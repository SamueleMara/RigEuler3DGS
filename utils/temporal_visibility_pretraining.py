"""Whole-sequence visibility inference for canonical Gaussian material.

This module deliberately knows nothing about static/dynamic material labels.
It only answers whether the canonical surface at a fixed pixel is present,
occluded, vacated, or unobservable in each frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm


PRESENT, OCCLUDED, VACATED, UNKNOWN = range(4)
STATE_NAMES = ("present", "occluded", "vacated", "unknown")
STATIC_PIXEL, DYNAMIC_PIXEL, UNKNOWN_PIXEL = range(3)


def _atomic_torch_save(value, path: Path) -> None:
    """Write required artifacts atomically, retrying PyTorch ZIP writer failures."""
    path = Path(path)
    # A save helper that cannot create its own directory turns a missing debug
    # folder into a stage-killing FileNotFoundError on the *temporary* file,
    # which reads as a serialisation failure rather than a missing path.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        try:
            torch.save(value, temporary)
        except (OSError, RuntimeError) as error:
            temporary.unlink(missing_ok=True)
            print(
                f"[Checkpoint] retrying {path.name} with legacy serialization "
                f"after: {error}"
            )
            torch.save(value, temporary, _use_new_zipfile_serialization=False)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class VisibilityConfig:
    depth_sigma: float = 0.02
    sigma_floor: float = 1.0e-3
    present_threshold: float = 0.60
    occlusion_margin: float = 2.5
    vacated_margin: float = 2.5
    min_canonical_alpha: float = 0.05
    boundary_relative_jump: float = 0.08
    hmm_stay_probability: float = 0.94


def inverse_depth_to_metric(invdepth: torch.Tensor) -> torch.Tensor:
    """Convert the repository's inverse-depth camera maps to metric depth."""
    invdepth = torch.as_tensor(invdepth).float()
    return torch.where(invdepth > 1.0e-8, invdepth.reciprocal(), torch.zeros_like(invdepth))


def load_frame0_reference_depth(
    frame0_depth: torch.Tensor,
    *,
    inverse_depth: bool = True,
    semantic_mask: torch.Tensor | None = None,
    canonical_alpha: torch.Tensor | None = None,
    min_canonical_alpha: float = 0.05,
    boundary_relative_jump: float = 0.08,
) -> dict[str, torch.Tensor]:
    """Build the physical reference and validity mask from measured frame-0 depth."""
    value = torch.as_tensor(frame0_depth).float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    depth = inverse_depth_to_metric(value) if inverse_depth else value
    valid = torch.isfinite(depth) & (depth > 0)
    if semantic_mask is not None:
        semantic = torch.as_tensor(semantic_mask).float()
        if semantic.ndim == 3:
            semantic = semantic[0]
        valid &= semantic > 0.5
    if canonical_alpha is not None:
        alpha = torch.as_tensor(canonical_alpha).float()
        if alpha.ndim == 3:
            alpha = alpha[0]
        valid &= alpha >= float(min_canonical_alpha)
    boundary = _depth_boundary_mask(depth, float(boundary_relative_jump))
    valid &= ~boundary
    return {
        "depth": torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth)),
        "valid": valid,
        "boundary": boundary,
    }


def classify_depth_states(
    reference_depth: torch.Tensor,
    observed_depth_sequence: torch.Tensor,
    valid_reference: torch.Tensor,
    *,
    present_abs: float,
    present_rel: float,
    occlusion_abs: float,
    vacated_abs: float,
) -> torch.Tensor:
    """Classify measured depth into deterministic PRESENT/OCCLUDED/VACATED/UNKNOWN."""
    reference = torch.as_tensor(reference_depth).float()
    observed = torch.as_tensor(observed_depth_sequence).float()
    if observed.ndim == 4 and observed.shape[1] == 1:
        observed = observed[:, 0]
    if observed.ndim != 3 or observed.shape[1:] != reference.shape:
        raise ValueError("expected reference [H,W] and observed depth [T,H,W]")
    valid = (
        torch.as_tensor(valid_reference).bool()[None]
        & torch.isfinite(observed)
        & (observed > 0)
    )
    delta = observed - reference[None]
    present_threshold = float(present_abs) + float(present_rel) * reference.abs()
    present = valid & (delta.abs() <= present_threshold[None])
    occluded = valid & (delta < -float(occlusion_abs)) & ~present
    vacated = valid & (delta > float(vacated_abs)) & ~present
    state = torch.full(observed.shape, UNKNOWN, dtype=torch.uint8, device=observed.device)
    state[present] = PRESENT
    state[occluded] = OCCLUDED
    state[vacated] = VACATED
    return state


def _minimum_run_mask(mask: torch.Tensor, minimum: int) -> torch.Tensor:
    """Mark elements belonging to a true temporal run of at least ``minimum``."""
    value = torch.as_tensor(mask).bool()
    minimum = max(1, int(minimum))
    if minimum == 1:
        return value
    strong = torch.zeros_like(value)
    for start in range(max(0, value.shape[0] - minimum + 1)):
        run = value[start : start + minimum].all(dim=0)
        for offset in range(minimum):
            strong[start + offset] |= run
    return strong


def cleanup_state_sequences(
    state: torch.Tensor,
    *,
    vacated_min_run: int,
    remove_isolated_flips: bool = True,
) -> torch.Tensor:
    """Remove one-frame flips and turn short VACATED runs into UNKNOWN evidence."""
    cleaned = torch.as_tensor(state).clone().to(torch.uint8)
    if cleaned.ndim != 3:
        raise ValueError("state must have shape [T,H,W]")
    if remove_isolated_flips and cleaned.shape[0] >= 3:
        original = cleaned.clone()
        agree = original[:-2] == original[2:]
        flip = agree & (original[1:-1] != original[:-2])
        cleaned[1:-1] = torch.where(flip, original[:-2], original[1:-1])
    strong_vacated = _minimum_run_mask(
        cleaned == VACATED, max(1, int(vacated_min_run))
    )
    cleaned[(cleaned == VACATED) & ~strong_vacated] = UNKNOWN
    return cleaned


def compute_pixel_sequence_stats(state: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return interpretable whole-sequence counts and run statistics per pixel."""
    value = torch.as_tensor(state).long()
    if value.ndim != 3:
        raise ValueError("state must have shape [T,H,W]")
    masks = [value == index for index in range(4)]
    counts = [mask.sum(dim=0) for mask in masks]
    observable = counts[PRESENT] + counts[VACATED]
    denominator = observable.clamp_min(1).float()
    vacated = masks[VACATED]
    previous = torch.cat((torch.zeros_like(vacated[:1]), vacated[:-1]), dim=0)
    return {
        "present_count": counts[PRESENT],
        "occluded_count": counts[OCCLUDED],
        "vacated_count": counts[VACATED],
        "unknown_count": counts[UNKNOWN],
        "observable_frame_count": observable,
        "present_ratio": counts[PRESENT].float() / denominator,
        "vacated_ratio": counts[VACATED].float() / denominator,
        "present_fraction_all": masks[PRESENT].float().mean(dim=0),
        "occluded_fraction_all": masks[OCCLUDED].float().mean(dim=0),
        "vacated_fraction_all": masks[VACATED].float().mean(dim=0),
        "unknown_fraction_all": masks[UNKNOWN].float().mean(dim=0),
        "longest_present_run": _longest_run(masks[PRESENT]),
        "longest_occluded_run": _longest_run(masks[OCCLUDED]),
        "longest_vacated_run": _longest_run(vacated),
        "number_vacated_runs": (vacated & ~previous).sum(dim=0),
    }


def classify_pixels_depth_only(
    stats: dict[str, torch.Tensor],
    *,
    min_observable_frames: int,
    static_present_ratio: float,
    dynamic_vacated_ratio: float,
    vacated_min_run: int,
) -> torch.Tensor:
    """Classify each camera pixel with explicit conservative depth conditions."""
    enough = stats["observable_frame_count"] >= int(min_observable_frames)
    static = (
        enough
        & (stats["present_ratio"] >= float(static_present_ratio))
        & (stats["longest_vacated_run"] < int(vacated_min_run))
    )
    dynamic = (
        enough
        & (stats["vacated_ratio"] >= float(dynamic_vacated_ratio))
        & (stats["longest_vacated_run"] >= int(vacated_min_run))
    )
    result = torch.full(
        stats["observable_frame_count"].shape,
        UNKNOWN_PIXEL,
        dtype=torch.uint8,
        device=stats["observable_frame_count"].device,
    )
    result[static & ~dynamic] = STATIC_PIXEL
    result[dynamic & ~static] = DYNAMIC_PIXEL
    return result


@torch.no_grad()
def build_canonical_visibility_cache(
    cameras,
    gaussians,
    pipe,
    *,
    use_trained_exp: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Render canonical metric depth and opacity for each fixed camera."""
    from gaussian_renderer import render, render_feature_field

    if not isinstance(cameras, (list, tuple)):
        cameras = [cameras]
    result = {}
    for camera in cameras:
        package = render(
            camera,
            gaussians,
            pipe,
            torch.zeros(3, device=gaussians.get_xyz.device),
            use_trained_exp=use_trained_exp,
        )
        alpha = render_feature_field(
            camera,
            gaussians,
            pipe,
            torch.ones((gaussians.get_xyz.shape[0], 1), device=gaussians.get_xyz.device),
            bg_feature=torch.zeros(1, device=gaussians.get_xyz.device),
            detach_geometry=True,
        )["render"][0]
        result[_camera_key(camera)] = {
            "canonical_depth": inverse_depth_to_metric(package["depth"][0]).detach(),
            "canonical_alpha": alpha.detach().clamp(0, 1),
        }
    return result


def _depth_boundary_mask(depth: torch.Tensor, relative_jump: float) -> torch.Tensor:
    valid = depth > 0
    result = torch.zeros_like(valid)
    dx = (depth[..., :, 1:] - depth[..., :, :-1]).abs()
    sx = torch.maximum(depth[..., :, 1:].abs(), depth[..., :, :-1].abs()).clamp_min(1.0e-6)
    bx = (dx / sx) > float(relative_jump)
    result[..., :, 1:] |= bx
    result[..., :, :-1] |= bx
    dy = (depth[..., 1:, :] - depth[..., :-1, :]).abs()
    sy = torch.maximum(depth[..., 1:, :].abs(), depth[..., :-1, :].abs()).clamp_min(1.0e-6)
    by = (dy / sy) > float(relative_jump)
    result[..., 1:, :] |= by
    result[..., :-1, :] |= by
    return result & valid


def compute_depth_state_likelihoods(
    canonical_depth: torch.Tensor,
    observed_depth_sequence: torch.Tensor,
    canonical_alpha: torch.Tensor,
    config: VisibilityConfig | None = None,
) -> torch.Tensor:
    """Return normalized emissions ``[T,4,H,W]`` in metric-depth space.

    ``depth_sigma`` is relative to canonical depth, with ``sigma_floor`` keeping
    the model well-defined near the camera. Margins are expressed in sigma.
    Invalid depth, weak ownership and depth discontinuities map to UNKNOWN.
    """
    cfg = config or VisibilityConfig()
    z0 = torch.as_tensor(canonical_depth).float()
    if z0.ndim == 3 and z0.shape[0] == 1:
        z0 = z0[0]
    observed = torch.as_tensor(observed_depth_sequence).float()
    if observed.ndim == 4 and observed.shape[1] == 1:
        observed = observed[:, 0]
    alpha = torch.as_tensor(canonical_alpha).float()
    if alpha.ndim == 3 and alpha.shape[0] == 1:
        alpha = alpha[0]
    if observed.ndim != 3 or z0.shape != observed.shape[-2:] or alpha.shape != z0.shape:
        raise ValueError("expected canonical [H,W], observations [T,H,W], alpha [H,W]")

    sigma = (z0.abs() * float(cfg.depth_sigma)).clamp_min(float(cfg.sigma_floor))
    delta = observed - z0[None]
    present = torch.exp(-0.5 * (delta / sigma[None]).square())
    occ_edge = -float(cfg.occlusion_margin) * sigma[None]
    vac_edge = float(cfg.vacated_margin) * sigma[None]
    occluded = torch.sigmoid((occ_edge - delta) / sigma[None].clamp_min(1.0e-8))
    vacated = torch.sigmoid((delta - vac_edge) / sigma[None].clamp_min(1.0e-8))

    valid = (z0[None] > 0) & (observed > 0) & torch.isfinite(observed)
    ownership = alpha[None] >= float(cfg.min_canonical_alpha)
    boundary = _depth_boundary_mask(z0, cfg.boundary_relative_jump)[None]
    known = valid & ownership & ~boundary
    unknown = (~known).float()
    emissions = torch.stack((present, occluded, vacated, unknown), dim=1)
    emissions[:, :3] *= known[:, None]
    # Leave a small unknown likelihood in valid regions so HMM transitions can
    # bridge one unreliable observation without manufacturing a material vote.
    emissions[:, UNKNOWN] += known.float() * 0.01
    return emissions / emissions.sum(dim=1, keepdim=True).clamp_min(1.0e-8)


def smooth_visibility_sequence(
    likelihoods: torch.Tensor,
    *,
    enabled: bool = True,
    stay_probability: float = 0.94,
) -> dict[str, torch.Tensor]:
    """Viterbi-smooth visibility only; return hard states and soft emissions."""
    emission = torch.as_tensor(likelihoods).float()
    if emission.ndim != 4 or emission.shape[1] != 4:
        raise ValueError("likelihoods must have shape [T,4,H,W]")
    if not enabled or emission.shape[0] <= 1:
        return {"state": emission.argmax(dim=1).to(torch.uint8), "probability": emission}

    stay = min(1.0 - 1.0e-6, max(0.25, float(stay_probability)))
    transition = torch.full((4, 4), (1.0 - stay) / 3.0, device=emission.device)
    transition.fill_diagonal_(stay)
    # Occlusion/vacating may return to present. Direct O<->V flicker is less plausible.
    transition[OCCLUDED, VACATED] *= 0.25
    transition[VACATED, OCCLUDED] *= 0.25
    transition /= transition.sum(dim=1, keepdim=True)
    log_transition = transition.clamp_min(1.0e-8).log()
    log_emission = emission.clamp_min(1.0e-8).log()
    t_count, _, height, width = emission.shape
    score = log_emission[0]
    back = torch.empty((t_count - 1, 4, height, width), device=emission.device, dtype=torch.uint8)
    for time_index in range(1, t_count):
        candidate = score[:, None] + log_transition[:, :, None, None]
        score, predecessor = candidate.max(dim=0)
        score = score + log_emission[time_index]
        back[time_index - 1] = predecessor.to(torch.uint8)
    state = torch.empty((t_count, height, width), device=emission.device, dtype=torch.uint8)
    state[-1] = score.argmax(dim=0).to(torch.uint8)
    for time_index in range(t_count - 2, -1, -1):
        state[time_index] = torch.gather(
            back[time_index].long(), 0, state[time_index + 1][None].long()
        )[0].to(torch.uint8)
    return {"state": state, "probability": emission}


def build_flow_identity_mask(
    visibility: torch.Tensor,
    *,
    present_threshold: float = 0.60,
) -> torch.Tensor:
    """Return soft P(t)*P(t+1) identity validity ``[T-1,H,W]``."""
    value = torch.as_tensor(visibility)
    if value.ndim == 3:
        present = (value == PRESENT).float()
    elif value.ndim == 4 and value.shape[1] == 4:
        present = value[:, PRESENT].float()
        present = torch.where(present >= float(present_threshold), present, torch.zeros_like(present))
    else:
        raise ValueError("visibility must be hard [T,H,W] or probabilities [T,4,H,W]")
    if present.shape[0] < 2:
        return present.new_zeros((0,) + tuple(present.shape[-2:]))
    return present[:-1] * present[1:]


def _longest_run(mask: torch.Tensor) -> torch.Tensor:
    """Longest true run over dimension zero for a boolean ``[T,...]`` tensor."""
    current = torch.zeros(mask.shape[1:], device=mask.device, dtype=torch.int16)
    longest = torch.zeros_like(current)
    for item in mask.bool():
        current = torch.where(item, current + 1, torch.zeros_like(current))
        longest = torch.maximum(longest, current)
    return longest


def summarize_pixel_temporal_evidence(
    visibility_state: torch.Tensor,
    dynamic_support: torch.Tensor | None = None,
    static_support: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    state = torch.as_tensor(visibility_state).long()
    if state.ndim != 3:
        raise ValueError("visibility_state must have shape [T,H,W]")
    summary = {}
    for index, name in enumerate(STATE_NAMES):
        mask = state == index
        summary[f"{name}_fraction"] = mask.float().mean(dim=0)
        summary[f"longest_{name}_run"] = _longest_run(mask)
    if dynamic_support is not None or static_support is not None:
        dyn = torch.zeros_like(state[:-1], dtype=torch.bool) if dynamic_support is None else torch.as_tensor(dynamic_support).bool()
        sta = torch.zeros_like(state[:-1], dtype=torch.bool) if static_support is None else torch.as_tensor(static_support).bool()
        informative = dyn | sta
        denominator = informative.sum(dim=0).clamp_min(1)
        summary["dynamic_time_fraction"] = dyn.sum(dim=0).float() / denominator
        summary["static_time_fraction"] = sta.sum(dim=0).float() / denominator
        summary["informative_count"] = informative.sum(dim=0)
        summary["longest_dynamic_run"] = _longest_run(dyn)
        summary["longest_static_run"] = _longest_run(sta)
    return summary


def compute_pixel_sequence_evidence(
    visibility_state: torch.Tensor,
    normalized_flow_residual: torch.Tensor,
    flow_confidence: torch.Tensor,
    *,
    static_residual_threshold: float = 2.0,
    dynamic_residual_threshold: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Build per-frame/interval evidence without allowing occlusions to vote."""
    state = torch.as_tensor(visibility_state).long()
    residual = torch.as_tensor(normalized_flow_residual).float()
    confidence = torch.as_tensor(flow_confidence).float()
    identity = build_flow_identity_mask(state)
    if residual.shape != identity.shape or confidence.shape != identity.shape:
        raise ValueError("flow tensors must have shape [T-1,H,W]")
    valid = identity * confidence.clamp(0.0, 1.0)
    static_flow = valid * (residual <= float(static_residual_threshold)).float()
    dynamic_flow = valid * (residual >= float(dynamic_residual_threshold)).float()
    vacated = (state == VACATED).float()
    present = (state == PRESENT).float()
    occluded = (state == OCCLUDED).float()
    unknown = (state == UNKNOWN).float()
    return {
        "present": present,
        "occluded": occluded,
        "vacated": vacated,
        "unknown": unknown,
        "identity_valid": identity,
        "static_flow": static_flow,
        "dynamic_flow": dynamic_flow,
        "informative": valid,
        "static_support": static_flow,
        "dynamic_support": torch.maximum(dynamic_flow, vacated[1:]),
    }


def save_visibility_debug_image(
    path: str | Path,
    state: torch.Tensor,
    canonical_depth: torch.Tensor | None = None,
    observed_depth: torch.Tensor | None = None,
) -> None:
    """Write the required P/O/V/U color diagnostic without extra dependencies."""
    from PIL import Image

    palette = torch.tensor(
        ((40, 200, 60), (40, 90, 220), (225, 50, 45), (128, 128, 128)),
        dtype=torch.uint8,
    )
    hard = torch.as_tensor(state).detach().cpu().long().clamp(0, 3)
    color = palette.index_select(0, hard.flatten()).reshape(*hard.shape, 3).numpy()
    Image.fromarray(color).save(Path(path))


def _state_rgb(state: torch.Tensor) -> torch.Tensor:
    palette = torch.tensor(
        ((40, 200, 60), (40, 90, 220), (225, 50, 45), (128, 128, 128)),
        dtype=torch.uint8,
    )
    hard = torch.as_tensor(state).detach().cpu().long().clamp(0, 3)
    return palette.index_select(0, hard.flatten()).reshape(*hard.shape, 3)


def _pixel_class_rgb(pixel_class: torch.Tensor) -> torch.Tensor:
    palette = torch.tensor(
        ((40, 200, 60), (225, 50, 45), (128, 128, 128)), dtype=torch.uint8
    )
    hard = torch.as_tensor(pixel_class).detach().cpu().long().clamp(0, 2)
    return palette.index_select(0, hard.flatten()).reshape(*hard.shape, 3)


def _save_pixel_class(path: Path, pixel_class: torch.Tensor) -> None:
    from PIL import Image

    Image.fromarray(_pixel_class_rgb(pixel_class).numpy()).save(path)


def _save_depth_state_panel(
    path: Path,
    reference_depth: torch.Tensor,
    current_depth: torch.Tensor,
    state: torch.Tensor,
) -> None:
    """Save frame-0/current/delta/P-O-V-U as one categorical diagnostic panel."""
    from PIL import Image, ImageDraw

    reference = torch.as_tensor(reference_depth).detach().cpu().float()
    current = torch.as_tensor(current_depth).detach().cpu().float()
    valid = torch.cat((reference[reference > 0], current[current > 0]))
    maximum = float(torch.quantile(valid, 0.98)) if valid.numel() else 1.0

    def gray(value):
        image = (value.clamp(0, maximum) / max(maximum, 1.0e-8) * 255).byte()
        return image[..., None].expand(-1, -1, 3)

    delta = current - reference
    finite = delta[torch.isfinite(delta)]
    scale = float(torch.quantile(finite.abs(), 0.98)) if finite.numel() else 1.0
    normalized = (delta / max(scale, 1.0e-8)).clamp(-1, 1)
    delta_rgb = torch.stack(
        (
            (normalized.clamp_min(0) * 255),
            ((1.0 - normalized.abs()) * 255),
            ((-normalized).clamp_min(0) * 255),
        ),
        dim=-1,
    ).byte()
    images = (gray(reference), gray(current), delta_rgb, _state_rgb(state))
    height, width = reference.shape
    canvas = Image.new("RGB", (4 * width, height + 20), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(
        zip(("frame-0 depth", "current depth", "signed delta", "P/O/V/U"), images)
    ):
        canvas.paste(Image.fromarray(image.numpy()), (index * width, 20))
        draw.text((index * width + 3, 3), label, fill="black")
    canvas.save(path)


def _save_temporal_strips(path: Path, rows: list[dict]) -> None:
    """Save categorical strips plus measured depth-difference curves."""
    from PIL import Image, ImageDraw

    if not rows:
        return
    width, row_height, strip_left, step = 900, 74, 410, 6
    canvas = Image.new("RGB", (width, row_height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    colors = ((40, 200, 60), (40, 90, 220), (225, 50, 45), (128, 128, 128))
    class_names = ("STATIC", "DYNAMIC", "UNKNOWN")
    for row_index, row in enumerate(rows):
        top = row_index * row_height
        state = torch.as_tensor(row["state"]).long().tolist()
        delta = torch.as_tensor(row["delta"]).float()
        scale = max(float(delta.abs().max()), 1.0e-8)
        draw.text(
            (4, top + 4),
            f"{row['camera']} {row['kind']} p=({row['x']},{row['y']}) "
            f"z0={row['reference_depth']:.4f} class={class_names[row['pixel_class']]}",
            fill="black",
        )
        for time_index, value in enumerate(state):
            left = strip_left + time_index * step
            draw.rectangle((left, top + 3, left + step - 1, top + 19), fill=colors[value])
        points = []
        for time_index, value in enumerate(delta.tolist()):
            x = strip_left + time_index * step + step // 2
            y = top + 49 - int(20 * max(-1.0, min(1.0, value / scale)))
            points.append((x, y))
        draw.line(points, fill=(25, 25, 25), width=1)
        draw.line((strip_left, top + 49, strip_left + max(0, len(state) - 1) * step, top + 49), fill=(180, 180, 180))
    canvas.save(path)


def _save_scalar_debug(path: Path, value: torch.Tensor, maximum: float | None = None) -> None:
    from PIL import Image

    tensor = torch.as_tensor(value).detach().cpu().float()
    if maximum is None:
        finite = tensor[torch.isfinite(tensor)]
        maximum = float(torch.quantile(finite, 0.98)) if finite.numel() else 1.0
    image = (tensor.clamp(0.0, max(maximum, 1.0e-8)) / max(maximum, 1.0e-8) * 255).byte().numpy()
    Image.fromarray(image).save(path)


def _camera_key(camera) -> str:
    return str(getattr(camera, "dynamic_camera_id", getattr(camera, "image_name", "camera")))


def _frame_groups(cameras) -> list[tuple[int, list]]:
    grouped = {}
    for camera in cameras:
        grouped.setdefault(int(getattr(camera, "frame_id", 0)), []).append(camera)
    keys = sorted({_camera_key(camera) for camera in cameras})
    result = []
    for frame_id in sorted(grouped):
        frame = sorted(grouped[frame_id], key=_camera_key)
        if [_camera_key(camera) for camera in frame] != keys:
            raise RuntimeError(f"frame {frame_id} is missing fixed-camera observations")
        result.append((frame_id, frame))
    return result


def _resize_map(value: torch.Tensor, size: tuple[int, int], mode: str = "nearest") -> torch.Tensor:
    tensor = torch.as_tensor(value).float()
    if tensor.ndim == 2:
        tensor = tensor[None]
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(tensor[None], size=size, mode=mode, **kwargs)[0]


@torch.no_grad()
def run_depth_sequence_pretraining(
    scene,
    gaussians,
    pipe,
    dataset,
    args,
    *,
    log_fn=None,
) -> dict:
    """Build frozen Gaussian cores from measured depth and canonical ownership only."""
    from PIL import Image
    from gaussian_renderer import render_feature_field
    from utils.gaussian_sequence_assignment import (
        accumulate_depth_pixel_classes_to_gaussians,
        build_depth_only_initial_cores,
        compute_gaussian_camera_votes,
    )
    from utils.loss_utils import _dynamic_root_for_train, _load_mask_for_view

    started = time.perf_counter()
    groups = _frame_groups(scene.getTrainCameras())
    if len(groups) < 2:
        raise RuntimeError("depth sequence pretraining needs at least two complete frames")
    camera_keys = [_camera_key(camera) for camera in groups[0][1]]
    camera_maps = [{_camera_key(camera): camera for camera in frame} for _, frame in groups]
    output_dir = Path(scene.model_path) / "dynamic_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_states, clean_states, pixel_stats, pixel_classes = {}, {}, {}, {}
    reference_depths, valid_reference_masks = {}, {}
    records, strip_rows = [], []
    state_totals = torch.zeros(4, dtype=torch.long)
    valid_pixel_total = 0
    pixel_counts = {}

    for camera_index, key in enumerate(
        tqdm(camera_keys, desc="Phase 1A measured-depth cores", unit="camera")
    ):
        canonical_camera = camera_maps[0][key]
        height, width = int(canonical_camera.image_height), int(canonical_camera.image_width)
        alpha = render_feature_field(
            canonical_camera,
            gaussians,
            pipe,
            torch.ones((gaussians.get_xyz.shape[0], 1), device=gaussians.get_xyz.device),
            bg_feature=torch.zeros(1, device=gaussians.get_xyz.device),
            detach_geometry=True,
        )["render"][0].detach().cpu().clamp(0, 1)
        try:
            semantic = _load_mask_for_view(
                canonical_camera, dataset, args, (height, width), device="cpu"
            )
        except Exception:
            semantic = None
        frame0_invdepth = getattr(canonical_camera, "invdepthmap", None)
        if frame0_invdepth is None:
            raise RuntimeError(f"missing measured frame-0 depth for camera {key}")
        frame0_invdepth = _resize_map(
            frame0_invdepth, (height, width), "nearest"
        )[0].cpu()
        reference = load_frame0_reference_depth(
            frame0_invdepth,
            inverse_depth=True,
            semantic_mask=semantic,
            canonical_alpha=alpha,
            min_canonical_alpha=float(args.visibility_min_canonical_alpha),
        )
        observed = []
        for frame_map in camera_maps:
            invdepth = getattr(frame_map[key], "invdepthmap", None)
            if invdepth is None:
                raise RuntimeError(
                    f"missing measured depth for camera {key}, "
                    f"frame {getattr(frame_map[key], 'frame_id', '?')}"
                )
            invdepth = _resize_map(invdepth, (height, width), "nearest")[0].cpu()
            observed.append(inverse_depth_to_metric(invdepth))
        observed = torch.stack(observed)
        raw = classify_depth_states(
            reference["depth"],
            observed,
            reference["valid"],
            present_abs=float(args.sequence_depth_present_abs),
            present_rel=float(args.sequence_depth_present_rel),
            occlusion_abs=float(args.sequence_depth_occlusion_abs),
            vacated_abs=float(args.sequence_depth_vacated_abs),
        )
        clean = cleanup_state_sequences(
            raw, vacated_min_run=int(args.sequence_vacated_min_run)
        )
        stats = compute_pixel_sequence_stats(clean)
        pixel_class = classify_pixels_depth_only(
            stats,
            min_observable_frames=int(args.sequence_depth_min_observable_frames),
            static_present_ratio=float(args.sequence_pixel_static_present_ratio),
            dynamic_vacated_ratio=float(args.sequence_pixel_dynamic_vacated_ratio),
            vacated_min_run=int(args.sequence_vacated_min_run),
        )
        raw_states[key] = raw.cpu()
        clean_states[key] = clean.cpu()
        pixel_stats[key] = {name: value.cpu() for name, value in stats.items()}
        pixel_classes[key] = pixel_class.cpu()
        reference_depths[key] = reference["depth"].cpu()
        valid_reference_masks[key] = reference["valid"].cpu()
        state_totals += torch.bincount(clean.flatten().long(), minlength=4)
        valid_pixel_total += int(reference["valid"].sum())
        pixel_counts[key] = {
            "static": int((pixel_class == STATIC_PIXEL).sum()),
            "dynamic": int((pixel_class == DYNAMIC_PIXEL).sum()),
            "unknown": int((pixel_class == UNKNOWN_PIXEL).sum()),
        }

        observable = (
            stats["observable_frame_count"]
            >= int(args.sequence_depth_min_observable_frames)
        ).float()
        features = torch.stack(
            (
                (pixel_class == STATIC_PIXEL).float(),
                (pixel_class == DYNAMIC_PIXEL).float(),
                (pixel_class == UNKNOWN_PIXEL).float(),
                observable,
                stats["occluded_fraction_all"],
                stats["vacated_ratio"],
                stats["present_ratio"],
            )
        )
        records.append(
            {
                "camera": canonical_camera,
                "camera_index": camera_index,
                "pixel_features": features,
                "pixel_confidence": reference["valid"].float()[None],
            }
        )

        selected_frames = sorted({0, len(groups) // 2, len(groups) - 1})
        for frame_index in selected_frames:
            _save_depth_state_panel(
                output_dir / f"depth_state_{key}_frame_{groups[frame_index][0]:06d}.png",
                reference["depth"],
                observed[frame_index],
                clean[frame_index],
            )
        for name in (
            "present_ratio",
            "vacated_ratio",
            "occluded_fraction_all",
            "unknown_fraction_all",
            "longest_vacated_run",
        ):
            _save_scalar_debug(
                output_dir / f"depth_{key}_{name}.png", stats[name]
            )
        _save_pixel_class(
            output_dir / f"depth_{key}_final_depth_only_pixel_class.png",
            pixel_class,
        )

        def add_strip(kind, score, mask):
            candidates = torch.as_tensor(mask).bool()
            if not bool(candidates.any()):
                return
            ranked = torch.where(
                candidates, torch.as_tensor(score).float(), torch.full_like(torch.as_tensor(score).float(), -float("inf"))
            )
            flat = int(torch.argmax(ranked))
            y, x = divmod(flat, width)
            strip_rows.append(
                {
                    "camera": key,
                    "kind": kind,
                    "x": x,
                    "y": y,
                    "reference_depth": float(reference["depth"][y, x]),
                    "pixel_class": int(pixel_class[y, x]),
                    "state": clean[:, y, x],
                    "delta": observed[:, y, x] - reference["depth"][y, x],
                }
            )

        add_strip("static", stats["present_ratio"], pixel_class == STATIC_PIXEL)
        add_strip("dynamic", stats["vacated_ratio"], pixel_class == DYNAMIC_PIXEL)
        add_strip("occluded", stats["occluded_fraction_all"], reference["valid"])
        add_strip("unknown", stats["unknown_fraction_all"], pixel_class == UNKNOWN_PIXEL)

    _save_temporal_strips(output_dir / "phase1_depth_temporal_strips.png", strip_rows)
    gaussian_evidence = accumulate_depth_pixel_classes_to_gaussians(
        gaussians, pipe, records, num_cameras=len(camera_keys)
    )
    votes = compute_gaussian_camera_votes(
        gaussian_evidence,
        min_camera_weight=float(args.sequence_gaussian_min_camera_weight),
        static_threshold=float(args.sequence_gaussian_static_camera_threshold),
        dynamic_threshold=float(args.sequence_gaussian_dynamic_camera_threshold),
    )
    cores = build_depth_only_initial_cores(
        votes,
        min_static_cameras=int(args.sequence_gaussian_min_static_cameras),
        min_dynamic_cameras=int(args.sequence_gaussian_min_dynamic_cameras),
        min_static_camera_fraction=float(args.sequence_gaussian_min_static_camera_fraction),
        min_dynamic_camera_fraction=float(args.sequence_gaussian_min_dynamic_camera_fraction),
        max_conflicting_cameras=int(args.sequence_gaussian_max_conflicting_cameras),
    )
    gaussian_evidence.update(
        {
            key: value.cpu() if torch.is_tensor(value) else value
            for key, value in votes.items()
        }
    )

    point_count = int(gaussians.get_xyz.shape[0])
    colors = torch.full((point_count, 3), 0.5, device=gaussians.get_xyz.device)
    colors[cores["static_core_mask"].to(colors.device)] = torch.tensor(
        (0.15, 0.75, 0.20), device=colors.device
    )
    colors[cores["dynamic_core_mask"].to(colors.device)] = torch.tensor(
        (0.90, 0.15, 0.10), device=colors.device
    )
    scalar_debug = {
        "dynamic_camera_count": votes["dynamic_camera_count"].float() / len(camera_keys),
        "static_camera_count": votes["static_camera_count"].float() / len(camera_keys),
        "visible_camera_count": votes["visible_camera_count"].float() / len(camera_keys),
        "mean_vacated_ratio": votes["mean_vacated_ratio"],
        "mean_present_ratio": votes["mean_present_ratio"],
    }
    for key, camera in zip(camera_keys, groups[0][1]):
        core_image = render_feature_field(
            camera,
            gaussians,
            pipe,
            colors,
            bg_feature=torch.zeros(3, device=colors.device),
            detach_geometry=True,
        )["render"].detach().cpu().permute(1, 2, 0).clamp(0, 1)
        Image.fromarray((core_image * 255).byte().numpy()).save(
            output_dir / f"depth_gaussian_cores_{key}.png"
        )
        for name, value in scalar_debug.items():
            image = render_feature_field(
                camera,
                gaussians,
                pipe,
                value.to(gaussians.get_xyz)[:, None],
                bg_feature=torch.zeros(1, device=gaussians.get_xyz.device),
                detach_geometry=True,
            )["render"][0].detach().cpu().clamp(0, 1)
            _save_scalar_debug(output_dir / f"depth_gaussian_{key}_{name}.png", image, 1.0)

    dynamic = cores["dynamic_core_mask"]
    static = cores["static_core_mask"]
    dynamic_camera_count = votes["dynamic_camera_count"]
    static_camera_count = votes["static_camera_count"]
    visible_camera_count = votes["visible_camera_count"]

    def fraction(mask, threshold, count):
        return float((count[mask] >= int(threshold)).float().mean()) if bool(mask.any()) else 0.0

    summary = {
        "schema_version": 1,
        "reference_depth_source": "measured_frame_0",
        "optical_flow_used": False,
        "rendered_depth_used": False,
        "hmm_used": False,
        "ground_truth_used_as_algorithm_input": False,
        "frames": len(groups),
        "cameras": len(camera_keys),
        "total_canonical_valid_pixels": valid_pixel_total,
        "present_state_count": int(state_totals[PRESENT]),
        "occluded_state_count": int(state_totals[OCCLUDED]),
        "vacated_state_count": int(state_totals[VACATED]),
        "unknown_state_count": int(state_totals[UNKNOWN]),
        "pixel_counts_per_camera": pixel_counts,
        "gaussian_static_core_size": int(static.sum()),
        "gaussian_dynamic_core_size": int(dynamic.sum()),
        "gaussian_unknown_size": int(cores["unknown_mask"].sum()),
        "median_visible_camera_count": float(visible_camera_count.float().median()),
        "median_static_camera_count_for_static_core": float(static_camera_count[static].float().median()) if bool(static.any()) else None,
        "median_dynamic_camera_count_for_dynamic_core": float(dynamic_camera_count[dynamic].float().median()) if bool(dynamic.any()) else None,
        "fraction_dynamic_core_ge2cams": fraction(dynamic, 2, dynamic_camera_count),
        "fraction_dynamic_core_ge3cams": fraction(dynamic, 3, dynamic_camera_count),
        "fraction_static_core_ge2cams": fraction(static, 2, static_camera_count),
        "fraction_static_core_ge3cams": fraction(static, 3, static_camera_count),
        "dynamic_core_exactly_1_camera": int((dynamic & (dynamic_camera_count == 1)).sum()),
        "dynamic_core_ge2_cameras": int((dynamic & (dynamic_camera_count >= 2)).sum()),
        "dynamic_core_ge3_cameras": int((dynamic & (dynamic_camera_count >= 3)).sum()),
        "gaussians_with_conflicting_camera_votes": int(((dynamic_camera_count > 0) & (static_camera_count > 0)).sum()),
        "runtime_seconds_before_artifact_write": time.perf_counter() - started,
        "stage_b_non_gt_checks": {
            "dynamic_core_nonempty": bool(dynamic.any()),
            "dynamic_core_all_multiview": bool(
                not dynamic.any() or (dynamic_camera_count[dynamic] >= 2).all()
            ),
            "static_core_nonempty": bool(static.any()),
            "no_one_camera_dynamic_core": not bool(
                (dynamic & (dynamic_camera_count == 1)).any()
            ),
        },
        "gt_evaluation": None,
    }
    if args.rigid_analytic_validation:
        # Evaluation begins only after depth labels are frozen and cannot feed back.
        from utils.rigid_killing_validation import load_rigid_dataset_ground_truth

        ground_truth = load_rigid_dataset_ground_truth(
            _dynamic_root_for_train(dataset, args),
            gaussians.get_xyz,
            moving_part=int(args.rigid_analytic_moving_part),
            gaussian_part_labels=getattr(gaussians, "_gt_part_labels", None),
        )
        gt_dynamic = ground_truth.moving_mask.cpu().bool()
        gt_static = ~gt_dynamic
        dynamic_tp = int((dynamic & gt_dynamic).sum())
        dynamic_fp = int((dynamic & gt_static).sum())
        dynamic_fn = int((~dynamic & gt_dynamic).sum())
        static_tp = int((static & gt_static).sum())
        static_fp = int((static & gt_dynamic).sum())
        static_fn = int((~static & gt_static).sum())
        summary["gt_evaluation"] = {
            "role": "post_pretrain_evaluation_only",
            "dynamic_core_precision": dynamic_tp / max(1, dynamic_tp + dynamic_fp),
            "dynamic_core_recall": dynamic_tp / max(1, dynamic_tp + dynamic_fn),
            "dynamic_core_iou": dynamic_tp / max(1, dynamic_tp + dynamic_fp + dynamic_fn),
            "static_core_precision": static_tp / max(1, static_tp + static_fp),
            "static_core_recall": static_tp / max(1, static_tp + static_fn),
            "confusion_matrix": {
                "gt_static": {
                    "pred_static": static_tp,
                    "pred_dynamic": dynamic_fp,
                    "pred_unknown": int((cores["unknown_mask"] & gt_static).sum()),
                },
                "gt_dynamic": {
                    "pred_static": static_fp,
                    "pred_dynamic": dynamic_tp,
                    "pred_unknown": int((cores["unknown_mask"] & gt_dynamic).sum()),
                },
            },
        }

    _atomic_torch_save(
        {"schema_version": 1, "camera_keys": camera_keys, "states": raw_states},
        output_dir / "phase1_depth_visibility_states_raw.pt",
    )
    _atomic_torch_save(
        {"schema_version": 1, "camera_keys": camera_keys, "states": clean_states},
        output_dir / "phase1_depth_visibility_states_clean.pt",
    )
    _atomic_torch_save(
        {
            "schema_version": 1,
            "camera_keys": camera_keys,
            "reference_depth": reference_depths,
            "valid_reference": valid_reference_masks,
            "stats": pixel_stats,
        },
        output_dir / "phase1_depth_pixel_sequence_stats.pt",
    )
    _atomic_torch_save(
        {"schema_version": 1, "camera_keys": camera_keys, "pixel_class": pixel_classes},
        output_dir / "phase1_depth_pixel_classification.pt",
    )
    _atomic_torch_save(
        gaussian_evidence, output_dir / "phase1_depth_gaussian_camera_evidence.pt"
    )
    _atomic_torch_save(
        {
            "schema_version": 1,
            "canonical_xyz": gaussians.get_xyz.detach().cpu(),
            **{key: value.cpu() if torch.is_tensor(value) else value for key, value in cores.items()},
            "old_attribution_score": None,
            "depth_only_state": cores["state"].cpu(),
        },
        output_dir / "phase1_depth_initial_gaussian_cores.pt",
    )
    with (output_dir / "phase1_depth_pretrain_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)

    if log_fn is not None:
        total_states = max(1, int(state_totals.sum()))
        metrics = {
            "depth_pretrain/present_fraction": float(state_totals[PRESENT]) / total_states,
            "depth_pretrain/occluded_fraction": float(state_totals[OCCLUDED]) / total_states,
            "depth_pretrain/vacated_fraction": float(state_totals[VACATED]) / total_states,
            "depth_pretrain/unknown_fraction": float(state_totals[UNKNOWN]) / total_states,
            "depth_pretrain/gaussian/static_core_size": int(static.sum()),
            "depth_pretrain/gaussian/dynamic_core_size": int(dynamic.sum()),
            "depth_pretrain/gaussian/unknown_size": int(cores["unknown_mask"].sum()),
            "depth_pretrain/gaussian/dynamic_core_ge2cams": summary["dynamic_core_ge2_cameras"],
            "depth_pretrain/gaussian/dynamic_core_ge3cams": summary["dynamic_core_ge3_cameras"],
        }
        if summary["gt_evaluation"] is not None:
            metrics.update(
                {
                    "depth_pretrain/gt/dynamic_precision": summary["gt_evaluation"]["dynamic_core_precision"],
                    "depth_pretrain/gt/dynamic_recall": summary["gt_evaluation"]["dynamic_core_recall"],
                    "depth_pretrain/gt/static_precision": summary["gt_evaluation"]["static_core_precision"],
                }
            )
        log_fn("depth_pretrain", 0, metrics)

    return {
        "canonical_xyz": gaussians.get_xyz.detach().cpu(),
        "motion_evidence": cores["dynamic_probability"].cpu(),
        "motion_evidence_normalized": cores["dynamic_probability"].cpu(),
        "static_consistency_score": cores["static_score"].cpu(),
        "initial_dynamic_mask": cores["dynamic_core_mask"].cpu(),
        "sequence_cores": cores,
        "summary": summary,
    }


@torch.no_grad()
def run_temporal_visibility_pretraining(
    scene,
    gaussians,
    pipe,
    dataset,
    args,
    *,
    log_fn=None,
) -> dict:
    """Execute Phase 1A on the complete, pre-compaction Gaussian scene."""
    pretrain_started = time.perf_counter()
    from gaussian_renderer import render_feature_field
    from utils.gaussian_sequence_assignment import (
        CoreThresholds,
        accumulate_sequence_evidence_to_gaussians,
        build_initial_cores,
    )
    from utils.loss_utils import _load_mask_for_view
    from utils.motion_correspondence import MotionCorrespondenceCache
    from utils.static_flow_calibration import (
        compute_camera_interval_reliability,
        evaluate_static_flow_residual,
        fit_robust_flow_null,
    )

    cameras = scene.getTrainCameras()
    groups = _frame_groups(cameras)
    if len(groups) < 2:
        raise RuntimeError("whole-sequence assignment needs at least two complete frames")
    camera_keys = [_camera_key(camera) for camera in groups[0][1]]
    camera_maps = [{_camera_key(camera): camera for camera in frame} for _, frame in groups]
    output_dir = Path(scene.model_path) / "dynamic_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = VisibilityConfig(
        depth_sigma=float(args.visibility_depth_sigma),
        present_threshold=float(args.visibility_present_threshold),
        occlusion_margin=float(args.visibility_occlusion_margin),
        vacated_margin=float(args.visibility_vacated_margin),
        min_canonical_alpha=float(args.visibility_min_canonical_alpha),
    )
    flow_cache = MotionCorrespondenceCache(dataset, args)
    visibility_by_camera = {}
    calibrations = {}
    visibility_totals = torch.zeros(4, dtype=torch.float64)
    visibility_pixels = 0

    visibility_started = time.perf_counter()
    for camera_index, key in enumerate(
        tqdm(camera_keys, desc="Phase 1A visibility", unit="camera")
    ):
        canonical_camera = camera_maps[0][key]
        height, width = int(canonical_camera.image_height), int(canonical_camera.image_width)
        canonical = build_canonical_visibility_cache(
            canonical_camera,
            gaussians,
            pipe,
            use_trained_exp=dataset.train_test_exp,
        )[key]
        canonical_metric = canonical["canonical_depth"].cpu()
        alpha = canonical["canonical_alpha"].cpu()
        observed = []
        for frame_map in camera_maps:
            invdepth = getattr(frame_map[key], "invdepthmap", None)
            if invdepth is None:
                raise RuntimeError(
                    f"sequence assignment requires depth for {key} frame "
                    f"{getattr(frame_map[key], 'frame_id', '?')}"
                )
            invdepth = _resize_map(invdepth, (height, width), "nearest")[0].cpu()
            observed.append(inverse_depth_to_metric(invdepth))
        emissions = compute_depth_state_likelihoods(
            canonical_metric, torch.stack(observed), alpha, cfg
        )
        smoothed = smooth_visibility_sequence(
            emissions,
            enabled=bool(args.visibility_hmm),
            stay_probability=cfg.hmm_stay_probability,
        )
        state = smoothed["state"].cpu()
        identity = build_flow_identity_mask(state).cpu()
        counts = torch.bincount(state.flatten().long(), minlength=4).double()
        visibility_totals += counts
        visibility_pixels += int(state.numel())
        visibility_by_camera[key] = {
            "state": state,
            "identity": identity.half(),
            "canonical_depth": canonical_metric.half(),
            "canonical_alpha": alpha.half(),
            "observed_depth": torch.stack(observed).half(),
        }
        for index in sorted({0, len(groups) // 2, len(groups) - 1}):
            save_visibility_debug_image(
                output_dir / f"visibility_{key}_frame_{groups[index][0]:06d}.png",
                state[index], canonical_metric, observed[index],
            )
    visibility_seconds = time.perf_counter() - visibility_started

    def evidence_records():
        progress = tqdm(
            total=len(camera_keys) * (len(groups) - 1),
            desc="Phase 1A calibrated flow + ownership",
            unit="pair",
        )
        for camera_index, key in enumerate(camera_keys):
            cache = visibility_by_camera[key]
            state = cache["state"].long()
            identity = cache["identity"].float()
            alpha = cache["canonical_alpha"].float()
            stable_present = (state == PRESENT).float().mean(dim=0) >= 0.90
            never_vacated = ~(state == VACATED).any(dim=0)
            canonical_camera = camera_maps[0][key]
            height, width = state.shape[-2:]
            try:
                semantic = _load_mask_for_view(
                    canonical_camera, dataset, args, (height, width), device="cpu"
                )
            except Exception:
                semantic = None
            background = torch.zeros((height, width), dtype=torch.bool)
            if semantic is not None:
                background = semantic.cpu() < 0.05
            known_static = (background | (stable_present & never_vacated))
            for interval in range(len(groups) - 1):
                source = camera_maps[interval][key]
                target = camera_maps[interval + 1][key]
                observation = flow_cache.get(source, target)
                flow = _resize_map(observation.flow, (height, width), "bilinear")
                raw_confidence = observation.calibration_confidence
                if raw_confidence is None:
                    raw_confidence = observation.confidence
                raw_confidence = _resize_map(raw_confidence, (height, width), "bilinear")[0].clamp(0, 1)
                calibration_mask = known_static & (identity[interval] > 0.5)
                if bool(args.static_flow_calibration):
                    calibration = fit_robust_flow_null(
                        flow, calibration_mask, raw_confidence,
                        mad_floor=float(args.static_flow_mad_floor),
                        min_samples=int(args.static_flow_min_samples),
                    )
                    reliability = compute_camera_interval_reliability(
                        calibration, min_samples=int(args.static_flow_min_samples)
                    )
                else:
                    calibration = {
                        "median": torch.zeros(2),
                        "mad": torch.full((2,), float(args.multiview_flow_noise_px)),
                        "scale": torch.tensor(float(args.multiview_flow_noise_px)),
                        "sample_count": int(calibration_mask.sum()),
                        "reliable": True,
                    }
                    reliability = 1.0
                calibration_cpu = {
                    "median": calibration["median"].cpu(),
                    "mad": calibration["mad"].cpu(),
                    "scale": calibration["scale"].cpu(),
                    "sample_count": calibration["sample_count"],
                    "reliable": calibration["reliable"],
                    "camera_reliability": reliability,
                }
                calibrations[(key, interval)] = calibration_cpu
                normalized = evaluate_static_flow_residual(flow, calibration)["normalized_residual"]
                calibrated_static = normalized[calibration_mask & (raw_confidence > 0.05)]
                calibration_cpu["static_normalized_residual_median"] = (
                    float(calibrated_static.median()) if calibrated_static.numel() else 0.0
                )
                foreground_confidence = _resize_map(
                    observation.confidence, (height, width), "bilinear"
                )[0].clamp(0, 1)
                valid_flow = identity[interval] * foreground_confidence * float(reliability)
                present = ((state[interval] == PRESENT) & (state[interval + 1] == PRESENT)).float()
                occluded = ((state[interval] == OCCLUDED) | (state[interval + 1] == OCCLUDED)).float()
                vacated = (state[interval + 1] == VACATED).float()
                features = torch.stack((
                    present,
                    occluded,
                    vacated,
                    valid_flow * (normalized <= 2.0).float(),
                    valid_flow * (normalized >= 4.0).float(),
                    valid_flow,
                ))
                if interval in {0, (len(groups) - 2) // 2, len(groups) - 2}:
                    _save_scalar_debug(
                        output_dir / f"flow_identity_{key}_interval_{interval:06d}.png",
                        identity[interval], maximum=1.0,
                    )
                    _save_scalar_debug(
                        output_dir / f"static_flow_score_{key}_interval_{interval:06d}.png",
                        normalized, maximum=8.0,
                    )
                yield {
                    "camera": canonical_camera,
                    "camera_index": camera_index,
                    "step_index": interval,
                    "pixel_features": features,
                    "pixel_confidence": (alpha >= cfg.min_canonical_alpha).float(),
                }
                progress.set_postfix(
                    camera=key,
                    interval=interval,
                    static_px=int(calibration["sample_count"]),
                    noise=f"{float(calibration['scale']):.3f}",
                )
                progress.update(1)
        progress.close()

    evidence_started = time.perf_counter()
    gaussian_evidence = accumulate_sequence_evidence_to_gaussians(
        gaussians, pipe, evidence_records(),
        num_cameras=len(camera_keys), num_steps=len(groups) - 1,
    )
    flow_evidence_seconds = time.perf_counter() - evidence_started
    cores = build_initial_cores(
        gaussian_evidence,
        CoreThresholds(
            dynamic_score=float(args.initial_dynamic_core_min_score),
            dynamic_time_fraction=float(args.initial_dynamic_core_min_time_fraction),
            dynamic_camera_fraction=float(args.initial_dynamic_core_min_camera_fraction),
            static_score=float(args.initial_static_core_min_score),
            min_observations=int(args.initial_core_min_observations),
        ),
    )
    core_colors = torch.full(
        (gaussians.get_xyz.shape[0], 3), 0.5,
        device=gaussians.get_xyz.device, dtype=torch.float32,
    )
    core_colors[cores["static_core_mask"].to(core_colors.device)] = torch.tensor(
        (0.15, 0.75, 0.20), device=core_colors.device
    )
    core_colors[cores["dynamic_core_mask"].to(core_colors.device)] = torch.tensor(
        (0.90, 0.15, 0.10), device=core_colors.device
    )
    core_render = render_feature_field(
        camera_maps[0][camera_keys[0]], gaussians, pipe, core_colors,
        bg_feature=torch.zeros(3, device=core_colors.device),
        detach_geometry=True,
    )["render"].detach().cpu().permute(1, 2, 0).clamp(0, 1)
    from PIL import Image
    Image.fromarray((core_render * 255).byte().numpy()).save(
        output_dir / "initial_gaussian_cores.png"
    )
    fractions = visibility_totals / max(1, visibility_pixels)
    calibration_scales = [float(value["scale"]) for value in calibrations.values()]
    calibration_residuals = [
        float(value["static_normalized_residual_median"])
        for value in calibrations.values()
    ]
    calibration_reliability = [
        float(value["camera_reliability"]) for value in calibrations.values()
    ]
    identity_values = [
        value["identity"].float().mean() for value in visibility_by_camera.values()
    ]
    identity_fraction = float(torch.stack(identity_values).mean())
    if identity_fraction <= 1.0e-6:
        raise RuntimeError("sequence pretraining found no identity-valid flow pixels")
    summary = {
        "schema_version": 1,
        "frames": len(groups),
        "cameras": len(camera_keys),
        "present_fraction": float(fractions[PRESENT]),
        "occluded_fraction": float(fractions[OCCLUDED]),
        "vacated_fraction": float(fractions[VACATED]),
        "unknown_fraction": float(fractions[UNKNOWN]),
        "static_flow_residual_median": float(torch.tensor(calibration_residuals).median()) if calibration_residuals else 0.0,
        "static_flow_noise_scale_median_px": float(torch.tensor(calibration_scales).median()) if calibration_scales else 0.0,
        "identity_valid_fraction": identity_fraction,
        "camera_reliability_mean": (
            float(torch.tensor(calibration_reliability).mean())
            if calibration_reliability else 0.0
        ),
        "dynamic_core_size": int(cores["dynamic_core_mask"].sum()),
        "static_core_size": int(cores["static_core_mask"].sum()),
        "unknown_size": int(cores["unknown_mask"].sum()),
        "ground_truth_used_as_algorithm_input": False,
        "runtime": {
            "visibility_seconds": visibility_seconds,
            "flow_evidence_seconds": flow_evidence_seconds,
            "total_seconds_before_artifact_write": time.perf_counter()
            - pretrain_started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    visibility_artifact = {
        "schema_version": 1,
        "camera_keys": camera_keys,
        "frame_ids": [frame_id for frame_id, _ in groups],
        "states": {key: value["state"] for key, value in visibility_by_camera.items()},
        "identity_masks": {key: value["identity"] for key, value in visibility_by_camera.items()},
        "canonical_depth": {key: value["canonical_depth"] for key, value in visibility_by_camera.items()},
        "canonical_alpha": {key: value["canonical_alpha"] for key, value in visibility_by_camera.items()},
    }
    _atomic_torch_save(
        visibility_artifact, output_dir / "phase1_visibility_sequence_pixel.pt"
    )
    _atomic_torch_save(
        {"schema_version": 1, "calibration": calibrations},
        output_dir / "phase1_static_flow_calibration.pt",
    )
    _atomic_torch_save(
        gaussian_evidence, output_dir / "phase1_gaussian_sequence_evidence.pt"
    )
    _atomic_torch_save(cores, output_dir / "phase1_initial_gaussian_cores.pt")
    with (output_dir / "phase1_visibility_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    with (output_dir / "phase1_static_flow_summary.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "camera_intervals": len(calibrations),
            "reliable_intervals": sum(bool(value["reliable"]) for value in calibrations.values()),
            "median_scale_px": summary["static_flow_noise_scale_median_px"],
            "median_normalized_residual": summary["static_flow_residual_median"],
        }, stream, indent=2)
    if log_fn is not None:
        log_fn("sequence_pretrain", 0, {
            "visibility/present_fraction": summary["present_fraction"],
            "visibility/occluded_fraction": summary["occluded_fraction"],
            "visibility/vacated_fraction": summary["vacated_fraction"],
            "flow_calibration/static_residual_median": summary["static_flow_residual_median"],
            "sequence/identity_valid_fraction": summary["identity_valid_fraction"],
            "camera/reliability_mean": summary["camera_reliability_mean"],
            "assignment_pretrain/dynamic_core_size": summary["dynamic_core_size"],
            "assignment_pretrain/static_core_size": summary["static_core_size"],
            "assignment_pretrain/unknown_size": summary["unknown_size"],
        })
    return {
        "canonical_xyz": gaussians.get_xyz.detach().cpu(),
        "motion_evidence": cores["dynamic_probability"].cpu(),
        "motion_evidence_normalized": cores["dynamic_probability"].cpu(),
        "static_consistency_score": cores["static_score"].cpu(),
        "initial_dynamic_mask": cores["dynamic_core_mask"].cpu(),
        "sequence_cores": cores,
        "summary": summary,
    }
