"""Per-outer-cycle artefacts, numerical diagnostics and visual diagnostics.

The closed loop is four modules correcting each other, so when it goes wrong the
question is never "is it wrong" but "which arrow broke".  Everything here exists
to answer that: each outer cycle writes a complete, self-contained directory, and
the numeric summary is designed so the known failure modes of spec 40 are
readable off a single line.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def _atomic_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _cpu_payload(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_payload(item) for item in value)
    return value


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@torch.no_grad()
def collect_diagnostics(
    state,
    *,
    responsibility_delta: float,
    consensus_velocity: torch.Tensor | None = None,
    killing_diagnostics: list | None = None,
    diffusion_diagnostics: dict | None = None,
    residual_decomposition: list | None = None,
    previous_labels: torch.Tensor | None = None,
    previous_modes: list | None = None,
    energy_scales: dict | None = None,
) -> dict:
    """One flat record per outer cycle, covering every quantity spec 38 asks for."""
    labels = state.hard_labels
    share = state.responsibilities
    entropy = state.assignment_entropy
    moving = share[:, 1:].sum(-1)
    ordered = share.sort(-1, descending=True).values
    ambiguous = (ordered[:, 0] < 0.8) | ((ordered[:, 0] - ordered[:, 1]) < 0.15)

    record: dict = {
        "outer_iteration": int(state.outer_iteration),
        "counts": state.label_counts(),
        "ambiguous_count": int(ambiguous.sum()),
        "moving_fraction": float((labels > 0).float().mean()),
        "mean_assignment_entropy": float(entropy.mean()),
        "p50_assignment_entropy": float(entropy.median()),
        "p90_assignment_entropy": float(entropy.quantile(0.9)),
        "responsibility_delta": _finite(float(responsibility_delta)),
        "mean_moving_probability": float(moving.mean()),
        "static_core_size": int(state.static_core_mask.sum()),
        "shell_size": int(state.shell_masks.any(0).sum()),
    }
    for part in range(state.num_moving_parts):
        record[f"core_size_part_{part + 1}"] = int(state.core_masks[part].sum())

    if previous_labels is not None:
        changed = labels != previous_labels.to(labels.device)
        record["label_changes"] = int(changed.sum())
        record["promotions_to_moving"] = int(((previous_labels == 0) & (labels > 0)).sum())
        record["demotions_to_static"] = int(((previous_labels > 0) & (labels == 0)).sum())
        record["part_switches"] = int(
            ((previous_labels > 0) & (labels > 0) & changed).sum()
        )

    # --- Killing geometry and its stability -----------------------------
    for part, mode in enumerate(state.killing_modes):
        prefix = f"part_{part + 1}"
        record[f"{prefix}_valid"] = bool(mode.valid)
        if mode.valid:
            axis = mode.angular_mode.double()
            norm = float(axis.norm())
            record[f"{prefix}_axis"] = (axis / max(norm, 1.0e-12)).tolist()
            record[f"{prefix}_rigid_rms"] = _finite(
                float(mode.diagnostics.get("rigid_rms", float("nan")))
            )
            record[f"{prefix}_static_rms"] = _finite(
                float(mode.diagnostics.get("static_rms", float("nan")))
            )
            if previous_modes is not None and part < len(previous_modes):
                other = previous_modes[part]
                if other is not None and other.valid:
                    before = other.angular_mode.double()
                    before = before / before.norm().clamp_min(1.0e-12)
                    cosine = abs(float((axis / max(norm, 1.0e-12)) @ before))
                    record[f"{prefix}_axis_change_degrees"] = math.degrees(
                        math.acos(min(1.0, max(-1.0, cosine)))
                    )
        speeds = state.killing_interval_speeds[part]
        record[f"{prefix}_speed_mean"] = float(speeds.mean())
        record[f"{prefix}_speed_max_abs"] = float(speeds.abs().max())

    # --- how far the modalities are from the physics --------------------
    valid = state.observed_valid
    for part in range(state.num_moving_parts):
        killing = state.killing_velocity[part]
        record[f"fm_minus_killing_part_{part + 1}"] = float(
            (state.fm_velocity - killing).norm(dim=-1).mean()
        )
        if bool(valid.any()):
            difference = (state.observed_velocity - killing).norm(dim=-1)
            record[f"observed_minus_killing_part_{part + 1}"] = float(difference[valid].mean())
    if consensus_velocity is not None:
        record["consensus_speed_mean"] = float(consensus_velocity.norm(dim=-1).mean())

    # --- diffusion residual, split by population ------------------------
    residual = state.diffusion_residual_mean.norm(dim=-1)
    active = state.diffusion_confidence > 0
    def population(mask: torch.Tensor) -> float | None:
        selected = active & mask[None].expand_as(active)
        return float(residual[selected].mean()) if bool(selected.any()) else None

    record["diffusion_residual_global"] = population(torch.ones_like(labels, dtype=torch.bool))
    record["diffusion_residual_static_assigned"] = population(labels == 0)
    record["diffusion_residual_moving_assigned"] = population(labels > 0)
    record["diffusion_residual_core"] = population(state.core_masks.any(0))
    record["diffusion_residual_shell"] = population(state.shell_masks.any(0))

    if residual_decomposition:
        for item in residual_decomposition:
            if not item.get("available"):
                continue
            part = item.get("part", "?")
            record[f"residual_parallel_part_{part}"] = _finite(item["parallel_norm_mean"])
            record[f"residual_orthogonal_part_{part}"] = _finite(item["orthogonal_norm_mean"])
            record[f"residual_parallel_ratio_part_{part}"] = _finite(
                item["parallel_over_orthogonal"]
            )
            record[f"residual_orthogonal_coherence_part_{part}"] = _finite(
                item["orthogonal_coherence"]
            )

    if diffusion_diagnostics:
        record.update({f"diffusion_{k}": _finite(v) for k, v in diffusion_diagnostics.items()
                       if not isinstance(v, (list, dict))})
    if killing_diagnostics:
        record["killing_refit"] = killing_diagnostics
    if energy_scales:
        record["energy_scales"] = {k: _finite(v) for k, v in energy_scales.items()}
    record.update(detect_failure_modes(record))
    return record


def detect_failure_modes(record: dict) -> dict:
    """Name the spec-40 failures directly rather than leaving them to be inferred.

    Each is a statement about numbers already in the record, so the warning and
    the evidence for it always travel together.
    """
    warnings = list(record.get("warnings", []))
    moving = record.get("moving_fraction", 0.0)
    if moving > 0.95:
        warnings.append("everything_moves: raise move_activation_cost or gamma, or gate the core harder")
    if moving < 0.005:
        warnings.append("everything_static: check energy normalisation and the diffusion energy scale")
    base = record.get("fm_base_velocity_norm_mean")
    current = record.get("fm_current_velocity_norm_mean")
    fm_pref = record.get("fm_3d_moving_preferred_fraction")
    observed_pref = record.get("observed_velocity_moving_preferred_fraction")
    if (
        base is not None and current is not None and base > 1.0e-8
        and current < 0.25 * base
        and fm_pref is not None and fm_pref < 0.05
        and (observed_pref is None or observed_pref > 0.05)
    ):
        warnings.append("FM_COLLAPSE_TOWARD_STATIC: restored FM is being erased by feedback")
    for key, value in record.items():
        if key.endswith("_axis_change_degrees") and value is not None and value > 20.0:
            warnings.append(f"{key}={value:.1f}: unstable axis, suspect a contaminated core")
    # FM and Killing agreeing while the renderer still complains is collusion:
    # the two have converged on each other rather than on the images.
    fm_gap = [v for k, v in record.items() if k.startswith("fm_minus_killing_") and v is not None]
    residual = record.get("diffusion_residual_core")
    if fm_gap and residual is not None and min(fm_gap) < 1.0e-3 and residual > 1.0e-2:
        warnings.append("fm_killing_collusion: lower lambda_fm_killing, trust the renderer more")
    for key, value in record.items():
        if key.startswith("residual_parallel_ratio_part_") and value is not None:
            part = key.rsplit("_", 1)[-1]
            coherence = record.get(f"residual_orthogonal_coherence_part_{part}")
            if value > 3.0:
                warnings.append(f"part {part}: residual mostly parallel -> speed or amplitude error")
            elif coherence is not None and coherence > 0.5:
                warnings.append(f"part {part}: residual coherently orthogonal -> Killing geometry error")
            elif coherence is not None and coherence < 0.2 and value < 1.0:
                warnings.append(f"part {part}: residual scattered orthogonal -> assignment error")
    return {"warnings": warnings}


def respond_to_failure_modes(record: dict, weights, consensus, core_config, *, log=print) -> dict:
    """Act on the failures of spec 40 instead of only naming them.

    The specification gives a response for each mode, and leaving them as text
    in a log means the run continues into the wall it just described.  Each
    adjustment below is the one the specification names for that mode, applied
    once per cycle and reported so the change is never silent.

    Deliberately conservative: one step per cycle, and nothing is loosened --
    only the safeguards tighten.  A loop that oscillates between responses is
    worse than one that drifts slowly in a diagnosable direction.
    """
    applied = []
    for warning in record.get("warnings", []):
        if warning.startswith("everything_moves"):
            # Spec 40.1: raise the cost of claiming motion, the static
            # regularisation, the core gate and the responsibility exponent.
            weights.move_activation_cost = weights.move_activation_cost * 2.0 + 0.05
            consensus.static *= 1.5
            consensus.gamma = min(4.0, consensus.gamma + 0.5)
            core_config.enter = min(0.98, core_config.enter + 0.02)
            applied.append(
                f"everything_moves -> move_cost={weights.move_activation_cost:.3f} "
                f"static={consensus.static:.2f} gamma={consensus.gamma:.1f} "
                f"core_enter={core_config.enter:.2f}"
            )
        elif warning.startswith("everything_static"):
            # Spec 40.2: the move cost and the diffusion scale are the suspects.
            # Lowering the move cost is the only safe direction -- the energies
            # are already normalised, so the remaining free parameter is how
            # much motion has to prove itself.
            weights.move_activation_cost = max(0.0, weights.move_activation_cost * 0.5)
            applied.append(
                f"everything_static -> move_cost={weights.move_activation_cost:.3f}"
            )
        elif "unstable axis" in warning:
            # Spec 40.3: contaminated core.  Gate it harder and weight the
            # confident points more.
            core_config.enter = min(0.98, core_config.enter + 0.03)
            core_config.persistence_cycles = min(6, core_config.persistence_cycles + 1)
            consensus.gamma = min(4.0, consensus.gamma + 0.5)
            applied.append(
                f"unstable_axis -> core_enter={core_config.enter:.2f} "
                f"persistence={core_config.persistence_cycles} gamma={consensus.gamma:.1f}"
            )
        elif warning.startswith("fm_killing_collusion"):
            # Spec 40.4: FM has converged on Killing rather than on the images.
            weights.flow_matching = max(0.1, weights.flow_matching * 0.5)
            applied.append(f"fm_killing_collusion -> fm_weight={weights.flow_matching:.3f}")
    if applied:
        for line in applied:
            log(f"    response: {line}")
    return {"applied": applied}


@torch.no_grad()
def save_closed_loop_debug(
    state,
    output_root,
    *,
    consensus_velocity: torch.Tensor | None = None,
    killing_transforms: torch.Tensor | None = None,
    energies: dict | None = None,
    diagnostics: dict | None = None,
) -> Path:
    """Write one self-contained directory for this outer cycle (spec 37)."""
    directory = Path(output_root) / f"outer_{int(state.outer_iteration):04d}"
    directory.mkdir(parents=True, exist_ok=True)

    tensors = {
        "responsibilities": state.responsibilities,
        "hard_labels": state.hard_labels,
        "assignment_entropy": state.assignment_entropy,
        "fm_base_displacement": state.fm_base_displacement,
        "fm_current_displacement": state.fm_current_displacement,
        "physical_displacement": state.physical_displacement,
        "fm_base_velocity": state.fm_base_velocity,
        "fm_velocity": state.fm_velocity,
        "fm_confidence": state.fm_confidence,
        "fm_physical_3d_residual": state.fm_physical_3d_residual,
        "killing_interval_speeds": state.killing_interval_speeds,
        "killing_velocity": state.killing_velocity,
        "physical_velocity": state.physical_velocity,
        "diffusion_residual_mean": state.diffusion_residual_mean,
        "diffusion_residual_cov": state.diffusion_residual_cov,
        "diffusion_confidence": state.diffusion_confidence,
        "core_masks": state.core_masks,
        "shell_masks": state.shell_masks,
        "static_core_mask": state.static_core_mask,
        "energy_fm_3d": state.energy_fm_3d,
        "energy_fm_render": state.energy_fm_render,
        "energy_obs_render": state.energy_obs_render,
        "energy_target_render": state.energy_target_render,
    }
    if consensus_velocity is not None:
        tensors["consensus_velocity"] = consensus_velocity
    if killing_transforms is not None:
        tensors["killing_transforms"] = killing_transforms
    for name, value in tensors.items():
        _atomic_save(_cpu_payload(value), directory / f"{name}.pt")

    _atomic_save(
        [mode.as_dict() for mode in state.killing_modes], directory / "killing_modes.pt"
    )
    for name, value in (energies or {}).items():
        if value is not None:
            _atomic_save(torch.as_tensor(value).detach().cpu(), directory / f"energy_{name}.pt")
    if diagnostics is not None:
        (directory / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8"
        )
    return directory


def format_diagnostics_line(record: dict, num_parts: int) -> str:
    """A single readable line per cycle for the console."""
    counts = record["counts"]
    parts = " ".join(
        f"p{index}={counts.get(f'part_{index}', 0)}" for index in range(1, num_parts + 1)
    )
    delta = record.get("responsibility_delta")
    return (
        f"[outer {record['outer_iteration']:03d}] "
        f"static={counts['static']} {parts} amb={record['ambiguous_count']} "
        f"H={record['mean_assignment_entropy']:.3f} "
        f"dR={'inf' if delta is None else f'{delta:.5f}'} "
        f"core={'/'.join(str(record.get(f'core_size_part_{i}', 0)) for i in range(1, num_parts + 1))} "
        f"res={_format(record.get('diffusion_residual_global'))}"
    )


def _format(value) -> str:
    return "n/a" if value is None else f"{value:.4g}"


@torch.no_grad()
def write_closed_loop_html(
    state,
    path,
    *,
    consensus_velocity: torch.Tensor | None = None,
    previous_labels: torch.Tensor | None = None,
    max_points: int = 20000,
    background_points: torch.Tensor | None = None,
    title: str = "Closed-loop motion state",
) -> Path:
    """Interactive point cloud of every field spec 39 asks to see.

    Background Gaussians are drawn as grey context only and carry no motion
    channel, which is also a visual check of the foreground-only invariant: if
    anything grey ever moves, the domain leaked.
    """
    points = state.canonical_fg.detach().cpu()
    count = points.shape[0]
    if count > int(max_points):
        stride = torch.linspace(0, count - 1, int(max_points)).round().long().unique()
    else:
        stride = torch.arange(count)
    sampled = points[stride]

    share = state.responsibilities.detach().cpu()[stride]
    channels = {
        "static probability": share[:, 0],
        "moving probability": share[:, 1:].sum(-1),
        "hard label": state.hard_labels.detach().cpu()[stride].float(),
        "assignment entropy": state.assignment_entropy.detach().cpu()[stride],
        "core membership": state.core_masks.any(0).detach().cpu()[stride].float(),
        "shell membership": state.shell_masks.any(0).detach().cpu()[stride].float(),
        "diffusion residual norm": state.diffusion_residual_mean.norm(dim=-1).mean(0).cpu()[stride],
    }
    for part in range(state.num_moving_parts):
        channels[f"probability part {part + 1}"] = share[:, part + 1]
    if state.fm_velocity.numel():
        channels["fm base speed"] = state.fm_base_velocity.norm(dim=-1).mean(0).detach().cpu()[stride]
        channels["fm speed"] = state.fm_velocity.norm(dim=-1).mean(0).detach().cpu()[stride]
        channels["fm current-base speed"] = (
            (state.fm_velocity - state.fm_base_velocity).norm(dim=-1).mean(0).detach().cpu()[stride]
        )
    if state.physical_velocity.numel():
        channels["physical speed"] = state.physical_velocity.norm(dim=-1).mean(0).detach().cpu()[stride]
    if state.fm_current_displacement.numel():
        channels["fm base displacement"] = state.fm_base_displacement.norm(dim=-1).detach().cpu()[stride]
        channels["fm current displacement"] = state.fm_current_displacement.norm(dim=-1).detach().cpu()[stride]
        channels["physical displacement"] = state.physical_displacement.norm(dim=-1).detach().cpu()[stride]
        channels["fm base minus physical"] = (
            (state.fm_base_displacement - state.physical_displacement).norm(dim=-1).detach().cpu()[stride]
        )
        channels["fm current minus physical"] = state.fm_physical_3d_residual.detach().cpu()[stride]
    if state.energy_fm_3d.numel():
        channels["FM3D prefers moving"] = (
            state.energy_fm_3d[:, 1:].amin(-1) < state.energy_fm_3d[:, 0]
        ).detach().cpu().float()[stride]
    if state.energy_fm_render.numel():
        channels["FM rendered prefers moving"] = (
            state.energy_fm_render[:, 1:].amin(-1) < state.energy_fm_render[:, 0]
        ).detach().cpu().float()[stride]
    if state.killing_velocity.numel():
        killing = state.killing_velocity.norm(dim=-1).mean(1).mean(0)
        channels["killing speed"] = killing.detach().cpu()[stride]
        channels["fm minus killing"] = (
            (state.fm_velocity - state.killing_velocity[0]).norm(dim=-1).mean(0).detach().cpu()[stride]
        )
    if consensus_velocity is not None:
        channels["consensus speed"] = (
            consensus_velocity.norm(dim=-1).mean(0).detach().cpu()[stride]
        )
    if previous_labels is not None:
        changed = (state.hard_labels != previous_labels.to(state.hard_labels.device))
        channels["label changed"] = changed.detach().cpu()[stride].float()

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_channel_scatter(
        destination,
        sampled,
        {name: value.float() for name, value in channels.items()},
        title=f"{title} (outer {state.outer_iteration})",
        context_points=None if background_points is None else background_points.detach().cpu(),
    )
    return destination


def _write_channel_scatter(path: Path, points, channels: dict, *, title: str, context_points=None):
    """One orbitable scene with a dropdown over every per-Gaussian channel.

    Written directly rather than through a shared helper because the closed loop
    needs a channel *selector* -- the whole diagnostic value is in flipping
    between, say, the assignment and the residual on the same geometry without
    the camera moving, so all channels share one trace and one fixed scene.
    """
    from utils.html_plot_layout import fixed_scene_bounds, left_colorbar, plotly_3d_config, plotly_3d_layout

    names = list(channels)
    if not names:
        raise ValueError("the scatter needs at least one channel")
    xyz = torch.as_tensor(points).detach().cpu().float()
    data = []
    if context_points is not None and context_points.numel():
        context = torch.as_tensor(context_points).detach().cpu().float()
        data.append({
            "type": "scatter3d", "mode": "markers", "name": "background (context)",
            "x": context[:, 0].tolist(), "y": context[:, 1].tolist(), "z": context[:, 2].tolist(),
            "marker": {"size": 1.2, "color": "rgba(150,150,150,0.35)"},
            "hoverinfo": "skip",
        })
    first = channels[names[0]]
    data.append({
        "type": "scatter3d", "mode": "markers", "name": "foreground",
        "x": xyz[:, 0].tolist(), "y": xyz[:, 1].tolist(), "z": xyz[:, 2].tolist(),
        "marker": {
            "size": 2.0,
            "color": first.tolist(),
            "colorscale": "Viridis",
            "colorbar": left_colorbar(names[0]),
            "showscale": True,
        },
        "text": [f"index {index}" for index in range(xyz.shape[0])],
    })
    channel_index = len(data) - 1
    buttons = [{
        "type": "dropdown",
        "buttons": [
            {
                "label": name,
                "method": "restyle",
                "args": [
                    {"marker.color": [channels[name].tolist()],
                     "marker.colorbar.title": name},
                    [channel_index],
                ],
            }
            for name in names
        ],
    }]
    bounds = fixed_scene_bounds(xyz)
    payload = {
        "data": data,
        "layout": plotly_3d_layout(title, buttons=buttons, scene_bounds=bounds),
    }
    path.write_text(
        "<html><head><meta charset='utf-8'>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head>"
        "<body><div id='plot' style='width:100vw;height:100vh'></div>"
        f"<script>const p={json.dumps(payload)};const config={json.dumps(plotly_3d_config())};"
        "Plotly.newPlot('plot',p.data,p.layout,config);</script></body></html>",
        encoding="utf-8",
    )
