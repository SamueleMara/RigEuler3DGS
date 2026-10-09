"""Inner loop: local Gaussian velocity field + analytic PARTWISE twist (FIRM v4).

The trainable quantity is one physical velocity per Gaussian, ``v_i`` (§1). Its
Jacobian supplies LOCAL strain rigidity (§4) and local skew diagnostics (§5).

The transported motion, however, is PARTWISE: for each moving part one
instantaneous twist ``xi_k = (omega_k, b_k)`` is fitted analytically from ALL
weighted Gaussian velocities of that part (§6-§8) by solving ``H_k omega_k = g_k``
-- no Kabsch -- and every Gaussian evaluated under candidate k is transported by
the SAME ``Delta T_k = Exp(xi_k dt)`` (§11, §12).

v3's mistake, corrected here: per-Gaussian ``xi_ik`` were integrated
independently, which is a field of independent rigid transforms rather than one
rigid body. ``local velocity field != local independent rigid transforms``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

from gaussian_renderer import render as render_gaussian
from utils.loss_utils import ssim
from utils.motion_correspondence import MotionCorrespondenceCache
from utils.rigid_transform_utils import (
    apply_rotation_to_gaussian_quaternions,
    normalize_quaternion,
    quaternion_multiply,
    se3_exp_spatial,
)

from .local_killing import (
    analytic_partwise_twist,
    effective_obs_lambda,
    evaluate_local_field,
    field_separation_loss,
    integrate_local_se3,
    local_killing_quantities,
    observed_velocity_loss,
    partwise_motion_separation,
    vee,
)
from .neighborhood import NeighborGraph, build_neighbor_graph, edge_part_weights


@dataclass
class NestedKillingMotionState:
    times: torch.Tensor
    positions: torch.Tensor
    rotations: torch.Tensor
    part_rotation: torch.Tensor
    part_translation: torch.Tensor
    part_quaternion: torch.Tensor
    omega: torch.Tensor
    b: torch.Tensor
    center_velocity: torch.Tensor
    strain: torch.Tensor
    motion_separation: torch.Tensor
    history: list[dict[str, float]]
    field_debug: list[dict[str, torch.Tensor | float | int]]
    # FIRM v3 §25: the real endpoint candidates are per-Gaussian, not a rigid
    # transform applied to the cloud.
    moving_endpoints: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 0, 3)))
    moving_endpoint_quaternions: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 0, 4)))
    acceptance: dict[str, float] = field(default_factory=dict)
    # Render-centric spec §18: detached per-Gaussian evidence [N, K] answering
    # "does Gaussian i actually behave like the part whose label it holds?"
    kinetic_energy: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 0)))
    inferred_speed: torch.Tensor = field(default_factory=lambda: torch.zeros((0,)))


def _vee(skew: torch.Tensor) -> torch.Tensor:  # kept for the existing tests
    return vee(skew)


def _jsonable(value):
    if torch.is_tensor(value):
        v = value.detach().cpu()
        return float(v) if v.numel() == 1 else v.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _compose_hard_partition_scene(
    xyz0: torch.Tensor,
    rot0: torch.Tensor,
    moving_xyz: torch.Tensor,
    moving_quat: torch.Tensor,
    hard_label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FINAL spec §3/§24: ONE concrete articulated scene, no soft blending.

        x_i(t) = x_i^0          if z_i = 0
               = T_k(t) x_i^0   if z_i = k > 0

    A Gaussian must never sit 48% with the moving hypothesis and 52% with the
    static one inside the physical scene. Implemented as a one-hot gather so the
    selection is exact and gradients reach only the selected class.
    """
    xyz_stack = torch.cat((xyz0[None], moving_xyz), dim=0)          # [K+1,N,3]
    quat_stack = torch.cat((rot0[None], moving_quat), dim=0)        # [K+1,N,4]
    k_plus_1 = int(xyz_stack.shape[0])
    z = torch.as_tensor(hard_label, device=xyz0.device).reshape(-1).long().clamp_(0, k_plus_1 - 1)
    select = torch.zeros((k_plus_1, int(xyz0.shape[0]), 1), dtype=xyz0.dtype, device=xyz0.device)
    select.scatter_(0, z[None, :, None], 1.0)
    xyz = (select * xyz_stack).sum(dim=0)
    quat = normalize_quaternion((select * quat_stack).sum(dim=0))
    return xyz, quat


def _compose_soft_positions(
    xyz0: torch.Tensor,
    rot0: torch.Tensor,
    moving_xyz: torch.Tensor,
    moving_quat: torch.Tensor,
    part_prob: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft membership blend. OUTER-belief use only (§36); never the inner scene."""
    xyz_stack = torch.cat((xyz0[None], moving_xyz), dim=0)
    quat_stack = torch.cat((rot0[None], moving_quat), dim=0)
    weights = part_prob.transpose(0, 1)[:, :, None]
    xyz = (weights * xyz_stack).sum(dim=0)
    quat = normalize_quaternion((weights * quat_stack).sum(dim=0))
    return xyz, quat


def _select_control_centers(points: torch.Tensor, weight: torch.Tensor, count: int) -> torch.Tensor:
    n = int(points.shape[0])
    if n <= 0:
        raise ValueError("cannot select controls from an empty point cloud")
    c = min(max(4, int(count)), n)
    w = torch.as_tensor(weight, dtype=points.dtype, device=points.device).reshape(-1)
    score = w
    if not bool(torch.isfinite(score).all()) or float(score.max().detach()) <= 0.0:
        order = torch.arange(n, device=points.device)
    else:
        order = torch.argsort(score, descending=True)
    candidate = points.index_select(0, order[: max(c, min(n, c * 4))])
    chosen = [0]
    dist2 = (candidate - candidate[0]).square().sum(dim=-1)
    for _ in range(1, c):
        idx = int(torch.argmax(dist2).detach().cpu())
        chosen.append(idx)
        dist2 = torch.minimum(dist2, (candidate - candidate[idx]).square().sum(dim=-1))
    return candidate[torch.as_tensor(chosen, device=points.device, dtype=torch.long)].detach()


def _kernel_scale(centers: torch.Tensor, override: float) -> torch.Tensor:
    if float(override) > 0.0:
        return centers.new_tensor(float(override))
    if centers.shape[0] <= 1:
        return centers.new_tensor(1.0)
    dist = torch.cdist(centers, centers)
    dist = dist.masked_fill(torch.eye(centers.shape[0], dtype=torch.bool, device=centers.device), float("inf"))
    nn = dist.min(dim=1).values
    nn = nn[torch.isfinite(nn) & (nn > 0)]
    if nn.numel() == 0:
        extent = torch.linalg.norm(centers.max(dim=0).values - centers.min(dim=0).values)
        return (0.1 * extent).clamp_min(1.0e-4)
    return nn.median().clamp_min(1.0e-4) * 1.5


def _local_field_velocity_and_jacobian(points_by_part, centers, control_velocity, sigma):
    """Backwards-compatible alias used by the existing test-suite."""
    return evaluate_local_field(points_by_part, centers, control_velocity, sigma)


def _render_camera_loss(
    camera, gaussians, pipe, background, dataset, args, positions, rotations, moving_mask=None
):
    """One camera's render loss plus its §12 validity weight.

    The validity weight is the fraction of the MOVING support this camera
    actually rasterises (``radii > 0``), floored so a view is only downweighted
    for genuinely missing supervision -- never dropped for having a high loss.
    """
    package = render_gaussian(
        camera,
        gaussians,
        pipe,
        background,
        use_trained_exp=bool(getattr(dataset, "train_test_exp", False)),
        separate_sh=False,
        means3D_override=positions,
        rotations_override=rotations,
    )
    image = package["render"]
    if getattr(camera, "alpha_mask", None) is not None:
        image = image * camera.alpha_mask.to(device=image.device)
    target = camera.original_image.to(device=image.device).clamp(0.0, 1.0)
    rgb = (image - target).abs().mean()
    dssim = 1.0 - ssim(image, target)
    depth = positions.new_tensor(0.0)
    has_depth = False
    if bool(getattr(args, "dynamic_use_depth", False)) and bool(getattr(camera, "depth_reliable", False)):
        mask = camera.depth_mask.to(device=image.device)
        depth = ((package["depth"] - camera.invdepthmap.to(device=image.device)).abs() * mask).mean()
        has_depth = True
    loss = (
        float(getattr(args, "lambda_inner_rgb", 1.0)) * rgb
        + float(getattr(args, "lambda_inner_ssim", 0.0)) * dssim
        + float(getattr(args, "lambda_inner_depth", 0.1)) * depth
    )

    # §12: eta_c in [floor, 1] from real rasteriser visibility of the moving support.
    visibility = 1.0
    if moving_mask is not None and bool(moving_mask.any()):
        radii = package.get("radii")
        if radii is not None and int(radii.numel()) == int(moving_mask.numel()):
            seen = (radii > 0) & moving_mask
            visibility = float(seen.sum()) / float(moving_mask.sum().clamp_min(1))
    floor = float(getattr(args, "multiview_visibility_floor", 0.1))
    visibility = min(1.0, max(floor, visibility))
    # CODEX_MERGED §38/§39 route L_render and L_depth to separate weights, so the
    # photometric and depth parts are returned separately as well as combined.
    photo = (float(getattr(args, "lambda_inner_rgb", 1.0)) * rgb
             + float(getattr(args, "lambda_inner_ssim", 0.0)) * dssim)
    return loss, rgb, dssim, depth, has_depth, visibility, photo


def multiview_render_loss_for_frame(
    *,
    cameras,
    gaussians,
    pipe,
    background,
    dataset,
    args,
    positions,
    rotations,
    moving_mask: torch.Tensor | None = None,
    lambda_view_var: float = 0.05,
    accumulate=None,
    eps: float = 1.0e-12,
) -> dict:
    """ALL cameras of one frame -> ONE aggregated differentiable loss (§2, §22).

    Every camera renders the SAME hard-partition scene and the SAME partwise
    transforms (§13, §14); there is no camera-specific twist and no per-camera
    optimizer step. This helper never calls ``optimizer.step()`` -- it only
    builds the loss and the diagnostics, so the single step stays outside after
    the physics terms are added (§3, §7).

        L_mv-mean = sum_c eta_c L_c / sum_c eta_c          (§6, §12)
        L_multiview = L_mv-mean + lambda_view_var * L_mv-std   (§10)

    ``accumulate`` is the optional §5 memory fallback: when given, it is called
    with each chunk's contribution to backward it immediately. The mean stays
    exact; the std term is then dropped because the per-camera graphs no longer
    coexist, and ``std_in_loss`` reports that.
    """
    if not cameras:
        zero = positions.sum() * 0.0
        return {
            "loss": zero, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "num_views": 0, "per_camera": [], "rgb": 0.0, "ssim": 0.0, "depth": 0.0,
            "std_in_loss": False, "value": 0.0,
            "photometric": zero, "depth_loss": zero, "has_depth": False,
        }

    n_cam = len(cameras)
    chunk = n_cam if accumulate is None else max(1, int(getattr(args, "killing_camera_chunk", 0)) or n_cam)
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    photo_terms: list[torch.Tensor] = []
    depth_terms: list = []
    per_camera: list[dict] = []
    accumulated_value = 0.0

    for start in range(0, n_cam, chunk):
        block = cameras[start : start + chunk]
        block_losses, block_weights = [], []
        for offset, camera in enumerate(block):
            loss_cam, rgb, dssim, depth, has_depth, visibility, photo_cam = _render_camera_loss(
                camera, gaussians, pipe, background, dataset, args,
                positions, rotations, moving_mask=moving_mask,
            )
            block_losses.append(loss_cam)
            block_weights.append(visibility)
            photo_terms.append(photo_cam)
            depth_terms.append(depth if has_depth else None)
            per_camera.append({
                "camera_id": start + offset,
                "rgb": float(rgb.detach()),
                "ssim": float(dssim.detach()),
                "depth": float(depth.detach()) if has_depth else float("nan"),
                "total_render": float(loss_cam.detach()),
                "visibility": float(visibility),
            })
        if accumulate is not None:
            # §5 memory fallback: this chunk's share of the UNIFORM mean,
            # backwarded now so the per-camera graphs can be freed. Visibility
            # weighting (§12) and the std term (§10) both need every camera loss
            # alive at once, so they are only available on the full-graph path.
            contribution = torch.stack(block_losses).sum() / n_cam
            accumulate(contribution)
            accumulated_value += float(contribution.detach())
        else:
            losses.extend(block_losses)
            weights.extend(block_weights)

    values = torch.tensor([c["total_render"] for c in per_camera], dtype=torch.float64)
    stats = {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
        "num_views": n_cam,
        "per_camera": per_camera,
        "rgb": float(np_mean([c["rgb"] for c in per_camera])),
        "ssim": float(np_mean([c["ssim"] for c in per_camera])),
        "depth": float(np_mean([c["depth"] for c in per_camera if c["depth"] == c["depth"]])),
    }

    if accumulate is not None:
        stats["loss"] = positions.sum() * 0.0
        stats["std_in_loss"] = False
        stats["value"] = accumulated_value
        # The chunked path already backwarded and freed the per-camera graphs,
        # so no separable differentiable terms survive.
        stats["photometric"] = positions.sum() * 0.0
        stats["depth_loss"] = positions.sum() * 0.0
        stats["has_depth"] = False
        return stats

    stack = torch.stack(losses)
    eta = torch.stack([w if torch.is_tensor(w) else stack.new_tensor(float(w)) for w in weights])
    # §12: downweight genuinely invalid views, never drop a view for being hard.
    mean = (eta * stack).sum() / eta.sum().clamp_min(float(eps))
    std = stack.std(unbiased=False) if stack.numel() > 1 else stack.new_tensor(0.0)
    stats["loss"] = mean + float(lambda_view_var) * std
    stats["std_in_loss"] = bool(float(lambda_view_var) > 0.0 and stack.numel() > 1)
    stats["value"] = float(stats["loss"].detach())
    stats["weighted_mean"] = float(mean.detach())
    # CODEX_MERGED §38/§39: separately weightable photometric and depth terms,
    # aggregated with the SAME visibility weights as the combined loss.
    if photo_terms:
        stats["photometric"] = (eta * torch.stack(photo_terms)).sum() / eta.sum().clamp_min(float(eps))
    else:
        stats["photometric"] = positions.sum() * 0.0
    live = [d for d in depth_terms if d is not None]
    if live:
        eta_d = torch.stack([w if torch.is_tensor(w) else stack.new_tensor(float(w))
                             for w, d in zip(weights, depth_terms) if d is not None])
        stats["depth_loss"] = (eta_d * torch.stack(live)).sum() / eta_d.sum().clamp_min(float(eps))
        stats["has_depth"] = True
    else:
        stats["depth_loss"] = positions.sum() * 0.0
        stats["has_depth"] = False
    return stats


def save_multiview_debug_images(
    *, cameras, gaussians, pipe, background, dataset, args, positions, rotations, out_dir: Path
) -> int:
    """§19: one GT / render / residual triple per camera at selected iterations."""
    try:
        import torchvision.utils as tvu
    except Exception:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with torch.no_grad():
        for idx, camera in enumerate(cameras):
            package = render_gaussian(
                camera, gaussians, pipe, background,
                use_trained_exp=bool(getattr(dataset, "train_test_exp", False)),
                separate_sh=False, means3D_override=positions, rotations_override=rotations,
            )
            image = package["render"].clamp(0.0, 1.0)
            target = camera.original_image.to(device=image.device).clamp(0.0, 1.0)
            residual = (image - target).abs().clamp(0.0, 1.0)
            tvu.save_image(target, out_dir / f"cam_{idx:03d}_gt.png")
            tvu.save_image(image, out_dir / f"cam_{idx:03d}_render.png")
            tvu.save_image(residual, out_dir / f"cam_{idx:03d}_residual.png")
            written += 3
    return written


def per_camera_gradient_norms(
    *, cameras, gaussians, pipe, background, dataset, args, positions, rotations, parameter, moving_mask=None
) -> list[float]:
    """§16: |d L_cam / d theta| per camera, to reveal a view dominating the field."""
    norms: list[float] = []
    for camera in cameras:
        loss_cam, *_ = _render_camera_loss(
            camera, gaussians, pipe, background, dataset, args, positions, rotations,
            moving_mask=moving_mask,
        )
        grad = torch.autograd.grad(loss_cam, parameter, retain_graph=True, allow_unused=True)[0]
        norms.append(0.0 if grad is None else float(grad.norm()))
    return norms


def np_mean(values):
    values = [v for v in values]
    return sum(values) / len(values) if values else 0.0


def _render_all_cameras(cameras, gaussians, pipe, background, dataset, args, positions, rotations):
    """Non-accumulating variant (single graph), used for evaluation only."""
    if not cameras:
        zero = positions.new_tensor(0.0)
        return zero, {"rgb": 0.0, "ssim": 0.0, "depth": 0.0, "views": 0}
    total = positions.new_tensor(0.0)
    rgb_t = positions.new_tensor(0.0)
    ssim_t = positions.new_tensor(0.0)
    depth_t = positions.new_tensor(0.0)
    depth_n = 0
    for camera in cameras:
        loss, rgb, dssim, depth, has_depth, _visibility, _photo = _render_camera_loss(
            camera, gaussians, pipe, background, dataset, args, positions, rotations
        )
        total = total + loss
        rgb_t = rgb_t + rgb
        ssim_t = ssim_t + dssim
        if has_depth:
            depth_t = depth_t + depth
            depth_n += 1
    n = max(1, len(cameras))
    return total / n, {
        "rgb": float(rgb_t.detach().cpu()) / n,
        "ssim": float(ssim_t.detach().cpu()) / n,
        "depth": float(depth_t.detach().cpu()) / max(1, depth_n),
        "views": len(cameras),
    }


def _sequence_motion_distance(part_rotation, part_translation, *, scale):
    rot = torch.as_tensor(part_rotation)
    trans = torch.as_tensor(part_translation, dtype=rot.dtype, device=rot.device)
    k = int(rot.shape[1])
    out = torch.zeros((k, k), dtype=rot.dtype, device=rot.device)
    denom = torch.as_tensor(scale, dtype=rot.dtype, device=rot.device).square().clamp_min(1.0e-8)
    for a in range(k):
        for bidx in range(a + 1, k):
            rel = rot[:, a].transpose(-1, -2) @ rot[:, bidx]
            trace = torch.diagonal(rel, dim1=-2, dim2=-1).sum(-1)
            angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
            dt = (trans[:, a] - trans[:, bidx]).square().sum(-1) / denom
            value = (angle.square() + dt).mean()
            out[a, bidx] = value
            out[bidx, a] = value
    return out


def _affine_killing_quantities(affine, linear, xyz_by_part, part_prob):
    """Analytic affine reference field, retained for the regression tests."""
    jac = affine
    strain = 0.5 * (jac + jac.transpose(-1, -2))
    omega_local = vee(0.5 * (jac - jac.transpose(-1, -2)))[:, None, :].expand_as(xyz_by_part)
    velocity = torch.einsum("kba,kna->knb", affine, xyz_by_part) + linear[:, None, :]
    b_local = velocity - torch.cross(omega_local, xyz_by_part, dim=-1)
    prob = part_prob[:, 1:].transpose(0, 1).to(xyz_by_part)
    mass = prob.sum(dim=1).clamp_min(1.0e-8)
    omega = (prob[:, :, None] * omega_local).sum(dim=1) / mass[:, None]
    b = (prob[:, :, None] * b_local).sum(dim=1) / mass[:, None]
    center = (prob[:, :, None] * xyz_by_part).sum(dim=1) / mass[:, None]
    center_velocity = b + torch.cross(omega, center, dim=-1)
    omega_cons = (((omega_local - omega[:, None, :]).square().sum(-1) * prob).sum(dim=1) / mass).sum()
    b_cons = (((b_local - b[:, None, :]).square().sum(-1) * prob).sum(dim=1) / mass).sum()
    return {
        "velocity": velocity,
        "strain": strain,
        "omega": omega,
        "b": b,
        "center_velocity": center_velocity,
        "omega_consistency": omega_cons,
        "b_consistency": b_cons,
    }


def _local_field_killing_quantities(centers, control_velocity, sigma, xyz_by_part, part_prob):
    """Backwards-compatible wrapper exposing the legacy key names."""
    q = local_killing_quantities(
        xyz_by_part, centers, control_velocity, sigma, part_prob[:, 1:]
    )
    q = dict(q)
    q["omega"] = q["omega_mean"]
    q["b"] = q["b_mean"]
    prob = part_prob[:, 1:].transpose(0, 1).to(xyz_by_part)
    mass = prob.sum(dim=1).clamp_min(1.0e-8)
    q["omega_consistency"] = (
        ((q["omega_local"] - q["omega_mean"][:, None, :]).square().sum(-1) * prob).sum(dim=1) / mass
    ).sum()
    q["b_consistency"] = (
        ((q["b_local"] - q["b_mean"][:, None, :]).square().sum(-1) * prob).sum(dim=1) / mass
    ).sum()
    return q


def _partwise_candidate_state(
    xyz0: torch.Tensor,
    rot0: torch.Tensor,
    part_r: torch.Tensor,
    part_t: torch.Tensor,
    part_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FIRM v4 §12: x_ik = T_k x_i^0, q_ik = q_k (x) q_i^0 -- ONE transform per part."""
    xyz = torch.einsum("kab,nb->kna", part_r, xyz0) + part_t[:, None, :]
    quat = torch.stack(
        [apply_rotation_to_gaussian_quaternions(rot0, part_q[k]) for k in range(int(part_q.shape[0]))],
        dim=0,
    )
    return xyz, quat


def _diagnostic_points(xyz0: torch.Tensor, part_prob: torch.Tensor, count: int) -> torch.Tensor:
    """§18: a shared point set for comparing local fields.

    Prefers uncertain Gaussians (support unions / boundaries), falling back to a
    uniform subsample when memberships are already confident.
    """
    n = int(xyz0.shape[0])
    c = min(max(16, int(count)), n)
    confidence = part_prob.max(dim=1).values
    order = torch.argsort(confidence)  # least confident first
    return xyz0.index_select(0, order[:c]).detach()


def solve_nested_killing_sequence(
    *,
    canonical_xyz: torch.Tensor,
    canonical_rotation: torch.Tensor,
    part_prob: torch.Tensor,
    frame_times: torch.Tensor,
    cameras_by_frame: list[list[object]],
    gaussians,
    pipe,
    background: torch.Tensor,
    dataset,
    args,
    output_dir: Path,
    log_fn=None,
    outer_iter: int = 0,
    neighbor_graph: NeighborGraph | None = None,
    hard_label: torch.Tensor | None = None,
    warm_start_field: list[torch.Tensor] | None = None,
) -> NestedKillingMotionState:
    xyz0 = torch.as_tensor(canonical_xyz).detach()
    rot0 = normalize_quaternion(torch.as_tensor(canonical_rotation, dtype=xyz0.dtype, device=xyz0.device).detach())
    prob = torch.as_tensor(part_prob, dtype=xyz0.dtype, device=xyz0.device).detach()
    times = torch.as_tensor(frame_times, dtype=xyz0.dtype, device=xyz0.device).reshape(-1)
    moving_parts = int(prob.shape[1]) - 1
    n_gauss = int(xyz0.shape[0])
    if moving_parts <= 0:
        raise ValueError("nested Killing solver needs at least one moving part")

    # FINAL spec §2: z_i = argmax_k p_ik, frozen for the ENTIRE inner solve.
    if hard_label is None:
        hard_label = prob.argmax(dim=-1)
    z = torch.as_tensor(hard_label, device=xyz0.device).reshape(-1).long().detach()
    if int(z.numel()) != n_gauss:
        raise ValueError("hard_label must have one entry per Gaussian")
    prob_moving = prob[:, 1:]
    if neighbor_graph is None:
        neighbor_graph = build_neighbor_graph(
            xyz0,
            k=int(getattr(args, "killing_neighbor_k", 16)),
            chunk_size=int(getattr(args, "killing_neighbor_chunk", 4096)),
        )
    graph = neighbor_graph.to(device=xyz0.device)
    edge_w = edge_part_weights(graph, prob_moving, z)

    # FIRM v4 §11/§12: ONE cumulative transform per moving part. Every Gaussian
    # evaluated under candidate k uses the SAME T_k -- there is no per-Gaussian
    # rigid increment. Static is exactly identity (§12).
    eye = torch.eye(3, dtype=xyz0.dtype, device=xyz0.device)
    part_r = eye[None].expand(moving_parts, -1, -1).clone()
    part_t = torch.zeros((moving_parts, 3), dtype=xyz0.dtype, device=xyz0.device)
    part_q = torch.zeros((moving_parts, 4), dtype=xyz0.dtype, device=xyz0.device)
    part_q[:, 0] = 1.0

    # §6: a_ik = p_ik * u_i. This repository's motion optimizer carries no
    # per-Gaussian visibility/reliability weight, so u_i is uniform; the hook is
    # kept so a real one can be supplied without touching the twist algebra.
    update_weight = torch.ones(n_gauss, dtype=xyz0.dtype, device=xyz0.device)

    moving_xyz, moving_quat = _partwise_candidate_state(xyz0, rot0, part_r, part_t, part_q)
    soft_xyz, soft_q = _compose_hard_partition_scene(xyz0, rot0, moving_xyz, moving_quat, z)
    position_frames = [soft_xyz.detach()]
    rotation_frames = [soft_q.detach()]
    omega_rows, b_rows, u_rows, strain_rows, history = [], [], [], [], []
    omega_history: list[torch.Tensor] = []
    kinetic_accum = torch.zeros((n_gauss, moving_parts), dtype=xyz0.dtype, device=xyz0.device)
    speed_accum = torch.zeros(n_gauss, dtype=xyz0.dtype, device=xyz0.device)
    kinetic_frames = 0
    u_history: list[torch.Tensor] = []
    part_r_frames, part_t_frames, part_q_frames = [], [], []
    field_debug: list[dict] = []
    prev_velocity = None
    prev_prev_velocity = None
    prev_dt = None
    dt = torch.diff(times).clamp_min(1.0e-8)
    inner_steps = max(0, int(getattr(args, "killing_inner_steps_per_frame", 10)))
    scale = torch.linalg.norm(xyz0.max(dim=0).values - xyz0.min(dim=0).values).clamp_min(1.0e-8)

    jacobian_mode = str(getattr(args, "killing_jacobian_mode", "analytic"))
    c_min = float(getattr(args, "observed_velocity_c_min", 0.60))
    gamma = float(getattr(args, "observed_velocity_gamma", 2.0))
    rho_obs_max = float(getattr(args, "observed_velocity_rho_max", 0.20))
    base_lambda_obs = float(getattr(args, "lambda_inner_observed_velocity", 0.0))
    min_high_conf = int(getattr(args, "observed_velocity_min_high_conf", 16))
    camera_chunk = int(getattr(args, "killing_camera_chunk", 0))
    lambda_view_var = float(getattr(args, "lambda_multiview_view_var", 0.05))
    # §12: only the moving support's visibility can make a view invalid.
    moving_mask = (z > 0)
    camera_rows: list[dict[str, float]] = []
    debug_every = int(getattr(args, "multiview_debug_every", 0))
    grad_probe_every = int(getattr(args, "multiview_grad_probe_every", 0))
    debug_root = output_dir / "multiview_debug"
    diag_points = _diagnostic_points(xyz0, prob, int(getattr(args, "killing_field_diagnostic_points", 512)))

    observed_cache = None
    if base_lambda_obs > 0.0:
        observed_cache = MotionCorrespondenceCache(dataset, args)
        if not observed_cache.enabled:
            print("[Nested Killing] observed-velocity requested but no flow cache is available; term stays zero")

    render_static_total, render_motion_total, strain_total, frames_counted = 0.0, 0.0, 0.0, 0

    frame_progress = tqdm(
        range(max(0, int(times.numel()) - 1)),
        desc=f"Nested inner Killing outer={outer_iter}",
        unit="frame",
        leave=False,
    )
    for interval in frame_progress:
        centers = torch.stack(
            [
                _select_control_centers(
                    moving_xyz[part],
                    prob_moving[:, part] * (z == part + 1).to(prob_moving),
                    int(getattr(args, "nested_killing_control_points", 64)),
                )
                for part in range(moving_parts)
            ],
            dim=0,
        )
        sigma = torch.stack(
            [_kernel_scale(centers[part], float(getattr(args, "nested_killing_kernel_scale", 0.0))) for part in range(moving_parts)]
        ).to(device=xyz0.device, dtype=xyz0.dtype)
        # §18: L_field-sep uses squared field differences, so its gradient
        # vanishes identically at EXACT collapse. Zero-initialising every part
        # would start each interval on that saddle and the separation term could
        # never act. Break the symmetry with a small per-part perturbation
        # (K=1 has nothing to separate, so it keeps the clean zero start).
        control_init = torch.zeros_like(centers)
        if moving_parts > 1:
            jitter = float(getattr(args, "killing_field_symmetry_break", 1.0e-4))
            if jitter > 0.0:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(outer_iter) * 1000003 + int(interval))
                noise = torch.randn(control_init.shape, generator=generator).to(control_init)
                control_init = control_init + jitter * float(scale) * noise
        # §47: warm-start from the previous outer's field where the shape still
        # matches, so the M-step does not restart from zero every outer.
        if warm_start_field is not None and interval < len(warm_start_field):
            previous = warm_start_field[interval].to(control_init)
            if previous.shape == control_init.shape:
                control_init = previous.clone()
        control_velocity = torch.nn.Parameter(control_init)
        optimizer = torch.optim.Adam([control_velocity], lr=float(getattr(args, "killing_lr", 1.0e-2)))

        cameras_next = cameras_by_frame[interval + 1] if interval + 1 < len(cameras_by_frame) else []
        observed_state = None
        if observed_cache is not None and observed_cache.enabled and interval + 1 < len(cameras_by_frame):
            observed_state = observed_cache.observed_velocity_targets(
                soft_xyz.detach(),
                cameras_by_frame[interval],
                cameras_by_frame[interval + 1],
                float(times[interval].detach().cpu()),
                float(times[interval + 1].detach().cpu()),
                gaussians=gaussians,
                pipe=pipe,
                query_rotations=soft_q.detach(),
            )

        base_xyz = moving_xyz.detach()
        base_quat = moving_quat.detach()
        base_r = part_r.detach()
        base_t = part_t.detach()
        base_q = part_q.detach()
        last = None
        step_progress = tqdm(
            range(max(1, inner_steps)),
            desc=f"Nested inner outer={outer_iter} frame={interval + 1}",
            unit="step",
            leave=False,
        )
        for step in step_progress:
            optimizer.zero_grad(set_to_none=True)

            # The trainable quantity is the Gaussian-by-Gaussian velocity field
            # (§1); its Jacobian supplies LOCAL strain rigidity (§4).
            q = local_killing_quantities(
                base_xyz, centers, control_velocity, sigma, prob_moving,
                graph, edge_w, jacobian_mode=jacobian_mode,
            )
            # §6-§8: ONE instantaneous twist per part, fitted analytically from
            # ALL weighted Gaussian velocities of that part. Differentiable (§13).
            twist = analytic_partwise_twist(
                base_xyz, q["velocity"], prob_moving, update_weight, z,
                reg=float(getattr(args, "partwise_twist_reg", 1.0e-8)),
            )
            # §11/§12: ONE exact SE(3) increment per part, shared by every
            # Gaussian evaluated under that part.
            r_inc, p_inc, q_inc = se3_exp_spatial(twist["omega"], twist["b"], dt[interval])
            trial_r = r_inc @ base_r
            trial_t = (r_inc @ base_t[..., None])[..., 0] + p_inc
            trial_q = normalize_quaternion(quaternion_multiply(q_inc, base_q))
            trial_xyz, trial_quat = _partwise_candidate_state(xyz0, rot0, trial_r, trial_t, trial_q)
            render_xyz, render_quat = _compose_hard_partition_scene(xyz0, rot0, trial_xyz, trial_quat, z)

            # §5: gradient accumulation only when explicitly requested for memory.
            accumulate_fn = None
            if camera_chunk > 0:
                def accumulate_fn(contribution):
                    contribution.backward(retain_graph=True)

            # ---- ALL CAMERAS BEFORE THE STEP (§2, §3, §13, §14) --------------
            mv = multiview_render_loss_for_frame(
                cameras=cameras_next,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                dataset=dataset,
                args=args,
                positions=render_xyz,
                rotations=render_quat,
                moving_mask=moving_mask,
                lambda_view_var=lambda_view_var,
                accumulate=accumulate_fn,
            )
            render_loss = mv["loss"]
            render_value = mv["value"]
            render_stats = {"rgb": mv["rgb"], "ssim": mv["ssim"], "depth": mv["depth"], "views": mv["num_views"]}
            # §18: every inner iteration must see the full camera set.
            if cameras_next and int(mv["num_views"]) != len(cameras_next):
                raise RuntimeError(
                    f"multiview contract violated: rendered {mv['num_views']} of {len(cameras_next)} cameras"
                )
            # §20: detailed per-camera rows go to their own CSV, not into the
            # already-wide inner history.
            for cam in mv["per_camera"]:
                camera_rows.append({
                    "outer": int(outer_iter), "frame": int(interval + 1), "inner_step": int(step),
                    **cam,
                })

            # ---- physics terms (§5, §8, §17, §18) ----------------------------
            strain_loss = q["strain_loss"]
            omega_coh = q["omega_coherence"]
            b_coh = q["b_coherence"]
            velocity = q["velocity"]
            if prev_velocity is None:
                temporal = velocity.sum() * 0.0
            else:
                # §17: temporal regularity of the LOCAL field, per-part normalised.
                d = (velocity - prev_velocity).square().sum(-1)
                w = prob_moving.transpose(0, 1)
                temporal = ((w * d).sum(dim=1) / w.sum(dim=1).clamp_min(1.0e-8)).sum()
            if prev_velocity is None or prev_prev_velocity is None:
                accel = velocity.sum() * 0.0
            else:
                d = (velocity - 2.0 * prev_velocity + prev_prev_velocity).square().sum(-1)
                w = prob_moving.transpose(0, 1)
                accel = ((w * d).sum(dim=1) / w.sum(dim=1).clamp_min(1.0e-8)).sum()
            # §16: separation acts on the PARTWISE twists, over the trajectory.
            omega_traj = torch.stack(omega_history + [twist["omega"]]) if omega_history else twist["omega"][None]
            u_traj = torch.stack(u_history + [twist["center_velocity"]]) if u_history else twist["center_velocity"][None]
            sep_loss, sep_matrix = partwise_motion_separation(
                omega_traj, u_traj,
                margin=float(getattr(args, "nested_motion_separation_margin", 0.01)),
                alpha_omega=float(getattr(args, "field_separation_alpha_omega", 1.0)),
                alpha_u=float(getattr(args, "field_separation_alpha_v", 1.0)),
            )

            # §9: does the velocity field of part k agree with ONE rigid twist?
            rigid_speed = twist["rigid_speed_residual"]

            w_strain = float(getattr(args, "lambda_inner_strain", 10.0))
            w_rigid = float(getattr(args, "lambda_inner_rigid_speed", 1.0))
            w_omega = float(getattr(args, "lambda_inner_omega_consistency", 1.0))
            w_b = float(getattr(args, "lambda_inner_b_consistency", 1.0))
            w_temp = float(getattr(args, "lambda_inner_twist_temporal", 0.01))
            w_acc = float(getattr(args, "lambda_inner_twist_acceleration", 0.0))
            w_sep = float(getattr(args, "lambda_inner_field_separation", 0.0))

            # §10: local strain and partwise rigid-speed have different roles and
            # both are required; neither replaces the other.
            physics = (
                w_strain * strain_loss
                + w_rigid * rigid_speed
                + w_omega * omega_coh
                + w_b * b_coh
                + w_temp * temporal
                + w_acc * accel
                + w_sep * sep_loss
            )

            # ---- §12/§13: confidence-gated, capped observed velocity ---------
            # §12: candidate motion is partwise, so the predicted per-Gaussian
            # velocity is the partwise displacement over the real interval.
            predicted_velocity = (trial_xyz - base_xyz) / dt[interval].clamp_min(1.0e-8)
            obs_loss, obs_stats = observed_velocity_loss(
                predicted_velocity, observed_state, prob_moving,
                velocity_scale=float(getattr(args, "eulerian_velocity_scale", 1.0)),
                c_min=c_min, gamma=gamma,
                huber_delta=float(getattr(args, "observed_velocity_huber_delta", 1.0)),
                min_high_conf=min_high_conf,
            )
            primary = physics.detach() + torch.as_tensor(
                render_value, dtype=physics.dtype, device=physics.device
            )
            lambda_obs_eff = effective_obs_lambda(
                base_lambda_obs, primary, obs_loss, rho_max=rho_obs_max
            )

            # Separate the render gradient from the physics/track gradient so a
            # dead or negligible render path is visible instead of being inferred
            # from a flat loss curve.
            # §16: per-camera gradient influence, on a diagnostic interval only.
            # Must run while the upstream graph is alive, i.e. before any
            # backward frees it; the probe itself keeps the graph (retain_graph).
            cam_grad_norms: list[float] = []
            if grad_probe_every > 0 and cameras_next and (step % grad_probe_every == 0):
                cam_grad_norms = per_camera_gradient_norms(
                    cameras=cameras_next, gaussians=gaussians, pipe=pipe, background=background,
                    dataset=dataset, args=args, positions=render_xyz, rotations=render_quat,
                    parameter=control_velocity, moving_mask=moving_mask,
                )

            # §7: physics is added ONCE, never once per camera.
            physics_total = physics + lambda_obs_eff * obs_loss
            total_loss = render_loss + physics_total

            # Separate the render gradient from the physics/track gradient.
            if accumulate_fn is None:
                render_loss.backward(retain_graph=True)
            grad_render = (
                control_velocity.grad.detach().clone()
                if control_velocity.grad is not None
                else torch.zeros_like(control_velocity)
            )
            physics_total.backward()
            grad_total = (
                control_velocity.grad.detach().clone()
                if control_velocity.grad is not None
                else torch.zeros_like(control_velocity)
            )
            grad_tail = grad_total - grad_render
            grad_render_norm = float(grad_render.norm())
            grad_tail_norm = float(grad_tail.norm())
            # §3/§24: EXACTLY ONE optimizer step after all cameras contributed.
            optimizer.step()
            optimizer_steps_this_inner_iteration = 1

            # §19: GT / render / residual per camera at selected iterations.
            # Selected frames only, at the END of their inner solve, so the
            # images show the motion that was actually committed.
            if (
                debug_every > 0
                and cameras_next
                and (interval % debug_every == 0)
                and (step == max(1, inner_steps) - 1)
            ):
                save_multiview_debug_images(
                    cameras=cameras_next, gaussians=gaussians, pipe=pipe, background=background,
                    dataset=dataset, args=args, positions=render_xyz.detach(), rotations=render_quat.detach(),
                    out_dir=debug_root / f"outer_{outer_iter:03d}" / f"frame_{interval + 1:03d}" / f"inner_{step:03d}",
                )

            total_value = render_value + float(physics_total.detach())
            obs_weighted = float(lambda_obs_eff.detach()) * float(obs_loss.detach())
            S = q["strain"].detach()
            last = {
                "outer": int(outer_iter),
                "frame": int(interval + 1),
                "step": int(step),
                "inner_total": total_value,
                "loss_total": total_value,
                "loss_render": render_value,
                "loss_strain": float(strain_loss.detach()),
                "loss_rigid_speed": float(rigid_speed.detach()),
                # Which term is actually driving the field?
                "grad_render_norm": grad_render_norm,
                "grad_physics_obs_norm": grad_tail_norm,
                "grad_render_fraction": grad_render_norm / max(grad_render_norm + grad_tail_norm, 1.0e-30),
                "grad_norm_render_multiview": grad_render_norm,
                "grad_norm_physics": grad_tail_norm,
                **{f"grad_norm_render_cam_{i:03d}": v for i, v in enumerate(cam_grad_norms)},
                "loss_omega_local_coherence": float(omega_coh.detach()),
                "loss_b_local_coherence": float(b_coh.detach()),
                "loss_temporal": float(temporal.detach()),
                "loss_acceleration": float(accel.detach()),
                "loss_field_separation": float(sep_loss.detach()),
                "loss_observed_velocity": float(obs_loss.detach()),
                "observed_velocity_high_conf": float(obs_loss.detach()),
                "render_rgb": render_stats["rgb"],
                "render_ssim": render_stats["ssim"],
                "render_depth": render_stats["depth"],
                # §18: the full camera set, and exactly one optimizer step.
                "render_views": float(render_stats["views"]),
                "optimizer_steps_this_inner_iteration": float(optimizer_steps_this_inner_iteration),
                # §9 aggregate multiview diagnostics
                "render_multiview_mean": mv["mean"],
                "render_multiview_std": mv["std"],
                "render_multiview_min": mv["min"],
                "render_multiview_max": mv["max"],
                "render_multiview_num_cameras": float(mv["num_views"]),
                "render_multiview_std_in_loss": float(bool(mv["std_in_loss"])),
                # §20 aggregate per-channel spread across views
                "render_rgb_mean": mv["rgb"],
                "render_rgb_std": float(torch.tensor([c["rgb"] for c in mv["per_camera"]]).std(unbiased=False)) if mv["per_camera"] else 0.0,
                "render_rgb_min": min((c["rgb"] for c in mv["per_camera"]), default=0.0),
                "render_rgb_max": max((c["rgb"] for c in mv["per_camera"]), default=0.0),
                "render_ssim_mean": mv["ssim"],
                "render_ssim_std": float(torch.tensor([c["ssim"] for c in mv["per_camera"]]).std(unbiased=False)) if mv["per_camera"] else 0.0,
                "render_ssim_min": min((c["ssim"] for c in mv["per_camera"]), default=0.0),
                "render_ssim_max": max((c["ssim"] for c in mv["per_camera"]), default=0.0),
                # §8 per-camera values
                **{f"render_rgb_cam_{c['camera_id']:03d}": c["rgb"] for c in mv["per_camera"]},
                **{f"render_ssim_cam_{c['camera_id']:03d}": c["ssim"] for c in mv["per_camera"]},
                **{f"render_visibility_cam_{c['camera_id']:03d}": c["visibility"] for c in mv["per_camera"]},
                # §36 confidence diagnostics
                "observed_velocity_num_valid": float(obs_stats["num_valid"]),
                "observed_velocity_coverage_fraction": float(obs_stats["coverage_fraction"]),
                "observed_velocity_mean_confidence": float(obs_stats["mean_confidence"]),
                "observed_velocity_median_confidence": float(obs_stats["median_confidence"]),
                "observed_velocity_num_above_cmin": float(obs_stats["num_above_cmin"]),
                "observed_velocity_high_conf_coverage": float(obs_stats["high_conf_coverage"]),
                "observed_velocity_base_lambda": float(base_lambda_obs),
                "observed_velocity_effective_lambda": float(lambda_obs_eff.detach()),
                "observed_velocity_weighted_fraction_of_inner_loss": (
                    obs_weighted / max(abs(total_value), 1.0e-12)
                ),
                # §37 field diagnostics
                "strain_norm": float(torch.linalg.matrix_norm(S).mean().cpu()),
                "strain_xx": float(S[..., 0, 0].abs().mean().cpu()),
                "strain_yy": float(S[..., 1, 1].abs().mean().cpu()),
                "strain_zz": float(S[..., 2, 2].abs().mean().cpu()),
                "strain_xy": float(S[..., 0, 1].abs().mean().cpu()),
                "strain_xz": float(S[..., 0, 2].abs().mean().cpu()),
                "strain_yz": float(S[..., 1, 2].abs().mean().cpu()),
                "omega_norm": float(q["omega_local"].detach().norm(dim=-1).mean().cpu()),
                "b_norm": float(q["b_local"].detach().norm(dim=-1).mean().cpu()),
                "u_norm": float(q["center_velocity"].detach().norm(dim=-1).mean().cpu()),
                "velocity_norm": float(velocity.detach().norm(dim=-1).mean().cpu()),
                "field_control_points": float(centers.shape[1]),
                "field_kernel_scale_mean": float(sigma.detach().mean().cpu()),
                # §21 degenerate-support diagnostics for the analytic twist fit.
                "twist_effective_weight": float(twist["effective_weight"].mean().cpu()),
                "twist_effective_num_points": float(twist["effective_num_points"].mean().cpu()),
                "twist_condition_number": float(twist["condition_number"].max().cpu()),
                "twist_eigenvalue_min": float(twist["eigenvalues"].min().cpu()),
                "twist_eigenvalue_max": float(twist["eigenvalues"].max().cpu()),
                "part_omega_norm": float(twist["omega"].detach().norm(dim=-1).mean().cpu()),
                "part_b_norm": float(twist["b"].detach().norm(dim=-1).mean().cpu()),
                "part_center_velocity_norm": float(twist["center_velocity"].detach().norm(dim=-1).mean().cpu()),
                **{
                    f"part_omega_norm_part_{p + 1}": float(twist["omega"].detach()[p].norm().cpu())
                    for p in range(moving_parts)
                },
                **{
                    f"twist_condition_number_part_{p + 1}": float(twist["condition_number"][p].cpu())
                    for p in range(moving_parts)
                },
                **{
                    f"omega_norm_part_{p + 1}": float(q["omega_local"].detach()[p].norm(dim=-1).mean().cpu())
                    for p in range(moving_parts)
                },
                **{
                    f"b_norm_part_{p + 1}": float(q["b_local"].detach()[p].norm(dim=-1).mean().cpu())
                    for p in range(moving_parts)
                },
                **{
                    f"strain_norm_part_{p + 1}": float(torch.linalg.matrix_norm(S[p]).mean().cpu())
                    for p in range(moving_parts)
                },
            }
            history.append(last)
            if log_fn is not None:
                log_fn(
                    "nested_inner",
                    outer_iter * 100000 + interval * max(1, inner_steps) + step,
                    {f"nested_inner/{k}": v for k, v in last.items() if isinstance(v, (int, float))},
                )
            step_progress.set_postfix(
                L=f"{total_value:.3g}",
                rgb=f"{render_stats['rgb']:.3g}",
                D=f"{float(strain_loss.detach()):.2e}",
                obs=int(obs_stats["num_above_cmin"]),
                lam=f"{float(lambda_obs_eff.detach()):.2e}",
                w=f"{last['omega_norm']:.2e}",
            )

        if last is not None:
            frame_progress.set_postfix(
                frame=int(interval + 1),
                L=f"{last['inner_total']:.3g}",
                rgb=f"{last['render_rgb']:.3g}",
                D=f"{last['loss_strain']:.2e}",
                hi=int(last["observed_velocity_num_above_cmin"]),
            )

        # ---- commit the interval -------------------------------------------
        with torch.no_grad():
            q = local_killing_quantities(
                base_xyz, centers, control_velocity, sigma, prob_moving,
                graph, edge_w, jacobian_mode=jacobian_mode,
            )
            twist = analytic_partwise_twist(
                base_xyz, q["velocity"], prob_moving, update_weight, z,
                reg=float(getattr(args, "partwise_twist_reg", 1.0e-8)),
            )
            # §11/§12: one increment per part, composed as T_k(t+1) = dT_k T_k(t).
            r_inc, p_inc, q_inc = se3_exp_spatial(twist["omega"], twist["b"], dt[interval])
            part_r = r_inc @ part_r
            part_t = (r_inc @ part_t[..., None])[..., 0] + p_inc
            part_q = normalize_quaternion(quaternion_multiply(q_inc, part_q))
            moving_xyz, moving_quat = _partwise_candidate_state(xyz0, rot0, part_r, part_t, part_q)
            soft_xyz, soft_q = _compose_hard_partition_scene(xyz0, rot0, moving_xyz, moving_quat, z)
            position_frames.append(soft_xyz.detach().clone())
            rotation_frames.append(soft_q.detach().clone())
            part_r_frames.append(part_r.detach().clone())
            part_t_frames.append(part_t.detach().clone())
            part_q_frames.append(part_q.detach().clone())
            omega_history.append(twist["omega"].detach().clone())
            u_history.append(twist["center_velocity"].detach().clone())

            # §18: E_ik^kin = aD||D_ik||^2 + av||v_i - v_ik^rigid||^2 + aw||w_ik^local - w_k||^2
            strain_term = q["strain"].square().sum(dim=(-2, -1))                       # [K,N]
            rigid_term = (q["velocity"] - twist["rigid_velocity"]).square().sum(-1)    # [K,N]
            omega_term = (q["omega_local"] - twist["omega"][:, None, :]).square().sum(-1)
            kinetic_accum += (
                float(getattr(args, "kinetic_alpha_strain", 1.0)) * strain_term
                + float(getattr(args, "kinetic_alpha_rigid", 1.0)) * rigid_term
                + float(getattr(args, "kinetic_alpha_omega", 1.0)) * omega_term
            ).transpose(0, 1)
            # §19: per-Gaussian inferred speed under its own current class.
            speed_accum += q["velocity"].norm(dim=-1).mean(dim=0)
            kinetic_frames += 1

            zeros3 = torch.zeros((prob.shape[1], 3), dtype=xyz0.dtype, device=xyz0.device)
            omega_full, b_full, u_full = zeros3.clone(), zeros3.clone(), zeros3.clone()
            strain_full = torch.zeros((prob.shape[1], 3, 3), dtype=xyz0.dtype, device=xyz0.device)
            omega_full[1:] = twist["omega"]
            b_full[1:] = twist["b"]
            u_full[1:] = twist["center_velocity"]
            w = prob_moving.transpose(0, 1)
            strain_full[1:] = (q["strain"] * w[:, :, None, None]).sum(dim=1) / w.sum(dim=1).clamp_min(1.0e-8)[:, None, None]
            omega_rows.append(omega_full.clone())
            b_rows.append(b_full.clone())
            u_rows.append(u_full.clone())
            strain_rows.append(strain_full.clone())

            # §19 acceptance evidence: render with motion vs render held static.
            if cameras_next:
                _, motion_stats = _render_all_cameras(
                    cameras_next, gaussians, pipe, background, dataset, args, soft_xyz, soft_q
                )
                _, static_stats = _render_all_cameras(
                    cameras_next, gaussians, pipe, background, dataset, args, xyz0, rot0
                )
                render_motion_total += float(motion_stats["rgb"])
                render_static_total += float(static_stats["rgb"])
                frames_counted += 1
            strain_total += float(torch.linalg.matrix_norm(q["strain"]).mean())

            field_debug.append(
                {
                    "outer": int(outer_iter),
                    "interval": int(interval),
                    "frame": int(interval + 1),
                    "dt": float(dt[interval].cpu()),
                    "centers": centers.detach().cpu(),
                    "control_velocity": control_velocity.detach().cpu(),
                    "kernel_scale": sigma.detach().cpu(),
                    "omega_local_mean": q["omega_mean"].cpu(),
                    "b_local_mean": q["b_mean"].cpu(),
                    "part_omega": twist["omega"].cpu(),
                    "part_b": twist["b"].cpu(),
                    "part_center": twist["center"].cpu(),
                    "center_velocity": twist["center_velocity"].cpu(),
                    "rigid_speed_residual": float(twist["rigid_speed_residual"].cpu()),
                    "twist_eigenvalues": twist["eigenvalues"].cpu(),
                    "twist_condition_number": twist["condition_number"].cpu(),
                    "twist_effective_weight": twist["effective_weight"].cpu(),
                    "twist_effective_num_points": twist["effective_num_points"].cpu(),
                    "strain_mean": float(q["strain"].square().sum(dim=(-2, -1)).sqrt().mean().cpu()),
                    "omega_local_coherence": float(q["omega_coherence"].cpu()),
                    "b_local_coherence": float(q["b_coherence"].cpu()),
                    "field_separation": sep_matrix.detach().cpu(),
                }
            )
            prev_prev_velocity = None if prev_velocity is None else prev_velocity.clone()
            prev_velocity = q["velocity"].detach().clone()
            prev_dt = dt[interval].clone()

    output_dir.mkdir(parents=True, exist_ok=True)

    # FIRM v4 §11/§18: these ARE the motion. One cumulative transform per part,
    # obtained by integrating the analytic partwise twists -- not Kabsch, and not
    # composed from per-Gaussian increments.
    part_rotation, part_translation, part_quaternion = _stack_part_transforms(
        part_r_frames, part_t_frames, part_q_frames, prob, xyz0
    )
    motion_separation = _partwise_separation_matrix(omega_history, u_history, prob)

    accept_render_motion = render_motion_total / max(1, frames_counted)
    accept_render_static = render_static_total / max(1, frames_counted)
    acceptance = {
        "render_rgb_motion": accept_render_motion,
        "render_rgb_static": accept_render_static,
        "render_improvement": accept_render_static - accept_render_motion,
        "mean_strain_norm": strain_total / max(1, int(times.numel()) - 1),
        "frames": float(frames_counted),
    }

    _write_csv(output_dir / "inner_history.csv", history)
    _write_csv(output_dir / "inner_camera_history.csv", camera_rows)
    (output_dir / "inner_history.json").write_text(json.dumps(_jsonable(history), indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "motion_separation.json").write_text(json.dumps(_jsonable(motion_separation), indent=2, allow_nan=False), encoding="utf-8")
    torch.save(
        {
            "schema_version": 3,
            "field_type": "local_gaussian_differential_killing_field",
            "field_debug": field_debug,
            "motion_separation": motion_separation,
            "acceptance": acceptance,
            "kabsch_used": False,
            "per_gaussian_twist_optimized": False,
            "global_hard_twist_imposed": False,
        },
        output_dir / "local_killing_field_debug.pt",
    )
    return NestedKillingMotionState(
        times=times.detach().cpu(),
        positions=torch.stack(position_frames).detach().cpu(),
        rotations=torch.stack(rotation_frames).detach().cpu(),
        part_rotation=part_rotation,
        part_translation=part_translation,
        part_quaternion=part_quaternion,
        omega=torch.stack(omega_rows).detach().cpu() if omega_rows else torch.zeros((0, prob.shape[1], 3)),
        b=torch.stack(b_rows).detach().cpu() if b_rows else torch.zeros((0, prob.shape[1], 3)),
        center_velocity=torch.stack(u_rows).detach().cpu() if u_rows else torch.zeros((0, prob.shape[1], 3)),
        strain=torch.stack(strain_rows).detach().cpu() if strain_rows else torch.zeros((0, prob.shape[1], 3, 3)),
        motion_separation=motion_separation,
        history=history,
        field_debug=field_debug,
        moving_endpoints=moving_xyz.detach(),
        moving_endpoint_quaternions=moving_quat.detach(),
        acceptance=acceptance,
        kinetic_energy=(kinetic_accum / max(1, kinetic_frames)).detach(),
        inferred_speed=(speed_accum / max(1, kinetic_frames)).detach(),
    )


def _stack_part_transforms(part_r_frames, part_t_frames, part_q_frames, prob, xyz0):
    """Stack the per-frame PARTWISE cumulative transforms, static class first.

    FIRM v4 §11/§18: these are the integrated analytic partwise twists. They are
    the motion itself -- not a Kabsch fit, and not a summary of per-Gaussian
    increments.
    """
    k_plus_1 = int(prob.shape[1])
    eye = torch.eye(3, dtype=xyz0.dtype, device=xyz0.device)
    identity_q = torch.zeros(4, dtype=xyz0.dtype, device=xyz0.device)
    identity_q[0] = 1.0
    if not part_r_frames:
        return (
            eye[None, None].expand(1, k_plus_1, -1, -1).detach().cpu(),
            torch.zeros((1, k_plus_1, 3), dtype=xyz0.dtype).cpu(),
            identity_q[None, None].expand(1, k_plus_1, -1).detach().cpu(),
        )
    # Frame 0 is the identity state. Without it these tensors carry one entry per
    # INTERVAL while positions/rotations/times carry one per FRAME, so indexing
    # part_rotation[f] alongside positions[f] would be shifted by a frame.
    rot = [eye[None].expand(k_plus_1, -1, -1).clone()]
    trans = [torch.zeros((k_plus_1, 3), dtype=xyz0.dtype, device=xyz0.device)]
    quat = [identity_q[None].expand(k_plus_1, -1).clone()]
    for r, t, q in zip(part_r_frames, part_t_frames, part_q_frames):
        full_r = eye[None].expand(k_plus_1, -1, -1).clone()
        full_t = torch.zeros((k_plus_1, 3), dtype=xyz0.dtype, device=xyz0.device)
        full_q = identity_q[None].expand(k_plus_1, -1).clone()
        full_r[1:] = r
        full_t[1:] = t
        full_q[1:] = q
        rot.append(full_r)
        trans.append(full_t)
        quat.append(full_q)
    return (
        torch.stack(rot).detach().cpu(),
        torch.stack(trans).detach().cpu(),
        torch.stack(quat).detach().cpu(),
    )


def _partwise_separation_matrix(omega_history, u_history, prob):
    """FIRM v4 §16: sequence-mean pairwise distance between PARTWISE twists."""
    k = int(prob.shape[1])
    out = torch.zeros((k, k), dtype=prob.dtype, device=prob.device)
    if not omega_history:
        return out.detach().cpu()
    w = torch.stack(omega_history)          # [F,K,3]
    u = torch.stack(u_history)
    moving = int(w.shape[1])
    for a in range(moving):
        for b in range(a + 1, moving):
            d = ((w[:, a] - w[:, b]).square().sum(-1) + (u[:, a] - u[:, b]).square().sum(-1)).mean()
            out[a + 1, b + 1] = d
            out[b + 1, a + 1] = d
    return out.detach().cpu()
