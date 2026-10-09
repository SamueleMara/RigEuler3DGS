"""Render-centric outer E-step (CODEX render-centric spec).

Inner loop: labels fixed, rendering optimizes motion.
Outer loop: motion frozen, rendering optimizes LABELS.

The forward physical scene stays hard -- every Gaussian follows exactly one
candidate motion (§3, §4) -- while a straight-through surrogate lets the video
residual reach the membership logits (§22). Motion, appearance and cameras are
all frozen here (§1, §23).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .neighborhood import NeighborGraph


# ---------------------------------------------------------------------------
# §3 straight-through membership
# ---------------------------------------------------------------------------
def straight_through_membership(p_soft: torch.Tensor) -> torch.Tensor:
    """p_st = onehot(argmax p) - stopgrad(p) + p.

    Forward: exactly one-hot, so the rendered scene is physically discrete.
    Backward: d p_st / d logits = d p_soft / d logits.
    """
    hard_idx = p_soft.argmax(dim=-1)
    p_hard = F.one_hot(hard_idx, num_classes=int(p_soft.shape[-1])).to(p_soft.dtype)
    return p_hard + p_soft - p_soft.detach()


def align_quaternion_hemisphere(candidate_quat: torch.Tensor) -> torch.Tensor:
    """§5: remove the q == -q sign ambiguity before any ST combination.

    ``candidate_quat`` is [N, K+1, 4]; class 0 (static) is the detached reference.
    """
    q_ref = candidate_quat[:, 0].detach()
    dot = (candidate_quat * q_ref[:, None, :]).sum(dim=-1, keepdim=True)
    return torch.where(dot < 0, -candidate_quat, candidate_quat)


def compose_straight_through_scene(
    candidate_xyz: torch.Tensor,
    candidate_quat: torch.Tensor,
    p_st: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """§4/§5: hard forward state, differentiable label backward.

    ``candidate_xyz``/``candidate_quat`` are [N, K+1, ...] and DETACHED (motion
    is frozen); ``p_st`` is the straight-through membership [N, K+1].
    """
    xyz = (p_st[..., None] * candidate_xyz).sum(dim=1)
    quat = (p_st[..., None] * align_quaternion_hemisphere(candidate_quat)).sum(dim=1)
    return xyz, F.normalize(quat, dim=-1)


def candidate_state_at_frame(
    xyz0: torch.Tensor,
    rot0: torch.Tensor,
    part_rotation: torch.Tensor,
    part_translation: torch.Tensor,
    part_quaternion: torch.Tensor,
    frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """§2/§28: lazily build the K+1 candidate states for one frame.

    Everything here is detached (frozen motion), so no motion graph is retained
    and the [F, K+1, N, 3] tensor is never materialised.
    """
    from utils.rigid_transform_utils import apply_rotation_to_gaussian_quaternions

    with torch.no_grad():
        rot = part_rotation[frame].to(xyz0)
        trans = part_translation[frame].to(xyz0)
        quat = part_quaternion[frame].to(xyz0)
        k_plus_1 = int(rot.shape[0])
        xyz = torch.stack(
            [xyz0 @ rot[k].transpose(0, 1) + trans[k] for k in range(k_plus_1)], dim=1
        )
        quats = torch.stack(
            [apply_rotation_to_gaussian_quaternions(rot0, quat[k]) for k in range(k_plus_1)],
            dim=1,
        )
    return xyz.detach(), quats.detach()


# ---------------------------------------------------------------------------
# §6-§10 the outer video loss
# ---------------------------------------------------------------------------
def outer_render_membership_loss(
    *,
    xyz0,
    rot0,
    part_rotation,
    part_translation,
    part_quaternion,
    p_st,
    cameras_by_frame,
    gaussians,
    pipe,
    background,
    dataset,
    args,
    frames=None,
    accumulate=None,
    render_fn=None,
    collect_rows=None,
) -> dict:
    """§6/§7/§10: the full multiframe / multiview video loss for the labels.

        L_outer-video = (1 / sum_t |C_t|) * sum_t sum_c L_video^{t,c}

    Every camera of a frame contributes before the label optimizer moves (§7),
    and one epoch covers every frame (§8). ``accumulate`` backwards each camera's
    share immediately so the whole video never needs to be in memory at once.

    This helper never touches motion, appearance or the inner optimizer (§10).
    """
    from .killing_inner_solver import _render_camera_loss

    render_fn = render_fn or _render_camera_loss
    frame_list = list(range(len(cameras_by_frame))) if frames is None else list(frames)
    total_views = sum(len(cameras_by_frame[f]) for f in frame_list)
    if total_views == 0:
        zero = p_st.sum() * 0.0
        return {"loss": zero, "rgb": 0.0, "ssim": 0.0, "depth": 0.0,
                "num_frames": 0, "num_views": 0, "per_frame": [], "per_camera": [], "value": 0.0}

    terms: list[torch.Tensor] = []
    per_frame, per_camera = [], []
    rgb_a = ssim_a = depth_a = 0.0
    depth_n = 0
    accumulated = 0.0

    for frame in frame_list:
        cameras = cameras_by_frame[frame]
        if not cameras:
            continue
        cand_xyz, cand_quat = candidate_state_at_frame(
            xyz0, rot0, part_rotation, part_translation, part_quaternion, frame
        )
        xyz, quat = compose_straight_through_scene(cand_xyz, cand_quat, p_st)

        frame_total = 0.0
        for cam_id, camera in enumerate(cameras):
            loss_cam, rgb, dssim, depth, has_depth, visibility, *_ = render_fn(
                camera, gaussians, pipe, background, dataset, args, xyz, quat
            )
            share = loss_cam / total_views
            if accumulate is not None:
                accumulate(share)
                accumulated += float(share.detach())
            else:
                terms.append(share)
            rgb_a += float(rgb.detach())
            ssim_a += float(dssim.detach())
            if has_depth:
                depth_a += float(depth.detach())
                depth_n += 1
            frame_total += float(loss_cam.detach())
            row = {
                "frame": int(frame), "camera_id": int(cam_id),
                "rgb": float(rgb.detach()), "ssim": float(dssim.detach()),
                "depth": float(depth.detach()) if has_depth else float("nan"),
                "total": float(loss_cam.detach()), "visibility": float(visibility),
            }
            per_camera.append(row)
            if collect_rows is not None:
                collect_rows.append(row)
        per_frame.append({"frame": int(frame), "total": frame_total / max(len(cameras), 1),
                          "num_views": len(cameras)})

    stats = {
        "rgb": rgb_a / total_views,
        "ssim": ssim_a / total_views,
        "depth": depth_a / max(1, depth_n),
        "num_frames": len(per_frame),
        "num_views": total_views,
        "per_frame": per_frame,
        "per_camera": per_camera,
    }
    if accumulate is not None:
        stats["loss"] = p_st.sum() * 0.0
        stats["value"] = accumulated
    else:
        loss = torch.stack(terms).sum()
        stats["loss"] = loss
        stats["value"] = float(loss.detach())
    return stats


# ---------------------------------------------------------------------------
# §17 motion-aware label graph coherence
# ---------------------------------------------------------------------------
def motion_aware_label_graph_loss(
    p_soft: torch.Tensor,
    delta_energy: torch.Tensor,
    graph: NeighborGraph,
    *,
    sigma_e: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """§17: coherence that does NOT smooth across the articulation boundary.

        w_ij^label = w_ij^spatial * exp(-(dE_i - dE_j)^2 / (2 sigma_E^2))
        L = sum w_ij^label ||p_i - p_j||^2 / (sum w_ij^label + eps)

    Nearby with compatible motion evidence -> same label; nearby with
    incompatible evidence -> a boundary is allowed.
    """
    if graph.num_edges == 0:
        return p_soft.sum() * 0.0
    de = torch.as_tensor(delta_energy, dtype=p_soft.dtype, device=p_soft.device).reshape(-1).detach()
    gap = de.index_select(0, graph.source) - de.index_select(0, graph.target)
    s = max(float(sigma_e), 1.0e-8)
    w = graph.weight.to(p_soft) * torch.exp(-gap.square() / (2.0 * s * s))
    diff = (p_soft.index_select(0, graph.source) - p_soft.index_select(0, graph.target)).square().sum(-1)
    return (w * diff).sum() / w.sum().clamp_min(float(eps))


# ---------------------------------------------------------------------------
# §18/§19 inner Killing evidence as weak label evidence
# ---------------------------------------------------------------------------
def kinetic_label_loss(p_soft: torch.Tensor, kinetic_energy: torch.Tensor) -> torch.Tensor:
    """§18: L = (1/N) sum_i sum_{k>0} p_ik E_ik^kin, with E detached."""
    e = torch.as_tensor(kinetic_energy, dtype=p_soft.dtype, device=p_soft.device).detach()
    if e.ndim != 2 or int(e.shape[0]) != int(p_soft.shape[0]):
        raise ValueError("kinetic energy must be [N, K] over the moving classes")
    moving = p_soft[:, 1:]
    k = min(int(moving.shape[1]), int(e.shape[1]))
    return (moving[:, :k] * e[:, :k]).sum(dim=1).mean()


def static_kinetic_energy(
    inferred_speed: torch.Tensor,
    reliability: torch.Tensor | None = None,
    *,
    alpha_0: float = 0.0,
) -> torch.Tensor:
    """§19: E_i0^kin = alpha_0 ||v_i||^2, gated by reliability.

    Defaults to zero: static-vs-moving must stay primarily render-driven, and a
    noisy velocity estimate must never push an invisible Gaussian to moving.
    """
    v = torch.as_tensor(inferred_speed).detach()
    energy = float(alpha_0) * v.square().sum(dim=-1) if v.ndim > 1 else float(alpha_0) * v.square()
    if reliability is not None:
        energy = energy * torch.as_tensor(reliability, dtype=energy.dtype, device=energy.device).detach()
    return energy


# ---------------------------------------------------------------------------
# §16 keep the 3D regularizer gradient subordinate to the video gradient
# ---------------------------------------------------------------------------
def compute_detached_3d_gradient_cap(
    video_loss: torch.Tensor,
    reg_loss: torch.Tensor,
    parameters,
    *,
    rho: float = 0.30,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, float, float]:
    """§16: s_3D = min(1, rho * |g_V| / (|g_3D| + eps)), DETACHED.

    Not a gate -- the outer update always runs. It only stops COM/inertia/NN
    from overwhelming the video-driven label gradient. Returns
    ``(scale, |g_V|, |g_3D|)``.
    """
    params = [p for p in parameters if p.requires_grad]
    if not params:
        return torch.as_tensor(1.0), 0.0, 0.0

    def _norm(loss):
        if not torch.is_tensor(loss) or not loss.requires_grad:
            return 0.0
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        total = 0.0
        for g in grads:
            if g is not None:
                total += float(g.detach().pow(2).sum())
        return float(total ** 0.5)

    g_video = _norm(video_loss)
    g_reg = _norm(reg_loss)
    scale = 1.0 if g_reg <= eps else min(1.0, float(rho) * g_video / (g_reg + eps))
    device = reg_loss.device if torch.is_tensor(reg_loss) else None
    return torch.as_tensor(scale, device=device).detach(), g_video, g_reg


# ---------------------------------------------------------------------------
# §12 optional: do not worsen a camera RELATIVE TO ITSELF
# ---------------------------------------------------------------------------
def view_degradation_penalty(
    current: dict[tuple[int, int], torch.Tensor],
    baseline: dict[tuple[int, int], float],
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """§12: L = (1/FC) sum_{t,c} [ (L_tc - L_tc^base) / (L_tc^base + eps) ]_+.

    Relative, not absolute: cameras keep their own irreducible loss scales, and
    this only discourages making one materially worse than it was at the start of
    the epoch. Optional, small weight, never a gate.
    """
    terms = []
    for key, value in current.items():
        base = float(baseline.get(key, 0.0))
        terms.append(torch.relu((value - base) / (base + float(eps))))
    if not terms:
        return torch.zeros(())
    return torch.stack(terms).mean()


def counterfactual_label_deltas(
    *,
    xyz0,
    rot0,
    part_rotation,
    part_translation,
    part_quaternion,
    hard_label,
    gaussian_ids,
    cameras_by_frame,
    gaussians,
    pipe,
    background,
    dataset,
    args,
    frames,
    render_fn=None,
) -> list[dict]:
    """§35: for boundary Gaussians, measure the ACTUAL render cost of each label.

        dL_{i,k} = L_render(z_i <- k) - L_render(z_i = current)

    Diagnostic only -- never part of the training loss. It validates that the
    straight-through label gradient points the same way asactual counterfactual
    rendering.
    """
    from .killing_inner_solver import _render_camera_loss

    render_fn = render_fn or _render_camera_loss
    k_plus_1 = int(part_rotation.shape[1])
    records: list[dict] = []

    def _loss_for(labels):
        total, views = 0.0, 0
        with torch.no_grad():
            for frame in frames:
                cams = cameras_by_frame[frame]
                if not cams:
                    continue
                cand_xyz, cand_quat = candidate_state_at_frame(
                    xyz0, rot0, part_rotation, part_translation, part_quaternion, frame
                )
                sel = torch.zeros((int(xyz0.shape[0]), k_plus_1), dtype=cand_xyz.dtype, device=cand_xyz.device)
                sel.scatter_(1, labels.reshape(-1, 1), 1.0)
                xyz, quat = compose_straight_through_scene(cand_xyz, cand_quat, sel)
                for camera in cams:
                    loss_cam, *_ = render_fn(
                        camera, gaussians, pipe, background, dataset, args, xyz, quat
                    )
                    total += float(loss_cam)
                    views += 1
        return total / max(views, 1)

    base_labels = torch.as_tensor(hard_label).reshape(-1).long().clone()
    base = _loss_for(base_labels)
    for gid in gaussian_ids:
        gid = int(gid)
        entry = {"gaussian_id": gid, "current_label": int(base_labels[gid]), "base_loss": base}
        for k in range(k_plus_1):
            if k == int(base_labels[gid]):
                entry[f"delta_label_{k}"] = 0.0
                continue
            trial = base_labels.clone()
            trial[gid] = k
            entry[f"delta_label_{k}"] = _loss_for(trial) - base
        records.append(entry)
    return records
