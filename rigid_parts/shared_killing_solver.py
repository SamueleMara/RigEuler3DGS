"""The inner cycle of CODEX_MERGED_ROBUST_SUPPORT_AND_DETERMINISTIC_LOSS_ROUTING.

That contract supersedes the earlier shared-scalar and decoupled formulations
wherever they conflict (§0). The physical model is unchanged,

    v_k(x, t) = s_k(t) A(x) xi_k,    xi_k = Pi_rigid[Phi_k(.; theta_k)],
    T_k(t) = Exp(q_k(t) xi_k),       q_k(t) = int_0^t s_k,

with ONE trainable spatial field per moving part and ONE scalar trajectory per
moving part (§29, §31, §87). What this file implements on top of that:

  SUPPORT (§26).  ``build_robust_inner_support`` is the single integration
  point. Nothing below reconstructs a fitting mask from hard labels (§27); every
  loss takes its support from that object, and F_k SUBSET-OF T_k holds by
  construction -- an ambiguous Gaussian may be MOVED but may not define the
  joint.

  GEOMETRY BLOCK (§35-§46).  Optimizes theta only, with s(t) frozen. Losses:
  render, depth, motion DIRECTION, final PCD, strain, rigid projection. The
  informative ones (render/depth/direction/final-PCD) are mandatory -- a
  geometry step driven only by strain and rigidity is explicitly not valid
  motion identification (§37).

  SCALAR BLOCK (§47-§55).  Optimizes s(t) only, with xi detached. Losses:
  render, depth, motion MAGNITUDE, final PCD, continuity, sign flip, jerk.

Routing is enforced twice over: the frozen side is ``detach()``-ed so gradient
cannot flow there mathematically, and each block additionally flips
``requires_grad_`` and asserts the other block received nothing (§46, §55).
Per-loss gradient norms are audited each round (§59-§61) because equal loss
VALUES say nothing about equal influence.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from utils.motion_correspondence import MotionCorrespondenceCache
from utils.rigid_transform_utils import (
    apply_rotation_to_gaussian_quaternions,
    matrix_to_quaternion,
    normalize_quaternion,
)

from .killing_inner_solver import (
    NestedKillingMotionState,
    _compose_hard_partition_scene,
    _kernel_scale,
    _select_control_centers,
    _stack_part_transforms,
    multiview_render_loss_for_frame,
)
from .decoupled_joint_scalar import (
    alpha_scalar_agreement,
    analytical_frame_scalar,
    frame_epoch_schedule,
    frame_joint_consistency,
    motion_direction_loss,
    trimmed_chamfer,
    velocity_magnitude_loss,
)
from .handoff_diagnostics import (
    format_softp_handoff,
    soft_probability_statistics,
)
from .inner_support import (
    build_robust_inner_support,
    format_support_summary,
    gt_support_metrics,
    write_support_debug_ply,
)
from .local_killing import local_killing_quantities
from .twist_geometry import (
    axis_distance_statistics,
    rotational_field_variation,
    screw_decomposition,
    twist_gradient_diagnostics,
)
from .neighborhood import NeighborGraph, build_neighbor_graph, edge_part_weights
from .robust_inner_killing import (
    integrate_scalar,
    motion_prior_losses,
    observability,
    project_field_to_rigid_twist,
    rigid_field,
    single_field_rigid_projection_losses,
    scalar_jerk_loss,
    scalar_sign_flip_loss,
    scalar_continuity_loss,
    se3_exp_twist,
    sign_flip_diagnostics,
)

EPS = 1.0e-8

#: §56's routing table, as data. Used by the diagnostics and by the tests that
#: assert no other routing exists.
LOSS_ROUTING = {
    "render": ("geometry", "scalar"),
    "depth": ("geometry", "scalar"),
    "motion_direction": ("geometry",),
    "motion_magnitude": ("scalar",),
    "final_pcd": ("geometry", "scalar"),
    "strain": ("geometry",),
    "rigid": ("geometry",),
    "continuity": ("scalar",),
    "flip": ("scalar",),
    "jerk": ("scalar",),
}

GEOMETRY_INFORMATIVE = ("render", "depth", "motion_direction", "final_pcd")   # §37, §59
SCALAR_INFORMATIVE = ("render", "depth", "motion_magnitude", "final_pcd")     # §60


def geometry_parameters(control: torch.Tensor, speed: torch.Tensor) -> list[torch.Tensor]:
    """§34: the geometry block owns the spatial controls and nothing else."""
    return [control]


def scalar_parameters(control: torch.Tensor, speed: torch.Tensor) -> list[torch.Tensor]:
    """§34: the scalar block owns s(t) and nothing else."""
    return [speed]


# Kept as aliases so earlier specs' names still resolve.
stage_a_parameters = geometry_parameters
stage_b_parameters = scalar_parameters


def set_requires_grad(params, flag: bool) -> None:
    for p in params:
        p.requires_grad_(bool(flag))


def clear_gradients(params) -> None:
    """Drop the frozen block's gradients before a backward.

    Each block's optimizer only zeroes ITS OWN parameters, so without this the
    other block still holds whatever the previous block left behind and
    ``assert_no_gradient`` would fire on stale state rather than on a genuine
    routing leak.
    """
    for p in params:
        p.grad = None


def assert_no_gradient(params, where: str) -> None:
    """§46/§55: the frozen block must come back with nothing ADDED.

    Meaningful only because ``clear_gradients`` runs first, so a non-None
    gradient here can only have come from this block's backward.
    """
    for p in params:
        if p.grad is not None and bool(p.grad.abs().sum() > 0):
            raise AssertionError(
                f"{where}: a frozen parameter received a non-zero gradient "
                f"(|grad| sum {float(p.grad.abs().sum()):.6g}) -- loss routing is broken")


def per_loss_gradient_norms(terms: dict, params: list[torch.Tensor]) -> dict[str, float]:
    """§59/§60/§61: ||dL_j/dparam|| for each loss SEPARATELY.

    Evaluated with ``torch.autograd.grad`` so nothing touches ``.grad`` and the
    optimizer is never stepped (§59). This exists because equal loss values do
    not imply equal influence -- the only way to know which term is actually
    driving a block is to measure its gradient.
    """
    live = [p for p in params if p.requires_grad]
    out: dict[str, float] = {}
    for name, value in terms.items():
        if not torch.is_tensor(value) or not value.requires_grad or not live:
            out[name] = 0.0
            continue
        try:
            grads = torch.autograd.grad(value, live, retain_graph=True, allow_unused=True)
        except RuntimeError:
            out[name] = float("nan")
            continue
        total = sum(float(g.detach().norm()) ** 2 for g in grads if g is not None)
        out[name] = total ** 0.5
    return out


def gradient_ratios(norms: dict[str, float], *, block: str, eps: float = 1.0e-12) -> dict:
    """§18/§19: data-vs-regulariser balance, from GRADIENT norms not loss values.

    §61 of the merged contract and §14 here make the same point: two losses with
    similar magnitudes can have wildly different optimizer influence. These
    ratios are the summary that makes the imbalance readable at a glance.
    """
    if block == "geometry":
        data_keys, reg_keys = GEOMETRY_INFORMATIVE, ("strain", "rigid")
        pairs = {
            "motion_dir_over_render": ("motion_direction", "render"),
            "motion_dir_over_final_pcd": ("motion_direction", "final_pcd"),
            "strain_over_render": ("strain", "render"),
            "rigid_over_render": ("rigid", "render"),
        }
    else:
        data_keys, reg_keys = SCALAR_INFORMATIVE, ("continuity", "flip", "jerk")
        pairs = {
            "flip_over_render": ("flip", "render"),
            "flip_over_motion_mag": ("flip", "motion_magnitude"),
            "continuity_over_render": ("continuity", "render"),
            "final_pcd_over_render": ("final_pcd", "render"),
        }
    data = sum(float(norms.get(k, 0.0)) for k in data_keys)
    reg = sum(float(norms.get(k, 0.0)) for k in reg_keys)
    out = {"G_data": data, "G_reg": reg, "R": reg / (data + eps)}
    for name, (a, b) in pairs.items():
        out[name] = float(norms.get(a, 0.0)) / (float(norms.get(b, 0.0)) + eps)
    return out


def _jsonable_report(value):
    """A NaN in a debug file is indistinguishable from a parse error downstream,
    and json.dumps refuses non-finite floats under allow_nan=False. Map them to
    null so a diverged run stays readable."""
    import math

    if isinstance(value, dict):
        return {k: _jsonable_report(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_report(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _jsonable_report(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic(payload, path: Path) -> None:
    from utils.temporal_visibility_pretraining import _atomic_torch_save

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, path)


@torch.no_grad()
def _gather_observed_velocity(
    observed_cache, cameras_by_frame, times, xyz, quats, gaussians, pipe,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Observed velocity per interval: [T-1, N, 3] and its confidence [T-1, N].

    Gathered for the whole sequence up front so one backward can carry gradient
    from every frame into the single shared field.
    """
    frames = int(times.numel())
    n = int(xyz.shape[0])
    velocity = torch.zeros((max(frames - 1, 0), n, 3), device=xyz.device, dtype=xyz.dtype)
    confidence = torch.zeros((max(frames - 1, 0), n), device=xyz.device, dtype=xyz.dtype)
    if observed_cache is None or not getattr(observed_cache, "enabled", False):
        return velocity, confidence
    for interval in range(frames - 1):
        if interval + 1 >= len(cameras_by_frame):
            break
        state = observed_cache.observed_velocity_targets(
            xyz, cameras_by_frame[interval], cameras_by_frame[interval + 1],
            float(times[interval]), float(times[interval + 1]),
            gaussians=gaussians, pipe=pipe, query_rotations=quats)
        if state is None:
            continue
        v, c = state.get("velocity"), state.get("confidence")
        if v is not None:
            velocity[interval] = torch.as_tensor(v, device=xyz.device, dtype=xyz.dtype)
        if c is not None:
            confidence[interval] = torch.as_tensor(c, device=xyz.device, dtype=xyz.dtype).reshape(-1)
    return velocity, confidence


def solve_shared_killing_sequence(
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
    final_xyz: torch.Tensor | None = None,
    gt_moving: torch.Tensor | None = None,
    soft_part_probs: torch.Tensor | None = None,
    soft_part_probs_name: str = "part_prob (frozen one-hot)",
) -> NestedKillingMotionState:
    """Drop-in replacement for ``solve_nested_killing_sequence`` (same contract)."""
    xyz0 = torch.as_tensor(canonical_xyz).detach()
    device, dtype = xyz0.device, xyz0.dtype
    rot0 = normalize_quaternion(
        torch.as_tensor(canonical_rotation, dtype=dtype, device=device).detach())
    prob = torch.as_tensor(part_prob, dtype=dtype, device=device).detach()
    times = torch.as_tensor(frame_times, dtype=dtype, device=device).reshape(-1)
    moving_parts = int(prob.shape[1]) - 1
    n = int(xyz0.shape[0])
    frames = int(times.numel())
    if moving_parts <= 0:
        raise ValueError("the shared Killing solver needs at least one moving part")

    z = (prob.argmax(dim=-1) if hard_label is None
         else torch.as_tensor(hard_label, device=device).reshape(-1).long()).detach()
    if neighbor_graph is None:
        neighbor_graph = build_neighbor_graph(
            xyz0, k=int(getattr(args, "killing_neighbor_k", 16)),
            chunk_size=int(getattr(args, "killing_neighbor_chunk", 4096)))
    graph = neighbor_graph.to(device=device)
    dt = torch.diff(times) if frames > 1 else torch.zeros(0, device=device, dtype=dtype)
    scale = float(torch.linalg.norm(xyz0.max(dim=0).values - xyz0.min(dim=0).values).clamp_min(1.0e-8))
    target_final = (None if final_xyz is None
                    else torch.as_tensor(final_xyz, device=device, dtype=dtype).detach())
    printer = print          # log_fn is the pipeline's structured logger, not a printer
    out = Path(output_dir)

    # ================================================================ SUPPORT
    # §26: the ONE place support is built. §1: P is read, never written.
    #
    # CODEX_DEBUG §3/§8: the support module's confidence gates are only
    # meaningful if this is the PERSISTENT SOFT P. The pipeline separately holds
    # a frozen one-hot projection for the hard gate, and passing that here would
    # silently make every gate inert. The caller names which tensor it handed
    # over and the statistics below are computed from THAT object, not from some
    # earlier one.
    support_probs = prob if soft_part_probs is None else torch.as_tensor(
        soft_part_probs, device=device, dtype=dtype).detach()
    if support_probs.shape != prob.shape:
        raise ValueError(f"soft_part_probs {tuple(support_probs.shape)} does not match "
                         f"part_prob {tuple(prob.shape)}")
    handoff = soft_probability_statistics(support_probs, z)                   # §4/§5/§6
    printer(format_softp_handoff(soft_part_probs_name, support_probs, handoff))  # §8
    if bool(getattr(args, "assert_soft_inner_probs", False)) \
            and handoff["input_is_effectively_hard"]:
        printer("  (--assert_soft_inner_probs) the inner support gates cannot "
                "discriminate on this input")

    prob_before = prob.clone()
    z_before = z.clone()
    support_probs_before = support_probs.clone()
    support = build_robust_inner_support(
        xyz=xyz0, soft_part_probs=support_probs, hard_labels=z, num_moving_parts=moving_parts,
        knn=int(getattr(args, "inner_support_knn", 16)),
        outlier_mad_scale=float(getattr(args, "inner_outlier_mad_scale", 3.5)),
        outlier_local_dynamic_min=float(getattr(args, "inner_outlier_local_dynamic_min", 0.35)),
        core_dynamic_confidence=float(getattr(args, "inner_core_dynamic_confidence", 0.80)),
        core_part_confidence=float(getattr(args, "inner_core_part_confidence", 0.70)),
        core_confidence_floor=float(getattr(args, "inner_core_confidence_floor", 0.60)),
        core_min_points=getattr(args, "inner_core_min_points", None),
        cluster_knn=int(getattr(args, "inner_cluster_knn", 16)),
        cluster_dynamic_gamma=float(getattr(args, "inner_cluster_confidence_gamma", 1.0)),
        cluster_part_gamma=float(getattr(args, "inner_cluster_part_gamma", 0.5)),
        attach_min_confidence=float(getattr(args, "inner_attach_min_confidence", 0.55)),
        attach_margin_ratio=float(getattr(args, "inner_attach_margin_ratio", 1.5)),
        attach_affinity_quantile=float(getattr(args, "inner_attach_affinity_quantile", 0.25)),
        fit_confidence_floor=float(getattr(args, "inner_fit_confidence_floor", 0.50)),
        fit_confidence_gamma=float(getattr(args, "inner_fit_confidence_gamma", 2.0)),
        fit_part_gamma=float(getattr(args, "inner_fit_part_gamma", 1.0)),
        seed=int(getattr(args, "seed", 20260919)))
    # §1's regression invariant, checked at runtime and not only in the tests.
    if not (torch.equal(prob, prob_before) and torch.equal(z, z_before)
            and torch.equal(support_probs, support_probs_before)):
        raise AssertionError("support preprocessing mutated the persistent outer labels")
    printer(f"[Decoupled] outer {outer_iter}")
    printer(format_support_summary(support))                                  # §63

    support_gt = None
    if gt_moving is not None:                                                 # §64
        support_gt = gt_support_metrics(support, gt_moving)
        printer(f"  GT (evaluation only): transport prec "
                f"{support_gt['transport']['precision']*100:.2f}%  fitting prec "
                f"{support_gt['fitting']['precision']*100:.2f}%  fitting rec "
                f"{support_gt['fitting']['recall']*100:.2f}%  "
                f"purity gain {'YES' if support_gt['fitting_precision_exceeds_transport'] else 'NO'}")

    # CODEX_DEBUG §33/§35 (storage ablations B and D): replace the FITTING
    # support with GT. Evaluation/debug only -- it answers "can the geometry
    # objective recover the right joint with perfect support?" and its result is
    # never evidence of unsupervised performance (§35). Transport is untouched,
    # so F_k SUBSET-OF T_k can still fail loudly rather than silently.
    if bool(getattr(args, "debug_gt_fitting_support", False)):
        if gt_moving is None:
            printer("[Decoupled] --debug_gt_fitting_support requested but no GT is "
                    "available; keeping the real support")
        else:
            gt = torch.as_tensor(gt_moving, device=device).reshape(-1).bool()
            printer(f"[Decoupled] DEBUG: OVERRIDING the fitting support with GT "
                    f"({int(gt.sum())} moving). This is a diagnostic upper bound, "
                    f"NOT an unsupervised result.")
            new_fit = torch.full_like(support.fitting_part_id, -1)
            for k in range(moving_parts):
                # keep the discovered cluster split where it exists, else part 1
                take = gt & (support.transport_part_id == (k + 1))
                new_fit[take] = k + 1
            orphan = gt & (new_fit < 0)
            if bool(orphan.any()):
                new_fit[orphan] = 1
            support.fitting_part_id = new_fit
            support.fitting_masks = torch.stack(
                [new_fit == (k + 1) for k in range(moving_parts)])
            support.fit_weight = torch.where(new_fit > 0,
                                             torch.ones_like(support.fit_weight),
                                             torch.zeros_like(support.fit_weight))
            support.diagnostics["gt_fitting_override"] = int(gt.sum())

    transport_mask = [support.transport_masks[k] for k in range(moving_parts)]
    fit_index = [support.fitting_index(k) for k in range(moving_parts)]
    fit_w = torch.stack([torch.where(support.fitting_masks[k], support.fit_weight,
                                     torch.zeros_like(support.fit_weight))
                         for k in range(moving_parts)])
    min_fit = int(getattr(args, "inner_min_fit_points", 6))
    for part in range(moving_parts):
        if int(fit_index[part].numel()) < min_fit:
            # §28: report, never inject all hard-moving candidates.
            printer(f"[Decoupled] part {part + 1}: fitting support "
                    f"{int(fit_index[part].numel())} < {min_fit}; motion marked INVALID "
                    f"(identity this outer, outer labels untouched)")

    # ------------------------------------------------------------- observed
    observed_cache = None
    if float(getattr(args, "lambda_inner_observed_velocity", 0.0)) > 0.0:
        observed_cache = MotionCorrespondenceCache(dataset, args)
    observed_v, observed_c = _gather_observed_velocity(
        observed_cache, cameras_by_frame, times, xyz0, rot0, gaussians, pipe)

    with torch.no_grad():
        supported = observed_c > 0
        typical = float(observed_v.norm(dim=-1)[supported].median()) if bool(supported.any()) else 0.0
        if not (typical > 0.0):
            typical = 1.0e-3 * scale
    robust_delta = float(getattr(args, "inner_frame_robust_scale", 1.0)) * typical
    robust_tau = float(getattr(args, "inner_robust_tau_scale", 2.0)) * typical
    robust_mode = str(getattr(args, "inner_frame_robust", "huber"))
    irls_steps = int(getattr(args, "inner_irls_steps", 2))

    # ------------------------------------------- §29: ONE spatial field per part
    prob_moving = prob[:, 1:]
    centers = torch.stack([
        _select_control_centers(
            xyz0, (prob_moving[:, part] * support.fitting_masks[part].to(dtype)),
            int(getattr(args, "nested_killing_control_points", 64)))
        for part in range(moving_parts)], dim=0)
    sigma = torch.stack([
        _kernel_scale(centers[part], float(getattr(args, "nested_killing_kernel_scale", 0.0)))
        for part in range(moving_parts)]).to(device=device, dtype=dtype)

    control = torch.zeros_like(centers)
    if warm_start_field:
        previous = warm_start_field[0].to(control)
        if previous.shape == control.shape:
            control = previous.clone()
    # §29/§30: ``control`` is the only trainable spatial object. No omega_g, no
    # u_g, no global twist, no per-frame spatial field.
    control = torch.nn.Parameter(control)
    # §31: the only time-varying motion parameter.
    speed = torch.nn.Parameter(torch.ones((moving_parts, frames), device=device, dtype=dtype))

    xyz_by_part = xyz0[None].expand(moving_parts, -1, -1)
    edge_w = edge_part_weights(graph, prob_moving, z)

    lam = {
        "render_G": float(getattr(args, "lambda_geometry_render", 1.0)),
        "depth_G": float(getattr(args, "lambda_geometry_depth", 1.0)),
        "motion_dir": float(getattr(args, "lambda_motion_direction", 1.0)),
        "final_G": float(getattr(args, "lambda_geometry_final_pcd", 1.0)),
        "strain": float(getattr(args, "lambda_inner_strain", 10.0)),
        "rigid": float(getattr(args, "lambda_rigid_proj", 1.0)),
        "render_S": float(getattr(args, "lambda_scalar_render", 1.0)),
        "depth_S": float(getattr(args, "lambda_scalar_depth", 1.0)),
        "motion_mag": float(getattr(args, "lambda_motion_magnitude", 1.0)),
        "final_S": float(getattr(args, "lambda_scalar_final_pcd", 1.0)),
        "cont": float(getattr(args, "lambda_s_continuity", 0.01)),
        "flip": float(getattr(args, "lambda_s_flip", 0.01)),
        "jerk": float(getattr(args, "lambda_s_jerk", 0.0)),
        "revolute": float(getattr(args, "lambda_revolute_prior", 0.0)),
        "prismatic": float(getattr(args, "lambda_prismatic_prior", 0.0)),
    }
    prior_mode = str(getattr(args, "inner_motion_prior", "none"))
    rounds = int(getattr(args, "inner_alternation_rounds", 3))
    geometry_steps = int(getattr(args, "inner_spatial_steps", 80))
    scalar_steps = int(getattr(args, "inner_scalar_steps", 80))
    geometry_lr = float(getattr(args, "killing_lr", 1.0e-2))
    scalar_lr = float(getattr(args, "inner_scalar_lr", 1.0e-2))

    history: list[dict] = []
    geometry_log: list[dict] = []
    scalar_log: list[dict] = []
    grad_audit: list[dict] = []
    seeded_from: dict[int, int] = {}
    anchored: dict[int, float] = {}
    alpha_bootstrap: dict[int, list[float]] = {}

    # ---------------------------------------------------------------- pieces
    def _field_and_projection():
        """Phi_local -> xi(theta) -> Phi_rigid. Recomputed every forward pass and
        NOT detached in the geometry block (§35)."""
        q = local_killing_quantities(
            xyz_by_part, centers, control, sigma, prob_moving, graph, edge_w,
            jacobian_mode=str(getattr(args, "killing_jacobian_mode", "analytic")))
        projections = []
        for part in range(moving_parts):
            idx = fit_index[part]
            if int(idx.numel()) < min_fit:
                projections.append(None)
                continue
            projections.append(project_field_to_rigid_twist(
                xyz0.index_select(0, idx),
                q["velocity"][part].index_select(0, idx),
                fit_w[part].index_select(0, idx),
                reg=float(getattr(args, "inner_projection_reg", 1.0e-8))))
        return q, projections

    def _field_rms(projections, part):
        proj, idx = projections[part], fit_index[part]
        if proj is None or int(idx.numel()) == 0:
            return None
        w = fit_w[part].index_select(0, idx)
        return ((w * proj["rigid_velocity"].square().sum(-1)).sum()
                / (w.sum() + EPS)).clamp_min(EPS).sqrt()

    def _observation(part):
        """§23: w = w_fit * w_visibility * w_motion-confidence, restricted to F_k.
        Outlier and unassigned points cannot appear here at all (§20, §33, §51)."""
        idx = fit_index[part]
        if int(idx.numel()) == 0 or int(observed_v.shape[0]) == 0:
            return None, None
        v_obs = observed_v.index_select(1, idx)
        w = (fit_w[part].index_select(0, idx)[None]
             * observed_c.index_select(1, idx)).clamp_min(0.0)
        if float(w.sum()) <= 0.0:
            return None, None
        return v_obs, w

    def _integrate(speed_now):
        return torch.stack([integrate_scalar(speed_now[part], dt) for part in range(moving_parts)])

    def _transported_scene(projections, q_all, t):
        """§21: transport uses T_k, which is wider than F_k."""
        moved_xyz, moved_q = xyz0, rot0
        for part in range(moving_parts):
            proj = projections[part]
            mask = transport_mask[part]
            if proj is None or not bool(mask.any()):
                continue
            T = se3_exp_twist(proj["omega"], proj["u"], q_all[part][t : t + 1])[0]
            shifted = xyz0 @ T[:3, :3].transpose(0, 1) + T[:3, 3]
            moved_xyz = torch.where(mask[:, None], shifted, moved_xyz)
            quat = normalize_quaternion(matrix_to_quaternion(T[:3, :3][None]))
            moved_q = torch.where(mask[:, None],
                                  apply_rotation_to_gaussian_quaternions(rot0, quat), moved_q)
        return moved_xyz, moved_q

    moving_any = torch.stack(transport_mask).any(dim=0)

    def _render_and_depth(projections, q_all, frame_subset, per_frame=None):
        """§38/§39 and §49/§50: separately weighted render and depth over all
        cameras of each sampled frame."""
        zero = control.sum() * 0.0 + speed.sum() * 0.0
        render_total, depth_total, counted = zero, zero, 0
        for t in frame_subset:
            cams = cameras_by_frame[t] if t < len(cameras_by_frame) else []
            if not cams:
                continue
            moved_xyz, moved_q = _transported_scene(projections, q_all, t)
            res = multiview_render_loss_for_frame(
                cameras=cams, gaussians=gaussians, pipe=pipe, background=background,
                dataset=dataset, args=args, positions=moved_xyz, rotations=moved_q,
                moving_mask=moving_any,
                lambda_view_var=float(getattr(args, "lambda_multiview_view_var", 0.0)))
            render_total = render_total + res["photometric"]
            depth_total = depth_total + res["depth_loss"]
            counted += 1
            if per_frame is not None:
                per_frame[int(t)] = {"render": float(res["photometric"].detach()),
                                     "depth": float(res["depth_loss"].detach())}
        denom = max(counted, 1)
        return render_total / denom, depth_total / denom

    def _final_pcd(projections, q_all, report=None):
        """§41/§52 with §42's moving-aware matching: forward from the transported
        moving support to the full final cloud, plus a BACKWARD term restricted
        to a budget of the nearest final points. §42 explicitly rejects a small
        moving prediction against the entire final scene under an unweighted
        symmetric Chamfer, which would let static geometry dominate."""
        total = control.sum() * 0.0 + speed.sum() * 0.0
        if target_final is None:
            return total
        for part in range(moving_parts):
            proj, mask = projections[part], transport_mask[part]
            if proj is None or not bool(mask.any()):
                continue
            T = se3_exp_twist(proj["omega"], proj["u"], q_all[part][-1:])[0]
            src = xyz0[mask]
            pred = src @ T[:3, :3].transpose(0, 1) + T[:3, 3]
            res = trimmed_chamfer(
                pred, target_final,
                forward_trim=float(getattr(args, "inner_final_pcd_forward_trim", 0.8)),
                backward_trim=float(getattr(args, "inner_final_pcd_backward_trim", 0.8)),
                backward_budget=float(getattr(args, "inner_final_pcd_budget", 1.5)))
            total = total + res["loss"] / (scale * scale)
            if report is not None:
                report[part] = {"forward": float(res["forward"].detach()) / (scale * scale),
                                "backward": float(res["backward"].detach()) / (scale * scale)}
        return total

    def _geometry_terms(frame_subset, per_frame=None):
        """§45. Every term backpropagates only to theta: s(t) enters detached."""
        q, projections = _field_and_projection()
        q_all = _integrate(speed.detach())                        # §36: frozen s
        render, depth = _render_and_depth(projections, q_all, frame_subset, per_frame)
        final = _final_pcd(projections, q_all)

        zero = control.sum() * 0.0
        strain, rigid, direction = q["strain_loss"], zero, zero
        rev = pri = zero
        for part in range(moving_parts):
            proj, idx = projections[part], fit_index[part]
            if proj is None or int(idx.numel()) == 0:
                continue
            terms = single_field_rigid_projection_losses(
                xyz0.index_select(0, idx), q["velocity"][part].index_select(0, idx),
                q["omega_local"][part].index_select(0, idx),
                q["b_local"][part].index_select(0, idx),
                proj, fit_w[part].index_select(0, idx))
            rigid = rigid + terms["rigid_proj"]
            v_obs, w = _observation(part)
            if v_obs is not None:
                direction = direction + motion_direction_loss(proj["rigid_velocity"], v_obs, w)
            out_prior = motion_prior_losses(proj["omega"], proj["u"], prior_mode)
            rev, pri = rev + out_prior["revolute"], pri + out_prior["prismatic"]
        return {"render": render, "depth": depth, "motion_direction": direction,
                "final_pcd": final, "strain": strain, "rigid": rigid,
                "revolute": rev, "prismatic": pri}, projections

    # §15/§16: the lambda actually applied to each term, so raw loss, lambda,
    # weighted loss and weighted gradient norm can all be reported together.
    _LAMBDA_OF = {
        "geometry": {"render": lam["render_G"], "depth": lam["depth_G"],
                     "motion_direction": lam["motion_dir"], "final_pcd": lam["final_G"],
                     "strain": lam["strain"], "rigid": lam["rigid"],
                     "revolute": lam["revolute"], "prismatic": lam["prismatic"]},
        "scalar": {"render": lam["render_S"], "depth": lam["depth_S"],
                   "motion_magnitude": lam["motion_mag"], "final_pcd": lam["final_S"],
                   "continuity": lam["cont"], "flip": lam["flip"], "jerk": lam["jerk"]},
    }

    def _weighted_terms(terms, block):
        """§14: probes must measure grad(lambda * L), not grad(L)."""
        table = _LAMBDA_OF[block]
        return {k: table.get(k, 0.0) * v for k, v in terms.items()}

    @torch.no_grad()
    def _twist_geometry(projections):
        """§26-§30: screw decomposition, axis distance to the fitting support and
        rotational-field variation, per part."""
        out: dict[int, dict] = {}
        for part in range(moving_parts):
            proj, idx = projections[part], fit_index[part]
            if proj is None or int(idx.numel()) == 0:
                continue
            omega, u = proj["omega"].detach(), proj["u"].detach()
            dec = screw_decomposition(omega, u)
            pts = xyz0.index_select(0, idx)
            w = fit_w[part].index_select(0, idx)
            out[part] = {
                "screw": dec,
                "axis_distance": axis_distance_statistics(pts, w, dec),
                "rotational_variation": rotational_field_variation(pts, w, omega, u),
            }
        return out

    def _geometry_total(terms):
        return (lam["render_G"] * terms["render"] + lam["depth_G"] * terms["depth"]
                + lam["motion_dir"] * terms["motion_direction"]
                + lam["final_G"] * terms["final_pcd"]
                + lam["strain"] * terms["strain"] + lam["rigid"] * terms["rigid"]
                + lam["revolute"] * terms["revolute"] + lam["prismatic"] * terms["prismatic"])

    def _scalar_terms(frozen, frame_subset, per_frame=None, pcd_report=None):
        """§54. ``frozen`` holds the DETACHED xi, so nothing reaches theta."""
        q_all = _integrate(speed)
        render, depth = _render_and_depth(frozen, q_all, frame_subset, per_frame)
        final = _final_pcd(frozen, q_all, report=pcd_report)
        zero = speed.sum() * 0.0
        magnitude = cont = flip = jerk = zero
        for part in range(moving_parts):
            proj = frozen[part]
            v_obs, w = _observation(part)
            if proj is not None and v_obs is not None:
                s_mid = 0.5 * (speed[part][:-1] + speed[part][1:])
                magnitude = magnitude + velocity_magnitude_loss(
                    proj["rigid_velocity"], v_obs, w, s_mid,
                    mode=robust_mode, delta=robust_delta)
            cont = cont + scalar_continuity_loss(speed[part], dt)
            flip = flip + scalar_sign_flip_loss(speed[part])
            jerk = jerk + scalar_jerk_loss(speed[part], dt)
        return {"render": render, "depth": depth, "motion_magnitude": magnitude,
                "final_pcd": final, "continuity": cont, "flip": flip, "jerk": jerk}

    def _scalar_total(terms):
        return (lam["render_S"] * terms["render"] + lam["depth_S"] * terms["depth"]
                + lam["motion_mag"] * terms["motion_magnitude"]
                + lam["final_S"] * terms["final_pcd"]
                + lam["cont"] * terms["continuity"] + lam["flip"] * terms["flip"]
                + lam["jerk"] * terms["jerk"])

    # ================================================= §32/§57: ONE-TIME BOOTSTRAP
    # Seeds the spatial field and anchors its magnitude from observed motion on
    # the FITTING support only (§33). alpha* is used here and for diagnostics --
    # it is never a third optimized state.
    target_rms: dict[int, float] = {}
    with torch.no_grad():
        for part in range(moving_parts):
            idx = fit_index[part]
            v_obs, w = _observation(part)
            if int(idx.numel()) < min_fit or v_obs is None:
                continue
            energy = (w * v_obs.norm(dim=-1)).sum(dim=1)
            t_star = int(energy.argmax())
            if float(energy[t_star]) <= 0.0:
                continue
            wt = w[t_star]
            target_rms[part] = float(((wt * v_obs[t_star].square().sum(-1)).sum()
                                      / (wt.sum() + EPS)).clamp_min(EPS).sqrt())
            if warm_start_field:
                continue
            seed = project_field_to_rigid_twist(
                xyz0.index_select(0, idx), v_obs[t_star], wt,
                reg=float(getattr(args, "inner_projection_reg", 1.0e-8)))
            field = rigid_field(centers[part], seed["omega"], seed["u"])
            if float(field.square().sum(-1).mean()) <= EPS:
                continue
            control[part] = field
            seeded_from[part] = t_star
        if seeded_from:
            printer("  bootstrap: seeded the shared field from observed motion at "
                    + ", ".join(f"part {p + 1} <- interval {t}"
                                for p, t in sorted(seeded_from.items())))

        # The gauge normalises Phi and compensates s(t), but that compensation
        # does not survive the outer loop: ``speed`` is rebuilt as ones on every
        # solve while ``control`` is warm-started, so the pair splits even though
        # only their PRODUCT is physical. Re-anchoring the field magnitude to the
        # observed motion restores it. Measured on blade without this, the
        # warm-started field arrived ~18x too large and the endpoint blew up.
        _, probe = _field_and_projection()
        for part in range(moving_parts):
            rms_now = _field_rms(probe, part)
            if rms_now is None or part not in target_rms or float(rms_now) <= EPS:
                continue
            control[part] *= target_rms[part] / float(rms_now)
            anchored[part] = target_rms[part]
        if anchored:
            printer("  bootstrap: anchored the field magnitude to observed motion: "
                    + ", ".join(f"part {p + 1} rms {v:.4g}" for p, v in sorted(anchored.items())))

        # §32: alpha* also initialises s(t). After the anchor Phi carries the
        # observed magnitude, so the gauge alone would give a constant s; alpha*
        # gives its temporal SHAPE. It is a bootstrap, not a constraint -- every
        # scalar loss is free to move s(t) from here.
        _, probe = _field_and_projection()
        for part in range(moving_parts):
            proj = probe[part]
            v_obs, w = _observation(part)
            if proj is None or v_obs is None:
                continue
            a = analytical_frame_scalar(proj["rigid_velocity"], v_obs, w)
            if int(a.numel()) == 0 or not bool(torch.isfinite(a).all()):
                continue
            s_init = torch.empty(frames, device=device, dtype=dtype)
            s_init[0], s_init[-1] = a[0], a[-1]
            if frames > 2:
                s_init[1:-1] = 0.5 * (a[:-1] + a[1:])
            speed[part] = s_init
            alpha_bootstrap[part] = [float(v) for v in a]
        if alpha_bootstrap:
            printer("  bootstrap: initialised s(t) from alpha*: "
                    + ", ".join(f"part {p + 1} |alpha|avg {sum(abs(v) for v in a)/max(len(a),1):.4g}"
                                for p, a in sorted(alpha_bootstrap.items())))

    # =========================================== §57: deterministic main loop
    all_frames = list(range(min(frames, len(cameras_by_frame))))
    stride = max(1, int(getattr(args, "inner_render_frame_stride", 1)))
    candidate_frames = all_frames[::stride]
    batch = int(getattr(args, "inner_render_frame_batch", 4))
    geometry_opt = torch.optim.Adam(geometry_parameters(control, speed), lr=geometry_lr)
    progress = tqdm(range(max(1, rounds)), desc=f"Inner outer={outer_iter}",
                    unit="round", leave=False)
    visited_geometry: set[int] = set()
    visited_scalar: set[int] = set()

    for rnd in progress:
        # ------------------------------------------------- A. GEOMETRY BLOCK
        set_requires_grad(geometry_parameters(control, speed), True)
        speed.requires_grad_(False)                                         # §46
        schedule = frame_epoch_schedule(max(1, geometry_steps), candidate_frames, batch,
                                        seed=int(getattr(args, "seed", 0)) + 17 * rnd)
        per_frame_G: dict[int, dict] = {}
        for step in range(max(1, geometry_steps)):
            geometry_opt.zero_grad(set_to_none=True)
            subset = schedule[step]
            visited_geometry.update(subset)
            last = step == max(1, geometry_steps) - 1
            terms, _ = _geometry_terms(subset, per_frame_G if last else None)
            total = _geometry_total(terms)
            if not bool(torch.isfinite(total)):
                printer(f"  GEOMETRY round {rnd} step {step}: NON-FINITE loss ("
                        + " ".join(f"{k}={float(v):.4g}" for k, v in terms.items())
                        + "); ending this block")
                break
            clear_gradients(scalar_parameters(control, speed))
            total.backward()
            assert_no_gradient(scalar_parameters(control, speed), "geometry block")   # §46
            geometry_opt.step()
            geometry_log.append({"round": rnd, "step": step, "block": "geometry",
                                 **{k: float(v.detach()) for k, v in terms.items()},
                                 "total": float(total.detach())})

        # §15/§59: per-loss geometry gradient norms of the WEIGHTED losses,
        # measured without stepping. §14: raw magnitudes cannot be compared.
        set_requires_grad(geometry_parameters(control, speed), True)
        speed.requires_grad_(False)
        terms, probe_projections = _geometry_terms(schedule[-1])
        weighted = _weighted_terms(terms, "geometry")
        g_norms = per_loss_gradient_norms(weighted, geometry_parameters(control, speed))
        g_raw = {k: float(v.detach()) if torch.is_tensor(v) else float(v)
                 for k, v in terms.items()}
        g_weighted = {k: float(v.detach()) if torch.is_tensor(v) else float(v)
                      for k, v in weighted.items()}
        g_ratios = gradient_ratios(g_norms, block="geometry")                     # §18
        informative = max(g_norms.get(k, 0.0) for k in GEOMETRY_INFORMATIVE)
        if not (informative > 0.0):
            printer(f"  round {rnd}: NO INFORMATIVE GEOMETRY GRADIENT "
                    f"(render {g_norms.get('render', 0.0):.3g} depth {g_norms.get('depth', 0.0):.3g} "
                    f"dir {g_norms.get('motion_direction', 0.0):.3g} "
                    f"final {g_norms.get('final_pcd', 0.0):.3g})")

        # §21-§24: the same per-loss gradients at the DERIVED twist. storage's
        # failure is physical (too much rotation), which the control gradients
        # cannot show but dL/domega and dL/du can.
        twist_grads: dict[int, dict] = {}
        for part in range(moving_parts):
            proj = probe_projections[part]
            if proj is None:
                continue
            twist_grads[part] = twist_gradient_diagnostics(
                weighted, proj["omega"], proj["u"])
        round_twist_geometry = _twist_geometry(probe_projections)                 # §26-§30

        # ---- gauge: unit-RMS field, magnitude compensated into s(t) --------
        with torch.no_grad():
            _, projections = _field_and_projection()
            for part in range(moving_parts):
                rms = _field_rms(projections, part)
                if rms is not None and float(rms) > EPS:
                    control[part] /= rms
                    speed[part] *= rms
                magnitude = speed[part].abs()
                if bool((magnitude > EPS).any()):
                    lead = int(torch.nonzero(magnitude > magnitude.max() * 0.1, as_tuple=False)[0])
                    if float(speed[part][lead]) < 0:      # anchor the sign gauge
                        control[part] *= -1
                        speed[part] *= -1

        # --------------------------------------------------- B. SCALAR BLOCK
        set_requires_grad(geometry_parameters(control, speed), False)        # §55
        speed.requires_grad_(True)
        with torch.no_grad():
            _, live = _field_and_projection()
        # §47: this detach is mandatory. It is what makes "final PCD updates the
        # trajectory, not the joint" true rather than nominal.
        frozen = [None if p is None else
                  {k: (v.detach() if torch.is_tensor(v) else v) for k, v in p.items()}
                  for p in live]
        mean_s = float(speed.detach().abs().mean())
        if not (mean_s > 0.0) or mean_s != mean_s:
            printer(f"  SCALAR round {rnd}: |s| is {mean_s}; using an absolute step")
            mean_s = 1.0
        # Adam's step is ~lr regardless of gradient magnitude, and s(t) carries
        # the physical scale of the motion (order 1e-3 here), so the scalar lr is
        # RELATIVE to |s| -- an absolute 1e-2 would move s by several times its
        # own value in one step.
        scalar_opt = torch.optim.Adam(scalar_parameters(control, speed), lr=scalar_lr * mean_s)
        schedule = frame_epoch_schedule(max(1, scalar_steps), candidate_frames, batch,
                                        seed=int(getattr(args, "seed", 0)) + 101 * rnd)
        per_frame_S: dict[int, dict] = {}
        pcd_report: dict[int, dict] = {}
        last_finite = speed.detach().clone()
        for step in range(max(1, scalar_steps)):
            scalar_opt.zero_grad(set_to_none=True)
            subset = schedule[step]
            visited_scalar.update(subset)
            last = step == max(1, scalar_steps) - 1
            terms = _scalar_terms(frozen, subset, per_frame_S if last else None,
                                  pcd_report if last else None)
            total = _scalar_total(terms)
            detail = " ".join(f"{k}={float(v):.4g}" for k, v in terms.items())
            if not bool(torch.isfinite(total)):
                printer(f"  SCALAR round {rnd} step {step}: NON-FINITE loss ({detail}); "
                        "rolling s(t) back and ending this block")
                with torch.no_grad():
                    speed.copy_(last_finite)
                break
            clear_gradients(geometry_parameters(control, speed))
            total.backward()
            assert_no_gradient(geometry_parameters(control, speed), "scalar block")   # §55
            if speed.grad is None or not bool(torch.isfinite(speed.grad).all()):
                printer(f"  SCALAR round {rnd} step {step}: NON-FINITE gradient ({detail}); "
                        "rolling s(t) back and ending this block")
                with torch.no_grad():
                    speed.copy_(last_finite)
                break
            scalar_opt.step()
            if not bool(torch.isfinite(speed.detach()).all()):
                printer(f"  SCALAR round {rnd} step {step}: s(t) went non-finite; rolling back")
                with torch.no_grad():
                    speed.copy_(last_finite)
                break
            last_finite = speed.detach().clone()
            row = {"round": rnd, "step": step, "block": "scalar",
                   **{k: float(v.detach()) for k, v in terms.items()},
                   "total": float(total.detach())}
            scalar_log.append(row)
            history.append(row)

        # §16/§60: per-loss scalar gradient norms of the WEIGHTED losses.
        terms = _scalar_terms(frozen, schedule[-1])
        weighted_s = _weighted_terms(terms, "scalar")
        s_norms = per_loss_gradient_norms(weighted_s, scalar_parameters(control, speed))
        s_raw = {k: float(v.detach()) if torch.is_tensor(v) else float(v)
                 for k, v in terms.items()}
        s_weighted = {k: float(v.detach()) if torch.is_tensor(v) else float(v)
                      for k, v in weighted_s.items()}
        s_ratios = gradient_ratios(s_norms, block="scalar")                       # §19
        if not (max(s_norms.get(k, 0.0) for k in SCALAR_INFORMATIVE) > 0.0):
            printer(f"  round {rnd}: NO INFORMATIVE SCALAR GRADIENT")
        grad_audit.append({
            "round": rnd,
            # §15/§16 want raw loss, lambda, weighted loss AND weighted gradient
            # norm for every term, so all four are kept side by side.
            "geometry": g_norms, "scalar": s_norms,
            "geometry_raw_loss": g_raw, "geometry_weighted_loss": g_weighted,
            "scalar_raw_loss": s_raw, "scalar_weighted_loss": s_weighted,
            "lambdas": {"geometry": _LAMBDA_OF["geometry"], "scalar": _LAMBDA_OF["scalar"]},
            "geometry_ratios": g_ratios, "scalar_ratios": s_ratios,
            "twist_gradients": {int(k): v for k, v in twist_grads.items()},
            "twist_geometry": round_twist_geometry,
            "geometry_per_frame": per_frame_G, "scalar_per_frame": per_frame_S,
            "final_pcd_parts": pcd_report})
        progress.set_postfix(
            G=f"{geometry_log[-1]['total']:.4g}" if geometry_log else "-",
            S=f"{scalar_log[-1]['total']:.4g}" if scalar_log else "-")

    set_requires_grad(geometry_parameters(control, speed), True)
    speed.requires_grad_(True)

    # ============================================================ integrate
    with torch.no_grad():
        speed_final = speed.detach()
        q_traj = _integrate(speed_final)
        _, final_projections = _field_and_projection()
        eye4 = torch.eye(4, device=device, dtype=dtype)
        transforms = torch.stack([
            (se3_exp_twist(final_projections[p]["omega"], final_projections[p]["u"], q_traj[p])
             if final_projections[p] is not None else eye4.repeat(frames, 1, 1))
            for p in range(moving_parts)])
        valid_motion = [p is not None and p["valid"] for p in final_projections]
        final_pcd_report: dict[int, dict] = {}
        if target_final is not None:
            _final_pcd(final_projections, q_traj, report=final_pcd_report)
        alpha_final: dict[int, list[float]] = {}
        joint_residual: dict[int, torch.Tensor] = {}
        frame_joint: dict[int, list[float]] = {}
        for part in range(moving_parts):
            proj = final_projections[part]
            v_obs, w = _observation(part)
            if proj is None or v_obs is None:
                continue
            fit = frame_joint_consistency(proj["rigid_velocity"], v_obs, w,
                                          mode=robust_mode, delta=robust_delta,
                                          robust_tau=robust_tau, irls_steps=irls_steps)
            alpha_final[part] = [float(v) for v in fit.alpha]
            joint_residual[part] = fit.point_residual.cpu()
            frame_joint[part] = [float(v) for v in fit.per_frame_loss]

    part_r, part_t, part_q = [], [], []
    positions, rotations = [], []
    for t in range(frames):
        R, T = transforms[:, t, :3, :3], transforms[:, t, :3, 3]
        part_r.append(R.clone())
        part_t.append(T.clone())
        quat = normalize_quaternion(matrix_to_quaternion(R))
        part_q.append(quat)
        moved = torch.stack([xyz0 @ R[p].transpose(0, 1) + T[p] for p in range(moving_parts)])
        quats = torch.stack([apply_rotation_to_gaussian_quaternions(rot0, quat[p : p + 1])
                             for p in range(moving_parts)])
        scene_xyz, scene_q = _compose_hard_partition_scene(
            xyz0, rot0, moved, quats, support.transport_part_id)
        positions.append(scene_xyz)
        rotations.append(scene_q)
    stack_r, stack_t, stack_q = _stack_part_transforms(part_r, part_t, part_q, prob, xyz0)

    # ================================================ §62/§65/§66 diagnostics
    diagnostics = {"outer": int(outer_iter), "parts": [],
                   "frame_coverage": {
                       "candidate_frames": len(candidate_frames),
                       "geometry_frames_visited": len(visited_geometry),
                       "scalar_frames_visited": len(visited_scalar),
                       "render_frame_stride": stride,
                       "render_frame_batch": batch,
                       "geometry_full_coverage": len(visited_geometry) >= len(candidate_frames),
                       "scalar_full_coverage": len(visited_scalar) >= len(candidate_frames)}}
    for part in range(moving_parts):
        idx = fit_index[part]
        pts = xyz0.index_select(0, idx) if int(idx.numel()) else xyz0[:0]
        w = fit_w[part].index_select(0, idx) if int(idx.numel()) else fit_w[part][:0]
        rot_deg = float(torch.rad2deg(torch.arccos(
            ((torch.diagonal(transforms[part, -1, :3, :3]).sum() - 1) / 2).clamp(-1, 1))))
        proj = final_projections[part]
        s_mid = (0.5 * (speed_final[part][:-1] + speed_final[part][1:])
                 if frames > 1 else speed_final[part])
        entry = {
            "part": part + 1,
            "transport_support": int(transport_mask[part].sum()),
            "fitting_support": int(idx.numel()),
            "valid_motion": bool(valid_motion[part]),
            "endpoint_rotation_deg": rot_deg,
            "endpoint_translation_over_bbox": float(transforms[part, -1, :3, 3].norm()) / scale,
            "omega_star_norm": 0.0 if proj is None else float(proj["omega"].norm()),
            "u_star_norm": 0.0 if proj is None else float(proj["u"].norm()),
            "rigid_projection_residual": (float("nan") if proj is None
                                          else float(proj["residual"])),
            "q_final": float(q_traj[part, -1]),
            "speed_rms": float(speed_final[part].square().mean().sqrt()),
            "median_transported_displacement": float(
                (positions[-1] - xyz0)[transport_mask[part]].norm(dim=-1).median())
            if bool(transport_mask[part].any()) else 0.0,
            **({} if part not in final_pcd_report else {
                "final_pcd_forward": final_pcd_report[part]["forward"],
                "final_pcd_backward": final_pcd_report[part]["backward"]}),
            **({} if part not in alpha_final else
               {f"alpha_vs_s_{k}": v for k, v in
                alpha_scalar_agreement(torch.tensor(alpha_final[part]), s_mid.cpu()).items()}),
            **sign_flip_diagnostics(speed_final[part]),
            **({} if int(idx.numel()) == 0 else observability(pts, w)),
        }
        diagnostics["parts"].append(entry)
        printer(f"  part {part + 1}: transport {entry['transport_support']} "
                f"fitting {entry['fitting_support']}  rot {rot_deg:.3f} deg  "
                f"|t|/bbox {entry['endpoint_translation_over_bbox']*100:.3f}%  "
                f"|w| {entry['omega_star_norm']:.4g} |u| {entry['u_star_norm']:.4g}  "
                f"rigidres {entry['rigid_projection_residual']:.3g}  "
                f"cond {entry.get('condition_number', float('nan')):.3e} "
                f"rank {entry.get('rank', -1)}")
    if grad_audit:
        last = grad_audit[-1]
        printer("  grad|geometry " + " ".join(f"{k}={v:.3g}" for k, v in last["geometry"].items()))
        printer("  grad|scalar   " + " ".join(f"{k}={v:.3g}" for k, v in last["scalar"].items()))

    out.mkdir(parents=True, exist_ok=True)
    # §48: the three diagnostic artifacts this spec requires per outer.
    (out / "loss_gradient_diagnostics.json").write_text(json.dumps(_jsonable_report({
        "outer": int(outer_iter),
        "note": "norms are of lambda*L (CODEX_DEBUG §14); probes use autograd.grad "
                "and never touch .grad (§17)",
        "rounds": [{k: v for k, v in row.items()
                    if k in ("round", "geometry", "scalar", "geometry_raw_loss",
                             "geometry_weighted_loss", "scalar_raw_loss",
                             "scalar_weighted_loss", "lambdas", "geometry_ratios",
                             "scalar_ratios")}
                   for row in grad_audit]}), indent=2))
    (out / "twist_gradient_diagnostics.json").write_text(json.dumps(_jsonable_report({
        "outer": int(outer_iter),
        "rounds": [{"round": row["round"], "parts": row.get("twist_gradients", {})}
                   for row in grad_audit]}), indent=2))
    (out / "twist_geometry_diagnostics.json").write_text(json.dumps(_jsonable_report({
        "outer": int(outer_iter),
        "rounds": [{"round": row["round"], "parts": row.get("twist_geometry", {})}
                   for row in grad_audit],
        "final": _twist_geometry(final_projections)}), indent=2))

    # §7: the exact object handed to the support builder, saved verbatim.
    _atomic({"schema_version": 1, "P": support_probs.detach().cpu(), "z": z.detach().cpu(),
             "dynamic_confidence": support.dynamic_confidence.detach().cpu(),
             "max_part_confidence": (support.part_confidence.max(dim=1).values.detach().cpu()
                                     if int(support.part_confidence.shape[1]) > 0
                                     else torch.zeros(n)),
             "source_name": soft_part_probs_name},
            out / "soft_probability_handoff.pt")
    out.mkdir(parents=True, exist_ok=True)
    (out / "soft_probability_handoff.json").write_text(json.dumps(_jsonable_report({
        "outer": int(outer_iter), "source_name": soft_part_probs_name,
        "shape": list(support_probs.shape), "dtype": str(support_probs.dtype),
        **handoff}), indent=2))

    # §62: support artifacts, in all three required forms.
    _atomic({"schema_version": 4,
             **{name: getattr(support, name).detach().cpu() for name in (
                 "transport_masks", "fitting_masks", "transport_part_id", "fitting_part_id",
                 "candidate_mask", "outlier_mask", "core_mask", "attached_mask",
                 "unassigned_mask", "fit_weight", "dynamic_confidence", "part_confidence",
                 "local_spacing", "local_dynamic_support")},
             "effective_thresholds": support.effective_thresholds,
             "diagnostics": support.diagnostics,
             "gt_metrics": support_gt}, out / "inner_support_cleaning.pt")
    out.mkdir(parents=True, exist_ok=True)
    (out / "inner_support_cleaning.json").write_text(json.dumps(_jsonable_report({
        "outer": int(outer_iter),
        "counts": {name: int(getattr(support, name).sum()) for name in (
            "candidate_mask", "outlier_mask", "core_mask", "attached_mask", "unassigned_mask")},
        "candidate_indices": torch.nonzero(support.candidate_mask).reshape(-1)[:20000],
        "outlier_indices": torch.nonzero(support.outlier_mask).reshape(-1)[:20000],
        "core_indices": torch.nonzero(support.core_mask).reshape(-1)[:20000],
        "attached_indices": torch.nonzero(support.attached_mask).reshape(-1)[:20000],
        "unassigned_indices": torch.nonzero(support.unassigned_mask).reshape(-1)[:20000],
        "effective_thresholds": support.effective_thresholds,
        "cluster_to_part_matrix": support.diagnostics.get("cluster_to_part_matrix"),
        "diagnostics": support.diagnostics,
        "gt_metrics": support_gt,
    }), indent=2))
    try:
        write_support_debug_ply(support, xyz0, out / "inner_support_debug.ply")
    except Exception as exc:                      # a debug file must never fail a run
        printer(f"  (support debug PLY not written: {exc})")

    _atomic({"schema_version": 4, "centers": centers.detach().cpu(),
             "control": control.detach().cpu(), "kernel_scale": sigma.detach().cpu(),
             "xi_star": [None if p is None else p["xi"].detach().cpu() for p in final_projections],
             "omega_star": [None if p is None else p["omega"].detach().cpu() for p in final_projections],
             "u_star": [None if p is None else p["u"].detach().cpu() for p in final_projections],
             "valid_motion": valid_motion, "xi_star_is_derived": True,
             "alpha_star": alpha_final, "alpha_bootstrap": alpha_bootstrap,
             "joint_residual_per_point": joint_residual,
             "frame_joint_loss_per_frame": frame_joint,
             "seeded_from_interval": seeded_from, "anchored_rms": anchored,
             "geometry_history": geometry_log}, out / "inner_killing_spatial.pt")
    _atomic({"schema_version": 3, "speed": speed_final.cpu(), "q": q_traj.cpu(),
             "s_mid": (0.5 * (speed_final[:, :-1] + speed_final[:, 1:])).cpu()
             if frames > 1 else speed_final.cpu(),
             "dt": dt.cpu(), "times": times.cpu(),
             "scalar_history": scalar_log, "history": history,
             "grad_audit": grad_audit}, out / "inner_scalar_trajectory.pt")

    report = {
        "outer": int(outer_iter),
        "loss_routing": {k: list(v) for k, v in LOSS_ROUTING.items()},
        "frame_coverage": diagnostics["frame_coverage"],
        "support": {"summary": format_support_summary(support),
                    "thresholds": support.effective_thresholds,
                    "diagnostics": support.diagnostics,
                    "gt": support_gt},
        "gradient_audit": grad_audit[-1] if grad_audit else {},
        "parts": [],
    }
    for part in range(moving_parts):
        entry = diagnostics["parts"][part]
        proj = final_projections[part]
        s_mid = (0.5 * (speed_final[part][:-1] + speed_final[part][1:])
                 if frames > 1 else speed_final[part])
        report["parts"].append({
            "part": part + 1,
            "valid_motion": bool(valid_motion[part]),
            "geometry": {                                                     # §65
                "xi": [] if proj is None else proj["xi"].detach().cpu().tolist(),
                "omega": [] if proj is None else proj["omega"].detach().cpu().tolist(),
                "u": [] if proj is None else proj["u"].detach().cpu().tolist(),
                "rigid_projection_residual": entry.get("rigid_projection_residual"),
                "strain": geometry_log[-1].get("strain") if geometry_log else None,
                "L_render_G": geometry_log[-1].get("render") if geometry_log else None,
                "L_depth_G": geometry_log[-1].get("depth") if geometry_log else None,
                "L_motion_dir": geometry_log[-1].get("motion_direction") if geometry_log else None,
                "L_final_PCD_G": geometry_log[-1].get("final_pcd") if geometry_log else None,
                "L_rigid": geometry_log[-1].get("rigid") if geometry_log else None,
                "gradient_norms": grad_audit[-1]["geometry"] if grad_audit else {},
                "condition_number": entry.get("condition_number"),
                "rank": entry.get("rank"),
                "support_covariance_eigenvalues": entry.get("eigenvalues"),
                "alpha_star": alpha_final.get(part, []),
                "frame_joint_loss_per_frame": frame_joint.get(part, []),
            },
            "scalar": {                                                       # §66
                "s": speed_final[part].cpu().tolist(),
                "s_mid": s_mid.cpu().tolist(),
                "q": q_traj[part].cpu().tolist(),
                "L_render_S": scalar_log[-1].get("render") if scalar_log else None,
                "L_depth_S": scalar_log[-1].get("depth") if scalar_log else None,
                "L_motion_mag": scalar_log[-1].get("motion_magnitude") if scalar_log else None,
                "L_final_PCD_S": scalar_log[-1].get("final_pcd") if scalar_log else None,
                "L_cont": scalar_log[-1].get("continuity") if scalar_log else None,
                "L_flip": scalar_log[-1].get("flip") if scalar_log else None,
                "L_jerk": scalar_log[-1].get("jerk") if scalar_log else None,
                "gradient_norms": grad_audit[-1]["scalar"] if grad_audit else {},
                "endpoint_rotation_deg": entry.get("endpoint_rotation_deg"),
                "endpoint_translation_over_bbox": entry.get("endpoint_translation_over_bbox"),
                "median_displacement": entry.get("median_transported_displacement"),
                "corr_alpha_s_mid": entry.get("alpha_vs_s_corr"),
                "sign_agreement_alpha_s_mid": entry.get("alpha_vs_s_sign_agreement"),
            },
            "final_pcd": {
                "q_final": entry.get("q_final"),
                "trimmed_chamfer_forward": entry.get("final_pcd_forward"),
                "trimmed_chamfer_backward": entry.get("final_pcd_backward"),
            },
        })
    (out / "inner_decoupled_report.json").write_text(json.dumps(_jsonable_report(report), indent=2))
    (out / "inner_motion_diagnostics.json").write_text(
        json.dumps(_jsonable_report(diagnostics), indent=2))
    (out / "inner_observability.json").write_text(json.dumps(_jsonable_report(
        {"parts": [{k: v for k, v in p.items() if k in
                    ("part", "eigenvalues", "condition_number", "rank",
                     "min_eigenvalue", "max_eigenvalue", "effective_points")}
                   for p in diagnostics["parts"]]}), indent=2))

    import csv as _csv

    with (out / "inner_scalar_trajectory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=[
            "part", "frame", "speed", "s_mid", "q", "alpha_star", "frame_joint_loss"])
        writer.writeheader()
        writer.writerows([{
            "part": p + 1, "frame": t, "speed": float(speed_final[p, t]),
            "s_mid": (float(0.5 * (speed_final[p, t] + speed_final[p, t + 1]))
                      if t + 1 < frames else ""),
            "q": float(q_traj[p, t]),
            "alpha_star": (float(alpha_final[p][t]) if p in alpha_final
                           and t < len(alpha_final[p]) else ""),
            "frame_joint_loss": (float(frame_joint[p][t]) if p in frame_joint
                                 and t < len(frame_joint[p]) else ""),
        } for p in range(moving_parts) for t in range(frames)])

    # ---------------------------------------------------------- final state
    omega_seq = torch.zeros((max(frames - 1, 0), moving_parts + 1, 3), device=device, dtype=dtype)
    b_seq = torch.zeros_like(omega_seq)
    for part in range(moving_parts):
        proj = final_projections[part]
        if proj is None or frames <= 1:
            continue
        s_mid = 0.5 * (speed_final[part][:-1] + speed_final[part][1:])
        omega_seq[:, part + 1] = proj["omega"].detach()[None] * s_mid[:, None]
        b_seq[:, part + 1] = proj["u"].detach()[None] * s_mid[:, None]

    return NestedKillingMotionState(
        times=times.detach().cpu(),
        positions=torch.stack(positions).detach().cpu(),
        rotations=torch.stack(rotations).detach().cpu(),
        part_rotation=stack_r, part_translation=stack_t, part_quaternion=stack_q,
        omega=omega_seq.detach().cpu(), b=b_seq.detach().cpu(),
        center_velocity=b_seq.detach().cpu(),
        strain=torch.zeros(1),
        motion_separation=torch.zeros((moving_parts + 1, moving_parts + 1)),
        history=[{"outer": int(outer_iter), **h} for h in history],
        field_debug=[{"outer": int(outer_iter), "centers": centers.detach().cpu(),
                      "control_velocity": control.detach().cpu(),
                      "kernel_scale": sigma.detach().cpu()}],
        moving_endpoints=torch.stack([
            xyz0 @ transforms[p, -1, :3, :3].transpose(0, 1) + transforms[p, -1, :3, 3]
            for p in range(moving_parts)], dim=0).detach(),
        moving_endpoint_quaternions=torch.stack([
            apply_rotation_to_gaussian_quaternions(
                rot0, normalize_quaternion(matrix_to_quaternion(transforms[p, -1, :3, :3][None])))
            for p in range(moving_parts)], dim=0).detach(),
        acceptance={"shared_field": 1.0,
                    "valid_motion": float(sum(1 for v in valid_motion if v)),
                    "mean_strain_norm": 0.0,
                    "fitting_support": float(sum(int(m.sum()) for m in support.fitting_masks))},
        kinetic_energy=torch.zeros((n, moving_parts), device=device, dtype=dtype),
        inferred_speed=torch.zeros(n, device=device, dtype=dtype),
    )
