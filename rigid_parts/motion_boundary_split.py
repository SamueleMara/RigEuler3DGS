"""CODEX_motion_boundary_3dgs_split -- fix the representation, not the labels.

Measured on USB: the falsely-dynamic Gaussians are not a label-optimization
failure. They are large, diffuse primitives whose ellipsoid physically covers
volume on BOTH sides of a rigid-motion boundary -- 3x the extent of a typical
static Gaussian, 83% of them reaching into the moving part with their 2-sigma
support. Excluding them, precision was 99.94% instead of 99.64%. For such a
primitive no single part label can be correct:

    one Gaussian  !<->  one rigid part

This module replaces each such parent with two smaller children placed on
opposite sides of the locally estimated motion boundary, with complementary soft
ownership. It is a NEW algorithm: selection comes from the soft ownership field,
placement is deterministic along the boundary normal, and no photometric
densification criterion or random covariance sampling is involved (spec
"CRITICAL IMPLEMENTATION RULE", sections 3 and 20). Only the low-level
tensor/optimizer mutation is reused from the repository's existing
``densification_postfix`` / ``prune_points`` machinery, which is
algorithm-agnostic.

Ground truth never influences selection, direction, placement or labels (§51).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .neighborhood import NeighborGraph, build_neighbor_graph

EPS = 1.0e-8


# ---------------------------------------------------------------------------
# Section 33: data structures
# ---------------------------------------------------------------------------


@dataclass
class BoundarySplitCandidate:
    """One parent that physically straddles a local rigid-part boundary."""

    parent_index: int
    class_a: int
    class_b: int
    normal: torch.Tensor          # [3] unit, +n points toward class b
    sigma_normal: float           # sqrt(n^T Sigma n)
    omega_a: float
    omega_b: float
    distance_to_b: float
    score: float


@dataclass
class BoundarySplitResult:
    """What the split actually did (section 33)."""

    num_parents: int
    num_children: int
    parent_indices: torch.Tensor
    child_a_indices: torch.Tensor
    child_b_indices: torch.Tensor
    split_group_id: torch.Tensor     # [N_new] long, -1 where not a split child
    split_parent_id: torch.Tensor    # [N_new] long, -1 where not a split child
    split_child_role: torch.Tensor   # [N_new] long, -1 / 0 / 1
    soft_part_probs: torch.Tensor    # [N_new, K+1]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Section 6: local class distribution
# ---------------------------------------------------------------------------


@torch.no_grad()
def local_class_distribution(
    soft_probs: torch.Tensor,
    graph: NeighborGraph,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """P_bar_i = sum_j w_ij P_j / (sum_j w_ij + eps), neighbours only (§6).

    Neighbours only -- not including self -- so the statistic answers "what do
    the points AROUND me look like", which is what a boundary test needs. A
    confidently-labelled parent must still be detectable (§5), and including its
    own vote would mask exactly that case.
    """
    p = torch.as_tensor(soft_probs)
    n = int(graph.num_nodes)
    if graph.num_edges == 0:
        return p.clone()
    src = torch.cat((graph.source, graph.target)).to(p.device)
    dst = torch.cat((graph.target, graph.source)).to(p.device)
    w = torch.cat((graph.weight, graph.weight)).to(device=p.device, dtype=p.dtype)

    numerator = torch.zeros_like(p)
    numerator.index_add_(0, src, w[:, None] * p.index_select(0, dst))
    degree = torch.zeros(n, dtype=p.dtype, device=p.device).index_add_(0, src, w)
    out = numerator / (degree[:, None] + float(eps))
    isolated = degree <= torch.finfo(p.dtype).tiny
    if bool(isolated.any()):
        out = torch.where(isolated[:, None], p, out)
    return out


# ---------------------------------------------------------------------------
# Sections 7, 8: boundary-normal estimation
# ---------------------------------------------------------------------------


@torch.no_grad()
def estimate_local_boundary_normal(
    index: int,
    xyz: torch.Tensor,
    soft_probs: torch.Tensor,
    neighbours: torch.Tensor,
    weights: torch.Tensor,
    class_a: int,
    class_b: int,
    *,
    ridge: float = 1.0e-6,
    confidence: float = 0.5,
    eps: float = EPS,
) -> torch.Tensor | None:
    """Section 7's weighted least-squares log-odds gradient, with §8's fallback.

        s_j = log(P_jb + eps) - log(P_ja + eps)
        g   = argmin_g sum_j w_ij [(s_j - s_i) - g^T (mu_j - mu_i)]^2 + ridge ||g||^2
        n   = g / ||g||          (+n -> increasing ownership of class b)

    Returns ``None`` when no stable direction exists, and that parent is then not
    split (§8) -- a random direction is explicitly forbidden.
    """
    if int(neighbours.numel()) < 3:
        return None
    mu = xyz[index]
    dmu = xyz.index_select(0, neighbours) - mu                       # [M, 3]
    p = soft_probs
    s = (torch.log(p[:, class_b] + float(eps)) - torch.log(p[:, class_a] + float(eps)))
    ds = s.index_select(0, neighbours) - s[index]                    # [M]
    w = weights.to(dmu.dtype)

    a = (w[:, None, None] * dmu[:, :, None] * dmu[:, None, :]).sum(dim=0)
    rhs = (w[:, None] * ds[:, None] * dmu).sum(dim=0)
    trace = float(torch.diagonal(a).sum())
    damping = float(ridge) * (abs(trace) / 3.0 + 1.0)
    eye = torch.eye(3, dtype=a.dtype, device=a.device)
    try:
        g = torch.linalg.solve(a + damping * eye, rhs)
    except Exception:
        g = None
    if g is not None and bool(torch.isfinite(g).all()) and float(g.norm()) > 1.0e-12:
        return g / g.norm()

    # Section 8 fallback: the direction between confident a and b neighbours.
    pa = p.index_select(0, neighbours)[:, class_a]
    pb = p.index_select(0, neighbours)[:, class_b]
    conf_a = neighbours[pa > float(confidence)]
    conf_b = neighbours[pb > float(confidence)]
    if int(conf_a.numel()) == 0 or int(conf_b.numel()) == 0:
        return None
    direction = xyz.index_select(0, conf_b).mean(0) - xyz.index_select(0, conf_a).mean(0)
    if float(direction.norm()) <= 1.0e-12:
        return None
    return direction / direction.norm()


# ---------------------------------------------------------------------------
# Sections 5, 9, 31, 32: candidate detection
# ---------------------------------------------------------------------------


@torch.no_grad()
def _nearest_confident_distance(
    query_xyz: torch.Tensor,
    reference_xyz: torch.Tensor,
    *,
    chunk: int = 4096,
) -> torch.Tensor:
    """Distance from each query to the nearest reference point, chunked."""
    if int(reference_xyz.shape[0]) == 0:
        return torch.full((int(query_xyz.shape[0]),), float("inf"),
                          device=query_xyz.device, dtype=query_xyz.dtype)
    out = []
    for start in range(0, int(query_xyz.shape[0]), max(1, int(chunk))):
        block = query_xyz[start:start + max(1, int(chunk))]
        out.append(torch.cdist(block, reference_xyz).min(dim=1).values)
    return torch.cat(out, dim=0)


@torch.no_grad()
def find_motion_boundary_split_candidates(
    *,
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    soft_probs: torch.Tensor,
    graph: NeighborGraph,
    trusted_static_mask: torch.Tensor | None = None,
    local_class_min: float = 0.15,
    extent_sigma: float = 2.0,
    size_threshold: float = 0.0,
    percent_dense: float = 0.01,
    scene_extent: float = 1.0,
    gradient_ridge: float = 1.0e-6,
    confident_class: float = 0.6,
    max_fraction: float = 0.01,
    max_count: int = 0,
) -> tuple[list[BoundarySplitCandidate], dict[str, Any]]:
    """Sections 5, 6, 9, 31, 32.

    Selection is deliberately NOT entropy-driven (§5): a representation-error
    primitive can be at P = [0.003, 0.997] and still span both regions. The test
    is large ellipsoid + locally competing classes + physical reach across the
    boundary.
    """
    device = xyz.device
    n = int(xyz.shape[0])
    k1 = int(soft_probs.shape[1])
    p_bar = local_class_distribution(soft_probs, graph)

    # --- local top-two classes with nontrivial support (§6) ----------------
    top2 = p_bar.topk(2, dim=1)
    class_a = top2.indices[:, 0]
    class_b = top2.indices[:, 1]
    m_a, m_b = top2.values[:, 0], top2.values[:, 1]
    competing = (m_a > float(local_class_min)) & (m_b > float(local_class_min))

    # --- large ellipsoid (§9) ---------------------------------------------
    largest = scaling.max(dim=1).values
    threshold = (float(size_threshold) if float(size_threshold) > 0.0
                 else float(percent_dense) * float(scene_extent))
    large = largest > threshold

    eligible = competing & large
    if trusted_static_mask is not None:
        # §23: a trusted static parent is never split in this first pass.
        trusted = torch.as_tensor(trusted_static_mask, device=device).reshape(-1).bool()
        eligible = eligible & ~trusted
    else:
        trusted = torch.zeros(n, dtype=torch.bool, device=device)

    diagnostics: dict[str, Any] = {
        "num_gaussians": n,
        "num_competing": int(competing.sum()),
        "num_large": int(large.sum()),
        "num_eligible_before_reach": int(eligible.sum()),
        "size_threshold": threshold,
        "num_trusted_skipped": int((competing & large & trusted).sum()),
    }
    if not bool(eligible.any()):
        diagnostics["num_selected"] = 0
        return [], diagnostics

    # --- distance to a high-confidence region of each class ----------------
    confident_sets = {
        c: xyz[(soft_probs[:, c] > float(confident_class))] for c in range(k1)
    }

    # --- covariance along the normal, and the physical reach test (§9) -----
    from utils.general_utils import build_rotation

    src_all = torch.cat((graph.source, graph.target)).to(device)
    dst_all = torch.cat((graph.target, graph.source)).to(device)
    w_all = torch.cat((graph.weight, graph.weight)).to(device=device, dtype=xyz.dtype)
    order = torch.argsort(src_all)
    src_sorted, dst_sorted, w_sorted = src_all[order], dst_all[order], w_all[order]
    counts = torch.bincount(src_sorted, minlength=n)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)))

    candidates: list[BoundarySplitCandidate] = []
    num_no_normal = 0
    num_no_reach = 0
    for i in torch.nonzero(eligible, as_tuple=False).reshape(-1).tolist():
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if hi <= lo:
            continue
        nb, wt = dst_sorted[lo:hi], w_sorted[lo:hi]
        a_i, b_i = int(class_a[i]), int(class_b[i])

        normal = estimate_local_boundary_normal(
            i, xyz, soft_probs, nb, wt, a_i, b_i,
            ridge=gradient_ridge, confidence=0.5)
        if normal is None:
            num_no_normal += 1
            continue

        # sigma_n = sqrt(n^T Sigma n) with Sigma = R diag(s^2) R^T
        rot = build_rotation(torch.nn.functional.normalize(rotation[i : i + 1], dim=-1))[0]
        local = rot.transpose(0, 1) @ normal
        sigma_n = float(torch.sqrt(((local * scaling[i]) ** 2).sum()).clamp_min(1.0e-12))

        reference = confident_sets.get(b_i)
        if reference is None or int(reference.shape[0]) == 0:
            num_no_reach += 1
            continue
        d_b = float(_nearest_confident_distance(xyz[i : i + 1], reference)[0])
        if not (float(extent_sigma) * sigma_n >= d_b):
            num_no_reach += 1
            continue

        denominator = float(m_a[i] + m_b[i]) + EPS
        omega_a = float(m_a[i]) / denominator
        omega_b = float(m_b[i]) / denominator
        # §31: score = size * mixture * reach, ordering only.
        s_size = float(largest[i]) / max(threshold, 1.0e-12)
        s_mix = 4.0 * omega_a * omega_b
        s_reach = float(extent_sigma) * sigma_n / max(d_b, 1.0e-12)
        candidates.append(BoundarySplitCandidate(
            parent_index=i, class_a=a_i, class_b=b_i, normal=normal.detach().clone(),
            sigma_normal=sigma_n, omega_a=omega_a, omega_b=omega_b,
            distance_to_b=d_b, score=s_size * s_mix * s_reach))

    candidates.sort(key=lambda c: -c.score)
    cap = max(0, int(float(max_fraction) * n))
    if int(max_count) > 0:
        cap = min(cap, int(max_count))
    trimmed = candidates[:cap]
    if candidates and not trimmed:
        # §32's fraction floors to zero on small clouds (1% of 81 is 0). That is
        # the cap behaving as specified, but silently splitting nothing while
        # candidates exist is exactly the kind of no-op that reads as "the
        # detector found nothing", so say so.
        print(f"[Boundary split] WARNING: {len(candidates)} candidate(s) found but the "
              f"cap is 0 (max_fraction={max_fraction} of N={n}); nothing will be split")

    diagnostics.update({
        "num_no_stable_normal": num_no_normal,
        "num_failed_reach_test": num_no_reach,
        "num_candidates": len(candidates),
        "num_selected": len(trimmed),
        "cap": cap,
        "score_max": float(trimmed[0].score) if trimmed else 0.0,
        "score_min": float(trimmed[-1].score) if trimmed else 0.0,
    })
    return trimmed, diagnostics


# ---------------------------------------------------------------------------
# Sections 10-16, 18, 35-37: child construction
# ---------------------------------------------------------------------------


@dataclass
class BoundarySplitChildren:
    """The child tensors, before they touch the model."""

    xyz_a: torch.Tensor
    xyz_b: torch.Tensor
    scaling_internal: torch.Tensor    # [2M, 3], already inverse-activated
    rotation: torch.Tensor            # [2M, 4]
    opacity_internal: torch.Tensor    # [2M, 1], already inverse-activated
    probs_a: torch.Tensor
    probs_b: torch.Tensor
    parent_indices: torch.Tensor


@torch.no_grad()
def build_boundary_split_children(
    candidates: list[BoundarySplitCandidate],
    *,
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    soft_probs: torch.Tensor,
    scaling_inverse_activation,
    inverse_opacity_activation,
    offset_sigma: float = 0.5,
    max_offset_sigma: float = 1.0,
    label_mix: float = 0.05,
    opacity_mode: str = "weighted_mass",
    num_children: int = 2,
) -> BoundarySplitChildren:
    """Sections 11-16, 18. Deterministic, boundary-aware, never random (§12)."""
    device = xyz.device
    dtype = xyz.dtype
    k1 = int(soft_probs.shape[1])
    idx = torch.tensor([c.parent_index for c in candidates], dtype=torch.long, device=device)
    normal = torch.stack([c.normal.to(device=device, dtype=dtype) for c in candidates])
    sigma_n = torch.tensor([c.sigma_normal for c in candidates], dtype=dtype, device=device)
    omega_a = torch.tensor([c.omega_a for c in candidates], dtype=dtype, device=device)
    omega_b = torch.tensor([c.omega_b for c in candidates], dtype=dtype, device=device)

    # §11: mu_a = mu - r w_b n, mu_b = mu + r w_a n, so w_a mu_a + w_b mu_b = mu.
    r = float(offset_sigma) * sigma_n
    cap = float(max_offset_sigma) * sigma_n
    delta_a = torch.minimum(r * omega_b, cap)
    delta_b = torch.minimum(r * omega_a, cap)
    parent_xyz = xyz.index_select(0, idx)
    xyz_a = parent_xyz - delta_a[:, None] * normal
    xyz_b = parent_xyz + delta_b[:, None] * normal

    # §13: the original activated-scale rule, s / (0.8 N).
    parent_scale = scaling.index_select(0, idx)
    child_scale = parent_scale / (0.8 * float(num_children))
    scaling_internal = scaling_inverse_activation(
        torch.cat((child_scale, child_scale), dim=0))

    # §14: rotation copied exactly.
    parent_rotation = rotation.index_select(0, idx)
    rotation_out = torch.cat((parent_rotation, parent_rotation), dim=0)

    # §16: opacity is SPLIT by ownership mass, in activated space (§37).
    parent_alpha = opacity.index_select(0, idx).reshape(-1)
    if str(opacity_mode) == "original_copy":
        alpha_a = parent_alpha
        alpha_b = parent_alpha
    else:
        alpha_a = (omega_a * parent_alpha).clamp(1.0e-6, 1.0 - 1.0e-6)
        alpha_b = (omega_b * parent_alpha).clamp(1.0e-6, 1.0 - 1.0e-6)
    opacity_internal = inverse_opacity_activation(
        torch.cat((alpha_a, alpha_b), dim=0)[:, None])

    # §18: complementary SOFT ownership priors -- never the same vector.
    parent_probs = soft_probs.index_select(0, idx)
    mix = float(label_mix)
    e_a = torch.zeros((len(candidates), k1), dtype=parent_probs.dtype, device=device)
    e_b = torch.zeros_like(e_a)
    for row, c in enumerate(candidates):
        e_a[row, c.class_a] = 1.0
        e_b[row, c.class_b] = 1.0
    probs_a = (1.0 - mix) * e_a + mix * parent_probs
    probs_b = (1.0 - mix) * e_b + mix * parent_probs
    probs_a = probs_a / probs_a.sum(dim=1, keepdim=True).clamp_min(EPS)
    probs_b = probs_b / probs_b.sum(dim=1, keepdim=True).clamp_min(EPS)

    return BoundarySplitChildren(
        xyz_a=xyz_a, xyz_b=xyz_b, scaling_internal=scaling_internal,
        rotation=rotation_out, opacity_internal=opacity_internal,
        probs_a=probs_a, probs_b=probs_b, parent_indices=idx)


# ---------------------------------------------------------------------------
# Sections 19, 20, 24, 25, 34: apply the split to the model
# ---------------------------------------------------------------------------


@torch.no_grad()
def apply_motion_boundary_split(
    gaussians,
    candidates: list[BoundarySplitCandidate],
    soft_probs: torch.Tensor,
    *,
    offset_sigma: float = 0.5,
    max_offset_sigma: float = 1.0,
    label_mix: float = 0.05,
    opacity_mode: str = "weighted_mass",
    extra_tensors: dict[str, torch.Tensor] | None = None,
) -> BoundarySplitResult:
    """1 parent -> 2 children, then prune the parent (§19, §25).

    Uses the repository's ``densification_postfix`` / ``prune_points`` ONLY as
    algorithm-agnostic tensor+optimizer infrastructure (§20). It never calls
    ``densify_and_split`` / ``densify_and_clone``: what to split and what the
    children are is decided entirely above.

    ``extra_tensors`` are caller-owned per-Gaussian arrays (masks, memory, ...)
    that must follow the identical append-then-prune ordering (§24, §25); the
    rebuilt versions are returned in ``diagnostics['extra']``.
    """
    device = gaussians.get_xyz.device
    n_old = int(gaussians.get_xyz.shape[0])
    probs = torch.as_tensor(soft_probs, device=device)
    if int(probs.shape[0]) != n_old:
        raise ValueError(f"soft_probs has {int(probs.shape[0])} rows for {n_old} gaussians")
    if not candidates:
        return BoundarySplitResult(
            num_parents=0, num_children=0,
            parent_indices=torch.zeros(0, dtype=torch.long, device=device),
            child_a_indices=torch.zeros(0, dtype=torch.long, device=device),
            child_b_indices=torch.zeros(0, dtype=torch.long, device=device),
            split_group_id=torch.full((n_old,), -1, dtype=torch.long, device=device),
            split_parent_id=torch.full((n_old,), -1, dtype=torch.long, device=device),
            split_child_role=torch.full((n_old,), -1, dtype=torch.long, device=device),
            soft_part_probs=probs.clone(),
            diagnostics={"num_selected": 0, "extra": dict(extra_tensors or {})})

    children = build_boundary_split_children(
        candidates,
        xyz=gaussians.get_xyz, scaling=gaussians.get_scaling,
        rotation=gaussians._rotation, opacity=gaussians.get_opacity,
        soft_probs=probs,
        scaling_inverse_activation=gaussians.scaling_inverse_activation,
        inverse_opacity_activation=gaussians.inverse_opacity_activation,
        offset_sigma=offset_sigma, max_offset_sigma=max_offset_sigma,
        label_mix=label_mix, opacity_mode=opacity_mode)

    parent_idx = children.parent_indices
    m = int(parent_idx.numel())

    # The infrastructure copies auxiliary per-Gaussian fields (semantic, DINO,
    # SAM3, gt_part) from a selected mask repeated `rep` times. Child a then
    # child b is exactly `sel` repeated twice, which matches our stacking order.
    selected_mask = torch.zeros(n_old, dtype=torch.bool, device=device)
    selected_mask[parent_idx] = True
    if int(selected_mask.sum()) != m:
        raise RuntimeError("duplicate parent indices in the split candidate list")
    # densification_postfix indexes with a boolean mask, which yields ascending
    # order; our per-candidate stacking must match it or children and auxiliary
    # fields would be mismatched.
    ascending = torch.argsort(parent_idx)
    xyz_a = children.xyz_a[ascending]
    xyz_b = children.xyz_b[ascending]
    probs_a = children.probs_a[ascending]
    probs_b = children.probs_b[ascending]
    scale_sorted = torch.cat((children.scaling_internal[:m][ascending],
                              children.scaling_internal[m:][ascending]), dim=0)
    rot_sorted = torch.cat((children.rotation[:m][ascending],
                            children.rotation[m:][ascending]), dim=0)
    opacity_sorted = torch.cat((children.opacity_internal[:m][ascending],
                                children.opacity_internal[m:][ascending]), dim=0)

    new_xyz = torch.cat((xyz_a, xyz_b), dim=0)
    parent_sorted = parent_idx[ascending]
    features_dc = gaussians._features_dc.index_select(0, parent_sorted)
    features_rest = gaussians._features_rest.index_select(0, parent_sorted)
    new_features_dc = torch.cat((features_dc, features_dc), dim=0)
    new_features_rest = torch.cat((features_rest, features_rest), dim=0)
    # ``tmp_radii`` is only populated once photometric densification has run, and
    # this split can fire in a run where it never did. Both densification_postfix
    # and prune_points index it, so materialise it rather than crash.
    existing = getattr(gaussians, "tmp_radii", None)
    if existing is None or not torch.is_tensor(existing) or int(existing.shape[0]) != n_old:
        gaussians.tmp_radii = torch.zeros(n_old, device=device, dtype=gaussians.get_xyz.dtype)
    tmp_radii = gaussians.tmp_radii.index_select(0, parent_sorted)
    new_tmp_radii = torch.cat((tmp_radii, tmp_radii), dim=0)

    gaussians._last_densify_selected_mask = selected_mask
    gaussians._last_densify_repeat = 2
    try:
        gaussians.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            opacity_sorted, scale_sorted, rot_sorted, new_tmp_radii)
    finally:
        gaussians._last_densify_selected_mask = None
        gaussians._last_densify_repeat = 1

    n_appended = int(gaussians.get_xyz.shape[0])
    if n_appended != n_old + 2 * m:
        raise RuntimeError(f"append produced {n_appended}, expected {n_old + 2 * m}")

    # --- extend every caller-owned per-Gaussian tensor identically (§24) ----
    group = torch.arange(m, dtype=torch.long, device=device)
    appended_probs = torch.cat((probs, probs_a, probs_b), dim=0)
    appended_group = torch.cat((
        torch.full((n_old,), -1, dtype=torch.long, device=device), group, group))
    appended_parent = torch.cat((
        torch.full((n_old,), -1, dtype=torch.long, device=device),
        parent_sorted, parent_sorted))
    appended_role = torch.cat((
        torch.full((n_old,), -1, dtype=torch.long, device=device),
        torch.zeros(m, dtype=torch.long, device=device),
        torch.ones(m, dtype=torch.long, device=device)))

    appended_extra: dict[str, torch.Tensor] = {}
    for name, tensor in (extra_tensors or {}).items():
        t = torch.as_tensor(tensor, device=device)
        if int(t.shape[0]) != n_old:
            raise ValueError(f"extra tensor {name} has {int(t.shape[0])} rows for {n_old}")
        inherited = t.index_select(0, parent_sorted)
        if name == "memory":
            # §22: child memories are exactly zero, never the parent's.
            inherited = torch.zeros_like(inherited)
        appended_extra[name] = torch.cat((t, inherited, inherited), dim=0)

    # --- prune the parents (§19) -------------------------------------------
    prune_mask = torch.zeros(n_appended, dtype=torch.bool, device=device)
    prune_mask[parent_sorted] = True
    gaussians.prune_points(prune_mask)

    keep = ~prune_mask
    n_new = int(gaussians.get_xyz.shape[0])
    if n_new != n_old + m:
        raise RuntimeError(f"§25 violated: N_new={n_new}, expected {n_old + m}")

    final_probs = appended_probs[keep]
    final_group = appended_group[keep]
    final_parent = appended_parent[keep]
    final_role = appended_role[keep]
    final_extra = {k: v[keep] for k, v in appended_extra.items()}

    # --- §24's explicit shape assertions ------------------------------------
    for name, tensor in (("soft_part_probs", final_probs), ("split_group_id", final_group),
                         ("split_parent_id", final_parent), ("split_child_role", final_role),
                         *final_extra.items()):
        if int(tensor.shape[0]) != n_new:
            raise RuntimeError(f"{name} has {int(tensor.shape[0])} rows, expected {n_new}")
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        t = getattr(gaussians, name)
        if int(t.shape[0]) != n_new:
            raise RuntimeError(f"gaussians.{name} has {int(t.shape[0])} rows, expected {n_new}")

    child_a = torch.nonzero(final_role == 0, as_tuple=False).reshape(-1)
    child_b = torch.nonzero(final_role == 1, as_tuple=False).reshape(-1)

    records = []
    for row, c in enumerate(sorted(candidates, key=lambda x: x.parent_index)):
        records.append({
            "parent_id": c.parent_index, "class_a": c.class_a, "class_b": c.class_b,
            "omega_a": c.omega_a, "omega_b": c.omega_b,
            "sigma_normal": c.sigma_normal, "distance_to_b": c.distance_to_b,
            "score": c.score,
            "normal_x": float(c.normal[0]), "normal_y": float(c.normal[1]),
            "normal_z": float(c.normal[2]),
            "parent_x": float(xyz_a[row, 0] + (xyz_b[row, 0] - xyz_a[row, 0]) * c.omega_a),
            "child_a_x": float(xyz_a[row, 0]), "child_b_x": float(xyz_b[row, 0]),
        })

    return BoundarySplitResult(
        num_parents=m, num_children=2 * m,
        parent_indices=parent_sorted, child_a_indices=child_a, child_b_indices=child_b,
        split_group_id=final_group, split_parent_id=final_parent,
        split_child_role=final_role, soft_part_probs=final_probs,
        diagnostics={"num_selected": m, "n_old": n_old, "n_new": n_new,
                     "extra": final_extra},
        records=records)


# ---------------------------------------------------------------------------
# Sections 26, 27: rebuild the graph and suppress sibling smoothing
# ---------------------------------------------------------------------------


@torch.no_grad()
def suppress_sibling_edges(graph: NeighborGraph, split_group_id: torch.Tensor) -> NeighborGraph:
    """Section 27, mandatory: w_ab = w_ba = 0 for children of the same parent.

    Without this the two children -- deliberately placed close together and
    given OPPOSITE labels -- are strong mutual neighbours, and the very first
    spatial propagation smooths them straight back into one ambiguous label,
    undoing the split.
    """
    gid = torch.as_tensor(split_group_id, device=graph.weight.device).reshape(-1)
    if graph.num_edges == 0:
        return graph
    src_gid = gid.index_select(0, graph.source)
    dst_gid = gid.index_select(0, graph.target)
    sibling = (src_gid >= 0) & (src_gid == dst_gid)
    if not bool(sibling.any()):
        return graph
    weight = graph.weight.clone()
    weight[sibling] = 0.0
    return NeighborGraph(source=graph.source, target=graph.target, weight=weight,
                         num_nodes=graph.num_nodes, k=graph.k)


@torch.no_grad()
def rebuild_state_after_boundary_split(
    gaussians,
    result: BoundarySplitResult,
    *,
    k: int = 16,
    chunk_size: int = 4096,
) -> NeighborGraph:
    """Sections 26 and 27: rebuild the KNN graph, then remove sibling edges."""
    graph = build_neighbor_graph(
        gaussians.get_canonical_xyz.detach().float(), k=int(k), chunk_size=int(chunk_size))
    return suppress_sibling_edges(graph, result.split_group_id)


def invalidate_per_gaussian_caches_after_split(*caches) -> None:
    """Section 29: every Gaussian-indexed cache is invalid once indices move.

    Pixel contributor ids, candidate-evidence caches, responsibility maps and
    per-Gaussian match caches all address Gaussians by index; after an
    append-then-prune every index may mean a different primitive. Clearing is
    the only safe option -- old ids must never be reused.
    """
    for cache in caches:
        if cache is None:
            continue
        if isinstance(cache, dict):
            cache.clear()
        elif isinstance(cache, list):
            del cache[:]


# ---------------------------------------------------------------------------
# Sections 17, 45: diagnostics and artifacts
# ---------------------------------------------------------------------------


@torch.no_grad()
def canonical_render_preservation(
    cameras, gaussians, pipe, background, dataset, *, max_views: int = 8
) -> dict[str, float]:
    """Section 17: render the canonical state with motion disabled.

    Call once before and once after; compare the returned images. The split must
    not catastrophically change the canonical rendering. Diagnostic only.
    """
    from gaussian_renderer import render as render_gaussian

    images = []
    for camera in list(cameras)[: int(max_views)]:
        package = render_gaussian(camera, gaussians, pipe, background,
                                  use_trained_exp=bool(getattr(dataset, "train_test_exp", False)),
                                  separate_sh=False)
        images.append(package["render"].detach().float().cpu())
    return {"num_views": len(images), "images": images}


def compare_canonical_renders(before: dict, after: dict) -> dict[str, float]:
    """Section 17: RGB L1, DSSIM and PSNR between the pre- and post-split renders."""
    import math

    from utils.loss_utils import ssim

    a, b = before.get("images", []), after.get("images", [])
    if not a or not b or len(a) != len(b):
        return {"available": False}
    l1_total, dssim_total, psnr_total = 0.0, 0.0, 0.0
    for x, y in zip(a, b):
        l1 = float((x - y).abs().mean())
        mse = float(((x - y) ** 2).mean())
        l1_total += l1
        dssim_total += float(1.0 - ssim(x[None].cuda(), y[None].cuda())) if x.is_cuda is False and torch.cuda.is_available() else 0.0
        psnr_total += 100.0 if mse <= 1.0e-12 else 10.0 * math.log10(1.0 / mse)
    n = len(a)
    return {"available": True, "num_views": n, "rgb_l1": l1_total / n,
            "dssim": dssim_total / n, "psnr": psnr_total / n}


def save_boundary_split_artifacts(
    directory: Path,
    result: BoundarySplitResult,
    xyz_before: torch.Tensor,
    xyz_after: torch.Tensor,
    *,
    render_comparison: dict | None = None,
) -> None:
    """Section 45. GT is never used for training-side visualization (§45, §51)."""
    from utils.static_endpoint_scene import write_labeled_points_ply
    from utils.temporal_visibility_pretraining import _atomic_torch_save

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    _atomic_torch_save({
        "schema_version": 1,
        "num_parents": result.num_parents,
        "num_children": result.num_children,
        "parent_indices": result.parent_indices.detach().cpu(),
        "child_a_indices": result.child_a_indices.detach().cpu(),
        "child_b_indices": result.child_b_indices.detach().cpu(),
        "split_group_id": result.split_group_id.detach().cpu(),
        "split_parent_id": result.split_parent_id.detach().cpu(),
        "split_child_role": result.split_child_role.detach().cpu(),
        "soft_part_probs": result.soft_part_probs.detach().cpu(),
        "diagnostics": {k: v for k, v in result.diagnostics.items() if k != "extra"},
        "render_comparison": render_comparison or {},
    }, directory / "motion_boundary_split.pt")

    if result.records:
        keys = sorted({k for r in result.records for k in r})
        with (directory / "motion_boundary_split.csv").open("w", newline="", encoding="utf-8") as s:
            writer = csv.DictWriter(s, fieldnames=keys)
            writer.writeheader()
            writer.writerows(result.records)

    import numpy as np

    for name, pts in (("motion_boundary_split_before.ply", xyz_before),
                      ("motion_boundary_split_after.ply", xyz_after)):
        p = torch.as_tensor(pts).detach().cpu().numpy().astype(np.float32)
        if name.endswith("after.ply"):
            role = result.split_child_role.detach().cpu().numpy()
            parts = np.where(role < 0, 0, role + 1).astype(np.int64)
        else:
            parts = np.zeros(p.shape[0], dtype=np.int64)
            idx = result.parent_indices.detach().cpu().numpy()
            if idx.size:
                parts[idx] = 1
        rgb = np.stack([
            np.where(parts == 0, 160, np.where(parts == 1, 220, 40)),
            np.where(parts == 0, 160, np.where(parts == 1, 50, 120)),
            np.where(parts == 0, 160, np.where(parts == 1, 40, 220)),
        ], axis=1).astype(np.uint8)
        write_labeled_points_ply(directory / name, p, rgb, parts)
