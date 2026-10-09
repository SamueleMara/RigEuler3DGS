"""CODEX_temporal_credit_outer_cycle -- judge a label change by its consequence.

The render-graph outer loop asks only "does class k look plausible for Gaussian
i right now". This module adds the missing question: *did assigning that class
actually make the next Killing solve better?*  The score difference between two
solved states is routed back to the label transitions that produced it and
accumulated in a per-Gaussian, per-class memory that biases the next proposal.

Nothing here touches the inner solver (section 2). Every function operates
around complete inner solves, on detached solved state:

    z^(m) -> INNER SOLVE -> T_k^(m)(t) -> S^(m)
    R^(m) = S^(m-1) - S^(m)   routed to  C^(m) = {i : z_i^(m) != z_i^(m-1)}
    memory M -> biases E_ik for the proposal of z^(m+1)

Outer iterations are indexed ``m``; ``t`` stays video-frame time (section 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .neighborhood import NeighborGraph

EPS = 1.0e-8


# ---------------------------------------------------------------------------
# Sections 3-6, 12-15: the quality scores of one SOLVED outer state
# ---------------------------------------------------------------------------


@dataclass
class QualityScore:
    """Section 3: S^(m) = lambda_R S_render + lambda_M S_match, lower is better."""

    raw_render: float
    raw_pixel_match: float
    raw_final_nn: float
    raw_match: float
    normalized_render: float
    normalized_match: float
    total: float
    num_frames: int = 0
    num_views: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalScore:
    """Sections 12-16: per-Gaussian s_i^(m) and its neighbourhood average."""

    render: torch.Tensor          # [N] s_{i,render}, raw
    match: torch.Tensor           # [N] s_{i,match}, raw
    match_valid: torch.Tensor     # [N] bool, section 14
    render_support: torch.Tensor  # [N] total compositing responsibility
    total: torch.Tensor           # [N] normalised and combined
    smoothed: torch.Tensor        # [N] section 16 neighbourhood average


@dataclass
class ScoreNormalization:
    """Section 6 / 15: computed once at m = 0 and then FIXED for the whole run."""

    c_render: float
    c_match: float
    c_render_local: float
    c_match_local: float
    established_at_outer: int = 0
    # §30: the set of Gaussians whose SELECTED candidate has final-frame support
    # shifts as T_k(t) changes -- on USB the moving class went 7856 -> 6798 valid
    # between two outers. Averaging only over the valid ones would make part of
    # S^(m) - S^(m-1) a change of population rather than of quality. So the score
    # runs over all N, with a FIXED penalty standing in for unsupported
    # candidates. §14's "do not assign zero cost" still holds: the penalty is the
    # baseline's 90th percentile of supported energies, not zero.
    unsupported_penalty: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "c_render": self.c_render, "c_match": self.c_match,
            "c_render_local": self.c_render_local, "c_match_local": self.c_match_local,
            "unsupported_penalty": self.unsupported_penalty,
            "established_at_outer": self.established_at_outer,
        }


# ---------------------------------------------------------------------------
# Section 4 + 13: render score and its per-Gaussian attribution
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_render_score(
    *,
    cameras_by_frame,
    gaussians,
    pipe,
    background,
    dataset,
    args,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    num_gaussians: int,
    attribute: bool = True,
    top_l: int = 4,
    contributor_min_weight: float = 1.0e-4,
    frame_stride: int = 1,
) -> tuple[float, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Section 4's S_render plus section 13's rho-weighted per-Gaussian share.

    ``positions``/``rotations`` are the SOLVED per-frame state [F, N, 3] / [F, N, 4],
    so this is pure evaluation -- no optimisation runs (section 4).

    The per-camera loss reuses the inner renderer's own definition
    (``lambda_inner_rgb`` L1 + ``lambda_inner_ssim`` DSSIM + ``lambda_inner_depth``),
    which is what section 4 and section 30 require.

    The per-pixel residual used for attribution is the *per-pixel* part of that
    same loss: the RGB L1 and, where the camera has reliable depth, the depth
    term. DSSIM is a windowed statistic, not a per-pixel residual, so it enters
    the global score only -- attributing it per pixel would invent a quantity the
    renderer never computed.
    """
    from gaussian_renderer import render as render_gaussian
    from utils.loss_utils import ssim

    from .pixel_matching import render_contributors

    device = positions.device
    l_rgb = float(getattr(args, "lambda_inner_rgb", 1.0))
    l_ssim = float(getattr(args, "lambda_inner_ssim", 0.0))
    l_depth = float(getattr(args, "lambda_inner_depth", 0.1))
    use_depth = bool(getattr(args, "dynamic_use_depth", False))

    numerator = torch.zeros(int(num_gaussians), dtype=torch.float64, device=device)
    denominator = torch.zeros(int(num_gaussians), dtype=torch.float64, device=device)

    total, count = 0.0, 0
    rgb_sum, ssim_sum, depth_sum, depth_views = 0.0, 0.0, 0.0, 0
    frames = list(range(0, len(cameras_by_frame), max(1, int(frame_stride))))

    for f in frames:
        cameras = cameras_by_frame[f]
        if not cameras:
            continue
        xyz_f = positions[f].contiguous()
        rot_f = rotations[f].contiguous()
        for camera in cameras:
            package = render_gaussian(
                camera, gaussians, pipe, background,
                use_trained_exp=bool(getattr(dataset, "train_test_exp", False)),
                separate_sh=False,
                means3D_override=xyz_f, rotations_override=rot_f,
            )
            image = package["render"]
            if getattr(camera, "alpha_mask", None) is not None:
                image = image * camera.alpha_mask.to(device=image.device)
            target = camera.original_image.to(device=image.device).clamp(0.0, 1.0)

            per_pixel = (image - target).abs().mean(dim=0)          # [H, W]
            rgb = float(per_pixel.mean())
            dssim = float(1.0 - ssim(image, target))
            residual = l_rgb * per_pixel
            depth_value = 0.0
            if use_depth and bool(getattr(camera, "depth_reliable", False)):
                mask = camera.depth_mask.to(device=image.device)
                d = (package["depth"] - camera.invdepthmap.to(device=image.device)).abs() * mask
                depth_value = float(d.mean())
                residual = residual + l_depth * d.reshape(per_pixel.shape)
                depth_sum += depth_value
                depth_views += 1

            total += l_rgb * rgb + l_ssim * dssim + l_depth * depth_value
            rgb_sum += rgb
            ssim_sum += dssim
            count += 1

            if not attribute:
                continue
            # Section 13: s_i = sum rho_i * residual / sum rho_i, over the pixels
            # where this Gaussian actually has compositing support.
            contributors = render_contributors(
                camera, gaussians, top_l=int(top_l),
                contributor_min_weight=float(contributor_min_weight),
                override_xyz=xyz_f, override_rotation=rot_f,
            )
            ids = contributors.ids.reshape(-1)
            rho = contributors.weights.reshape(-1).to(torch.float64)
            flat = residual.reshape(-1).to(torch.float64)
            per_slot = flat[:, None].expand(-1, int(contributors.ids.shape[-1])).reshape(-1)
            keep = ids >= 0
            if bool(keep.any()):
                idx = ids[keep].long()
                w = rho[keep]
                numerator.scatter_add_(0, idx, w * per_slot[keep])
                denominator.scatter_add_(0, idx, w)

    if count == 0:
        zero = torch.zeros(int(num_gaussians), device=device)
        return 0.0, zero, zero, {"num_views": 0, "num_frames": 0}

    attribution = (numerator / (denominator + EPS)).to(positions.dtype)
    detail = {
        "num_views": count,
        "num_frames": len(frames),
        "mean_rgb": rgb_sum / count,
        "mean_dssim": ssim_sum / count,
        "mean_depth": (depth_sum / depth_views) if depth_views else float("nan"),
        "attributed_gaussians": int((denominator > 0).sum()),
    }
    return total / count, attribution, denominator.to(positions.dtype), detail


# ---------------------------------------------------------------------------
# Sections 5 + 14: final matching score and its per-Gaussian share
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_match_score(
    *,
    pixel_energy: torch.Tensor,
    pixel_num_views: torch.Tensor,
    final_nn_energy: torch.Tensor | None,
    labels: torch.Tensor,
    lambda_final_nn: float = 0.1,
    min_support_views: int = 2,
    unsupported_penalty: float | None = None,
) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:
    """Section 5's S_match and section 14's s_{i,match} for the SELECTED class.

    Returns ``(S_match, S_pixel, S_final_nn, s_i_match, valid)``.

    Section 14: a selected candidate with unsupported final-frame evidence is
    marked invalid -- never given zero cost, which would reward exactly the
    hypotheses the matcher could not see.

    Section 30: when ``unsupported_penalty`` is given, the GLOBAL score runs over
    all N Gaussians with that fixed value substituted for unsupported selected
    candidates. That keeps the evaluated population identical at every outer
    iteration, which averaging over the (moving) valid set would not.
    """
    energy = torch.as_tensor(pixel_energy)
    z = torch.as_tensor(labels, device=energy.device).reshape(-1).long().clamp(0, energy.shape[1] - 1)
    selected = energy.gather(1, z[:, None]).squeeze(1)
    views = torch.as_tensor(pixel_num_views, device=energy.device).gather(1, z[:, None]).squeeze(1)
    valid = views >= float(min_support_views)

    if unsupported_penalty is None:
        # Baseline pass: the penalty does not exist yet, so fall back to the
        # supported population. establish_normalization() then fixes the penalty
        # and every later call receives it.
        pixel_score = float(selected[valid].mean()) if bool(valid.any()) else float(selected.mean())
    else:
        filled = torch.where(valid, selected,
                             torch.full_like(selected, float(unsupported_penalty)))
        pixel_score = float(filled.mean())

    nn_score = 0.0
    if final_nn_energy is not None and int(torch.as_tensor(final_nn_energy).numel()):
        nn = torch.as_tensor(final_nn_energy, device=energy.device)
        nn_score = float(nn.gather(1, z[:, None]).squeeze(1).mean())

    return (pixel_score + float(lambda_final_nn) * nn_score,
            pixel_score, nn_score, selected, valid)


# ---------------------------------------------------------------------------
# Sections 6 + 15: fixed normalisation, then the combined scores
# ---------------------------------------------------------------------------


def unsupported_penalty_from(
    selected_energy: torch.Tensor,
    valid: torch.Tensor,
    *,
    quantile: float = 0.9,
) -> float:
    """Section 30's fixed stand-in cost for an unsupported selected candidate.

    The baseline's 90th percentile of SUPPORTED energies: clearly expensive, so
    section 14's "do not assign zero cost" holds, but still inside the range the
    matcher actually produces rather than an arbitrary large number that would
    let the population term dominate the score.
    """
    e = torch.as_tensor(selected_energy).reshape(-1)
    v = torch.as_tensor(valid, device=e.device).reshape(-1).bool()
    pool = e[v] if bool(v.any()) else e
    if int(pool.numel()) == 0:
        return 0.0
    return float(torch.quantile(pool.float(), float(quantile)))


def establish_normalization(
    render_score: float,
    match_score: float,
    local_render: torch.Tensor,
    local_match: torch.Tensor,
    local_match_valid: torch.Tensor,
    *,
    outer: int = 0,
    unsupported_penalty: float = 0.0,
    eps: float = EPS,
) -> ScoreNormalization:
    """Sections 6 and 15: the constants are taken ONCE, at the baseline state.

    ``match_score`` must already be computed with the section 30 convention
    (all N Gaussians, ``unsupported_penalty`` for the unsupported ones), or the
    baseline would sit on a different scale from every later iteration.
    """
    valid = torch.as_tensor(local_match_valid).reshape(-1).bool()
    match_pool = local_match[valid] if bool(valid.any()) else local_match
    return ScoreNormalization(
        c_render=float(render_score) + float(eps),
        c_match=float(match_score) + float(eps),
        c_render_local=float(local_render.median()) + float(eps),
        c_match_local=float(match_pool.median()) + float(eps),
        established_at_outer=int(outer),
        unsupported_penalty=float(unsupported_penalty),
    )


def combine_global_score(
    render_score: float,
    match_score: float,
    pixel_score: float,
    nn_score: float,
    normalization: ScoreNormalization,
    *,
    lambda_render: float = 1.0,
    lambda_match: float = 0.05,
    detail: dict | None = None,
) -> QualityScore:
    """Section 6 / corrections §22: S = lambda_R S_render/c_R + lambda_M S_match/c_M.

    ``lambda_match`` defaults to 0.05, not 1.0 (corrections §21-§22): measured on
    the soft USB run, the render term correlates with F1 in the right direction
    while the final-matching terms can move the wrong way, so at equal weight the
    combined score failed to track label quality and the temporal memory was
    credited against a signal carrying little information. The score is now
    render-dominant; matching is weak auxiliary supervision.
    """
    nr = float(render_score) / normalization.c_render
    nm = float(match_score) / normalization.c_match
    return QualityScore(
        raw_render=float(render_score),
        raw_pixel_match=float(pixel_score),
        raw_final_nn=float(nn_score),
        raw_match=float(match_score),
        normalized_render=nr,
        normalized_match=nm,
        total=float(lambda_render) * nr + float(lambda_match) * nm,
        num_frames=int((detail or {}).get("num_frames", 0)),
        num_views=int((detail or {}).get("num_views", 0)),
        detail=dict(detail or {}),
    )


def combine_local_score(
    local_render: torch.Tensor,
    local_match: torch.Tensor,
    local_match_valid: torch.Tensor,
    render_support: torch.Tensor,
    normalization: ScoreNormalization,
    graph: NeighborGraph,
    *,
    lambda_render: float = 1.0,
    lambda_match: float = 1.0,
) -> LocalScore:
    """Sections 12, 15, 16: normalise, combine, then smooth over the KNN graph.

    Where the match term is invalid (section 14) the combined score falls back to
    the render attribution alone, rescaled by ``lambda_render + lambda_match`` so
    a Gaussian is not made to look cheap merely because one term was unavailable.
    """
    render = torch.as_tensor(local_render).reshape(-1)
    match = torch.as_tensor(local_match, device=render.device).reshape(-1)
    valid = torch.as_tensor(local_match_valid, device=render.device).reshape(-1).bool()

    nr = render / normalization.c_render_local
    nm = match / normalization.c_match_local
    both = float(lambda_render) * nr + float(lambda_match) * nm
    render_only = (float(lambda_render) + float(lambda_match)) * nr
    total = torch.where(valid, both, render_only)

    return LocalScore(
        render=render, match=match, match_valid=valid,
        render_support=torch.as_tensor(render_support, device=render.device).reshape(-1),
        total=total,
        smoothed=neighborhood_average(graph, total),
    )


# ---------------------------------------------------------------------------
# Section 16: neighbourhood average on the canonical spatial graph
# ---------------------------------------------------------------------------


def neighborhood_average(graph: NeighborGraph, values: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """s_bar_i = sum_{j in N_i u {i}} w_ij s_j / (sum w_ij + eps) (section 16).

    Spatial weights only. The render-preference graph is deliberately NOT used:
    weighting credit by the same preference the credit is meant to correct would
    make the attribution circular.
    """
    v = torch.as_tensor(values).reshape(-1)
    n = int(graph.num_nodes)
    device = v.device
    # The self term: w_ii = 1, so an isolated Gaussian keeps its own score
    # instead of collapsing to zero.
    num = v.clone()
    den = torch.ones(n, dtype=v.dtype, device=device)
    if graph.num_edges:
        src = torch.cat((graph.source, graph.target)).to(device)
        dst = torch.cat((graph.target, graph.source)).to(device)
        w = torch.cat((graph.weight, graph.weight)).to(device=device, dtype=v.dtype)
        num = num.scatter_add(0, src, w * v.index_select(0, dst))
        den = den.scatter_add(0, src, w)
    return num / (den + float(eps))


# ---------------------------------------------------------------------------
# Section 31: align moving-part identities before diffing labels
# ---------------------------------------------------------------------------


@torch.no_grad()
def align_part_permutation(
    previous_labels: torch.Tensor,
    new_labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, list[int], bool]:
    """Remap the new moving labels onto the previous part identities (section 31).

    Static class 0 is fixed. Returns ``(aligned_new_labels, permutation, changed)``
    where ``permutation[k]`` is the previous-iteration id that new class ``k``
    was matched to. Without this a pure renaming of parts would read as tens of
    thousands of false transitions.
    """
    k1 = int(num_classes)
    identity = list(range(k1))
    new = torch.as_tensor(new_labels).reshape(-1).long()
    if k1 <= 2:
        # K = 1: only {static, part 1}; there is nothing to permute.
        return new, identity, False

    prev = torch.as_tensor(previous_labels, device=new.device).reshape(-1).long()
    overlap = torch.zeros((k1 - 1, k1 - 1), dtype=torch.float64)
    for a in range(1, k1):
        mask_a = prev == a
        if not bool(mask_a.any()):
            continue
        for b in range(1, k1):
            overlap[a - 1, b - 1] = float((mask_a & (new == b)).sum())

    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-overlap.numpy())
    except Exception:
        # Greedy maximum-overlap fallback; scipy is not a hard dependency here.
        rows, cols, taken = [], [], set()
        order = torch.argsort(overlap.reshape(-1), descending=True)
        for flat in order.tolist():
            a, b = divmod(flat, k1 - 1)
            if a in rows or b in taken:
                continue
            rows.append(a)
            cols.append(b)
            taken.add(b)
        rows, cols = list(rows), list(cols)

    # permutation maps NEW class id -> PREVIOUS class id.
    permutation = identity[:]
    for a, b in zip(rows, cols):
        permutation[int(b) + 1] = int(a) + 1
    lookup = torch.tensor(permutation, device=new.device, dtype=torch.long)
    aligned = lookup.index_select(0, new)
    return aligned, permutation, permutation != identity


# ---------------------------------------------------------------------------
# Section 32: connected components of the changed set
# ---------------------------------------------------------------------------


@torch.no_grad()
def changed_components(graph: NeighborGraph, changed: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Component id per changed Gaussian (-1 elsewhere) and the component count."""
    mask = torch.as_tensor(changed).reshape(-1).bool()
    n = int(graph.num_nodes)
    out = torch.full((n,), -1, dtype=torch.long, device=mask.device)
    if not bool(mask.any()):
        return out, 0

    label = torch.arange(n, device=mask.device)
    if graph.num_edges:
        keep = (mask.index_select(0, graph.source.to(mask.device))
                & mask.index_select(0, graph.target.to(mask.device)))
        src = torch.cat((graph.source.to(mask.device)[keep], graph.target.to(mask.device)[keep]))
        dst = torch.cat((graph.target.to(mask.device)[keep], graph.source.to(mask.device)[keep]))
        for _ in range(n):
            candidate = label.index_select(0, dst)
            updated = label.scatter_reduce(0, src, candidate, reduce="amin", include_self=True)
            if bool(torch.equal(updated, label)):
                break
            label = updated

    roots = torch.unique(label[mask])
    remap = torch.full((n,), -1, dtype=torch.long, device=mask.device)
    remap[roots] = torch.arange(int(roots.numel()), device=mask.device)
    out[mask] = remap[label[mask]]
    return out, int(roots.numel())


# ---------------------------------------------------------------------------
# Sections 17, 18, 33: the transition reward
# ---------------------------------------------------------------------------


def local_transition_reward(
    smoothed_previous: torch.Tensor,
    smoothed_new: torch.Tensor,
    *,
    clip: float = 1.0,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 17: r_i = (s_bar^(m-1) - s_bar^(m)) / (|s_bar^(m-1)| + eps), clipped."""
    prev = torch.as_tensor(smoothed_previous).reshape(-1)
    new = torch.as_tensor(smoothed_new, device=prev.device).reshape(-1)
    relative = (prev - new) / (prev.abs() + float(eps))
    return relative.clamp(-abs(float(clip)), abs(float(clip)))


def combine_reward(
    global_relative_reward: float,
    local_reward: torch.Tensor,
    *,
    beta_global: float = 0.25,
    beta_component: float = 0.0,
    component_id: torch.Tensor | None = None,
    changed: torch.Tensor | None = None,
    clip: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sections 18 and 33: hat_r = beta_G R + beta_C r_q + (1-beta_G-beta_C) r_i.

    ``beta_component`` defaults to 0, which is section 33's required first
    setting, and then this reduces exactly to section 18.
    Returns ``(hat_r, component_reward)``.
    """
    local = torch.as_tensor(local_reward).reshape(-1)
    bg = float(beta_global)
    bc = float(beta_component)
    if bg < 0.0 or bc < 0.0 or bg + bc > 1.0:
        raise ValueError(f"need beta_global, beta_component >= 0 and their sum <= 1; got {bg}, {bc}")

    component_reward = torch.zeros_like(local)
    if bc > 0.0 and component_id is not None:
        cid = torch.as_tensor(component_id, device=local.device).reshape(-1)
        valid = cid >= 0
        if bool(valid.any()):
            count = int(cid[valid].max()) + 1
            total = torch.zeros(count, dtype=local.dtype, device=local.device)
            size = torch.zeros(count, dtype=local.dtype, device=local.device)
            total.scatter_add_(0, cid[valid], local[valid])
            size.scatter_add_(0, cid[valid], torch.ones_like(local[valid]))
            mean = total / size.clamp_min(1.0)
            component_reward[valid] = mean.index_select(0, cid[valid])

    hat = bg * float(global_relative_reward) + bc * component_reward + (1.0 - bg - bc) * local
    hat = hat.clamp(-abs(float(clip)), abs(float(clip)))
    if changed is not None:
        # Only transitions carry credit (section 11); everything else stays 0 so
        # it cannot leak into the memory update.
        hat = torch.where(torch.as_tensor(changed, device=hat.device).reshape(-1).bool(),
                          hat, torch.zeros_like(hat))
    return hat, component_reward


# ---------------------------------------------------------------------------
# Sections 19-22: the temporal label memory
# ---------------------------------------------------------------------------


def update_temporal_memory(
    memory: torch.Tensor,
    old_labels: torch.Tensor,
    new_labels: torch.Tensor,
    changed_mask: torch.Tensor,
    reward: torch.Tensor,
    *,
    decay: float = 0.8,
    global_decay: float = 1.0,
) -> torch.Tensor:
    """Sections 20-22, out of place.

        M_ib <- gamma M_ib + (1 - gamma) hat_r_i
        M_ia <- gamma M_ia - (1 - gamma) hat_r_i        for i in C^(m) only

    The optional global decay (section 21) is applied BEFORE the transition
    updates, as the spec requires, and defaults to 1.0 (no decay).
    """
    m = torch.as_tensor(memory).clone()
    changed = torch.as_tensor(changed_mask, device=m.device).reshape(-1).bool()
    if float(global_decay) != 1.0:
        m = m * float(global_decay)
    if not bool(changed.any()):
        return m.clamp(-1.0, 1.0)

    gamma = float(decay)
    idx = torch.nonzero(changed, as_tuple=False).reshape(-1)
    a = torch.as_tensor(old_labels, device=m.device).reshape(-1).long()[idx]
    b = torch.as_tensor(new_labels, device=m.device).reshape(-1).long()[idx]
    r = torch.as_tensor(reward, device=m.device).reshape(-1).to(m.dtype)[idx]

    # A transition a -> b with a == b is not a transition; guard so a degenerate
    # entry cannot apply +r and -r to the same cell and silently cancel.
    real = a != b
    idx, a, b, r = idx[real], a[real], b[real], r[real]
    if int(idx.numel()) == 0:
        return m.clamp(-1.0, 1.0)

    m[idx, b] = gamma * m[idx, b] + (1.0 - gamma) * r
    m[idx, a] = gamma * m[idx, a] - (1.0 - gamma) * r
    return m.clamp(-1.0, 1.0)                                   # section 22


# ---------------------------------------------------------------------------
# Sections 23-24: bias the next candidate score, with a margin cap
# ---------------------------------------------------------------------------


def apply_temporal_memory(
    energy: torch.Tensor,
    memory: torch.Tensor,
    *,
    raw_margin: torch.Tensor | None = None,
    lambda_memory: float = 0.25,
    margin_cap: float = 0.5,
    margin_floor: float = 1.0e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Section 23: E_tilde = E - lambda_mem M, capped by section 24.

        |lambda_mem M_ik| <= rho_mem * max(m_i^render, m_floor)

    so memory can settle ambiguous decisions but cannot overturn a Gaussian
    whose current render evidence is overwhelming. Current evidence stays
    primary; memory is a bias, never an override.
    """
    e = torch.as_tensor(energy)
    m = torch.as_tensor(memory, device=e.device, dtype=e.dtype)
    if m.shape != e.shape:
        raise ValueError(f"memory {tuple(m.shape)} does not match energy {tuple(e.shape)}")

    bias = float(lambda_memory) * m
    stats: dict[str, float] = {
        "mean_abs_bias_uncapped": float(bias.abs().mean()),
        "max_abs_bias_uncapped": float(bias.abs().max()) if bias.numel() else 0.0,
    }
    if raw_margin is not None:
        margin = torch.as_tensor(raw_margin, device=e.device, dtype=e.dtype).reshape(-1)
        limit = (float(margin_cap) * margin.clamp_min(float(margin_floor)))[:, None]
        capped = bias.clamp(-limit, limit)
        stats["fraction_capped"] = float((bias.abs() > limit + EPS).float().mean())
        bias = capped
    else:
        stats["fraction_capped"] = 0.0

    stats["mean_abs_bias"] = float(bias.abs().mean())
    stats["max_abs_bias"] = float(bias.abs().max()) if bias.numel() else 0.0
    biased = e - bias
    stats["num_preference_flips"] = float((biased.argmin(dim=1) != e.argmin(dim=1)).sum())
    return biased, stats


# ---------------------------------------------------------------------------
# Section 26: optional differentiable transition-credit loss (diagnostic)
# ---------------------------------------------------------------------------


def temporal_credit_loss(
    probabilities: torch.Tensor,
    old_labels: torch.Tensor,
    new_labels: torch.Tensor,
    changed_mask: torch.Tensor,
    reward: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Section 26: -mean_i hat_r_i [log P_{i,b} - log P_{i,a}] over i in C^(m).

    Logged as a diagnostic. Section 26 is explicit that the memory-biased
    deterministic update is sufficient for the first implementation, so this is
    never added to an optimiser here.
    """
    p = torch.as_tensor(probabilities)
    changed = torch.as_tensor(changed_mask, device=p.device).reshape(-1).bool()
    if not bool(changed.any()):
        return p.sum() * 0.0
    idx = torch.nonzero(changed, as_tuple=False).reshape(-1)
    a = torch.as_tensor(old_labels, device=p.device).reshape(-1).long()[idx]
    b = torch.as_tensor(new_labels, device=p.device).reshape(-1).long()[idx]
    r = torch.as_tensor(reward, device=p.device).reshape(-1).to(p.dtype)[idx]
    log_b = torch.log(p[idx, b] + float(eps))
    log_a = torch.log(p[idx, a] + float(eps))
    return -(r * (log_b - log_a)).mean()


# ---------------------------------------------------------------------------
# Sections 34, 37: oscillation and transition-type diagnostics
# ---------------------------------------------------------------------------


@torch.no_grad()
def detect_label_oscillation(
    label_two_back: torch.Tensor | None,
    label_one_back: torch.Tensor,
    label_now: torch.Tensor,
) -> torch.Tensor:
    """Section 34: z^(m) == z^(m-2) != z^(m-1) is an a -> b -> a oscillation."""
    now = torch.as_tensor(label_now).reshape(-1)
    if label_two_back is None:
        return torch.zeros_like(now, dtype=torch.bool)
    two = torch.as_tensor(label_two_back, device=now.device).reshape(-1)
    one = torch.as_tensor(label_one_back, device=now.device).reshape(-1)
    return (now == two) & (now != one)


@torch.no_grad()
def transition_type_report(
    old_labels: torch.Tensor,
    new_labels: torch.Tensor,
    changed_mask: torch.Tensor,
    reward: torch.Tensor,
    local_previous: torch.Tensor,
    local_new: torch.Tensor,
    render_previous: torch.Tensor,
    render_new: torch.Tensor,
    match_previous: torch.Tensor,
    match_new: torch.Tensor,
) -> list[dict[str, Any]]:
    """Section 37: per transition type a -> b, count and reward statistics."""
    changed = torch.as_tensor(changed_mask).reshape(-1).bool()
    if not bool(changed.any()):
        return []
    a = torch.as_tensor(old_labels, device=changed.device).reshape(-1).long()
    b = torch.as_tensor(new_labels, device=changed.device).reshape(-1).long()
    r = torch.as_tensor(reward, device=changed.device).reshape(-1)

    rows: list[dict[str, Any]] = []
    pairs = torch.unique(torch.stack((a[changed], b[changed]), dim=1), dim=0)
    for pair in pairs.tolist():
        src, dst = int(pair[0]), int(pair[1])
        mask = changed & (a == src) & (b == dst)
        count = int(mask.sum())
        if count == 0:
            continue
        rewards = r[mask]
        rows.append({
            "transition": f"{src}->{dst}",
            "from_class": src,
            "to_class": dst,
            "count": count,
            "mean_reward": float(rewards.mean()),
            "median_reward": float(rewards.median()),
            "fraction_positive_reward": float((rewards > 0).float().mean()),
            # Improvement = previous - new, so positive means it got better.
            "mean_local_improvement": float((local_previous[mask] - local_new[mask]).mean()),
            "mean_render_improvement": float((render_previous[mask] - render_new[mask]).mean()),
            "mean_match_improvement": float((match_previous[mask] - match_new[mask]).mean()),
        })
    rows.sort(key=lambda row: -row["count"])
    return rows


# ---------------------------------------------------------------------------
# Sections 27, 28, 38, 39: the cycle itself
# ---------------------------------------------------------------------------


def score_f1_correlation(
    score_total,
    score_render,
    score_match,
    f1,
) -> dict[str, Any]:
    """Corrections §25, evaluation only: does S actually track segmentation quality?

    Lower S is better, so a working score is NEGATIVELY correlated with F1.
    Never used to tune anything inside a run.
    """
    import math as _math

    def _corr(a, b) -> float:
        xs = [float(v) for v in a]
        ys = [float(v) for v in b]
        pairs = [(x, y) for x, y in zip(xs, ys)
                 if _math.isfinite(x) and _math.isfinite(y)]
        if len(pairs) < 2:
            return float("nan")
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = _math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = _math.sqrt(sum((y - my) ** 2 for y in ys))
        return num / dx / dy if dx > 0 and dy > 0 else float("nan")

    def _rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for position, index in enumerate(order):
            ranks[index] = float(position)
        return ranks

    out = {
        "pearson_score_f1": _corr(score_total, f1),
        "spearman_score_f1": _corr(_rank(list(score_total)), _rank(list(f1))),
        "pearson_render_f1": _corr(score_render, f1),
        "pearson_match_f1": _corr(score_match, f1),
        "expected_sign": "negative",
    }
    p = out["pearson_score_f1"]
    out["tracks_quality"] = bool(p == p and p < 0.0)
    return out


def _finite(*tensors) -> bool:
    for t in tensors:
        if t is None:
            continue
        v = torch.as_tensor(t)
        if v.numel() and not bool(torch.isfinite(v).all()):
            return False
    return True


class TemporalCreditCycle:
    """Holds the across-outer state: normalisation, memory, previous scores.

    Usage mirrors section 28 exactly. Per outer iteration ``m``, after the inner
    solve has produced ``motion^(m)`` for the labels ``z^(m)`` currently in use::

        record = cycle.observe(outer=m, labels=z_m, quality=..., local=...)
        biased, stats = cycle.bias_energy(energy, raw_margin)
        # ... render-graph proposes z^(m+1) from ``biased`` ...

    ``observe`` credits the transition ``z^(m-1) -> z^(m)`` that a previous
    proposal caused, which is the only point at which a reward exists (section 7:
    no reward is assigned to ``z^(0)``).
    """

    def __init__(
        self,
        num_gaussians: int,
        num_classes: int,
        *,
        device=None,
        beta_global: float = 0.25,
        beta_component: float = 0.0,
        reward_clip: float = 1.0,
        memory_decay: float = 0.8,
        memory_global_decay: float = 1.0,
        lambda_memory: float = 0.25,
        memory_margin_cap: float = 0.5,
        lambda_render: float = 1.0,
        lambda_match: float = 0.05,
    ) -> None:
        self.num_classes = int(num_classes)
        # Section 19 / 40: memory starts at exactly zero. It is never preloaded
        # from the endpoint-contrastive initialisation -- no transition has
        # happened yet, so there is nothing to have observed.
        self.memory = torch.zeros((int(num_gaussians), int(num_classes)), device=device)
        self.normalization: ScoreNormalization | None = None
        self.beta_global = float(beta_global)
        self.beta_component = float(beta_component)
        self.reward_clip = float(reward_clip)
        self.memory_decay = float(memory_decay)
        self.memory_global_decay = float(memory_global_decay)
        self.lambda_memory = float(lambda_memory)
        self.memory_margin_cap = float(memory_margin_cap)
        self.lambda_render = float(lambda_render)
        self.lambda_match = float(lambda_match)

        self.previous_score: QualityScore | None = None
        self.previous_local: LocalScore | None = None
        self.previous_labels: torch.Tensor | None = None
        self.label_two_back: torch.Tensor | None = None
        self.history: list[dict[str, Any]] = []
        self.last_record: dict[str, Any] | None = None
        self.last_energy_stats: dict[str, float] = {}

    # -- section 28 step C/D ------------------------------------------------
    def observe(
        self,
        *,
        outer: int,
        labels: torch.Tensor,
        quality: QualityScore,
        local: LocalScore,
        graph: NeighborGraph,
        probabilities: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Score the solved state and credit the transition that produced it."""
        z = torch.as_tensor(labels).reshape(-1).long()
        record: dict[str, Any] = {
            "outer": int(outer),
            # Corrections §24: never hide the subterms inside one scalar.
            "score_total": quality.total,
            "score_render_raw": quality.raw_render,
            "score_render_normalized": quality.normalized_render,
            "score_match_raw": quality.raw_match,
            "score_match_normalized": quality.normalized_match,
            "score_pixel_match_raw": quality.raw_pixel_match,
            "score_final_nn_raw": quality.raw_final_nn,
            "lambda_match": self.lambda_match,
            "score_render": quality.normalized_render,
            "score_match": quality.normalized_match,
            "raw_score_render": quality.raw_render,
            "raw_score_pixel_match": quality.raw_pixel_match,
            "raw_score_final_nn": quality.raw_final_nn,
            "num_score_frames": quality.num_frames,
            "num_score_views": quality.num_views,
            # §35 names the static column explicitly; num_part_0 is the same set.
            "num_static": int((z == 0).sum()),
            **{f"num_part_{c}": int((z == c).sum()) for c in range(self.num_classes)},
        }

        if self.previous_score is None or self.previous_labels is None:
            # Section 7: this is the baseline. No temporal comparison exists and
            # no artificial S^(-1) is invented.
            record.update({
                "is_baseline": True,
                "relative_global_reward": float("nan"),
                "num_changed": 0, "changed_fraction": 0.0, "num_changed_components": 0,
                "mean_local_reward_changed": float("nan"),
                "median_local_reward_changed": float("nan"),
                "num_positive_reward_changes": 0, "num_negative_reward_changes": 0,
                "mean_abs_memory": float(self.memory.abs().mean()),
                "max_abs_memory": float(self.memory.abs().max()) if self.memory.numel() else 0.0,
                "num_label_oscillations": 0,
                "part_permutation_applied": False,
                "temporal_credit_loss": float("nan"),
            })
            self._advance(z, quality, local)
            self.history.append(record)
            self.last_record = record
            return record

        # Section 31: align part identities before diffing, or a pure renaming
        # reads as thousands of transitions.
        aligned, permutation, permuted = align_part_permutation(
            self.previous_labels, z, self.num_classes)
        changed = aligned != self.previous_labels

        # Section 9.
        prev_total = self.previous_score.total
        global_reward = prev_total - quality.total
        global_relative = global_reward / (abs(prev_total) + EPS)

        # Sections 17, 32, 18.
        local_reward = local_transition_reward(
            self.previous_local.smoothed, local.smoothed, clip=self.reward_clip)
        component_id, num_components = changed_components(graph, changed)
        final_reward, component_reward = combine_reward(
            global_relative, local_reward,
            beta_global=self.beta_global, beta_component=self.beta_component,
            component_id=component_id, changed=changed, clip=self.reward_clip,
        )

        # Sections 38/39: a worse global score is credited negatively; it never
        # triggers a blanket revert. Only non-finite state is a reason to refuse
        # the update, and that is numerical safety, not an optimisation gate.
        numerically_valid = _finite(quality.total, local.total, local.smoothed, final_reward)
        memory_before = self.memory.clone()
        if numerically_valid:
            self.memory = update_temporal_memory(
                self.memory, self.previous_labels, aligned, changed, final_reward,
                decay=self.memory_decay, global_decay=self.memory_global_decay,
            )

        oscillation = detect_label_oscillation(self.label_two_back, self.previous_labels, aligned)
        changed_rewards = final_reward[changed]
        report = transition_type_report(
            self.previous_labels, aligned, changed, final_reward,
            self.previous_local.smoothed, local.smoothed,
            self.previous_local.render, local.render,
            self.previous_local.match, local.match,
        )

        record.update({
            "is_baseline": False,
            "relative_global_reward": float(global_relative),
            "absolute_global_reward": float(global_reward),
            "num_changed": int(changed.sum()),
            "changed_fraction": float(changed.float().mean()),
            "num_changed_components": int(num_components),
            "mean_local_reward_changed": (
                float(changed_rewards.mean()) if int(changed_rewards.numel()) else float("nan")),
            "median_local_reward_changed": (
                float(changed_rewards.median()) if int(changed_rewards.numel()) else float("nan")),
            "num_positive_reward_changes": int((changed_rewards > 0).sum()),
            "num_negative_reward_changes": int((changed_rewards < 0).sum()),
            "mean_abs_memory": float(self.memory.abs().mean()),
            "max_abs_memory": float(self.memory.abs().max()) if self.memory.numel() else 0.0,
            "num_label_oscillations": int(oscillation.sum()),
            "part_permutation_applied": bool(permuted),
            "part_permutation": permutation,
            "numerically_valid": bool(numerically_valid),
            "temporal_credit_loss": (
                float(temporal_credit_loss(probabilities, self.previous_labels, aligned,
                                           changed, final_reward))
                if probabilities is not None else float("nan")),
        })

        self.last_record = record
        self.history.append(record)
        self._artifact = {
            "old_label": self.previous_labels.detach().cpu(),
            "new_label": aligned.detach().cpu(),
            "changed_mask": changed.detach().cpu(),
            "local_score_old": self.previous_local.smoothed.detach().cpu(),
            "local_score_new": local.smoothed.detach().cpu(),
            "local_score_old_unsmoothed": self.previous_local.total.detach().cpu(),
            "local_score_new_unsmoothed": local.total.detach().cpu(),
            "local_reward": local_reward.detach().cpu(),
            "final_reward": final_reward.detach().cpu(),
            "component_reward": component_reward.detach().cpu(),
            "changed_connected_component_id": component_id.detach().cpu(),
            "memory_before": memory_before.detach().cpu(),
            "memory_after": self.memory.detach().cpu(),
            "label_oscillation_mask": oscillation.detach().cpu(),
            "transition_report": report,
            "part_permutation": permutation,
        }
        self._advance(z, quality, local)
        return record

    def _advance(self, labels: torch.Tensor, quality: QualityScore, local: LocalScore) -> None:
        """Section 28 step E, plus section 34's two-step label history."""
        self.label_two_back = self.previous_labels
        self.previous_labels = labels.detach().clone()
        self.previous_score = quality
        self.previous_local = local

    # -- section 28 step A --------------------------------------------------
    def bias_energy(
        self,
        energy: torch.Tensor,
        raw_margin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Section 23/24: E_tilde for the next render-graph proposal."""
        biased, stats = apply_temporal_memory(
            energy, self.memory.to(energy.device),
            raw_margin=raw_margin,
            lambda_memory=self.lambda_memory,
            margin_cap=self.memory_margin_cap,
        )
        self.last_energy_stats = stats
        return biased, stats

    def artifact(self, render_energy: torch.Tensor, biased_energy: torch.Tensor) -> dict[str, Any]:
        """Section 36's per-transition debug payload."""
        payload = dict(getattr(self, "_artifact", {}) or {})
        if payload:
            payload["render_energy"] = torch.as_tensor(render_energy).detach().cpu()
            payload["memory_biased_energy"] = torch.as_tensor(biased_energy).detach().cpu()
            payload["schema_version"] = 1
        return payload
