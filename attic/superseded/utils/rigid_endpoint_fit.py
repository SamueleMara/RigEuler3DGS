"""Finite SE(3) fitting used by progressive rigid discovery.

This module intentionally deals with endpoint transforms, not velocity fields.
Stage 1 needs a small set of robust finite rigid hypotheses before any temporal
Killing/twist refinement is allowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def skew(vector: torch.Tensor) -> torch.Tensor:
    vector = torch.as_tensor(vector)
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )


def hat_se3(xi: torch.Tensor) -> torch.Tensor:
    xi = torch.as_tensor(xi)
    if xi.shape[-1] != 6:
        raise ValueError("xi must end in 6")
    v = xi[..., :3]
    w = xi[..., 3:]
    mat = torch.zeros(xi.shape[:-1] + (4, 4), device=xi.device, dtype=xi.dtype)
    mat[..., :3, :3] = skew(w)
    mat[..., :3, 3] = v
    return mat


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """SE(3) exponential using torch's differentiable matrix exponential."""
    return torch.matrix_exp(hat_se3(xi))


def _so3_log(R: torch.Tensor) -> torch.Tensor:
    trace = R.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    vee = torch.stack(
        (R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]),
        dim=-1,
    )
    small = theta < 1.0e-5
    scale = theta / (2.0 * torch.sin(theta).clamp_min(1.0e-8))
    omega = scale[..., None] * vee
    omega = torch.where(small[..., None], 0.5 * vee, omega)
    return omega


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """SE(3) logarithm with the standard inverse left-Jacobian for translation."""
    T = torch.as_tensor(T)
    if T.shape[-2:] != (4, 4):
        raise ValueError("T must end in 4x4")
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    w = _so3_log(R)
    theta = torch.linalg.norm(w, dim=-1)
    W = skew(w)
    I = torch.eye(3, device=T.device, dtype=T.dtype).expand(T.shape[:-2] + (3, 3))
    half = 0.5
    theta2 = theta.square()
    small = theta < 1.0e-5
    coef = (
        (1.0 / theta2.clamp_min(1.0e-12))
        - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta).clamp_min(1.0e-8))
    )
    Vinv = I - half * W + coef[..., None, None] * (W @ W)
    Vinv_small = I - half * W + (1.0 / 12.0) * (W @ W)
    Vinv = torch.where(small[..., None, None], Vinv_small, Vinv)
    v = torch.einsum("...ij,...j->...i", Vinv, t)
    return torch.cat((v, w), dim=-1)


def transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    T = torch.as_tensor(T, device=points.device, dtype=points.dtype)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    if T.ndim == 2:
        return points @ R.transpose(-1, -2) + t
    return torch.einsum("kij,nj->kni", R, points) + t[:, None, :]


def weighted_kabsch(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor | float | int]:
    """Fit ``target ~= R source + t`` with weighted Kabsch."""
    source = torch.as_tensor(source).float()
    target = torch.as_tensor(target, device=source.device, dtype=source.dtype)
    if source.ndim != 2 or target.shape != source.shape or source.shape[-1] != 3:
        raise ValueError("source and target must be matching [N,3]")
    if int(source.shape[0]) < 3:
        raise ValueError("weighted_kabsch needs at least three points")
    if weight is None:
        w = torch.ones(source.shape[0], device=source.device, dtype=source.dtype)
    else:
        w = torch.as_tensor(weight, device=source.device, dtype=source.dtype).reshape(-1).clamp_min(0.0)
    valid = torch.isfinite(source).all(-1) & torch.isfinite(target).all(-1) & torch.isfinite(w) & (w > 0)
    if int(valid.sum()) < 3 or float(w[valid].sum()) <= eps:
        raise ValueError("weighted_kabsch has fewer than three valid weighted points")
    x = source[valid]
    y = target[valid]
    wv = w[valid]
    wnorm = wv / wv.sum().clamp_min(eps)
    xbar = (wnorm[:, None] * x).sum(0)
    ybar = (wnorm[:, None] * y).sum(0)
    xc = x - xbar
    yc = y - ybar
    H = xc.transpose(0, 1) @ (wnorm[:, None] * yc)
    U, _, Vh = torch.linalg.svd(H)
    R = Vh.transpose(0, 1) @ U.transpose(0, 1)
    if torch.det(R) < 0:
        Vh = Vh.clone()
        Vh[-1] *= -1
        R = Vh.transpose(0, 1) @ U.transpose(0, 1)
    t = ybar - R @ xbar
    T = torch.eye(4, device=source.device, dtype=source.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    pred = source @ R.transpose(0, 1) + t
    residual = torch.linalg.norm(pred - target, dim=-1)
    weighted_rms = torch.sqrt((w * residual.square()).sum() / w.sum().clamp_min(eps))
    return {
        "T": T,
        "R": R,
        "t": t,
        "residual_per_point": residual,
        "weighted_rms": float(weighted_rms.detach().cpu()),
        "valid_count": int(valid.sum()),
        "weight_sum": float(w[valid].sum().detach().cpu()),
    }


def huber_weight(residual: torch.Tensor, delta: float) -> torch.Tensor:
    delta = max(float(delta), 1.0e-8)
    r = torch.as_tensor(residual).clamp_min(0.0)
    return torch.where(r <= delta, torch.ones_like(r), delta / r.clamp_min(1.0e-8))


def robust_fit_se3_irls(
    source: torch.Tensor,
    target: torch.Tensor,
    base_weight: torch.Tensor,
    *,
    init_T: torch.Tensor | None = None,
    iters: int = 5,
    huber_delta: float = 0.02,
) -> dict[str, Any]:
    weight = torch.as_tensor(base_weight, device=source.device, dtype=source.dtype).reshape(-1).clamp_min(0.0)
    if init_T is not None:
        residual = torch.linalg.norm(transform_points(init_T.to(source), source) - target, dim=-1)
        weight = weight * huber_weight(residual, huber_delta)
    result = None
    diagnostics = []
    for iteration in range(max(1, int(iters))):
        result = weighted_kabsch(source, target, weight)
        residual = torch.as_tensor(result["residual_per_point"], device=source.device, dtype=source.dtype)
        diagnostics.append({
            "iteration": int(iteration),
            "weighted_rms": float(result["weighted_rms"]),
            "weight_sum": float(weight.sum().detach().cpu()),
            "median_residual": float(residual.median().detach().cpu()),
        })
        weight = torch.as_tensor(base_weight, device=source.device, dtype=source.dtype).reshape(-1).clamp_min(0.0)
        weight = weight * huber_weight(residual, huber_delta)
    assert result is not None
    result["robust_weight"] = weight.detach()
    result["diagnostics"] = diagnostics
    result["xi"] = se3_log(result["T"])
    return result


@dataclass
class RansacConfig:
    iters: int = 256
    inlier_threshold: float = 0.03
    irls_iters: int = 5
    huber_delta: float = 0.02
    min_samples: int = 3
    seed: int = 20260912
    prosac_fraction: float = 0.8


def transform_summary(T: torch.Tensor) -> dict[str, Any]:
    T = torch.as_tensor(T).detach().cpu().float()
    xi = se3_log(T)
    angle = float(torch.linalg.norm(xi[3:]))
    axis = torch.zeros(3)
    if angle > 1.0e-8:
        axis = xi[3:] / angle
    return {
        "rotation_angle_deg": math.degrees(angle),
        "rotation_axis": [float(v) for v in axis],
        "translation": [float(v) for v in T[:3, 3]],
        "det_R": float(torch.det(T[:3, :3])),
    }


def target_median_nn_scale(target_xyz: torch.Tensor, *, chunk_size: int = 2048) -> float:
    target = torch.as_tensor(target_xyz).detach().float()
    n = int(target.shape[0])
    if n <= 1:
        return 1.0
    values = []
    k = min(2, n)
    for start in range(0, n, max(1, int(chunk_size))):
        d = torch.cdist(target[start:start + int(chunk_size)], target)
        values.append(torch.topk(d, k=k, largest=False).values[:, 1])
    scale = torch.cat(values)
    finite = scale[torch.isfinite(scale) & (scale > 0)]
    return float(finite.median()) if finite.numel() else 1.0


def _chunked_knn(query: torch.Tensor, target: torch.Tensor, k: int, *, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    indices = []
    kk = min(max(1, int(k)), int(target.shape[0]))
    for start in range(0, int(query.shape[0]), max(1, int(chunk_size))):
        d2 = torch.cdist(query[start:start + int(chunk_size)], target).square()
        v, i = torch.topk(d2, k=kk, largest=False, sorted=True)
        values.append(v)
        indices.append(i)
    return torch.cat(values, 0), torch.cat(indices, 0)


def build_source_neighbor_graph(source: torch.Tensor, k: int, *, chunk_size: int = 2048) -> dict[str, torch.Tensor]:
    """Self-excluding t0 k-NN graph over canonical positions, with t0 distances.

    Used to enforce that a source point's distance to its own t0 neighbours is
    preserved by whatever it and they get matched to at frame T -- a rigid part
    moving as one piece cannot stretch or compress its own local neighbourhood.
    """
    source = torch.as_tensor(source).detach().float()
    n = int(source.shape[0])
    kk = min(max(1, int(k)) + 1, n)
    if kk <= 1:
        empty_idx = torch.zeros((n, 0), dtype=torch.long, device=source.device)
        empty_dist = torch.zeros((n, 0), dtype=source.dtype, device=source.device)
        return {"neighbor_idx": empty_idx, "neighbor_dist0": empty_dist}
    idx_chunks = []
    dist_chunks = []
    for start in range(0, n, max(1, int(chunk_size))):
        d2 = torch.cdist(source[start:start + int(chunk_size)], source).square()
        values, indices = torch.topk(d2, k=kk, largest=False, sorted=True)
        idx_chunks.append(indices[:, 1:])
        dist_chunks.append(values[:, 1:].clamp_min(0.0).sqrt())
    return {
        "neighbor_idx": torch.cat(idx_chunks, dim=0),
        "neighbor_dist0": torch.cat(dist_chunks, dim=0),
    }


def build_fm_target_candidates(
    *,
    fm_endpoint_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    legacy_indices: torch.Tensor | None,
    legacy_costs: torch.Tensor | None,
    endpoint_knn: int,
    max_candidates: int,
    spcd: float,
    chunk_size: int = 1024,
    source_xyz: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Store actual frame-T target candidates for every source Gaussian."""
    endpoint = torch.as_tensor(fm_endpoint_xyz).detach().float()
    target = torch.as_tensor(target_xyz, device=endpoint.device, dtype=endpoint.dtype).detach()
    parts_idx = []
    parts_prior = []
    parts_source = []
    if legacy_indices is not None:
        idx = torch.as_tensor(legacy_indices, device=endpoint.device).long()
        valid = (idx >= 0) & (idx < int(target.shape[0]))
        if legacy_costs is not None:
            cost = torch.as_tensor(legacy_costs, device=endpoint.device, dtype=endpoint.dtype)
            prior = torch.softmax(-torch.where(valid, cost, torch.full_like(cost, 1.0e6)), dim=1)
        else:
            prior = valid.to(endpoint.dtype)
            prior = prior / prior.sum(1, keepdim=True).clamp_min(1.0e-8)
        parts_idx.append(idx.clamp(0, max(0, int(target.shape[0]) - 1)))
        parts_prior.append(prior * valid.to(prior.dtype))
        parts_source.append(torch.ones_like(idx))

    sigma2 = max(float(spcd), 1.0e-8) ** 2
    dist2, idx = _chunked_knn(endpoint, target, int(endpoint_knn), chunk_size=int(chunk_size))
    prior = torch.softmax(-0.5 * dist2 / sigma2, dim=1)
    parts_idx.append(idx)
    parts_prior.append(prior)
    parts_source.append(torch.full_like(idx, 2))

    if source_xyz is not None:
        # Guarantee every point can propose "I did not move": k-NN around its
        # own t0 position, not only around the FM-predicted endpoint.  A
        # point whose true motion is (near) zero can otherwise lose its own
        # near-identity match here whenever the FM prediction itself carries
        # even small error, since the endpoint-KNN branch above searches
        # around that (slightly off) prediction rather than the t0 position.
        source = torch.as_tensor(source_xyz, device=endpoint.device, dtype=endpoint.dtype).detach()
        dist2_t0, idx_t0 = _chunked_knn(source, target, int(endpoint_knn), chunk_size=int(chunk_size))
        prior_t0 = torch.softmax(-0.5 * dist2_t0 / sigma2, dim=1)
        parts_idx.append(idx_t0)
        parts_prior.append(prior_t0)
        parts_source.append(torch.full_like(idx_t0, 3))

    all_idx = torch.cat(parts_idx, dim=1)
    all_prior = torch.cat(parts_prior, dim=1).clamp_min(0.0)
    all_source = torch.cat(parts_source, dim=1)
    if int(max_candidates) > 0 and all_idx.shape[1] > int(max_candidates):
        score, order = torch.topk(all_prior, k=int(max_candidates), dim=1, largest=True, sorted=True)
        all_idx = torch.gather(all_idx, 1, order)
        all_source = torch.gather(all_source, 1, order)
        all_prior = score
    valid = all_prior > 0
    all_prior = all_prior / all_prior.sum(1, keepdim=True).clamp_min(1.0e-8)
    xyz = target[all_idx.reshape(-1)].reshape(all_idx.shape[0], all_idx.shape[1], 3)
    return {
        "candidate_target_xyz": xyz,
        "candidate_target_idx": all_idx,
        "candidate_fm_prior": all_prior,
        "candidate_valid": valid,
        "candidate_source_id": all_source,
    }


def correspondence_estep(
    source: torch.Tensor,
    T: torch.Tensor,
    candidate_xyz: torch.Tensor,
    candidate_prior: torch.Tensor,
    candidate_valid: torch.Tensor,
    fm_endpoint_xyz: torch.Tensor,
    *,
    spcd: float,
    neighbor_idx: torch.Tensor | None = None,
    neighbor_dist0: torch.Tensor | None = None,
    neighbor_target_xyz: torch.Tensor | None = None,
    isometry_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    pred = transform_points(T, source)
    geo = torch.linalg.norm(pred[:, None] - candidate_xyz, dim=-1)
    fm_spread = torch.linalg.norm(candidate_xyz - fm_endpoint_xyz[:, None], dim=-1)
    median = fm_spread.median(dim=1, keepdim=True).values
    mad = (fm_spread - median).abs().median(dim=1).values
    sigma = torch.maximum(
        torch.full_like(mad, max(float(spcd), 1.0e-8)),
        mad.clamp_min(1.0e-8),
    )
    likelihood = torch.exp(-0.5 * geo.square() / sigma[:, None].square().clamp_min(1.0e-12))
    if (
        neighbor_idx is not None
        and neighbor_dist0 is not None
        and neighbor_target_xyz is not None
        and int(neighbor_idx.shape[1]) > 0
        and float(isometry_weight) > 0.0
    ):
        # Does proposing THIS candidate as i's target, alongside whatever i's t0
        # neighbours are currently proposed to match, preserve their t0 pairwise
        # distances?  A single shared rigid transform preserves these exactly,
        # so a candidate that stretches/compresses them is evidence it belongs
        # to a different part (often: nearby static structure) rather than
        # evidence about the transform itself.
        dT = torch.linalg.norm(
            candidate_xyz[:, :, None, :] - neighbor_target_xyz[:, None, :, :], dim=-1
        )
        iso_err = dT - neighbor_dist0[:, None, :]
        iso_cost = iso_err.square().mean(dim=-1)
        iso_sigma2 = max(float(spcd), 1.0e-8) ** 2
        likelihood = likelihood * torch.exp(-0.5 * float(isometry_weight) * iso_cost / iso_sigma2)
    posterior_raw = candidate_prior.clamp_min(0.0) * likelihood * candidate_valid.to(likelihood.dtype)
    mass = posterior_raw.sum(1, keepdim=True)
    # When the current transform sits far enough from EVERY candidate that
    # all of a point's likelihoods underflow together, `mass` is not merely
    # small -- it is no longer numerically meaningful.  Dividing by a fixed
    # floor (as this used to do) does not renormalise it back to summing to
    # one; it silently manufactures a near-zero "posterior" that then
    # collapses `expected_target` toward the coordinate origin (verified:
    # this produced sub-metre-to-multi-metre-wrong EM targets on real data,
    # which the IRLS refit then fit toward, corrupting the whole rigid
    # transform on the very first EM iteration). Fall back to a uniform
    # prior over the valid candidates instead -- "no discriminating
    # evidence" should mean "don't know", not "answer is the origin, with
    # full confidence".
    no_evidence = mass <= torch.finfo(mass.dtype).tiny * 10.0
    uniform = candidate_valid.to(likelihood.dtype)
    uniform = uniform / uniform.sum(1, keepdim=True).clamp_min(1.0e-8)
    posterior = torch.where(no_evidence, uniform, posterior_raw / mass.clamp_min(torch.finfo(mass.dtype).tiny))
    expected = (posterior[..., None] * candidate_xyz).sum(1)
    entropy = -(posterior * torch.log(posterior.clamp_min(1.0e-12))).sum(1)
    valid_count = candidate_valid.sum(1).clamp_min(1)
    denom = torch.log(valid_count.to(entropy.dtype).clamp_min(2.0))
    corr_conf = torch.where(valid_count > 1, 1.0 - entropy / denom, torch.ones_like(entropy))
    corr_conf = corr_conf.clamp(0.0, 1.0)
    selected = posterior.argmax(1)
    selected_xyz = candidate_xyz[torch.arange(source.shape[0], device=source.device), selected]
    selected_residual = torch.linalg.norm(pred - selected_xyz, dim=-1)
    return {
        "posterior": posterior,
        "expected_target": expected,
        "corr_confidence": corr_conf,
        "entropy": entropy,
        "selected": selected,
        "selected_target_xyz": selected_xyz,
        "selected_residual": selected_residual,
        "min_candidate_residual": geo.masked_fill(~candidate_valid, float("inf")).min(1).values,
    }


def _sample_candidate_targets(
    candidate_idx: torch.Tensor,
    candidate_prior: torch.Tensor,
    source_ids: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    weights = candidate_prior[source_ids].clamp_min(0.0)
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1.0e-8)
    return torch.multinomial(weights, num_samples=1, replacement=True, generator=generator).reshape(-1)


def ransac_candidate_se3(
    source: torch.Tensor,
    candidate_xyz: torch.Tensor,
    candidate_idx: torch.Tensor,
    candidate_prior: torch.Tensor,
    candidate_valid: torch.Tensor,
    fm_endpoint_xyz: torch.Tensor,
    seed_mask: torch.Tensor,
    dynamic_confidence: torch.Tensor,
    *,
    threshold: float,
    spcd: float,
    iters: int,
    seed: int,
    print_fn=print,
) -> dict[str, Any]:
    seed_mask = torch.as_tensor(seed_mask, device=source.device).bool()
    dyn = torch.as_tensor(dynamic_confidence, device=source.device, dtype=source.dtype).clamp_min(0.0)
    ids = torch.nonzero(seed_mask & (dyn > 0), as_tuple=False).reshape(-1)
    if int(ids.numel()) < 3:
        raise RuntimeError("candidate RANSAC needs at least three high-confidence dynamic seeds")
    generator = torch.Generator(device=source.device).manual_seed(int(seed))
    weights = dyn[ids] / dyn[ids].sum().clamp_min(1.0e-8)
    best = None
    for trial in range(max(1, int(iters))):
        local = torch.multinomial(weights, num_samples=3, replacement=False, generator=generator)
        src_ids = ids[local]
        cand_draw = _sample_candidate_targets(candidate_idx, candidate_prior, src_ids, generator)
        y = candidate_xyz[src_ids, cand_draw]
        try:
            fit = weighted_kabsch(source[src_ids], y)
        except ValueError:
            continue
        estep = correspondence_estep(
            source, fit["T"], candidate_xyz, candidate_prior, candidate_valid,
            fm_endpoint_xyz, spcd=spcd,
        )
        residual = estep["min_candidate_residual"]
        inlier = seed_mask & (residual <= float(threshold))
        mass = dyn[inlier] * estep["corr_confidence"][inlier]
        score = float((mass * (1.0 - residual[inlier] / max(float(threshold), 1.0e-8)).clamp_min(0.0)).sum().detach().cpu())
        if best is None or score > best["score"]:
            selected_target = candidate_idx[
                torch.arange(source.shape[0], device=source.device), estep["selected"]
            ]
            collision = selected_target_collision_stats(candidate_idx, estep["selected"], inlier)
            unique = int(collision["unique_selected_targets"])
            summary = {
                "trial": int(trial),
                "seed_source_count": int(ids.numel()),
                "inliers": int(inlier.sum()),
                "seed_points": int(ids.numel()),
                "weighted_inlier_mass": float(mass.sum().detach().cpu()) if mass.numel() else 0.0,
                "threshold": float(threshold),
                "median_residual": float(residual[inlier].median().detach().cpu()) if bool(inlier.any()) else float("inf"),
                "p90_residual": float(torch.quantile(residual[inlier], 0.9).detach().cpu()) if int(inlier.sum()) > 1 else float("inf"),
                "candidate_entropy_med": float(estep["entropy"][inlier].median().detach().cpu()) if bool(inlier.any()) else float("inf"),
                "unique_target_ratio": float(unique / max(1, int(inlier.sum()))),
                "maximum_target_multiplicity": collision["maximum_target_multiplicity"],
                "p90_target_multiplicity": collision["p90_target_multiplicity"],
                **transform_summary(fit["T"]),
                "T_matrix": fit["T"].detach().cpu().tolist(),
            }
            print_fn(
                "[RANSAC_BEST]\n"
                f"trial                 = {summary['trial']}\n"
                f"seed source count     = {summary['seed_source_count']}\n"
                f"inliers / seed points = {summary['inliers']} / {summary['seed_points']}\n"
                f"weighted inlier mass  = {summary['weighted_inlier_mass']:.6g}\n"
                f"threshold             = {summary['threshold']:.6g}\n"
                f"median residual       = {summary['median_residual']:.6g}\n"
                f"p90 residual          = {summary['p90_residual']:.6g}\n"
                f"rotation angle deg    = {summary['rotation_angle_deg']:.6g}\n"
                f"rotation axis         = {summary['rotation_axis']}\n"
                f"translation           = {summary['translation']}\n"
                f"det(R)                = {summary['det_R']:.6g}\n"
                f"candidate entropy med = {summary['candidate_entropy_med']:.6g}\n"
                f"unique target ratio   = {summary['unique_target_ratio']:.6g}\n"
                f"maximum target multiplicity = {summary['maximum_target_multiplicity']}\n"
                f"p90 target multiplicity     = {summary['p90_target_multiplicity']:.6g}"
            )
            best = {"T": fit["T"], "score": score, "inlier": inlier, "summary": summary}
    if best is None:
        raise RuntimeError("candidate RANSAC found no transform")
    return best


def _candidate_geometry_objective(
    estep: dict[str, torch.Tensor],
    support: torch.Tensor,
    dynamic_confidence: torch.Tensor,
    *,
    huber_delta: float,
) -> torch.Tensor:
    residual = estep["min_candidate_residual"]
    delta = max(float(huber_delta), 1.0e-8)
    robust = torch.where(
        residual <= delta,
        0.5 * residual.square() / delta,
        residual - 0.5 * delta,
    )
    weight = (
        dynamic_confidence.clamp(0.0, 1.0)
        * estep["corr_confidence"].clamp(0.0, 1.0)
        * support.to(residual.dtype)
    )
    return (weight * robust).sum() / weight.sum().clamp_min(1.0e-8)


def selected_target_collision_stats(candidate_idx: torch.Tensor, selected: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    ids = torch.nonzero(mask.bool(), as_tuple=False).reshape(-1)
    if int(ids.numel()) == 0:
        return {
            "unique_selected_targets": 0,
            "total_source_matches": 0,
            "unique_target_ratio": 0.0,
            "maximum_target_multiplicity": 0,
            "p90_target_multiplicity": 0.0,
        }
    target_ids = candidate_idx[ids, selected[ids]]
    unique, counts = torch.unique(target_ids, return_counts=True)
    return {
        "unique_selected_targets": int(unique.numel()),
        "total_source_matches": int(ids.numel()),
        "unique_target_ratio": float(unique.numel() / max(1, int(ids.numel()))),
        "maximum_target_multiplicity": int(counts.max()) if counts.numel() else 0,
        "p90_target_multiplicity": float(torch.quantile(counts.float(), 0.9)) if counts.numel() > 1 else float(counts.max()) if counts.numel() else 0.0,
    }


def fm_guided_correspondence_em(
    source: torch.Tensor,
    candidate_xyz: torch.Tensor,
    candidate_idx: torch.Tensor,
    candidate_prior: torch.Tensor,
    candidate_valid: torch.Tensor,
    fm_endpoint_xyz: torch.Tensor,
    support_mask: torch.Tensor,
    dynamic_confidence: torch.Tensor,
    *,
    init_T: torch.Tensor,
    spcd: float,
    irls_iters: int,
    huber_delta: float,
    em_iters: int,
    neighbor_idx: torch.Tensor | None = None,
    neighbor_dist0: torch.Tensor | None = None,
    isometry_weight: float = 0.0,
    print_fn=print,
) -> dict[str, Any]:
    support = torch.as_tensor(support_mask, device=source.device).bool()
    if int(support.sum()) < 3:
        raise RuntimeError("correspondence EM needs at least three support points")
    dyn = torch.as_tensor(dynamic_confidence, device=source.device, dtype=source.dtype).clamp(0.0, 1.0)
    T = init_T.to(device=source.device, dtype=source.dtype).clone()
    use_isometry = (
        neighbor_idx is not None
        and neighbor_dist0 is not None
        and int(neighbor_idx.shape[1]) > 0
        and float(isometry_weight) > 0.0
    )
    # Gauss-Seidel style: each iteration's isometry term is checked against the
    # PREVIOUS iteration's selected targets for a point's t0 neighbours, then
    # this iteration's own selection becomes the reference for the next one.
    neighbor_ref = transform_points(T, source) if use_isometry else None

    def _neighbor_targets() -> torch.Tensor | None:
        return neighbor_ref[neighbor_idx] if use_isometry else None

    previous_selected = None
    history = []
    best_estep = correspondence_estep(
        source, T, candidate_xyz, candidate_prior, candidate_valid,
        fm_endpoint_xyz, spcd=spcd,
        neighbor_idx=neighbor_idx, neighbor_dist0=neighbor_dist0,
        neighbor_target_xyz=_neighbor_targets(), isometry_weight=isometry_weight,
    )
    best_objective = _candidate_geometry_objective(
        best_estep, support, dyn, huber_delta=float(huber_delta)
    )
    best_T = T.clone()
    for iteration in range(max(1, int(em_iters))):
        estep = correspondence_estep(
            source, T, candidate_xyz, candidate_prior, candidate_valid,
            fm_endpoint_xyz, spcd=spcd,
            neighbor_idx=neighbor_idx, neighbor_dist0=neighbor_dist0,
            neighbor_target_xyz=_neighbor_targets(), isometry_weight=isometry_weight,
        )
        selected = estep["selected"]
        changed = int((selected[support] != previous_selected[support]).sum()) if previous_selected is not None else int(support.sum())
        robust_current = huber_weight(estep["min_candidate_residual"], huber_delta)
        weight = dyn * estep["corr_confidence"] * robust_current * support.to(source.dtype)
        if int((weight > 0).sum()) < 3:
            raise RuntimeError("correspondence EM lost rigid support")
        fit = robust_fit_se3_irls(
            source,
            estep["expected_target"],
            weight,
            init_T=T,
            iters=int(irls_iters),
            huber_delta=float(huber_delta),
        )
        new_T = fit["T"].to(device=source.device, dtype=source.dtype)
        next_estep = correspondence_estep(
            source, new_T, candidate_xyz, candidate_prior, candidate_valid,
            fm_endpoint_xyz, spcd=spcd,
            neighbor_idx=neighbor_idx, neighbor_dist0=neighbor_dist0,
            neighbor_target_xyz=_neighbor_targets(), isometry_weight=isometry_weight,
        )
        objective = _candidate_geometry_objective(
            next_estep, support, dyn, huber_delta=float(huber_delta)
        )
        if bool(objective > best_objective + torch.finfo(source.dtype).eps * 10.0):
            print_fn(
                "[CORR_EM_ROLLBACK]\n"
                f"iteration             = {iteration}\n"
                f"previous objective    = {float(best_objective.detach().cpu()):.6g}\n"
                f"proposed objective    = {float(objective.detach().cpu()):.6g}\n"
                "candidate geometry worsened; keeping previous transform"
            )
            break
        residual = next_estep["min_candidate_residual"]
        dT = new_T @ torch.linalg.inv(T)
        residual_support = residual[support]
        collision = selected_target_collision_stats(candidate_idx, next_estep["selected"], support)
        row = {
            "iteration": int(iteration),
            "active_moving_seeds": int(support.sum()),
            "mean_corr_confidence": float(next_estep["corr_confidence"][support].mean().detach().cpu()),
            "median_corr_confidence": float(next_estep["corr_confidence"][support].median().detach().cpu()),
            "candidate_changed": int(changed),
            "candidate_changed_fraction": float(changed / max(1, int(support.sum()))),
            **collision,
            "candidate_geometry_objective": float(objective.detach().cpu()),
            "residual_mean": float(residual_support.mean().detach().cpu()),
            "residual_median": float(residual_support.median().detach().cpu()),
            "residual_p90": float(torch.quantile(residual_support, 0.9).detach().cpu()) if residual_support.numel() > 1 else float(residual_support.mean().detach().cpu()),
            "residual_max": float(residual_support.max().detach().cpu()),
            "T_delta": transform_summary(dT),
            "current_transform": transform_summary(new_T),
        }
        if use_isometry:
            selected_xyz = next_estep["selected_target_xyz"]
            neighbor_targets_now = selected_xyz[neighbor_idx]
            drift = (torch.linalg.norm(selected_xyz[:, None, :] - neighbor_targets_now, dim=-1) - neighbor_dist0).abs()
            drift_support = drift[support]
            row["neighbor_distance_drift_mean"] = float(drift_support.mean().detach().cpu())
            row["neighbor_distance_drift_median"] = float(drift_support.median().detach().cpu())
        history.append(row)
        print_fn(
            f"[CORR_EM {iteration:03d}]\n"
            f"active moving seeds      = {row['active_moving_seeds']}\n"
            f"mean corr confidence     = {row['mean_corr_confidence']:.6g}\n"
            f"median corr confidence   = {row['median_corr_confidence']:.6g}\n"
            f"candidate changed        = {row['candidate_changed']} / {row['active_moving_seeds']}\n"
            f"candidate changed frac   = {row['candidate_changed_fraction']:.6g}\n"
            f"unique target count      = {row['unique_selected_targets']}\n"
            f"unique target ratio      = {row['unique_target_ratio']:.6g}\n"
            f"maximum target multiplicity = {row['maximum_target_multiplicity']}\n"
            f"p90 target multiplicity     = {row['p90_target_multiplicity']:.6g}\n"
            f"candidate geometry objective = {row['candidate_geometry_objective']:.6g}\n"
            f"rigid residual:\n"
            f"    mean   = {row['residual_mean']:.6g}\n"
            f"    median = {row['residual_median']:.6g}\n"
            f"    p90    = {row['residual_p90']:.6g}\n"
            f"    max    = {row['residual_max']:.6g}\n"
            + (
                "neighbour distance drift (|t0 dist - T dist| to t0 neighbours):\n"
                f"    mean   = {row['neighbor_distance_drift_mean']:.6g}\n"
                f"    median = {row['neighbor_distance_drift_median']:.6g}\n"
                if use_isometry else ""
            )
            + f"T delta:\n"
            f"    rotation deg = {row['T_delta']['rotation_angle_deg']:.6g}\n"
            f"    translation  = {row['T_delta']['translation']}\n"
            f"current rotation:\n"
            f"    angle deg = {row['current_transform']['rotation_angle_deg']:.6g}\n"
            f"    axis      = {row['current_transform']['rotation_axis']}\n"
            f"    t         = {row['current_transform']['translation']}"
        )
        T = new_T
        best_T = new_T.clone()
        best_estep = next_estep
        best_objective = objective
        previous_selected = next_estep["selected"].detach().clone()
        if use_isometry:
            neighbor_ref = next_estep["selected_target_xyz"].detach().clone()
    estep = best_estep
    T = best_T
    residual = estep["min_candidate_residual"]
    weight = dyn * estep["corr_confidence"] * huber_weight(residual, huber_delta) * support.to(source.dtype)
    return {
        "T": T,
        "xi": se3_log(T),
        "history": history,
        "posterior": estep["posterior"].detach(),
        "expected_target": estep["expected_target"].detach(),
        "corr_confidence": estep["corr_confidence"].detach(),
        "selected": estep["selected"].detach(),
        "selected_target_xyz": estep["selected_target_xyz"].detach(),
        "residual_per_point": residual.detach(),
        "support_weight": weight.detach(),
        "collision": selected_target_collision_stats(candidate_idx, estep["selected"], support),
    }


def ransac_fit_se3(
    source: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    cfg: RansacConfig,
) -> dict[str, Any]:
    source = torch.as_tensor(source).float()
    target = torch.as_tensor(target, device=source.device, dtype=source.dtype)
    confidence = torch.as_tensor(confidence, device=source.device, dtype=source.dtype).reshape(-1).clamp_min(0.0)
    valid = (
        torch.isfinite(source).all(-1)
        & torch.isfinite(target).all(-1)
        & torch.isfinite(confidence)
        & (confidence > 0)
    )
    ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if int(ids.numel()) < int(cfg.min_samples):
        raise RuntimeError("RANSAC SE(3) has fewer than three valid matches")

    generator = torch.Generator(device=source.device).manual_seed(int(cfg.seed))
    order = ids[torch.argsort(confidence[ids], descending=True)]
    prosac_count = max(int(cfg.min_samples), int(math.ceil(float(order.numel()) * float(cfg.prosac_fraction))))
    pool = order[:prosac_count]
    best = None
    threshold = float(cfg.inlier_threshold)
    for iteration in range(max(1, int(cfg.iters))):
        if int(pool.numel()) < int(cfg.min_samples):
            sample = order[: int(cfg.min_samples)]
        else:
            perm = torch.randperm(int(pool.numel()), generator=generator, device=source.device)
            sample = pool[perm[: int(cfg.min_samples)]]
        try:
            candidate = weighted_kabsch(source[sample], target[sample])
        except ValueError:
            continue
        residual = torch.linalg.norm(transform_points(candidate["T"], source) - target, dim=-1)
        inlier = valid & (residual <= threshold)
        score = float((confidence[inlier] * (1.0 - residual[inlier] / max(threshold, 1.0e-8)).clamp_min(0.0)).sum().detach().cpu())
        if best is None or score > best["score"]:
            best = {
                "T": candidate["T"],
                "residual": residual,
                "inlier": inlier,
                "score": score,
                "iteration": int(iteration),
            }
    if best is None:
        raise RuntimeError("RANSAC SE(3) found no usable hypothesis")
    inlier_weight = confidence * best["inlier"].to(confidence.dtype)
    refined = robust_fit_se3_irls(
        source, target, inlier_weight,
        init_T=best["T"], iters=int(cfg.irls_iters), huber_delta=float(cfg.huber_delta),
    )
    refined["inlier"] = torch.as_tensor(refined["residual_per_point"], device=source.device) <= threshold
    refined["ransac_score"] = best["score"]
    refined["ransac_iteration"] = best["iteration"]
    refined["inlier_weight"] = inlier_weight.detach()
    return refined


def relative_distance_rigid_weights(
    source: torch.Tensor,
    target: torch.Tensor,
    knn_idx: torch.Tensor,
    match_idx: torch.Tensor | None = None,
    *,
    sigma: float = 0.02,
) -> torch.Tensor:
    """Edge weights high only when local relative distances are preserved."""
    sigma = max(float(sigma), 1.0e-8)
    source = torch.as_tensor(source).float()
    target = torch.as_tensor(target, device=source.device, dtype=source.dtype)
    idx = torch.as_tensor(knn_idx, device=source.device).long()
    if match_idx is None:
        matched = target
    else:
        matched = target[torch.as_tensor(match_idx, device=source.device).long()]
    src_nei = source[idx]
    tgt_nei = matched[idx]
    ds = torch.linalg.norm(source[:, None] - src_nei, dim=-1)
    dt = torch.linalg.norm(matched[:, None] - tgt_nei, dim=-1)
    return torch.exp(-0.5 * ((ds - dt) / sigma).square()).clamp(0.0, 1.0)
