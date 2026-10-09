"""Robust K-part inner support construction.

CODEX_MERGED_ROBUST_SUPPORT_AND_DETERMINISTIC_LOSS_ROUTING, Parts I-VII.

    soft outer labels P
        -> hard dynamic candidate gate            (§2)
        -> conservative spatial outlier rejection (§5, §6, §7)
        -> true high-confidence core              (§8, §9, §10)
        -> confidence-weighted KNN graph          (§11)
        -> exactly K moving graph clusters        (§12)
        -> cluster-ID alignment                   (§13)
        -> conservative attachment                (§15, §16, §17)
        -> explicit unassigned state              (§18)
        -> fitting supports F_k / transport T_k   (§19, §20, §21)
        -> continuous fit weights                 (§22)

The point of the whole module is the single invariant F_k SUBSET-OF T_k (§86):
a Gaussian may be *moved* on the strength of a weak outer label, but it may only
*define the joint* if it survived every gate. Everything downstream consumes
``InnerSupportResult`` and is forbidden from rebuilding masks from hard labels
(§26, §27).

Persistent outer state is never mutated (§1): every tensor here is derived, and
the builder only ever reads ``soft_part_probs`` and ``hard_labels``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

EPS = 1.0e-8

#: §10's relaxation schedule for the core dynamic-confidence threshold.
CORE_CONFIDENCE_SCHEDULE = (0.80, 0.75, 0.70, 0.65)


@dataclass
class InnerSupportResult:
    """§24. Every field is [N]-aligned with the canonical Gaussian order."""

    transport_masks: torch.Tensor          # [K, N] bool
    fitting_masks: torch.Tensor            # [K, N] bool

    transport_part_id: torch.Tensor        # [N] long, 0 static, 1..K moving
    fitting_part_id: torch.Tensor          # [N] long, -1 excluded, 1..K fitted

    candidate_mask: torch.Tensor           # [N] bool
    outlier_mask: torch.Tensor             # [N] bool
    core_mask: torch.Tensor                # [N] bool
    attached_mask: torch.Tensor            # [N] bool
    unassigned_mask: torch.Tensor          # [N] bool

    fit_weight: torch.Tensor               # [N]
    dynamic_confidence: torch.Tensor       # [N]
    part_confidence: torch.Tensor          # [N, K]

    local_spacing: torch.Tensor            # [N]
    local_dynamic_support: torch.Tensor    # [N]

    effective_thresholds: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def num_parts(self) -> int:
        return int(self.fitting_masks.shape[0])

    def fitting_index(self, part: int) -> torch.Tensor:
        """Indices of F_k, with ``part`` 0-based."""
        return torch.nonzero(self.fitting_masks[part], as_tuple=False).reshape(-1)

    def valid(self, part: int, minimum: int) -> bool:
        return int(self.fitting_masks[part].sum()) >= int(minimum)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _knn_within(points: torch.Tensor, k: int, *, chunk: int = 2048):
    """KNN restricted to the given subset (§5). Returns (idx, dist), self excluded."""
    m = int(points.shape[0])
    if m <= 1:
        return (points.new_zeros((m, 0), dtype=torch.long), points.new_zeros((m, 0)))
    kk = min(int(k), m - 1)
    idx_all, dist_all = [], []
    for start in range(0, m, int(chunk)):
        block = points[start : start + int(chunk)]
        d = torch.cdist(block, points)
        # exclude self
        rows = torch.arange(block.shape[0], device=points.device)
        d[rows, rows + start] = float("inf")
        dist, idx = torch.topk(d, kk, dim=1, largest=False)
        idx_all.append(idx)
        dist_all.append(dist)
    return torch.cat(idx_all), torch.cat(dist_all)


def _mad(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    median = values.median()
    return median, 1.4826 * (values - median).abs().median()


def _percentiles(values: torch.Tensor) -> dict[str, float]:
    if int(values.numel()) == 0:
        return {p: float("nan") for p in ("p10", "p25", "median", "p75", "p90")}
    q = torch.quantile(values.double(),
                       torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=torch.float64,
                                    device=values.device))
    return {"p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4])}


def _spectral_partition(
    affinity: torch.Tensor, k: int, *, seed: int, eps: float = EPS
) -> torch.Tensor:
    """§12: normalised-cut spectral embedding + deterministic K partition.

    Euclidean K-means in XYZ is explicitly forbidden (§12) because it cuts a
    connected elongated part in half whenever that part is longer than the gap
    to a different one. The normalised Laplacian's low eigenvectors follow graph
    CONNECTIVITY, which is what "one rigid part" actually means.
    """
    m = int(affinity.shape[0])
    if k <= 1 or m <= k:
        return torch.zeros(m, dtype=torch.long, device=affinity.device)

    degree = affinity.sum(dim=1)
    d_inv_sqrt = degree.clamp_min(eps).rsqrt()
    normalised = affinity * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    normalised = 0.5 * (normalised + normalised.transpose(0, 1))
    try:
        _, vectors = torch.linalg.eigh(normalised.double())
    except Exception:
        return torch.zeros(m, dtype=torch.long, device=affinity.device)
    embedding = vectors[:, -k:]
    embedding = embedding / embedding.norm(dim=1, keepdim=True).clamp_min(eps)

    # Deterministic k-means++ seeding on the spectral embedding, then Lloyd.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    centres = [int(torch.randint(m, (1,), generator=generator).item())]
    for _ in range(k - 1):
        d = torch.cdist(embedding, embedding[centres]).min(dim=1).values
        centres.append(int(d.argmax()))
    centroids = embedding[centres].clone()
    assignment = torch.zeros(m, dtype=torch.long, device=affinity.device)
    for _ in range(50):
        assignment = torch.cdist(embedding, centroids).argmin(dim=1)
        for c in range(k):
            members = assignment == c
            if bool(members.any()):
                centroids[c] = embedding[members].mean(dim=0)
    # §12 requires exactly K NON-EMPTY clusters: re-seed any empty cluster with
    # the point worst explained by its current centroid.
    for c in range(k):
        if bool((assignment == c).any()):
            continue
        d = (embedding - centroids[assignment]).norm(dim=1)
        # never steal the last member of another cluster
        counts = torch.bincount(assignment, minlength=k)
        d = torch.where(counts[assignment] > 1, d, torch.full_like(d, -1.0))
        victim = int(d.argmax())
        assignment[victim] = c
        centroids[c] = embedding[victim]
    return assignment


def _align_clusters_to_parts(
    cluster_id: torch.Tensor, part_probs: torch.Tensor, k: int
) -> tuple[torch.Tensor, list[list[float]]]:
    """§13: Hungarian alignment of permutation-invariant cluster labels to part
    IDs. This changes IDs only -- never membership."""
    m_matrix = torch.zeros((k, k), dtype=torch.float64)
    for a in range(k):
        members = cluster_id == a
        if bool(members.any()):
            m_matrix[a] = part_probs[members].sum(dim=0).double().cpu()
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-m_matrix.numpy())
        mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    except Exception:
        # Deterministic greedy fallback, still one-to-one.
        mapping, used = {}, set()
        order = torch.argsort(m_matrix.reshape(-1), descending=True)
        for flat in order.tolist():
            a, c = divmod(flat, k)
            if a in mapping or c in used:
                continue
            mapping[a] = c
            used.add(c)
        for a in range(k):
            if a not in mapping:
                mapping[a] = next(c for c in range(k) if c not in used)
                used.add(mapping[a])
    aligned = torch.full_like(cluster_id, -1)
    for a, c in mapping.items():
        aligned[cluster_id == a] = c
    return aligned, m_matrix.tolist()


# ---------------------------------------------------------------------------
# §25: the builder
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_robust_inner_support(
    xyz: torch.Tensor,
    soft_part_probs: torch.Tensor,
    hard_labels: torch.Tensor | None,
    num_moving_parts: int,
    *,
    knn: int = 16,
    outlier_mad_scale: float = 3.5,
    outlier_local_dynamic_min: float = 0.35,
    core_dynamic_confidence: float = 0.80,
    core_part_confidence: float = 0.70,
    core_confidence_floor: float = 0.60,
    core_min_points: int | None = None,
    cluster_knn: int = 16,
    cluster_dynamic_gamma: float = 1.0,
    cluster_part_gamma: float = 0.5,
    attach_min_confidence: float = 0.55,
    attach_margin_ratio: float = 1.5,
    attach_affinity_quantile: float = 0.25,
    fit_confidence_floor: float = 0.50,
    fit_confidence_gamma: float = 2.0,
    fit_part_gamma: float = 1.0,
    seed: int = 20260919,
    eps: float = EPS,
) -> InnerSupportResult:
    """Build F_k and T_k from the outer soft labels without ever mutating them."""
    x = torch.as_tensor(xyz)
    device, dtype = x.device, x.dtype
    probs = torch.as_tensor(soft_part_probs, device=device, dtype=dtype)
    n = int(x.shape[0])
    k = int(num_moving_parts)
    if k < 1:
        raise ValueError(f"num_moving_parts must be >= 1, got {k}")
    if int(probs.shape[1]) != k + 1:
        raise ValueError(f"soft_part_probs must be [N, K+1] with K={k}, got {tuple(probs.shape)}")

    # §2: temporary hard gate. Persistent P is read only.
    z = (probs.argmax(dim=-1) if hard_labels is None
         else torch.as_tensor(hard_labels, device=device).reshape(-1).long())
    c_dyn = (1.0 - probs[:, 0]).clamp(0.0, 1.0)              # §2
    p_mov = probs[:, 1:]                                      # §2
    candidate = z > 0                                         # §2: D
    cand_idx = torch.nonzero(candidate, as_tuple=False).reshape(-1)

    local_spacing = torch.zeros(n, device=device, dtype=dtype)
    local_dyn = torch.zeros(n, device=device, dtype=dtype)
    outlier = torch.zeros(n, dtype=torch.bool, device=device)
    core = torch.zeros(n, dtype=torch.bool, device=device)
    attached = torch.zeros(n, dtype=torch.bool, device=device)
    fitting_part_id = torch.full((n,), -1, dtype=torch.long, device=device)

    thresholds: dict[str, Any] = {
        "core_dynamic_confidence": float(core_dynamic_confidence),
        "core_part_confidence": float(core_part_confidence),
        "attach_min_confidence": float(attach_min_confidence),
        "attach_margin_ratio": float(attach_margin_ratio),
        "outlier_mad_scale": float(outlier_mad_scale),
        "outlier_local_dynamic_min": float(outlier_local_dynamic_min),
    }
    diagnostics: dict[str, Any] = {"candidate_count": int(candidate.sum())}

    if int(cand_idx.numel()) >= 2:
        pts = x.index_select(0, cand_idx)
        nbr, dist = _knn_within(pts, knn)                     # §5

        # §5: r_i = median KNN distance, then a MAD cut on it.
        r = dist.median(dim=1).values
        local_spacing[cand_idx] = r
        r_med, r_mad = _mad(r)
        spacing_cut = float(r_med + float(outlier_mad_scale) * r_mad)
        isolated = r > spacing_cut

        # §6: local dynamic support, on the SAME neighbourhood.
        sigma_x = float(r_med.clamp_min(eps))
        w_space = torch.exp(-(dist.square()) / (2.0 * sigma_x * sigma_x + eps))
        c_nbr = c_dyn.index_select(0, cand_idx)[nbr]
        d_local = (w_space * c_nbr).sum(dim=1) / (w_space.sum(dim=1) + eps)
        local_dyn[cand_idx] = d_local
        weak = d_local < float(outlier_local_dynamic_min)

        # §6: BOTH conditions. Spacing alone never rejects a thin structure.
        is_outlier = isolated & weak
        outlier[cand_idx] = is_outlier
        thresholds.update({"spacing_median": float(r_med), "spacing_mad": float(r_mad),
                           "spacing_cut": spacing_cut, "sigma_x": sigma_x})
        diagnostics.update({
            "isolated_count": int(isolated.sum()),
            "weak_support_count": int(weak.sum()),
            "outlier_count": int(is_outlier.sum()),
        })

    clean = candidate & ~outlier                              # §7
    clean_idx = torch.nonzero(clean, as_tuple=False).reshape(-1)
    max_part = p_mov.max(dim=1).values                        # §8: m_i

    # §9: the confidence distribution must be visible BEFORE clustering, so a
    # core that happens to equal the clean set can be proven rather than assumed.
    diagnostics["dynamic_confidence"] = _percentiles(c_dyn[clean_idx])
    diagnostics["max_part_confidence"] = _percentiles(max_part[clean_idx])
    for level in (0.50, 0.70, 0.80, 0.90):
        diagnostics[f"clean_dyn_above_{level:.2f}"] = int((c_dyn[clean_idx] > level).sum())
    diagnostics["clean_count"] = int(clean.sum())

    # §8 + §10: confidence-selected core, relaxed on a fixed schedule, never
    # below the floor, and NEVER silently replaced by the clean set.
    min_core = int(core_min_points if core_min_points is not None else max(24, 6 * k))
    schedule = [t for t in CORE_CONFIDENCE_SCHEDULE if t <= float(core_dynamic_confidence) + 1e-12]
    if not schedule or schedule[0] != float(core_dynamic_confidence):
        schedule = [float(core_dynamic_confidence)] + schedule
    schedule = [t for t in schedule if t >= float(core_confidence_floor) - 1e-12]
    if not schedule:
        schedule = [float(core_confidence_floor)]

    used_dyn, used_part = schedule[0], float(core_part_confidence)
    for level in schedule:
        part_level = min(float(core_part_confidence), level)
        selected = clean & (c_dyn >= level) & (max_part >= part_level)
        used_dyn, used_part = level, part_level
        if int(selected.sum()) >= min_core:
            break
    core = selected
    thresholds.update({"core_dynamic_used": float(used_dyn),
                       "core_part_used": float(used_part),
                       "core_min_points": min_core,
                       "core_schedule": [float(s) for s in schedule]})
    diagnostics["core_count"] = int(core.sum())
    # §9: if the core did not actually filter anything, prove it.
    diagnostics["core_equals_clean"] = bool(int(core.sum()) == int(clean.sum()))
    if diagnostics["core_equals_clean"] and int(clean.sum()) > 0:
        diagnostics["core_equals_clean_min_dynamic"] = float(c_dyn[clean].min())
        diagnostics["core_equals_clean_min_max_part"] = float(max_part[clean].min())
    # §10: an insufficient core is REPORTED, never widened to all candidates.
    diagnostics["support_low_confidence"] = bool(int(core.sum()) < min_core)

    core_idx = torch.nonzero(core, as_tuple=False).reshape(-1)
    cluster_to_part: list[list[float]] = []
    attach_cut = float("nan")

    if int(core_idx.numel()) > 0:
        core_pts = x.index_select(0, core_idx)
        m_core = int(core_idx.numel())
        kk = min(int(cluster_knn), max(m_core - 1, 1))
        sigma_x = float(thresholds.get("sigma_x", 1.0)) or 1.0

        # §11: w_ij = w_space * w_dyn * w_part, as a dense affinity on the core.
        d2 = torch.cdist(core_pts, core_pts).square()
        w_space = torch.exp(-d2 / (2.0 * sigma_x * sigma_x + eps))
        if m_core > 1:                       # symmetric KNN sparsification
            keep = torch.zeros_like(w_space, dtype=torch.bool)
            nn_idx = torch.topk(d2, min(kk + 1, m_core), dim=1, largest=False).indices
            keep.scatter_(1, nn_idx, True)
            keep = keep | keep.transpose(0, 1)
            keep.fill_diagonal_(False)
            w_space = w_space * keep.to(w_space.dtype)
        cd = c_dyn.index_select(0, core_idx)
        w_dyn = (cd[:, None] * cd[None, :]).clamp_min(0.0).pow(float(cluster_dynamic_gamma))
        if k > 1:
            pm = p_mov.index_select(0, core_idx)
            w_part = (pm @ pm.transpose(0, 1)).clamp_min(0.0).pow(float(cluster_part_gamma))
        else:
            w_part = torch.ones_like(w_dyn)   # §11: w_part = 1 for K = 1
        affinity = w_space * w_dyn * w_part

        # §12 + §13: real graph partition, then ID alignment only.
        raw_cluster = _spectral_partition(affinity, k, seed=seed)
        aligned, cluster_to_part = _align_clusters_to_parts(
            raw_cluster, p_mov.index_select(0, core_idx), k)
        fitting_part_id[core_idx] = aligned + 1
        diagnostics["cluster_sizes"] = [int((aligned == c).sum()) for c in range(k)]

        # §15-§17: conservative attachment of non-core clean candidates.
        rest = clean & ~core
        rest_idx = torch.nonzero(rest, as_tuple=False).reshape(-1)
        if int(rest_idx.numel()) > 0:
            rest_pts = x.index_select(0, rest_idx)
            d2r = torch.cdist(rest_pts, core_pts).square()
            ws = torch.exp(-d2r / (2.0 * sigma_x * sigma_x + eps))
            cr = c_dyn.index_select(0, rest_idx)
            wd = (cr[:, None] * cd[None, :]).clamp_min(0.0).pow(float(cluster_dynamic_gamma))
            if k > 1:
                wp = (p_mov.index_select(0, rest_idx)
                      @ p_mov.index_select(0, core_idx).transpose(0, 1)
                      ).clamp_min(0.0).pow(float(cluster_part_gamma))
            else:
                wp = torch.ones_like(wd)
            w_rest = ws * wd * wp                                    # [R, |C|]

            # §15: mean affinity per cluster, so cluster size cannot decide.
            a_bar = torch.zeros((int(rest_idx.numel()), k), device=device, dtype=dtype)
            for c in range(k):
                members = aligned == c
                size = float(members.sum())
                if size > 0:
                    a_bar[:, c] = w_rest[:, members].sum(dim=1) / (size + eps)

            best_val, best_c = a_bar.max(dim=1)
            if k > 1:
                second = a_bar.clone()
                second.scatter_(1, best_c[:, None], -1.0)
                second_val = second.max(dim=1).values.clamp_min(0.0)
                margin_ok = best_val / (second_val + eps) >= float(attach_margin_ratio)
            else:
                margin_ok = torch.ones_like(best_val, dtype=torch.bool)   # §17

            # §16: "the affinity threshold should be normalized/adaptive and
            # saved explicitly" -- a quantile of the CORE's own internal
            # affinity, so it means "as well connected as a core point is".
            core_self = torch.zeros(m_core, device=device, dtype=dtype)
            for c in range(k):
                members = aligned == c
                size = float(members.sum())
                if size > 0:
                    core_self[members] = affinity[members][:, members].sum(dim=1) / (size + eps)
            positive = core_self[core_self > 0]
            attach_cut = (float(torch.quantile(positive.double(),
                                               float(attach_affinity_quantile)))
                          if int(positive.numel()) > 0 else 0.0)
            ok = (best_val >= attach_cut) & margin_ok & (cr >= float(attach_min_confidence))
            take = rest_idx[ok]
            fitting_part_id[take] = best_c[ok] + 1
            attached[take] = True
        thresholds["attach_affinity_cut"] = attach_cut
        thresholds["attach_affinity_quantile"] = float(attach_affinity_quantile)

    # §18: everything clean that is neither core nor attached stays UNASSIGNED.
    # Not static, not deleted, just not trusted to define the joint.
    unassigned = clean & ~core & ~attached
    fitting_part_id = torch.where(core | attached, fitting_part_id,
                                  torch.full_like(fitting_part_id, -1))       # §19

    # §21: transport falls back to the outer hard label for anything excluded,
    # so an ambiguous Gaussian still MOVES without contaminating the fit.
    transport_part_id = torch.where(core | attached, fitting_part_id.clamp_min(0), z)
    transport_part_id = torch.where(candidate, transport_part_id,
                                    torch.zeros_like(transport_part_id))

    fitting_masks = torch.stack([fitting_part_id == (c + 1) for c in range(k)])   # §20
    transport_masks = torch.stack([transport_part_id == (c + 1) for c in range(k)])

    # §22: points inside F_k are NOT equally trusted.
    floor = float(fit_confidence_floor)
    w_label = ((c_dyn - floor) / max(1.0 - floor, eps)).clamp_min(0.0).pow(float(fit_confidence_gamma))
    part_gather = torch.zeros(n, device=device, dtype=dtype)
    for c in range(k):
        part_gather = torch.where(fitting_masks[c], p_mov[:, c], part_gather)
    fit_weight = w_label * part_gather.clamp_min(0.0).pow(float(fit_part_gamma))
    fit_weight = torch.where(fitting_part_id > 0, fit_weight, torch.zeros_like(fit_weight))

    diagnostics.update({
        "attached_count": int(attached.sum()),
        "unassigned_count": int(unassigned.sum()),
        "fitting_count": int((fitting_part_id > 0).sum()),
        "transport_count": int((transport_part_id > 0).sum()),
        "cluster_to_part_matrix": cluster_to_part,
        "per_part": [{
            "part": c + 1,
            "transport": int(transport_masks[c].sum()),
            "core": int((core & (fitting_part_id == c + 1)).sum()),
            "attached": int((attached & (fitting_part_id == c + 1)).sum()),
            "fitting": int(fitting_masks[c].sum()),
            "fallback_transport": int((transport_masks[c] & ~fitting_masks[c]).sum()),
            "mean_fit_weight": float(fit_weight[fitting_masks[c]].mean())
            if bool(fitting_masks[c].any()) else 0.0,
            "median_fit_weight": float(fit_weight[fitting_masks[c]].median())
            if bool(fitting_masks[c].any()) else 0.0,
        } for c in range(k)],
    })

    return InnerSupportResult(
        transport_masks=transport_masks, fitting_masks=fitting_masks,
        transport_part_id=transport_part_id, fitting_part_id=fitting_part_id,
        candidate_mask=candidate, outlier_mask=outlier, core_mask=core,
        attached_mask=attached, unassigned_mask=unassigned,
        fit_weight=fit_weight, dynamic_confidence=c_dyn, part_confidence=p_mov,
        local_spacing=local_spacing, local_dynamic_support=local_dyn,
        effective_thresholds=thresholds, diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# §63/§64: reporting
# ---------------------------------------------------------------------------
def format_support_summary(support: InnerSupportResult) -> str:
    """§63's console block."""
    d = support.diagnostics
    t = support.effective_thresholds
    dyn = d.get("dynamic_confidence", {})
    mp = d.get("max_part_confidence", {})

    def row(stats):
        return "/".join(f"{stats.get(key, float('nan')):.3f}"
                        for key in ("p10", "p25", "median", "p75", "p90"))

    lines = [
        "[InnerSupport]",
        f"  candidates: {d.get('candidate_count', 0)}   clean: {d.get('clean_count', 0)}   "
        f"outliers: {d.get('outlier_count', 0)}",
        f"  core: {d.get('core_count', 0)}   attached: {d.get('attached_count', 0)}   "
        f"unassigned: {d.get('unassigned_count', 0)}   fitting total: {d.get('fitting_count', 0)}",
        f"  confidence: dyn p10/p25/median/p75/p90 {row(dyn)}",
        f"              maxpart p10/p25/median/p75/p90 {row(mp)}",
        f"  thresholds: core dyn {t.get('core_dynamic_used', float('nan')):.2f}  "
        f"core part {t.get('core_part_used', float('nan')):.2f}  "
        f"attach conf {t.get('attach_min_confidence', float('nan')):.2f}  "
        f"attach affinity {t.get('attach_affinity_cut', float('nan')):.4g}",
    ]
    if d.get("support_low_confidence"):
        lines.append(f"  WARNING: core {d.get('core_count', 0)} < minimum "
                     f"{t.get('core_min_points')} -- support marked LOW CONFIDENCE "
                     "(no all-candidate fallback was applied)")
    if d.get("core_equals_clean"):
        lines.append(f"  NOTE: core == clean; every clean candidate exceeded both "
                     f"thresholds (min dyn {d.get('core_equals_clean_min_dynamic', float('nan')):.3f}, "
                     f"min maxpart {d.get('core_equals_clean_min_max_part', float('nan')):.3f})")
    for entry in d.get("per_part", []):
        lines.append(
            f"  part {entry['part']}: transport {entry['transport']}  core {entry['core']}  "
            f"attached {entry['attached']}  fitting {entry['fitting']}  "
            f"fallback {entry['fallback_transport']}  "
            f"fit w mean {entry['mean_fit_weight']:.3f} median {entry['median_fit_weight']:.3f}")
    return "\n".join(lines)


@torch.no_grad()
def gt_support_metrics(support: InnerSupportResult, gt_moving: torch.Tensor) -> dict:
    """§64. Evaluation only -- GT never enters support construction.

    The effect to look for is fitting precision > transport precision: the gates
    are supposed to buy purity for the joint fit at some cost in recall.
    """
    gt = torch.as_tensor(gt_moving, device=support.fit_weight.device).reshape(-1).bool()

    def pr(mask):
        m = mask.bool()
        tp = int((m & gt).sum())
        return {"count": int(m.sum()),
                "precision": tp / max(int(m.sum()), 1),
                "recall": tp / max(int(gt.sum()), 1)}

    out = {
        "transport": pr(support.transport_part_id > 0),
        "core": pr(support.core_mask),
        "fitting": pr(support.fitting_part_id > 0),
        "outlier_gt_moving_fraction": (float(gt[support.outlier_mask].float().mean())
                                       if bool(support.outlier_mask.any()) else 0.0),
        "unassigned_gt_moving_fraction": (float(gt[support.unassigned_mask].float().mean())
                                          if bool(support.unassigned_mask.any()) else 0.0),
    }
    out["fitting_precision_exceeds_transport"] = bool(
        out["fitting"]["precision"] > out["transport"]["precision"])
    return out


@torch.no_grad()
def write_support_debug_ply(support: InnerSupportResult, xyz: torch.Tensor, path) -> None:
    """§62's ``inner_support_debug.ply``: one colour per support state, so the
    gates can be inspected in a viewer instead of only in aggregate counts."""
    from pathlib import Path

    x = torch.as_tensor(xyz).detach().cpu()
    n = int(x.shape[0])
    colour = torch.zeros((n, 3), dtype=torch.uint8)
    colour[:] = torch.tensor([60, 60, 60], dtype=torch.uint8)            # static
    colour[support.candidate_mask.cpu()] = torch.tensor([90, 90, 160], dtype=torch.uint8)
    colour[support.unassigned_mask.cpu()] = torch.tensor([230, 200, 60], dtype=torch.uint8)
    colour[support.outlier_mask.cpu()] = torch.tensor([220, 50, 50], dtype=torch.uint8)
    colour[support.attached_mask.cpu()] = torch.tensor([60, 170, 230], dtype=torch.uint8)
    colour[support.core_mask.cpu()] = torch.tensor([50, 200, 80], dtype=torch.uint8)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"ply\nformat ascii 1.0\nelement vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property int fitting_part\nproperty int transport_part\n"
              "property float fit_weight\nproperty float dynamic_confidence\n"
              "end_header\n")
    fp = support.fitting_part_id.cpu()
    tp = support.transport_part_id.cpu()
    fw = support.fit_weight.detach().cpu()
    dc = support.dynamic_confidence.detach().cpu()
    rows = [
        f"{float(x[i, 0]):.6f} {float(x[i, 1]):.6f} {float(x[i, 2]):.6f} "
        f"{int(colour[i, 0])} {int(colour[i, 1])} {int(colour[i, 2])} "
        f"{int(fp[i])} {int(tp[i])} {float(fw[i]):.6f} {float(dc[i]):.6f}"
        for i in range(n)
    ]
    out.write_text(header + "\n".join(rows) + "\n")
