"""CODEX_robust_inner_killing_shared_spatial_scalar -- repair the inner solve.

Measured failures this targets, from the overnight sweep:

    blade_103706    23.9%-of-bbox translation while ZERO moving Gaussians
                    displace more than 5% -- divergence
    storage_45135   0.1-0.3 deg, 0.03% of bbox -- identity collapse

Both are scenes with weak endpoint separation (2.9x), and both come from the
same structural weakness: the current solver fits an INDEPENDENT spatial Killing
field per frame from the whole hard-labelled support. With little render signal
each frame is free to invent its own geometry, so the sequence either cancels to
identity or runs away.

The repair, per moving part k:

    v_k(x, t) = s_k(t) * Phi_k(x)

one spatial field for the entire video, one scalar speed per frame. Plus:
support cleaning before fitting, clustering into exactly K parts, and exact
rigid-field consistency added ALONGSIDE (never replacing) the existing strain
term.

Outer labels, temporal memory, the render contrast and the boundary split are
all untouched (§55).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .neighborhood import NeighborGraph, build_neighbor_graph

EPS = 1.0e-8


# ---------------------------------------------------------------------------
# Sections 2-5: candidate set, outlier rejection, fitting weights
# ---------------------------------------------------------------------------


@dataclass
class SupportCleaning:
    """Section 3's three states, kept distinct."""

    candidate: torch.Tensor          # [N] bool, z_i > 0
    outlier: torch.Tensor            # [N] bool, grossly isolated AND unsupported
    clean: torch.Tensor              # [N] bool, candidate and not outlier
    confidence: torch.Tensor         # [N] c_i = 1 - P_i0
    local_dynamic: torch.Tensor      # [N] s_i^dyn
    spacing: torch.Tensor            # [N] r_i
    diagnostics: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def clean_moving_support(
    xyz: torch.Tensor,
    soft_probs: torch.Tensor,
    hard_labels: torch.Tensor,
    graph: NeighborGraph,
    *,
    mad_scale: float = 3.5,
    local_dynamic_min: float = 0.35,
    eps: float = EPS,
) -> SupportCleaning:
    """Sections 2 and 3.

    A point is rejected only when it is BOTH grossly isolated in space AND
    unsupported by dynamic neighbours (§3). Spacing alone never removes a point:
    a thin but genuinely moving structure is sparse by nature, and removing it
    would shrink exactly the support the fit needs.
    """
    z = torch.as_tensor(hard_labels).reshape(-1)
    p = torch.as_tensor(soft_probs)
    candidate = z > 0
    confidence = (1.0 - p[:, 0]).clamp(0.0, 1.0)          # §2: c_i = 1 - P_i0

    # r_i = median neighbour distance, from the existing canonical graph.
    n = int(xyz.shape[0])
    device = xyz.device
    src = torch.cat((graph.source, graph.target)).to(device)
    dst = torch.cat((graph.target, graph.source)).to(device)
    dist = (xyz.index_select(0, src) - xyz.index_select(0, dst)).norm(dim=-1)
    spacing = torch.zeros(n, device=device, dtype=xyz.dtype)
    counts = torch.zeros(n, device=device, dtype=xyz.dtype)
    spacing.index_add_(0, src, dist)
    counts.index_add_(0, src, torch.ones_like(dist))
    spacing = spacing / counts.clamp_min(1.0)             # mean is a fine proxy here

    # §3: local dynamic support from the confidence field.
    w = torch.cat((graph.weight, graph.weight)).to(device=device, dtype=xyz.dtype)
    num = torch.zeros(n, device=device, dtype=xyz.dtype)
    den = torch.zeros(n, device=device, dtype=xyz.dtype)
    num.index_add_(0, src, w * confidence.index_select(0, dst))
    den.index_add_(0, src, w)
    local_dynamic = num / (den + float(eps))

    outlier = torch.zeros(n, dtype=torch.bool, device=device)
    if bool(candidate.any()):
        r = spacing[candidate]
        r_med = r.median()
        mad = 1.4826 * (r - r_med).abs().median()
        threshold = r_med + float(mad_scale) * mad
        isolated = torch.zeros(n, dtype=torch.bool, device=device)
        isolated[candidate] = spacing[candidate] > threshold
        unsupported = local_dynamic < float(local_dynamic_min)
        outlier = candidate & isolated & unsupported     # BOTH conditions (§3)

    clean = candidate & ~outlier
    return SupportCleaning(
        candidate=candidate, outlier=outlier, clean=clean, confidence=confidence,
        local_dynamic=local_dynamic, spacing=spacing,
        diagnostics={
            "candidate_moving_count": int(candidate.sum()),
            "outlier_count": int(outlier.sum()),
            "clean_count": int(clean.sum()),
            "mean_confidence_candidate": (
                float(confidence[candidate].mean()) if bool(candidate.any()) else float("nan")),
            "mean_local_dynamic_candidate": (
                float(local_dynamic[candidate].mean()) if bool(candidate.any()) else float("nan")),
        })


def fitting_weights(
    confidence: torch.Tensor,
    *,
    floor: float = 0.50,
    gamma: float = 2.0,
    part_prob: torch.Tensor | None = None,
    eta: float = 1.0,
) -> torch.Tensor:
    """Section 5: w_i = [ (c_i - tau_0) / (1 - tau_0) ]_+^gamma.

    Below the floor the weight is exactly zero, so a transported-but-uncertain
    point contributes nothing to the fit (§44 Test H) while still being carried
    by its part's transform.
    """
    c = torch.as_tensor(confidence).reshape(-1)
    tau = min(max(float(floor), 0.0), 0.999999)
    w = ((c - tau) / (1.0 - tau)).clamp_min(0.0).pow(float(gamma))
    if part_prob is not None:
        w = w * torch.as_tensor(part_prob, device=w.device, dtype=w.dtype).clamp_min(0.0).pow(float(eta))
    return w


# ---------------------------------------------------------------------------
# Sections 6-9: clustering the moving support into exactly K parts
# ---------------------------------------------------------------------------


@dataclass
class SupportClusters:
    """Section 4's fitting support, per part."""

    assignment: torch.Tensor         # [N] long, -1 = unassigned/not moving
    core: torch.Tensor               # [N] bool, the high-confidence clustering core
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def confidence_weighted_graph(
    xyz: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor,
    *,
    k: int = 16,
    gamma: float = 1.0,
    chunk_size: int = 4096,
) -> tuple[NeighborGraph, torch.Tensor]:
    """Section 7: w_ij = w_ij^x (c_i c_j)^gamma, on the masked subset.

    Returns the graph over the SUBSET plus the index map back to full indices.
    """
    idx = torch.nonzero(torch.as_tensor(mask).reshape(-1), as_tuple=False).reshape(-1)
    pts = xyz.index_select(0, idx)
    graph = build_neighbor_graph(pts, k=int(k), chunk_size=int(chunk_size))
    if graph.num_edges:
        c = torch.as_tensor(confidence, device=pts.device).reshape(-1).index_select(0, idx)
        ci = c.index_select(0, graph.source)
        cj = c.index_select(0, graph.target)
        weight = graph.weight * (ci * cj).clamp_min(0.0).pow(float(gamma))
        graph = NeighborGraph(graph.source, graph.target, weight, graph.num_nodes, graph.k)
    return graph, idx


@torch.no_grad()
def spectral_cluster(
    graph: NeighborGraph,
    num_parts: int,
    *,
    seed: int = 20260919,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 8: normalised-cut spectral clustering on the weighted graph.

    Plain Euclidean K-means is explicitly not the default (§8): it cuts through
    a connected elongated part whenever that part is longer than the gap to a
    different one. The normalised Laplacian's low eigenvectors instead follow
    graph connectivity, which is what "one rigid part" actually means here.
    """
    n = int(graph.num_nodes)
    k = max(1, int(num_parts))
    if k == 1 or n <= k:
        return torch.zeros(n, dtype=torch.long, device=graph.weight.device)

    device = graph.weight.device
    src = torch.cat((graph.source, graph.target)).to(device)
    dst = torch.cat((graph.target, graph.source)).to(device)
    w = torch.cat((graph.weight, graph.weight)).to(device)

    degree = torch.zeros(n, device=device, dtype=w.dtype).index_add_(0, src, w)
    d_inv_sqrt = degree.clamp_min(eps).rsqrt()
    # Normalised affinity S = D^-1/2 W D^-1/2 as a dense matrix; the moving
    # support is a small fraction of the cloud so this stays affordable.
    affinity = torch.zeros((n, n), device=device, dtype=w.dtype)
    affinity.index_put_((src, dst), w * d_inv_sqrt[src] * d_inv_sqrt[dst], accumulate=True)
    affinity = 0.5 * (affinity + affinity.transpose(0, 1))

    try:
        # Top-k eigenvectors of S == bottom-k of the normalised Laplacian.
        values, vectors = torch.linalg.eigh(affinity.double())
        embedding = vectors[:, -k:].to(w.dtype)
    except Exception:
        return torch.zeros(n, dtype=torch.long, device=device)

    embedding = embedding / embedding.norm(dim=1, keepdim=True).clamp_min(eps)
    from .soft_labels import soft_spatial_clustering

    pi, _ = soft_spatial_clustering(
        torch.nn.functional.pad(embedding, (0, max(0, 3 - embedding.shape[1])))[:, :3]
        if embedding.shape[1] < 3 else embedding[:, :3],
        torch.ones(n, device=device), k, iterations=30, seed=seed)
    return pi.argmax(dim=1)


@torch.no_grad()
def cluster_moving_support(
    xyz: torch.Tensor,
    cleaning: SupportCleaning,
    num_parts: int,
    *,
    core_confidence: float = 0.80,
    knn: int = 16,
    confidence_gamma: float = 1.0,
    attach_min_connectivity: float = 0.0,
    attach_min_confidence: float = 0.50,
    attach_preference_ratio: float = 1.5,
    seed: int = 20260919,
    eps: float = EPS,
) -> SupportClusters:
    """Sections 6-9: cluster the confident core, then attach conservatively."""
    device = xyz.device
    n = int(xyz.shape[0])
    assignment = torch.full((n,), -1, dtype=torch.long, device=device)

    core = cleaning.clean & (cleaning.confidence > float(core_confidence))    # §6
    summary: dict[str, Any] = {
        "core_count": int(core.sum()),
        "clean_count": int(cleaning.clean.sum()),
        "core_confidence_threshold": float(core_confidence),
    }
    if not bool(core.any()):
        # Nothing confident enough to cluster: fall back to the clean set so the
        # solve still has support, and say so rather than silently fitting noise.
        core = cleaning.clean
        summary["core_fallback_to_clean"] = True
    if not bool(core.any()):
        return SupportClusters(assignment=assignment, core=core, summary=summary)

    graph, idx = confidence_weighted_graph(
        xyz, cleaning.confidence, core, k=knn, gamma=confidence_gamma)
    labels = spectral_cluster(graph, num_parts, seed=seed)
    assignment[idx] = labels

    # §9: attach remaining clean candidates only on clear graph preference.
    remaining = cleaning.clean & (assignment < 0) & (cleaning.confidence >= float(attach_min_confidence))
    attached = 0
    if bool(remaining.any()) and bool((assignment >= 0).any()):
        full_graph = build_neighbor_graph(xyz, k=int(knn))
        src = torch.cat((full_graph.source, full_graph.target)).to(device)
        dst = torch.cat((full_graph.target, full_graph.source)).to(device)
        w = torch.cat((full_graph.weight, full_graph.weight)).to(device)
        k_parts = max(1, int(num_parts))
        votes = torch.zeros((n, k_parts), device=device, dtype=w.dtype)
        assigned_dst = assignment.index_select(0, dst)
        valid = assigned_dst >= 0
        if bool(valid.any()):
            votes.index_put_((src[valid], assigned_dst[valid]), w[valid], accumulate=True)
        best2 = votes.topk(min(2, k_parts), dim=1)
        top = best2.values[:, 0]
        second = best2.values[:, 1] if k_parts > 1 else torch.zeros_like(top)
        clear = (top > float(attach_min_connectivity)) & (
            top >= float(attach_preference_ratio) * second.clamp_min(eps))
        take = remaining & clear
        assignment[take] = best2.indices[take, 0]
        attached = int(take.sum())

    summary.update({
        "attached_count": attached,
        "assigned_count": int((assignment >= 0).sum()),
        "unassigned_clean_count": int((cleaning.clean & (assignment < 0)).sum()),
    })

    diagnostics = []
    for part in range(max(1, int(num_parts))):
        m = assignment == part
        entry: dict[str, Any] = {"part": part, "size": int(m.sum())}
        if bool(m.any()):
            pts = xyz[m]
            centred = pts - pts.mean(dim=0, keepdim=True)
            cov = (centred.transpose(0, 1) @ centred) / max(int(m.sum()), 1)
            entry.update({
                "mean_confidence": float(cleaning.confidence[m].mean()),
                "bbox_extent": (pts.max(dim=0).values - pts.min(dim=0).values).tolist(),
                "covariance_eigenvalues": torch.linalg.eigvalsh(cov.double()).tolist(),
            })
        diagnostics.append(entry)
    return SupportClusters(assignment=assignment, core=core,
                           diagnostics=diagnostics, summary=summary)


# ---------------------------------------------------------------------------
# Sections 11-14: exact rigid-field consistency (added, never replacing strain)
# ---------------------------------------------------------------------------


def rigid_field(points: torch.Tensor, omega: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Section 13: Phi^rigid(x) = u + omega x x."""
    return u.reshape(1, 3) + torch.cross(
        omega.reshape(1, 3).expand(points.shape[0], 3), points, dim=-1)


def rigid_basis(points: torch.Tensor) -> torch.Tensor:
    """A_i = [ -[x_i]_x  I_3 ] in R^{3x6}, so that A_i xi = u + omega x x_i.

    ONE-FIELD correction §6. Note the sign: -[x]_x v = v x x = -(x x v), and
    since omega x x = -(x x omega), the first block must be -[x]_x for
    A [omega; u] to equal u + omega x x.
    """
    x = torch.as_tensor(points)
    n = int(x.shape[0])
    zero = torch.zeros(n, device=x.device, dtype=x.dtype)
    hat = torch.stack((
        torch.stack((zero, -x[:, 2], x[:, 1]), dim=1),
        torch.stack((x[:, 2], zero, -x[:, 0]), dim=1),
        torch.stack((-x[:, 1], x[:, 0], zero), dim=1)), dim=1)      # [n, 3, 3] = [x]_x
    eye = torch.eye(3, device=x.device, dtype=x.dtype).expand(n, 3, 3)
    return torch.cat((-hat, eye), dim=2)                             # [n, 3, 6]


def project_field_to_rigid_twist(
    xyz: torch.Tensor,
    velocity: torch.Tensor,
    weights: torch.Tensor,
    *,
    reg: float = 1.0e-8,
    eps: float = EPS,
) -> dict[str, Any]:
    """ONE-FIELD correction §6, §7, §35, §36: xi* = argmin_xi sum_i w_i ||phi_i - A_i xi||^2.

    Closed form xi* = (H + reg I)^-1 g with H = sum w A^T A and g = sum w A^T phi.

    This is DERIVED, never a Parameter (§8, §9). It is differentiable with
    respect to ``velocity``, so gradients flow

        loss -> xi* -> phi -> control/theta

    which is the whole point: the rigid motion that transports the Gaussians is
    a function of the one optimized field, not a second competing variable.
    """
    x = torch.as_tensor(xyz)
    phi = torch.as_tensor(velocity)
    w = torch.as_tensor(weights, device=x.device, dtype=x.dtype).reshape(-1)
    n = int(x.shape[0])
    if n == 0:
        zero6 = torch.zeros(6, device=x.device, dtype=x.dtype)
        return {"xi": zero6, "omega": zero6[:3], "u": zero6[3:],
                "rigid_velocity": phi.new_zeros((0, 3)), "H": torch.zeros((6, 6), device=x.device),
                "residual": phi.sum() * 0.0, "rank": 0, "condition_number": float("inf"),
                "eigenvalues": [], "fallback": False, "valid": False}

    A = rigid_basis(x)                                               # [n, 3, 6]
    wA = w[:, None, None] * A
    H = torch.einsum("nij,nik->jk", A, wA)                           # [6, 6]
    g = torch.einsum("nij,ni->j", wA, phi)                           # [6]
    eye6 = torch.eye(6, device=x.device, dtype=x.dtype)
    damped = H + float(reg) * eye6

    fallback = False
    try:
        xi = torch.linalg.solve(damped, g.unsqueeze(-1)).squeeze(-1)
    except Exception:
        # §7: guarded fallback only, and it is logged rather than silent.
        fallback = True
        xi = torch.linalg.lstsq(damped, g.unsqueeze(-1)).solution.squeeze(-1)

    omega, u = xi[:3], xi[3:]
    rigid_velocity = torch.einsum("nij,j->ni", A, xi)
    residual = ((w * (phi - rigid_velocity).square().sum(dim=-1)).sum()
                / (w.sum() + float(eps)))

    with torch.no_grad():
        values = torch.linalg.eigvalsh(H.double())
        peak = values.max()
        rank = int((values > peak * 1.0e-10).sum()) if float(peak) > 0 else 0
        condition = float(peak / values.min().clamp_min(1.0e-300)) if float(peak) > 0 else float("inf")
    return {"xi": xi, "omega": omega, "u": u, "rigid_velocity": rigid_velocity,
            "H": H, "residual": residual, "rank": rank,
            "condition_number": condition, "eigenvalues": values.tolist(),
            "fallback": fallback, "valid": rank >= 6}


def single_field_rigid_projection_losses(
    points: torch.Tensor,
    phi_local: torch.Tensor,
    local_omega: torch.Tensor,
    local_b: torch.Tensor,
    projection: dict[str, Any],
    weights: torch.Tensor,
    *,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """ONE-FIELD correction §11, §12, §34, §36.

    ``L_rigid_proj`` compares the optimized field against ITS OWN best rigid
    projection -- not against a second learned field. ``L_omega`` / ``L_u`` keep
    the local cell statistics referenced to that same derived twist.
    """
    w = torch.as_tensor(weights, device=points.device, dtype=points.dtype).reshape(-1)
    denom = w.sum() + float(eps)
    omega_star, u_star = projection["omega"], projection["u"]

    rigid = projection["rigid_velocity"]
    l_proj = (w * (phi_local - rigid).square().sum(dim=-1)).sum() / denom
    l_omega = (w * (local_omega - omega_star.reshape(1, 3)).square().sum(dim=-1)).sum() / denom
    predicted_b = u_star.reshape(1, 3) + torch.cross(
        omega_star.reshape(1, 3).expand(points.shape[0], 3), points, dim=-1)
    l_u = (w * (local_b - predicted_b).square().sum(dim=-1)).sum() / denom
    return {"rigid_proj": l_proj, "omega": l_omega, "u": l_u}


def exact_rigid_consistency_losses(
    cell_omega: torch.Tensor,
    cell_u: torch.Tensor,
    cell_center: torch.Tensor,
    omega: torch.Tensor,
    u: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """Section 13's L_omega and L_u.

        L_omega = sum_c w_c ||omega_c - omega||^2
        L_u     = sum_c w_c ||u_c - (u + omega x b_c)||^2

    These pin every local cell to ONE global twist. The existing strain term only
    says each cell is locally rigid; it cannot stop neighbouring cells from being
    rigid about different axes, which is how a per-frame field drifts.
    """
    w = (torch.ones(cell_omega.shape[0], device=cell_omega.device, dtype=cell_omega.dtype)
         if weights is None else torch.as_tensor(weights).reshape(-1).to(cell_omega))
    denom = w.sum() + float(eps)
    l_omega = (w * (cell_omega - omega.reshape(1, 3)).square().sum(dim=-1)).sum() / denom
    predicted_u = u.reshape(1, 3) + torch.cross(
        omega.reshape(1, 3).expand(cell_center.shape[0], 3), cell_center, dim=-1)
    l_u = (w * (cell_u - predicted_u).square().sum(dim=-1)).sum() / denom
    return {"omega": l_omega, "u": l_u}


def rigid_field_probe_loss(
    points: torch.Tensor,
    field_values: torch.Tensor,
    omega: torch.Tensor,
    u: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 13's optional direct probe loss (weight 0 by default, §14)."""
    w = (torch.ones(points.shape[0], device=points.device, dtype=points.dtype)
         if weights is None else torch.as_tensor(weights).reshape(-1).to(points))
    residual = field_values - rigid_field(points, omega, u)
    return (w * residual.square().sum(dim=-1)).sum() / (w.sum() + float(eps))


# ---------------------------------------------------------------------------
# Sections 16-19: shared spatial field and the temporal scalar
# ---------------------------------------------------------------------------


def field_rms(
    points: torch.Tensor,
    omega: torch.Tensor,
    u: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 17's weighted RMS magnitude of Phi over the fitting support."""
    values = rigid_field(points, omega, u)
    w = torch.as_tensor(weights).reshape(-1).to(values)
    return ((w * values.square().sum(dim=-1)).sum() / (w.sum() + float(eps))).clamp_min(eps).sqrt()


@torch.no_grad()
def normalize_gauge(
    omega: torch.Tensor,
    u: torch.Tensor,
    speed: torch.Tensor,
    points: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Sections 17 and 18: fix the scale and sign gauge without changing physics.

        Phi -> Phi / rms,   s -> s * rms          (velocity s*Phi unchanged)
        then orient so the first reliably nonzero s(t) is positive.
    """
    rms = float(field_rms(points, omega, u, weights, eps=eps))
    if rms <= eps:
        return omega, u, speed, 1.0
    omega_n, u_n, speed_n = omega / rms, u / rms, speed * rms

    # §18: sign is pure convention; a genuine later reversal stays allowed.
    magnitude = speed_n.abs()
    if bool((magnitude > eps).any()):
        first = int(torch.nonzero(magnitude > magnitude.max() * 0.1, as_tuple=False)[0])
        if float(speed_n[first]) < 0:
            omega_n, u_n, speed_n = -omega_n, -u_n, -speed_n
    return omega_n, u_n, speed_n, rms


def scalar_from_observation(
    observed_velocity: torch.Tensor,
    field_values: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 19: s_hat_i(t) = <v_hat, Phi> / ||Phi||^2, the per-point implied speed."""
    num = (observed_velocity * field_values).sum(dim=-1)
    den = field_values.square().sum(dim=-1) + float(eps)
    return num / den


def huber(residual: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    a = residual.abs()
    d = float(delta)
    return torch.where(a <= d, 0.5 * residual.square(), d * (a - 0.5 * d))


def scalar_consistency_loss(
    implied: torch.Tensor,
    speed: torch.Tensor,
    weights: torch.Tensor,
    *,
    delta: float = 1.0,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 19, robust: sum w rho(s_hat_i(t) - s(t)) / sum w."""
    w = torch.as_tensor(weights).to(implied)
    return (w * huber(implied - speed, delta)).sum() / (w.sum() + float(eps))


@torch.no_grad()
def scalar_inconsistency(implied: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
    """Section 20: E_i = median_t |s_hat_i(t) - s(t)|. Logged, never written back
    into the outer labels."""
    return (implied - speed.reshape(1, -1)).abs().median(dim=1).values


def robust_scalar_weight(error: torch.Tensor, *, tau: float = 1.0) -> torch.Tensor:
    """Section 21: 1 / (1 + (e/tau)^2). Never a hard delete."""
    return 1.0 / (1.0 + (torch.as_tensor(error) / max(float(tau), EPS)).square())


# ---------------------------------------------------------------------------
# Sections 24-27: temporal regularization of s(t)
# ---------------------------------------------------------------------------


def scalar_continuity_loss(speed: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Section 24: sum a(t)^2 with a(t) = (s(t+1) - s(t)) / dt."""
    if int(speed.numel()) < 2:
        return speed.sum() * 0.0
    d = torch.as_tensor(dt, device=speed.device, dtype=speed.dtype).reshape(-1)
    step = d[: speed.numel() - 1].clamp_min(EPS)
    return ((speed[1:] - speed[:-1]) / step).square().mean()


def scalar_jerk_loss(speed: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Section 25, weak by default."""
    if int(speed.numel()) < 3:
        return speed.sum() * 0.0
    d = torch.as_tensor(dt, device=speed.device, dtype=speed.dtype).reshape(-1)
    step = d[: speed.numel() - 1].clamp_min(EPS)
    accel = (speed[1:] - speed[:-1]) / step
    return (accel[1:] - accel[:-1]).square().mean()


def scalar_sign_flip_loss(speed: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """Section 26: ReLU(-s(t) s(t+1) / s_rms^2).

    Penalises an ABRUPT high-speed reversal while leaving a smooth zero crossing
    essentially free -- the product is tiny near zero, so
    +0.1, +0.01, 0, -0.01, -0.1 costs almost nothing, but +1 -> -1 costs a lot.
    """
    if int(speed.numel()) < 2:
        return speed.sum() * 0.0
    rms = speed.square().mean().clamp_min(eps)
    return torch.relu(-(speed[:-1] * speed[1:]) / rms).mean()


@torch.no_grad()
def sign_flip_diagnostics(speed: torch.Tensor, *, high_fraction: float = 0.5) -> dict[str, Any]:
    """Section 27."""
    s = torch.as_tensor(speed).reshape(-1)
    if int(s.numel()) < 2:
        return {"num_sign_flips": 0, "num_high_speed_sign_flips": 0,
                "max_flip_magnitude": 0.0, "strongest_flip_frame": -1}
    product = s[:-1] * s[1:]
    flips = product < 0
    magnitude = torch.where(flips, -product, torch.zeros_like(product))
    threshold = float(high_fraction) * float(s.abs().max().clamp_min(EPS)) ** 2
    return {
        "num_sign_flips": int(flips.sum()),
        "num_high_speed_sign_flips": int((magnitude > threshold).sum()),
        "max_flip_magnitude": float(magnitude.max()) if int(magnitude.numel()) else 0.0,
        "strongest_flip_frame": int(magnitude.argmax()) if bool(flips.any()) else -1,
    }


# ---------------------------------------------------------------------------
# Sections 28-29: integration and exact SE(3) transport
# ---------------------------------------------------------------------------


def integrate_scalar(speed: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Section 28: trapezoidal q(t), q(t_0) = 0, real dt -- no unit-step assumption."""
    s = torch.as_tensor(speed).reshape(-1)
    d = torch.as_tensor(dt, device=s.device, dtype=s.dtype).reshape(-1)
    steps = 0.5 * (s[:-1] + s[1:]) * d[: s.numel() - 1]
    return torch.cat((torch.zeros(1, device=s.device, dtype=s.dtype), steps.cumsum(0)))


def se3_exp_twist(omega: torch.Tensor, u: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Section 29: T(t) = Exp(q(t) * xi), returned as [T, 4, 4]."""
    device, dtype = omega.device, omega.dtype
    qq = torch.as_tensor(q, device=device, dtype=dtype).reshape(-1)
    out = torch.eye(4, device=device, dtype=dtype).repeat(int(qq.numel()), 1, 1)
    w = omega.reshape(3) * qq[:, None]                       # [T, 3]
    v = u.reshape(3) * qq[:, None]

    def _hat(a):
        z = torch.zeros(a.shape[0], device=device, dtype=dtype)
        return torch.stack((
            torch.stack((z, -a[:, 2], a[:, 1]), dim=1),
            torch.stack((a[:, 2], z, -a[:, 0]), dim=1),
            torch.stack((-a[:, 1], a[:, 0], z), dim=1)), dim=1)

    W = _hat(w)
    eye = torch.eye(3, device=device, dtype=dtype).expand(int(qq.numel()), 3, 3)

    # Work in theta^2, never in ||w||. Two reasons, both of which bit us:
    #
    # 1. GRADIENTS. d||w||/dw = w/||w|| is 0/0 at w = 0, and q(t_0) = 0 by
    #    construction, so frame 0 ALWAYS hits it. Masking the small-angle rows
    #    afterwards does not help: the masked-out rows still contribute
    #    0 * NaN = NaN in backward. With a near-prismatic joint (|omega| ~ 1e-4)
    #    almost every frame lands there, which is exactly how 41 of 60 s(t)
    #    gradients came back non-finite.
    # 2. PRECISION. (1 - cos th)/th^2 loses catastrophically in float32 for
    #    small th: at th = 1e-2 the numerator is 5e-5 with ~1e-7 absolute error.
    #
    # theta_sq is a polynomial in w, so it differentiates cleanly at zero, and
    # below th = 0.1 the Taylor series is both exact to ~1e-11 and free of
    # cancellation. Both branches are evaluated on a SAFE theta so neither can
    # produce a NaN for torch.where to propagate.
    theta_sq = (w * w).sum(dim=1).reshape(-1, 1, 1)                  # [T, 1, 1]
    small = theta_sq < 1.0e-2
    theta = torch.sqrt(torch.where(small, torch.ones_like(theta_sq), theta_sq))
    t2 = torch.where(small, theta_sq, torch.zeros_like(theta_sq))    # Taylor argument

    #  A = sin(th)/th          B = (1-cos th)/th^2      C = (th - sin th)/th^3
    # Four terms, not three: Rodrigues' identity only makes R orthogonal when A
    # and B are exact, so truncation error shows up directly as R R^T - I. At the
    # switch (t2 = 1e-2) three terms leave ~1e-10 of non-orthogonality; four
    # leave ~1e-13, below float64 round-off in the products below.
    t4 = t2 * t2
    t6 = t4 * t2
    a_coef = torch.where(small,
                         1.0 - t2 / 6.0 + t4 / 120.0 - t6 / 5040.0,
                         torch.sin(theta) / theta)
    b_coef = torch.where(small,
                         0.5 - t2 / 24.0 + t4 / 720.0 - t6 / 40320.0,
                         (1.0 - torch.cos(theta)) / theta.square())
    c_coef = torch.where(small,
                         1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0 - t6 / 362880.0,
                         (theta - torch.sin(theta)) / theta.pow(3))

    WW = W @ W
    R = eye + a_coef * W + b_coef * WW
    V = eye + b_coef * W + c_coef * WW
    out[:, :3, :3] = R
    out[:, :3, 3] = (V @ v[:, :, None]).squeeze(-1)
    return out


def transport_points(transforms: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """x_i(t) = T(t) x_i^0, returned as [T, N, 3] (§29)."""
    R = transforms[:, :3, :3]
    t = transforms[:, :3, 3]
    return torch.einsum("tij,nj->tni", R, points) + t[:, None, :]


# ---------------------------------------------------------------------------
# Sections 30-33: optional motion-type priors, OFF by default
# ---------------------------------------------------------------------------


def screw_pitch(omega: torch.Tensor, u: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """Section 32: h = <omega, u> / ||omega||^2. Pure revolute motion has h = 0."""
    return (omega * u).sum() / (omega.square().sum() + float(eps))


def motion_prior_losses(
    omega: torch.Tensor,
    u: torch.Tensor,
    mode: str = "none",
    *,
    omega_min: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Sections 30-33. ``none`` sets BOTH weights to exactly zero (§50 Test U)."""
    zero = omega.sum() * 0.0
    m = str(mode).strip().lower()
    if m == "none":
        return {"revolute": zero, "prismatic": zero}
    if m == "prismatic":
        return {"revolute": zero, "prismatic": omega.square().sum()}
    if m == "revolute":
        loss = screw_pitch(omega, u).square()
        if float(omega_min) > 0.0:
            loss = loss + torch.nn.functional.softplus(
                torch.as_tensor(float(omega_min), device=omega.device) - omega.norm()).square()
        return {"revolute": loss, "prismatic": zero}
    raise ValueError(f"unknown inner_motion_prior: {mode}")


# ---------------------------------------------------------------------------
# Section 38: observability
# ---------------------------------------------------------------------------


@torch.no_grad()
def observability(points: torch.Tensor, weights: torch.Tensor) -> dict[str, Any]:
    """Section 38: H = sum_i w_i A_i^T A_i with A_i = [-[x]_x  I].

    A poorly conditioned H is exactly the situation where a rigid fit can run
    away (blade) or vanish (storage), so this is the number to look at first
    when a solve misbehaves.
    """
    x = torch.as_tensor(points).double()
    w = torch.as_tensor(weights).reshape(-1).double()
    n = int(x.shape[0])
    if n == 0:
        return {"rank": 0, "condition_number": float("inf"), "eigenvalues": []}
    zeros = torch.zeros(n, dtype=x.dtype, device=x.device)
    hat = torch.stack((
        torch.stack((zeros, -x[:, 2], x[:, 1]), dim=1),
        torch.stack((x[:, 2], zeros, -x[:, 0]), dim=1),
        torch.stack((-x[:, 1], x[:, 0], zeros), dim=1)), dim=1)
    eye = torch.eye(3, dtype=x.dtype, device=x.device).expand(n, 3, 3)
    A = torch.cat((-hat, eye), dim=2)                      # [n, 3, 6]
    H = torch.einsum("n,nij,nik->jk", w, A, A)
    values = torch.linalg.eigvalsh(H)
    positive = values[values > values.max() * 1.0e-10] if float(values.max()) > 0 else values
    return {
        "eigenvalues": values.tolist(),
        "condition_number": float(values.max() / values.min().clamp_min(1.0e-300)),
        "rank": int((values > values.max() * 1.0e-10).sum()) if float(values.max()) > 0 else 0,
        "min_eigenvalue": float(values.min()),
        "max_eigenvalue": float(values.max()),
        "effective_points": float(w.sum()),
    }


# ---------------------------------------------------------------------------
# Sections 34-37: the alternating solve
# ---------------------------------------------------------------------------


@dataclass
class SharedFieldSolution:
    omega: torch.Tensor              # [3] global twist rotation
    u: torch.Tensor                  # [3] global twist translation
    speed: torch.Tensor              # [T] s_k(t)
    q: torch.Tensor                  # [T] integrated q_k(t)
    transforms: torch.Tensor         # [T, 4, 4]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)


def solve_shared_field_and_scalar(
    points: torch.Tensor,
    observed_velocity: torch.Tensor,
    weights: torch.Tensor,
    dt: torch.Tensor,
    *,
    spatial_steps: int = 60,
    scalar_steps: int = 60,
    rounds: int = 3,
    lr: float = 1.0e-2,
    lambda_obs: float = 1.0,
    lambda_scalar: float = 1.0,
    lambda_continuity: float = 0.01,
    lambda_flip: float = 0.01,
    lambda_jerk: float = 0.0,
    lambda_revolute: float = 0.0,
    lambda_prismatic: float = 0.0,
    motion_prior: str = "none",
    huber_delta: float = 1.0,
    scalar_robust_tau: float = 1.0,
    seed: int = 20260919,
) -> SharedFieldSolution:
    """Sections 35-37: alternate spatial and scalar steps, ONE field for all frames.

    ``observed_velocity`` is [T, M, 3] over the fitting support; ``weights`` is
    [M]. The spatial step sees every frame at once (§22), which is what stops a
    per-frame field from inventing frame-specific geometry.
    """
    device = points.device
    dtype = points.dtype
    frames = int(observed_velocity.shape[0])
    w = torch.as_tensor(weights, device=device, dtype=dtype).reshape(-1)

    # Initialise the twist from a weighted least-squares fit to the mean motion,
    # so the alternation starts somewhere physical rather than at zero.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    mean_v = (observed_velocity * w[None, :, None]).sum(dim=(0, 1)) / (frames * w.sum() + EPS)
    omega = torch.nn.Parameter((torch.randn(3, generator=generator).to(device=device, dtype=dtype) * 1.0e-3))
    u = torch.nn.Parameter(mean_v.clone().to(device=device, dtype=dtype))
    speed = torch.nn.Parameter(torch.ones(frames, device=device, dtype=dtype))

    history: list[dict[str, float]] = []
    robust = torch.ones((int(points.shape[0]), frames), device=device, dtype=dtype)

    for _ in range(max(1, int(rounds))):
        # --- 1. spatial step: Phi with s(t) frozen (§36) --------------------
        optimizer = torch.optim.Adam([omega, u], lr=float(lr))
        for _ in range(max(1, int(spatial_steps))):
            values = rigid_field(points, omega, u)                     # [M, 3]
            predicted = speed.detach()[:, None, None] * values[None]   # [T, M, 3]
            weight = w[None, :] * robust.transpose(0, 1)
            obs = (weight * huber((predicted - observed_velocity).norm(dim=-1),
                                  huber_delta)).sum() / (weight.sum() + EPS)
            implied = scalar_from_observation(observed_velocity, values[None].expand_as(observed_velocity))
            # implied is [T, M]; s(t) broadcasts as [T, 1]; weight is [T, M].
            scal = scalar_consistency_loss(implied, speed.detach()[:, None],
                                           weight, delta=huber_delta)
            priors = motion_prior_losses(omega, u, motion_prior)
            total = (float(lambda_obs) * obs + float(lambda_scalar) * scal
                     + float(lambda_revolute) * priors["revolute"]
                     + float(lambda_prismatic) * priors["prismatic"])
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

        # --- 2. gauge normalisation (§17, §18) ------------------------------
        with torch.no_grad():
            o, uu, s, rms = normalize_gauge(omega.detach(), u.detach(), speed.detach(), points, w)
            omega.copy_(o); u.copy_(uu); speed.copy_(s)

        # --- 3. scalar step: s(t) with Phi frozen (§37) ---------------------
        optimizer = torch.optim.Adam([speed], lr=float(lr))
        values_frozen = rigid_field(points, omega.detach(), u.detach())
        for _ in range(max(1, int(scalar_steps))):
            predicted = speed[:, None, None] * values_frozen[None]
            weight = w[None, :] * robust.transpose(0, 1)
            obs = (weight * huber((predicted - observed_velocity).norm(dim=-1),
                                  huber_delta)).sum() / (weight.sum() + EPS)
            implied = scalar_from_observation(
                observed_velocity, values_frozen[None].expand_as(observed_velocity))
            scal = scalar_consistency_loss(implied, speed[:, None], weight, delta=huber_delta)
            total = (float(lambda_obs) * obs + float(lambda_scalar) * scal
                     + float(lambda_continuity) * scalar_continuity_loss(speed, dt)
                     + float(lambda_flip) * scalar_sign_flip_loss(speed)
                     + float(lambda_jerk) * scalar_jerk_loss(speed, dt))
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            history.append({"obs": float(obs.detach()), "scalar": float(scal.detach()),
                            "total": float(total.detach())})

        # --- 4. robust re-weighting from scalar inconsistency (§20, §21) ----
        with torch.no_grad():
            values_frozen = rigid_field(points, omega, u)
            implied = scalar_from_observation(
                observed_velocity, values_frozen[None].expand_as(observed_velocity))
            error = (implied - speed[None, :].transpose(0, 1)).abs().transpose(0, 1)
            robust = robust_scalar_weight(error, tau=scalar_robust_tau)

    with torch.no_grad():
        o, uu, s, _ = normalize_gauge(omega.detach(), u.detach(), speed.detach(), points, w)
        q = integrate_scalar(s, dt)
        transforms = se3_exp_twist(o, uu, q)
        values = rigid_field(points, o, uu)
        implied = scalar_from_observation(observed_velocity, values[None].expand_as(observed_velocity))
        inconsistency = scalar_inconsistency(implied.transpose(0, 1), s)
        displacement = (transport_points(transforms, points)[-1] - points).norm(dim=-1)
        diagnostics = {
            **sign_flip_diagnostics(s),
            **observability(points, w),
            "omega_norm": float(o.norm()),
            "u_norm": float(uu.norm()),
            "speed_rms": float(s.square().mean().sqrt()),
            "q_final": float(q[-1]),
            "median_transported_displacement": float(displacement.median()),
            "median_observed_speed": float(observed_velocity.norm(dim=-1).median()),
            "median_scalar_inconsistency": float(inconsistency.median()),
            "num_frames": frames,
            "fitting_points": int(points.shape[0]),
        }
    return SharedFieldSolution(omega=o, u=uu, speed=s, q=q, transforms=transforms,
                               diagnostics=diagnostics, history=history)
