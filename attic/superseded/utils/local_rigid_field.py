"""Local finite-rigidity estimation on fixed spatial neighbourhoods.

This module answers one question per Gaussian: *does the displacement field
around this point look like a coherent rigid motion, and if so how large is
it?*  It never fits a single global transform, never requires an exact
source/target Gaussian correspondence, and never needs to know how many
rigid parts the scene contains -- a neighbourhood inside part 1 and a
neighbourhood inside part 2 are fitted independently and never averaged.

The local transforms produced here are motion *diagnostics*.  They are not
part transforms, joint transforms, or correspondence transforms, and nothing
downstream may treat them as such.

Centering convention: the spec writes the local differences relative to the
centre point x_i, but the translation as a Kabsch offset between weighted
centroids.  Those two are only consistent when the centroid is the centre,
and centring on x_i would feed the centre's own displacement (via z_i) back
into a fit that is explicitly supposed to exclude it (leave-one-centre-out).
Both are therefore centred on the weighted neighbour centroid, which keeps
the Jacobian translation-invariant and keeps the centre genuinely held out.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LocalDeformationResult:
    """Per-point local motion model and the evidence behind it."""

    J: torch.Tensor                  # [N, 3, 3] local deformation gradient
    R: torch.Tensor                  # [N, 3, 3] closest proper rotation to J
    t: torch.Tensor                  # [N, 3] local translation, Kabsch offset
    singular_values: torch.Tensor    # [N, 3] singular values of J

    strain_error: torch.Tensor       # [N] sum (sigma_a - 1)^2
    rigid_fit_error: torch.Tensor    # [N] E^R, weighted residual of R,t
    static_fit_error: torch.Tensor   # [N] E^S, weighted residual of F(x)=x
    rigid_gain: torch.Tensor         # [N] E^S - E^R
    rigid_failure_fraction: torch.Tensor  # [N] E^R / (E^R + E^S), 0.5 = no information

    predicted_disp: torch.Tensor     # [N, 3] neighbour-predicted centre motion
    motion_magnitude: torch.Tensor   # [N] ||predicted_disp|| / effective scale

    inlier_fraction: torch.Tensor    # [N] fraction within the adaptive robust threshold
    robust_delta: torch.Tensor       # [N] the adaptive threshold actually used
    effective_scale: torch.Tensor    # [N] length unit used for every normalisation
    cov_eigenvalues: torch.Tensor    # [N, 3] ascending eigenvalues of the source covariance
    cov_condition: torch.Tensor      # [N] lambda1 / lambda3, surface-degeneracy diagnostic
    valid: torch.Tensor              # [N] bool


def build_knn_graph(xyz: torch.Tensor, k: int, *, chunk_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-excluding k-NN graph over one cloud, from geometry only.

    Geometry only and built once: the graph must not depend on the dynamic
    labels it is used to refine, or the refinement feeds on itself.
    """
    xyz = torch.as_tensor(xyz).detach()
    n = int(xyz.shape[0])
    if n <= 1:
        empty_i = torch.zeros((n, 0), dtype=torch.long, device=xyz.device)
        return empty_i, torch.zeros((n, 0), dtype=xyz.dtype, device=xyz.device)
    kk = min(max(1, int(k)) + 1, n)
    idx_chunks: list[torch.Tensor] = []
    dist_chunks: list[torch.Tensor] = []
    for start in range(0, n, max(1, int(chunk_size))):
        d = torch.cdist(xyz[start:start + int(chunk_size)], xyz)
        values, indices = torch.topk(d, k=kk, largest=False, sorted=True)
        idx_chunks.append(indices[:, 1:])
        dist_chunks.append(values[:, 1:])
    return torch.cat(idx_chunks, 0), torch.cat(dist_chunks, 0)


def query_knn(query_xyz: torch.Tensor, target_xyz: torch.Tensor, k: int, *, chunk_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """k-NN of arbitrary continuous query locations into a target cloud.

    The queries are transported positions, not cloud members, so nothing is
    excluded here -- there is no "self" to drop.
    """
    query = torch.as_tensor(query_xyz).detach()
    target = torch.as_tensor(target_xyz).detach().to(device=query.device, dtype=query.dtype)
    m = int(target.shape[0])
    if m == 0:
        empty_i = torch.zeros((int(query.shape[0]), 0), dtype=torch.long, device=query.device)
        return empty_i, torch.zeros((int(query.shape[0]), 0), dtype=query.dtype, device=query.device)
    kk = min(max(1, int(k)), m)
    idx_chunks: list[torch.Tensor] = []
    dist_chunks: list[torch.Tensor] = []
    for start in range(0, int(query.shape[0]), max(1, int(chunk_size))):
        d = torch.cdist(query[start:start + int(chunk_size)], target)
        values, indices = torch.topk(d, k=kk, largest=False, sorted=True)
        idx_chunks.append(indices)
        dist_chunks.append(values)
    return torch.cat(idx_chunks, 0), torch.cat(dist_chunks, 0)


def local_scale_from_knn(neighbor_distance: torch.Tensor, *, minimum: float = 1.0e-6) -> torch.Tensor:
    """Per-point length unit: the median distance to its own neighbours."""
    if int(neighbor_distance.shape[1]) == 0:
        return torch.full((int(neighbor_distance.shape[0]),), float(minimum), device=neighbor_distance.device, dtype=neighbor_distance.dtype)
    scale = neighbor_distance.median(dim=1).values
    finite = scale[torch.isfinite(scale) & (scale > 0)]
    fallback = finite.median() if finite.numel() else torch.tensor(float(minimum), device=scale.device, dtype=scale.dtype)
    return torch.where(torch.isfinite(scale) & (scale > 0), scale, fallback).clamp_min(float(minimum))


def compute_effective_local_scale(
    raw_scale: torch.Tensor,
    knn_indices: torch.Tensor,
    *,
    mode: str = "neighborhood_median",
    clip_low_quantile: float = 0.10,
    clip_high_quantile: float = 0.90,
) -> torch.Tensor:
    """A density-robust length unit for every normalised quantity.

    The raw per-point nearest-neighbour spacing tracks sampling density, not
    geometry: a dense patch gets a tiny unit, so a millimetre of motion there
    normalises to a larger number than a centimetre in a sparse patch.  That
    manufactures false dynamics in dense regions and false statics in sparse
    ones.  Pooling over the neighbourhood and clipping to scene-relative
    quantiles keeps the unit geometric without hard-coding any metric
    threshold.
    """
    raw = torch.as_tensor(raw_scale).detach().reshape(-1)
    mode = str(mode)
    if mode == "raw" or int(knn_indices.shape[1]) == 0:
        effective = raw
    else:
        gathered = raw[torch.as_tensor(knn_indices).detach().to(device=raw.device).long()]
        both = torch.cat((raw[:, None], gathered), dim=1)
        if mode == "neighborhood_mean":
            effective = both.mean(dim=1)
        elif mode == "neighborhood_median":
            effective = both.median(dim=1).values
        else:
            raise ValueError("local scale mode must be raw, neighborhood_median or neighborhood_mean")
    finite = effective[torch.isfinite(effective) & (effective > 0)]
    if finite.numel() > 1:
        low = torch.quantile(finite, float(clip_low_quantile))
        high = torch.quantile(finite, float(clip_high_quantile))
        effective = effective.clamp(min=float(low), max=float(high))
    fallback = finite.median() if finite.numel() else torch.tensor(1.0e-6, device=raw.device, dtype=raw.dtype)
    effective = torch.where(torch.isfinite(effective) & (effective > 0), effective, fallback)
    return effective.clamp_min(1.0e-6)


def compute_robust_residual_scale(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-neighbourhood (median, 1.4826*MAD) of already-normalised residuals."""
    median = residual.median(dim=1, keepdim=True).values
    mad = (residual - median).abs().median(dim=1).values
    return median.reshape(-1), 1.4826 * mad


def adaptive_huber_delta(
    residual: torch.Tensor,
    *,
    factor: float = 2.5,
    minimum: float = 0.05,
) -> torch.Tensor:
    """Robust per-neighbourhood outlier threshold, in effective-scale units.

    A fixed threshold accepts essentially every neighbour once residuals are
    normalised, which makes the robust stage a no-op and the inlier fraction
    a constant 1.  Scaling it to each neighbourhood's own residual spread
    makes "outlier" mean what it should: inconsistent *relative to how well
    this neighbourhood fits*.  The floor keeps a perfectly rigid
    neighbourhood (MAD = 0) from declaring all of its neighbours outliers.
    """
    _median, robust_sigma = compute_robust_residual_scale(residual)
    return torch.clamp(float(factor) * robust_sigma, min=float(minimum))


def compute_motion_gate(
    motion_magnitude: torch.Tensor,
    *,
    threshold: float = 1.0,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Soft gate that is ~0 for low-motion neighbourhoods."""
    temperature = max(float(temperature), 1.0e-6)
    return torch.sigmoid((motion_magnitude - float(threshold)) / temperature)


def compute_motion_gated_rigid_evidence(
    motion_magnitude: torch.Tensor,
    rigid_failure_fraction: torch.Tensor,
    *,
    threshold: float = 1.0,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Positive only when a neighbourhood *moves* and moves rigidly.

    A 6-DOF rigid model has more freedom than the identity, so it fits noise
    better even where nothing moved: ``E_R < E_S`` on its own is not evidence
    of motion.  Gating by motion magnitude removes that bias -- a static
    neighbourhood contributes ~0 no matter how much better the rigid model
    fits its noise.
    """
    gate = compute_motion_gate(motion_magnitude, threshold=threshold, temperature=temperature)
    return gate * (0.5 - rigid_failure_fraction)


def closest_rotation(J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Closest proper rotation to each 3x3 matrix, plus its singular values."""
    U, S, Vh = torch.linalg.svd(J.double())
    det = torch.linalg.det(U @ Vh)
    correction = torch.ones_like(S)
    correction[..., -1] = torch.sign(det).clamp(min=-1.0, max=1.0)
    correction = torch.where(correction == 0, torch.ones_like(correction), correction)
    R = U @ torch.diag_embed(correction) @ Vh
    return R.to(J.dtype), S.to(J.dtype)


def huber_weight(residual: torch.Tensor, delta: float | torch.Tensor) -> torch.Tensor:
    """Huber weights; ``delta`` may be a scalar or broadcastable per-row tensor."""
    r = residual.clamp_min(0.0)
    if torch.is_tensor(delta):
        d = delta.to(device=r.device, dtype=r.dtype).clamp_min(1.0e-8)
    else:
        d = torch.full_like(r, max(float(delta), 1.0e-8))
    return torch.where(r <= d, torch.ones_like(r), d / r.clamp_min(1.0e-8))


def _weighted_fit(
    neighbor_xyz: torch.Tensor,     # [N,K,3]
    neighbor_z: torch.Tensor,       # [N,K,3]
    weight: torch.Tensor,           # [N,K]
    *,
    jacobian_reg: float,
    scale: torch.Tensor,            # [N]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One weighted local (J, R, t) fit. Returns J, R, S, t, centroids."""
    mass = weight.sum(dim=1, keepdim=True).clamp_min(1.0e-12)          # [N,1]
    x_bar = (weight[..., None] * neighbor_xyz).sum(dim=1) / mass       # [N,3]
    z_bar = (weight[..., None] * neighbor_z).sum(dim=1) / mass         # [N,3]
    dx = neighbor_xyz - x_bar[:, None, :]                               # [N,K,3]
    dz = neighbor_z - z_bar[:, None, :]                                 # [N,K,3]
    wdx = weight[..., None] * dx
    A = torch.einsum("nka,nkb->nab", wdx, dx)                           # [N,3,3]
    B = torch.einsum("nka,nkb->nab", weight[..., None] * dz, dx)        # [N,3,3]
    # Regularise in the local length unit so the same setting means the same
    # thing on a millimetre-scale and a metre-scale reconstruction.
    reg = (float(jacobian_reg) * scale.square()).clamp_min(1.0e-12)     # [N]
    eye = torch.eye(3, device=A.device, dtype=A.dtype).expand_as(A)
    A_reg = A + reg[:, None, None] * eye
    # J A = B  =>  A J^T = B^T (A symmetric) -- solved, never inverted.
    J = torch.linalg.solve(A_reg, B.transpose(-1, -2)).transpose(-1, -2)
    R, S = closest_rotation(J)
    t = z_bar - torch.einsum("nab,nb->na", R, x_bar)
    return J, R, S, t, A, x_bar


@torch.no_grad()
def fit_local_deformation_at(
    neighbor_xyz: torch.Tensor,
    neighbor_disp: torch.Tensor,
    *,
    center_xyz: torch.Tensor,
    neighbor_distance: torch.Tensor | None = None,
    neighbor_confidence: torch.Tensor | None = None,
    local_scale: torch.Tensor | None = None,
    jacobian_reg: float = 1.0e-5,
    irls_iters: int = 2,
    huber_delta: float = 1.0,
    robust_factor: float = 2.5,
    huber_delta_min: float = 0.05,
    spatial_sigma_scale: float = 2.0,
    min_valid_neighbors: int = 6,
    degeneracy_ratio: float = 1.0e-3,
    confidence_floor: float = 0.25,
) -> LocalDeformationResult:
    """Fit a local rigid motion model around arbitrary centre locations.

    ``neighbor_*`` are already-gathered ``[Q,K,...]`` tensors, so this works
    both for cloud members (Stage D) and for transported continuous query
    positions (Stage J option B) without a second code path.

    ``local_scale`` should be the *effective* (neighbourhood-pooled, clipped)
    scale -- every normalised quantity this returns uses it as its unit.
    """
    neighbor_xyz = torch.as_tensor(neighbor_xyz).detach()
    neighbor_disp = torch.as_tensor(neighbor_disp).detach().to(neighbor_xyz)
    center_xyz = torch.as_tensor(center_xyz).detach().to(neighbor_xyz)
    q, k = int(neighbor_xyz.shape[0]), int(neighbor_xyz.shape[1])
    device, dtype = neighbor_xyz.device, neighbor_xyz.dtype

    finite_neighbor = torch.isfinite(neighbor_xyz).all(-1) & torch.isfinite(neighbor_disp).all(-1)
    neighbor_disp = torch.where(finite_neighbor[..., None], neighbor_disp, torch.zeros_like(neighbor_disp))
    neighbor_z = neighbor_xyz + neighbor_disp

    if neighbor_distance is None:
        neighbor_distance = torch.linalg.norm(neighbor_xyz - center_xyz[:, None, :], dim=-1)
    if local_scale is None:
        local_scale = local_scale_from_knn(neighbor_distance)
    local_scale = torch.as_tensor(local_scale).detach().to(device=device, dtype=dtype).reshape(-1).clamp_min(1.0e-6)
    local_scale = torch.where(torch.isfinite(local_scale), local_scale, torch.full_like(local_scale, 1.0e-6))

    sigma = (float(spatial_sigma_scale) * local_scale).clamp_min(1.0e-6)
    w_space = torch.exp(-0.5 * neighbor_distance.square() / sigma[:, None].square())
    w_space = torch.where(finite_neighbor, w_space, torch.zeros_like(w_space))

    if neighbor_confidence is None:
        w_conf = torch.ones((q, k), device=device, dtype=dtype)
    else:
        w_conf = torch.as_tensor(neighbor_confidence).detach().to(device=device, dtype=dtype).clamp(0.0, 1.0)
        # Never let confidence alone erase a neighbourhood: an early wrong
        # label must not be able to delete the support that would correct it.
        w_conf = float(confidence_floor) + (1.0 - float(confidence_floor)) * w_conf
    # Weights used for BOTH hypotheses in the competition below. They are
    # deliberately free of any rigid-residual term: reweighting by the rigid
    # model's own residual and then asking whether the rigid model fits
    # better than static would be judging one hypothesis on its own evidence.
    w_base = w_space * w_conf

    weight = w_base
    J = R = S = t = A = None
    delta = torch.full((q,), float(huber_delta), device=device, dtype=dtype)
    for _ in range(max(1, int(irls_iters))):
        J, R, S, t, A, _x_bar = _weighted_fit(
            neighbor_xyz, neighbor_z, weight, jacobian_reg=jacobian_reg, scale=local_scale
        )
        predicted = torch.einsum("nab,nkb->nka", R, neighbor_xyz) + t[:, None, :]
        residual = torch.linalg.norm(neighbor_z - predicted, dim=-1) / local_scale[:, None].clamp_min(1.0e-8)
        delta = adaptive_huber_delta(residual, factor=float(robust_factor), minimum=float(huber_delta_min))
        weight = w_base * huber_weight(residual, delta[:, None])

    predicted = torch.einsum("nab,nkb->nka", R, neighbor_xyz) + t[:, None, :]
    residual = torch.linalg.norm(neighbor_z - predicted, dim=-1) / local_scale[:, None].clamp_min(1.0e-8)
    # Same normalised residual and same adaptive threshold as the weighting,
    # so the inlier fraction actually reports what the robust stage did.
    inlier = (residual <= delta[:, None]) & finite_neighbor
    inlier_fraction = inlier.to(dtype).sum(dim=1) / max(1, k)

    base_mass = w_base.sum(dim=1).clamp_min(1.0e-12)
    static_error = (w_base * torch.linalg.norm(neighbor_z - neighbor_xyz, dim=-1).square()).sum(dim=1) / base_mass
    rigid_error = (w_base * torch.linalg.norm(neighbor_z - predicted, dim=-1).square()).sum(dim=1) / base_mass
    rigid_gain = static_error - rigid_error

    strain = (S - 1.0).square().sum(dim=-1)

    # Leave-one-centre-out by construction: the centre never entered the fit,
    # so this is a genuine prediction of its motion from its neighbours, and
    # a false-static centre inside a moving part can still be recovered.
    predicted_center = torch.einsum("nab,nb->na", R, center_xyz) + t
    predicted_disp = predicted_center - center_xyz
    motion_magnitude = torch.linalg.norm(predicted_disp, dim=-1) / local_scale.clamp_min(1.0e-8)

    rigid_failure_fraction = torch.where(
        (rigid_error + static_error) > 0,
        rigid_error / (rigid_error + static_error).clamp_min(1.0e-30),
        torch.full_like(rigid_error, 0.5),
    )

    support_count = (w_base > 0).sum(dim=1)
    eig = torch.linalg.eigvalsh(A.double()).clamp_min(0.0)
    spread = (eig[:, 0] / eig[:, 2].clamp_min(1.0e-30)).to(dtype)
    condition = (eig[:, 2] / eig[:, 0].clamp_min(1.0e-30)).to(dtype)
    valid = (
        (support_count >= int(min_valid_neighbors))
        & (spread >= float(degeneracy_ratio))
        & torch.isfinite(J).all(-1).all(-1)
        & torch.isfinite(t).all(-1)
        & torch.isfinite(predicted_disp).all(-1)
        & torch.isfinite(strain)
        & torch.isfinite(rigid_gain)
    )

    zero3 = torch.zeros(3, device=device, dtype=dtype)
    eye3 = torch.eye(3, device=device, dtype=dtype)
    # A degenerate neighbourhood must not inject an arbitrary rotation.
    R = torch.where(valid[:, None, None], R, eye3.expand_as(R))
    J = torch.where(valid[:, None, None], J, eye3.expand_as(J))
    t = torch.where(valid[:, None], t, center_xyz - torch.einsum("ab,nb->na", eye3, center_xyz))
    predicted_disp = torch.where(valid[:, None], predicted_disp, zero3.expand_as(predicted_disp))
    motion_magnitude = torch.where(valid, motion_magnitude, torch.zeros_like(motion_magnitude))

    return LocalDeformationResult(
        J=J,
        R=R,
        t=t,
        singular_values=S,
        strain_error=strain,
        rigid_fit_error=rigid_error,
        static_fit_error=static_error,
        rigid_gain=rigid_gain,
        rigid_failure_fraction=rigid_failure_fraction,
        predicted_disp=predicted_disp,
        motion_magnitude=motion_magnitude,
        inlier_fraction=inlier_fraction,
        robust_delta=delta,
        effective_scale=local_scale,
        cov_eigenvalues=eig.to(dtype),
        cov_condition=condition,
        valid=valid,
    )


@torch.no_grad()
def fit_local_deformation(
    xyz: torch.Tensor,
    displacement: torch.Tensor,
    knn_indices: torch.Tensor,
    point_confidence: torch.Tensor | None = None,
    local_scale: torch.Tensor | None = None,
    *,
    knn_distance: torch.Tensor | None = None,
    jacobian_reg: float = 1.0e-5,
    irls_iters: int = 2,
    huber_delta: float = 1.0,
    robust_factor: float = 2.5,
    huber_delta_min: float = 0.05,
    leave_center_out: bool = True,
    spatial_sigma_scale: float = 2.0,
    min_valid_neighbors: int = 6,
    degeneracy_ratio: float = 1.0e-3,
    confidence_floor: float = 0.25,
) -> LocalDeformationResult:
    """Fit a local rigid motion model around every point of one cloud."""
    xyz = torch.as_tensor(xyz).detach()
    displacement = torch.as_tensor(displacement).detach().to(xyz)
    knn_indices = torch.as_tensor(knn_indices).detach().to(device=xyz.device).long()
    neighbor_xyz = xyz[knn_indices]
    neighbor_disp = displacement[knn_indices]
    neighbor_conf = None if point_confidence is None else torch.as_tensor(point_confidence).detach().to(xyz)[knn_indices]
    if not bool(leave_center_out):
        # Keeping the centre in its own fit is only meaningful as an ablation:
        # it lets a false-static centre hold its own neighbourhood static.
        neighbor_xyz = torch.cat((xyz[:, None, :], neighbor_xyz), dim=1)
        neighbor_disp = torch.cat((displacement[:, None, :], neighbor_disp), dim=1)
        if neighbor_conf is not None and point_confidence is not None:
            center_conf = torch.as_tensor(point_confidence).detach().to(xyz)[:, None]
            neighbor_conf = torch.cat((center_conf, neighbor_conf), dim=1)
        if knn_distance is not None:
            knn_distance = torch.cat((torch.zeros_like(knn_distance[:, :1]), knn_distance), dim=1)
    return fit_local_deformation_at(
        neighbor_xyz,
        neighbor_disp,
        center_xyz=xyz,
        neighbor_distance=knn_distance,
        neighbor_confidence=neighbor_conf,
        local_scale=local_scale,
        jacobian_reg=jacobian_reg,
        irls_iters=irls_iters,
        huber_delta=huber_delta,
        robust_factor=robust_factor,
        huber_delta_min=huber_delta_min,
        spatial_sigma_scale=spatial_sigma_scale,
        min_valid_neighbors=min_valid_neighbors,
        degeneracy_ratio=degeneracy_ratio,
        confidence_floor=confidence_floor,
    )


@torch.no_grad()
def denoise_displacement_field(
    displacement: torch.Tensor,
    knn_indices: torch.Tensor,
    knn_distance: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    local_scale: torch.Tensor | None = None,
    spatial_sigma_scale: float = 2.0,
    huber_delta: float = 1.0,
    iters: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Robustly denoise a displacement field, returning (displacement, sigma).

    This operates on displacements, never on discrete endpoint identities.
    It is the deterministic stand-in for the learned displacement-diffusion
    mode: a confidence-weighted, Huber-robust local consensus, which is what
    a denoiser is being asked to supply here (a cleaner, physically
    plausible field plus an uncertainty).  A learned denoiser can replace it
    behind the same signature without touching anything downstream.
    """
    displacement = torch.as_tensor(displacement).detach()
    knn_indices = torch.as_tensor(knn_indices).detach().to(device=displacement.device).long()
    if local_scale is None:
        local_scale = local_scale_from_knn(knn_distance)
    local_scale = local_scale.to(displacement).reshape(-1).clamp_min(1.0e-6)
    sigma = (float(spatial_sigma_scale) * local_scale).clamp_min(1.0e-6)
    w_space = torch.exp(-0.5 * knn_distance.to(displacement).square() / sigma[:, None].square())
    if confidence is not None:
        w_space = w_space * torch.as_tensor(confidence).detach().to(displacement).clamp(0.0, 1.0)[knn_indices]

    current = displacement.clone()
    for _ in range(max(1, int(iters))):
        neighbor_disp = current[knn_indices]
        deviation = torch.linalg.norm(neighbor_disp - current[:, None, :], dim=-1) / local_scale[:, None]
        weight = w_space * huber_weight(deviation, huber_delta)
        mass = weight.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        current = (weight[..., None] * neighbor_disp).sum(dim=1) / mass

    neighbor_disp = current[knn_indices]
    spread = torch.linalg.norm(neighbor_disp - current[:, None, :], dim=-1)
    uncertainty = spread.median(dim=1).values / local_scale.clamp_min(1.0e-8)
    current = torch.where(torch.isfinite(current).all(-1, keepdim=True), current, displacement)
    uncertainty = torch.where(torch.isfinite(uncertainty), uncertainty, torch.zeros_like(uncertainty))
    return current, uncertainty
