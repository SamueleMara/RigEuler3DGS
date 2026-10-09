"""P4/P5 -- "can that intrinsic match be explained by one rigid transform?"

Metric isometry is necessary but not sufficient.  A patch can be metrically
compatible with a target patch and still not be reachable from it by any
element of SE(3): a mirrored patch has identical internal distances, and so
does a patch matched to a rearrangement of a symmetric one.  Only fitting an
actual ``R, t`` and measuring what it leaves behind can tell those apart.

Which is why the determinant correction below is load-bearing rather than
decorative.  Without it the SVD happily returns a reflection, that reflection
fits a mirrored patch perfectly, and the pipeline accepts a motion no rigid
body can perform.

``E_SE3`` is kept separate from ``E_metric`` and never summed with it (§22).
They are consecutive feasibility tests, so a candidate must pass both; a
weighted sum would let a very isometric candidate buy its way past a rigid
residual it should have failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

EPS = 1.0e-12


@dataclass
class RigidHypothesis:
    """A batch of rigid transforms recovered from soft patch couplings."""

    rotation: torch.Tensor      # [B, 3, 3]
    translation: torch.Tensor   # [B, 3]
    residual: torch.Tensor      # [B] E_SE3, in units of patch scale squared
    mass: torch.Tensor          # [B] coupling mass the fit used
    degenerate: torch.Tensor    # [B] bool -- the fit was not determined
    stats: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def weighted_kabsch_from_soft_coupling(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    coupling: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(R, t, mass)`` from a soft coupling, per §21.

    The soft coupling is used directly rather than being hardened into
    correspondences first.  There is no Gaussian identity between the two
    endpoint models, so a hard assignment would be inventing one; the
    weighted centroids and cross-covariance are exactly the sufficient
    statistics the soft plan provides.
    """
    source_points = torch.as_tensor(source_points).detach().float()
    target_points = torch.as_tensor(target_points).detach().float().to(source_points.device)
    coupling = torch.as_tensor(coupling).detach().float().to(source_points.device)

    mass = coupling.sum(dim=(1, 2)).clamp_min(EPS)                 # [B]
    p = coupling.sum(dim=2)                                        # [B, m]
    q = coupling.sum(dim=1)                                        # [B, n]
    mean_x = (p[:, :, None] * source_points).sum(dim=1) / mass[:, None]
    mean_y = (q[:, :, None] * target_points).sum(dim=1) / mass[:, None]

    centred_x = source_points - mean_x[:, None, :]
    centred_y = target_points - mean_y[:, None, :]
    # H = sum_ac Pi_ac (x_a - xbar)(y_c - ybar)^T, as one batched product.
    cross = centred_x.transpose(1, 2) @ coupling @ centred_y        # [B, 3, 3]

    u, singular, vh = torch.linalg.svd(cross.double())
    v = vh.transpose(1, 2)
    # det correction: without it the SVD returns a reflection whenever that
    # fits better, and a reflection is not a motion any rigid body can make.
    sign = torch.sign(torch.linalg.det(v @ u.transpose(1, 2)))
    correction = torch.diag_embed(torch.stack([
        torch.ones_like(sign), torch.ones_like(sign), sign
    ], dim=1))
    rotation = (v @ correction @ u.transpose(1, 2)).float()
    translation = mean_y - torch.einsum("bij,bj->bi", rotation, mean_x)

    # A rank-deficient cross-covariance means the rotation is not determined
    # by this patch; reporting it as a confident transform would be a lie.
    degenerate = (singular[:, 2] / singular[:, 0].clamp_min(1.0e-30)) < 1.0e-6
    return rotation, translation, mass, degenerate


@torch.no_grad()
def compute_se3_residual(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    coupling: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    *,
    patch_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """§22's ``E_SE3``, normalised by the source patch intrinsic scale squared.

    The normalisation is what makes the number comparable between a dense
    patch and a sparse one; without it a large patch would always look worse
    simply for spanning more space.
    """
    source_points = torch.as_tensor(source_points).detach().float()
    target_points = torch.as_tensor(target_points).detach().float().to(source_points.device)
    moved = torch.einsum("bij,bkj->bki", rotation, source_points) + translation[:, None, :]
    squared = torch.cdist(moved, target_points).square()            # [B, m, n]
    mass = coupling.sum(dim=(1, 2)).clamp_min(EPS)
    residual = (coupling * squared).sum(dim=(1, 2)) / mass
    if patch_scale is not None:
        residual = residual / torch.as_tensor(patch_scale).detach().to(residual).reshape(-1).clamp_min(EPS).square()
    return residual


@torch.no_grad()
def fit_rigid_hypothesis(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    coupling: torch.Tensor,
    *,
    patch_scale: torch.Tensor | None = None,
) -> RigidHypothesis:
    """P4 end to end: weighted Kabsch, then its own residual."""
    rotation, translation, mass, degenerate = weighted_kabsch_from_soft_coupling(
        source_points, target_points, coupling
    )
    residual = compute_se3_residual(
        source_points, target_points, coupling, rotation, translation, patch_scale=patch_scale
    )
    return RigidHypothesis(
        rotation=rotation,
        translation=translation,
        residual=residual,
        mass=mass,
        degenerate=degenerate,
        stats={
            "median_residual": float(residual[~degenerate].median()) if bool((~degenerate).any()) else float("nan"),
            "degenerate_fraction": float(degenerate.float().mean()),
        },
    )


ALIGNMENT_LADDER = (0.3, 0.1, 0.03, 0.01, 0.003)


@torch.no_grad()
def refine_alignment_by_residual(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    *,
    patch_scale: torch.Tensor | None = None,
    ladder: tuple[float, ...] = ALIGNMENT_LADDER,
    steps_per_level: int = 6,
):
    """Minimise §22's residual directly, annealing the coupling as it goes.

    GW cannot do this job: distortion is a function of distances alone, so a
    reflection is exactly as good as the truth under it, and driving the
    solver harder only finds better reflections.  The rigid residual is not
    reflection-invariant, so it is the quantity to descend on once metric
    compatibility has been established -- which is the lexicographic order
    §4 requires, not a weighted combination of the two.

    Annealing matters: at a fixed soft epsilon every alignment reports a
    similar inflated residual and the comparison between restarts carries no
    information.  The ladder ends sharp so the selection is made where the
    residual actually discriminates.
    """
    from utils.local_metric_isometry import coupling_from_alignment

    for epsilon in ladder:
        for _ in range(max(1, int(steps_per_level))):
            coupling = coupling_from_alignment(
                source_points, target_points, rotation, translation,
                source_valid, target_valid, patch_scale=patch_scale, epsilon=float(epsilon),
            )
            rotation, translation, _mass, _degenerate = weighted_kabsch_from_soft_coupling(
                source_points, target_points, coupling
            )
    coupling = coupling_from_alignment(
        source_points, target_points, rotation, translation,
        source_valid, target_valid, patch_scale=patch_scale, epsilon=float(ladder[-1]),
    )
    residual = compute_se3_residual(
        source_points, target_points, coupling, rotation, translation, patch_scale=patch_scale
    )
    return rotation, translation, coupling, residual


@torch.no_grad()
def solve_isometry_with_rigid_selection(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    d0: torch.Tensor,
    dt: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    patch_scale: torch.Tensor | None = None,
    angles: int = 12,
    screen_to: int = 3,
    **solver_kwargs,
):
    """Metric isometry tests compatibility; the rigid residual picks the motion.

    The lexicographic composition §4 requires, with each stage doing the job
    it can actually do:

      * GW supplies ``E_metric`` (§19) -- a reflection-invariant statement
        about whether the two patches are isometric at all, which is exactly
        what the §20 test needs and all GW can honestly provide;
      * the transform is then recovered by descending §22's own residual from
        several frame restarts, because that quantity is not
        reflection-invariant and can therefore separate a motion from its
        mirror image.

    The two numbers are never summed.  A pair must be metrically compatible
    to be considered, and among the restarts the rigid residual chooses.

    Restarts are evaluated as ONE batch rather than in a loop.  They are
    independent by construction, the patches are tiny (tens of points), and
    the cost here is kernel-launch latency rather than arithmetic: looping 24
    restarts and 3 refinements cost 147 sequential solver calls and 10,560
    sequential Sinkhorn iterations for a single 16x16 evaluation.  Stacking
    them into the batch dimension changes no result and removes almost all of
    that.

    Returns ``(coupling, distortion, RigidHypothesis)``.
    """
    from utils.local_metric_isometry import (
        build_pca_frame_initialisations,
        coupling_from_alignment,
        solve_partial_local_gromov_wasserstein,
    )

    # E_metric: the metric-compatibility statement, solved once from the
    # unbiased product coupling so it does not inherit any restart's bias.
    metric = solve_partial_local_gromov_wasserstein(
        d0, dt, source_valid=source_valid, target_valid=target_valid, **solver_kwargs
    )

    starts = build_pca_frame_initialisations(
        source_points, target_points, source_valid, target_valid, angles=int(angles)
    )
    restarts, batch = int(starts.shape[0]), int(source_points.shape[0])

    def tile(tensor, count):
        """Repeat along a new leading axis and fold it into the batch."""
        return tensor[None].expand(count, *tensor.shape).reshape(count * tensor.shape[0], *tensor.shape[1:])

    all_source = tile(source_points, restarts)
    all_target = tile(target_points, restarts)
    all_svalid = tile(source_valid, restarts)
    all_tvalid = tile(target_valid, restarts)
    all_scale = tile(patch_scale, restarts) if patch_scale is not None else None
    flat = starts.reshape(restarts * batch, 4, 3)

    # Screening: every restart in a single solver call.
    screen_coupling = coupling_from_alignment(
        all_source, all_target, flat[:, :3, :], flat[:, 3, :],
        all_svalid, all_tvalid, patch_scale=all_scale, epsilon=0.01,
    )
    screen = compute_se3_residual(
        all_source, all_target, screen_coupling, flat[:, :3, :], flat[:, 3, :], patch_scale=all_scale
    ).reshape(restarts, batch)

    keep = min(max(1, int(screen_to)), restarts)
    ranking = screen.argsort(dim=0)[:keep]                       # [keep, B]
    index = (ranking * batch + torch.arange(batch, device=ranking.device)[None, :]).reshape(-1)

    # Refinement: all surviving restarts of all patches, still one batch.
    picked = flat[index]
    kept_source = tile(source_points, keep)
    kept_target = tile(target_points, keep)
    kept_svalid = tile(source_valid, keep)
    kept_tvalid = tile(target_valid, keep)
    kept_scale = tile(patch_scale, keep) if patch_scale is not None else None

    rotation, translation, coupling, residual = refine_alignment_by_residual(
        kept_source, kept_target, kept_svalid, kept_tvalid,
        picked[:, :3, :], picked[:, 3, :], patch_scale=kept_scale,
    )
    _r, _t, _mass, degenerate = weighted_kabsch_from_soft_coupling(kept_source, kept_target, coupling)

    # A degenerate fit must not win by posting an artificially low residual.
    score = torch.where(degenerate, torch.full_like(residual, float("inf")), residual).reshape(keep, batch)
    winner = score.argmin(dim=0)                                  # [B]
    take = winner * batch + torch.arange(batch, device=winner.device)

    hypothesis = RigidHypothesis(
        rotation=rotation[take],
        translation=translation[take],
        residual=residual[take],
        mass=coupling[take].sum(dim=(1, 2)),
        degenerate=degenerate[take],
        stats={
            "restarts": restarts,
            "refined": keep,
            "median_residual": float(residual[take].median()),
            "median_distortion": float(metric.distortion.median()),
        },
    )
    return coupling[take], metric.distortion, hypothesis


@torch.no_grad()
def build_se3_null_distribution(
    source_points: torch.Tensor,
    target_pool_points: torch.Tensor,
    couplings: torch.Tensor,
    *,
    patch_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """§23: the rigid residual achieved by metric-invalid / randomised pairs.

    Built from the same null draws the metric test used, so the two tests
    describe the same alternative hypothesis rather than two different ones.
    Returns ``[B, samples]``.
    """
    samples = int(target_pool_points.shape[1])
    out = torch.empty((int(source_points.shape[0]), samples), device=source_points.device, dtype=torch.float32)
    for s in range(samples):
        hypothesis = fit_rigid_hypothesis(
            source_points, target_pool_points[:, s], couplings[:, s], patch_scale=patch_scale
        )
        out[:, s] = hypothesis.residual
    return out


@torch.no_grad()
def rigid_hypothesis_test(
    residual: torch.Tensor,
    null_residual: torch.Tensor,
    *,
    alpha: float = 0.05,
) -> dict[str, torch.Tensor | float]:
    """Empirical p-value of ``E_SE3``, corrected with the same fixed FDR."""
    from utils.local_metric_isometry import benjamini_hochberg

    residual = torch.as_tensor(residual).detach().float().reshape(-1)
    null = torch.as_tensor(null_residual).detach().float()
    better = (null <= residual[:, None]).sum(dim=1).float()
    p_value = (better + 1.0) / (float(null.shape[1]) + 1.0)
    accepted, cut = benjamini_hochberg(p_value, alpha=float(alpha))
    return {
        "p_value": p_value,
        "accepted": accepted,
        "fdr_cut": cut,
        "alpha": float(alpha),
        "null_median": null.median(dim=1).values,
        "effect": null.median(dim=1).values - residual,
    }
