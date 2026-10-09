"""P3 -- "is there a target patch with the same intrinsic metric?"

This is the theoretical core.  A rigid motion changes every absolute
coordinate but preserves internal distances, so the question of where a
moved patch went is a question about metric structure alone:

    find Pi minimising  sum_abcd Pi_ac Pi_bd (D0_ab - DT_cd)^2

Nothing in this module may see an absolute position, an appearance, or a
displacement (§51).  Those are later stages, and letting any of them in here
would let a candidate that is not isometric be rescued by looking similar --
which is exactly the failure the whole architecture is built to prevent.

Acceptance is statistical, not a threshold.  "Distortion below 0.1" has no
meaning that carries between scenes; "distortion that this scene's own
non-matching patch pairs essentially never achieve" does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

EPS = 1.0e-12


@dataclass
class GromovWassersteinResult:
    """A batch of solved patch couplings and their unregularised distortion."""

    coupling: torch.Tensor      # [B, m, n] transport plan, partial
    distortion: torch.Tensor    # [B] E_metric, evaluated unregularised
    mass: torch.Tensor          # [B] total transported mass
    iterations: int
    # (epsilon, iteration, median unregularised distortion) per solver step.
    # The trace is the only way to tell "converged" from "never moved", and
    # those two produce identical final numbers.
    trace: list[tuple[float, int, float]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _gw_gradient(d0: torch.Tensor, dt: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
    """Gradient of the squared-loss GW objective w.r.t. the coupling.

    For ``L(a,b) = (a-b)^2`` the fourth-order tensor factorises (Peyre et
    al. 2016) into ``f1(D0) p 1^T + 1 q^T f2(DT)^T - 2 D0 Pi DT^T``, which
    is what makes this tractable at all -- the naive form is ``O(m^2 n^2)``
    per evaluation and would put the whole method out of reach.
    """
    p = coupling.sum(dim=2, keepdim=True)                    # [B, m, 1]
    q = coupling.sum(dim=1, keepdim=True)                    # [B, 1, n]
    constant = d0.square() @ p + (q @ dt.square().transpose(1, 2))
    return 2.0 * (constant - 2.0 * (d0 @ coupling @ dt.transpose(1, 2)))


@torch.no_grad()
def compute_metric_distortion(
    d0: torch.Tensor,
    dt: torch.Tensor,
    coupling: torch.Tensor,
) -> torch.Tensor:
    """§19's ``E_metric``, always the UNREGULARISED distortion.

    Reported separately from whatever the solver optimised, per §18: an
    entropic solver's own objective includes a term that has nothing to do
    with isometry, so scoring candidates by it would rank them partly by how
    diffuse their coupling is.
    """
    p = coupling.sum(dim=2, keepdim=True)
    q = coupling.sum(dim=1, keepdim=True)
    constant = (d0.square() @ p * p).sum(dim=(1, 2)) + ((q @ dt.square().transpose(1, 2)) * q).sum(dim=(1, 2))
    cross = (coupling * (d0 @ coupling @ dt.transpose(1, 2))).sum(dim=(1, 2))
    total = constant - 2.0 * cross
    normaliser = (p.sum(dim=(1, 2)) * q.sum(dim=(1, 2))).clamp_min(EPS)
    return (total / normaliser).clamp_min(0.0)


def _semi_relaxed_sinkhorn(
    cost: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    row_equality: torch.Tensor,
    *,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Entropic OT: equality on the smaller side, per-point CAPS on the larger.

    This is what makes the transport partial without introducing a mass
    parameter.  The two endpoint patches contain different numbers of points
    -- independent densification and pruning guarantee it -- so the smaller
    patch is fully explained while the larger one is allowed to leave points
    unmatched, which is exactly what occlusion and resampling produce.

    Capping BOTH sides instead is degenerate and silently so: the duals can
    then never grow to meet any marginal, the cheapest feasible plan is the
    empty one, and the solver returns a near-zero-mass coupling that the
    Kabsch fit downstream happily turns into a confident garbage transform.
    Measured here, mass collapsed to 0.005 and the recovered rotation was at
    maximum possible error while reporting no degeneracy at all.
    """
    log_k = -cost / max(float(epsilon), 1.0e-12)
    log_u = torch.zeros_like(p)
    log_v = torch.zeros_like(q)
    log_p, log_q = torch.log(p.clamp_min(EPS)), torch.log(q.clamp_min(EPS))
    zero_u, zero_v = torch.zeros_like(log_u), torch.zeros_like(log_v)
    rows = row_equality[:, None]
    for _ in range(max(1, int(iterations))):
        raw_u = log_p - torch.logsumexp(log_k + log_v[:, None, :], dim=2)
        log_u = torch.where(rows, raw_u, torch.minimum(raw_u, zero_u))
        raw_v = log_q - torch.logsumexp(log_k + log_u[:, :, None], dim=1)
        log_v = torch.where(rows, torch.minimum(raw_v, zero_v), raw_v)
    return torch.exp(log_k + log_u[:, :, None] + log_v[:, None, :])


@torch.no_grad()
def build_pca_frame_initialisations(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    angles: int = 8,
) -> torch.Tensor:
    """Deterministic initial alignments sweeping the patch's own normal frame.

    Distance-only matching is blind to reflection.  A patch and its mirror
    image have identical internal distances, so the GW objective genuinely
    cannot tell them apart -- measured here, driving the solver harder took
    the distortion from 0.032 down to 0.0144, at or below the true
    correspondence's own 0.009-0.017, while the recovered rotation stayed
    maximally wrong.  That is a degeneracy of the criterion, not a failure of
    the solver, and no amount of optimisation will fix it.

    So the restarts have to bracket the right basin and let §22's rigid
    residual choose, which it can: the true alignment's ``E_SE3`` was 0.010
    against 0.036-0.15 for the reflected ones.

    Enumerating the four proper sign flips of the PCA frame is not enough.
    A surface patch has a well-determined normal but near-isotropic tangent
    eigenvalues, so its in-plane axes are essentially arbitrary and the four
    flips all inherit the same arbitrary frame.  Sweeping rotations about the
    normal covers that degeneracy instead; ``angles`` is an accuracy/runtime
    control, not a physical parameter.

    Returns ``[2 * angles, B, 4, 3]`` -- rows 0-2 rotation, row 3 translation.
    """
    source_points = torch.as_tensor(source_points).detach().float()
    target_points = torch.as_tensor(target_points).detach().float().to(source_points.device)
    ws = source_valid.float()[:, :, None]
    wt = target_valid.float()[:, :, None]

    mean_s = (source_points * ws).sum(1) / ws.sum(1).clamp_min(EPS)
    mean_t = (target_points * wt).sum(1) / wt.sum(1).clamp_min(EPS)
    cs = (source_points - mean_s[:, None, :]) * ws
    ct = (target_points - mean_t[:, None, :]) * wt
    # eigh returns ascending eigenvalues, so column 0 is the patch normal.
    # Reorder to (t1, t2, n) and make each frame right-handed.
    frame_s = _right_handed(torch.linalg.eigh((cs.transpose(1, 2) @ cs).double())[1].flip(-1))
    frame_t = _right_handed(torch.linalg.eigh((ct.transpose(1, 2) @ ct).double())[1].flip(-1))

    packed = []
    for step in range(max(1, int(angles))):
        theta = 2.0 * torch.pi * float(step) / float(max(1, int(angles)))
        cos, sin = torch.cos(torch.tensor(theta, dtype=frame_s.dtype)), torch.sin(torch.tensor(theta, dtype=frame_s.dtype))
        spin = torch.tensor(
            [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
            dtype=frame_s.dtype, device=frame_s.device,
        )
        # diag(1,-1,-1) is itself a proper rotation (by pi about the first
        # tangent axis), so flipping the normal this way stays inside SO(3).
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=frame_s.dtype, device=frame_s.device))
        for mirror in (torch.eye(3, dtype=frame_s.dtype, device=frame_s.device), flip):
            rotation = (frame_t @ spin @ mirror @ frame_s.transpose(1, 2)).float()
            translation = mean_t - torch.einsum("bij,bj->bi", rotation, mean_s)
            packed.append(torch.cat((rotation, translation[:, None, :]), dim=1))
    return torch.stack(packed, dim=0)


def _right_handed(frame: torch.Tensor) -> torch.Tensor:
    """Flip the last axis where needed so the basis has determinant +1."""
    determinant = torch.linalg.det(frame)
    correction = torch.ones_like(frame)
    correction[:, :, 2] = torch.where(determinant < 0, -1.0, 1.0)[:, None]
    return frame * correction


@torch.no_grad()
def coupling_from_alignment(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    patch_scale: torch.Tensor | None = None,
    epsilon: float = 0.01,
    iterations: int = 80,
) -> torch.Tensor:
    """A soft coupling implied by a candidate rigid alignment.

    Soft rather than nearest-neighbour: a hard assignment is a vertex of the
    transport polytope, and conditional gradient started from a vertex has
    nowhere to descend.

    ``epsilon`` controls how soft, and it matters more than it looks.  A
    diffuse coupling inflates every residual towards the same value and
    destroys the comparison between candidate alignments: measured with the
    TRUE transform on one scene, the residual read 0.116 at epsilon 0.1 and
    0.026 at 0.003, against 0.010 for a hard assignment.  Candidates were
    then being ranked by how diffuse their coupling was rather than by how
    well they explained the geometry, so the selection must always be made
    at the sharp end of the ladder.
    """
    moved = torch.einsum("bij,bkj->bki", rotation, source_points) + translation[:, None, :]
    cost = torch.cdist(moved, target_points).square()
    if patch_scale is not None:
        cost = cost / patch_scale[:, None, None].clamp_min(EPS).square()
    else:
        cost = cost / cost.flatten(1).median(dim=1).values[:, None, None].clamp_min(EPS)
    source_count = source_valid.float().sum(1).clamp_min(1.0)
    target_count = target_valid.float().sum(1).clamp_min(1.0)
    unit = 1.0 / torch.minimum(source_count, target_count)
    return _semi_relaxed_sinkhorn(
        cost, source_valid.float() * unit[:, None], target_valid.float() * unit[:, None],
        source_count <= target_count, epsilon=float(epsilon), iterations=int(iterations),
    )


@torch.no_grad()
def solve_partial_local_gromov_wasserstein(
    d0: torch.Tensor,
    dt: torch.Tensor,
    *,
    source_valid: torch.Tensor | None = None,
    target_valid: torch.Tensor | None = None,
    initial_coupling: torch.Tensor | None = None,
    outer_iterations: int = 30,
    inner_iterations: int = 40,
    continuation: tuple[float, ...] = (1.0, 0.3, 0.1, 0.03, 0.01),
    tolerance: float = 1.0e-7,
    trace: bool = False,
) -> GromovWassersteinResult:
    """Conditional-gradient partial GW over a batch of patch pairs.

    §18 forbids an exposed entropy coefficient.  The solver anneals epsilon
    down a FIXED deterministic ladder to the smallest numerically stable
    value and evaluates the result with the unregularised distortion, so the
    regularisation is an internal numerical device and changing the ladder
    cannot change which hypotheses are accepted beyond tolerance.

    All inputs are already normalised by their own patch scale, so ``d0``
    and ``dt`` are dimensionless and one ladder serves every scene.
    """
    d0 = torch.as_tensor(d0).detach().float()
    dt = torch.as_tensor(dt).detach().float().to(d0.device)
    b, m, _ = d0.shape
    n = int(dt.shape[1])

    source_mask = torch.ones((b, m), device=d0.device) if source_valid is None else source_valid.float()
    target_mask = torch.ones((b, n), device=d0.device) if target_valid is None else target_valid.float()
    source_count = source_mask.sum(dim=1).clamp_min(1.0)
    target_count = target_mask.sum(dim=1).clamp_min(1.0)
    # One unit of mass per point of the SMALLER patch. Both sides then carry
    # the same per-point mass, so "unmatched" means a point of the larger
    # patch had no counterpart -- a physical statement, not a budget choice.
    unit = 1.0 / torch.minimum(source_count, target_count)
    p = source_mask * unit[:, None]
    q = target_mask * unit[:, None]
    row_equality = source_count <= target_count

    if initial_coupling is None:
        coupling = p[:, :, None] * q[:, None, :]
        coupling = coupling / coupling.sum(dim=(1, 2), keepdim=True).clamp_min(EPS)
    else:
        coupling = torch.as_tensor(initial_coupling).detach().float().to(d0.device)
    total_iterations = 0
    objective: list[tuple[float, int, float]] = []
    for epsilon in continuation:
        for _ in range(max(1, int(outer_iterations) // max(1, len(continuation)))):
            total_iterations += 1
            gradient = _gw_gradient(d0, dt, coupling)
            # Rescaled per patch so one epsilon ladder means the same thing
            # for every patch; the LP limit is unchanged by a positive scale.
            scale = gradient.abs().flatten(1).max(dim=1).values.clamp_min(EPS)
            direction = _semi_relaxed_sinkhorn(
                gradient / scale[:, None, None], p, q, row_equality,
                epsilon=float(epsilon), iterations=int(inner_iterations),
            )
            # Exact line search: the GW objective is quadratic along the
            # segment, so the optimal step has a closed form and no step-size
            # parameter is needed.
            delta = direction - coupling
            a2 = -2.0 * (delta * (d0 @ delta @ dt.transpose(1, 2))).sum(dim=(1, 2))
            a1 = (delta * _gw_gradient(d0, dt, coupling)).sum(dim=(1, 2))
            step = torch.where(a2 > EPS, (-a1 / (2.0 * a2)).clamp(0.0, 1.0), (a1 < 0).float())
            new_coupling = coupling + step[:, None, None] * delta
            moved = float((new_coupling - coupling).abs().max())
            coupling = new_coupling
            if trace:
                objective.append((
                    float(epsilon), total_iterations,
                    float(compute_metric_distortion(d0, dt, coupling).median()),
                ))
            if moved < float(tolerance):
                break

    distortion = compute_metric_distortion(d0, dt, coupling)
    return GromovWassersteinResult(
        coupling=coupling,
        distortion=distortion,
        mass=coupling.sum(dim=(1, 2)),
        iterations=total_iterations,
        trace=objective,
        stats={"epsilon_ladder": list(continuation), "median_distortion": float(distortion.median())},
    )


@torch.no_grad()
def build_metric_null_distribution(
    source_d0: torch.Tensor,
    target_pool_dt: torch.Tensor,
    *,
    samples: int = 16,
    generator: torch.Generator | None = None,
    **solver_kwargs,
) -> torch.Tensor:
    """§20: what distortion does a NON-matching patch pair achieve here?

    Drawn from the same allowed dynamic target domain as the real candidates,
    because the null has to answer "how exceptional is this match among the
    matches this scene could have produced".  A null built from unrelated
    geometry would make every candidate look significant.

    Returns ``[B, samples]``.
    """
    b = int(source_d0.shape[0])
    pool = int(target_pool_dt.shape[0])
    generator = generator or torch.Generator(device="cpu").manual_seed(0)
    picks = torch.randint(0, pool, (b, int(samples)), generator=generator).to(source_d0.device)

    out = torch.empty((b, int(samples)), device=source_d0.device, dtype=source_d0.dtype)
    for s in range(int(samples)):
        result = solve_partial_local_gromov_wasserstein(source_d0, target_pool_dt[picks[:, s]], **solver_kwargs)
        out[:, s] = result.distortion
    return out


def benjamini_hochberg(p_values: torch.Tensor, *, alpha: float = 0.05) -> tuple[torch.Tensor, float]:
    """FDR control across all tested source patches (§20).

    Thousands of patches are tested, so an uncorrected 5% cut would accept
    hundreds of pure-noise matches and every downstream stage would inherit
    them.  BH is fixed globally and never tuned per dataset (§46).
    """
    p = torch.as_tensor(p_values).detach().float().reshape(-1)
    n = int(p.numel())
    if n == 0:
        return torch.zeros(0, dtype=torch.bool), 0.0
    order = torch.argsort(p)
    ranked = p[order]
    limits = float(alpha) * torch.arange(1, n + 1, device=p.device, dtype=p.dtype) / float(n)
    passed = ranked <= limits
    if not bool(passed.any()):
        return torch.zeros(n, dtype=torch.bool, device=p.device), 0.0
    cut = float(ranked[int(torch.nonzero(passed).max())])
    return p <= cut, cut


@torch.no_grad()
def metric_hypothesis_test(
    distortion: torch.Tensor,
    null_distortion: torch.Tensor,
    *,
    alpha: float = 0.05,
) -> dict[str, torch.Tensor | float]:
    """Empirical p-value of the observed distortion against the scene's null.

    The ``+1`` in the numerator and denominator is the standard finite-sample
    correction: with a null of 16 draws, an uncorrected estimate can report
    p = 0 and claim more significance than the sample can support.
    """
    distortion = torch.as_tensor(distortion).detach().float().reshape(-1)
    null = torch.as_tensor(null_distortion).detach().float()
    better = (null <= distortion[:, None]).sum(dim=1).float()
    p_value = (better + 1.0) / (float(null.shape[1]) + 1.0)
    accepted, cut = benjamini_hochberg(p_value, alpha=float(alpha))
    return {
        "p_value": p_value,
        "accepted": accepted,
        "fdr_cut": cut,
        "alpha": float(alpha),
        "null_median": null.median(dim=1).values,
        "effect": null.median(dim=1).values - distortion,
    }
