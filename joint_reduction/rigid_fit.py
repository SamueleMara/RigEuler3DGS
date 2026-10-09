"""Section 28 -- rigid motion compression.

The free per-Gaussian IEKF of sections 4-17 is a bootstrap stage only.  Once a
group of Gaussians moves coherently we compress its ``12N``-dimensional state
into a single 6-vector twist

    v_i ~= v0 + omega x (mu_i - c),        xi = [v0; omega] in R^6

and hand the result to the prismatic/revolute reduction of section 29.  Every
solve here is closed form weighted least squares: no Adam, no autograd, no
``loss.backward()`` (section 0 rules 4/5 and section 36).

Conventions
-----------
* The twist is ordered ``[v0; omega]`` -- translation first -- to match the
  ``[delta mu; delta phi]`` ordering of the 6D pose error state of section 4 and
  the pose tangents ``G`` of section 29.  ``rigid_parts.robust_inner_killing``
  uses the opposite ``[omega; u]`` order; the two orderings are never mixed, and
  nothing in this package imports that module.
* ``c`` is the reduction centre and ``v0`` is the velocity *at* ``c``.  Moving
  the centre re-parameterises the twist as ``v0(c') = v0(c) + omega x (c' - c)``;
  the rigid field itself is unchanged.  The default centre is the weighted
  centroid because it makes the translation and rotation blocks of the normal
  matrix orthogonal (``sum_i w_i (mu_i - c) = 0``), which is the only cheap thing
  one can do about the conditioning of this problem.
* Rank deficiency is *reported*, never hidden: the solve is a truncated
  eigen-pseudo-inverse, so an unobservable direction returns zero instead of a
  confident number amplified by a tiny eigenvalue.
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch

EPS = 1.0e-12
#: Huber / Tukey tuning constants at the usual 95% Gaussian efficiency.
HUBER_DELTA = 1.345
TUKEY_C = 4.685


# ---------------------------------------------------------------------------
# Small linear-algebra helpers (self-contained on purpose: `joint_reduction`
# must not import `estimator/`, which is being written in parallel).
# ---------------------------------------------------------------------------


def hat(v: torch.Tensor) -> torch.Tensor:
    """``[v]_x`` for a batch of 3-vectors, so that ``hat(v) @ w == cross(v, w)``."""
    v = torch.as_tensor(v)
    if v.shape[-1] != 3:
        raise ValueError(f"hat expects [...,3], got {tuple(v.shape)}")
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        (
            torch.stack((zero, -v[..., 2], v[..., 1]), dim=-1),
            torch.stack((v[..., 2], zero, -v[..., 0]), dim=-1),
            torch.stack((-v[..., 1], v[..., 0], zero), dim=-1),
        ),
        dim=-2,
    )


def rigid_design(mu: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Per-point design ``A_i = [I, -[mu_i - c]_x]`` in ``R^{3x6}`` (section 28).

    The minus sign is forced by the twist ordering: ``-[r]_x omega = omega x r``,
    so ``A_i [v0; omega] = v0 + omega x (mu_i - c)`` exactly as written in the
    spec.  This is the *per-point* design -- the dense Jacobian of rule 6 is the
    pixel-level ``H``, which is never formed anywhere in this package.
    """
    r = torch.as_tensor(mu) - torch.as_tensor(center).reshape(3)
    n = r.shape[0]
    eye = torch.eye(3, device=r.device, dtype=r.dtype).expand(n, 3, 3)
    return torch.cat((eye, -hat(r)), dim=2)


def _sym(m: torch.Tensor) -> torch.Tensor:
    """Kill the asymmetry that accumulates in ``sum A^T W A`` before ``eigh``."""
    return 0.5 * (m + m.transpose(-1, -2))


def _truncated_eigen_solve(
    mat: torch.Tensor, rhs: torch.Tensor, rank_tol: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Minimum-norm solve of ``mat x = rhs`` for a symmetric PSD ``mat``.

    Directions whose eigenvalue falls below ``rank_tol * lambda_max`` are *not*
    inverted.  A rank-deficient normal matrix (pure translation of a collinear
    group, a single Gaussian, a static group) then yields exactly zero along the
    unobservable direction instead of noise divided by ~0, and the caller still
    sees the deficiency through the returned rank/eigenvalues.
    """
    evals, evecs = torch.linalg.eigh(_sym(mat))
    lam_max = evals.max().clamp_min(0.0)
    tol = rank_tol * lam_max
    keep = evals > tol
    inv = torch.where(keep, 1.0 / torch.where(keep, evals, torch.ones_like(evals)),
                      torch.zeros_like(evals))
    sol = evecs @ (inv * (evecs.transpose(-1, -2) @ rhs))
    pinv = evecs @ torch.diag(inv) @ evecs.transpose(-1, -2)
    return sol, evals, pinv, int(keep.sum().item())


def _condition_number(evals: torch.Tensor, rank_tol: float) -> float:
    lam_max = float(evals.max().item())
    lam_min = float(evals.min().item())
    if lam_max <= 0.0 or lam_min <= rank_tol * lam_max:
        return math.inf
    return lam_max / lam_min


def _jacobi_scaled(mat: torch.Tensor) -> torch.Tensor:
    """``D^-1/2 M D^-1/2`` with ``D = diag(M)``.

    The twist blocks carry different units (``1`` for ``v0``, ``length`` for
    ``omega``), so the raw condition number of the normal matrix is a function of
    the scene scale and is useless as a threshold.  Symmetric Jacobi scaling
    removes that, which makes `condition_number_scaled` the number to actually
    threshold on.
    """
    d = torch.diagonal(mat).clamp_min(0.0)
    safe = torch.where(d > 0, d, torch.ones_like(d))
    inv_sqrt = torch.where(d > 0, safe.rsqrt(), torch.zeros_like(d))
    return _sym(mat * inv_sqrt[:, None] * inv_sqrt[None, :])


def _resolve_information(
    cov: torch.Tensor | None, n: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor | None:
    """Turn a velocity covariance into the per-point information ``W_i``.

    ``None`` is propagated instead of materialising ``N`` identity blocks: at a
    million Gaussians that would be 72 MB of nothing, and the unweighted einsum
    path below is both cheaper and better conditioned.
    """
    if cov is None:
        return None
    cov = torch.as_tensor(cov, device=device, dtype=dtype)
    if cov.dim() == 2 and cov.shape == (3, 3):
        cov = cov.expand(n, 3, 3)
    elif cov.dim() == 2 and cov.shape[-1] == 3:
        cov = torch.diag_embed(cov)  # [N,3] = per-axis variances
    elif cov.dim() == 1 and cov.numel() == n:
        eye = torch.eye(3, device=device, dtype=dtype)
        cov = cov.reshape(n, 1, 1) * eye
    if cov.shape != (n, 3, 3):
        raise ValueError(f"cov must broadcast to [{n},3,3], got {tuple(cov.shape)}")
    # Cholesky inverse rather than `inv`: the covariance is SPD by construction
    # and a failed factorisation is a real error we want to surface, not absorb.
    chol, info = torch.linalg.cholesky_ex(_sym(cov))
    if int(info.max().item()) != 0:
        raise ValueError("velocity covariance is not positive definite")
    return torch.cholesky_inverse(chol)


def _whitened_sq(res: torch.Tensor, info: torch.Tensor | None) -> torch.Tensor:
    """``r_i^T W_i r_i`` per point."""
    if info is None:
        return (res * res).sum(dim=-1)
    return torch.einsum("na,nab,nb->n", res, info, res).clamp_min(0.0)


def _robust_weights(u: torch.Tensor, kind: str, scale: torch.Tensor) -> torch.Tensor:
    """IRLS attenuation from the whitened residual norms ``u``."""
    ratio = u / scale
    if kind == "huber":
        return torch.clamp(HUBER_DELTA / ratio.clamp_min(EPS), max=1.0)
    if kind == "tukey":
        t = (ratio / TUKEY_C).clamp(max=1.0)
        return (1.0 - t * t) ** 2
    raise ValueError(f"unknown robust kernel {kind!r}")


# ---------------------------------------------------------------------------
# Section 28 -- the fit itself
# ---------------------------------------------------------------------------


@torch.no_grad()
def fit_rigid_twist(
    mu: torch.Tensor,
    v: torch.Tensor,
    weights: torch.Tensor | None = None,
    cov: torch.Tensor | None = None,
    *,
    center: torch.Tensor | None = None,
    robust: str | None = None,
    irls_iters: int = 8,
    ridge: float = 0.0,
    rank_tol: float = 1.0e-10,
) -> dict[str, Any]:
    """Weighted least-squares rigid twist ``xi = [v0; omega]`` (section 28).

    Minimises ``sum_i w_i (v_i - A_i xi)^T W_i (v_i - A_i xi)`` with
    ``A_i = [I, -[mu_i - c]_x]`` and ``W_i = cov_i^-1`` (identity when ``cov`` is
    ``None``), i.e. the filtered velocities are fused with their own IEKF
    uncertainty rather than being treated as equally trustworthy.

    Parameters
    ----------
    mu, v : ``[N,3]`` centres and filtered velocities of one candidate group.
    weights : ``[N]`` non-negative membership/confidence weights (default 1).
    cov : ``[N,3,3]`` (or ``[N,3]`` diagonal, ``[N]`` isotropic, ``[3,3]`` shared)
        velocity covariances.  ``[3,3]`` is read as one shared full covariance,
        so with exactly three Gaussians pass ``[N,3,3]`` to stay unambiguous.
    center : reduction centre ``c``; defaults to the weighted centroid.
    robust : ``None``, ``"huber"`` or ``"tukey"``.  Both run IRLS; ``"tukey"`` is
        non-convex so it is seeded with a few Huber iterations, which is the
        standard remedy for its sensitivity to the starting point.
    ridge : optional Tikhonov term added to the normal matrix.  Left at 0 by
        default -- the truncated pseudo-inverse already handles deficiency, and a
        ridge would *hide* it behind a finite condition number.

    Returns a dict with the twist, its split ``v0``/``omega``, the centre, the
    weighted and per-point residuals, the 6x6 normal matrix with its
    eigenvalues/condition numbers/rank, the IRLS weights, and the
    static-vs-moving evidence.
    """
    mu = torch.as_tensor(mu)
    v = torch.as_tensor(v, device=mu.device, dtype=mu.dtype)
    if mu.dim() != 2 or mu.shape[1] != 3:
        raise ValueError(f"mu must be [N,3], got {tuple(mu.shape)}")
    if v.shape != mu.shape:
        raise ValueError(f"v must match mu {tuple(mu.shape)}, got {tuple(v.shape)}")
    n = mu.shape[0]
    if n == 0:
        raise ValueError("fit_rigid_twist needs at least one Gaussian")
    device, dtype = mu.device, mu.dtype

    if weights is None:
        base_w = torch.ones(n, device=device, dtype=dtype)
    else:
        base_w = torch.as_tensor(weights, device=device, dtype=dtype).reshape(n).clamp_min(0.0)
    if float(base_w.sum().item()) <= 0.0:
        raise ValueError("all weights are zero")

    info = _resolve_information(cov, n, device, dtype)

    if center is None:
        c = (base_w[:, None] * mu).sum(dim=0) / base_w.sum()
    else:
        c = torch.as_tensor(center, device=device, dtype=dtype).reshape(3)

    design = rigid_design(mu, c)  # [N,3,6]
    eye6 = torch.eye(6, device=device, dtype=dtype)

    # Data scale used as the floor for the robust scale: a group whose inliers
    # are *noiseless* drives the MAD to exactly zero, and 0/0 would turn every
    # point into an outlier on the next IRLS sweep.
    scale_floor = (1.0e-9 * v.norm(dim=-1).median() + EPS).to(dtype)

    kinds: list[str | None]
    if robust is None:
        kinds = [None]
    elif robust == "huber":
        kinds = ["huber"] * max(1, irls_iters)
    elif robust == "tukey":
        seed = max(1, irls_iters // 3)
        kinds = ["huber"] * seed + ["tukey"] * max(1, irls_iters - seed)
    else:
        raise ValueError(f"unknown robust kernel {robust!r}")

    rob_w = torch.ones(n, device=device, dtype=dtype)
    twist = torch.zeros(6, device=device, dtype=dtype)
    normal = torch.zeros(6, 6, device=device, dtype=dtype)
    rhs = torch.zeros(6, device=device, dtype=dtype)
    evals = torch.zeros(6, device=device, dtype=dtype)
    pinv = torch.zeros(6, 6, device=device, dtype=dtype)
    rank = 0
    res = torch.zeros_like(v)
    sq = torch.zeros(n, device=device, dtype=dtype)
    iterations = 0

    for kind in kinds:
        w = base_w * rob_w
        if info is None:
            normal = torch.einsum("n,nai,naj->ij", w, design, design)
            rhs = torch.einsum("n,nai,na->i", w, design, v)
        else:
            normal = torch.einsum("n,nai,nab,nbj->ij", w, design, info, design)
            rhs = torch.einsum("n,nai,nab,nb->i", w, design, info, v)
        normal = _sym(normal)
        twist, evals, pinv, rank = _truncated_eigen_solve(
            normal + ridge * eye6, rhs, rank_tol
        )
        res = v - torch.einsum("nai,i->na", design, twist)
        sq = _whitened_sq(res, info)
        iterations += 1
        if kind is None:
            break
        u = sq.clamp_min(0.0).sqrt()
        # MAD about zero: `u` is a non-negative residual magnitude, so the usual
        # median-of-absolute-deviations-about-the-median would under-estimate the
        # spread of the inlier half.
        scale = torch.maximum(1.4826 * u.median(), scale_floor)
        rob_w = _robust_weights(u, kind, scale)

    w_final = base_w * rob_w
    w_sum = w_final.sum().clamp_min(EPS)
    chi2_t = (w_final * sq).sum()
    chi2 = float(chi2_t.item())
    residual = float((chi2_t / w_sum).clamp_min(0.0).sqrt().item())
    per_point = res.norm(dim=-1)

    # Static evidence: xi = 0 is a feasible point of the same objective, so
    # chi2_static >= chi2_fit always and the ratio lives in [0,1].  1 means the
    # twist bought nothing (static group), 0 means the motion is entirely
    # explained by the twist.  A group that is exactly static drives both to 0,
    # which is the 0/0 the guard below resolves in favour of "static".
    sq_static = _whitened_sq(v, info)
    chi2_static = float((w_final * sq_static).sum().item())
    if chi2_static <= EPS * max(1.0, abs(chi2)):
        static_evidence = 1.0
    else:
        static_evidence = min(1.0, max(0.0, chi2 / chi2_static))

    scaled = _jacobi_scaled(normal)
    evals_scaled = torch.linalg.eigvalsh(scaled)

    return {
        "twist": twist,
        "v0": twist[:3],
        "omega": twist[3:],
        "center": c,
        "residual": residual,
        "residual_chi2": chi2,
        "per_point_residual": per_point,
        "per_point_chi2": sq,
        "velocity_fit": v - res,
        "normal_matrix": normal,
        "normal_rhs": rhs,
        "eigenvalues": evals,
        "eigenvalues_scaled": evals_scaled,
        "condition_number": _condition_number(evals, rank_tol),
        "condition_number_scaled": _condition_number(evals_scaled, rank_tol),
        "rank": rank,
        "rank_deficient": rank < 6,
        "twist_covariance": pinv,
        "weights": w_final,
        "robust_weights": rob_w,
        "robust": robust,
        "iterations": iterations,
        "static_evidence": static_evidence,
        "static_chi2": chi2_static,
        "num_gaussians": n,
    }


@torch.no_grad()
def membership_compatibility(
    mu: torch.Tensor,
    v: torch.Tensor,
    twist: torch.Tensor,
    c: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Per-Gaussian compatibility with a rigid twist, in ``[0,1]`` (section 28).

    ``exp(-||v_i - (v0 + omega x (mu_i - c))||^2 / (2 sigma^2))`` -- the Gaussian
    likelihood ratio of "this Gaussian belongs to the part" against a perfect
    fit, which is exactly the membership weight to feed back into
    :func:`fit_rigid_twist` for the next EM/IRLS sweep.  ``sigma`` is the
    expected velocity disagreement (scalar or ``[N]``).
    """
    mu = torch.as_tensor(mu)
    v = torch.as_tensor(v, device=mu.device, dtype=mu.dtype)
    twist = torch.as_tensor(twist, device=mu.device, dtype=mu.dtype).reshape(6)
    design = rigid_design(mu, torch.as_tensor(c, device=mu.device, dtype=mu.dtype))
    res = v - torch.einsum("nai,i->na", design, twist)
    s = torch.as_tensor(sigma, device=mu.device, dtype=mu.dtype)
    if bool((s <= 0).any().item()):
        raise ValueError("sigma must be positive")
    s = s.reshape(-1) if s.dim() else s.reshape(())
    z = res.norm(dim=-1) / s
    return torch.exp(-0.5 * z * z)


# ---------------------------------------------------------------------------
# Debug export (section 28: "debug visualization/export")
# ---------------------------------------------------------------------------


def rigid_fit_report(fit: dict[str, Any]) -> str:
    """One-line-per-fact human summary for the diagnostics log of section 34."""
    cond = fit["condition_number_scaled"]
    lines = [
        f"rigid twist fit over N={fit['num_gaussians']} gaussians"
        f" (robust={fit['robust']}, irls_iters={fit['iterations']})",
        f"  v0      = {[round(float(x), 6) for x in fit['v0']]}",
        f"  omega   = {[round(float(x), 6) for x in fit['omega']]}",
        f"  center  = {[round(float(x), 6) for x in fit['center']]}",
        f"  residual(weighted rms) = {fit['residual']:.6e}   chi2 = {fit['residual_chi2']:.6e}",
        f"  normal matrix rank = {fit['rank']}/6"
        + ("  <-- RANK DEFICIENT" if fit["rank_deficient"] else ""),
        f"  cond(raw) = {fit['condition_number']:.6e}   cond(jacobi) = {cond:.6e}",
        f"  eigenvalues = {[f'{float(x):.3e}' for x in fit['eigenvalues']]}",
        f"  static_evidence = {fit['static_evidence']:.6f}"
        f" (chi2_static = {fit['static_chi2']:.6e})",
    ]
    return "\n".join(lines)


def export_rigid_fit(fit: dict[str, Any], path: str) -> dict[str, str]:
    """Dump a fit for offline inspection: ``<path>.npz`` + ``<path>.json``.

    Per-Gaussian arrays go to the npz (residuals and IRLS weights are what one
    actually wants to colour a point cloud by); the scalars that decide
    accept/reject go to the json so they are greppable across a sweep.
    """
    import numpy as np

    stem = path[:-4] if path.endswith((".npz", ".json")) else path
    arrays = {
        k: fit[k].detach().cpu().numpy()
        for k in (
            "twist", "v0", "omega", "center", "per_point_residual", "per_point_chi2",
            "velocity_fit", "normal_matrix", "normal_rhs", "eigenvalues",
            "eigenvalues_scaled", "twist_covariance", "weights", "robust_weights",
        )
    }
    np.savez(stem + ".npz", **arrays)
    scalars = {
        k: fit[k]
        for k in (
            "residual", "residual_chi2", "condition_number", "condition_number_scaled",
            "rank", "rank_deficient", "static_evidence", "static_chi2", "iterations",
            "num_gaussians", "robust",
        )
    }
    scalars["twist"] = [float(x) for x in fit["twist"]]
    scalars["center"] = [float(x) for x in fit["center"]]
    with open(stem + ".json", "w", encoding="utf-8") as handle:
        json.dump(scalars, handle, indent=2, sort_keys=True)
    return {"npz": stem + ".npz", "json": stem + ".json"}
