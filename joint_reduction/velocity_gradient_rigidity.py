"""Sections 8-14: the instantaneous (differential) half of the dual theory.

CODEX_IMPLEMENT_DUAL_DIFFERENTIAL_FINITE_JOINT_DISCOVERY.md sections 8-14.

Around a reference point c a smooth velocity field is, to first order,

    v(x, t) = v_c(t) + G_t (x - c),        G_t = grad_x v,

and G splits into

    D = (G + G^T)/2        (rate of deformation)
    Omega = (G - G^T)/2    (rigid rotation rate),

with D = 0 and Omega = [omega]_x exactly when the field is a rigid-body
velocity field. That single fact is what this module measures: D says whether
the support really moves as one rigid body AT THIS INSTANT, and omega says
whether it is turning.

Section 2 is emphatic about what this decomposition may NOT be applied to. For
a finite transform x' = Rx + t the displacement field is u(x) = (R - I)x + t,
and sym(R - I) is NOT zero for a finite rotation even though the motion is
perfectly rigid -- a 90-degree turn would be reported as enormous strain. The
decomposition belongs to the velocity field alone; finite motion stays on
SE(3) and is handled by `finite_relative_rotation.py`. The synthetic test
`test_53_finite_rotation_is_not_strain` exists to keep that line drawn.
"""

from __future__ import annotations

import math

import torch

from estimator.manifold import so3_hat, so3_vee

from .joint_information import information_inverse

DIRECT = "direct"
CENTRAL = "central_difference"
NONUNIFORM = "nonuniform_difference"
ONE_SIDED = "one_sided"

# vech ordering used everywhere in this module for the symmetric part:
VECH_INDEX = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


# ---------------------------------------------------------------------------
# section 8: instantaneous velocity from Phase-1 tracks
# ---------------------------------------------------------------------------
def velocity_weight_matrix(times) -> tuple[torch.Tensor, list]:
    """The linear operator W with v = W x, plus the rule used at each frame.

    Returning the operator rather than the velocities is what makes section 9
    possible: the same W propagates the position covariance to the velocity
    covariance, so the uncertainty of a one-sided boundary estimate is not
    quietly assumed equal to that of an interior central difference.

    Interior frames use the three-point rule for the actual spacing, which
    reduces to the familiar (x[t+1] - x[t-1]) / (2 dt) on a uniform grid.
    Boundaries use the three-point one-sided rule, which is second order --
    a first-order difference there would bias exactly the frames where a
    joint's motion starts and stops.
    """
    t = torch.as_tensor(times, dtype=torch.float64).reshape(-1)
    T = int(t.numel())
    if T < 2:
        raise ValueError("section 8: at least two frames are needed for a "
                         "velocity estimate")
    W = torch.zeros((T, T), dtype=torch.float64)
    rules = []
    uniform = bool(T < 3 or torch.allclose(t[1:] - t[:-1],
                                           (t[1] - t[0]).expand(T - 1),
                                           rtol=1e-9, atol=1e-12))
    if T == 2:
        dt = float(t[1] - t[0])
        W[0, 0], W[0, 1] = -1.0 / dt, 1.0 / dt
        W[1, 0], W[1, 1] = -1.0 / dt, 1.0 / dt
        return W, [ONE_SIDED, ONE_SIDED]

    for k in range(T):
        if 0 < k < T - 1:
            h1 = float(t[k] - t[k - 1])
            h2 = float(t[k + 1] - t[k])
            W[k, k - 1] = -h2 / (h1 * (h1 + h2))
            W[k, k] = (h2 - h1) / (h1 * h2)
            W[k, k + 1] = h1 / (h2 * (h1 + h2))
            rules.append(CENTRAL if uniform else NONUNIFORM)
        else:
            # three-point one-sided, second order, for the real spacing
            i0, i1, i2 = (0, 1, 2) if k == 0 else (T - 1, T - 2, T - 3)
            h1 = float(t[i1] - t[i0])
            h2 = float(t[i2] - t[i0])
            det = h1 * h2 * (h2 - h1)
            W[k, i0] = -(h2 * h2 - h1 * h1) / det
            W[k, i1] = h2 * h2 / det
            W[k, i2] = -h1 * h1 / det
            rules.append(ONE_SIDED)
    return W, rules


def instantaneous_velocity(tracks, *, times=None, dt: float = 1.0,
                           position_covariance=None,
                           velocity=None) -> dict:
    """Section 8 + 9: velocities, with their covariance, from Phase-1 tracks.

    `tracks` is [T, N, 3]. If Phase 1 already supplies velocities, pass them as
    `velocity` and they are used unchanged (`velocity_source = "direct"`).

    `position_covariance` is [T, N, 3, 3] (or [T, N] isotropic variances).
    Section 9 forbids giving every Gaussian the same confidence when better
    information exists, so the covariance is propagated through the SAME
    operator that produced the velocity rather than assumed. Frames are taken
    to be independently estimated, which makes Sigma_v[t] = sum_k W[t,k]^2
    Sigma_x[k]; that is an assumption about Phase 1, and it is recorded in the
    result as `covariance_model`.
    """
    tracks = torch.as_tensor(tracks).double()
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError("section 8: tracks must be [T, N, 3]")
    T, N = tracks.shape[0], tracks.shape[1]
    if times is None:
        times = torch.arange(T, dtype=torch.float64) * float(dt)
    W, rules = velocity_weight_matrix(times)

    if velocity is not None:
        v = torch.as_tensor(velocity).double().reshape(T, N, 3)
        source = [DIRECT] * T
    else:
        v = torch.einsum("tk,kni->tni", W, tracks)
        source = rules

    out = {"velocity": v, "weight_matrix": W, "velocity_source": source,
           "times": torch.as_tensor(times, dtype=torch.float64).reshape(-1)}

    if position_covariance is None:
        out["velocity_covariance"] = None
        out["velocity_information"] = None
        out["covariance_model"] = ("none supplied: callers fall back to "
                                   "isotropic unit weights (section 9 allows "
                                   "this only when no directional information "
                                   "exists)")
        return out

    cov = torch.as_tensor(position_covariance).double()
    if cov.ndim == 2:                                   # [T, N] isotropic
        eye = torch.eye(3, dtype=torch.float64, device=cov.device)
        cov = cov[..., None, None] * eye
    if cov.shape[:2] != (T, N) or cov.shape[-2:] != (3, 3):
        raise ValueError("section 9: position_covariance must be [T,N,3,3] "
                         "or [T,N]")
    if velocity is not None:
        vcov = cov
        model = "direct Phase-1 velocity covariance, used unchanged"
    else:
        vcov = torch.einsum("tk,knij->tnij", W * W, cov)
        model = ("propagated through the section 8 difference operator, "
                 "frames assumed independently estimated")
    out["velocity_covariance"] = vcov
    out["velocity_information"] = torch.linalg.pinv(vcov, hermitian=True)
    out["covariance_model"] = model
    return out


# ---------------------------------------------------------------------------
# section 10: the affine velocity field, by weighted least squares
# ---------------------------------------------------------------------------
def _design(r: torch.Tensor) -> torch.Tensor:
    """B_i = [I3 | r_i^T (x) I3], so B_i theta = v_c + G r_i.

    The Kronecker block is laid out so that the G half of theta is the
    COLUMN-MAJOR vec of G: (r^T (x) I3) vec_col(G) = r0 G[:,0] + r1 G[:,1] +
    r2 G[:,2] = G r. `test_10_design_matrix_recovers_G` pins this down, because
    getting the convention backwards silently transposes G and swaps D with
    Omega's sign.
    """
    n = r.shape[0]
    eye = torch.eye(3, dtype=torch.float64, device=r.device)
    blocks = [eye.expand(n, 3, 3)]
    for k in range(3):
        blocks.append(r[:, k, None, None] * eye)
    return torch.cat(blocks, dim=2)                       # [N, 3, 12]


def fit_affine_velocity_field(x, v, *, information=None, weights=None,
                              centre=None) -> dict:
    """Section 10: v_i = v_c + G (x_i - c), weighted least squares.

    Solved with an explicit pseudoinverse so that rank and singular values are
    reported rather than hidden. Section 10 forbids covering a rank deficiency
    with strong Tikhonov damping: a support that does not span three
    dimensions genuinely cannot determine all nine entries of G, and saying so
    is the useful answer.
    """
    x = torch.as_tensor(x).double()
    v = torch.as_tensor(v).double()
    n = x.shape[0]
    if weights is None:
        w = torch.ones(n, dtype=torch.float64, device=x.device)
    else:
        w = torch.as_tensor(weights).double().reshape(-1).clamp_min(0.0)
    if information is None:
        eye = torch.eye(3, dtype=torch.float64, device=x.device)
        lam = w[:, None, None] * eye
    else:
        lam = torch.as_tensor(information).double() * w[:, None, None]

    c = ((w[:, None] * x).sum(0) / w.sum().clamp_min(1e-300)
         if centre is None else torch.as_tensor(centre).double().reshape(3))
    r = x - c
    B = _design(r)
    A = torch.einsum("nij,nik,nkl->jl", B, lam, B)
    b = torch.einsum("nij,nik,nk->j", B, lam, v)
    inv = information_inverse(A)
    theta = inv["inverse"] @ b
    G = theta[3:].reshape(3, 3).T                 # undo the column-major vec
    pred = torch.einsum("nij,j->ni", B, theta)
    resid = v - pred
    return {"centre": c, "v_c": theta[:3], "G": G, "theta": theta,
            "information": A, "covariance": inv["inverse"],
            "rank": inv["rank"], "condition": inv["condition"],
            "spectrum": inv["spectrum"], "null_basis": inv["null_basis"],
            "residual": resid,
            "weighted_residual": float(
                torch.einsum("ni,nij,nj->", resid, lam, resid)),
            "num_points": int(n)}


# ---------------------------------------------------------------------------
# sections 11, 12: the decomposition
# ---------------------------------------------------------------------------
def decompose_velocity_gradient(G) -> dict:
    """Section 11: G = D + Omega, and omega = vee(Omega).

    Section 12 names the mistake this must not make. It is NOT
    "diagonal = strain, off-diagonal = rotation". The correct reading is:

      * the diagonal of D is normal/volumetric deformation rate;
      * the SYMMETRIC off-diagonal of D is shear -- deformation, not rotation;
      * only the ANTISYMMETRIC part Omega is rigid rotation;
      * translation is the constant v_c and is not part of G at all.

    A pure shear therefore has large off-diagonal entries in G and yet
    Omega = 0, and must never be read as a turning body. `test_52_pure_shear`
    enforces exactly that.
    """
    G = torch.as_tensor(G).double()
    D = 0.5 * (G + G.transpose(-1, -2))
    Om = 0.5 * (G - G.transpose(-1, -2))
    omega = so3_vee(Om)
    return {"D": D, "Omega": Om, "omega": omega,
            "strain_norm": float(D.norm()),
            "spin_norm": float(Om.norm()),
            "volumetric_rate": float(D.diagonal(dim1=-2, dim2=-1).sum(-1)),
            "convention": ("omega = vee(Omega) with the repository's "
                           "so3_hat/so3_vee pair; v = v_c + omega x r")}


def vech_selector(device=None) -> torch.Tensor:
    """S with vech(D) = S vec_col(G), for propagating the fit covariance."""
    S = torch.zeros((6, 9), dtype=torch.float64, device=device)
    for row, (i, j) in enumerate(VECH_INDEX):
        a = j * 3 + i                      # G[i,j] in column-major vec
        b = i * 3 + j                      # G[j,i]
        if a == b:
            S[row, a] = 1.0
        else:
            S[row, a] = S[row, b] = 0.5
    return S


def strain_covariance(affine: dict) -> dict:
    """Section 13: Sigma_D from the affine fit's own covariance."""
    S = vech_selector(affine["covariance"].device)
    cov_g = affine["covariance"][3:, 3:]
    return {"vech_selector": S, "covariance": S @ cov_g @ S.T}


def strain_statistic(D, cov_D) -> dict:
    """Section 13: Q_D = vech(D)^T Sigma_D^+ vech(D).

    A continuous statement of how far the instantaneous field is from rigid,
    in units of its own uncertainty. Section 13 forbids gating support on a
    bare ||D|| threshold, so this returns a statistic and its degrees of
    freedom; what to do with it is the caller's decision.
    """
    D = torch.as_tensor(D).double()
    d = torch.stack([D[i, j] for i, j in VECH_INDEX])
    inv = information_inverse(torch.as_tensor(cov_D).double())
    q = float(d @ (inv["inverse"] @ d))
    return {"vech_D": d, "Q_D": q, "dof": int(inv["rank"]),
            "Q_D_per_dof": q / max(int(inv["rank"]), 1),
            "note": ("section 13: a continuous rigid-model consistency "
                     "statistic, not a gate")}


# ---------------------------------------------------------------------------
# section 14: differential pairwise rigidity
# ---------------------------------------------------------------------------
def differential_edge_rigidity(x, v, edges, *, velocity_covariance=None,
                               eps: float = 1e-12) -> dict:
    """Section 14: delta_ij = r^T (v_i - v_j) / ||r||^2, with its variance.

    Why this and not |v_i - v_j|: for a rigid body v_i - v_j = omega x r_ij,
    which is PERPENDICULAR to r_ij. Projecting onto r_ij therefore removes the
    rotation exactly and leaves only the part that changes the distance --

        d/dt ||r||^2 = 2 r^T (v_i - v_j) = 2 r^T D r,

    since r^T Omega r = 0 for any skew Omega. So delta is zero for a rigid
    body whether or not it is spinning, which is precisely what a rigidity
    test must measure.

    The raw value is returned alongside its variance and the normalized value;
    section 14 forbids thresholding the raw magnitude.
    """
    x = torch.as_tensor(x).double()
    v = torch.as_tensor(v).double()
    e = torch.as_tensor(edges).long().reshape(-1, 2)
    r = x[e[:, 0]] - x[e[:, 1]]
    dv = v[e[:, 0]] - v[e[:, 1]]
    r2 = (r * r).sum(1) + eps
    delta = (r * dv).sum(1) / r2

    if velocity_covariance is None:
        var = torch.full_like(delta, float("nan"))
    else:
        cov = torch.as_tensor(velocity_covariance).double()
        if cov.ndim == 1:
            cov = cov[:, None, None] * torch.eye(3, dtype=torch.float64,
                                                 device=x.device)
        # d(delta)/d(v_i) = r / ||r||^2 and d/d(v_j) = -r / ||r||^2; the two
        # endpoints are different Gaussians, so their errors add.
        g = r / r2[:, None]
        var = (torch.einsum("ei,eij,ej->e", g, cov[e[:, 0]], g)
               + torch.einsum("ei,eij,ej->e", g, cov[e[:, 1]], g))
    normalized = delta / var.clamp_min(1e-300).sqrt()
    return {"edges": e, "rho_diff": delta, "rho_diff_variance": var,
            "rho_diff_normalized": normalized,
            "separation": r.norm(dim=1),
            "note": ("section 14: r^T(v_i - v_j)/||r||^2; zero for a rigid "
                     "body because r^T Omega r = 0")}


def rigidity_likelihood_ratio(affine: dict, twist: dict) -> dict:
    """Section 13 + 35 A: is this velocity field rigid? A nested test.

    The rigid model v = v_c + omega x r is EXACTLY the affine model
    v = v_c + G r with its symmetric part held at zero, so rigid (6
    parameters) is nested inside affine (12) and the six extra parameters are
    precisely vech(D). That makes the comparison a textbook nested test rather
    than a threshold:

        sigma^2 = weighted residual of the AFFINE fit / (3N - 12)
        chi2    = (rigid residual - affine residual) / sigma^2   ~  chi2(6)

    The noise scale comes from the more general model, so it is not estimated
    from the very residual being tested -- which is what makes the statistic
    mean something. `Q_D` of section 13 answers the same question from the
    parameter side and is kept as the reported diagnostic; this is the form
    the decision can actually use, because its null distribution is known.

    A support holding two differently-moving bodies inflates the rigid
    residual while the affine model absorbs part of the difference, so this
    statistic rises sharply -- which is section 55's expected behaviour and
    section 32's reminder that the cause is mixed support, not material
    strain.
    """
    n = int(affine["num_points"])
    dof_affine = max(3 * n - 12, 1)
    sigma2 = affine["weighted_residual"] / dof_affine
    # floor at the arithmetic noise of the velocities themselves
    sigma2 = max(sigma2, twist.get("residual_sigma2", 0.0) * 0.0 + sigma2,
                 float(torch.finfo(torch.float64).eps) ** 2)
    excess = twist["rigid_fit_error"] - affine["weighted_residual"]
    chi2 = max(excess, 0.0) / max(sigma2, 1e-300)
    return {"chi2_rigid_vs_affine": chi2, "dof": 6,
            "sigma2_from_affine": sigma2,
            "affine_residual": affine["weighted_residual"],
            "rigid_residual": twist["rigid_fit_error"],
            "excess": excess,
            "note": ("section 35 A: rigid is nested in affine and the six "
                     "extra parameters are vech(D), so this is a chi2(6) with "
                     "a known null distribution")}
