"""Sections 16-19, 27-29: the exact rigid twist, and propagating with it.

CODEX_IMPLEMENT_DUAL_DIFFERENTIAL_FINITE_JOINT_DISCOVERY.md.

The affine fit of section 10 is a diagnostic: it asks what the velocity field
IS. This module asks a different question -- what is the best RIGID velocity
field for this support -- by fitting

    v_i = v_c + omega x r_i

directly, which is the exact rigid subspace rather than a projection taken
after the fact. Section 28 is the reason the distinction matters: the free
per-Gaussian field is a MEASUREMENT, while the projected rigid twist is the
PROCESS MODEL, and propagating the former would let per-Gaussian noise leak
into the part's trajectory.
"""

from __future__ import annotations

import math

import torch

from estimator.manifold import so3_exp, so3_hat, so3_left_jacobian, so3_vee

from .joint_information import information_inverse

SPATIAL = "spatial"
BODY = "body"


# ---------------------------------------------------------------------------
# section 16: the exact rigid twist by weighted least squares
# ---------------------------------------------------------------------------
def fit_rigid_twist(x, v, *, information=None, weights=None,
                    centre=None, dt: float = 1.0) -> dict:
    """Section 16: v_i = v_c + omega x r_i, solved for xi_c = [v_c; omega].

    Using omega x r = -[r]_x omega makes the model linear in the unknown, so
    this is one weighted least-squares solve with no iteration and no
    initialisation. The design block is

        A_i = [ I3 | -[r_i]_x ].

    The fit is centred on the support because the two halves of xi are then
    nearly uncorrelated; referred to a distant world origin the translation
    and rotation blocks mix, and `H_omega_given_v` below -- which is the whole
    point of section 17 -- would be dominated by that choice of origin rather
    than by the data.
    """
    x = torch.as_tensor(x).double()
    v = torch.as_tensor(v).double()
    n = x.shape[0]
    if weights is None:
        w = torch.ones(n, dtype=torch.float64, device=x.device)
    else:
        w = torch.as_tensor(weights).double().reshape(-1).clamp_min(0.0)
    eye = torch.eye(3, dtype=torch.float64, device=x.device)
    lam = (w[:, None, None] * eye if information is None
           else torch.as_tensor(information).double() * w[:, None, None])

    c = ((w[:, None] * x).sum(0) / w.sum().clamp_min(1e-300)
         if centre is None else torch.as_tensor(centre).double().reshape(3))
    r = x - c
    A = torch.cat([eye.expand(n, 3, 3), -so3_hat(r)], dim=2)      # [N,3,6]
    H = torch.einsum("nij,nik,nkl->jl", A, lam, A)
    b = torch.einsum("nij,nik,nk->j", A, lam, v)
    inv = information_inverse(H)
    xi_c = inv["inverse"] @ b
    pred = torch.einsum("nij,j->ni", A, xi_c)
    resid = v - pred
    wres = float(torch.einsum("ni,nij,nj->", resid, lam, resid))

    # The scale of the measurement noise, estimated from the fit itself.
    #
    # Without it the information matrix is only defined up to an arbitrary
    # factor, and every hypothesis test built on it is meaningless: passing
    # Lambda = I asserts that each velocity is known to +-1 world unit, so a
    # part genuinely turning at 0.06 rad/frame is reported as indistinguishable
    # from stationary -- not because it is, but because it was told the
    # measurements are twenty times noisier than the signal. Section 6 requires
    # the test to be made against real uncertainty, so when the caller supplies
    # none the residual of this very fit provides it:
    #
    #     sigma^2 = weighted residual / (3N - 6).
    #
    # The floor keeps a numerically exact fit from producing an infinite
    # information rather than merely a very large one.
    dof = max(3 * n - 6, 1)
    # The floor is the ARITHMETIC noise of the velocity estimate, which is set
    # by the positions it was differenced from, not by the velocity's own
    # magnitude: v = (x(t+h) - x(t-h)) / 2h carries absolute rounding of about
    # eps |x| / dt whatever v happens to be. Using eps |v| instead would make
    # a slow, perfectly-tracked part look infinitely precise, so that its own
    # last-bit rounding registered as a significant rotation.
    xscale = float(x.abs().max()) if x.numel() else 1.0
    floor = (float(torch.finfo(torch.float64).eps) * max(xscale, 1e-300)
             / max(float(dt), 1e-300)) ** 2
    sigma2 = max(wres / dof, floor)
    scaled = H / sigma2 if information is None else H

    return {"centre": c, "v_c": xi_c[:3], "omega": xi_c[3:], "xi_c": xi_c,
            "information": H, "covariance": inv["inverse"],
            "information_scaled": scaled,
            "residual_sigma2": sigma2, "residual_dof": dof,
            "information_is_calibrated": bool(information is not None),
            "rank": inv["rank"], "condition": inv["condition"],
            "spectrum": inv["spectrum"], "null_basis": inv["null_basis"],
            "residual": resid,
            "rigid_fit_error": wres,
            "num_points": int(n)}


def twist_covariance(fit: dict) -> dict:
    """Sections 16, 17: the twist covariance and the angular block that
    survives after translation is eliminated as a nuisance.

    Partitioning

        H = [[H_vv, H_vw], [H_wv, H_ww]],

    the angular information left once translation is free to absorb whatever
    it can is the Schur complement

        H_{w|v} = H_ww - H_wv H_vv^+ H_vw.

    This is the quantity that decides whether rotation is OBSERVABLE at all,
    which section 6 insists must replace a threshold on ||omega||: a support
    sliding past the cameras can produce a nonzero omega that carries no
    information once translation is accounted for, and H_{w|v} says so.
    """
    # Built on the CALIBRATED information: with no external covariance this is
    # the residual-scaled version, so chi2_omega_zero is a real statistic
    # rather than a number in arbitrary units.
    H = fit.get("information_scaled", fit["information"])
    Hvv, Hvw = H[:3, :3], H[:3, 3:]
    Hwv, Hww = H[3:, :3], H[3:, 3:]
    inv_vv = information_inverse(Hvv)
    H_w_given_v = Hww - Hwv @ inv_vv["inverse"] @ Hvw
    schur = information_inverse(H_w_given_v)
    omega = fit["omega"]
    # Mahalanobis distance of omega from the null hypothesis omega = 0
    # (section 6's H_P), in the metric that already discounts translation.
    chi2 = float(omega @ (H_w_given_v @ omega))
    return {"omega_information": H_w_given_v,
            "omega_covariance": fit["covariance"][3:, 3:],
            "omega_rank": schur["rank"],
            "omega_condition": schur["condition"],
            "omega_spectrum": schur["spectrum"],
            "chi2_omega_zero": chi2,
            "dof": int(schur["rank"]),
            "null_hypothesis": "section 6: H_P is omega = 0",
            "note": ("section 17: angular information after eliminating the "
                     "translation nuisance, H_ww - H_wv H_vv^+ H_vw")}


def center_twist_to_world_twist(v_c, omega, centre) -> torch.Tensor:
    """Section 19: v_0 = v_c - omega x c, giving the spatial twist [v_0; omega].

    A twist referred to the support centre and the same twist referred to the
    world origin are the same motion written two ways; mixing them is a silent
    error, so the conversion is explicit and has its own test.
    """
    v_c = torch.as_tensor(v_c).double().reshape(3)
    omega = torch.as_tensor(omega).double().reshape(3)
    c = torch.as_tensor(centre).double().reshape(3)
    return torch.cat([v_c - torch.linalg.cross(omega, c), omega])


def se3_exp(xi, dt: float = 1.0) -> torch.Tensor:
    """exp(dt * xi^) as a 4x4, with xi = [v; omega] in the spatial convention.

    The translation block carries the SO(3) left Jacobian, not a bare dt * v:
    over a finite step the point is also being carried around by the rotation,
    and dropping J turns a rotation about a distant axis into a rotation plus a
    spurious drift.
    """
    xi = torch.as_tensor(xi).double().reshape(6)
    phi = xi[3:] * float(dt)
    u = xi[:3] * float(dt)
    T = torch.eye(4, dtype=torch.float64, device=xi.device)
    T[:3, :3] = so3_exp(phi.unsqueeze(0))[0].double()
    T[:3, 3] = so3_left_jacobian(phi.unsqueeze(0))[0].double() @ u
    return T


def propagate_spatial_twist(T, xi, dt: float = 1.0,
                            convention: str = SPATIAL) -> torch.Tensor:
    """Sections 27, 28: compose one step of the rigid process model.

    Spatial (world) twist composes on the LEFT:   T_{t+1} = exp(dt xi^) T_t
    Body twist composes on the RIGHT:             T_{t+1} = T_t exp(dt xi^)

    Section 27 says to prefer the spatial convention here because Gaussian
    positions are stored in world coordinates, and never to mix the two. Both
    are implemented so the composition test of section 58 can check them
    against each other; production passes `convention=SPATIAL`.
    """
    T = torch.as_tensor(T).double()
    step = se3_exp(xi, dt)
    if convention == SPATIAL:
        return step @ T
    if convention == BODY:
        return T @ step
    raise ValueError(f"section 27: convention must be '{SPATIAL}' or "
                     f"'{BODY}', not {convention!r}")


def propagate_gaussians(mu, xi, dt: float = 1.0, *, rotations=None,
                        covariances=None, convention: str = SPATIAL) -> dict:
    """Sections 28, 29: carry centres, orientations and covariances together.

    A Gaussian is not a point. Moving only its centre and leaving its
    orientation behind makes an anisotropic splat point the wrong way after a
    rotation, which the renderer then sees as a photometric error that no
    joint can explain. Section 29 requires all three to move with the same
    increment:

        mu    -> R mu + t
        R^G   -> R R^G
        Sigma -> R Sigma R^T
    """
    step = se3_exp(xi, dt)
    R, t = step[:3, :3], step[:3, 3]
    if convention == BODY:
        raise ValueError("section 29: Gaussian propagation is defined in the "
                         "spatial convention; a body twist must be converted "
                         "first")
    mu = torch.as_tensor(mu).double()
    out = {"increment": step, "mu": mu @ R.T + t}
    if rotations is not None:
        out["rotations"] = R @ torch.as_tensor(rotations).double()
    if covariances is not None:
        S = torch.as_tensor(covariances).double()
        out["covariances"] = R @ S @ R.T
    return out


def twist_sequence(tracks, *, times=None, dt: float = 1.0,
                   information=None, weights=None, velocity=None) -> dict:
    """Sections 16-19 over a whole window: xi_t for every frame.

    Returns both the centre-referenced and the spatial twist, because section
    19's rank-1 common-screw test is stated for the spatial twist while the
    per-frame fit is conditioned at the centre.
    """
    from .velocity_gradient_rigidity import instantaneous_velocity

    tracks = torch.as_tensor(tracks).double()
    T = tracks.shape[0]
    vel = instantaneous_velocity(tracks, times=times, dt=dt, velocity=velocity)
    V = vel["velocity"]
    fits, xi_c, xi_s, omegas, infos = [], [], [], [], []
    for t in range(T):
        info_t = None if information is None else information[t]
        f = fit_rigid_twist(tracks[t], V[t], information=info_t,
                            weights=weights, dt=float(dt))
        f.update(twist_covariance(f))
        fits.append(f)
        xi_c.append(f["xi_c"])
        xi_s.append(center_twist_to_world_twist(f["v_c"], f["omega"],
                                                f["centre"]))
        omegas.append(f["omega"])
        infos.append(f["information"])
    return {"fits": fits,
            "xi_centre": torch.stack(xi_c),
            "xi_spatial": torch.stack(xi_s),
            "omega": torch.stack(omegas),
            "information": torch.stack(infos),
            "velocity": V,
            "velocity_source": vel["velocity_source"],
            "times": vel["times"]}
