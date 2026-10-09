"""Section 2 + 9.2: the coordinate-invariant Gaussian transport metric.

CODEX_DIRECT_GAUSSIAN_JOINT_RGBD_RELIABILITY.md sections 2 and 9.2.

Section 2.1 found a real defect in the previous objective. It scored a
candidate by (p_obs - p)/L, where p is the translation of x -> R x + p about
the WORLD ORIGIN. That is not a physical quantity: shifting the world origin
by g sends p to p + g - R g, so whenever two candidates differ in rotation --
which is exactly the prismatic-versus-revolute comparison -- the residual
between them depends on where the origin happens to sit. Measured on two
candidates 0.3 and 0.22 rad apart, it grew from 0.054 to 10.1 as the origin
moved 113 units away, so the same scene could have been classified either way
depending on the world frame it was reconstructed in.

What replaces it is the motion the candidate actually induces on the part's
own Gaussians, which no choice of origin can change.
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# section 2: the coordinate-invariant Gaussian-transport objective
# ---------------------------------------------------------------------------
def canonical_moments(mu0: torch.Tensor, weights: torch.Tensor | None = None):
    """Section 2.2: the part's centroid c, second moment S, and L^2 = tr S.

    These are fixed properties of the CANONICAL support: they do not depend on
    any candidate transform, so neither P nor R can shift them in its favour.
    """
    mu0 = mu0.double()
    w = (torch.ones(mu0.shape[0], dtype=torch.float64, device=mu0.device)
         if weights is None else weights.double().clamp_min(0.0))
    W = w.sum().clamp_min(1e-300)
    c = (w.unsqueeze(1) * mu0).sum(0) / W
    dx = mu0 - c
    S = (w.unsqueeze(1) * dx).T @ dx / W
    L2 = float(S.diagonal().sum())
    return {"c": c, "S": S, "L2": L2, "L": math.sqrt(max(L2, 0.0)), "W": float(W)}


def psd_sqrt(S: torch.Tensor) -> torch.Tensor:
    """The symmetric PSD square root of S, via its eigendecomposition.

    S is a weighted second moment, so it is symmetric PSD by construction; a
    thin part has a near-zero eigenvalue, hence the clamp before the sqrt.
    """
    evals, evecs = torch.linalg.eigh(S.double())
    return evecs @ torch.diag(evals.clamp_min(0.0).sqrt()) @ evecs.T


def gaussian_transport_residual(R_obs, p_obs, R, p, c, S, L, *,
                                include_frame_zero: bool = False,
                                S_half: torch.Tensor | None = None):
    """Section 2.3's E_G, as a least-squares RESIDUAL VECTOR.

        E_G = 1/((T-1) W L^2) sum_t sum_i w_i || That_t mu_i - T_t mu_i ||^2
            = 1/((T-1) L^2)   sum_t [ ||Delta d_c,t||^2
                                      + tr(Delta R_t S Delta R_t^T) ]

    The second line is EXACT, not an approximation: writing mu_i = c + (mu_i-c),
    the cross term carries sum_i w_i (mu_i - c) = 0 by the definition of c, so
    it vanishes identically. That is what lets Phase 3 avoid touching every
    Gaussian on every step while still measuring motion of the real geometry.

    Why this replaces (p_obs - p)/L: the world-origin translation p is NOT a
    physical quantity. Under a global shift g it becomes p + g - R g, so a
    residual built from it depends on where the origin happens to sit whenever
    the two candidates' rotations differ. Measured on two candidates 0.3 and
    0.22 rad apart, that residual grew from 0.054 to 8.95 as the origin moved
    113 units -- it would have ranked P against R differently in different
    world frames. The centre displacement d_c,t = R c + p - c is invariant.

    tr(Delta R S Delta R^T) = ||Delta R S^{1/2}||_F^2, so the rotation part is
    returned as Delta R S^{1/2} flattened, keeping the whole thing an exact
    sum of squares for Levenberg-Marquardt.
    """
    sl = slice(0, None) if include_frame_zero else slice(1, None)
    Ro, po, Rm, pm = R_obs[sl].double(), p_obs[sl].double(), R[sl].double(), p[sl].double()
    T = int(Ro.shape[0])
    if T == 0:
        return torch.zeros(0, dtype=torch.float64, device=c.device)
    c = c.reshape(3).double()
    # section 2.2: displacement of the canonical centre, origin-invariant
    d_obs = torch.einsum("tij,j->ti", Ro, c) + po - c
    d_mod = torch.einsum("tij,j->ti", Rm, c) + pm - c
    dd = d_obs - d_mod

    # S is constant across the whole fit, so its root is computed once by the
    # caller and passed in; eigh inside the LM residual would otherwise be
    # redone on every Jacobian column.
    if S_half is None:
        S_half = psd_sqrt(S)
    dR = (Ro - Rm) @ S_half                                   # [T,3,3]

    scale = 1.0 / (max(float(L), 1e-300) * math.sqrt(max(T, 1)))
    return torch.cat([(dd * scale).reshape(-1), (dR * scale).reshape(-1)])


def gaussian_transport_error_pointwise(mu0, weights, R_obs, p_obs, R, p, L, *,
                                       include_frame_zero: bool = False):
    """The literal O(T N) point sum of section 2.3, for the equivalence test."""
    sl = slice(0, None) if include_frame_zero else slice(1, None)
    Ro, po, Rm, pm = R_obs[sl].double(), p_obs[sl].double(), R[sl].double(), p[sl].double()
    T = int(Ro.shape[0])
    if T == 0:
        return 0.0
    mu0 = mu0.double()
    w = (torch.ones(mu0.shape[0], dtype=torch.float64, device=mu0.device)
         if weights is None else weights.double().clamp_min(0.0))
    W = float(w.sum())
    a = torch.einsum("tij,nj->tni", Ro, mu0) + po.unsqueeze(1)
    b = torch.einsum("tij,nj->tni", Rm, mu0) + pm.unsqueeze(1)
    sq = ((a - b) ** 2).sum(-1)                               # [T,N]
    return float((w.unsqueeze(0) * sq).sum() / (T * W * max(L, 1e-300) ** 2))


def transport_audit(R_obs, p_obs, R, p, moments, *,
                    include_frame_zero: bool = False) -> dict:
    """Section 2.5: every term of the fit, separated and recorded.

    The point of splitting them is that a wrong P/R verdict has to come from
    somewhere, and these say where. `E_center_translation` is how far the
    part's centre ends up from where it should; `E_center_rotation_moment` is
    how much of the error is orientation weighted by the part's actual extent;
    `delta_rotation_geodesic` is the same orientation error in radians, with
    no shape weighting, so comparing the two shows whether a part is being
    judged mostly on its length or mostly on its turn.

    `E_legacy_origin_translation` is the superseded section-10 quantity. It is
    recorded so that a change of verdict can be traced to the change of
    objective, and for no other purpose -- section 2.1 shows it is not
    invariant to the world origin, so nothing may branch on it.
    """
    sl = slice(0, None) if include_frame_zero else slice(1, None)
    Ro, po = R_obs[sl].double(), p_obs[sl].double()
    Rm, pm = R[sl].double(), p[sl].double()
    T = int(Ro.shape[0])
    c = moments["c"].reshape(3).double()
    S = moments["S"].double()
    L = float(moments["L"])
    denom = max(L, 1e-300) ** 2 * max(T, 1)

    d_obs = torch.einsum("tij,j->ti", Ro, c) + po - c
    d_mod = torch.einsum("tij,j->ti", Rm, c) + pm - c
    e_trans = float(((d_obs - d_mod) ** 2).sum()) / denom

    dR = Ro - Rm
    e_rot = float(torch.einsum("tij,jk,tik->", dR, S, dR)) / denom

    # imported here, not at module scope: the chart module imports this one
    # for its objective, so a top-level import would close the cycle.
    from .continuous_joint_geometry import _so3_log_safe
    rel = torch.einsum("tji,tjk->tik", Rm, Ro)
    geo = _so3_log_safe(rel).norm(dim=1)

    legacy = float((((po - pm) / max(L, 1e-300)) ** 2).sum()
                   + (geo ** 2).sum()) / max(T, 1)

    evals = torch.linalg.eigvalsh(S).tolist()
    return {
        "E_center_translation": e_trans,
        "E_center_rotation_moment": e_rot,
        "E_gaussian_transport_total": e_trans + e_rot,
        "delta_rotation_geodesic_rad": geo.tolist(),
        "delta_rotation_geodesic_max_rad": float(geo.max()) if T else 0.0,
        "delta_rotation_geodesic_rms_rad": (float((geo ** 2).mean().sqrt())
                                            if T else 0.0),
        "E_legacy_origin_translation": legacy,
        "centroid_c": c.tolist(),
        "second_moment_S": S.tolist(),
        "second_moment_eigenvalues": evals,
        "part_radius_L": L,
        "frames_scored": T,
    }


# ---------------------------------------------------------------------------
# section 9.2: where does a revolute preference actually come from?
# ---------------------------------------------------------------------------
def sl_of(include_frame_zero: bool):
    """Frame 0 is the gauge (T_0 = I on both sides), so it is normally cut."""
    return slice(0, None) if include_frame_zero else slice(1, None)


def gain_attribution(R_obs, p_obs, fit_P, fit_R, moments, *,
                     include_frame_zero: bool = False) -> dict:
    """Split E_P - E_R into its centre-trajectory and rotation parts.

    Section 9.2: because E_G = E_center_translation + E_center_rotation_moment,
    the gain that makes a part look revolute splits the same way:

        Delta E_total = Delta E_center + Delta E_rotation.

    The two mean very different things. `Delta E_center` is curvature of the
    path the part's centre actually travels -- a bent trajectory is what a
    revolute joint IS. `Delta E_rotation` is only the part's orientation
    changing, which a free rigid fit can produce from noise at a magnitude far
    below anything visible.

    So a revolute preference built almost entirely out of `Delta E_rotation`,
    with `Delta E_center` near zero or NEGATIVE (the straight-line model
    fitting the centre path better), is not evidence of curvature. It is a
    tiny unstable rotation, and this function labels it as such rather than
    letting it pass as a joint type.

    Both P and R preserve pairwise distances and inertia, so those invariants
    cannot make this distinction (section 9.4); this decomposition can.
    """
    aP = transport_audit(R_obs, p_obs, fit_P["R"], fit_P["p"], moments,
                         include_frame_zero=include_frame_zero)
    aR = transport_audit(R_obs, p_obs, fit_R["R"], fit_R["p"], moments,
                         include_frame_zero=include_frame_zero)
    d_center = aP["E_center_translation"] - aR["E_center_translation"]
    d_rot = aP["E_center_rotation_moment"] - aR["E_center_rotation_moment"]
    d_total = d_center + d_rot
    denom = abs(d_center) + abs(d_rot)
    frac_rot = (abs(d_rot) / denom) if denom > 0 else 0.0

    # A gain is "rotation-only" when the straight-line model explains the
    # centre path at LEAST AS WELL as the curved one does. Then no curvature
    # was measured at all: only orientation wobble.
    #
    # Note that `frac_rot` alone does NOT distinguish the two cases, and must
    # not be used as if it did. On a genuine revolute joint the rotation term
    # legitimately dominates too -- measured here, USB (true revolute) puts
    # 97.7% of its gain in rotation and Storage (true prismatic) puts 88.8%,
    # so a threshold on the fraction would have ranked them the wrong way
    # round. What separates them is the SIGN of `delta_E_center`: USB's centre
    # path is fit 46x better by the curved model (rms 8.4e-02 -> 1.8e-03),
    # while Storage's is fit better by the straight one. The fraction is
    # reported for the record, not as a selector.
    rotation_only = bool(d_total > 0.0 and d_center <= 0.0)

    # Scale context: how big the residual rotations are next to the rotation
    # actually present in the observations. A prismatic part that was fitted a
    # spurious axis shows a residual far below the observed amplitude.
    from .continuous_joint_geometry import _so3_log_safe
    obs_rot = _so3_log_safe(R_obs[sl_of(include_frame_zero)].double()).norm(dim=1)
    obs_amp = float(obs_rot.max()) if obs_rot.numel() else 0.0
    obs_disp = float((torch.einsum("tij,j->ti", R_obs[sl_of(include_frame_zero)].double(),
                                   moments["c"].reshape(3).double())
                      + p_obs[sl_of(include_frame_zero)].double()
                      - moments["c"].reshape(3).double()).norm(dim=1).max()
                     ) if R_obs.shape[0] else 0.0
    return {
        "observed_rotation_amplitude_rad": obs_amp,
        "observed_center_displacement_max": obs_disp,
        "part_radius_L": float(moments["L"]),
        "audit_P": aP, "audit_R": aR,
        "delta_E_center": d_center,
        "delta_E_rotation": d_rot,
        "delta_E_total": d_total,
        "rotation_fraction_of_gain": frac_rot,
        "gain_is_rotation_only": rotation_only,
        "max_residual_rotation_P_rad": aP["delta_rotation_geodesic_max_rad"],
        "max_residual_rotation_R_rad": aR["delta_rotation_geodesic_max_rad"],
        "center_path_rms_P": math.sqrt(max(aP["E_center_translation"], 0.0)),
        "center_path_rms_R": math.sqrt(max(aR["E_center_translation"], 0.0)),
    }
