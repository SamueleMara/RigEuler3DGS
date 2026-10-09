"""Sections 26, 39: do the differential and finite pictures agree?

CODEX_IMPLEMENT_DUAL_DIFFERENTIAL_FINITE_JOINT_DISCOVERY.md.

Two independent routes reach the same physical quantity. Phase-1 velocities
give an instantaneous angular velocity omega^diff through the rigid-twist fit;
the Phase-2 finite rotations give omega^finite by differentiating the rotation
sequence. A correct rigid model must satisfy

    omega^diff ~ omega^finite,

and section 26 makes the comparison mandatory because it is the only
instrument that says WHERE a spurious rotation enters. That is section 39's
question for Storage, stated without anything Storage-specific in it:

  * omega^diff ~ 0 while omega^finite != 0  -> the angular motion is not in
    the measurements; it was introduced downstream, by Phase 2 or the joint
    fit;
  * both nonzero and parallel                -> the rotation is genuinely
    present in the reconstructed motion;
  * both inconsistent with D large            -> the support or the motion
    model is unreliable, and neither number means anything yet.

Nothing here decides a joint type. It locates a disagreement.
"""

from __future__ import annotations

import math

import torch

from estimator.manifold import project_to_so3, so3_log, so3_vee

from .common_screw import chi2_quantile
from .joint_information import information_inverse


def incremental_spatial_omega(rotations, *, times=None, dt: float = 1.0,
                              convention: str = "spatial") -> dict:
    """Section 26: omega^finite_t = log(R_{t+1} R_t^T)^vee / dt.

    The SPATIAL increment is R_{t+1} R_t^T, composing on the left, matching
    `propagate_spatial_twist`. The body increment R_t^T R_{t+1} is a different
    vector -- the same rotation expressed in the moving frame -- and section 27
    forbids mixing them, so the convention is a parameter and both are
    computed only where a test asks for it.
    """
    R = project_to_so3(torch.as_tensor(rotations).double()).detach()
    T = int(R.shape[0])
    t = (torch.arange(T, dtype=torch.float64, device=R.device) * float(dt)
         if times is None else torch.as_tensor(times).double().reshape(-1))
    if T < 2:
        return {"omega": torch.zeros((0, 3), dtype=torch.float64),
                "times": t, "convention": convention}
    if convention == "spatial":
        dR = torch.einsum("tij,tkj->tik", R[1:], R[:-1])
    elif convention == "body":
        dR = torch.einsum("tji,tjk->tik", R[:-1], R[1:])
    else:
        raise ValueError("section 27: convention must be 'spatial' or 'body'")
    h = (t[1:] - t[:-1]).clamp_min(1e-300)
    omega = so3_log(dR) / h[:, None]
    # stamped at the interval midpoint, which is where a centred difference of
    # the same data would place it
    return {"omega": omega, "times": 0.5 * (t[1:] + t[:-1]),
            "increment": dR, "convention": convention,
            "note": ("section 26: spatial increment R_{t+1} R_t^T, "
                     "differentiated; stamped at interval midpoints")}


def compare_angular_velocity(omega_diff, omega_finite, *,
                             covariance_diff=None,
                             covariance_finite=None) -> dict:
    """Section 26: Euclidean AND covariance-normalised disagreement.

    Both are required. The Euclidean difference says how large the
    disagreement is in rad/s, which is what a physicist reads; the
    Mahalanobis distance says whether it is large COMPARED TO WHAT EITHER
    ESTIMATE CLAIMS TO KNOW, which is what decides whether it matters. A
    well-observed part disagreeing by 0.01 rad/s is a real problem; a barely
    observed one disagreeing by the same amount is not.
    """
    a = torch.as_tensor(omega_diff).double().reshape(-1, 3)
    b = torch.as_tensor(omega_finite).double().reshape(-1, 3)
    T = min(a.shape[0], b.shape[0])
    a, b = a[:T], b[:T]
    d = a - b
    diff_norm = d.norm(dim=1)

    na, nb = a.norm(dim=1), b.norm(dim=1)
    cos = (a * b).sum(1) / (na.clamp_min(1e-300) * nb.clamp_min(1e-300))
    cos = torch.where((na > 0) & (nb > 0), cos,
                      torch.full_like(cos, float("nan")))

    maha = torch.full((T,), float("nan"), dtype=torch.float64)
    if covariance_diff is not None or covariance_finite is not None:
        eye = torch.zeros((3, 3), dtype=torch.float64)
        ca = (torch.as_tensor(covariance_diff).double()[:T] if
              covariance_diff is not None else eye.expand(T, 3, 3))
        cb = (torch.as_tensor(covariance_finite).double()[:T] if
              covariance_finite is not None else eye.expand(T, 3, 3))
        tot = ca + cb
        for k in range(T):
            inv = information_inverse(tot[k])["inverse"]
            maha[k] = d[k] @ (inv @ d[k])

    return {"omega_diff": a, "omega_finite": b, "difference": d,
            "difference_norm": diff_norm,
            "mahalanobis_difference": maha,
            "axis_cosine": cos,
            "median_difference_norm": float(diff_norm.median()) if T else 0.0,
            "median_axis_cosine": float(cos[~cos.isnan()].median())
            if T and bool((~cos.isnan()).any()) else float("nan"),
            "median_mahalanobis": float(maha[~maha.isnan()].median())
            if T and bool((~maha.isnan()).any()) else float("nan")}


def attribute_angular_origin(comparison: dict, *, strain_statistic=None,
                             omega_information=None,
                             confidence: float = 0.95) -> dict:
    """Section 39: say WHERE the angular motion is, without naming a scene.

    Every cut here is a chi-square quantile of the statistic's own null
    distribution at one `confidence` shared by the whole run -- section 6's
    H_P: omega = 0 for "is there any rotation", chi2(6) for "is the support
    rigid", chi2(3) for "do the two routes agree". Section 68 forbids the
    alternative, and in particular forbids deciding agreement with a cut on
    the axis cosine: a cosine of 0.9 means something quite different for a
    precisely-measured axis than for a barely-observed one. The cosine is
    reported as a diagnostic and, where no covariance exists to normalise the
    disagreement, the attribution says so instead of guessing.

    This returns a LABEL for a disagreement, not a joint type.
    """
    a = comparison["omega_diff"]
    b = comparison["omega_finite"]
    T = a.shape[0]
    zero_cut = chi2_quantile(3, confidence)
    if omega_information is not None:
        F = torch.as_tensor(omega_information).double()[:T]
        chi_diff = torch.einsum("ti,tij,tj->t", a, F, a)
    else:
        chi_diff = torch.full((T,), float("nan"), dtype=torch.float64)

    diff_present = (bool((chi_diff > zero_cut).float().mean() > 0.5)
                    if not bool(chi_diff.isnan().all()) else None)
    fin_scale = float(b.norm(dim=1).median()) if T else 0.0
    dif_scale = float(a.norm(dim=1).median()) if T else 0.0
    cosm = comparison["median_axis_cosine"]
    maha = comparison.get("median_mahalanobis", float("nan"))
    # The two estimates agree when their difference is consistent with the
    # uncertainty of the estimates themselves -- a chi2(3) on the difference
    # vector -- not when a cosine clears a constant.
    agree_cut = chi2_quantile(3, confidence)
    if not math.isnan(maha):
        aligned = maha <= agree_cut
        agreement_basis = "mahalanobis difference against chi2(3)"
    else:
        # No covariance was supplied for either estimate. Rather than fall
        # back to a cosine threshold, record that agreement is untestable.
        aligned = None
        agreement_basis = ("no covariance supplied; agreement between the two "
                           "angular estimates could not be tested")

    strain_large = None
    if strain_statistic is not None:
        st = torch.as_tensor(strain_statistic).double().reshape(-1)
        # Q_D has six degrees of freedom; a median well above its quantile
        # means the one-rigid-body model does not describe this support.
        strain_large = bool(float(st.median()) > chi2_quantile(6, confidence))

    # Scale below which an angular velocity is numerically indistinguishable
    # from zero whatever the statistics say. Both routes being exactly zero is
    # the prismatic case, and must not be reported as a disagreement about an
    # axis that neither of them has.
    tiny = 1e-12 * max(1.0, dif_scale, fin_scale)
    both_absent = dif_scale <= tiny and fin_scale <= tiny

    if strain_large:
        verdict = "SUPPORT_UNRELIABLE"
        why = ("section 39: the differential strain statistic is large, so "
               "neither angular estimate describes one rigid body")
    elif both_absent:
        verdict = "ANGULAR_ABSENT"
        why = ("section 39: neither route finds any angular motion at all, "
               "which is the prismatic case and not a disagreement")
    elif diff_present is None:
        verdict = "ANGULAR_UNDETERMINED"
        why = ("section 6: no calibrated angular information was supplied, so "
               "whether omega differs from zero cannot be tested")
    elif diff_present is False and fin_scale > 4.0 * max(dif_scale, tiny):
        verdict = "ANGULAR_INTRODUCED_DOWNSTREAM"
        why = ("section 39: Phase-1 velocities carry no angular component "
               "distinguishable from zero, yet the finite rotations do -- the "
               "rotation enters after the measurements")
    elif diff_present is False:
        verdict = "ANGULAR_ABSENT"
        why = ("section 39: no angular component distinguishable from zero in "
               "either route")
    elif aligned is None:
        verdict = "ANGULAR_UNDETERMINED"
        why = ("section 68: the two angular estimates carry no covariance, so "
               "whether they agree cannot be tested without a cosine "
               "threshold, which is forbidden")
    elif aligned:
        verdict = "ANGULAR_GENUINELY_PRESENT"
        why = ("section 39: both estimates are nonzero and share an axis, so "
               "the rotation is in the reconstructed motion itself")
    else:
        verdict = "ANGULAR_INCONSISTENT"
        why = ("section 26: the two estimates do not share an axis; the "
               "motion model is not yet trustworthy")
    return {"attribution": verdict, "reason": why,
            "median_omega_diff": dif_scale,
            "median_omega_finite": fin_scale,
            "median_axis_cosine": cosm,
            "differential_angular_significant": diff_present,
            "chi2_omega_diff": chi_diff,
            "axes_agree": aligned, "agreement_basis": agreement_basis,
            "agreement_cut": agree_cut, "omega_zero_cut": zero_cut,
            "confidence": confidence}


def compare_screw_trajectory(q_diff, q_finite) -> dict:
    """Section 26: the two routes must also agree on the joint coordinate.

    Compared after removing the q(t_0) = 0 gauge from both, because an offset
    is not a disagreement about the motion.
    """
    a = torch.as_tensor(q_diff).double().reshape(-1)
    b = torch.as_tensor(q_finite).double().reshape(-1)
    T = min(a.numel(), b.numel())
    a, b = a[:T] - (a[0] if T else 0.0), b[:T] - (b[0] if T else 0.0)
    d = a - b
    rng = float((b.max() - b.min())) if T else 0.0
    return {"q_diff": a, "q_finite": b, "difference": d,
            "max_difference": float(d.abs().max()) if T else 0.0,
            "rms_difference": float((d * d).mean().sqrt()) if T else 0.0,
            "finite_range": rng,
            "relative_difference": (float(d.abs().max()) / rng
                                    if rng > 0 else float("nan"))}
