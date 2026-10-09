"""Sections 22-25: the finite half of the dual theory, from centred tracks.

CODEX_IMPLEMENT_DUAL_DIFFERENTIAL_FINITE_JOINT_DISCOVERY.md.

Section 4 gives the finite P/R signature in one line. For a rigid part,

    r_ij(t) = x_i(t) - x_j(t) = R(t) r_ij(0),

so translation cancels EXACTLY -- no fitting, no nuisance parameter, no
dependence on where the world origin sits. A prismatic joint leaves the whole
relative vector unchanged; a revolute joint preserves its length but turns its
direction. The existing finite rigidity graph only ever looked at ||r||, which
is blind to that turn; this module looks at the direction too.

Subtracting the weighted centre is the same cancellation applied to all points
at once, and is what makes the Kabsch rotation below a statement about the
part's configuration rather than about its position.
"""

from __future__ import annotations

import math

import torch

from estimator.manifold import so3_exp, so3_hat, so3_log, so3_vee

from .joint_information import information_inverse


def weighted_center(x, weights=None) -> torch.Tensor:
    """Section 22: c_t = sum_i w_i x_i(t) / sum_i w_i."""
    x = torch.as_tensor(x).double()
    if weights is None:
        return x.mean(-2)
    w = torch.as_tensor(weights).double().reshape(-1).clamp_min(0.0)
    return (w[:, None] * x).sum(-2) / w.sum().clamp_min(1e-300)


def centered_tracks(tracks, weights=None) -> dict:
    """Section 22: y_i(t) = x_i(t) - c_t, with translation removed exactly."""
    tracks = torch.as_tensor(tracks).double()
    c = torch.stack([weighted_center(tracks[t], weights)
                     for t in range(tracks.shape[0])])
    return {"centres": c, "centered": tracks - c[:, None, :]}


def weighted_kabsch_rotation(y0, yt, weights=None) -> dict:
    """Section 23: R_t = argmin_R sum_i w_i ||y_i(t) - R y_i(0)||^2.

    The SVD solution with an explicit reflection correction: a least-squares
    3x3 fit to noisy points can land on an improper matrix, and silently
    accepting det = -1 turns a mirror into a "rotation" of about pi.
    """
    y0 = torch.as_tensor(y0).double()
    yt = torch.as_tensor(yt).double()
    n = y0.shape[0]
    w = (torch.ones(n, dtype=torch.float64, device=y0.device) if weights is None
         else torch.as_tensor(weights).double().reshape(-1).clamp_min(0.0))
    H = torch.einsum("n,ni,nj->ij", w, yt, y0)
    U, S, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.det(U @ Vh))
    flip = torch.diag(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64,
                                   device=y0.device))
    flip[2, 2] = d
    R = U @ flip @ Vh
    resid = yt - y0 @ R.T
    return {"R": R, "cross_covariance_singular_values": S,
            "reflection_corrected": bool(float(d) < 0),
            "residual": resid,
            "kabsch_error": float((w[:, None] * resid * resid).sum()),
            "weight_total": float(w.sum())}


def rotation_observability(y, weights=None, information=None) -> dict:
    """Section 23: can this support's shape determine a rotation at all?

    Perturbing R by exp(dphi) moves the model point z = R y_0 to
    z - [z]_x dphi, so the Fisher information for the rotation is

        M = sum_i [z_i]_x^T Lambda_i [z_i]_x,

    which for isotropic weights is tr(S) I - S with S the weighted shape
    covariance. Its SMALLEST eigenvalue names the least observable axis:

      * a collinear support gives S = diag(0, 0, s), hence M = diag(s, s, 0) --
        rotation about its own axis is invisible, exactly as it should be;
      * a planar support still determines all three, because a plane is not
        symmetric under any nontrivial rotation.

    Reporting this is what lets section 25's prohibition stand: a small angle
    on a support that could not have measured rotation means nothing, and a
    small angle on a well-conditioned support means something.
    """
    y = torch.as_tensor(y).double()
    n = y.shape[0]
    w = (torch.ones(n, dtype=torch.float64, device=y.device) if weights is None
         else torch.as_tensor(weights).double().reshape(-1).clamp_min(0.0))
    Z = so3_hat(y)                                           # [N,3,3]
    if information is None:
        eye = torch.eye(3, dtype=torch.float64, device=y.device)
        lam = w[:, None, None] * eye
    else:
        lam = torch.as_tensor(information).double() * w[:, None, None]
    M = torch.einsum("nji,njk,nkl->il", Z, lam, Z)
    inv = information_inverse(M)
    shape = torch.einsum("n,ni,nj->ij", w, y, y)
    sv = torch.linalg.eigvalsh(shape)
    return {"rotation_information": M, "rotation_covariance": inv["inverse"],
            "spectrum": inv["spectrum"], "rank": inv["rank"],
            "condition": inv["condition"],
            "least_observable_axis": inv["spectrum"][0].item()
            if inv["spectrum"].numel() else 0.0,
            "shape_covariance": shape, "shape_singular_values": sv,
            "shape_rank": int((sv > sv.max().clamp_min(1e-300)
                               * len(sv) * torch.finfo(torch.float64).eps).sum()),
            "note": ("section 23: Fisher information of a rotation "
                     "perturbation; zero eigenvalue means that axis is "
                     "unobservable from this support's shape")}


def rotation_log_trajectory(rotations) -> dict:
    """Section 24: phi_t = log(R_t)^vee, the quantity that is rank 0 or rank 1.

    For an ideal prismatic joint phi_t = 0 for every t; for an ideal one-DoF
    revolute joint phi_t = q_t e with ONE fixed axis e. So the question is not
    how big phi is -- section 25 forbids `if max_angle < 0.02: prismatic` --
    but whether phi is distinguishable from zero and, if it is, whether every
    frame points along the same axis.
    """
    R = torch.as_tensor(rotations).double()
    phi = so3_log(R)
    angle = phi.norm(dim=-1)
    return {"phi": phi, "angle": angle,
            "max_angle": float(angle.max()) if angle.numel() else 0.0,
            "note": ("section 24: rank 0 means prismatic geometry, rank 1 "
                     "means one-DoF revolute geometry; decided by covariance, "
                     "never by an angle threshold (section 25)")}


def finite_rotation_sequence(tracks, *, weights=None, information=None) -> dict:
    """Sections 22-24 over a whole window: c_t, R_t, phi_t and observability."""
    tracks = torch.as_tensor(tracks).double()
    T = tracks.shape[0]
    cen = centered_tracks(tracks, weights)
    y = cen["centered"]
    n = int(y.shape[1])
    xscale = float(tracks.abs().max()) if tracks.numel() else 1.0
    floor = (float(torch.finfo(torch.float64).eps) * max(xscale, 1e-300)) ** 2
    rows, Rs, covs = [], [], []
    for t in range(T):
        k = weighted_kabsch_rotation(y[0], y[t], weights)
        obs = rotation_observability(y[t], weights, information)
        Rs.append(k["R"])
        # Calibrated rotation covariance. The information matrix M is only
        # defined up to the measurement noise scale, so without this the
        # covariance is in arbitrary units and any test built on it is
        # meaningless -- the same trap the twist fit avoids. A weighted Kabsch
        # over N points spends 3 parameters, so sigma^2 = residual / (3N - 3),
        # floored at the arithmetic noise of the positions themselves.
        sigma2 = max(k["kabsch_error"] / max(3 * n - 3, 1), floor)
        covs.append(obs["rotation_covariance"] * sigma2)
        rows.append({**k, "rotation_sigma2": sigma2,
                     "rotation_covariance_calibrated": covs[-1],
                     **{"observability_" + a: b for a, b in obs.items()}})
    R = torch.stack(Rs)
    log = rotation_log_trajectory(R)
    obs0 = rotation_observability(y[0], weights, information)
    return {"centres": cen["centres"], "centered": y, "R": R,
            "per_frame": rows, "rotation_observability": obs0,
            "rotation_covariance": torch.stack(covs), **log}


def relative_vector_rotation(tracks, edges) -> dict:
    """Section 31's `relative_vector_rotation`, the finite P/R signature itself.

    r_ij(t) = R(t) r_ij(0), so for prismatic motion the DIRECTION of every
    relative vector is frozen while for revolute motion it turns. This is the
    per-edge version of the test and needs no part-level fit at all, which
    makes it usable as graph evidence in Phase 2.
    """
    x = torch.as_tensor(tracks).double()
    e = torch.as_tensor(edges).long().reshape(-1, 2)
    r = x[:, e[:, 0], :] - x[:, e[:, 1], :]                  # [T, E, 3]
    n0 = r[0].norm(dim=1).clamp_min(1e-300)
    nt = r.norm(dim=2).clamp_min(1e-300)
    cos = (r * r[0]).sum(2) / (nt * n0)
    ang = torch.arccos(cos.clamp(-1.0, 1.0))
    return {"edges": e, "angle_per_frame": ang,
            "max_angle": ang.max(dim=0).values,
            "length_drift": (nt - n0).abs().max(dim=0).values / n0,
            "note": ("section 4: length preserved but direction turning is "
                     "the finite revolute signature; both preserved is "
                     "prismatic")}
