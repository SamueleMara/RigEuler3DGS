"""Chronological twists for the backward pass.

CODEX_BIDIRECTIONAL_IEKF_AND_ARTICULATED_3DGS_MERGE.md section E.

The backward filter walks the sequence last-to-first with a POSITIVE dt, so
its internal velocity state is a reverse-time twist. Reversing that array and
calling the result chronological velocity is wrong by a sign: the state says
"how the body moves as the processing advances", and processing advances
backwards through physical time.

The fix is not to negate the stored array either. The stored twist is a
filter state carrying its own linearisation and its own covariance; what the
comparison needs is the twist implied by the POSES once they are back in
original order. So this module reconstructs it:

    xi_t = (1 / dt_t) * Log( T_{t+1} (T_t)^-1 )^v

That is the left/spatial increment, matching the convention the rest of the
pipeline uses (`boxplus` is R+ = Exp(hat(dphi)) R, a left perturbation), and
it is computed identically for both directions so the two are comparable by
construction.
"""

from __future__ import annotations

import torch

from estimator.joint_manifold import se3_log


def _as_pose_arrays(poses):
    """Accept [T,4,4] or a pair (R [T,3,3], p [T,3]) and return both."""
    if isinstance(poses, (tuple, list)) and len(poses) == 2:
        R, p = poses
        R = torch.as_tensor(R).double()
        p = torch.as_tensor(p).double()
    else:
        M = torch.as_tensor(poses).double()
        if M.ndim != 3 or M.shape[-2:] != (4, 4):
            raise ValueError(f"expected [T,4,4] poses, got {tuple(M.shape)}")
        R, p = M[:, :3, :3], M[:, :3, 3]
    if R.shape[0] != p.shape[0]:
        raise ValueError(f"{R.shape[0]} rotations but {p.shape[0]} translations")
    return R.contiguous(), p.contiguous()


def relative_increments(poses):
    """Left/spatial increments dT_t = T_{t+1} T_t^-1, as [T-1,6] twists.

    Indexed by the EARLIER frame: entry t describes the motion from t to t+1.
    """
    R, p = _as_pose_arrays(poses)
    T = R.shape[0]
    if T < 2:
        raise ValueError(f"need at least two poses to form an increment, got {T}")
    Ra, pa = R[:-1], p[:-1]
    Rb, pb = R[1:], p[1:]
    # T_{t+1} T_t^-1 : rotation Rb Ra^T, translation pb - (Rb Ra^T) pa
    dR = torch.einsum("nij,nkj->nik", Rb, Ra)
    dp = pb - torch.einsum("nij,nj->ni", dR, pa)
    return se3_log(dR, dp), dR, dp


def chronological_twists(poses, dt):
    """Twist per frame implied by chronologically ordered poses.

    `poses` must ALREADY be in original chronological order -- restore the
    order first, then call this. Returns [T,6] with [rho; phi] per frame; the
    last entry repeats the previous one so the array lines up with the poses,
    which is a padding convention, not a measurement.
    """
    eta, _, _ = relative_increments(poses)
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(
            f"dt must be positive, got {dt}. The backward pass reverses the "
            "ORDER and keeps dt positive; it never negates time.")
    xi = eta / dt
    return torch.cat([xi, xi[-1:]], dim=0)


def restore_chronological(values, direction):
    """Put a per-processing-step array back into original frame order.

    Forward is already chronological. Backward is reversed. This is the only
    thing that should ever be reversed: poses and covariances are per-frame
    quantities and reorder cleanly. Velocities do NOT -- they are per-step
    rates in processing time, so after restoring the poses the twist is
    RECOMPUTED by chronological_twists rather than reordered.
    """
    from utils.sequence_direction import BACKWARD, normalise

    if normalise(direction) == BACKWARD:
        return torch.as_tensor(values).flip(0).contiguous()
    return torch.as_tensor(values).contiguous()


def twist_sign_agreement(forward_xi, backward_xi):
    """Do the two directions agree on where the body is going?

    Returns the fraction of frames whose translational and rotational parts
    point the same way, plus the mean cosine. Section E's test is exactly
    this: a constant positive motion processed backward must come back with
    the SAME sign once poses are restored and the twist recomputed.
    """
    f = torch.as_tensor(forward_xi).double().reshape(-1, 6)
    b = torch.as_tensor(backward_xi).double().reshape(-1, 6)
    n = min(f.shape[0], b.shape[0])
    f, b = f[:n], b[:n]

    def _agreement(a, c):
        na, nc = a.norm(dim=-1), c.norm(dim=-1)
        live = (na > 1e-12) & (nc > 1e-12)
        if not bool(live.any()):
            return float("nan"), float("nan")
        cos = (a[live] * c[live]).sum(-1) / (na[live] * nc[live])
        return float((cos > 0).double().mean()), float(cos.mean())

    rho_frac, rho_cos = _agreement(f[:, :3], b[:, :3])
    phi_frac, phi_cos = _agreement(f[:, 3:], b[:, 3:])
    return {
        "frames_compared": int(n),
        "translation_same_sign_fraction": rho_frac,
        "translation_mean_cosine": rho_cos,
        "rotation_same_sign_fraction": phi_frac,
        "rotation_mean_cosine": phi_cos,
    }
