"""Sections 4-6, 8: the exact prismatic null and its curvature directions.

CODEX_NEXT_EFFICIENT_P_NULL_AND_PHASE2_DEBUG.md.

The physical null is `H_P: beta = 0` evaluated AT THE BEST P-CONSTRAINED
SOLUTION (section 2), not at an unrestricted screw whose omega is then
inspected. So this module does two things:

  * fit the null -- run the same Phase-2 IEKF with `omega` pinned to zero, so
    the prismatic direction and the whole q trajectory get every chance to
    explain the images (section 6);
  * name the curvature directions -- the two tangent coordinates that move
    `omega` off zero (section 8).

At `zeta_P = [u, 0]` with `|u| = 1` the constraint Jacobian is
`[[0, u^T], [2u^T, 0]]`, so the tangent space is exactly

    delta_u perpendicular to u   (2 DOF, the P direction: gamma)
    delta_omega perpendicular to u   (2 DOF, curvature: beta)

and ONE orthonormal basis `E` of `u`'s orthogonal complement serves both. That
is why the curvature hypothesis has exactly two degrees of freedom and not
three (section 8 forbids treating the three Cartesian omega components as
independent directions).

Building the basis explicitly like this also means the existing, already
validated `joint_jvp` computes both blocks: no second derivation of the
kinematics, and therefore no second chance to get it wrong.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .joint_kinematics import current_points, joint_jvp, joint_vjp
from .joint_manifold import constraint_jacobian, constraint_values

GAMMA_DIM = 2        # the P direction's two tangent DOF
BETA_DIM = 2         # section 8: curvature has exactly two DOF
NULL_ERROR_DIM = 4   # [dq, dqdot, dgamma(2)]


def perpendicular_basis(u: Tensor) -> Tensor:
    """E in R^{3x2}, orthonormal, with E^T u = 0 (section 4)."""
    u = u.double().reshape(3)
    n = u.norm().clamp_min(1e-300)
    e = u / n
    tmp = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=u.device)
    if abs(float(e @ tmp)) > 0.9:
        tmp = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64,
                           device=u.device)
    e1 = torch.linalg.cross(e, tmp)
    e1 = e1 / e1.norm().clamp_min(1e-300)
    e2 = torch.linalg.cross(e, e1)
    return torch.stack([e1, e2], dim=1)            # [3,2]


def null_tangent_basis(zeta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(B, B_gamma, B_beta) at a prismatic state.

    `B = [B_gamma | B_beta]` is a valid 6x4 tangent basis -- orthonormal, and
    annihilated by the constraint Jacobian -- so it can be handed straight to
    the existing `joint_jvp`. The split is what makes gamma and beta separately
    addressable:

        B_gamma = [[E], [0]]   moves the prismatic direction
        B_beta  = [[0], [E]]   moves omega off zero
    """
    z = zeta.double().reshape(6)
    E = perpendicular_basis(z[:3])
    zero = torch.zeros((3, 2), dtype=torch.float64, device=z.device)
    B_gamma = torch.cat([E, zero], dim=0)          # [6,2]
    B_beta = torch.cat([zero, E], dim=0)           # [6,2]
    return torch.cat([B_gamma, B_beta], dim=1), B_gamma, B_beta


def is_prismatic_state(zeta: Tensor, tol: float = 1e-12) -> bool:
    return float(zeta.double().reshape(6)[3:].abs().max()) <= tol


def project_to_prismatic(part) -> None:
    """Pin omega to exactly zero and renormalise u (section 4).

    `omega = 0` exactly, not approximately: the null must BE prismatic, so
    that the score below is evaluated on the constraint surface rather than
    near it.
    """
    z = part.zeta.double().reshape(6).clone()
    z[3:] = 0.0
    n = z[:3].norm().clamp_min(1e-300)
    z[:3] = z[:3] / n
    part.zeta = z


def null_jvp(part, frame: int, delta: Tensor, mu_now: Tensor) -> Tensor:
    """Pose tangent for `[dq, dqdot, dgamma(2)]` at a prismatic state.

    Delegates to the validated `joint_jvp` with the explicit basis, mapping
    `dgamma` into the first two coordinates of `delta_theta_J` and leaving the
    curvature coordinates at zero -- which IS the null constraint
    `delta_omega = 0`.
    """
    d = torch.as_tensor(delta).double().reshape(-1)
    B, _bg, _bb = null_tangent_basis(part.zeta)
    full = torch.zeros(6, dtype=torch.float64)
    full[0] = d[0]
    full[1] = d[1] if d.numel() > 1 else 0.0
    full[2:4] = d[2:4] if d.numel() >= 4 else 0.0
    return joint_jvp(part, frame, full, mu_now=mu_now, basis=B)


def curvature_jvp(part, frame: int, dbeta: Tensor, mu_now: Tensor) -> Tensor:
    """Pose tangent for the two curvature coordinates `beta` (section 8).

    At frame 0 this is EXACTLY zero: the geometry perturbation enters through
    `J_l(q xi) q delta_xi`, and `q(0) = 0` by the gauge. Section 20 requires
    that identity to hold to numerical precision, and it holds structurally
    here rather than by a special case.
    """
    d = torch.as_tensor(dbeta).double().reshape(-1)
    B, _bg, _bb = null_tangent_basis(part.zeta)
    full = torch.zeros(6, dtype=torch.float64)
    full[4:6] = d[:2]
    return joint_jvp(part, frame, full, mu_now=mu_now, basis=B)


def null_vjp(part, frame: int, pose_adjoint: Tensor, mu_now: Tensor) -> Tensor:
    """Adjoint of `null_jvp` -> `[dq, dqdot, dgamma(2)]`."""
    B, _bg, _bb = null_tangent_basis(part.zeta)
    full = joint_vjp(part, frame, pose_adjoint, mu_now=mu_now, basis=B)
    out = torch.zeros(NULL_ERROR_DIM, dtype=torch.float64, device=full.device)
    out[0] = full[0]
    out[1] = full[1]
    out[2:4] = full[2:4]
    return out


def curvature_vjp(part, frame: int, pose_adjoint: Tensor,
                  mu_now: Tensor) -> Tensor:
    """Adjoint of `curvature_jvp` -> the two beta coordinates."""
    B, _bg, _bb = null_tangent_basis(part.zeta)
    full = joint_vjp(part, frame, pose_adjoint, mu_now=mu_now, basis=B)
    return full[4:6].clone()


# ---------------------------------------------------------------------------
# section 6: fit the null before testing it
# ---------------------------------------------------------------------------
def run_prismatic_null_sequence(state, mu0, R0, observations, linearize,
                                precision_fn, times=None, cfg=None,
                                max_passes: int = 4,
                                stationarity_tol: float = 1e-3) -> dict:
    """Section 6: run the SAME Phase-2 IEKF with omega pinned to zero.

    Repeated passes from frame 0 until the measurement-only nuisance score
    stops moving the P direction materially. The number of passes is NOT a
    dataset-specific constant: it is whatever it takes for the direction to
    stop changing, bounded by `max_passes`, and every pass is recorded.
    """
    from .joint_iekf import JointIEKFConfig, run_window

    cfg = cfg or JointIEKFConfig()
    for p in state.parts:
        project_to_prismatic(p)

    passes, prev_dir = [], state.parts[0].zeta[:3].clone()
    out = None
    for k in range(int(max_passes)):
        for p in state.parts:
            p.q = torch.zeros_like(p.q)
            p.qdot = torch.zeros_like(p.qdot)
        out = run_window(state, mu0, R0, observations, linearize,
                         precision_fn, times=times, cfg=cfg,
                         label=f"P-null pass {k}")
        # omega is held at zero throughout: the unrestricted runtime would
        # have moved it, and that is exactly what the null forbids.
        for p in state.parts:
            project_to_prismatic(p)
        new_dir = state.parts[0].zeta[:3].clone()
        change = float(torch.arccos(
            (new_dir @ prev_dir).clamp(-1.0, 1.0)))
        energy = float(sum(r.energy_after for r in out["records"]))
        passes.append({"pass": k, "objective": energy,
                       "direction_change_rad": change,
                       "direction": new_dir.tolist()})
        prev_dir = new_dir
        if k > 0 and change < stationarity_tol:
            break
    return {"state": state, "result": out, "passes": passes,
            "null_pass_count": len(passes),
            "null_objective_per_pass": [p["objective"] for p in passes],
            "null_direction_change": [p["direction_change_rad"]
                                      for p in passes]}
