"""Sections 12-14: initialize M_R, and say honestly when it cannot be fitted.

CODEX_COMPLETE_PHASE2_DEBUG_AND_PARALLEL_PR_MODELS.md.

At `theta_t = 0` the finite axis is inherently unobservable: the revolute
geometry tangent is

    dphi = theta_t J_l(theta_t e_R) E_R deta

which vanishes identically, so no RGB-D residual can move `e_R` or `rho`.
That is a property of the model, not a bug (section 12).

But "the data contains no rotation" and "the optimizer was handed a singular
start and never got to ask" are different claims, and the previous
implementation could not tell them apart -- it reported the energy of an
arbitrary initial axis as if it were a fitted R model.

Section 13's deterministic hierarchy fixes that:

    A. finite-axis initialization from the generic screw, when its angular
       block is observable and nonzero;
    B. otherwise the continuous zero-pitch screw chart as a BOOTSTRAP, where
       `eta = 0` is exactly prismatic and is a regular interior point -- so
       the question "does the residual want curvature?" can be asked without
       ever dividing by a vanishing angle;
    C. if the bootstrap is still curvature-unobservable, return
       `R_GEOMETRY_UNOBSERVABLE` and do not pretend otherwise.

No step uses a hand-tuned axis-radius, omega or angle threshold. Observability
is decided by the numerical rank of the geometry information and by whether
the nonlinear solve accepted a decrease (section 13B).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .joint_manifold import revolute_axis
from .phase2_revolute import RevolutePart, axis_basis

FITTED = "FITTED"
UNOBSERVABLE = "R_GEOMETRY_UNOBSERVABLE"
FAILED = "OPTIMIZATION_FAILED"


def information_rank(F: Tensor, tol: float | None = None) -> dict:
    """Numerical rank of a geometry information block, with no magic number.

    The tolerance is the standard one for a symmetric PSD matrix --
    `max(eig) * n * eps` -- so it scales with the problem rather than with a
    constant someone picked once on one scene.
    """
    F = torch.as_tensor(F).double()
    ev = torch.linalg.eigvalsh(0.5 * (F + F.T))
    top = float(ev.max()) if ev.numel() else 0.0
    n = F.shape[0]
    t = (top * n * torch.finfo(torch.float64).eps) if tol is None else float(tol)
    return {"eigenvalues": ev.tolist(), "max": top,
            "tolerance": t, "rank": int((ev > t).sum()), "full": n}


def finite_axis_from_screw(zeta: Tensor, length: float, centroid: Tensor,
                           q: Tensor) -> dict | None:
    """Section 13A: the generic screw's chart -> `(e_R, a_R, theta_t)`.

    `theta_t = q_t |omega|` because the chart's `q` is dimensionless and the
    physical angle carries the generator's magnitude. Reading `q` as the angle
    directly would scale the whole trajectory by `|omega|`.
    """
    ax = revolute_axis(torch.as_tensor(zeta).double(), float(length),
                       torch.as_tensor(centroid).double())
    if ax is None:
        return None
    theta = torch.as_tensor(q).double().reshape(-1) * ax["omega_norm"]
    return {"axis": ax["axis"], "axis_point": ax["axis_point"],
            "theta": theta, "omega_norm": ax["omega_norm"],
            "pitch": ax["pitch"]}


def revolute_part_from_axis(members, mu0, axis, axis_point, theta,
                            thetadot=None) -> RevolutePart:
    """Build `(e_R, rho)` so that `e_R^T(a_R - c_0) = 0` holds exactly."""
    members = torch.as_tensor(members).long().reshape(-1)
    c0 = mu0.double().cpu()[members].mean(0)
    e = torch.as_tensor(axis).double().reshape(3).cpu()
    e = e / e.norm().clamp_min(1e-300)
    E = axis_basis(e)
    a = torch.as_tensor(axis_point).double().reshape(3).cpu()
    # Only the transverse component of the offset is representable; the part
    # along the axis moves no Gaussian and is pure gauge.
    rho = E.T @ (a - c0)
    return RevolutePart(members, c0, e, rho,
                        torch.as_tensor(theta).double().reshape(-1).cpu(),
                        thetadot)


def screw_curvature_observability(screw_state, screw_records) -> dict:
    """Is the generic screw's ANGULAR block observable and nonzero?

    Two separate questions, and both must be yes:

      * does the fitted generator actually have an angular part (`|omega| > 0`
        as a number, not as a threshold), and
      * did the data have information about the geometry -- i.e. is the
        accumulated geometry information block numerically full rank?

    A nonzero `omega` with a rank-deficient information block is an arbitrary
    axis that nothing pushed on, which is precisely the case section 13C is
    there to catch.
    """
    p = screw_state.parts[0]
    z = p.zeta.double().reshape(6)
    omega_norm = float(z[3:].norm())
    F = torch.zeros((4, 4), dtype=torch.float64)
    accepted = 0
    for r in (screw_records or []):
        accepted += int(getattr(r, "accepted", 0))
        ob = getattr(r, "observability", None) or {}
        ev = ob.get("geometry_information_eigenvalues")
        if ev:
            F = F + torch.diag(torch.tensor(ev, dtype=torch.float64))
    rk = information_rank(F)
    return {"omega_norm": omega_norm,
            "angular_block_nonzero": omega_norm > 0.0,
            "geometry_information_rank": rk["rank"],
            "geometry_information_full": rk["full"],
            "geometry_information_eigenvalues": rk["eigenvalues"],
            "geometry_information_tolerance": rk["tolerance"],
            "accepted_iterations": accepted,
            "observable": bool(omega_norm > 0.0 and rk["rank"] == rk["full"]
                               and accepted > 0)}


def initialize_revolute(members, mu0, positions, screw_state=None,
                        screw_records=None, bootstrap=None) -> dict:
    """Section 13: run the hierarchy and report which branch produced the state.

    `bootstrap` is an optional callable with no arguments that runs the
    continuous zero-pitch chart on the real data and returns
    `(state, records)`. It is only consulted when branch A is not available,
    and its result is NEVER reported as the final R joint (section 13B) -- it
    exists to answer "does the residual want curvature?" and, if so, to hand
    over a finite-axis starting point.
    """
    members = torch.as_tensor(members).long().reshape(-1)
    diag = {"hierarchy": []}

    # ---- A: the generic screw, if its angular block is observable --------
    if screw_state is not None:
        obs = screw_curvature_observability(screw_state, screw_records)
        diag["hierarchy"].append({"branch": "A_generic_screw", **obs})
        if obs["observable"]:
            p = screw_state.parts[0]
            fa = finite_axis_from_screw(p.zeta, p.length, p.centroid, p.q)
            if fa is not None:
                part = revolute_part_from_axis(members, mu0, fa["axis"],
                                               fa["axis_point"], fa["theta"])
                diag.update(initialization_source="A_generic_screw",
                            status=FITTED,
                            max_abs_theta=float(fa["theta"].abs().max()),
                            omega_norm=fa["omega_norm"], pitch=fa["pitch"])
                return {"part": part, **diag}

    # ---- B: the continuous near-P curvature bootstrap --------------------
    if bootstrap is not None:
        try:
            bstate, brecords = bootstrap()
        except Exception as exc:                       # noqa: BLE001
            diag["hierarchy"].append({"branch": "B_curvature_bootstrap",
                                      "error": repr(exc)})
            bstate = None
        if bstate is not None:
            obs = screw_curvature_observability(bstate, brecords)
            diag["hierarchy"].append({"branch": "B_curvature_bootstrap", **obs})
            if obs["observable"]:
                p = bstate.parts[0]
                fa = finite_axis_from_screw(p.zeta, p.length, p.centroid, p.q)
                if fa is not None:
                    part = revolute_part_from_axis(members, mu0, fa["axis"],
                                                   fa["axis_point"],
                                                   fa["theta"])
                    diag.update(initialization_source="B_curvature_bootstrap",
                                status=FITTED,
                                max_abs_theta=float(fa["theta"].abs().max()),
                                omega_norm=fa["omega_norm"],
                                pitch=fa["pitch"],
                                note=("section 13B: the bootstrap only "
                                      "supplied the starting point; the "
                                      "reported R joint is the finite-axis "
                                      "model fitted from here"))
                    return {"part": part, **diag}

    # ---- C: curvature is not observable ---------------------------------
    diag.update(initialization_source=None, status=UNOBSERVABLE,
                max_abs_theta=0.0,
                reason=("sections 13C, 14: neither the generic screw nor the "
                        "continuous curvature bootstrap produced an observable "
                        "nonzero rotation, so there is no finite axis to fit. "
                        "The energy of an arbitrary initial axis is NOT an R "
                        "hypothesis and is not reported as one."))
    return {"part": None, **diag}


def fit_status(part, records, before_axis, before_rho) -> dict:
    """Section 14: the per-run validity block saved beside every R fit."""
    accepted = sum(int(getattr(r, "accepted", 0)) for r in (records or []))
    F = torch.zeros((4, 4), dtype=torch.float64)
    for r in (records or []):
        ev = (getattr(r, "observability", None) or {}).get(
            "geometry_information_eigenvalues")
        if ev:
            F = F + torch.diag(torch.tensor(ev, dtype=torch.float64))
    rk = information_rank(F)
    th = float(part.theta.abs().max()) if part is not None else 0.0
    d_axis = (float((part.axis - torch.as_tensor(before_axis).double()).norm())
              if part is not None else 0.0)
    d_rho = (float((part.rho - torch.as_tensor(before_rho).double()).norm())
             if part is not None else 0.0)
    if part is None:
        status = UNOBSERVABLE
    elif th <= 0.0 or rk["rank"] == 0:
        status = UNOBSERVABLE
    elif accepted == 0:
        status = FAILED
    else:
        status = FITTED
    return {"max_abs_theta": th, "accepted_iterations": accepted,
            "geometry_information_eigenvalues": rk["eigenvalues"],
            "geometry_information_rank": rk["rank"],
            "geometry_information_full": rk["full"],
            "axis_update_norm": d_axis, "axis_offset_update_norm": d_rho,
            "R_fit_valid": status == FITTED, "R_fit_status": status}
