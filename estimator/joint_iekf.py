"""Sections 13-16, 19, 22: the Phase-2 rigid-screw IEKF runtime.

CODEX_CONTINUE_PHASE2_RIGID_SCREW_IEKF_INTEGRATION.md.

This is the wiring, not a new estimator (section 1):

    delta y = H_G J_{G,J} delta x_J

The renderer cache, the RGB-D residual, the measurement precision, the
truncated-CG solve and the trust-region acceptance are Phase 1's own functions,
reached through callbacks. Nothing here re-derives a measurement model, adds a
loss, or fits a free SE(3) trajectory first.

The state is six numbers for K=1 -- `[q, qdot, theta_J(4)]` -- so the posterior
is kept as a FULL dense 6x6 (section 4). Section 5 is emphatic that the
off-diagonal blocks are not optional: `P_q,qdot` is the only route by which
image evidence reaches `qdot` at all, since a single frame has no `qdot`
derivative (section 7), and `P_q,theta` is how the trajectory and the static
geometry resolve each other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .joint_kinematics import state_to_gaussians
from .joint_manifold import tangent_basis
from .joint_process import predict as process_predict
from .joint_renderer_bridge import joint_apply_H, joint_apply_HT
from .joint_state import GEOMETRY_DIM, JOINT_ERROR_DIM
from .trust_region import ScalingMetric, TrustRegionConfig, adapt_radius, steihaug_cg


@dataclass
class JointIEKFConfig:
    max_iters: int = 8
    max_cg_iters: int = 40
    radius_init: float = 1.0
    sigma_qddot: float = 1.0         # process noise on qdot
    prior_weight: float = 1.0
    psd_floor: float = 1e-12
    verbose: bool = False
    # "map" (default, every Phase 2 / Phase 3 runner): steps are solved AND accepted on
    # E_total = E_meas + E_prior, chart-mode prior information transported as
    # G^-T Lam G^-1 (e_new = G e_old). "pre_repair_legacy": the pre-2026-10-21 frame
    # step exactly -- acceptance on E_meas alone and the old G^-1 Lam G^-T transport --
    # kept ONLY for the legacy joint-identification diagnostic estimator/prismatic_null.py
    # (not in the active pipeline; the repair spec does not change joint identification).
    acceptance: str = "map"

    def __post_init__(self):
        if self.acceptance not in ("map", "pre_repair_legacy"):
            raise ValueError(f"acceptance must be 'map' or 'pre_repair_legacy', got {self.acceptance!r}")


@dataclass
class FrameRecord:
    frame: int
    iterations: int
    accepted: int
    energy_before: float
    energy_after: float
    radius: float
    rho_last: float
    information: Tensor              # [6K,6K] posterior information
    posterior: Tensor                # [6K,6K] posterior covariance
    q: float
    qdot: float
    sign_flipped: bool
    observability: dict


def _metric(prior_information: Tensor, cfg: JointIEKFConfig) -> ScalingMetric:
    """Phase-1's trust-region metric, built from the ACTUAL prior.

    `ScalingMetric`'s contract is that radius 1 means "the RMS state component
    moves this many PRIOR SIGMAS". Hardcoding those sigmas breaks that: `q` is
    a trajectory coordinate whose natural scale is how far the joint moves in
    one frame, not a small perturbation. With a fixed `sigma_q = 0.05` the
    region capped `|dq|` at 0.1225 per iteration, which was ample for a slow
    prismatic slide and left a revolute joint sweeping 0.6 rad over seven
    frames permanently behind -- it tracked two frames and then lost the
    object, ending 27 degrees out.

    The prior information already states the answer: its diagonal IS the
    inverse prior variance of each coordinate, and it is rebuilt every frame
    from the process model. Using it makes the radius self-calibrating and
    removes three constants that would otherwise need tuning per scene.
    """
    d = prior_information.diagonal().clamp_min(1e-12)
    n = int(d.numel())
    return ScalingMetric(weight=(d / n).to(torch.float64))


def apply_sign_gauge(state, reference=None) -> list:
    """Section 16: remove the remaining Z_2 gauge (q, xi) == (-q, -xi).

    The physical transform is invariant under flipping the generator and the
    coordinate together, so a recursive filter can walk across that symmetry
    from one frame to the next and produce a trajectory that jumps sign while
    rendering identically. Continuity is enforced against the previous
    geometry rather than against any fixed direction, and `q`, `qdot` and the
    covariance's sign-dependent coordinates are flipped with it.

    Returns a per-part flag so the caller can record what happened.
    """
    flips = []
    if any(not hasattr(p, "zeta") for p in state.parts):
        # Stage B's M_P / M_R have no `zeta` and no sign ambiguity to gauge:
        # their retraction rotates a carried frame, so the nominal never flips.
        return [False] * len(state.parts)
    for k, p in enumerate(state.parts):
        ref = None if reference is None else reference[k]
        flip = bool(ref is not None and float(p.zeta @ ref) < 0.0)
        if flip:
            p.zeta = -p.zeta
            p.q = -p.q
            p.qdot = -p.qdot
        flips.append(flip)
    return flips


def sign_jacobian(state, flips) -> Tensor:
    """The error-state Jacobian of the sign flip (section 16).

    `q` and `qdot` negate, so their error coordinates do too. The geometry
    tangent basis is recomputed at `-zeta`, and its relation to the old basis
    is `B_new^T (-B_old)` -- the same first-order transport as any other
    change of tangent frame, with the reflection folded in.
    """
    G = torch.eye(state.dim, dtype=torch.float64)
    for k, (p, flip) in enumerate(zip(state.parts, flips)):
        if not flip:
            continue
        i = k * JOINT_ERROR_DIM
        G[i, i] = -1.0
        G[i + 1, i + 1] = -1.0
        B_new = tangent_basis(p.zeta)
        B_old = tangent_basis(-p.zeta)
        G[i + 2:i + 6, i + 2:i + 6] = B_new.T @ (-B_old)
    return G


def _to_view_device(ob, view):
    """Phase 1 stages frames on the host by default (`--frames-on-cpu`), so an
    observation arrives on the CPU while the rendered views are on the device.
    Phase 1 moves them per frame through its own `_to_device`; this does the
    same at the one boundary where the joint runtime touches an observation.
    """
    if ob is None:
        return None
    return tuple(t.to(view.rgb.device) if torch.is_tensor(t) else t
                 for t in ob)


def _camera_terms(lin, observation, precision_fn, model=None):
    """The frame's per-camera `CameraTerms` at a linearization.

    `precision_fn(lin)` either returns CameraTerms itself (the production runners:
    estimator/phase2_shared.py, robust Student-t NLL objective with the same sigma
    as the GN weights), or the legacy per-camera (p_rgb, p_depth) tuples; those are
    taken as a FIXED quadratic precision supplied by the caller, with residuals
    from `observation` (masked: Y(b) - C, r_D on V_used, r_A = M - A)."""
    from .observation_terms import CameraTerms
    res = precision_fn(lin)
    if len(res) != len(lin.views):
        raise AssertionError(f"{len(lin.views)} views but {len(res)} precision entries: "
                             "refusing a silent zip truncation")
    if res and isinstance(res[0], CameraTerms):
        return res
    if len(observation) != len(lin.views):
        raise AssertionError(f"{len(lin.views)} views but {len(observation)} observations: "
                             "refusing a silent zip truncation")
    out = []
    for v, ob0, (pc, pd) in zip(lin.views, observation, res):
        kap = r_a = None
        if model is not None:
            from .masked_observation import residuals
            ob = ob0.to(v.rgb.device)
            r_c, r_d, r_a_ = residuals(v, ob, model.background_tensor(v.rgb.device))
            if model.alpha_term:
                kap, r_a = model.kappa(pc), r_a_
        else:
            ob = _to_view_device(ob0, v)
            if ob is None:
                r_c, r_d = torch.zeros_like(v.rgb), torch.zeros_like(v.depth)
            else:
                r_c = ob[0] - v.rgb
                r_d = ob[1] - v.depth
        r_c = torch.where(pc > 0, r_c, torch.zeros_like(r_c)) if pc.dim() == r_c.dim() else r_c
        r_d = torch.where(pd > 0, r_d, torch.zeros_like(r_d))
        e_c = 0.5 * float((pc.double() * r_c.double().square()).sum())
        e_d = 0.5 * float((pd.double() * r_d.double().square()).sum())
        e_a = 0.0 if kap is None else 0.5 * float((kap.double() * r_a.double().square()).sum())
        n = int((pd > 0).sum())
        out.append(CameraTerms(pc, pd, r_c, r_d, kap, r_a, e_c, e_d, e_a, n, n, 0))
    return out


def _energy(linearize, state, frame, mu0, R0, observation, precision_fn,
            observation_model=None):
    """The measurement part of the frame objective at `state`, on the TRUE
    nonlinear render. Returns (energy, lin); the per-camera terms and the split
    are attached to `lin` (`frame_terms`, `energy_parts`) so the solve at this
    linearization uses exactly the terms the energy was computed from."""
    from .observation_terms import summarize
    lin = linearize(state, frame)
    terms = _camera_terms(lin, observation, precision_fn, observation_model)
    lin.frame_terms = terms
    sm = summarize(terms)
    lin.energy_parts = sm
    return sm["rgb"] + sm["depth"] + sm["alpha"], lin


class _FramePrior:
    """E_prior(x) = w/2 e(x)^T Lambda_minus e(x) for one frame (P0-2).

    x_minus (the predicted nominal) and Lambda_minus are snapshotted ONCE, after
    the process prediction and before any correction, and never re-applied.

    exact   the state defines `prior_error(frame, x_minus)` = x boxminus x_minus
            in its own tangent convention (rigid: left SE(3) log + twist
            difference; displacement: (q, qdot) difference). J = d e(x boxplus d)/dd
            comes from `prior_jacobian` when the state has it, else from a central
            difference through the state's OWN retraction (n <= 12, no rendering).
            E_prior depends on the physical state only, so it is invariant to the
            tangent chart the solver happens to be in.
    chart   a state without `prior_error`: the previous first-order bookkeeping
            (tracked offset delta, transported by the reset Jacobian G), now also
            used for acceptance, so the solved and the accepted objective agree.
    """

    FD_STEP = 1e-6

    def __init__(self, state, frame, Lambda_minus, weight):
        import copy
        self.frame = int(frame)
        self.w = float(weight)
        self.Lam = 0.5 * (Lambda_minus.double() + Lambda_minus.double().T)
        self.exact = hasattr(state, "prior_error")
        self.x_minus = copy.deepcopy(state) if self.exact else None
        self.delta = torch.zeros(state.dim, dtype=torch.float64)

    def error(self, x, step=None) -> Tensor:
        if self.exact:
            return x.prior_error(self.frame, self.x_minus).double().reshape(-1)
        return self.delta + (0.0 if step is None else torch.as_tensor(step).double().reshape(-1))

    def jacobian(self, x) -> Tensor:
        n = x.dim
        if not self.exact:
            return torch.eye(n, dtype=torch.float64)
        if hasattr(x, "prior_jacobian"):
            return x.prior_jacobian(self.frame, self.x_minus).double()
        import copy
        from .joint_process import apply_correction
        J = torch.zeros((n, n), dtype=torch.float64)
        h = self.FD_STEP
        for k in range(n):
            d = torch.zeros(n, dtype=torch.float64)
            d[k] = h
            xp = apply_correction(copy.deepcopy(x), self.frame, d, transport=False)[0]
            xm = apply_correction(copy.deepcopy(x), self.frame, -d, transport=False)[0]
            J[:, k] = (self.error(xp) - self.error(xm)) / (2.0 * h)
        return J

    def energy(self, x, step=None) -> float:
        e = self.error(x, step)
        return 0.5 * self.w * float(e @ (self.Lam @ e))

    def gradient_parts(self, x):
        """(w J^T Lam e, w J^T Lam J) at the iterate x."""
        e, J = self.error(x), self.jacobian(x)
        return self.w * (J.T @ (self.Lam @ e)), self.w * (J.T @ self.Lam @ J)

    def accepted(self, step, G, legacy_transport=False):
        """Chart bookkeeping after an accepted step (no-op in the exact mode).

        e_new = G e_old, so E = 1/2 e_old^T Lam e_old = 1/2 e_new^T G^-T Lam G^-1 e_new.
        `legacy_transport` reproduces the pre-2026-10-21 G^-1 Lam G^-T (wrong unless G
        is orthogonal), for the pre_repair_legacy acceptance only."""
        if self.exact:
            return
        self.delta = G @ (self.delta + torch.as_tensor(step).double().reshape(-1))
        if abs(float(torch.det(G))) > 1e-12:
            if legacy_transport:
                self.Lam = torch.linalg.solve(G, torch.linalg.solve(G, self.Lam).T).T
            else:
                Gi = torch.linalg.inv(G)
                self.Lam = Gi.T @ self.Lam @ Gi


def frame_step(state, frame: int, mu0: Tensor, R0, observation, linearize,
               precision_fn, prior_information: Tensor,
               cfg: JointIEKFConfig | None = None, visibility=None,
               observation_model=None) -> FrameRecord:
    """Section 13: one Phase-2/3 frame, solved and accepted on ONE objective.

        E_total(x) = E_meas(x) + w/2 e(x)^T Lambda_minus e(x)

    E_meas is the measurement part of the frame (RGB, robust valid depth NLL and,
    in the expected-quad mode, the alpha term) on the true render; the prior is
    the predicted nominal x_minus with information Lambda_minus, snapshotted once.
    The GN model at the iterate x is
        b = J_meas^T W r - w J^T Lam e(x),   A = J_meas^T W J_meas + w J^T Lam J,
    the predicted decrease is b^T s - s^T A s / 2 for exactly the trial step, and
    the actual decrease is E_total(x) - E_total(x boxplus s) -- the same terms.
    (Before 2026-10-21 the solve used the prior but acceptance did not.)

    `linearize(state, frame)` and `precision_fn(lin)` are the caller's bindings
    to the renderer cache and the measurement model (see `_camera_terms`).
    """
    import copy
    from .joint_process import apply_correction
    cfg = cfg or JointIEKFConfig()
    dev = mu0.device
    n = state.dim
    metric = _metric(prior_information, cfg)
    om = observation_model
    if om is not None and visibility is not None:
        raise ValueError("the masked observation mode and the negative-ray constraint are "
                         "different measurement models; choose one")
    # Negative-ray constraint (opt-in, visibility_constraint.py): object-valid
    # RGB-D only, visible alpha attached to every linearization, the stage's
    # own lift J_g = state.jvp / state.vjp for C = J_A J_g.
    if visibility is not None:
        _lin0, _prec0 = linearize, precision_fn

        def linearize(st, fr, _f=_lin0):
            L = _f(st, fr)
            L.visibility_values = visibility.evaluate(L)
            return L

        def precision_fn(L, _f=_prec0):
            return [(pc * M.to(pc.device, pc.dtype)[None] if pc.dim() == 3 else pc * M.to(pc.device, pc.dtype),
                     pd * M.to(pd.device, pd.dtype))
                    for (pc, pd), M in zip(_f(L), visibility.masks, strict=True)]
    tr = TrustRegionConfig(radius_init=cfg.radius_init,
                           max_pcg_iters=cfg.max_cg_iters)
    radius = cfg.radius_init
    prior = _FramePrior(state, frame, prior_information, cfg.prior_weight)

    meas, lin = _energy(linearize, state, frame, mu0, R0, observation, precision_fn, om)
    e_prior = prior.energy(state)
    energy = meas + e_prior
    energy0 = energy
    legacy = cfg.acceptance == "pre_repair_legacy"
    meas0 = meas_cur = meas
    parts0 = {**lin.energy_parts, "prior": e_prior, "total": energy}
    accepted_count = 0
    rho_last = float("nan")
    it = 0
    history = []
    vis_ref = vis_cur = vis_last = None
    vis_log = []
    if visibility is not None:
        visibility.check_views(lin)
        if len(observation) != visibility.ncam:
            raise AssertionError(f"{len(observation)} observations but {visibility.ncam} cameras of masks: "
                                 "refusing a silent zip truncation")
        vis_ref = visibility.row_values(lin.visibility_values)
        vis_cur = vis_ref

    def _weighted(terms):
        return [((t.p_rgb * t.r_rgb, t.p_d * t.r_d, t.kappa * t.r_a) if t.kappa is not None
                 else (t.p_rgb * t.r_rgb, t.p_d * t.r_d)) for t in terms]

    def _meas_normal(L, st, terms):
        alpha = any(t.kappa is not None for t in terms)

        def op(p: Tensor) -> Tensor:
            tang = joint_apply_H(L, st, frame, p.to(dev), mu0, R0, alpha=alpha)
            if alpha:
                w = [(t.p_rgb * dc, t.p_d * dd, (t.kappa if t.kappa is not None else torch.zeros_like(da)) * da)
                     for t, (dc, dd, da) in zip(terms, tang)]
            else:
                w = [(t.p_rgb * dc, t.p_d * dd) for t, (dc, dd) in zip(terms, tang)]
            return joint_apply_HT(L, st, frame, w, mu0, R0).cpu()
        return op

    # Each iteration re-renders the whole scene, so at full scale this is
    # seconds per step. It also breaks early, hence absolute positioning.
    from utils.progress import ibar
    _bar = ibar(int(cfg.max_iters), f"solve f{frame:03d}")
    for it in range(1, int(cfg.max_iters) + 1):
        _bar.to(it - 1)
        terms = lin.frame_terms
        # ---- right-hand side: J_meas^T W r - w J^T Lam e ----------------
        rhs_meas = joint_apply_HT(lin, state, frame, _weighted(terms), mu0, R0).cpu()
        g_prior, N_prior = prior.gradient_parts(state)
        b = rhs_meas - g_prior
        meas_op = _meas_normal(lin, state, terms)

        # ---- matrix-free normal operator (section 12) --------------------
        def A_mv(p: Tensor, _m=meas_op, _N=N_prior) -> Tensor:
            return _m(p) + _N @ p.double()

        # Section 14: for K=1 the operator is small, so probing it with the
        # basis vectors gives it EXACTLY; its inverse is the CG preconditioner.
        # Steihaug-CG starts along b, whose qdot component is exactly zero (no
        # image qdot derivative), and the coupling that carries information into
        # qdot only appears after multiplication by A -- with the image
        # information on q orders of magnitude above the prior, unpreconditioned
        # CG met its tolerance before qdot moved.
        eye = torch.eye(n, dtype=torch.float64)
        A_dense = torch.stack([A_mv(eye[:, k]) for k in range(n)], dim=1)
        A_dense = 0.5 * (A_dense + A_dense.T)
        try:
            evals = torch.linalg.eigvalsh(A_dense)
            A_pc = (A_dense if float(evals.min()) > cfg.psd_floor
                    else A_dense + (cfg.psd_floor - float(evals.min()))
                    * torch.eye(n, dtype=torch.float64))
            A_inv = torch.linalg.inv(A_pc)
            M_inv = lambda r: A_inv @ r                        # noqa: E731
        except Exception:
            M_inv = None

        res = steihaug_cg(A_mv, b, radius, metric, M_inv=M_inv,
                          max_iters=cfg.max_cg_iters)
        step = res.step
        if not torch.isfinite(step).all() or float(step.norm()) == 0.0:
            history.append({"iteration": it, "stop": "zero_or_nonfinite_step", "radius": radius})
            break

        vis_audit = vis_info = None
        if visibility is not None:
            # constrained correction (projection of the trust-region step onto
            # C d <= u, C = J_A J_g) with up to two second-order corrections
            from .visibility_constraint import constrained_step

            def _lift(v, _st=state):
                return _st.jvp(frame, v.to(dev).double(), mu0, R0).float()

            def _liftT(pose, _st=state):
                return _st.vjp(frame, pose.double(), mu0, R0).cpu().double()
            step_tr = step
            margin = torch.zeros_like(vis_cur)
            n_soc = 0
            while True:
                step, vis_info, lin_pred = constrained_step(
                    visibility, lin, lin.visibility_values, vis_ref, vis_cur, A_mv, b, _lift, _liftT,
                    step_tr, radius, metric, dim_state=n, M_inv=M_inv, margin=margin)
                vis_last = (lin, A_mv, b, _lift, _liftT, M_inv)
                trial = copy.deepcopy(state)
                trial, _G = apply_correction(trial, frame, step, transport=False)
                m_new, lin_new = _energy(linearize, trial, frame, mu0, R0, observation,
                                         precision_fn, om)
                vis_new = visibility.row_values(lin_new.visibility_values)
                vis_audit = visibility.audit(vis_ref, vis_cur, vis_new)
                if vis_audit["ok"] or n_soc >= 2:
                    break
                hard = visibility.hard_rows(vis_ref)
                over = hard & (vis_new > visibility.policy.epsilon + visibility.policy.audit_tol)
                if not bool(over.any()):
                    break
                err = (vis_new - (vis_cur + torch.nan_to_num(lin_pred, nan=0.0))).clamp_min(0.0)
                margin = margin + torch.where(over, err + visibility.policy.audit_tol, torch.zeros_like(err))
                n_soc += 1
            vis_info["second_order_corrections"] = n_soc

        # ---- predicted reduction from the SAME quadratic MAP model -------
        pred = float((b * step).sum() - 0.5 * (step * A_mv(step)).sum())

        # ---- trial retraction, rerender, acceptance on E_total ----------
        if visibility is None:
            trial = copy.deepcopy(state)
            trial, _G = apply_correction(trial, frame, step, transport=False)
            m_new, lin_new = _energy(linearize, trial, frame, mu0, R0, observation,
                                     precision_fn, om)
        p_new = prior.energy(trial, step)
        e_new = m_new + p_new
        actual = (meas_cur - m_new) if legacy else (energy - e_new)
        rho = actual / pred if pred > 0 else float("-inf")
        if vis_audit is not None and not vis_audit["ok"]:
            rho = float("nan")              # nonlinear audit failed: reject and shrink
        rho_last = rho
        radius_before = radius
        radius, accept = adapt_radius(radius, rho, res.boundary_hit, actual, tr)
        took = False
        if accept and math.isfinite(e_new) and ((m_new < meas_cur) if legacy else (e_new < energy)):
            took = True
            # the accepted step is applied to the REAL state
            state, G = apply_correction(state, frame, step, transport=False)
            prior.accepted(step, G, legacy_transport=legacy)
            energy, lin = e_new, lin_new
            meas_cur = m_new
            accepted_count += 1
            if visibility is not None:
                vis_cur = vis_new
        tp = lin_new.energy_parts
        history.append({"iteration": it, "measurement_rgb": tp["rgb"], "measurement_alpha": tp["alpha"],
                        "measurement_depth": tp["depth"], "prior_energy": p_new, "total_energy": e_new,
                        "predicted_decrease": pred, "actual_decrease": actual, "rho": rho,
                        "accepted": took, "radius_before": radius_before, "radius": radius,
                        "depth_used": tp["depth_used"], "depth_gated_by_alpha": tp["depth_gated_by_alpha"],
                        "prior_mode": "exact_manifold" if prior.exact else "chart_offset"})
        if visibility is not None:
            vis_log.append({"iteration": it, "accepted": took,
                            **(vis_info or {}), "audit": vis_audit})
        _bar.note(f"rho {rho:+.3f} E {energy:.4e} "
                  f"{'accept' if took else 'reject'}")
        if cfg.verbose:
            print(f"      it {it}: rho {rho:+.3f} radius {radius:.3e} "
                  f"energy {energy:.6e} {'accept' if took else 'reject'}")
        if radius <= tr.radius_min:
            break

    _bar.close()
    vis_summary = None
    if visibility is not None:
        final = visibility.audit(vis_ref, vis_ref, vis_cur)
        persistent = None
        if final["violating_now"] == 0:
            status = "FEASIBLE"
        elif vis_last is not None:
            from .visibility_constraint import classify_persistent
            L_, A_, b_, lf_, lfT_, M_ = vis_last
            persistent = classify_persistent(visibility, L_, L_.visibility_values, vis_ref,
                                             visibility.row_values(L_.visibility_values),
                                             A_, b_, lf_, lfT_, dim_state=n, M_inv=M_)
            status = persistent["status"]
        else:
            status = "UNRESOLVED_INFEASIBLE"
        vis_summary = {"status": status, "final_audit": final, "coverage": visibility.coverage(),
                       "rays_violated_at_reference": int((vis_ref > visibility.policy.epsilon).sum()),
                       "persistent_violation_classification": persistent,
                       "constrained_steps": sum(1 for r in vis_log if r.get("status") not in (None, "INACTIVE")),
                       "rejected_by_audit": sum(1 for r in vis_log if r.get("audit") and not r["audit"]["ok"]),
                       "posterior_note": "unconstrained information inverse; truncated by active inequalities",
                       "iterations": vis_log}

    # ---- section 14: the posterior at the ACCEPTED state -----------------
    # measurement information by probing the operator at the final
    # linearization, plus the prior information pulled back to the tangent of
    # the accepted state (w J^T Lambda_minus J): the same tangent the next
    # frame's prediction propagates.
    meas_final = _meas_normal(lin, state, lin.frame_terms)
    H_info = torch.stack([meas_final(torch.eye(n, dtype=torch.float64)[:, k])
                          for k in range(n)], dim=1)
    H_info = 0.5 * (H_info + H_info.T)
    _, N_prior = prior.gradient_parts(state)
    info = N_prior + H_info
    info = 0.5 * (info + info.T)
    evals = torch.linalg.eigvalsh(info)
    if float(evals.min()) <= cfg.psd_floor:
        info = info + (cfg.psd_floor - float(evals.min())) * torch.eye(
            n, dtype=torch.float64)
    post = torch.linalg.inv(info)
    post = 0.5 * (post + post.T)

    p0 = state.parts[0]
    final_parts = {**lin.energy_parts, "prior": prior.energy(state), "total": energy}
    return FrameRecord(
        frame=frame, iterations=it, accepted=accepted_count,
        energy_before=meas0 if legacy else energy0, energy_after=meas_cur if legacy else energy,
        radius=radius,
        rho_last=rho_last, information=info, posterior=post,
        q=float(p0.q[frame]), qdot=float(p0.qdot[frame]), sign_flipped=False,
        observability={"measurement_information_eigenvalues":
                       torch.linalg.eigvalsh(H_info).tolist(),
                       "posterior_eigenvalues": evals.tolist(),
                       "geometry_information_eigenvalues":
                       torch.linalg.eigvalsh(
                           H_info[getattr(state, "geometry_slice",
                                          slice(2, 6)),
                                  getattr(state, "geometry_slice",
                                          slice(2, 6))]).tolist(),
                       "objective": {"initial": parts0, "final": final_parts, "acceptance": cfg.acceptance,
                                     "prior_mode": "exact_manifold" if prior.exact else "chart_offset"},
                       "iterations": history,
                       **({"visibility": vis_summary} if vis_summary is not None else {})})


def run_window(state, mu0: Tensor, R0, observations, linearize, precision_fn,
               times=None, cfg: JointIEKFConfig | None = None,
               label: str = "", visibility=None, observation_model=None) -> dict:
    """Sections 6, 13-16: the recursive filter over a whole window.

    Predict, update, reset, with the joint covariance carried from frame to
    frame. Section 5's cross-covariances are never dropped: the posterior is
    inverted and propagated as one dense matrix.
    """
    cfg = cfg or JointIEKFConfig()
    T = len(observations)
    t_arr = (torch.arange(T, dtype=torch.float64) if times is None
             else torch.as_tensor(times).double().reshape(-1))
    records = []
    reference = [p.zeta.clone() if hasattr(p, "zeta") else None
                 for p in state.parts]

    # `label` names which Phase-2 model is sweeping the window -- screw, P, R
    # or the P-constrained null -- because all four run the same code here and
    # a bar reading "window" four times over tells the operator nothing.
    from utils.progress import ibar
    wbar = ibar(T, f"{label} window" if label else "phase2 window", leave=True)
    for frame in range(T):
        wbar.to(frame)
        if frame == 0:
            # Frame 0 is the canonical gauge, not a measurement. The canonical
            # Gaussians ARE the frame-0 geometry, so q(0) = 0 by definition and
            # T(0) = I whatever the screw is -- which means frame 0 carries no
            # information about ANY coordinate: the geometry is unobservable
            # there too. Updating on it would spend the residual that belongs
            # to model error on a motion that did not happen, and would break
            # the s_0 = 0 gauge the backend's chart is built on.
            for p in state.parts:
                p.q[0] = 0.0
                p.qdot[0] = 0.0
            info0 = torch.linalg.inv(state.covariance)
            e0, lin0 = _energy(linearize, state, 0, mu0, R0, observations[0],
                               precision_fn, observation_model)
            records.append(FrameRecord(
                frame=0, iterations=0, accepted=0, energy_before=e0,
                energy_after=e0, radius=cfg.radius_init,
                rho_last=float("nan"), information=info0,
                posterior=state.covariance.clone(), q=0.0, qdot=0.0,
                sign_flipped=False,
                observability={"measurement_information_eigenvalues":
                               [0.0] * state.dim,
                               "posterior_eigenvalues":
                               torch.linalg.eigvalsh(state.covariance).tolist(),
                               "geometry_information_eigenvalues": [0.0] * 4,
                               "note": ("frame 0 is the canonical gauge: "
                                        "T(q=0) = I, so no coordinate is "
                                        "observable here")}))
            if cfg.verbose:
                print(f"   frame 0: canonical gauge, q = 0 pinned "
                      f"(energy {e0:.4e} is model error, not motion)")
            continue

        dt = float(t_arr[frame] - t_arr[frame - 1])
        if hasattr(state, "predict"):
            state.predict(frame - 1, dt, cfg.sigma_qddot)
        else:
            process_predict(state, frame - 1, dt, cfg.sigma_qddot)
        prior_info = torch.linalg.inv(state.covariance)
        prior_info = 0.5 * (prior_info + prior_info.T)

        rec = frame_step(state, frame, mu0, R0, observations[frame],
                         linearize, precision_fn, prior_info, cfg,
                         visibility=None if visibility is None else visibility.for_frame(frame),
                         observation_model=observation_model)

        # section 16: keep the representation continuous before storing
        flips = apply_sign_gauge(state, reference)
        if any(flips):
            S = sign_jacobian(state, flips)
            rec.posterior = S @ rec.posterior @ S.T
            rec.information = S @ rec.information @ S.T
            rec.sign_flipped = True
            rec.q, rec.qdot = -rec.q, -rec.qdot
        reference = [p.zeta.clone() if hasattr(p, "zeta") else None
                     for p in state.parts]

        state.covariance = rec.posterior
        records.append(rec)
        wbar.note(f"q {rec.q:+.5f} E {rec.energy_after:.4e} "
                  f"{rec.accepted}/{rec.iterations} acc")
        if cfg.verbose:
            print(f"   frame {frame}: q {rec.q:+.5f} qdot {rec.qdot:+.5f} "
                  f"energy {rec.energy_before:.4e} -> {rec.energy_after:.4e} "
                  f"({rec.accepted}/{rec.iterations} accepted)")
    wbar.to(T)
    wbar.close()
    return {"state": state, "records": records,
            "q": [r.q for r in records], "qdot": [r.qdot for r in records],
            "final_covariance": state.covariance,
            "geometry": state.parts[0].geometry()}
