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


def _energy(linearize, state, frame, mu0, R0, observation, precision_fn):
    """The true nonlinear MAP energy at the current nominal, via Phase 1."""
    lin = linearize(state, frame)
    tot = 0.0
    for v, ob0, prec in zip(lin.views, observation, precision_fn(lin)):
        ob = _to_view_device(ob0, v)
        if ob is None:
            continue
        rgb_r = (ob[0] - v.rgb).double()
        d_r = (ob[1] - v.depth).double()
        pc, pd = prec
        tot += float((pc.double() * rgb_r * rgb_r).sum())
        tot += float((pd.double() * d_r * d_r).sum())
    return 0.5 * tot, lin


def frame_step(state, frame: int, mu0: Tensor, R0, observation, linearize,
               precision_fn, prior_information: Tensor,
               cfg: JointIEKFConfig | None = None) -> FrameRecord:
    """Section 13: one Phase-2 frame, with Phase-1's solver and acceptance.

    `linearize(state, frame) -> FrameLinearization` and
    `precision_fn(lin) -> [(p_rgb, p_depth)]` are the caller's bindings to the
    EXISTING Phase-1 renderer cache and measurement precision; this function
    never builds either itself.

    `prior_information` is the predicted information at this frame. The prior
    is frozen for all nonlinear iterations (section 13 step 2) and the
    correction is measured from it, so relinearizing does not quietly reuse
    the prior more than once.
    """
    cfg = cfg or JointIEKFConfig()
    dev = mu0.device
    n = state.dim
    metric = _metric(prior_information, cfg)
    tr = TrustRegionConfig(radius_init=cfg.radius_init,
                           max_pcg_iters=cfg.max_cg_iters)
    radius = cfg.radius_init
    delta_from_prior = torch.zeros(n, dtype=torch.float64)

    energy, lin = _energy(linearize, state, frame, mu0, R0, observation,
                          precision_fn)
    energy0 = energy
    accepted_count = 0
    rho_last = float("nan")
    it = 0

    # Each iteration re-renders the whole scene, so at full scale this is
    # seconds per step. It also breaks early, hence absolute positioning.
    from utils.progress import ibar
    _bar = ibar(int(cfg.max_iters), f"solve f{frame:03d}")
    for it in range(1, int(cfg.max_iters) + 1):
        _bar.to(it - 1)
        prec = precision_fn(lin)
        # ---- measurement right-hand side: J^T H^T W r --------------------
        weighted = []
        for v, ob0, (pc, pd) in zip(lin.views, observation, prec):
            ob = _to_view_device(ob0, v)
            if ob is None:
                weighted.append((torch.zeros_like(v.rgb),
                                 torch.zeros_like(v.depth)))
                continue
            weighted.append((pc * (ob[0] - v.rgb), pd * (ob[1] - v.depth)))
        rhs_meas = joint_apply_HT(lin, state, frame, weighted, mu0, R0).cpu()
        b = rhs_meas - cfg.prior_weight * (prior_information
                                           @ delta_from_prior)

        # ---- matrix-free normal operator (section 12) --------------------
        def A_mv(p: Tensor) -> Tensor:
            tang = joint_apply_H(lin, state, frame, p.to(dev), mu0, R0)
            w = [(pc * dc, pd * dd)
                 for (pc, pd), (dc, dd) in zip(prec, tang)]
            meas = joint_apply_HT(lin, state, frame, w, mu0, R0).cpu()
            return meas + cfg.prior_weight * (prior_information @ p)

        # Section 14: for K=1 the operator is 6x6, so probing it with the
        # basis vectors gives it EXACTLY -- no pixel Jacobian is built, just
        # six applications of the matrix-free operator. Its inverse is then
        # used as the CG preconditioner.
        #
        # This is not an optimisation. Steihaug-CG starts along `b`, whose
        # qdot component is exactly zero (section 7: the image has no qdot
        # derivative), and the coupling that should carry information into
        # qdot only appears after the residual has been multiplied by A. With
        # the image information on q four orders of magnitude larger than the
        # prior, CG met its forcing tolerance in one or two iterations and
        # qdot never moved: measured on real data it stayed at 7e-06 while q
        # advanced 0.018 per frame, even though the posterior variance of qdot
        # was correctly falling. Preconditioning by the exact inverse makes
        # the first direction already the right one.
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
            break

        # ---- predicted reduction from the SAME quadratic model ----------
        pred = float((b * step).sum() - 0.5 * (step * A_mv(step)).sum())

        # ---- trial retraction, rerender, Phase-1 acceptance -------------
        import copy
        trial = copy.deepcopy(state)
        from .joint_process import apply_correction
        trial, _G = apply_correction(trial, frame, step, transport=False)
        e_new, lin_new = _energy(linearize, trial, frame, mu0, R0, observation,
                                 precision_fn)
        actual = energy - e_new
        rho = actual / pred if pred > 0 else float("-inf")
        rho_last = rho
        radius, accept = adapt_radius(radius, rho, res.boundary_hit, actual, tr)
        if accept and math.isfinite(e_new) and e_new < energy:
            # the accepted step is applied to the REAL state, with the
            # covariance carried across the basis change (section 15)
            state, G = apply_correction(state, frame, step, transport=False)
            delta_from_prior = G @ (delta_from_prior + step)
            prior_information = (torch.linalg.solve(
                G, torch.linalg.solve(G, prior_information).T).T
                if abs(float(torch.det(G))) > 1e-12 else prior_information)
            energy, lin = e_new, lin_new
            accepted_count += 1
        _bar.note(f"rho {rho:+.3f} E {energy:.4e} "
                  f"{'accept' if accept else 'reject'}")
        if cfg.verbose:
            print(f"      it {it}: rho {rho:+.3f} radius {radius:.3e} "
                  f"energy {energy:.6e} {'accept' if accept else 'reject'}")
        if radius <= tr.radius_min:
            break

    _bar.close()

    # ---- section 14: the exact 6x6 posterior, by probing the operator ----
    prec = precision_fn(lin)

    def meas_only(p: Tensor) -> Tensor:
        tang = joint_apply_H(lin, state, frame, p.to(dev), mu0, R0)
        w = [(pc * dc, pd * dd) for (pc, pd), (dc, dd) in zip(prec, tang)]
        return joint_apply_HT(lin, state, frame, w, mu0, R0).cpu()

    # The state is tiny, so applying the matrix-free operator to the six basis
    # vectors gives the EXACT measurement information -- no pixel Jacobian is
    # ever materialised (section 14).
    H_info = torch.stack([meas_only(torch.eye(n, dtype=torch.float64)[:, k])
                          for k in range(n)], dim=1)
    H_info = 0.5 * (H_info + H_info.T)
    info = prior_information + H_info
    info = 0.5 * (info + info.T)
    evals = torch.linalg.eigvalsh(info)
    if float(evals.min()) <= cfg.psd_floor:
        info = info + (cfg.psd_floor - float(evals.min())) * torch.eye(
            n, dtype=torch.float64)
    post = torch.linalg.inv(info)
    post = 0.5 * (post + post.T)

    p0 = state.parts[0]
    return FrameRecord(
        frame=frame, iterations=it, accepted=accepted_count,
        energy_before=energy0, energy_after=energy, radius=radius,
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
                                          slice(2, 6))]).tolist()})


def run_window(state, mu0: Tensor, R0, observations, linearize, precision_fn,
               times=None, cfg: JointIEKFConfig | None = None,
               label: str = "") -> dict:
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
                               precision_fn)
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
                         linearize, precision_fn, prior_info, cfg)

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
