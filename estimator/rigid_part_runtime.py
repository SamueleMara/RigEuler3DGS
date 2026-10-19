"""Sections 9, 13-16, 18: the rigid SE(3) Phase-2 runtime.

CODEX_IMPLEMENT_RIGID_SE3_PHASE2_AND_POSTHOC_JOINT_REDUCTION.md.

This is wiring, not a new estimator (section 13). The renderer cache, the
RGB-D residual, `R_eff = R_sensor + R_model`, the trust region, the
acceptance test and the Steihaug-CG solve are Phase 1's own and are reached
through the SAME `frame_step` the screw runtime uses. The only new component
is the exact rigid state-to-Gaussian map in `rigid_part_se3`.

Section 16's initializer is weighted Kabsch on the Phase-1 free trajectories
restricted to the frozen coherence support. Section 17 forbids the obvious
alternative: no axis is fitted first and no joint model is used to seed the
pose. There is no joint geometry at this point in the pipeline, by design.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .joint_iekf import FrameRecord, JointIEKFConfig, _energy, frame_step
from .joint_manifold import se3_adjoint, se3_exp, se3_left_jacobian, se3_log
from .manifold import project_to_so3
from .rigid_part_se3 import (RIGID_ERROR_DIM, TAU, XI, RigidPart, rigid_jvp,
                             rigid_vjp, state_to_gaussians)

# Section 8: frame 0 is the canonical gauge, not an estimate.
CANONICAL_FRAME = 0


@dataclass
class RigidIEKFConfig:
    """Section 9. `sigma_vdot` and `sigma_wdot` are separate because a linear
    and an angular acceleration do not share units; one scalar for both would
    make the process prior depend on whether the scene is measured in metres.
    """
    max_iters: int = 8
    max_cg_iters: int = 40
    sigma_vdot: float = 1.0
    sigma_wdot: float = 1.0
    radius_init: float = 1.0
    prior_weight: float = 1.0
    psd_floor: float = 1e-12
    verbose: bool = False
    # The initial pose/velocity uncertainty at the canonical frame.
    sigma_tau0: float = 1e-2
    sigma_xi0: float = 1.0

    def to_joint(self) -> JointIEKFConfig:
        """The shared solver's config. Only solver knobs cross over."""
        return JointIEKFConfig(
            max_iters=self.max_iters, max_cg_iters=self.max_cg_iters,
            radius_init=self.radius_init, prior_weight=self.prior_weight,
            psd_floor=self.psd_floor, verbose=self.verbose)


class RigidState:
    """The shared loop's view of one rigid part.

    It exposes exactly what `frame_step` reads -- `dim`, `num_parts`, `parts`,
    `covariance`, `jvp`, `vjp`, `geometry_slice`, `retract_in_place` -- and
    `gaussians` for the debug/energy paths.

    `geometry_slice` is the POSE block here. For the screw it selected the
    four static joint-geometry coordinates; this model has no static geometry,
    so the quantity worth reporting as "what did the image constrain" is the
    six-dimensional pose. The velocity block is deliberately excluded: its
    measurement information is exactly zero (section 11) and reporting its
    eigenvalues alongside the pose's would suggest otherwise.
    """

    def __init__(self, part: RigidPart, cfg: RigidIEKFConfig | None = None):
        cfg = cfg or RigidIEKFConfig()
        self.parts = [part]
        self.error_dim = RIGID_ERROR_DIM
        self.times = torch.arange(part.num_frames, dtype=torch.float64)
        c = torch.cat([torch.full((6,), cfg.sigma_tau0 ** 2,
                                  dtype=torch.float64),
                       torch.full((6,), cfg.sigma_xi0 ** 2,
                                  dtype=torch.float64)])
        self.covariance = torch.diag(c)
        self._cfg = cfg

    # -- shape ------------------------------------------------------------
    @property
    def num_parts(self) -> int:
        return 1

    @property
    def dim(self) -> int:
        return RIGID_ERROR_DIM

    @property
    def geometry_slice(self) -> slice:
        return TAU

    # -- the exact rigid map ----------------------------------------------
    def jvp(self, frame, delta, mu0, R0=None):
        part = self.parts[0]
        mu_now, _ = state_to_gaussians(part, frame, mu0, R0)
        idx = part.members.to(mu_now.device)
        full = torch.zeros((mu0.shape[0], 6), dtype=torch.float64,
                           device=mu_now.device)
        full[idx] = rigid_jvp(part, frame, delta, mu_now[idx])
        return full

    def vjp(self, frame, pose_adjoint, mu0, R0=None):
        part = self.parts[0]
        mu_now, _ = state_to_gaussians(part, frame, mu0, R0)
        idx = part.members.to(mu_now.device)
        a = torch.as_tensor(pose_adjoint).double().reshape(-1, 6)
        return rigid_vjp(part, frame, a[idx], mu_now[idx]).cpu()

    def gaussians(self, frame, mu0, R0=None):
        return state_to_gaussians(self.parts[0], frame, mu0, R0)

    # -- section 9: the process model -------------------------------------
    def process_matrix(self, frame: int, dt: float) -> Tensor:
        """F for `T_{t+1} = Exp(dt xi^) T_t`, `xi_{t+1} = xi_t`.

            delta_tau_{t+1} = Ad_E delta_tau_t + dt J_l(dt xi) delta_xi_t
            delta_xi_{t+1}  = delta_xi_t

        with `E = Exp(dt xi^)`. The off-diagonal block is the ONLY route by
        which image evidence about the pose reaches the velocity, because a
        single image has no velocity derivative at all (section 11). At
        `xi = 0` it degenerates to the familiar `[[I, dt I], [0, I]]`.
        """
        p = self.parts[0]
        xi = p.twist(frame) * float(dt)
        E_R, E_t = se3_exp(xi.reshape(1, 6))
        F = torch.eye(RIGID_ERROR_DIM, dtype=torch.float64)
        F[TAU, TAU] = se3_adjoint(E_R, E_t)[0]
        F[TAU, XI] = float(dt) * se3_left_jacobian(xi.reshape(1, 6))[0]
        return F

    def process_noise(self, dt: float) -> Tensor:
        """Q: white acceleration on the twist, integrated once.

        The pose block is not zero -- unlike the screw's static geometry, the
        pose is driven by the velocity, so an unmodelled acceleration reaches
        it through `dt^3/3`.
        """
        dt = float(dt)
        Q = torch.zeros((RIGID_ERROR_DIM, RIGID_ERROR_DIM),
                        dtype=torch.float64)
        s2 = torch.tensor([self._cfg.sigma_vdot ** 2] * 3
                          + [self._cfg.sigma_wdot ** 2] * 3,
                          dtype=torch.float64)
        Q[TAU, TAU] = torch.diag(s2 * dt ** 3 / 3.0)
        Q[TAU, XI] = torch.diag(s2 * dt ** 2 / 2.0)
        Q[XI, TAU] = torch.diag(s2 * dt ** 2 / 2.0)
        Q[XI, XI] = torch.diag(s2 * dt)
        return Q

    def predict(self, frame: int, dt: float, sigma_qddot: float = 1.0):
        """Advance the nominal and the covariance by one frame (section 9)."""
        p = self.parts[0]
        F = self.process_matrix(frame, dt)
        Q = self.process_noise(dt)
        if frame + 1 < p.num_frames:
            E_R, E_t = se3_exp((p.twist(frame) * float(dt)).reshape(1, 6))
            p.R[frame + 1] = project_to_so3(
                (E_R[0] @ p.R[frame]).reshape(1, 3, 3))[0]
            p.p[frame + 1] = E_R[0] @ p.p[frame] + E_t[0]
            p.v[frame + 1] = p.v[frame].clone()
            p.omega[frame + 1] = p.omega[frame].clone()
        self.covariance = F @ self.covariance @ F.T + Q
        return self

    def retract_in_place(self, delta, frame):
        self.parts[0] = self.parts[0].retract(delta, frame)


# ---------------------------------------------------------------------------
# Section 16: initialization from the Phase-1 coherent trajectories
# ---------------------------------------------------------------------------

def weighted_kabsch(X0: Tensor, Xt: Tensor, w: Tensor | None = None):
    """`argmin_{R,p} sum_i w_i ||x_i(t) - (R x_i(0) + p)||^2`.

    The reflection guard is not optional: without `det(U V^T)` folded in, a
    nearly planar support (a drawer front, a blade face) produces a mirror
    whenever the smallest singular value is at noise level, and the resulting
    "rotation" has determinant -1.
    """
    X0 = X0.double()
    Xt = Xt.double()
    n = X0.shape[0]
    w = (torch.ones(n, dtype=torch.float64) if w is None
         else torch.as_tensor(w).double().reshape(-1))
    W = w.sum().clamp_min(1e-300)
    c0 = (w[:, None] * X0).sum(0) / W
    ct = (w[:, None] * Xt).sum(0) / W
    A = X0 - c0
    B = Xt - ct
    M = (w[:, None] * B).T @ A
    U, S, Vh = torch.linalg.svd(M)
    D = torch.diag(torch.tensor([1.0, 1.0, float(torch.det(U @ Vh))],
                                dtype=torch.float64))
    R = U @ D @ Vh
    p = ct - R @ c0
    return R, p, S


def kabsch_initialize(members: Tensor, mu0: Tensor, positions,
                      times=None, weights=None) -> dict:
    """Section 16: a rigid pose per frame, then the twist that connects them.

    WEIGHTING POLICY, stated once. Every member of the frozen coherence
    support is weighted equally. The support is already precision-first
    (section 1) -- it was built to be pure rather than complete -- so a second
    reweighting here would be tuning on top of a selector designed not to
    need it. `weights` exists for the ablation in RSE3-9 and is not used by
    the production path.

    Frame 0 is forced to the identity gauge (section 8) rather than taken from
    Kabsch, so the gauge holds exactly even if the Phase-1 frame-0 centers
    were ever written back with rounding.
    """
    members = torch.as_tensor(members).long().reshape(-1).cpu()
    X0 = torch.as_tensor(positions[0]).double().cpu()[members]
    T = len(positions)
    t_arr = (torch.arange(T, dtype=torch.float64) if times is None
             else torch.as_tensor(times).double().reshape(-1))
    R = torch.eye(3, dtype=torch.float64).expand(T, 3, 3).clone()
    p = torch.zeros((T, 3), dtype=torch.float64)
    sing = []
    for t in range(T):
        if t == CANONICAL_FRAME:
            sing.append([float("nan")] * 3)
            continue
        Xt = torch.as_tensor(positions[t]).double().cpu()[members]
        Rt, pt, S = weighted_kabsch(X0, Xt, weights)
        R[t] = Rt
        p[t] = pt
        sing.append(S.tolist())

    # xi_t = (1/dt) Log(T_{t+1} T_t^{-1}) -- the SPATIAL twist, matching the
    # process model's left composition and section 25's increment convention.
    v = torch.zeros((T, 3), dtype=torch.float64)
    om = torch.zeros((T, 3), dtype=torch.float64)
    for t in range(T - 1):
        dt = float(t_arr[t + 1] - t_arr[t])
        if abs(dt) < 1e-300:
            continue
        dR = R[t + 1] @ R[t].T
        dp = p[t + 1] - dR @ p[t]
        eta = se3_log(dR.reshape(1, 3, 3), dp.reshape(1, 3))[0] / dt
        v[t] = eta[:3]
        om[t] = eta[3:]
    if T >= 2:                      # the last frame has no successor
        v[T - 1] = v[T - 2]
        om[T - 1] = om[T - 2]

    part = RigidPart(members, R, p, v, om)
    resid = []
    for t in range(T):
        Xt = torch.as_tensor(positions[t]).double().cpu()[members]
        resid.append(float((Xt - (X0 @ R[t].T + p[t])).norm(dim=1).mean()))

    # HOW MUCH OF THE OBSERVED MOTION ONE RIGID BODY DOES NOT EXPLAIN.
    #
    # The residual alone is a length and means nothing on its own. The right
    # denominator is the DISPLACEMENT, not the part size: measured across
    # seven scenes the per-frame residual tracks how far the part has
    # travelled (correlation 0.84 to 0.99) and not how big it is. Dividing by
    # the part scale made blade read 133% purely because its part is small,
    # while its one-DOF statistic is a clean 0.993.
    #
    # WHAT IT DOES NOT DO, stated because the obvious guess is wrong. It is
    # not a body count and it does not single out the scissor. Measured:
    # stapler 16, blade 22, USB 24, SCISSOR 24, fridge 26, laptop 33,
    # storage 44 percent. The scissor -- two blades turning oppositely about
    # one pivot, the one scene that genuinely is not a single body -- sits in
    # the middle, because Phase 1 recovers each blade well and the single
    # Kabsch fit splits the difference. gamma_1DOF is what finds it: 0.59,
    # with the first two singular values at 0.59 / 0.37, against 0.92 or
    # better everywhere else.
    #
    # So read this as a quality number for the support and its Phase-1
    # trajectories -- how much motion no rigid model accounts for -- and read
    # gamma for whether the motion that IS rigid has one degree of freedom.
    #
    # The denominator is the mean per-point DISPLACEMENT, not |p_t|. Two
    # halves rotating oppositely about a shared pivot cancel in the centre of
    # mass, so |p_t| collapses toward zero exactly in the case this is meant
    # to catch and the ratio diverges -- 9.2 on a synthetic scissor. The mean
    # per-point displacement does not collapse: every point still moves.
    c0 = X0.mean(0)
    L = float(((X0 - c0) ** 2).sum(1).mean().sqrt())
    travel = []
    for t in range(T):
        Xt = torch.as_tensor(positions[t]).double().cpu()[members]
        travel.append(float((Xt - X0).norm(dim=1).mean()))
    unexplained = [float(r / d) if d > 1e-9 else float("nan")
                   for r, d in zip(resid, travel)]
    finite = [u for u in unexplained if u == u]
    return {"part": part, "kabsch_singular_values": sing,
            "mean_residual_per_frame": resid,
            "max_mean_residual": max(resid) if resid else 0.0,
            "part_scale_L": L,
            "displacement_per_frame": travel,
            "unexplained_fraction_per_frame": unexplained,
            "unexplained_fraction_final": (finite[-1] if finite else float("nan")),
            "max_unexplained_fraction": (max(finite) if finite else float("nan")),
            "weighting": "uniform over the frozen coherence support",
            "initialization_source": "phase1_kabsch",
            "used_axis_or_joint_model": False,
            "unexplained_fraction_note": (
                "mean Kabsch residual / mean per-point displacement: the "
                "fraction of the observed motion that the best single rigid "
                "unexplained. It rises both when the support spans several "
                "bodies and when Phase 1 is noisy over it, so read it with "
                "gamma_1DOF rather than alone. Never thresholded."),
            "note": ("section 17: no screw, prismatic or revolute model was "
                     "fitted to produce this initialization")}


# ---------------------------------------------------------------------------
# Sections 15, 18: the window
# ---------------------------------------------------------------------------

def run_rigid_window(state: RigidState, mu0: Tensor, R0, observations,
                     linearize, precision_fn, times=None,
                     cfg: RigidIEKFConfig | None = None,
                     label: str = "rigid", visibility=None) -> dict:
    """Predict / update / reset over the whole window.

    Frame 0 is the canonical gauge and carries no information about anything:
    `T_0 = I` whatever the motion is, so updating on it would spend the
    residual that belongs to model error on a motion that did not happen.
    """
    cfg = cfg or RigidIEKFConfig()
    jcfg = cfg.to_joint()
    T = len(observations)
    t_arr = (torch.arange(T, dtype=torch.float64) if times is None
             else torch.as_tensor(times).double().reshape(-1))
    part = state.parts[0]
    records = []

    from utils.progress import ibar
    wbar = ibar(T, f"{label} window", leave=True)

    for frame in range(T):
        wbar.to(frame)
        if frame == CANONICAL_FRAME:
            part.R[frame] = torch.eye(3, dtype=torch.float64)
            part.p[frame] = torch.zeros(3, dtype=torch.float64)
            info0 = torch.linalg.inv(state.covariance)
            e0, _lin0 = _energy(linearize, state, frame, mu0, R0,
                                observations[frame], precision_fn)
            records.append(FrameRecord(
                frame=frame, iterations=0, accepted=0, energy_before=e0,
                energy_after=e0, radius=cfg.radius_init,
                rho_last=float("nan"), information=info0,
                posterior=state.covariance.clone(), q=0.0, qdot=0.0,
                sign_flipped=False,
                observability={
                    "measurement_information_eigenvalues": [0.0] * state.dim,
                    "posterior_eigenvalues":
                        torch.linalg.eigvalsh(state.covariance).tolist(),
                    "geometry_information_eigenvalues": [0.0] * 6,
                    "note": ("section 8: frame 0 is the canonical gauge, "
                             "T_0 = I, so no pose coordinate is observable "
                             "here")}))
            if cfg.verbose:
                print(f"   frame 0: canonical gauge, T_0 = I pinned "
                      f"(energy {e0:.4e} is model error, not motion)")
            continue

        dt = float(t_arr[frame] - t_arr[frame - 1])
        state.predict(frame - 1, dt)
        prior_info = torch.linalg.inv(state.covariance)
        prior_info = 0.5 * (prior_info + prior_info.T)

        rec = frame_step(state, frame, mu0, R0, observations[frame],
                         linearize, precision_fn, prior_info, jcfg,
                         visibility=None if visibility is None else visibility.for_frame(frame))
        state.covariance = rec.posterior
        part = state.parts[0]
        records.append(rec)
        ang = float(torch.linalg.norm(
            torch.as_tensor(_rotvec(part.R[frame]))))
        wbar.note(f"|p| {float(part.p[frame].norm()):.4f} "
                  f"ang {ang:.4f} E {rec.energy_after:.3e}")
        if cfg.verbose:
            print(f"   frame {frame}: |p| {float(part.p[frame].norm()):.5f} "
                  f"angle {ang:.5f} energy {rec.energy_before:.4e} -> "
                  f"{rec.energy_after:.4e} "
                  f"({rec.accepted}/{rec.iterations} accepted)")
    wbar.to(T)
    wbar.close()

    part = state.parts[0]
    return {"state": state, "records": records, "part": part,
            "poses_R": part.R.clone(), "poses_p": part.p.clone(),
            "velocities_v": part.v.clone(), "velocities_omega": part.omega.clone(),
            "pose_covariances": torch.stack(
                [r.posterior[TAU, TAU] for r in records]),
            "covariances": torch.stack([r.posterior for r in records]),
            "final_covariance": state.covariance,
            "geometry": part.geometry()}


def _rotvec(R: Tensor) -> Tensor:
    from .manifold import so3_log
    return so3_log(torch.as_tensor(R).double().reshape(1, 3, 3))[0]
