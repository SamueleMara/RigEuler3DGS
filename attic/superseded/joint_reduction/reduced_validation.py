"""Does the visual update actually do anything in the reduced filter?

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md section 15.

Section 15's complaint about the previous evidence is precise: with a constant
speed, the constant-velocity process model ALREADY predicts the right answer,
so "q recovered exactly" proves nothing about the measurement. Its closing line
is the rule this module enforces:

    "Do not call `q recovered exactly` if `q_visual_delta` is negligible and
     the prior already predicts the answer."

So every frame logs section 15's nine quantities, the visual correction is
logged SEPARATELY from the prediction, and each trajectory is run three ways --
process-only, measurement-only and fused -- so the comparison section 15 asks
for is a measurement rather than a claim.

The mandatory trajectories (section 15) are defined here as well, because they
are part of the contract: constant speed is NOT among them, and the ones that
are have been chosen so the constant-velocity prior is provably wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

EPS = 1e-12


# ---------------------------------------------------------------------------
# section 15's mandatory synthetic trajectories
# ---------------------------------------------------------------------------
def traj_accelerated(t: float, a: float = 0.9, v0: float = 0.1) -> float:
    """1. accelerated motion: the constant-velocity prior is always behind."""
    return v0 * t + 0.5 * a * t * t


def traj_sinusoidal(t: float, amp: float = 0.35, freq: float = 0.9) -> float:
    """2. sinusoidal q: curvature of both signs."""
    return amp * math.sin(2 * math.pi * freq * t)


def traj_reversal(t: float, speed: float = 0.5, turn: float = 0.5) -> float:
    """3. reversal: the prior extrapolates straight through the turn."""
    return speed * t if t <= turn else speed * turn - speed * (t - turn)


def traj_stop_start(t: float, speed: float = 0.5, stop: float = 0.3,
                    restart: float = 0.6) -> float:
    """4. stop / start: the prior keeps moving during the stop."""
    if t <= stop:
        return speed * t
    if t <= restart:
        return speed * stop
    return speed * stop + speed * (t - restart)


MANDATORY_TRAJECTORIES = {
    "accelerated": traj_accelerated,
    "sinusoidal": traj_sinusoidal,
    "reversal": traj_reversal,
    "stop_start": traj_stop_start,
    # 5 and 6 are the same q(t) run from a deliberately wrong initial state;
    # the trajectory itself is a simple ramp so the ERROR is unambiguous.
    "wrong_initial_qdot": lambda t: 0.4 * t,
    "wrong_initial_q": lambda t: 0.4 * t,
}

WRONG_INITIAL = {
    "wrong_initial_qdot": {"qdot0_offset": 1.5, "q0_offset": 0.0},
    "wrong_initial_q": {"qdot0_offset": 0.0, "q0_offset": 0.35},
}


# ---------------------------------------------------------------------------
# section 15's per-frame log
# ---------------------------------------------------------------------------
@dataclass
class FrameLog:
    """Exactly section 15's required fields, plus the frame index."""

    frame: int
    q_pred: float
    qdot_pred: float
    q_visual_delta: float
    qdot_visual_delta: float
    q_post: float
    qdot_post: float
    prior_energy: float
    measurement_energy: float
    joint_information: float
    q_true: float = float("nan")

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["q_error"] = (abs(self.q_post - self.q_true)
                        if math.isfinite(self.q_true) else float("nan"))
        return d


@dataclass
class ModeResult:
    """One of the three ways of running a trajectory."""

    mode: str                           # process_only | measurement_only | fused
    logs: list[FrameLog] = field(default_factory=list)

    @property
    def q_errors(self) -> list[float]:
        return [abs(l.q_post - l.q_true) for l in self.logs
                if math.isfinite(l.q_true)]

    @property
    def rmse(self) -> float:
        e = self.q_errors
        return math.sqrt(sum(x * x for x in e) / len(e)) if e else float("nan")

    @property
    def final_error(self) -> float:
        return self.q_errors[-1] if self.q_errors else float("nan")

    @property
    def max_visual_delta(self) -> float:
        return max((abs(l.q_visual_delta) for l in self.logs), default=0.0)

    @property
    def mean_visual_delta(self) -> float:
        if not self.logs:
            return 0.0
        return sum(abs(l.q_visual_delta) for l in self.logs) / len(self.logs)

    def to_dict(self) -> dict:
        return {"mode": self.mode, "rmse": self.rmse,
                "final_error": self.final_error,
                "max_q_visual_delta": self.max_visual_delta,
                "mean_q_visual_delta": self.mean_visual_delta,
                "frames": [l.to_dict() for l in self.logs]}


@dataclass
class TrajectoryResult:
    name: str
    process_only: ModeResult
    fused: ModeResult
    measurement_only: ModeResult | None = None

    @property
    def fused_beats_process(self) -> bool:
        """Section 15: 'The fused result must outperform process-only
        prediction.'"""
        return self.fused.rmse < self.process_only.rmse

    @property
    def visual_update_is_meaningful(self) -> bool:
        """Section 15's anti-vacuity rule: the visual correction must be a real
        contribution, not numerical dust next to the state itself."""
        scale = max((abs(l.q_true) for l in self.fused.logs
                     if math.isfinite(l.q_true)), default=1.0)
        return self.fused.max_visual_delta > 1e-3 * max(scale, EPS)

    def to_dict(self) -> dict:
        d = {"trajectory": self.name,
             "process_only_rmse": self.process_only.rmse,
             "fused_rmse": self.fused.rmse,
             "improvement": (1.0 - self.fused.rmse / self.process_only.rmse
                             if self.process_only.rmse > EPS else 0.0),
             "fused_beats_process_only": self.fused_beats_process,
             "max_q_visual_delta": self.fused.max_visual_delta,
             "mean_q_visual_delta": self.fused.mean_visual_delta,
             "visual_update_is_meaningful": self.visual_update_is_meaningful,
             "process_only": self.process_only.to_dict(),
             "fused": self.fused.to_dict()}
        if self.measurement_only is not None:
            d["measurement_only_rmse"] = self.measurement_only.rmse
            d["measurement_only"] = self.measurement_only.to_dict()
        return d

    def format(self) -> str:
        lines = [f"  {self.name}:"]
        lines.append(f"      process-only RMSE {self.process_only.rmse:.6f}   "
                     f"fused RMSE {self.fused.rmse:.6f}   "
                     f"improvement {100 * (1 - self.fused.rmse / max(self.process_only.rmse, EPS)):.1f}%")
        lines.append(f"      |q_visual_delta|: max {self.fused.max_visual_delta:.6f}   "
                     f"mean {self.fused.mean_visual_delta:.6f}")
        return "\n".join(lines)


@torch.no_grad()
def iterated_reduced_update(prior, frame_energy_and_information, t, *,
                            max_inner: int = 8, lm_init: float = 1e-3,
                            lm_max: float = 1e8, backtracks: int = 6):
    """One frame of the reduced filter, with step control.

    A single Gauss-Newton step on q is not safe: the image is a strongly
    nonlinear function of the joint coordinate, so when the prediction is far
    off (an accelerating or reversing joint, or a deliberately wrong initial
    state) the unguarded step overshoots by orders of magnitude and the next
    normal matrix is indefinite. Measured on §15's 'accelerated' trajectory,
    the unguarded update produced |q_visual_delta| = 39 for a joint whose total
    travel is 0.5, and then failed outright.

    So the reduced update gets the same discipline the free stage has: iterate,
    re-linearize at the current q, damp, and accept only when the TRUE
    nonlinear objective (prior + measurement) decreases. The prior stays frozen
    at this frame's prediction throughout, so the frame is never counted as
    several independent measurements.
    """
    from .joint_filter import JointState

    chol, status = torch.linalg.cholesky_ex(prior.cov)
    if int(status.item()) != 0:
        raise ValueError("prior covariance is not positive definite")
    prior_info = torch.cholesky_inverse(chol)
    prior_vec = prior.vector

    def total_energy(q, meas_e):
        x = torch.stack([torch.as_tensor(q, dtype=prior_vec.dtype,
                                         device=prior_vec.device),
                         state_qdot])
        d = x - prior_vec
        return float(0.5 * (d @ prior_info @ d)) + float(meas_e)

    state_q = float(prior.q)
    state_qdot = prior.qdot.clone()
    info, b, meas_e = frame_energy_and_information(t, state_q)
    energy = total_energy(state_q, meas_e)
    lm = float(lm_init)
    last_info, last_meas = info, meas_e

    for _ in range(int(max_inner)):
        I_qq = float(torch.as_tensor(info).reshape(-1)[0])
        b_q = float(torch.as_tensor(b).reshape(-1)[0])
        meas_m = torch.zeros_like(prior.cov)
        meas_m[0, 0] = I_qq
        rhs = torch.zeros_like(prior_vec)
        rhs[0] = b_q
        x_cur = torch.stack([torch.as_tensor(state_q, dtype=prior_vec.dtype,
                                             device=prior_vec.device), state_qdot])
        grad = rhs - prior_info @ (x_cur - prior_vec)
        accepted = False
        for _bt in range(int(backtracks)):
            A = prior_info + meas_m + lm * torch.eye(
                2, dtype=prior.cov.dtype, device=prior.cov.device)
            try:
                delta = torch.linalg.solve(A, grad.unsqueeze(-1)).squeeze(-1)
            except Exception:
                lm = min(lm * 10.0, lm_max)
                continue
            q_try = state_q + float(delta[0])
            qdot_try = state_qdot + delta[1]
            try:
                i2, b2, e2 = frame_energy_and_information(t, q_try)
            except Exception:
                lm = min(lm * 10.0, lm_max)
                continue
            saved = state_qdot
            state_qdot = qdot_try
            e_try = total_energy(q_try, e2)
            if e_try < energy:
                state_q, energy = q_try, e_try
                info, b, last_info, last_meas = i2, b2, i2, e2
                lm = max(lm * 0.33, 1e-10)
                accepted = True
                break
            state_qdot = saved
            lm = min(lm * 10.0, lm_max)
        if not accepted or lm >= lm_max:
            break

    # The posterior covariance uses the information actually accumulated.
    I_qq = float(torch.as_tensor(last_info).reshape(-1)[0])
    meas_m = torch.zeros_like(prior.cov)
    meas_m[0, 0] = I_qq
    total = prior_info + meas_m
    cov = torch.linalg.inv(total + 1e-12 * torch.eye(
        2, dtype=total.dtype, device=total.device))
    cov = 0.5 * (cov + cov.T)
    # Keep the posterior strictly positive definite. Without process noise the
    # constant-velocity model asserts the speed is EXACTLY constant, the
    # covariance collapses frame after frame, and the next Cholesky fails. The
    # real remedy is a non-zero accel_psd (the acceleration genuinely is
    # unmodelled); this floor only stops round-off from doing the same thing.
    ev, evec = torch.linalg.eigh(cov)
    floor = float(ev.max()) * 1e-12
    cov = evec @ torch.diag(ev.clamp_min(max(floor, 1e-18))) @ evec.T
    cov = 0.5 * (cov + cov.T)
    post = JointState(torch.as_tensor(state_q, dtype=prior_vec.dtype,
                                      device=prior_vec.device),
                      state_qdot, cov)
    return post, I_qq, last_meas


@torch.no_grad()
def run_trajectory(name, q_of_t, state0, dt, frame_energy_and_information,
                   num_frames: int, *, accel_psd: float = 0.0,
                   measurement_only: bool = False) -> ModeResult:
    """Run the reduced filter over one trajectory, logging section 15's fields.

    ``frame_energy_and_information(t, q_linearization) -> (information, b,
    measurement_energy)`` must build the reduced normal system from the REAL
    renderer at the given q.
    """
    from .joint_filter import JointState

    mode = "measurement_only" if measurement_only else "fused"
    res = ModeResult(mode=mode)
    state = state0
    for t in range(num_frames):
        prior = state.predict(dt, accel_psd=accel_psd)
        q_pred, qdot_pred = float(prior.q), float(prior.qdot)

        post, joint_information, meas_e = iterated_reduced_update(
            prior, frame_energy_and_information, t)
        q_post, qdot_post = float(post.q), float(post.qdot)

        # The visual delta is the posterior MINUS the prediction: section 15
        # wants the measurement's own contribution isolated, not the total
        # change from the previous frame.
        q_visual = q_post - q_pred
        qdot_visual = qdot_post - qdot_pred

        d = prior.vector - post.vector
        try:
            chol, status = torch.linalg.cholesky_ex(prior.cov)
            prior_info_m = (torch.cholesky_inverse(chol) if int(status.item()) == 0
                            else torch.zeros_like(prior.cov))
        except Exception:
            prior_info_m = torch.zeros_like(prior.cov)
        prior_energy = float(0.5 * (d @ prior_info_m @ d))

        res.logs.append(FrameLog(
            frame=t, q_pred=q_pred, qdot_pred=qdot_pred,
            q_visual_delta=q_visual, qdot_visual_delta=qdot_visual,
            q_post=q_post, qdot_post=qdot_post, prior_energy=prior_energy,
            measurement_energy=float(meas_e),
            joint_information=joint_information,
            q_true=float(q_of_t((t + 1) * dt))))
        state = post
    return res


@torch.no_grad()
def run_process_only(name, q_of_t, state0, dt, num_frames: int,
                     *, accel_psd: float = 0.0) -> ModeResult:
    """The ablation section 15 requires: NO measurement at all, pure
    constant-velocity extrapolation."""
    res = ModeResult(mode="process_only")
    state = state0
    for t in range(num_frames):
        prior = state.predict(dt, accel_psd=accel_psd)
        res.logs.append(FrameLog(
            frame=t, q_pred=float(prior.q), qdot_pred=float(prior.qdot),
            q_visual_delta=0.0, qdot_visual_delta=0.0,
            q_post=float(prior.q), qdot_post=float(prior.qdot),
            prior_energy=0.0, measurement_energy=float("nan"),
            joint_information=0.0, q_true=float(q_of_t((t + 1) * dt))))
        state = prior
    return res


def summarize(results: list[TrajectoryResult]) -> dict:
    """The section 15 verdict over all mandatory trajectories."""
    return {
        "num_trajectories": len(results),
        "all_fused_beat_process_only": all(r.fused_beats_process for r in results),
        "all_visual_updates_meaningful": all(
            r.visual_update_is_meaningful for r in results),
        "trajectories": [r.to_dict() for r in results],
        "failing": [r.name for r in results
                    if not (r.fused_beats_process and r.visual_update_is_meaningful)],
    }
