"""Stage B (sections 27-33): run M_P and M_R through the SAME IEKF.

CODEX_VALIDATE_GENERIC_TWIST_THEN_PARALLEL_PRISMATIC_REVOLUTE_PHASE2.md.

Section 28 is the requirement this file exists to satisfy: the two physical
models do not get their own optimizers.  They get their own *state-to-Gaussian
map*, and everything downstream of it -- the renderer cache, the RGB-D
residual, the robust measurement precision, the truncated-CG solve, the
trust-region acceptance, the information-form update -- is the Phase-1 code
reached through `run_window`.  No Adam, no external optimizer, no free-SE(3)
intermediate.

Section 27 is the other requirement: ONE innovation updates the trajectory and
the geometry together.  That falls out of the state layout -- `[s, sdot, gamma]`
and `[theta, thetadot, eta, rho]` are single vectors with one dense covariance,
so the cross-covariance between displacement and geometry is carried, never
dropped, and there is no stage in which one is fitted with the other held
fixed.

Section 33: nothing here classifies.  It produces E_P and E_R, their splits,
and the held-out fold differences, and stops.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor

from . import phase2_prismatic as mp
from . import phase2_revolute as mr
from .joint_iekf import JointIEKFConfig, run_window
from .joint_process import process_matrix, process_noise


class _Phase2ModelState:
    """A K=1 state that `run_window` can drive, for either physical model.

    It exposes exactly what the shared loop asks for -- `dim`, `num_parts`,
    `parts`, `covariance`, `jvp`, `vjp`, `predict`, `geometry_slice` -- and
    nothing else.  `q` / `qdot` are aliases onto the model's own displacement
    and velocity so the shared `FrameRecord` stays meaningful.
    """

    def __init__(self, part, error_dim: int, jvp, vjp, to_gauss,
                 prior_scale=None):
        self.parts = [part]
        self.error_dim = int(error_dim)
        self._jvp, self._vjp, self._to_gauss = jvp, vjp, to_gauss
        self.times = torch.arange(part.num_frames, dtype=torch.float64)
        c = torch.ones(self.error_dim, dtype=torch.float64)
        if prior_scale is not None:
            c = torch.as_tensor(prior_scale).double().reshape(-1)
        self.covariance = torch.diag(c)

    # -- shape ------------------------------------------------------------
    @property
    def num_parts(self) -> int:
        return 1

    @property
    def dim(self) -> int:
        return self.error_dim

    @property
    def geometry_slice(self) -> slice:
        return slice(2, self.error_dim)

    # -- the model-specific map ------------------------------------------
    def jvp(self, frame, delta, mu0, R0=None):
        part = self.parts[0]
        mu_now, _ = self._to_gauss(part, frame, mu0, R0)
        idx = part.members.to(mu_now.device)
        full = torch.zeros((mu0.shape[0], 6), dtype=torch.float64,
                           device=mu_now.device)
        full[idx] = self._jvp(part, frame, delta, mu_now[idx])
        return full

    def vjp(self, frame, pose_adjoint, mu0, R0=None):
        part = self.parts[0]
        mu_now, _ = self._to_gauss(part, frame, mu0, R0)
        idx = part.members.to(mu_now.device)
        a = torch.as_tensor(pose_adjoint).double().reshape(-1, 6)
        return self._vjp(part, frame, a[idx], mu_now[idx]).cpu()

    def gaussians(self, frame, mu0, R0=None):
        return self._to_gauss(self.parts[0], frame, mu0, R0)

    # -- process (sections 19, 24) ----------------------------------------
    def predict(self, frame: int, dt: float, sigma_qddot: float = 1.0):
        """Identical to the generic process model: the displacement/velocity
        pair random-walks in acceleration and the geometry is exactly static."""
        F = process_matrix(self, dt)
        Q = process_noise(self, dt, sigma_qddot)
        p = self.parts[0]
        if frame + 1 < p.num_frames:
            p.q[frame + 1] = p.q[frame] + float(dt) * p.qdot[frame]
            p.qdot[frame + 1] = p.qdot[frame]
        self.covariance = F @ self.covariance @ F.T + Q

    def retract_in_place(self, delta, frame):
        self.parts[0] = self.parts[0].retract(delta, frame)


def _alias(part, disp: str, vel: str):
    """Give a physical part the `q`/`qdot` names the shared loop reads."""
    cls = type(part)
    if not hasattr(cls, "q"):
        setattr(cls, "q", property(lambda s, a=disp: getattr(s, a),
                                   lambda s, v, a=disp: setattr(s, a, v)))
        setattr(cls, "qdot", property(lambda s, a=vel: getattr(s, a),
                                      lambda s, v, a=vel: setattr(s, a, v)))
    return part


_alias(mp.PrismaticPart.__new__(mp.PrismaticPart), "s", "sdot")
_alias(mr.RevolutePart.__new__(mr.RevolutePart), "theta", "thetadot")


def prismatic_state(part: mp.PrismaticPart, prior_scale=None) -> _Phase2ModelState:
    return _Phase2ModelState(part, mp.PRISMATIC_ERROR_DIM, mp.prismatic_jvp,
                             mp.prismatic_vjp, mp.state_to_gaussians,
                             prior_scale)


def revolute_state(part: mr.RevolutePart, prior_scale=None) -> _Phase2ModelState:
    return _Phase2ModelState(part, mr.REVOLUTE_ERROR_DIM, mr.revolute_jvp,
                             mr.revolute_vjp, mr.state_to_gaussians,
                             prior_scale)


# ---------------------------------------------------------------------------
# Section 29: initializers
# ---------------------------------------------------------------------------

def init_prismatic(mu0: Tensor, members: Tensor, positions) -> mp.PrismaticPart:
    """Direction from the robust COM displacement; `s_t` by projection."""
    members = torch.as_tensor(members).long().reshape(-1)
    com = torch.stack([torch.as_tensor(p).double()[members].mean(0)
                       for p in positions])
    d = com - com[0]
    # Robust direction: the dominant singular vector of the displacement
    # stack, not the last frame, so one bad frame cannot set the axis.
    if float(d.abs().max()) < 1e-12:
        e = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    else:
        _u, _s, vh = torch.linalg.svd(d, full_matrices=False)
        e = vh[0]
        if float(d[-1] @ e) < 0:
            e = -e
    s = d @ e
    return mp.PrismaticPart(members, e, s)


def init_revolute(mu0: Tensor, members: Tensor, positions,
                  axis=None, angles=None, axis_point=None) -> mr.RevolutePart:
    """Axis from the generic screw when identifiable, else from the data.

    When the generic Phase-2 solution is passed in (`axis`, `angles`,
    `axis_point`) it is used only as a starting point, as section 29 allows.
    Otherwise the axis comes from the frame-to-frame relative rotation that
    best explains the observed point motion, and the axis point from the
    least-squares centre of the resulting circular arcs.
    """
    members = torch.as_tensor(members).long().reshape(-1)
    X = mu0.double().cpu()[members]
    centroid = X.mean(0)
    if axis is None or angles is None:
        from .manifold import so3_log
        P = [torch.as_tensor(p).double().cpu()[members] for p in positions]
        axes, angs = [], [0.0]
        for t in range(1, len(P)):
            A, B = P[0] - P[0].mean(0), P[t] - P[t].mean(0)
            U, _S, Vh = torch.linalg.svd(B.T @ A)
            D = torch.diag(torch.tensor(
                [1.0, 1.0, float(torch.det(U @ Vh))], dtype=torch.float64))
            w = so3_log((U @ D @ Vh).unsqueeze(0))[0]
            th = float(w.norm())
            angs.append(th)
            if th > 1e-9:
                axes.append(w / th)
        if not axes:
            axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        else:
            A = torch.stack(axes)
            axis = torch.linalg.svd(A, full_matrices=False)[2][0]
            if float(A.sum(0) @ axis) < 0:
                axis = -axis
        angles = torch.tensor(angs, dtype=torch.float64)
    axis = torch.as_tensor(axis).double().reshape(3).cpu()
    axis = axis / axis.norm().clamp_min(1e-300)
    angles = torch.as_tensor(angles).double().reshape(-1).cpu()
    E = mr.axis_basis(axis)
    if axis_point is None:
        rho = torch.zeros(2, dtype=torch.float64)
    else:
        a = torch.as_tensor(axis_point).double().reshape(3).cpu()
        rho = E.T @ (a - centroid)
    return mr.RevolutePart(members, centroid, axis, rho, angles, plane=E)


# ---------------------------------------------------------------------------
# Section 30: fit one model, and split the residual
# ---------------------------------------------------------------------------

SPLIT_HEADER = ["model", "frame", "camera", "rgb_energy", "depth_energy",
                "rgb_pixels", "depth_pixels"]


def residual_split(state, mu0, truth, cameras, frames, cfg, precision_fn,
                   build_linearization, label: str):
    """E_M split by RGB / depth / frame / camera (section 30)."""
    rows, tot_rgb, tot_d = [], 0.0, 0.0
    for t in range(len(frames)):
        mu, _ = state.gaussians(t, mu0, None)
        s = dataclasses.replace(truth, mu=mu.to(truth.mu.dtype))
        lin = build_linearization(s, cameras, eps_depth=cfg.eps_depth)
        obs = frames[t][0]
        for ci, (v, ob) in enumerate(zip(lin.views, obs)):
            if ob is None:
                rows.append([label, t, ci, 0.0, 0.0, 0, 0])
                continue
            pc, pd = precision_fn(lin, t)[ci]
            rr = (ob[0].to(v.rgb.device) - v.rgb).double()
            dr = (ob[1].to(v.depth.device) - v.depth).double()
            er = 0.5 * float((pc.double() * rr * rr).sum())
            ed = 0.5 * float((pd.double() * dr * dr).sum())
            rows.append([label, t, ci, er, ed,
                         int((pc > 0).sum()), int((pd > 0).sum())])
            tot_rgb += er
            tot_d += ed
    return {"total": tot_rgb + tot_d, "rgb": tot_rgb, "depth": tot_d,
            "rows": rows}


def fit_model(state, mu0, observations, linearize, precision_fn,
              cfg: JointIEKFConfig, label: str = "", visibility=None, observation_model=None):
    """Section 28: the shared IEKF, unchanged, driving a physical model.

    `label` only names the model on the progress bar; the fit is identical
    whatever it says.
    """
    return run_window(state, mu0, None, observations, linearize, precision_fn,
                      cfg=cfg, label=label, visibility=visibility,
                      observation_model=observation_model)


def heldout_delta(e_p: float, e_r: float) -> float:
    """Section 31's per-fold number: Delta E_heldout = E_P - E_R."""
    return float(e_p - e_r)


def predictive_nll(residual: Tensor, precision: Tensor, H: Tensor,
                   P: Tensor) -> float:
    """Section 32 via Woodbury, never a dense pixel covariance.

        S = R_eff + H P H^T,  with R_eff = diag(1/precision)

        nu^T S^-1 nu = nu^T W nu - (H^T W nu)^T (P^-1 + H^T W H)^-1 (H^T W nu)
        log|S|       = -sum log(precision) + log|P^-1 + H^T W H| + log|P|

    `H` here is the already-reduced [n_pix, p] measurement block, so the only
    determinant taken is p x p.
    """
    w = precision.double().reshape(-1)
    nu = residual.double().reshape(-1)
    keep = w > 0
    w, nu, H = w[keep], nu[keep], H.double()[keep]
    P = P.double()
    Pinv = torch.linalg.inv(P)
    HtW = H.T * w
    M = Pinv + HtW @ H
    g = HtW @ nu
    quad = float(w @ (nu * nu)) - float(g @ torch.linalg.solve(M, g))
    logdet = (-float(torch.log(w).sum())
              + float(torch.linalg.slogdet(M)[1])
              + float(torch.linalg.slogdet(P)[1]))
    return 0.5 * quad + 0.5 * logdet
