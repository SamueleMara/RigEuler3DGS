"""Sections 9, 12, 13: rigid part transforms estimated through the renderer.

CODEX_IMPLEMENT_GEOMETRIC_RIGID_MOTION_JOINT_DISCOVERY.md sections 8-13.

Section 8 makes {T_{k,t}} in SE(3) the central state during part discovery, and
is explicit that no P/R structure may be imposed yet. Section 9 says the part
transforms are to be optimised against the real RGB-D window through the
analytic renderer Jacobian, compressing the whole Gaussian state into K rigid
transforms per frame:

    delta xi_{i,t} = G_{i,k,t} delta zeta_{k,t}
    H_{k,t}        = sum_{i: z_i = k} H_{i,t} G_{i,k,t}

so one 6K x 6K Gauss-Newton system per frame replaces a scene-scale solve. The
cross-part blocks are kept because two parts share pixels.

Nothing in this file knows what a joint is. That is section 48's ordering: the
renderer decides who moves and how much, and only afterwards does
`joint_reduction.geometric_joint_fit` decide which joint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from estimator.manifold import project_to_so3, so3_exp, so3_hat, so3_log

EPS = 1e-12


@dataclass
class RigidTransforms:
    """{T_{k,t}} for K moving parts over T frames, in section 13's form.

        x_{i,t} = R_{k,t} x_i^0 + t_{k,t},     R^G_{i,t} = R_{k,t} R_i^0

    The static part is index 0 and is always the identity (section 8's
    T_{0,t} = I); it is never stored and never optimised.
    """
    R: torch.Tensor          # [T,K,3,3]
    t: torch.Tensor          # [T,K,3]

    @staticmethod
    def identity(T: int, K: int, device=None, dtype=torch.float64):
        eye = torch.eye(3, device=device, dtype=dtype)
        return RigidTransforms(
            R=eye.reshape(1, 1, 3, 3).repeat(T, K, 1, 1).clone(),
            t=torch.zeros((T, K, 3), device=device, dtype=dtype))

    @property
    def num_frames(self) -> int:
        return int(self.R.shape[0])

    @property
    def K(self) -> int:
        return int(self.R.shape[1])

    def clone(self) -> "RigidTransforms":
        return RigidTransforms(R=self.R.clone(), t=self.t.clone())

    def part_trajectory(self, k: int):
        """(R, t) over time for part k (1-based), for the geometric joint fit."""
        return self.R[:, k - 1], self.t[:, k - 1]

    def to_dict(self) -> dict:
        return {"R": self.R.detach().cpu().tolist(),
                "t": self.t.detach().cpu().tolist(),
                "num_frames": self.num_frames, "K": self.K}


def apply_transforms(x0: torch.Tensor, R0: torch.Tensor | None,
                     labels: torch.Tensor, transforms: RigidTransforms,
                     frame: int):
    """Section 8's G_t = union_k T_{k,t} G^0_k.

    Static Gaussians (label 0) keep their canonical geometry exactly -- that is
    T_{0,t} = I, not an approximation of it.
    """
    mu = x0.clone()
    R = None if R0 is None else R0.clone()
    for k in range(1, transforms.K + 1):
        mask = labels == k
        if not bool(mask.any()):
            continue
        Rk = transforms.R[frame, k - 1].to(x0.dtype)
        tk = transforms.t[frame, k - 1].to(x0.dtype)
        mu[mask] = x0[mask] @ Rk.T + tk
        if R is not None:
            R[mask] = Rk.unsqueeze(0) @ R0[mask]
    return mu, R


def rigid_tangent(mu: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """G_{i,k,t} as six [N,6] columns, stacked to [N,6,6].

    A rigid increment (dv, dw) about the part's CURRENT centroid moves a member
    by dx_i = dv + dw x (x_i - c) and turns it by dphi_i = dw, so

        G_i = [[ I, -[x_i - c]_x ],
               [ 0,  I          ]]

    in the state layout (mu, phi). Taking the increment about the centroid
    rather than the origin decouples rotation from translation and is what
    keeps the 6x6 block well conditioned when the part sits far from the
    world origin.
    """
    n = mu.shape[0]
    d = mu - center.reshape(1, 3)
    G = torch.zeros((n, 6, 6), device=mu.device, dtype=mu.dtype)
    eye = torch.eye(3, device=mu.device, dtype=mu.dtype)
    G[:, :3, :3] = eye.unsqueeze(0)
    G[:, :3, 3:] = -so3_hat(d)
    G[:, 3:, 3:] = eye.unsqueeze(0)
    return G


class RigidPartObservation:
    """H_{k,t} = sum_{i: z_i = k} H_{i,t} G_{i,k,t}, matrix free (section 9).

    Every column is one Gaussian-level JVP through the real rasterizer, so the
    renderer needs no change -- exactly as section 2.2 requires.
    """

    def __init__(self, linearization, mu: torch.Tensor, labels: torch.Tensor,
                 K: int):
        self.lin = linearization
        self.mu = mu
        self.labels = labels
        self.K = int(K)
        self.centers = []
        self._cols = []
        for k in range(1, self.K + 1):
            mask = labels == k
            c = (mu[mask].mean(dim=0) if bool(mask.any())
                 else torch.zeros(3, device=mu.device, dtype=mu.dtype))
            self.centers.append(c)
            G = rigid_tangent(mu, c)                 # [N,6,6]
            cols = []
            for j in range(6):
                col = G[:, :, j].clone()
                col[~mask] = 0.0                     # zero outside the part
                cols.append(col.contiguous())
            self._cols.append(cols)

    def column(self, k: int, j: int) -> torch.Tensor:
        return self._cols[k - 1][j]

    def normal_system(self, precisions, residuals):
        """(A, b) of size [6K,6K] and [6K] for ONE frame.

        A = H^T R_eff^-1 H and b = H^T R_eff^-1 r in the stacked rigid
        coordinate (dzeta_1 .. dzeta_K).
        """
        n = 6 * self.K
        dev = self.mu.device
        weighted = [(p * r, q * d) for (p, q), (r, d)
                    in zip(precisions, residuals)]
        # ONE adjoint for the whole gradient: H^T R^-1 r does not depend on
        # which column it is later contracted with, so computing it inside the
        # loop would cost 6K identical VJPs.
        g_r = self.lin.apply_HT(weighted)
        b = torch.zeros(n, dtype=torch.float64, device=dev)
        hv = []
        for k in range(1, self.K + 1):
            for j in range(6):
                col = self.column(k, j)
                b[(k - 1) * 6 + j] = (g_r * col).sum().double()
                hv.append(self.lin.apply_H(col))
        A = torch.zeros((n, n), dtype=torch.float64, device=dev)
        for ci in range(n):
            w = [(p * a, q * d) for (p, q), (a, d)
                 in zip(precisions, hv[ci])]
            g = self.lin.apply_HT(w)
            for cj in range(ci, n):
                k2, j2 = divmod(cj, 6)
                A[ci, cj] = (g * self.column(k2 + 1, j2)).sum().double()
                A[cj, ci] = A[ci, cj]
        return A, b


def _increment(transforms: RigidTransforms, frame: int, k: int,
               dzeta: torch.Tensor, x0: torch.Tensor, labels: torch.Tensor):
    """Apply a rigid increment about the part's current centroid.

    With c_t = R c_0 + t the current centroid, the increment
    x -> Exp(dw)(x - c_t) + c_t + dv gives

        R <- Exp(dw) R,      t <- c_t + dv - Exp(dw) R c_0

    which is written out rather than approximated, so a large step stays an
    exact rigid motion instead of drifting off SE(3).
    """
    mask = labels == k
    if not bool(mask.any()):
        return
    dv = dzeta[:3].double()
    dw = dzeta[3:].double()
    R = transforms.R[frame, k - 1].double()
    t = transforms.t[frame, k - 1].double()
    c0 = x0[mask].double().mean(dim=0)
    ct = R @ c0 + t
    dR = so3_exp(dw.reshape(1, 3))[0]
    transforms.R[frame, k - 1] = project_to_so3((dR @ R).unsqueeze(0))[0]
    transforms.t[frame, k - 1] = ct + dv - (dR @ R) @ c0


@torch.no_grad()
def refine_rigid_transforms(objective, x0, R0, labels,
                            transforms: RigidTransforms, *,
                            max_iters: int = 12, lam0: float = 1e-3,
                            lam_up: float = 10.0, lam_down: float = 3.0,
                            max_trials: int = 12, tau_energy: float = 1e-6,
                            warm_start: bool = True, progress: str | None = None,
                            verbose: bool = False) -> dict:
    """Section 9: optimise {T_{k,t}} against the real RGB-D window.

    Each frame is solved independently because T_{k,t} enters only frame t's
    render -- the same block structure the trajectory solver exploits. The step
    is Levenberg-damped and accepted only on a measured decrease of the TRUE
    nonlinear objective (section 10 F).

    `warm_start` seeds a still-identity frame from the frame before it. This is
    temporal continuation, not a prior: a render has a finite basin of
    attraction, and starting every frame from the identity makes the late
    frames of a long sweep unreachable. Measured on a 5-frame revolute sweep of
    0.36 rad, cold starts recovered almost no rotation (|dR| growing to 0.32 by
    the last frame) while the energy still fell, i.e. the solver explained a
    rotation with a translation. The seed is only a starting point; every frame
    is still optimised against its own data and accepted only on a measured
    decrease.
    """
    K = transforms.K
    info = {"per_frame": [], "energy0": 0.0, "energy": 0.0}
    total0 = total1 = 0.0

    eye3 = torch.eye(3, dtype=transforms.R.dtype, device=transforms.R.device)
    frames = list(enumerate(objective.frames))
    if progress:
        from tools.kpart_pipeline import _progress
        frames = _progress(frames, progress, True)
    for ti, (obs, fid) in frames:
        if warm_start and ti > 0:
            for k in range(K):
                is_identity = (
                    float((transforms.R[ti, k] - eye3).abs().max()) < 1e-12
                    and float(transforms.t[ti, k].abs().max()) < 1e-12)
                if is_identity:
                    transforms.R[ti, k] = transforms.R[ti - 1, k].clone()
                    transforms.t[ti, k] = transforms.t[ti - 1, k].clone()
        mu, R = apply_transforms(x0, R0, labels, transforms, ti)
        er, ed = objective.frame_energy(mu, R, obs)
        E = er + ed
        E_start = E
        lam = float(lam0)
        hist = [E]
        status = "max_iters"

        for _it in range(int(max_iters)):
            mu, R = apply_transforms(x0, R0, labels, transforms, ti)
            lin, prec, res = objective.linearize_frame(mu, R, obs)
            ob = RigidPartObservation(lin, mu, labels, K)
            A, b = ob.normal_system(prec, res)
            del lin, ob
            n = 6 * K
            eye = torch.eye(n, dtype=torch.float64, device=A.device)
            accepted = False
            for _trial in range(int(max_trials)):
                try:
                    L = torch.linalg.cholesky(A + lam * eye)
                    step = torch.cholesky_solve(b.unsqueeze(1), L).squeeze(1)
                except Exception:
                    lam *= lam_up
                    continue
                trial = transforms.clone()
                for k in range(1, K + 1):
                    _increment(trial, ti, k, step[(k - 1) * 6:k * 6], x0, labels)
                mu_t, R_t = apply_transforms(x0, R0, labels, trial, ti)
                er2, ed2 = objective.frame_energy(mu_t, R_t, obs)
                if er2 + ed2 < E - 1e-9:
                    transforms.R[ti] = trial.R[ti]
                    transforms.t[ti] = trial.t[ti]
                    E = er2 + ed2
                    hist.append(E)
                    lam = max(lam / lam_down, 1e-10)
                    accepted = True
                    break
                lam *= lam_up
            if not accepted:
                status = "no_decrease"
                break
            if len(hist) >= 2 and abs(hist[-2] - hist[-1]) <= tau_energy * max(1.0, abs(hist[-2])):
                status = "converged"
                break

        total0 += E_start
        total1 += E
        info["per_frame"].append({"frame": fid, "energy0": E_start,
                                  "energy": E, "status": status,
                                  "iterations": len(hist) - 1})
        if verbose:
            print(f"       frame {fid}: E {E_start:.2f} -> {E:.2f} ({status})")

    info["energy0"] = total0
    info["energy"] = total1
    info["converged"] = all(f["status"] in ("converged", "no_decrease")
                            for f in info["per_frame"])
    info["all_frames_converged"] = all(f["status"] == "converged"
                                       for f in info["per_frame"])
    # Which frames ran out of budget rather than converging. Without this an
    # `all_frames_converged = False` says only that SOMETHING did not finish,
    # and there is no way to tell one stubborn frame from every frame hitting
    # the cap -- i.e. no way to know whether raising --rigid-refine-iters would
    # help at all.
    from collections import Counter
    counts = Counter(f["status"] for f in info["per_frame"])
    info["status_counts"] = dict(counts)
    info["frames_at_iteration_cap"] = int(counts.get("max_iters", 0))
    info["iteration_cap"] = int(max_iters)
    used = [f["iterations"] for f in info["per_frame"]]
    info["iterations_used"] = {
        "mean": (sum(used) / len(used)) if used else 0.0,
        "max": max(used) if used else 0,
        "at_cap_fraction": (counts.get("max_iters", 0) / len(used)) if used else 0.0}
    return info


@torch.no_grad()
def instantaneous_twist(mu: torch.Tensor, v: torch.Tensor,
                        weights: torch.Tensor | None = None) -> dict:
    """Section 12's weighted least-squares twist, as an INITIALIZER only.

        xi_k = argmin_xi sum_{i: z_i = k} w_i || v_i - (v + omega x x_i) ||^2

    Section 12 is explicit that this is a rigid-consistency diagnostic, a
    transform initialization and a reassignment cue -- never the P/R decision.
    """
    from joint_reduction.rigid_fit import fit_rigid_twist
    out = fit_rigid_twist(mu, v, weights=weights)
    out["use"] = ("section 12: initializer, rigidity diagnostic and "
                  "reassignment cue only; never a joint-type decision")
    return out


@torch.no_grad()
def initialize_transforms(x0: torch.Tensor, labels: torch.Tensor,
                          positions: list, K: int,
                          weights: torch.Tensor | None = None
                          ) -> RigidTransforms:
    """Sections 10 A / 12 / 13: seed {T_{k,t}} from the motion evidence.

    `positions` is, per frame, the per-Gaussian position implied by the
    multiview motion bootstrap (canonical geometry plus the recovered
    displacement field). For each part and frame the seed is the weighted
    Procrustes solution over that part's members.

    This step is not optional in practice. Measured on a 300-Gaussian part at
    48x48 over two cameras: refining from the identity recovered a 0.30 rad
    rotation only to |dR| = 0.34 after 60 Levenberg iterations and stalled with
    a large residual, because a render has a finite basin of attraction and
    rotation is the slowest direction in it. Seeding with Procrustes on the
    displacement field gave |dR| = 5.2e-08 immediately, after which the
    renderer refinement correctly had nothing left to do. Section 12 names
    exactly this use -- "transform initialization" -- and section 10 orders it
    before the fit.
    """
    from joint_reduction.joint_geometry import fit_rigid_transform

    T = len(positions)
    tr = RigidTransforms.identity(T, K, device=x0.device)
    for k in range(1, K + 1):
        mask = labels == k
        if int(mask.sum()) < 3:
            continue
        w = None if weights is None else weights[mask]
        src = x0[mask].double()
        for t in range(T):
            R, tt = fit_rigid_transform(src, positions[t][mask].double(), w)
            tr.R[t, k - 1] = R
            tr.t[t, k - 1] = tt
    return tr


@torch.no_grad()
def rigid_residual_per_gaussian(x0: torch.Tensor, labels: torch.Tensor,
                                positions: list,
                                transforms: RigidTransforms) -> torch.Tensor:
    """Section 17 / 29: how badly each Gaussian violates its part's rigid motion.

    Returns ``[N]`` RMS position error over the window. Section 40 requires a
    non-rigid subgroup to show a HIGH value here so it can be split off, and
    section 29 requires it persisted.
    """
    n = x0.shape[0]
    acc = torch.zeros(n, device=x0.device, dtype=torch.float64)
    cnt = 0
    for t, pos in enumerate(positions):
        pred = x0.double().clone()
        for k in range(1, transforms.K + 1):
            mask = labels == k
            if not bool(mask.any()):
                continue
            R = transforms.R[t, k - 1].double()
            tt = transforms.t[t, k - 1].double()
            pred[mask] = x0[mask].double() @ R.T + tt
        acc += ((pos.double() - pred) ** 2).sum(dim=1)
        cnt += 1
    return (acc / max(cnt, 1)).sqrt()


# ---------------------------------------------------------------------------
# CODEX_IMPLEMENT_WHOLE_WINDOW_RIGID_COMPONENT_DISCOVERY.md sections 12, 13
# ---------------------------------------------------------------------------
@torch.no_grad()
def robust_fit_frame_transform(x0: torch.Tensor, xt: torch.Tensor,
                               weights: torch.Tensor | None = None, *,
                               loss: str = "tukey", tukey_c: float = 4.685,
                               huber_c: float = 1.345, iters: int = 12,
                               floor: float = 1e-9) -> dict:
    """Section 12's robust T_{k,t} for one component at one frame.

        T = argmin_T  sum_i  w_i rho( || x_i^t - T x_i^0 || )

    IRLS over the exact Procrustes solution, with the scale recomputed from
    the residuals each pass as section 13's 1.4826 * MAD -- the real MAD,
    median | r - median(r) |, not 1.4826 * median(r).

    Section 20: these weights DOWNWEIGHT outliers within a component. They are
    not a membership rule; membership is decided by the whole-track cost of
    section 14.
    """
    from joint_reduction.joint_geometry import fit_rigid_transform
    from joint_reduction.rigidity_graph import robust_scale

    x0 = x0.double()
    xt = xt.double()
    n = x0.shape[0]
    w = (torch.ones(n, dtype=torch.float64, device=x0.device)
         if weights is None else weights.double().clamp_min(0.0))
    w0 = w.clone()
    R = torch.eye(3, dtype=torch.float64, device=x0.device)
    t = torch.zeros(3, dtype=torch.float64, device=x0.device)
    res = torch.zeros(n, dtype=torch.float64, device=x0.device)
    scale = torch.tensor(float(floor), dtype=torch.float64)
    for _ in range(int(iters)):
        R, t = fit_rigid_transform(x0, xt, w)
        res = (xt - (x0 @ R.T + t)).norm(dim=1)
        scale = robust_scale(res, floor=floor)
        u = res / scale.clamp_min(floor)
        if loss == "huber":
            rw = torch.where(u <= huber_c, torch.ones_like(u), huber_c / u.clamp_min(floor))
        elif loss == "cauchy":
            rw = 1.0 / (1.0 + (u / huber_c) ** 2)
        else:                                            # tukey, redescending
            uu = (u / tukey_c).clamp(max=1.0)
            rw = (1.0 - uu ** 2) ** 2
        w = w0 * rw
        if float(w.sum()) <= floor:                      # everything rejected
            w = w0.clone()
            break
    return {"R": R, "t": t, "weights": w, "residual": res,
            "robust_scale": float(scale), "loss": loss}


@torch.no_grad()
def robust_fit_se3_trajectory(x0: torch.Tensor, tracks: torch.Tensor,
                              member: torch.Tensor,
                              confidence: torch.Tensor | None = None,
                              **kw) -> dict:
    """Section 12 over the whole window: {T_{k,t}} for one component.

    `tracks` is ``[T, C, 3]`` and `member` selects this component's supernodes.
    `confidence` is section 19's omega_{i,t} -- visibility, information and
    track stability -- so a badly observed node cannot dominate the fit.
    """
    T = int(tracks.shape[0])
    dev = tracks.device
    idx = torch.nonzero(member, as_tuple=False).reshape(-1)
    R = torch.eye(3, dtype=torch.float64, device=dev).reshape(1, 3, 3).repeat(T, 1, 1)
    t = torch.zeros((T, 3), dtype=torch.float64, device=dev)
    info = {"frames": [], "members": int(idx.numel())}
    if idx.numel() < 3:
        info["degenerate"] = True
        return {"R": R, "t": t, **info}
    src = x0[idx].double()
    for f in range(T):
        w = None if confidence is None else confidence[f][idx]
        out = robust_fit_frame_transform(src, tracks[f][idx], w, **kw)
        R[f], t[f] = out["R"], out["t"]
        info["frames"].append({
            "frame": f, "robust_scale": out["robust_scale"],
            "residual_median": float(out["residual"].median()),
            "downweighted": int((out["weights"] < 0.1 * out["weights"].max()
                                 .clamp_min(1e-12)).sum())})
    info["degenerate"] = False
    return {"R": R, "t": t, **info}


@torch.no_grad()
def se3_distance_to_identity(R: torch.Tensor, t: torch.Tensor,
                             length: float) -> float:
    """Section 11's D_k^static = mean_t d_SE(3)(T_{k,t}, I)^2 with

        d^2 = ||t||^2 / L^2 + ||log R||^2

    so a translation of one component radius counts the same as one radian.
    """
    from joint_reduction.joint_geometry import normalized_geometric_error
    T = int(R.shape[0])
    eye = torch.eye(3, dtype=torch.float64, device=R.device)
    out = normalized_geometric_error(
        R.double(), t.double(), eye.reshape(1, 3, 3).repeat(T, 1, 1),
        torch.zeros((T, 3), dtype=torch.float64, device=R.device), length)
    return float(out["error"])
