"""The nonlinear full-window IEKS: render, linearise, solve, retract, rerender.

P4B of CODEX_NEXT_STEPS_UNIFIED_PHASE1_AFTER_T5.md, section 3.5.

This is the piece that makes the rest an estimator rather than a solver and a
renderer that have never met. It runs

    1  validate the sealed factor inventory and the frame mapping
    2  evaluate the TRUE nonlinear posterior at the current nominal
    3  linearise every (frame, camera) and every adjacent process factor
    4  build A and b and a documented preconditioner
    5  solve A d = b by matrix-free PCG, with a reported true residual
    6  retract the whole trajectory and RERENDER for each trial
    7  accept on Armijo sufficient decrease of the SAME objective
    8  on rejection restore everything and raise damping
    9  stop on objective / gradient / step stationarity, with a status

Three traps this file is written around.

**Accepting on a surrogate.** The quadratic model built in step 3 is not the
objective. Step 6 rerenders; the line search compares the real posterior
before and after, never the model. With `robust=True` the effective precision
is state-dependent IRLS and the weighted quadratic is NOT the corresponding
negative log-likelihood, so robust mode is REFUSED here rather than reported
as a Student-t MAP. The Gaussian path is implemented and validated first,
which is what section 3.4 asks for.

**Counting an image twice.** The inventory is sealed before the loop. Each
outer iteration and each backtracking trial re-renders the same sealed set;
nothing registers a factor again. A rejected trial leaves the inventory
byte-identical.

**Calling the preconditioner a covariance.** `block_jacobi_preconditioner`
accelerates PCG and is never returned as posterior uncertainty.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import torch

from estimator.state import GaussianMotionState, boxplus

from .full_window_free import (
    FullWindowFreeOperator,
    ProcessFactor,
    block_jacobi_preconditioner,
    pcg,
)
from .process_factors import process_residual, process_residual_jacobians

CONVERGED = "CONVERGED"
MAX_ITERS = "MAX_ITERS"
LINE_SEARCH_FAILED = "LINE_SEARCH_FAILED"
PCG_FAILED = "PCG_FAILED"
NONFINITE = "NONFINITE"


@dataclass
class IEKSConfig:
    max_outer: int = 12
    pcg_max_iters: int = 200
    pcg_tol: float = 1e-8
    damping: float = 1e-3
    damping_up: float = 10.0
    damping_down: float = 0.3
    backtracking: tuple = (1.0, 0.5, 0.25, 0.1, 0.03)
    armijo_c: float = 1e-4
    grad_tol: float = 1e-8
    step_tol: float = 1e-10
    rel_objective_tol: float = 1e-10
    # Section 3.1: which policy anchors the source pose. 'fixed_conditional'
    # eliminates those tangent columns exactly; it never fakes a huge prior
    # precision to imitate a constraint.
    source_pose_policy: str = "optimizable_reference"
    anchor_frames: tuple = ()
    verbose: bool = True


@dataclass
class Trajectory:
    """The nominal trajectory: one GaussianMotionState per original frame."""

    states: list
    frame_ids: list
    dt: float

    @property
    def T(self) -> int:
        return len(self.states)

    @property
    def N(self) -> int:
        return int(self.states[0].mu.shape[0])

    def retract(self, step: torch.Tensor, scale: float = 1.0) -> "Trajectory":
        return Trajectory(
            states=[boxplus(s, (scale * step[t]).to(device=s.mu.device,
                                                    dtype=s.mu.dtype))
                    for t, s in enumerate(self.states)],
            frame_ids=list(self.frame_ids), dt=self.dt)

    def centers(self) -> torch.Tensor:
        return torch.stack([s.mu for s in self.states])

    def rotations(self) -> torch.Tensor:
        return torch.stack([s.R for s in self.states])

    def velocities(self) -> torch.Tensor:
        return torch.stack([s.v for s in self.states])

    def angular_velocities(self) -> torch.Tensor:
        return torch.stack([s.omega for s in self.states])


class UnifiedFreeIEKS:
    """One posterior over the whole window, solved by iterated smoothing."""

    def __init__(self, observation, Q_inv, prior_information=None,
                 prior_mean: Trajectory | None = None,
                 config: IEKSConfig | None = None):
        self.obs = observation
        self.cfg = config or IEKSConfig()
        self.Q_inv = torch.as_tensor(Q_inv).double()   # [N,12,12]
        self.prior_information = prior_information
        self.prior_mean = prior_mean
        if getattr(observation, "robust", False):
            raise NotImplementedError(
                "robust=True gives a state-dependent IRLS effective precision, "
                "and 0.5*sum(w*r^2) recomputed at each state is NOT the "
                "Student-t negative log-likelihood whose gradient those "
                "weights represent. Accepting a line-search step on that "
                "quantity while calling the result a robust MAP would be "
                "wrong. The Gaussian path is implemented and validated; the "
                "robust scalar potential has to be derived from the "
                "production weights before this is unblocked.")
        if self.cfg.source_pose_policy not in ("fixed_conditional",
                                               "optimizable_reference"):
            raise ValueError(
                "source_pose_policy must be 'fixed_conditional' (tangent "
                "columns eliminated) or 'optimizable_reference' (anchor kept "
                "as reference data, source-time pose free). Mixing them "
                "silently makes every reported uncertainty ambiguous.")

    # -- objective ---------------------------------------------------------
    def prior_residual(self, trajectory: Trajectory) -> torch.Tensor:
        """X_0 boxminus Xbar_0, in the current chart. Zero if no prior."""
        if self.prior_information is None or self.prior_mean is None:
            return torch.zeros(trajectory.N, 12, dtype=torch.float64,
                               device=trajectory.states[0].mu.device)
        from estimator.state import boxminus
        return boxminus(trajectory.states[0], self.prior_mean.states[0]).double()

    def evaluate(self, trajectory: Trajectory) -> dict:
        """The TRUE nonlinear posterior. Rerenders every frame."""
        rgb = depth = 0.0
        for t in range(trajectory.T):
            cost = self.obs.nonlinear_cost(trajectory.states[t], t)
            rgb += cost["rgb"]
            depth += cost["depth"]

        process = 0.0
        for t in range(trajectory.T - 1):
            e = process_residual(trajectory.states[t], trajectory.states[t + 1],
                                 trajectory.dt)
            Q = self.Q_inv.to(e.device)
            process += 0.5 * float(torch.einsum("ni,nij,nj->", e, Q, e))

        prior = 0.0
        if self.prior_information is not None:
            e0 = self.prior_residual(trajectory)
            P0 = self.prior_information.double().to(e0.device)
            prior = 0.5 * float(torch.einsum("ni,nij,nj->", e0, P0, e0))

        total = rgb + depth + process + prior
        return {"E_rgb": rgb, "E_depth": depth, "E_process": process,
                "E_prior": prior, "E_total": total}

    # -- linearisation -----------------------------------------------------
    def build_system(self, trajectory: Trajectory, damping: float):
        caches = [self.obs.linearize_frame(trajectory.states[t], t)
                  for t in range(trajectory.T)]

        process_factors = []
        for t in range(trajectory.T - 1):
            A, B = process_residual_jacobians(
                trajectory.states[t], trajectory.states[t + 1], trajectory.dt)
            e = process_residual(trajectory.states[t], trajectory.states[t + 1],
                                 trajectory.dt)
            process_factors.append(ProcessFactor(
                frame=t, A=A, B=B, Q_inv=self.Q_inv.to(A.device), residual=e))

        operator = _CudaWindowOperator(
            num_frames=trajectory.T, num_gaussians=trajectory.N,
            process_factors=process_factors, observation=self.obs,
            caches=caches, prior_information=self.prior_information,
            prior_residual=(self.prior_residual(trajectory)
                            if self.prior_information is not None else None),
            damping=damping,
            frozen_frames=(tuple(self.cfg.anchor_frames)
                           if self.cfg.source_pose_policy == "fixed_conditional"
                           else ()))
        return operator, caches

    # -- the outer loop ----------------------------------------------------
    def run(self, trajectory: Trajectory) -> dict:
        inventory = self.obs.build_inventory()
        expected_factors = inventory["num_factors"]
        trace = []
        damping = self.cfg.damping
        current = trajectory
        energy = self.evaluate(current)
        status = MAX_ITERS
        started = time.time()

        if self.cfg.verbose:
            print(f"  IEKS start  E_total={energy['E_total']:.6e} "
                  f"(rgb {energy['E_rgb']:.4e}, depth {energy['E_depth']:.4e}, "
                  f"proc {energy['E_process']:.4e}, prior {energy['E_prior']:.4e})"
                  f"  factors={expected_factors}")

        for outer in range(self.cfg.max_outer):
            operator, _caches = self.build_system(current, damping)
            b = operator.rhs()
            grad_norm = float(b.norm())
            if not math.isfinite(grad_norm):
                status = NONFINITE
                break
            if grad_norm < self.cfg.grad_tol:
                status = CONVERGED
                break

            step, solve_info = pcg(
                operator, b, preconditioner=block_jacobi_preconditioner(operator),
                max_iters=self.cfg.pcg_max_iters, tol=self.cfg.pcg_tol)
            if solve_info["status"] == "NON_POSITIVE_CURVATURE":
                damping *= self.cfg.damping_up
                trace.append({"outer": outer, "accepted": False,
                              "reason": "non_positive_curvature",
                              "damping": damping, **energy})
                continue
            if not torch.isfinite(step).all():
                status = NONFINITE
                break

            # Armijo needs g^T d < 0 with g = -b the true gradient.
            directional = -float((b * step).sum())
            step_norm = float(step.norm())
            if step_norm < self.cfg.step_tol:
                status = CONVERGED
                break

            accepted = False
            trials = []
            for scale in self.cfg.backtracking:
                candidate = current.retract(step, scale)
                trial_energy = self.evaluate(candidate)
                threshold = (energy["E_total"]
                             + self.cfg.armijo_c * scale * directional)
                ok = (math.isfinite(trial_energy["E_total"])
                      and trial_energy["E_total"] <= threshold)
                trials.append({"scale": scale,
                               "E_total": trial_energy["E_total"],
                               "threshold": threshold, "accepted": bool(ok)})
                if ok:
                    # On acceptance only: the nominal moves and the next outer
                    # iteration relinearises here.
                    current = candidate
                    energy = trial_energy
                    accepted = True
                    damping = max(damping * self.cfg.damping_down, 1e-12)
                    break

            # A rejected trial must leave the factor set untouched.
            after = self.obs.inventory.as_manifest()["num_factors"]
            if after != expected_factors:
                raise AssertionError(
                    f"the factor inventory changed from {expected_factors} to "
                    f"{after} during outer iteration {outer}. Relinearisation "
                    "and line-search trials must never register a factor.")

            record = {"outer": outer, "accepted": accepted,
                      "damping": damping, "grad_norm": grad_norm,
                      "step_norm": step_norm, "directional": directional,
                      "pcg_iterations": solve_info["iterations"],
                      "pcg_residual": solve_info["residual"],
                      "pcg_status": solve_info["status"],
                      "trials": trials, **energy}
            trace.append(record)
            if self.cfg.verbose:
                print(f"   outer {outer:2d} {'ACCEPT' if accepted else 'reject'}"
                      f"  E_total={energy['E_total']:.6e}"
                      f"  |g|={grad_norm:.3e} |d|={step_norm:.3e}"
                      f"  pcg {solve_info['iterations']}@{solve_info['residual']:.1e}"
                      f"  lambda={damping:.1e}")

            if not accepted:
                damping *= self.cfg.damping_up
                if damping > 1e12:
                    status = LINE_SEARCH_FAILED
                    break
                continue

            if len(trace) >= 2 and trace[-2].get("accepted"):
                rel = abs(trace[-2]["E_total"] - energy["E_total"]) / max(
                    abs(trace[-2]["E_total"]), 1e-30)
                if rel < self.cfg.rel_objective_tol:
                    status = CONVERGED
                    break

        return {"status": status, "energy": energy, "trace": trace,
                "trajectory": current, "factor_inventory": inventory,
                "wall_time_s": time.time() - started,
                "covariance_contract": {
                    "representation": "none_retained",
                    "note": ("The block-Jacobi operator used to precondition "
                             "PCG is NOT the posterior covariance: it drops "
                             "the temporal off-diagonals and the co-visible "
                             "cross-Gaussian coupling. No marginal covariance "
                             "is claimed at this scale."),
                }}


class _CudaWindowOperator(FullWindowFreeOperator):
    """FullWindowFreeOperator whose measurement blocks are the real renderer."""

    def __init__(self, *, num_frames, num_gaussians, process_factors,
                 observation, caches, prior_information, prior_residual,
                 damping, frozen_frames=()):
        super().__init__(num_frames=num_frames, num_gaussians=num_gaussians,
                         process_factors=process_factors,
                         measurement_factors=[],
                         prior_information=prior_information,
                         prior_residual=prior_residual, damping=damping)
        self.obs = observation
        self.caches = caches
        self.frozen = tuple(frozen_frames)

    def _freeze(self, u: torch.Tensor) -> torch.Tensor:
        """Exact elimination of a conditionally fixed source-time pose.

        The pose columns are zeroed, not given a large precision: a fake prior
        is a soft constraint that leaks, and its magnitude would show up in
        every reported uncertainty.
        """
        if not self.frozen:
            return u
        out = u.clone()
        for t in self.frozen:
            out[t, :, 0:6] = 0.0
        return out

    def information_matvec(self, u: torch.Tensor) -> torch.Tensor:
        u = self._freeze(self._check(u))
        out = torch.zeros_like(u)
        if self.prior_information is not None:
            out[0] += torch.einsum("nij,nj->ni", self.prior_information.double(), u[0])
        for t, cache in enumerate(self.caches):
            contribution = self.obs.information_apply(cache, u[t].to(cache.linearization.views[0].rgb.dtype))
            out[t] += contribution.double()
        for factor in self.process:
            t = factor.frame
            Ku = (torch.einsum("nij,nj->ni", factor.A, u[t])
                  + torch.einsum("nij,nj->ni", factor.B, u[t + 1]))
            QKu = torch.einsum("nij,nj->ni", factor.Q_inv, Ku)
            out[t] += torch.einsum("nji,nj->ni", factor.A, QKu)
            out[t + 1] += torch.einsum("nji,nj->ni", factor.B, QKu)
        if self.damping:
            out += self.damping * u
        return self._freeze(out)

    def rhs(self) -> torch.Tensor:
        b = self.zeros()
        if self.prior_information is not None and self.prior_residual is not None:
            b[0] -= torch.einsum("nij,nj->ni", self.prior_information.double(),
                                 self.prior_residual.double())
        for t, cache in enumerate(self.caches):
            b[t] += self.obs.gradient(cache).double()
        for factor in self.process:
            t = factor.frame
            Qe = torch.einsum("nij,nj->ni", factor.Q_inv, factor.residual.double())
            b[t] -= torch.einsum("nji,nj->ni", factor.A, Qe)
            b[t + 1] -= torch.einsum("nji,nj->ni", factor.B, Qe)
        return self._freeze(b)

    def dense_information(self):
        raise NotImplementedError(
            "the CUDA-backed operator is matrix-free by construction; "
            "dense_information exists only on the synthetic linear operator "
            "used by the T7 reference tests")


def save_run(out_dir, result: dict) -> dict:
    """Original-time arrays plus the traces section 3.5 step 10 requires."""
    import numpy as np

    out_dir = __import__("pathlib").Path(out_dir)
    free = out_dir / "unified_free"
    free.mkdir(parents=True, exist_ok=True)
    trajectory = result["trajectory"]
    np.save(free / "positions_original_order.npy",
            trajectory.centers().detach().cpu().numpy())
    np.save(free / "rotations_original_order.npy",
            trajectory.rotations().detach().cpu().numpy())
    np.save(free / "velocities_physical_original_order.npy",
            trajectory.velocities().detach().cpu().numpy())
    np.save(free / "angular_velocities_physical_original_order.npy",
            trajectory.angular_velocities().detach().cpu().numpy())
    with open(free / "solver_trace.jsonl", "w") as handle:
        for record in result["trace"]:
            handle.write(json.dumps(record) + "\n")
    (free / "frame_camera_factor_inventory.json").write_text(
        json.dumps(result["factor_inventory"], indent=1))
    (free / "covariance_contract.json").write_text(
        json.dumps(result["covariance_contract"], indent=1))
    (out_dir / "run_status.json").write_text(json.dumps({
        "status": result["status"], "energy": result["energy"],
        "wall_time_s": result["wall_time_s"],
        "original_frame_ids": list(trajectory.frame_ids)}, indent=1))
    return {"out_dir": str(out_dir)}
