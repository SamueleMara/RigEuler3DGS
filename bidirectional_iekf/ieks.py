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
    ERROR_DIM,
    FullWindowFreeOperator,
    ProcessFactor,
    block_jacobi_preconditioner,
    pcg,
)
from .process_factors import process_residual, process_residual_jacobians

CONVERGED = "CONVERGED"
STALLED = "STALLED"
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
    # ||b|| relative to the objective value. A stall with a large gradient is
    # STALLED; only genuine stationarity is CONVERGED.
    grad_rel_tol: float = 1e-3
    # DIAGNOSTIC ONLY since the certificate below replaced it. |g|/|g_0| made
    # the verdict a property of the initialisation: a fixture whose noise was
    # enlarged crossed this threshold without the solver stopping anywhere
    # different.
    grad_decay_tol: float = 1e-3
    # The stationarity certificate. 0.5 b'A^-1 b -- the quadratic model's
    # prediction of ALL remaining reduction -- as a fraction of the current
    # objective. It depends on the current point alone, so it cannot be moved
    # by changing where the run started. 1e-6 is the float32 rendering
    # reproducibility floor measured in tests/test_unified_streaming_window.py
    # (~2e-7 relative, three resident-vs-resident rebuilds): below it the
    # model predicts less improvement than the renderer can resolve.
    decrement_tol: float = 1e-6
    # ... OR the renderer's own reproducibility, measured per run. The
    # rasterizer composites in float32 with a non-reproducible tile order, so
    # evaluating the same nominal twice gives two different objectives. A
    # model that predicts less reduction than THAT is predicting something
    # the renderer cannot resolve, and the solve has reached the floor of
    # what this arithmetic can say. Measured at the start, from two
    # evaluations of the warm start, so it is a property of the scene rather
    # than a constant guessed for it.
    measure_objective_noise: bool = True
    noise_factor: float = 1.0
    # A tiny step is only a stall if it PERSISTS while the certificate stays
    # unmet. Breaking on the first one declared a stall at a point where the
    # model still predicted progress and the next few steps delivered it.
    tiny_step_patience: int = 6
    # STATE stationarity. The gradient cannot be the sole criterion here:
    # it floors at the float32 rendering noise (measured ~4e-2 on a fixture
    # whose objective had already reached 6e-8), while the step norm falls to
    # ~2e-7 and the state has genuinely stopped moving. Requiring |g| -> 0
    # would report a converged estimate as MAX_ITERS forever.
    step_tol: float = 1e-5
    rel_objective_tol: float = 1e-10
    # Section 3.1: which policy anchors the source pose. 'fixed_conditional'
    # eliminates those tangent columns exactly; it never fakes a huge prior
    # precision to imitate a constraint.
    source_pose_policy: str = "optimizable_reference"
    anchor_frames: tuple = ()
    # Per-GROUP anchors: ((window_frame_index, candidate_indices), ...). The
    # two-source union needs the initial group's pose fixed at original frame
    # 0 and the final group's at frame 59 -- and NOT the other way round.
    # `anchor_frames` freezes a frame for every candidate, which in a union
    # would clamp the opposite source at a time it was never reconstructed
    # at. Only the 6 pose columns are eliminated; rates stay free.
    anchor_groups: tuple = ()
    # Called with (outer, trajectory, energy) after every ACCEPTED step, so a
    # caller can keep the best accepted iterate on disk. The objective is
    # monotone over accepted steps, so the latest one is the best.
    on_accept: object = None
    # Called at the top of every outer iteration; a non-empty string return
    # stops the run with that string as its status (e.g. TIME_BUDGET,
    # STOPPED_BY_USER). The nominal at that point is the last ACCEPTED one,
    # so nothing half-applied is ever left behind.
    stop_check: object = None
    # Tangent columns held at zero. A DIAGNOSTIC ablation, not a model: used
    # to isolate whether a stall comes from the weakly observed rate blocks.
    frozen_columns: tuple = ()
    # Keep ONE frame's rasterizer cache resident instead of the whole window.
    # Measured on the real Washer scene (24,582 Gaussians, 640x480, 8 GiB
    # card): 878 MiB at 4 (frame, camera) factors, 1901 MiB at 16, and an
    # out-of-memory at 64 -- about 85 MiB a factor, so the 480 factors of a
    # 60-frame 8-camera window need roughly 41 GiB and cannot fit. Streaming
    # re-renders each frame instead of reading its cache; the arithmetic is
    # unchanged and tests/test_unified_streaming_window.py checks that.
    stream_frames: bool = False
    # Keep the per-frame-pair process Jacobians in host memory and move one
    # pair at a time. A and B are dense [N,12,12] float64, so at 24,582
    # Gaussians one pair is 2 x 28 MiB and a 60-frame window is 3.3 GiB of
    # device memory that the rasterizer then cannot have. Streaming the frames
    # alone is not enough: with 30 frames x 8 cameras the window evaluates in
    # 544 MiB but the system build still ran out at 4438 MiB on an 8 GiB card.
    # The transfer is about 4% of a matvec whose renders dominate.
    process_factors_on_cpu: bool = False
    # Precondition on the DIAGONAL instead of the 12x12 blocks. A
    # preconditioner changes how fast PCG converges, never what it converges
    # to, so this is a memory/iteration trade and not a modelling choice. It
    # is what the full real window needs: the block form holds [T,N,12,12]
    # float64 three times over (the Fisher memo, the blocks, their inverse),
    # which is 1.70 GiB each at 60 frames and 24,582 Gaussians, while the
    # diagonal form holds [T,N,12] -- 142 MiB.
    diagonal_preconditioner: bool = False
    # Recompute ||b - A d|| / ||b|| at the end of every PCG solve. One extra
    # matvec per outer iteration. The recursive residual CG carries is what
    # the stopping rule uses, but it drifts, and on a stiff window it can
    # read small while the iterate is not a solution -- so a run that claims
    # convergence should be able to show the real number.
    pcg_true_residual: bool = False
    # Keep the full-window float64 vectors (PCG state, preconditioner, LM
    # scale) in HOST memory; only one frame's tangent visits the GPU at a
    # time. At 45,920 Gaussians x 60 frames each such vector is 264 MB and
    # PCG holds about eight, which is what an 8 GiB card shared with other
    # jobs cannot spare. The arithmetic is unchanged (float64 either way).
    solve_on_cpu: bool = False
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

    def evaluate(self, trajectory: Trajectory, *, with_support: bool = False) -> dict:
        """The TRUE nonlinear posterior. Rerenders every frame.

        `with_support` also returns the number of pixels that actually carry
        depth. The alpha > min_alpha_for_depth gate means the objective can
        fall simply because fewer depth terms are valid, so the count is
        recorded alongside the energy and an 'improvement' of that kind is
        visible rather than silent.
        """
        from .observations import depth_support

        rgb = depth = 0.0
        support = {"total": 0, "per_frame": []}
        for t in range(trajectory.T):
            if with_support:
                cache = self.obs.linearize_frame(trajectory.states[t], t)
                counts = depth_support(self.obs, cache)
                support["total"] += counts["total"]
                support["per_frame"].append(counts["total"])
                for residual, precision in zip(cache.residuals, cache.precisions):
                    if residual is None:
                        continue
                    from estimator.measurement import measurement_energy
                    rgb += float(measurement_energy(residual[0], precision[0]))
                    depth += float(measurement_energy(residual[1], precision[1]))
                continue
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
        out = {"E_rgb": rgb, "E_depth": depth, "E_process": process,
               "E_prior": prior, "E_total": total}
        if with_support:
            out["depth_support"] = support
        return out

    # -- linearisation -----------------------------------------------------
    def build_system(self, trajectory: Trajectory, damping: float):
        caches = (_StreamedCaches(self.obs, trajectory.states)
                  if self.cfg.stream_frames
                  else [self.obs.linearize_frame(trajectory.states[t], t)
                        for t in range(trajectory.T)])

        host = (torch.device("cpu") if self.cfg.process_factors_on_cpu
                else None)
        process_factors = []
        for t in range(trajectory.T - 1):
            A, B = process_residual_jacobians(
                trajectory.states[t], trajectory.states[t + 1], trajectory.dt)
            e = process_residual(trajectory.states[t], trajectory.states[t + 1],
                                 trajectory.dt)
            Q = self.Q_inv.to(A.device)
            if host is not None:
                A, B, Q, e = A.to(host), B.to(host), Q.to(host), e.to(host)
            process_factors.append(ProcessFactor(frame=t, A=A, B=B, Q_inv=Q,
                                                 residual=e))

        operator = _CudaWindowOperator(
            num_frames=trajectory.T, num_gaussians=trajectory.N,
            process_factors=process_factors, observation=self.obs,
            caches=caches, prior_information=self.prior_information,
            prior_residual=(self.prior_residual(trajectory)
                            if self.prior_information is not None else None),
            damping=damping,
            frozen_frames=(tuple(self.cfg.anchor_frames)
                           if self.cfg.source_pose_policy == "fixed_conditional"
                           else ()),
            frozen_groups=(tuple(self.cfg.anchor_groups)
                           if self.cfg.source_pose_policy == "fixed_conditional"
                           else ()),
            work_device=(torch.device("cpu") if self.cfg.solve_on_cpu
                         else trajectory.states[0].mu.device))
        return operator, caches

    # -- the outer loop ----------------------------------------------------
    def run(self, trajectory: Trajectory) -> dict:
        inventory = self.obs.build_inventory()
        expected_factors = inventory["num_factors"]
        trace = []
        damping = self.cfg.damping
        initial_grad = None
        tiny_steps = 0
        current = trajectory
        energy = self.evaluate(current)
        objective_noise = 0.0
        if self.cfg.measure_objective_noise:
            repeat = self.evaluate(current)
            objective_noise = abs(repeat["E_total"] - energy["E_total"])
        status = MAX_ITERS
        started = time.time()

        if self.cfg.verbose and self.cfg.measure_objective_noise:
            print(f"  renderer objective reproducibility: {objective_noise:.3e} "
                  f"absolute on E_total={energy['E_total']:.6e} "
                  f"({objective_noise / max(abs(energy['E_total']), 1e-30):.2e} "
                  "relative); a predicted reduction below it is below what "
                  "the renderer can resolve")
        if self.cfg.verbose:
            print(f"  IEKS start  E_total={energy['E_total']:.6e} "
                  f"(rgb {energy['E_rgb']:.4e}, depth {energy['E_depth']:.4e}, "
                  f"proc {energy['E_process']:.4e}, prior {energy['E_prior']:.4e})"
                  f"  factors={expected_factors}")

        for outer in range(self.cfg.max_outer):
            if self.cfg.stop_check is not None:
                reason = self.cfg.stop_check()
                if reason:
                    status = str(reason)
                    break
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
                operator, b,
                # The preconditioner's 12x12 blocks mix pose and rate, so a
                # residual that is zero on an eliminated pose coordinate can
                # come back nonzero there, and CG would then move an anchor
                # it was told is fixed. Eliminating its output too keeps
                # every CG vector inside the constrained subspace.
                preconditioner=_eliminated(operator, block_jacobi_preconditioner(
                    operator, damping=damping,
                    diagonal_only=self.cfg.diagonal_preconditioner)),
                max_iters=self.cfg.pcg_max_iters, tol=self.cfg.pcg_tol,
                true_residual=self.cfg.pcg_true_residual,
                progress=(_pcg_progress(outer, self.cfg.pcg_max_iters, started)
                          if self.cfg.verbose else None))
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
            # d' A0 d with the damping removed: A_damped = A0 + lambda*D, so
            # A0 d = A_damped d - lambda * D d exactly.
            Ad = operator.information_matvec(step)
            if damping:
                Ad = Ad - damping * operator.curvature_scale() * step
            curvature_quadratic = float((step * Ad).sum())
            # 3.4: scaled projected stationarity ||D^-1/2 g||, with D the
            # undamped curvature diagonal. A raw gradient norm mixes metres,
            # radians and rates and cannot be compared across directions.
            scaled_gradient = float(
                (b / operator.curvature_scale().clamp_min(1e-30).sqrt()).norm())
            step_norm = float(step.norm())
            if initial_grad is None:
                initial_grad = grad_norm
            # ||g_final|| / ||g_initial|| is kept as a DIAGNOSTIC and no
            # longer decides anything. It made the verdict a property of
            # where the run started: enlarging a fixture's initialisation
            # noise moved the ratio across the threshold without changing the
            # point the solver stopped at.
            decayed = grad_norm <= self.cfg.grad_decay_tol * max(initial_grad, 1e-30)
            # The certificate instead. For an exact Newton step d = A^-1 b,
            #     b.d - 0.5 d'A d = 0.5 b'A^-1 b
            # is the quadratic model's prediction of ALL the reduction that
            # remains -- the Newton decrement, which depends on the current
            # point alone. Below a tolerance times the objective, the model
            # says there is nothing left to gain and the state is stationary;
            # above it the solver stopped somewhere the model still expects
            # progress, which is a stall however small the step was.
            predicted_full = -directional - 0.5 * curvature_quadratic
            objective_scale = max(abs(energy["E_total"]), 1.0)
            decrement_ratio = predicted_full / objective_scale
            noise_floor = self.cfg.noise_factor * objective_noise
            certified = (0.0 <= predicted_full
                         <= max(self.cfg.decrement_tol * objective_scale,
                                noise_floor))
            # The certificate has to be VISIBLE. Breaking here used to leave
            # the last trace row belonging to the previous iteration, so the
            # numbers a CONVERGED verdict rests on were not in the record.
            def terminal(verdict):
                trace.append({
                    "outer": outer, "accepted": False, "terminal": True,
                    "status": verdict, "damping": damping,
                    "grad_norm": grad_norm,
                    "scaled_projected_gradient": scaled_gradient,
                    "curvature_quadratic": curvature_quadratic,
                    "step_norm": step_norm, "directional": directional,
                    "predicted_full_reduction": predicted_full,
                    "decrement_ratio": decrement_ratio,
                    "decrement_threshold": max(
                        self.cfg.decrement_tol * objective_scale, noise_floor),
                    "objective_noise": objective_noise,
                    "certified": certified,
                    "gradient_decay_ratio": grad_norm / max(initial_grad, 1e-30),
                    "pcg_iterations": solve_info["iterations"],
                    "pcg_residual": solve_info["residual"],
                    "pcg_true_residual": solve_info.get("true_residual"),
                    "pcg_status": solve_info["status"], **energy})

            if step_norm < self.cfg.step_tol and certified:
                status = CONVERGED
                terminal(status)
                break
            if step_norm < self.cfg.step_tol:
                tiny_steps += 1
                if tiny_steps >= self.cfg.tiny_step_patience:
                    status = STALLED
                    terminal(status)
                    break
            else:
                tiny_steps = 0


            accepted = False
            trials = []
            for scale in self.cfg.backtracking:
                candidate = current.retract(step, scale)
                trial_energy = self.evaluate(candidate, with_support=True)
                threshold = (energy["E_total"]
                             + self.cfg.armijo_c * scale * directional)
                ok = (math.isfinite(trial_energy["E_total"])
                      and trial_energy["E_total"] <= threshold)
                # The quadratic model's PREDICTED REDUCTION, which is not
                # the Armijo requirement. For m(d) = F + g'd + 1/2 d'A0 d the
                # predicted decrease along alpha*d is
                #
                #     pred = -alpha g'd - 0.5 alpha^2 d'A0 d
                #
                # with A0 the UNDAMPED operator. The previous code recorded
                # -armijo_c * alpha * g'd, which is the sufficient-decrease
                # threshold; calling that the model's prediction makes the
                # ratio rho meaningless. Armijo is still checked separately
                # below, as it should be.
                predicted = (-scale * directional
                             - 0.5 * scale * scale * curvature_quadratic)
                trials.append({
                    "scale": scale, "threshold": threshold,
                    "accepted": bool(ok),
                    "E_total": trial_energy["E_total"],
                    "E_rgb": trial_energy["E_rgb"],
                    "E_depth": trial_energy["E_depth"],
                    "E_process": trial_energy["E_process"],
                    "E_prior": trial_energy["E_prior"],
                    "depth_support_total": trial_energy.get(
                        "depth_support", {}).get("total"),
                    "predicted_reduction": predicted,
                    "actual_reduction": energy["E_total"] - trial_energy["E_total"],
                    # rho is only defined for a model that predicts a decrease.
                    "rho": ((energy["E_total"] - trial_energy["E_total"]) / predicted
                            if predicted > 0 else None),
                    "depth_support_change": (
                        trial_energy.get("depth_support", {}).get("total", 0)
                        - energy.get("depth_support", {}).get("total", 0)),
                    "finite": bool(math.isfinite(trial_energy["E_total"]))})
                if ok:
                    # On acceptance only: the nominal moves and the next outer
                    # iteration relinearises here.
                    current = candidate
                    energy = trial_energy
                    accepted = True
                    if self.cfg.on_accept is not None:
                        self.cfg.on_accept(outer, current, energy)
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
                      "scaled_projected_gradient": scaled_gradient,
                      "curvature_quadratic": curvature_quadratic,
                      "step_norm": step_norm, "directional": directional,
                      "predicted_full_reduction": predicted_full,
                      "decrement_ratio": decrement_ratio,
                      "decrement_threshold": max(
                          self.cfg.decrement_tol * objective_scale, noise_floor),
                      "objective_noise": objective_noise,
                      "certified": certified,
                      "gradient_decay_ratio": grad_norm / max(initial_grad, 1e-30),
                      "pcg_iterations": solve_info["iterations"],
                      "pcg_residual": solve_info["residual"],
                      "pcg_residual_kind": solve_info.get("residual_kind"),
                      "pcg_true_residual": solve_info.get("true_residual"),
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

            if accepted and trace[-1]["step_norm"] * trials[-1]["scale"] < self.cfg.step_tol:
                if certified:
                    status = CONVERGED
                    terminal(status)
                    break
                tiny_steps += 1
                if tiny_steps >= self.cfg.tiny_step_patience:
                    status = STALLED
                    terminal(status)
                    break
            if len(trace) >= 2 and trace[-2].get("accepted"):
                rel = abs(trace[-2]["E_total"] - energy["E_total"]) / max(
                    abs(trace[-2]["E_total"]), 1e-30)
                if rel < self.cfg.rel_objective_tol and not certified:
                    tiny_steps += 1
                    if tiny_steps < self.cfg.tiny_step_patience:
                        continue
                if rel < self.cfg.rel_objective_tol:
                    # A flat objective is NOT convergence on its own. If the
                    # gradient is still large the optimiser has stalled, and
                    # reporting that as CONVERGED is exactly the misreport the
                    # review warns about.
                    status = CONVERGED if certified else STALLED
                    terminal(status)
                    break

        if status == MAX_ITERS and trace and not any(r["accepted"] for r in trace):
            # A loop of rejected steps is not a budget exhaustion; saying
            # MAX_ITERS there would describe a stalled solver as one that
            # simply ran out of iterations.
            status = LINE_SEARCH_FAILED
        return {"status": status, "energy": energy, "trace": trace,
                "converged": status == CONVERGED,
                "objective_noise": objective_noise,
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


def _pcg_progress(outer, max_iters, started, every=10):
    """A line every `every` PCG iterations, so a long step is visibly alive."""
    def report(k, rel):
        if k % every == 0 or k == max_iters:
            print(f"     outer {outer:>2}  PCG {k:>3}/{max_iters}  residual "
                  f"{rel:.3e}  ({time.time() - started:.0f} s)", flush=True)
    return report


def _eliminated(operator, preconditioner):
    if preconditioner is None:
        return None

    def apply(r):
        return operator._freeze(preconditioner(r))

    return apply


class _StreamedCaches:
    """One frame's linearisation resident at a time, rebuilt on demand.

    The resident path holds every frame's frozen rasterizer cache alive for
    the whole outer iteration, which is what makes the full real window
    impossible on an 8 GiB card (see `IEKSConfig.stream_frames` for the
    measured numbers). This provider answers `caches[t]` by relinearising
    frame `t` at the SAME nominal state the resident path used, and lets the
    previous one go.

    It is not an approximation and not a reduced state: `linearize_frame` is
    the same call with the same arguments, so the operator's action is
    identical. What it costs is one render per frame per pass -- a matvec that
    used to read T cached linearisations now rebuilds them.

    `build_inventory` is deliberately NOT touched here: the factor set is
    sealed once, and relinearisation updates factors rather than adding them.
    """

    def __init__(self, observation, states):
        self.obs = observation
        self.states = list(states)
        self.rebuilds = 0

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        if isinstance(index, slice):
            raise TypeError(
                "a streamed window is read one frame at a time; slicing it "
                "would materialise exactly the residency this avoids")
        t = int(index)
        if t < 0:
            t += len(self.states)
        if not 0 <= t < len(self.states):
            raise IndexError(t)
        self.rebuilds += 1
        return self.obs.linearize_frame(self.states[t], t)


class _CudaWindowOperator(FullWindowFreeOperator):
    """FullWindowFreeOperator whose measurement blocks are the real renderer."""

    def __init__(self, *, num_frames, num_gaussians, process_factors,
                 observation, caches, prior_information, prior_residual,
                 damping, frozen_frames=(), frozen_columns=(), frozen_groups=(),
                 work_device=None):
        super().__init__(num_frames=num_frames, num_gaussians=num_gaussians,
                         process_factors=process_factors,
                         measurement_factors=[],
                         prior_information=prior_information,
                         prior_residual=prior_residual, damping=damping,
                         work_device=work_device)
        self.obs = observation
        self.caches = caches
        self.frozen = tuple(frozen_frames)
        self.frozen_groups = tuple(
            (int(t), torch.as_tensor(idx, dtype=torch.long).reshape(-1))
            for t, idx in frozen_groups)
        self.frozen_columns = tuple(frozen_columns)
        self._curvature = None
        self._measurement_blocks = None
        self._measurement_diag = None

    def _freeze(self, u: torch.Tensor) -> torch.Tensor:
        """Exact elimination of a conditionally fixed source-time pose.

        The pose columns are zeroed, not given a large precision: a fake prior
        is a soft constraint that leaks, and its magnitude would show up in
        every reported uncertainty.
        """
        if not self.frozen and not self.frozen_columns and not self.frozen_groups:
            return u
        out = u.clone()
        for t in self.frozen:
            out[t, :, 0:6] = 0.0
        for t, idx in self.frozen_groups:
            out[t, idx.to(out.device), 0:6] = 0.0
        for c in self.frozen_columns:
            out[:, :, c] = 0.0
        return out

    def measurement_block_diagonal(self) -> torch.Tensor:
        """Per-frame, per-Gaussian measurement Fisher, for preconditioning.

        Memoised: the curvature scale and the block-Jacobi preconditioner both
        ask for it within one outer iteration, and under `stream_frames` each
        ask would otherwise re-render the whole window. The operator is rebuilt
        from scratch at every outer iteration, so the memo never outlives the
        linearisation it belongs to.
        """
        if self._measurement_blocks is not None:
            return self._measurement_blocks
        blocks = torch.zeros(self.T, self.N, ERROR_DIM, ERROR_DIM,
                             dtype=torch.float64, device=self.work_device)
        for t, cache in enumerate(self.caches):
            blocks[t] = self.obs.block_fisher(cache).to(blocks.device)
        self._measurement_blocks = blocks
        return blocks

    def measurement_diagonal(self) -> torch.Tensor:
        """Just diag(H'R^-1 H) per frame, [T,N,12].

        The full blocks are [T,N,12,12]; on the real window that is 1.70 GiB
        against 142 MiB, and the LM scaling and the diagonal preconditioner
        only ever use the diagonal. One frame's blocks are materialised at a
        time and dropped, so the peak is one [N,12,12] rather than T of them.
        """
        if self._measurement_diag is not None:
            return self._measurement_diag
        if self._measurement_blocks is not None:
            self._measurement_diag = self._measurement_blocks.diagonal(
                dim1=-2, dim2=-1).clone()
            return self._measurement_diag
        diag = torch.zeros(self.T, self.N, ERROR_DIM, dtype=torch.float64,
                           device=self.work_device)
        for t, cache in enumerate(self.caches):
            diag[t] = self.obs.block_fisher(cache).diagonal(
                dim1=-2, dim2=-1).to(diag.device)
        self._measurement_diag = diag
        return diag

    def curvature_scale(self) -> torch.Tensor:
        """diag(A) per tangent coordinate, for Levenberg-Marquardt damping.

        Uniform lambda*I cannot work here. The image carries NO velocity
        Jacobian, so the velocity directions get their curvature only from the
        process term and sit around seven orders of magnitude below the pose
        directions. A lambda small enough to leave the pose block alone does
        nothing to the velocity block, the solve takes a step of norm ~40
        where the solution is 0.2 away, the velocity part explodes the process
        energy and every trial is rejected. Scaling by diag(A) damps each
        direction relative to its own curvature, which is what LM is for.
        """
        if self._curvature is None:
            diag = self.measurement_diagonal().clone()
            for factor in self.process:
                t = factor.frame
                A, B, Q = (factor.A.to(diag.device), factor.B.to(diag.device),
                           factor.Q_inv.to(diag.device))
                # einsum straight to the diagonal: the full AtQA is [N,12,12]
                # and only its 12 diagonal entries are wanted.
                diag[t] += torch.einsum("nji,njk,nki->ni", A, Q, A)
                diag[t + 1] += torch.einsum("nji,njk,nki->ni", B, Q, B)
            if self.prior_information is not None:
                diag[0] += self.prior_information.double().to(diag.device).diagonal(
                    dim1=-2, dim2=-1)
            self._curvature = diag.abs().clamp_min(1e-12)
        return self._curvature

    def information_matvec(self, u: torch.Tensor) -> torch.Tensor:
        u = self._freeze(self._check(u))
        out = torch.zeros_like(u)
        if self.prior_information is not None:
            out[0] += torch.einsum(
                "nij,nj->ni", self.prior_information.double().to(u.device), u[0])
        for t, cache in enumerate(self.caches):
            # One frame's tangent goes to the renderer's device and its
            # information comes back: the full-window vectors may live on the
            # host (solve_on_cpu) while the JVP/VJP stays on the GPU.
            ref = cache.linearization.views[0].rgb
            contribution = self.obs.information_apply(
                cache, u[t].to(device=ref.device, dtype=ref.dtype))
            out[t] += contribution.double().to(out.device)
        for factor in self.process:
            t = factor.frame
            A, B, Q = (factor.A.to(u.device), factor.B.to(u.device),
                       factor.Q_inv.to(u.device))
            Ku = (torch.einsum("nij,nj->ni", A, u[t])
                  + torch.einsum("nij,nj->ni", B, u[t + 1]))
            QKu = torch.einsum("nij,nj->ni", Q, Ku)
            out[t] += torch.einsum("nji,nj->ni", A, QKu)
            out[t + 1] += torch.einsum("nji,nj->ni", B, QKu)
        if self.damping:
            out += self.damping * self.curvature_scale() * u
        return self._freeze(out)

    def rhs(self) -> torch.Tensor:
        b = self.zeros()
        if self.prior_information is not None and self.prior_residual is not None:
            P0 = self.prior_information.double().to(b.device)
            b[0] -= torch.einsum("nij,nj->ni", P0,
                                 self.prior_residual.double().to(b.device))
        for t, cache in enumerate(self.caches):
            b[t] += self.obs.gradient(cache).double().to(b.device)
        for factor in self.process:
            t = factor.frame
            A, B, Q = (factor.A.to(b.device), factor.B.to(b.device),
                       factor.Q_inv.to(b.device))
            Qe = torch.einsum("nij,nj->ni", Q, factor.residual.double().to(b.device))
            b[t] -= torch.einsum("nji,nj->ni", A, Qe)
            b[t + 1] -= torch.einsum("nji,nj->ni", B, Qe)
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
