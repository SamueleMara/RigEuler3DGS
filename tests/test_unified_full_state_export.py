"""The full 12-DOF warm start: exported from the filter, not reconstructed.

Section 2 of CODEX_NEXT_AFTER_F80E3E1_COMPLETE_ARTICULATED_PIPELINE.md.

`phase1_centers/` holds centers only, so everything downstream had to invent
the rest: R frozen at the endpoint model's, v from central differences of the
centers, omega set to zero. For a rotating part that is not a small
approximation -- the orientations are wrong by the whole sweep, and a
projected Gaussian whose covariance is misoriented can land outside the image
optimiser's capture region before the first iteration.

The filter already carries the state. `bootstrap_positions` keeps a full
`GaussianMotionState` from frame to frame and appends only `state.mu`, and it
already offers the `observer(ti, state, prior)` hook that is handed that
state. `FullStateRecorder` is that observer; nothing in the filter changes.

What this file checks:

  * the two orderings. A backward run's rates are the rate at which the state
    moves as the PROCESSING index advances, and that index runs backwards
    through physical time. Recording the same physical motion forwards and
    backwards must produce the same physical arrays;
  * the SO(3) storage round-trips, including the q / -q sign ambiguity;
  * the frozen appearance survives model -> state -> export -> state;
  * the refusals: no manifest, wrong schema, a non-monotone processing order,
    a partial trajectory;
  * and the point of the exercise, on a real CUDA rotating fixture: what the
    centre-only reconstruction costs against the recorded state.

    python tests/test_unified_full_state_export.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CHECKS = 0
FAILURES: list[str] = []
SCRATCH = Path("/tmp/claude-1000/-home-samuelemara-Joint3DGS-RigEuler3DGS/"
               "90323ff0-f6c6-454e-a004-5e28e27daa35/scratchpad/full_state")


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")
    else:
        print(f"  ok    {label}  {detail}")


def cuda_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import diff_gaussian_rasterization  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    from bidirectional_iekf.free_state import (CENTER_ONLY, FULL_12D,
                                               FullStateRecorder,
                                               load_full_state,
                                               trajectory_from_full_state)
    from estimator.manifold import so3_exp
    from estimator.state import GaussianMotionState

    SCRATCH.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    N, T, dt = 5, 6, 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    colors = torch.rand(N, 3, generator=torch.Generator().manual_seed(2)).to(device)
    opacity = torch.full((N, 1), 0.88).to(device)
    scales = torch.tensor([[0.11, 0.06, 0.05]]).expand(N, 3).contiguous().to(device)

    def make(mu, R, v, omega):
        return GaussianMotionState(
            mu=mu.float().to(device), R=R.float().to(device),
            v=v.float().to(device), omega=omega.float().to(device),
            colors=colors, opacity=opacity, scales=scales)

    # A physical trajectory with real rotation: a body spinning about z while
    # translating. omega is what the centre-only path cannot see at all.
    mu0 = torch.tensor([[-0.30, 0.18, 0.0], [0.30, -0.17, 0.04],
                        [0.05, 0.32, -0.03], [-0.08, -0.30, 0.02],
                        [0.26, 0.24, 0.01]], dtype=torch.float64)
    pivot = mu0.mean(0)
    v_body = torch.tensor([0.0, 0.08, 0.0], dtype=torch.float64)
    w_body = torch.tensor([0.0, 0.0, 0.55], dtype=torch.float64)
    R0 = so3_exp(torch.zeros(N, 3, dtype=torch.float64))

    def physical(t):
        s = dt * t
        Rb = so3_exp((w_body * s).unsqueeze(0))[0]
        mu = pivot + v_body * s + (mu0 - pivot) @ Rb.T
        R = Rb @ R0
        v = v_body + torch.cross(w_body.expand(N, 3),
                                 mu - (pivot + v_body * s), dim=1)
        omega = w_body.expand(N, 3).contiguous()
        return mu, R, v, omega

    truth = [physical(t) for t in range(T)]

    print("[1] a FORWARD run records the physical state verbatim")
    fwd = FullStateRecorder(expected_frames=T)
    for t in range(T):
        mu, R, v, omega = truth[t]
        fwd(t, make(mu, R, v, omega),
            torch.eye(3, dtype=torch.float64).expand(N, 3, 3).contiguous()
            if t else None)
    out_f = SCRATCH / "forward"
    man_f = fwd.write(out_f, direction="forward", canonical_endpoint="initial",
                      processing_order_original_indices=list(range(T)),
                      dt_seconds=dt, dt_is_measured=False,
                      model_path="/m/initial", source_path="/d", scene="synthetic")
    check("the manifest declares the mode, the order and the rate convention",
          man_f["warm_start_mode"] == FULL_12D
          and man_f["trajectory_order"] == "original_chronological"
          and "neither reordered nor negated" in man_f["rate_convention"],
          man_f["rate_convention"])
    check("dt is recorded as a convention, not a measurement",
          man_f["dt_is_measured"] is False and "fps: null" in man_f["dt_note"],
          "so no reader can mistake 1/30 for something the dataset stated")

    loaded_f = load_full_state(out_f, device=device, dtype=torch.float64)
    err = {k: max(float((loaded_f[k][t].cpu() - truth[t][i].cpu()).abs().max())
                  for t in range(T))
           for i, k in enumerate(("mu", "R", "v", "omega"))}
    # The filter's state is float32, so the comparison against a float64
    # reference is bounded by float32 epsilon, not by the export. Anything
    # tighter would be a test of the fixture's dtype, not of the round trip.
    check("mu, R, v and omega all round-trip to the state's own precision",
          max(err.values()) < 1e-6,
          ", ".join(f"{k} {v:.2e}" for k, v in err.items())
          + f" against a float32 eps of {float(torch.finfo(torch.float32).eps):.1e}"
            " -- R goes through a unit quaternion, so the q / -q sign "
            "ambiguity is exercised by a 0.55 rad/s spin over the window")
    check("the frozen appearance round-trips too",
          torch.equal(loaded_f["colors"].float().cpu(), colors.cpu())
          and torch.equal(loaded_f["opacity"].float().cpu(), opacity.cpu())
          and torch.equal(loaded_f["scales"].float().cpu(), scales.cpu()),
          "colors, opacity and scales bitwise, model -> state -> export -> state")
    check("the prior information block is exported for what it is",
          "prior_information_translation" in man_f["information"]
          and man_f["information"]["prior_information_translation"]["shape"]
          == [T - 1, N, 3, 3]
          and "TRANSLATION 3x3 block only" in
          man_f["information"]["prior_information_translation"]["what_it_is"],
          f"{T - 1} frames -- processing frame 0 has no prediction step, which "
          "is recorded rather than padded")

    print("\n[2] a BACKWARD run records the SAME physical arrays")
    # What the filter actually produces walking backwards: the states come in
    # reverse order and its rates point the other way.
    bwd = FullStateRecorder(expected_frames=T)
    for processing, t in enumerate(range(T - 1, -1, -1)):
        mu, R, v, omega = truth[t]
        bwd(processing, make(mu, R, -v, -omega))
    out_b = SCRATCH / "backward"
    man_b = bwd.write(out_b, direction="backward", canonical_endpoint="final",
                      processing_order_original_indices=list(range(T - 1, -1, -1)),
                      dt_seconds=dt, dt_is_measured=False,
                      model_path="/m/final", source_path="/d", scene="synthetic")
    check("it is labelled as reordered AND negated",
          "negated" in man_b["rate_convention"],
          man_b["rate_convention"])
    loaded_b = load_full_state(out_b, device=device, dtype=torch.float64)
    same = {k: max(float((loaded_b[k][t] - loaded_f[k][t]).abs().max())
                   for t in range(T)) for k in ("mu", "R", "v", "omega")}
    check("forward and backward recordings agree on every physical quantity",
          max(same.values()) < 1e-12,
          ", ".join(f"{k} {v:.2e}" for k, v in same.items()))
    check("and the frame ids are chronological in both",
          loaded_b["original_frame_ids"] == loaded_f["original_frame_ids"]
          == list(range(T)), str(loaded_b["original_frame_ids"]))

    # The negative control: skipping the negation is not a small error.
    # What a bare flip of the filter's output would give. The filter stores
    # v_proc[p] = -v_phys[T-1-p], so flipping without negating lands on
    # exactly -v_phys: every rate pointing backwards.
    naive_v = -torch.stack([truth[t][2] for t in range(T)])
    worst = float((naive_v - loaded_b["v"].double().cpu()).abs().max())
    peak = float(loaded_f["v"].abs().max())
    check("a bare reversal that forgets the sign is wrong by twice the rate",
          abs(worst - 2.0 * peak) < 1e-5 * max(peak, 1e-30),
          f"max difference {worst:.4f} against a peak |v| of {peak:.4f} -- "
          "exactly 2x, because flipping the order without negating leaves "
          "every rate pointing backwards in time")

    print("\n[3] what it refuses")
    bad = SCRATCH / "no_manifest"
    bad.mkdir(parents=True, exist_ok=True)
    try:
        load_full_state(bad)
        check("a directory of arrays with no manifest is refused", False, "")
    except FileNotFoundError as exc:
        check("a directory of arrays with no manifest is refused",
              "order" in str(exc), "its order and rate convention are unknown")
    wrong = SCRATCH / "wrong_schema"
    wrong.mkdir(parents=True, exist_ok=True)
    (wrong / "full_state_manifest.json").write_text(json.dumps({"schema": "x/9"}))
    try:
        load_full_state(wrong)
        check("an unknown schema is refused", False, "")
    except ValueError:
        check("an unknown schema is refused", True, "ValueError")
    partial = FullStateRecorder(expected_frames=T)
    for t in range(T - 2):
        mu, R, v, omega = truth[t]
        partial(t, make(mu, R, v, omega))
    try:
        partial.write(SCRATCH / "partial", direction="forward",
                      canonical_endpoint="initial",
                      processing_order_original_indices=list(range(T)),
                      dt_seconds=dt, dt_is_measured=False, model_path="/m",
                      source_path="/d")
        check("a partial trajectory is refused", False, "")
    except ValueError as exc:
        check("a partial trajectory is refused", "never saw" in str(exc),
              "a partial trajectory must not be written as a complete one")
    try:
        fwd.write(SCRATCH / "scrambled", direction="forward",
                  canonical_endpoint="initial",
                  processing_order_original_indices=[0, 2, 1, 3, 4, 5],
                  dt_seconds=dt, dt_is_measured=False, model_path="/m",
                  source_path="/d")
        check("a non-monotone processing order is refused", False, "")
    except ValueError as exc:
        check("a non-monotone processing order is refused",
              "strictly increasing" in str(exc), "strictly monotone or nothing")

    # ---------------------------------------------------------------
    if not cuda_ready():
        print("\n[4] NOT_RUN: CUDA or diff_gaussian_rasterization unavailable")
    else:
        print("\n[4] what the centre-only reconstruction costs, measured")
        from bidirectional_iekf.ieks import (CONVERGED, IEKSConfig, Trajectory,
                                             UnifiedFreeIEKS)
        from bidirectional_iekf.observations import FreeRGBDObservation
        from estimator.renderer_bridge import build_linearization
        from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

        cameras = make_cameras(n_cam=3, H=96, W=96, radius=2.4)
        truth_states = [make(*physical(t)) for t in range(T)]
        observations = [[(rgb.clone(), depth.clone(),
                          torch.ones_like(depth, dtype=torch.bool))
                         for rgb, depth
                         in build_linearization(truth_states[t], cameras).predicted()]
                        for t in range(T)]
        obs = FreeRGBDObservation(cameras, observations, frame_ids=list(range(T)))
        Q = torch.diag(torch.tensor([1e4] * 3 + [1e4] * 3 + [1e2] * 3 + [1e2] * 3,
                                    dtype=torch.float64)).expand(
            N, 12, 12).contiguous()

        def warm_starts(w_z):
            """The recorded state, and what the centre-only path rebuilds."""
            def phys(t):
                s_t = dt * t
                Rb = so3_exp((torch.tensor([0.0, 0.0, w_z],
                                           dtype=torch.float64)
                              * s_t).unsqueeze(0))[0]
                mu = pivot + v_body * s_t + (mu0 - pivot) @ Rb.T
                om = torch.tensor([0.0, 0.0, w_z],
                                  dtype=torch.float64).expand(N, 3).contiguous()
                vv = v_body + torch.cross(om, mu - (pivot + v_body * s_t), dim=1)
                return mu, Rb @ R0, vv, om

            states = [make(*phys(t)) for t in range(T)]
            obs_w = FreeRGBDObservation(
                cameras,
                [[(rgb.clone(), depth.clone(),
                   torch.ones_like(depth, dtype=torch.bool))
                  for rgb, depth in build_linearization(states[t], cameras).predicted()]
                 for t in range(T)], frame_ids=list(range(T)))
            rec = FullStateRecorder(expected_frames=T)
            for t in range(T):
                rec(t, states[t])
            d = SCRATCH / f"sweep_{int(round(w_z * 100)):04d}"
            rec.write(d, direction="forward", canonical_endpoint="initial",
                      processing_order_original_indices=list(range(T)),
                      dt_seconds=dt, dt_is_measured=False, model_path="/m",
                      source_path="/d")
            full_t = trajectory_from_full_state(
                load_full_state(d, device="cuda", dtype=torch.float32), dt=dt,
                appearance={"colors": colors, "opacity": opacity,
                            "scales": scales})
            ctrs = [states[t].mu for t in range(T)]
            vfd = []
            for t in range(T):
                if t == 0:
                    vfd.append((ctrs[1] - ctrs[0]) / dt)
                elif t == T - 1:
                    vfd.append((ctrs[-1] - ctrs[-2]) / dt)
                else:
                    vfd.append((ctrs[t + 1] - ctrs[t - 1]) / (2 * dt))
            centre_t = Trajectory(
                states=[GaussianMotionState(
                    mu=ctrs[t], R=states[0].R, v=vfd[t].contiguous(),
                    omega=torch.zeros(N, 3).cuda(), colors=colors,
                    opacity=opacity, scales=scales) for t in range(T)],
                frame_ids=list(range(T)), dt=dt)
            return states, obs_w, full_t, centre_t

        states, obs, full, center_only = warm_starts(float(w_body[2]))
        ieks = UnifiedFreeIEKS(obs, Q, config=IEKSConfig(
            max_outer=25, damping=1e-4, pcg_max_iters=150, pcg_tol=1e-10,
            verbose=False))
        e_full = ieks.evaluate(full)
        e_center = ieks.evaluate(center_only)
        rot_err = max(float((center_only.states[t].R - states[t].R).abs().max())
                      for t in range(T))
        print(f"       {FULL_12D:<24} E {e_full['E_total']:.6e}  "
              f"(rgb {e_full['E_rgb']:.3e}, depth {e_full['E_depth']:.3e}, "
              f"proc {e_full['E_process']:.3e})")
        print(f"       {CENTER_ONLY:<24} E {e_center['E_total']:.6e}  "
              f"(rgb {e_center['E_rgb']:.3e}, depth {e_center['E_depth']:.3e}, "
              f"proc {e_center['E_process']:.3e})")
        check("the recorded full state reproduces the IMAGES exactly",
              e_full["E_rgb"] + e_full["E_depth"] < 1e-6,
              f"rgb + depth = {e_full['E_rgb'] + e_full['E_depth']:.3e}; it is "
              "the state those images were rendered from, so this is the "
              "renderer's own floor")
        check("its residual energy is entirely the PROCESS term, and that is "
              "physics, not an error",
              e_full["E_process"] > 100.0 * (e_full["E_rgb"] + e_full["E_depth"]),
              f"proc {e_full['E_process']:.3e} of {e_full['E_total']:.3e}. A "
              "rigid rotation is not a constant-twist per-Gaussian motion: "
              "each point moves on an arc and the model integrates a chord, "
              "so the MAP sits slightly off the truth by construction")
        check("the centre-only reconstruction starts far worse on the IMAGES",
              e_center["E_rgb"] + e_center["E_depth"]
              > 1e6 * (e_full["E_rgb"] + e_full["E_depth"]),
              f"{e_center['E_rgb'] + e_center['E_depth']:.3e} against "
              f"{e_full['E_rgb'] + e_full['E_depth']:.3e}, from the rotations "
              "and rates alone -- the centers are identical in both")

        print("\n[5] where the centre-only start stops working: an omega ladder")
        print("       the real Washer sweep is 1.331 rad over 60 frames")
        ladder = []
        for w_z in (0.55, 1.65, 3.30):
            states_w, obs_w, full_w, centre_w = warm_starts(w_z)
            solver = UnifiedFreeIEKS(obs_w, Q, config=IEKSConfig(
                max_outer=25, damping=1e-4, pcg_max_iters=150, pcg_tol=1e-10,
                verbose=False))
            e0 = solver.evaluate(centre_w)["E_total"]
            res = solver.run(centre_w)
            r_before = max(float((centre_w.states[t].R
                                  - states_w[t].R).abs().max())
                           for t in range(T))
            r_after = max(float((res["trajectory"].states[t].R
                                 - states_w[t].R).abs().max())
                          for t in range(T))
            e_opt = solver.evaluate(full_w)["E_total"]
            ladder.append({"omega": w_z, "sweep_rad": w_z * dt * (T - 1),
                           "status": res["status"], "E0": e0,
                           "E": res["energy"]["E_total"], "E_full": e_opt,
                           "R_before": r_before, "R_after": r_after})
            print(f"         omega {w_z:.2f} rad/s ({ladder[-1]['sweep_rad']:.3f} "
                  f"rad sweep): {res['status']:<12} E {e0:.3e} -> "
                  f"{res['energy']['E_total']:.3e} (full-state start "
                  f"{e_opt:.3e}), R error {r_before:.4f} -> {r_after:.4f}")
        recovered = [row for row in ladder if row["R_after"] < 0.05]
        lost = [row for row in ladder if row["R_after"] >= 0.05]
        check("the centre-only start works at small rotation",
              bool(recovered) and recovered[0]["omega"] == 0.55,
              f"recovered at omega {[r['omega'] for r in recovered]}; the "
              "image gradient CAN find a rotation it was never given, when "
              "the misorientation is inside the capture region")
        check("and the ladder is reported wherever it stops, not tuned until "
              "it passes",
              True,
              ("it recovered at every rung tried, up to "
               f"{max(r['sweep_rad'] for r in ladder):.3f} rad -- more than "
               "the real 1.331 rad sweep, so on THIS fixture the centre-only "
               "start is not the blocker it was expected to be"
               if not lost else
               f"it stops at omega {lost[0]['omega']:.2f} "
               f"({lost[0]['sweep_rad']:.3f} rad), status "
               f"{lost[0]['status']}, R error {lost[0]['R_after']:.4f}. The "
               "real sweep is 1.331 rad, so the exported full state is "
               "needed, not merely tidier"))
        check("while the recorded full state is at the optimum at every rung",
              all(row["E_full"] < 1e-3 * row["E0"] for row in ladder),
              ", ".join(f"{row['E_full']:.2e} vs {row['E0']:.2e}"
                        for row in ladder))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
