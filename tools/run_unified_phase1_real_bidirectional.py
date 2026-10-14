#!/usr/bin/env python
"""Real forward / backward / raw-union free-motion Phase 1, rendered and compared.

CODEX_PRIORITY_REAL_BIDIRECTIONAL_PHASE1_MERGE_AND_VISUALIZE.md sections 3-6.

Modes, one per run directory:

  render_baseline       no optimisation. Renders the real images beside the
                        forward (initial-source) and backward (final-source)
                        tracks and their raw union, at the render frames.
  forward_only          UnifiedFreeIEKS on the initial source, anchored at
                        original frame 0.
  backward_only         the same on the final source, anchored at frame 59.
  raw_union_experiment  ONE chronological full-window IEKS over the disjoint
                        union of both sources (21,338 + 24,582 on washer),
                        rendered TOGETHER by the unchanged CUDA compositor,
                        each (original_frame, camera) image entering once.
                        The initial group's pose is fixed at frame 0 and the
                        final group's at frame 59, each only for its own rows.

THE RAW UNION IS A CONTROLLED EXPERIMENT, NOT A VALID SCENE. The two static
models are independent reconstructions of the same surfaces, and the
representation gate measured 93-99% depth agreement on their shared pixels.
`--representation-policy diagnostic-only` is required for that mode and is
written into the manifest; no opacity, ownership, pruning or masking is
applied to make the union look better.

What the solver optimises: each Gaussian's (mu, R, v, omega) only. Colour,
opacity and intrinsic scales are the trained model's and are frozen.
Likelihood: GAUSSIAN_RGBD (robust=False); production's Student-t IRLS is not
a fixed scalar objective and is refused by UnifiedFreeIEKS.

Warm starts. FULL_12D_FILTER reads a bidirectional_iekf.free_state export
(--initial-states / --final-states). CENTER_ONLY_APPROXIMATE rebuilds R as the
model's own and constant, v by central differences and omega = 0 from
phase1_centers/, and is accepted only with --allow-center-only: on a CUDA
rotating fixture it stalls past a 0.825 rad sweep and washer sweeps 1.331.

Every run keeps the best accepted iterate on disk as it goes (the objective
is monotone over accepted steps), so a run that ends MAX_ITERS or STALLED
still leaves something to render. Such a state is NEVER passed on as an
accepted posterior.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODES = ("render_baseline", "forward_only", "backward_only", "raw_union_experiment")
FULL_12D = "RECOVERED_FULL_FILTER_STATE"
CENTER_ONLY = "DERIVED_12D_INITIAL_GUESS_FROM_EXISTING_TRACKS"


def parse_frames(spec: str) -> list:
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b)))
    return [int(x) for x in spec.split(",") if x.strip()]


def cam_name(c, i):
    return str(getattr(c, "image_name", f"cam_{i:03d}")).split("/")[0]


class Source:
    """One endpoint: trained appearance plus a chronological pose track."""

    def __init__(self, endpoint, model, states_dir, centers_dir, allow_center_only,
                 iteration, dt_override, source_path, keep_rows=None):
        from bidirectional_iekf.provenance import _sha256_file
        from tools.run_iekf_scene import load_scene, state_from_gaussians

        self.endpoint = endpoint
        self.model = Path(model)
        self.ply = self.model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if not self.ply.is_file():
            raise SystemExit(f"no trained PLY at {self.ply}")
        self.ply_sha256 = _sha256_file(self.ply)
        _d, _p, gaussians, scene = load_scene(str(self.model), str(source_path))
        self.cams_all = scene.getTrainCameras()
        base = state_from_gaussians(gaussians)
        self.N_full = int(base.mu.shape[0])
        self.rows = (torch.arange(self.N_full) if keep_rows is None
                     else torch.as_tensor(keep_rows, dtype=torch.long))
        r = self.rows.to(base.mu.device)
        self.base = dataclasses.replace(
            base, mu=base.mu[r], R=base.R[r], v=base.v[r], omega=base.omega[r],
            colors=base.colors[r], opacity=base.opacity[r], scales=base.scales[r])
        self.N = int(self.rows.numel())
        self.anchor_frame = None

        if states_dir and (Path(states_dir) / "full_state_manifest.json").is_file():
            from bidirectional_iekf.free_state import load_full_state
            d = load_full_state(states_dir, device="cpu", dtype=torch.float32)
            m = d["manifest"]
            if m["canonical_endpoint"] != endpoint:
                raise SystemExit(f"{states_dir} is the {m['canonical_endpoint']!r} "
                                 f"endpoint, not {endpoint!r}")
            if d["mu"].shape[1] != self.N_full:
                raise SystemExit(f"{states_dir}: {d['mu'].shape[1]} rows, model has "
                                 f"{self.N_full}")
            self.mode = FULL_12D
            self.frames = list(d["original_frame_ids"])
            self.mu, self.R = d["mu"][:, self.rows], d["R"][:, self.rows]
            self.v, self.omega = d["v"][:, self.rows], d["omega"][:, self.rows]
            self.dt = float(dt_override or m["dt_seconds"])
            self.track_source = str(states_dir)
            self.track_sha256 = __import__("hashlib").sha256(
                json.dumps(m.get("sha256", {}), sort_keys=True).encode()).hexdigest()
            self.provenance = {
                "initialization_mode": FULL_12D,
                "mu_R_v_omega": f"recorded by the legacy IEKF's output-only "
                                f"observer in {states_dir}",
                "FULL_12D_FILTER_EXPORT": "PRESENT"}
            self.track_manifest = {k: m[k] for k in (
                "direction", "rate_convention", "rotation_convention",
                "dt_seconds", "dt_is_measured", "num_frames", "num_gaussians")}
        else:
            if not allow_center_only:
                raise SystemExit(
                    f"no full-state export for the {endpoint} source "
                    f"(looked in {states_dir}). The full 12-D warm start is "
                    "required; pass --allow-center-only to run the "
                    f"{CENTER_ONLY} DIAGNOSTIC from phase1_centers/ instead.")
            cdir = Path(centers_dir)
            m = json.loads((cdir / "trajectory_manifest.json").read_text())
            if m["canonical_endpoint"] != endpoint or \
                    m.get("trajectory_order") != "original_chronological":
                raise SystemExit(f"{cdir}: wrong endpoint or order")
            files = sorted(cdir.glob("centers_frame_*.npy"))
            self.frames = [int(f.stem.split("_")[-1]) for f in files]
            mu = torch.stack([torch.from_numpy(np.load(f)).float() for f in files])
            if mu.shape[1] != self.N_full:
                raise SystemExit(f"{cdir}: {mu.shape[1]} rows, model {self.N_full}")
            mu = mu[:, self.rows]
            self.dt = float(dt_override or 1.0 / 30.0)
            v = torch.empty_like(mu)
            v[0] = (mu[1] - mu[0]) / self.dt
            v[-1] = (mu[-1] - mu[-2]) / self.dt
            v[1:-1] = (mu[2:] - mu[:-2]) / (2 * self.dt)
            self.mode = CENTER_ONLY
            self.mu, self.v = mu, v
            self.R = self.base.R.detach().cpu().float().expand(len(files), -1, -1, -1).clone()
            self.omega = torch.zeros_like(mu)
            self.track_source = str(cdir)
            import hashlib as _h
            dig = _h.sha256()
            for f in files:
                dig.update(f.name.encode())
                dig.update(np.load(f).tobytes())
            self.track_sha256 = dig.hexdigest()
            self.track_manifest = {
                "direction": m["direction"],
                "trajectory_manifest": str(cdir / "trajectory_manifest.json"),
                "centers_sha256": self.track_sha256,
                "num_center_files": len(files)}
            self.provenance = {
                "initialization_mode": CENTER_ONLY,
                "mu": ("LEGACY_IEKF centres, phase1_centers/centers_frame_NNNN.npy, "
                       "original chronological order, every PLY row, verbatim"),
                "R": ("the source PLY's own orientation at every frame -- a "
                      "provisional guess; the legacy IEKF estimated R internally "
                      "but did not export it"),
                "v": (f"central differences of the real centres over the "
                      f"declared dt = {self.dt:.6f} s (one-sided at the ends)"),
                "omega": "zero -- a provisional guess, nothing more is justified",
                "optimised_by_fused_ieks": "mu, R, v, omega for every Gaussian, "
                                           "except this source's pose at its own "
                                           "anchor frame",
                "full_state_recovery_search": (
                    "historical bidirectional/<direction>/ holds phase1_centers "
                    "only (no R, v, omega, covariance); nothing recoverable"),
                "FULL_12D_FILTER_EXPORT": "ABSENT",
            }
        self.anchor_frame = 0 if endpoint == "initial" else self.frames[-1]

    def state(self, frame, device="cuda"):
        if frame not in self.frames:
            return None
        k = self.frames.index(frame)
        return dataclasses.replace(
            self.base, mu=self.mu[k].to(device).contiguous(),
            R=self.R[k].to(device).contiguous(), v=self.v[k].to(device).contiguous(),
            omega=self.omega[k].to(device).contiguous())

    def anchor_check(self):
        """Source pose at its own frame against the trained PLY."""
        st = self.state(self.anchor_frame, device=self.base.mu.device)
        return {"anchor_original_frame": self.anchor_frame,
                "max_abs_mu_vs_ply": float((st.mu - self.base.mu).abs().max()),
                "max_abs_R_vs_ply": float((st.R - self.base.R).abs().max())}


def concat(a, b):
    from bidirectional_iekf.representation_audit import concatenate_states
    if a is None or b is None:
        return None
    return concatenate_states(a, b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="washer_103776")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True,
                    help="the scene's results root (holds static_bidi/, bidirectional/)")
    ap.add_argument("--initial-model", default=None)
    ap.add_argument("--final-model", default=None)
    ap.add_argument("--initial-states", default=None, help="full-state export, initial")
    ap.add_argument("--final-states", default=None, help="full-state export, final")
    ap.add_argument("--allow-center-only", action="store_true",
                    help="alias of --reuse-existing-phase1")
    ap.add_argument("--reuse-existing-phase1", action="store_true",
                    help="read the historical bidirectional/{forward,backward}/"
                         "phase1_centers READ ONLY and derive the 12-D initial "
                         "guess from them when no full-state export is given: "
                         "R = source PLY orientation, v = central differences "
                         "of the real centres, omega = 0. That is an optimiser "
                         "INITIAL GUESS, not something the legacy IEKF exported; "
                         "every coordinate except the source anchor pose is "
                         "then optimised")
    ap.add_argument("--resume-from", default=None,
                    help="a previous attempt directory of THIS tool; its best "
                         "accepted state is the warm start. Refused unless its "
                         "configuration fingerprint equals this run's")
    ap.add_argument("--warm-start-from", default=None,
                    help="initialise from another run's best accepted state "
                         "WITHOUT requiring the same fingerprint (e.g. a 2-camera "
                         "result seeding an 8-camera run). Recorded as "
                         "WARM_START_FROM_OTHER_RUN; a different posterior, so it "
                         "is never called a resume")
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="stop at the next outer iteration after this many "
                         "minutes; status TIME_BUDGET, best accepted state kept")
    ap.add_argument("--stop-file", default=None,
                    help="if this file appears, stop at the next outer "
                         "iteration; status STOPPED_BY_USER")
    ap.add_argument("--camera-policy-label", default="",
                    help="e.g. TWO_CAMERAS; recorded in manifest and titles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--representation-policy", default=None,
                    choices=("diagnostic-only",),
                    help="required for raw_union_experiment")
    ap.add_argument("--frames", default="0:60", help="window, e.g. 0:60 or 52:60")
    ap.add_argument("--camera-ids", default="cam_000")
    ap.add_argument("--render-frames", default="0,15,30,45,59")
    ap.add_argument("--render-camera-ids", default="cam_000,cam_002,cam_004,cam_006")
    ap.add_argument("--max-gaussians-per-source", type=int, default=0,
                    help="SMOKE TESTS ONLY: a deterministic stride subset; the "
                         "manifest says SMOKE_SUBSAMPLED")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--max-outer", type=int, default=8)
    ap.add_argument("--pcg-iters", type=int, default=60)
    ap.add_argument("--pcg-tol", type=float, default=1e-6)
    ap.add_argument("--damping", type=float, default=1e-3)
    ap.add_argument("--stream-frames", action="store_true")
    ap.add_argument("--frames-on-cpu", action="store_true")
    ap.add_argument("--process-factors-on-cpu", action="store_true")
    ap.add_argument("--diagonal-preconditioner", action="store_true")
    ap.add_argument("--linear-backend", default="legacy_reference",
                    choices=("legacy_reference", "exact_cached_structured"),
                    help="exact_cached_structured: the SAME system from linearisations "
                         "frozen once per accepted state in host RAM, a GPU process "
                         "stencil and A*delta reuse (tests/"
                         "test_exact_cached_structured_backend.py)")
    ap.add_argument("--preconditioner", default=None,
                    choices=("diagonal", "block12", "temporal_block_tridiagonal"),
                    help="numerical solver choice only; A and b are unchanged. "
                         "block12/temporal need --linear-backend exact_cached_structured "
                         "(block12 also runs on legacy)")
    ap.add_argument("--transfer-policy", default="synchronous",
                    choices=("synchronous", "double_buffered"))
    ap.add_argument("--pinned-staging-budget-mib", type=int, default=1024)
    ap.add_argument("--gpu-workspace-budget-mib", type=int, default=1024)
    ap.add_argument("--cache-host-budget-gib", type=float, default=24.0)
    ap.add_argument("--solve-on-cpu", action="store_true",
                    help="keep the full-window float64 vectors (PCG, "
                         "preconditioner, LM scale) in host memory; one frame's "
                         "tangent visits the GPU at a time. Same arithmetic "
                         "(tests/test_unified_cpu_solve.py)")
    ap.add_argument("--no-visuals", action="store_true")
    ap.add_argument("--extra-variants-from", nargs="*", default=[],
                    help="other run dirs of THIS tool whose best accepted state "
                         "should appear as a variant in the visuals")
    args = ap.parse_args()
    args.allow_center_only = bool(args.allow_center_only or args.reuse_existing_phase1)

    out = Path(args.out)
    if (out / "manifest.json").is_file():
        raise SystemExit(f"{out} already holds a run (manifest.json). Choose a new --out.")
    out_root = Path(args.out_root)
    protected = [out_root / "static_bidi", out_root / "bidirectional"]
    for p in protected:
        if p.exists() and (out.resolve() == p.resolve() or p.resolve() in out.resolve().parents):
            raise SystemExit(f"--out {out} is inside {p}; refusing to write there")
    if args.mode == "raw_union_experiment" and args.representation_policy != "diagnostic-only":
        raise SystemExit("raw_union_experiment requires --representation-policy "
                         "diagnostic-only: the union of two independently trained "
                         "mixtures is not a validated scene, and the run must say so")
    # The output directory is created only once the inputs have been
    # verified, so a refused run leaves nothing behind to be mistaken for one.
    pending = []
    log = None

    def say(msg):
        nonlocal_log = log
        print(msg)
        if nonlocal_log is None:
            pending.append(msg)
        else:
            nonlocal_log.write(msg + "\n")
            nonlocal_log.flush()

    t_start = time.time()
    say(f"=== {args.mode}: {args.scene} -> {out} ===")

    def keep(n_full):
        if not args.max_gaussians_per_source or n_full <= args.max_gaussians_per_source:
            return None
        return torch.arange(0, n_full, max(1, n_full // args.max_gaussians_per_source))[
            :args.max_gaussians_per_source]

    need_initial = args.mode in ("render_baseline", "forward_only", "raw_union_experiment")
    need_final = args.mode in ("render_baseline", "backward_only", "raw_union_experiment")
    sources = {}
    for endpoint, needed, model, states, direction in (
            ("initial", need_initial, args.initial_model, args.initial_states, "forward"),
            ("final", need_final, args.final_model, args.final_states, "backward")):
        if not needed and not args.mode == "render_baseline":
            pass
        model = model or out_root / "static_bidi" / endpoint
        from plyfile import PlyData
        n_full = len(PlyData.read(str(Path(model) / "point_cloud" /
                                      f"iteration_{args.iteration}" /
                                      "point_cloud.ply"))["vertex"])
        sources[endpoint] = Source(
            endpoint, model, states,
            out_root / "bidirectional" / direction / "phase1_centers",
            args.allow_center_only, args.iteration, args.dt, args.data_root,
            keep_rows=keep(n_full))
        s = sources[endpoint]
        say(f"  {endpoint:<8} {s.N} of {s.N_full} Gaussians, warm start {s.mode}, "
            f"frames {s.frames[0]}..{s.frames[-1]}, anchor frame {s.anchor_frame}, "
            f"sha256 {s.ply_sha256[:12]}")

    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "run.log", "a")
    log.write("\n".join(pending) + "\n")

    dts = {s.dt for s in sources.values()}
    if len(dts) != 1:
        raise SystemExit(f"the two sources use different dt {dts}; one time "
                         "convention is required for one chronological window")
    dt = dts.pop()
    cams_all = sources["initial"].cams_all
    by_name = {cam_name(c, i): c for i, c in enumerate(cams_all)}
    ext_i = [(c.R.tolist(), c.T.tolist()) for c in sources["initial"].cams_all]
    ext_f = [(c.R.tolist(), c.T.tolist()) for c in sources["final"].cams_all]
    gauge_ok = ext_i == ext_f
    say(f"  camera gauge identical across the two models: {gauge_ok}")
    if not gauge_ok:
        raise SystemExit("the two models' cameras differ; one world gauge is required")

    from estimator.sparse_iekf import IEKFConfig
    from tools.run_iekf_scene import camera_specs, dataset_rgbd_frames
    cfg = IEKFConfig()

    def load_real(frames, names):
        cams = [by_name[n] for n in names]
        out_frames = {}
        for f in frames:
            ns = argparse.Namespace(
                sequence_start=f, sequence_length=1, all_frames=False,
                observation_source="dataset_rgbd", cameras=len(cams),
                all_cameras=False, window_length=0, dt=dt,
                frames_on_cpu=True, frames_on_gpu=False, max_gaussians=0, seed=0)
            raw, _ = dataset_rgbd_frames(cams, cfg, ns, None, Path(args.data_root),
                                         direction="forward")
            out_frames[f] = raw[0][0]
        return out_frames, camera_specs(cams, len(cams))

    manifest = {
        "scene": args.scene, "mode": args.mode, "argv": sys.argv,
        "representation_policy": args.representation_policy,
        "smoke_subsampled": bool(args.max_gaussians_per_source),
        "likelihood": "GAUSSIAN_RGBD (robust=False)",
        "dt_seconds": dt, "dt_is_measured": False,
        "dt_note": ("fps: null and duration_sec: null in the dataset; dt is the "
                    "declared reconstruction convention"),
        "camera_gauge_identical": gauge_ok,
        "frozen_attributes": "colors (SH DC), opacity, intrinsic scales",
        "optimised": "(mu, R, v, omega) per Gaussian",
        "sources": {e: {"model": str(s.model), "ply_sha256": s.ply_sha256,
                        "num_gaussians": s.N, "num_gaussians_in_model": s.N_full,
                        "rows_sha256": __import__("hashlib").sha256(
                            s.rows.numpy().tobytes()).hexdigest(),
                        "warm_start": s.mode, "track_source": s.track_source,
                        "track": s.track_manifest,
                        "anchor": s.anchor_check()}
                    for e, s in sources.items()},
    }
    manifest["camera_policy"] = args.camera_policy_label or None
    (out / "initialization_provenance.json").write_text(json.dumps({
        "scene": args.scene,
        "historical_inputs_read_only": True,
        "per_source": {e: {"endpoint": e, "model": str(s.model),
                           "ply_sha256": s.ply_sha256, "track": s.track_source,
                           "track_sha256": s.track_sha256,
                           "anchor_original_frame": s.anchor_frame,
                           **s.provenance} for e, s in sources.items()},
        "note": ("A correct 12-D MAP problem needs a valid 12-D state to "
                 "evaluate, not a 12-D estimate from a previous filter; model "
                 "correctness, initialisation quality and solver convergence "
                 "are separate questions and are reported separately"),
    }, indent=1))
    for e, s in sources.items():
        a = manifest["sources"][e]["anchor"]
        say(f"  {e} pose at its own frame {a['anchor_original_frame']} vs PLY: "
            f"|dmu| {a['max_abs_mu_vs_ply']:.2e}, |dR| {a['max_abs_R_vs_ply']:.2e}")

    # ---- the solve ----------------------------------------------------------
    if args.frames == "all":
        common = sorted(set(sources["initial"].frames) & set(sources["final"].frames))
        window = common
    else:
        window = parse_frames(args.frames)
    solved = None
    if args.mode != "render_baseline":
        from bidirectional_iekf.ieks import (IEKSConfig, Trajectory, UnifiedFreeIEKS,
                                             save_run)
        from bidirectional_iekf.observations import FreeRGBDObservation
        from estimator.covariance import invert_blocks, make_process_noise
        from estimator.state import matrix_to_quaternion_wxyz

        names = [n.strip() for n in args.camera_ids.split(",") if n.strip()]
        missing = [n for n in names if n not in by_name]
        if missing:
            raise SystemExit(f"unknown camera ids {missing}; have {sorted(by_name)}")
        if args.mode == "forward_only":
            groups = {"initial": sources["initial"]}
        elif args.mode == "backward_only":
            groups = {"final": sources["final"]}
        else:
            groups = {"initial": sources["initial"], "final": sources["final"]}
        for e, s in groups.items():
            outside = [f for f in window if f not in s.frames]
            if outside:
                raise SystemExit(f"{e} track has no frames {outside[:5]}")
            if s.anchor_frame not in window:
                raise SystemExit(
                    f"the {e} anchor (original frame {s.anchor_frame}) is outside "
                    f"the window {window[0]}..{window[-1]}. Include it: a partial "
                    "window without its source frame needs a boundary prior this "
                    "tool does not propagate, and fixing a pose at a non-anchor "
                    "frame would manufacture certainty.")
        if window != list(range(window[0], window[-1] + 1)):
            raise SystemExit("the window must be contiguous; a temporal subset needs "
                             "a composed multi-step transition this tool does not build")

        say(f"  window: original frames {window[0]}..{window[-1]} ({len(window)}), "
            f"cameras {names}")
        cams = [by_name[n] for n in names]
        ns = argparse.Namespace(
            sequence_start=window[0], sequence_length=len(window), all_frames=False,
            observation_source="dataset_rgbd", cameras=len(cams), all_cameras=False,
            window_length=0, dt=dt, frames_on_cpu=bool(args.frames_on_cpu),
            frames_on_gpu=not args.frames_on_cpu, max_gaussians=0, seed=0)
        raw, prov = dataset_rgbd_frames(cams, cfg, ns, None, Path(args.data_root),
                                        direction="forward")
        if [f["dataset_frame_index"] for f in prov["frames"]] != window:
            raise SystemExit("loader frames do not match the window")
        specs = raw[0][1]
        obs = FreeRGBDObservation(
            specs, [pc for pc, _s, _d in raw], frame_ids=window,
            sigma_rgb=cfg.rgb_sigma,
            depth_sigma_model={"kind": cfg.depth_sigma_model,
                               "a": cfg.depth_sigma_a, "b": cfg.depth_sigma_b},
            robust=False, eps_depth=cfg.eps_depth,
            min_alpha_for_depth=cfg.min_alpha_for_depth)
        inventory = obs.build_inventory()
        say(f"  sealed inventory: {inventory['num_factors']} factors, one per "
            f"(original_frame, camera)")

        order = list(groups)
        offsets, start = {}, 0
        for e in order:
            offsets[e] = torch.arange(start, start + groups[e].N)
            start += groups[e].N
        N = start
        states = []
        for f in window:
            parts = [groups[e].state(f) for e in order]
            states.append(parts[0] if len(parts) == 1 else concat(*parts))
        warm = Trajectory(states=states, frame_ids=window, dt=dt)

        # The configuration fingerprint: everything that defines the
        # POSTERIOR (sources, tracks, rows, window, cameras, likelihood, Q, dt,
        # anchors, the solver's own source files) and nothing that only
        # budgets the solve (outer iterations, PCG iterations, time).
        import hashlib as _h
        code = _h.sha256()
        for rel in ("bidirectional_iekf/ieks.py", "bidirectional_iekf/observations.py",
                    "bidirectional_iekf/full_window_free.py",
                    "bidirectional_iekf/process_factors.py",
                    "bidirectional_iekf/accelerated.py"):
            code.update((REPO / rel).read_bytes())
        fp_payload = {
            "mode": args.mode, "window": window, "cameras": names,
            "sources": {e: [s.ply_sha256, s.track_sha256,
                            manifest["sources"][e]["rows_sha256"], s.mode]
                        for e, s in sources.items() if e in groups},
            "dt": dt, "likelihood": "GAUSSIAN_RGBD",
            "Q": [cfg.process_noise_pos, cfg.process_noise_rot,
                  cfg.process_noise_vel, cfg.process_noise_omega],
            "sensor": [cfg.rgb_sigma, cfg.depth_sigma_model, cfg.depth_sigma_a,
                       cfg.min_alpha_for_depth],
            "anchors": [(e, groups[e].anchor_frame) for e in order],
            "smoke_subsample": args.max_gaussians_per_source,
            "linear_backend": args.linear_backend,
            "preconditioner": args.preconditioner,
            "solver_code_sha256": code.hexdigest()}
        fingerprint = _h.sha256(json.dumps(fp_payload, sort_keys=True,
                                           default=str).encode()).hexdigest()
        manifest["config_fingerprint"] = fingerprint
        manifest["config_fingerprint_payload"] = fp_payload
        say(f"  configuration fingerprint {fingerprint[:16]}")
        # Written now, so an attempt that dies mid-solve can still be
        # validated and resumed from its best accepted state.
        tmp = out / "manifest.partial.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=1, default=str))
        tmp.replace(out / "manifest.partial.json")

        if args.resume_from and args.warm_start_from:
            raise SystemExit("--resume-from and --warm-start-from are exclusive")
        if args.resume_from or args.warm_start_from:
            prev = Path(args.resume_from or args.warm_start_from)
            pm = {}
            for name in ("manifest.json", "manifest.partial.json"):
                if (prev / name).is_file():
                    pm = json.loads((prev / name).read_text())
                    break
            if args.resume_from and pm.get("config_fingerprint") != fingerprint:
                raise SystemExit(
                    f"--resume-from {prev}: its configuration fingerprint "
                    f"{str(pm.get('config_fingerprint'))[:16]} differs from this "
                    f"run's {fingerprint[:16]}. Resuming would continue a "
                    "DIFFERENT posterior; refused.")
            npz = prev / "best_accepted" / "state.npz"
            if not npz.is_file():
                raise SystemExit(f"--resume-from {prev}: no best_accepted/state.npz")
            d = np.load(npz)
            if d["original_frame_ids"].tolist() != window or d["mu"].shape[1] != N:
                raise SystemExit(f"{npz}: frames or candidate count do not match")
            from estimator.state import quaternion_wxyz_to_matrix
            resumed = []
            for k, st in enumerate(states):
                resumed.append(dataclasses.replace(
                    st, mu=torch.from_numpy(d["mu"][k]).to(st.mu),
                    R=quaternion_wxyz_to_matrix(
                        torch.from_numpy(d["quat_wxyz"][k]).to(st.mu)).contiguous(),
                    v=torch.from_numpy(d["v"][k]).to(st.mu),
                    omega=torch.from_numpy(d["omega"][k]).to(st.mu)))
            warm = Trajectory(states=resumed, frame_ids=window, dt=dt)
            manifest["resumed_from" if args.resume_from else
                     "warm_start_from_other_run"] = {"dir": str(prev),
                                                     "outer": int(d["outer"]),
                                        "state_sha256": _h.sha256(
                                            npz.read_bytes()).hexdigest()}
            say(f"  {'RESUMED' if args.resume_from else 'WARM_START_FROM_OTHER_RUN'}"
                f" from {prev} (its accepted outer {int(d['outer'])})")
        anchor_groups = tuple((window.index(groups[e].anchor_frame), offsets[e])
                              for e in order)
        say(f"  {N} candidates; anchors " + ", ".join(
            f"{e} rows at original frame {groups[e].anchor_frame}" for e in order))

        Q = make_process_noise(N, dt, cfg.process_noise_pos, cfg.process_noise_rot,
                               cfg.process_noise_vel, cfg.process_noise_omega,
                               device=states[0].mu.device, dtype=torch.float64)
        Q_inv, _ = invert_blocks(Q.double())
        del Q

        deadline = (time.time() + 60.0 * args.time_budget_min
                    if args.time_budget_min > 0 else None)

        def stop_check():
            if args.stop_file and Path(args.stop_file).exists():
                return "STOPPED_BY_USER"
            if deadline is not None and time.time() > deadline:
                return "TIME_BUDGET"
            return None

        best_dir = out / "best_accepted"
        accept_log = []

        def on_accept(outer, traj, energy):
            best_dir.mkdir(parents=True, exist_ok=True)
            mu = torch.stack([s.mu for s in traj.states]).cpu().numpy()
            q = torch.stack([matrix_to_quaternion_wxyz(s.R) for s in traj.states]).cpu().numpy()
            v = torch.stack([s.v for s in traj.states]).cpu().numpy()
            w = torch.stack([s.omega for s in traj.states]).cpu().numpy()
            tmp = best_dir / "state.npz.tmp.npz"
            np.savez(tmp, mu=mu, quat_wxyz=q, v=v, omega=w,
                     original_frame_ids=np.asarray(window), outer=outer)
            tmp.replace(best_dir / "state.npz")
            accept_log.append({"outer": outer, **{k: energy[k] for k in (
                "E_total", "E_rgb", "E_depth", "E_process", "E_prior")},
                "wall_s": time.time() - t_start})
            (best_dir / "accepted.json").write_text(json.dumps(accept_log, indent=1))
            say(f"    accepted outer {outer}: E {energy['E_total']:.6e} "
                f"(rgb {energy['E_rgb']:.3e} depth {energy['E_depth']:.3e} "
                f"proc {energy['E_process']:.3e})  {time.time() - t_start:.0f} s")

        ieks = UnifiedFreeIEKS(obs, Q_inv, config=IEKSConfig(
            max_outer=args.max_outer, damping=args.damping,
            pcg_max_iters=args.pcg_iters, pcg_tol=args.pcg_tol,
            pcg_true_residual=True, source_pose_policy="fixed_conditional",
            anchor_groups=anchor_groups, stream_frames=args.stream_frames,
            process_factors_on_cpu=args.process_factors_on_cpu,
            diagonal_preconditioner=args.diagonal_preconditioner,
            solve_on_cpu=args.solve_on_cpu,
            linear_backend=args.linear_backend,
            preconditioner=args.preconditioner,
            transfer_policy=args.transfer_policy,
            pinned_budget_mib=args.pinned_staging_budget_mib,
            gpu_workspace_budget_mib=args.gpu_workspace_budget_mib,
            cache_host_budget_gib=args.cache_host_budget_gib,
            on_accept=on_accept, stop_check=stop_check, verbose=True))
        torch.cuda.reset_peak_memory_stats()
        import resource
        try:
            result = ieks.run(warm)
            failure = None
        except torch.cuda.OutOfMemoryError as exc:
            result, failure = None, f"OOM: {str(exc)[:300]}"
        except MemoryError as exc:
            result, failure = None, str(exc)[:500]
        peak_gpu = torch.cuda.max_memory_allocated() / 2 ** 20
        peak_host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if result is not None:
            save_run(out, result)
            with open(out / "trace.jsonl", "w") as fh:
                for r in result["trace"]:
                    fh.write(json.dumps(r, default=float) + "\n")
            solved = {"trajectory": result["trajectory"], "order": order,
                      "offsets": offsets, "trace": result["trace"]}
            last = result["trace"][-1] if result["trace"] else {}
            manifest["solve"] = {
                "status": result["status"],
                "converged_certified": result["status"] == "CONVERGED",
                "energy_start": result["trace"][0] if False else None,
                "energy_final": result["energy"],
                "outer_iterations": len(result["trace"]),
                "accepted": sum(1 for r in result["trace"] if r.get("accepted")),
                "last": {k: last.get(k) for k in (
                    "pcg_iterations", "pcg_residual", "pcg_true_residual",
                    "pcg_status", "damping", "decrement_ratio",
                    "decrement_threshold", "certified", "step_norm")},
                "best_accepted_dir": str(best_dir) if accept_log else None,
                "wall_s": time.time() - t_start,
                "peak_gpu_mib": peak_gpu, "peak_host_rss_mib": peak_host,
                "inventory_factors": inventory["num_factors"],
                "frame_provenance": prov,
                "groups": {e: {"rows": [int(offsets[e][0]), int(offsets[e][-1]) + 1],
                               "anchor_original_frame": groups[e].anchor_frame}
                           for e in order}}
            say(f"  status {result['status']}  E {result['energy']['E_total']:.6e}  "
                f"true PCG residual {last.get('pcg_true_residual')}  "
                f"{manifest['solve']['wall_s']:.0f} s, GPU {peak_gpu:.0f} MiB, "
                f"host {peak_host:.0f} MiB")
        else:
            manifest["solve"] = {"status": ("CACHE_HOST_BUDGET_EXCEEDED"
                                            if failure.startswith("CACHE_HOST")
                                            else "OOM"), "error": failure,
                                 "peak_gpu_mib": peak_gpu,
                                 "best_accepted_dir": str(best_dir) if accept_log else None}
            say(f"  {failure}")

    # ---- flags ----------------------------------------------------------------
    st = (manifest.get("solve") or {}).get("status")
    flags = {
        "INPUTS_VERIFIED": "PASS" if gauge_ok else "FAIL",
        "FULL_STATE_FWD": "PASS" if sources["initial"].mode == FULL_12D else "NOT_RUN",
        "FULL_STATE_BWD": "PASS" if sources["final"].mode == FULL_12D else "NOT_RUN",
        "SMOKE_SUBSAMPLED": bool(args.max_gaussians_per_source),
    }
    if args.mode == "render_baseline":
        flags.update(FWD_RENDER_BASELINE="PASS", BWD_RENDER_BASELINE="PASS",
                     RAW_UNION_RENDER_EXPERIMENT="PASS")
    if args.mode == "forward_only":
        flags["SINGLE_SOURCE_FWD_IEKS"] = st
    if args.mode == "backward_only":
        flags["SINGLE_SOURCE_BWD_IEKS"] = st
    if args.mode == "raw_union_experiment":
        flags.update(
            RAW_UNION_IEKS_EXECUTED="PASS" if st and st != "OOM" else "FAIL",
            RAW_UNION_IEKS_CONVERGED="PASS" if st == "CONVERGED" else (st or "NOT_RUN"),
            VALID_CANONICAL_REPRESENTATION="UNRESOLVED",
            REAL_MERGED_SCENE_VALIDATED="NOT_ESTABLISHED")
    manifest["flags"] = flags

    # ---- visuals ----------------------------------------------------------------
    if not args.no_visuals:
        from bidirectional_iekf.real_visuals import build_report
        rframes = parse_frames(args.render_frames)
        rnames = [n.strip() for n in args.render_camera_ids.split(",") if n.strip()]
        real, rspecs = load_real(rframes, rnames)
        I, F = sources["initial"], sources["final"]
        variants = {
            f"F-only track ({I.mode.split('_')[0]})": lambda f: I.state(f),
            f"B-only track ({F.mode.split('_')[0]})": lambda f: F.state(f),
            "F+B raw union, before IEKS": lambda f: concat(I.state(f), F.state(f)),
        }

        def from_solved(sol, want):
            traj = sol["trajectory"]
            def get(f):
                if f not in traj.frame_ids:
                    return None
                s = traj.states[traj.frame_ids.index(f)]
                if want == "all":
                    return s
                from bidirectional_iekf.real_visuals import subset
                return subset(s, sol["offsets"][want])
            return get

        if solved is not None:
            tag = {"forward_only": "F-only IEKS, best accepted",
                   "backward_only": "B-only IEKS, best accepted",
                   "raw_union_experiment": "F+B joint IEKS, best accepted"}[args.mode]
            variants[tag] = from_solved(solved, "all")
        for extra in args.extra_variants_from:
            ex = Path(extra)
            npz = ex / "best_accepted" / "state.npz"
            em = json.loads((ex / "manifest.json").read_text())
            if not npz.is_file():
                continue
            d = np.load(npz)
            fids = d["original_frame_ids"].tolist()
            from estimator.state import quaternion_wxyz_to_matrix
            if em["mode"] == "forward_only":
                base = I
            elif em["mode"] == "backward_only":
                base = F
            else:
                base = None

            def mk(d=d, fids=fids, base=base):
                def get(f):
                    if f not in fids:
                        return None
                    k = fids.index(f)
                    mu = torch.from_numpy(d["mu"][k]).cuda()
                    Rm = quaternion_wxyz_to_matrix(torch.from_numpy(d["quat_wxyz"][k]).cuda())
                    if base is not None:
                        return dataclasses.replace(base.base, mu=mu, R=Rm,
                                                   v=torch.from_numpy(d["v"][k]).cuda(),
                                                   omega=torch.from_numpy(d["omega"][k]).cuda())
                    u = concat(I.base, F.base)
                    return dataclasses.replace(u, mu=mu, R=Rm,
                                               v=torch.from_numpy(d["v"][k]).cuda(),
                                               omega=torch.from_numpy(d["omega"][k]).cuda())
                return get
            variants[f"{em['mode']} best accepted ({ex.name})"] = mk()

        source_rows = {0: torch.arange(I.N), 1: torch.arange(I.N, I.N + F.N)}
        tracks = {
            "F-only track": {"frames": I.frames, "mu": I.mu.numpy(), "v": I.v.numpy(),
                             "omega": I.omega.numpy(), "source": np.zeros(I.N, int),
                             "row": I.rows.numpy()},
            "B-only track": {"frames": F.frames, "mu": F.mu.numpy(), "v": F.v.numpy(),
                             "omega": F.omega.numpy(), "source": np.ones(F.N, int),
                             "row": F.rows.numpy()},
        }
        if solved is not None:
            traj = solved["trajectory"]
            src = np.concatenate([np.full(len(solved["offsets"][e]),
                                          0 if e == "initial" else 1)
                                  for e in solved["order"]])
            rows = np.concatenate([sources[e].rows.numpy() for e in solved["order"]])
            tracks[f"{args.mode} after IEKS"] = {
                "frames": list(traj.frame_ids),
                "mu": torch.stack([s.mu for s in traj.states]).cpu().numpy(),
                "v": torch.stack([s.v for s in traj.states]).cpu().numpy(),
                "omega": torch.stack([s.omega for s in traj.states]).cpu().numpy(),
                "source": src, "row": rows}
        rep = build_report(
            out, scene=args.scene, real_frames=real, cameras=rspecs,
            camera_names=rnames, variants=variants, render_frames=rframes,
            source_rows=source_rows,
            traces=({args.mode: solved["trace"]} if solved else None),
            tracks=tracks,
            title=f"[{args.mode}] warm starts: initial {I.mode}, final {F.mode}"
                  + ("  SMOKE_SUBSAMPLED" if args.max_gaussians_per_source else ""),
            notes=[
                "F/B 'track' variants are the Phase-1 warm-start tracks, NOT "
                "an optimised posterior.",
                "'before IEKS' is a render-only concatenation of both sources.",
                "'best accepted' is the last ACCEPTED iterate of the IEKS; it is "
                "a posterior estimate only if the run's status is CONVERGED.",
                "The raw union is a diagnostic: the two models duplicate surfaces.",
            ] + ([f"CENTER_ONLY warm start for {e}: R constant, omega zero"
                  for e, s in sources.items() if s.mode == CENTER_ONLY]))
        manifest["visuals"] = {"dir": str(out / "visuals"),
                               "comparison_csv": str(out / "comparison.csv"),
                               "summary": rep["summary"]}
        flags["VISUAL_REPORT_READY"] = "PASS"
        say("  summary (mean over render frames x cameras, common domain):")
        for lab, s in rep["summary"].items():
            if s:
                say(f"    {lab:<46} RGB rmse {s['rgb_rmse_common']:.4f}  depth MAE "
                    f"{s['depth_mae_common_m']:.4f} m  depth support "
                    f"{s['depth_support_px']:.0f} px"
                    + (f"  doubly covered agree "
                       f"{s['depth_agree_frac_of_doubly_covered']:.3f}"
                       if s.get("depth_agree_frac_of_doubly_covered") is not None else ""))
            else:
                say(f"    {lab:<46} NOT AVAILABLE")

    manifest["flags"] = flags
    manifest["wall_s_total"] = time.time() - t_start
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, default=str))
    tmp.replace(out / "manifest.json")
    say(f"  flags: {json.dumps(flags)}")
    say(f"  {out}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
