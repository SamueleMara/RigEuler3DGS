#!/usr/bin/env python
"""Orchestration for the fused (forward + backward) Phase-1 IEKS.

CODEX_IMPLEMENT_FUSED_PHASE1_REUSE_EXISTING_FB_WASHER_THEN_OVERNIGHT_ALL.md.

Two entry points, reached through thin shell wrappers in scripts/:

  scene      scripts/run_fused_phase1.sh            one scene, REUSE_EXISTING_FB
  overnight  scripts/run_fused_phase1_overnight.sh  every eligible scene, serially

The OPT-IN legacy forward/backward rerun is a separate module,
tools/legacy_phase1_rerun.py, which this file does not import or call.

The estimator is NOT here. Every optimisation and render is a subprocess of
tools/run_unified_phase1_real_bidirectional.py; this file only discovers
inputs, validates them, plans, locks, resumes and summarises. The `scene` and
`overnight` paths contain no reference to the legacy IEKF runner, static
training or run_bidirectional_all.sh, and tests/test_fused_phase1_wrappers.py
checks that by parsing this file.

Layout of one scene's experiment (everything new, nothing historical touched):

  <RESULTS_ROOT>/<SCENE>/unified_phase1_experiments/<TAG>/
    run_manifest.json              experiment fingerprint, sources, plan
    preflight.json
    status.json                    machine flags, written atomically
    render_baselines/              real | F track | B track | raw union
    optimizer/attempt_NNN/         one fused IEKS attempt each; --resume adds one
    optimizer/best_accepted_full_state.npz   best accepted state of any attempt
    optimizer/iterations.jsonl     every attempt's trace, tagged by attempt
    report/                        renders incl. the best accepted joint state
    visuals/  comparison.csv       copies of the report's, for the fixed layout
    docs/REAL_BIDIRECTIONAL_PHASE1_RESULTS.md
    docs/PHASE1_FUSED_THEORY_COMPLIANCE.{md,json}
    logs/

Resume policy (deterministic, documented):
  * an experiment directory is identified by its FINGERPRINT: source PLY and
    track hashes, frames, camera set, sensor model, Q, Gaussian population and
    the solver's code hash. The profile NAME and the time budget are not part
    of it, so a Washer test and the overnight run with the same --tag share
    one experiment and the overnight run CONTINUES it;
  * a directory whose fingerprint differs is never written: the scene is
    reported TAG_CONFIG_MISMATCH and skipped;
  * a finished stage (manifest present) is reused; the optimiser is skipped
    only when its latest attempt is CONVERGED. Otherwise --resume runs ONE new
    attempt warm-started from the best accepted state so far. MAX_ITERS,
    STALLED and TIME_BUDGET never pass silently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"
DEFAULT_DATASET = ("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/"
                   "output/preparation_8cams/partnet_subset_fixed_render_linear")
CLI = REPO / "tools" / "run_unified_phase1_real_bidirectional.py"
REPORT = REPO / "tools" / "write_fused_phase1_report.py"

# Effective settings. washer-validation and overnight are IDENTICAL in
# everything that defines the posterior; they differ only in solve budget
# (PCG iterations, time), which is why they can share one experiment.
PROFILES = {
    "washer-validation": dict(n_cams=2, max_outer=10, pcg_iters=60, pcg_tol=1e-6,
                              damping=1e-3, time_budget_min=0.0, render_cams=4,
                              subsample=0),
    # Same posterior as washer-validation; a smaller solve budget so ten
    # scenes fit a night. Measured on washer (45,920 Gaussians, 60 frames,
    # 2 cameras, solve on host): ~7.1 s per PCG iteration plus ~60 s fixed
    # per run. Larger scenes cost proportionally more per iteration, and the
    # budget then stops them with TIME_BUDGET and their best accepted state.
    "overnight": dict(n_cams=2, max_outer=10, pcg_iters=40, pcg_tol=1e-6,
                      damping=1e-3, time_budget_min=55.0, render_cams=4,
                      subsample=0),
    # bounded validation only: a stride subset, labelled SMOKE_SUBSAMPLED,
    # written under <TAG>__smoke so it can never be mistaken for a full run
    "smoke": dict(n_cams=1, max_outer=2, pcg_iters=8, pcg_tol=1e-6, damping=1e-3,
                  time_budget_min=10.0, render_cams=1, subsample=1500),
}
STREAM = ["--stream-frames", "--frames-on-cpu", "--process-factors-on-cpu",
          "--diagonal-preconditioner", "--solve-on-cpu"]

EXIT_OK, EXIT_FAILED, EXIT_BLOCKED, EXIT_PARTIAL = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


class Lock:
    """mkdir lock with a PID file; a stale lock (dead PID) is reclaimed.

    mkdir is atomic on every filesystem this runs on, including the exFAT
    results disk, where advisory file locks are not something to rely on.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.held = False

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                self.path.mkdir()
                (self.path / "pid").write_text(f"{os.getpid()}\n{time.ctime()}\n")
                self.held = True
                return
            except FileExistsError:
                pid = None
                try:
                    pid = int((self.path / "pid").read_text().split()[0])
                except Exception:
                    pass
                alive = False
                if pid:
                    try:
                        os.kill(pid, 0)
                        alive = True
                    except OSError:
                        alive = False
                if alive:
                    raise SystemExit(
                        f"LOCKED: {self.path} is held by running PID {pid}. Another "
                        "fused Phase-1 run is using this experiment (or the GPU). "
                        "Not starting a second one.")
                shutil.rmtree(self.path, ignore_errors=True)    # stale: reclaim
        raise SystemExit(f"could not acquire {self.path}")

    def release(self):
        if self.held:
            shutil.rmtree(self.path, ignore_errors=True)
            self.held = False


# ---------------------------------------------------------------------------
# discovery and eligibility
# ---------------------------------------------------------------------------

def inspect_scene(scene: str, results_root: Path, dataset_root: Path,
                  hash_sources: bool = True) -> dict:
    """Everything the fused run needs, or the precise reason it cannot run."""
    out = results_root / scene
    src = dataset_root / scene
    info = {"scene": scene, "results_dir": str(out), "dataset_dir": str(src),
            "eligible": False, "reasons": []}
    ply = {e: out / "static_bidi" / e / "point_cloud" / "iteration_30000" /
           "point_cloud.ply" for e in ("initial", "final")}
    cams_json = {e: out / "static_bidi" / e / "cameras.json" for e in ("initial", "final")}
    tracks = {d: out / "bidirectional" / d / "phase1_centers"
              for d in ("forward", "backward")}
    if not ply["initial"].is_file():
        info["reasons"].append("MISSING_INITIAL_PLY")
    if not ply["final"].is_file():
        info["reasons"].append("MISSING_FINAL_PLY")
    for d, endpoint in (("forward", "initial"), ("backward", "final")):
        m = read_json(tracks[d] / "trajectory_manifest.json")
        if m is None:
            info["reasons"].append(f"MISSING_{d.upper()}_TRACKS")
            continue
        bad = []
        if m.get("direction") != d:
            bad.append(f"direction={m.get('direction')!r}")
        if m.get("canonical_endpoint") != endpoint:
            bad.append(f"canonical_endpoint={m.get('canonical_endpoint')!r}")
        if m.get("trajectory_order") != "original_chronological":
            bad.append(f"trajectory_order={m.get('trajectory_order')!r}")
        n_frames = m.get("num_frames")
        files = sorted(tracks[d].glob("centers_frame_*.npy"))
        if not isinstance(n_frames, int) or len(files) != n_frames:
            bad.append(f"{len(files)} centre files for num_frames={n_frames}")
        if bad:
            info["reasons"].append(f"BAD_{d.upper()}_TRACK_MANIFEST: " + "; ".join(bad))
            continue
        info[f"{d}_frames"] = n_frames
        info[f"{d}_gaussians"] = m.get("num_gaussians")
    have_cams = [e for e in ("initial", "final") if cams_json[e].is_file()]
    if not have_cams:
        info["reasons"].append("MISSING_CAMERAS: neither model has cameras.json")
    info["warnings"] = [
        f"{e.upper()}_MODEL_HAS_NO_CAMERAS_JSON: camera names taken from the other "
        "model; the gauge is checked at run time from the cameras both models "
        "actually load (the CLI refuses a mismatch)"
        for e in ("initial", "final") if e not in have_cams]
    for e in ("initial", "final"):
        if not (out / "static_bidi" / e / "cfg_args").is_file():
            info["warnings"].append(
                f"{e.upper()}_MODEL_HAS_NO_CFG_ARGS: load_scene uses its defaults "
                "for this model, as the historical Phase 1 did")
    if not (src / "rgb").is_dir():
        info["reasons"].append("MISSING_RGBD: no rgb/")
    if not (src / "depth_npy").is_dir():
        info["reasons"].append("MISSING_RGBD: no depth_npy/")
    if info["reasons"]:
        return info

    if info["forward_frames"] != info["backward_frames"]:
        info["reasons"].append(
            f"FRAME_COUNT_MISMATCH: forward {info['forward_frames']}, "
            f"backward {info['backward_frames']}")
        return info
    T = info["forward_frames"]
    cams = json.loads(cams_json[have_cams[0]].read_text())
    names = [str(c["img_name"]).split("/")[0] for c in cams]
    same_gauge = True
    if len(have_cams) == 2:
        camf = json.loads(cams_json["final"].read_text())
        same_gauge = ([(c["position"], c["rotation"]) for c in cams]
                      == [(c["position"], c["rotation"]) for c in camf])
    info["gauge_checked_from"] = ("both cameras.json" if len(have_cams) == 2
                                  else "run time (one cameras.json)")
    if not same_gauge:
        info["reasons"].append("CAMERA_GAUGE_MISMATCH between the two models")
        return info
    missing_rgb = [n for n in names if not (src / "rgb" / n).is_dir()]
    if missing_rgb:
        info["reasons"].append(f"MISSING_RGBD: cameras {missing_rgb}")
        return info
    info.update(eligible=True, num_frames=T, camera_names=names,
                num_cameras=len(names),
                inputs={"initial_ply": str(ply["initial"]),
                        "final_ply": str(ply["final"]),
                        "forward_tracks": str(tracks["forward"]),
                        "backward_tracks": str(tracks["backward"])})
    if hash_sources:
        info["sha256"] = {
            "initial_ply": sha256_file(ply["initial"]),
            "final_ply": sha256_file(ply["final"]),
            "forward_trajectory_manifest": sha256_file(
                tracks["forward"] / "trajectory_manifest.json"),
            "backward_trajectory_manifest": sha256_file(
                tracks["backward"] / "trajectory_manifest.json"),
            "forward_centers": _centers_hash(tracks["forward"]),
            "backward_centers": _centers_hash(tracks["backward"]),
        }
    return info


def _centers_hash(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.glob("centers_frame_*.npy")):
        h.update(f.name.encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def discover(results_root: Path, dataset_root: Path) -> list:
    """Case-preserving scene directory names, sorted; nothing hard-coded."""
    out = []
    for d in sorted(results_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or d.name.startswith((".", "unified_phase1_overnight")):
            continue
        out.append(inspect_scene(d.name, results_root, dataset_root,
                                 hash_sources=False))
    return out


# ---------------------------------------------------------------------------
# the single-scene plan
# ---------------------------------------------------------------------------

def choose(names, n):
    if n >= len(names):
        return list(names)
    idx = sorted({round(k * len(names) / n) % len(names) for k in range(n)})
    return [names[i] for i in idx]


def scene_plan(info: dict, profile: str, tag: str, args) -> dict:
    p = PROFILES[profile]
    T = info["num_frames"]
    names = info["camera_names"]
    cams = choose(names, p["n_cams"])
    rcams = choose(names, p["render_cams"])
    rframes = sorted({0, T // 4, T // 2, (3 * T) // 4, T - 1})
    if profile == "smoke":
        rframes = [0, T - 1]
    exp_tag = tag + ("__smoke" if profile == "smoke" else "")
    exp = Path(info["results_dir"]) / "unified_phase1_experiments" / exp_tag
    label = {1: "ONE_CAMERA", 2: "TWO_CAMERAS", 4: "FOUR_CAMERAS"}.get(
        len(cams), f"{len(cams)}_CAMERAS")
    if len(cams) == len(names):
        label = "ALL_CAMERAS"
    code = hashlib.sha256()
    for rel in ("bidirectional_iekf/ieks.py", "bidirectional_iekf/observations.py",
                "bidirectional_iekf/full_window_free.py",
                "bidirectional_iekf/process_factors.py",
                "tools/run_unified_phase1_real_bidirectional.py"):
        code.update((REPO / rel).read_bytes())
    fingerprint_payload = {
        "scene": info["scene"], "frames": [0, T], "cameras": cams,
        "subsample": p["subsample"], "sources": info.get("sha256"),
        "initial_full_state": args.initial_full_state,
        "final_full_state": args.final_full_state,
        "mode": "raw_union_experiment", "likelihood": "GAUSSIAN_RGBD",
        "code_sha256": code.hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True,
                                            default=str).encode()).hexdigest()
    return {"scene": info["scene"], "profile": profile, "tag": exp_tag,
            "experiment_dir": str(exp), "num_frames": T,
            "frames": f"0..{T - 1} (all original frames, chronological)",
            "camera_ids": cams, "camera_policy": label,
            "render_frames": rframes, "render_camera_ids": rcams,
            "settings": p, "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "smoke_subsampled": bool(p["subsample"])}


def cli_base(info, args, out_dir: Path, mode: str, plan: dict) -> list:
    cmd = [sys.executable, str(CLI), "--scene", info["scene"],
           "--data-root", info["dataset_dir"], "--out-root", info["results_dir"],
           "--mode", mode, "--reuse-existing-phase1", "--frames", "all",
           "--render-frames", ",".join(map(str, plan["render_frames"])),
           "--render-camera-ids", ",".join(plan["render_camera_ids"]),
           "--camera-policy-label", plan["camera_policy"], "--out", str(out_dir)]
    if args.initial_full_state:
        cmd += ["--initial-states", args.initial_full_state]
    if args.final_full_state:
        cmd += ["--final-states", args.final_full_state]
    if plan["settings"]["subsample"]:
        cmd += ["--max-gaussians-per-source", str(plan["settings"]["subsample"])]
    return cmd


def run_logged(cmd, log: Path, dry: bool) -> int:
    pretty = " ".join(f"'{c}'" if " " in str(c) else str(c) for c in cmd)
    print(f"    $ {pretty}")
    if dry:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(f"\n### {time.ctime()}\n### {pretty}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    return proc.returncode


def latest_attempts(opt_dir: Path) -> list:
    return sorted(p for p in opt_dir.glob("attempt_*") if p.is_dir())


def attempt_status(a: Path):
    m = read_json(a / "manifest.json")
    if m is None:
        return "INCOMPLETE" if (a / "manifest.partial.json").is_file() else "EMPTY"
    return (m.get("solve") or {}).get("status", "UNKNOWN")


def best_attempt(opt_dir: Path):
    """The newest attempt that holds an accepted state (they only improve)."""
    for a in reversed(latest_attempts(opt_dir)):
        if (a / "best_accepted" / "state.npz").is_file():
            return a
    return None


def run_scene(args) -> int:
    results_root, dataset_root = Path(args.results_root), Path(args.dataset_root)
    if not results_root.is_dir():
        print(f"BLOCKED: results root {results_root} is not mounted")
        return EXIT_BLOCKED
    if not args.reuse_existing_phase1:
        print("BLOCKED: --reuse-existing-phase1 is required. This wrapper only "
              "consumes the existing forward/backward Phase 1; it never re-runs "
              "it (see scripts/run_legacy_free_phase1_export.sh for that).")
        return EXIT_BLOCKED
    info = inspect_scene(args.scene, results_root, dataset_root)
    if not info["eligible"]:
        print(json.dumps({"scene": args.scene, "status": "SKIPPED_MISSING_PRECOMPUTED_FB",
                          "reasons": info["reasons"]}, indent=1))
        return EXIT_BLOCKED
    plan = scene_plan(info, args.profile, args.tag, args)
    exp = Path(plan["experiment_dir"])
    if args.time_budget_min is not None:
        plan["settings"] = dict(plan["settings"], time_budget_min=args.time_budget_min)

    import torch
    cuda = bool(torch.cuda.is_available())
    try:
        import diff_gaussian_rasterization  # noqa: F401
        raster = True
    except Exception:
        raster = False
    free_gb = shutil.disk_usage(results_root).free / 2 ** 30
    existing = read_json(exp / "run_manifest.json")
    preflight = {
        "scene": args.scene, "profile": args.profile, "tag": plan["tag"],
        "experiment_dir": str(exp), "python": sys.executable,
        "cuda_available": cuda, "rasterizer_importable": raster,
        "disk_free_gib": round(free_gb, 1),
        "initialization_mode": ("RECOVERED_FULL_FILTER_STATE"
                                if args.initial_full_state and args.final_full_state
                                else "DERIVED_12D_INITIAL_GUESS_FROM_EXISTING_TRACKS"),
        "historical_inputs_read_only": info["inputs"],
        "warnings": info.get("warnings", []),
        "source_sha256": info["sha256"],
        "num_frames": plan["num_frames"], "camera_ids": plan["camera_ids"],
        "camera_policy": plan["camera_policy"],
        "gaussians": {"initial": info["forward_gaussians"],
                      "final": info["backward_gaussians"],
                      "union": info["forward_gaussians"] + info["backward_gaussians"]},
        "smoke_subsampled": plan["smoke_subsampled"],
        "fingerprint": plan["fingerprint"],
        "existing_experiment": (None if existing is None else
                                {"fingerprint": existing.get("fingerprint"),
                                 "matches": existing.get("fingerprint") == plan["fingerprint"]}),
        "legacy_runner_invoked": False, "static_training_invoked": False,
    }
    blockers = []
    if not cuda:
        blockers.append("CUDA_UNAVAILABLE")
    if not raster:
        blockers.append("RASTERIZER_NOT_IMPORTABLE")
    if free_gb < 5:
        blockers.append(f"LOW_DISK: {free_gb:.1f} GiB free")
    if existing is not None and existing.get("fingerprint") != plan["fingerprint"]:
        blockers.append("TAG_CONFIG_MISMATCH: the experiment directory was made "
                        "with a different configuration; choose another --tag")
    if existing is not None and not args.resume and not args.preflight and not args.dry_run:
        blockers.append("TAG_EXISTS: pass --resume to continue it, or use a new --tag")
    preflight["blockers"] = blockers
    print(json.dumps(preflight, indent=1, default=str))
    if args.preflight:
        return EXIT_BLOCKED if blockers else EXIT_OK
    if blockers:
        return EXIT_BLOCKED

    s = plan["settings"]
    opt = exp / "optimizer"
    logs = exp / "logs"
    stop_file = exp / "STOP"
    attempts = latest_attempts(opt)
    last = attempts[-1] if attempts else None
    converged = last is not None and attempt_status(last) == "CONVERGED"
    prev_best = best_attempt(opt) if opt.is_dir() else None
    next_attempt = opt / f"attempt_{len(attempts) + 1:03d}"

    print(f"\n=== plan: {args.scene} [{args.profile}] -> {exp}")
    steps = []
    rb = exp / "render_baselines"
    if not (rb / "manifest.json").is_file():
        steps.append(("render_baselines", cli_base(info, args, rb, "render_baseline", plan)))
    else:
        print("    render_baselines: present, reused")
    if converged:
        print(f"    optimizer: {last.name} is CONVERGED, reused")
    else:
        cmd = cli_base(info, args, next_attempt, "raw_union_experiment", plan) + [
            "--representation-policy", "diagnostic-only",
            "--camera-ids", ",".join(plan["camera_ids"]),
            "--max-outer", str(s["max_outer"]), "--pcg-iters", str(s["pcg_iters"]),
            "--pcg-tol", str(s["pcg_tol"]), "--damping", str(s["damping"]),
            "--stop-file", str(stop_file), "--no-visuals"] + STREAM
        if s["time_budget_min"]:
            cmd += ["--time-budget-min", str(s["time_budget_min"])]
        if prev_best is not None:
            cmd += ["--resume-from", str(prev_best)]
            print(f"    optimizer: resuming from {prev_best.name} "
                  f"(last status {attempt_status(last)})")
        steps.append(("optimizer", cmd))
    if args.dry_run:
        for name, cmd in steps:
            print(f"  [{name}]")
            run_logged(cmd, logs / f"{name}.log", dry=True)
        print("  [report] render_baseline with the best accepted attempt as a variant")
        print(f"  [docs]   {REPORT.name} -> {exp / 'docs'}")
        print("  outputs:")
        for rel in ("run_manifest.json", "initialization_provenance.json",
                    "render_baselines/", "optimizer/attempt_NNN/",
                    "optimizer/best_accepted_full_state.npz",
                    "optimizer/iterations.jsonl", "report/", "visuals/",
                    "comparison.csv", "status.json", "docs/", "logs/"):
            print(f"    {exp / rel}")
        return EXIT_OK

    lock = Lock(exp / ".lock")
    lock.acquire()
    rc_total = EXIT_OK
    try:
        if stop_file.exists():
            stop_file.unlink()
        if existing is None:
            atomic_json(exp / "run_manifest.json", {
                "fingerprint": plan["fingerprint"], "plan": plan,
                "sources": info["sha256"], "inputs_read_only": info["inputs"],
                "created": time.ctime(), "created_by_profile": args.profile})
        atomic_json(exp / "preflight.json", preflight)
        for name, cmd in steps:
            if name == "optimizer":
                next_attempt.parent.mkdir(parents=True, exist_ok=True)
            print(f"  [{name}] {time.ctime()}")
            rc = run_logged(cmd, logs / f"{name}.log", dry=False)
            print(f"  [{name}] exit {rc}")
            if rc != 0:
                rc_total = EXIT_FAILED
            if name == "optimizer":
                _publish_optimizer(opt)
            if stop_file.exists():
                print("  STOP file present: not starting further stages")
                break

        best = best_attempt(opt) if opt.is_dir() else None
        report_dir = exp / "report"
        made_from = (read_json(report_dir / "manifest.json") or {}).get("argv", [])
        need_report = best is not None and str(best) not in made_from
        if need_report:
            if report_dir.exists():
                shutil.rmtree(report_dir)          # our own, regenerated output
            cmd = cli_base(info, args, report_dir, "render_baseline", plan) + [
                "--extra-variants-from", str(best)]
            print(f"  [report] {time.ctime()}")
            rc = run_logged(cmd, logs / "report.log", dry=False)
            if rc == 0:
                for item in ("visuals", "comparison.csv"):
                    src_ = report_dir / item
                    dst = exp / item
                    tmp = exp / (item + ".incoming")
                    if tmp.exists():
                        shutil.rmtree(tmp) if tmp.is_dir() else tmp.unlink()
                    (shutil.copytree if src_.is_dir() else shutil.copy2)(src_, tmp)
                    if dst.exists():
                        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
                    tmp.replace(dst)
            else:
                rc_total = EXIT_FAILED
        elif best is None and (rb / "visuals").is_dir() and not (exp / "visuals").exists():
            shutil.copytree(rb / "visuals", exp / "visuals")
            shutil.copy2(rb / "comparison.csv", exp / "comparison.csv")

        rc = run_logged([sys.executable, str(REPORT), "--experiment", str(exp)],
                        logs / "docs.log", dry=False)
        if rc != 0:
            rc_total = EXIT_FAILED
        status = read_json(exp / "status.json") or {}
        print(json.dumps({"scene": args.scene, "status": status.get("SCENE_STATUS"),
                          "flags": status.get("flags")}, indent=1))
        if rc_total == EXIT_OK and status.get("SCENE_STATUS") != "CONVERGED":
            rc_total = EXIT_PARTIAL
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        (exp / "logs").mkdir(parents=True, exist_ok=True)
        (exp / "logs" / "driver_traceback.txt").write_text(traceback.format_exc())
        rc_total = EXIT_FAILED
    finally:
        lock.release()
    return rc_total


def _publish_optimizer(opt: Path) -> None:
    """Copy the best accepted state and concatenate every attempt's trace."""
    import numpy as np  # noqa: F401
    best = best_attempt(opt)
    if best is not None:
        tmp = opt / "best_accepted_full_state.npz.tmp"
        shutil.copy2(best / "best_accepted" / "state.npz", tmp)
        tmp.replace(opt / "best_accepted_full_state.npz")
    lines = []
    for a in latest_attempts(opt):
        tr = a / "trace.jsonl"
        if tr.is_file():
            for line in tr.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    rec["attempt"] = a.name
                    lines.append(json.dumps(rec))
    tmp = opt / "iterations.jsonl.tmp"
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
    tmp.replace(opt / "iterations.jsonl")
    atomic_json(opt / "status.json", {
        "attempts": [{"attempt": a.name, "status": attempt_status(a),
                      "has_best_accepted": (a / "best_accepted" / "state.npz").is_file()}
                     for a in latest_attempts(opt)],
        "best_attempt": best.name if best else None})


# ---------------------------------------------------------------------------
# overnight: discovery, serial queue, summary
# ---------------------------------------------------------------------------

def run_overnight(args) -> int:
    results_root, dataset_root = Path(args.results_root), Path(args.dataset_root)
    if not results_root.is_dir() or not dataset_root.is_dir():
        print(f"BLOCKED: results root {results_root} or dataset root "
              f"{dataset_root} is not mounted. Not falling back to anything else.")
        return EXIT_BLOCKED
    if not args.all_existing_scenes:
        print("BLOCKED: --all-existing-scenes is required")
        return EXIT_BLOCKED
    if not args.reuse_existing_phase1:
        print("BLOCKED: --reuse-existing-phase1 is required")
        return EXIT_BLOCKED
    scenes = discover(results_root, dataset_root)
    eligible = [s for s in scenes if s["eligible"]]
    run_dir = results_root / "unified_phase1_overnight" / args.tag
    queue = [{"order": i + 1, "scene": s["scene"], "frames": s["num_frames"],
              "cameras": s["num_cameras"],
              "gaussians": [s["forward_gaussians"], s["backward_gaussians"]],
              "warnings": s.get("warnings", []),
              "experiment_dir": str(Path(s["results_dir"]) / "unified_phase1_experiments"
                                    / args.tag)}
             for i, s in enumerate(eligible)]
    inventory = {"results_root": str(results_root), "dataset_root": str(dataset_root),
                 "tag": args.tag, "profile": args.profile,
                 "eligible": queue,
                 "skipped": [{"scene": s["scene"], "reasons": s["reasons"]}
                             for s in scenes if not s["eligible"]]}
    if args.list_scenes or args.preflight or args.dry_run:
        print(json.dumps(inventory, indent=1))
    if args.list_scenes:
        return EXIT_OK
    if not eligible:
        print("BLOCKED: no eligible scene")
        return EXIT_BLOCKED
    if args.preflight:
        atomic_json(run_dir / "eligible_scenes.json", inventory)
        print(f"\n  eligible_scenes.json -> {run_dir / 'eligible_scenes.json'}")
        return EXIT_OK
    runner = os.environ.get("FUSED_PHASE1_SCENE_RUNNER")   # test hook only
    import shlex
    base = (shlex.split(runner) if runner else
            ["bash", str(REPO / "scripts" / "run_fused_phase1.sh")])
    if args.dry_run:
        for q in queue:
            cmd = base + ["--scene", q["scene"], "--reuse-existing-phase1",
                          "--profile", args.profile, "--tag", args.tag] + (
                ["--resume"] if args.resume else [])
            print(f"  [{q['order']}/{len(queue)}] " + " ".join(cmd))
        print(f"\n  summary -> {run_dir / 'ALL_SCENES_SUMMARY.md'}")
        print(f"  stop     -> touch '{run_dir / 'STOP'}'")
        return EXIT_OK

    lock = Lock(run_dir / ".lock")
    lock.acquire()
    stop = run_dir / "STOP"
    if stop.exists():
        stop.unlink()
    atomic_json(run_dir / "eligible_scenes.json", inventory)
    print(f"overnight PID {os.getpid()}; run dir {run_dir}")
    print(f"  clean stop: touch '{stop}'  (finishes the current outer iteration)")
    rows = []
    try:
        for q in queue:
            if stop.exists():
                rows.append({**q, "status": "NOT_STARTED_STOPPED_BY_USER", "exit": None})
                continue
            exp = Path(q["experiment_dir"])
            if exp.exists() and not args.resume:
                rows.append({**q, "status": "SKIPPED_TAG_EXISTS_USE_RESUME", "exit": None})
                print(f"[{q['order']}/{len(queue)}] {q['scene']}: experiment exists, "
                      "pass --resume")
                continue
            cmd = base + ["--scene", q["scene"], "--reuse-existing-phase1",
                          "--profile", args.profile, "--tag", args.tag] + (
                ["--resume"] if args.resume else [])
            if args.scene_time_budget_min is not None:
                cmd += ["--time-budget-min", str(args.scene_time_budget_min)]
            log = run_dir / "logs" / f"{q['scene']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            print(f"[{q['order']}/{len(queue)}] {q['scene']}  start {time.ctime()}  "
                  f"log {log}", flush=True)
            with open(log, "a") as fh:
                proc = subprocess.Popen(cmd, cwd=REPO, stdout=fh,
                                        stderr=subprocess.STDOUT)
                # forward a STOP to the scene's own stop file
                while proc.poll() is None:
                    time.sleep(5)
                    if stop.exists():
                        (exp / "STOP").parent.mkdir(parents=True, exist_ok=True)
                        (exp / "STOP").touch()
                rc = proc.returncode
            st = read_json(exp / "status.json") or {}
            status = {EXIT_OK: "CONVERGED", EXIT_PARTIAL: "PARTIAL",
                      EXIT_BLOCKED: "BLOCKED"}.get(rc, "FAILED")
            if st.get("SCENE_STATUS"):
                status = st["SCENE_STATUS"] if rc in (EXIT_OK, EXIT_PARTIAL) else status
            row = {**q, "status": status, "exit": rc,
                   "wall_min": round((time.time() - t0) / 60, 1), "log": str(log),
                   **{k: st.get(k) for k in ("objective_initial", "objective_best",
                                             "outer_iterations", "pcg_true_residual_last",
                                             "peak_gpu_mib", "peak_host_mib",
                                             "camera_policy", "visuals")}}
            rows.append(row)
            print(f"    -> {status} (exit {rc}) in {row['wall_min']} min", flush=True)
            _write_summary(run_dir, rows, queue, inventory)
    finally:
        _write_summary(run_dir, rows, queue, inventory)
        lock.release()
    statuses = [r["status"] for r in rows]
    if any(s in ("FAILED", "BLOCKED") for s in statuses):
        return EXIT_FAILED
    if all(s == "CONVERGED" for s in statuses):
        return EXIT_OK
    return EXIT_PARTIAL


def _write_summary(run_dir: Path, rows, queue, inventory):
    atomic_json(run_dir / "ALL_SCENES_SUMMARY.json",
                {"rows": rows, "queue": queue, "skipped": inventory["skipped"],
                 "updated": time.ctime()})
    keys = ["order", "scene", "status", "exit", "wall_min", "frames", "cameras",
            "camera_policy", "gaussians", "objective_initial", "objective_best",
            "outer_iterations", "pcg_true_residual_last", "peak_gpu_mib",
            "peak_host_mib", "experiment_dir", "visuals", "log"]
    tmp = run_dir / "ALL_SCENES_SUMMARY.csv.tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tmp.replace(run_dir / "ALL_SCENES_SUMMARY.csv")
    lines = ["# Fused Phase 1 — all scenes", "",
             f"Updated {time.ctime()}. Statuses are measured; PARTIAL means the "
             "joint IEKS ran but was not certified CONVERGED (best accepted state "
             "kept). The raw union is a diagnostic representation in every scene.", "",
             "| # | scene | status | frames | cams | Gaussians (F/B) | E start -> best "
             "| outer | true PCG res. | GPU MiB | min | results |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e0, e1 = r.get("objective_initial"), r.get("objective_best")
        en = (f"{e0:.3e} -> {e1:.3e}" if isinstance(e0, (int, float))
              and isinstance(e1, (int, float)) else "")
        lines.append(
            f"| {r['order']} | {r['scene']} | {r['status']} | {r['frames']} | "
            f"{r.get('camera_policy') or r['cameras']} | {r['gaussians'][0]}/"
            f"{r['gaussians'][1]} | {en} | {r.get('outer_iterations') or ''} | "
            f"{r.get('pcg_true_residual_last') or ''} | {r.get('peak_gpu_mib') or ''} | "
            f"{r.get('wall_min') or ''} | `{r['experiment_dir']}` |")
    if inventory["skipped"]:
        lines += ["", "## Not eligible (never retrained)", ""]
        for s in inventory["skipped"]:
            lines.append(f"- `{s['scene']}`: {'; '.join(s['reasons'])}")
    tmp = run_dir / "ALL_SCENES_SUMMARY.md.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(run_dir / "ALL_SCENES_SUMMARY.md")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--results-root", default=os.environ.get("RESULTS_ROOT", DEFAULT_RESULTS))
        p.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", DEFAULT_DATASET))
        p.add_argument("--tag", required=True)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--preflight", action="store_true")
        p.add_argument("--dry-run", action="store_true")

    sc = sub.add_parser("scene")
    common(sc)
    sc.add_argument("--scene", required=True)
    sc.add_argument("--reuse-existing-phase1", action="store_true")
    sc.add_argument("--profile", choices=sorted(PROFILES), default="washer-validation")
    sc.add_argument("--time-budget-min", type=float, default=None)
    sc.add_argument("--initial-full-state", default=None)
    sc.add_argument("--final-full-state", default=None)

    ov = sub.add_parser("overnight")
    common(ov)
    ov.add_argument("--all-existing-scenes", action="store_true")
    ov.add_argument("--reuse-existing-phase1", action="store_true")
    ov.add_argument("--profile", choices=sorted(PROFILES), default="overnight")
    ov.add_argument("--list-scenes", action="store_true")
    ov.add_argument("--scene-time-budget-min", type=float, default=None)

    args = ap.parse_args()
    if args.cmd == "scene":
        return run_scene(args)
    return run_overnight(args)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit(main())
