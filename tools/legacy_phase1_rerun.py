#!/usr/bin/env python
"""OPT-IN, non-destructive rerun of ONE historical legacy Phase-1 direction.

Section 1.3 of CODEX_IMPLEMENT_FUSED_PHASE1_REUSE_EXISTING_FB_WASHER_THEN_OVERNIGHT_ALL.md.

The fused Phase 1 never needs this. It exists so the historical forward or
backward IEKF can be reproduced on request -- the SAME
tools/run_motion_coherence_support.py, the same settings, a NEW output
directory -- optionally with the output-only full-state recorder attached.

  * nothing runs without exactly one of --preflight / --dry-run / --execute;
  * output only to <RESULTS>/<SCENE>/unified_phase1_experiments/<TAG>/
    legacy_iekf_<direction>/, refused if it would fall inside static_bidi/ or
    bidirectional/;
  * never run_bidirectional_all.sh (it rm -rf's its stage directory), never
    static training;
  * after an --execute, the rerun's centres are compared with the historical
    ones frame by frame.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from fused_phase1_driver import (DEFAULT_DATASET, DEFAULT_RESULTS, EXIT_BLOCKED,  # noqa: E402
                                 EXIT_FAILED, EXIT_OK, Lock, atomic_json,
                                 inspect_scene, read_json, run_logged)

LEGACY_DEFAULTS = dict(bootstrap_iters=12, iekf_supernode_size=64, coherence_knn=12,
                       supernode_count=192, cv_folds=5)


def run_legacy(args) -> int:
    modes = [m for m in ("preflight", "dry_run", "execute") if getattr(args, m)]
    if len(modes) != 1:
        print("FAIL CLOSED: give exactly one of --preflight, --dry-run, --execute. "
              "Nothing runs without an explicit --execute.")
        return EXIT_BLOCKED
    results_root, dataset_root = Path(args.results_root), Path(args.dataset_root)
    info = inspect_scene(args.scene, results_root, dataset_root, hash_sources=False)
    endpoint = "initial" if args.direction == "forward" else "final"
    hist = results_root / args.scene / "bidirectional" / args.direction
    cfg = read_json(hist / "config.json") or {}
    out = (results_root / args.scene / "unified_phase1_experiments" / args.tag /
           f"legacy_iekf_{args.direction}")
    protected = [results_root / args.scene / "static_bidi",
                 results_root / args.scene / "bidirectional"]
    for p in protected:
        if p.resolve() == out.resolve() or p.resolve() in out.resolve().parents:
            print(f"REFUSED: {out} lies inside historical {p}")
            return EXIT_BLOCKED
    settings = dict(LEGACY_DEFAULTS, frames=cfg.get("frames", info.get("num_frames")),
                    cameras=cfg.get("cameras", info.get("num_cameras")),
                    coherence_knn=cfg.get("coherence_knn", LEGACY_DEFAULTS["coherence_knn"]),
                    seed=cfg.get("seed", 0))
    cmd = [sys.executable, str(REPO / "tools" / "run_motion_coherence_support.py"),
           "--scene", args.scene, "--direction", args.direction,
           "--canonical-endpoint", endpoint,
           "--model", str(results_root / args.scene / "static_bidi" / endpoint),
           "--source", str(dataset_root / args.scene),
           "--frames", str(settings["frames"]), "--cameras", str(settings["cameras"]),
           "--bootstrap-iters", str(settings["bootstrap_iters"]),
           "--iekf-supernode-size", str(settings["iekf_supernode_size"]),
           "--coherence-knn", str(settings["coherence_knn"]),
           "--supernode-count", str(settings["supernode_count"]),
           "--cv-folds", str(settings["cv_folds"]), "--seed", str(settings["seed"]),
           "--frames-on-cpu", "--out", str(out)]
    if args.save_full_state:
        cmd += ["--full-state-out", str(out / "full_state")]
    complete = (out / "predicted_support.json").is_file() and (
        not args.save_full_state or (out / "full_state" / "full_state_manifest.json").is_file())
    report = {"scene": args.scene, "direction": args.direction,
              "endpoint_model": endpoint, "historical_config": cfg,
              "settings": settings,
              "settings_source": ("frames/cameras/coherence_knn/seed from the "
                                  "historical bidirectional/<dir>/config.json; "
                                  "bootstrap_iters, iekf_supernode_size, "
                                  "supernode_count, cv_folds from "
                                  "run_bidirectional_all.sh defaults (they are "
                                  "not recorded in config.json)"),
              "output": str(out), "historical_left_untouched": str(hist),
              "already_complete": complete, "eligible": info["eligible"],
              "reasons": info["reasons"]}
    print(json.dumps(report, indent=1, default=str))
    pretty = " ".join(f"'{c}'" if " " in str(c) else str(c) for c in cmd)
    if args.preflight:
        return EXIT_OK if info["eligible"] else EXIT_BLOCKED
    print(f"\n  $ {pretty}")
    if args.dry_run:
        return EXIT_OK
    if not info["eligible"]:
        return EXIT_BLOCKED
    if out.exists() and not complete:
        print(f"REFUSED: {out} exists but is incomplete; the legacy runner cannot "
              "resume a partial run. Use a new --tag.")
        return EXIT_BLOCKED
    if complete:
        if args.resume:
            print("already complete; nothing to do")
            return EXIT_OK
        print(f"REFUSED: {out} already holds a completed rerun; use a new --tag")
        return EXIT_BLOCKED
    lock = Lock(out.parent / ".lock_legacy")
    lock.acquire()
    try:
        rc = run_logged(cmd, out.parent / "logs" / f"legacy_{args.direction}.log", dry=False)
        if rc == 0:
            _compare_legacy(hist / "phase1_centers", out / "phase1_centers",
                            out / "reproduction_vs_historical.json")
        return EXIT_OK if rc == 0 else EXIT_FAILED
    finally:
        lock.release()


def _compare_legacy(old: Path, new: Path, dst: Path) -> None:
    import numpy as np
    rows = []
    for f in sorted(new.glob("centers_frame_*.npy")):
        o = old / f.name
        if o.is_file():
            d = np.linalg.norm(np.load(f) - np.load(o), axis=1)
            rows.append({"frame": f.name, "median": float(np.median(d)),
                         "p95": float(np.percentile(d, 95)), "max": float(d.max())})
    atomic_json(dst, {"historical": str(old), "rerun": str(new), "per_frame": rows,
                      "determinism_expected": False,
                      "note": ("CUDA atomics in the rasterizer's backward pass make "
                               "a rerun not bitwise identical; deviations are "
                               "measured, not assumed away")})
    if rows:
        print(f"  reproduction vs historical: worst p95 "
              f"{max(r['p95'] for r in rows):.3e} m, worst max "
              f"{max(r['max'] for r in rows):.3e} m -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=os.environ.get("RESULTS_ROOT", DEFAULT_RESULTS))
    ap.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", DEFAULT_DATASET))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--direction", choices=("forward", "backward"), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--save-full-state", action="store_true")
    ap.add_argument("--resume", action="store_true")
    return run_legacy(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
