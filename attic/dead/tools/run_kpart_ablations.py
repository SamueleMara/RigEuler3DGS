"""Section 28's mandatory real-data ablation suite.

CODEX_REQUIRED_MODIFICATIONS_MULTIVIEW_MULTIFRAME_KPART_IEKF.md section 28.

    1. 4 cameras / 16 frames  -- the V2-comparable baseline
    2. 8 cameras / 16 frames
    3. 8 cameras / 30 frames
    4. 8 cameras / 60 frames
    5. sequential IEKF selector
    6. window discovery
    7. window discovery + model-error covariance
    8. final pipeline + canonical handover

Reports joint type, rate, support size, support stability, objective,
observability, runtime and memory for each. GT q(t) is used ONLY to score.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

ABLATIONS = [
    ("1_4cam_16frames", ["--cameras", "4", "--sequence-length", "16",
                         "--window-discovery", "--supernodes"],
     "V2-comparable baseline: 4 cameras, 16 frames"),
    ("2_8cam_16frames", ["--all-cameras", "--sequence-length", "16",
                         "--window-discovery", "--supernodes"],
     "all 8 cameras, 16 frames"),
    ("3_8cam_30frames", ["--all-cameras", "--sequence-length", "30",
                         "--window-discovery", "--supernodes"],
     "all 8 cameras, 30 frames"),
    ("4_8cam_60frames", ["--all-cameras", "--all-frames",
                         "--window-discovery", "--supernodes"],
     "all 8 cameras, all 60 frames"),
    ("5_sequential_selector", ["--all-cameras", "--sequence-length", "16",
                               "--supernodes"],
     "sequential IEKF selector (no window discovery)"),
    ("6_window_discovery", ["--all-cameras", "--sequence-length", "16",
                            "--window-discovery", "--supernodes",
                            "--no-model-error"],
     "window discovery, sensor noise only"),
    ("7_window_plus_model_error", ["--all-cameras", "--sequence-length", "16",
                                   "--window-discovery", "--supernodes"],
     "window discovery + model-error covariance"),
    ("8_final_plus_handover", ["--all-cameras", "--sequence-length", "16",
                               "--window-discovery", "--supernodes",
                               "--canonical-handover", "--split-supernodes"],
     "final pipeline + canonical handover + partition refinement"),
]


def run_one(name, extra, note, args) -> dict:
    out = Path(args.out_root) / name
    cmd = [PY, "-u", str(REPO / "tools" / "run_iekf_scene.py"),
           "--model", args.model,
           "--observation-source", "dataset_rgbd",
           "--sequence-start", "0",
           "--full-gaussians",
           "--num-moving-parts", str(args.K),
           "--depth-sigma", args.depth_sigma,
           "--bootstrap-iters", str(args.bootstrap_iters),
           "--discovery-outer", str(args.discovery_outer),
           "--out", str(out)] + extra
    print(f"\n{'=' * 74}\n  ABLATION {name}: {note}\n{'=' * 74}")
    print("  " + " ".join(cmd[1:]))
    t0 = time.perf_counter()
    env = {"PYTHONPATH": str(REPO),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    import os
    e = dict(os.environ)
    e.update(env)
    p = subprocess.run(cmd, capture_output=True, text=True, env=e,
                       timeout=args.timeout)
    dt = time.perf_counter() - t0
    tail = "\n".join(p.stdout.strip().splitlines()[-6:])
    print(tail)
    if p.returncode != 0:
        print(f"  FAILED rc={p.returncode}")
        print("\n".join(p.stderr.strip().splitlines()[-8:]))
    return {"name": name, "note": note, "command": cmd[1:],
            "returncode": p.returncode, "wall_clock_s": dt,
            "out_dir": str(out),
            "stdout_tail": tail}


def summarise(rec: dict) -> dict:
    """Pull section 28's required columns out of the persisted artifacts."""
    d = Path(rec["out_dir"])
    s = {"ablation": rec["name"], "note": rec["note"],
         "runtime_s": rec["wall_clock_s"], "ok": rec["returncode"] == 0}

    def load(fn):
        p = d / fn
        return json.loads(p.read_text()) if p.exists() else None

    cam = load("camera_usage.json")
    if cam:
        s["cameras"] = cam.get("num_cameras_used")
    fr = load("frame_usage.json")
    if fr:
        s["frames"] = fr.get("num_frames")
    k = load("kpart_summary.json")
    if k:
        s["status"] = k.get("status")
        s["peak_cuda_gib"] = k.get("peak_cuda_gib")
        sc = k.get("scene") or {}
        parts = sc.get("parts") or []
        s["joint_type"] = [p.get("kind") for p in parts]
        q = k.get("q_traj") or []
        if q and len(q[0]) > 1:
            dq = [b - a for a, b in zip(q[0], q[0][1:])]
            s["dq_per_frame"] = sum(dq) / len(dq)
        ro = k.get("reduced_observability") or {}
        s["reduced_rank"] = ro.get("observable_rank")
        s["reduced_condition"] = ro.get("condition_number")
    sup = load("support_kpart.json")
    if sup:
        s["support_counts"] = sup.get("label_counts")
        s["support_energy"] = sup.get("total_support_energy")
    wd = load("window_discovery.json")
    if wd:
        s["objective"] = (wd.get("energy") or {}).get("total")
        s["converged"] = wd.get("converged")
        s["selected"] = wd.get("selected")
    ho = load("handover_refit.json")
    if ho:
        s["handover_before"] = ho.get("energy_before")
        s["handover_after"] = ho.get("energy_after")
        s["handover_accepted"] = ho.get("accepted")
    sel = load("joint_selected.json")
    if sel and "joint_type" not in s:
        s["joint_type"] = sel.get("selected")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="output/merged_USB_100109")
    ap.add_argument("--out-root", default="output/kpart_ablations")
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--depth-sigma", default="constant:0.05:0.0")
    ap.add_argument("--bootstrap-iters", type=int, default=6)
    ap.add_argument("--discovery-outer", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these ablation names")
    args = ap.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    records, rows = [], []
    for name, extra, note in ABLATIONS:
        if args.only and name not in args.only:
            continue
        rec = run_one(name, extra, note, args)
        records.append(rec)
        rows.append(summarise(rec))
        Path(args.out_root, "ablation_summary.json").write_text(
            json.dumps({"rows": rows, "records": records}, indent=2))

    print(f"\n{'=' * 100}\n  SECTION 28 ABLATION SUMMARY\n{'=' * 100}")
    hdr = (f"  {'ablation':<26} {'cam':>4} {'frm':>4} {'type':>12} "
           f"{'dq/frame':>11} {'support':>14} {'objective':>13} {'rank':>5} "
           f"{'runtime':>9}")
    print(hdr)
    print("  " + "-" * 104)
    for r in rows:
        jt = r.get("joint_type")
        jt = (",".join(jt) if isinstance(jt, list) else str(jt))[:12]
        dq = r.get("dq_per_frame")
        dq_s = f"{dq:+.6f}" if dq is not None else "-"
        obj = r.get("objective")
        obj_s = f"{obj:.1f}" if obj is not None else "-"
        print(f"  {r['ablation']:<26} {str(r.get('cameras', '-')):>4} "
              f"{str(r.get('frames', '-')):>4} {jt:>12} {dq_s:>11} "
              f"{str(r.get('support_counts', '-')):>14} {obj_s:>13} "
              f"{str(r.get('reduced_rank', '-')):>5} {r['runtime_s']:>8.1f}s")

    print(f"\n  written to {args.out_root}/ablation_summary.json")


if __name__ == "__main__":
    main()
