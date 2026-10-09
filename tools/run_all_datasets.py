#!/usr/bin/env python
"""Run the geometric joint-discovery pipeline on every dataset that has a
frame-0 static 3DGS model.

A dataset is RUNNABLE when both of these exist:

    input/<scene>/                                       the RGB-D sequence
    output/<scene>_static_endpoints_14k/shared/frame0_static   the frame-0 model

The second is the static model trained on frame 0; its sibling `frame_t_static`
is the endpoint model and is deliberately NOT used -- the estimator must start
from canonical geometry and discover the motion itself.

Every dataset is run with the SAME algorithm and the same flags. Nothing here
is per-scene except the camera and frame counts, which are read from the
dataset rather than chosen: that is the point of running them all.

    python tools/run_all_datasets.py                 # discover and run everything
    python tools/run_all_datasets.py --list          # show what would run, run nothing
    python tools/run_all_datasets.py --only USB_100109 storage_45135
    python tools/run_all_datasets.py --max-frames 30 --quick
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Static-model directory suffixes, PREFERRED FIRST. 30k with depth is the
# current default; the 14k models predate it and are still perfectly usable,
# so discovery keeps finding them rather than orphaning seven trained scenes
# the day the default moved. A scene with both resolves to the 30k one.
SUFFIXES = ("_static_endpoints_30k", "_static_endpoints_14k")
SUFFIX = SUFFIXES[0]
RGB_EXT = (".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
def discover(repo: Path) -> list[dict]:
    """Every (scene, model, source) triple that can actually be run."""
    out = []
    roots = {}
    for suf in reversed(SUFFIXES):          # later writes win, so prefer first
        for d in sorted((repo / "output").glob(f"*{suf}")):
            roots[d.name[: -len(suf)]] = (d, suf)
    for scene in sorted(roots):
        model_root, suf = roots[scene]
        model = model_root / "shared" / "frame0_static"
        source = repo / "input" / scene
        rec = {"scene": scene, "model": model, "source": source,
               "tag": suf.lstrip("_"), "runnable": True, "why_not": []}

        if not (model / "cfg_args").exists():
            rec["runnable"] = False
            rec["why_not"].append("no cfg_args in shared/frame0_static")
        if not sorted(model.glob("point_cloud/iteration_*")):
            rec["runnable"] = False
            rec["why_not"].append("no point_cloud/iteration_* checkpoint")
        if not source.is_dir():
            rec["runnable"] = False
            rec["why_not"].append(f"no input/{scene}")

        rgb = source / "rgb"
        cams = sorted(p.name for p in rgb.iterdir() if p.is_dir()) if rgb.is_dir() else []
        rec["cameras"] = len(cams)
        frames = 0
        if cams:
            d = rgb / cams[0]
            frames = max(len(list(d.glob(f"frame_*{e}"))) for e in RGB_EXT)
        rec["frames"] = frames
        if not cams:
            rec["runnable"] = False
            rec["why_not"].append("no rgb/<cam>/ directories")
        if frames < 2:
            rec["runnable"] = False
            rec["why_not"].append(f"only {frames} RGB frames")
        if not (source / "depth_npy").is_dir():
            rec["runnable"] = False
            rec["why_not"].append("no depth_npy/")

        # Relative cost, for ordering and for the up-front plan. The dominant
        # term is rendering, which scales with Gaussians x cameras; USB is the
        # unit because it is the one scene with a measured full-run time.
        n_gauss = 0
        ply = sorted(model.glob("point_cloud/iteration_*/point_cloud.ply"))
        if ply:
            try:
                from plyfile import PlyData
                n_gauss = int(PlyData.read(str(ply[-1])).elements[0].count)
            except Exception:
                n_gauss = 0
        rec["gaussians"] = n_gauss
        rec["cost"] = (n_gauss * max(len(cams), 1)) / 90000.0

        ji = source / "metadata" / "joint_info.json"
        rec["gt_type"] = None
        if ji.exists():
            try:
                sel = json.loads(ji.read_text()).get("selected_joint") or {}
                rec["gt_type"] = {"continuous": "revolute",
                                  "revolute": "revolute",
                                  "prismatic": "prismatic"}.get(sel.get("type"))
            except Exception:
                pass
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
def build_command(rec: dict, args, out_dir: Path) -> list[str]:
    n = min(rec["frames"], args.max_frames) if args.max_frames else rec["frames"]
    cmd = [sys.executable, "tools/run_iekf_scene.py",
           "--model", str(rec["model"]),
           "--source", str(rec["source"]),
           "--observation-source", "dataset_rgbd",
           "--sequence-start", "0", "--sequence-length", str(n),
           "--all-cameras",
           "--supernodes",
           "--window-discovery", "--num-moving-parts", str(args.K),
           "--depth-sigma", args.depth_sigma,
           "--no-visualizations",
           "--out", str(out_dir)]
    for flag in ('joint_direct_gaussian','joint_direct_rgbd','joint_heldout_validation'):
        if getattr(args,flag,False):cmd.append('--'+flag.replace('_','-'))
    cmd += ['--joint-confidence',str(getattr(args,'joint_confidence',.95)),
            '--joint-block-frames',str(getattr(args,'joint_block_frames',10)),
            '--joint-iters',str(getattr(args,'joint_iters',30))]
    if args.camera_folds > 1:
        # Section 8: each fold holds out a different camera block, so the
        # refits differ in their EVIDENCE while the algorithm is untouched.
        cmd += ["--camera-folds", str(args.camera_folds),
                "--camera-fold", str(args.camera_fold)]
    cmd += ["--max-gaussians", str(args.max_gaussians)] if args.max_gaussians \
        else ["--full-gaussians"]
    if args.quick:
        cmd += ["--discovery-outer", "2", "--rigid-refine-iters", "6",
                "--reopen-support-rounds", "1", "--bootstrap-iters", "6",
                "--iters", "6", "--pcg-iters", "40"]
    else:
        cmd += ["--discovery-outer", str(args.discovery_outer),
                "--rigid-refine-iters", str(args.rigid_refine_iters),
                "--reopen-support-rounds", str(args.reopen_support_rounds),
                "--bootstrap-iters", str(args.bootstrap_iters),
                "--iters", str(args.iters), "--pcg-iters", str(args.pcg_iters)]
    if args.camera_folds > 1 and args.no_debug:
        # Section 8: the fold refits exist to be compared against each other,
        # and that comparison reads debug_numeric/rigid_transforms.pt. The
        # heavy debug output (render panels, exported Gaussian frames) is not
        # needed for it, so folding turns on the numerics alone.
        cmd += ["--debug-numerics"]
    if not args.no_debug:
        cmd += ["--debug-all",
                "--debug-frame-stride", str(args.debug_frame_stride),
                "--debug-gaussian-subsample", str(args.debug_gaussian_subsample)]
    return cmd


def summarise(scene: str, out_dir: Path, rec: dict, seconds: float,
              status: str) -> dict:
    """Read back what the run decided. Missing files are reported, not guessed."""
    row = {"scene": scene, "status": status, "runtime_min": seconds / 60.0,
           "cameras": rec["cameras"], "frames_available": rec["frames"],
           "gt_type": rec["gt_type"], "selected": None, "fit_status": None,
           "E_P": None, "E_R": None, "gap": None, "floor": None,
           "axis": None, "axis_point": None, "q_min": None, "q_max": None,
           "rigid_converged": None, "label_counts": None,
           "gt_dq_per_frame": None, "est_dq_per_frame": None,
           "sweep_error_rad": None, "type_match": None}

    gj = out_dir / "geometric_joints.json"
    if gj.exists():
        try:
            j = (json.loads(gj.read_text()).get("per_part") or [None])[0]
            if j:
                ne = j.get("normalized_error") or {}
                q = j.get("q") or []
                row.update(selected=j.get("selected"), fit_status=j.get("status"),
                           E_P=ne.get("prismatic"), E_R=ne.get("revolute"),
                           gap=ne.get("absolute_gap"),
                           floor=ne.get("indistinguishable_floor"),
                           axis=j.get("axis"), axis_point=j.get("axis_point"),
                           q_min=min(q) if q else None,
                           q_max=max(q) if q else None)
                if len(q) > 1:
                    row["est_dq_per_frame"] = (q[-1] - q[0]) / (len(q) - 1)
        except Exception as exc:
            row["status"] = f"{status} (unreadable geometric_joints.json: {exc})"

    rp = out_dir / "rigid_parts.json"
    if rp.exists():
        try:
            d = json.loads(rp.read_text())
            row["rigid_converged"] = d.get("converged")
            row["label_counts"] = d.get("counts")
        except Exception:
            pass

    gt = out_dir / "dataset_joint_ground_truth.json"
    if gt.exists():
        try:
            g = json.loads(gt.read_text())
            row["gt_dq_per_frame"] = g.get("gt_dq_per_frame")
            gq = g.get("gt_q") or []
            if gq and row["q_max"] is not None:
                n = min(len(gq), int(row.get("frames_used") or len(gq)))
                sweep_gt = gq[n - 1] - gq[0]
                row["sweep_error_rad"] = abs((row["q_max"] - row["q_min"])
                                             - abs(sweep_gt))
        except Exception:
            pass
    if row["gt_type"] and row["selected"]:
        row["type_match"] = (row["selected"] == row["gt_type"])
    return row


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default="output/geometric_all",
                    help="parent directory for the per-scene run directories")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these scene names")
    ap.add_argument("--skip", nargs="+", default=[], help="scenes to leave out")
    ap.add_argument("--list", action="store_true",
                    help="show what would run and exit")
    ap.add_argument("--force", action="store_true",
                    help="rerun scenes that already have a result")
    ap.add_argument("--continue-on-error", action="store_true", default=True)
    ap.add_argument("--stop-on-error", action="store_false",
                    dest="continue_on_error")
    ap.add_argument("--max-frames", type=int, default=60,
                    help="cap the window; 0 uses every frame the dataset has")
    ap.add_argument("--max-gaussians", type=int, default=0,
                    help="0 means --full-gaussians")
    ap.add_argument("--K", type=int, default=1, help="moving parts")
    ap.add_argument("--quick", action="store_true",
                    help="small iteration budgets, for a shakedown pass")
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--discovery-outer", type=int, default=8)
    ap.add_argument("--rigid-refine-iters", type=int, default=25)
    ap.add_argument("--reopen-support-rounds", type=int, default=4)
    ap.add_argument("--bootstrap-iters", type=int, default=12)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--pcg-iters", type=int, default=80)
    ap.add_argument("--debug-frame-stride", type=int, default=5)
    ap.add_argument("--debug-gaussian-subsample", type=int, default=6000)
    ap.add_argument("--depth-sigma", default="constant:0.05:0.0")
    ap.add_argument('--joint-direct-gaussian',action='store_true')
    ap.add_argument('--joint-direct-rgbd',action='store_true')
    ap.add_argument('--joint-heldout-validation',action='store_true')
    ap.add_argument('--joint-confidence',type=float,default=.95)
    ap.add_argument('--joint-block-frames',type=int,default=10)
    ap.add_argument('--joint-iters',type=int,default=30)
    ap.add_argument("--camera-folds", type=int, default=0,
                    help="section 8: number of camera blocks for the "
                         "repeatability refits (0 disables)")
    ap.add_argument("--camera-fold", type=int, default=-1,
                    help="section 8: which camera block this run holds out")
    ap.add_argument("--order", choices=("cost", "name"), default="cost",
                    help="cost runs the cheapest scenes first, so an "
                         "overnight run that overruns still finishes the most "
                         "datasets")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop LAUNCHING new scenes once this much wall clock "
                         "has passed; 0 means no limit. A running scene is "
                         "never interrupted.")
    args = ap.parse_args()

    found = discover(REPO)
    if args.only:
        found = [r for r in found if r["scene"] in set(args.only)]
    found = [r for r in found if r["scene"] not in set(args.skip)]
    if not found:
        print("no datasets matched")
        return 1

    runnable = [r for r in found if r["runnable"]]
    blocked = [r for r in found if not r["runnable"]]
    if args.order == "cost":
        runnable.sort(key=lambda r: r.get("cost", 0.0))

    print(f"\n{'scene':<46}{'cam':>4}{'frames':>8}{'gauss':>8}{'cost':>7}"
          f"{'GT':>11}")
    print("-" * 92)
    for r in runnable + blocked:
        mark = "" if r["runnable"] else "  [SKIP] " + "; ".join(r["why_not"])
        print(f"{r['scene']:<46}{r['cameras']:>4}{r['frames']:>8}"
              f"{r.get('gaussians', 0):>8}{r.get('cost', 0):>7.2f}"
              f"{str(r['gt_type']):>11}{mark}")
    total_cost = sum(r.get("cost", 0.0) for r in runnable)
    # The anchor: USB (cost 1.0) took 32 min at --discovery-outer 8 /
    # --rigid-refine-iters 25. Phase 2 is roughly two thirds of that and scales
    # with the refine budget, so a different budget rescales the estimate.
    scale = 0.35 + 0.65 * (args.rigid_refine_iters / 25.0)
    est = total_cost * 32.0 * (1.0 if args.quick else scale)
    print(f"\n{len(runnable)} runnable, {len(blocked)} skipped; "
          f"total relative cost {total_cost:.1f}x USB")
    print(f"ROUGH estimate at these settings: {est / 60:.1f} h "
          f"(anchor: USB measured 32 min at refine=25; this extrapolates "
          f"from one point and ignores per-scene convergence)")
    if args.max_hours:
        print(f"will stop launching new scenes after {args.max_hours:.1f} h")
    if args.list:
        return 0

    out_root = REPO / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    rows, t_all = [], time.perf_counter()

    for i, rec in enumerate(runnable, 1):
        scene = rec["scene"]
        out_dir = out_root / scene
        done = (out_dir / "geometric_joints.json").exists() \
            or (out_dir / "kpart_summary.json").exists()
        if done and not args.force:
            print(f"\n[{i}/{len(runnable)}] {scene}: already has a result, "
                  f"skipping (--force to rerun)")
            rows.append(summarise(scene, out_dir, rec, 0.0, "cached"))
            continue

        if args.max_hours and (time.perf_counter() - t_all) / 3600.0 >= args.max_hours:
            print(f"\n[{i}/{len(runnable)}] {scene}: wall-clock budget of "
                  f"{args.max_hours:.1f} h reached, not launching. Rerun the "
                  f"same command later -- finished scenes are cached.")
            rows.append({**summarise(scene, out_dir, rec, 0.0, "not started"),
                         "status": "not started (time budget)"})
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(rec, args, out_dir)
        log = out_dir / "run.log"
        print(f"\n{'=' * 104}\n[{i}/{len(runnable)}] {scene}  "
              f"({rec['cameras']} cameras, "
              f"{min(rec['frames'], args.max_frames) if args.max_frames else rec['frames']} frames)"
              f"\n  -> {out_dir}\n  log: {log}\n{'=' * 104}")
        env = {**os.environ, "PYTHONPATH": ".",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               # Python block-buffers stdout when it is a file rather than a
               # terminal, so a redirected overnight log shows nothing for
               # minutes at a time and `tail -f` is useless. Unbuffered output
               # costs nothing here and makes the run followable.
               "PYTHONUNBUFFERED": "1"}
        env.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
        t0 = time.perf_counter()
        with log.open("w") as fh:
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT)
        dt = time.perf_counter() - t0
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"  {status} in {dt / 60:.1f} min")
        if proc.returncode != 0:
            tail = log.read_text().strip().splitlines()[-12:]
            print("  --- last lines of the log ---")
            for ln in tail:
                print("   ", ln[:160])
            if not args.continue_on_error:
                rows.append(summarise(scene, out_dir, rec, dt, status))
                break
        rows.append(summarise(scene, out_dir, rec, dt, status))

    # ---- combined report -------------------------------------------------
    total = time.perf_counter() - t_all
    (out_root / "all_datasets_summary.json").write_text(json.dumps(
        {"total_minutes": total / 60.0,
         "settings": {k: v for k, v in vars(args).items()},
         "skipped": [{"scene": r["scene"], "why_not": r["why_not"]}
                     for r in blocked],
         "results": rows}, indent=2, default=str))

    def f(v, d=5):
        return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))

    lines = ["# Geometric joint discovery — all datasets", "",
             f"{len(rows)} run(s), {total / 60:.1f} min total.", "",
             "| scene | status | GT | selected | match | E_P | E_R | gap vs floor | q range | rigid conv | min |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        match = "—" if r["type_match"] is None else ("**yes**" if r["type_match"] else "**NO**")
        qr = "—" if r["q_min"] is None else f"{r['q_min']:.4f} … {r['q_max']:.4f}"
        gv = "—" if r["gap"] is None else f"{r['gap']:.2e} / {r['floor']:.2e}"
        lines.append(
            f"| {r['scene']} | {r['status']} | {r['gt_type']} | "
            f"{r['selected'] or r['fit_status']} | {match} | "
            f"{f(r['E_P'], 3) if r['E_P'] is None else format(r['E_P'], '.3e')} | "
            f"{f(r['E_R'], 3) if r['E_R'] is None else format(r['E_R'], '.3e')} | "
            f"{gv} | {qr} | {r['rigid_converged']} | {r['runtime_min']:.1f} |")
    if blocked:
        lines += ["", "## Skipped", ""]
        lines += [f"- **{r['scene']}** — {'; '.join(r['why_not'])}" for r in blocked]
    md = "\n".join(lines) + "\n"
    (out_root / "all_datasets_summary.md").write_text(md)
    print("\n" + md)
    print(f"summary: {out_root / 'all_datasets_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
