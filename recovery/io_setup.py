"""Shared CLI plumbing for Phase 4 / Phase 5: upstream contract, cameras exactly as Phase 3 used them, lazy
views over the Phase-3 window, run manifests."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .contracts import load_upstream
from .material_completion import ViewRef
from .views import ViewStore, camera_from_3dgs, check_against_metadata


def add_common_cli(ap):
    ap.add_argument("--source", required=True, help="scene directory (rgb/, depth_npy/, mask/, metadata/)")
    ap.add_argument("--results-root", default=None, help="recorded only")
    ap.add_argument("--direction", default="auto", choices=("auto", "forward", "backward"))
    ap.add_argument("--model", required=True, help="static_bidi/initial (forward) or static_bidi/final (backward)")
    ap.add_argument("--phase2-dir", required=True)
    ap.add_argument("--phase3-dir", required=True)
    ap.add_argument("--joint-type", required=True, choices=("prismatic", "revolute"),
                    help="explicit: Phase 3 fits both and does not record a selection")
    ap.add_argument("--cameras", default=None, help="must equal Phase 3's camera set (default: Phase 3's)")
    ap.add_argument("--frame-window", default=None, help="original_start:original_end, must equal Phase 3's")
    ap.add_argument("--sensor-model", default=None, help="recorded and hashed")
    ap.add_argument("--frames-on-cpu", action="store_true", default=True,
                    help="views stay on the CPU, one at a time on the GPU (always on)")
    ap.add_argument("--view-cache", type=int, default=128)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke-frames", type=int, default=0,
                    help="DEBUG ONLY: keep this many evenly spaced Phase-3 frames (outputs marked SMOKE, not results)")


def git_state(repo: Path) -> dict:
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": [l for l in run("status", "--porcelain").splitlines() if l and "submodules/sam3" not in l]}


def setup(args, log=print):
    up = load_upstream(args.source, args.model, args.phase2_dir, args.phase3_dir, args.joint_type, args.direction,
                       args.cameras, args.frame_window, args.sensor_model)
    if args.dry_run:
        return up, None, None, None
    from tools.run_iekf_scene import load_scene
    from utils.camera_selection import select_cameras
    _, _, _, sc = load_scene(str(up.model_dir), str(up.source), inference_only=True, camera_device="cpu")
    cams = [camera_from_3dgs(c) for c in select_cameras(sc.getTrainCameras(), ",".join(up.cameras))]
    if [c.name for c in cams] != up.cameras:
        raise RuntimeError(f"cameras {[c.name for c in cams]} != Phase 3's {up.cameras}")
    cam_check = [check_against_metadata(c, up.source) for c in cams]
    store = ViewStore(up.source, {c.name: c for c in cams}, cache=args.view_cache)
    frames = sorted(up.joint.frames)
    smoke = None
    if args.smoke_frames:
        import numpy as np
        keep = sorted(set(np.linspace(0, len(frames) - 1, args.smoke_frames).round().astype(int).tolist()))
        smoke = [frames[k] for k in keep]
        log(f"SMOKE RUN: {len(smoke)} of {len(frames)} frames -- not a result")
        frames = smoke
    views = [ViewRef(c, f, (lambda c=c.name, f=f: store.get(c, f))) for c in cams for f in frames]
    return up, cams, views, {"camera_check": cam_check, "smoke_frames": smoke}


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def setup_phase4(args, log=print, spill_dir=None):
    """Phase-4 setup: same contract and cameras as `setup`, but every view is decoded ONCE into a
    PreparedViews store (recovery/view_cache.py); the ViewRef getters read that store and never decode.
    Phase 5 keeps using `setup` (unchanged)."""
    from .view_cache import PreparedViews, ViewKey
    up = load_upstream(args.source, args.model, args.phase2_dir, args.phase3_dir, args.joint_type, args.direction,
                       args.cameras, args.frame_window, args.sensor_model)
    from tools.run_iekf_scene import load_scene
    from utils.camera_selection import select_cameras
    _, _, _, sc = load_scene(str(up.model_dir), str(up.source), inference_only=True, camera_device="cpu")
    cams = [camera_from_3dgs(c) for c in select_cameras(sc.getTrainCameras(), ",".join(up.cameras))]
    if [c.name for c in cams] != up.cameras:
        raise RuntimeError(f"cameras {[c.name for c in cams]} != Phase 3's {up.cameras}")
    cam_check = [check_against_metadata(c, up.source) for c in cams]
    frames = sorted(up.joint.frames)
    subset = None
    if getattr(args, "view_subset", 0):
        import numpy as np
        keep = sorted(set(np.linspace(0, len(frames) - 1, args.view_subset).round().astype(int).tolist()))
        subset = [frames[k] for k in keep]
        log(f"VIEW SUBSET: {len(subset)} of {len(frames)} frames per camera -- benchmark only, NOT_FINAL")
        frames = subset
    stride = int(getattr(args, "frame_stride", 0) or 1)
    if stride > 1:                         # 4B / 4C recovery views: every stride-th Phase-3 frame (declared, recorded)
        frames = frames[::stride]
        subset = frames
        log(f"RECOVERY FRAME STRIDE {stride}: {len(frames)} of {len(sorted(up.joint.frames))} frames per camera")
    order = [(c, f) for c in cams for f in frames]
    pv = PreparedViews(up.source, [ViewKey(c.name, int(f), c.W, c.H) for c, f in order],
                       max_host_gb=getattr(args, "max_host_gb", 8.0), spill_dir=spill_dir, log=log)
    if getattr(args, "dry_run", False):
        return up, cams, None, pv, {"camera_check": cam_check, "view_subset": subset}
    pv.prepare()
    views = [ViewRef(c, f, (lambda i=i: pv.cpu_view(i))) for i, (c, f) in enumerate(order)]
    return up, cams, views, pv, {"camera_check": cam_check, "view_subset": subset}
