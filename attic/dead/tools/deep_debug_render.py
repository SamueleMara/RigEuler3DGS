#!/usr/bin/env python
"""Sections 32, 33, 47-49: render the fixed-camera stage sequence to disk.

CODEX_DEEP_STAGEWISE_DEBUG_AND_THEORY_AUDIT.md.

    python tools/deep_debug_render.py --run output/geometric_all/storage_45135

Unlike `tools/deep_debug.py`, this one needs the model, the dataset images and
a GPU: these are the quantities section 58 says cannot be reconstructed from
cached tensors. It writes

    <run>/deep_debug/renders/camera_<id>/frame_XXXX/*.png
    <run>/deep_debug/renders/camera_<id>/sequence_manifest.json
    <run>/deep_debug/renders/strips/*.png

Section 49's same-camera rule is enforced structurally: every stage at a given
frame is rendered through the identical CameraSpec with the identical
observation, and the depth and residual colour maps are normalised once per
camera for the whole sequence rather than per image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from debug.deep_debug_renders import (DepthNorm, STAGES,  # noqa: E402
                                      keyframe_strip, render_state,
                                      write_frame_images)
from tools.dual_joint_discovery import DEFAULT_DT, load_phase1_tracks  # noqa: E402


def _part_members(root: Path, part: int):
    """Gaussian indices of the moving part, from the cached labels."""
    lf = root / "rigid_part_labels.pt"
    if not lf.is_file():
        return None
    lab = torch.load(lf, map_location="cpu", weights_only=False)
    lab = (lab["labels"] if isinstance(lab, dict) else lab).reshape(-1)
    return (lab == part) if int((lab == part).sum()) else (lab != 0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=None, help="trained 3DGS model dir")
    ap.add_argument("--source", default=None, help="dataset source dir")
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)
    ap.add_argument("--cameras", type=int, default=2,
                    help="how many cameras to render (section 38 wants more "
                         "than one so a joint is not supported by a single view)")
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--frames", type=int, default=60,
                    help="dataset frames to load, matching the original run")
    ap.add_argument("--max-gaussians", type=int, default=0,
                    help="0 = every Gaussian; a positive value subsamples for "
                         "speed and is recorded in the manifest")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.run)
    scene = root.name
    out = Path(args.out) if args.out else (root / "deep_debug" / "renders")
    out.mkdir(parents=True, exist_ok=True)

    # ---- resolve the model/source from the dataset registry --------------
    model, source = args.model, args.source
    if model is None or source is None:
        from tools.run_all_datasets import discover
        for rec in discover(REPO):
            if rec["scene"] == scene:
                model = model or str(rec["model"])
                source = source or str(rec["source"])
                break
    if model is None or source is None:
        raise SystemExit(
            f"could not resolve the model/source for {scene}; pass --model "
            f"and --source explicitly")
    print(f"=== renders: {scene} ===\n  model  {model}\n  source {source}")

    from tools.run_iekf_scene import (camera_specs, dataset_rgbd_frames,
                                      load_scene, state_from_gaussians)
    from joint_reduction.kpart_window import WindowObjective
    from estimator.sparse_iekf import IEKFConfig

    dataset, pipe, gaussians, sc = load_scene(model, source)
    cams_all = sc.getTrainCameras()
    ncam = min(args.cameras, len(cams_all))
    cameras = camera_specs(cams_all, ncam)
    truth = state_from_gaussians(gaussians)
    print(f"  {truth.mu.shape[0]} Gaussians, using {ncam} of "
          f"{len(cams_all)} cameras")

    cfg = IEKFConfig()
    # The loader reads its settings off a Namespace; these mirror what the
    # batch runner passes, so the frames here are the frames the run saw.
    ns = argparse.Namespace(sequence_start=0, sequence_length=args.frames,
                            observation_source="dataset_rgbd",
                            depth_sigma="constant:0.05:0.0",
                            frames_on_cpu=True, no_rgb=False, no_depth=False,
                            all_frames=False, all_cameras=False,
                            cameras=ncam, window_length=0, dt=args.dt)
    frames, frame_ids = dataset_rgbd_frames(cams_all[:ncam], cfg, ns, truth,
                                            Path(source))
    window = [(obs, fid) for (obs, _s, _d), fid in
              zip(frames, [f["dataset_frame_index"]
                           for f in frame_ids["frames"]])]
    T = len(window)
    print(f"  {T} dataset frames")

    objective = WindowObjective(truth.mu, truth.R, cameras, window, cfg,
                                state_template=truth)

    # ---- per-stage Gaussian geometry -------------------------------------
    members = _part_members(root, args.part)
    ph1 = load_phase1_tracks(root, args.part)
    tf = root / "debug_numeric" / "rigid_transforms.pt"
    T4 = (torch.load(tf, map_location="cpu", weights_only=False).double()
          if tf.is_file() else None)

    dev = truth.mu.device
    mu0, R0 = truth.mu, truth.R
    unavailable: dict = {}

    # Phase 1 free: carry each member Gaussian by its supernode's displacement.
    # A Gaussian belongs to the nearest supernode centre at frame 0, which is
    # how the grouping was formed, so this is the field Phase 1 reconstructed
    # rather than a re-estimate of it.
    ph1_disp = None
    if members is not None and int(members.sum()):
        c0 = ph1["x0"].to(dev).float()
        sel = members.to(dev)
        d = torch.cdist(mu0[sel].float(), c0)
        near = d.argmin(1)
        ph1_disp = (ph1["tracks"].to(dev).float() - c0[None])   # [T,S,3]
    else:
        unavailable["phase1_free"] = "no part labels in this run"

    if T4 is None:
        unavailable["phase2_rigid"] = "debug_numeric/rigid_transforms.pt absent"

    # P and R charts from the offline geometric candidate
    pch = rch = None
    try:
        from joint_reduction.direct_gaussian_joint import (
            dual_differential_finite_evidence, geometric_joint_candidate)
        ev = dual_differential_finite_evidence(ph1["x0"], ph1["tracks"],
                                               dt=args.dt)
        cand = geometric_joint_candidate(ev, x0=ph1["x0"], dt=args.dt)
        pch, rch = cand.get("chart_prismatic"), cand.get("chart_revolute")
        if pch is None:
            unavailable["P"] = (f"no chart: the geometric candidate returned "
                                f"{cand['status']}")
            unavailable["R"] = unavailable["P"]
    except Exception as exc:                                   # pragma: no cover
        unavailable["P"] = unavailable["R"] = f"{type(exc).__name__}: {exc}"
    unavailable.setdefault("refined", "Phase-4 refined geometry is not "
                                      "persisted per frame by this run")

    def geometry(stage: str, t: int):
        """(mu, R) for one stage at one frame, or None when unavailable."""
        if stage == "canonical":
            return mu0, R0
        if stage == "phase1_free" and ph1_disp is not None:
            mu = mu0.clone()
            mu[sel] = mu0[sel] + ph1_disp[t][near]
            return mu, R0
        if stage == "phase2_rigid" and T4 is not None:
            idx = min(max(args.part - 1, 0), T4.shape[0] - 1)
            M = T4[idx, t].to(dev).float()
            mu = mu0.clone()
            Rg = R0.clone() if R0 is not None else None
            mu[sel] = mu0[sel] @ M[:3, :3].T + M[:3, 3]
            if Rg is not None:
                Rg[sel] = M[:3, :3] @ R0[sel]
            return mu, Rg
        if stage in ("P", "R") and pch is not None:
            chart = pch if stage == "P" else rch
            Rm, p = chart.transform(t)
            Rm, p = Rm.to(dev).float(), p.to(dev).float()
            mu = mu0.clone()
            Rg = R0.clone() if R0 is not None else None
            mu[sel] = mu0[sel] @ Rm.T + p
            if Rg is not None:
                Rg[sel] = Rm @ R0[sel]
            return mu, Rg
        return None

    todo = list(range(0, T, max(1, args.frame_stride)))
    manifests: dict = {}
    for ci in range(ncam):
        cam_dir = out / f"camera_{ci:03d}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        # one depth normalisation for the whole camera (section 49)
        lo, hi = [], []
        for t in todo:
            ob = objective._to_device(window[t][0], dev)[ci]
            if ob is None:
                continue
            gd, gv = ob[1], ob[2]
            v = gd[gv]
            if v.numel():
                lo.append(float(v.min()))
                hi.append(float(v.max()))
        dnorm = DepthNorm(min(lo) if lo else 0.0, max(hi) if hi else 1.0)
        print(f"  camera {ci}: depth normalised over "
              f"[{dnorm.lo:.3f}, {dnorm.hi:.3f}] m for the whole sequence")

        frames_rec = []
        for t in todo:
            obs_dev = objective._to_device(window[t][0], dev)
            ob = obs_dev[ci]
            folder = cam_dir / f"frame_{t:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            rec = {"frame": t, "images": {}, "unavailable": {}}
            gt_rgb = gt_d = gt_v = None
            if ob is not None:
                gt_rgb, gt_d, gt_v = (ob[0].float().cpu(), ob[1].float().cpu(),
                                      ob[2].cpu())
                rec["images"].update(write_frame_images(
                    folder, "gt", gt_rgb, gt_d, None, None, None, None,
                    dnorm, cfg.min_alpha_for_depth, 1.0))
            else:
                rec["unavailable"]["gt"] = "no observation for this camera"

            for stage in STAGES:
                if stage == "gt":
                    continue
                g = geometry(stage, t)
                if g is None:
                    rec["unavailable"][stage] = unavailable.get(
                        stage, "stage not produced by this run")
                    continue
                mu, Rg = g
                with torch.no_grad():
                    rgb, depth, alpha = render_state(objective, mu, Rg,
                                                     obs_dev, ci)
                rec["images"].update(write_frame_images(
                    folder, stage, rgb, depth, alpha, gt_rgb, gt_d, gt_v,
                    dnorm, cfg.min_alpha_for_depth, 0.25))
            frames_rec.append(rec)
            if t == todo[0]:
                print(f"    frame {t}: wrote {len(rec['images'])} images, "
                      f"{len(rec['unavailable'])} stage(s) unavailable")

        man = {"scene": scene, "camera": ci, "part": args.part,
               "frames": frames_rec, "stages": list(STAGES),
               "frame_stride": args.frame_stride,
               "depth_normalisation": {"low_m": dnorm.lo, "high_m": dnorm.hi,
                                       "fixed_for_whole_sequence": True},
               "rgb_residual_gain": 0.25,
               "same_camera_rule": ("section 49: every stage uses this "
                                    "camera's intrinsics and extrinsics, this "
                                    "frame, and one depth/residual "
                                    "normalisation for the whole sequence"),
               "unavailable_stages": unavailable,
               "model": model, "source": source}
        (cam_dir / "sequence_manifest.json").write_text(
            json.dumps(man, indent=1, default=str))
        manifests[ci] = man

    # ---- section 48: the key-frame strips --------------------------------
    key = sorted({todo[0], todo[len(todo) // 4], todo[len(todo) // 2],
                  todo[3 * len(todo) // 4], todo[-1]})
    strips = []
    for ci in range(ncam):
        for resid in (False, True):
            name = (f"camera_{ci:03d}_"
                    f"{'residual' if resid else 'rgb'}_strip.png")
            p = keyframe_strip(out, ci, key, list(STAGES),
                               out / "strips" / name, residual=resid)
            if p:
                strips.append(str(p.relative_to(out)))
    (out / "strips_manifest.json").write_text(json.dumps(
        {"keyframes": key, "strips": strips,
         "layout": "GT | canonical | phase1_free | phase2_rigid | P | R | refined",
         "note": "section 48; a missing cell is drawn as 'n/a', never filled in"},
        indent=1))
    print(f"\n  renders -> {out}")
    print(f"  strips  -> {out / 'strips'} ({len(strips)} sheets, "
          f"key frames {key})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
