#!/usr/bin/env python
"""Calibrated RGB-D -> surface-centred bootstrap -> time-consistent Phase 1/2/3.

RigEuler3DGS_Calibrated_Bootstrap_IEKF_Implementation_Spec.md section 1. One
dataset scene / episode per call:

    real       --dataset-kind real [--rgbd-noise-calibration
                --calibration-start-frame A --calibration-end-frame B]
               --motion-start-frame S --motion-end-frame E
    synthetic  --dataset-kind synthetic --motion-start-frame S --motion-end-frame E
               (no calibration, sensor policy ideal)

Stages (--stages, default "calibration bootstrap phase1 phase2 phase3"):

  calibration  tools/calibrate_static_rgbd.py on [A, B]   -> RESULTS/calibration/
  bootstrap    tools/train_tracking_bootstrap.py at S and, independently, at E
                                                          -> RESULTS/bootstrap/{initial,final}/
  phase1-3     run_bidirectional_all.sh on RESULTS/pipeline/<scene>/ with the
               bootstrap models as the endpoint models, the motion window, the
               capture timestamps, the shared sensor model and the bootstrap P0

Validation runs BEFORE any CUDA work and refuses: unknown / out-of-range frame IDs,
non-increasing timestamps, an incomplete endpoint or calibration frame, missing
masks over the motion window, calibration flags on synthetic data, a calibration
interval without --rgbd-noise-calibration (or the flag without the interval), a
calibration interval overlapping the motion window, and any mismatch between the
dataset kind, the calibration choice and --sensor-noise-model. Nothing falls back
silently: a real capture without calibration runs only with
--sensor-noise-model assumed and is labelled UNCALIBRATED.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
EXT = (".png", ".jpg", ".jpeg")


def sha(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frame_ids(src: Path, cam: str) -> list:
    return sorted(int(f.stem.split("_")[-1]) for f in (src / "rgb" / cam).iterdir() if f.suffix.lower() in EXT)


def has_frame(src: Path, cam: str, t: int) -> list:
    miss = []
    if not any((src / "rgb" / cam / f"frame_{t:06d}{e}").is_file() for e in EXT):
        miss.append(f"rgb/{cam}/frame_{t:06d}")
    if not (src / "depth_npy" / cam / f"frame_{t:06d}.npy").is_file():
        miss.append(f"depth_npy/{cam}/frame_{t:06d}.npy")
    if not any((src / "mask" / cam / f"frame_{t:06d}{e}").is_file() for e in EXT):
        miss.append(f"mask/{cam}/frame_{t:06d}")
    return miss


def validate(a) -> dict:
    """Schema validation; returns the facts it established. SystemExit on any violation."""
    src = Path(a.source)
    errs = []
    if not src.is_dir():
        raise SystemExit(f"--source {src} does not exist")
    cams = sorted(p.name for p in (src / "rgb").glob("cam_*") if p.is_dir())
    if not cams:
        raise SystemExit(f"{src}/rgb has no cam_* folders")
    if a.cameras:
        cams = cams[:a.cameras]
    ids = frame_ids(src, cams[0])
    for c in cams[1:]:
        if frame_ids(src, c) != ids:
            errs.append(f"camera {c} has a different frame set than {cams[0]}")
    S, E = a.motion_start_frame, a.motion_end_frame
    if not (S in ids and E in ids and E > S):
        errs.append(f"motion interval [{S}, {E}] is not a valid range of frame IDs {ids[0]}..{ids[-1]}")
    elif ids != list(range(ids[0], ids[-1] + 1)):
        errs.append("frame IDs are not contiguous; the loader reads frame_NNNNNN consecutively")
    # dataset kind x calibration x sensor model, no silent fallback
    cal_given = a.calibration_start_frame is not None or a.calibration_end_frame is not None
    if a.dataset_kind == "synthetic":
        if a.rgbd_noise_calibration or cal_given:
            errs.append("synthetic data has no physical sensor: --rgbd-noise-calibration / calibration frames are refused")
        if a.sensor_noise_model != "ideal":
            errs.append(f"synthetic data requires --sensor-noise-model ideal, not {a.sensor_noise_model}")
    else:
        if cal_given and not a.rgbd_noise_calibration:
            errs.append("a calibration interval was given without --rgbd-noise-calibration")
        if a.rgbd_noise_calibration:
            if a.calibration_start_frame is None or a.calibration_end_frame is None:
                errs.append("--rgbd-noise-calibration needs BOTH --calibration-start-frame and --calibration-end-frame")
            elif a.calibration_end_frame - a.calibration_start_frame + 1 < 4:
                errs.append("the calibration interval needs >= 4 frames (variance + held-out model selection)")
            elif not (a.calibration_start_frame in ids and a.calibration_end_frame in ids):
                errs.append("calibration frames outside the dataset's frame IDs")
            elif not (a.calibration_end_frame < S or a.calibration_start_frame > E):
                errs.append(f"calibration interval [{a.calibration_start_frame}, {a.calibration_end_frame}] overlaps the "
                            f"motion interval [{S}, {E}]; calibration frames must be static and not motion frames")
            if a.sensor_noise_model != "calibrated":
                errs.append(f"--rgbd-noise-calibration produces a calibrated model; --sensor-noise-model is {a.sensor_noise_model}")
        elif a.sensor_noise_model != "assumed":
            errs.append("a real capture without --rgbd-noise-calibration runs only with --sensor-noise-model assumed "
                        "(status UNCALIBRATED)")
    if errs:
        raise SystemExit("validation failed:\n  - " + "\n  - ".join(errs))
    # data completeness and time
    for t in (S, E):
        for c in cams:
            m = has_frame(src, c, t)
            if m:
                errs.append(f"endpoint frame {t}: missing {m}")
    if a.rgbd_noise_calibration:
        for t in range(a.calibration_start_frame, a.calibration_end_frame + 1):
            for c in cams:
                m = has_frame(src, c, t)
                if m:
                    errs.append(f"calibration frame {t}: missing {m}")
    ts = None
    tsp = src / "metadata" / "frame_timestamps.csv"
    if a.frame_timestamps == "auto":
        if not tsp.is_file():
            errs.append(f"--frame-timestamps auto but {tsp} does not exist")
        else:
            rows = {int(r["frame_index"]): float(r["timestamp_s"]) for r in csv.DictReader(open(tsp))}
            ts = [rows.get(t) for t in range(S, E + 1)]
            if any(x is None for x in ts):
                errs.append("frame_timestamps.csv does not cover the motion interval")
            elif any(b <= a_ for a_, b in zip(ts, ts[1:])):
                errs.append("timestamps are not strictly increasing over the motion interval")
    elif a.dataset_kind == "real":
        errs.append("real captures need physical timestamps: --frame-timestamps auto (or a path)")
    sys.path.insert(0, str(REPO / "tools"))
    from check_masks import check
    probs, _info = check(src, src / "mask", len(cams), E - S + 1, start=S)
    errs += [f"masks: {p}" for p in probs[:5]]
    for d in ("colmap",):
        if not (src / d / "sparse" / "0").is_dir():
            errs.append(f"{src / d}/sparse/0 missing (camera geometry)")
    if errs:
        raise SystemExit("validation failed:\n  - " + "\n  - ".join(errs))
    return {"cameras": cams, "frames_available": [ids[0], ids[-1]], "timestamps_motion_s": ts,
            "dt_s": (None if not ts else [round(b - a_, 6) for a_, b in zip(ts, ts[1:])][:3])}


def git(*args) -> str | None:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def run(cmd, log: Path, dry: bool, env=None) -> int:
    print("  + " + " ".join(f"'{c}'" if " " in str(c) else str(c) for c in cmd))
    if dry:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        p = subprocess.run([str(c) for c in cmd], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if p.returncode != 0:
        print(f"    !! failed (rc {p.returncode}), log {log}")
        print("    " + "\n    ".join(log.read_text().splitlines()[-8:]))
    return p.returncode


def bootstrap_reusable(d: Path, want: dict) -> bool | None:
    """True reuse / None nothing there / False refuse (a different configuration)."""
    m = d / "bootstrap_manifest.json"
    if not d.exists() or not any(d.iterdir()):
        return None
    if not m.is_file():
        return False
    j = json.loads(m.read_text())
    return all(j.get(k) == v for k, v in want.items())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True, help="one dataset scene / episode directory")
    ap.add_argument("--dataset-kind", choices=("real", "synthetic"), required=True)
    ap.add_argument("--rgbd-noise-calibration", action="store_true")
    ap.add_argument("--calibration-start-frame", type=int, default=None)
    ap.add_argument("--calibration-end-frame", type=int, default=None)
    ap.add_argument("--motion-start-frame", type=int, required=True)
    ap.add_argument("--motion-end-frame", type=int, required=True)
    ap.add_argument("--bootstrap-mode", choices=("covariance_surface",), default="covariance_surface")
    ap.add_argument("--sensor-noise-model", choices=("calibrated", "ideal", "assumed"), required=True)
    ap.add_argument("--frame-timestamps", default=None, help="auto (real default) | PATH | off (synthetic default)")
    ap.add_argument("--cameras", type=int, default=None, help="first N cameras (default all)")
    ap.add_argument("--num-gaussians", type=int, default=30000)
    ap.add_argument("--refine-iters", type=int, default=1000)
    ap.add_argument("--rgb-model-sigma", type=float, default=0.0, help="declared model-mismatch sd (not sensor)")
    ap.add_argument("--depth-model-sigma", type=float, default=0.0, help="declared model-mismatch sd [m]")
    ap.add_argument("--p0-velocity-sigma", type=float, nargs=2, default=(1.0, 0.316227766))
    ap.add_argument("--observation-mode", default="masked-rgb-expected-quad")
    ap.add_argument("--stages", default="calibration bootstrap phase1 phase2 phase3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.frame_timestamps is None:
        a.frame_timestamps = "auto" if a.dataset_kind == "real" else "off"
    t0 = time.time()
    src, out = Path(a.source).resolve(), Path(a.out)
    facts = validate(a)
    stages = a.stages.split()
    scene = src.name
    print(f"validated {scene}: cameras {facts['cameras']}, motion [{a.motion_start_frame}, {a.motion_end_frame}], "
          f"sensor {a.sensor_noise_model}, timestamps {a.frame_timestamps}")
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    cal_json = out / "calibration" / "sensor_model.json"
    man = {"schema": 1, "scene": scene, "source": str(src), "dataset_kind": a.dataset_kind,
           "sensor": a.sensor_noise_model,
           "status": {"calibrated": "CALIBRATED", "ideal": "SYNTHETIC_NO_SENSOR_NOISE",
                      "assumed": "UNCALIBRATED"}[a.sensor_noise_model],
           "intervals": {"calibration": ([a.calibration_start_frame, a.calibration_end_frame]
                                         if a.rgbd_noise_calibration else None),
                         "motion": [a.motion_start_frame, a.motion_end_frame],
                         "forward_canonical_frame": a.motion_start_frame,
                         "backward_canonical_frame": a.motion_end_frame},
           "validation": facts, "config": vars(a),
           "code": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "*.py", "*.sh"))},
           "inputs_sha256": {"colmap_cameras.bin": sha(src / "colmap" / "sparse" / "0" / "cameras.bin"),
                             "colmap_images.bin": sha(src / "colmap" / "sparse" / "0" / "images.bin"),
                             "metadata_cameras.json": sha(src / "metadata" / "cameras.json"),
                             "frame_timestamps.csv": sha(src / "metadata" / "frame_timestamps.csv")},
           "stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def save():
        if not a.dry_run:
            (out / "run_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    save()
    sensor_args = []
    # ---- calibration ---------------------------------------------------------
    if a.sensor_noise_model == "calibrated":
        if "calibration" in stages:
            if cal_json.is_file():
                print(f"  [calibration] present, reusing {cal_json}")
                man["stages"]["calibration"] = "reused"
            else:
                rc = run([PY, "tools/calibrate_static_rgbd.py", "--source", src,
                          "--calibration-start-frame", a.calibration_start_frame,
                          "--calibration-end-frame", a.calibration_end_frame,
                          "--cameras", ",".join(facts["cameras"]), "--out", out / "calibration"],
                         out / "logs" / "calibration.log", a.dry_run)
                man["stages"]["calibration"] = "complete" if rc == 0 else "failed"
                save()
                if rc:
                    return 1
        elif not cal_json.is_file() and not a.dry_run:
            raise SystemExit(f"--sensor-noise-model calibrated but {cal_json} does not exist and the calibration "
                             "stage was not requested")
        sensor_args = ["--sensor-model", cal_json]
        if cal_json.is_file():
            man["sensor_model_id"] = json.loads(cal_json.read_text()).get("sensor_model_id")
    else:
        sensor_args = ["--sensor-policy", a.sensor_noise_model]
    # ---- bootstrap: two independent endpoints --------------------------------
    for end, frame in (("initial", a.motion_start_frame), ("final", a.motion_end_frame)):
        d = out / "bootstrap" / end
        if "bootstrap" in stages:
            want = {"endpoint_frame": frame, "num_gaussians": None, "refine_iters": a.refine_iters,
                    "sensor_policy": a.sensor_noise_model}
            want.pop("num_gaussians")
            r = bootstrap_reusable(d, want)
            if r is True:
                print(f"  [bootstrap/{end}] present with the same configuration, reusing")
                man["stages"][f"bootstrap_{end}"] = "reused"
            elif r is False:
                raise SystemExit(f"{d} holds a bootstrap with a different configuration; use a new --out")
            else:
                rc = run([PY, "tools/train_tracking_bootstrap.py", "--source", src, "--frame", frame,
                          "--endpoint", end, "--out", d, *sensor_args,
                          "--num-gaussians", a.num_gaussians, "--refine-iters", a.refine_iters,
                          "--rgb-model-sigma", a.rgb_model_sigma, "--depth-model-sigma", a.depth_model_sigma],
                         out / "logs" / f"bootstrap_{end}.log", a.dry_run)
                man["stages"][f"bootstrap_{end}"] = "complete" if rc == 0 else "failed"
                save()
                if rc:
                    return 1
        # the driver reads the endpoint models from <root>/<scene>/static_bidi/<end> (copies: exFAT)
        dst = out / "pipeline" / scene / "static_bidi" / end
        if not a.dry_run and d.is_dir() and not dst.exists():
            shutil.copytree(d, dst, ignore=shutil.ignore_patterns("source_responsibility.npz"))
    # ---- Phase 1/2/3 through the driver ---------------------------------------
    ph = [s for s in stages if s in ("phase1", "phase2", "phase3")]
    if ph:
        cmd = ["bash", "run_bidirectional_all.sh", "-i", src.parent, "-o", out / "pipeline", "--scenes", scene,
               "--stages", " ".join(ph), "--cameras", str(a.cameras or len(facts["cameras"])),
               "--motion-start-frame", a.motion_start_frame, "--motion-end-frame", a.motion_end_frame,
               "--frame-timestamps", a.frame_timestamps, *sensor_args, "--bootstrap-p0",
               "--p0-velocity-sigma", *a.p0_velocity_sigma,
               "--rgb-model-sigma", a.rgb_model_sigma, "--depth-model-sigma", a.depth_model_sigma,
               "--observation-mode", a.observation_mode]
        env = dict(os.environ, ITERS=str(a.refine_iters))
        print(f"  (driver ITERS={a.refine_iters}: the bootstrap's final checkpoint is the endpoint model)")
        rc = run(cmd, out / "logs" / "pipeline.log", a.dry_run, env=env)
        man["stages"]["phase123"] = {"rc": rc, "log": str(out / "logs" / "pipeline.log"),
                                     "results": str(out / "pipeline" / scene / "bidirectional")}
        save()
        if rc:
            man["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save()
            return rc
    man["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    man["runtime_s"] = round(time.time() - t0, 1)
    save()
    print(f"done {scene}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
