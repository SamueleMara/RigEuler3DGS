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
sys.path.insert(0, str(REPO))          # utils.*, estimator.* for the declared (ideal / assumed) sensor model
from utils.camera_selection import resolve  # noqa: E402

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
    try:
        cams = resolve(a.cameras, cams)
    except ValueError as ex:
        raise SystemExit(str(ex))
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
    # direction(s): explicit flag, else the dataset's own recommendation (no guessing)
    a.direction = getattr(a, "direction", "auto")
    if a.direction in ("forward", "backward"):
        dirs, dsrc = [a.direction], f"--direction {a.direction}"
    elif a.direction == "both":
        dirs, dsrc = ["forward", "backward"], "--direction both"
    else:
        f = src / "metadata" / "direction" / "summary.json"
        rec = str(json.loads(f.read_text()).get("recommended_direction", "")).lower() if f.is_file() else ""
        if rec not in ("forward", "backward"):
            raise SystemExit(f"validation failed:\n  - no recommended_direction in {f}; pass --direction "
                             "forward|backward|both")
        dirs, dsrc = [rec], str(f)
    ends = [("initial", S) if d == "forward" else ("final", E) for d in dirs]
    # data completeness and time (only the endpoints the chosen direction(s) need)
    for t in [fr for _e, fr in ends]:
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
            "directions": dirs, "direction_source": dsrc, "endpoints": ends,
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


def cached_stage(name, d: Path, sig: dict, dry: bool, man: dict) -> str:
    """'run' or 'reuse' for a stage directory under tools/stage_cache.py's contract;
    SystemExit on a refusal (nothing deleted). A same-signature interrupted attempt is
    moved aside (never deleted) and redone; in a dry run nothing is touched."""
    import stage_cache as sc
    decision, msg = sc.decide(d, sig)
    if decision == "refuse":
        raise SystemExit(f"  !! [{name}] {msg}")
    if decision == "reuse":
        print(f"  [{name}] reusing: {msg}")
        man["stages"][name] = {"status": "reused", "signature_sha256": sc.sig_hash(sig)}
        return "reuse"
    if decision == "redo":
        if dry:
            print(f"  [{name}] {msg}: would be moved aside and redone (dry run: untouched)")
        else:
            moved = sc.move_aside(d)
            print(f"  [{name}] {msg}: moved aside to {moved}, redoing")
    return "run"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True, help="one dataset scene / episode directory")
    ap.add_argument("--dataset-kind", choices=("real", "synthetic"), required=True)
    ap.add_argument("--rgbd-noise-calibration", action="store_true")
    ap.add_argument("--calibration-start-frame", type=int, default=None)
    ap.add_argument("--calibration-end-frame", type=int, default=None)
    ap.add_argument("--direction", choices=("auto", "forward", "backward", "both"), default="auto",
                    help="pass(es) to run; auto (default) = the dataset's metadata/direction/summary.json "
                         "recommended_direction, refused when absent")
    ap.add_argument("--motion-start-frame", type=int, required=True)
    ap.add_argument("--motion-end-frame", type=int, required=True)
    ap.add_argument("--bootstrap-mode", choices=("covariance_surface",), default="covariance_surface")
    ap.add_argument("--sensor-noise-model", choices=("calibrated", "ideal", "assumed"), required=True)
    ap.add_argument("--frame-timestamps", default=None, help="auto (real default) | PATH | off (synthetic default)")
    ap.add_argument("--cameras", default="all", help="all | first N | cam_001,cam_002, for every stage")
    ap.add_argument("--num-gaussians", type=int, default=30000)
    ap.add_argument("--refine-iters", type=int, default=1000)
    ap.add_argument("--rgb-model-sigma", type=float, default=0.0, help="declared model-mismatch sd (not sensor)")
    ap.add_argument("--depth-model-sigma", type=float, default=0.0, help="declared model-mismatch sd [m]")
    ap.add_argument("--p0-velocity-sigma", type=float, nargs=2, default=(1.0, 0.316227766))
    # bootstrap / calibration settings, passed explicitly so the stage signatures see them
    ap.add_argument("--knn", type=int, default=24)
    ap.add_argument("--checkpoints", default="0,250,500,1000,2000")
    ap.add_argument("--heldout-fraction", type=float, default=0.1)
    ap.add_argument("--silhouette-sigma", type=float, default=0.1)
    ap.add_argument("--thickness-weight", type=float, default=0.01)
    ap.add_argument("--p0-prior-sigma-pos", type=float, default=0.05)
    ap.add_argument("--p0-prior-sigma-rot", type=float, default=0.5)
    ap.add_argument("--lr-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration-bins", type=int, default=8)
    ap.add_argument("--calibration-min-bin-samples", type=int, default=200)
    ap.add_argument("--observation-mode", default="masked-rgb-expected-quad")
    # Phase 1 only (RigEuler3DGS_Phase1_Covariance_Velocity_Corrections_Codex.md), passed to the driver
    ap.add_argument("--phase1-partition", choices=("canonical", "per-frame"), default="canonical")
    ap.add_argument("--phase1-covariance-transport", action="store_true")
    ap.add_argument("--phase1-alpha-precision", choices=("coupled", "declared"), default="coupled")
    ap.add_argument("--phase1-alpha-sd", type=float, default=None)
    ap.add_argument("--phase1-rgb-support", choices=("full", "foreground"), default="full")
    ap.add_argument("--phase1-first-return-mode", choices=("off", "audit", "depth"), default="off")
    ap.add_argument("--phase1-first-return-prior", default=None)
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
    print(f"validated {scene}: directions {facts['directions']} ({facts['direction_source']}), "
          f"cameras {facts['cameras']}, motion [{a.motion_start_frame}, {a.motion_end_frame}], "
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
           "directions": facts["directions"], "direction_source": facts["direction_source"],
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
    sys.path.insert(0, str(REPO / "tools"))
    import stage_cache as sc
    # ---- calibration ---------------------------------------------------------
    if a.sensor_noise_model == "calibrated":
        copt = {"bins": a.calibration_bins, "min_bin_samples": a.calibration_min_bin_samples}
        csig = sc.calibration_signature(src, facts["cameras"], a.calibration_start_frame,
                                        a.calibration_end_frame, copt)
        cdir = out / "calibration"
        if "calibration" in stages:
            if cached_stage("calibration", cdir, csig, a.dry_run, man) == "run":
                if not a.dry_run:
                    sc.record(cdir, csig, "running")
                rc = run([PY, "tools/calibrate_static_rgbd.py", "--source", src,
                          "--calibration-start-frame", a.calibration_start_frame,
                          "--calibration-end-frame", a.calibration_end_frame,
                          "--cameras", ",".join(facts["cameras"]), "--bins", a.calibration_bins,
                          "--min-bin-samples", a.calibration_min_bin_samples, "--out", cdir],
                         out / "logs" / "calibration.log", a.dry_run)
                if not a.dry_run:
                    bad = sc.validate("calibration", cdir, csig) if rc == 0 else ["calibration tool failed"]
                    sc.record(cdir, csig, "complete" if not bad else "failed",
                              {"problems": bad} if bad else None)
                    man["stages"]["calibration"] = {"status": "complete" if not bad else "failed",
                                                    "signature_sha256": sc.sig_hash(csig), "problems": bad}
                    save()
                    if bad:
                        return 1
        elif not a.dry_run and sc.decide(cdir, csig)[0] != "reuse":
            raise SystemExit(f"--sensor-noise-model calibrated: {cdir} is not a complete calibration with "
                             "this signature and the calibration stage was not requested")
        sensor_args = ["--sensor-model", cal_json]
        sid = json.loads(cal_json.read_text()).get("sensor_model_id") if cal_json.is_file() else "PENDING_CALIBRATION"
        man["sensor_model_id"] = sid
        sensor = {"policy": "calibrated", "sensor_model_id": sid}
    else:
        sensor_args = ["--sensor-policy", a.sensor_noise_model]
        from utils.noise.sensor_model import declared_model
        sensor = {"policy": a.sensor_noise_model, "sensor_model_id": declared_model(a.sensor_noise_model).model_id}
    # ---- bootstrap: two independent endpoints --------------------------------
    bopt = {"num_gaussians": a.num_gaussians, "knn": a.knn, "refine_iters": a.refine_iters,
            "checkpoints": a.checkpoints, "heldout_fraction": a.heldout_fraction,
            "silhouette_sigma": a.silhouette_sigma, "thickness_weight": a.thickness_weight,
            "rgb_model_sigma": a.rgb_model_sigma, "depth_model_sigma": a.depth_model_sigma,
            "p0_prior_sigma_pos": a.p0_prior_sigma_pos, "p0_prior_sigma_rot": a.p0_prior_sigma_rot,
            "lr_scale": a.lr_scale, "seed": a.seed,
            # EVALUATION ONLY labels: synthetic inputs carry labelled endpoint clouds, real captures none
            "gt_labels": "off" if a.dataset_kind == "real" else "auto"}
    for end, frame in facts["endpoints"]:
        d = out / "bootstrap" / end
        bsig = sc.bootstrap_signature(src, facts["cameras"], end, frame, sensor, bopt)
        if "bootstrap" in stages:
            if cached_stage(f"bootstrap_{end}", d, bsig, a.dry_run, man) == "run":
                if not a.dry_run:
                    sc.record(d, bsig, "running")
                rc = run([PY, "tools/train_tracking_bootstrap.py", "--source", src, "--frame", frame,
                          "--endpoint", end, "--cameras", ",".join(facts["cameras"]), "--out", d, *sensor_args,
                          "--num-gaussians", a.num_gaussians, "--knn", a.knn, "--refine-iters", a.refine_iters,
                          "--checkpoints", a.checkpoints, "--heldout-fraction", a.heldout_fraction,
                          "--silhouette-sigma", a.silhouette_sigma, "--thickness-weight", a.thickness_weight,
                          "--rgb-model-sigma", a.rgb_model_sigma, "--depth-model-sigma", a.depth_model_sigma,
                          "--p0-prior-sigma-pos", a.p0_prior_sigma_pos, "--p0-prior-sigma-rot", a.p0_prior_sigma_rot,
                          "--lr-scale", a.lr_scale, "--seed", a.seed, "--gt-labels", bopt["gt_labels"]],
                         out / "logs" / f"bootstrap_{end}.log", a.dry_run)
                if not a.dry_run:
                    bad = sc.validate("bootstrap", d, bsig) if rc == 0 else ["bootstrap tool failed"]
                    sc.record(d, bsig, "complete" if not bad else "failed", {"problems": bad} if bad else None)
                    man["stages"][f"bootstrap_{end}"] = {"status": "complete" if not bad else "failed",
                                                         "signature_sha256": sc.sig_hash(bsig), "problems": bad}
                    save()
                    if bad:
                        return 1
        # the driver reads the endpoint models from <root>/<scene>/static_bidi/<end> (copies: exFAT);
        # an existing copy must be byte-identical to this bootstrap, never silently stale
        dst = out / "pipeline" / scene / "static_bidi" / end
        if not a.dry_run and d.is_dir():
            keys = [f"point_cloud/iteration_{a.refine_iters}/point_cloud.ply", "geometry_information.npz",
                    "bootstrap_manifest.json"]
            if dst.exists():
                stale = [k for k in keys if sc.sha_file(dst / k) != sc.sha_file(d / k)]
                if stale:
                    raise SystemExit(f"{dst} differs from {d} ({', '.join(stale)}): a stale endpoint copy -- "
                                     "choose a new --out")
            else:
                shutil.copytree(d, dst, ignore=shutil.ignore_patterns("source_responsibility.npz",
                                                                      "stage_signature.json*"))
    # ---- Phase 1/2/3 through the driver ---------------------------------------
    ph = [s for s in stages if s in ("phase1", "phase2", "phase3")]
    if ph:
        cmd = ["bash", "run_bidirectional_all.sh", "-i", src.parent, "-o", out / "pipeline", "--scenes", scene,
               "--stages", " ".join(ph), "--cameras", ",".join(facts["cameras"]),
               "--motion-start-frame", a.motion_start_frame, "--motion-end-frame", a.motion_end_frame,
               "--direction", ("both" if len(facts["directions"]) == 2 else facts["directions"][0]),
               "--frame-timestamps", a.frame_timestamps, *sensor_args, "--bootstrap-p0",
               "--p0-velocity-sigma", *a.p0_velocity_sigma,
               "--rgb-model-sigma", a.rgb_model_sigma, "--depth-model-sigma", a.depth_model_sigma,
               "--observation-mode", a.observation_mode,
               "--phase1-partition", a.phase1_partition, "--phase1-alpha-precision", a.phase1_alpha_precision,
               "--phase1-rgb-support", a.phase1_rgb_support,
               *(("--phase1-alpha-sd", a.phase1_alpha_sd) if a.phase1_alpha_sd is not None else ()),
               *(("--phase1-covariance-transport",) if a.phase1_covariance_transport else ()),
               "--phase1-first-return-mode", a.phase1_first_return_mode,
               *(("--phase1-first-return-prior", a.phase1_first_return_prior) if a.phase1_first_return_prior else ())]
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
