"""One CLI / configuration contract for the Phase 1, 2 and 3 runners.

RigEuler3DGS_Calibrated_Bootstrap_IEKF_Implementation_Spec.md sections 1 and 4.1:
the motion interval, physical time, and ONE frozen sensor measurement model shared
by every IEKF phase, recorded with an explicit policy in each stage's outputs.

    --motion-start-frame S      inclusive original frame ID of the first motion frame
                                (the forward canonical endpoint); default 0
    --motion-end-frame E        inclusive last motion frame (the backward canonical
                                endpoint); default S + --frames - 1
    --frame-timestamps auto     use <source>/metadata/frame_timestamps.csv (or a path);
                                default off = constant dt (synthetic, uniform)
    --sensor-model PATH         calibration/sensor_model.json -> policy calibrated
    --sensor-policy P           ideal (synthetic) | assumed (real, uncalibrated);
                                ignored when --sensor-model is given
    --rgb-model-sigma s         DECLARED model-mismatch scale added in variance to the
    --depth-model-sigma s       sensor term (renderer / appearance error, not sensor
                                noise); default 0
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path


def add_cli(ap, phase1: bool = False) -> None:
    ap.add_argument("--motion-start-frame", type=int, default=0,
                    help="inclusive original frame ID of the first motion frame (forward canonical endpoint)")
    ap.add_argument("--motion-end-frame", type=int, default=None,
                    help="inclusive original frame ID of the last motion frame (backward canonical endpoint); "
                         "default motion-start + --frames - 1")
    ap.add_argument("--frame-timestamps", default="off",
                    help="auto = <source>/metadata/frame_timestamps.csv, or a csv path; off = constant dt")
    ap.add_argument("--sensor-model", default=None,
                    help="calibrated sensor model json (tools/calibrate_static_rgbd.py) shared by Phase 1/2/3")
    ap.add_argument("--sensor-policy", choices=("ideal", "assumed"), default="ideal",
                    help="without --sensor-model: ideal (synthetic) or assumed (real, UNCALIBRATED)")
    ap.add_argument("--rgb-model-sigma", type=float, default=0.0,
                    help="declared RGB model-mismatch sd added in variance (not a sensor statistic)")
    ap.add_argument("--depth-model-sigma", type=float, default=0.0,
                    help="declared depth model-mismatch sd [m] added in variance (not a sensor statistic)")
    if phase1:
        ap.add_argument("--p0-geometry", default=None,
                        help="bootstrap geometry covariance npz (P0_geometry [N,6,6]) for the Phase-1 pose prior")
        ap.add_argument("--p0-velocity-sigma", type=float, nargs=2, default=(1.0, 0.316227766),
                        metavar=("V_MPS", "W_RADPS"),
                        help="DECLARED initial velocity prior sd (m/s, rad/s) used with --p0-geometry")


def window(args, frames_default: int) -> tuple[int, int]:
    """(start, length) of the motion window in original frame IDs."""
    s = int(getattr(args, "motion_start_frame", 0) or 0)
    e = getattr(args, "motion_end_frame", None)
    if s < 0:
        raise SystemExit(f"--motion-start-frame {s} < 0")
    if e is None:
        return s, int(frames_default)
    e = int(e)
    if e <= s:
        raise SystemExit(f"--motion-end-frame {e} must be > --motion-start-frame {s}")
    n = e - s + 1
    if frames_default and int(frames_default) not in (0, n) and getattr(args, "_frames_explicit", True):
        raise SystemExit(f"--frames {frames_default} contradicts the motion interval [{s}, {e}] ({n} frames)")
    return s, n


def sensor(args):
    """(SceneNoiseModel, policy) from the CLI; never a silent fallback."""
    from utils.noise.sensor_model import declared_model, load_sensor_model
    path = getattr(args, "sensor_model", None)
    if path:
        m = load_sensor_model(path)
        if m.policy != "calibrated":
            raise SystemExit(f"--sensor-model {path} has policy {m.policy!r}, expected calibrated")
        return m, "calibrated"
    pol = getattr(args, "sensor_policy", "ideal") or "ideal"
    return declared_model(pol), pol


def configure(args, cfg, camera_names):
    """The runner's IEKFConfig with the shared sensor model and camera identities."""
    model, policy = sensor(args)
    names = tuple(camera_names)
    if model.is_physical:
        unknown = [c for c in names if c not in model.cameras]
        if unknown:
            raise SystemExit(f"sensor model has no calibration for cameras {unknown} "
                             f"(calibrated: {sorted(model.cameras)}); cameras are never substituted")
    return dataclasses.replace(cfg, sensor_model=model if model.is_physical else None, camera_names=names,
                               rgb_model_sigma=float(getattr(args, "rgb_model_sigma", 0.0) or 0.0),
                               depth_model_sigma=float(getattr(args, "depth_model_sigma", 0.0) or 0.0))


def manifest(args, cfg, frame_ids=None) -> dict:
    """What every stage writes as sensor_manifest.json (and folds into its signature)."""
    model, policy = sensor(args)
    recs = (frame_ids or {}).get("frames", []) if isinstance(frame_ids, dict) else []
    return {"sensor_policy": policy, "sensor_model_id": model.model_id,
            "sensor_model_path": getattr(args, "sensor_model", None),
            "status": {"calibrated": "CALIBRATED", "ideal": "SYNTHETIC_NO_SENSOR_NOISE",
                       "assumed": "UNCALIBRATED"}[policy],
            "camera_names": list(getattr(cfg, "camera_names", ())),
            "declared_model_mismatch": {"rgb_sd": float(getattr(cfg, "rgb_model_sigma", 0.0)),
                                        "depth_sd_m": float(getattr(cfg, "depth_model_sigma", 0.0)),
                                        "note": "renderer / appearance-model error, NOT sensor noise"},
            "fallback_sigmas_when_not_calibrated": {"rgb_sigma": cfg.rgb_sigma,
                                                    "depth_sigma_model": cfg.depth_sigma_model,
                                                    "depth_sigma_a": cfg.depth_sigma_a,
                                                    "depth_sigma_b": cfg.depth_sigma_b},
            "motion_interval": {"start": int(getattr(args, "motion_start_frame", 0) or 0),
                                "end": getattr(args, "motion_end_frame", None)},
            "frame_timestamps": getattr(args, "frame_timestamps", "off"),
            "frames_original_indices": [r.get("dataset_frame_index") for r in recs],
            "timestamps_s": [r.get("timestamp_s") for r in recs] if recs and "timestamp_s" in recs[0] else None,
            "covariance_note": ("independent per-pixel diagonal measurement model; shared-camera "
                                "correlations (common-mode offset, calibration) are NOT represented")}


def write_manifest(path, args, cfg, frame_ids=None) -> dict:
    m = manifest(args, cfg, frame_ids)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(m, indent=1, default=str))
    return m


def check_canonical_endpoint(model_dir, direction: str, start: int, end: int, explicit: bool) -> dict:
    """The model must BE the canonical endpoint of this window: forward = the motion
    start frame, backward = the motion end frame. Evidence: the bootstrap's own
    manifest (bootstrap_manifest.json) or the single-frame COLMAP export the model
    was trained from. Enforced whenever a motion interval was given explicitly."""
    from utils.sequence_direction import endpoint_frame_of_colmap, model_source_path
    want = int(start) if str(direction).startswith("f") else int(end)
    m = Path(model_dir) / "bootstrap_manifest.json"
    if m.is_file():
        frame, basis = int(json.loads(m.read_text())["endpoint_frame"]), str(m)
    else:
        src = model_source_path(model_dir)
        frame, basis = (endpoint_frame_of_colmap(src) if src else None), f"COLMAP {src}"
    ev = {"direction": direction, "expected_canonical_frame": want, "model_endpoint_frame": frame,
          "basis": basis, "consistent": frame is None or frame == want}
    if explicit and frame is not None and frame != want:
        raise SystemExit(f"{model_dir} is the endpoint model of original frame {frame} ({basis}), but the "
                         f"{direction} pass of motion interval [{start}, {end}] needs frame {want}")
    if explicit and frame is None:
        raise SystemExit(f"{model_dir}: cannot establish which frame this endpoint model represents "
                         "(no bootstrap_manifest.json, no single-frame COLMAP provenance)")
    return ev
