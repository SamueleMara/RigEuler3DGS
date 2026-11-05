"""Content-safe reuse of the calibration and bootstrap stages of tools/run_noise_aware_rigeuler.py.

The same contract as tools/stage_signature.py (which guards Phase 1/2/3): a stage
directory is reused only if it holds `stage_signature.json` with status "complete",
a signature EQUAL to the requested one, and outputs that still match the hashes
recorded when the stage completed. Everything else is refused with the differing
fields; nothing is ever deleted. An interrupted attempt with the SAME signature is
moved aside (<dir>.incomplete-<UTC time>) and redone.

Signatures hash file CONTENT (never names or mtimes) and fingerprint the
implementation by the bytes of the source files the stage executes -- not the git
commit, so an unrelated commit does not invalidate a result, while any change to
the code that produced it does.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCHEMA = 1
FILE = "stage_signature.json"
EXT = (".png", ".jpg", ".jpeg")

IMPL = {
    "calibration": ["tools/calibrate_static_rgbd.py", "utils/noise/sensor_model.py"],
    "bootstrap": ["tools/train_tracking_bootstrap.py", "estimator/surface_bootstrap.py",
                  "utils/noise/sensor_model.py", "estimator/observation_terms.py", "estimator/renderer_bridge.py",
                  "estimator/sparse_iekf.py", "gaussian_renderer/__init__.py", "scene/__init__.py",
                  "scene/gaussian_model.py", "scene/dataset_readers.py", "tools/run_iekf_scene.py",
                  "observation_diagnostics/renderer_access.py"],
}


def sha_file(p: Path) -> str | None:
    p = Path(p)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def implementation(stage: str) -> dict:
    return {f: sha_file(REPO / f) for f in IMPL[stage]}


def _frame_file(src: Path, kind: str, cam: str, t: int) -> Path | None:
    if kind == "depth_npy":
        p = src / "depth_npy" / cam / f"frame_{t:06d}.npy"
        return p if p.is_file() else None
    return next((src / kind / cam / f"frame_{t:06d}{e}" for e in EXT
                 if (src / kind / cam / f"frame_{t:06d}{e}").is_file()), None)


def frames_content(src: Path, cams, frames) -> str:
    """sha256 over (camera, frame, kind, file sha) of every rgb / depth / mask file."""
    h = hashlib.sha256()
    for c in cams:
        for t in frames:
            for kind in ("rgb", "depth_npy", "mask"):
                p = _frame_file(src, kind, c, t)
                h.update(f"{c}/{t}/{kind}:{sha_file(p) if p else 'MISSING'};".encode())
    return h.hexdigest()


def calibration_signature(src: Path, cams, start: int, end: int, options: dict) -> dict:
    return {"schema": SCHEMA, "stage": "calibration", "source": str(src), "cameras": list(cams),
            "interval": [int(start), int(end)],
            "inputs_sha256": frames_content(src, cams, range(int(start), int(end) + 1)),
            "options": dict(options), "implementation": implementation("calibration")}


def bootstrap_signature(src: Path, cams, endpoint: str, frame: int, sensor: dict, options: dict,
                        colmap_subdir: str = "colmap") -> dict:
    sp = src / colmap_subdir / "sparse" / "0"
    return {"schema": SCHEMA, "stage": "bootstrap", "source": str(src), "cameras": list(cams),
            "endpoint": endpoint, "frame": int(frame),
            "camera_geometry_sha256": {f: sha_file(sp / f) for f in ("cameras.bin", "images.bin")},
            "inputs_sha256": frames_content(src, cams, [int(frame)]),
            "sensor": dict(sensor), "options": dict(options), "implementation": implementation("bootstrap")}


def sig_hash(sig: dict) -> str:
    return hashlib.sha256(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()


def _diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
    elif a != b:
        out.append(f"{path}: {str(a)[:50]} -> {str(b)[:50]}")
    return out


# ---------------------------------------------------------------------------
# structural validation of completed outputs
# ---------------------------------------------------------------------------
def outputs(stage: str, d: Path, options: dict) -> dict:
    """{relative path: sha256} of the stage's mandatory outputs (None = missing)."""
    if stage == "calibration":
        names = ["sensor_model.json", "validation.json", "covariance.npz"]
    else:
        it = int(options["refine_iters"])
        names = [f"point_cloud/iteration_{it}/point_cloud.ply", "point_cloud.ply",
                 "geometry_information.npz", "bootstrap_manifest.json"]
    return {n: sha_file(d / n) for n in names}


def validate(stage: str, d: Path, sig: dict) -> list:
    """Problems with a stage's outputs beyond hash equality (empty = sound)."""
    bad = []
    if stage == "calibration":
        try:
            import sys
            sys.path.insert(0, str(REPO))
            from utils.noise.sensor_model import SceneNoiseModel
            m = SceneNoiseModel.from_json(d / "sensor_model.json")       # verifies the content id
            if sorted(m.cameras) != sorted(sig["cameras"]):
                bad.append(f"sensor model cameras {sorted(m.cameras)} != {sorted(sig['cameras'])}")
        except Exception as ex:                                            # noqa: BLE001
            bad.append(f"sensor_model.json: {type(ex).__name__}: {ex}"[:200])
        return bad
    import numpy as np
    it = int(sig["options"]["refine_iters"])
    ply = d / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
    try:
        from plyfile import PlyData
        n = len(PlyData.read(str(ply))["vertex"])
    except Exception as ex:                                                # noqa: BLE001
        return [f"{ply}: unreadable ({type(ex).__name__})"]
    if sha_file(ply) != sha_file(d / "point_cloud.ply"):
        bad.append("top-level point_cloud.ply is not the requested checkpoint")
    try:
        P = np.load(d / "geometry_information.npz")["P0_geometry"]
        if P.shape != (n, 6, 6):
            bad.append(f"P0_geometry {P.shape} != ({n}, 6, 6)")
    except Exception as ex:                                                # noqa: BLE001
        bad.append(f"geometry_information.npz: {type(ex).__name__}")
    try:
        m = json.loads((d / "bootstrap_manifest.json").read_text())
        if m.get("num_gaussians") != n:
            bad.append(f"manifest num_gaussians {m.get('num_gaussians')} != PLY {n}")
        if m.get("sensor_model_id") != sig["sensor"].get("sensor_model_id"):
            bad.append("manifest sensor_model_id differs from the requested sensor model")
        if m.get("endpoint_frame") != sig["frame"] or m.get("endpoint") != sig["endpoint"]:
            bad.append("manifest endpoint / frame differ")
    except Exception as ex:                                                # noqa: BLE001
        bad.append(f"bootstrap_manifest.json: {type(ex).__name__}")
    return bad


# ---------------------------------------------------------------------------
# decide / record
# ---------------------------------------------------------------------------
def decide(d: Path, sig: dict) -> tuple[str, str]:
    """('run' | 'reuse' | 'redo' | 'refuse', message). Read-only."""
    d = Path(d)
    if not d.exists() or not any(d.iterdir()):
        return "run", "nothing there"
    f = d / FILE
    if not f.is_file():
        return "refuse", (f"{d} holds outputs without {FILE} (legacy or foreign): refusing to reuse or "
                          "overwrite it -- choose a new --out")
    rec = json.loads(f.read_text())
    if rec.get("signature_sha256") != sig_hash(sig):
        diffs = _diff(rec.get("signature", {}), sig)
        return "refuse", (f"{d}: produced with different inputs/settings -- " + "; ".join(diffs[:6]) +
                          (" ..." if len(diffs) > 6 else "") + " -- choose a new --out")
    if rec.get("status") != "complete":
        return "redo", f"{d}: own {rec.get('status')} attempt with the same signature"
    now = outputs(sig["stage"], d, sig["options"])
    if now != rec.get("outputs"):
        changed = [k for k in now if now[k] != (rec.get("outputs") or {}).get(k)]
        return "refuse", f"{d}: outputs changed or missing since completion ({', '.join(changed)}) -- choose a new --out"
    bad = validate(sig["stage"], d, sig)
    if bad:
        return "refuse", f"{d}: complete but invalid ({'; '.join(bad[:3])}) -- choose a new --out"
    return "reuse", f"{d}: complete, same signature, outputs intact"


def move_aside(d: Path) -> Path:
    """Never delete: an interrupted same-signature attempt is renamed, then redone."""
    dst = Path(f"{d}.incomplete-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}")
    os.replace(d, dst)
    return dst


def record(d: Path, sig: dict, status: str, extra: dict | None = None) -> None:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"schema": SCHEMA, "status": status, "signature_sha256": sig_hash(sig), "signature": sig,
           "recorded": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}
    if status == "complete":
        rec["outputs"] = outputs(sig["stage"], d, sig["options"])
    tmp = d / (FILE + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, d / FILE)
