#!/usr/bin/env python
"""Experiment signatures for the motion stages (P0-5): never reuse a stage
output that was produced under a different configuration.

A stage directory is reusable only if it holds `stage_manifest.json` with
status "complete", a signature EQUAL to the requested one, and outputs that
still pass the stage's structural validation. The signature covers:

    code        git commit + hash of the uncommitted diff of tracked code
    stage, direction, scene
    endpoint    static model dir, sha256 of its point_cloud.ply and cfg_args
    observation mode, background, hard-constraint options, mask root and a
                content hash of every mask file of the used cameras
    source      dataset dir, frame/camera counts, rgb/depth file listing hash
    config      the driver's IEKF / coherence / Phase-2/3 settings
    upstream    the signature hash of the Phase-1 (for Phase 2) or Phase-2 (for
                Phase 3) manifest this stage consumes

Commands (all read-only except `write`):

    compute ... > sig.json          print the requested signature
    check --dir D --signature F     exit 0 reuse | 10 nothing there | 11 our own
                                    incomplete attempt (same signature, safe to
                                    redo) | 12 refuse (legacy output without a
                                    manifest, or a different signature)
    write --dir D --signature F --status running|complete|failed
                                    atomic; "complete" first validates outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = "stage_manifest.json"
EXT = (".png", ".jpg", ".jpeg")


def _sha_file(p: Path, chunk: int = 1 << 20) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _listing_hash(root: Path, cams, pattern_ext) -> str | None:
    """sha256 over (relative path, size) of every file of the given cameras."""
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    for c in cams:
        d = root / c
        if not d.is_dir():
            h.update(f"missing:{c}".encode())
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in pattern_ext:
                h.update(f"{c}/{f.name}:{f.stat().st_size}".encode())
    return h.hexdigest()


def _content_hash(root: Path, cams) -> str | None:
    """sha256 over the BYTES of every mask of the given cameras (masks are small)."""
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    for c in cams:
        d = root / c
        if not d.is_dir():
            h.update(f"missing:{c}".encode())
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in EXT:
                h.update(f"{c}/{f.name}".encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def _code_version() -> dict:
    def git(*a):
        r = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    diff = git("diff", "HEAD", "--", "*.py", "*.sh", "*.cu", "*.cpp", "*.h", "*.cuh")
    return {"commit": git("rev-parse", "HEAD"),
            "uncommitted_diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest() if diff else None}


def compute(a) -> dict:
    src = Path(a.source)
    cams = sorted(p.name for p in (src / "rgb").glob("cam_*") if p.is_dir())[: int(a.cameras)]
    model = Path(a.model)
    ply = model / "point_cloud" / f"iteration_{a.iterations}" / "point_cloud.ply"
    mode = a.observation_mode
    mask_root = Path(a.mask_source) if a.mask_source else src / "mask"
    needs_masks = mode != "original" or (a.object_mask_constraint or "off") != "off"
    up = None
    if a.upstream:
        m = Path(a.upstream) / MANIFEST
        if m.is_file():
            up = json.loads(m.read_text()).get("signature_sha256")
        else:
            up = "MISSING_UPSTREAM_MANIFEST"
    sig = {
        "schema": 1,
        "code": _code_version(),
        "stage": a.stage, "direction": a.direction, "scene": a.scene,
        "endpoint": {"model_dir": str(model), "ply_sha256": _sha_file(ply),
                     "cfg_args_sha256": _sha_file(model / "cfg_args")},
        "observation": {"mode": mode, "rgb_background_distribution": a.rgb_background_distribution,
                        "rgb_background_constant": a.rgb_background_constant,
                        "object_mask_constraint": a.object_mask_constraint or "off",
                        "mask_options": a.mask_options,
                        "mask_root": str(mask_root) if needs_masks else None,
                        "mask_content_sha256": _content_hash(mask_root, cams) if needs_masks else None},
        "source": {"dir": str(src), "frames": int(a.frames), "cameras": cams,
                   "rgb_listing_sha256": _listing_hash(src / "rgb", cams, EXT),
                   "depth_listing_sha256": _listing_hash(src / "depth_npy", cams, (".npy",))},
        "config": dict(kv.split("=", 1) for kv in (a.config or [])),
        "upstream_signature_sha256": up,
    }
    return sig


def sig_hash(sig: dict) -> str:
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------
# structural validation of a finished stage
# --------------------------------------------------------------------------
def validate(stage: str, d: Path) -> list:
    """Problems with a stage's outputs (empty list = structurally valid)."""
    import numpy as np
    bad = []

    def jload(p):
        try:
            return json.loads(p.read_text())
        except Exception as ex:          # noqa: BLE001
            bad.append(f"{p.name}: unreadable JSON ({type(ex).__name__})")
            return None
    if stage == "phase1":
        j = jload(d / "predicted_support.json")
        if j is not None and not ("status" in j and "dynamic_gaussians" in j):
            bad.append("predicted_support.json: missing status / dynamic_gaussians")
        try:
            np.load(d / "phase1_centers" / "canonical_xyz.npy")
        except Exception:                # noqa: BLE001
            bad.append("phase1_centers/canonical_xyz.npy unreadable")
        if not list((d / "phase1_centers").glob("centers_frame_*.npy")):
            bad.append("no phase1_centers/centers_frame_*.npy")
    elif stage == "phase2":
        try:
            P = np.load(d / "phase2_rigid" / "poses.npy")
            if P.ndim != 3 or P.shape[1:] != (4, 4):
                bad.append(f"phase2_rigid/poses.npy shape {P.shape}")
        except Exception:                # noqa: BLE001
            bad.append("phase2_rigid/poses.npy unreadable")
        if jload(d / "support_coherence" / "support.json") is None:
            pass
        if not (d / "joint_reduction").is_dir():
            bad.append("joint_reduction/ missing")
        jload(d / "summary.json")
    elif stage == "phase3":
        j = jload(d / "summary.json")
        if j is not None and not j:
            bad.append("summary.json empty")
    else:
        bad.append(f"unknown stage {stage}")
    return bad


def check(d: Path, sig: dict) -> tuple[int, str]:
    m = d / MANIFEST
    if not d.exists() or (d.is_dir() and not any(d.iterdir())):
        return 10, "nothing there"
    if not m.is_file():
        return 12, (f"{d} holds outputs without {MANIFEST} (legacy or foreign run): refusing to "
                    "reuse or overwrite it; use a new output root, or FORCE=1 to overwrite")
    rec = json.loads(m.read_text())
    want = sig_hash(sig)
    if rec.get("signature_sha256") != want:
        diffs = _diff(rec.get("signature", {}), sig)
        return 12, (f"{d}: produced with a different configuration -- " + "; ".join(diffs[:6]) +
                    (" ..." if len(diffs) > 6 else "") + " (use a new output root, or FORCE=1)")
    if rec.get("status") != "complete":
        return 11, f"{d}: our own {rec.get('status')} attempt with the same signature -- redoing it"
    bad = validate(sig["stage"], d)
    if bad:
        return 11, f"{d}: manifest says complete but outputs are invalid ({'; '.join(bad[:3])}) -- redoing"
    return 0, f"{d}: complete, same signature -- reusing"


def _diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
    elif a != b:
        out.append(f"{path}: {str(a)[:60]} -> {str(b)[:60]}")
    return out


def write(d: Path, sig: dict, status: str) -> int:
    rec = {"status": status, "signature_sha256": sig_hash(sig), "signature": sig}
    if status == "complete":
        bad = validate(sig["stage"], d)
        if bad:
            rec["status"] = "failed"
            rec["validation_problems"] = bad
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, d / MANIFEST)
    if rec["status"] != status:
        print(f"stage outputs failed validation: {rec['validation_problems']}")
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    for k in ("stage", "direction", "scene", "source", "model"):
        c.add_argument(f"--{k}", required=True)
    c.add_argument("--frames", required=True)
    c.add_argument("--cameras", required=True)
    c.add_argument("--iterations", default="30000")
    c.add_argument("--observation-mode", required=True)
    c.add_argument("--rgb-background-distribution", default=None)
    c.add_argument("--rgb-background-constant", default=None)
    c.add_argument("--object-mask-constraint", default="off")
    c.add_argument("--mask-options", default=None)
    c.add_argument("--mask-source", default=None)
    c.add_argument("--upstream", default=None)
    c.add_argument("--config", nargs="*")
    k = sub.add_parser("check")
    k.add_argument("--dir", required=True)
    k.add_argument("--signature", required=True)
    w = sub.add_parser("write")
    w.add_argument("--dir", required=True)
    w.add_argument("--signature", required=True)
    w.add_argument("--status", required=True, choices=("running", "complete", "failed"))
    a = ap.parse_args(argv)
    if a.cmd == "compute":
        print(json.dumps(compute(a), indent=1))
        return 0
    sig = json.loads(Path(a.signature).read_text())
    if a.cmd == "check":
        rc, msg = check(Path(a.dir), sig)
        print(msg)
        return rc
    return write(Path(a.dir), sig, a.status)


if __name__ == "__main__":
    raise SystemExit(main())
