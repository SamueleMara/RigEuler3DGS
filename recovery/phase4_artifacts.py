"""Immutable Phase-4 stage artifacts (unified spec 2026-11-11, sections 3.5, 4, 5.7, 6.6, 8).

<phase4_out>/
  4a_original_membership/       written by 4A only
  4b_new_surface_proposals/     written by 4B only (requires a valid 4A complete.json)
  4c_accepted_moving/           written by 4C only (requires valid 4A and 4B complete.json)

A stage directory is IMMUTABLE once its `complete.json` exists:
  * complete.json carries `artifact_sha256` = sha256 over (relative path, sha256 of contents) of every other file
    in the directory (checkpoint/ excluded), plus the stage's own input proof;
  * downstream stages call `require_complete(dir)` which recomputes that hash and refuses on any mismatch, so a
    later stage can neither run on a tampered artifact nor modify one without being detected;
  * a stage that fails its own gate writes `incomplete.json` (reasons) and never `complete.json` -- the absence of
    complete.json is never an empty result or a success.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .runlog import atomic_json

STAGE_A, STAGE_B, STAGE_C = "4a_original_membership", "4b_new_surface_proposals", "4c_accepted_moving"
EXCLUDE_FROM_HASH = ("complete.json", "incomplete.json")
EXCLUDE_DIRS = ("checkpoint",)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tensor_sha256(t) -> str:
    a = t.detach().cpu().contiguous().numpy() if torch.is_tensor(t) else np.ascontiguousarray(t)
    return hashlib.sha256(a.tobytes()).hexdigest()


def canonical_hashes(g) -> dict:
    """Per-tensor sha256 of the canonical Gaussian parameters (raw stored tensors, row order) + combined."""
    out = {k: tensor_sha256(getattr(g, k)) for k in ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc",
                                                     "_features_rest")}
    out["n"] = int(g._xyz.shape[0])
    out["combined"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    return out


def dir_hash(d) -> dict:
    d = Path(d)
    files = {}
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(d).as_posix()
        if rel in EXCLUDE_FROM_HASH or rel.split("/")[0] in EXCLUDE_DIRS:
            continue
        files[rel] = sha256_file(p)
    return {"files": files, "combined": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def write_complete(d, payload: dict) -> dict:
    d = Path(d)
    if (d / "complete.json").exists():
        raise RuntimeError(f"{d}/complete.json exists: a completed stage artifact is immutable")
    if (d / "incomplete.json").exists():
        os.replace(d / "incomplete.json", d / f"incomplete_superseded_{int(time.time())}.json.bak")
    h = dir_hash(d)
    body = {**payload, "artifact_sha256": h["combined"], "artifact_files": h["files"],
            "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_json(d / "complete.json", body)
    return body


def write_incomplete(d, reasons: list, payload: dict | None = None):
    atomic_json(Path(d) / "incomplete.json", {"complete": False, "reasons": list(reasons), **(payload or {}),
                                              "written": time.strftime("%Y-%m-%d %H:%M:%S")})


def require_complete(d, stage_name: str) -> dict:
    """complete.json of a stage directory, after verifying that no file of the artifact changed."""
    d = Path(d)
    p = d / "complete.json"
    if not p.is_file():
        why = ""
        if (d / "incomplete.json").is_file():
            why = f" (incomplete.json: {json.loads((d / 'incomplete.json').read_text()).get('reasons')})"
        raise SystemExit(f"{stage_name}: {d} has no complete.json{why}; refusing to start")
    c = json.loads(p.read_text())
    if c.get("stage") != stage_name.split("_")[0]:
        raise SystemExit(f"{p} belongs to stage {c.get('stage')!r}, not {stage_name}")
    h = dir_hash(d)
    if h["combined"] != c.get("artifact_sha256"):
        changed = sorted(set(h["files"]) ^ set(c.get("artifact_files", {})) |
                         {k for k in h["files"] if c.get("artifact_files", {}).get(k) != h["files"][k]})
        raise SystemExit(f"{d}: artifact hash mismatch -- files changed after completion: {changed[:10]}")
    return c


def save_npz(path, **arrays):
    np.savez_compressed(path, **arrays)


def write_jsonl(path, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=_json_default) + "\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def joint_fingerprint(joint) -> dict:
    """Immutable fingerprint of the fixed Phase-3 joint (files + parameters + the per-frame transform)."""
    from .contracts import sha256
    files = {}
    for k in ("geometry_file", "trajectory_file"):
        f = getattr(joint, k, None)
        files[k] = sha256(f) if f and Path(str(f)).is_file() else None
    d = joint.to_dict()
    T = []
    for f in sorted(joint.frames):
        Q, p = joint.transform(f)
        T.append(np.concatenate([np.asarray(Q, float).ravel(), np.asarray(p, float).ravel()]))
    tsha = hashlib.sha256(np.asarray(T, np.float64).tobytes()).hexdigest()
    return {"files": files, "parameters_sha256": hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest(),
            "transforms_sha256": tsha, "frames": [int(f) for f in joint.frames]}


# ---------------------------------------------------------------------------- shared all-view model audit
def view_audit_row(out: dict, tgt, tol: float, loss_vec) -> dict:
    """Per-view RGB-D-mask residual / coverage / contradiction metrics of one full render."""
    A, Dh = out["alpha"][0], out["depth"][0]
    M, V, D = tgt.M, tgt.V, tgt.D
    cov = A >= 0.5
    vc = V & cov
    n_m = int(M.sum())
    r = (Dh - D)[vc]
    return {"loss": [float(x) for x in loss_vec], "E": float(sum(loss_vec)), "mask_px": n_m, "valid_px": int(V.sum()),
            "mask_coverage": float((cov & M).sum()) / max(1, n_m),
            "offmask_alpha_mean": float(A[~M].mean()) if bool((~M).any()) else 0.0,
            "offmask_covered_px": int((cov & ~M).sum()),
            "uncovered_valid_px": int((V & ~cov).sum()),
            "depth_mae_valid_covered_m": float(r.abs().mean()) if r.numel() else None,
            "front_contradiction_px": int((vc & (Dh < D - tol)).sum()),
            "behind_px": int((vc & (Dh > D + tol)).sum())}
