#!/usr/bin/env python
"""Machine-readable report of motion-stage runs (CODEX_FIX_MASKED_IEKF_DEFAULT_AND_VALIDATION_2026-10-21.md, F.3).

Reads, for every <root>/<scene>/bidirectional/<direction>{,_rigid,_joint}, only
what the stages themselves wrote, and collects per stage:

  signature     stage_manifest.json: status, signature hash, observation mode,
                code commit / diff hash, static PLY + cfg_args sha256, source
                content hashes, mask content hash, config, upstream hash
  steps         accepted / rejected iterations, stop reasons
  objective     MAP split (RGB, alpha, depth, prior, total) initial -> final
  covariance    Phase 1: per-frame posterior health and the prior source
  depth         observed-valid, used, gated-by-alpha counts
  leakage       mask_diagnostics.json (GT-free, tracked centres + orientations)
  resources     stage_resources.json: wall time, peak GPU memory

Missing files are listed under "missing", never filled in.

    python tools/repair_report.py --root <results root> [--root <another>] --out report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGES = (("phase1", ""), ("phase2", "_rigid"), ("phase3", "_joint"))


def _load(p: Path, missing: list):
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        missing.append(str(p))
    except Exception as ex:                          # noqa: BLE001
        missing.append(f"{p} (unreadable: {type(ex).__name__})")
    return None


def _signature(d: Path, missing: list) -> dict | None:
    m = _load(d / "stage_manifest.json", missing)
    if m is None:
        return None
    s = m.get("signature", {})
    return {"status": m.get("status"), "signature_sha256": m.get("signature_sha256"),
            "validation_problems": m.get("validation_problems"),
            "code": s.get("code"), "endpoint": s.get("endpoint"), "observation": s.get("observation"),
            "source": {k: v for k, v in (s.get("source") or {}).items() if k != "cameras"},
            "cameras": len((s.get("source") or {}).get("cameras") or []),
            "config": s.get("config"), "upstream_signature_sha256": s.get("upstream_signature_sha256")}


def _leak(md):
    if md is None:
        return None
    return {"status": md.get("status"), "reason": md.get("reason"), "mean_over_frames": md.get("mean_over_frames"),
            "per_frame": [{k: r.get(k) for k in ("processing_index", "alpha_outside_mean", "leaking_fraction",
                                                 "foreground_alpha_mean", "depth_valid_original",
                                                 "depth_valid_in_mask")} for r in md.get("rows", [])]}


def phase1(d: Path, missing: list) -> dict:
    fr = _load(d / "phase1_frames.json", missing) or {}
    rows = fr.get("per_frame", [])
    mo = _load(d / "masked_observation.json", []) or {}
    return {
        "signature": _signature(d, missing),
        "frames": fr.get("frames"), "runtime_s": fr.get("runtime_s"),
        "stop_reasons": [r.get("stop_reason") for r in rows],
        "covariance": [{"frame": r.get("frame"), **(r.get("covariance") or {})} for r in rows],
        "objective": mo.get("frames"),
        "leakage": _leak(_load(d / "mask_diagnostics.json", missing)),
        "resources": _load(d / "stage_resources.json", missing),
    }


def _map_frames(p: Path, missing: list):
    j = _load(p, missing)
    if j is None:
        return None
    out = []
    for f in j.get("frames", []):
        it = f.get("iterations") or []
        acc = [h for h in it if h.get("accepted")]
        out.append({"processing_index": f.get("processing_index"),
                    "accepted": len(acc), "rejected": sum(1 for h in it if "accepted" in h and not h["accepted"]),
                    "stop": next((h.get("stop") for h in it if h.get("stop")), None),
                    "objective_initial": (f.get("objective") or {}).get("initial"),
                    "objective_final": (f.get("objective") or {}).get("final"),
                    "depth_used_final": acc[-1].get("depth_used") if acc else None,
                    "depth_gated_final": acc[-1].get("depth_gated_by_alpha") if acc else None,
                    "prior_mode": it[0].get("prior_mode") if it else None})
    return out


def phase2(d: Path, missing: list) -> dict:
    p2 = d / "phase2_rigid"
    return {"signature": _signature(d, missing),
            "frames": _map_frames(p2 / "map_objective.json", missing),
            "model_error_audit": (_load(p2 / "model_error_audit.json", missing) or {}).get(
                "background_leaking_W_over_sensor_mean"),
            "leakage": _leak(_load(p2 / "mask_diagnostics.json", missing)),
            "resources": _load(p2 / "stage_resources.json", missing)}


def phase3(d: Path, missing: list) -> dict:
    s = _load(d / "summary.json", missing) or {}
    hyps = {}
    for t in s.get("joint_types_run") or []:
        hyps[t] = {"frames": _map_frames(d / t / "map_objective.json", missing),
                   "geometry_unchanged_bitwise": (s.get("hypotheses", {}).get(t) or {}).get(
                       "geometry_unchanged_bitwise"),
                   "leakage": _leak(_load(d / t / "mask_diagnostics.json", missing))}
    return {"signature": _signature(d, missing), "hypotheses": hyps,
            "model_error_audit": (_load(d / "model_error_audit.json", missing) or {}).get(
                "background_leaking_W_over_sensor_mean"),
            "resources": _load(d / "stage_resources.json", missing)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--directions", default="forward backward")
    a = ap.parse_args(argv)
    report = {"schema": 1, "roots": {}}
    for root in a.root:
        R = Path(root)
        scenes = sorted(p.name for p in R.iterdir() if (p / "bidirectional").is_dir()) if R.is_dir() else []
        rr = {}
        for sc in scenes:
            for dname in a.directions.split():
                b = R / sc / "bidirectional"
                missing: list = []
                rr[f"{sc}/{dname}"] = {"phase1": phase1(b / dname, missing),
                                       "phase2": phase2(b / f"{dname}_rigid", missing),
                                       "phase3": phase3(b / f"{dname}_joint", missing),
                                       "missing": missing}
        report["roots"][str(R)] = rr
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=str))
    n = sum(len(v) for v in report["roots"].values())
    miss = sum(len(x["missing"]) for v in report["roots"].values() for x in v.values())
    print(f"wrote {a.out}: {n} scene x direction runs, {miss} missing file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
