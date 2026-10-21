#!/usr/bin/env python
"""Fair original-vs-masked A/B of the motion stages, evaluation only.

CODEX_FIX_MASKED_IEKF_DEFAULT_AND_VALIDATION_2026-10-21.md, P1-4. Both arms
must be FRESH Phase 1/2/3 runs (scripts/run_object_mask_ab.sh) from the same
static endpoints; this tool checks that before it scores anything:

  manifests  every stage dir of both arms holds a COMPLETE stage_manifest.json
             (tools/stage_signature.py); the two signatures may differ only in
             the observation block, the output-root paths and the upstream hash
             that follows from them -- anything else is reported as an
             UNINTENDED difference and fails the comparison
  endpoint   the static PLY / cfg_args hashes are identical in both arms

Scores, per scene x direction:

  GT (evaluation only; never used in fitting)
    Phase 1   <dir>/evaluation/confusion_matrix.json      support precision/recall/F1
    Phase 2   <dir>_rigid/joint_reduction/gt_evaluation.json, summary.json
    Phase 3   <dir>_joint/evaluation_gt/phase3_gt_evaluation.json
  GT-free (estimator/mask_diagnostics.py: the SAME definition in both modes, the
  tracked centres AND orientations re-rendered on one fixed evaluation background)
    alpha outside the mask (mean, sum), leaking pixel share, foreground alpha
    coverage, RGB MAE on foreground / background of the same composite, depth
    MAE on valid in-mask depth, valid-depth counts and the share of originally
    valid depth the mask discards
  resources  stage_resources.json: wall time, peak GPU memory

No total energy is compared: the two arms optimise different objectives.
Missing files are reported as MISSING (and make the exit status 1), never
silently treated as an absent measurement.

    python tools/compare_object_masks.py --original <root> --masked <root> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

STAGES = (("phase1", ""), ("phase2", "_rigid"), ("phase3", "_joint"))
ALLOWED_DIFF_PREFIXES = ("observation.", "endpoint.model_dir", "upstream_signature_sha256")

METRICS = [  # (key, label, better)
    ("p1_precision", "P1 support precision", "high"), ("p1_recall", "P1 support recall", "high"),
    ("p1_f1", "P1 support F1", "high"),
    ("p2_axis_deg", "P2 axis error [deg]", "low"), ("p2_line_m", "P2 axis-line error [m]", "low"),
    ("p2_incr_rms", "P2 increment RMS", "low"), ("p2_gamma", "P2 gamma_1DOF", "high"),
    ("p2_unexplained", "P2 unexplained share", "low"),
    ("p3_axis_deg", "P3 axis error [deg]", "low"), ("p3_line_m", "P3 axis-line error [m]", "low"),
    ("p3_coord_mae", "P3 coordinate MAE", "low"),
]
GT_FREE = [  # (key in mean_over_frames, label, better)
    ("alpha_outside_mean", "alpha outside mask (mean)", "low"),
    ("alpha_outside_sum", "alpha outside mask (sum/frame)", "low"),
    ("leaking_fraction", "leaking bg pixel share", "low"),
    ("foreground_alpha_mean", "foreground alpha coverage", "high"),
    ("rgb_mae_foreground", "RGB MAE foreground", "low"),
    ("rgb_mae_background", "RGB MAE background", "low"),
    ("depth_mae_in_mask", "depth MAE in mask [m]", "low"),
    ("depth_discarded_by_mask", "valid depth discarded by mask", None),
]
for _s in ("p1", "p2", "p3"):
    METRICS += [(f"{_s}_{k}", f"{_s.upper()} {l}", b) for k, l, b in GT_FREE]
    METRICS += [(f"{_s}_wall_s", f"{_s.upper()} wall time [s]", "low"),
                (f"{_s}_gpu_mib", f"{_s.upper()} peak GPU [MiB]", "low")]


def _load(p: Path, problems: list | None = None, what: str = ""):
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        if problems is not None:
            problems.append(f"MISSING {what or p}")
        return None
    except Exception as ex:                          # noqa: BLE001
        if problems is not None:
            problems.append(f"UNREADABLE {p} ({type(ex).__name__})")
        return None


def _num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:                                # noqa: BLE001
        return None


def _flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def manifest_check(bo: Path, bm: Path, d: str) -> tuple[list, list, dict]:
    """(problems, intended differences, endpoint hashes) for one direction."""
    problems, intended, endpoint = [], [], {}
    for stage, suffix in STAGES:
        mo = _load(bo / f"{d}{suffix}" / "stage_manifest.json", problems, f"original {d}{suffix}/stage_manifest.json")
        mm = _load(bm / f"{d}{suffix}" / "stage_manifest.json", problems, f"masked {d}{suffix}/stage_manifest.json")
        if mo is None or mm is None:
            continue
        for arm, m in (("original", mo), ("masked", mm)):
            if m.get("status") != "complete":
                problems.append(f"{arm} {d}{suffix}: manifest status {m.get('status')!r}, not complete")
        so, sm = _flat(mo.get("signature", {})), _flat(mm.get("signature", {}))
        for k in sorted(set(so) | set(sm)):
            if so.get(k) == sm.get(k):
                continue
            line = f"{stage}: {k}: {str(so.get(k))[:70]} -> {str(sm.get(k))[:70]}"
            if k.startswith(ALLOWED_DIFF_PREFIXES):
                intended.append(line)
            else:
                problems.append(f"UNINTENDED {d} {line}")
        mode = mo.get("signature", {}).get("observation", {}).get("mode")
        if mode != "original":
            problems.append(f"original arm {d}{suffix} ran mode {mode!r}")
        if stage == "phase1":
            endpoint = {"ply_sha256": (so.get("endpoint.ply_sha256"), sm.get("endpoint.ply_sha256")),
                        "cfg_args_sha256": (so.get("endpoint.cfg_args_sha256"), sm.get("endpoint.cfg_args_sha256"))}
            for k, (a, b) in endpoint.items():
                if a is None or a != b:
                    problems.append(f"{d}: static endpoint {k} differs or is missing ({a} vs {b})")
    return problems, intended, endpoint


def _gt_free(r: dict, prefix: str, md: dict | None, res: dict | None, problems: list, where: str):
    if md is None:
        return
    if md.get("status") != "OK":
        problems.append(f"{where}: mask diagnostics {md.get('status')} ({md.get('reason')})")
        return
    for k, _l, _b in GT_FREE:
        r[f"{prefix}_{k}"] = _num((md.get("mean_over_frames") or {}).get(k))
    if res is not None:
        r[f"{prefix}_wall_s"] = _num(res.get("wall_s"))
        r[f"{prefix}_gpu_mib"] = _num(res.get("peak_gpu_allocated_mib"))


def scores(bidi: Path, d: str, arm: str, problems: list, notes: list) -> dict:
    r = {}
    gt3 = _load(bidi / d / "evaluation" / "optimized_3dgs_gt.json", problems,
                f"{arm} {d}/evaluation/optimized_3dgs_gt.json")
    if gt3 is not None and gt3.get("status") != "FOUND":
        # e.g. backward: the FINAL endpoint carries no labelled-3DGS GT -- a
        # recorded absence of GT, not a missing output
        notes.append(f"{arm} {d}: Phase-1 support GT {gt3.get('status')}")
        c = None
    else:
        c = _load(bidi / d / "evaluation" / "confusion_matrix.json", problems,
                  f"{arm} {d}/evaluation/confusion_matrix.json")
    if c and "coherence" in c:
        m = c["coherence"]
        tp, fp, fn = m.get("tp", 0), m.get("fp", 0), m.get("fn", 0)
        r["p1_precision"] = tp / (tp + fp) if tp + fp else None
        r["p1_recall"] = tp / (tp + fn) if tp + fn else None
        if r["p1_precision"] is not None and r["p1_recall"] is not None and r["p1_precision"] + r["p1_recall"] > 0:
            r["p1_f1"] = 2 * r["p1_precision"] * r["p1_recall"] / (r["p1_precision"] + r["p1_recall"])
    _gt_free(r, "p1", _load(bidi / d / "mask_diagnostics.json", problems, f"{arm} {d}/mask_diagnostics.json"),
             _load(bidi / d / "stage_resources.json", problems, f"{arm} {d}/stage_resources.json"),
             problems, f"{arm} {d}")
    g = _load(bidi / f"{d}_rigid" / "joint_reduction" / "gt_evaluation.json", problems,
              f"{arm} {d}_rigid/joint_reduction/gt_evaluation.json")
    gt_type = None
    if g and g.get("available"):
        gt_type = g.get("gt_joint_type")
        h = g.get(gt_type) or {}
        r["p2_axis_deg"] = _num(h.get("axis_angle_deg"))
        r["p2_line_m"] = _num((h.get("axis_line_error") or {}).get("distance"))
        r["p2_incr_rms"] = _num((h.get("q_match") or {}).get("increment_rms_error"))
    s = _load(bidi / f"{d}_rigid" / "summary.json", problems, f"{arm} {d}_rigid/summary.json")
    if s:
        r["p2_gamma"] = _num((s.get("one_dof") or {}).get("gamma_1dof"))
        r["p2_unexplained"] = _num((s.get("support_rigidity") or {}).get("unexplained_fraction_final"))
    p2 = bidi / f"{d}_rigid" / "phase2_rigid"
    _gt_free(r, "p2", _load(p2 / "mask_diagnostics.json", problems,
                            f"{arm} {d}_rigid/phase2_rigid/mask_diagnostics.json"),
             _load(p2 / "stage_resources.json", problems, f"{arm} {d}_rigid/phase2_rigid/stage_resources.json"),
             problems, f"{arm} {d}_rigid")
    e3 = _load(bidi / f"{d}_joint" / "evaluation_gt" / "phase3_gt_evaluation.json", problems,
               f"{arm} {d}_joint/evaluation_gt/phase3_gt_evaluation.json")
    s3 = _load(bidi / f"{d}_joint" / "summary.json", problems, f"{arm} {d}_joint/summary.json")
    t = gt_type or ((e3 or {}).get("gt_joint") or {}).get("type")
    if e3 and e3.get("fits"):
        f = e3["fits"].get(t) or {}
        r["p3_axis_deg"] = _num(f.get("axis_angle_deg"))
        r["p3_line_m"] = _num(f.get("line_distance_m"))
        r["p3_coord_mae"] = _num(f.get("coord_mae")) if f.get("matches_gt_type") else None
    run = (s3 or {}).get("joint_types_run") or []
    t3 = t if t in run else (run[0] if run else None)
    if t3 is None:
        problems.append(f"{arm} {d}_joint: no fitted joint type")
    else:
        _gt_free(r, "p3", _load(bidi / f"{d}_joint" / t3 / "mask_diagnostics.json", problems,
                                f"{arm} {d}_joint/{t3}/mask_diagnostics.json"),
                 _load(bidi / f"{d}_joint" / "stage_resources.json", problems,
                       f"{arm} {d}_joint/stage_resources.json"), problems, f"{arm} {d}_joint")
    r["gt_type"] = gt_type
    return r


def fmt(x, key):
    if x is None:
        return "n/a"
    if key.endswith("_m"):
        return f"{100 * x:.2f} cm"
    if key.endswith("_deg"):
        return f"{x:.3f}"
    if key.endswith(("_wall_s", "_gpu_mib")):
        return f"{x:.0f}"
    return f"{x:.4g}"


def med(v):   # true median: the mean of the middle two for an even count
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--original", "--baseline", dest="original", required=True,
                    help="results root of the explicit --observation-mode original arm")
    ap.add_argument("--masked", "--masks", dest="masked", required=True,
                    help="results root of the masked arm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default=None, help="space-separated subset")
    ap.add_argument("--directions", default="forward backward")
    args = ap.parse_args(argv)
    A, B, out = Path(args.original), Path(args.masked), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = (args.scenes.split() if args.scenes else
              sorted({p.name for root in (A, B) if root.is_dir() for p in root.iterdir()
                      if (p / "bidirectional").is_dir()}))
    rows, problems, intended, notes = [], [], {}, []
    for sc in scenes:
        for d in args.directions.split():
            bo, bm = A / sc / "bidirectional", B / sc / "bidirectional"
            pr, it, ep = manifest_check(bo, bm, d)
            nt = []
            a = scores(bo, d, "original", pr, nt)
            b = scores(bm, d, "masked", pr, nt)
            notes += [f"{sc} {d}: {x}" for x in nt]
            row = {"scene": sc, "direction": d, "gt_type": a.get("gt_type") or b.get("gt_type"),
                   "status": "OK" if not pr else "PROBLEMS", "problems": len(pr),
                   "static_ply_sha256": (ep.get("ply_sha256") or (None,))[0]}
            for k, _l, _b in METRICS:
                row[f"{k}__original"] = a.get(k)
                row[f"{k}__masked"] = b.get(k)
            rows.append(row)
            problems += [f"{sc} {d}: {p}" for p in pr]
            intended[f"{sc} {d}"] = it
    with open(out / "masked_ab.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["scene"])
        w.writeheader()
        w.writerows(rows)
    (out / "masked_ab.json").write_text(json.dumps(
        {"original_root": str(A), "masked_root": str(B), "rows": rows, "problems": problems, "notes": notes,
         "intended_manifest_differences": intended}, indent=1, default=str))
    L = ["# Original vs masked observation A/B (both arms fresh Phase 1/2/3)\n",
         f"original arm: `{A}`  \nmasked arm: `{B}`  \n",
         f"{len(rows)} scene x direction runs; {len(problems)} problem(s). GT used for scoring only; "
         "no objective energies are compared (different objectives).\n"]
    if problems:
        L += ["\n## Problems (the comparison is NOT valid where these apply)\n"] + [f"- {p}" for p in problems]
    if notes:
        L += ["\n## Notes (recorded absence of GT, not a failure)\n"] + [f"- {x}" for x in notes]
    L += ["\n## Manifest differences between the arms (intended)\n"]
    for key, it in intended.items():
        L.append(f"- **{key}**: " + ("; ".join(sorted({x.split(': ', 2)[1] for x in it})) if it else "none"))
    L += ["\n## Summary per metric\n",
          "| metric | runs | better masked | worse | median original | median masked |", "|---|---|---|---|---|---|"]
    for k, label, better in METRICS:
        pairs = [(r[f"{k}__original"], r[f"{k}__masked"]) for r in rows
                 if r[f"{k}__original"] is not None and r[f"{k}__masked"] is not None]
        if not pairs:
            L.append(f"| {label} | 0 | | | | |")
            continue
        if better is None:
            up = dn = "-"
        else:
            sgn = 1 if better == "high" else -1
            up = sum(1 for x, y in pairs if sgn * (y - x) > 1e-9)
            dn = sum(1 for x, y in pairs if sgn * (y - x) < -1e-9)
        L.append(f"| {label} | {len(pairs)} | {up} | {dn} | {fmt(med([x for x, _ in pairs]), k)} | "
                 f"{fmt(med([y for _, y in pairs]), k)} |")
    for title, keys in (("Phase 2 joint (GT-type hypothesis)", ["p2_axis_deg", "p2_line_m", "p2_gamma"]),
                        ("Phase 3 per-frame displacement", ["p3_coord_mae", "p3_axis_deg"]),
                        ("Phase 1 support", ["p1_precision", "p1_recall", "p1_f1"]),
                        ("Background leakage, final state (GT-free)",
                         ["p3_alpha_outside_mean", "p3_foreground_alpha_mean", "p3_rgb_mae_background"]),
                        ("Resources", ["p1_wall_s", "p2_wall_s", "p3_wall_s", "p1_gpu_mib"])):
        L += [f"\n## {title}\n", "| scene | dir | " + " | ".join(
            f"{l} original / masked" for k, l, _ in METRICS if k in keys) + " |", "|---|---|" + "---|" * len(keys)]
        for r in rows:
            L.append(f"| {r['scene']} | {r['direction'][0].upper()} | " +
                     " | ".join(f"{fmt(r[k + '__original'], k)} / {fmt(r[k + '__masked'], k)}" for k in keys) + " |")
    (out / "masked_ab.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12 + len(problems)]))
    print(f"\nwrote {out / 'masked_ab.md'}, masked_ab.csv, masked_ab.json")
    return 1 if problems or not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
