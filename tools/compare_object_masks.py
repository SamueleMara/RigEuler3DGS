#!/usr/bin/env python
"""A/B of the motion stages with and without the hard negative-ray constraint
(--object-mask-constraint visible-negative-hard), evaluation only.

Reads, for every scene x direction present in BOTH results roots, the scores
each stage already wrote after its inference finished:

  Phase 1   <dir>/evaluation/confusion_matrix.json (coherence support vs the
            labelled-3DGS GT)                      -> support precision/recall/F1
  Phase 2   <dir>_rigid/joint_reduction/gt_evaluation.json, summary.json
            -> axis angle, axis-line distance, increment RMS of the GT-type
               hypothesis; gamma_1DOF, unexplained share (GT-free)
  Phase 3   <dir>_joint/evaluation_gt/phase3_gt_evaluation.json
            -> axis angle, line distance, coordinate MAE of the GT-type fit
  with      visibility_constraint.json per stage (GT-free): share of frames that
            end FEASIBLE and the mean number of violating negative rays

The GT type is used only to pick which of the two reported hypotheses to
score; nothing here feeds back into a run.

    python tools/compare_object_masks.py --baseline <root without masks> \\
        --masks <root with masks> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

METRICS = [  # (key, label, better)
    ("p1_precision", "P1 support precision", "high"), ("p1_recall", "P1 support recall", "high"),
    ("p1_f1", "P1 support F1", "high"),
    ("p2_axis_deg", "P2 axis error [deg]", "low"), ("p2_line_m", "P2 axis-line error [m]", "low"),
    ("p2_incr_rms", "P2 increment RMS", "low"), ("p2_gamma", "P2 gamma_1DOF", "high"),
    ("p2_unexplained", "P2 unexplained share", "low"),
    ("p3_axis_deg", "P3 axis error [deg]", "low"), ("p3_line_m", "P3 axis-line error [m]", "low"),
    ("p3_coord_mae", "P3 coordinate MAE", "low"),
    ("vis_p1_feasible", "P1 frames FEASIBLE (constraint)", "high"),
    ("vis_p2_feasible", "P2 frames FEASIBLE (constraint)", "high"),
    ("vis_p3_feasible", "P3 frames FEASIBLE (constraint)", "high"),
    ("vis_p2_violating", "P2 mean violating rays", "low"),
]


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def scores(bidi: Path, d: str) -> dict:
    r = {}
    c = _load(bidi / d / "evaluation" / "confusion_matrix.json")
    if c and "coherence" in c:
        m = c["coherence"]
        tp, fp, fn = m.get("tp", 0), m.get("fp", 0), m.get("fn", 0)
        r["p1_precision"] = tp / (tp + fp) if tp + fp else None
        r["p1_recall"] = tp / (tp + fn) if tp + fn else None
        if r["p1_precision"] is not None and r["p1_recall"] is not None and r["p1_precision"] + r["p1_recall"] > 0:
            r["p1_f1"] = 2 * r["p1_precision"] * r["p1_recall"] / (r["p1_precision"] + r["p1_recall"])
    def vis_stats(js):
        fr = (js or {}).get("frames") or []
        if not fr:
            return None, None
        feas = sum(1 for f in fr if f.get("status") == "FEASIBLE") / len(fr)
        viol = sum((f.get("final_audit") or {}).get("violating_now", 0) for f in fr) / len(fr)
        return feas, viol
    r["vis_p1_feasible"], _ = vis_stats(_load(bidi / d / "visibility_constraint.json"))
    g = _load(bidi / f"{d}_rigid" / "joint_reduction" / "gt_evaluation.json")
    gt_type = None
    if g and g.get("available"):
        gt_type = g.get("gt_joint_type")
        h = g.get(gt_type) or {}
        r["p2_axis_deg"] = _num(h.get("axis_angle_deg"))
        r["p2_line_m"] = _num((h.get("axis_line_error") or {}).get("distance"))
        r["p2_incr_rms"] = _num((h.get("q_match") or {}).get("increment_rms_error"))
    s = _load(bidi / f"{d}_rigid" / "summary.json")
    if s:
        r["p2_gamma"] = _num((s.get("one_dof") or {}).get("gamma_1dof"))
        r["p2_unexplained"] = _num((s.get("support_rigidity") or {}).get("unexplained_fraction_final"))
    r["vis_p2_feasible"], r["vis_p2_violating"] = vis_stats(
        _load(bidi / f"{d}_rigid" / "phase2_rigid" / "visibility_constraint.json"))
    e3 = _load(bidi / f"{d}_joint" / "evaluation_gt" / "phase3_gt_evaluation.json")
    if e3 and e3.get("fits"):
        t = gt_type or (e3.get("gt_joint") or {}).get("type")
        f = e3["fits"].get(t) or {}
        r["p3_axis_deg"] = _num(f.get("axis_angle_deg"))
        r["p3_line_m"] = _num(f.get("line_distance_m"))
        r["p3_coord_mae"] = _num(f.get("coord_mae")) if f.get("matches_gt_type") else None
        r["vis_p3_feasible"], _ = vis_stats(_load(bidi / f"{d}_joint" / str(t) / "visibility_constraint.json"))
    r["gt_type"] = gt_type
    return r


def fmt(x, key):
    if x is None:
        return "n/a"
    if key.endswith("_m"):
        return f"{100 * x:.2f} cm"
    if key.endswith("_deg"):
        return f"{x:.3f}"
    return f"{x:.4f}" if abs(x) < 10 else f"{x:.1f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", required=True, help="results root WITHOUT --object-masks")
    ap.add_argument("--masks", required=True, help="results root WITH --object-masks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default=None, help="space-separated subset")
    args = ap.parse_args(argv)
    A, B, out = Path(args.baseline), Path(args.masks), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = (args.scenes.split() if args.scenes else
              sorted(p.name for p in B.iterdir() if (p / "bidirectional").is_dir() and (A / p.name / "bidirectional").is_dir()))
    rows = []
    for sc in scenes:
        for d in ("forward", "backward"):
            a, b = scores(A / sc / "bidirectional", d), scores(B / sc / "bidirectional", d)
            row = {"scene": sc, "direction": d, "gt_type": a.get("gt_type") or b.get("gt_type")}
            for k, _l, _b in METRICS:
                row[f"{k}__without"] = a.get(k)
                row[f"{k}__with"] = b.get(k)
            rows.append(row)
    with open(out / "object_masks_ab.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["scene"])
        w.writeheader()
        w.writerows(rows)
    # summary: per metric, how many runs improve / worsen, and the medians
    L = [f"# Negative-ray constraint A/B\n", f"baseline (without): `{A}`  \nwith masks: `{B}`  \n"
         f"{len(rows)} scene x direction runs. GT used for scoring only.\n",
         "\n## Summary per metric\n", "| metric | runs | better with masks | worse | median without | median with |",
         "|---|---|---|---|---|---|"]
    for k, label, better in METRICS:
        pairs = [(r[f"{k}__without"], r[f"{k}__with"]) for r in rows
                 if r[f"{k}__without"] is not None and r[f"{k}__with"] is not None]
        if not pairs:
            L.append(f"| {label} | 0 | | | | |")
            continue
        sgn = 1 if better == "high" else -1
        up = sum(1 for x, y in pairs if sgn * (y - x) > 1e-9)
        dn = sum(1 for x, y in pairs if sgn * (y - x) < -1e-9)
        med = lambda v: sorted(v)[len(v) // 2]
        L.append(f"| {label} | {len(pairs)} | {up} | {dn} | {fmt(med([x for x, _ in pairs]), k)} | "
                 f"{fmt(med([y for _, y in pairs]), k)} |")
    for title, keys in (("Phase 2 joint (GT-type hypothesis)", ["p2_axis_deg", "p2_line_m", "p2_gamma"]),
                        ("Phase 3 per-frame displacement", ["p3_coord_mae", "p3_axis_deg"]),
                        ("Phase 1 support", ["p1_precision", "p1_recall"]),
                        ("Constraint status (GT-free)", ["vis_p1_feasible", "vis_p2_feasible", "vis_p3_feasible"])):
        L += [f"\n## {title}\n", "| scene | dir | " + " | ".join(f"{l} without / with" for k, l, _ in METRICS if k in keys) + " |",
              "|---|---|" + "---|" * len(keys)]
        for r in rows:
            L.append(f"| {r['scene']} | {r['direction'][0].upper()} | " +
                     " | ".join(f"{fmt(r[k + '__without'], k)} / {fmt(r[k + '__with'], k)}" for k in keys) + " |")
    (out / "object_masks_ab.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:6 + len(METRICS)]))
    print(f"\nwrote {out / 'object_masks_ab.md'} and object_masks_ab.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
