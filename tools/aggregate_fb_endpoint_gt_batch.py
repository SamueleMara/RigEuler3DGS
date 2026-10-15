#!/usr/bin/env python
"""EVALUATION ONLY: collect every scene's evaluation_gt/ tables of one tag into
<results-root>/endpoint_correspondence_batch/<tag>/. Post hoc reporting only:
no winner, no threshold, nothing written back into any scene.

    python tools/aggregate_fb_endpoint_gt_batch.py --results-root R --tag T [--status scenes_status.tsv]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(p: Path):
    if not p.is_file() or p.stat().st_size == 0:
        return []
    return list(csv.DictReader(open(p)))


def write_csv(p: Path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys or ["scene"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--status", default=None)
    a = ap.parse_args()
    R = Path(a.results_root)
    out = R / "endpoint_correspondence_batch" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    status = {r["scene"]: r for r in csv.DictReader(open(a.status), delimiter="\t")} if a.status and Path(a.status).is_file() else {}
    scenes = sorted({p.parents[1].name for p in R.glob(f"*/endpoint_correspondence/{a.tag}/run_state.json")} | set(status))
    agg = {k: [] for k in ("joint", "support", "geometry")}
    cross, rows = [], []
    for s in scenes:
        d = R / s / "endpoint_correspondence" / a.tag
        ev = d / "evaluation_gt"
        sealed = (d / "run_state.json").is_file() and json.loads((d / "run_state.json").read_text()).get("complete")
        summ = json.loads((ev / "gt_evaluation_summary.json").read_text()) if (ev / "gt_evaluation_summary.json").is_file() else {}
        rows.append({"scene": s, "diagnosis_sealed": bool(sealed), "gt_status": summ.get("gt_status", "NOT_EVALUATED"),
                     "evaluation_status": summ.get("status", "NOT_EVALUATED"),
                     "primary_outputs_unchanged": summ.get("primary_outputs_unchanged"),
                     **({k: status[s][k] for k in ("preflight", "diagnosis", "gt_evaluation", "minutes")} if s in status else {}),
                     "diagnosis_dir": str(d), "evaluation_dir": str(ev) if ev.is_dir() else ""})
        for k, f in (("joint", "metrics_joint.csv"), ("support", "metrics_support.csv"), ("geometry", "metrics_geometry.csv")):
            for r in read_csv(ev / f):
                agg[k].append({"scene": s, **r})
        tab = {}
        for r in read_csv(ev / "metrics_patches.csv"):
            lab = r.get("frozen_J0_label") or r.get("frozen_screen_J0_label")
            key = (r["type"], lab, r.get("posthoc_gt_reading") or r.get("status"))
            tab[key] = tab.get(key, 0) + 1
        for (ty, lab, rd), n in sorted(tab.items(), key=lambda x: tuple(map(str, x[0]))):
            cross.append({"scene": s, "type": ty, "frozen_label": lab, "posthoc_gt_reading": rd, "patches": n})
    write_csv(out / "scenes_status.csv", rows)
    write_csv(out / "metrics_joint_all.csv", agg["joint"])
    write_csv(out / "metrics_support_all.csv", agg["support"])
    write_csv(out / "metrics_geometry_all.csv", agg["geometry"])
    write_csv(out / "patch_label_vs_gt_reading_all.csv", cross)
    L = [f"# Endpoint correspondence batch `{a.tag}` — GT evaluation summary (EVALUATION ONLY)\n",
         "Post-hoc scores collected from each scene's `evaluation_gt/`. No hypothesis, scene or threshold is "
         "selected here; denominators and abstentions are in the CSVs.\n",
         "| scene | diagnosis sealed | GT | evaluation | primary unchanged | output |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['scene']} | {r['diagnosis_sealed']} | {r['gt_status']} | {r['evaluation_status']} | "
                 f"{r['primary_outputs_unchanged']} | `{r['diagnosis_dir']}` |")
    L.append("\n## Joint estimates (per direction and hypothesis; type never chosen by GT)\n")
    L.append("| scene | direction | hypothesis | estimate | GT type | line angle (deg) | line dist (m) | coord MAE | rot err (deg) | point err (m) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in agg["joint"]:
        L.append(f"| {r['scene']} | {r['direction']} | {r['hypothesis']} | {r.get('estimate_status')} | {r.get('gt_type')} | "
                 f"{r.get('axis_line_angle_deg') or ''} | {r.get('axis_line_distance_m') or ''} | "
                 f"{(r.get('coord_mae') or '')} {r.get('coord_units') or ''} | {r.get('pose_rotation_err_deg_mean') or ''} | "
                 f"{r.get('point_transfer_err_m_mean') or ''} |")
    L.append("\nTables: scenes_status.csv, metrics_joint_all.csv, metrics_support_all.csv, metrics_geometry_all.csv, "
             "patch_label_vs_gt_reading_all.csv.")
    (out / "batch_gt_report.md").write_text("\n".join(L) + "\n")
    print(f"batch tables: {out}  ({len(rows)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
