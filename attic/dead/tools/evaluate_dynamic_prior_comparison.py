#!/usr/bin/env python3
"""Recompute prior-level metrics for a dynamic-prior comparison directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.dynamic_prior import load_dynamic_prior, prior_metrics  # noqa: E402


def _safe(value: Any) -> str:
    if isinstance(value, float):
        return "" if value != value else f"{value:.6g}"
    return "" if value is None else str(value)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--gt-mask", default="")
    args = parser.parse_args()
    root = Path(args.comparison_dir).expanduser().resolve()
    gt_path = Path(args.gt_mask).expanduser() if args.gt_mask else root / "shared" / "gt_moving_mask.pt"
    gt = torch.load(gt_path, map_location="cpu", weights_only=False).bool() if gt_path.is_file() else None
    rows = []
    for prior_path in sorted(root.glob("*/prior.pt")):
        prior = load_dynamic_prior(prior_path)
        metrics = prior_metrics(prior, gt)
        metrics["path"] = str(prior_path)
        rows.append(metrics)
        (prior_path.parent / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2, allow_nan=False), encoding="utf-8")
    fields = [
        "method",
        "path",
        "num_gaussians",
        "valid_fraction",
        "mean_probability",
        "mean_confidence",
        "all_precision",
        "all_recall",
        "all_f1",
        "all_iou",
        "all_pr_auc",
        "all_roc_auc",
        "high_confidence_f1",
    ]
    with (root / "comparison_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _safe(row.get(field)) for field in fields})
    (root / "comparison_summary.json").write_text(
        json.dumps(_clean({"schema_version": 1, "metrics": rows}), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[comparison-eval] wrote metrics for {len(rows)} priors under {root}")


if __name__ == "__main__":
    main()
