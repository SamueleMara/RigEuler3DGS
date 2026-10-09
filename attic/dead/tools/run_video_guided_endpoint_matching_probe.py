#!/usr/bin/env python3
"""Probe sparse video-supported endpoint candidate ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.endpoint_video_candidate_search import measure_candidate_recall_on_exhaustive_subset, rank_target_gaussians_from_video_support
from utils.video_endpoint_transport import build_sparse_video_endpoint_gaussian_support


def run(args) -> dict:
    output = Path(args.output_dir).expanduser().resolve()
    if args.synthetic:
        source_own = torch.eye(4)
        target_own = torch.eye(4)
        source_pixels = torch.arange(4)
        target_pixels = torch.tensor([1, 2, 3, 0])
        truth = target_pixels
    else:
        payload = torch.load(args.track_ownership, map_location="cpu")
        source_own = payload["source_ownership"].float()
        target_own = payload["target_ownership"].float()
        source_pixels = payload["track_source_pixels"].long()
        target_pixels = payload["track_target_pixels"].long()
        truth = payload.get("true_target_ids")
    support = build_sparse_video_endpoint_gaussian_support(source_own, target_own, source_pixels, target_pixels, torch.ones_like(source_pixels, dtype=torch.float32), topk_gaussians=int(args.topk_gaussians))
    ranked, score = rank_target_gaussians_from_video_support(support, num_sources=int(source_own.shape[1]), k=int(args.candidates))
    summary = {
        "support_pairs": int(support.source_ids.numel()),
        "sources_with_support": int((ranked[:, 0] >= 0).sum()),
        "top_score_median": float(score[:, 0].median()) if score.numel() else 0.0,
    }
    if truth is not None:
        summary["recall"] = measure_candidate_recall_on_exhaustive_subset(ranked, truth.long(), depths=(1, 4, 8, 16))
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"ranked_targets": ranked, "scores": score, "support": support}, output / "video_endpoint_candidates.pt")
    (output / "video_endpoint_candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--track-ownership", default="")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--topk-gaussians", type=int, default=4)
    p.add_argument("--candidates", type=int, default=16)
    args = p.parse_args()
    if not args.synthetic and not args.track_ownership:
        p.error("--track-ownership is required unless --synthetic is used")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
