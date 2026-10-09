"""Section 23 three-stage evaluation of a render_graph run.

Reports the raw render matcher, the seed purity and the final graph labels
separately, because that is what tells you whether a failure came from render
matching, seed selection, or graph propagation.

Ground truth is read HERE ONLY, after inference. It never influences training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def score(pred, gt, mask=None):
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    if pred.size == 0:
        return None
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {
        "n": int(pred.size),
        "accuracy": float((pred == gt).mean() * 100),
        "iou": float(tp / max(tp + fp + fn, 1) * 100),
        "precision": float(p * 100),
        "recall": float(r * 100),
        "f1": float(2 * p * r / max(p + r, 1e-9) * 100),
    }


def _load_ground_truth(run_path: Path, tracks, fallback_ply: str):
    """Per-Gaussian GT if the run saved it, else NN-map the sparse cloud.

    The pipeline tracks ``_gt_part_labels`` through densification and writes them
    into its own output ply as ``gt_part``. That is the correct per-Gaussian
    label. Nearest-neighbour mapping the 1380-point sparse cloud onto 11320
    Gaussians instead disagrees with it on ~14% of them, which is enough to move
    F1 by more than ten points -- so the sparse cloud is only a fallback.
    """
    from plyfile import PlyData

    plys = sorted(run_path.glob("point_cloud/iteration_*/point_cloud.ply"),
                  key=lambda q: int(q.parent.name.rsplit("_", 1)[-1]))
    if plys:
        vertex = PlyData.read(str(plys[-1]))["vertex"]
        if "gt_part" in {p.name for p in vertex.properties}:
            gt = np.asarray(vertex["gt_part"]).astype(np.int64)
            n = int(tracks["canonical_xyz"].shape[0])
            if gt.shape[0] == n:
                return gt, f"per-Gaussian gt_part from {plys[-1].parent.name}"
            print(f"  warning: gt_part has {gt.shape[0]} entries for {n} gaussians; "
                  f"falling back to the sparse cloud")

    if not fallback_ply:
        raise SystemExit(
            "no per-Gaussian gt_part in the run ply and no --gt_ply fallback given")
    from scipy.spatial import cKDTree

    g = PlyData.read(fallback_ply)["vertex"]
    gx = np.stack([g["x"], g["y"], g["z"]], 1)
    gl = np.asarray(g["part"])
    _, idx = cKDTree(gx).query(tracks["canonical_xyz"].numpy())
    return gl[idx], f"NN-mapped from {Path(fallback_ply).name} (APPROXIMATE)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="output/<run name>")
    ap.add_argument("--outer", type=int, default=-1, help="which outer iteration (-1 = last)")
    ap.add_argument("--gt_ply", default="",
                    help="fallback sparse GT; only used if the run ply has no gt_part field")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = REPO / args.run / "dynamic_debug" / "nested_part_embedding_killing"
    files = sorted(d.glob("render_graph_outer_*.pt"))
    if not files:
        raise SystemExit(f"no render_graph_outer_*.pt in {d} (was this a render_graph run?)")
    path = files[args.outer]
    rg = torch.load(path, map_location="cpu", weights_only=False)
    tracks = torch.load(d / "nested_killing_tracks.pt", map_location="cpu", weights_only=False)

    gt, gt_source = _load_ground_truth(REPO / args.run, tracks, args.gt_ply)
    print(f"ground truth: {gt_source}")

    best = rg["best_render_label"].numpy()
    margin = rg["normalized_margin"].numpy()
    comparable = (rg["candidate_valid"].sum(1) >= 2).numpy()
    seeds = rg["seed_mask"].numpy()
    seed_label = rg["seed_label"].numpy()
    final = rg["final_label"].numpy()
    k1 = int(rg["render_preference"].shape[1])

    report = {"run": args.run, "file": path.name,
              "ground_truth": gt_source,
              "coverage": float(comparable.mean() * 100)}

    print(f"=== {args.run}  ({path.name}) ===")
    print(f"\nA. RAW RENDER MATCHER   coverage {comparable.mean()*100:.1f}%")
    report["raw_matcher"] = {}
    for label, frac in (("all valid", 1.0), ("top 75%", 0.75), ("top 50%", 0.50),
                        ("top 25%", 0.25), ("top 10%", 0.10)):
        thr = np.quantile(margin[comparable], 1 - frac) if frac < 1.0 else -1.0
        s = score(best, gt, comparable & (margin >= thr))
        report["raw_matcher"][label] = s
        print(f"   {label:10s} n={s['n']:6d}  acc {s['accuracy']:6.2f}  IoU {s['iou']:6.2f}"
              f"  prec {s['precision']:6.2f}  rec {s['recall']:6.2f}  F1 {s['f1']:6.2f}")

    print("\nB. SEED PURITY")
    report["seeds"] = {"total": int(seeds.sum())}
    for c in range(k1):
        m = seeds & (seed_label == c)
        if int(m.sum()):
            purity = float((gt[m] == c).mean() * 100)
            report["seeds"][f"class_{c}"] = {"n": int(m.sum()), "purity": purity}
            print(f"   class {c}: n={int(m.sum()):6d}  purity {purity:6.2f}%")
    overall = float((seed_label[seeds] == gt[seeds]).mean() * 100) if seeds.any() else 0.0
    report["seeds"]["purity"] = overall
    print(f"   all    : n={int(seeds.sum()):6d}  purity {overall:6.2f}%")

    print("\nC. FINAL GRAPH LABELS")
    s = score(final, gt)
    report["final"] = s
    print(f"   n={s['n']:6d}  acc {s['accuracy']:6.2f}  IoU {s['iou']:6.2f}"
          f"  prec {s['precision']:6.2f}  rec {s['recall']:6.2f}  F1 {s['f1']:6.2f}")
    print(f"\n   moving fraction: predicted {(final==1).mean()*100:.1f}%  "
          f"ground truth {(gt==1).mean()*100:.1f}%")

    print("\nPER-OUTER FINAL LABELS")
    report["per_outer"] = {}
    for f in files:
        rg_i = torch.load(f, map_location="cpu", weights_only=False)
        si = score(rg_i["final_label"].numpy(), gt)
        report["per_outer"][f.stem[-3:]] = si
        print(f"   outer {f.stem[-3:]}  acc {si['accuracy']:6.2f}  IoU {si['iou']:6.2f}"
              f"  prec {si['precision']:6.2f}  rec {si['recall']:6.2f}  F1 {si['f1']:6.2f}")
    best = max(report["per_outer"].items(), key=lambda kv: kv[1]["f1"])
    print(f"   best: outer {best[0]} at F1 {best[1]['f1']:.2f}")
    report["best_outer"] = {"outer": best[0], **best[1]}

    out = Path(args.out) if args.out else d / "render_graph_evaluation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
