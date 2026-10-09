"""CODEX_soft_labels section 51/52/53 -- soft init + soft outer, memory OFF vs ON.

    soft init + soft outer + memory OFF
    soft init + soft outer + memory ON

Nothing else differs between the arms, and no Killing-solver change is mixed in
(section 51). Ground truth is read HERE ONLY, after each run.

    python tools/run_soft_ablation.py
    python tools/run_soft_ablation.py --report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.run_render_graph import run  # noqa: E402

SCENES = ("USB_100109", "storage_45135")
ARMS = (("soft_memory_off", False), ("soft_memory_on", True))


def run_name(scene: str, arm: str) -> str:
    return f"soft_{scene}_{arm}"


def _debug(out: str) -> Path:
    return REPO / "output" / out / "dynamic_debug" / "nested_part_embedding_killing"


def _gt(out: str, n: int):
    from plyfile import PlyData

    plys = sorted((REPO / "output" / out).glob("point_cloud/iteration_*/point_cloud.ply"),
                  key=lambda q: int(q.parent.name.rsplit("_", 1)[-1]))
    if not plys:
        return None
    vertex = PlyData.read(str(plys[-1]))["vertex"]
    if "gt_part" not in {p.name for p in vertex.properties}:
        return None
    gt = np.asarray(vertex["gt_part"]).astype(np.int64)
    return gt if gt.shape[0] == n else None


def _score(pred: np.ndarray, gt: np.ndarray) -> dict:
    moving, truth = pred > 0, gt > 0
    tp = int((moving & truth).sum())
    fp = int((moving & ~truth).sum())
    fn = int((~moving & truth).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"accuracy": float((moving == truth).mean() * 100),
            "precision": float(p * 100), "recall": float(r * 100),
            "f1": float(2 * p * r / max(p + r, 1.0e-9) * 100),
            "moving_iou": float(tp / max(tp + fp + fn, 1) * 100)}


def score_run(out: str) -> dict:
    """Sections 52 and 53: label quality plus the soft-state diagnostics."""
    debug = _debug(out)
    files = sorted(debug.glob("soft_outer_*.pt"))
    if not files:
        return {"available": False, "reason": f"no soft_outer_*.pt in {debug}"}

    states = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    n = int(states[0]["soft_part_probs_after"].shape[0])
    gt = _gt(out, n)
    row: dict = {"available": True, "num_outer": len(states), "num_gaussians": n}

    # Section 52: the initialization, before the first inner solve.
    init_path = debug / "initialization" / "soft_binary_init.pt"
    if init_path.is_file():
        init = torch.load(init_path, map_location="cpu", weights_only=False)
        s = init.get("summary", {})
        row["init"] = {
            "mean_entropy": s.get("mean_entropy"),
            "fraction_peak_gt_075": s.get("fraction_peak_gt_075"),
            "fraction_peak_gt_090": s.get("fraction_peak_gt_090"),
            "mean_q_dynamic": s.get("mean_q_dynamic"),
            "gt": init.get("gt_evaluation", {}),
        }

    if gt is not None:
        per_outer = [_score(s["soft_part_probs_after"].argmax(dim=1).numpy(), gt) for s in states]
        best = max(range(len(per_outer)), key=lambda i: per_outer[i]["f1"])
        row.update({
            "gt": True,
            "gt_moving_fraction": float((gt > 0).mean() * 100),
            "per_outer_f1": [x["f1"] for x in per_outer],
            "final": per_outer[-1],
            "best": {**per_outer[best], "outer": best},
        })
    else:
        row["gt"] = False

    # Section 53: entropy, probability change, churn, soft reversal.
    history = debug / "render_graph_history.csv"
    if history.is_file():
        rows = list(csv.DictReader(history.open()))

        def col(key):
            out_vals = []
            for r in rows:
                try:
                    out_vals.append(float(r[key]))
                except (KeyError, TypeError, ValueError):
                    out_vals.append(float("nan"))
            return out_vals

        row["soft"] = {
            "mean_entropy_after": col("mean_entropy_after"),
            "mean_l1_probability_change": col("mean_l1_probability_change"),
            "fraction_argmax_changes": col("fraction_argmax_changes"),
            "fraction_max_prob_gt_090": col("fraction_max_prob_gt_090"),
            "fraction_soft_update_reversal": col("fraction_soft_update_reversal"),
            "mean_render_alpha": col("mean_render_alpha"),
        }

    temporal = debug / "temporal_outer_history.csv"
    if temporal.is_file():
        rows = list(csv.DictReader(temporal.open()))

        def tcol(key, cast=float):
            vals = []
            for r in rows:
                try:
                    vals.append(cast(float(r[key])))
                except (KeyError, TypeError, ValueError):
                    vals.append(float("nan"))
            return vals

        row["temporal"] = {
            "score_total": tcol("score_total"),
            "relative_global_reward": tcol("relative_global_reward"),
            "mean_abs_memory": tcol("mean_abs_memory"),
            "num_temporal_credit_updates": tcol("num_temporal_credit_updates", int),
        }
    return row


def run_ablation(scenes=SCENES, extra: dict | None = None) -> dict:
    results = {}
    for scene in scenes:
        for arm, memory in ARMS:
            out = run_name(scene, arm)
            overrides = {"outer_soft_labels": True, "enable_temporal_credit": bool(memory)}
            overrides.update(extra or {})
            print(f"\n{'='*78}\n=== {scene}  {arm}  memory={memory}  -> output/{out}\n{'='*78}")
            run(scene, out, overrides=overrides, evaluate=False)
            results[f"{scene}/{arm}"] = score_run(out)
    return results


def report(scenes=SCENES) -> dict:
    return {f"{s}/{a}": score_run(run_name(s, a)) for s in scenes for a, _ in ARMS}


def print_report(results: dict) -> None:
    print("\n=== SOFT LABELS: initialization (section 52, evaluation only) ===")
    print(f"{'run':<36s} {'F1':>7s} {'prec':>7s} {'rec':>7s} {'H':>7s} {'>.75':>7s} {'>.90':>7s}")
    for key, r in results.items():
        init = r.get("init")
        if not init:
            continue
        g = init.get("gt") or {}
        print(f"{key:<36s} {g.get('f1', float('nan')):>7.2f} {g.get('precision', float('nan')):>7.2f} "
              f"{g.get('recall', float('nan')):>7.2f} {init['mean_entropy']:>7.4f} "
              f"{init['fraction_peak_gt_075']*100:>6.1f}% {init['fraction_peak_gt_090']*100:>6.1f}%")

    print("\n=== SOFT LABELS: outer loop (section 53) ===")
    print(f"{'run':<36s} {'per-outer F1':<40s} {'best':>6s} {'final':>6s}")
    for key, r in results.items():
        if not r.get("available") or not r.get("gt"):
            print(f"{key:<36s} {r.get('reason', 'no gt')}")
            continue
        f = ' '.join('%.2f' % v for v in r["per_outer_f1"])
        print(f"{key:<36s} {f:<40s} {r['best']['f1']:>6.2f} {r['final']['f1']:>6.2f}")

    for key, r in results.items():
        s = r.get("soft")
        if not s:
            continue
        print(f"\n{key}")
        print(f"   entropy        {['%.4f' % v for v in s['mean_entropy_after']]}")
        print(f"   mean|dP|_1     {['%.4f' % v for v in s['mean_l1_probability_change']]}")
        print(f"   argmax churn   {['%.2f%%' % (v*100) for v in s['fraction_argmax_changes']]}")
        print(f"   max(P)>0.90    {['%.1f%%' % (v*100) for v in s['fraction_max_prob_gt_090']]}")
        print(f"   soft reversal  {['%.1f%%' % (v*100) for v in s['fraction_soft_update_reversal']]}")
        t = r.get("temporal")
        if t:
            print(f"   S              {['%.4f' % v for v in t['score_total']]}")
            print(f"   R_rel          {['%+.4f' % v for v in t['relative_global_reward']]}")
            print(f"   credit updates {t['num_temporal_credit_updates']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--outer", type=int, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    extra = {} if args.outer is None else {"part_outer_iters": int(args.outer)}
    results = report(args.scenes) if args.report else run_ablation(args.scenes, extra=extra)
    print_report(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
