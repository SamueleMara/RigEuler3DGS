"""Spec section 42/43 -- render_graph with and without temporal credit memory.

    A. render_graph without temporal memory
    B. render_graph + temporal credit memory

Everything else is held identical (section 42): initialization, inner Killing
solver, warm start, render matcher, graph parameters, seed threshold, number of
outer iterations. The ONLY difference between the arms is
``--enable_temporal_credit``.

Section 43 says to inspect the temporal mechanism separately from the improved
initialization, so the ablation is crossed with ``--init_mode``: run it from the
legacy initialized state, from the endpoint-contrastive one, or both.

    python tools/run_temporal_ablation.py --init endpoint_contrastive
    python tools/run_temporal_ablation.py --init both
    python tools/run_temporal_ablation.py --report            # score existing runs

Ground truth is read HERE ONLY, after each run. It never influences training.
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
INITS = ("legacy", "endpoint_contrastive")
ARMS = (("A_no_memory", False), ("B_temporal_memory", True))


def run_name(scene: str, init: str, arm: str) -> str:
    return f"temporal_{scene}_{init}_{arm}"


def _debug_dir(out: str) -> Path:
    return REPO / "output" / out / "dynamic_debug" / "nested_part_embedding_killing"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _load_gt(out: str, n: int):
    """Per-Gaussian gt_part from the run's own ply (evaluation only)."""
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


def _f1(pred: np.ndarray, gt: np.ndarray) -> dict:
    moving = pred > 0
    truth = gt > 0
    tp = int((moving & truth).sum())
    fp = int((moving & ~truth).sum())
    fn = int((~moving & truth).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {
        "accuracy": float((moving == truth).mean() * 100),
        "precision": float(p * 100),
        "recall": float(r * 100),
        "f1": float(2 * p * r / max(p + r, 1.0e-9) * 100),
        "moving_iou": float(tp / max(tp + fp + fn, 1) * 100),
    }


def score_run(out: str) -> dict:
    """Label quality per outer iteration plus the temporal-credit diagnostics."""
    debug = _debug_dir(out)
    files = sorted(debug.glob("render_graph_outer_*.pt"))
    if not files:
        return {"available": False, "reason": f"no render_graph_outer_*.pt in {debug}"}

    labels = [torch.load(f, map_location="cpu", weights_only=False)["final_label"].numpy()
              for f in files]
    gt = _load_gt(out, int(labels[0].shape[0]))
    row: dict = {"available": True, "num_outer": len(labels),
                 "num_gaussians": int(labels[0].shape[0])}

    if gt is not None:
        per_outer = [_f1(z, gt) for z in labels]
        best = max(range(len(per_outer)), key=lambda i: per_outer[i]["f1"])
        row.update({
            "gt": True,
            "gt_moving_fraction": float((gt > 0).mean() * 100),
            "per_outer_f1": [s["f1"] for s in per_outer],
            "final": per_outer[-1],
            "best": {**per_outer[best], "outer": best},
        })
    else:
        row["gt"] = False

    # Temporal history, when the arm had memory enabled.
    history_path = debug / "temporal_outer_history.csv"
    if history_path.is_file():
        rows = list(csv.DictReader(history_path.open()))
        def _f(r, k, default=float("nan")):
            try:
                return float(r[k])
            except (KeyError, TypeError, ValueError):
                return default
        row["temporal"] = {
            "num_rows": len(rows),
            "score_total": [_f(r, "score_total") for r in rows],
            "relative_global_reward": [_f(r, "relative_global_reward") for r in rows],
            "num_changed": [int(_f(r, "num_changed", 0)) for r in rows],
            "num_changed_components": [int(_f(r, "num_changed_components", 0)) for r in rows],
            "num_positive_reward_changes": [int(_f(r, "num_positive_reward_changes", 0)) for r in rows],
            "num_negative_reward_changes": [int(_f(r, "num_negative_reward_changes", 0)) for r in rows],
            "num_label_oscillations": [int(_f(r, "num_label_oscillations", 0)) for r in rows],
            "mean_abs_memory": [_f(r, "mean_abs_memory") for r in rows],
            "max_abs_memory": [_f(r, "max_abs_memory") for r in rows],
            "bias_num_preference_flips": [_f(r, "bias_num_preference_flips") for r in rows],
            "bias_fraction_capped": [_f(r, "bias_fraction_capped") for r in rows],
        }

    memory_path = debug / "temporal_credit_memory.pt"
    if memory_path.is_file():
        state = torch.load(memory_path, map_location="cpu", weights_only=False)
        memory = state["memory"]
        row["memory"] = {
            "mean_abs": float(memory.abs().mean()),
            "max_abs": float(memory.abs().max()),
            "fraction_nonzero": float((memory.abs() > 1.0e-12).float().mean()),
            "normalization": state.get("normalization"),
        }
    return row


# ---------------------------------------------------------------------------
# Driving
# ---------------------------------------------------------------------------


def run_ablation(scenes=SCENES, inits=INITS, arms=ARMS, extra: dict | None = None) -> dict:
    results: dict[str, dict] = {}
    for scene in scenes:
        for init in inits:
            for arm, temporal in arms:
                out = run_name(scene, init, arm)
                overrides = {"init_mode": init, "enable_temporal_credit": bool(temporal)}
                overrides.update(extra or {})
                print(f"\n{'='*78}\n=== {scene}  init={init}  arm={arm}  "
                      f"temporal={temporal}  -> output/{out}\n{'='*78}")
                run(scene, out, overrides=overrides, evaluate=False)
                results[f"{scene}/{init}/{arm}"] = score_run(out)
    return results


def report(scenes=SCENES, inits=INITS, arms=ARMS) -> dict:
    return {f"{scene}/{init}/{arm}": score_run(run_name(scene, init, arm))
            for scene in scenes for init in inits for arm, _ in arms}


def print_report(results: dict) -> None:
    print("\n=== SECTION 42 ABLATION -- temporal credit memory ON vs OFF ===")
    print(f"{'scene / init / arm':<52s} {'best F1':>8s} {'@outer':>7s} {'final F1':>9s} "
          f"{'prec':>7s} {'rec':>7s}")
    for key, row in results.items():
        if not row.get("available"):
            print(f"{key:<52s} {row.get('reason', 'missing')}")
            continue
        if not row.get("gt"):
            print(f"{key:<52s} (no per-Gaussian gt_part in the run ply)")
            continue
        print(f"{key:<52s} {row['best']['f1']:>8.2f} {row['best']['outer']:>7d} "
              f"{row['final']['f1']:>9.2f} {row['final']['precision']:>7.2f} "
              f"{row['final']['recall']:>7.2f}")

    print("\n--- temporal diagnostics (arm B only) ---")
    for key, row in results.items():
        t = row.get("temporal")
        if not t:
            continue
        print(f"\n{key}")
        print(f"   S^(m)            {['%.5f' % v for v in t['score_total']]}")
        print(f"   R_rel^(m)        {['%+.5f' % v for v in t['relative_global_reward']]}")
        print(f"   changed          {t['num_changed']}  in components {t['num_changed_components']}")
        print(f"   reward +/-       {t['num_positive_reward_changes']} / "
              f"{t['num_negative_reward_changes']}")
        print(f"   oscillations     {t['num_label_oscillations']}")
        print(f"   |M| mean / max   {['%.4f' % v for v in t['mean_abs_memory']]} / "
              f"{['%.4f' % v for v in t['max_abs_memory']]}")
        print(f"   argmin flips     {t['bias_num_preference_flips']}   "
              f"capped fraction {['%.3f' % v for v in t['bias_fraction_capped']]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--init", choices=["legacy", "endpoint_contrastive", "both"],
                    default="endpoint_contrastive",
                    help="§43: inspect the temporal mechanism from a fixed initialized state")
    ap.add_argument("--report", action="store_true",
                    help="score existing runs instead of launching new ones")
    ap.add_argument("--outer", type=int, default=None, help="override part_outer_iters")
    ap.add_argument("--out", default="", help="write the scores as json")
    args = ap.parse_args()

    inits = list(INITS) if args.init == "both" else [args.init]
    extra = {} if args.outer is None else {"part_outer_iters": int(args.outer)}
    results = (report(args.scenes, inits) if args.report
               else run_ablation(args.scenes, inits, extra=extra))
    print_report(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
