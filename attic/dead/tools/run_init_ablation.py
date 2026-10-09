"""Spec section 28 -- legacy vs endpoint_contrastive initialization, two stages.

Stage A stops after initialization (``--init_only``) and reports, before the
first M-step:

    predicted moving support size
    moving support purity   against the debug GT
    moving support recall   against the debug GT
    binary F1

Stage B then runs exactly ONE inner Killing solve (``--part_outer_iters 1``) for
each arm and compares:

    endpoint transform translation
    endpoint transform rotation
    motion-vs-static render improvement
    temporal twist coherence

Ground truth is read HERE ONLY, after each run. It never influences training.

    python tools/run_init_ablation.py --stage A
    python tools/run_init_ablation.py --stage B
    python tools/run_init_ablation.py --report          # score existing runs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.run_render_graph import RENDER_GRAPH_CONFIG, run  # noqa: E402

SCENES = ("USB_100109", "storage_45135")
MODES = ("legacy", "endpoint_contrastive")


def run_name(scene: str, mode: str, stage: str) -> str:
    return f"init_ablation_{scene}_{mode}_{stage}"


def _debug_dir(out: str) -> Path:
    return REPO / "output" / out / "dynamic_debug" / "nested_part_embedding_killing"


# ---------------------------------------------------------------------------
# Stage A -- before the first M-step
# ---------------------------------------------------------------------------


def score_initialization(out: str) -> dict:
    """The four section 28 stage-A numbers, from one --init_only run."""
    path = _debug_dir(out) / "initial_labels.pt"
    if not path.is_file():
        return {"available": False, "reason": f"missing {path}"}
    state = torch.load(path, map_location="cpu", weights_only=False)

    z = torch.as_tensor(state["initial_labels"]).reshape(-1).long()
    gt_raw = state.get("gt_part_labels")
    moving = (z > 0)
    row = {
        "available": True,
        "init_mode": str(state.get("init_mode", "?")),
        "num_gaussians": int(z.numel()),
        "moving_support_size": int(moving.sum()),
        "moving_support_fraction": float(moving.float().mean() * 100.0),
    }
    if gt_raw is None:
        row.update({"gt": False, "reason": "the run saved no gt_part_labels"})
        return row

    gt = (torch.as_tensor(gt_raw).reshape(-1).long() > 0)
    tp = int((moving & gt).sum())
    fp = int((moving & ~gt).sum())
    fn = int((~moving & gt).sum())
    purity = tp / max(tp + fp, 1)          # precision of the moving support
    recall = tp / max(tp + fn, 1)
    row.update({
        "gt": True,
        "gt_moving_fraction": float(gt.float().mean() * 100.0),
        "moving_support_purity": float(purity * 100.0),
        "moving_support_recall": float(recall * 100.0),
        "binary_f1": float(2 * purity * recall / max(purity + recall, 1.0e-9) * 100.0),
        "binary_accuracy": float((moving == gt).float().mean() * 100.0),
        "moving_iou": float(tp / max(tp + fp + fn, 1) * 100.0),
    })
    return row


# ---------------------------------------------------------------------------
# Stage B -- after exactly one inner Killing solve
# ---------------------------------------------------------------------------


def _rotation_angle_deg(rotation: torch.Tensor) -> float:
    trace = float(torch.diagonal(rotation).sum())
    return math.degrees(math.acos(min(max((trace - 1.0) / 2.0, -1.0), 1.0)))


def score_first_killing_solve(out: str) -> dict:
    """The four section 28 stage-B numbers, from one part_outer_iters=1 run."""
    tracks_path = _debug_dir(out) / "nested_killing_tracks.pt"
    history_path = _debug_dir(out) / "inner_history.csv"
    if not tracks_path.is_file():
        return {"available": False, "reason": f"missing {tracks_path}"}
    tracks = torch.load(tracks_path, map_location="cpu", weights_only=False)

    translation = torch.as_tensor(tracks["part_translation"])      # [F, K+1, 3]
    rotation = torch.as_tensor(tracks["part_rotation"])            # [F, K+1, 3, 3]
    omega = torch.as_tensor(tracks["omega"])                       # [F-1, K+1, 3]
    k1 = int(translation.shape[1])

    parts = []
    for part in range(1, k1):
        w = omega[:, part]
        # Temporal twist coherence: mean cosine between consecutive omega_k.
        # 1.0 = one steady axis over the whole sequence (a real articulation),
        # ~0 = the axis reorients every frame (noise dressed as motion).
        if int(w.shape[0]) >= 2:
            a, b = w[:-1], w[1:]
            denom = (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1.0e-12)
            cosine = ((a * b).sum(dim=-1) / denom)
            coherence = float(cosine.mean())
        else:
            coherence = float("nan")
        parts.append({
            "part": part,
            "endpoint_translation_norm": float(translation[-1, part].norm()),
            "endpoint_rotation_deg": _rotation_angle_deg(rotation[-1, part]),
            "mean_omega_norm": float(omega[:, part].norm(dim=-1).mean()),
            "temporal_twist_coherence": coherence,
        })

    row = {
        "available": True,
        "parts": parts,
        # part_labels in the tracks is written AFTER the outer E-step relabels,
        # so it is NOT the support the inner solve consumed. The solve runs first
        # (nested_pipeline.py: solve_nested_killing_sequence before
        # run_render_graph_outer), on the initialization labels; read those from
        # the matching stage-A run instead.
        "num_moving_after_estep": int((torch.as_tensor(tracks["part_labels"]) > 0).sum()),
    }

    if history_path.is_file():
        rows = list(csv.DictReader(history_path.open()))
        outer0 = [r for r in rows if int(float(r.get("outer", 0))) == 0]
        by_frame: dict[int, list[dict]] = {}
        for r in outer0:
            by_frame.setdefault(int(float(r["frame"])), []).append(r)

        # Every frame's first inner step is logged at omega = u = 0. For frame 1
        # that is the true static scene, so (first - last) there is exactly the
        # motion-vs-static render improvement. For frame f > 1 the baseline is
        # the scene as frame f-1's solve left it, so the same difference is the
        # per-frame INCREMENTAL gain -- comparable across arms, but not a
        # motion-vs-static comparison. Both are reported, labelled apart.
        deltas, statics, identity_baselined = [], [], None
        for frame in sorted(by_frame):
            seq = by_frame[frame]
            first, last = float(seq[0]["loss_render"]), float(seq[-1]["loss_render"])
            if float(seq[0]["omega_norm"]) > 1.0e-9 or float(seq[0]["u_norm"]) > 1.0e-9:
                continue                      # not an identity start; skip it
            statics.append(first)
            deltas.append(first - last)
            if identity_baselined is None:    # the first frame, true static baseline
                identity_baselined = {
                    "frame": frame, "static": first, "moving": last,
                    "improvement": first - last,
                    "relative": (first - last) / max(first, 1.0e-12) * 100.0,
                }
        if deltas:
            improved = [d for d in deltas if d > 0]
            row.update({
                "motion_vs_static_frame1": identity_baselined,
                "incremental_baseline_mean": sum(statics) / len(statics),
                "incremental_improvement_mean": sum(deltas) / len(deltas),
                "incremental_frames": len(deltas),
                "incremental_improved_fraction": len(improved) / len(deltas) * 100.0,
                "incremental_improvement_relative": (
                    sum(deltas) / max(sum(statics), 1.0e-12) * 100.0),
            })
    return row


# ---------------------------------------------------------------------------
# Driving the runs
# ---------------------------------------------------------------------------


def run_stage(stage: str, scenes=SCENES, modes=MODES, extra: dict | None = None) -> dict:
    """Run every (scene, mode) arm of one stage and return the scores."""
    if stage not in ("A", "B"):
        raise ValueError("stage must be A (init only) or B (one Killing solve)")

    results: dict[str, dict] = {}
    for scene in scenes:
        for mode in modes:
            out = run_name(scene, mode, stage)
            overrides = {"init_mode": mode}
            if stage == "A":
                overrides["init_only"] = True
            else:
                overrides["part_outer_iters"] = 1
            overrides.update(extra or {})

            print(f"\n{'='*78}\n=== stage {stage}  {scene}  init_mode={mode}  -> output/{out}\n{'='*78}")
            run(scene, out, overrides=overrides, evaluate=False)
            results[f"{scene}/{mode}"] = (score_initialization(out) if stage == "A"
                                          else score_first_killing_solve(out))
    return results


def report(stage: str, scenes=SCENES, modes=MODES) -> dict:
    """Score whatever arms already exist on disk, without re-running anything."""
    results = {}
    for scene in scenes:
        for mode in modes:
            out = run_name(scene, mode, stage)
            results[f"{scene}/{mode}"] = (score_initialization(out) if stage == "A"
                                          else score_first_killing_solve(out))
    return results


def print_stage_a(results: dict) -> None:
    print("\n=== SECTION 28 STAGE A -- before the first M-step ===")
    print(f"{'scene / init_mode':<40s} {'support':>8s} {'purity':>8s} {'recall':>8s} "
          f"{'F1':>8s} {'acc':>8s} {'GT mov':>8s}")
    for key, row in results.items():
        if not row.get("available"):
            print(f"{key:<40s} {row.get('reason', 'missing')}")
            continue
        if not row.get("gt"):
            print(f"{key:<40s} {row['moving_support_size']:>8d} "
                  f"{'(no gt)':>8s}")
            continue
        print(f"{key:<40s} {row['moving_support_size']:>8d} "
              f"{row['moving_support_purity']:>8.2f} {row['moving_support_recall']:>8.2f} "
              f"{row['binary_f1']:>8.2f} {row['binary_accuracy']:>8.2f} "
              f"{row['gt_moving_fraction']:>8.2f}")


def print_stage_b(results: dict) -> None:
    print("\n=== SECTION 28 STAGE B -- after exactly one inner Killing solve ===")
    print("(the solve consumed the STAGE A support; num_moving below is post-E-step)")
    for key, row in results.items():
        if not row.get("available"):
            print(f"{key:<40s} {row.get('reason', 'missing')}")
            continue
        print(f"\n{key}   (support after the outer E-step: {row['num_moving_after_estep']})")
        for part in row["parts"]:
            print(f"   part {part['part']}: |t| {part['endpoint_translation_norm']:.6f}   "
                  f"rot {part['endpoint_rotation_deg']:7.3f} deg   "
                  f"mean|omega| {part['mean_omega_norm']:.6f}   "
                  f"twist coherence {part['temporal_twist_coherence']:+.4f}")
        mvs = row.get("motion_vs_static_frame1")
        if mvs:
            print(f"   motion vs static (frame {mvs['frame']}, true identity baseline): "
                  f"{mvs['static']:.6f} -> {mvs['moving']:.6f}  "
                  f"({mvs['relative']:+.2f}%)")
        if "incremental_improvement_mean" in row:
            print(f"   per-frame incremental gain: {row['incremental_improvement_mean']:+.6f} "
                  f"({row['incremental_improvement_relative']:+.2f}%, positive on "
                  f"{row['incremental_improved_fraction']:.1f}% of "
                  f"{row['incremental_frames']} frames)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["A", "B", "both"], default="A")
    ap.add_argument("--report", action="store_true",
                    help="score existing runs instead of launching new ones")
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--out", default="", help="write the scores as json")
    args = ap.parse_args()

    stages = ["A", "B"] if args.stage == "both" else [args.stage]
    everything = {}
    for stage in stages:
        results = (report(stage, args.scenes, args.modes) if args.report
                   else run_stage(stage, args.scenes, args.modes))
        everything[stage] = results
        (print_stage_a if stage == "A" else print_stage_b)(results)

    if args.out:
        Path(args.out).write_text(json.dumps(everything, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
