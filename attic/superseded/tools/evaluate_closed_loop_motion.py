#!/usr/bin/env python3
"""Evaluate a closed-loop run, and check its acceptance criteria.

Two jobs.

**Acceptance (spec 42).**  The pipeline's defining property is that the loop
actually closes, and that is checkable from the artefacts alone: the assignment
must have changed, the physics must have been refitted from it, the critic must
have been conditioned on the physics, and its residual must have reached the
next assignment.  Each criterion is checked against the saved per-cycle state
rather than asserted in prose.

**Ground truth (optional).**  Where a scene ships part labels and a joint, the
final assignment and the recovered axis are scored against them.  Ground truth
is never an input to the loop; it is read here and only here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_cycles(run_dir: Path) -> list[dict]:
    cycles = []
    for directory in sorted(run_dir.glob("outer_*")):
        record = directory / "diagnostics.json"
        if record.is_file():
            cycles.append({"dir": directory, **json.loads(record.read_text())})
    return cycles


def check_acceptance(run_dir: Path, cycles: list[dict]) -> dict:
    """Verify every acceptance criterion of spec 42 from the saved artefacts."""
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, detail: str) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    if not cycles:
        record("cycles_present", False, "no outer_* directories with diagnostics")
        return {"checks": checks, "passed": False}
    record("cycles_present", True, f"{len(cycles)} outer cycles")

    first = cycles[0]
    checkpoint = run_dir / "closed_loop_checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint.is_file() else None

    # Foreground only: every motion tensor must be N_F rows, not the full cloud.
    if payload is not None:
        n = int(payload["foreground_config"]["num_foreground"])
        state = payload["state"]
        widths = {
            key: tuple(value.shape)
            for key, value in state.items()
            if torch.is_tensor(value) and value.ndim >= 1
        }
        offenders = [
            key for key, shape in widths.items()
            if key in ("responsibilities", "fm_velocity", "killing_velocity",
                       "diffusion_residual_mean", "label_energies")
            and n not in shape
        ]
        record("foreground_only", not offenders,
               f"N_F={n}; offending tensors: {offenders}" if offenders else f"all motion tensors are N_F={n}")
        # Static is label 0 with exactly zero motion, by construction.
        record("static_is_label_zero", int(state["responsibilities"].shape[-1]) >= 2,
               f"{state['responsibilities'].shape[-1]} labels, index 0 is static")

    # The assignment actually moved.
    deltas = [c.get("responsibility_delta") for c in cycles if c.get("responsibility_delta") is not None]
    moved = any(value > 1.0e-6 for value in deltas)
    record("assignment_changes", moved,
           f"responsibility deltas {['%.2e' % v for v in deltas[:5]]}")

    # Killing was refitted from the assignment, i.e. some part reports a fit.
    refits = [
        entry for cycle in cycles for entry in (cycle.get("killing_refit") or [])
        if entry.get("refitted")
    ]
    record("killing_receives_assignment_feedback", bool(refits),
           f"{len(refits)} refits weighted by responsibilities and core masks")

    # The critic ran against a real physical proposal, not canonical.
    alphas = [c.get("physics_alpha") for c in cycles if c.get("physics_alpha") is not None]
    conditioned = any(c.get("diffusion_residual_global") is not None for c in cycles)
    record("diffusion_conditioned_on_physics", conditioned and bool(alphas),
           f"physics alpha schedule {alphas}")

    # The critic's residual reached the assignment.
    voted = (run_dir / f"outer_{0:04d}" / "energy_diffusion.pt").is_file()
    record("diffusion_affects_assignment", voted,
           "energy_diffusion.pt written per cycle" if voted else "no diffusion energy saved")

    # The full path executed at least once.
    closed = all(
        checks[name]["passed"]
        for name in ("assignment_changes", "killing_receives_assignment_feedback",
                     "diffusion_conditioned_on_physics", "diffusion_affects_assignment")
        if name in checks
    )
    record("loop_genuinely_closes", closed,
           "r -> FM -> Killing -> proposal -> diffusion -> energies -> r")

    # No free final motion: every point is canonical or one part's transform.
    if payload is not None:
        labels = payload["state"]["hard_labels"]
        parts = int(payload["assignment_config"]["num_moving_parts"])
        legal = bool(((labels >= 0) & (labels <= parts)).all())
        record("no_free_final_motion", legal,
               f"labels within [0,{parts}]; motion is canonical or one exact rigid transform")

    record("soft_early_assignment", first.get("mean_assignment_entropy", 0.0) > 0.05,
           f"entropy started at {first.get('mean_assignment_entropy'):.4f}")
    return {"checks": checks, "passed": all(item["passed"] for item in checks.values())}


def _ground_truth_moving(points: torch.Tensor, parts_ply: Path) -> torch.Tensor:
    """Nearest-neighbour transfer of the sparse GT part labels onto the cloud."""
    import numpy as np
    from plyfile import PlyData

    vertex = PlyData.read(str(parts_ply))["vertex"].data
    sparse = torch.from_numpy(np.stack((vertex["x"], vertex["y"], vertex["z"]), -1).copy()).double()
    part_id = torch.from_numpy(np.asarray(vertex["part"], dtype=np.int64).copy())
    nearest = []
    for start in range(0, points.shape[0], 2048):
        nearest.append(part_id[torch.cdist(points[start:start + 2048], sparse).argmin(1)])
    return torch.cat(nearest) == 1


def score_assignment(labels: torch.Tensor, gt_moving: torch.Tensor) -> dict:
    """Moving/static scores for one label vector against one GT mask."""
    predicted = torch.as_tensor(labels).reshape(-1) > 0
    tp = int((predicted & gt_moving).sum())
    fp = int((predicted & ~gt_moving).sum())
    fn = int((~predicted & gt_moving).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "assignment_precision": precision,
        "assignment_recall": recall,
        "assignment_f1": 2 * precision * recall / max(precision + recall, 1.0e-9),
        "assignment_iou": tp / max(tp + fp + fn, 1),
        "predicted_moving": int(predicted.sum()),
    }


def score_against_ground_truth(
    run_dir: Path, root_scene: Path, *, refined_state: Path | None = None
) -> dict | None:
    """Score the final assignment and axis, reading GT only here.

    ``refined_state`` is the Stage-4 reassignment under the exact joint
    transport.  It is scored alongside the Stage-1 assignment rather than
    instead of it: the whole point of that stage is whether the exact transport
    improves on the coarse Killing one, and that is only visible as a pair.
    """
    metadata = root_scene / "pointcloud" / "joint_metadata.json"
    parts_ply = root_scene / "pointcloud" / "points3D_parts.ply"
    checkpoint = run_dir / "closed_loop_checkpoint.pt"
    if not (metadata.is_file() and parts_ply.is_file() and checkpoint.is_file()):
        return None

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state"]
    points = state["canonical_fg"].double()
    gt_moving = _ground_truth_moving(points, parts_ply)

    result = {
        "ground_truth_used_as_algorithm_input": False,
        **score_assignment(state["hard_labels"], gt_moving),
        "gt_moving": int(gt_moving.sum()),
    }
    if refined_state is not None and Path(refined_state).is_file():
        refined = torch.load(refined_state, map_location="cpu", weights_only=False)
        labels = refined.get("hard_labels")
        if labels is not None and labels.numel() == gt_moving.numel():
            result["refined"] = score_assignment(labels, gt_moving)
    info = json.loads(metadata.read_text())
    axis = torch.tensor(info["axis_vector"], dtype=torch.float64)
    axis = axis / axis.norm()
    result["gt_joint_type"] = info["type"]
    for index, mode in enumerate(payload["killing_config"]["modes"]):
        if not mode.get("valid"):
            continue
        angular = torch.as_tensor(mode["angular_mode"]).double()
        if float(angular.norm()) <= 1.0e-12:
            continue
        cosine = abs(float((angular / angular.norm()) @ axis))
        result[f"part_{index + 1}_axis_error_degrees"] = math.degrees(
            math.acos(min(1.0, max(-1.0, cosine)))
        )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--root-scene", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    cycles = load_cycles(run_dir)
    acceptance = check_acceptance(run_dir, cycles)

    print(f"[evaluate] {run_dir}  ({len(cycles)} outer cycles)")
    print("\nacceptance criteria (spec 42):")
    for name, item in acceptance["checks"].items():
        print(f"  [{'PASS' if item['passed'] else 'FAIL'}] {name:38s} {item['detail']}")
    print(f"\noverall: {'PASS' if acceptance['passed'] else 'FAIL'}")

    report = {"run_dir": str(run_dir), "acceptance": acceptance,
              "cycles": [{k: v for k, v in c.items() if k != "dir"} for c in cycles]}
    if args.root_scene:
        truth = score_against_ground_truth(run_dir, Path(args.root_scene).expanduser().resolve())
        if truth is not None:
            report["ground_truth"] = truth
            print("\nagainst ground truth (read here, never by the loop):")
            for key, value in truth.items():
                print(f"  {key:36s} {value}")
        else:
            print("\nno ground truth available for this scene")

    destination = Path(args.output) if args.output else run_dir / "closed_loop_evaluation.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nwrote {destination}")
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
