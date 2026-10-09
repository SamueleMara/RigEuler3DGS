"""Re-run joint reduction on a finished rigid Phase-2 run. No GPU, no scene.

CODEX_IMPLEMENT_RIGID_SE3_PHASE2_AND_POSTHOC_JOINT_REDUCTION.md section 24:
reduction reads the recovered rigid trajectory and its uncertainty, and
nothing else. That is also what makes this possible -- everything it needs is
already on disk:

    phase2_rigid/poses.npy              T_t
    phase2_rigid/pose_covariances.npy   P_{T,t}
    phase1/centers/centers_frame_0000.npy   the canonical geometry
    support_coherence/support.json      the frozen moving support

Rigid Phase 2 costs minutes per scene; the reduction costs seconds. When the
reduction changes -- a new diagnostic, a different weighting, a fix -- this
re-derives it for every finished run instead of repeating the renderer.

    python tools/rereduce_rigid_trajectory.py output/rigid_se3_*

Section 34 still holds: the poses are READ. This never writes them back.
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


def rereduce(run: Path, refine: bool = True) -> dict:
    from joint_reduction import reduce_trajectory
    from joint_reduction.gt_evaluation import evaluate, load_joint_gt

    poses = np.load(run / "phase2_rigid" / "poses.npy")
    P_tau = np.load(run / "phase2_rigid" / "pose_covariances.npy")
    R = torch.from_numpy(poses[:, :3, :3]).double()
    p = torch.from_numpy(poses[:, :3, 3]).double()
    mu0 = torch.from_numpy(
        np.load(run / "phase1" / "centers" / "centers_frame_0000.npy")).double()
    sup = json.loads((run / "support_coherence" / "support.json").read_text())
    members = torch.tensor(sup["members"], dtype=torch.long)

    red = reduce_trajectory(R, p, mu0, members,
                            P_tau=torch.from_numpy(P_tau).double(),
                            refine=refine)

    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() \
        else {"scene": run.name}
    source = summary.get("source")
    gt = load_joint_gt(source) if source else {"available": False,
                                               "reason": "no source recorded"}
    ev = evaluate(red, gt, frame_ids=list(range(R.shape[0])))

    # reuse the pipeline's own writers so the tree stays identical
    from tools.run_rigid_se3_pipeline import _geometry_only, _json, _write_reduction
    _write_reduction(run / "joint_reduction", red, R.shape[0])
    _json(run / "joint_reduction" / "gt_evaluation.json", ev)
    summary["one_dof"] = red["one_dof"]
    summary["prismatic"] = _geometry_only(red["prismatic"])
    summary["revolute"] = _geometry_only(red["revolute"])
    summary["gt_evaluation"] = ev
    summary["reduction_rerun"] = True
    _json(summary_path, summary)
    return {"run": str(run), "frames": int(R.shape[0]),
            "gamma": red["one_dof"]["gamma_1dof"],
            "rad_over_L": red["revolute"]["axis_radius_over_part_scale"],
            "gt": ev}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="output/rigid_se3_<scene> directories")
    ap.add_argument("--no-refine", action="store_true")
    args = ap.parse_args()
    for r in args.runs:
        run = Path(r)
        if not (run / "phase2_rigid" / "poses.npy").exists():
            print(f"  {run}: no phase2_rigid/poses.npy -- skipped")
            continue
        out = rereduce(run, refine=not args.no_refine)
        g = out["gt"]
        tail = (f"   P {g['prismatic']['axis_angle_deg']:6.2f} deg"
                f"   R {g['revolute']['axis_angle_deg']:6.2f} deg"
                f"   line {g['revolute']['axis_line_error']['distance']:.4f}"
                if g.get("available") else "   (no GT)")
        print(f"  {run.name:<34} gamma {out['gamma']:.4f}"
              f"  rad/L {out['rad_over_L']:7.1f}{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
