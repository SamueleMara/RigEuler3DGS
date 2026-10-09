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


def rebuild_html(run: Path, red: dict, ev: dict) -> list:
    """Rebuild the 3D debug pages from artifacts. No renderer, no scene.

    The pages are derived data like the reduction is, so when the debug code
    changes there is no reason to repeat Phase 2 to see the new view.
    """
    from debug.rigid_debug import (write_rigid_phase2_3d,
                                   write_rigid_to_joint_3d, write_timelines)
    from estimator.rigid_part_se3 import RigidPart

    poses = np.load(run / "phase2_rigid" / "poses.npy")
    kab = np.load(run / "phase2_rigid" / "phase1_kabsch_poses.npy")
    vel = np.load(run / "phase2_rigid" / "velocities.npy")
    sup = json.loads((run / "support_coherence" / "support.json").read_text())
    members = torch.tensor(sup["members"], dtype=torch.long)
    cdir = run / "phase1" / "centers"
    files = sorted(cdir.glob("centers_frame_*.npy"))
    positions = [torch.from_numpy(np.load(f)).double() for f in files]
    mu0 = positions[0].clone()
    part = RigidPart(members,
                     R=torch.from_numpy(poses[:, :3, :3]).double(),
                     p=torch.from_numpy(poses[:, :3, 3]).double(),
                     v=torch.from_numpy(vel[:, :3]).double(),
                     omega=torch.from_numpy(vel[:, 3:]).double())
    kR = torch.from_numpy(kab[:, :3, :3]).double()
    kp = torch.from_numpy(kab[:, :3, 3]).double()
    summ = json.loads((run / "summary.json").read_text()) \
        if (run / "summary.json").exists() else {}
    unexp = (summ.get("support_rigidity") or {}).get(
        "unexplained_fraction_final")
    scene = run.name.replace("rigid_se3_", "")
    html = run / "html"
    out = [write_rigid_phase2_3d(html / "rigid_phase2_debug.html", mu0,
                                 members, positions, part, kR, kp, None,
                                 title=f"{scene} - rigid SE(3) Phase 2",
                                 unexplained=unexp),
           write_rigid_to_joint_3d(html / "rigid_to_joint_debug.html", mu0,
                                   members, positions, part, kR, kp, red, ev,
                                   unexplained=unexp,
                                   title=f"{scene} - rigid reconstruction "
                                         f"and joint reduction")]
    return out


def rereduce(run: Path, refine: bool = True, html: bool = True) -> dict:
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
    # Recompute the support-rigidity diagnostic through the SAME function the
    # pipeline uses, rather than re-deriving the formula here. Kabsch on the
    # stored Phase-1 centers is cheap and CPU-only, and one source of truth
    # means the two paths cannot drift apart.
    from estimator.rigid_part_runtime import kabsch_initialize
    cdir = run / "phase1" / "centers"
    pos = [torch.from_numpy(np.load(f)).double()
           for f in sorted(cdir.glob("centers_frame_*.npy"))]
    init = kabsch_initialize(members, pos[0], pos)
    summary["initialization"] = {k: v for k, v in init.items() if k != "part"}
    summary["support_rigidity"] = {
        "max_mean_kabsch_residual": init["max_mean_residual"],
        "part_scale_L": init["part_scale_L"],
        "unexplained_fraction_final": init["unexplained_fraction_final"],
        "max_unexplained_fraction": init["max_unexplained_fraction"],
        "note": init["unexplained_fraction_note"]}
    summary["reduction_rerun"] = True
    _json(summary_path, summary)
    pages = rebuild_html(run, red, ev) if html else []
    return {"run": str(run), "frames": int(R.shape[0]), "pages": pages,
            "gamma": red["one_dof"]["gamma_1dof"],
            "rad_over_L": red["revolute"]["axis_radius_over_part_scale"],
            "gt": ev}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="output/rigid_se3_<scene> directories")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--no-html", action="store_true",
                    help="skip rebuilding the 3D debug pages")
    args = ap.parse_args()
    for r in args.runs:
        run = Path(r)
        if not (run / "phase2_rigid" / "poses.npy").exists():
            print(f"  {run}: no phase2_rigid/poses.npy -- skipped")
            continue
        out = rereduce(run, refine=not args.no_refine, html=not args.no_html)
        g = out["gt"]
        tail = (f"   P {g['prismatic']['axis_angle_deg']:6.2f} deg"
                f"   R {g['revolute']['axis_angle_deg']:6.2f} deg"
                f"   line {g['revolute']['axis_line_error']['distance']:.4f}"
                if g.get("available") else "   (no GT)")
        print(f"  {run.name:<34} gamma {out['gamma']:.4f}"
              f"  rad/L {out['rad_over_L']:7.1f}{tail}")
        for pg in out["pages"]:
            print(f"      {pg}  ({pg.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
