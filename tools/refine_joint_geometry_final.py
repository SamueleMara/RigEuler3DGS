#!/usr/bin/env python
"""OPTIONAL final joint-geometry refinement after Phase 5 -- DISABLED BY DEFAULT, NOT PART OF ANY PIPELINE.

Without --enable this is a dry run: it validates the inputs and reports what WOULD be optimised, and writes
nothing. With --enable it requires <phase5-dir>/final/joint_unchanged_proof.json (identical = true, hashes
equal to the Phase-3 files), freezes every Gaussian tensor / label / camera / the topology, optimises only the
minimal joint geometry (recovery/joint_refine_final.py) over full-sequence sweeps, and writes a SEPARATE
phase5_joint_final/ next to phase5/. It never overwrites Phase 3, 4 or 5.

    python tools/refine_joint_geometry_final.py --phase5-dir RES/SCENE/phase5 \\
        --phase3-dir RES/SCENE/bidirectional/DIR_joint --joint-type revolute --steps 10 --full-sequence-sweep --enable
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase5-dir", required=True)
    ap.add_argument("--phase3-dir", required=True)
    ap.add_argument("--joint-type", required=True, choices=("prismatic", "revolute"))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--full-sequence-sweep", action="store_true")
    ap.add_argument("--allow-q-correction", action="store_true")
    ap.add_argument("--enable", action="store_true", help="actually optimise (otherwise: dry run, nothing written)")
    ap.add_argument("--output", default=None, help="default: <phase5-dir>/../phase5_joint_final")
    a = ap.parse_args(argv)
    from recovery.contracts import load_joint, sha256
    p5, p3 = Path(a.phase5_dir), Path(a.phase3_dir)
    direction = p3.name.split("_")[0]
    joint = load_joint(p3, a.joint_type, direction)
    from recovery.joint_refine_final import MinimalJoint
    mj = MinimalJoint(joint, a.allow_q_correction)
    proof_p = p5 / "final" / "joint_unchanged_proof.json"
    report = {"status": "DRY_RUN" if not a.enable else "REQUESTED", "joint_type": a.joint_type,
              "dof": mj.dof(), "would_optimise": ["axis tangent (2)"] + (["axis point, perpendicular (2)"]
                                                                         if a.joint_type == "revolute" else [])
              + ([f"dq ({len(joint.q) - 1}, anchor fixed)"] if a.allow_q_correction else []),
              "frozen": ["all Gaussian tensors", "material labels", "topology", "cameras", "exposure"],
              "proof_present": proof_p.is_file()}
    if not a.enable:
        report["note"] = "NOT RUN: --enable not given; nothing written"
        print(json.dumps(report, indent=1))
        return 0
    if not proof_p.is_file():
        print(f"refusing: {proof_p} missing (Phase 5 must complete with an unchanged joint first)", file=sys.stderr)
        return 2
    proof = json.loads(proof_p.read_text())
    files = proof.get("fingerprint_after", {}).get("files", {})
    if not proof.get("identical") or files.get("geometry_file") != joint.geometry_sha256 or \
            files.get("trajectory_file") != joint.trajectory_sha256:
        print("refusing: the Phase-5 proof does not match the Phase-3 joint files", file=sys.stderr)
        return 2
    print("refusing: the optimisation loop is intentionally not wired in this revision (spec section 8: "
          "create, test the parameterisation, DO NOT RUN). Report:", json.dumps(report), file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
