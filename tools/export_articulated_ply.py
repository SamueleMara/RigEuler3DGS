#!/usr/bin/env python
"""Back-fill the articulated fields (part_id, lineage, joint type / axis / origin / range, per-frame q) into the final
PLY of an already finished Phase-5 run. CPU only, seconds. The 3DGS fields are copied unchanged (checked).

    python tools/export_articulated_ply.py --phase5-dir OUT/<scene>/phase5/<run>_gaussian_rgbd [--phase5-dir ...]

Writes <phase5-dir>/final/point_cloud.ply in place (atomic); a PLY that already has the fields is skipped.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plyfile import PlyData  # noqa: E402

from recovery.articulated_ply import read_articulated_ply, write_articulated_ply  # noqa: E402
from recovery.contracts import load_joint  # noqa: E402
from recovery.gaussians import GaussianTensors  # noqa: E402


def export(d: Path) -> None:
    fp = d / "final" / "point_cloud.ply"
    before = PlyData.read(str(fp))["vertex"].data
    if "part_id" in before.dtype.names:
        print(f"{fp}: already articulated, skipped")
        return
    m = json.loads((d / "manifest.json").read_text())
    st = m.get("setup") or m.get("upstream", {}).get("setup")       # the trainer nests the CLI manifest under "upstream"
    if not st:
        raise SystemExit(f"{d}/manifest.json: no setup (neither top level nor under 'upstream')")
    joint = load_joint(Path(st["phase3_dir"]), st["joint_type"], st["direction"])
    proof = json.loads((d / "input_proof.json").read_text())
    if joint.geometry_sha256 is None or not proof.get("joint_fingerprint"):
        raise SystemExit(f"{d}: no joint fingerprint to check against")
    from recovery import phase4_artifacts as AR
    if AR.joint_fingerprint(joint) != proof["joint_fingerprint"]:
        raise SystemExit(f"{d}: the Phase-3 joint on disk differs from the one Phase 5 used")
    und = np.load(d / "unresolved_originals.npz")["gaussian_uid"]
    info = write_articulated_ply(fp, GaussianTensors.load_lineage(d / "final"), joint, fp, unresolved_uids=und,
                                 extra_comments=[f"back-filled by tools/export_articulated_ply.py"])
    after = PlyData.read(str(fp))["vertex"].data
    if any(not np.array_equal(before[k], after[k]) for k in before.dtype.names):
        raise SystemExit(f"{fp}: 3DGS fields changed while back-filling")
    r = read_articulated_ply(fp)
    print(f"{fp}: {info['joint_type']} axis {np.round(r['axis'], 4).tolist()} origin {np.round(r['origin'], 4).tolist()} "
          f"q {r['q_range'][0]:.4f}..{r['q_range'][1]:.4f} {info['q_units']}; {info['n_moving']}/{info['n_gaussians']} "
          f"moving, {info['n_unresolved']} unresolved")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase5-dir", action="append", required=True)
    for d in ap.parse_args().phase5_dir:
        export(Path(d))


if __name__ == "__main__":
    main()
