#!/usr/bin/env python
"""Post-hoc, non-mutating simulation-asset export of an already COMPLETED Phase-5 run.

    python tools/export_phase5_sim_asset.py --phase5-dir <P5> [--asset-physics-spec SPEC.yaml]
        [--asset-registration REG.yaml] [--asset-export-mode auto|required|visual_only]
        [--asset-usd-python PY] [--asset-mujoco-python PY] [--asset-sapien-python PY] [--asset-revision NAME]

Reads <P5>/complete.json, final/point_cloud.ply, final/joint/, final/joint_unchanged_proof.json, manifest.json; writes
ONLY <P5>/assets/<scene>/ (or --out-root). Never rewrites complete.json, final/point_cloud.ply, the joint files or any
trained tensor. Identical inputs -> the existing asset is verified and reused; different inputs -> refused unless
--asset-revision gives a new revision. Exit status: 0 exported / reused, 2 input or physics error, 3 validation
failure or required-level not reached.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    from recovery.sim_asset_export import ExportError, export_phase5_sim_asset
    from recovery.sim_asset_export.ply_split import sha256_file
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase5-dir", required=True)
    ap.add_argument("--asset-physics-spec", default=None)
    ap.add_argument("--asset-registration", default=None)
    ap.add_argument("--asset-export-mode", default="auto", choices=("auto", "required", "visual_only"))
    ap.add_argument("--asset-usd-python", default=None)
    ap.add_argument("--asset-mujoco-python", default=None)
    ap.add_argument("--asset-sapien-python", default=None)
    ap.add_argument("--asset-scene", default=None)
    ap.add_argument("--asset-revision", default=None)
    ap.add_argument("--asset-verify", default="true", choices=("true", "false"))
    ap.add_argument("--out-root", default=None, help="default <phase5-dir>/assets")
    a = ap.parse_args(argv)
    p5 = Path(a.phase5_dir)
    guard = [p5 / "complete.json", p5 / "final" / "point_cloud.ply", p5 / "final" / "joint_unchanged_proof.json",
             *sorted((p5 / "final" / "joint").glob("*"))]
    before = {str(p): sha256_file(p) for p in guard if p.is_file()}
    try:
        r = export_phase5_sim_asset(p5, scene=a.asset_scene, physics_spec=a.asset_physics_spec,
                                    registration=a.asset_registration, mode=a.asset_export_mode,
                                    out_root=a.out_root, usd_python=a.asset_usd_python,
                                    mujoco_python=a.asset_mujoco_python, sapien_python=a.asset_sapien_python,
                                    verify=a.asset_verify == "true", revision=a.asset_revision)
        rc = 0
    except ExportError as e:
        print(f"asset export failed: {e}", file=sys.stderr)
        msg = str(e)
        rc = 3 if ("validation FAILED" in msg or "mode required" in msg) else 2
        r = None
    after = {str(p): sha256_file(p) for p in guard if p.is_file()}
    if before != after:
        print("INTERNAL ERROR: a Phase-5 file changed during the export", file=sys.stderr)
        return 4
    if r is not None:
        print(json.dumps({"asset_export_status": r.status, "asset_root": str(r.asset_root), "reused": r.reused,
                          "visual_complete": r.visual_complete, "physics_complete": r.physics_complete,
                          "usd_complete": r.usd_complete, "missing_inputs": r.missing_inputs,
                          "NOT_FOR_RESULTS": r.NOT_FOR_RESULTS, "warnings": r.warnings,
                          "manifest_sha256": r.manifest_sha256}, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
