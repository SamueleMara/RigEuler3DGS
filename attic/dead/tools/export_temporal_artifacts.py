"""Package the temporal-ablation arms for Gaussian-by-Gaussian causal analysis.

Exports exactly what the per-Gaussian analysis needs -- memory flips, reward
correctness, oscillations, per-transition-type outcomes -- and nothing else, so
the archive stays small enough to move around:

    <arm>/render_graph_outer_XXX.pt      labels, raw + biased energies, seeds, margins
    <arm>/temporal_credit_outer_XXX.pt   old/new label, changed mask, rewards,
                                         memory before/after, oscillation mask,
                                         changed-component ids            (ON arms only)
    <arm>/temporal_outer_history.csv     per-outer score / reward / churn  (ON arms only)
    <arm>/temporal_credit_memory.pt      final memory + fixed normalisation (ON arms only)
    <arm>/nested_killing_tracks.pt       canonical_xyz, part transforms, omega/b
    <arm>/static_anchors.pt              trusted static anchor mask
    <arm>/initialization/                endpoint-contrastive init artifacts (ec arms)
    <arm>/gt_part.npy                    per-Gaussian ground truth, extracted from the
                                         run ply so the multi-hundred-MB ply is not shipped
    <arm>/manifest.json                  completeness record for this arm

A memory-OFF arm legitimately has NO temporal_credit_* / temporal_outer_history
files: temporal credit is disabled, so nothing writes them. Completeness for an
OFF arm therefore means 6 render_graph_outer_*.pt plus gt_part, and the manifest
records that explicitly so a downstream check does not read "no credit files" as
"run died".

    python tools/export_temporal_artifacts.py --out /tmp/temporal_export
    python tools/export_temporal_artifacts.py --out /tmp/x --zip --only storage_45135
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCENES = ("USB_100109", "storage_45135")
INITS = ("legacy", "endpoint_contrastive")
ARMS = (("A_no_memory", False), ("B_temporal_memory", True))
EXPECTED_OUTERS = 6


def _extract_gt(run_dir: Path, destination: Path) -> dict:
    """Pull gt_part out of the run ply into a small .npy (the ply is huge)."""
    from plyfile import PlyData

    plys = sorted(run_dir.glob("point_cloud/iteration_*/point_cloud.ply"),
                  key=lambda q: int(q.parent.name.rsplit("_", 1)[-1]))
    if not plys:
        return {"gt_part": False, "reason": "no point_cloud/iteration_*/point_cloud.ply"}
    vertex = PlyData.read(str(plys[-1]))["vertex"]
    if "gt_part" not in {p.name for p in vertex.properties}:
        return {"gt_part": False, "reason": "the run ply carries no gt_part field"}
    gt = np.asarray(vertex["gt_part"]).astype(np.int64)
    np.save(destination / "gt_part.npy", gt)
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    np.save(destination / "canonical_xyz.npy", xyz)
    return {"gt_part": True, "num_gaussians": int(gt.shape[0]),
            "gt_moving_fraction": float((gt > 0).mean() * 100),
            "source_ply": plys[-1].parent.name}


def export_arm(scene: str, init: str, arm: str, temporal: bool, out_root: Path) -> dict:
    run = f"temporal_{scene}_{init}_{arm}"
    run_dir = REPO / "output" / run
    debug = run_dir / "dynamic_debug" / "nested_part_embedding_killing"
    destination = out_root / run
    manifest: dict = {"run": run, "scene": scene, "init_mode": init,
                      "arm": arm, "temporal_credit": bool(temporal)}

    if not debug.is_dir():
        manifest.update({"complete": False, "reason": "run directory absent"})
        return manifest
    destination.mkdir(parents=True, exist_ok=True)

    render_graph = sorted(debug.glob("render_graph_outer_*.pt"))
    credit = sorted(debug.glob("temporal_credit_outer_*.pt"))
    for path in render_graph + credit:
        shutil.copy2(path, destination / path.name)
    for name in ("temporal_outer_history.csv", "temporal_credit_memory.pt",
                 "nested_killing_tracks.pt", "static_anchors.pt",
                 "render_graph_history.csv", "initialization_summary.json"):
        if (debug / name).is_file():
            shutil.copy2(debug / name, destination / name)
    if (debug / "initialization").is_dir():
        shutil.copytree(debug / "initialization", destination / "initialization",
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.ply"))

    manifest.update(_extract_gt(run_dir, destination))
    manifest.update({
        "num_render_graph_outers": len(render_graph),
        "num_temporal_credit_files": len(credit),
        "expected_outers": EXPECTED_OUTERS,
    })

    # A memory-OFF arm has no credit artifacts BY DESIGN, not by failure.
    expected_credit = EXPECTED_OUTERS - 1 if temporal else 0
    manifest["expected_temporal_credit_files"] = expected_credit
    manifest["complete"] = bool(
        len(render_graph) >= EXPECTED_OUTERS
        and len(credit) == expected_credit
        and manifest.get("gt_part", False)
    )
    if not manifest["complete"] and "reason" not in manifest:
        manifest["reason"] = (
            f"{len(render_graph)}/{EXPECTED_OUTERS} render-graph outers, "
            f"{len(credit)}/{expected_credit} credit files")
    if not temporal:
        manifest["note"] = ("memory OFF: temporal_credit_*.pt and "
                            "temporal_outer_history.csv are absent by design")

    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="export directory")
    ap.add_argument("--only", nargs="+", default=list(SCENES), help="restrict to these scenes")
    ap.add_argument("--zip", action="store_true", help="also produce <out>.zip")
    ap.add_argument("--require-complete", action="store_true",
                    help="exit non-zero if any arm is incomplete")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifests = []
    for scene in args.only:
        for init in INITS:
            for arm, temporal in ARMS:
                manifests.append(export_arm(scene, init, arm, temporal, out_root))

    print(f"{'arm':<54s} {'outers':>7s} {'credit':>7s} {'gt':>4s}  status")
    for m in manifests:
        status = "complete" if m.get("complete") else f"INCOMPLETE ({m.get('reason','?')})"
        print(f"{m['run']:<54s} {m.get('num_render_graph_outers', 0):>7d} "
              f"{m.get('num_temporal_credit_files', 0):>7d} "
              f"{'yes' if m.get('gt_part') else 'no':>4s}  {status}")

    (out_root / "manifest.json").write_text(json.dumps({
        "expected_outers": EXPECTED_OUTERS,
        "note": ("memory-OFF arms have no temporal_credit_*.pt or "
                 "temporal_outer_history.csv by design; judge them on "
                 "render_graph_outer_*.pt and gt_part.npy"),
        "arms": manifests,
    }, indent=2))

    complete = sum(1 for m in manifests if m.get("complete"))
    print(f"\n{complete}/{len(manifests)} arms complete -> {out_root}")

    if args.zip:
        archive = shutil.make_archive(str(out_root), "zip", root_dir=out_root)
        size = Path(archive).stat().st_size / 1e6
        print(f"wrote {archive} ({size:.1f} MB)")

    if args.require_complete and complete != len(manifests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
