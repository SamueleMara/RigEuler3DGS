"""Audit: ground truth cannot reach the estimator, and the two sources share
one gauge.

Sections 1.3 to 1.6 of
CODEX_NEXT_AFTER_F80E3E1_COMPLETE_ARTICULATED_PIPELINE.md.

The real loader prints

    [GT parts] attached sparse part labels from .../points3D_parts.ply

on every scene load, including the ones this pipeline makes. Loading a field
and using it are different things, and the directive is right that the
difference has to be demonstrated rather than presumed. Four questions, each
answered by measurement:

  1.3  can the part labels reach the state, the factors or the support?
  1.4  are the two endpoint models in one world gauge at one metric scale?
  1.5  what is the physical inter-frame duration, really?
  1.6  which likelihood is the real CLI running?

The real-data parts need the Washer tree on the external disk and are skipped
with an explicit NOT_RUN line when it is absent; the static parts always run.

    python tests/test_unified_gt_isolation_and_gauge.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

CHECKS = 0
FAILURES: list[str] = []

SCENE_ROOT = Path("/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/"
                  "ours/washer_103776")
DATASET = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/"
               "output/preparation_8cams/partnet_subset_fixed_render_linear/"
               "washer_103776")

# Every name by which ground truth is known in this repository. The scan is
# by PARSED IDENTIFIER, not by substring: a comment that mentions GT must not
# be able to fail the test, and a string that happens to contain the word
# must not be able to pass it.
GT_NAMES = {
    "_gt_part_labels", "gt_part_labels", "part_indices", "part_labels",
    "load_optimized_gt", "supernode_gt", "load_static_dynamic_gt",
    "optimized_gt", "motion_coherence_eval", "gt_edge_coherence",
    "classification_metrics", "transport_sparse_gt_parts",
    "has_sparse_gt_parts",
}
# The modules the unified estimator actually runs. If GT is reachable at all,
# it is reachable from one of these.
PRODUCTION_MODULES = [
    "bidirectional_iekf/ieks.py",
    "bidirectional_iekf/observations.py",
    "bidirectional_iekf/full_window_free.py",
    "bidirectional_iekf/process_factors.py",
    "bidirectional_iekf/provenance.py",
    "bidirectional_iekf/free_dynamics.py",
    "bidirectional_iekf/unified_coherence.py",
    "bidirectional_iekf/representation_audit.py",
    "estimator/motion_coherence_support.py",
    "estimator/coherence_extraction.py",
    "tools/run_unified_phase1_single_source.py",
    "tools/run_unified_representation_gate.py",
]


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")
    else:
        print(f"  ok    {label}  {detail}")


def identifiers(path: Path) -> set:
    """Every name the module actually REFERS to, from its parsed tree."""
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.update(node.module.split("."))
            for alias in node.names:
                names.update(alias.name.split("."))
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
    return names


def main() -> int:
    print("[1.3a] no production module even NAMES ground truth")
    offenders = {}
    for rel in PRODUCTION_MODULES:
        path = REPO / rel
        if not path.is_file():
            offenders[rel] = {"MISSING_FILE"}
            continue
        hit = identifiers(path) & GT_NAMES
        if hit:
            offenders[rel] = hit
    check("the unified estimator refers to no GT symbol",
          not offenders,
          f"{len(PRODUCTION_MODULES)} modules scanned by parsed identifier; "
          + ("none refers to any of "
             f"{sorted(GT_NAMES)[:4]}... " if not offenders
             else f"OFFENDERS {offenders}"))

    # The getattr escape hatch: a module could reach GT through a string.
    string_hits = {}
    for rel in PRODUCTION_MODULES:
        path = REPO / rel
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text())
        found = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("getattr", "setattr", "hasattr")):
                for arg in node.args[1:2]:
                    if isinstance(arg, ast.Constant) and arg.value in GT_NAMES:
                        found.add(arg.value)
        if found:
            string_hits[rel] = found
    check("and reaches none of them through getattr/setattr either",
          not string_hits, f"{string_hits or 'no dynamic attribute access to GT'}")

    print("\n[1.3b] the state the estimator builds cannot carry a label")
    from estimator.state import GaussianMotionState
    fields = set(GaussianMotionState.__dataclass_fields__)
    check("GaussianMotionState has no label field",
          not (fields & GT_NAMES) and fields == {
              "mu", "R", "v", "omega", "colors", "opacity", "scales"},
          f"fields {sorted(fields)}")
    import inspect
    from tools.run_iekf_scene import state_from_gaussians
    source = inspect.getsource(state_from_gaussians)
    read = {node.attr for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Attribute)}
    check("state_from_gaussians reads only appearance and geometry",
          not (read & GT_NAMES),
          "it touches " + ", ".join(sorted(
              a for a in read if a.startswith("get_"))))

    print("\n[1.3c] the sealed factor inventory is a function of frames and "
          "cameras alone")
    from bidirectional_iekf.observations import FactorKey
    key_fields = set(FactorKey.__dataclass_fields__)
    check("a factor is identified by (original_frame, camera) and nothing else",
          key_fields == {"original_frame", "camera"}, f"{sorted(key_fields)}")

    # ---------------------------------------------------------------
    if not (SCENE_ROOT.is_dir() and DATASET.is_dir()):
        print(f"\n[1.3d-1.5] NOT_RUN: {SCENE_ROOT} or {DATASET} not mounted")
    else:
        print("\n[1.3d] on the real scene, the labels are not even populated")
        import torch
        from tools.run_iekf_scene import load_scene
        _d, _p, gaussians, _s = load_scene(
            str(SCENE_ROOT / "static_bidi" / "final"), str(DATASET))
        labels = getattr(gaussians, "_gt_part_labels", None)
        populated = labels is not None and torch.is_tensor(labels) and labels.numel()
        check("a model loaded from its trained PLY carries no part labels",
              not populated,
              f"_gt_part_labels.numel() = "
              f"{0 if labels is None else int(labels.numel())}; the labels "
              "enter at create_from_pcd, and loading iteration_30000 does "
              "not go through it. The [GT parts] line is printed while "
              "READING the scene's sparse cloud, before that choice")

        print("\n[1.3e] and if they WERE populated, the state would not change")
        baseline = state_from_gaussians(gaussians)
        gaussians._gt_part_labels = torch.arange(
            int(gaussians.get_xyz.shape[0]), device="cuda", dtype=torch.long)
        poisoned = state_from_gaussians(gaussians)
        identical = all(
            torch.equal(getattr(baseline, f), getattr(poisoned, f))
            for f in ("mu", "R", "v", "omega", "colors", "opacity", "scales"))
        check("injecting a distinct label per Gaussian changes nothing",
              identical,
              "all seven state tensors bitwise equal; this is the strong form "
              "of the claim -- not that the field is empty, but that it is "
              "unread")
        gaussians._gt_part_labels = torch.empty(0, dtype=torch.long, device="cuda")

        print("\n[1.4] the two endpoint models share one world gauge")
        cams = {}
        for endpoint in ("initial", "final"):
            path = SCENE_ROOT / "static_bidi" / endpoint / "cameras.json"
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            cams[endpoint] = json.loads(path.read_text())
        check("both models were trained against the same 8 cameras",
              len(cams["initial"]) == len(cams["final"]) == 8,
              f"{len(cams['initial'])} and {len(cams['final'])}")
        pos = {e: np.array([c["position"] for c in cams[e]]) for e in cams}
        rot = {e: np.array([c["rotation"] for c in cams[e]]) for e in cams}
        dpos = float(np.abs(pos["initial"] - pos["final"]).max())
        drot = float(np.abs(rot["initial"] - rot["final"]).max())
        check("their extrinsics are identical, so there is no world-to-world map",
              dpos == 0.0 and drot == 0.0,
              f"max |dposition| {dpos:.3e}, max |drotation| {drot:.3e}. The "
              "two COLMAP reconstructions (colmap and colmap_final) were "
              "written with the same rig, so no alignment is derived and "
              "none is guessed")
        fx = {e: cams[e][0]["fx"] for e in cams}
        check("and so are their intrinsics",
              fx["initial"] == fx["final"]
              and cams["initial"][0]["width"] == cams["final"][0]["width"],
              f"fx {fx['initial']:.6f}, "
              f"{cams['initial'][0]['width']}x{cams['initial'][0]['height']}")
        names = {e: [c["img_name"] for c in cams[e]] for e in cams}
        check("the two differ only in which FRAME they were registered at",
              all("frame_000000" in n for n in names["initial"])
              and all("frame_000059" in n for n in names["final"]),
              f"{names['initial'][0]} against {names['final'][0]} -- which is "
              "the endpoint provenance, not a gauge difference")

        from plyfile import PlyData
        box = {}
        for endpoint in ("initial", "final"):
            ply = (SCENE_ROOT / "static_bidi" / endpoint / "point_cloud"
                   / "iteration_30000" / "point_cloud.ply")
            v = PlyData.read(str(ply))["vertex"]
            xyz = np.stack([v["x"], v["y"], v["z"]], 1)
            box[endpoint] = (xyz.min(0), xyz.max(0))
            check(f"the {endpoint} PLY carries no 'part' property",
                  "part" not in {p.name for p in v.properties},
                  "so nothing downstream could read a label out of it even "
                  "by accident")
        overlap = np.minimum(box["initial"][1], box["final"][1]) - np.maximum(
            box["initial"][0], box["final"][0])
        check("both populations occupy the same metric box",
              bool(np.all(overlap > 0)) and float(np.abs(
                  box["initial"][0] - box["final"][0]).max()) < 0.05,
              f"initial x in [{box['initial'][0][0]:.3f}, {box['initial'][1][0]:.3f}], "
              f"final x in [{box['final'][0][0]:.3f}, {box['final'][1][0]:.3f}]; "
              f"the final model reaches z = {box['final'][1][2]:.3f} against "
              f"{box['initial'][1][2]:.3f}, which is the door being open, not "
              "a scale difference")

        print("\n[1.5] the physical inter-frame duration is NOT a measured fact")
        cfg = (DATASET / "metadata" / "job_config_resolved.yaml").read_text()
        values = (DATASET / "metadata" / "frame_values.csv").read_text().strip()
        rows = [r.split(",") for r in values.splitlines()[1:]]
        q = np.array([float(r[1]) for r in rows])
        dq = np.diff(q)
        check("the dataset records fps and duration as null",
              "fps: null" in cfg and "duration_sec: null" in cfg,
              "so --dt is a declared convention, not a measurement, and this "
              "is written into every manifest rather than implied")
        check("what IS uniform is the joint coordinate, not a clock",
              float(dq.std()) / float(dq.mean()) < 1e-9,
              f"dq = {dq.mean():.10f} rad, relative spread "
              f"{float(dq.std()) / float(dq.mean()):.2e} over "
              f"{len(q)} frames; q runs 0 -> {q[-1]:.6f} rad")
        check("frames are evenly spaced, so a constant dt is self-consistent",
              len(q) == 60,
              f"{len(q)} frames; any constant dt only rescales v and omega "
              "together with Q, and no reported quantity depends on its value")
        info = json.loads((DATASET / "metadata" / "joint_info.json").read_text())
        check("and the dataset's own joint type is recorded for LATER "
              "evaluation only",
              info["selected_joint"]["type"] == "revolute",
              f"{info['selected_joint']['type']}, limits "
              f"{info['selected_joint']['limits'][0]} -- read here, in a test, "
              "and nowhere in the estimator")

    print("\n[1.6] which likelihood the real CLI runs, stated not implied")
    from estimator.sparse_iekf import IEKFConfig
    cfg = IEKFConfig()
    check("production IEKFConfig advertises robust_measurement=True",
          cfg.robust_measurement is True,
          f"student_t_nu = {cfg.student_t_nu}")
    cli = ast.parse((REPO / "tools/run_unified_phase1_single_source.py").read_text())
    robust_false = [
        node for node in ast.walk(cli)
        if isinstance(node, ast.keyword) and node.arg == "robust"
        and isinstance(node.value, ast.Constant) and node.value.value is False]
    check("and the unified CLI sets robust=False explicitly",
          len(robust_false) == 1,
          "a Gaussian RGB-D likelihood: mathematically legitimate, but a "
          "DIFFERENT likelihood from production's Student-t IRLS")
    from bidirectional_iekf.ieks import UnifiedFreeIEKS
    import inspect
    src = inspect.getsource(UnifiedFreeIEKS.__init__)
    check("robust=True is refused rather than approximated",
          "NotImplementedError" in src and "IRLS" in src,
          "0.5*sum(w*r^2) recomputed at each state is not the Student-t "
          "negative log likelihood whose gradient those weights represent, "
          "so accepting a line-search step on it would not be a MAP")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
