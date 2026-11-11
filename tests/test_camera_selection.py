#!/usr/bin/env python
"""--cameras all | N | cam_001,cam_002 through the pipeline (CPU; driver with stub stages).

  [S1] parsing: all / auto / count / names, spelling and order, refusals
  [S2] static training on a subset drops the other cameras' views AND seed points,
       and recomputes the scene extent; no selection is unchanged
  [S3] endpoint provenance: a name list needs the exact camera set, a count needs coverage
  [S4] the stage signature and the mask preflight see only the selected cameras
  [S5] driver: names reach every runner, an endpoint of other cameras is refused, an
       unknown camera skips the scene, the default runs every camera

    python tests/test_camera_selection.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from utils.camera_selection import endpoint_cameras, endpoint_problem, resolve, select_cameras  # noqa: E402

CHECKS, FAILURES = 0, []
CAMS = ["cam_000", "cam_001", "cam_002"]


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(label)


def refused(spec):
    try:
        resolve(spec, CAMS)
    except ValueError:
        return True
    return False


def test_parsing():
    print("\n[S1] parsing")
    check("[S1] default, all and auto select every camera", resolve(None, CAMS) == resolve("all", CAMS)
          == resolve("auto", CAMS) == resolve("", CAMS) == CAMS)
    check("[S1] a count selects the first N, clamped", resolve("2", CAMS) == CAMS[:2] and resolve(8, CAMS) == CAMS)
    want = ["cam_001", "cam_002"]
    check("[S1] spelling does not matter", all(resolve(s, CAMS) == want for s in
                                               ("cam_001,cam_002", "cam001,cam002", "cam_1,cam_2", "cam_001 cam_002")))
    check("[S1] the order given does not matter", resolve("cam_002,cam_001", CAMS) == want)
    check("[S1] a repeated camera counts once", resolve("cam_001,cam_001", CAMS) == ["cam_001"])
    check("[S1] unknown names, bare indices in a list and zero are refused",
          refused("cam_009") and refused("1,2") and refused("cam_001,") and refused("0"))
    objs = [types.SimpleNamespace(image_name=f"{c}/frame_000000.png") for c in CAMS]
    check("[S1] Camera objects keep their order", [o.image_name[:7] for o in select_cameras(objs, "cam_002,cam_000")]
          == ["cam_000", "cam_002"])


def colmap_fixture(d: Path):
    """Three cameras on a line; each seed point seen by exactly one, cam_000's far away."""
    sp = d / "sparse" / "0"
    sp.mkdir(parents=True)
    (sp / "cameras.txt").write_text("1 PINHOLE 8 8 10 10 4 4\n")
    lines = []
    for i, c in enumerate(CAMS):
        lines += [f"{i + 1} 1 0 0 0 {float(i)} 0 0 1 {c}/frame_000000.png", ""]
        im = d.parent / "rgb" / c
        im.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(im / "frame_000000.png")
    (sp / "images.txt").write_text("\n".join(lines) + "\n")
    pts = [(1, 100.0, 1), (2, 1.0, 2), (3, 2.0, 2), (4, 3.0, 3)]
    (sp / "points3D.txt").write_text("".join(f"{i} {x} 0 1 10 20 30 0.0 {im} 0\n" for i, x, im in pts))


def test_static_subset():
    print("\n[S2] static training on a subset")
    from scene.dataset_readers import readColmapSceneInfo
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "scene" / "colmap"
        colmap_fixture(d)
        full = readColmapSceneInfo(str(d), "../rgb", "", False, False, load_gt_parts=False)
        sub = readColmapSceneInfo(str(d), "../rgb", "", False, False, load_gt_parts=False, cameras="cam_001,cam_002")
        check("[S2] no selection keeps every view and seed point",
              len(full.train_cameras) == 3 and len(full.point_cloud.points) == 4)
        check("[S2] the views are the selected ones",
              [c.image_name.split("/")[0] for c in sub.train_cameras] == ["cam_001", "cam_002"])
        check("[S2] the excluded camera's seed points are gone",
              sorted(sub.point_cloud.points[:, 0].tolist()) == [1.0, 2.0, 3.0], str(sub.point_cloud.points[:, 0]))
        check("[S2] colours stay aligned with the points", len(sub.point_cloud.colors) == 3)
        check("[S2] the extent comes from the selected views",
              sub.nerf_normalization["radius"] < full.nerf_normalization["radius"],
              f'{sub.nerf_normalization["radius"]:.3f} < {full.nerf_normalization["radius"]:.3f}')
        ply = d / "sparse" / "0" / "points3D.ply"
        from plyfile import PlyData, PlyElement
        v = PlyData.read(ply)["vertex"].data.copy()
        v["x"][0] += 1.0
        PlyData([PlyElement.describe(v, "vertex")]).write(ply)
        try:
            readColmapSceneInfo(str(d), "../rgb", "", False, False, load_gt_parts=False, cameras="cam_001")
            check("[S2] a cloud that does not match the tracks is refused", False)
        except ValueError as ex:
            check("[S2] a cloud that does not match the tracks is refused", True, str(ex)[-60:])


def test_endpoint():
    print("\n[S3] endpoint provenance")
    with tempfile.TemporaryDirectory() as td:
        m = Path(td)
        (m / "cfg_args").write_text("Namespace(sh_degree=3, depths='depth_npy')")
        check("[S3] an older model counts as every camera", endpoint_cameras(m) is None)
        check("[S3] a name list refuses it", endpoint_problem(m, "cam_001,cam_002", CAMS) is not None)
        check("[S3] all and a count accept it", endpoint_problem(m, "all", CAMS) is None
              and endpoint_problem(m, "2", CAMS) is None)
        check("[S3] the same list naming every camera accepts it", endpoint_problem(m, "cam_000,cam_001,cam_002", CAMS) is None)
        (m / "cfg_args").write_text("Namespace(cameras='cam_001,cam_002', sh_degree=3)")
        check("[S3] cfg_args records the training cameras", endpoint_cameras(m) == ["cam_001", "cam_002"])
        check("[S3] the matching list accepts it", endpoint_problem(m, "cam_002,cam_001", CAMS) is None)
        check("[S3] all refuses it, the model never saw cam_000", endpoint_problem(m, "all", CAMS) is not None)
        check("[S3] another list refuses it", endpoint_problem(m, "cam_001", CAMS) is not None)
        (m / "bootstrap_manifest.json").write_text('{"camera_names": ["cam_000"]}')
        check("[S3] a bootstrap manifest is read", endpoint_cameras(m) == ["cam_000"])


def scene_fixture(root: Path, out: Path, name: str, static_cfg="Namespace()"):
    d = root / name
    for k in ("colmap/sparse/0", "colmap_final/sparse/0"):
        (d / k).mkdir(parents=True)
    for c in CAMS:
        for k in ("rgb", "mask", "depth_npy"):
            (d / k / c).mkdir(parents=True)
        for f in range(2):
            Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(d / "rgb" / c / f"frame_{f:06d}.png")
            Image.fromarray(np.zeros((4, 4), np.uint8)).save(d / "mask" / c / f"frame_{f:06d}.png")
            np.save(d / "depth_npy" / c / f"frame_{f:06d}.npy", np.ones((4, 4), np.float32))
    for e in ("initial", "final"):
        m = out / name / "static_bidi" / e
        (m / "point_cloud" / "iteration_30000").mkdir(parents=True)
        (m / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_bytes(b"ply\n" + b"0" * 120000)
        (m / "cfg_args").write_text(static_cfg)
    return d


def test_signature_and_masks():
    print("\n[S4] signature and mask preflight")
    import check_masks
    import stage_signature as ss
    with tempfile.TemporaryDirectory() as td:
        d = scene_fixture(Path(td) / "in", Path(td) / "out", "s")
        problems, info = check_masks.check(d, d / "mask", "cam_001,cam_002", None)
        check("[S4] masks of the selected cameras are checked", not problems and info["cameras"] == 2, str(info))
        shutil.rmtree(d / "mask" / "cam_000")
        problems, _ = check_masks.check(d, d / "mask", "cam_001,cam_002", None)
        check("[S4] a missing mask of an excluded camera does not matter", not problems)
        problems, _ = check_masks.check(d, d / "mask", "all", None)
        check("[S4] it does when that camera is selected", bool(problems))
        check("[S4] an unknown camera is a problem", bool(check_masks.check(d, d / "mask", "cam_007", None)[0]))
        a = argparse.Namespace(source=str(d), model=str(Path(td) / "m"), iterations="30000", observation_mode="original",
                               mask_source=None, object_mask_constraint="off", upstream=None, stage="phase1",
                               direction="forward", scene="s", rgb_background_distribution=None,
                               rgb_background_constant=None, mask_options=None, frames="2", config=[], cameras="cam_002,cam_001")
        sub = ss.compute(a)
        check("[S4] the signature names the selected cameras", sub["source"]["cameras"] == ["cam_001", "cam_002"])
        a.cameras = "2"
        first2 = ss.compute(a)
        check("[S4] a different camera set is a different signature", first2["source"]["cameras"] == CAMS[:2]
              and ss.sig_hash(first2) != ss.sig_hash(sub))


STUB = r'''#!REALPY
import json, os, sys
from pathlib import Path
REAL = "REALPY"
stages = {"tools/run_motion_coherence_support.py": "phase1", "tools/run_rigid_se3_pipeline.py": "phase2",
          "tools/run_joint_phase3.py": "phase3"}
if len(sys.argv) < 2 or sys.argv[1] not in stages:
    os.execv(REAL, [REAL] + sys.argv[1:])
a = sys.argv[2:]
out = Path(a[a.index("--out") + 1])
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write(f"{stages[sys.argv[1]]} {a[a.index('--cameras') + 1]}\n")
sys.exit(3)
'''


def test_driver():
    print("\n[S5] driver (stub stages)")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        stub = td / "stubpy"
        stub.write_text(STUB.replace("REALPY", sys.executable))
        stub.chmod(0o755)

        def run(case, cfg, *extra):
            root, out = td / case / "in", td / case / "out"
            scene_fixture(root, out, "s", static_cfg=cfg)
            log = td / f"{case}.log"
            log.write_text("")
            env = dict(os.environ, PY=str(stub), STUB_LOG=str(log), STALL_MIN="0", STALL_POLL="1")
            r = subprocess.run(["bash", "run_bidirectional_all.sh", "-i", str(root), "-o", str(out), "--scenes", "s",
                                "--stages", "phase1", "--direction", "forward", *extra], cwd=REPO, env=env,
                               capture_output=True, text=True, timeout=300)
            return r.stdout, log.read_text().split("\n")[0]
        sel = "Namespace(cameras='cam_001,cam_002')"
        out, seen = run("names", sel, "--cameras", "cam002,cam001")
        check("[S5] the runner receives the canonical names", seen == "phase1 cam_001,cam_002", seen)
        out, seen = run("dflt", "Namespace()")
        check("[S5] the default runs every camera", seen == "phase1 cam_000,cam_001,cam_002", seen)
        out, seen = run("count", "Namespace()", "--cameras", "2")
        check("[S5] a count runs the first N on an all-camera endpoint", seen == "phase1 cam_000,cam_001", seen)
        out, seen = run("refuse", "Namespace()", "--cameras", "cam_001,cam_002")
        check("[S5] an all-camera endpoint is refused for a name list",
              not seen and "REFUSED(static/initial, built from other cameras)" in out)
        out, seen = run("unk", "Namespace()", "--cameras", "cam_007")
        check("[S5] an unknown camera skips the scene", not seen and "SKIPPED(cameras not in the scene)" in out)


if __name__ == "__main__":
    test_parsing()
    test_static_subset()
    test_endpoint()
    test_signature_and_masks()
    test_driver()
    print(f"\n{CHECKS - len(FAILURES)} of {CHECKS} checks passed")
    for f in FAILURES:
        print("  FAILED:", f)
    raise SystemExit(1 if FAILURES else 0)
