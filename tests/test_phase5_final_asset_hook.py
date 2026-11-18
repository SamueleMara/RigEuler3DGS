"""Gate E: the asset export as the LAST Phase-5 operation (tools/train_phase5_rgbd_gaussians.py).

  * CUDA: a small real Phase-5 training (synthetic revolute scene, recovery.training.train_phase5) -> articulated PLY ->
    the driver's final_asset_export -> assets/<scene>/ with both PLYs and, with physics, URDF / USD / MJCF; the
    articulated PLY, the joint fingerprint, loss_log.csv and the final metrics are unchanged by the export
  * failure injection through the driver: required -> SystemExit (Phase 5 not completed), auto -> EXPORT_FAILED recorded
  * --asset-export-mode required fails in main() BEFORE training when the physics spec is missing
"""
import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import recovery  # noqa: E402,F401
from recovery import phase4_artifacts as AR  # noqa: E402
from recovery.contracts import load_joint  # noqa: E402
from recovery.sim_asset_export import export as EX  # noqa: E402
from recovery.sim_asset_export.ply_split import sha256_file  # noqa: E402
from sim_runtimes import run_all  # noqa: E402

spec_ = importlib.util.spec_from_file_location("p5driver", REPO / "tools" / "train_phase5_rgbd_gaussians.py")
D = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(D)

SMALL = dict(iters_A=20, iters_B=60, iters_C=20, densify_from=40, densify_every=40, densify_until=50, sh_interval=50,
             val_every=0, render_frames=0)


def _asset_args(argv):
    ap = argparse.ArgumentParser()
    D.add_asset_args(ap)
    return ap.parse_args(argv)


def _trained_run(tmp):
    """Real (small) Phase-5 training + articulated PLY, laid out as the driver leaves it before complete.json."""
    import torch
    from synthetic_articulated import make_scene
    from recovery.articulated_ply import write_articulated_ply
    from recovery.gaussians import GaussianTensors
    from recovery.training import Phase5Config, train_phase5
    s = make_scene("revolute", "forward", withhold_frac=0.0, seed=7)
    p3 = Path(tmp) / "phase3"
    (p3 / "revolute").mkdir(parents=True)
    j0 = s.joint
    (p3 / "revolute" / "geometry.json").write_text(json.dumps(
        {"joint_type": "revolute", "status": "FITTED", "geometry_unchanged_bitwise": True,
         "geometry_fixed": {"axis": list(j0.axis), "axis_point": list(j0.point)}}))
    (p3 / "revolute" / "trajectory.csv").write_text("processing_index,original_frame,q,posterior_sd_q\n" + "".join(
        f"{k},{f},{float(q)!r},0.001\n" for k, (f, q) in enumerate(zip(j0.frames, j0.q))))
    joint = load_joint(p3, "revolute", "forward")
    m = s.model
    m.lineage["material"] = s.gt_moving_model.astype(np.int8)
    m.max_sh_degree = 1
    out = Path(tmp) / "phase5" / "run"
    torch.manual_seed(0)
    r = train_phase5(m, joint, s.views, s.cameras, Phase5Config(mode="gaussian_rgbd", **SMALL), out,
                     log=lambda *a: None)
    fp = AR.joint_fingerprint(joint)
    lin = GaussianTensors.load_lineage(out / "final")
    write_articulated_ply(out / "final" / "point_cloud.ply", lin, joint, out / "final" / "point_cloud.ply",
                          unresolved_uids=lin["gaussian_uid"][:3])
    mf = float(np.mean([x["total"] for x in r["final"]]))
    completion = {"stage": "phase5", "train_mode": "gaussian_rgbd", "joint_fingerprint": fp,
                  "final_gaussians": int(len(lin["gaussian_uid"])), "joint_unchanged": True, "NOT_FOR_RESULTS": True,
                  "all_view_metrics_final": {"total": mf}}
    setup = {"source": "/nonexistent/final_synthetic_hook_episode_000000", "joint_type": "revolute",
             "direction": "forward", "phase3_dir": str(p3)}
    return out, joint, completion, setup, fp


def _physics(out, joint, tmp):
    """Box collision meshes fitted to the trained model's own part extents (fixture physics, declared)."""
    import yaml
    from plyfile import PlyData
    from synthetic_phase5_run import box_inertia, box_obj_text
    v = PlyData.read(str(out / "final" / "point_cloud.ply"))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(float)
    d = {"schema_version": 1, "registration": {"T_world_capture": None, "gauge": "capture_world",
                                               "gravity_alignment": "declared", "provenance": "test fixture"},
         "collision": {}, "links": {},
         "joint": {"lower": -0.2, "upper": 1.6, "velocity_limit": 2.0, "effort_limit": 4.0, "damping": 0.1,
                   "coulomb_friction": 0.0, "provenance": "measured"}}
    for side, lk, pid in (("static", "static_link", 0), ("moving", "moving_link", 1)):
        p = xyz[v["part_id"] == pid]
        lo, hi = np.percentile(p, 1, 0) - 0.01, np.percentile(p, 99, 0) + 0.01
        (Path(tmp) / f"{side}.obj").write_text(box_obj_text(lo, hi))
        d["collision"][side] = {"source": f"{side}.obj", "units": "m", "T_capture_mesh": np.eye(4).tolist(),
                                "provenance": "measured"}
        d["links"][lk] = {"mass_kg": 2.0, "com_capture_m": (0.5 * (lo + hi)).tolist(),
                          "inertia_com_capture_kg_m2": box_inertia(lo, hi, 2.0).tolist(), "provenance": "measured"}
        if side == "moving":
            d["collision"][side]["at_original_frame"] = joint.anchor_frame
            d["links"][lk]["at_original_frame"] = joint.anchor_frame
    sp = Path(tmp) / "physics.yaml"
    sp.write_text(yaml.safe_dump(d))
    return sp


def test_hook_writes_asset_last_and_changes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        out, joint, completion, setup, fp = _trained_run(tmp)
        guard = {p: sha256_file(out / p) for p in ("final/point_cloud.ply", "loss_log.csv",
                                                   "final/joint_unchanged_proof.json", "final/joint/geometry.json",
                                                   "final/joint/trajectory.csv")}
        sp = _physics(out, joint, tmp)
        a = _asset_args(["--asset-physics-spec", str(sp)])
        fields = D.final_asset_export(a, out, completion, setup, log=lambda *x: None)
        assert fields["asset_export_status"] == "FULL_PHYSICS_READY", fields
        assert fields["asset_export_dir"] == "assets/final_synthetic_hook_episode_000000"
        root = out / fields["asset_export_dir"]
        for f in ("visual/static.ply", "visual/moving.ply", "asset.yaml", "object.yaml", "joint_trajectory.csv",
                  "export_manifest.json", "lineage.npz", "maniskill/model.urdf", "isaac/model.usd",
                  "mujoco/model.xml", "physics/dynamics.yaml", "physics/collision_report.json", "export_complete.json"):
            assert (root / f).is_file(), f
        assert sha256_file(root / "export_manifest.json") == fields["asset_export_manifest_sha256"]
        assert {p: sha256_file(out / p) for p in guard} == guard, "export changed a Phase-5 file"
        assert AR.joint_fingerprint(joint) == fp
        print(f"  trained Phase 5 ({completion['final_gaussians']} Gaussians, final loss "
              f"{completion['all_view_metrics_final']['total']:.4f}) -> {fields['asset_export_status']} "
              f"{fields['asset_readiness']}")
        # visual-only default path of a run without physics
        f2 = D.final_asset_export(_asset_args(["--asset-revision", "nophys"]), out, completion, setup,
                                  log=lambda *x: None)
        assert f2["asset_export_status"] == "BLOCKED_MISSING_PHYSICS"
        assert D.final_asset_export(_asset_args(["--asset-export-mode", "off"]), out, completion, setup) == \
            {"asset_export_status": "OFF", "asset_export_mode": "off"}

        real = EX.write_mjcf
        EX.write_mjcf = lambda *x, **k: (_ for _ in ()).throw(RuntimeError("injected"))
        try:
            try:
                D.final_asset_export(_asset_args(["--asset-physics-spec", str(sp), "--asset-export-mode", "required",
                                                  "--asset-revision", "inj"]), out, completion, setup,
                                     log=lambda *x: None)
                raise AssertionError("required must stop Phase-5 completion")
            except SystemExit as e:
                assert "NOT completed" in str(e)
            f3 = D.final_asset_export(_asset_args(["--asset-physics-spec", str(sp), "--asset-revision", "inj"]), out,
                                      completion, setup, log=lambda *x: None)
            assert f3["asset_export_status"] == "EXPORT_FAILED" and "injected" in f3["asset_export_error"]
            assert not (out / "assets" / "final_synthetic_hook_episode_000000.rev-inj").exists()
        finally:
            EX.write_mjcf = real


def test_required_fails_before_training():
    import recovery.io_setup as IO
    import recovery.training as TR
    with tempfile.TemporaryDirectory() as tmp:
        from synthetic_phase5_run import make_phase5_run
        run = make_phase5_run(Path(tmp) / "fx", "revolute", "forward")
        fp = AR.joint_fingerprint(run["joint"])
        called = {"train": False}
        saved = (D.preflight, IO.setup, TR.train_phase5)

        class Up:
            hashes = {}
            joint = run["joint"]
        D.preflight = lambda p4: {"4a": {"input_proof": {"upstream_sha256": {}}}, "4c": {"joint_fingerprint": fp},
                                  "setup": {"source": "/x/s", "model": "m", "phase2_dir": "a", "phase3_dir": "b",
                                            "joint_type": "revolute", "direction": "forward"}}
        IO.setup = lambda ns: (Up, [], [], {})

        def no_train(*a, **k):
            called["train"] = True
            raise AssertionError("training started")
        TR.train_phase5 = no_train
        try:
            try:
                D.main(["--phase4-dir", str(tmp), "--output", str(Path(tmp) / "o"), "--asset-export-mode", "required"])
                raise AssertionError("expected SystemExit")
            except SystemExit as e:
                assert "--asset-physics-spec is required" in str(e)
            assert not called["train"] and not (Path(tmp) / "o").exists()
        finally:
            D.preflight, IO.setup, TR.train_phase5 = saved


if __name__ == "__main__":
    run_all(dict(globals()))
