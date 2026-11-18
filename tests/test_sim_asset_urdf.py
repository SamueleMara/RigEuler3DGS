"""Gate D / URDF: generated URDF read back as XML (independent RPY decoding), then loaded by the real SAPIEN URDF loader
(ManiSkill's physics backend) in its own interpreter: 2 links, 1 DOF, axis / limits, q = 0 = canonical, live PhysX
link poses == analytic SE(3) for every recorded q and the limits; masses / inertias read back; engine-specific joint
parameters reported (not equated); PArtTwin1D's own loader + LinkAdapter transports the unchanged PLYs from the live
SAPIEN link poses; contact smoke test. Also after moving the asset folder to another absolute path.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from recovery.sim_asset_export import export_phase5_sim_asset  # noqa: E402
from recovery.sim_asset_export import validate as V  # noqa: E402
from sim_runtimes import report_not_run, run_all, runtime  # noqa: E402
from synthetic_phase5_run import make_phase5_run, make_physics_spec  # noqa: E402


def _export(tmp, jt, d):
    run = make_phase5_run(Path(tmp) / "run", jt, d)
    spec = make_physics_spec(run, Path(tmp) / "phys")
    return run, export_phase5_sim_asset(run["phase5_dir"], physics_spec=spec, log=lambda *a: None)


def test_urdf_text_and_sapien_runtime():
    for jt, d in (("revolute", "forward"), ("prismatic", "backward")):
        with tempfile.TemporaryDirectory() as tmp:
            run, r = _export(tmp, jt, d)
            u = r.validation_report["urdf"]
            assert u["status"] == "PASS" and u["fk_max_error_all_frames"] < 1e-12, u
            txt = (r.asset_root / "maniskill" / "model.urdf").read_text()
            assert "<visual" not in txt and ".ply" not in txt, "PLYs are never URDF meshes"
            assert txt.count("<link ") == 2 and txt.count("<joint ") == 1
            s = r.validation_report.get("sapien_parttwin1d")
            if not runtime("sapien"):
                report_not_run("SAPIEN URDF import", "sapien")
                assert s["status"] == "NOT_RUN"
                continue
            assert s["status"] == "PASS", s
            assert s["parttwin1d_consumer"]["status"] in ("PASS", "NOT_RUN")
            print(f"  {jt}: SAPIEN {s['sapien_version']} {s['q_evaluated']} q, link pose error "
                  f"{s['pose_max_error']:.2e} m/rad; PArtTwin1D adapter {s['parttwin1d_consumer']['status']} "
                  f"(means {s['parttwin1d_consumer'].get('means_max_error_m', float('nan')):.2e} m); contact "
                  f"{s['contact']['status']} ({s['contact']['final_height_above_static_top_m'] * 1e3:.2f} mm above "
                  f"the top, r = {s['contact']['radius_m'] * 1e3:.1f} mm); engine deviations "
                  f"{ {k: v['sapien'] for k, v in s['engine_deviations'].items()} }")


def test_moved_folder_still_loads():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as elsewhere:
        run, r = _export(tmp, "revolute", "forward")
        moved = Path(elsewhere) / "deep" / "copy" / r.asset_root.name
        shutil.copytree(r.asset_root, moved)
        shutil.rmtree(run["phase5_dir"])                              # nothing may point back into the run
        # the Phase-5 run is gone: the copy must be self-contained (no absolute path back) and load in the engine
        assert V.check_portable(moved, [str(Path(tmp).resolve()), str(r.asset_root)])["status"] == "PASS"
        if runtime("sapien"):
            import json
            import subprocess
            q = {"q": [0.0, 0.5], "T_world_root": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                 "object_yaml": str(moved / "object.yaml"), "adapter_q": [0.0]}
            (Path(elsewhere) / "req.json").write_text(json.dumps(q))
            out = subprocess.run([runtime("sapien"), str(Path(V.__file__).with_name("_sapien_check.py")),
                                  str(moved / "maniskill" / "model.urdf"), str(Path(elsewhere) / "req.json"),
                                  str(Path(elsewhere) / "o.json"), str(Path(elsewhere) / "o.npz")],
                                 capture_output=True, text=True)
            assert out.returncode == 0, out.stderr[-2000:]
            o = json.loads((Path(elsewhere) / "o.json").read_text())
            assert o["dof"] == 1 and all(o["inertial"][k]["collision_shapes"] >= 1 for k in o["inertial"])
            assert o.get("parttwin", {}).get("counts", [1])[0] > 0
            print(f"  moved asset: SAPIEN loads URDF + collision meshes, PArtTwin1D loads object.yaml from {moved}")
        else:
            report_not_run("moved-folder SAPIEN load", "sapien")


if __name__ == "__main__":
    run_all(dict(globals()))
