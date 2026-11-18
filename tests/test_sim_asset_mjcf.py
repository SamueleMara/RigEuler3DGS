"""Gate D / MJCF: model.xml compiled by mujoco.MjModel.from_xml_path (separate interpreter); mj_forward at every
recorded q, at the stops, at negative / positive edge values: xpos / xmat of both bodies == analytic SE(3); exactly one
DOF (no spurious joints); hinge / slide on +Z at the child frame; mass, COM and full inertia read back; mesh geom
vertices land where the link-frame proxies say; contact smoke test reported; compiles from a moved folder.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from recovery.sim_asset_export import export_phase5_sim_asset  # noqa: E402
from sim_runtimes import report_not_run, run_all, runtime  # noqa: E402
from synthetic_phase5_run import make_phase5_run, make_physics_spec  # noqa: E402


def test_mjcf_compiles_and_matches_analytic():
    if not runtime("mujoco"):
        report_not_run("MJCF compile", "mujoco")
        return
    for jt, d in (("revolute", "forward"), ("prismatic", "backward")):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            run = make_phase5_run(Path(tmp) / "run", jt, d)
            spec = make_physics_spec(run, Path(tmp) / "phys", moving_inertia="density")
            r = export_phase5_sim_asset(run["phase5_dir"], physics_spec=spec, log=lambda *a: None, verify=False)
            m = r.validation_report["mjcf"]
            assert m["status"] == "PASS", m
            assert m["q_evaluated"] == len(run["joint"].frames) + 5
            assert m["joint_readback"]["type_name"] == ("hinge" if jt == "revolute" else "slide")
            assert max(v["inertia_rel"] for v in m["inertial_error"].values()) < 1e-6
            assert m["contact"]["status"] in ("RESTING_ON_STATIC", "DEVIATION")
            print(f"  {jt}: MuJoCo {m['mujoco_version']}, {m['q_evaluated']} q, pose error {m['pose_max_error']:.1e}, "
                  f"mesh vertices {m['mesh_vertex_max_error_m']:.1e} m, inertia rel "
                  f"{max(v['inertia_rel'] for v in m['inertial_error'].values()):.1e}; contact {m['contact']['status']} "
                  f"(penetration {m['contact']['penetration_m'] * 1e3:.2f} mm)")
            moved = Path(other) / r.asset_root.name
            shutil.copytree(r.asset_root, moved)
            code = ("import mujoco,sys;m=mujoco.MjModel.from_xml_path(sys.argv[1]);"
                    "print(m.nq,m.nv,m.njnt,m.nbody,m.nmesh)")
            out = subprocess.run([runtime("mujoco"), "-c", code, str(moved / "mujoco" / "model.xml")],
                                 capture_output=True, text=True)
            assert out.returncode == 0, out.stderr
            assert out.stdout.split()[:4] == ["1", "1", "1", "3"], out.stdout


if __name__ == "__main__":
    run_all(dict(globals()))
