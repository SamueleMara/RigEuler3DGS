"""Gate D / USD: model.usd is a real binary OpenUSD stage (PXR-USDC), opened with pxr.Usd.Stage.Open in a separate
interpreter: ArticulationRootAPI, 2 rigid bodies with MassAPI, Revolute/PrismaticJoint body0/body1, local anchors that
reproduce the analytic link pose at every q, limits in DEGREES for revolute, collisionEnabled false, convex-hull
collision meshes, relative PLY references + T_link_ply. Without a USD runtime the asset is honestly
PHYSICS_INCOMPLETE_RUNTIME (no model.usd, USD NOT_RUN) and `required` fails. The Isaac PhysX runtime is NOT_RUN here.
"""
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from recovery.sim_asset_export import ExportError, export_phase5_sim_asset  # noqa: E402
from recovery.sim_asset_export import export as EX  # noqa: E402
from recovery.sim_asset_export.usd_export import read_usd  # noqa: E402
from sim_runtimes import report_not_run, run_all, runtime  # noqa: E402
from synthetic_phase5_run import make_phase5_run, make_physics_spec  # noqa: E402


def test_usd_stage_schema_units_and_fk():
    py = runtime("usd")
    if not py:
        report_not_run("USD stage", "usd")
        return
    for jt, d in (("revolute", "backward"), ("prismatic", "forward")):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            run = make_phase5_run(Path(tmp) / "run", jt, d)
            spec = make_physics_spec(run, Path(tmp) / "phys", lower=-0.5, upper=1.25 if jt == "revolute" else 0.3)
            r = export_phase5_sim_asset(run["phase5_dir"], physics_spec=spec, log=lambda *a: None, verify=False)
            u = r.validation_report["usd"]
            assert u["status"] == "PASS" and r.usd_complete, u
            assert (r.asset_root / "isaac" / "model.usd").read_bytes()[:8] == b"PXR-USDC"
            moved = Path(other) / r.asset_root.name
            shutil.copytree(r.asset_root, moved)
            rb = read_usd(moved / "isaac" / "model.usd", py)             # re-open from another absolute path
            J = rb["joints"]["articulation_joint"]
            k = 180 / math.pi if jt == "revolute" else 1.0
            hi = 1.25 if jt == "revolute" else 0.3
            assert abs(J["lower"] - (-0.5) * k) < 1e-4 and abs(J["upper"] - hi * k) < 1e-4, (J["lower"], J["upper"])
            assert J["collision_enabled"] is False and J["axis"] == "Z"
            for lk in ("static_link", "moving_link"):
                ply = rb["links"][lk]["visual_ply"]
                assert not Path(ply).is_absolute() and (moved / "isaac" / ply).resolve().is_file()
            assert all(c["approximation"] == "convexHull" for c in rb["collisions"].values())
            print(f"  {jt}: USD {tuple(u['usd_version'])}, joint-frame FK error {u['joint_frame_fk_error']:.2e}, "
                  f"limits {J['lower']:.4f} .. {J['upper']:.4f} {'deg' if jt == 'revolute' else 'm'}, Isaac runtime "
                  f"{u['isaac_runtime']}")


def test_missing_usd_runtime_is_not_a_pass():
    real = EX.V.find_runtime

    def no_usd(kind, explicit=None):
        return {"python": None, "source": None} if kind == "usd" else real(kind, explicit)
    EX.V.find_runtime = no_usd
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run = make_phase5_run(Path(tmp) / "run", "revolute", "forward")
            spec = make_physics_spec(run, Path(tmp) / "phys")
            r = export_phase5_sim_asset(run["phase5_dir"], physics_spec=spec, log=lambda *a: None, verify=False)
            assert r.status == "PHYSICS_INCOMPLETE_RUNTIME" and not r.usd_complete and not r.physics_complete
            assert r.validation_report["usd"]["status"] == "NOT_RUN"
            assert not (r.asset_root / "isaac" / "model.usd").exists()
            assert any("usd" in m for m in r.missing_inputs)
            try:
                export_phase5_sim_asset(run["phase5_dir"], physics_spec=spec, mode="required", revision="req",
                                        log=lambda *a: None)
                raise AssertionError("required must fail without a USD runtime")
            except ExportError as e:
                assert "usd" in str(e)
            assert not (run["phase5_dir"] / "assets" / f"{r.scene}.rev-req").exists()
    finally:
        EX.V.find_runtime = real


if __name__ == "__main__":
    run_all(dict(globals()))
