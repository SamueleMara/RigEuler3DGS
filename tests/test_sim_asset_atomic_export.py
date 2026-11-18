"""Transaction semantics (spec 7): atomic promotion, failure injection, idempotence, refusal of silent replacement,
revisions, tamper detection, readiness statuses, and that a post-hoc export never modifies the Phase-5 run.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from recovery.sim_asset_export import ExportError, export_phase5_sim_asset  # noqa: E402
from recovery.sim_asset_export import export as EX  # noqa: E402
from recovery.sim_asset_export.ply_split import sha256_file  # noqa: E402
from sim_runtimes import run_all  # noqa: E402
from synthetic_phase5_run import make_phase5_run, make_physics_spec  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
Q = dict(log=lambda *a: None, verify=False)


def _p5_hashes(p5: Path):
    return {str(p.relative_to(p5)): sha256_file(p) for p in sorted(p5.rglob("*"))
            if p.is_file() and "assets" not in p.relative_to(p5).parts}


def _expect_export_error(fn, needle):
    try:
        fn()
    except ExportError as e:
        assert needle in str(e), f"{needle!r} not in {e}"
        return
    raise AssertionError(f"expected ExportError ({needle})")


def test_failure_injection_leaves_nothing_that_looks_complete():
    with tempfile.TemporaryDirectory() as tmp:
        run = make_phase5_run(Path(tmp) / "run", "revolute", "forward")
        spec = make_physics_spec(run, Path(tmp) / "phys")
        real = EX.write_mjcf

        def boom(*a, **k):
            raise RuntimeError("injected MJCF writer failure")
        EX.write_mjcf = boom
        try:
            for mode in ("auto", "required"):
                _expect_export_error(lambda: export_phase5_sim_asset(run["phase5_dir"], physics_spec=spec, mode=mode,
                                                                     **Q), "injected")
                assets = run["phase5_dir"] / "assets"
                assert not (assets / run["scene"]).exists(), "no asset directory after a failure"
                assert not [p for p in assets.iterdir() if p.name.startswith(".staging")], "staging removed"
                assert not list(assets.rglob("export_complete.json"))
                f = json.loads((assets / f"{run['scene']}.export_failed.json").read_text())
                assert "injected" in f["error"] and f["asset_dir_written"] is False
        finally:
            EX.write_mjcf = real
        r = export_phase5_sim_asset(run["phase5_dir"], physics_spec=spec, **Q)
        assert r.status == "FULL_PHYSICS_READY"
        assert not (run["phase5_dir"] / "assets" / f"{run['scene']}.export_failed.json").exists()


def test_idempotence_refusal_revision_and_tamper():
    with tempfile.TemporaryDirectory() as tmp:
        run = make_phase5_run(Path(tmp) / "run", "prismatic", "forward")
        p5 = run["phase5_dir"]
        before = _p5_hashes(p5)
        spec = make_physics_spec(run, Path(tmp) / "phys")
        r1 = export_phase5_sim_asset(p5, physics_spec=spec, **Q)
        r2 = export_phase5_sim_asset(p5, physics_spec=spec, **Q)
        assert r2.reused and r2.status == r1.status and r2.manifest_sha256 == r1.manifest_sha256
        assert _p5_hashes(p5) == before, "the Phase-5 run is never modified"
        spec2 = make_physics_spec(run, Path(tmp) / "phys2", lower=-0.05, upper=0.4)
        _expect_export_error(lambda: export_phase5_sim_asset(p5, physics_spec=spec2, **Q), "DIFFERENT inputs")
        r3 = export_phase5_sim_asset(p5, physics_spec=spec2, revision="stops2", **Q)
        assert r3.asset_root.name == f"{run['scene']}.rev-stops2" and not r3.reused
        (r1.asset_root / "maniskill" / "model.urdf").write_text("<robot/>")
        _expect_export_error(lambda: export_phase5_sim_asset(p5, physics_spec=spec, **Q), "changed after export")


def test_statuses_never_overclaim():
    with tempfile.TemporaryDirectory() as tmp:
        run = make_phase5_run(Path(tmp) / "run", "revolute", "forward")
        p5 = run["phase5_dir"]
        r = export_phase5_sim_asset(p5, **Q)                                       # no physics
        assert r.status == "BLOCKED_MISSING_PHYSICS" and r.visual_complete and not r.physics_complete
        for f in ("maniskill/model.urdf", "isaac/model.usd", "mujoco/model.xml", "physics"):
            assert not (r.asset_root / f).exists(), f"{f} must not exist without validated physics"
        assert "NOT generated" in (r.asset_root / "object.yaml").read_text() and r.missing_inputs
        _expect_export_error(lambda: export_phase5_sim_asset(p5, mode="required", revision="x", **Q), "required")
        r = export_phase5_sim_asset(p5, mode="visual_only", revision="vis", **Q)
        assert r.status == "VISUAL_ONLY" and not (r.asset_root / "physics").exists()
        bad = make_physics_spec(run, Path(tmp) / "bad", extra={"links.static_link.mass_kg": -1.0})
        r = export_phase5_sim_asset(p5, physics_spec=bad, revision="bad", **Q)
        assert r.status == "PHYSICS_REJECTED" and not (r.asset_root / "maniskill").exists()
        _expect_export_error(lambda: export_phase5_sim_asset(p5, physics_spec=bad, mode="required", revision="bad2",
                                                             **Q), "rejected")
        pre = make_physics_spec(run, Path(tmp) / "prev", extra={"joint.lower": None, "joint.upper": None,
                                                               "joint.limits_from_observed_range": True})
        r = export_phase5_sim_asset(p5, physics_spec=pre, revision="preview", **Q)
        assert r.status == "PHYSICS_PREVIEW_ASSUMPTIONS" and not r.physics_complete
        noreg = make_physics_spec(run, Path(tmp) / "noreg", registration=False)
        r = export_phase5_sim_asset(p5, physics_spec=noreg, revision="noreg", **Q)
        assert r.status == "PHYSICS_PREVIEW_ASSUMPTIONS"
        assert any("gravity" in a for a in json.loads((r.asset_root / "export_manifest.json").read_text())
                   ["readiness"]["physics_assumptions"])


def test_post_hoc_cli_is_non_mutating():
    with tempfile.TemporaryDirectory() as tmp:
        run = make_phase5_run(Path(tmp) / "run", "revolute", "backward")
        p5 = run["phase5_dir"]
        before = _p5_hashes(p5)
        spec = make_physics_spec(run, Path(tmp) / "phys")
        cmd = [sys.executable, str(REPO / "tools" / "export_phase5_sim_asset.py"), "--phase5-dir", str(p5),
               "--asset-physics-spec", str(spec), "--asset-verify", "false"]
        o1 = subprocess.run(cmd, capture_output=True, text=True)
        assert o1.returncode == 0, o1.stderr[-2000:]
        o2 = subprocess.run(cmd, capture_output=True, text=True)
        assert o2.returncode == 0 and json.loads(o2.stdout[o2.stdout.index("{"):])["reused"] is True
        assert _p5_hashes(p5) == before
        o3 = subprocess.run(cmd + ["--asset-export-mode", "required", "--asset-revision", "r",
                                   "--asset-mujoco-python", "/nonexistent/python"], capture_output=True, text=True)
        assert o3.returncode == 2 and "cannot import" in o3.stderr


if __name__ == "__main__":
    run_all(dict(globals()))
