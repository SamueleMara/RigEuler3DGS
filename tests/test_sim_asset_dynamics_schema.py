"""Gate C (CPU): physics inputs are validated, never guessed.

Rejected with an actionable error: missing / negative mass, non-finite COM, non-symmetric / non-positive / non-physical
inertia, invalid or missing joint stops (no silent observed-range limits), wrong mesh units, a scaled or reflecting
T_capture_mesh, an empty / corrupted mesh, swapped (mismatched) collision link frames, a moving mesh declared at the
wrong frame, an RGB-D-reconstructed surface without end-effector exclusion, a non-convex surface without a
decomposition, unknown keys.
Known synthetic meshes: unit scaling, rigid placement, transport of the moving inputs from frame f to the anchor,
COM / inertia rotation, density integration, convex-proxy bookkeeping.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from recovery.sim_asset_export.frames import inv_se3, joint_frame  # noqa: E402
from recovery.sim_asset_export.load_phase5 import load_phase5  # noqa: E402
from recovery.sim_asset_export.physics import resolve_physics  # noqa: E402
from recovery.sim_asset_export.schema import PhysicsSpecError, parse_physics_spec  # noqa: E402
from sim_runtimes import run_all  # noqa: E402
from synthetic_phase5_run import (DELETE, MOVING_BOX, STATIC_BOX, box_inertia, box_obj_text, box_vertices,  # noqa: E402
                                  make_phase5_run, make_physics_spec)


def _ctx(tmp, jt="revolute", d="forward"):
    run = make_phase5_run(Path(tmp) / "run", jt, d)
    inp = load_phase5(run["phase5_dir"])
    T = (joint_frame(jt, inp.joint.axis, inp.joint.point)[0] if jt == "revolute" else
         joint_frame(jt, inp.joint.axis, moving_xyz=inp.xyz[inp.part_id == 1])[0])
    return run, inp, T


def _resolve(run, inp, T, tmp, name, **kw):
    spec = make_physics_spec(run, Path(tmp) / name, **kw)
    return resolve_physics(parse_physics_spec(spec), inp, T)


def _rejects(fn, needle):
    try:
        fn()
    except PhysicsSpecError as e:
        assert needle.lower() in str(e).lower(), f"{needle!r} not in: {e}"
        return
    raise AssertionError(f"accepted; expected rejection containing {needle!r}")


def test_rejections():
    with tempfile.TemporaryDirectory() as tmp:
        run, inp, T = _ctx(tmp)
        R = lambda name, **kw: _resolve(run, inp, T, tmp, name, **kw)          # noqa: E731
        bad_I = [[1, 0.5, 0], [0.2, 1, 0], [0, 0, 1]]
        cases = [
            ("no_mass", dict(extra={"links.static_link.mass_kg": None}), "mass_kg"),
            ("neg_mass", dict(extra={"links.static_link.mass_kg": -2.0}), "must be >"),
            ("nan_com", dict(extra={"links.static_link.com_capture_m": [0.0, float("nan"), 2.1]}), "not finite"),
            ("asym_I", dict(extra={"links.static_link.inertia_com_capture_kg_m2": bad_I}), "symmetric"),
            ("npd_I", dict(extra={"links.static_link.inertia_com_capture_kg_m2": [[1, 0, 0], [0, -1, 0], [0, 0, 1]]}),
             "positive definite"),
            ("tri_I", dict(extra={"links.static_link.inertia_com_capture_kg_m2": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]]}),
             "triangle"),
            ("lo_gt_hi", dict(lower=0.5, upper=-0.1, extra={"joint.lower": 0.5, "joint.upper": -0.1}), "must be <"),
            ("no_stops", dict(extra={"joint.lower": None, "joint.upper": None}), "NOT a mechanical limit"),
            ("anchor_out", dict(extra={"joint.lower": 0.1, "joint.upper": 1.0}), "outside the stops"),
            ("no_effort", dict(extra={"joint.effort_limit": DELETE}), "effort_limit"),
            ("typo", dict(extra={"joint.dampnig": 0.1}), "unknown keys"),
            ("rgbd", dict(extra={"collision.static.source_kind": "reconstructed_rgbd",
                                 "collision.static.provenance": "estimated"}), "COLLISION_UNRESOLVED"),
            ("units", dict(units="mm", extra={"collision.static.units": "m", "collision.moving.units": "m"}),
             "extent"),
            ("no_frame", dict(extra={"collision.moving.at_original_frame": DELETE}), "at_original_frame"),
            ("bad_frame", dict(extra={"collision.moving.at_original_frame": 9999}), "not a Phase-3 frame"),
        ]
        for name, kw, needle in cases:
            _rejects(lambda: R(name, **kw), needle)
        # scaled / reflecting rigid transform
        S = (2 * np.eye(4)).tolist()
        S[3][3] = 1.0
        _rejects(lambda: R("scaled", extra={"collision.moving.T_capture_mesh": S}), "orthonormal")
        F = np.diag([1.0, 1.0, -1.0, 1.0]).tolist()
        _rejects(lambda: R("reflect", extra={"collision.moving.T_capture_mesh": F}), "reflection")
        # swapped meshes -> mismatched link frames
        spec = make_physics_spec(run, Path(tmp) / "swap")
        d = yaml.safe_load(spec.read_text())
        d["collision"]["static"]["source"], d["collision"]["moving"]["source"] = "moving_mesh.obj", "static_mesh.obj"
        d["collision"]["static"]["T_capture_mesh"] = np.eye(4).tolist()
        spec.write_text(yaml.safe_dump(d))
        _rejects(lambda: resolve_physics(parse_physics_spec(spec), inp, T), "collision")
        # moving mesh posed at the last frame but declared at the anchor
        fr = run["joint"].chronological()[0]
        _rejects(lambda: R("wrong_pose", moving_frame=int(fr[-1]),
                           extra={"collision.moving.at_original_frame": int(run["joint"].anchor_frame)}),
                 "mismatched collision link frames")
        # empty / corrupted mesh
        for nm, txt, needle in (("empty", "# nothing\n", "empty mesh"),
                                ("corrupt", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n", "out of range")):
            spec = make_physics_spec(run, Path(tmp) / nm)
            (spec.parent / "static_mesh.obj").write_text(txt)
            _rejects(lambda: parse_physics_spec(spec), needle)
        # non-convex surface without a decomposition
        spec = make_physics_spec(run, Path(tmp) / "L")
        (spec.parent / "static_mesh.obj").write_text(_l_shape_obj(np.linalg.inv(np.asarray(
            yaml.safe_load(spec.read_text())["collision"]["static"]["T_capture_mesh"]))))
        _rejects(lambda: resolve_physics(parse_physics_spec(spec), inp, T), "not convex")


def _l_shape_obj(T=np.eye(4)):
    """Union of two boxes inside STATIC_BOX: an L (non-convex), closed, outward."""
    lo, hi = STATIC_BOX
    mid = 0.5 * (lo + hi)
    # polygon in x-y extruded along z (closed prism); vertices of the L outline
    P = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], mid[1]), (mid[0], mid[1]), (mid[0], hi[1]), (lo[0], hi[1])]
    V = [(x, y, lo[2]) for x, y in P] + [(x, y, hi[2]) for x, y in P]
    V = np.asarray(V) @ T[:3, :3].T + T[:3, 3]
    F = []
    tri = [(0, 1, 2), (0, 2, 3), (0, 3, 5), (3, 4, 5)]                  # L outline, counter-clockwise from +z
    F += [(c, b2, a2) for a2, b2, c in tri]                               # bottom (facing -z)
    F += [(a2 + 6, b2 + 6, c + 6) for a2, b2, c in tri]                   # top
    for i in range(6):
        j = (i + 1) % 6
        F += [(i, j, j + 6), (i, j + 6, i + 6)]
    return "".join(f"v {x!r} {y!r} {z!r}\n" for x, y, z in V) + "".join(f"f {i + 1} {j + 1} {k + 1}\n"
                                                                         for i, j, k in F)


def test_known_meshes_and_inertials():
    with tempfile.TemporaryDirectory() as tmp:
        for jt in ("revolute", "prismatic"):
            run, inp, T = _ctx(Path(tmp) / jt, jt, "backward" if jt == "prismatic" else "forward")
            for units in ("m", "mm"):
                ph = _resolve(run, inp, T, tmp, f"{jt}_{units}", units=units)
                s = ph.collision["static_link"]
                lo, hi = STATIC_BOX
                v = s.surface.vertices
                assert np.abs(np.sort(v, 0) - np.sort(box_vertices(lo, hi), 0)).max() < 1e-9, "scale + placement"
                m = ph.collision["moving_link"]
                lo_m, hi_m = MOVING_BOX[jt]
                want = box_vertices(lo_m, hi_m) @ inv_se3(T)[:3, :3].T + inv_se3(T)[:3, 3]
                assert np.abs(np.sort(m.surface.vertices, 0) - np.sort(want, 0)).max() < 1e-9, \
                    "moving mesh posed at frame f transported to J0 with the frozen joint"
                assert s.approximation == "source_is_convex" and len(s.parts) == 1
            mi = ph.inertial["moving_link"]
            Tl = inv_se3(T)
            com_want = Tl[:3, :3] @ (0.5 * (lo_m + hi_m)) + Tl[:3, 3]
            I_want = Tl[:3, :3] @ box_inertia(lo_m, hi_m, 1.5) @ Tl[:3, :3].T
            assert np.abs(mi.com - com_want).max() < 1e-12 and np.abs(mi.inertia - I_want).max() < 1e-12
            assert mi.provenance == "measured" and ph.provenance_summary["all_measured"] is False  # contact estimated
            ph = _resolve(run, inp, T, tmp, f"{jt}_dens", moving_inertia="density")
            mi = ph.inertial["moving_link"]
            vol = float(np.prod(hi_m - lo_m))
            assert abs(mi.mass - 600 * vol) < 1e-9 * mi.mass and mi.provenance == "density_assumption"
            assert np.abs(mi.inertia - I_want / 1.5 * 600 * vol).max() < 1e-9 * np.abs(I_want).max()
            assert np.abs(mi.com - com_want).max() < 1e-12


def test_convex_proxy_bookkeeping():
    with tempfile.TemporaryDirectory() as tmp:
        run, inp, T = _ctx(tmp)
        spec = make_physics_spec(run, Path(tmp) / "dec")
        d = yaml.safe_load(spec.read_text())
        Tcm = np.asarray(d["collision"]["static"]["T_capture_mesh"])
        (spec.parent / "static_mesh.obj").write_text(_l_shape_obj(np.linalg.inv(Tcm)))
        lo, hi = STATIC_BOX
        mid = 0.5 * (lo + hi)
        (spec.parent / "part_a.obj").write_text(box_obj_text(lo, np.array([hi[0], mid[1], hi[2]]), np.linalg.inv(Tcm)))
        (spec.parent / "part_b.obj").write_text(box_obj_text(lo, np.array([mid[0], hi[1], hi[2]]), np.linalg.inv(Tcm)))
        d["collision"]["static"]["convex_parts"] = ["part_a.obj", "part_b.obj"]
        d["collision"]["static"]["convex_coverage_tolerance_m"] = 1e-6
        spec.write_text(yaml.safe_dump(d))
        ph = resolve_physics(parse_physics_spec(spec), inp, T)
        s = ph.collision["static_link"]
        assert s.approximation == "supplied_convex_decomposition" and len(s.parts) == 2 and not ph.assumptions[1:]
        assert s.report["coverage"]["max_surface_vertex_distance_m"] <= 1e-6
        # parts that leave part of the surface uncovered are refused
        d["collision"]["static"]["convex_parts"] = ["part_a.obj"]
        spec.write_text(yaml.safe_dump(d))
        _rejects(lambda: resolve_physics(parse_physics_spec(spec), inp, T), "outside")
        # opt-in single hull: estimated, recorded as an assumption
        d["collision"]["static"].pop("convex_parts")
        d["collision"]["static"].pop("convex_coverage_tolerance_m")
        d["collision"]["static"]["approximation"] = "convex_hull"
        spec.write_text(yaml.safe_dump(d))
        ph = resolve_physics(parse_physics_spec(spec), inp, T)
        assert ph.collision["static_link"].provenance == "estimated"
        assert any("convex hull" in a for a in ph.assumptions)
        hv = ph.collision["static_link"].report["hull"]["volume"]
        # hull of the L prism: the bounding box minus the corner triangle (W/2 x H/2 / 2) -> 7/8 of the box
        assert abs(hv - 7.0 / 8.0 * float(np.prod(hi - lo))) < 1e-9, hv


if __name__ == "__main__":
    run_all(dict(globals()))
