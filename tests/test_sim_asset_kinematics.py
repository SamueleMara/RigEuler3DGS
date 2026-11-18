"""Gate B (CPU, float64): exact transport T_B_Lmoving(q_f) T_Lmoving_Pmoving x == Q_f x + p_f.

  * revolute and prismatic, arbitrary axes, a non-origin hinge point, negative q, forward and BACKWARD anchors:
    every recorded Phase-3 frame, random canonical points and the moving Gaussians
  * q = 0 is the identity; static Gaussians are invariant; covariance R S R^T and quaternion composition
    quat(R) ⊗ q_P; SH direction invariance when the camera moves with the link; points on the hinge stay fixed
  * R_B_J: deterministic, right-handed, +Z = axis, including axes along / near the basis vectors
  * conventions: URDF fixed-axis RPY (checked against scipy from_euler('xyz'), including gimbal lock), scalar-first
    Hamilton quaternions (checked against scipy), USD degrees, MuJoCo wxyz body quaternion
  * the demonstration range and the mechanical limits stay distinct in asset.yaml, the URDF and the USD
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from recovery.sim_asset_export import export_phase5_sim_asset  # noqa: E402
from recovery.sim_asset_export.frames import (R_from_urdf_rpy, T_B_L, T_B_P_moving, apply, axis_frame, inv_se3,  # noqa: E402
                                              joint_frame, quat_mul, quat_wxyz_from_R, se3, urdf_rpy_from_R)
from recovery.sim_asset_export.load_phase5 import load_phase5  # noqa: E402
from recovery.sim_asset_export.model import build_asset  # noqa: E402
from recovery.sim_asset_export.ply_split import split_articulated_ply  # noqa: E402
from recovery.sim_asset_export.usd_export import usd_units  # noqa: E402
from recovery.sim_asset_export.validate import check_analytic  # noqa: E402
from sim_runtimes import run_all  # noqa: E402
from synthetic_phase5_run import make_phase5_run, make_physics_spec  # noqa: E402

TOL = 1e-9


def test_axis_frame_is_deterministic_and_right_handed():
    rng = np.random.default_rng(0)
    axes = [np.eye(3)[i] * s for i in range(3) for s in (1, -1)] + [np.array([1e-12, 1.0, 1e-12]),
                                                                   np.array([1, 1, 1.0])] + list(rng.normal(size=(200, 3)))
    for a in axes:
        R = axis_frame(a)
        u = a / np.linalg.norm(a)
        assert np.abs(R[:, 2] - u).max() < 1e-15 and abs(np.linalg.det(R) - 1) < 1e-12
        assert np.abs(R.T @ R - np.eye(3)).max() < 1e-12
        assert np.array_equal(R, axis_frame(np.array(a, copy=True))), "deterministic"


def test_transport_equals_joint_transform_every_frame():
    rng = np.random.default_rng(1)
    cases = [("revolute", "forward", None, None), ("revolute", "backward", (-0.6, 0.3, 0.74), (0.31, -0.42, 2.2)),
             ("prismatic", "forward", None, None), ("prismatic", "backward", (0.7, -0.7, 0.14), None)]
    for jt, d, axis, point in cases:
        with tempfile.TemporaryDirectory() as tmp:
            run = make_phase5_run(tmp, jt, d, axis=axis, point=point, q_scale=1.2 if jt == "revolute" else 0.2)
            j = run["joint"]
            assert min(j.q) < 0, "the fixture has negative q"
            assert (j.anchor_frame == min(j.frames)) == (d == "forward")
            inp = load_phase5(run["phase5_dir"])
            T, o, kind = (joint_frame(jt, j.axis, j.point) if jt == "revolute" else
                          joint_frame(jt, j.axis, moving_xyz=inp.xyz[inp.part_id == 1]))
            if jt == "revolute":
                assert np.array_equal(o, np.asarray(j.point)), "o is exactly joint.point"
            pts = np.concatenate([rng.uniform(-3, 3, (500, 3)), inp.xyz[inp.part_id == 1]])
            for f in j.frames:
                Q, p = j.transform(f)
                Tp = T_B_P_moving(jt, T, j.q_of(f))
                assert np.abs(apply(Tp, pts) - (pts @ Q.T + p)).max() < TOL
                assert np.abs(Tp[:3, :3] - Q).max() < TOL
                Tl = T_B_L(jt, T, j.q_of(f))
                assert np.abs(apply(Tl @ inv_se3(T), pts) - (pts @ Q.T + p)).max() < TOL
            assert np.abs(T_B_P_moving(jt, T, 0.0) - np.eye(4)).max() < 1e-12
            if jt == "revolute":
                on_axis = np.asarray(j.point) + np.outer(rng.normal(size=50), j.axis)
                assert np.abs(apply(T_B_P_moving(jt, T, 0.9), on_axis) - on_axis).max() < TOL
            else:
                d0 = apply(T_B_P_moving(jt, T, 0.15), pts) - pts
                assert np.abs(d0 - 0.15 * np.asarray(j.axis)).max() < TOL, "prismatic moves along +axis"
            # full analytic validator (covariance, quaternion composition, SH direction)
            spl = split_articulated_ply(inp.ply_path, {"static_link": Path(tmp) / "s.ply",
                                                       "moving_link": Path(tmp) / "m.ply"})
            asset = build_asset(inp, spl, None, None)
            rep = check_analytic(asset, j, inp.xyz[inp.part_id == 1])
            assert rep["status"] == "PASS", rep
            assert asset.T_link_ply["static_link"].tolist() == np.eye(4).tolist()


def test_world_gaussians_static_invariant_and_quaternion_rule():
    rng = np.random.default_rng(2)
    TWB = se3(Rotation.random(random_state=3).as_matrix(), [0.4, -1.0, 0.2])
    T_B_J0 = se3(axis_frame([0.2, 0.3, -0.9]), [0.1, 0.2, 2.0])
    mu = rng.normal(size=(100, 3))
    qP = rng.normal(size=(100, 4))
    qP /= np.linalg.norm(qP, axis=1, keepdims=True)
    for q in (-0.7, 0.0, 0.4, 1.3):
        # static link: T_W_P = T_W_B @ I at every q
        T_W_Ps = TWB @ np.eye(4)
        assert np.array_equal(apply(T_W_Ps, mu), apply(TWB, mu))
        T_W_Pm = TWB @ T_B_L("revolute", T_B_J0, q) @ inv_se3(T_B_J0)
        R = T_W_Pm[:3, :3]
        qw = quat_mul(np.broadcast_to(quat_wxyz_from_R(R), qP.shape), qP)
        Rw = Rotation.from_quat(qw[:, [1, 2, 3, 0]]).as_matrix()
        Rp = Rotation.from_quat(qP[:, [1, 2, 3, 0]]).as_matrix()
        assert np.abs(Rw - R @ Rp).max() < 1e-12, "quat_W = quat(R_W_P) ⊗ quat_P (Hamilton, scalar first)"
        cam_P = rng.normal(size=(5, 3)) * 3
        d_P = (apply(T_W_Pm, mu)[:, None] - apply(T_W_Pm, cam_P)[None]) @ R
        assert np.abs(d_P - (mu[:, None] - cam_P[None])).max() < 1e-12, "no residual frame rotation in SH"


def test_rotation_conventions():
    rng = np.random.default_rng(4)
    Rs = list(Rotation.random(300, random_state=5).as_matrix()) + [
        Rotation.from_euler("xyz", [0.3, math.pi / 2, -0.2]).as_matrix(),
        Rotation.from_euler("xyz", [0.3, -math.pi / 2, 0.5]).as_matrix()]
    for R in Rs:
        rpy = urdf_rpy_from_R(R)
        assert np.abs(Rotation.from_euler("xyz", rpy).as_matrix() - R).max() < 1e-9, "URDF fixed-axis xyz"
        assert np.abs(R_from_urdf_rpy(rpy) - R).max() < 1e-9
        q = quat_wxyz_from_R(R)
        s = Rotation.from_matrix(R).as_quat()                      # x y z w
        s = np.r_[s[3], s[:3]]
        assert min(np.abs(q - s).max(), np.abs(q + s).max()) < 1e-12 and q[0] >= 0
    del rng


def test_usd_units_and_distinct_limits():
    with tempfile.TemporaryDirectory() as tmp:
        run = make_phase5_run(Path(tmp) / "r", "revolute", "forward")
        spec = make_physics_spec(run, Path(tmp) / "phys", lower=-0.6, upper=1.4)
        inp = load_phase5(run["phase5_dir"])
        from recovery.sim_asset_export.physics import resolve_physics
        from recovery.sim_asset_export.schema import parse_physics_spec
        spl = split_articulated_ply(inp.ply_path, {"static_link": Path(tmp) / "s.ply",
                                                   "moving_link": Path(tmp) / "m.ply"})
        T = joint_frame("revolute", inp.joint.axis, inp.joint.point)[0]
        a = build_asset(inp, spl, resolve_physics(parse_physics_spec(spec), inp, T), None)
        u = usd_units(a)
        assert abs(u["lower_usd"] - math.degrees(-0.6)) < 1e-12 and abs(u["upper_usd"] - math.degrees(1.4)) < 1e-12
        assert abs(u["damping_usd"] - 0.3 * math.pi / 180) < 1e-15 and u["limit_units"] == "deg"
        # exported files: observed range != physical stops, each labelled
        r = export_phase5_sim_asset(run["phase5_dir"], physics_spec=spec, log=lambda *a: None, verify=False)
        y = yaml.safe_load((r.asset_root / "asset.yaml").read_text())
        obs = y["joint"]["observed_demonstration_range"]
        lim = y["joint"]["physical_limits"]
        assert obs == [float(min(inp.joint.q)), float(max(inp.joint.q))] and (lim["lower"], lim["upper"]) == (-0.6, 1.4)
        assert lim["provenance"] == "physical_stops_measured" and y["joint"]["observed_range_is_not_a_mechanical_limit"]
        urdf = (r.asset_root / "maniskill" / "model.urdf").read_text()
        assert f'lower="{-0.6!r}"' in urdf and f'upper="{1.4!r}"' in urdf
        print(f"  revolute limits [-0.6, 1.4] rad -> USD [{u['lower_usd']:.6f}, {u['upper_usd']:.6f}] deg; "
              f"observed {obs}; USD file {'PASS' if r.usd_complete else 'NOT_RUN'}")


if __name__ == "__main__":
    run_all(dict(globals()))
