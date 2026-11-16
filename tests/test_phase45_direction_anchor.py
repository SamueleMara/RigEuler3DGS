"""Gate 2: articulated canonical rendering under the fixed Phase-3 joint (spec 2.2, 3.3 item 5, 7.4).

  * identity at the anchor (positions, quaternions and the rendered image, bitwise)
  * static Gaussians never move; moving pairwise distances preserved; covariance rotates as Q Sigma Q^T
  * exact pose relative to the anchor = the pipeline's own Phase-3 transport (estimator states), forward
    and backward runs (real v6 fridge / blue box, synthetic v5 USB / blade)
  * original-frame mapping (backward: processing row k is original frame anchor - k)
  * gradients: through a tensor-only P / R joint into positions AND quaternions (finite differences), joint
    gradients of static Gaussians identically zero, rendered-image joint gradients CUDA == oracle
  * no joint tensor requires grad in the Phase 4/5 transport
  * local_SH == world_SH at the anchor; local_SH colour of a moving Gaussian rotates with the part
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery  # noqa: E402,F401
from recovery import raster_oracle as O  # noqa: E402
from recovery.articulated_scene import (FrameTransform, JointTransport, joint_transform_torch, posed,  # noqa: E402
                                        quat_to_rotmat, render_articulated)
from recovery.contracts import Joint, load_upstream, rot_exp  # noqa: E402
from recovery.views import camera_from_K  # noqa: E402
import test_phase45_contracts as TC  # noqa: E402

DEV = "cuda"


def toy_joint(jtype="revolute", direction="forward", T=8, q_max=0.9):
    frames = list(range(10, 10 + T)) if direction == "forward" else list(range(10 + T - 1, 9, -1))
    q = [0.0] + [q_max * k / (T - 1) for k in range(1, T)]
    axis = np.array([0.3, -0.2, 0.93]); axis /= np.linalg.norm(axis)
    return Joint(type=jtype, axis=tuple(axis), point=(0.1, -0.05, 2.3) if jtype == "revolute" else None,
                 frames=tuple(frames), q=tuple(q), posterior_sd_q=tuple([0.0] * T), direction=direction,
                 geometry_file="toy", trajectory_file="toy", geometry_sha256="", trajectory_sha256="")


class ToyGaussians:
    def __init__(self, N=60, seed=0, deg=3):
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.rand(*s, generator=g)
        self._xyz = torch.cat([(r(N, 2) - 0.5) * 0.8, 2.2 + 0.4 * r(N, 1)], 1)
        self._rotation = torch.randn(N, 4, generator=g)
        self._scaling = torch.log(0.03 + 0.05 * r(N, 3))
        self._opacity = torch.logit(0.3 + 0.5 * r(N, 1))
        self._features_dc = (r(N, 1, 3) - 0.5) * 2
        self._features_rest = (r(N, (deg + 1) ** 2 - 1, 3) - 0.5) * 0.4
        self.active_sh_degree = deg
        self.moving = torch.arange(N) < N // 2

    def to(self, dev):
        for k in ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest", "moving"):
            setattr(self, k, getattr(self, k).to(dev))
        return self


def toy_camera(eye=(0.4, 0.1, 0.0), W=64, H=48):
    eye = np.asarray(eye, float)
    z = np.array([0.0, 0.0, 2.4]) - eye
    z /= np.linalg.norm(z)
    x = np.cross([0.0, -1.0, 0.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4); c2w[:3, :3] = np.stack([x, y, z], 1); c2w[:3, 3] = eye
    f = W / (2 * math.tan(math.radians(25)))
    return camera_from_K("cam_toy", [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], c2w, W, H)


def test_identity_static_rigidity_covariance():
    for jt in ("revolute", "prismatic"):
        for d in ("forward", "backward"):
            j = toy_joint(jt, d)
            g = ToyGaussians().to(DEV)
            tr = JointTransport(j, DEV)
            rot = torch.nn.functional.normalize(g._rotation, dim=1)
            mu, q = posed(g._xyz, rot, g.moving, tr.at(j.anchor_frame))
            assert torch.equal(mu, g._xyz) and torch.equal(q, rot), "anchor must be the identity"
            cam = toy_camera()
            bg = torch.ones(3, device=DEV)
            a = render_articulated(cam, tr.at(j.anchor_frame), g, g.moving, bg, "world_SH")
            b = render_articulated(cam, tr.at(j.anchor_frame), g, torch.zeros_like(g.moving), bg, "world_SH")
            assert torch.equal(a["color"], b["color"]) and torch.equal(a["depth"], b["depth"])
            f = j.frames[-1]
            ft = tr.at(f)
            mu, q = posed(g._xyz, rot, g.moving, ft)
            s = ~g.moving
            assert torch.equal(mu[s], g._xyz[s]) and torch.equal(q[s], rot[s]), "static Gaussians moved"
            m = g.moving
            d0 = torch.cdist(g._xyz[m].double(), g._xyz[m].double())
            d1 = torch.cdist(mu[m].double(), mu[m].double())
            assert float((d0 - d1).abs().max()) < 1e-5, "moving part is not rigid"
            Q = torch.as_tensor(j.transform(f)[0], device=DEV, dtype=torch.float32)
            S = torch.diag_embed(torch.exp(g._scaling[m]) ** 2)
            R0, R1 = quat_to_rotmat(rot[m]), quat_to_rotmat(q[m])
            Sig0, Sig1 = R0 @ S @ R0.transpose(1, 2), R1 @ S @ R1.transpose(1, 2)
            assert float((Q @ Sig0 @ Q.T - Sig1).abs().max()) < 1e-6, "covariance did not rotate with Q"
            # backward mapping: processing row k is original frame anchor -/+ k
            k = 3
            assert j.frames[k] == j.anchor_frame + (k if d == "forward" else -k)
            assert j.q_of(j.frames[k]) == j.q[k]


def _pipeline_pose(up, mu0, R0, k):
    """The pipeline's own Phase-3 transport at processing row k (estimator states)."""
    from estimator import phase2_prismatic as mp
    from estimator.phase2_model_comparison import init_revolute, prismatic_state, revolute_state
    m = torch.as_tensor(up.members)
    q = torch.as_tensor(np.asarray(up.joint.q))
    if up.joint.type == "revolute":
        part = init_revolute(mu0, m, None, axis=torch.as_tensor(up.joint.axis), angles=q,
                             axis_point=torch.as_tensor(up.joint.point))
        st = revolute_state(part)
    else:
        part = mp.PrismaticPart(m, torch.as_tensor(up.joint.axis), q, torch.zeros_like(q))
        st = prismatic_state(part)
    return st.gaussians(k, mu0, R0)


def test_pose_equals_phase3_transport_on_real_runs():
    if not TC._available():
        print("SKIP: result disk absent")
        return
    from plyfile import PlyData
    cases = [("real", "red_fridge", "majority", "revolute"), ("real", "light_blue_box", "C", "prismatic"),
             ("synthetic", "USB_100109", None, "revolute"), ("synthetic", "blade_103706", None, "prismatic")]
    for kind, scene, arm, jt in cases:
        p, d = TC.run_paths(kind, scene, arm)
        up = load_upstream(**p, joint_type=jt)
        v = PlyData.read(up.ply)["vertex"]
        mu0 = torch.as_tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float64)
        qv = torch.as_tensor(np.stack([v[f"rot_{i}"] for i in range(4)], 1), dtype=torch.float64)
        qv = torch.nn.functional.normalize(qv, dim=1)
        R0 = quat_to_rotmat(qv)
        moving = torch.zeros(len(mu0), dtype=torch.bool)
        moving[torch.as_tensor(up.members)] = True
        tr = JointTransport(up.joint, "cpu", torch.float64)
        k = int(np.argmax(np.abs(up.joint.q)))
        mu_p, R_p = _pipeline_pose(up, mu0, R0, k)
        mu_m, q_m = posed(mu0, qv, moving, tr.at(up.joint.frames[k]))
        e_mu = float((mu_m - mu_p).abs().max())
        e_R = float((quat_to_rotmat(q_m) - R_p).abs().max())
        assert e_mu < 1e-8 and e_R < 1e-8, f"{scene} ({d}, {jt}) row {k}: |dmu| {e_mu:.2e} |dR| {e_R:.2e}"
        assert float((mu_m - mu0).abs().max()) > 1e-2, "the test frame must actually move the part"


def test_joint_gradients_positions_quaternions_fd():
    torch.manual_seed(0)
    N = 12
    xyz = torch.randn(N, 3, dtype=torch.float64)
    rot = torch.nn.functional.normalize(torch.randn(N, 4, dtype=torch.float64), dim=1)
    moving = torch.arange(N) < 7
    wm, wq = torch.randn(N, 3, dtype=torch.float64), torch.randn(N, 4, dtype=torch.float64)
    for jt in ("revolute", "prismatic"):
        axis = torch.tensor([0.2, -0.4, 0.9], dtype=torch.float64, requires_grad=True)
        point = torch.tensor([0.3, 0.1, -0.2], dtype=torch.float64, requires_grad=True)
        q = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)

        def L(axis, point, q, only_static=False):
            mu, qq = posed(xyz, rot, moving, joint_transform_torch(jt, axis, point, q))
            sel = ~moving if only_static else torch.ones(N, dtype=torch.bool)
            return (wm[sel] * mu[sel]).sum() + (wq[sel] * qq[sel]).sum()

        L(axis, point, q).backward()
        for name, t in (("axis", axis), ("point", point), ("q", q)):
            flat = t.detach().reshape(-1)
            for i in range(flat.numel()):
                args = [axis.detach().clone(), point.detach().clone(), q.detach().clone()]
                idx = ("axis", "point", "q").index(name)
                ap, am = [a.clone() for a in args], [a.clone() for a in args]
                ap[idx].reshape(-1)[i] += 1e-6
                am[idx].reshape(-1)[i] -= 1e-6
                fd = (float(L(*ap)) - float(L(*am))) / 2e-6
                an = 0.0 if t.grad is None else float(t.grad.reshape(-1)[i])
                assert abs(fd - an) < 1e-6 * max(1, abs(fd)), f"{jt} d/d{name}[{i}]: fd {fd} autograd {an}"
        for t in (axis, point, q):
            t.grad = None
        L(axis, point, q, only_static=True).backward()
        for t in (axis, point, q):
            assert t.grad is None or float(t.grad.abs().max()) == 0.0, "static Gaussians depend on the joint"
    # rendered image: CUDA joint gradient == oracle joint gradient (revolute, q and axis point)
    g = ToyGaussians(N=30)
    cam = toy_camera()
    w = torch.randn(3, cam.H, cam.W, dtype=torch.float64)
    grads = []
    for backend in ("cuda", "oracle"):
        dev = "cuda" if backend == "cuda" else "cpu"
        dt = torch.float32 if backend == "cuda" else torch.float64
        axis = torch.tensor([0.1, -0.3, 0.95], dtype=dt, device=dev, requires_grad=True)
        point = torch.tensor([0.05, 0.0, 2.3], dtype=dt, device=dev, requires_grad=True)
        q = torch.tensor(0.4, dtype=dt, device=dev, requires_grad=True)
        ft = joint_transform_torch("revolute", axis, point, q)
        X = g._xyz.to(dev, dt)
        R = torch.nn.functional.normalize(g._rotation.to(dev, dt), dim=1)
        mu, qq = posed(X, R, g.moving.to(dev), ft)
        sc, op = torch.exp(g._scaling.to(dev, dt)), torch.sigmoid(g._opacity.to(dev, dt))
        cols = torch.sigmoid(g._features_dc[:, 0].to(dev, dt))
        if backend == "cuda":
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
            s = GaussianRasterizationSettings(cam.H, cam.W, cam.tanfovx, cam.tanfovy, torch.ones(3, device=dev), 1.0,
                                              cam.V.to(dev), cam.P.to(dev), 0, cam.campos.to(dev), False, False, False)
            color = GaussianRasterizer(s).forward_rgbd(mu, torch.zeros_like(mu, requires_grad=True), op,
                                                       colors_precomp=cols, scales=sc, rotations=qq)[0]
        else:
            color = O.render(mu, sc, qq, op, cam.V.double(), cam.P.double(), cam.campos.double(), cam.W, cam.H,
                             cam.tanfovx, cam.tanfovy, torch.ones(3, dtype=torch.float64), colors=cols)["color"]
        (w.to(dev, color.dtype) * color).sum().backward()
        grads.append(torch.cat([axis.grad.double().cpu(), point.grad.double().cpu(), q.grad.double().cpu().reshape(1)]))
    e = float((grads[0] - grads[1]).norm() / grads[1].norm())
    assert e < 1e-3, f"rendered joint gradient CUDA vs oracle: relative error {e:.2e}"


def test_phase45_transport_registers_no_joint_parameter():
    tr = JointTransport(toy_joint(), DEV)
    for f in tr.joint.frames:
        ft = tr.at(f)
        assert not (ft.Q.requires_grad or ft.p.requires_grad or ft.qQ.requires_grad)


def test_local_sh_matches_world_at_anchor_and_rotates_with_part():
    j = toy_joint("revolute")
    g = ToyGaussians().to(DEV)
    tr = JointTransport(j, DEV)
    cam = toy_camera()
    bg = torch.ones(3, device=DEV)
    a = render_articulated(cam, tr.at(j.anchor_frame), g, g.moving, bg, "world_SH")
    b = render_articulated(cam, tr.at(j.anchor_frame), g, g.moving, bg, "local_SH")
    assert float((a["color"] - b["color"]).abs().max()) < 2e-5
    # local_SH: a moving Gaussian seen at frame f from C has the colour its canonical self shows to Q^T (C - p)
    f = j.frames[-1]
    Q, p = j.transform(f)
    C = cam.campos.double().numpy()
    sh = torch.cat([g._features_dc, g._features_rest], 1).double().cpu()
    m = g.moving.cpu()
    c_local = O.sh_to_rgb(g._xyz.double().cpu()[m], torch.as_tensor(Q.T @ (C - p)), sh[m], 3)
    mu_f = g._xyz.double().cpu()[m] @ torch.as_tensor(Q).T + torch.as_tensor(p)
    c_world = O.sh_to_rgb(mu_f, torch.as_tensor(C), sh[m], 3)
    c_canon = O.sh_to_rgb(g._xyz.double().cpu()[m], torch.as_tensor(C), sh[m], 3)
    assert float((c_local - c_canon).abs().max()) > 1e-3, "the part's rotation must change its local view"
    assert float((c_local - c_world).abs().max()) > 1e-3, "world SH ignores the part's rotation"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
