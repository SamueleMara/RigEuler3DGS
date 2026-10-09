"""FIRM v4 §22 non-negotiable contract.

v3's error: per-Gaussian xi_ik integrated independently, i.e. a field of
independent rigid transforms rather than one rigid body. These tests pin the
corrected contract: the velocity field stays Gaussian-by-Gaussian, but the
transported twist is ONE per part, fitted analytically, with no Kabsch.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts import killing_inner_solver
from rigid_parts.local_killing import (
    analytic_partwise_twist,
    local_killing_quantities,
    partwise_motion_separation,
)
from rigid_parts.neighborhood import build_neighbor_graph, edge_part_weights
from utils.rigid_transform_utils import se3_exp_spatial

DT = torch.float64


def _rigid(points, omega, b):
    return torch.cross(omega.expand_as(points), points, dim=-1) + b


# --------------------------------------------------------------------------
# §6-§8 the analytic fit
# --------------------------------------------------------------------------
def test_analytic_twist_recovers_a_rigid_field_exactly():
    torch.manual_seed(0)
    n = 300
    x = (torch.rand((1, n, 3), dtype=DT) * 2 - 1)
    omega = torch.tensor([0.4, -0.3, 0.9], dtype=DT)
    b = torch.tensor([0.2, 0.5, -0.1], dtype=DT)
    v = _rigid(x[0], omega, b)[None]
    prob = torch.ones((n, 1), dtype=DT)

    q = analytic_partwise_twist(x, v, prob, reg=0.0)
    assert float((q["omega"][0] - omega).abs().max()) < 1e-14, float((q["omega"][0] - omega).abs().max())
    assert float((q["b"][0] - b).abs().max()) < 1e-14
    assert float(q["rigid_speed_residual"]) < 1e-28

    # b_k = u_k - omega_k x c_k exactly (§8).
    expect_b = q["center_velocity"][0] - torch.cross(q["omega"][0], q["center"][0], dim=-1)
    assert torch.allclose(q["b"][0], expect_b, atol=1e-14)

    # H_k omega_k = g_k is the normal equation actually solved (§7).
    residual = q["H"][0] @ q["omega"][0] - q["g"][0]
    assert float(residual.abs().max()) < 1e-10, float(residual.abs().max())


def test_analytic_twist_is_a_least_squares_projection_not_a_mean():
    """The fit must beat the naive per-Gaussian mean on a non-rigid field."""
    torch.manual_seed(1)
    n = 400
    x = (torch.rand((1, n, 3), dtype=DT) * 2 - 1)
    omega = torch.tensor([0.0, 0.0, 0.7], dtype=DT)
    b = torch.tensor([0.1, 0.0, 0.0], dtype=DT)
    v = _rigid(x[0], omega, b)[None] + 0.05 * torch.randn((1, n, 3), dtype=DT)
    prob = torch.ones((n, 1), dtype=DT)
    q = analytic_partwise_twist(x, v, prob, reg=0.0)

    graph = build_neighbor_graph(x[0], k=12)
    local = local_killing_quantities(
        x, torch.rand((1, 40, 3), dtype=DT), torch.randn((1, 40, 3), dtype=DT),
        torch.full((1,), 0.4, dtype=DT), prob, graph, edge_part_weights(graph, prob),
    )
    # Fit error of the analytic twist vs. of the membership-weighted local mean.
    def fit_error(w, bb):
        pred = torch.cross(w.expand_as(x[0]), x[0], dim=-1) + bb
        return float((v[0] - pred).square().sum(-1).mean())

    assert fit_error(q["omega"][0], q["b"][0]) < fit_error(local["omega_mean"][0], local["b_mean"][0])
    assert float((q["omega"][0] - omega).abs().max()) < 0.05


def test_update_weights_select_which_velocities_build_each_twist():
    """§6: a_ik = p_ik u_i decides which Gaussian velocities feed part k."""
    torch.manual_seed(2)
    n = 300
    x = (torch.rand((1, n, 3), dtype=DT) * 2 - 1).expand(2, -1, -1).contiguous()
    w_a = torch.tensor([0.4, -0.3, 0.9], dtype=DT)
    w_b = torch.tensor([0.0, 0.0, -0.7], dtype=DT)
    b = torch.tensor([0.2, 0.5, -0.1], dtype=DT)
    v = torch.stack((_rigid(x[0], w_a, b), _rigid(x[0], w_b, b)))

    prob = torch.zeros((n, 2), dtype=DT)
    prob[: n // 2, 0] = 1.0
    prob[n // 2 :, 1] = 1.0
    q = analytic_partwise_twist(x, v, prob, reg=0.0)
    assert float((q["omega"][0] - w_a).abs().max()) < 1e-12
    assert float((q["omega"][1] - w_b).abs().max()) < 1e-12

    # A zero update weight must remove a Gaussian from the fit entirely.
    update = torch.ones(n, dtype=DT)
    update[: n // 4] = 0.0
    q2 = analytic_partwise_twist(x, v, prob, update, reg=0.0)
    assert float((q2["omega"][0] - w_a).abs().max()) < 1e-12  # still rigid, so unchanged
    assert float(q2["effective_weight"][0]) < float(q["effective_weight"][0])


def test_analytic_twist_is_differentiable_end_to_end():
    """§13: nothing on render -> T_k -> omega_k,b_k -> v_i is detached."""
    torch.manual_seed(3)
    n = 120
    # Make BOTH inputs differentiable so an explicit .detach() anywhere on the
    # c_k / u_k / H_k / g_k / omega_k / b_k path would show up as a missing grad.
    x = ((torch.rand((1, n, 3), dtype=DT) * 2 - 1)).requires_grad_(True)
    v = torch.randn((1, n, 3), dtype=DT, requires_grad=True)
    prob = torch.rand((n, 1), dtype=DT)
    q = analytic_partwise_twist(x, v, prob)
    for key in ("center", "center_velocity", "omega", "b", "H", "g", "rigid_velocity"):
        assert q[key].requires_grad, f"§13 violated: {key} was detached"

    loss = q["omega"].sum() + q["b"].sum() + q["rigid_speed_residual"]
    grad_v, grad_x = torch.autograd.grad(loss, (v, x))
    assert torch.isfinite(grad_v).all() and float(grad_v.abs().max()) > 0.0, "no gradient to velocities"
    assert torch.isfinite(grad_x).all() and float(grad_x.abs().max()) > 0.0, "no gradient to positions"

    # Conditioning diagnostics are the one thing that SHOULD be detached.
    for key in ("effective_weight", "effective_num_points", "eigenvalues", "condition_number"):
        assert not q[key].requires_grad, f"{key} should be a detached diagnostic"


def test_rigid_speed_residual_is_zero_only_for_rigid_fields():
    torch.manual_seed(4)
    n = 250
    x = (torch.rand((1, n, 3), dtype=DT) * 2 - 1)
    prob = torch.ones((n, 1), dtype=DT)
    rigid = _rigid(x[0], torch.tensor([0.3, 0.1, -0.6], dtype=DT), torch.tensor([0.2, 0.0, 0.1], dtype=DT))[None]
    assert float(analytic_partwise_twist(x, rigid, prob, reg=0.0)["rigid_speed_residual"]) < 1e-28

    stretch = x * torch.tensor([0.5, 0.0, 0.0], dtype=DT)  # pure strain, not rigid
    assert float(analytic_partwise_twist(x, stretch, prob, reg=0.0)["rigid_speed_residual"]) > 1e-3


# --------------------------------------------------------------------------
# §11, §12, §22 partwise transport
# --------------------------------------------------------------------------
def test_all_gaussians_of_a_part_share_one_transform():
    """§12/§22: candidate k transport uses the SAME Delta T_k for every Gaussian."""
    torch.manual_seed(5)
    n = 200
    xyz0 = torch.rand((n, 3), dtype=DT) * 2 - 1
    rot0 = torch.zeros((n, 4), dtype=DT)
    rot0[:, 0] = 1.0
    part_r = torch.stack([torch.linalg.qr(torch.randn((3, 3), dtype=DT))[0] for _ in range(2)])
    part_t = torch.randn((2, 3), dtype=DT)
    part_q = torch.nn.functional.normalize(torch.randn((2, 4), dtype=DT), dim=-1)

    xyz, quat = killing_inner_solver._partwise_candidate_state(xyz0, rot0, part_r, part_t, part_q)
    for k in range(2):
        expect = xyz0 @ part_r[k].transpose(0, 1) + part_t[k]
        assert torch.allclose(xyz[k], expect, atol=1e-12), f"part {k} is not one shared transform"

    # A shared transform preserves every pairwise distance exactly.
    def pairwise(a):
        return (a[:, None, :] - a[None, :, :]).norm(dim=-1)

    for k in range(2):
        assert float((pairwise(xyz[k]) - pairwise(xyz0)).abs().max()) < 1e-12


def test_partwise_integration_is_exact_se3_and_left_composed():
    """§11: Delta T_k = Exp(xi_k dt), T_k(t+1) = Delta T_k T_k(t)."""
    torch.manual_seed(6)
    frames = 30
    omega = torch.randn((frames, 2, 3), dtype=DT) * 0.5
    b = torch.randn((frames, 2, 3), dtype=DT) * 0.3
    dts = 0.02 + 0.05 * torch.rand(frames, dtype=DT)

    part_r = torch.eye(3, dtype=DT)[None].expand(2, -1, -1).clone()
    part_t = torch.zeros((2, 3), dtype=DT)
    reference = torch.eye(4, dtype=DT)[None].expand(2, -1, -1).clone()
    for f in range(frames):
        r_inc, p_inc, _ = se3_exp_spatial(omega[f], b[f], dts[f])
        part_r = r_inc @ part_r
        part_t = (r_inc @ part_t[..., None])[..., 0] + p_inc
        delta = torch.eye(4, dtype=DT)[None].expand(2, -1, -1).clone()
        delta[:, :3, :3] = r_inc
        delta[:, :3, 3] = p_inc
        reference = delta @ reference
    assert float((part_r - reference[:, :3, :3]).abs().max()) < 1e-12
    assert float((part_t - reference[:, :3, 3]).abs().max()) < 1e-12


def test_local_field_still_drives_the_partwise_twist():
    """The whole chain: velocity field -> analytic twist -> SE(3) -> positions."""
    torch.manual_seed(7)
    n = 200
    xyz0 = torch.rand((n, 3), dtype=DT) * 2 - 1
    centers = torch.rand((1, 48, 3), dtype=DT) * 2 - 1
    omega = torch.tensor([0.0, 0.0, 0.8], dtype=DT)
    b = torch.tensor([0.3, 0.0, 0.0], dtype=DT)
    control = torch.nn.Parameter(_rigid(centers[0], omega, b)[None].clone())
    sigma = torch.full((1,), 0.5, dtype=DT)
    prob = torch.ones((n, 1), dtype=DT)
    graph = build_neighbor_graph(xyz0, k=12)

    q = local_killing_quantities(xyz0[None], centers, control, sigma, prob, graph, edge_part_weights(graph, prob))
    twist = analytic_partwise_twist(xyz0[None], q["velocity"], prob, reg=0.0)
    r_inc, p_inc, _ = se3_exp_spatial(twist["omega"], twist["b"], torch.tensor(0.1, dtype=DT))
    moved = xyz0 @ r_inc[0].transpose(0, 1) + p_inc[0]

    assert float((moved - xyz0).norm(dim=-1).mean()) > 1e-3, "the part did not move"
    grad = torch.autograd.grad(moved.sum(), control)[0]
    assert torch.isfinite(grad).all() and float(grad.abs().max()) > 0.0, (
        "§13 gradient does not reach the velocity-field parameters"
    )
    cos = torch.nn.functional.cosine_similarity(twist["omega"][0], omega, dim=0)
    assert float(cos) > 0.95, f"partwise omega lost the field's rotation axis (cos {float(cos):.3f})"


# --------------------------------------------------------------------------
# §19 the v3 behaviour that must be gone
# --------------------------------------------------------------------------
def test_no_per_gaussian_se3_transport_remains_in_the_solver():
    """§5/§19: per-Gaussian Exp_SE3 transport must not drive the motion."""
    source = inspect.getsource(killing_inner_solver.solve_nested_killing_sequence)
    assert "integrate_local_se3" not in source, (
        "solver still transports Gaussians with per-Gaussian SE(3) increments"
    )
    assert "analytic_partwise_twist" in source
    assert "_partwise_candidate_state" in source

    # And no Kabsch CALL on the motion path (§7, §18) -- mentions in comments
    # explaining the prohibition are fine, invocations are not.
    import re

    calls = re.findall(r"\b\w*[Kk]absch\w*\s*\(", source)
    assert not calls, f"Kabsch invoked on the motion path: {calls}"
    assert "rigid_fit" not in source


def test_local_strain_and_rigid_speed_are_both_present_and_distinct():
    """§10: the two rigidity terms have different roles; neither replaces the other."""
    source = inspect.getsource(killing_inner_solver.solve_nested_killing_sequence)
    assert "strain_loss" in source and "rigid_speed" in source
    assert "lambda_inner_strain" in source and "lambda_inner_rigid_speed" in source

    # They genuinely measure different things: a field can be locally
    # strain-free per-neighbourhood yet disagree with ONE partwise twist.
    torch.manual_seed(8)
    n = 300
    x = torch.rand((1, n, 3), dtype=DT) * 2 - 1
    prob = torch.ones((n, 1), dtype=DT)
    # Two rigid halves with different twists: locally rigid nearly everywhere,
    # but no single partwise twist explains both.
    left = x[0, :, 0] < 0
    v = torch.where(
        left[:, None],
        _rigid(x[0], torch.tensor([0.0, 0.0, 0.9], dtype=DT), torch.zeros(3, dtype=DT)),
        _rigid(x[0], torch.tensor([0.0, 0.0, -0.9], dtype=DT), torch.zeros(3, dtype=DT)),
    )[None]
    twist = analytic_partwise_twist(x, v, prob, reg=0.0)
    assert float(twist["rigid_speed_residual"]) > 1e-2, (
        "partwise residual failed to see two incompatible rigid motions"
    )


# --------------------------------------------------------------------------
# §16 separation, §21 degeneracy
# --------------------------------------------------------------------------
def test_separation_acts_on_partwise_twists():
    w = torch.stack([torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]], dtype=DT)] * 5)
    u = torch.zeros_like(w)
    far, d_far = partwise_motion_separation(w, u, margin=0.1)
    collapsed = w.clone()
    collapsed[:, 1] = collapsed[:, 0]
    near, d_near = partwise_motion_separation(collapsed, u, margin=0.1)
    assert float(far) == 0.0 and float(near) > 0.0
    assert float(d_near[0, 1]) == 0.0 < float(d_far[0, 1])
    assert float(partwise_motion_separation(w[:, :1], u[:, :1], margin=0.1)[0]) == 0.0


def test_degenerate_support_cannot_produce_a_giant_angular_velocity():
    """§21: collinear support is regularised, flagged, and stays bounded."""
    torch.manual_seed(9)
    n = 60
    x = torch.zeros((1, n, 3), dtype=DT)
    x[0, :, 0] = torch.linspace(-1.0, 1.0, n, dtype=DT)
    omega = torch.tensor([0.4, -0.3, 0.9], dtype=DT)
    v = _rigid(x[0], omega, torch.tensor([0.2, 0.0, 0.1], dtype=DT))[None]
    prob = torch.ones((n, 1), dtype=DT)

    q = analytic_partwise_twist(x, v, prob, reg=1.0e-8)
    assert torch.isfinite(q["omega"]).all()
    assert float(q["omega"].norm()) < 100.0, f"degenerate support exploded: {float(q['omega'].norm())}"
    assert float(q["condition_number"][0]) > 1e6, "degeneracy not visible in the conditioning"
    for key in ("effective_weight", "effective_num_points", "eigenvalues", "condition_number"):
        assert key in q, f"§21 diagnostic missing: {key}"

    # A well-conditioned support must look clearly different.
    x_full = torch.rand((1, n, 3), dtype=DT) * 2 - 1
    v_full = _rigid(x_full[0], omega, torch.zeros(3, dtype=DT))[None]
    good = analytic_partwise_twist(x_full, v_full, prob, reg=1.0e-8)
    assert float(good["condition_number"][0]) < float(q["condition_number"][0]) / 100.0


def test_twist_solve_is_closed_form_and_avoids_cusolver():
    """The 3x3 twist solve must not go through torch.linalg.solve/cuSOLVER.

    A stalled v4 run traced to cuSOLVER on small batched systems; this repo has
    hit that before. The adjugate solve is exact, differentiable and sync-free.
    """
    from rigid_parts import local_killing

    source = inspect.getsource(local_killing.analytic_partwise_twist)
    assert "torch.linalg.solve" not in source, "twist path reintroduced torch.linalg.solve"
    assert "_solve_symmetric_3x3" in source
    # eigvalsh is diagnostics only and must stay off the GPU path.
    eig = inspect.getsource(local_killing._symmetric_3x3_eigenvalues)
    assert ".cpu()" in eig

    # Exactness against the reference solver on well-conditioned systems.
    torch.manual_seed(11)
    m = torch.randn((9, 3, 3), dtype=DT)
    h = m @ m.transpose(-1, -2) + 3.0 * torch.eye(3, dtype=DT)
    g = torch.randn((9, 3), dtype=DT)
    mine = local_killing._solve_symmetric_3x3(h, g)
    ref = torch.linalg.solve(h, g[..., None])[..., 0]
    assert float((mine - ref).abs().max()) < 1e-10

    # Singular H must stay bounded rather than producing a giant omega (§21).
    singular = torch.zeros((1, 3, 3), dtype=DT)
    assert float(local_killing._solve_symmetric_3x3(singular, g[:1]).abs().max()) < 1e12


def test_fabricated_fps_timescale_is_not_used_as_dt():
    """dt scales Exp(xi dt), so a synthesised fps rescales every twist.

    With no timestamp table the dataset reader invents ``frame_id / fps`` from
    --fps (default 30). Using that as dt attenuated the render gradient ~30x and
    forced the optimiser to find velocities ~30x larger; the measured render
    share of the gradient went from 0.5% to 16.7% once dt became 1 per frame.
    """
    from types import SimpleNamespace

    from rigid_parts.nested_pipeline import _camera_groups, _has_real_timestamps

    class _Cam:
        def __init__(self, fid, ts):
            self.dynamic_frame_id = fid
            self.timestamp = ts

    class _Scene:
        def __init__(self, cams):
            self._c = cams

        def getTrainCameras(self):
            return self._c

    cams = [_Cam(f, f / 30.0) for f in range(10)]
    scene = _Scene(cams)

    # No timestamp file anywhere -> the 30 fps timescale is fabricated.
    args = SimpleNamespace(source_path="/nonexistent/scene", dynamic_timestamps_file="", nested_time_unit="auto")
    assert not _has_real_timestamps(args)
    _, times, _ = _camera_groups(scene, args)
    dt = float(times[1] - times[0])
    assert abs(dt - 1.0) < 1e-12, f"fabricated fps leaked into dt: {dt}"

    # An explicit request for seconds is still honoured.
    args_seconds = SimpleNamespace(source_path="/nonexistent/scene", dynamic_timestamps_file="", nested_time_unit="seconds")
    _, times_s, _ = _camera_groups(scene, args_seconds)
    assert abs(float(times_s[1] - times_s[0]) - 1.0 / 30.0) < 1e-6  # times are float32

    # A supplied timestamp table counts as measured, so seconds are used.
    args_real = SimpleNamespace(source_path="/nonexistent/scene", dynamic_timestamps_file="/tmp/ts.json", nested_time_unit="auto")
    assert _has_real_timestamps(args_real)
    _, times_r, _ = _camera_groups(scene, args_real)
    assert abs(float(times_r[1] - times_r[0]) - 1.0 / 30.0) < 1e-6


def test_part_transform_frames_align_with_position_frames():
    """part_rotation[f] must describe the SAME frame as positions[f].

    The commit block runs once per interval, so without an explicit identity
    frame these tensors carried F-1 entries against F frames of positions and
    every consumer indexing them was shifted by one frame.
    """
    torch.manual_seed(12)
    n, frames, k_moving = 40, 7, 2
    xyz0 = torch.rand((n, 3), dtype=DT)
    prob = torch.rand((n, k_moving + 1), dtype=DT)

    part_r, part_t, part_q = [], [], []
    r = torch.eye(3, dtype=DT)[None].expand(k_moving, -1, -1).clone()
    t = torch.zeros((k_moving, 3), dtype=DT)
    q = torch.zeros((k_moving, 4), dtype=DT)
    q[:, 0] = 1.0
    for _ in range(frames - 1):  # one commit per interval
        omega = torch.randn((k_moving, 3), dtype=DT) * 0.2
        b = torch.randn((k_moving, 3), dtype=DT) * 0.2
        r_inc, p_inc, q_inc = se3_exp_spatial(omega, b, torch.tensor(1.0, dtype=DT))
        r = r_inc @ r
        t = (r_inc @ t[..., None])[..., 0] + p_inc
        from utils.rigid_transform_utils import normalize_quaternion, quaternion_multiply

        q = normalize_quaternion(quaternion_multiply(q_inc, q))
        part_r.append(r.clone())
        part_t.append(t.clone())
        part_q.append(q.clone())

    rot, trans, quat = killing_inner_solver._stack_part_transforms(part_r, part_t, part_q, prob, xyz0)
    assert rot.shape[0] == frames, f"{rot.shape[0]} transform frames vs {frames} position frames"
    assert trans.shape[0] == frames and quat.shape[0] == frames
    assert rot.shape[1] == k_moving + 1, "static class missing from the part index"

    # Frame 0 must be the identity state for every part.
    assert torch.allclose(rot[0], torch.eye(3, dtype=DT).expand_as(rot[0]), atol=1e-12)
    assert float(trans[0].abs().max()) == 0.0
    # Static class stays identity at every frame.
    assert torch.allclose(rot[:, 0], torch.eye(3, dtype=DT).expand_as(rot[:, 0]), atol=1e-12)
    assert float(trans[:, 0].abs().max()) == 0.0
    # And the interval transforms land at the frames after the identity.
    assert torch.allclose(rot[1:, 1:], torch.stack(part_r), atol=1e-12)


if __name__ == "__main__":
    module = sys.modules["__main__"]
    tests = [
        (name, fn)
        for name, fn in sorted(vars(module).items())
        if name.startswith("test_") and inspect.isfunction(fn)
    ]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"PASS firm_v4_partwise_twist ({len(tests)} tests)")
