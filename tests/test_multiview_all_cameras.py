"""CODEX addendum §17/§24: true all-camera multiview inner optimization.

The contract: C camera renders -> one aggregated loss -> one backward -> one
optimizer step, with physics applied once rather than once per camera.
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
from rigid_parts.killing_inner_solver import multiview_render_loss_for_frame

DT = torch.float64


# --------------------------------------------------------------------------
# A fake camera/renderer so the contract can be tested without the rasteriser.
# --------------------------------------------------------------------------
class _FakeCamera:
    def __init__(self, target, index, alpha_mask=None):
        self.original_image = target
        self.index = index
        self.alpha_mask = alpha_mask
        self.depth_reliable = False


class _Recorder:
    """Stands in for gaussian_renderer.render and counts the calls."""

    def __init__(self, targets, radii=None):
        self.targets = targets
        self.calls = 0
        self.radii = radii

    def __call__(self, camera, gaussians, pipe, background, **kwargs):
        idx = camera.index
        self.calls += 1
        positions = kwargs["means3D_override"]
        # A differentiable, camera-specific image so each view has its own gradient.
        image = (positions.mean() * (idx + 1.0)).expand(3, 4, 4)
        radii = self.radii if self.radii is not None else torch.ones(int(positions.shape[0]))
        return {"render": image, "depth": image[:1], "radii": radii,
                "viewspace_points": positions, "visibility_filter": None}


def _args(**over):
    from types import SimpleNamespace

    base = dict(lambda_inner_rgb=1.0, lambda_inner_ssim=0.0, lambda_inner_depth=0.0,
                dynamic_use_depth=False, multiview_visibility_floor=0.1,
                killing_camera_chunk=0)
    base.update(over)
    return SimpleNamespace(**base)


def _setup(n_cameras=8, n_points=16):
    targets = [torch.full((3, 4, 4), 0.1 * (i + 1), dtype=DT) for i in range(n_cameras)]
    cameras = [_FakeCamera(t, i) for i, t in enumerate(targets)]
    recorder = _Recorder(targets)
    positions = torch.rand((n_points, 3), dtype=DT, requires_grad=True)
    rotations = torch.zeros((n_points, 4), dtype=DT)
    rotations[:, 0] = 1.0
    return cameras, recorder, positions, rotations


def _run(recorder, cameras, positions, rotations, *, args=None, **kw):
    original = killing_inner_solver.render_gaussian
    killing_inner_solver.render_gaussian = recorder
    try:
        return multiview_render_loss_for_frame(
            cameras=cameras, gaussians=None, pipe=None, background=None,
            dataset=None, args=args or _args(), positions=positions, rotations=rotations, **kw
        )
    finally:
        killing_inner_solver.render_gaussian = original


# --------------------------------------------------------------------------
# §17.1 / §17.2 / §17.3
# --------------------------------------------------------------------------
def test_one_inner_step_sees_all_cameras():
    cameras, recorder, positions, rotations = _setup(8)
    mv = _run(recorder, cameras, positions, rotations, lambda_view_var=0.0)
    assert recorder.calls == 8, f"rendered {recorder.calls} of 8 cameras"
    assert mv["num_views"] == 8
    assert len(mv["per_camera"]) == 8
    assert [c["camera_id"] for c in mv["per_camera"]] == list(range(8))


def test_gradients_from_all_cameras_are_present_before_the_step():
    cameras, recorder, positions, rotations = _setup(8)
    mv = _run(recorder, cameras, positions, rotations, lambda_view_var=0.0)
    mv["loss"].backward()
    assert positions.grad is not None and float(positions.grad.abs().max()) > 0.0

    # Every camera individually contributes: removing any one changes the gradient.
    full = positions.grad.detach().clone()
    for drop in range(8):
        subset = [c for i, c in enumerate(cameras) if i != drop]
        p2 = positions.detach().clone().requires_grad_(True)
        mv2 = _run(_Recorder(recorder.targets), subset, p2, rotations, lambda_view_var=0.0)
        mv2["loss"].backward()
        assert float((p2.grad - full).abs().max()) > 1e-12, f"camera {drop} contributed nothing"


# --------------------------------------------------------------------------
# §17.4 the gradient equals that of the mean camera loss
# --------------------------------------------------------------------------
def test_aggregate_gradient_equals_gradient_of_the_mean_camera_loss():
    cameras, recorder, positions, rotations = _setup(2)
    mv = _run(recorder, cameras, positions, rotations, lambda_view_var=0.0)
    mv["loss"].backward()
    aggregate = positions.grad.detach().clone()

    # Reference: grad of 0.5 * (L_1 + L_2), computed one camera at a time.
    per_cam_grads = []
    for cam in cameras:
        p = positions.detach().clone().requires_grad_(True)
        single = _run(_Recorder(recorder.targets), [cam], p, rotations, lambda_view_var=0.0)
        single["loss"].backward()
        per_cam_grads.append(p.grad.detach().clone())
    reference = 0.5 * (per_cam_grads[0] + per_cam_grads[1])

    assert torch.allclose(aggregate, reference, atol=1e-12), (
        f"max diff {float((aggregate - reference).abs().max()):.3e}"
    )


def test_physics_is_added_once_not_once_per_camera():
    """§7: L_inner = mean(render) + physics, never sum_c(render_c + physics)."""
    cameras, recorder, positions, rotations = _setup(8)
    mv = _run(recorder, cameras, positions, rotations, lambda_view_var=0.0)
    physics = (positions * 2.0).sum()
    total = mv["loss"] + physics
    total.backward()
    got = positions.grad.detach().clone()

    p2 = positions.detach().clone().requires_grad_(True)
    mv2 = _run(_Recorder(recorder.targets), cameras, p2, rotations, lambda_view_var=0.0)
    mv2["loss"].backward()
    render_only = p2.grad.detach().clone()

    physics_component = got - render_only
    expected = torch.full_like(physics_component, 2.0)          # d(2*sum x)/dx = 2
    assert torch.allclose(physics_component, expected, atol=1e-10), (
        f"physics was scaled by {float(physics_component.mean()) / 2.0:.2f}x the camera count"
    )


# --------------------------------------------------------------------------
# §9 / §10 aggregate statistics and the consistency regularizer
# --------------------------------------------------------------------------
def test_multiview_statistics_are_reported():
    cameras, recorder, positions, rotations = _setup(8)
    mv = _run(recorder, cameras, positions, rotations, lambda_view_var=0.0)
    values = torch.tensor([c["total_render"] for c in mv["per_camera"]], dtype=DT)
    assert abs(mv["mean"] - float(values.mean())) < 1e-12
    assert abs(mv["std"] - float(values.std(unbiased=False))) < 1e-12
    assert abs(mv["min"] - float(values.min())) < 1e-12
    assert abs(mv["max"] - float(values.max())) < 1e-12
    assert mv["num_views"] == 8
    assert mv["std"] > 0.0, "the fake cameras should disagree, so std must be non-zero"


def test_view_variance_term_penalises_uneven_views():
    cameras, recorder, positions, rotations = _setup(8)
    plain = _run(recorder, cameras, positions, rotations, lambda_view_var=0.0)
    regularised = _run(_Recorder(recorder.targets), cameras, positions, rotations, lambda_view_var=0.05)
    assert float(regularised["loss"]) > float(plain["loss"]), "std term had no effect"
    expected = float(plain["loss"]) + 0.05 * plain["std"]
    assert abs(float(regularised["loss"]) - expected) < 1e-9
    assert regularised["std_in_loss"] is True and plain["std_in_loss"] is False


# --------------------------------------------------------------------------
# §12 visibility weighting
# --------------------------------------------------------------------------
def test_invalid_view_is_downweighted_but_never_dropped():
    n_points = 16
    cameras, recorder, positions, rotations = _setup(4, n_points)
    moving = torch.zeros(n_points, dtype=torch.bool)
    moving[:8] = True

    # Camera 0 sees none of the moving support.
    blind = torch.ones(n_points)
    blind[:8] = 0.0
    original = killing_inner_solver.render_gaussian

    class _Mixed(_Recorder):
        def __call__(self, camera, *a, **kw):
            idx = camera.index
            self.radii = blind if idx == 0 else torch.ones(n_points)
            return super().__call__(camera, *a, **kw)

    killing_inner_solver.render_gaussian = _Mixed(recorder.targets)
    try:
        mv = multiview_render_loss_for_frame(
            cameras=cameras, gaussians=None, pipe=None, background=None, dataset=None,
            args=_args(), positions=positions, rotations=rotations,
            moving_mask=moving, lambda_view_var=0.0,
        )
    finally:
        killing_inner_solver.render_gaussian = original

    vis = [c["visibility"] for c in mv["per_camera"]]
    assert vis[0] == 0.1, f"blind view should hit the floor, got {vis[0]}"
    assert all(v == 1.0 for v in vis[1:]), vis
    assert mv["num_views"] == 4, "an invalid view must be downweighted, not dropped"
    assert len(mv["per_camera"]) == 4


# --------------------------------------------------------------------------
# §3 / §4 / §24 structural guarantees in the solver
# --------------------------------------------------------------------------
def test_solver_takes_exactly_one_optimizer_step_per_inner_iteration():
    source = inspect.getsource(killing_inner_solver.solve_nested_killing_sequence)
    step_lines = [l for l in source.splitlines() if "optimizer.step()" in l and not l.strip().startswith("#")]
    assert len(step_lines) == 1, f"expected one optimizer.step() call site, found {len(step_lines)}"
    assert "optimizer_steps_this_inner_iteration = 1" in source

    # The §4 forbidden pattern: zero_grad inside the camera loop.
    render_block = source[source.index("ALL CAMERAS BEFORE THE STEP"):source.index("optimizer.step()")]
    assert "zero_grad" not in render_block, "optimizer is reset inside the camera loop"
    assert "multiview_render_loss_for_frame" in render_block


def test_helper_never_steps_the_optimizer():
    """§22: the helper computes the loss and diagnostics only."""
    source = inspect.getsource(multiview_render_loss_for_frame)
    # Strip the docstring: it explains the prohibition, it does not perform it.
    body = source.split('"""', 2)[-1]
    assert "optimizer.step()" not in body, "the helper steps the optimizer"
    assert "zero_grad" not in body
    for key in ("loss", "mean", "std", "min", "max", "num_views", "per_camera"):
        assert f'"{key}"' in source, f"helper does not return {key}"


def test_render_views_assertion_and_per_camera_logging_exist():
    source = inspect.getsource(killing_inner_solver.solve_nested_killing_sequence)
    assert "multiview contract violated" in source, "§18 render_views assertion missing"
    for key in ("render_rgb_cam_", "render_ssim_cam_", "render_multiview_mean",
                "render_multiview_std", "render_multiview_min", "render_multiview_max",
                "render_multiview_num_cameras", "optimizer_steps_this_inner_iteration"):
        assert key in source, f"§8/§9/§18 diagnostic missing: {key}"
    assert "inner_camera_history.csv" in source, "§20 per-camera CSV missing"


def test_hard_partition_and_twist_are_shared_across_cameras():
    """§13/§14: the scene and the twist are built once, outside the camera loop."""
    source = inspect.getsource(killing_inner_solver.solve_nested_killing_sequence)
    scene_at = source.index("_compose_hard_partition_scene(xyz0, rot0, trial_xyz")
    cameras_at = source.index("ALL CAMERAS BEFORE THE STEP")
    twist_at = source.index("twist = analytic_partwise_twist")
    assert twist_at < scene_at < cameras_at, (
        "the twist/scene must be computed before the camera loop, not per camera"
    )


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
    print(f"PASS multiview_all_cameras ({len(tests)} tests)")
