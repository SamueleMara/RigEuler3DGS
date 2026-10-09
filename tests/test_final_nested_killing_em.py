"""FINAL spec §49 mandatory regression tests (A-J).

The contract: softness lives in the OUTER belief, the INNER physical scene is one
discrete hypothesis per Gaussian, the rigid twist is partwise and analytic, and
no global gate may freeze the EM loop.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts import killing_inner_solver, nested_pipeline
from rigid_parts.embeddings import PartEmbeddingModel
from rigid_parts.local_killing import analytic_partwise_twist
from rigid_parts.neighborhood import build_neighbor_graph, edge_part_weights
from rigid_parts.outer_embedding_solver import (
    build_outer_targets,
    candidates_from_local_endpoints,
    geometry_guard_value,
    outer_embedding_losses,
)
from rigid_parts.static_init import diffuse_static_anchors, initial_part_logits

DT = torch.float64


def _identity_quats(n):
    q = torch.zeros((n, 4), dtype=DT)
    q[:, 0] = 1.0
    return q


# --------------------------------------------------------------------------
# A. No soft physical blending
# --------------------------------------------------------------------------
def test_A_no_soft_physical_blending_in_the_inner_scene():
    n = 5
    xyz0 = torch.rand((n, 3), dtype=DT)
    rot0 = _identity_quats(n)
    shift = torch.tensor([10.0, 0.0, 0.0], dtype=DT)
    moving_xyz = (xyz0 + shift)[None]
    moving_quat = rot0[None]

    # p_i = [0.51, 0.49] -> the moving hypothesis is nearly as likely...
    prob = torch.tensor([[0.51, 0.49]] * n, dtype=DT)
    z = prob.argmax(dim=-1)
    assert int(z.sum()) == 0, "hard label should be static here"

    xyz, quat = killing_inner_solver._compose_hard_partition_scene(
        xyz0, rot0, moving_xyz, moving_quat, z
    )
    # ...but the PHYSICAL position must be exactly canonical.
    assert torch.allclose(xyz, xyz0, atol=1e-15), "inner scene blended the hypotheses"
    soft, _ = killing_inner_solver._compose_soft_positions(xyz0, rot0, moving_xyz, moving_quat, prob)
    assert float((soft - xyz0).abs().max()) > 4.0, "test is vacuous: soft blend did not move"

    # Flip the winner: now the Gaussian sits exactly on the moving hypothesis.
    prob_moving = torch.tensor([[0.49, 0.51]] * n, dtype=DT)
    xyz_m, _ = killing_inner_solver._compose_hard_partition_scene(
        xyz0, rot0, moving_xyz, moving_quat, prob_moving.argmax(dim=-1)
    )
    assert torch.allclose(xyz_m, xyz0 + shift, atol=1e-15)


# --------------------------------------------------------------------------
# B. Twist support
# --------------------------------------------------------------------------
def test_B_twist_uses_only_its_current_hard_support():
    torch.manual_seed(1)
    n = 200
    x = (torch.rand((1, n, 3), dtype=DT) * 2 - 1)
    omega = torch.tensor([0.0, 0.0, 0.8], dtype=DT)
    b = torch.tensor([0.1, 0.0, 0.0], dtype=DT)

    z = torch.zeros(n, dtype=torch.long)
    z[: n // 2] = 1                      # first half is the moving support
    v = torch.zeros((1, n, 3), dtype=DT)
    v[0, : n // 2] = torch.cross(omega.expand(n // 2, 3), x[0, : n // 2], dim=-1) + b
    # The static half carries a completely different, contaminating velocity.
    v[0, n // 2 :] = torch.tensor([5.0, -5.0, 5.0], dtype=DT)

    # p_ik = 0.49 for the static half: high enough to contaminate if unmasked.
    prob = torch.full((n, 1), 0.49, dtype=DT)
    prob[: n // 2] = 0.9

    masked = analytic_partwise_twist(x, v, prob, None, z, reg=0.0)
    assert torch.allclose(masked["omega"][0], omega, atol=1e-10), (
        f"static Gaussians leaked into the twist: {masked['omega'][0]}"
    )
    assert torch.allclose(masked["b"][0], b, atol=1e-10)

    unmasked = analytic_partwise_twist(x, v, prob, None, None, reg=0.0)
    assert float((unmasked["omega"][0] - omega).abs().max()) > 0.1, (
        "test is vacuous: the contaminating half did not change the unmasked fit"
    )


def test_B2_edge_weights_are_zero_across_the_hard_support_boundary():
    torch.manual_seed(2)
    n = 120
    x = torch.rand((n, 3), dtype=DT)
    graph = build_neighbor_graph(x, k=10)
    prob = torch.full((n, 1), 0.6, dtype=DT)
    z = torch.zeros(n, dtype=torch.long)
    z[: n // 2] = 1

    w = edge_part_weights(graph, prob, z)
    crosses = (z[graph.source] == 1) != (z[graph.target] == 1)
    assert int(crosses.sum()) > 0, "no boundary-crossing edges to test"
    assert float(w[0][crosses].abs().max()) == 0.0, "static and moving regularise each other"
    inside = (z[graph.source] == 1) & (z[graph.target] == 1)
    assert float(w[0][inside].max()) > 0.0, "in-support edges were also zeroed"


# --------------------------------------------------------------------------
# C / D. Partwise transform
# --------------------------------------------------------------------------
def test_C_all_gaussians_in_a_part_receive_the_same_transform():
    torch.manual_seed(3)
    n = 150
    xyz0 = torch.rand((n, 3), dtype=DT) * 2 - 1
    rot0 = _identity_quats(n)
    part_r = torch.stack([torch.linalg.qr(torch.randn((3, 3), dtype=DT))[0] for _ in range(2)])
    part_t = torch.randn((2, 3), dtype=DT)
    part_q = torch.nn.functional.normalize(torch.randn((2, 4), dtype=DT), dim=-1)
    xyz, _ = killing_inner_solver._partwise_candidate_state(xyz0, rot0, part_r, part_t, part_q)
    for k in range(2):
        assert torch.allclose(xyz[k], xyz0 @ part_r[k].T + part_t[k], atol=1e-12)

    def pairwise(a):
        return (a[:, None, :] - a[None, :, :]).norm(dim=-1)

    for k in range(2):
        assert float((pairwise(xyz[k]) - pairwise(xyz0)).abs().max()) < 1e-12


def test_D_local_velocities_vary_but_the_transform_stays_partwise():
    torch.manual_seed(4)
    n = 200
    x = (torch.rand((1, n, 3), dtype=DT) * 2 - 1)
    # A deliberately non-rigid, spatially varying velocity field.
    v = (x * torch.tensor([1.0, -0.5, 0.3], dtype=DT)) + 0.2 * torch.randn((1, n, 3), dtype=DT)
    z = torch.ones(n, dtype=torch.long)
    prob = torch.ones((n, 1), dtype=DT)
    twist = analytic_partwise_twist(x, v, prob, None, z, reg=0.0)

    assert float(v.std()) > 0.1, "velocities do not actually vary"
    assert twist["omega"].shape == (1, 3), "more than one twist per part"
    assert twist["b"].shape == (1, 3)
    assert float(twist["rigid_speed_residual"]) > 0.0, "non-rigid field should leave a residual"


# --------------------------------------------------------------------------
# E. Initial spatial static decay
# --------------------------------------------------------------------------
def test_E_initial_static_logits_decay_away_from_anchors_but_stay_random():
    torch.manual_seed(5)
    n = 900
    x = torch.rand((n, 3), dtype=DT)
    graph = build_neighbor_graph(x, k=12)
    anchors = x[:, 0] < 0.08                     # a spatial patch of trusted anchors
    assert 10 < int(anchors.sum()) < n // 4

    s = diffuse_static_anchors(graph, anchors, alpha=0.85, iterations=40)
    assert float(s[anchors].min()) == 1.0, "anchors are not clamped at 1"
    assert 0.0 <= float(s.min()) and float(s.max()) <= 1.0

    near = (~anchors) & (x[:, 0] < 0.25)
    far = x[:, 0] > 0.7
    assert float(s[near].mean()) > float(s[far].mean()), "affinity does not decay with distance"

    logits = initial_part_logits(s, anchors, 3, static_gain=4.0, init_std=1.0)
    prob = torch.softmax(logits, dim=-1)
    assert float(prob[near, 0].mean()) > float(prob[far, 0].mean()), (
        "neighbours of anchors did not receive a larger static logit"
    )
    # Far points stay randomized: not collapsed onto static, not onto moving.
    assert float(prob[far, 0].std()) > 0.05, "far-field initialisation is not randomized"
    assert 0.1 < float(prob[far, 0].mean()) < 0.75, (
        f"far from anchors should be uncertain, got mean p_static {float(prob[far, 0].mean()):.3f}"
    )
    # §9: moving mass is split across parts, not dumped on part 1.
    hard = prob.argmax(dim=-1)
    assert int((hard == 1).sum()) > 0 and int((hard == 2).sum()) > 0, "no moving symmetry break"


# --------------------------------------------------------------------------
# F. No global gate deadlock
# --------------------------------------------------------------------------
def test_F_no_global_gate_can_freeze_the_outer_loop():
    source = inspect.getsource(nested_pipeline.run_nested_part_embedding_killing_discovery)
    for forbidden in ("evaluate_inner_solve", "decision.accepted", "inner_motion_accepted"):
        assert forbidden not in source, f"a global permission gate survives: {forbidden}"
    assert "endpoint_usefulness" in source, "delta-E should still be reported as a diagnostic"

    # Even when the articulated render is worse than static, the embedding
    # optimizer must still move non-anchor probabilities.
    torch.manual_seed(6)
    n = 120
    xyz0 = torch.rand((n, 3), dtype=DT)
    moving = (xyz0 + torch.tensor([9.0, 0.0, 0.0], dtype=DT))[None]  # a terrible hypothesis
    candidates = candidates_from_local_endpoints(xyz0, moving)
    final_xyz = xyz0.clone()                                          # truth: nothing moved
    targets = build_outer_targets(
        final_xyz=final_xyz, candidates=candidates,
        temperature_final_assignment=0.01, chunk_size=64,
    )
    anchors = torch.zeros(n, dtype=torch.bool)
    anchors[:10] = True
    model = PartEmbeddingModel(n, 2, 6, temperature=1.0, static_anchor_mask=anchors,
                               init_logits=torch.randn((n, 2), dtype=torch.float32) * 0.5)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    before = model.probabilities().detach().clone()
    for _ in range(25):
        losses, _ = outer_embedding_losses(
            canonical_xyz=xyz0, final_xyz=final_xyz, part_prob=model.probabilities().to(DT),
            static_anchor_mask=anchors, prototypes=model.prototype.to(DT),
            candidates=candidates, targets=targets,
            bbox_diag=torch.tensor(1.0, dtype=DT), chunk_size=64, margin=0.001,
        )
        total = losses["candidate_nn"] + losses["final_nn"] + losses["part_mass"]
        opt.zero_grad()
        total.backward()
        opt.step()
    after = model.probabilities().detach()
    moved = float((after[~anchors] - before[~anchors]).abs().max())
    assert moved > 1e-3, f"outer E-step did not move non-anchor probabilities ({moved:.2e})"
    # And it moved them the right way: towards static, which explains the endpoint.
    assert float(after[~anchors, 0].mean()) > float(before[~anchors, 0].mean())


# --------------------------------------------------------------------------
# G / H. Recovery in both directions
# --------------------------------------------------------------------------
def _recovery(truth_moves: bool):
    torch.manual_seed(7)
    n = 160
    xyz0 = torch.rand((n, 3), dtype=DT) * 2 - 1
    shift = torch.tensor([0.5, 0.0, 0.0], dtype=DT)
    target = xyz0[:, 0] > 0.0                     # the genuinely moving half
    moving = (xyz0 + shift)[None]
    candidates = candidates_from_local_endpoints(xyz0, moving)
    final_xyz = torch.where(target[:, None], xyz0 + shift, xyz0)
    targets = build_outer_targets(
        final_xyz=final_xyz, candidates=candidates,
        temperature_final_assignment=0.01, chunk_size=64,
    )
    # Start every Gaussian on the WRONG hypothesis.
    logits = torch.zeros((n, 2), dtype=torch.float32)
    wrong = target if not truth_moves else ~target
    logits[wrong, 0] = 2.0
    logits[~wrong, 1] = 2.0
    model = PartEmbeddingModel(n, 2, 6, temperature=1.0,
                               static_anchor_mask=torch.zeros(n, dtype=torch.bool),
                               init_logits=logits)
    opt = torch.optim.Adam(model.parameters(), lr=0.2)
    for _ in range(120):
        losses, _ = outer_embedding_losses(
            canonical_xyz=xyz0, final_xyz=final_xyz, part_prob=model.probabilities().to(DT),
            static_anchor_mask=torch.zeros(n, dtype=torch.bool), prototypes=model.prototype.to(DT),
            candidates=candidates, targets=targets,
            bbox_diag=torch.tensor(2.0, dtype=DT), chunk_size=64, margin=0.001,
        )
        total = (losses["candidate_nn"] + losses["final_nn"]
                 + losses["part_mass"] + losses["part_com"] + losses["candidate_margin"])
        opt.zero_grad()
        total.backward()
        opt.step()
    return model.probabilities().detach().argmax(dim=-1), target


def test_G_true_static_gaussians_can_return_to_static():
    hard, target = _recovery(truth_moves=False)
    correct = float((hard[~target] == 0).float().mean())
    assert correct > 0.8, f"only {correct*100:.1f}% of true-static Gaussians recovered"


def test_H_true_moving_gaussians_can_join_the_moving_support():
    hard, target = _recovery(truth_moves=True)
    correct = float((hard[target] == 1).float().mean())
    assert correct > 0.8, f"only {correct*100:.1f}% of true-moving Gaussians recovered"


# --------------------------------------------------------------------------
# I. Inertia-collapse protection
# --------------------------------------------------------------------------
def test_I_inertia_collapse_is_caught_by_support_and_geometry_guard():
    from rigid_parts.outer_embedding_solver import _soft_inertia_at_positions

    torch.manual_seed(8)
    n = 300
    xyz0 = torch.rand((n, 3), dtype=DT) * 2 - 1
    shift = torch.tensor([0.6, 0.0, 0.0], dtype=DT)
    truly_moving = xyz0[:, 0] > 0.0
    candidates = candidates_from_local_endpoints(xyz0, (xyz0 + shift)[None])
    final_xyz = torch.where(truly_moving[:, None], xyz0 + shift, xyz0)
    targets = build_outer_targets(final_xyz=final_xyz, candidates=candidates,
                                  temperature_final_assignment=0.01, chunk_size=256)

    honest = torch.zeros((n, 2), dtype=DT)
    honest[truly_moving, 1] = 1.0
    honest[~truly_moving, 0] = 1.0
    collapsed = honest.clone()
    sliver = truly_moving & (xyz0[:, 1].abs() > 0.35)
    collapsed[sliver, 1] = 0.0
    collapsed[sliver, 0] = 1.0

    inertia_of_collapsed, _ = _soft_inertia_at_positions(candidates, collapsed)
    _, com_of_honest = _soft_inertia_at_positions(candidates, honest)
    targets["part_inertia"] = inertia_of_collapsed.detach()
    targets["part_com"] = com_of_honest.detach()
    targets["part_mass"] = (honest.sum(dim=0) / n).detach()

    weights = {"candidate_nn": 1.0, "final_nn": 1.0, "global_com": 1.0, "part_com": 2.0, "part_mass": 2.0}

    def losses_for(prob):
        return outer_embedding_losses(
            canonical_xyz=xyz0, final_xyz=final_xyz, part_prob=prob,
            static_anchor_mask=torch.zeros(n, dtype=torch.bool),
            prototypes=torch.zeros((2, 4), dtype=DT),
            candidates=candidates, targets=targets,
            bbox_diag=torch.tensor(2.0, dtype=DT), chunk_size=256, margin=0.001,
        )[0]

    lh, lc = losses_for(honest), losses_for(collapsed)
    assert float(lc["part_inertia"]) < float(lh["part_inertia"]), "test vacuous"
    assert float(lc["part_mass"]) > float(lh["part_mass"]), "§39 support term missed the collapse"
    assert float(geometry_guard_value(lc, weights)) > float(geometry_guard_value(lh, weights))

    # §42: the guard backtracks, it does not deadlock the outer iteration.
    source = inspect.getsource(nested_pipeline.run_nested_part_embedding_killing_discovery)
    assert "load_state_dict" in source and "group[\"lr\"]" in source


# --------------------------------------------------------------------------
# J. Time scale
# --------------------------------------------------------------------------
def test_J_frame_dt_is_one_and_no_hidden_rescaling():
    from types import SimpleNamespace

    from rigid_parts.nested_pipeline import _camera_groups

    class _Cam:
        def __init__(self, fid, ts):
            self.dynamic_frame_id = fid
            self.timestamp = ts

    class _Scene:
        def __init__(self, cams):
            self._c = cams

        def getTrainCameras(self):
            return self._c

    frames = 60
    scene = _Scene([_Cam(f, f / 30.0) for f in range(frames)])
    args = SimpleNamespace(source_path="/nonexistent", dynamic_timestamps_file="", nested_time_unit="auto")
    _, times, _ = _camera_groups(scene, args)

    dt = torch.diff(times)
    assert torch.allclose(dt, torch.ones_like(dt)), f"dt is not 1 per frame: {dt[:3].tolist()}"
    # Neither the 1/30 fps timescale nor a [0,1] sequence normalisation leaked in.
    assert abs(float(dt[0]) - 1.0 / 30.0) > 1e-6
    assert abs(float(dt[0]) - 1.0 / (frames - 1)) > 1e-6
    assert float(times[-1]) == float(frames - 1), "sequence was normalised to [0,1]"


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
    print(f"PASS final_nested_killing_em ({len(tests)} tests)")
