"""Motion-boundary 3DGS split: required tests A-V, K>1 and integration (§38-§44).

The mutating tests run against a REAL ``GaussianModel`` with a real Adam
optimizer, because §39 Test J is specifically about optimizer-state extension
going through the existing machinery -- a stub would test nothing.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rigid_parts.motion_boundary_split import (
    apply_motion_boundary_split,
    build_boundary_split_children,
    estimate_local_boundary_normal,
    find_motion_boundary_split_candidates,
    invalidate_per_gaussian_caches_after_split,
    local_class_distribution,
    rebuild_state_after_boundary_split,
    suppress_sibling_edges,
)
from rigid_parts.neighborhood import build_neighbor_graph

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _model(xyz, scales, *, k1=2, opacity=0.6):
    """A real GaussianModel with a real optimizer, built from explicit tensors."""
    from argparse import ArgumentParser

    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud

    n = int(xyz.shape[0])
    pcd = BasicPointCloud(points=np.asarray(xyz, dtype=np.float64),
                          colors=np.full((n, 3), 0.5), normals=np.zeros((n, 3)))
    g = GaussianModel(0)
    g.create_from_pcd(pcd, [], 1.0)
    opt = OptimizationParams(ArgumentParser())
    args = opt.extract(ArgumentParser().parse_args([])) if False else opt
    class _A:  # minimal training_args
        percent_dense = 0.01
        position_lr_init = 1.6e-4
        position_lr_final = 1.6e-6
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 30000
        feature_lr = 2.5e-3
        opacity_lr = 2.5e-2
        scaling_lr = 5e-3
        rotation_lr = 1e-3
        exposure_lr_init = 0.0
        exposure_lr_final = 0.0
        exposure_lr_delay_steps = 0
        exposure_lr_delay_mult = 0.0
        iterations = 30000
    g.training_setup(_A())

    with torch.no_grad():
        g._scaling.copy_(g.scaling_inverse_activation(
            torch.as_tensor(scales, dtype=torch.float32, device=g._scaling.device)))
        g._opacity.copy_(g.inverse_opacity_activation(
            torch.full((n, 1), float(opacity), device=g._opacity.device)))
        g._rotation.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=g._rotation.device
                                       ).repeat(n, 1))
        g._features_dc.copy_(torch.linspace(0, 1, n, device=g._features_dc.device
                                            )[:, None, None].expand_as(g._features_dc).contiguous())
    # Take one optimizer step so Adam moments EXIST. Without this the state dict
    # is empty and §39 Test J would assert nothing: cat_tensors_to_optimizer's
    # "no stored state" branch would be the one exercised.
    loss = (g._xyz.square().sum() + g._scaling.square().sum() + g._opacity.square().sum()
            + g._rotation.square().sum() + g._features_dc.square().sum()
            + g._features_rest.square().sum())
    loss.backward()
    g.optimizer.step()
    g.optimizer.zero_grad(set_to_none=True)
    return g


def _boundary_scene(n_side=40, gap=0.25, big_scale=0.30, small_scale=0.01):
    """Static slab at x<0, moving slab at x>0, with one huge straddling parent."""
    g = torch.Generator().manual_seed(7)
    left = torch.rand((n_side, 3), generator=g) * torch.tensor([0.4, 0.4, 0.4]) - torch.tensor([0.5, 0.2, 0.2])
    right = torch.rand((n_side, 3), generator=g) * torch.tensor([0.4, 0.4, 0.4]) + torch.tensor([0.1, -0.2, -0.2])
    straddler = torch.tensor([[0.0, 0.0, 0.0]])
    xyz = torch.cat((left, right, straddler), 0)
    scales = torch.full((xyz.shape[0], 3), float(small_scale))
    scales[-1] = float(big_scale)                       # the parent
    probs = torch.zeros((xyz.shape[0], 2))
    probs[:n_side, 0] = 0.98; probs[:n_side, 1] = 0.02          # static
    probs[n_side:2 * n_side, 0] = 0.02; probs[n_side:2 * n_side, 1] = 0.98   # moving
    probs[-1] = torch.tensor([0.003, 0.997])            # §5: confident, yet straddling
    return xyz, scales, probs, int(xyz.shape[0]) - 1


def _candidates(xyz, scales, probs, **kw):
    graph = build_neighbor_graph(xyz.to(DEV), k=12)
    rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEV).repeat(xyz.shape[0], 1)
    return find_motion_boundary_split_candidates(
        xyz=xyz.to(DEV), scaling=scales.to(DEV), rotation=rot,
        soft_probs=probs.to(DEV), graph=graph,
        size_threshold=kw.pop("size_threshold", 0.05),
        max_fraction=kw.pop("max_fraction", 1.0), **kw), graph


# ---------------------------------------------------------------------------
# §41 Tests P, Q, R, S -- the trigger
# ---------------------------------------------------------------------------


def test_p_large_gaussian_crossing_a_boundary_is_selected():
    xyz, scales, probs, parent = _boundary_scene()
    (cands, diag), _ = _candidates(xyz, scales, probs)
    assert any(c.parent_index == parent for c in cands), \
        f"the straddling parent was not selected (diag {diag})"


def test_q_small_gaussian_at_the_same_boundary_is_not_selected():
    xyz, scales, probs, parent = _boundary_scene(big_scale=0.01)
    (cands, _), _ = _candidates(xyz, scales, probs)
    assert not any(c.parent_index == parent for c in cands), \
        "a small Gaussian at the boundary was selected"


def test_r_large_gaussian_in_a_homogeneous_region_is_not_selected():
    xyz, scales, probs, parent = _boundary_scene()
    probs[:] = torch.tensor([0.02, 0.98])          # everything one class
    (cands, diag), _ = _candidates(xyz, scales, probs)
    assert len(cands) == 0, f"a homogeneous region produced {len(cands)} candidates"
    assert diag["num_competing"] == 0, "competing classes detected in a homogeneous region"


def test_s_confident_but_straddling_gaussian_is_still_selected():
    """§41 Test S, critical: selection must NOT require high label entropy."""
    xyz, scales, probs, parent = _boundary_scene()
    probs[parent] = torch.tensor([0.001, 0.999])   # essentially one-hot
    entropy = float(-(probs[parent] * torch.log(probs[parent] + 1e-12)).sum())
    assert entropy < 0.02, "the fixture parent is not confident"
    (cands, _), _ = _candidates(xyz, scales, probs)
    assert any(c.parent_index == parent for c in cands), \
        "a confident boundary-straddling Gaussian was missed; selection is entropy-gated"


def test_trusted_static_parents_are_never_split():
    """§23."""
    xyz, scales, probs, parent = _boundary_scene()
    graph = build_neighbor_graph(xyz.to(DEV), k=12)
    rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEV).repeat(xyz.shape[0], 1)
    trusted = torch.zeros(xyz.shape[0], dtype=torch.bool, device=DEV)
    trusted[parent] = True
    cands, diag = find_motion_boundary_split_candidates(
        xyz=xyz.to(DEV), scaling=scales.to(DEV), rotation=rot, soft_probs=probs.to(DEV),
        graph=graph, trusted_static_mask=trusted, size_threshold=0.05)
    assert not any(c.parent_index == parent for c in cands), "a trusted static parent was split"
    assert diag["num_trusted_skipped"] >= 1, "the skip was not logged"


def test_split_count_is_capped():
    """§32."""
    xyz, scales, probs, _ = _boundary_scene()
    scales[:] = 0.30                                # make everything eligible by size
    (cands, diag), _ = _candidates(xyz, scales, probs, max_fraction=0.01)
    assert len(cands) <= max(0, int(0.01 * xyz.shape[0])), "the fraction cap was exceeded"
    (cands2, _), _ = _candidates(xyz, scales, probs, max_fraction=1.0, max_count=3)
    assert len(cands2) <= 3, "the absolute cap was exceeded"


# ---------------------------------------------------------------------------
# §7/§8 -- boundary normal
# ---------------------------------------------------------------------------


def test_boundary_normal_points_toward_class_b():
    xyz, scales, probs, parent = _boundary_scene()
    graph = build_neighbor_graph(xyz.to(DEV), k=12)
    src = torch.cat((graph.source, graph.target)); dst = torch.cat((graph.target, graph.source))
    w = torch.cat((graph.weight, graph.weight))
    keep = src == parent
    n = estimate_local_boundary_normal(parent, xyz.to(DEV), probs.to(DEV),
                                       dst[keep], w[keep], 0, 1)
    assert n is not None, "no normal was estimated at an obvious boundary"
    # class 1 (moving) lies at +x, so the normal must have a positive x component.
    assert float(n[0]) > 0.5, f"the normal does not point toward class b: {n.tolist()}"
    assert abs(float(n.norm()) - 1.0) < 1.0e-5, "the normal is not unit length"


def test_degenerate_normal_returns_none_rather_than_random():
    """§8: no stable direction -> do not split. A random direction is forbidden."""
    xyz = torch.zeros((5, 3), device=DEV)            # all coincident
    probs = torch.full((5, 2), 0.5, device=DEV)
    nb = torch.arange(1, 5, device=DEV)
    w = torch.ones(4, device=DEV)
    assert estimate_local_boundary_normal(0, xyz, probs, nb, w, 0, 1) is None, \
        "a degenerate neighbourhood produced a direction"


# ---------------------------------------------------------------------------
# §38 Tests A-E -- geometry
# ---------------------------------------------------------------------------


def _split_once(**kw):
    xyz, scales, probs, parent = _boundary_scene()
    g = _model(xyz.numpy(), scales.numpy())
    graph = build_neighbor_graph(g.get_canonical_xyz.detach().float(), k=12)
    cands, _ = find_motion_boundary_split_candidates(
        xyz=g.get_canonical_xyz.detach().float(), scaling=g.get_scaling.detach(),
        rotation=g._rotation.detach(), soft_probs=probs.to(g.get_xyz.device),
        graph=graph, size_threshold=0.05, max_fraction=1.0)
    cands = [c for c in cands if c.parent_index == parent]
    assert cands, "fixture did not produce the expected candidate"
    before = {
        "n": int(g.get_xyz.shape[0]),
        "xyz": g.get_xyz.detach().clone(), "scaling": g.get_scaling.detach().clone(),
        "rotation": g._rotation.detach().clone(), "opacity": g.get_opacity.detach().clone(),
        "f_dc": g._features_dc.detach().clone(), "f_rest": g._features_rest.detach().clone(),
    }
    result = apply_motion_boundary_split(
        g, cands, probs.to(g.get_xyz.device),
        extra_tensors={"memory": torch.zeros((int(xyz.shape[0]), 2), device=g.get_xyz.device)},
        **kw)
    return g, cands[0], result, before, parent


def test_a_one_parent_becomes_two_children_and_is_pruned():
    g, c, result, before, parent = _split_once()
    assert result.num_parents == 1 and result.num_children == 2
    assert int(g.get_xyz.shape[0]) == before["n"] + 1, \
        f"N went {before['n']} -> {int(g.get_xyz.shape[0])}, expected +1"
    assert int(result.child_a_indices.numel()) == 1 and int(result.child_b_indices.numel()) == 1
    # §40 Test N: the parent row is gone.
    kept = g.get_xyz.detach()
    parent_xyz = before["xyz"][parent]
    assert float((kept - parent_xyz).norm(dim=1).min()) > 1.0e-6, "the parent survived"


def test_b_children_are_on_opposite_sides_of_the_normal():
    g, c, result, before, parent = _split_once()
    mu = before["xyz"][parent]
    a = g.get_xyz.detach()[result.child_a_indices[0]]
    b = g.get_xyz.detach()[result.child_b_indices[0]]
    n = c.normal.to(mu.device)
    assert float((a - mu) @ n) < 0, "child a is not on the -n side"
    assert float((b - mu) @ n) > 0, "child b is not on the +n side"
    # §12: never coincident clones.
    assert float((a - b).norm()) > 1.0e-6, "the children are coincident"


def test_c_weighted_centroid_is_preserved():
    g, c, result, before, parent = _split_once()
    mu = before["xyz"][parent]
    a = g.get_xyz.detach()[result.child_a_indices[0]]
    b = g.get_xyz.detach()[result.child_b_indices[0]]
    centroid = c.omega_a * a + c.omega_b * b
    assert float((centroid - mu).norm()) < 1.0e-5, \
        f"w_a mu_a + w_b mu_b = {centroid.tolist()}, parent {mu.tolist()}"


def test_d_offsets_stay_within_the_configured_maximum():
    g, c, result, before, parent = _split_once()
    mu = before["xyz"][parent]
    for idx in (result.child_a_indices[0], result.child_b_indices[0]):
        offset = float((g.get_xyz.detach()[idx] - mu).norm())
        assert offset <= 1.0 * c.sigma_normal + 1.0e-6, \
            f"offset {offset:.6f} exceeds max_offset_sigma * sigma_n = {c.sigma_normal:.6f}"


def test_e_child_scale_is_parent_over_1_point_6():
    g, c, result, before, parent = _split_once()
    parent_scale = before["scaling"][parent]
    for idx in (result.child_a_indices[0], result.child_b_indices[0]):
        child = g.get_scaling.detach()[idx]
        assert torch.allclose(child, parent_scale / 1.6, rtol=1.0e-4), \
            f"child scale {child.tolist()} != parent/1.6 {(parent_scale/1.6).tolist()}"


# ---------------------------------------------------------------------------
# §39 Tests F-J -- appearance and optimizer
# ---------------------------------------------------------------------------


def test_f_g_appearance_and_rotation_are_copied():
    g, c, result, before, parent = _split_once()
    for idx in (result.child_a_indices[0], result.child_b_indices[0]):
        assert torch.allclose(g._features_dc.detach()[idx], before["f_dc"][parent]), \
            "SH DC was not copied exactly"
        assert torch.allclose(g._features_rest.detach()[idx], before["f_rest"][parent]), \
            "SH rest was not copied exactly"
        assert torch.allclose(g._rotation.detach()[idx], before["rotation"][parent]), \
            "rotation was not copied exactly"


def test_h_opacity_mass_is_split_not_duplicated():
    g, c, result, before, parent = _split_once()
    parent_alpha = float(before["opacity"][parent])
    a = float(g.get_opacity.detach()[result.child_a_indices[0]])
    b = float(g.get_opacity.detach()[result.child_b_indices[0]])
    assert abs((a + b) - parent_alpha) < 1.0e-4, \
        f"opacity mass {a + b:.5f} != parent {parent_alpha:.5f}"
    assert a < parent_alpha and b < parent_alpha, "a child kept the full parent opacity"
    assert abs(a - c.omega_a * parent_alpha) < 1.0e-4


def test_i_all_gaussian_tensors_have_the_new_length():
    g, c, result, before, parent = _split_once()
    n = int(g.get_xyz.shape[0])
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        assert int(getattr(g, name).shape[0]) == n, f"{name} has the wrong length"
    for name, t in (("soft_part_probs", result.soft_part_probs),
                    ("split_group_id", result.split_group_id),
                    ("split_parent_id", result.split_parent_id),
                    ("split_child_role", result.split_child_role)):
        assert int(t.shape[0]) == n, f"{name} has the wrong length"
    assert int(g.tmp_radii.shape[0]) == n and int(g.max_radii2D.shape[0]) == n
    gt = getattr(g, "_gt_part_labels", None)
    if gt is not None and gt.numel():
        assert int(gt.shape[0]) == n, "gt_part_labels was not resized with the model"


def test_j_optimizer_state_is_extended_through_existing_machinery():
    g, c, result, before, parent = _split_once()
    n = int(g.get_xyz.shape[0])
    checked = 0
    for group in g.optimizer.param_groups:
        param = group["params"][0]
        if param.shape[0] != n:
            continue
        state = g.optimizer.state.get(param, None)
        if state is None or "exp_avg" not in state:
            continue
        checked += 1
        assert int(state["exp_avg"].shape[0]) == n, \
            f"exp_avg for {group['name']} has {int(state['exp_avg'].shape[0])} rows, expected {n}"
        assert int(state["exp_avg_sq"].shape[0]) == n, f"exp_avg_sq wrong for {group['name']}"
        # The appended rows start at zero, exactly as the Graphdeco machinery
        # does. (features_rest is [N, 0, 3] at sh_degree 0, so skip empty rows.)
        last = state["exp_avg"][-1]
        if int(last.numel()) > 0:
            assert float(last.abs().max()) == 0.0, \
                f"appended {group['name']} moments are not zero-initialised"
    assert checked > 0, "no Adam moments existed; the test asserted nothing"


# ---------------------------------------------------------------------------
# §40 Tests K-O -- labels and memory
# ---------------------------------------------------------------------------


def test_k_l_m_children_get_complementary_soft_labels():
    g, c, result, before, parent = _split_once()
    pa = result.soft_part_probs[result.child_a_indices[0]]
    pb = result.soft_part_probs[result.child_b_indices[0]]
    assert int(pa.argmax()) == c.class_a, f"child a is not class a: {pa.tolist()}"
    assert int(pb.argmax()) == c.class_b, f"child b is not class b: {pb.tolist()}"
    # §18: strong but SOFT, and never the same vector.
    assert float(pa.max()) < 1.0 and float(pb.max()) < 1.0, "a child label is one-hot"
    assert float(pa.max()) > 0.9 and float(pb.max()) > 0.9, "a child label is not strong"
    assert float((pa - pb).abs().sum()) > 0.5, "the children got the same distribution"
    for name, p in (("a", pa), ("b", pb)):
        assert abs(float(p.sum()) - 1.0) < 1.0e-5, f"child {name} row does not sum to 1"


def test_o_child_memories_are_exactly_zero():
    """§22: children never inherit parent temporal credit."""
    xyz, scales, probs, parent = _boundary_scene()
    g = _model(xyz.numpy(), scales.numpy())
    graph = build_neighbor_graph(g.get_canonical_xyz.detach().float(), k=12)
    cands, _ = find_motion_boundary_split_candidates(
        xyz=g.get_canonical_xyz.detach().float(), scaling=g.get_scaling.detach(),
        rotation=g._rotation.detach(), soft_probs=probs.to(DEV), graph=graph,
        size_threshold=0.05, max_fraction=1.0)
    cands = [c for c in cands if c.parent_index == parent]
    memory = torch.full((int(xyz.shape[0]), 2), 0.7, device=DEV)   # non-zero parent memory
    result = apply_motion_boundary_split(g, cands, probs.to(DEV),
                                         extra_tensors={"memory": memory})
    new_memory = result.diagnostics["extra"]["memory"]
    for idx in (result.child_a_indices[0], result.child_b_indices[0]):
        assert float(new_memory[idx].abs().max()) == 0.0, \
            f"a child inherited memory {new_memory[idx].tolist()}"
    assert int(new_memory.shape[0]) == int(g.get_xyz.shape[0])


# ---------------------------------------------------------------------------
# §42 Tests T, U, V -- graph
# ---------------------------------------------------------------------------


def test_t_u_v_graph_is_rebuilt_without_sibling_edges():
    g, c, result, before, parent = _split_once()
    graph = rebuild_state_after_boundary_split(g, result, k=12)
    n = int(g.get_xyz.shape[0])
    assert graph.num_nodes == n, "the graph was not rebuilt at the new size"       # Test T
    a = int(result.child_a_indices[0]); b = int(result.child_b_indices[0])
    sibling = (((graph.source == a) & (graph.target == b))
               | ((graph.source == b) & (graph.target == a)))
    if bool(sibling.any()):
        assert float(graph.weight[sibling].abs().max()) == 0.0, \
            "the sibling smoothing edge was not suppressed"                       # Test U
    # Test V: the children still have real connections to other neighbours.
    for child in (a, b):
        touching = (graph.source == child) | (graph.target == child)
        live = touching & (graph.weight > 0)
        assert int(live.sum()) > 0, f"child {child} lost every neighbour"


def test_u_sibling_suppression_is_exact():
    pts = torch.tensor([[0.0, 0, 0], [0.01, 0, 0], [1.0, 0, 0], [1.01, 0, 0]], dtype=torch.float32)
    graph = build_neighbor_graph(pts, k=3)
    gid = torch.tensor([0, 0, -1, -1])
    out = suppress_sibling_edges(graph, gid)
    sib = (((out.source == 0) & (out.target == 1)) | ((out.source == 1) & (out.target == 0)))
    assert bool(sib.any()), "the fixture has no sibling edge to suppress"
    assert float(out.weight[sib].max()) == 0.0, "sibling weight is not zero"
    assert float(out.weight[~sib].max()) > 0.0, "non-sibling edges were destroyed"
    assert torch.equal(out.source, graph.source), "the edge list was altered"


# ---------------------------------------------------------------------------
# §43 -- K > 1 without hard-coded binary logic
# ---------------------------------------------------------------------------


def test_k_greater_than_one_supports_both_0_1_and_1_2_splits():
    """§43: local top-two must work for static/part1 AND part1/part2."""
    g0 = torch.Generator().manual_seed(9)
    # Three slabs along x with narrow gaps, so a straddler at each gap really
    # does have neighbours from BOTH adjacent classes within k = 12.
    static = torch.rand((30, 3), generator=g0) * 0.3 - torch.tensor([0.70, 0.15, 0.15])
    part1 = torch.rand((30, 3), generator=g0) * 0.3 - torch.tensor([0.15, 0.15, 0.15])
    part2 = torch.rand((30, 3), generator=g0) * 0.3 + torch.tensor([0.40, -0.15, -0.15])
    straddle_01 = torch.tensor([[-0.275, 0.0, 0.0]])       # between static and part1
    straddle_12 = torch.tensor([[0.275, 0.0, 0.0]])        # between part1 and part2
    xyz = torch.cat((static, part1, part2, straddle_01, straddle_12), 0).to(DEV)

    probs = torch.zeros((xyz.shape[0], 3), device=DEV)
    probs[:30] = torch.tensor([0.96, 0.02, 0.02], device=DEV)
    probs[30:60] = torch.tensor([0.02, 0.96, 0.02], device=DEV)
    probs[60:90] = torch.tensor([0.02, 0.02, 0.96], device=DEV)
    probs[90] = torch.tensor([0.50, 0.48, 0.02], device=DEV)
    probs[91] = torch.tensor([0.02, 0.50, 0.48], device=DEV)

    scales = torch.full((xyz.shape[0], 3), 0.01, device=DEV)
    scales[90] = 0.30
    scales[91] = 0.30
    graph = build_neighbor_graph(xyz, k=12)
    rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEV).repeat(xyz.shape[0], 1)
    cands, diag = find_motion_boundary_split_candidates(
        xyz=xyz, scaling=scales, rotation=rot, soft_probs=probs, graph=graph,
        size_threshold=0.05, max_fraction=1.0)

    pairs = {c.parent_index: tuple(sorted((c.class_a, c.class_b))) for c in cands}
    assert 90 in pairs, f"the static/part1 straddler was not selected (diag {diag})"
    assert 91 in pairs, f"the part1/part2 straddler was not selected (diag {diag})"
    assert pairs[90] == (0, 1), f"wrong class pair at the 0/1 boundary: {pairs[90]}"
    assert pairs[91] == (1, 2), f"wrong class pair at the 1/2 boundary: {pairs[91]}"


def test_local_class_distribution_uses_neighbours_only():
    """§5/§6: a confident parent must still show competing NEIGHBOURS."""
    pts = torch.tensor([[0.0, 0, 0], [-0.1, 0, 0], [0.1, 0, 0]], dtype=torch.float32)
    graph = build_neighbor_graph(pts, k=2)
    probs = torch.tensor([[0.001, 0.999], [0.99, 0.01], [0.01, 0.99]])
    bar = local_class_distribution(probs, graph)
    assert float(bar[0, 0]) > 0.15 and float(bar[0, 1]) > 0.15, \
        f"a confident parent's neighbourhood did not show competition: {bar[0].tolist()}"


# ---------------------------------------------------------------------------
# §44 -- integration
# ---------------------------------------------------------------------------


def test_integration_split_between_outer0_and_inner1():
    """§44: split at the right point, memory still zero, count refined."""
    from rigid_parts.soft_labels import TemporalOrderGuard

    guard = TemporalOrderGuard()
    guard.record_solve()                                   # INNER 0
    memory = torch.zeros((81, 2), device=DEV)
    guard.check_memory_is_zero(memory)                     # still zero before the split

    xyz, scales, probs, parent = _boundary_scene()
    g = _model(xyz.numpy(), scales.numpy())
    n_before = int(g.get_xyz.shape[0])
    graph = build_neighbor_graph(g.get_canonical_xyz.detach().float(), k=12)
    cands, _ = find_motion_boundary_split_candidates(
        xyz=g.get_canonical_xyz.detach().float(), scaling=g.get_scaling.detach(),
        rotation=g._rotation.detach(), soft_probs=probs.to(DEV), graph=graph,
        size_threshold=0.05, max_fraction=1.0)
    cands = [c for c in cands if c.parent_index == parent]

    memory = torch.zeros((n_before, 2), device=DEV)
    result = apply_motion_boundary_split(g, cands, probs.to(DEV),
                                         extra_tensors={"memory": memory})
    new_memory = result.diagnostics["extra"]["memory"]

    # No temporal credit update happened during the split.
    assert guard.as_dict()["num_temporal_credit_updates"] == 0, \
        "a credit update occurred during the split"
    guard.check_memory_is_zero(new_memory)
    assert float(new_memory.abs().max()) == 0.0, "memory is not zero before INNER 1"

    # The inner solver would receive the REFINED count and the intended classes.
    z = result.soft_part_probs.argmax(dim=1)
    assert int(z.shape[0]) == int(g.get_xyz.shape[0]) == n_before + 1
    assert int(z[result.child_a_indices[0]]) == cands[0].class_a, \
        "child a's temporary hard label is not class a"
    assert int(z[result.child_b_indices[0]]) == cands[0].class_b, \
        "child b's temporary hard label is not class b"

    caches = [{"stale": 1}, [1, 2, 3]]
    invalidate_per_gaussian_caches_after_split(*caches)
    assert caches[0] == {} and caches[1] == [], "§29: caches were not invalidated"


def test_no_candidates_is_a_safe_no_op():
    xyz, scales, probs, _ = _boundary_scene(big_scale=0.01)
    g = _model(xyz.numpy(), scales.numpy())
    n = int(g.get_xyz.shape[0])
    result = apply_motion_boundary_split(g, [], probs.to(DEV))
    assert result.num_parents == 0 and int(g.get_xyz.shape[0]) == n, "a no-op changed the model"
    assert int(result.soft_part_probs.shape[0]) == n


def test_cli_defaults_match_the_spec():
    from argparse import ArgumentParser

    from arguments import ModelParams, OptimizationParams, PipelineParams, add_dynamic_args

    parser = ArgumentParser()
    ModelParams(parser); OptimizationParams(parser); PipelineParams(parser)
    add_dynamic_args(parser)
    a = parser.parse_args([])
    for flag, expected in (
        ("enable_motion_boundary_split", False),
        ("boundary_split_local_class_min", 0.15),
        ("boundary_split_extent_sigma", 2.0),
        ("boundary_split_offset_sigma", 0.5),
        ("boundary_split_max_offset_sigma", 1.0),
        ("boundary_split_gradient_ridge", 1.0e-6),
        ("boundary_split_label_mix", 0.05),
        ("boundary_split_max_fraction", 0.01),
        ("boundary_split_max_count", 0),
        ("boundary_split_opacity_mode", "weighted_mass"),
    ):
        assert hasattr(a, flag), f"--{flag} is not registered"
        assert getattr(a, flag) == expected, \
            f"--{flag} default is {getattr(a, flag)}, spec says {expected}"


if __name__ == "__main__":
    module = sys.modules["__main__"]
    tests = [(n, f) for n, f in sorted(vars(module).items())
             if n.startswith("test_") and inspect.isfunction(f)]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"PASS motion_boundary_split ({len(tests)} tests)")
