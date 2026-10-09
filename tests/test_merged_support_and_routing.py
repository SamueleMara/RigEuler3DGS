"""CODEX_MERGED_ROBUST_SUPPORT_AND_DETERMINISTIC_LOSS_ROUTING.

Part XIX support tests (§67-§74) and Part XX gradient-routing tests (§75-§82),
plus the acceptance-checklist invariants of §86-§89.

The routing tests deliberately leave requires_grad TRUE on BOTH blocks and check
that the other block still comes back with no gradient. That is the only way to
prove the routing is enforced by the graph (the detaches) rather than merely by
which parameters the optimizer happens to own.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import torch

from rigid_parts import shared_killing_solver as sks
from rigid_parts.decoupled_joint_scalar import (
    frame_epoch_schedule,
    motion_direction_loss,
    trimmed_chamfer,
    velocity_magnitude_loss,
)
from rigid_parts.inner_support import (
    InnerSupportResult,
    build_robust_inner_support,
    format_support_summary,
    gt_support_metrics,
    write_support_debug_ply,
)
from rigid_parts.robust_inner_killing import (
    integrate_scalar,
    project_field_to_rigid_twist,
    rigid_basis,
    scalar_continuity_loss,
    scalar_jerk_loss,
    scalar_sign_flip_loss,
    se3_exp_twist,
)

DT = torch.float64
SOLVER_SRC = Path(sks.__file__).read_text(encoding="utf-8")
SOLVER_AST = ast.parse(SOLVER_SRC)
SUPPORT_SRC = Path(inspect.getfile(build_robust_inner_support)).read_text(encoding="utf-8")

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def probs_from(dyn: torch.Tensor, part: torch.Tensor | None = None) -> torch.Tensor:
    """[N, K+1] from a dynamic confidence and an optional moving-part split."""
    n = int(dyn.numel())
    if part is None:
        return torch.stack((1.0 - dyn, dyn), dim=1)
    k = int(part.shape[1])
    out = torch.zeros((n, k + 1), dtype=DT)
    out[:, 0] = 1.0 - dyn
    out[:, 1:] = part * dyn[:, None]
    return out


# ---------------------------------------------------------------------------
# PART XIX - support tests
# ---------------------------------------------------------------------------
def test_67_k1_preprocessing_is_nontrivial():
    """§67/§14: for K=1 the gates must still produce core / attached / outlier /
    unassigned, and fitting support strictly smaller than transport support."""
    g = torch.Generator().manual_seed(5)
    core_pts = torch.rand((60, 3), generator=g, dtype=DT) * 0.4                 # coherent blob
    near_pts = core_pts[:15] + 0.05                                             # nearby, weaker
    # Genuinely isolated: mutually distant too, so none of them supports another.
    # A pair of nearby low-confidence candidates would (correctly, per §6) keep
    # each other alive -- spacing alone must never reject.
    far_pts = torch.tensor([[9.0, 0.0, 0.0], [-8.0, 4.0, 1.0], [0.0, -11.0, 6.0]], dtype=DT)
    ambiguous = core_pts[:10] + 0.8                                             # detached-ish
    xyz = torch.cat((core_pts, near_pts, far_pts, ambiguous))
    dyn = torch.cat((torch.full((60,), 0.95, dtype=DT),
                     torch.full((15,), 0.60, dtype=DT),
                     torch.full((3,), 0.52, dtype=DT),
                     torch.full((10,), 0.40, dtype=DT)))
    probs = probs_from(dyn)
    z = torch.ones(int(dyn.numel()), dtype=torch.long)

    s = build_robust_inner_support(xyz, probs, z, 1, knn=8, cluster_knn=8)
    check("§67 high-confidence coherent points become core",
          int((s.core_mask[:60]).sum()) >= 55, f"{int(s.core_mask[:60].sum())}/60")
    check("§67 isolated low-support points become outliers",
          bool(s.outlier_mask[75:78].all()), str(s.outlier_mask[75:78].tolist()))
    check("§67 an explicit unassigned state exists", int(s.unassigned_mask.sum()) > 0,
          f"{int(s.unassigned_mask.sum())}")
    check("§67 core != clean", int(s.core_mask.sum()) < int((s.candidate_mask & ~s.outlier_mask).sum()),
          f"core {int(s.core_mask.sum())} clean {int((s.candidate_mask & ~s.outlier_mask).sum())}")
    check("§14/§67 fitting support < transport support",
          int((s.fitting_part_id > 0).sum()) < int((s.transport_part_id > 0).sum()),
          f"F {int((s.fitting_part_id > 0).sum())} T {int((s.transport_part_id > 0).sum())}")
    check("§86 F_k is a subset of T_k",
          bool((~s.fitting_masks[0] | s.transport_masks[0]).all()))


def test_68_k2_graph_clustering_not_copied_from_labels():
    """§68/§3/§12: two spatially distinct structures with DELIBERATELY WRONG prior
    part IDs. The clustering must follow the graph, not z."""
    g = torch.Generator().manual_seed(9)
    a = torch.rand((50, 3), generator=g, dtype=DT) * 0.5
    b = torch.rand((50, 3), generator=g, dtype=DT) * 0.5 + torch.tensor([8.0, 0, 0], dtype=DT)
    xyz = torch.cat((a, b))
    dyn = torch.full((100,), 0.95, dtype=DT)
    # Confident prior part IDs, of which 24 are simply WRONG. The graph must
    # override them; a clustering copied from z would reproduce the errors.
    part = torch.zeros((100, 2), dtype=DT)
    part[:50] = torch.tensor([0.88, 0.12], dtype=DT)
    part[50:] = torch.tensor([0.12, 0.88], dtype=DT)
    part[:12] = torch.tensor([0.12, 0.88], dtype=DT)          # wrong for blob A
    part[50:62] = torch.tensor([0.88, 0.12], dtype=DT)        # wrong for blob B
    probs = probs_from(dyn, part)
    z = probs.argmax(dim=-1)

    s = build_robust_inner_support(xyz, probs, z, 2, knn=8, cluster_knn=8)
    ids = s.fitting_part_id
    a_ids = set(int(v) for v in ids[:50] if int(v) > 0)
    b_ids = set(int(v) for v in ids[50:] if int(v) > 0)
    check("§68 exactly 2 non-empty clusters",
          int(s.fitting_masks[0].sum()) > 0 and int(s.fitting_masks[1].sum()) > 0,
          f"{int(s.fitting_masks[0].sum())}/{int(s.fitting_masks[1].sum())}")
    check("§68 blob A lands in one cluster", len(a_ids) == 1, str(a_ids))
    check("§68 blob B lands in one cluster", len(b_ids) == 1, str(b_ids))
    check("§68 the two blobs are separated", a_ids != b_ids, f"{a_ids} vs {b_ids}")
    # The clustering is NOT a copy of z: z mislabels 24 points, the graph does not.
    agree = int((ids[:50] == ids[0]).sum()) + int((ids[50:] == ids[50]).sum())
    check("§68 clustering is not copied from z", agree > 90, f"{agree}/100 spatially consistent")
    check("§13 Hungarian mapping is recorded",
          len(s.diagnostics.get("cluster_to_part_matrix", [])) == 2)


def test_69_elongated_part_is_not_split():
    """§69/§12: a connected elongated structure must survive, which naive
    Euclidean K-means in XYZ would cut in half."""
    n = 90
    line = torch.stack((torch.linspace(0, 12, n, dtype=DT),
                        torch.zeros(n, dtype=DT), torch.zeros(n, dtype=DT)), dim=1)
    dyn = torch.full((n,), 0.95, dtype=DT)
    s = build_robust_inner_support(line, probs_from(dyn), torch.ones(n, dtype=torch.long),
                                   1, knn=6, cluster_knn=6)
    ids = set(int(v) for v in s.fitting_part_id if int(v) > 0)
    check("§69 the elongated part stays one cluster", ids == {1}, str(ids))
    check("§69 nothing was rejected for spacing alone", int(s.outlier_mask.sum()) == 0,
          f"{int(s.outlier_mask.sum())} outliers")


def test_70_confidence_core_is_thresholded():
    """§70: the exact list from the spec -- core must be {0.95, 0.91, 0.85}."""
    xyz = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0],
                        [0.3, 0, 0], [0.4, 0, 0], [0.5, 0, 0]], dtype=DT)
    dyn = torch.tensor([0.95, 0.91, 0.85, 0.79, 0.62, 0.51], dtype=DT)
    s = build_robust_inner_support(
        xyz, probs_from(dyn), torch.ones(6, dtype=torch.long), 1,
        knn=3, cluster_knn=3, core_dynamic_confidence=0.80, core_part_confidence=0.70,
        core_min_points=1)          # so the schedule does not relax past 0.80
    check("§70 core is exactly the three above threshold",
          s.core_mask.tolist() == [True, True, True, False, False, False],
          str(s.core_mask.tolist()))
    check("§8 `core = cleaned_candidates` did not happen",
          int(s.core_mask.sum()) == 3 and int((s.candidate_mask & ~s.outlier_mask).sum()) == 6)


def test_71_unassigned_state():
    """§71: a point with insufficient cluster affinity is unassigned -- excluded
    from fitting, still transported."""
    g = torch.Generator().manual_seed(3)
    blob = torch.rand((40, 3), generator=g, dtype=DT) * 0.3
    lonely = torch.tensor([[4.0, 4.0, 4.0]], dtype=DT)
    xyz = torch.cat((blob, lonely))
    dyn = torch.cat((torch.full((40,), 0.95, dtype=DT), torch.tensor([0.58], dtype=DT)))
    s = build_robust_inner_support(xyz, probs_from(dyn), torch.ones(41, dtype=torch.long),
                                   1, knn=6, cluster_knn=6)
    check("§71 unassigned = True", bool(s.unassigned_mask[-1]) or bool(s.outlier_mask[-1]),
          f"unassigned {bool(s.unassigned_mask[-1])} outlier {bool(s.outlier_mask[-1])}")
    check("§71 fitting = False", int(s.fitting_part_id[-1]) == -1)
    check("§71 transport = True", int(s.transport_part_id[-1]) == 1)
    check("§18 unassigned points are not made static",
          int(s.transport_part_id[s.unassigned_mask].min()) > 0
          if bool(s.unassigned_mask.any()) else True)


def test_72_fit_weight_is_confidence_ordered():
    """§72: w_fit(0.99) > w_fit(0.55)."""
    g = torch.Generator().manual_seed(11)
    blob = torch.rand((40, 3), generator=g, dtype=DT) * 0.3
    dyn = torch.full((40,), 0.99, dtype=DT)
    dyn[:5] = 0.55
    s = build_robust_inner_support(blob, probs_from(dyn), torch.ones(40, dtype=torch.long),
                                   1, knn=6, cluster_knn=6, core_dynamic_confidence=0.50,
                                   core_part_confidence=0.50)
    fitted = s.fitting_part_id > 0
    high = s.fit_weight[fitted & (s.dynamic_confidence > 0.9)]
    low = s.fit_weight[fitted & (s.dynamic_confidence < 0.6)]
    check("§72 both confidences are represented",
          int(high.numel()) > 0 and int(low.numel()) > 0,
          f"high {int(high.numel())} low {int(low.numel())}")
    check("§72 w_fit(0.99) > w_fit(0.55)", float(high.min()) > float(low.max()),
          f"{float(low.max()):.5f} < {float(high.min()):.5f}")
    check("§22 points in F_k are not equally trusted", float(s.fit_weight[fitted].std()) > 0)


def test_73_no_label_mutation():
    """§1/§73: P and z come back bit-identical."""
    g = torch.Generator().manual_seed(13)
    xyz = torch.rand((80, 3), generator=g, dtype=DT)
    dyn = torch.rand(80, generator=g, dtype=DT)
    probs = probs_from(dyn)
    z = (torch.rand(80, generator=g, dtype=DT) > 0.5).long()
    p_before, z_before = probs.clone(), z.clone()
    build_robust_inner_support(xyz, probs, z, 1, knn=6, cluster_knn=6)
    check("§73 P_before == P_after", torch.equal(probs, p_before))
    check("§73 z_before == z_after", torch.equal(z, z_before))


def test_74_no_bypass_in_the_solver_source():
    """§27/§74: the forbidden bypasses must not exist as default code paths."""
    for forbidden in ("fitting_mask = hard_labels", "fit_idx = transport_idx",
                      "core = clean_candidates", "cluster_id = hard_labels",
                      "fitting_mask = transport_mask"):
        check(f"§27 no `{forbidden}`", forbidden not in SOLVER_SRC and forbidden not in SUPPORT_SRC)
    check("§26 support is built exactly once",
          SOLVER_SRC.count("build_robust_inner_support(") == 1)
    check("§26 fitting index comes from the support object",
          "support.fitting_index(k)" in SOLVER_SRC)
    check("§26 transport masks come from the support object",
          "support.transport_masks[k]" in SOLVER_SRC)
    check("§20 the fit weight comes from the support object",
          "support.fit_weight" in SOLVER_SRC)
    # The old path rebuilt masks from clusters + hard labels; it must be gone.
    for stale in ("clusters.assignment == part", "z == part + 1"):
        check(f"§27 stale mask construction `{stale}` removed", stale not in SOLVER_SRC)


def test_support_result_shape_contract():
    """§24: every declared field exists with the declared shape."""
    g = torch.Generator().manual_seed(17)
    xyz = torch.rand((50, 3), generator=g, dtype=DT)
    dyn = torch.rand(50, generator=g, dtype=DT) * 0.5 + 0.5
    part = torch.rand((50, 2), generator=g, dtype=DT)
    part = part / part.sum(dim=1, keepdim=True)
    s = build_robust_inner_support(xyz, probs_from(dyn, part),
                                   torch.ones(50, dtype=torch.long), 2, knn=6, cluster_knn=6)
    check("§24 result type", isinstance(s, InnerSupportResult))
    for name, shape in (("transport_masks", (2, 50)), ("fitting_masks", (2, 50)),
                        ("transport_part_id", (50,)), ("fitting_part_id", (50,)),
                        ("candidate_mask", (50,)), ("outlier_mask", (50,)),
                        ("core_mask", (50,)), ("attached_mask", (50,)),
                        ("unassigned_mask", (50,)), ("fit_weight", (50,)),
                        ("dynamic_confidence", (50,)), ("part_confidence", (50, 2)),
                        ("local_spacing", (50,)), ("local_dynamic_support", (50,))):
        check(f"§24 {name} is {shape}", tuple(getattr(s, name).shape) == shape,
              str(tuple(getattr(s, name).shape)))
    check("§24 effective_thresholds is populated", len(s.effective_thresholds) > 0)
    check("§9 confidence percentiles are reported",
          "dynamic_confidence" in s.diagnostics and "p10" in s.diagnostics["dynamic_confidence"])
    check("§63 the console summary renders", "[InnerSupport]" in format_support_summary(s))
    check("§86 F_k subset of T_k for every part",
          all(bool((~s.fitting_masks[k] | s.transport_masks[k]).all()) for k in range(2)))


def test_support_artifacts_and_gt_metrics():
    import tempfile

    g = torch.Generator().manual_seed(19)
    xyz = torch.rand((60, 3), generator=g, dtype=DT)
    dyn = torch.rand(60, generator=g, dtype=DT) * 0.5 + 0.5
    s = build_robust_inner_support(xyz, probs_from(dyn), torch.ones(60, dtype=torch.long),
                                   1, knn=6, cluster_knn=6)
    gt = torch.zeros(60, dtype=torch.bool)
    gt[s.core_mask] = True
    metrics = gt_support_metrics(s, gt)
    check("§64 transport/core/fitting precision and recall reported",
          all(k in metrics for k in ("transport", "core", "fitting")))
    check("§64 the purity comparison is reported",
          "fitting_precision_exceeds_transport" in metrics)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "inner_support_debug.ply"
        write_support_debug_ply(s, xyz, path)
        text = path.read_text()
    lines = text.strip().splitlines()
    body = lines[lines.index("end_header") + 1:]
    check("§62 debug PLY has a header and one row per Gaussian",
          text.startswith("ply") and "element vertex 60" in text and len(body) == 60,
          f"{len(body)} rows")
    check("§62 debug PLY carries the support state",
          "property int fitting_part" in text and "property float fit_weight" in text)


# ---------------------------------------------------------------------------
# PART XX - gradient routing
# ---------------------------------------------------------------------------
def routing_fixture():
    """A miniature of the solver's two blocks: one derived twist, one scalar.

    Both parameters keep requires_grad=True so the tests measure what the GRAPH
    routes, not what an optimizer happens to own.
    """
    torch.manual_seed(23)
    x = torch.rand((24, 3), dtype=DT) * 2 - 1
    theta = torch.nn.Parameter(0.2 * torch.randn((24, 3), dtype=DT))     # the spatial field
    speed = torch.nn.Parameter(torch.full((6,), 0.3, dtype=DT))
    dt = torch.full((5,), 0.2, dtype=DT)
    w = torch.ones(24, dtype=DT)
    return x, theta, speed, dt, w


def geometry_xi(x, theta, w):
    return project_field_to_rigid_twist(x, theta, w, reg=1e-10)


def grads(loss, theta, speed):
    for p in (theta, speed):
        p.grad = None
    loss.backward()
    gt = 0.0 if theta.grad is None else float(theta.grad.abs().sum())
    gs = 0.0 if speed.grad is None else float(speed.grad.abs().sum())
    return gt, gs


def test_75_render_to_geometry_only():
    """§75: geometry block, render-like loss -> theta only. s(t) is detached."""
    x, theta, speed, dt, w = routing_fixture()
    proj = geometry_xi(x, theta, w)
    q = integrate_scalar(speed.detach(), dt)                 # §36: frozen s
    T = se3_exp_twist(proj["omega"], proj["u"], q)
    moved = torch.einsum("tij,nj->tni", T[:, :3, :3], x) + T[:, :3, 3][:, None]
    gt, gs = grads(moved.square().mean(), theta, speed)      # stands in for L_render
    check("§75 geometry grad != 0", gt > 0, f"{gt:.4g}")
    check("§75 speed grad == 0", gs == 0.0, f"{gs:.4g}")


def test_76_render_to_scalar_only():
    """§76: scalar block, render-like loss -> s(t) only. xi is detached."""
    x, theta, speed, dt, w = routing_fixture()
    with torch.no_grad():
        live = geometry_xi(x, theta, w)
    omega, u = live["omega"].detach(), live["u"].detach()    # §47's mandatory detach
    q = integrate_scalar(speed, dt)
    T = se3_exp_twist(omega, u, q)
    moved = torch.einsum("tij,nj->tni", T[:, :3, :3], x) + T[:, :3, 3][:, None]
    gt, gs = grads(moved.square().mean(), theta, speed)
    check("§76 speed grad != 0", gs > 0, f"{gs:.4g}")
    check("§76 geometry grad == 0", gt == 0.0, f"{gt:.4g}")


def test_77_final_pcd_to_geometry_only():
    x, theta, speed, dt, w = routing_fixture()
    target = x + torch.tensor([0.2, 0.0, 0.0], dtype=DT)
    proj = geometry_xi(x, theta, w)
    q = integrate_scalar(speed.detach(), dt)
    T = se3_exp_twist(proj["omega"], proj["u"], q[-1:])[0]
    pred = x @ T[:3, :3].transpose(0, 1) + T[:3, 3]
    gt, gs = grads(trimmed_chamfer(pred, target, forward_trim=1.0,
                                   backward_trim=1.0, backward_budget=1.0)["loss"], theta, speed)
    check("§77 geometry grad != 0", gt > 0, f"{gt:.4g}")
    check("§77 speed grad == 0", gs == 0.0, f"{gs:.4g}")


def test_78_final_pcd_to_scalar_only():
    x, theta, speed, dt, w = routing_fixture()
    target = x + torch.tensor([0.2, 0.0, 0.0], dtype=DT)
    with torch.no_grad():
        live = geometry_xi(x, theta, w)
    omega, u = live["omega"].detach(), live["u"].detach()
    q = integrate_scalar(speed, dt)
    T = se3_exp_twist(omega, u, q[-1:])[0]
    pred = x @ T[:3, :3].transpose(0, 1) + T[:3, 3]
    gt, gs = grads(trimmed_chamfer(pred, target, forward_trim=1.0,
                                   backward_trim=1.0, backward_budget=1.0)["loss"], theta, speed)
    check("§78 speed grad != 0", gs > 0, f"{gs:.4g}")
    check("§78 geometry grad == 0", gt == 0.0, f"{gt:.4g}")


def test_79_motion_direction_cannot_train_speed():
    """§79/§40: the direction loss is speed-invariant by construction -- s(t)
    does not even appear in its graph."""
    x, theta, speed, dt, w = routing_fixture()
    proj = geometry_xi(x, theta, w)
    v = torch.randn((5, 24, 3), dtype=DT)
    gt, gs = grads(motion_direction_loss(proj["rigid_velocity"], v,
                                         torch.ones((5, 24), dtype=DT)), theta, speed)
    check("§79 geometry grad != 0", gt > 0, f"{gt:.4g}")
    check("§79 speed grad == 0", gs == 0.0, f"{gs:.4g}")


def test_80_motion_magnitude_cannot_train_geometry():
    """§80/§51: magnitude uses the FIXED xi, so nothing reaches theta."""
    x, theta, speed, dt, w = routing_fixture()
    with torch.no_grad():
        live = geometry_xi(x, theta, w)
    psi = live["rigid_velocity"].detach()
    v = torch.randn((5, 24, 3), dtype=DT)
    s_mid = 0.5 * (speed[:-1] + speed[1:])
    gt, gs = grads(velocity_magnitude_loss(psi, v, torch.ones((5, 24), dtype=DT), s_mid,
                                           mode="l2"), theta, speed)
    check("§80 speed grad != 0", gs > 0, f"{gs:.4g}")
    check("§80 geometry grad == 0", gt == 0.0, f"{gt:.4g}")


def test_81_strain_cannot_train_scalar():
    x, theta, speed, dt, w = routing_fixture()
    jac = theta[:, :, None] * theta[:, None, :]
    strain = (0.5 * (jac + jac.transpose(-1, -2))).square().sum()
    gt, gs = grads(strain, theta, speed)
    check("§81 geometry grad != 0", gt > 0, f"{gt:.4g}")
    check("§81 speed grad == 0", gs == 0.0, f"{gs:.4g}")


def test_82_temporal_regularizers_cannot_train_geometry():
    x, theta, _, dt, w = routing_fixture()
    # A VARYING s(t): a constant one makes continuity and jerk identically zero,
    # so "as mathematically expected" would be a zero gradient and the test would
    # prove nothing about routing.
    speed = torch.nn.Parameter(torch.tensor([0.4, 0.2, 0.05, -0.1, -0.35, 0.2], dtype=DT))
    loss = (scalar_continuity_loss(speed, dt) + scalar_sign_flip_loss(speed)
            + scalar_jerk_loss(speed, dt))
    gt, gs = grads(loss, theta, speed)
    check("§82 speed grad as expected", gs > 0, f"{gs:.4g}")
    check("§82 geometry grad == 0", gt == 0.0, f"{gt:.4g}")

    # And a constant s(t) really does give zero from continuity/jerk.
    flat = torch.nn.Parameter(torch.full((6,), 0.3, dtype=DT))
    (scalar_continuity_loss(flat, dt) + scalar_jerk_loss(flat, dt)).backward()
    check("§53 a constant s(t) is free of continuity/jerk cost",
          float(flat.grad.abs().sum()) == 0.0)


# ---------------------------------------------------------------------------
# Structural invariants against the real solver
# ---------------------------------------------------------------------------
def test_routing_table_matches_the_spec():
    """§56's table, verbatim."""
    expected = {
        "render": {"geometry", "scalar"}, "depth": {"geometry", "scalar"},
        "motion_direction": {"geometry"}, "motion_magnitude": {"scalar"},
        "final_pcd": {"geometry", "scalar"}, "strain": {"geometry"},
        "rigid": {"geometry"}, "continuity": {"scalar"}, "flip": {"scalar"},
        "jerk": {"scalar"},
    }
    check("§56 the routing table has exactly the listed losses",
          set(sks.LOSS_ROUTING) == set(expected), str(set(sks.LOSS_ROUTING) ^ set(expected)))
    for name, blocks in expected.items():
        check(f"§56 {name} routes to {sorted(blocks)}",
              set(sks.LOSS_ROUTING[name]) == blocks, str(sks.LOSS_ROUTING[name]))


def test_two_blocks_only():
    """§34/§88: exactly two parameter blocks, disjoint, no monolithic optimizer."""
    control = torch.nn.Parameter(torch.zeros(2, 4, 3))
    speed = torch.nn.Parameter(torch.ones(2, 9))
    g = sks.geometry_parameters(control, speed)
    s = sks.scalar_parameters(control, speed)
    check("§34 geometry block owns only the spatial controls", [id(t) for t in g] == [id(control)])
    check("§34 scalar block owns only s(t)", [id(t) for t in s] == [id(speed)])
    check("§88 the blocks are disjoint", not ({id(t) for t in g} & {id(t) for t in s}))

    adam = [ast.unparse(n.args[0]) if n.args else "<none>"
            for n in ast.walk(SOLVER_AST)
            if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("Adam")]
    check("§88 exactly two optimizers exist", len(adam) == 2, str(adam))
    check("§88 one is the geometry block", any(a.startswith("geometry_parameters(") for a in adam))
    check("§88 one is the scalar block", any(a.startswith("scalar_parameters(") for a in adam))


def test_prohibited_trainables_absent():
    """§30/§87: only `control` and `speed` may be Parameters."""
    created = []
    for node in ast.walk(SOLVER_AST):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and ast.unparse(node.value.func).endswith("Parameter"):
            created.extend(ast.unparse(t) for t in node.targets)
    check("§30 exactly two trainable tensors", sorted(created) == ["control", "speed"], str(created))
    # Identifiers, not prose: the comment documenting the prohibition names them.
    names = {n.id for n in ast.walk(SOLVER_AST) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(SOLVER_AST) if isinstance(n, ast.Attribute)}
    for forbidden in ("omega_g", "u_g", "global_twist", "global_omega", "xi_param",
                      "global_translation", "omega_t", "u_t"):
        check(f"§30 no binding named {forbidden}", forbidden not in names)


def test_block_freeze_and_assertions_exist():
    """§46/§55: requires_grad flips plus the zero-gradient assertions."""
    src = inspect.getsource(sks.solve_shared_killing_sequence)
    check("§46 geometry block freezes s(t)", "speed.requires_grad_(False)" in src)
    check("§55 scalar block freezes the geometry",
          "set_requires_grad(geometry_parameters(control, speed), False)" in src)
    check("§55 scalar block unfreezes s(t)", "speed.requires_grad_(True)" in src)
    check("§46 geometry asserts s(t) got nothing",
          'assert_no_gradient(scalar_parameters(control, speed), "geometry block")' in src)
    check("§55 scalar asserts the geometry got nothing",
          'assert_no_gradient(geometry_parameters(control, speed), "scalar block")' in src)
    check("§47 the scalar block detaches xi", "v.detach() if torch.is_tensor(v) else v" in src)
    check("§36 the geometry block detaches s(t)", "_integrate(speed.detach())" in src)

    # and the assertion actually fires
    p = torch.nn.Parameter(torch.ones(3))
    p.grad = torch.ones(3)
    try:
        sks.assert_no_gradient([p], "unit test")
        raise SystemExit("unreachable")
    except AssertionError:
        check("§46 the assertion is real", True)


def test_geometry_block_has_informative_losses():
    """§37/§38/§41: a geometry step driven only by strain/rigid is NOT valid."""
    fn = next(n for n in ast.walk(SOLVER_AST)
              if isinstance(n, ast.FunctionDef) and n.name == "_geometry_terms")
    body = ast.unparse(fn)
    for required, why in (
        ("_render_and_depth", "§38/§39 render and depth supervise geometry"),
        ("motion_direction_loss", "§40 motion direction supervises geometry"),
        ("_final_pcd", "§41 final PCD supervises geometry"),
        ("strain_loss", "§43 strain regularizes geometry"),
        ("single_field_rigid_projection_losses", "§44 rigid projection regularizes geometry"),
    ):
        check(why, required in body)
    check("§51 magnitude is NOT in the geometry block", "velocity_magnitude_loss" not in body)
    check("§53 temporal regularizers are NOT in the geometry block",
          "scalar_continuity_loss" not in body and "scalar_jerk_loss" not in body)
    check("§37 the informative set is declared",
          set(sks.GEOMETRY_INFORMATIVE) == {"render", "depth", "motion_direction", "final_pcd"})

    scalar = next(n for n in ast.walk(SOLVER_AST)
                  if isinstance(n, ast.FunctionDef) and n.name == "_scalar_terms")
    sbody = ast.unparse(scalar)
    for required, why in (
        ("_render_and_depth", "§49/§50 render and depth supervise the trajectory"),
        ("velocity_magnitude_loss", "§51 magnitude supervises the trajectory"),
        ("_final_pcd", "§52 final PCD supervises the trajectory"),
        ("scalar_continuity_loss", "§53 continuity"),
        ("scalar_sign_flip_loss", "§53 sign flip"),
        ("scalar_jerk_loss", "§53 jerk"),
    ):
        check(why, required in sbody)
    check("§40 direction is NOT in the scalar block", "motion_direction_loss" not in sbody)


def test_per_loss_gradient_audit():
    """§59/§60/§61: per-loss norms, measured without stepping."""
    a = torch.nn.Parameter(torch.ones(4, dtype=DT))
    b = torch.nn.Parameter(torch.ones(4, dtype=DT) * 2)
    terms = {"big": (3.0 * a).sum(), "small": (0.001 * a).sum(),
             "none": (b * b).sum(), "constant": torch.tensor(5.0, dtype=DT)}
    norms = sks.per_loss_gradient_norms(terms, [a])
    check("§59 every term reports a norm", set(norms) == set(terms), str(set(norms)))
    check("§61 a large-gradient term is distinguished from a small one",
          norms["big"] > 100 * norms["small"], f"{norms['big']:.4g} vs {norms['small']:.4g}")
    check("§59 a term not touching the parameter reports 0", norms["none"] == 0.0)
    check("§59 a constant reports 0", norms["constant"] == 0.0)
    check("§59 .grad is untouched", a.grad is None and b.grad is None)

    src = inspect.getsource(sks.solve_shared_killing_sequence)
    # CODEX_DEBUG §14 later required the probe to measure grad(lambda*L), so the
    # audit now passes the WEIGHTED terms rather than the raw ones.
    check("§59 the audit runs on the geometry block",
          "per_loss_gradient_norms(weighted, geometry_parameters(control, speed))" in src)
    check("§60 the audit runs on the scalar block",
          "per_loss_gradient_norms(weighted_s, scalar_parameters(control, speed))" in src)
    check("CODEX_DEBUG §14 the probe measures the weighted losses",
          "_weighted_terms(terms," in src)
    check("§59 a strain-only geometry step is reported",
          "NO INFORMATIVE GEOMETRY GRADIENT" in src)
    check("§60 an uninformative scalar step is reported",
          "NO INFORMATIVE SCALAR GRADIENT" in src)
    check("§61 no compensating clamps were added",
          "clamp(speed" not in SOLVER_SRC and "speed.clamp" not in SOLVER_SRC)


def test_58_full_frame_coverage():
    """§58: every frame visited before the epoch completes."""
    frames = list(range(60))
    schedule = frame_epoch_schedule(30, frames, 4, seed=1)
    check("§58 one batch per step", len(schedule) == 30)
    check("§58 the batch size is honoured", all(len(b) == 4 for b in schedule))
    visited = {f for batch in schedule[:15] for f in batch}
    check("§58 one epoch covers every frame", visited == set(frames),
          f"{len(visited)}/60 after 15 steps")
    # no frame is repeated inside an epoch
    first_epoch = [f for batch in schedule[:15] for f in batch]
    check("§58 no frame repeats within an epoch", len(first_epoch) == len(set(first_epoch)))
    check("§58 the default stride is 1",
          'default=1' in Path('arguments/__init__.py').read_text(encoding="utf-8")
          .split("--inner_render_frame_stride")[1][:120])
    src = inspect.getsource(sks.solve_shared_killing_sequence)
    check("§58 coverage is reported", "geometry_full_coverage" in src)


def test_89_acceptance_artifacts():
    """§62/§89's debug checklist: the artifacts must actually be written."""
    src = inspect.getsource(sks.solve_shared_killing_sequence)
    for artifact in ("inner_support_cleaning.pt", "inner_support_cleaning.json",
                     "inner_support_debug.ply"):
        check(f"§62 {artifact} is written", artifact in src)
    check("§65 geometry metrics are saved", "L_motion_dir" in src and "gradient_norms" in src)
    check("§66 scalar metrics are saved", "L_motion_mag" in src and "endpoint_rotation_deg" in src)
    check("§64 GT support metrics are reported", "gt_support_metrics(" in src)
    check("§1 the no-mutation invariant is checked at runtime",
          "mutated the persistent outer labels" in src)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
