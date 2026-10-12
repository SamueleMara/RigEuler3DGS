"""Synthetic acceptance tests for the motion-coherence static/dynamic filter.

CODEX_IMPLEMENT_MOTION_COHERENCE_STATIC_DYNAMIC_FILTER.md sections 36 (S1-S7),
37 (the differential identity) and 46 (leakage).

Renderer-independent by construction: every case builds free trajectories
directly, which is the point -- the filter's claim is about trajectory
geometry, so it must be testable without any imaging at all.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.motion_coherence_support import (  # noqa: E402
    UNRESOLVED, canonical_knn_graph, differential_strain_from_velocity,
    edge_strain, propagate_to_gaussians, run_motion_coherence,
)
from estimator.manifold import so3_exp  # noqa: E402

PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# scene builders: two blobs, one static, one moving
# ---------------------------------------------------------------------------

def _blobs(n_static=120, n_dyn=80, seed=0, gap=3.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand((n_static, 3), generator=g, dtype=torch.float64) * 2.0
    b = torch.rand((n_dyn, 3), generator=g, dtype=torch.float64) * 2.0
    b[:, 0] += gap
    x0 = torch.cat([a, b])
    is_dyn = torch.zeros(x0.shape[0], dtype=torch.bool)
    is_dyn[n_static:] = True
    # one Gaussian per supernode keeps the test about the filter, not the
    # reduction; the reduction is exercised separately below.
    sid = torch.arange(x0.shape[0])
    return x0, is_dyn, sid


def _apply(x0, is_dyn, transforms):
    """transforms: list of (R, p) applied to the dynamic blob only."""
    out = []
    for (R, p) in transforms:
        x = x0.clone()
        x[is_dyn] = x0[is_dyn] @ R.T + p
        out.append(x)
    return out


def _run(positions, sid, knn=10):
    return run_motion_coherence(positions, sid, knn=knn, verbose=False)


def _score(res, is_dyn):
    """Dynamic-class IoU of the prediction against the construction."""
    if res["status"] == UNRESOLVED:
        return None
    pred = torch.zeros_like(is_dyn)
    pred[res["dynamic_nodes"]] = True
    tp = int((pred & is_dyn).sum())
    fp = int((pred & ~is_dyn).sum())
    fn = int((~pred & is_dyn).sum())
    return tp / max(tp + fp + fn, 1)


# ---------------------------------------------------------------------------
# S1-S7
# ---------------------------------------------------------------------------

def test_S1_static_plus_translation():
    x0, dyn, sid = _blobs(seed=1)
    I = torch.eye(3, dtype=torch.float64)
    pos = _apply(x0, dyn, [(I, torch.tensor([0.0, 0.0, 0.0])
                            + torch.tensor([0.08 * t, 0.0, 0.0],
                                           dtype=torch.float64))
                           for t in range(12)])
    iou = _score(_run(pos, sid), dyn)
    check("S1 static + translated rigid body", iou is not None and iou > 0.95,
          f"dynamic IoU {iou:.3f}")


def test_S2_static_plus_revolute_offset_axis():
    x0, dyn, sid = _blobs(seed=2)
    e = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    a = x0[dyn].mean(0) + torch.tensor([1.2, 0.0, 0.0], dtype=torch.float64)
    tf = []
    for t in range(12):
        R = so3_exp((0.06 * t * e).reshape(1, 3))[0].double()
        tf.append((R, a - R @ a))
    iou = _score(_run(_apply(x0, dyn, tf), sid), dyn)
    check("S2 static + revolute about an offset axis",
          iou is not None and iou > 0.95, f"dynamic IoU {iou:.3f}")


def test_S3_speed_change_stop_restart_reversal():
    x0, dyn, sid = _blobs(seed=3)
    I = torch.eye(3, dtype=torch.float64)
    s = [0.0, .02, .06, .14, .14, .14, .14, .20, .26, .20, .10, .02]
    pos = _apply(x0, dyn, [(I, torch.tensor([v, 0.0, 0.0],
                                            dtype=torch.float64)) for v in s])
    iou = _score(_run(pos, sid), dyn)
    check("S3 speed change / stop / restart / reversal",
          iou is not None and iou > 0.95, f"dynamic IoU {iou:.3f}")


def test_S4_chaotic_false_dynamics():
    """A static region given independent per-point jitter of an amplitude
    COMPARABLE to the real motion. A displacement rule cannot tell these apart;
    metric coherence must."""
    x0, dyn, sid = _blobs(seed=4)
    g = torch.Generator().manual_seed(99)
    I = torch.eye(3, dtype=torch.float64)
    true = _apply(x0, dyn, [(I, torch.tensor([0.08 * t, 0.0, 0.0],
                                             dtype=torch.float64))
                            for t in range(12)])
    amp = 0.08 * 11 / 2.0
    false_idx = torch.nonzero(~dyn, as_tuple=False).reshape(-1)[:60]
    pos = []
    for t, x in enumerate(true):
        y = x.clone()
        if t > 0:
            y[false_idx] += amp * torch.randn(
                (false_idx.numel(), 3), generator=g, dtype=torch.float64)
        pos.append(y)
    res = _run(pos, sid)
    iou = _score(res, dyn)
    # The chaotic points must NOT be absorbed into the dynamic component.
    pred = torch.zeros_like(dyn)
    if res["status"] != UNRESOLVED:
        pred[res["dynamic_nodes"]] = True
    leaked = int((pred[false_idx]).sum())
    check("S4 chaotic false dynamics are not a coherent dynamic component",
          leaked <= 0.1 * false_idx.numel(),
          f"{leaked}/{false_idx.numel()} chaotic points leaked, IoU "
          f"{iou if iou is None else round(iou, 3)}")


def test_S5_smooth_nonrigid_shear():
    """Low temporal jitter, finite metric strain. Must fail rigid coherence:
    the sheared block's internal edges must score worse than a rigid block's."""
    x0, dyn, sid = _blobs(seed=5)
    pos = []
    for t in range(12):
        x = x0.clone()
        k = 0.03 * t
        x[dyn, 0] = x0[dyn, 0] + k * x0[dyn, 1]       # smooth shear
        pos.append(x)
    traj = torch.stack(pos)
    g = canonical_knn_graph(traj[0], k=10)
    st = edge_strain(traj, g["edge_i"], g["edge_j"])
    inside = dyn[g["edge_i"]] & dyn[g["edge_j"]]
    outside = (~dyn[g["edge_i"]]) & (~dyn[g["edge_j"]])
    c_shear = float(st["C"][inside].median())
    c_rigid = float(st["C"][outside].median())
    j_shear = float(st["J"][inside].median())
    check("S5 smooth shear fails rigid coherence",
          c_shear > 50 * max(c_rigid, 1e-12),
          f"C shear {c_shear:.3e} vs rigid {c_rigid:.3e}")
    check("S5 and it is smooth, not jitter",
          j_shear < c_shear, f"J {j_shear:.3e} < C {c_shear:.3e}")


def test_S6_move_and_return():
    x0, dyn, sid = _blobs(seed=6)
    I = torch.eye(3, dtype=torch.float64)
    s = [0.0, .05, .10, .15, .20, .25, .20, .15, .10, .05, .0, .0]
    pos = _apply(x0, dyn, [(I, torch.tensor([v, 0.0, 0.0],
                                            dtype=torch.float64)) for v in s])
    iou = _score(_run(pos, sid), dyn)
    check("S6 move and return to start is still recovered",
          iou is not None and iou > 0.95, f"dynamic IoU {iou:.3f}")


def test_S7_tiny_motion_points_near_the_axis():
    """Points near a revolute axis barely move. They must stay DYNAMIC through
    coherence connectivity, not be called static for having small displacement."""
    g = torch.Generator().manual_seed(7)
    static = torch.rand((120, 3), generator=g, dtype=torch.float64) * 2.0
    arm = torch.rand((80, 3), generator=g, dtype=torch.float64)
    arm[:, 0] = arm[:, 0] * 2.5 + 3.0          # spread away from the axis
    arm[:, 1] *= 0.6
    x0 = torch.cat([static, arm])
    dyn = torch.zeros(x0.shape[0], dtype=torch.bool)
    dyn[120:] = True
    sid = torch.arange(x0.shape[0])
    e = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    a = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)   # axis at the base
    tf = []
    for t in range(12):
        R = so3_exp((0.05 * t * e).reshape(1, 3))[0].double()
        tf.append((R, a - R @ a))
    pos = _apply(x0, dyn, tf)
    disp = (pos[-1] - pos[0]).norm(dim=1)
    near = torch.nonzero(dyn & (disp < float(disp[dyn].median()) * 0.25),
                         as_tuple=False).reshape(-1)
    res = _run(pos, sid)
    pred = torch.zeros_like(dyn)
    if res["status"] != UNRESOLVED:
        pred[res["dynamic_nodes"]] = True
    kept = int(pred[near].sum())
    check("S7 near-axis tiny-motion points stay DYNAMIC",
          near.numel() > 0 and kept >= 0.9 * near.numel(),
          f"{kept}/{near.numel()} kept (their displacement is "
          f"{float(disp[near].max()):.4f} vs body median "
          f"{float(disp[dyn].median()):.4f})")


# ---------------------------------------------------------------------------
# Section 37: the differential identity
# ---------------------------------------------------------------------------

def test_S37_differential_identity():
    g = torch.Generator().manual_seed(11)
    x0 = torch.rand((80, 3), generator=g, dtype=torch.float64) * 2.0
    gr = canonical_knn_graph(x0, k=8)
    worst_match, worst_rigid = 0.0, 0.0
    for dt in (1e-2, 1e-3, 1e-4):
        # smooth non-rigid field, so the identity is tested where it is not
        # trivially zero on both sides
        def f(t):
            x = x0.clone()
            x[:, 0] = x0[:, 0] + 0.5 * t * x0[:, 1]
            return x
        T = 4
        traj = torch.stack([f(k * dt) for k in range(T)])
        st = edge_strain(traj, gr["edge_i"], gr["edge_j"],
                         times=torch.arange(T, dtype=torch.float64) * dt)
        r = traj[0, gr["edge_j"]] - traj[0, gr["edge_i"]]
        v = (f(dt) - f(0.0)) / dt
        dv = v[gr["edge_j"]] - v[gr["edge_i"]]
        want = differential_strain_from_velocity(r, dv)
        got = st["epsilon"][0]
        rel = float((got - want).abs().max() / want.abs().max().clamp_min(1e-30))
        worst_match = max(worst_match, rel) if dt == 1e-4 else worst_match

        # exact rigid motion: both forms must go to zero
        e = torch.tensor([0.1, -0.3, 0.95], dtype=torch.float64)
        e = e / e.norm()
        rig = []
        for k in range(T):
            R = so3_exp((0.7 * k * dt * e).reshape(1, 3))[0].double()
            rig.append(x0 @ R.T + torch.tensor([0.3, -0.1, 0.2],
                                               dtype=torch.float64) * k * dt)
        strig = edge_strain(torch.stack(rig), gr["edge_i"], gr["edge_j"],
                            times=torch.arange(T, dtype=torch.float64) * dt)
        worst_rigid = max(worst_rigid, float(strig["epsilon"].abs().max()))
    check("S37 finite log-strain rate matches r^T(v_j-v_i)/|r|^2",
          worst_match < 1e-3, f"rel {worst_match:.2e} at dt=1e-4")
    check("S37 exact rigid motion gives epsilon -> 0",
          worst_rigid < 1e-9, f"max |epsilon| {worst_rigid:.2e}")


# ---------------------------------------------------------------------------
# Section 46: leakage
# ---------------------------------------------------------------------------

def test_S46_no_leakage():
    import ast
    src_path = REPO / "estimator" / "motion_coherence_support.py"
    src = src_path.read_text()
    tree = ast.parse(src)
    banned_mod = {"estimator.moving_support", "moving_support"}
    banned_name = {"otsu_threshold", "select_moving_support",
                   "trajectory_excursion"}
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if (node.module or "") in banned_mod:
                bad.append(f"from {node.module}")
            for a in node.names:
                if a.name in banned_name:
                    bad.append(f"import {a.name}")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in banned_mod:
                    bad.append(f"import {a.name}")
        if isinstance(node, (ast.Name, ast.Attribute)):
            nm = getattr(node, "id", None) or getattr(node, "attr", "")
            if nm in banned_name:
                bad.append(f"use of {nm}")
    check("S46 the coherence module never imports the Otsu selector", not bad,
          "; ".join(sorted(set(bad))))
    for word in ("ground_truth", "load_static_dynamic_gt", "points3D_parts",
                 "gt_label", "gt_parts"):
        check(f"S46 no GT reference: {word!r}", word not in src)
    # Displacement must not drive the partition: the only displacement-like
    # quantity allowed is INSIDE validate_component, which runs after the cut.
    # Check the CODE of the partition, not its prose: the docstring says in
    # words that displacement is not used, so a raw substring scan flags the
    # very sentence that documents the guarantee.
    fn = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
              and f.name == "normalized_cut")
    ids = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            ids.add(node.id)
        elif isinstance(node, ast.Attribute):
            ids.add(node.attr)
    for word in ("median", "topk", "displacement", "excursion", "otsu",
                 "quantile", "kthvalue"):
        check(f"S46 the partition calls nothing named {word!r}",
              word not in ids)
    # and it must take no displacement-derived argument
    check("S46 the partition takes only the graph and its affinity",
          [a.arg for a in fn.args.args] == ["edge_i", "edge_j", "W", "n"],
          str([a.arg for a in fn.args.args]))


def test_S46_supernode_reduction_and_propagation():
    """The reduction path, with the repo's OWN spatial supernodes.

    Grouping Gaussians arbitrarily would not test anything real: Phase-1
    supernodes are spatially compact, so a supernode lies on one side of the
    joint and its centroid trajectory is meaningful. A random grouping puts
    members of both bodies in one supernode, and no method can then label it.
    """
    from estimator.supernodes import build_supernodes

    g = torch.Generator().manual_seed(13)
    a = torch.rand((400, 3), generator=g, dtype=torch.float64) * 2.0
    b = torch.rand((260, 3), generator=g, dtype=torch.float64) * 2.0
    b[:, 0] += 4.0
    x0 = torch.cat([a, b])
    is_dyn_pt = torch.zeros(x0.shape[0], dtype=torch.bool)
    is_dyn_pt[400:] = True

    part = build_supernodes(x0.float(), target_size=16, seed=0)
    sid = part.supernode_id.long()
    pos = [x0.clone()]
    for t in range(1, 10):
        x = x0.clone()
        x[is_dyn_pt] += torch.tensor([0.06 * t, 0.0, 0.0], dtype=torch.float64)
        pos.append(x)

    res = run_motion_coherence(pos, sid, knn=8, verbose=False)
    gi = propagate_to_gaussians(res["dynamic_nodes"], sid)
    pred = torch.zeros(x0.shape[0], dtype=torch.bool)
    pred[gi] = True
    tp = int((pred & is_dyn_pt).sum())
    fp = int((pred & ~is_dyn_pt).sum())
    fn_ = int((~pred & is_dyn_pt).sum())
    iou = tp / max(tp + fp + fn_, 1)
    check("S46 supernode reduction + propagation recovers the moving body",
          iou > 0.95, f"Gaussian-level dynamic IoU {iou:.3f} "
                      f"({int(sid.max()) + 1} supernodes, tp {tp} fp {fp} fn {fn_})")


# ===========================================================================
# Section 20: S8-S13, the regressions the two-way Ncut failed
# ===========================================================================

from estimator.coherence_extraction import (  # noqa: E402
    INCOHERENT, MOVING_RIGID, NO_MOVING, candidate_hierarchy,
    extract_moving_component,
)
from estimator.motion_coherence_support import (  # noqa: E402
    canonical_knn_graph as _knn, edge_strain as _strain,
)


def _extract(positions, knn=10, **kw):
    """Run the v2 extractor on raw per-node trajectories."""
    traj = torch.stack([torch.as_tensor(p).double() for p in positions])
    g = _knn(traj[0], k=knn)
    st = _strain(traj, g["edge_i"], g["edge_j"])
    return extract_moving_component(traj, g["edge_i"], g["edge_j"], st["C"],
                                    traj.shape[1], verbose=False, **kw), g, st


def _iou(res, is_dyn):
    pred = torch.tensor([l == MOVING_RIGID for l in res["labels"]])
    tp = int((pred & is_dyn).sum())
    fp = int((pred & ~is_dyn).sum())
    fn = int((~pred & is_dyn).sum())
    return tp / max(tp + fp + fn, 1), pred


def _small_body(frac, n_total=240, seed=0, rot=False):
    g = torch.Generator().manual_seed(seed)
    n_dyn = max(6, int(round(frac * n_total)))
    n_sta = n_total - n_dyn
    a = torch.rand((n_sta, 3), generator=g, dtype=torch.float64) * 4.0
    b = torch.rand((n_dyn, 3), generator=g, dtype=torch.float64) * 1.0
    b[:, 0] += 5.0
    x0 = torch.cat([a, b])
    dyn = torch.zeros(n_total, dtype=torch.bool)
    dyn[n_sta:] = True
    pos = []
    e = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    axis = x0[dyn].mean(0) + torch.tensor([1.5, 0.0, 0.0], dtype=torch.float64)
    for t in range(12):
        x = x0.clone()
        if rot:
            R = so3_exp((0.06 * t * e).reshape(1, 3))[0].double()
            x[dyn] = x0[dyn] @ R.T + (axis - R @ axis)
        else:
            x[dyn] = x0[dyn] + torch.tensor([0.07 * t, 0.0, 0.0],
                                            dtype=torch.float64)
        pos.append(x)
    return pos, dyn


def test_S8_small_translating_body():
    pos, dyn = _small_body(0.08, seed=81)
    res, _g, _s = _extract(pos)
    iou, pred = _iou(res, dyn)
    check("S8 an 8% translating body is recovered without a balanced split",
          iou > 0.9, f"IoU {iou:.3f}, predicted {100 * pred.double().mean():.1f}% "
                     f"moving vs {100 * dyn.double().mean():.1f}% true")


def test_S9_small_revolute_body():
    pos, dyn = _small_body(0.15, seed=91, rot=True)
    res, _g, _s = _extract(pos)
    iou, pred = _iou(res, dyn)
    check("S9 a 15% revolute body is recovered",
          iou > 0.9, f"IoU {iou:.3f}, predicted {100 * pred.double().mean():.1f}%")


def test_S10_chaotic_outliers_stay_ambiguous():
    """25% of the scene given independent per-frame noise. They must NOT be
    forced into the moving body."""
    pos, dyn = _small_body(0.15, seed=101)
    g = torch.Generator().manual_seed(7)
    n = pos[0].shape[0]
    chaos = torch.nonzero(~dyn, as_tuple=False).reshape(-1)[: int(0.25 * n)]
    amp = 0.07 * 11 / 2
    noisy = [pos[0]]
    for t, x in enumerate(pos[1:], 1):
        y = x.clone()
        y[chaos] += amp * torch.randn((chaos.numel(), 3), generator=g,
                                      dtype=torch.float64)
        noisy.append(y)
    res, _g, _s = _extract(noisy)
    pred = torch.tensor([l == MOVING_RIGID for l in res["labels"]])
    leaked = int(pred[chaos].sum())
    check("S10 25% chaotic outliers are not forced into the moving body",
          leaked <= 0.1 * chaos.numel(),
          f"{leaked}/{chaos.numel()} leaked; "
          f"{res['counts'][INCOHERENT]} nodes left ambiguous")


def test_S11_small_hinge_next_to_a_large_static_body():
    """Cross-body kNN edges exist because the hinge touches the static body."""
    g = torch.Generator().manual_seed(111)
    sta = torch.rand((200, 3), generator=g, dtype=torch.float64) * 3.0
    hinge = torch.rand((30, 3), generator=g, dtype=torch.float64) * 0.8
    hinge[:, 0] += 2.9                      # overlapping the static block
    x0 = torch.cat([sta, hinge])
    dyn = torch.zeros(x0.shape[0], dtype=torch.bool)
    dyn[200:] = True
    e = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    a = torch.tensor([2.9, 0.0, 0.0], dtype=torch.float64)
    pos = []
    for t in range(12):
        R = so3_exp((0.07 * t * e).reshape(1, 3))[0].double()
        x = x0.clone()
        x[dyn] = x0[dyn] @ R.T + (a - R @ a)
        pos.append(x)
    res, _g, _s = _extract(pos)
    iou, _p = _iou(res, dyn)
    check("S11 a small hinge adjacent to a large static body is recovered",
          iou > 0.8, f"IoU {iou:.3f}")


def test_S12_one_accidental_low_C_cross_edge():
    """A single cheap cross edge must not chain the two bodies together."""
    pos, dyn = _small_body(0.12, seed=121)
    traj = torch.stack(pos)
    g = _knn(traj[0], k=10)
    st = _strain(traj, g["edge_i"], g["edge_j"])
    C = st["C"].clone()
    cross = torch.nonzero(dyn[g["edge_i"]] != dyn[g["edge_j"]],
                          as_tuple=False).reshape(-1)
    check("S12 the scene really has cross edges", cross.numel() > 0,
          f"{cross.numel()} cross edges")
    C[cross[0]] = 0.0                       # one accidental perfect edge
    res = extract_moving_component(traj, g["edge_i"], g["edge_j"], C,
                                   traj.shape[1], verbose=False)
    iou, pred = _iou(res, dyn)
    check("S12 one accidental low-C cross edge does not chain the bodies",
          iou > 0.8, f"IoU {iou:.3f}, predicted "
                     f"{100 * pred.double().mean():.1f}% moving")


def test_S13_static_body_with_tiny_coherent_drift():
    """Everything drifts rigidly by a whisper. There is no articulated motion,
    so the honest answer is no moving component -- not 'the half that drifted
    slightly more'."""
    g = torch.Generator().manual_seed(131)
    x0 = torch.rand((200, 3), generator=g, dtype=torch.float64) * 3.0
    pos = [x0 + torch.tensor([1e-4 * t, 0.0, 0.0], dtype=torch.float64)
           for t in range(12)]
    res, _g, _s = _extract(pos)
    moving = res["counts"][MOVING_RIGID]
    check("S13 a uniformly drifting static body yields no large moving body",
          res["status"] == NO_MOVING or moving <= 0.25 * 200,
          f"status {res['status']}, {moving}/200 marked moving")


def test_S20_hierarchy_has_no_balance_prior():
    """Structural: the candidate generator sees only the graph and C."""
    import ast
    src = (REPO / "estimator" / "coherence_extraction.py").read_text()
    tree = ast.parse(src)
    fn = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
              and f.name == "candidate_hierarchy")
    # The only extra arguments permitted are structural minimums (how many
    # nodes make a component, how many edges make a merge). Neither is derived
    # from the data, so neither can encode a target fraction.
    args_ = [a.arg for a in fn.args.args]
    check("S20 the candidate hierarchy takes only the graph, C and structural "
          "minimums",
          args_ == ["edge_i", "edge_j", "C", "n", "min_nodes",
                    "min_merge_support"], str(args_))
    ids = {getattr(x, "id", None) or getattr(x, "attr", "")
           for x in ast.walk(fn) if isinstance(x, (ast.Name, ast.Attribute))}
    for w in ("fiedler", "eigh", "ncut", "otsu", "median", "topk",
              "displacement"):
        check(f"S20 the hierarchy calls nothing named {w!r}", w not in ids)
    # Check imports, not prose: the module docstring states in words that it
    # uses no Otsu, so a substring scan flags the sentence documenting the
    # guarantee.
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
    bad = [m for m in mods
           if "moving_support" in m or "otsu" in m.lower()
           or "optimized_gt" in m or "eval" in m]
    check("S20 the extractor imports no selector, GT or evaluation module",
          not bad, "; ".join(sorted(bad)))


def test_iekf_grouping_is_independent():
    """The Phase-1 IEKF grouping must not follow the coherence-graph flag.

    They shared --supernode-size until the count-based coherence default
    landed. That dropped it to 0, build_supernodes turned `n // max(ts, 1)`
    into one group per Gaussian, and Phase 1 went from 384 pose blocks to
    24,582 on washer -- two orders of magnitude slower, with output that
    still looked right.
    """
    import argparse, io as _io, re, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from estimator.supernodes import build_supernodes
    import torch

    src = _io.open(Path(__file__).resolve().parents[1]
                   / "tools" / "run_motion_coherence_support.py").read()
    assert "supernode_size=args.iekf_supernode_size" in src, \
        "the Phase-1 IEKF must read its own grouping flag"
    assert "supernode_size=args.supernode_size," not in src, \
        "the coherence-graph flag must not reach the Phase-1 IEKF"
    assert re.search(r'--iekf-supernode-size".*?default=64', src, re.S), \
        "the IEKF grouping default must stay 64"

    x = torch.randn(4096, 3)
    degenerate = int(build_supernodes(x, target_size=0, seed=0).supernode_id.max()) + 1
    healthy = int(build_supernodes(x, target_size=64, seed=0).supernode_id.max()) + 1
    assert degenerate == 4096, degenerate
    assert healthy == 64, healthy
    print("PASS Phase-1 IEKF grouping is independent of the coherence graph")


def main():
    for fn in (test_iekf_grouping_is_independent,
               test_S1_static_plus_translation,
               test_S2_static_plus_revolute_offset_axis,
               test_S3_speed_change_stop_restart_reversal,
               test_S4_chaotic_false_dynamics,
               test_S5_smooth_nonrigid_shear,
               test_S6_move_and_return,
               test_S7_tiny_motion_points_near_the_axis,
               test_S37_differential_identity,
               test_S46_no_leakage,
               test_S46_supernode_reduction_and_propagation,
               test_S8_small_translating_body,
               test_S9_small_revolute_body,
               test_S10_chaotic_outliers_stay_ambiguous,
               test_S11_small_hinge_next_to_a_large_static_body,
               test_S12_one_accidental_low_C_cross_edge,
               test_S13_static_body_with_tiny_coherent_drift,
               test_S20_hierarchy_has_no_balance_prior):
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed")


if __name__ == "__main__":
    main()


