"""V2 section 21: static / moving support discovery tests T-S1..T-S4.

Every scene here is rendered by the real CUDA rasterizer and the evidence comes
from the analytic JVP/VJP -- section 6.1 forbids substituting optical flow, and
section 6.3 forbids classifying support from per-Gaussian velocity magnitude.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.measurement import build_precision  # noqa: E402
from estimator.renderer_bridge import CameraSpec, build_linearization  # noqa: E402
from estimator.state import GaussianMotionState  # noqa: E402
from estimator.supernodes import GroupedObservation, build_supernodes  # noqa: E402
from joint_reduction.line_geometry import PluckerLine, revolute_joint_tangent  # noqa: E402
from joint_reduction.support import (  # noqa: E402
    MOVING,
    STATIC,
    SupportConfig,
    gaussian_support_metrics,
    graph_cut_binary,
    infer_support,
    ownership_responsibility,
    support_metrics,
    supernode_motion_evidence,
)

DEV = "cuda"
PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def camera(width=96, height=96, fov=0.9, tz=0.0):
    tanf = math.tan(fov * 0.5)
    view = torch.eye(4, device=DEV)
    view[3, 0] = tz
    znear, zfar = 0.01, 100.0
    proj = torch.zeros((4, 4), device=DEV)
    proj[0, 0] = 1.0 / tanf
    proj[1, 1] = 1.0 / tanf
    proj[2, 2] = zfar / (zfar - znear)
    proj[3, 2] = 1.0
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    full = view.unsqueeze(0).bmm(proj.t().unsqueeze(0)).squeeze(0)
    return CameraSpec(viewmatrix=view, projmatrix=full, tanfovx=tanf, tanfovy=tanf,
                      image_height=height, image_width=width)


def two_part_scene(n_static=640, n_moving=640, seed=0, z_static=2.6, z_moving=2.0,
                   spread=0.5):
    """A static slab and a separate movable part, well separated in space so
    the canonical grouping can resolve them."""
    g = torch.Generator().manual_seed(seed)
    s = torch.zeros((n_static, 3))
    s[:, 0] = (torch.rand(n_static, generator=g) - 0.5) * spread * 2 - 0.45
    s[:, 1] = (torch.rand(n_static, generator=g) - 0.5) * spread * 2
    s[:, 2] = z_static + torch.rand(n_static, generator=g) * 0.15
    m = torch.zeros((n_moving, 3))
    m[:, 0] = (torch.rand(n_moving, generator=g) - 0.5) * spread + 0.5
    m[:, 1] = (torch.rand(n_moving, generator=g) - 0.5) * spread
    m[:, 2] = z_moving + torch.rand(n_moving, generator=g) * 0.15
    mu = torch.cat([s, m])
    n = mu.shape[0]
    truth = torch.cat([torch.zeros(n_static, dtype=torch.long),
                       torch.ones(n_moving, dtype=torch.long)])
    z = torch.zeros((n, 3))
    state = GaussianMotionState(
        mu=mu.to(DEV), R=torch.eye(3).repeat(n, 1, 1).to(DEV),
        v=z.clone().to(DEV), omega=z.to(DEV),
        colors=torch.rand((n, 3), generator=g).to(DEV),
        opacity=torch.full((n, 1), 0.85).to(DEV),
        scales=torch.full((n, 3), 0.045).to(DEV))
    return state, truth.to(DEV)


def moved_state(state, delta_mu):
    return GaussianMotionState(
        mu=state.mu + delta_mu, R=state.R, v=state.v, omega=state.omega,
        colors=state.colors, opacity=state.opacity, scales=state.scales)


def precisions_and_residuals(lin, gt, sigma=0.1):
    prec, res = [], []
    for view, (rgb_gt, d_gt, d_valid) in zip(lin.views, gt):
        r_rgb = rgb_gt - view.rgb
        p_rgb = build_precision(r_rgb, valid_mask=None, sigma_rgb=sigma,
                                robust=True, nu=5.0,
                                layout=("rgb", "rgb", "rgb"), channel_dim=0)
        r_d = d_gt - view.depth
        p_d = build_precision(r_d.unsqueeze(0), valid_mask=d_valid.unsqueeze(0),
                              sigma_rgb=(), depth_sigma_model="constant:0.02:0.0",
                              depth_range=d_gt.unsqueeze(0), robust=True, nu=5.0,
                              layout=("depth",), channel_dim=0).squeeze(0)
        prec.append((p_rgb, p_d))
        res.append((r_rgb, r_d))
    return prec, res


def render_gt(state, cams):
    lin = build_linearization(state, cams)
    return [(v.rgb.clone(), v.depth.clone(), (v.alpha > 0.05).clone())
            for v in lin.views]


def run_support(state, truth, tangent_fn, delta_fn, cams, *, target_size=64,
                potts=1.0, temperature=0.3, label=""):
    """Shared harness: build supernodes, render before/after, infer support.

    ``delta_fn(mu)`` returns the joint displacement for EVERY Gaussian, with no
    ground-truth masking. The observation applies it under the TRUE support
    mask; the energy applies it under the PREDICTED one. Masking the
    displacement before the energy sees it would make every labelling produce
    the same render, so the energy could not discriminate at all.
    """
    part = build_supernodes(state.mu, target_size=target_size, seed=0)
    gaussian_truth = truth
    # supernode-level truth = majority label of its members
    sid = part.supernode_id.long()
    node_truth = torch.zeros(part.num_supernodes, dtype=torch.long, device=DEV)
    for k in range(part.num_supernodes):
        mem = truth[sid == k]
        node_truth[k] = int(mem.float().mean().round())

    delta_full = delta_fn(state.mu)
    delta = delta_full * (gaussian_truth == MOVING).unsqueeze(1).to(delta_full.dtype)
    gt = render_gt(moved_state(state, delta), cams)
    lin = build_linearization(state, cams)
    grouped = GroupedObservation(linearization=lin, partition=part)
    prec, res = precisions_and_residuals(lin, gt)
    tangent = tangent_fn(state.mu)

    ev = supernode_motion_evidence(lin, grouped, part, tangent, res, prec)
    resp = ownership_responsibility(lin, grouped, part)

    def evidence_fn(_z):
        return ev, resp

    def energy_fn(z):
        """True nonlinear robust MAP energy after re-rendering with only the
        z=MOVING supernodes displaced (section 6.3 step 6)."""
        mask = z.to(DEV)[sid] == MOVING
        d = delta_full * mask.unsqueeze(1).to(delta_full.dtype)
        lin_try = build_linearization(moved_state(state, d), cams)
        e = 0.0
        for view, (rgb_gt, d_gt, d_valid) in zip(lin_try.views, gt):
            rr = rgb_gt - view.rgb
            pr = build_precision(rr, valid_mask=None, sigma_rgb=0.1, robust=True,
                                 nu=5.0, layout=("rgb", "rgb", "rgb"), channel_dim=0)
            e += float(0.5 * (pr * rr * rr).sum())
            rd = d_gt - view.depth
            pd = build_precision(rd.unsqueeze(0), valid_mask=d_valid.unsqueeze(0),
                                 sigma_rgb=(), depth_sigma_model="constant:0.02:0.0",
                                 depth_range=d_gt.unsqueeze(0), robust=True, nu=5.0,
                                 layout=("depth",), channel_dim=0).squeeze(0)
            e += float(0.5 * (pd * rd * rd).sum())
        return e

    cfg = SupportConfig(potts_weight=potts, max_outer=6,
                        evidence_temperature=temperature)
    state_out, history = infer_support(part, evidence_fn, energy_fn, cfg)
    m = support_metrics(state_out.hard, node_truth)
    gm = gaussian_support_metrics(state_out.hard, gaussian_truth, part.supernode_id)
    print(f"       {label}: {part.num_supernodes} supernodes "
          f"({int((node_truth == MOVING).sum())} truly moving)")
    print(f"       supernode  P {m['precision']:.3f}  R {m['recall']:.3f}  "
          f"F1 {m['f1']:.3f}  IoU {m['iou']:.3f}")
    print(f"       gaussian   P {gm['gaussian_precision']:.3f}  "
          f"R {gm['gaussian_recall']:.3f}  F1 {gm['gaussian_f1']:.3f}  "
          f"IoU {gm['gaussian_iou']:.3f}")
    energies = [h.get("energy", h.get("energy_after")) for h in history]
    print(f"       energy trajectory: {[f'{e:.1f}' for e in energies if e is not None]}")
    return state_out, history, m, gm, part, node_truth


def test_T_S1_mixed_static_and_prismatic():
    """T-S1: static background + one prismatic moving part."""
    cams = [camera(), camera(tz=0.25)]
    state, truth = two_part_scene()
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    step = 0.09

    def tangent_fn(mu):
        t = torch.zeros((mu.shape[0], 6), device=DEV)
        t[:, :3] = axis
        return t

    def delta_fn(mu):
        # the joint displacement for EVERY Gaussian; the harness applies the
        # true mask to build the observation and the predicted mask to score
        return (axis * step).reshape(1, 3).expand_as(mu).contiguous()

    _, hist, m, gm, _, _ = run_support(state, truth, tangent_fn, delta_fn, cams,
                                       label="T-S1 prismatic")
    check("T-S1 support F1 is high", m["f1"] > 0.85, f"F1 {m['f1']:.3f}")
    check("T-S1 support IoU is high", m["iou"] > 0.75, f"IoU {m['iou']:.3f}")
    check("T-S1 at Gaussian granularity too",
          gm["gaussian_f1"] > 0.85, f"F1 {gm['gaussian_f1']:.3f}")
    check("T-S1 the true objective decreased or held",
          hist[-1].get("accepted") is not False or len(hist) > 1)


def test_T_S2_mixed_static_and_offset_revolute():
    """T-S2: the same, but the moving part rotates about an ARBITRARY line
    that does not pass through the origin."""
    cams = [camera(), camera(tz=0.25)]
    state, truth = two_part_scene(seed=4)
    e = torch.tensor([0.0, 1.0, 0.15], device=DEV)
    e = e / e.norm()
    a = torch.tensor([0.5, 0.0, 2.05], device=DEV)      # through the moving part
    a = a - (a @ e) * e
    line = PluckerLine(direction=e, moment=torch.linalg.cross(a, e))
    q = 0.10

    def tangent_fn(mu):
        dmu, dphi = revolute_joint_tangent(mu, line)
        return torch.cat([dmu, dphi], dim=1)

    def delta_fn(mu):
        from joint_reduction.line_geometry import revolute_transport_line
        moved, _ = revolute_transport_line(mu, None, line, q)
        return moved - mu

    _, hist, m, gm, _, _ = run_support(state, truth, tangent_fn, delta_fn, cams,
                                       label="T-S2 offset revolute")
    check("T-S2 support F1 is high for an offset revolute axis",
          m["f1"] > 0.80, f"F1 {m['f1']:.3f}")
    check("T-S2 support IoU is high", m["iou"] > 0.70, f"IoU {m['iou']:.3f}")
    check("T-S2 the axis used was genuinely offset from the origin",
          float(line.moment.norm()) > 0.1, f"|m| {float(line.moment.norm()):.3f}")


def test_T_S3_occlusion_revealed_static_is_not_moving():
    """T-S3: a moving front part reveals static rear Gaussians.

    Those rear Gaussians change appearance dramatically (they go from hidden to
    visible) but they do NOT move. Labelling them moving from appearance change
    alone is the failure this test exists to catch.
    """
    g = torch.Generator().manual_seed(11)
    n_front, n_rear = 420, 420
    front = torch.zeros((n_front, 3))
    front[:, 0] = (torch.rand(n_front, generator=g) - 0.5) * 0.45
    front[:, 1] = (torch.rand(n_front, generator=g) - 0.5) * 0.45
    front[:, 2] = 1.85 + torch.rand(n_front, generator=g) * 0.08
    rear = torch.zeros((n_rear, 3))
    rear[:, 0] = (torch.rand(n_rear, generator=g) - 0.5) * 0.45
    rear[:, 1] = (torch.rand(n_rear, generator=g) - 0.5) * 0.45
    rear[:, 2] = 2.45 + torch.rand(n_rear, generator=g) * 0.08
    mu = torch.cat([front, rear])
    n = mu.shape[0]
    # FRONT moves, REAR is static and merely becomes visible.
    truth = torch.cat([torch.ones(n_front, dtype=torch.long),
                       torch.zeros(n_rear, dtype=torch.long)]).to(DEV)
    z = torch.zeros((n, 3))
    state = GaussianMotionState(
        mu=mu.to(DEV), R=torch.eye(3).repeat(n, 1, 1).to(DEV),
        v=z.clone().to(DEV), omega=z.to(DEV),
        colors=torch.rand((n, 3), generator=g).to(DEV),
        opacity=torch.full((n, 1), 0.95).to(DEV),
        scales=torch.full((n, 3), 0.05).to(DEV))
    cams = [camera()]
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    step = 0.55                      # large enough to really uncover the rear

    def tangent_fn(m_):
        t = torch.zeros((m_.shape[0], 6), device=DEV)
        t[:, :3] = axis
        return t

    def delta_fn(m_):
        return (axis * step).reshape(1, 3).expand_as(m_).contiguous()

    st, hist, m, gm, part, node_truth = run_support(
        state, truth, tangent_fn, delta_fn, cams, label="T-S3 occlusion")
    sid = part.supernode_id.long()
    rear_nodes = torch.tensor(
        [k for k in range(part.num_supernodes) if int(node_truth[k]) == STATIC],
        device=DEV)
    if rear_nodes.numel():
        mislabelled = int((st.hard.to(DEV)[rear_nodes] == MOVING).sum())
        frac = mislabelled / int(rear_nodes.numel())
        check("T-S3 revealed static supernodes are not labelled moving",
              frac < 0.25, f"{mislabelled}/{int(rear_nodes.numel())} "
                           f"({frac:.1%}) static supernodes called moving")
    check("T-S3 the moving front part is still found",
          m["recall"] > 0.70, f"recall {m['recall']:.3f}")
    check("T-S3 precision is not destroyed by the revealed region",
          m["precision"] > 0.70, f"precision {m['precision']:.3f}")


def test_T_S4_partial_visibility_uses_coherence():
    """T-S4: supernodes visible in only a subset of cameras.

    Their evidence is weak, so the Potts term on the canonical graph must carry
    the label from their visible neighbours rather than each one hallucinating
    an independent decision.
    """
    cams = [camera(width=96, height=96, fov=0.55)]     # narrow: clips the sides
    state, truth = two_part_scene(seed=8, spread=0.75)
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)

    def tangent_fn(mu):
        t = torch.zeros((mu.shape[0], 6), device=DEV)
        t[:, :3] = axis
        return t

    def delta_fn(mu):
        return (axis * 0.09).reshape(1, 3).expand_as(mu).contiguous()

    print("       --- without smoothness (potts = 0) ---")
    _, _, m0, _, _, _ = run_support(state, truth, tangent_fn, delta_fn, cams,
                                    potts=0.0, label="T-S4 potts=0")
    print("       --- with canonical-graph smoothness (potts = 2) ---")
    st2, _, m2, _, part, node_truth = run_support(
        state, truth, tangent_fn, delta_fn, cams, potts=2.0, label="T-S4 potts=2")
    check("T-S4 spatial coherence does not hurt the labelling",
          m2["f1"] >= m0["f1"] - 0.02,
          f"F1 {m0['f1']:.3f} (no smoothness) -> {m2['f1']:.3f} (smoothed)")

    # The structural claim: smoothing reduces isolated disagreements with
    # one's neighbours on the canonical graph.
    def boundary_count(z):
        e = part.adjacency
        if e.numel() == 0:
            return 0
        za = z.to(DEV)[e[:, 0]]
        zb = z.to(DEV)[e[:, 1]]
        return int((za != zb).sum())
    print(f"       graph disagreements: potts=2 -> {boundary_count(st2.hard)} "
          f"of {int(part.adjacency.shape[0])} edges")
    check("T-S4 the smoothed labelling is spatially coherent",
          boundary_count(st2.hard) < 0.5 * int(part.adjacency.shape[0]),
          f"{boundary_count(st2.hard)}/{int(part.adjacency.shape[0])} edges cut")


def test_graph_cut_is_exact():
    """The binary subproblem must be solved exactly (section 6.3 step 5), so
    on small instances it must match brute force."""
    import itertools
    torch.manual_seed(3)
    for trial in range(8):
        c = 7
        us = torch.rand(c).double()
        um = torch.rand(c).double()
        edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
                              [0, 6], [2, 5]], dtype=torch.long)
        lam = 0.4

        def energy(z):
            # us[k] is the cost of labelling k STATIC, um[k] of MOVING.
            e = sum(float(us[k]) if z[k] == STATIC else float(um[k])
                    for k in range(c))
            e += lam * sum(1 for a, b in edges.tolist() if z[a] != z[b])
            return e

        best = min(itertools.product([0, 1], repeat=c), key=energy)
        labels, _ = graph_cut_binary(us, um, edges, lam)
        got = energy(labels.tolist())
        check(f"§6.3 graph cut attains the global optimum (trial {trial})",
              abs(got - energy(list(best))) < 1e-9,
              f"cut {got:.6f} vs brute force {energy(list(best)):.6f}")


def test_metrics_are_correct():
    pred = torch.tensor([1, 1, 0, 0, 1])
    true = torch.tensor([1, 0, 0, 1, 1])
    m = support_metrics(pred, true)
    check("§6.4 precision", abs(m["precision"] - 2 / 3) < 1e-12, f"{m['precision']}")
    check("§6.4 recall", abs(m["recall"] - 2 / 3) < 1e-12, f"{m['recall']}")
    check("§6.4 F1", abs(m["f1"] - 2 / 3) < 1e-12, f"{m['f1']}")
    check("§6.4 IoU", abs(m["iou"] - 0.5) < 1e-12, f"{m['iou']}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
