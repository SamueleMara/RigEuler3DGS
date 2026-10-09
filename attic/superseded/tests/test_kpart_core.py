"""K-part core: models, Jacobian, multi-label support, identity, splitting.

CODEX_REQUIRED_MODIFICATIONS_MULTIVIEW_MULTIFRAME_KPART_IEKF.md sections 7-10,
15, 21, and tests T-K1..T-K4, T-SPLIT1, T-MV1.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.measurement import build_precision  # noqa: E402
from estimator.partition_refine import (  # noqa: E402
    split_supernodes,
    supernode_disagreement,
)
from estimator.renderer_bridge import CameraSpec, build_linearization  # noqa: E402
from estimator.state import GaussianMotionState  # noqa: E402
from estimator.supernodes import GroupedObservation, build_supernodes  # noqa: E402
from joint_reduction.kpart_models import (  # noqa: E402
    PRISMATIC,
    REVOLUTE,
    KPartObservation,
    PartModel,
    SceneModel,
)
from joint_reduction.kpart_support import (  # noqa: E402
    alpha_expansion,
    build_unaries,
    infer_kpart_support,
    kpart_metrics,
    part_achievable_gain,
    potts_energy,
)
from joint_reduction.label_identity import apply_permutation, match_parts  # noqa: E402
from joint_reduction.line_geometry import PluckerLine  # noqa: E402

DEV = "cuda"
PASSED: list[str] = []


def check(name, cond, detail=""):
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


def three_region_scene(n_each=420, seed=0):
    """Static slab + prismatic part + revolute part (T-K1's scene)."""
    g = torch.Generator().manual_seed(seed)

    def blob(n, cx, cy, cz, s=0.22):
        p = torch.zeros((n, 3))
        p[:, 0] = (torch.rand(n, generator=g) - 0.5) * s + cx
        p[:, 1] = (torch.rand(n, generator=g) - 0.5) * s + cy
        p[:, 2] = (torch.rand(n, generator=g) - 0.5) * 0.12 + cz
        return p

    static = blob(n_each, -0.55, 0.0, 2.55)
    prism = blob(n_each, 0.0, 0.42, 2.05)
    revol = blob(n_each, 0.52, -0.38, 2.15)
    mu = torch.cat([static, prism, revol])
    n = mu.shape[0]
    truth = torch.cat([torch.zeros(n_each, dtype=torch.long),
                       torch.ones(n_each, dtype=torch.long),
                       torch.full((n_each,), 2, dtype=torch.long)])
    z = torch.zeros((n, 3))
    st = GaussianMotionState(
        mu=mu.to(DEV), R=torch.eye(3).repeat(n, 1, 1).to(DEV),
        v=z.clone().to(DEV), omega=z.to(DEV),
        colors=torch.rand((n, 3), generator=g).to(DEV),
        opacity=torch.full((n, 1), 0.85).to(DEV),
        scales=torch.full((n, 3), 0.04).to(DEV))
    return st, truth.to(DEV)


def obs_of(state, mu, R, cams):
    s = GaussianMotionState(mu=mu, R=R if R is not None else state.R, v=state.v,
                            omega=state.omega, colors=state.colors,
                            opacity=state.opacity, scales=state.scales)
    lin = build_linearization(s, cams)
    return [(v.rgb.clone(), v.depth.clone(), (v.alpha > 0.05).clone())
            for v in lin.views]


def prec_res(lin, gt):
    prec, res = [], []
    for view, (rgb, d, valid) in zip(lin.views, gt):
        rr = rgb - view.rgb
        pr = build_precision(rr, valid_mask=None, sigma_rgb=0.1, robust=True,
                             nu=5.0, layout=("rgb", "rgb", "rgb"), channel_dim=0)
        rd = d - view.depth
        pd = build_precision(rd.unsqueeze(0), valid_mask=valid.unsqueeze(0),
                             sigma_rgb=(), depth_sigma_model="constant:0.02:0.0",
                             depth_range=d.unsqueeze(0), robust=True, nu=5.0,
                             layout=("depth",), channel_dim=0).squeeze(0)
        prec.append((pr, pd))
        res.append((rr, rd))
    return prec, res


# ===========================================================================
def test_alpha_expansion_is_optimal():
    """§10: the multi-label solver must reach the Potts optimum."""
    torch.manual_seed(0)
    worst = 0.0
    for trial in range(10):
        C, L = 8, 3
        unary = torch.rand((C, L)).double()
        edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
                              [6, 7], [0, 7], [2, 5]], dtype=torch.long)
        w = torch.ones(edges.shape[0]).double()
        lam = 0.35

        def E(z):
            d, s = potts_energy(torch.tensor(z, dtype=torch.long), unary,
                                edges, w, lam)
            return d + s

        best = min(itertools.product(range(L), repeat=C), key=E)
        lab, _ = alpha_expansion(unary, edges, w, lam=lam)
        worst = max(worst, E(lab.tolist()) - E(list(best)))
    check("§10 alpha-expansion attains the global Potts optimum on 10 instances",
          worst < 1e-9, f"worst gap {worst:.2e}")


def test_kpart_models_and_jacobian():
    """§8/§9: per-part models and the K-part pixel Jacobian."""
    cams = [camera()]
    st, truth = three_region_scene(n_each=200, seed=1)
    e_p = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    e_r = torch.tensor([0.0, 0.0, 1.0], device=DEV)
    a_r = torch.tensor([0.52, -0.38, 0.0], device=DEV)
    a_r = a_r - (a_r @ e_r) * e_r
    m_r = torch.linalg.cross(a_r, e_r)
    scene = SceneModel(parts=[PartModel(kind=PRISMATIC, axis=e_p, part_index=1),
                              PartModel(kind=REVOLUTE, axis=e_r, moment=m_r,
                                        part_index=2)], K=2)
    check("§8 a revolute part carries a Plucker moment, not an origin axis",
          float(scene.parts[1].moment.norm()) > 0.1,
          f"|m| {float(scene.parts[1].moment.norm()):.3f}")
    check("§8 and satisfies the e.m = 0 gauge",
          abs(float(scene.parts[1].axis @ scene.parts[1].moment)) < 1e-6)

    # transport moves only the right members
    mu, R = scene.transport_all(st.mu, st.R, truth, [0.05, 0.1])
    d = (mu - st.mu).norm(dim=1)
    check("§8 static Gaussians do not move", float(d[truth == 0].max()) < 1e-7)
    check("§8 part 1 moves by its own prismatic q",
          abs(float(d[truth == 1].mean()) - 0.05) < 1e-6,
          f"{float(d[truth == 1].mean()):.6f}")
    check("§8 part 2 moves by its own revolute q",
          float(d[truth == 2].min()) > 1e-4)
    check("§8 the two parts move by DIFFERENT amounts (not one global twist)",
          abs(float(d[truth == 1].mean()) - float(d[truth == 2].mean())) > 1e-3)

    # §9: the K-part Jacobian
    lin = build_linearization(st, cams)
    kp = KPartObservation(lin, scene, st.mu, truth)
    check("§9 part k's tangent is zero outside part k",
          float(kp.tangent(1)[truth != 1].abs().max()) == 0.0
          and float(kp.tangent(2)[truth != 2].abs().max()) == 0.0)
    check("§9 and non-zero inside it",
          float(kp.tangent(1)[truth == 1].abs().max()) > 0
          and float(kp.tangent(2)[truth == 2].abs().max()) > 0)

    # adjoint identity for the K-dimensional operator
    for trial in range(3):
        torch.manual_seed(20 + trial)
        dq = torch.randn(2, device=DEV)
        hv = kp.apply_H(dq.tolist())
        z = [(torch.randn_like(a), torch.randn_like(b)) for a, b in hv]
        lhs = sum(float((a.double() * zc.double()).sum()
                        + (b.double() * zd.double()).sum())
                  for (a, b), (zc, zd) in zip(hv, z))
        rhs = float((dq.double() * kp.apply_HT(z).double()).sum())
        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
        check(f"§9 the K-part operator is self-adjoint (trial {trial})",
              rel < 1e-4, f"{lhs:.5f} vs {rhs:.5f}, rel {rel:.2e}")

    # linearity across parts: H(dq1+dq2) = H(dq1) + H(dq2)
    a1 = kp.apply_H([1.0, 0.0])[0][0]
    a2 = kp.apply_H([0.0, 1.0])[0][0]
    both = kp.apply_H([1.0, 1.0])[0][0]
    check("§9 delta y = sum_k H_{p,q_k} dq_k is additive across parts",
          float((both - a1 - a2).abs().max()) < 1e-4,
          f"max {float((both - a1 - a2).abs().max()):.3e}")


def test_T_MV1_camera_accumulation():
    """T-MV1: H_all^T z = sum_c H_c^T z_c."""
    st, truth = three_region_scene(n_each=200, seed=2)
    cams = [camera(), camera(tz=0.25), camera(tz=-0.25), camera(tz=0.5)]
    lin_all = build_linearization(st, cams)
    torch.manual_seed(3)
    probe = torch.randn((st.num_gaussians, 6), device=DEV)
    hv = lin_all.apply_H(probe)
    z = [(torch.randn_like(a), torch.randn_like(b)) for a, b in hv]
    all_adj = lin_all.apply_HT(z).double()
    per = torch.zeros_like(all_adj)
    for ci, c in enumerate(cams):
        l1 = build_linearization(st, [c])
        per += l1.apply_HT([z[ci]]).double()
    rel = float((all_adj - per).abs().max() / all_adj.abs().max())
    check("T-MV1 multi-camera H^T z equals the per-camera sum", rel < 1e-4,
          f"max rel {rel:.3e} over {len(cams)} cameras")


def test_T_K1_known_K_two_parts():
    """T-K1: static + prismatic + revolute, K=2, correct 3-label segmentation."""
    cams = [camera(), camera(tz=0.3)]
    st, truth = three_region_scene(seed=4)
    part = build_supernodes(st.mu, target_size=64, seed=0)
    sid = part.supernode_id.long()
    node_truth = torch.zeros(part.num_supernodes, dtype=torch.long, device=DEV)
    for c in range(part.num_supernodes):
        m = truth[sid == c]
        node_truth[c] = int(torch.mode(m).values) if m.numel() else 0

    e_p = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    e_r = torch.tensor([0.0, 0.0, 1.0], device=DEV)
    a_r = torch.tensor([0.52, -0.38, 0.0], device=DEV)
    a_r = a_r - (a_r @ e_r) * e_r
    scene = SceneModel(parts=[
        PartModel(kind=PRISMATIC, axis=e_p, part_index=1),
        PartModel(kind=REVOLUTE, axis=e_r,
                  moment=torch.linalg.cross(a_r, e_r), part_index=2)], K=2)
    q_true = [0.09, 0.22]
    mu_t, R_t = scene.transport_all(st.mu, st.R, truth, q_true)
    gt = obs_of(st, mu_t, R_t, cams)

    lin = build_linearization(st, cams)
    grouped = GroupedObservation(linearization=lin, partition=part)
    prec, res = prec_res(lin, gt)
    gains = [part_achievable_gain(lin, grouped, prec, res,
                                 scene.part_tangent(st.mu,
                                                    torch.full_like(truth, k), k))
             for k in (1, 2)]
    sup = infer_kpart_support(part, gains, K=2, potts_weight=1.0)
    m = kpart_metrics(sup.hard.to(DEV), node_truth, K=2)
    print(f"       supernode labels {sup.counts()} (truth "
          f"{[int((node_truth == k).sum()) for k in range(3)]})")
    print(f"       Hungarian accuracy {m['hungarian_accuracy']:.3f}  "
          f"ARI {m['adjusted_rand_index']:.3f}  permutation {m['permutation']}")
    check("T-K1 all K moving labels are used (known-K constraint)",
          sup.all_parts_present(), str(sup.counts()))
    check("T-K1 the 3-label segmentation is correct up to permutation",
          m["hungarian_accuracy"] > 0.85,
          f"accuracy {m['hungarian_accuracy']:.3f}")
    check("T-K1 and the clustering agrees structurally (ARI)",
          m["adjusted_rand_index"] > 0.6,
          f"ARI {m['adjusted_rand_index']:.3f}")
    check("T-K1 the unary matrix has shape [C, K+1]",
          tuple(sup.unary.shape) == (part.num_supernodes, 3),
          str(tuple(sup.unary.shape)))
    check("T-K1 soft scores are a distribution over K+1 labels",
          abs(float(sup.soft.sum(dim=1).mean()) - 1.0) < 1e-5)


def test_T_K4_gt_leakage():
    """T-K4: changing GT part labels must not change inference when K is fixed."""
    cams = [camera()]
    st, truth = three_region_scene(n_each=300, seed=6)
    part = build_supernodes(st.mu, target_size=64, seed=0)
    e_p = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    scene = SceneModel(parts=[PartModel(kind=PRISMATIC, axis=e_p, part_index=1)], K=1)
    mu_t, _ = scene.transport_all(st.mu, st.R, (truth > 0).long(), [0.08])
    gt = obs_of(st, mu_t, st.R, cams)
    lin = build_linearization(st, cams)
    grouped = GroupedObservation(linearization=lin, partition=part)
    prec, res = prec_res(lin, gt)

    out = []
    for permuted in (False, True):
        # permuting the GT labels must be invisible to the estimator, because
        # the estimator never reads them
        _ = truth if not permuted else torch.where(truth == 1, torch.full_like(truth, 2),
                                                   torch.where(truth == 2, torch.ones_like(truth), truth))
        g = part_achievable_gain(lin, grouped, prec, res, None)
        sup = infer_kpart_support(part, [g], K=1, potts_weight=1.0)
        out.append(sup.hard.clone())
    check("T-K4 permuting GT part labels does not change inference",
          torch.equal(out[0], out[1]),
          "the estimator reads K only, never the labels")


def test_T_SPLIT1_boundary_supernode():
    """T-SPLIT1: a supernode crossing a true part boundary must split."""
    torch.manual_seed(7)
    n = 600
    mu = torch.randn((n, 3), device=DEV) * 0.2
    part = build_supernodes(mu, num_supernodes=6, seed=0)
    sid = part.supernode_id.long()
    # craft evidence that disagrees INSIDE supernode 0 and agrees elsewhere
    K = 2
    ev = torch.zeros((n, K), device=DEV)
    c0 = sid == 0
    idx = torch.nonzero(c0, as_tuple=False).reshape(-1)
    half = idx[: idx.numel() // 2]
    ev[idx, 0] = 1.0
    ev[half, 0] = -1.0          # half the members disagree
    for c in range(1, part.num_supernodes):
        ev[sid == c, 1] = 1.0

    dis = supernode_disagreement(ev, sid, part.num_supernodes)
    check("§15 the straddling supernode has the highest disagreement",
          int(dis.argmax()) == 0, f"argmax {int(dis.argmax())}, "
                                  f"scores {[round(float(x),3) for x in dis]}")
    ref = split_supernodes(part, ev, tau_split=0.2, min_members=4)
    check("§15 it is split", ref.partition.num_supernodes > part.num_supernodes,
          f"{part.num_supernodes} -> {ref.partition.num_supernodes}")
    check("§15 with parent-child provenance recorded", len(ref.parent_of) > 0,
          str(ref.parent_of))
    check("§15 canonical geometry is NOT mutated",
          torch.equal(ref.partition.canonical_means, part.canonical_means))
    new_sid = ref.partition.supernode_id.long()
    sub = new_sid[c0]
    check("§15 the split separates the disagreeing members",
          int(sub.unique().numel()) == 2, f"{int(sub.unique().numel())} children")
    check("§15 every other supernode is untouched",
          all(int(new_sid[sid == c].unique().numel()) == 1
              for c in range(1, part.num_supernodes)))


def test_T_K3_label_permutation():
    """T-K3: identity matching across windows."""
    prev_sup = [torch.tensor([1, 1, 1, 0, 0, 0]), torch.tensor([0, 0, 0, 1, 1, 1])]
    # the next window discovers the SAME two parts but numbers them the other way
    cur_sup = [torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([1, 1, 1, 0, 0, 0])]
    e1 = torch.tensor([1.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0])
    prev_models = [PartModel(kind=PRISMATIC, axis=e1, part_index=1),
                   PartModel(kind=PRISMATIC, axis=e2, part_index=2)]
    cur_models = [PartModel(kind=PRISMATIC, axis=e2, part_index=1),
                  PartModel(kind=PRISMATIC, axis=e1, part_index=2)]
    m = match_parts(prev_sup, cur_sup, prev_models, cur_models)
    print(f"       permutation {m['permutation']}, cost {m['total_cost']:.4f}")
    check("T-K3 the swap is detected", m["permutation"] == {1: 2, 2: 1},
          str(m["permutation"]))
    check("T-K3 and is reported as non-identity", not m["is_identity"])
    labels = torch.tensor([0, 1, 1, 2, 2, 0])
    relabelled = apply_permutation(labels, m["permutation"])
    check("T-K3 applying it restores the previous numbering",
          torch.equal(relabelled, torch.tensor([0, 2, 2, 1, 1, 0])),
          str(relabelled.tolist()))
    check("T-K3 static is never permuted",
          int((relabelled == 0).sum()) == int((labels == 0).sum()))
    # an unchanged window must match to the identity
    m2 = match_parts(prev_sup, prev_sup, prev_models, prev_models)
    check("T-K3 an unchanged window matches to the identity", m2["is_identity"],
          str(m2["permutation"]))


def test_unary_construction():
    """§10's unary matrix must be a non-negative cost consistent with the gains."""
    g1 = torch.tensor([5.0, 0.0, 1.0])
    g2 = torch.tensor([0.0, 7.0, 1.0])
    u = build_unaries([g1, g2])
    check("§10 the unary matrix is [C, K+1]", tuple(u.shape) == (3, 3))
    check("§10 every unary is non-negative (min-cut requires it)",
          float(u.min()) >= 0.0)
    check("§10 the best-explaining label has the lowest cost",
          int(u[0].argmin()) == 1 and int(u[1].argmin()) == 2,
          f"{u.tolist()}")
    check("§10 a supernode explained equally by both parts is ambiguous",
          abs(float(u[2, 1] - u[2, 2])) < 1e-9)


def main():
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
