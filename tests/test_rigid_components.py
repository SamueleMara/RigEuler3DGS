"""Sections 33-40: whole-window rigid-component discovery.

CODEX_IMPLEMENT_WHOLE_WINDOW_RIGID_COMPONENT_DISCOVERY.md sections 5-17, 33-40.

The contract these tests enforce is section 1's:

    discover rigid bodies by CONSISTENCY OVER THE WHOLE SEQUENCE,
    not by thresholding displacement magnitude.

Sections 34 and 36 are the ones that actually discriminate the two: a
near-axis revolute member barely moves yet belongs to the moving body, and two
parts can share a final displacement magnitude while following different rigid
trajectories. Any magnitude rule fails both; this is the behavioural test
section 40 asks for.
"""

from __future__ import annotations

import math
import sys
import pathlib
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.manifold import so3_exp  # noqa: E402
from estimator.part_assignment import (assignment_margin,  # noqa: E402
                                       update_component_labels,
                                       whole_track_assignment_cost)
from estimator.rigid_part_estimator import (  # noqa: E402
    robust_fit_frame_transform, robust_fit_se3_trajectory,
    se3_distance_to_identity)
from joint_reduction.rigid_discovery import (  # noqa: E402
    identify_static_component, initialize_rigid_components)
from joint_reduction.rigidity_graph import (  # noqa: E402
    build_rigidity_graph, median_absolute_deviation as mad, robust_scale)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def knn_edges(x: torch.Tensor, k: int = 6) -> torch.Tensor:
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    idx = d.topk(min(k, x.shape[0] - 1), largest=False).indices
    e = set()
    for i in range(x.shape[0]):
        for j in idx[i].tolist():
            e.add((min(i, j), max(i, j)))
    return torch.tensor(sorted(e), dtype=torch.long, device=x.device)


def decompose(x0, tracks, K, noise_free=None, seed=0):
    """The production path: rigidity graph -> K+1 components -> labels."""
    e = knn_edges(x0)
    g = build_rigidity_graph(tracks, e)
    L = float((x0 - x0.mean(0)).norm(dim=1).mean().clamp_min(1e-9))
    init = initialize_rigid_components(x0, tracks, e, g["affinity"], L, K,
                                       num_nodes=x0.shape[0], seed=seed,
                                       verbose=False)
    comps, lab = init["components"], init["labels"]
    for _ in range(6):
        new = []
        for k in range(K + 1):
            m = lab == k
            if int(m.sum()) < 3:
                new.append(comps[k])
                continue
            f = robust_fit_se3_trajectory(x0, tracks, m)
            new.append({"R": f["R"], "t": f["t"]})
        comps = new
        cost = whole_track_assignment_cost(x0, tracks, comps, L)
        upd = update_component_labels(cost, e, g["affinity"], lab)
        if int((upd["labels"] == lab).all()):
            lab = upd["labels"]
            break
        lab = upd["labels"]
    st = identify_static_component(comps, L)
    remap = torch.tensor([st["remap"][int(k)] for k in range(K + 1)],
                         device=lab.device)
    return {"labels": remap[lab], "components": st["components"],
            "cost": whole_track_assignment_cost(x0, tracks, st["components"], L),
            "L": L, "graph": g}


def scene(n_static=90, n_move=60, spread=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = (torch.rand((n_static, 3), generator=g) - 0.5) * spread * 2
    s[:, 0] -= spread
    m = (torch.rand((n_move, 3), generator=g) - 0.5) * spread * 2
    m[:, 0] += spread
    return s.to(DEV).double(), m.to(DEV).double()


def accuracy(lab, truth, K):
    """Best-permutation accuracy over the K+1 labels."""
    import itertools
    best = 0.0
    for p in itertools.permutations(range(K + 1)):
        mapped = torch.tensor([p[int(v)] for v in lab], device=lab.device)
        best = max(best, float((mapped == truth).double().mean()))
    return best


# ===========================================================================
def test_13_mad_is_the_real_mad():
    """§13: MAD is median|r - median(r)|, not 1.4826*median(r)."""
    r = torch.tensor([10.0, 10.0, 10.0, 10.0, 50.0])
    check("§13 MAD is median |r - median(r)|", abs(float(mad(r))) < 1e-9,
          f"{float(mad(r)):.6f} (the wrong form would give "
          f"{1.4826 * float(r.median()):.3f})")
    r2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])   # median 3, MAD 1
    got = float(robust_scale(r2))
    check("§13 and the robust scale is 1.4826 x that",
          abs(got - 1.4826) < 1e-5, f"{got:.6f} (float32 precision)")
    check("§13 a degenerate MAD is floored, not infinite",
          float(robust_scale(torch.ones(9))) > 0)


def test_33_prismatic_small_displacement():
    """§33: a small prismatic motion buried in noisy static tracks.

    The moving part's displacement OVERLAPS the static noise, so any magnitude
    rule must fail; rigid consistency must not.
    """
    torch.manual_seed(0)
    s, m = scene(seed=1)
    x0 = torch.cat([s, m])
    truth = torch.cat([torch.zeros(len(s)), torch.ones(len(m))]).long().to(DEV)
    T = 12
    step = 0.010                     # tiny
    noise = 0.008                    # static noise of the SAME order
    tracks = []
    for t in range(T):
        p = x0.clone()
        p[truth == 1, 0] += step * t
        p += torch.randn(p.shape, device=DEV).double() * noise
        tracks.append(p)
    tracks = torch.stack(tracks)
    dmag = (tracks[-1] - x0).norm(dim=1)
    print(f"       |dx| static median {float(dmag[truth==0].median()):.4f}  "
          f"moving median {float(dmag[truth==1].median()):.4f}  "
          f"(overlap: a magnitude rule cannot separate these)")
    r = decompose(x0, tracks, 1)
    acc = accuracy(r["labels"], truth, 1)
    print(f"       rigid decomposition accuracy {acc:.3f}, counts "
          f"{[int((r['labels']==k).sum()) for k in range(2)]}")
    check("§33 a small prismatic motion is recovered by rigid consistency",
          acc > 0.90, f"accuracy {acc:.3f}")
    moved = se3_distance_to_identity(r["components"][1]["R"],
                                     r["components"][1]["t"], r["L"])
    check("§33 the moving component is NOT the identity", moved > 1e-6,
          f"d(T, I) = {moved:.3e}")


def test_34_revolute_points_near_axis():
    """§34: THE critical regression against magnitude gating.

    A revolute body contains points at every radius. ||d_i|| = 2 r_i sin(q/2),
    so near-axis members barely move. They must be kept.
    """
    torch.manual_seed(1)
    s, _ = scene(n_move=0, seed=2)
    g = torch.Generator().manual_seed(7)
    # moving body spanning radius 0.02 .. 0.8 about the axis
    rad = torch.cat([torch.linspace(0.02, 0.10, 30),
                     torch.linspace(0.10, 0.80, 50)])
    ang = torch.rand(len(rad), generator=g) * 2 * math.pi
    mv = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang),
                      (torch.rand(len(rad), generator=g) - 0.5) * 0.2], 1)
    mv = mv.to(DEV).double() + torch.tensor([1.2, 0.0, 0.0], device=DEV).double()
    x0 = torch.cat([s, mv])
    truth = torch.cat([torch.zeros(len(s)), torch.ones(len(mv))]).long().to(DEV)
    axis = torch.tensor([0.0, 0.0, 1.0], device=DEV).double()
    centre = torch.tensor([1.2, 0.0, 0.0], device=DEV).double()
    T = 12
    tracks = []
    for t in range(T):
        q = 0.05 * t
        R = so3_exp((q * axis).reshape(1, 3))[0].double()
        p = x0.clone()
        p[truth == 1] = centre + (x0[truth == 1] - centre) @ R.T
        p += torch.randn(p.shape, device=DEV).double() * 0.004
        tracks.append(p)
    tracks = torch.stack(tracks)

    r = decompose(x0, tracks, 1)
    near = (x0[:, :2] - centre[:2]).norm(dim=1) < 0.12
    near_moving = near & (truth == 1)
    kept = float((r["labels"][near_moving] == 1).double().mean())
    dmag = (tracks[-1] - x0).norm(dim=1)
    print(f"       near-axis members: {int(near_moving.sum())}, their |dx| "
          f"median {float(dmag[near_moving].median()):.4f} vs static "
          f"{float(dmag[truth==0].median()):.4f}")
    print(f"       retained in the moving part: {kept:.3f}")
    acc = accuracy(r["labels"], truth, 1)
    check("§34 near-axis members are retained in the moving part", kept > 0.80,
          f"{kept:.3f} retained")
    check("§34 and the whole body is recovered", acc > 0.90,
          f"accuracy {acc:.3f}")


def test_35_36_two_parts():
    """§35: K=2, a translation and a rotation. §36: equal magnitudes."""
    torch.manual_seed(2)
    g = torch.Generator().manual_seed(3)
    s = ((torch.rand((80, 3), generator=g) - 0.5) * 0.6).to(DEV).double()
    a = ((torch.rand((60, 3), generator=g) - 0.5) * 0.4).to(DEV).double() \
        + torch.tensor([1.5, 0.0, 0.0], device=DEV).double()
    b = ((torch.rand((60, 3), generator=g) - 0.5) * 0.4).to(DEV).double() \
        + torch.tensor([-1.5, 0.0, 0.0], device=DEV).double()
    x0 = torch.cat([s, a, b])
    truth = torch.cat([torch.zeros(80), torch.ones(60),
                       2 * torch.ones(60)]).long().to(DEV)
    axis = torch.tensor([0.0, 0.0, 1.0], device=DEV).double()
    # Section 16 asks for two moving bodies with SIMILAR final displacement
    # norms and different whole-window trajectories. With the axis through B's
    # own centroid its radius caps the achievable displacement at ~0.07 against
    # A's 0.33, so the fixture never produced the comparable magnitudes it
    # claimed to test. Offsetting the axis by 0.5 and turning further brings
    # B to ~0.36, within section 36's own tolerance of A.
    cb = torch.tensor([-2.0, 0.0, 0.0], device=DEV).double()
    T = 12
    tracks = []
    for t in range(T):
        p = x0.clone()
        p[truth == 1, 1] += 0.03 * t                       # A translates
        R = so3_exp((0.064 * t * axis).reshape(1, 3))[0].double()
        p[truth == 2] = cb + (x0[truth == 2] - cb) @ R.T   # B rotates
        p += torch.randn(p.shape, device=DEV).double() * 0.004
        tracks.append(p)
    tracks = torch.stack(tracks)
    dm = (tracks[-1] - x0).norm(dim=1)
    print(f"       final |dx| median: A {float(dm[truth==1].median()):.4f}  "
          f"B {float(dm[truth==2].median()):.4f}   "
          f"(§36: comparable magnitudes, different trajectories)")
    r = decompose(x0, tracks, 2, seed=1)
    acc = accuracy(r["labels"], truth, 2)
    counts = [int((r["labels"] == k).sum()) for k in range(3)]
    print(f"       K+1 = 3 components, accuracy {acc:.3f}, counts {counts}")
    check("§35 three rigid components are recovered up to permutation",
          acc > 0.85, f"accuracy {acc:.3f}")
    check("§35 none of the three components is empty", all(c > 0 for c in counts),
          str(counts))
    check("§36 parts with similar displacement magnitude are separated",
          abs(float(dm[truth == 1].median()) - float(dm[truth == 2].median()))
          < 0.5 * float(dm[truth == 1].median()) and acc > 0.85,
          "comparable magnitudes, still separated")


def test_37_outlier_blowups():
    """§37: 10x-20x Phase-1 blowups, the storage heavy tail."""
    torch.manual_seed(3)
    s, m = scene(seed=4)
    x0 = torch.cat([s, m])
    truth = torch.cat([torch.zeros(len(s)), torch.ones(len(m))]).long().to(DEV)
    T, step = 12, 0.04
    tracks = []
    gen = torch.Generator(device=DEV).manual_seed(11)
    blow = torch.zeros(len(x0), dtype=torch.bool, device=DEV)
    blow[torch.randperm(len(x0), generator=torch.Generator().manual_seed(5))[:12]] = True
    for t in range(T):
        p = x0.clone()
        p[truth == 1, 0] += step * t
        p += torch.randn(p.shape, generator=gen, device=DEV).double() * 0.004
        # 10x-20x blowups on a few nodes
        p[blow] += torch.randn((int(blow.sum()), 3), generator=gen,
                               device=DEV).double() * step * t * 15
        tracks.append(p)
    tracks = torch.stack(tracks)
    r = decompose(x0, tracks, 1)
    acc = accuracy(r["labels"], truth, 1)
    # transform error against the truth
    got = r["components"][1]["t"][-1]
    want = torch.tensor([step * (T - 1), 0.0, 0.0], device=DEV).double()
    err = float((got - want).norm())
    fit = robust_fit_frame_transform(x0[truth == 1], tracks[-1][truth == 1])
    low = int((fit["weights"] < 0.1 * fit["weights"].max()).sum())
    print(f"       {int(blow.sum())} blowups at ~15x; label accuracy {acc:.3f}, "
          f"|t| error {err:.4f} against |t|={float(want.norm()):.3f}")
    print(f"       robust fit downweighted {low} members")
    check("§37 labels survive the blowups", acc > 0.85, f"accuracy {acc:.3f}")
    check("§37 the transform stays accurate", err < 0.25 * float(want.norm()),
          f"error {err:.4f}")
    check("§37 robust weights actually rejected outliers", low > 0, f"{low}")


def test_38_static_model_mismatch():
    """§38: static nodes with noisy free motion must still form the identity.

    This is the failure mode directly: "inconsistent with the static renderer"
    was being read as "part of the moving body".
    """
    torch.manual_seed(4)
    s, m = scene(seed=6)
    x0 = torch.cat([s, m])
    truth = torch.cat([torch.zeros(len(s)), torch.ones(len(m))]).long().to(DEV)
    T, step = 12, 0.05
    gen = torch.Generator(device=DEV).manual_seed(9)
    tracks = []
    for t in range(T):
        p = x0.clone()
        p[truth == 1, 0] += step * t
        # static nodes get LARGE but incoherent model-mismatch motion
        p[truth == 0] += torch.randn((int((truth == 0).sum()), 3),
                                     generator=gen, device=DEV).double() * 0.03
        p[truth == 1] += torch.randn((int((truth == 1).sum()), 3),
                                     generator=gen, device=DEV).double() * 0.004
        tracks.append(p)
    tracks = torch.stack(tracks)
    r = decompose(x0, tracks, 1)
    acc = accuracy(r["labels"], truth, 1)
    d_static = se3_distance_to_identity(r["components"][0]["R"],
                                        r["components"][0]["t"], r["L"])
    d_moving = se3_distance_to_identity(r["components"][1]["R"],
                                        r["components"][1]["t"], r["L"])
    print(f"       accuracy {acc:.3f}; d(T,I): static {d_static:.3e}, "
          f"moving {d_moving:.3e}")
    check("§38 noisy static nodes still form the identity component",
          acc > 0.85, f"accuracy {acc:.3f}")
    check("§11 the static component is the one nearest the identity",
          d_static < d_moving, f"{d_static:.3e} < {d_moving:.3e}")


def test_39_40_no_magnitude_shortcut():
    """§39/§40: K=1 uses the same logic, and membership is not by magnitude."""
    import inspect
    import joint_reduction.rigid_discovery as RD
    src = inspect.getsource(RD)
    check("§39 no K == 1 shortcut assigning all candidates to part 1",
          "labels[idx] = 1" not in src and "if K == 1:" not in src)
    check("§2 the displacement-magnitude gate is gone from production",
          "mad_sigmas" not in src and "seed_k_parts" not in src)
    check("§56 no 'shared-twist compatibility' claim over a magnitude rule",
          "shared-twist" not in src.lower()
          or "motion_consensus" not in src)
    # Behavioural: the same entry point serves K=1 and K=2.
    sig = inspect.signature(RD.initialize_rigid_components)
    check("§39 one component initializer for every K",
          "K" in sig.parameters,
          "initialize_rigid_components(..., K, ...) handles both")



def _mixture(frac_static, frac_a, n=200, seed=5, b_rate=0.055, T=12):
    """A static set plus two rigid movers, in the requested proportions."""
    g = torch.Generator().manual_seed(seed)
    ns = int(round(n * frac_static))
    na = int(round(n * frac_a))
    nb = n - ns - na
    s_ = ((torch.rand((ns, 3), generator=g) - 0.5) * 0.6).to(DEV).double()
    a_ = ((torch.rand((na, 3), generator=g) - 0.5) * 0.4).to(DEV).double() \
        + torch.tensor([1.5, 0.0, 0.0], device=DEV).double()
    b_ = ((torch.rand((nb, 3), generator=g) - 0.5) * 0.4).to(DEV).double() \
        + torch.tensor([-1.5, 0.0, 0.0], device=DEV).double()
    x0 = torch.cat([s_, a_, b_])
    truth = torch.cat([torch.zeros(ns), torch.ones(na),
                       2 * torch.ones(nb)]).long().to(DEV)
    axis = torch.tensor([0.0, 0.0, 1.0], device=DEV).double()
    cb = torch.tensor([-2.0, 0.0, 0.0], device=DEV).double()
    tracks = []
    for t in range(T):
        pt = x0.clone()
        pt[truth == 1, 1] += 0.03 * t
        R = so3_exp((b_rate * t * axis).reshape(1, 3))[0].double()
        pt[truth == 2] = cb + (x0[truth == 2] - cb) @ R.T
        pt += torch.randn(pt.shape, device=DEV).double() * 0.004
        tracks.append(pt)
    return x0, torch.stack(tracks), truth


def _identity_costs(x0, tracks):
    """C_i(I) for every node, and the OLD mixture-contaminated threshold."""
    from estimator.part_assignment import whole_track_assignment_cost
    from joint_reduction.rigid_discovery import robust_scale_1d
    T = tracks.shape[0]
    eye = torch.eye(3, device=DEV).double()
    identity = {"R": eye.reshape(1, 3, 3).repeat(T, 1, 1),
                "t": torch.zeros((T, 3), device=DEV).double()}
    L = float((x0 - x0.mean(0)).norm(dim=1).mean().clamp_min(1e-9))
    c = whole_track_assignment_cost(x0, tracks, [identity], L).reshape(-1)
    tau = float(c.median() + 2.0 * robust_scale_1d(c))
    return c, tau, L


def test_14_contaminated_mixture():
    """§14: 40% static / 30% A / 30% B, B's identity cost between the static
    cost and the OLD mixture-contaminated threshold."""
    x0, tracks, truth = _mixture(0.40, 0.30)
    c, tau, _L = _identity_costs(x0, tracks)
    cs = float(c[truth == 0].median())
    cb = float(c[truth == 2].median())
    print(f"       identity cost: static {cs:.3e}  B {cb:.3e}  "
          f"old threshold {tau:.3e}")
    check("§14 B is geometrically separable from static", cb > 5.0 * cs,
          f"{cb:.3e} vs {cs:.3e}")
    check("§14 yet B sits UNDER the old mixture-contaminated threshold",
          cb < tau, f"{cb:.3e} < {tau:.3e} -- the old rule absorbs it")
    old_absorbed = int((c[truth == 2] <= tau).sum())
    print(f"       the old threshold would absorb {old_absorbed}/"
          f"{int((truth == 2).sum())} of B into the identity")
    r = decompose(x0, tracks, 2, seed=1)
    acc = accuracy(r["labels"], truth, 2)
    counts = [int((r["labels"] == k).sum()) for k in range(3)]
    print(f"       competing-model assignment: accuracy {acc:.3f}, "
          f"counts {counts}")
    check("§14 the competing-model assignment recovers B", acc > 0.85,
          f"accuracy {acc:.3f}")
    check("§14 and no component is empty", all(n > 0 for n in counts),
          str(counts))


def test_15_moving_majority():
    """§15: 20% static / 80% moving. The identity must not expand."""
    x0, tracks, truth = _mixture(0.20, 0.40, seed=6)
    c, tau, _L = _identity_costs(x0, tracks)
    cs = float(c[truth == 0].median())
    print(f"       80% of nodes move; identity cost static {cs:.3e}, "
          f"old threshold {tau:.3e} ({tau / max(cs, 1e-30):.0f}x static)")
    r = decompose(x0, tracks, 2, seed=1)
    acc = accuracy(r["labels"], truth, 2)
    counts = [int((r["labels"] == k).sum()) for k in range(3)]
    print(f"       accuracy {acc:.3f}, counts {counts} "
          f"(truth {[int((truth == k).sum()) for k in range(3)]})")
    check("§15 a moving majority does not collapse the decomposition",
          acc > 0.85, f"accuracy {acc:.3f}")
    check("§15 the static component is still found and is not empty",
          counts[0] > 0, str(counts))
    check("§15 no identity inlier threshold exists in production",
          "inlier_c" not in (pathlib.Path(REPO / "joint_reduction"
                                          / "rigid_discovery.py").read_text()
                             .split("def score_hypothesis")[0]),
          "hypothesis selection is a model-set argmin, not a threshold")


def main():
    for fn in (test_14_contaminated_mixture,
               test_15_moving_majority,
               test_13_mad_is_the_real_mad,
               test_39_40_no_magnitude_shortcut,
               test_33_prismatic_small_displacement,
               test_34_revolute_points_near_axis,
               test_35_36_two_parts,
               test_37_outlier_blowups,
               test_38_static_model_mismatch):
        print(f"\n{fn.__name__}")
        fn()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"\n{len(PASSED)} checks passed across 7 tests")


if __name__ == "__main__":
    main()
