"""V2 section 3.3: the CUDA supernode kernels must be the true G and G^T.

The CPU reference lives in test_v2_trust_region_and_lines.py. This file checks
that the kernels actually compiled into diff_gaussian_rasterization agree with
it, and that the adjoint identity <Gv, z> = <v, G^T z> holds on device.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from diff_gaussian_rasterization import _C  # noqa: E402
from estimator.supernodes import GroupedObservation, build_supernodes  # noqa: E402

PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def cpu_expand(sid, means, cent, dz):
    c = sid.long()
    d = means - cent[c]
    t, th = dz[c, :3], dz[c, 3:]
    return torch.cat([t + torch.linalg.cross(th, d), th], dim=1)


def cpu_reduce(sid, means, cent, adj, C):
    c = sid.long()
    d = means - cent[c]
    mb, pb = adj[:, :3], adj[:, 3:]
    out = torch.zeros((C, 6), device=adj.device, dtype=adj.dtype)
    out.index_add_(0, c, torch.cat([mb, torch.linalg.cross(d, mb) + pb], dim=1))
    return out


def setup(n=5000, seed=0):
    torch.manual_seed(seed)
    means = (torch.randn((n, 3), device="cuda") * 2.0).contiguous()
    part = build_supernodes(means, target_size=64, seed=seed)
    return means, part


def test_expand_matches_reference():
    means, part = setup()
    sid, cent = part.supernode_id, part.centroids
    C = part.num_supernodes
    print(f"       {part.num_gaussians} Gaussians -> {C} supernodes "
          f"({part.num_gaussians / C:.1f} members avg)")
    torch.manual_seed(7)
    dz = torch.randn((C, 6), device="cuda")
    got = _C.iekf_expand_group_tangent(sid, means, cent, dz, False)
    want = cpu_expand(sid, means, cent, dz)
    err = float((got - want).abs().max())
    check("§3.2 the CUDA group expansion matches the reference tangent",
          err < 1e-5, f"max abs {err:.3e}")
    check("§3.2 the phi block is the group rotation verbatim",
          float((got[:, 3:] - dz[sid.long(), 3:]).abs().max()) < 1e-6)


def test_reduce_matches_reference():
    means, part = setup(seed=1)
    sid, cent = part.supernode_id, part.centroids
    C = part.num_supernodes
    torch.manual_seed(8)
    adj = torch.randn((part.num_gaussians, 6), device="cuda")
    got = _C.iekf_reduce_group_adjoint(sid, means, cent, adj, C, False)
    want = cpu_reduce(sid, means, cent, adj, C)
    rel = float((got - want).abs().max() / want.abs().max())
    check("§3.3 the CUDA group adjoint matches the reference reduction",
          rel < 1e-4, f"max rel {rel:.3e}")


def test_adjoint_identity_on_device():
    """<G v, z> = <v, G^T z>. If the kernels disagree the IEKF's gradient is
    simply wrong, so this is the gate section 3.3 cares about."""
    means, part = setup(n=20000, seed=2)
    sid, cent = part.supernode_id, part.centroids
    C = part.num_supernodes
    for trial in range(5):
        torch.manual_seed(100 + trial)
        v = torch.randn((C, 6), device="cuda")
        z = torch.randn((part.num_gaussians, 6), device="cuda")
        Gv = _C.iekf_expand_group_tangent(sid, means, cent, v, False)
        GTz = _C.iekf_reduce_group_adjoint(sid, means, cent, z, C, False)
        lhs = float((Gv.double() * z.double()).sum())
        rhs = float((v.double() * GTz.double()).sum())
        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
        check(f"§3.3 adjoint identity holds (trial {trial})", rel < 1e-5,
              f"{lhs:.6f} vs {rhs:.6f}, rel {rel:.2e}")


def test_fisher_congruence():
    """F_c = sum_i G_i^T F_i G_i, and it must stay positive semi-definite."""
    means, part = setup(n=3000, seed=3)
    sid, cent = part.supernode_id, part.centroids
    C, N = part.num_supernodes, part.num_gaussians

    torch.manual_seed(12)
    # build genuinely PSD per-Gaussian blocks
    L = torch.randn((N, 6, 6), device="cuda") * 0.3
    F = L @ L.transpose(1, 2)
    idx = torch.triu_indices(6, 6, device="cuda")
    packed = F[:, idx[0], idx[1]].contiguous()

    grouped = _C.iekf_reduce_group_fisher(sid, means, cent, packed, C, False)
    out = torch.zeros((C, 6, 6), device="cuda")
    t = 0
    for a in range(6):
        for b in range(a, 6):
            out[:, a, b] = grouped[:, t]
            out[:, b, a] = grouped[:, t]
            t += 1

    # reference
    c = sid.long()
    d = means - cent[c]
    G = torch.zeros((N, 6, 6), device="cuda")
    G[:, 0, 0] = G[:, 1, 1] = G[:, 2, 2] = 1.0
    G[:, 3, 3] = G[:, 4, 4] = G[:, 5, 5] = 1.0
    G[:, 0, 4], G[:, 0, 5] = d[:, 2], -d[:, 1]
    G[:, 1, 3], G[:, 1, 5] = -d[:, 2], d[:, 0]
    G[:, 2, 3], G[:, 2, 4] = d[:, 1], -d[:, 0]
    ref = torch.zeros((C, 6, 6), device="cuda")
    ref.index_add_(0, c, G.transpose(1, 2) @ F @ G)

    rel = float((out - ref).abs().max() / ref.abs().max())
    check("§3.3 the group Fisher is the congruence G^T F G", rel < 1e-3,
          f"max rel {rel:.3e}")
    ev = torch.linalg.eigvalsh(out.double())
    check("§3.3 the grouped Fisher stays positive semi-definite",
          float(ev.min()) > -1e-3 * float(ev.max()),
          f"min eig {float(ev.min()):.3e} vs max {float(ev.max()):.3e}")

    # And the whole point of grouping: conditioning must improve.
    ev_i = torch.linalg.eigvalsh(F.double())
    cond_i = float((ev_i.max(dim=1).values
                    / ev_i.min(dim=1).values.clamp_min(1e-30)).median())
    cond_c = float((ev.max(dim=1).values
                    / ev.min(dim=1).values.clamp_min(1e-30)).median())
    print(f"       median block condition: per-Gaussian {cond_i:.3e} -> "
          f"per-supernode {cond_c:.3e}")
    check("§3 grouping improves the median block condition number",
          cond_c < cond_i, f"{cond_i:.3e} -> {cond_c:.3e}")


def test_partition_is_deterministic_and_canonical():
    means, p1 = setup(n=8000, seed=5)
    p2 = build_supernodes(means, target_size=64, seed=5)
    check("§3.1 the partition is deterministic for a fixed seed+config",
          p1.canonical_hash() == p2.canonical_hash(), p1.canonical_hash()[:16])
    p3 = build_supernodes(means, target_size=32, seed=5)
    check("§3.1 a different config gives a different canonical hash",
          p1.canonical_hash() != p3.canonical_hash())
    check("§3.1 every Gaussian is assigned",
          int((p1.supernode_id < 0).sum()) == 0)
    check("§3.1 no supernode is empty",
          int(p1.member_counts.min()) > 0, f"min {int(p1.member_counts.min())}")
    check("§3.1 the counts sum to N",
          int(p1.member_counts.sum()) == p1.num_gaussians)
    check("§3.1 the adjacency graph is non-trivial",
          int(p1.adjacency.shape[0]) > p1.num_supernodes,
          f"{int(p1.adjacency.shape[0])} edges / {p1.num_supernodes} nodes")
    check("§3.1 adjacency is stored canonically (a<b, sorted, unique)",
          bool((p1.adjacency[:, 0] < p1.adjacency[:, 1]).all()))
    # the partition must NOT depend on motion -- it only ever saw frame-0 means
    check("§3.1 the partition is built from canonical geometry only",
          p1.canonical_means.shape == means.shape
          and float((p1.canonical_means - means).abs().max()) == 0.0)


def test_unassigned_gaussians_are_inert():
    means, part = setup(n=2000, seed=6)
    sid = part.supernode_id.clone()
    sid[:50] = -1                      # mark some unassigned
    C = part.num_supernodes
    torch.manual_seed(31)
    dz = torch.randn((C, 6), device="cuda")
    got = _C.iekf_expand_group_tangent(sid, means, part.centroids, dz, False)
    check("§3.2 unassigned Gaussians get an exactly zero tangent",
          float(got[:50].abs().max()) == 0.0)
    check("§3.2 assigned Gaussians are unaffected by them",
          float(got[50:].abs().max()) > 0.0)
    adj = torch.randn((part.num_gaussians, 6), device="cuda")
    g_all = _C.iekf_reduce_group_adjoint(sid, means, part.centroids, adj, C, False)
    # The reduction sums with atomicAdd, whose ordering varies per launch, so
    # the comparison floor is a repeat of the SAME input -- not exact equality.
    # One repeat under-samples that noise, so take the worst of several.
    floor = 0.0
    for _ in range(5):
        g_rep = _C.iekf_reduce_group_adjoint(sid, means, part.centroids, adj, C, False)
        floor = max(floor, float((g_all - g_rep).abs().max()))
    adj2 = adj.clone()
    adj2[:50] = torch.randn((50, 6), device="cuda") * 1e3
    g_two = _C.iekf_reduce_group_adjoint(sid, means, part.centroids, adj2, C, False)
    delta = float((g_all - g_two).abs().max())
    # The substantive claim: perturbing the unassigned rows by 1e3 must move the
    # result by no more than the summation noise does, and by far less than the
    # injected magnitude would if it leaked through.
    check("§3.2 and contribute nothing to the group adjoint",
          delta <= max(4.0 * floor, 1e-4) and delta < 1e-3 * float(g_all.abs().max()),
          f"delta {delta:.2e} vs atomic floor {floor:.2e} "
          f"(output scale {float(g_all.abs().max()):.2e}, injected 1e3)")


def test_grouped_observation_wrapper():
    means, part = setup(n=4000, seed=4)
    go = GroupedObservation(linearization=None, partition=part)
    torch.manual_seed(44)
    v = torch.randn((part.num_supernodes, 6), device="cuda")
    z = torch.randn((part.num_gaussians, 6), device="cuda")
    lhs = float((go.expand(v).double() * z.double()).sum())
    rhs = float((v.double() * go.reduce(z).double()).sum())
    check("the GroupedObservation wrapper preserves the adjoint identity",
          abs(lhs - rhs) / max(abs(lhs), 1e-30) < 1e-5, f"{lhs:.5f} vs {rhs:.5f}")
    check("the wrapper exposes the supernode count",
          go.num_supernodes == part.num_supernodes)


def test_scale_11320():
    """Section 1.2's full-scene size: the state drops from 6N to 6C."""
    means, part = setup(n=11320, seed=99)
    free = 6 * part.num_gaussians
    grouped = 6 * part.num_supernodes
    print(f"       free state {free} DoF -> grouped state {grouped} DoF "
          f"({free / grouped:.0f}x reduction)")
    check("§3 the full-scene state is reduced by more than an order of magnitude",
          grouped * 10 < free, f"{free} -> {grouped}")
    torch.manual_seed(77)
    v = torch.randn((part.num_supernodes, 6), device="cuda")
    z = torch.randn((part.num_gaussians, 6), device="cuda")
    Gv = _C.iekf_expand_group_tangent(part.supernode_id, means, part.centroids,
                                      v, False)
    GTz = _C.iekf_reduce_group_adjoint(part.supernode_id, means, part.centroids,
                                       z, part.num_supernodes, False)
    lhs = float((Gv.double() * z.double()).sum())
    rhs = float((v.double() * GTz.double()).sum())
    check("§3 the adjoint identity holds at full scene scale",
          abs(lhs - rhs) / max(abs(lhs), 1e-30) < 1e-5, f"rel "
          f"{abs(lhs - rhs) / max(abs(lhs), 1e-30):.2e}")


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
