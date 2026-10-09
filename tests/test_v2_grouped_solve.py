"""V2 section 3: the grouped supernode state must actually drive the solve.

Section 3 does not ask for a partition to be reported alongside a free-Gaussian
solve -- it asks for the free state to be REPLACED. A flag that is parsed but
does not change the estimator is the dead flag section 13 condemns, so these
tests check the solved dimension, the adjoint of the composed operator, and
that the grouped filter actually recovers rigid motion.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.renderer_bridge import CameraSpec, build_linearization  # noqa: E402
from estimator.sparse_iekf import IEKFConfig, iekf_frame_step  # noqa: E402
from estimator.state import GaussianMotionState  # noqa: E402
from estimator.supernodes import GroupedObservation, build_supernodes  # noqa: E402

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


def scene(n=1024, seed=0, v=None):
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros((n, 3))
    mu[:, 0] = (torch.rand(n, generator=g) - 0.5) * 0.7
    mu[:, 1] = (torch.rand(n, generator=g) - 0.5) * 0.7
    mu[:, 2] = 2.0 + torch.rand(n, generator=g) * 0.3
    z = torch.zeros((n, 3))
    return GaussianMotionState(
        mu=mu.to(DEV), R=torch.eye(3).repeat(n, 1, 1).to(DEV),
        v=(z if v is None else v).to(DEV), omega=z.to(DEV),
        colors=torch.rand((n, 3), generator=g).to(DEV),
        opacity=torch.full((n, 1), 0.8).to(DEV),
        scales=torch.full((n, 3), 0.045).to(DEV))


def render_gt(state, cams, cfg):
    lin = build_linearization(state, cams, eps_depth=cfg.eps_depth)
    return [(v.rgb.clone(), v.depth.clone(),
             (v.alpha > cfg.min_alpha_for_depth).clone()) for v in lin.views]


def identity_cov(n):
    P = torch.zeros((n, 12, 12), device=DEV)
    P[:, range(12), range(12)] = torch.tensor(
        [1e-2] * 3 + [1e-2] * 3 + [1.0] * 3 + [1e-1] * 3, device=DEV)
    return P


def moved(state, d):
    return GaussianMotionState(mu=state.mu + d, R=state.R, v=state.v,
                               omega=state.omega, colors=state.colors,
                               opacity=state.opacity, scales=state.scales)


def test_grouped_solve_changes_the_solved_dimension():
    """§3: the flag must change the state that is actually solved for."""
    cams = [camera()]
    truth = scene()
    n = truth.num_gaussians
    d = torch.zeros_like(truth.mu)
    d[:, 0] = 0.01
    gt = render_gt(moved(truth, d), cams, IEKFConfig())

    rows = {}
    for use in (False, True):
        cfg = IEKFConfig(max_iters=2, pcg_max_iters=30, use_supernodes=use,
                         supernode_target_size=64)
        _, _, diag = iekf_frame_step(truth, identity_cov(n), gt, cams,
                                     1 / 30.0, cfg)
        init = next(r for r in diag.rows if r.get("event") == "initial")
        rows[use] = init
        print(f"       use_supernodes={str(use):<5} "
              f"parameterization={init['state_parameterization']:<20} "
              f"solved {init['solved_dimension']:>6} of {init['free_dimension']} DoF")

    check("§3 the free run solves the per-Gaussian state",
          rows[False]["state_parameterization"] == "free_per_gaussian"
          and rows[False]["solved_dimension"] == 12 * n)
    check("§3 the grouped run solves the supernode state",
          rows[True]["state_parameterization"] == "supernode_grouped")
    check("§3 the grouped state is more than an order of magnitude smaller",
          rows[True]["solved_dimension"] * 10 < rows[False]["solved_dimension"],
          f"{rows[False]['solved_dimension']} -> {rows[True]['solved_dimension']}")
    check("§3 the flag is NOT dead: it changes the solved dimension",
          rows[True]["solved_dimension"] != rows[False]["solved_dimension"])


def test_grouped_operator_keeps_the_adjoint_identity():
    """The composed operator G^T H^T R^-1 H G must still be self-adjoint, and
    G^T must be the exact transpose of G on the 12D state (both blocks)."""
    cams = [camera(), camera(tz=0.2)]
    st = scene(n=2048, seed=3)
    part = build_supernodes(st.mu, target_size=64, seed=0)
    lin = build_linearization(st, cams)
    go = GroupedObservation(linearization=lin, partition=part)

    def G(vg):
        return torch.cat([go.expand(vg[:, :6]), go.expand(vg[:, 6:])], dim=1)

    def GT(vf):
        return torch.cat([go.reduce(vf[:, :6].contiguous()),
                          go.reduce(vf[:, 6:].contiguous())], dim=1)

    for trial in range(4):
        torch.manual_seed(50 + trial)
        vg = torch.randn((part.num_supernodes, 12), device=DEV)
        zf = torch.randn((part.num_gaussians, 12), device=DEV)
        lhs = float((G(vg).double() * zf.double()).sum())
        rhs = float((vg.double() * GT(zf).double()).sum())
        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
        check(f"§3.3 the 12D group map is self-adjoint (trial {trial})",
              rel < 1e-5, f"{lhs:.5f} vs {rhs:.5f}, rel {rel:.2e}")

    # and the velocity block uses the SAME rigid structure as the pose block
    torch.manual_seed(7)
    vg = torch.randn((part.num_supernodes, 12), device=DEV)
    vg[:, 6:] = vg[:, :6]
    full = G(vg)
    check("§3.2 the velocity block is grouped by the same rigid map as pose",
          float((full[:, :6] - full[:, 6:]).abs().max()) < 1e-5,
          f"max diff {float((full[:, :6] - full[:, 6:]).abs().max()):.3e}")


def test_grouped_filter_recovers_a_rigid_translation():
    """The point of the grouping: a rigid motion is exactly representable in
    the grouped state, so the filter should recover it."""
    cams = [camera(), camera(tz=0.2)]
    truth = scene(n=2048, seed=5)
    n = truth.num_gaussians
    shift = 0.012
    d = torch.zeros_like(truth.mu)
    d[:, 0] = shift
    cfg_gt = IEKFConfig()
    gt = render_gt(moved(truth, d), cams, cfg_gt)

    out = {}
    for use in (True, False):
        cfg = IEKFConfig(max_iters=6, pcg_max_iters=60, use_supernodes=use,
                         supernode_target_size=64)
        est, _, diag = iekf_frame_step(truth, identity_cov(n), gt, cams,
                                       1 / 30.0, cfg)
        rows = [r for r in diag.rows if r.get("event") in ("initial", "accepted",
                                                           "rejected")]
        e0, eN = rows[0]["energy"], rows[-1]["energy"]
        recovered = float((est.mu - truth.mu)[:, 0].mean())
        err = float((est.mu - (truth.mu + d)).norm(dim=1).mean())
        out[use] = (e0, eN, recovered, err)
        print(f"       {'grouped ' if use else 'free    '}: energy "
              f"{e0:9.2f} -> {eN:8.2f}   mean dx {recovered:.5f} "
              f"(true {shift})   residual error {err:.5f}")

    e0, eN, rec, err = out[True]
    _, eN_free, rec_free, err_free = out[False]
    check("§3 the grouped filter reduces the objective", eN < e0,
          f"{e0:.2f} -> {eN:.2f}")
    check("§3 the grouped filter recovers most of the translation",
          abs(rec - shift) < 0.5 * shift,
          f"recovered {rec:.5f} of {shift} ({100 * rec / shift:.0f}%)")
    check("§3 and the remaining per-Gaussian error is below the motion",
          err < 0.5 * shift, f"{err:.5f} vs motion {shift}")
    # The substantive section 3 claim: on a rigid motion the grouped state --
    # in which that motion is exactly representable -- beats the free state,
    # which has to discover the same motion independently per Gaussian from a
    # structurally ill-conditioned system.
    check("§3 grouping beats the free per-Gaussian state on a rigid motion",
          eN < eN_free and err < err_free,
          f"energy {eN:.2f} vs {eN_free:.2f}; residual {err:.5f} vs {err_free:.5f}")
    check("§3 and recovers more of the true displacement",
          abs(rec - shift) < abs(rec_free - shift),
          f"{100 * rec / shift:.0f}% vs {100 * rec_free / shift:.0f}%")


def test_grouped_partition_is_reused_not_rebuilt():
    """§3.1: the partition is immutable. Passing one in must be honoured."""
    cams = [camera()]
    truth = scene(n=512, seed=11)
    n = truth.num_gaussians
    d = torch.zeros_like(truth.mu)
    d[:, 0] = 0.01
    gt = render_gt(moved(truth, d), cams, IEKFConfig())
    part = build_supernodes(truth.mu, num_supernodes=7, seed=0)
    h0 = part.canonical_hash()
    cfg = IEKFConfig(max_iters=2, pcg_max_iters=20, use_supernodes=True,
                     supernode_target_size=64)
    _, _, diag = iekf_frame_step(truth, identity_cov(n), gt, cams, 1 / 30.0,
                                 cfg, partition=part)
    init = next(r for r in diag.rows if r.get("event") == "initial")
    check("§3.1 the supplied partition is used, not a freshly built one",
          init["solved_dimension"] == 12 * 7,
          f"solved {init['solved_dimension']} DoF for 7 supernodes")
    check("§3.1 and the partition was not mutated",
          part.canonical_hash() == h0)


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
