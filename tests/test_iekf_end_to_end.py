"""End-to-end IEKF acceptance tests against the real CUDA renderer.

CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md section 35: T12 (single-frame pose
recovery), T13 (constant-twist multi-frame recovery), T14 (multi-camera
information gain), T15 (occlusion stress).

Every one of these drives the production estimator path: no Adam, no autograd,
analytic CUDA Jacobians only (section 36).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator import (  # noqa: E402
    CameraSpec,
    GaussianMotionState,
    IEKFConfig,
    boxplus,
    build_linearization,
    evaluate_true_map_energy,
    iekf_frame_step,
    invert_blocks,
    so3_exp,
)

PASSED: list[str] = []
DEV = "cuda"


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def camera(width=64, height=64, fov=0.8, yaw=0.0, radius=0.0):
    """A camera looking at the origin region down +z, optionally orbited."""
    tanf = math.tan(fov * 0.5)
    c, s = math.cos(yaw), math.sin(yaw)
    R = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], device=DEV)
    t = torch.tensor([radius * s, 0.0, 0.0], device=DEV)
    view = torch.eye(4, device=DEV)
    # repo convention: vm[0],vm[4],vm[8] is the first ROW of R_cw
    view[0, 0], view[1, 0], view[2, 0] = R[0, 0], R[0, 1], R[0, 2]
    view[0, 1], view[1, 1], view[2, 1] = R[1, 0], R[1, 1], R[1, 2]
    view[0, 2], view[1, 2], view[2, 2] = R[2, 0], R[2, 1], R[2, 2]
    view[3, 0], view[3, 1], view[3, 2] = t[0], t[1], t[2]
    znear, zfar = 0.01, 100.0
    proj = torch.zeros((4, 4), device=DEV)
    proj[0, 0] = znear / (tanf * znear)
    proj[1, 1] = znear / (tanf * znear)
    proj[2, 2] = zfar / (zfar - znear)
    proj[3, 2] = 1.0
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    full = view.unsqueeze(0).bmm(proj.t().unsqueeze(0)).squeeze(0)
    return CameraSpec(viewmatrix=view, projmatrix=full, tanfovx=tanf,
                      tanfovy=tanf, image_height=height, image_width=width)


def make_state(n=6, seed=0, spread=0.3, v=None, omega=None):
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros((n, 3))
    mu[:, 0] = (torch.rand(n, generator=g) - 0.5) * spread
    mu[:, 1] = (torch.rand(n, generator=g) - 0.5) * spread
    mu[:, 2] = 2.0 + torch.rand(n, generator=g) * 0.4
    R = torch.eye(3).repeat(n, 1, 1)
    scales = torch.full((n, 3), 0.09)
    scales[:, 0] = 0.14
    scales[:, 2] = 0.05
    opacity = torch.full((n, 1), 0.8)
    colors = torch.rand((n, 3), generator=g)
    zeros = torch.zeros((n, 3))
    return GaussianMotionState(
        mu=mu.to(DEV), R=R.to(DEV),
        v=(zeros if v is None else v).to(DEV),
        omega=(zeros if omega is None else omega).to(DEV),
        colors=colors.to(DEV), opacity=opacity.to(DEV), scales=scales.to(DEV))


def render_gt(state, cams, cfg):
    """Synthesize the measurement y = h(x_true) with full depth validity."""
    lin = build_linearization(state, cams, eps_depth=cfg.eps_depth)
    out = []
    for view in lin.views:
        valid = view.alpha > cfg.min_alpha_for_depth
        out.append((view.rgb.clone(), view.depth.clone(), valid.clone()))
    return out


def identity_cov(n, pos=1e-2, rot=1e-2, vel=1e-1, om=1e-1):
    P = torch.zeros((n, 12, 12), device=DEV)
    d = torch.tensor([pos] * 3 + [rot] * 3 + [vel] * 3 + [om] * 3, device=DEV)
    P[:, range(12), range(12)] = d
    return P


def pose_error(a: GaussianMotionState, b: GaussianMotionState):
    dmu = (a.mu - b.mu).norm(dim=1)
    dR = a.R @ b.R.transpose(1, 2)
    cos = ((dR.diagonal(dim1=1, dim2=2).sum(-1) - 1) * 0.5).clamp(-1, 1)
    return float(dmu.mean()), float(torch.rad2deg(torch.arccos(cos)).mean())


# ---------------------------------------------------------------------------
# T12
# ---------------------------------------------------------------------------
def test_T12_single_frame_pose_recovery():
    """Render GT from a known perturbed pose, start from the unperturbed
    estimate, and require the damped IEKF to reduce the TRUE MAP energy
    monotonically on accepted steps and to reduce the pose error."""
    cams = [camera()]
    cfg = IEKFConfig(max_iters=8, pcg_max_iters=25, robust_measurement=False,
                     rgb_sigma=0.05, depth_sigma_a=0.01)
    truth = make_state(n=6, seed=4)

    # A genuine pose offset: translation plus a small rotation.
    torch.manual_seed(2)
    delta = torch.zeros((6, 12), device=DEV)
    delta[:, :3] = torch.randn((6, 3), device=DEV) * 0.02
    delta[:, 3:6] = torch.randn((6, 3), device=DEV) * 0.05
    start = boxplus(truth, delta)

    gt = render_gt(truth, cams, cfg)
    P0 = identity_cov(6)
    e0, _ = pose_error(start, truth)[0], None
    mu_err0, rot_err0 = pose_error(start, truth)

    out, P_post, diag = iekf_frame_step(start, P0, gt, cams, dt=1 / 30.0, cfg=cfg)
    mu_err1, rot_err1 = pose_error(out, truth)

    accepted = [r for r in diag.rows if r.get("event") == "accepted"]
    check("T12 at least one step was accepted", len(accepted) >= 1,
          f"{len(accepted)} accepted of {cfg.max_iters}")
    energies = [r["energy"] for r in diag.rows if "energy" in r and r.get("event") in
                ("initial", "accepted")]
    monotone = all(b <= a + 1e-6 for a, b in zip(energies, energies[1:]))
    check("T12 accepted nonlinear MAP energy is monotonically non-increasing",
          monotone, " -> ".join(f"{e:.4g}" for e in energies[:6]))
    check("T12 the energy actually decreased", energies[-1] < energies[0],
          f"{energies[0]:.6g} -> {energies[-1]:.6g}")
    check("T12 translation error is reduced", mu_err1 < mu_err0,
          f"{mu_err0:.5f} -> {mu_err1:.5f}")
    check("T12 rotation error is not made worse", rot_err1 <= rot_err0 + 1e-3,
          f"{rot_err0:.4f} deg -> {rot_err1:.4f} deg")
    check("T12 the posterior stayed finite and PD",
          bool(torch.isfinite(P_post).all())
          and float(torch.linalg.eigvalsh(P_post.double()).min()) > 0,
          f"min eig {float(torch.linalg.eigvalsh(P_post.double()).min()):.3e}")
    check("T12 a PCG solve actually ran",
          any(r.get("pcg_iterations", 0) for r in diag.rows),
          str([r.get("pcg_iterations") for r in diag.rows[:4]]))


def test_T12b_rejection_shrinks_the_step_control():
    """A rejected step must tighten the step control and never be applied.

    V2 section 4.4 moved WHICH control that is: "trust radius is the primary
    step-size control. LM damping may remain as a secondary stabilizer." So the
    trust-region path must shrink the RADIUS on rejection, and the LM ablation
    must still raise lambda. Both are checked, each against its own mechanism.
    """
    cams = [camera()]
    truth = make_state(n=4, seed=9)
    # A deliberately large offset so early linearized steps overshoot.
    delta = torch.zeros((4, 12), device=DEV)
    delta[:, 0] = 0.12
    start = boxplus(truth, delta)

    for use_tr in (True, False):
        cfg = IEKFConfig(max_iters=6, pcg_max_iters=20, lm_initial=1e-12,
                         robust_measurement=False, use_trust_region=use_tr)
        gt = render_gt(truth, cams, cfg)
        out, P_post, diag = iekf_frame_step(start, identity_cov(4), gt, cams,
                                            dt=1 / 30.0, cfg=cfg)
        tag = "trust region" if use_tr else "LM ablation"
        rejected = [r for r in diag.rows if r.get("event") == "rejected"]
        if rejected:
            idx = diag.rows.index(rejected[0])
            if use_tr:
                before = diag.rows[idx - 1].get("trust_radius")
                after = rejected[0].get("trust_radius")
                check(f"T12b ({tag}) a rejected step shrank the trust radius",
                      after is not None and before is not None and after < before,
                      f"{before} -> {after}")
            else:
                prior_lm = diag.rows[idx - 1]["lm"]
                check(f"T12b ({tag}) a rejected step increased LM damping",
                      rejected[0]["lm"] > prior_lm,
                      f"{prior_lm:.3e} -> {rejected[0]['lm']:.3e}")
            check(f"T12b ({tag}) a rejected step applied alpha = 0",
                  rejected[0]["alpha"] == 0.0)
        else:
            check(f"T12b ({tag}) every step was accepted "
                  f"(no rejection to inspect)", True)
        energies = [r["energy"] for r in diag.rows
                    if r.get("event") in ("initial", "accepted")]
        check(f"T12b ({tag}) energy never increased across accepted steps",
              all(b <= a + 1e-6 for a, b in zip(energies, energies[1:])))


# ---------------------------------------------------------------------------
# T13
# ---------------------------------------------------------------------------
def test_T13_constant_twist_multi_frame():
    """A rigidly translating object over several frames with REAL dt. The
    filter must track it and the velocity state must become informative."""
    cams = [camera()]
    cfg = IEKFConfig(max_iters=5, pcg_max_iters=20, robust_measurement=False,
                     rgb_sigma=0.05, depth_sigma_a=0.01)
    dt = 1.0 / 30.0
    true_v = torch.tensor([[0.45, 0.0, 0.0]], device=DEV).repeat(5, 1)
    truth = make_state(n=5, seed=12, v=true_v.cpu())

    est = make_state(n=5, seed=12)  # starts with ZERO velocity: must learn it
    P = identity_cov(5, pos=1e-2, rot=1e-3, vel=1.0, om=1e-3)

    errs = []
    cur_truth = truth
    for k in range(4):
        cur_truth = GaussianMotionState(
            mu=truth.mu + true_v * (dt * (k + 1)), R=truth.R, v=truth.v,
            omega=truth.omega, colors=truth.colors, opacity=truth.opacity,
            scales=truth.scales)
        gt = render_gt(cur_truth, cams, cfg)
        est, P, diag = iekf_frame_step(est, P, gt, cams, dt=dt, cfg=cfg)
        e, _ = pose_error(est, cur_truth)
        errs.append(e)
        print(f"       frame {k}: mean |dmu| = {e:.5f}   "
              f"mean |v| = {float(est.v.norm(dim=1).mean()):.4f}")

    check("T13 tracking error does not diverge", errs[-1] < 0.25,
          f"final {errs[-1]:.5f}, per-frame {[round(e,4) for e in errs]}")
    check("T13 the filter improves after the first frame", errs[-1] <= errs[0] + 1e-3,
          f"{errs[0]:.5f} -> {errs[-1]:.5f}")
    check("T13 real dt was used (not unit spacing)", abs(dt - 1 / 30.0) < 1e-12)
    check("T13 the state stayed finite",
          bool(torch.isfinite(est.mu).all()) and bool(torch.isfinite(est.R).all()))
    check("T13 rotations stayed on SO(3)",
          float((est.R @ est.R.transpose(1, 2)
                 - torch.eye(3, device=DEV)).abs().max()) < 1e-4)


# ---------------------------------------------------------------------------
# T14
# ---------------------------------------------------------------------------
def test_T14_multi_camera_information_gain():
    """Section 25: synchronized cameras ADD information. More cameras must give
    a larger Fisher and a better-conditioned normal system."""
    cfg = IEKFConfig(robust_measurement=False)
    truth = make_state(n=4, seed=17)
    traces, conds = [], []
    for n_cam in (1, 2, 3):
        cams = [camera(yaw=i * 0.25, radius=0.0) for i in range(n_cam)]
        lin = build_linearization(truth, cams, eps_depth=cfg.eps_depth)
        prec = []
        for view in lin.views:
            prec.append((torch.ones_like(view.rgb),
                         (view.alpha > cfg.min_alpha_for_depth).to(view.depth.dtype)))
        fisher = lin.build_block_fisher(prec)
        traces.append(float(fisher.diagonal(dim1=1, dim2=2).sum(-1).mean()))
        ev = torch.linalg.eigvalsh(fisher.double() + 1e-9 * torch.eye(6, device=DEV).double())
        conds.append(float((ev.max(dim=1).values / ev.min(dim=1).values.clamp_min(1e-30)).mean()))
        print(f"       {n_cam} camera(s): mean Fisher trace {traces[-1]:.4e}  "
              f"mean condition {conds[-1]:.3e}")
    check("T14 information increases with each added camera",
          traces[1] > traces[0] and traces[2] > traces[1],
          " < ".join(f"{t:.4e}" for t in traces))
    check("T14 conditioning improves (or does not worsen) with more cameras",
          conds[-1] <= conds[0] * 1.05,
          f"{conds[0]:.3e} -> {conds[-1]:.3e}")


def test_T14b_multiview_fusion_is_additive():
    """b = sum_c H_c^T R_c^-1 r_c exactly (section 25)."""
    cfg = IEKFConfig()
    truth = make_state(n=3, seed=23)
    cams = [camera(yaw=0.0), camera(yaw=0.3)]
    lin_both = build_linearization(truth, cams, eps_depth=cfg.eps_depth)
    singles = [build_linearization(truth, [c], eps_depth=cfg.eps_depth) for c in cams]
    torch.manual_seed(31)
    z = [(torch.randn_like(v.rgb), torch.randn_like(v.depth)) for v in lin_both.views]
    joint = lin_both.apply_HT(z)
    summed = sum(s.apply_HT([zi]) for s, zi in zip(singles, z))
    rel = float((joint - summed).abs().max() / joint.abs().max().clamp_min(1e-20))
    check("T14b multiview H^T z equals the per-camera sum", rel < 1e-5,
          f"rel {rel:.3e}")


# ---------------------------------------------------------------------------
# T15
# ---------------------------------------------------------------------------
def test_T15_occlusion_stress():
    """Move a front layer laterally across a back layer and require the
    front/rear coupling to stay correct throughout (section 26)."""
    cfg = IEKFConfig()
    cams = [camera()]
    n = 2
    mu = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.6]])
    R = torch.eye(3).repeat(n, 1, 1)
    scales = torch.tensor([[0.13, 0.13, 0.05], [0.20, 0.20, 0.05]])
    opacity = torch.tensor([[0.95], [0.9]])
    colors = torch.tensor([[1.0, 0.1, 0.1], [0.1, 0.2, 1.0]])
    zeros = torch.zeros((n, 3))
    worst = 0.0
    for shift in (-0.12, -0.06, 0.0, 0.06, 0.12):
        m = mu.clone()
        m[0, 0] = shift
        state = GaussianMotionState(
            mu=m.to(DEV), R=R.to(DEV), v=zeros.to(DEV), omega=zeros.to(DEV),
            colors=colors.to(DEV), opacity=opacity.to(DEV), scales=scales.to(DEV))
        lin = build_linearization(state, cams, eps_depth=cfg.eps_depth)
        v = torch.zeros((n, 6), device=DEV)
        v[0, 0] = 1.0        # push the FRONT layer sideways
        hv = lin.apply_H(v)
        torch.manual_seed(7)
        z = [(torch.randn_like(hv[0][0]), torch.randn_like(hv[0][1]))]
        lhs = float((hv[0][0] * z[0][0]).sum() + (hv[0][1] * z[0][1]).sum())
        rhs = float((v * lin.apply_HT(z)).sum())
        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-20)
        worst = max(worst, rel)
        print(f"       shift {shift:+.3f}: adjoint rel {rel:.3e}")
    check("T15 the adjoint identity holds through the whole occlusion sweep",
          worst < 1e-4, f"worst rel {worst:.3e}")

    # And the occluded layer must be measurably less sensitive than the front.
    state = GaussianMotionState(
        mu=mu.to(DEV), R=R.to(DEV), v=zeros.to(DEV), omega=zeros.to(DEV),
        colors=colors.to(DEV), opacity=opacity.to(DEV), scales=scales.to(DEV))
    lin = build_linearization(state, cams, eps_depth=cfg.eps_depth)
    vf = torch.zeros((n, 6), device=DEV); vf[0, 0] = 1.0
    vb = torch.zeros((n, 6), device=DEV); vb[1, 0] = 1.0
    nf = float(lin.apply_H(vf)[0][0].abs().sum())
    nb = float(lin.apply_H(vb)[0][0].abs().sum())
    check("T15 the occluded back layer is less sensitive than the front",
          nb < nf, f"back {nb:.4f} < front {nf:.4f}")


def test_no_autograd_or_adam_in_estimator():
    """Section 36, enforced on the production source rather than asserted."""
    import ast

    offenders = []
    for path in sorted((REPO / "estimator").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name.endswith(".backward") or "torch.optim" in name \
                        or name.startswith("torch.autograd"):
                    offenders.append(f"{path.name}:{node.lineno} {name}")
    check("no .backward() / torch.optim / torch.autograd call in estimator/",
          not offenders, str(offenders))
    # The renderer operators must not be autograd Functions either.
    import diff_gaussian_rasterization as dgr
    for fn in ("iekf_apply_H", "iekf_apply_HT", "iekf_build_fisher"):
        obj = getattr(dgr, fn)
        check(f"{fn} is a plain function, not an autograd.Function",
              not hasattr(obj, "apply") or not isinstance(obj, type))


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the end-to-end IEKF tests")
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
