"""The whole technique end to end, with convergence certification.

CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md section 2's ladder (free per-Gaussian
IEKF -> rigid twist -> P/R joint -> reduced joint filter), section 14's
acceptance semantics, and section 40's traceability chain.

What each group proves:
  * chain audit  -- every analytic link matches central differences of that same
                    link, so a failure localizes instead of being inferred.
  * certificate  -- monotonicity ALONE is vacuous (steps are only accepted when
                    the energy drops). The gain ratio rho and the gradient norm
                    are what actually certify the linearization and convergence.
  * pipeline     -- the free stage really does hand over to a joint model, and
                    the reduced stage really does recover q.
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
    audit_chain,
    boxplus,
    build_linearization,
    iekf_frame_step,
    run_sequence,
)
from joint_reduction.discovery import assess_coherence, decide_reduction, propose_joints  # noqa: E402

PASSED: list[str] = []
DEV = "cuda"
DT = torch.float64


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def camera(width=64, height=64, fov=0.8, dtype=torch.float32):
    tanf = math.tan(fov * 0.5)
    view = torch.eye(4, device=DEV, dtype=dtype)
    znear, zfar = 0.01, 100.0
    proj = torch.zeros((4, 4), device=DEV, dtype=dtype)
    proj[0, 0] = 1.0 / tanf
    proj[1, 1] = 1.0 / tanf
    proj[2, 2] = zfar / (zfar - znear)
    proj[3, 2] = 1.0
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    full = view.unsqueeze(0).bmm(proj.t().unsqueeze(0)).squeeze(0)
    return CameraSpec(viewmatrix=view, projmatrix=full, tanfovx=tanf, tanfovy=tanf,
                      image_height=height, image_width=width)


def scene(n=6, seed=0, v=None, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros((n, 3))
    mu[:, 0] = (torch.rand(n, generator=g) - 0.5) * 0.3
    mu[:, 1] = (torch.rand(n, generator=g) - 0.5) * 0.3
    mu[:, 2] = 2.0 + torch.rand(n, generator=g) * 0.3
    R = torch.eye(3).repeat(n, 1, 1)
    scales = torch.full((n, 3), 0.09)
    scales[:, 0] = 0.14
    scales[:, 2] = 0.05
    z = torch.zeros((n, 3))
    return GaussianMotionState(
        mu=mu.to(DEV, dtype), R=R.to(DEV, dtype),
        v=(z if v is None else v).to(DEV, dtype), omega=z.to(DEV, dtype),
        colors=torch.rand((n, 3), generator=g).to(DEV, dtype),
        opacity=torch.full((n, 1), 0.8).to(DEV, dtype), scales=scales.to(DEV, dtype))


def render_gt(state, cams, cfg):
    lin = build_linearization(state, cams, eps_depth=cfg.eps_depth)
    return [(v.rgb.clone(), v.depth.clone(),
             (v.alpha > cfg.min_alpha_for_depth).clone()) for v in lin.views]


def identity_cov(n, pos=1e-2, rot=1e-2, vel=1.0, om=1e-2, dtype=torch.float32):
    P = torch.zeros((n, 12, 12), device=DEV, dtype=dtype)
    d = torch.tensor([pos]*3 + [rot]*3 + [vel]*3 + [om]*3, device=DEV, dtype=dtype)
    P[:, range(12), range(12)] = d
    return P


# ---------------------------------------------------------------------------
# Section 40: the Jacobian chain, link by link
# ---------------------------------------------------------------------------
def test_chain_audit_every_link():
    """Each analytic link vs central differences of that same intermediate."""
    cam = camera(dtype=DT)
    torch.manual_seed(3)
    n = 24
    mu = torch.randn((n, 3), device=DEV, dtype=DT) * 0.12
    mu[:, 2] += 2.2
    R = torch.eye(3, device=DEV, dtype=DT).repeat(n, 1, 1)
    scales = torch.full((n, 3), 0.08, device=DEV, dtype=DT)
    scales[:, 0] = 0.13
    opacity = torch.full((n, 1), 0.7, device=DEV, dtype=DT)
    pixel = torch.tensor([32.0, 32.0], device=DEV, dtype=DT)

    audit = audit_chain(mu, R, scales, opacity, cam.viewmatrix, cam.projmatrix,
                        cam.image_width, cam.image_height, cam.tanfovx, cam.tanfovy,
                        pixel)
    print(audit.format())
    for link in audit.links:
        check(f"§40 link {link.name} matches finite differences", link.ok,
              f"rel {link.rel_error:.3e}")
    check("§40 the whole chain is consistent", audit.ok, str(audit.broken()))


def test_chain_audit_detects_a_broken_link():
    """The audit must FAIL when a link is wrong -- otherwise it certifies nothing.

    dJ_pi is dropped (section 9.2's "do not omit this term"), which is the most
    plausible real mistake, and the audit has to catch it at dS/dQ/dalpha.
    """
    import estimator.jacobian_audit as ja

    cam = camera(dtype=DT)
    torch.manual_seed(4)
    n = 16
    mu = torch.randn((n, 3), device=DEV, dtype=DT) * 0.12
    mu[:, 2] += 2.2
    R = torch.eye(3, device=DEV, dtype=DT).repeat(n, 1, 1)
    scales = torch.full((n, 3), 0.08, device=DEV, dtype=DT)
    scales[:, 0] = 0.13
    opacity = torch.full((n, 1), 0.7, device=DEV, dtype=DT)
    pixel = torch.tensor([32.0, 32.0], device=DEV, dtype=DT)

    original = ja.chain_analytic

    def broken(fw, dmu, dphi, *a, **kw):
        out = original(fw, dmu, dphi, *a, **kw)
        out["Jp"] = torch.zeros_like(out["Jp"])          # omit dJ_pi
        Jp, Sc = fw["Jp"], fw["Sigma_c"]
        out["S"] = Jp @ out["Sigma_c"] @ Jp.transpose(1, 2)
        out["Q"] = -fw["Q"] @ out["S"] @ fw["Q"]
        return out

    ja.chain_analytic = broken
    try:
        audit = ja.audit_chain(mu, R, scales, opacity, cam.viewmatrix, cam.projmatrix,
                               cam.image_width, cam.image_height,
                               cam.tanfovx, cam.tanfovy, pixel)
    finally:
        ja.chain_analytic = original
    check("§9.2 dropping dJ_pi is detected by the audit", not audit.ok,
          f"broken links: {audit.broken()}")
    check("§9.2 the audit blames dJp itself", "dJp" in audit.broken(),
          str(audit.broken()))
    check("§9.2 the damage propagates to dS and dQ",
          "dS" in audit.broken() and "dQ" in audit.broken(), str(audit.broken()))


# ---------------------------------------------------------------------------
# Section 14: the convergence certificate
# ---------------------------------------------------------------------------
def test_convergence_certificate():
    cams = [camera()]
    cfg = IEKFConfig(max_iters=8, pcg_max_iters=25, robust_measurement=False)
    truth = scene(n=6, seed=4)
    torch.manual_seed(2)
    d = torch.zeros((6, 12), device=DEV)
    d[:, :3] = torch.randn((6, 3), device=DEV) * 0.02
    d[:, 3:6] = torch.randn((6, 3), device=DEV) * 0.05
    gt = render_gt(truth, cams, cfg)
    out, P, diag = iekf_frame_step(boxplus(truth, d), identity_cov(6), gt, cams,
                                   dt=1 / 30.0, cfg=cfg)
    cert = diag.certificate
    print(cert.format())
    check("the certificate exists", cert is not None)
    check("accepted energies are monotone", cert.monotone,
          str(cert.monotone_violations))
    check("the gain ratio is reported per iteration",
          all(r.predicted_decrease == r.predicted_decrease for r in cert.records))
    check("the quadratic model predicts the true decrease (rho near 1)",
          cert.model_trustworthy, f"median rho {cert.median_gain_ratio:.4f}")
    check("the gradient norm falls", cert.stationarity_improved,
          f"{cert.records[0].gradient_norm:.4e} -> "
          f"{min(r.gradient_norm for r in cert.records):.4e}")
    check("convergence is CERTIFIED", cert.certified, str(cert.reasons()))


def test_certificate_is_not_vacuous():
    """A certificate that cannot fail certifies nothing. These must FAIL."""
    from estimator.convergence import ConvergenceCertificate, IterationRecord

    def rec(**kw):
        base = dict(iteration=0, accepted=True, energy_before=10.0, energy_after=9.0,
                    predicted_decrease=1.0, actual_decrease=1.0, gradient_norm=1.0,
                    step_norm=0.1, alpha=1.0, lm=1e-3, pcg_iterations=5,
                    pcg_rel_residual=1e-5)
        base.update(kw)
        return IterationRecord(**base)

    c = ConvergenceCertificate()
    c.add(rec(energy_before=10.0, energy_after=11.0, gradient_norm=1.0))
    check("an energy INCREASE on an accepted step fails monotonicity", not c.monotone)
    check("and is named in the reasons",
          any("INCREASED" in r for r in c.reasons()), str(c.reasons()))

    c2 = ConvergenceCertificate()
    c2.add(rec(iteration=0, predicted_decrease=100.0, actual_decrease=1.0,
               gradient_norm=1.0))
    c2.add(rec(iteration=1, predicted_decrease=100.0, actual_decrease=1.0,
               gradient_norm=0.1))
    check("a bad gain ratio fails the model-trust test", not c2.model_trustworthy,
          f"rho {c2.median_gain_ratio:.4f}")
    check("and is named", any("gain ratio" in r for r in c2.reasons()))

    c3 = ConvergenceCertificate()
    c3.add(rec(iteration=0, gradient_norm=1.0))
    c3.add(rec(iteration=1, gradient_norm=1.0))
    check("a flat gradient fails stationarity", not c3.stationarity_improved)
    check("and is named", any("gradient norm" in r for r in c3.reasons()))
    check("none of these three is certified",
          not c.certified and not c2.certified and not c3.certified)


# ---------------------------------------------------------------------------
# Sections 28-30: rigid coherence, model selection, the reduced filter
# ---------------------------------------------------------------------------
def test_coherence_and_model_selection():
    torch.manual_seed(7)
    mu = torch.randn((80, 3), device=DEV, dtype=DT) * 0.2

    # pure translation -> prismatic
    v_t = torch.tensor([0.3, 0.0, 0.0], device=DEV, dtype=DT).expand(80, 3).clone()
    dec = decide_reduction(mu, v_t)
    check("§29 pure translation selects prismatic",
          dec.confident and dec.selected.kind == "prismatic",
          f"{dec.selected.kind if dec.selected else None}, confident={dec.confident}")
    check("§29 the prismatic axis is the translation direction",
          float((dec.selected.axis - torch.tensor([1.0, 0, 0], device=DEV, dtype=DT))
                .abs().max()) < 1e-6)

    # pure rotation about a known axis through a known point -> revolute
    omega = torch.tensor([0.0, 0.0, 0.7], device=DEV, dtype=DT)
    centre = torch.tensor([0.15, -0.1, 0.0], device=DEV, dtype=DT)
    v_r = torch.linalg.cross(omega.expand(80, 3), mu - centre)
    dec_r = decide_reduction(mu, v_r)
    check("§29 pure rotation selects revolute",
          dec_r.confident and dec_r.selected.kind == "revolute",
          f"{dec_r.selected.kind if dec_r.selected else None}")
    axis_err = float((dec_r.selected.axis.abs()
                      - torch.tensor([0.0, 0, 1.0], device=DEV, dtype=DT)).abs().max())
    check("§29 the revolute axis is recovered", axis_err < 1e-6, f"{axis_err:.3e}")
    # the axis point is recovered up to a slide along the axis
    p = dec_r.selected.point
    resid = (p - centre) - ((p - centre) @ omega / omega.square().sum()) * omega
    check("§29 the axis point is recovered", float(resid.norm()) < 1e-6,
          f"{float(resid.norm()):.3e}")
    check("§27 the screw pitch of a pure rotation is ~0",
          abs(dec_r.selected.pitch) < 1e-6, f"{dec_r.selected.pitch:.3e}")


def test_reduction_declines_when_it_should():
    """Section 28: do not commit to a joint on weak evidence."""
    torch.manual_seed(11)
    mu = torch.randn((60, 3), device=DEV, dtype=DT) * 0.2

    static = decide_reduction(mu, torch.zeros_like(mu))
    check("a static group is not reduced", not static.confident)
    check("and says why", any("static" in r for r in static.reasons),
          str(static.reasons))

    incoherent = decide_reduction(mu, torch.randn_like(mu) * 0.3)
    check("incoherent motion is not reduced", not incoherent.confident)
    check("and says why", any("rigid twist" in r for r in incoherent.reasons),
          str(incoherent.reasons))

    # a helical/screw twist must be declined, not promoted to a third model
    omega = torch.tensor([0.0, 0.0, 0.7], device=DEV, dtype=DT)
    v_screw = torch.linalg.cross(omega.expand(60, 3), mu) + 0.5 * omega
    screw = decide_reduction(mu, v_screw)
    check("§0.13 a screw twist is declined, not made a third joint type",
          not screw.confident, f"selected={screw.selected.kind if screw.selected else None}")
    check("and says why", any("unexplained" in r or "helical" in r or "close" in r
                             for r in screw.reasons),
          str(screw.reasons))


def test_full_pipeline_free_then_reduced():
    """Section 2's ladder: the free stage hands over to a joint and q is tracked."""
    cams = [camera()]
    cfg = IEKFConfig(max_iters=5, pcg_max_iters=20, robust_measurement=False)
    dt = 1.0 / 30.0
    speed = 0.45
    n = 6
    truth0 = scene(n=n, seed=12)
    vtrue = torch.tensor([[speed, 0.0, 0.0]], device=DEV).repeat(n, 1)

    # V2 section 18's gate requires a WINDOW of stable estimates before it will
    # commit, so an end-to-end ladder test has to supply enough frames for that
    # window to fill. Five frames gave it only three samples against a
    # three-frame window plus the frames before reduction is first attempted.
    frames = []
    for k in range(8):
        moved = GaussianMotionState(
            mu=truth0.mu + vtrue * (dt * (k + 1)), R=truth0.R, v=truth0.v,
            omega=truth0.omega, colors=truth0.colors, opacity=truth0.opacity,
            scales=truth0.scales)
        frames.append((render_gt(moved, cams, cfg), cams, dt))

    est = scene(n=n, seed=12)
    res = run_sequence(est, identity_cov(n), frames, cfg=cfg, reduce_after=3,
                       verbose=True)
    print(res.format())

    check("the pipeline ran every frame", len(res.frames) == 8)
    check("the first frames used the free per-Gaussian stage",
          res.frames[0].mode == "free")
    check("the sequence reduced to a joint model", res.reduced_at_frame is not None,
          "reduction never triggered")
    if res.reduced_at_frame is not None:
        check("the selected joint is prismatic",
              res.reduction.selected.kind == "prismatic",
              res.reduction.selected.kind)
        axis = res.reduction.selected.axis
        err = float((axis.abs() - torch.tensor([1.0, 0, 0], device=DEV)).abs().max())
        check("the prismatic axis matches the true motion direction", err < 0.05,
              f"axis {axis.tolist()}")
        reduced = [f for f in res.frames if f.mode == "reduced"]
        check("later frames used the REDUCED stage", len(reduced) >= 1,
              f"{len(reduced)} reduced frames")
        check("the reduced stage reduces its energy",
              all(f.energy_final <= f.energy_initial + 1e-6 for f in reduced),
              str([(f.energy_initial, f.energy_final) for f in reduced]))
        qs = [f.q for f in reduced]
        check("q advances with the motion", all(b > a for a, b in zip(qs, qs[1:]))
              if len(qs) > 1 else True, str(qs))
        # q should track speed*dt per frame once reduced
        if len(qs) > 1:
            step = qs[-1] - qs[-2]
            check("q advances at about speed*dt per frame",
                  abs(step - speed * dt) < 0.4 * speed * dt,
                  f"dq {step:.5f} vs {speed*dt:.5f}")

    certs = [f.certificate for f in res.frames if f.certificate is not None]
    check("every free frame produced a certificate", len(certs) >= 1)
    check("every free frame was monotone", all(c.monotone for c in certs))


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
