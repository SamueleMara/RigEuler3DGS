"""Sections 24, 40: the composed renderer tests, H_J = H_G J_{G,J}.

CODEX_IMPLEMENT_PHASE2_RIGID_SCREW_IEKF.md.

These run the REAL CUDA rasterizer. The point of section 40 is that the joint
layer composes with Phase 1's analytic operators without either side being
re-derived: the renderer JVP/VJP are the existing corrected ones, and the only
new factor is `J_{G,J}`.

    R1  H_G J p   against central finite differences of the rendered image
    R2  <H_G J p, z> == <p, J^T H_G^T z>
    R3  RGB and depth separately
    R4  a static Gaussian's derivative is exactly zero
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.joint_kinematics import (full_jvp, full_vjp,  # noqa: E402
                                        state_to_gaussians)
from estimator.joint_manifold import retract_joint_geometry  # noqa: E402
from estimator.joint_renderer_bridge import (joint_apply_H,  # noqa: E402
                                             joint_apply_HT)
from estimator.joint_state import JointState, initialise_from_twist  # noqa: E402
from estimator.renderer_bridge import build_linearization  # noqa: E402
from tests.test_strict_trajectory import Cfg, DEV, camera, scene_of  # noqa: E402

PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def setup(n=260, moving=160, seed=0, xi=(0.25, -0.15, 0.08, 0.10, 0.22, -0.12)):
    truth = scene_of(n, seed)
    cams = [camera(), camera(tz=0.12)]
    mu0 = truth.mu.double()
    members = torch.arange(0, moving)
    part = initialise_from_twist(members, mu0,
                                 torch.tensor(xi, dtype=torch.float64),
                                 torch.linspace(0.0, 0.9, 8), None)
    st = JointState([part], torch.arange(8, dtype=torch.float64))
    return truth, cams, mu0, st


def render(truth, cams, st, frame, mu0):
    """One linearization of the CURRENT joint state -- Phase 1's own call."""
    import dataclasses
    mu, _Rg = state_to_gaussians(st, frame, mu0, None)
    state = dataclasses.replace(truth, mu=mu.to(truth.mu.dtype))
    return build_linearization(state, cams, eps_depth=Cfg.eps_depth)


# ===========================================================================
def test_R1_composed_jvp():
    """R1: H_G J p against central finite differences of the rendered RGB-D."""
    truth, cams, mu0, st = setup()
    frame = 5
    g = torch.Generator().manual_seed(3)
    p = torch.randn(st.dim, generator=g, dtype=torch.float64)
    p[1] = 0.0                                   # qdot has no direct effect

    lin = render(truth, cams, st, frame, mu0)
    tangents = joint_apply_H(lin, st, frame, p, mu0)

    def perturbed(h):
        s2 = copy.deepcopy(st)
        d = p * h
        s2.parts[0].q[frame] = s2.parts[0].q[frame] + d[0]
        z, _G = retract_joint_geometry(s2.parts[0].zeta, d[2:6])
        s2.parts[0].zeta = z
        return render(truth, cams, s2, frame, mu0)

    # The cached renderer JVP is the derivative at FIXED visibility: Phase 1's
    # contributor traversal is frozen at the linearization point. A finite
    # difference across a real step also moves the contributor set, so the
    # handful of silhouette pixels where a Gaussian enters or leaves a tile
    # disagree by construction. That is a property of the Phase-1 operator,
    # which sections 0 and 38 forbid modifying, so the comparison is made on
    # the bulk and the outlier fraction is reported rather than hidden by a
    # loose tolerance on the maximum.
    h = 1e-3
    lp, lm = perturbed(h), perturbed(-h)
    worst_rgb = worst_depth = 0.0
    frac_rgb = frac_depth = 0.0
    corr = []
    excluded = 0
    thr = Cfg.min_alpha_for_depth
    for ci in range(len(cams)):
        # Pixels whose DEPTH VALIDITY flips between the two perturbed renders
        # have no derivative at all: the depth there is gated on
        # `alpha > min_alpha_for_depth`, so it jumps rather than varies. These
        # are excluded by their mechanism, not by a magnitude cutoff -- one
        # such pixel in camera 1 showed a finite difference of 1.1e+03 against
        # an analytic maximum of 0.94, which is a step, not a slope.
        valid = ((lp.views[ci].alpha > thr) == (lm.views[ci].alpha > thr))
        excluded += int((~valid).sum())
        fd_rgb = (lp.views[ci].rgb.double() - lm.views[ci].rgb.double()) / (2 * h)
        fd_d = (lp.views[ci].depth.double() - lm.views[ci].depth.double()) / (2 * h)
        an_rgb, an_d = tangents[ci][0].double(), tangents[ci][1].double()
        for fd, an, tag in ((fd_rgb, an_rgb, "rgb"), (fd_d, an_d, "depth")):
            m = valid.expand_as(fd) if fd.shape != valid.shape else valid
            fdv, anv = fd[m], an[m]
            sc = max(float(anv.abs().max()), 1e-12)
            rel = ((fdv - anv).abs() / sc).reshape(-1)
            q99 = float(rel.quantile(0.99))
            frac = float((rel > 1e-2).double().mean())
            if tag == "rgb":
                worst_rgb, frac_rgb = max(worst_rgb, q99), max(frac_rgb, frac)
            else:
                worst_depth, frac_depth = max(worst_depth, q99), max(frac_depth, frac)
            corr.append(float(torch.corrcoef(torch.stack(
                [fdv.reshape(-1), anv.reshape(-1)]))[0, 1]))
    print(f"       excluded {excluded} pixel(s) whose depth validity flipped "
          f"(no derivative exists there)")
    print(f"       99th-percentile relative error: RGB {worst_rgb:.3e}, "
          f"depth {worst_depth:.3e}")
    print(f"       pixels disagreeing by >1%: RGB {100 * frac_rgb:.2f}%, "
          f"depth {100 * frac_depth:.2f}%   (contributor-set changes)")
    print(f"       analytic/finite-difference correlation: "
          f"{min(corr):.6f} .. {max(corr):.6f}")
    check("R1 the composed RGB JVP matches finite differences on the bulk",
          worst_rgb < 5e-2, f"99th pct {worst_rgb:.2e}")
    check("R3 and the depth JVP does too, separately",
          worst_depth < 5e-2, f"99th pct {worst_depth:.2e}")
    check("R1 the residual disagreement is confined to silhouette pixels",
          frac_rgb < 0.05 and frac_depth < 0.05,
          f"{100 * max(frac_rgb, frac_depth):.2f}% of pixels")
    # A correlation is the wrong summary here: one silhouette pixel with a
    # finite difference of 1e+03 against an analytic 0.9 dominates the
    # covariance and drives it to zero however well the other 6910 pixels
    # agree. The property that actually distinguishes "discretization
    # artifact" from "wrong Jacobian" is whether the disagreement SHRINKS with
    # the step: a systematic error would not.
    fracs = []
    for hh in (4e-3, 2e-3, 1e-3, 5e-4):
        a, b = perturbed(hh), perturbed(-hh)
        bad = tot = 0
        for ci in range(len(cams)):
            fd = (a.views[ci].rgb.double() - b.views[ci].rgb.double()) / (2 * hh)
            an = tangents[ci][0].double()
            rel = (fd - an).abs() / max(float(an.abs().max()), 1e-12)
            bad += int((rel > 1e-2).sum())
            tot += rel.numel()
        fracs.append(bad / tot)
    print("       disagreeing RGB pixels vs step size: "
          + ", ".join(f"h={hh:.0e} -> {100 * fr:.2f}%"
                      for hh, fr in zip((4e-3, 2e-3, 1e-3, 5e-4), fracs)))
    check("R1 the disagreement shrinks with the step, so it is a "
          "discretization artifact and not a wrong Jacobian",
          fracs[-1] < fracs[0],
          f"{100 * fracs[0]:.2f}% -> {100 * fracs[-1]:.2f}%")


def test_R2_composed_adjoint():
    """R2 / section 24: <H_G J p, z> == <p, J^T H_G^T z>, exactly."""
    truth, cams, mu0, st = setup(seed=1)
    frame = 4
    lin = render(truth, cams, st, frame, mu0)
    g = torch.Generator(device=DEV).manual_seed(5)
    worst = 0.0
    for _ in range(6):
        p = torch.randn(st.dim, dtype=torch.float64)
        tang = joint_apply_H(lin, st, frame, p, mu0)
        z = [(torch.randn(t[0].shape, generator=g, device=DEV),
              torch.randn(t[1].shape, generator=g, device=DEV))
             for t in tang]
        lhs = sum(float((a.double() * b.double()).sum())
                  for t, zz in zip(tang, z) for a, b in zip(t, zz))
        adj = joint_apply_HT(lin, st, frame, z, mu0)
        rhs = float((p.to(adj.device) * adj).sum())
        worst = max(worst, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30))
    # The tolerance is set by the RENDERER, not by the joint layer: Phase 1's
    # apply_H / apply_HT are float32 and these inner products accumulate over
    # ~7000 pixels x 2 cameras x 4 channels, so a few times float32 epsilon is
    # the floor. J9 measures the kinematic adjoint on its own in float64 and
    # gets 4.6e-15, which locates the entire loss in the float32 operators the
    # spec forbids changing.
    eps32 = float(torch.finfo(torch.float32).eps)
    print(f"       worst relative adjoint mismatch over 6 pairs: {worst:.3e}"
          f"   (float32 eps {eps32:.2e}; J9 gives 4.6e-15 in float64)")
    check("R2 the composed operator and its adjoint agree to renderer "
          "precision", worst < 100 * eps32, f"{worst:.2e}")


def test_R4_static_derivative_is_zero():
    """R4: Gaussians outside the moving support have exactly zero derivative."""
    truth, cams, mu0, st = setup(n=260, moving=120, seed=2)
    frame = 6
    g = torch.Generator().manual_seed(7)
    p = torch.randn(st.dim, generator=g, dtype=torch.float64)
    pose = full_jvp(st, frame, p, mu0)
    print(f"       moving block max |J p| {float(pose[:120].abs().max()):.3e}, "
          f"static block {float(pose[120:].abs().max()):.3e}")
    check("R4 the static Gaussians' pose tangent is exactly zero",
          float(pose[120:].abs().max()) == 0.0)
    # and the adjoint ignores whatever lands on them
    a = torch.zeros((mu0.shape[0], 6), dtype=torch.float64)
    a[120:] = 1.0
    check("R4 and adjoints on static Gaussians contribute nothing",
          float(full_vjp(st, frame, a, mu0).abs().max()) == 0.0)


def test_R5_shared_part_geometry():
    """R5: every member of a part shares one screw, so K=2 blocks stay separate."""
    truth = scene_of(300, 11)
    cams = [camera()]
    mu0 = truth.mu.double()
    pa = initialise_from_twist(torch.arange(0, 120), mu0,
                               torch.tensor([0.3, 0.0, 0.0, 0.0, 0.0, 0.0]),
                               torch.linspace(0, 0.8, 8), None)
    pb = initialise_from_twist(torch.arange(120, 240), mu0,
                               torch.tensor([0.0, 0.2, 0.0, 0.1, 0.0, 0.3]),
                               torch.linspace(0, 0.6, 8), None)
    st = JointState([pa, pb], torch.arange(8, dtype=torch.float64))
    g = torch.Generator().manual_seed(13)
    v = torch.zeros(st.dim, dtype=torch.float64)
    v[st.slice_of(0)] = torch.randn(6, generator=g, dtype=torch.float64)
    pose = full_jvp(st, 5, v, mu0)
    print(f"       part A perturbed: |J v| on A {float(pose[:120].abs().max()):.3e}, "
          f"on B {float(pose[120:240].abs().max()):.3e}, "
          f"static {float(pose[240:].abs().max()):.3e}")
    check("R5 perturbing one part leaves the other exactly unmoved",
          float(pose[120:].abs().max()) == 0.0)
    check("R5 and the perturbed part does move",
          float(pose[:120].abs().max()) > 1e-6)
    lin = build_linearization(truth, cams, eps_depth=Cfg.eps_depth)
    z = [(torch.randn(t.rgb.shape, device=DEV),
          torch.randn(t.depth.shape, device=DEV)) for t in lin.views]
    adj = joint_apply_HT(lin, st, 5, z, mu0)
    check("R5 the adjoint returns one block per part",
          adj.numel() == 12)
    check("R5 and each part's block is nonzero where that part is visible",
          float(adj[:6].abs().max()) > 0.0 and float(adj[6:].abs().max()) > 0.0)


def main():
    if not torch.cuda.is_available():
        print("CUDA is required for the composed renderer tests")
        return
    tests = [test_R1_composed_jvp, test_R2_composed_adjoint,
             test_R4_static_derivative_is_zero, test_R5_shared_part_geometry]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
