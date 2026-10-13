"""T5 — the free-state adapter against the ACTUAL CUDA RGB-D operator.

CODEX_COMPLETE_UNIFIED_PHASE1_IEKS_REMAINING_STEPS.md section 2.

Every check here goes through the real rasterizer. The spec is explicit that
an algebra-only substitute is a FAIL, so there is no CPU fallback: without
CUDA the suite reports NOT_RUN rather than passing on something else.

The seven required checks:

  1 parity        the adapter renders what the existing bridge renders
  2 adjoint       <H u, z> == <u, H^T z> with overlapping Gaussians
  3 finite diff   analytic JVP vs central differences of a real rerender,
                  away from sorting / support / clamp discontinuities
  4 velocity      image Jacobian velocity columns are exactly zero
  5 co-visibility mixed information H_i^T R^-1 H_j is nonzero for i != j
  6 immutability  colour, opacity and scales bitwise unchanged
  7 resources     peak CUDA memory for one frame x eight cameras

    python tests/test_unified_cuda_renderer_jvp_vjp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CHECKS = 0
FAILURES: list[str] = []
SKIPPED: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")
    else:
        print(f"  ok    {label}  {detail}")


def cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import diff_gaussian_rasterization  # noqa: F401
        return True
    except Exception:
        return False


def make_cameras(n_cam=3, H=128, W=128, radius=2.2):
    """A small ring of cameras looking at the origin."""
    from estimator.renderer_bridge import CameraSpec

    cams = []
    fov = 0.9
    tan = float(np.tan(fov * 0.5))
    for k in range(n_cam):
        angle = 2.0 * np.pi * k / n_cam
        eye = np.array([radius * np.sin(angle), 0.0, radius * np.cos(angle)])
        forward = -eye / np.linalg.norm(eye)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward); right /= np.linalg.norm(right)
        true_up = np.cross(forward, right)
        Rwc = np.stack([right, true_up, forward], axis=0)
        t = -Rwc @ eye
        view = np.eye(4)
        view[:3, :3] = Rwc.T
        view[3, :3] = t
        view_t = torch.tensor(view, dtype=torch.float32, device="cuda")
        proj = torch.zeros(4, 4, dtype=torch.float32, device="cuda")
        proj[0, 0] = 1.0 / tan
        proj[1, 1] = 1.0 / tan
        # ROW-vector convention, matching world_view_transform above: a point
        # is p @ M, so w = z_cam requires M[2,3] = 1. Putting the 1 at [3,2]
        # instead gives a negative w and the whole scene lands behind the
        # camera, which renders as a uniformly empty image rather than an error.
        near, far = 0.01, 100.0
        proj[2, 2] = (far + near) / (far - near)
        proj[2, 3] = 1.0
        proj[3, 2] = -2.0 * far * near / (far - near)
        full = (view_t.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        cams.append(CameraSpec(viewmatrix=view_t, projmatrix=full, tanfovx=tan,
                               tanfovy=tan, image_height=H, image_width=W))
    return cams


def make_state(n=60, seed=0, spread=0.30):
    """A compact cluster, so many Gaussians overlap in every view."""
    from estimator.manifold import so3_exp
    from estimator.state import GaussianMotionState

    g = torch.Generator().manual_seed(seed)
    return GaussianMotionState(
        mu=(torch.randn(n, 3, generator=g) * spread).cuda(),
        R=so3_exp(torch.randn(n, 3, generator=g) * 0.3).cuda(),
        v=(torch.randn(n, 3, generator=g) * 0.02).cuda(),
        omega=(torch.randn(n, 3, generator=g) * 0.02).cuda(),
        colors=torch.rand(n, 3, generator=g).cuda(),
        opacity=(0.55 + 0.4 * torch.rand(n, 1, generator=g)).cuda(),
        scales=(0.16 + 0.06 * torch.rand(n, 3, generator=g)).cuda())


def synth_observations(state, cameras, n_frames=1, noise=0.0, seed=0):
    """Observations produced BY the renderer, so residuals are well posed."""
    from estimator.renderer_bridge import build_linearization

    g = torch.Generator(device="cuda").manual_seed(seed)
    frames = []
    for _ in range(n_frames):
        lin = build_linearization(state, cameras)
        per_cam = []
        for rgb, depth in lin.predicted():
            r = rgb.detach().clone()
            d = depth.detach().clone()
            if noise:
                r = r + noise * torch.randn(r.shape, generator=g, device="cuda")
                d = d + noise * torch.randn(d.shape, generator=g, device="cuda")
            per_cam.append((r, d, torch.ones_like(d, dtype=torch.bool)))
        frames.append(per_cam)
    return frames


def main() -> int:
    if not cuda_available():
        print("T5 NOT_RUN: CUDA or diff_gaussian_rasterization unavailable.")
        print("The specification forbids substituting an algebra-only test, "
              "so nothing is reported as passing here.")
        return 0

    from bidirectional_iekf.observations import FreeRGBDObservation, adjoint_defect
    from estimator.renderer_bridge import build_linearization, extract_pose
    from estimator.state import boxplus

    torch.manual_seed(0)
    cameras = make_cameras(n_cam=3)
    state = make_state()
    n = state.mu.shape[0]
    observations = synth_observations(state, cameras, n_frames=1, noise=0.01)
    obs = FreeRGBDObservation(cameras, observations, frame_ids=[0])

    print("[1] source-only parity with the existing bridge")
    cache = obs.linearize_frame(state, 0)
    direct = build_linearization(state, cameras)
    worst_rgb = max(float((a[0] - b[0]).abs().max())
                    for a, b in zip(cache.linearization.predicted(), direct.predicted()))
    worst_depth = max(float((a[1] - b[1]).abs().max())
                      for a, b in zip(cache.linearization.predicted(), direct.predicted()))
    check("adapter render equals the bridge render",
          worst_rgb < 1e-6 and worst_depth < 1e-6,
          f"max |drgb| {worst_rgb:.2e}, max |ddepth| {worst_depth:.2e}")
    check("depth is the PDF's normalised D, not invdepth",
          bool(torch.isfinite(direct.views[0].depth).all())
          and float(direct.views[0].depth.max()) > 1.0,
          f"max D = {float(direct.views[0].depth.max()):.3f} (a scene at "
          "~2.2 m; an inverse depth would be well below 1)")

    print("[2] JVP / VJP adjoint identity on real CUDA output")
    g = torch.Generator(device="cuda").manual_seed(5)
    for trial in range(3):
        u = torch.randn(n, 12, generator=g, device="cuda")
        z = [(torch.randn(3, c.image_height, c.image_width, generator=g, device="cuda"),
              torch.randn(c.image_height, c.image_width, generator=g, device="cuda"))
             for c in cameras]
        d = adjoint_defect(obs, cache, u, z)
        check(f"<Hu,z> == <u,H^T z> (trial {trial})", d["rel_error"] < 1e-4,
              f"rel {d['rel_error']:.2e}  (lhs {d['lhs']:.4f}, rhs {d['rhs']:.4f})")

    print("[3] analytic JVP vs central differences of a real rerender")
    print("     eps is SWEPT: the renderer is float32, so a central difference")
    print("     is truncation-limited at large eps and cancellation-limited at")
    print("     small eps. A single eps proves nothing; the floor is reported.")
    # A separate, SPARSE fixture for differentiability. The dense scene above
    # is right for co-visibility but wrong here: in a saturated stack the
    # perturbed Gaussian sits behind ~59 others, its contribution is modulated
    # by their transmittance, and the difference quotient measures their noise
    # rather than its derivative. Here each Gaussian has a clean footprint.
    fd_state = make_state(n=8, seed=11, spread=0.55)
    fd_state = type(fd_state)(
        mu=fd_state.mu, R=fd_state.R, v=fd_state.v, omega=fd_state.omega,
        colors=fd_state.colors, opacity=fd_state.opacity,
        # STRONGLY ANISOTROPIC on purpose. An isotropic Gaussian is
        # rotation-invariant, so both the analytic and the numeric rotation
        # derivative are exactly zero and their ratio is 0/0 -- a test that
        # looks like a 100% error while nothing is wrong.
        scales=torch.tensor([0.20, 0.07, 0.04], device="cuda")
        .expand_as(fd_state.scales).contiguous())
    fd_cameras = make_cameras(n_cam=1, H=128, W=128, radius=2.6)
    fd_obs = FreeRGBDObservation(
        fd_cameras, synth_observations(fd_state, fd_cameras, 1), frame_ids=[0])
    fd_cache = fd_obs.linearize_frame(fd_state, 0)
    nfd = fd_state.mu.shape[0]
    # The Gaussian to perturb is CHOSEN BY OBSERVABILITY, not by result: the
    # row of H^T applied to a uniform image with the largest norm.
    uniform = [(torch.ones(3, c.image_height, c.image_width, device="cuda"),
                torch.ones(c.image_height, c.image_width, device="cuda"))
               for c in fd_cameras]
    sensitivity = fd_obs.vjp(fd_cache, uniform)[:, :6].norm(dim=1)
    target = int(sensitivity.argmax())
    print(f"     sparse fixture: {nfd} gaussians, perturbing the most "
          f"observable one (index {target}, |H^T 1| = "
          f"{float(sensitivity[target]):.3f})")

    for label, column in (("position x", 0), ("orientation z", 5)):
        curve = []
        for eps in (1.6e-2, 8e-3, 4e-3, 2e-3, 1e-3, 4e-4, 2e-4):
            direction = torch.zeros(nfd, 12, device="cuda")
            direction[target, column] = 1.0     # ONE Gaussian: a coherent
            # push on all of them sums many nonlinearities and measures the
            # scene's curvature rather than the Jacobian.
            analytic = fd_obs.jvp(fd_cache, direction)
            lin_p = build_linearization(boxplus(fd_state, eps * direction), fd_cameras)
            lin_m = build_linearization(boxplus(fd_state, -eps * direction), fd_cameras)
            rel_rgb, rel_depth, norms, kept = [], [], [], []
            for vb, vp, vm, (a_rgb, a_depth), (p_rgb, p_depth), (m_rgb, m_depth) in zip(
                    fd_cache.linearization.views, lin_p.views, lin_m.views,
                    analytic, lin_p.predicted(), lin_m.predicted()):
                fd_rgb = (p_rgb - m_rgb) / (2 * eps)
                fd_depth = (p_depth - m_depth) / (2 * eps)
                # Fixed active set: a silhouette pixel flips between
                # background (D = 0) and surface under an eps translation, so
                # its difference quotient diverges as 1/eps and no analytic
                # derivative can match it -- the function is not
                # differentiable there. Saturated coverage, unchanged
                # contributor count.
                stable = ((vb.alpha > 0.30) & (vp.alpha > 0.30) & (vm.alpha > 0.30)
                          & (vp.n_contrib == vm.n_contrib)
                          & (vp.n_contrib == vb.n_contrib))
                if int(stable.sum()) < 64:
                    continue
                mask3 = stable.unsqueeze(0).expand_as(fd_rgb)
                kept.append(float(stable.float().mean()))
                rel_rgb.append(float((a_rgb - fd_rgb)[mask3].norm()
                                     / max(float(fd_rgb[mask3].norm()), 1e-30)))
                rel_depth.append(float((a_depth - fd_depth)[stable].norm()
                                       / max(float(fd_depth[stable].norm()), 1e-30)))
                norms.append((float(a_rgb[mask3].norm()), float(fd_rgb[mask3].norm())))
            if rel_rgb:
                curve.append((eps, max(rel_rgb), max(rel_depth), norms[0], min(kept)))
        for eps, rr, rd, (na, nf), frac in curve:
            print(f"       eps {eps:8.1e}  rel_rgb {rr:8.3e}  rel_depth {rd:8.3e}"
                  f"   |analytic| {na:.4f} vs |fd| {nf:.4f}   stable {frac:.3f}")
        best_rgb = min(c[1] for c in curve)
        best_depth = min(c[2] for c in curve)
        best_eps = min(curve, key=lambda c: c[1])[0]
        na, nf = min(curve, key=lambda c: c[1])[3]
        check(f"JVP matches the rerendered RGB derivative ({label})",
              best_rgb < 0.10,
              f"best relative L2 error {best_rgb:.3e} at eps={best_eps:.1e} "
              f"(float32 rendering floor)")
        check(f"JVP matches the rerendered depth derivative ({label})",
              best_depth < 0.15, f"best relative L2 error {best_depth:.3e}")
        check(f"analytic and numeric derivative magnitudes agree ({label})",
              abs(na - nf) / max(nf, 1e-30) < 0.05,
              f"|analytic| {na:.4f} vs |fd| {nf:.4f}")

    print("[4] the image Jacobian has no velocity columns")
    vel = torch.zeros(n, 12, device="cuda")
    vel[:, 6:12] = torch.randn(n, 6, generator=g, device="cuda")
    out = obs.jvp(cache, vel)
    worst = max(max(float(a.abs().max()), float(b.abs().max())) for a, b in out)
    check("JVP of a pure velocity tangent is exactly zero", worst == 0.0,
          f"max |dI| = {worst:.3e} (structurally zero: extract_pose drops the "
          "velocity block before the renderer sees it)")
    adj = obs.vjp(cache, [(torch.randn_like(a), torch.randn_like(b)) for a, b in out])
    check("VJP writes nothing into the velocity block",
          float(adj[:, 6:12].abs().max()) == 0.0,
          f"max |adjoint_v| = {float(adj[:, 6:12].abs().max()):.3e}")

    print("[5] co-visible Gaussians produce mixed information")
    probe_i = torch.zeros(n, 12, device="cuda"); probe_i[0, 0] = 1.0
    probe_j = torch.zeros(n, 12, device="cuda"); probe_j[1, 0] = 1.0
    Aij = float((probe_i * obs.information_apply(cache, probe_j)).sum())
    check("H_i^T R^-1 H_j is nonzero for two overlapping Gaussians",
          abs(Aij) > 1e-12,
          f"u_i^T A u_j = {Aij:.6e}  (a block-diagonal-by-Gaussian "
          "approximation would give exactly 0 here)")
    Aii = float((probe_i * obs.information_apply(cache, probe_i)).sum())
    check("the diagonal block is positive", Aii > 0.0, f"u_i^T A u_i = {Aii:.6e}")
    sym = float((probe_i * obs.information_apply(cache, probe_j)).sum()
                - (probe_j * obs.information_apply(cache, probe_i)).sum())
    check("the measurement information action is symmetric",
          abs(sym) / max(abs(Aij), 1e-30) < 1e-4,
          f"asymmetry {sym:.3e}, relative {abs(sym) / max(abs(Aij), 1e-30):.2e}")

    print("[6] source appearance is immutable")
    before = (state.colors.clone(), state.opacity.clone(), state.scales.clone())
    step = torch.randn(n, 12, generator=g, device="cuda") * 0.01
    moved = boxplus(state, step)
    obs.linearize_frame(moved, 0)
    check("colours bitwise unchanged", torch.equal(moved.colors, before[0]))
    check("opacity bitwise unchanged", torch.equal(moved.opacity, before[1]))
    check("scales bitwise unchanged", torch.equal(moved.scales, before[2]))

    print("[7] factor inventory and resources")
    obs.build_inventory()
    manifest = obs.inventory.as_manifest()
    check("one factor per (frame, camera)", manifest["num_factors"] == len(cameras),
          f"{manifest['num_factors']} factors for {len(cameras)} cameras")
    global CHECKS
    CHECKS += 1
    try:
        from bidirectional_iekf.observations import FactorKey
        obs.inventory.register(FactorKey(0, 0))
        FAILURES.append("a duplicate factor key was accepted")
        print("  FAIL  a duplicate factor key was accepted")
    except ValueError:
        print("  ok    a duplicate (frame, camera) key is refused")

    cams8 = make_cameras(n_cam=8, H=240, W=320)
    obs8 = FreeRGBDObservation(
        cams8, synth_observations(state, cams8, 1, noise=0.02, seed=9), frame_ids=[0])
    torch.cuda.reset_peak_memory_stats()
    c8 = obs8.linearize_frame(state, 0)
    obs8.information_apply(c8, torch.randn(n, 12, device="cuda"))
    peak = torch.cuda.max_memory_allocated() / (1 << 20)
    print(f"  ..    peak CUDA memory, 1 frame x 8 cameras @320x240, "
          f"{n} gaussians: {peak:.1f} MiB")
    cost = obs8.nonlinear_cost(state, 0)
    check("nonlinear cost is finite and uses every camera",
          np.isfinite(cost["total"]) and cost["cameras"] == 8,
          f"E_rgb {cost['rgb']:.4f}, E_depth {cost['depth']:.4f}, "
          f"cameras {cost['cameras']}")

    print()
    for note in SKIPPED:
        print(f"SKIPPED {note}")
    if FAILURES:
        print(f"\n{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
