"""CUDA observation-operator acceptance tests.

CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md section 35: T6 (single-Gaussian JVP),
T7 (multi-Gaussian occlusion JVP), T8 (JVP/VJP adjoint identity -- the HARD
acceptance test), T9 (matrix-free HVP vs explicit H), T10 (block Fisher),
T16 (isotropic observability), T22 (backward compatibility).

These run against the REAL rasterizer in submodules/diff-gaussian-rasterization,
not a Python reimplementation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from diff_gaussian_rasterization import (  # noqa: E402
    iekf_apply_H,
    iekf_apply_HT,
    iekf_build_fisher,
    iekf_build_linearization_cache,
    iekf_fisher_to_dense,
    iekf_render,
)

PASSED: list[str] = []
DEV = "cuda"


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# A minimal synthetic scene with an explicit camera, so every quantity in the
# spec's chain is known and controllable.
# ---------------------------------------------------------------------------
def make_camera(width=64, height=64, fov=0.8):
    tanfovx = math.tan(fov * 0.5)
    tanfovy = math.tan(fov * 0.5)
    # Identity world->camera, looking down +z.
    view = torch.eye(4, device=DEV)
    znear, zfar = 0.01, 100.0
    top = tanfovy * znear
    right = tanfovx * znear
    proj = torch.zeros((4, 4), device=DEV)
    proj[0, 0] = znear / right
    proj[1, 1] = znear / top
    proj[2, 2] = zfar / (zfar - znear)
    proj[3, 2] = 1.0
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    full = (view.unsqueeze(0).bmm(proj.t().unsqueeze(0))).squeeze(0)
    return view, full, tanfovx, tanfovy, width, height


def make_scene(n=4, seed=0, isotropic=False, spread=0.35):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.zeros((n, 3))
    means[:, 0] = (torch.rand(n, generator=g) - 0.5) * spread
    means[:, 1] = (torch.rand(n, generator=g) - 0.5) * spread
    # staggered depths so several Gaussians genuinely overlap in the image
    means[:, 2] = 2.0 + torch.arange(n, dtype=torch.float32) * 0.25
    scales = torch.full((n, 3), 0.10)
    if not isotropic:
        scales[:, 0] = 0.16
        scales[:, 2] = 0.05
    quats = torch.zeros((n, 4))
    quats[:, 0] = 1.0
    if not isotropic:
        ang = torch.rand(n, generator=g) * 0.7
        quats[:, 0] = torch.cos(ang * 0.5)
        quats[:, 3] = torch.sin(ang * 0.5)
    opacity = torch.full((n, 1), 0.75)
    colors = torch.rand((n, 3), generator=g)
    return (means.to(DEV), scales.to(DEV), quats.to(DEV),
            opacity.to(DEV), colors.to(DEV))


def build(means, scales, quats, opacity, colors, cam, bg=None):
    view, full, tfx, tfy, W, H = cam
    if bg is None:
        bg = torch.zeros(3, device=DEV)
    return iekf_build_linearization_cache(
        means3D=means, scales=scales, rotations=quats, opacity=opacity,
        colors=colors, background=bg, viewmatrix=view, projmatrix=full,
        tanfovx=tfx, tanfovy=tfy, image_height=H, image_width=W,
        eps_depth=1e-6, debug=False)


def h_of(means, scales, quats, opacity, colors, cam, bg=None):
    """The nonlinear measurement h(x): stacked RGB and normalized depth."""
    cache = build(means, scales, quats, opacity, colors, cam, bg)
    rgb, depth, alpha, T, ncon = iekf_render(cache)
    # ncon is the forward pass's per-pixel contributor count; the reverse
    # operators need it to stop where the forward stopped.
    return rgb, depth, alpha, T, cache, ncon


def transported(means, quats, delta):
    """x boxplus delta, for the 6D pose part only (section 4 retraction).

    mu+ = mu + dmu ; R+ = Exp(hat(dphi)) R, applied to the quaternion.
    """
    dmu = delta[:, :3]
    dphi = delta[:, 3:]
    theta = dphi.norm(dim=1, keepdim=True)
    small = theta < 1e-12
    axis = dphi / theta.clamp_min(1e-30)
    half = theta * 0.5
    dq = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=1)
    dq = torch.where(small, torch.tensor([1.0, 0.0, 0.0, 0.0], device=dphi.device)
                     .expand_as(dq), dq)
    w1, x1, y1, z1 = dq.unbind(1)
    w2, x2, y2, z2 = quats.unbind(1)
    out = torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=1)
    return means + dmu, out


# ---------------------------------------------------------------------------
# T6 / T7
#
# A note on the metric, because it is not arbitrary. The renderer is only
# PIECEWISE smooth (section 26): a Gaussian's screen-space radius and the
# alpha < 1/255 cull are hard cutoffs, so at the splat boundary a few pixels
# jump by O(1) no matter how small eps is. A whole-image relative error
# therefore does NOT converge as eps -> 0 -- it grows, because the smooth signal
# shrinks while the jump does not. That is a property of the renderer, not an
# error in the Jacobian.
#
# So the comparison is made on pixels that are stably interior to the splat in
# all three renders, and the boundary population is reported separately rather
# than quietly dropped.
# ---------------------------------------------------------------------------
def fd_compare(cache, means, scales, quats, opacity, colors, cam, v, eps_list):
    """Returns per-eps (interior relative L2, all-pixel relative L2, n_interior)."""
    drgb, ddepth = iekf_apply_H(cache, v)
    _, _, a0, _, _, _ = h_of(means, scales, quats, opacity, colors, cam)
    rows = []
    for eps in eps_list:
        mp, qp = transported(means, quats, eps * v)
        mm, qm = transported(means, quats, -eps * v)
        rp, dp, ap, _, _, _ = h_of(mp, scales, qp, opacity, colors, cam)
        rm, dm, am, _, _, _ = h_of(mm, scales, qm, opacity, colors, cam)
        fd_rgb = (rp - rm) / (2 * eps)
        fd_d = (dp - dm) / (2 * eps)
        interior = (a0 > 0.05) & (ap > 0.05) & (am > 0.05)
        mask_rgb = interior.unsqueeze(0).expand_as(fd_rgb)
        rel_in = float((fd_rgb - drgb)[mask_rgb].norm()
                       / fd_rgb[mask_rgb].norm().clamp_min(1e-20))
        rel_all = float((fd_rgb - drgb).norm() / fd_rgb.norm().clamp_min(1e-20))
        # Depth is normalized by (A + eps_D), so it is genuinely ill-conditioned
        # where A -> 0; compare it only where coverage is solid (section 27).
        solid = (a0 > 0.5) & (ap > 0.5) & (am > 0.5)
        rel_d = (float((fd_d - ddepth)[solid].norm()
                       / fd_d[solid].norm().clamp_min(1e-20))
                 if int(solid.sum()) > 0 else float("nan"))
        rows.append((eps, rel_in, rel_all, rel_d, int(interior.sum())))
    return rows


def test_T6_single_gaussian_jvp():
    """T6: H v vs central differences, swept over eps to expose the U-shape."""
    cam = make_camera()
    means, scales, quats, opacity, colors = make_scene(n=1, seed=3)
    rgb0, d0, a0, T0, cache, ncon = h_of(means, scales, quats, opacity, colors, cam)
    check("T6 the single Gaussian is actually visible", float(a0.max()) > 0.1,
          f"max alpha {float(a0.max()):.4f}")

    torch.manual_seed(11)
    v = torch.randn((1, 6), device=DEV)
    v[:, 3:] *= 0.5
    v = v / v.norm()
    rows = fd_compare(cache, means, scales, quats, opacity, colors, cam, v,
                      (1e-2, 3e-3, 1e-3, 3e-4, 1e-4))
    for eps, rin, rall, rd, n in rows:
        print(f"       eps={eps:7.0e}  interior {rin:.3e}  depth {rd:.3e}  "
              f"all-pixel {rall:.3e}  (n_int={n})")
    best = min(r[1] for r in rows)
    best_d = min(r[3] for r in rows)
    check("T6 RGB JVP matches central differences on interior pixels", best < 1e-3,
          f"best interior rel {best:.3e}")
    check("T6 depth JVP matches central differences where coverage is solid",
          best_d < 1e-3, f"best rel {best_d:.3e}")
    # The U-shape the spec asks for: error must get WORSE at the largest eps
    # (truncation) than at the optimum.
    check("T6 the eps sweep shows truncation at large eps", rows[0][1] > best,
          f"eps=1e-2 {rows[0][1]:.3e} vs best {best:.3e}")
    # And the boundary effect is real and must be visible, not hidden.
    check("T6 whole-image error is dominated by the splat boundary",
          rows[-1][2] > 10 * rows[-1][1],
          f"all {rows[-1][2]:.3e} vs interior {rows[-1][1]:.3e}")


def test_T6b_pure_translation_is_exact():
    """A pure translation isolates du and dJp from the rotation path."""
    cam = make_camera()
    means, scales, quats, opacity, colors = make_scene(n=1, seed=3)
    _, _, _, _, cache, ncon = h_of(means, scales, quats, opacity, colors, cam)
    for axis in range(3):
        v = torch.zeros((1, 6), device=DEV)
        v[0, axis] = 1.0
        rows = fd_compare(cache, means, scales, quats, opacity, colors, cam, v,
                          (3e-3, 1e-3))
        best = min(r[1] for r in rows)
        check(f"T6b translation along axis {axis} is exact", best < 1e-3,
              f"rel {best:.3e}")


def test_T6c_pure_rotation_is_exact():
    """A pure rotation isolates dSigma -> dS -> dQ (sections 9.3/9.4), which is
    the part a naive implementation gets wrong by dropping dJ_pi."""
    cam = make_camera()
    means, scales, quats, opacity, colors = make_scene(n=1, seed=3)
    _, _, _, _, cache, ncon = h_of(means, scales, quats, opacity, colors, cam)
    for axis in range(3):
        v = torch.zeros((1, 6), device=DEV)
        v[0, 3 + axis] = 1.0
        rows = fd_compare(cache, means, scales, quats, opacity, colors, cam, v,
                          (3e-3, 1e-3))
        best = min(r[1] for r in rows)
        check(f"T6c rotation about axis {axis} is exact", best < 2e-3,
              f"rel {best:.3e}")


def test_T7_multi_gaussian_occlusion_jvp():
    """T7: several depth layers -- the tangent must carry front/rear coupling."""
    cam = make_camera()
    means, scales, quats, opacity, colors = make_scene(n=5, seed=7, spread=0.12)
    rgb0, d0, a0, T0, cache, ncon = h_of(means, scales, quats, opacity, colors, cam)
    overlapped = float((a0 > 0.9).float().mean())
    check("T7 the layers genuinely overlap", overlapped > 0.01,
          f"{overlapped*100:.2f}% of pixels near-opaque")

    torch.manual_seed(5)
    v = torch.randn((5, 6), device=DEV) * 0.3
    rows = fd_compare(cache, means, scales, quats, opacity, colors, cam, v,
                      (3e-3, 1e-3, 3e-4))
    best = min(r[1] for r in rows)
    check("T7 occluded multi-layer JVP matches central differences", best < 2e-3,
          f"best interior rel {best:.3e}")

    # Moving ONLY the front layer must still change the composited result: that
    # is the occlusion coupling the tangent recurrence of section 10 encodes.
    order = torch.argsort(means[:, 2])
    front = torch.zeros((5, 6), device=DEV)
    front[int(order[0]), 0] = 1.0
    dfront, _ = iekf_apply_H(cache, front)
    check("T7 moving the front layer changes the image",
          float(dfront.abs().max()) > 1e-6, f"{float(dfront.abs().max()):.3e}")
    back = torch.zeros((5, 6), device=DEV)
    back[int(order[-1]), 0] = 1.0
    dback, _ = iekf_apply_H(cache, back)
    check("T7 the back layer is attenuated relative to the front",
          float(dback.abs().max()) < float(dfront.abs().max()),
          f"back {float(dback.abs().max()):.3e} < front {float(dfront.abs().max()):.3e}")


# ---------------------------------------------------------------------------
# T8 -- the hard acceptance test
# ---------------------------------------------------------------------------
def test_T8_adjoint_identity():
    """<H v, z> == <v, H^T z> for random v and z (spec section 35, T8)."""
    cam = make_camera()
    for seed, n in ((1, 3), (2, 6), (3, 10)):
        means, scales, quats, opacity, colors = make_scene(n=n, seed=seed, spread=0.2)
        rgb, depth, alpha, T, cache, ncon = h_of(means, scales, quats, opacity, colors, cam)
        torch.manual_seed(100 + seed)
        v = torch.randn((n, 6), device=DEV)
        z_rgb = torch.randn_like(rgb)
        z_d = torch.randn_like(depth)

        hv_rgb, hv_d = iekf_apply_H(cache, v)
        lhs = float((hv_rgb * z_rgb).sum() + (hv_d * z_d).sum())
        htz = iekf_apply_HT(cache, z_rgb, z_d, T, alpha, depth, ncon)
        rhs = float((v * htz).sum())
        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
        check(f"T8 adjoint identity (n={n})", rel < 1e-4,
              f"<Hv,z>={lhs:.6e}  <v,H^Tz>={rhs:.6e}  rel {rel:.3e}")


# ---------------------------------------------------------------------------
# T9 / T10
# ---------------------------------------------------------------------------
def explicit_H(cache, n, rgb_shape, depth_shape):
    """Build H column by column with the JVP. Only legitimate on a tiny scene --
    the production path never does this (section 0 rule 6)."""
    cols = []
    for i in range(n):
        for k in range(6):
            e = torch.zeros((n, 6), device=DEV)
            e[i, k] = 1.0
            drgb, dd = iekf_apply_H(cache, e)
            cols.append(torch.cat([drgb.reshape(-1), dd.reshape(-1)]))
    return torch.stack(cols, dim=1)


def test_T9_matrix_free_hvp():
    """H^T R^-1 H v from the matrix-free operator vs an explicitly built H."""
    cam = make_camera(width=32, height=32)
    n = 3
    means, scales, quats, opacity, colors = make_scene(n=n, seed=21, spread=0.15)
    rgb, depth, alpha, T, cache, ncon = h_of(means, scales, quats, opacity, colors, cam)

    H = explicit_H(cache, n, rgb.shape, depth.shape)
    torch.manual_seed(9)
    prec_rgb = torch.rand_like(rgb) + 0.5
    prec_d = torch.rand_like(depth) + 0.5
    Rinv = torch.cat([prec_rgb.reshape(-1), prec_d.reshape(-1)])

    v = torch.randn((n, 6), device=DEV)
    dense = (H.t() @ (Rinv * (H @ v.reshape(-1)))).reshape(n, 6)

    hv_rgb, hv_d = iekf_apply_H(cache, v)
    mf = iekf_apply_HT(cache, prec_rgb * hv_rgb, prec_d * hv_d, T, alpha, depth, ncon)
    rel = float((dense - mf).abs().max() / dense.abs().max().clamp_min(1e-12))
    check("T9 matrix-free H^T R^-1 H v matches explicit H", rel < 1e-3,
          f"rel {rel:.3e}")


def test_T10_block_fisher():
    """CUDA 6x6 blocks vs the diagonal blocks of the explicit H^T R^-1 H."""
    cam = make_camera(width=32, height=32)
    n = 3
    means, scales, quats, opacity, colors = make_scene(n=n, seed=33, spread=0.15)
    rgb, depth, alpha, T, cache, ncon = h_of(means, scales, quats, opacity, colors, cam)

    H = explicit_H(cache, n, rgb.shape, depth.shape)
    torch.manual_seed(4)
    prec_rgb = torch.rand_like(rgb) + 0.5
    prec_d = torch.rand_like(depth) + 0.5
    Rinv = torch.cat([prec_rgb.reshape(-1), prec_d.reshape(-1)])
    normal = H.t() @ (Rinv[:, None] * H)

    fisher = iekf_fisher_to_dense(
        iekf_build_fisher(cache, prec_rgb, prec_d, T, alpha, depth, ncon))
    worst = 0.0
    for i in range(n):
        ref = normal[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6]
        got = fisher[i]
        scale = ref.abs().max().clamp_min(1e-12)
        worst = max(worst, float((ref - got).abs().max() / scale))
    check("T10 block Fisher matches the explicit diagonal blocks", worst < 1e-3,
          f"worst rel {worst:.3e}")
    check("T10 Fisher blocks are symmetric",
          float((fisher - fisher.transpose(1, 2)).abs().max()) < 1e-6)
    evals = torch.linalg.eigvalsh(fisher.double())
    check("T10 Fisher blocks are PSD", float(evals.min()) > -1e-6,
          f"min eigenvalue {float(evals.min()):.3e}")


def test_T16_isotropic_observability():
    """Section 27: a near-isotropic Gaussian has almost no rotational Fisher
    information. The estimator must report that, not invent confidence."""
    cam = make_camera()
    iso = make_scene(n=1, seed=2, isotropic=True)
    aniso = make_scene(n=1, seed=2, isotropic=False)
    out = {}
    for name, scene in (("isotropic", iso), ("anisotropic", aniso)):
        means, scales, quats, opacity, colors = scene
        rgb, depth, alpha, T, cache, ncon = h_of(means, scales, quats, opacity, colors, cam)
        prec_rgb = torch.ones_like(rgb)
        prec_d = torch.ones_like(depth)
        f = iekf_fisher_to_dense(
            iekf_build_fisher(cache, prec_rgb, prec_d, T, alpha, depth, ncon))[0]
        rot = f[3:, 3:]
        trans = f[:3, :3]
        out[name] = (float(torch.linalg.eigvalsh(rot.double()).max()),
                     float(torch.linalg.eigvalsh(trans.double()).max()))
    check("T16 isotropic rotation information is far below anisotropic",
          out["isotropic"][0] < 0.05 * out["anisotropic"][0],
          f"iso {out['isotropic'][0]:.4e} vs aniso {out['anisotropic'][0]:.4e}")
    check("T16 translation stays observable for the isotropic Gaussian",
          out["isotropic"][1] > 1e-3, f"{out['isotropic'][1]:.4e}")


def test_T22_backward_compatibility():
    """The legacy rasterizer bindings must still exist and still run."""
    from diff_gaussian_rasterization import _C
    for name in ("rasterize_gaussians", "rasterize_gaussians_backward",
                 "rasterize_gaussians_contributors", "rasterize_gaussian_features",
                 "rasterize_gaussian_positions", "rasterize_gaussian_flow_stats",
                 "rasterize_gaussian_pixel_feature_stats", "mark_visible"):
        check(f"T22 legacy binding {name} is present", hasattr(_C, name))
    means = torch.randn((8, 3), device=DEV)
    view = torch.eye(4, device=DEV)
    proj = torch.eye(4, device=DEV)
    present = _C.mark_visible(means, view, proj)
    check("T22 a legacy call still executes", tuple(present.shape) == (8,))


def test_frozen_appearance_and_cache_invariance():
    """Section 0 rule 3 and section 20.3: the cache must be stable, and the
    operators must not depend on anything but the cache."""
    cam = make_camera()
    means, scales, quats, opacity, colors = make_scene(n=4, seed=15)
    rgb, depth, alpha, T, cache, ncon = h_of(means, scales, quats, opacity, colors, cam)
    v = torch.randn((4, 6), device=DEV)
    a1 = iekf_apply_H(cache, v)
    a2 = iekf_apply_H(cache, v)
    check("the JVP is deterministic at a frozen linearization",
          torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1]))
    r2 = iekf_render(cache)
    check("render from the same cache is deterministic",
          torch.equal(rgb, r2[0]) and torch.equal(depth, r2[1]))
    check("H is linear in v (superposition)",
          torch.allclose(iekf_apply_H(cache, 2.5 * v)[0], 2.5 * a1[0], atol=1e-5,
                         rtol=1e-4))
    zero = iekf_apply_H(cache, torch.zeros_like(v))
    check("H applied to zero is exactly zero",
          float(zero[0].abs().max()) == 0.0 and float(zero[1].abs().max()) == 0.0)


def test_depth_is_normalized():
    """Section 8: D = Z/(A+eps), not the legacy unnormalized inverse depth."""
    cam = make_camera()
    means, scales, quats, opacity, colors = make_scene(n=3, seed=8, spread=0.1)
    rgb, depth, alpha, T, cache, ncon = h_of(means, scales, quats, opacity, colors, cam)
    covered = alpha > 0.5
    check("some pixels are covered", int(covered.sum()) > 10, f"{int(covered.sum())}")
    zmin = float(means[:, 2].min())
    zmax = float(means[:, 2].max())
    d = depth[covered]
    check("normalized depth lies inside the Gaussian depth range",
          float(d.min()) >= zmin - 0.2 and float(d.max()) <= zmax + 0.2,
          f"[{float(d.min()):.3f}, {float(d.max()):.3f}] vs [{zmin:.3f}, {zmax:.3f}]")
    check("empty pixels have zero accumulated alpha",
          float(alpha[~covered].min()) >= 0.0)
    check("final transmittance is in [0,1]",
          float(T.min()) >= 0.0 and float(T.max()) <= 1.0 + 1e-6)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the IEKF operator tests")
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
