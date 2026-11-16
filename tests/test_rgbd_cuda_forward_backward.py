"""Gate 1: native CUDA RGB + alpha + metric depth, forward and backward (spec section 3.3).

The reference is recovery/raster_oracle.py, a float64 torch re-implementation of the same kernel (same
projection, EWA covariance, tiles, depth order, alpha clipping, early stop). Its autograd gradients are the
exact derivatives of that function; they are themselves checked by central finite differences. The CUDA
gradients are compared to them. Scene: two cameras, overlapping front (semi-transparent) and back layers,
partly transparent splats, non-zero background, Gaussians straddling 16x16 tile borders.

    python tests/test_rgbd_cuda_forward_backward.py      (or pytest)
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recovery  # noqa: E402,F401  (pins this checkout's rasterizer)
from diff_gaussian_rasterization import (GaussianRasterizationSettings, GaussianRasterizer,  # noqa: E402
                                         _C, metric_depth_from)
from recovery import raster_oracle as O  # noqa: E402

DEV = "cuda"
W, H = 40, 36
FOV = math.radians(50.0)
BG = torch.tensor([0.2, 0.4, 0.7])


def look_at(eye, target, up=(0.0, -1.0, 0.0)):
    """OpenCV camera (x right, y down, z forward) -> row-vector view matrix V (p_view = [p,1] @ V)."""
    eye, target, up = (torch.tensor(v, dtype=torch.float64) for v in (eye, target, up))
    z = target - eye
    z = z / z.norm()
    x = torch.linalg.cross(up, z)
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    Rcw = torch.stack([x, y, z])                     # rows: camera axes in world
    w2c = torch.eye(4, dtype=torch.float64)
    w2c[:3, :3] = Rcw
    w2c[:3, 3] = -Rcw @ eye
    return w2c.T


def projection(znear=0.01, zfar=100.0):
    t = math.tan(FOV / 2)
    P = torch.zeros(4, 4, dtype=torch.float64)
    P[0, 0] = 1 / t
    P[1, 1] = 1 / t
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P.T


def cameras():
    out = []
    for eye in ((0.0, 0.0, 0.0), (0.55, 0.12, 0.15)):
        V = look_at(eye, (0.0, 0.0, 2.4))
        Pm = V @ projection()
        out.append(dict(V=V, P=Pm, campos=torch.tensor(eye, dtype=torch.float64)))
    return out


def scene(seed=0, n_front=5, n_back=5, sh_degree=3):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    front = torch.cat([(r(n_front, 2) - 0.5) * 0.9, 2.0 + 0.1 * r(n_front, 1)], 1)
    back = torch.cat([(r(n_back, 2) - 0.5) * 1.1, 2.7 + 0.1 * r(n_back, 1)], 1)
    xyz = torch.cat([front, back])
    N = xyz.shape[0]
    quat = torch.nn.functional.normalize(torch.randn(N, 4, generator=g, dtype=torch.float64), dim=1)
    log_scale = torch.log(0.06 + 0.12 * r(N, 3))
    # opacity well below 0.99 so the CUDA alpha clamp (whose gradient CUDA does not zero) never binds
    op = torch.cat([0.35 + 0.2 * r(n_front), 0.6 + 0.2 * r(n_back)])
    op_logit = torch.log(op / (1 - op))[:, None]
    dc = (r(N, 1, 3) - 0.5) * 2.0 + 0.4
    rest = (r(N, (sh_degree + 1) ** 2 - 1, 3) - 0.5) * 0.3
    return dict(xyz=xyz, quat=quat, log_scale=log_scale, op_logit=op_logit, dc=dc, rest=rest, deg=sh_degree)


def activated(p):
    return dict(means=p["xyz"], scales=torch.exp(p["log_scale"]),
                rotations=torch.nn.functional.normalize(p["quat"], dim=1),
                opacities=torch.sigmoid(p["op_logit"]), sh=torch.cat([p["dc"], p["rest"]], 1))


def cuda_render(a, cam, colors=None, antialiasing=False, sh_degree=3):
    s = GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=math.tan(FOV / 2), tanfovy=math.tan(FOV / 2),
        bg=BG.float().to(DEV), scale_modifier=1.0, viewmatrix=cam["V"].float().to(DEV),
        projmatrix=cam["P"].float().to(DEV), sh_degree=sh_degree, campos=cam["campos"].float().to(DEV),
        prefiltered=False, debug=False, antialiasing=antialiasing)
    R = GaussianRasterizer(s)
    f = lambda x: x.float().to(DEV)
    means2D = torch.zeros(a["means"].shape[0], 3, device=DEV, requires_grad=True)
    kw = dict(colors_precomp=f(colors)) if colors is not None else dict(shs=f(a["sh"]))
    color, radii, invd, alpha, zd = R.forward_rgbd(f(a["means"]), means2D, f(a["opacities"]),
                                                   scales=f(a["scales"]), rotations=f(a["rotations"]), **kw)
    return dict(color=color, radii=radii, invdepth=invd, alpha=alpha, zdepth=zd), R, means2D


def oracle_render(a, cam, colors=None, antialiasing=False, sh_degree=3):
    return O.render(a["means"], a["scales"], a["rotations"], a["opacities"], cam["V"], cam["P"], cam["campos"],
                    W, H, math.tan(FOV / 2), math.tan(FOV / 2), BG.double(),
                    colors=colors, sh=None if colors is not None else a["sh"], sh_degree=sh_degree,
                    antialiasing=antialiasing)


def loss_weights(seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(c=torch.randn(3, H, W, generator=g, dtype=torch.float64),
                i=torch.randn(1, H, W, generator=g, dtype=torch.float64),
                a=torch.randn(1, H, W, generator=g, dtype=torch.float64),
                z=torch.randn(1, H, W, generator=g, dtype=torch.float64),
                d0=2.0 + torch.rand(1, H, W, generator=g, dtype=torch.float64))


def total_loss(out, w, terms=("c", "i", "a", "z", "d")):
    dt, dev = torch.float64, out["color"].device
    t = lambda x: x.to(dev, dt)
    L = 0.0
    if "c" in terms:
        L = L + (t(w["c"]) * out["color"].to(dt)).sum()
    if "i" in terms:
        L = L + (t(w["i"]) * out["invdepth"].to(dt)).sum()
    if "a" in terms:
        L = L + (t(w["a"]) * out["alpha"].to(dt)).sum()
    if "z" in terms:
        L = L + (t(w["z"]) * out["zdepth"].to(dt)).sum()
    if "d" in terms:  # the torch-side quotient D = Z / (A + eps), only where there is coverage
        A = out["alpha"].to(dt)
        D = metric_depth_from(out["zdepth"].to(dt), A, 1e-6)
        m = (A.detach() > 0.05).to(dt)
        L = L + 0.5 * (m * (D - t(w["d0"])) ** 2).sum()
    return L


def rel(a, b):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).norm() / max(float(b.norm()), 1e-12))


# ----------------------------------------------------------------------------------------- forward
def test_forward_matches_oracle():
    p = scene()
    a = activated(p)
    for ci, cam in enumerate(cameras()):
        for aa in (False, True):
            cu, _, _ = cuda_render(a, cam, antialiasing=aa)
            ref = oracle_render(a, cam, antialiasing=aa)
            assert torch.equal(cu["radii"].cpu(), ref["radii"].cpu()), f"cam {ci} aa {aa}: radii differ"
            for k in ("color", "alpha", "zdepth", "invdepth"):
                err = float((cu[k].double().cpu() - ref[k].detach().cpu()).abs().max())
                assert err < 2e-5 * max(1.0, float(ref[k].abs().max())), f"cam {ci} aa {aa} {k}: {err:.3g}"
            cov = ref["alpha"] > 0.05
            assert cov.float().mean() > 0.2, "the test scene must cover a good part of the image"
            assert float(ref["alpha"].max()) < 1.0 and float(ref["alpha"].min()) >= 0.0


def test_alpha_is_one_minus_final_T_and_zero_without_contributors():
    """C = sum w f + (1 - A) bg exactly, so A = 1 - T_final; an empty scene gives A = Z = 0."""
    p = scene()
    a = activated(p)
    cam = cameras()[0]
    cols = torch.zeros_like(a["means"])                   # black splats: color = (1 - A) bg
    cu, _, _ = cuda_render(a, cam, colors=cols)
    pred = (1 - cu["alpha"]) * BG.to(DEV)[:, None, None]
    assert float((cu["color"] - pred).abs().max()) < 1e-6
    far = dict(a, means=a["means"] + torch.tensor([0.0, 0.0, -10.0], dtype=torch.float64))   # behind camera
    cu0, _, _ = cuda_render(far, cam)
    assert float(cu0["alpha"].abs().max()) == 0.0 and float(cu0["zdepth"].abs().max()) == 0.0
    assert torch.allclose(cu0["color"], BG.to(DEV)[:, None, None].expand(3, H, W))


def test_legacy_outputs_unchanged_in_this_build():
    """Same kernel: the legacy entry point and the RGB-D entry point agree bit for bit on RGB / invdepth / radii."""
    p = scene()
    a = activated(p)
    for cam in cameras():
        cu, R, m2 = cuda_render(a, cam)
        f = lambda x: x.float().to(DEV)
        color, radii, invd = R(f(a["means"]), m2, f(a["opacities"]), shs=f(a["sh"]),
                               scales=f(a["scales"]), rotations=f(a["rotations"]))
        assert torch.equal(color, cu["color"]) and torch.equal(radii, cu["radii"]) and torch.equal(invd, cu["invdepth"])


def test_metric_depth_is_weighted_camera_z_not_inverse_or_first_return():
    """Two coincident-in-image layers far apart in depth: Z/A, 1/(invdepth/A) and the first return differ."""
    N = 2
    a = dict(means=torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]], dtype=torch.float64),
             scales=torch.full((N, 3), 0.25, dtype=torch.float64),
             rotations=torch.tensor([[1.0, 0, 0, 0]] * N, dtype=torch.float64),
             opacities=torch.tensor([[0.5], [0.9]], dtype=torch.float64),
             sh=torch.zeros(N, 16, 3, dtype=torch.float64))
    cam = cameras()[0]
    cu, _, _ = cuda_render(a, cam)
    ref = oracle_render(a, cam)
    y, x = H // 2, W // 2
    A, Z, I = (float(cu[k][0, y, x]) for k in ("alpha", "zdepth", "invdepth"))
    D = float(metric_depth_from(cu["zdepth"], cu["alpha"])[0, y, x])
    D_ref = float(ref["zdepth"][0, y, x] / ref["alpha"][0, y, x])
    assert abs(D - D_ref) < 1e-4, (D, D_ref)
    harmonic = A / I
    assert D > 2.4 and abs(D - harmonic) > 0.2 and abs(D - 2.0) > 0.4, (D, harmonic)


# ---------------------------------------------------------------------------------------- backward
def _grads_activated(render_fn, a, cam, w, colors=None, terms=("c", "i", "a", "z", "d"), aa=False):
    leaves = {k: v.detach().clone().double().requires_grad_(True) for k, v in a.items()}
    col = None if colors is None else colors.detach().clone().double().requires_grad_(True)
    out = render_fn(leaves, cam, colors=col, antialiasing=aa)
    out = out[0] if isinstance(out, tuple) else out
    L = total_loss(out, w, terms)
    L.backward()
    g = {k: v.grad for k, v in leaves.items() if v.grad is not None}
    if col is not None:
        g["colors"] = col.grad
    return float(L), g


def test_backward_matches_oracle_gradients():
    """CUDA dL/d{means, scales, rotations, opacities, SH} and dL/dcolors_precomp == exact oracle gradients
    (antialiasing off, the repository default and the only mode Phase 4/5 accept)."""
    p = scene()
    a = activated(p)
    w = loss_weights()
    for ci, cam in enumerate(cameras()):
        Lc, gc = _grads_activated(cuda_render, a, cam, w)
        Lr, gr = _grads_activated(oracle_render, a, cam, w)
        assert abs(Lc - Lr) < 1e-4 * max(1.0, abs(Lr)), (Lc, Lr)
        for k in ("means", "scales", "rotations", "opacities", "sh"):
            e = rel(gc[k], gr[k])
            assert e < 1e-4, f"cam {ci} d/d{k}: relative error {e:.2e}"
        cols = torch.rand(a["means"].shape[0], 3, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        _, gc = _grads_activated(cuda_render, a, cam, w, colors=cols)
        _, gr = _grads_activated(oracle_render, a, cam, w, colors=cols)
        for k in ("colors", "means", "opacities"):
            e = rel(gc[k], gr[k])
            assert e < 1e-4, f"cam {ci} colors_precomp d/d{k}: relative error {e:.2e}"


def test_antialiasing_gradient_gap_is_preexisting():
    """Known upstream defect, NOT introduced here: with antialiasing on, computeCov2DCUDA differentiates
    det_cov / det_cov_plus_h with c_xx, c_yy that already include the 0.3 low-pass, so the covariance-path
    gradients (scales, rotations, a little of the means) are off by ~1%. The legacy RGB-only backward shows
    the same gap; opacity / SH gradients stay exact. Phase 4/5 therefore refuse antialiasing=True."""
    p = scene()
    a = activated(p)
    w = loss_weights()
    cam = cameras()[0]
    _, gl = _grads_activated(cuda_render, a, cam, w, terms=("c",), aa=True)
    _, gr = _grads_activated(oracle_render, a, cam, w, terms=("c",), aa=True)
    assert rel(gl["scales"], gr["scales"]) > 1e-3          # the gap exists on the legacy terms alone
    assert rel(gl["opacities"], gr["opacities"]) < 1e-4 and rel(gl["sh"], gr["sh"]) < 1e-4


def test_each_loss_term_reaches_only_its_path():
    """dL/dZ and dL/dA alone give zero colour/SH gradient; RGB-only gradients equal the legacy backward."""
    p = scene()
    a = activated(p)
    w = loss_weights()
    cam = cameras()[1]
    _, g_geo = _grads_activated(cuda_render, a, cam, w, terms=("a", "z", "d"))
    assert float(g_geo["sh"].abs().max()) == 0.0
    assert float(g_geo["means"].abs().max()) > 0 and float(g_geo["opacities"].abs().max()) > 0
    _, g_rgb = _grads_activated(cuda_render, a, cam, w, terms=("c",))
    # legacy backward, RGB only
    leaves = {k: v.detach().clone().float().to(DEV).requires_grad_(True) for k, v in a.items()}
    _, R, m2 = cuda_render(a, cam)
    color, _, _ = R(leaves["means"], m2, leaves["opacities"], shs=leaves["sh"], scales=leaves["scales"],
                    rotations=leaves["rotations"])
    (w["c"].to(DEV).float() * color).sum().backward()
    for k in ("means", "scales", "rotations", "opacities", "sh"):
        assert rel(g_rgb[k], leaves[k].grad) < 1e-6, k


def test_oracle_gradients_match_finite_differences():
    """The reference itself: autograd vs central differences w.r.t. the raw GaussianModel parameters
    (xyz, quaternion, log-scale, opacity logit, DC, SH rest), float64, a sample of coordinates."""
    p = scene(n_front=3, n_back=3)
    w = loss_weights()
    cam = cameras()[1]

    def L_of(params):
        return total_loss(oracle_render(activated(params), cam), w)

    leaves = {k: (v.clone().requires_grad_(True) if torch.is_tensor(v) else v) for k, v in p.items()}
    L_of(leaves).backward()
    g = torch.Generator().manual_seed(5)
    for k in ("xyz", "quat", "log_scale", "op_logit", "dc", "rest"):
        flat = p[k].reshape(-1)
        for idx in torch.randperm(flat.numel(), generator=g)[:6].tolist():
            eps = 1e-6
            pp = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in p.items()}
            pm = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in p.items()}
            pp[k].reshape(-1)[idx] += eps
            pm[k].reshape(-1)[idx] -= eps
            fd = (float(L_of(pp)) - float(L_of(pm))) / (2 * eps)
            an = float(leaves[k].grad.reshape(-1)[idx])
            assert abs(fd - an) <= 1e-5 * max(1.0, abs(fd)) + 1e-6, f"{k}[{idx}]: fd {fd:.6g} autograd {an:.6g}"


def test_raw_parameter_gradients_cuda_vs_oracle():
    """Through the GaussianModel activations: CUDA gradients w.r.t. raw xyz / quaternion / log-scale /
    opacity logit / DC / SH equal the (finite-difference-checked) oracle gradients."""
    p = scene()
    w = loss_weights()
    for cam in cameras():
        gs = []
        for fn in (cuda_render, oracle_render):
            leaves = {k: (v.clone().requires_grad_(True) if torch.is_tensor(v) else v) for k, v in p.items()}
            out = fn(activated(leaves), cam)
            out = out[0] if isinstance(out, tuple) else out
            total_loss(out, w).backward()
            gs.append({k: leaves[k].grad for k in ("xyz", "quat", "log_scale", "op_logit", "dc", "rest")})
        for k in gs[0]:
            e = rel(gs[0][k], gs[1][k])
            assert e < 1e-4, f"d/d{k}: relative error {e:.2e}"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print(f"{fails} failed")
    sys.exit(1 if fails else 0)
