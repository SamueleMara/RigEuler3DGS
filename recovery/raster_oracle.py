"""Float64 torch re-implementation of the native CUDA rasterizer, for parity tests only.

It mirrors cuda_rasterizer/{forward.cu,auxiliary.h} step by step so that its forward pass equals the CUDA
kernel up to float32 rounding and its autograd gradients are the exact derivatives of that same function:

  * near culling z_view <= 0.2; 2D mean ndc2Pix(p, S) = ((p + 1) S - 1) / 2 with p_w = 1 / (w + 1e-7)
  * EWA covariance with the 1.3 tan(fov) clamp of t, + 0.3 low-pass, optional antialiasing opacity scale
  * radius ceil(3 sqrt(max eigenvalue)) with the 0.1 floor, 16x16 tile rectangle, tiles of zero area culled
  * per tile, Gaussians in increasing view depth (the [tile | depth] radix sort)
  * per pixel: skip power > 0, alpha = min(0.99, o G), skip alpha < 1/255, stop before T would fall
    below 1e-4; C = sum w f + T_final bg, invdepth = sum w / z, A = 1 - T_final, Z = sum w z
  * SH colour: the CUDA basis and constants, + 0.5, clamped at 0

Discrete decisions (culling, skips, the 0.99 clamp, early stop) are piecewise constant, as in CUDA. Tests
keep splats away from them; the CUDA backward does not zero the gradient of a clamped alpha.
"""
import math

import torch

BLOCK = 16
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
         1.445305721320277, -0.5900435899266435)


def quat_to_rot(q):
    """CUDA computeCov3D: q = (r, x, y, z), used unnormalized."""
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def sh_to_rgb(means, campos, sh, deg):
    """CUDA computeColorFromSH. sh: [N, M, 3]."""
    d = means - campos
    d = d / d.norm(dim=-1, keepdim=True)
    x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    res = SH_C0 * sh[:, 0]
    if deg > 0:
        res = res - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
        if deg > 1:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            res = (res + SH_C2[0] * xy * sh[:, 4] + SH_C2[1] * yz * sh[:, 5]
                   + SH_C2[2] * (2 * zz - xx - yy) * sh[:, 6] + SH_C2[3] * xz * sh[:, 7]
                   + SH_C2[4] * (xx - yy) * sh[:, 8])
            if deg > 2:
                res = (res + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9] + SH_C3[1] * xy * z * sh[:, 10]
                       + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
                       + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
                       + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + SH_C3[5] * z * (xx - yy) * sh[:, 14]
                       + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return torch.clamp_min(res + 0.5, 0.0)


def render(means, scales, rotations, opacities, viewmatrix, projmatrix, campos, W, H, tanfovx, tanfovy, bg,
           colors=None, sh=None, sh_degree=0, scale_modifier=1.0, antialiasing=False):
    """Returns dict(color [3,H,W], alpha [1,H,W], zdepth [1,H,W], invdepth [1,H,W], radii [N], order).

    viewmatrix / projmatrix are the 3DGS row-vector matrices (p_view = [p, 1] @ V)."""
    dt = torch.float64
    means, scales, rotations = means.to(dt), scales.to(dt), rotations.to(dt)
    opacities = opacities.to(dt).reshape(-1)
    V, Pm = viewmatrix.to(dt), projmatrix.to(dt)
    bg = bg.to(dt)
    N = means.shape[0]
    hom = torch.cat([means, torch.ones(N, 1, dtype=dt, device=means.device)], 1)
    t = (hom @ V)[:, :3]
    ph = hom @ Pm
    pw = 1.0 / (ph[:, 3] + 1e-7)
    pproj = ph[:, :2] * pw[:, None]
    xy = torch.stack([((pproj[:, 0] + 1) * W - 1) * 0.5, ((pproj[:, 1] + 1) * H - 1) * 0.5], 1)
    fy, fx = H / (2 * tanfovy), W / (2 * tanfovx)
    # EWA: J W Sigma W^T J^T with the clamped t (CUDA computeCov2D)
    limx, limy = 1.3 * tanfovx, 1.3 * tanfovy
    tz = t[:, 2]
    tx = torch.clamp(t[:, 0] / tz, -limx, limx) * tz
    ty = torch.clamp(t[:, 1] / tz, -limy, limy) * tz
    zero = torch.zeros_like(tz)
    J = torch.stack([torch.stack([fx / tz, zero, -fx * tx / tz ** 2], -1),
                     torch.stack([zero, fy / tz, -fy * ty / tz ** 2], -1)], 1)          # [N,2,3]
    Rcw = V[:3, :3].T                                                                    # p_cam = Rcw p + t
    # CUDA: M = S * R with glm's column-major R = quat_to_rot^T, Sigma = M^T M = Rq S^2 Rq^T
    Rq = quat_to_rot(rotations)
    S = torch.diag_embed(scales * scale_modifier)
    Mc = S @ Rq.transpose(1, 2)
    Sigma = Mc.transpose(1, 2) @ Mc
    cov2 = J @ Rcw @ Sigma @ Rcw.T @ J.transpose(1, 2)
    a, b, c = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    det0 = a * c - b * b
    a, c = a + 0.3, c + 0.3
    det = a * c - b * b
    op = opacities * (torch.sqrt(torch.clamp_min(det0 / det, 0.000025)) if antialiasing else 1.0)
    conic = torch.stack([c / det, -b / det, a / det], 1)
    mid = 0.5 * (a + c)
    lam1 = mid + torch.sqrt(torch.clamp_min(mid * mid - det, 0.1))
    lam2 = mid - torch.sqrt(torch.clamp_min(mid * mid - det, 0.1))
    radius = torch.ceil(3 * torch.sqrt(torch.maximum(lam1, lam2))).detach()
    gx, gy = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK
    p = xy.detach()
    rmin_x = torch.clamp(((p[:, 0] - radius) / BLOCK).to(torch.int64), 0, gx)
    rmin_y = torch.clamp(((p[:, 1] - radius) / BLOCK).to(torch.int64), 0, gy)
    rmax_x = torch.clamp(((p[:, 0] + radius + BLOCK - 1) / BLOCK).to(torch.int64), 0, gx)
    rmax_y = torch.clamp(((p[:, 1] + radius + BLOCK - 1) / BLOCK).to(torch.int64), 0, gy)
    keep = (tz.detach() > 0.2) & (det.detach() != 0) & ((rmax_x - rmin_x) * (rmax_y - rmin_y) > 0)
    radii = torch.where(keep, radius, torch.zeros_like(radius)).to(torch.int32)
    if colors is None:
        colors = sh_to_rgb(means, campos.to(dt), sh.to(dt), sh_degree)
    colors = colors.to(dt)
    order = torch.argsort(tz.detach())
    # pixel grid; tile membership per (pixel, Gaussian)
    ys, xs = torch.meshgrid(torch.arange(H, device=means.device), torch.arange(W, device=means.device),
                            indexing="ij")
    px, py = xs.reshape(-1).to(dt), ys.reshape(-1).to(dt)
    tx_i, ty_i = (xs.reshape(-1) // BLOCK), (ys.reshape(-1) // BLOCK)
    T = torch.ones(H * W, dtype=dt, device=means.device)
    done = torch.zeros(H * W, dtype=torch.bool, device=means.device)
    C = torch.zeros(H * W, 3, dtype=dt, device=means.device)
    Z = torch.zeros(H * W, dtype=dt, device=means.device)
    I = torch.zeros(H * W, dtype=dt, device=means.device)
    for g in order.tolist():
        if not bool(keep[g]):
            continue
        intile = (tx_i >= rmin_x[g]) & (tx_i < rmax_x[g]) & (ty_i >= rmin_y[g]) & (ty_i < rmax_y[g])
        dx, dy = xy[g, 0] - px, xy[g, 1] - py
        power = -0.5 * (conic[g, 0] * dx * dx + conic[g, 2] * dy * dy) - conic[g, 1] * dx * dy
        alpha = torch.clamp_max(op[g] * torch.exp(power), 0.99)
        live = intile & ~done & (power.detach() <= 0) & (alpha.detach() >= 1.0 / 255.0)
        test_T = T * (1 - alpha)
        stop = live & (test_T.detach() < 0.0001)
        done = done | stop
        use = live & ~stop
        w = torch.where(use, alpha * T, torch.zeros_like(T))
        C = C + w[:, None] * colors[g]
        Z = Z + w * tz[g]
        I = I + w / tz[g]
        T = torch.where(use, test_T, T)
    color = (C + T[:, None] * bg[None, :]).T.reshape(3, H, W)
    return dict(color=color, alpha=(1 - T).reshape(1, H, W), zdepth=Z.reshape(1, H, W),
                invdepth=I.reshape(1, H, W), radii=radii, order=order)
