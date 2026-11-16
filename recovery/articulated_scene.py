"""Canonical static / moving Gaussians transported by the FIXED Phase-3 joint, rendered natively (spec 2.2, 7.4).

    mu_i(f) = mu_i^0                     (static)        R_i(f) = R_i^0
    mu_i(f) = Q_f mu_i^0 + p_f           (moving)        R_i(f) = Q_f R_i^0     (quaternion q(Q_f) * q_i^0)

(Q_f, p_f) come from contracts.Joint.transform (relative to the anchor). They are constant buffers here: no
joint parameter is ever a tensor that requires grad in this module (Phase 5 asserts it). The canonical
tensors are NOT detached, so gradients reach the canonical Gaussians through the transport.

Colour (spec 3.4):
  * world_SH  -- the legacy mode: SH evaluated by the CUDA kernel along the world direction mu_i(f) - C.
                 Exact parity with the upstream renderer; a moving Gaussian's view-dependent colour does
                 NOT rotate with the part.
  * local_SH  -- SH evaluated in torch along the canonical-frame direction Q_f^T (mu_i(f) - C) for moving
                 Gaussians (world direction for static ones), passed as colors_precomp. The SH coefficient
                 rows are never rotated as if they were 3D vectors. Equal to world_SH at the anchor.

Rendering: one native pass returning RGB, alpha = 1 - T_final, Z = sum w z (camera axial) and the legacy
inverse depth; depth = Z / (alpha + eps). antialiasing is refused (pre-existing gradient defect, audit 2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import recovery  # noqa: F401  (pins the RGB-D rasterizer)
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, metric_depth_from

from .raster_oracle import sh_to_rgb

DEPTH_EPS = 1e-6


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, (w, x, y, z), broadcasting."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a proper rotation, w >= 0 (float64)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q if q[0] >= 0 else -q


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


@dataclass
class FrameTransform:
    frame: int
    Q: torch.Tensor      # [3,3]
    p: torch.Tensor      # [3]
    qQ: torch.Tensor     # [4] quaternion of Q


def skew(e: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((), dtype=e.dtype, device=e.device)
    return torch.stack([torch.stack([z, -e[2], e[1]]), torch.stack([e[2], z, -e[0]]),
                        torch.stack([-e[1], e[0], z])])


def joint_transform_torch(jtype: str, axis: torch.Tensor, point, q: torch.Tensor) -> FrameTransform:
    """Tensor-only (Q, p, quaternion) of a P / R joint, differentiable in axis, point and q.

    Used ONLY by the separate, disabled-by-default final joint refinement and by gradient tests. Phase 4
    and Phase 5 use JointTransport (constant buffers)."""
    e = axis / axis.norm()
    I = torch.eye(3, dtype=e.dtype, device=e.device)
    if jtype == "prismatic":
        one = torch.ones((), dtype=e.dtype, device=e.device)
        return FrameTransform(-1, I, q * e, torch.stack([one, one * 0, one * 0, one * 0]))
    K = skew(e)
    Q = I + torch.sin(q) * K + (1 - torch.cos(q)) * (K @ K)
    p = (I - Q) @ point
    qQ = torch.cat([torch.cos(q / 2).reshape(1), torch.sin(q / 2) * e])
    return FrameTransform(-1, Q, p, qQ)


class JointTransport:
    """Fixed per-frame (Q_f, p_f) of the selected joint, as constant device buffers."""

    def __init__(self, joint, device="cuda", dtype=torch.float32):
        self.joint = joint
        self.device, self.dtype = device, dtype
        self._cache = {}

    def at(self, frame: int) -> FrameTransform:
        f = int(frame)
        if f not in self._cache:
            Q, p = self.joint.transform(f)
            t = lambda x: torch.as_tensor(np.asarray(x, float), device=self.device, dtype=self.dtype)
            self._cache[f] = FrameTransform(f, t(Q), t(p), t(rotmat_to_quat(Q)))
        return self._cache[f]


def canonical_tensors(g):
    """Activated canonical tensors of a GaussianModel-like object (attributes _xyz, _rotation, _scaling,
    _opacity, _features_dc, _features_rest; exp / sigmoid / normalize activations)."""
    for name in ("_current_xyz", "_current_rotation"):
        if getattr(g, name, None) is not None:
            raise RuntimeError(f"{name} is set: the model holds a transported state, not canonical tensors")
    return dict(xyz=g._xyz, rot=F.normalize(g._rotation, dim=-1), scale=torch.exp(g._scaling),
                opacity=torch.sigmoid(g._opacity), dc=g._features_dc, rest=g._features_rest)


def posed(xyz, rot, moving: torch.Tensor, ft: FrameTransform):
    """World means [N,3] and quaternions [N,4] at frame f. `moving` is a bool [N] (hard material label)."""
    m = moving[:, None]
    mu = torch.where(m, xyz @ ft.Q.T + ft.p, xyz)
    q = torch.where(m, quat_mul(ft.qQ.expand_as(rot), rot), rot)
    return mu, q


def render_articulated(cam, ft: FrameTransform, g, moving: torch.Tensor, bg: torch.Tensor, sh_mode="local_SH",
                       active_sh_degree=None, scaling_modifier=1.0, antialiasing=False, tensors=None):
    """One native CUDA RGB-D-alpha pass of the whole scene at frame ft.frame from camera `cam`.

    Returns color [3,H,W], alpha [1,H,W], zdepth [1,H,W] (= sum w z), depth [1,H,W] (= Z / (A + eps)),
    invdepth [1,H,W] (legacy), radii [N], viewspace (means2D leaf, for densification statistics)."""
    if antialiasing:
        raise ValueError("antialiasing=True is refused: its covariance gradient is wrong upstream (audit 2)")
    if sh_mode not in ("world_SH", "local_SH"):
        raise ValueError(f"sh_mode {sh_mode!r}")
    t = tensors if tensors is not None else canonical_tensors(g)
    dev = t["xyz"].device
    mu, q = posed(t["xyz"], t["rot"], moving, ft)
    deg = int(active_sh_degree if active_sh_degree is not None else getattr(g, "active_sh_degree", 0))
    ncoef = (deg + 1) ** 2
    sh = torch.cat([t["dc"], t["rest"]], 1)[:, :max(ncoef, 1)].contiguous()
    settings = GaussianRasterizationSettings(
        image_height=cam.H, image_width=cam.W, tanfovx=cam.tanfovx, tanfovy=cam.tanfovy, bg=bg.to(dev).float(),
        scale_modifier=scaling_modifier, viewmatrix=cam.V.to(dev), projmatrix=cam.P.to(dev), sh_degree=deg,
        campos=cam.campos.to(dev), prefiltered=False, debug=False, antialiasing=False)
    R = GaussianRasterizer(settings)
    means2D = torch.zeros(mu.shape[0], 3, device=dev, dtype=mu.dtype, requires_grad=True)
    try:
        means2D.retain_grad()
    except RuntimeError:
        pass
    if sh_mode == "world_SH":
        color, radii, invd, alpha, zd = R.forward_rgbd(mu, means2D, t["opacity"], shs=sh, scales=t["scale"],
                                                       rotations=q)
    else:
        C = cam.campos.to(dev)
        # canonical-frame camera centre for moving Gaussians: Q^T (C - p); world centre for static ones
        c_can = torch.where(moving[:, None], ((C - ft.p) @ ft.Q)[None, :].expand_as(mu), C[None, :].expand_as(mu))
        colors = sh_to_rgb(t["xyz"], c_can, sh, deg)
        color, radii, invd, alpha, zd = R.forward_rgbd(mu, means2D, t["opacity"], colors_precomp=colors,
                                                       scales=t["scale"], rotations=q)
    return dict(color=color, alpha=alpha, zdepth=zd, depth=metric_depth_from(zd, alpha, DEPTH_EPS),
                invdepth=invd, radii=radii, viewspace=means2D, means=mu)
