"""R0 -- the rasteriser as an operator, not a picture generator.

The rasteriser already defines how much each Gaussian owns each pixel:

    Omega_cpi = w_cpi / sum_k w_cpk,    w_cpi = T_cpi * alpha_cpi

and that ownership is the bridge between image-space and Gaussian-space
transport.  The two are not independent problems; the renderer relates them.

This project's ``diff_gaussian_rasterization`` does NOT expose the per-pixel
per-Gaussian weights, and nothing here pretends otherwise.  What it does
expose is ``render_feature_field``, which splats arbitrary per-Gaussian
features through exactly the same alpha compositing -- so for any per-Gaussian
quantity ``f``:

    splat(f)[p] = sum_i Omega_cpi f_i   (up to the normaliser splat(1)[p])

Every quantity this stage needs is of that form, which is why an explicit
``[P, N]`` ownership matrix is never built.  It would also be impossible to
build: 300k pixels by 40k Gaussians is 12 billion entries per camera, and §6
forbids materialising it for exactly that reason.

Verified against the rasteriser: splatting a constant 1 reproduces the
accumulated mass exactly (max error 0.0), so the identity above is not an
approximation of the compositing -- it is the compositing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

EPS = 1.0e-8


@dataclass
class OwnershipMaps:
    """Ownership-weighted quantities for one camera, in image space."""

    total_mass: torch.Tensor        # [H, W] sum_i Omega_cpi (the normaliser)
    channels: torch.Tensor          # [C, H, W] sum_i Omega_cpi f_i
    camera_index: int
    stats: dict[str, Any] = field(default_factory=dict)

    def normalised(self, index: int = 0) -> torch.Tensor:
        """Channel ``index`` divided by the accumulated mass.

        The ratio is the ownership-weighted MEAN of the feature over the
        Gaussians covering that pixel, which is the quantity with a
        probabilistic reading; the unnormalised splat also carries how much
        geometry is there at all, which is why both are kept.
        """
        return self.channels[index] / self.total_mass.clamp_min(EPS)


@torch.no_grad()
def splat_gaussian_features(
    camera,
    gaussians,
    pipe,
    features: torch.Tensor,
    *,
    chunk_channels: int = 8,
    render_feature_fn: Callable | None = None,
) -> torch.Tensor:
    """``sum_i Omega_cpi f_i`` for every channel of ``f``, as ``[C, H, W]``.

    Chunked over channels because the CUDA extension renders a bounded number
    at a time; this is a runtime control and changes nothing numerically.
    """
    if render_feature_fn is None:
        from gaussian_renderer import render_feature_field as render_feature_fn

    features = torch.as_tensor(features).detach().float()
    if features.ndim == 1:
        features = features[:, None]
    outputs: list[torch.Tensor] = []
    for start in range(0, int(features.shape[1]), max(1, int(chunk_channels))):
        block = features[:, start:start + int(chunk_channels)].contiguous()
        outputs.append(render_feature_fn(camera, gaussians, pipe, block)["render"].detach())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def extract_sparse_gaussian_pixel_ownership(
    camera,
    gaussians,
    pipe,
    features: torch.Tensor | None = None,
    *,
    camera_index: int = 0,
    chunk_channels: int = 8,
    render_feature_fn: Callable | None = None,
) -> OwnershipMaps:
    """Ownership normaliser plus any ownership-weighted features for one camera.

    ``features`` is whatever the caller needs projected -- the dynamic
    posterior, a set of patch indicators, per-Gaussian velocities.  Passing
    them together costs one render per channel block rather than one per
    quantity.
    """
    count = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    ones = torch.ones((count, 1), device=device)
    payload = ones if features is None else torch.cat(
        (ones, torch.as_tensor(features).detach().float().to(device).reshape(count, -1)), dim=1
    )
    splatted = splat_gaussian_features(
        camera, gaussians, pipe, payload,
        chunk_channels=int(chunk_channels), render_feature_fn=render_feature_fn,
    )
    total = splatted[0]
    channels = splatted[1:] if int(splatted.shape[0]) > 1 else splatted[:0]
    covered = total > 1.0e-4
    return OwnershipMaps(
        total_mass=total,
        channels=channels,
        camera_index=int(camera_index),
        stats={
            "covered_pixel_fraction": float(covered.float().mean()),
            "median_covered_mass": float(total[covered].median()) if bool(covered.any()) else 0.0,
            "height": int(total.shape[0]), "width": int(total.shape[1]),
        },
    )


@torch.no_grad()
def extract_sparse_visibility_support(
    camera,
    gaussians,
    pipe,
    *,
    render_fn: Callable | None = None,
) -> torch.Tensor:
    """Which Gaussians this camera can see at all, as a ``[N]`` bool mask.

    Visibility is a property of the camera, and a Gaussian invisible from a
    view contributes no evidence there -- it must not be scored as if the
    view had disagreed with it.
    """
    if render_fn is None:
        from gaussian_renderer import render as render_fn
    output = render_fn(camera, gaussians, pipe, torch.zeros(3, device=gaussians.get_xyz.device))
    radii = output["radii"].detach()
    return radii > 0


@torch.no_grad()
def accumulate_gaussian_ownership_mass(
    cameras,
    gaussians,
    pipe,
    *,
    max_cameras: int = 8,
    render_fn: Callable | None = None,
) -> torch.Tensor:
    """How many views each Gaussian is visible in, as ``[N]``.

    Used to weight multi-view fusion by actual observability rather than by
    an arbitrary per-camera constant (§14 forbids camera weights).
    """
    count = int(gaussians.get_xyz.shape[0])
    total = torch.zeros(count, device=gaussians.get_xyz.device)
    for camera in list(cameras)[: int(max_cameras)]:
        total += extract_sparse_visibility_support(camera, gaussians, pipe, render_fn=render_fn).float()
    return total


@torch.no_grad()
def build_patch_indicator_features(
    count: int,
    patch_indices: torch.Tensor,
    patch_valid: torch.Tensor,
    *,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """One channel per patch, 1 on its member Gaussians. ``[N, P]``.

    Splatting these gives each patch's image footprint directly, which is what
    §16's patch-level support score is defined on.  Working at patch
    granularity is also what keeps the lift tractable: the Gaussian-level
    ``P_img`` is an ``N x N`` object nobody can store, while the patch-level
    score is exactly what the candidate ordering consumes.
    """
    patch_indices = torch.as_tensor(patch_indices).detach().long()
    patch_valid = torch.as_tensor(patch_valid).detach().bool()
    patches = int(patch_indices.shape[0])
    features = torch.zeros((int(count), patches), device=device)
    for patch in range(patches):
        members = patch_indices[patch][patch_valid[patch]]
        features[members.to(device), patch] = 1.0
    return features
