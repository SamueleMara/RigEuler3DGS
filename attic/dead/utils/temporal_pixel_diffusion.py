"""Foreground-only local pixel diffusion sampler.

The sampler is deliberately local and masked: it is used only to expose
multimodal adjacent-frame candidates without ever sampling background pixels.
"""

from __future__ import annotations

import torch

from utils.semantic_foreground_transport import build_foreground_pixel_indices


def _chw(image: torch.Tensor) -> torch.Tensor:
    img = torch.as_tensor(image).detach().float()
    if img.ndim == 2:
        img = img[None]
    if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
        img = img.permute(2, 0, 1)
    return img[:3]


def _xy(idx: torch.Tensor, width: int) -> torch.Tensor:
    return torch.stack((idx % width, idx // width), dim=-1).float()


def sample_forward_pixel_transport(
    source_pixel_indices: torch.Tensor,
    target_foreground_mask: torch.Tensor,
    *,
    image_shape: tuple[int, int],
    samples: int = 8,
    radius: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    src = torch.as_tensor(source_pixel_indices).long()
    targets = build_foreground_pixel_indices(target_foreground_mask)
    if int(src.numel()) == 0 or int(targets.numel()) == 0:
        return torch.empty((int(src.numel()), 0), dtype=torch.long)
    h, w = int(image_shape[0]), int(image_shape[1])
    dist = torch.cdist(_xy(src, w), _xy(targets, w))
    if radius is not None:
        dist = torch.where(dist <= float(radius), dist, torch.full_like(dist, float("inf")))
    k = min(int(samples), int(targets.numel()))
    order = torch.topk(dist, k=k, largest=False, dim=1).indices
    out = targets[order]
    if generator is not None and k > 1:
        perm = torch.stack([torch.randperm(k, generator=generator) for _ in range(int(src.numel()))])
        out = out.gather(1, perm)
    return out


def sample_backward_pixel_transport(*args, **kwargs) -> torch.Tensor:
    return sample_forward_pixel_transport(*args, **kwargs)


def score_samples_with_rgbd(
    source_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    source_pixel_indices: torch.Tensor,
    sampled_target_indices: torch.Tensor,
    target_foreground_mask: torch.Tensor,
) -> torch.Tensor:
    src = torch.as_tensor(source_pixel_indices).long()
    tgt = torch.as_tensor(sampled_target_indices).long()
    fg = torch.as_tensor(target_foreground_mask).bool().reshape(-1)
    if tgt.numel() == 0:
        return torch.empty_like(tgt, dtype=torch.float32)
    si = _chw(source_rgb).reshape(_chw(source_rgb).shape[0], -1).T[src]
    ti = _chw(target_rgb).reshape(_chw(target_rgb).shape[0], -1).T[tgt.reshape(-1)].reshape(*tgt.shape, -1)
    sd = torch.as_tensor(source_depth).float().reshape(-1)[src]
    td = torch.as_tensor(target_depth).float().reshape(-1)[tgt]
    cost = torch.linalg.norm(ti - si[:, None, :], dim=-1) + (td - sd[:, None]).abs() / sd.abs().median().clamp_min(1.0e-6)
    return torch.where(fg[tgt], -cost, torch.full_like(cost, -float("inf")))
