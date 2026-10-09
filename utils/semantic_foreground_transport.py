"""Semantic-foreground firewall for image/video transport.

Every image-side dynamic path must make the foreground domain explicit.  These
helpers are intentionally boring: they zero invalid pixels before any scoring
or normalisation can accidentally see background support.
"""

from __future__ import annotations

import torch

EPS = 1.0e-12


def _flat_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(mask).detach().bool().reshape(-1)


def apply_foreground_domain(value: torch.Tensor, semantic_foreground_mask: torch.Tensor, *, fill: float = 0.0) -> torch.Tensor:
    """Return ``value`` with non-foreground pixels set to ``fill``."""
    tensor = torch.as_tensor(value)
    mask = torch.as_tensor(semantic_foreground_mask, device=tensor.device).bool()
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(0)
    return torch.where(mask, tensor, torch.as_tensor(fill, device=tensor.device, dtype=tensor.dtype))


def build_foreground_pixel_indices(
    semantic_foreground_mask: torch.Tensor,
    *,
    max_pixels: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Flat pixel indices where the semantic foreground mask is true."""
    index = torch.nonzero(_flat_mask(semantic_foreground_mask), as_tuple=False).reshape(-1)
    if max_pixels is not None and int(max_pixels) > 0 and int(index.numel()) > int(max_pixels):
        pick = torch.randperm(int(index.numel()), generator=generator, device=index.device)[: int(max_pixels)]
        index = index[pick]
    return index.long()


def mask_sparse_pixel_transport(
    source_pixel_indices: torch.Tensor,
    target_pixel_indices: torch.Tensor,
    values: torch.Tensor,
    source_foreground_mask: torch.Tensor,
    target_foreground_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter/zero sparse transport so every surviving edge is fg->fg.

    ``target_pixel_indices`` may be ``[E]`` or ``[S, K]``.  In the row-wise
    case rows with no legal target are removed and illegal target entries get
    zero value.
    """
    src = torch.as_tensor(source_pixel_indices).long()
    tgt = torch.as_tensor(target_pixel_indices).long()
    val = torch.as_tensor(values)
    src_fg = _flat_mask(source_foreground_mask).to(src.device)
    tgt_fg = _flat_mask(target_foreground_mask).to(tgt.device)
    if tgt.ndim == 1:
        keep = src_fg[src] & tgt_fg[tgt]
        return src[keep], tgt[keep], val[keep]
    if tgt.ndim != 2:
        raise ValueError("target_pixel_indices must be [E] or [S,K]")
    legal = src_fg[src][:, None] & tgt_fg[tgt]
    kept_rows = legal.any(dim=1)
    clean_values = torch.where(legal.to(val.device), val, torch.zeros((), device=val.device, dtype=val.dtype))
    return src[kept_rows], tgt[kept_rows], clean_values[kept_rows]


def renormalize_foreground_ownership(
    ownership: torch.Tensor,
    *,
    semantic_foreground_mask: torch.Tensor | None = None,
    foreground_gaussian_mask: torch.Tensor | None = None,
    gaussian_dim: int = -1,
) -> torch.Tensor:
    """Mask ownership and normalise over the Gaussian/support dimension."""
    own = torch.as_tensor(ownership).detach().float().clone()
    dim = gaussian_dim if gaussian_dim >= 0 else own.ndim + gaussian_dim
    if foreground_gaussian_mask is not None:
        gmask = torch.as_tensor(foreground_gaussian_mask, device=own.device).bool()
        shape = [1] * own.ndim
        shape[dim] = int(gmask.numel())
        own = own * gmask.reshape(shape).float()
    if semantic_foreground_mask is not None:
        pmask = torch.as_tensor(semantic_foreground_mask, device=own.device).bool()
        if own.ndim >= 3 and pmask.ndim == 2:
            while pmask.ndim < own.ndim:
                pmask = pmask.unsqueeze(0)
            own = own * pmask.float()
        elif own.ndim == 2:
            own = own * pmask.reshape(-1, 1).float()
        else:
            own = own * pmask.float()
    return own / own.sum(dim=dim, keepdim=True).clamp_min(EPS)
