"""Lineage-safe densification and pruning for Phase 5 (spec 7.3).

The clone / split / prune rules are those of scene/gaussian_model.py (densify_and_clone, densify_and_split,
prune_points, cat_tensors_to_optimizer, _prune_optimizer), applied to the CANONICAL tensors only:
  * clone:  |grad_2D| >= thr and max scale <= percent_dense * extent  -> exact copy
  * split:  |grad_2D| >= thr and max scale >  percent_dense * extent  -> N samples ~ N(0, S) rotated by the
            CANONICAL rotation around the CANONICAL mean, scales / (0.8 N); the parent is removed
Every child inherits the parent's HARD material and gets a new persistent gaussian_uid, parent_uid = parent's
uid, original_id = -1, provenance DENSIFY_CLONE / DENSIFY_SPLIT. Parameters, Adam moments and every lineage
array change in one step (new rows get zero moments, as in GaussianModel).

Densification never reads a transported position: the trainer renders mu(f) = Q_f mu0 + p_f from the
canonical leaf, and only the leaf (_xyz) is cloned / split / saved.

Pruning (declared, not GaussianModel's screen-size rule): a Gaussian is removed only if
  (opacity < min_opacity AND it was rasterized in >= 1 view of the last full sweep -- the data saw it and
   drove it transparent)  OR  max scale > big_ws * extent.
A Gaussian unseen in the whole sweep is never pruned for low opacity (no evidence), and Phase-4 geometry is
not pruned on an opacity reset (Phase 5 performs no opacity reset).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .articulated_scene import quat_to_rotmat
from .gaussians import PROV_DENSIFY_CLONE, PROV_DENSIFY_SPLIT

PARAMS = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")
ATTR = {"xyz": "_xyz", "f_dc": "_features_dc", "f_rest": "_features_rest", "opacity": "_opacity",
        "scaling": "_scaling", "rotation": "_rotation"}


def _set(g, name, t):
    setattr(g, ATTR[name], t)


def cat_rows(g, opt, new: dict):
    """Append rows to every parameter (trainable or frozen) and to its Adam state."""
    for name in PARAMS:
        old = getattr(g, ATTR[name])
        ext = new[name].to(old.device, old.dtype)
        group = next((gr for gr in opt.param_groups if gr["name"] == name), None) if opt else None
        if group is None:
            t = torch.cat([old.detach(), ext], 0)
            _set(g, name, nn.Parameter(t, requires_grad=old.requires_grad) if isinstance(old, nn.Parameter) else t)
            continue
        st = opt.state.pop(group["params"][0], None)
        p = nn.Parameter(torch.cat([group["params"][0].detach(), ext], 0).requires_grad_(True))
        if st is not None:
            st["exp_avg"] = torch.cat([st["exp_avg"], torch.zeros_like(ext)], 0)
            st["exp_avg_sq"] = torch.cat([st["exp_avg_sq"], torch.zeros_like(ext)], 0)
            opt.state[p] = st
        group["params"][0] = p
        _set(g, name, p)


def keep_rows(g, opt, keep: torch.Tensor):
    for name in PARAMS:
        old = getattr(g, ATTR[name])
        group = next((gr for gr in opt.param_groups if gr["name"] == name), None) if opt else None
        if group is None:
            t = old.detach()[keep]
            _set(g, name, nn.Parameter(t, requires_grad=old.requires_grad) if isinstance(old, nn.Parameter) else t)
            continue
        st = opt.state.pop(group["params"][0], None)
        p = nn.Parameter(group["params"][0].detach()[keep].requires_grad_(True))
        if st is not None:
            st["exp_avg"], st["exp_avg_sq"] = st["exp_avg"][keep], st["exp_avg_sq"][keep]
            opt.state[p] = st
        group["params"][0] = p
        _set(g, name, p)
    k = keep.cpu().numpy()
    for kk in list(g.lineage):
        g.lineage[kk] = g.lineage[kk][k]


def _append_lineage(g, parents: np.ndarray, repeat: int, prov: int):
    lin = g.lineage
    start = int(lin["gaussian_uid"].max()) + 1
    n = len(parents) * repeat
    par = np.tile(parents, repeat)
    lin["material"] = np.concatenate([lin["material"], lin["material"][par]])
    lin["parent_uid"] = np.concatenate([lin["parent_uid"], lin["gaussian_uid"][par]])
    lin["gaussian_uid"] = np.concatenate([lin["gaussian_uid"], np.arange(start, start + n, dtype=np.int64)])
    lin["original_id"] = np.concatenate([lin["original_id"], np.full(n, -1, np.int64)])
    lin["provenance"] = np.concatenate([lin["provenance"], np.full(n, prov, np.int8)])


def densify(g, opt, grad_norm: torch.Tensor, thr: float, extent: float, percent_dense: float, N: int = 2,
            generator=None) -> dict:
    """grad_norm [N] = accumulated mean |viewspace grad| (as GaussianModel). Returns counts."""
    if getattr(g, "_current_xyz", None) is not None:
        raise RuntimeError("refusing to densify a transported state")
    scale = torch.exp(g._scaling.detach())
    big = scale.max(1).values > percent_dense * extent
    sel = grad_norm >= thr
    clone = (sel & ~big).nonzero().reshape(-1)
    split = (sel & big).nonzero().reshape(-1)
    n0 = len(g)
    if len(clone):
        new = {n: getattr(g, ATTR[n]).detach()[clone] for n in PARAMS}
        cat_rows(g, opt, new)
        _append_lineage(g, clone.cpu().numpy(), 1, PROV_DENSIFY_CLONE)
    if len(split):
        s = scale[split].repeat(N, 1)
        samples = torch.normal(torch.zeros_like(s), s, generator=generator)
        R = quat_to_rotmat(g._rotation.detach()[split]).repeat(N, 1, 1)
        new = {"xyz": (R @ samples[..., None])[..., 0] + g._xyz.detach()[split].repeat(N, 1),
               "scaling": torch.log(s / (0.8 * N)),
               "rotation": g._rotation.detach()[split].repeat(N, 1),
               "f_dc": g._features_dc.detach()[split].repeat(N, 1, 1),
               "f_rest": g._features_rest.detach()[split].repeat(N, 1, 1),
               "opacity": g._opacity.detach()[split].repeat(N, 1)}
        cat_rows(g, opt, new)
        _append_lineage(g, split.cpu().numpy(), N, PROV_DENSIFY_SPLIT)
        keep = torch.ones(len(g), dtype=torch.bool, device=g._xyz.device)
        keep[split] = False
        keep_rows(g, opt, keep)
    g.check()
    return {"cloned": int(len(clone)), "split": int(len(split)), "before": n0, "after": len(g)}


def prune(g, opt, seen_in_sweep: torch.Tensor, min_opacity: float, big_ws: float, extent: float) -> dict:
    op = torch.sigmoid(g._opacity.detach()).reshape(-1)
    big = torch.exp(g._scaling.detach()).max(1).values > big_ws * extent
    drop = ((op < min_opacity) & seen_in_sweep) | big
    n0 = len(g)
    if bool(drop.any()):
        keep_rows(g, opt, ~drop)
    g.check()
    return {"pruned": int(drop.sum()), "pruned_low_opacity_seen": int(((op < min_opacity) & seen_in_sweep).sum()),
            "pruned_too_big": int(big.sum()), "kept_low_opacity_unseen": int(((op < min_opacity) & ~seen_in_sweep).sum()),
            "before": n0, "after": len(g)}
