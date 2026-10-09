"""Foreground-only SAM3 overlapping-mask affinity distillation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from utils.loss_utils import _dynamic_root_for_train, _load_mask_for_view


@dataclass
class SAM3MaskPack:
    masks: torch.Tensor
    scores: torch.Tensor | None
    foreground_fraction: torch.Tensor | None
    image_hw: tuple[int, int]
    mask_hw: tuple[int, int]
    path: str = ""


@dataclass
class PreparedSAM3Masks:
    masks: torch.Tensor
    rejected_masks: torch.Tensor
    weights: torch.Tensor
    valid_mask: torch.Tensor
    membership_count: torch.Tensor
    foreground_mask: torch.Tensor


def _sam3_mask_root_for_train(dataset, args) -> Path:
    requested = str(getattr(args, "sam3_mask_root", "") or "").strip()
    if requested:
        path = Path(requested).expanduser()
        path = path if path.is_absolute() else (_dynamic_root_for_train(dataset, args) / path)
        if path.name == "sam3":
            sibling = path.with_name("sam3_masks")
            if _root_has_mask_cache(sibling) and not _root_has_mask_cache(path):
                print(f"[SAM3 affinity] {path} contains dense features; using sibling mask cache {sibling}")
                return sibling.resolve()
        return path.resolve()
    return _dynamic_root_for_train(dataset, args) / "sam3_masks"


def _is_mask_cache_file(path: Path) -> bool:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return isinstance(obj, dict) and ("masks" in obj or "mask" in obj)


def _root_has_mask_cache(root: Path) -> bool:
    root = Path(root)
    if not root.exists():
        return False
    for path in root.rglob("frame_*.pt"):
        return _is_mask_cache_file(path)
    return False


def _camera_key(camera):
    cam_id = getattr(camera, "dynamic_camera_id", None)
    frame_id = getattr(camera, "frame_id", None)
    if cam_id is not None and frame_id is not None:
        return str(cam_id), int(frame_id)
    image_path = getattr(camera, "dynamic_image_path", "")
    if image_path:
        p = Path(image_path)
        match = re.search(r"frame[_-]?(\d+)", p.stem)
        if match:
            return p.parent.name, int(match.group(1))
    name = str(getattr(camera, "image_name", ""))
    cam_match = re.search(r"cam_\d+", name)
    frame_match = re.search(r"frame[_-]?(\d+)", name)
    if cam_match is not None:
        return cam_match.group(0), int(frame_match.group(1)) if frame_match else 0
    return Path(name).stem, 0


class SAM3MaskCache:
    def __init__(self, root, device="cpu"):
        self.root = Path(root).expanduser().resolve()
        self.device = device
        self._cache: dict[Path, SAM3MaskPack] = {}

    def path_for_camera(self, camera) -> Path | None:
        cam_id, frame_id = _camera_key(camera)
        candidates = [
            self.root / cam_id / f"frame_{int(frame_id):06d}.pt",
            self.root / cam_id / f"{Path(str(getattr(camera, 'image_name', ''))).stem}.pt",
            self.root / f"{Path(str(getattr(camera, 'image_name', ''))).stem}.pt",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def get(self, camera) -> SAM3MaskPack:
        path = self.path_for_camera(camera)
        if path is None:
            cam_id, frame_id = _camera_key(camera)
            raise FileNotFoundError(
                f"SAM3 mask cache missing for {cam_id} frame {int(frame_id):06d} under {self.root}"
            )
        if path not in self._cache:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            masks = obj.get("masks", obj.get("mask")) if isinstance(obj, dict) else obj
            if masks is None:
                keys = sorted(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
                raise RuntimeError(
                    f"{path} is not a SAM3 mask-cache file; keys={keys}. "
                    "You passed the dense SAM3 feature root. Use ROOT/sam3_masks, "
                    "or run tools/extract_sam3_masks.py first."
                )
            scores = obj.get("scores") if isinstance(obj, dict) else None
            fraction = obj.get("foreground_fraction") if isinstance(obj, dict) else None
            image_hw = tuple(obj.get("image_hw", masks.shape[-2:])) if isinstance(obj, dict) else tuple(masks.shape[-2:])
            mask_hw = tuple(obj.get("mask_hw", masks.shape[-2:])) if isinstance(obj, dict) else tuple(masks.shape[-2:])
            masks = torch.as_tensor(masks).float()
            if masks.ndim == 2:
                masks = masks[None]
            if masks.ndim != 3:
                raise RuntimeError(f"SAM3 masks must be [M,H,W], got {tuple(masks.shape)} in {path}")
            scores_t = None if scores is None else torch.as_tensor(scores).float().reshape(-1)
            frac_t = None if fraction is None else torch.as_tensor(fraction).float().reshape(-1)
            self._cache[path] = SAM3MaskPack(
                masks=masks.to(self.device),
                scores=None if scores_t is None else scores_t.to(self.device),
                foreground_fraction=None if frac_t is None else frac_t.to(self.device),
                image_hw=(int(image_hw[0]), int(image_hw[1])),
                mask_hw=(int(mask_hw[0]), int(mask_hw[1])),
                path=str(path),
            )
        return self._cache[path]


def prepare_foreground_sam3_masks(mask_pack, foreground_mask, args):
    fg = torch.as_tensor(foreground_mask, dtype=torch.float32, device=mask_pack.masks.device)
    if fg.ndim == 3:
        fg = fg[0]
    fg_bin = fg > 0.5
    masks = mask_pack.masks.float().to(fg.device)
    if masks.shape[-2:] != fg_bin.shape:
        masks = F.interpolate(
            masks[:, None],
            size=tuple(fg_bin.shape),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    masks = masks.clamp(0.0, 1.0) * fg_bin.float()[None]
    binary = masks > 0.5
    fg_area = fg_bin.sum().float().clamp_min(1.0)
    fractions = binary.flatten(1).sum(1).float() / fg_area
    keep = (
        (fractions >= float(getattr(args, "sam3_mask_min_fg_fraction", 0.001)))
        & (fractions <= float(getattr(args, "sam3_mask_max_fg_fraction", 0.95)))
        & (binary.flatten(1).sum(1) > 0)
    )
    rejected = masks[~keep]
    masks = masks[keep]
    if masks.numel() == 0:
        h, w = fg_bin.shape
        return PreparedSAM3Masks(
            masks=masks.reshape(0, h, w),
            rejected_masks=rejected.reshape(-1, h, w),
            weights=torch.empty(0, device=fg.device),
            valid_mask=torch.zeros((h, w), dtype=torch.bool, device=fg.device),
            membership_count=torch.zeros((h, w), dtype=torch.float32, device=fg.device),
            foreground_mask=fg_bin,
        )
    fractions = fractions[keep]
    # Broad preference for nontrivial sub-object masks. Confidence only scales
    # proposals; it never turns missing SAM3 evidence into negative evidence.
    weights = torch.ones_like(fractions)
    weights = weights * (1.0 - ((fractions - 0.35).abs() / 0.35).clamp(0.0, 1.0) * 0.5)
    if bool(getattr(args, "sam3_mask_use_confidence", True)) and mask_pack.scores is not None:
        score = mask_pack.scores.to(fg.device)
        if score.numel() == keep.numel():
            weights = weights * score[keep].clamp(0.05, 1.0)
    membership_count = (masks > 0.5).float().sum(0)
    valid_mask = fg_bin & (
        (masks.square() * weights.square()[:, None, None]).sum(0).sqrt() > 1.0e-8
    )
    return PreparedSAM3Masks(
        masks=masks,
        rejected_masks=rejected,
        weights=weights.float(),
        valid_mask=valid_mask,
        membership_count=membership_count,
        foreground_mask=fg_bin,
    )


def sample_foreground_pixels(foreground_mask, sam3_valid_mask, n_pixels, generator=None):
    valid = torch.as_tensor(foreground_mask).bool() & torch.as_tensor(sam3_valid_mask).bool()
    ids = torch.nonzero(valid, as_tuple=False)
    if ids.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=valid.device)
    take = min(int(n_pixels), int(ids.shape[0]))
    perm = torch.randperm(ids.shape[0], device=valid.device, generator=generator)[:take]
    return ids.index_select(0, perm).long()


def gather_membership_matrix(masks, weights, pixel_idx):
    if pixel_idx.numel() == 0 or masks.numel() == 0:
        return torch.empty((0, int(masks.shape[0])), device=masks.device, dtype=torch.float32)
    y = pixel_idx[:, 0].long()
    x = pixel_idx[:, 1].long()
    return (masks[:, y, x].T.float() * weights.float()[None]).contiguous()


def compute_target_affinity(H, eps=1.0e-8):
    if H.numel() == 0:
        return torch.empty((H.shape[0], H.shape[0]), device=H.device, dtype=torch.float32)
    Hn = F.normalize(H.float(), dim=-1, eps=eps)
    return (Hn @ Hn.T).clamp(0.0, 1.0)


def compute_rendered_affinity(Z, eps=1.0e-8):
    if Z.numel() == 0:
        return torch.empty((Z.shape[0], Z.shape[0]), device=Z.device, dtype=torch.float32)
    Zn = F.normalize(Z.float(), dim=-1, eps=eps)
    return (Zn @ Zn.T).clamp(-1.0, 1.0)


def _sample_bin(mask, count):
    ids = torch.nonzero(mask, as_tuple=False)
    if ids.numel() == 0 or int(count) <= 0:
        return torch.empty((0, 2), dtype=torch.long, device=mask.device)
    if ids.shape[0] <= int(count):
        return ids.long()
    perm = torch.randperm(ids.shape[0], device=mask.device)[: int(count)]
    return ids.index_select(0, perm).long()


def sample_balanced_pairs(target_affinity, valid_pair_mask, args, predicted_affinity=None):
    n = int(target_affinity.shape[0])
    offdiag = ~torch.eye(n, dtype=torch.bool, device=target_affinity.device)
    valid = valid_pair_mask.bool() & offdiag
    pos = valid & (target_affinity >= float(getattr(args, "sam3_affinity_pos_threshold", 0.75)))
    neg = valid & (target_affinity <= float(getattr(args, "sam3_affinity_neg_threshold", 0.25)))
    mid = valid & ~(pos | neg)
    pairs = [
        _sample_bin(pos, int(getattr(args, "sam3_affinity_positive_pairs", 512))),
        _sample_bin(mid, int(getattr(args, "sam3_affinity_intermediate_pairs", 256))),
        _sample_bin(neg, int(getattr(args, "sam3_affinity_negative_pairs", 512))),
    ]
    frac = float(getattr(args, "sam3_affinity_hard_pair_fraction", 0.0))
    if frac > 0.0 and predicted_affinity is not None:
        pred01 = ((predicted_affinity + 1.0) * 0.5).detach()
        hard_pos = pos & (pred01 < target_affinity)
        hard_neg = neg & (pred01 > target_affinity)
        hard_count = int(frac * max(1, sum(int(p.shape[0]) for p in pairs)))
        pairs.extend([_sample_bin(hard_pos, hard_count // 2), _sample_bin(hard_neg, hard_count - hard_count // 2)])
    pairs = [p for p in pairs if p.numel() > 0]
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=target_affinity.device)
    return torch.cat(pairs, dim=0)


def sam3_affinity_loss(target_affinity, predicted_affinity, pair_idx, args):
    if pair_idx.numel() == 0:
        zero = predicted_affinity.sum() * 0.0
        return zero, {}
    y = target_affinity[pair_idx[:, 0], pair_idx[:, 1]].float()
    pred = predicted_affinity[pair_idx[:, 0], pair_idx[:, 1]].float()
    if str(getattr(args, "sam3_affinity_loss", "soft_bce")) == "l1":
        loss = (((pred + 1.0) * 0.5) - y).abs().mean()
    else:
        logits = (pred - float(getattr(args, "sam3_affinity_bias", 0.0))) / float(
            getattr(args, "sam3_affinity_temperature", 0.1)
        )
        loss = F.binary_cross_entropy_with_logits(logits, y)
    pos = y >= float(getattr(args, "sam3_affinity_pos_threshold", 0.75))
    neg = y <= float(getattr(args, "sam3_affinity_neg_threshold", 0.25))
    mid = ~(pos | neg)
    stats = {
        "mean_target_affinity": float(y.detach().mean()) if y.numel() else 0.0,
        "mean_pred_affinity": float(pred.detach().mean()) if pred.numel() else 0.0,
        "positive_pred_affinity": float(pred.detach()[pos].mean()) if bool(pos.any()) else 0.0,
        "negative_pred_affinity": float(pred.detach()[neg].mean()) if bool(neg.any()) else 0.0,
        "intermediate_pred_affinity": float(pred.detach()[mid].mean()) if bool(mid.any()) else 0.0,
        "mae_affinity": float((((pred.detach() + 1.0) * 0.5) - y.detach()).abs().mean()) if y.numel() else 0.0,
        "pair_count": int(pair_idx.shape[0]),
        "positive_pairs": int(pos.sum()),
        "negative_pairs": int(neg.sum()),
        "intermediate_pairs": int(mid.sum()),
    }
    return loss, stats


def gaussian_affinity(features, idx_a, idx_b):
    fa = F.normalize(features[idx_a].float(), dim=-1, eps=1.0e-8)
    fb = F.normalize(features[idx_b].float(), dim=-1, eps=1.0e-8)
    return (fa * fb).sum(-1)


def build_local_affinity_graph(xyz, sam3_features, knn):
    xyz = torch.as_tensor(xyz).detach().float().cpu()
    feat = torch.as_tensor(sam3_features).detach().float().cpu()
    n = int(xyz.shape[0])
    if n == 0:
        return {"edge_index": torch.empty((2, 0), dtype=torch.long), "affinity": torch.empty(0)}
    k = min(max(1, int(knn)), max(1, n - 1))
    src_chunks = []
    dst_chunks = []
    for start in range(0, n, 1024):
        end = min(n, start + 1024)
        dist = torch.cdist(xyz[start:end], xyz)
        ids = dist.topk(k + 1, largest=False).indices[:, 1:]
        src_chunks.append(torch.arange(start, end).repeat_interleave(k))
        dst_chunks.append(ids.reshape(-1))
    src = torch.cat(src_chunks)
    dst = torch.cat(dst_chunks)
    aff = ((gaussian_affinity(feat, src, dst) + 1.0) * 0.5).clamp(0.0, 1.0)
    return {"edge_index": torch.stack((src.long(), dst.long())), "affinity": aff.float()}


def _write_tensor_image(path: Path, image: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = image.detach().float().cpu()
    if value.ndim == 2:
        finite = torch.isfinite(value)
        if bool(finite.any()):
            lo, hi = torch.quantile(value[finite], torch.tensor([0.02, 0.98]))
        else:
            lo, hi = torch.tensor(0.0), torch.tensor(1.0)
        value = ((value - lo) / (hi - lo).clamp_min(1.0e-6)).clamp(0.0, 1.0)
        value = value[None].repeat(3, 1, 1)
    if value.shape[0] == 1:
        value = value.repeat(3, 1, 1)
    if value.shape[0] > 3:
        value = value[:3]
    arr = (value.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def _mask_overlay(masks: torch.Tensor, image: torch.Tensor | None = None):
    if image is None:
        h, w = masks.shape[-2:] if masks.numel() else (1, 1)
        rgb = torch.zeros((3, h, w), dtype=torch.float32)
    else:
        rgb = image.detach().float().cpu().clamp(0.0, 1.0).clone()
    if masks.numel() == 0:
        return rgb
    colors = torch.tensor(
        [
            [0.90, 0.20, 0.15],
            [0.20, 0.70, 0.95],
            [0.25, 0.85, 0.30],
            [0.95, 0.75, 0.20],
            [0.75, 0.35, 0.95],
            [0.95, 0.45, 0.10],
        ],
        dtype=torch.float32,
    )
    for idx, mask in enumerate(masks.detach().float().cpu()):
        alpha = (0.35 * mask.clamp(0.0, 1.0))[None]
        color = colors[idx % colors.shape[0]][:, None, None]
        rgb = rgb * (1.0 - alpha) + color * alpha
    return rgb.clamp(0.0, 1.0)


def _query_affinity_image(queries, source_features, membership=None):
    rows = []
    for y, x in queries:
        if membership is not None:
            q = membership[:, int(y), int(x)]
            feat = membership.flatten(1).T
            qn = F.normalize(q[None].float(), dim=-1, eps=1.0e-8)
            fn = F.normalize(feat.float(), dim=-1, eps=1.0e-8)
            aff = (fn @ qn.T).reshape(source_features.shape[-2:])
        else:
            feature = F.normalize(source_features.float(), dim=0, eps=1.0e-8)
            q = feature[:, int(y), int(x)]
            aff = (feature * q[:, None, None]).sum(0)
        rows.append(aff.clamp(0.0, 1.0) if membership is not None else aff.clamp(-1.0, 1.0))
    return torch.cat(rows, dim=1) if rows else torch.empty(0)


def _write_debug(output_dir, stage_name, camera, prepared, render_pkg, pixel_idx):
    cam_id, frame_id = _camera_key(camera)
    root = Path(output_dir) / "sam3_debug" / str(stage_name) / str(cam_id) / f"frame_{int(frame_id):06d}"
    root.mkdir(parents=True, exist_ok=True)
    image = getattr(camera, "original_image", None)
    _write_tensor_image(root / "foreground.png", prepared.foreground_mask.float())
    _write_tensor_image(root / "accepted_masks_overlay.png", _mask_overlay(prepared.masks, image))
    _write_tensor_image(root / "rejected_masks_overlay.png", _mask_overlay(prepared.rejected_masks, image))
    _write_tensor_image(root / "membership_count.png", prepared.membership_count)
    if pixel_idx.numel() == 0:
        return
    queries = pixel_idx[: min(4, pixel_idx.shape[0])].detach().cpu().tolist()
    membership = prepared.masks.float() * prepared.weights[:, None, None]
    target = _query_affinity_image(queries, prepared.valid_mask.float(), membership=membership)
    pred = _query_affinity_image(queries, render_pkg["feature"].detach(), membership=None)
    _write_tensor_image(root / "sam3_target_affinity_queries.png", target)
    _write_tensor_image(root / "rendered_gaussian_affinity_queries.png", ((pred + 1.0) * 0.5).clamp(0.0, 1.0))


def _write_gaussian_html(path: Path, xyz: torch.Tensor, features: torch.Tensor, max_points: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = xyz.detach().float().cpu()
    feat = features.detach().float().cpu()
    n = int(xyz.shape[0])
    if n > int(max_points):
        ids = torch.linspace(0, n - 1, int(max_points)).round().long().unique()
        xyz = xyz.index_select(0, ids)
        feat = feat.index_select(0, ids)
    centered = feat - feat.mean(0, keepdim=True)
    if centered.shape[0] >= 3 and centered.shape[1] >= 3:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        rgb = centered @ vh[:3].T
    else:
        rgb = F.pad(centered, (0, max(0, 3 - centered.shape[1])))[:, :3]
    lo = rgb.quantile(0.02, dim=0)
    hi = rgb.quantile(0.98, dim=0)
    rgb = ((rgb - lo) / (hi - lo).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    colors = [
        f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        for r, g, b in rgb.tolist()
    ]
    data = {
        "x": xyz[:, 0].tolist(),
        "y": xyz[:, 1].tolist(),
        "z": xyz[:, 2].tolist(),
        "color": colors,
    }
    path.write_text(
        """<!doctype html><meta charset="utf-8"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<div id="plot" style="width:100vw;height:100vh"></div>
<script>
const d = DATA;
Plotly.newPlot('plot', [{type:'scatter3d', mode:'markers', x:d.x, y:d.y, z:d.z,
 marker:{size:2,color:d.color,opacity:0.85}}], {margin:{l:0,r:0,t:0,b:0}, scene:{aspectmode:'data'}});
</script>
""".replace("DATA", json.dumps(data)),
        encoding="utf-8",
    )


def _freeze_for_affinity(gaussians):
    trainable_name = "_sam3_affinity_field"
    for name in (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
        "_exposure",
        "_semantic_logits",
        "_dino_feature_field",
        "_sam3_feature_field",
    ):
        value = getattr(gaussians, name, None)
        if torch.is_tensor(value) and value.numel() > 0:
            value.requires_grad_(False)
    field = getattr(gaussians, trainable_name, None)
    if not torch.is_tensor(field) or field.numel() == 0:
        raise RuntimeError("SAM3 affinity field was not initialized")
    field.requires_grad_(True)
    for name in (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
        "_semantic_logits",
        "_dino_feature_field",
        "_sam3_feature_field",
    ):
        value = getattr(gaussians, name, None)
        if torch.is_tensor(value) and value.numel() > 0 and value.requires_grad:
            raise RuntimeError(f"SAM3 affinity phase would optimize {name}; this is forbidden")


def _field_training_config(args):
    keys = [
        "sam3_affinity_dim",
        "sam3_affinity_iters",
        "sam3_affinity_lr",
        "sam3_mask_min_fg_fraction",
        "sam3_mask_max_fg_fraction",
        "sam3_mask_use_confidence",
        "sam3_affinity_pixels_per_iter",
        "sam3_affinity_positive_pairs",
        "sam3_affinity_intermediate_pairs",
        "sam3_affinity_negative_pairs",
        "sam3_affinity_pos_threshold",
        "sam3_affinity_neg_threshold",
        "sam3_affinity_loss",
        "sam3_affinity_temperature",
        "sam3_affinity_bias",
        "sam3_affinity_norm_weight",
        "sam3_affinity_hard_pair_fraction",
    ]
    return {key: getattr(args, key) for key in keys if hasattr(args, key)}


def train_sam3_affinity_field(
    gaussians,
    cameras,
    pipe,
    dataset,
    args,
    output_dir,
    *,
    writer=None,
    wb_run=None,
    start_step=0,
    stage_name="canonical",
    frame_idx=None,
):
    if int(getattr(args, "sam3_affinity_iters", 0)) <= 0:
        return {}
    from gaussian_renderer import render_sam3_affinity_field

    cameras = list(cameras)
    if not cameras:
        raise RuntimeError(f"No cameras available for SAM3 affinity training ({stage_name})")
    gaussians.init_sam3_affinity_field(
        int(getattr(args, "sam3_affinity_dim", 16)),
        init_random=True,
        requires_grad=True,
    )
    _freeze_for_affinity(gaussians)
    optimizer = torch.optim.Adam(
        [{"params": [gaussians.get_sam3_affinity], "lr": float(getattr(args, "sam3_affinity_lr", 0.0025)), "name": "sam3_affinity"}],
        lr=0.0,
        eps=1.0e-15,
    )
    if len(optimizer.param_groups) != 1 or optimizer.param_groups[0]["name"] != "sam3_affinity":
        raise RuntimeError("SAM3 affinity optimizer must contain only the affinity field")
    xyz_before = gaussians.get_canonical_xyz.detach().clone()
    rot_before = gaussians.get_rotation.detach().clone()
    root = _sam3_mask_root_for_train(dataset, args)
    cache = SAM3MaskCache(root, device=gaussians.get_xyz.device)
    output_dir = Path(output_dir)
    debug_interval = int(getattr(args, "sam3_affinity_debug_interval", 500))
    generator = torch.Generator(device=gaussians.get_xyz.device)
    generator.manual_seed(20_260_904 + int(start_step))
    ema: dict[str, float] = {}
    stack = []
    last_stats = {}
    used_updates = 0
    progress = tqdm(
        range(1, int(getattr(args, "sam3_affinity_iters", 0)) + 1),
        desc=f"Phase 0D SAM3 affinity {stage_name}",
        unit="iter",
    )
    for local_iter in progress:
        if not stack:
            stack = cameras.copy()
        camera = stack.pop(randint(0, len(stack) - 1))
        h = int(camera.original_image.shape[-2])
        w = int(camera.original_image.shape[-1])
        foreground = _load_mask_for_view(camera, dataset, args, (h, w), device=gaussians.get_xyz.device)
        if foreground is None:
            raise RuntimeError("SAM3 affinity requires existing foreground masks; none resolved for camera")
        pack = cache.get(camera)
        prepared = prepare_foreground_sam3_masks(pack, foreground, args)
        pixel_idx = sample_foreground_pixels(
            prepared.foreground_mask,
            prepared.valid_mask,
            int(getattr(args, "sam3_affinity_pixels_per_iter", 768)),
            generator=generator,
        )
        if pixel_idx.shape[0] < 2 or prepared.masks.shape[0] == 0:
            continue
        H = gather_membership_matrix(prepared.masks, prepared.weights, pixel_idx)
        valid_rows = torch.linalg.norm(H, dim=-1) > 1.0e-8
        pixel_idx = pixel_idx[valid_rows]
        H = H[valid_rows]
        if H.shape[0] < 2:
            continue
        render_pkg = render_sam3_affinity_field(camera, gaussians, pipe)
        y = pixel_idx[:, 0].long()
        x = pixel_idx[:, 1].long()
        Z = render_pkg["feature"][:, y, x].T.contiguous()
        finite = torch.isfinite(Z).all(-1) & (render_pkg["alpha"][y, x] > 1.0e-6)
        H = H[finite]
        pixel_idx = pixel_idx[finite]
        Z = Z[finite]
        if Z.shape[0] < 2:
            continue
        target = compute_target_affinity(H)
        predicted = compute_rendered_affinity(Z)
        valid_pairs = torch.isfinite(target) & torch.isfinite(predicted)
        pairs = sample_balanced_pairs(target, valid_pairs, args, predicted_affinity=predicted)
        if pairs.numel() == 0:
            continue
        loss_affinity, stats = sam3_affinity_loss(target, predicted, pairs, args)
        z_norm = torch.linalg.norm(render_pkg["feature"], dim=0).clamp(0.0, 1.0)
        norm_weight = (
            prepared.foreground_mask.float()
            * prepared.valid_mask.float()
            * render_pkg["alpha"].detach().clamp(0.0, 1.0)
        )
        loss_norm = ((1.0 - z_norm) * norm_weight).sum() / norm_weight.sum().clamp_min(1.0e-6)
        loss = loss_affinity + float(getattr(args, "sam3_affinity_norm_weight", 0.1)) * loss_norm
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        used_updates += 1
        stats.update(
            {
                "loss_total": float(loss.detach()),
                "loss_affinity": float(loss_affinity.detach()),
                "loss_norm": float(loss_norm.detach()),
                "valid_foreground_fraction": float(prepared.valid_mask.float().mean()),
                "accepted_mask_count": int(prepared.masks.shape[0]),
                "feature_norm": float(z_norm.detach()[prepared.valid_mask].mean()) if bool(prepared.valid_mask.any()) else 0.0,
                "updates": int(used_updates),
            }
        )
        last_stats = stats
        global_step = int(start_step) + int(local_iter)
        if writer is not None:
            for key, value in stats.items():
                writer.add_scalar(f"sam3/{key}", value, global_step)
                writer.add_scalar(f"sam3/{stage_name}/{key}", value, global_step)
        if wb_run is not None and local_iter % int(getattr(args, "wandb_log_interval", 1)) == 0:
            payload = {"iteration": global_step}
            for key, value in stats.items():
                payload[f"sam3/{key}"] = value
                payload[f"sam3/{stage_name}/{key}"] = value
            wb_run.log(
                payload,
                step=global_step,
            )
        for key in ("loss_total", "loss_affinity", "loss_norm", "positive_pred_affinity", "negative_pred_affinity"):
            ema[key] = 0.4 * float(stats[key]) + 0.6 * ema.get(key, float(stats[key]))
        if local_iter % 10 == 0:
            progress.set_postfix(
                loss=f"{ema.get('loss_total', 0.0):.4f}",
                aff=f"{ema.get('loss_affinity', 0.0):.4f}",
                norm=f"{ema.get('loss_norm', 0.0):.4f}",
                masks=int(stats["accepted_mask_count"]),
            )
        if debug_interval > 0 and (local_iter == 1 or local_iter % debug_interval == 0 or local_iter == int(getattr(args, "sam3_affinity_iters", 0))):
            _write_debug(output_dir, stage_name, camera, prepared, render_pkg, pixel_idx)

    if used_updates == 0:
        raise RuntimeError(f"SAM3 affinity training made zero updates; check mask cache under {root}")
    if not torch.allclose(xyz_before, gaussians.get_canonical_xyz.detach(), atol=0.0, rtol=0.0):
        raise RuntimeError("SAM3 affinity training changed Gaussian XYZ")
    if not torch.allclose(rot_before, gaussians.get_rotation.detach(), atol=0.0, rtol=0.0):
        raise RuntimeError("SAM3 affinity training changed Gaussian rotations")
    field = gaussians.get_sam3_affinity.detach()
    sidecar_dir = output_dir / "point_cloud" / f"iteration_{int(getattr(args, 'static_train_iters', 0)) if stage_name == 'canonical' else int(getattr(args, 'endpoint_static_iters', 0))}"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    config = _field_training_config(args)
    gaussians._sam3_affinity_training_config = config
    gaussians._sam3_affinity_frame_idx = None if frame_idx is None else int(frame_idx)
    gaussians.save_sam3_affinity_field(sidecar_dir / "sam3_affinity_field.pt", training_config=config, frame_idx=frame_idx)
    graph = build_local_affinity_graph(
        gaussians.get_canonical_xyz.detach().cpu(),
        field.cpu(),
        int(getattr(args, "sam3_affinity_graph_knn", 16)),
    )
    torch.save(graph, sidecar_dir / "sam3_affinity_graph.pt")
    _write_gaussian_html(
        Path(output_dir) / "sam3_debug" / str(stage_name) / "sam3_affinity_gaussians.html",
        gaussians.get_canonical_xyz,
        field,
        int(getattr(args, "sam3_affinity_html_max_points", 5000)),
    )
    summary = {
        "schema_version": 1,
        "stage": str(stage_name),
        "frame_idx": None if frame_idx is None else int(frame_idx),
        "sam3_mask_root": str(root),
        "updates": int(used_updates),
        "gaussians": int(field.shape[0]),
        "dim": int(field.shape[1]),
        "sidecar": str(sidecar_dir / "sam3_affinity_field.pt"),
        "graph": str(sidecar_dir / "sam3_affinity_graph.pt"),
        "last_stats": last_stats,
        "training_config": config,
        "ground_truth_used": False,
        "no_cross_frame_raw_cosine": True,
    }
    summary_path = Path(output_dir) / "sam3_debug" / str(stage_name) / "sam3_affinity_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary
