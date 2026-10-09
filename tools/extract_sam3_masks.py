#!/usr/bin/env python3
"""Cache foreground-clipped SAM3 overlapping masks for affinity-field training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from extract_sam3_features import (
    _image_table,
    _load_model,
    _resolve_frames,
    _sam3_autocast,
)


def _load_foreground(root: Path, mask_dir: str, cam: str, frame: int, hw):
    h, w = hw
    for ext in (".png", ".jpg", ".jpeg"):
        path = root / mask_dir / cam / f"frame_{int(frame):06d}{ext}"
        if path.exists():
            arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            if arr.shape != (h, w):
                arr = np.array(
                    Image.fromarray((arr * 255).astype(np.uint8)).resize(
                        (w, h), Image.Resampling.NEAREST
                    ),
                    dtype=np.float32,
                ) / 255.0
            return arr > 0.5
    raise FileNotFoundError(
        f"Foreground mask missing for {cam} frame {int(frame):06d} under {root / mask_dir}"
    )


def _prompt_grid(fg: np.ndarray, points_per_side: int, max_prompts: int):
    ys, xs = np.nonzero(fg)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    px = np.linspace(x0, x1, max(1, int(points_per_side)))
    py = np.linspace(y0, y1, max(1, int(points_per_side)))
    points = []
    for y in py:
        for x in px:
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if 0 <= yi < fg.shape[0] and 0 <= xi < fg.shape[1] and fg[yi, xi]:
                points.append((float(xi), float(yi)))
    if len(points) > int(max_prompts):
        keep = np.linspace(0, len(points) - 1, int(max_prompts)).round().astype(np.int64)
        points = [points[int(i)] for i in keep]
    return np.asarray(points, dtype=np.float32)


def _dedupe_by_iou(masks: torch.Tensor, scores: torch.Tensor, threshold: float):
    keep = []
    binary = masks > 0.5
    order = torch.argsort(scores, descending=True)
    for idx in order.tolist():
        candidate = binary[idx]
        if not keep:
            keep.append(idx)
            continue
        prev = binary[torch.tensor(keep)]
        inter = (prev & candidate).flatten(1).sum(1).float()
        union = (prev | candidate).flatten(1).sum(1).float().clamp_min(1.0)
        if float((inter / union).max()) < float(threshold):
            keep.append(idx)
    return torch.tensor(keep, dtype=torch.long)


def _overlay(image: Image.Image, masks: torch.Tensor, path: Path):
    rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    if masks.numel() == 0:
        Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
        return
    rng = np.random.default_rng(12345)
    for mask in masks.float().numpy():
        color = rng.random(3)
        alpha = 0.35 * np.clip(mask[..., None], 0.0, 1.0)
        rgb = rgb * (1.0 - alpha) + color * alpha
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


@torch.inference_mode()
def _extract_image_masks(model, processor, image: Image.Image, fg: np.ndarray, args):
    if getattr(model, "inst_interactive_predictor", None) is None:
        raise RuntimeError("SAM3 model was not built with instance interactivity enabled")
    points = _prompt_grid(fg, args.points_per_side, args.max_prompts)
    if points.shape[0] == 0:
        h, w = fg.shape
        return torch.empty((0, h, w), dtype=torch.bool), torch.empty(0), torch.empty((0, 2))

    with _sam3_autocast(args.device, args.sam3_amp_dtype):
        inference_state = processor.set_image(image)

    masks_out = []
    scores_out = []
    prompt_out = []
    for xy in points:
        with _sam3_autocast(args.device, args.sam3_amp_dtype):
            masks, scores, _ = model.predict_inst(
                inference_state,
                point_coords=xy[None],
                point_labels=np.ones((1,), dtype=np.int32),
                multimask_output=True,
                return_logits=bool(args.soft_masks),
                normalize_coords=True,
            )
        mask_t = torch.from_numpy(np.asarray(masks)).float()
        if bool(args.soft_masks):
            mask_t = mask_t.sigmoid()
        score_t = torch.from_numpy(np.asarray(scores)).float().reshape(-1)
        for mi in range(mask_t.shape[0]):
            clipped = mask_t[mi].clamp(0.0, 1.0)
            clipped = clipped * torch.from_numpy(fg).float()
            fg_frac = float((clipped > 0.5).sum()) / max(1, int(fg.sum()))
            if fg_frac < float(args.min_fg_fraction) or fg_frac > float(args.max_fg_fraction):
                continue
            if int((clipped > 0.5).sum()) < int(args.min_area_px):
                continue
            masks_out.append(clipped)
            scores_out.append(float(score_t[min(mi, score_t.numel() - 1)]))
            prompt_out.append(tuple(float(v) for v in xy))
    if not masks_out:
        h, w = fg.shape
        return torch.empty((0, h, w), dtype=torch.bool), torch.empty(0), torch.empty((0, 2))
    masks_t = torch.stack(masks_out)
    scores_t = torch.tensor(scores_out, dtype=torch.float32)
    prompt_t = torch.tensor(prompt_out, dtype=torch.float32)
    keep = _dedupe_by_iou(masks_t, scores_t, float(args.dedupe_iou))
    masks_t = masks_t.index_select(0, keep)
    scores_t = scores_t.index_select(0, keep)
    prompt_t = prompt_t.index_select(0, keep)
    if not bool(args.soft_masks):
        masks_t = masks_t > 0.5
    return masks_t.cpu(), scores_t.cpu(), prompt_t.cpu()


def main():
    parser = argparse.ArgumentParser(description="Extract foreground-clipped SAM3 mask proposals")
    parser.add_argument("--root", required=True, help="Dataset root containing rgb/ and mask/")
    parser.add_argument("--rgb-dir", default="rgb")
    parser.add_argument("--mask-dir", default="mask")
    parser.add_argument("--frames", nargs="*", default=[])
    parser.add_argument("--output", default="", help="Defaults to ROOT/sam3_masks")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam3-repo", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--sam3-amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--points-per-side", type=int, default=12)
    parser.add_argument("--max-prompts", type=int, default=96)
    parser.add_argument("--min-fg-fraction", type=float, default=0.001)
    parser.add_argument("--max-fg-fraction", type=float, default=0.95)
    parser.add_argument("--min-area-px", type=int, default=16)
    parser.add_argument("--dedupe-iou", type=float, default=0.90)
    parser.add_argument("--soft-masks", action="store_true")
    parser.add_argument("--debug-overlays", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else root / "sam3_masks"
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "SAM3 mask extraction was requested on CUDA, but torch.cuda.is_available() is false. "
            "Run this command from the GPU-enabled rigeuler3dgs shell."
        )
    table = _image_table(root, args.rgb_dir)
    frames = _resolve_frames(table, args.frames)
    output.mkdir(parents=True, exist_ok=True)
    model = _load_model(args, enable_segmentation=True, enable_inst_interactivity=True)
    from sam3.model.sam3_image_processor import Sam3Processor

    processor = Sam3Processor(model, resolution=int(args.resolution), device=args.device)

    items = [(cam, frame, table[cam][frame]) for cam in sorted(table) for frame in frames]
    metadata = {
        "schema_version": 1,
        "source": "sam3_interactive_point_prompts",
        "frames": [int(f) for f in frames],
        "cameras": sorted(table),
        "root": str(root),
        "foreground_mask_dir": str(root / args.mask_dir),
        "points_per_side": int(args.points_per_side),
        "max_prompts": int(args.max_prompts),
        "min_fg_fraction": float(args.min_fg_fraction),
        "max_fg_fraction": float(args.max_fg_fraction),
        "dedupe_iou": float(args.dedupe_iou),
        "soft_masks": bool(args.soft_masks),
        "input_resolution": int(args.resolution),
        "ground_truth_used": False,
    }
    total_masks = 0
    for cam, frame, path in tqdm(items, desc="SAM3 masks", unit="image"):
        image = Image.open(path).convert("RGB")
        w, h = image.size
        fg = _load_foreground(root, args.mask_dir, cam, frame, (h, w))
        masks, scores, prompt_xy = _extract_image_masks(model, processor, image, fg, args)
        fg_area = max(1, int(fg.sum()))
        mask_binary = masks.float() > 0.5
        fg_fraction = mask_binary.flatten(1).sum(1).float() / float(fg_area) if masks.numel() else torch.empty(0)
        cam_out = output / cam
        cam_out.mkdir(parents=True, exist_ok=True)
        payload = {
            "masks": masks.half() if bool(args.soft_masks) else masks.bool(),
            "scores": scores.float(),
            "foreground_fraction": fg_fraction.float(),
            "image_hw": (int(h), int(w)),
            "mask_hw": (int(h), int(w)),
            "prompt_xy": prompt_xy.float(),
            "ground_truth_used": False,
        }
        torch.save(payload, cam_out / f"frame_{int(frame):06d}.pt")
        total_masks += int(masks.shape[0])
        if args.debug_overlays:
            debug = output / "sam3_mask_debug" / cam
            debug.mkdir(parents=True, exist_ok=True)
            Image.fromarray((fg.astype(np.uint8) * 255)).save(debug / f"frame_{int(frame):06d}_foreground.png")
            _overlay(image, masks.float(), debug / f"frame_{int(frame):06d}_accepted_masks_overlay.png")
    metadata["total_cached_masks"] = int(total_masks)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[SAM3 masks] wrote {len(items)} files / {total_masks} masks to {output}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
