#!/usr/bin/env python3
"""
Tiny scene-specific video diffusion test for dynamic/static motion cues.

Goal
----
Train a very small noise-conditioned temporal denoiser directly on one short
video (or several synchronized camera folders) in a few minutes, then extract
coarse temporal features and local feature correspondences.

This is deliberately NOT a full video generator. It is a compact experiment to
answer: does scene-specific diffusion-style denoising learn temporal features
that separate static pixels, moving pixels, and low-confidence/occluded areas?

Dependencies
------------
- torch
- numpy
- Pillow

No torchvision, diffusers, xformers, OpenCV, or external models are required.

Expected input layouts
----------------------
1) One camera / one video:
   input_dir/
      frame_000000.png
      frame_000001.png
      ...

2) Multi-camera root (auto-detected):
   input_dir/
      cam_000/
         frame_000000.png
         ...
      cam_001/
         frame_000000.png
         ...

The script treats each image-containing folder as an independent temporal
sequence and samples clips only within that folder.

Typical run
-----------
python tiny_video_diffusion_test.py \
    --input_dir /path/to/rgb \
    --output_dir /path/to/output/tiny_diffusion_test \
    --resolution 160 \
    --clip_len 4 \
    --diffusion_steps 32 \
    --base_channels 32 \
    --max_steps 6000 \
    --max_minutes 7 \
    --analyze

Outputs
-------
- checkpoint.pt
- train_log.csv
- config.json
- diagnostics/<sequence>/features.pt
- diagnostics/<sequence>/motion_gap1.npz
- diagnostics/<sequence>/avg_feature_motion.png
- diagnostics/<sequence>/avg_match_confidence.png
- diagnostics/<sequence>/avg_same_pixel_similarity.png

Interpretation of diagnostic images
-----------------------------------
- avg_feature_motion.png:
    Bright = learned feature correspondence moves spatially.
    With fixed cameras, this is a candidate dynamic cue.
- avg_match_confidence.png:
    Bright = a strong local match exists. Dark regions are ambiguous and often
    include occlusions, disocclusions, textureless areas, or failures.
- avg_same_pixel_similarity.png:
    Bright = the same pixel location remains feature-consistent over time.

For a moving/static decision in the real 3DGS pipeline, the next step is to
compare the learned feature displacement to the STATIC projection predicted by
3DGS geometry/cameras, rather than using feature motion magnitude alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_sequences(input_dir: Path, min_frames: int) -> Dict[str, List[Path]]:
    """Find all folders containing >= min_frames images.

    If input_dir itself contains enough images, it is used as one sequence and
    nested folders are ignored. Otherwise, all qualifying nested folders are
    returned.
    """
    input_dir = input_dir.resolve()
    direct = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )
    if len(direct) >= min_frames:
        return {input_dir.name: direct}

    sequences: Dict[str, List[Path]] = {}
    for root, _, files in os.walk(input_dir):
        root_p = Path(root)
        imgs = sorted(
            [root_p / f for f in files if Path(f).suffix.lower() in IMAGE_EXTS],
            key=natural_key,
        )
        if len(imgs) >= min_frames:
            rel = root_p.relative_to(input_dir)
            name = str(rel).replace(os.sep, "__") if str(rel) != "." else input_dir.name
            sequences[name] = imgs

    if not sequences:
        raise FileNotFoundError(
            f"No image sequence with at least {min_frames} frames found under {input_dir}"
        )
    return sequences


def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Center-crop to square, then resize. Deliberately simple and dependency-free."""
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.Resampling.BILINEAR)


def load_sequences(
    sequence_paths: Dict[str, List[Path]],
    resolution: int,
    max_frames: int,
    pin_memory: bool,
) -> Dict[str, torch.Tensor]:
    """Load images as uint8 CPU tensors [N, 3, H, W]."""
    loaded: Dict[str, torch.Tensor] = {}
    for name, paths in sequence_paths.items():
        if max_frames > 0:
            paths = paths[:max_frames]
        frames = []
        for p in paths:
            with Image.open(p) as im:
                im = resize_center_crop(im, resolution)
                arr = np.asarray(im, dtype=np.uint8).copy()
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))
        seq = torch.stack(frames, dim=0).contiguous()
        if pin_memory and torch.cuda.is_available():
            seq = seq.pin_memory()
        loaded[name] = seq
        print(f"[data] {name}: {len(frames)} frames @ {resolution}x{resolution}")
    return loaded


def sample_clip(
    sequences: Dict[str, torch.Tensor],
    clip_len: int,
    temporal_stride_max: int,
    device: torch.device,
) -> Tuple[str, int, int, torch.Tensor]:
    names = list(sequences.keys())
    for _ in range(100):
        name = random.choice(names)
        seq = sequences[name]
        stride = random.randint(1, temporal_stride_max)
        needed = 1 + (clip_len - 1) * stride
        if len(seq) >= needed:
            start = random.randint(0, len(seq) - needed)
            idx = torch.arange(start, start + needed, stride)
            clip = seq[idx]
            # [T,C,H,W] -> [1,C,T,H,W], [-1,1]
            clip = clip.to(device=device, dtype=torch.float32, non_blocking=True)
            clip = clip / 127.5 - 1.0
            clip = clip.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
            return name, start, stride, clip
    raise RuntimeError("Could not sample a valid clip. Check clip_len / temporal_stride_max.")


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half < 1:
            return t.float().unsqueeze(-1)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class ResBlock3D(nn.Module):
    """Factorized spatial + temporal residual block.

    Uses (1,3,3) spatial convolution followed by (3,1,1) temporal convolution,
    which is substantially cheaper than dense 3x3x3 convolutions.
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        g1 = min(8, in_ch)
        while in_ch % g1 != 0:
            g1 -= 1
        g2 = min(8, out_ch)
        while out_ch % g2 != 0:
            g2 -= 1

        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.spatial = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.temporal = nn.Conv3d(
            out_ch,
            out_ch,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=max(1, out_ch // 8),
        )
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.time_proj = nn.Linear(t_dim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.spatial(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(temb))[:, :, None, None, None]
        h = self.temporal(F.silu(h))
        h = self.conv2(F.silu(self.norm2(h)))
        return x if isinstance(self.skip, nn.Identity) and False else self.skip(x) + h


class TinyTemporalUNet(nn.Module):
    def __init__(self, base: int = 32, time_dim: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.stem = nn.Conv3d(3, base, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.enc1 = ResBlock3D(base, base, time_dim)
        self.down1 = nn.Conv3d(base, base * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.enc2 = ResBlock3D(base * 2, base * 2, time_dim)
        self.down2 = nn.Conv3d(base * 2, base * 3, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.mid1 = ResBlock3D(base * 3, base * 3, time_dim)
        self.mid2 = ResBlock3D(base * 3, base * 3, time_dim)

        self.up2 = nn.ConvTranspose3d(
            base * 3,
            base * 2,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.dec2 = ResBlock3D(base * 4, base * 2, time_dim)
        self.up1 = nn.ConvTranspose3d(
            base * 2,
            base,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.dec1 = ResBlock3D(base * 2, base, time_dim)

        g = min(8, base)
        while base % g != 0:
            g -= 1
        self.out_norm = nn.GroupNorm(g, base)
        self.out = nn.Conv3d(base, 3, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_features: bool = False,
    ):
        temb = self.time_embed(t)

        x0 = self.stem(x)
        e1 = self.enc1(x0, temb)
        e2 = self.enc2(self.down1(e1), temb)
        m = self.mid2(self.mid1(self.down2(e2), temb), temb)

        d2 = self.up2(m)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), temb)
        d1 = self.up1(d2)
        feat = self.dec1(torch.cat([d1, e1], dim=1), temb)

        eps = self.out(F.silu(self.out_norm(feat)))
        if return_features:
            return eps, feat
        return eps


def make_beta_schedule(steps: int, device: torch.device):
    # Gentle linear schedule is adequate for this tiny experiment.
    betas = torch.linspace(1e-4, 0.02, steps, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def q_sample(x0: torch.Tensor, t: torch.Tensor, alpha_bars: torch.Tensor, noise: torch.Tensor):
    a = alpha_bars[t].view(-1, 1, 1, 1, 1)
    return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise


@dataclass
class TrainConfig:
    input_dir: str
    output_dir: str
    resolution: int
    clip_len: int
    temporal_stride_max: int
    diffusion_steps: int
    base_channels: int
    max_steps: int
    max_minutes: float
    lr: float
    weight_decay: float
    seed: int
    max_frames: int
    temporal_dropout_prob: float
    amp: bool


def maybe_temporal_dropout(x: torch.Tensor, probability: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Occasionally corrupt one full frame to force temporal context use.

    This is an auxiliary self-supervised corruption, not part of the formal
    forward diffusion process. The target remains the known diffusion noise.
    It is intentionally lightweight for the proof-of-concept.
    """
    if probability <= 0.0 or random.random() >= probability:
        return x, torch.zeros((x.shape[0], 1, x.shape[2], 1, 1), device=x.device, dtype=x.dtype)

    out = x.clone()
    B, _, T, _, _ = x.shape
    mask = torch.zeros((B, 1, T, 1, 1), device=x.device, dtype=x.dtype)
    for b in range(B):
        tt = random.randrange(T)
        mask[b, 0, tt] = 1.0
        # Strongly corrupt only that frame. Temporal neighbors are untouched.
        out[b, :, tt] = torch.randn_like(out[b, :, tt])
    return out, mask


def train(args, sequences: Dict[str, torch.Tensor], device: torch.device) -> Path:
    model = TinyTemporalUNet(base=args.base_channels).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[model] parameters: {params/1e6:.2f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    use_amp = bool(args.amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    _, _, alpha_bars = make_beta_schedule(args.diffusion_steps, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"

    model.train()
    start_wall = time.monotonic()
    last_print = start_wall
    ema_loss = None
    steps_done = 0

    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "wall_seconds", "loss", "ema_loss", "seq", "start", "stride"])

        for step in range(1, args.max_steps + 1):
            elapsed = time.monotonic() - start_wall
            if elapsed >= args.max_minutes * 60.0 and step > 100:
                print(f"[train] hard wall-clock cutoff reached at {elapsed/60:.2f} min")
                break

            seq_name, clip_start, stride, x0 = sample_clip(
                sequences,
                args.clip_len,
                args.temporal_stride_max,
                device,
            )

            t = torch.randint(0, args.diffusion_steps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, alpha_bars, noise)
            xt, _drop_mask = maybe_temporal_dropout(xt, args.temporal_dropout_prob)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(xt, t)
                    loss = F.mse_loss(pred, noise)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xt, t)
                loss = F.mse_loss(pred, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss_val = float(loss.detach().item())
            ema_loss = loss_val if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_val
            steps_done = step

            if step == 1 or step % args.log_every == 0:
                elapsed = time.monotonic() - start_wall
                writer.writerow(
                    [step, f"{elapsed:.3f}", f"{loss_val:.8f}", f"{ema_loss:.8f}", seq_name, clip_start, stride]
                )
                f.flush()

            if step == 1 or step % args.print_every == 0:
                now = time.monotonic()
                elapsed = now - start_wall
                rate = step / max(elapsed, 1e-6)
                remaining_by_time = max(0.0, args.max_minutes * 60 - elapsed)
                print(
                    f"[train] step {step:5d} | loss {loss_val:.5f} | ema {ema_loss:.5f} | "
                    f"{rate:.2f} step/s | {elapsed/60:.2f} min | time left <= {remaining_by_time/60:.2f} min"
                )
                last_print = now

    elapsed = time.monotonic() - start_wall
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "base_channels": args.base_channels,
            "diffusion_steps": args.diffusion_steps,
            "resolution": args.resolution,
            "clip_len": args.clip_len,
            "steps_done": steps_done,
            "train_seconds": elapsed,
        },
        ckpt_path,
    )
    print(f"[train] saved {ckpt_path}")
    print(f"[train] finished {steps_done} steps in {elapsed/60:.2f} min")
    return ckpt_path


def load_model(checkpoint: Path, device: torch.device) -> Tuple[TinyTemporalUNet, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = TinyTemporalUNet(base=int(ckpt["base_channels"]))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, ckpt


@torch.inference_mode()
def extract_sequence_features(
    model: TinyTemporalUNet,
    seq_u8: torch.Tensor,
    clip_len: int,
    diffusion_steps: int,
    pool: int,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    """Extract overlapping temporal features and average contributions per frame.

    Returns CPU float32 tensor [N,C,H/pool,W/pool].
    """
    N, _, H, W = seq_u8.shape
    if N < clip_len:
        raise ValueError("Sequence shorter than clip_len")

    accum = None
    counts = torch.zeros(N, dtype=torch.float32)
    eval_t = torch.zeros((1,), dtype=torch.long, device=device)
    use_amp = bool(amp and device.type == "cuda")

    for start in range(0, N - clip_len + 1):
        clip = seq_u8[start : start + clip_len].to(device=device, dtype=torch.float32, non_blocking=True)
        clip = clip / 127.5 - 1.0
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1,C,T,H,W]

        # Near-clean evaluation. No random noise is injected so features are deterministic.
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, feat = model(clip, eval_t, return_features=True)
        else:
            _, feat = model(clip, eval_t, return_features=True)

        # [1,C,T,H,W] -> [T,C,h,w]
        feat = feat[0].permute(1, 0, 2, 3).float()
        if pool > 1:
            feat = F.avg_pool2d(feat, kernel_size=pool, stride=pool)
        feat = F.normalize(feat, dim=1)
        feat_cpu = feat.cpu()

        if accum is None:
            accum = torch.zeros((N,) + feat_cpu.shape[1:], dtype=torch.float32)

        accum[start : start + clip_len] += feat_cpu
        counts[start : start + clip_len] += 1.0

    assert accum is not None
    accum /= counts[:, None, None, None].clamp_min(1.0)
    accum = F.normalize(accum, dim=1)
    return accum


@torch.inference_mode()
def local_feature_match(
    f0: torch.Tensor,
    f1: torch.Tensor,
    radius: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Local nearest-neighbor feature matching.

    f0,f1: CPU [C,H,W], expected L2-normalized in channel dimension.

    Returns dx, dy, best_score, same_pixel_score as float32 numpy [H,W].
    dx/dy are measured in feature-grid pixels.
    """
    a = f0.to(device, non_blocking=True).unsqueeze(0).float()
    b = f1.to(device, non_blocking=True).unsqueeze(0).float()
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)

    _, C, H, W = a.shape
    k = 2 * radius + 1
    patches = F.unfold(b, kernel_size=k, padding=radius)  # [1,C*k*k,H*W]
    patches = patches.view(1, C, k * k, H * W)
    a_flat = a.view(1, C, 1, H * W)
    sims = (a_flat * patches).sum(dim=1)[0]  # [k*k,H*W]

    best_score, best_idx = sims.max(dim=0)
    center = radius * k + radius
    same = sims[center]

    dy = torch.div(best_idx, k, rounding_mode="floor") - radius
    dx = (best_idx % k) - radius

    return (
        dx.view(H, W).float().cpu().numpy(),
        dy.view(H, W).float().cpu().numpy(),
        best_score.view(H, W).float().cpu().numpy(),
        same.view(H, W).float().cpu().numpy(),
    )


def save_grayscale(arr: np.ndarray, path: Path, robust: bool = True, invert: bool = False) -> None:
    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        img = np.zeros_like(a, dtype=np.uint8)
    else:
        vals = a[finite]
        if robust and vals.size > 20:
            lo, hi = np.percentile(vals, [2.0, 98.0])
        else:
            lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo + 1e-12:
            norm = np.zeros_like(a, dtype=np.float32)
        else:
            norm = (a - lo) / (hi - lo)
        norm = np.clip(norm, 0.0, 1.0)
        if invert:
            norm = 1.0 - norm
        img = (norm * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def analyze(
    args,
    sequences: Dict[str, torch.Tensor],
    checkpoint: Path,
    device: torch.device,
):
    model, ckpt = load_model(checkpoint, device)
    clip_len = int(ckpt.get("clip_len", args.clip_len))
    diffusion_steps = int(ckpt.get("diffusion_steps", args.diffusion_steps))

    root = Path(args.output_dir) / "diagnostics"
    root.mkdir(parents=True, exist_ok=True)

    for seq_name, seq in sequences.items():
        seq_dir = root / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        print(f"[analyze] extracting features for {seq_name} ...")
        feat = extract_sequence_features(
            model,
            seq,
            clip_len=clip_len,
            diffusion_steps=diffusion_steps,
            pool=args.feature_pool,
            device=device,
            amp=args.amp,
        )

        # Save compact half-precision coarse features for later 3DGS integration.
        torch.save(
            {
                "features": feat.half(),
                "feature_pool": args.feature_pool,
                "resolution": args.resolution,
                "sequence": seq_name,
            },
            seq_dir / "features.pt",
        )

        for gap in range(1, args.max_gap + 1):
            if len(feat) <= gap:
                continue
            dxs, dys, scores, sames = [], [], [], []
            t0 = time.monotonic()
            for i in range(0, len(feat) - gap):
                dx, dy, score, same = local_feature_match(
                    feat[i], feat[i + gap], args.search_radius, device
                )
                dxs.append(dx)
                dys.append(dy)
                scores.append(score)
                sames.append(same)

            dxs = np.stack(dxs)
            dys = np.stack(dys)
            scores = np.stack(scores)
            sames = np.stack(sames)
            mag = np.sqrt(dxs * dxs + dys * dys)

            np.savez_compressed(
                seq_dir / f"motion_gap{gap}.npz",
                dx=dxs.astype(np.float16),
                dy=dys.astype(np.float16),
                best_score=scores.astype(np.float16),
                same_pixel_score=sames.astype(np.float16),
                feature_pool=np.int32(args.feature_pool),
                search_radius=np.int32(args.search_radius),
                gap=np.int32(gap),
            )

            if gap == 1:
                save_grayscale(mag.mean(axis=0), seq_dir / "avg_feature_motion.png")
                save_grayscale(scores.mean(axis=0), seq_dir / "avg_match_confidence.png")
                save_grayscale(sames.mean(axis=0), seq_dir / "avg_same_pixel_similarity.png")
                # Ambiguity map: low best match confidence. This is not an occlusion GT mask,
                # but it is a useful place to inspect occlusion/disocclusion behavior.
                save_grayscale(scores.mean(axis=0), seq_dir / "avg_ambiguity.png", invert=True)

            print(
                f"[analyze] {seq_name} gap={gap}: {len(dxs)} pairs in "
                f"{time.monotonic()-t0:.1f}s"
            )

    print(f"[analyze] diagnostics saved under {root}")


def parse_args():
    p = argparse.ArgumentParser(description="Tiny scene-specific video diffusion test")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resolution", type=int, default=160)
    p.add_argument("--clip_len", type=int, default=4)
    p.add_argument("--temporal_stride_max", type=int, default=2)
    p.add_argument("--diffusion_steps", type=int, default=32)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--max_minutes", type=float, default=7.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_frames", type=int, default=60, help="0 = all frames per sequence")
    p.add_argument(
        "--temporal_dropout_prob",
        type=float,
        default=0.10,
        help="Probability of replacing one noisy frame by pure noise to force temporal context",
    )
    p.add_argument("--no_amp", action="store_true", help="Disable CUDA mixed precision")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda", help="Execution device. Default: cuda (fail loudly if unavailable)")
    p.add_argument("--resume", type=str, default="", help="Analyze an existing checkpoint instead of training")
    p.add_argument("--analyze", action="store_true", help="Run feature/correspondence diagnostics after training")
    p.add_argument("--feature_pool", type=int, default=4, help="Spatial pooling for saved features")
    p.add_argument("--search_radius", type=int, default=5, help="Local matching radius in pooled feature pixels")
    p.add_argument("--max_gap", type=int, default=1, help="Analyze temporal gaps 1..max_gap")
    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    args.amp = not args.no_amp

    if args.clip_len < 2:
        raise ValueError("clip_len must be >= 2")
    if args.resolution % 4 != 0:
        raise ValueError("resolution must be divisible by 4")
    if args.feature_pool < 1:
        raise ValueError("feature_pool must be >= 1")
    if args.search_radius < 0:
        raise ValueError("search_radius must be >= 0")

    seed_everything(args.seed)
    print(f"[torch] version={torch.__version__} | built_cuda={torch.version.cuda} | cuda_available={torch.cuda.is_available()} | devices={torch.cuda.device_count()}")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Check `nvidia-smi`, `which python`, `python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\"`, "
                "and CUDA_VISIBLE_DEVICES before training."
            )
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_paths = find_sequences(input_dir, min_frames=args.clip_len)
    print(f"[data] found {len(seq_paths)} sequence(s)")
    sequences = load_sequences(
        seq_paths,
        resolution=args.resolution,
        max_frames=args.max_frames,
        pin_memory=True,
    )

    cfg = TrainConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        resolution=args.resolution,
        clip_len=args.clip_len,
        temporal_stride_max=args.temporal_stride_max,
        diffusion_steps=args.diffusion_steps,
        base_channels=args.base_channels,
        max_steps=args.max_steps,
        max_minutes=args.max_minutes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_frames=args.max_frames,
        temporal_dropout_prob=args.temporal_dropout_prob,
        amp=args.amp,
    )
    with (output_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    if args.resume:
        checkpoint = Path(args.resume)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    else:
        checkpoint = train(args, sequences, device)

    if args.analyze or args.resume:
        analyze(args, sequences, checkpoint, device)
    else:
        print("[done] training complete. Re-run with --resume checkpoint.pt to analyze features.")


if __name__ == "__main__":
    main()
