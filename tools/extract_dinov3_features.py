#!/usr/bin/env python3
"""Extract dense DINOv3 patch-token feature maps for dynamic RGB sequences.

Default backend is Hugging Face Transformers because the official HF DINOv3
repos store Transformers-format weights, not the torch-hub .pth filename.

Expected input:
  ROOT/rgb/cam_000/frame_000000.png
  ROOT/rgb/cam_001/frame_000000.png

Output:
  ROOT/dinov3/cam_000/frame_000000.pt

Each .pt contains:
  {
    "features": Tensor[C,Hf,Wf],
    "image_path": str,
    "model_name": str,
    "backend": str,
    "image_size": int,
    "patch_size": int,
  }
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Global DINOv3 defaults.
# Override without touching code:
#   export DINOV3_HF_LOCAL_DIR=/path/to/facebook/dinov3-vitl16-pretrain-lvd1689m
#   export DINOV3_HF_MODEL=facebook/dinov3-vitl16-pretrain-lvd1689m
DINOv3_HF_MODEL = os.environ.get(
    "DINOV3_HF_MODEL",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
)
DINOv3_HF_LOCAL_DIR = os.environ.get(
    "DINOV3_HF_LOCAL_DIR",
    str(Path.home() / ".cache" / "huggingface" / "dinov3-vitl16-pretrain-lvd1689m"),
)
DINOv3_BACKEND = os.environ.get("DINOV3_BACKEND", "transformers")
DINOv3_IMAGE_SIZE = int(os.environ.get("DINOV3_IMAGE_SIZE", "512"))
DINOv3_TORCHHUB_REPO = os.environ.get("DINOV3_TORCHHUB_REPO", "submodules/dinov3")
DINOv3_TORCHHUB_MODEL_NAME = os.environ.get("DINOV3_TORCHHUB_MODEL_NAME", "dinov3_vitl16")


def iter_images(sequence_root: Path, rgb_dir: str) -> Iterable[Path]:
    root = sequence_root / rgb_dir
    if not root.exists():
        raise FileNotFoundError(f"RGB root not found: {root}")
    imgs = []
    for ext in IMAGE_EXTS:
        imgs.extend(root.glob(f"cam_*/*{ext}"))
        imgs.extend(root.glob(f"*{ext}"))
    seen = set()
    for p in sorted(imgs):
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        yield p


def output_path_for(image_path: Path, sequence_root: Path, rgb_dir: str, out_dir: str) -> Path:
    rgb_root = sequence_root / rgb_dir
    rel = image_path.relative_to(rgb_root)
    return (sequence_root / out_dir / rel).with_suffix(".pt")


def preprocess_image(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    # np.array creates a writable copy and avoids torch.frombuffer warnings.
    arr = torch.from_numpy(np.array(img, dtype=np.uint8))
    arr = arr.view(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = (arr - mean) / std
    return x.unsqueeze(0).to(device, non_blocking=True)


def load_transformers_model(hf_model: str, local_dir: str, device: torch.device):
    try:
        from transformers import AutoModel
    except Exception as exc:
        raise RuntimeError(
            "Transformers backend requested but transformers is not installed.\n"
            "Run: python -m pip install -U 'transformers>=4.56.0' huggingface_hub safetensors"
        ) from exc

    local_dir = str(local_dir or "").strip()
    if local_dir:
        local_path = Path(local_dir).expanduser()
        if local_path.exists():
            model_src = str(local_path)
            print(f"[DINO] Loading Transformers model from local checkpoint dir: {model_src}")
        else:
            model_src = hf_model
            print(f"[DINO] Local checkpoint dir does not exist: {local_path}")
            print(f"[DINO] Falling back to HF model id/cache: {model_src}")
    else:
        model_src = hf_model
        print(f"[DINO] Loading Transformers model from HF id/cache: {model_src}")

    model = AutoModel.from_pretrained(model_src, trust_remote_code=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_patch_size(model, fallback=16) -> int:
    cfg = getattr(model, "config", None)
    ps = getattr(cfg, "patch_size", None)
    if isinstance(ps, int):
        return int(ps)
    pe = getattr(model, "patch_embed", None)
    ps = getattr(pe, "patch_size", None)
    if isinstance(ps, tuple):
        return int(ps[0])
    if isinstance(ps, int):
        return int(ps)
    return int(fallback)


def extract_transformers_patch_tokens(model, x: torch.Tensor, patch_size: int) -> torch.Tensor:
    with torch.no_grad():
        out = model(pixel_values=x)

    tokens = None
    if hasattr(out, "last_hidden_state"):
        tokens = out.last_hidden_state
    elif isinstance(out, dict):
        for key in ("last_hidden_state", "hidden_states", "x_norm_patchtokens", "patch_tokens"):
            if key in out and torch.is_tensor(out[key]):
                tokens = out[key]
                break
    elif torch.is_tensor(out):
        tokens = out

    if tokens is None or tokens.ndim != 3 or tokens.shape[0] != 1:
        raise RuntimeError(f"Could not get [1,N,C] tokens from Transformers output: {type(out)!r}")

    n = int(tokens.shape[1])

    h_floor = x.shape[-2] // patch_size
    w_floor = x.shape[-1] // patch_size
    grid_n = h_floor * w_floor

    # DINOv3 Transformers models may return prefix tokens before patch tokens:
    # CLS plus register tokens. Example for ViT-L/16 at 512:
    #   32*32 patch tokens = 1024
    #   output tokens       = 1029
    #   prefix tokens       = 5
    # Drop any small prefix block when the suffix is exactly the image patch grid.
    if n > grid_n and (n - grid_n) <= 32:
        prefix = n - grid_n
        print(f"[DINO] Dropping {prefix} prefix tokens; using {grid_n} patch tokens")
        tokens = tokens[:, prefix:, :]
        n = grid_n
    elif n == grid_n + 1:
        tokens = tokens[:, 1:, :]
        n = grid_n
    elif n > 1 and int(round(math.sqrt(n - 1))) ** 2 == n - 1:
        tokens = tokens[:, 1:, :]
        n -= 1

    side = int(round(math.sqrt(n)))
    if side * side == n:
        h = w = side
    elif h_floor * w_floor == n:
        h, w = h_floor, w_floor
    else:
        raise RuntimeError(
            f"Cannot reshape {n} tokens to a grid. "
            f"patch_size={patch_size}, image={tuple(x.shape[-2:])}, floor_grid={h_floor}x{w_floor}"
        )

    feat = tokens[0].reshape(h, w, tokens.shape[-1]).permute(2, 0, 1).contiguous().float().cpu()
    feat = F.normalize(feat, dim=0)
    return feat


def find_cached_checkpoint(model_name: str, explicit_cache: str = "") -> Path | None:
    roots = []
    if explicit_cache:
        roots.append(Path(explicit_cache).expanduser())
    torch_home = os.environ.get("TORCH_HOME", "")
    if torch_home:
        roots.append(Path(torch_home).expanduser() / "hub" / "checkpoints")
    roots.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints")
    patterns = [
        f"{model_name}*.pth",
        f"{model_name}*.pt",
        f"*{model_name}*.pth",
        f"*{model_name}*.pt",
        "*dinov3*.pth",
        "*dinov3*.pt",
        "*dino*.pth",
        "*dino*.pt",
    ]
    for r in roots:
        if not r.exists():
            continue
        for pat in patterns:
            hits = sorted(r.glob(pat))
            if hits:
                return hits[0]
    return None


def clean_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "teacher", "student", "backbone"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)!r}")
    cleaned = {}
    for k, v in ckpt.items():
        nk = str(k)
        for pref in ("module.", "backbone.", "teacher.", "student.", "model.", "encoder."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v
    return cleaned


def instantiate_torchhub_model(repo: str, model_name: str, pretrained: bool):
    repo_path = Path(repo).expanduser()
    if repo_path.exists():
        print(f"[DINO] Loading local torch.hub repo: {repo_path} model={model_name} pretrained={pretrained}")
        return torch.hub.load(str(repo_path), model_name, source="local", pretrained=pretrained)
    print(f"[DINO] Loading remote torch.hub repo: {repo} model={model_name} pretrained={pretrained}")
    return torch.hub.load(repo, model_name, pretrained=pretrained)


def load_torchhub_model(repo: str, model_name: str, weights: str, device: torch.device, allow_download: bool, cache_dir: str):
    ckpt_path = None
    if weights:
        ckpt_path = Path(weights).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--weights does not exist: {ckpt_path}")
    else:
        ckpt_path = find_cached_checkpoint(model_name, cache_dir)

    if ckpt_path is not None:
        model = instantiate_torchhub_model(repo, model_name, pretrained=False)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        missing, unexpected = model.load_state_dict(clean_state_dict(ckpt), strict=False)
        print(f"[DINO] Loaded local torchhub weights={ckpt_path}")
        print(f"[DINO] state_dict load: missing={len(missing)}, unexpected={len(unexpected)}")
    elif allow_download:
        model = instantiate_torchhub_model(repo, model_name, pretrained=True)
    else:
        raise RuntimeError(
            "Torchhub backend requested but no local .pth checkpoint was found. "
            "Use --backend transformers for HF repos, or pass --weights /path/to/torchhub_checkpoint.pth."
        )

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_torchhub_patch_tokens(model, x: torch.Tensor, patch_size: int) -> torch.Tensor:
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            out = model.forward_features(x)
        else:
            out = model(x)

    if isinstance(out, dict):
        tokens = None
        for key in ("x_norm_patchtokens", "x_norm_patch_tokens", "patch_tokens", "patchtokens", "tokens"):
            if key in out:
                tokens = out[key]
                break
        if tokens is None:
            for v in out.values():
                if torch.is_tensor(v) and v.ndim == 3:
                    tokens = v
                    break
        if tokens is None:
            raise RuntimeError(f"Could not find patch tokens in model output keys={list(out.keys())}")
    elif torch.is_tensor(out) and out.ndim == 3:
        tokens = out
    else:
        raise RuntimeError(f"Unsupported torchhub output: {type(out)!r}")

    n = int(tokens.shape[1])
    side = int(round(math.sqrt(n)))
    if side * side == n:
        h = w = side
    else:
        h = x.shape[-2] // patch_size
        w = x.shape[-1] // patch_size
        if h * w != n:
            raise RuntimeError(f"Cannot reshape {n} patch tokens to grid")
    feat = tokens[0].reshape(h, w, tokens.shape[-1]).permute(2, 0, 1).contiguous().float().cpu()
    feat = F.normalize(feat, dim=0)
    return feat



def make_saved_feature_target(
    patch_feat: torch.Tensor,
    original_hw: tuple[int, int],
    full_resolution: bool,
    feature_dim: int,
    fp16: bool,
) -> torch.Tensor:
    """Prepare feature tensor for saving.

    patch_feat is [C,h,w]. If full_resolution=True, this returns [D,H,W] where
    D defaults to 32 unless explicitly specified. This is the sane dense target
    for training: whole-image supervision without storing 1024 channels per pixel.
    """
    if patch_feat.ndim != 3:
        raise RuntimeError(f"Expected patch_feat [C,h,w], got {tuple(patch_feat.shape)}")

    C = int(patch_feat.shape[0])
    if feature_dim > 0:
        D = min(int(feature_dim), C)
        patch_feat = patch_feat[:D]
    elif full_resolution:
        # Full-res 1024-channel ViT-L targets are enormous. Default to 32
        # channels, matching --dino_feature_field_dim 32.
        D = min(32, C)
        print(f"[DINO] --full_resolution set with no --feature_dim; saving first {D}/{C} channels")
        patch_feat = patch_feat[:D]

    patch_feat = F.normalize(patch_feat.float(), dim=0)

    if full_resolution:
        H, W = int(original_hw[0]), int(original_hw[1])
        patch_feat = F.interpolate(
            patch_feat.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0]
        patch_feat = F.normalize(patch_feat, dim=0)

    if fp16:
        patch_feat = patch_feat.half()
    return patch_feat.contiguous()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence_root", required=True, type=str)
    ap.add_argument("--rgb_dir", default="rgb", type=str)
    ap.add_argument("--out_dir", default="dinov3", type=str)
    ap.add_argument("--backend", default=DINOv3_BACKEND, choices=["transformers", "torchhub"],
                    help="Use HF Transformers weights by default. Torchhub needs a .pth checkpoint.")
    ap.add_argument("--hf_model", default=DINOv3_HF_MODEL, type=str)
    ap.add_argument("--hf_local_dir", default=DINOv3_HF_LOCAL_DIR, type=str,
                    help="Local snapshot dir from `hf download ... --local-dir DIR`. Defaults to the global DINOv3_HF_LOCAL_DIR.")
    ap.add_argument("--repo", default=DINOv3_TORCHHUB_REPO, type=str,
                    help="Torchhub backend local repo path.")
    ap.add_argument("--model_name", default=DINOv3_TORCHHUB_MODEL_NAME, type=str,
                    help="Torchhub backend model function name.")
    ap.add_argument("--weights", default="", type=str,
                    help="Torchhub backend local .pth checkpoint.")
    ap.add_argument("--cache_dir", default="", type=str)
    ap.add_argument("--allow_download", action="store_true",
                    help="Torchhub backend only: explicitly allow online .pth download.")
    ap.add_argument("--image_size", default=DINOv3_IMAGE_SIZE, type=int)
    ap.add_argument("--full_resolution", action="store_true",
                    help="Save whole-image dense feature targets [D,H,W] by upsampling patch tokens to original RGB resolution.")
    ap.add_argument("--feature_dim", default=0, type=int,
                    help="Number of feature channels to save. Recommended 32 for full-resolution targets. 0 keeps all patch channels unless --full_resolution.")
    ap.add_argument("--fp32", action="store_true",
                    help="Save features as float32. Default saves float16 to keep full-image targets small.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_images", default=-1, type=int)
    args = ap.parse_args()

    sequence_root = Path(args.sequence_root).expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backend == "transformers":
        model = load_transformers_model(args.hf_model, args.hf_local_dir, device)
        patch_size = get_patch_size(model, fallback=16)
        extract_fn = extract_transformers_patch_tokens
        model_label = args.hf_model
    else:
        model = load_torchhub_model(args.repo, args.model_name, args.weights, device, args.allow_download, args.cache_dir)
        patch_size = get_patch_size(model, fallback=16)
        extract_fn = extract_torchhub_patch_tokens
        model_label = args.model_name

    images = list(iter_images(sequence_root, args.rgb_dir))
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found under {sequence_root / args.rgb_dir}")

    print(
        f"[DINO] Extracting {len(images)} images, backend={args.backend}, "
        f"image_size={args.image_size}, patch_size={patch_size}, device={device}"
    )

    done = 0
    skipped = 0
    for i, img_path in enumerate(images, 1):
        out_path = output_path_for(img_path, sequence_root, args.rgb_dir, args.out_dir)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(img_path) as im:
            original_hw = (int(im.height), int(im.width))
        x = preprocess_image(img_path, args.image_size, device)
        patch_feat = extract_fn(model, x, patch_size=patch_size)
        feat = make_saved_feature_target(
            patch_feat=patch_feat,
            original_hw=original_hw,
            full_resolution=bool(args.full_resolution),
            feature_dim=int(args.feature_dim),
            fp16=not bool(args.fp32),
        )
        torch.save(
            {
                "features": feat,
                "image_path": str(img_path),
                "model_name": model_label,
                "backend": args.backend,
                "image_size": int(args.image_size),
                "patch_size": int(patch_size),
                "original_hw": tuple(original_hw),
                "full_resolution": bool(args.full_resolution),
                "feature_dim": int(feat.shape[0]),
                "dtype": str(feat.dtype),
            },
            out_path,
        )
        done += 1
        if i == 1 or i % 25 == 0:
            print(f"[DINO] {i}/{len(images)} saved={done} skipped={skipped} last={out_path}")
    print(f"[DINO] Done. saved={done}, skipped={skipped}, out={sequence_root / args.out_dir}")


if __name__ == "__main__":
    main()
