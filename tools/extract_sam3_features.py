#!/usr/bin/env python3
"""Extract cached dense SAM3 image embeddings for RigEuler3DGS.

Default SAM3 code path is this repo's vendored checkout:
``submodules/sam3``. The trainer consumes the saved tensors under ROOT/sam3.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def _find_cached_sam3_checkpoint() -> Path | None:
    cache = Path("/home/samuelemara/.cache/huggingface/hub")
    candidates = [
        *cache.glob("models--facebook--sam3.1/snapshots/*/sam3.1_multiplex.pt"),
        *cache.glob("models--facebook--sam3/snapshots/*/sam3.pt"),
    ]
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size > 100_000_000:
                return path
        except OSError:
            continue
    return None


def _image_table(root: Path, rgb_dir: str):
    table = {}
    for cam_dir in sorted((root / rgb_dir).glob("cam_*")):
        if not cam_dir.is_dir():
            continue
        frames = {}
        for path in sorted(cam_dir.glob("frame_*.png")) + sorted(cam_dir.glob("frame_*.jpg")) + sorted(cam_dir.glob("frame_*.jpeg")):
            try:
                frame = int(path.stem.split("_")[-1])
            except ValueError:
                continue
            frames[frame] = path
        if frames:
            table[cam_dir.name] = frames
    if not table:
        raise FileNotFoundError(f"No RGB frames found under {root / rgb_dir}/cam_*/frame_*.png")
    return table


def _resolve_frames(table, requested):
    available = sorted(set.intersection(*(set(frames) for frames in table.values())))
    if not available:
        raise RuntimeError("No synchronized frame ids are shared by all cameras")
    if not requested:
        return [available[0], available[-1]]
    out = []
    for item in requested:
        text = str(item).strip().upper()
        if text in ("LAST", "END", "FINAL"):
            frame = available[-1]
        elif text in ("FIRST", "START"):
            frame = available[0]
        else:
            frame = int(item)
        if frame not in available:
            raise RuntimeError(f"Requested frame {frame} is not synchronized across all cameras")
        out.append(frame)
    return sorted(set(out))


def _sam3_autocast(device: str, amp_dtype: str):
    if not str(device).startswith("cuda") or amp_dtype == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_model(args, *, enable_segmentation=False, enable_inst_interactivity=False):
    repo = Path(args.sam3_repo).expanduser().resolve() if args.sam3_repo else (
        Path(__file__).resolve().parents[1] / "submodules" / "sam3"
    )
    if repo.exists():
        sys.path.insert(0, str(repo))
    from sam3.model_builder import build_sam3_image_model

    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else _find_cached_sam3_checkpoint()
    )
    if checkpoint is None and not args.allow_download:
        raise FileNotFoundError(
            "No complete cached SAM3 checkpoint found. Expected one of:\n"
            "  /home/samuelemara/.cache/huggingface/hub/models--facebook--sam3.1/snapshots/*/sam3.1_multiplex.pt\n"
            "  /home/samuelemara/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt\n"
            "Pass --checkpoint explicitly, or pass --allow-download to fetch from Hugging Face."
        )
    if checkpoint is not None:
        print(f"[SAM3] using checkpoint: {checkpoint}")
    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=str(checkpoint) if checkpoint is not None else None,
        load_from_HF=bool(args.allow_download and checkpoint is None),
        enable_segmentation=bool(enable_segmentation),
        enable_inst_interactivity=bool(enable_inst_interactivity),
    )
    model.eval()
    return model


def _collect_feature_candidates(obj, candidates, prefix="root"):
    if torch.is_tensor(obj):
        tensor = obj.detach()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim == 3:
            # Accept [C,H,W] or [H,W,C]. Prefer real spatial maps, not tokens.
            if tensor.shape[0] >= 8 and tensor.shape[1] >= 4 and tensor.shape[2] >= 4:
                candidates.append((prefix, tensor.float()))
            elif tensor.shape[-1] >= 8 and tensor.shape[0] >= 4 and tensor.shape[1] >= 4:
                candidates.append((prefix, tensor.permute(2, 0, 1).contiguous().float()))
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            _collect_feature_candidates(value, candidates, f"{prefix}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _collect_feature_candidates(value, candidates, f"{prefix}[{index}]")


@torch.inference_mode()
def extract_sam3_dense_embedding(
    model,
    image: Image.Image,
    *,
    device="cuda",
    resolution=1008,
    amp_dtype="bfloat16",
):
    """Return one dense SAM3 embedding tensor [C,Hf,Wf].

    All SAM3-internal assumptions are isolated here. The preferred path uses
    Sam3Processor.set_image(), then selects the highest-resolution tensor from
    the returned backbone output.
    """
    from sam3.model.sam3_image_processor import Sam3Processor

    processor = Sam3Processor(model, resolution=int(resolution), device=device)
    with _sam3_autocast(device, amp_dtype):
        state = processor.set_image(image)
    backbone_out = state.get("backbone_out", state)
    candidates = []
    _collect_feature_candidates(backbone_out, candidates)
    if not candidates:
        raise RuntimeError(
            "SAM3 adapter could not find a dense [C,H,W] image tensor in backbone_out. "
            "Update extract_sam3_dense_embedding() for this SAM3 version."
        )
    # Use the largest spatial map; for FPN outputs this keeps the most local structure.
    name, feature = max(candidates, key=lambda item: int(item[1].shape[-2]) * int(item[1].shape[-1]))
    return name, feature.detach().cpu()


def _fit_projection(sample_rows, dim):
    sample = torch.cat(sample_rows, dim=0).float()
    mean = sample.mean(0)
    centered = sample - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[: int(dim)].contiguous()
    return {"mean": mean.contiguous(), "components": components}


def _apply_projection(feature, projection):
    c, h, w = feature.shape
    flat = feature.permute(1, 2, 0).reshape(-1, c).float()
    projected = (flat - projection["mean"]) @ projection["components"].T
    projected = projected.reshape(h, w, -1).permute(2, 0, 1).contiguous()
    return projected


def _save_feature(path, feature, *, dtype):
    feature = F.normalize(feature.float(), dim=0)
    if dtype == "float16":
        feature = feature.half()
    torch.save({"features": feature.cpu(), "normalization": "channel_l2"}, path)


def main():
    parser = argparse.ArgumentParser(description="Extract SAM3 dense feature maps")
    parser.add_argument("--root", required=True, help="Dataset root containing rgb/cam_*/frame_*.png")
    parser.add_argument("--rgb-dir", default="rgb")
    parser.add_argument("--frames", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sam3-repo",
        default="",
        help="SAM3 source checkout. Defaults to this repo's submodules/sam3.",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face download if no local checkpoint is found.")
    parser.add_argument("--sam3-amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--feature-dim", type=int, default=32, help="Shared PCA output dimension. Use 0 to save raw SAM3 channels.")
    parser.add_argument("--pca-samples-per-image", type=int, default=2048)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    table = _image_table(root, args.rgb_dir)
    frames = _resolve_frames(table, args.frames)
    output.mkdir(parents=True, exist_ok=True)
    model = _load_model(args)

    image_items = [(cam, frame, table[cam][frame]) for cam in sorted(table) for frame in frames]
    projection = None
    feature_source = None
    raw_dim = None
    if int(args.feature_dim) > 0:
        samples = []
        generator = torch.Generator().manual_seed(12345)
        for _, _, path in tqdm(image_items, desc="SAM3 PCA sample", unit="image"):
            image = Image.open(path).convert("RGB")
            feature_source, feature = extract_sam3_dense_embedding(
                model,
                image,
                device=args.device,
                resolution=args.resolution,
                amp_dtype=args.sam3_amp_dtype,
            )
            raw_dim = int(feature.shape[0])
            flat = feature.permute(1, 2, 0).reshape(-1, feature.shape[0])
            take = min(int(args.pca_samples_per_image), int(flat.shape[0]))
            ids = torch.randperm(flat.shape[0], generator=generator)[:take]
            samples.append(flat[ids].float())
        projection = _fit_projection(samples, int(args.feature_dim))
        torch.save(projection, output / "shared_projection.pt")

    for cam, frame, path in tqdm(image_items, desc="SAM3 extract", unit="image"):
        image = Image.open(path).convert("RGB")
        feature_source, feature = extract_sam3_dense_embedding(
            model,
            image,
            device=args.device,
            resolution=args.resolution,
            amp_dtype=args.sam3_amp_dtype,
        )
        raw_dim = int(feature.shape[0])
        saved = _apply_projection(feature, projection) if projection is not None else feature
        cam_out = output / cam
        cam_out.mkdir(parents=True, exist_ok=True)
        _save_feature(cam_out / f"frame_{int(frame):06d}.pt", saved, dtype=args.dtype)

    metadata = {
        "schema_version": 1,
        "model": "facebook/sam3.1" if not args.checkpoint else "local_checkpoint",
        "checkpoint": str(args.checkpoint or "huggingface:facebook/sam3.1"),
        "feature_source": feature_source,
        "feature_dim_raw": int(raw_dim or 0),
        "feature_dim_saved": int(args.feature_dim) if int(args.feature_dim) > 0 else int(raw_dim or 0),
        "normalization": "per-pixel channel L2 after shared PCA" if projection is not None else "per-pixel channel L2 raw",
        "input_resolution": int(args.resolution),
        "sam3_amp_dtype": args.sam3_amp_dtype,
        "output_stride": None,
        "frames": frames,
        "cameras": sorted(table),
        "ground_truth_used": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[SAM3] wrote {len(image_items)} feature maps to {output}")


if __name__ == "__main__":
    main()
