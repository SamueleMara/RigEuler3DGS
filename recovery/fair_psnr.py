"""PSNR exactly as FreeArtGS evaluates it, so our numbers and FreeArtGS's are comparable.

compute_psnr / compute_psnr_masked are line-for-line copies of FreeArtGS evaluate/psnr_utils.py
(_compute_psnr, _compute_psnr_masked); score_view applies them the way render_gaussians_and_psnr_gsplat does
(masked PSNR with the un-eroded object mask, full-image PSNR if the mask is empty; the scene value is the mean of
per-image PSNRs). Both methods are scored on 8-bit renders on a white background against the SAME recorded files
<source>/rgb/<cam>/<stem>.png and <source>/mask/<cam>/<stem>.png. Our float renders are quantized the way
splatfacto-art writes its eval_results PNGs: (clip(rgb, 0, 1) * 255).astype(uint8).
FreeArtGS's evaluate_freeartgs.py additionally searches a per-frame joint offset against the GT image before
scoring (refine_motion_delta); that is NOT applied here, to either method.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image

DEFINITION = ("FreeArtGS evaluate/psnr_utils.py _compute_psnr_masked: MSE over mask pixels x 3 channels of 8-bit "
              "images / 255 (mask > 127, not eroded), 10 log10(1 / MSE), 99 dB if MSE <= 1e-12; full-image PSNR "
              "when the mask is empty; scene value = mean of per-image PSNRs; white background; no per-frame joint "
              "refinement")


def compute_psnr(img: np.ndarray, ref: np.ndarray) -> float:
    a = img.astype(np.float32)
    b = ref.astype(np.float32)
    if a.max() > 1.5 or b.max() > 1.5:
        a = a / 255.0
        b = b / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def compute_psnr_masked(img: np.ndarray, ref: np.ndarray, mask: np.ndarray):
    a = img.astype(np.float32)
    b = ref.astype(np.float32)
    if a.max() > 1.5 or b.max() > 1.5:
        a = a / 255.0
        b = b / 255.0
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype != np.bool_:
        mask = mask > 0.5 if mask.max() <= 1.0 else mask > 127
    mask = mask.astype(np.bool_)
    valid = int(mask.sum())
    if valid < 1:
        return None
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    mse = float(((a - b) ** 2)[mask3].mean())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def quantize_like_splatfacto(rgb_hw3_float: np.ndarray) -> np.ndarray:
    return (np.clip(rgb_hw3_float, 0, 1) * 255).astype("uint8")


def _find(d: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        if (d / f"{stem}{ext}").is_file():
            return d / f"{stem}{ext}"
    raise FileNotFoundError(f"{d}/{stem}.png")


def load_reference(source, camera: str, stem: str):
    """(rgb uint8 HxWx3, mask HxW, sha256 of the two files) from <source>/rgb|mask/<camera>/<stem>.*"""
    rp, mp = _find(Path(source) / "rgb" / camera, stem), _find(Path(source) / "mask" / camera, stem)
    gt = np.asarray(Image.open(rp))
    if gt.ndim == 2:
        gt = np.repeat(gt[..., None], 3, axis=2)
    gt = gt[..., :3]
    mask = np.asarray(Image.open(mp))
    if mask.ndim == 3 and mask.shape[2] > 1:
        mask = mask[:, :, 0]
    h = hashlib.sha256(rp.read_bytes() + mp.read_bytes()).hexdigest()
    return gt, mask, h


def score_view(pred_u8: np.ndarray, gt_u8: np.ndarray, mask: np.ndarray) -> dict:
    if pred_u8.shape != gt_u8.shape or mask.shape != gt_u8.shape[:2]:
        raise ValueError(f"shape mismatch: render {pred_u8.shape}, reference {gt_u8.shape}, mask {mask.shape}")
    p = compute_psnr_masked(pred_u8, gt_u8, mask)
    return {"psnr": compute_psnr(pred_u8, gt_u8) if p is None else p, "mask_empty": p is None}


def summarize(rows: list) -> dict:
    """rows: dicts with camera, frame, psnr, mask_empty."""
    if not rows:
        return {"psnr": None, "frames": 0}
    cams = sorted({r["camera"] for r in rows})
    return {"psnr": float(np.mean([r["psnr"] for r in rows])), "frames": len(rows),
            "psnr_median": float(np.median([r["psnr"] for r in rows])),
            "per_camera": {c: float(np.mean([r["psnr"] for r in rows if r["camera"] == c])) for c in cams},
            "mask_empty_views": int(sum(r["mask_empty"] for r in rows)), "definition": DEFINITION}
