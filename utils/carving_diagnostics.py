"""Logging-only diagnostics for the transparency-carving A/B.

Nothing here touches the training objective. Every render runs under
``torch.no_grad()``, nothing is passed to ``backward()``, and no densification
statistic is updated. Enabling ``--carving_debug`` must change what is written
to disk and nothing about what is learned.

Two quantities are measured.

**Background opacity leakage.** The rasteriser does not expose the accumulated
alpha A(p), but it composites front to back, so rendering every Gaussian as
constant white over a black background gives

    render(p) = sum_i T_i(p) a_i(p) * 1 + (1 - A(p)) * 0 = A(p)

exactly. No new CUDA kernel, and in particular no estimate of A from the
black-background training image -- that would confuse "no Gaussian here" with
"a black Gaussian here", which is the very thing under test.

**Anisotropy.** The ratio of largest to smallest activated scale per Gaussian.
Reported, never thresholded: this file may not prune, clamp or filter anything.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

# Order is fixed and part of the fingerprint: two runs are comparable only if
# the same tensors are hashed in the same sequence.
FINGERPRINT_TENSORS = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
)

VIEW_FIELDS = (
    "iteration", "camera_name", "branch", "L1", "SSIM", "training_total_loss",
    "E_background_camera", "foreground_opacity_mean", "background_pixels",
)

GLOBAL_FIELDS = (
    "iteration", "branch", "E_background_global", "foreground_opacity_global",
    "cameras_used", "cameras_without_background", "N_gaussians",
    "anisotropy_median", "anisotropy_p99", "anisotropy_max",
    "max_scale", "bbox_diagonal", "max_scale_over_bbox_diag",
    "nonfinite_parameters", "L1", "SSIM", "training_total_loss",
)


@torch.no_grad()
def render_projected_alpha(viewpoint_cam, gaussians, pipe):
    """Accumulated projected opacity A(p) in [0,1], shape [1,H,W].

    White Gaussians over a black background. `separate_sh=False` because the
    override-colour branch leaves the split DC term undefined, and
    `use_trained_exp=False` because the learned exposure transform would scale
    the very quantity being measured.
    """
    from gaussian_renderer import render

    ones = torch.ones_like(gaussians.get_xyz)
    black = torch.zeros(3, device=ones.device, dtype=ones.dtype)
    package = render(
        viewpoint_cam, gaussians, pipe, black,
        override_color=ones,
        separate_sh=False,
        use_trained_exp=False,
    )
    return package["render"][0:1].clamp(0.0, 1.0)


def _camera_mask(viewpoint_cam, shape):
    """Foreground coverage for this camera at the rendered resolution."""
    mask = getattr(viewpoint_cam, "object_alpha", None)
    if mask is None:
        return None
    mask = mask.to(device="cuda", dtype=torch.float32)
    if tuple(mask.shape[-2:]) != tuple(shape[-2:]):
        mask = torch.nn.functional.interpolate(
            mask[None], size=tuple(shape[-2:]), mode="nearest")[0]
    return mask.clamp(0.0, 1.0)


@torch.no_grad()
def camera_leakage(viewpoint_cam, gaussians, pipe):
    """Numerator and denominator of E_background for one camera.

    Returned unreduced on purpose. The global figure is a pixel-weighted sum
    over all views -- sum(1-M)A / sum(1-M) -- not the mean of per-view ratios,
    which would weight a camera that barely sees the object as heavily as one
    that fills the frame.
    """
    alpha = render_projected_alpha(viewpoint_cam, gaussians, pipe)
    mask = _camera_mask(viewpoint_cam, alpha.shape)
    if mask is None:
        return None
    background = 1.0 - mask
    bg_weight = float(background.sum())
    fg_weight = float(mask.sum())
    return {
        "bg_numerator": float((background * alpha).sum()),
        "bg_denominator": bg_weight,
        "fg_numerator": float((mask * alpha).sum()),
        "fg_denominator": fg_weight,
        # A camera with no background pixels has no leakage to report; it is
        # recorded as N/A and left out of the global sums rather than counted
        # as a zero it never earned.
        "E_background_camera": (float((background * alpha).sum()) / bg_weight
                                if bg_weight > 0 else float("nan")),
        "foreground_opacity_mean": (float((mask * alpha).sum()) / fg_weight
                                    if fg_weight > 0 else float("nan")),
        "background_pixels": bg_weight,
        "alpha": alpha,
        "mask": mask,
    }


@torch.no_grad()
def anisotropy_stats(gaussians):
    """Shape statistics of the current primitives. Reported, never applied."""
    out = {
        "N_gaussians": 0, "anisotropy_median": float("nan"),
        "anisotropy_p99": float("nan"), "anisotropy_max": float("nan"),
        "max_scale": float("nan"), "bbox_diagonal": float("nan"),
        "max_scale_over_bbox_diag": float("nan"), "nonfinite_parameters": 0,
    }
    xyz = gaussians.get_xyz
    if xyz is None or xyz.numel() == 0:
        return out
    scales = gaussians.get_scaling.detach().float()
    out["N_gaussians"] = int(scales.shape[0])

    # Counted and reported, never dropped: a NaN scale is a training failure,
    # and silently filtering it would hide the failure behind clean numbers.
    nonfinite = int((~torch.isfinite(scales)).any(dim=-1).sum())
    nonfinite += int((~torch.isfinite(xyz)).any(dim=-1).sum())
    out["nonfinite_parameters"] = nonfinite

    finite = torch.isfinite(scales).all(dim=-1)
    usable = scales[finite]
    if usable.numel() == 0:
        return out
    smallest = usable.amin(dim=-1)
    largest = usable.amax(dim=-1)
    positive = smallest > 0
    if positive.any():
        ratio = largest[positive] / smallest[positive]
        out["anisotropy_median"] = float(torch.quantile(ratio.double(), 0.5))
        out["anisotropy_p99"] = float(torch.quantile(ratio.double(), 0.99))
        out["anisotropy_max"] = float(ratio.max())
    out["max_scale"] = float(largest.max())

    points = xyz.detach().float()
    points = points[torch.isfinite(points).all(dim=-1)]
    if points.shape[0] >= 2:
        diagonal = float((points.amax(dim=0) - points.amin(dim=0)).norm())
        out["bbox_diagonal"] = diagonal
        if diagonal > 0:
            out["max_scale_over_bbox_diag"] = out["max_scale"] / diagonal
    return out


@torch.no_grad()
def initial_state_fingerprint(gaussians):
    """SHA256 per initial parameter tensor, plus a combined digest.

    Hashes the raw bytes together with shape and dtype. Two runs that disagree
    here are not an A/B of one intervention, whatever their flags say.
    """
    per_tensor, combined = {}, hashlib.sha256()
    for name in FINGERPRINT_TENSORS:
        tensor = getattr(gaussians, name, None)
        if tensor is None or not torch.is_tensor(tensor):
            per_tensor[name] = None
            combined.update(f"{name}:absent".encode())
            continue
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        header = f"{name}:{tuple(array.shape)}:{array.dtype.str}".encode()
        digest = hashlib.sha256()
        digest.update(header)
        digest.update(array.tobytes())
        per_tensor[name] = digest.hexdigest()
        combined.update(header)
        combined.update(array.tobytes())
    return {"per_tensor": per_tensor, "combined": combined.hexdigest()}


class CarvingDebugWriter:
    """CSV/JSON sink for the diagnostics. Creates nothing until enabled."""

    def __init__(self, model_path, branch: str, enabled: bool):
        self.enabled = bool(enabled)
        self.branch = branch
        self.root = Path(model_path) / "carving_debug"
        if not self.enabled:
            return
        (self.root / "alpha_previews").mkdir(parents=True, exist_ok=True)
        self._views = self._open("metrics.csv", VIEW_FIELDS)
        self._globals = self._open("metrics_global.csv", GLOBAL_FIELDS)
        self._trace = self._open("camera_trace.csv", ("iteration", "image_name"))

    def _open(self, name, fields):
        handle = open(self.root / name, "w", newline="")
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        handle.flush()
        return handle, writer

    def trace(self, iteration: int, image_name: str) -> None:
        """One row per training iteration. No rendering, no extra work."""
        if not self.enabled:
            return
        handle, writer = self._trace
        writer.writerow({"iteration": int(iteration), "image_name": str(image_name)})
        handle.flush()

    def view_row(self, row: dict) -> None:
        if not self.enabled:
            return
        handle, writer = self._views
        writer.writerow({k: row.get(k, "") for k in VIEW_FIELDS})
        handle.flush()

    def global_row(self, row: dict) -> None:
        if not self.enabled:
            return
        handle, writer = self._globals
        writer.writerow({k: row.get(k, "") for k in GLOBAL_FIELDS})
        handle.flush()

    def initial_state(self, payload: dict) -> None:
        if not self.enabled:
            return
        (self.root / "initial_state.json").write_text(json.dumps(payload, indent=1))

    def preview(self, iteration: int, camera_name: str, alpha, mask) -> None:
        """One alpha/mask pair per debug checkpoint, not per training step."""
        if not self.enabled:
            return
        try:
            from PIL import Image
            safe = str(camera_name).replace("/", "_")
            for tag, tensor in (("alpha", alpha), ("mask", mask)):
                array = (tensor[0].detach().cpu().numpy().clip(0, 1) * 255)
                Image.fromarray(array.astype(np.uint8)).save(
                    self.root / "alpha_previews" / f"{tag}_{safe}_{int(iteration):06d}.png")
        except Exception:
            pass  # previews are a convenience; never fail a run over one

    def close(self) -> None:
        if not self.enabled:
            return
        for handle, _ in (self._views, self._globals, self._trace):
            try:
                handle.close()
            except Exception:
                pass


@torch.no_grad()
def evaluate(cameras, gaussians, pipe, iteration, branch, writer,
             photometric=None, previews=0):
    """Full-dataset diagnostics at one checkpoint.

    `cameras` is iterated in the order given and never shuffled: shuffling here
    would consume the same RNG that picks the training camera and desynchronise
    the two runs this is meant to compare.
    """
    photometric = photometric or {}
    bg_num = bg_den = fg_num = fg_den = 0.0
    used = skipped = 0
    for index, camera in enumerate(cameras):
        stats = camera_leakage(camera, gaussians, pipe)
        if stats is None:
            continue
        if stats["bg_denominator"] > 0:
            bg_num += stats["bg_numerator"]
            bg_den += stats["bg_denominator"]
            used += 1
        else:
            skipped += 1
        fg_num += stats["fg_numerator"]
        fg_den += stats["fg_denominator"]
        writer.view_row({
            "iteration": iteration, "camera_name": camera.image_name,
            "branch": branch,
            "L1": photometric.get("L1", ""), "SSIM": photometric.get("SSIM", ""),
            "training_total_loss": photometric.get("total", ""),
            "E_background_camera": stats["E_background_camera"],
            "foreground_opacity_mean": stats["foreground_opacity_mean"],
            "background_pixels": stats["background_pixels"],
        })
        if index < previews:
            writer.preview(iteration, camera.image_name, stats["alpha"], stats["mask"])

    shape = anisotropy_stats(gaussians)
    row = {
        "iteration": iteration, "branch": branch,
        # "N/A", not a fabricated zero: no background pixels anywhere means the
        # metric is undefined, not satisfied.
        "E_background_global": (bg_num / bg_den) if bg_den > 0 else "N/A",
        "foreground_opacity_global": (fg_num / fg_den) if fg_den > 0 else "N/A",
        "cameras_used": used, "cameras_without_background": skipped,
        "L1": photometric.get("L1", ""), "SSIM": photometric.get("SSIM", ""),
        "training_total_loss": photometric.get("total", ""),
        **shape,
    }
    writer.global_row(row)
    return row
