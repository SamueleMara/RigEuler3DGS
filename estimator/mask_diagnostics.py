"""GT-free mask diagnostics with ONE evaluation definition for every observation mode.

CODEX_FIX_MASKED_IEKF_DEFAULT_AND_VALIDATION_2026-10-21.md, P1-4: the original
and the masked arm of the A/B optimise different objectives, so their energies
are not comparable. These diagnostics are: every arm's tracked state -- centres
AND orientations, never canonical rotations -- is re-rendered on the same fixed
evaluation background and scored against the same object masks.

Masks are observations, not ground truth, and nothing here feeds back into
inference: the numbers are written next to a stage's outputs only.

Per camera and frame:

    alpha_outside_sum / _mean   accumulated alpha A on M = 0 (background leakage)
    leaking_fraction            fraction of M = 0 pixels with A > leak_alpha
    foreground_alpha_mean       mean A on M = 1 (object coverage)
    rgb_mae_foreground/_background
                                |C(b_eval) - Y(b_eval)|, Y = M I + (1 - M) b_eval
    depth_valid_original        positive finite observed depth (V_orig)
    depth_valid_in_mask         V_orig AND M
    depth_discarded_by_mask     1 - in_mask / original
    depth_mae_in_mask           |D_obs - D| on V_orig AND M AND A > min_alpha_for_depth
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

EVAL_BACKGROUND = (0.5, 0.5, 0.5)


def lazy_mask_loader(mask_root, cam_names, original_frames, height, width):
    """k -> [camera] bool masks, read from disk on demand (64 cameras x T frames
    of resident masks would not fit next to the solve)."""
    from .masked_observation import load_object_masks

    def get(k):
        return load_object_masks(Path(mask_root), cam_names, [original_frames[k]], height, width)[0]
    return get


def masks_for_run(args, source, frames_raw, frame_ids, mseq=None):
    """(mask_fn, root, reason): the resident masks of a masked run, else a lazy
    loader; mask_fn is None (with the reason) when the dataset has no masks."""
    cams = list(frame_ids.get("cameras") or [])
    ncam = len(frames_raw[0][0])
    orig = [int(f["original_frame_index"]) if isinstance(f, dict) and "original_frame_index" in f
            else int(f["dataset_frame_index"]) if isinstance(f, dict) else int(f) for f in frame_ids["frames"]]
    root = Path(args.mask_source) if getattr(args, "mask_source", None) else Path(source) / "mask"
    if mseq is not None:
        return (lambda k: mseq.masks[k]), str(mseq.mask_root), None
    if not root.is_dir():
        return None, str(root), f"no mask directory {root}"
    rgb0 = frames_raw[0][0][0][0]
    H, W = int(rgb0.shape[-2]), int(rgb0.shape[-1])
    try:
        get = lazy_mask_loader(root, cams[:ncam], orig, H, W)
        get(0)
    except SystemExit as ex:              # a strict-loader refusal: report, do not guess
        return None, str(root), str(ex)
    return get, str(root), None


@torch.no_grad()
def camera_diagnostics(view, mask, obs, cfg, leak_alpha=0.05) -> dict:
    dev = view.alpha.device
    M = mask.to(dev).bool()
    A = view.alpha.squeeze()
    out, fg = ~M, M
    n_out, n_fg = int(out.sum()), int(fg.sum())
    rgb, depth = obs[0].to(dev), obs[1].to(dev)
    valid = obs[2].to(dev).bool() if len(obs) > 2 and obs[2] is not None else depth > 0
    b = torch.tensor(EVAL_BACKGROUND, device=dev, dtype=rgb.dtype)[:, None, None]
    Y = torch.where(M[None], rgb, b.expand_as(rgb))
    err = (view.rgb.to(rgb.dtype) - Y).abs().mean(0)
    v_orig = valid & torch.isfinite(depth) & (depth > 0)
    v_in = v_orig & M
    n_orig, n_in = int(v_orig.sum()), int(v_in.sum())
    used = v_in & (A > cfg.min_alpha_for_depth)
    dz = torch.where(used, (depth - view.depth.squeeze()).abs(), torch.zeros_like(depth))
    return {
        "pixels_outside_mask": n_out, "pixels_in_mask": n_fg,
        "alpha_outside_sum": float(A[out].double().sum()) if n_out else 0.0,
        "alpha_outside_mean": float(A[out].double().mean()) if n_out else None,
        "leaking_fraction": float((A[out] > leak_alpha).double().mean()) if n_out else None,
        "foreground_alpha_mean": float(A[fg].double().mean()) if n_fg else None,
        "rgb_mae_foreground": float(err[fg].double().mean()) if n_fg else None,
        "rgb_mae_background": float(err[out].double().mean()) if n_out else None,
        "depth_valid_original": n_orig, "depth_valid_in_mask": n_in,
        "depth_discarded_by_mask": (1.0 - n_in / n_orig) if n_orig else None,
        "depth_mae_in_mask": float(dz[used].double().mean()) if int(used.sum()) else None,
        "depth_scored": int(used.sum()),
    }


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


class MaskDiagnostics:
    """Accumulates per-frame diagnostics of one stage; `report()` is the JSON."""

    def __init__(self, stage, mask_fn, mask_root, cameras, cfg, truth, build_linearization,
                 reason=None, leak_alpha=0.05):
        self.stage, self.mask_fn, self.mask_root = stage, mask_fn, mask_root
        self.cameras, self.cfg, self.truth = cameras, cfg, truth
        self.build_linearization, self.reason, self.leak_alpha = build_linearization, reason, leak_alpha
        self.rows, self.seconds = [], 0.0

    @property
    def available(self) -> bool:
        return self.mask_fn is not None

    @torch.no_grad()
    def frame(self, k, frame_id, mu, R, obs) -> dict | None:
        """Re-render (mu, R) camera by camera on the evaluation background."""
        if not self.available:
            return None
        from .phase2_shared import render_state
        t0 = time.perf_counter()
        masks = self.mask_fn(k)
        if len(masks) != len(obs) or len(obs) != len(self.cameras):
            raise AssertionError(f"frame {k}: {len(masks)} masks, {len(obs)} observations, "
                                 f"{len(self.cameras)} cameras")
        bg = torch.tensor(EVAL_BACKGROUND, device=self.truth.mu.device, dtype=torch.float32)
        cams = []
        for c, (cam, m, ob) in enumerate(zip(self.cameras, masks, obs)):
            if ob is None:
                cams.append({"camera": c, "missing_observation": True})
                continue
            lin = render_state(self.truth, mu, R, [cam], self.cfg, self.build_linearization, background=bg)
            cams.append(camera_diagnostics(lin.views[0], m, ob, self.cfg, self.leak_alpha))
            del lin
        row = {"processing_index": int(k), "frame": frame_id,
               "cameras_scored": sum(1 for r in cams if not r.get("missing_observation")),
               "alpha_outside_sum": sum(r.get("alpha_outside_sum", 0.0) for r in cams),
               "alpha_outside_mean": _mean(cams, "alpha_outside_mean"),
               "leaking_fraction": _mean(cams, "leaking_fraction"),
               "foreground_alpha_mean": _mean(cams, "foreground_alpha_mean"),
               "rgb_mae_foreground": _mean(cams, "rgb_mae_foreground"),
               "rgb_mae_background": _mean(cams, "rgb_mae_background"),
               "depth_valid_original": sum(r.get("depth_valid_original", 0) for r in cams),
               "depth_valid_in_mask": sum(r.get("depth_valid_in_mask", 0) for r in cams),
               "depth_mae_in_mask": _mean(cams, "depth_mae_in_mask"),
               "per_camera": cams}
        o = row["depth_valid_original"]
        row["depth_discarded_by_mask"] = (1.0 - row["depth_valid_in_mask"] / o) if o else None
        self.rows.append(row)
        self.seconds += time.perf_counter() - t0
        return row

    def report(self) -> dict:
        base = {"stage": self.stage, "definition": "estimator/mask_diagnostics.py (GT-free, same in every mode)",
                "evaluation_background": list(EVAL_BACKGROUND), "leak_alpha": self.leak_alpha,
                "mask_root": self.mask_root, "state_rendered": "tracked centres AND orientations",
                "used_in_inference": False}
        if not self.available:
            return {**base, "status": "NOT_AVAILABLE", "reason": self.reason}
        keys = ("alpha_outside_sum", "alpha_outside_mean", "leaking_fraction", "foreground_alpha_mean",
                "rgb_mae_foreground", "rgb_mae_background", "depth_discarded_by_mask", "depth_mae_in_mask")
        return {**base, "status": "OK", "frames": len(self.rows), "seconds": round(self.seconds, 2),
                "mean_over_frames": {k: _mean(self.rows, k) for k in keys},
                "rows": self.rows}


def resources(wall_s: float) -> dict:
    """Wall time and peak GPU memory of the current process (for stage_resources.json)."""
    out = {"wall_s": round(float(wall_s), 2)}
    if torch.cuda.is_available():
        out.update({"peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                    "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
                    "device": torch.cuda.get_device_name(0)})
    return out
