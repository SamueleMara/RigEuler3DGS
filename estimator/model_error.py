"""Model error as a first-class measurement term.

CODEX_REQUIRED_MODIFICATIONS_MULTIVIEW_MULTIFRAME_KPART_IEKF.md section 13.

The framework's `R` is a SENSOR model. On real data it is not the dominant
error. Measured on USB with 4 cameras, evaluating the trained model's own
geometry against dataset frame t:

    frames apart   total robust energy   fraction that is motion
        0                20,654                  -- (geometry matches)
        1                23,652                 12.7%
        5                38,720                 46.7%
       15                77,348                 73.3%

At zero motion the residual is already 20,654: reconstruction and appearance
error the motion state cannot reduce, because the geometry already matches.
Treating that as sensor noise makes the filter chase it.

Section 13's remedy, implemented here:

    e_ref_{c,p}       = y_{0,c,p}^GT - h_{c,p}(G_0)        reference residual
    sigma^2_model     = max(sigma_floor^2, beta * e_ref^2)
    R_eff             = R_sensor + R_model

A pixel that is wrong at zero motion is wrong for reasons motion cannot fix, so
it is down-weighted during discovery -- per pixel, per camera, not globally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

EPS = 1e-12


@dataclass
class ModelErrorMap:
    """Per-camera, per-pixel model-error variance from a reference frame."""

    rgb_var: list                      # [C] each [3,H,W] sigma_model^2
    depth_var: list                    # [C] each [H,W]
    reference_frame: int
    beta: float
    rgb_floor: float
    depth_floor: float
    stats: dict = field(default_factory=dict)

    @property
    def num_cameras(self) -> int:
        return len(self.rgb_var)

    def effective_rgb_precision(self, camera: int, sensor_sigma: float,
                                robust_weight: torch.Tensor | None = None):
        """R_eff^-1 for RGB: 1 / (sigma_sensor^2 + sigma_model^2).

        The robust weight multiplies the EFFECTIVE precision, so the Student-t
        reweighting still applies on top -- model error explains the systematic
        part, the robust weight the outliers.
        """
        var = sensor_sigma * sensor_sigma + self.rgb_var[camera]
        p = 1.0 / var.clamp_min(EPS)
        return p if robust_weight is None else p * robust_weight

    def effective_depth_precision(self, camera: int, sensor_var: torch.Tensor,
                                  robust_weight: torch.Tensor | None = None):
        var = sensor_var + self.depth_var[camera]
        p = 1.0 / var.clamp_min(EPS)
        return p if robust_weight is None else p * robust_weight

    def to_dict(self) -> dict:
        return {"reference_frame": self.reference_frame, "beta": self.beta,
                "rgb_floor": self.rgb_floor, "depth_floor": self.depth_floor,
                "num_cameras": self.num_cameras, **self.stats}


@torch.no_grad()
def estimate_model_error(linearization, reference_frame_obs, *,
                         reference_index: int = 0, beta: float = 1.0,
                         rgb_floor: float = 1e-4, depth_floor: float = 1e-6,
                         blur: int = 0) -> ModelErrorMap:
    """Section 13: estimate persistent model mismatch at a reference frame.

    ``linearization`` must be built on the CANONICAL geometry G_0, and
    ``reference_frame_obs`` must be the observation of the frame that geometry
    was reconstructed from -- otherwise genuine motion is absorbed into the
    model-error map and the discovery stage goes blind to it.
    """
    rgb_var, depth_var = [], []
    stats = {"per_camera": []}
    for ci, (view, (rgb_gt, d_gt, d_valid)) in enumerate(
            zip(linearization.views, reference_frame_obs)):
        e_rgb = (rgb_gt - view.rgb)
        v_rgb = (beta * e_rgb * e_rgb).clamp_min(rgb_floor)
        e_d = (d_gt - view.depth)
        # Where the sensor has no return there is no evidence about model error
        # either; the floor applies rather than a residual computed against a
        # zero depth, which would invent an enormous variance.
        v_d = torch.where(d_valid, (beta * e_d * e_d).clamp_min(depth_floor),
                          torch.full_like(e_d, depth_floor))
        if blur > 0:
            v_rgb = _box_blur(v_rgb, blur)
            v_d = _box_blur(v_d.unsqueeze(0), blur).squeeze(0)
        rgb_var.append(v_rgb)
        depth_var.append(v_d)
        stats["per_camera"].append({
            "camera": ci,
            "rgb_reference_residual_l1": float(e_rgb.abs().mean()),
            "rgb_model_sigma_median": float(v_rgb.sqrt().median()),
            "rgb_model_sigma_max": float(v_rgb.sqrt().max()),
            "depth_reference_residual_l1": float(e_d[d_valid].abs().mean())
            if bool(d_valid.any()) else None,
            "depth_model_sigma_median": float(v_d[d_valid].sqrt().median())
            if bool(d_valid.any()) else None,
            "depth_valid_pixels": int(d_valid.sum()),
        })
    return ModelErrorMap(rgb_var=rgb_var, depth_var=depth_var,
                         reference_frame=reference_index, beta=beta,
                         rgb_floor=rgb_floor, depth_floor=depth_floor,
                         stats=stats)


def _box_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Smooth the variance map: model error is spatially correlated (a
    mis-reconstructed region, not independent pixels), and an unsmoothed map
    can zero out a single pixel that happened to match by luck."""
    k = 2 * int(radius) + 1
    c = x.shape[0]
    w = torch.ones((c, 1, k, k), device=x.device, dtype=x.dtype) / (k * k)
    return torch.nn.functional.conv2d(x.unsqueeze(0), w, padding=radius,
                                      groups=c).squeeze(0)


@torch.no_grad()
def motion_excess_map(linearization, obs, model_error: ModelErrorMap,
                      camera: int):
    """The residual a motion estimate could still explain, after model error.

    Section 13 asks for this to be persisted: it is the part of the picture the
    discovery stage is actually allowed to chase.
    """
    view = linearization.views[camera]
    dev = view.rgb.device
    rgb_gt, d_gt, d_valid = obs
    if rgb_gt.device.type != dev.type:
        rgb_gt, d_gt, d_valid = rgb_gt.to(dev), d_gt.to(dev), d_valid.to(dev)
    e_rgb = (rgb_gt - view.rgb).abs()
    m_rgb = model_error.rgb_var[camera].sqrt()
    e_d = (d_gt - view.depth).abs()
    m_d = model_error.depth_var[camera].sqrt()
    return {
        "rgb_excess": (e_rgb - m_rgb).clamp_min(0.0),
        "depth_excess": torch.where(d_valid, (e_d - m_d).clamp_min(0.0),
                                    torch.zeros_like(e_d)),
        "rgb_excess_fraction": float(((e_rgb - m_rgb).clamp_min(0.0).sum()
                                      / e_rgb.sum().clamp_min(EPS))),
        "depth_excess_fraction": float(
            ((e_d - m_d).clamp_min(0.0)[d_valid].sum()
             / e_d[d_valid].sum().clamp_min(EPS))) if bool(d_valid.any()) else None,
    }
