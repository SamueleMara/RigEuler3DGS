"""One definition of the per-camera RGB-D measurement terms for every stage.

Each camera at one linearization yields a `CameraTerms` record:

    p_rgb, p_d      Gauss-Newton precisions (the IRLS local model)
    r_rgb, r_d      residuals (r_d exactly 0 where depth is not used)
    kappa, r_a      the accumulated-alpha term of the masked expected-quad mode
                    (None otherwise): kappa = trace(W_rgb Sigma_b), r_a = M - A
    e_rgb, e_d, e_a the objective contributions, float64

The objective of a channel is ONE fixed function of the state for the whole
frame (CODEX_FIX_MASKED_IEKF_DEFAULT_AND_VALIDATION_2026-10-21.md, P1-1):

  robust channel   sum_valid (nu+1)/2 log1p(z^2/nu),  z = r / sigma_eff
                   (its derivative is the IRLS score w(z) r / sigma_eff^2)
  fixed quadratic  sum 1/2 r^T W r                      (masked RGB, W fixed)

with sigma_eff^2 = sigma_sensor^2 + sigma_model^2 when a model-error map is
supplied (fixed for the run, so it carries no state derivative), and the same
sigma_eff in the weight, the precision and the objective.

Depth availability = observed validity (V_orig, or V_orig AND M in the masked
mode; always finite and > 0) AND the PRE-EXISTING predicted-alpha gate
alpha > min_alpha_for_depth.
The gate depends on the state, so the depth term can switch samples on/off
between trial states; the counts of observed-valid, used and gated-out samples
are reported per camera so that discontinuity is visible, never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .measurement import channel_sigma, robust_term


@dataclass
class CameraTerms:
    p_rgb: torch.Tensor
    p_d: torch.Tensor
    r_rgb: torch.Tensor
    r_d: torch.Tensor
    kappa: torch.Tensor | None
    r_a: torch.Tensor | None
    e_rgb: float
    e_d: float
    e_a: float
    depth_observed: int          # observed-valid depth samples (before the alpha gate)
    depth_used: int              # samples actually in the depth term
    depth_gated: int             # observed-valid samples dropped by the alpha gate

    @property
    def energy(self) -> float:
        return self.e_rgb + self.e_d + self.e_a

    def as_prec4(self):
        return (self.p_rgb, self.p_d, self.r_rgb, self.r_d)


def _depth_sigma_model(cfg):
    return (cfg.depth_sigma_model if ":" in str(cfg.depth_sigma_model)
            else {"kind": cfg.depth_sigma_model, "a": cfg.depth_sigma_a, "b": cfg.depth_sigma_b})


def depth_sensor_sigma(depth_obs: torch.Tensor, valid: torch.Tensor, cfg) -> torch.Tensor:
    """sigma_D(z) of the configured depth model at the OBSERVED depth (fixed per frame)."""
    rng = torch.where(valid, depth_obs, torch.zeros_like(depth_obs))
    return channel_sigma(rng.unsqueeze(0), sigma_rgb=(), depth_sigma_model=_depth_sigma_model(cfg),
                         depth_range=rng.unsqueeze(0), layout=("depth",), channel_dim=0).squeeze(0)


def camera_terms(view, rgb_target: torch.Tensor, depth_obs: torch.Tensor, depth_valid_obs: torch.Tensor,
                 cfg, *, rgb_robust: bool, rgb_model_var: torch.Tensor | None = None,
                 depth_model_var: torch.Tensor | None = None,
                 coverage: torch.Tensor | None = None, kappa_fn=None,
                 rgb_valid: torch.Tensor | None = None) -> CameraTerms:
    """The measurement of one camera.

    rgb_target   the RGB observation (I in the original mode, Y(b) in the masked modes)
    rgb_robust   Student-t on RGB (original mode); False = fixed quadratic (masked modes)
    *_model_var  fixed model-error variances (Phase 2/3), or None
    coverage     M (masked modes); with kappa_fn, adds the alpha term 1/2 kappa (M - A)^2
    rgb_valid    optional [H,W] RGB sample mask (the hard negative-ray mode restricts RGB to M = 1)
    """
    # ---- RGB ---------------------------------------------------------------
    r_rgb = rgb_target - view.rgb
    if cfg.use_rgb:
        s2 = torch.full_like(r_rgb, float(cfg.rgb_sigma) ** 2)
        if rgb_model_var is not None:
            s2 = s2 + rgb_model_var.to(s2.device, s2.dtype).expand_as(s2)
        valid_rgb = torch.isfinite(r_rgb)
        if rgb_valid is not None:
            valid_rgb = valid_rgb & rgb_valid.to(r_rgb.device).bool()[None]
        p_rgb, e_rgb = robust_term(r_rgb, s2.sqrt(), valid_rgb, rgb_robust, cfg.student_t_nu)
    else:
        p_rgb, e_rgb = torch.zeros_like(r_rgb), 0.0
    # ---- depth (observed validity AND the pre-existing predicted-alpha gate) --
    # positive finite camera Z only: zero / negative / NaN are never a measurement
    obs_valid = depth_valid_obs.bool() & torch.isfinite(depth_obs) & (depth_obs > 0)
    used = obs_valid & (view.alpha > cfg.min_alpha_for_depth)
    r_d = torch.where(used, depth_obs - view.depth, torch.zeros_like(view.depth))
    if cfg.use_depth:
        sd = depth_sensor_sigma(depth_obs, obs_valid, cfg)
        if depth_model_var is not None:
            sd = (sd * sd + depth_model_var.to(sd.device, sd.dtype)).sqrt()
        p_d, e_d = robust_term(r_d, sd, used, cfg.robust_measurement, cfg.student_t_nu)
    else:
        p_d, e_d = torch.zeros_like(r_d), 0.0
    # ---- accumulated alpha (masked expected-quad only) -----------------------
    kappa = r_a = None
    e_a = 0.0
    if coverage is not None and kappa_fn is not None:
        kappa = kappa_fn(p_rgb)
        r_a = coverage.to(view.alpha.device, view.alpha.dtype) - view.alpha
        e_a = 0.5 * float((kappa.double() * r_a.double().square()).sum())
    n_obs, n_used = int(obs_valid.sum()), int(used.sum())
    return CameraTerms(p_rgb, p_d, r_rgb, r_d, kappa, r_a, float(e_rgb), float(e_d), float(e_a),
                       n_obs, n_used, n_obs - n_used)


def summarize(terms) -> dict:
    """Frame totals for reports: energies and depth coverage."""
    return {"rgb": sum(t.e_rgb for t in terms), "depth": sum(t.e_d for t in terms),
            "alpha": sum(t.e_a for t in terms),
            "depth_observed_valid": sum(t.depth_observed for t in terms),
            "depth_used": sum(t.depth_used for t in terms),
            "depth_gated_by_alpha": sum(t.depth_gated for t in terms)}
