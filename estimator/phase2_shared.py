"""Section 4: the ONE measurement-precision path every Phase-2 consumer uses.

CODEX_COMPLETE_PHASE2_DEBUG_AND_PARALLEL_PR_MODELS.md.

The production Phase-2 runtime weights its residuals with

    R_eff = R_sensor + R_model

and so must the Stage-A audit, exact P, exact R and the held-out evaluation.
Section 4 is explicit that the formula must not be duplicated: a debug path
that rebuilds sensor precision with `build_precision` alone is weighting a
DIFFERENT objective from the filter it is auditing, and the energies it
reports are then not comparable with the filter's.

On Storage this is not a detail. The model-error term previously changed the
apparent evidence by orders of magnitude -- the image energy is dominated by
canonical/appearance mismatch, and treating that as sensor noise inflates the
posterior information enough to reject a drawer's zero-rotation null at
chi2 ~ 1e+07.
"""

from __future__ import annotations

import torch

from .measurement import build_precision


def effective_precision(lin, obs_all, cam_ids, cfg, model_error=None):
    """R_eff^-1 per camera as `(p_rgb, p_depth)`, for a camera subset.

    `obs_all` is indexed by ABSOLUTE camera id; `lin.views` is indexed by
    position within `cam_ids`. Keeping the two indexings separate is what lets
    a camera fold render three of eight cameras without silently pairing view
    0 with observation 0 when the fold does not start at camera 0.
    """
    out = []
    for j, c in enumerate(cam_ids):
        v = lin.views[j]
        ob = obs_all[c]
        if ob is None:
            out.append((torch.zeros_like(v.rgb), torch.zeros_like(v.depth)))
            continue
        gt_rgb = ob[0].to(v.rgb.device)
        gt_d = ob[1].to(v.depth.device)
        valid = (ob[2].to(v.depth.device)
                 if len(ob) > 2 and ob[2] is not None else None)
        pr = build_precision(gt_rgb - v.rgb, valid_mask=None,
                             sigma_rgb=cfg.rgb_sigma,
                             robust=cfg.robust_measurement,
                             nu=cfg.student_t_nu,
                             layout=("rgb", "rgb", "rgb"), channel_dim=0)
        m = (v.alpha > cfg.min_alpha_for_depth)
        if valid is not None:
            m = m & valid
        pd = build_precision((gt_d - v.depth).unsqueeze(0),
                             valid_mask=m.unsqueeze(0), sigma_rgb=(),
                             depth_sigma_model=cfg.depth_sigma_model,
                             robust=cfg.robust_measurement,
                             nu=cfg.student_t_nu, layout=("depth",),
                             channel_dim=0, use_rgb=False).squeeze(0)
        if model_error is not None:
            w = pr * (cfg.rgb_sigma ** 2)
            pr = model_error.effective_rgb_precision(c, cfg.rgb_sigma,
                                                     robust_weight=w)
            sv = torch.ones_like(pd) * (cfg.depth_sigma_a ** 2)
            pd = model_error.effective_depth_precision(c, sv,
                                                       robust_weight=pd * sv)
            pd = pd * m.to(pd.dtype)
        out.append((pr, pd))
    return out


def weighted_energy(lin, obs_all, cam_ids, precisions):
    """0.5 r^T R_eff^-1 r per camera, split into RGB and depth."""
    rows = []
    for j, c in enumerate(cam_ids):
        v = lin.views[j]
        ob = obs_all[c]
        if ob is None:
            rows.append({"camera": c, "rgb": 0.0, "depth": 0.0, "total": 0.0})
            continue
        pc, pd = precisions[j]
        rr = (ob[0].to(v.rgb.device) - v.rgb).double()
        rd = (ob[1].to(v.depth.device) - v.depth).double()
        er = 0.5 * float((pc.double() * rr * rr).sum())
        ed = 0.5 * float((pd.double() * rd * rd).sum())
        rows.append({"camera": c, "rgb": er, "depth": ed, "total": er + ed})
    return rows


def render_state(truth, mu, Rg, cameras, cfg, build_linearization):
    """Linearize a (centers, orientations) pair with the REAL renderer.

    Section 2: the orientations must reach the rasterizer. A debug helper that
    accepts `Rg` and drops it renders every state with the canonical
    covariances, so `Sigma_i(t) = R_t Sigma_i^0 R_t^T` is never exercised and
    a rotation-only state is indistinguishable from a translation of the same
    centers.
    """
    import dataclasses

    kw = {"mu": mu.to(truth.mu.dtype)}
    if Rg is not None:
        kw["R"] = Rg.to(truth.R.dtype)
    s = dataclasses.replace(truth, **kw)
    return build_linearization(s, cameras, eps_depth=cfg.eps_depth)
