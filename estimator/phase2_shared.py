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


def measurement_terms(lin, obs_frame, cam_ids, cfg, model_error=None, model=None):
    """Per-camera `CameraTerms` (estimator/observation_terms.py) of Phase 2/3.

    ONE sigma per sample, sigma_eff^2 = sigma_sensor^2 + sigma_model^2 (the
    model-error map is estimated once per run and is fixed), used in the IRLS
    weight, the GN precision AND the robust objective (Student-t NLL), so the
    solved and the accepted objective are the same function (P1-1).

      model None    original mode: Student-t RGB (cfg.robust_measurement) on every
                    pixel, Student-t depth on V_orig AND the alpha gate
      model given   masked modes: fixed-quadratic RGB on Y(b) - C over the whole
                    image, robust depth on V_orig AND M, alpha term in expected-quad
    `obs_frame` is indexed by ABSOLUTE camera id, `lin.views` by position in cam_ids.
    """
    from .observation_terms import CameraTerms, camera_terms
    if model is not None:
        from .masked_observation import phase1_camera_terms
        return phase1_camera_terms(lin.views, [obs_frame[c] for c in cam_ids], cfg, model,
                                   model_error=model_error, cam_ids=cam_ids)
    out = []
    for j, c in enumerate(cam_ids):
        v = lin.views[j]
        ob = obs_frame[c]
        if ob is None:
            z3, z = torch.zeros_like(v.rgb), torch.zeros_like(v.depth)
            out.append(CameraTerms(z3, z, z3, z, None, None, 0.0, 0.0, 0.0, 0, 0, 0))
            continue
        rgb = ob[0].to(v.rgb.device)
        d = ob[1].to(v.depth.device)
        valid = (ob[2].to(v.depth.device) if len(ob) > 2 and ob[2] is not None else d > 0)
        out.append(camera_terms(v, rgb, d, valid, cfg, rgb_robust=bool(cfg.robust_measurement),
                                rgb_model_var=None if model_error is None else model_error.rgb_var[c],
                                depth_model_var=None if model_error is None else model_error.depth_var[c]))
    return out


def effective_precision_masked(lin, prepared_all, cam_ids, cfg, model_error=None):
    """The masked observation mode's (W_rgb, W_depth) per camera, for a camera subset.

    W_rgb is FIXED: 1/(sigma_sensor^2 + sigma_model^2) per pixel and channel (the
    same sensor + model-error variance as `effective_precision`, estimated once),
    WITHOUT the residual-dependent Student-t weight, because the exact background
    expectation is defined for a fixed quadratic RGB cost. Depth keeps the existing
    robust precision, on V_used = V_orig AND M (and the pre-existing coverage gate).
    """
    from .masked_observation import depth_precision
    out = []
    for j, c in enumerate(cam_ids):
        v = lin.views[j]
        ob = prepared_all[c].to(v.rgb.device)
        if model_error is not None:
            pr = model_error.effective_rgb_precision(c, cfg.rgb_sigma).expand_as(v.rgb).contiguous()
        else:
            pr = torch.full_like(v.rgb, 1.0 / (cfg.rgb_sigma ** 2))
        r_d = torch.where(ob.depth_valid, ob.depth_observed - v.depth, torch.zeros_like(v.depth))
        pd, valid = depth_precision(v, ob, r_d, cfg)
        if model_error is not None:
            sv = torch.ones_like(pd) * (cfg.depth_sigma_a ** 2)
            pd = model_error.effective_depth_precision(c, sv, robust_weight=pd * sv)
            pd = pd * valid.to(pd.dtype)
        out.append((pr, pd))
    return out


def weighted_energy_masked(lin, prepared_all, cam_ids, precisions, model):
    """The masked objective per camera: RGB on the known background, foreground depth
    and (expected mode) 1/2 kappa (A - M)^2."""
    from .masked_observation import residuals
    rows = []
    for j, c in enumerate(cam_ids):
        v = lin.views[j]
        ob = prepared_all[c].to(v.rgb.device)
        pc, pd = precisions[j]
        r_c, r_d, r_a = residuals(v, ob, model.background_tensor(v.rgb.device))
        er = 0.5 * float((pc.double() * r_c.double().square()).sum())
        ed = 0.5 * float((pd.double() * r_d.double().square()).sum())
        ea = 0.5 * float((model.kappa(pc).double() * r_a.double().square()).sum())
        rows.append({"camera": c, "rgb": er, "depth": ed, "alpha": ea, "total": er + ed + ea})
    return rows


class PreparedFrames:
    """Lazy, per-frame PreparedRGBD lists (processing order) for the Phase 2/3 filters:
    a frame is prepared when first touched and kept until another frame is asked for,
    so a long real sequence is never duplicated in memory."""

    def __init__(self, masked_seq, obs_all):
        if len(obs_all) != len(masked_seq.masks):
            raise AssertionError(f"{len(obs_all)} observation frames but {len(masked_seq.masks)} mask frames")
        self.seq, self.obs_all = masked_seq, obs_all
        self._k, self._v = None, None

    def __len__(self):
        return len(self.obs_all)

    def __getitem__(self, k):
        if self._k != k:
            self._v = self.seq.prepare_frame(k, [o for o in self.obs_all[k]], device="cpu")
            self._k = k
        return self._v


def render_state(truth, mu, Rg, cameras, cfg, build_linearization, background=None):
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
    return build_linearization(s, cameras, eps_depth=cfg.eps_depth, background=background)
