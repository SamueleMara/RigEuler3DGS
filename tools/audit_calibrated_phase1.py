#!/usr/bin/env python
"""READ-ONLY Phase-1 covariance and camera-conflict audit.

RigEuler3DGS_Codex_Surgical_Fixes_4c88725.md section 4. Runs the production Phase 1
(tools/kpart_pipeline.bootstrap_positions -> estimator.sparse_iekf.iekf_frame_step)
on a short window with a diagnostic callback. The callback gets clones of the
covariances, renders the accepted state once more and never feeds anything back:
tracking states and covariances are identical with and without it (tested).

Per frame:
  covariance   P_prev (P0 on the first step), F P_prev F^T, Q(dt), P_pred, P_post:
               per-Gaussian position / rotation / velocity / angular-velocity SDs
               (quantiles), pose-velocity cross-block norms, PSD minima, and the pose
               variance contributed by the velocity block alone, F_pv P_vv F_pv^T
  per camera   RGB / depth / alpha supports and energies, standardized residuals
               z = r sqrt(p) (quantiles, |z| < 1/2/3 coverage) -- an EMPIRICAL view,
               not a chi-square NIS claim (Student-t weights, correlated pixels);
               sensor sd, declared model-mismatch sd and their combination, labelled
               separately; mask IoU and off-mask opacity of the rendered alpha;
               per-camera Fisher (trace, weak directions) and gradient
               g_c = J_c^T W_c r_c with its agreement with the other cameras and with
               the accepted update (diagnostic only: no gating, no ownership)
  state        accepted / rejected iterations, stop reason, displacement quantiles
               vs the canonical state and vs the previous frame
  GT           evaluation only, optional: per-Gaussian moving / static labels from a
               labelled point cloud -> static drift and moving displacement reported apart

Gate A of RigEuler3DGS_Phase1_Covariance_Velocity_Corrections_Codex.md (all read-only):
  fisher_check     per-camera Fisher INCLUDING the alpha block (same kappa as production),
                   sum_c F_c vs the production measurement_operators Fisher on the same
                   linearization and channel terms
  forces_at        per-camera channel energies and forces g_c = J_c^T W_c r_c (+ alpha) at
                   the CANONICAL state (zero displacement) and at the PROPAGATED PRIOR, i.e.
                   at a common pre-update state, not only at the posterior optimum
  velocity_gain    accepted twist change vs the linear pose->twist conditional
                   P_tp P_pp^-1 dpose of the predicted block, induced speed change
  attribution      Gaussians with their own measurement information vs moved only through
                   their group vs prediction-only (no information, no update)
  ray_motion       accepted update along / across each camera ray
  solver           every inner iteration (energy, gradient, step, gain ratio, PCG) and the
                   stop reason; iteration_budget is reported as NOT converged
  fd_check         analytic J v vs central finite differences of the production render at
                   the accepted state (RGB, alpha, depth)
  residuals        depth / RGB residuals split into mask interior, mask boundary band and
                   observed-depth discontinuities, signed (camera by camera)
  --snapshot-dir   per-frame npz with the 12D predicted / posterior means and the full
                   12x12 P_prev, P_pred, P_post blocks (float32)
  --residual-report-only   no tracking: the canonical render against every window frame,
                   per-pixel temporal mean (persistent part) vs temporal sd, by region

    python tools/audit_calibrated_phase1.py --model <endpoint model dir> --source <scene> \\
        --motion-start-frame S --motion-end-frame E [--frame-timestamps auto] \\
        [--sensor-model calibration/sensor_model.json] [--p0-geometry <model>/geometry_information.npz] \\
        --out audit.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))


def q(x, qs=(0.5, 0.9, 0.99)):
    x = torch.as_tensor(x).double().reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return None
    return [float(v) for v in torch.quantile(x.cpu(), torch.tensor(qs, dtype=torch.float64))]


def block_report(P: torch.Tensor) -> dict:
    d = torch.diagonal(P, dim1=1, dim2=2).double().clamp_min(0).sqrt()
    ev = torch.linalg.eigvalsh(0.5 * (P + P.transpose(1, 2)).double().cpu())
    return {"pos_sd_m": q(d[:, 0:3].mean(1)), "rot_sd_rad": q(d[:, 3:6].mean(1)),
            "vel_sd_mps": q(d[:, 6:9].mean(1)), "omega_sd_radps": q(d[:, 9:12].mean(1)),
            "pose_vel_cross_fro": q(P[:, :6, 6:].double().flatten(1).norm(dim=1)),
            "min_eigenvalue": float(ev.min()), "psd": bool(ev.min() >= -1e-12 * ev.abs().max())}


def camera_centres(cameras) -> torch.Tensor:
    """[C,3] world camera centres (row-vector viewmatrix: p_cam = [p, 1] @ V)."""
    return torch.stack([torch.linalg.inv(c.viewmatrix.double())[3, :3] for c in cameras])


def _dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    return torch.nn.functional.max_pool2d(m[None, None].float(), 2 * r + 1, 1, r)[0, 0] > 0.5


def residual_breakdown(view, term, ob, sd_d, band=3, edge_m=0.02) -> dict:
    """Signed residuals of one camera by image region (diagnostic categories, never gates):
    interior = used depth away from the mask boundary and observed-depth discontinuities,
    boundary = within `band` px of the mask edge, depth_edge = 3x3 observed-depth range
    > edge_m. r = observed - rendered: r < 0 means the render lies BEHIND the observation."""
    if isinstance(ob, tuple):
        D, M = ob[1].to(view.depth.device), view.alpha > 0.5
    else:
        D, M = ob.depth_observed.to(view.depth.device), ob.object_coverage.to(view.depth.device) > 0.5
    bnd = _dilate(M, band) & _dilate(~M, band)
    Dv = torch.where(torch.isfinite(D) & (D > 0), D, torch.zeros_like(D))
    big = torch.where(Dv > 0, Dv, torch.full_like(Dv, -1e9))
    small = torch.where(Dv > 0, Dv, torch.full_like(Dv, 1e9))
    rng = (torch.nn.functional.max_pool2d(big[None, None], 3, 1, 1)
           + torch.nn.functional.max_pool2d(-small[None, None], 3, 1, 1))[0, 0]
    edge = (rng > edge_m) & (Dv > 0)
    used = term.p_d > 0
    wsum = float((term.p_d * term.r_d.square())[used].sum()) or 1.0
    out = {"thresholds": {"boundary_band_px": band, "depth_edge_range_m": edge_m}}
    for name, sel in (("interior", used & ~bnd & ~edge), ("boundary", used & bnd),
                      ("depth_edge", used & edge & ~bnd)):
        r = term.r_d[sel].double()
        out[name] = None if r.numel() == 0 else {
            "samples": int(r.numel()), "fraction_of_used": float(r.numel() / max(int(used.sum()), 1)),
            "signed_median_mm": 1000 * float(r.median()), "signed_mean_mm": 1000 * float(r.mean()),
            "abs_mm_q": q(1000 * r.abs()), "fraction_render_behind": float((r < 0).double().mean()),
            "sensor_sd_mm_median": 1000 * float(sd_d[sel].median()),
            "weighted_energy_share": float((term.p_d * term.r_d.square())[sel].sum()) / wsum}
    rr = term.r_rgb[:, M & ~bnd].double()
    out["rgb_mask_interior"] = None if rr.numel() == 0 else {
        "samples": int(rr.shape[1]), "mean": rr.mean(1).tolist(), "rms": rr.square().mean(1).sqrt().tolist()}
    return out


def camera_forces(state, obs, cfg, cameras, observation_model, upd=None):
    """Per camera channel energies and translational forces g_c at `state` (read-only)."""
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import _frame_terms
    bg = None if observation_model is None else observation_model.background_tensor(state.mu.device)
    lin = build_linearization(state, cameras, eps_depth=cfg.eps_depth, background=bg)
    terms = _frame_terms(lin.views, obs, cfg, observation_model=observation_model)
    grads, rows = [], []
    for c, t in enumerate(terms):
        wr = [(torch.zeros_like(tt.r_rgb), torch.zeros_like(tt.r_d)) for tt in terms]
        wr[c] = (t.p_rgb * t.r_rgb, t.p_d * t.r_d)
        g = lin.apply_HT(wr)
        if t.kappa is not None:
            wa = [torch.zeros_like(tt.r_d) for tt in terms]
            wa[c] = t.kappa * t.r_a
            g = g + lin.apply_HT_alpha(wa)
        g = g[:, :3].double()
        grads.append(g)
        gn = g.norm(dim=1)
        row = {"camera": cfg.camera_names[c] if c < len(cfg.camera_names) else f"#{c}",
               "energy": {"rgb": t.e_rgb, "depth": t.e_d, "alpha": t.e_a},
               "samples": {"rgb": int((t.p_rgb > 0).sum()), "depth_observed": t.depth_observed,
                           "depth_used": t.depth_used},
               "force_norm_q": q(gn[gn > 0])}
        if upd is not None:
            ok = (gn > 0) & (upd.norm(dim=1) > 0)
            cs = torch.nn.functional.cosine_similarity(g[ok], upd[ok], dim=1)
            row["cos_with_accepted_update_q"] = q(cs)
            row["fraction_agreeing_with_update"] = float((cs > 0).double().mean()) if cs.numel() else None
        rows.append(row)
    agree = {}
    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            ok = (grads[i].norm(dim=1) > 0) & (grads[j].norm(dim=1) > 0)
            cs = torch.nn.functional.cosine_similarity(grads[i][ok], grads[j][ok], dim=1)
            agree[f"{rows[i]['camera']}~{rows[j]['camera']}"] = {
                "shared_gaussians": int(ok.sum()), "cos_q": q(cs),
                "fraction_conflicting": float((cs < 0).double().mean()) if cs.numel() else None}
    total = sum(grads) if grads else None
    return {"cameras": rows, "camera_agreement": agree,
            "total_force_norm_q": q(total.norm(dim=1)[total.norm(dim=1) > 0]) if total is not None else None}


def velocity_gain(prior_state, state, Ppred, dt) -> dict:
    """Accepted twist change vs the linear pose->twist conditional of the predicted block:
    dtw_lin = P_tp P_pp^-1 dpose. In the per-Gaussian linear MAP the twist update is
    exactly this conditional (the image observes pose only); a grouped solve or a
    nonlinear iterate departs from it, and the residual says by how much."""
    from estimator.process import process_predict
    from estimator.state import boxminus
    pred = process_predict(prior_state, dt)
    d = boxminus(state, pred).double()
    dpose, dtw = d[:, :6], d[:, 6:]
    P = Ppred.double()
    Ppp = 0.5 * (P[:, :6, :6] + P[:, :6, :6].transpose(1, 2))
    try:
        lin = (P[:, 6:, :6] @ torch.linalg.solve(Ppp, dpose.unsqueeze(-1))).squeeze(-1)
    except RuntimeError:
        lin = (P[:, 6:, :6] @ torch.linalg.pinv(Ppp) @ dpose.unsqueeze(-1)).squeeze(-1)
    dv, dvl = dtw[:, :3], lin[:, :3]
    dw, dwl = dtw[:, 3:], lin[:, 3:]
    nz = dv.norm(dim=1) > 0
    speed_pred = pred.v.double().norm(dim=1)
    speed_post = state.v.double().norm(dim=1)
    dmu = dpose[:, :3].norm(dim=1)
    gain = dv.norm(dim=1)[dmu > 0] / (dmu[dmu > 0] / float(dt))
    return {"dv_mps_q": q(dv.norm(dim=1)), "dv_linear_conditional_mps_q": q(dvl.norm(dim=1)),
            "dv_minus_linear_mps_q": q((dv - dvl).norm(dim=1)),
            "explained_fraction_q": q((1 - (dv - dvl).square().sum(1) / dv.square().sum(1).clamp_min(1e-30))[nz]),
            "cos_dv_linear_q": q(torch.nn.functional.cosine_similarity(dv[nz], dvl[nz], dim=1)),
            "domega_radps_q": q(dw.norm(dim=1)), "domega_linear_conditional_radps_q": q(dwl.norm(dim=1)),
            "speed_change_mps_q": q(speed_post - speed_pred), "speed_post_mps_q": q(speed_post),
            "dv_over_dmu_per_dt_q": q(gain),
            "note": "1.0 in dv_over_dmu_per_dt means the whole position correction was booked as velocity"}


def fd_check(state, lin, cameras, cfg, bg, seed=0, eps_mu=(1e-3, 1e-4, 1e-5, 1e-6), phi_over_mu=10.0) -> dict:
    """Analytic J v (the cached linearization) vs central differences of the production render,
    over a ladder of step sizes: large steps cross sorting / early-termination discontinuities,
    tiny ones hit float32 round-off, so every step is reported and the best one summarized."""
    from estimator.renderer_bridge import build_linearization, embed_pose_to_full_state
    from estimator.state import boxplus
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn((state.mu.shape[0], 6), generator=g, dtype=torch.float64)

    def rel(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-30))
    ladder = []
    for e in eps_mu:
        v = base.clone()
        v[:, :3] *= e
        v[:, 3:] *= e * phi_over_mu
        v = v.to(state.mu.device, state.mu.dtype)
        Hv, HA = lin.apply_H(v), lin.apply_H_alpha(v)
        full = embed_pose_to_full_state(v)
        Lp = build_linearization(boxplus(state, full), cameras, eps_depth=cfg.eps_depth, background=bg)
        Lm = build_linearization(boxplus(state, -full), cameras, eps_depth=cfg.eps_depth, background=bg)
        ladder.append({"eps_mu_m": e, "eps_phi_rad": e * phi_over_mu, "cameras": [
            {"rgb_rel_err": rel(Hv[c][0], 0.5 * (vp.rgb - vm.rgb)),
             "alpha_rel_err": rel(HA[c], 0.5 * (vp.alpha - vm.alpha)),
             "depth_rel_err": rel(Hv[c][1], 0.5 * (vp.depth - vm.depth))}
            for c, (vp, vm) in enumerate(zip(Lp.views, Lm.views))]})
        del Lp, Lm
    best = []
    for c in range(len(cameras)):
        best.append({k: min(r["cameras"][c][k] for r in ladder) for k in ("rgb_rel_err", "alpha_rel_err", "depth_rel_err")})
    return {"ladder": ladder, "cameras": best,
            "note": "per camera the best relative error over the step ladder; depth is normalized "
                    "Z/(A+eps), so a large error at silhouettes is expected"}


class Phase1Audit:
    def __init__(self, labels=None, weak_tau=1e-6, snapshot_dir=None, fd=True, forces=True):
        self.rows, self.labels, self.weak_tau = [], labels, weak_tau
        self.snapshot_dir = None if snapshot_dir is None else Path(snapshot_dir)
        self.fd, self.forces = fd, forces

    @torch.no_grad()
    def __call__(self, *, ti, fid, prior_state, P_prev, state, P_post, diag, obs, cfg, dt, cameras,
                 observation_model, canonical):
        from estimator.covariance import covariance_predict, make_process_noise
        from estimator.observation_terms import calibrated_camera
        from estimator.process import error_state_transition, process_predict
        from estimator.renderer_bridge import build_linearization
        from estimator.sparse_iekf import _frame_terms
        n = state.mu.shape[0]
        dev, dt_ = state.mu.device, state.mu.dtype
        F = error_state_transition(prior_state, dt, mode="so3")
        Q = make_process_noise(n, dt, cfg.process_noise_pos, cfg.process_noise_rot, cfg.process_noise_vel,
                               cfg.process_noise_omega, device=dev, dtype=dt_)
        FPF = F @ P_prev @ F.transpose(1, 2)
        Ppred = covariance_predict(P_prev, F, Q)
        Fpv = F[:, :6, 6:]
        vel_origin = Fpv @ P_prev[:, 6:, 6:] @ Fpv.transpose(1, 2)
        row = {"processing_index": ti, "frame": fid, "dt_s": float(dt),
               "covariance": {"P_prev": block_report(P_prev), "FPF^T": block_report(FPF), "Q": block_report(Q),
                              "P_pred": block_report(Ppred), "P_post": block_report(P_post),
                              "P_pred_equals_FPF+Q_max_abs": float((Ppred - FPF - Q).abs().max()),
                              "pose_variance_from_velocity_prior": {
                                  "pos_sd_m": q(torch.diagonal(vel_origin, dim1=1, dim2=2)[:, :3].clamp_min(0).sqrt().mean(1)),
                                  "rot_sd_rad": q(torch.diagonal(vel_origin, dim1=1, dim2=2)[:, 3:].clamp_min(0).sqrt().mean(1))}}}
        bg = None if observation_model is None else observation_model.background_tensor(dev)
        lin = build_linearization(state, cameras, eps_depth=cfg.eps_depth, background=bg)
        terms = _frame_terms(lin.views, obs, cfg, observation_model=observation_model)
        pred_mu = process_predict(prior_state, dt).mu
        upd = (state.mu - pred_mu).double()
        cams, grads = [], []
        for c, (v, t) in enumerate(zip(lin.views, terms)):
            name = cfg.camera_names[c] if c < len(cfg.camera_names) else f"#{c}"
            cal = calibrated_camera(cfg, name if c < len(cfg.camera_names) else None)
            # standardized with the UNWEIGHTED sd (sensor [+ declared mismatch]): the IRLS
            # precision p = w / sd^2 would cap |z| at sqrt(nu + 1) and hide large residuals
            from estimator.observation_terms import depth_sensor_sigma
            rms, dms = float(cfg.rgb_model_sigma), float(cfg.depth_model_sigma)
            sd_rgb = (torch.as_tensor(np.sqrt(cal.rgb_variance() + rms * rms), device=t.r_rgb.device,
                                      dtype=t.r_rgb.dtype).reshape(3, 1, 1)
                      if cal else torch.full((3, 1, 1), math.sqrt(cfg.rgb_sigma ** 2 + rms * rms),
                                             device=t.r_rgb.device, dtype=t.r_rgb.dtype))
            zr = (t.r_rgb / sd_rgb)[t.p_rgb > 0]
            used = t.p_d > 0
            zobs = v.depth + t.r_d
            sd_d = (torch.as_tensor(cal.depth_variance(zobs.double()), device=zobs.device).sqrt().to(zobs.dtype)
                    if cal else depth_sensor_sigma(zobs, used, cfg))
            sd_d = (sd_d * sd_d + dms * dms).sqrt()
            zd = (t.r_d / sd_d)[used]
            cov = getattr(obs[c], "object_coverage", None) if not isinstance(obs[c], tuple) else None
            A = v.alpha
            mask_stats = None
            if cov is not None:
                M = cov.to(A.device) > 0.5
                Ab = A > 0.5
                mask_stats = {"iou": float((M & Ab).sum() / ((M | Ab).sum()).clamp_min(1)),
                              "offmask_opacity_mean": float(A[~M].mean()) if bool((~M).any()) else None,
                              "foreground_alpha_mean": float(A[M].mean()) if bool(M.any()) else None}
            zero = [(torch.zeros_like(tt.p_rgb), torch.zeros_like(tt.p_d)) for tt in terms]
            prec_c = list(zero)
            prec_c[c] = (t.p_rgb, t.p_d)
            Fc = lin.build_block_fisher(prec_c)
            if t.kappa is not None:
                # the alpha block of the SAME production Fisher, with the same kappa
                ka = [torch.zeros_like(tt.r_d) for tt in terms]
                ka[c] = t.kappa
                Fc = Fc + lin.build_alpha_fisher(ka)
            F_sum = Fc if c == 0 else F_sum + Fc
            tr = torch.diagonal(Fc, dim1=1, dim2=2).sum(1)
            seen = tr > 0
            ev = torch.linalg.eigvalsh(0.5 * (Fc[seen] + Fc[seen].transpose(1, 2)).double().cpu()) if bool(seen.any()) else None
            weak = (ev < self.weak_tau * ev.max(1, keepdim=True).values.clamp_min(1e-30)).sum(1) if ev is not None else None
            wr = list((torch.zeros_like(tt.r_rgb), torch.zeros_like(tt.r_d)) for tt in terms)
            wr[c] = (t.p_rgb * t.r_rgb, t.p_d * t.r_d)
            g = lin.apply_HT(wr)
            if t.kappa is not None:
                wa = [torch.zeros_like(tt.r_d) for tt in terms]
                wa[c] = t.kappa * t.r_a
                g = g + lin.apply_HT_alpha(wa)
            grads.append(g.double())
            gn = g[:, :3].norm(dim=1).double()
            cosu = torch.nn.functional.cosine_similarity(g[:, :3].double(), upd, dim=1)
            ok = (gn > 0) & (upd.norm(dim=1) > 0)
            cams.append({
                "camera": name,
                "sigma": {"rgb_sensor_sd": (np.sqrt(cal.rgb_variance()).tolist() if cal else [cfg.rgb_sigma] * 3),
                          "rgb_model_sd": float(cfg.rgb_model_sigma), "depth_model_sd_m": float(cfg.depth_model_sigma),
                          "depth_sensor_sd_m_at_median_z": (math.sqrt(float(cal.depth_variance(
                              np.array([float(torch.median(v.depth[v.alpha > 0.5]))]))[0])) if cal and bool((v.alpha > 0.5).any()) else None),
                          "source": "calibrated" if cal else "configured (ideal / assumed)"},
                "rgb": {"samples": int((t.p_rgb > 0).sum()), "energy": t.e_rgb, "abs_z_q": q(zr.abs()),
                        "coverage_abs_z_lt_1_2_3": [float((zr.abs() < k).double().mean()) for k in (1, 2, 3)] if zr.numel() else None},
                "depth": {"observed_valid": t.depth_observed, "used": t.depth_used, "gated_by_alpha": t.depth_gated,
                          "energy": t.e_d, "abs_z_q": q(zd.abs()), "abs_residual_mm_q": q(1000 * t.r_d[t.p_d > 0].abs()),
                          "coverage_abs_z_lt_1_2_3": [float((zd.abs() < k).double().mean()) for k in (1, 2, 3)] if zd.numel() else None},
                "alpha": {"energy": t.e_a, "mask": mask_stats},
                "fisher": {"gaussians_seen": int(seen.sum()), "trace_q": q(tr[seen]),
                           "weak_directions_hist": (torch.bincount(weak, minlength=7).tolist() if weak is not None else None)},
                "gradient": {"translation_norm_q": q(gn[gn > 0]),
                             "cos_with_accepted_update_q": q(cosu[ok]),
                             "fraction_agreeing_with_update": float((cosu[ok] > 0).double().mean()) if bool(ok.any()) else None},
            })
        agree = {}
        for i in range(len(grads)):
            for j in range(i + 1, len(grads)):
                a, b = grads[i][:, :3], grads[j][:, :3]
                ok = (a.norm(dim=1) > 0) & (b.norm(dim=1) > 0)
                cs = torch.nn.functional.cosine_similarity(a[ok], b[ok], dim=1)
                agree[f"{cams[i]['camera']}~{cams[j]['camera']}"] = {
                    "shared_gaussians": int(ok.sum()), "cos_q": q(cs),
                    "fraction_conflicting": float((cs < 0).double().mean()) if cs.numel() else None}
        from estimator.sparse_iekf import measurement_operators
        alpha_terms = ([(tt.kappa, tt.r_a) for tt in terms] if terms and terms[0].kappa is not None else None)
        F_prod = measurement_operators(lin, [tt.as_prec4() for tt in terms], alpha_terms)[2]()
        row["fisher_check"] = {
            "includes_alpha": alpha_terms is not None,
            "rel_err_sum_cameras_vs_production": float((F_sum - F_prod).norm() / F_prod.norm().clamp_min(1e-30)),
            "note": "per-camera blocks are summed with the alpha block; production = measurement_operators"}
        # information attribution: own measurement / group-coupled / prediction only
        own = torch.diagonal(F_prod, dim1=1, dim2=2).sum(1) > 0
        moved = upd.norm(dim=1) > 0
        stp = (state.mu - prior_state.mu).double().norm(dim=1)
        row["attribution"] = {
            "own_measurement": int(own.sum()),
            "no_own_information_moved_by_group": int((~own & moved).sum()),
            "prediction_only": int((~own & ~moved).sum()),
            "step_from_previous_mm_q": {
                "own_measurement": q(1000 * stp[own]),
                "no_own_information_moved_by_group": q(1000 * stp[~own & moved]),
                "prediction_only": q(1000 * stp[~own & ~moved])}}
        # accepted update along / across each camera ray
        C = camera_centres(cameras).to(dev)
        un = upd.norm(dim=1)
        top = un >= torch.quantile(un.cpu(), 0.99).to(dev) if un.numel() > 100 else un > 0
        ray_rows = []
        for c, cc in enumerate(C):
            ray = state.mu.double() - cc
            ray = ray / ray.norm(dim=1, keepdim=True).clamp_min(1e-12)
            along = (upd * ray).sum(1)
            ray_rows.append({"camera": cams[c]["camera"] if c < len(cams) else f"#{c}",
                             "abs_cos_q": q((along.abs() / un.clamp_min(1e-30))[un > 0]),
                             "abs_cos_top1pct_q": q((along.abs() / un.clamp_min(1e-30))[top]),
                             "along_mm_top1pct_q": q(1000 * along[top].abs())})
        row["ray_motion"] = {"update_mm_q": q(1000 * un), "cameras": ray_rows}
        row["velocity_gain"] = velocity_gain(prior_state, state, Ppred, dt)
        if self.forces:
            from estimator.process import process_predict
            row["forces_at"] = {
                "canonical": camera_forces(canonical, obs, cfg, cameras, observation_model, upd),
                "prior": camera_forces(process_predict(prior_state, dt), obs, cfg, cameras, observation_model, upd)}
        if self.fd:
            row["fd_check"] = fd_check(state, lin, cameras, cfg, bg)
        row["residuals"] = []
        for c, (v, t) in enumerate(zip(lin.views, terms)):
            name = cfg.camera_names[c] if c < len(cfg.camera_names) else None
            cal = calibrated_camera(cfg, name)
            zobs = v.depth + t.r_d
            from estimator.observation_terms import depth_sensor_sigma
            sd = (torch.as_tensor(cal.depth_variance(zobs.double()), device=zobs.device).sqrt().to(zobs.dtype)
                  if cal else depth_sensor_sigma(zobs, t.p_d > 0, cfg))
            row["residuals"].append({"camera": name or f"#{c}", **residual_breakdown(v, t, obs[c], sd)})
        it_keys = ("iteration", "event", "energy", "trial_energy", "rgb_energy", "depth_energy", "alpha_energy",
                   "prior", "gradient_norm", "step_norm", "gain_ratio", "pcg_iterations", "pcg_rel_residual",
                   "pcg_reason", "radius", "trust_radius", "mean_dmu", "max_dmu", "reason")
        stop_row = next((r for r in diag.rows if r.get("event") == "stop"), {})
        post_row = next((r for r in diag.rows if r.get("event") == "posterior"), {})
        row["solver"] = {"iterations": [{k: r.get(k) for k in it_keys if k in r} for r in diag.rows
                                        if r.get("event") in ("initial", "accepted", "rejected", "stop")],
                         "stop_reason": stop_row.get("reason"),
                         "converged": stop_row.get("reason") == "converged",
                         # the IEKF's own label (POSTERIOR_UNCERTIFIED in first-return depth mode)
                         "posterior_status": post_row.get("posterior_status") or (
                             "LOCAL_HESSIAN_AT_CONVERGED_STATE" if stop_row.get("reason") == "converged"
                             else "UNCERTIFIED_NOT_CONVERGED"),
                         "curvature": post_row.get("curvature")}
        if self.snapshot_dir is not None:
            from estimator.process import process_predict
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            pred = process_predict(prior_state, dt)
            f32 = lambda x: x.detach().float().cpu().numpy()            # noqa: E731
            np.savez_compressed(self.snapshot_dir / f"frame_{ti:04d}.npz",
                                mu_pred=f32(pred.mu), R_pred=f32(pred.R), v_pred=f32(pred.v), omega_pred=f32(pred.omega),
                                mu_post=f32(state.mu), R_post=f32(state.R), v_post=f32(state.v), omega_post=f32(state.omega),
                                P_prev=f32(P_prev), P_pred=f32(Ppred), P_post=f32(P_post), Q=f32(Q[:1]),
                                F=f32(F), fisher_pose=f32(F_prod), dt=np.float64(dt))
        disp0 = (state.mu - canonical.mu).double().norm(dim=1)
        disp1 = (state.mu - prior_state.mu).double().norm(dim=1)
        ev = [r for r in diag.rows if r.get("event") in ("accepted", "rejected")]
        stop = next((r.get("reason") for r in diag.rows if r.get("event") == "stop"), None)
        row.update({"cameras": cams, "camera_gradient_agreement": agree,
                    "state": {"accepted": sum(r["event"] == "accepted" for r in ev),
                              "rejected": sum(r["event"] == "rejected" for r in ev), "stop_reason": stop,
                              "disp_from_canonical_mm_q": q(1000 * disp0), "disp_from_previous_mm_q": q(1000 * disp1)}})
        if self.labels is not None:
            lab = torch.as_tensor(self.labels, device=dev)
            row["gt_evaluation_only"] = {
                "static_drift_mm_q": q(1000 * disp0[lab == 0]) if bool((lab == 0).any()) else None,
                "moving_disp_mm_q": q(1000 * disp0[lab > 0]) if bool((lab > 0).any()) else None,
                "num_static": int((lab == 0).sum()), "num_moving": int((lab > 0).sum())}
        self.rows.append(row)
        del lin


def gt_labels(source: Path, mu: torch.Tensor, frame_kind: str):
    """Evaluation-only: nearest labelled sparse point -> part label per Gaussian."""
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    f = source / "pointcloud" / ("points3D_parts.ply" if frame_kind == "initial" else "points3D_parts_final.ply")
    if not f.is_file():
        return None, None
    v = PlyData.read(str(f))["vertex"]
    if "part" not in v.data.dtype.names:
        return None, None
    P = np.stack([v["x"], v["y"], v["z"]], 1)
    d, i = cKDTree(P).query(mu.detach().cpu().numpy())
    return np.asarray(v["part"])[i].astype(np.int64), {"file": str(f), "median_nn_mm": float(1000 * np.median(d))}


def setup_window(args):
    """The production Phase-1 setup of tools/run_motion_coherence_support.py."""
    from estimator import run_config as rc
    from estimator.masked_observation import from_args as masked_from_args
    from estimator.sparse_iekf import IEKFConfig
    from tools.run_iekf_scene import camera_specs, dataset_rgbd_frames, load_scene, state_from_gaussians
    _ds, _pipe, gs, sc = load_scene(args.model, str(Path(args.source) / "colmap"), camera_device="cpu",
                                    inference_only=True)
    cams_all = sorted(sc.getTrainCameras() + sc.getTestCameras(), key=lambda c: c.image_name)
    ncam = len(cams_all)
    truth = state_from_gaussians(gs)
    cfg = IEKFConfig()
    s0, n = rc.window(args, 0)
    ns = argparse.Namespace(sequence_start=s0, sequence_length=n, all_frames=False, observation_source="dataset_rgbd",
                            cameras=ncam, bootstrap_iters=args.bootstrap_iters, supernode_size=args.iekf_supernode_size,
                            seed=0, frames_on_cpu=True, frames_on_gpu=False, depth_sigma=None,
                            joint_direct_rgbd=False, joint_direct_gaussian=False, window_length=0, all_cameras=False,
                            dt=1.0 / 30.0, max_gaussians=0, no_model_error=False, model_error_beta=1.0,
                            model_error_blur=1, frame_timestamps=args.frame_timestamps, p0_geometry=args.p0_geometry,
                            p0_velocity_sigma=tuple(args.p0_velocity_sigma), phase1_covariance_reset=False,
                            visibility_constraint=None,
                            phase1_partition=getattr(args, "phase1_partition", "canonical"),
                            phase1_covariance_transport=bool(getattr(args, "phase1_covariance_transport", False)))
    frames_raw, frame_ids = dataset_rgbd_frames(cams_all[:ncam], cfg, ns, truth, Path(args.source), direction=args.direction)
    cfg = rc.configure(args, cfg, frame_ids["cameras"][:ncam])
    ns.masked_observation = masked_from_args(args, args.source, frames_raw, frame_ids, cfg)
    from estimator.first_return_observation import from_args as fr_from_args
    ns.first_return = fr_from_args(args, list(cfg.camera_names))
    window = [(ob, fid) for (ob, _s, _d), fid in zip(frames_raw, frame_ids["frames"])]
    return truth, camera_specs(cams_all, ncam), window, cfg, ns, frame_ids


def _frame_obs(ns, k, window, dev):
    obs = window[k][0]
    obs = [tuple(x.to(dev) if torch.is_tensor(x) else x for x in o) for o in obs]
    return obs if ns.masked_observation is None else ns.masked_observation.prepare_frame(k, obs, device=dev)


@torch.no_grad()
def canonical_control(truth, cameras, window, cfg, ns) -> dict:
    """The canonical state against its OWN endpoint frame: the bootstrap was fitted to this
    frame, so this is a control of the model discrepancy, not an independent test."""
    from estimator.observation_terms import calibrated_camera, depth_sensor_sigma
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import _frame_terms
    model = None if ns.masked_observation is None else ns.masked_observation.model
    dev = truth.mu.device
    obs = _frame_obs(ns, 0, window, dev)
    bg = None if model is None else model.background_tensor(dev)
    lin = build_linearization(truth, cameras, eps_depth=cfg.eps_depth, background=bg)
    terms = _frame_terms(lin.views, obs, cfg, observation_model=model)
    out = []
    for c, (v, t) in enumerate(zip(lin.views, terms)):
        name = cfg.camera_names[c] if c < len(cfg.camera_names) else None
        cal = calibrated_camera(cfg, name)
        zobs = v.depth + t.r_d
        sd = (torch.as_tensor(cal.depth_variance(zobs.double()), device=zobs.device).sqrt().to(zobs.dtype)
              if cal else depth_sensor_sigma(zobs, t.p_d > 0, cfg))
        out.append({"camera": name or f"#{c}", "energy": {"rgb": t.e_rgb, "depth": t.e_d, "alpha": t.e_a},
                    **residual_breakdown(v, t, obs[c], sd)})
    del lin
    return {"frame": window[0][1], "note": "NOT an independent test: the endpoint bootstrap used this frame",
            "cameras": out}


@torch.no_grad()
def residual_report(truth, cameras, window, cfg, ns, band=3, edge_m=0.02) -> dict:
    """Section C1 decomposition on a window the user asserts is STATIONARY: the fixed
    canonical render against every frame. Per pixel, the temporal mean of the residual is
    its persistent part (model / geometry / registration bias -- not identified as noise)
    and the temporal sd its fluctuating part (comparable with the static sensor sd).
    Nothing here estimates or sets a sigma."""
    from estimator.observation_terms import calibrated_camera, depth_sensor_sigma
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import _frame_terms
    model = None if ns.masked_observation is None else ns.masked_observation.model
    dev = truth.mu.device
    bg = None if model is None else model.background_tensor(dev)
    lin = build_linearization(truth, cameras, eps_depth=cfg.eps_depth, background=bg)
    acc = None
    for k in range(len(window)):
        obs = _frame_obs(ns, k, window, dev)
        terms = _frame_terms(lin.views, obs, cfg, observation_model=model)
        if acc is None:
            acc = [{"n": torch.zeros_like(t.r_d, dtype=torch.float64), "s": torch.zeros_like(t.r_d, dtype=torch.float64),
                    "ss": torch.zeros_like(t.r_d, dtype=torch.float64),
                    "rn": torch.zeros_like(t.r_d, dtype=torch.float64),
                    "rs": torch.zeros_like(t.r_rgb, dtype=torch.float64),
                    "rss": torch.zeros_like(t.r_rgb, dtype=torch.float64), "ob0": obs[c], "t0": t}
                   for c, t in enumerate(terms)]
        for a, t, ob, v in zip(acc, terms, obs, lin.views):
            u = (t.p_d > 0).double()
            a["n"] += u
            a["s"] += u * t.r_d.double()
            a["ss"] += u * t.r_d.double().square()
            M = (ob.object_coverage.to(dev) > 0.5) if not isinstance(ob, tuple) else (v.alpha > 0.5)
            a["rn"] += M.double()
            a["rs"] += M.double()[None] * t.r_rgb.double()
            a["rss"] += M.double()[None] * t.r_rgb.double().square()
    out = []
    for c, (a, v) in enumerate(zip(acc, lin.views)):
        name = cfg.camera_names[c] if c < len(cfg.camera_names) else None
        cal = calibrated_camera(cfg, name)
        ok = a["n"] >= 2
        mean = a["s"] / a["n"].clamp_min(1)
        var = ((a["ss"] - a["n"] * mean.square()) / (a["n"] - 1).clamp_min(1)).clamp_min(0)
        zobs = v.depth + a["t0"].r_d
        sd = (torch.as_tensor(cal.depth_variance(zobs.double()), device=dev).sqrt()
              if cal else depth_sensor_sigma(zobs, ok, cfg).double())
        ob = a["ob0"]
        M = (ob.object_coverage.to(dev) > 0.5) if not isinstance(ob, tuple) else (v.alpha > 0.5)
        bnd = _dilate(M, band) & _dilate(~M, band)
        regions = {"all": ok, "interior": ok & ~bnd, "boundary": ok & bnd}
        row = {"camera": name or f"#{c}", "frames": len(window)}
        for rn, sel in regions.items():
            if not bool(sel.any()):
                row[rn] = None
                continue
            row[rn] = {"pixels": int(sel.sum()),
                       "persistent_mean_mm_q": q(1000 * mean[sel]),
                       "persistent_rms_mm": 1000 * float(mean[sel].square().mean().sqrt()),
                       "temporal_sd_mm_q": q(1000 * var[sel].sqrt()),
                       "sensor_sd_mm_median": 1000 * float(sd[sel].median()),
                       "temporal_sd_over_sensor_sd_median": float((var[sel].sqrt() / sd[sel].clamp_min(1e-12)).median())}
        rok = a["rn"] >= 2
        rmean = a["rs"] / a["rn"].clamp_min(1)[None]
        rvar = ((a["rss"] - a["rn"][None] * rmean.square()) / (a["rn"] - 1).clamp_min(1)[None]).clamp_min(0)
        row["rgb_in_mask"] = None if not bool(rok.any()) else {
            "pixels": int(rok.sum()),
            "persistent_rms": rmean[:, rok].square().mean(1).sqrt().tolist(),
            "temporal_sd_median": rvar[:, rok].sqrt().median(1).values.tolist(),
            "sensor_sd": (np.sqrt(cal.rgb_variance()).tolist() if cal else [cfg.rgb_sigma] * 3)}
        out.append(row)
    del lin
    return {"assumption": "the window is stationary (asserted by the caller, not verified here)",
            "thresholds": {"boundary_band_px": band}, "cameras": out}


def run_window(args, audit):
    """setup_window, then bootstrap_positions with `audit` (or None). Returns positions, info, extras."""
    from tools.kpart_pipeline import bootstrap_positions
    truth, cameras, window, cfg, ns, frame_ids = setup_window(args)
    ns.phase1_audit = audit
    control = canonical_control(truth, cameras, window, cfg, ns) if audit is not None else None
    pos, info = bootstrap_positions(truth, cameras, window, cfg, None, ns, verbose=False)
    return pos, info, {"truth": truth, "cfg": cfg, "frame_ids": frame_ids, "canonical_control": control}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--direction", choices=("forward", "backward"), default="forward")
    ap.add_argument("--bootstrap-iters", type=int, default=12)
    ap.add_argument("--iekf-supernode-size", type=int, default=64)
    ap.add_argument("--gt-labels", choices=("auto", "off"), default="auto", help="evaluation only")
    ap.add_argument("--snapshot-dir", default=None, help="per-frame 12D means / 12x12 blocks (npz, float32)")
    ap.add_argument("--no-fd-check", action="store_true")
    ap.add_argument("--no-forces", action="store_true", help="skip the canonical / prior force renders")
    ap.add_argument("--residual-report-only", action="store_true",
                    help="no tracking: canonical render vs every window frame (window asserted stationary)")
    ap.add_argument("--phase1-partition", choices=("canonical", "per-frame"), default="canonical")
    ap.add_argument("--phase1-covariance-transport", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deterministic", action="store_true", help="bitwise reproducible reductions")
    from estimator.masked_observation import add_cli as mask_cli
    from estimator.run_config import add_cli as rc_cli
    from estimator.first_return_observation import add_cli as fr_cli
    mask_cli(ap)
    fr_cli(ap)
    rc_cli(ap, phase1=True)
    a = ap.parse_args(argv)
    if a.deterministic:
        from estimator.deterministic_ops import enable as _det_enable
        print(f"deterministic reductions: {_det_enable()}")
    t0 = time.time()
    if a.residual_report_only:
        truth, cameras, window, cfg, ns, _ids = setup_window(a)
        out = {"tool": "tools/audit_calibrated_phase1.py --residual-report-only", "read_only": True,
               "model": a.model, "source": a.source, "window": [a.motion_start_frame, a.motion_end_frame],
               "sensor_model_id": (cfg.sensor_model.model_id if cfg.sensor_model is not None else None),
               "declared_model_sd": {"rgb": float(cfg.rgb_model_sigma), "depth_m": float(cfg.depth_model_sigma)},
               "canonical_control": canonical_control(truth, cameras, window, cfg, ns),
               "stationary_window": residual_report(truth, cameras, window, cfg, ns),
               "runtime_s": round(time.time() - t0, 1)}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
        print(f"wrote {a.out} (residual report, {len(window)} frames, {out['runtime_s']} s)")
        return 0
    audit = Phase1Audit(snapshot_dir=a.snapshot_dir, fd=not a.no_fd_check, forces=not a.no_forces)
    if a.gt_labels == "auto":
        from tools.run_iekf_scene import load_scene, state_from_gaussians
        _d, _p, gs, _s = load_scene(a.model, str(Path(a.source) / "colmap"), camera_device="cpu", inference_only=True)
        audit.labels, gt_info = gt_labels(Path(a.source), state_from_gaussians(gs).mu,
                                          "initial" if a.direction == "forward" else "final")
    else:
        gt_info = None
    pos, info, ex = run_window(a, audit)
    out = {"tool": "tools/audit_calibrated_phase1.py", "read_only": True, "model": a.model, "source": a.source,
           "direction": a.direction, "window": [a.motion_start_frame, a.motion_end_frame],
           "sensor_policy": "calibrated" if a.sensor_model else a.sensor_policy,
           "sensor_model_id": (ex["cfg"].sensor_model.model_id if ex["cfg"].sensor_model is not None else None),
           "camera_names": list(ex["cfg"].camera_names), "P0_source": info.get("P0_source"),
           "gt_labels": gt_info, "canonical_control": ex.get("canonical_control"),
           "phase1_partition": info.get("partition"), "first_return": info.get("first_return"),
           "first_return_per_frame": [r.get("first_return") for r in info.get("per_frame", [])],
           "rows": audit.rows, "runtime_s": round(time.time() - t0, 1),
           "notes": ["P0 / P_post are per-Gaussian block approximations (BLOCK_INFORMATION_APPROXIMATION): "
                     "inter-Gaussian and shared-camera covariance are not represented",
                     "a posterior after an iteration_budget stop is UNCERTIFIED_NOT_CONVERGED",
                     "standardized residuals are an empirical diagnostic; no chi-square NIS calibration is claimed "
                     "(Student-t weights, spatially correlated pixels)",
                     "camera gradients are diagnostics only: no gating, no ownership"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {a.out} ({len(audit.rows)} audited frames, {out['runtime_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
