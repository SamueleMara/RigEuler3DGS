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


class Phase1Audit:
    def __init__(self, labels=None, weak_tau=1e-6):
        self.rows, self.labels, self.weak_tau = [], labels, weak_tau

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


def run_window(args, audit):
    """The production Phase-1 setup of tools/run_motion_coherence_support.py, then
    bootstrap_positions with `audit` (or None). Returns positions, info, extras."""
    from estimator import run_config as rc
    from estimator.masked_observation import from_args as masked_from_args
    from estimator.sparse_iekf import IEKFConfig
    from tools.kpart_pipeline import bootstrap_positions
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
                            visibility_constraint=None)
    frames_raw, frame_ids = dataset_rgbd_frames(cams_all[:ncam], cfg, ns, truth, Path(args.source), direction=args.direction)
    cfg = rc.configure(args, cfg, frame_ids["cameras"][:ncam])
    ns.masked_observation = masked_from_args(args, args.source, frames_raw, frame_ids, cfg)
    ns.phase1_audit = audit
    window = [(ob, fid) for (ob, _s, _d), fid in zip(frames_raw, frame_ids["frames"])]
    pos, info = bootstrap_positions(truth, camera_specs(cams_all, ncam), window, cfg, None, ns, verbose=False)
    return pos, info, {"truth": truth, "cfg": cfg, "frame_ids": frame_ids}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--direction", choices=("forward", "backward"), default="forward")
    ap.add_argument("--bootstrap-iters", type=int, default=12)
    ap.add_argument("--iekf-supernode-size", type=int, default=64)
    ap.add_argument("--gt-labels", choices=("auto", "off"), default="auto", help="evaluation only")
    ap.add_argument("--out", required=True)
    from estimator.masked_observation import add_cli as mask_cli
    from estimator.run_config import add_cli as rc_cli
    mask_cli(ap)
    rc_cli(ap, phase1=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    audit = Phase1Audit()
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
           "gt_labels": gt_info, "rows": audit.rows, "runtime_s": round(time.time() - t0, 1),
           "notes": ["P0 / P_post are per-Gaussian block approximations: inter-Gaussian and shared-camera "
                     "covariance are not represented",
                     "standardized residuals are an empirical diagnostic; no chi-square NIS calibration is claimed "
                     "(Student-t weights, spatially correlated pixels)",
                     "camera gradients are diagnostics only: no gating, no ownership"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {a.out} ({len(audit.rows)} audited frames, {out['runtime_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
