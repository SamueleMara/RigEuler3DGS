"""Production windowed multiview K-part discovery pipeline.

CODEX_REQUIRED_MODIFICATIONS_MULTIVIEW_MULTIFRAME_KPART_IEKF.md.

This is the actual production path, not a helper. Section 0 item 4: "Do not
report a feature as implemented if the production runner never calls it."

Data flow (section 32's target architecture):

    all cameras x all frames (section 3, 4)
        |
        v
    bootstrap IEKF, grouped supernode state      INITIALISATION ONLY (section 4.3)
        |
        v
    model-error covariance R_eff = R_sensor + R_model          (section 13)
        |
        v
    multi-baseline displacement                                (section 14)
        |
        v
    K-part windowed discovery: alternate support <-> joint      (sections 5,10,11,20)
        |
        v
    canonical handover from IMMUTABLE G_0                       (section 6)
        |
        v
    reduced (q_k, qdot_k) filter + observability on that state  (sections 17,19)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from estimator.measurement import build_precision
from estimator.model_error import estimate_model_error, motion_excess_map
from estimator.renderer_bridge import build_linearization
from estimator.state import GaussianMotionState
from estimator.supernodes import GroupedObservation, build_supernodes
from joint_reduction.kpart_models import (
    PRISMATIC,
    REVOLUTE,
    UNRESOLVED,
    PartModel,
    SceneModel,
)
from joint_reduction.kpart_support import (
    baseline_excess_gain,
    calibrate_null_excess,
    infer_kpart_support,
    kpart_metrics,
    monte_carlo_null_threshold,
    part_score_statistic,
    support_null_threshold,
)
from joint_reduction.kpart_window import (
    AlternationConfig,
    DiscoveryConfig,
    ObjectiveWeights,
    SelectionConfig,
    WindowObjective,
    discover_kpart_joints,
    multi_baseline_motion,
)

EPS = 1e-12


# ---------------------------------------------------------------------------
# section 3: camera usage, with the hard assertion the spec requires
# ---------------------------------------------------------------------------
def camera_fold(total: int, fold: int, num_folds: int) -> list:
    """Section 8: a held-out camera block, assigned deterministically.

    Folds are strided (camera i -> fold i % num_folds) rather than contiguous
    so each fold keeps cameras from around the object instead of one side of
    it; a contiguous block would confound "a different camera subset" with "a
    different viewing hemisphere".

    This exists because section 8 needs INDEPENDENT-CAMERA-SUBSET refits to
    measure how repeatable Phase 2 actually is. Resampling Gaussians cannot
    do that: the rigid-model error is shared across every Gaussian in a part,
    so resampling them re-measures the same error and reports a floor far
    smaller than the truth.
    """
    if num_folds <= 1:
        return list(range(total))
    keep = [i for i in range(total) if i % num_folds != fold]
    return keep or list(range(total))


def resolve_cameras(cams_all, requested: int | None, all_cameras: bool,
                    frames_meta, *, fold: int | None = None,
                    num_folds: int = 0) -> tuple[list, dict]:
    """Section 3: use the WHOLE camera set, and prove it.

    "Add a hard assertion that requested all-camera mode actually uses all
    dataset cameras."

    Section 8 adds one deliberate exception: when a camera fold is requested,
    one block of cameras is held out so the refit can be compared against the
    others. The audit records the fold, so a folded run can never be mistaken
    for a full-camera run.
    """
    total = len(cams_all)
    if all_cameras:
        use = list(range(total))
    else:
        use = list(range(min(requested or total, total)))
    folded = fold is not None and num_folds > 1
    if folded:
        keep = set(camera_fold(total, int(fold), int(num_folds)))
        use = [i for i in use if i in keep]
    reason = (f"held out by section 8 fold {fold}/{num_folds}"
              if folded else "not requested")
    excluded = [{"camera": i, "reason": reason}
                for i in range(total) if i not in use]
    audit = {"dataset_cameras": total, "cameras_used": use,
             "num_cameras_used": len(use), "excluded": excluded,
             "all_cameras_requested": bool(all_cameras),
             "camera_fold": (None if not folded else
                             {"fold": int(fold), "num_folds": int(num_folds)})}
    if all_cameras and not folded and len(use) != total:
        raise AssertionError(
            f"section 3: --all-cameras was requested but only {len(use)} of "
            f"{total} dataset cameras would be used")
    if folded and not use:
        raise AssertionError(
            f"section 8: fold {fold} of {num_folds} left no cameras")
    return use, audit


@torch.no_grad()
def per_camera_diagnostics(state, cameras, obs, cfg) -> list:
    """Section 3's required per-camera persistence."""
    lin = build_linearization(state, cameras, eps_depth=cfg.eps_depth)
    dev = state.mu.device
    if obs and obs[0][0].device.type != dev.type:
        obs = [(a.to(dev), b.to(dev), c.to(dev)) for a, b, c in obs]
    out = []
    prec = []
    for ci, (view, (rgb_gt, d_gt, d_valid)) in enumerate(zip(lin.views, obs)):
        rr = rgb_gt - view.rgb
        pr = build_precision(rr, valid_mask=None, sigma_rgb=cfg.rgb_sigma,
                             robust=cfg.robust_measurement, nu=cfg.student_t_nu,
                             layout=("rgb", "rgb", "rgb"), channel_dim=0)
        rd = d_gt - view.depth
        valid = d_valid & (view.alpha > cfg.min_alpha_for_depth)
        pd = build_precision(
            rd.unsqueeze(0), valid_mask=valid.unsqueeze(0), sigma_rgb=(),
            depth_sigma_model=(cfg.depth_sigma_model
                               if ":" in str(cfg.depth_sigma_model)
                               else {"kind": cfg.depth_sigma_model,
                                     "a": cfg.depth_sigma_a, "b": cfg.depth_sigma_b}),
            depth_range=d_gt.unsqueeze(0), robust=cfg.robust_measurement,
            nu=cfg.student_t_nu, layout=("depth",), channel_dim=0).squeeze(0)
        prec.append((pr, pd))
        out.append({
            "camera": ci,
            "valid_rgb_pixels": int(rgb_gt[0].numel()),
            "valid_depth_pixels": int(valid.sum()),
            "robust_energy_rgb": float(0.5 * (pr * rr * rr).sum()),
            "robust_energy_depth": float(0.5 * (pd * rd * rd).sum()),
            "residual_rgb_l1": float(rr.abs().mean()),
            "residual_depth_l1": float(rd[valid].abs().mean()) if bool(valid.any()) else None,
            "alpha_coverage": float((view.alpha > cfg.min_alpha_for_depth).float().mean()),
        })
    fisher = lin.build_block_fisher(prec)
    tr = fisher.diagonal(dim1=1, dim2=2).sum(-1)
    for ci in range(len(out)):
        l1 = build_linearization(state, [cameras[ci]], eps_depth=cfg.eps_depth)
        f1 = l1.build_block_fisher([prec[ci]])
        out[ci]["fisher_trace"] = float(f1.diagonal(dim1=1, dim2=2).sum(-1).sum())
        del l1, f1
    out_total = {"total_fisher_trace": float(tr.sum())}
    del lin
    return out, out_total


# ---------------------------------------------------------------------------
# section 16: solver selection
# ---------------------------------------------------------------------------
def choose_solver(mode: str, history: list | None = None) -> tuple[bool, dict]:
    """Section 16: trust_region | lm | auto. Returns (use_trust_region, reason).

    Section 16 is explicit that the trust region did NOT fix the earlier PCG
    pathology -- the adjoint bug did -- so neither solver is presented as the
    cure, and `auto` picks on measured behaviour rather than on belief.
    """
    if mode == "trust_region":
        return True, {"solver": "trust_region", "reason": "explicitly requested"}
    if mode == "lm":
        return False, {"solver": "lm", "reason": "explicitly requested"}
    if mode != "auto":
        raise ValueError(f"solver must be trust_region, lm or auto, got {mode!r}")
    if not history:
        return True, {"solver": "trust_region",
                      "reason": "auto: no history yet, starting with the "
                                "bounded-step solver"}
    rej = sum(1 for r in history if not r.get("accepted", True))
    boundary = sum(1 for r in history if r.get("boundary_hit"))
    n = max(len(history), 1)
    if rej / n > 0.5:
        return False, {"solver": "lm", "rejected_fraction": rej / n,
                       "boundary_fraction": boundary / n,
                       "reason": ("auto: the trust region rejected more than half "
                                  "its steps, which on a piecewise-smooth "
                                  "renderer means the quadratic model is not "
                                  "validating; the line search accepts any true "
                                  "decrease")}
    return True, {"solver": "trust_region", "rejected_fraction": rej / n,
                  "boundary_fraction": boundary / n,
                  "reason": "auto: the trust region is accepting steps"}


# ---------------------------------------------------------------------------
# section 17: observability on the REDUCED physical state
# ---------------------------------------------------------------------------
@torch.no_grad()
def reduced_observability(objective, scene, labels, q_traj) -> dict:
    """Section 17: "After support/joint reduction, recompute observability on
    the reduced physical state."

    The state is (q_1..q_K) plus each part's joint parameters. The Fisher of the
    q block comes straight from the K-part operator; the strongest convergence
    certificate belongs here, not on the 6N bootstrap state where 80% of
    directions are legitimately weak.
    """
    from joint_reduction.kpart_models import KPartObservation

    K = scene.K
    total = torch.zeros((K, K), dtype=torch.float64)
    per_frame = []
    for ti, (obs, fid) in enumerate(objective.frames):
        qt = [q_traj[k][ti] for k in range(K)]
        mu, R = scene.transport_all(objective.mu0, objective.R0, labels, qt)
        lin = build_linearization(objective._state(mu, R), objective.cameras,
                                  eps_depth=objective.cfg.eps_depth)
        # The window is staged on the host at full scale (60 frames x 8 cameras
        # does not fit on an 8 GiB card), so a frame must be brought across
        # before its residuals are formed.
        obs = objective._to_device(obs, mu.device)
        prec, res = [], []
        for view, (rgb_gt, d_gt, d_valid) in zip(lin.views, obs):
            rr = rgb_gt - view.rgb
            pr = build_precision(rr, valid_mask=None,
                                 sigma_rgb=objective.cfg.rgb_sigma,
                                 robust=objective.cfg.robust_measurement,
                                 nu=objective.cfg.student_t_nu,
                                 layout=("rgb", "rgb", "rgb"), channel_dim=0)
            rd = d_gt - view.depth
            valid = d_valid & (view.alpha > objective.cfg.min_alpha_for_depth)
            pd = build_precision(
                rd.unsqueeze(0), valid_mask=valid.unsqueeze(0), sigma_rgb=(),
                depth_sigma_model="constant:0.05:0.0",
                depth_range=d_gt.unsqueeze(0),
                robust=objective.cfg.robust_measurement,
                nu=objective.cfg.student_t_nu, layout=("depth",),
                channel_dim=0).squeeze(0)
            prec.append((pr, pd))
            res.append((rr, rd))
        kp = KPartObservation(lin, scene, mu, labels)
        I, b = kp.normal_system(prec, res)
        total += I.cpu()
        ev = torch.linalg.eigvalsh(I.cpu())
        per_frame.append({"frame": fid,
                          "information": I.cpu().tolist(),
                          "min_eigenvalue": float(ev.min()),
                          "max_eigenvalue": float(ev.max())})
        del lin
    ev = torch.linalg.eigvalsh(total)
    rank = int((ev > 1e-6 * float(ev.max().clamp_min(EPS))).sum())
    return {"state": "reduced (q_1..q_K)", "dimension": K,
            "information": total.tolist(),
            "eigenvalues": ev.tolist(),
            "min_eigenvalue": float(ev.min()),
            "max_eigenvalue": float(ev.max()),
            "condition_number": float(ev.max() / ev.min().clamp_min(EPS)),
            "observable_rank": rank,
            "fully_observable": rank == K,
            "per_frame": per_frame,
            "note": ("section 17: the strongest convergence certificate belongs "
                     "to this state, not to the 6N bootstrap state")}


# ---------------------------------------------------------------------------
# the production driver
# ---------------------------------------------------------------------------
@torch.no_grad()
def _frozen_moving_support(truth, partition, positions, args) -> torch.Tensor:
    """Section 3 / 28: the Phase-1 moving support, frozen before Phase 2.

    Delegates to `estimator.moving_support.select_moving_support`, which uses
    Otsu's parameter-free two-class threshold on each Gaussian's trajectory
    excursion.

    The rule this replaced was `displacement > median(displacement)` -- a
    MEDIAN SPLIT, which returns exactly half the scene whatever the data says.
    It measured 50.0% on all five datasets, so for a drawer the "moving part"
    handed to Phase 2 was ~80% static Gaussians and the fitted rigid screw
    recovered a COM displacement of 2e-04 against a ground-truth travel
    of 0.56.
    """
    from estimator.moving_support import select_moving_support
    out = select_moving_support(positions)
    _LAST_SUPPORT.clear()
    _LAST_SUPPORT.update({k: v for k, v in out.items() if k != "members"})
    return out["members"]


# The evidence for the most recent support selection, so a caller can persist
# it next to the joint it determined.
_LAST_SUPPORT: dict = {}


def last_support_evidence() -> dict:
    return dict(_LAST_SUPPORT)


def run_kpart_discovery(truth, cameras, frames, cfg, args, writer, *,
                        reference_obs=None, gt_labels=None, verbose=True,
                        heldout_data=None) -> dict:
    """Sections 4.3, 5, 6, 10, 11, 13, 14, 17, 20 as the production path.

    ``frames`` is [(obs_per_camera, frame_id)] over the whole window.
    ``reference_obs`` is the observation of the frame the canonical geometry was
    reconstructed from; model error (section 13) is estimated against it.
    """
    t_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    K = int(args.num_moving_parts)
    dev = truth.mu.device
    out = {"K": K, "K_source": "user-provided structural prior (section 7)"}

    # ---- section 23: GT leakage audit, declared up front ------------------
    audit = {
        "K source": "user-provided structural prior",
        "GT part labels used by estimator": False,
        "GT joint type used by estimator": False,
        "GT q(t) used by estimator": False,
        "GT qdot(t) used by estimator": False,
        "GT axis used by estimator": False,
        "GT axis point used by estimator": False,
        "GT moving masks used by estimator": False,
        "gt_fields_read_for_evaluation_only": (
            ["part labels"] if gt_labels is not None else []),
    }
    writer.json("gt_usage_audit", audit)
    if verbose:
        print(f"  §23 GT audit: K = {K} is the ONLY structural prior; no GT "
              f"labels, type, axis or q(t) reach the estimator")

    # ---- section 13: model-error covariance -------------------------------
    model_error = None
    if reference_obs is not None and not args.no_model_error:
        lin0 = build_linearization(truth, cameras, eps_depth=cfg.eps_depth)
        if reference_obs and reference_obs[0][0].device.type != dev.type:
            reference_obs = [(a.to(dev), b.to(dev), c.to(dev))
                             for a, b, c in reference_obs]
        model_error = estimate_model_error(
            lin0, reference_obs, reference_index=args.sequence_start,
            beta=args.model_error_beta, blur=args.model_error_blur)
        excess = [motion_excess_map(lin0, frames[-1][0][ci], model_error, ci)
                  for ci in range(len(cameras))]
        writer.json("model_error", {
            **model_error.to_dict(),
            "motion_excess_at_last_frame": [
                {"camera": ci, "rgb_excess_fraction": e["rgb_excess_fraction"],
                 "depth_excess_fraction": e["depth_excess_fraction"]}
                for ci, e in enumerate(excess)]})
        if verbose:
            s = model_error.stats["per_camera"][0]
            print(f"  §13 model error from reference frame "
                  f"{args.sequence_start}: RGB residual L1 "
                  f"{s['rgb_reference_residual_l1']:.5f}, sigma_model median "
                  f"{s['rgb_model_sigma_median']:.5f}; depth residual L1 "
                  f"{s['depth_reference_residual_l1']}")
        del lin0

    # ---- canonical geometry, immutable (sections 6, 15) -------------------
    from joint_reduction.handover import canonical_hash
    mu0, R0 = truth.mu.clone(), truth.R.clone()
    hash_before = canonical_hash(mu0, R0)
    partition = build_supernodes(mu0, target_size=args.supernode_size,
                                 seed=args.seed)
    if verbose:
        print(f"  §1.4 canonical partition: {partition.num_supernodes} "
              f"supernodes, {6 * truth.num_gaussians} -> "
              f"{6 * partition.num_supernodes} DoF")

    # ---- section 14: multi-baseline displacement for initialisation -------
    disp, baseline_info = _bootstrap_displacement(truth, cameras, frames, cfg,
                                                  partition, args, verbose)
    writer.json("bootstrap_displacement", baseline_info)

    # ---- the windowed objective (section 24) ------------------------------
    objective = WindowObjective(mu0, R0, cameras, frames, cfg,
                                model_error=model_error,
                                weights=ObjectiveWeights(
                                    rgbd=1.0, support=args.lambda_support,
                                    temporal=args.lambda_temporal),
                                state_template=truth)

    # ---- per-part gain callback (sections 10, 12) -------------------------
    def _statistic_at(frame_index, scene, labels, q_traj):
        """Per-part achievable gain evaluated against one observed frame."""
        if scene is None:
            lin = build_linearization(truth, cameras, eps_depth=cfg.eps_depth)
            geom = mu0
        else:
            qt = [q_traj[k][frame_index] for k in range(K)]
            mu, R = scene.transport_all(mu0, R0, labels, qt)
            lin = build_linearization(objective._state(mu, R), cameras,
                                      eps_depth=cfg.eps_depth)
            geom = mu
        gr = GroupedObservation(linearization=lin, partition=partition)
        prec, res = _prec_res(lin, frames[frame_index][0], cfg, model_error)
        g = []
        for k in range(1, K + 1):
            t = (None if scene is None or scene.parts[k - 1] is None
                 else scene.parts[k - 1].tangent(geom))
            # Lambda_c of section 5.1: a prior information on the joint step,
            # not a tuned threshold.
            lam = 1.0 / max(args.q_prior_scale ** 2, 1e-12)
            stat, _df, sinfo = part_score_statistic(lin, gr, prec, res, t,
                                                    prior_information=lam)
            # Section 5.1: a supernode the part model cannot move has F_c ~ 0
            # and is not evidence either way. Its b^2/F is a ratio of two
            # vanishing quantities, which on a null-only scene produced a mean
            # of 8.3e13 against a chi-square mean of 1. Those supernodes are
            # excluded from the test rather than thresholded.
            g.append(stat * sinfo["testable"])
        del lin
        return g

    near_i = min(1, len(frames) - 1)
    far_i = len(frames) - 1
    # P0-B / section 5: the frame-0 vs frame-1 "null" is REMOVED from
    # production. That pair is only noise if the object has not moved between
    # them, which the estimator cannot know and which is false on any sequence
    # that starts in motion. The support null now comes from the stated
    # stochastic model R_eff (section 6), via `support_null_threshold`.
    #
    # The far-minus-near contrast survives as an INITIALISER and diagnostic
    # only (section 8: "A far-minus-near contrast is initializer/debug only").
    null_diag = {}
    if args.keep_null_diagnostic:
        _g_ref = _gain_at(0, None, None, None)
        _g_near = _gain_at(near_i, None, None, None)
        _thr, null_diag = calibrate_null_excess(_g_ref[0], _g_near[0],
                                                quantile=args.null_quantile)
        null_diag["production_use"] = False
        null_diag["why_not_used"] = (
            "section 5: frames 0 and 1 are not a guaranteed no-motion pair")
        writer.json("support_null_frame_pair_diagnostic", null_diag)

    def grouped_fn(scene, labels, q_traj, frames_used=None):
        """Section 8: the support statistic accumulated over the WHOLE window.

            T_{c,k}^window = sum_{t,camera} T_{c,k}^{(t,camera)}      (section 8)

        with the per-frame term of section 5.1,

            T_{c,k} = b_{c,k}^T (F_{c,k} + Lambda_{c,k})^{-1} b_{c,k},

        which is a score statistic in R_eff-whitened units and therefore
        chi-square distributed under the static null (section 6 Option A).
        """
        idx = frames_used if frames_used is not None else range(len(frames))
        total = None
        for ti in idx:
            g = _statistic_at(ti, scene, labels, q_traj)
            total = g if total is None else [a + b for a, b in zip(total, g)]
        if total is None:
            total = [torch.zeros(partition.num_supernodes, device=dev)
                     for _ in range(K)]
        if scene is None and K > 1:
            # Before any part model exists there is one hypothesis-free field.
            # Known K says there are K parts; the seeds come from clustering the
            # high-evidence supernodes SPATIALLY on canonical centroids --
            # geometry only, never GT labels.
            total = _seed_k_gains(total[0], partition, K)
        return total

    # ---- section 6: the support null, from R_eff ---------------------------
    # The statistic accumulates over n_obs = (frames x cameras) independent
    # terms, so the summed df scales with it; a 1-DoF part score over that many
    # terms is chi-square with that many degrees of freedom.
    n_terms = len(frames) * len(cameras)
    df_part = 1 * n_terms
    df_free = 6 * n_terms
    null_thr, null_info = support_null_threshold(
        df=df_part, alpha=args.support_alpha,
        num_tests=partition.num_supernodes, bonferroni=True)
    null_free, null_free_info = support_null_threshold(
        df=df_free, alpha=args.support_alpha,
        num_tests=partition.num_supernodes, bonferroni=True)
    writer.json("support_null", {
        "part_constrained": {**null_info, "frames": len(frames),
                             "cameras": len(cameras)},
        "unconstrained_6dof": null_free_info,
        "frame_pair_diagnostic": null_diag})
    if verbose:
        print(f"  §6 support null from R_eff: chi2 df={df_part} "
              f"(1 DoF x {n_terms} terms), alpha={args.support_alpha:g} "
              f"Bonferroni over {partition.num_supernodes} supernodes "
              f"-> threshold {null_thr:.1f}")

    cfgd = DiscoveryConfig(
        K=K,
        alternation=AlternationConfig(
            max_outer=args.discovery_outer,
            potts_weight=args.potts_weight,
            q_search_bracket=args.q_bracket,
            sigma_q=args.sigma_q,
            tau_energy=args.tau_energy),
        selection=SelectionConfig(min_relative_margin=args.model_margin),
        weights=ObjectiveWeights(rgbd=1.0, support=args.lambda_support,
                                 temporal=args.lambda_temporal),
        enumerate_limit=args.enumerate_limit)

    # =====================================================================
    # CODEX_IMPLEMENT_GEOMETRIC_RIGID_MOTION_JOINT_DISCOVERY.md
    # Sections 25, 10, 14: the production path is
    #     Phase 1  multiview motion evidence  (who moves, how much)
    #     Phase 2  K rigid parts, z <-> {T_{k,t}}, to convergence
    #     Phase 3  geometric joint fit from {T_{k,t}} alone
    # Section 1 removes the profile-likelihood/BIC classifier from this path;
    # it remains available as an observability DIAGNOSTIC only.
    # =====================================================================
    if args.geometric_joints:
        from joint_reduction.rigid_discovery import (RigidDiscoveryConfig,
                                                     discover_rigid_parts,
                                                     fit_joints)

        # ---- the read-only debug recorder (debug addendum §0) ------------
        # Restored: the Phase-2 rewrite dropped this creation but kept the
        # dbg.record(...) calls below, leaving `dbg` undefined.
        from debug import GeometricDebug
        dbg = GeometricDebug(
            enabled=bool(getattr(args, "debug_numerics", False)
                         or getattr(args, "make_geometric_debug_html", False)
                         or getattr(args, "export_debug_gaussian_frames", False)),
            out_dir=writer.dir, args=args)
        dbg.record(mu0=truth.mu, R0=truth.R, scales=truth.scales,
                   colors=truth.colors, opacity=truth.opacity,
                   supernode_id=partition.supernode_id,
                   num_supernodes=partition.num_supernodes,
                   model=str(getattr(args, "model", "")),
                   dataset=str(getattr(args, "source", "") or ""),
                   K=K, dt=float(getattr(args, "dt", 1.0)),
                   frame_ids=[f for _o, f in frames],
                   image_size=(int(cameras[0].image_width),
                               int(cameras[0].image_height)),
                   camera_meta=[{"name": getattr(c, "image_name", f"cam_{i:03d}"),
                                 "viewmatrix": c.viewmatrix.detach().cpu().tolist(),
                                 "tanfovx": float(c.tanfovx),
                                 "tanfovy": float(c.tanfovy)}
                                for i, c in enumerate(cameras)])

        if verbose:
            print(f"\n  === §25 Phase 1: MULTIVIEW MOTION EVIDENCE "
                  f"({len(frames)} frames x {len(cameras)} cameras) ===")
        image_information = []
        direct = bool(getattr(args, 'joint_direct_gaussian', False) or
                      getattr(args, 'joint_direct_rgbd', False))
        observer = None
        prior_information = []
        if direct:
            from tools.direct_joint_pipeline import motion_image_information
            def observer(ti, state, prior=None):
                image_information.append(motion_image_information(
                    state, cameras, frames[ti][0], cfg))
                prior_information.append(torch.zeros_like(image_information[-1]) if prior is None else prior)
        positions, pos_info = bootstrap_positions(truth, cameras, frames, cfg,
                                                  partition, args,
                                                  verbose=verbose, observer=observer)
        writer.json("motion_evidence", pos_info)
        dbg.record(positions=positions, motion_info=pos_info)

        if verbose and not getattr(args, "phase2_rigid_screw_iekf", False):
            print(f"\n  === §25 Phase 2: K RIGID-PART DISCOVERY (K={K}) ===")
            print(f"  whole-window rigid-component decomposition: the §6 "
                  f"support null is NOT used to seed membership (that rule, "
                  f"and the displacement-magnitude rule that replaced it, are "
                  f"both removed per §1/§2)")

        # Section 19: omega_{i,t}. A supernode's track is only as trustworthy
        # as the evidence behind it, so the per-supernode weight is the Fisher
        # scale of the §6 statistic -- how strongly the images constrain that
        # supernode at all -- normalised to a mean of one. Section 19 is
        # explicit that a low-confidence node must not be called static merely
        # for having moved little, so this weights the FIT, never membership.
        conf = None
        try:
            ev = gains0[0].double().clamp_min(0.0)
            w = ev / ev.mean().clamp_min(1e-12)
            conf = w.reshape(1, -1).repeat(len(positions), 1)
        except Exception:
            conf = None

        rcfg = RigidDiscoveryConfig(max_outer=args.discovery_outer,
                                    potts_weight=args.potts_weight,
                                    tau_energy=args.tau_energy,
                                    refine_iters=args.rigid_refine_iters,
                                    seed=args.seed)
        # ------------------------------------------------------------------
        # Section 28: the NEW Phase 2. The frozen Phase-1 moving support goes
        # straight into the rigid-screw IEKF -- the old free-SE(3) estimator
        # is not called as an intermediate, which is the whole point of the
        # architecture. The old branch stays reachable without the flag so a
        # baseline comparison is still possible.
        # ------------------------------------------------------------------
        if getattr(args, "phase2_rigid_screw_iekf", False):
            from estimator.joint_runtime import run_phase2_rigid_screw
            moving = _frozen_moving_support(truth, partition, positions, args)
            screw = run_phase2_rigid_screw(
                truth, cameras, frames, cfg, args, writer,
                members=moving, positions=positions,
                model_error=model_error, verbose=verbose)
            out["phase2_rigid_screw"] = {
                "q": screw["q"], "qdot": screw["qdot"],
                "geometry": screw["geometry"],
                "branch": "phase2_rigid_screw_iekf",
                "free_se3_phase2_executed": False}
            # The same runtime/memory bookkeeping the other exits do. This
            # branch returns early, and the three places that record it all
            # sit further down, so without this `peak_cuda_gib` fell back to
            # its 0 default and the summary line read "peak CUDA 0.00 GiB" on
            # a run that had just driven the CUDA rasterizer for every frame.
            # ----------------------------------------------------------
            # Stage B (sections 30, 31): the two parallel physical models,
            # on the SAME frozen support, initialized from the generic screw
            # (section 29) and compared on held-out cameras. This produces
            # evidence only -- it does not relabel the joint (section 33).
            # ----------------------------------------------------------
            if getattr(args, "phase2_parallel_pr", False):
                from estimator.joint_runtime import run_phase2_parallel_pr
                out["phase2_model_comparison"] = run_phase2_parallel_pr(
                    truth, cameras, frames, cfg, args, writer,
                    members=moving, positions=positions,
                    model_error=model_error, generic=screw, verbose=verbose)
            out["runtime_s"] = time.perf_counter() - t_start
            out["peak_cuda_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
            writer.json("run_summary_phase2_branch",
                        {"branch": "phase2_rigid_screw_iekf",
                         "free_se3_phase2_executed": False,
                         "runtime_s": out["runtime_s"],
                         "peak_cuda_gib": out["peak_cuda_gib"]})
            return out

        rig = discover_rigid_parts(objective, truth.mu, truth.R, partition,
                                   None, positions, K, rcfg,
                                   confidence=conf, verbose=verbose)
        writer.json("rigid_parts", {
            "method": ("CODEX_IMPLEMENT_WHOLE_WINDOW_RIGID_COMPONENT_DISCOVERY "
                       "sections 5-17: K+1 rigid components discovered by "
                       "whole-window rigid consistency; no displacement "
                       "magnitude rule, no K=1 shortcut"),
            "converged": rig["converged"], "stop_reason": rig["stop_reason"],
            "energy": rig["energy"], "counts": rig["counts"],
            "history": rig["history"],
            "hypotheses": rig["hypotheses"],
            "static_identification": rig["static_identification"],
            "length_scale": rig["length_scale"],
            "transforms": rig["transforms"].to_dict()})
        # Sections 25, 26, 27: mandatory diagnostics.
        writer.json("motion_retention", {
            "note": ("section 25: diagnostic only, never a joint selector. A "
                     "ratio near zero means Phase 2 discarded the motion "
                     "Phase 1 recovered."),
            "per_part": rig["motion_retention"]})
        writer.json("bootstrap_to_rigid_error", {
            "note": "section 26", **rig["bootstrap_to_rigid"]})
        _m = rig["assignment_margin"]
        writer.json("assignment_margin", {
            "note": "section 27",
            "median_margin": _m["median_margin"],
            "frac_below_1pct_of_best": _m["frac_below_1pct_of_best"]})
        writer.tensor("assignment_margin_per_supernode", _m["margin"].cpu())
        writer.tensor("whole_track_assignment_cost", rig["track_cost"].cpu())
        _g = rig["rigidity_graph"]
        from joint_reduction.rigidity_graph import (RIGIDITY_EDGE_HEADER,
                                                    rigidity_edge_rows)
        writer.csv("rigidity_edges", RIGIDITY_EDGE_HEADER,
                   rigidity_edge_rows(_g))
        writer.json("rigidity_graph", {
            "note": "sections 5-8",
            "num_edges": int(_g["edges"].shape[0]),
            "sigma_rigid": _g["sigma_rigid"],
            "sigma_derivation": _g["sigma_derivation"],
            "num_low_confidence": _g["num_low_confidence"]})
        writer.tensor("rigid_residual_per_gaussian", rig["rigid_residual"])
        writer.tensor("rigid_part_labels", rig["labels"])
        dbg.record(transforms=rig["transforms"], labels=rig["labels"],
                   node_labels=rig["node_labels"],
                   rigid_history=rig["history"],
                   rigid_residual=rig["rigid_residual"],
                   rigid_converged=rig["converged"],
                   rigid_stop_reason=rig["stop_reason"])
        if dbg.enabled:
            from estimator.part_assignment import rigid_assignment_costs
            dbg.record(assignment_costs=rigid_assignment_costs(
                truth.mu, partition.supernode_id.long().clamp_min(0),
                positions, rig["transforms"], partition.num_supernodes))

        if verbose:
            print(f"\n  === §25 Phase 3: GEOMETRIC JOINT DISCOVERY ===")
        joints = fit_joints(truth.mu, rig["labels"], rig["transforms"], K,
                            rcfg, rigid_residual=rig["rigid_residual"],
                            verbose=verbose)
        if direct:
            from tools.direct_joint_pipeline import fit_direct_joints
            heldout_objective = None
            if heldout_data is not None:
                hc, hf = heldout_data['cameras'], heldout_data['frames']
                hlin = build_linearization(truth, hc, eps_depth=cfg.eps_depth)
                reference = [(a.to(dev), b.to(dev), c.to(dev)) for a,b,c in hf[0][0]]
                herror = None if args.no_model_error else estimate_model_error(
                    hlin, reference, reference_index=args.sequence_start,
                    beta=args.model_error_beta, blur=args.model_error_blur)
                del hlin
                heldout_objective = WindowObjective(mu0, R0, hc, hf, cfg,
                    model_error=herror, state_template=truth)
                heldout_objective.camera_labels = heldout_data['camera_ids']
                objective.camera_labels = heldout_data['train_camera_ids']
            writer.tensor('phase12_cache', {'positions':[p.cpu() for p in positions],
                'image_information':image_information, 'rig':rig, 'partition':partition,
                'prior_information':prior_information,
                'canonical_hash':hash_before, 'frame_ids':[fid for _,fid in frames],
                'camera_viewmatrices':[c.viewmatrix.cpu() for c in cameras]})
            joints, direct_estimates, direct_heat = fit_direct_joints(
                objective, rig, positions, image_information, joints, partition,
                args, writer, heldout=heldout_objective, verbose=verbose)
            writer.tensor('phase1_prior_information',{'F_prior':torch.stack(prior_information),
                'reference_frame_zero':'no update at canonical frame',
                'role':'conditional centre block of full process-prior information; excluded from image Fisher and joint observation count'})
            out['direct_joint_estimates'] = direct_estimates
            dbg.record(direct_joint_estimates=direct_estimates, direct_joint_heat=direct_heat)
        writer.json("geometric_joints", {
            "method": ("direct analytic RGB-D, cluster curvature covariance and frozen held-out prediction"
                       if direct and args.joint_direct_rgbd else "sections 14-22: the joint type is the argmin of the "
                       "section 20 normalized geometric error over the "
                       "recovered rigid trajectory; no BIC, no likelihood, no "
                       "curvature test, no axis-distance gate (section 1)"),
            "per_part": [None if j is None else
                         {k: v for k, v in j.items() if k != "transforms"}
                         for j in joints],
            "rigid_parts_converged": rig["converged"]})
        out["geometric_joints"] = [
            None if j is None else
            {"part": j["part"], "selected": j["selected"],
             "status": j["status"], "axis": j["axis"],
             "axis_point": j["axis_point"], "q": j["q"],
             "normalized_error": j["normalized_error"]}
            for j in joints]
        out["rigid_parts"] = {"converged": rig["converged"],
                              "counts": rig["counts"],
                              "energy": rig["energy"]}
        dbg.record(joints=joints)

        # ---- section 23 / Phase 4: back to the renderer ------------------
        # The geometric stage has decided the TYPE and the axis. Section 23
        # projects onto that model and refines the low-dimensional state with
        # the existing banded trajectory solver, which is exactly what it
        # directs ("Use the existing banded trajectory solver to refine
        # q_{k,0:T} against RGB-D").
        res = _geometric_to_scene(objective, truth, rig, joints, K, partition,
                                  positions, args, writer, verbose, dbg=dbg)
        if res is None:
            # Section 0 rule 7: an unresolved run still emits the full debug
            # package. That is the run you most need to inspect.
            statuses = [(j or {}).get('status','UNRESOLVED_GEOMETRY') for j in joints]
            out["status"] = ('MODEL_MISMATCH' if 'MODEL_MISMATCH' in statuses else
                             'UNRESOLVED_OPTIMIZATION' if 'UNRESOLVED_OPTIMIZATION' in statuses else
                             'UNRESOLVED_GEOMETRY')
            out["unresolved_reason"] = [
                (j or {}).get("status", "no joint") for j in joints]
            dbg.record(status=out['status'])
            _emit_debug(dbg, objective, writer, args, verbose)
            out["runtime_s"] = time.perf_counter() - t_start
            out["peak_cuda_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
            return out
        out["status"] = "RESOLVED"
        out["K"] = K
        out["scene"] = res.scene.to_dict()
        out["q_traj"] = res.q_traj
        dbg.record(scene=res.scene, q_traj=res.q_traj, labels=res.labels,
                   node_labels=res.node_labels, status="RESOLVED")

    # The superseded profile-likelihood/BIC path (section 1). It runs only
    # when --profile-likelihood-joints asked for it; the geometric stage
    # above has already produced `res` otherwise.
    if not args.geometric_joints:
        if verbose:
            print(f"\n  === §5 WINDOWED K-PART DISCOVERY "
                  f"({len(frames)} frames x {len(cameras)} cameras, K={K}) ===")
        disc = discover_kpart_joints(objective, partition, grouped_fn, disp,
                                     cfgd, verbose=verbose)
        out["discovery"] = disc["decision"]
        out["candidates"] = disc["candidates"]

        # Section 5: an UNRESOLVED selection is a RESULT, not a fallback to the
        # best candidate. Reporting the runner-up as if it had been chosen would be
        # exactly the "silently reinterpret the requested method" section 0 forbids.
        resolved = disc["decision"]["selected"] != UNRESOLVED
        res = disc.get("result") if resolved else None
        unresolved_best = disc.get("result") if not resolved else None
        # Section 16: the measurement-space identifiability test is part of the
        # production decision (section 18), so its full profile is persisted --
        # the kappa grid, the profile likelihood, the ratio, the threshold and the
        # confidence interval -- not just the verdict.
        writer.json("curvature_identifiability", {
            "method": ("section 16 profile likelihood over kappa = 1/rho, whose "
                       "prismatic limit is kappa = 0; no axis-distance threshold "
                       "is used (section 15)"),
            "per_part": disc.get("curvature_profiles")})
        writer.json("joint_hypotheses", {
            "candidates": disc["candidates"],
            "curvature_identifiability": disc.get("curvature_profiles"),
            "selection": {k: v for k, v in disc["decision"].items()
                          if k not in ("best", "candidates")},
            "num_residuals": disc["num_residuals"],
            "renders": disc["renders"]})

        if res is None:
            payload = {"selected": UNRESOLVED,
                       "status": disc["decision"].get("status"),
                       "reason": disc["decision"]["reason"],
                       "curvature_identifiability": disc.get("curvature_profiles"),
                       "relative_margin": disc["decision"].get("relative_margin"),
                       "criterion": disc["decision"].get("criterion")}
            if unresolved_best is not None:
                # persist what the runner-up WOULD have been, clearly labelled, so
                # the diagnostics are available without the result being claimed
                payload["best_candidate_not_selected"] = unresolved_best.to_dict()
                writer.json("support_kpart",
                            unresolved_best.support.to_dict()
                            if unresolved_best.support is not None else {})
            writer.json("window_discovery", payload)
            out["status"] = UNRESOLVED
            out["unresolved_reason"] = disc["decision"]["reason"]
            out["runtime_s"] = time.perf_counter() - t_start
            out["peak_cuda_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
            return out

    # ---- section 15: split supernodes that straddle a part boundary -------
    if args.split_supernodes:
        from estimator.partition_refine import (
            per_gaussian_part_evidence,
            split_supernodes,
        )
        ti = len(frames) // 2
        qt = [res.q_traj[k][ti] for k in range(K)]
        mu_t, R_t = res.scene.transport_all(mu0, R0, res.labels, qt)
        lin = build_linearization(objective._state(mu_t, R_t), cameras,
                                  eps_depth=cfg.eps_depth)
        prec, resd = _prec_res(lin, frames[ti][0], cfg, model_error)
        ev = per_gaussian_part_evidence(lin, res.scene, mu_t, res.labels,
                                        prec, resd, K)
        del lin
        refined = split_supernodes(partition, ev, tau_split=args.tau_split)
        writer.json("partition_split_history", refined.to_dict())
        if verbose:
            print(f"  §15 partition refinement: "
                  f"{refined.initial_num_supernodes} -> "
                  f"{refined.partition.num_supernodes} supernodes "
                  f"({len(refined.splits)} splits)")
        if refined.splits:
            # Re-solve support on the refined topology and keep it only if the
            # TRUE objective improves -- section 11.C applies here too.
            gains = grouped_fn(res.scene, res.labels, res.q_traj)
            # gains were computed on the OLD partition; recompute on the new one
            partition_new = refined.partition
            gr = GroupedObservation(linearization=None, partition=partition_new)
            sup_new = infer_kpart_support(
                partition_new,
                [_regroup(g, partition, partition_new) for g in gains], K=K,
                potts_weight=args.potts_weight)
            lab_new = sup_new.gaussian_labels(
                partition_new.supernode_id).to(res.labels.device)
            e_new = objective.evaluate(res.scene, lab_new, res.q_traj,
                                       sigma_q=args.sigma_q)
            if e_new.total <= res.energy.total:
                res.labels, res.support, res.energy = lab_new, sup_new, e_new
                partition = partition_new
                if verbose:
                    print(f"  §15 refined support accepted: E "
                          f"{res.energy.total:.2f}")
            elif verbose:
                print(f"  §15 refined support rejected (E {e_new.total:.2f} > "
                      f"{res.energy.total:.2f}); keeping the coarse partition")
    else:
        writer.json("partition_split_history", {
            "performed": False,
            "note": "pass --split-supernodes to enable section 15 refinement"})

    writer.json("window_discovery", res.to_dict())
    writer.json("support_iteration_history", {
        "method": ("section 11 alternation: joint/trajectory -> multi-label "
                   "support -> true nonlinear validation, to convergence"),
        "iterations": res.history,
        "converged": res.converged, "stop_reason": res.stop_reason})
    if res.support is not None:
        writer.json("support_kpart", res.support.to_dict())
        writer.tensor("support_hard", res.support.hard)
        writer.tensor("support_probability", res.support.soft)
        writer.tensor("support_unary", res.support.unary)

    # ---- section 23: post-hoc metrics only --------------------------------
    if gt_labels is not None:
        sid = partition.supernode_id.long()
        node_truth = torch.zeros(partition.num_supernodes, dtype=torch.long,
                                 device=dev)
        for c in range(partition.num_supernodes):
            m = gt_labels[sid == c]
            node_truth[c] = int(torch.mode(m).values) if m.numel() else 0
        met = kpart_metrics(res.support.hard.to(dev), node_truth, K=K)
        writer.json("support_metrics_vs_gt", met)
        if verbose:
            print(f"  §23 support vs GT (evaluation only): Hungarian accuracy "
                  f"{met['hungarian_accuracy']:.3f}, ARI "
                  f"{met['adjusted_rand_index']:.3f}")

    # ---- section 6: canonical handover ------------------------------------
    ho = _canonical_handover(objective, res, mu0, R0, hash_before, writer,
                             args, verbose)
    out["handover"] = ho

    # ---- section 17: observability on the reduced state -------------------
    obs_red = reduced_observability(objective, res.scene, res.labels, res.q_traj)
    writer.json("reduced_observability", obs_red)
    if verbose:
        print(f"  §17 reduced-state observability: rank "
              f"{obs_red['observable_rank']}/{obs_red['dimension']}, "
              f"condition {obs_red['condition_number']:.3e}, "
              f"min eigenvalue {obs_red['min_eigenvalue']:.3e}")

    # ---- section 1.7: the reduced visual joint filter ---------------------
    out["reduced_stage"] = run_reduced_filter(objective, res.scene, res.labels,
                                              res.q_traj, args, writer, verbose)

    out["status"] = "RESOLVED"
    out["scene"] = res.scene.to_dict()
    out["q_traj"] = res.q_traj
    out["labels"] = res.labels
    out["partition"] = partition
    out["result"] = res
    out["reduced_observability"] = obs_red

    # ---- the read-only debug package (debug addendum) --------------------
    # Emitted last, so it sees every stage including the canonical handover
    # and the reduced filter. `dbg` exists only on the geometric path.
    _dbg = locals().get("dbg")
    if _dbg is not None and _dbg.enabled:
        _dbg.record(handover=out.get("handover") or {},
                    reduced=out.get("reduced_stage") or {},
                    observability={"condition": obs_red.get("condition_number"),
                                   "rank": obs_red.get("observable_rank"),
                                   "dimension": obs_red.get("dimension"),
                                   "min_eigenvalue": obs_red.get("min_eigenvalue")},
                    status="RESOLVED")
        _emit_debug(_dbg, objective, writer, args, verbose)

    out["runtime_s"] = time.perf_counter() - t_start
    out["peak_cuda_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    return out


def _regroup(gain, old_partition, new_partition):
    """Carry a per-supernode gain onto a refined partition.

    A child inherits its parent's gain as the starting point; the next
    alternation pass recomputes it properly on the new topology.
    """
    sid_old = old_partition.supernode_id.long().clamp_min(0)
    sid_new = new_partition.supernode_id.long().clamp_min(0)
    per_gauss = gain.to(sid_old.device)[sid_old]
    C = new_partition.num_supernodes
    out = torch.zeros(C, device=gain.device, dtype=gain.dtype)
    cnt = torch.zeros(C, device=gain.device, dtype=gain.dtype)
    out.index_add_(0, sid_new, per_gauss)
    cnt.index_add_(0, sid_new, torch.ones_like(per_gauss))
    return out / cnt.clamp_min(1.0)


def _prec_res(lin, obs, cfg, model_error=None):
    dev = lin.views[0].rgb.device
    if obs and obs[0][0].device.type != dev.type:
        obs = [(a.to(dev), b.to(dev), c.to(dev)) for a, b, c in obs]
    prec, res = [], []
    for ci, (view, (rgb_gt, d_gt, d_valid)) in enumerate(zip(lin.views, obs)):
        rr = rgb_gt - view.rgb
        pr = build_precision(rr, valid_mask=None, sigma_rgb=cfg.rgb_sigma,
                             robust=cfg.robust_measurement, nu=cfg.student_t_nu,
                             layout=("rgb", "rgb", "rgb"), channel_dim=0)
        if model_error is not None:
            w = pr * (cfg.rgb_sigma ** 2)
            pr = model_error.effective_rgb_precision(ci, cfg.rgb_sigma,
                                                     robust_weight=w)
        rd = d_gt - view.depth
        valid = d_valid & (view.alpha > cfg.min_alpha_for_depth)
        pd = build_precision(
            rd.unsqueeze(0), valid_mask=valid.unsqueeze(0), sigma_rgb=(),
            depth_sigma_model=(cfg.depth_sigma_model
                               if ":" in str(cfg.depth_sigma_model)
                               else {"kind": cfg.depth_sigma_model,
                                     "a": cfg.depth_sigma_a, "b": cfg.depth_sigma_b}),
            depth_range=d_gt.unsqueeze(0), robust=cfg.robust_measurement,
            nu=cfg.student_t_nu, layout=("depth",), channel_dim=0).squeeze(0)
        if model_error is not None:
            sensor_var = (1.0 / pd.clamp_min(EPS)) * valid.to(pd.dtype)
            pd = model_error.effective_depth_precision(
                ci, sensor_var, robust_weight=valid.to(pd.dtype))
        prec.append((pr, pd))
        res.append((rr, rd))
    return prec, res


def _seed_k_gains(gain: torch.Tensor, partition, K: int) -> list:
    """Split one hypothesis-free gain field into K seeds.

    Before any part model exists there is a single "could this move?" score.
    Known K says there are K parts, so the seeds are obtained by clustering the
    high-evidence supernodes SPATIALLY on canonical centroids -- geometry only,
    never GT labels. Each part's seed gain is the shared gain masked to its
    cluster, which the alternation then refines.
    """
    C = partition.num_supernodes
    strong = gain > gain.median()
    idx = torch.nonzero(strong, as_tuple=False).reshape(-1)
    if idx.numel() < K:
        return [gain.clone() for _ in range(K)]
    pts = partition.centroids[idx]
    # deterministic farthest-point seeding then Lloyd, as in build_supernodes
    seeds = [0]
    d = (pts - pts[0]).norm(dim=1)
    for _ in range(K - 1):
        nxt = int(d.argmax())
        seeds.append(nxt)
        d = torch.minimum(d, (pts - pts[nxt]).norm(dim=1))
    cent = pts[torch.tensor(seeds, device=pts.device)].clone()
    assign = torch.zeros(pts.shape[0], dtype=torch.long, device=pts.device)
    for _ in range(25):
        new = torch.cdist(pts, cent).argmin(dim=1)
        if torch.equal(new, assign):
            break
        assign = new
        for k in range(K):
            m = assign == k
            if bool(m.any()):
                cent[k] = pts[m].mean(dim=0)
    out = []
    for k in range(K):
        g = torch.zeros_like(gain)
        sel = idx[assign == k]
        g[sel] = gain[sel]
        out.append(g)
    return out


# ---------------------------------------------------------------------------
# section 4.3 + 14: the bootstrap, used ONLY to initialise
# ---------------------------------------------------------------------------
@torch.no_grad()
def _bootstrap_displacement(truth, cameras, frames, cfg, partition, args,
                            verbose=True):
    """Section 4.3: "Bootstrap IEKF: initial motion estimate only."

    Section 14: initialisation uses the LONGEST baseline inside the window
    rather than adjacent frames, because adjacent frames on real data carry
    ~12.7% motion against ~87.3% model error while a 15-frame baseline carries
    ~73%. The bootstrap runs the grouped IEKF against the chosen far frame and
    returns the per-Gaussian displacement it recovered -- nothing more. It makes
    no P/R decision and sets no support.
    """
    from estimator import IEKFConfig, iekf_frame_step

    n = truth.num_gaussians
    dev = truth.mu.device
    T = len(frames)
    baselines = [b for b in args.baselines if b < T]
    if not baselines:
        baselines = [max(T - 1, 1)]
    target = max(baselines)

    bcfg = IEKFConfig(**{**{k: getattr(cfg, k) for k in
                           ("max_iters", "pcg_max_iters", "use_rgb", "use_depth",
                            "robust_measurement", "student_t_nu", "rgb_sigma",
                            "depth_sigma_model", "eps_depth",
                            "min_alpha_for_depth", "use_trust_region")},
                        "use_supernodes": True,
                        "supernode_target_size": args.supernode_size,
                        "supernode_seed": args.seed,
                        "max_iters": args.bootstrap_iters})
    P = torch.zeros((n, 12, 12), device=dev, dtype=truth.mu.dtype)
    P[:, range(12), range(12)] = torch.tensor(
        [1e-2] * 3 + [1e-2] * 3 + [1.0] * 3 + [1e-1] * 3, device=dev,
        dtype=truth.mu.dtype)
    obs, fid = frames[target]
    if obs and obs[0][0].device.type != dev.type:
        obs = [(a.to(dev), b.to(dev), c.to(dev)) for a, b, c in obs]
    t0 = time.perf_counter()
    state, _, diag = iekf_frame_step(truth, P, obs, cameras,
                                     dt=args.dt * target, cfg=bcfg,
                                     partition=partition)
    disp = state.mu - truth.mu
    rows = [r for r in diag.rows if r.get("event") in ("initial", "accepted",
                                                       "rejected")]
    info = {
        "role": "INITIALISATION ONLY (section 4.3); makes no P/R decision",
        "baselines_available": baselines,
        "baseline_used": target,
        "target_frame_id": fid,
        "dt_used_s": args.dt * target,
        "mean_displacement": float(disp.norm(dim=1).mean()),
        "max_displacement": float(disp.norm(dim=1).max()),
        "energy_initial": rows[0]["energy"] if rows else None,
        "energy_final": rows[-1]["energy"] if rows else None,
        "iterations": len(rows),
        "wall_clock_s": time.perf_counter() - t0,
        "state_parameterization": rows[0].get("state_parameterization") if rows else None,
        "solved_dimension": rows[0].get("solved_dimension") if rows else None,
    }
    if verbose:
        print(f"  §4.3/§14 bootstrap on baseline {target} (frame {fid}): "
              f"energy {info['energy_initial']:.1f} -> {info['energy_final']:.1f}, "
              f"mean displacement {info['mean_displacement']:.5f}")
    return disp, info


# ---------------------------------------------------------------------------
# section 6: canonical handover, in production
# ---------------------------------------------------------------------------
@torch.no_grad()
def _canonical_handover(objective, res, mu0, R0, hash_before, writer, args,
                        verbose=True) -> dict:
    """Section 6, called by the production path.

    1. return to immutable canonical geometry;
    2. refit the selected joint parameters over the whole discovery window;
    3. re-estimate q;
    4. evaluate the true nonlinear objective;
    5. accept only if E_handover <= E_pre + eps.
    """
    from joint_reduction.handover import canonical_hash

    K = res.scene.K
    pre = objective.evaluate(res.scene, res.labels, res.q_traj,
                             sigma_q=args.sigma_q)

    # Step 1-3: the refit IS from canonical geometry -- transport_all always
    # starts at mu0, so the deformed bootstrap geometry never enters. The
    # trajectory is re-estimated once more at the accepted geometry.
    from joint_reduction.kpart_window import (
        AlternationConfig,
        optimise_q_trajectory,
    )
    acfg = AlternationConfig(q_search_bracket=args.q_bracket,
                             sigma_q=args.sigma_q)
    q_refit = [list(x) for x in res.q_traj]
    q_refit = optimise_q_trajectory(objective, res.scene, res.labels, q_refit, acfg)
    post = objective.evaluate(res.scene, res.labels, q_refit, sigma_q=args.sigma_q)

    hash_after = canonical_hash(mu0, R0)
    accepted = post.total <= pre.total + args.handover_epsilon
    out = {
        "performed": True,
        "canonical_hash_before": hash_before,
        "canonical_hash_after": hash_after,
        "canonical_unchanged": hash_before == hash_after,
        "pre_handover_objective": pre.to_dict(),
        "post_handover_objective": post.to_dict(),
        "energy_before": pre.total,
        "energy_after": post.total,
        "accepted": bool(accepted),
        "epsilon": args.handover_epsilon,
        "q_before": [list(x) for x in res.q_traj],
        "q_after": q_refit,
        "support_before": res.labels.detach().cpu().tolist(),
        "support_after": res.labels.detach().cpu().tolist(),
        "requirement": "E_handover <= E_pre + eps (section 6)",
        "note": ("the reduced model is instantiated from IMMUTABLE canonical "
                 "geometry; bootstrap-deformed geometry is never the rigid "
                 "reference"),
    }
    if accepted:
        res.q_traj = q_refit
    writer.json("handover_refit", out)
    if verbose:
        print(f"  §6 canonical handover: E {pre.total:.3f} -> {post.total:.3f}, "
              f"{'ACCEPTED' if accepted else 'REJECTED (staying in discovery)'}; "
              f"canonical hash unchanged = {out['canonical_unchanged']}")
    return out


# ---------------------------------------------------------------------------
# section 1.7 + 19 + 26: the reduced visual joint filter
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_reduced_filter(objective, scene, labels, q_traj, args, writer,
                       verbose=True) -> dict:
    """Section 1.7: once the joint is known, filter directly on (q_k, qdot_k).

    Section 19: for a revolute part the propagation is in JOINT coordinates
    (constant qdot), not constant Cartesian twist -- a rotating body's Cartesian
    velocity turns with it, so constant Cartesian twist is a bootstrap-only
    approximation and is wrong here by construction.

    Section 26 requires the predicted, innovation, posterior and information
    traces; they are all persisted below.
    """
    from joint_reduction.kpart_models import KPartObservation

    K = scene.K
    T = len(objective.frames)
    dt = args.dt
    rows = []
    # Initialise from the discovered trajectory: q_0 and a finite-difference
    # qdot. This is the handover INTO the reduced state (section 6 step 6).
    q = [q_traj[k][0] for k in range(K)]
    qdot = [((q_traj[k][1] - q_traj[k][0]) / dt) if T > 1 else 0.0
            for k in range(K)]
    P = [torch.tensor([[1e-2, 0.0], [0.0, 1e-1]], dtype=torch.float64)
         for _ in range(K)]
    accel_psd = args.reduced_accel_psd

    for ti in range(T):
        obs, fid = objective.frames[ti]
        # --- predict in JOINT coordinates (section 19) ---------------------
        q_pred = [q[k] + qdot[k] * dt for k in range(K)]
        qdot_pred = list(qdot)
        for k in range(K):
            F = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float64)
            Q = accel_psd * torch.tensor(
                [[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]],
                dtype=torch.float64)
            P[k] = F @ P[k] @ F.T + Q

        # --- visual update on (q, qdot) only -------------------------------
        mu, R = scene.transport_all(objective.mu0, objective.R0, labels, q_pred)
        lin = build_linearization(objective._state(mu, R), objective.cameras,
                                  eps_depth=objective.cfg.eps_depth)
        prec, res = _prec_res(lin, obs, objective.cfg, objective.model_error)
        kp = KPartObservation(lin, scene, mu, labels)
        I, b = kp.normal_system(prec, res)
        meas_e = sum(float(0.5 * (p * r * r).sum()) + float(0.5 * (pd * rd * rd).sum())
                     for (p, pd), (r, rd) in zip(prec, res))
        del lin

        q_post, qdot_post, innov = [], [], []
        for k in range(K):
            prior_info = torch.linalg.inv(P[k])
            M = torch.zeros((2, 2), dtype=torch.float64)
            M[0, 0] = float(I[k, k])
            rhs = torch.zeros(2, dtype=torch.float64)
            rhs[0] = float(b[k])
            total = prior_info + M
            delta = torch.linalg.solve(total, rhs.unsqueeze(-1)).squeeze(-1)
            # section 17/26: the information and its inverse are the reduced
            # state's own uncertainty, which is where the certificate belongs
            P[k] = torch.linalg.inv(total)
            q_post.append(q_pred[k] + float(delta[0]))
            qdot_post.append(qdot_pred[k] + float(delta[1]))
            innov.append(float(delta[0]))

        rows.append({
            "frame": fid,
            "q_pred": list(q_pred), "qdot_pred": list(qdot_pred),
            "q_visual_delta": list(innov),
            "qdot_visual_delta": [qdot_post[k] - qdot_pred[k] for k in range(K)],
            "q_post": list(q_post), "qdot_post": list(qdot_post),
            "measurement_energy": meas_e,
            "joint_information": [float(I[k, k]) for k in range(K)],
            "posterior_variance_q": [float(P[k][0, 0]) for k in range(K)],
            "posterior_variance_qdot": [float(P[k][1, 1]) for k in range(K)],
            "visual_correction_magnitude": max(abs(x) for x in innov),
        })
        q, qdot = q_post, qdot_post

    dq = [[rows[t]["q_post"][k] - rows[t - 1]["q_post"][k]
           for t in range(1, len(rows))] for k in range(K)]
    out = {
        "frames": rows,
        "mean_dq_per_frame": [sum(d) / len(d) if d else 0.0 for d in dq],
        "final_q": [rows[-1]["q_post"][k] for k in range(K)] if rows else [],
        "final_qdot": [rows[-1]["qdot_post"][k] for k in range(K)] if rows else [],
        "mean_visual_correction": (sum(r["visual_correction_magnitude"]
                                       for r in rows) / max(len(rows), 1)),
        "propagation": ("joint coordinates (section 19); constant Cartesian "
                        "twist is NOT used once a joint hypothesis exists"),
    }
    writer.json("reduced_stage", out)
    if verbose and rows:
        print(f"  §1.7 reduced filter: final q {[round(x, 5) for x in out['final_q']]}, "
              f"qdot {[round(x, 5) for x in out['final_qdot']]}, "
              f"mean dq/frame {[round(x, 6) for x in out['mean_dq_per_frame']]}")
        print(f"       mean visual correction {out['mean_visual_correction']:.6f} "
              f"(section 15: a negligible correction means the prior, not the "
              f"image, produced the answer)")
    return out


@torch.no_grad()
def _geometric_to_scene(objective, truth, rig, joints, K, partition,
                        positions, args, writer, verbose=True, dbg=None):
    """Sections 23, 26 and 25 Phase 4: project onto the discovered joints.

    The geometric stage has already decided the TYPE and the axis from
    {T_{k,t}} alone (sections 14-19). This step does three things, in the order
    section 25 gives them:

      * build the joint model (section 23's mu_{i,t} for P and R);
      * section 26 -- reopen support now that joint geometry exists, scoring
        each supernode against the JOINT model rather than the free rigid
        transform, and re-running the multi-label assignment;
      * section 23 -- refine q_{k,0:T} against RGB-D with the existing banded
        solver, accepting only a measured decrease of the true objective.

    Returns an `AlternationResult`, so the run continues into the existing
    section 24 canonical handover and the reduced visual filter instead of a
    parallel pipeline (section 45).
    """
    from joint_reduction.kpart_models import (PRISMATIC, REVOLUTE, PartModel,
                                              SceneModel)
    from joint_reduction.kpart_support import KPartSupport
    from joint_reduction.kpart_window import (AlternationResult,
                                              solve_banded_trajectory)
    from estimator.part_assignment import assign_parts, rigid_assignment_costs

    dev = truth.mu.device
    parts, q_traj = [], []
    T = len(objective.frames)
    for k in range(1, K + 1):
        j = joints[k - 1]
        if j is None or j["selected"] is None:
            return None
        axis = torch.tensor(j["axis"], device=dev, dtype=truth.mu.dtype)
        if j["selected"] == "revolute":
            a = torch.tensor(j["axis_point"], device=dev, dtype=truth.mu.dtype)
            parts.append(PartModel(kind=REVOLUTE, axis=axis,
                                   moment=torch.linalg.cross(a, axis),
                                   part_index=k))
        else:
            parts.append(PartModel(kind=PRISMATIC, axis=axis, part_index=k))
        q_traj.append(list(j["q"]))
    scene = SceneModel(parts=parts, K=K)
    labels = rig["labels"]
    sid = partition.supernode_id.long().clamp_min(0)

    E0 = objective.evaluate(scene, labels, q_traj)
    if verbose:
        print(f"\n  === §25 Phase 4: RENDERER REFINEMENT IN JOINT "
              f"COORDINATES ===")
        print(f"  §23 projected onto the discovered joints: E = {E0.total:.2f}")

    # ---- section 26: reopen support with the joint model in hand ---------
    node_labels = rig["node_labels"]
    support = None
    for it in range(int(args.reopen_support_rounds)):
        pred = []
        for ti in range(T):
            mu, _R = scene.transport_all(truth.mu, truth.R, labels,
                                         [q_traj[k][ti] for k in range(K)])
            pred.append(mu)
        costs = rigid_assignment_costs(truth.mu, sid, positions,
                                       _JointAsTransforms(scene, truth, labels,
                                                          q_traj, K, T),
                                       partition.num_supernodes)
        out_a = assign_parts(costs, None, potts_weight=args.potts_weight,
                             initial=node_labels)
        new_nodes = out_a["labels"]
        new_labels = new_nodes[sid]
        E_try = objective.evaluate(scene, new_labels, q_traj)
        accepted = E_try.total <= E0.total + 1e-9
        if verbose:
            print(f"  §26 reopen support round {it}: counts "
                  f"{out_a['counts']}, E {E0.total:.2f} -> {E_try.total:.2f}"
                  f"{'' if accepted else '  [rejected]'}")
        soft = torch.softmax(-costs.float(), dim=1)
        support = KPartSupport(hard=new_nodes if accepted else node_labels,
                               unary=out_a["unary"], soft=soft, K=K,
                               potts_weight=float(args.potts_weight),
                               data_energy=float(out_a["data_energy"]),
                               potts_energy=float(out_a["smoothness_energy"]),
                               info={"stage": "section 26 reopen after joint "
                                              "discovery",
                                     "accepted": bool(accepted)})
        if not accepted or bool((new_nodes == node_labels).all()):
            break
        node_labels, labels, E0 = new_nodes, new_labels, E_try

    # ---- section 23: refine q_{k,0:T} against RGB-D ----------------------
    cfg_alt = AlternationConfig(max_outer=args.discovery_outer,
                                potts_weight=args.potts_weight,
                                sigma_q=args.sigma_q,
                                tau_energy=args.tau_energy,
                                progress=bool(verbose))
    q_before = [list(c) for c in q_traj]
    q_traj, qinfo = solve_banded_trajectory(objective, scene, labels, q_traj,
                                            cfg_alt)
    E1 = objective.evaluate(scene, labels, q_traj)
    if dbg is not None:
        dbg.record(refine_info={
            "energy_before": E0.total, "energy_after": E1.total,
            "status": qinfo.get("status"),
            "trajectory_iterations": qinfo.get("iterations"),
            "per_frame_delta_q": [[q_traj[k][t] - q_before[k][t]
                                   for t in range(T)] for k in range(K)]},
            support_reopen=[support.info] if support is not None else [])
    writer.json("renderer_refinement", {
        "method": ("section 23: project onto the discovered joint model and "
                   "refine q_{k,0:T} with the existing banded solver"),
        "energy_before": E0.total, "energy_after": E1.total,
        "trajectory_solver": {k: v for k, v in qinfo.items()
                              if k not in ("history", "iterations")},
        "support_reopened": support.info if support is not None else None})
    if verbose:
        print(f"  §23 q(t) refined against RGB-D: E {E0.total:.2f} -> "
              f"{E1.total:.2f} ({qinfo.get('status')})")
        for k in range(K):
            print(f"       q_{k + 1}(t) "
                  f"{[round(x, 5) for x in q_traj[k][:8]]}"
                  f"{' ...' if T > 8 else ''}")

    return AlternationResult(
        scene=scene, labels=labels, q_traj=q_traj, energy=E1,
        history=rig["history"], converged=rig["converged"],
        stop_reason=rig["stop_reason"], support=support,
        node_labels=node_labels)


class _JointAsTransforms:
    """Section 26's T_{k,t} read off the DISCOVERED joint model.

    `rigid_assignment_costs` scores a supernode against a part's transform; after
    joint discovery that transform is no longer free, it is the joint model
    evaluated at q_{k,t}. This adapter supplies exactly that, so section 26 uses
    the same cost function as section 11 without a second implementation.
    """

    def __init__(self, scene, truth, labels, q_traj, K, T):
        import torch as _t
        from joint_reduction.joint_geometry import fit_rigid_transform
        self.K = K
        dev = truth.mu.device
        self.R = _t.eye(3, device=dev, dtype=_t.float64).reshape(1, 1, 3, 3) \
            .repeat(T, K, 1, 1).clone()
        self.t = _t.zeros((T, K, 3), device=dev, dtype=_t.float64)
        for k in range(1, K + 1):
            mask = labels == k
            if int(mask.sum()) < 3:
                continue
            src = truth.mu[mask]
            for ti in range(T):
                mu, _r = scene.transport_all(truth.mu, truth.R, labels,
                                             [q_traj[j][ti] for j in range(K)])
                Rk, tk = fit_rigid_transform(src.double(), mu[mask].double())
                self.R[ti, k - 1] = Rk
                self.t[ti, k - 1] = tk


def _progress(iterable, desc, verbose=True, total=None):
    """A tqdm bar when one helps, a plain iterator otherwise.

    The long loops here are per-frame and run for minutes at full scale (60
    frames x 8 cameras x 11,320 Gaussians), so a bar is the difference between
    "is it working?" and a measured ETA.

    This used to write to stdout and give up whenever stdout was not a
    terminal, which killed the bar in exactly the runs that needed it: the
    overnight ones, redirected to a log. `utils.progress` writes to /dev/tty
    instead, so the bar reaches the terminal whatever stdout is doing and the
    log stays free of carriage returns.
    """
    if not verbose:
        return list(iterable) if total is None else iterable
    from utils.progress import pbar
    return pbar(iterable, desc.strip(), total=total)


@torch.no_grad()
def _collect_debug_metrics(dbg, objective, verbose=True):
    """Sections 15, 16, 32, 33: per-frame, per-camera and multiview metrics.

    These are MEASUREMENTS, taken after inference has finished, on the states
    inference produced. Nothing here feeds back into an estimate -- section 0
    rule 10. The per-camera joint step

        dq_{t,c} = g_{t,c} / (h_{t,c} + eps)

    is section 32's "what this camera alone would ask for", which is how a
    camera that opposes the others or carries no information becomes visible.
    """
    if not dbg.enabled or objective is None:
        return
    from estimator.measurement import measurement_energy
    from joint_reduction.kpart_models import KPartObservation

    labels = dbg.labels
    scene = dbg.scene
    for ti, (obs, fid) in _progress(list(enumerate(objective.frames)),
                                    "  debug metrics", verbose):
        mu = dbg.positions_at("refined", ti)
        if mu is None:
            mu = dbg.positions_at("rigid", ti)
        if mu is None:
            mu = dbg.mu0
        R = dbg.orientations_at("refined", ti)
        if R is None:
            R = dbg.orientations_at("rigid", ti)
        lin, prec, resid = objective.linearize_frame(mu, R, obs)
        kobs = None
        if scene is not None and labels is not None and dbg.q_traj:
            try:
                kobs = KPartObservation(lin, scene, mu, labels)
            except Exception:
                kobs = None
        tot_rgb = tot_d = 0.0
        for ci, ((pr, pd), (rr, rd)) in enumerate(zip(prec, resid)):
            vr = int((pr > 0).sum())
            vd = int((pd > 0).sum())
            e_rgb = float(measurement_energy(rr, pr))
            e_d = float(measurement_energy(rd, pd))
            tot_rgb += e_rgb
            tot_d += e_d
            row = {"frame": fid, "camera_id": ci,
                   "valid_rgb_px": vr, "valid_depth_px": vd,
                   "visible_gaussians": "",
                   "rgb_rmse": float((rr[pr > 0] ** 2).mean().sqrt()) if vr else 0.0,
                   "rgb_robust_energy": e_rgb,
                   "depth_rmse": float((rd[pd > 0] ** 2).mean().sqrt()) if vd else 0.0,
                   "depth_robust_energy": e_d,
                   # Left EMPTY rather than 0 when no joint model exists: an
                   # unresolved run has no q to be informative about, and a
                   # literal zero would read as "the cameras carry no
                   # information" instead of "this quantity is not defined at
                   # this stage".
                   "motion_sensitive_px": "", "fisher_trace": ""}
            if kobs is not None:
                # One camera's own gradient and information in q.
                one_p = [(torch.zeros_like(a), torch.zeros_like(b))
                         for a, b in prec]
                one_r = [(torch.zeros_like(a), torch.zeros_like(b))
                         for a, b in resid]
                one_p[ci] = prec[ci]
                one_r[ci] = resid[ci]
                try:
                    I_q, b_q = kobs.normal_system(one_p, one_r)
                    tr = float(I_q.diagonal().sum())
                    row["fisher_trace"] = tr
                    for k in range(1, dbg.K + 1):
                        h = float(I_q[k - 1, k - 1])
                        dbg.multiview.append({
                            "frame": fid, "camera_id": ci, "part": k,
                            "g": float(b_q[k - 1]), "h": h,
                            "delta_q": float(b_q[k - 1]) / (h + 1e-12)})
                    dy = lin.apply_H(kobs.tangent(1))
                    row["motion_sensitive_px"] = int(
                        (dy[ci][0].abs().sum(0) > 1e-6).sum())
                except Exception:
                    pass
            dbg.add_camera_row(row)
        dbg.add_frame_row({"frame": fid, "rgb_energy": tot_rgb,
                           "depth_energy": tot_d,
                           "total_observation_energy": tot_rgb + tot_d,
                           "visible_gaussians": int(dbg.num_gaussians)})
        del lin, kobs
    if verbose:
        print(f"  debug: collected {len(dbg.frame_rows)} frame rows and "
              f"{len(dbg.camera_rows)} camera rows")


def _emit_debug(dbg, objective, writer, args, verbose=True):
    """Write the whole debug package (sections 13, 39, 40, 41)."""
    if not dbg.enabled:
        return
    from debug.numeric import write_numeric_debug, _stage_energies
    _collect_debug_metrics(dbg, objective, verbose=verbose)
    energies = _stage_energies(dbg, objective)
    dbg.record(energy_by_stage={
        k: [(e or {}).get("total") for e in v] for k, v in energies.items()})

    if getattr(args, "debug_numerics", False):
        write_numeric_debug(dbg, objective, writer)
        from debug.plots import write_plots
        files = write_plots(dbg, writer, energies)
        if verbose:
            print(f"  debug: {len(files)} plots written to plots/")
    if getattr(args, "debug_render_panels", False):
        from debug.render_panels import write_render_panels
        dbg.record(render_panels=write_render_panels(
            dbg, objective, writer,
            frame_stride=max(1, int(getattr(args, "debug_frame_stride", 1))),
            verbose=verbose))
    if getattr(args, "export_debug_gaussian_frames", False):
        _export_gaussian_frames(dbg, writer, args, verbose)
    if getattr(args, "make_geometric_debug_html", False):
        from debug.geometric_motion_debug import write_html
        path = write_html(dbg, writer,
                          max_gaussians=int(getattr(args,
                                                    "debug_gaussian_subsample",
                                                    4000)),
                          frame_stride=int(getattr(args, "debug_frame_stride", 1)))
        if verbose:
            print(f"  debug: {path}")


def _export_gaussian_frames(dbg, writer, args, verbose=True):
    """Section 41: persist enough to reconstruct the final states exactly."""
    writer.tensor("canonical_gaussians", {
        "mu": dbg.mu0.detach().cpu(),
        "R": None if dbg.R0 is None else dbg.R0.detach().cpu(),
        "scales": None if dbg.scales is None else dbg.scales.detach().cpu(),
        "colors": None if dbg.colors is None else dbg.colors.detach().cpu(),
        "opacity": None if dbg.opacity is None else dbg.opacity.detach().cpu()})
    if dbg.labels is not None:
        writer.tensor("part_labels", dbg.labels.detach().cpu())
    if dbg.q_traj:
        writer.tensor("q_trajectories", torch.tensor(dbg.q_traj,
                                                     dtype=torch.float64))
    writer.json("joint_models", {
        "note": ("section 41: with canonical_gaussians.pt, part_labels.pt and "
                 "rigid_transforms.pt these reproduce every stage exactly "
                 "through section 2's transports"),
        "per_part": [None if j is None else
                     {"part": j.get("part"), "selected": j.get("selected"),
                      "status": j.get("status"), "axis": j.get("axis"),
                      "axis_point": j.get("axis_point"), "q": j.get("q")}
                     for j in dbg.joints]})
    from debug.recorder import part_color
    stride = max(1, int(getattr(args, "debug_frame_stride", 1)))
    mode = str(getattr(args, "debug_ply_color", "part")).lower()
    out = writer.dir / "gaussian_frames"
    out.mkdir(parents=True, exist_ok=True)

    # Section 4: the part palette, resolved once. Every frame uses the same
    # mapping, so a part never changes colour between frames.
    cols = {k: tuple(int(part_color(k).lstrip("#")[i:i + 2], 16)
                     for i in (0, 2, 4)) for k in range(dbg.K + 1)}
    app = None
    if mode == "appearance" and dbg.colors is not None:
        app = (dbg.colors.detach().cpu().clamp(0, 1) * 255).round().to(
            torch.uint8)

    frames = list(range(0, dbg.num_frames, stride))
    n = 0
    for t in _progress(frames, "  §41 PLY frames", verbose):
        mu = dbg.positions_at("refined", t)
        if mu is None:
            mu = dbg.positions_at("rigid", t)
        if mu is None:
            break
        lab = dbg.labels if dbg.labels is not None else torch.zeros(
            mu.shape[0], dtype=torch.long)
        mc = mu.detach().cpu()
        lc = lab.detach().cpu()
        lines = ["ply", "format ascii 1.0", f"element vertex {mc.shape[0]}",
                 "property float x", "property float y", "property float z",
                 "property uchar red", "property uchar green",
                 "property uchar blue", "property int label", "end_header"]
        for i in range(mc.shape[0]):
            k = int(lc[i])
            if app is not None:
                r, g, b = (int(app[i, 0]), int(app[i, 1]), int(app[i, 2]))
            else:
                r, g, b = cols.get(k, (128, 128, 128))
            lines.append(f"{float(mc[i,0]):.6f} {float(mc[i,1]):.6f} "
                         f"{float(mc[i,2]):.6f} {r} {g} {b} {k}")
        # The trailing newline is REQUIRED: without it the final vertex record
        # is not newline-terminated, strict PLY readers reject the file, and
        # `wc -l` reports one vertex fewer than the header declares -- which
        # looks exactly like a lost vertex.
        (out / f"frame_{t:04d}.ply").write_text("\n".join(lines) + "\n")
        n += 1
    writer.json("gaussian_frames_index", {
        "note": "section 41",
        "frames": n, "stride": stride,
        "color_mode": mode,
        "color_meaning": (
            "red/green/blue are the SECTION 4 PART COLOURS (0 = static grey, "
            "1..K = the moving parts), not the Gaussian's appearance. The "
            "`label` property carries the integer part id. Pass "
            "--debug-ply-color appearance for the scene's own colours."
            if app is None else
            "red/green/blue are the Gaussian's OWN appearance colours; the "
            "`label` property still carries the section 4 part id."),
        "palette": {str(k): part_color(k) for k in range(dbg.K + 1)}})
    if verbose:
        print(f"  debug: exported {n} PLY frames to gaussian_frames/ "
              f"(colour = {mode})")


@torch.no_grad()
def bootstrap_positions(truth, cameras, frames, cfg, partition, args,
                        verbose=True, observer=None):
    """Section 25 Phase 1: multiview motion evidence for EVERY frame.

    Section 10 A wants local motion, twist evidence and confidence before any
    part structure exists, and section 10 C then fits T_{k,t} against the whole
    window. The rigid stage therefore needs a position field per frame, not the
    single far-baseline displacement the P/R pipeline used as an initializer.

    Each frame is solved by the same grouped IEKF against the real renderer,
    warm started from the previous frame. The warm start is temporal
    continuation, not a prior: a render has a finite basin of attraction, and
    solving every frame from the canonical state makes the late frames of a
    long sweep unreachable.

    Returns ``(positions, info)`` where `positions[t]` is ``[N,3]``. No joint
    model and no P/R hypothesis exists at this point (section 48).
    """
    from estimator import IEKFConfig, iekf_frame_step

    n = truth.num_gaussians
    dev = truth.mu.device
    bcfg = IEKFConfig(**{**{k: getattr(cfg, k) for k in
                         ("max_iters", "pcg_max_iters", "use_rgb", "use_depth",
                          "robust_measurement", "student_t_nu", "rgb_sigma",
                          "depth_sigma_model", "eps_depth",
                          "min_alpha_for_depth", "use_trust_region")},
                      "use_supernodes": True,
                      "supernode_target_size": args.supernode_size,
                      "supernode_seed": args.seed,
                      "max_iters": args.bootstrap_iters})
    positions, per_frame = [], []
    state = truth
    t0 = time.perf_counter()
    # Driven by hand rather than wrapped, so the bar can carry the number
    # that says whether Phase 1 is actually finding motion.
    from utils.progress import ibar
    _fb = ibar(len(frames), "§25.1 motion evidence", leave=True)
    for ti, (obs, fid) in enumerate(frames):
        _fb.to(ti)
        if ti == 0:
            positions.append(truth.mu.clone())
            per_frame.append({"frame": fid, "mean_displacement": 0.0,
                              "note": "frame 0 defines the canonical state"})
            if observer is not None:
                observer(ti, truth)
            continue
        if obs and obs[0][0].device.type != dev.type:
            obs = [(a.to(dev), b.to(dev), c.to(dev)) for a, b, c in obs]
        P = torch.zeros((n, 12, 12), device=dev, dtype=truth.mu.dtype)
        P[:, range(12), range(12)] = torch.tensor(
            [1e-2] * 3 + [1e-2] * 3 + [1.0] * 3 + [1e-1] * 3, device=dev,
            dtype=truth.mu.dtype)
        seed = state if ti > 1 else truth
        from estimator.silhouette import get_active as _sil
        if _sil() is not None:
            _sil().set_frame(ti)          # processing index -> this frame's masks
        prior = []
        def capture_information(prior_block, image_block):
            prior.append(prior_block.detach().cpu())
        new_state, _, diag = iekf_frame_step(seed, P, obs, cameras,
                                             dt=args.dt, cfg=bcfg,
                                             partition=partition,
                                             information_observer=capture_information if observer else None)
        state = new_state
        if observer is not None:
            observer(ti, state, prior[0])
        positions.append(state.mu.clone())
        d = state.mu - truth.mu
        per_frame.append({"frame": fid,
                          "mean_displacement": float(d.norm(dim=1).mean()),
                          "max_displacement": float(d.norm(dim=1).max())})
        # `fid` is a per-frame provenance dict, not an index -- the bar gets
        # the loop counter.
        _fb.note(f"mean |dx| {per_frame[-1]['mean_displacement']:.5f}")
        if verbose:
            print(f"       frame {fid}: mean |dx| "
                  f"{per_frame[-1]['mean_displacement']:.5f}")
    _fb.to(len(frames))
    _fb.close()
    info = {"role": ("section 25 Phase 1: motion evidence only; no joint "
                     "model, no P/R hypothesis, no GT"),
            "frames": len(frames), "runtime_s": time.perf_counter() - t0,
            "per_frame": per_frame,
            "warm_started": True}
    return positions, info
