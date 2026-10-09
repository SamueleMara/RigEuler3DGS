"""Fixed-prior iterated EKF over the sparse alpha-aware renderer.

CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md sections 13-15, 22, 24, 25.

The loop below is section 24's reference algorithm, implemented with its exact
semantics. The parts that are easy to get subtly wrong, and why they are the
way they are:

  * The prediction happens ONCE per video frame (section 14). Re-predicting
    inside the inner iterations would count the same RGB-D frame as several
    independent measurements and shrink the covariance repeatedly.

  * ``Lambda_prior`` and ``x_prior`` are captured before the loop and never
    touched again, so every inner iteration solves
        (Lambda^- + H_j^T R^-1 H_j + lambda_j D_j) Delta_j
            = H_j^T R^-1 r_j - Lambda^- delta_j
    which is section 14's damped step, NOT a fresh Kalman update.

  * A step is accepted only if the TRUE nonlinear MAP objective decreases
    (section 14). The linearized model's prediction of the decrease is not
    trusted, because the renderer is only piecewise smooth.

  * After an accepted step the scene is re-rendered, re-sorted and
    re-linearized (section 0 rule 10). Inside one PCG solve nothing changes
    (section 20.3), which is what makes conjugacy valid.

No Adam, no autograd, no loss.backward() anywhere in this file (section 36).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from .convergence import (
    ConvergenceCertificate,
    IterationRecord,
    quadratic_model_decrease,
)
from .covariance import (
    apply_prior,
    approximate_posterior_information,
    invert_blocks,
    project_psd,
)
from .measurement import build_precision, measurement_energy
from .observability import (
    analyze_blocks,
    inflate_weak_covariance,
    weak_direction_policy,
)
from .pcg import block_jacobi_preconditioner, pcg
from .radiance import resolve_radiance_mode
from .trust_region import (
    STOP_FORCING,
    ScalingMetric,
    TrustRegionConfig,
    TrustRegionHistory,
    TrustRegionRecord,
    adapt_radius,
    forcing_eta,
    predicted_reduction,
    steihaug_cg,
)
from .renderer_bridge import (
    CameraSpec,
    FrameLinearization,
    build_linearization,
    embed_pose_to_full_state,
    extract_pose,
)
from .state import GaussianMotionState, boxminus, boxplus

STATE_DIM = 12
POSE_DIM = 6


@dataclass
class IEKFConfig:
    """Section 33's controls, with the spec's defaults."""

    max_iters: int = 10
    lm_initial: float = 1e-3
    lm_min: float = 1e-8
    lm_max: float = 1e8
    lm_accept_scale: float = 0.33
    lm_reject_scale: float = 4.0
    line_search_shrink: float = 0.5
    alpha_min: float = 1e-3
    pcg_max_iters: int = 30
    pcg_tol: float = 1e-4
    use_rgb: bool = True
    use_depth: bool = True
    robust_measurement: bool = True
    student_t_nu: float = 5.0
    rgb_sigma: float = 0.1
    depth_sigma_model: str = "constant"
    depth_sigma_a: float = 0.02
    depth_sigma_b: float = 0.0
    # Section 33's process-noise controls. These are the weak constant-twist
    # prior of section 5: unmodelled acceleration is absorbed here.
    process_noise_pos: float = 1e-3
    process_noise_rot: float = 1e-3
    process_noise_vel: float = 1e-1
    process_noise_omega: float = 1e-1
    eps_depth: float = 1e-6
    antialiasing: bool = False
    # Section 27: a pixel with no coverage carries no depth measurement. This
    # is a validity gate, not a tuning weight.
    min_alpha_for_depth: float = 0.05
    # Section 3: appearance is frozen during motion estimation. The flag does
    # not CHOOSE whether to freeze (the estimator never writes appearance); it
    # turns on a runtime assertion that nothing did, which is what makes the
    # guarantee checkable rather than merely intended.
    frozen_appearance: bool = True
    strict_frozen_radiance: bool = True
    # Section 25: fuse every synchronized camera. False restricts the update to
    # the first camera, which is the controlled ablation for information gain.
    multiview: bool = True
    # Section 35 T8: assert <Hv,z> == <v,H^T z> on the live linearization each
    # frame. Cheap (one extra JVP/VJP) and catches a mis-built cache in place.
    debug_adjoint_test: bool = False
    adjoint_tolerance: float = 1e-3
    # V2 section 13 Option B: Mode B (precomputed pose-basis derivatives) is
    # NOT implemented and its CLI flag has been removed from production, since
    # "raises if enabled" is not a feature. Mode A -- recomputing the
    # differentials inside each JVP/VJP -- is the validated and only path.
    # The field survives so an explicit request still fails loudly rather than
    # being silently ignored by a caller that sets it programmatically.
    cache_derivative_basis: bool = False
    rel_energy_tol: float = 1e-6
    step_norm_tol: float = 1e-7
    debug: bool = False

    # ---- V2 section 4: explicit scaled trust region --------------------
    # The default is ON. Section 1.3 showed the LM + line-search path only made
    # progress at a 25-iteration PCG budget and accepted ZERO steps at 100 or
    # 400 -- the linear-solver budget was acting as the step controller. The LM
    # path is kept only so that failure remains reproducible as an ablation.
    use_trust_region: bool = True
    trust_radius_init: float = 1.0
    trust_radius_max: float = 100.0
    trust_radius_min: float = 1e-8
    trust_eta_accept: float = 0.1
    trust_eta_weak: float = 0.25
    trust_eta_good: float = 0.75
    # Section 4.1's scaling metric: the trust radius is measured in units of
    # these prior sigmas, so "radius 1" means the same physical thing for a
    # translation and for a rotation.
    scale_sigma_mu: float = 0.05
    scale_sigma_phi: float = 0.05
    scale_sigma_v: float = 0.5
    scale_sigma_omega: float = 0.5
    scale_from_prior: bool = False
    # Section 4.3's inexact-Newton forcing rule. Section 4.3 allows the exact
    # formula to be tuned provided it is explicit and persisted, and it is
    # persisted in every trust-region record as `forcing_eta`.
    #
    # eta_max is deliberately TIGHT (0.01, not the 0.5 the formula's example
    # suggests). The loose-far-from-the-solution argument assumes an accurate
    # linear solve is expensive to afford; before the trust region it was worse
    # than expensive, it was destructive -- a 400-iteration PCG budget drove
    # accepted steps to zero (section 1.3). With the radius controlling the
    # step, a tight solve is simply safe, and it measurably matters: on the
    # multi-frame constant-twist test, eta_max 0.5 tracked at 1.6e-3 mean
    # position error while eta_max 0.01 reached 3.6e-4, matching the exact
    # solve. The looser setting remains available and is what `forcing_max`
    # selects.
    forcing_c: float = 0.5
    forcing_min: float = 1e-6
    forcing_max: float = 0.01

    # ---- V2 section 3: grouped / supernode state -----------------------
    use_supernodes: bool = False
    supernode_target_size: int = 64
    supernode_count: int | None = None
    supernode_knn: int = 8
    supernode_seed: int = 0

    # ---- V2 section 6: support inference before joint reduction --------
    # On a real scene only part of the cloud moves, so the joint must be fitted
    # to the moving supernodes only. These control section 6.3 step 1's
    # robust threshold on the alpha-aware achievable-motion gain.
    support_k_mad: float = 8.0
    support_max_moving_fraction: float = 0.5

    # ---- V2 section 5: observability-aware update ----------------------
    observability_tau: float = 1e-2
    project_weak_directions: bool = False
    inflate_weak_posterior: bool = True
    observability_report: bool = True

    # ---- V2 section 12: radiance mode ----------------------------------
    # strict_frozen_radiance above selects the mode; this acknowledges the
    # unvalidated view-dependent one deliberately (section 12.2).
    allow_unvalidated_view_dependent: bool = False


@dataclass
class IEKFDiagnostics:
    """Section 34's per-iteration log."""

    rows: list[dict] = field(default_factory=list)
    certificate: "ConvergenceCertificate | None" = None
    # V2 section 4.5 / 5.3: the trust-region and observability evidence for
    # this frame, kept alongside the raw rows so a caller can persist them
    # without re-parsing the log.
    trust_region: "TrustRegionHistory | None" = None
    observability: "object | None" = None

    def add(self, **kw) -> None:
        self.rows.append(dict(kw))

    def last(self) -> dict:
        return self.rows[-1] if self.rows else {}


def _precisions(views, rgbd_gt, cfg: IEKFConfig):
    """Per-camera (RGB precision, depth precision, RGB residual, depth residual).

    Sections 6 and 13.1. RGB and depth are built separately because they are
    different sensors with different noise models, and because section 27 wants
    depth to dominate early geometry through calibrated R^-1 rather than through
    an optimizer coefficient.
    """
    out = []
    for view, (rgb_gt, depth_gt, depth_valid) in zip(views, rgbd_gt):
        r_rgb = rgb_gt - view.rgb
        if cfg.use_rgb:
            p_rgb = build_precision(
                r_rgb, valid_mask=None, sigma_rgb=cfg.rgb_sigma,
                robust=cfg.robust_measurement, nu=cfg.student_t_nu,
                layout=("rgb", "rgb", "rgb"), channel_dim=0)
        else:
            p_rgb = torch.zeros_like(r_rgb)

        r_d = depth_gt - view.depth
        if cfg.use_depth:
            # Section 6: an invalid / hole depth gets EXACTLY zero precision,
            # never a fake depth value. Section 27 adds the coverage gate -- a
            # pixel the renderer barely covers has no depth prediction to
            # correct, and D = Z/(A+eps) is ill-conditioned there.
            valid = depth_valid & (view.alpha > cfg.min_alpha_for_depth)
            # sigma_rgb=() because this call has a depth-only layout; the
            # module rejects an RGB sigma it has no RGB channel for, which is
            # the right behaviour rather than silently ignoring it.
            p_d = build_precision(
                r_d.unsqueeze(0), valid_mask=valid.unsqueeze(0),
                sigma_rgb=(),
                # A "kind:a:b" string already carries its coefficients, so it
                # is forwarded as-is; a bare kind takes the config's a/b.
                depth_sigma_model=(cfg.depth_sigma_model
                                   if ":" in str(cfg.depth_sigma_model)
                                   else {"kind": cfg.depth_sigma_model,
                                         "a": cfg.depth_sigma_a,
                                         "b": cfg.depth_sigma_b}),
                depth_range=depth_gt.unsqueeze(0),
                robust=cfg.robust_measurement, nu=cfg.student_t_nu,
                layout=("depth",), channel_dim=0).squeeze(0)
        else:
            p_d = torch.zeros_like(r_d)
        out.append((p_rgb, p_d, r_rgb, r_d))
    return out


@torch.no_grad()
def group_precond(M_blocks: torch.Tensor, grouped, G, GT) -> Callable:
    """Block-Jacobi preconditioner in the supernode state (section 3.3).

    The per-Gaussian 12x12 normal blocks are reduced to [C,12,12] by the same
    congruence G^T M G that defines the grouped operator, done here by applying
    G^T to M's columns. This is the PRECONDITIONER only: the exact operator
    still runs the full JVP/VJP, so every cross-supernode co-visibility
    coupling survives untouched.
    """
    c = grouped.num_supernodes
    dev, dt = M_blocks.device, M_blocks.dtype
    cols = []
    for k in range(STATE_DIM):
        # Probe every group with the SAME basis direction at once. M is
        # block-diagonal per Gaussian and the reduction sums only within a
        # group, so the groups do not contaminate each other and one pass
        # yields column k of G^T M G for all of them.
        vg = torch.zeros((c, STATE_DIM), device=dev, dtype=dt)
        vg[:, k] = 1.0
        cols.append(GT(torch.einsum("nij,nj->ni", M_blocks, G(vg))))
    G_blocks = torch.stack(cols, dim=2)                       # [C,12,12]
    G_blocks = 0.5 * (G_blocks + G_blocks.transpose(1, 2))
    eye = torch.eye(STATE_DIM, device=dev, dtype=dt)
    trace = G_blocks.diagonal(dim1=1, dim2=2).abs().sum(-1).clamp_min(1e-12)
    G_blocks = G_blocks + (1e-6 * trace / STATE_DIM).reshape(c, 1, 1) * eye
    return block_jacobi_preconditioner(G_blocks)


def evaluate_true_map_energy(
    state: GaussianMotionState,
    x_prior: GaussianMotionState,
    Lambda_prior: torch.Tensor,
    rgbd_gt,
    cameras: Sequence[CameraSpec],
    cfg: IEKFConfig,
    background: torch.Tensor | None = None,
) -> tuple[float, dict, FrameLinearization]:
    """Section 13's J(delta) = prior energy + measurement energy, evaluated on
    the TRUE nonlinear render -- this is what acceptance is judged on."""
    lin = build_linearization(state, cameras, background=background,
                              eps_depth=cfg.eps_depth,
                              antialiasing=cfg.antialiasing, debug=cfg.debug)
    prec = _precisions(lin.views, rgbd_gt, cfg)
    meas = 0.0
    for p_rgb, p_d, r_rgb, r_d in prec:
        meas += float(measurement_energy(r_rgb, p_rgb))
        meas += float(measurement_energy(r_d, p_d))
    delta = boxminus(state, x_prior)
    prior = 0.5 * float((delta * apply_prior(Lambda_prior, delta)).sum())
    return meas + prior, {"measurement": meas, "prior": prior}, lin


def iekf_frame_step(
    state_prev_post: GaussianMotionState,
    P_prev_post: torch.Tensor,
    rgbd_gt,
    cameras: Sequence[CameraSpec],
    dt: float,
    cfg: IEKFConfig | None = None,
    *,
    process_predict: Callable | None = None,
    covariance_predict: Callable | None = None,
    process_noise: torch.Tensor | None = None,
    background: torch.Tensor | None = None,
    partition=None,
    information_observer: Callable | None = None,
) -> tuple[GaussianMotionState, torch.Tensor, IEKFDiagnostics]:
    """One video frame. Section 24's reference algorithm, semantics preserved.

    ``rgbd_gt`` is a sequence, one entry per camera, of
    ``(rgb [3,H,W], depth [H,W], depth_valid [H,W] bool)``.
    ``dt`` is the REAL inter-frame time in seconds (section 0 rule 11).
    """
    cfg = cfg or IEKFConfig()
    if cfg.cache_derivative_basis:
        raise NotImplementedError(
            "cache_derivative_basis selects section 20.2's Mode B (precomputed "
            "pose-basis derivatives), which is not implemented. V2 section 13 "
            "required either implementing it or removing it from the production "
            "CLI; Option B was taken, so --iekf_cache_derivative_basis no "
            "longer exists and Mode B is documented as a future optimization. "
            "Mode A is the validated path.")
    # V2 section 12: the radiance mode is resolved explicitly, so a run always
    # records which of section 12.1 / 12.2 it used.
    radiance_mode = resolve_radiance_mode(
        cfg.strict_frozen_radiance, cfg.allow_unvalidated_view_dependent)
    if not cfg.multiview and len(cameras) > 1:
        cameras = list(cameras)[:1]
        rgbd_gt = list(rgbd_gt)[:1]
    diag = IEKFDiagnostics()
    frozen_before = ((state_prev_post.colors.clone(), state_prev_post.opacity.clone(),
                      state_prev_post.scales.clone()) if cfg.frozen_appearance else None)
    # Section 14/34: the convergence evidence is collected as the loop runs,
    # because the gain ratio needs the quadratic model that the PCG actually
    # solved and cannot be reconstructed afterwards.
    certificate = ConvergenceCertificate()

    # --- 1. Predict ONCE for this video frame (section 14) -----------------
    from .covariance import covariance_predict as _cov_pred
    from .covariance import make_process_noise
    from .process import error_state_transition
    from .process import process_predict as _proc_pred

    _proc = process_predict or _proc_pred
    _cov = covariance_predict or _cov_pred
    x_prior = _proc(state_prev_post, dt)
    F = error_state_transition(state_prev_post, dt, mode="so3")
    if process_noise is None:
        process_noise = make_process_noise(
            state_prev_post.num_gaussians, dt,
            cfg.process_noise_pos, cfg.process_noise_rot,
            cfg.process_noise_vel, cfg.process_noise_omega,
            device=state_prev_post.mu.device, dtype=state_prev_post.mu.dtype)
    P_prior = _cov(P_prev_post, F, process_noise)
    # The prior information must be PSD or the MAP objective is unbounded below
    # along the offending direction, and "minimizing" it is meaningless.
    P_prior, prior_psd = project_psd(P_prior.double(),
                                     storage_dtype=P_prev_post.dtype)
    Lambda_prior, inv_diag = invert_blocks(P_prior)
    Lambda_prior, lam_psd = project_psd(Lambda_prior,
                                        storage_dtype=P_prev_post.dtype)
    Lambda_prior = Lambda_prior.to(P_prev_post.dtype)
    P_prior = P_prior.to(P_prev_post.dtype)

    # --- 2. The prior is now FIXED for every inner iteration ---------------
    x = x_prior
    lm = float(cfg.lm_initial)
    energy, parts, lin = evaluate_true_map_energy(
        x, x_prior, Lambda_prior, rgbd_gt, cameras, cfg, background)

    # --- 2b. V2 section 4: the scaled trust region -------------------------
    # The metric is built ONCE per frame and never changes inside the loop, so
    # the radius means the same thing at every iteration and the accept/reject
    # history is comparable across them.
    if cfg.scale_from_prior:
        metric = ScalingMetric.from_prior_information(Lambda_prior)
    else:
        metric = ScalingMetric.from_physical_scales(
            (x.num_gaussians, STATE_DIM), x.mu.device, x.mu.dtype,
            sigma_mu=cfg.scale_sigma_mu, sigma_phi=cfg.scale_sigma_phi,
            sigma_v=cfg.scale_sigma_v, sigma_omega=cfg.scale_sigma_omega)
    tr_cfg = TrustRegionConfig(
        radius_init=cfg.trust_radius_init, radius_max=cfg.trust_radius_max,
        radius_min=cfg.trust_radius_min, eta_accept=cfg.trust_eta_accept,
        eta_weak=cfg.trust_eta_weak, eta_good=cfg.trust_eta_good,
        forcing_c=cfg.forcing_c, forcing_min=cfg.forcing_min,
        forcing_max=cfg.forcing_max, max_pcg_iters=cfg.pcg_max_iters)
    # --- 2c. V2 section 3: the canonical supernode partition ---------------
    # Built from CANONICAL geometry (the predicted state's means, which is the
    # frame's reference) and immutable for the whole frame. Passing a partition
    # in keeps it immutable across the whole sequence, which is what section
    # 3.1 actually requires.
    grouped = None
    group_metric = None
    if cfg.use_supernodes:
        from .supernodes import GroupedObservation, build_supernodes
        part = partition if partition is not None else build_supernodes(
            x_prior.mu, target_size=cfg.supernode_target_size,
            num_supernodes=cfg.supernode_count, knn=cfg.supernode_knn,
            seed=cfg.supernode_seed)
        grouped = GroupedObservation(linearization=None, partition=part)
        group_metric = ScalingMetric.from_physical_scales(
            (part.num_supernodes, STATE_DIM), x.mu.device, x.mu.dtype,
            sigma_mu=cfg.scale_sigma_mu, sigma_phi=cfg.scale_sigma_phi,
            sigma_v=cfg.scale_sigma_v, sigma_omega=cfg.scale_sigma_omega)

    radius = float(cfg.trust_radius_init)
    tr_history = TrustRegionHistory()
    g0_norm: float | None = None
    obs_report = None
    # A predicted reduction below this fraction of the objective is round-off,
    # and rho computed from it is the ratio of two noise terms.
    significance_floor = 1e-12

    diag.add(iteration=-1, event="initial", energy=energy, **parts,
             lm=lm, prior_inverse_fallbacks=int(inv_diag.n_fallback),
             prior_jittered=int(inv_diag.n_jittered),
             prior_psd_blocks_floored=prior_psd["blocks_floored"],
             prior_psd_negative_before=prior_psd["blocks_with_negative_eigenvalue"],
             prior_information_psd_floored=lam_psd["blocks_floored"],
             trust_radius=radius, step_control=(
                 "trust_region" if cfg.use_trust_region else "lm_line_search"),
             scaling_metric=metric.to_dict(), radiance_mode=radiance_mode,
             state_parameterization=("supernode_grouped" if grouped is not None
                                     else "free_per_gaussian"),
             solved_dimension=(STATE_DIM * grouped.num_supernodes
                               if grouped is not None
                               else STATE_DIM * x.num_gaussians),
             free_dimension=STATE_DIM * x.num_gaussians)

    # The loop below is where a full-scale frame spends its minutes, and it
    # exits early on convergence, so progress is reported by absolute position
    # rather than by counting iterations -- `break` then leaves the bar honest.
    from utils.progress import ibar
    _bar = ibar(int(cfg.max_iters), "iekf solve")
    for j in range(int(cfg.max_iters)):
        _bar.to(j)
        # --- 3. Nonlinear render + frozen linearization cache --------------
        prec = _precisions(lin.views, rgbd_gt, cfg)
        precisions = [(p_rgb, p_d) for p_rgb, p_d, _, _ in prec]
        residuals = [(r_rgb, r_d) for _, _, r_rgb, r_d in prec]

        # --- 4b. optional T8 self-test on THIS frame's linearization --------
        if cfg.debug_adjoint_test:
            torch.manual_seed(0)
            probe = torch.randn((x.num_gaussians, POSE_DIM), device=x.mu.device,
                                dtype=x.mu.dtype)
            hv = lin.apply_H(probe)
            z = [(torch.randn_like(a), torch.randn_like(b)) for a, b in hv]
            lhs = sum(float((a * zc).sum() + (b * zd).sum())
                      for (a, b), (zc, zd) in zip(hv, z))
            rhs_adj = float((probe * lin.apply_HT(z)).sum())
            rel = abs(lhs - rhs_adj) / max(abs(lhs), abs(rhs_adj), 1e-30)
            diag.add(iteration=j, event="adjoint_test", adjoint_lhs=lhs,
                     adjoint_rhs=rhs_adj, adjoint_rel=rel)
            if rel > cfg.adjoint_tolerance:
                raise AssertionError(
                    f"adjoint identity violated at frame iteration {j}: "
                    f"<Hv,z>={lhs:.6e} vs <v,H^Tz>={rhs_adj:.6e} (rel {rel:.3e}). "
                    f"The linearization cache and the operators disagree.")

        # --- 5. RHS measurement information (section 22) -------------------
        weighted = [(p_rgb * r_rgb, p_d * r_d) for p_rgb, p_d, r_rgb, r_d in prec]
        rhs_pose = lin.apply_HT(weighted)
        rhs = embed_pose_to_full_state(rhs_pose)
        delta_from_prior = boxminus(x, x_prior)
        b = rhs - apply_prior(Lambda_prior, delta_from_prior)

        # --- 6. Block Fisher preconditioner (section 16) -------------------
        fisher_pose = lin.build_block_fisher(precisions)

        # V2 section 5.1: the SAME Fisher, scaled and eigendecomposed, tells us
        # which pose directions this frame actually measures. It costs one
        # 6x6 eigendecomposition per block on a matrix we already have.
        if cfg.observability_report:
            sigma = torch.tensor(
                [cfg.scale_sigma_mu] * 3 + [cfg.scale_sigma_phi] * 3,
                device=fisher_pose.device, dtype=torch.float64)
            obs_report = analyze_blocks(fisher_pose, tau=cfg.observability_tau,
                                        scale=sigma)
            if j == 0:
                diag.add(iteration=j, event="observability",
                         **obs_report.to_dict())

        M_blocks = Lambda_prior.clone()
        M_blocks[:, :POSE_DIM, :POSE_DIM] += fisher_pose
        damp = torch.eye(STATE_DIM, device=M_blocks.device, dtype=M_blocks.dtype)
        M_blocks = M_blocks + lm * damp.expand_as(M_blocks)
        precond = block_jacobi_preconditioner(M_blocks)

        # --- 7. Frozen linearized normal operator (section 15) -------------
        # Never forms H or H^T R^-1 H: the measurement term is three CUDA
        # passes, so cross-Gaussian co-visibility coupling is preserved exactly
        # (section 12) without assembling any off-diagonal block.
        def A_mv(v: torch.Tensor) -> torch.Tensor:
            Hv = lin.apply_H(extract_pose(v))
            weighted_Hv = [(pc * a, pd * d) for (pc, pd), (a, d) in zip(precisions, Hv)]
            meas = embed_pose_to_full_state(lin.apply_HT(weighted_Hv))
            return apply_prior(Lambda_prior, v) + meas + lm * v

        # --- 7b. V2 section 3: solve in the GROUPED state ------------------
        # The free per-Gaussian state is structurally ill-conditioned at scale
        # (section 1.2: 80% weak directions on the full USB scene). Section 3
        # replaces it with 6 DoF per canonical supernode. The operators are
        # unchanged -- the group tangent is expanded to a Gaussian tangent, the
        # validated analytic chain runs, and the adjoint is reduced back -- so
        # this adds a change of variables, it does not fork the measurement
        # model (section 24).
        #
        # Velocity is grouped by the SAME map: a rigid group's velocity field
        # has the identical structure, dv_i = dv_c + domega_c x (mu_i^0 - c_c^0)
        # and domega_i = domega_c, so G acts blockwise on [pose | velocity].
        if grouped is not None:
            def _G(vg: torch.Tensor) -> torch.Tensor:
                return torch.cat([grouped.expand(vg[:, :POSE_DIM]),
                                  grouped.expand(vg[:, POSE_DIM:])], dim=1)

            def _GT(vf: torch.Tensor) -> torch.Tensor:
                return torch.cat([grouped.reduce(vf[:, :POSE_DIM].contiguous()),
                                  grouped.reduce(vf[:, POSE_DIM:].contiguous())],
                                 dim=1)

            solve_b = _GT(b)
            _A_full = A_mv

            def A_solve(v: torch.Tensor) -> torch.Tensor:
                return _GT(_A_full(_G(v)))

            solve_precond = group_precond(M_blocks, grouped, _G, _GT)
            solve_metric = group_metric
            expand_step = _G
        else:
            solve_b, A_solve = b, A_mv
            solve_precond, solve_metric = precond, metric
            expand_step = None

        # ||b|| is the gradient of the true objective at this iterate; it is the
        # quantity that must go to zero for convergence to mean anything.
        grad_norm = float(solve_b.norm())
        energy_before = energy

        # --- 8/9. Step control ---------------------------------------------
        if cfg.use_trust_region:
            # V2 section 4. The step length is set by the radius; the PCG
            # tolerance is set by the forcing rule. Neither is set by the
            # iteration budget, which is the whole point (section 1.3).
            if g0_norm is None:
                g0_norm = grad_norm
            eta = forcing_eta(grad_norm, g0_norm, c=cfg.forcing_c,
                              eta_min=cfg.forcing_min, eta_max=cfg.forcing_max)
            sres = steihaug_cg(A_solve, solve_b, radius, solve_metric,
                               M_inv=solve_precond,
                               max_iters=cfg.pcg_max_iters, eta=eta)
            solve_step = sres.step
            # The radius and the forcing rule act in the space actually solved;
            # the retraction needs the per-Gaussian step.
            step = expand_step(solve_step) if expand_step is not None else solve_step
            info = {"iterations": sres.iterations,
                    "relative_residual": sres.rel_residual,
                    "converged": sres.stop_reason == STOP_FORCING,
                    "reason": sres.stop_reason}

            # Section 5.2: never spend the step on directions the measurement
            # did not constrain.
            obs_info = {}
            if (obs_report is not None and cfg.project_weak_directions
                    and grouped is None):
                pose_step = extract_pose(step)
                projected, obs_info = weak_direction_policy(
                    pose_step, obs_report, project=True)
                step = step.clone()
                step[:, :POSE_DIM] = projected

            # The model decrease is the one the solver actually minimized, so
            # it is evaluated in the solved space.
            predicted = predicted_reduction(solve_step, solve_b, A_solve)
            x_try = boxplus(x, step)
            e_try, parts_try, lin_try = evaluate_true_map_energy(
                x_try, x_prior, Lambda_prior, rgbd_gt, cameras, cfg, background)
            actual = energy_before - e_try
            # Section 4.4's gain ratio. A predicted reduction at round-off is
            # not evidence of anything, so rho is NaN there and adapt_radius
            # rejects -- rather than reporting the ratio of two noise terms,
            # which is how a gain ratio of 2211 was produced before.
            rho = (actual / predicted
                   if predicted > significance_floor * max(abs(energy_before), 1.0)
                   else float("nan"))
            radius_before = radius
            radius, accepted = adapt_radius(radius, rho, sres.boundary_hit,
                                            actual, tr_cfg)
            trial_energy = e_try
            if accepted:
                x, energy, lin, parts = x_try, e_try, lin_try, parts_try
            alpha = 1.0 if accepted else 0.0

            with torch.no_grad():
                t_norm = float(step[:, :3].norm())
                r_norm = float(step[:, 3:6].norm())
            tr_history.add(TrustRegionRecord(
                iteration=j, objective_before=energy_before,
                objective_after=energy if accepted else energy_before,
                predicted_reduction=predicted, actual_reduction=actual,
                rho=rho, trust_radius_before=radius_before,
                trust_radius_after=radius,
                step_norm_scaled=sres.step_norm_scaled,
                step_norm_translation=t_norm, step_norm_rotation=r_norm,
                pcg_stop_reason=sres.stop_reason,
                pcg_iterations=sres.iterations,
                pcg_rel_residual=sres.rel_residual, forcing_eta=eta,
                boundary_hit=sres.boundary_hit, lm_lambda=lm,
                accepted=accepted,
                gradient_norm_scaled=solve_metric.scaled_gradient_norm(solve_b),
                optimality_eta_g=grad_norm / max(g0_norm, 1e-30)))
            extra_diag = {"trust_radius": radius,
                          "trust_radius_before": radius_before,
                          "rho": rho, "forcing_eta": eta,
                          "boundary_hit": sres.boundary_hit,
                          "step_norm_scaled": sres.step_norm_scaled,
                          "predicted_reduction": predicted,
                          "actual_reduction": actual, **obs_info}
        else:
            # Section 1.3's reproducible failure mode, kept only as an ablation.
            solve_step, info = pcg(A_solve, solve_b, solve_precond,
                                   max_iters=cfg.pcg_max_iters, tol=cfg.pcg_tol)
            step = expand_step(solve_step) if expand_step is not None else solve_step
            alpha = 1.0
            accepted = False
            trial_energy = energy
            while alpha >= cfg.alpha_min:
                x_try = boxplus(x, alpha * step)
                e_try, parts_try, lin_try = evaluate_true_map_energy(
                    x_try, x_prior, Lambda_prior, rgbd_gt, cameras, cfg, background)
                if e_try < energy:
                    x, energy, lin, parts = x_try, e_try, lin_try, parts_try
                    lm = max(cfg.lm_min, lm * cfg.lm_accept_scale)
                    accepted = True
                    trial_energy = e_try
                    break
                trial_energy = e_try
                alpha *= cfg.line_search_shrink
            if not accepted:
                lm = min(cfg.lm_max, lm * cfg.lm_reject_scale)
            predicted = quadratic_model_decrease(
                alpha * solve_step if accepted else solve_step, solve_b, A_solve)
            extra_diag = {}

        certificate.add(IterationRecord(
            iteration=j, accepted=accepted,
            energy_before=energy_before,
            energy_after=energy if accepted else energy_before,
            predicted_decrease=predicted,
            actual_decrease=(energy_before - energy) if accepted else 0.0,
            gradient_norm=grad_norm, step_norm=float(step.norm()),
            alpha=(alpha if accepted else 0.0), lm=lm,
            pcg_iterations=int(info.get("iterations", 0) or 0),
            # `x or nan` would turn a legitimate 0.0 residual into NaN.
            pcg_rel_residual=float(info["relative_residual"])
            if "relative_residual" in info else float("nan")))

        with torch.no_grad():
            dmu = step[:, :3].norm(dim=1)
            dphi = step[:, 3:6].norm(dim=1)
        diag.add(iteration=j, event="accepted" if accepted else "rejected",
                 energy=energy, trial_energy=trial_energy, **parts,
                 lm=lm, alpha=(alpha if accepted else 0.0),
                 pcg_iterations=info.get("iterations"),
                 pcg_rel_residual=info.get("relative_residual"),
                 pcg_converged=info.get("converged"),
                 pcg_reason=info.get("reason"),
                 mean_dmu=float(dmu.mean()), max_dmu=float(dmu.max()),
                 mean_dphi=float(dphi.mean()), max_dphi=float(dphi.max()),
                 step_norm=float(step.norm()), gradient_norm=grad_norm,
                 gain_ratio=certificate.records[-1].gain_ratio, **extra_diag)

        _bar.note(f"E {energy:.4e} |g| {grad_norm:.2e} "
                  f"{'accept' if accepted else 'reject'}")

        # --- 10. Convergence (section 14: BOTH criteria) -------------------
        if accepted:
            rel = abs(trial_energy - energy) / max(abs(energy), 1e-30)
            if rel < cfg.rel_energy_tol and float(step.norm()) < cfg.step_norm_tol:
                break
        elif cfg.use_trust_region:
            # Section 4.4: a radius driven to its floor means no step of any
            # length helps -- that is the converged/stalled signal here, the
            # way lm >= lm_max is for the LM path.
            if radius <= tr_cfg.radius_min * 1.0000001:
                break
        elif lm >= cfg.lm_max:
            break

    _bar.close()

    # --- 11. Approximate posterior marginals, ONCE (section 17) ------------
    prec = _precisions(lin.views, rgbd_gt, cfg)
    fisher_pose = lin.build_block_fisher([(p, d) for p, d, _, _ in prec])
    Lambda_post = approximate_posterior_information(Lambda_prior, fisher_pose)
    # Invert in float64 and project back onto the PSD cone. The measurement
    # Fisher spans ~1e6 between a well-seen and an unseen direction, so the
    # block condition number is past what float32 can invert: without this the
    # posterior acquires a negative eigenvalue after a few frames, the next
    # frame's prior information inherits it, and the prior ENERGY goes negative
    # -- which the step control then happily "minimizes".
    P_post, post_diag = invert_blocks(Lambda_post.double())

    # V2 section 5.2: "inflate posterior uncertainty rather than inventing
    # motion". Along a direction the measurement did not constrain, the
    # posterior must not claim to be more certain than the prior was.
    inflate_info: dict = {}
    if cfg.observability_report and cfg.inflate_weak_posterior:
        sigma = torch.tensor(
            [cfg.scale_sigma_mu] * 3 + [cfg.scale_sigma_phi] * 3,
            device=fisher_pose.device, dtype=torch.float64)
        final_report = analyze_blocks(fisher_pose, tau=cfg.observability_tau,
                                      scale=sigma)
        pose_post = P_post[:, :POSE_DIM, :POSE_DIM]
        pose_prior = P_prior[:, :POSE_DIM, :POSE_DIM].double()
        inflated, inflate_info = inflate_weak_covariance(
            pose_post, final_report, prior=pose_prior)
        P_post = P_post.clone()
        P_post[:, :POSE_DIM, :POSE_DIM] = inflated
        diag.add(iteration=-2, event="observability_final",
                 **final_report.to_dict(), **inflate_info)

    # The PSD projection runs LAST, after the inflation. Inflating replaces the
    # 6x6 pose sub-block while leaving the pose-velocity cross-covariance
    # untouched, and [[A',C],[C^T,D]] need not be PSD even when A' and D are --
    # measured on USB, projecting before the inflation still let a -4e-4
    # eigenvalue reach the next frame.
    P_post, psd_diag = project_psd(P_post, storage_dtype=Lambda_post.dtype)
    P_post = P_post.to(Lambda_post.dtype)

    diag.add(iteration=-2, event="posterior",
             posterior_inverse_fallbacks=int(post_diag.n_fallback),
             mean_fisher_trace=float(fisher_pose.diagonal(dim1=1, dim2=2).sum(-1).mean()),
             posterior_psd_blocks_floored=psd_diag["blocks_floored"],
             posterior_psd_negative_before=psd_diag["blocks_with_negative_eigenvalue"],
             posterior_min_eigenvalue_before=psd_diag["min_correlation_eigenvalue_before"],
             posterior_min_eigenvalue_after=psd_diag["min_correlation_eigenvalue_after"],
             **inflate_info)
    if frozen_before is not None:
        c0, o0, s0 = frozen_before
        if not (torch.equal(x.colors, c0) and torch.equal(x.opacity, o0)
                and torch.equal(x.scales, s0)):
            raise AssertionError(
                "section 3 violated: the estimator modified frozen appearance "
                "(colour / opacity / scales) during a motion update")
    diag.certificate = certificate
    diag.trust_region = tr_history if cfg.use_trust_region else None
    diag.observability = obs_report
    if information_observer is not None:
        # Read-only diagnostics; no change to the prediction or update.
        information_observer(Lambda_prior[:, :3, :3], fisher_pose[:, :3, :3])
    return x, P_post, diag


def config_from_args(args) -> IEKFConfig:
    """Build an :class:`IEKFConfig` from the section 33 CLI flags.

    Kept here rather than in arguments/ so the estimator stays importable
    without the training argument parser.
    """
    def g(name, default):
        return getattr(args, name, default)

    return IEKFConfig(
        max_iters=int(g("iekf_max_iters", 10)),
        lm_initial=float(g("iekf_lm_initial", 1e-3)),
        lm_min=float(g("iekf_lm_min", 1e-8)),
        lm_max=float(g("iekf_lm_max", 1e8)),
        line_search_shrink=float(g("iekf_line_search_shrink", 0.5)),
        alpha_min=float(g("iekf_alpha_min", 1e-3)),
        pcg_max_iters=int(g("iekf_pcg_max_iters", 30)),
        pcg_tol=float(g("iekf_pcg_tol", 1e-4)),
        use_rgb=bool(g("iekf_use_rgb", True)),
        use_depth=bool(g("iekf_use_depth", True)),
        robust_measurement=bool(g("iekf_robust_measurement", True)),
        student_t_nu=float(g("iekf_student_t_nu", 5.0)),
        rgb_sigma=float(g("iekf_rgb_sigma", 0.1)),
        depth_sigma_model=str(g("iekf_depth_sigma_model", "constant")),
        process_noise_pos=float(g("iekf_process_noise_pos", 1e-3)),
        process_noise_rot=float(g("iekf_process_noise_rot", 1e-3)),
        process_noise_vel=float(g("iekf_process_noise_vel", 1e-1)),
        process_noise_omega=float(g("iekf_process_noise_omega", 1e-1)),
        eps_depth=float(g("iekf_eps_depth", 1e-6)),
        min_alpha_for_depth=float(g("iekf_min_alpha_for_depth", 0.05)),
        debug=bool(g("iekf_debug_jacobians", False)),
        frozen_appearance=bool(g("iekf_frozen_appearance", True)),
        strict_frozen_radiance=bool(g("iekf_strict_frozen_radiance", True)),
        multiview=bool(g("iekf_multiview", True)),
        debug_adjoint_test=bool(g("iekf_debug_adjoint_test", False)),
        cache_derivative_basis=bool(g("iekf_cache_derivative_basis", False)),
        # ---- V2 section 4 ----------------------------------------------
        use_trust_region=bool(g("iekf_trust_region", True)),
        trust_radius_init=float(g("iekf_trust_radius_init", 1.0)),
        trust_radius_max=float(g("iekf_trust_radius_max", 100.0)),
        trust_radius_min=float(g("iekf_trust_radius_min", 1e-8)),
        trust_eta_accept=float(g("iekf_trust_eta_accept", 0.1)),
        trust_eta_weak=float(g("iekf_trust_eta_weak", 0.25)),
        trust_eta_good=float(g("iekf_trust_eta_good", 0.75)),
        scale_sigma_mu=float(g("iekf_scale_sigma_mu", 0.05)),
        scale_sigma_phi=float(g("iekf_scale_sigma_phi", 0.05)),
        scale_sigma_v=float(g("iekf_scale_sigma_v", 0.5)),
        scale_sigma_omega=float(g("iekf_scale_sigma_omega", 0.5)),
        scale_from_prior=bool(g("iekf_scale_from_prior", False)),
        forcing_c=float(g("iekf_forcing_c", 0.5)),
        forcing_min=float(g("iekf_forcing_min", 1e-6)),
        forcing_max=float(g("iekf_forcing_max", 0.01)),
        # ---- V2 section 3 ----------------------------------------------
        use_supernodes=bool(g("iekf_supernodes", False)),
        supernode_target_size=int(g("iekf_supernode_size", 64)),
        supernode_count=(int(g("iekf_supernode_count", 0)) or None),
        supernode_knn=int(g("iekf_supernode_knn", 8)),
        supernode_seed=int(g("iekf_supernode_seed", 0)),
        # ---- V2 section 5 ----------------------------------------------
        support_k_mad=float(g("iekf_support_k_mad", 8.0)),
        support_max_moving_fraction=float(g("iekf_support_max_moving", 0.5)),
        observability_tau=float(g("iekf_observability_tau", 1e-2)),
        project_weak_directions=bool(g("iekf_project_weak", False)),
        inflate_weak_posterior=bool(g("iekf_inflate_weak", True)),
        observability_report=bool(g("iekf_observability_report", True)),
        # ---- V2 section 12 ---------------------------------------------
        allow_unvalidated_view_dependent=bool(
            g("iekf_allow_view_dependent_radiance", False)),
    )
