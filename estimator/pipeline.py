"""The whole technique, end to end.

CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md section 2's ladder, run as one loop:

    previous posterior
        -> constant-twist prediction with REAL dt
        -> fixed-prior IEKF against the RGB-D frame (sections 14, 15, 24)
        -> filtered per-Gaussian {mu, R, v, omega} + 12x12 marginals
        -> rigid coherence test on the filtered velocities (section 28)
        -> P/R model selection (section 29)
        -> REDUCED joint state (q, qdot); free Gaussian updates STOP (section 30)

The per-Gaussian stage is a bootstrap by design: section 2 says so, and the
observability diagnostics say why -- with N free 6-DoF poses and a handful of
views, most pose blocks carry almost no measurement information, so the prior
dominates them. Reduction is what converts that under-determined problem into
an over-determined one: thousands of pixels constraining a single q.

The transition is guarded. ``decide_reduction`` declines when the twist is not
coherent, when the group is static, or when prismatic and revolute are too
close to call, and the pipeline then simply keeps filtering freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .renderer_bridge import CameraSpec, build_linearization
from .sparse_iekf import IEKFConfig, iekf_frame_step
from .state import GaussianMotionState


@dataclass
class FrameReport:
    index: int
    dt: float
    mode: str                       # "free" | "reduced"
    energy_initial: float
    energy_final: float
    certificate: object | None
    q: float | None = None
    qdot: float | None = None
    extra: dict = field(default_factory=dict)
    # V2 sections 4.5 / 5.3 / 23: the per-frame diagnostics object, carried so
    # the trust-region history, the linear-solver history and the observability
    # report can be persisted. Without it they exist only inside the frame step
    # and section 23's artifacts would be empty.
    diagnostics: object | None = None


@dataclass
class PipelineResult:
    frames: list[FrameReport] = field(default_factory=list)
    reduction: object | None = None
    reduced_at_frame: int | None = None
    support: dict | None = None
    gate: dict | None = None
    state: GaussianMotionState | None = None
    joint_state: object | None = None

    def format(self) -> str:
        rows = [f"  {'frame':>5} {'mode':>8} {'dt':>7} {'E_init':>13} {'E_final':>13} "
                f"{'rho':>7} {'cert':>6} {'q':>10}"]
        rows.append("  " + "-" * 78)
        for f in self.frames:
            cert = f.certificate
            rho = cert.median_gain_ratio if cert else float("nan")
            ok = ("Y" if cert.certified else "n") if cert else "-"
            q = f"{f.q:.5f}" if f.q is not None else "-"
            rows.append(f"  {f.index:>5} {f.mode:>8} {f.dt:>7.4f} "
                        f"{f.energy_initial:>13.4f} {f.energy_final:>13.4f} "
                        f"{rho:>7.3f} {ok:>6} {q:>10}")
        if self.reduced_at_frame is not None:
            rows.append(f"  reduced to a joint model at frame {self.reduced_at_frame}")
        else:
            rows.append("  never reduced: the rigid-coherence test was not satisfied")
        return "\n".join(rows)


@torch.no_grad()
def _support_mask(state, cameras, rgbd_gt, partition, cfg):
    """V2 section 6.3 step 1 on the live frame: which supernodes move?

    Uses the exact analytic renderer sensitivity (section 6.1), never velocity
    magnitude, and returns a per-GAUSSIAN boolean mask so the joint fit sees
    only the moving part.
    """
    from joint_reduction.support import (
        achievable_motion_gain,
        initialize_support_from_evidence,
    )

    from .measurement import build_precision
    from .supernodes import GroupedObservation

    lin = build_linearization(state, cameras, eps_depth=cfg.eps_depth)
    go = GroupedObservation(linearization=lin, partition=partition)
    prec, res = [], []
    for view, (rgb_gt, d_gt, d_valid) in zip(lin.views, rgbd_gt):
        rr = rgb_gt - view.rgb
        pr = build_precision(rr, valid_mask=None, sigma_rgb=cfg.rgb_sigma,
                             robust=cfg.robust_measurement, nu=cfg.student_t_nu,
                             layout=("rgb", "rgb", "rgb"), channel_dim=0)
        rd = d_gt - view.depth
        pd = build_precision(
            rd.unsqueeze(0),
            valid_mask=(d_valid & (view.alpha > cfg.min_alpha_for_depth)).unsqueeze(0),
            sigma_rgb=(),
            depth_sigma_model=(cfg.depth_sigma_model
                               if ":" in str(cfg.depth_sigma_model)
                               else {"kind": cfg.depth_sigma_model,
                                     "a": cfg.depth_sigma_a, "b": cfg.depth_sigma_b}),
            depth_range=d_gt.unsqueeze(0), robust=cfg.robust_measurement,
            nu=cfg.student_t_nu, layout=("depth",), channel_dim=0).squeeze(0)
        prec.append((pr, pd))
        res.append((rr, rd))
    gain, ginfo = achievable_motion_gain(lin, go, prec, res)
    # Release the extra linearization and its per-camera precision/residual
    # images before returning. This runs while the frame step's own
    # linearization is still alive, and at full scene scale with several
    # 640x480 cameras holding both at once exhausts the device.
    gain = gain.detach().clone()
    del lin, go, prec, res
    torch.cuda.empty_cache()
    hard, info = initialize_support_from_evidence(
        gain, k_mad=cfg.support_k_mad,
        max_moving_fraction=cfg.support_max_moving_fraction)
    sid = partition.supernode_id.long().clamp_min(0)
    mask = hard.to(sid.device)[sid] > 0
    info.update({"num_supernodes": partition.num_supernodes,
                 "median_gain": ginfo["median_gain"],
                 "max_gain": ginfo["max_gain"],
                 "num_moving_gaussians": int(mask.sum()),
                 "supernode_gain": gain.detach().cpu().tolist(),
                 "support_hard": hard.detach().cpu().tolist()})
    return mask, info


@torch.no_grad()
def _aggregate_to_supernodes(state, partition, mask):
    """Supernode centroids and mean velocities for the moving supernodes.

    A supernode's members share one rigid motion, so their mean velocity IS the
    velocity at the centroid and the group contributes exactly one independent
    sample to the twist fit.
    """
    sid = partition.supernode_id.long().clamp_min(0)
    c = partition.num_supernodes
    dev, dt = state.mu.device, state.mu.dtype
    w = mask.to(dt)
    cnt = torch.zeros(c, device=dev, dtype=dt).index_add_(0, sid, w)
    mu = torch.zeros((c, 3), device=dev, dtype=dt).index_add_(
        0, sid, state.mu * w.unsqueeze(1))
    v = torch.zeros((c, 3), device=dev, dtype=dt).index_add_(
        0, sid, state.v * w.unsqueeze(1))
    keep = cnt > 0
    denom = cnt[keep].unsqueeze(1)
    return mu[keep] / denom, v[keep] / denom, int(keep.sum())


@torch.no_grad()
def _velocity_sigma(P, mask) -> float:
    """Mean per-axis velocity standard deviation the filter reports, over the
    moving set. This is section 18's 'predicted velocity uncertainty'."""
    vv = P[:, 6:9, 6:9]
    if mask is not None and bool(mask.any()):
        vv = vv[mask]
    tr = vv.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(0.0) / 3.0
    return float(tr.sqrt().mean())


def _gate_status(cert) -> str:
    """Map the per-frame certificate onto section 17's status vocabulary."""
    from estimator.certificate_v2 import (
        CERTIFIED_LOCAL_MAP,
        CONVERGED_BUT_UNOBSERVABLE,
        REJECTED_MODEL,
    )
    if cert is None:
        return REJECTED_MODEL
    return CERTIFIED_LOCAL_MAP if cert.certified else CONVERGED_BUT_UNOBSERVABLE


def run_sequence(
    state0: GaussianMotionState,
    P0: torch.Tensor,
    frames: Sequence,              # each: (rgbd_gt, cameras, dt)
    cfg: IEKFConfig | None = None,
    *,
    reduce_after: int = 2,
    reduce_only_after_convergence: bool = False,
    member: torch.Tensor | None = None,
    allow_reduction: bool = True,
    background: torch.Tensor | None = None,
    reduction_kwargs: dict | None = None,
    infer_support_before_reduction: bool = True,
    use_multi_criterion_gate: bool = True,
    gate_config=None,
    support_patience: int = 3,
    verbose: bool = True,
) -> PipelineResult:
    """Run the full ladder over a sequence of synchronized RGB-D frames."""
    from joint_reduction.discovery import decide_reduction
    from joint_reduction.joint_filter import JointState
    from joint_reduction.renderer_reduction import joint_update

    cfg = cfg or IEKFConfig()
    result = PipelineResult()
    state, P = state0, P0
    joint_model = None
    joint_state = None
    mu0 = R0 = None
    # V2 section 3.1: ONE canonical partition for the whole sequence. Rebuilding
    # it per frame would let the grouping drift with the estimate, and section
    # 3.1 requires it to be immutable.
    # V2 section 18: the reduction decision is a multi-criterion, windowed,
    # uncertainty-normalized test, not a single absolute residual threshold.
    gate = None
    if use_multi_criterion_gate:
        from joint_reduction.coherence_gate import CoherenceGate, GateConfig
        gate = CoherenceGate(gate_config or GateConfig())
    support_sticky = None
    support_streak = None
    partition = None
    if getattr(cfg, "use_supernodes", False):
        from .supernodes import build_supernodes
        partition = build_supernodes(
            state.mu, target_size=cfg.supernode_target_size,
            num_supernodes=cfg.supernode_count, knn=cfg.supernode_knn,
            seed=cfg.supernode_seed)


    for k, (rgbd_gt, cameras, dt) in enumerate(frames):
        if joint_model is None:
            # ---- FREE per-Gaussian stage (the bootstrap) -------------------
            state, P, diag = iekf_frame_step(state, P, rgbd_gt, cameras, dt=dt,
                                             cfg=cfg, background=background,
                                             partition=partition)
            cert = getattr(diag, "certificate", None)
            e0 = diag.rows[0].get("energy", float("nan"))
            eN = min((r.get("energy", float("inf")) for r in diag.rows
                      if "energy" in r), default=float("nan"))
            result.frames.append(FrameReport(
                index=k, dt=dt, mode="free", energy_initial=e0, energy_final=eN,
                certificate=cert, diagnostics=diag))
            if verbose:
                print(f"  frame {k}: free stage, energy {e0:.4f} -> {eN:.4f}"
                      + (f", rho {cert.median_gain_ratio:.3f}, "
                         f"certified={cert.certified}" if cert else ""))

            # ---- section 28: is the motion rigidly coherent yet? -----------
            # --iekf/--joint_reduce_after_convergence: only test coherence on
            # a frame whose own IEKF solve was certified. Velocities from an
            # uncertified frame are not trustworthy enough to commit a joint to.
            converged_ok = (cert.certified if (cert is not None) else False)
            if reduce_only_after_convergence and not converged_ok:
                if verbose:
                    print("  reduction skipped: this frame's IEKF solve was not "
                          "certified, and --joint_reduce_after_convergence is set")
            elif allow_reduction and k + 1 >= reduce_after:
                # V2 section 6: on a real scene only PART of it moves, so a
                # single rigid twist fitted over every Gaussian cannot explain
                # the field and reduction is declined forever. Infer which
                # canonical supernodes move first, then fit the joint to those.
                active = member
                if partition is not None and infer_support_before_reduction:
                    active, support_info = _support_mask(
                        state, cameras, rgbd_gt, partition, cfg)
                    # Hysteresis. The gain-based evidence RANKS supernodes, so
                    # which ones clear the threshold shifts between frames even
                    # when the physical moving part is unchanged -- measured
                    # Jaccard 0.34-0.46 frame to frame on an unchanged part,
                    # which made section 18's support-stability criterion fire
                    # on noise. A supernode joins the moving set when flagged
                    # and leaves only after `support_patience` consecutive
                    # frames without evidence, so the set tracks the part
                    # rather than the ranking.
                    flagged = torch.tensor(support_info["support_hard"],
                                           device=state.mu.device) > 0
                    if support_streak is None:
                        support_streak = torch.zeros(partition.num_supernodes,
                                                     device=state.mu.device)
                    support_streak = torch.where(
                        flagged, torch.zeros_like(support_streak),
                        support_streak + 1.0)
                    sticky = (support_sticky | flagged) if support_sticky is not None else flagged
                    sticky = sticky & (support_streak < support_patience)
                    support_sticky = sticky
                    sid_ = partition.supernode_id.long().clamp_min(0)
                    active = sticky[sid_]
                    support_info["sticky_num_moving"] = int(sticky.sum())
                    support_info["sticky_num_gaussians"] = int(active.sum())
                    support_info["support_patience"] = support_patience
                    result.support = support_info
                    if verbose:
                        print(f"  §6 support: {support_info['num_moving']} "
                              f"flagged this frame, {support_info['sticky_num_moving']}"
                              f" of {support_info['num_supernodes']} supernodes "
                              f"moving after hysteresis ({int(active.sum())} of "
                              f"{state.num_gaussians} Gaussians); gain median "
                              f"{support_info['median_gain']:.1f} max "
                              f"{support_info['max_gain']:.1f}")
                    if int(active.sum()) < 16:
                        if verbose:
                            print("  §6 support found too few moving Gaussians "
                                  "to fit a joint; keeping the free stage")
                        active = member
                # Under grouping the state's independent quantities are the
                # supernode twists, not the per-Gaussian velocities: every
                # member of a supernode shares one rigid motion by
                # construction. Fitting the coherence twist to thousands of
                # individually noisy per-Gaussian velocities therefore measures
                # noise the state does not even carry, and the relative
                # residual never falls below the gate. Aggregate to the
                # supernodes the state actually estimates.
                if partition is not None and active is not None:
                    fit_mu, fit_v, agg = _aggregate_to_supernodes(
                        state, partition, active)
                    if verbose:
                        print(f"  §3 coherence fitted on {agg} moving "
                              f"supernodes, not {int(active.sum())} "
                              f"individually-noisy Gaussian velocities")
                else:
                    sel = slice(None) if active is None else active
                    fit_mu, fit_v = state.mu[sel], state.v[sel]
                decision = decide_reduction(fit_mu, fit_v, member=None,
                                            **(reduction_kwargs or {}))
                if verbose:
                    print(decision.format())

                commit = decision.confident
                if gate is not None:
                    from joint_reduction.coherence_gate import WindowSample
                    coh = decision.coherence
                    # Section 18: judge the residual against the velocity
                    # uncertainty the filter itself reports, not against an
                    # absolute constant. An absolute threshold cannot know that
                    # a 0.5 residual is decisive when sigma_v is 0.01 and
                    # meaningless when sigma_v is 0.4.
                    sigma_v = _velocity_sigma(P, active)
                    gate.push(WindowSample(
                        frame=k, omega=coh.omega.detach().cpu().double(),
                        v=coh.v0.detach().cpu().double(),
                        residual=float(coh.residual),
                        residual_sigma=float(sigma_v),
                        twist_covariance_trace=float("nan"),
                        observable_rank=int(coh.rank),
                        support=(active.detach().cpu().long()
                                 if active is not None else None),
                        certificate_status=_gate_status(cert),
                        model_name=(decision.selected.kind
                                    if decision.selected else ""),
                        model_score=(decision.selected.score
                                     if decision.selected else float("nan"))))
                    verdict = gate.evaluate()
                    result.gate = gate.to_dict()
                    if verbose:
                        if verdict["reduce"]:
                            print("  §18 multi-criterion gate: ALL criteria "
                                  "satisfied over the window -> reduce")
                        else:
                            for r in verdict["reason"]:
                                print(f"  §18 gate declines: {r}")
                    # The gate is the authority on WHEN; decide_reduction stays
                    # the authority on WHICH model and keeps every veto that
                    # section 18 does NOT replace (identifiability, model
                    # quality). Section 18 replaces exactly one thing: the
                    # absolute coherence-residual threshold.
                    commit = bool(verdict["reduce"]) and (
                        decision.confident
                        or decision.declined_only_on_coherence_residual)
                    if commit and decision.model is None:
                        from joint_reduction.renderer_reduction import RenderJointModel
                        b = decision.selected
                        decision.model = RenderJointModel(
                            kind=b.kind, axis=b.axis, point=b.point,
                            member=active)
                        if verbose:
                            print(f"  §18 the windowed gate overrides the "
                                  f"absolute residual threshold "
                                  f"({coh.relative_residual:.3f}); every other "
                                  f"veto passed, so the {b.kind} model is built")
                if commit:
                    joint_model = decision.model
                    result.reduction = decision
                    result.reduced_at_frame = k
                    # The joint generates the poses from here on, so the
                    # reference geometry is frozen at the moment of reduction.
                    mu0, R0 = state.mu.clone(), state.R.clone()
                    # The reduced state starts at q = 0 relative to the frozen
                    # reference geometry, with a loose prior: section 29.3 says
                    # not to discard uncertainty during reduction.
                    joint_state = JointState(
                        q=torch.zeros(1, device=state.mu.device, dtype=state.mu.dtype),
                        qdot=torch.zeros(1, device=state.mu.device, dtype=state.mu.dtype),
                        cov=torch.eye(2, device=state.mu.device,
                                      dtype=state.mu.dtype))
                    if verbose:
                        print(f"  --> reduced to a {joint_model.kind} joint; free "
                              f"per-Gaussian updates stop here (section 30)")
        else:
            # ---- REDUCED joint stage ---------------------------------------
            # Section 30: the Gaussians are generated rigidly from q. They are
            # never deformed independently again.
            q = float(joint_state.q.reshape(-1)[0])
            qdot = float(joint_state.qdot.reshape(-1)[0])
            q_pred = q + qdot * dt                       # kinematic prediction
            e0 = eN = float("nan")
            for _ in range(max(cfg.max_iters, 1)):
                mu_q, R_q = joint_model.transport(mu0, R0, q_pred)
                cur = GaussianMotionState(
                    mu=mu_q, R=R_q, v=state.v, omega=state.omega,
                    colors=state.colors, opacity=state.opacity, scales=state.scales)
                lin = build_linearization(cur, cameras, background=background,
                                          eps_depth=cfg.eps_depth)
                res, prec = [], []
                energy = 0.0
                for view, (rgb_gt, depth_gt, depth_valid) in zip(lin.views, rgbd_gt):
                    r_rgb = rgb_gt - view.rgb
                    r_d = depth_gt - view.depth
                    p_rgb = torch.ones_like(r_rgb) if cfg.use_rgb else torch.zeros_like(r_rgb)
                    ok = depth_valid & (view.alpha > cfg.min_alpha_for_depth)
                    p_d = ok.to(r_d.dtype) if cfg.use_depth else torch.zeros_like(r_d)
                    res.append((r_rgb, r_d))
                    prec.append((p_rgb, p_d))
                    energy += 0.5 * float((p_rgb * r_rgb * r_rgb).sum()
                                          + (p_d * r_d * r_d).sum())
                if e0 != e0:
                    e0 = energy
                dq, info = joint_update(lin, joint_model, mu_q, res, prec)
                step = float(dq.reshape(-1)[0])
                if not (step == step):
                    break
                q_pred += step
                eN = energy
                if abs(step) < 1e-7:
                    break
            new_qdot = (q_pred - q) / max(dt, 1e-12)
            joint_state = JointState(
                q=torch.tensor([q_pred], device=state.mu.device, dtype=state.mu.dtype),
                qdot=torch.tensor([new_qdot], device=state.mu.device, dtype=state.mu.dtype),
                cov=joint_state.cov)
            mu_q, R_q = joint_model.transport(mu0, R0, q_pred)
            state = GaussianMotionState(
                mu=mu_q, R=R_q, v=state.v, omega=state.omega,
                colors=state.colors, opacity=state.opacity, scales=state.scales)
            result.frames.append(FrameReport(
                index=k, dt=dt, mode="reduced", energy_initial=e0, energy_final=eN,
                certificate=None, q=q_pred, qdot=new_qdot))
            if verbose:
                print(f"  frame {k}: reduced stage, q {q:.5f} -> {q_pred:.5f} "
                      f"(qdot {new_qdot:+.4f}), energy {e0:.4f} -> {eN:.4f}")

    result.state = state
    result.joint_state = joint_state
    return result
