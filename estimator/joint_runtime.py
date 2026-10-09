"""Sections 19, 28, 29: the production entry point for the Phase-2 screw IEKF.

CODEX_CONTINUE_PHASE2_RIGID_SCREW_IEKF_INTEGRATION.md.

    Phase 1 -> frozen support -> initialize joint state
            -> new rigid-screw Phase-2 IEKF -> posterior trajectory/geometry

Section 28 is explicit that this branch must NOT call the old free-SE(3)
Phase-2 estimator as a mandatory intermediate. It does not: the only Phase-1
products it consumes are the frozen moving support and a coarse initial twist,
both of which are initialization (section 19), and section 20's test proves the
filter corrects a deliberately wrong one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor

from .joint_iekf import JointIEKFConfig, run_window
from .joint_kinematics import state_to_gaussians
from .joint_state import JointState, initialise_from_twist
from .measurement import build_precision
from .renderer_bridge import build_linearization


def initial_twist_from_phase1(mu0: Tensor, members: Tensor, positions) -> Tensor:
    """Section 19: a coarse generator from the frozen Phase-1 support.

    A single weighted rigid-twist fit to the Phase-1 displacement of the
    moving support. This is INITIALIZATION, not classification: it is allowed
    to be wrong, and section 20 requires a test that proves the filter
    corrects it.
    """
    from joint_reduction.rigid_twist_from_tracks import fit_rigid_twist

    idx = members.to(mu0.device)
    x0 = mu0.double()[idx]
    last = torch.as_tensor(positions[-1]).double().to(mu0.device)[idx]
    v = last - x0
    fit = fit_rigid_twist(x0, v)
    xi = torch.cat([fit["v_c"] - torch.linalg.cross(fit["omega"],
                                                    fit["centre"]),
                    fit["omega"]]).cpu()
    if float(xi.norm()) <= 0.0:
        xi = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    return xi


def initial_q_from_phase1(part, mu0: Tensor, members: Tensor, positions) -> Tensor:
    """Section 29: initialize q(t) by projecting Phase-1 COM motion onto the screw.

    `q` used to start at zero on every frame. The process model then predicts
    `q_{t+1} = q_t + dt qdot_t` from `qdot = 0`, so each frame begins exactly
    where the last one ended and the filter has to discover the whole
    trajectory incrementally, one frame at a time, inside a per-frame trust
    region. Measured on an 8-frame storage slice at 25 iterations it reached
    q = 0.0094 -- about an eighth of the travel the data contains -- with the
    trust radius sitting on its floor (1e-08) for five of the eight frames.

    The displacement per unit q is taken from the transform itself rather than
    from the chart's algebra, so it stays correct for a screw with rotation:

        g = c(q=1) - c(0),      q_t = <Delta c_t, g> / |g|^2

    Frame 0 keeps q = 0 exactly -- it is the canonical gauge, not an estimate.
    This is INITIALIZATION; the filter still corrects it, and section 20's
    wrong-initialization test still has to pass.
    """
    from .joint_kinematics import part_transform

    idx = torch.as_tensor(members).long().reshape(-1)
    c0 = mu0.double().cpu()[idx].mean(0)
    R1, t1 = part_transform(1.0, part.zeta, part.length)
    g = (R1.double() @ c0 + t1.double()) - c0
    gn = float(g @ g)
    T = len(positions)
    q = torch.zeros(T, dtype=torch.float64)
    if gn <= 0.0:
        return q
    for t in range(1, T):
        dc = torch.as_tensor(positions[t]).double().cpu()[idx].mean(0) - c0
        q[t] = float(dc @ g) / gn
    return q


def run_phase2_rigid_screw(truth, cameras, frames, cfg, args, writer, *,
                           members: Tensor, positions, model_error=None,
                           verbose: bool = True) -> dict:
    """Section 28's production branch, with the section 29 artifacts."""
    mu0 = truth.mu.double()
    T = len(frames)
    members = torch.as_tensor(members).long().reshape(-1).cpu()
    if verbose:
        print(f"\n  === Phase 2: RIGID-SCREW IEKF "
              f"({int(members.numel())} frozen moving Gaussians, "
              f"{T} frames, K=1) ===")

    xi0 = initial_twist_from_phase1(mu0, members, positions)
    part = initialise_from_twist(members, mu0, xi0,
                                 torch.zeros(T, dtype=torch.float64), None)
    q_init = initial_q_from_phase1(part, mu0, members, positions)
    part.q = q_init.clone()
    dt0 = 1.0
    part.qdot = torch.cat([(q_init[1:] - q_init[:-1]) / dt0,
                           torch.zeros(1, dtype=torch.float64)])
    state = JointState([part], torch.arange(T, dtype=torch.float64))
    init_record = {"initial_twist": xi0.tolist(),
                   "zeta": part.zeta.tolist(), "length_scale": part.length,
                   "centroid": part.centroid.tolist(),
                   "q0": 0.0, "qdot0": float(part.qdot[0]),
                   "q_init": q_init.tolist(),
                   "q_init_source": "phase1_com_projection",
                   "q_init_max": float(q_init.abs().max()),
                   "note": ("section 29: q(t) initialized by projecting the "
                            "Phase-1 COM motion onto the screw; q(0) = 0 "
                            "remains the canonical gauge. The geometry is a "
                            "coarse Phase-1 twist and the filter is still "
                            "required to correct both (section 20)")}

    counter = {"frame": 0}

    def linearize(st, frame):
        counter["frame"] = frame
        mu, _Rg = state_to_gaussians(st, frame, mu0, None)
        import dataclasses
        s = dataclasses.replace(truth, mu=mu.to(truth.mu.dtype))
        return build_linearization(s, cameras, eps_depth=cfg.eps_depth)

    def precision_fn(lin):
        obs = frames[counter["frame"]][0]
        out = []
        for v, ob in zip(lin.views, obs):
            if ob is None:
                out.append((torch.zeros_like(v.rgb), torch.zeros_like(v.depth)))
                continue
            gt_rgb, gt_d = ob[0].to(v.rgb.device), ob[1].to(v.depth.device)
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
                # Section 13 of the Phase-1 spec: R_eff = R_sensor + R_model.
                # Phase 1 applies this and Phase 2 must use the SAME
                # precision (section 2). Without it the filter treats a
                # residual dominated by canonical/appearance mismatch as if it
                # were sensor noise: on Storage the image energy is 1.2e+06,
                # the posterior information is inflated by orders of
                # magnitude, and the section 22 test then rejects omega = 0
                # at chi2 = 1e+07 on a joint that is physically a drawer.
                ci = len(out)
                w = pr * (cfg.rgb_sigma ** 2)
                pr = model_error.effective_rgb_precision(
                    ci, cfg.rgb_sigma, robust_weight=w)
                sv = torch.ones_like(pd) * (cfg.depth_sigma_a ** 2)
                wd = pd * sv
                pd = model_error.effective_depth_precision(
                    ci, sv, robust_weight=wd)
                pd = pd * m.to(pd.dtype)
            out.append((pr, pd))
        return out

    obs = []
    for (ob, _fid) in frames:
        obs.append([(o[0], o[1]) if o is not None else None for o in ob])

    jcfg = JointIEKFConfig(
        max_iters=int(getattr(args, "phase2_screw_iters", 8)),
        max_cg_iters=int(getattr(args, "phase2_screw_cg_iters", 40)),
        verbose=verbose)
    out = run_window(state, mu0, None, obs, linearize, precision_fn,
                     cfg=jcfg, label="screw")

    # ---- section 29 artifacts --------------------------------------------
    recs = out["records"]
    writer.json("phase2_rigid_screw/config", {
        "K": 1, "frames": T, "cameras": len(cameras),
        "max_iters": jcfg.max_iters, "max_cg_iters": jcfg.max_cg_iters,
        "sigma_qddot": jcfg.sigma_qddot,
        "branch": "phase2_rigid_screw_iekf",
        "old_free_se3_phase2_called": False,
        "identity": ("Phase 2 = Phase 1 IEKF + rigid-screw state-to-Gaussian "
                     "map; delta y = H_G J_{G,J} delta x_J")})
    writer.json("phase2_rigid_screw/frozen_support", {
        "num_members": int(members.numel()),
        "members": members.tolist()[:4096],
        "truncated": bool(members.numel() > 4096),
        "frozen_before_phase2": True})
    writer.json("phase2_rigid_screw/initialization", init_record)
    writer.csv("phase2_rigid_screw/joint_trace",
               ["frame", "q", "qdot", "energy_before", "energy_after",
                "iterations", "accepted", "radius", "rho_last",
                "sign_flipped"],
               [[r.frame, r.q, r.qdot, r.energy_before, r.energy_after,
                 r.iterations, r.accepted, r.radius, r.rho_last,
                 int(r.sign_flipped)] for r in recs])
    writer.csv("phase2_rigid_screw/information_trace",
               ["frame"] + [f"post_eig_{i}" for i in range(6)]
               + [f"meas_eig_{i}" for i in range(6)]
               + [f"geom_eig_{i}" for i in range(4)],
               [[r.frame]
                + list(r.observability["posterior_eigenvalues"])
                + list(r.observability["measurement_information_eigenvalues"])
                + list(r.observability["geometry_information_eigenvalues"])
                for r in recs])
    writer.json("phase2_rigid_screw/convergence", {
        "per_frame": [{"frame": r.frame, "iterations": r.iterations,
                       "accepted": r.accepted, "radius": r.radius,
                       "energy_before": r.energy_before,
                       "energy_after": r.energy_after} for r in recs],
        "total_energy_reduction": (recs[0].energy_before - recs[-1].energy_after
                                   if recs else None)})
    geom = out["geometry"]
    writer.json("phase2_rigid_screw/summary", {
        "q": out["q"], "qdot": out["qdot"], "geometry": geom,
        "final_covariance": out["final_covariance"].tolist(),
        "physical_prismatic_travel":
            state.parts[0].physical_coordinate("prismatic").tolist(),
        "physical_revolute_angle":
            state.parts[0].physical_coordinate("revolute").tolist()})
    # ------------------------------------------------------------------
    # Sections 3, 16, 26: the production label.
    #
    # The unrestricted screw above recovers the geometry and trajectory. It
    # does NOT decide the joint type: section 2 requires the null to be tested
    # at the best P-CONSTRAINED solution, so a second pass of the same IEKF
    # runs with omega pinned to zero and the efficient score is evaluated
    # there. The old posterior chi-square is kept as `legacy_angular_chi2`
    # and no longer sets the label (section 26).
    # ------------------------------------------------------------------
    legacy = posterior_prismatic_test(out["state"], recs[-1].information) \
        if recs else {"status": "UNRESOLVED_GEOMETRY",
                      "reason": "no frames were updated"}

    from .curvature_score import (PRISMATIC_AT_RESOLUTION, REVOLUTE_CANDIDATE,
                                  run_efficient_p_test)
    from .prismatic_null import run_prismatic_null_sequence

    null_state = JointState(
        [initialise_from_twist(members, mu0, xi0,
                               torch.zeros(T, dtype=torch.float64), None)],
        torch.arange(T, dtype=torch.float64))
    if verbose:
        print("   fitting the P-constrained null (section 6)")
    nullfit = run_prismatic_null_sequence(
        null_state, mu0, None, obs, linearize, precision_fn, cfg=jcfg,
        max_passes=int(getattr(args, "phase2_null_passes", 2)))
    test = run_efficient_p_test(
        null_state, mu0, None, obs, linearize, precision_fn,
        # Section 13: one globally fixed window length. The default is the
        # FINEST partition -- one block per camera per frame -- because the
        # cluster covariance needs degrees of freedom: with the unrelated
        # `--joint-block-frames 10` default this collapsed to 2 blocks on an
        # 8-frame window, giving an F(1,1) cut of 161 and a test with no
        # power at all.
        block_frames=int(getattr(args, "phase2_block_frames", 1) or 1),
        confidence=float(getattr(args, "joint_confidence", 0.95)),
        # Section 19: the null fit watches its own P direction stop moving
        # across passes, which is the convergence the section describes.
        null_converged=(len(nullfit["null_direction_change"]) > 1
                        and nullfit["null_direction_change"][-1] < 0.05))
    t = test["test"]

    status = t["status"]
    if status == REVOLUTE_CANDIDATE:
        # Section 16: the axis comes from the ALREADY FITTED unrestricted
        # runtime; no new R optimizer is introduced.
        ax = geom.get("revolute")
        status = "REVOLUTE" if (ax and ax.get("identifiable")) else \
            REVOLUTE_CANDIDATE
    # Section 22: persist every diagnostic, for every scene, without
    # exception -- and without tuning anything after seeing them.
    sc, sh, bl, rb = test["scores"], test["schur"], test["blocks"], test["robust"]
    writer.json("phase2_rigid_screw/curvature_diagnostics", {
        "F_alpha_alpha": sc["F_aa"].tolist(),
        "F_beta_alpha": sc["F_ab"].T.tolist(),
        "F_beta_beta": sc["F_bb"].tolist(),
        "g_alpha": sc["g_alpha"].tolist(),
        "g_beta": sc["g_beta"].tolist(),
        "F_beta_given_alpha": sh["F_beta_given_alpha"].tolist(),
        "g_beta_given_alpha": sh["g_beta_given_alpha"].tolist(),
        "A_beta_alpha": sh["A_beta_alpha"].tolist(),
        "u_blocks": bl["u_blocks"].tolist(),
        "block_index": [{"camera": b["camera"], "window": b["window"],
                         "frames": b["frames"]} for b in sc["blocks"]],
        "B_U": rb["B_U"].tolist(),
        "singular_values_B": rb.get("singular_values_B", []),
        "rank_B": rb.get("rank_B"),
        "S_curv": t["S_curv"], "F_statistic": t.get("F_statistic"),
        "p_value": t["p_value"], "rank_curv": t["rank_curv"],
        "beta_one_step": t["beta_one_step"],
        "Sigma_beta": t["Sigma_beta"],
        "alpha_layout": sc["layout"],
        "P_direction": out["state"].parts[0].zeta[:3].tolist(),
        "P_null_direction": null_state.parts[0].zeta[:3].tolist(),
        "P_null_q": null_state.parts[0].q.tolist(),
        "P_null_objective_per_pass": nullfit["null_objective_per_pass"],
        "legacy_angular_chi2": legacy.get("chi2_omega_zero"),
        "note": "section 22: persisted for every scene; nothing tuned after"})
    writer.json("phase2_rigid_screw/classification", {
        "status": status, "efficient_score_test": t,
        "null_fit": {k: v for k, v in nullfit.items()
                     if k not in ("state", "result")},
        "legacy_angular_chi2": legacy,
        "unrestricted_geometry": geom,
        "decision_rule": (
            "sections 2, 16: H_P: beta = 0 tested at the best P-constrained "
            "solution by a cluster-robust efficient score; the unrestricted "
            "screw supplies the axis only after P is rejected. The legacy "
            "posterior chi-square is recorded but does not set this label "
            "(section 26)")})
    out["classification"] = {"status": status, "test": t, "legacy": legacy}
    if verbose:
        print(f"   efficient P-null test: {status}")
        print(f"     S_curv {t['S_curv']:.4e}  F {t.get('F_statistic', 0):.4f}"
              f"  cut {t['cut']:.3f}  rank {t['rank_curv']}"
              f"  blocks {t.get('num_blocks_used', 0)}")
        print(f"     legacy angular chi2 "
              f"{legacy.get('chi2_omega_zero', float('nan')):.3e} "
              f"(diagnostic only)")

    try:
        import numpy as np
        p = Path(writer.dir) / "phase2_rigid_screw"
        p.mkdir(parents=True, exist_ok=True)
        np.savez(p / "posterior_covariance.npz",
                 covariance=torch.stack([r.posterior for r in recs]).numpy(),
                 information=torch.stack([r.information for r in recs]).numpy())
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# section 22: the minimal posterior P test
# ---------------------------------------------------------------------------
def posterior_prismatic_test(state, information: Tensor, part: int = 0,
                             confidence: float = 0.95) -> dict:
    """Section 22: is the departure from P supported by its own information?

    The two angular curvature directions are the geometry tangent coordinates
    that change `omega`. Projecting the posterior information onto them,

        F_beta = B_beta^T F_theta B_beta,

    gives what the data actually knows about curvature after `q`, `qdot` and
    the in-plane direction have absorbed what they can.

    Section 22 forbids an arbitrary norm threshold, so the decision is a
    chi-square on `omega = 0` against its own null distribution, and an
    unobserved pair of directions returns UNRESOLVED_GEOMETRY rather than a
    verdict. This runs on the SAME physical state and the SAME RGB-D
    measurement model as the filter -- it is a read of the posterior, not a
    second estimator.
    """
    from joint_reduction.common_screw import chi2_quantile
    from .joint_manifold import tangent_basis

    p = state.parts[part]
    i0 = part * 6 + 2
    F_theta = information[i0:i0 + 4, i0:i0 + 4].double()
    B = tangent_basis(p.zeta)                       # [6,4]
    # the curvature directions are those whose ambient image moves omega
    ang = B[3:, :]                                  # [3,4]
    u, sv, _vh = torch.linalg.svd(ang, full_matrices=False)
    keep = int((sv > sv.max().clamp_min(1e-300) * 1e-8).sum())
    if keep == 0:
        return {"status": "UNRESOLVED_GEOMETRY",
                "reason": "no geometry direction changes omega at this state"}
    # Every tangent direction that changes omega is a curvature direction.
    # Section 22 says "two", and that is right AT A PRISMATIC STATE, where
    # `ang` has rank 2 (section 12: two angular directions orthogonal to the
    # translation). Away from P it has rank 3, and hard-coding two discards
    # the dominant one: on a state with |omega| = 0.993 the component thrown
    # away carried 8.29 of the 8.41 total, so the test reported chi2 = 2e-26
    # on an unmistakably revolute geometry. The rank is used instead.
    _u2, _s2, vh2 = torch.linalg.svd(ang, full_matrices=True)
    B_beta = vh2[:keep].T                           # [4, rank(ang)]
    F_beta = B_beta.T @ F_theta @ B_beta
    ev = torch.linalg.eigvalsh(F_beta)
    omega = p.zeta[3:]
    # The departure from P in TANGENT coordinates. `[0; omega]` is not a
    # tangent vector -- it is part of zeta itself -- so projecting it with
    # `B^T` measures nothing and returned chi2 = 2e-29 on a state with
    # |omega| = 0.98. What is wanted is the least-norm tangent step that
    # PRODUCES this omega, which is the pseudo-inverse of the map
    # `delta_theta -> delta_omega`.
    beta = B_beta.T @ (torch.linalg.pinv(ang) @ omega)
    chi2 = float(beta @ (F_beta @ beta))
    dof = int(B_beta.shape[1])
    cut = chi2_quantile(dof, confidence)
    # Power, not magnitude. The smallest departure this posterior could
    # distinguish from zero is `sqrt(cut / lambda_min)`; because zeta is
    # normalised, the whole available range of beta is at most 1, so a
    # detection floor above that means the test could not have rejected ANY
    # curvature and the honest answer is that nothing was resolved.
    lam = float(ev.min())
    beta_min = (math.sqrt(cut / lam) if lam > 0 else float("inf"))
    observed = beta_min <= 1.0
    if not observed:
        status = "UNRESOLVED_GEOMETRY"
    elif chi2 > cut:
        status = "REVOLUTE"
    else:
        status = "PRISMATIC_AT_RESOLUTION"
    return {"status": status, "chi2_omega_zero": chi2, "dof": dof, "cut": cut,
            "smallest_detectable_beta": beta_min,
            "F_beta_eigenvalues": ev.tolist(),
            "omega_norm": float(omega.norm()),
            "curvature_directions": int(B_beta.shape[1]),
            "confidence": confidence,
            "note": ("section 22: a chi-square on omega = 0 against the "
                     "posterior information in the curvature directions; no "
                     "norm threshold, and an unobserved pair returns "
                     "UNRESOLVED_GEOMETRY")}


# ===========================================================================
# Stage B (sections 30, 31, 41): run the generic screw, M_P and M_R on the
# same frozen support and report the evidence. This function CLASSIFIES
# NOTHING (section 33).
# ===========================================================================

def run_phase2_parallel_pr(truth, cameras, frames, cfg, args, writer, *,
                           members: Tensor, positions, model_error=None,
                           generic=None, verbose: bool = True) -> dict:
    """Sections 30/31: E_P and E_R on train and held-out cameras.

    `generic` is an optional `run_phase2_rigid_screw` result, used only to
    initialize M_R's axis (section 29) and to report E_G alongside.
    """
    import dataclasses

    from .phase2_model_comparison import (
        fit_model, heldout_delta, init_prismatic, init_revolute,
        prismatic_state, revolute_state,
    )

    mu0 = truth.mu.double()
    T = len(frames)
    members = torch.as_tensor(members).long().reshape(-1).cpu()
    n_cam = len(cameras)
    if verbose:
        print(f"\n  === Phase 2: PARALLEL P/R MODELS "
              f"({int(members.numel())} frozen moving Gaussians, "
              f"{T} frames, {n_cam} cameras) ===")

    # ---- section 31: a fixed camera-fold protocol shared by all scenes ----
    n_folds = max(1, min(int(getattr(args, "phase2_pr_folds", 3)), n_cam))
    folds = [[c for c in range(n_cam) if c % n_folds != f] for f in range(n_folds)]
    heldout = [[c for c in range(n_cam) if c % n_folds == f] for f in range(n_folds)]

    def _bind(state, train_cams):
        """The SAME callbacks the generic branch uses, restricted to a fold."""
        counter = {"frame": 0}
        sub = [cameras[c] for c in train_cams]

        def linearize(st, frame):
            counter["frame"] = frame
            mu, _ = st.gaussians(frame, mu0, None)
            s = dataclasses.replace(truth, mu=mu.to(truth.mu.dtype))
            return build_linearization(s, sub, eps_depth=cfg.eps_depth)

        def precision_fn(lin):
            return _measurement_precision(lin, frames[counter["frame"]][0],
                                          train_cams, cfg, model_error)

        obs = []
        for (ob, _fid) in frames:
            obs.append([(ob[c][0], ob[c][1]) if ob[c] is not None else None
                        for c in train_cams])
        return linearize, precision_fn, obs

    def _energy(state, cam_ids):
        """E_M on a camera subset, split by RGB / depth / frame / camera."""
        rows, tot = [], {"rgb": 0.0, "depth": 0.0}
        sub = [cameras[c] for c in cam_ids]
        for t in range(T):
            mu, _ = state.gaussians(t, mu0, None)
            s = dataclasses.replace(truth, mu=mu.to(truth.mu.dtype))
            lin = build_linearization(s, sub, eps_depth=cfg.eps_depth)
            prec = _measurement_precision(lin, frames[t][0], cam_ids, cfg,
                                          model_error)
            for j, c in enumerate(cam_ids):
                ob = frames[t][0][c]
                if ob is None:
                    rows.append([t, c, 0.0, 0.0])
                    continue
                v = lin.views[j]
                pc, pd = prec[j]
                rr = (ob[0].to(v.rgb.device) - v.rgb).double()
                dr = (ob[1].to(v.depth.device) - v.depth).double()
                er = 0.5 * float((pc.double() * rr * rr).sum())
                ed = 0.5 * float((pd.double() * dr * dr).sum())
                rows.append([t, c, er, ed])
                tot["rgb"] += er
                tot["depth"] += ed
        tot["total"] = tot["rgb"] + tot["depth"]
        return tot, rows

    jcfg = JointIEKFConfig(
        max_iters=int(getattr(args, "phase2_screw_iters", 8)),
        max_cg_iters=int(getattr(args, "phase2_screw_cg_iters", 40)),
        verbose=False)

    gen_axis = gen_angles = gen_point = None
    if generic is not None:
        g = generic.get("geometry") or {}
        gen_axis = g.get("revolute_axis") or g.get("omega")
        gen_point = g.get("axis_point")
        ang = generic.get("physical_revolute_angle")
        if ang is None:
            ang = (generic.get("summary") or {}).get("physical_revolute_angle")
        gen_angles = ang

    results, per_frame = {}, []
    for fi, (train, test) in enumerate(zip(folds, heldout)):
        if not test or not train:
            continue
        pos = [torch.as_tensor(p).double().cpu() for p in positions]
        built = {
            "prismatic": lambda: prismatic_state(
                init_prismatic(mu0.cpu(), members, pos)),
            "revolute": lambda: revolute_state(
                init_revolute(mu0.cpu(), members, pos, axis=gen_axis,
                              angles=gen_angles, axis_point=gen_point)),
        }
        for name, make in built.items():
            st = make()
            before = st.parts[0].geometry()
            lin, prec, obs = _bind(st, train)
            fit = fit_model(st, mu0, obs, lin, prec, jcfg,
                            label=f"fold {fi} {name}")
            tr, rows_tr = _energy(st, train)
            ho, rows_ho = _energy(st, test)
            results.setdefault(name, []).append(
                {"fold": fi, "train_cameras": train, "heldout_cameras": test,
                 "train": tr, "heldout": ho,
                 "geometry": st.parts[0].geometry(),
                 "observability": _model_observability(name, before, st, fit)})
            for split, rows in (("train", rows_tr), ("heldout", rows_ho)):
                for (t, c, er, ed) in rows:
                    per_frame.append([name, fi, split, t, c, er, ed])
            if verbose:
                print(f"    fold {fi} {name:>9}: train {tr['total']:.4e}  "
                      f"heldout {ho['total']:.4e}")

    degenerate = [r["fold"] for r in results.get("revolute", [])
                  if not r["observability"].get("fair_comparison", True)]
    comparison = {
        "revolute_degenerate_folds": degenerate,
        "comparison_is_interpretable": not degenerate,
        "note": ("sections 30-33: evidence only. No label is produced here, "
                 "no loss-gap threshold is applied, and R is NOT declared "
                 "the winner by a margin rule."),
        "folds": [{"fold": i, "train_cameras": t, "heldout_cameras": h}
                  for i, (t, h) in enumerate(zip(folds, heldout))],
        "per_fold": results,
        "delta_heldout_per_fold": [
            heldout_delta(p["heldout"]["total"], r["heldout"]["total"])
            for p, r in zip(results.get("prismatic", []),
                            results.get("revolute", []))],
    }
    if generic is not None:
        comparison["generic_reference"] = {
            "geometry": generic.get("geometry"),
            "note": "E_G is the generic screw branch's own training energy"}

    writer.json("phase2_model_comparison/comparison", comparison)
    writer.csv("phase2_model_comparison/per_frame_camera",
               ["model", "fold", "split", "frame", "camera",
                "rgb_energy", "depth_energy"], per_frame)
    for name in ("prismatic", "revolute"):
        if name in results:
            writer.json(f"phase2_model_comparison/{name}/folds", results[name])
    _write_comparison_html(writer, comparison)
    writer.json("phase2_model_comparison/heldout/summary", {
        "delta_heldout_per_fold": comparison["delta_heldout_per_fold"],
        "definition": "delta = E_P,heldout - E_P,heldout(R); positive favours R",
        "not_a_classifier": True})
    return comparison


def _measurement_precision(lin, obs_all, cam_ids, cfg, model_error):
    """Section 4: delegate to the ONE shared R_eff path.

    Kept as a name so existing call sites do not change, but it owns no
    formula: duplicating `R_sensor + R_model` is exactly what section 4
    forbids, because the duplicate drifts.
    """
    from .phase2_shared import effective_precision
    return effective_precision(lin, obs_all, cam_ids, cfg, model_error)


def _write_comparison_html(writer, comparison: dict) -> None:
    """Section 41's `joint_debug.html`: an index over the comparison numbers.

    It reports E_P and E_R per fold split by RGB and depth, and -- in the same
    table, not in a footnote -- whether each fold's M_R was actually fitted.
    A fold whose revolute geometry was degenerate is marked in place, because a
    reader scanning for "which model won" would otherwise read a frozen
    hypothesis as a fitted one.
    """
    import html as _html
    import json as _json

    rows = []
    for m in ("prismatic", "revolute"):
        for r in comparison.get("per_fold", {}).get(m, []):
            ob = r.get("observability", {})
            fair = ob.get("fair_comparison", True)
            warn = "" if fair else ' class="bad"'
            note = "" if fair else " &#9888; NOT FITTED"
            rows.append(
                f"<tr{warn}><td>{m}</td><td>{r['fold']}</td>"
                f"<td>{r['train']['rgb']:.4e}</td>"
                f"<td>{r['train']['depth']:.4e}</td>"
                f"<td>{r['heldout']['rgb']:.4e}</td>"
                f"<td>{r['heldout']['depth']:.4e}</td>"
                f"<td>{r['heldout']['total']:.4e}{note}</td></tr>")
    interp = comparison.get("comparison_is_interpretable", True)
    banner = ("" if interp else
              '<p class="bad">At least one fold\'s revolute model was '
              'degenerate (theta ~ 0, so its geometry Jacobian vanishes). '
              'The held-out difference on this scene is NOT P-versus-R '
              'evidence.</p>')
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Phase 2 model comparison</title><style>
 body{{font:13px/1.6 system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
 table{{border-collapse:collapse;margin:12px 0}} td,th{{border:1px solid #333;
 padding:5px 9px;text-align:right}} td:first-child,th:first-child{{text-align:left}}
 th{{background:#1b1b1b}} tr.bad td{{color:#f99}} p.bad{{color:#f99;
 border-left:3px solid #f55;padding-left:10px}}
 pre{{background:#1b1b1b;padding:12px;overflow:auto;border:1px solid #333}}
</style></head><body>
<h1>Phase 2 &mdash; parallel P/R comparison</h1>
<p>Sections 30&ndash;33. Evidence only: no label is produced here and no
loss-gap threshold is applied.</p>
{banner}
<table><tr><th>model</th><th>fold</th><th>train RGB</th><th>train depth</th>
<th>heldout RGB</th><th>heldout depth</th><th>heldout total</th></tr>
{''.join(rows)}</table>
<h2>&Delta;E<sub>heldout</sub> = E<sub>P</sub> &minus; E<sub>R</sub> per fold</h2>
<pre>{_html.escape(_json.dumps(comparison.get("delta_heldout_per_fold"), indent=1))}</pre>
<h2>comparison.json</h2>
<pre>{_html.escape(_json.dumps(comparison, indent=1, default=str))}</pre>
</body></html>"""
    path = writer.dir / "phase2_model_comparison" / "joint_debug.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    writer.written.append("phase2_model_comparison/joint_debug.html")
