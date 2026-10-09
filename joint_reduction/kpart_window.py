"""Windowed multiview K-part joint discovery.

CODEX_REQUIRED_MODIFICATIONS_MULTIVIEW_MULTIFRAME_KPART_IEKF.md
sections 5, 11, 14, 20, 24.

This is the production discovery stage. It replaces
"framewise motion -> binary moving mask -> one twist -> joint label" with

    multiview RGB-D sequence + analytic alpha-aware renderer + known K
        -> joint support + joint geometry + joint trajectories      (section 32)

estimated TOGETHER over a temporal window.

The objective is section 24's

    E = E_RGBD + lam_sup E_support + lam_temp E_temporal + lam_model E_model

with

    E_RGBD = sum_{t,c,p} rho( r^T R_eff^-1 r ),
    r_{tcp} = y_GT - h_cp( T(z, eta, q_t) G_0 )                      (section 5)

evaluated on the ACTUAL multiview frames, from IMMUTABLE canonical geometry
G_0 -- never from bootstrap-deformed geometry.

Section 11's alternation:
    A. (eta, q) <- argmin E(z, .)      joint geometry and trajectories
    B. z        <- argmin E(., eta, q) multi-label support
    C. rerender the true nonlinear model; accept only if E_true decreases
run to convergence on all four criteria, not for one pass.

Section 20: for small K every assignment (M_1..M_K) in {P,R}^K is enumerated and
scored on the SAME objective; larger K falls back to per-part local selection.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import torch

from .kpart_models import (PRISMATIC, REVOLUTE, UNRESOLVED, KPartObservation,
                           PartModel, SceneModel)
from .identifiability import (CONVERGED, UNRESOLVED_MODEL,
                              UNRESOLVED_OPTIMIZATION)
from .kpart_support import infer_kpart_support, part_achievable_gain
from .line_geometry import PluckerLine, fit_plucker_from_motion

EPS = 1e-12


# ---------------------------------------------------------------------------
# section 24: the windowed objective
# ---------------------------------------------------------------------------
@dataclass
class ObjectiveWeights:
    rgbd: float = 1.0
    support: float = 1.0
    temporal: float = 1.0
    model: float = 1.0          # reserved: model error enters through R_eff


@dataclass
class WindowEnergy:
    rgbd: float = 0.0
    rgb: float = 0.0
    depth: float = 0.0
    temporal: float = 0.0
    support: float = 0.0
    total: float = 0.0
    per_frame: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"total": self.total, "rgbd": self.rgbd, "rgb": self.rgb,
                "depth": self.depth, "temporal": self.temporal,
                "support": self.support, "per_frame": self.per_frame}


def temporal_precision(sigma_a: float = 1.0, dt: float = 1.0) -> float:
    """Section 12's process precision W_a = 1 / (sigma_a^2 dt^4).

    This is the acceleration process noise of a constant-velocity model, not a
    tuned regulariser: sigma_a is an acceleration in joint-coordinate units per
    unit time squared, so the temporal term lands on the same probabilistic
    scale as the data term (section 21) and the two are directly comparable.
    """
    return 1.0 / max(float(sigma_a) ** 2 * float(dt) ** 4, EPS)


def second_difference(q: list) -> list:
    """D_2 q = q_{t+1} - 2 q_t + q_{t-1} (section 12)."""
    return [q[i + 1] - 2 * q[i] + q[i - 1] for i in range(1, len(q) - 1)]


def acceleration_operator(T: int, times=None, dt: float = 1.0, *,
                          device=None, dtype=torch.float64) -> torch.Tensor:
    """The [T-2, T] operator A with (A q)_i = the acceleration at frame i.

    Uniformly sampled, this is section 12's D_2 divided by dt^2:

        (A q)_i = (q_{i+1} - 2 q_i + q_{i-1}) / dt^2,

    so (1/sigma_a^2) A^T A is exactly section 12's D_2^T W_a D_2 with
    W_a = 1/(sigma_a^2 dt^4). Section 12 also permits "the exact equivalent
    process covariance", which is what the non-uniform branch builds: with
    frame times t_i and steps h_i = t_{i+1} - t_i,

        (A q)_i = 2/(h_{i-1}+h_i) [ (q_{i+1}-q_i)/h_i - (q_i-q_{i-1})/h_{i-1} ],

    the standard second divided difference. This is what section 30's
    "irregular dt" case requires; it reduces to the uniform form when all
    h_i are equal, which `tests/test_strict_trajectory.py` checks numerically
    rather than assuming.
    """
    A = torch.zeros((max(T - 2, 0), T), device=device, dtype=dtype)
    if T < 3:
        return A
    if times is None:
        h_prev = torch.full((T - 2,), float(dt), device=device, dtype=dtype)
        h_next = h_prev
    else:
        tt = torch.as_tensor(times, device=device, dtype=dtype).reshape(-1)
        if int(tt.numel()) != T:
            raise ValueError(f"times has {int(tt.numel())} entries, expected {T}")
        h = tt[1:] - tt[:-1]
        if bool((h <= 0).any()):
            raise ValueError("frame times must be strictly increasing")
        h_prev, h_next = h[:-1], h[1:]
    rows = torch.arange(T - 2, device=device)
    scale = 2.0 / (h_prev + h_next)
    A[rows, rows] = scale / h_prev
    A[rows, rows + 1] = -scale * (1.0 / h_prev + 1.0 / h_next)
    A[rows, rows + 2] = scale / h_next
    return A


def temporal_energy(q: list, sigma_q: float = 1.0, *,
                    w_a: float | None = None, sigma_a: float | None = None,
                    times=None, dt: float = 1.0) -> float:
    """Section 12's E_temporal = 1/2 || D_2 q ||^2_{W_a}.

    A second-difference penalty: it leaves any constant-rate trajectory free and
    charges only curvature, so it regularises noise without forbidding the
    acceleration the trajectories deliberately contain.

    The 1/2 is not cosmetic -- it is what makes section 12's H_temp =
    D_2^T W_a D_2 the exact Hessian of this energy, so the banded Gauss-Newton
    solve minimises precisely the objective that `evaluate` reports.
    """
    if len(q) < 3:
        return 0.0
    if sigma_a is not None or times is not None:
        A = acceleration_operator(len(q), times=times, dt=dt)
        v = torch.as_tensor([float(x) for x in q], dtype=torch.float64)
        s_a = 1.0 if sigma_a is None else float(sigma_a)
        return float(0.5 * ((A @ v) ** 2).sum() / max(s_a * s_a, EPS))
    w = float(w_a) if w_a is not None else 1.0 / max(sigma_q * sigma_q, EPS)
    return float(0.5 * w * sum(d * d for d in second_difference(q)))


def temporal_hessian(T: int, w_a: float | None = None, *, device=None,
                     dtype=torch.float64, sigma_a: float | None = None,
                     times=None, dt: float = 1.0) -> torch.Tensor:
    """Section 12's H_temp = D_2^T W_a D_2 for ONE part's trajectory.

    [T,T], symmetric positive semi-definite, banded with half-bandwidth 2. Its
    null space is exactly the constant-velocity trajectories, which is the
    point: the prior charges acceleration and nothing else.
    """
    if T < 3:
        return torch.zeros((T, T), device=device, dtype=dtype)
    if sigma_a is not None or times is not None:
        A = acceleration_operator(T, times=times, dt=dt, device=device,
                                  dtype=dtype)
        s_a = 1.0 if sigma_a is None else float(sigma_a)
        return (A.T @ A) / max(s_a * s_a, EPS)
    D2 = torch.zeros((T - 2, T), device=device, dtype=dtype)
    rows = torch.arange(T - 2, device=device)
    D2[rows, rows] = 1.0
    D2[rows, rows + 1] = -2.0
    D2[rows, rows + 2] = 1.0
    return float(w_a if w_a is not None else 1.0) * (D2.T @ D2)


class WindowObjective:
    """Evaluates section 24's E on the real multiview window.

    One instance owns the frames, the canonical geometry and the measurement
    model, so every hypothesis is scored on exactly the same data -- section 5's
    "Both P and R must use the same frames, cameras, residuals, robust weights,
    support regularisation, and uncertainty model."
    """

    def __init__(self, mu0, R0, cameras, frames, cfg, *,
                 model_error=None, weights: ObjectiveWeights | None = None,
                 colors=None, opacity=None, scales=None, state_template=None):
        self.mu0, self.R0 = mu0, R0
        self.cameras = cameras
        self.frames = frames                      # [(obs_per_camera, frame_id)]
        self.cfg = cfg
        self.model_error = model_error
        self.w = weights or ObjectiveWeights()
        self.template = state_template
        self.n_render = 0

    def _state(self, mu, R):
        from estimator.state import GaussianMotionState
        t = self.template
        return GaussianMotionState(mu=mu, R=R if R is not None else t.R,
                                   v=t.v, omega=t.omega, colors=t.colors,
                                   opacity=t.opacity, scales=t.scales)

    @staticmethod
    def _to_device(obs, device):
        """Move one frame's observations to the device if staged on the host.

        The whole window does not fit on an 8 GiB card at 60 frames x 8
        cameras, and a per-frame transfer (~39 MiB, a few ms) is far cheaper
        than the render it feeds.
        """
        first = next((o for o in (obs or []) if o is not None), None)
        if first is None or first[0].device.type == device.type:
            return obs
        return [None if o is None else
                (o[0].to(device, non_blocking=True),
                 o[1].to(device, non_blocking=True),
                 o[2].to(device, non_blocking=True)) for o in obs]

    @torch.no_grad()
    def linearize_frame(self, mu, R, obs):
        """(linearization, precisions, residuals) for one frame.

        This is the SINGLE place the measurement model is formed. The energy,
        the trajectory normal equations of sections 11-13 and the support
        statistic of section 5.1 all read R_eff from here, so section 5's
        "Both P and R must use the same frames, cameras, residuals, robust
        weights, support regularisation, and uncertainty model" holds by
        construction rather than by two code paths agreeing.
        """
        obs = self._to_device(obs, mu.device)
        from estimator.measurement import build_precision
        from estimator.renderer_bridge import build_linearization

        lin = build_linearization(self._state(mu, R), self.cameras,
                                  eps_depth=self.cfg.eps_depth)
        self.n_render += 1
        precisions, residuals = [], []
        for ci, (view, ob) in enumerate(zip(lin.views, obs)):
            if ob is None:
                # Section 30's "missing cameras" / section 20's valid-observation
                # count: a camera with no observation at this frame contributes
                # nothing to the energy, the gradient or the Fisher. Zero
                # precision is the exact statement of "no measurement", not a
                # down-weighting.
                precisions.append((torch.zeros_like(view.rgb),
                                   torch.zeros_like(view.depth)))
                residuals.append((torch.zeros_like(view.rgb),
                                  torch.zeros_like(view.depth)))
                continue
            rgb_gt, d_gt, d_valid = ob
            rr = rgb_gt - view.rgb
            pr = build_precision(rr, valid_mask=None, sigma_rgb=self.cfg.rgb_sigma,
                                 robust=self.cfg.robust_measurement,
                                 nu=self.cfg.student_t_nu,
                                 layout=("rgb", "rgb", "rgb"), channel_dim=0)
            if self.model_error is not None:
                # Section 13: R_eff = R_sensor + R_model, applied as a precision
                # so the robust weight still multiplies on top.
                w = pr * (self.cfg.rgb_sigma ** 2)
                pr = self.model_error.effective_rgb_precision(
                    ci, self.cfg.rgb_sigma, robust_weight=w)

            rd = d_gt - view.depth
            valid = d_valid & (view.alpha > self.cfg.min_alpha_for_depth)
            pd = build_precision(
                rd.unsqueeze(0), valid_mask=valid.unsqueeze(0), sigma_rgb=(),
                depth_sigma_model=(self.cfg.depth_sigma_model
                                   if ":" in str(self.cfg.depth_sigma_model)
                                   else {"kind": self.cfg.depth_sigma_model,
                                         "a": self.cfg.depth_sigma_a,
                                         "b": self.cfg.depth_sigma_b}),
                depth_range=d_gt.unsqueeze(0), robust=self.cfg.robust_measurement,
                nu=self.cfg.student_t_nu, layout=("depth",),
                channel_dim=0).squeeze(0)
            if self.model_error is not None:
                base = pd.clamp_min(EPS)
                sensor_var = (1.0 / base) * valid.to(pd.dtype)
                pd = self.model_error.effective_depth_precision(
                    ci, sensor_var, robust_weight=valid.to(pd.dtype))

            precisions.append((pr, pd))
            residuals.append((rr, rd))
        return lin, precisions, residuals

    @torch.no_grad()
    def frame_energy(self, mu, R, obs):
        """rho( r^T R_eff^-1 r ) for one frame, summed over all cameras."""
        from estimator.measurement import measurement_energy
        lin, precisions, residuals = self.linearize_frame(mu, R, obs)
        e_rgb = e_d = 0.0
        for (pr, pd), (rr, rd) in zip(precisions, residuals):
            e_rgb += float(measurement_energy(rr, pr))
            e_d += float(measurement_energy(rd, pd))
        del lin
        return e_rgb, e_d

    @torch.no_grad()
    def frame_normal_system(self, scene: SceneModel, labels, q_t, obs):
        """Sections 11 and 13: (h_t, g_t) in the K-dimensional q coordinate.

            g_t = H_{q,t}^T R_t^-1 r_t          [K]
            h_t = H_{q,t}^T R_t^-1 H_{q,t}      [K,K]

        No renderer cross-frame Hessian is involved: each q_{k,t} enters only
        frame t's render, so the data Hessian is block diagonal in t and this
        builds one K x K block.
        """
        mu, R = scene.transport_all(self.mu0, self.R0, labels, q_t)
        lin, precisions, residuals = self.linearize_frame(mu, R, obs)
        kobs = KPartObservation(lin, scene, mu, labels)
        h_t, g_t = kobs.normal_system(precisions, residuals)
        del lin, kobs
        return h_t, g_t

    @torch.no_grad()
    def evaluate(self, scene: SceneModel, labels, q_traj, support_energy=0.0,
                 sigma_q: float = 1.0) -> WindowEnergy:
        """E over the whole window (section 24)."""
        out = WindowEnergy()
        for ti, (obs, fid) in enumerate(self.frames):
            qt = [q_traj[k][ti] for k in range(scene.K)]
            mu, R = scene.transport_all(self.mu0, self.R0, labels, qt)
            er, ed = self.frame_energy(mu, R, obs)
            out.rgb += er
            out.depth += ed
            out.per_frame.append({"frame": fid, "rgb": er, "depth": ed})
        out.rgbd = out.rgb + out.depth
        out.temporal = sum(temporal_energy(q_traj[k], sigma_q)
                           for k in range(scene.K))
        out.support = float(support_energy)
        out.total = (self.w.rgbd * out.rgbd
                     + self.w.temporal * out.temporal
                     + self.w.support * out.support)
        return out


# ---------------------------------------------------------------------------
# section 14: multi-baseline initialisation
# ---------------------------------------------------------------------------
@torch.no_grad()
def multi_baseline_motion(mu0, displacement_fn, baselines=(1, 2, 3, 5, 10)):
    """Section 14: exploit temporal separations with a better motion-to-model
    -error ratio.

    Adjacent frames on real data carry ~12.7% motion against ~87.3% model error;
    a 15-frame baseline carries ~73%. Initialisation therefore uses the LONGEST
    available baseline whose displacement is still within the renderer's basin,
    not the adjacent-frame difference.
    """
    best = None
    tried = []
    for b in baselines:
        d = displacement_fn(b)
        if d is None:
            continue
        mag = float(d.norm(dim=1).mean())
        tried.append({"baseline": b, "mean_displacement": mag})
        if best is None or mag > best[1]:
            best = (b, mag, d)
    if best is None:
        return None, {"baselines_tried": tried, "chosen": None}
    return best[2], {"baselines_tried": tried, "chosen_baseline": best[0],
                     "chosen_displacement": best[1]}


# ---------------------------------------------------------------------------
# section 20: per-part P/R hypotheses on the same objective
# ---------------------------------------------------------------------------
@torch.no_grad()
def initialise_part_models(mu0, displacement, mask, k, dt_total: float = 1.0):
    """Initialise BOTH a prismatic and a revolute hypothesis for part k.

    Section 5: the fitted twist "may only initialise eta_P, eta_R. It must not
    perform the final P/R selection."
    """
    pts = mu0[mask]
    d = displacement[mask]
    out = {}
    if pts.shape[0] < 4 or float(d.norm()) < EPS:
        return {PRISMATIC: None, REVOLUTE: None,
                "info": {"reason": "insufficient moving geometry"}}

    # prismatic: the dominant direction of the displacement field
    U, S, Vh = torch.linalg.svd(d - d.mean(dim=0, keepdim=True) * 0.0,
                                full_matrices=False)
    e = Vh[0] / Vh[0].norm().clamp_min(EPS)
    if float((d @ e).sum()) < 0:
        e = -e
    out[PRISMATIC] = PartModel(kind=PRISMATIC, axis=e, part_index=k)

    # revolute: a Plucker line fitted to the same field, as a VELOCITY
    v = d / max(dt_total, EPS)
    line, info = fit_plucker_from_motion(pts, v, min_omega=1e-5)
    out[REVOLUTE] = (None if line is None else
                     PartModel(kind=REVOLUTE, axis=line.direction,
                               moment=line.moment, part_index=k))
    out["info"] = {"prismatic_direction_share": float((S[0] ** 2) / (S ** 2).sum().clamp_min(EPS)),
                   "revolute_observable": line is not None,
                   "revolute_reason": info.get("reason"),
                   "qdot_init": info.get("qdot")}
    return out


@dataclass
class SelectionConfig:
    """Section 5's model comparison, on comparable evidence."""
    max_normalized_data_term: float = float("inf")
    min_relative_margin: float = 0.02
    # Section 18: BIC assumes regular identifiable models and the
    # revolute-to-prismatic limit is near-singular, so BIC is a DIAGNOSTIC and
    # the decision rests on the measurement-space identifiability test of
    # section 16.
    use_bic: bool = True
    # Section 0.5 / 17: only converged candidates may be compared.
    require_convergence: bool = True
    # Section 18: production selection MUST include the section 16
    # measurement-space test. Turning this off makes the result a
    # diagnostic ranking, not a production selection.
    require_identifiability: bool = True
    # Section 16: significance for the profile-likelihood curvature interval.
    alpha_identifiability: float = 0.05


@torch.no_grad()
def axis_identifiability(model, support_points: torch.Tensor) -> dict:
    """How far the fitted revolute axis is, in units of the support extent."""
    if model is None or model.kind != REVOLUTE or support_points.numel() == 0:
        return {"applicable": False}
    c = support_points.mean(dim=0)
    extent = float((support_points - c).norm(dim=1).max())
    e = model.axis
    v = c - model.line.closest_point
    d = float((v - (v @ e) * e).norm())
    ratio = d / max(extent, EPS)
    return {"applicable": True, "support_extent": extent,
            "axis_distance": d, "distance_over_extent": ratio,
            "degenerate": ratio > 0.0}


def bic(energy: float, num_parameters: int, num_residuals: int) -> float:
    return 2.0 * energy + num_parameters * math.log(max(num_residuals, 1))


def model_status(results: list) -> tuple:
    """Sections 17 and 0.5: which hypotheses may be compared at all.

    Returns (converged, status). Section 0.5 forbids a final P/R selection from
    an unfinished optimization, so a candidate that did not converge is not
    evidence for or against anything -- it is simply not comparable.
    """
    converged = [r for r in results
                 if math.isfinite(r.get("objective", float("inf")))
                 and bool(r.get("converged", False))]
    if not converged:
        return [], UNRESOLVED_OPTIMIZATION
    return converged, CONVERGED


def select_assignment(results: list, cfg: SelectionConfig,
                      curvature: list | None = None) -> dict:
    """Pick among enumerated (M_1..M_K) assignments on the SAME criterion.

    Section 5: "Do not compare unrelated heuristic scores." Every entry here is
    the same windowed objective, optionally BIC-corrected for the fact that a
    revolute part has 4 free parameters against a prismatic part's 2.

    Sections 17 and 18 govern what may be concluded from that ranking:

      * Section 17 -- only CONVERGED hypotheses may be compared. A candidate
        that did not converge yields UNRESOLVED_OPTIMIZATION, never a model.
      * Section 18 -- "BIC may remain a diagnostic, but production selection
        must include the measurement-space identifiability test above." The
        ranking therefore does not decide P vs R on its own; the section 16
        profile-likelihood verdict does, and its absence is itself a reason to
        report UNRESOLVED_MODEL rather than to fall back on BIC.

    `curvature` is section 16's per-part profile verdicts, one entry per part,
    supplied for the window rather than per candidate.

    Note a direct consequence of section 16, which this function implements
    literally: the profile test's outcomes are REVOLUTE (nonzero curvature is
    significantly supported) and UNRESOLVED_MODEL (0 lies inside the interval).
    There is no outcome that AFFIRMS prismatic, so a prismatic part is reported
    as UNRESOLVED_MODEL together with the curvature interval that bounds it.
    Section 45 requires exactly this -- "Do not force the simpler model" -- and
    forbids converting a tight interval around 0 into a PRISMATIC verdict by
    picking a smallest-curvature-that-matters, which would be a new threshold.
    """
    usable = [r for r in results if math.isfinite(r["objective"])]
    if not usable:
        return {"selected": UNRESOLVED, "reason": ["no assignment converged"],
                "status": UNRESOLVED_OPTIMIZATION, "candidates": results,
                "notes": [], "identifiability_guard_applied": False,
                "relative_margin": float("inf")}

    key = (lambda r: r["bic"]) if cfg.use_bic else (lambda r: r["objective"])
    reasons: list[str] = []      # vetoes: any entry forces UNRESOLVED
    notes: list[str] = []        # explanations that do NOT veto
    guard_applied = False

    # P0-A / section 15: the far-axis override is REMOVED from production.
    # `axis_identifiability` is still computed and persisted, but d/R is a
    # DIAGNOSTIC and never a selector -- section 15 is explicit that a true
    # revolute can have d/R >> 1 and remain identifiable when the angular sweep
    # and measurement quality are sufficient. Identifiability is decided in
    # observation space instead (section 16, `profile_curvature`).

    # ---- section 17 / 0.5: convergence gates the comparison ---------------
    comparable, conv_status = model_status(usable)
    if cfg.require_convergence and conv_status == UNRESOLVED_OPTIMIZATION:
        ranked_all = sorted(usable, key=key)
        return {"selected": UNRESOLVED, "status": UNRESOLVED_OPTIMIZATION,
                "reason": ["no candidate converged, so section 0.5 forbids "
                           "selecting a model from the ranking"],
                "notes": [f"best by {('bic' if cfg.use_bic else 'objective')} "
                          f"was {ranked_all[0]['assignment']} (diagnostic "
                          f"only, not a selection)"],
                "identifiability_guard_applied": False,
                "relative_margin": float("inf"),
                "criterion": "bic" if cfg.use_bic else "objective",
                "best": ranked_all[0], "candidates": ranked_all}
    pool = comparable if cfg.require_convergence else usable
    if cfg.require_convergence and len(pool) < len(usable):
        notes.append(f"{len(usable) - len(pool)} of {len(usable)} candidates "
                     f"did not converge and were excluded from the comparison "
                     f"(section 17)")

    ranked = sorted(pool, key=key)
    best = ranked[0]

    ok_quality = best["normalized_data_term"] <= cfg.max_normalized_data_term
    if not ok_quality:
        reasons.append(f"the best assignment does not explain the window well "
                       f"enough: normalized data term "
                       f"{best['normalized_data_term']:.5f} > "
                       f"{cfg.max_normalized_data_term:.5f}")
    margin = float("inf")
    if len(ranked) >= 2:
        a, b = key(best), key(ranked[1])
        margin = (b - a) / max(abs(b), EPS)
        if margin < cfg.min_relative_margin:
            reasons.append(f"assignments {best['assignment']} and "
                           f"{ranked[1]['assignment']} are not distinguishable "
                           f"on the same criterion (relative margin "
                           f"{margin:.5f} < {cfg.min_relative_margin:.5f})")

    # ---- section 16 / 18: the measurement-space identifiability test ------
    # This, not the ranking, decides P vs R for each moving part.
    #
    # `curvature` is indexed by PART and is a property of that part's motion,
    # not of any one candidate assignment: the profile is computed once on the
    # optimised model and its verdict holds whichever assignment happens to
    # rank first. Reading it off the best-ranked candidate would make the
    # verdict depend on the ranking it is supposed to override.
    status = CONVERGED
    if curvature is None:
        curvature = best.get("curvature") or []
    final = list(best["assignment"])
    if cfg.require_identifiability:
        for k, kind in enumerate(best["assignment"]):
            prof = curvature[k] if k < len(curvature) else None
            if prof is None:
                final[k] = UNRESOLVED
                status = UNRESOLVED_MODEL
                reasons.append(f"part {k + 1}: section 18 requires the "
                               f"measurement-space identifiability test before "
                               f"a production P/R choice, and no curvature "
                               f"profile was supplied")
                continue
            verdict = prof.get("verdict")
            if verdict == UNRESOLVED_OPTIMIZATION:
                final[k] = UNRESOLVED
                status = UNRESOLVED_OPTIMIZATION
                reasons.append(f"part {k + 1}: the curvature profile did not "
                               f"converge at the compared hypotheses "
                               f"(section 17)")
            elif verdict == REVOLUTE:
                final[k] = REVOLUTE
                notes.append(
                    f"part {k + 1}: revolute, profile likelihood ratio "
                    f"{prof.get('likelihood_ratio', float('nan')):.3f} > "
                    f"{prof.get('threshold', float('nan')):.3f} "
                    f"(section 16)")
            else:
                final[k] = UNRESOLVED
                if status == CONVERGED:
                    status = UNRESOLVED_MODEL
                lo, hi = prof.get("confidence_interval", (0.0, 0.0))
                reasons.append(
                    f"part {k + 1}: 0 lies inside the curvature interval "
                    f"[{lo:.5g}, {hi:.5g}], so the observations do not "
                    f"distinguish curved motion from its best translational "
                    f"approximation (section 16 -> UNRESOLVED_MODEL)")
        if status != CONVERGED:
            notes.append(f"the ranking still prefers {best['assignment']} on "
                         f"{'bic' if cfg.use_bic else 'objective'}; section 18 "
                         f"keeps that as a diagnostic only")

        # The section 16 verdict decides the KIND, so the result that is
        # returned must be the optimised model OF THAT KIND -- otherwise the
        # run would report "revolute" while handing back the prismatic scene
        # that happened to rank first. If the matching hypothesis exists but
        # did not converge, section 17 forbids reporting it at all.
        if status == CONVERGED and list(final) != list(best["assignment"]):
            match = next((r for r in usable
                          if list(r["assignment"]) == list(final)), None)
            if match is None:
                reasons.append(
                    f"the identifiability test selected {final} but no "
                    f"hypothesis of that assignment was optimised")
                status = UNRESOLVED_MODEL
            elif not match.get("converged", False):
                reasons.append(
                    f"the identifiability test selected {final}, but that "
                    f"hypothesis did not converge, so section 0.5 forbids "
                    f"reporting it")
                status = UNRESOLVED_OPTIMIZATION
            else:
                notes.append(f"the returned model is the {final} hypothesis, "
                             f"not the {best['assignment']} one that ranked "
                             f"first (section 18)")
                best = match
    selected = (final if cfg.require_identifiability else best["assignment"]) \
        if not reasons else UNRESOLVED
    return {"selected": selected, "reason": reasons, "notes": notes,
            "status": status if not reasons or status != CONVERGED
            else UNRESOLVED_MODEL,
            "ranking_only_assignment": best["assignment"],
            "identifiability_guard_applied": guard_applied,
            "relative_margin": margin,
            "criterion": "bic" if cfg.use_bic else "objective",
            "best": best, "candidates": ranked,
            "comparable": True}


# ---------------------------------------------------------------------------
# section 11: alternate support and joint parameters to convergence
# ---------------------------------------------------------------------------
@dataclass
class AlternationConfig:
    """Section 11's stopping criteria -- all four, not just the objective."""

    max_outer: int = 8
    tau_energy: float = 1e-4          # |E^{j+1}-E^j| / E^j
    tau_support_jaccard: float = 0.98
    tau_eta: float = 1e-3             # manifold distance between joint params
    tau_q: float = 1e-4               # max_t |q^{j+1}-q^j|
    potts_weight: float = 1.0
    static_prior_fraction: float = 0.01
    q_search_bracket: float = 0.5
    q_golden_iters: int = 24
    geometry_rounds: int = 2
    sigma_q: float = 1.0
    accept_epsilon: float = 1e-9
    # Section 12's process likelihood: W_a = 1 / (sigma_a^2 dt^4).
    sigma_a: float = 1.0
    dt: float = 1.0
    # Section 12's lambda in [H_data + H_temp + lambda I] dq = ...
    # A BUDGET, not a stopping rule: the loop stops on the section 23
    # criteria (relative energy decrease or a vanished gradient) and this
    # only bounds the work if they are never met. Measured: a 10-frame
    # window starting at q = 0 with |q_true| up to 0.19 needs 11 iterations
    # to reach a zero gradient, and stops itself there.
    trajectory_iters: int = 30
    tau_grad: float = 1e-6          # ||g|| <= tau_grad ||g_0||
    tau_grad_stall: float = 1e-3    # no-decrease at this ||g||/||g_0||
                                    # is convergence, not a stall
    lm_lambda0: float = 1e-3
    lm_lambda_min: float = 1e-9
    lm_up: float = 10.0
    lm_down: float = 3.0
    # Measured: near the bottom of a curved valley LM needs to damp well
    # past lambda = 1e3 before a step decreases the objective; at 6 trials
    # a genuinely convergent solve was reported as stalled.
    lm_max_trials: int = 14
    # Section 30 keeps the coordinate-wise search reachable so the two can
    # be compared; production uses the joint banded solve (section 11).
    trajectory_solver: str = "banded"
    # Show a per-frame bar while assembling the normal equations.
    # Display only; it changes nothing the solver computes.
    progress: bool = False


def _eta_distance(a: PartModel | None, b: PartModel | None) -> float:
    """Manifold distance between two part models (section 11's ||.||_manifold)."""
    if a is None or b is None:
        return float("inf") if a is not b else 0.0
    if a.kind != b.kind:
        return float("inf")
    c = float(torch.clamp((a.axis @ b.axis).abs(), 0.0, 1.0))
    d = math.acos(c)
    if a.kind == REVOLUTE:
        d += float((a.moment - b.moment).norm())
    return d


@torch.no_grad()
def window_trajectory_energy(objective: WindowObjective, scene: SceneModel,
                             labels, q_traj, w_a: float = 1.0, *,
                             sigma_a: float = 1.0, times=None,
                             dt: float = 1.0) -> float:
    """The TRUE nonlinear objective the trajectory solve minimises.

    Data term plus section 12's process likelihood, weighted exactly as
    `WindowObjective.evaluate` weights them, so a decrease here is a decrease of
    the reported E (section 11.C: "accept only if E_true decreases").
    """
    e = 0.0
    for ti, (obs, _fid) in enumerate(objective.frames):
        qt = [q_traj[k][ti] for k in range(scene.K)]
        mu, R = scene.transport_all(objective.mu0, objective.R0, labels, qt)
        er, ed = objective.frame_energy(mu, R, obs)
        e += er + ed
    e *= objective.w.rgbd
    e += objective.w.temporal * sum(
        temporal_energy(q_traj[k], sigma_a=sigma_a, times=times, dt=dt)
        for k in range(scene.K))
    return float(e)


@torch.no_grad()
def solve_banded_trajectory(objective: WindowObjective, scene: SceneModel,
                            labels, q_traj, cfg: AlternationConfig):
    """Sections 11, 12 and 13: ONE joint Gauss-Newton solve over all K*T
    trajectory coordinates.

        g_t    = H_{q,t}^T R_t^-1 r_t                              (section 11)
        h_t    = H_{q,t}^T R_t^-1 H_{q,t}                          (section 11)
        H_data = diag(h_0, ..., h_{T-1})                           (section 11)
        H_temp = D_2^T W_a D_2,  W_a = 1/(sigma_a^2 dt^4)          (section 12)

        [H_data + H_temp + lambda I] dq = g - H_temp q             (section 12)

    For K parts the per-frame block h_t is K x K and the temporal coupling is
    per part, so the assembled system is block-banded over K*T variables and is
    solved for all of them at once (section 13). At T = 60 and K <= 3 this is a
    180 x 180 SPD solve -- done densely and exactly, as section 12 directs
    ("For T=60, solve this small banded SPD system directly").

    This REPLACES the coordinate-wise golden-section search as the production
    solver (section 11: "Remove coordinate-wise / independent scalar searches
    as the primary production trajectory solver"). Coordinate-wise descent
    cannot move two parts together when they share pixels, and it cannot move a
    whole trajectory against the temporal prior, which couples every frame.

    lambda is the Levenberg damping that section 12's boxed equation already
    carries. It is adapted on the measured decrease of the true nonlinear
    objective -- not a tuned constant and not a threshold on any result.
    """
    T = len(objective.frames)
    K = scene.K
    if T == 0 or K == 0:
        return q_traj, {"status": "empty"}

    dev = objective.mu0.device
    times = getattr(objective, "frame_times", None)
    w_a = temporal_precision(cfg.sigma_a, cfg.dt)
    n = K * T

    # Section 12: H_temp is fixed across iterations (the prior is quadratic).
    H_temp = torch.zeros((n, n), device=dev, dtype=torch.float64)
    Hk = temporal_hessian(T, w_a, device=dev, dtype=torch.float64,
                          sigma_a=cfg.sigma_a, times=times, dt=cfg.dt)
    idx = torch.arange(T, device=dev)
    for k in range(K):
        rows = (idx * K + k).unsqueeze(1).expand(T, T)
        cols = (idx * K + k).unsqueeze(0).expand(T, T)
        H_temp[rows.reshape(-1), cols.reshape(-1)] += Hk.reshape(-1)

    def flat(qt):
        v = torch.zeros(n, device=dev, dtype=torch.float64)
        for k in range(K):
            for t in range(T):
                v[t * K + k] = float(qt[k][t])
        return v

    def unflat(v):
        return [[float(v[t * K + k]) for t in range(T)] for k in range(K)]

    lam = float(cfg.lm_lambda0)
    def energy_of(qt):
        return window_trajectory_energy(objective, scene, labels, qt,
                                        sigma_a=cfg.sigma_a, times=times,
                                        dt=cfg.dt)

    E = energy_of(q_traj)
    hist = [E]
    info = {"w_a": w_a, "energy0": E, "iterations": [], "n_variables": n,
            "solver": "banded Gauss-Newton (sections 11-13)"}

    for it in range(int(cfg.trajectory_iters)):
        # ---- section 11: per-frame data gradient and Hessian --------------
        g = torch.zeros(n, device=dev, dtype=torch.float64)
        H_data = torch.zeros((n, n), device=dev, dtype=torch.float64)
        _fr = list(enumerate(objective.frames))
        if getattr(cfg, "progress", False):
            from tools.kpart_pipeline import _progress
            _fr = _progress(_fr, f"  §23 q solve (iter {it + 1})", True)
        for t, (obs, _fid) in _fr:
            qt = [q_traj[k][t] for k in range(K)]
            h_t, g_t = objective.frame_normal_system(scene, labels, qt, obs)
            sl = slice(t * K, t * K + K)
            g[sl] = g_t
            H_data[sl, sl] = h_t

        q_vec = flat(q_traj)
        A0 = objective.w.rgbd * H_data + objective.w.temporal * H_temp
        rhs = objective.w.rgbd * g - objective.w.temporal * (H_temp @ q_vec)
        gnorm = float(rhs.norm())
        if it == 0:
            info["grad_norm0"] = gnorm

        accepted = False
        predicted = None
        for _trial in range(int(cfg.lm_max_trials)):
            A = A0 + lam * torch.eye(n, device=dev, dtype=torch.float64)
            try:
                L = torch.linalg.cholesky(A)
                dq = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)
            except Exception:
                lam *= float(cfg.lm_up)
                continue
            if predicted is None:
                # The Newton decrement: the decrease this quadratic model
                # predicts. Unlike ||g||, it is scale free and is meaningful
                # even when the solve STARTS at the optimum -- which is exactly
                # what happens when the trajectory is re-solved after a
                # geometry refinement, and what a purely relative gradient test
                # cannot see.
                predicted = 0.5 * float(dq @ rhs)
                info["predicted_decrease"] = predicted
                if predicted <= cfg.tau_energy * max(abs(E), 1.0):
                    info["status"] = "converged"
                    info["converged_on"] = "stationary"
                    info["predicted_decrease"] = predicted
                    break
            cand = unflat(q_vec + dq)
            E_new = energy_of(cand)
            if E_new < E - cfg.accept_epsilon:
                q_traj = cand
                info["iterations"].append(
                    {"iter": it, "energy": E_new, "decrease": E - E_new,
                     "lambda": lam, "step_norm": float(dq.norm()),
                     "grad_norm": gnorm, "accepted": True})
                E = E_new
                hist.append(E)
                lam = max(lam / float(cfg.lm_down), float(cfg.lm_lambda_min))
                accepted = True
                break
            lam *= float(cfg.lm_up)
        if info.get("status") == "converged":
            break
        if not accepted:
            info["iterations"].append({"iter": it, "energy": E, "decrease": 0.0,
                                       "lambda": lam, "grad_norm": gnorm,
                                       "accepted": False})
            # No step decreases the objective. That is convergence if the
            # gradient has collapsed, and a stall if it has not -- section 32
            # needs those distinguished, because a stall is
            # UNRESOLVED_OPTIMIZATION while convergence is a real answer.
            rel_g = gnorm / max(info.get("grad_norm0", gnorm), EPS)
            pred = predicted if predicted is not None else 0.0
            if pred <= cfg.tau_grad_stall * max(abs(E), 1.0):
                info["status"] = "converged"
                info["converged_on"] = "no_decrease_at_stationary_point"
            else:
                info["status"] = "stalled"
            info["grad_norm_rel"] = rel_g
            info["predicted_decrease"] = pred
            break
        if len(hist) >= 2 and abs(hist[-2] - hist[-1]) <= cfg.tau_energy * abs(hist[-2]):
            info["status"] = "converged"
            info["converged_on"] = "energy"
            break
    else:
        info["status"] = "max_iters"

    info["energy"] = E
    info["monotone"] = all(hist[i + 1] <= hist[i] + cfg.accept_epsilon
                           for i in range(len(hist) - 1))
    info["history"] = hist
    info.setdefault("grad_norm_rel",
                    (info["iterations"][-1]["grad_norm"]
                     / max(info.get("grad_norm0", 1.0), EPS))
                    if info["iterations"] else 0.0)
    info["converged"] = info.get("status") == "converged"
    return q_traj, info


@torch.no_grad()
def optimise_q_trajectory_coordinatewise(objective: WindowObjective,
                                         scene: SceneModel, labels, q_traj,
                                         cfg: AlternationConfig):
    """One 1-D golden-section search per (part, frame) on the true objective.

    NOT the production solver. Section 11 requires the joint solve
    (`solve_banded_trajectory`) because coordinate descent cannot move two
    parts together when they share pixels, and cannot move a whole trajectory
    against the temporal prior that couples every frame. This is retained so
    section 30 can MEASURE the difference between the two rather than assert
    it, and as the fallback when the banded system is not formable.
    """
    phi = 0.5 * (5 ** 0.5 - 1)
    K = scene.K
    for k in range(1, K + 1):
        if scene.parts[k - 1] is None:
            continue
        for ti, (obs, _fid) in enumerate(objective.frames):
            def E(x, _ti=ti, _k=k, _obs=obs):
                trial = [q_traj[j][_ti] for j in range(K)]
                trial[_k - 1] = float(x)
                mu, R = scene.transport_all(objective.mu0, objective.R0,
                                            labels, trial)
                er, ed = objective.frame_energy(mu, R, _obs)
                nb = [q_traj[_k - 1][i] for i in
                      range(max(0, _ti - 1), min(len(q_traj[_k - 1]), _ti + 2))]
                nb[min(_ti, 1)] = float(x)
                return er + ed + objective.w.temporal * temporal_energy(
                    nb, cfg.sigma_q)

            lo = q_traj[k - 1][ti] - cfg.q_search_bracket
            hi = q_traj[k - 1][ti] + cfg.q_search_bracket
            a, b = lo, hi
            c_, d_ = b - phi * (b - a), a + phi * (b - a)
            fc, fd = E(c_), E(d_)
            for _ in range(cfg.q_golden_iters):
                if abs(b - a) < 1e-9:
                    break
                if fc < fd:
                    b, d_, fd = d_, c_, fc
                    c_ = b - phi * (b - a)
                    fc = E(c_)
                else:
                    a, c_, fc = c_, d_, fd
                    d_ = a + phi * (b - a)
                    fd = E(d_)
            q_traj[k - 1][ti] = 0.5 * (a + b)
    return q_traj


@torch.no_grad()
def optimise_q_trajectory(objective: WindowObjective, scene: SceneModel,
                          labels, q_traj, cfg: AlternationConfig):
    """Section 11.A for the trajectory.

    Production is the joint banded Gauss-Newton solve over all K*T coordinates
    (sections 11-13). `cfg.trajectory_solver` selects the coordinate-wise
    search only for the section 30 comparison.
    """
    if str(cfg.trajectory_solver).lower().startswith("coord"):
        return optimise_q_trajectory_coordinatewise(objective, scene, labels,
                                                    q_traj, cfg)
    q_traj, info = solve_banded_trajectory(objective, scene, labels, q_traj, cfg)
    if info.get("status") == "empty":
        return optimise_q_trajectory_coordinatewise(objective, scene, labels,
                                                    q_traj, cfg)
    return q_traj


@torch.no_grad()
def refine_part_geometry(objective: WindowObjective, scene: SceneModel,
                         labels, q_traj, cfg: AlternationConfig):
    """Section 11.A for the geometry: coordinate search on each part's manifold.

    Section 5 requires the axis LOCATION to be optimised jointly with its
    direction and the q trajectory, so a revolute part searches all four of its
    degrees of freedom (2 direction + 2 moment), not just the direction.
    """
    def total(sc):
        e = 0.0
        for ti, (obs, _f) in enumerate(objective.frames):
            qt = [q_traj[j][ti] for j in range(sc.K)]
            mu, R = sc.transport_all(objective.mu0, objective.R0, labels, qt)
            er, ed = objective.frame_energy(mu, R, obs)
            e += er + ed
        return e

    best = total(scene)
    for k in range(1, scene.K + 1):
        model = scene.parts[k - 1]
        if model is None:
            continue
        e = model.axis
        tmp = torch.tensor([1.0, 0.0, 0.0], device=e.device, dtype=e.dtype)
        if abs(float(e @ tmp)) > 0.9:
            tmp = torch.tensor([0.0, 1.0, 0.0], device=e.device, dtype=e.dtype)
        u1 = torch.linalg.cross(e, tmp)
        u1 = u1 / u1.norm().clamp_min(EPS)
        u2 = torch.linalg.cross(e, u1)
        step_dir = 0.05
        step_mom = 0.05
        for _ in range(cfg.geometry_rounds):
            improved = False
            moves = [(u1, step_dir, True), (u2, step_dir, True)]
            if model.kind == REVOLUTE:
                moves += [(u1, step_mom, False), (u2, step_mom, False)]
            for basis, scale, is_dir in moves:
                for sign in (+1.0, -1.0):
                    if is_dir:
                        cand = PartModel(kind=model.kind,
                                         axis=model.axis + sign * scale * basis,
                                         moment=model.moment, part_index=k)
                    else:
                        cand = PartModel(kind=model.kind, axis=model.axis,
                                         moment=model.moment + sign * scale * basis,
                                         part_index=k)
                    trial = SceneModel(parts=list(scene.parts), K=scene.K)
                    trial.parts[k - 1] = cand
                    val = total(trial)
                    if val < best - 1e-12:
                        best, model, improved = val, cand, True
                        scene.parts[k - 1] = cand
            if not improved:
                step_dir *= 0.5
                step_mom *= 0.5
    return scene, best


@dataclass
class AlternationResult:
    scene: SceneModel
    labels: torch.Tensor
    q_traj: list
    energy: WindowEnergy
    history: list = field(default_factory=list)
    converged: bool = False
    stop_reason: str = ""
    support: object | None = None
    node_labels: torch.Tensor | None = None

    def to_dict(self) -> dict:
        return {"scene": self.scene.to_dict(),
                "labels": self.labels.detach().cpu().tolist(),
                "q_trajectories": self.q_traj,
                "energy": self.energy.to_dict(),
                "converged": self.converged,
                "stop_reason": self.stop_reason,
                "outer_iterations": len(self.history),
                "history": self.history,
                "support": (self.support.to_dict() if self.support is not None
                            else None)}


@torch.no_grad()
def alternate_to_convergence(objective: WindowObjective, partition, scene,
                             node_labels, q_traj, gain_fn,
                             cfg: AlternationConfig, K: int, *,
                             freeze_support: bool = False) -> AlternationResult:
    """Section 11's full A -> B -> C loop, run until all four criteria hold.

    C is the part that makes this honest: every proposal is re-rendered through
    the TRUE nonlinear model and accepted only if E_true does not increase.

    `freeze_support` implements section 10 Stage B: during the direct P/R
    comparison the support is held at the stabilized S so that S_P = S_R = S.
    Without it each hypothesis invents the support that flatters it, and the
    comparison measures the support search rather than the models.
    """
    from .kpart_support import KPartSupport

    # The support state is per-SUPERNODE (length C) -- that is the variable the
    # multi-label cut optimises. The renderer needs a per-GAUSSIAN expansion
    # (length N); keeping both and never confusing them is the whole point of
    # carrying `node_labels` separately.
    sid = partition.supernode_id.long().clamp_min(0)
    node_labels = node_labels.to(sid.device).long()
    labels = node_labels[sid]
    support = None
    energy = objective.evaluate(scene, labels, q_traj, sigma_q=cfg.sigma_q)
    history = [{"outer": -1, "event": "initial", **energy.to_dict(),
                "label_counts": [int((labels == k).sum()) for k in range(K + 1)]}]
    stop_reason = "max_outer_reached"
    converged = False

    for j in range(int(cfg.max_outer)):
        prev_scene = SceneModel(parts=list(scene.parts), K=K)
        prev_labels = labels.clone()
        prev_nodes = node_labels.clone()
        prev_q = [list(x) for x in q_traj]
        prev_total = energy.total

        # --- A. joint geometry and trajectory, support fixed ---------------
        q_traj = optimise_q_trajectory(objective, scene, labels, q_traj, cfg)
        scene, _ = refine_part_geometry(objective, scene, labels, q_traj, cfg)

        # --- B. support, joint fixed ---------------------------------------
        if freeze_support:
            # Section 10 Stage B: S_P = S_R = S. Only (eta, q) move here.
            energy = objective.evaluate(scene, labels, q_traj,
                                        sigma_q=cfg.sigma_q)
            history.append({"outer": j, "event": "A only (support frozen, "
                                                 "section 10 Stage B)",
                            **energy.to_dict(),
                            "label_counts": [int((labels == k).sum())
                                             for k in range(K + 1)]})
            d_eta = max((_eta_distance(prev_scene.parts[k], scene.parts[k])
                         for k in range(K)), default=0.0)
            d_q = max((abs(a - b) for k in range(K)
                       for a, b in zip(prev_q[k], q_traj[k])), default=0.0)
            rel = abs(prev_total - energy.total) / max(abs(prev_total), EPS)
            if rel <= cfg.tau_energy and d_eta <= cfg.tau_eta and d_q <= cfg.tau_q:
                converged, stop_reason = True, "converged (support frozen)"
                break
            continue
        gains = gain_fn(scene, labels, q_traj)
        sup = infer_kpart_support(partition, gains, K=K,
                                  potts_weight=cfg.potts_weight,
                                  static_prior_fraction=cfg.static_prior_fraction,
                                  initial=node_labels.cpu())
        new_nodes = sup.hard.to(sid.device).long()
        new_labels = new_nodes[sid]

        # --- C. nonlinear validation ---------------------------------------
        trial_energy = objective.evaluate(scene, new_labels, q_traj,
                                          support_energy=sup.total_energy,
                                          sigma_q=cfg.sigma_q)
        accept = trial_energy.total <= prev_total + cfg.accept_epsilon
        if accept:
            labels, node_labels, energy, support = (new_labels, new_nodes,
                                                    trial_energy, sup)
        else:
            # keep the improved geometry/trajectory, reject only the relabelling
            energy = objective.evaluate(scene, labels, q_traj,
                                        support_energy=(support.total_energy
                                                        if support else 0.0),
                                        sigma_q=cfg.sigma_q)

        # --- section 11's four stopping criteria ---------------------------
        rel_e = abs(energy.total - prev_total) / max(abs(prev_total), EPS)
        inter = int(((labels > 0) & (prev_labels > 0) &
                     (labels == prev_labels)).sum())
        union = int(((labels > 0) | (prev_labels > 0)).sum())
        jac = 1.0 if union == 0 else inter / union
        d_eta = max((_eta_distance(prev_scene.parts[k], scene.parts[k])
                     for k in range(K)), default=0.0)
        d_q = max((max((abs(a - b) for a, b in zip(prev_q[k], q_traj[k])),
                       default=0.0) for k in range(K)), default=0.0)

        history.append({
            "outer": j, "event": "accepted" if accept else "support_rejected",
            **energy.to_dict(),
            "rel_energy_change": rel_e, "support_jaccard": jac,
            "eta_change": d_eta, "q_change": d_q,
            "label_counts": [int((labels == k).sum()) for k in range(K + 1)],
            "support_energy": sup.total_energy,
            "all_parts_present": sup.all_parts_present(),
            "renders_so_far": objective.n_render})

        if (rel_e < cfg.tau_energy and jac > cfg.tau_support_jaccard
                and d_eta < cfg.tau_eta and d_q < cfg.tau_q):
            converged = True
            stop_reason = ("all four section 11 criteria satisfied: "
                           f"dE/E {rel_e:.2e} < {cfg.tau_energy:.0e}, "
                           f"Jaccard {jac:.4f} > {cfg.tau_support_jaccard}, "
                           f"d_eta {d_eta:.2e} < {cfg.tau_eta:.0e}, "
                           f"d_q {d_q:.2e} < {cfg.tau_q:.0e}")
            break

    result = AlternationResult(scene=scene, labels=labels, q_traj=q_traj,
                               energy=energy, history=history,
                               converged=converged, stop_reason=stop_reason,
                               support=support)
    result.node_labels = node_labels
    return result


# ---------------------------------------------------------------------------
# section 5 + section 20: the production discovery entry point
# ---------------------------------------------------------------------------
@dataclass
class DiscoveryConfig:
    K: int = 1
    alternation: AlternationConfig = field(default_factory=AlternationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    enumerate_limit: int = 3      # enumerate {P,R}^K exactly up to this K
    baselines: tuple = (1, 2, 3, 5, 10, 15)
    sigma_q: float = 1.0
    # Section 16: the measurement-space identifiability test. Section 18 makes
    # it mandatory for a production P/R choice, so it is on by default; turning
    # it off downgrades the run to a diagnostic ranking and `select_assignment`
    # says so.
    # Section 10 Stage B: freeze S_P = S_R = S for the direct comparison.
    freeze_support_for_pr: bool = True
    identifiability: bool = True
    identifiability_alpha: float = 0.05
    identifiability_grid: int = 11


@torch.no_grad()
def discover_kpart_joints(objective: WindowObjective, partition, grouped_fn,
                          bootstrap_displacement, cfg: DiscoveryConfig,
                          verbose: bool = True) -> dict:
    """The production replacement for twist-based P/R selection.

    Sections 4.3 and 5: the bootstrap twist enters ONLY through
    ``bootstrap_displacement``, as an initialiser. Every decision below is made
    by minimising the same windowed multiview objective.

    Section 20: for K <= enumerate_limit every assignment in {P,R}^K is fitted
    and scored; beyond that each part's model is selected locally against the
    same objective, because 2^K assignments stop being affordable.
    """
    K = cfg.K
    mu0 = objective.mu0
    dev = mu0.device
    n_res = sum(3 * c.image_height * c.image_width + c.image_height * c.image_width
                for c in objective.cameras) * len(objective.frames)

    # ---- initial labelling from hypothesis-free alpha-aware evidence ------
    gains0 = grouped_fn(None, None, None)          # [K] list of [C] gains
    sup0 = infer_kpart_support(partition, gains0, K=K,
                               potts_weight=cfg.alternation.potts_weight,
                               static_prior_fraction=cfg.alternation.static_prior_fraction)
    nodes0 = sup0.hard.to(dev).long()
    labels0 = nodes0[partition.supernode_id.long().clamp_min(0)]
    if verbose:
        print(f"  §10 initial K-part support: counts {sup0.counts()}, "
              f"all parts present = {sup0.all_parts_present()}")

    # ---- per-part P and R initialisation (section 5: initialiser only) ----
    inits = {}
    for k in range(1, K + 1):
        mask = labels0 == k
        inits[k] = initialise_part_models(mu0, bootstrap_displacement, mask, k)
        if verbose:
            i = inits[k]["info"]
            print(f"  §20 part {k}: prismatic init (direction share "
                  f"{i.get('prismatic_direction_share', float('nan')):.3f}), "
                  f"revolute init {'ok' if i.get('revolute_observable') else 'unobservable'}")

    # ---- enumerate assignments (section 20) -------------------------------
    if K <= cfg.enumerate_limit:
        assignments = list(itertools.product((PRISMATIC, REVOLUTE), repeat=K))
    else:
        assignments = [tuple(PRISMATIC if (inits[k][REVOLUTE] is None) else REVOLUTE
                             for k in range(1, K + 1))]
        if verbose:
            print(f"  §20 K={K} exceeds the enumeration limit; using local "
                  f"per-part selection")

    # ---- section 10 Stage A: stabilize the support BEFORE comparing -------
    # A reference hypothesis is alternated to convergence purely to settle the
    # support; its MODEL is then discarded. Section 10 forbids comparing P and
    # R while each invents a different support, so the comparison below starts
    # from this one S.
    stable_nodes = nodes0.clone()
    stage_a = None
    if cfg.freeze_support_for_pr:
        ref_kind = PRISMATIC if inits[1].get(PRISMATIC) is not None else REVOLUTE
        ref_parts = [inits[k].get(ref_kind) or inits[k].get(REVOLUTE)
                     or inits[k].get(PRISMATIC) for k in range(1, K + 1)]
        if all(m is not None for m in ref_parts):
            ref_scene = SceneModel(parts=ref_parts, K=K)
            q_ref = [[0.0] * len(objective.frames) for _ in range(K)]
            stage_a = alternate_to_convergence(
                objective, partition, ref_scene, nodes0.clone(), q_ref,
                lambda sc, lab, qt, _k=None: grouped_fn(sc, lab, qt),
                cfg.alternation, K)
            stable_nodes = (stage_a.node_labels if stage_a.node_labels is not None
                            else nodes0).clone()
            if verbose:
                cnt = [int((stable_nodes == k).sum()) for k in range(K + 1)]
                print(f"  §10 Stage A: support stabilized on a {ref_kind} "
                      f"reference, counts {cnt}, "
                      f"{'converged' if stage_a.converged else 'not converged'}"
                      f" ({stage_a.stop_reason}); the reference MODEL is "
                      f"discarded")

    results = []
    for assign in assignments:
        parts = []
        ok = True
        for k, kind in enumerate(assign, start=1):
            m = inits[k].get(kind)
            if m is None:
                ok = False
                break
            parts.append(m)
        if not ok:
            results.append({"assignment": list(assign), "objective": float("inf"),
                            "bic": float("inf"), "normalized_data_term": float("inf"),
                            "converged": False,
                            "reason": "a hypothesis could not be initialised"})
            continue
        scene = SceneModel(parts=parts, K=K)
        q0 = [[0.0] * len(objective.frames) for _ in range(K)]

        def gain_fn(sc, lab, qt, _k=None):
            return grouped_fn(sc, lab, qt)

        # Section 10 Stage B: every hypothesis is scored on the SAME frozen
        # support S, so the comparison is between the MODELS and not between
        # the two support searches they would otherwise each run.
        res = alternate_to_convergence(objective, partition, scene,
                                       stable_nodes.clone(), q0, gain_fn,
                                       cfg.alternation, K,
                                       freeze_support=cfg.freeze_support_for_pr)
        # section 5: record how identifiable each revolute axis actually is,
        # measured against the support the part ended up with
        ident = []
        for k in range(1, K + 1):
            mask = res.labels == k
            pts = mu0[mask] if bool(mask.any()) else mu0[:0]
            ident.append(axis_identifiability(res.scene.parts[k - 1], pts))
        results.append({
            "assignment": list(assign),
            "identifiability": ident,
            "objective": res.energy.total,
            "bic": bic(res.energy.total, scene.num_parameters + K * len(objective.frames),
                       n_res),
            "normalized_data_term": res.energy.rgbd / max(n_res, 1),
            "converged": res.converged,
            "stop_reason": res.stop_reason,
            "result": res})
        if verbose:
            idn = "".join(
                f"  axis@{g['distance_over_extent']:.2f}R" for g in ident
                if g.get("applicable"))
            print(f"  §5 assignment {list(assign)}: E = {res.energy.total:.2f} "
                  f"(RGB {res.energy.rgb:.1f}, depth {res.energy.depth:.1f}, "
                  f"temporal {res.energy.temporal:.3f}), "
                  f"BIC {results[-1]['bic']:.1f},{idn} "
                  f"{'converged' if res.converged else 'not converged'}")

    # ---- section 16: the measurement-space identifiability test -----------
    # Run ONCE per part on the best converged candidate's optimised model. The
    # verdict is a property of the part's motion, so it is computed here and
    # handed to the selector rather than being derived from the ranking it is
    # meant to override (section 18).
    curvature = None
    if cfg.identifiability:
        from .identifiability import profile_curvature
        pool = [r for r in results if r.get("converged")
                and math.isfinite(r["objective"])] or \
               [r for r in results if math.isfinite(r["objective"])]
        if pool:
            ref = min(pool, key=lambda r: r.get("bic", r["objective"]))
            rres = ref.get("result")
            curvature = []
            for k in range(1, K + 1):
                mask = rres.labels == k
                pts = mu0[mask] if bool(mask.any()) else mu0[:0]
                if rres.scene.parts[k - 1] is None or int(mask.sum()) < 3:
                    curvature.append(None)
                    continue
                prof = profile_curvature(
                    objective, rres.scene, rres.labels, k, cfg.alternation,
                    support_points=pts, q_init=rres.q_traj,
                    alpha=cfg.identifiability_alpha,
                    n_grid=cfg.identifiability_grid)
                curvature.append(prof.to_dict())
                if verbose:
                    print(f"  §16 part {k} curvature profile: kappa_hat = "
                          f"{prof.kappa_hat:.5g} (axis {1 / prof.kappa_hat:.4g} "
                          f"away)" if prof.kappa_hat else
                          f"  §16 part {k} curvature profile: kappa_hat = 0 "
                          f"(the prismatic limit)")
                    print(f"       LR = {prof.likelihood_ratio:.3f} vs "
                          f"chi2 threshold {prof.threshold:.3f} at alpha="
                          f"{prof.alpha:g}; interval "
                          f"[{prof.interval[0]:.5g}, {prof.interval[1]:.5g}] "
                          f"-> {prof.verdict}")

    decision = select_assignment(results, cfg.selection, curvature=curvature)
    if verbose:
        if decision["selected"] == UNRESOLVED:
            print(f"  §5 selection: UNRESOLVED "
                  f"({decision.get('status', '')})")
            for r in decision["reason"]:
                print(f"       - {r}")
        else:
            print(f"  §5 selection: {decision['selected']} "
                  f"(margin {decision['relative_margin']:.4f} on "
                  f"{decision['criterion']})")
        for n in decision.get("notes", []):
            print(f"       note: {n}")
    best = decision.get("best")
    # ---- section 10 Stage C: reopen support after the model is selected ----
    stage_c = None
    if (cfg.freeze_support_for_pr and best is not None
            and decision["selected"] != UNRESOLVED and best.get("result")):
        rb = best["result"]
        if verbose:
            print("  §10 Stage C: support reopened, alternating "
                  "joint <-> support to convergence")
        stage_c = alternate_to_convergence(
            objective, partition, rb.scene, rb.node_labels
            if rb.node_labels is not None else stable_nodes, rb.q_traj,
            lambda sc, lab, qt, _k=None: grouped_fn(sc, lab, qt),
            cfg.alternation, K)
        if stage_c.energy.total <= rb.energy.total + cfg.alternation.accept_epsilon:
            best["result"] = stage_c
            if verbose:
                print(f"       E {rb.energy.total:.2f} -> "
                      f"{stage_c.energy.total:.2f}, "
                      f"{'converged' if stage_c.converged else stage_c.stop_reason}")
        elif verbose:
            print(f"       rejected: E would rise "
                  f"{rb.energy.total:.2f} -> {stage_c.energy.total:.2f}")

    return {"decision": decision,
            "curvature_profiles": curvature,
            "stage_a_support": (None if stage_a is None else
                                {"converged": stage_a.converged,
                                 "stop_reason": stage_a.stop_reason,
                                 "counts": [int((stable_nodes == k).sum())
                                            for k in range(K + 1)]}),
            "stage_c_refinement": (None if stage_c is None else
                                   {"converged": stage_c.converged,
                                    "energy": stage_c.energy.total,
                                    "stop_reason": stage_c.stop_reason}),
            "result": (best or {}).get("result"),
            "initial_support": sup0,
            "candidates": [{k: v for k, v in r.items() if k != "result"}
                           for r in results],
            "num_residuals": n_res,
            "renders": objective.n_render}
