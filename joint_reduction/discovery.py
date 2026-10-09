"""Rigid-coherence detection and P/R model selection.

CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md sections 28, 29, 30.

This is the step that turns the per-Gaussian bootstrap into the permanent
representation. Section 2 lays out the ladder:

    12N Gaussian error state
        -> 6D rigid-body twist per coherent part
        -> P/R joint model
        -> low-dimensional joint state (q, qdot)

and section 30 is explicit that once the joint is known the free Gaussian
updates STOP for that part: "all moving Gaussian centers and rotations are
generated rigidly from the joint; the renderer may no longer deform them
independently."

Model selection between prismatic and revolute is done on the fitted twist
itself rather than on a separate heuristic: a twist with |omega| ~ 0 is a
translation, and one whose screw pitch h = <omega,u>/|omega|^2 is ~ 0 is a pure
rotation about an axis. The two are reported with their evidence so a weak
decision is visible as a weak decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .renderer_reduction import RenderJointModel
from .rigid_fit import fit_rigid_twist

EPS = 1e-12


@dataclass
class CoherenceReport:
    """Whether a group of Gaussians moves as one rigid body, and how certainly."""

    twist: torch.Tensor                 # [6] = [v0; omega]
    v0: torch.Tensor
    omega: torch.Tensor
    center: torch.Tensor
    residual: float                     # weighted fit residual
    relative_residual: float            # residual / observed speed scale
    condition_number: float
    rank: int
    is_static: bool
    is_coherent: bool
    n_points: int
    diagnostics: dict = field(default_factory=dict)


@dataclass
class JointHypothesis:
    kind: str
    score: float                        # lower is better: normalized residual
    axis: torch.Tensor
    point: torch.Tensor | None
    pitch: float
    omega_norm: float
    u_norm: float
    note: str = ""


@dataclass
class ReductionDecision:
    coherence: CoherenceReport
    hypotheses: list[JointHypothesis]
    selected: JointHypothesis | None
    model: RenderJointModel | None
    confident: bool
    reasons: list[str] = field(default_factory=list)
    # True when the ONLY thing standing in the way was the absolute
    # coherence-residual threshold. V2 section 18 replaces that single number
    # with a windowed, uncertainty-normalized gate, so a caller running the
    # section 18 gate may override this one -- but it must not override the
    # identifiability or model-quality vetoes, which section 18 does not touch.
    declined_only_on_coherence_residual: bool = False

    def format(self) -> str:
        c = self.coherence
        rows = [
            f"  rigid fit: |v0| {float(c.v0.norm()):.5f}  |omega| {float(c.omega.norm()):.5f}"
            f"  residual {c.residual:.4e}  rel {c.relative_residual:.4f}",
            f"             rank {c.rank}/6  cond {c.condition_number:.3e}  "
            f"static={c.is_static}  coherent={c.is_coherent}  n={c.n_points}",
        ]
        for h in self.hypotheses:
            mark = "<-- selected" if self.selected is h else ""
            rows.append(f"  hypothesis {h.kind:<10} score {h.score:.5f}  "
                        f"|omega| {h.omega_norm:.5f}  |u| {h.u_norm:.5f}  "
                        f"pitch {h.pitch:+.5f} {mark}")
            if h.note:
                rows.append(f"      {h.note}")
        rows.append(f"  => reduction {'ACCEPTED' if self.confident else 'DECLINED'}")
        for r in self.reasons:
            rows.append(f"     - {r}")
        return "\n".join(rows)


@torch.no_grad()
def assess_coherence(mu: torch.Tensor, v: torch.Tensor, weights: torch.Tensor | None = None,
                     *, static_speed: float = 1e-4, max_relative_residual: float = 0.35,
                     min_rank: int = 6) -> CoherenceReport:
    """Section 28: does this group share one rigid twist v_i = v0 + omega x (mu_i - c)?

    ``static_speed`` is in the same units as ``v`` (world units per second), so
    the caller must pass filtered velocities, not per-frame displacements.
    """
    n = int(mu.shape[0])
    if weights is None:
        weights = torch.ones(n, device=mu.device, dtype=mu.dtype)
    fit = fit_rigid_twist(mu, v, weights)
    twist = fit["twist"].reshape(-1)
    v0, omega = twist[:3], twist[3:]
    speed_scale = float((weights * v.norm(dim=1)).sum() / weights.sum().clamp_min(EPS))
    residual = float(fit["residual"])
    rel = residual / max(speed_scale, EPS)
    is_static = speed_scale < static_speed
    rank = int(fit.get("rank", 0))
    coherent = (not is_static) and rel <= max_relative_residual and rank >= min_rank
    return CoherenceReport(
        twist=twist, v0=v0, omega=omega,
        center=fit.get("center", mu.mean(dim=0)),
        residual=residual, relative_residual=rel,
        condition_number=float(fit.get("condition_number", float("inf"))),
        rank=rank, is_static=is_static, is_coherent=coherent, n_points=n,
        diagnostics={"speed_scale": speed_scale, **{k: val for k, val in fit.items()
                                                    if not torch.is_tensor(val)}})


@torch.no_grad()
def propose_joints(coh: CoherenceReport, mu: torch.Tensor,
                   v: torch.Tensor | None = None,
                   weights: torch.Tensor | None = None,
                   *, far_axis_ratio: float = 5.0) -> list[JointHypothesis]:
    """Section 29: prismatic and revolute hypotheses, scored COMPARABLY.

    The scores must measure the same thing or the comparison is meaningless.
    An earlier version scored prismatic by "rotation share" and revolute by
    "pitch share" -- different quantities on different scales -- and duly chose
    revolute for a pure translation whose filtered twist carried a small
    spurious omega. Both models are now scored by the fraction of the OBSERVED
    velocity field they fail to explain:

        prismatic   v_i ~ v0            (one constant translation)
        revolute    v_i ~ omega x (mu_i - a)   (zero-pitch twist about an axis)

    A far-away rotation axis is also penalised. Over a bounded support,
    omega x (mu - a) with |a| >> extent is indistinguishable from a constant
    translation, so a revolute model with a distant axis "explains" a
    translation while being the more complex hypothesis; parsimony says take
    the prismatic one.
    """
    omega = coh.omega
    # fit_rigid_twist solves v_i = v0 + omega x (mu_i - c), i.e. the twist is
    # referenced to the fit's own centroid c. The screw formulas below assume
    # the ORIGIN-referenced convention v(x) = u + omega x x, so convert:
    #     v(x) = v0 + omega x (x - c) = (v0 - omega x c) + omega x x.
    # The pitch is invariant under this shift (omega . (omega x c) = 0) but the
    # axis POINT is not, and using v0 directly puts the axis in the wrong place.
    u = coh.v0 - torch.linalg.cross(omega, coh.center)
    w_norm = float(omega.norm())
    u_norm = float(u.norm())
    if weights is None:
        weights = torch.ones(mu.shape[0], device=mu.device, dtype=mu.dtype)
    wsum = weights.sum().clamp_min(EPS)

    def residual_share(model_v: torch.Tensor) -> float:
        """Weighted ||v - model|| / ||v||: the share left unexplained."""
        if v is None:
            return float("nan")
        num = float((weights * (v - model_v).norm(dim=1)).sum() / wsum)
        den = float((weights * v.norm(dim=1)).sum() / wsum)
        return num / max(den, EPS)

    out: list[JointHypothesis] = []

    # --- prismatic: one constant translation ------------------------------
    v_bar = ((weights[:, None] * v).sum(dim=0) / wsum) if v is not None else coh.v0
    axis_p = v_bar / max(float(v_bar.norm()), EPS)
    score_p = (residual_share(v_bar.reshape(1, 3).expand_as(v))
               if v is not None else w_norm / max(w_norm + u_norm, EPS))
    out.append(JointHypothesis(
        kind="prismatic", score=score_p, axis=axis_p, point=None, pitch=0.0,
        omega_norm=w_norm, u_norm=float(v_bar.norm())))

    # --- revolute: zero-pitch twist about an axis -------------------------
    if w_norm > EPS:
        pitch = float((omega * u).sum() / (omega.square().sum() + EPS))
        axis_r = omega / w_norm
        point = torch.linalg.cross(omega, u) / (omega.square().sum() + EPS)
        model_v = torch.linalg.cross(omega.expand_as(mu), mu - point.reshape(1, 3))
        score_r = residual_share(model_v) if v is not None else abs(pitch) * w_norm / max(
            u_norm + abs(pitch) * w_norm, EPS)
        # Far-axis / pseudo-translation guard (see the docstring).
        extent = float((mu.max(dim=0).values - mu.min(dim=0).values).norm().clamp_min(EPS))
        centroid = (weights[:, None] * mu).sum(dim=0) / wsum
        delta = centroid - point
        perp = delta - (delta @ axis_r) * axis_r
        axis_distance = float(perp.norm()) / extent
        note = ""
        if axis_distance > far_axis_ratio:
            note = (f"axis sits {axis_distance:.1f} support-extents away: over this "
                    f"support the rotation is indistinguishable from a translation, "
                    f"so the simpler prismatic model is preferred")
            score_r = float("inf")
        out.append(JointHypothesis(
            kind="revolute", score=score_r, axis=axis_r, point=point, pitch=pitch,
            omega_norm=w_norm, u_norm=u_norm, note=note))
    else:
        out.append(JointHypothesis(
            kind="revolute", score=float("inf"),
            axis=torch.tensor([0.0, 0.0, 1.0], device=mu.device, dtype=mu.dtype),
            point=None, pitch=float("nan"), omega_norm=w_norm, u_norm=u_norm,
            note="omega is numerically zero: no rotation axis is identifiable"))
    return out


@torch.no_grad()
def decide_reduction(mu: torch.Tensor, v: torch.Tensor,
                     weights: torch.Tensor | None = None,
                     member: torch.Tensor | None = None,
                     *, margin: float = 1.5, max_unexplained: float = 0.20,
                     **kw) -> ReductionDecision:
    """The whole section 28 -> 29 step: coherence, hypotheses, and a decision.

    The decision is DECLINED rather than forced when the evidence is weak --
    section 28 wants the free per-Gaussian stage kept until rigid coherence is
    actually established, and section 32.4's principle (do not return confident
    parameters from a rank-deficient fit) applies here too.
    """
    coh = assess_coherence(mu, v, weights, **{k: val for k, val in kw.items()
                                              if k in ("static_speed",
                                                       "max_relative_residual",
                                                       "min_rank")})
    hyps = propose_joints(coh, mu, v, weights)
    reasons: list[str] = []
    if coh.is_static:
        reasons.append(f"the group is static (mean speed "
                       f"{coh.diagnostics['speed_scale']:.3e}); nothing to reduce")
    coherence_only = False
    if not coh.is_coherent and not coh.is_static:
        reasons.append(f"no single rigid twist explains the motion "
                       f"(relative residual {coh.relative_residual:.3f}, rank {coh.rank}/6)")
        coherence_only = True

    ranked = sorted(hyps, key=lambda h: h.score)
    best = ranked[0] if ranked and ranked[0].score < float("inf") else None
    confident = coh.is_coherent and best is not None
    # The winner must actually EXPLAIN the motion, not merely beat the loser.
    # A screw twist, for instance, can be "won" by prismatic while leaving a
    # third of the velocity field unexplained; section 0 rule 13 keeps helical
    # motion out of the joint family, so the honest answer is to decline and
    # keep filtering freely rather than commit to the wrong model.
    if confident and best is not None and best.score == best.score \
            and best.score > max_unexplained:
        confident = False
        reasons.append(f"the best model ({best.kind}) leaves "
                       f"{best.score*100:.0f}% of the velocity field unexplained "
                       f"(limit {max_unexplained*100:.0f}%); neither a prismatic "
                       f"nor a revolute joint describes this motion")
    if confident and len(ranked) > 1 and ranked[1].score < float("inf"):
        # Require the winner to actually win: an ambiguous P-vs-R call should
        # keep the free state rather than commit to the wrong joint.
        if ranked[0].score * margin > ranked[1].score:
            confident = False
            reasons.append(f"prismatic and revolute scores are too close "
                           f"({ranked[0].score:.4f} vs {ranked[1].score:.4f}); "
                           f"the joint type is not identifiable yet")
    if confident and best.note:
        confident = False
        reasons.append(best.note)

    model = None
    if confident:
        model = RenderJointModel(kind=best.kind, axis=best.axis, point=best.point,
                           member=member)
    # Only the residual threshold stood in the way: no other veto fired, and a
    # model could be built if a better gate says so.
    only_coherence = bool(coherence_only and best is not None
                          and len(reasons) == 1 and coh.rank >= 6)
    return ReductionDecision(coherence=coh, hypotheses=ranked, selected=best,
                             model=model, confident=confident, reasons=reasons,
                             declined_only_on_coherence_residual=only_coherence)
