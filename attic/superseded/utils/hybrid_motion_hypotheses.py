"""Static-XYZ vs dynamic-relational hypothesis comparison for frame-0 labels.

Two questions, deliberately kept apart because they have different right
answers:

    XYZ        : "did this Gaussian stay where it was?"
    relational : "if it moved, where did the same rigid structure go?"

Measured on real data, raw-XYZ proximity separates moving from static far
better than rigid-invariant descriptors do (AUC 0.845 vs 0.568 on
USB_100109) -- a smooth, uniformly sampled 3DGS surface simply has no local
geometric signature to localise with, and for over half of its Gaussians
another point of the SAME cloud is a closer descriptor match than anything
in the other frame.  So the static cue is not replaced.  It is kept as the
primary hypothesis, and the relational branch is asked only the narrower
question it can actually answer: where the static explanation has already
failed, is there a confident non-local rigid-compatible region to move to?

The final label is a comparison of the two hypotheses, never a threshold on
one field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from utils.dynamic_label_refinement import (
    LABEL_AMBIGUOUS,
    LABEL_DYNAMIC,
    LABEL_STATIC,
    assign_discrete_labels,
    centered_unit_score,
    compute_prior_logit,
    ema_update,
)
from utils.hybrid_dynamic_feedback import (
    compute_context_distance_change,
    compute_internal_distance_change,
    compute_relational_dynamic_evidence,
    compute_static_context_anchors,
    saturating_score,
)


@dataclass
class HybridMotionEvidence:
    """Per-Gaussian evidence from both hypotheses, all in logit units."""

    static_mismatch: torch.Tensor        # [N] E_S, normalised by local scale
    dynamic_gate: torch.Tensor           # [N] m, soft "the static story failed"
    relational_confidence: torch.Tensor  # [N] c_rel = 1 - H/log(M)
    relational_peak_mass: torch.Tensor   # [N] max_j P_ij
    relational_evidence: torch.Tensor    # [N] confident non-local support
    candidate_displacement: torch.Tensor # [N] |expected motion| / scale
    rigidity_evidence: torch.Tensor      # [N] -g(R_int), internal metric kept?
    context_evidence: torch.Tensor       # [N] e_rel, already gated by c_rel
    cycle_evidence: torch.Tensor         # [N] reserved for later phases
    diffusion_evidence: torch.Tensor     # [N] reserved for later phases
    render_evidence: torch.Tensor        # [N] reserved for later phases
    internal_change: torch.Tensor | None = None  # [N] R_int, raw diagnostic
    context_change: torch.Tensor | None = None   # [N] C_ctx, raw diagnostic
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridWeights:
    """Logit weights. Later-phase terms default to zero until their stage lands."""

    static_mismatch: float = 2.0
    relational: float = 2.0
    rigidity: float = 0.0
    context: float = 0.0
    cycle: float = 0.0
    diffusion: float = 0.0
    render: float = 0.0


@torch.no_grad()
def compute_static_mismatch(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    effective_scale: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """``E_S``: is there plausible target geometry where this Gaussian already was?

    The nearest frame-T Gaussian in absolute position, normalised by the local
    length unit.  Small means the static explanation stands on its own; large
    means nothing is left where this point used to be, which is the only
    situation in which the dynamic branch is worth consulting.
    """
    source_xyz = torch.as_tensor(source_xyz).detach()
    target_xyz = torch.as_tensor(target_xyz).detach().to(source_xyz)
    # Both clouds are shifted by the SAME offset, so every distance is exactly
    # preserved -- but in float32 the shift is not cosmetic.  cdist evaluates
    # ||a||^2 + ||b||^2 - 2a.b, which cancels catastrophically for a cloud far
    # from the origin: a COLMAP frame centred 500 units out reported E_S ~ 0.96
    # for points that had not moved at all, i.e. a full local scale of pure
    # arithmetic error on the quantity that is supposed to detect sub-scale
    # displacement.  Centred, the same case gives ~1e-5.
    origin = source_xyz.mean(dim=0, keepdim=True)
    source_centred = source_xyz - origin
    target_centred = target_xyz - origin
    nearest: list[torch.Tensor] = []
    for start in range(0, int(source_xyz.shape[0]), max(1, int(chunk_size))):
        d2 = torch.cdist(source_centred[start:start + int(chunk_size)], target_centred).square()
        nearest.append(d2.min(dim=1).values)
    squared = torch.cat(nearest, 0).clamp_min(0.0)
    scale = torch.as_tensor(effective_scale).detach().to(squared).reshape(-1).clamp_min(1.0e-8)
    return squared / scale.square()


def compute_dynamic_gate(
    static_mismatch: torch.Tensor,
    *,
    threshold: float | None = None,
    temperature: float | None = None,
) -> tuple[torch.Tensor, float, float]:
    """``m = sigmoid((E_S - tau)/T)``, a soft "consult the dynamic branch".

    Deliberately soft, and with ``tau``/``T`` derived from the scene's own
    ``E_S`` distribution rather than hard-coded: a metric threshold would not
    transfer between datasets, and a hard gate would forbid the dynamic
    branch from ever rescuing a point the static branch narrowly explained.
    """
    value = torch.as_tensor(static_mismatch).detach()
    finite = value[torch.isfinite(value)]
    if threshold is None:
        threshold = float(finite.median()) if finite.numel() else 0.0
    if temperature is None:
        spread = float((finite - finite.median()).abs().median()) if finite.numel() else 0.0
        temperature = max(1.4826 * spread, 1.0e-6)
    return torch.sigmoid((value - float(threshold)) / float(temperature)), float(threshold), float(temperature)


@torch.no_grad()
def compute_relational_confidence(coupling) -> tuple[torch.Tensor, torch.Tensor]:
    """``(c_rel, peak_mass)`` -- how localised the dynamic coupling actually is.

    A diffuse row means the geometry could not localise the moved region.
    That must read as "do not know", never as a physical endpoint: forcing a
    near-uniform posterior into a displacement is how symmetric or repeated
    structure invents motion that is not there.
    """
    mass = (coupling.mass * coupling.valid).to(torch.float32)
    total = mass.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    normalised = mass / total
    entropy = -(normalised * torch.log(normalised.clamp_min(1.0e-12))).sum(dim=1)
    candidates = coupling.valid.sum(dim=1).clamp_min(2).to(entropy.dtype)
    confidence = (1.0 - entropy / torch.log(candidates)).clamp(0.0, 1.0)
    # A row that kept almost no matched mass is not confident, however peaked
    # the little mass it has may look.
    matched = coupling.row_mass.to(confidence) / (coupling.row_mass + coupling.unmatched_src).to(confidence).clamp_min(1.0e-12)
    return (confidence * matched).clamp(0.0, 1.0), normalised.max(dim=1).values


@torch.no_grad()
def compute_localisation_confidence(coupling, *, eps: float = 1.0e-12) -> torch.Tensor:
    """Relative cost margin between the best candidate and its best rival.

    This, not the entropy of §12, is what gates the dynamic branch.  The row
    softmax uses one global temperature derived from the whole cost matrix,
    so every row comes out nearly uniform and the normalised entropy is
    dominated by the candidate-pool size rather than by match quality:
    measured on two intrinsically identical copies, resolving the ambiguity
    completely (top-1 correctness 0.506 -> 1.000) moved the entropy
    confidence by less than 0.01, while this margin moved 0.0 -> 1.0.

    Being a ratio of costs it is also temperature-free, so it cannot be
    tuned into looking confident.
    """
    cost = torch.where(coupling.valid, coupling.total_cost, torch.full_like(coupling.total_cost, float("inf")))
    if int(cost.shape[1]) < 2:
        return torch.ones(int(cost.shape[0]), device=cost.device, dtype=cost.dtype)
    top2 = torch.topk(cost, k=2, dim=1, largest=False).values
    margin = (top2[:, 1] - top2[:, 0]) / (top2[:, 1] + top2[:, 0]).clamp_min(float(eps))
    return torch.where(torch.isfinite(margin), margin, torch.zeros_like(margin)).clamp(0.0, 1.0)


@torch.no_grad()
def compute_peak_candidate(coupling) -> torch.Tensor:
    """Index into the candidate axis of the highest-mass valid candidate."""
    mass = torch.where(coupling.valid, coupling.mass, torch.full_like(coupling.mass, -1.0))
    return mass.argmax(dim=1)


@torch.no_grad()
def compute_transported_position(
    coupling,
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    min_confidence: float = 0.05,
) -> torch.Tensor:
    """``F_D(x_i)``: the COMMITTED candidate, not the posterior mean.

    The transport is deliberately soft, so its rows stay spread across the
    whole candidate pool even when the top candidate is exactly right --
    measured on an unmoved cloud whose top-1 match was correct for every
    point, the posterior mean still sat far enough off to report a context
    distance change of 0.8 local scales.  A posterior mean over a diffuse row
    is a centroid, and a centroid is not a place anything went.

    A row that cannot localise falls back to no displacement, so an
    unresolvable point reads as "no evidence" rather than as a fabricated
    jump to the middle of its candidates.
    """
    source_xyz = torch.as_tensor(source_xyz).detach()
    target_xyz = torch.as_tensor(target_xyz).detach().to(source_xyz)
    peak = compute_peak_candidate(coupling)
    chosen = target_xyz[coupling.tgt_idx.gather(1, peak[:, None]).reshape(-1).clamp_min(0)]
    matched = coupling.valid.gather(1, peak[:, None]).reshape(-1) & (coupling.row_mass > 0)
    if confidence is not None:
        matched = matched & (confidence.to(source_xyz).reshape(-1) >= float(min_confidence))
    return torch.where(matched[:, None], chosen, source_xyz)


@torch.no_grad()
def compute_dynamic_evidence(
    coupling,
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    effective_scale: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    relational_support: torch.Tensor | None = None,
    locality_radius: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Is this point's BEST explanation somewhere it could not have stayed?

    Summing transport mass beyond a radius does not work: the row softmax
    spreads mass over the entire candidate pool by construction, so on an
    unmoved cloud with perfect top-1 matches roughly a third of every row
    still lands beyond three local scales.  That statistic measures the
    transport temperature, not motion.  The committed candidate's
    displacement does not have that failure mode.

    Returns ``(evidence, displacement of the committed candidate)``.
    """
    source_xyz = torch.as_tensor(source_xyz).detach()
    target_xyz = torch.as_tensor(target_xyz).detach().to(source_xyz)
    scale = torch.as_tensor(effective_scale).detach().to(source_xyz).reshape(-1).clamp_min(1.0e-8)

    peak = compute_peak_candidate(coupling)
    chosen = target_xyz[coupling.tgt_idx.gather(1, peak[:, None]).reshape(-1).clamp_min(0)]
    displacement = torch.linalg.norm(chosen - source_xyz, dim=-1) / scale
    matched = coupling.valid.gather(1, peak[:, None]).reshape(-1) & (coupling.row_mass > 0)

    radius = max(float(locality_radius), 1.0e-6)
    evidence = (displacement / radius).clamp_min(0.0)
    evidence = evidence.square() / (1.0 + evidence.square())   # 0.5 at the radius, saturating
    if confidence is not None:
        evidence = evidence * confidence.to(evidence).reshape(-1).clamp(0.0, 1.0)
    if relational_support is not None:
        support = relational_support.to(evidence)
        if support.dim() == 2:
            support = support.gather(1, peak[:, None]).reshape(-1)
        evidence = evidence * support.clamp(0.0, 1.0)
    evidence = torch.where(matched, evidence, torch.zeros_like(evidence))
    return evidence, torch.where(matched, displacement, torch.zeros_like(displacement))


@torch.no_grad()
def combine_hypothesis_evidence(
    evidence: HybridMotionEvidence,
    *,
    weights: HybridWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Signed logit evidence: static failure plus a corroborated dynamic story.

    Every dynamic-branch term is attenuated by the coupling confidence, so a
    geometry that could not localise contributes nothing rather than noise.
    """
    weights = weights or HybridWeights()
    static_term = weights.static_mismatch * centered_unit_score(evidence.static_mismatch)
    confidence = evidence.relational_confidence.clamp(0.0, 1.0)
    gated = evidence.dynamic_gate.clamp(0.0, 1.0) * confidence
    terms = {
        "static_mismatch": static_term,
        "relational": weights.relational * gated * centered_unit_score(evidence.relational_evidence),
        "rigidity": weights.rigidity * confidence * evidence.rigidity_evidence,
        # `context_evidence` is e_rel, which §21.3 already gates by c_rel;
        # gating it a second time here would square the attenuation and mute
        # the one term that can tell rigid motion from a bad match.
        "context": weights.context * evidence.context_evidence,
        "cycle": weights.cycle * confidence * evidence.cycle_evidence,
        "diffusion": weights.diffusion * confidence * evidence.diffusion_evidence,
        "render": weights.render * evidence.render_evidence,
    }
    total = torch.stack(list(terms.values()), dim=0).sum(dim=0)
    total = torch.where(torch.isfinite(total), total, torch.zeros_like(total))
    return total, terms


@torch.no_grad()
def update_from_immutable_prior(
    prior_probability: torch.Tensor,
    current_probability: torch.Tensor,
    combined_evidence: torch.Tensor,
    *,
    prior_logit_cap: float = 0.0,
    ema: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``p* = sigmoid(L0 + E)`` then EMA. Returns ``(updated, proposal)``.

    The base is always the immutable prior logit, never the running
    posterior: re-deriving it from the previous probability would re-add the
    same evidence every iteration, so unchanged evidence must give an
    unchanged proposal and leave only the EMA moving.
    """
    prior_logit = compute_prior_logit(prior_probability, cap=float(prior_logit_cap))
    proposal = torch.sigmoid(prior_logit + combined_evidence.to(prior_logit))
    updated = ema_update(current_probability.to(proposal), proposal, float(ema))
    return updated, proposal


def assign_static_dynamic_ambiguous(
    probability: torch.Tensor,
    *,
    tau_low: float = 0.30,
    tau_high: float = 0.70,
    relational_confidence: torch.Tensor | None = None,
    dynamic_gate: torch.Tensor | None = None,
    min_confidence: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Labels, with ambiguity also allowed when neither branch can commit.

    A point whose static story failed but whose dynamic coupling is
    unlocalised has no supported explanation either way; calling it static by
    default would hide exactly the cases this phase exists to find.
    """
    label, ambiguous = assign_discrete_labels(
        probability, ambiguous_low=float(tau_low), ambiguous_high=float(tau_high)
    )
    if relational_confidence is not None and dynamic_gate is not None and float(min_confidence) > 0.0:
        unsupported = (dynamic_gate > 0.5) & (relational_confidence < float(min_confidence))
        ambiguous = ambiguous | unsupported
        label = torch.where(unsupported, torch.full_like(label, LABEL_AMBIGUOUS), label)
    return label, ambiguous


@torch.no_grad()
def build_hybrid_evidence(
    coupling,
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    effective_scale: torch.Tensor,
    *,
    static_mismatch: torch.Tensor | None = None,
    relational_support: torch.Tensor | None = None,
    locality_radius: float = 3.0,
    gate_threshold: float | None = None,
    gate_temperature: float | None = None,
    knn_indices: torch.Tensor | None = None,
    lambda_internal: float = 1.0,
    max_anchors: int = 64,
) -> HybridMotionEvidence:
    """Assemble both branches' evidence for one coupling.

    Passing ``knn_indices`` enables the §21 context-contrast branch, which is
    the only part that can tell a rigid motion from a confidently wrong
    match; without it the dynamic branch can only report that a confident
    non-local candidate exists, not whether it is structurally credible.
    """
    if static_mismatch is None:
        static_mismatch = compute_static_mismatch(source_xyz, target_xyz, effective_scale)
    gate, threshold, temperature = compute_dynamic_gate(
        static_mismatch, threshold=gate_threshold, temperature=gate_temperature
    )
    entropy_confidence, peak = compute_relational_confidence(coupling)
    confidence = compute_localisation_confidence(coupling)
    relational, displacement = compute_dynamic_evidence(
        coupling, source_xyz, target_xyz, effective_scale, confidence=confidence,
        relational_support=relational_support, locality_radius=float(locality_radius),
    )
    zeros = torch.zeros_like(static_mismatch)

    internal_change = context_change = None
    rigidity_evidence, context_evidence = zeros.clone(), zeros.clone()
    if knn_indices is not None:
        transported = compute_transported_position(
            coupling, source_xyz, target_xyz, confidence=confidence,
        )
        internal_change, _edge_confidence = compute_internal_distance_change(
            source_xyz, transported, knn_indices, effective_scale, trust=confidence,
        )
        anchors = compute_static_context_anchors(static_mismatch)
        context_change = compute_context_distance_change(
            source_xyz, transported, anchors, effective_scale, max_anchors=int(max_anchors),
        )
        context_evidence = compute_relational_dynamic_evidence(
            context_change, internal_change, confidence, lambda_internal=float(lambda_internal),
        )
        rigidity_evidence = -saturating_score(internal_change, knee=0.5)

    return HybridMotionEvidence(
        static_mismatch=static_mismatch,
        dynamic_gate=gate,
        relational_confidence=confidence,
        relational_peak_mass=peak,
        relational_evidence=relational,
        candidate_displacement=displacement,
        rigidity_evidence=rigidity_evidence,
        context_evidence=context_evidence,
        cycle_evidence=zeros.clone(),
        diffusion_evidence=zeros.clone(),
        render_evidence=zeros.clone(),
        internal_change=internal_change,
        context_change=context_change,
        stats={
            "gate_threshold": float(threshold),
            "gate_temperature": float(temperature),
            "mean_static_mismatch": float(static_mismatch.mean()),
            "median_static_mismatch": float(static_mismatch.median()),
            "mean_dynamic_gate": float(gate.mean()),
            "mean_relational_confidence": float(confidence.mean()),
            "mean_entropy_confidence": float(entropy_confidence.mean()),
            "mean_relational_peak_mass": float(peak.mean()),
            "uniform_peak_mass": float(1.0 / max(1, int(coupling.valid.sum(dim=1).float().median()))),
            "mean_relational_evidence": float(relational.mean()),
            "locality_radius": float(locality_radius),
        },
    )
