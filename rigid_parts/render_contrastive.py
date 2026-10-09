"""CODEX_render_response_contrastive_outer_loss -- coordinate label UPDATES.

An addition to the soft outer step, nothing replaced. The base proposal
(render unary + soft KNN + existing temporal memory) stays exactly as it is and
remains the anchor; this refines it for a few label-only steps.

The idea it rests on: under the *currently solved and frozen* Killing motion,
two neighbouring Gaussians that react the same way to the moving-vs-static
hypothesis should move their labels in the same direction, and two that react
oppositely should move apart. Crucially the contrast acts on

    delta_l_i = l_i^ref - l_i^old            (§11)

not on the absolute label, so coordinating a correction never erases existing
confidence differences.

This is different in kind from the initializer contrast that failed earlier
(§48): that one keyed on endpoint-distance similarity, which does not imply
different rigid ownership -- two points on the same rotating part legitimately
have very different endpoint displacement. This one keys on the actual render
consequence of the current motion hypothesis. The safeguard is kept either way:
weak disagreement NEVER creates repulsion (§9, §48); only strong, confident
negative correlation makes a negative pair.

Roles stay separate (§22): temporal memory is cross-outer historical credit;
this is within-current-outer coordination. The graph is rebuilt from scratch
every outer iteration and discarded (§49).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .neighborhood import NeighborGraph

EPS = 1.0e-8


# ---------------------------------------------------------------------------
# Sections 4, 5, 26: render-response fingerprints
# ---------------------------------------------------------------------------


@dataclass
class RenderResponseFingerprints:
    """Per-Gaussian response to the frozen Killing motion."""

    vectors: torch.Tensor        # [N, D] L2-normalised, 0 where unusable
    valid: torch.Tensor          # [N, V] bool, per-observation usability
    confidence: torch.Tensor     # [N] rho_i in [0, 1]
    raw_norm: torch.Tensor       # [N] ||r_i|| before normalisation


@torch.no_grad()
def build_render_response_fingerprints(
    per_view_energy: torch.Tensor,
    per_view_valid: torch.Tensor,
    *,
    num_classes: int,
    tau_rho: float = 1.0,
    eps: float = EPS,
) -> RenderResponseFingerprints:
    """Sections 4 and 26.

    K = 1:  r_i^(v) = E_iS^(v) - E_iD^(v)               (positive -> moving better)
    K > 1:  e_i^(v) = -E_i^(v) - mean_k(-E_i^(v)),      class-centred, concatenated

    An observation counts only where BOTH the needed candidates have evidence
    (§25); a Gaussian with no usable observation gets rho_i = 0 and so forms no
    edges. Fingerprints are L2-normalised and NOT mean-centred (§7).
    """
    e = torch.as_tensor(per_view_energy)                 # [N, K+1, V]
    v = torch.as_tensor(per_view_valid, device=e.device).bool()
    n, k1, views = e.shape
    if int(num_classes) != k1:
        raise ValueError(f"per_view_energy has {k1} classes, expected {num_classes}")

    if k1 == 2:
        usable = v[:, 0] & v[:, 1]                       # [N, V]
        r = (e[:, 0] - e[:, 1]) * usable.to(e.dtype)     # E_S - E_D
    else:
        # §26: negative energies centred over classes, then flattened over views.
        neg = -e
        neg = neg - neg.mean(dim=1, keepdim=True)
        usable = v.all(dim=1)                            # [N, V]
        r = (neg * usable[:, None, :].to(e.dtype)).reshape(n, k1 * views)

    norm = r.norm(dim=1)
    vectors = r / norm.clamp_min(eps)[:, None]
    vectors = torch.where((norm > eps)[:, None], vectors, torch.zeros_like(vectors))
    # §8: response confidence from the magnitude of the response itself.
    confidence = (norm / max(float(tau_rho), eps)).clamp(0.0, 1.0)
    confidence = torch.where(usable.any(dim=1), confidence, torch.zeros_like(confidence))
    return RenderResponseFingerprints(
        vectors=vectors, valid=usable, confidence=confidence, raw_norm=norm)


# ---------------------------------------------------------------------------
# Sections 6-10: the contrast graph
# ---------------------------------------------------------------------------


@dataclass
class RenderContrastGraph:
    """Section 32."""

    positive_src: torch.Tensor
    positive_dst: torch.Tensor
    positive_weight: torch.Tensor
    negative_src: torch.Tensor
    negative_dst: torch.Tensor
    negative_weight: torch.Tensor
    response_confidence: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def num_positive(self) -> int:
        return int(self.positive_src.numel())

    @property
    def num_negative(self) -> int:
        return int(self.negative_src.numel())


@torch.no_grad()
def build_render_contrast_graph(
    fingerprints: RenderResponseFingerprints,
    spatial_graph: NeighborGraph,
    *,
    positive_threshold: float = 0.70,
    negative_threshold: float = 0.70,
    min_shared_obs: int = 2,
    eps: float = EPS,
) -> RenderContrastGraph:
    """Sections 6-10. Local KNN edges only, so cost is O(|E|), never O(N^2)."""
    src, dst = spatial_graph.source, spatial_graph.target
    device = fingerprints.vectors.device
    src = src.to(device)
    dst = dst.to(device)
    w_space = spatial_graph.weight.to(device=device, dtype=fingerprints.vectors.dtype)

    if int(src.numel()) == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return RenderContrastGraph(empty, empty, w_space.new_zeros(0), empty, empty,
                                   w_space.new_zeros(0), fingerprints.confidence,
                                   {"num_positive": 0, "num_negative": 0})

    # §5: only shared valid observations count toward the pair.
    shared = (fingerprints.valid.index_select(0, src)
              & fingerprints.valid.index_select(0, dst)).sum(dim=1)
    enough = shared >= int(min_shared_obs)

    vi = fingerprints.vectors.index_select(0, src)
    vj = fingerprints.vectors.index_select(0, dst)
    cosine = (vi * vj).sum(dim=1)                        # already unit norm (§7)

    rho_i = fingerprints.confidence.index_select(0, src)
    rho_j = fingerprints.confidence.index_select(0, dst)
    rho = torch.sqrt((rho_i * rho_j).clamp_min(0.0))     # §8

    tp = float(positive_threshold)
    tn = float(negative_threshold)
    # §10: ramp so that only clearly-extreme correlations carry real weight.
    g_pos = ((cosine - tp) / max(1.0 - tp, eps)).clamp(0.0, 1.0)
    g_neg = ((-cosine - tn) / max(1.0 - tn, eps)).clamp(0.0, 1.0)

    pos_mask = enough & (cosine > tp) & (rho > 0)
    neg_mask = enough & (cosine < -tn) & (rho > 0)
    pos_w = (w_space * rho * g_pos) * pos_mask.to(w_space.dtype)
    neg_w = (w_space * rho * g_neg) * neg_mask.to(w_space.dtype)
    pos_keep = pos_w > 0
    neg_keep = neg_w > 0

    diagnostics = {
        "num_edges_considered": int(src.numel()),
        "num_insufficient_shared_obs": int((~enough).sum()),
        "num_positive": int(pos_keep.sum()),
        "num_negative": int(neg_keep.sum()),
        "num_ambiguous": int((enough & ~pos_mask & ~neg_mask).sum()),
        "positive_edge_mean_similarity": (
            float(cosine[pos_keep].mean()) if bool(pos_keep.any()) else float("nan")),
        "negative_edge_mean_similarity": (
            float(cosine[neg_keep].mean()) if bool(neg_keep.any()) else float("nan")),
        "mean_response_confidence": float(fingerprints.confidence.mean()),
        "median_response_confidence": float(fingerprints.confidence.median()),
        "fraction_high_response_confidence": float((fingerprints.confidence > 0.5).float().mean()),
    }
    return RenderContrastGraph(
        positive_src=src[pos_keep], positive_dst=dst[pos_keep],
        positive_weight=pos_w[pos_keep],
        negative_src=src[neg_keep], negative_dst=dst[neg_keep],
        negative_weight=neg_w[neg_keep],
        response_confidence=fingerprints.confidence, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Sections 11-15, 18, 27-29: the losses
# ---------------------------------------------------------------------------


def centred_update(logits_ref: torch.Tensor, logits_old: torch.Tensor) -> torch.Tensor:
    """Sections 11 and 27: Delta L with the irrelevant common shift removed.

    For K = 1 the centred 2-vector is +-delta_l/2, so the K>1 formulation reduces
    to the binary one up to a constant factor -- one code path, both cases.
    """
    delta = logits_ref - logits_old
    return delta - delta.mean(dim=1, keepdim=True)


def render_contrastive_losses(
    logits_ref: torch.Tensor,
    logits_old: torch.Tensor,
    graph: RenderContrastGraph,
    *,
    sign_temperature: float = 0.10,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """Sections 13/14 (K=1) and 28/29 (K>1), on the CENTRED update vectors."""
    d = centred_update(logits_ref, logits_old)
    zero = logits_ref.sum() * 0.0

    if graph.num_positive:
        di = d.index_select(0, graph.positive_src)
        dj = d.index_select(0, graph.positive_dst)
        w = graph.positive_weight.to(d.dtype)
        agree = (w * (di - dj).square().sum(dim=1)).sum() / (w.sum() + float(eps))
    else:
        agree = zero

    if graph.num_negative:
        di = d.index_select(0, graph.negative_src)
        dj = d.index_select(0, graph.negative_dst)
        w = graph.negative_weight.to(d.dtype)
        # §29: cosine of the centred updates, so the penalty cannot be dodged by
        # inflating logit magnitude.
        cos = ((di * dj).sum(dim=1)
               / (di.norm(dim=1) * dj.norm(dim=1) + float(eps)))
        disagree = (w * torch.nn.functional.softplus(
            cos / max(float(sign_temperature), eps))).sum() / (w.sum() + float(eps))
    else:
        disagree = zero

    return {"agree": agree, "disagree": disagree}


def preserve_loss(
    probs_ref: torch.Tensor,
    probs_base: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 18: mean KL(P_ref || P_base). Keeps this a small correction."""
    p = probs_ref.clamp_min(eps)
    q = probs_base.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Sections 17, 19, 20, 21, 31: the refinement
# ---------------------------------------------------------------------------


@dataclass
class RenderContrastResult:
    probs: torch.Tensor
    base_probs: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


def refine_soft_labels_with_render_contrast(
    *,
    old_probs: torch.Tensor,
    base_probs: torch.Tensor,
    contrast_graph: RenderContrastGraph,
    trusted_static_mask: torch.Tensor | None = None,
    num_steps: int = 10,
    lr: float = 1.0e-2,
    lambda_preserve: float = 1.0,
    lambda_agree: float = 1.0,
    lambda_disagree: float = 0.25,
    sign_temperature: float = 0.10,
    eps: float = EPS,
) -> RenderContrastResult:
    """Sections 17-21 and 31.

    Starts from ``P_base``'s logits, never from random and never from ``P_old``
    (§17). Only the label logits are trainable; the graph, the weights and the
    fingerprints all arrive detached (§21). Trusted static anchors participate in
    pair construction but their own logits are frozen and restored (§31).
    """
    base = torch.as_tensor(base_probs)
    old = torch.as_tensor(old_probs, device=base.device, dtype=base.dtype)
    logits_base = torch.log(base.clamp_min(eps))
    # §30: the update is measured against P_old, the state entering this outer
    # iteration -- not against P_base -- so the contrast reasons about the whole
    # current correction P_old -> P_ref.
    logits_old = torch.log(old.clamp_min(eps)).detach()

    no_edges = contrast_graph.num_positive == 0 and contrast_graph.num_negative == 0
    if no_edges or (float(lambda_agree) == 0.0 and float(lambda_disagree) == 0.0) \
            or int(num_steps) <= 0:
        # §43 Test L / §42 Test H: with nothing to contrast, return P_base exactly.
        return RenderContrastResult(
            probs=base.clone(), base_probs=base,
            diagnostics={"applied": False, "reason": "no usable contrast edges",
                         "num_positive": contrast_graph.num_positive,
                         "num_negative": contrast_graph.num_negative,
                         "loss_total": float("nan"), "loss_preserve": float("nan"),
                         "loss_agree": float("nan"), "loss_disagree": float("nan"),
                         "num_argmax_changes_due_to_contrast": 0})

    logits = logits_base.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=max(float(lr), 1.0e-8))

    frozen = None
    if trusted_static_mask is not None:
        frozen = torch.as_tensor(trusted_static_mask, device=base.device).reshape(-1).bool()

    history = []
    for _ in range(max(1, int(num_steps))):
        probs = torch.softmax(logits, dim=1)
        parts = render_contrastive_losses(
            logits, logits_old, contrast_graph, sign_temperature=sign_temperature)
        pres = preserve_loss(probs, base)
        total = (float(lambda_preserve) * pres
                 + float(lambda_agree) * parts["agree"]
                 + float(lambda_disagree) * parts["disagree"])
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        history.append({"total": float(total.detach()), "preserve": float(pres.detach()),
                        "agree": float(parts["agree"].detach()),
                        "disagree": float(parts["disagree"].detach())})

    with torch.no_grad():
        refined = torch.softmax(logits.detach(), dim=1)
        if frozen is not None and bool(frozen.any()):
            # §31: anchors influenced their neighbours but never moved themselves.
            refined[frozen] = base[frozen]
        before = centred_update(logits_base, logits_old).norm(dim=1)
        after = centred_update(torch.log(refined.clamp_min(eps)), logits_old).norm(dim=1)
        changed = int((refined.argmax(dim=1) != base.argmax(dim=1)).sum())

    last = history[-1]
    return RenderContrastResult(
        probs=refined, base_probs=base,
        diagnostics={
            "applied": True,
            "num_positive": contrast_graph.num_positive,
            "num_negative": contrast_graph.num_negative,
            "loss_total": last["total"], "loss_preserve": last["preserve"],
            "loss_agree": last["agree"], "loss_disagree": last["disagree"],
            "mean_update_norm_before_contrast": float(before.mean()),
            "mean_update_norm_after_contrast": float(after.mean()),
            "num_argmax_changes_due_to_contrast": changed,
            "steps": len(history),
            **contrast_graph.diagnostics,
        })


# ---------------------------------------------------------------------------
# Section 37: evaluation-only GT diagnostics
# ---------------------------------------------------------------------------


@torch.no_grad()
def contrast_gt_diagnostics(
    graph: RenderContrastGraph,
    base_probs: torch.Tensor,
    refined_probs: torch.Tensor,
    gt_part_labels: torch.Tensor,
) -> dict[str, Any]:
    """Section 37. Never influences training or pair construction."""
    gt = (torch.as_tensor(gt_part_labels).reshape(-1) > 0).to(base_probs.device)
    out: dict[str, Any] = {}
    if graph.num_positive:
        same = (gt.index_select(0, graph.positive_src) == gt.index_select(0, graph.positive_dst))
        out["positive_edges_same_gt_class"] = float(same.float().mean() * 100.0)
    if graph.num_negative:
        diff = (gt.index_select(0, graph.negative_src) != gt.index_select(0, graph.negative_dst))
        out["negative_edges_different_gt_class"] = float(diff.float().mean() * 100.0)

    before = base_probs.argmax(dim=1) > 0
    after = refined_probs.argmax(dim=1) > 0
    moved = before != after
    if bool(moved.any()):
        was_right = before[moved] == gt[moved]
        now_right = after[moved] == gt[moved]
        out["contrast_changes_total"] = int(moved.sum())
        out["contrast_changes_improving_gt"] = int((~was_right & now_right).sum())
        out["contrast_changes_worsening_gt"] = int((was_right & ~now_right).sum())
    else:
        out["contrast_changes_total"] = 0
        out["contrast_changes_improving_gt"] = 0
        out["contrast_changes_worsening_gt"] = 0
    return out
