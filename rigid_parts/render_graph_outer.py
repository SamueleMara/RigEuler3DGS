"""Simple deterministic render-matching outer loop.

    render-based candidate matching -> high-confidence seeds -> coherent expansion

No learned outer objective, no composite loss, no gradient balancing, no
temperature annealing (sections 4, 16, 17, 18). The only question this mode asks
is whether render-derived Gaussian matching plus graph propagation can recover
static + K moving parts.

Section 5 is the load-bearing decision: the class ordering comes DIRECTLY from
E_ik^render. The old candidate confidence never multiplies the energy -- that
product is what previously collapsed the pixel signal to ~2% of the label
gradient. Here the margin between best and second-best IS the confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .neighborhood import NeighborGraph


@dataclass
class RenderGraphResult:
    """Everything section 21 requires, per outer iteration."""

    candidate_energy: torch.Tensor        # [N, K+1]
    candidate_valid: torch.Tensor         # [N, K+1] bool
    candidate_num_views: torch.Tensor     # [N, K+1]
    best_render_label: torch.Tensor       # [N]
    raw_margin: torch.Tensor              # [N]
    normalized_margin: torch.Tensor       # [N]
    render_preference: torch.Tensor       # [N, K+1]
    seed_mask: torch.Tensor               # [N] bool
    seed_label: torch.Tensor              # [N]
    propagated_probabilities: torch.Tensor  # [N, K+1]
    final_label: torch.Tensor             # [N]
    graph_iterations: int = 0
    graph_final_delta: float = 0.0
    graph_converged: bool = False

    def as_dict(self) -> dict:
        return {
            "candidate_energy": self.candidate_energy,
            "candidate_valid": self.candidate_valid,
            "candidate_num_views": self.candidate_num_views,
            "best_render_label": self.best_render_label,
            "raw_margin": self.raw_margin,
            "normalized_margin": self.normalized_margin,
            "render_preference": self.render_preference,
            "seed_mask": self.seed_mask,
            "seed_label": self.seed_label,
            "propagated_probabilities": self.propagated_probabilities,
            "final_label": self.final_label,
            "graph_iterations": self.graph_iterations,
            "graph_final_delta": self.graph_final_delta,
            "graph_converged": self.graph_converged,
        }


# ---------------------------------------------------------------------------
# sections 5, 6, 14: best class and its normalised margin
# ---------------------------------------------------------------------------
def render_class_preference(
    energy: torch.Tensor,
    num_views: torch.Tensor,
    *,
    min_support_views: int = 2,
    tau_render: float = 1.0,
    eps: float = 1.0e-8,
) -> dict:
    """k* = argmin_k E_ik, margin m_i = E^(2) - E^(1), robustly normalised.

    A hypothesis with fewer than ``min_support_views`` supporting final cameras
    is not a usable comparison (section 8), so it is pushed out of the ranking.
    Insufficient support makes a Gaussian a NON-SEED -- never automatically
    static.
    """
    e = torch.as_tensor(energy)
    if not e.is_floating_point():
        e = e.float()
    views = torch.as_tensor(num_views).to(e)
    k_plus_1 = int(e.shape[1])

    valid = views >= float(min_support_views)
    # Rank only among supported hypotheses.
    big = e.max().detach() + 1.0 if e.numel() else e.new_tensor(1.0)
    ranked = torch.where(valid, e, torch.full_like(e, float(big) + 1.0e3))

    order = ranked.argsort(dim=1)
    best = order[:, 0]
    e1 = ranked.gather(1, order[:, :1]).squeeze(1)
    e2 = (
        ranked.gather(1, order[:, 1:2]).squeeze(1)
        if k_plus_1 > 1
        else torch.full_like(e1, float("inf"))
    )
    raw_margin = (e2 - e1).clamp_min(0.0)

    # A Gaussian can only be ranked if at least two hypotheses are supported
    # (otherwise "best vs second best" is meaningless, section 8).
    comparable = valid.sum(dim=1) >= 2
    raw_margin = torch.where(comparable, raw_margin, torch.zeros_like(raw_margin))

    # section 6: robust P10/P90 normalisation over the comparable Gaussians.
    pool = raw_margin[comparable]
    if pool.numel() > 1:
        q10 = torch.quantile(pool, 0.10)
        q90 = torch.quantile(pool, 0.90)
    else:
        q10 = raw_margin.new_tensor(0.0)
        q90 = raw_margin.new_tensor(1.0)
    normalized = ((raw_margin - q10) / (q90 - q10 + eps)).clamp(0.0, 1.0)
    normalized = torch.where(comparable, normalized, torch.zeros_like(normalized))

    # section 10: render preference vector from robustly normalised energies.
    med = e[valid].median() if bool(valid.any()) else e.new_tensor(1.0)
    scaled = e / (med.abs() + eps)
    logits = -scaled / max(float(tau_render), 1.0e-6)
    logits = torch.where(valid, logits, torch.full_like(logits, -1.0e4))
    all_invalid = ~valid.any(dim=1)
    preference = torch.softmax(logits, dim=1)
    if bool(all_invalid.any()):
        preference[all_invalid] = 1.0 / float(k_plus_1)

    return {
        "best_label": best,
        "raw_margin": raw_margin,
        "normalized_margin": normalized,
        "preference": preference,
        "valid": valid,
        "comparable": comparable,
    }


# ---------------------------------------------------------------------------
# section 7: high-confidence render seeds
# ---------------------------------------------------------------------------
def select_render_seeds(
    best_label: torch.Tensor,
    normalized_margin: torch.Tensor,
    comparable: torch.Tensor,
    anchor_mask: torch.Tensor,
    *,
    seed_confidence: float = 0.75,
    per_class: bool = True,
    min_seeds_per_class: int = 32,
    absolute_margin_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeds = confident render classifications, plus the trusted static anchors.

    A single global margin threshold is not class-neutral. The normalised margin
    distributions differ per class, so one cut-off admits one class's marginal
    candidates while rejecting another's good ones: on USB it gave 88.4% pure
    static seeds against only 55.8% pure moving seeds, and that asymmetry is what
    capped propagation.

    ``per_class`` instead reads ``seed_confidence`` as a QUANTILE taken inside
    each class's own margin distribution, so every class contributes its most
    confident fraction and each is held to the same standard relative to itself.
    ``per_class=False`` restores the plain global threshold.

    ``absolute_margin_floor`` is the guard the per-class quantile cannot
    override: a genuinely ambiguous point (margin at or near zero) must never
    become a seed just because its class happens to be small.

    If a trusted anchor conflicts with render matching, the anchor wins.
    """
    device = normalized_margin.device
    n = int(normalized_margin.numel())
    seed_mask = torch.zeros(n, dtype=torch.bool, device=device)

    if per_class:
        q = min(max(float(seed_confidence), 0.0), 0.999999)
        for k in range(int(best_label.max()) + 1 if n else 0):
            in_class = comparable & (best_label == k)
            count = int(in_class.sum())
            if count == 0:
                continue
            margins = normalized_margin[in_class]
            if count <= int(min_seeds_per_class):
                # Too few candidates to take a quantile of; keep the ones that
                # clear the absolute floor rather than the whole class.
                seed_mask |= in_class & (normalized_margin > float(absolute_margin_floor))
                continue
            thr = torch.quantile(margins, q)
            picked = in_class & (normalized_margin >= thr)
            # Guarantee the class is represented even if its margins are flat.
            if int(picked.sum()) < int(min_seeds_per_class):
                idx = torch.nonzero(in_class, as_tuple=False).reshape(-1)
                top = idx[margins.argsort(descending=True)[: int(min_seeds_per_class)]]
                picked = picked.clone()
                picked[top] = True
            # The floor applies to the top-up too.
            seed_mask |= picked & (normalized_margin > float(absolute_margin_floor))
    else:
        seed_mask = comparable & (normalized_margin >= float(seed_confidence))
    seed_mask = seed_mask & (normalized_margin > float(absolute_margin_floor))

    seed_label = torch.where(seed_mask, best_label, torch.zeros_like(best_label))

    anchors = torch.as_tensor(anchor_mask, device=device).bool().reshape(-1)
    seed_mask = seed_mask | anchors
    seed_label = torch.where(anchors, torch.zeros_like(seed_label), seed_label)
    return seed_mask, seed_label


# ---------------------------------------------------------------------------
# sections 9, 10: boundary-aware graph weights
# ---------------------------------------------------------------------------
def build_label_graph_weights(
    graph: NeighborGraph,
    preference: torch.Tensor,
    *,
    boundary_aware: bool = False,
    sigma_floor_frac: float = 0.10,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """w_ij = w_ij^space * exp(-||R_i - R_j||^2 / (2 sigma_R^2)).

    sigma_R is the median edge preference distance, so the boundary scale comes
    from the data rather than a hand-set constant. Nearby with similar render
    preference propagates; nearby with different preference is an articulation
    boundary and the edge weakens.
    """
    if graph.num_edges == 0:
        return graph.weight.new_zeros(0)
    if not boundary_aware:
        # Purely spatial: w_ij = w_ij^space.
        #
        # Render already decides the seeds. Letting it ALSO define the graph
        # boundary makes the propagation circular -- the same evidence both
        # picks the labels and decides which neighbours may correct them, so a
        # confident mistake reinforces itself instead of being outvoted.
        return graph.weight.to(preference)
    ri = preference.index_select(0, graph.source)
    rj = preference.index_select(0, graph.target)
    dist = (ri - rj).norm(dim=-1)

    # sigma_R is data-derived, but the median alone collapses when almost every
    # edge is near-identical: it goes to ~0 and then ANY dissimilar edge is
    # annihilated, disconnecting exactly the boundary points this term exists to
    # describe. Floor it against the largest possible distance between two
    # simplex vectors (sqrt 2) so a boundary edge becomes weak, never zero.
    sigma = torch.maximum(
        dist.median(),
        dist.new_tensor(float(sigma_floor_frac) * (2.0 ** 0.5)),
    ).clamp_min(1.0e-6)
    w_render = torch.exp(-dist.square() / (2.0 * sigma * sigma))
    return graph.weight.to(preference) * w_render


# ---------------------------------------------------------------------------
# sections 11, 12, 13: render-anchored propagation
# ---------------------------------------------------------------------------
def propagate_labels(
    graph: NeighborGraph,
    edge_weight: torch.Tensor,
    preference: torch.Tensor,
    normalized_margin: torch.Tensor,
    seed_mask: torch.Tensor,
    seed_label: torch.Tensor,
    *,
    iterations: int = 1000,
    tol: float = 1.0e-4,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, int, float]:
    """P_i <- (1 - a_i) R_i + a_i * (sum_j w_ij P_j) / (sum_j w_ij), a_i = 1 - m_hat.

    Confident points follow their own render evidence; uncertain points follow
    spatially coherent neighbours. Seeds are re-clamped exactly after every
    iteration (section 12).

    ``iterations`` is a safety bound, not the stopping rule: propagation runs
    until ``max_i ||P^(n+1) - P^(n)||_inf < tol``. A fixed 50-step cut stopped
    short of the fixed point, so the labels depended on how long it happened to
    run.
    """
    n, k_plus_1 = preference.shape
    device = preference.device
    p = preference.clone()
    seeds = torch.as_tensor(seed_mask, device=device).bool().reshape(-1)
    if bool(seeds.any()):
        onehot = torch.zeros((n, k_plus_1), dtype=p.dtype, device=device)
        onehot.scatter_(1, seed_label.reshape(-1, 1).long(), 1.0)
        p = torch.where(seeds[:, None], onehot, p)

    if graph.num_edges == 0:
        return p, 0, 0.0

    src = torch.cat((graph.source, graph.target))
    dst = torch.cat((graph.target, graph.source))
    w = torch.cat((edge_weight, edge_weight)).to(p)
    degree_raw = torch.zeros(n, dtype=p.dtype, device=device).scatter_add_(0, src, w)
    # A node whose edges all vanished must fall back to its own render evidence,
    # not to a zero vector (whose argmax would silently read as class 0).
    # The aggregate is a RATIO, so the weights cancel: a tiny-but-finite degree
    # is still a well-defined neighbourhood average. Only a genuinely zero
    # degree needs the fallback, otherwise legitimately weak boundary edges
    # would be mistaken for disconnection.
    tiny = torch.finfo(p.dtype).tiny
    isolated = degree_raw <= tiny
    degree = degree_raw.clamp_min(tiny)
    alpha = (1.0 - normalized_margin.to(p)).clamp(0.0, 1.0)[:, None]

    delta = 0.0
    done = 0
    for step in range(max(0, int(iterations))):
        agg = torch.zeros_like(p)
        agg.index_add_(0, src, w[:, None] * p.index_select(0, dst))
        agg = agg / degree[:, None]
        nxt = (1.0 - alpha) * preference + alpha * agg
        if bool(isolated.any()):
            nxt = torch.where(isolated[:, None], preference, nxt)
        if bool(seeds.any()):
            nxt = torch.where(seeds[:, None], p, nxt)
        delta = float((nxt - p).abs().max())
        p = nxt
        done = step + 1
        if delta < float(tol):
            break
    return p, done, delta


def run_render_graph_outer(
    *,
    energy: torch.Tensor,
    num_views: torch.Tensor,
    graph: NeighborGraph,
    anchor_mask: torch.Tensor,
    seed_confidence: float = 0.75,
    min_support_views: int = 2,
    tau_render: float = 1.0,
    iterations: int = 1000,
    tol: float = 1.0e-4,
    boundary_aware_graph: bool = False,
    per_class_seeds: bool = True,
    min_seeds_per_class: int = 32,
    absolute_margin_floor: float = 0.05,
) -> RenderGraphResult:
    """The whole deterministic outer step (section 27)."""
    pref = render_class_preference(
        energy, num_views, min_support_views=min_support_views, tau_render=tau_render
    )
    seed_mask, seed_label = select_render_seeds(
        pref["best_label"], pref["normalized_margin"], pref["comparable"],
        anchor_mask, seed_confidence=seed_confidence,
        per_class=per_class_seeds, min_seeds_per_class=min_seeds_per_class,
        absolute_margin_floor=absolute_margin_floor,
    )
    edge_w = build_label_graph_weights(graph, pref["preference"], boundary_aware=boundary_aware_graph)
    prob, iters, delta = propagate_labels(
        graph, edge_w, pref["preference"], pref["normalized_margin"],
        seed_mask, seed_label, iterations=iterations, tol=tol,
    )
    return RenderGraphResult(
        candidate_energy=energy.detach(),
        candidate_valid=pref["valid"].detach(),
        candidate_num_views=torch.as_tensor(num_views).detach(),
        best_render_label=pref["best_label"].detach(),
        raw_margin=pref["raw_margin"].detach(),
        normalized_margin=pref["normalized_margin"].detach(),
        render_preference=pref["preference"].detach(),
        seed_mask=seed_mask.detach(),
        seed_label=seed_label.detach(),
        propagated_probabilities=prob.detach(),
        final_label=prob.argmax(dim=1).detach(),
        graph_iterations=int(iters),
        graph_final_delta=float(delta),
        graph_converged=bool(delta < float(tol)),
    )
