"""CODEX_soft_labels_outer_temporal_cycle -- soft labels are the persistent state.

    Labels are always soft outside the inner solver; they are temporarily
    argmaxed only for a Killing solve; memory exists only after observing the
    effect of a completed label-to-motion transition.              (section 55)

Three things this supersedes in the hard-label pipeline:

* Initialization no longer thresholds at ``p_dynamic > 0.5`` and no longer
  seeds logits from the rank percentile of the endpoint distance -- that
  construction put half the cloud on each side of 0.5 by definition (section 12).
  Logits start neutral and the pairwise / polarization / anchor losses decide.
* The outer loop no longer picks hard seeds and clamps them. It produces a soft
  distribution from render evidence, spatial propagation and a temporal logit
  bias (sections 21-24, 29).
* ``z = argmax(P)`` exists only for the duration of one inner Killing solve
  (sections 18, 34). ``P`` is never overwritten by its own one-hot.

The inner Killing solver is untouched (section 19).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .neighborhood import NeighborGraph

EPS = 1.0e-8


# ---------------------------------------------------------------------------
# Sections 4, 7-12: the soft binary initializer
# ---------------------------------------------------------------------------


@dataclass
class SoftBinaryResult:
    """Section 13: q stays probabilistic; nothing is thresholded."""

    logits: torch.Tensor          # [N, 2]  (static, dynamic)
    q: torch.Tensor               # [N, 2]  softmax of the logits
    history: list[dict[str, float]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    initial_logits: torch.Tensor | None = None   # corrections §33

    @property
    def q_static(self) -> torch.Tensor:
        return self.q[:, 0]

    @property
    def q_dynamic(self) -> torch.Tensor:
        return self.q[:, 1]


def binary_entropy(q: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """H(q_i) = -sum_c q_ic log q_ic, per Gaussian (section 10)."""
    p = torch.as_tensor(q)
    return -(p * torch.log(p + float(eps))).sum(dim=-1)


def pair_confidence(similarity: torch.Tensor, *, gamma: float = 2.0) -> torch.Tensor:
    """c_ij = |2 s_ij - 1|^gamma (section 9).

    Pairs near ``s_ij = 0.5`` carry no information about whether the two
    Gaussians belong together, so they are down-weighted rather than allowed to
    pull every probability toward the middle.
    """
    s = torch.as_tensor(similarity)
    return (2.0 * s - 1.0).abs().pow(float(gamma))


def positive_similarity_confidence(similarity: torch.Tensor, *, gamma: float = 2.0) -> torch.Tensor:
    """Corrections §5: c_ij = (s_ij^+)^gamma -- positive only, never signed.

    Contrast this with the superseded ``pair_confidence`` (|2s-1|^gamma), which
    gave a DISSIMILAR pair maximal weight so it could be pushed apart.
    """
    return torch.as_tensor(similarity).clamp_min(0.0).pow(float(gamma))


def positive_smoothness_loss(
    q: torch.Tensor,
    graph: NeighborGraph,
    similarity: torch.Tensor,
    *,
    gamma: float = 2.0,
    eps: float = EPS,
) -> torch.Tensor:
    """Corrections §5: L_smooth = sum w^space c_ij ||q_i - q_j||^2 / sum w^space c_ij.

    Positive-only by construction. Corrections §3: for one rigid rotating part
    ``d(x) = ||Rx + t - x||`` varies continuously with the lever arm, so two
    neighbours on the SAME part can have very different endpoint distances.
    Low ``s_ij`` therefore means *no constraint*, never "opposite labels" --
    ``s_ij -> 0`` drives ``c_ij -> 0`` and the pair simply drops out.
    """
    if graph.num_edges == 0:
        return q.sum() * 0.0
    s = torch.as_tensor(similarity, device=q.device, dtype=q.dtype)
    c = positive_similarity_confidence(s, gamma=gamma)
    w = graph.weight.to(device=q.device, dtype=q.dtype) * c
    qi = q.index_select(0, graph.source.to(q.device))
    qj = q.index_select(0, graph.target.to(q.device))
    diff = (qi - qj).square().sum(dim=-1)
    return (w * diff).sum() / (w.sum() + float(eps))


def weak_endpoint_logits(
    normalized_distance: torch.Tensor,
    *,
    epsilon_d: float = 0.2,
) -> torch.Tensor:
    """Corrections §8: a weak, zero-centred endpoint-informed asymmetry.

        r_i = 2 rankPercentile(d_i) - 1  in [-1, 1]
        l^D = +eps_d r_i / 2,  l^S = -eps_d r_i / 2   =>  l^D - l^S = eps_d r_i

    This is NOT the rank-percentile probability initialization that §8 forbids:
    that mapped the rank straight onto q and so encoded a balanced 50/50 split
    around 0.5. Here the rank only tilts the logit difference by at most
    ``eps_d``, which at 0.2 keeps even the extremes near [0.45, 0.55] -- enough
    to break the exact symmetry that left the entropy term stationary
    (corrections §7), without imposing any class ratio.
    """
    from .endpoint_contrastive_init import rank_percentile

    r = 2.0 * rank_percentile(normalized_distance) - 1.0
    half = 0.5 * float(epsilon_d) * r
    return torch.stack((-half, half), dim=1)


def polarization_weight(
    progress: float,
    *,
    lambda_max: float = 0.5,
    ramp_start: float = 0.20,
    ramp_end: float = 0.70,
) -> float:
    """Corrections §9: lambda_polar(r) = lambda_max * clip((r - r0)/(r1 - r0), 0, 1).

    Full-strength entropy minimisation from iteration 0 freezes early, uncertain
    mistakes at high confidence before the anchors and the smoothing term have
    organised the field.
    """
    r0, r1 = float(ramp_start), float(ramp_end)
    if r1 <= r0:
        return float(lambda_max) if float(progress) >= r1 else 0.0
    return float(lambda_max) * min(max((float(progress) - r0) / (r1 - r0), 0.0), 1.0)


def pairwise_contrastive_loss(
    q: torch.Tensor,
    graph: NeighborGraph,
    similarity: torch.Tensor,
    *,
    gamma: float = 2.0,
    eps: float = EPS,
) -> torch.Tensor:
    """SUPERSEDED by ``positive_smoothness_loss`` (corrections §4).

    Kept only so the superseding behaviour can be tested against it. Its
    ``(1 - s_ij) log(1 - P_same)`` term is exactly the repulsion corrections §3
    identifies as physically invalid; it is no longer used in training.
    """
    if graph.num_edges == 0:
        return q.sum() * 0.0
    s = torch.as_tensor(similarity, device=q.device, dtype=q.dtype)
    c = pair_confidence(s, gamma=gamma)
    w = graph.weight.to(device=q.device, dtype=q.dtype) * c

    qi = q.index_select(0, graph.source.to(q.device))
    qj = q.index_select(0, graph.target.to(q.device))
    p_same = (qi * qj).sum(dim=-1).clamp(0.0, 1.0)

    bce = (s * torch.log(p_same + float(eps))
           + (1.0 - s) * torch.log(1.0 - p_same + float(eps)))
    return -(w * bce).sum() / (w.sum() + float(eps))


def polarization_loss(q: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """Section 10: mean binary entropy. Essential -- the pairwise term alone
    is minimised just as well by every q sitting at [0.5, 0.5]."""
    return binary_entropy(q, eps=eps).mean()


def anchor_loss_terms(
    q: torch.Tensor,
    trusted_static: torch.Tensor,
    geometric_static: torch.Tensor,
    dynamic: torch.Tensor,
    *,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """Section 11, split so the weights can be ordered trusted > geom > dynamic."""
    def _nll(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(mask, device=q.device).reshape(-1).bool()
        if not bool(m.any()):
            return q.sum() * 0.0
        return -torch.log(values[m] + float(eps)).mean()

    return {
        "trusted_static": _nll(q[:, 0], trusted_static),
        "geometric_static": _nll(q[:, 0], geometric_static),
        "dynamic": _nll(q[:, 1], dynamic),
    }


def train_soft_binary_labels(
    graph: NeighborGraph,
    similarity: torch.Tensor,
    trusted_static: torch.Tensor,
    geometric_static: torch.Tensor,
    dynamic_anchors: torch.Tensor,
    *,
    num_gaussians: int,
    normalized_distance: torch.Tensor | None = None,
    iterations: int = 300,
    lr: float = 1.0e-2,
    gamma: float = 2.0,
    lambda_smooth: float = 1.0,
    lambda_polar: float = 0.5,
    lambda_anchor: float = 1.0,
    lambda_trusted_static: float = 2.0,
    lambda_geometric_static: float = 1.0,
    lambda_dynamic: float = 0.25,
    endpoint_logit_bias: float = 0.2,
    polarization_ramp_start: float = 0.20,
    polarization_ramp_end: float = 0.70,
    log_every: int = 10,
    log_fn=None,
    device=None,
) -> SoftBinaryResult:
    """Corrections §10: L = lambda_smooth L_smooth + lambda_anchor L_anchor
    + lambda_polar(r) L_polar.

    Three corrections relative to the superseded objective:
      * the repulsive pairwise BCE is gone, replaced by positive-only smoothing
        (corrections §3-§5);
      * logits start with a weak endpoint-informed asymmetry rather than at exact
        symmetry, where the entropy term is stationary (corrections §7-§8);
      * polarization ramps in rather than firing from iteration 0 (corrections §9).

    There is still no ``memory`` argument and none can be passed (soft-labels
    §3/§44, corrections §2).
    """
    dev = device if device is not None else graph.weight.device
    n = int(num_gaussians)
    if normalized_distance is None:
        initial = torch.zeros((n, 2), device=dev)
    else:
        initial = weak_endpoint_logits(
            torch.as_tensor(normalized_distance).to(dev), epsilon_d=endpoint_logit_bias)
    logits = initial.clone().detach().to(dev).requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=max(float(lr), 1.0e-8))

    rank = None
    if normalized_distance is not None:
        from .endpoint_contrastive_init import rank_percentile

        rank = rank_percentile(torch.as_tensor(normalized_distance).to(dev))

    history: list[dict[str, float]] = []
    steps = max(0, int(iterations))
    every = max(1, int(log_every))

    for step in range(steps + 1):
        q = torch.softmax(logits, dim=-1)
        progress = step / max(steps, 1)
        lambda_polar_now = polarization_weight(
            progress, lambda_max=lambda_polar,
            ramp_start=polarization_ramp_start, ramp_end=polarization_ramp_end)

        parts = anchor_loss_terms(q, trusted_static, geometric_static, dynamic_anchors)
        parts["smooth"] = positive_smoothness_loss(q, graph, similarity, gamma=gamma)
        parts["polar"] = polarization_loss(q)
        anchor = (float(lambda_trusted_static) * parts["trusted_static"]
                  + float(lambda_geometric_static) * parts["geometric_static"]
                  + float(lambda_dynamic) * parts["dynamic"])
        total = (float(lambda_smooth) * parts["smooth"]
                 + lambda_polar_now * parts["polar"]
                 + float(lambda_anchor) * anchor)

        if step % every == 0 or step == steps:
            with torch.no_grad():
                entropy = binary_entropy(q)
                peak = q.max(dim=-1).values
                row = {
                    "iteration": int(step),
                    "loss_total": float(total.detach()),
                    "loss_smooth": float(parts["smooth"].detach()),
                    "loss_polar": float(parts["polar"].detach()),
                    "loss_anchor": float(anchor.detach()),
                    "lambda_polar_current": float(lambda_polar_now),
                    "loss_trusted_static": float(parts["trusted_static"].detach()),
                    "loss_geometric_static": float(parts["geometric_static"].detach()),
                    "loss_dynamic": float(parts["dynamic"].detach()),
                    "mean_entropy": float(entropy.mean()),
                    "median_entropy": float(entropy.median()),
                    "mean_q_dynamic": float(q[:, 1].mean()),
                    "fraction_peak_gt_060": float((peak > 0.60).float().mean()),
                    "fraction_peak_gt_075": float((peak > 0.75).float().mean()),
                    "fraction_peak_gt_090": float((peak > 0.90).float().mean()),
                    "fraction_peak_gt_095": float((peak > 0.95).float().mean()),
                }
                if rank is not None:
                    # Corrections §11: is the learned field still ordered by
                    # endpoint distance, or did it invert it?
                    predicted_dynamic = q[:, 1] > q[:, 0]
                    row["mean_endpoint_rank_static_pred"] = (
                        float(rank[~predicted_dynamic].mean())
                        if bool((~predicted_dynamic).any()) else float("nan"))
                    row["mean_endpoint_rank_dynamic_pred"] = (
                        float(rank[predicted_dynamic].mean())
                        if bool(predicted_dynamic.any()) else float("nan"))
            history.append(row)
            if log_fn is not None:
                log_fn(f"[Soft init] it {step:4d}  L {row['loss_total']:.5f} "
                       f"(smooth {row['loss_smooth']:.5f} polar {row['loss_polar']:.5f}"
                       f"@{row['lambda_polar_current']:.3f} anchor {row['loss_anchor']:.5f})  "
                       f"H {row['mean_entropy']:.4f}  q_dyn {row['mean_q_dynamic']:.4f}  "
                       f">0.90 {row['fraction_peak_gt_090']*100:.1f}%")
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

    with torch.no_grad():
        final_logits = logits.detach().clone()
        q = torch.softmax(final_logits, dim=-1)
        entropy = binary_entropy(q)
        peak = q.max(dim=-1).values
        levels = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=q.device)
        summary = {
            "mean_entropy": float(entropy.mean()),
            "median_entropy": float(entropy.median()),
            "mean_q_dynamic": float(q[:, 1].mean()),
            "median_q_dynamic": float(q[:, 1].median()),
            "q_dynamic_quantile_levels": levels.tolist(),
            "q_dynamic_quantiles": torch.quantile(q[:, 1].float(), levels).tolist(),
            "fraction_peak_gt_060": float((peak > 0.60).float().mean()),
            "fraction_peak_gt_075": float((peak > 0.75).float().mean()),
            "fraction_peak_gt_090": float((peak > 0.90).float().mean()),
            "fraction_peak_gt_095": float((peak > 0.95).float().mean()),
            "num_trusted_static_anchors": int(torch.as_tensor(trusted_static).sum()),
            "num_geometric_static_anchors": int(torch.as_tensor(geometric_static).sum()),
            "num_dynamic_anchors": int(torch.as_tensor(dynamic_anchors).sum()),
            "iterations": steps,
            "endpoint_logit_bias": float(endpoint_logit_bias),
            "polarization_ramp": [float(polarization_ramp_start), float(polarization_ramp_end)],
            "lambda_polar_max": float(lambda_polar),
        }
        if rank is not None:
            # Corrections §12: does the learned field preserve endpoint ordering?
            summary["endpoint_rank_auc"] = float(_rank_auc(rank, q[:, 1]))
    result = SoftBinaryResult(logits=final_logits, q=q, history=history, summary=summary)
    result.initial_logits = initial.detach().cpu()
    return result


def _rank_auc(reference: torch.Tensor, scores: torch.Tensor) -> float:
    """Spearman-style agreement in [0, 1]: 1 = same ordering, 0.5 = unrelated.

    Corrections §12 asks whether the learned q_dynamic still orders Gaussians the
    way raw endpoint rank does. Implemented as a rank correlation mapped to the
    AUC scale so 0.5 reads as 'no relationship', matching the spec's framing.
    """
    a = torch.as_tensor(reference).reshape(-1).float()
    b = torch.as_tensor(scores).reshape(-1).float()
    if int(a.numel()) < 2:
        return 0.5
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = (ra.norm() * rb.norm()).clamp_min(1.0e-12)
    rho = float((ra * rb).sum() / denominator)
    return 0.5 * (rho + 1.0)


# ---------------------------------------------------------------------------
# Sections 14-16: soft clustering of the dynamic mass
# ---------------------------------------------------------------------------


@torch.no_grad()
def soft_spatial_clustering(
    xyz: torch.Tensor,
    q_dynamic: torch.Tensor,
    num_parts: int,
    *,
    iterations: int = 20,
    temperature: float = 0.0,
    seed: int = 20260919,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sections 14-15: pi_ik = softmax(-||x - mu_k||^2 / tau_K), mu from q^D pi.

    Returns ``(pi, centroids)``. No hard K-means labels are produced (section 15).
    ``temperature <= 0`` picks tau_K from the cloud scale so the softmax is neither
    saturated nor flat: the mean squared spread divided by K.
    """
    pts = torch.as_tensor(xyz).float()
    w = torch.as_tensor(q_dynamic, dtype=pts.dtype, device=pts.device).reshape(-1).clamp_min(0.0)
    k = max(1, int(num_parts))
    n = int(pts.shape[0])
    if k == 1:
        return torch.ones((n, 1), dtype=pts.dtype, device=pts.device), pts.mean(dim=0, keepdim=True)

    tau = float(temperature)
    if tau <= 0.0:
        spread = float((pts - pts.mean(dim=0, keepdim=True)).square().sum(dim=-1).mean())
        tau = max(spread / float(k), 1.0e-8)

    # Deterministic, weight-aware seeding: farthest-point on the dynamic mass.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    weights = w if float(w.sum()) > 0 else torch.ones_like(w)
    first = int(torch.multinomial(weights.float().cpu(), 1, generator=generator).item())
    centroids = [pts[first]]
    for _ in range(1, k):
        d2 = torch.cdist(pts, torch.stack(centroids)).square().min(dim=1).values
        score = weights * d2
        centroids.append(pts[int(score.argmax())])
    mu = torch.stack(centroids)

    for _ in range(max(1, int(iterations))):
        pi = torch.softmax(-torch.cdist(pts, mu).square() / tau, dim=1)      # section 14
        mass = (w[:, None] * pi)                                            # section 15
        denominator = mass.sum(dim=0).clamp_min(float(eps))
        updated = (mass.transpose(0, 1) @ pts) / denominator[:, None]
        # An emptied cluster keeps its previous centre rather than collapsing to
        # the origin, which would silently reduce the effective K.
        alive = mass.sum(dim=0) > float(eps)
        mu = torch.where(alive[:, None], updated, mu)

    pi = torch.softmax(-torch.cdist(pts, mu).square() / tau, dim=1)
    return pi, mu


def compose_soft_initial_state(
    q: torch.Tensor,
    pi: torch.Tensor | None,
    num_moving_parts: int,
) -> torch.Tensor:
    """Section 16: P_i0 = q^S, P_ik = q^D pi_ik. Sums to 1 exactly."""
    q = torch.as_tensor(q)
    k = max(1, int(num_moving_parts))
    if k == 1 or pi is None:
        # Section 16: K = 1 is exactly [q^S, q^D]; no clustering is involved.
        return q.clone()
    p = torch.as_tensor(pi, dtype=q.dtype, device=q.device)
    if int(p.shape[1]) != k:
        raise ValueError(f"pi has {int(p.shape[1])} parts, expected {k}")
    return torch.cat((q[:, :1], q[:, 1:2] * p), dim=1)


# ---------------------------------------------------------------------------
# Sections 21-24, 29-31: the soft outer update
# ---------------------------------------------------------------------------


def render_probabilities(
    energy: torch.Tensor,
    *,
    tau_render: float = 1.0,
    valid: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 21: R = softmax_k(-E_ik / tau_R).

    Hypotheses without final-frame support are pushed out of the softmax rather
    than allowed to win by having a small unmeasured energy; a Gaussian with no
    supported hypothesis at all falls back to uniform, which then contributes
    nothing but is still a valid distribution.
    """
    e = torch.as_tensor(energy)
    if not e.is_floating_point():
        e = e.float()
    logits = -e / max(float(tau_render), 1.0e-6)
    if valid is not None:
        v = torch.as_tensor(valid, device=e.device).bool()
        logits = torch.where(v, logits, torch.full_like(logits, -1.0e4))
        none_valid = ~v.any(dim=1)
        out = torch.softmax(logits, dim=1)
        if bool(none_valid.any()):
            out = out.clone()
            out[none_valid] = 1.0 / float(e.shape[1])
        return out
    return torch.softmax(logits, dim=1)


def spatial_probability_propagation(
    probabilities: torch.Tensor,
    graph: NeighborGraph,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 23: N_i = sum_j w_ij P_j / (sum_j w_ij + eps), spatial weights only.

    Render preference deliberately does NOT weaken edges here (section 23) --
    letting the evidence being corrected also shape the graph that corrects it
    makes the propagation circular. An isolated Gaussian keeps its own
    distribution instead of becoming the zero vector, which is not a distribution.
    """
    p = torch.as_tensor(probabilities)
    n = int(graph.num_nodes)
    if graph.num_edges == 0:
        return p.clone()
    src = torch.cat((graph.source, graph.target)).to(p.device)
    dst = torch.cat((graph.target, graph.source)).to(p.device)
    w = torch.cat((graph.weight, graph.weight)).to(device=p.device, dtype=p.dtype)

    numerator = torch.zeros_like(p)
    numerator.index_add_(0, src, w[:, None] * p.index_select(0, dst))
    degree = torch.zeros(n, dtype=p.dtype, device=p.device).index_add_(0, src, w)

    isolated = degree <= torch.finfo(p.dtype).tiny
    out = numerator / (degree[:, None] + float(eps))
    if bool(isolated.any()):
        out = torch.where(isolated[:, None], p, out)
    return out


def render_alpha(
    normalized_margin: torch.Tensor,
    comparable: torch.Tensor | None = None,
) -> torch.Tensor:
    """LEGACY (corrections §17): alpha = 1 - normalized_margin.

    Superseded as the default by ``validity_alpha``. This rule mapped every
    ``comparable = False`` Gaussian to alpha = 1, discarding the render evidence
    even when exactly one candidate was valid -- which is the strongest evidence
    the matcher produces, not the weakest. Reachable via
    ``--outer_confidence_mode normalized_margin`` for comparison only.
    """
    margin = torch.as_tensor(normalized_margin).reshape(-1).clamp(0.0, 1.0)
    alpha = 1.0 - margin
    if comparable is not None:
        c = torch.as_tensor(comparable, device=alpha.device).reshape(-1).bool()
        alpha = torch.where(c, alpha, torch.ones_like(alpha))
    return alpha


def validity_alpha(
    render_probs: torch.Tensor,
    num_valid: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """Corrections §15, exactly:

        alpha_i = 1                  if v_i = 0     (no render information)
                = 0                  if v_i = 1     (one candidate: strong evidence)
                = H(R_i)/log(K+1)    if v_i >= 2    (confidence from render entropy)

    The v = 1 case is the correction that matters: the superseded margin rule
    treated "only one hypothesis is supported" as *unusable* because best-vs-
    second-best is undefined, when it is in fact the matcher stating that only
    one hypothesis can explain this Gaussian at all.
    """
    r = torch.as_tensor(render_probs)
    v = torch.as_tensor(num_valid, device=r.device).reshape(-1)
    k1 = int(r.shape[1])
    normaliser = float(torch.log(torch.tensor(float(k1))))
    entropy = -(r * torch.log(r + float(eps))).sum(dim=1)
    alpha = (entropy / max(normaliser, float(eps))).clamp(0.0, 1.0)
    alpha = torch.where(v <= 0, torch.ones_like(alpha), alpha)
    alpha = torch.where(v == 1, torch.zeros_like(alpha), alpha)
    return alpha


@torch.no_grad()
def validity_diagnostics(
    num_valid: torch.Tensor,
    alpha: torch.Tensor,
    hard_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
) -> dict[str, Any]:
    """Corrections §19: the per-validity-case breakdown, required every outer."""
    v = torch.as_tensor(num_valid).reshape(-1)
    a = torch.as_tensor(alpha, device=v.device).reshape(-1)
    z = torch.as_tensor(hard_labels, device=v.device).reshape(-1)
    groups = {"valid0": v == 0, "valid1": v == 1, "valid2plus": v >= 2}

    out: dict[str, Any] = {"mean_alpha_all": float(a.mean())}
    for name, mask in groups.items():
        out[f"num_{name}"] = int(mask.sum())
        out[f"mean_alpha_{name}"] = float(a[mask].mean()) if bool(mask.any()) else float("nan")
        for c in range(int(num_classes)):
            out[f"num_{name}_class_{c}"] = int((mask & (z == c)).sum())

    if int(num_classes) == 2:
        # Corrections §19: the K=1 breakdown by WHICH candidate was valid.
        valid = torch.as_tensor(valid_mask, device=v.device)
        static_ok, moving_ok = valid[:, 0], valid[:, 1]
        out.update({
            "static_only_valid": int((static_ok & ~moving_ok).sum()),
            "moving_only_valid": int((~static_ok & moving_ok).sum()),
            "both_valid": int((static_ok & moving_ok).sum()),
            "neither_valid": int((~static_ok & ~moving_ok).sum()),
        })
    return out


@torch.no_grad()
def render_evidence_against_gt(
    render_probs: torch.Tensor,
    num_valid: torch.Tensor,
    gt_part_labels: torch.Tensor,
) -> dict[str, Any]:
    """Corrections §20, evaluation only: is one-valid-candidate evidence reliable?

    Never touches training; called after the update, on GT that the loop cannot see.
    """
    pred = torch.as_tensor(render_probs).argmax(dim=1).cpu()
    v = torch.as_tensor(num_valid).reshape(-1).cpu()
    gt = (torch.as_tensor(gt_part_labels).reshape(-1).cpu() > 0).long()
    if int(pred.numel()) != int(gt.numel()):
        return {"available": False}

    out: dict[str, Any] = {"available": True}
    for name, mask in (("valid0", v == 0), ("valid1", v == 1),
                       ("valid2plus", v >= 2), ("all", torch.ones_like(v, dtype=torch.bool))):
        if not bool(mask.any()):
            out[name] = {"n": 0}
            continue
        p = (pred[mask] > 0).long()
        g = gt[mask]
        tp = int(((p == 1) & (g == 1)).sum())
        fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        out[name] = {
            "n": int(mask.sum()),
            "accuracy": float((p == g).float().mean() * 100),
            "precision": float(prec * 100),
            "recall": float(rec * 100),
            "f1": float(2 * prec * rec / max(prec + rec, 1.0e-9) * 100),
        }
    return out


def combine_render_and_spatial(
    render_probs: torch.Tensor,
    neighbor_probs: torch.Tensor,
    alpha: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 24: U = (1 - alpha) R + alpha N, renormalised."""
    r = torch.as_tensor(render_probs)
    n = torch.as_tensor(neighbor_probs, device=r.device, dtype=r.dtype)
    a = torch.as_tensor(alpha, device=r.device, dtype=r.dtype).reshape(-1, 1)
    u = (1.0 - a) * r + a * n
    return u / u.sum(dim=1, keepdim=True).clamp_min(float(eps))


def apply_memory_in_logit_space(
    proposal: torch.Tensor,
    memory: torch.Tensor | None,
    *,
    lambda_memory: float = 0.25,
    eps: float = EPS,
) -> torch.Tensor:
    """Section 29: P = softmax(log(U + eps) + lambda_mem * M).

    Section 30: memory is a smooth logit bias on a distribution, never an
    argmin flip on an energy.
    """
    u = torch.as_tensor(proposal)
    logits = torch.log(u + float(eps))
    if memory is not None and float(lambda_memory) != 0.0:
        logits = logits + float(lambda_memory) * torch.as_tensor(
            memory, device=u.device, dtype=u.dtype)
    return torch.softmax(logits, dim=1)


def clamp_trusted_static(probabilities: torch.Tensor, trusted_mask: torch.Tensor) -> torch.Tensor:
    """Sections 22/31: trusted static anchors are the ONLY persistent hard state."""
    p = torch.as_tensor(probabilities)
    mask = torch.as_tensor(trusted_mask, device=p.device).reshape(-1).bool()
    if not bool(mask.any()):
        return p
    out = p.clone()
    out[mask] = 0.0
    out[mask, 0] = 1.0
    return out


@dataclass
class SoftOuterUpdate:
    """Section 39's artifact for one outer transition."""

    soft_before: torch.Tensor
    render_probs: torch.Tensor
    neighbor_probs: torch.Tensor
    proposal: torch.Tensor
    soft_after: torch.Tensor
    alpha: torch.Tensor
    hard_before: torch.Tensor
    hard_after: torch.Tensor
    valid_mask: torch.Tensor | None = None       # corrections §33
    num_valid: torch.Tensor | None = None        # corrections §14
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, torch.Tensor]:
        out = {
            "soft_part_probs_before": self.soft_before,
            "render_probs": self.render_probs,
            "neighbor_probs": self.neighbor_probs,
            "soft_proposal": self.proposal,
            "soft_part_probs_after": self.soft_after,
            "render_alpha": self.alpha,
            "alpha": self.alpha,
            "temporary_hard_labels_before": self.hard_before,
            "temporary_hard_labels_after": self.hard_after,
            # Section 38/39: kept only for compatibility, and it is by definition
            # argmax of the soft state -- never a separate algorithmic variable.
            "final_label": self.hard_after,
        }
        if self.valid_mask is not None:
            out["candidate_valid_mask"] = self.valid_mask
        if self.num_valid is not None:
            out["num_valid_candidates"] = self.num_valid
        return out


def soft_outer_update(
    *,
    soft_probs: torch.Tensor,
    energy: torch.Tensor,
    normalized_margin: torch.Tensor,
    comparable: torch.Tensor,
    graph: NeighborGraph,
    trusted_mask: torch.Tensor,
    memory: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    tau_render: float = 1.0,
    lambda_memory: float = 0.25,
    smoothing: float = 0.0,
    confidence_mode: str = "candidate_validity_entropy",
) -> SoftOuterUpdate:
    """Section 33's whole transition: P^(m) -> P^(m+1), all soft.

    ``confidence_mode`` selects corrections §15's validity rule (the default) or
    the superseded ``normalized_margin`` rule (§17, comparison only).
    ``smoothing`` is section 46's ``--outer_probability_smoothing``.
    """
    p_before = torch.as_tensor(soft_probs)
    r = render_probabilities(energy, tau_render=tau_render, valid=valid)
    n = spatial_probability_propagation(p_before, graph)

    num_valid = None
    if valid is not None:
        num_valid = torch.as_tensor(valid, device=p_before.device).sum(dim=1)
    mode = str(confidence_mode).strip().lower()
    if mode == "candidate_validity_entropy":
        # No validity mask supplied means every candidate is supported, i.e.
        # v = K+1 >= 2, which is exactly corrections §15's entropy branch.
        v = (num_valid if num_valid is not None
             else torch.full((int(p_before.shape[0]),), int(p_before.shape[1]),
                             device=p_before.device, dtype=torch.long))
        a = validity_alpha(r, v)
    elif mode == "normalized_margin":
        a = render_alpha(normalized_margin, comparable)
    else:
        raise ValueError(f"unknown outer_confidence_mode: {confidence_mode}")
    u = combine_render_and_spatial(r, n, a)
    p_after = apply_memory_in_logit_space(u, memory, lambda_memory=lambda_memory)

    beta = min(max(float(smoothing), 0.0), 1.0)
    if beta > 0.0:
        p_after = (1.0 - beta) * p_after + beta * p_before
        p_after = p_after / p_after.sum(dim=1, keepdim=True).clamp_min(EPS)
    p_after = clamp_trusted_static(p_after, trusted_mask)

    hard_before = p_before.argmax(dim=1)
    hard_after = p_after.argmax(dim=1)
    diagnostics = soft_update_diagnostics(p_before, p_after, hard_before, hard_after, memory)
    if num_valid is not None:
        diagnostics.update(validity_diagnostics(                       # corrections §19
            num_valid, a, hard_after,
            torch.as_tensor(valid, device=p_before.device), int(p_before.shape[1])))
    diagnostics["outer_confidence_mode"] = mode
    return SoftOuterUpdate(
        soft_before=p_before, render_probs=r, neighbor_probs=n, proposal=u,
        soft_after=p_after, alpha=a, hard_before=hard_before, hard_after=hard_after,
        valid_mask=None if valid is None else torch.as_tensor(valid, device=p_before.device),
        num_valid=num_valid, diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Sections 41-42: diagnostics
# ---------------------------------------------------------------------------


def probability_entropy(p: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    q = torch.as_tensor(p)
    return -(q * torch.log(q + float(eps))).sum(dim=-1)


@torch.no_grad()
def soft_update_diagnostics(
    before: torch.Tensor,
    after: torch.Tensor,
    hard_before: torch.Tensor,
    hard_after: torch.Tensor,
    memory: torch.Tensor | None,
) -> dict[str, Any]:
    """Section 41."""
    changed = hard_after != hard_before
    peak = after.max(dim=1).values
    return {
        "mean_entropy_before": float(probability_entropy(before).mean()),
        "mean_entropy_after": float(probability_entropy(after).mean()),
        "mean_max_probability": float(peak.mean()),
        "fraction_max_prob_gt_075": float((peak > 0.75).float().mean()),
        "fraction_max_prob_gt_090": float((peak > 0.90).float().mean()),
        "mean_l1_probability_change": float((after - before).abs().sum(dim=1).mean()),
        "num_argmax_changes": int(changed.sum()),
        "fraction_argmax_changes": float(changed.float().mean()),
        "mean_memory_magnitude": (0.0 if memory is None
                                  else float(torch.as_tensor(memory).abs().mean())),
    }


@torch.no_grad()
def soft_oscillation(
    p_now: torch.Tensor,
    p_one_back: torch.Tensor | None,
    p_two_back: torch.Tensor | None,
) -> dict[str, Any]:
    """Section 42: O_i = <P^(m) - P^(m-1), P^(m-1) - P^(m-2)>; O_i < 0 is a reversal."""
    if p_one_back is None or p_two_back is None:
        return {"fraction_soft_update_reversal": float("nan"),
                "mean_soft_update_reversal_magnitude": float("nan")}
    a = torch.as_tensor(p_now) - torch.as_tensor(p_one_back, device=p_now.device)
    b = (torch.as_tensor(p_one_back, device=p_now.device)
         - torch.as_tensor(p_two_back, device=p_now.device))
    o = (a * b).sum(dim=1)
    reversed_mask = o < 0
    return {
        "fraction_soft_update_reversal": float(reversed_mask.float().mean()),
        "mean_soft_update_reversal_magnitude": (
            float(o[reversed_mask].abs().mean()) if bool(reversed_mask.any()) else 0.0),
    }


@torch.no_grad()
def initialization_diagnostics(q: torch.Tensor, summary: dict[str, Any]) -> str:
    """Section 40's required print."""
    quantiles = "  ".join(
        f"q{l:.2f}={v:.4f}" for l, v in zip(summary["q_dynamic_quantile_levels"],
                                            summary["q_dynamic_quantiles"]))
    return "\n".join([
        "[Soft init] ===== initialization diagnostics (section 40) =====",
        f"  mean entropy q                 {summary['mean_entropy']:.5f}",
        f"  median entropy q               {summary['median_entropy']:.5f}",
        f"  fraction max(q) > 0.60         {summary['fraction_peak_gt_060']*100:6.2f}%",
        f"  fraction max(q) > 0.75         {summary['fraction_peak_gt_075']*100:6.2f}%",
        f"  fraction max(q) > 0.90         {summary['fraction_peak_gt_090']*100:6.2f}%",
        f"  fraction max(q) > 0.95         {summary['fraction_peak_gt_095']*100:6.2f}%",
        f"  mean q_dynamic                 {summary['mean_q_dynamic']:.5f}",
        f"  q_dynamic quantiles            {quantiles}",
        f"  dynamic anchor count           {summary['num_dynamic_anchors']}",
        f"  static anchor count            {summary['num_trusted_static_anchors']} trusted"
        f" + {summary['num_geometric_static_anchors']} geometric",
    ])


# ---------------------------------------------------------------------------
# Sections 44-45: explicit guards
# ---------------------------------------------------------------------------


def assert_no_memory_in_initialization(function=train_soft_binary_labels) -> None:
    """Section 44: initialization must not even be able to receive memory."""
    import inspect

    names = set(inspect.signature(function).parameters)
    offending = {n for n in names if "memory" in n.lower() or n.lower() in {"m", "credit"}}
    if offending:
        raise AssertionError(
            f"section 44 violated: {function.__name__} accepts {sorted(offending)}; "
            "initialization must contain no temporal memory")


class TemporalOrderGuard:
    """Sections 44, 45, 54: enforce the causal order at run time.

    Counts memory updates and refuses one before two solved states exist, so a
    future edit cannot quietly reintroduce a reward for ``z^(0)``.
    """

    def __init__(self) -> None:
        self.num_solved_states = 0
        self.num_memory_updates = 0

    def record_solve(self) -> None:
        self.num_solved_states += 1

    def check_memory_is_zero(self, memory: torch.Tensor) -> None:
        peak = float(torch.as_tensor(memory).abs().max()) if memory is not None else 0.0
        if self.num_solved_states <= 1 and peak != 0.0:
            raise AssertionError(
                f"section 44 violated: memory has magnitude {peak} after "
                f"{self.num_solved_states} solved state(s); it must be exactly zero "
                "until two solves can be compared")

    def allow_memory_update(self) -> bool:
        return self.num_solved_states >= 2

    def record_memory_update(self) -> None:
        if not self.allow_memory_update():
            raise AssertionError(
                "section 45 violated: a temporal memory update was attempted with "
                f"only {self.num_solved_states} solved state(s)")
        self.num_memory_updates += 1

    def as_dict(self) -> dict[str, int]:
        return {"num_solved_states": self.num_solved_states,
                "num_temporal_credit_updates": self.num_memory_updates}
