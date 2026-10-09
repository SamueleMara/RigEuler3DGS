"""CODEX_high_precision_bootstrap_storage -- seed, do not segment.

The storage deadlock measured earlier runs like this: endpoint mismatch marks a
large false-dynamic population, global polarization hardens it to [0.01, 0.99]
before any motion evidence exists, INNER 0's moving support is then dominated by
genuinely static points, the Killing motion collapses toward identity, the static
and moving render hypotheses become indistinguishable, the outer render
probabilities sit at ~[0.5, 0.5] and the labels stop refining.

This initializer breaks that chain *before* INNER 0 by refusing to segment. It
produces only:

    ultra-confident static core  +  ultra-confident dynamic core  +  soft remainder

Everything else starts at exactly [0.5, 0.5] and stays near it. The outer
render/motion loop is the segmentation algorithm (§1, §14).

Differences from the ``soft_endpoint_contrastive`` initializer, all mandated:
  * no global entropy polarization at all -- not even ramped (§3);
  * the dynamic core is the top 2% of endpoint distance, not the top 5% (§4);
  * non-anchor Gaussians get NO signed endpoint-rank bias; their median rank must
    not decide a class sign (§7);
  * anchors are driven to explicit soft targets rather than saturated (§6).

It receives no temporal memory and cannot (§16).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .neighborhood import NeighborGraph
from .soft_labels import binary_entropy, positive_smoothness_loss

EPS = 1.0e-8


# ---------------------------------------------------------------------------
# Sections 4, 5, 23: the two confident cores
# ---------------------------------------------------------------------------


@dataclass
class BootstrapAnchors:
    """Seeds only -- never a segmentation (§1)."""

    trusted_static: torch.Tensor      # [N] bool, permanently clamped downstream
    geometric_static: torch.Tensor    # [N] bool, initialization-only evidence
    dynamic_core: torch.Tensor        # [N] bool, NOT clamped after init (§6)
    undecided: torch.Tensor           # [N] bool, everything else
    tau_static: float
    tau_dynamic: float


@torch.no_grad()
def build_high_precision_bootstrap_anchors(
    normalized_distance: torch.Tensor,
    trusted_static_mask: torch.Tensor | None,
    *,
    dynamic_quantile: float = 0.98,
    static_quantile: float = 0.10,
) -> BootstrapAnchors:
    """Sections 4 and 5.

    ``dynamic_quantile`` is the quantile itself, so 0.98 means the top 2% -- a
    deliberately extreme tail. The 5% default of the previous initializer is too
    permissive here: on storage that tail is where the coherent false-dynamic
    static regions live. The threshold is configuration-driven and never
    GT-driven (§4).

    Precedence is trusted static > dynamic core > geometric static, so a point is
    never in two cores at once and the strongest evidence always wins.
    """
    d = torch.as_tensor(normalized_distance).reshape(-1)
    n = int(d.numel())
    q_d = min(max(float(dynamic_quantile), 0.5), 1.0)
    q_s = min(max(float(static_quantile), 0.0), 0.5)
    tau_d = float(torch.quantile(d.float(), q_d))
    tau_s = float(torch.quantile(d.float(), q_s))

    if trusted_static_mask is None:
        trusted = torch.zeros(n, dtype=torch.bool, device=d.device)
    else:
        trusted = torch.as_tensor(trusted_static_mask, device=d.device).bool().reshape(-1)
        if int(trusted.numel()) != n:
            raise ValueError("trusted static mask does not match the Gaussian count")

    dynamic = (d >= tau_d) & ~trusted
    geometric = (d <= tau_s) & ~trusted & ~dynamic
    undecided = ~(trusted | geometric | dynamic)
    return BootstrapAnchors(trusted_static=trusted, geometric_static=geometric,
                            dynamic_core=dynamic, undecided=undecided,
                            tau_static=tau_s, tau_dynamic=tau_d)


# ---------------------------------------------------------------------------
# Sections 6, 8, 10: the bootstrap objective
# ---------------------------------------------------------------------------


def soft_target_cross_entropy(
    q: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """-mean_i sum_c t_c log q_ic over ``mask``; minimised exactly at q = t.

    Section 6 gives explicit anchor probabilities ([0.999, 0.001] and so on), not
    "as confident as possible". A plain -log(q_class) would saturate past those
    targets and re-introduce the premature hardening this spec exists to remove.
    """
    m = torch.as_tensor(mask, device=q.device).reshape(-1).bool()
    if not bool(m.any()):
        return q.sum() * 0.0
    t = torch.as_tensor(target, device=q.device, dtype=q.dtype).reshape(1, -1)
    return -(t * torch.log(q[m] + float(eps))).sum(dim=1).mean()


@dataclass
class BootstrapInitResult:
    """Section 23."""

    part_probs: torch.Tensor          # [N, K+1] persistent soft state
    q: torch.Tensor                   # [N, 2]
    logits: torch.Tensor              # [N, 2]
    anchors: BootstrapAnchors
    pi: torch.Tensor | None = None
    centroids: torch.Tensor | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def q_static(self) -> torch.Tensor:
        return self.q[:, 0]

    @property
    def q_dynamic(self) -> torch.Tensor:
        return self.q[:, 1]


def train_soft_bootstrap_probabilities(
    graph: NeighborGraph,
    similarity: torch.Tensor,
    anchors: BootstrapAnchors,
    *,
    num_gaussians: int,
    iterations: int = 200,
    lr: float = 1.0e-2,
    lambda_knn: float = 1.0,
    lambda_anchor: float = 1.0,
    lambda_trusted_static: float = 2.0,
    lambda_geometric_static: float = 0.5,
    lambda_dynamic_core: float = 1.0,
    trusted_static_target: float = 0.001,
    geometric_static_target: float = 0.01,
    dynamic_core_target: float = 0.01,
    lambda_polar: float = 0.0,
    log_every: int = 10,
    log_fn=None,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    """Section 10: L = lambda_knn L_knn + lambda_anchor L_anchor. Nothing else.

    ``lambda_polar`` exists only so the legacy polarizing behaviour remains
    reachable as an ablation (§3); it is 0 in bootstrap mode and no ramp is
    applied. There is deliberately no ``memory`` argument (§16).
    """
    dev = device if device is not None else graph.weight.device
    n = int(num_gaussians)
    # §7: every Gaussian starts at exactly [0.5, 0.5]. No signed endpoint-rank
    # bias -- the median endpoint rank must not decide any class sign.
    logits = torch.zeros((n, 2), device=dev, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=max(float(lr), 1.0e-8))

    targets = {
        "trusted_static": (anchors.trusted_static,
                           torch.tensor([1.0 - trusted_static_target, trusted_static_target]),
                           float(lambda_trusted_static)),
        "geometric_static": (anchors.geometric_static,
                             torch.tensor([1.0 - geometric_static_target, geometric_static_target]),
                             float(lambda_geometric_static)),
        "dynamic_core": (anchors.dynamic_core,
                         torch.tensor([dynamic_core_target, 1.0 - dynamic_core_target]),
                         float(lambda_dynamic_core)),
    }

    history: list[dict[str, float]] = []
    steps = max(0, int(iterations))
    every = max(1, int(log_every))
    undecided = anchors.undecided.to(dev)

    for step in range(steps + 1):
        q = torch.softmax(logits, dim=-1)
        knn = positive_smoothness_loss(q, graph, similarity, gamma=1.0)   # §8, plain s_ij
        parts = {name: soft_target_cross_entropy(q, mask, target)
                 for name, (mask, target, _) in targets.items()}
        anchor = sum(weight * parts[name] for name, (_, _, weight) in targets.items())
        total = float(lambda_knn) * knn + float(lambda_anchor) * anchor
        if float(lambda_polar) != 0.0:
            # Ablation path only; §3 forbids this in bootstrap mode.
            total = total + float(lambda_polar) * binary_entropy(q).mean()

        if step % every == 0 or step == steps:
            with torch.no_grad():
                peak = q.max(dim=-1).values
                row = {
                    "iteration": int(step),
                    "loss_total": float(total.detach()),
                    "loss_knn": float(knn.detach()),
                    "loss_anchor": float(anchor.detach()),
                    "loss_trusted_static": float(parts["trusted_static"].detach()),
                    "loss_geometric_static": float(parts["geometric_static"].detach()),
                    "loss_dynamic_core": float(parts["dynamic_core"].detach()),
                    "lambda_polar_current": float(lambda_polar),
                    "mean_entropy_all": float(binary_entropy(q).mean()),
                    "mean_q_dynamic_all": float(q[:, 1].mean()),
                    "fraction_peak_gt_090": float((peak > 0.90).float().mean()),
                    "mean_q_dynamic_undecided": (
                        float(q[undecided, 1].mean()) if bool(undecided.any()) else float("nan")),
                    "mean_entropy_undecided": (
                        float(binary_entropy(q)[undecided].mean())
                        if bool(undecided.any()) else float("nan")),
                }
            history.append(row)
            if log_fn is not None:
                log_fn(f"[Bootstrap] it {step:4d}  L {row['loss_total']:.5f} "
                       f"(knn {row['loss_knn']:.5f} anchor {row['loss_anchor']:.5f})  "
                       f"H_all {row['mean_entropy_all']:.4f}  "
                       f"H_undec {row['mean_entropy_undecided']:.4f}  "
                       f"q_dyn_undec {row['mean_q_dynamic_undecided']:.4f}  "
                       f">0.90 {row['fraction_peak_gt_090']*100:.1f}%")
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

    with torch.no_grad():
        final = logits.detach().clone()
        q = torch.softmax(final, dim=-1)
    return final, q, history


# ---------------------------------------------------------------------------
# Sections 11, 17, 18, 19: composition and diagnostics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compose_bootstrap_part_probs(
    q: torch.Tensor,
    xyz: torch.Tensor,
    num_moving_parts: int,
    *,
    cluster_iterations: int = 20,
    cluster_temperature: float = 0.0,
    seed: int = 20260919,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Section 11: P_i0 = q^S, P_ik = q^D pi_ik. No hard cluster label persists."""
    from .soft_labels import compose_soft_initial_state, soft_spatial_clustering

    k = max(1, int(num_moving_parts))
    if k == 1:
        return q.clone(), None, xyz.new_zeros((0, 3))
    pi, centroids = soft_spatial_clustering(
        xyz, q[:, 1], k, iterations=cluster_iterations,
        temperature=cluster_temperature, seed=seed)
    return compose_soft_initial_state(q, pi, k), pi, centroids


@torch.no_grad()
def bootstrap_diagnostics(
    q: torch.Tensor,
    part_probs: torch.Tensor,
    anchors: BootstrapAnchors,
) -> tuple[dict[str, Any], list[str]]:
    """Sections 17 and 18, including the re-binarization guard."""
    n = int(q.shape[0])
    entropy = binary_entropy(q)
    peak = part_probs.max(dim=1).values
    anchor_any = anchors.trusted_static | anchors.geometric_static | anchors.dynamic_core
    undecided = anchors.undecided

    def _mean(t, m):
        return float(t[m].mean()) if bool(m.any()) else float("nan")

    q_dyn = q[:, 1]
    diagnostics: dict[str, Any] = {
        "num_trusted_static": int(anchors.trusted_static.sum()),
        "num_geometric_static": int(anchors.geometric_static.sum()),
        "num_dynamic_core": int(anchors.dynamic_core.sum()),
        "num_undecided": int(undecided.sum()),
        "fraction_trusted_static": float(anchors.trusted_static.float().mean()),
        "fraction_geometric_static": float(anchors.geometric_static.float().mean()),
        "fraction_dynamic_core": float(anchors.dynamic_core.float().mean()),
        "fraction_undecided": float(undecided.float().mean()),
        "mean_entropy_all": float(entropy.mean()),
        "mean_entropy_anchors": _mean(entropy, anchor_any),
        "mean_entropy_undecided": _mean(entropy, undecided),
        "mean_q_dynamic_all": float(q_dyn.mean()),
        "mean_q_dynamic_dynamic_core": _mean(q_dyn, anchors.dynamic_core),
        "mean_q_dynamic_undecided": _mean(q_dyn, undecided),
        "undecided_fraction_below_040": (
            float((q_dyn[undecided] < 0.4).float().mean()) if bool(undecided.any()) else float("nan")),
        "undecided_fraction_040_to_060": (
            float(((q_dyn[undecided] >= 0.4) & (q_dyn[undecided] <= 0.6)).float().mean())
            if bool(undecided.any()) else float("nan")),
        "undecided_fraction_above_060": (
            float((q_dyn[undecided] > 0.6).float().mean()) if bool(undecided.any()) else float("nan")),
        "fraction_peak_gt_090": float((peak > 0.90).float().mean()),
        "anchor_coverage": float(anchor_any.float().mean()),
        "tau_static": anchors.tau_static,
        "tau_dynamic": anchors.tau_dynamic,
    }

    warnings: list[str] = []
    # §18: polarization must not spread far past the anchors before motion exists.
    polarized = diagnostics["fraction_peak_gt_090"]
    coverage = diagnostics["anchor_coverage"]
    if polarized > coverage + 0.25:
        message = (f"non-anchor population has become highly polarized before motion "
                   f"evidence exists: max(P)>0.9 on {polarized*100:.1f}% of Gaussians "
                   f"against {coverage*100:.1f}% anchor coverage")
        warnings.append(message)
    return diagnostics, warnings


@torch.no_grad()
def first_support_diagnostics(
    part_probs: torch.Tensor,
    anchors: BootstrapAnchors,
    gt_part_labels: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Section 19: what INNER 0 will actually receive.

    Section 12: an exact [0.5, 0.5] tie must resolve to static class 0, which is
    what ``argmax`` does for ties (first maximal index). Undecided Gaussians are
    therefore temporarily static while staying persistently soft.
    """
    z = part_probs.argmax(dim=1)
    entropy = binary_entropy(part_probs[:, :2]) if int(part_probs.shape[1]) >= 2 else None
    out: dict[str, Any] = {
        "temporary_static_count": int((z == 0).sum()),
        "temporary_moving_count": int((z > 0).sum()),
        "persistent_mean_entropy": float(entropy.mean()) if entropy is not None else float("nan"),
        "persistent_undecided_count": int(anchors.undecided.sum()),
    }
    if gt_part_labels is not None and int(torch.as_tensor(gt_part_labels).numel()):
        gt = (torch.as_tensor(gt_part_labels).reshape(-1).cpu() > 0)
        moving = (z > 0).cpu()
        if int(moving.numel()) == int(gt.numel()):
            tp = int((moving & gt).sum())
            fp = int((moving & ~gt).sum())
            fn = int((~moving & gt).sum())
            # §19: high precision FIRST; recall is the outer loop's job.
            out["moving_support_precision"] = float(tp / max(tp + fp, 1) * 100.0)
            out["moving_support_recall"] = float(tp / max(tp + fn, 1) * 100.0)
            out["moving_support_size"] = int(moving.sum())
    return out


def format_bootstrap_diagnostics(diagnostics: dict[str, Any]) -> str:
    """Section 17's required print."""
    d = diagnostics
    return "\n".join([
        "[Bootstrap] ===== high-precision bootstrap initialization (§17) =====",
        f"  trusted static      {d['num_trusted_static']:7d}  ({d['fraction_trusted_static']*100:6.2f}%)",
        f"  geometric static    {d['num_geometric_static']:7d}  ({d['fraction_geometric_static']*100:6.2f}%)",
        f"  dynamic core        {d['num_dynamic_core']:7d}  ({d['fraction_dynamic_core']*100:6.2f}%)",
        f"  undecided           {d['num_undecided']:7d}  ({d['fraction_undecided']*100:6.2f}%)",
        f"  mean entropy   all {d['mean_entropy_all']:.4f}   anchors {d['mean_entropy_anchors']:.4f}"
        f"   undecided {d['mean_entropy_undecided']:.4f}",
        f"  mean q_dynamic all {d['mean_q_dynamic_all']:.4f}   core {d['mean_q_dynamic_dynamic_core']:.4f}"
        f"   undecided {d['mean_q_dynamic_undecided']:.4f}",
        f"  undecided split    <0.4 {d['undecided_fraction_below_040']*100:5.1f}%"
        f"   0.4-0.6 {d['undecided_fraction_040_to_060']*100:5.1f}%"
        f"   >0.6 {d['undecided_fraction_above_060']*100:5.1f}%",
        f"  max(P)>0.90        {d['fraction_peak_gt_090']*100:6.2f}%"
        f"   (anchor coverage {d['anchor_coverage']*100:6.2f}%)",
    ])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_high_precision_bootstrap_initialization(
    canonical_xyz: torch.Tensor,
    final_xyz: torch.Tensor,
    num_moving_parts: int,
    *,
    trusted_static_mask: torch.Tensor | None = None,
    args=None,
    result_dir=None,
    gt_part_labels: torch.Tensor | None = None,
    log_fn=None,
) -> BootstrapInitResult:
    """Sections 1-19 end to end. Seeds only; the outer loop segments."""
    from pathlib import Path

    from .endpoint_contrastive_init import (
        compute_endpoint_matching,
        endpoint_edge_similarity,
    )
    from .neighborhood import build_neighbor_graph
    from .soft_labels import assert_no_memory_in_initialization

    # §16: initialization cannot receive temporal memory.
    assert_no_memory_in_initialization(train_soft_bootstrap_probabilities)

    def opt(name: str, default):
        return default if args is None else type(default)(getattr(args, name, default))

    printer = log_fn if log_fn is not None else print
    xyz = torch.as_tensor(canonical_xyz).detach().float()
    k = max(1, int(num_moving_parts))
    seed = int(getattr(args, "seed", 20260919)) if args is not None else 20260919

    matching = compute_endpoint_matching(
        xyz, final_xyz, chunk_size=opt("rigid_part_endpoint_chunk", 2048))
    graph = build_neighbor_graph(
        xyz, k=opt("init_graph_k", 16), chunk_size=opt("killing_neighbor_chunk", 4096))
    similarity, _ = endpoint_edge_similarity(graph, matching.normalized)

    anchors = build_high_precision_bootstrap_anchors(
        matching.normalized, trusted_static_mask,
        dynamic_quantile=opt("bootstrap_dynamic_quantile", 0.98),
        static_quantile=opt("bootstrap_static_quantile", 0.10))

    logits, q, history = train_soft_bootstrap_probabilities(
        graph, similarity, anchors, num_gaussians=int(xyz.shape[0]),
        iterations=opt("bootstrap_iterations", 200),
        lr=opt("init_binary_lr", 1.0e-2),
        lambda_knn=opt("lambda_bootstrap_knn", 1.0),
        lambda_anchor=opt("lambda_bootstrap_anchor", 1.0),
        lambda_trusted_static=opt("lambda_init_trusted_static", 2.0),
        lambda_geometric_static=opt("lambda_bootstrap_geometric_static", 0.5),
        lambda_dynamic_core=opt("lambda_bootstrap_dynamic_core", 1.0),
        lambda_polar=opt("lambda_bootstrap_polar", 0.0),
        log_fn=printer, device=xyz.device)

    part_probs, pi, centroids = compose_bootstrap_part_probs(
        q, xyz, k,
        cluster_iterations=opt("init_soft_cluster_iters", 20),
        cluster_temperature=opt("init_soft_cluster_temperature", 0.0), seed=seed)

    total = part_probs.sum(dim=1)
    if not bool(torch.allclose(total, torch.ones_like(total), atol=1.0e-5)):
        raise RuntimeError("bootstrap P^(0) rows do not sum to 1")

    diagnostics, warnings = bootstrap_diagnostics(q, part_probs, anchors)
    diagnostics["init_mode"] = "high_precision_bootstrap"
    diagnostics["num_moving_parts"] = k
    printer(format_bootstrap_diagnostics(diagnostics))
    for message in warnings:
        printer(f"[Bootstrap warning] {message}")

    support = first_support_diagnostics(part_probs, anchors, gt_part_labels)
    diagnostics["first_support"] = support
    printer(f"[Bootstrap] §19 first temporary support: static {support['temporary_static_count']}"
            f"  moving {support['temporary_moving_count']}"
            f"  persistent undecided {support['persistent_undecided_count']}")
    if "moving_support_precision" in support:
        printer(f"    GT (evaluation only): moving-support precision "
                f"{support['moving_support_precision']:.2f}%  recall "
                f"{support['moving_support_recall']:.2f}%  (high precision is the target)")

    if result_dir is not None:
        from utils.temporal_visibility_pretraining import _atomic_torch_save

        directory = Path(result_dir) / "initialization"
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save({
            "schema_version": 1,
            "endpoint_nn_distance": matching.distance.detach().cpu(),
            "endpoint_nn_distance_normalized": matching.normalized.detach().cpu(),
            "logits": logits.detach().cpu(),
            "q": q.detach().cpu(),
            "q_static": q[:, 0].detach().cpu(),
            "q_dynamic": q[:, 1].detach().cpu(),
            "entropy": binary_entropy(q).detach().cpu(),
            "soft_part_probs": part_probs.detach().cpu(),
            "trusted_static_mask": anchors.trusted_static.detach().cpu(),
            "geometric_static_mask": anchors.geometric_static.detach().cpu(),
            "dynamic_core_mask": anchors.dynamic_core.detach().cpu(),
            "undecided_mask": anchors.undecided.detach().cpu(),
            "pi": None if pi is None else pi.detach().cpu(),
            "diagnostics": diagnostics,
            "warnings": warnings,
        }, directory / "bootstrap_init.pt")
        import csv as _csv

        with (directory / "bootstrap_init_history.csv").open("w", newline="", encoding="utf-8") as s:
            keys = sorted({key for row in history for key in row}) if history else []
            if keys:
                writer = _csv.DictWriter(s, fieldnames=keys)
                writer.writeheader()
                writer.writerows(history)

    return BootstrapInitResult(
        part_probs=part_probs, q=q, logits=logits, anchors=anchors, pi=pi,
        centroids=centroids, history=history, diagnostics=diagnostics, warnings=warnings)
