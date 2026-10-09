"""Sections 10, 14, 25: the rigid-part discovery loop and the joint stage.

CODEX_IMPLEMENT_GEOMETRIC_RIGID_MOTION_JOINT_DISCOVERY.md sections 10, 14, 19, 25, 28.

Section 48 fixes the order and forbids reversing it:

    images + multiview analytic gradients  ->  WHO MOVES AND HOW MUCH
    rigid geometry of each dynamic part    ->  WHICH JOINT

so this module runs section 10's alternation

    A initialize motion evidence
    B initialize K rigid parts
    C fit T_{k,t} against the real RGB-D window
    D reassign labels
    E spatial regularization on the supernode graph
    F true nonlinear accept/reject
    G repeat to convergence

to completion, and only then (section 14) hands {T_{k,t}} to the geometric
joint fit. No P/R hypothesis exists anywhere above this line, which is the
point of section 48's "Do not let P/R assumptions decide which Gaussian moves
before rigid motion has been recovered".
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import torch

from estimator.part_assignment import assign_parts, rigid_assignment_costs
from estimator.rigid_part_estimator import (RigidTransforms, apply_transforms,
                                            initialize_transforms,
                                            refine_rigid_transforms,
                                            rigid_residual_per_gaussian)

from .geometric_joint_fit import decide_joint_type
from .joint_geometry import rms_radius, se3_distance

EPS = 1e-12


@dataclass
class RigidDiscoveryConfig:
    """Section 28's rigid-part convergence criteria."""
    max_outer: int = 6
    tau_energy: float = 1e-4          # |E^{j+1}-E^j| / max(1,E^j)
    tau_transform: float = 1e-3       # max_{k,t} d_SE(3)
    tau_jaccard: float = 0.98         # Jaccard(z^{j+1}, z^j)
    potts_weight: float = 1.0
    refine_iters: int = 12
    # Section 18's weights, each term logged separately.
    lambda_track: float = 1.0
    lambda_pair: float = 1.0
    lambda_graph: float = 1.0
    lambda_render: float = 0.0
    # Sections 7, 12, 13.
    min_visibility_overlap: int = 2
    robust_loss: str = "tukey"
    seed: int = 0
    tau_indistinguishable: float = 1e-9
    # A BUDGET, not a stopping rule: section 28's criteria stop the fit.
    # Measured on real USB transforms, the revolute fit settled at 68
    # iterations, so a budget of 60 reported UNRESOLVED_OPTIMIZATION on a
    # fit that was about to converge.
    joint_fit_iters: int = 200


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = int((a == b).sum())
    return inter / max(int(a.numel()), 1)


@torch.no_grad()
def grow_rigid_seed(seed: int, edges: torch.Tensor, affinity: torch.Tensor,
                    num_nodes: int, *, min_affinity: float,
                    max_size: int = 400) -> torch.Tensor:
    """A graph-coherent seed subset: breadth-first growth over rigid edges.

    Section 49 asks for hypotheses from "graph-coherent seed subsets". Growth
    follows only edges whose rigidity affinity clears `min_affinity`, which is
    derived from the observed affinity distribution rather than chosen, so the
    subset is a set of supernodes that are mutually consistent with ONE rigid
    body -- spatially connected by construction, because the graph is the kNN
    adjacency.
    """
    dev = affinity.device
    adj = {}
    keep = affinity >= min_affinity
    e = edges[keep]
    for a, b in e.tolist():
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen = {int(seed)}
    frontier = [int(seed)]
    while frontier and len(seen) < max_size:
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
                    if len(seen) >= max_size:
                        break
            if len(seen) >= max_size:
                break
        frontier = nxt
    m = torch.zeros(num_nodes, dtype=torch.bool, device=dev)
    m[torch.tensor(sorted(seen), dtype=torch.long, device=dev)] = True
    return m


@torch.no_grad()
def score_hypothesis(x0, tracks, comp, length, *, inlier_c: float = 2.0,
                     lam: float = 1.0) -> dict:
    """Section 50: score a rigid hypothesis by WHOLE-TRACK support.

        S(T) = sum_i support_i(T) - lambda E_track(T)

    `support_i` counts a supernode whose entire Phase-1 trajectory is
    explained by this one T_t sequence. Section 50 forbids scoring by final
    displacement amplitude, which is what made the previous magnitude rule
    both biased toward large motion and blind to points near a revolute axis.
    """
    from estimator.part_assignment import whole_track_assignment_cost
    c = whole_track_assignment_cost(x0, tracks, [comp], length).reshape(-1)
    scale = robust_scale_1d(c)
    inlier = c <= (c.median() + inlier_c * scale)
    return {"cost": c, "inliers": inlier, "support": int(inlier.sum()),
            "score": float(inlier.sum()) - lam * float(c[inlier].sum()),
            "track_energy": float(c.sum())}


def robust_scale_1d(v: torch.Tensor) -> torch.Tensor:
    from .rigidity_graph import robust_scale
    return robust_scale(v)


@torch.no_grad()
def build_hypothesis_pool(x0, tracks, edges, affinity, length, *,
                          num_nodes: int, confidence=None, pool_size: int = 32,
                          seed: int = 0, min_members: int = 3,
                          verbose: bool = True) -> list:
    """Section 7: a POOL of candidate whole-window rigid trajectories.

    Candidates come from graph-coherent seed subsets of the rigidity graph --
    high rigid-compatibility neighbourhoods -- each grown, then robustly fitted
    over the whole window. Section 7 forbids generating candidates from
    displacement magnitude, and nothing here reads a displacement norm.

    This REPLACES the previous sequential scheme, which fitted one model,
    declared its "identity inliers" by a threshold on the full mixed cost
    distribution, removed them, and looked for the next model in the remainder.
    Section 2 removes that threshold: with moving parts a large fraction of the
    scene, the identity's own cost statistics are contaminated by the very
    nodes it fails to explain, so it absorbed them and the pool ran dry before
    K models were found. Measured on the section 35 case: the identity swallowed
    59 of 60 nodes of a genuinely rotating part whose cost was 35x the static
    cost, because the mixed-population threshold sat 93x above it.
    """
    from estimator.rigid_part_estimator import (robust_fit_se3_trajectory,
                                                se3_distance_to_identity)
    dev = tracks.device
    finite = affinity[affinity > 0]
    min_aff = float(finite.median()) if finite.numel() else 0.0

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(int(num_nodes), generator=g).tolist()
    pool, seen_masks = [], []
    for sd in order:
        if len(pool) >= int(pool_size):
            break
        sub = grow_rigid_seed(sd, edges, affinity, num_nodes,
                              min_affinity=min_aff)
        n = int(sub.sum())
        if n < int(min_members):
            continue
        # Skip a seed that regrows a subset we already fitted; the pool wants
        # DISTINCT hypotheses, not the same body found from every member.
        dup = False
        for m in seen_masks:
            inter = int((sub & m).sum())
            if inter >= 0.9 * min(n, int(m.sum())):
                dup = True
                break
        if dup:
            continue
        fit = robust_fit_se3_trajectory(x0, tracks, sub, confidence)
        if fit["degenerate"]:
            continue
        d_id = se3_distance_to_identity(fit["R"], fit["t"], float(length))
        seen_masks.append(sub)
        pool.append({"R": fit["R"], "t": fit["t"], "seed_node": int(sd),
                     "seed_size": n, "distance_to_identity": float(d_id)})
    if verbose:
        nz = sum(1 for h in pool if h["distance_to_identity"] > 1e-12)
        print(f"       §7 hypothesis pool: {len(pool)} candidates "
              f"({nz} non-identity) from graph-coherent seeds")
    return pool


@torch.no_grad()
def score_model_set(costs: torch.Tensor, chosen: tuple, edges=None,
                    affinity=None, *, potts_weight: float = 1.0,
                    pair_weight: float = 1.0) -> dict:
    """Section 8's E_modelset for one candidate set {I, H_a1, ... H_aK}.

        E = sum_i min_{h in set} C_i(h)  +  E_graph  +  E_pair

    `costs` is [M+1, C] with row 0 the identity, precomputed once so a set is
    scored by an elementwise minimum rather than by refitting anything.

    Every model competes for every supernode directly (section 5). There is no
    identity inlier set and no threshold: a node belongs to whichever
    hypothesis explains its WHOLE track best.
    """
    sel = torch.stack([costs[0]] + [costs[c] for c in chosen])   # [K+1, C]
    best, labels = sel.min(dim=0)
    E = float(best.sum())
    e_graph = e_pair = 0.0
    if edges is not None and edges.numel() and affinity is not None:
        i, j = edges[:, 0].long(), edges[:, 1].long()
        diff = (labels[i] != labels[j]).double()
        w = affinity.double().clamp_min(0.0)
        e_graph = float(potts_weight) * float((w * diff).sum())
        # Section 15: a rigidly consistent edge whose endpoints disagree is
        # evidence against this partition, over and above the Potts count.
        e_pair = float(pair_weight) * float((w * diff).sum())
    return {"energy": E + e_graph + e_pair, "data": E, "graph": e_graph,
            "pair": e_pair, "labels": labels, "chosen": tuple(chosen)}


@torch.no_grad()
def select_model_set(costs: torch.Tensor, K: int, edges=None, affinity=None, *,
                     potts_weight: float = 1.0, pair_weight: float = 1.0,
                     exhaustive_limit: int = 2, verbose: bool = True) -> dict:
    """Section 8: choose the K non-identity hypotheses that, TOGETHER with the
    identity, explain the scene best.

        (a*, b*) = argmin_{a != b} E(I, H_a, H_b)          for K = 2

    Section 8 asks for direct evaluation of combinations where feasible. With
    the per-hypothesis costs precomputed, a set costs one elementwise minimum,
    so K <= `exhaustive_limit` is enumerated exactly. Larger K falls back to a
    greedy forward selection followed by one swap pass, which is the standard
    remedy when the exact search stops being affordable; the fallback is
    reported so it is never mistaken for the exact answer.
    """
    M = int(costs.shape[0]) - 1
    if M < 1 or K < 1:
        return {"chosen": tuple(), "labels": torch.zeros(costs.shape[1],
                                                         dtype=torch.long,
                                                         device=costs.device),
                "exhaustive": True, "energy": float(costs[0].sum())}
    cand = list(range(1, M + 1))
    if K <= int(exhaustive_limit) and len(cand) <= 64:
        best = None
        for combo in itertools.combinations(cand, min(K, len(cand))):
            sc = score_model_set(costs, combo, edges, affinity,
                                 potts_weight=potts_weight,
                                 pair_weight=pair_weight)
            if best is None or sc["energy"] < best["energy"]:
                best = sc
        best["exhaustive"] = True
        if verbose:
            print(f"       §8 model-set search: exhaustive over "
                  f"{len(cand)} candidates choose {K}, best E = "
                  f"{best['energy']:.6g}")
        return best
    # greedy forward selection, then one improving-swap pass
    chosen = []
    for _ in range(min(K, len(cand))):
        best_add = None
        for c in cand:
            if c in chosen:
                continue
            sc = score_model_set(costs, tuple(chosen + [c]), edges, affinity,
                                 potts_weight=potts_weight,
                                 pair_weight=pair_weight)
            if best_add is None or sc["energy"] < best_add["energy"]:
                best_add = sc
        if best_add is None:
            break
        chosen = list(best_add["chosen"])
    best = score_model_set(costs, tuple(chosen), edges, affinity,
                           potts_weight=potts_weight, pair_weight=pair_weight)
    improved = True
    while improved:
        improved = False
        for idx in range(len(chosen)):
            for c in cand:
                if c in chosen:
                    continue
                trial = list(chosen)
                trial[idx] = c
                sc = score_model_set(costs, tuple(trial), edges, affinity,
                                     potts_weight=potts_weight,
                                     pair_weight=pair_weight)
                if sc["energy"] < best["energy"] - 1e-12:
                    best, chosen, improved = sc, trial, True
    best["exhaustive"] = False
    if verbose:
        print(f"       §8 model-set search: greedy+swap over {len(cand)} "
              f"candidates choose {K}, best E = {best['energy']:.6g}")
    return best


@torch.no_grad()
def extract_nonidentity_rigid_hypotheses(x0, tracks, edges, affinity, length,
                                         n_models: int, *, num_nodes: int,
                                         confidence=None, n_seeds: int = 24,
                                         seed: int = 0, verbose: bool = True
                                         ) -> list:
    """Sections 7 and 8: the pool, scored as SETS.

    Kept under its original name because callers and tests use it, but the
    mechanism is now section 8's joint model-set selection rather than the
    sequential identity-inlier removal of before (section 2).
    """
    from estimator.part_assignment import whole_track_assignment_cost
    pool = build_hypothesis_pool(x0, tracks, edges, affinity, length,
                                 num_nodes=num_nodes, confidence=confidence,
                                 pool_size=max(int(n_seeds), 4 * int(n_models)),
                                 seed=seed, verbose=verbose)
    pool = [h for h in pool if h["distance_to_identity"] > 1e-12]
    if not pool:
        return []
    T = int(tracks.shape[0])
    eye = torch.eye(3, dtype=torch.float64, device=tracks.device)
    identity = {"R": eye.reshape(1, 3, 3).repeat(T, 1, 1),
                "t": torch.zeros((T, 3), dtype=torch.float64,
                                 device=tracks.device)}
    models = [identity] + [{"R": h["R"], "t": h["t"]} for h in pool]
    costs = whole_track_assignment_cost(x0, tracks, models, length,
                                        confidence).T.contiguous()  # [M+1, C]
    sel = select_model_set(costs, int(n_models), edges, affinity,
                           verbose=verbose)
    out = []
    for rank, c in enumerate(sel["chosen"]):
        h = pool[c - 1]
        occ = int((sel["labels"] == rank + 1).sum())
        out.append({"component": {"R": h["R"], "t": h["t"]},
                    "seed_node": h["seed_node"], "seed_size": h["seed_size"],
                    "distance_to_identity": h["distance_to_identity"],
                    "occupancy": occ,
                    "support": occ,
                    "score": -float(sel["energy"])})
        if verbose:
            print(f"       hypothesis {rank + 1}: seed node {h['seed_node']} "
                  f"(grew to {h['seed_size']}), occupancy {occ}, "
                  f"d(T, I) = {h['distance_to_identity']:.4g}")
    return out


@torch.no_grad()
def initialize_rigid_components(x0, tracks, edges, affinity, length, K, *,
                                num_nodes: int, confidence=None,
                                seed: int = 0, verbose: bool = True) -> dict:
    """Sections 6, 9 and 10: K+1 rigid components, static NOT hard-coded.

    Section 6: the initializer must produce K DISTINCT non-identity
    hypotheses, and must not reach them by removing identity inliers and
    hoping the remainder contains K models.

    Returns the K+1 component trajectories and an initial labelling. Which one
    is static is decided afterwards, by section 11, from the trajectories
    themselves.
    """
    from estimator.part_assignment import whole_track_assignment_cost
    dev = tracks.device
    T = int(tracks.shape[0])
    eye = torch.eye(3, dtype=torch.float64, device=dev)
    identity = {"R": eye.reshape(1, 3, 3).repeat(T, 1, 1),
                "t": torch.zeros((T, 3), dtype=torch.float64, device=dev)}

    models = extract_nonidentity_rigid_hypotheses(
        x0, tracks, edges, affinity, length, K, num_nodes=num_nodes,
        confidence=confidence, seed=seed, verbose=verbose)

    comps = [identity] + [m["component"] for m in models]
    # Section 17 E / section 10: a component that could not be extracted is not
    # fabricated; it is carried as a duplicate identity and will either pick up
    # members through the alternation or be reported empty.
    while len(comps) < K + 1:
        comps.append({"R": identity["R"].clone(), "t": identity["t"].clone()})

    # Section 4: the label is the argmin over ALL models. No identity
    # threshold, no pre-accepted identity membership (section 5).
    cost = whole_track_assignment_cost(x0, tracks, comps, length, confidence)
    labels = cost.argmin(dim=1)
    return {"components": comps, "labels": labels, "cost": cost,
            "hypotheses": [{k: v for k, v in m.items() if k != "component"}
                           for m in models]}


@torch.no_grad()
def identify_static_component(comps: list, length: float) -> dict:
    """Section 11: the static component is the one closest to the identity.

        k_static = argmin_k  (1/T) sum_t d_SE(3)(T_{k,t}, I)^2

    Decided from the recovered trajectories, never from GT and never by
    assuming index 0 was static to begin with (section 9).
    """
    from estimator.rigid_part_estimator import se3_distance_to_identity
    d = [se3_distance_to_identity(c["R"], c["t"], length) for c in comps]
    k_static = int(torch.tensor(d).argmin())
    order = [k_static] + [k for k in range(len(comps)) if k != k_static]
    remap = {old: new for new, old in enumerate(order)}
    return {"k_static": k_static, "distances": d, "remap": remap,
            "components": [comps[o] for o in order]}


@torch.no_grad()
def motion_retention(x0, tracks, comps, labels, K) -> list:
    """Section 25's MANDATORY motion-retention diagnostic.

        M_k^boot  = median_{i in k} || x_i^{boot,T} - x_i^0 ||
        M_k^rigid = median_{i in k} || T_{k,T} x_i^0 - x_i^0 ||
        eta_k     = M_k^rigid / (M_k^boot + eps)

    A ratio near zero is the storage failure: Phase 2 discarded the motion
    Phase 1 found, and the near-zero P/R residuals that follow are an artifact,
    not genuine ambiguity. Section 25 is explicit that this is a DIAGNOSTIC and
    never a joint selector.
    """
    out = []
    last = tracks.shape[0] - 1
    x0d = x0.double()
    for k in range(1, K + 1):
        m = labels == k
        if int(m.sum()) == 0:
            out.append({"part": k, "members": 0,
                        "motion_retention_bootstrap": 0.0,
                        "motion_retention_rigid": 0.0,
                        "motion_retention_ratio": 0.0, "collapsed": True})
            continue
        boot = (tracks[last][m].double() - x0d[m]).norm(dim=1).median()
        R, t = comps[k]["R"][last].double(), comps[k]["t"][last].double()
        rigid = ((x0d[m] @ R.T + t) - x0d[m]).norm(dim=1).median()
        ratio = float(rigid / (boot + EPS))
        out.append({"part": k, "members": int(m.sum()),
                    "motion_retention_bootstrap": float(boot),
                    "motion_retention_rigid": float(rigid),
                    "motion_retention_ratio": ratio,
                    "collapsed": ratio < 0.1})
    return out


@torch.no_grad()
def bootstrap_to_rigid_error(x0, tracks, comps, labels, length) -> dict:
    """Section 26: does the rigid decomposition explain the Phase-1 field?

        E_{B->R} = sum_{i,t} rho( || x_i^{boot,t} - T_{z_i,t} x_i^0 || )

    normalised by the valid count and L^2 so it is comparable across scenes.
    """
    T = int(tracks.shape[0])
    x0d = x0.double()
    per_frame, total, n = [], 0.0, 0
    L2 = max(float(length) ** 2, EPS)
    for f in range(T):
        pred = x0d.clone()
        for k, c in enumerate(comps):
            m = labels == k
            if not bool(m.any()):
                continue
            pred[m] = x0d[m] @ c["R"][f].double().T + c["t"][f].double()
        e = ((tracks[f].double() - pred) ** 2).sum(dim=1)
        per_frame.append(float(e.sum() / max(e.numel(), 1) / L2))
        total += float(e.sum())
        n += int(e.numel())
    return {"total": total, "normalized": total / max(n, 1) / L2,
            "per_frame": per_frame}


@torch.no_grad()
def discover_rigid_parts(objective, x0, R0, partition, node_labels0,
                         positions: list, K: int,
                         cfg: RigidDiscoveryConfig,
                         edges: torch.Tensor | None = None,
                         edge_weights: torch.Tensor | None = None,
                         confidence: torch.Tensor | None = None,
                         verbose: bool = True) -> dict:
    """Section 17's alternating rigid-component decomposition.

        A  initialize K+1 rigid components from the whole-window rigidity graph
        B  robustly fit T_{k,t} for every component and frame
        C  evaluate whole-track costs
        D  update labels (track + pairwise rigidity + graph)
        E  handle degenerate components without fabricating members
        F  refit trajectories
        G  evaluate the true objective
        H  iterate to section 28's convergence

    Section 1: this discovers rigid bodies by CONSISTENCY OVER THE WHOLE
    SEQUENCE. There is no displacement-magnitude membership rule anywhere in
    it, and no K == 1 shortcut.
    """
    from estimator.part_assignment import (assignment_margin,
                                           update_component_labels,
                                           whole_track_assignment_cost)
    from estimator.rigid_part_estimator import robust_fit_se3_trajectory
    from .rigidity_graph import build_rigidity_graph

    sid = partition.supernode_id.long().clamp_min(0)
    dev = x0.device
    C = partition.num_supernodes
    T = len(positions)

    # ---- whole Phase-1 supernode trajectories (section 4) -----------------
    cnt = torch.zeros(C, device=dev).index_add_(
        0, sid, torch.ones(sid.shape[0], device=dev))
    tracks = torch.stack([
        torch.zeros((C, 3), device=dev).index_add_(0, sid, p)
        / cnt.clamp_min(1).unsqueeze(1) for p in positions])          # [T,C,3]
    sx0 = tracks[0]
    cen = sx0 - sx0.mean(0, keepdim=True)
    L = float(cen.norm(dim=1).mean().clamp_min(EPS))

    if edges is None:
        edges = partition.adjacency
    edges = edges.to(dev)

    # ---- A: the rigidity graph and K+1 components (sections 5-10) ---------
    graph = build_rigidity_graph(tracks, edges, None,
                                 min_overlap=cfg.min_visibility_overlap)
    if verbose:
        fin = graph["rho"][~torch.isnan(graph["rho"])]
        print(f"  §5 rigidity graph: {int(edges.shape[0])} edges, "
              f"rho median {float(fin.median()) if fin.numel() else float('nan'):.4f}, "
              f"sigma_rigid {graph['sigma_rigid']:.4g}, "
              f"{graph['num_low_confidence']} low-confidence")
    init = initialize_rigid_components(
        sx0, tracks, edges, graph["affinity"], L, K, num_nodes=C,
        confidence=confidence, seed=cfg.seed, verbose=verbose)
    comps = init["components"]
    node_labels = init["labels"]

    history, E_prev = [], None
    stop_reason, converged = "max_outer_reached", False
    prev_comps = None

    for j in range(int(cfg.max_outer)):
        prev_nodes = node_labels.clone()

        # ---- B/F: robustly refit every component ------------------------
        new_comps, degenerate = [], []
        for k in range(K + 1):
            m = node_labels == k
            if int(m.sum()) < 3:
                degenerate.append(k)
                new_comps.append(comps[k])          # section 17 E: keep, do
                continue                            # not fabricate members
            fit = robust_fit_se3_trajectory(sx0, tracks, m, confidence,
                                            loss=cfg.robust_loss)
            new_comps.append({"R": fit["R"], "t": fit["t"]})
        comps = new_comps

        # ---- C/D: whole-track cost, then labels -------------------------
        cost = whole_track_assignment_cost(sx0, tracks, comps, L, confidence)
        upd = update_component_labels(cost, edges, graph["affinity"],
                                      node_labels, lam_pair=cfg.lambda_pair,
                                      lam_graph=cfg.potts_weight)
        node_labels = upd["labels"]
        margin = upd["margin"]

        # ---- G: the true objective (section 18) -------------------------
        labels_g = node_labels[sid]
        E_track = float(cost.gather(1, node_labels.view(-1, 1)).sum())
        E_graph = float(upd["history"][-1]["energy"]) if upd["history"] else 0.0
        E_render = _window_energy(objective, x0, R0, labels_g,
                                  _as_transforms(comps, K, dev))
        E = cfg.lambda_track * E_track + cfg.lambda_graph * E_graph \
            + cfg.lambda_render * E_render

        d_T = 0.0
        if prev_comps is not None:
            from .joint_geometry import se3_distance
            d_T = max((se3_distance(prev_comps[k]["R"][f], prev_comps[k]["t"][f],
                                    comps[k]["R"][f], comps[k]["t"][f], L)
                       for k in range(K + 1) for f in range(T)), default=0.0)
        prev_comps = [{"R": c["R"].clone(), "t": c["t"].clone()} for c in comps]

        jac = _jaccard(prev_nodes, node_labels)
        rel = (abs(E_prev - E) / max(1.0, abs(E_prev))) if E_prev is not None else 1.0
        counts = [int((labels_g == k).sum()) for k in range(K + 1)]
        history.append({"outer": j, "energy": E, "E_track": E_track,
                        "E_graph": E_graph, "E_render": E_render,
                        "relative_change": rel, "max_se3_step": d_T,
                        "label_jaccard": jac, "counts": counts,
                        "node_counts": [int((node_labels == k).sum())
                                        for k in range(K + 1)],
                        "median_margin": margin["median_margin"],
                        "frac_near_tie": margin["frac_below_1pct_of_best"],
                        "degenerate_components": degenerate})
        if verbose:
            print(f"  §17 outer {j}: E {E:.1f} (track {E_track:.3f}, "
                  f"render {E_render:.0f}), rel {rel:.2e}, dSE3 {d_T:.2e}, "
                  f"Jaccard {jac:.4f}, labels {counts}, "
                  f"margin {margin['median_margin']:.4g}")
        E_prev = E

        # ---- H: section 28's conditions, all of them --------------------
        if (rel < cfg.tau_energy and d_T < cfg.tau_transform
                and jac > cfg.tau_jaccard):
            converged, stop_reason = True, (
                f"section 28: rel {rel:.2e} < {cfg.tau_energy:g}, d_SE3 "
                f"{d_T:.2e} < {cfg.tau_transform:g}, Jaccard {jac:.4f} > "
                f"{cfg.tau_jaccard:g}")
            break

    # ---- section 11: which component is static ---------------------------
    st = identify_static_component(comps, L)
    comps = st["components"]
    remap = torch.tensor([st["remap"][int(k)] for k in range(K + 1)],
                         device=dev)
    node_labels = remap[node_labels]
    labels = node_labels[sid]
    if verbose:
        print(f"  §11 static component = index {st['k_static']} "
              f"(d to identity {[round(d, 5) for d in st['distances']]}) "
              f"-> relabelled to 0")

    transforms = _as_transforms(comps, K, dev)
    retention = motion_retention(sx0, tracks, comps, node_labels, K)
    b2r = bootstrap_to_rigid_error(sx0, tracks, comps, node_labels, L)
    if verbose:
        for r in retention:
            flag = "  PHASE2_MOTION_COLLAPSE" if r["collapsed"] else ""
            print(f"  §25 part {r['part']} motion retention: "
                  f"bootstrap {r['motion_retention_bootstrap']:.5f} -> rigid "
                  f"{r['motion_retention_rigid']:.5f}  ratio "
                  f"{r['motion_retention_ratio']:.3f}{flag}")
        print(f"  §26 bootstrap->rigid projection error (normalised): "
              f"{b2r['normalized']:.5e}")

    from estimator.rigid_part_estimator import rigid_residual_per_gaussian
    resid = rigid_residual_per_gaussian(x0, labels, positions, transforms)
    final_cost = whole_track_assignment_cost(sx0, tracks, comps, L, confidence)
    return {"transforms": transforms, "labels": labels,
            "node_labels": node_labels, "energy": E_prev, "history": history,
            "converged": converged, "stop_reason": stop_reason,
            "rigid_residual": resid,
            "counts": [int((labels == k).sum()) for k in range(K + 1)],
            "rigidity_graph": graph, "hypotheses": init["hypotheses"],
            "static_identification": {k: v for k, v in st.items()
                                      if k != "components"},
            "motion_retention": retention,
            "bootstrap_to_rigid": b2r,
            "assignment_margin": assignment_margin(final_cost),
            "track_cost": final_cost, "length_scale": L,
            "supernode_tracks": tracks}


def _as_transforms(comps: list, K: int, dev) -> RigidTransforms:
    """Components 1..K as the RigidTransforms the rest of the pipeline uses.

    Component 0 is the static one and is the identity by construction of
    section 11's remap, so it is not stored (section 8's T_{0,t} = I).
    """
    T = int(comps[0]["R"].shape[0])
    tr = RigidTransforms.identity(T, K, device=dev)
    for k in range(1, K + 1):
        tr.R[:, k - 1] = comps[k]["R"]
        tr.t[:, k - 1] = comps[k]["t"]
    return tr


@torch.no_grad()
def _window_energy(objective, x0, R0, labels, transforms, verbose=False,
                   desc="  window energy") -> float:
    e = 0.0
    frames = list(enumerate(objective.frames))
    if verbose:
        from tools.kpart_pipeline import _progress
        frames = _progress(frames, desc, True)
    for ti, (obs, _f) in frames:
        mu, R = apply_transforms(x0, R0, labels, transforms, ti)
        er, ed = objective.frame_energy(mu, R, obs)
        e += er + ed
    return float(e)


@torch.no_grad()
def fit_joints(x0, labels, transforms: RigidTransforms, K: int,
               cfg: RigidDiscoveryConfig, rigid_residual=None,
               verbose: bool = True) -> list:
    """Section 14: infer each part's joint, AFTER the rigid trajectories exist.

    Nothing here sees an image. The only inputs are the recovered {T_{k,t}},
    the part's own canonical extent (section 20's L_k) and -- for section 19's
    "numerically indistinguishable" -- how precisely those transforms were
    themselves recovered.

    `rigid_residual` is the per-Gaussian rigid RMS error. It sets the floor
    below which E_P and E_R are two numbers inside their own error bars.

    That floor used to be `(sigma / (L sqrt(N)))^2`, which assumes the N
    members give N INDEPENDENT looks at the part's rotation. Section 8 of
    CODEX_DIRECT_GAUSSIAN_JOINT_RGBD_RELIABILITY.md rules that out: every
    member of a part is fitted through the same images, the same calibration
    and the same canonical Gaussian model, so a systematic error in any of
    them moves all the members together. Dividing by sqrt(N) with N in the
    thousands then claims a resolution the reconstruction never had, and any
    tiny gain clears it. A synthetic case with correlated per-frame rotation
    noise makes a spurious revolute gain read as 179x that floor
    (tests/test_geometric_floor.py).

    What is used here instead is the OPPOSITE extreme, `(sigma / L)^2`: the
    rigid-model error is treated as fully correlated across members, so the
    effective sample size is one. That is deliberately conservative, and it is
    the only honest choice available inside a single run -- the real floor is
    the spread of the gain across independent camera-block refits, which needs
    several runs. Produce them with `--camera-folds` / `--camera-fold` and
    decide with `tools/joint_decide_from_folds.py`; every verdict here carries
    `floor_source` saying which of the two it was.

    sigma is taken as the MEDIAN per-Gaussian residual, not the maximum. The
    maximum is an outlier statistic: measured on real USB it was 0.61 against a
    part radius of 0.57, which would declare every comparison indistinguishable
    on the strength of a handful of badly tracked Gaussians.
    """
    from .gaussian_transport_metric import canonical_moments, gain_attribution
    from .geometric_joint_fit import decide_joint_type_continuous
    out = []
    for k in range(1, K + 1):
        mask = labels == k
        if int(mask.sum()) < 3:
            out.append(None)
            continue
        L = rms_radius(x0[mask].double())
        R_k, t_k = transforms.part_trajectory(k)
        n_k = max(int(mask.sum()), 1)
        sigma = (float(rigid_residual[mask].median())
                 if rigid_residual is not None else 0.0)
        floor = (sigma / max(L, EPS)) ** 2
        moments = canonical_moments(x0[mask].double())
        d = decide_joint_type_continuous(
            R_k, t_k, moments["c"], moments["L"], epsilon_geom=floor,
            max_iters=cfg.joint_fit_iters, moments=moments)
        # Section 9.2: record whether any revolute preference came from the
        # centre path bending or only from the part's orientation wobbling.
        d["floor_source"] = (
            "within-run conservative bound (sigma/L)^2 with the rigid-model "
            "error treated as fully correlated across members (section 8). "
            "The authoritative floor is the spread of the gain across "
            "independent camera-block refits: see "
            "tools/joint_decide_from_folds.py")
        d["normalized_error"] = {
            "prismatic": d["E_P"], "revolute": d["E_R"],
            "absolute_gap": d["curvature_gain"],
            "relative_gap": d["curvature_gain"],
            "noise_floor": floor,
            "indistinguishable_floor": d["floor"],
            "tau_indistinguishable": cfg.tau_indistinguishable}
        d["part"] = k
        d["length_scale"] = L
        d["rigid_rmse_median"] = sigma
        d["normalized_rigid_rmse"] = sigma / max(L, EPS)
        d["transform_precision"] = sigma / max(L, EPS)
        d["transform_precision_optimistic_1_over_sqrtN"] = (
            sigma / (max(L, EPS) * math.sqrt(n_k)))
        d["num_gaussians"] = int(mask.sum())
        # Section 9.2, recorded with every part: a revolute preference built
        # only out of orientation wobble, with the centre path fitted at
        # least as well by a straight line, is not measured curvature.
        from .continuous_joint_geometry import fit_continuous_joint
        from .geometric_joint_fit import relative_trajectory
        _Rr, _tr = relative_trajectory(R_k, t_k)
        _fp = fit_continuous_joint(moments["c"], _Rr, _tr, moments["L"],
                                   max_iters=cfg.joint_fit_iters,
                                   fix_eta_zero=True, moments=moments)
        _fr = fit_continuous_joint(moments["c"], _Rr, _tr, moments["L"],
                                   max_iters=cfg.joint_fit_iters,
                                   fix_eta_zero=False, moments=moments)
        d["gain_attribution"] = gain_attribution(_Rr, _tr, _fp, _fr, moments)
        out.append(d)
        if verbose:
            ep = d["normalized_error"]["prismatic"]
            er = d["normalized_error"]["revolute"]
            ne = d["normalized_error"]
            print(f"  §19 part {k}: E_P = {ep:.6e}, E_R = {er:.6e}")
            print(f"       gap {ne['absolute_gap']:.3e} vs indistinguishable "
                  f"floor {ne['indistinguishable_floor']:.3e}")
            print(f"       (median rigid residual {sigma:.3e} over {n_k} "
                  f"Gaussians, L = {L:.4f} -> transform precision "
                  f"{d['transform_precision']:.3e}) "
                  f"-> {d['selected'] or d['status']}")
            if d["selected"] == "revolute":
                a = d["axis_point"]
                print(f"       axis {[round(x, 4) for x in d['axis']]} "
                      f"through {[round(x, 4) for x in a]}")
            elif d["selected"] == "prismatic":
                print(f"       axis {[round(x, 4) for x in d['axis']]}")
            if d["q"] is not None:
                print(f"       q(t) {[round(x, 5) for x in d['q'][:6]]}"
                      f"{' ...' if len(d['q']) > 6 else ''}")
    return out
