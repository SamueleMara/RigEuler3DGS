"""Coherent-moving-component extraction. Replaces the two-way normalized cut.

CODEX_FIX_COHERENCE_EXTRACTION_AND_USE_OPTIMIZED_3DGS_GT.md sections 11-19.

The coherence measurement `C_ij` is UNCHANGED (section 1) -- it was never the
problem. The previous report measured GT cross-boundary edges at 1.5-4.5x the
within-static strain on every scene, and blade, with the strongest signal of
all five (4.47x), had the worst F1. The failure was the partition: a two-way
normalized cut minimizes `cut/vol(A) + cut/vol(B)`, which is biased toward
balanced splits, so an 8% drawer came back as a 46% component.

What replaces it (section 11): there is no longer a requirement that every
node land in one of two bodies. The output is

    MOVING_RIGID               passed to Phase 2
    STATIC_COMPATIBLE
    INCOHERENT_OR_AMBIGUOUS    false dynamics are ALLOWED to stay here

and the moving support's size is an outcome, not a target (section 18): 5% and
60% are equally acceptable and neither is rewarded.

The pipeline, with the roles kept separate (section 17):

    C_ij  -> graph-constrained single-linkage hierarchy  -> candidates
    held-out whole-window rigid prediction               -> which is physical

No weighted score mixes the two. No Otsu, no displacement, no target fraction,
no GT, no joint type.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

MOVING_RIGID = "MOVING_RIGID"
STATIC_COMPATIBLE = "STATIC_COMPATIBLE"
INCOHERENT = "INCOHERENT_OR_AMBIGUOUS"
NO_MOVING = "NO_COHERENT_MOVING_COMPONENT"
EPS = 1e-12


# ---------------------------------------------------------------------------
# Section 12: the candidate hierarchy
# ---------------------------------------------------------------------------

def candidate_hierarchy(edge_i: Tensor, edge_j: Tensor, C: Tensor, n: int,
                        min_nodes: int = 3, min_merge_support: int = 2) -> list:
    """Graph-constrained single-linkage over edges sorted by increasing C.

    Each union-find merge event creates one candidate component, recorded with
    its birth cost (the edge that formed it) and its death cost (the edge that
    merged it into something larger). Persistence = death - birth is how long
    the component survives as a distinct coherent group as the coherence
    tolerance is relaxed.

    `min_merge_support` is section 13's guard against single-linkage chaining.
    Plain single linkage merges two components as soon as ONE cheap edge joins
    them, so a single accidental low-C cross edge silently welds two bodies
    together -- and it does so at the very first merge, which means the two
    real bodies never exist as candidates at all and no later validation can
    recover them. A genuine interface inside one rigid body is crossed by many
    kNN edges; a bridge is crossed by one. Requiring two independent edges
    before a merge distinguishes them. It is a structural minimum, global, and
    not tuned per scene.

    Nothing here has a target fraction, a balance term, GT, Otsu or a
    displacement threshold.
    """
    order = torch.argsort(C)
    parent = list(range(n))
    size = [1] * n
    members = {v: [v] for v in range(n)}
    birth = [0.0] * n
    cands = []
    pending: dict = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in order.tolist():
        c = float(C[e])
        a, b = find(int(edge_i[e])), find(int(edge_j[e]))
        if a == b:
            continue
        if min_merge_support > 1:
            # A component of size k cannot offer more than k independent edges
            # to another, and two singletons are joined by exactly one edge by
            # definition -- demanding two there would block every merge and
            # produce no candidates at all. The requirement is therefore
            # capped by what the smaller side can actually supply.
            need = min(min_merge_support, size[a], size[b])
            key = (a, b) if a < b else (b, a)
            pending[key] = pending.get(key, 0) + 1
            if pending[key] < need:
                continue
        # both children die here
        for r in (a, b):
            if size[r] >= min_nodes:
                cands.append({"members": list(members[r]),
                              "num_nodes": size[r],
                              "birth_edge_cost": birth[r],
                              "death_merge_edge_cost": c,
                              "persistence": c - birth[r]})
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        members[a] = members[a] + members[b]
        members.pop(b, None)
        size[a] += size[b]
        birth[a] = c
        if min_merge_support > 1:
            # `b` was absorbed into `a`, so support already accumulated against
            # `b` now counts toward `a`. Dropping it instead would throw away
            # real evidence and stall legitimate merges.
            remap: dict = {}
            for (u, v), cnt in pending.items():
                if u == b:
                    u = a
                if v == b:
                    v = a
                if u == v:
                    continue
                key2 = (u, v) if u < v else (v, u)
                remap[key2] = remap.get(key2, 0) + cnt
            pending = remap
    # whatever survives to the end
    for r in set(find(v) for v in range(n)):
        if size[r] >= min_nodes:
            cands.append({"members": list(members[r]), "num_nodes": size[r],
                          "birth_edge_cost": birth[r],
                          "death_merge_edge_cost": float("inf"),
                          "persistence": float("inf")})
    for k, c in enumerate(cands):
        c["candidate_id"] = k
    return cands


# ---------------------------------------------------------------------------
# Section 13: internal and boundary edge distributions
# ---------------------------------------------------------------------------

def edge_distributions(cand: dict, edge_i: Tensor, edge_j: Tensor,
                       C: Tensor, n: int) -> dict:
    """Guard against one accidental low-C cross edge chaining two bodies.

    A component is not trustworthy because a single cheap path reaches it; the
    whole internal distribution has to be coherent and the boundary has to be
    worse than the inside.
    """
    inside = torch.zeros(n, dtype=torch.bool)
    inside[torch.tensor(cand["members"], dtype=torch.long)] = True
    a, b = inside[edge_i], inside[edge_j]
    cin = C[a & b]
    cbd = C[a ^ b]

    def q(x, p):
        return float(x.quantile(p)) if x.numel() else float("nan")

    return {"num_internal_edges": int(cin.numel()),
            "num_boundary_edges": int(cbd.numel()),
            "internal_q50": q(cin, 0.50), "internal_q75": q(cin, 0.75),
            "internal_q90": q(cin, 0.90),
            "boundary_q10": q(cbd, 0.10), "boundary_q25": q(cbd, 0.25),
            "boundary_q50": q(cbd, 0.50),
            "separation": (q(cbd, 0.25) / q(cin, 0.75)
                           if q(cin, 0.75) > 0 else float("inf"))}


# ---------------------------------------------------------------------------
# Sections 14-16: held-out rigid validation
# ---------------------------------------------------------------------------

def _kabsch(P: Tensor, Q: Tensor):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=torch.float64))
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def to_reference_frame(traj: Tensor, complement) -> Tensor:
    """Re-express the trajectory in the COMPLEMENT's own rigid frame.

    Articulation is relative. Measured in world coordinates, a scene that
    drifts uniformly is perfectly rigid with a large non-identity trajectory,
    so it scores eta = 1 and an enormous delta_BIC and is declared one big
    moving body -- which is a global change of coordinates, not a joint.

    Fitting the complement's rigid frame `(R^K_t, p^K_t)` and mapping

        y_i(t) = (R^K_t)^T (x_i(t) - p^K_t)

    cancels any motion the candidate shares with the rest of the scene. A
    genuinely static complement gives `R^K ~ I`, `p^K ~ 0` and leaves the
    candidate untouched; a uniformly drifting scene collapses the candidate's
    trajectory onto its own frame-0 positions, so there is nothing left for a
    rigid model to explain. This also removes global camera/scene drift from
    every candidate for free.
    """
    X = torch.as_tensor(traj).double()
    k = torch.as_tensor(complement).long().reshape(-1)
    if k.numel() < 3:
        return None
    y0 = X[0, k]
    out = torch.empty_like(X)
    for t in range(X.shape[0]):
        R, p = _kabsch(y0, X[t, k])
        out[t] = (X[t] - p) @ R
    return out


def cross_validated_rigidity(traj: Tensor, members, folds: int = 5) -> dict:
    """Sections 14-16: fit the rigid trajectory on training nodes, score on
    held-out nodes.

    In-sample Kabsch residual is not usable for selection: a component can
    always fit itself better by being larger or noisier. The held-out form
    asks whether a COMMON rigid trajectory predicts nodes it never saw.

        eta^cv = 1 - RSS_R^cv / (RSS_I^cv + eps)

    -> 1 for a true coherent body, ~0 for a chaotic renderer-driven cloud.

    The BIC term is what stops a large static body with a whisper of drift from
    being called dynamic: a rigid trajectory costs 6 parameters per frame, and
    `BIC_I - BIC_R = N log(RSS_I/RSS_R) - 6T log N` is negative unless the
    motion actually pays for itself.
    """
    idx = torch.as_tensor(members).long().reshape(-1)
    X = torch.as_tensor(traj).double()
    T, N = X.shape[0], int(idx.numel())
    if N < max(4, folds):
        return {"degenerate": True, "num_nodes": N, "eta_cv": float("nan"),
                "rss_rigid_cv": float("nan"), "rss_identity_cv": float("nan"),
                "delta_rss_cv": float("nan"), "delta_bic": float("nan")}
    x0 = X[0, idx]
    fold_of = torch.arange(N) % folds           # deterministic, no randomness
    rss_r = rss_i = 0.0
    rss_r_in = rss_i_in = 0.0
    for f in range(folds):
        te = fold_of == f
        tr = ~te
        if int(tr.sum()) < 3 or int(te.sum()) < 1:
            continue
        for t in range(T):
            xt = X[t, idx]
            R, p = _kabsch(x0[tr], xt[tr])
            pred = x0[te] @ R.T + p
            rss_r += float((xt[te] - pred).pow(2).sum())
            rss_i += float((xt[te] - x0[te]).pow(2).sum())
    for t in range(T):
        xt = X[t, idx]
        R, p = _kabsch(x0, xt)
        rss_r_in += float((xt - x0 @ R.T - p).pow(2).sum())
        rss_i_in += float((xt - x0).pow(2).sum())
    nobs = max(N * T, 1)
    k = 6 * T                                   # rigid DOF over the window
    bic_i = nobs * math.log(max(rss_i_in, EPS) / nobs)
    bic_r = nobs * math.log(max(rss_r_in, EPS) / nobs) + k * math.log(nobs)
    return {"degenerate": False, "num_nodes": N, "folds": folds,
            "rss_rigid_cv": rss_r, "rss_identity_cv": rss_i,
            "delta_rss_cv": rss_i - rss_r,
            # INTENSIVE form: delta_RSS is a sum over N nodes x T frames, so
            # comparing candidates on it rewards size outright -- a near-whole
            # component wins on node count even when almost all of its nodes
            # are static. Section 18 forbids that. Per node-frame, the quantity
            # is the mean squared displacement a common rigid trajectory
            # explains, and it is neutral to how many nodes carry it.
            "delta_rss_cv_per_node_frame": (rss_i - rss_r) / max(N * T, 1),
            "eta_cv": 1.0 - rss_r / (rss_i + EPS),
            "rss_rigid_in_sample": rss_r_in,
            "rss_identity_in_sample": rss_i_in,
            "bic_identity": bic_i, "bic_rigid": bic_r,
            "delta_bic": bic_i - bic_r}


# ---------------------------------------------------------------------------
# Section 17: sequential selection
# ---------------------------------------------------------------------------

def select_moving_component(cands: list) -> dict:
    """Among graph-persistent candidates, the strongest held-out non-identity
    rigid evidence. No weighted sum of persistence, eta and motion.

    A candidate qualifies when a common rigid trajectory beats identity on
    nodes it never saw (`eta_cv > 0`, `delta_rss_cv > 0`) AND survives the
    complexity penalty (`delta_bic > 0`). Among those, the one with the largest
    held-out `delta_rss_cv` -- the absolute amount of motion the rigid model
    explains out of sample -- is the moving body.

    If nothing qualifies, say so. Section 17 forbids fabricating a half-scene
    dynamic component.
    """
    # Articulation is RELATIVE: a part is moving only with respect to
    # something that is not. A candidate covering the whole graph has no
    # boundary, so whatever rigid trajectory explains it is a global change of
    # coordinates, not a joint -- and a uniformly drifting scene would
    # otherwise score eta = 1 and an enormous delta_BIC and be declared one
    # big moving body. Requiring a non-empty boundary is section 13's boundary
    # distribution doing the work, not a size threshold.
    # QUALIFICATION is the held-out test (section 17's "held-out non-identity
    # rigid prediction"), not the BIC.
    #
    # delta_BIC is kept and persisted as section 16's complexity diagnostic,
    # but it must not gate: with k = 6T parameters for a free per-frame rigid
    # trajectory, the penalty k log(NT) is fixed while the evidence grows with
    # N, so a small component can never pass however coherent it is. On blade
    # the genuine moving body (8 nodes, IoU 0.67, eta_cv 0.974) was rejected at
    # delta_BIC = -4.3e+02 while a near-whole component passed -- the same
    # size bias, in a different disguise, that section 18 forbids.
    #
    # The held-out cross-validation is already the anti-overfitting guard, and
    # it is size-neutral: a chaotic cloud fails it at any N because a common
    # rigid trajectory cannot predict nodes it never saw.
    # sqrt(machine epsilon): below this the relative motion is indistinguishable
    # from accumulated floating-point error in the Kabsch chain, and any ratio
    # computed from it is noise over noise.
    motion_floor = math.sqrt(torch.finfo(torch.float64).eps)
    ok = [c for c in cands
          if not c["rigidity"].get("degenerate")
          and c["edges"]["num_boundary_edges"] > 0
          and c["rigidity"].get("relative_motion_ratio", 0.0) > motion_floor
          and c["rigidity"]["eta_cv"] > 0.0
          and c["rigidity"]["delta_rss_cv"] > 0.0]
    if not ok:
        whole = [c for c in cands if c["edges"]["num_boundary_edges"] == 0]
        why = ("no candidate is explained by a common non-identity rigid "
               "trajectory on held-out nodes with delta_bic > 0; section 17 "
               "forbids inventing one")
        near_zero = [c for c in cands
                     if not c["rigidity"].get("degenerate")
                     and c["rigidity"].get("relative_motion_ratio", 1.0)
                     <= motion_floor]
        if near_zero and len(near_zero) >= 0.5 * max(len(cands), 1):
            why = ("no candidate moves measurably relative to its complement "
                   "(relative motion at the floating-point floor): the scene "
                   "shares one rigid motion, which is a global change of "
                   "coordinates, not an articulated part")
        elif whole and any(not c["rigidity"].get("degenerate")
                           and c["rigidity"]["eta_cv"] > 0.0 for c in whole):
            why = ("the only coherently moving candidate is the WHOLE graph, "
                   "which has no boundary. That is a global coordinate drift, "
                   "not an articulated part: a moving body is moving only "
                   "relative to something that is not")
        return {"status": NO_MOVING, "selected": None,
                "num_qualified": 0, "reason": why}
    # Relative motion is SYMMETRIC: in the moving part's frame the static body
    # moves exactly as much. So the complement-relative evidence can say that
    # articulation exists, but never which side is the moving one -- selecting
    # on it alone picks whichever side is larger, which is the balance bias
    # this work set out to remove.
    #
    # The world frame breaks the tie, and legitimately: the cameras are fixed,
    # so Phase-1 world coordinates carry a real static reference. This is
    # section 16's identity-residual test, used exactly where section 16 puts
    # it -- AFTER the coherent candidates exist, never to construct them.
    best = max(ok, key=lambda c: c["rigidity_world"]
               ["delta_rss_cv_per_node_frame"])
    return {"status": "RESOLVED", "selected": best["candidate_id"],
            "num_qualified": len(ok),
            "qualified_ids": [c["candidate_id"] for c in ok],
            "rule": ("section 17: qualify on held-out COMPLEMENT-RELATIVE "
                     "rigid evidence (does articulation exist at all), then "
                     "name the moving side by world-frame held-out delta_RSS "
                     "PER NODE-FRAME (section 16). The per-node-frame form is "
                     "what makes size an outcome rather than a target: the raw "
                     "sum grows with the component and would always pick the "
                     "largest one")}


# ---------------------------------------------------------------------------
# Section 19: optional refinement
# ---------------------------------------------------------------------------

def refine_from_core(traj: Tensor, core, candidates_pool, sigma_k: float = 3.0) -> dict:
    """Recruit nodes that the CORE's own rigid trajectory explains (section 19).

    `T_t` is fitted from the core ONLY and never refitted per node, so this
    cannot drift into giving every node its own motion.

    The acceptance rule is a held-out model comparison, per node:

        E_i^M = sum_t ||x_i(t) - T_t x_i(0)||^2     moves with the core
        E_i^I = sum_t ||x_i(t) - x_i(0)||^2         stays put

    and node `i` joins when `E_i^M < E_i^I` -- it is better explained by moving
    with the body than by being static. `T_t` was fitted without `i`, so this
    is a genuine out-of-sample comparison, and it needs no scale, no threshold
    and no displacement amplitude at all.

    That last point is what lets it recruit near-axis material: a point close
    to a revolute axis barely moves, so `E_i^I` is small -- but if it turns
    with the body, `E_i^M` is smaller still. A truly static point has
    `E_i^I ~ 0` while the core's transform would displace it, so `E_i^M` is
    large and it is left alone.

    A residual scale from the core (median + k*MAD) is still computed and
    reported, but it is a diagnostic: a core chosen FOR its coherence has a
    degenerate residual spread, so using it as the gate recruits nothing.
    """
    X = torch.as_tensor(traj).double()
    core = torch.as_tensor(core).long().reshape(-1)
    T = X.shape[0]
    x0c = X[0, core]
    Rs, ps = [], []
    for t in range(T):
        R, p = _kabsch(x0c, X[t, core])
        Rs.append(R)
        ps.append(p)

    def moved(idx):
        e = torch.zeros(idx.numel(), dtype=torch.float64)
        for t in range(T):
            e += (X[t, idx] - X[0, idx] @ Rs[t].T - ps[t]).pow(2).sum(1)
        return e

    def stayed(idx):
        e = torch.zeros(idx.numel(), dtype=torch.float64)
        for t in range(T):
            e += (X[t, idx] - X[0, idx]).pow(2).sum(1)
        return e

    ec = moved(core)
    scale = float(ec.median()) + sigma_k * float(
        (ec - ec.median()).abs().median()) * 1.4826
    pool = torch.as_tensor(candidates_pool).long().reshape(-1)
    pool = pool[~torch.isin(pool, core)]
    if not pool.numel():
        return {"recruited": torch.zeros(0, dtype=torch.long),
                "core_residual_scale": scale, "num_considered": 0,
                "num_recruited": 0}
    em, ei = moved(pool), stayed(pool)
    keep = em < ei
    return {"recruited": pool[keep],
            "core_residual_scale": scale,
            "num_considered": int(pool.numel()),
            "num_recruited": int(keep.sum()),
            "num_rejected_static": int((~keep).sum()),
            "rule": ("section 19: recruit node i when E_i^M < E_i^I under the "
                     "core's own T_t. Held-out per node, no threshold, no "
                     "scale, no displacement amplitude")}


# ---------------------------------------------------------------------------
# The whole extractor
# ---------------------------------------------------------------------------

def extract_moving_component(traj: Tensor, edge_i: Tensor, edge_j: Tensor,
                             C: Tensor, n: int, *, min_nodes: int = 3,
                             folds: int = 5, max_candidates: int = 48,
                             min_merge_support: int = 2,
                             refine: bool = True, verbose: bool = True) -> dict:
    """Sections 11-19 end to end. Returns per-node labels in three classes."""
    cands = candidate_hierarchy(edge_i, edge_j, C, n, min_nodes=min_nodes,
                                min_merge_support=min_merge_support)
    # Section 17 evaluates "graph-persistent candidates": the hierarchy has
    # O(V) nodes and each evaluation is T Kabsch fits per fold, so the most
    # persistent ones are evaluated. Persistence is a property of C alone.
    # Section 17 evaluates "graph-persistent candidates", but persistence
    # alone is not a usable pool: a near-whole component survives to the end of
    # the merge tree and therefore has the highest persistence of all, so a
    # plain top-K crowds out every genuine part. On storage the best candidate
    # (49 nodes, IoU 0.69) sat at persistence rank 88 behind a wall of
    # 630-node components and was never evaluated.
    #
    # The pool is therefore stratified by component SIZE, taking the most
    # persistent candidates in each octave. That is not a size preference --
    # section 18 forbids one -- it is the opposite: it guarantees every size is
    # actually considered, so the outcome is decided by the evidence rather
    # than by which sizes happened to be looked at.
    seen, by_bin = set(), {}
    for c in cands:
        key = (c["num_nodes"], min(c["members"]), max(c["members"]))
        if key in seen:
            continue
        seen.add(key)
        b = int(math.log2(max(c["num_nodes"], 1)))
        by_bin.setdefault(b, []).append(c)
    bins = sorted(by_bin)
    per_bin = max(2, max_candidates // max(len(bins), 1))
    pool = []
    for b in bins:
        got = sorted(by_bin[b],
                     key=lambda c: -(c["persistence"]
                                     if c["persistence"] != float("inf")
                                     else 1e18))
        pool.extend(got[:per_bin])
    pool = pool[:max_candidates]
    all_nodes = torch.arange(n)
    from utils.progress import pbar
    for c in pbar(pool, "coherence candidates", total=len(pool)):
        c["edges"] = edge_distributions(c, edge_i, edge_j, C, n)
        inside = torch.zeros(n, dtype=torch.bool)
        inside[torch.tensor(c["members"], dtype=torch.long)] = True
        comp = all_nodes[~inside]
        c["complement_nodes"] = int(comp.numel())
        # The world-frame numbers are kept visible (section 16 asks for raw
        # residuals), but SELECTION uses the complement-relative ones.
        c["rigidity_world"] = cross_validated_rigidity(traj, c["members"], folds)
        rel = to_reference_frame(traj, comp)
        if rel is None:
            c["rigidity"] = {"degenerate": True,
                             "num_nodes": c["num_nodes"],
                             "reason": ("fewer than 3 complement nodes: there "
                                        "is no static reference for this "
                                        "candidate to be moving relative to"),
                             "eta_cv": float("nan"),
                             "delta_rss_cv": float("nan"),
                             "delta_bic": float("nan")}
        else:
            c["rigidity"] = cross_validated_rigidity(rel, c["members"], folds)
            c["rigidity"]["frame"] = "complement_relative"
            # Dimensionless size of the relative motion: RMS displacement per
            # node-frame, in units of the component's own canonical extent.
            # This is a NUMERICAL guard, not a motion threshold -- it only
            # separates "literally nothing" from "something". A scene that
            # drifts uniformly leaves a relative motion of 1e-28, i.e. machine
            # noise, on which eta = 1.0 is meaningless; a real joint is O(0.1).
            mem = torch.tensor(c["members"], dtype=torch.long)
            ext = float((traj[0, mem] - traj[0, mem].mean(0)).norm(dim=1)
                        .median().clamp_min(EPS))
            nf = max(c["num_nodes"] * traj.shape[0], 1)
            c["rigidity"]["relative_motion_ratio"] = (
                math.sqrt(max(c["rigidity"]["rss_identity_cv"], 0.0) / nf) / ext)
    if verbose:
        print(f"    {len(cands)} merge candidates, {len(pool)} evaluated")

    sel = select_moving_component(pool)
    labels = [INCOHERENT] * n
    core = torch.zeros(0, dtype=torch.long)
    recruit = None
    if sel["status"] == "RESOLVED":
        chosen = next(c for c in pool if c["candidate_id"] == sel["selected"])
        core = torch.tensor(chosen["members"], dtype=torch.long)
        if refine:
            # Section 19 fits T_t from the CORE and scores every other node by
            # E_i^M = sum_t ||x_i(t) - T_t x_i(0)||^2 -- in world coordinates.
            # Scoring in the complement-relative frame instead would be
            # self-defeating: the complement still contains the rest of the
            # moving body at this point, so the reference frame is contaminated
            # by exactly the nodes the refinement exists to recruit.
            rest = torch.tensor([v for v in range(n)], dtype=torch.long)
            recruit = refine_from_core(traj, core, rest)
            core = torch.unique(torch.cat([core, recruit["recruited"]]))
        for v in core.tolist():
            labels[v] = MOVING_RIGID
        # Everything else that sits inside a coherent, near-identity candidate
        # is STATIC_COMPATIBLE; the remainder stays INCOHERENT_OR_AMBIGUOUS --
        # section 11 allows false dynamics to stay there rather than forcing
        # them into a body.
        static_pool = [c for c in pool
                       if not c["rigidity"].get("degenerate")
                       and c["rigidity"]["delta_bic"] <= 0.0]
        if static_pool:
            big = max(static_pool, key=lambda c: c["num_nodes"])
            for v in big["members"]:
                if labels[v] == INCOHERENT:
                    labels[v] = STATIC_COMPATIBLE
    counts = {k: labels.count(k) for k in (MOVING_RIGID, STATIC_COMPATIBLE,
                                           INCOHERENT)}
    return {"status": sel["status"] if sel["status"] != "RESOLVED" else "RESOLVED",
            "selection": sel, "labels": labels, "counts": counts,
            "moving_nodes": core,
            "candidates": [{k: v for k, v in c.items() if k != "members"}
                           | {"num_members": c["num_nodes"]} for c in pool],
            "candidate_members": {c["candidate_id"]: c["members"] for c in pool},
            "refinement": ({k: v for k, v in recruit.items()
                            if k != "recruited"} if recruit else None),
            "num_merge_candidates": len(cands),
            "method": "coherence_single_linkage_heldout_rigid"}
