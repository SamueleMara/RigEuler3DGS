"""Static/dynamic segmentation from trajectory metric coherence alone.

CODEX_IMPLEMENT_MOTION_COHERENCE_STATIC_DYNAMIC_FILTER.md.

This module answers one question (section 49):

    Which groups of Phase-1 free trajectories preserve their internal metric
    over the whole sequence?

NOT "which points moved far?". The distinction matters because Phase 1 spends
free per-Gaussian degrees of freedom compensating illumination, appearance
mismatch, alpha ordering and depth/model error. Those trajectories can have an
amplitude comparable to the real moving body, so any displacement-magnitude
rule confuses them with it. They do not, however, preserve their neighbours'
relative geometry, and a real rigid body does -- under translation, rotation,
arbitrary SE(3), variable speed, stop/restart and reversal alike.

The invariant (section 5):

    ||x_i(t) - x_j(t)|| = constant      for i, j on the same rigid body

equivalently, differentially (section 8):

    r_ij^T (v_j - v_i) = r_ij^T D r_ij = 0   since r^T Omega r = 0 and D = 0

STRICT SEPARATION (section 18). This module deliberately imports nothing from
`estimator.moving_support`, uses no Otsu, no median or top-k displacement, no
hand-picked displacement threshold, no RGB residual, no semantic feature, no
P/R type, no joint axis, no Phase-2 render loss and no ground truth. Ground
truth is for evaluation, after the prediction is on disk, and lives in
`motion_coherence_eval.py` so it cannot be reached from here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

UNRESOLVED = "UNRESOLVED_COHERENCE_PARTITION"
STATIC, DYNAMIC = "STATIC", "DYNAMIC"

# Numerical guards. These are machine/data-scale floors, NOT motion thresholds:
# nothing is declared moving or static by comparing against them.
EPS = 1e-12


# ---------------------------------------------------------------------------
# Section 3: supernode trajectories
# ---------------------------------------------------------------------------

def supernode_trajectories(positions, supernode_id: Tensor,
                           weights: Tensor | None = None) -> dict:
    """`x_s(t)` from the fixed Phase-1 supernode membership.

        x_s(t) = sum_{i in s} w_i x_i(t) / sum_{i in s} w_i

    The weight convention is uniform `w_i = 1` unless the caller supplies one,
    and it is recorded in the output so the reduction is reproducible. The
    membership is Phase-1's own and is never recomputed here -- an edge must
    refer to the same two supernodes for the whole sequence (section 4).
    """
    sid = torch.as_tensor(supernode_id).long().reshape(-1).cpu()
    valid = sid >= 0
    n_s = int(sid[valid].max()) + 1 if int(valid.sum()) else 0
    w = (torch.ones(sid.numel(), dtype=torch.float64) if weights is None
         else torch.as_tensor(weights).double().reshape(-1).cpu())
    w = torch.where(valid, w, torch.zeros_like(w))
    denom = torch.zeros(n_s, dtype=torch.float64).index_add_(
        0, sid.clamp_min(0), w)
    traj = []
    for p in positions:
        x = torch.as_tensor(p).double().cpu()
        acc = torch.zeros((n_s, 3), dtype=torch.float64).index_add_(
            0, sid.clamp_min(0), x * w[:, None])
        traj.append(acc / denom.clamp_min(EPS)[:, None])
    keep = denom > 0
    return {"trajectories": torch.stack(traj),          # [T, n_s, 3]
            "valid": keep, "num_supernodes": n_s,
            "weight_convention": ("uniform w_i = 1" if weights is None
                                  else "caller-supplied"),
            "members_per_supernode": torch.zeros(n_s, dtype=torch.long)
            .index_add_(0, sid.clamp_min(0), valid.long())}


# ---------------------------------------------------------------------------
# Section 4: one fixed canonical graph
# ---------------------------------------------------------------------------

def canonical_knn_graph(x0: Tensor, k: int = 12) -> dict:
    """Symmetric kNN edges built ONCE in the canonical frame t = 0.

    Rebuilding neighbours per frame would let the graph track the motion it is
    supposed to measure: an edge that stretches would simply be replaced by a
    shorter one and the strain would vanish by construction.
    """
    x = torch.as_tensor(x0).double().cpu()
    n = x.shape[0]
    k = max(1, min(int(k), n - 1))
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    nb = d.topk(k, largest=False).indices                 # [n, k]
    src = torch.arange(n).repeat_interleave(k)
    dst = nb.reshape(-1)
    a = torch.minimum(src, dst)
    b = torch.maximum(src, dst)
    key = a * n + b
    uniq = torch.unique(key)
    ei, ej = uniq // n, uniq % n
    return {"edge_i": ei, "edge_j": ej, "k": k, "num_nodes": n,
            "num_edges": int(ei.numel()),
            "canonical_distance": (x[ei] - x[ej]).norm(dim=1)}


# ---------------------------------------------------------------------------
# Sections 5-9: strain, coherence, diagnostics
# ---------------------------------------------------------------------------

def edge_strain(traj: Tensor, edge_i: Tensor, edge_j: Tensor,
                times: Tensor | None = None, w_t: Tensor | None = None,
                eps_len: float | None = None) -> dict:
    """`l_ij(t)`, `z_ij(t)`, `C_ij`, `eps_ij(t)`, `J_ij`.

    `z_ij(t) = log((l + eps) / (lbar + eps))` with `lbar = median_t l` -- the
    median over the WHOLE window, not frame 0 and not the endpoint, so an edge
    that moves and returns (section 36 S6) is still measured against its own
    typical length, and an isolated bad frame cannot set the reference.

    `eps_len` is a numerical guard derived from the data scale, never a motion
    threshold: it only keeps the log finite for degenerate zero-length edges.
    """
    X = torch.as_tensor(traj).double()                    # [T, n, 3]
    T = X.shape[0]
    r = X[:, edge_j, :] - X[:, edge_i, :]                 # [T, E, 3]
    ell = r.norm(dim=2)                                   # [T, E]
    if eps_len is None:
        eps_len = float(ell[ell > 0].median()) * 1e-9 if (ell > 0).any() else EPS
    lbar = ell.median(dim=0).values                       # [E]
    z = torch.log((ell + eps_len) / (lbar + eps_len).clamp_min(EPS))

    w = (torch.ones_like(z) if w_t is None
         else torch.as_tensor(w_t).double().reshape(T, -1).expand_as(z))
    C = ((w * z * z).sum(0) / (w.sum(0) + EPS)).clamp_min(0.0).sqrt()

    t_arr = (torch.arange(T, dtype=torch.float64) if times is None
             else torch.as_tensor(times).double().reshape(-1))
    dt = (t_arr[1:] - t_arr[:-1]).clamp_min(EPS)[:, None]
    dz = z[1:] - z[:-1]
    epsilon = dz / dt                                     # [T-1, E]
    J = ((dz * dz).sum(0) / max(T - 1, 1)).sqrt()
    return {"length": ell, "z": z, "C": C, "epsilon": epsilon, "J": J,
            "reference_length": lbar, "eps_len": eps_len}


def differential_strain_from_velocity(r: Tensor, dv: Tensor) -> Tensor:
    """Section 8's identity form: `r^T (v_j - v_i) / ||r||^2`.

    Used by the section 37 numerical check that this and the finite log-length
    difference agree, and that both go to zero for exact rigid motion.
    """
    r = torch.as_tensor(r).double()
    dv = torch.as_tensor(dv).double()
    return (r * dv).sum(-1) / (r * r).sum(-1).clamp_min(EPS)


# ---------------------------------------------------------------------------
# Sections 11-12: affinity and the coherence-only binary cut
# ---------------------------------------------------------------------------

def coherence_affinity(C: Tensor, edge_i: Tensor, edge_j: Tensor, n: int,
                       sigma_floor: float | None = None) -> dict:
    """`W_ij = exp(-C_ij^2 / (sigma_i sigma_j + eps))`, self-scaled.

    `sigma_i = median_{j in N(i)} C_ij`, so the scale is local and no global
    coherence threshold is introduced. Displacement amplitude appears nowhere
    (section 11).
    """
    C = torch.as_tensor(C).double().reshape(-1)
    sig = torch.zeros(n, dtype=torch.float64)
    # local median per node, over its incident edges
    buckets: list[list[float]] = [[] for _ in range(n)]
    for e in range(C.numel()):
        buckets[int(edge_i[e])].append(float(C[e]))
        buckets[int(edge_j[e])].append(float(C[e]))
    med_all = float(C.median()) if C.numel() else 0.0
    floor = (sigma_floor if sigma_floor is not None
             else max(med_all * 1e-6, EPS))
    for v in range(n):
        b = buckets[v]
        sig[v] = (torch.tensor(b, dtype=torch.float64).median()
                  if b else torch.tensor(0.0, dtype=torch.float64))
    sig = sig.clamp_min(floor)
    w = torch.exp(-(C * C) / (sig[edge_i] * sig[edge_j] + EPS))
    return {"W": w, "sigma": sig, "sigma_floor": floor,
            "global_median_C": med_all}


def normalized_cut(edge_i: Tensor, edge_j: Tensor, W: Tensor, n: int) -> dict:
    """Two-way normalized cut from the Fiedler vector of L_sym.

    Nothing seeds this: no Otsu, no displacement, no current support, no GT,
    and no target dynamic fraction (section 12). The split point on the
    Fiedler vector is the one minimizing the normalized cut among its own
    sorted values, which is a property of the eigenvector, not an outside
    threshold.
    """
    A = torch.zeros((n, n), dtype=torch.float64)
    A[edge_i, edge_j] = W
    A[edge_j, edge_i] = W
    deg = A.sum(1)
    keep = deg > 0
    dm = deg.clamp_min(EPS).pow(-0.5)
    L = torch.eye(n, dtype=torch.float64) - (dm[:, None] * A * dm[None, :])
    L = 0.5 * (L + L.T)
    evals, evecs = torch.linalg.eigh(L)
    f = evecs[:, 1] * dm                                  # generalized Fiedler

    # sweep: the cut point is chosen by the Ncut objective itself
    order = torch.argsort(f)
    best, best_i = float("inf"), None
    volA = 0.0
    total = float(deg.sum())
    inA = torch.zeros(n, dtype=torch.bool)
    cut = 0.0
    for i in range(n - 1):
        v = int(order[i])
        inA[v] = True
        volA += float(deg[v])
        # edges from v: those to nodes already in A reduce the cut
        row = A[v]
        cut += float(row.sum()) - 2.0 * float(row[inA].sum())
        volB = total - volA
        if volA <= 0 or volB <= 0:
            continue
        nc = cut / volA + cut / volB
        if nc < best:
            best, best_i = nc, i
    if best_i is None:
        return {"labels": torch.zeros(n, dtype=torch.long), "ncut": float("inf"),
                "fiedler": f, "eigenvalues": evals[:4].tolist(),
                "degenerate": True}
    labels = torch.zeros(n, dtype=torch.long)
    labels[order[: best_i + 1]] = 1
    return {"labels": labels, "ncut": best, "fiedler": f,
            "eigenvalues": evals[:4].tolist(),
            "spectral_gap": float(evals[2] - evals[1]),
            "degenerate": bool(~keep.any()),
            "isolated_nodes": int((~keep).sum())}


# ---------------------------------------------------------------------------
# Section 14: ambiguity
# ---------------------------------------------------------------------------

def node_ambiguity(C: Tensor, edge_i: Tensor, edge_j: Tensor,
                   labels: Tensor, n: int) -> dict:
    """`C_i^in` and `C_i^out`: median coherence to same- and other-label
    neighbours. Persisted as confidence; ambiguous nodes are NOT repaired with
    a displacement rule (section 14)."""
    ins: list[list[float]] = [[] for _ in range(n)]
    outs: list[list[float]] = [[] for _ in range(n)]
    for e in range(C.numel()):
        a, b, c = int(edge_i[e]), int(edge_j[e]), float(C[e])
        same = labels[a] == labels[b]
        (ins if same else outs)[a].append(c)
        (ins if same else outs)[b].append(c)
    def med(xs):
        return torch.tensor([
            (torch.tensor(v, dtype=torch.float64).median() if v else float("nan"))
            for v in xs], dtype=torch.float64)
    c_in, c_out = med(ins), med(outs)
    return {"C_in": c_in, "C_out": c_out,
            "margin": c_out - c_in,
            "ambiguous": ~(c_out > c_in)}


# ---------------------------------------------------------------------------
# Sections 15-17: validate each component, then name it
# ---------------------------------------------------------------------------

def kabsch(P: Tensor, Q: Tensor, w: Tensor | None = None):
    """Weighted rigid fit mapping P -> Q. Validation only (section 15): this
    never constructs or seeds the partition."""
    P, Q = P.double(), Q.double()
    w = torch.ones(P.shape[0], dtype=torch.float64) if w is None else w.double()
    W = w.sum().clamp_min(EPS)
    pc, qc = (w[:, None] * P).sum(0) / W, (w[:, None] * Q).sum(0) / W
    H = (w[:, None] * (P - pc)).T @ (Q - qc)
    U, _S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=torch.float64))
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def validate_component(traj: Tensor, idx: Tensor, w: Tensor | None = None) -> dict:
    """Whole-window rigid residual `E_k^R`, identity residual `E_k^I`, gain."""
    X = torch.as_tensor(traj).double()
    idx = torch.as_tensor(idx).long().reshape(-1)
    if idx.numel() < 3:
        return {"num_nodes": int(idx.numel()), "E_rigid": float("nan"),
                "E_identity": float("nan"), "gain": float("nan"),
                "per_frame": [], "degenerate": True}
    x0 = X[0, idx]
    ww = torch.ones(idx.numel(), dtype=torch.float64) if w is None else w.double()
    num_r = num_i = den = 0.0
    rows = []
    for t in range(X.shape[0]):
        xt = X[t, idx]
        R, p = kabsch(x0, xt, ww)
        pred = x0 @ R.T + p
        er = float((ww * (xt - pred).pow(2).sum(1)).sum())
        ei = float((ww * (xt - x0).pow(2).sum(1)).sum())
        num_r += er
        num_i += ei
        den += float(ww.sum())
        th = float(torch.linalg.matrix_norm(R - torch.eye(3, dtype=torch.float64)))
        rows.append({"frame": t, "rigid_residual": er / max(float(ww.sum()), EPS),
                     "identity_residual": ei / max(float(ww.sum()), EPS),
                     "translation_norm": float(p.norm()),
                     "rotation_fro": th})
    ER, EI = num_r / max(den, EPS), num_i / max(den, EPS)
    return {"num_nodes": int(idx.numel()), "E_rigid": ER, "E_identity": EI,
            "gain": EI - ER, "per_frame": rows, "degenerate": False}


def identify_static_dynamic(comp0: dict, comp1: dict) -> dict:
    """Section 16: the component with the smaller coherent-motion gain is
    STATIC. This runs only AFTER the coherence partition exists, and uses no
    per-point displacement threshold."""
    g0, g1 = comp0.get("gain"), comp1.get("gain")
    if not (isinstance(g0, float) and isinstance(g1, float)) \
            or math.isnan(g0) or math.isnan(g1):
        return {"status": UNRESOLVED,
                "reason": "a component has no valid rigid fit"}
    if g0 == g1:
        return {"status": UNRESOLVED,
                "reason": "the two components have identical motion gain"}
    dyn = 0 if g0 > g1 else 1
    lo, hi = min(g0, g1), max(g0, g1)
    return {"status": "RESOLVED", "dynamic_component": dyn,
            "static_component": 1 - dyn,
            "gain": [g0, g1],
            "gain_ratio": (hi / lo) if lo > 0 else float("inf"),
            "rule": ("section 16: larger E^I - E^R is DYNAMIC; the comparison "
                     "is between components, never against a threshold")}


def component_motion_magnitude(comp: dict, length_scale: float) -> list:
    """Section 17's `M_k(t)`. Debug only; never reaches the partition."""
    L = max(float(length_scale), EPS)
    return [math.sqrt((r["translation_norm"] / L) ** 2 + r["rotation_fro"] ** 2)
            for r in comp.get("per_frame", [])]


# ---------------------------------------------------------------------------
# The whole branch, in one call
# ---------------------------------------------------------------------------

def run_motion_coherence(positions, supernode_id: Tensor, *, knn: int = 12,
                         times=None, weights=None, verbose: bool = True) -> dict:
    """Sections 3-17, in order. GT never enters here (section 21)."""
    sn = supernode_trajectories(positions, supernode_id, weights)
    traj = sn["trajectories"]
    keep = sn["valid"]
    node_ids = torch.nonzero(keep, as_tuple=False).reshape(-1)
    traj = traj[:, node_ids, :]
    n = int(node_ids.numel())
    if verbose:
        print(f"    {n} supernodes x {traj.shape[0]} frames")

    g = canonical_knn_graph(traj[0], k=knn)
    st = edge_strain(traj, g["edge_i"], g["edge_j"], times=times)
    aff = coherence_affinity(st["C"], g["edge_i"], g["edge_j"], n)
    cut = normalized_cut(g["edge_i"], g["edge_j"], aff["W"], n)
    labels = cut["labels"]
    amb = node_ambiguity(st["C"], g["edge_i"], g["edge_j"], labels, n)

    comps = []
    for k in (0, 1):
        comps.append(validate_component(traj, torch.nonzero(
            labels == k, as_tuple=False).reshape(-1)))
    ident = identify_static_dynamic(comps[0], comps[1])

    L = float((traj[0] - traj[0].mean(0)).norm(dim=1).median().clamp_min(EPS))
    for k in (0, 1):
        comps[k]["motion_magnitude"] = component_motion_magnitude(comps[k], L)

    out = {
        "status": ident["status"], "identification": ident,
        "num_supernodes": n, "node_ids": node_ids,
        "graph": {k: v for k, v in g.items()
                  if k not in ("edge_i", "edge_j", "canonical_distance")},
        "edge_i": g["edge_i"], "edge_j": g["edge_j"],
        "canonical_distance": g["canonical_distance"],
        "C": st["C"], "J": st["J"], "z": st["z"], "epsilon": st["epsilon"],
        "W": aff["W"], "sigma": aff["sigma"],
        "labels": labels, "cut": {k: v for k, v in cut.items()
                                  if k not in ("labels", "fiedler")},
        "fiedler": cut["fiedler"], "ambiguity": amb,
        "components": comps, "length_scale": L,
        "method": "trajectory_metric_coherence",
    }
    if ident["status"] == "RESOLVED":
        out["dynamic_nodes"] = node_ids[labels == ident["dynamic_component"]]
        out["static_nodes"] = node_ids[labels == ident["static_component"]]
    else:
        out["dynamic_nodes"] = torch.zeros(0, dtype=torch.long)
        out["static_nodes"] = torch.zeros(0, dtype=torch.long)
    return out


def propagate_to_gaussians(dynamic_nodes: Tensor, supernode_id: Tensor) -> Tensor:
    """Section 3: supernode labels -> member Gaussian indices."""
    sid = torch.as_tensor(supernode_id).long().reshape(-1).cpu()
    dyn = torch.zeros(int(sid.max()) + 1 if sid.numel() else 0,
                      dtype=torch.bool)
    d = torch.as_tensor(dynamic_nodes).long().reshape(-1)
    if d.numel():
        dyn[d] = True
    mask = (sid >= 0) & dyn[sid.clamp_min(0)]
    return torch.nonzero(mask, as_tuple=False).reshape(-1)
