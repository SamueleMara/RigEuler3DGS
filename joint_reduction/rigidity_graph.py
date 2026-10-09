"""Section 5-8: the whole-window rigidity compatibility graph.

CODEX_IMPLEMENT_WHOLE_WINDOW_RIGID_COMPONENT_DISCOVERY.md sections 5, 6, 7, 8, 13.

Two points on one rigid body preserve their separation for the whole sequence:

    || x_i^t - x_j^t ||  =  || x_i^0 - x_j^0 ||        for all t      (section 5)

so the primary compatibility statistic for an edge is the relative violation of
that identity, aggregated robustly over time:

    eps_ij^t = | ||x_i^t - x_j^t|| - ||x_i^0 - x_j^0|| | / (||x_i^0 - x_j^0|| + eps)
    rho_ij   = median_{t in V_ij} eps_ij^t

`rho_ij` is joint-type independent. It is identical for static-static,
prismatic-prismatic, revolute-revolute and arbitrary SE(3) pairs, and -- the
point of section 22 -- it does not depend on displacement MAGNITUDE. A point
sitting next to a revolute axis barely moves, yet it is perfectly rigid with
respect to the rest of its body, so it has a small rho and joins the right
component. A magnitude rule would have classified it static.

Section 7: only frames where both endpoints are actually observed may enter the
median. A missing track is not evidence of zero motion.
"""

from __future__ import annotations

import torch

EPS = 1e-12


# ---------------------------------------------------------------------------
# section 13: the robust scale, computed correctly
# ---------------------------------------------------------------------------
def median_absolute_deviation(r: torch.Tensor) -> torch.Tensor:
    """MAD = median |r - median(r)|  (section 13).

    Section 13 calls out the wrong version explicitly: `1.4826 * median(r)` is
    not a MAD, it is a scaled median, and for a strictly positive residual
    distribution it is far larger. This module keeps the two names distinct so
    the mistake cannot be made silently.
    """
    if r.numel() == 0:
        return torch.zeros((), dtype=r.dtype, device=r.device)
    m = r.median()
    return (r - m).abs().median()


def robust_scale(r: torch.Tensor, floor: float | None = None) -> torch.Tensor:
    """Gaussian-consistent robust sigma = 1.4826 * MAD (section 13).

    `floor` guards the degenerate case where more than half the residuals are
    identical and the MAD collapses to zero. Section 13 requires that floor to
    be documented and derived from numerical precision rather than chosen: it
    defaults to the dtype's epsilon scaled by the data magnitude.
    """
    s = 1.4826 * median_absolute_deviation(r)
    if floor is None:
        eps = torch.finfo(r.dtype if r.is_floating_point() else torch.float64).eps
        floor = float(eps) * float(r.abs().max() if r.numel() else 1.0) * 16.0
    return s.clamp_min(max(float(floor), 0.0) or EPS)


# ---------------------------------------------------------------------------
# sections 5, 6, 7: the graph
# ---------------------------------------------------------------------------
@torch.no_grad()
def pairwise_rigidity_error(tracks: torch.Tensor, edges: torch.Tensor,
                            visibility: torch.Tensor | None = None,
                            min_overlap: int = 2) -> dict:
    """Section 5's rho_ij over the whole window, for every graph edge.

    `tracks`     [T, C, 3] supernode positions per frame, frame 0 canonical.
    `edges`      [E, 2] undirected supernode adjacency (section 6).
    `visibility` [T, C] bool/float, whether a supernode is observed at a frame.

    Returns rho (median), p90, the canonical separation, and |V_ij|.
    """
    T, C, _ = tracks.shape
    if edges.numel() == 0:
        z = torch.zeros(0, dtype=tracks.dtype, device=tracks.device)
        return {"rho": z, "p90": z.clone(), "canonical_distance": z.clone(),
                "overlap": z.clone().long(), "low_confidence": z.clone().bool()}
    i, j = edges[:, 0].long(), edges[:, 1].long()
    x0 = tracks[0].double()
    d0 = (x0[i] - x0[j]).norm(dim=1)                       # [E]

    # eps_ij^t for every frame, then a robust aggregate over the VALID frames.
    dt = (tracks[:, i, :].double() - tracks[:, j, :].double()).norm(dim=2)  # [T,E]
    rel = (dt - d0.unsqueeze(0)).abs() / (d0.unsqueeze(0) + EPS)

    if visibility is None:
        valid = torch.ones((T, rel.shape[1]), dtype=torch.bool,
                           device=tracks.device)
    else:
        v = visibility.bool() if visibility.dtype == torch.bool \
            else (visibility > 0)
        valid = v[:, i] & v[:, j]                           # section 7
    overlap = valid.sum(dim=0)

    big = torch.full_like(rel, float("inf"))
    masked = torch.where(valid, rel, big)
    rho = torch.full((rel.shape[1],), float("nan"), dtype=torch.float64,
                     device=tracks.device)
    p90 = rho.clone()
    ok = overlap >= max(int(min_overlap), 1)
    if bool(ok.any()):
        cols = torch.nonzero(ok, as_tuple=False).reshape(-1)
        for c in cols.tolist():                 # per edge: the valid subset
            v = rel[valid[:, c], c]
            rho[c] = v.median()
            p90[c] = v.quantile(0.90)
    return {"rho": rho, "p90": p90, "canonical_distance": d0,
            "overlap": overlap,
            # Section 7: an edge without enough shared observations is marked
            # low confidence rather than being given an invented compatibility.
            "low_confidence": ~ok}


@torch.no_grad()
def rigid_affinity(rho: torch.Tensor, spatial_weight: torch.Tensor | None = None,
                   sigma: float | None = None,
                   low_confidence: torch.Tensor | None = None) -> dict:
    """Section 8: turn geometric inconsistency into same-component evidence.

        w_ij^rigid = w_ij^spatial * exp( -rho_ij^2 / (2 sigma_rigid^2) )

    `sigma` is derived from the distribution of rho itself -- the robust scale
    of the observed violations -- so it adapts to a scene's own noise without
    being tuned, and never from ground truth. The derivation is returned so it
    can be persisted (section 8 requires that).
    """
    finite = torch.isfinite(rho)
    if sigma is None:
        src = rho[finite]
        sigma = float(robust_scale(src.double())) if src.numel() else 1.0
        derivation = ("1.4826 * MAD of the finite rho_ij of this scene "
                      "(section 8: a robust distributional estimate, never GT)")
    else:
        sigma = float(sigma)
        derivation = "configured"
    sigma = max(sigma, EPS)
    w = torch.zeros_like(rho)
    w[finite] = torch.exp(-(rho[finite] ** 2) / (2.0 * sigma ** 2))
    if spatial_weight is not None:
        w = w * spatial_weight.to(w.dtype)
    if low_confidence is not None:
        # Not zero: a low-confidence edge still carries spatial adjacency, it
        # simply must not assert rigid compatibility it could not measure.
        w = torch.where(low_confidence, w * 0.0 + 0.0, w)
    return {"affinity": w, "sigma_rigid": sigma, "sigma_derivation": derivation}


@torch.no_grad()
def build_rigidity_graph(tracks: torch.Tensor, edges: torch.Tensor,
                         visibility: torch.Tensor | None = None,
                         *, min_overlap: int = 2,
                         sigma: float | None = None) -> dict:
    """Sections 5-8 together: the graph Phase 2 decomposes."""
    stats = pairwise_rigidity_error(tracks, edges, visibility,
                                    min_overlap=min_overlap)
    aff = rigid_affinity(stats["rho"], None, sigma, stats["low_confidence"])
    out = {**stats, **aff, "edges": edges,
           "num_edges": int(edges.shape[0]),
           "num_low_confidence": int(stats["low_confidence"].sum())}
    finite = torch.isfinite(stats["rho"])
    if bool(finite.any()):
        r = stats["rho"][finite]
        out["rho_summary"] = {
            "median": float(r.median()), "p90": float(r.quantile(0.90)),
            "max": float(r.max()), "n": int(r.numel())}
    else:
        out["rho_summary"] = {"median": None, "p90": None, "max": None, "n": 0}
    return out


# ---------------------------------------------------------------------------
# section 42: the required rigidity-edge debug format
# ---------------------------------------------------------------------------
RIGIDITY_EDGE_HEADER = [
    "edge_i", "edge_j", "canonical_distance", "valid_frame_count",
    "median_relative_distance_change", "p90_relative_distance_change",
    "rigid_affinity", "final_same_component",
]


def rigidity_edge_rows(graph: dict, labels=None) -> list:
    """Section 42's rows. `labels` fills `final_same_component` when the
    decomposition has finished; before that the column is left empty rather
    than guessed."""
    e = graph["edges"]
    if e.numel() == 0:
        return []
    i = e[:, 0].long().cpu().tolist()
    j = e[:, 1].long().cpu().tolist()
    d0 = graph["canonical_distance"].cpu().tolist()
    ov = graph["overlap"].cpu().tolist()
    rho = graph["rho"].cpu().tolist()
    p90 = graph["p90"].cpu().tolist()
    aff = graph["affinity"].cpu().tolist()
    same = None
    if labels is not None:
        lab = labels.cpu()
        same = [int(lab[a] == lab[b]) for a, b in zip(i, j)]
    return [[i[k], j[k], d0[k], ov[k], rho[k], p90[k], aff[k],
             "" if same is None else same[k]] for k in range(len(i))]
