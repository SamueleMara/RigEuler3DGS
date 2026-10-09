"""Sections 10, 11: assigning supernodes to the static part or one of K parts.

CODEX_IMPLEMENT_GEOMETRIC_RIGID_MOTION_JOINT_DISCOVERY.md sections 7, 10, 11, 27.

Section 11's energy is

    E(z,T) = E_render(z,T) + E_rigid(z,T) + E_graph(z)

with, in the shared-transform form of section 11.2,

    E_rigid = sum_{t,k} sum_{i: z_i = k} || x_{i,t}^inferred - T_{k,t} x_i^0 ||^2

and section 11.3's Potts term on the supernode adjacency graph. Section 7 makes
RIGIDITY the main structural prior: a label is good when the member actually
moves the way that part's single rigid transform says it should.

The unaries are built per supernode because that is the variable the multi-label
cut optimises, and the move is the same exact alpha-expansion the support stage
already uses -- there is no second, weaker optimiser here.
"""

from __future__ import annotations

import torch

EPS = 1e-12


@torch.no_grad()
def rigid_assignment_costs(x0: torch.Tensor, supernode_id: torch.Tensor,
                           positions: list, transforms,
                           num_supernodes: int,
                           weights: torch.Tensor | None = None
                           ) -> torch.Tensor:
    """Section 11.2's per-supernode cost of each label, including static.

    Returns ``[C, K+1]``: column 0 is the static hypothesis T_{0,t} = I
    (section 8), column k the cost of explaining the supernode with part k's
    rigid transform.

    The cost is the mean squared position error over the window, so it is
    directly comparable across labels and does not grow with the number of
    members a supernode happens to contain.
    """
    K = transforms.K
    dev = x0.device
    sid = supernode_id.long().clamp_min(0)
    cost = torch.zeros((num_supernodes, K + 1), dtype=torch.float64, device=dev)
    count = torch.zeros(num_supernodes, dtype=torch.float64, device=dev)
    w = (torch.ones(x0.shape[0], dtype=torch.float64, device=dev)
         if weights is None else weights.double().clamp_min(0.0))
    count.index_add_(0, sid, w)

    x0d = x0.double()
    for t, pos in enumerate(positions):
        posd = pos.double()
        # label 0: static, so the prediction is the canonical geometry itself
        d0 = ((posd - x0d) ** 2).sum(dim=1) * w
        cost[:, 0].index_add_(0, sid, d0)
        for k in range(1, K + 1):
            R = transforms.R[t, k - 1].double()
            tt = transforms.t[t, k - 1].double()
            dk = ((posd - (x0d @ R.T + tt)) ** 2).sum(dim=1) * w
            cost[:, k].index_add_(0, sid, dk)
    return cost / count.clamp_min(EPS).unsqueeze(1) / max(len(positions), 1)


@torch.no_grad()
def assign_parts(costs: torch.Tensor, edges: torch.Tensor | None = None, *,
                 edge_weights: torch.Tensor | None = None,
                 potts_weight: float = 1.0,
                 initial: torch.Tensor | None = None,
                 max_cycles: int = 8) -> dict:
    """Sections 10 D/E and 11.3: multi-label assignment with a Potts prior.

    Reuses the exact alpha-expansion already used for support, so the spatial
    regularisation of section 10 E is solved by the same min-cut machinery
    rather than by a greedy pass.
    """
    from joint_reduction.kpart_support import alpha_expansion, potts_energy

    C, L = costs.shape
    dev = costs.device
    # alpha-expansion needs non-negative unaries; subtracting the per-node
    # minimum is an additive constant per node and changes no argmin.
    unary = (costs - costs.min(dim=1, keepdim=True).values).clamp_min(0.0).float()
    if edges is None:
        edges = torch.zeros((0, 2), dtype=torch.long)
    edges = edges.cpu().long()
    w = (torch.ones(edges.shape[0]) if edge_weights is None
         else edge_weights.cpu().float())
    init = (costs.argmin(dim=1) if initial is None else initial.long()).cpu()
    labels, info = alpha_expansion(unary.cpu(), edges, w,
                                   lam=float(potts_weight), initial=init,
                                   max_cycles=int(max_cycles))
    data, smooth = potts_energy(labels, unary.cpu(), edges, w,
                                float(potts_weight))
    labels = labels.to(dev)
    return {"labels": labels, "energy": data + smooth,
            "data_energy": data, "smoothness_energy": smooth,
            "unary": unary, "expansion": info,
            "counts": [int((labels == l).sum()) for l in range(L)],
            "argmin_labels": costs.argmin(dim=1)}


@torch.no_grad()
def pairwise_distance_violation(x0: torch.Tensor, positions: list,
                                edges: torch.Tensor) -> torch.Tensor:
    """Section 11.2's pairwise form, as a per-edge rigidity violation.

    Renamed from `pairwise_rigidity_cost` so it no longer collides with the
    section 15 label-cost function of that name, which shadowed it silently.

        ( ||x_i,t - x_j,t|| - ||x_i,0 - x_j,0|| )^2

    Kept alongside the shared-transform form because section 7 asks for
    pairwise-distance preservation explicitly, and because it needs no
    transform -- it can therefore CHECK the transform rather than assume it.
    """
    if edges.numel() == 0:
        return torch.zeros(0, dtype=torch.float64, device=x0.device)
    i, j = edges[:, 0].long(), edges[:, 1].long()
    d0 = (x0.double()[i] - x0.double()[j]).norm(dim=1)
    acc = torch.zeros(i.shape[0], dtype=torch.float64, device=x0.device)
    for pos in positions:
        dt = (pos.double()[i] - pos.double()[j]).norm(dim=1)
        acc += (dt - d0) ** 2
    return acc / max(len(positions), 1)


# ---------------------------------------------------------------------------
# CODEX_IMPLEMENT_WHOLE_WINDOW_RIGID_COMPONENT_DISCOVERY.md sections 14, 15, 27
# ---------------------------------------------------------------------------
@torch.no_grad()
def whole_track_assignment_cost(x0: torch.Tensor, tracks: torch.Tensor,
                                components: list, length: torch.Tensor | float,
                                confidence: torch.Tensor | None = None, *,
                                huber_c: float = 2.0) -> torch.Tensor:
    """Section 14's C_ik^track over the WHOLE window.

        C_ik = sum_{t in V_i} omega_{i,t} rho( ||x_i^t - T_{k,t} x_i^0|| / L_i )

    `components` is a list of {"R": [T,3,3], "t": [T,3]} -- one per candidate
    rigid component, the static one included as an ordinary competitor
    (section 48). `length` normalises so the cost is dimensionless and
    comparable across components and scenes.

    Section 14 is explicit that this must not be computed from the final-frame
    displacement, nor from one instantaneous twist. The storage failure is
    exactly what happens when membership is decided on a single number: a
    component whose transform collapsed to the identity scores the same as the
    static hypothesis, and the labels stop carrying information.
    """
    T, C, _ = tracks.shape
    dev = tracks.device
    L = (length if torch.is_tensor(length)
         else torch.full((C,), float(length), dtype=torch.float64, device=dev))
    L = L.double().clamp_min(EPS)
    x0d = x0.double()
    cost = torch.zeros((C, len(components)), dtype=torch.float64, device=dev)
    for k, comp in enumerate(components):
        R, t = comp["R"].double(), comp["t"].double()
        acc = torch.zeros(C, dtype=torch.float64, device=dev)
        wsum = torch.zeros(C, dtype=torch.float64, device=dev)
        for f in range(T):
            pred = x0d @ R[f].T + t[f]
            r = (tracks[f].double() - pred).norm(dim=1) / L
            # Huber on the normalised residual: quadratic near zero so small
            # disagreements still discriminate, linear far out so one bad
            # frame cannot decide membership by itself.
            rho = torch.where(r <= huber_c, 0.5 * r * r,
                              huber_c * (r - 0.5 * huber_c))
            w = (torch.ones(C, dtype=torch.float64, device=dev)
                 if confidence is None else confidence[f].double())
            acc += w * rho
            wsum += w
        cost[:, k] = acc / wsum.clamp_min(EPS)
    return cost


@torch.no_grad()
def pairwise_rigidity_cost(labels: torch.Tensor, edges: torch.Tensor,
                           affinity: torch.Tensor, num_labels: int,
                           ) -> torch.Tensor:
    """Section 15's C_ik^pair: local same-component evidence.

        C_ik^pair = - sum_{j in N(i)} w_ij 1[z_j = k]

    Negative because a neighbour that is rigidly compatible and already in k is
    evidence FOR k. It keeps components spatially coherent and stops isolated
    noisy supernodes flipping label on their own track alone.
    """
    C = int(labels.shape[0])
    out = torch.zeros((C, num_labels), dtype=torch.float64, device=labels.device)
    if edges.numel() == 0:
        return out
    i, j = edges[:, 0].long(), edges[:, 1].long()
    w = affinity.double()
    for a, b in ((i, j), (j, i)):
        lab = labels[b].long()
        out.index_put_((a, lab), -w, accumulate=True)
    return out


@torch.no_grad()
def assignment_margin(cost: torch.Tensor) -> dict:
    """Section 27: how separable the labels actually are.

        dC_i = C_i^second - C_i^best

    A margin near zero on every supernode is the signature of the storage
    collapse: the static and moving hypotheses explained the tracks equally
    well because the moving transform had become the identity.
    """
    srt, order = cost.sort(dim=1)
    best = srt[:, 0]
    second = srt[:, 1] if cost.shape[1] > 1 else torch.full_like(best, float("nan"))
    margin = second - best
    return {"best": best, "second": second, "margin": margin,
            "best_label": order[:, 0],
            "median_margin": float(margin[~torch.isnan(margin)].median())
            if int((~torch.isnan(margin)).sum()) else float("nan"),
            "frac_below_1pct_of_best": float(
                ((margin < 0.01 * best.clamp_min(EPS))
                 & ~torch.isnan(margin)).double().mean())}


@torch.no_grad()
def update_component_labels(track_cost: torch.Tensor, edges, affinity,
                            labels: torch.Tensor, *, lam_pair: float = 1.0,
                            lam_graph: float = 1.0, inner: int = 3,
                            max_cycles: int = 8) -> dict:
    """Sections 14, 15 and 18: labels from the combined cost.

    The pairwise term depends on the labels it is helping to choose, so the
    unary is rebuilt from the current labelling and the exact alpha-expansion
    is re-run a few times -- the same min-cut machinery the support stage
    uses, not a weaker greedy pass.
    """
    num_labels = int(track_cost.shape[1])
    lab = labels.clone().long()
    info = []
    for it in range(int(inner)):
        pair = pairwise_rigidity_cost(lab, edges, affinity, num_labels)
        unary = track_cost + lam_pair * pair
        out = assign_parts(unary, edges, edge_weights=affinity,
                           potts_weight=lam_graph, initial=lab,
                           max_cycles=max_cycles)
        new = out["labels"]
        changed = int((new != lab).sum())
        info.append({"inner": it, "changed": changed,
                     "energy": out["energy"], "counts": out["counts"]})
        lab = new
        if changed == 0:
            break
    return {"labels": lab, "history": info,
            "margin": assignment_margin(track_cost)}
