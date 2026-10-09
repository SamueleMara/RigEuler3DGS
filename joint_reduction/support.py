"""Static / moving support discovery over canonical supernodes.

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md section 6.

Section 6 opens with "The method is incomplete until it determines which
canonical regions move." A joint fitted to every Gaussian in the scene is a
joint fitted partly to the static background, which drags the axis and the
trajectory. So each canonical supernode carries a support variable

    z_c in {0, 1}        1 = moves with the joint, 0 = static

and the static supernodes must stay at canonical geometry under the reduced
model.

The three things section 6 is strict about, and which shape this file:

  * 6.1 The motion evidence must come from the EXACT analytic renderer
    sensitivity a_{p,c} = sum_i H_{p,i} G_i(eta), not from optical flow and
    not from per-Gaussian velocity magnitude. That is what makes the evidence
    alpha-aware: a supernode hidden behind another contributes nothing to any
    pixel, so the renderer reports no evidence about it -- which is correct,
    and is exactly what a velocity-magnitude heuristic gets wrong.

  * 6.2 Responsibilities GATE, they do not duplicate. "The residual exists
    once per pixel." A pixel is attributed to the supernode that owns it, and
    attribution never turns one residual into several pseudo-measurements.

  * 6.3 The binary subproblem is solved exactly (graph cut), and the resulting
    support is accepted only if the TRUE nonlinear robust MAP objective
    decreases after a re-render. A smaller surrogate energy is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

EPS = 1e-12
STATIC, MOVING = 0, 1


@dataclass
class SupportState:
    """z_c and the evidence it came from."""

    probability: torch.Tensor        # [C] soft support probability in [0,1]
    hard: torch.Tensor               # [C] int64 in {0,1}
    unary_static: torch.Tensor       # [C] cost of labelling c STATIC
    unary_moving: torch.Tensor       # [C] cost of labelling c MOVING
    evidence: dict = field(default_factory=dict)

    @property
    def num_supernodes(self) -> int:
        return int(self.hard.shape[0])

    @property
    def num_moving(self) -> int:
        return int((self.hard == MOVING).sum())

    def gaussian_mask(self, supernode_id: torch.Tensor) -> torch.Tensor:
        """Expand z_c to a per-Gaussian boolean moving mask."""
        sid = supernode_id.long().clamp_min(0)
        mask = self.hard.to(sid.device)[sid] == MOVING
        return mask & (supernode_id.long() >= 0)

    def to_dict(self) -> dict:
        return {"num_supernodes": self.num_supernodes,
                "num_moving": self.num_moving,
                "num_static": self.num_supernodes - self.num_moving,
                "moving_fraction": self.num_moving / max(self.num_supernodes, 1),
                "mean_probability": float(self.probability.mean()),
                **self.evidence}


# ---------------------------------------------------------------------------
# 6.1 alpha-aware motion evidence
# ---------------------------------------------------------------------------
@torch.no_grad()
def joint_motion_sensitivity(linearization, grouped, joint_tangent: torch.Tensor
                             ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Section 6.1's a_{p,c}, summed over the supernode's members.

    ``joint_tangent`` is the per-Gaussian tangent d(mu, phi)/dq of the
    candidate joint -- for a revolute line that is ``revolute_joint_tangent``,
    for a prismatic axis it is a constant direction. Pushing it through the
    renderer's analytic JVP gives d(image)/dq, which IS the alpha-aware
    evidence: occluded or transparent supernodes produce no image change.
    """
    return linearization.apply_H(joint_tangent)


@torch.no_grad()
def supernode_motion_evidence(linearization, grouped, partition,
                              joint_tangent: torch.Tensor,
                              residuals, precisions) -> dict:
    """How much of each pixel residual the candidate joint motion explains,
    attributed back to supernodes through the adjoint.

    The attribution uses H^T, so it is the same operator the filter uses --
    a supernode gets credit for a residual exactly in proportion to how much
    moving it would change that pixel. Section 6.2: this is a GATE built from
    one residual per pixel, never a per-supernode copy of the residual.
    """
    dy = linearization.apply_H(joint_tangent)
    sid = partition.supernode_id.long().clamp_min(0)
    c = partition.num_supernodes

    # g_i = (H^T R^-1 r)_i -- the gradient of the measurement energy w.r.t.
    # moving Gaussian i. Contracted with the joint tangent t_i and summed over
    # the supernode's members this is
    #
    #     b_c = d/dq [ -E_meas ]  for moving ONLY supernode c along the joint
    #
    # which is the first-order evidence that c's motion explains the image.
    weighted = [(pc * rc, pd * rd)
                for (pc, pd), (rc, rd) in zip(precisions, residuals)]
    g = linearization.apply_HT(weighted)                     # [N,6]
    b_i = (g * joint_tangent).sum(dim=1)                     # [N]
    b_c = torch.zeros(c, device=b_i.device, dtype=b_i.dtype)
    b_c.index_add_(0, sid, b_i)

    # a_c = t^T H^T R^-1 H t restricted to c: the curvature, i.e. how strongly
    # the joint motion of c is VISIBLE. This is where alpha-awareness enters --
    # an occluded supernode changes no pixel, so a_c is ~0 and no amount of
    # residual can be attributed to it.
    w_dy = [(pc * a, pd * d) for (pc, pd), (a, d) in zip(precisions, dy)]
    h = linearization.apply_HT(w_dy)                         # [N,6]
    a_i = (h * joint_tangent).sum(dim=1)
    a_c = torch.zeros(c, device=a_i.device, dtype=a_i.dtype)
    a_c.index_add_(0, sid, a_i)

    visible = torch.zeros(c, device=b_c.device)
    visible.index_add_(0, sid, torch.ones(partition.num_gaussians,
                                          device=b_c.device))
    return {"alignment": b_c, "sensitivity": a_c.clamp_min(0.0),
            "visibility": visible, "image_tangent": dy,
            "total_alignment": float(b_c.sum()),
            "total_sensitivity": float(a_c.clamp_min(0.0).sum())}


# ---------------------------------------------------------------------------
# 6.2 depth-aware ownership responsibility (a gate, not a measurement)
# ---------------------------------------------------------------------------
@torch.no_grad()
def ownership_responsibility(linearization, grouped, partition,
                             min_sensitivity: float = 1e-8) -> torch.Tensor:
    """gamma_c in [0,1]: how much of the image a supernode is actually
    responsible for.

    Section 6.2 allows this to gate unreliable contributors and to diagnose
    occlusion, provided it never splits a pixel into several independent
    measurements. It is used here only as a multiplicative confidence on the
    unary evidence.
    """
    probe = torch.ones((partition.num_gaussians, 6), device=partition.centroids.device)
    dy = linearization.apply_H(probe)
    mag = [(a.abs(), d.abs()) for a, d in dy]
    share = grouped.reduce(linearization.apply_HT(mag)).abs().sum(dim=1)
    total = float(share.sum())
    if total <= min_sensitivity:
        return torch.zeros_like(share)
    return share / share.max().clamp_min(EPS)


# ---------------------------------------------------------------------------
# 6.3 step 4-5: Potts smoothness + exact binary graph cut
# ---------------------------------------------------------------------------
def graph_cut_binary(unary_static: torch.Tensor, unary_moving: torch.Tensor,
                     edges: torch.Tensor, potts_weight: float = 1.0
                     ) -> tuple[torch.Tensor, dict]:
    """Exact global minimum of a binary Potts energy by s-t min-cut.

        E(z) = sum_c U_c(z_c) + lambda * sum_{(a,b) in E} [z_a != z_b]

    With a non-negative Potts penalty the pairwise term is submodular, so the
    min-cut IS the global optimum -- which is why section 6.3 asks for a graph
    cut rather than an iterative relabeling that could stall in a local one.

    Source = MOVING, sink = STATIC. In the standard construction a node left on
    the SINK side has its source edge cut and takes the STATIC label, so that
    edge must carry the cost of being STATIC; symmetrically the sink edge
    carries the cost of being MOVING.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    c = int(unary_static.shape[0])
    us = unary_static.detach().cpu().double().numpy()
    um = unary_moving.detach().cpu().double().numpy()
    # Min-cut needs non-negative integer capacities; shift to >= 0 then scale.
    shift = float(min(us.min(), um.min(), 0.0))
    us, um = us - shift, um - shift
    lam = float(potts_weight)
    scale = 1000.0 / max(float(max(us.max(), um.max(), lam)), EPS)

    n = c + 2
    src, snk = c, c + 1
    rows, cols, data = [], [], []

    def add(u, v, cap):
        cap_i = int(round(cap * scale))
        if cap_i <= 0:
            return
        rows.append(u)
        cols.append(v)
        data.append(cap_i)

    for k in range(c):
        add(src, k, us[k])     # cut => k lands on the sink side => STATIC
        add(k, snk, um[k])     # cut => k stays on the source side => MOVING
    e = edges.detach().cpu().numpy() if edges.numel() else np.zeros((0, 2), int)
    for a, b in e:
        add(int(a), int(b), lam)
        add(int(b), int(a), lam)

    if not data:
        return (torch.zeros(c, dtype=torch.long),
                {"reason": "no finite capacities", "flow": 0.0})

    graph = csr_matrix((np.asarray(data, dtype=np.int32),
                        (np.asarray(rows), np.asarray(cols))), shape=(n, n))
    res = maximum_flow(graph, src, snk)

    # The source side of the residual graph is the MOVING set.
    flow = res.flow.toarray()
    cap = graph.toarray()
    residual = cap - flow
    seen = np.zeros(n, dtype=bool)
    stack = [src]
    seen[src] = True
    while stack:
        u = stack.pop()
        for v in np.nonzero(residual[u] > 0)[0]:
            if not seen[v]:
                seen[v] = True
                stack.append(int(v))
    labels = torch.from_numpy(seen[:c].astype(np.int64))
    return labels, {"flow_value": float(res.flow_value) / scale,
                    "reachable_from_source": int(seen[:c].sum()),
                    "potts_weight": lam, "capacity_scale": scale}


# ---------------------------------------------------------------------------
# 6.3 the alternating loop
# ---------------------------------------------------------------------------
@dataclass
class SupportConfig:
    # RELATIVE to the mean unary magnitude, not an absolute energy. The unaries
    # are true measurement energies, so their scale depends on the scene, the
    # image size and the number of cameras; an absolute Potts weight silently
    # changes meaning between scenes. It did exactly that here -- correcting the
    # VJP's contributor cutoff lowered the unary scale and a fixed weight of 1.0
    # went from negligible to overwhelming, smoothing every label into one
    # class. Expressed relative to the data term, "how much smoothing" means the
    # same thing everywhere.
    potts_weight: float = 1.0
    max_outer: int = 6
    evidence_temperature: float = 1.0
    min_moving_fraction: float = 0.0
    require_true_decrease: bool = True
    # The null hypothesis is STATIC. A supernode the renderer cannot see --
    # fully occluded at the linearization point, so a_c ~ 0 and b_c ~ 0 -- has
    # no evidence either way, and its unaries are both ~0. Without a prior the
    # graph cut would be free to call it moving on a numerical tie. Charging a
    # small cost for the MOVING label makes "it moves" the claim that has to be
    # paid for, which is the same discipline section 5.2 applies to weak
    # directions: do not invent motion where nothing was measured.
    moving_prior_fraction: float = 0.05


def _moving_prior(delta_e: torch.Tensor, fraction: float) -> float:
    """A prior in the same energy units as the evidence, scaled to the scene so
    it stays meaningful without being a tuned absolute constant.

    The scale is taken from the supernodes that actually FAVOUR moving, not from
    all of them. A static supernode that the joint motion would wreck carries an
    enormous positive delta_e (~1e4 on the offset-revolute test), so a prior
    scaled to the overall mean came out at 395 -- larger than the true moving
    supernodes' own evidence of 284 to 577, and it relabelled the entire moving
    part as static. The prior only has to outweigh numerical dust on supernodes
    with no evidence either way; sizing it against the moving population keeps it
    small next to real evidence while still breaking ties.
    """
    favouring = delta_e[delta_e < 0]
    if favouring.numel() == 0:
        return 0.0
    return float(fraction) * float(favouring.abs().median())


@torch.no_grad()
def achievable_motion_gain(linearization, grouped, precisions, residuals,
                           ridge: float = 1e-6):
    """Section 6.3 step 1: hypothesis-free, alpha-aware motion evidence.

    Before any joint hypothesis exists there is still a well-posed question per
    supernode: how much of the image residual could this supernode explain if it
    were free to move rigidly? That is the Gauss-Newton energy reduction

        Delta_c = 1/2 b_c^T F_c^-1 b_c,
        b_c = (G^T H^T R^-1 r)_c,   F_c = (G^T H^T R^-1 H G)_c

    computed from the exact analytic renderer sensitivity, so it is alpha-aware
    by construction: an occluded or transparent supernode changes no pixel, its
    F_c is ~0, and no amount of residual can be attributed to it.

    Why not per-supernode velocity magnitude, which is far cheaper. Measured on
    the real USB sequence, the filtered supernode speeds ran median 0.147 against
    a maximum of 1.11 -- no separation at all, because reconstruction and
    lighting residual make the filter attribute motion to the static background
    too. The same scene's achievable-gain distribution ran median 108 against a
    maximum of 6285. Section 6.3's "do not classify support from only
    per-Gaussian velocity magnitude" is not merely a stylistic preference.
    """
    weighted = [(p * r, q * d) for (p, q), (r, d) in zip(precisions, residuals)]
    b = grouped.reduce(linearization.apply_HT(weighted))        # [C,6]
    F = grouped.block_fisher(precisions)                        # [C,6,6]
    eye = torch.eye(6, device=F.device, dtype=F.dtype).expand_as(F)
    # Relative ridge: a supernode nothing can see has F ~ 0, and an absolute
    # epsilon there would manufacture a huge "achievable" gain out of noise.
    tr = F.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-12).reshape(-1, 1, 1)
    sol = torch.linalg.solve(F + ridge * tr / 6.0 * eye, b.unsqueeze(-1)).squeeze(-1)
    gain = (0.5 * (b * sol).sum(dim=1)).clamp_min(0.0)
    return gain, {"b": b, "fisher": F,
                  "median_gain": float(gain.median()),
                  "max_gain": float(gain.max())}


@torch.no_grad()
def initialize_support_from_evidence(gain: torch.Tensor, *, k_mad: float = 8.0,
                                     max_moving_fraction: float = 0.5
                                     ) -> tuple[torch.Tensor, dict]:
    """Threshold the achievable gain robustly into an initial z_c.

    Median + k*MAD rather than a mean/std rule: the static population is the
    majority and the moving tail is exactly the outlier we are looking for, so
    a non-robust statistic is dragged up by the thing it is meant to detect.
    """
    med = gain.median()
    mad = (gain - med).abs().median().clamp_min(1e-12)
    thr = med + float(k_mad) * mad
    hard = (gain > thr).long()
    frac = float(hard.float().mean())
    info = {"threshold": float(thr), "median": float(med), "mad": float(mad),
            "k_mad": float(k_mad), "num_moving": int(hard.sum()),
            "moving_fraction": frac, "capped": False}
    if frac > max_moving_fraction:
        # More than half the scene "moving" means the threshold found no
        # outlier population; keep only the strongest, and say so.
        keep = max(1, int(max_moving_fraction * gain.numel()))
        idx = gain.topk(keep).indices
        hard = torch.zeros_like(hard)
        hard[idx] = 1
        info.update({"capped": True, "num_moving": int(hard.sum()),
                     "moving_fraction": float(hard.float().mean()),
                     "note": (f"the median+{k_mad}MAD rule selected "
                              f"{frac:.0%} of supernodes, which is not an "
                              f"outlier population; capped to the strongest "
                              f"{max_moving_fraction:.0%}")})
    return hard, info


@torch.no_grad()
def estimate_joint_rate(b: torch.Tensor, a: torch.Tensor) -> tuple[float, dict]:
    """The joint's own rate q*, estimated so the static background cannot set it.

    The obvious estimator, q* = sum_c b_c / sum_c a_c, is the best SINGLE rate
    for the hypothesis "the whole scene moves together". On a scene that is
    mostly static that hypothesis is false and its answer is near zero: measured
    on the offset-revolute test it returned 8.8e-4 for a true q of 0.10, because
    the static supernodes contribute ~1e6 each to sum(a) and nothing to sum(b).
    With q* that small the quadratic penalty vanishes and well-lit static
    supernodes with a little residual alignment get labelled moving.

    What actually separates the two populations is the per-supernode implied
    rate q_c = b_c / a_c: on the same test the moving supernodes sat in
    [0.065, 0.158] around the true 0.10, while every static one was inside
    [-0.001, 0.003]. So q* is chosen as the common rate that maximizes the total
    ACHIEVABLE energy reduction,

        score(q) = sum_c max(0, q b_c - 1/2 q^2 a_c),

    counting each supernode only where moving at that rate actually helps. A
    static supernode contributes nothing at the moving group's q (its 1/2 q^2 a
    dwarfs its q b), so it cannot drag the estimate toward zero; the candidates
    are the q_c themselves, so no search range has to be guessed.
    """
    a_safe = a.clamp_min(EPS)
    q_c = b / a_safe
    cand = q_c[q_c > 0]
    if cand.numel() == 0:
        return 0.0, {"q_star": 0.0, "reason": "no supernode has positive alignment"}
    # score every candidate rate at once: [K,C] -> [K]
    q = cand.reshape(-1, 1)
    gain = (q * b.reshape(1, -1) - 0.5 * q * q * a.reshape(1, -1)).clamp_min(0.0)
    score = gain.sum(dim=1)
    k = int(score.argmax())
    q_star = float(cand[k])
    naive = float(b.sum() / a.sum().clamp_min(EPS))
    return q_star, {
        "q_star": q_star,
        "q_star_naive_pooled": naive,
        "num_candidates": int(cand.numel()),
        "best_score": float(score[k]),
        "num_supernodes_helped": int((gain[k] > 0).sum()),
    }


def unary_from_evidence(evidence: dict, responsibility: torch.Tensor | None = None,
                        temperature: float = 1.0, moving_prior_fraction: float = 0.0
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Section 6.3 step 3: unary costs from the joint-conditioned pixel
    likelihood, in TRUE ENERGY UNITS.

    With b_c = -dE/dq and a_c = d^2E/dq^2 restricted to supernode c, moving c
    by the joint's displacement q changes the measurement energy by

        dE_c(q) = -q b_c + 1/2 q^2 a_c.

    The joint's own q is shared, so it is estimated from the totals,
    q* = (sum_c b_c) / (sum_c a_c) -- the single scalar that best explains the
    whole image under the hypothesis. A supernode should then be labelled
    MOVING exactly when dE_c(q*) < 0, i.e. when moving it with the joint
    genuinely lowers the energy.

    Expressing the unaries as real energies is what makes the Potts weight
    meaningful: it is traded against the same units the data term is in, not
    against an arbitrary normalized score.
    """
    b = evidence["alignment"]
    a = evidence["sensitivity"].clamp_min(0.0)
    q_star, q_info = estimate_joint_rate(b, a)
    delta_e = -q_star * b + 0.5 * (q_star ** 2) * a
    if responsibility is not None:
        # Section 6.2: a supernode nothing can see has no evidence either way,
        # so its unaries are gated towards zero and the Potts term -- i.e. its
        # neighbours -- decides. This is the opposite of letting it hallucinate
        # an independent label.
        delta_e = delta_e * responsibility
    t = max(float(temperature), EPS)
    # cost of STATIC is the energy forgone by not moving; cost of MOVING is the
    # energy incurred by moving. Exactly one of them is non-zero.
    unary_static = torch.clamp(-delta_e, min=0.0) / t
    unary_moving = torch.clamp(delta_e, min=0.0) / t
    if moving_prior_fraction > 0.0:
        unary_moving = unary_moving + _moving_prior(delta_e, moving_prior_fraction) / t
    return unary_static, unary_moving


@torch.no_grad()
def infer_support(partition, evidence_fn, energy_fn, cfg: SupportConfig | None = None,
                  initial: torch.Tensor | None = None) -> tuple[SupportState, list[dict]]:
    """Section 6.3's alternating support/joint inference.

    ``evidence_fn(support_hard) -> (evidence dict, responsibility or None)``
        re-fits / re-conditions the joint with the support fixed and returns
        the motion evidence (step 2 and 3).
    ``energy_fn(support_hard) -> float``
        the TRUE nonlinear robust MAP objective after a re-render (step 6).

    A support update is accepted only when that true objective decreases
    (step 7), never because the surrogate graph-cut energy fell.
    """
    cfg = cfg or SupportConfig()
    c = partition.num_supernodes
    # The null hypothesis is that nothing moves. Starting from all-MOVING would
    # make "everything moves" the fallback whenever a proposal is rejected,
    # which is precisely the claim that needs evidence.
    hard = (torch.full((c,), STATIC, dtype=torch.long) if initial is None
            else initial.detach().cpu().long().clone())
    best_energy = energy_fn(hard)
    history = [{"outer": -1, "event": "initial", "energy": best_energy,
                "num_moving": int((hard == MOVING).sum()), "accepted": True}]

    state = SupportState(probability=hard.double(), hard=hard.clone(),
                         unary_static=torch.zeros(c), unary_moving=torch.zeros(c))

    for outer in range(int(cfg.max_outer)):
        evidence, responsibility = evidence_fn(hard)
        us, um = unary_from_evidence(evidence, responsibility,
                                     cfg.evidence_temperature,
                                     cfg.moving_prior_fraction)
        # Scale the Potts weight to the data term AND to the graph degree. Two
        # separate traps here, both measured on the offset-revolute test:
        #
        #  * degree. A supernode has ~knn neighbours, so a per-edge weight equal
        #    to a node's own evidence gives it a smoothing cost knn times that
        #    evidence and the labelling collapses to one class. Dividing by the
        #    mean degree makes `potts_weight` mean "per-node smoothing budget as
        #    a multiple of the per-node data term".
        #
        #  * which data scale. A static supernode that the joint motion would
        #    wreck carries a colossal unary (4.4e4 here) while a genuinely
        #    moving one carries a modest 3e2. Any mean is set by the static
        #    majority, and smoothing at that scale erases the moving part. The
        #    scale is therefore a LOW quantile of each node's own evidence
        #    strength: smoothing is allowed to override the weakest evidence in
        #    the scene, never the strongest.
        strength = torch.maximum(us, um)
        unary_scale = float(torch.quantile(strength.double(), 0.25))
        n_edges = int(partition.adjacency.shape[0])
        mean_degree = max(2.0 * n_edges / max(c, 1), 1.0)
        lam = cfg.potts_weight * unary_scale / mean_degree
        labels, cut_info = graph_cut_binary(us, um, partition.adjacency, lam)
        cut_info.update({"unary_scale": unary_scale,
                         "mean_degree": mean_degree,
                         "potts_weight_relative": cfg.potts_weight,
                         "potts_weight_absolute": lam})
        changed = int((labels != hard).sum())
        # Soft probability for the artifact: a calibrated read of the same
        # evidence, not a second independent estimate.
        margin = (um - us)
        prob = torch.sigmoid(margin / max(cfg.evidence_temperature, EPS))

        if changed == 0:
            history.append({"outer": outer, "event": "converged",
                            "energy": best_energy, "changed": 0,
                            "num_moving": int((hard == MOVING).sum()),
                            "accepted": True, **cut_info})
            state = SupportState(probability=prob, hard=hard.clone(),
                                 unary_static=us.cpu(), unary_moving=um.cpu(),
                                 evidence={"converged_at_outer": outer, **cut_info})
            break

        trial_energy = energy_fn(labels)
        accept = (trial_energy < best_energy) if cfg.require_true_decrease else True
        history.append({
            "outer": outer, "event": "proposal", "changed": changed,
            "energy_before": best_energy, "energy_after": trial_energy,
            "accepted": bool(accept),
            "num_moving_before": int((hard == MOVING).sum()),
            "num_moving_after": int((labels == MOVING).sum()),
            **cut_info})
        if accept:
            hard = labels
            best_energy = trial_energy
            state = SupportState(probability=prob, hard=hard.clone(),
                                 unary_static=us.cpu(), unary_moving=um.cpu(),
                                 evidence={"accepted_at_outer": outer, **cut_info})
        else:
            # Section 6.3 step 7: a proposal that does not reduce the true
            # objective is discarded, and the loop stops rather than cycling.
            history.append({"outer": outer, "event": "rejected",
                            "energy": best_energy, "accepted": False})
            break

    state.evidence.setdefault("final_energy", best_energy)
    state.evidence["outer_iterations"] = len(history)
    return state, history


# ---------------------------------------------------------------------------
# 6.4 metrics against ground truth
# ---------------------------------------------------------------------------
def support_metrics(predicted: torch.Tensor, truth: torch.Tensor) -> dict:
    """Section 6.4's precision / recall / F1 / IoU, with MOVING as positive."""
    p = predicted.detach().cpu().long().reshape(-1) == MOVING
    t = truth.detach().cpu().long().reshape(-1) == MOVING
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, EPS)
    iou = tp / max(tp + fp + fn, 1)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "true_negative": tn, "precision": prec, "recall": rec, "f1": f1,
            "iou": iou, "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "num_predicted_moving": tp + fp, "num_true_moving": tp + fn}


def gaussian_support_metrics(predicted_z: torch.Tensor, truth_moving_mask: torch.Tensor,
                             supernode_id: torch.Tensor) -> dict:
    """The same metrics at Gaussian granularity, which is what a downstream
    consumer of the support actually sees."""
    sid = supernode_id.long().clamp_min(0)
    pred = predicted_z.to(sid.device)[sid]
    out = support_metrics(pred, truth_moving_mask.long())
    return {f"gaussian_{k}": v for k, v in out.items()}
