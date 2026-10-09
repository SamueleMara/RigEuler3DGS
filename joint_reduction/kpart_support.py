"""Known-K multi-label support: z_c in {0, 1, ..., K}.

CODEX_REQUIRED_MODIFICATIONS_MULTIVIEW_MULTIFRAME_KPART_IEKF.md sections 7, 10, 12.

Binary support answers "does this supernode move?". With K known moving rigid
parts the question is "which part does it belong to, if any?":

    z_c = 0   static
    z_c = k   moving rigid part k                              (section 7)

K is a CARDINALITY prior, never a label source. Section 7's forbidden estimator
inputs -- GT part labels, GT joint type, GT axis, GT q(t), GT moving masks --
appear nowhere in this module; the only thing it consumes is the integer K and
the analytic renderer evidence.

The energy is section 10's Potts model over the canonical supernode graph

    E(z) = sum_c U_c(z_c) + lambda sum_{(c,d) in E} w_cd [z_c != z_d]

minimised by ALPHA-EXPANSION, which for a Potts (metric) pairwise term has the
standard 2-approximation guarantee and in practice reaches the global optimum on
problems this size. Each expansion step is an exact binary graph cut, reusing
the same exact min-cut machinery as the binary solver.

Unaries come from section 12's alpha-aware achievable reduction

    Delta E_{c,k} ~= 1/2 b_{c,k}^T F_{c,k}^-1 b_{c,k}

evaluated under part k's current kinematic basis -- never from raw speed, which
section 12 forbids and which was measured not to separate moving from static on
real data (median 0.147 against max 1.11).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .support import graph_cut_binary

EPS = 1e-12
STATIC = 0


@dataclass
class KPartSupport:
    """z_c in {0..K} plus everything section 10 requires to be persisted."""

    hard: torch.Tensor                  # [C] int64 in {0..K}
    unary: torch.Tensor                 # [C, K+1]
    soft: torch.Tensor                  # [C, K+1] posterior-like scores
    K: int
    potts_weight: float = 0.0
    potts_energy: float = 0.0
    data_energy: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def num_supernodes(self) -> int:
        return int(self.hard.shape[0])

    @property
    def total_energy(self) -> float:
        return self.data_energy + self.potts_energy

    def counts(self) -> list[int]:
        return [int((self.hard == k).sum()) for k in range(self.K + 1)]

    def all_parts_present(self) -> bool:
        """Section 7's known-K constraint: every moving label must be used."""
        return all(int((self.hard == k).sum()) > 0 for k in range(1, self.K + 1))

    def gaussian_labels(self, supernode_id: torch.Tensor) -> torch.Tensor:
        sid = supernode_id.long().clamp_min(0)
        return self.hard.to(sid.device)[sid]

    def part_mask(self, supernode_id: torch.Tensor, k: int) -> torch.Tensor:
        return self.gaussian_labels(supernode_id) == int(k)

    def to_dict(self) -> dict:
        return {"K": self.K, "num_supernodes": self.num_supernodes,
                "hard_labels": self.hard.detach().cpu().tolist(),
                "unary": self.unary.detach().cpu().tolist(),
                "soft": self.soft.detach().cpu().tolist(),
                "label_counts": self.counts(),
                "all_parts_present": self.all_parts_present(),
                "potts_weight": self.potts_weight,
                "potts_energy": self.potts_energy,
                "data_energy": self.data_energy,
                "total_support_energy": self.total_energy, **self.info}


# ---------------------------------------------------------------------------
# section 10: alpha-expansion over the Potts energy
# ---------------------------------------------------------------------------
def potts_energy(labels: torch.Tensor, unary: torch.Tensor,
                 edges: torch.Tensor, weights: torch.Tensor,
                 lam: float) -> tuple[float, float]:
    """(data, smoothness) energy of a labelling."""
    idx = torch.arange(labels.numel(), device=unary.device)
    data = float(unary[idx, labels.to(unary.device)].sum())
    if edges.numel() == 0:
        return data, 0.0
    a, b = edges[:, 0], edges[:, 1]
    diff = (labels[a] != labels[b]).to(unary.dtype)
    return data, float(lam * (weights * diff).sum())


def _expansion_move(unary: torch.Tensor, labels: torch.Tensor, alpha: int,
                    edges: torch.Tensor, weights: torch.Tensor,
                    lam: float) -> torch.Tensor:
    """One exact alpha-expansion move (Boykov-Veksler-Zabih).

    Binary variable per node: x_c = 1 means "take alpha", x_c = 0 means "keep
    z_c". With the Potts pairwise term the move energy is

        V(0,0) = w[z_a != z_b]   V(0,1) = w[z_a != alpha]
        V(1,0) = w[alpha != z_b] V(1,1) = 0

    which is submodular (A + D <= B + C for Potts), so the move is solved
    EXACTLY by one s-t min-cut. The standard reparameterisation

        A + (C-A)x_a + (D-C)x_b + (B+C-A-D)[x_a=0][x_b=1]

    turns the pairwise part into node terms plus a single DIRECTED edge of
    non-negative capacity, which is why this builds the graph itself instead of
    reusing the undirected binary helper -- that helper can only express a
    symmetric Potts edge, and the three distinct V values here are not
    symmetric once z_a and z_b differ.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    n_nodes = unary.shape[0]
    dev = unary.device
    u = unary.double().cpu().numpy()
    z = labels.cpu().numpy()

    # theta_c(0) = cost of KEEPING z_c, theta_c(1) = cost of TAKING alpha.
    # A node already labelled alpha needs no special case: U_c(z_c) IS U_c(alpha)
    # there, so both sides cost the same and either assignment means alpha.
    th0 = u[np.arange(n_nodes), z].copy()
    th1 = u[:, alpha].copy()

    directed = []
    if edges.numel():
        e = edges.cpu().numpy()
        w = (lam * weights).double().cpu().numpy()
        za, zb = z[e[:, 0]], z[e[:, 1]]
        A = w * (za != zb)          # V(keep, keep)
        B = w * (za != alpha)       # V(keep_a, alpha_b)
        Cc = w * (alpha != zb)      # V(alpha_a, keep_b)
        D = np.zeros_like(A)        # V(alpha, alpha)
        # E(x_a,x_b) = A + (C-A) x_a + (D-C) x_b + (B+C-A-D) [x_a=0][x_b=1]
        np.add.at(th1, e[:, 0], Cc - A)
        np.add.at(th1, e[:, 1], D - Cc)
        cap = B + Cc - A - D        # >= 0 by submodularity of a Potts term
        # The residual term costs when x_a = 0 (sink side) and x_b = 1 (source
        # side). An edge is cut when its TAIL is on the source side and its head
        # on the sink side, so this is the edge b -> a, not a -> b.
        directed = [(int(b), int(a), float(c))
                    for a, b, c in zip(e[:, 0], e[:, 1], cap) if c > 0]

    # Per-node constant shift so both capacities are non-negative; subtracting
    # the same constant from theta(0) and theta(1) cannot change the argmin.
    m = np.minimum(th0, th1)
    th0, th1 = th0 - m, th1 - m

    allcap = np.concatenate([th0, th1,
                             np.array([c for _, _, c in directed] or [0.0])])
    big = float(np.max(allcap)) + 1.0
    scale = 1e6 / max(big, EPS)

    n = n_nodes + 2
    src, snk = n_nodes, n_nodes + 1
    rows, cols, data = [], [], []

    def add(a, b, c):
        ci = int(round(c * scale))
        if ci > 0:
            rows.append(a); cols.append(b); data.append(ci)

    # Source side means x = 1 ("take alpha"). The source edge is cut when the
    # node ends on the sink side, i.e. x = 0, so it carries theta(0).
    for c in range(n_nodes):
        add(src, c, th0[c])
        add(c, snk, th1[c])
    for a, b, c in directed:
        add(a, b, c)

    if not data:
        return labels.clone()
    g = csr_matrix((np.asarray(data, dtype=np.int32),
                    (np.asarray(rows), np.asarray(cols))), shape=(n, n))
    res = maximum_flow(g, src, snk)
    residual = g.toarray() - res.flow.toarray()
    seen = np.zeros(n, dtype=bool)
    stack = [src]
    seen[src] = True
    while stack:
        q = stack.pop()
        for v in np.nonzero(residual[q] > 0)[0]:
            if not seen[v]:
                seen[v] = True
                stack.append(int(v))
    out = labels.clone()
    out[torch.from_numpy(seen[:n_nodes]).to(dev)] = alpha
    return out


def alpha_expansion(unary: torch.Tensor, edges: torch.Tensor,
                    weights: torch.Tensor | None = None, *,
                    lam: float = 1.0, max_cycles: int = 8,
                    initial: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, dict]:
    """Section 10's preferred solver.

    Cycles over every label, applying the exact expansion move above and
    keeping it only when the TRUE multi-label energy decreases. For a Potts
    (metric) pairwise term this is the standard 2-approximation; on graphs this
    size it reaches the global optimum, which the test suite checks against
    brute force.

    Why not solve the multi-label problem directly: it is NP-hard in general.
    Why not iterated conditional modes: it is a local method that stalls on
    exactly the label-swap configurations this problem produces.
    """
    C, L = unary.shape
    dev = unary.device
    if weights is None:
        weights = torch.ones(max(edges.shape[0], 1), device=dev, dtype=unary.dtype)
    labels = (torch.zeros(C, dtype=torch.long, device=dev) if initial is None
              else initial.clone().to(dev).long())

    d, s = potts_energy(labels, unary, edges, weights, lam)
    best = d + s
    history = [{"cycle": -1, "label": None, "data": d, "smooth": s,
                "total": best, "changed": 0}]

    for cycle in range(int(max_cycles)):
        improved = False
        for alpha in range(L):
            trial = _expansion_move(unary, labels, alpha, edges, weights, lam)
            td, ts = potts_energy(trial, unary, edges, weights, lam)
            if td + ts < best - 1e-12:
                changed = int((trial != labels).sum())
                labels, best = trial, td + ts
                improved = True
                history.append({"cycle": cycle, "label": alpha, "data": td,
                                "smooth": ts, "total": best, "changed": changed})
        if not improved:
            break

    d, s = potts_energy(labels, unary, edges, weights, lam)
    return labels, {"data_energy": d, "smoothness_energy": s, "total": d + s,
                    "cycles_run": len(history), "history": history,
                    "solver": "alpha_expansion"}


# ---------------------------------------------------------------------------
# section 12: per-part alpha-aware unaries
# ---------------------------------------------------------------------------
@torch.no_grad()
def part_score_statistic(linearization, grouped, precisions, residuals,
                         part_tangent: torch.Tensor | None,
                         prior_information: float = 0.0):
    """Section 5.1's support statistic.

        b_c   = H_c^T R_eff^-1 r
        F_c   = H_c^T R_eff^-1 H_c
        T_c   = b_c^T (F_c + Lambda_c)^-1 b_c                    (section 5.1)

    and, for a candidate part k, the same with the part-constrained Jacobian
    H_{c,k} = H_c G_{c,k}:

        T_{c,k} = b_{c,k}^T (F_{c,k} + Lambda_{c,k})^-1 b_{c,k}

    This is a SCORE statistic in R_eff-whitened units, so under the static null
    r ~ N(0, R_eff) it is asymptotically chi-square with the dimension of the
    local identifiable subspace (section 6 Option A). That is what makes a
    dataset-independent significance level meaningful; no empirical "null
    frame" is involved (section 5).

    ``prior_information`` is Lambda_c. It is a prior/damping information, not a
    tuned threshold: it bounds the implied step and keeps T_c finite where the
    part model barely moves the supernode.

    Returns ``(T, df, info)``. ``info["testable"]`` marks the supernodes whose
    own Fisher dominates Lambda_c -- i.e. where the DATA, not the prior, decides
    the statistic. A supernode the part model cannot move has F_c ~ 0 and is
    not evidence either way; with Lambda_c = 0 its b^2/F diverges (measured: a
    null-only scene produced a mean of 8.3e13 against a chi-square mean of 1,
    while the MEDIAN was a correct 0.595). Reporting it as "moving" would be
    reading noise through a singular normalisation, so it is excluded from the
    test rather than thresholded.

    Where F_c >> Lambda_c the statistic follows the chi-square law of section 6;
    where Lambda_c shrinks it, the statistic is stochastically smaller, so the
    test is CONSERVATIVE there, never anti-conservative.
    """
    weighted = [(p * r, q * d) for (p, q), (r, d) in zip(precisions, residuals)]
    g = linearization.apply_HT(weighted)                       # [N,6]

    if part_tangent is None:
        # unconstrained 6-DoF local score: d = 6
        b = grouped.reduce(g)                                  # [C,6]
        F = grouped.block_fisher(precisions)                   # [C,6,6]
        eye = torch.eye(6, device=F.device, dtype=F.dtype).expand_as(F)
        lam = float(prior_information)
        sol = torch.linalg.solve(F + lam * eye, b.unsqueeze(-1)).squeeze(-1)
        T = (b * sol).sum(dim=1).clamp_min(0.0)
        scale = F.diagonal(dim1=1, dim2=2).sum(-1) / 6.0
        testable = scale > max(lam, EPS)
        return T, 6, {"fisher_scale": scale, "testable": testable,
                      "lambda": lam}

    # part-constrained: the local subspace is 1-D (the joint coordinate), d = 1
    sid = grouped.part.supernode_id.long().clamp_min(0)
    C = grouped.num_supernodes
    b_i = (g * part_tangent).sum(dim=1)
    b_c = torch.zeros(C, device=b_i.device, dtype=b_i.dtype).index_add_(0, sid, b_i)
    dy = linearization.apply_H(part_tangent)
    w_dy = [(p * a, q * d) for (p, q), (a, d) in zip(precisions, dy)]
    h = linearization.apply_HT(w_dy)
    a_i = (h * part_tangent).sum(dim=1)
    a_c = torch.zeros(C, device=a_i.device, dtype=a_i.dtype).index_add_(0, sid, a_i)
    lam = float(prior_information)
    T = (b_c * b_c / (a_c.clamp_min(0.0) + lam).clamp_min(EPS)).clamp_min(0.0)
    testable = a_c > max(lam, EPS)
    return T, 1, {"fisher_scale": a_c, "testable": testable, "lambda": lam}


def chi2_quantile(p: float, df: int) -> float:
    """Upper quantile of chi-square with `df` degrees of freedom.

    Exact via SciPy. The Wilson-Hilferty cube-root approximation was tried
    first and rejected on measurement: at df = 1 it is 11% high at p = 0.99999,
    and that is exactly the regime a Bonferroni correction over ~176 supernodes
    lands in (alpha/test = 5.7e-6). An 11% error in the decision threshold
    would make section 29 A's "false-positive rate should match the chosen
    significance" untestable. The fallback is kept only so the module imports
    without SciPy; it is never silently used without saying so.
    """
    try:
        from scipy.stats import chi2 as _chi2
        return float(_chi2.ppf(p, df))
    except Exception:                                   # pragma: no cover
        import math
        # Wilson-Hilferty, accurate to ~1% near the centre, worse in the tail
        z = math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)
        k = float(df)
        return k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3


def _erfinv(x: float) -> float:                         # pragma: no cover
    import math
    a = 0.147
    ln = math.log(1.0 - x * x) if abs(x) < 1 else -745.0
    t = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(max(math.sqrt(t * t - ln / a) - t, 0.0)), x)


def support_null_threshold(df: int, alpha: float = 1e-3,
                           num_tests: int = 1, bonferroni: bool = True
                           ) -> tuple[float, dict]:
    """Section 6 Option A: the null comes from the stochastic model, not data.

    Under r ~ N(0, R_eff) the score statistic of section 5.1 is approximately
    chi-square with `df` degrees of freedom, so the decision threshold at
    significance alpha is its upper quantile. Testing every supernode means
    `num_tests` simultaneous tests; a Bonferroni correction keeps the
    FAMILY-WISE false-positive rate at alpha, which is what "false-positive
    rate should match the chosen statistical significance" (section 29 A) asks
    for.

    Nothing here depends on the dataset: alpha and df are the only inputs.
    """
    a = alpha / max(num_tests, 1) if bonferroni else alpha
    a = min(max(a, 1e-12), 0.5)
    thr = chi2_quantile(1.0 - a, df)
    return thr, {"null_model": "chi2 from R_eff (section 6 Option A)",
                 "alpha": alpha, "alpha_per_test": a, "df": df,
                 "num_tests": num_tests, "bonferroni": bonferroni,
                 "threshold": thr,
                 "note": ("dataset-independent: derived from the stated "
                          "stochastic model, not from any observed frame")}


@torch.no_grad()
def monte_carlo_null_threshold(linearization, grouped, precisions,
                               part_tangent, *, samples: int = 32,
                               quantile: float = 0.999,
                               prior_information: float = 0.0,
                               generator=None) -> tuple[float, dict]:
    """Section 6 Option B: empirical null by sampling eps ~ N(0, R_eff).

    Provided as the cross-check for Option A's asymptotic chi-square: the same
    statistic is recomputed on synthetic residuals drawn from the stated model,
    so any departure from the asymptotic law shows up directly.
    """
    vals = []
    for _ in range(int(samples)):
        synth = []
        for (pc, pd) in precisions:
            # eps ~ N(0, R_eff)  <=>  eps = sigma * n  with sigma = 1/sqrt(prec)
            sc = torch.rsqrt(pc.clamp_min(EPS))
            sd = torch.rsqrt(pd.clamp_min(EPS))
            nc = torch.randn(pc.shape, device=pc.device, generator=generator)
            nd = torch.randn(pd.shape, device=pd.device, generator=generator)
            synth.append((sc * nc * (pc > 0), sd * nd * (pd > 0)))
        t, _df, info = part_score_statistic(linearization, grouped, precisions,
                                            synth, part_tangent,
                                            prior_information=prior_information)
        vals.append(t[info["testable"]] if bool(info["testable"].any()) else t)
    stack = torch.stack(vals)
    thr = float(torch.quantile(stack.reshape(-1).double(), float(quantile)))
    return thr, {"null_model": "Monte Carlo from R_eff (section 6 Option B)",
                 "samples": int(samples), "quantile": quantile,
                 "threshold": thr,
                 "empirical_mean": float(stack.mean()),
                 "empirical_max": float(stack.max())}


@torch.no_grad()
def part_achievable_gain(linearization, grouped, precisions, residuals,
                         part_tangent: torch.Tensor | None,
                         ridge: float = 1e-6, q_prior_scale: float = 1.0):
    """DIAGNOSTIC ONLY (kept for the historical comparison in the report).

    This is the earlier 1/2 b^T F^-1 b form with a q-prior ridge. Production
    support now uses `part_score_statistic` with the section 6 null; this
    remains so the two can be compared in diagnostics.
    """
    t, _df, _info = part_score_statistic(
        linearization, grouped, precisions, residuals, part_tangent,
        prior_information=1.0 / max(q_prior_scale ** 2, EPS))
    return 0.5 * t


@torch.no_grad()
def baseline_excess_gain(gain_far: torch.Tensor, gain_near: torch.Tensor,
                         *, baseline_ratio: float = 1.0) -> torch.Tensor:
    """INITIALISER / DIAGNOSTIC ONLY (section 8).

        x_c = max(0, T_c(far) - r * T_c(near))

    Section 8: "A far-minus-near contrast is initializer/debug only." Final
    support must use the whole window (`T_c^window = sum_{t,camera} T_c`), which
    is what the production path accumulates. This remains because the contrast
    is still a useful initialiser and because the report compares the two.
    """
    return (gain_far - float(baseline_ratio) * gain_near).clamp_min(0.0)


@torch.no_grad()
def calibrate_null_excess(gain_a: torch.Tensor, gain_b: torch.Tensor,
                          quantile: float = 0.99) -> tuple[float, dict]:
    """DIAGNOSTIC ONLY (section 5). NOT a production threshold.

    Section 5 removes this from production: it treats a frame pair as a
    no-motion reference, which is invalid whenever the object already moves
    between those frames -- and the estimator has no way to know that it does
    not. The production null comes from the stated stochastic model instead
    (`support_null_threshold`, section 6).

    Kept so the two can be compared in the report, and so section 29 E can
    assert that production does not consult it.
    """
    null = (gain_b - gain_a).clamp_min(0.0)
    thr = float(torch.quantile(null.double(), float(quantile)))
    return thr, {"null_quantile": float(quantile), "null_threshold": thr,
                 "null_median": float(null.median()),
                 "null_max": float(null.max()),
                 "production_use": False,
                 "why_not_used": ("section 5: frames 0 and 1 are not a "
                                  "guaranteed no-motion pair")}


@torch.no_grad()
def build_unaries(gains: list[torch.Tensor], *, static_prior: float = 0.0,
                  temperature: float = 1.0) -> torch.Tensor:
    """Section 10's `[num_supernodes, K+1]` unary matrix.

    ``gains[k-1]`` is Delta E_{c,k} for moving part k: the energy a supernode
    could REMOVE by moving under part k. A unary is a COST, so

        U_c(0) = max_j Delta E_{c,j}                   energy forgone by staying static
        U_c(k) = max_j Delta E_{c,j} - Delta E_{c,k}   + moving prior

    Only differences matter to the cut, and every entry is non-negative, which
    the min-cut construction requires.

    The prior is charged to the MOVING labels, not to static. "This supernode
    moves" is the claim that needs evidence; pricing static instead inverts the
    null hypothesis and labels the whole scene moving -- measured on the K=2
    synthetic scene, that produced label counts [0, 1, 18] against a truth of
    [6, 6, 7], i.e. no static supernodes at all.
    """
    G = torch.stack(gains, dim=1)                              # [C,K]
    best = G.max(dim=1).values                                 # [C]
    K = G.shape[1]
    unary = torch.zeros((G.shape[0], K + 1), device=G.device, dtype=G.dtype)
    unary[:, 0] = best
    unary[:, 1:] = best.unsqueeze(1) - G + static_prior
    return unary / max(float(temperature), EPS)


def soft_scores(unary: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Section 10's required soft score for labels 0..K: a Gibbs posterior over
    the same unaries, so the soft and hard outputs cannot disagree about which
    label the data prefers."""
    return torch.softmax(-unary / max(float(temperature), EPS), dim=1)


@torch.no_grad()
def infer_kpart_support(partition, gains: list[torch.Tensor], *,
                        K: int, potts_weight: float = 1.0,
                        static_prior_fraction: float = 0.01,
                        temperature: float = 1.0,
                        initial: torch.Tensor | None = None,
                        enforce_all_parts: bool = True) -> KPartSupport:
    """One multi-label support solve (sections 7 and 10).

    ``enforce_all_parts`` implements section 7's known-K constraint: if the cut
    leaves a moving label empty, the K parts were not all used and the labelling
    is repaired by assigning that label its strongest claimant. Reporting an
    empty part would silently turn a known-K problem into a known-(K-1) one.
    """
    if len(gains) != K:
        raise ValueError(f"expected {K} per-part gain vectors, got {len(gains)}")
    G = torch.stack(gains, dim=1)
    # The prior charged to the moving labels is scaled to a LOW quantile of the
    # positive evidence, not its median. Its only job is to break ties on
    # supernodes with NO evidence either way -- the occluded case -- so it must
    # sit below the weakest real evidence in the scene. Measured on the K=2
    # scene the positive gains spanned 0.55 to 306 with a median of 110; a prior
    # of 5% of the median (5.5) buried the two weakest genuine movers and cost
    # 10 points of accuracy (0.947 -> 0.842), while 1% of the 25th percentile
    # (0.29) sits under all of them.
    favouring = G.max(dim=1).values
    pos = favouring[favouring > 0]
    scale = float(torch.quantile(pos.double(), 0.25)) if pos.numel() else 0.0
    unary = build_unaries(gains, static_prior=static_prior_fraction * scale,
                          temperature=temperature)

    edges = partition.adjacency
    w = torch.ones(edges.shape[0], device=unary.device, dtype=unary.dtype)
    # Potts weight relative to a LOW quantile of per-node evidence strength and
    # divided by the mean degree: smoothing may override the weakest evidence
    # in the scene, never the strongest, and a node with knn neighbours must not
    # pay knn times its own evidence.
    strength = unary.max(dim=1).values - unary.min(dim=1).values
    q25 = float(torch.quantile(strength.double(), 0.25)) if strength.numel() else 0.0
    degree = max(2.0 * edges.shape[0] / max(partition.num_supernodes, 1), 1.0)
    lam = potts_weight * q25 / degree

    labels, info = alpha_expansion(unary, edges, w, lam=lam, initial=initial)

    repaired = []
    if enforce_all_parts:
        for k in range(1, K + 1):
            if int((labels == k).sum()) == 0:
                # strongest claimant for the empty label, excluding nodes that
                # are the sole member of another moving label
                order = torch.argsort(G[:, k - 1], descending=True)
                for c in order.tolist():
                    cur = int(labels[c])
                    if cur == 0 or int((labels == cur).sum()) > 1:
                        labels[c] = k
                        repaired.append({"label": k, "supernode": c,
                                         "gain": float(G[c, k - 1])})
                        break
    d, s = potts_energy(labels, unary, edges, w, lam)
    return KPartSupport(
        hard=labels.cpu(), unary=unary.cpu(), soft=soft_scores(unary).cpu(),
        K=K, potts_weight=lam, potts_energy=s, data_energy=d,
        info={**info, "potts_weight_relative": potts_weight,
              "unary_scale_q25": q25, "mean_degree": degree,
              "empty_labels_repaired": repaired})


# ---------------------------------------------------------------------------
# section 23: post-hoc metrics only (never estimator input)
# ---------------------------------------------------------------------------
def kpart_metrics(predicted: torch.Tensor, truth: torch.Tensor,
                  K: int) -> dict:
    """Hungarian-matched accuracy, ARI, and per-label P/R/F1/IoU.

    Section 23 permits GT part labels ONLY here. The matching is required
    because the estimator has no way to know which of its labels corresponds to
    which GT part -- only that there are K of them.
    """
    from scipy.optimize import linear_sum_assignment

    p = predicted.detach().cpu().long().reshape(-1).numpy()
    t = truth.detach().cpu().long().reshape(-1).numpy()
    L = K + 1
    conf = np.zeros((L, L), dtype=np.int64)
    for a, b in zip(p, t):
        if 0 <= a < L and 0 <= b < L:
            conf[a, b] += 1
    # Static (0) is not permutable: it means the same thing in both labellings.
    sub = conf[1:, 1:]
    row, col = linear_sum_assignment(-sub)
    perm = {0: 0}
    for r, c in zip(row, col):
        perm[int(r) + 1] = int(c) + 1
    matched = np.array([perm.get(int(x), 0) for x in p])
    acc = float((matched == t).mean()) if t.size else 0.0

    per_label = {}
    for k in range(L):
        tp = int(((matched == k) & (t == k)).sum())
        fp = int(((matched == k) & (t != k)).sum())
        fn = int(((matched != k) & (t == k)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per_label[k] = {"precision": prec, "recall": rec,
                        "f1": 2 * prec * rec / max(prec + rec, EPS),
                        "iou": tp / max(tp + fp + fn, 1),
                        "support": int((t == k).sum())}
    return {"hungarian_accuracy": acc, "permutation": perm,
            "adjusted_rand_index": _ari(matched, t),
            "per_label": per_label,
            "confusion": conf.tolist(),
            "note": "GT labels used for evaluation only (section 23)"}


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand Index, computed from the contingency table."""
    if a.size == 0:
        return 0.0
    la, lb = np.unique(a), np.unique(b)
    cont = np.zeros((la.size, lb.size), dtype=np.float64)
    ia = {v: i for i, v in enumerate(la)}
    ib = {v: i for i, v in enumerate(lb)}
    for x, y in zip(a, b):
        cont[ia[x], ib[y]] += 1
    n = cont.sum()
    comb = lambda x: x * (x - 1) / 2.0            # noqa: E731
    sum_ij = comb(cont).sum()
    sum_i = comb(cont.sum(axis=1)).sum()
    sum_j = comb(cont.sum(axis=0)).sum()
    total = comb(n)
    exp = sum_i * sum_j / max(total, EPS)
    mx = 0.5 * (sum_i + sum_j)
    return float((sum_ij - exp) / max(mx - exp, EPS))
