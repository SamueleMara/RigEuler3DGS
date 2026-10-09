"""Sections 10, 12-19: the robust efficient-score test of H_P: beta = 0.

CODEX_NEXT_EFFICIENT_P_NULL_AND_PHASE2_DEBUG.md.

The question is section 10's:

> after allowing the best P direction and the whole q trajectory to explain
> everything they can, what RGB-D residual remains specifically along
> curvature?

Section 12 is the correction that matters. A model-based Fisher chi-square,

    g^T F^+ g,

assumes the whitened residual is independent across pixels. It is not: even
with `R_eff`, what is left is structured -- canonical-model mismatch,
calibration, shading -- and a statistic whose power grows with the pixel count
will reject ANY null once the residual has any structure at all. That is
exactly what was observed: Storage scored 4.29e+02 against a chi2(3) cut of
7.81 on a drawer. So the covariance of the score is estimated EMPIRICALLY from
camera/time blocks instead of being assumed.

No `1/sqrt(N)` over Gaussians or pixels appears anywhere here (section 14).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

CURVATURE_NOT_OBSERVABLE = "UNRESOLVED_GEOMETRY"
PRISMATIC_AT_RESOLUTION = "PRISMATIC_AT_RESOLUTION"
REVOLUTE_CANDIDATE = "REVOLUTE_CANDIDATE"
MODEL_MISMATCH = "MODEL_MISMATCH"


def _pinv(A: Tensor, rtol: float = 1e-10) -> tuple[Tensor, int, Tensor]:
    A = 0.5 * (A.double() + A.double().T)
    ev, V = torch.linalg.eigh(A)
    tol = rtol * float(ev.abs().max()) if ev.numel() else 0.0
    keep = ev > tol
    inv = (V[:, keep] * (1.0 / ev[keep])) @ V[:, keep].T if bool(keep.any()) \
        else torch.zeros_like(A)
    return inv, int(keep.sum()), ev


def schur_eliminate(scores: dict) -> dict:
    """Section 10: remove what the P nuisance can already explain."""
    F_aa, F_ab, F_bb = scores["F_aa"], scores["F_ab"], scores["F_bb"]
    g_a, g_b = scores["g_alpha"], scores["g_beta"]
    Faa_inv, rank_a, ev_a = _pinv(F_aa)
    A_ba = F_ab.T @ Faa_inv                       # [nb, na]
    F_cond = F_bb - A_ba @ F_ab
    g_cond = g_b - A_ba @ g_a
    inv_c, rank_c, ev_c = _pinv(F_cond)
    # Section 19: the null-stationarity audit. A large nuisance score means
    # the P null was not actually optimized, and beta cannot be interpreted
    # until it is.
    D_alpha = float(g_a @ (Faa_inv @ g_a))
    return {"A_beta_alpha": A_ba, "F_beta_given_alpha": F_cond,
            "g_beta_given_alpha": g_cond,
            "F_beta_given_alpha_pinv": inv_c,
            "rank_alpha": rank_a, "rank_beta_given_alpha": rank_c,
            "eigenvalues_alpha": ev_a.tolist(),
            "eigenvalues_beta_given_alpha": ev_c.tolist(),
            "D_alpha": D_alpha, "alpha_dim": int(F_aa.shape[0]),
            "fisher_chi2": float(g_cond @ (inv_c @ g_cond)),
            "fisher_note": ("section 12: the model-based chi-square is kept "
                            "as a diagnostic only; it assumes independent "
                            "pixels and therefore over-rejects")}


def block_efficient_scores(scores: dict, schur: dict) -> dict:
    """Section 13: u_b = s_beta,b - A_beta_alpha s_alpha,b, and their sum.

    The per-block NUISANCE scores are carried through too, so section 19's
    stationarity audit can be made with the same cluster-robust machinery as
    the curvature test rather than with a model-based chi-square.
    """
    A = schur["A_beta_alpha"]
    u, sa = [], []
    for b in scores["blocks"]:
        u.append(b["s_beta"] - A @ b["s_alpha"])
        sa.append(b["s_alpha"])
    U = torch.stack(u).sum(0) if u else torch.zeros(scores["beta_dim"],
                                                    dtype=torch.float64)
    return {"u_blocks": torch.stack(u) if u else torch.zeros(
        (0, scores["beta_dim"]), dtype=torch.float64),
            "alpha_blocks": torch.stack(sa) if sa else torch.zeros(
                (0, scores["alpha_dim"]), dtype=torch.float64),
            "U": U, "num_blocks": len(u),
            "consistency_with_g_beta_given_alpha":
                float((U - schur["g_beta_given_alpha"]).abs().max())}


def cluster_robust_covariance(blocks: dict) -> dict:
    """Section 14: B_U from the spread of the block scores themselves.

    One documented finite-cluster correction, applied globally: the usual
    `B/(B-1)` for an estimated mean. Nothing is divided by a pixel or Gaussian
    count (section 14 forbids it), because the unit of independent evidence
    here is the camera/time block, not the pixel.
    """
    u = blocks["u_blocks"]
    B = int(u.shape[0])
    if B < 2:
        return {"B_U": torch.zeros((u.shape[-1], u.shape[-1]),
                                   dtype=torch.float64),
                "num_blocks": B, "rank_B": 0,
                "singular_values_B": [],
                "reason": "at least two blocks are needed for a spread"}
    ubar = u.mean(0)
    d = u - ubar
    B_U = d.T @ d * (B / (B - 1.0))
    sv = torch.linalg.svdvals(B_U)
    tol = float(sv.max()) * len(sv) * torch.finfo(torch.float64).eps
    return {"B_U": B_U, "num_blocks": B,
            "rank_B": int((sv > tol).sum()),
            "singular_values_B": sv.tolist(),
            "finite_cluster_correction": "B/(B-1), applied globally"}


def curvature_statistic(scores: dict, schur: dict, blocks: dict,
                        robust: dict, *, confidence: float = 0.95,
                        null_converged: bool | None = None) -> dict:
    """Sections 15, 16, 18, 19: the robust statistic and the decision."""
    from joint_reduction.common_screw import chi2_quantile

    U = blocks["U"]
    B_U = robust["B_U"]
    nblocks = int(robust.get("num_blocks", 0))
    inv_B, rank_B, ev_B = _pinv(B_U)
    S = float(U @ (inv_B @ U))
    # Section 15: the observable rank is what BOTH the Fisher information and
    # the empirical block covariance support. A direction the data cannot see,
    # or one whose block spread is degenerate, is not a testable direction.
    rank = min(int(schur["rank_beta_given_alpha"]), int(rank_B))
    p = max(rank, 1)

    # Section 14's finite-cluster correction, applied globally.
    #
    # `B_U` is a covariance ESTIMATED from B blocks, so `U^T B_U^+ U` is a
    # Hotelling T^2, not a chi-square: referring it to chi2(p) ignores the
    # uncertainty in the covariance itself and over-rejects badly when B is
    # small. With 6 blocks and pure independent pixel noise -- a case with no
    # curvature whatsoever -- it scored 11.6 against a chi2(2) cut of 5.99 and
    # called exact prismatic data revolute.
    #
    # The standard correction is the F reference,
    #
    #     (B - p) / (p (B - 1)) * T^2  ~  F(p, B - p),
    #
    # which converges to chi2(p)/p as B grows, so one rule covers both
    # regimes and nothing is tuned per scene.
    if nblocks > p:
        scale = (nblocks - p) / (p * (nblocks - 1.0))
        stat = scale * S
        cut = _f_quantile(confidence, p, nblocks - p)
        reference = f"F({p}, {nblocks - p}) with the finite-cluster scaling"
    else:
        # Too few blocks to estimate the covariance at all.
        stat, cut, reference = S, float("inf"), (
            f"undefined: {nblocks} blocks cannot support {p} parameters")

    # Section 18: the one-step curvature estimate with its robust covariance.
    Fp = schur["F_beta_given_alpha_pinv"]
    beta_hat = Fp @ schur["g_beta_given_alpha"]
    Sigma_beta = Fp @ B_U @ Fp
    sig = torch.linalg.eigvalsh(0.5 * (Sigma_beta + Sigma_beta.T))

    # Section 19: beta may not be interpreted while the null is unconverged.
    #
    # The audit uses the SAME cluster-robust form as the curvature test. The
    # model-based version, `g_alpha^T F_alpha_alpha^+ g_alpha` against a
    # chi-square, has exactly the defect section 12 identifies: over thousands
    # of correlated pixels it never looks small. Measured on a synthetic
    # revolute scene it read 4.5e+02 against a cut of 14.07 and GREW to
    # 7.5e+03 when the null was fitted harder -- so it was reporting the
    # residual's structure, not the optimiser's progress, and blocked every
    # verdict. Referring the nuisance score to the spread of its own blocks
    # measures what section 19 actually asks: whether the fit still has
    # unexploited slack.
    # Which directions of alpha are actually replicated across blocks.
    #
    # alpha = [gamma_1, gamma_2, q_1 ... q_{T-1}], and a `q_t` coordinate is
    # nonzero only in the blocks belonging to frame t. The between-block
    # covariance is therefore structurally degenerate in those directions, and
    # a robust statistic built on all of alpha measures that degeneracy rather
    # than the optimiser's progress. Only the two gamma directions -- the
    # prismatic direction, which every block sees -- are replicated, so the
    # audit is restricted to them.
    sa_full = blocks.get("alpha_blocks")
    sa = None if sa_full is None else sa_full[:, :2]
    if sa is not None and int(sa.shape[0]) > 1:
        a_bar = sa.mean(0)
        da = sa - a_bar
        B_a = da.T @ da * (sa.shape[0] / (sa.shape[0] - 1.0))
        inv_a, rank_a_rob, _ev = _pinv(B_a)
        g_a_tot = sa.sum(0)
        D_alpha_robust = float(g_a_tot @ (inv_a @ g_a_tot))
        pa = max(int(rank_a_rob), 1)
        if nblocks > pa:
            alpha_stat = (nblocks - pa) / (pa * (nblocks - 1.0)) * D_alpha_robust
            alpha_cut = _f_quantile(confidence, pa, nblocks - pa)
        else:
            alpha_stat, alpha_cut = 0.0, float("inf")
    else:
        D_alpha_robust, alpha_stat, alpha_cut = 0.0, 0.0, float("inf")
    # Section 19 calls this "a convergence diagnostic, not a physical
    # classifier". When the caller knows whether the null fit converged -- it
    # watches the P direction stop moving across passes, which is the thing
    # section 19 actually describes -- that signal is authoritative and the
    # statistic is reported alongside it.
    audit_converged = alpha_stat <= alpha_cut
    null_ok = audit_converged if null_converged is None else bool(null_converged)

    if rank == 0:
        status = CURVATURE_NOT_OBSERVABLE
        why = ("section 16: no curvature direction is observable -- the "
               "conditional information or the block covariance is "
               "degenerate, so neither P nor R is supported")
    elif not null_ok:
        status = CURVATURE_NOT_OBSERVABLE
        why = (f"section 19: the P null is not stationary (robust nuisance "
               f"statistic {alpha_stat:.3e} against a cut of "
               f"{alpha_cut:.2f}), so the curvature score cannot be "
               f"interpreted yet")
    elif nblocks <= p:
        status = CURVATURE_NOT_OBSERVABLE
        why = (f"section 14: {nblocks} camera/time blocks cannot support a "
               f"{p}-parameter robust covariance, so no test is possible")
    elif stat > cut:
        status = REVOLUTE_CANDIDATE
        why = ("section 16: after the best P direction and the whole q "
               "trajectory have explained what they can, a curvature "
               "component remains that the block-to-block spread cannot "
               "account for")
    else:
        status = PRISMATIC_AT_RESOLUTION
        why = ("section 16: RGB-D does not require resolvable curvature once "
               "the best P nuisance is removed")

    return {"status": status, "reason": why,
            "S_curv": S, "F_statistic": stat, "rank_curv": rank, "cut": cut,
            "reference_distribution": reference, "num_blocks_used": nblocks,
            "confidence": confidence,
            "p_value": _f_sf(stat, p, max(nblocks - p, 1))
            if nblocks > p else 1.0,
            "U": U.tolist(), "B_U": B_U.tolist(),
            "B_U_eigenvalues": ev_B.tolist(),
            "rank_B": int(rank_B),
            "rank_fisher": int(schur["rank_beta_given_alpha"]),
            "beta_one_step": beta_hat.tolist(),
            "Sigma_beta": Sigma_beta.tolist(),
            "Sigma_beta_eigenvalues": sig.tolist(),
            "D_alpha_robust": D_alpha_robust,
            "D_alpha_statistic": alpha_stat, "D_alpha_cut": alpha_cut,
            "D_alpha_fisher": schur["D_alpha"],
            "null_converged": bool(null_ok),
            "nuisance_audit_converged": bool(audit_converged),
            "null_converged_source": ("null-fit direction convergence"
                                      if null_converged is not None
                                      else "robust nuisance audit"),
            "legacy_fisher_chi2": schur["fisher_chi2"],
            "num_blocks": blocks["num_blocks"],
            "score_consistency": blocks["consistency_with_g_beta_given_alpha"],
            "note": ("section 12: `legacy_fisher_chi2` is the model-based "
                     "statistic and is reported for comparison only; it does "
                     "not set this label")}


def _f_quantile(conf: float, d1: int, d2: int) -> float:
    try:
        from scipy.stats import f as _f
        return float(_f.ppf(conf, d1, d2))
    except Exception:                                      # pragma: no cover
        from joint_reduction.common_screw import chi2_quantile
        return chi2_quantile(d1, conf) / max(d1, 1)


def _f_sf(x: float, d1: int, d2: int) -> float:
    try:
        from scipy.stats import f as _f
        return float(_f.sf(max(x, 0.0), d1, d2))
    except Exception:                                      # pragma: no cover
        return _chi2_sf(max(x, 0.0) * d1, d1)


def _chi2_sf(x: float, k: int) -> float:
    try:
        from scipy.stats import chi2
        return float(chi2.sf(max(x, 0.0), k))
    except Exception:
        # Wilson-Hilferty, the same approximation used for the quantiles
        if x <= 0:
            return 1.0
        t = (x / k) ** (1.0 / 3.0)
        m = 1.0 - 2.0 / (9.0 * k)
        s = math.sqrt(2.0 / (9.0 * k))
        z = (t - m) / max(s, 1e-300)
        return 0.5 * math.erfc(z / math.sqrt(2.0))


def run_efficient_p_test(state, mu0, R0, observations, linearize,
                         precision_fn, *, block_frames: int = 5,
                         confidence: float = 0.95, part: int = 0,
                         null_converged: bool | None = None) -> dict:
    """The whole chain of sections 9-19, on an already-fitted P null."""
    from .joint_score_blocks import accumulate_scores

    sc = accumulate_scores(state, mu0, R0, observations, linearize,
                           precision_fn, block_frames=block_frames, part=part)
    schur = schur_eliminate(sc)
    blocks = block_efficient_scores(sc, schur)
    robust = cluster_robust_covariance(blocks)
    stat = curvature_statistic(sc, schur, blocks, robust,
                               confidence=confidence,
                               null_converged=null_converged)
    return {"scores": sc, "schur": schur, "blocks": blocks,
            "robust": robust, "test": stat}
