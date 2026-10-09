"""Section 8: how big must a P-versus-R gain be before it means anything?

CODEX_DIRECT_GAUSSIAN_JOINT_RGBD_RELIABILITY.md section 8.

The floor this replaces was `epsilon_geom = (sigma_rigid / (L sqrt(N)))^2`,
which treats each of the N supernodes as an independent measurement of the
part's rotation. They are not. Every supernode in a part is fitted through the
same images, the same calibration and the same canonical Gaussian model, so a
systematic error in any of those moves all of them together. Dividing by
sqrt(N) then shrinks the floor by a factor that was never earned -- with N in
the thousands it reports a resolution far finer than the reconstruction has,
and any tiny gain clears it.

What this module measures instead is how much the answer MOVES when the
evidence changes in a way that leaves the physics alone: refit Phase 2 on
independent camera blocks and watch the fitted transforms, and the resulting
P/R gain, disagree with each other.

Two separate quantities come out, and section 8 is explicit that they are not
interchangeable:

  * transform repeatability `D_bc` -- how far apart two folds place the same
    part, in the same units the P/R objective uses;
  * `epsilon_geom` -- a floor on the GAIN `E_P - E_R`, calibrated from how
    much that gain itself fluctuates across folds.

A spread in transform position does not convert into a spread in the gain,
because the gain is a difference of two fits that both move when the
transforms move. Only the second may be compared with a measured gain.
"""

from __future__ import annotations

import math

import torch

from .gaussian_transport_metric import canonical_moments, psd_sqrt


def transform_discrepancy(T_b, T_c, moments, *, include_frame_zero=False):
    """Section 8's D_bc, per frame, via the same exact moment reduction.

        D_bc,t^2 = (1 / (W L^2)) sum_i w_i || T_t^(b) mu_i - T_t^(c) mu_i ||^2

    Expanding mu_i = c + (mu_i - c) kills the cross term (the weighted centred
    sum is zero), leaving a centre part and a moment part -- the same split the
    P/R objective uses, so these numbers are directly comparable with it.

    T_b, T_c are [T,4,4].
    """
    sl = slice(0, None) if include_frame_zero else slice(1, None)
    Rb, pb = T_b[sl, :3, :3].double(), T_b[sl, :3, 3].double()
    Rc, pc = T_c[sl, :3, :3].double(), T_c[sl, :3, 3].double()
    c = moments["c"].reshape(3).double()
    S = moments["S"].double()
    L2 = max(float(moments["L"]) ** 2, 1e-300)

    db = torch.einsum("tij,j->ti", Rb, c) + pb - c
    dc = torch.einsum("tij,j->ti", Rc, c) + pc - c
    trans = ((db - dc) ** 2).sum(1) / L2

    dR = Rb - Rc
    rot = torch.einsum("tij,jk,tik->t", dR, S, dR) / L2

    return {"D_bc_per_frame": (trans + rot).sqrt(),
            "translation_part_per_frame": trans.sqrt(),
            "rotation_part_per_frame": rot.sqrt(),
            "D_bc_rms": float((trans + rot).mean().sqrt()),
            "D_bc_max": float((trans + rot).max().sqrt()),
            "translation_rms": float(trans.mean().sqrt()),
            "rotation_rms": float(rot.mean().sqrt())}


def effective_independent_folds(values) -> dict:
    """How many of these folds are actually independent?

    Section 8: "do not claim more effective samples than supported." Folds
    share most of their cameras, so their errors are correlated and the naive
    count overstates the evidence. The lag-1 autocorrelation of the ordered
    values gives the usual variance-inflation correction

        n_eff = n (1 - rho) / (1 + rho),

    which is reported alongside n, never silently in place of it.
    """
    v = torch.as_tensor(list(values), dtype=torch.float64).reshape(-1)
    n = int(v.numel())
    if n < 3:
        return {"n_folds": n, "rho_lag1": None, "n_effective": float(n)}
    x = v - v.mean()
    denom = float((x * x).sum())
    rho = float((x[:-1] * x[1:]).sum() / denom) if denom > 0 else 0.0
    rho = max(min(rho, 0.99), -0.99)
    n_eff = n * (1.0 - rho) / (1.0 + rho)
    return {"n_folds": n, "rho_lag1": rho,
            "n_effective": float(max(1.0, min(float(n), n_eff)))}


def gain_floor_from_folds(fold_gains, *, epsilon_opt: float = 0.0) -> dict:
    """Section 8: a floor on `E_P - E_R`, from how much that gain itself moves.

    `fold_gains` holds the signed gain measured on each independent camera-fold
    refit. The floor is the spread of those gains, NOT the spread of the
    transforms -- section 8 is explicit that a transform-position variance "is
    not automatically the same as a floor on the gain".

    The floor is reported as the fold-to-fold standard deviation of the gain
    carried onto the mean, inflated by the correlation between folds, plus the
    separately measured optimization error. Nothing here looks at ground truth
    and nothing is tuned per scene.
    """
    g = torch.as_tensor(list(fold_gains), dtype=torch.float64).reshape(-1)
    n = int(g.numel())
    if n == 0:
        return {"status": "NO_FOLDS", "epsilon_geom": None}
    mean = float(g.mean())
    if n == 1:
        return {"status": "INSUFFICIENT_FOLDS", "epsilon_geom": None,
                "mean_gain": mean, "n_folds": 1,
                "reason": "section 8 needs at least two independent camera "
                          "blocks before any spread can be measured"}
    sd = float(g.std(unbiased=True))
    eff = effective_independent_folds(g)
    se = sd / math.sqrt(max(eff["n_effective"], 1.0))
    eps_geom = se
    floor = eps_geom + float(epsilon_opt)
    # The signs must agree too: folds that disagree on WHICH model is better
    # are not measuring a joint type, whatever the mean says.
    pos = int((g > 0).sum())
    return {"status": "OK",
            "mean_gain": mean, "median_gain": float(g.median()),
            "sd_gain": sd, "min_gain": float(g.min()),
            "max_gain": float(g.max()),
            "folds_favouring_revolute": pos,
            "folds_favouring_prismatic": n - pos,
            "sign_is_unanimous": bool(pos == n or pos == 0),
            **eff,
            "epsilon_geom": eps_geom,
            "epsilon_opt": float(epsilon_opt),
            "floor": floor,
            "gain_over_floor": (mean / floor) if floor > 0 else float("inf"),
            "derivation": "section 8: standard error of the per-fold signed "
                          "gain across independent camera blocks, corrected "
                          "for lag-1 correlation between folds; no 1/sqrt(N) "
                          "over supernodes or Gaussians anywhere"}


def optimization_error(fit_fn, n_starts: int = 4) -> dict:
    """Section 8's `epsilon_opt`: how much the SOLVER alone moves the answer.

    `fit_fn(seed)` must return an energy. Restarting the same fit from
    different initialisations and taking the spread of the results separates
    solver noise from evidence: a gain smaller than this is a property of the
    optimizer, not of the scene.
    """
    vals = [float(fit_fn(k)) for k in range(int(n_starts))]
    t = torch.tensor(vals, dtype=torch.float64)
    return {"energies": vals, "best": float(t.min()),
            "spread": float(t.max() - t.min()),
            "epsilon_opt": float(t.max() - t.min()),
            "n_starts": int(n_starts)}
