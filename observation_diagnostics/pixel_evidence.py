"""Per-pixel coverage, observation support and depth-layer agreement.

Definitions (also written into every run's provenance.json):

  covers_d(p)   alpha_d(p) >= alpha_cover          "model COVERS"
  e_d(p)        D_d(p) - D_gt(p), D_d = Z/(A+eps)  the production depth convention
  c_d(p)        RMS over channels of RGB_d(p) - RGB_gt(p)
  S_d^depth(p)  valid_gt(p) AND covers_d(p) AND |e_d(p)| <= depth_tol
                                                   "model EXPLAINS the depth"
  S_d^rgbd(p)   S_d^depth(p) AND c_d(p) <= rgb_tol

`valid_gt` is the dataset's own depth return (> 0). An invalid pixel is
GT_INVALID: never "missing geometry", never "unexplained" (T14).

These are DIAGNOSTIC predicates with declared, swept thresholds -- not
probabilities, and not rules for a future optimiser. The default scales are
the Phase-1 IEKF's own sensor model (estimator/sparse_iekf.IEKFConfig:
rgb_sigma = 0.1, constant depth sigma = 0.02 m, min_alpha_for_depth = 0.05),
with bands at 2 sigma.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

RGB_SIGMA = 0.1
DEPTH_SIGMA_M = 0.02
MIN_ALPHA_FOR_DEPTH = 0.05


@dataclass(frozen=True)
class Thresholds:
    alpha_cover: float = 0.5
    depth_tol_m: float = 2 * DEPTH_SIGMA_M
    rgb_tol: float = 2 * RGB_SIGMA
    # contributor-level: a Gaussian's view depth (its centre's z) against the
    # observed depth. A centre is not a surface point, so this is looser.
    contrib_depth_tol_m: float = 2 * DEPTH_SIGMA_M
    advantage_depth_m: float = DEPTH_SIGMA_M
    advantage_rgb: float = RGB_SIGMA
    low_visibility_alpha: float = MIN_ALPHA_FOR_DEPTH
    improvement_cap_m: float = 5 * DEPTH_SIGMA_M

    def as_dict(self) -> dict:
        d = asdict(self)
        d["provenance"] = ("estimator/sparse_iekf.IEKFConfig defaults used by "
                           "tools/run_motion_coherence_support.py (the historical "
                           "Phase 1): rgb_sigma 0.1, depth sigma 0.02 m constant, "
                           "min_alpha_for_depth 0.05; bands at 2 sigma")
        return d


SWEEP_ALPHA = (0.05, 0.2, 0.5, 0.9)
SWEEP_DEPTH_M = (0.01, 0.02, 0.04, 0.06, 0.1)
SWEEP_RGB = (0.05, 0.1, 0.2, 0.3)

# integer codes of the saved maps
COVER_CODES = {0: "NEITHER_COVERS", 1: "F_ONLY_COVER", 2: "B_ONLY_COVER", 3: "BOTH_COVER"}
SUPPORT_CODES = {0: "NEITHER_VALID_SUPPORT", 1: "F_SUPPORTED_B_UNEXPLAINED",
                 2: "B_SUPPORTED_F_UNEXPLAINED", 3: "BOTH_VALID_SUPPORT",
                 4: "GT_INVALID"}
LAYER_CODES = {0: "NOT_BOTH_COVER", 1: "BOTH_SAME_OBSERVED_SURFACE_CANDIDATE",
               2: "BOTH_COVER_DIFFERENT_DEPTH",
               3: "BOTH_COVER_AGREE_OFF_OBSERVED_SURFACE", 4: "BOTH_COVER_GT_INVALID"}


def _rms_rgb(a, b):
    return ((a - b) ** 2).mean(0).sqrt()


def classify(F: dict, B: dict, gt: dict, th: Thresholds) -> dict:
    """All per-pixel fields for one (frame, camera).

    F, B: {"rgb": [3,H,W], "depth": [H,W], "alpha": [H,W]} from the production
    renderer, each source rendered ALONE. gt: {"rgb", "depth", "valid"}.
    """
    valid = gt["valid"]
    aF, aB = F["alpha"], B["alpha"]
    covF, covB = aF >= th.alpha_cover, aB >= th.alpha_cover
    eF = torch.where(valid, F["depth"] - gt["depth"], torch.zeros_like(aF))
    eB = torch.where(valid, B["depth"] - gt["depth"], torch.zeros_like(aB))
    cF, cB = _rms_rgb(F["rgb"], gt["rgb"]), _rms_rgb(B["rgb"], gt["rgb"])
    SdF = valid & covF & (eF.abs() <= th.depth_tol_m)
    SdB = valid & covB & (eB.abs() <= th.depth_tol_m)
    SrF, SrB = SdF & (cF <= th.rgb_tol), SdB & (cB <= th.rgb_tol)
    both = covF & covB
    dFB = (F["depth"] - B["depth"]).abs()

    cover = covF.to(torch.uint8) + 2 * covB.to(torch.uint8)
    support = SdF.to(torch.uint8) + 2 * SdB.to(torch.uint8)
    support = torch.where(valid, support, torch.full_like(support, 4))
    layer = torch.zeros_like(cover)
    same_depth = dFB <= th.depth_tol_m
    on_obs = (eF.abs() <= th.depth_tol_m) & (eB.abs() <= th.depth_tol_m)
    layer[both & valid & same_depth & on_obs] = 1
    layer[both & ~same_depth] = 2          # valid or not: never a matched surface
    layer[both & valid & same_depth & ~on_obs] = 3
    layer[both & ~valid & same_depth] = 4

    # explanation error used for the continuous improvement: an uncovering
    # source gets the cap, so "B covers and fits, F has nothing" is maximal.
    cap = th.improvement_cap_m
    errF = torch.where(covF, eF.abs().clamp(max=cap), torch.full_like(eF, cap))
    errB = torch.where(covB, eB.abs().clamp(max=cap), torch.full_like(eB, cap))
    improve_B = torch.where(valid & covB, (errF - errB).clamp(min=0), torch.zeros_like(eF))
    improve_F = torch.where(valid & covF, (errB - errF).clamp(min=0), torch.zeros_like(eF))

    return {"valid": valid, "covF": covF, "covB": covB, "eF": eF, "eB": eB,
            "cF": cF, "cB": cB, "SdF": SdF, "SdB": SdB, "SrF": SrF, "SrB": SrB,
            "dFB": dFB, "cover": cover, "support": support, "layer": layer,
            "W_Bonly": SdB & ~SdF, "W_Fonly": SdF & ~SdB,
            "improve_B": improve_B, "improve_F": improve_F}


def _n(mask) -> int:
    return int(mask.sum())


def _q(x, mask):
    x = x[mask]
    if x.numel() == 0:
        return [None, None, None]
    qs = torch.quantile(x.float()[:1_000_000], torch.tensor([0.1, 0.5, 0.9], device=x.device))
    return [float(v) for v in qs]


def view_metrics(fields: dict, F: dict, B: dict, th: Thresholds) -> dict:
    """Counts with explicit denominators, residual summaries and edge cases."""
    v = fields["valid"]
    covF, covB = fields["covF"], fields["covB"]
    both = covF & covB
    eF, eB, cF, cB = fields["eF"], fields["eB"], fields["cF"], fields["cB"]
    HW = int(v.numel())
    nv = _n(v)
    out = {"pixels_total": HW, "gt_valid": nv, "gt_invalid": HW - nv}
    for code, name in COVER_CODES.items():
        out[name] = _n(fields["cover"] == code)
        out[name + "__on_gt_valid"] = _n((fields["cover"] == code) & v)
    for code, name in SUPPORT_CODES.items():
        out[name] = _n(fields["support"] == code)
    rgbd = fields["SrF"].to(torch.uint8) + 2 * fields["SrB"].to(torch.uint8)
    for code, name in ((0, "NEITHER_RGBD_SUPPORT"), (1, "F_RGBD_SUPPORTED_B_UNEXPLAINED"),
                       (2, "B_RGBD_SUPPORTED_F_UNEXPLAINED"), (3, "BOTH_RGBD_SUPPORT")):
        out[name] = _n((rgbd == code) & v)
    for code, name in LAYER_CODES.items():
        if code:
            out[name] = _n(fields["layer"] == code)
    d_m, r_m = th.advantage_depth_m, th.advantage_rgb
    bv = both & v
    out["F_DEPTH_ADVANTAGE"] = _n(bv & (eF.abs() + d_m < eB.abs()))
    out["B_DEPTH_ADVANTAGE"] = _n(bv & (eB.abs() + d_m < eF.abs()))
    depth_tie = bv & ((eF.abs() - eB.abs()).abs() <= d_m)
    out["F_COLOR_ONLY_ADVANTAGE"] = _n(depth_tie & (cF + r_m < cB))
    out["B_COLOR_ONLY_ADVANTAGE"] = _n(depth_tie & (cB + r_m < cF))
    t = th.depth_tol_m
    out["F_IN_FRONT_OF_OBS_B_CORRECT"] = _n(bv & (eF < -t) & (eB.abs() <= t))
    out["B_IN_FRONT_OF_OBS_F_CORRECT"] = _n(bv & (eB < -t) & (eF.abs() <= t))
    out["F_BEHIND_OBS_B_CORRECT"] = _n(bv & (eF > t) & (eB.abs() <= t))
    out["B_BEHIND_OBS_F_CORRECT"] = _n(bv & (eB > t) & (eF.abs() <= t))
    aF, aB = F["alpha"], B["alpha"]
    amax = torch.maximum(aF, aB)
    out["LOW_VISIBILITY"] = _n(v & (amax >= th.low_visibility_alpha) & (amax < th.alpha_cover))
    out["GT_INVALID_BUT_F_COVERS"] = _n(~v & covF)
    out["GT_INVALID_BUT_B_COVERS"] = _n(~v & covB)
    # residuals on the pixels each source covers, and on the production gate
    for key, e, c, cov, a in (("F", eF, cF, covF, aF), ("B", eB, cB, covB, aB)):
        m = v & cov
        out[f"{key}_depth_mae_m__covered"] = float(e[m].abs().mean()) if _n(m) else None
        out[f"{key}_depth_rmse_m__covered"] = float(e[m].pow(2).mean().sqrt()) if _n(m) else None
        out[f"{key}_rgb_rmse__covered"] = float(c[m].pow(2).mean().sqrt()) if _n(m) else None
        g = v & (a > MIN_ALPHA_FOR_DEPTH)
        out[f"{key}_depth_mae_m__production_gate"] = float(e[g].abs().mean()) if _n(g) else None
        out[f"{key}_rgb_rmse__all_valid"] = float(c[v].pow(2).mean().sqrt()) if nv else None
        out[f"{key}_alpha_sum"] = float(a.sum())
        out[f"{key}_depth_err_q10_q50_q90__covered"] = _q(e, m)
    out["dFB_q10_q50_q90__both_cover"] = _q(fields["dFB"], both)
    return out


def sweep_rows(F: dict, B: dict, gt: dict) -> list[dict]:
    """Category sizes across the declared alpha / depth / RGB sweeps (T11)."""
    rows = []
    for a in SWEEP_ALPHA:
        for d in SWEEP_DEPTH_M:
            th = Thresholds(alpha_cover=a, depth_tol_m=d)
            f = classify(F, B, gt, th)
            v = f["valid"]
            rows.append({"alpha_cover": a, "depth_tol_m": d, "rgb_tol": th.rgb_tol,
                         "BOTH_COVER": _n(f["covF"] & f["covB"] & v),
                         "F_ONLY_COVER": _n(f["covF"] & ~f["covB"] & v),
                         "B_ONLY_COVER": _n(~f["covF"] & f["covB"] & v),
                         "BOTH_VALID_SUPPORT": _n(f["support"] == 3),
                         "F_SUPPORTED_B_UNEXPLAINED": _n(f["support"] == 1),
                         "B_SUPPORTED_F_UNEXPLAINED": _n(f["support"] == 2),
                         "NEITHER_VALID_SUPPORT": _n(f["support"] == 0),
                         "SAME_SURFACE_CANDIDATE": _n(f["layer"] == 1),
                         "DIFFERENT_DEPTH": _n(f["layer"] == 2),
                         "gt_valid": _n(v)})
    base = Thresholds()
    for r in SWEEP_RGB:
        th = Thresholds(rgb_tol=r)
        f = classify(F, B, gt, th)
        v = f["valid"]
        rows.append({"alpha_cover": th.alpha_cover, "depth_tol_m": th.depth_tol_m,
                     "rgb_tol": r,
                     "BOTH_RGBD_SUPPORT": _n(f["SrF"] & f["SrB"]),
                     "F_RGBD_SUPPORTED_B_UNEXPLAINED": _n(f["SrF"] & ~f["SrB"]),
                     "B_RGBD_SUPPORTED_F_UNEXPLAINED": _n(f["SrB"] & ~f["SrF"]),
                     "gt_valid": _n(v)})
    del base
    return rows
