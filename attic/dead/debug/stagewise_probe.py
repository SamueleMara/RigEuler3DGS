"""Sections 6-42: per-stage numeric probes over a finished run.

CODEX_DEEP_STAGEWISE_DEBUG_AND_THEORY_AUDIT.md.

Each probe answers one of the mission questions for one stage and returns
both its numbers and a status from section 56's vocabulary. Nothing here
decides a joint type and nothing is tuned (section 1): the probes read what
the stages produced and say whether the theoretical invariant holds, is
low-signal, is unobservable, or was never computed.

Section 58 says to work from cached artifacts wherever that is mathematically
valid. Where a quantity genuinely cannot be reconstructed without re-running
the renderer, the probe records `NOT_AVAILABLE` together with the reason and
the command that would produce it, rather than inventing a number.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from estimator.manifold import project_to_so3, so3_log
from joint_reduction.common_screw import chi2_quantile
from joint_reduction.differential_finite_consistency import \
    incremental_spatial_omega
from joint_reduction.finite_relative_rotation import (
    centered_tracks, relative_vector_rotation, rotation_observability,
    weighted_kabsch_rotation)
from joint_reduction.rigidity_graph import pairwise_rigidity_error

# section 56 vocabulary
CONSISTENT = "CONSISTENT"
INCONSISTENT = "INCONSISTENT"
LOW_SIGNAL = "LOW_SIGNAL"
UNOBSERVABLE = "UNOBSERVABLE"
UNCALIBRATED = "UNCALIBRATED"
RANK_DEFICIENT = "RANK_DEFICIENT"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
NOT_WIRED = "NOT_WIRED"
DIFFERENT_STATISTIC = "DIFFERENT_STATISTIC"
NOT_AVAILABLE = "NOT_AVAILABLE"


def _f(v):
    if v is None:
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if math.isnan(x) else x


def _flat(m):
    return [float(x) for x in torch.as_tensor(m).double().reshape(-1)]


# ---------------------------------------------------------------------------
# section 6: Phase 0 canonical model
# ---------------------------------------------------------------------------
def probe_phase0(root: Path) -> dict:
    p = root / "canonical_gaussians.pt"
    if not p.is_file():
        return {"status": NOT_AVAILABLE,
                "reason": f"{p.name} absent; re-run with --debug-all",
                "artifacts": {}}
    can = torch.load(p, map_location="cpu", weights_only=False)
    mu = can["mu"].double()
    c = mu.mean(0)
    rad = (mu - c).norm(dim=1)
    out = {"num_gaussians": int(mu.shape[0]),
           "centroid": _flat(c),
           "extent_min": _flat(mu.min(0).values),
           "extent_max": _flat(mu.max(0).values),
           "rms_radius": float((rad ** 2).mean().sqrt()),
           "radius_p50": float(rad.median()),
           "radius_p99": float(rad.quantile(0.99)),
           "has_rotations": can.get("R") is not None,
           "has_scales": can.get("scales") is not None,
           "has_opacity": can.get("opacity") is not None,
           "non_finite_centres": int((~torch.isfinite(mu)).any(1).sum())}
    status = CONSISTENT if out["non_finite_centres"] == 0 else INCONSISTENT
    return {"status": status, "artifacts": out,
            "answers": "what the frozen canonical model contains"}


# ---------------------------------------------------------------------------
# sections 8-11: Phase 1 free motion
# ---------------------------------------------------------------------------
def probe_phase1_free_motion(ph1: dict, dt: float) -> dict:
    X, x0 = ph1["tracks"], ph1["x0"]
    disp = (X - X[0]).norm(dim=2)
    centre = X.mean(1)
    travel = (centre - centre[0]).norm(dim=1)
    L = float((x0 - x0.mean(0)).norm(dim=1).square().mean().sqrt())
    moved = (disp[-1] > 0.05 * L)
    out = {"num_supernodes": int(X.shape[1]), "num_frames": int(X.shape[0]),
           "part_radius_L": L, "dt": dt,
           "centre_travel_max": float(travel.max()),
           "centre_travel_over_L": float(travel.max()) / max(L, 1e-300),
           "per_node_displacement_p50": float(disp[-1].median()),
           "per_node_displacement_p99": float(disp[-1].quantile(0.99)),
           "nodes_moving_over_5pct_L": int(moved.sum()),
           "fraction_moving": float(moved.double().mean()),
           "stored_velocity_available": ph1["stored_velocity"] is not None}
    status = CONSISTENT
    if out["centre_travel_over_L"] < 0.1:
        status = LOW_SIGNAL
    return {"status": status, "artifacts": out,
            "answers": "which nodes Phase 1 moved and how far"}


def probe_velocity_estimators(ph1: dict, dt: float) -> dict:
    """Section 10: compare the temporal derivative estimators.

    The stored velocity column is a first-order backward difference of the
    same positions, so this quantifies how much the choice of estimator
    changes the angular conclusion before any of it is used.
    """
    from joint_reduction.velocity_gradient_rigidity import \
        instantaneous_velocity
    X = ph1["tracks"]
    central = instantaneous_velocity(X, dt=dt)["velocity"]
    back = torch.zeros_like(X)
    back[1:] = (X[1:] - X[:-1]) / dt
    back[0] = back[1]
    out = {"central_vs_backward_median_abs_diff":
               float((central - back).norm(dim=2).median()),
           "central_median_speed": float(central.norm(dim=2).median()),
           "backward_median_speed": float(back.norm(dim=2).median()),
           "relative_difference":
               float((central - back).norm(dim=2).median()
                     / central.norm(dim=2).median().clamp_min(1e-300))}
    if ph1["stored_velocity"] is not None:
        st = ph1["stored_velocity"]
        out["stored_vs_backward_max_abs"] = float((st - back).abs().max())
        out["stored_is_backward_difference"] = \
            bool(out["stored_vs_backward_max_abs"] < 1e-5)
    return {"status": CONSISTENT, "artifacts": out,
            "answers": "whether the derivative estimator changes the answer"}


def probe_temporal_covariance(ph1: dict, dt: float) -> dict:
    """Section 37: the cross-covariance the difference operator induces.

    Velocities at neighbouring frames share position samples, so they are NOT
    independent. The operator's own correlation is computable even when the
    Phase-1 position covariance is unknown, and reporting it is what stops a
    later chi-square from assuming T independent angular measurements.
    """
    from joint_reduction.velocity_gradient_rigidity import \
        velocity_weight_matrix
    T = int(ph1["tracks"].shape[0])
    W, rules = velocity_weight_matrix(
        torch.arange(T, dtype=torch.float64) * dt)
    # Sigma_v = W Sigma_x W^T; with Sigma_x = sigma^2 I the structure is W W^T
    G = W @ W.T
    d = G.diagonal().clamp_min(1e-300)
    corr = G / torch.sqrt(d[:, None] * d[None, :])
    off1 = corr.diagonal(1)
    off2 = corr.diagonal(2)
    return {"status": UNCALIBRATED, "artifacts": {
        "note": ("the OPERATOR-induced correlation only; the Phase-1 position "
                 "covariance is not persisted, so the reconstruction's own "
                 "correlations remain unknown"),
        "rules": list(dict.fromkeys(rules)),
        "lag1_correlation_median": float(off1.median()),
        "lag1_correlation_min": float(off1.min()),
        "lag2_correlation_median": float(off2.median()),
        "effective_independent_frames":
            float(T * (1 - float(off1.median())) / (1 + float(off1.median()))),
        "num_frames": T,
        "unknown_correlations": ("IEKF posterior cross-covariance between "
                                 "frames and between cameras")},
        "answers": "whether velocity estimates may be treated as independent"}


# ---------------------------------------------------------------------------
# section 17: Phase 2A finite rigidity graph
# ---------------------------------------------------------------------------
def probe_phase2_graph(ph1: dict, seed: int = 0, num_edges: int = 4000) -> dict:
    X = ph1["tracks"]
    n = int(X.shape[1])
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, n, (num_edges,), generator=g)
    j = torch.randint(0, n, (num_edges,), generator=g)
    keep = i != j
    edges = torch.stack([i[keep], j[keep]], dim=1)
    rho = pairwise_rigidity_error(X, edges)
    rho = rho["rho"] if isinstance(rho, dict) else rho
    rv = relative_vector_rotation(X, edges)
    out = {"num_edges": int(edges.shape[0]),
           "rho_finite_p50": float(rho.median()),
           "rho_finite_p90": float(rho.quantile(0.90)),
           "rho_finite_p99": float(rho.quantile(0.99)),
           "rho_finite_max": float(rho.max()),
           "relative_vector_rotation_p50_deg":
               float(rv["max_angle"].median()) * 57.2957795,
           "relative_vector_rotation_p90_deg":
               float(rv["max_angle"].quantile(0.90)) * 57.2957795,
           "length_drift_p50": float(rv["length_drift"].median())}
    # A rigid support preserves length; a revolute one turns the direction.
    status = CONSISTENT if out["rho_finite_p90"] < 0.05 else INCONSISTENT
    return {"status": status, "artifacts": out, "edges": edges,
            "rho": rho, "relative_rotation": rv,
            "answers": ("whether raw Phase-1 relative configuration preserves "
                        "distance, and whether its DIRECTION turns")}


# ---------------------------------------------------------------------------
# sections 21-22: the three-way angular comparison
# ---------------------------------------------------------------------------
def consecutive_phase1_rotation(ph1: dict, dt: float) -> dict:
    """Section 21 B: consecutive-frame Kabsch on RAW Phase-1 tracks.

    Centred independently at t and t+1, so this asks whether the raw tracks
    themselves rotate -- without Phase 2, and without the frame-0 reference
    that a cumulative Kabsch would impose.
    """
    X = ph1["tracks"]
    cen = centered_tracks(X)["centered"]
    T = int(X.shape[0])
    Rs, errs, obs = [], [], []
    n = int(X.shape[1])
    xscale = float(X.abs().max())
    floor = (float(torch.finfo(torch.float64).eps) * max(xscale, 1e-300)) ** 2
    covs = []
    for t in range(T - 1):
        k = weighted_kabsch_rotation(cen[t], cen[t + 1])
        Rs.append(k["R"])
        errs.append(k["kabsch_error"])
        o = rotation_observability(cen[t])
        sigma2 = max(k["kabsch_error"] / max(3 * n - 3, 1), floor)
        covs.append(o["rotation_covariance"] * sigma2 / (dt * dt))
        obs.append(o["least_observable_axis"])
    if not Rs:
        return {"omega": torch.zeros((0, 3), dtype=torch.float64),
                "covariance": torch.zeros((0, 3, 3), dtype=torch.float64),
                "kabsch_error": [], "least_observable": []}
    R = torch.stack(Rs)
    omega = so3_log(project_to_so3(R).detach()) / dt
    return {"omega": omega, "covariance": torch.stack(covs),
            "kabsch_error": errs, "least_observable": obs, "R": R}


THREE_WAY_HEADER = [
    "frame",
    "omega_diff_p1_x", "omega_diff_p1_y", "omega_diff_p1_z",
    "omega_finite_p1_x", "omega_finite_p1_y", "omega_finite_p1_z",
    "omega_p2_x", "omega_p2_y", "omega_p2_z",
    "norm_diff_p1", "norm_finite_p1", "norm_p2",
    "cos_diff_vs_finite_p1", "cos_diff_p1_vs_p2", "cos_finite_p1_vs_p2",
    "mahal_diff_vs_finite_p1", "mahal_diff_p1_vs_p2", "mahal_finite_p1_vs_p2",
    "diff_observable", "finite_p1_observable", "p2_covariance_available"]


def _cos(a, b):
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    ok = (na > 0) & (nb > 0)
    c = (a * b).sum(-1) / (na.clamp_min(1e-300) * nb.clamp_min(1e-300))
    return torch.where(ok, c, torch.full_like(c, float("nan")))


def _mahal(d, cov_a, cov_b):
    from joint_reduction.joint_information import information_inverse
    if cov_a is None and cov_b is None:
        return torch.full((d.shape[0],), float("nan"), dtype=torch.float64)
    tot = torch.zeros((d.shape[0], 3, 3), dtype=torch.float64)
    if cov_a is not None:
        tot = tot + cov_a[:d.shape[0]]
    if cov_b is not None:
        tot = tot + cov_b[:d.shape[0]]
    out = torch.zeros(d.shape[0], dtype=torch.float64)
    for k in range(d.shape[0]):
        inv = information_inverse(tot[k])["inverse"]
        out[k] = d[k] @ (inv @ d[k])
    return out


def three_way_angular(evidence: dict, ph1: dict, p2_omega, dt: float,
                      confidence: float = 0.95) -> dict:
    """Sections 21-23. Mahalanobis is left blank where no covariance exists."""
    diff = evidence["twist"]["omega"]
    diff_mid = 0.5 * (diff[1:] + diff[:-1])
    om_cov = torch.stack([f["omega_covariance"]
                          for f in evidence["twist"]["fits"]])
    diff_cov = 0.25 * (om_cov[1:] + om_cov[:-1])
    diff_info = evidence["omega_information"]
    diff_info_mid = 0.5 * (diff_info[1:] + diff_info[:-1])

    fin = consecutive_phase1_rotation(ph1, dt)
    fin_om, fin_cov = fin["omega"], fin["covariance"]

    T = min(diff_mid.shape[0], fin_om.shape[0],
            p2_omega.shape[0] if p2_omega is not None else 10 ** 9)
    diff_mid, diff_cov = diff_mid[:T], diff_cov[:T]
    diff_info_mid = diff_info_mid[:T]
    fin_om, fin_cov = fin_om[:T], fin_cov[:T]
    p2 = (p2_omega[:T] if p2_omega is not None
          else torch.full((T, 3), float("nan"), dtype=torch.float64))

    zero_cut = chi2_quantile(3, confidence)
    chi_diff = torch.einsum("ti,tij,tj->t", diff_mid, diff_info_mid, diff_mid)
    diff_obs = chi_diff > zero_cut
    # the finite-P1 estimate is observable when its own magnitude is
    # significant against its own covariance
    fin_obs = _mahal(fin_om, fin_cov, None) > zero_cut

    rows = []
    for t in range(T):
        rows.append([t] + _flat(diff_mid[t]) + _flat(fin_om[t]) + _flat(p2[t])
                    + [_f(diff_mid[t].norm()), _f(fin_om[t].norm()),
                       _f(p2[t].norm()),
                       _f(_cos(diff_mid[t], fin_om[t])),
                       _f(_cos(diff_mid[t], p2[t])),
                       _f(_cos(fin_om[t], p2[t])),
                       _f(_mahal(diff_mid[t:t+1], diff_cov[t:t+1],
                                 fin_cov[t:t+1] - fin_cov[t:t+1]
                                 + fin_cov[t:t+1])[0]
                          if True else None),
                       "",            # section 22: no Phase-2 covariance
                       "",            # exists, so no Mahalanobis is fabricated
                       bool(diff_obs[t]), bool(fin_obs[t]), False])
    # the one Mahalanobis we can honestly compute: diff-P1 vs finite-P1
    m_df = _mahal(diff_mid - fin_om, diff_cov, fin_cov)
    for t in range(T):
        rows[t][16] = _f(m_df[t])

    cos_df = _cos(diff_mid, fin_om)
    cos_dp = _cos(diff_mid, p2)
    cos_fp = _cos(fin_om, p2)
    summary = {
        "frames_compared": T,
        "median_norm_diff_p1": _f(diff_mid.norm(dim=1).median()),
        "median_norm_finite_p1": _f(fin_om.norm(dim=1).median()),
        "median_norm_p2": _f(p2.norm(dim=1).median()),
        "median_cos_diff_vs_finite_p1": _f(cos_df[~cos_df.isnan()].median()
                                           if bool((~cos_df.isnan()).any()) else None),
        "median_cos_diff_p1_vs_p2": _f(cos_dp[~cos_dp.isnan()].median()
                                       if bool((~cos_dp.isnan()).any()) else None),
        "median_cos_finite_p1_vs_p2": _f(cos_fp[~cos_fp.isnan()].median()
                                         if bool((~cos_fp.isnan()).any()) else None),
        "median_mahal_diff_vs_finite_p1": _f(m_df.median()),
        "fraction_diff_observable": float(diff_obs.double().mean()),
        "fraction_finite_p1_observable": float(fin_obs.double().mean()),
        "p2_covariance_available": False,
        "chi2_3dof_cut": zero_cut,
        "median_consecutive_kabsch_error":
            float(torch.tensor(fin["kabsch_error"]).median())
            if fin["kabsch_error"] else None,
    }

    # Section 23's interpretation, reported and never used as production
    # logic. The three findings are reported INDEPENDENTLY rather than as a
    # single first-match verdict: observability is a qualifier on a cosine,
    # not a replacement for it. Ordering observability first hid the most
    # informative statement a scene can make -- that the two Phase-1 routes
    # agree with each other and both disagree with Phase 2, which localises
    # the divergence to the rigid projection.
    d_obs = summary["fraction_diff_observable"] > 0.5
    f_obs = summary["fraction_finite_p1_observable"] > 0.5
    agree_df = (summary["median_mahal_diff_vs_finite_p1"] != ""
                and float(summary["median_mahal_diff_vs_finite_p1"]) <= zero_cut)

    def _absf(x):
        return abs(float(x)) if x not in ("", None) else None

    cos_dp2 = _absf(summary["median_cos_diff_p1_vs_p2"])
    cos_fp2 = _absf(summary["median_cos_finite_p1_vs_p2"])
    p2_present = p2_omega is not None and p2.shape[0] > 0
    # Phase 1 and Phase 2 part company when BOTH Phase-1 routes point
    # somewhere else than Phase 2 does.
    diverges_at_p2 = bool(p2_present and cos_dp2 is not None
                          and cos_fp2 is not None
                          and cos_dp2 < 0.5 and cos_fp2 < 0.5)

    findings = []
    if agree_df:
        findings.append("the two Phase-1 routes agree with each other")
    else:
        findings.append("diff-P1 disagrees with finite-P1, so the raw tracks, "
                        "the derivative or the support are internally "
                        "inconsistent")
    if p2_present:
        if diverges_at_p2:
            findings.append("and both disagree with Phase 2 -> PHASE 2 IS THE "
                            "FIRST PLACE ANGULAR MOTION CHANGES")
        elif agree_df:
            findings.append("and so does Phase 2 -> angular motion is present "
                            "before and after rigid projection")
    else:
        findings.append("no Phase-2 transform was available for comparison")
    if not d_obs and not f_obs:
        findings.append("CAVEAT: neither Phase-1 angular estimate is "
                        "individually distinguishable from zero, so the "
                        "agreement above is between two small quantities")
    elif not (d_obs and f_obs):
        findings.append("CAVEAT: only one of the two Phase-1 angular "
                        "estimates is observable")
    interp = "; ".join(findings)
    summary["interpretation"] = interp
    summary["phase1_routes_agree"] = bool(agree_df)
    summary["diverges_at_phase2"] = diverges_at_p2
    summary["both_phase1_observable"] = bool(d_obs and f_obs)
    # Section 24: a cosine between an unobservable pair means nothing.
    summary["axis_cosine_interpretable"] = bool(d_obs and f_obs)
    if not summary["axis_cosine_interpretable"]:
        summary["axis_cosine_note"] = "AXIS_COSINE_NOT_INTERPRETABLE"

    if not agree_df:
        status = INCONSISTENT
    elif diverges_at_p2:
        status = INCONSISTENT          # Phase 1 and Phase 2 do not agree
    elif not (d_obs or f_obs):
        status = UNOBSERVABLE
    else:
        status = CONSISTENT
    return {"status": status, "rows": rows, "summary": summary,
            "finite_p1": fin,
            "answers": ("whether the raw Phase-1 configuration rotates, and "
                        "whether Phase 2 changed the angular motion")}
