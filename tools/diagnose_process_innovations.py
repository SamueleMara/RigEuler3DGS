#!/usr/bin/env python
"""Whitened process innovations of a free-motion trajectory, done correctly.

Section 2 of CODEX_PRIORITY_REAL_BIDIRECTIONAL_PHASE1_MERGE_AND_VISUALIZE.md.

An earlier report compared the largest frame-to-frame CENTRE DISPLACEMENT
(0.0697 m) with sqrt(Q[mu,mu]) (3.96e-4 m) and called the ratio a "176 sigma
process mismatch". That is not a process statistic. The constant-twist model
predicts

    mu_pred   = mu_t + dt v_t          R_pred = Exp(dt hat(omega_t)) R_t
    v_pred    = v_t                    omega_pred = omega_t

and the innovation is the residual against that PREDICTION, in the tested
12-D local coordinates of `bidirectional_iekf.process_factors.process_residual`:

    e_mu = mu_{t+1} - mu_t - dt v_t       e_R = Log(R_{t+1} R_pred^T)
    e_v  = v_{t+1} - v_t                  e_w = omega_{t+1} - omega_t

The consistency statistic is chi2 = e' Q^-1 e with the Q the solver actually
assembles (`estimator.covariance.make_process_noise`, cross terms included).
Under the model chi2 ~ chi2_12 (mean 12) and each marginal block statistic
e_b' Q_bb^-1 e_b ~ chi2_3 (mean 3). This tool reports their DISTRIBUTIONS,
per frame and overall, beside the invalid displacement ratio so the two can
be compared.

Two input kinds, never confused:

  --full-state DIR    a bidirectional_iekf.free_state export: the filter's own
                      mu, R, v, omega                        -> FULL_12D_FILTER
  --centers DIR       phase1_centers/: centres only. R, v and omega are
                      rebuilt exactly as the single-source CLI rebuilds them
                      (R constant, central-difference v, omega = 0)
                                                             -> CENTER_ONLY_APPROXIMATE

Reads only; writes a JSON and an HTML into --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PCTS = (50.0, 90.0, 99.0, 99.9, 100.0)
BLOCKS = {"mu": slice(0, 3), "phi": slice(3, 6), "v": slice(6, 9),
          "omega": slice(9, 12)}


def _pct(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {f"p{p:g}": float(np.percentile(x, p)) for p in PCTS} | {
        "mean": float(x.mean())}


def _tail_share(chi2: torch.Tensor) -> dict:
    """Which fraction of E_process the largest terms carry.

    E_process is a SUM over (Gaussian, pair) terms, so a handful of large
    ones can dominate it while the typical Gaussian sits at one sigma. That
    distinction is the whole difference between "Q is mis-specified" and
    "a small set of tracks jumped".
    """
    flat = chi2.reshape(-1).double()
    total = float(flat.sum())
    order = torch.sort(flat, descending=True).values
    out = {}
    for frac in (0.001, 0.01, 0.05):
        k = max(1, int(round(frac * flat.numel())))
        out[f"top_{frac * 100:g}pct_of_terms"] = float(order[:k].sum()) / max(total, 1e-300)
    return out


def load_states(args):
    """(mu, R, v, omega) as [T,N,...] float64 CPU, plus a provenance dict."""
    from estimator.state import quaternion_wxyz_to_matrix

    if args.full_state:
        d = Path(args.full_state)
        m = json.loads((d / "full_state_manifest.json").read_text())
        ids = np.load(d / "original_frame_ids.npy").tolist()
        mu = torch.from_numpy(np.load(d / "mu_original_order.npy")).double()
        q = torch.from_numpy(np.load(d / "quat_wxyz_original_order.npy")).double()
        R = quaternion_wxyz_to_matrix(q.reshape(-1, 4)).reshape(*q.shape[:2], 3, 3)
        v = torch.from_numpy(np.load(d / "v_physical_original_order.npy")).double()
        w = torch.from_numpy(np.load(d / "omega_physical_original_order.npy")).double()
        dt = float(args.dt if args.dt else m["dt_seconds"])
        return mu, R, v, w, ids, dt, {
            "mode": "FULL_12D_FILTER", "source": str(d),
            "direction": m["direction"], "endpoint": m["canonical_endpoint"],
            "rate_convention": m["rate_convention"],
            "dt_is_measured": m.get("dt_is_measured", False)}

    d = Path(args.centers)
    m = json.loads((d / "trajectory_manifest.json").read_text())
    if m.get("trajectory_order") != "original_chronological":
        raise SystemExit(f"{d}: trajectory_order={m.get('trajectory_order')!r}")
    files = sorted(d.glob("centers_frame_*.npy"))
    ids = [int(f.stem.split("_")[-1]) for f in files]
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise SystemExit(f"{d}: frames are not contiguous: {ids[:5]}...")
    mu = torch.stack([torch.from_numpy(np.load(f)).double() for f in files])
    dt = float(args.dt if args.dt else 1.0 / 30.0)
    T, N = mu.shape[:2]
    v = torch.empty_like(mu)
    v[0] = (mu[1] - mu[0]) / dt
    v[-1] = (mu[-1] - mu[-2]) / dt
    v[1:-1] = (mu[2:] - mu[:-2]) / (2 * dt)
    # R constant and omega = 0 make e_R and e_omega identically zero, so the
    # actual rotation values cannot affect any number below; identity is used
    # rather than reading the PLY, and that is recorded.
    R = torch.eye(3, dtype=torch.float64).expand(T, N, 3, 3).contiguous()
    w = torch.zeros_like(mu)
    return mu, R, v, w, ids, dt, {
        "mode": "CENTER_ONLY_APPROXIMATE", "source": str(d),
        "direction": m["direction"], "endpoint": m["canonical_endpoint"],
        "rate_convention": "central differences of chronological centres",
        "rotation_note": ("R constant and omega = 0, exactly as the "
                          "single-source CLI rebuilds them, so e_R = e_omega "
                          "= 0 by construction"),
        "dt_is_measured": False}


def innovations(mu, R, v, w, dt, sigmas, chunk=8192):
    """Per pair, per Gaussian: total chi2 and marginal block chi2."""
    from bidirectional_iekf.process_factors import process_residual
    from estimator.covariance import make_process_noise
    from estimator.state import GaussianMotionState

    T, N = mu.shape[:2]
    Q = make_process_noise(1, dt, *sigmas, dtype=torch.float64)[0]   # [12,12]
    Q_inv = torch.linalg.inv(Q)
    Qb_inv = {k: torch.linalg.inv(Q[s, s]) for k, s in BLOCKS.items()}
    chi2 = torch.empty(T - 1, N, dtype=torch.float64)
    block = {k: torch.empty(T - 1, N, dtype=torch.float64) for k in BLOCKS}
    norm = {k: torch.empty(T - 1, N, dtype=torch.float64) for k in BLOCKS}
    app = None
    for t in range(T - 1):
        for s in range(0, N, chunk):
            e_ = slice(s, min(s + chunk, N))
            n = e_.stop - e_.start
            if app is None or app[0].shape[0] != n:
                app = (torch.zeros(n, 3), torch.full((n, 1), 0.5),
                       torch.full((n, 3), 0.01))

            def st(k):
                return GaussianMotionState(mu=mu[k, e_], R=R[k, e_], v=v[k, e_],
                                           omega=w[k, e_], colors=app[0],
                                           opacity=app[1], scales=app[2])
            e = process_residual(st(t), st(t + 1), dt)            # [n,12]
            chi2[t, e_] = torch.einsum("ni,ij,nj->n", e, Q_inv, e)
            for k, sl in BLOCKS.items():
                eb = e[:, sl]
                block[k][t, e_] = torch.einsum("ni,ij,nj->n", eb, Qb_inv[k], eb)
                norm[k][t, e_] = eb.norm(dim=1)
    return Q, chi2, block, norm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--full-state")
    src.add_argument("--centers")
    ap.add_argument("--label", required=True, help="e.g. forward / backward")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=None,
                    help="override the declared step; v and Q are then "
                         "consistent with it because v is re-read in the "
                         "export's own units only for --centers")
    args = ap.parse_args()

    from estimator.sparse_iekf import IEKFConfig
    cfg = IEKFConfig()
    sigmas = (cfg.process_noise_pos, cfg.process_noise_rot,
              cfg.process_noise_vel, cfg.process_noise_omega)

    mu, R, v, w, ids, dt, prov = load_states(args)
    T, N = mu.shape[:2]
    Q, chi2, block, norm = innovations(mu, R, v, w, dt, sigmas)

    # The statistic the earlier report used, for direct comparison.
    disp = (mu[1:] - mu[:-1]).norm(dim=2)
    invalid = disp / float(Q[0, 0].sqrt())

    # Time reversal checks on the same chronological state.
    flip = lambda x: x.flip(0).contiguous()  # noqa: E731
    _, chi2_rev, _, _ = innovations(flip(mu), flip(R), -flip(v), -flip(w), dt, sigmas)
    _, chi2_naive, _, _ = innovations(flip(mu), flip(R), flip(v), flip(w), dt, sigmas)

    sd = Q.diagonal().sqrt()
    corr_mu_v = float(Q[0, 6] / (sd[0] * sd[6]))
    result = {
        "label": args.label, **prov,
        "num_frames": T, "num_gaussians": N,
        "original_frame_ids": ids,
        "dt_seconds": dt,
        "dt_semantics": ("DECLARED convention: job_config_resolved.yaml records "
                         "fps: null and duration_sec: null; the static "
                         "training's fps=30 is a reconstruction convention, "
                         "not a measurement. Rates are 'per declared second'. "
                         "Rescaling time by k scales v by 1/k and changes Q "
                         "through dt; the tool recomputes both together."),
        "Q": {
            "built_by": "estimator.covariance.make_process_noise",
            "densities_per_sqrt_s": dict(zip(("pos", "rot", "vel", "omega"),
                                             sigmas)),
            "diagonal_sd": {k: float(sd[s.start]) for k, s in BLOCKS.items()},
            "corr_mu_v": corr_mu_v,
            "corr_phi_omega": float(Q[3, 9] / (sd[3] * sd[9])),
            "eigenvalues": torch.linalg.eigvalsh(Q).tolist(),
            "matrix_12x12": Q.tolist(),
        },
        "expected_under_the_model": {"chi2_total_mean": 12.0,
                                     "chi2_block_mean": 3.0},
        "chi2_total": _pct(chi2.numpy()),
        "standardized_total_sqrt_chi2": _pct(chi2.sqrt().numpy()),
        "chi2_block": {k: _pct(b.numpy()) for k, b in block.items()},
        "innovation_norm_raw": {k: _pct(b.numpy()) for k, b in norm.items()},
        "E_process_total": float(0.5 * chi2.sum()),
        "E_process_tail_share": _tail_share(chi2),
        "per_gaussian_worst_pair_chi2": _pct(chi2.max(0).values.numpy()),
        "E_process_per_pair": (0.5 * chi2.sum(1)).tolist(),
        "chi2_median_per_pair": chi2.median(1).values.tolist(),
        "chi2_p99_per_pair": torch.quantile(chi2, 0.99, dim=1).tolist(),
        "time_reversal": {
            "chronological_E_process": float(0.5 * chi2.sum()),
            "reversed_with_rates_negated_E_process": float(0.5 * chi2_rev.sum()),
            "reversed_WITHOUT_negation_E_process": float(0.5 * chi2_naive.sum()),
            "note": ("the constant-velocity model is time-reversible when rates "
                     "are negated, so the first two should be comparable; "
                     "flipping without negating should be far worse"),
        },
        "the_176_sigma_claim": {
            "invalid_statistic": "max ||mu_{t+1} - mu_t|| / sqrt(Q[mu,mu])",
            "invalid_value_max": float(invalid.max()),
            "invalid_value_p99": float(torch.quantile(invalid.reshape(-1)[
                ::max(1, invalid.numel() // 2_000_000)], 0.99)),
            "correct_statistic": "sqrt(e' Q^-1 e) with e the 12-D innovation",
            "correct_value_p50": float(chi2.sqrt().median()),
            "correct_value_p99": float(torch.quantile(chi2.sqrt().reshape(-1)[
                ::max(1, chi2.numel() // 2_000_000)], 0.99)),
            "correct_value_max": float(chi2.sqrt().max()),
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / f"process_innovations_{args.label}.json"
    jpath.write_text(json.dumps(result, indent=1))

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    x = ids[:-1]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("E_process per frame pair",
                                        "chi2 per Gaussian: median and p99 (model mean 12)",
                                        "marginal block chi2, median (model mean 3)"))
    fig.add_trace(go.Scatter(x=x, y=result["E_process_per_pair"], name="E_process"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=result["chi2_median_per_pair"], name="chi2 median"), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=result["chi2_p99_per_pair"], name="chi2 p99"), 2, 1)
    fig.add_hline(y=12.0, line_dash="dot", row=2, col=1)
    for k, b in block.items():
        fig.add_trace(go.Scatter(x=x, y=b.median(1).values.tolist(),
                                 name=f"{k} median"), 3, 1)
    fig.add_hline(y=3.0, line_dash="dot", row=3, col=1)
    fig.update_yaxes(type="log")
    fig.update_layout(title=f"{args.label}: whitened process innovations "
                            f"({prov['mode']}, N={N}, dt={dt:.5f} s declared)",
                      height=900)
    hpath = out / f"process_innovations_{args.label}.html"
    fig.write_html(str(hpath), include_plotlyjs="cdn")

    c = result["the_176_sigma_claim"]
    print(f"[{args.label}] {prov['mode']}  N={N}  T={T}  dt={dt:.5f} (declared)")
    print(f"  Q sd: mu {sd[0]:.3e}  phi {sd[3]:.3e}  v {sd[6]:.3e}  "
          f"omega {sd[9]:.3e};  corr(mu,v) {corr_mu_v:.3f}")
    print(f"  chi2 total   p50 {result['chi2_total']['p50']:.3e}  "
          f"p99 {result['chi2_total']['p99']:.3e}  max "
          f"{result['chi2_total']['p100']:.3e}  (model mean 12)")
    for k in BLOCKS:
        b = result["chi2_block"][k]
        print(f"  chi2 {k:<6}  p50 {b['p50']:.3e}  p99 {b['p99']:.3e}  "
              f"max {b['p100']:.3e}  (model mean 3)")
    ts = result["E_process_tail_share"]
    print(f"  share of E_process in the largest terms: top 0.1% "
          f"{100 * ts['top_0.1pct_of_terms']:.1f}%, top 1% "
          f"{100 * ts['top_1pct_of_terms']:.1f}%, top 5% "
          f"{100 * ts['top_5pct_of_terms']:.1f}%")
    print(f"  E_process {result['E_process_total']:.4e}; time reversal: "
          f"negated {result['time_reversal']['reversed_with_rates_negated_E_process']:.4e}, "
          f"NOT negated {result['time_reversal']['reversed_WITHOUT_negation_E_process']:.4e}")
    print(f"  the '176 sigma' statistic (displacement / sqrt(Q_mu)): max "
          f"{c['invalid_value_max']:.1f}; the correct sqrt(chi2): p50 "
          f"{c['correct_value_p50']:.2f}, p99 {c['correct_value_p99']:.2f}, "
          f"max {c['correct_value_max']:.1f}")
    print(f"  {jpath}\n  {hpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
