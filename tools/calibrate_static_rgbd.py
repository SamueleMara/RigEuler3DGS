#!/usr/bin/env python
"""Per-camera RGB-D sensor-noise calibration from a user-supplied STATIC interval.

RigEuler3DGS_Calibrated_Bootstrap_IEKF_Implementation_Spec.md section 2.

Measurement model at pixel p of camera c and static frame t:

    D_c,t,p = d*_c,p + b_c,p + e_c,t,p

The static interval identifies the fluctuating part e (its variance as a
function of depth, its temporal correlation, missing returns). It does NOT
identify the persistent bias b apart from the true depth d*; that is stated in
the output, never estimated.

Estimators (all on pixels inside the object mask in EVERY calibration frame,
raw finite depth > 0 only -- background pixels are never used):

  location     per-pixel temporal median of D (robust)
  residual     e = D - median_t D, after removing each frame's common-mode
               offset (median over the camera's pixels), which is reported
               separately as `global_offset_std_m` -- it is a shared, correlated
               term the independent per-pixel model does not represent
  variance     pooled 1.4826 MAD of e per depth bin, x T/(T-1) small-sample
               factor; the consecutive-difference estimator
               Var(dD) / (2 (1 - rho_1)) is reported as a cross-check (the naive
               Var(dD)/2 under-estimates autocorrelated noise)
  depth model  sigma^2(z) = a [+ b z^2 [+ d z^4]], non-negative, fitted on the
               EVEN calibration frames and selected by the Gaussian NLL of the
               ODD frames' residuals; forms the depth range cannot support
               (too few bins / too narrow a range) are not fitted
  quantization the measured depth step q; q^2/12 is the variance floor
  temporal     pooled lag-k autocorrelation of e, k = 1..5, with a 95 % interval
               from a pixel bootstrap; a DIAGNOSTIC (section 4.2), not a model
  RGB          3x3 channel covariance of the temporal residual, robustly trimmed
  invalid      fraction of (in-mask pixel, frame) samples with no valid return

Outputs (into --out, normally RESULTS/calibration):

  sensor_model.json   utils/noise/sensor_model.py SceneNoiseModel (policy calibrated)
  covariance.npz      per-camera binned variances, intervals, autocorrelation, RGB covariance
  validation.json     interval checks, held-out model selection, split-half consistency,
                      stationarity diagnostics

    python tools/calibrate_static_rgbd.py --source <episode> \\
        --calibration-start-frame 0 --calibration-end-frame 99 --out RESULTS/calibration
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MADK = 1.4826
RGB_EXT = (".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
# pure estimators (arrays in, numbers out) -- unit-tested
# ---------------------------------------------------------------------------
def robust_var(x: np.ndarray, q: float = 0.0, k: float = 5.0) -> float:
    """Outlier-trimmed sample variance, safe on QUANTIZED data.

    A MAD-based scale collapses to 0 when more than half the samples share one
    quantization level (registered depth in ~1 mm steps, 8-bit RGB), so the
    trimming bound uses max(MAD scale, q) and keeps everything within k of it;
    the variance itself is the ordinary sample variance of the kept samples. It
    INCLUDES the quantization noise, which is part of what the filter observes
    (Sheppard: ~ sigma^2 + q^2/12)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    m = np.median(x)
    scale = max(MADK * float(np.median(np.abs(x - m))), float(q), 1e-12)
    keep = np.abs(x - m) <= k * scale
    xk = x[keep]
    return float(np.var(xk, ddof=1)) if xk.size >= 2 else float("nan")


def measured_quantization(D: np.ndarray, V: np.ndarray | None = None) -> float:
    """Per-pixel step: the smallest positive difference between distinct values of
    ONE pixel's time series (registered depth is quantized on a per-pixel grid),
    pooled as the median over pixels showing >= 2 levels. 0 if none do.
    D: [T, P] (or 1-D, treated as one pixel)."""
    X = np.asarray(D, float)
    if X.ndim == 1:
        X = X[:, None]
    if V is not None:
        X = np.where(V, X, np.nan)
    Xs = np.sort(X, axis=0)
    d = np.diff(Xs, axis=0)
    d = np.where(np.isfinite(d) & (d > 1e-9), d, np.inf)
    step = d.min(axis=0)
    step = step[np.isfinite(step)]
    return float(np.median(step)) if step.size else 0.0


def temporal_residuals(D: np.ndarray, V: np.ndarray):
    """D, V: [T, P] depth and validity. Returns (median [P], residual [T,P] NaN where
    invalid, per-frame common-mode offset [T])."""
    X = np.where(V, D, np.nan)
    med = np.nanmedian(X, axis=0)
    E = X - med[None]
    offset = np.nanmedian(E, axis=1)                      # common mode per frame
    offset = np.where(np.isfinite(offset), offset, 0.0)
    return med, E - offset[:, None], offset


def autocorrelation(E: np.ndarray, max_lag: int = 5, boot: int = 200, seed: int = 0):
    """Pooled lag-k autocorrelation of residuals E [T,P] with a pixel-bootstrap 95% CI."""
    T = E.shape[0]
    out = []
    rng = np.random.default_rng(seed)
    num = {}
    for k in range(1, min(max_lag, T - 1) + 1):
        a, b = E[:-k], E[k:]
        ok = np.isfinite(a) & np.isfinite(b)
        prod = np.where(ok, a * b, 0.0).sum(0)
        den = np.where(ok, 0.5 * (a * a + b * b), 0.0).sum(0)
        num[k] = (prod, den)
        rho = float(prod.sum() / max(den.sum(), 1e-300))
        P = prod.size
        idx = rng.integers(0, P, size=(boot, P))
        bs = prod[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-300)
        out.append({"lag": k, "rho": rho, "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]})
    return out


def binned_variance(z: np.ndarray, E: np.ndarray, T: int, bins: int = 8, min_count: int = 200,
                    boot: int = 100, seed: int = 0, q: float = 0.0) -> list:
    """Pool residuals E [T,P] by the pixels' median depth z [P] into quantile bins."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(z)
    zq = np.quantile(z[ok], np.linspace(0, 1, bins + 1)) if ok.sum() else np.array([])
    rows = []
    corr = T / max(T - 1, 1)
    for i in range(len(zq) - 1):
        sel = ok & (z >= zq[i]) & ((z <= zq[i + 1]) if i == len(zq) - 2 else (z < zq[i + 1]))
        e = E[:, sel]
        e = e[np.isfinite(e)]
        if e.size < min_count:
            continue
        var = robust_var(e, q) * corr
        pix = np.nonzero(sel)[0]
        bs = []
        for _ in range(boot):                               # pixel bootstrap of the pooled MAD
            pick = rng.choice(pix, size=pix.size, replace=True)
            ee = E[:, pick]
            bs.append(robust_var(ee[np.isfinite(ee)], q) * corr)
        rows.append({"z_center": float(np.median(z[sel])), "z_lo": float(zq[i]), "z_hi": float(zq[i + 1]),
                     "variance": var,
                     "ci95": ([float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))] if bs
                              else [float("nan"), float("nan")]),
                     "pixels": int(sel.sum()), "samples": int(e.size)})
    return rows


FORMS = {"a": (0,), "a+bz2": (0, 2), "a+bz2+dz4": (0, 2, 4)}


def fit_form(rows: list, form: str):
    """Non-negative weighted least squares of the binned variances on z^powers."""
    from scipy.optimize import nnls
    pw = FORMS[form]
    z = np.array([r["z_center"] for r in rows])
    v = np.array([r["variance"] for r in rows])
    w = 1.0 / np.maximum(v, 1e-18)                         # relative error weighting
    A = np.stack([z ** p for p in pw], 1) * w[:, None]
    coef, _ = nnls(A, v * w)
    full = [0.0, 0.0, 0.0]
    for p, c in zip(pw, coef):
        full[{0: 0, 2: 1, 4: 2}[p]] = float(c)
    return tuple(full)


def heldout_nll(coeffs, q, z: np.ndarray, E: np.ndarray) -> float:
    """Mean Gaussian NLL of residuals E [T,P] at pixel depth z [P] under sigma^2(z)."""
    a, b, d = coeffs
    s2 = np.maximum(a + b * z ** 2 + d * z ** 4, max(q * q / 12.0, 1e-12))[None]
    ok = np.isfinite(E) & np.isfinite(s2)
    r2 = np.where(ok, E * E / s2 + np.log(s2), 0.0)
    return float(0.5 * r2.sum() / max(ok.sum(), 1))


def select_depth_model(z, E_even, E_odd, T_even, q, bins=8, min_count=200):
    """Fit every supported form on the even frames, choose by odd-frame NLL."""
    rows = binned_variance(z, E_even, T_even, bins=bins, min_count=min_count, q=q)
    out = {"bins_even": rows, "candidates": {}}
    if not rows:
        return None, out
    zr = (min(r["z_center"] for r in rows), max(r["z_center"] for r in rows))
    for form, pw in FORMS.items():
        supported = len(rows) >= len(pw) + 1 and (len(pw) == 1 or zr[1] / max(zr[0], 1e-9) > 1.2)
        if not supported:
            out["candidates"][form] = {"supported": False,
                                       "reason": f"{len(rows)} bins over z in [{zr[0]:.3f}, {zr[1]:.3f}] m"}
            continue
        c = fit_form(rows, form)
        out["candidates"][form] = {"supported": True, "coeffs": c, "heldout_nll": heldout_nll(c, q, z, E_odd)}
    ok = {k: v for k, v in out["candidates"].items() if v.get("supported")}
    best = min(ok, key=lambda k: ok[k]["heldout_nll"])
    out["selected"] = best
    out["depth_range_m"] = zr
    return best, out


def rgb_covariance(rgb: np.ndarray, V: np.ndarray, T: int, q: float = 1.0 / 255.0) -> np.ndarray:
    """rgb [T,3,P] in [0,1]; residual about the per-pixel median, trimmed at
    5 x max(MAD scale, q) -- 8-bit images are quantized at q = 1/255."""
    R = rgb - np.median(rgb, axis=0, keepdims=True)
    R = R.transpose(1, 0, 2).reshape(3, -1)
    scale = np.maximum(MADK * np.median(np.abs(R), axis=1, keepdims=True), q)
    keep = np.all(np.abs(R) <= 5 * scale, axis=0)
    R = R[:, keep]
    if R.shape[1] < 10:
        return np.zeros((3, 3))
    return np.cov(R) * T / max(T - 1, 1)


def calibrate_camera(name, D, V, rgb, fg, rng_seed=0, bins=8, min_count=200) -> tuple:
    """D [T,H,W] metres, V valid, rgb [T,3,H,W] in [0,1], fg [T,H,W] masks.
    Returns (CameraNoise, diagnostics dict, arrays dict)."""
    from utils.noise.sensor_model import CameraNoise
    T = D.shape[0]
    always_fg = fg.all(0)
    valid = V & always_fg[None]
    pix = np.nonzero(always_fg.reshape(-1))[0]
    Dp = D.reshape(T, -1)[:, pix]
    Vp = valid.reshape(T, -1)[:, pix]
    enough = Vp.sum(0) >= max(2, int(0.5 * T))
    Dp, Vp = Dp[:, enough], Vp[:, enough]
    if Dp.shape[1] < min_count:
        raise SystemExit(f"{name}: only {Dp.shape[1]} in-mask pixels with valid depth in >= half the "
                         "calibration frames -- not enough to calibrate")
    q = measured_quantization(Dp, Vp)
    med, E, offset = temporal_residuals(Dp, Vp)
    even, odd = np.arange(0, T, 2), np.arange(1, T, 2)
    if odd.size < 2 or even.size < 2:
        raise SystemExit(f"{name}: {T} frames -- need >= 4 for the held-out model selection")
    _, Ee, _ = temporal_residuals(Dp[even], Vp[even])
    _, Eo, _ = temporal_residuals(Dp[odd], Vp[odd])
    form, sel = select_depth_model(med, Ee, Eo, even.size, q, bins=bins, min_count=min_count)
    if form is None:
        raise SystemExit(f"{name}: no depth bin has >= {min_count} samples")
    rows_all = binned_variance(med, E, T, bins=bins, min_count=min_count, seed=rng_seed, q=q)
    coeffs = fit_form(rows_all, form)
    ac = autocorrelation(E)
    rho1 = ac[0]["rho"] if ac else 0.0
    dD = np.diff(np.where(Vp, Dp, np.nan), axis=0) - np.diff(offset)[:, None]
    var_diff = robust_var(dD[np.isfinite(dD)], q)
    alt = {"naive_var_dD_over_2": var_diff / 2, "corrected_var_dD_over_2(1-rho1)":
           var_diff / (2 * max(1 - rho1, 1e-6)), "residual_about_median": robust_var(E[np.isfinite(E)], q) * T / (T - 1)}
    # split half: first vs second half of the interval
    h = T // 2
    _, E1, _ = temporal_residuals(Dp[:h], Vp[:h])
    _, E2, _ = temporal_residuals(Dp[h:], Vp[h:])
    r1 = binned_variance(med, E1, h, bins=bins, min_count=min_count // 2, boot=0, q=q)
    r2 = binned_variance(med, E2, T - h, bins=bins, min_count=min_count // 2, boot=0, q=q)
    split = [{"z_center": a["z_center"], "var_first": a["variance"], "var_second": b["variance"],
              "ratio": b["variance"] / max(a["variance"], 1e-18)} for a, b in zip(r1, r2)]
    rgbp = rgb.reshape(T, 3, -1)[:, :, pix][:, :, enough]
    C = rgb_covariance(rgbp, Vp, T)
    inv = float(1.0 - valid.sum() / max(always_fg.sum() * T, 1))
    zr = sel.get("depth_range_m", (float(np.nanmin(med)), float(np.nanmax(med))))
    cam = CameraNoise(camera=name, depth_coeffs=coeffs, depth_model=form, depth_quantization_m=q,
                      depth_range_m=tuple(float(x) for x in zr), rgb_covariance=tuple(tuple(map(float, r)) for r in C),
                      invalid_fraction=inv, lag1_autocorrelation=float(rho1),
                      autocorrelation_ci95=tuple(ac[0]["ci95"]) if ac else (0.0, 0.0),
                      global_offset_std_m=float(np.std(offset)), num_pixels=int(Dp.shape[1]), num_frames=int(T))
    diag = {"camera": name, "selection": sel, "bins": rows_all, "autocorrelation": ac,
            "variance_estimators_pooled": alt, "split_half": split, "quantization_m": q,
            "global_offset_m": offset.tolist(), "rgb_covariance": C.tolist(),
            "rgb_correlation": (C / np.sqrt(np.outer(np.diag(C), np.diag(C)) + 1e-30)).tolist(),
            "invalid_fraction": inv}
    arrays = {f"{name}_bins_z": np.array([r["z_center"] for r in rows_all]),
              f"{name}_bins_var": np.array([r["variance"] for r in rows_all]),
              f"{name}_bins_ci": np.array([r["ci95"] for r in rows_all]),
              f"{name}_autocorr": np.array([[a["lag"], a["rho"], *a["ci95"]] for a in ac]),
              f"{name}_rgb_cov": C}
    return cam, diag, arrays


# ---------------------------------------------------------------------------
# dataset io
# ---------------------------------------------------------------------------
def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_interval(source: Path, cam: str, frames: list):
    from PIL import Image
    D, V, RGB, FG, files = [], [], [], [], []
    for t in frames:
        rp = next((source / "rgb" / cam / f"frame_{t:06d}{e}" for e in RGB_EXT
                   if (source / "rgb" / cam / f"frame_{t:06d}{e}").is_file()), None)
        dp = source / "depth_npy" / cam / f"frame_{t:06d}.npy"
        mp = next((source / "mask" / cam / f"frame_{t:06d}{e}" for e in RGB_EXT
                   if (source / "mask" / cam / f"frame_{t:06d}{e}").is_file()), None)
        if rp is None or not dp.is_file() or mp is None:
            raise SystemExit(f"{cam} frame {t}: missing rgb/depth/mask "
                             f"({rp}, {dp if dp.is_file() else None}, {mp}) -- the calibration "
                             "interval must be complete")
        d = np.load(dp).astype(np.float32)
        D.append(d)
        V.append(np.isfinite(d) & (d > 0))
        RGB.append(np.asarray(Image.open(rp).convert("RGB"), np.float32).transpose(2, 0, 1) / np.float32(255.0))
        m = np.asarray(Image.open(mp))
        FG.append((m[..., 0] if m.ndim == 3 else m) > 0)
        files += [rp, dp, mp]
    return np.stack(D), np.stack(V), np.stack(RGB), np.stack(FG), files


def frame_timestamps(source: Path):
    p = source / "metadata" / "frame_timestamps.csv"
    if not p.is_file():
        return None
    import csv
    return {int(r["frame_index"]): (float(r["timestamp_s"]), int(r.get("source_frame_index", r["frame_index"])))
            for r in csv.DictReader(open(p))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True, help="dataset episode / scene directory")
    ap.add_argument("--calibration-start-frame", type=int, required=True, help="inclusive frame ID")
    ap.add_argument("--calibration-end-frame", type=int, required=True, help="inclusive frame ID")
    ap.add_argument("--cameras", default=None, help="comma list (default: every rgb/cam_*)")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--min-bin-samples", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = Path(a.source)
    cs, ce = a.calibration_start_frame, a.calibration_end_frame
    if ce - cs + 1 < 4:
        raise SystemExit(f"calibration interval [{cs}, {ce}] has {ce - cs + 1} frames; at least 4 are "
                         "needed (two for a variance, two more for the held-out model selection)")
    cams = (a.cameras.split(",") if a.cameras else
            sorted(p.name for p in (src / "rgb").glob("cam_*") if p.is_dir()))
    frames = list(range(cs, ce + 1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from utils.noise.sensor_model import SceneNoiseModel
    cams_noise, diags, arrays, hashes = {}, {}, {}, {}
    for c in cams:
        D, V, RGB, FG, files = load_interval(src, c, frames)
        cam, diag, arr = calibrate_camera(c, D, V, RGB, FG, bins=a.bins, min_count=a.min_bin_samples)
        cams_noise[c], diags[c] = cam, diag
        arrays.update(arr)
        hashes[c] = hashlib.sha256("".join(_sha(f) for f in files).encode()).hexdigest()
        s = math.sqrt(float(cam.depth_variance(np.array([float(np.mean(cam.depth_range_m))]))[0]))
        print(f"  {c}: depth model {cam.depth_model} {tuple(f'{x:.3e}' for x in cam.depth_coeffs)}, "
              f"sigma_D at {np.mean(cam.depth_range_m):.2f} m = {1000 * s:.2f} mm, q = {1000 * cam.depth_quantization_m:.2f} mm, "
              f"rho1 {cam.lag1_autocorrelation:+.2f}, rgb sd {np.sqrt(np.diag(np.asarray(cam.rgb_covariance))).round(4).tolist()}, "
              f"invalid {100 * cam.invalid_fraction:.1f} %")
    ts = frame_timestamps(src)
    prov = {"tool": "tools/calibrate_static_rgbd.py", "source": str(src), "calibration_frames": [cs, ce],
            "source_frame_indices": [ts[t][1] for t in frames] if ts else None,
            "input_sha256_per_camera": hashes, "mask_policy": "object foreground in every calibration frame",
            "estimators": "per-pixel temporal median residual, common mode removed; pooled MAD x T/(T-1)"}
    model = SceneNoiseModel(policy="calibrated", cameras=cams_noise, provenance=prov)
    mid = model.to_json(out / "sensor_model.json")
    np.savez_compressed(out / "covariance.npz", **arrays)
    stationarity = {c: {"global_offset_std_m": d["global_offset_m"] and float(np.std(d["global_offset_m"])),
                        "global_offset_range_m": float(np.ptp(d["global_offset_m"]))} for c, d in diags.items()}
    (out / "validation.json").write_text(json.dumps({
        "sensor_model_id": mid, "interval": {"start": cs, "end": ce, "frames": len(frames)},
        "identifiability": model.identifiability, "stationarity": stationarity,
        "per_camera": diags,
        "notes": ["the interval is user-supplied: nothing here checks that the scene really was static "
                  "beyond the reported common-mode offsets and split-half ratios",
                  "temporal autocorrelation is a diagnostic only, never used as an AR(1) model of motion residuals",
                  "the global per-frame offset is a camera-shared correlated term NOT represented by the "
                  "independent per-pixel variance"]}, indent=1, default=float))
    print(f"wrote {out / 'sensor_model.json'} (sensor_model_id {mid[:12]}), covariance.npz, validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
