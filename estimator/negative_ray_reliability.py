"""Per-pixel reliability q_p of the Phase-1 negative-ray (KNOWN_NONOBJECT) observation.

The negative-ray likelihood (estimator/occlusion_likelihood.py, iekf_negative_ray.cu) is
the two-component observation model

    p(ray p | x) = (1 - q_p) * 1  +  q_p * E[T_front(x; D_p)]
    L_p          = -log((1 - q_p) + q_p (1 - A_p)) = -log(1 - q_p A_p)

where the clean component says "no modeled object surface in front of the measured depth"
and the contaminated component (mask false negative, or a surface the sensor cannot tell
from the object's own) carries no information. q_p depends ONLY on the observation (mask,
raw depth, calibrated depth sd), never on the state, so L stays a likelihood and its
derivatives are those of -log(1 - q A) with q fixed per pixel.

Calibration (per camera, pooled over the frames of the run; observation data only, no GT)
-----------------------------------------------------------------------------------------
For every candidate pixel (M = 0, valid raw depth, not on the 3x3 mask edge, not a
depth edge -- see below):
  d_p        Euclidean distance (px) to the nearest object pixel with valid depth (M = 1)
  cons_p     |D_p - D_m(p)| <= z_a sqrt(sigma_p^2 + sigma_m^2): the measured depth cannot be
             told apart from the object's own surface at that nearest pixel (two-sided test
             at level a; sigma = the run's depth sd at the OBSERVED depth)
Declared assumptions:
  (A1) a contaminated pixel is depth-consistent with the nearest object pixel (a mask false
       negative IS object surface; an indistinguishable touching surface is consistent by
       definition);
  (A2) a clean pixel is depth-consistent at a rate r_inf that does not depend on d, measured
       on the far half of the candidates (d >= d_far = the median candidate distance);
  (A3) the contamination fraction depends on d only (per camera).
Then r(d) = c(d) + (1 - c(d)) r_inf, so
  c(d)              = clip((r(d) - r_inf) / (1 - r_inf), 0, 1),  r(d) fitted non-increasing
                      in d (weighted pool-adjacent-violators) on d < d_far, c = 0 beyond
  q_p (cons_p)      = P(clean | consistent, d) = (1 - c(d)) r_inf / r(d)      (<= 1)
  q_p (not cons_p)  = 1   (by A1 a contaminated pixel is never inconsistent)
A2 is violated by surfaces continuous with the object (a floor under its feet): those raise
r(d) near the object, so c is OVER-estimated and q UNDER-estimated -- the error removes
negative evidence, it never adds free-space pressure.

Depth edges (mixed returns): a pixel whose 3x3 valid depths have a range larger than the
(1 - a) quantile of the range of n iid N(0, sigma_p^2) samples is a depth discontinuity;
its measured depth may be a mixture of the two surfaces and no reliability can be
calibrated for it with the data available. It is UNKNOWN (q = 0, no term), as the spec
requires when reliability cannot be supported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

LEVEL = 0.0027          # two-sided 3-sigma level of both tests (consistency and depth edge)


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (bisection on erfc; exact to ~1e-12)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def range_quantile(n: int, level: float = LEVEL) -> float:
    """(1 - level) quantile of the range of n iid N(0,1) samples:
    F_R(w) = n int phi(x) [Phi(x + w) - Phi(x)]^(n-1) dx (trapezoid on [-9, 9])."""
    if n < 2:
        return float("inf")
    x = np.linspace(-9.0, 9.0, 6001)
    phi = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    from scipy.special import ndtr
    Px = ndtr(x)

    def cdf(w):
        return float(n * np.trapz(phi * np.clip(ndtr(x + w) - Px, 0.0, 1.0) ** (n - 1), x))
    lo, hi = 0.0, 20.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < 1.0 - level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_RANGE_Q: dict = {}


def _range_table(level: float) -> torch.Tensor:
    if level not in _RANGE_Q:
        _RANGE_Q[level] = torch.tensor([float("inf"), float("inf")] + [range_quantile(n, level) for n in range(2, 10)],
                                       dtype=torch.float64)
    return _RANGE_Q[level]


def depth_edge(D: torch.Tensor, V: torch.Tensor, sigma: torch.Tensor, level: float = LEVEL) -> torch.Tensor:
    """[H,W] bool: valid pixel whose valid 3x3 depths (itself included) have a range above the
    (1 - level) quantile of the range of n iid N(0, sigma_p^2) samples."""
    Dd = torch.where(V, D.double(), torch.zeros_like(D, dtype=torch.float64))
    big = torch.finfo(torch.float64).max
    hi = torch.nn.functional.max_pool2d(torch.where(V, Dd, torch.full_like(Dd, -big))[None, None], 3, 1, 1)[0, 0]
    lo = -torch.nn.functional.max_pool2d(torch.where(V, -Dd, torch.full_like(Dd, -big))[None, None], 3, 1, 1)[0, 0]
    n = torch.nn.functional.avg_pool2d(V.double()[None, None], 3, 1, 1, count_include_pad=True)[0, 0] * 9.0
    n = n.round().long().clamp(0, 9)
    w = _range_table(level).to(D.device)[n]
    return V & (n >= 2) & ((hi - lo) > w * sigma.double())


@dataclass
class Features:
    candidate: torch.Tensor     # [H,W] bool: M = 0, valid raw depth, off the 3x3 mask edge, not a depth edge
    distance: torch.Tensor      # [H,W] float64 px to the nearest valid M = 1 pixel (inf when none)
    consistent: torch.Tensor    # [H,W] bool
    edge: torch.Tensor          # [H,W] bool depth edge (M = 0 pixels only are relevant)
    near_mask: torch.Tensor     # [H,W] bool M = 0 pixel with an M = 1 pixel in its 3x3


def features(M: torch.Tensor, D: torch.Tensor, V: torch.Tensor, sigma: torch.Tensor, level: float = LEVEL,
             mask_edge: str = "unknown") -> Features:
    from scipy.ndimage import distance_transform_edt
    M, V = M.bool(), V.bool()
    near = (torch.nn.functional.max_pool2d(M.float()[None, None], 3, 1, 1)[0, 0] > 0) & ~M
    edge = depth_edge(D, V, sigma, level) & ~M
    obj = (M & V).cpu().numpy()
    Dn = D.detach().double().cpu().numpy()
    sn = sigma.detach().double().cpu().numpy()
    if obj.any():
        dist, (iy, ix) = distance_transform_edt(~obj, return_indices=True)
        Dm, sm = Dn[iy, ix], sn[iy, ix]
        z = _norm_ppf(1.0 - level / 2.0)
        cons = np.abs(Dn - Dm) <= z * np.sqrt(sn * sn + sm * sm)
    else:
        dist = np.full(obj.shape, np.inf)
        cons = np.zeros(obj.shape, bool)
    dev = M.device
    dist_t = torch.from_numpy(dist).to(dev)
    cons_t = torch.from_numpy(cons).to(dev)
    # no object pixel in this image: every M = 0 pixel is infinitely far from the mask, i.e.
    # far field (A2: c = 0); it stays a candidate with q = 1 (consistency is undefined -> False)
    cand = (~M) & V & ~edge
    if mask_edge == "unknown":
        cand = cand & ~near
    return Features(cand, dist_t, cons_t & cand, edge, near)


def _pav_nonincreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-increasing) regression by pool-adjacent-violators."""
    blocks = []                       # [value, weight, length]
    for yi, wi in zip(y.tolist(), w.tolist()):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            ww = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / ww if ww > 0 else 0.5 * (v1 + v2), ww, n1 + n2])
    out = []
    for v, _, n in blocks:
        out += [v] * n
    return np.asarray(out, dtype=np.float64)


@dataclass
class CameraReliability:
    camera: str
    level: float
    d_far: int
    r_inf: float
    bins: np.ndarray           # integer distance bins 0..d_far-1 (bin k = ceil(d) == k)
    counts: np.ndarray
    consistent: np.ndarray
    r_fit: np.ndarray           # isotonic r(d) on the bins
    contamination: np.ndarray   # c(d) on the bins
    stats: dict = field(default_factory=dict)

    def q_map(self, f: Features) -> torch.Tensor:
        """[H,W] float32 q_p: 0 off the candidates (UNKNOWN), 1 for inconsistent candidates,
        (1 - c(d)) r_inf / r(d) for consistent ones."""
        dev = f.candidate.device
        q = torch.zeros(f.candidate.shape, dtype=torch.float64, device=dev)
        q[f.candidate] = 1.0
        k = torch.ceil(f.distance).clamp(0, 1e9).long()
        inside = f.consistent & (k < self.d_far)
        if bool(inside.any()):
            r = torch.as_tensor(self.r_fit, device=dev)
            c = torch.as_tensor(self.contamination, device=dev)
            kk = k[inside]
            rq = r[kk]
            val = torch.where(rq > 0, (1.0 - c[kk]) * self.r_inf / rq.clamp_min(1e-300), torch.ones_like(rq))
            q[inside] = val.clamp(0.0, 1.0)
        return q.float()

    def to_manifest(self) -> dict:
        nz = self.counts > 0
        show = [int(b) for b in self.bins[nz][:24]]
        return {"camera": self.camera, "level": self.level, "d_far_px": int(self.d_far), "r_inf": self.r_inf,
                "r_fit_first_bins": {str(b): float(self.r_fit[b]) for b in show},
                "contamination_first_bins": {str(b): float(self.contamination[b]) for b in show},
                "q_consistent_first_bins": {str(b): (float((1 - self.contamination[b]) * self.r_inf / self.r_fit[b])
                                                     if self.r_fit[b] > 0 else 1.0) for b in show},
                **self.stats}


def calibrate(camera: str, frames, level: float = LEVEL, mask_edge: str = "unknown") -> CameraReliability:
    """frames: iterable of (M, D, V_raw, sigma) for ONE camera (any device)."""
    dists, conss = [], []
    n_m0_valid = n_edge = n_near = n_frames = 0
    for M, D, V, sig in frames:
        f = features(M, D, V, sig, level, mask_edge)
        c = f.candidate
        dists.append(f.distance[c].cpu().numpy())
        conss.append(f.consistent[c].cpu().numpy())
        m0v = (~M.bool()) & V.bool()
        n_m0_valid += int(m0v.sum())
        n_edge += int((f.edge & m0v).sum())
        n_near += int((f.near_mask & m0v).sum())
        n_frames += 1
    d = np.concatenate(dists) if dists else np.zeros(0)
    cs = np.concatenate(conss) if conss else np.zeros(0, bool)
    n_inf = int((~np.isfinite(d)).sum())
    fin = np.isfinite(d)
    d, cs = d[fin], cs[fin]
    if d.size == 0:
        # the object is never visible in this camera: no distance-dependent contamination can be
        # measured; every candidate is far field (c = 0, q = 1 for inconsistent = all of them)
        return CameraReliability(camera, float(level), 2, 0.0, np.arange(2), np.zeros(2), np.zeros(2),
                                 np.zeros(2), np.zeros(2), {"frames": n_frames, "candidate_pixels": n_inf,
                                                            "note": "no M = 1 pixel in any calibration frame"})
    d_far = max(int(math.ceil(float(np.median(d)))), 2)
    far = d >= d_far
    r_inf = float(cs[far].mean()) if far.any() else 0.0
    k = np.ceil(d).astype(np.int64)
    near = k < d_far
    counts = np.bincount(k[near], minlength=d_far).astype(np.float64)
    consistent = np.bincount(k[near], weights=cs[near].astype(np.float64), minlength=d_far)
    raw = np.where(counts > 0, consistent / np.maximum(counts, 1), 0.0)
    used = counts > 0
    r_fit = np.full(d_far, r_inf, dtype=np.float64)
    if used.any():
        r_fit[used] = _pav_nonincreasing(raw[used], counts[used])
        # empty bins take the next populated bin's value (non-increasing step function)
        last = r_inf
        for b in range(d_far - 1, -1, -1):
            if used[b]:
                last = r_fit[b]
            else:
                r_fit[b] = last
    contam = np.clip((r_fit - r_inf) / max(1.0 - r_inf, 1e-12), 0.0, 1.0) if r_inf < 1.0 else np.zeros(d_far)
    stats = {"frames": n_frames, "candidate_pixels": int(d.size), "candidates_without_object_in_view": n_inf,
             "consistent_fraction": float(cs.mean()),
             "m0_valid_pixels": n_m0_valid, "depth_edge_fraction_of_m0_valid": n_edge / max(n_m0_valid, 1),
             "mask_edge_fraction_of_m0_valid": n_near / max(n_m0_valid, 1),
             "assumptions": ["A1 contaminated => depth-consistent with the nearest object pixel",
                             "A2 clean pixels consistent at the far-field rate r_inf, independent of d",
                             "A3 contamination depends on the distance to the mask only (per camera)"]}
    return CameraReliability(camera, float(level), d_far, r_inf, np.arange(d_far), counts, consistent, r_fit,
                             contam, stats)
