"""Cross-source co-contribution evidence: exact enumeration, bounded storage.

For F Gaussian i and B Gaussian j, over the SAME original (frame, camera)
observations, each source rendered alone:

  O_ij          = sum_p qF_pi qB_pj                                  same pixel
  O_surf_ij[t]  = sum_p qF_pi qB_pj kappa_t(zF_i, zB_j, D_gt(p))      same surface
      kappa_t   = valid_gt(p) AND |zF - zB| <= t AND |zF - D_gt| <= t AND |zB - D_gt| <= t
                  (box kernel in metres, t in PAIR_TOLS_M; primary = middle value)
  O_incompat_ij = sum_p qF qB [|zF - zB| > t_primary]                  different layers
  O_surf_rgb_ij = sum_p qF qB kappa_primary [RMS(cF_i - cB_j) <= rgb_tol]
  rgb_dist_ij   = sum_p qF qB RMS(cF_i - cB_j)            (/ O_ij = weighted mean)
  dist3d_ij     = sum_p qF qB |muF_i(t) - muB_j(t)|        centres at that frame
  dist3d_surf_ij= sum_p qF qB kappa_primary |muF_i(t) - muB_j(t)|

zF, zB are each contributor's own view depth, never the sources' mean
rendered depths. Colour is a separate column; nothing is gated on it.

WHAT IS EXACT AND WHAT IS BOUNDED. Measured on storage_45135: ~1.4e8
co-contributor pair records per view (up to ~330 contributors per pixel per
source), 7.4e6 distinct (i, j) per view, ~1.0e7 per frame, ~2.5e7 over the
sequence -- with 27% of a view's pairs carrying 99.86% of its pair mass.

  * Every (F contributor, B contributor) pair at every pixel is accounted
    for: the per-VIEW sums O_ij, O_surf_ij, ... are computed exactly as sparse
    matrix products (see view_pairs; checked against a CPU enumeration of all
    pairs in tests/test_fb_pixel_overlap_core.py T9).
  * A view's pair is stored only if its per-view O_ij >= prune_tau (declared,
    default 1e-4, recorded in the fingerprint; 0 = store everything). The
    pruned per-view mass is NOT discarded silently: it is accumulated exactly
    per F row and per B row (omitted_O, omitted_O_surf) and per view.
  * Hence a stored O_surf_ij is a LOWER bound of the true sequence value, the
    deficit of any pair (stored or not) involving F row i is at most
    omitted_O_surf_F[i] (and B row j: omitted_O_surf_B[j]), and a best partner
    is CERTIFIED when best - second >= that row's omitted mass.
  * One full frame is also enumerated with prune_tau = 0 as an exact
    reference, and the pruned and exact frame-level answers are compared.
"""

from __future__ import annotations

import numpy as np
import torch

from .gaussian_evidence import PAIR_TOLS_M
from .pixel_evidence import Thresholds

PRIMARY = 1
SUM_COLS = ("O", "O_surf_0", "O_surf_1", "O_surf_2", "O_incompat", "O_surf_rgb",
            "rgb_dist", "dist3d", "dist3d_surf", "npix", "npix_surf")


def _reduce_np(keys: np.ndarray, cols: dict):
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]]) if k.size else np.empty(0, np.int64)
    out = {c: np.add.reduceat(v[order], starts) if k.size else v[:0] for c, v in cols.items()}
    return k[starts] if k.size else k, out


def _csr(rows, cols, vals, shape):
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, shape).coalesce().to_sparse_csr()


def _coo_keys(m: torch.Tensor, ncols: int):
    crow, col, val = m.crow_indices(), m.col_indices(), m.values()
    row = torch.repeat_interleave(torch.arange(crow.numel() - 1, device=crow.device),
                                  crow[1:] - crow[:-1])
    return row * ncols + col, val


def view_pairs(cF, cB, zF_all, zB_all, colF, colB, muF, muB, gt_depth, valid,
               NF: int, NB: int, th: Thresholds, *, prune_tau: float = 1e-4,
               max_pairs_per_chunk: int = 0, only_cols=None, block_budget=None) -> dict:
    """Every co-contributor pair of one view, as exact sparse products, then pruned.

    With Q_F the [pixels x N_F] matrix of composited weights q (one entry per
    contributor record) and Q_B likewise:
        O                = Q_F^T Q_B
        S_t              = (Q_F o a_t)^T (Q_B o b_t),   a_t(p,i) = valid(p) [|zF_i - D(p)| <= t]
        O_surf_t         = S_t o [|zF_i - zB_j| <= t]   (zF, zB constant per view)
        O_incompat       = O o [|zF_i - zB_j| > t_primary]
        npix / npix_surf = the same products with q replaced by 1
    which is exactly sum_p over every co-contributor pair (each Gaussian
    contributes at most once per pixel). Colour and centre-distance columns are
    constant per pair and view, so they are O or O_surf_primary times the pair's
    value. cuSPARSE SpGEMM in float64, F rows in blocks bounded by their multiply
    count (memory only; every (i, j) belongs to exactly one block).
    """
    dev = cF.offsets.device
    HW = cF.H * cF.W
    nF, nB = cF.counts, cB.counts
    both = (nF > 0) & (nB > 0)
    total = int((nF[both] * nB[both]).sum())
    cols_wanted = only_cols or SUM_COLS
    stats = {"pixels_with_both_contributors": int(both.sum()), "pairs": total}
    zero = {"omitted_O_F": np.zeros(NF), "omitted_O_surf_F": np.zeros(NF),
            "omitted_O_B": np.zeros(NB), "omitted_O_surf_B": np.zeros(NB)}
    if total == 0:
        return {"keys": np.empty(0, np.int64),
                "cols": {c: np.empty(0, np.float64) for c in cols_wanted},
                "stats": stats | {"pair_mass": 0.0, "pair_mass_surf": 0.0, "distinct_pairs": 0,
                                  "stored_pairs": 0, "omitted_mass": 0.0,
                                  "omitted_mass_surf": 0.0}, **zero}
    tols = PAIR_TOLS_M
    D = gt_depth.reshape(-1)
    V = valid.reshape(-1)

    def masks(c, z_all):
        pix, g = c.pixel, c.gid.long()
        z, d, v = z_all[g], D[pix], V[pix]
        return pix, g, c.q.double(), [(v & ((z - d).abs() <= t)).double() for t in tols]

    pB, gB, qB, aB = masks(cB, zB_all)
    one_B = torch.ones_like(qB)
    need = {"O": True, "S0": "O_surf_0" in cols_wanted, "S1": True,
            "S2": "O_surf_2" in cols_wanted, "N": "npix" in cols_wanted,
            "NS": "npix_surf" in cols_wanted}
    Bvals = {"O": qB, "S0": qB * aB[0], "S1": qB * aB[1], "S2": qB * aB[2],
             "N": one_B, "NS": aB[1]}
    BM = {k: _csr(pB, gB, Bvals[k], (HW, NB)) for k in Bvals if need[k]}
    del Bvals
    pF, gF, qF, aF = masks(cF, zF_all)
    Fvals = {"O": qF, "S0": qF * aF[0], "S1": qF * aF[1], "S2": qF * aF[2],
             "N": torch.ones_like(qF), "NS": aF[1]}
    # multiply count per F Gaussian = sum over its pixels of B's contributor count
    work = torch.zeros(NF, dtype=torch.float64, device=dev).index_add_(0, gF, nB[pF].double())
    free, _ = torch.cuda.mem_get_info(dev)
    # ~60 bytes of cuSPARSE workspace per multiply (measured), products run one at a time
    budget = float(block_budget) if block_budget else max(1e5, 0.3 * free / 60.0)
    stats["multiplies"] = float(work.sum())
    order = torch.argsort(gF)
    gFs, pFs = gF[order], pF[order]
    Fv = {k: v[order] for k, v in Fvals.items() if need[k]}
    del Fvals, order
    cw = torch.cumsum(work, 0)
    bounds, start = [], 0
    while start < NF:
        lim = (float(cw[start - 1]) if start else 0.0) + budget
        end = int(torch.searchsorted(cw, torch.tensor(lim, device=dev, dtype=torch.float64), right=True))
        end = max(end, start + 1)
        bounds.append((start, end))
        start = end
    stats["spgemm_blocks"] = len(bounds)
    cut = torch.searchsorted(gFs, torch.tensor([b[0] for b in bounds] + [NF], device=dev)).tolist()
    keys_all, cols_all = [], {c: [] for c in cols_wanted}
    mass = mass_surf = om_mass = om_surf = 0.0
    distinct = stored = 0
    om = {k: torch.zeros(NF if k.endswith("_F") else NB, dtype=torch.float64, device=dev)
          for k in zero}
    t1 = tols[PRIMARY]
    for k, (a, b) in enumerate(bounds):
        s0, s1 = cut[k], cut[k + 1]
        if s1 == s0:
            continue
        rows = gFs[s0:s1] - a
        pp = pFs[s0:s1]
        prod = {}
        for name in BM:
            m = torch.sparse.mm(_csr(rows, pp, Fv[name][s0:s1], (b - a, HW)), BM[name])
            prod[name] = _coo_keys(m, NB)
            del m
        keyO, valO = prod["O"]
        srt = torch.argsort(keyO)
        keyO, valO = keyO[srt], valO[srt]

        def align(name):
            kk, vv = prod[name]
            out = torch.zeros_like(valO)
            if kk.numel():
                idx = torch.searchsorted(keyO, kk)
                out[idx] = vv
            return out
        i_loc, j = keyO // NB, keyO % NB
        i = i_loc + a
        dz = (zF_all[i] - zB_all[j]).abs()
        res = {"O": valO}
        S1 = align("S1") * (dz <= t1)
        res["O_surf_1"] = S1
        if need["S0"]:
            res["O_surf_0"] = align("S0") * (dz <= tols[0])
        if need["S2"]:
            res["O_surf_2"] = align("S2") * (dz <= tols[2])
        if "O_incompat" in cols_wanted:
            res["O_incompat"] = valO * (dz > t1)
        if need["N"]:
            res["npix"] = align("N")
        if need["NS"]:
            res["npix_surf"] = align("NS") * (dz <= t1)
        if "O_surf_rgb" in cols_wanted or "rgb_dist" in cols_wanted:
            rgbd = ((colF[i] - colB[j]) ** 2).mean(1).sqrt().double()
            if "O_surf_rgb" in cols_wanted:
                res["O_surf_rgb"] = S1 * (rgbd <= th.rgb_tol)
            if "rgb_dist" in cols_wanted:
                res["rgb_dist"] = valO * rgbd
        if "dist3d" in cols_wanted or "dist3d_surf" in cols_wanted:
            d3 = (muF[i] - muB[j]).norm(dim=1).double()
            if "dist3d" in cols_wanted:
                res["dist3d"] = valO * d3
            if "dist3d_surf" in cols_wanted:
                res["dist3d_surf"] = S1 * d3
        del prod
        mass += float(valO.sum())
        mass_surf += float(S1.sum())
        distinct += int(keyO.numel())
        keep = valO >= prune_tau if prune_tau > 0 else torch.ones_like(valO, dtype=torch.bool)
        drop = ~keep
        om["omitted_O_F"].index_add_(0, i[drop], valO[drop])
        om["omitted_O_B"].index_add_(0, j[drop], valO[drop])
        om["omitted_O_surf_F"].index_add_(0, i[drop], S1[drop])
        om["omitted_O_surf_B"].index_add_(0, j[drop], S1[drop])
        om_mass += float(valO[drop].sum())
        om_surf += float(S1[drop].sum())
        stored += int(keep.sum())
        keys_all.append((i[keep] * NB + j[keep]).cpu().numpy())
        for c in cols_wanted:
            # float32 after the float64 sums and the pruning decision: halves host RAM
            cols_all[c].append(res[c][keep].float().cpu().numpy())
        del res, keyO, valO, i, j, dz, S1, keep, drop
    keys = np.concatenate(keys_all) if keys_all else np.empty(0, np.int64)
    cols = {c: (np.concatenate(v) if v else np.empty(0)) for c, v in cols_all.items()}
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    cols = {c: v[order] for c, v in cols.items()}
    stats |= {"pair_mass": mass, "pair_mass_surf": mass_surf, "distinct_pairs": distinct,
              "stored_pairs": stored, "omitted_mass": om_mass, "omitted_mass_surf": om_surf}
    return {"keys": keys, "cols": cols, "stats": stats,
            **{k: v.cpu().numpy() for k, v in om.items()}}


def frame_table(views: list) -> dict:
    """Merge one frame's views [(camera_index, view_pairs result)]. Each key
    appears at most once per view, so a sum of distinct camera bits is their OR."""
    keys, cols = [], {c: [] for c in SUM_COLS}
    cams, cams_surf = [], []
    for ci, v in views:
        if v["keys"].size == 0:
            continue
        keys.append(v["keys"])
        for c in SUM_COLS:
            cols[c].append(v["cols"][c])
        cams.append(np.full(v["keys"].size, 1 << ci, np.int64))
        cams_surf.append((v["cols"]["O_surf_1"] > 0).astype(np.int64) << ci)
    if not keys:
        return {"keys": np.empty(0, np.int64), **{c: np.empty(0, np.float32) for c in SUM_COLS},
                "cams": np.empty(0, np.uint8), "cams_surf": np.empty(0, np.uint8)}
    allc = {c: np.concatenate(v) for c, v in cols.items()}
    allc["cams"] = np.concatenate(cams)
    allc["cams_surf"] = np.concatenate(cams_surf)
    k, red = _reduce_np(np.concatenate(keys), allc)
    out = {"keys": k}
    for c in SUM_COLS:
        out[c] = red[c].astype(np.float32)
    out["cams"] = red["cams"].astype(np.uint8)
    out["cams_surf"] = red["cams_surf"].astype(np.uint8)
    return out


_GCOLS = SUM_COLS + ("frames", "frames_surf")


class _Shard:
    def __init__(self):
        self.t = None

    def add(self, frame: int, ft: dict, lo: int, hi: int):
        n = hi - lo
        if n <= 0:
            return
        sl = slice(lo, hi)
        surf = ft["O_surf_1"][sl] > 0
        new = {"keys": ft["keys"][sl], **{c: ft[c][sl] for c in SUM_COLS},
               "cams": ft["cams"][sl], "cams_surf": ft["cams_surf"][sl],
               "frames": np.ones(n, np.int16), "frames_surf": surf.astype(np.int16),
               "first": np.full(n, frame, np.int16), "last": np.full(n, frame, np.int16),
               "first_surf": np.where(surf, frame, 9999).astype(np.int16),
               "last_surf": np.where(surf, frame, -1).astype(np.int16),
               "best_frame_O_surf": np.where(surf, ft["O_surf_1"][sl], 0).astype(np.float32),
               "best_frame": np.where(surf, frame, -1).astype(np.int16)}
        if self.t is None:
            self.t = new
            return
        cat = {k: np.concatenate([self.t[k], new[k]]) for k in new}
        self.t = None
        order = np.argsort(cat["keys"], kind="stable")
        for k in cat:
            cat[k] = cat[k][order]
        del order
        keys = cat["keys"]
        st = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        out = {"keys": keys[st]}
        for c in _GCOLS:
            out[c] = np.add.reduceat(cat[c], st).astype(cat[c].dtype)
        for c in ("cams", "cams_surf"):
            out[c] = np.bitwise_or.reduceat(cat[c], st)
        for c, fn in (("first", np.minimum), ("first_surf", np.minimum),
                      ("last", np.maximum), ("last_surf", np.maximum)):
            out[c] = fn.reduceat(cat[c], st)
        # frame of the largest same-surface overlap (ties: the earlier row)
        bo = cat["best_frame_O_surf"]
        seg = np.repeat(np.arange(st.size), np.diff(np.r_[st, bo.size]))
        order = np.lexsort((-np.arange(bo.size), bo, seg))
        pick = order[np.r_[st[1:], bo.size] - 1]
        out["best_frame_O_surf"] = bo[pick]
        out["best_frame"] = cat["best_frame"][pick]
        self.t = out


class GlobalPairs:
    """Sequence-wide merge, sharded by F row so a merge's temporaries stay
    one shard large. Exact sums of the stored per-view values."""

    def __init__(self, NF: int, NB: int, shards: int = 16):
        self.NB = NB
        self.bounds = np.linspace(0, NF, shards + 1).astype(np.int64) * NB
        self.shards = [_Shard() for _ in range(shards)]

    def add(self, frame: int, ft: dict):
        cut = np.searchsorted(ft["keys"], self.bounds)
        for s, sh in enumerate(self.shards):
            sh.add(frame, ft, int(cut[s]), int(cut[s + 1]))

    def rows(self) -> int:
        return sum(0 if s.t is None else s.t["keys"].size for s in self.shards)

    def nbytes(self) -> int:
        return sum(0 if s.t is None else sum(v.nbytes for v in s.t.values()) for s in self.shards)

    def table(self) -> dict:
        parts = [s.t for s in self.shards if s.t is not None]
        if not parts:
            return {}
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def save(self, path):
        np.savez(path, bounds=self.bounds, **{f"s{i:02d}_{k}": v for i, s in enumerate(self.shards)
                                              if s.t is not None for k, v in s.t.items()})

    def load(self, path):
        d = np.load(path)
        if not np.array_equal(d["bounds"], self.bounds):
            raise ValueError("checkpoint shard bounds differ")
        for i, s in enumerate(self.shards):
            ks = [k for k in d.files if k.startswith(f"s{i:02d}_")]
            s.t = {k[4:]: d[k] for k in ks} if ks else None


def popcount(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(np.int64)
    c = np.zeros_like(x)
    while np.any(x):
        c += x & 1
        x >>= 1
    return c
