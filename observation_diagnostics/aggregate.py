"""Sequence-level aggregation: tables, Gaussian labels, candidate patches,
the initially-occluded test, case studies, galleries and the reports.

Everything here reads the per-frame records written by the coverage and pair
passes (tools/analyze_fb_pixel_overlap.py); renders are redone only for the
nominated case studies.

Gaussian labels are HYPOTHESES from declared thresholds (LABEL_THRESHOLDS,
swept in metrics/threshold_sensitivity_labels.csv), never ground truth:

  NEVER_VISIBLE             no composited contribution in any processed view
  UNRESOLVED_LOW_EVIDENCE   supported mass < S_MIN: too little observation to say
  LIKELY_COMMON_SURFACE     best partner rho_surf >= RHO_STRONG, seen on the same
                            surface in >= MIN_FRAMES frames, no comparable rival
  AMBIGUOUS                 rho_surf >= RHO_STRONG but a second partner within
                            AMBIG_RATIO of the best
  WEAK_COMMON_SURFACE       RHO_WEAK <= rho_surf < RHO_STRONG, or strong but in
                            fewer than MIN_FRAMES frames
  OVERLAPPING_DIFFERENT_DEPTH  rho_surf < RHO_WEAK, but a partner shares its
                            pixels (rho_pix >= RHO_STRONG) mostly at another depth
  NO_CREDIBLE_MATCH         rho_surf < RHO_WEAK otherwise

rho_surf_ij = O_surf_ij / sqrt(Q2F_i Q2B_j), Q2 = sum q^2 over the Gaussian's
GT-depth-compatible contributions: a cosine of the two same-surface
contribution maps, in [0, 1] by Cauchy-Schwarz, 1 for identical maps.
rho_pix uses O_ij and sum q^2 over all contributions.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from . import gaussian_evidence as ge
from . import pair_evidence as pv
from . import visualize as vz

LABEL_THRESHOLDS = {"S_MIN": 1.0, "RHO_STRONG": 0.3, "RHO_WEAK": 0.1,
                    "AMBIG_RATIO": 0.7, "MIN_FRAMES": 3, "INCOMPAT_FRACTION": 0.5,
                    "U_MIN": 1.0, "MAX_GROUPS": 30, "CASE_STUDY_GROUPS": 12}
CASE_FRAMES = (0, 10, 20, 30, 40, 50, 59)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _flat(row: dict, prefix="") -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, dict):
            out |= _flat(v, f"{prefix}{k}.")
        elif isinstance(v, list):
            out[prefix + k] = json.dumps(v)
        else:
            out[prefix + k] = v
    return out


def _write_csv(path: Path, rows: list):
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _pct(a, b):
    return None if not b else round(100.0 * a / b, 3)


COUNT_KEYS = None


def _count_keys(row):
    return [k for k, v in row.items() if isinstance(v, int) and not isinstance(v, bool)
            and k not in ("frame",)]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def load_coverage(out: Path, frames, fingerprint):
    views, missing = [], []
    for f in frames:
        p = out / "coverage" / f"frame_{f:04d}.json"
        if not p.is_file() or json.loads(p.read_text()).get("fingerprint") != fingerprint:
            missing.append(f)
            continue
        views += json.loads(p.read_text())["views"]
    return views, missing


def per_frame_rows(views):
    by = {}
    for v in views:
        by.setdefault(v["frame"], []).append(v)
    rows = []
    for f in sorted(by):
        vs = by[f]
        r = {"frame": f, "cameras": len(vs)}
        for k in _count_keys(vs[0]):
            r[k] = sum(v[k] for v in vs)
        for s in ("F", "B"):
            # weighted by the pixels each statistic was computed on
            ncov = [v["BOTH_COVER__on_gt_valid"] + v[f"{s}_ONLY_COVER__on_gt_valid"] for v in vs]
            for stat in ("depth_mae_m__covered",):
                vals = [(n, v[f"{s}_{stat}"]) for n, v in zip(ncov, vs) if v[f"{s}_{stat}"] is not None]
                r[f"{s}_{stat}"] = (sum(n * x for n, x in vals) / max(sum(n for n, _ in vals), 1)
                                    if vals else None)
            vals = [(n, v[f"{s}_depth_rmse_m__covered"]) for n, v in zip(ncov, vs)
                    if v[f"{s}_depth_rmse_m__covered"] is not None]
            r[f"{s}_depth_rmse_m__covered"] = (
                (sum(n * x * x for n, x in vals) / max(sum(n for n, _ in vals), 1)) ** 0.5
                if vals else None)
            vals = [(v["gt_valid"], v[f"{s}_rgb_rmse__all_valid"]) for v in vs
                    if v[f"{s}_rgb_rmse__all_valid"] is not None]
            r[f"{s}_rgb_rmse__all_valid"] = (
                (sum(n * x * x for n, x in vals) / max(sum(n for n, _ in vals), 1)) ** 0.5
                if vals else None)
            # production gate: weight by GT-valid pixels (the gate support is not stored)
            vals = [(v["gt_valid"], v[f"{s}_depth_mae_m__production_gate"]) for v in vs
                    if v[f"{s}_depth_mae_m__production_gate"] is not None]
            r[f"{s}_depth_mae_m__production_gate"] = (
                sum(n * x for n, x in vals) / max(sum(n for n, _ in vals), 1) if vals else None)
        rows.append(r)
    return rows


def sweep_table(out: Path, frames):
    acc = {}
    for f in frames:
        p = out / "coverage" / f"frame_{f:04d}_sweep.json"
        if not p.is_file():
            continue
        for r in json.loads(p.read_text()):
            key = (r["alpha_cover"], r["depth_tol_m"], r["rgb_tol"])
            a = acc.setdefault(key, {})
            for k, v in r.items():
                if k in ("frame", "camera", "alpha_cover", "depth_tol_m", "rgb_tol"):
                    continue
                a[k] = a.get(k, 0) + v
    rows = []
    for (a, d, r), vals in sorted(acc.items()):
        row = {"alpha_cover": a, "depth_tol_m": d, "rgb_tol": r, "gt_valid_pixels": vals["gt_valid"]}
        for k, v in vals.items():
            if k != "gt_valid":
                row[k] = v
                row[k + "__pct_of_gt_valid"] = _pct(v, vals["gt_valid"])
        rows.append(row)
    return rows


def load_gaussian_arrays(out: Path, frames, N_F, N_B):
    cols = list(ge.COLUMNS) + ["cams_visible", "cams_supported", "cams_witness"]
    arr = {s: {c: np.zeros((len(frames), n), np.float32 if c in ge.COLUMNS else np.int32)
               for c in cols} for s, n in (("F", N_F), ("B", N_B))}
    for k, f in enumerate(frames):
        d = np.load(out / "coverage" / f"frame_{f:04d}.npz")
        for s in ("F", "B"):
            for c in cols:
                arr[s][c][k] = d[f"{s}_{c}"]
    return arr


def _bit_or(x):
    o = np.zeros(x.shape[1], np.int64)
    for row in x:
        o |= row.astype(np.int64)
    return o


def _first_last(mask2d, frames):
    fr = np.asarray(frames)
    any_ = mask2d.any(0)
    first = np.where(any_, fr[np.argmax(mask2d, axis=0)], -1)
    last = np.where(any_, fr[mask2d.shape[0] - 1 - np.argmax(mask2d[::-1], axis=0)], -1)
    return first, last


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------

def merge_pairs(out: Path, frames, fingerprint, NF, NB, log):
    """The pair pass's sequence table, its per-frame stats and the omitted mass."""
    st_path, final = out / "pairs" / "checkpoint.json", out / "pairs" / "global_final.npz"
    stats, missing = [], list(frames)
    om = {"omitted_O_F": np.zeros(NF), "omitted_O_surf_F": np.zeros(NF),
          "omitted_O_B": np.zeros(NB), "omitted_O_surf_B": np.zeros(NB)}
    if not (st_path.is_file() and final.is_file()):
        return {}, missing, stats, om
    st = json.loads(st_path.read_text())
    if st.get("fingerprint") != fingerprint or not st.get("complete"):
        return {}, missing, stats, om
    t0 = time.time()
    gp = pv.GlobalPairs(NF, NB)
    gp.load(final)
    t = gp.table()
    missing = [f for f in frames if f not in st["frames_merged"]]
    for f in frames:
        rec = out / "pairs" / f"frame_{f:04d}.json"
        if rec.is_file():
            stats.append(json.loads(rec.read_text()))
        o = out / "pairs" / f"frame_{f:04d}_omitted.npz"
        if o.is_file():
            d = np.load(o)
            for k in om:
                om[k] += d[k]
    log(f"  pair table: {t['keys'].size if t else 0:,} stored (F,B) pairs over "
        f"{len(frames) - len(missing)} frames, loaded in {time.time() - t0:.0f} s")
    return t, missing, stats, om


def best_partners(keys_side, other, rho_surf, rho_pix, incompat_frac, frames_surf, n_side):
    """Per Gaussian of one side: best and second partner by rho_surf; best by rho_pix."""
    order = np.lexsort((-rho_surf, keys_side))
    ks, rs = keys_side[order], rho_surf[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    ends = np.r_[starts[1:], ks.size]
    res = {"best": np.full(n_side, -1, np.int64), "rho_surf": np.zeros(n_side),
           "second_rho_surf": np.zeros(n_side), "frames_surf": np.zeros(n_side, np.int32),
           "partners": np.zeros(n_side, np.int32),
           "partners_comparable": np.zeros(n_side, np.int32),
           "best_pix": np.full(n_side, -1, np.int64), "rho_pix": np.zeros(n_side),
           "best_pix_incompat_frac": np.zeros(n_side)}
    g = ks[starts]
    res["best"][g] = other[order][starts]
    res["rho_surf"][g] = rs[starts]
    has2 = ends - starts > 1
    res["second_rho_surf"][g[has2]] = rs[starts[has2] + 1]
    res["frames_surf"][g] = frames_surf[order][starts]
    res["partners"][g] = ends - starts
    best_of = np.repeat(rs[starts], ends - starts)
    comp = (rs >= 0.5 * best_of) & (best_of > 0)
    res["partners_comparable"][g] = np.add.reduceat(comp.astype(np.int32), starts)
    return _pix_best(res, keys_side, other, rho_pix, incompat_frac)


def certify(res, keys_side, other, osurf, omitted, n_side):
    """Best partner by the stored O_surf, and whether pruning could change it:
    any pair's missing same-surface mass is at most the row's omitted mass."""
    order = np.lexsort((-osurf, keys_side))
    ks, os_ = keys_side[order], osurf[order]
    st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    en = np.r_[st[1:], ks.size]
    g = ks[st]
    res["best_by_O"] = np.full(n_side, -1, np.int64)
    res["O_surf_best"] = np.zeros(n_side)
    res["O_surf_second"] = np.zeros(n_side)
    res["best_by_O"][g] = other[order][st]
    res["O_surf_best"][g] = os_[st]
    two = en - st > 1
    res["O_surf_second"][g[two]] = os_[st[two] + 1]
    res["omitted_O_surf"] = omitted
    res["certified"] = (res["O_surf_best"] - res["O_surf_second"]) >= omitted
    tot = np.bincount(keys_side, weights=osurf, minlength=n_side)
    res["omitted_fraction"] = np.divide(omitted, omitted + tot, out=np.zeros(n_side),
                                        where=(omitted + tot) > 0)
    return res


def _pix_best(res, keys_side, other, rho_pix, incompat_frac):
    order2 = np.lexsort((-rho_pix, keys_side))
    k2 = keys_side[order2]
    s2 = np.flatnonzero(np.r_[True, k2[1:] != k2[:-1]])
    g2 = k2[s2]
    res["best_pix"][g2] = other[order2][s2]
    res["rho_pix"][g2] = rho_pix[order2][s2]
    res["best_pix_incompat_frac"][g2] = incompat_frac[order2][s2]
    return res


def label(total, supported, bp, th):
    n = total.size
    lab = np.full(n, "NO_CREDIBLE_MATCH", dtype=object)
    strong = bp["rho_surf"] >= th["RHO_STRONG"]
    weak = (bp["rho_surf"] >= th["RHO_WEAK"]) & ~strong
    ratio = np.divide(bp["second_rho_surf"], bp["rho_surf"],
                      out=np.zeros(n), where=bp["rho_surf"] > 0)
    persistent = bp["frames_surf"] >= th["MIN_FRAMES"]
    lab[weak | (strong & ~persistent)] = "WEAK_COMMON_SURFACE"
    lab[strong & persistent & (ratio < th["AMBIG_RATIO"])] = "LIKELY_COMMON_SURFACE"
    lab[strong & persistent & (ratio >= th["AMBIG_RATIO"])] = "AMBIGUOUS"
    nomatch = bp["rho_surf"] < th["RHO_WEAK"]
    odd = nomatch & (bp["rho_pix"] >= th["RHO_STRONG"]) & \
        (bp["best_pix_incompat_frac"] >= th["INCOMPAT_FRACTION"])
    lab[odd] = "OVERLAPPING_DIFFERENT_DEPTH"
    lab[supported < th["S_MIN"]] = "UNRESOLVED_LOW_EVIDENCE"
    lab[total <= 0] = "NEVER_VISIBLE"
    return lab, ratio


# ---------------------------------------------------------------------------
# geometry helpers for occlusion and case studies
# ---------------------------------------------------------------------------

def project(spec, xyz):
    """Pixel coordinates and view depth, the renderer's own projection."""
    import torch
    p = torch.as_tensor(xyz, dtype=torch.float32, device=spec.viewmatrix.device)
    ph = torch.cat([p, torch.ones_like(p[:, :1])], 1)
    hom = ph @ spec.projmatrix
    w = 1.0 / (hom[:, 3:4] + 1e-7)
    ndc = hom[:, :2] * w
    view = ph @ spec.viewmatrix
    u = ((ndc[:, 0] + 1.0) * spec.image_width - 1.0) * 0.5
    v = ((ndc[:, 1] + 1.0) * spec.image_height - 1.0) * 0.5
    return u.cpu().numpy(), v.cpu().numpy(), view[:, 2].cpu().numpy()


def occlusion_status(ctx, members, B, t_first, tol):
    """Why is the group not witnessed before t_first? Decided from the GT
    depth at the members' projections, per camera, at sampled early frames."""
    from .sources import load_gt
    out = {"t_first_witness": int(t_first), "frames": []}
    if t_first <= ctx["frames"][0]:
        out["status"] = "WITNESSED_FROM_FIRST_FRAME"
        return out
    early = [f for f in ctx["frames"] if f < t_first]
    early = sorted(set(early[:: max(1, len(early) // 6)] + [early[-1]]))
    votes = {"OCCLUDED": 0, "VISIBLE_AT_OBSERVED_DEPTH": 0, "OUT_OF_VIEW": 0, "GT_INVALID": 0,
             "BEHIND_OBSERVED_SURFACE_GAP": 0}
    for f in early:
        xyz = B.mu[B.frames.index(f)][members]
        per_cam = {}
        for cam in ctx["cams"]:
            spec = ctx["specs"][ctx["all_cams"].index(cam)]
            u, v, z = project(spec, xyz)
            H, W = spec.image_height, spec.image_width
            inside = (z > 0.01) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
            if not inside.any():
                per_cam[cam] = {"in_fov": 0}
                votes["OUT_OF_VIEW"] += 1
                continue
            _r, d, valid, _ = load_gt(ctx["data"], cam, f, H, W, device="cpu")
            d, valid = d.numpy(), valid.numpy()
            ui = np.clip(np.round(u[inside]).astype(int), 0, W - 1)
            vi = np.clip(np.round(v[inside]).astype(int), 0, H - 1)
            dg, vg, zz = d[vi, ui], valid[vi, ui], z[inside]
            occ = vg & (dg < zz - tol)
            vis = vg & (np.abs(dg - zz) <= tol)
            gap = vg & (dg > zz + tol)
            inval = ~vg
            c = {"in_fov": int(inside.sum()), "gt_in_front": int(occ.sum()),
                 "gt_at_member_depth": int(vis.sum()), "gt_behind": int(gap.sum()),
                 "gt_invalid": int(inval.sum())}
            per_cam[cam] = c
            k = max(("OCCLUDED", c["gt_in_front"]), ("VISIBLE_AT_OBSERVED_DEPTH", c["gt_at_member_depth"]),
                    ("BEHIND_OBSERVED_SURFACE_GAP", c["gt_behind"]), ("GT_INVALID", c["gt_invalid"]),
                    key=lambda x: x[1])[0]
            votes[k] += 1
        out["frames"].append({"frame": f, "cameras": per_cam})
    out["votes_over_early_frame_cameras"] = votes
    in_view = sum(v for k, v in votes.items() if k != "OUT_OF_VIEW")
    if in_view == 0:
        out["status"] = "OUT_OF_VIEW"
    elif votes["OCCLUDED"] >= 0.6 * in_view:
        out["status"] = "OCCLUSION_HYPOTHESIS"
    elif votes["VISIBLE_AT_OBSERVED_DEPTH"] >= 0.5 * in_view:
        out["status"] = "WEAK_EVIDENCE"  # observed at its depth yet not exclusive: F explains it too, or B did not cover
    else:
        out["status"] = "UNDETERMINED"
    out["rule"] = ("per (early frame, camera) the majority GT relation at the members' "
                   f"projected pixels with tol {tol} m; OCCLUSION_HYPOTHESIS if >= 60% of "
                   "in-view votes see GT in front of the members, WEAK_EVIDENCE if >= 50% see "
                   "GT at their depth, OUT_OF_VIEW if no camera sees them, else UNDETERMINED. "
                   "A hypothesis from GT depth, not a proof of physical occlusion.")
    return out


def _render_pair(ctx, f, cam):
    from .renderer_access import pixel_contributions
    from .sources import load_gt
    from estimator.renderer_bridge import build_linearization
    spec = ctx["specs"][ctx["all_cams"].index(cam)]
    out = {}
    for key, src in (("F", ctx["F"]), ("B", ctx["B"])):
        st = src.state(f)
        v = build_linearization(st, [spec]).views[0]
        out[key] = (v, pixel_contributions(v.cache), st)
    rgb, d, valid, _ = load_gt(ctx["data"], cam, f, spec.image_height, spec.image_width)
    out["gt"] = (rgb, d, valid)
    return out


def _member_mass(contrib, members_t, HW):
    import torch
    sel = torch.isin(contrib.gid.long(), members_t)
    pix = contrib.pixel[sel]
    return torch.zeros(HW, dtype=torch.float64, device=pix.device).index_add_(
        0, pix, contrib.q[sel].double())


def _crop_images(r, mass_map, box, th, drange):
    import torch
    (vF, _cF, _), (vB, _cB, _) = r["F"], r["B"]
    rgb, d, valid = r["gt"]
    y0, y1, x0, x1 = box
    gn = {"rgb": rgb.cpu().numpy(), "depth": d.cpu().numpy(), "valid": valid.cpu().numpy()}
    eF = np.where(gn["valid"], vF.depth.cpu().numpy() - gn["depth"], 0)
    eB = np.where(gn["valid"], vB.depth.cpu().numpy() - gn["depth"], 0)
    aF, aB = vF.alpha.cpu().numpy(), vB.alpha.cpu().numpy()
    H, W = gn["depth"].shape
    m = mass_map.reshape(H, W).cpu().numpy() if isinstance(mass_map, torch.Tensor) else mass_map
    base = vz._to_u8(gn["rgb"].transpose(1, 2, 0)).astype(np.float32) * 0.45
    over = np.clip(base + 255.0 * np.clip(m / max(m.max(), 1e-6), 0, 1)[..., None], 0, 255)
    imgs = {"GT_RGB": vz._to_u8(gn["rgb"].transpose(1, 2, 0)),
            "GT_depth": vz.depth_rgb(gn["depth"], gn["valid"], drange),
            "F_RGB": vz._to_u8(vF.rgb.cpu().numpy().transpose(1, 2, 0)),
            "F_depth": vz.depth_rgb(vF.depth.cpu().numpy(), aF > 0.05, drange),
            "B_RGB": vz._to_u8(vB.rgb.cpu().numpy().transpose(1, 2, 0)),
            "B_depth": vz.depth_rgb(vB.depth.cpu().numpy(), aB > 0.05, drange),
            "F_depth_residual": vz.residual_rgb(eF, gn["valid"] & (aF >= th.alpha_cover)),
            "B_depth_residual": vz.residual_rgb(eB, gn["valid"] & (aB >= th.alpha_cover)),
            "contribution": over.astype(np.uint8)}
    return {k: v[y0:y1, x0:x1] for k, v in imgs.items()}


def _box(u, v, H, W, margin=24):
    if u.size == 0:
        return (0, H, 0, W)
    x0, x1 = int(max(np.percentile(u, 2) - margin, 0)), int(min(np.percentile(u, 98) + margin, W))
    y0, y1 = int(max(np.percentile(v, 2) - margin, 0)), int(min(np.percentile(v, 98) + margin, H))
    # at least 96 px, same aspect for every frame of the study
    if x1 - x0 < 96:
        c = (x0 + x1) // 2
        x0, x1 = max(c - 48, 0), min(c + 48, W)
    if y1 - y0 < 96:
        c = (y0 + y1) // 2
        y0, y1 = max(c - 48, 0), min(c + 48, H)
    return (y0, y1, x0, x1)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def run(ctx, log, skip_report=False) -> dict:
    import torch
    out, F, B, th = ctx["out"], ctx["F"], ctx["B"], ctx["th"]
    frames, fp = ctx["frames"], ctx["fingerprint"]
    met, gdir, mdir, vdir = (out / "metrics", out / "gaussians", out / "matches",
                             out / "visuals")
    for d in (met, gdir, mdir, vdir):
        d.mkdir(parents=True, exist_ok=True)
    LT = LABEL_THRESHOLDS
    T_AGG = time.time()
    summary = {"scene": ctx["scene"], "frames": [frames[0], frames[-1]],
               "num_frames": len(frames), "cameras": ctx["cams"],
               "views_per_source": len(frames) * len(ctx["cams"]),
               "N_F": F.N, "N_B": B.N, "state_mode": {"F": F.state_mode, "B": B.state_mode},
               "thresholds": th.as_dict(), "label_thresholds": LT,
               "pilot": bool(frames != list(range(len(F.frames))) or
                             ctx["cams"] != ctx["all_cams"])}

    # ---- coverage tables ------------------------------------------------
    views, missing = load_coverage(out, frames, fp)
    summary["coverage_frames_missing"] = missing
    if missing:
        log(f"  coverage incomplete: {len(missing)} frames missing; aggregation stops here")
        summary["status"] = "COVERAGE_INCOMPLETE"
        (met / "global_summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return summary
    _write_csv(met / "per_frame_camera.csv", [_flat(v) for v in views])
    pf = per_frame_rows(views)
    _write_csv(met / "per_frame.csv", pf)
    sweep = sweep_table(out, frames)
    _write_csv(met / "threshold_sensitivity.csv", sweep)
    keys = _count_keys(views[0])
    tot = {k: sum(v[k] for v in views) for k in keys}
    nv = tot["gt_valid"]
    summary["pixel_totals"] = tot
    summary["pixel_pct_of_gt_valid"] = {k: _pct(tot[k], nv) for k in keys
                                        if k not in ("pixels_total", "gt_invalid", "gt_valid")}
    per_cam = {}
    for cam in ctx["cams"]:
        vs = [v for v in views if v["camera"] == cam]
        n = sum(v["gt_valid"] for v in vs)
        per_cam[cam] = {"gt_valid": n, **{k: _pct(sum(v[k] for v in vs), n) for k in (
            "BOTH_COVER__on_gt_valid", "F_ONLY_COVER__on_gt_valid", "B_ONLY_COVER__on_gt_valid",
            "NEITHER_COVERS__on_gt_valid", "BOTH_VALID_SUPPORT", "F_SUPPORTED_B_UNEXPLAINED",
            "B_SUPPORTED_F_UNEXPLAINED", "NEITHER_VALID_SUPPORT",
            "BOTH_SAME_OBSERVED_SURFACE_CANDIDATE", "BOTH_COVER_DIFFERENT_DEPTH")}}
    summary["per_camera_pct_of_gt_valid"] = per_cam
    both_cover = tot["BOTH_COVER__on_gt_valid"]
    summary["of_both_cover_gt_valid_pct"] = {
        k: _pct(tot[k], both_cover) for k in ("BOTH_SAME_OBSERVED_SURFACE_CANDIDATE",
                                              "BOTH_COVER_DIFFERENT_DEPTH",
                                              "BOTH_COVER_AGREE_OFF_OBSERVED_SURFACE")}
    # contributor statistics and T1 on real data
    cs = {}
    for s in ("F", "B"):
        cons = [v[f"{s}_conservation"] for v in views]
        con = [v[f"{s}_contributors"] for v in views]
        cs[s] = {"records_total": sum(c.get("records", 0) for c in con),
                 "pixels_with_contributors_total": sum(c.get("pixels_with_contributors", 0) for c in con),
                 "per_pixel_p50_median_over_views": float(np.median([c.get("per_pixel_p50", 0) for c in con])),
                 "per_pixel_p99_max_over_views": max(c.get("per_pixel_p99", 0) for c in con),
                 "per_pixel_max": max(c.get("per_pixel_max", 0) for c in con),
                 "early_terminated_pixels_total": sum(c.get("early_terminated_pixels", 0) for c in con),
                 "T1_max_abs_sum_q_minus_A": max(c["max_abs_sum_q_minus_A"] for c in cons),
                 "T1_max_abs_depth_reconstruction_m": max(c["max_abs_depth_recon_on_alpha_gt_0.05"] for c in cons),
                 "T1_final_T_bitwise_all_views": all(c["final_T_bitwise_equal"] for c in cons),
                 "T1_records_le_n_contrib_all_views": all(c["records_le_n_contrib"] for c in cons)}
    (met / "contributor_statistics.json").write_text(json.dumps(cs, indent=1))
    summary["contributors"] = cs
    vz.timeline_plots(vdir, pf, ctx["scene"])
    log(f"  coverage tables: {len(views)} views, GT-valid pixels {nv:,}")

    # ---- per-Gaussian arrays --------------------------------------------
    log(f"  [{time.time() - T_AGG:6.1f} s] per-Gaussian arrays")
    G = load_gaussian_arrays(out, frames, F.N, B.N)
    tot_g = {s: {c: G[s][c].sum(0).astype(np.float64) for c in ge.COLUMNS} for s in G}
    np.savez_compressed(gdir / "per_frame_curves.npz", frames=np.asarray(frames),
                        **{f"{s}_{c}": G[s][c] for s in G for c in
                           ("mass_total", "mass_supported", "witness", "witness_improve",
                            "cams_supported", "cams_witness")})

    # ---- pairs ----------------------------------------------------------
    log(f"  [{time.time() - T_AGG:6.1f} s] pairs")
    table, pmissing, pstats, omitted = merge_pairs(out, frames, fp, F.N, B.N, log)
    pairs_complete = not pmissing and bool(table)
    views_st = [v for s_ in pstats for v in s_["views"]]
    ref_path = out / "pairs" / "exact_reference.json"
    prune = ctx.get("pair_prune", 0.0)
    pst = {"exactness": ("EXACT" if prune == 0 else
                         "APPROXIMATE_BOUNDED: every co-contributor pair enumerated and summed "
                         "exactly per view; a view's pair stored only if its per-view O >= "
                         f"{prune}; the pruned mass is accounted per Gaussian (omitted_*), so a "
                         "stored O is a lower bound and a best partner is certified when "
                         "best - second >= the row's omitted same-surface mass"),
           "prune_tau": prune,
           "frames_processed": len(frames) - len(pmissing), "frames_requested": len(frames),
           "frames_missing": pmissing, "views_processed": len(views_st),
           "pair_records_total": int(sum(v["pairs"] for v in views_st)),
           "pair_mass_total": float(sum(v.get("pair_mass", 0) for v in views_st)),
           "pair_mass_same_surface_primary": float(sum(v.get("pair_mass_surf", 0) for v in views_st)),
           "omitted_mass_total": float(sum(v.get("omitted_mass", 0) for v in views_st)),
           "omitted_mass_same_surface": float(sum(v.get("omitted_mass_surf", 0) for v in views_st)),
           "view_level_distinct_pairs_total": int(sum(v.get("distinct_pairs", 0) for v in views_st)),
           "view_level_stored_pairs_total": int(sum(v.get("stored_pairs", 0) for v in views_st)),
           "pixels_with_both_contributors_total": int(sum(v["pixels_with_both_contributors"]
                                                          for v in views_st)),
           "distinct_pairs_stored_sequence": int(table["keys"].size) if table else 0,
           "pass_seconds": float(sum(s_["seconds"] for s_ in pstats)),
           "exact_reference_frame": (json.loads(ref_path.read_text()) if ref_path.is_file() else None),
           "pair_tolerances_m": list(ge.PAIR_TOLS_M), "primary_tolerance_m": ge.PAIR_TOLS_M[pv.PRIMARY]}
    if pst["pair_mass_total"]:
        pst["omitted_fraction_of_pair_mass"] = pst["omitted_mass_total"] / pst["pair_mass_total"]
        pst["omitted_fraction_of_same_surface_mass"] = (
            pst["omitted_mass_same_surface"] / max(pst["pair_mass_same_surface_primary"], 1e-30))
    bpF = bpB = None
    labF = labB = None
    if pairs_complete:
        k = table["keys"]
        iF, jB = k // B.N, k % B.N
        q2F, q2B = tot_g["F"]["q2_kappa_1"], tot_g["B"]["q2_kappa_1"]
        rho_surf = table["O_surf_1"] / np.sqrt(np.maximum(q2F[iF] * q2B[jB], 1e-30))
        rho_pix = table["O"] / np.sqrt(np.maximum(tot_g["F"]["q2_all"][iF] * tot_g["B"]["q2_all"][jB], 1e-30))
        incompat = np.divide(table["O_incompat"], table["O"], out=np.zeros_like(table["O"]),
                             where=table["O"] > 0)
        pst["rho_surf_max"] = float(rho_surf.max()) if rho_surf.size else 0.0
        pst["cauchy_schwarz_violations_rho_surf_gt_1"] = int((rho_surf > 1 + 1e-4).sum())
        np.savez(mdir / "pair_scores.npz", F_row=iF.astype(np.int32), B_row=jB.astype(np.int32),
                 rho_surf=rho_surf.astype(np.float32), rho_pix=rho_pix.astype(np.float32))
        bpF = best_partners(iF, jB, rho_surf, rho_pix, incompat, table["frames_surf"], F.N)
        bpB = best_partners(jB, iF, rho_surf, rho_pix, incompat, table["frames_surf"], B.N)
        for bp, side, ks, oth, n in ((bpF, "F", iF, jB, F.N), (bpB, "B", jB, iF, B.N)):
            certify(bp, ks, oth, table["O_surf_1"], omitted[f"omitted_O_surf_{side}"], n)
            has = bp["O_surf_best"] > 0
            pst[f"{side}_rows_with_same_surface_partner"] = int(has.sum())
            pst[f"{side}_best_by_O_surf_certified_fraction"] = (
                float(bp["certified"][has].mean()) if has.any() else None)
            ev = has & (tot_g[side]["mass_supported"] >= LT["S_MIN"])
            pst[f"{side}_best_by_O_surf_certified_fraction__supported_ge_S_MIN"] = (
                float(bp["certified"][ev].mean()) if ev.any() else None)
            pst[f"{side}_best_by_rho_equals_best_by_O_surf_fraction"] = (
                float((bp["best"][has] == bp["best_by_O"][has]).mean()) if has.any() else None)
            pst[f"{side}_median_omitted_fraction_of_same_surface"] = (
                float(np.median(bp["omitted_fraction"][has])) if has.any() else None)
        labF, ratioF = label(tot_g["F"]["mass_total"], tot_g["F"]["mass_supported"], bpF, LT)
        labB, ratioB = label(tot_g["B"]["mass_total"], tot_g["B"]["mass_supported"], bpB, LT)
        # presentation table: each Gaussian's best partner + every pair with rho_surf >= RHO_WEAK
        import pandas as pd
        # top-20 partners per F Gaussian by rho_surf (rho_surf >= RHO_WEAK), plus every
        # Gaussian's best partner on either side
        o = np.lexsort((-rho_surf, iF))
        st_ = np.flatnonzero(np.r_[True, iF[o][1:] != iF[o][:-1]])
        rank = np.arange(o.size) - np.repeat(st_, np.diff(np.r_[st_, o.size]))
        keep = np.zeros(k.size, bool)
        keep[o[rank < 20]] = True
        keep &= rho_surf >= LT["RHO_WEAK"]
        okF, okB = bpF["best"] >= 0, bpB["best"] >= 0
        best_keys = np.concatenate([np.flatnonzero(okF) * B.N + bpF["best"][okF],
                                    bpB["best"][okB] * B.N + np.flatnonzero(okB)])
        keep |= np.isin(k, best_keys)
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(-rho_surf[idx], kind="stable")]
        O = table["O"][idx].astype(np.float64)
        Os = table["O_surf_1"][idx].astype(np.float64)
        fs = table["frames_surf"][idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            df = pd.DataFrame({
                "F_row": iF[idx], "B_row": jB[idx],
                "rho_surf": np.round(rho_surf[idx], 5), "rho_pix": np.round(rho_pix[idx], 5),
                "O": O, "O_surf_2cm": table["O_surf_0"][idx], "O_surf_4cm": Os,
                "O_surf_8cm": table["O_surf_2"][idx], "O_incompat": table["O_incompat"][idx],
                "O_surf_rgb_compatible": table["O_surf_rgb"][idx],
                "mean_rgb_dist": np.where(O > 0, table["rgb_dist"][idx] / O, np.nan),
                "mean_center_dist_m": np.where(O > 0, table["dist3d"][idx] / O, np.nan),
                "mean_center_dist_same_surface_m": np.where(Os > 0, table["dist3d_surf"][idx] / Os, np.nan),
                "shared_pixels": table["npix"][idx].astype(np.int64),
                "shared_same_surface_pixels": table["npix_surf"][idx].astype(np.int64),
                "frames": table["frames"][idx], "frames_same_surface": fs,
                "cams": pv.popcount(table["cams"][idx]),
                "cams_same_surface": pv.popcount(table["cams_surf"][idx]),
                "first_frame": table["first"][idx], "last_frame": table["last"][idx],
                "first_same_surface_frame": np.where(fs > 0, table["first_surf"][idx], -1),
                "last_same_surface_frame": np.where(fs > 0, table["last_surf"][idx], -1),
                "best_frame": table["best_frame"][idx],
                "is_F_best": bpF["best"][iF[idx]] == jB[idx],
                "is_B_best": bpB["best"][jB[idx]] == iF[idx]})
        df.to_csv(mdir / "candidate_pairs.csv", index=False, float_format="%.6g")
        rows = df
        pst["candidate_pairs_csv_rows"] = len(rows)
        pst["candidate_pairs_csv_selection"] = (
            "PRESENTATION: every Gaussian's best partner (either side) plus each F Gaussian's "
            f"top 20 partners by rho_surf with rho_surf >= {LT['RHO_WEAK']}. The complete stored table ({k.size:,} pairs, every column) is "
            "pairs/global_final.npz (GlobalPairs.load), with rho in matches/pair_scores.npz.")
        sweep_lab = []
        for rs in (0.2, 0.3, 0.5):
            for rw in (0.05, 0.1, 0.2):
                if rw >= rs:
                    continue
                th2 = dict(LT, RHO_STRONG=rs, RHO_WEAK=rw)
                for s, bp, tg in (("F", bpF, tot_g["F"]), ("B", bpB, tot_g["B"])):
                    lab, _ = label(tg["mass_total"], tg["mass_supported"], bp, th2)
                    u, c = np.unique(lab, return_counts=True)
                    sweep_lab.append({"source": s, "RHO_STRONG": rs, "RHO_WEAK": rw,
                                      **{str(a): int(b) for a, b in zip(u, c)}})
        _write_csv(met / "threshold_sensitivity_labels.csv", sweep_lab)
    else:
        log(f"  pairs incomplete ({len(pmissing)} frames missing): match labels are "
            "PAIRS_INCOMPLETE")
    (mdir / "pair_statistics.json").write_text(json.dumps(pst, indent=1))
    summary["pairs"] = pst

    # ---- per-Gaussian evidence tables -----------------------------------
    log(f"  [{time.time() - T_AGG:6.1f} s] evidence tables")
    support_ann = {}
    for s, src in (("F", F), ("B", B)):
        sp = Path(ctx["support_paths"][s])
        if sp.is_file():
            d = json.loads(sp.read_text())
            a = np.full(src.N, "absent", dtype=object)
            a[np.asarray(d.get("static_gaussians", []), dtype=np.int64)] = "static"
            a[np.asarray(d.get("dynamic_gaussians", []), dtype=np.int64)] = "dynamic"
            support_ann[s] = (a, d.get("status"))
    tables = {}
    for s, src, bp, lab in (("F", F, bpF, labF), ("B", B, bpB, labB)):
        tg, A = tot_g[s], G[s]
        mu = src.mu
        steps = np.linalg.norm(np.diff(mu, axis=0), axis=2)
        sup_m, wit_m, vis_m = A["mass_supported"] > 0, A["witness"] > 0, A["mass_total"] > 0
        f_sup, l_sup = _first_last(sup_m, frames)
        f_wit, l_wit = _first_last(wit_m, frames)
        peak = np.asarray(frames)[np.argmax(A["witness"], axis=0)]
        cams_sup, cams_wit = _bit_or(A["cams_supported"]), _bit_or(A["cams_witness"])
        opac = src.base.opacity.reshape(-1).cpu().numpy()
        smax = src.base.scales.max(1).values.cpu().numpy()
        col = src.base.colors.cpu().numpy()
        excl = np.divide(tg["witness"], tg["mass_supported"], out=np.zeros(src.N),
                         where=tg["mass_supported"] > 0)
        t = {"row": np.arange(src.N), "mass_total": tg["mass_total"], "mass_valid": tg["mass_valid"],
             "mass_kappa": tg["mass_kappa"], "mass_supported": tg["mass_supported"],
             "witness_U": tg["witness"], "witness_improve": tg["witness_improve"],
             "exclusive_fraction": excl, "mass_front_of_observed": tg["mass_front"],
             "mass_on_gt_invalid": tg["mass_gt_invalid"],
             "frames_visible": vis_m.sum(0), "frames_supported": sup_m.sum(0),
             "frames_witness": wit_m.sum(0), "first_supported_frame": f_sup,
             "last_supported_frame": l_sup, "first_witness_frame": f_wit,
             "last_witness_frame": l_wit, "peak_witness_frame": np.where(wit_m.any(0), peak, -1),
             "cams_supported": pv.popcount(cams_sup), "cams_witness": pv.popcount(cams_wit),
             "track_path_m": steps.sum(0), "track_net_m": np.linalg.norm(mu[-1] - mu[0], axis=1),
             "track_max_step_m": steps.max(0), "opacity": opac, "max_scale_m": smax,
             "colour_r": col[:, 0], "colour_g": col[:, 1], "colour_b": col[:, 2],
             "x_anchor": mu[src.frames.index(int(src.manifest.get("canonical_original_frame", 0)))][:, 0],
             "y_anchor": mu[src.frames.index(int(src.manifest.get("canonical_original_frame", 0)))][:, 1],
             "z_anchor": mu[src.frames.index(int(src.manifest.get("canonical_original_frame", 0)))][:, 2]}
        if bp is not None:
            t |= {"best_partner": bp["best"], "best_rho_surf": bp["rho_surf"],
                  "second_rho_surf": bp["second_rho_surf"],
                  "best_partner_frames_same_surface": bp["frames_surf"],
                  "partners": bp["partners"], "partners_comparable": bp["partners_comparable"],
                  "best_pixel_partner": bp["best_pix"], "best_rho_pix": bp["rho_pix"],
                  "best_pixel_partner_incompat_frac": bp["best_pix_incompat_frac"],
                  "best_partner_by_O_surf": bp["best_by_O"],
                  "best_partner_certified_under_pruning": bp["certified"],
                  "omitted_same_surface_mass": bp["omitted_O_surf"],
                  "omitted_fraction_of_same_surface": bp["omitted_fraction"],
                  "match_label": lab}
        else:
            t["match_label"] = np.full(src.N, "PAIRS_INCOMPLETE", dtype=object)
        if s in support_ann:
            t["coherence_annotation"] = support_ann[s][0]
        tables[s] = t
        import pandas as pd
        pd.DataFrame(t).to_csv(gdir / f"{'forward' if s == 'F' else 'backward'}_evidence.csv",
                               index=False, float_format="%.6g")
    summary["label_counts"] = {s: dict(zip(*[x.tolist() for x in np.unique(
        tables[s]["match_label"].astype(str), return_counts=True)])) for s in tables}
    summary["coherence_annotation_source"] = {s: v[1] for s, v in support_ann.items()}

    # ---- unique candidates and their patches ----------------------------
    log(f"  [{time.time() - T_AGG:6.1f} s] unique candidates")
    groups = {}
    for s, src, other in (("B", B, "F"), ("F", F, "B")):
        t = tables[s]
        cand = np.flatnonzero(t["witness_U"] >= LT["U_MIN"])
        cand = cand[np.argsort(-t["witness_U"][cand])]
        name = "backward" if s == "B" else "forward"
        cols = ["row", "witness_U", "witness_improve", "exclusive_fraction", "mass_supported",
                "frames_witness", "cams_witness", "first_witness_frame", "last_witness_frame",
                "peak_witness_frame", "first_supported_frame", "match_label", "best_partner",
                "best_rho_surf", "best_rho_pix", "best_pixel_partner_incompat_frac",
                "track_path_m", "track_net_m", "opacity", "max_scale_m"] + \
            (["coherence_annotation"] if "coherence_annotation" in t else [])
        _write_csv(gdir / f"{name}_unique_candidates.csv",
                   [{c: (t[c][i].item() if hasattr(t[c][i], "item") else t[c][i])
                     for c in cols if c in t} for i in cand])
        # patches: candidates whose own label does not already claim a common surface
        pool = cand[~np.isin(t["match_label"][cand], ["LIKELY_COMMON_SURFACE"])]
        groups[s] = group_candidates(src, pool, t, frames)
        log(f"  {name}: {cand.size} Gaussians with witness U >= {LT['U_MIN']}, "
            f"{len(groups[s])} patches")
    amb = []
    for s in ("F", "B"):
        t = tables[s]
        for i in np.flatnonzero(t["match_label"] == "AMBIGUOUS"):
            amb.append({"source": s, "row": int(i), "mass_supported": float(t["mass_supported"][i]),
                        "best_partner": int(t["best_partner"][i]),
                        "best_rho_surf": float(t["best_rho_surf"][i]),
                        "second_rho_surf": float(t["second_rho_surf"][i]),
                        "partners_comparable": int(t["partners_comparable"][i])})
    amb.sort(key=lambda r: -r["mass_supported"])
    _write_csv(gdir / "ambiguous_candidates.csv", amb)

    # ---- occlusion test, counterfactuals, case studies -------------------
    log(f"  [{time.time() - T_AGG:6.1f} s] case studies")
    wdir = gdir / "trajectory_witnesses"
    wdir.mkdir(exist_ok=True)
    cases = {}
    for s in ("B", "F"):
        src = B if s == "B" else F
        cases[s] = []
        for gi, g in enumerate(groups[s][:LT["MAX_GROUPS"]]):
            members = g["members"]
            A = G[s]
            curve = A["witness"][:, members].sum(1)
            sup_curve = A["mass_supported"][:, members].sum(1)
            vis_curve = A["mass_total"][:, members].sum(1)
            t_first = frames[int(np.argmax(curve > 0))] if (curve > 0).any() else frames[-1]
            peak_f = frames[int(np.argmax(curve))]
            occ = occlusion_status(ctx, members, src, t_first, th.contrib_depth_tol_m)
            rec = {"source": s, "group": gi, "members": members.tolist(), "size": int(members.size),
                   "witness_U": float(g["witness"]), "peak_frame": peak_f,
                   "first_witness_frame": t_first,
                   "frames": list(frames), "witness_curve": curve.tolist(),
                   "supported_curve": sup_curve.tolist(), "visible_curve": vis_curve.tolist(),
                   "cams_witness_per_frame": [
                       int(pv.popcount(np.array([np.bitwise_or.reduce(A["cams_witness"][k, members])]))[0])
                       for k in range(len(frames))],
                   "match_labels": dict(zip(*[x.tolist() for x in np.unique(
                       tables[s]["match_label"][members].astype(str), return_counts=True)])),
                   "coherence_annotation": (dict(zip(*[x.tolist() for x in np.unique(
                       tables[s]["coherence_annotation"][members].astype(str), return_counts=True)]))
                       if "coherence_annotation" in tables[s] else None),
                   "occlusion": occ, "centre_at_peak": src.mu[src.frames.index(peak_f)][members].mean(0).tolist(),
                   "note": "a patch is a 3-D proximity group of witnessed Gaussians, NOT a physical part"}
            if gi < LT["CASE_STUDY_GROUPS"]:
                rec |= case_study(ctx, s, gi, members, peak_f, t_first, out)
            (wdir / f"{'B' if s == 'B' else 'F'}_group_{gi:03d}.json").write_text(
                json.dumps(rec, default=float))
            cases[s].append(rec)
            vz.curve_plot(vdir / "missing_surface_case_studies" / f"{s}_group_{gi:03d}_curves.png",
                          f"{ctx['scene']} {s}-exclusive patch {gi}: {members.size} Gaussians, "
                          f"occlusion test {occ['status']}",
                          {"witness U (exclusive, depth-compatible)": (frames, curve),
                           "supported mass": (frames, sup_curve),
                           "visible mass (any contribution)": (frames, vis_curve)}, "sum of q")
        log(f"  {s}: {len(cases[s])} patch case studies")
    summary["groups"] = {s: [{k: c[k] for k in ("group", "size", "witness_U", "first_witness_frame",
                                                "peak_frame")} | {"occlusion": c["occlusion"]["status"]}
                             for c in cases[s]] for s in cases}

    # ---- pair galleries ---------------------------------------------------
    log(f"  [{time.time() - T_AGG:6.1f} s] pair galleries")
    pair_examples = []
    if pairs_complete:
        pair_examples = pair_gallery(ctx, table, rho_surf, rho_pix, incompat, tables, out)
    summary["pair_examples"] = pair_examples

    # ---- 3-D page and index ----------------------------------------------
    log(f"  [{time.time() - T_AGG:6.1f} s] 3-D page")
    mid = frames[len(frames) // 2]
    rng = np.random.default_rng(0)
    layers = []
    for s, src, colr in (("F", F, "rgba(40,110,230,0.25)"), ("B", B, "rgba(245,140,20,0.25)")):
        xyz = src.mu[src.frames.index(mid)]
        sub = rng.choice(src.N, size=min(src.N, 15000), replace=False)
        layers.append((f"{s} all centres (uniform display subsample {sub.size}/{src.N})",
                       xyz[sub], colr, 1.2))
    for s, src, colr in (("B", B, "rgb(255,60,0)"), ("F", F, "rgb(0,60,255)")):
        for c in cases[s][:10]:
            m = np.asarray(c["members"])
            layers.append((f"{s} patch {c['group']} ({m.size} G, {c['occlusion']['status']})",
                           src.mu[src.frames.index(mid)][m], colr, 3))
    vz.gaussian_3d_html(vdir / "gaussian_3d_diagnostics.html", ctx["scene"], mid, layers)
    status = ("COMPLETE" if pairs_complete and not summary["pilot"] else
              "PILOT_COMPLETE" if pairs_complete else "COVERAGE_ONLY")
    summary["status"] = status
    vz.write_index(vdir, ctx["scene"], frames, ctx["cams"], ctx["depth_range"],
                   {"status": status, "pilot": summary["pilot"],
                    "GT-valid pixels": nv,
                    "both cover (% GT-valid)": summary["pixel_pct_of_gt_valid"]["BOTH_COVER__on_gt_valid"],
                    "B explains, F does not (%)": summary["pixel_pct_of_gt_valid"]["B_SUPPORTED_F_UNEXPLAINED"],
                    "F explains, B does not (%)": summary["pixel_pct_of_gt_valid"]["F_SUPPORTED_B_UNEXPLAINED"]})
    (met / "global_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if not skip_report:
        from . import report
        report.write(out, summary, pf, cases, pair_examples, tables, ctx)
    return summary


def group_candidates(src, pool, t, frames):
    """3-D proximity patches over the whole track: two candidates are linked if
    their centres stay within r at seven sampled frames (RMS)."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    if pool.size == 0:
        return []
    samp = [frames[int(round(k))] for k in np.linspace(0, len(frames) - 1, 7)]
    feat = np.concatenate([src.mu[src.frames.index(f)][pool] for f in samp], axis=1)
    r = max(2.0 * float(np.median(t["max_scale_m"][pool])), 0.01)
    tree = cKDTree(feat)
    pairs = tree.query_pairs(r * np.sqrt(len(samp)), output_type="ndarray")
    from scipy.sparse import coo_matrix
    n = pool.size
    m = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) \
        else coo_matrix((n, n))
    nc, lab = connected_components(m, directed=False)
    groups = []
    for c in range(nc):
        mem = pool[lab == c]
        groups.append({"members": np.sort(mem), "witness": float(t["witness_U"][mem].sum()),
                       "link_radius_m": r})
    groups.sort(key=lambda g: -g["witness"])
    return groups


def case_study(ctx, s, gi, members, peak_f, t_first, out) -> dict:
    """Matched temporal crops for one patch at its two best cameras, and the
    within-source counterfactual (hide the patch, re-render)."""
    import dataclasses
    import torch
    from estimator.renderer_bridge import build_linearization
    th, frames = ctx["th"], ctx["frames"]
    src = ctx["B"] if s == "B" else ctx["F"]
    mt = torch.as_tensor(members, device="cuda", dtype=torch.long)
    # best cameras at the peak frame by the patch's own witnessed mass
    per_cam = {}
    renders = {}
    for cam in ctx["cams"]:
        r = _render_pair(ctx, peak_f, cam)
        renders[cam] = r
        v, c, _ = r[s]
        HW = c.H * c.W
        mass = _member_mass(c, mt, HW)
        rgb, d, valid = r["gt"]
        per_cam[cam] = float(mass[valid.reshape(-1)].sum())
    best = sorted(per_cam, key=lambda k: -per_cam[k])[:2]
    rows_frames = sorted(set([f for f in CASE_FRAMES if f in frames] + [t_first, peak_f]))
    res = {"best_cameras": best, "patch_mass_per_camera_at_peak": per_cam, "case_study_pngs": []}
    for cam in best:
        spec = ctx["specs"][ctx["all_cams"].index(cam)]
        xyz = src.mu[src.frames.index(peak_f)][members]
        u, v, _z = project(spec, xyz)
        box = _box(u, v, spec.image_height, spec.image_width)
        rows = []
        for f in rows_frames:
            r = renders[cam] if f == peak_f else _render_pair(ctx, f, cam)
            vv, cc, _ = r[s]
            mass = _member_mass(cc, mt, cc.H * cc.W)
            crops = _crop_images(r, mass, box, th, ctx["depth_range"])
            crops["label"] = (f"frame {f}" + (" first witness" if f == t_first else "")
                              + (" PEAK" if f == peak_f else ""))
            rows.append(crops)
        p = out / "visuals" / "missing_surface_case_studies" / f"{s}_group_{gi:03d}_{cam}.png"
        vz.case_study(p, f"{ctx['scene']}: {s}-exclusive patch {gi} ({members.size} Gaussians), "
                         f"{cam}, crop fixed at the peak frame {peak_f}", rows, ctx["depth_range"])
        res["case_study_pngs"].append(str(p.relative_to(out)))
    # counterfactual within the unchanged source, at the best view
    cam = best[0]
    spec = ctx["specs"][ctx["all_cams"].index(cam)]
    st = src.state(peak_f)
    op = st.opacity.clone()
    op[mt] = 0.0
    hidden = build_linearization(dataclasses.replace(st, opacity=op), [spec]).views[0]
    r = renders[cam]
    v0, c0, _ = r[s]
    rgb, d, valid = r["gt"]
    mass = _member_mass(c0, mt, c0.H * c0.W).reshape(d.shape)
    wp = valid & (mass >= 0.1)
    e0 = (v0.depth - d).abs()[wp]
    e1 = (hidden.depth - d).abs()[wp]
    a1 = hidden.alpha[wp]
    res["counterfactual"] = {
        "kind": "DIAGNOSTIC_COUNTERFACTUAL_WITHIN_SOURCE: the patch's opacity set to 0 in "
                "its own source, one view re-rendered; not a third-model insertion test",
        "frame": peak_f, "camera": cam, "pixels_where_patch_q_ge_0.1": int(wp.sum()),
        "mean_abs_depth_err_with_m": float(e0.mean()) if e0.numel() else None,
        "mean_abs_depth_err_hidden_m": float(e1[a1 >= th.alpha_cover].mean()) if bool((a1 >= th.alpha_cover).any()) else None,
        "fraction_still_covered_when_hidden": float((a1 >= th.alpha_cover).float().mean()) if a1.numel() else None,
        "mean_rgb_rms_with": float(((v0.rgb - rgb) ** 2).mean(0).sqrt()[wp].mean()) if e0.numel() else None,
        "mean_rgb_rms_hidden": float(((hidden.rgb - rgb) ** 2).mean(0).sqrt()[wp].mean()) if e0.numel() else None}
    return res


def pair_curves(ctx, sel):
    """Per-frame O and O_surf (primary tolerance) of a few nominated pairs, by a
    render sweep over every processed view. Exact: one Gaussian contributes at
    most once per pixel, so the pair's pixel product is the two maps' product."""
    import torch
    from .renderer_access import pixel_contributions
    from .sources import load_gt
    from estimator.renderer_bridge import build_linearization
    tol = ge.PAIR_TOLS_M[pv.PRIMARY]
    fr_idx = {f: k for k, f in enumerate(ctx["frames"])}
    res = {p: {"O": [0.0] * len(fr_idx), "O_surf": [0.0] * len(fr_idx), "best_view": None,
               "best": -1.0} for p in sel}
    Fset = torch.tensor(sorted({a for a, _ in sel}), device="cuda")
    Bset = torch.tensor(sorted({b for _, b in sel}), device="cuda")
    for f in ctx["frames"]:
        stF, stB = ctx["F"].state(f), ctx["B"].state(f)
        for cam in ctx["cams"]:
            spec = ctx["specs"][ctx["all_cams"].index(cam)]
            vF = build_linearization(stF, [spec]).views[0]
            vB = build_linearization(stB, [spec]).views[0]
            cF, cB = pixel_contributions(vF.cache), pixel_contributions(vB.cache)
            _r, d, valid, _ = load_gt(ctx["data"], cam, f, spec.image_height, spec.image_width)
            HW = cF.H * cF.W
            maps = {}
            for key, c, S in (("F", cF, Fset), ("B", cB, Bset)):
                m = torch.isin(c.gid.long(), S)
                g, pix, q = c.gid[m].long(), c.pixel[m], c.q[m].double()
                for gg in S.tolist():
                    mm = g == gg
                    maps[(key, gg)] = torch.zeros(HW, dtype=torch.float64, device="cuda").index_add_(
                        0, pix[mm], q[mm])
            dd, vv = d.reshape(-1).double(), valid.reshape(-1)
            for (a, b) in sel:
                zF, zB = float(vF.cache.depth[a]), float(vB.cache.depth[b])
                prod = maps[("F", a)] * maps[("B", b)]
                kap = vv & (abs(zF - zB) <= tol) & ((zF - dd).abs() <= tol) & ((zB - dd).abs() <= tol)
                o, os_ = float(prod.sum()), float((prod * kap).sum())
                r = res[(a, b)]
                r["O"][fr_idx[f]] += o
                r["O_surf"][fr_idx[f]] += os_
                score = os_ if os_ > 0 else 1e-9 * o
                if score > r["best"]:
                    r["best"], r["best_view"] = score, (f, cam)
            del vF, vB, cF, cB, maps
    return res


def pair_gallery(ctx, table, rho_surf, rho_pix, incompat, tables, out):
    """Strong, weak, ambiguous and false (same pixel, different depth) examples."""
    import torch
    LT = LABEL_THRESHOLDS
    k = table["keys"]
    NB = ctx["B"].N
    iF, jB = k // NB, k % NB
    tF = tables["F"]
    picks = []
    strong = np.flatnonzero((rho_surf >= LT["RHO_STRONG"]) & (table["frames_surf"] >= LT["MIN_FRAMES"]))
    picks += [("STRONG_SAME_SURFACE", i) for i in strong[np.argsort(-table["O_surf_1"][strong])][:3]]
    weak = np.flatnonzero((rho_surf >= LT["RHO_WEAK"]) & (rho_surf < LT["RHO_STRONG"]))
    picks += [("WEAK_SAME_SURFACE", i) for i in weak[np.argsort(-table["O_surf_1"][weak])][:2]]
    ambF = np.flatnonzero(tF["match_label"] == "AMBIGUOUS")
    if ambF.size:
        f0 = ambF[np.argmax(tF["mass_supported"][ambF])]
        cand = np.flatnonzero(iF == f0)
        picks += [("AMBIGUOUS_PARTNER_1", cand[np.argsort(-rho_surf[cand])][0])]
        if cand.size > 1:
            picks += [("AMBIGUOUS_PARTNER_2", cand[np.argsort(-rho_surf[cand])][1])]
    false = np.flatnonzero((rho_pix >= LT["RHO_STRONG"]) & (rho_surf < LT["RHO_WEAK"]) &
                           (incompat >= LT["INCOMPAT_FRACTION"]))
    picks += [("FALSE_SAME_PIXEL_DIFFERENT_DEPTH", i) for i in false[np.argsort(-table["O"][false])][:3]]
    res = []
    pdir = out / "visuals" / "pair_case_studies"
    pdir.mkdir(parents=True, exist_ok=True)
    frames = ctx["frames"]
    sel = [(int(iF[i]), int(jB[i])) for _n, i in picks]
    curves = pair_curves(ctx, sel)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for n, (kind, i) in enumerate(picks):
        fi, bj = int(iF[i]), int(jB[i])
        cv = curves[(fi, bj)]
        f, best_cam = cv["best_view"]
        best_r = _render_pair(ctx, f, best_cam)
        cF, cB = best_r["F"][1], best_r["B"][1]
        bmF = _member_mass(cF, torch.tensor([fi], device="cuda"), cF.H * cF.W)
        bmB = _member_mass(cB, torch.tensor([bj], device="cuda"), cB.H * cB.W)
        rgb, d, valid = best_r["gt"]
        H, W = d.shape
        co = (bmF * bmB).reshape(H, W).cpu().numpy()
        ys, xs = np.nonzero(co > 0)
        box = _box(xs.astype(float), ys.astype(float), H, W, margin=30)
        y0, y1, x0, x1 = box
        vF, vB = best_r["F"][0], best_r["B"][0]
        fig, ax = plt.subplots(1, 6, figsize=(17, 3.4))
        panels = [("GT RGB", vz._to_u8(rgb.cpu().numpy().transpose(1, 2, 0))),
                  ("GT depth", vz.depth_rgb(d.cpu().numpy(), valid.cpu().numpy(), ctx["depth_range"])),
                  (f"F depth (row {fi})", vz.depth_rgb(vF.depth.cpu().numpy(), vF.alpha.cpu().numpy() > 0.05, ctx["depth_range"])),
                  (f"B depth (row {bj})", vz.depth_rgb(vB.depth.cpu().numpy(), vB.alpha.cpu().numpy() > 0.05, ctx["depth_range"])),
                  ("qF_i (red) qB_j (green)", np.stack([np.clip(bmF.reshape(H, W).cpu().numpy() * 3, 0, 1),
                                                         np.clip(bmB.reshape(H, W).cpu().numpy() * 3, 0, 1),
                                                         np.zeros((H, W))], -1))]
        for a, (t, img) in zip(ax[:5], panels):
            a.imshow(img[y0:y1, x0:x1])
            a.set_title(t, fontsize=8)
            a.axis("off")
        ax[5].plot(frames, cv["O"], label="O_ij (same pixel)")
        ax[5].plot(frames, cv["O_surf"], label="O_surf_ij (same surface, 4 cm)")
        ax[5].set_xlabel("original frame")
        ax[5].legend(fontsize=7)
        fig.suptitle(f"{ctx['scene']} {kind}: F{fi} - B{bj}  rho_surf {rho_surf[i]:.3f}  "
                     f"rho_pix {rho_pix[i]:.3f}  incompat {incompat[i]:.2f}  "
                     f"frames same-surface {int(table['frames_surf'][i])}  view frame {f} {best_cam}",
                     fontsize=9)
        fig.tight_layout()
        p = pdir / f"{n:02d}_{kind}_F{fi}_B{bj}.png"
        fig.savefig(p, dpi=85)
        plt.close(fig)
        res.append({"kind": kind, "F_row": fi, "B_row": bj, "rho_surf": float(rho_surf[i]),
                    "rho_pix": float(rho_pix[i]), "incompat_fraction": float(incompat[i]),
                    "frames_same_surface": int(table["frames_surf"][i]),
                    "mean_center_dist_m": float(table["dist3d"][i] / max(table["O"][i], 1e-30)),
                    "view": [f, best_cam], "png": str(p.relative_to(out))})
        (out / "matches" / "selected_pair_witnesses").mkdir(exist_ok=True)
        (out / "matches" / "selected_pair_witnesses" / f"{n:02d}_{kind}.json").write_text(
            json.dumps(res[-1] | {"per_frame_O": cv["O"], "per_frame_O_surf": cv["O_surf"],
                                  "frames": list(frames)}))
    return res
