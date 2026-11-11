#!/usr/bin/env python
"""Phase-0 geometry-consistency evidence and an OPT-IN counterfactual repair.

The endpoint model is rendered with the production renderer at its own (fixed) state; on
each evidence frame, every composited contributor of every pixel (production CSR:
observation_diagnostics/renderer_access.pixel_contributions, q = alpha T, T_STOP included)
is compared with that camera's observation:

    off-mask        M = 0:  the object mask says nothing is there, the Gaussian is visible
    in front        M = 1, valid depth, z_i < D - k sigma_eff  (free space it occupies)
    matched         M = 1, valid depth, |z_i - D| <= k sigma_eff
    behind / other  z_i > D + k sigma_eff, or no valid depth: neutral (occluded / unknown)

sigma_eff^2 = sigma_D(D)^2 (calibrated or configured) + declared depth model sd^2 +
s_i^2, with s_i the Gaussian's own extent along that camera's ray (r^T Sigma_i r): the full
footprint in depth, not the centre alone. Per Gaussian, over cameras and frames:

    C_i = sum q (off-mask) + sum q (in front)       contradiction mass
    S_i = sum q (matched)                           support mass

Repair rules (opt-in, one comparison each, no tuned weight):
    majority    remove i iff C_i > S_i (masses summed over cameras)
    unanimous   remove i iff C_ic > S_ic in EVERY camera c that sees it (C_ic + S_ic > 0):
                a Gaussian matched in one camera and contradicted in another is kept,
                because that pattern is also what a cross-camera depth/registration
                disagreement produces (see cross_view_depth_offset in the report)
The repaired model is a NEW directory (PLYs, geometry_information.npz rows, manifest);
the source model is never modified. The report records every count, the per-camera
masses and, when given, the overlap with other audits' Gaussian sets.

    python tools/phase0_geometry_consistency.py --model M --source S --sensor-model CAL \\
        --depth-model-sigma 0.003 --frames 60 80 --report R.json [--out-model M_repaired]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))


def ray_extent(view, gid_np, mu, Sigma, cam_centre):
    """s_i = sqrt(r^T Sigma_i r), r the unit ray from the camera centre to the Gaussian centre."""
    r = mu[gid_np] - cam_centre[None]
    r = r / np.linalg.norm(r, axis=1, keepdims=True).clip(1e-12)
    return np.sqrt(np.einsum("ni,nij,nj->n", r, Sigma[gid_np], r).clip(0))


@torch.no_grad()
def evidence(a):
    from audit_calibrated_phase1 import _frame_obs, setup_window
    from audit_phase1_wrong_camera import make_sigma_fn
    from estimator.renderer_bridge import build_linearization
    from estimator.wrong_camera_audit import camera_centres
    from observation_diagnostics.renderer_access import pixel_contributions
    s0, s1 = a.frames
    ns_args = argparse.Namespace(**vars(a))
    # a single evidence frame (START == END, e.g. a synthetic endpoint with no stationary
    # window): the loader needs two frames, so widen by one and use only [START, END]
    w0, w1 = (s0, s1) if s1 > s0 else ((s0 - 1, s1) if a.direction == "backward" else (s0, s1 + 1))
    ns_args.motion_start_frame, ns_args.motion_end_frame = w0, w1
    truth, cams, window, cfg, ns, ids = setup_window(ns_args)
    n = truth.mu.shape[0]
    model = None if ns.masked_observation is None else ns.masked_observation.model
    bg = None if model is None else model.background_tensor(truth.mu.device)
    lin = build_linearization(truth, cams, eps_depth=cfg.eps_depth, background=bg)
    sig_fn = make_sigma_fn()
    mu = truth.mu.double().cpu().numpy()
    R = truth.R.double().cpu().numpy()
    sc = truth.scales.double().cpu().numpy()
    Sigma = np.einsum("nij,nj,nkj->nik", R, sc ** 2, R)
    centres = camera_centres(lin).cpu().numpy()
    names = list(cfg.camera_names)
    keys = ("off_mask", "in_front", "matched", "behind", "no_depth", "qd_valid", "q_valid", "qD_valid")
    per = {nm: {k: np.zeros(n) for k in keys} for nm in names}
    csr = []
    for c, view in enumerate(lin.views):
        cc = pixel_contributions(view.cache)
        gid = cc.gid.long().cpu().numpy()
        q = (cc.alpha * cc.T).double().cpu().numpy()
        pix = cc.pixel.cpu().numpy()
        z = view.cache.depth.double().cpu().numpy()[gid]
        ext = ray_extent(view, gid, mu, Sigma, centres[c])
        csr.append((gid, q, pix, z, ext))
    frames_used = []
    for k in range(len(window)):
        fid = window[k][1]
        of = fid.get("original_frame_index") if isinstance(fid, dict) else fid
        if not (min(s0, s1) <= int(of) <= max(s0, s1)):
            continue
        obs = _frame_obs(ns, k, window, truth.mu.device)
        frames_used.append(of)
        for c, (gid, q, pix, z, ext) in enumerate(csr):
            ob = obs[c]
            D = ob.depth_observed.float()
            valid = ob.depth_valid.bool() & torch.isfinite(D) & (D > 0)
            M = (ob.object_coverage > 0.5).reshape(-1).cpu().numpy()
            sd = sig_fn(c, D, valid, cfg).double().reshape(-1).cpu().numpy()
            Dn, vn = D.double().reshape(-1).cpu().numpy(), valid.reshape(-1).cpu().numpy()
            m_p, v_p = M[pix], vn[pix]
            se = np.sqrt(sd[pix] ** 2 + ext ** 2)
            d = z - Dn[pix]
            off = ~m_p
            front = m_p & v_p & (d < -a.k_sigma * se)
            match = m_p & v_p & (np.abs(d) <= a.k_sigma * se)
            behind = m_p & v_p & (d > a.k_sigma * se)
            nodep = m_p & ~v_p
            for key, sel in (("off_mask", off), ("in_front", front), ("matched", match), ("behind", behind),
                             ("no_depth", nodep)):
                per[names[c]][key] += np.bincount(gid[sel], weights=q[sel], minlength=n)
            # q-weighted signed centre-to-surface offset and observed depth, every in-mask valid
            # record: the per-camera cross-view consistency diagnostic (z_i - D, metres)
            vv = m_p & v_p
            per[names[c]]["qd_valid"] += np.bincount(gid[vv], weights=(q * d)[vv], minlength=n)
            per[names[c]]["q_valid"] += np.bincount(gid[vv], weights=q[vv], minlength=n)
            per[names[c]]["qD_valid"] += np.bincount(gid[vv], weights=(q * Dn[pix])[vv], minlength=n)
    C = sum(per[nm]["off_mask"] + per[nm]["in_front"] for nm in names)
    S = sum(per[nm]["matched"] for nm in names)
    return truth, cfg, names, per, C, S, frames_used, n


def cross_view(per, names) -> dict:
    """Gaussians MATCHED in camera A (most of their in-mask mass within k sigma in A) are
    common 3D points; their q-weighted signed offset z - D in camera B measures how B's
    depth disagrees with A's geometry. A linear fit offset = a + b * D_B separates a
    constant offset from a depth-proportional (scale) disagreement."""
    out = {}
    for A in names:
        own = per[A]["matched"] + per[A]["in_front"] + per[A]["off_mask"] + per[A]["behind"]
        mA = (per[A]["matched"] > 0.5 * own) & (own > 0)
        for B in names:
            if A == B:
                continue
            sel = mA & (per[B]["q_valid"] > 0.05)
            if sel.sum() < 30:
                out[f"{A}->{B}"] = {"n": int(sel.sum())}
                continue
            off = per[B]["qd_valid"][sel] / per[B]["q_valid"][sel]
            DB = per[B]["qD_valid"][sel] / per[B]["q_valid"][sel]
            b, a0 = np.polyfit(DB, off, 1)
            out[f"{A}->{B}"] = {"n": int(sel.sum()), "median_offset_mm": 1000 * float(np.median(off)),
                                "q10_q90_mm": [1000 * float(np.quantile(off, 0.1)), 1000 * float(np.quantile(off, 0.9))],
                                "fraction_in_front_>2cm": float((off < -0.02).mean()),
                                "fraction_behind_>2cm": float((off > 0.02).mean()),
                                "fit_offset_mm_at_0": 1000 * float(a0), "fit_slope_mm_per_m": 1000 * float(b),
                                "depth_range_m": [float(DB.min()), float(DB.max())]}
    return out


def write_repaired(src: Path, dst: Path, keep: np.ndarray, info: dict):
    from plyfile import PlyData, PlyElement
    if dst.exists():
        raise SystemExit(f"{dst} exists: refusing to overwrite")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("source_responsibility.npz", "stage_signature.json*"))
    n = int(keep.size)
    for ply in [dst / "point_cloud.ply"] + sorted((dst / "point_cloud").glob("iteration_*/point_cloud.ply")):
        v = PlyData.read(str(ply))["vertex"]
        if len(v) != n:
            raise SystemExit(f"{ply}: {len(v)} vertices, the evidence has {n}")
        PlyData([PlyElement.describe(v.data[keep], "vertex")]).write(str(ply))
    gi = dst / "geometry_information.npz"
    if gi.is_file():
        z = dict(np.load(gi))
        np.savez_compressed(gi, **{k: (v[keep] if v.ndim and v.shape[0] == n else v) for k, v in z.items()})
    for name in ("bootstrap_manifest.json", "manifest.json"):
        p = dst / name
        if p.is_file():
            m = json.loads(p.read_text())
            m["num_gaussians"] = int(keep.sum())
            m["phase0_geometry_repair"] = info
            p.write_text(json.dumps(m, indent=1, default=float))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--frames", type=int, nargs=2, required=True, metavar=("START", "END"),
                    help="evidence frames with the object in the endpoint configuration (stationary)")
    ap.add_argument("--direction", choices=("forward", "backward"), default="forward",
                    help="backward for a FINAL endpoint (its window is read in reverse)")
    ap.add_argument("--cameras", default="all", help="all | first N | cam_001,cam_002 (utils/camera_selection.py)")
    ap.add_argument("--k-sigma", type=float, default=3.0, help="depth-agreement width in units of sigma_eff")
    ap.add_argument("--report", required=True)
    ap.add_argument("--out-model", default=None, help="write the repaired model here (opt-in)")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="name=path pairs of Gaussian-id sets (json list or selected_contributor_traces.jsonl)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--rule", choices=("majority", "unanimous"), default="majority",
                    help="which repair rule writes --out-model (the report counts both)")
    from estimator.masked_observation import add_cli as mask_cli
    from estimator.run_config import add_cli as rc_cli
    mask_cli(ap)
    rc_cli(ap, phase1=True)
    a = ap.parse_args(argv)
    if a.deterministic:
        from estimator.deterministic_ops import enable
        enable()
    for k, v in {"bootstrap_iters": 12, "iekf_supernode_size": 64, "phase1_partition": "canonical",
                 "phase1_covariance_transport": False, "first_return_mode": "off", "first_return_prior": None,
                 "gt_labels": "off"}.items():
        setattr(a, k, v)
    t0 = time.time()
    truth, cfg, names, per, C, S, frames, n = evidence(a)
    rules = {"majority": C > S}
    seen = np.stack([(per[nm]["off_mask"] + per[nm]["in_front"] + per[nm]["matched"]) > 0 for nm in names])
    contra = np.stack([(per[nm]["off_mask"] + per[nm]["in_front"]) > per[nm]["matched"] for nm in names])
    rules["unanimous"] = seen.any(0) & np.all(contra | ~seen, axis=0)
    remove = rules[a.rule]
    frac = C / np.maximum(C + S, 1e-30)
    rep = {"tool": "tools/phase0_geometry_consistency.py", "model": a.model, "source": a.source,
           "evidence_frames": frames, "k_sigma": a.k_sigma, "num_gaussians": n,
           "rule": a.rule,
           "rule_definitions": {"majority": "C_i (off-mask + in-front q) > S_i (matched q), summed over cameras",
                                "unanimous": "C_ic > S_ic in every camera that sees i"},
           "num_removed_by_rule": {k: int(v.sum()) for k, v in rules.items()},
           "num_removed": int(remove.sum()), "fraction_removed": float(remove.mean()),
           "num_invisible": int(((C + S) == 0).sum()),
           "per_camera_mass": {nm: {k: float(v.sum()) for k, v in d.items() if k in ("off_mask", "in_front", "matched",
                                                                                       "behind", "no_depth")}
                               for nm, d in per.items()},
           "cross_view_depth_offset": cross_view(per, names),
           "contradiction_fraction_q": [float(x) for x in np.quantile(frac[(C + S) > 0], [0.5, 0.9, 0.99])],
           "removed_by_dominant_camera": {nm: int(((np.argmax(np.stack([per[x]["off_mask"] + per[x]["in_front"]
                                                                          for x in names]), 0) == i) & remove).sum())
                                          for i, nm in enumerate(names)},
           "removed_by_dominant_kind": {k: int(((np.stack([sum(per[x][kk] for x in names) for kk in ("off_mask", "in_front")])
                                                .argmax(0) == j) & remove).sum()) for j, k in enumerate(("off_mask", "in_front"))},
           "seconds": round(time.time() - t0, 1)}
    for pair in a.compare:
        name, path = pair.split("=", 1)
        p = Path(path)
        if p.suffix == ".jsonl":
            sets = {}
            for line in p.read_text().splitlines():
                r = json.loads(line)
                sets.setdefault(r.get("set", "all"), set()).add(int(r["gaussian_id"]))
        else:
            sets = {name: set(int(i) for i in json.loads(p.read_text()))}
        for sn, ids in sets.items():
            ids = np.array(sorted(ids))
            ids = ids[ids < n]
            rep.setdefault("overlap", {})[f"{name}:{sn}"] = {
                "size": int(ids.size), "removed": int(remove[ids].sum()),
                "removed_by_rule": {k: int(v[ids].sum()) for k, v in rules.items()},
                "fraction_removed": float(remove[ids].mean()) if ids.size else None,
                "median_contradiction_fraction": float(np.median(frac[ids])) if ids.size else None}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.report).with_suffix(".npz"), C=C, S=S, remove=remove,
                        **{f"remove_{k}": v for k, v in rules.items()},
                        **{f"{nm}_{k}": v for nm, d in per.items() for k, v in d.items()})
    if a.out_model:
        write_repaired(Path(a.model), Path(a.out_model), ~remove,
                       {"source_model": a.model, "report": a.report, "removed": int(remove.sum()),
                        "rule": rep["rule"], "k_sigma": a.k_sigma, "evidence_frames": frames})
        rep["repaired_model"] = a.out_model
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, indent=1, default=float))
    print(f"removed {rep['num_removed']} of {n} ({100 * rep['fraction_removed']:.2f}%), wrote {a.report}"
          + (f" and {a.out_model}" if a.out_model else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
