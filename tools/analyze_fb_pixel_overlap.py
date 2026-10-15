#!/usr/bin/env python
"""Forward/backward pixel overlap, same-surface evidence and complementary
coverage of the EXISTING Phase-1 sources (read only).

CODEX_PHASE1_FORWARD_BACKWARD_PIXEL_OVERLAP_COVERAGE_DIAGNOSTICS.md.

Each source (forward = static_bidi/initial + bidirectional/forward tracks,
backward = static_bidi/final + bidirectional/backward tracks) is rendered
ALONE at every requested original (frame, camera) with the production IEKF
renderer, against the dataset's own RGB-D. Nothing is trained, merged, pruned
or optimised; the raw F+B union is never rendered. Outputs go only to

    <results_root>/<scene>/observation_overlap_coverage/<tag>/

Passes (resumable per original frame, refused if inputs or code changed):
  coverage  every pixel's coverage / explanation category, every Gaussian's
            exact contribution accounting, maps, threshold sweeps
  pairs     exact enumeration of every (F, B) co-contributor pair at every
            pixel both sources reach, merged over the sequence
  then      aggregation, Gaussian labels, case studies, galleries, reports

    python tools/analyze_fb_pixel_overlap.py --scene washer_103776 --frames all \
        --cameras all --tag overlap_coverage_v1 --mode all --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_RESULTS = "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"
DEFAULT_DATASET = ("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/"
                   "preparation_8cams/partnet_subset_fixed_render_linear")
OUT_SUBDIR = "observation_overlap_coverage"


class Log:
    """print() that also appends to logs/<name>.log."""

    def __init__(self, out: Path):
        self.dir = out / "logs"
        self.name = "coverage"

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{self.name}.log", "a") as fh:
            fh.write(time.strftime("%H:%M:%S ") + msg + "\n")


def parse_list(spec: str, universe: list, kind: str) -> list:
    if spec == "all":
        return list(universe)
    if kind == "frames" and ":" in spec:
        a, b = spec.split(":")
        out = list(range(int(a), int(b)))
    elif kind == "frames":
        out = [int(x) for x in spec.split(",") if x.strip()]
    else:
        out = [x.strip() for x in spec.split(",") if x.strip()]
    bad = [x for x in out if x not in universe]
    if bad:
        raise SystemExit(f"--{kind}: {bad} not available; available: {universe}")
    return out


MEASUREMENT_CODE = ("renderer_access.py", "sources.py", "pixel_evidence.py",
                    "gaussian_evidence.py", "pair_evidence.py")
PRESENTATION_CODE = ("aggregate.py", "visualize.py", "report.py")


def code_hash(names=MEASUREMENT_CODE, include_tool=True) -> str:
    """Hash of the code that produces the per-frame records. Aggregation,
    figures and reports are re-derived from those records on every run, so
    their hash is recorded in provenance but does not block a resume."""
    h = hashlib.sha256()
    files = [REPO / "observation_diagnostics" / n for n in names]
    if include_tool:
        files.append(Path(__file__))
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def mem_available_gib() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("inf")


class HostRamLow(RuntimeError):
    pass


def peak() -> dict:
    import torch
    return {"peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
            "peak_host_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def preflight(args, paths, out: Path) -> dict:
    import torch
    from observation_diagnostics.sources import protected_files

    rep = {"scene": args.scene, "results_root": args.results_root,
           "dataset_root": args.dataset_root, "out": str(out), "tag": args.tag,
           "python": sys.executable, "blockers": [], "warnings": []}
    rep["cuda_available"] = torch.cuda.is_available()
    if not rep["cuda_available"]:
        rep["blockers"].append("CUDA unavailable")
    else:
        free, total = torch.cuda.mem_get_info()
        rep["gpu"] = {"name": torch.cuda.get_device_name(), "free_mib": free // 2 ** 20,
                      "total_mib": total // 2 ** 20}
        if free < 1.0 * 2 ** 30:
            rep["warnings"].append(f"only {free / 2**30:.2f} GiB GPU free")
    try:
        import diff_gaussian_rasterization  # noqa: F401
        rep["rasterizer_importable"] = True
    except Exception as e:  # pragma: no cover
        rep["rasterizer_importable"] = False
        rep["blockers"].append(f"rasterizer import failed: {e}")
    disk = shutil.disk_usage(Path(args.results_root))
    rep["disk_free_gib"] = round(disk.free / 2 ** 30, 1)
    if disk.free < 20 * 2 ** 30:
        rep["blockers"].append("less than 20 GiB free on the results disk")
    data = Path(args.dataset_root) / args.scene
    rep["dataset_dir"] = str(data)
    for k, s in paths["sources"].items():
        e = {"direction": s["direction"], "endpoint": s["endpoint"],
             "ply": str(s["ply"]) if s["ply"] else None}
        for name in ("manifest", "support"):
            e[name] = str(s[name]) if Path(s[name]).is_file() else None
        if s["ply"] is None:
            rep["blockers"].append(f"{k}: no trained PLY under {s['model']}")
        if e["manifest"] is None:
            rep["blockers"].append(f"{k}: no {s['manifest']}")
        else:
            m = json.loads(Path(s["manifest"]).read_text())
            e["manifest_summary"] = {x: m.get(x) for x in (
                "direction", "canonical_endpoint", "canonical_original_frame",
                "trajectory_order", "num_frames", "num_gaussians")}
            po = m.get("processing_order_original_indices", [])
            e["processing_order"] = f"{po[0]}..{po[-1]}" if po else None
            if m.get("trajectory_order") != "original_chronological":
                rep["blockers"].append(f"{k}: trajectory_order {m.get('trajectory_order')!r}")
        files = sorted(Path(s["centers"]).glob("centers_frame_*.npy"))
        e["center_files"] = len(files)
        e["full_state_export"] = (Path(s["full_state"]) / "full_state_manifest.json").is_file()
        e["state_mode"] = ("RECORDED_FULL_12D_STATE" if e["full_state_export"]
                           else "CENTER_ONLY_FIXED_ORIENTATION")
        rep[f"source_{k}"] = e
    cams = sorted(p.name for p in (data / "rgb").glob("cam_*")) if (data / "rgb").is_dir() else []
    rep["dataset_cameras"] = cams
    if not cams:
        rep["blockers"].append(f"no {data}/rgb/cam_* directories")
    nframes = [len(list((data / "depth_npy" / c).glob("frame_*.npy"))) for c in cams]
    rep["dataset_frames_per_camera"] = dict(zip(cams, nframes))
    if not (data / "depth_npy").is_dir():
        rep["blockers"].append("no depth_npy/ (metric depth) in the dataset")
    for name in ("colmap", "colmap_final"):
        if not (data / name / "sparse" / "0").is_dir():
            rep["warnings"].append(f"{data}/{name}/sparse/0 missing")
    prot = protected_files(paths)
    rep["protected_files"] = len(prot)
    gt_dir = data / "metadata"
    rep["gt_metadata_present_but_unused"] = gt_dir.is_dir()
    return rep


# ---------------------------------------------------------------------------
# the two passes
# ---------------------------------------------------------------------------

def render_view(state, spec):
    from estimator.renderer_bridge import build_linearization
    v = build_linearization(state, [spec]).views[0]
    return v


def conservation(contrib, view) -> dict:
    """T1 on real data: sum q = A, sum q z / (A + eps) = D, final T equal."""
    import torch
    pix = contrib.pixel
    q = contrib.q.double()
    HW = contrib.H * contrib.W
    A = torch.zeros(HW, dtype=torch.float64, device=q.device).index_add_(0, pix, q)
    Z = torch.zeros_like(A).index_add_(0, pix, q * view.cache.depth[contrib.gid.long()].double())
    a_ref = view.alpha.reshape(-1).double()
    covered = a_ref > 0.05
    dz = (Z / (A + view.cache.eps_depth) - view.depth.reshape(-1).double()).abs()
    return {"max_abs_sum_q_minus_A": float((A - a_ref).abs().max()),
            "max_abs_depth_recon_on_alpha_gt_0.05": float(dz[covered].max()) if bool(covered.any()) else 0.0,
            "final_T_bitwise_equal": bool(torch.equal(contrib.final_T, view.transmittance.reshape(-1))),
            "records_le_n_contrib": bool((contrib.counts <= view.n_contrib.reshape(-1)).all())}


def contributor_counts(contrib) -> dict:
    import torch
    c = contrib.counts
    nz = c[c > 0].float()
    if nz.numel() == 0:
        return {"pixels_with_contributors": 0, "records": 0}
    qs = torch.quantile(nz, torch.tensor([0.5, 0.9, 0.99], device=nz.device))
    return {"pixels_with_contributors": int(nz.numel()), "records": int(c.sum()),
            "per_pixel_p50": float(qs[0]), "per_pixel_p90": float(qs[1]),
            "per_pixel_p99": float(qs[2]), "per_pixel_max": int(c.max()),
            "early_terminated_pixels": int(contrib.terminated.sum()),
            "residual_T_at_termination_max": (float(contrib.final_T[contrib.terminated > 0].max())
                                              if bool((contrib.terminated > 0).any()) else None)}


def coverage_pass(ctx, log):
    import torch
    from observation_diagnostics import gaussian_evidence as ge
    from observation_diagnostics import pixel_evidence as pe
    from observation_diagnostics import visualize as vz
    from observation_diagnostics.renderer_access import pixel_contributions
    from observation_diagnostics.sources import load_gt

    log.name = "coverage"
    out, F, B, th = ctx["out"], ctx["F"], ctx["B"], ctx["th"]
    cdir = out / "coverage"
    cdir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    done_before = 0
    for n_f, f in enumerate(ctx["frames"]):
        rec = cdir / f"frame_{f:04d}.json"
        if rec.is_file() and json.loads(rec.read_text()).get("fingerprint") == ctx["fingerprint"]:
            done_before += 1
            continue
        t0 = time.time()
        stF, stB = F.state(f), B.state(f)
        accF = ge.new_frame_accumulator(F.N, "cuda")
        accB = ge.new_frame_accumulator(B.N, "cuda")
        views, sweeps = [], []
        for cam in ctx["cams"]:
            ci = ctx["all_cams"].index(cam)
            spec = ctx["specs"][ci]
            vF, vB = render_view(stF, spec), render_view(stB, spec)
            cF, cB = pixel_contributions(vF.cache), pixel_contributions(vB.cache)
            g_rgb, g_d, g_valid, g_files = load_gt(ctx["data"], cam, f, spec.image_height,
                                                   spec.image_width)
            Fd = {"rgb": vF.rgb, "depth": vF.depth, "alpha": vF.alpha}
            Bd = {"rgb": vB.rgb, "depth": vB.depth, "alpha": vB.alpha}
            gt = {"rgb": g_rgb, "depth": g_d, "valid": g_valid}
            fields = pe.classify(Fd, Bd, gt, th)
            row = {"frame": f, "camera": cam, **pe.view_metrics(fields, Fd, Bd, th)}
            row["F_conservation"] = conservation(cF, vF)
            row["B_conservation"] = conservation(cB, vB)
            row["F_contributors"] = contributor_counts(cF)
            row["B_contributors"] = contributor_counts(cB)
            row["F_accum"] = ge.accumulate(accF, cF, vF.cache.depth, fields, g_d, "F", ci,
                                           th.contrib_depth_tol_m)
            row["B_accum"] = ge.accumulate(accB, cB, vB.cache.depth, fields, g_d, "B", ci,
                                           th.contrib_depth_tol_m)
            row["gt_files"] = g_files
            views.append(row)
            if not ctx["no_sweep"]:
                for s in pe.sweep_rows(Fd, Bd, gt):
                    sweeps.append({"frame": f, "camera": cam, **s})
            # maps
            fn = {k: (fields[k].cpu().numpy()) for k in (
                "cover", "support", "layer", "eF", "eB", "dFB", "covF", "covB",
                "W_Bonly", "W_Fonly")}
            Fn = {k: v.cpu().numpy() for k, v in Fd.items()}
            Bn = {k: v.cpu().numpy() for k, v in Bd.items()}
            gn = {"depth": g_d.cpu().numpy(), "valid": g_valid.cpu().numpy(),
                  "rgb": g_rgb.cpu().numpy()}
            if f in ctx["map_frames"]:
                mdir = out / "pixel_maps" / f"frame_{f:04d}" / cam
                imgs = vz.view_maps(fn, Fn, Bn, gn, ctx["depth_range"])
                imgs["GT_RGB"] = vz._to_u8(gn["rgb"].transpose(1, 2, 0))
                imgs["F_RGB"] = vz._to_u8(Fn["rgb"].transpose(1, 2, 0))
                imgs["B_RGB"] = vz._to_u8(Bn["rgb"].transpose(1, 2, 0))
                for name, img in imgs.items():
                    vz.save_png(mdir / f"{name}.png", img)
                np.savez_compressed(
                    mdir / "numeric.npz", F_alpha=Fn["alpha"], B_alpha=Bn["alpha"],
                    F_depth=Fn["depth"], B_depth=Bn["depth"], cover=fn["cover"],
                    support=fn["support"], layer=fn["layer"],
                    codes=json.dumps({"cover": pe.COVER_CODES, "support": pe.SUPPORT_CODES,
                                      "layer": pe.LAYER_CODES}))
                if f in ctx["panel_frames"]:
                    vz.comparison_panel(
                        out / "visuals" / "source_comparison_panels" / f"frame_{f:04d}_{cam}.png",
                        f"{ctx['scene']}  original frame {f}  {cam}  (each source rendered alone)",
                        imgs, ctx["depth_range"])
            del vF, vB, cF, cB, fields
        arrs = {f"F_{k}": v for k, v in ge.to_numpy(accF).items()}
        arrs |= {f"B_{k}": v for k, v in ge.to_numpy(accB).items()}
        np.savez(cdir / f"frame_{f:04d}.npz", **arrs)
        if sweeps:
            (cdir / f"frame_{f:04d}_sweep.json").write_text(json.dumps(sweeps))
        dt = time.time() - t0
        rec.write_text(json.dumps({"fingerprint": ctx["fingerprint"], "frame": f,
                                   "seconds": dt, "views": views, **peak()}, default=float))
        n_left = len(ctx["frames"]) - n_f - 1
        cons = max(max(v["F_conservation"]["max_abs_sum_q_minus_A"],
                       v["B_conservation"]["max_abs_sum_q_minus_A"]) for v in views)
        log(f"  coverage frame {f:3d}: {len(views)} views, {dt:5.1f} s, "
            f"max|sum q - A| {cons:.1e}, ETA {n_left * dt / 60:5.1f} min  "
            f"GPU peak {peak()['peak_gpu_mib']} MiB")
    log(f"  coverage pass: {len(ctx['frames'])} frames ({done_before} reused), "
        f"{time.time() - t_all:.0f} s")
    return {"seconds": time.time() - t_all, "frames_reused": done_before, **peak()}


def _frame_views(ctx, f, prune_tau, only_cols=None, cams=None):
    from observation_diagnostics import pair_evidence as pv
    from observation_diagnostics.renderer_access import pixel_contributions
    from observation_diagnostics.sources import load_gt
    F, B, th = ctx["F"], ctx["B"], ctx["th"]
    stF, stB = F.state(f), B.state(f)
    views = []
    for cam in (cams or ctx["cams"]):
        ci = ctx["all_cams"].index(cam)
        spec = ctx["specs"][ci]
        vF, vB = render_view(stF, spec), render_view(stB, spec)
        cF, cB = pixel_contributions(vF.cache), pixel_contributions(vB.cache)
        _rgb, g_d, g_valid, _ = load_gt(ctx["data"], cam, f, spec.image_height,
                                        spec.image_width)
        vp = pv.view_pairs(cF, cB, vF.cache.depth, vB.cache.depth,
                           stF.colors.float(), stB.colors.float(),
                           stF.mu.float(), stB.mu.float(), g_d, g_valid, F.N, B.N, th,
                           prune_tau=prune_tau, max_pairs_per_chunk=ctx["pair_chunk"],
                           only_cols=only_cols)
        vp["stats"]["camera"] = cam
        views.append((ci, vp))
        del vF, vB, cF, cB
    return views


def exact_reference(ctx, f, log) -> dict:
    """A representative ROI enumerated with every pair stored (prune_tau = 0):
    the middle frame at two cameras (the first and the middle one), against the
    same views pruned. Reports the dropped mass and whether best partners move.
    (All eight cameras unpruned need ~4x the host RAM on the densest scenes.)"""
    import torch
    from observation_diagnostics import pair_evidence as pv
    t0 = time.time()
    cols = ("O", "O_surf_1")
    ref_cams = sorted({ctx["cams"][0], ctx["cams"][len(ctx["cams"]) // 2]})
    exact = _frame_views(ctx, f, 0.0, only_cols=cols, cams=ref_cams)
    pruned = []
    for ci, v in exact:
        keep = v["cols"]["O"] >= ctx["pair_prune"]
        pruned.append((ci, {"keys": v["keys"][keep],
                            "cols": {c: v["cols"][c][keep] for c in cols}}))

    def merge(views):
        k = np.concatenate([v["keys"] for _c, v in views])
        cs = {c: np.concatenate([v["cols"][c] for _c, v in views]) for c in cols}
        return pv._reduce_np(k, cs)

    ke, ce = merge(exact)
    kp, cp = merge(pruned)
    del exact, pruned
    NB = ctx["B"].N

    def best(keys, osurf, side):
        g = keys // NB if side == "F" else keys % NB
        o = np.lexsort((-osurf, g))
        g2 = g[o]
        st = np.flatnonzero(np.r_[True, g2[1:] != g2[:-1]])
        part = (keys % NB if side == "F" else keys // NB)[o][st]
        return dict(zip(g2[st].tolist(), part.tolist())), dict(zip(g2[st].tolist(), osurf[o][st].tolist()))

    res = {"frame": f, "cameras": ref_cams, "prune_tau": ctx["pair_prune"],
           "distinct_pairs_exact": int(ke.size), "distinct_pairs_stored": int(kp.size),
           "mass_O_exact": float(ce["O"].sum()), "mass_O_stored": float(cp["O"].sum()),
           "mass_O_surf_exact": float(ce["O_surf_1"].sum()),
           "mass_O_surf_stored": float(cp["O_surf_1"].sum())}
    res["fraction_O_surf_mass_dropped"] = 1 - res["mass_O_surf_stored"] / max(res["mass_O_surf_exact"], 1e-30)
    for side in ("F", "B"):
        be, ve = best(ke, ce["O_surf_1"], side)
        bp, _vp = best(kp, cp["O_surf_1"], side)
        rows = [g for g in be if ve[g] > 0]
        agree = sum(1 for g in rows if bp.get(g) == be[g])
        res[f"{side}_rows_with_same_surface_partner"] = len(rows)
        res[f"{side}_best_partner_agreement"] = agree / max(len(rows), 1)
        strong = [g for g in rows if ve[g] >= 1e-2]
        res[f"{side}_best_partner_agreement_best_O_surf_ge_1e-2"] = (
            sum(1 for g in strong if bp.get(g) == be[g]) / max(len(strong), 1))
        res[f"{side}_rows_best_O_surf_ge_1e-2"] = len(strong)
        wt = sum(ve[g] for g in rows)
        res[f"{side}_best_partner_agreement_weighted_by_O_surf"] = (
            sum(ve[g] for g in rows if bp.get(g) == be[g]) / max(wt, 1e-30))
    res["seconds"] = time.time() - t0
    torch.cuda.empty_cache()
    log(f"  exact reference, frame {f} {ref_cams}: {res['distinct_pairs_exact']:,} pairs exact vs "
        f"{res['distinct_pairs_stored']:,} stored; same-surface mass dropped "
        f"{100 * res['fraction_O_surf_mass_dropped']:.3f}%; best-partner agreement "
        f"F {100 * res['F_best_partner_agreement']:.1f}% / B {100 * res['B_best_partner_agreement']:.1f}% "
        f"of all rows, F {100 * res['F_best_partner_agreement_best_O_surf_ge_1e-2']:.1f}% / "
        f"B {100 * res['B_best_partner_agreement_best_O_surf_ge_1e-2']:.1f}% where the best O_surf >= 1e-2, "
        f"O_surf-weighted F {100 * res['F_best_partner_agreement_weighted_by_O_surf']:.1f}% / "
        f"B {100 * res['B_best_partner_agreement_weighted_by_O_surf']:.1f}%")
    return res


def pairs_pass(ctx, log):
    import torch
    from observation_diagnostics import pair_evidence as pv

    log.name = "pairs"
    out, F, B = ctx["out"], ctx["F"], ctx["B"]
    pdir = out / "pairs"
    pdir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    final = pdir / "global_final.npz"
    state_path = pdir / "checkpoint.json"
    if final.is_file() and state_path.is_file():
        st = json.loads(state_path.read_text())
        if st.get("fingerprint") == ctx["fingerprint"] and st.get("complete"):
            log("  pairs pass already complete for this fingerprint; reused")
            return {"seconds": 0.0, "reused": True}
    gp = pv.GlobalPairs(F.N, B.N)
    merged = []
    ckpt = pdir / "global_checkpoint.npz"
    if state_path.is_file() and ckpt.is_file():
        st = json.loads(state_path.read_text())
        if st.get("fingerprint") == ctx["fingerprint"]:
            gp.load(ckpt)
            merged = st["frames_merged"]
            log(f"  resumed the pair table from its checkpoint: {len(merged)} frames, "
                f"{gp.rows():,} pairs")
    todo = [f for f in ctx["frames"] if f not in merged]
    since = 0
    for n_f, f in enumerate(todo):
        t0 = time.time()
        views = _frame_views(ctx, f, ctx["pair_prune"])
        stats = [v["stats"] for _c, v in views]
        om = {k: sum(v[k] for _c, v in views) for k in (
            "omitted_O_F", "omitted_O_surf_F", "omitted_O_B", "omitted_O_surf_B")}
        ft = pv.frame_table(views)
        del views
        torch.cuda.empty_cache()
        gp.add(f, ft)
        np.savez(pdir / f"frame_{f:04d}_omitted.npz",
                 **{k: v.astype(np.float32) for k, v in om.items()})
        merged.append(f)
        since += 1
        dt = time.time() - t0
        (pdir / f"frame_{f:04d}.json").write_text(json.dumps(
            {"fingerprint": ctx["fingerprint"], "frame": f, "seconds": dt,
             "distinct_pairs_in_frame": int(ft["keys"].size), "views": stats, **peak()},
            default=float))
        if mem_available_gib() < ctx["min_free_ram_gib"] and n_f < len(todo) - 1:
            gp.save(ckpt)
            state_path.write_text(json.dumps({"fingerprint": ctx["fingerprint"],
                                              "frames_merged": merged, "complete": False}))
            raise HostRamLow(
                f"HOST_RAM_LOW: {mem_available_gib():.1f} GiB available after frame {f} "
                f"(sequence pair table {gp.nbytes() / 2**30:.1f} GiB). Checkpointed "
                f"{len(merged)} frames; rerun with --resume once memory is free.")
        if since >= 10 and n_f < len(todo) - 1:
            gp.save(ckpt)
            state_path.write_text(json.dumps({"fingerprint": ctx["fingerprint"],
                                              "frames_merged": merged, "complete": False}))
            since = 0
        n_left = len(todo) - n_f - 1
        log(f"  pairs frame {f:3d}: {sum(s['pairs'] for s in stats):,} pair records, "
            f"{sum(s['stored_pairs'] for s in stats):,} stored, frame {ft['keys'].size:,} distinct, "
            f"sequence table {gp.rows():,} ({gp.nbytes() / 2**30:.2f} GiB), {dt:5.1f} s, "
            f"ETA {n_left * dt / 60:5.1f} min, GPU peak {peak()['peak_gpu_mib']} MiB, "
            f"host peak {peak()['peak_host_rss_mib'] / 1024:.1f} GiB")
        del ft
    ref_frame = ctx["frames"][len(ctx["frames"]) // 2]
    ref = exact_reference(ctx, ref_frame, log)
    (pdir / "exact_reference.json").write_text(json.dumps(ref, indent=1))
    gp.save(final)
    state_path.write_text(json.dumps({"fingerprint": ctx["fingerprint"], "frames_merged": merged,
                                      "complete": True}))
    if ckpt.is_file():
        ckpt.unlink()
    log(f"  pairs pass: {len(ctx['frames'])} frames, {gp.rows():,} stored pairs, "
        f"{time.time() - t_all:.0f} s")
    return {"seconds": time.time() - t_all, "frames_resumed": len(ctx["frames"]) - len(todo),
            "stored_pairs": gp.rows(), "table_gib": gp.nbytes() / 2 ** 30, **peak()}


def depth_range(data: Path, cams, frames) -> tuple:
    vals = []
    for f in sorted({frames[0], frames[len(frames) // 2], frames[-1]}):
        for c in cams:
            p = data / "depth_npy" / c / f"frame_{f:06d}.npy"
            if p.is_file():
                d = np.load(p)
                vals.append(d[d > 0])
    v = np.concatenate(vals) if vals else np.array([0.0, 1.0])
    return (float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--results-root", default=os.environ.get("RESULTS_ROOT", DEFAULT_RESULTS))
    ap.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", DEFAULT_DATASET))
    ap.add_argument("--frames", default="all", help="all | a:b | comma list (original indices)")
    ap.add_argument("--cameras", default="all", help="all | comma list of camera names")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=("coverage", "pairs", "all"), default="all")
    ap.add_argument("--preflight", action="store_true", help="checks only, then exit")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, then exit")
    ap.add_argument("--resume", action="store_true",
                    help="continue an experiment with the same fingerprint")
    ap.add_argument("--maps", default="all",
                    help="frames whose per-pixel maps are written: all | none | comma list")
    ap.add_argument("--panel-frames", default="0,10,20,30,40,50,59")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the per-view threshold sweep (T11 then UNAVAILABLE)")
    ap.add_argument("--pair-chunk", type=int, default=4_000_000,
                    help="upper bound on pairs per GPU chunk; also capped by free GPU "
                         "memory (memory only: the sums are exact)")
    ap.add_argument("--pair-prune", type=float, default=1e-4,
                    help="store a view's (F,B) pair only if its per-view O_ij >= this; the "
                         "pruned mass is accounted per Gaussian and bounds every result "
                         "(0 = store all pairs, exact; needs far more host RAM)")
    ap.add_argument("--min-free-ram-gib", type=float, default=3.0,
                    help="checkpoint and stop the pair pass if available host RAM falls "
                         "below this (resume later with --resume)")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="write here instead of <results-root>/<scene>/"
                         f"{OUT_SUBDIR}/<tag> (tests and smoke runs)")
    args = ap.parse_args()

    from observation_diagnostics.sources import scene_paths
    paths = scene_paths(Path(args.results_root), Path(args.dataset_root), args.scene)
    out = (Path(args.out_dir) if args.out_dir
           else Path(args.results_root) / args.scene / OUT_SUBDIR / args.tag)
    log = Log(out)
    pf = preflight(args, paths, out)
    if args.preflight or args.dry_run:
        print(json.dumps(pf, indent=1, default=str))
        if args.preflight:
            out.mkdir(parents=True, exist_ok=True)
            (out / "preflight.json").write_text(json.dumps(pf, indent=1, default=str))
            return 2 if pf["blockers"] else 0
    if pf["blockers"]:
        print(json.dumps(pf, indent=1, default=str))
        print("BLOCKED:", "; ".join(pf["blockers"]))
        return 2
    all_frames_data = min(pf["dataset_frames_per_camera"].values())
    universe_frames = list(range(min(all_frames_data, pf["source_F"]["center_files"],
                                     pf["source_B"]["center_files"])))
    frames = parse_list(args.frames, universe_frames, "frames")
    cams = parse_list(args.cameras, pf["dataset_cameras"], "cameras")
    pilot = (frames != universe_frames) or (cams != pf["dataset_cameras"])
    map_frames = (set(frames) if args.maps == "all" else set() if args.maps == "none"
                  else {int(x) for x in args.maps.split(",")})
    panel_frames = {int(x) for x in args.panel_frames.split(",") if x.strip()}
    panel_frames = {f for f in panel_frames if f in frames} or {frames[0], frames[-1]}
    if args.dry_run:
        print(f"\nplan: {args.scene}  frames {frames[0]}..{frames[-1]} ({len(frames)}), "
              f"cameras {cams}, mode {args.mode}, {'PILOT' if pilot else 'FULL'}\n  -> {out}")
        return 0

    import torch
    from observation_diagnostics import aggregate
    from observation_diagnostics.pixel_evidence import Thresholds
    from observation_diagnostics.renderer_access import KERNEL_SHA256
    from observation_diagnostics.sources import (SourceTrack, check_gauge, hash_files,
                                                 protected_files)
    from tools.run_iekf_scene import camera_specs

    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight.json").write_text(json.dumps(pf, indent=1, default=str))
    t_start = time.time()
    prot = protected_files(paths)
    log(f"=== {args.scene}: hashing {len(prot)} protected files (T12, before)")
    before = hash_files(prot)
    data = Path(args.dataset_root) / args.scene
    log("  loading forward source (static_bidi/initial + bidirectional/forward)")
    F = SourceTrack("F", paths["sources"]["F"], data)
    log("  loading backward source (static_bidi/final + bidirectional/backward)")
    B = SourceTrack("B", paths["sources"]["B"], data)
    gauge = check_gauge(F, B)
    specs = camera_specs(F.cams_all, len(F.cams_all))
    all_cams = gauge["F_cameras"]
    if any(c not in all_cams for c in cams):
        raise SystemExit(f"cameras {cams} not in the models' {all_cams}")
    th = Thresholds()
    fp_payload = {"scene": args.scene, "frames": frames, "cameras": cams,
                  "F": [F.ply_sha256, F.track_sha256, F.state_mode],
                  "B": [B.ply_sha256, B.track_sha256, B.state_mode],
                  "thresholds": th.as_dict(), "code_sha256": code_hash(),
                  "kernel_sha256": KERNEL_SHA256, "no_sweep": args.no_sweep,
                  "pair_prune": args.pair_prune}
    fingerprint = hashlib.sha256(json.dumps(fp_payload, sort_keys=True).encode()).hexdigest()
    state_file = out / "run_state.json"
    if state_file.is_file():
        old = json.loads(state_file.read_text())
        if old.get("fingerprint") != fingerprint:
            raise SystemExit(
                f"{out} holds an experiment with fingerprint {old.get('fingerprint', '')[:16]}, "
                f"this run is {fingerprint[:16]} (inputs, frames, cameras, thresholds or code "
                "changed). Use a new --tag; resuming would mix two experiments.")
        if not args.resume:
            raise SystemExit(f"{out} already holds this experiment; pass --resume to continue "
                             "it, or use a new --tag")
    state_file.write_text(json.dumps({"fingerprint": fingerprint, "payload": fp_payload},
                                     indent=1))
    drange = depth_range(data, cams, frames)
    log(f"  F {F.N} Gaussians, B {B.N} Gaussians, {len(frames)} frames x {len(cams)} cameras"
        f" = {len(frames) * len(cams)} views per source, {'PILOT' if pilot else 'FULL'}; "
        f"state {F.state_mode}; fingerprint {fingerprint[:16]}")
    ctx = {"out": out, "F": F, "B": B, "th": th, "frames": frames, "cams": cams,
           "all_cams": all_cams, "specs": specs, "data": data, "scene": args.scene,
           "fingerprint": fingerprint, "depth_range": drange, "map_frames": map_frames,
           "panel_frames": panel_frames, "no_sweep": args.no_sweep,
           "pair_chunk": args.pair_chunk, "pair_prune": args.pair_prune,
           "min_free_ram_gib": args.min_free_ram_gib,
           "support_paths": {k: str(v["support"]) for k, v in paths["sources"].items()}}
    prov = {"scene": args.scene, "pilot": pilot, "frames": frames, "cameras": cams,
            "fingerprint": fingerprint, "fingerprint_payload": fp_payload,
            "sources": {"F": F.describe(), "B": B.describe()}, "camera_gauge": gauge,
            "depth_range_m": drange, "thresholds": th.as_dict(),
            "renderer": {"path": "estimator.renderer_bridge.build_linearization -> "
                                 "diff_gaussian_rasterization iekf_build_cache + iekf_render "
                                 "(production, unmodified)",
                         "contributions": "observation_diagnostics.renderer_access "
                                          "(standalone diagnostic kernel, same loop)",
                         "kernel_sha256": KERNEL_SHA256,
                         "background": "black (0,0,0), the estimator default",
                         "colour": "DC SH term only (frozen radiance), as Phase 1 rendered"},
            "gt_usage": "dataset RGB + metric depth only; no GT part labels, joint "
                        "parameters or masks are read",
            "code_sha256": code_hash(),
            "presentation_code_sha256": code_hash(PRESENTATION_CODE, include_tool=False),
            "protected_hashes_before": before,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    runtime = {}
    if args.mode in ("coverage", "all"):
        log(f"--- coverage pass ({len(frames)} frames)")
        runtime["coverage"] = coverage_pass(ctx, log)
    if args.mode in ("pairs", "all"):
        log(f"--- pair-evidence pass ({len(frames)} frames)")
        try:
            runtime["pairs"] = pairs_pass(ctx, log)
        except HostRamLow as e:
            log(str(e))
            return 3
    log.name = "visuals"
    log("--- aggregation, labels, case studies, reports")
    t0 = time.time()
    agg = aggregate.run(ctx, log, skip_report=args.skip_report)
    runtime["aggregate_visuals_report"] = {"seconds": time.time() - t0, **peak()}
    after = hash_files(prot)
    changed = [p for p in before if before[p] != after.get(p)]
    prov["protected_hashes_after"] = after
    prov["protected_files_changed"] = changed
    prov["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    runtime["total_seconds"] = time.time() - t_start
    (out / "metrics").mkdir(exist_ok=True)
    (out / "metrics" / "runtime_memory.json").write_text(json.dumps(runtime, indent=1))
    if changed:
        log(f"!! T12 FAILED: protected files changed: {changed}")
        return 1
    log(f"=== done: {args.scene} -> {out}  ({runtime['total_seconds'] / 60:.1f} min); "
        f"T12 no-mutation PASS ({len(before)} files); status {agg.get('status')}")
    return 0 if agg.get("status") in ("COMPLETE", "PILOT_COMPLETE") else 3


if __name__ == "__main__":
    raise SystemExit(main())
