#!/usr/bin/env python
"""Gate A / Gate B benchmark of the corrected Phase 4A on one real (or synthetic) upstream run. NOT a result.

  1. caching: the pre-correction ViewStore(cache=64) over two sequential sweeps vs PreparedViews (decode
     counters, bytes read, seconds)
  2. groups of the non-core originals (sizes)
  3. a FIXED deterministic set of --groups groups (evenly spaced in group id) scored on the selected views by
     (a) the exact ROI scorer (phase4a_exact.ExactEvaluator) and (b) an independent FULL-IMAGE reference
     (full render + full per-pixel maps of both hypotheses); per-channel parity and decision agreement
  4. measured throughput (item-views / s), per-100-group cost, peak RSS / CUDA, and an extrapolation of the
     complete run's score operations from the MEASURED counts (not a wall-time guess)

Writes <output>/benchmark_4a.json and a line-buffered <output>/benchmark.log. Every output is NOT_FINAL.

    python tools/benchmark_phase4a.py --source SRC --direction forward --model RES/static_bidi/initial \\
        --phase2-dir RES/bidirectional/forward_rigid --phase3-dir RES/bidirectional/forward_joint \\
        --joint-type revolute --cameras cam_001,cam_002 --groups 50 --view-subset 24 --output OUT
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import recovery  # noqa: E402,F401


def main(argv=None):
    import numpy as np
    import torch
    from recovery import footprint as FP
    from recovery import phase4a_exact as X
    from recovery.exact_objective import CHANNELS, ExactLossConfig, ViewTargets
    from recovery.gaussians import GaussianTensors
    from recovery.io_setup import add_common_cli, setup_phase4
    from recovery.material_completion import Phase4Scene, build_components, surface_normals
    from recovery.rgbd_loss import LossConfig
    from recovery.runlog import RunLog, atomic_json, stdout_line_buffered
    from recovery.views import ViewStore
    stdout_line_buffered()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_cli(ap)
    ap.add_argument("--groups", type=int, default=50)
    ap.add_argument("--view-subset", type=int, default=24, help="frames per camera (0 = all); NOT_FINAL either way")
    ap.add_argument("--thrash-demo", type=int, default=1, help="1: also run the old ViewStore over two sweeps")
    ap.add_argument("--reference", type=int, default=1, help="1: also score with the full-image reference")
    ap.add_argument("--max-host-gb", type=float, default=8.0)
    a = ap.parse_args(argv)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    log = RunLog(out / "benchmark.log")
    rep = {"NOT_FINAL": True, "argv": sys.argv}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    up, cams, views, pv, extra = setup_phase4(a, log=log)
    rep["prepare"] = {"seconds": time.time() - t0, **{k: v for k, v in pv.manifest().items() if k != "per_view"}}
    log(f"prepared {len(pv)} views once: {pv.counters.to_dict()}")
    # 1. caching
    t1 = time.time()
    for _ in range(2):
        for i in range(len(pv)):
            pv.gpu_view(i)
    torch.cuda.synchronize()
    rep["prepared_two_sweeps"] = {"seconds": time.time() - t1, "counters_after": pv.counters.to_dict()}
    log(f"PreparedViews, two more sweeps: {time.time() - t1:.2f} s, decodes still {pv.counters.image_decodes}")
    if a.thrash_demo:
        st = ViewStore(up.source, {c.name: c for c in cams}, cache=64)
        t1 = time.time()
        for _ in range(2):
            for k in pv.keys:
                st.get(k.camera, k.frame)
        rep["old_viewstore_cache64_two_sweeps"] = {"seconds": time.time() - t1, "counters": st.counters,
                                                    "views": len(pv)}
        log(f"old ViewStore(cache=64), two sweeps over {len(pv)} views: {st.counters} in {time.time() - t1:.1f} s")
    # 2. groups
    g = GaussianTensors.load_ply(up.ply).to("cuda")
    g.active_sh_degree = 0
    scene = Phase4Scene(g, up.joint, views, LossConfig(), sh_degree=0)
    pol = X.Policy4A()
    tt = scene.tensors()
    xyz = tt["xyz"].cpu().numpy()
    cand = np.setdiff1d(np.arange(len(g)), up.members)
    nrm, flat = surface_normals(tt["rot"].cpu().numpy(), tt["scale"].cpu().numpy())
    comps = build_components(xyz, cand, pol.component_voxel, pol.max_component, nrm, flat, pol.normal_cos, pol.flat_max,
                             extent=tt["scale"].cpu().numpy().max(1), link_factor=pol.link_factor, layer_tol=pol.layer_tol)
    sz = np.array([len(c) for c in comps])
    rep["groups"] = {"n": len(comps), "candidates": int(len(cand)), "core": int(len(up.members)),
                     "size_quantiles": np.percentile(sz, [0, 25, 50, 75, 90, 100]).tolist(), "singletons": int((sz == 1).sum()),
                     "link_factor": pol.link_factor}
    log(f"groups: {rep['groups']}")
    # 3. fixed set
    pick = sorted(set(np.linspace(0, len(comps) - 1, min(a.groups, len(comps))).round().astype(int).tolist()))
    groups = [X.Group(int(k), comps[k]) for k in pick]
    cfg = ExactLossConfig()
    ev = X.ExactEvaluator(scene, pv, cfg, log, progress_every_s=30)
    labels = torch.zeros(len(g), dtype=torch.bool, device="cuda")
    labels[torch.as_tensor(up.members, device="cuda")] = True
    t1 = time.time()
    ev.footprints()
    rects = ev.group_rects(groups)
    rep["footprints_seconds"] = time.time() - t1
    torch.cuda.synchronize()
    t1 = time.time()
    r = ev.sweep([[x] for x in groups], labels, rects, tag="exact ROI")
    torch.cuda.synchronize()
    dt = time.time() - t1
    scored = int((r["skip"] == X.SKIP_NONE).sum())
    rep["exact_roi"] = {"seconds": dt, "item_views_scored": scored, "item_views_per_s": scored / dt,
                        "seconds_per_100_groups_all_selected_views": dt / len(groups) * 100,
                        "skip_counts": {str(k): int((r["skip"] == k).sum()) for k in (0, 1, 2)},
                        "counters": ev.counters.to_dict()}
    log(f"exact ROI: {scored} group-views in {dt:.1f} s ({scored / dt:.0f}/s)")
    # 4. full-image reference
    if a.reference:
        dev = "cuda"
        ref = np.zeros((len(groups), len(views), 4))
        t1 = time.time()
        for vi, vr in enumerate(views):
            tgt = ViewTargets(pv.gpu_view(vi, dev), cfg)
            base = scene.render(vr, labels)
            bm = tgt.full_maps(base)
            for k, x in enumerate(groups):
                if r["skip"][k, vi] != X.SKIP_NONE:
                    continue
                alt = labels.clone()
                alt[torch.as_tensor(x.members, device=dev)] = True
                am = tgt.full_maps(scene.render(vr, alt))
                ref[k, vi] = torch.stack([(am[c] - bm[c]).sum() for c in CHANNELS]).cpu().numpy()
        dtr = time.time() - t1
        diff = np.abs(r["d_alt"] - ref)
        scale = np.abs(ref).max()
        blocks = X.blocks_of(views, pol.block_frames)
        agree = 0
        for k in range(len(groups)):
            d1 = X.decide(-r["d_alt"][k], r["changed"][k], ev.HW, blocks, ev.cam_of_view, pol)[0]
            d2 = X.decide(-ref[k], r["changed"][k], ev.HW, blocks, ev.cam_of_view, pol)[0]
            agree += d1 == d2
        rep["full_reference"] = {"seconds": dtr, "max_abs_diff": float(diff.max()),
                                 "max_abs_reference": float(scale), "max_rel_diff": float(diff.max() / max(scale, 1e-300)),
                                 "decisions_agree": int(agree), "groups": len(groups),
                                 "speedup_exact_vs_reference": dtr / max(dt, 1e-9)}
        log(f"full reference: {dtr:.1f} s; max |ROI - full| {diff.max():.3e} (max |full| {scale:.3e}); "
            f"decisions agree {agree}/{len(groups)}")
    # extrapolation from measured counts
    frac = scored / max(1, len(groups) * len(views))
    rate = scored / dt
    V_all = len(up.joint.frames) * len(cams)
    est = len(comps) * V_all * frac
    rep["extrapolation_from_measured"] = {
        "all_groups": len(comps), "all_views": V_all, "measured_scored_fraction": frac, "measured_rate_per_s": rate,
        "one_full_sweep_item_views": est, "one_full_sweep_minutes_at_measured_rate": est / rate / 60,
        "note": "4A needs >= 2 full sweeps (initial + final verification with OMIT ~ 2x renders) plus batch re-scores "
                "of interacting groups; the batch count is data-dependent and is logged by the real run"}
    rep["resources"] = log.resources()
    atomic_json(out / "benchmark_4a.json", rep)
    log(f"wrote {out / 'benchmark_4a.json'}: {json.dumps(rep['extrapolation_from_measured'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
