#!/usr/bin/env python3
"""R7/R8 on real data: does render refinement improve the transform?

The stage §20 calls crucial, actually executed.  Physical hypotheses come
from the metric/SE(3) stages; this probe then runs

    T* = argmax_T log p(O | T)

against real images, and measures the LLR before and after so the gain can be
read rather than assumed (§45).  It also projects the refined 3D motion into
pixel flow and compares that against the measured pixel transport (§47 clause
6), which is the one check that can tell a transform that explains the images
from one that merely moved something somewhere.

Writes the §46 HTML: static render, initial dynamic render, refined dynamic
render, the observed frame, ownership and marginal heatmaps, and the LLR
table.  A refinement that did nothing is visible there immediately.

Run:
    PYTHONPATH=. python tools/run_render_refined_rigid_hypothesis_probe.py \
        --frame0-model-dir <f0> --frameT-model-dir <fT> \
        --root-scene <scene> --output-dir <out>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils import gpu_memory, progress
from utils.dynamic_pixel_marginals import project_gaussian_dynamic_posterior_to_pixels
from utils.local_rigid_hypothesis import solve_isometry_with_rigid_selection
from utils.metric_patch_graph import (
    build_full_geometry_knn, build_nested_metric_patches,
    compute_pairwise_metric_matrix, gather_patches, normalise_metric_matrix,
)
from utils.pixel_transport import solve_partial_pixel_transport
from utils.projected_pixel_flow import (
    compare_projected_flow_with_pixel_transport,
    rasterize_pixel_flow_from_gaussian_velocities,
)
from utils.render_diagnostics_html import image_entry, write_render_diagnostics_html
from utils.render_hypothesis_likelihood import (
    fit_predictive_rgb_residual_model, observed_image,
    render_rigid_patch_hypothesis, render_static,
)
from utils.render_refinement import refine_rigid_transform_by_render_likelihood
from utils.spatial_motion_posterior import compute_motion_posterior
from utils.static_pixel_calibration import calibrate_from_static_pixels
from tools.run_relational_graph_probe import _json_safe, _write_json


def json_finite(value):
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(v) for v in value]
    return value


def load_scene(model_dir, root_scene, iteration):
    from tools.test_render_denoise_discovery import build_scene
    return build_scene(SimpleNamespace(
        static_model_dir=model_dir, root_scene=root_scene, iteration=int(iteration),
        sh_degree=None, resolution=None, white_background=None,
        camera_data_device="cpu", use_depth=True, cameras="", fps=30.0,
    ))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame0-model-dir", required=True)
    p.add_argument("--frameT-model-dir", required=True)
    p.add_argument("--root-scene", required=True)
    p.add_argument("--iteration", type=int, default=14000)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--knn", type=int, default=48)
    p.add_argument("--patch-level", type=int, default=1)
    p.add_argument("--source-patches", type=int, default=16)
    p.add_argument("--target-patches", type=int, default=32)
    p.add_argument("--hypotheses", type=int, default=6, help="how many to refine")
    p.add_argument("--render-cameras", type=int, default=3)
    p.add_argument("--refine-iterations", type=int, default=8)
    p.add_argument("--html-panels", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-html", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    progress.set_enabled(not bool(args.no_progress))
    out = Path(args.output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(int(args.seed))
    results: dict[str, Any] = {"config": {k: _json_safe(v) for k, v in vars(args).items()}}

    with progress.stage("load endpoints"):
        scene0, g0, pipe, white = load_scene(args.frame0_model_dir, args.root_scene, int(args.iteration))
        sceneT, gT, _p, _w = load_scene(args.frameT_model_dir, args.root_scene, int(args.iteration))
    from tools.test_render_denoise_discovery import _ordered_complete_frame_groups
    groups = _ordered_complete_frame_groups(scene0.getTrainCameras())
    if len(groups) < 2:
        raise SystemExit("need at least two complete multi-view frames")
    frame0_id, cameras0 = groups[0]
    frameT_id, camerasT = groups[-1]
    device = g0.get_xyz.device
    background = torch.tensor([1.0, 1.0, 1.0] if white else [0.0, 0.0, 0.0], device=device)
    xyz0, xyzT = g0.get_xyz.detach(), gT.get_xyz.detach()
    progress.write(f"[load] frames {frame0_id}->{frameT_id}, {len(cameras0)} views | "
                   f"{int(xyz0.shape[0])} / {int(xyzT.shape[0])} Gaussians")

    # -------- P0/P2: get physical hypotheses to refine ---------------------
    with gpu_memory.scope("P0/P2"), progress.stage("P0/P2 hypotheses"):
        knn0, d0dist = build_full_geometry_knn(xyz0, int(args.knn), chunk_size=int(args.chunk_size), desc="  knn0")
        knnT, dTdist = build_full_geometry_knn(xyzT, int(args.knn), chunk_size=int(args.chunk_size), desc="  knnT")
        posterior = compute_motion_posterior(
            xyz0, xyzT, d0dist[:, 1:].median(1).values.clamp_min(1e-8),
            dTdist[:, 1:].median(1).values.clamp_min(1e-8), chunk_size=int(args.chunk_size))
        patches0 = build_nested_metric_patches(xyz0, knn0, desc="  rank0")
        patchesT = build_nested_metric_patches(xyzT, knnT, desc="  rankT")
        gpu_memory.release(d0dist, dTdist)

    pool0 = (posterior.source_map_dynamic & patches0.full_rank).nonzero(as_tuple=True)[0]
    poolT = (posterior.target_map_dynamic & patchesT.full_rank).nonzero(as_tuple=True)[0]
    if int(pool0.numel()) < 4 or int(poolT.numel()) < 4:
        results["verdict"] = {"result": "no dynamic domain"}
        _write_json(out / "render_refined_probe.json", json_finite(_json_safe(results))); return 0

    def pick(pool, n):
        if int(pool.numel()) <= int(n): return pool
        c = torch.randperm(int(pool.numel()), generator=generator)[: int(n)]
        return pool[c.sort().values.to(pool.device)]

    anchors0, anchorsT = pick(pool0, int(args.source_patches)), pick(poolT, int(args.target_patches))
    idx0, valid0 = gather_patches(patches0, anchors0, level=int(args.patch_level))
    idxT, validT = gather_patches(patchesT, anchorsT, level=int(args.patch_level))
    d0, scale0 = normalise_metric_matrix(compute_pairwise_metric_matrix(xyz0, idx0), valid=valid0)
    dT, _ = normalise_metric_matrix(compute_pairwise_metric_matrix(xyzT, idxT), valid=validT)

    with progress.stage("P3-P4 best physical hypothesis per source patch"):
        best_res = torch.full((int(anchors0.numel()),), float("inf"))
        best = {}
        for t in progress.bar(range(int(anchorsT.numel())), desc="  targets", unit="tgt", leave=True):
            index = torch.full((int(anchors0.numel()),), t, device=device, dtype=torch.long)
            _c, _d, hyp = solve_isometry_with_rigid_selection(
                xyz0[idx0], xyzT[idxT[index]], d0, dT[index], valid0, validT[index], patch_scale=scale0)
            res = torch.where(hyp.degenerate, torch.full_like(hyp.residual, float("inf")), hyp.residual).cpu()
            better = res < best_res
            for key, value in (("rotation", hyp.rotation.cpu()), ("translation", hyp.translation.cpu())):
                stored = best.get(key)
                best[key] = value if stored is None else torch.where(
                    better[:, None, None] if value.ndim == 3 else better[:, None], value, stored)
            best["target"] = torch.where(better, torch.full((int(anchors0.numel()),), t), best.get(
                "target", torch.zeros(int(anchors0.numel()), dtype=torch.long)))
            best_res = torch.where(better, res, best_res)

    order = torch.argsort(best_res)[: int(args.hypotheses)]
    progress.write(f"[P4] refining the {int(order.numel())} best of {int(anchors0.numel())} hypotheses "
                   f"(SE3 residual {float(best_res[order].min()):.4f}-{float(best_res[order].max()):.4f})")

    # -------- R2: calibrate on held-out frame-T views -----------------------
    with progress.stage("R2 predictive calibration"):
        m0 = project_gaussian_dynamic_posterior_to_pixels(
            cameras0[0], g0, pipe, posterior.source_dynamic_posterior)
        mT = project_gaussian_dynamic_posterior_to_pixels(
            camerasT[0], gT, pipe, posterior.target_dynamic_posterior)
        image0 = observed_image(cameras0[0]).to(device)
        imageT = observed_image(camerasT[0]).to(device)
        calibration = calibrate_from_static_pixels(m0, mT, image0, imageT, generator=generator)
        static_mask = (posterior.source_dynamic_posterior <= 0.5).to(device)
        model = fit_predictive_rgb_residual_model(
            camerasT, g0, pipe, background, static_gaussian_mask=static_mask,
            max_cameras=int(args.render_cameras))
    progress.write(f"[R2] predictive residual scale {model.scale:.5g} from {model.samples} px "
                   f"on held-out frame-{frameT_id} views")
    results["R2"] = {"residual_scale": model.scale, "residual_samples": model.samples,
                     "appearance_scale": calibration.appearance_scale,
                     "calibration": "held-out frame-T views, static regions"}

    # -------- R3: pixel transport (for the R8 comparison) -------------------
    with progress.stage("R3 pixel transport"):
        transport = solve_partial_pixel_transport(
            m0, mT, image0, imageT, calibration, camera_id=0, generator=generator)
    results["R3"] = dict(transport.stats)

    # -------- R7 + R8 -------------------------------------------------------
    rows, panels = [], []
    width = int(m0.total_mass.shape[1])
    with progress.stage("R7 render refinement + R8 projected flow"):
        for rank, index in enumerate(progress.bar(order.tolist(), desc="  hypotheses", unit="hyp", leave=True)):
            with gpu_memory.scope(f"hyp {index}"):
                mask = torch.zeros(int(xyz0.shape[0]), dtype=torch.bool)
                mask[idx0[index].cpu()[valid0[index].cpu()]] = True
                mask = mask.to(device)
                rotation0 = best["rotation"][index].to(device)
                translation0 = best["translation"][index].to(device)

                refined = refine_rigid_transform_by_render_likelihood(
                    camerasT, g0, pipe, background, mask, rotation0, translation0, model,
                    max_cameras=int(args.render_cameras), iterations=int(args.refine_iterations),
                    patch_scale=float(scale0[index]))

                # R8: does the refined 3D motion's projected flow agree with
                # the measured pixel transport?
                velocity = torch.zeros_like(xyz0)
                moved = xyz0[mask] @ refined.rotation.T + refined.translation
                velocity[mask] = moved - xyz0[mask]
                flow = rasterize_pixel_flow_from_gaussian_velocities(
                    cameras0[0], g0, pipe, xyz0, velocity)
                agreement = compare_projected_flow_with_pixel_transport(flow, transport, width)

                row = {
                    "hypothesis": int(index), "rank": rank,
                    "se3_residual": float(best_res[index]),
                    "LLR_before": refined.log_likelihood_ratio_before,
                    "LLR_after": refined.log_likelihood_ratio_after,
                    "LLR_gain": refined.stats["llr_gain"],
                    "improved": bool(refined.improved),
                    "evaluations": refined.evaluations,
                    "flow_error_px": agreement.get("median_flow_error_px"),
                    "flow_agreement": agreement.get("agreement_fraction"),
                }
                rows.append(row)
                progress.write(
                    f"    hyp {int(index):3d}  LLR {refined.log_likelihood_ratio_before:+.4g} -> "
                    f"{refined.log_likelihood_ratio_after:+.4g} (gain {refined.stats['llr_gain']:+.4g}) "
                    f"| flow err {agreement.get('median_flow_error_px', float('nan')):.2f}px")

                if not args.no_html and rank < int(args.html_panels):
                    camera = camerasT[0]
                    static_render = render_static(camera, g0, pipe, background)
                    initial = render_rigid_patch_hypothesis(
                        camera, g0, pipe, background, mask, rotation0, translation0)
                    final = render_rigid_patch_hypothesis(
                        camera, g0, pipe, background, mask, refined.rotation, refined.translation)
                    target_observed = observed_image(camera).to(device)
                    panels.append({
                        "title": f"hypothesis {int(index)} (rank {rank}) -- LLR "
                                 f"{refined.log_likelihood_ratio_before:+.3g} -> {refined.log_likelihood_ratio_after:+.3g}",
                        "images": [
                            image_entry(target_observed, f"observed frame {frameT_id}"),
                            image_entry(static_render, "static hypothesis"),
                            image_entry(initial, "dynamic hypothesis (initial)"),
                            image_entry(final, "dynamic hypothesis (refined)"),
                            image_entry((target_observed - static_render).abs().mean(0), "static residual"),
                            image_entry((target_observed - final).abs().mean(0), "refined residual"),
                            image_entry((initial - final).abs().amax(0), "what refinement changed"),
                        ],
                        "table": {"cols": ["metric", "value"], "rows": [
                            {"metric": "SE3 residual", "value": float(best_res[index])},
                            {"metric": "LLR before", "value": refined.log_likelihood_ratio_before},
                            {"metric": "LLR after", "value": refined.log_likelihood_ratio_after},
                            {"metric": "LLR gain", "value": refined.stats["llr_gain"]},
                            {"metric": "flow error px", "value": agreement.get("median_flow_error_px", float("nan"))},
                            {"metric": "flow agreement", "value": agreement.get("agreement_fraction", float("nan"))},
                        ]},
                    })
                    del static_render, initial, final, target_observed
                del flow, velocity

    gains = [r["LLR_gain"] for r in rows]
    improved = [r for r in rows if r["improved"]]
    results["R7_R8"] = {
        "hypotheses_refined": len(rows),
        "improved_count": len(improved),
        "mean_llr_gain": float(sum(gains) / max(1, len(gains))),
        "median_flow_error_px": float(sorted(r["flow_error_px"] or 0.0 for r in rows)[len(rows) // 2]) if rows else None,
        "per_hypothesis": rows,
    }
    results["verdict"] = {
        "render_refinement_improves_transform": bool(len(improved) > 0),
        "improved_fraction": float(len(improved) / max(1, len(rows))),
        "result": (f"RENDER REFINEMENT IMPROVED {len(improved)}/{len(rows)} TRANSFORMS"
                   if improved else "RENDER REFINEMENT IMPROVED NOTHING"),
    }

    if not args.no_html:
        with progress.stage("R46 html"):
            ownership = m0.total_mass
            payload = {"views": {
                "refinement panels": {
                    "note": ("Each row is one physical hypothesis. If 'dynamic (refined)' looks identical "
                             "to 'dynamic (initial)', refinement did nothing -- which no summary number "
                             "says as directly."),
                    "panels": panels,
                    "table": {"cols": ["hypothesis", "se3_residual", "LLR_before", "LLR_after",
                                       "LLR_gain", "flow_error_px", "flow_agreement"], "rows": rows},
                },
                "scene maps": {
                    "note": "R0/R1: rasteriser ownership and the dynamic pixel marginals it carries.",
                    "panels": [{"title": "ownership and marginals", "images": [
                        image_entry(ownership, "R0 ownership mass (frame 0)"),
                        image_entry(m0.dynamic_mass, "R1 dynamic pixel mass (frame 0)"),
                        image_entry(m0.static_mass, "R1 static pixel mass (frame 0)"),
                        image_entry(m0.dynamic_fraction, "R1 dynamic fraction"),
                        image_entry(mT.dynamic_mass, f"R1 dynamic pixel mass (frame {frameT_id})"),
                        image_entry(image0, f"observed frame {frame0_id}"),
                        image_entry(imageT, f"observed frame {frameT_id}"),
                    ]}],
                },
                "calibration": {
                    "note": ("R2 calibrates on held-out frame-T views and static regions only -- "
                             "training-view residuals measure reconstruction overfit, not predictive noise."),
                    "table": {"cols": ["quantity", "value"], "rows": [
                        {"quantity": "predictive RGB residual scale", "value": model.scale},
                        {"quantity": "calibration pixels", "value": model.samples},
                        {"quantity": "static appearance scale", "value": calibration.appearance_scale},
                        {"quantity": "static displacement scale px", "value": calibration.displacement_scale},
                        {"quantity": "transported pixel mass", "value": transport.stats.get("transported_mass", 0.0)},
                        {"quantity": "unmatched source fraction", "value": transport.stats.get("unmatched_source_fraction", 0.0)},
                    ]},
                },
            }}
            html = out / "render_refinement_diagnostics.html"
            write_render_diagnostics_html(html, payload)
            progress.write(f"[html] wrote {html}")
            results["html"] = str(html)

    _write_json(out / "render_refined_probe.json", json_finite(_json_safe(results)))
    print(f"\n=== {results['verdict']['result']} ===")
    print(f"  mean LLR gain {results['R7_R8']['mean_llr_gain']:+.4g} over {len(rows)} hypotheses")
    print(f"  median projected-flow error {results['R7_R8']['median_flow_error_px']} px")
    print(f"\n[probe] wrote {out / 'render_refined_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
