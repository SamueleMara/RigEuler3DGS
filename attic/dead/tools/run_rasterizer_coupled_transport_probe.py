#!/usr/bin/env python3
"""R0-R5 on real data: does image support fix candidate starvation?

The physics-first pipeline's binding constraint was never the physics -- it
was that each source patch could only afford to test a few dozen of several
thousand target anchors, drawn at random, so the true counterpart was almost
never in the candidate set.  §52.1 makes the claim this probe has to test:
image-supported search materially increases true physical candidate recall.

The chain is P0 -> R0 ownership -> R1 marginals -> R2 static calibration ->
R3 pixel transport -> R4 lift -> R5 ordering, and the measurement at the end
is recall of the exhaustive physical best candidate under the image-supported
order versus under random order (§44).

§15: image support ORDERS candidates. It never accepts one.

Run:
    PYTHONPATH=. python tools/run_rasterizer_coupled_transport_probe.py \
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
from utils.lifted_image_transport import (
    candidate_recall_audit,
    fuse_multiview_gaussian_support,
    lift_pixel_transport_to_gaussian_support,
    score_patch_pairs_from_gaussian_support,
)
from utils.metric_patch_graph import (
    build_full_geometry_knn,
    build_nested_metric_patches,
    compute_pairwise_metric_matrix,
    gather_patches,
    normalise_metric_matrix,
)
from utils.pixel_transport import solve_partial_pixel_transport, summarise_pixel_transport
from utils.rasterizer_ownership import build_patch_indicator_features, splat_gaussian_features
from utils.render_hypothesis_likelihood import observed_image
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


def load_scene(model_dir: str, root_scene: str, iteration: int):
    from tools.test_render_denoise_discovery import build_scene

    return build_scene(SimpleNamespace(
        static_model_dir=model_dir, root_scene=root_scene, iteration=int(iteration),
        sh_degree=None, resolution=None, white_background=None,
        camera_data_device="cpu", use_depth=True, cameras="", fps=30.0,
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame0-model-dir", required=True)
    parser.add_argument("--frameT-model-dir", required=True)
    parser.add_argument("--root-scene", required=True)
    parser.add_argument("--iteration", type=int, default=14000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cameras", type=int, default=6)
    parser.add_argument("--knn", type=int, default=48)
    parser.add_argument("--patch-level", type=int, default=1)
    parser.add_argument("--source-patches", type=int, default=48)
    parser.add_argument("--target-patches", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--transport-pixels", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--save-tensors", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    progress.set_enabled(not bool(args.no_progress))
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(int(args.seed))
    results: dict[str, Any] = {"config": {k: _json_safe(v) for k, v in vars(args).items()}}

    with progress.stage("load endpoints"):
        scene0, g0, pipe, white = load_scene(args.frame0_model_dir, args.root_scene, int(args.iteration))
        sceneT, gT, _pipeT, _whiteT = load_scene(args.frameT_model_dir, args.root_scene, int(args.iteration))

    # The two endpoint MODELS are different, but both scenes expose the whole
    # video's cameras. Taking index i from each gives the SAME frame twice,
    # so the "endpoint" pixel transport would compare an image with itself:
    # measured that way the static appearance scale came out at exactly
    # 0.0000 and the transport cost was degenerate. The source must be the
    # first complete multi-view frame and the target the last.
    from tools.test_render_denoise_discovery import _ordered_complete_frame_groups

    groups = _ordered_complete_frame_groups(scene0.getTrainCameras())
    if len(groups) < 2:
        raise SystemExit("the rasterizer coupling needs at least two complete multi-view frames")
    frame0_id, cameras0 = groups[0]
    frameT_id, camerasT = groups[-1]
    progress.write(f"[load] endpoint frames: {frame0_id} -> {frameT_id} "
                   f"({len(cameras0)} synchronised views each, {len(groups)} frames available)")
    device = g0.get_xyz.device
    background = torch.tensor([1.0, 1.0, 1.0] if white else [0.0, 0.0, 0.0], device=device)
    xyz0, xyzT = g0.get_xyz.detach(), gT.get_xyz.detach()
    progress.write(f"[load] frame0 {int(xyz0.shape[0])} Gaussians, frameT {int(xyzT.shape[0])}")
    results["endpoint_frames"] = {"source": int(frame0_id), "target": int(frameT_id),
                                  "views_per_frame": len(cameras0)}

    # ---------------- P0 ---------------------------------------------------
    with gpu_memory.scope("P0"), progress.stage("P0 posterior"):
        knn0, dist0 = build_full_geometry_knn(xyz0, int(args.knn), chunk_size=int(args.chunk_size), desc="  knn0")
        knnT, distT = build_full_geometry_knn(xyzT, int(args.knn), chunk_size=int(args.chunk_size), desc="  knnT")
        posterior = compute_motion_posterior(
            xyz0, xyzT, dist0[:, 1:].median(1).values.clamp_min(1e-8),
            distT[:, 1:].median(1).values.clamp_min(1e-8), chunk_size=int(args.chunk_size),
        )
        gpu_memory.release(dist0, distT)
    q0, qT = posterior.source_dynamic_posterior, posterior.target_dynamic_posterior
    progress.write(f"[P0] MAP dynamic src {posterior.stats['source_map_dynamic_fraction']:.3f} "
                   f"tgt {posterior.stats['target_map_dynamic_fraction']:.3f}")
    results["P0"] = dict(posterior.stats)

    # ---------------- P2 patches -------------------------------------------
    with gpu_memory.scope("P2"), progress.stage("P2 patches"):
        patches0 = build_nested_metric_patches(xyz0, knn0, desc="  rank0")
        patchesT = build_nested_metric_patches(xyzT, knnT, desc="  rankT")
    pool0 = (posterior.source_map_dynamic & patches0.full_rank).nonzero(as_tuple=True)[0]
    poolT = (posterior.target_map_dynamic & patchesT.full_rank).nonzero(as_tuple=True)[0]
    if int(pool0.numel()) < 4 or int(poolT.numel()) < 4:
        results["verdict"] = {"result": "no dynamic domain"}
        _write_json(out / "rasterizer_coupled_probe.json", json_finite(_json_safe(results)))
        return 0

    def pick(pool, count):
        if int(pool.numel()) <= int(count):
            return pool
        chosen = torch.randperm(int(pool.numel()), generator=generator)[: int(count)]
        return pool[chosen.sort().values.to(pool.device)]

    anchors0, anchorsT = pick(pool0, int(args.source_patches)), pick(poolT, int(args.target_patches))
    idx0, valid0 = gather_patches(patches0, anchors0, level=int(args.patch_level))
    idxT, validT = gather_patches(patchesT, anchorsT, level=int(args.patch_level))
    progress.write(f"[P2] source patches {int(anchors0.numel())} target patches {int(anchorsT.numel())} "
                   f"(dynamic domain {int(poolT.numel())})")

    # ---------------- R0/R1/R2/R3/R4 per camera ----------------------------
    footprints0 = build_patch_indicator_features(int(xyz0.shape[0]), idx0.cpu(), valid0.cpu(), device=device)
    footprintsT = build_patch_indicator_features(int(xyzT.shape[0]), idxT.cpu(), validT.cpu(), device=device)

    per_camera_support: list[torch.Tensor] = []
    transports = []
    marginal_stats = []
    with progress.stage("R0-R4 rasterizer coupling"):
        for index in progress.bar(range(min(int(args.cameras), len(cameras0))), desc="  cameras", unit="cam", leave=True):
            with gpu_memory.scope(f"camera {index}"):
                camera0, cameraT = cameras0[index], camerasT[index]
                m0 = project_gaussian_dynamic_posterior_to_pixels(camera0, g0, pipe, q0, camera_index=index)
                mT = project_gaussian_dynamic_posterior_to_pixels(cameraT, gT, pipe, qT, camera_index=index)
                marginal_stats.append(m0.stats)

                image0 = observed_image(camera0).to(device)
                imageT = observed_image(cameraT).to(device)
                calibration = calibrate_from_static_pixels(m0, mT, image0, imageT, generator=generator)

                transport = solve_partial_pixel_transport(
                    m0, mT, image0, imageT, calibration, camera_id=index,
                    max_source_pixels=int(args.transport_pixels),
                    max_target_pixels=int(args.transport_pixels), generator=generator,
                )
                transports.append(transport)

                splat0 = splat_gaussian_features(camera0, g0, pipe, footprints0)
                splatT = splat_gaussian_features(cameraT, gT, pipe, footprintsT)
                per_camera_support.append(
                    lift_pixel_transport_to_gaussian_support(transport, splat0, splatT).cpu()
                )
                progress.write(
                    f"    cam {index}: dyn pixel mass {m0.stats['dynamic_mass_fraction']:.3f} | "
                    f"calib disp {calibration.displacement_scale:.2f}px appearance {calibration.appearance_scale:.4f} | "
                    f"transported {transport.stats.get('transported_mass', 0.0):.1f} "
                    f"unmatched src {transport.stats.get('unmatched_source_fraction', 0.0):.3f}"
                )
                del splat0, splatT, image0, imageT

    results["R1_marginals"] = {
        "cameras": len(marginal_stats),
        "mean_dynamic_mass_fraction": float(
            sum(s["dynamic_mass_fraction"] for s in marginal_stats) / max(1, len(marginal_stats))
        ),
        "mean_covered_pixel_fraction": float(
            sum(s["covered_pixel_fraction"] for s in marginal_stats) / max(1, len(marginal_stats))
        ),
    }
    results["R3_transport"] = summarise_pixel_transport(transports)

    # ---------------- R5 ordering and the recall audit ---------------------
    lifted = fuse_multiview_gaussian_support(per_camera_support)
    order, ordering_stats = score_patch_pairs_from_gaussian_support(
        lifted, candidates=int(args.candidates), fallback_generator=generator
    )
    results["R4_R5_support"] = {**lifted.stats, **ordering_stats}
    progress.write(
        f"[R5] image support: {ordering_stats['anchors_with_image_support']}/{ordering_stats['sources']} "
        f"anchors supported | pairs with support {lifted.stats['pairs_with_support']} "
        f"| density {lifted.stats['support_density']:.4f}"
    )

    # §44 audit: the exhaustive physical best candidate for each source patch,
    # then where the image-supported order puts it.
    with progress.stage("R5 exhaustive recall audit"):
        from utils.local_rigid_hypothesis import solve_isometry_with_rigid_selection

        d0, scale0 = normalise_metric_matrix(compute_pairwise_metric_matrix(xyz0, idx0), valid=valid0)
        dT, _ = normalise_metric_matrix(compute_pairwise_metric_matrix(xyzT, idxT), valid=validT)
        best_residual = torch.full((int(anchors0.numel()),), float("inf"))
        best_target = torch.zeros(int(anchors0.numel()), dtype=torch.long)
        for target in progress.bar(range(int(anchorsT.numel())), desc="  exhaustive", unit="tgt", leave=True):
            index = torch.full((int(anchors0.numel()),), target, device=device, dtype=torch.long)
            _coupling, _distortion, hypothesis = solve_isometry_with_rigid_selection(
                xyz0[idx0], xyzT[idxT[index]], d0, dT[index], valid0, validT[index], patch_scale=scale0,
            )
            residual = torch.where(
                hypothesis.degenerate, torch.full_like(hypothesis.residual, float("inf")), hypothesis.residual
            ).cpu()
            better = residual < best_residual
            best_residual = torch.where(better, residual, best_residual)
            best_target = torch.where(better, torch.full_like(best_target, target), best_target)

    image_recall = candidate_recall_audit(order.cpu(), best_target)
    random_order = torch.stack([
        torch.randperm(int(anchorsT.numel()), generator=generator)[: int(args.candidates)]
        for _ in range(int(anchors0.numel()))
    ])
    random_recall = candidate_recall_audit(random_order, best_target)
    results["R5_recall_audit"] = {
        "image_supported": image_recall, "random": random_recall,
        "target_domain": int(poolT.numel()), "targets_audited": int(anchorsT.numel()),
        "note": ("recall of the EXHAUSTIVE physical best candidate among the audited target "
                 "patches; this is the §52.1 claim measured, not asserted"),
    }
    gain = image_recall["recall_at_8"] - random_recall["recall_at_8"]
    results["verdict"] = {
        "image_support_improves_recall": bool(gain > 0.0),
        "recall_at_8_gain": float(gain),
        "result": ("IMAGE SUPPORT IMPROVES CANDIDATE RECALL" if gain > 0.0
                   else "IMAGE SUPPORT DOES NOT IMPROVE CANDIDATE RECALL"),
    }

    if bool(args.save_tensors):
        tensors = out / "tensors"; tensors.mkdir(parents=True, exist_ok=True)
        torch.save(lifted.support.cpu(), tensors / "patch_image_support.pt")
        torch.save(order.cpu(), tensors / "image_supported_order.pt")
        torch.save(best_target, tensors / "exhaustive_best_target.pt")
        torch.save(best_residual, tensors / "exhaustive_best_residual.pt")

    _write_json(out / "rasterizer_coupled_probe.json", json_finite(_json_safe(results)))
    print(f"\n=== {results['verdict']['result']} ===")
    print(f"  recall@1  image {image_recall['recall_at_1']:.3f}  random {random_recall['recall_at_1']:.3f}")
    print(f"  recall@8  image {image_recall['recall_at_8']:.3f}  random {random_recall['recall_at_8']:.3f}")
    print(f"  recall@16 image {image_recall['recall_at_16']:.3f}  random {random_recall['recall_at_16']:.3f}")
    print(f"\n[probe] wrote {out / 'rasterizer_coupled_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
