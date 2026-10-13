#!/usr/bin/env python
"""Section 4.4 — is the union of the two endpoint models one physical scene?

CODEX_COMPLETE_MISSING_UNIFIED_BIDI_PHASE1_AFTER_9B78EFA.md section 4.4.

The two endpoint models are independent trainings of independent COLMAP
reconstructions of the SAME object. Their union is a candidate state space;
whether it is a nonredundant scene is a question about the renderer, and this
tool asks the renderer.

Each source is placed at the SAME real frame using its OWN free Phase-1
trajectory -- `bidirectional/forward/phase1_centers` for the initial model,
`bidirectional/backward/phase1_centers` for the final one, both indexed by
ORIGINAL frame, so the two arrays are simultaneous. The unchanged compositing
CUDA renderer then renders initial only, final only, and the concatenation,
and the audit counts the pixels that receive coverage from BOTH.

That count is the whole answer. Where two sources cover one pixel, alpha
compositing gives 1 - (1-a1)(1-a2): a surface represented twice reads as more
opaque and its normalised depth D = Z/(A+eps) shifts toward the duplicate.
That is a consequence of the compositing equations, not a proposed model, and
nothing here rescales opacity, deduplicates by distance, introduces ownership
variables or consults a ground-truth mask. If the overlap is nonzero the
result is UNRESOLVED_SCENE_REPRESENTATION and only the MERGED run stops.

Rotations are the models' own and are held constant: Phase 1 persisted centers
only. The gate measures coverage overlap, which the centers determine; the
manifest says so rather than implying the orientations were transported.

    python tools/run_unified_representation_gate.py \
      --source <dataset root> --out-root <scene results root> --frames 0 30 59
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="washer_103776")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out", default=None,
                    help="default <out-root>/unified_phase1_tests/representation_gate")
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 30, 59],
                    help="original frames to audit")
    ap.add_argument("--cameras", type=int, default=2)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--overlap-alpha-tol", type=float, default=1e-3)
    ap.add_argument("--depth-sigma", type=float, default=0.02,
                    help="the production depth model's constant a. Used to "
                         "ask whether two sources covering one pixel are at "
                         "the SAME depth, which is the distinction a pixel "
                         "intersection cannot make")
    ap.add_argument("--k-sigma", type=float, default=3.0)
    ap.add_argument("--with-observations", action="store_true",
                    help="also report the RGB and depth residual of A alone, "
                         "B alone and A+B against the real calibrated frame. "
                         "Diagnostic: a lower union residual is not evidence "
                         "of a unique representation")
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out = Path(args.out or out_root / "unified_phase1_tests" / "representation_gate")
    from bidirectional_iekf.unified_coherence import reject_stale_support
    reject_stale_support(out, what="the output directory")

    from bidirectional_iekf.representation_audit import audit_pair, write_audit
    from estimator.state import GaussianMotionState
    from tools.run_iekf_scene import load_scene, state_from_gaussians
    from tools.run_unified_phase1_single_source import sha256_file

    print(f"=== representation gate: {args.scene} ===")
    sources, states, cameras = {}, {}, None
    for endpoint, direction in (("initial", "forward"), ("final", "backward")):
        model = out_root / "static_bidi" / endpoint
        centers_dir = out_root / "bidirectional" / direction / "phase1_centers"
        ply = model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
        for path in (model, centers_dir, ply):
            if not path.exists():
                raise SystemExit(f"missing: {path}")
        p1 = json.loads((centers_dir / "trajectory_manifest.json").read_text())
        if p1.get("canonical_endpoint") != endpoint:
            raise SystemExit(f"{centers_dir} is the {p1.get('canonical_endpoint')!r} "
                             f"endpoint, not {endpoint!r}")
        if p1.get("trajectory_order") != "original_chronological":
            raise SystemExit(f"{centers_dir} is not in original chronological order")
        _d, _p, gaussians, scene = load_scene(str(model), str(args.source))
        state = state_from_gaussians(gaussians)
        if cameras is None:
            from estimator.renderer_bridge import CameraSpec  # noqa: F401
            from tools.run_iekf_scene import camera_specs
            cams_all = scene.getTrainCameras()
            all_cameras = cams_all
            ncam = min(args.cameras, len(cams_all))
            cameras = camera_specs(cams_all, ncam)
            camera_names = [str(getattr(c, "image_name", f"cam_{i:03d}")).split("/")[0]
                            for i, c in enumerate(cams_all[:ncam])]
        states[endpoint] = state
        sources[endpoint] = {
            "model_path": str(model), "ply_path": str(ply),
            "ply_sha256": sha256_file(ply),
            "num_gaussians": int(state.mu.shape[0]),
            "phase1_centers": str(centers_dir),
            "phase1_direction": p1["direction"],
            "phase1_frames": p1["num_frames"],
        }
        print(f"  {endpoint:<8} {state.mu.shape[0]} Gaussians from {model.name}, "
              f"{p1['direction']} Phase 1")

    def transported(endpoint, direction, frame):
        path = (out_root / "bidirectional" / direction / "phase1_centers"
                / f"centers_frame_{frame:04d}.npy")
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        base = states[endpoint]
        mu = torch.from_numpy(np.load(path)).to(base.mu.device, base.mu.dtype)
        if mu.shape != base.mu.shape:
            raise SystemExit(f"{path} is {tuple(mu.shape)} but the model is "
                             f"{tuple(base.mu.shape)}")
        import dataclasses
        return dataclasses.replace(base, mu=mu.contiguous())

    def real_frame(frame):
        """The dataset's own RGB-D for this original frame, per camera."""
        import argparse as _argparse
        from estimator.sparse_iekf import IEKFConfig
        from tools.run_iekf_scene import dataset_rgbd_frames
        ns = _argparse.Namespace(
            sequence_start=int(frame), sequence_length=1, all_frames=False,
            observation_source="dataset_rgbd", cameras=len(cameras),
            all_cameras=False, window_length=0, dt=1.0 / 30.0,
            frames_on_cpu=False, frames_on_gpu=True, max_gaussians=0, seed=0)
        raw, _prov = dataset_rgbd_frames(all_cameras[:len(cameras)],
                                         IEKFConfig(), ns, None,
                                         Path(args.source), direction="forward")
        return raw[0][0]

    entries = []
    for frame in args.frames:
        a = transported("initial", "forward", frame)
        b = transported("final", "backward", frame)
        observed = real_frame(frame) if args.with_observations else None
        entry = audit_pair(a, b, cameras, label=f"original_frame_{frame:04d}",
                           overlap_alpha_tol=args.overlap_alpha_tol,
                           depth_sigma=args.depth_sigma, k_sigma=args.k_sigma,
                           observations=observed)
        entry["original_frame"] = int(frame)
        entry["cameras"] = camera_names
        covered = sum(c["covered_pixels_a"] + c["covered_pixels_b"]
                      for c in entry["per_camera"])
        entry["overlap_fraction_of_covered"] = (
            2.0 * entry["overlap_pixels_total"] / max(covered, 1))
        entries.append(entry)
        print(f"  frame {frame:>3}: {entry['status']} / "
              f"{entry['overlap_character']}  "
              f"{entry['overlap_pixels_total']} doubly covered "
              f"({100 * entry['overlap_fraction_of_covered']:.1f}% of coverage); "
              f"of those {100 * entry['depth_agreeing_fraction']:.1f}% agree "
              f"in depth to {args.k_sigma:g}x{args.depth_sigma:g} m, "
              f"{entry['depth_separated_pixels']} are separated layers; "
              f"max alpha gain {entry['max_alpha_gain_over_single_source']:.4f}")
        if args.with_observations:
            res = entry["per_camera"][0].get("residual_vs_observation", {})
            if res:
                print("            RGB rmse / depth rmse vs the real frame: "
                      + ", ".join(
                          f"{k} {res[k].get('rgb_rmse', float('nan')):.4f} / "
                          f"{res[k].get('depth_rmse_m', float('nan')):.4f} m"
                          for k in ("initial_only", "final_only", "union")))

        if args.save_images:
            from bidirectional_iekf.representation_audit import (
                _render, concatenate_states)
            img = out / "images" / f"frame_{frame:04d}"
            img.mkdir(parents=True, exist_ok=True)
            renders = {"initial": _render(a, cameras), "final": _render(b, cameras),
                       "union": _render(concatenate_states(a, b), cameras)}
            for name, views in renders.items():
                for ci, v in enumerate(views):
                    np.savez_compressed(
                        img / f"{name}_{camera_names[ci]}.npz",
                        rgb=v["rgb"].detach().cpu().numpy(),
                        depth=v["depth"].detach().cpu().numpy(),
                        alpha=v["alpha"].detach().cpu().numpy(),
                        n_contrib=v["n_contrib"].detach().cpu().numpy())
            for ci in range(len(cameras)):
                va, vb = renders["initial"][ci], renders["final"][ci]
                both = ((va["alpha"] > args.overlap_alpha_tol)
                        & (vb["alpha"] > args.overlap_alpha_tol))
                np.save(img / f"overlap_mask_{camera_names[ci]}.npy",
                        both.detach().cpu().numpy())

    payload = write_audit(out, entries, sources=sources)
    (out / "gate_config.json").write_text(json.dumps({
        "scene": args.scene, "source": str(args.source),
        "frames": list(args.frames), "cameras": camera_names,
        "overlap_alpha_tol": args.overlap_alpha_tol,
        "approximation": "CENTER_ONLY_FIXED_ORIENTATION",
        "orientation_treatment": (
            "each model's own rotations, held constant: Phase 1 persisted "
            "centers only. A splat's projected footprint depends on its "
            "covariance orientation as well as its centre, so this is an "
            "approximation and is labelled one. A full-state export "
            "(bidirectional_iekf.free_state) removes it."),
        "depth_sigma_m": args.depth_sigma, "k_sigma": args.k_sigma,
        "what_projected_overlap_does_not_prove": (
            "two valid depth-separated parts, one behind the other, cover "
            "the same pixels; and no overlap in a sampled set of cameras "
            "does not prove global nonredundancy. The depth-agreement "
            "fraction is what distinguishes a duplicate from a layer."),
        "placement": ("each source at the SAME original frame through its own "
                      "free Phase-1 trajectory; the two arrays are "
                      "simultaneous by construction"),
    }, indent=1))
    print(f"\n  OVERALL: {payload['status']}")
    print(f"  {out / 'representation_audit.json'}")
    return 0 if payload["status"] == "PASSED" else 4


if __name__ == "__main__":
    raise SystemExit(main())
