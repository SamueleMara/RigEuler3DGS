#!/usr/bin/env python
"""Run the motion-coherence static/dynamic filter on one scene.

CODEX_IMPLEMENT_MOTION_COHERENCE_STATIC_DYNAMIC_FILTER.md sections 18-20,
38-40, 44-45.

    Phase 1 free trajectories -> coherence -> static/dynamic -> frozen support

The prediction is written to disk BEFORE any ground truth is read (section 21),
and the order is enforced by `require_prediction_on_disk`, not by convention.
The Otsu baseline is evaluated too, but only as a row in the comparison table:
the two selectors are never unioned, intersected, averaged or voted, and
neither seeds the other (section 20).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _write_phase1_direction_debug(out, frame_ids, ctx, scene, num_positions,
                                  endpoint_evidence=None):
    """Debug for the backward Phase-1 phase.

    The failure this catches is silent: a trajectory saved under the wrong
    frame numbers is well formed, loads cleanly and runs the sequence
    backwards. So the mapping actually used is written out beside the centers
    and checked, rather than trusted.
    """
    from debug.sequence_direction_debug import write_phase_debug

    dbg = out / "phase1_direction"
    report = write_phase_debug(dbg, frame_ids, scene=scene)
    payload = {
        **ctx.as_dict(),
        "num_phase1_frames": int(num_positions),
        "centers_named_by": "original_frame_index",
        "canonical_geometry": (
            "final endpoint model; NOT reordered -- the backward solve lives "
            "in the final gauge and only the time axis is restored"
            if ctx.is_backward else "initial endpoint model"),
        "frame_order_checks": report["checks"],
        "endpoint_provenance": endpoint_evidence or {},
        "trajectory_objects": {
            "canonical_xyz": "endpoint model geometry, never a trajectory frame",
            "positions_processing_order": "as the filter produced them",
            "positions_original_order": "chronological, what coherence reads",
        },
    }
    import json as _jsonlib
    (dbg / "phase1_direction.json").write_text(_jsonlib.dumps(payload, indent=1))
    if not report["checks"]["passes"]:
        print(f"  !! frame-order checks FAILED: {report['checks']}")
    else:
        print(f"  frame order verified ({ctx.direction}, "
              f"{num_positions} frames) -> {report['html']}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--cameras", type=int, default=8)
    ap.add_argument("--max-gaussians", type=int, default=0)
    ap.add_argument("--bootstrap-iters", type=int, default=6)
    # 0 means "size the graph by COUNT" -- see the comment at the call site.
    # A positive value pins Gaussians-per-node instead, reproducing the older
    # behaviour, and takes precedence.
    ap.add_argument("--supernode-size", type=int, default=0)
    # SEPARATE from the coherence graph below. This one sizes the GROUPED
    # STATE of the Phase-1 IEKF, which is a different object with a different
    # job: it decides how many pose blocks the PCG solve carries. The two
    # shared one flag until the count-based coherence default landed, at
    # which point --supernode-size dropped to 0 and build_supernodes' own
    # `n // max(target_size, 1)` turned that into ONE GROUP PER GAUSSIAN --
    # 24,582 blocks instead of 384 on washer, and Phase 1 slowed by about two
    # orders of magnitude while still producing correct-looking output.
    ap.add_argument("--iekf-supernode-size", type=int, default=64,
                    help="Gaussians per group in the Phase-1 IEKF state. Not "
                         "the coherence graph; see --supernode-count.")
    ap.add_argument("--supernode-count", type=int, default=192,
                    help="how many supernodes the coherence graph has, "
                         "independent of how many Gaussians the model holds. "
                         "ONE global value, never tuned per scene.")
    # 192 was chosen by sweeping {128, 192, 256, 384} on five scenes and
    # taking the best mean AND best worst case (mean F1 0.733 / worst 0.404,
    # against 0.617 / 0.381 for the old fixed size of 64). 128 has the best
    # single score, on USB, but collapses blade to F1 0.395 and storage to
    # 0.191 -- too few nodes and the graph cannot resolve a boundary, so the
    # selector takes the whole scene.
    #
    # Choosing it on GT F1 is tuning on GT, so it was then validated on the
    # four scenes NOT in the sweep: +0.033 mean F1, better on three of four
    # (scissor +0.098, oven +0.026, foldchair +0.013, fridge -0.006).
    ap.add_argument("--coherence-knn", type=int, default=12,
                    help="section 4: ONE global default for every dataset; "
                         "do not tune it per scene")
    ap.add_argument("--phase1-centers", default=None,
                    help="reuse an existing phase1/centers directory instead "
                         "of re-running Phase 1")
    ap.add_argument("--frames-on-cpu", action="store_true")
    ap.add_argument("--extractor", choices=("v2", "ncut"), default="v2",
                    help="v2 = section 11-19 coherent-component extraction "
                         "(default); ncut = the superseded two-way normalized "
                         "cut, kept only for comparison")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    # Bidirectional: additive, forward stays the default and is unchanged.
    ap.add_argument("--direction", choices=("forward", "backward"),
                    default="forward",
                    help="backward walks the sequence last-to-first from the "
                         "FINAL endpoint model; --model must be that model")
    ap.add_argument("--canonical-endpoint", choices=("initial", "final"),
                    default=None,
                    help="which endpoint --model is; defaults to the one the "
                         "direction implies, and a mismatch is refused")
    ap.add_argument("--out", default=None)
    ap.add_argument("--full-state-out", default=None,
                    help="also export the FULL 12-DOF free state (mu, R as a "
                         "unit quaternion, v, omega) at every original frame, "
                         "plus the prior information translation block the "
                         "filter's own observer provides. Output only: it "
                         "attaches bidirectional_iekf.free_state."
                         "FullStateRecorder as the existing observer hook and "
                         "changes nothing in the filter. phase1_centers/ "
                         "stores centers alone, which forces every downstream "
                         "stage to invent R and the rates.")
    args = ap.parse_args()

    model, source = args.model, args.source
    if model is None or source is None:
        from tools.run_all_datasets import discover
        for rec in discover(REPO):
            if rec["scene"] == args.scene:
                model = model or str(rec["model"])
                source = source or str(rec["source"])
                break
    out = Path(args.out or f"output/motion_coherence_{args.scene}")
    (out / "numerical").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"=== MOTION COHERENCE: {args.scene} ===")

    from estimator.motion_coherence_support import (
        UNRESOLVED, propagate_to_gaussians, run_motion_coherence)
    from estimator.supernodes import build_supernodes

    # ---- section 48 step 2: Phase-1 free trajectories --------------------
    # This branch is renderer-independent by construction: it reads only the
    # Phase-1 free trajectories. When those are already on disk nothing here
    # needs CUDA, the 3DGS model or the cameras -- frame 0 of the trajectory
    # IS the canonical state. That is not a convenience; a filter whose claim
    # is about trajectory geometry should be runnable, and testable, without
    # the renderer it is meant to be independent of.
    positions, mu0, ncam = None, None, args.cameras
    if args.phase1_centers:
        d = Path(args.phase1_centers)
        files = sorted(d.glob("centers_frame_*.npy"))[: args.frames or None]
        if not files:
            raise SystemExit(f"no centers_frame_*.npy under {d}")
        positions = [torch.from_numpy(np.load(f)).double() for f in files]
        # Section 5. The centers are named by ORIGINAL frame, so for a
        # backward run centers_frame_0000 is the FINAL model's Gaussians
        # transported to original frame 0 -- it is a pose of the canonical
        # geometry, not the canonical geometry. Taking mu0 from it would put
        # the whole solve in the wrong gauge while every array kept its
        # shape. The canonical set is stored explicitly and read back.
        manifest_path = d / "trajectory_manifest.json"
        canonical_path = d / "canonical_xyz.npy"
        manifest = (json.loads(manifest_path.read_text())
                    if manifest_path.is_file() else {})
        reload_direction = str(manifest.get("direction", "forward"))
        if canonical_path.is_file():
            mu0 = torch.from_numpy(np.load(canonical_path)).double()
        elif reload_direction == "backward":
            raise SystemExit(
                f"{d} holds a BACKWARD trajectory but no canonical_xyz.npy. "
                "Its centers are named by original frame, so frame 0 is the "
                "final model transported to the start of the sequence, not "
                "the canonical geometry. Re-run Phase 1 to write the "
                "canonical set rather than guessing it.")
        else:
            # Legacy forward trajectories predate the manifest; there frame 0
            # IS the canonical frame, so this stays compatible.
            mu0 = positions[0].clone()
        if manifest.get("trajectory_order") not in (None, "original_chronological"):
            raise SystemExit(
                f"{d} records trajectory_order="
                f"{manifest.get('trajectory_order')!r}; this loader expects "
                "original chronological order")
        print(f"  reusing {len(positions)} Phase-1 center frames from {d}"
              f"  ({mu0.shape[0]} Gaussians, {reload_direction}, "
              f"no renderer needed)")
    if positions is None:
        from estimator.sparse_iekf import IEKFConfig
        from tools.run_iekf_scene import (dataset_rgbd_frames, load_scene,
                                          state_from_gaussians)
        dataset, pipe, gaussians, sc = load_scene(model, source, camera_device="cpu")
        cams_all = sc.getTrainCameras()
        ncam = min(args.cameras, len(cams_all))
        truth = state_from_gaussians(gaussians)
        if args.max_gaussians and truth.mu.shape[0] > args.max_gaussians:
            import dataclasses
            g = torch.Generator().manual_seed(args.seed)
            keep = torch.randperm(
                truth.mu.shape[0],
                generator=g)[:args.max_gaussians].sort().values
            truth = dataclasses.replace(
                truth, mu=truth.mu[keep], R=truth.R[keep], v=truth.v[keep],
                omega=truth.omega[keep], colors=truth.colors[keep],
                opacity=truth.opacity[keep], scales=truth.scales[keep])
        mu0 = truth.mu.double()
        cfg = IEKFConfig()
        print(f"  {mu0.shape[0]} Gaussians, {ncam} cameras")
        from tools.kpart_pipeline import bootstrap_positions

        ns = argparse.Namespace(
            sequence_start=0, sequence_length=args.frames, all_frames=False,
            observation_source="dataset_rgbd", cameras=ncam,
            bootstrap_iters=args.bootstrap_iters,
            supernode_size=args.iekf_supernode_size, seed=args.seed,
            frames_on_cpu=bool(args.frames_on_cpu),
            frames_on_gpu=not args.frames_on_cpu, depth_sigma=None,
            joint_direct_rgbd=False, joint_direct_gaussian=False,
            window_length=0, all_cameras=False, dt=1.0 / 30.0,
            max_gaussians=args.max_gaussians, no_model_error=False,
            model_error_beta=1.0, model_error_blur=1)
        frames_raw, frame_ids = dataset_rgbd_frames(
            cams_all[:ncam], cfg, ns, truth, Path(source),
            direction=args.direction)
        from utils.sequence_direction import (context_from_frame_ids,
                                              verify_endpoint_provenance)
        ctx = context_from_frame_ids(frame_ids)
        # Section 6: enforced in the runtime, not only documented. The check
        # reads the model's cfg_args and the COLMAP images list rather than
        # the directory name, because ".../final" is a string and proves
        # nothing about which frame the cameras were registered at.
        endpoint_evidence = verify_endpoint_provenance(
            args.direction, model, total_frames=len(frames_raw), strict=True)
        print(f"  endpoint check: {endpoint_evidence['observed_endpoint'] or 'unknown'}"
              f" (frame {endpoint_evidence['endpoint_original_frame']}),"
              f" expected {endpoint_evidence['expected_endpoint']}")
        cameras = frames_raw[0][1]
        window = [(ob, fid) for (ob, _s, _d), fid
                  in zip(frames_raw, frame_ids["frames"])]
        print(f"  Phase 1 over {len(window)} frames ...")
        recorder = None
        if args.full_state_out:
            from bidirectional_iekf.free_state import FullStateRecorder
            recorder = FullStateRecorder(expected_frames=len(window))
            print(f"  recording the full 12-DOF state to {args.full_state_out}")
        positions, _info = bootstrap_positions(truth, cameras, window, cfg,
                                               None, ns, verbose=True,
                                               observer=recorder)
        # Step 5: persist in ORIGINAL CHRONOLOGICAL indexing.
        #
        # `positions` comes back in PROCESSING order, so for a backward run
        # positions[0] is original frame T-1. Saving that as
        # centers_frame_0000 would hand every downstream stage a trajectory
        # that runs the wrong way while looking perfectly well formed. The
        # file name carries the ORIGINAL frame number in both directions, so
        # a forward and a backward run can be compared file by file.
        #
        # The canonical geometry is NOT reordered: for backward it is the
        # final endpoint's Gaussians, which is the gauge the whole backward
        # solve lives in. Only the time axis is restored.
        cdir = out / "phase1_centers"
        cdir.mkdir(parents=True, exist_ok=True)
        order = ctx.order()
        if len(order) != len(positions):
            raise SystemExit(
                f"{len(positions)} Phase-1 frames but the direction context "
                f"describes {len(order)}; they must correspond one to one")
        for processing_index, p in enumerate(positions):
            original = order[processing_index]
            np.save(cdir / f"centers_frame_{original:04d}.npy",
                    torch.as_tensor(p).double().cpu().numpy())

        # Section 4. Renaming the files did NOT reorder the trajectory that
        # coherence consumes. Three objects, never interchangeable:
        #
        #   canonical_xyz             the endpoint model's own geometry
        #   positions_processing      as the filter produced them
        #   positions_original        chronological, what coherence reads
        #
        # For a backward run the canonical set is the FINAL model even though
        # the chronological trajectory starts at original frame 0.
        positions_processing = list(positions)
        positions = list(ctx.to_original(positions_processing))
        np.save(cdir / "canonical_xyz.npy", mu0.double().cpu().numpy())
        np.save(cdir / "original_gaussian_indices.npy",
                np.arange(int(mu0.shape[0]), dtype=np.int64))
        (cdir / "trajectory_manifest.json").write_text(json.dumps({
            "direction": ctx.direction,
            "canonical_endpoint": ctx.endpoint,
            "canonical_model_path": str(model),
            "canonical_original_frame": (order[0] if ctx.is_backward else order[0]),
            "trajectory_order": "original_chronological",
            "processing_order_original_indices": order,
            "num_frames": len(positions),
            "num_gaussians": int(mu0.shape[0]),
            "note": ("centers_frame_NNNN.npy is indexed by ORIGINAL frame. "
                     "canonical_xyz.npy is the endpoint model's geometry and "
                     "is NOT centers_frame_0000 for a backward run."),
        }, indent=1))
        if recorder is not None:
            import hashlib
            ply = (Path(model) / "point_cloud"
                   / f"iteration_{args.iteration if hasattr(args, 'iteration') else 30000}"
                   / "point_cloud.ply")
            digest = ""
            if ply.is_file():
                h = hashlib.sha256()
                with open(ply, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        h.update(chunk)
                digest = h.hexdigest()
            fs = recorder.write(
                args.full_state_out, direction=ctx.direction,
                canonical_endpoint=ctx.endpoint,
                processing_order_original_indices=order,
                dt_seconds=float(ns.dt), dt_is_measured=False,
                model_path=str(model), source_path=str(source),
                ply_sha256=digest, scene=args.scene,
                extra={"max_gaussians": int(args.max_gaussians),
                       "seed": int(args.seed),
                       "num_cameras": int(ncam),
                       "subset_applied": bool(args.max_gaussians
                                              and truth.mu.shape[0] < mu0.shape[0])})
            print(f"  full state: {fs['num_frames']} frames x "
                  f"{fs['num_gaussians']} Gaussians, rates "
                  f"{fs['rate_convention']}")
        _write_phase1_direction_debug(out, frame_ids, ctx, args.scene,
                                      len(positions), endpoint_evidence)

    # ---- sections 3-17: the coherence branch -----------------------------
    #
    # THE GRAPH IS SIZED BY NODE COUNT, NOT BY GAUSSIANS PER NODE.
    #
    # `target_size` makes the node count scale with the model: training the
    # same object for 30k instead of 14k iterations roughly doubles the
    # Gaussians, and with a fixed 64 per node the graph doubled too -- USB
    # 177 -> 323 nodes, laptop 438 -> 722. The kNN degree `k` stayed at 12, so
    # the same physical part came to span twice as many nodes at the same
    # connectivity. The extracted component did NOT keep up (USB 93 -> 89
    # nodes, laptop 120 -> 90), so it covered half the body and recall halved:
    # USB 0.874 -> 0.446, blade 0.897 -> 0.555, storage 0.532 -> 0.283.
    # Precision was unaffected, which is what rules out the selector being
    # permissive -- the body fragmented and single linkage could no longer
    # span it.
    #
    # Fixing the COUNT makes the graph the same shape whatever the Gaussian
    # density, so `k` means the same thing and a result is comparable across
    # models. `--supernode-size` still works and takes precedence when given,
    # for reproducing an earlier run.
    mu0 = mu0.double().cpu()
    if args.supernode_size:
        part = build_supernodes(mu0.float(), target_size=args.supernode_size,
                                seed=args.seed)
        how = f"target size {args.supernode_size}"
    else:
        want = min(int(args.supernode_count), int(mu0.shape[0]))
        part = build_supernodes(mu0.float(), num_supernodes=want,
                                seed=args.seed)
        how = (f"target count {args.supernode_count}, "
               f"{mu0.shape[0] / max(want, 1):.0f} Gaussians per node")
    sid = part.supernode_id.long().cpu()
    print(f"  {int(sid.max()) + 1} supernodes ({how})")
    res = run_motion_coherence(positions, sid, knn=args.coherence_knn,
                               verbose=True)

    # ---- sections 11-19: replace the two-way Ncut ------------------------
    if args.extractor == "v2":
        from estimator.coherence_extraction import (
            MOVING_RIGID, NO_MOVING, extract_moving_component)
        from estimator.motion_coherence_support import supernode_trajectories
        sn = supernode_trajectories(positions, sid)
        traj_nodes = sn["trajectories"][:, res["node_ids"], :]
        ex = extract_moving_component(
            traj_nodes, res["edge_i"], res["edge_j"], res["C"],
            res["num_supernodes"], folds=args.cv_folds, verbose=True)
        res["extraction"] = ex
        res["status"] = ex["status"]
        res["dynamic_nodes"] = res["node_ids"][ex["moving_nodes"]] \
            if ex["moving_nodes"].numel() else torch.zeros(0, dtype=torch.long)
        keep = torch.tensor([l != MOVING_RIGID for l in ex["labels"]])
        res["static_nodes"] = res["node_ids"][keep]
        res["node_class"] = ex["labels"]
        print(f"    extraction: {ex['status']}  {ex['counts']}")
        if ex["status"] != NO_MOVING:
            sel = next(c for c in ex["candidates"]
                       if c["candidate_id"] == ex["selection"]["selected"])
            r, rw = sel["rigidity"], sel["rigidity_world"]
            print(f"    selected candidate {sel['candidate_id']}: "
                  f"{sel['num_members']} nodes, persistence "
                  f"{sel['persistence']:.4f}, eta_cv(rel) {r['eta_cv']:.4f}, "
                  f"dRSS_world {rw['delta_rss_cv']:.4e}, "
                  f"dBIC(rel) {r['delta_bic']:.3e}")
        else:
            print(f"    {ex['selection']['reason']}")
    status = res["status"]
    print(f"  status: {status}")
    for k, c in enumerate(res["components"]):
        tag = ""
        if status != UNRESOLVED:
            tag = (" DYNAMIC" if k == res["identification"]["dynamic_component"]
                   else " STATIC")
        print(f"    component {k}{tag}: {c['num_nodes']} nodes  "
              f"E_rigid {c['E_rigid']:.4e}  E_identity {c['E_identity']:.4e}  "
              f"gain {c['gain']:.4e}")

    dyn_g = propagate_to_gaussians(res["dynamic_nodes"], sid)
    sta_g = propagate_to_gaussians(res["static_nodes"], sid)

    # ---- section 40: the prediction, written BEFORE any GT ---------------
    pred_path = out / "predicted_support.json"
    pred_path.write_text(json.dumps({
        "method": res["method"],
        "status": status,
        "num_supernodes": res["num_supernodes"],
        "num_gaussians": int(mu0.shape[0]),
        "static_supernodes": res["static_nodes"].tolist(),
        "dynamic_supernodes": res["dynamic_nodes"].tolist(),
        "static_gaussians": sta_g.tolist(),
        "dynamic_gaussians": dyn_g.tolist(),
        "predicted_dynamic_fraction": float(dyn_g.numel()) / max(int(mu0.shape[0]), 1),
        "graph_config": {**res["graph"], "coherence_knn": args.coherence_knn,
                         "supernode_target_size": args.supernode_size,
                         "supernode_count": args.supernode_count,
                         "num_supernodes_built": int(sid.max()) + 1},
        "component_stats": [{k: v for k, v in c.items() if k != "per_frame"}
                            for c in res["components"]],
        "identification": res["identification"],
        "cut": res["cut"],
        "note": ("sections 0, 40: inferred from trajectory metric coherence "
                 "ALONE. No Otsu, no displacement threshold, no RGB residual, "
                 "no semantics, no joint type, no GT."),
    }, indent=1, default=str))
    (out / "config.json").write_text(json.dumps({
        "scene": args.scene, "model": model, "source": source,
        "frames": len(positions), "cameras": ncam,
        "gaussians": int(mu0.shape[0]),
        "coherence_knn": args.coherence_knn,
        "supernode_target_size": args.supernode_size,
        "seed": args.seed,
        "weight_convention": "uniform w_i = 1",
    }, indent=1))

    # ---- section 25/26: per-edge and per-node debug -----------------------
    ei, ej, C, J = res["edge_i"], res["edge_j"], res["C"], res["J"]
    lab = res["labels"]
    _csv(out / "numerical" / "edges.csv",
         ["edge_id", "node_i", "node_j", "canonical_distance", "C_rigid",
          "J_jitter", "same_predicted_component"],
         [[e, int(ei[e]), int(ej[e]), float(res["canonical_distance"][e]),
           float(C[e]), float(J[e]), int(lab[ei[e]] == lab[ej[e]])]
          for e in range(int(ei.numel()))])
    amb = res["ambiguity"]
    _csv(out / "numerical" / "nodes.csv",
         ["node", "supernode", "label", "predicted_class", "C_in", "C_out",
          "margin", "ambiguous"],
         [[i, int(res["node_ids"][i]), int(lab[i]),
           ("DYNAMIC" if status != UNRESOLVED
            and int(lab[i]) == res["identification"]["dynamic_component"]
            else "STATIC" if status != UNRESOLVED else "UNRESOLVED"),
           float(amb["C_in"][i]), float(amb["C_out"][i]),
           float(amb["margin"][i]), int(bool(amb["ambiguous"][i]))]
          for i in range(res["num_supernodes"])])
    (out / "numerical" / "components.json").write_text(json.dumps(
        {"status": status, "identification": res["identification"],
         "components": res["components"], "cut": res["cut"],
         "length_scale": res["length_scale"]}, indent=1, default=str))
    # section 10: the whole temporal signal
    sub = torch.argsort(C)[:: max(1, int(ei.numel()) // 4000)]
    np.savez_compressed(
        out / "numerical" / "edge_strain_timeseries.npz",
        edge_i=ei.numpy(), edge_j=ej.numpy(),
        C_edge=C.numpy(), J_edge=J.numpy(),
        subset=sub.numpy(),
        z=res["z"][:, sub].numpy().astype(np.float32),
        epsilon=res["epsilon"][:, sub].numpy().astype(np.float32))

    # ---- sections 27-33: debug pages -------------------------------------
    from debug.motion_coherence_debug import (
        write_differential_strain_heatmap, write_finite_strain_heatmap,
        write_representative_edges, write_support_3d)
    dbg = out / "debug"
    write_finite_strain_heatmap(dbg / "finite_strain_heatmap.html", res)
    write_differential_strain_heatmap(dbg / "differential_strain_heatmap.html", res)
    write_representative_edges(dbg / "representative_edges.html", res)
    if res.get("extraction"):
        from debug.motion_coherence_debug import write_candidate_tree
        write_candidate_tree(dbg / "coherence_component_tree.html",
                             res["extraction"],
                             f"{args.scene} — coherence candidate tree")

    # ======================================================================
    # EVALUATION ONLY from here. GT has not been touched above.
    # ======================================================================
    from estimator.motion_coherence_eval import (
        classification_metrics, compare_selectors, gt_edge_coherence,
        load_static_dynamic_gt, require_prediction_on_disk, write_json)

    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    require_prediction_on_disk(pred_path)          # section 21 order guard
    from estimator.optimized_gt import (NOT_FOUND, compare_with_sparse_nn,
                                        load_optimized_gt, supernode_gt)
    # Sections 2-6: the optimized labelled 3DGS model is the PRIMARY GT.
    gt = load_optimized_gt(model, mu0)
    sparse = load_static_dynamic_gt(source, mu0)
    if gt.get("status") == NOT_FOUND:
        # Section 5: do not silently call sparse-NN labels ground truth.
        write_json(out / "evaluation" / "optimized_3dgs_gt.json", gt)
        print(f"  {NOT_FOUND}: {gt.get('reason')}")
        gt = {"available": False, "reason": NOT_FOUND}
    else:
        np.save(out / "evaluation" / "optimized_3dgs_gt_labels.npy",
                gt["labels"].numpy())
        write_json(out / "evaluation" / "optimized_3dgs_gt.json",
                   {k: v for k, v in gt.items() if k != "labels"})
        if sparse.get("available"):
            write_json(out / "evaluation" / "sparse_nn_vs_optimized.json",
                       compare_with_sparse_nn(gt["labels"], sparse["labels"]))
        write_json(out / "numerical" / "supernode_gt.json",
                   supernode_gt(gt["labels"], sid))
    ev = out / "evaluation"
    rows = []
    gt_node = None

    # Section 24: the Otsu baseline. It reads the SAME Phase-1 trajectories the
    # coherence extractor read and nothing else -- it is not seeded by, and
    # does not seed, the coherence support.
    from estimator.moving_support import select_moving_support
    base = select_moving_support(positions)
    pred_base = torch.zeros(mu0.shape[0], dtype=torch.bool)
    pred_base[base["members"]] = True
    pred_dyn = torch.zeros(mu0.shape[0], dtype=torch.bool)
    pred_dyn[dyn_g] = True
    m_coh = m_base = None

    if gt.get("available"):
        gtl = gt["labels"]
        write_json(out / "numerical" / "gt_mapping.json",
                   {k: v for k, v in gt.items() if k != "labels"})
        print(f"  GT: {gt['mapping_method']} from {gt['iteration']}, "
              f"{100 * gt['gt_dynamic_fraction']:.1f}% dynamic, "
              f"alignment {gt['alignment'].get('max_alignment_error', 0):.1e}")
        m_coh = classification_metrics(pred_dyn, gtl)
        m_base = classification_metrics(pred_base, gtl)
        for name, m in (("current", m_base), ("coherence", m_coh)):
            rows.append({"scene": args.scene, "selector": name, **m})
            print(f"  [{name:>9}] dyn P {m['dynamic_precision']:.3f} "
                  f"R {m['dynamic_recall']:.3f} F1 {m['dynamic_f1']:.3f} "
                  f"IoU {m['dynamic_iou']:.3f} balAcc "
                  f"{m['balanced_accuracy']:.3f}  pred dyn "
                  f"{100 * m['predicted_dynamic_fraction']:.1f}% "
                  f"(GT {100 * m['gt_dynamic_fraction']:.1f}%)")
        write_json(ev / "confusion_matrix.json",
                   {"coherence": m_coh["confusion_matrix"],
                    "current": m_base["confusion_matrix"]})
        write_json(ev / "current_vs_coherence.json", compare_selectors(rows))
        write_json(out / "numerical" / "gt_metrics.json",
                   {"coherence": m_coh, "current": m_base})

        # section 35: is GT itself coherent?
        gt_node = torch.full((res["num_supernodes"],), -1, dtype=torch.long)
        for i in range(res["num_supernodes"]):
            mem = torch.nonzero(sid == int(res["node_ids"][i]),
                                as_tuple=False).reshape(-1)
            l = gtl[mem]
            l = l[l >= 0]
            if l.numel():
                gt_node[i] = int(l.double().mean().round())
        write_json(ev / "gt_coherence.json",
                   gt_edge_coherence(C, ei, ej, gt_node, lab))
    else:
        print(f"  no GT available: {gt.get('reason')}")
        write_json(out / "numerical" / "gt_mapping.json", gt)

    # Section 24: the two selectors on the same Gaussians, in 3D, over the
    # Phase-1 trajectories -- so a point called STATIC that visibly travels is
    # a false negative you can see rather than one you have to infer.
    from debug.motion_coherence_debug import write_selector_comparison_3d
    gt_vec = (gt["labels"] if gt.get("available")
              else torch.full((mu0.shape[0],), -1, dtype=torch.long))
    sel3d = write_selector_comparison_3d(
        dbg / "static_dynamic_comparison_3d.html", mu0, positions,
        gt_vec, pred_base, pred_dyn,
        metrics=({"current": m_base, "coherence": m_coh}
                 if m_base and m_coh else {}),
        title=f"{args.scene} — static/dynamic: motion baseline vs coherence")

    if res.get("node_class"):
        res["node_class_counts"] = res["extraction"]["counts"]
    write_support_3d(dbg / "motion_coherence_support_3d.html", res,
                     node_xyz=torch.stack([
                         mu0.cpu()[sid == int(res["node_ids"][i])].mean(0)
                         for i in range(res["num_supernodes"])]),
                     traj=None, gt_node=gt_node,
                     title=f"{args.scene} — motion coherence support")

    print(f"\n  === {args.scene} DONE in {time.perf_counter() - t0:.1f}s ===")
    print(f"  {out}")
    print(f"  3D A/B   {sel3d}")
    print(f"  3D       {dbg / 'motion_coherence_support_3d.html'}")
    print(f"  heatmaps {dbg / 'finite_strain_heatmap.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
