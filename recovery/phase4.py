"""Phase 4 driver: 4A material completion -> 4B discovery -> 4C acceptance -> one canonical model.

Shared by tools/run_phase4_recovery.py and the tests. The joint (J_3, q_3) is read, never written; its
files are hashed before and after.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import candidate_acceptance as C
from . import canonical_fusion as FU
from . import material_completion as MC
from . import residual_discovery as RD
from .gaussians import MOVING, PROV_NAMES, PROV_ORIGINAL, PROV_ORIGINAL_CORE, GaussianTensors
from .rgbd_loss import view_loss


def scene_loss(scene, moving):
    rows = []
    for vr in scene.views:
        t = view_loss(scene.render(vr, moving), scene.view(vr), scene.loss_cfg)
        rows.append({"camera": vr.cam.name, "frame": int(vr.frame),
                     **{k: (float(v) if torch.is_tensor(v) else v) for k, v in t.items()}})
    return rows


def run_phase4(scene: MC.Phase4Scene, core: np.ndarray, pol: MC.PolicyConfig, dcfg: RD.DiscoveryConfig,
               fcfg: FU.FusionConfig, acfg: C.AcceptConfig, out_dir=None, upstream_manifest=None,
               sigma_depth_fn=None, log=print, discovery=True) -> dict:
    t0 = time.time()
    n_orig = len(scene.g)
    lin = scene.g.lineage
    lin["original_id"] = np.arange(n_orig, dtype=np.int64)
    lin["provenance"] = np.full(n_orig, PROV_ORIGINAL, np.int8)
    lin["provenance"][core] = PROV_ORIGINAL_CORE
    mv0 = torch.zeros(n_orig, dtype=torch.bool, device=scene.device)
    mv0[torch.as_tensor(core, device=scene.device)] = True
    audit_before = scene_loss(scene, mv0)
    # ---------------- 4A
    log("Phase 4A: original-population material completion")
    res = MC.run_phase4a(scene, core, pol, log=log)
    if out_dir:
        MC.write_outputs(out_dir, scene, res, pol, upstream_manifest or {})
    if res["aborted"]:
        log("Phase 4A: UPSTREAM_JOINT_INVALID -- Phase 4 stops; no material change, no birth")
        return {"aborted": True, "phase4a": res, "seconds": time.time() - t0}
    lin["material"] = res["material"].astype(np.int8)
    moving = torch.as_tensor(res["material"] == MOVING, device=scene.device)
    decisions, cells, maps_stats = [], {}, []
    if discovery:
        # ---------------- 4B
        log("Phase 4B: unexplained foreground at every (camera, frame)")
        maps, maps_stats = RD.discovery_maps(scene, moving, dcfg)
        obs, sub = FU.collect(scene, maps, fcfg, sigma_depth_fn, moving=moving)
        for h in ("S", "M"):
            cells[h] = FU.fuse(obs[h], fcfg)
        log(f"  4B proposals: static {len(cells['S'][0])} cells / {len(np.unique(cells['S'][1]))} patches, "
            f"moving {len(cells['M'][0])} cells / {len(np.unique(cells['M'][1]))} patches; coherence "
            f"S {FU.coherence(cells['S'][0]):.2f} M {FU.coherence(cells['M'][0]):.2f} frames per cell")
        if out_dir:
            d = Path(out_dir) / "discovery"
            (d / "residual_maps").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "residual_maps" / "codes.npz", codes=np.stack([m.numpy() for m in maps]),
                                legend=json.dumps(RD.CODES),
                                views=json.dumps([{"camera": v.cam.name, "frame": int(v.frame)} for v in scene.views]))
            np.savez_compressed(d / "pixel_evidence.npz", **{
                f"{h}_{k}": np.concatenate([getattr(o, k) for o in obs[h]]) if obs[h] else np.zeros(0)
                for h in ("S", "M") for k in ("x", "n", "rgb", "view", "frame")})
            np.savez_compressed(d / "proposed_patches.npz", **{
                f"{h}_{k}": np.array([c[k] for c in cells[h][0]]) for h in ("S", "M")
                for k in ("mu", "normal", "rgb", "count", "views", "frames", "first_frame", "thickness")},
                **{f"{h}_patch": cells[h][1] for h in ("S", "M")})
            (d / "source_provenance.json").write_text(json.dumps({
                "pixel_convention": "K^-1 [u + 0.5, v + 0.5, 1], metric axial depth from depth_npy",
                "eligible_codes": ["COVERAGE_GAP", "FRONT_SURFACE"], "subsampled_views": sub,
                "per_view_counts": maps_stats, "first_frame_is_provenance_only": True}, indent=1))
            (d / "layer_association.json").write_text(json.dumps({
                "rule": "cells keyed by (canonical voxel, dominant signed normal axis); patches = touching cells "
                        "with the same normal bin; opposite faces never fuse", "fusion": fcfg.to_dict(),
                "discovery": dcfg.to_dict()}, indent=1))
            (d / "proposal_scores.json").write_text(json.dumps({
                h: {"cells": len(cells[h][0]), "patches": int(len(np.unique(cells[h][1]))),
                    "coherence_frames_per_cell": FU.coherence(cells[h][0])} for h in ("S", "M")}, indent=1))
        # ---------------- 4C
        log("Phase 4C: whole-sequence acceptance")
        cands = C.pair_patches(cells["S"][0], cells["S"][1], cells["M"][0], cells["M"][1], acfg)
        g_new, moving, decisions = C.run_phase4c(scene, moving, cands, acfg, fcfg.voxel, log=log)
    n_acc = len(scene.g) - n_orig
    assert len(scene.g) == n_orig + n_acc and np.array_equal(scene.g.lineage["original_id"][:n_orig],
                                                              np.arange(n_orig))
    scene.g.lineage["material"] = moving.cpu().numpy().astype(np.int8)
    scene.g.check()
    audit_after = scene_loss(scene, moving)
    if out_dir:
        d = Path(out_dir) / "final_canonical"
        d.mkdir(parents=True, exist_ok=True)
        scene.g.save_ply(d / "point_cloud.ply")
        scene.g.save_lineage(d)
        torch.save({"xyz": scene.g._xyz.cpu(), "rotation": scene.g._rotation.cpu(), "scaling": scene.g._scaling.cpu(),
                    "opacity": scene.g._opacity.cpu(), "features_dc": scene.g._features_dc.cpu(),
                    "features_rest": scene.g._features_rest.cpu(), "lineage": scene.g.lineage,
                    "max_sh_degree": scene.g.max_sh_degree}, d / "checkpoint.pt")
        prov = scene.g.lineage["provenance"]
        (d / "source_provenance.json").write_text(json.dumps({
            "counts_by_provenance": {PROV_NAMES[int(k)]: int((prov == k).sum()) for k in np.unique(prov)},
            "n_original": n_orig, "n_accepted": n_acc, "n_final": len(scene.g),
            "status_counts_4A": {s: int((res["status"] == s).sum()) for s in MC.STATUS_CODES},
            "decisions_4C": decisions}, indent=1, default=str))
        (d / "joint_reference.json").write_text(json.dumps(scene.joint.to_dict(), indent=1))
        (d / "counts.json").write_text(json.dumps({
            "n_original": n_orig, "n_accepted": n_acc, "n_final": len(scene.g),
            "moving": int(scene.g.lineage["material"].sum()), "static": int((scene.g.lineage["material"] == 0).sum())}))
        (d / "full_sequence_render_audit.json").write_text(json.dumps({
            "before_phase4": audit_before, "after_phase4": audit_after,
            "mean_total_before": float(np.mean([r["total"] for r in audit_before])),
            "mean_total_after": float(np.mean([r["total"] for r in audit_after])),
            "validation_note": "4C validation views are the odd Phase-3 processing rows; Phase 3 used every frame, so "
                               "validation is conditional on the upstream joint"}, indent=1))
    return {"aborted": False, "phase4a": res, "decisions": decisions, "n_original": n_orig, "n_accepted": n_acc,
            "moving": moving, "audit_before": audit_before, "audit_after": audit_after, "cells": cells,
            "seconds": time.time() - t0}
