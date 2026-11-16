#!/usr/bin/env python
"""Phase 5 (unified spec 2026-11-11, PART II): train the completed Phase-4 canonical model on ALL views, joint FROZEN.

    python -u tools/train_phase5_rgbd_gaussians.py --phase4-dir OUT/phase4/RUN --output OUT/phase5/RUN \\
        [--train-mode gaussian_rgbd|sh_only] [--loss-ssim-impl fused_ssim|3dgs_python] [--loss-use-depth false ...]

--phase4-dir is the Phase-4 run directory holding 4a_original_membership/ and 4c_accepted_moving/. Both must have a
valid complete.json (hash-verified), 4C must have been built from that 4A, and the upstream files (source, model,
Phase-2/3 directories, cameras, frame window) are read from the 4A manifest and re-verified by hash and joint
fingerprint. Phase 5 trains on EVERY (camera, Phase-3 frame) view, whatever frame stride 4B / 4C used.

Loss (recovery/rgbd_loss.py): lambda_rgb [(1 - 0.2) L1 + 0.2 (1 - SSIM)] on the masked image with the 3DGS
utils/loss_utils.l1_loss and the CUDA fused_ssim used by 3DGS train.py, + lambda_depth L_depth + lambda_mask L_mask.
--train-mode sh_only trains DC/SH only (depth and mask are monitors). Never started by Phase 4; never refines the joint.
Outputs: input_proof.json, unresolved_originals.npz, config_resolved.json, manifest.json, loss_log.csv,
frame_camera_metrics.csv, renderer_parity.json, checkpoints/, final/{point_cloud.ply, lineage, joint_unchanged_proof.json,
joint/}, initial/final_all_view_metrics.json, complete.json.
"""
import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import recovery  # noqa: E402,F401


def _bool(s):
    return str(s).lower() in ("1", "true", "yes")


def _add_cfg(ap, cls, prefix, skip=()):
    for f in fields(cls):
        if f.name in skip:
            continue
        t = {"int": int, "float": float, "bool": _bool, "str": str}.get(f.type if isinstance(f.type, str) else f.type.__name__)
        if t is not None:
            ap.add_argument(f"{prefix}{f.name.replace('_', '-')}", type=t, default=f.default)


def preflight(phase4_dir: Path) -> dict:
    """Refuse anything but a complete, untampered 4A -> 4C chain. Returns both complete.json payloads + setup."""
    from recovery import phase4_artifacts as AR
    a_dir, c_dir = phase4_dir / AR.STAGE_A, phase4_dir / AR.STAGE_C
    ca = AR.require_complete(a_dir, AR.STAGE_A)
    cc = AR.require_complete(c_dir, AR.STAGE_C)
    if cc["phase4a_artifact_sha256"] != ca["artifact_sha256"]:
        raise SystemExit(f"{c_dir} was built from a different 4A artifact than {a_dir}")
    if cc["canonical_sha256_originals"] != ca["canonical_sha256"]:
        raise SystemExit("4C's original canonical Gaussians differ from 4A's")
    setup = json.loads((a_dir / "manifest.json").read_text())["setup"]
    if setup != ca["setup"]:
        raise SystemExit("4A manifest.json setup differs from its complete.json")
    fc = c_dir / "final_canonical"
    if not (fc / "point_cloud.ply").is_file():
        raise SystemExit(f"{fc}/point_cloud.ply missing")
    return {"4a": ca, "4c": cc, "setup": setup, "a_dir": a_dir, "c_dir": c_dir, "final_canonical": fc}


def main(argv=None):
    from recovery.rgbd_loss import LossConfig
    from recovery.training import Phase5Config
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase4-dir", required=True, help="Phase-4 run directory (4a_original_membership/, 4c_accepted_moving/)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-mode", default="gaussian_rgbd", choices=("gaussian_rgbd", "sh_only"))
    ap.add_argument("--view-cache", type=int, default=128, help="decoded views kept on the CPU (LRU)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-frames", type=int, default=0, help="DEBUG ONLY: evenly spaced frames; output NOT_FOR_RESULTS")
    ap.add_argument("--dry-run", action="store_true", help="preflight + upstream contract only")
    _add_cfg(ap, Phase5Config, "--", skip=("mode", "loss", "bg", "seed"))
    _add_cfg(ap, LossConfig, "--loss-")
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from recovery import phase4_artifacts as AR
    from recovery.gaussians import GaussianTensors
    from recovery.io_setup import git_state, setup, write_json
    from recovery.phase4_stages import load_membership
    from recovery.phase4a_exact import UNDECIDED
    from recovery.training import train_phase5
    t0 = time.time()
    p4 = Path(a.phase4_dir)
    pf = preflight(p4)
    st = pf["setup"]
    ns = argparse.Namespace(source=st["source"], model=st["model"], phase2_dir=st["phase2_dir"],
                            phase3_dir=st["phase3_dir"], joint_type=st["joint_type"], direction=st["direction"],
                            cameras=st.get("cameras"), frame_window=st.get("frame_window"),
                            sensor_model=st.get("sensor_model"), dry_run=a.dry_run, view_cache=a.view_cache,
                            smoke_frames=a.smoke_frames)
    up, cams, views, extra = setup(ns)
    if dict(up.hashes) != pf["4a"]["input_proof"]["upstream_sha256"]:
        raise SystemExit("the upstream files changed since 4A (hash mismatch)")
    if AR.joint_fingerprint(up.joint) != pf["4c"]["joint_fingerprint"]:
        raise SystemExit("the Phase-3 joint differs from the one Phase 4 used")
    if a.dry_run:
        print(json.dumps({"DRY_RUN": True, "phase4_dir": str(p4), "N_final": pf["4c"]["N_final"],
                          "recovery_views": pf["4c"].get("recovery_views"), "joint": up.joint.to_dict()},
                         indent=1, default=str))
        return 0
    out = Path(a.output)
    if (out / "complete.json").is_file():
        print(f"{out}/complete.json exists: Phase 5 output is immutable; not re-run")
        return 0
    fc = pf["final_canonical"]
    g = GaussianTensors.load_ply(fc / "point_cloud.ply")
    g.lineage = GaussianTensors.load_lineage(fc)
    g.check()
    if len(g) != pf["4c"]["N_final"]:
        raise SystemExit(f"4C final_canonical has {len(g)} rows, complete.json says {pf['4c']['N_final']}")
    status, render_label, z = load_membership(pf["a_dir"])
    n0 = len(status)
    if not np.array_equal(g.lineage["gaussian_uid"][:n0], z["uid"]) or \
            not np.array_equal(g.lineage["material"][:n0].astype(bool), render_label):
        raise SystemExit("4C canonical rows / material labels do not match the 4A membership")
    out.mkdir(parents=True, exist_ok=True)
    und = np.isin(status, UNDECIDED)
    AR.save_npz(out / "unresolved_originals.npz", gaussian_uid=z["uid"][und], row=np.nonzero(und)[0],
                status=status[und].astype(str), frozen_render_label=render_label[und])
    loss = LossConfig(**{f.name: getattr(a, f"loss_{f.name}") for f in fields(LossConfig)})
    kw = {f.name: getattr(a, f.name) for f in fields(Phase5Config) if f.name not in ("mode", "loss", "bg", "seed")}
    cfg = Phase5Config(mode=a.train_mode, seed=a.seed, loss=loss, **kw)
    not_for_results = bool(a.smoke_frames) or bool(pf["4a"].get("NOT_FOR_RESULTS")) or bool(pf["4c"].get("NOT_FOR_RESULTS"))
    proof = {"phase4_dir": str(p4), "phase4a_artifact_sha256": pf["4a"]["artifact_sha256"],
             "phase4c_artifact_sha256": pf["4c"]["artifact_sha256"], "upstream_sha256": dict(up.hashes),
             "joint_fingerprint": pf["4c"]["joint_fingerprint"], "N_final_phase4": pf["4c"]["N_final"],
             "recovery_views_phase4": pf["4c"].get("recovery_views"), "phase5_views": len(views),
             "unresolved_originals": int(und.sum()),
             "unresolved_rule": "rendered with 4A's frozen render_label; never converted inside Phase 5"}
    write_json(out / "input_proof.json", proof)
    manifest = {"stage": "phase5", "train_mode": cfg.mode, "upstream": up.manifest(), "input_proof": proof,
                "setup": st, "code": git_state(REPO), **extra, "NOT_FOR_RESULTS": not_for_results, "argv": sys.argv,
                "rgb_losses": {"l1": "utils/loss_utils.l1_loss (3DGS)",
                               "ssim": {"fused_ssim": "fused_ssim CUDA package (as called by 3DGS train.py)",
                                        "3dgs_python": "utils/loss_utils.ssim (3DGS reference)"}[loss.ssim_impl],
                               "lambda_dssim": loss.lambda_ssim}}
    write_json(out / "manifest.json", manifest)
    print(f"Phase 5 [{cfg.mode}] {p4} -> {out}: {len(g)} Gaussians ({int(und.sum())} unresolved originals, frozen "
          f"labels), {len(views)} views ({len(cams)} cameras x {len(views) // max(1, len(cams))} frames); loss "
          f"{loss.lambda_rgb} x [L1 + SSIM({loss.ssim_impl})], depth x {loss.lambda_depth if loss.use_depth else 0}, "
          f"mask x {loss.lambda_mask if loss.use_mask else 0}", flush=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    r = train_phase5(g, up.joint, views, cams, cfg, out, manifest=manifest)
    AR.require_complete(pf["a_dir"], AR.STAGE_A)                    # Phase 4 untouched
    AR.require_complete(pf["c_dir"], AR.STAGE_C)
    if AR.joint_fingerprint(up.joint) != pf["4c"]["joint_fingerprint"]:
        raise RuntimeError("the joint fingerprint changed during Phase 5")
    proof_j = json.loads((out / "final" / "joint_unchanged_proof.json").read_text())
    if not proof_j["identical"] or proof_j["joint_optimized"]:
        raise RuntimeError("joint_unchanged_proof failed")
    keys = ("total", "psnr_fg", "depth_l1", "mask_l1")
    mi = {k: float(np.mean([x[k] for x in r["initial"]])) for k in keys}
    mf = {k: float(np.mean([x[k] for x in r["final"]])) for k in keys}
    write_json(out / "initial_all_view_metrics.json", {"mean": mi, "views": len(r["initial"])})
    write_json(out / "final_all_view_metrics.json", {"mean": mf, "views": len(r["final"])})
    write_json(out / "resource_report.json", {"seconds": time.time() - t0, "peak_gpu_memory_mb": r["peak_gpu_memory_mb"]})
    if cfg.render_frames > 0 and not (out / "renders" / "index.json").is_file():
        raise RuntimeError("the visual comparison renders were not written")
    from recovery.articulated_ply import write_articulated_ply
    fp = out / "final" / "point_cloud.ply"
    art = write_articulated_ply(fp, GaussianTensors.load_lineage(out / "final"), up.joint, fp,
                                unresolved_uids=z["uid"][und], extra_comments=[f"phase5 {cfg.mode} from {p4}"])
    print(f"articulated PLY: {fp} (part_id, lineage, joint {art['joint_type']} axis/origin/range, per-frame q)")
    write_json(out / "complete.json", {"stage": "phase5", "train_mode": cfg.mode, **proof, "final_gaussians": len(g),
                                       "visual_comparison": str(out / "renders"),
                                       "all_view_metrics_initial": mi, "all_view_metrics_final": mf,
                                       "joint_unchanged": True, "NOT_FOR_RESULTS": not_for_results,
                                       "seconds": time.time() - t0})
    print(f"Phase 5 [{cfg.mode}] done in {(time.time() - t0) / 60:.1f} min: {len(g)} Gaussians; all-view mean loss "
          f"{mi['total']:.5f} -> {mf['total']:.5f}, PSNR_fg {mi['psnr_fg']:.2f} -> {mf['psnr_fg']:.2f} dB, depth L1 "
          f"{mi['depth_l1'] * 1000:.2f} -> {mf['depth_l1'] * 1000:.2f} mm; joint unchanged -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
