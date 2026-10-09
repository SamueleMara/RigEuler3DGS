"""Section 50: offline diagnostic for pixel-mediated matching.

Runs E_pixel on an ALREADY solved trajectory and scores the pixel-only classifier

    k_i^pix = argmin_k E_ik^pixel

against the debug ground truth, stratified by confidence. Ground truth is read
ONLY here, for evaluation -- never by the matching module itself (section 37).

The point is to find out whether the signal is useful BEFORE it is allowed to
change the EM loop.
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

from rigid_parts.pixel_matching import (  # noqa: E402
    build_target_cache,
    candidate_pixel_evidence,
    covariance_from_scale_rotation,
)
from scene.gaussian_model import GaussianModel  # noqa: E402


def _final_cameras(scene_cameras, frame_ids):
    """Cameras belonging to the LAST dynamic frame."""
    last = max(frame_ids)
    out = []
    for idx, cam in enumerate(scene_cameras):
        fid = getattr(cam, "dynamic_frame_id", getattr(cam, "frame_id", idx))
        if int(fid) == int(last):
            out.append(cam)
    return out


def score(pred, gt, mask=None):
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    if pred.size == 0:
        return {"n": 0}
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "n": int(pred.size),
        "accuracy": float((pred == gt).mean() * 100),
        "iou": float(tp / max(tp + fp + fn, 1) * 100),
        "precision": float(prec * 100),
        "recall": float(rec * 100),
        "f1": float(2 * prec * rec / max(prec + rec, 1e-9) * 100),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="output/<run> whose solved trajectory to use")
    ap.add_argument("--source_path", default="input/USB_100109")
    ap.add_argument("--final_3dgs_path", default="output/USB_100109_static_endpoints_14k/shared/frame_t_static")
    ap.add_argument("--gt_ply", default="input/USB_100109/pointcloud/points3D_parts.ply")
    ap.add_argument("--top_l", type=int, default=4)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--sigma_z_ratio", type=float, default=0.02)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = REPO / args.run / "dynamic_debug" / "nested_part_embedding_killing"
    tracks = torch.load(run_dir / "nested_killing_tracks.pt", map_location="cpu", weights_only=False)
    xyz0 = tracks["canonical_xyz"].cuda()
    rot0 = tracks["canonical_rotation"].cuda()
    part_R = tracks["part_rotation"][-1].cuda()
    part_t = tracks["part_translation"][-1].cuda()
    part_q = tracks["part_rotation_quaternions"][-1].cuda()
    k_plus_1 = int(part_R.shape[0])
    n = int(xyz0.shape[0])
    bbox = float((xyz0.max(0).values - xyz0.min(0).values).norm())

    # ---- final model + its cameras ----------------------------------------
    final_dir = Path(args.final_3dgs_path)
    plys = sorted(final_dir.glob("point_cloud/iteration_*/point_cloud.ply"),
                  key=lambda p: int(p.parent.name.rsplit("_", 1)[-1]))
    final_model = GaussianModel(3)
    final_model.load_ply(str(plys[-1]))
    final_xyz = final_model.get_xyz.detach().cuda()
    print(f"final 3DGS: {final_xyz.shape[0]} gaussians from {plys[-1]}")

    # Reuse the run's own recorded configuration so the scene loads exactly as
    # it did during training (the dynamic validator has many coupled flags).
    import ast

    from arguments import GroupParams

    cfg_text = (REPO / args.run / "cfg_args").read_text().strip()
    body = cfg_text[cfg_text.index("(") + 1: cfg_text.rindex(")")]
    cfg = {}
    for part in body.split(", "):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            cfg[key.strip()] = ast.literal_eval(value)
        except Exception:
            cfg[key.strip()] = value
    model_args = GroupParams()
    for key, value in cfg.items():
        setattr(model_args, key, value)
    model_args.source_path = str((REPO / args.source_path).resolve())
    model_args.model_path = str(REPO / args.run)

    from scene import Scene
    # dynamic_args is what expands the fixed colmap cameras into the
    # per-frame dynamic sequence (60 frames x 8 views).
    scene = Scene(model_args, GaussianModel(3), shuffle=False, dynamic_args=model_args)
    cams = scene.getTrainCameras()
    frame_ids = [int(getattr(c, "dynamic_frame_id", getattr(c, "frame_id", i))) for i, c in enumerate(cams)]
    final_cams = _final_cameras(cams, frame_ids)
    print(f"cameras: {len(cams)} total, {len(final_cams)} at final frame {max(frame_ids)}")

    # ---- section 7 cache, section 14-21 candidate evidence -----------------
    cache_dir = REPO / args.run / "dynamic_debug" / "pixel_matching" / "target_cache"
    target_maps = build_target_cache(final_cams, final_model, cache_dir,
                                     top_l=args.top_l, final_model_id=str(plys[-1]))
    print(f"cached {len(target_maps)} contributor maps -> {cache_dir}")

    with torch.no_grad():
        cand_xyz = torch.stack([xyz0 @ part_R[k].T + part_t[k] for k in range(k_plus_1)], dim=1)
        from utils.rigid_transform_utils import apply_rotation_to_gaussian_quaternions
        cand_q = torch.stack(
            [apply_rotation_to_gaussian_quaternions(rot0, part_q[k]) for k in range(k_plus_1)], dim=1)
        # The canonical model is the run's OWN trained output: the endpoint ply
        # has a different Gaussian count after densification/pruning.
        src_plys = sorted((REPO / args.run).glob("point_cloud/iteration_*/point_cloud.ply"),
                          key=lambda q: int(q.parent.name.rsplit("_", 1)[-1]))
        src = GaussianModel(3)
        src.load_ply(str(src_plys[-1]))
        scaling = src.get_scaling.detach().cuda()
        opacity = src.get_opacity.detach().cuda()
        if int(scaling.shape[0]) != n:
            raise RuntimeError(
                f"source model has {scaling.shape[0]} gaussians but the trajectory has {n}")
        cov = covariance_from_scale_rotation(
            scaling[:, None, :].expand(n, k_plus_1, 3).reshape(-1, 3),
            cand_q.reshape(-1, 4),
        ).reshape(n, k_plus_1, 3, 3)

    ev = candidate_pixel_evidence(
        candidate_xyz=cand_xyz, candidate_cov3d=cov, opacity=opacity,
        cameras=final_cams, target_maps=target_maps, final_xyz=final_xyz,
        bbox_diag=bbox, window=args.window, sigma_z_ratio=args.sigma_z_ratio,
    )

    # ---- section 50 report (GT used for EVALUATION ONLY) -------------------
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    g = PlyData.read(args.gt_ply)["vertex"]
    gx = np.stack([g["x"], g["y"], g["z"]], 1)
    gl = np.asarray(g["part"])
    _, idx = cKDTree(gx).query(xyz0.cpu().numpy())
    gt = gl[idx]

    pred = ev.energy.argmin(dim=1).cpu().numpy()
    conf = ev.confidence.max(dim=1).values.cpu().numpy()
    covered = (ev.num_views.max(dim=1).values > 0).cpu().numpy()

    report = {
        "num_gaussians": n,
        "coverage_fraction": float(covered.mean()),
        "overall": score(pred, gt),
        "covered_only": score(pred, gt, covered),
        "confidence_quantiles": {},
        "mean_confidence": float(conf.mean()),
        "mean_energy_margin": float(
            (ev.energy.topk(2, dim=1, largest=False).values[:, 1]
             - ev.energy.min(dim=1).values).mean()),
    }
    qs = np.quantile(conf[covered], [0.25, 0.5, 0.75]) if covered.any() else [0, 0, 0]
    for name, lo in (("q1", -np.inf), ("q2", qs[0]), ("q3", qs[1]), ("q4", qs[2])):
        m = covered & (conf > lo)
        report["confidence_quantiles"][name] = score(pred, gt, m)
    report["high_confidence"] = score(pred, gt, covered & (conf > qs[2]))

    print(json.dumps(report, indent=2))
    out = Path(args.out) if args.out else run_dir.parent / "pixel_matching" / "diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    torch.save({k: v.cpu() for k, v in ev.as_dict().items()},
               out.parent / "candidate_evidence.pt")
    print(f"\nwrote {out} and candidate_evidence.pt")


if __name__ == "__main__":
    main()
