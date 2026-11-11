#!/usr/bin/env python
"""Fast, surface-centred endpoint 3DGS for tracking, from ONE multiview RGB-D frame.

RigEuler3DGS_Calibrated_Bootstrap_IEKF_Implementation_Spec.md section 3.

  1. back-project every camera's masked, valid depth at --frame with its sensor
     covariance (utils/noise/sensor_model.py; policy ideal/assumed -> the declared
     IEKF depth sigma), fit one common surface (estimator/surface_bootstrap.py)
  2. write a fixed-count 3DGS model (DC colour only, SH rest = 0) that every
     existing loader reads: <out>/point_cloud/iteration_0/point_cloud.ply, cfg_args
  3. short refinement (Adam on means, quaternions, scales, opacity, DC colour):
     no densification, pruning, opacity reset or SH promotion; Gaussian IDs and
     count are asserted unchanged. Loss per camera:
        masked RGB    sum M |C - I|^2 / (2 sigma_rgb,c^2)      (fixed W)
        silhouette    sum (A - M)^2 / (2 sigma_A^2)            (declared sigma_A)
        depth         Student-t NLL of inverse depth on V & M & A > 0.5,
                      Var(1/z) = Var(z) / z^4 from the per-camera model
        thickness     declared weight * mean (s_normal / s_tangent)^2  (geometric
                      regularizer, NOT a sensor statement)
     The seed fit and this loss use the same depth samples SEQUENTIALLY (fit, then
     refine); they are never counted as two independent measurements.
     A held-out pixel set (never in the loss) is scored at every checkpoint.
  4. at the final checkpoint, with the IEKF renderer: exact source responsibility
     w = T_before * alpha per (Gaussian, camera, pixel), verified against the
     rendered alpha; per-Gaussian camera ownership (diagnostic); the per-Gaussian
     6x6 Fisher under the shared sensor model and P0 = (F + Lambda_prior)^-1.

Outputs in --out (a model directory):

  point_cloud/iteration_<N>/point_cloud.ply  checkpoints (N in --checkpoints, <= --refine-iters)
  point_cloud.ply                            the final checkpoint, also at top level
  cfg_args                                   minimal layout for tools/run_iekf_scene.load_scene
  bootstrap_manifest.json / manifest.json    endpoint frame, sensor model id + policy, counts,
                                             input hashes, held-out scores, verifications
  geometry_information.npz                   P0_geometry [N,6,6], fisher, eigenvalues, seed-fit
                                             normal variances, Sigma_render (kept separate)
  source_responsibility.npz                  COO [gaussian_id, camera_id, flat_pixel_id, weight]
                                             + ownership diagnostics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
RGB_EXT = (".png", ".jpg", ".jpeg")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def colmap_cameras(colmap_dir: Path):
    """[(name, geometry dict)] in COLMAP order, from the sparse model alone."""
    from scene.dataset_readers import readColmapSceneInfo
    from utils.graphics_utils import getWorld2View2
    info = readColmapSceneInfo(str(colmap_dir), "../rgb", "", False, False, load_gt_parts=False)
    cams = sorted(info.train_cameras + info.test_cameras, key=lambda c: c.image_name)
    out = []
    for c in cams:
        w2c = getWorld2View2(c.R, c.T).astype(np.float64)
        c2w = np.linalg.inv(w2c)
        W, H = int(c.width), int(c.height)
        out.append((str(c.image_name).split("/")[0],
                    {"W": W, "H": H, "fx": W / (2 * math.tan(c.FovX / 2)), "fy": H / (2 * math.tan(c.FovY / 2)),
                     "cx": W / 2.0, "cy": H / 2.0, "R_wc": c2w[:3, :3], "t_wc": c2w[:3, 3], "w2c": w2c}))
    return out


def load_frame(source: Path, cam: str, t: int):
    from PIL import Image
    rp = next((source / "rgb" / cam / f"frame_{t:06d}{e}" for e in RGB_EXT
               if (source / "rgb" / cam / f"frame_{t:06d}{e}").is_file()), None)
    dp = source / "depth_npy" / cam / f"frame_{t:06d}.npy"
    mp = next((source / "mask" / cam / f"frame_{t:06d}{e}" for e in RGB_EXT
               if (source / "mask" / cam / f"frame_{t:06d}{e}").is_file()), None)
    if rp is None or not dp.is_file() or mp is None:
        raise SystemExit(f"{cam} frame {t}: rgb/depth/mask incomplete ({rp}, {dp}, {mp})")
    rgb = np.asarray(Image.open(rp).convert("RGB"), np.float32) / 255.0
    d = np.load(dp).astype(np.float32)
    m = np.asarray(Image.open(mp))
    m = (m[..., 0] if m.ndim == 3 else m) > 0
    return rgb, d, m, [rp, dp, mp]


def write_model(out: Path, it: int, params: dict, source: Path, colmap_subdir: str, gt_part=None):
    """A 3DGS model directory (sh_degree 3, SH rest = 0) every loader in the repo reads.
    `gt_part` (EVALUATION ONLY, synthetic inputs): written as the per-vertex fields gt_part /
    gt_dynamic exactly as GaussianModel.save_ply does (-1 = unlabelled in both)."""
    from plyfile import PlyData, PlyElement
    n = params["xyz"].shape[0]
    dc = (params["rgb"] - 0.5) / 0.28209479177387814
    op = np.clip(params["opacity"], 1e-4, 1 - 1e-4)
    attrs = {"x": params["xyz"][:, 0], "y": params["xyz"][:, 1], "z": params["xyz"][:, 2],
             "nx": np.zeros(n), "ny": np.zeros(n), "nz": np.zeros(n),
             "f_dc_0": dc[:, 0], "f_dc_1": dc[:, 1], "f_dc_2": dc[:, 2]}
    for i in range(45):
        attrs[f"f_rest_{i}"] = np.zeros(n)
    attrs["opacity"] = np.log(op / (1 - op))
    for i in range(3):
        attrs[f"scale_{i}"] = np.log(np.maximum(params["scales"][:, i], 1e-7))
    for i in range(4):
        attrs[f"rot_{i}"] = params["quat_wxyz"][:, i]
    if gt_part is not None:
        from utils.gt_part_labels import dynamic_from_part
        if len(gt_part) != n:
            raise AssertionError(f"{len(gt_part)} GT labels for {n} Gaussians")
        attrs["gt_part"] = np.asarray(gt_part, np.float64)
        attrs["gt_dynamic"] = dynamic_from_part(gt_part).astype(np.float64)
    dtype = [(k, "f4") for k in attrs]
    arr = np.empty(n, dtype=dtype)
    for k, v in attrs.items():
        arr[k] = v
    d = out / "point_cloud" / f"iteration_{it}"
    d.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(d / "point_cloud.ply"))
    (out / "cfg_args").write_text(
        f"Namespace(sh_degree=3, source_path='{source / colmap_subdir}', model_path='{out}', "
        f"images='../rgb', depths='depth_npy', masks='{source / 'mask'}', sapien_depth_is_metric=True, "
        f"resolution=1, white_background=False, train_test_exp=False, data_device='cpu', eval=False)")
    return d / "point_cloud.ply"


def params_from_gaussians(g) -> dict:
    from tools.run_iekf_scene import state_from_gaussians
    s = state_from_gaussians(g)
    q = torch.nn.functional.normalize(g.get_rotation.detach(), dim=1)
    return {"xyz": s.mu.cpu().numpy().astype(np.float64), "quat_wxyz": q.cpu().numpy().astype(np.float64),
            "scales": s.scales.cpu().numpy().astype(np.float64), "opacity": s.opacity.cpu().numpy()[:, 0],
            "rgb": s.colors.cpu().numpy().astype(np.float64)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame", type=int, required=True, help="the endpoint's original frame ID")
    ap.add_argument("--endpoint", choices=("initial", "final"), required=True)
    ap.add_argument("--out", required=True, help="new model directory")
    ap.add_argument("--colmap-subdir", default="colmap", help="camera geometry (identical for every frame)")
    ap.add_argument("--cameras", default="all", help="all | first N | cam_001,cam_002 (utils/camera_selection.py)")
    ap.add_argument("--sensor-model", default=None)
    ap.add_argument("--sensor-policy", choices=("ideal", "assumed"), default="ideal")
    ap.add_argument("--num-gaussians", type=int, default=30000)
    ap.add_argument("--knn", type=int, default=24)
    ap.add_argument("--refine-iters", type=int, default=1000)
    ap.add_argument("--checkpoints", default="0,250,500,1000,2000")
    ap.add_argument("--heldout-fraction", type=float, default=0.1)
    ap.add_argument("--rgb-model-sigma", type=float, default=0.0, help="declared, added in variance")
    ap.add_argument("--depth-model-sigma", type=float, default=0.0, help="declared [m], added in variance")
    ap.add_argument("--silhouette-sigma", type=float, default=0.1, help="declared sd of the A - M residual")
    ap.add_argument("--thickness-weight", type=float, default=0.01, help="declared regularizer strength")
    ap.add_argument("--p0-prior-sigma-pos", type=float, default=0.05, help="declared weak prior [m]")
    ap.add_argument("--p0-prior-sigma-rot", type=float, default=0.5, help="declared weak prior [rad]")
    ap.add_argument("--lr-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt-labels", choices=("auto", "off"), default="auto",
                    help="EVALUATION ONLY: auto = label the seeds from pointcloud/points3D_parts[_final].ply "
                         "of this endpoint when the dataset has it (synthetic); off = never (real captures)")
    a = ap.parse_args(argv)
    t_start = time.time()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    src, out = Path(a.source), Path(a.out)
    # a new directory; the orchestrator's 'running' stage_signature.json is the only allowed content
    if out.exists() and any(p.name != "stage_signature.json" for p in out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; give a new directory")
    out.mkdir(parents=True, exist_ok=True)

    # ---- sensor model ------------------------------------------------------
    from estimator.sparse_iekf import IEKFConfig
    from utils.noise.sensor_model import declared_model, load_sensor_model
    sm = load_sensor_model(a.sensor_model) if a.sensor_model else declared_model(a.sensor_policy)
    base = IEKFConfig()

    def depth_var(cam, z):
        v = (sm.depth_variance(cam, z) if sm.is_physical else
             np.full_like(z, base.depth_sigma_a ** 2 if base.depth_sigma_model == "constant" else
                          (base.depth_sigma_a + base.depth_sigma_b * z) ** 2))
        return v + a.depth_model_sigma ** 2

    def rgb_var(cam):
        v = sm.camera(cam).rgb_variance() if sm.is_physical else np.full(3, base.rgb_sigma ** 2)
        return v + a.rgb_model_sigma ** 2

    # ---- 1. samples and the surface fit ------------------------------------
    from estimator.surface_bootstrap import backproject, fit_surface, gaussian_parameters, voxel_seeds
    from utils.camera_selection import resolve, select_cameras
    cams = colmap_cameras(src / a.colmap_subdir)
    try:
        chosen = resolve(a.cameras, [c[0] for c in cams])
    except ValueError as ex:
        raise SystemExit(str(ex))
    cams = [c for c in cams if c[0] in chosen]
    Y, D, V, C, S, files = [], [], [], [], [], []
    obs = {}
    for ci, (name, geo) in enumerate(cams):
        rgb, dep, msk, fl = load_frame(src, name, a.frame)
        files += fl
        valid = msk & np.isfinite(dep) & (dep > 0)
        y, d, z, pix = backproject(dep, valid, geo)
        Y.append(y), D.append(d), V.append(depth_var(name, z)), C.append(rgb.reshape(-1, 3)[pix])
        S.append(np.stack([np.full(pix.size, ci), pix], 1))
        obs[name] = (rgb, dep, msk)
        print(f"  {name}: {pix.size} masked valid depth samples, sigma_D median "
              f"{1000 * math.sqrt(float(np.median(V[-1]))):.2f} mm")
    Y, D, V, C, S = (np.concatenate(x) for x in (Y, D, V, C, S))
    seeds, h = voxel_seeds(Y, a.num_gaussians)
    fit = fit_surface(Y, D, V, C, seeds, k=a.knn)
    params = gaussian_parameters(fit)
    n = params["xyz"].shape[0]
    print(f"  {n} seeds (voxel {1000 * h:.1f} mm) from {len(Y)} samples; median sigma_surface "
          f"{1000 * math.sqrt(float(np.median(fit['sigma_surface2']))):.2f} mm, inliers "
          f"{100 * float(np.median(fit['inlier_fraction'])):.0f} %")
    # EVALUATION ONLY: static / dynamic GT at the seeds (utils/gt_part_labels.py). The
    # population is fixed from here on, so the seed's label is the Gaussian's label.
    gt_part, gt_info = None, {"status": "OFF" if a.gt_labels == "off" else "NO_LABELLED_CLOUD",
                              "evaluation_only": True}
    if a.gt_labels == "auto":
        from utils.gt_part_labels import labelled_cloud, nn_labels, summary
        cloud = labelled_cloud(src, a.endpoint)
        if cloud is not None:
            lab = nn_labels(params["xyz"], cloud)
            gt_part = lab["part"]
            gt_info = {"status": "LABELLED", "evaluation_only": True, "cloud": cloud["path"],
                       "cloud_sha256": cloud["sha256"], "cloud_points": int(len(cloud["part"])),
                       "method": "nearest labelled sample of the same endpoint at the SEED position; "
                                 "ids fixed through the refinement",
                       "max_dist_m": lab["max_dist"], "factor_of_median_spacing": lab["factor"],
                       **summary(gt_part, lab["nn_distance"])}
            print(f"  GT labels (evaluation only): {gt_info['num_static']} static, {gt_info['num_dynamic']} "
                  f"dynamic, {gt_info['num_unlabelled']} unlabelled from {cloud['path']}")
    write_model(out, 0, params, src, a.colmap_subdir, gt_part)

    # ---- 2. refinement ------------------------------------------------------
    from gaussian_renderer import render
    from tools.run_iekf_scene import load_scene
    _ds, pipe, gs, sc = load_scene(str(out), str(src / a.colmap_subdir), camera_device="cpu", inference_only=True)
    cams_s = select_cameras(sorted(sc.getTrainCameras() + sc.getTestCameras(), key=lambda c: c.image_name), a.cameras)
    names = [str(c.image_name).split("/")[0] for c in cams_s]
    if names != [c[0] for c in cams]:
        raise SystemExit(f"camera order mismatch {names} vs {[c[0] for c in cams]}")
    from estimator.surface_bootstrap import bootstrap_loss_terms, pixel_partition
    data = []
    for ci, (name, cam) in enumerate(zip(names, cams_s)):
        rgb, dep, msk = obs[name]
        # one seeded split of the WHOLE image per camera, fixed for the whole refinement
        train_px, held = pixel_partition(msk.shape[0], msk.shape[1], a.heldout_fraction, a.seed, ci)
        t = lambda x: torch.as_tensor(x, device="cuda")                       # noqa: E731
        z = np.where(np.isfinite(dep) & (dep > 0), dep, 1.0)
        vz = depth_var(name, z)
        data.append({"cam": cam, "name": name, "rgb": t(rgb).permute(2, 0, 1), "M": t(msk.astype(np.float32)),
                     "train": t(train_px), "held": t(held),
                     "invz": t((1.0 / z).astype(np.float32)), "V": t(np.isfinite(dep) & (dep > 0)),
                     "var_invz": t((vz / z ** 4).astype(np.float32)),
                     "w_rgb": t((1.0 / rgb_var(name)).astype(np.float32)).reshape(3, 1, 1)})
    for p in (gs._features_rest,):
        p.requires_grad_(False)
    groups = [{"params": [gs._xyz], "lr": 1.6e-4 * a.lr_scale}, {"params": [gs._rotation], "lr": 1e-3 * a.lr_scale},
              {"params": [gs._scaling], "lr": 5e-3 * a.lr_scale}, {"params": [gs._opacity], "lr": 2.5e-2 * a.lr_scale},
              {"params": [gs._features_dc], "lr": 2.5e-3 * a.lr_scale}]
    opt = torch.optim.Adam(groups, eps=1e-15)
    bg0 = torch.zeros(3, device="cuda")
    ones = None
    nu = 5.0

    def losses(dd, keyset, with_stats=False):
        nonlocal ones
        cam = dd["cam"]
        r = render(cam, gs, pipe, bg0)
        if ones is None or ones.shape[0] != gs.get_xyz.shape[0]:
            ones = torch.ones((gs.get_xyz.shape[0], 3), device="cuda")
        A = render(cam, gs, pipe, bg0, override_color=ones)["render"][0]
        t_ = bootstrap_loss_terms(r["render"], A, r["depth"].squeeze(0), dd, dd[keyset],
                                  silhouette_sigma=a.silhouette_sigma, nu=nu)
        out = (t_["rgb"], t_["alpha"], t_["depth"])
        return (out, t_["stats"]) if with_stats else out

    ckpts = sorted({int(x) for x in a.checkpoints.split(",") if int(x) <= a.refine_iters} | {a.refine_iters})
    report = []

    def evaluate(it):
        with torch.no_grad():
            got = [losses(dd, part, with_stats=True) for part in ("held", "train") for dd in data]
        n = len(data)
        held, train = got[:n], got[n:]
        rep = {"iteration": it, "heldout_rgb": float(np.mean([float(p[0][0]) for p in held])),
               "heldout_silhouette": float(np.mean([float(p[0][1]) for p in held])),
               "heldout_depth_nll": float(np.mean([float(p[0][2]) for p in held])),
               "train_rgb": float(np.mean([float(p[0][0]) for p in train])),
               "train_silhouette": float(np.mean([float(p[0][1]) for p in train])),
               "train_depth_nll": float(np.mean([float(p[0][2]) for p in train])),
               "heldout_per_camera": [{"camera": dd["name"], **p[1]} for dd, p in zip(data, held)],
               "num_gaussians": int(gs.get_xyz.shape[0])}
        report.append(rep)
        print(f"  it {it:5d}: held-out rgb {rep['heldout_rgb']:.4g}  sil {rep['heldout_silhouette']:.4g}  "
              f"depth NLL {rep['heldout_depth_nll']:.4g}")

    n0 = gs.get_xyz.shape[0]
    evaluate(0)
    for it in range(1, a.refine_iters + 1):
        dd = data[(it - 1) % len(data)]
        lr_, ls_, ld_ = losses(dd, "train")
        s = gs.get_scaling
        thick = ((s.min(1).values / s.max(1).values.clamp_min(1e-9)) ** 2).mean()
        loss = lr_ + ls_ + ld_ + a.thickness_weight * thick
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if gs.get_xyz.shape[0] != n0:
            raise AssertionError("the Gaussian count changed during the bootstrap refinement")
        if it in ckpts:
            evaluate(it)
            write_model(out, it, params_from_gaussians(gs), src, a.colmap_subdir, gt_part)
    final_it = a.refine_iters
    if final_it == 0:
        write_model(out, 0, params, src, a.colmap_subdir, gt_part)
    shutil.copy(out / "point_cloud" / f"iteration_{final_it}" / "point_cloud.ply", out / "point_cloud.ply")

    # ---- 3. responsibility, ownership, Fisher, P0 ----------------------------
    from estimator.observation_terms import camera_terms
    from estimator.renderer_bridge import build_linearization
    from estimator.surface_bootstrap import geometry_covariance, ownership, source_responsibility
    from tools.run_iekf_scene import camera_specs, state_from_gaussians
    import dataclasses
    state = state_from_gaussians(gs)
    specs = camera_specs(cams_s, len(cams_s))
    with torch.no_grad():
        lin = build_linearization(state, specs, eps_depth=base.eps_depth)
        resp = source_responsibility(lin)
        own = ownership(resp, n, len(cams_s))
        cfg = dataclasses.replace(base, sensor_model=sm if sm.is_physical else None, camera_names=tuple(names),
                                  rgb_model_sigma=a.rgb_model_sigma, depth_model_sigma=a.depth_model_sigma)
        prec = []
        for v, dd in zip(lin.views, data):
            dz = torch.where(dd["V"], 1.0 / dd["invz"], torch.zeros_like(dd["invz"]))
            tm = camera_terms(v, dd["rgb"], dz, dd["V"] & (dd["M"] > 0), cfg, rgb_robust=False, camera=dd["name"])
            prec.append((tm.p_rgb * dd["M"][None], tm.p_d))
        F = lin.build_block_fisher(prec)
    cov = geometry_covariance(F, a.p0_prior_sigma_pos, a.p0_prior_sigma_rot)
    Sig = np.einsum("nij,nj,nkj->nik", state.R.double().cpu().numpy(),
                    state.scales.double().cpu().numpy() ** 2, state.R.double().cpu().numpy())
    np.savez_compressed(out / "geometry_information.npz", P0_geometry=cov["P0_geometry"], fisher=cov["fisher"],
                        fisher_eigenvalues=cov["fisher_eigenvalues"], P0_eigenvalues=cov["P0_eigenvalues"],
                        seed_fit_var_normal=fit["var_normal"], seed_fit_sigma_surface2=fit["sigma_surface2"],
                        Sigma_render=Sig)
    np.savez_compressed(out / "source_responsibility.npz", **{k: v for k, v in resp.items() if isinstance(v, np.ndarray)},
                        camera_share=own["camera_share"], dominant_camera=own["dominant_camera"],
                        camera_names=np.array(names))
    obsv = (cov["fisher_eigenvalues"] > 1e-6 * np.maximum(cov["fisher_eigenvalues"].max(1, keepdims=True), 1e-30)).sum(1)
    man = {"endpoint": a.endpoint, "endpoint_frame": a.frame, "source": str(src),
           "sensor_policy": sm.policy, "sensor_model_id": sm.model_id, "sensor_model_path": a.sensor_model,
           "num_gaussians": n, "count_unchanged": True, "refine_iters": final_it, "checkpoints": ckpts,
           "heldout": report, "seed_voxel_m": h, "knn": a.knn,
           "declared": {"rgb_model_sigma": a.rgb_model_sigma, "depth_model_sigma": a.depth_model_sigma,
                        "silhouette_sigma": a.silhouette_sigma, "thickness_weight": a.thickness_weight,
                        "p0_prior_sigma_pos_m": a.p0_prior_sigma_pos, "p0_prior_sigma_rot_rad": a.p0_prior_sigma_rot},
           "responsibility_check": {"max_abs_sum_minus_rendered_alpha": resp["max_abs_alpha_sum_error_per_camera"],
                                    "entries": int(resp["weight"].size), "unseen_gaussians": own["unseen"]},
           "observability": {"observable_directions_histogram": np.bincount(obsv, minlength=7).tolist(),
                             "note": cov["approximation"]},
           "camera_names": names, "gt_labels": gt_info,
           "input_sha256": hashlib.sha256("".join(_sha(f) for f in files).encode()).hexdigest(),
           "policy_notes": ["only the endpoint frame's own observations are used",
                            "Sigma_render (rendering footprint) is stored separately from P0 (centre uncertainty)",
                            "no densification / pruning / opacity reset / SH promotion; ids fixed"],
           "runtime_s": round(time.time() - t_start, 1)}
    for f in ("bootstrap_manifest.json", "manifest.json"):
        (out / f).write_text(json.dumps(man, indent=1, default=float))
    print(f"  responsibility: {resp['weight'].size} entries, max |sum w - A| "
          f"{max(resp['max_abs_alpha_sum_error_per_camera']):.2e}, unseen Gaussians {own['unseen']}")
    print(f"  P0: observable directions per Gaussian {man['observability']['observable_directions_histogram']}")
    print(f"wrote {out} in {man['runtime_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
