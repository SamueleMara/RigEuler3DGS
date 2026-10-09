"""Independent endpoint-frame static 3DGS reconstruction for endpoint transport."""

from __future__ import annotations

import json
from pathlib import Path
from random import randint
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from gaussian_renderer import render, render_feature_field, render_sam3_feature_field
from scene.gaussian_model import GaussianModel
from utils.general_utils import get_expon_lr_func
from utils.graphics_utils import BasicPointCloud, fov2focal
from utils.loss_utils import _load_mask_for_view, _load_sam3_target_for_view
from utils.system_utils import mkdir_p


def _frame_id(camera):
    value = getattr(camera, "frame_id", None)
    if value is not None:
        return int(value)
    name = str(getattr(camera, "image_name", ""))
    import re

    match = re.search(r"frame[_-]?(\d+)", name)
    return int(match.group(1)) if match else 0


def resolve_endpoint_frame(dynamic_cameras, requested):
    frames = sorted({_frame_id(camera) for camera in dynamic_cameras})
    if not frames:
        raise RuntimeError("Endpoint static model requires dynamic cameras")
    if int(requested) >= 0:
        frame = int(requested)
        if frame not in frames:
            raise RuntimeError(f"Endpoint frame {frame} is not present in dynamic cameras: {frames[:5]}...{frames[-5:]}")
        return frame
    return int(frames[-1])


def build_static_snapshot_cameras(dynamic_cameras, frame_idx: int):
    cameras = [camera for camera in dynamic_cameras if _frame_id(camera) == int(frame_idx)]
    if not cameras:
        raise RuntimeError(f"No synchronized cameras found for endpoint frame {int(frame_idx)}")
    cameras.sort(key=lambda camera: str(getattr(camera, "dynamic_camera_id", getattr(camera, "image_name", ""))))
    return cameras


def _endpoint_root(scene_model_path, args, frame_idx):
    requested = str(getattr(args, "endpoint_model_path", "") or "").strip()
    if requested:
        return Path(requested).expanduser().resolve()
    return Path(scene_model_path).expanduser().resolve() / "endpoint_static" / f"frame_{int(frame_idx):06d}"


def _find_model_ply(root, preferred_iteration=None):
    path = Path(root).expanduser()
    if path.is_file() and path.suffix.lower() == ".ply":
        return path.resolve()
    if preferred_iteration is not None:
        candidate = path / "point_cloud" / f"iteration_{int(preferred_iteration)}" / "point_cloud.ply"
        if candidate.exists():
            return candidate.resolve()
    pc_root = path / "point_cloud"
    if not pc_root.is_dir():
        return None
    candidates = []
    for child in pc_root.glob("iteration_*"):
        try:
            iteration = int(child.name.split("_")[-1])
        except ValueError:
            continue
        ply = child / "point_cloud.ply"
        if ply.exists():
            candidates.append((iteration, ply))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1].resolve()


def _camera_depth(camera):
    depth = getattr(camera, "invdepthmap", None)
    if depth is None:
        return None
    depth = depth.detach().float()
    if depth.ndim == 3:
        depth = depth[0]
    return depth


def _backproject_camera_depth(camera, dataset, args, *, max_pixels_per_camera=120_000):
    depth = _camera_depth(camera)
    if depth is None or not bool(getattr(camera, "depth_reliable", False)):
        return None
    device = depth.device
    h, w = int(depth.shape[-2]), int(depth.shape[-1])
    valid = torch.isfinite(depth) & (depth > 1.0e-8)
    mask = _load_mask_for_view(camera, dataset, args, (h, w), device=device)
    if mask is not None:
        valid &= mask > 0.5
    ids = torch.nonzero(valid, as_tuple=False)
    if ids.numel() == 0:
        return None
    if ids.shape[0] > int(max_pixels_per_camera):
        select = torch.linspace(0, ids.shape[0] - 1, int(max_pixels_per_camera), device=device).round().long().unique()
        ids = ids.index_select(0, select)
    v = ids[:, 0].float()
    u = ids[:, 1].float()
    z = depth[ids[:, 0], ids[:, 1]].float()
    fx = float(fov2focal(float(camera.FoVx), w))
    fy = float(fov2focal(float(camera.FoVy), h))
    cx = 0.5 * float(max(w - 1, 1))
    cy = 0.5 * float(max(h - 1, 1))
    camera_points = torch.stack(((u - cx) / fx * z, (v - cy) / fy * z, z), dim=-1)
    homogeneous = torch.cat((camera_points, torch.ones_like(z[:, None])), dim=-1)
    world = homogeneous @ camera.world_view_transform.to(device=device, dtype=homogeneous.dtype).inverse()
    xyz = world[:, :3] / world[:, 3:].clamp_min(1.0e-8)
    rgb = camera.original_image.detach().to(device=device).float()
    color = rgb[:, ids[:, 0], ids[:, 1]].T.clamp(0.0, 1.0)
    finite = torch.isfinite(xyz).all(-1) & torch.isfinite(color).all(-1)
    if not bool(finite.any()):
        return None
    return xyz[finite].detach().cpu(), color[finite].detach().cpu()


def _voxel_downsample(points, colors, voxel_size):
    xyz = torch.as_tensor(points).float()
    rgb = torch.as_tensor(colors).float().clamp(0.0, 1.0)
    if xyz.numel() == 0 or float(voxel_size) <= 0.0:
        return xyz.numpy(), rgb.numpy()
    lower = xyz.amin(0)
    key = torch.floor((xyz - lower) / float(voxel_size)).long()
    unique, inverse = torch.unique(key, dim=0, return_inverse=True)
    del unique
    count = torch.bincount(inverse).float().clamp_min(1.0)
    out_xyz = torch.zeros((count.numel(), 3), dtype=torch.float32)
    out_rgb = torch.zeros_like(out_xyz)
    out_xyz.index_add_(0, inverse, xyz)
    out_rgb.index_add_(0, inverse, rgb)
    out_xyz = out_xyz / count[:, None]
    out_rgb = out_rgb / count[:, None]
    return out_xyz.numpy(), out_rgb.clamp(0.0, 1.0).numpy()


def _point_cloud_from_depth(cameras, dataset, args):
    chunks_xyz, chunks_rgb = [], []
    for camera in tqdm(cameras, desc="Endpoint depth fusion", unit="camera", leave=False):
        chunk = _backproject_camera_depth(camera, dataset, args)
        if chunk is None:
            continue
        xyz, rgb = chunk
        chunks_xyz.append(xyz)
        chunks_rgb.append(rgb)
    if not chunks_xyz:
        raise RuntimeError("endpoint_init=depth_fusion found no valid endpoint depth pixels")
    xyz = torch.cat(chunks_xyz, dim=0)
    rgb = torch.cat(chunks_rgb, dim=0)
    xyz_np, rgb_np = _voxel_downsample(xyz, rgb, float(getattr(args, "endpoint_depth_voxel_size", 0.01)))
    normals = np.zeros_like(xyz_np, dtype=np.float32)
    return BasicPointCloud(points=xyz_np.astype(np.float32), colors=rgb_np.astype(np.float32), normals=normals)


def _point_cloud_from_canonical(canonical_gaussians):
    xyz = canonical_gaussians.get_canonical_xyz.detach().cpu().float().numpy()
    colors = np.full_like(xyz, 0.5, dtype=np.float32)
    normals = np.zeros_like(xyz, dtype=np.float32)
    return BasicPointCloud(points=xyz.astype(np.float32), colors=colors, normals=normals)


def _clone_static_args(args, *, dynamic_motion=False):
    values = vars(args).copy()
    values["dynamic_motion"] = bool(dynamic_motion)
    return SimpleNamespace(**values)


def _write_rgb_depth_compare(path, camera, rendered):
    target_rgb = camera.original_image.detach().float().cpu().clamp(0.0, 1.0)
    pred_rgb = rendered["render"].detach().float().cpu().clamp(0.0, 1.0)
    rgb_residual = (pred_rgb - target_rgb).abs().mean(0, keepdim=True).repeat(3, 1, 1).clamp(0.0, 1.0)
    if bool(getattr(camera, "depth_reliable", False)):
        target_depth = camera.invdepthmap.detach().float().cpu()
        if target_depth.ndim == 3:
            target_depth = target_depth[0]
        pred_depth = rendered["depth"].detach().float().cpu().squeeze()
        valid = torch.isfinite(target_depth) & torch.isfinite(pred_depth) & (target_depth > 0)
        finite = torch.cat((target_depth[valid], pred_depth[valid])) if bool(valid.any()) else torch.tensor([])
        if finite.numel():
            lo, hi = torch.quantile(finite, torch.tensor([0.02, 0.98]))
        else:
            lo, hi = torch.tensor(0.0), torch.tensor(1.0)
        denom = (hi - lo).clamp_min(1.0e-6)
        target_vis = ((target_depth - lo) / denom).clamp(0.0, 1.0)[None].repeat(3, 1, 1)
        pred_vis = ((pred_depth - lo) / denom).clamp(0.0, 1.0)[None].repeat(3, 1, 1)
        depth_res = ((pred_depth - target_depth).abs() / denom).clamp(0.0, 1.0)[None].repeat(3, 1, 1)
    else:
        h, w = target_rgb.shape[-2:]
        target_vis = torch.zeros((3, h, w))
        pred_vis = torch.zeros_like(target_vis)
        depth_res = torch.zeros_like(target_vis)
    grid = torch.cat(
        (
            torch.cat((target_rgb, pred_rgb, rgb_residual), dim=-1),
            torch.cat((target_vis, pred_vis, depth_res), dim=-1),
        ),
        dim=-2,
    )
    image = (255.0 * grid.permute(1, 2, 0).numpy()).round().astype(np.uint8)
    Image.fromarray(image).save(path)


def _feature_rgb(feature):
    value = torch.as_tensor(feature).detach().float().cpu()
    if value.ndim == 3:
        value = value[:3]
    if value.shape[0] < 3:
        value = F.pad(value, (0, 0, 0, 0, 0, 3 - value.shape[0]))
    value = value[:3]
    flat = value.reshape(3, -1)
    lo = flat.quantile(0.02, dim=1)[:, None, None]
    hi = flat.quantile(0.98, dim=1)[:, None, None]
    return ((value - lo) / (hi - lo).clamp_min(1.0e-6)).clamp(0.0, 1.0)


def _write_sam3_compare(path, camera, gaussians, pipe, dataset, args):
    feature = gaussians.get_sam3_feature_field
    if not torch.is_tensor(feature) or feature.numel() == 0:
        return
    target = _load_sam3_target_for_view(
        camera,
        dataset,
        args,
        (int(camera.original_image.shape[-2]), int(camera.original_image.shape[-1])),
        device=feature.device,
    )
    if target is None:
        return
    k = min(feature.shape[1], target.shape[0])
    rendered = render_sam3_feature_field(
        camera,
        gaussians,
        pipe,
        features=feature[:, :k],
        detach_geometry=True,
    )["render"]
    alpha = render_feature_field(
        camera,
        gaussians,
        pipe,
        torch.ones((feature.shape[0], 1), device=feature.device, dtype=feature.dtype),
        bg_feature=torch.zeros(1, device=feature.device),
        detach_geometry=True,
    )["render"][0]
    pred = F.normalize(rendered / alpha.clamp_min(1.0e-4)[None], dim=0)
    target_n = F.normalize(target[:k], dim=0)
    residual = (1.0 - (pred * target_n).sum(0).clamp(-1.0, 1.0))[None].repeat(3, 1, 1)
    grid = torch.cat((_feature_rgb(target_n), _feature_rgb(pred), residual.clamp(0.0, 1.0)), dim=-1)
    image = (255.0 * grid.permute(1, 2, 0).cpu().numpy()).round().astype(np.uint8)
    Image.fromarray(image).save(path)


@torch.no_grad()
def _estimate_normals(points, k=12, max_points=15000):
    xyz = torch.as_tensor(points).detach().cpu().float()
    if xyz.shape[0] < 4 or xyz.shape[0] > int(max_points):
        return None
    normals = torch.zeros_like(xyz)
    for start in range(0, xyz.shape[0], 512):
        chunk = xyz[start : start + 512]
        dist = torch.cdist(chunk, xyz)
        ids = dist.topk(min(int(k) + 1, xyz.shape[0]), largest=False).indices[:, 1:]
        neigh = xyz[ids] - chunk[:, None]
        cov = torch.einsum("nki,nkj->nij", neigh, neigh) / max(1, ids.shape[1])
        eigvec = torch.linalg.eigh(cov).eigenvectors[:, :, 0]
        normals[start : start + chunk.shape[0]] = F.normalize(eigvec, dim=-1)
    return normals


def _load_endpoint_model(ply_path, dataset, opt, args):
    del args
    model = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    model.load_ply(str(ply_path), dataset.train_test_exp)
    return model


def prepare_endpoint_static_model(
    scene,
    canonical_gaussians,
    pipe,
    background,
    dataset,
    args,
    opt,
    *,
    static_visual_loss_fn,
):
    dynamic_cameras = scene.getTrainCameras()
    frame_idx = resolve_endpoint_frame(dynamic_cameras, int(getattr(args, "endpoint_frame", -1)))
    endpoint_cameras = build_static_snapshot_cameras(dynamic_cameras, frame_idx)
    root = _endpoint_root(scene.model_path, args, frame_idx)
    preferred_iteration = int(getattr(args, "endpoint_static_iters", 0))
    existing_ply = _find_model_ply(root, preferred_iteration=preferred_iteration)
    summary_path = root / "endpoint_summary.json"
    if existing_ply is not None:
        endpoint = _load_endpoint_model(existing_ply, dataset, opt, args)
        if (
            bool(getattr(args, "use_sam3_feature_field", False))
            and float(getattr(args, "endpoint_sam3_weight", 0.0)) > 0.0
            and (
                not torch.is_tensor(endpoint.get_sam3_feature_field)
                or endpoint.get_sam3_feature_field.numel() == 0
            )
        ):
            raise RuntimeError(
                f"Existing endpoint model {existing_ply} has no sam3_feature_field.pt. "
                "Build/train an endpoint model with --use_sam3_feature_field before enabling --endpoint_sam3_weight."
            )
        summary = {
            "schema_version": 1,
            "frame": frame_idx,
            "model_path": str(root),
            "point_cloud": str(existing_ply),
            "loaded_existing": True,
            "ground_truth_used": False,
        }
        root.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        if not bool(getattr(args, "endpoint_build_if_missing", False)):
            raise FileNotFoundError(
                f"Endpoint static model missing at {root}. Pass --endpoint_build_if_missing to build it."
            )
        mkdir_p(str(root))
        init = str(getattr(args, "endpoint_init", "depth_fusion"))
        if init == "depth_fusion":
            pcd = _point_cloud_from_depth(endpoint_cameras, dataset, args)
        elif init == "canonical":
            pcd = _point_cloud_from_canonical(canonical_gaussians)
        else:
            colmap_ply = _find_model_ply(root)
            if colmap_ply is None:
                raise FileNotFoundError(
                    "--endpoint_init colmap requires --endpoint_model_path containing a point_cloud.ply"
                )
            endpoint = _load_endpoint_model(colmap_ply, dataset, opt, args)
            pcd = None
        if init != "colmap":
            endpoint = GaussianModel(dataset.sh_degree, opt.optimizer_type)
            endpoint.create_from_pcd(pcd, endpoint_cameras, scene.cameras_extent)
        endpoint.init_semantic_and_feature_fields(
            bool(getattr(args, "use_semantic_feature_field", False)),
            0,
            sam3_dim=int(args.sam3_feature_dim) if bool(getattr(args, "use_sam3_feature_field", False)) else 0,
        )
        opt_endpoint = SimpleNamespace(**vars(opt))
        opt_endpoint.semantic_feature_lr = float(getattr(args, "semantic_feature_lr", 1.0e-3))
        opt_endpoint.dino_feature_lr = float(getattr(args, "dino_feature_lr", 1.0e-3))
        opt_endpoint.sam3_feature_lr = float(getattr(args, "sam3_feature_lr", 1.0e-3))
        endpoint.training_setup(opt_endpoint)
        train_args = _clone_static_args(args, dynamic_motion=False)
        depth_schedule = get_expon_lr_func(
            opt.depth_l1_weight_init,
            opt.depth_l1_weight_final,
            max_steps=max(1, preferred_iteration),
        )
        stack = []
        progress = tqdm(range(1, preferred_iteration + 1), desc=f"Endpoint static frame {frame_idx}", unit="iter")
        for iteration in progress:
            endpoint.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                endpoint.oneupSHdegree()
            if not stack:
                stack = list(endpoint_cameras)
            camera = stack.pop(randint(0, len(stack) - 1))
            total, stats, package = static_visual_loss_fn(
                camera,
                endpoint,
                pipe,
                background,
                dataset,
                train_args,
                opt_endpoint,
                depth_schedule(iteration),
            )
            total.backward()
            endpoint.optimizer.step()
            endpoint.optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{float(total.detach()):.4g}")
        iteration_dir = root / "point_cloud" / f"iteration_{preferred_iteration}"
        ply = iteration_dir / "point_cloud.ply"
        endpoint.save_ply(str(ply))
        summary = {
            "schema_version": 1,
            "frame": frame_idx,
            "model_path": str(root),
            "point_cloud": str(ply),
            "loaded_existing": False,
            "init": init,
            "iterations": preferred_iteration,
            "camera_count": len(endpoint_cameras),
            "gaussian_count": int(endpoint.get_xyz.shape[0]),
            "ground_truth_used": False,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with torch.no_grad():
            for camera in endpoint_cameras:
                rendered = render(
                    camera,
                    endpoint,
                    pipe,
                    background,
                    use_trained_exp=bool(dataset.train_test_exp),
                )
                cam_name = str(getattr(camera, "dynamic_camera_id", Path(getattr(camera, "image_name", "camera")).stem))
                _write_rgb_depth_compare(
                    root / f"endpoint_static_rgb_depth_compare_{cam_name}.png",
                    camera,
                    rendered,
                )
                _write_sam3_compare(
                    root / f"sam3_endpoint_{cam_name}.png",
                    camera,
                    endpoint,
                    pipe,
                    dataset,
                    train_args,
                )
    normals = None
    if (
        str(getattr(args, "endpoint_geometry_metric", "point")) == "point_plane"
        or float(getattr(args, "endpoint_normal_weight", 0.0)) > 0.0
    ):
        normals = _estimate_normals(endpoint.get_xyz.detach().cpu())
    state = {
        "schema_version": 1,
        "frame": frame_idx,
        "cameras": endpoint_cameras,
        "model": endpoint,
        "model_path": str(root),
        "summary_path": str(summary_path),
        "point_cloud_path": summary.get("point_cloud", str(existing_ply) if existing_ply else ""),
        "xyz": endpoint.get_xyz.detach(),
        "normals": None if normals is None else normals,
        "sam3_features": (
            endpoint.get_sam3_feature_field.detach()
            if torch.is_tensor(endpoint.get_sam3_feature_field)
            and endpoint.get_sam3_feature_field.numel() > 0
            else None
        ),
        "summary": summary,
    }
    canonical_gaussians.endpoint_static_state = state
    return state
