#!/usr/bin/env python3
"""Compare static/dynamic priors before the shared rigid Killing solver."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.dynamic_prior import (  # noqa: E402
    DynamicPrior,
    load_dynamic_prior,
    make_all_foreground_prior,
    make_all_gaussians_prior,
    make_gt_prior,
    prior_metrics,
    save_dynamic_prior,
)
from utils.endpoint_flow_matching import (  # noqa: E402
    EndpointFlowMLP,
    FlowMatchingConfig,
    build_dynamic_prior_from_flow,
    flow_checkpoint_payload,
    integrate_flow,
    train_endpoint_flow_matching,
)
from utils.endpoint_transport_prior import (  # noqa: E402
    EndpointTransportConfig,
    EndpointTransportPrior,
    EndpointTransportPriorResult,
    save_endpoint_transport_result,
)
from utils.sam3_gaussian_affinity import _root_has_mask_cache  # noqa: E402
from utils.static_endpoint_scene import (  # noqa: E402
    build_labeled_frame_static_scene,
    has_inherited_gt_labels,
    has_sparse_gt_parts,
)


def _split_methods(text: str) -> list[str]:
    aliases = {
        "endpoint_ot": "endpoint_transport_debug",
        "endpoint_transport": "endpoint_transport_debug",
        "closed_loop_rgbd": "strict_closed_loop_rgbd",
        "strict_rgbd": "strict_closed_loop_rgbd",
    }
    allowed = {
        "baseline",
        "all_fg",
        "gt",
        "denoise",
        "flow_matching",
        "endpoint_transport_debug",
        "strict_closed_loop_rgbd",
    }
    methods = []
    def add_method(name: str) -> None:
        if name not in methods:
            methods.append(name)

    for item in str(text).replace(",", " ").split():
        item = item.strip()
        if item == "all":
            for name in ("baseline", "denoise", "flow_matching"):
                add_method(name)
        elif item:
            item = aliases.get(item, item)
            if item not in allowed:
                raise ValueError(
                    f"Unknown comparison method {item!r}. "
                    f"Allowed: {', '.join(sorted(allowed))}"
                )
            add_method(item)
    return methods or ["baseline", "denoise", "flow_matching"]


def _ply_from_model_dir(model_dir: str | Path, iteration: int) -> Path:
    return Path(model_dir).expanduser().resolve() / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"


def _load_ported_gt_mask(static_ply: str | Path, moving_part: int, expected_count: int) -> torch.Tensor | None:
    ply = Path(static_ply).expanduser().resolve()
    sidecar = ply.parent / "gt_parts.pt"
    if sidecar.is_file():
        obj = torch.load(sidecar, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            source = str(obj.get("source", "unknown"))
            if "gt_moving_mask" in obj:
                mask = torch.as_tensor(obj["gt_moving_mask"]).detach().cpu().bool().reshape(-1)
            elif "gt_part_labels" in obj:
                mask = torch.as_tensor(obj["gt_part_labels"]).detach().cpu().long().reshape(-1) == int(moving_part)
            else:
                mask = None
            if mask is not None and mask.numel() == int(expected_count):
                if source != "gaussian_model_inherited_sparse_parts":
                    print(
                        "[comparison] WARNING: GT loaded from a post-hoc/legacy "
                        f"sidecar ({source}); retrain static 3DGS for exact lineage labels: {sidecar}"
                    )
                else:
                    print(f"[comparison] GT loaded from inherited static 3DGS sidecar: {sidecar}")
                return mask
    try:
        from plyfile import PlyData

        vertex = PlyData.read(str(ply))["vertex"].data
        names = set(vertex.dtype.names or ())
        if "gt_dynamic" in names:
            mask = torch.as_tensor(vertex["gt_dynamic"].astype("bool")).detach().cpu().bool().reshape(-1)
        elif "gt_part" in names:
            mask = torch.as_tensor(vertex["gt_part"].astype("int64")).detach().cpu().long().reshape(-1) == int(moving_part)
        else:
            mask = None
        if mask is not None and mask.numel() == int(expected_count):
            print(f"[comparison] GT loaded from static 3DGS PLY fields: {ply}")
            return mask
    except Exception as exc:
        print(f"[comparison] static 3DGS GT read skipped: {exc}")
    return None


def _ply_or_sidecar_has_field(ply: str | Path, *, prefix: str, sidecar: str, sidecar_keys: tuple[str, ...]) -> bool:
    ply = Path(ply).expanduser().resolve()
    try:
        from plyfile import PlyData

        props = [p.name for p in PlyData.read(str(ply)).elements[0].properties]
        if any(name.startswith(prefix) for name in props):
            return True
    except Exception:
        pass

    sidecar_path = ply.parent / sidecar
    if not sidecar_path.is_file():
        return False
    try:
        obj = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            value = None
            for key in sidecar_keys:
                if key in obj:
                    value = obj[key]
                    break
        else:
            value = obj
        tensor = torch.as_tensor(value)
        return tensor.numel() > 0
    except Exception:
        return False


def _ply_has_dino_field(ply: str | Path) -> bool:
    return _ply_or_sidecar_has_field(
        ply,
        prefix="dino_",
        sidecar="dino_feature_field.pt",
        sidecar_keys=("dino_feature_field", "features"),
    )


def _ply_has_sam3_field(ply: str | Path) -> bool:
    return _ply_or_sidecar_has_field(
        ply,
        prefix="sam3_feature_",
        sidecar="sam3_feature_field.pt",
        sidecar_keys=("sam3_feature_field", "features"),
    )


def _input_scene_dir(root_scene: str | Path) -> Path:
    root = Path(root_scene).expanduser().resolve()
    if (root / "rgb").is_dir() or (root / "pointcloud").is_dir():
        return root
    if root.name == "colmap" and ((root.parent / "rgb").is_dir() or (root.parent / "pointcloud").is_dir()):
        return root.parent
    return REPO / "input" / root.name


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("[comparison] running:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def _has_frame_cache(root: str | Path, frames: set[int] | None = None) -> bool:
    path = Path(root).expanduser()
    if not path.exists():
        return False
    paths = list(path.rglob("frame_*.pt"))
    if not paths:
        return False
    if frames is None:
        return True
    seen: set[int] = set()
    for item in paths:
        try:
            seen.add(int(item.stem.split("_")[-1]))
        except ValueError:
            continue
    return set(int(x) for x in frames).issubset(seen)


def _required_sam3_frame_ids(args) -> set[int]:
    return {0, int(_resolve_frame_t_idx(args))}


def _canonical_sam3_mask_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    return root.with_name("sam3_masks") if root.name == "sam3" else root


def _sam3_env(args) -> dict[str, str]:
    env = os.environ.copy()
    if not bool(args.sam3_allow_download):
        env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def _resolve_aux_paths(args, output: Path) -> None:
    if not str(args.endpoint_model_dir).strip():
        args.endpoint_model_dir = str(output / "shared" / "endpoint_static")
    if not str(args.denoise_output_dir).strip():
        args.denoise_output_dir = str(output / "denoise_discovery")
    input_dir = _input_scene_dir(args.root_scene)
    if not str(args.sam3_feature_root).strip():
        args.sam3_feature_root = str(input_dir / "sam3")
    if not str(args.sam3_mask_root).strip():
        args.sam3_mask_root = str(input_dir / "sam3_masks")
    else:
        args.sam3_mask_root = str(_canonical_sam3_mask_root(args.sam3_mask_root))


def _sam3_frame_args(args) -> list[str]:
    endpoint = str(int(args.endpoint_frame)) if int(args.endpoint_frame) >= 0 else "LAST"
    return ["0", endpoint]


def _ensure_sam3_caches(args) -> None:
    required_frames = _required_sam3_frame_ids(args)
    if bool(args.ensure_sam3_features) and not _has_frame_cache(args.sam3_feature_root, required_frames):
        cmd = [
            sys.executable,
            str(REPO / "tools" / "extract_sam3_features.py"),
            "--root",
            str(Path(args.root_scene).expanduser().resolve()),
            "--frames",
            *_sam3_frame_args(args),
            "--output",
            str(Path(args.sam3_feature_root).expanduser().resolve()),
            "--device",
            str(args.sam3_device),
            "--feature-dim",
            str(int(args.sam3_feature_dim)),
            "--resolution",
            str(int(args.sam3_resolution)),
        ]
        if args.sam3_checkpoint:
            cmd.extend(["--checkpoint", str(Path(args.sam3_checkpoint).expanduser().resolve())])
        if bool(args.sam3_allow_download):
            cmd.append("--allow-download")
        _run(cmd, env=_sam3_env(args))

    mask_root = Path(args.sam3_mask_root).expanduser()
    if bool(args.ensure_sam3_masks) and not (
        _root_has_mask_cache(mask_root) and _has_frame_cache(mask_root, required_frames)
    ):
        cmd = [
            sys.executable,
            str(REPO / "tools" / "extract_sam3_masks.py"),
            "--root",
            str(Path(args.root_scene).expanduser().resolve()),
            "--frames",
            *_sam3_frame_args(args),
            "--output",
            str(_canonical_sam3_mask_root(args.sam3_mask_root).resolve()),
            "--device",
            str(args.sam3_device),
            "--resolution",
            str(int(args.sam3_resolution)),
        ]
        if args.sam3_checkpoint:
            cmd.extend(["--checkpoint", str(Path(args.sam3_checkpoint).expanduser().resolve())])
        if bool(args.sam3_allow_download):
            cmd.append("--allow-download")
        if bool(args.sam3_mask_debug_overlays):
            cmd.append("--debug-overlays")
        _run(cmd, env=_sam3_env(args))


def _ensure_static_model(args) -> Path:
    args.static_model_dir = str(Path(args.static_model_dir).expanduser().resolve())
    ply = _ply_from_model_dir(args.static_model_dir, args.static_iteration)
    if ply.is_file() and not bool(args.force_static_train):
        if has_sparse_gt_parts(_input_scene_dir(args.root_scene)) and not has_inherited_gt_labels(ply):
            print(
                f"[comparison] existing frame-0 static model lacks inherited GT labels; rebuilding: {ply}"
            )
        else:
            if bool(args.train_sam3_feature_field) and not _ply_or_sidecar_has_field(
                ply,
                prefix="sam3_feature_",
                sidecar="sam3_feature_field.pt",
                sidecar_keys=("sam3_feature_field", "features"),
            ):
                raise RuntimeError(
                    f"Existing static model has no dense SAM3 feature field: {ply}. "
                    "Use --force-static-train to rebuild it with --train-sam3-feature-field, "
                    "or omit --train-sam3-feature-field for this comparison."
                )
            return ply
    if not bool(args.build_static_if_missing) and not bool(args.force_static_train):
        raise FileNotFoundError(f"Static model missing or stale without inherited GT labels: {ply}")
    cmd = [
        sys.executable,
        str(REPO / "train.py"),
        "-s",
        str(Path(args.root_scene).expanduser().resolve()),
        "-m",
        args.static_model_dir,
        "--iterations",
        str(int(args.static_iteration)),
        "--static_train_iters",
        str(int(args.static_iteration)),
        "--use_semantic_feature_field",
        "--disable_viewer",
        "--no_wandb",
    ]
    dino_root = _input_scene_dir(args.root_scene) / "dinov3"
    if _has_frame_cache(dino_root, {0}):
        cmd.extend(["--use_dino_feature_field", "--dinov3_root", str(dino_root)])
    if bool(args.train_sam3_feature_field):
        cmd.extend(
            [
                "--use_sam3_feature_field",
                "--sam3_feature_root",
                str(Path(args.sam3_feature_root).expanduser().resolve()),
                "--sam3_feature_dim",
                str(int(args.sam3_feature_dim)),
            ]
        )
    _run(cmd)
    if not ply.is_file():
        raise FileNotFoundError(f"Static training finished but did not write {ply}")
    if has_sparse_gt_parts(_input_scene_dir(args.root_scene)) and not has_inherited_gt_labels(ply):
        raise RuntimeError(f"Static training wrote a model without inherited GT labels: {ply}")
    return ply


def _resolve_frame_t_idx(args) -> int:
    """Resolve --endpoint-frame -1 to an absolute frame index."""
    idx = int(args.endpoint_frame)
    if idx >= 0:
        return idx
    # Negative: count from end — read num_frames from Sapien metadata.
    scene_root = Path(args.root_scene).expanduser().resolve()
    meta_yaml = scene_root / "metadata" / "job_config_resolved.yaml"
    if meta_yaml.exists():
        import yaml
        with open(meta_yaml) as f:
            cfg = yaml.safe_load(f)
        num_frames = int(cfg.get("num_frames", 0))
        if num_frames > 0:
            return num_frames + idx
    # Fallback: count frames in the input rgb directory.
    input_root = REPO / "input" / scene_root.name
    rgb_cams = sorted((input_root / "rgb").iterdir()) if (input_root / "rgb").is_dir() else []
    if rgb_cams:
        frame_ids: list[int] = []
        for path in rgb_cams[0].glob("frame_*.*"):
            suffix = path.suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg"}:
                continue
            token = path.stem.rsplit("_", 1)[-1]
            if token.isdigit():
                frame_ids.append(int(token))
        frame_ids = sorted(set(frame_ids))
        if frame_ids:
            try:
                return frame_ids[idx]
            except IndexError as exc:
                raise RuntimeError(
                    f"Cannot resolve endpoint_frame={idx}: only {len(frame_ids)} rgb frames found."
                ) from exc
    raise RuntimeError(
        f"Cannot resolve endpoint_frame={idx}: no Sapien metadata or input rgb dir found."
    )


def _find_rgb_frame(rgb_root: Path, cam_name: str, frame_str: str) -> Path | None:
    cam_dir = rgb_root / cam_name
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        path = cam_dir / f"{frame_str}{ext}"
        if path.is_file():
            return path
    matches = sorted(
        p for p in cam_dir.glob(f"{frame_str}.*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    return matches[0] if matches else None


def _build_frame_t_colmap(
    scene_root: Path,
    frame_t_idx: int,
    source_colmap_dir: Path,
    output_colmap_dir: Path,
    *,
    sample_stride: int = 8,
    max_points_per_image: int = 1000,
) -> None:
    """Build a mock COLMAP for frame T without re-running SAPIEN.

    Reuses camera intrinsics/extrinsics from the frame-0 COLMAP (cameras are
    fixed), backprojects frame-T depth maps from input/depth_npy/ to produce 3D
    points, and writes cameras/images/points3D to output_colmap_dir/sparse/0/.
    Image files are symlinked from input/rgb/.
    """
    import numpy as np
    from utils.read_write_model import (
        read_model,
        write_model,
        Image,
        Point3D,
        qvec2rotmat,
    )

    cams, frame0_imgs, _ = read_model(str(source_colmap_dir / "sparse" / "0"), ext=".bin")

    frame_str = f"frame_{frame_t_idx:06d}"
    # Derive input directories relative to the Sapien render root.  The
    # input/USB_XXXXX/ mirror is tried first; the Sapien output dir is the
    # fallback (depth and rgb live in both places for this dataset).
    input_root = REPO / "input" / scene_root.name
    depth_root = input_root / "depth_npy" if (input_root / "depth_npy").is_dir() else scene_root / "depth_npy"
    rgb_root = input_root / "rgb" if (input_root / "rgb").is_dir() else scene_root / "rgb"

    # Map cam_name -> frame-0 Image (to copy extrinsics).
    cam_name_to_img0: dict[str, Any] = {}
    for img in frame0_imgs.values():
        cam_name = img.name.split("/")[0]
        cam_name_to_img0[cam_name] = img

    new_images: dict[int, Image] = {}
    new_points3D: dict[int, Point3D] = {}
    next_image_id = 1
    next_point_id = 1

    for cam_name in sorted(cam_name_to_img0.keys()):
        img0 = cam_name_to_img0[cam_name]
        cam = cams[img0.camera_id]
        fx, fy, cx, cy = float(cam.params[0]), float(cam.params[1]), float(cam.params[2]), float(cam.params[3])
        h, w = int(cam.height), int(cam.width)
        R = qvec2rotmat(img0.qvec)  # world_to_cam rotation
        t = img0.tvec

        depth_path = depth_root / cam_name / f"{frame_str}.npy"
        if not depth_path.exists():
            print(f"[frame_t_colmap] depth missing for {cam_name}/{frame_str}, skipping")
            continue
        rgb_path = _find_rgb_frame(rgb_root, cam_name, frame_str)
        if rgb_path is None:
            print(f"[frame_t_colmap] rgb missing for {cam_name}/{frame_str}, skipping")
            continue
        depth = np.load(str(depth_path)).astype(np.float32)
        if depth.shape != (h, w):
            import cv2
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        valid = np.isfinite(depth) & (depth > 0.0)
        grid = np.zeros_like(valid, dtype=bool)
        grid[::sample_stride, ::sample_stride] = True
        ys, xs = np.nonzero(valid & grid)
        if max_points_per_image > 0 and len(ys) > max_points_per_image:
            keep = np.linspace(0, len(ys) - 1, max_points_per_image, dtype=int)
            ys, xs = ys[keep], xs[keep]

        d = depth[ys, xs].astype(np.float64)
        pts_cam = np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], axis=1)
        pts_world = (R.T @ (pts_cam - t).T).T  # [N,3]

        colors = np.full((len(ys), 3), 128, dtype=np.uint8)
        try:
            import imageio.v2 as imageio
            rgb_img = imageio.imread(str(rgb_path))
            if rgb_img.shape[:2] != (h, w):
                import cv2
                rgb_img = cv2.resize(rgb_img, (w, h), interpolation=cv2.INTER_LINEAR)
            if rgb_img.ndim == 2:
                colors = np.repeat(rgb_img[ys, xs, None], 3, axis=1).astype(np.uint8)
            elif rgb_img.ndim == 3 and rgb_img.shape[2] >= 3:
                colors = rgb_img[ys, xs, :3].astype(np.uint8)
        except Exception as exc:
            print(f"[frame_t_colmap] rgb read failed for {rgb_path}: {exc}; using gray fallback")

        image_id = next_image_id
        next_image_id += 1
        image_rel = f"{cam_name}/{rgb_path.name}"
        point3D_ids = np.full(len(ys), -1, dtype=np.int64)
        xys_arr = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

        for k in range(len(ys)):
            pid = next_point_id
            next_point_id += 1
            new_points3D[pid] = Point3D(
                id=pid,
                xyz=pts_world[k],
                rgb=colors[k],
                error=0.0,
                image_ids=np.array([image_id], dtype=np.int32),
                point2D_idxs=np.array([k], dtype=np.int32),
            )
            point3D_ids[k] = pid

        new_images[image_id] = Image(
            id=image_id,
            qvec=img0.qvec.copy(),
            tvec=img0.tvec.copy(),
            camera_id=img0.camera_id,
            name=image_rel,
            xys=xys_arr,
            point3D_ids=point3D_ids,
        )

    if not new_images:
        raise RuntimeError(f"No valid depth frames found for frame {frame_t_idx} — cannot build frame-T COLMAP")

    sparse_dir = output_colmap_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    write_model(cams, new_images, new_points3D, str(sparse_dir), ext=".bin")
    write_model(cams, new_images, new_points3D, str(sparse_dir), ext=".txt")

    # Symlink frame-T RGB images.
    images_dir = output_colmap_dir / "images"
    for img in new_images.values():
        parts = img.name.split("/")
        dest = images_dir / parts[0] / parts[1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            src = rgb_root / parts[0] / parts[1]
            if src.exists():
                import os as _os
                dest.symlink_to(_os.path.relpath(src, dest.parent))

    print(
        f"[frame_t_colmap] wrote {len(new_images)} cameras, "
        f"{len(new_points3D)} points to {output_colmap_dir}"
    )


def _frame_t_colmap_is_valid(colmap_dir: Path, rgb_root: Path) -> bool:
    sparse_dir = colmap_dir / "sparse" / "0"
    if not (sparse_dir / "cameras.bin").is_file() or not (sparse_dir / "images.bin").is_file():
        return False
    try:
        import numpy as np
        from utils.read_write_model import read_model

        _, images, points = read_model(str(sparse_dir), ext=".bin")
    except Exception:
        return False
    if not images:
        return False
    for image in images.values():
        path = rgb_root / image.name
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not path.is_file():
            return False
    if points:
        rgb = np.stack([p.rgb for p in points.values()])
        unique = np.unique(rgb, axis=0)
        if unique.shape[0] == 1 and unique[0].tolist() == [128, 128, 128]:
            return False
    return True


def _ensure_endpoint_model(args, output: Path) -> Path:
    """Build the frame-T static 3DGS the same way as frame-0: mock COLMAP + full training."""
    args.endpoint_model_dir = str(
        Path(args.endpoint_model_dir).expanduser().resolve()
        if args.endpoint_model_dir
        else output / "shared" / "frame_t_static"
    )
    explicit_ply = Path(args.endpoint_ply).expanduser().resolve() if args.endpoint_ply else None
    if explicit_ply is not None and explicit_ply.is_file() and not bool(args.force_endpoint_train):
        input_root = _input_scene_dir(args.root_scene)
        if has_sparse_gt_parts(input_root) and not has_inherited_gt_labels(explicit_ply):
            raise RuntimeError(
                f"Explicit endpoint PLY lacks inherited GT labels: {explicit_ply}"
            )
        args.endpoint_ply = str(explicit_ply)
        return explicit_ply
    ply = _ply_from_model_dir(args.endpoint_model_dir, args.endpoint_iteration)
    if ply.is_file() and not bool(args.force_endpoint_train):
        input_root = _input_scene_dir(args.root_scene)
        if has_sparse_gt_parts(input_root) and not has_inherited_gt_labels(ply):
            print(
                f"[comparison] existing endpoint static model lacks inherited GT labels; rebuilding: {ply}"
            )
        else:
            if bool(args.train_sam3_feature_field) and not _ply_or_sidecar_has_field(
                ply,
                prefix="sam3_feature_",
                sidecar="sam3_feature_field.pt",
                sidecar_keys=("sam3_feature_field", "features"),
            ):
                raise RuntimeError(
                    f"Existing endpoint model has no dense SAM3 feature field: {ply}. "
                    "Use --force-endpoint-train to rebuild it with SAM3, or omit "
                    "--train-sam3-feature-field for this comparison."
                )
            args.endpoint_ply = str(ply)
            return ply
    if not bool(args.build_endpoint_if_missing) and not bool(args.force_endpoint_train):
        raise FileNotFoundError(f"Frame-T model missing or stale without inherited GT labels: {ply}")

    frame_t_idx = _resolve_frame_t_idx(args)
    input_root = _input_scene_dir(args.root_scene)
    frame_t_scene = output / "shared" / "frame_t_scene"
    build_labeled_frame_static_scene(
        input_root,
        frame_t_scene,
        int(frame_t_idx),
        force=bool(args.force_endpoint_train),
    )

    # --- Train full static 3DGS on frame T (semantic + DINO + SAM3) ---
    cmd = [
        sys.executable,
        str(REPO / "train.py"),
        "-s", str(frame_t_scene),
        "-m", args.endpoint_model_dir,
        "--iterations", str(int(args.endpoint_iteration)),
        "--static_train_iters", str(int(args.endpoint_iteration)),
        "--use_semantic_feature_field",
        "--disable_viewer",
        "--no_wandb",
    ]

    # DINO: include if frame-T features exist in input/dinov3/.
    dino_root = input_root / "dinov3"
    dino_probe = dino_root / "cam_000" / f"frame_{frame_t_idx:06d}.pt"
    if dino_probe.exists():
        cmd.extend(["--use_dino_feature_field", "--dinov3_root", str(dino_root)])

    # SAM3 dense features.
    sam3_root = (
        Path(args.sam3_feature_root).expanduser().resolve()
        if args.sam3_feature_root
        else input_root / "sam3"
    )
    sam3_probe = sam3_root / "cam_000" / f"frame_{frame_t_idx:06d}.pt"
    if sam3_probe.exists():
        cmd.extend([
            "--use_sam3_feature_field",
            "--sam3_feature_root", str(sam3_root),
            "--sam3_feature_dim", str(int(args.sam3_feature_dim)),
        ])

    # SAM3 affinity (trains inside train.py on frame-T masks).
    if bool(args.ensure_sam3_affinity):
        sam3_mask_root = (
            Path(args.sam3_mask_root).expanduser().resolve()
            if args.sam3_mask_root
            else input_root / "sam3_masks"
        )
        sam3_mask_probe = sam3_mask_root / "cam_000" / f"frame_{frame_t_idx:06d}.pt"
        if sam3_mask_probe.exists():
            cmd.extend([
                "--use_sam3_affinity_field",
                "--sam3_mask_root", str(sam3_mask_root),
                "--sam3_affinity_iters", str(int(args.sam3_affinity_iters)),
                "--sam3_affinity_dim", str(int(args.sam3_affinity_dim)),
            ])

    _run(cmd)
    if not ply.is_file():
        raise FileNotFoundError(f"Frame-T training finished but did not write {ply}")
    if has_sparse_gt_parts(input_root) and not has_inherited_gt_labels(ply):
        raise RuntimeError(f"Frame-T training wrote a model without inherited GT labels: {ply}")
    args.endpoint_ply = str(ply)
    return ply


def _ensure_canonical_sam3_affinity(args) -> None:
    if not bool(args.ensure_sam3_affinity):
        return
    cmd = [
        sys.executable,
        str(REPO / "tools" / "train_sam3_affinity_field.py"),
        "--root-scene",
        str(Path(args.root_scene).expanduser().resolve()),
        "--static-model-dir",
        str(Path(args.static_model_dir).expanduser().resolve()),
        "--output-root",
        str(Path(args.static_model_dir).expanduser().resolve()),
        "--iteration",
        str(int(args.static_iteration)),
        "--sam3-mask-root",
        str(Path(args.sam3_mask_root).expanduser().resolve()),
        "--sam3-affinity-iters",
        str(int(args.sam3_affinity_iters)),
        "--sam3-affinity-dim",
        str(int(args.sam3_affinity_dim)),
        "--",
        "--no_wandb",
    ]
    _run(cmd)


def _load_gaussian_ply(path: str | Path, sh_degree: int):
    from scene import GaussianModel

    ply = Path(path).expanduser().resolve()
    if not ply.is_file():
        raise FileNotFoundError(f"Gaussian PLY does not exist: {ply}")
    model = GaussianModel(int(sh_degree))
    model.load_ply(str(ply))
    return model


def _foreground_ids(model, threshold: float, min_points: int) -> torch.Tensor:
    prob = model.get_semantic_probability
    if prob is None or prob.numel() == 0:
        return torch.arange(model.get_xyz.shape[0], dtype=torch.long)
    mask = prob.detach().reshape(-1).cpu() > float(threshold)
    if int(mask.sum()) < int(min_points):
        raise RuntimeError(
            f"Only {int(mask.sum())} semantic foreground Gaussians at threshold {threshold}; "
            f"need at least {int(min_points)}"
        )
    return torch.nonzero(mask, as_tuple=False).reshape(-1).long()


def _tensor_for_ids(value: Any, ids: torch.Tensor, full_count: int, default: float) -> torch.Tensor:
    if value is None:
        return torch.full((ids.numel(),), float(default), dtype=torch.float32)
    tensor = torch.as_tensor(value).detach().cpu().float()
    if tensor.ndim > 1 and tensor.shape[-1] in (full_count, ids.numel()):
        tensor = tensor.reshape(-1, tensor.shape[-1]).mean(dim=0)
    tensor = tensor.reshape(-1)
    if tensor.numel() == full_count:
        return tensor.index_select(0, ids)
    if tensor.numel() == ids.numel():
        return tensor
    return torch.full((ids.numel(),), float(default), dtype=torch.float32)


def _temporal_motion_ratio_for_ids(value: Any, ids: torch.Tensor, full_count: int, default: float) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().float()
    if tensor.ndim <= 1:
        return _tensor_for_ids(tensor, ids, full_count, default)
    tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.shape[-1] == full_count:
        tensor = tensor.index_select(1, ids)
    elif tensor.shape[-1] != ids.numel():
        return torch.full((ids.numel(),), float(default), dtype=torch.float32)
    finite = torch.isfinite(tensor) & (tensor > 0)
    safe = torch.where(finite, tensor, torch.ones_like(tensor))
    counts = finite.float().sum(dim=0).clamp_min(1.0)
    geom = torch.exp(torch.log(safe.clamp_min(1.0e-8)).sum(dim=0) / counts)
    return torch.where(finite.any(dim=0), geom, torch.full_like(geom, float(default)))


def _denoise_score_for_ids(payload: dict[str, Any], gaussian_ids: torch.Tensor, full_count: int) -> tuple[torch.Tensor, dict[str, Any]]:
    for key in ("dynamic_score", "raw_score", "motion_score"):
        value = payload.get(key)
        if value is not None:
            return _tensor_for_ids(value, gaussian_ids, full_count, 0.0), {
                "score_source": key,
                "score_aggregation": "provided",
            }
    for key in ("motion_ratio_corrected", "motion_ratio_raw"):
        value = payload.get(key)
        if value is not None:
            return _temporal_motion_ratio_for_ids(value, gaussian_ids, full_count, 0.0), {
                "score_source": key,
                "score_aggregation": "geometric_mean_over_targets",
            }
    return torch.zeros(gaussian_ids.numel(), dtype=torch.float32), {
        "score_source": "missing",
        "score_aggregation": "zeros",
    }


def _denoise_candidates(args, output: Path) -> list[Path]:
    candidates: list[Path] = []
    if args.denoise_evidence:
        candidates.append(Path(args.denoise_evidence).expanduser())
    static_dir = Path(args.static_model_dir).expanduser().resolve()
    candidates.extend(
        [
            static_dir / "dynamic_debug" / "render_guided_motion_probe" / "per_gaussian_evidence.pt",
            static_dir / "render_denoise_discovery" / "per_gaussian_evidence.pt",
            static_dir / "dynamic_debug" / "render_denoise_discovery" / "per_gaussian_evidence.pt",
            output / "denoise" / "per_gaussian_evidence.pt",
        ]
    )
    denoise_root = Path(args.denoise_output_dir).expanduser().resolve()
    candidates.extend(sorted(denoise_root.glob("evaluation/checkpoint_target_*/temporal/frame0_temporal_motion.pt")))
    candidates.extend(sorted(denoise_root.glob("evaluation/checkpoint_target_*/frame_*/per_gaussian_motion.pt")))
    return candidates


def _ensure_denoise_discovery(args, output: Path) -> None:
    if not bool(args.build_denoise_if_missing):
        return
    if next((p for p in _denoise_candidates(args, output) if p.is_file()), None) is not None:
        return
    cmd = [
        sys.executable,
        str(REPO / "tools" / "run_frame0_motion_pipeline.py"),
        "--root-scene",
        str(Path(args.root_scene).expanduser().resolve()),
        "--static-model-dir",
        str(Path(args.static_model_dir).expanduser().resolve()),
        "--iteration",
        str(int(args.static_iteration)),
        "--output-root",
        str(Path(args.denoise_output_dir).expanduser().resolve()),
        "--train-targets",
        str(args.denoise_train_targets),
        "--eval-targets",
        str(args.denoise_eval_targets),
        "--inference-trials",
        str(int(args.denoise_inference_trials)),
        "--knn",
        str(int(args.denoise_knn)),
        "--training-iterations",
        str(int(args.denoise_training_iterations)),
        "--max-html-points",
        str(int(args.max_html_points)),
        "--skip-existing",
    ]
    cmd.append("--use-depth" if bool(args.denoise_use_depth) else "--no-use-depth")
    cmd.append("--debug-html" if bool(args.denoise_debug_html) else "--no-debug-html")
    cmd.append("--debug-2d" if bool(args.denoise_debug_2d) else "--no-debug-2d")
    if str(getattr(args, "denoise_cameras", "")).strip():
        cmd.extend(["--cameras", str(args.denoise_cameras).strip()])
    if str(getattr(args, "denoise_camera_data_device", "")).strip():
        cmd.extend(["--camera-data-device", str(args.denoise_camera_data_device).strip()])
    _run(cmd)
    evidence = next((p for p in _denoise_candidates(args, output) if p.is_file()), None)
    if evidence is not None:
        args.denoise_evidence = str(evidence)


def _denoise_prior(args, gaussian_ids: torch.Tensor, full_count: int) -> DynamicPrior:
    candidates = _denoise_candidates(args, Path(args.output).expanduser().resolve())
    evidence_path = next((p for p in candidates if p.is_file()), None)
    if evidence_path is None:
        half = torch.full((gaussian_ids.numel(),), 0.5, dtype=torch.float32)
        return DynamicPrior(
            probability=half,
            raw_score=torch.zeros_like(half),
            valid=torch.zeros_like(half, dtype=torch.bool),
            confidence=torch.zeros_like(half),
            method="denoise",
            gaussian_ids=gaussian_ids,
            metadata={
                "ground_truth_used_as_algorithm_input": False,
                "missing_evidence": True,
                "searched": [str(p) for p in candidates],
            },
        ).validate()

    payload = torch.load(evidence_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "per_gaussian" in payload and isinstance(payload["per_gaussian"], dict):
        payload = payload["per_gaussian"]
    raw, score_meta = _denoise_score_for_ids(payload, gaussian_ids, full_count)
    probability_value = payload.get(
        "final_dynamic_probability",
        payload.get("probability", payload.get("dynamic_probability")),
    )
    if probability_value is None:
        temperature = max(float(args.comparison_denoise_prob_temperature), 1.0e-6)
        probability = torch.sigmoid(
            (torch.log(raw.clamp_min(1.0e-8)) - float(args.comparison_denoise_prob_bias))
            / temperature
        )
    else:
        probability = _tensor_for_ids(probability_value, gaussian_ids, full_count, 0.5).clamp(0.0, 1.0)
    confidence = _tensor_for_ids(
        payload.get(
            "confidence",
            payload.get("seed_consistency_corrected", payload.get("seed_consistency_raw")),
        ),
        gaussian_ids,
        full_count,
        1.0,
    ).clamp(0.0, 1.0)
    unknown = _tensor_for_ids(payload.get("unknown_score"), gaussian_ids, full_count, 0.0).clamp(0.0, 1.0)
    confidence = torch.minimum(confidence, 1.0 - unknown)
    valid = confidence > 0.0
    return DynamicPrior(
        probability=probability,
        raw_score=raw,
        valid=valid,
        confidence=confidence,
        method="denoise",
        gaussian_ids=gaussian_ids,
        metadata={
            "ground_truth_used_as_algorithm_input": False,
            "source": str(evidence_path),
            **score_meta,
            "probability_model": "sigmoid((log(score)-bias)/temperature)",
        },
    ).validate()


def _load_gt_mask(args, canonical_xyz: torch.Tensor, static_ply: str | Path | None = None) -> torch.Tensor | None:
    if args.no_gt:
        return None
    if static_ply is not None:
        ported = _load_ported_gt_mask(
            static_ply,
            int(args.rigid_analytic_moving_part),
            int(canonical_xyz.shape[0]),
        )
        if ported is not None:
            return ported
    try:
        from utils.rigid_killing_validation import load_rigid_dataset_ground_truth

        gt = load_rigid_dataset_ground_truth(
            args.root_scene,
            canonical_xyz,
            moving_part=int(args.rigid_analytic_moving_part),
        )
        return gt.moving_mask.detach().cpu().bool()
    except Exception as exc:
        print(f"[comparison] GT unavailable: {exc}")
        return None


def _hash_payload(payload: Any) -> str:
    stream = io.BytesIO()
    torch.save(payload, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, tuple):
            return [clean(v) for v in value]
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _safe_metric(value: Any) -> str:
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "num_gaussians",
        "valid_fraction",
        "mean_probability",
        "mean_confidence",
        "all_precision",
        "all_recall",
        "all_f1",
        "all_iou",
        "all_pr_auc",
        "all_roc_auc",
        "high_confidence_f1",
        "prior_seed_parts",
        "prior_seed_static_gaussians",
        "prior_seed_dynamic_gaussians",
        "prior_seed_part_sizes",
        "prior_seed_precision",
        "prior_seed_recall",
        "prior_seed_f1",
        "prior_seed_iou",
        "killing_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _safe_metric(row.get(field)) for field in fields})


def _subsample(n: int, max_points: int, seed: int) -> torch.Tensor:
    if n <= max_points:
        return torch.arange(n)
    gen = torch.Generator().manual_seed(int(seed))
    return torch.randperm(n, generator=gen)[: int(max_points)]


def _write_prior_html(path: Path, xyz: torch.Tensor, priors: dict[str, DynamicPrior], gt_mask: torch.Tensor | None, max_points: int, seed: int) -> None:
    if not priors:
        return
    id_sets = [set(prior.gaussian_ids.detach().cpu().long().tolist()) for prior in priors.values()]
    common_ids = set.intersection(*id_sets) if id_sets else set()
    if common_ids:
        ids = torch.tensor(sorted(common_ids), dtype=torch.long)
        id_policy = "intersection"
    else:
        ids = torch.unique(torch.cat([prior.gaussian_ids.long() for prior in priors.values()]), sorted=True)
        id_policy = "union_fallback"
    points = xyz.index_select(0, ids).detach().cpu()
    sel = _subsample(points.shape[0], max_points, seed)
    selected_ids = ids.index_select(0, sel)
    scalars: dict[str, list[float]] = {}
    if gt_mask is not None:
        scalars["gt_label"] = gt_mask.index_select(0, selected_ids).float().tolist()
    for name, prior in priors.items():
        probability = torch.full((ids.numel(),), 0.5, dtype=torch.float32)
        confidence = torch.zeros(ids.numel(), dtype=torch.float32)
        prior_ids = prior.gaussian_ids.detach().cpu().long()
        dst_all = torch.searchsorted(ids, prior_ids)
        in_range = dst_all < ids.numel()
        matched = torch.zeros_like(in_range, dtype=torch.bool)
        matched[in_range] = ids.index_select(0, dst_all[in_range]) == prior_ids[in_range]
        dst = dst_all[matched]
        probability.index_copy_(0, dst, prior.probability.detach().cpu().float()[matched])
        confidence.index_copy_(0, dst, prior.confidence.detach().cpu().float()[matched])
        scalars[f"{name}_probability"] = probability.index_select(0, sel).tolist()
        scalars[f"{name}_confidence"] = confidence.index_select(0, sel).tolist()
    first = next(iter(scalars))
    payload = {
        "xyz": points.index_select(0, sel).tolist(),
        "scalars": scalars,
        "first": first,
        "id_policy": id_policy,
        "num_ids": int(ids.numel()),
    }
    path.write_text(
        """<!doctype html><meta charset="utf-8"><title>Dynamic prior comparison</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<select id="scalar"></select><div id="plot" style="width:100vw;height:92vh"></div>
<script>
const payload = """
        + json.dumps(payload)
        + """;
const sel = document.getElementById('scalar');
for (const k of Object.keys(payload.scalars)) { const o=document.createElement('option'); o.value=k; o.textContent=k; sel.appendChild(o); }
function draw(k) {
  const xyz = payload.xyz;
	  Plotly.react('plot', [{type:'scatter3d', mode:'markers', x:xyz.map(p=>p[0]), y:xyz.map(p=>p[1]), z:xyz.map(p=>p[2]),
	    marker:{size:2, color:payload.scalars[k], colorscale:'Viridis', cmin:0, cmax:1, showscale:true}}], {margin:{l:0,r:0,b:0,t:20}, title:k+' | ids: '+payload.id_policy+' n='+payload.num_ids, scene:{aspectmode:'data'}});
}
sel.onchange = () => draw(sel.value); sel.value = payload.first; draw(payload.first);
</script>""",
        encoding="utf-8",
    )


def _write_transport_html(path: Path, source: torch.Tensor, target: torch.Tensor, probability: torch.Tensor, confidence: torch.Tensor, max_points: int, seed: int) -> None:
    sel = _subsample(source.shape[0], max_points, seed)
    s = source.index_select(0, sel).detach().cpu().tolist()
    t = target.index_select(0, sel).detach().cpu().tolist()
    p = probability.index_select(0, sel).detach().cpu().tolist()
    c = confidence.index_select(0, sel).detach().cpu().tolist()
    payload = {"source": s, "target": t, "probability": p, "confidence": c}
    path.write_text(
        """<!doctype html><meta charset="utf-8"><title>Endpoint transport arrows</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script><div id="plot" style="width:100vw;height:96vh"></div>
<script>
const p = """
        + json.dumps(payload)
        + """;
let xs=[], ys=[], zs=[];
for (let i=0;i<p.source.length;i++) { xs.push(p.source[i][0], p.target[i][0], null); ys.push(p.source[i][1], p.target[i][1], null); zs.push(p.source[i][2], p.target[i][2], null); }
Plotly.newPlot('plot', [
 {type:'scatter3d', mode:'markers', x:p.source.map(q=>q[0]), y:p.source.map(q=>q[1]), z:p.source.map(q=>q[2]), marker:{size:2,color:p.probability,colorscale:'Viridis',cmin:0,cmax:1,showscale:true}, name:'source'},
 {type:'scatter3d', mode:'lines', x:xs, y:ys, z:zs, line:{width:2,color:'rgba(30,30,30,0.35)'}, name:'transport'}
	], {margin:{l:0,r:0,b:0,t:20}, title:'endpoint transport arrows', scene:{aspectmode:'data'}});
</script>""",
        encoding="utf-8",
    )


def _write_disagreement_html(path: Path, xyz: torch.Tensor, denoise: DynamicPrior, other: DynamicPrior, other_name: str, max_points: int, seed: int) -> None:
    if not torch.equal(denoise.gaussian_ids, other.gaussian_ids):
        return
    points = xyz.index_select(0, denoise.gaussian_ids).detach().cpu()
    sel = _subsample(points.shape[0], max_points, seed)
    d = (denoise.probability >= 0.5).long()
    o = (other.probability >= 0.5).long()
    category = (2 * d + o).index_select(0, sel).tolist()
    payload = {"xyz": points.index_select(0, sel).tolist(), "category": category, "other_name": other_name}
    path.write_text(
        """<!doctype html><meta charset="utf-8"><title>Denoise disagreement</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script><div id="plot" style="width:100vw;height:96vh"></div>
<script>
const p = """
        + json.dumps(payload)
        + """;
const xyz = p.xyz;
Plotly.newPlot('plot', [{type:'scatter3d', mode:'markers', x:xyz.map(q=>q[0]), y:xyz.map(q=>q[1]), z:xyz.map(q=>q[2]),
 marker:{size:2,color:p.category,colorscale:[[0,'#3b4cc0'],[0.33,'#19a974'],[0.66,'#ff9f1c'],[1,'#d7263d']],cmin:0,cmax:3,showscale:true}}], {
	 margin:{l:0,r:0,b:0,t:32}, title:'0 D static / other static, 1 D static / other dynamic, 2 D dynamic / other static, 3 both dynamic: '+p.other_name, scene:{aspectmode:'data'}
	});
</script>""",
        encoding="utf-8",
    )


def _write_histograms(path: Path, priors: dict[str, DynamicPrior], gt_mask: torch.Tensor | None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = list(priors)
    fig, axes = plt.subplots(max(1, len(names)), 1, figsize=(7, max(2.5, 2.2 * len(names))))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        prior = priors[name]
        prob = prior.probability.detach().cpu().numpy()
        if gt_mask is None:
            ax.hist(prob, bins=40, range=(0.0, 1.0), alpha=0.8)
        else:
            truth = gt_mask.index_select(0, prior.gaussian_ids).cpu().numpy().astype(bool)
            ax.hist(prob[~truth], bins=40, range=(0.0, 1.0), alpha=0.65, label="GT static")
            ax.hist(prob[truth], bins=40, range=(0.0, 1.0), alpha=0.65, label="GT dynamic")
            ax.legend(loc="best")
        ax.set_title(name)
        ax.set_xlabel("P(dynamic)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_prior_scatter(path: Path, priors: dict[str, DynamicPrior], gt_mask: torch.Tensor | None) -> None:
    if "denoise" not in priors:
        return
    other_name = (
        "flow_matching"
        if "flow_matching" in priors
        else "endpoint_transport_debug"
        if "endpoint_transport_debug" in priors
        else None
    )
    if other_name is None:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    denoise = priors["denoise"]
    other = priors[other_name]
    if not torch.equal(denoise.gaussian_ids, other.gaussian_ids):
        return
    x = denoise.probability.detach().cpu().numpy()
    y = other.probability.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    if gt_mask is None:
        ax.scatter(x, y, s=3, alpha=0.35)
    else:
        truth = gt_mask.index_select(0, denoise.gaussian_ids).detach().cpu().numpy()
        ax.scatter(x[~truth], y[~truth], s=3, alpha=0.25, label="GT static")
        ax.scatter(x[truth], y[truth], s=3, alpha=0.25, label="GT dynamic")
        ax.legend(loc="best")
    ax.set_xlabel("denoise P(dynamic)")
    ax.set_ylabel(f"{other_name} P(dynamic)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _correlation(rows: dict[str, DynamicPrior]) -> dict[str, float]:
    if "denoise" not in rows:
        return {}
    other_name = (
        "flow_matching"
        if "flow_matching" in rows
        else "endpoint_transport_debug"
        if "endpoint_transport_debug" in rows
        else None
    )
    if other_name is None:
        return {}
    a = rows["denoise"].probability.float()
    b = rows[other_name].probability.float()
    if a.numel() != b.numel() or a.numel() == 0:
        return {}
    a0 = a - a.mean()
    b0 = b - b.mean()
    pearson = float((a0 * b0).mean() / (a0.std(unbiased=False) * b0.std(unbiased=False)).clamp_min(1.0e-8))
    ar = torch.argsort(torch.argsort(a)).float()
    br = torch.argsort(torch.argsort(b)).float()
    ar = ar - ar.mean()
    br = br - br.mean()
    spearman = float((ar * br).mean() / (ar.std(unbiased=False) * br.std(unbiased=False)).clamp_min(1.0e-8))
    return {"denoise_vs_" + other_name + "_pearson": pearson, "denoise_vs_" + other_name + "_spearman": spearman}


def _smooth_prior_3d(
    prior: DynamicPrior,
    full_xyz: torch.Tensor,
    knn: int = 16,
    iters: int = 3,
    alpha: float = 0.7,
) -> DynamicPrior:
    """KNN diffusion smoothing: iteratively replaces each Gaussian's probability with a
    confidence-weighted blend of its spatial neighbours.  Makes the static/dynamic
    boundary sharper without touching Gaussians whose neighbours all agree."""
    xyz = full_xyz.index_select(0, prior.gaussian_ids.long()).float()
    prob = prior.probability.float().clone()
    conf = prior.confidence.float().clamp_min(0.0)
    n = xyz.shape[0]
    k = min(max(1, int(knn)), n - 1)
    if k <= 0 or iters <= 0:
        return prior
    chunk = 1024
    nn_idx_list: list[torch.Tensor] = []
    for start in range(0, n, chunk):
        d2 = torch.cdist(xyz[start : start + chunk], xyz)
        _, idx = torch.topk(d2, k=k + 1, dim=1, largest=False)
        nn_idx_list.append(idx[:, 1:])
    nn_idx = torch.cat(nn_idx_list, 0)  # [N, k]
    for _ in range(iters):
        neighbor_prob = prob[nn_idx.reshape(-1)].reshape(n, k)
        neighbor_conf = conf[nn_idx.reshape(-1)].reshape(n, k)
        w = (neighbor_conf + 1.0e-6) / (neighbor_conf + 1.0e-6).sum(dim=1, keepdim=True)
        smoothed = (neighbor_prob * w).sum(dim=1)
        prob = (alpha * prob + (1.0 - alpha) * smoothed).clamp(0.0, 1.0)
    return DynamicPrior(
        probability=prob.cpu(),
        raw_score=prior.raw_score,
        valid=prior.valid,
        confidence=prior.confidence,
        method=prior.method,
        gaussian_ids=prior.gaussian_ids,
        metadata={**prior.metadata, "smoothed_knn": knn, "smoothed_iters": iters, "smoothed_alpha": alpha},
    ).validate()


def _prior_seed_labels(
    prior: DynamicPrior,
    full_xyz: torch.Tensor,
    *,
    k_parts: int = 1,
    max_iters: int = 150,
) -> torch.Tensor:
    """Static/dynamic prior seed labels only.

    Real K-moving-part clustering is Stage-1 `--kpart_rigid_discovery` inside
    the branch training run. This quick mask exists only to inspect the prior
    before the physical K-part/Killing solver.

    Returns a long tensor of shape [len(prior.gaussian_ids)]:
      0   = static  (prob ≤ 0.5)
      1   = dynamic (prob > 0.5)
    """
    del full_xyz, k_parts, max_iters
    prob = prior.probability.float().clamp(0.0, 1.0)
    labels = torch.zeros(prob.shape[0], dtype=torch.long)
    labels[prob > 0.5] = 1
    return labels


def _prior_seed_metrics(
    labels: torch.Tensor,
    gaussian_ids: torch.Tensor,
    gt_mask: torch.Tensor | None,
) -> dict[str, Any]:
    labels = torch.as_tensor(labels).detach().cpu().long().reshape(-1)
    if labels.numel() == 0:
        return {
            "prior_seed_parts": 0,
            "prior_seed_static_gaussians": 0,
            "prior_seed_dynamic_gaussians": 0,
            "prior_seed_part_sizes": "{}",
        }
    part_sizes = {
        int(part): int((labels == part).sum())
        for part in range(int(labels.max().item()) + 1)
    }
    metrics: dict[str, Any] = {
        "prior_seed_parts": int(labels.max().item()),
        "prior_seed_static_gaussians": int((labels == 0).sum()),
        "prior_seed_dynamic_gaussians": int((labels > 0).sum()),
        "prior_seed_part_sizes": json.dumps(part_sizes, sort_keys=True),
    }
    if gt_mask is not None:
        ids = torch.as_tensor(gaussian_ids).detach().cpu().long().reshape(-1)
        gt = torch.as_tensor(gt_mask).detach().cpu().bool().reshape(-1)
        if ids.numel() == labels.numel() and (ids.numel() == 0 or int(ids.max()) < int(gt.numel())):
            truth = gt.index_select(0, ids)
            pred = labels > 0
            tp = float((pred & truth).sum())
            fp = float((pred & ~truth).sum())
            fn = float((~pred & truth).sum())
            precision = tp / max(1.0, tp + fp)
            recall = tp / max(1.0, tp + fn)
            metrics.update(
                {
                    "prior_seed_precision": precision,
                    "prior_seed_recall": recall,
                    "prior_seed_f1": (
                        2.0 * precision * recall / max(1.0e-12, precision + recall)
                    ),
                    "prior_seed_iou": tp / max(1.0, tp + fp + fn),
                }
            )
    return metrics


def _write_prior_seed_html(
    path: Path,
    xyz: torch.Tensor,
    seed_labels: torch.Tensor,
    gaussian_ids: torch.Tensor,
    prior: "DynamicPrior",
    smoothed_prior: "DynamicPrior | None",
    max_points: int,
    seed: int,
    gt_mask: "torch.Tensor | None" = None,
) -> None:
    """Rich debug HTML: dropdown switches between prior seed labels,
    raw probability, smoothed probability (if available), and GT (if available)."""
    points = xyz.index_select(0, gaussian_ids.long()).detach().cpu()
    sel = _subsample(points.shape[0], max_points, seed)
    labels_sel = seed_labels.index_select(0, sel).tolist()
    prob_sel = prior.probability.float().index_select(0, sel).tolist()
    conf_sel = prior.confidence.float().index_select(0, sel).tolist()
    k = int(seed_labels.max().item())
    # per-part sizes for debug info
    part_sizes = {int(p): int((seed_labels == p).sum()) for p in range(0, k + 1)}

    scalars: dict[str, list] = {
        "probability (raw)": prob_sel,
        "confidence (raw)": conf_sel,
    }
    if smoothed_prior is not None:
        scalars["probability (smoothed)"] = smoothed_prior.probability.float().index_select(0, sel).tolist()
    if gt_mask is not None:
        scalars["GT dynamic"] = gt_mask.index_select(0, gaussian_ids.long()).float().index_select(0, sel).tolist()

    payload = {
        "xyz": points.index_select(0, sel).tolist(),
        "label": labels_sel,
        "scalars": scalars,
        "k": k,
        "part_sizes": part_sizes,
        "method": prior.method,
    }
    path.write_text(
        """<!doctype html><meta charset="utf-8">
<title>Prior seed debug</title>
<style>
body{margin:0;font-family:sans-serif;background:#111;color:#eee}
#controls{position:fixed;top:8px;left:8px;z-index:9;display:flex;gap:8px;align-items:center;background:rgba(0,0,0,.6);padding:6px 10px;border-radius:6px}
#info{position:fixed;bottom:8px;left:8px;z-index:9;background:rgba(0,0,0,.6);padding:6px 10px;border-radius:6px;font-size:12px;white-space:pre}
select,button{background:#333;color:#eee;border:1px solid #555;padding:3px 6px;border-radius:4px;cursor:pointer}
</style>
<div id="controls">
  <label>View: <select id="mode">
    <option value="kpart">Prior seed labels</option>
    <option value="scalar">Scalar</option>
  </select></label>
  <label id="scalar_label">Scalar: <select id="scalar"></select></label>
</div>
<div id="info"></div>
<div id="plot" style="width:100vw;height:100vh"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>
<script>
const p = """ + json.dumps(payload) + """;
const COLORS = ['#888888','#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a0d020','#a65628','#f781bf','#999999'];
const xyz = p.xyz;
const sel_scalar = document.getElementById('scalar');
const sel_mode = document.getElementById('mode');
const scalar_label = document.getElementById('scalar_label');
const info_div = document.getElementById('info');

for (const k of Object.keys(p.scalars)) {
  const o = document.createElement('option'); o.value = k; o.textContent = k; sel_scalar.appendChild(o);
}

function infoText() {
  let s = 'Method: ' + p.method + '  |  prior seed dynamic groups=' + p.k + '\\n';
  for (const [part, size] of Object.entries(p.part_sizes)) {
    const label = part == 0 ? 'static' : 'part ' + part;
    s += '  ' + label + ': ' + size + ' Gaussians\\n';
  }
  return s.trim();
}
info_div.textContent = infoText();

function drawPriorSeed() {
  const traces = [];
  for (let part = 0; part <= p.k; part++) {
    const idx = p.label.map((l,i) => l===part ? i : -1).filter(i=>i>=0);
    if (!idx.length) continue;
    traces.push({
      type:'scatter3d', mode:'markers',
      x:idx.map(i=>xyz[i][0]), y:idx.map(i=>xyz[i][1]), z:idx.map(i=>xyz[i][2]),
      marker:{size:2, color:COLORS[part % COLORS.length], opacity:0.85},
      name:(part===0 ? 'static ('+p.part_sizes[0]+')' : 'part '+part+' ('+p.part_sizes[part]+')')
    });
  }
  Plotly.react('plot', traces, {
    margin:{l:0,r:0,b:0,t:0}, paper_bgcolor:'#111', plot_bgcolor:'#111',
    scene:{aspectmode:'data', bgcolor:'#111',
           xaxis:{color:'#aaa'}, yaxis:{color:'#aaa'}, zaxis:{color:'#aaa'}},
    legend:{font:{color:'#eee'}, bgcolor:'rgba(0,0,0,0.5)'}
  });
}

function drawScalar(key) {
  const vals = p.scalars[key];
  Plotly.react('plot', [{
    type:'scatter3d', mode:'markers',
    x:xyz.map(q=>q[0]), y:xyz.map(q=>q[1]), z:xyz.map(q=>q[2]),
    marker:{size:2, color:vals, colorscale:'Viridis', cmin:0, cmax:1,
            showscale:true, colorbar:{tickfont:{color:'#eee'}}},
    name: key
  }], {
    margin:{l:0,r:0,b:0,t:0}, paper_bgcolor:'#111', plot_bgcolor:'#111',
    scene:{aspectmode:'data', bgcolor:'#111',
           xaxis:{color:'#aaa'}, yaxis:{color:'#aaa'}, zaxis:{color:'#aaa'}}
  });
}

function refresh() {
  const mode = sel_mode.value;
  scalar_label.style.display = mode === 'scalar' ? '' : 'none';
  if (mode === 'kpart') drawPriorSeed();
  else drawScalar(sel_scalar.value);
}
sel_mode.onchange = refresh;
sel_scalar.onchange = () => drawScalar(sel_scalar.value);
refresh();
</script>""",
        encoding="utf-8",
    )


def _cfg_matches(saved: dict[str, Any], current: dict[str, Any], ignore: set[str] | None = None) -> bool:
    """Return True iff saved and current dicts agree on every key not in ignore."""
    skip = ignore or set()
    for k, v in current.items():
        if k in skip:
            continue
        if saved.get(k) != v:
            return False
    return True


def _load_transport_result(endpoint_dir: Path) -> EndpointTransportPriorResult | None:
    """Reconstruct EndpointTransportPriorResult from files saved by save_endpoint_transport_result
    plus the prior.pt written by this tool.  Returns None if any required file is missing."""
    required = [
        "nearest_distance.pt", "displacement.pt", "transport_entropy.pt",
        "motion_scale.pt", "barycentric_target.pt", "prior.pt",
    ]
    if not all((endpoint_dir / f).is_file() for f in required):
        return None
    try:
        prior = load_dynamic_prior(endpoint_dir / "prior.pt")
        load = lambda name: torch.load(endpoint_dir / name, map_location="cpu", weights_only=False)
        return EndpointTransportPriorResult(
            prior=prior,
            barycentric_target=load("barycentric_target.pt"),
            displacement=load("displacement.pt"),
            nearest_distance=load("nearest_distance.pt"),
            entropy=load("transport_entropy.pt"),
            target_indices=load("target_indices.pt") if (endpoint_dir / "target_indices.pt").is_file() else None,
            transport_weights=load("transport_weights.pt") if (endpoint_dir / "transport_weights.pt").is_file() else None,
            motion_scale=load("motion_scale.pt"),
        )
    except Exception as exc:
        print(f"[comparison] transport cache load failed ({exc}) — recomputing")
        return None


def _run_strict_closed_loop_rgbd_prior(args, output: Path) -> DynamicPrior:
    branch = output / "strict_closed_loop_rgbd" / "prior_probe"
    prior_path = branch / "dynamic_prior.pt"
    summary_path = branch / "closed_loop_endpoint_fm_summary.json"
    if prior_path.is_file() and summary_path.is_file() and not bool(getattr(args, "force_closed_loop_rgbd", False)):
        print("[comparison:strict_closed_loop_rgbd] cached prior — skipping probe")
        return load_dynamic_prior(prior_path)
    branch.mkdir(parents=True, exist_ok=True)
    endpoint_model = Path(args.endpoint_ply).expanduser().resolve() if str(args.endpoint_ply).strip() else Path(args.endpoint_model_dir).expanduser().resolve()
    cmd = [
        sys.executable,
        str(REPO / "tools" / "run_closed_loop_endpoint_fm_probe.py"),
        "--root-scene",
        str(Path(args.root_scene).expanduser().resolve()),
        "--static-model-dir",
        str(Path(args.static_model_dir).expanduser().resolve()),
        "--endpoint-static-model-dir",
        str(endpoint_model),
        "--static-iteration",
        str(int(args.static_iteration)),
        "--endpoint-static-iteration",
        str(int(args.endpoint_iteration)),
        "--output-dir",
        str(branch),
        "--fm-iters",
        str(int(args.closed_loop_fm_iters)),
        "--fm-batch-size",
        str(int(args.closed_loop_fm_batch_size)),
        "--fm-knn",
        str(int(args.closed_loop_fm_knn)),
        "--chunk-size",
        str(int(args.closed_loop_chunk_size)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.closed_loop_device),
        "--video-max-cameras",
        str(int(args.closed_loop_video_max_cameras)),
        "--video-max-pixels",
        str(int(args.closed_loop_video_max_pixels)),
        "--video-frame-stride",
        str(int(args.closed_loop_video_frame_stride)),
        "--video-pixel-candidates",
        str(int(args.closed_loop_video_pixel_candidates)),
        "--video-pixel-search-radius",
        str(int(args.closed_loop_video_pixel_search_radius)),
        "--video-topk-gaussians",
        str(int(args.closed_loop_video_topk_gaussians)),
        "--video-max-pixel-radius",
        str(float(args.closed_loop_video_max_pixel_radius)),
        "--video-depth-sigma",
        str(float(args.closed_loop_video_depth_sigma)),
        "--video-likelihood-weight",
        str(float(args.closed_loop_video_likelihood_weight)),
        "--static-threshold",
        str(float(args.closed_loop_static_threshold)),
        "--dynamic-threshold",
        str(float(args.closed_loop_dynamic_threshold)),
        "--max-html-points",
        str(int(args.max_html_points)),
    ]
    if bool(args.closed_loop_use_video_rgbd):
        cmd.append("--use-video-rgbd")
    print("[comparison:strict_closed_loop_rgbd] running strict endpoint/full-video RGB-D prior probe")
    _run(cmd)
    return load_dynamic_prior(prior_path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_shared_killing(args, method: str, branch_dir: Path, prior_path: Path) -> dict[str, Any]:
    if not args.run_killing:
        payload = {"status": "skipped", "reason": "run_killing disabled"}
        torch.save(payload, branch_dir / "killing_result.pt")
        return payload
    branch_run_kpart = bool(args.run_kpart) and method != "baseline"
    branch_fit_joints = bool(args.fit_joints) and branch_run_kpart
    branch_optimize_joint_trajectory = bool(args.optimize_joint_trajectory) and branch_fit_joints
    run_dir = branch_dir / "dynamic_run"
    start_ply = _ply_from_model_dir(args.static_model_dir, args.static_iteration)
    start_has_sam3 = _ply_has_sam3_field(start_ply)
    pipeline_config = {
        "run_kpart": branch_run_kpart,
        "kpart_k": int(args.kpart_k) if branch_run_kpart else None,
        "kpart_cycles": int(args.kpart_cycles) if branch_run_kpart else None,
        "kpart_anneal_cycles": int(args.kpart_anneal_cycles) if branch_run_kpart else None,
        "kpart_plateau_cycles": int(args.kpart_plateau_cycles) if branch_run_kpart else None,
        "kpart_confidence_threshold": float(args.kpart_confidence_threshold) if branch_run_kpart else None,
        "kpart_dynamic_prior_weight": float(args.kpart_dynamic_prior_weight) if branch_run_kpart else None,
        "kpart_prior_anchor_threshold": float(args.kpart_prior_anchor_threshold) if branch_run_kpart else None,
        "kpart_prior_boundary_expansion": int(args.kpart_prior_boundary_expansion) if branch_run_kpart else None,
        "kpart_prior_cleanup_min_size": int(args.kpart_prior_cleanup_min_size) if branch_run_kpart else None,
        "phase1_motion_representation": str(args.phase1_motion_representation),
        "fit_joints": branch_fit_joints,
        "optimize_joint_trajectory": branch_optimize_joint_trajectory,
        "joint_types": str(args.joint_types),
        "joint_compare_iters": int(args.joint_compare_iters),
        "joint_trajectory_mode": str(args.joint_trajectory_mode),
        "joint_trajectory_iters": int(args.joint_trajectory_iters),
        "start_ply": str(start_ply),
        "start_has_sam3": bool(start_has_sam3),
        "kpart_uses_dino": False,
    }
    result_path = branch_dir / "killing_result.pt"
    if not bool(getattr(args, "force_killing", False)) and result_path.is_file() and prior_path.is_file():
        try:
            existing = torch.load(result_path, map_location="cpu", weights_only=False)
            if isinstance(existing, dict) and existing.get("status") == "complete":
                stored_hash = existing.get("prior_hash", "")
                current_hash = _sha256_file(prior_path)
                # skip if no hash stored (legacy result) OR hash matches; re-run only if hash present and mismatches
                if (not stored_hash or stored_hash == current_hash) and existing.get("pipeline_config") == pipeline_config:
                    print(f"[comparison:{method}] killing already complete and prior unchanged — skipping")
                    return existing
        except Exception:
            pass
    cmd = [
        sys.executable,
        str(REPO / "train.py"),
        "-s",
        str(Path(args.root_scene).expanduser().resolve()),
        "-m",
        str(run_dir),
        "--iterations",
        str(int(args.static_iteration)),
        "--start_ply",
        str(start_ply),
        "--start_ply_iteration",
        str(int(args.static_iteration)),
        "--static_train_iters",
        str(int(args.static_iteration)),
        "--dynamic_motion",
        "--phase1_motion_representation",
        str(args.phase1_motion_representation),
        "--phase1_only",
        "--use_semantic_feature_field",
        "--dynamic_use_depth",
        "--use_motion_correspondence",
        "--dynamic_prior_path",
        str(prior_path),
        "--prior_hard_threshold",
        str(float(args.prior_hard_threshold)),
        "--semantic_dynamic_threshold",
        str(float(args.semantic_dynamic_threshold)),
        "--curl_grid_target_cells",
        str(int(args.curl_grid_target_cells)),
        "--disable_viewer",
    ]
    cmd.extend(["--no_kpart_dino", "--kpart_dino_3d_weight", "0.0", "--kpart_dino_2d_seam_weight", "0.0"])
    if start_has_sam3:
        cmd.extend(
            [
                "--use_sam3_feature_field",
                "--sam3_feature_root",
                str(Path(args.sam3_feature_root).expanduser().resolve()),
                "--sam3_feature_dim",
                str(int(args.sam3_feature_dim)),
            ]
        )
    if branch_run_kpart:
        cmd.extend(
            [
                "--kpart_rigid_discovery",
                "--num_moving_parts",
                str(int(args.kpart_k)),
                "--kpart_dynamic_prior_weight",
                str(float(args.kpart_dynamic_prior_weight)),
                "--kpart_prior_boundary_expansion",
                str(int(args.kpart_prior_boundary_expansion)),
                "--kpart_prior_anchor_threshold",
                str(float(args.kpart_prior_anchor_threshold)),
                "--kpart_prior_cleanup_min_size",
                str(int(args.kpart_prior_cleanup_min_size)),
                "--kpart_confidence_threshold",
                str(float(args.kpart_confidence_threshold)),
                "--kpart_anneal_cycles",
                str(int(args.kpart_anneal_cycles)),
                "--kpart_plateau_cycles",
                str(int(args.kpart_plateau_cycles)),
            ]
        )
        if int(args.kpart_cycles) > 0:
            cmd.extend(["--kpart_cycles", str(int(args.kpart_cycles))])
        if branch_fit_joints:
            cmd.extend(
                [
                    "--fit_joints",
                    "--joint_types",
                    str(args.joint_types),
                    "--joint_topology",
                    str(args.joint_topology),
                    "--joint_compare_iters",
                    str(int(args.joint_compare_iters)),
                    "--joint_compare_batch_size",
                    str(int(args.joint_compare_batch_size)),
                    "--joint_compare_validation_cameras",
                    str(int(args.joint_compare_validation_cameras)),
                ]
            )
        if branch_optimize_joint_trajectory:
            cmd.extend(
                [
                    "--optimize_joint_trajectory",
                    "--joint_trajectory_mode",
                    str(args.joint_trajectory_mode),
                    "--joint_trajectory_iters",
                    str(int(args.joint_trajectory_iters)),
                    "--joint_trajectory_batch_size",
                    str(int(args.joint_trajectory_batch_size)),
                    "--joint_trajectory_acceleration_weight",
                    str(float(args.joint_trajectory_acceleration_weight)),
                    "--joint_trajectory_jerk_weight",
                    str(float(args.joint_trajectory_jerk_weight)),
                    "--joint_trajectory_reverse_weight",
                    str(float(args.joint_trajectory_reverse_weight)),
                ]
            )
    elif method != "baseline" and (bool(args.fit_joints) or bool(args.optimize_joint_trajectory)):
        raise RuntimeError("Joint stages require --run-kpart")
    if method == "baseline":
        cmd.append("--dynamic_use_all_gaussians")
    if bool(getattr(args, "freeze_prior_labels", False)) and not branch_run_kpart:
        cmd.append("--freeze_dynamic_prior_labels")
    cmd.append("--prior_use_soft_weights" if args.prior_use_soft_weights else "--no-prior_use_soft_weights")
    if not args.killing_wandb:
        cmd.append("--no_wandb")
    if args.extra and args.extra[0] == "--":
        cmd.extend(args.extra[1:])
    elif args.extra:
        cmd.extend(args.extra)
    print(f"[comparison:{method}] running dynamic branch pipeline")
    completed = subprocess.run(cmd, cwd=REPO)
    prior_hash = _sha256_file(prior_path) if prior_path.is_file() else ""
    payload = {
        "status": "complete" if completed.returncode == 0 else "failed",
        "returncode": int(completed.returncode),
        "run_dir": str(run_dir),
        "command": cmd,
        "prior_hash": prior_hash,
        "pipeline_config": pipeline_config,
        "pipeline_stages": [
            "static_dynamic_prior",
            "stage1_prior_kpart" if branch_run_kpart else "plain_all_foreground_killing",
            str(args.phase1_motion_representation),
            "stage2_revolute_prismatic_joint_fit" if branch_fit_joints else "skip_joint_fit",
            "stage3_joint_trajectory_reoptimization" if branch_optimize_joint_trajectory else "skip_joint_reoptimization",
        ],
    }
    torch.save(payload, branch_dir / "killing_result.pt")
    return payload


def _save_branch_prior(
    args,
    method: str,
    prior: DynamicPrior,
    output: Path,
    gt_mask: torch.Tensor | None,
    canonical_xyz: torch.Tensor | None = None,
) -> tuple[Path, dict[str, Any]]:
    branch = output / method
    branch.mkdir(parents=True, exist_ok=True)
    if method == "denoise":
        torch.save(prior.raw_score, branch / "denoise_raw_score.pt")
        torch.save(prior.confidence, branch / "denoise_confidence.pt")

    # 3D smoothing (optional) + prior seed mask. Real K-part runs in train.py.
    smooth_iters = int(getattr(args, "prior_smooth_iters", 0))
    smoothed: DynamicPrior | None = None
    effective_prior = prior
    seed_summary: dict[str, Any] | None = None
    if canonical_xyz is not None:
        if smooth_iters > 0:
            smoothed = _smooth_prior_3d(
                prior,
                canonical_xyz,
                knn=int(getattr(args, "prior_smooth_knn", 16)),
                iters=smooth_iters,
                alpha=float(getattr(args, "prior_smooth_alpha", 0.7)),
            )
            save_dynamic_prior(prior, branch / "raw_prior.pt")
            save_dynamic_prior(smoothed, branch / "smoothed_prior.pt")
            effective_prior = smoothed
        seed_labels = _prior_seed_labels(
            effective_prior,
            canonical_xyz,
            k_parts=int(getattr(args, "kpart_k", 1)),
        )
        seed_summary = _prior_seed_metrics(seed_labels, effective_prior.gaussian_ids, gt_mask)
        torch.save(seed_labels, branch / "prior_seed_labels.pt")
        dynamic_count = int((seed_labels > 0).sum().item())
        print(f"[comparison:{method}] prior seed dynamic Gaussians: {dynamic_count}")
        _write_prior_seed_html(
            branch / "prior_seed_labels.html",
            canonical_xyz,
            seed_labels,
            effective_prior.gaussian_ids,
            prior,
            smoothed,
            int(getattr(args, "max_html_points", 5000)),
            int(getattr(args, "seed", 0)),
            gt_mask=gt_mask,
        )

    prior_path = save_dynamic_prior(effective_prior, branch / "prior.pt")
    metrics = prior_metrics(effective_prior, gt_mask)
    if seed_summary is not None:
        metrics.update(seed_summary)
    if method == "gt":
        killing = {
            "status": "skipped",
            "reason": "GT prior is an evaluation upper bound only and is never passed to the Killing solver",
        }
        torch.save(killing, branch / "killing_result.pt")
    else:
        killing = _run_shared_killing(args, method, branch, prior_path)
    metrics["killing_status"] = killing.get("status", "unknown")
    _write_json(branch / "metrics.json", metrics)
    return prior_path, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-scene", required=True)
    parser.add_argument("--static-model-dir", required=True)
    parser.add_argument("--static-iteration", type=int, default=30000)
    parser.add_argument("--build-static-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-static-train", action="store_true")
    parser.add_argument("--output", "--comparison-output-dir", required=True)
    parser.add_argument(
        "--methods",
        "--comparison-methods",
        default="baseline,denoise,flow_matching",
        help=(
            "Comma/space separated methods: baseline all_fg gt denoise "
            "flow_matching endpoint_transport_debug strict_closed_loop_rgbd."
        ),
    )
    parser.add_argument("--semantic-dynamic-threshold", type=float, default=0.2)
    parser.add_argument("--semantic-dynamic-min-points", type=int, default=32)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--seed", "--comparison-seed", type=int, default=20260904)
    parser.add_argument("--max-html-points", type=int, default=5000)
    parser.add_argument("--no-gt", action="store_true")
    parser.add_argument("--rigid-analytic-moving-part", type=int, default=1)
    parser.add_argument("--denoise-evidence", default="")
    parser.add_argument("--build-denoise-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--denoise-output-dir", default="")
    parser.add_argument("--denoise-train-targets", default="1,59")
    parser.add_argument("--denoise-eval-targets", default="1,30,59")
    parser.add_argument("--denoise-inference-trials", type=int, default=16)
    parser.add_argument("--denoise-knn", type=int, default=16)
    parser.add_argument("--denoise-training-iterations", type=int, default=1000)
    parser.add_argument("--denoise-use-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--denoise-debug-2d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--denoise-debug-html", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--denoise-cameras",
        default="",
        help="Optional comma-separated camera IDs forwarded to denoise train/eval to reduce CUDA image/depth residency.",
    )
    parser.add_argument(
        "--denoise-camera-data-device",
        default="cpu",
        help="Device for cached denoise camera RGB/depth tensors; cpu avoids preloading all dynamic frames on CUDA.",
    )
    parser.add_argument("--comparison-denoise-prob-bias", type=float, default=0.0)
    parser.add_argument("--comparison-denoise-prob-temperature", type=float, default=0.35)
    parser.add_argument("--endpoint-model-dir", default="")
    parser.add_argument("--endpoint-iteration", type=int, default=30000)
    parser.add_argument("--endpoint-ply", default="")
    parser.add_argument("--endpoint-frame", type=int, default=-1)
    parser.add_argument("--endpoint-init", choices=("depth_fusion", "canonical", "colmap"), default="depth_fusion")
    parser.add_argument("--endpoint-depth-voxel-size", type=float, default=0.01)
    parser.add_argument("--build-endpoint-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-endpoint-train", action="store_true")
    parser.add_argument("--fm-endpoint-knn", type=int, default=16)
    parser.add_argument("--fm-transport-temperature", type=float, default=0.02)
    parser.add_argument("--fm-transport-robust-delta", type=float, default=0.02)
    parser.add_argument("--fm-motion-scale", choices=("local_knn", "global_nn", "gaussian_scale"), default="local_knn")
    parser.add_argument("--fm-confidence-entropy-scale", type=float, default=1.0)
    parser.add_argument("--fm-prior-prob-bias", type=float, default=1.0)
    parser.add_argument("--fm-prior-prob-temperature", type=float, default=0.35)
    parser.add_argument("--fm-transport-chunk-size", type=int, default=1024)
    parser.add_argument("--fm-hidden-dim", type=int, default=128)
    parser.add_argument("--fm-layers", type=int, default=4)
    parser.add_argument("--fm-lr", type=float, default=1.0e-3)
    parser.add_argument("--fm-iters", type=int, default=1000)
    parser.add_argument("--fm-batch-size", type=int, default=2048)
    parser.add_argument("--fm-identity-weight", type=float, default=0.01)
    parser.add_argument("--fm-smooth-weight", type=float, default=0.0)
    parser.add_argument("--fm-integrator", choices=("euler", "rk4"), default="euler")
    parser.add_argument("--fm-integration-steps", type=int, default=16)
    parser.add_argument("--closed-loop-use-video-rgbd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-closed-loop-rgbd", action="store_true")
    parser.add_argument("--closed-loop-device", default="cuda")
    parser.add_argument("--closed-loop-fm-iters", type=int, default=100)
    parser.add_argument("--closed-loop-fm-batch-size", type=int, default=2048)
    parser.add_argument("--closed-loop-fm-knn", type=int, default=16)
    parser.add_argument("--closed-loop-chunk-size", type=int, default=1024)
    parser.add_argument("--closed-loop-video-max-cameras", type=int, default=8)
    parser.add_argument("--closed-loop-video-max-pixels", type=int, default=4096)
    parser.add_argument("--closed-loop-video-frame-stride", type=int, default=1)
    parser.add_argument("--closed-loop-video-pixel-candidates", type=int, default=169)
    parser.add_argument("--closed-loop-video-pixel-search-radius", type=int, default=8)
    parser.add_argument("--closed-loop-video-topk-gaussians", type=int, default=8)
    parser.add_argument("--closed-loop-video-max-pixel-radius", type=float, default=6.0)
    parser.add_argument("--closed-loop-video-depth-sigma", type=float, default=0.03)
    parser.add_argument("--closed-loop-video-likelihood-weight", type=float, default=1.0)
    parser.add_argument("--closed-loop-static-threshold", type=float, default=0.35)
    parser.add_argument("--closed-loop-dynamic-threshold", type=float, default=0.65)
    parser.add_argument("--run-killing", action="store_true")
    parser.add_argument("--force-killing", action="store_true")
    parser.add_argument("--run-kpart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-transport", action="store_true",
                        help="Recompute endpoint transport even if cached result exists")
    parser.add_argument("--force-flow-matching", action="store_true",
                        help="Retrain flow matching even if cached checkpoint exists")
    parser.add_argument("--killing-wandb", action="store_true")
    parser.add_argument("--phase1-motion-representation", choices=("rigid_killing_eulerian", "rigid_killing_lagrangian", "rigid_killing_1d"), default="rigid_killing_eulerian")
    parser.add_argument("--freeze-prior-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prior-use-soft-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prior-hard-threshold", type=float, default=0.5)
    parser.add_argument("--prior-smooth-knn", type=int, default=16)
    parser.add_argument("--prior-smooth-iters", type=int, default=0,
                        help="KNN diffusion smoothing iterations (0 = disabled)")
    parser.add_argument("--prior-smooth-alpha", type=float, default=0.7,
                        help="Weight for own probability vs neighbours (0=fully smooth, 1=no change)")
    parser.add_argument("--kpart-k", type=int, default=1,
                        help="Number of dynamic moving parts passed to Stage-1 K-part discovery.")
    parser.add_argument("--kpart-cycles", type=int, default=0)
    parser.add_argument("--kpart-anneal-cycles", type=int, default=8)
    parser.add_argument("--kpart-plateau-cycles", type=int, default=5)
    parser.add_argument("--kpart-confidence-threshold", type=float, default=0.20)
    parser.add_argument("--kpart-dynamic-prior-weight", type=float, default=1.0)
    parser.add_argument("--kpart-prior-boundary-expansion", type=int, default=0)
    parser.add_argument("--kpart-prior-anchor-threshold", type=float, default=0.75)
    parser.add_argument("--kpart-prior-cleanup-min-size", type=int, default=16)
    parser.add_argument("--fit-joints", action="store_true")
    parser.add_argument("--optimize-joint-trajectory", action="store_true")
    parser.add_argument("--joint-types", choices=("auto", "revolute", "prismatic", "screw"), default="auto")
    parser.add_argument("--joint-topology", choices=("auto", "known"), default="known")
    parser.add_argument("--joint-compare-iters", type=int, default=12)
    parser.add_argument("--joint-compare-batch-size", type=int, default=4)
    parser.add_argument("--joint-compare-validation-cameras", type=int, default=2)
    parser.add_argument("--joint-trajectory-mode", choices=("monotonic", "profile_scale", "per_frame", "spline"), default="monotonic")
    parser.add_argument("--joint-trajectory-iters", type=int, default=200)
    parser.add_argument("--joint-trajectory-batch-size", type=int, default=8)
    parser.add_argument("--joint-trajectory-acceleration-weight", type=float, default=1.0e-5)
    parser.add_argument("--joint-trajectory-jerk-weight", type=float, default=1.0e-8)
    parser.add_argument("--joint-trajectory-reverse-weight", type=float, default=1.0e-2)
    parser.add_argument("--curl-grid-target-cells", type=int, default=4096)
    parser.add_argument("--ensure-sam3-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-sam3-feature-field", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ensure-sam3-masks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ensure-sam3-affinity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sam3-feature-root", default="")
    parser.add_argument("--sam3-mask-root", default="")
    parser.add_argument("--sam3-feature-dim", type=int, default=32)
    parser.add_argument("--sam3-affinity-iters", type=int, default=500)
    parser.add_argument("--sam3-affinity-dim", type=int, default=16)
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument("--sam3-checkpoint", default="")
    parser.add_argument("--sam3-allow-download", action="store_true")
    parser.add_argument("--sam3-mask-debug-overlays", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.prior_smooth_knn = max(1, int(args.prior_smooth_knn))
    args.prior_smooth_iters = max(0, int(args.prior_smooth_iters))
    args.prior_smooth_alpha = min(1.0, max(0.0, float(args.prior_smooth_alpha)))
    args.kpart_k = max(1, int(args.kpart_k))
    args.kpart_cycles = max(0, int(args.kpart_cycles))
    args.kpart_anneal_cycles = max(1, int(args.kpart_anneal_cycles))
    args.kpart_plateau_cycles = max(1, int(args.kpart_plateau_cycles))
    args.kpart_confidence_threshold = min(1.0, max(0.0, float(args.kpart_confidence_threshold)))
    args.kpart_dynamic_prior_weight = max(0.0, float(args.kpart_dynamic_prior_weight))
    args.kpart_prior_boundary_expansion = max(0, int(args.kpart_prior_boundary_expansion))
    args.kpart_prior_anchor_threshold = min(1.0, max(0.5, float(args.kpart_prior_anchor_threshold)))
    args.kpart_prior_cleanup_min_size = max(1, int(args.kpart_prior_cleanup_min_size))
    args.joint_trajectory_acceleration_weight = max(0.0, float(args.joint_trajectory_acceleration_weight))
    args.joint_trajectory_jerk_weight = max(0.0, float(args.joint_trajectory_jerk_weight))
    args.joint_trajectory_reverse_weight = max(0.0, float(args.joint_trajectory_reverse_weight))
    args.closed_loop_fm_iters = max(1, int(args.closed_loop_fm_iters))
    args.closed_loop_fm_batch_size = max(1, int(args.closed_loop_fm_batch_size))
    args.closed_loop_fm_knn = max(1, int(args.closed_loop_fm_knn))
    args.closed_loop_chunk_size = max(1, int(args.closed_loop_chunk_size))
    args.closed_loop_video_max_cameras = max(1, int(args.closed_loop_video_max_cameras))
    args.closed_loop_video_max_pixels = max(1, int(args.closed_loop_video_max_pixels))
    args.closed_loop_video_frame_stride = max(1, int(args.closed_loop_video_frame_stride))
    args.closed_loop_video_pixel_candidates = max(1, int(args.closed_loop_video_pixel_candidates))
    args.closed_loop_video_pixel_search_radius = max(1, int(args.closed_loop_video_pixel_search_radius))
    args.closed_loop_video_topk_gaussians = max(1, int(args.closed_loop_video_topk_gaussians))
    args.closed_loop_video_max_pixel_radius = max(1.0e-6, float(args.closed_loop_video_max_pixel_radius))
    args.closed_loop_video_depth_sigma = max(1.0e-8, float(args.closed_loop_video_depth_sigma))
    args.closed_loop_video_likelihood_weight = max(0.0, float(args.closed_loop_video_likelihood_weight))
    if bool(args.optimize_joint_trajectory):
        args.fit_joints = True

    torch.manual_seed(int(args.seed))
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    methods = _split_methods(args.methods)
    _resolve_aux_paths(args, output)
    if bool(args.ensure_sam3_affinity):
        args.ensure_sam3_masks = True
    if bool(args.train_sam3_feature_field):
        args.ensure_sam3_features = True
    _ensure_sam3_caches(args)
    static_ply = _ensure_static_model(args)
    needs_transport = "endpoint_transport_debug" in methods or "flow_matching" in methods
    needs_endpoint = needs_transport or "strict_closed_loop_rgbd" in methods
    endpoint_static_ply = None
    if needs_endpoint:
        endpoint_static_ply = _ensure_endpoint_model(args, output)
    if bool(args.ensure_sam3_affinity):
        _ensure_canonical_sam3_affinity(args)
    if "denoise" in methods:
        _ensure_denoise_discovery(args, output)
    _write_json(
        output / "pipeline_order.json",
        {
            "schema_version": 1,
            "order": [
                "prepare_sam3_input_caches_if_requested",
                "train_or_rebuild_frame0_static_3dgs",
                "train_or_rebuild_endpoint_static_3dgs" if needs_endpoint else "skip_endpoint_static_3dgs",
                "train_canonical_sam3_affinity_if_requested",
                "for_each_branch_build_static_dynamic_prior",
                "strict_closed_loop_rgbd_probe" if "strict_closed_loop_rgbd" in methods else "skip_strict_closed_loop_rgbd_probe",
                "for_each_branch_run_stage1_k_moving_part_discovery",
                "for_each_branch_fit_eulerian_or_lagrangian_killing_fields",
                "optionally_fit_revolute_prismatic_joints",
                "optionally_reoptimize_joint_trajectory",
            ],
            "dynamic_branches": methods,
            "static_ply": str(static_ply),
            "endpoint_static_ply": None if endpoint_static_ply is None else str(endpoint_static_ply),
            "prior_smoothing": {
                "knn": int(args.prior_smooth_knn),
                "iters": int(args.prior_smooth_iters),
                "alpha": float(args.prior_smooth_alpha),
            },
            "prior_seed": {
                "input": "branch_prior",
                "dynamic_threshold": 0.5,
                "role": "quick static/dynamic diagnostic only; real Stage-1 K-part runs inside each dynamic branch when --run-killing is set",
            },
            "branch_dynamic_training": {
                "enabled": bool(args.run_killing),
                "stage1_kpart_enabled_for": [
                    name
                    for name in methods
                    if name in {"denoise", "flow_matching", "strict_closed_loop_rgbd"} and bool(args.run_killing and args.run_kpart)
                ],
                "baseline_stage": "plain_all_foreground_killing_no_kpart_no_joint",
                "prior_branch_stage": "prior_probabilities_to_k_static_clustering_then_killing_then_optional_joint_trajectory",
                "support": str(args.phase1_motion_representation),
                "joint_fit_enabled": bool(args.fit_joints),
                "joint_trajectory_reoptimization_enabled": bool(args.optimize_joint_trajectory),
                "joint_candidate_policy": str(args.joint_types),
            },
        },
    )

    canonical = _load_gaussian_ply(static_ply, args.sh_degree)
    canonical_xyz = canonical.get_canonical_xyz.detach().cpu().float()
    gaussian_ids = _foreground_ids(canonical, args.semantic_dynamic_threshold, args.semantic_dynamic_min_points)
    source_xyz = canonical_xyz.index_select(0, gaussian_ids)
    source_scale = canonical.get_scaling.detach().cpu().float().amax(dim=1).index_select(0, gaussian_ids)
    shared = output / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    torch.save(gaussian_ids, shared / "gaussian_ids.pt")
    gt_mask = _load_gt_mask(args, canonical_xyz, static_ply)
    if gt_mask is not None:
        torch.save(gt_mask, shared / "gt_moving_mask.pt")
    killing_config = {
        "phase1_motion_representation": args.phase1_motion_representation,
        "static_model_dir": str(Path(args.static_model_dir).expanduser().resolve()),
        "static_iteration": int(args.static_iteration),
        "semantic_dynamic_threshold": float(args.semantic_dynamic_threshold),
        "freeze_prior_labels": bool(args.freeze_prior_labels),
        "prior_use_soft_weights": bool(args.prior_use_soft_weights),
        "prior_hard_threshold": float(args.prior_hard_threshold),
    }
    _write_json(shared / "killing_config.json", killing_config)
    torch.save(
        {
            "schema_version": 1,
            "note": "train.py builds the shared motion correspondence observations inside the shared Killing path",
            "root_scene": str(Path(args.root_scene).expanduser().resolve()),
            "killing_config_hash": _hash_payload(killing_config),
            "gaussian_ids_hash": _hash_payload(gaussian_ids),
        },
        shared / "motion_observations.pt",
    )
    _write_json(
        shared / "canonical_summary.json",
        {
            "num_canonical_gaussians": int(canonical_xyz.shape[0]),
            "num_foreground_gaussians": int(gaussian_ids.numel()),
            "gaussian_ids_hash": _hash_payload(gaussian_ids),
            "ground_truth_available": gt_mask is not None,
        },
    )

    priors: dict[str, DynamicPrior] = {}
    metric_rows: list[dict[str, Any]] = []

    if "baseline" in methods:
        priors["baseline"] = make_all_gaussians_prior(int(canonical_xyz.shape[0]))
    if "all_fg" in methods:
        priors["all_fg"] = make_all_foreground_prior(gaussian_ids)
    if "gt" in methods and gt_mask is not None:
        priors["gt"] = make_gt_prior(gaussian_ids, gt_mask)
    if "denoise" in methods:
        priors["denoise"] = _denoise_prior(args, gaussian_ids, int(canonical_xyz.shape[0]))

    endpoint_result = None
    endpoint_target_xyz = None
    if needs_transport:
        endpoint_ply = Path(args.endpoint_ply).expanduser() if args.endpoint_ply else None
        if endpoint_ply is None or not endpoint_ply.is_file():
            if not args.endpoint_model_dir:
                raise FileNotFoundError(
                    "--endpoint-model-dir or --endpoint-ply is required for "
                    "endpoint_transport_debug/flow_matching"
                )
            endpoint_ply = _ply_from_model_dir(args.endpoint_model_dir, args.endpoint_iteration)
        endpoint = _load_gaussian_ply(endpoint_ply, args.sh_degree)
        endpoint_ids = _foreground_ids(endpoint, args.semantic_dynamic_threshold, args.semantic_dynamic_min_points)
        endpoint_target_xyz = endpoint.get_canonical_xyz.detach().cpu().float().index_select(0, endpoint_ids)
        transport_cfg = EndpointTransportConfig(
            knn=int(args.fm_endpoint_knn),
            temperature=float(args.fm_transport_temperature),
            robust_delta=float(args.fm_transport_robust_delta),
            motion_scale_mode=str(args.fm_motion_scale),
            confidence_distance_scale=1.0,
            confidence_entropy_scale=float(args.fm_confidence_entropy_scale),
            chunk_size=int(args.fm_transport_chunk_size),
            prior_prob_bias=float(args.fm_prior_prob_bias),
            prior_prob_temperature=float(args.fm_prior_prob_temperature),
        )
        transport_cfg_dict = dataclasses.asdict(transport_cfg)
        endpoint_dir = output / "shared" / "endpoint_transport"
        endpoint_dir.mkdir(parents=True, exist_ok=True)
        transport_cfg_path = endpoint_dir / "transport_config.json"
        # Skip transport if all cached files exist and config matches.
        _cached = None
        if not bool(getattr(args, "force_transport", False)):
            saved_cfg = json.loads(transport_cfg_path.read_text()) if transport_cfg_path.is_file() else {}
            if _cfg_matches(saved_cfg, transport_cfg_dict):
                _cached = _load_transport_result(endpoint_dir)
                if _cached is not None:
                    print("[comparison] endpoint transport cached — skipping recompute")
        if _cached is not None:
            endpoint_result = _cached
        else:
            endpoint_result = EndpointTransportPrior(
                endpoint_target_xyz,
                {"source_scale": source_scale},
                transport_cfg,
            ).compute(source_xyz, gaussian_ids)
            save_endpoint_transport_result(endpoint_result, endpoint_dir)
            save_dynamic_prior(endpoint_result.prior, endpoint_dir / "prior.pt")
            _write_json(transport_cfg_path, transport_cfg_dict)
        if "endpoint_transport_debug" in methods:
            endpoint_result.prior.method = "endpoint_transport_debug"
            priors["endpoint_transport_debug"] = endpoint_result.prior

    if "flow_matching" in methods:
        if endpoint_result is None or endpoint_target_xyz is None:
            raise RuntimeError("flow_matching requires endpoint transport result")
        fm_cfg = FlowMatchingConfig(
            hidden_dim=int(args.fm_hidden_dim),
            layers=int(args.fm_layers),
            lr=float(args.fm_lr),
            iters=int(args.fm_iters),
            batch_size=int(args.fm_batch_size),
            identity_weight=float(args.fm_identity_weight),
            smooth_weight=float(args.fm_smooth_weight),
            integration_steps=int(args.fm_integration_steps),
            integrator=str(args.fm_integrator),
            prior_prob_bias=float(args.fm_prior_prob_bias),
            prior_prob_temperature=float(args.fm_prior_prob_temperature),
            seed=int(args.seed),
        )
        fm_cfg_dict = dataclasses.asdict(fm_cfg)
        fm_cfg_dict["endpoint_transport_hash"] = _hash_payload(
            {
                "transport_cfg": transport_cfg_dict,
                "barycentric_target": endpoint_result.barycentric_target,
                "transport_confidence": endpoint_result.prior.confidence,
                "transport_valid": endpoint_result.prior.valid,
            }
        )
        fm_dir = output / "flow_matching"
        fm_dir.mkdir(parents=True, exist_ok=True)
        fm_checkpoint_path = fm_dir / "fm_checkpoint.pt"
        fm_integrated_path = fm_dir / "integrated_endpoint.pt"
        fm_cfg_path = fm_dir / "fm_config.json"
        # Skip FM training if checkpoint + integrated exist and config matches.
        _fm_loaded = False
        if not bool(getattr(args, "force_flow_matching", False)) and fm_checkpoint_path.is_file() and fm_integrated_path.is_file():
            saved_fm_cfg = json.loads(fm_cfg_path.read_text()) if fm_cfg_path.is_file() else {}
            if _cfg_matches(saved_fm_cfg, fm_cfg_dict, ignore={"seed"}):
                try:
                    ckpt = torch.load(fm_checkpoint_path, map_location="cpu", weights_only=False)
                    model = EndpointFlowMLP(hidden_dim=fm_cfg.hidden_dim, layers=fm_cfg.layers)
                    model.load_state_dict(ckpt.get("model_state_dict") or ckpt["state_dict"])
                    model.eval()
                    integrated = torch.load(fm_integrated_path, map_location="cpu", weights_only=False)
                    _fm_loaded = True
                    print("[comparison:flow_matching] loaded cached checkpoint — skipping training")
                except Exception as exc:
                    print(f"[comparison:flow_matching] checkpoint load failed ({exc}) — retraining")
        if not _fm_loaded:
            model, logs = train_endpoint_flow_matching(source_xyz, endpoint_target_xyz, endpoint_result, fm_cfg)
            integrated = integrate_flow(model, source_xyz, fm_cfg)
            torch.save(integrated, fm_integrated_path)
            torch.save(integrated - source_xyz, fm_dir / "displacement.pt")
            torch.save(flow_checkpoint_payload(model.cpu(), fm_cfg, logs), fm_checkpoint_path)
            _write_json(fm_dir / "training_log.json", logs)
            _write_json(fm_cfg_path, fm_cfg_dict)
        fm_prior = build_dynamic_prior_from_flow(
            source_xyz,
            integrated,
            gaussian_ids,
            fm_cfg,
            confidence=endpoint_result.prior.confidence,
            valid=endpoint_result.prior.valid,
        )
        priors["flow_matching"] = fm_prior

    if "strict_closed_loop_rgbd" in methods:
        priors["strict_closed_loop_rgbd"] = _run_strict_closed_loop_rgbd_prior(args, output)

    for method in methods:
        if method not in priors:
            continue
        prior_path, metrics = _save_branch_prior(args, method, priors[method], output, gt_mask, canonical_xyz)
        del prior_path
        metric_rows.append(metrics)

    _write_table(output / "comparison_table.csv", metric_rows)
    summary = {
        "schema_version": 1,
        "methods": methods,
        "num_foreground_gaussians": int(gaussian_ids.numel()),
        "ground_truth_available": gt_mask is not None,
        "correlation": _correlation(priors),
        "metrics": metric_rows,
    }
    _write_json(output / "comparison_summary.json", summary)
    _write_prior_html(output / "prior_comparison.html", canonical_xyz, priors, gt_mask, int(args.max_html_points), int(args.seed))
    if endpoint_result is not None:
        _write_transport_html(
            output / "endpoint_transport_arrows.html",
            source_xyz,
            endpoint_result.barycentric_target,
            endpoint_result.prior.probability,
            endpoint_result.prior.confidence,
            int(args.max_html_points),
            int(args.seed),
        )
    if "denoise" in priors:
        other_name = (
            "flow_matching"
            if "flow_matching" in priors
            else "endpoint_transport_debug"
            if "endpoint_transport_debug" in priors
            else None
        )
        if other_name is not None:
            _write_disagreement_html(
                output / "denoise_vs_fm_disagreement.html",
                canonical_xyz,
                priors["denoise"],
                priors[other_name],
                other_name,
                int(args.max_html_points),
                int(args.seed),
            )
    _write_histograms(output / "prior_score_histograms.png", priors, gt_mask)
    _write_prior_scatter(output / "denoise_vs_fm_prior_scatter.png", priors, gt_mask)
    failed = [
        str(row.get("method", "unknown"))
        for row in metric_rows
        if row.get("killing_status") == "failed"
    ]
    if failed:
        print(f"[comparison] failed dynamic branches: {', '.join(failed)}")
        sys.exit(1)
    print(f"[comparison] complete: {output}")


if __name__ == "__main__":
    main()
