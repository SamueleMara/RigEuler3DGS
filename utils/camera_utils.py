#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
from pathlib import Path
import os
import cv2
import torch

WARNED = False

MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def object_mask_root(args):
    """Absolute directory holding the object masks, or None when --masks is unset.

    Read relative to --source_path, the way --images and --depths are, and
    taken as given when absolute. The pipeline's scenes keep their masks at
    <scene>/mask while COLMAP lives at <scene>/colmap, so the absolute form is
    what the endpoint trainer passes: the synthesised frame_t scene has its own
    colmap/ but no mask/ of its own.
    """
    masks = getattr(args, "masks", "") or ""
    if not masks:
        return None
    root = Path(masks)
    if not root.is_absolute():
        root = Path(getattr(args, "source_path", "")) / root
    root = Path(os.path.normpath(str(root)))
    if not root.is_dir():
        raise FileNotFoundError(
            f"--masks points at {root}, which is not a directory. Give a path "
            "relative to --source_path, or an absolute one.")
    return root


def image_root_candidates(args):
    """Directories an image path may be relative to, most specific first."""
    source = str(getattr(args, "source_path", "") or "")
    if not source:
        return []
    images_dir = str(getattr(args, "images", "") or "images")
    roots = []
    try:
        from scene.dynamic_sequence_dataset import infer_colmap_source_path
        roots.append(Path(infer_colmap_source_path(source)) / images_dir)
    except Exception:
        pass
    roots.append(Path(source) / images_dir)
    roots.append(Path(source))
    seen, unique = set(), []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def resolve_object_mask_path(image_path, args, mask_root):
    """The one mask belonging to this image, or an exception.

    Matching is on the path RELATIVE to the image root, never on the basename.
    Every camera in these datasets names its frames identically --
    cam_000/frame_000000.png and cam_001/frame_000000.png -- so a basename
    match would hand one camera another camera's silhouette and the error
    would look like a calibration problem, not a lookup bug.
    """
    image_path = Path(image_path)
    roots = image_root_candidates(args)
    rel = None
    # abspath BEFORE resolve, and never only resolve: the synthesised frame_t
    # scenes fill colmap/images/ with symlinks back into <scene>/rgb/, so
    # resolving first silently rewrites cam_004/frame_000000.png into
    # rgb/cam_004/frame_000000.png and then looks for mask/rgb/cam_004/... .
    # The link's own path is the one the mask tree mirrors.
    for normalise in (os.path.abspath, lambda q: str(Path(q).resolve())):
        for root in roots:
            try:
                rel = Path(normalise(str(image_path))).relative_to(
                    Path(normalise(str(root))))
                break
            except (ValueError, OSError):
                continue
        if rel is not None:
            break
    if rel is None:
        raise ValueError(
            f"Cannot place {image_path} under any image root "
            f"({', '.join(str(r) for r in roots) or 'none known'}). The mask "
            "tree mirrors the image tree, so an image outside it has no "
            "unambiguous mask. Move it under the root, or drop --masks.")
    found = []
    for suffix in (image_path.suffix, *MASK_SUFFIXES):
        if not suffix:
            continue
        candidate = mask_root / rel.with_suffix(suffix)
        if candidate.is_file() and candidate.resolve() not in {f.resolve() for f in found}:
            found.append(candidate)
    if not found:
        raise FileNotFoundError(
            f"No mask for {image_path}. Expected {mask_root / rel}. --masks "
            "requires one for every image: there is no all-ones fallback, "
            "because a silent one supervises the whole background as object "
            "and is indistinguishable from the bug it would cause.")
    if len(found) > 1:
        raise ValueError(
            f"Ambiguous mask for {image_path}: "
            + ", ".join(str(f) for f in found)
            + ". Keep one file per image.")
    return found[0]


def load_object_alpha_premult(image, image_path, mask_path, resolution):
    """Ground-truth alpha and premultiplied colour at the training resolution.

    Background pixels in these renders are black, and black is not the
    object's colour there -- it means "nothing was seen". Downsampling the
    blacked-out RGB and the mask as two separate images averages that black
    into the object's edge, which is the dark halo the border Gaussians then
    faithfully learn. Premultiplying FIRST and resizing colour and coverage
    with the same kernel keeps `premul + (1 - alpha) * b` equal to the area
    average of the per-pixel composite, so the edge carries no colour the
    camera never recorded.

    Returns (premultiplied RGB [3,H,W], alpha [1,H,W]), both float32 in [0,1].
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if mask.shape != rgb.shape[:2]:
        raise ValueError(
            "Mask and image disagree on size.\n"
            f"  image {image_path} -> {rgb.shape[1]}x{rgb.shape[0]}\n"
            f"  mask  {mask_path} -> {mask.shape[1]}x{mask.shape[0]}\n"
            "They must match at source resolution; both are resized together "
            "afterwards.")
    off_binary = np.unique(mask[(mask != 0) & (mask != 255)])
    if off_binary.size:
        shown = ", ".join(str(int(v)) for v in off_binary[:8])
        raise ValueError(
            f"{mask_path} is not binary: it also holds {shown}"
            + (", and more" if off_binary.size > 8 else "")
            + ". White is foreground, black is background. Nothing is "
              "thresholded here on purpose -- a guessed cut silently decides "
              "which pixels are supervised as object.")

    alpha0 = mask.astype(np.float32) / 255.0
    premul0 = rgb * alpha0[..., None]

    target_w, target_h = int(resolution[0]), int(resolution[1])
    src_h, src_w = alpha0.shape
    if (target_w, target_h) == (src_w, src_h):
        premul, alpha = premul0, alpha0
    else:
        # INTER_AREA downward so edge alpha means coverage; never INTER_NEAREST,
        # which would snap a half-covered edge pixel to fully object or fully
        # background and put the halo straight back.
        interp = cv2.INTER_AREA if (target_w < src_w or target_h < src_h) else cv2.INTER_LINEAR
        premul = cv2.resize(premul0, (target_w, target_h), interpolation=interp)
        alpha = cv2.resize(alpha0, (target_w, target_h), interpolation=interp)

    premul_t = torch.from_numpy(
        np.ascontiguousarray(premul.transpose(2, 0, 1))).float().clamp_(0.0, 1.0)
    alpha_t = torch.from_numpy(
        np.ascontiguousarray(alpha[None])).float().clamp_(0.0, 1.0)
    if not (torch.isfinite(premul_t).all() and torch.isfinite(alpha_t).all()):
        raise ValueError(f"Non-finite values after resizing {mask_path}")
    return premul_t, alpha_t

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    image = Image.open(cam_info.image_path)

    if cam_info.depth_path != "":
        try:
            depth_path = str(cam_info.depth_path)
            if depth_path.endswith(".npy"):
                depth_arr = np.load(depth_path).astype(np.float32)
                if depth_arr.ndim == 3:
                    depth_arr = depth_arr[..., 0]
                if getattr(args, "sapien_depth_is_metric", False):
                    valid = depth_arr > 1e-8
                    invdepthmap = np.zeros_like(depth_arr, dtype=np.float32)
                    invdepthmap[valid] = 1.0 / depth_arr[valid]
                else:
                    _v = depth_arr[depth_arr > 1e-8]
                    if _v.size and float(np.median(_v)) > 1.0:
                        raise ValueError(
                            f"{depth_path} has a median of {float(np.median(_v)):.3f} "
                            "among its valid pixels. Inverse depth that large "
                            "would put the surface 1/3 of a metre away over the "
                            "whole image, so this is almost certainly metres. "
                            "Pass --sapien_depth_is_metric, or the depth term "
                            "trains against a map it has misread by a "
                            "reciprocal.")
                    invdepthmap = depth_arr
            elif depth_path.endswith(".npz"):
                data = np.load(depth_path)
                key = "depth" if "depth" in data else list(data.keys())[0]
                depth_arr = data[key].astype(np.float32)
                if depth_arr.ndim == 3:
                    depth_arr = depth_arr[..., 0]
                if getattr(args, "sapien_depth_is_metric", False):
                    valid = depth_arr > 1e-8
                    invdepthmap = np.zeros_like(depth_arr, dtype=np.float32)
                    invdepthmap[valid] = 1.0 / depth_arr[valid]
                else:
                    invdepthmap = depth_arr
            elif is_nerf_synthetic:
                invdepthmap = cv2.imread(depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(depth_path, -1).astype(np.float32) / float(2**16)
                if getattr(args, "sapien_depth_is_metric", False):
                    valid = invdepthmap > 1e-8
                    converted = np.zeros_like(invdepthmap, dtype=np.float32)
                    converted[valid] = 1.0 / invdepthmap[valid]
                    invdepthmap = converted

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    cam = Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                 FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                 image=image, invdepthmap=invdepthmap,
                 image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                 train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

    # Optional dynamic-sequence metadata. The original Camera class is unchanged;
    # these attributes are attached only when provided by the dataset reader.
    for attr in ["camera_id", "frame_id", "timestamp", "raw_timestamp", "mask_path", "depth_path"]:
        if hasattr(cam_info, attr):
            setattr(cam, attr, getattr(cam_info, attr))

    # Object masks as ground-truth alpha. Attached only when --masks is given;
    # without it every attribute below stays None and the camera is exactly
    # what it was before. This is NOT cam.alpha_mask, which stays the
    # loss-validity mask for the --train_test_exp half-image split.
    mask_root = object_mask_root(args)
    if mask_root is not None:
        mask_path = resolve_object_mask_path(cam_info.image_path, args, mask_root)
        premult, alpha = load_object_alpha_premult(
            image, cam_info.image_path, mask_path, resolution)
        if tuple(alpha.shape[-2:]) != (cam.image_height, cam.image_width):
            raise ValueError(
                f"{mask_path} resized to {tuple(alpha.shape[-2:])} but the "
                f"camera is {(cam.image_height, cam.image_width)}")
        cam.object_rgb_premult = premult.to(cam.data_device)
        cam.object_alpha = alpha.to(cam.data_device)
        cam.object_mask_path = str(mask_path)
    return cam

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry