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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        # The extension is resolved, not assumed. These datasets store depth as
        # float32 .npy in metres; the .png beside it is a normalised
        # visualisation (0.76-1.0 after /2**16 on scissor), not a measurement.
        # Hardcoding .png here is why --depths could never reach depth_npy/.
        depth_path = ""
        if depths_folder != "":
            stem = os.path.join(depths_folder, extr.name[:-n_remove])
            for ext in (".npy", ".npz", ".png"):
                if os.path.exists(stem + ext):
                    depth_path = stem + ext
                    break
            if depth_path == "":
                depth_path = stem + ".png"

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def _load_matching_part_indices(path, positions):
    """Load exact sparse GT part labels for a COLMAP PLY when the dataset has them.

    The SAPIEN/processed inputs keep semantic part labels in
    ``<scene>/pointcloud/points3D_parts.ply`` while the 3DGS loader reads
    ``<scene>/colmap/sparse/0/points3D.ply``.  For correct Gaussian lineage we
    attach the part labels at initialization, before clone/split densification.
    """
    prop_names = None
    try:
        plydata = PlyData.read(path)
        vertices = plydata["vertex"]
        if "part" in {p.name for p in vertices.properties}:
            return np.asarray(vertices["part"], dtype=np.int64)
    except Exception:
        return None

    ply_path = Path(path).resolve()
    # The FINAL endpoint keeps its labels under a different name. A scene
    # exported with both endpoints has:
    #
    #     pointcloud/points3D_parts.ply          labels for colmap/
    #     pointcloud/points3D_parts_final.ply    labels for colmap_final/
    #
    # Looking only for the first leaves every final-endpoint model with no
    # GT at all, which does not affect training -- GT is diagnostics only --
    # but empties the GT and metric panels of every debug page built from it.
    # Which name to prefer is decided by the COLMAP directory the model was
    # read from, never by guessing.
    names = ["points3D_parts.ply"]
    if any(part == "colmap_final" for part in ply_path.parts):
        names.insert(0, "points3D_parts_final.ply")
    candidates = [ply_path.with_name(n) for n in names]
    for parent in ply_path.parents:
        candidates.extend(parent / "pointcloud" / n for n in names)

    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            part_ply = PlyData.read(candidate)
            part_vertices = part_ply["vertex"]
            part_props = {p.name for p in part_vertices.properties}
            if "part" not in part_props:
                continue
            part_xyz = np.vstack(
                [part_vertices["x"], part_vertices["y"], part_vertices["z"]]
            ).T.astype(np.float32)
            parts = np.asarray(part_vertices["part"], dtype=np.int64)
        except Exception:
            continue
        if part_xyz.shape[0] != positions.shape[0]:
            continue
        ordered_error = np.linalg.norm(part_xyz - positions, axis=1)
        if ordered_error.size > 0 and float(ordered_error.max()) <= 1.0e-4:
            print(f"[GT parts] attached sparse part labels from {candidate}")
            return parts

        # Last safe fallback: same point set, different order.  This remains
        # exact only when every nearest sparse point is unique and within the
        # same numerical tolerance.
        if positions.shape[0] > 10000:
            continue
        diff = positions[:, None, :] - part_xyz[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        nn = np.argmin(dist2, axis=1)
        nn_dist = np.sqrt(dist2[np.arange(dist2.shape[0]), nn])
        if float(nn_dist.max()) <= 1.0e-4 and np.unique(nn).size == positions.shape[0]:
            print(f"[GT parts] attached reordered sparse part labels from {candidate}")
            return parts[nn]
    return None


def fetchPly(path, load_gt_parts=True):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    prop_names = {p.name for p in vertices.properties}
    if {"nx", "ny", "nz"}.issubset(prop_names):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    part_indices = (_load_matching_part_indices(path, positions.astype(np.float32))
                    if load_gt_parts else None)
    return BasicPointCloud(points=positions, colors=colors, normals=normals, part_indices=part_indices)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def _resolve_depths_folder(path, depths):
    """Find the depth directory, under the COLMAP root or beside it.

    `path` here is the COLMAP root, so the obvious join gives
    `<scene>/colmap/depth_npy`. These datasets put the depth one level up, at
    `<scene>/depth_npy`, next to `colmap/` rather than inside it -- which is
    also where every other tool in this repository reads it from. Checking
    both means `--depths depth_npy` works without the caller having to know
    which layout it is looking at, and a wrong name still fails loudly
    instead of silently resolving to nothing.
    """
    if depths == "":
        return ""
    for cand in (os.path.join(path, depths),
                 os.path.join(os.path.dirname(os.path.abspath(path)), depths)):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        f"--depths {depths!r} matches no directory: looked in "
        f"{os.path.join(path, depths)} and "
        f"{os.path.join(os.path.dirname(os.path.abspath(path)), depths)}")


def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8, load_gt_parts=True):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            # Not fatal. depth_params.json carries the per-image scale/offset
            # that aligns MONOCULAR estimated inverse depth to the COLMAP
            # scale. Depth that is already metric -- a sensor, or a simulator
            # export -- needs no alignment, and refusing to train without the
            # file blocks exactly the case that needs it least.
            print(f"No depth_params.json at '{depth_params_file}'; using the "
                  f"depth maps as they are. That is correct for metric depth "
                  f"and wrong for unscaled monocular depth.")
            depths_params = None
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=_resolve_depths_folder(path, depths),
        test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path, load_gt_parts=load_gt_parts)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
