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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.dynamic_loss_utils import normalize_camera_timestamps
from scene.dynamic_sequence_dataset import (
    infer_colmap_source_path,
    dynamic_sequence_available,
    expand_fixed_colmap_cameras_to_dynamic_sequence,
)

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], dynamic_args=None):
        """Load the canonical GraphDECO scene and optionally its video cameras."""
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self._source_path = args.source_path
        self._scene_args = args
        self._resolution_scales = list(resolution_scales)
        self._base_train_cameras = {}
        self._base_test_cameras = {}

        # Keep the canonical initialization exactly in the vanilla 3DGS format.
        # If source_path points to ROOT, but ROOT/colmap/sparse exists, we load
        # the vanilla COLMAP scene from ROOT/colmap. If source_path already points
        # to ROOT/colmap, this is unchanged. Dynamic RGB/depth frames are wired
        # later by cloning these fixed COLMAP cameras over time.
        colmap_source_path = infer_colmap_source_path(args.source_path)
        if os.path.exists(os.path.join(colmap_source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](colmap_source_path, args.images, args.depths, args.eval, args.train_test_exp,
                load_gt_parts=getattr(args, 'load_gt_parts', True))
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type! Expected source_path/sparse or source_path/colmap/sparse."

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            base_train = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            base_test = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

            self._base_train_cameras[resolution_scale] = base_train
            self._base_test_cameras[resolution_scale] = base_test

            # Dynamic fixed-camera sequence expansion. This preserves vanilla
            # COLMAP initialization and only replaces the camera/image list used
            # for dynamic training. Every rgb/cam_xxx/frame_yyyyyy image reuses
            # the corresponding fixed COLMAP camera intrinsics/extrinsics.
            if dynamic_args is not None and getattr(dynamic_args, "dynamic_motion", False) and dynamic_sequence_available(args.source_path, dynamic_args):
                train_dyn, test_dyn, expanded = expand_fixed_colmap_cameras_to_dynamic_sequence(
                    base_train, base_test, args.source_path, args, dynamic_args, resolution_scale
                )
                self.train_cameras[resolution_scale] = train_dyn if expanded else base_train
                self.test_cameras[resolution_scale] = test_dyn if expanded else base_test
            else:
                self.train_cameras[resolution_scale] = base_train
                self.test_cameras[resolution_scale] = base_test

            # Attach normalized timestamps for any non-expanded fallback camera
            # list. Expanded dynamic cameras already receive exact frame-based
            # timestamps at construction time.
            fps = float(getattr(dynamic_args, "fps", 30.0)) if dynamic_args is not None else 30.0
            time_scale = float(getattr(dynamic_args, "time_scale", 1.0)) if dynamic_args is not None else 1.0
            time_mode = str(getattr(dynamic_args, "motion_time_mode", "seconds")) if dynamic_args is not None else "seconds"
            all_dynamic_cameras = list(self.train_cameras[resolution_scale]) + list(self.test_cameras[resolution_scale])
            normalize_camera_timestamps(
                all_dynamic_cameras, fps=fps, time_scale=time_scale, time_mode=time_mode
            )

        if self.loaded_iter:
            point_cloud_path = os.path.join(
                self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter)
            )
            self.gaussians.load_ply(
                os.path.join(point_cloud_path, "point_cloud.ply"), args.train_test_exp,
                load_gt_parts=getattr(args, 'load_gt_parts', True)
            )
            dynamic_path = os.path.join(point_cloud_path, "dynamic_motion.pth")
            if dynamic_args is not None and getattr(dynamic_args, "dynamic_motion", False) and os.path.exists(dynamic_path):
                import torch
                self.gaussians.load_dynamic_state(
                    torch.load(dynamic_path, map_location="cpu", weights_only=False), dynamic_args
                )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def enable_dynamic_sequence(self, args=None, dynamic_args=None):
        """Replace train/test camera lists by cloned dynamic RGB-D sequence cameras.

        This is used by the two-stage trainer after the static canonical 3DGS
        stage has finished. The Gaussian parameters are not reinitialized.
        """
        if args is None:
            args = self._scene_args
        if dynamic_args is None:
            return False
        changed_any = False
        for resolution_scale in self._resolution_scales:
            base_train = self._base_train_cameras[resolution_scale]
            base_test = self._base_test_cameras[resolution_scale]
            if dynamic_sequence_available(args.source_path, dynamic_args):
                train_dyn, test_dyn, expanded = expand_fixed_colmap_cameras_to_dynamic_sequence(
                    base_train, base_test, args.source_path, args, dynamic_args, resolution_scale
                )
                if expanded:
                    self.train_cameras[resolution_scale] = train_dyn
                    self.test_cameras[resolution_scale] = test_dyn
                    changed_any = True
            fps = float(getattr(dynamic_args, "fps", 30.0))
            time_scale = float(getattr(dynamic_args, "time_scale", 1.0))
            time_mode = str(getattr(dynamic_args, "motion_time_mode", "seconds"))
            all_dynamic_cameras = list(self.train_cameras[resolution_scale]) + list(self.test_cameras[resolution_scale])
            normalize_camera_timestamps(
                all_dynamic_cameras, fps=fps, time_scale=time_scale, time_mode=time_mode
            )
        return changed_any

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

        # Save dynamic state next to the canonical point cloud when enabled. The
        # canonical PLY remains compatible with the original 3DGS repo.
        dynamic_state = self.gaussians.capture_dynamic_state()
        if dynamic_state is not None:
            import torch
            torch.save(dynamic_state, os.path.join(point_cloud_path, "dynamic_motion.pth"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
