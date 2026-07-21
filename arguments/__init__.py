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

from argparse import ArgumentParser, Namespace
import sys
import os
from pathlib import Path

# Global DINOv3 defaults used by train.py and the extractor.
DINOv3_HF_MODEL = os.environ.get(
    "DINOV3_HF_MODEL",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
)
DINOv3_HF_LOCAL_DIR = os.environ.get(
    "DINOV3_HF_LOCAL_DIR",
    str(Path.home() / ".cache" / "huggingface" / "dinov3-vitl16-pretrain-lvd1689m"),
)
DINOv3_BACKEND = os.environ.get("DINOV3_BACKEND", "transformers")
DINOv3_IMAGE_SIZE = int(os.environ.get("DINOV3_IMAGE_SIZE", "512"))

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def add_dynamic_args(parser):
    """Small user-facing argument surface for generic dynamic scenes.

    Expected scene root:
        ROOT/
          colmap/
          rgb/cam_XXX/frame_YYYYYY.png
          depth_npy/cam_XXX/frame_YYYYYY.npy      optional
          depth_png/cam_XXX/frame_YYYYYY.png      optional
          dinov3/cam_XXX/frame_YYYYYY.pt          optional/precomputed

    Preferred usage:
        --root_scene ROOT

    Then the code internally sets:
        source_path = ROOT/colmap

    The original -s/--source_path still exists for static 3DGS compatibility,
    but for dynamic scenes you should pass only --root_scene.
    """
    group = parser.add_argument_group("Generic Dynamic Scene")

    # Main switches.
    group.add_argument("--dynamic_motion", action="store_true", default=False,
                       help="Enable static-then-dynamic Eulerian training.")
    group.add_argument("--root_scene", type=str, default="",
                       help="Scene root containing colmap/, rgb/, depth_npy/, depth_png/, metadata/, pointcloud/. If set, -s is automatically ROOT/colmap.")

    # Iteration split.
    group.add_argument("--static_train_iters", type=int, default=30000,
                       help="Static canonical 3DGS iterations.")
    group.add_argument("--euler_field_train_iters", type=int, default=15000,
                       help="Dynamic Eulerian field iterations. Default 15000 gives a real dynamic stage after the 30000 static stage.")

    # Resume from a saved static GraphDeco PLY. This is different from
    # --start_checkpoint, which only accepts chkpnt*.pth files.
    group.add_argument("--start_ply", type=str, default="",
                       help="Resume dynamic training from a saved static point_cloud.ply instead of a chkpnt*.pth checkpoint.")
    group.add_argument("--start_ply_iteration", type=int, default=-1,
                       help="Iteration associated with --start_ply. -1 = use --static_train_iters.")

    group.add_argument("--force_static_retrain", action="store_true", default=False,
                       help="Ignore an existing iteration_<static_train_iters>/point_cloud.ply and retrain the static stage from COLMAP.")
    group.add_argument("--no_auto_static_resume", action="store_true", default=False,
                       help="Disable automatic resume from an existing static PLY in the output folder.")

    # Generic sequence controls.
    group.add_argument("--fps", type=float, default=30.0)
    group.add_argument("--time_scale", type=float, default=1.0)
    group.add_argument("--cameras", type=str, default="",
                       help="Comma-separated cameras, e.g. cam_000,cam_001. Empty = all.")
    group.add_argument("--frame_start", type=int, default=0)
    group.add_argument("--frame_end", type=int, default=-1,
                       help="Inclusive maximum frame id. -1 = all.")
    group.add_argument("--frame_stride", type=int, default=1)
    group.add_argument("--test_stride", type=int, default=0,
                       help="If >0, every Nth frame per camera goes to test set.")

    # Optional modalities.
    group.add_argument("--use_depth", action="store_true", default=True,
                       help="Use ROOT/depth_npy or ROOT/depth_png as depth supervision.")
    group.add_argument("--no_depth", action="store_true", default=False,
                       help="Disable depth supervision. Depth is enabled by default for this pipeline.")
    group.add_argument("--depth_format", type=str, default="npy", choices=["npy", "png"],
                       help="Depth folder to use when --use_depth is active.")
    group.add_argument("--use_dinov3", action="store_true", default=True,
                       help="Use ROOT/dinov3 dense feature tensors.")
    group.add_argument("--extract_dinov3", action="store_true", default=False,
                       help="Extract DINOv3 features at training start if missing.")
    group.add_argument("--dinov3_weights", type=str, default="",
                       help="Local DINOv3 backbone checkpoint path. Empty = auto-detect ~/.cache/torch/hub/checkpoints/*dinov3*.pth. No online download by default.")
    group.add_argument("--dinov3_cache_dir", type=str, default="",
                       help="Optional local directory where DINOv3 checkpoints are searched.")
    group.add_argument("--dinov3_allow_download", action="store_true", default=False,
                       help="Explicitly allow torch.hub online pretrained checkpoint download. Default off because DINOv3 URL may return HTTP 403.")
    group.add_argument("--dinov3_backend", type=str, default=DINOv3_BACKEND, choices=["transformers", "torchhub"],
                       help="DINO feature extraction backend. transformers uses HF model repos; torchhub uses local submodules/dinov3 and .pth weights.")
    group.add_argument("--dinov3_hf_model", type=str, default=DINOv3_HF_MODEL,
                       help="HF Transformers model id for DINOv3 feature extraction.")
    group.add_argument("--dinov3_hf_local_dir", type=str, default=DINOv3_HF_LOCAL_DIR,
                       help="Local HF snapshot directory. Defaults to ~/.cache/huggingface/dinov3-vitl16-pretrain-lvd1689m or DINOV3_HF_LOCAL_DIR.")
    group.add_argument("--dinov3_full_resolution", action="store_true", default=True,
                       help="Save extracted DINO targets as whole-image dense maps [D,H,W]. Default true.")
    group.add_argument("--dinov3_patch_resolution", action="store_true", default=False,
                       help="Store patch-grid DINO targets instead of full-image dense maps.")
    group.add_argument("--dinov3_target_dim", type=int, default=32,
                       help="Number of DINO channels saved for full-image targets. Recommended: same as --dino_feature_field_dim.")
    group.add_argument("--dinov3_save_fp32", action="store_true", default=False,
                       help="Save DINO targets in float32. Default saves float16.")
    group.add_argument("--dinov3_required", action="store_true", default=False,
                       help="If set, abort when DINOv3 feature extraction fails. Default: warn, disable DINOv3, and continue dynamic training.")

    # W&B logging. Authentication is intentionally external: run `wandb login`
    # or export WANDB_API_KEY before launching train.py. Never hardcode keys.
    group.add_argument("--wandb", action="store_true", default=True,
                       help="Enable Weights & Biases logging. Default: enabled; use --no_wandb or --wandb_mode disabled to disable.")
    group.add_argument("--no_wandb", action="store_true", default=False,
                       help="Disable W&B logging even though it is enabled by default.")
    group.add_argument("--wandb_project", type=str, default="RigEuler3DGS",
                       help="W&B project name.")
    group.add_argument("--wandb_entity", type=str, default="samuelemara-my-institution",
                       help="Optional W&B entity/team. Empty = default account.")
    group.add_argument("--wandb_name", type=str, default="",
                       help="Optional W&B run name. Empty = model folder name.")
    group.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"],
                       help="W&B mode.")
    group.add_argument("--wandb_tags", type=str, default="",
                       help="Comma-separated W&B tags.")
    group.add_argument("--wandb_log_interval", type=int, default=10,
                       help="Log scalar metrics every N iterations.")
    group.add_argument("--wandb_log_artifacts", action="store_true", default=False,
                       help="Upload cfg, summaries, and HTML outputs as a W&B artifact at the end.")
    group.add_argument("--wandb_log_tracks", action="store_true", default=False,
                       help="Also upload gaussian_tracks.pt when --wandb_log_artifacts is active. This can be large.")

    # Optional post-training rendering / visualization. These are deliberately
    # off by default because rendering all cameras/frames can be expensive.
    group.add_argument("--render_after_training", action="store_true", default=False,
                       help="After dynamic training, render all selected dynamic frames/cameras using gaussian_tracks.pt when available.")
    group.add_argument("--render_save_video", action="store_true", default=False,
                       help="Also write one MP4 video per camera when --render_after_training is active.")
    group.add_argument("--render_no_tracks", action="store_true", default=False,
                       help="Render by integrating the in-memory Euler field instead of injecting exported gaussian_tracks.pt positions.")
    group.add_argument("--render_split", type=str, default="all", choices=["train", "test", "all"],
                       help="Camera split to render after training.")
    group.add_argument("--render_cameras", type=str, default="",
                       help="Comma-separated camera ids to render after training. Empty = inherit --cameras; if that is empty, all cameras.")
    group.add_argument("--render_frame_start", type=int, default=-2,
                       help="First frame to render. -2 = inherit --frame_start.")
    group.add_argument("--render_frame_end", type=int, default=-2,
                       help="Last frame to render. -2 = inherit --frame_end.")
    group.add_argument("--render_frame_stride", type=int, default=0,
                       help="Frame stride for rendering. 0 = inherit --frame_stride.")
    group.add_argument("--render_output_name", type=str, default="dynamic_renders",
                       help="Output folder name under model_path for post-training renders.")

    group.add_argument("--make_track_html", action="store_true", default=True,
                       help="After dynamic training, create an interactive Plotly HTML animation from gaussian_tracks.pt.")
    group.add_argument("--track_html_name", type=str, default="gaussian_tracks_visualization.html",
                       help="HTML filename under model_path.")
    group.add_argument("--track_html_max_points", type=int, default=30000,
                       help="Maximum number of Gaussians shown in the HTML visualization.")
    group.add_argument("--track_html_dynamic_threshold", type=float, default=0.5,
                       help="If dynamic_prob exists, classify dynamic when prob >= threshold. Otherwise threshold normalized max speed.")
    group.add_argument("--track_html_frame_stride", type=int, default=1,
                       help="Frame stride for the HTML animation.")
    group.add_argument("--track_html_label_mode", type=str, default="auto",
                       choices=["auto", "prob", "speed", "combined"],
                       help="How to label Gaussians in the track HTML. auto uses dynamic_prob only if it is non-degenerate, otherwise speed.")
    group.add_argument("--track_html_speed_percentile", type=float, default=80.0,
                       help="Percentile of max speed used by speed/auto track labelling.")

    group.add_argument("--make_velocity_html", action="store_true", default=True,
                       help="After dynamic training, create an interactive 3D velocity-field HTML heatmap from gaussian_tracks.pt.")
    group.add_argument("--velocity_html_name", type=str, default="velocity_field_visualization.html",
                       help="Velocity-field HTML filename under model_path.")
    group.add_argument("--velocity_html_max_points", type=int, default=30000,
                       help="Maximum number of Gaussian samples shown in the velocity-field HTML.")
    group.add_argument("--velocity_html_max_arrows", type=int, default=3000,
                       help="Maximum number of velocity arrows shown in the velocity-field HTML.")
    group.add_argument("--velocity_html_frame_stride", type=int, default=1,
                       help="Frame stride for the velocity-field HTML animation.")
    group.add_argument("--velocity_html_speed_clip_percentile", type=float, default=98.0,
                       help="Speed percentile used to clip the velocity heatmap color range.")
    group.add_argument("--velocity_html_arrow_scale", type=float, default=0.0,
                       help="Velocity arrow length scale. 0 = automatic scale from scene radius and max speed.")

    # Camera-frustum visualization. Camera centers and object points are always
    # kept in true world coordinates. The only display parameter is the frustum
    # depth, in world units/meters, matching the older point-cloud/joint HTML
    # visualization utilities.
    group.add_argument("--html_frustum_scale", type=float, default=0.12,
                       help="Camera frustum depth in world units/meters. Camera centers and object points are not rescaled.")
    group.add_argument("--html_no_camera_deduplicate", action="store_true", default=False,
                       help="Do not deduplicate repeated dynamic camera entries in cameras.json.")
    group.add_argument("--html_aspectmode", type=str, default="cube", choices=["cube", "data", "auto"],
                       help="Plotly 3D aspect mode for HTML outputs. cube matches older debug visualizers; data preserves metric visual aspect but shallow scenes look flat.")
    group.add_argument("--html_axis_scope", type=str, default="all", choices=["object", "cameras", "all"],
                       help="Which coordinates define the initial HTML axis range. Coordinates are never rescaled.")

    # Phase-boundary debug snapshots. These are for diagnosing whether points were
    # already stuck after dynamic/static labelling or became stuck after rigid motion losses.
    group.add_argument("--export_phase_debug_snapshots", action="store_true", default=True,
                       help="Export tracks, velocity HTML, and motion-grid state at phase boundaries.")
    group.add_argument("--disable_phase_debug_snapshots", action="store_true", default=False,
                       help="Disable phase-boundary debug snapshots.")
    group.add_argument("--phase_debug_dir", type=str, default="debug_phase_snapshots",
                       help="Subfolder inside model_path where phase debug snapshots are saved.")
    group.add_argument("--phase_debug_make_html", action="store_true", default=True,
                       help="Create track and velocity-field HTML files for each phase debug snapshot.")
    group.add_argument("--phase_debug_no_html", action="store_true", default=False,
                       help="Only save PT/JSON debug files, skip debug HTML generation.")
    group.add_argument("--phase_debug_html_max_points", type=int, default=30000,
                       help="Maximum points shown in phase debug track/velocity HTML.")
    group.add_argument("--phase_debug_html_frame_stride", type=int, default=1,
                       help="Frame stride used in phase debug HTML visualizations.")
    group.add_argument("--phase_debug_velocity_max_arrows", type=int, default=3000,
                       help="Maximum arrows shown in phase debug velocity-field HTML.")
    group.add_argument("--phase_debug_grid_octomap_html", action="store_true", default=True,
                       help="Create octomap-style HTML visualization of the adaptive motion grid and learned cell feature field for each phase snapshot.")
    group.add_argument("--phase_debug_no_grid_octomap_html", action="store_true", default=False,
                       help="Disable phase snapshot motion_grid_octomap.html generation.")
    group.add_argument("--phase_debug_grid_octomap_color_mode", type=str, default="feature_pca",
                       choices=["feature_pca", "feature_norm", "dynamic_prob", "semantic_prob", "mean_speed", "max_speed", "max_displacement", "score_ema", "level", "count"],
                       help="Color mode for motion_grid_octomap.html. feature_pca displays learned grid feature vectors as PCA RGB.")
    group.add_argument("--phase_debug_grid_octomap_max_cells", type=int, default=5000,
                       help="Maximum adaptive-grid cells shown in motion_grid_octomap.html.")
    group.add_argument("--phase_debug_grid_octomap_min_count", type=int, default=1,
                       help="Minimum Gaussian count per cell shown in motion_grid_octomap.html.")
    group.add_argument("--phase_debug_grid_octomap_opacity", type=float, default=0.24,
                       help="Cube opacity for motion_grid_octomap.html.")
    group.add_argument("--phase_debug_grid_octomap_include_inactive", action="store_true", default=False,
                       help="Also show inactive adaptive-grid cells in motion_grid_octomap.html.")
    group.add_argument("--phase_debug_deep_diagnostics", action="store_true", default=True,
                       help="Write deep per-point/per-cell phase diagnostics: seed distance/DINO similarity, rigid/static residuals, q_model, stuck masks.")
    group.add_argument("--phase_debug_no_deep_diagnostics", action="store_true", default=False,
                       help="Disable deep per-point/per-cell phase diagnostics.")
    group.add_argument("--phase_debug_write_csv", action="store_true", default=True,
                       help="Also write per_point_diagnostics.csv for quick spreadsheet inspection.")
    group.add_argument("--phase_debug_no_csv", action="store_true", default=False,
                       help="Skip per_point_diagnostics.csv and only write PT/JSON diagnostics.")
    group.add_argument("--phase_debug_seed_quantile", type=float, default=0.80,
                       help="Quantile of max-speed foreground points used as high-motion seeds for diagnostics.")
    group.add_argument("--phase_debug_seed_min_points", type=int, default=128,
                       help="Minimum number of high-motion seed points for diagnostic twist fitting.")
    group.add_argument("--phase_debug_seed_max_points", type=int, default=1024,
                       help="Maximum number of high-motion seed points for nearest-seed and twist diagnostics.")
    group.add_argument("--phase_debug_nearest_chunk", type=int, default=2048,
                       help="Chunk size for nearest high-motion seed distance computation.")
    group.add_argument("--phase_debug_q_temperature", type=float, default=-1.0,
                       help="Temperature for diagnostic q_model=sigmoid((e_static-e_rigid)/T). <=0 uses robust auto temperature.")
    group.add_argument("--phase_debug_twist_ridge", type=float, default=1e-6,
                       help="Ridge regularization for diagnostic per-frame twist fitting.")
    group.add_argument("--phase_debug_stuck_speed_threshold", type=float, default=0.05,
                       help="Max-speed threshold below which static foreground points are counted as dead/stuck.")
    group.add_argument("--phase_debug_stuck_disp_threshold", type=float, default=0.02,
                       help="Max-displacement threshold below which static foreground points are counted as dead/stuck.")
    group.add_argument("--phase_debug_moving_speed_threshold", type=float, default=0.10,
                       help="Max-speed threshold above which static foreground points are counted as moving-but-static.")
    group.add_argument("--phase_debug_moving_disp_threshold", type=float, default=0.10,
                       help="Max-displacement threshold above which static foreground points are counted as moving-but-static.")
    group.add_argument("--phase_debug_close_seed_distance", type=float, default=0.10,
                       help="Canonical distance threshold for static foreground points close to high-motion seeds.")
    group.add_argument("--phase_debug_max_problem_cells_json", type=int, default=48,
                       help="Number of top problematic cells written in deep_diagnostics_summary.json.")

    # Exact joint static/dynamic + one-rigid-part optimization.
    group.add_argument("--joint_single_rigid_training", action="store_true", default=True,
                       help="Jointly optimize the cosine part field, proposal motion, and one explicit rigid twist at every dynamic iteration.")
    group.add_argument("--disable_joint_single_rigid_training", action="store_true", default=False)

    # Explicit rigid velocity v=omega(t)x(x-c)+u(t).
    group.add_argument("--joint_twist_lr", type=float, default=5e-4)
    group.add_argument("--joint_twist_hidden_dim", type=int, default=64)
    group.add_argument("--joint_twist_num_layers", type=int, default=3)
    group.add_argument("--joint_twist_time_frequencies", type=int, default=8)
    group.add_argument("--joint_max_omega", type=float, default=4.0)
    group.add_argument("--joint_max_translation_velocity", type=float, default=2.0)
    group.add_argument("--joint_total_max_velocity", type=float, default=3.0)
    group.add_argument("--joint_grad_clip", type=float, default=1.0)

    # Exact per-Gaussian initialization from canonical XYZ and the frozen
    # statically trained DINO field. A full pairwise matrix is computed.
    group.add_argument("--joint_num_moving_parts", type=int, default=1,
                       help="Known number of moving rigid parts. Current physical model requires 1.")
    group.add_argument("--joint_init_euclidean_weight", type=float, default=1.0,
                       help="Weight of normalized canonical-center Euclidean distance.")
    group.add_argument("--joint_init_cosine_weight", type=float, default=1.0,
                       help="Weight of 1-cosine between each pair of frozen static DINO Gaussian descriptors.")
    group.add_argument("--joint_init_euclidean_scale", type=float, default=-1.0,
                       help="Euclidean distance normalizer. <=0 uses robust median pair distance.")
    group.add_argument("--joint_init_distance_block_size", type=int, default=512)
    group.add_argument("--joint_init_distance_device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device for the one-time full XYZ+DINO distance computation. CPU is the safe default; CUDA is not recommended for full N^2 initialization.")
    group.add_argument("--dynamic_camera_data_device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Storage device for expanded dynamic-sequence camera images. CPU keeps all frames out of GPU memory; only the sampled frame is copied to CUDA.")
    group.add_argument("--joint_init_distance_storage", type=str, default="float16", choices=["float16", "float32"])
    group.add_argument("--joint_init_kmedoids_iters", type=int, default=12,
                       help="PAM-style iterations directly on the full precomputed combined distance matrix.")
    group.add_argument("--joint_save_initial_distance_matrix", action="store_true", default=True)
    group.add_argument("--disable_joint_save_initial_distance_matrix", action="store_true", default=False)

    # Trainable per-Gaussian dynamic/static cosine field. Geometry and DINO are
    # initialization-only and are never reused by these clustering losses.
    group.add_argument("--joint_part_field_dim", type=int, default=8)
    group.add_argument("--joint_part_assignment_temperature", type=float, default=1.0)
    group.add_argument("--joint_part_field_init_noise", type=float, default=0.02)
    group.add_argument("--joint_part_field_lr", type=float, default=1e-3)
    group.add_argument("--joint_part_prototype_lr", type=float, default=5e-4)
    group.add_argument("--joint_field_pair_samples", type=int, default=1024,
                       help="Subset size for the complete point-to-point cosine matrix each iteration; <=0 uses all Gaussians.")
    group.add_argument("--joint_field_different_margin", type=float, default=0.20)
    group.add_argument("--joint_minimum_part_fraction", type=float, default=0.02)

    # Correctly routed joint losses.
    group.add_argument("--lambda_joint_twist_fit", type=float, default=0.10,
                       help="Fits omega,u to a detached moving proposal selected by the current cosine assignment.")
    group.add_argument("--lambda_joint_proposal_rigid", type=float, default=0.05,
                       help="Aligns the proposal to a detached rigid expert only for confident moving Gaussians.")
    group.add_argument("--lambda_joint_proposal_static", type=float, default=0.02,
                       help="Suppresses proposal velocity only for confident static Gaussians.")
    group.add_argument("--lambda_joint_label", type=float, default=0.05)
    group.add_argument("--lambda_joint_field_compact", type=float, default=0.02)
    group.add_argument("--lambda_joint_field_pairwise", type=float, default=0.01)
    group.add_argument("--lambda_joint_prototype_separation", type=float, default=0.01)
    group.add_argument("--lambda_joint_assignment_entropy", type=float, default=0.002)
    group.add_argument("--lambda_joint_part_occupancy", type=float, default=0.02)
    group.add_argument("--lambda_joint_twist_smooth", type=float, default=0.01)

    group.add_argument("--joint_label_temperature", type=float, default=-1.0,
                       help="<=0 enables adaptive residual-scale temperature; positive fixes it explicitly.")
    group.add_argument("--joint_label_temperature_quantile", type=float, default=0.50)
    group.add_argument("--joint_label_temperature_min", type=float, default=1e-3)
    group.add_argument("--joint_label_temperature_max", type=float, default=5e-2)
    group.add_argument("--joint_confident_static_threshold", type=float, default=0.20)
    group.add_argument("--joint_confident_dynamic_threshold", type=float, default=0.80)
    group.add_argument("--joint_robust_delta", type=float, default=0.05)
    group.add_argument("--joint_twist_smooth_dt", type=float, default=0.02)

    # Continuous trust transfer from proposal to rigid transport. All losses
    # remain active from the first dynamic iteration; this is not a phase split.
    group.add_argument("--joint_rigid_handover_start", type=int, default=0)
    group.add_argument("--joint_rigid_handover_iters", type=int, default=3000)
    group.add_argument("--joint_debug_early_iter", type=int, default=1000)
    group.add_argument("--joint_debug_mid_iter", type=int, default=-1)

    # Legacy aliases accepted only so older command lines do not crash. They are
    # not used by the exact joint objective.
    group.add_argument("--joint_initial_clusters", type=int, default=2)
    group.add_argument("--joint_cluster_kmeans_iters", type=int, default=12)
    group.add_argument("--joint_cluster_xyz_weight", type=float, default=1.0)
    group.add_argument("--joint_cluster_dino_weight", type=float, default=1.0)
    group.add_argument("--lambda_joint_rigid_velocity", type=float, default=0.0)
    group.add_argument("--lambda_joint_static_velocity", type=float, default=0.0)
    group.add_argument("--lambda_joint_cluster", type=float, default=0.0)
    group.add_argument("--lambda_joint_residual", type=float, default=0.0)
    group.add_argument("--lambda_joint_label_binary", type=float, default=0.0)
    group.add_argument("--joint_loss_ramp_iters", type=int, default=0)
    group.add_argument("--joint_residual_ramp_iters", type=int, default=0)

    # Staged dynamic losses.
    group.add_argument("--dynamic_discovery_iters", type=int, default=1000,
                       help="Euler-stage iterations dedicated to static/dynamic discovery before motion-understanding losses dominate.")
    group.add_argument("--motion_understanding_from_iter", type=int, default=-1,
                       help="Start motion-understanding phase at this Euler iteration. -1 = dynamic_discovery_iters.")
    group.add_argument("--lambda_static", type=float, default=0.01,
                       help="Discovery-phase static velocity penalty.")
    group.add_argument("--lambda_dyn_sparse", type=float, default=0.001,
                       help="Dynamic label sparsity prior. Kept as a real parser arg so old commands do not crash.")
    group.add_argument("--lambda_dyn_bin", type=float, default=0.001,
                       help="Dynamic label binarization prior. Kept as a real parser arg so old commands do not crash.")
    group.add_argument("--lambda_static_motion", type=float, default=0.001,
                       help="Motion-phase static penalty applied only to confident static Gaussians.")
    group.add_argument("--static_confident_threshold", type=float, default=0.05,
                       help="Only p_dyn below this threshold is considered confident static during motion phase.")
    group.add_argument("--lambda_cycle", type=float, default=0.01,
                       help="Motion-phase cycle consistency by integrating the transported points back to t=0.")
    group.add_argument("--cycle_from_iter", type=int, default=-1,
                       help="Euler iteration where cycle loss starts. -1 = motion_understanding_from_iter.")
    group.add_argument("--lambda_velocity_rigid", type=float, default=0.0,
                       help="Deprecated/no-op. Removed; rigid motion is learned with --lambda_global_twist.")
    group.add_argument("--lambda_cell_label_coherence", type=float, default=0.0,
                       help="Optional label coherence inside cells. Default 0 because mixed hinge cells should split, not be smoothed.")
    group.add_argument("--num_moving_parts", type=int, default=1,
                       help="Known number of moving rigid parts. 1 for a single articulated moving link.")

    # Deprecated no-op compatibility switches. The losses are removed from the
    # active objective but old commands will not crash.
    group.add_argument("--lambda_global_se3", type=float, default=0.0,
                       help="Deprecated/no-op. Global SE(3) is disabled; SE(3) is recovered by integrating the fitted twist.")
    group.add_argument("--lambda_velocity_temporal", type=float, default=0.0, help="Deprecated/no-op.")
    group.add_argument("--lambda_velocity_accel", type=float, default=0.0, help="Deprecated/no-op.")

    # Global Eulerian twist prior. This is the velocity-field equivalent of SE(3):
    # v(x,t) = omega(t) x x + u(t). It constrains the learned Eulerian field
    # before integrating it into a global pose trajectory.
    group.add_argument("--lambda_global_twist", type=float, default=0.05,
                       help="Fit one Eulerian rigid-body twist per moving part and penalize velocity residuals.")
    group.add_argument("--global_twist_from_iter", type=int, default=1000,
                       help="Euler-stage warmup iterations before the global twist prior is enabled.")
    group.add_argument("--global_twist_dynamic_threshold", type=float, default=0.5,
                       help="Dynamic-probability threshold used to select Gaussians for global twist fitting.")
    group.add_argument("--global_twist_min_points", type=int, default=128,
                       help="Minimum selected dynamic Gaussians required for each/global twist fit.")
    group.add_argument("--global_twist_max_points", type=int, default=8192,
                       help="Maximum dynamic Gaussians used per iteration for global twist fitting.")
    group.add_argument("--global_twist_cluster_iters", type=int, default=8,
                       help="Small k-means iterations used only when --num_moving_parts > 1 for twist fitting.")
    group.add_argument("--global_twist_robust_delta", type=float, default=0.03,
                       help="Huber delta in velocity units for the global twist residual. Set <=0 for squared residual.")
    group.add_argument("--global_twist_ridge", type=float, default=1e-6,
                       help="Ridge regularization for the weighted twist least-squares solve.")
    group.add_argument("--global_twist_detach_fit", action="store_true", default=True,
                       help="Detach fitted omega,u so gradients update the velocity field, not the linear solve.")



    # Learnable per-Gaussian DINO feature field. Dense per-frame DINO maps are
    # rendered as Gaussian features with override_color and supervised across
    # frames, so moving Gaussians carry a temporally stable semantic/part signal.
    group.add_argument("--use_dino_feature_field", action="store_true", default=True,
                       help="Learn a per-Gaussian feature field supervised by dense DINO maps when available.")
    group.add_argument("--disable_dino_feature_field", action="store_true", default=False,
                       help="Disable the per-Gaussian DINO feature field.")
    group.add_argument("--lambda_dino_feature_field", type=float, default=0.02,
                       help="Static-stage weight for rendered DINO feature-field supervision.")
    group.add_argument("--lambda_dino_feature_field_dynamic", type=float, default=0.003,
                       help="Dynamic-stage DINO transport weight. Keep lower than RGB/depth because dense DINO maps are noisy correspondences.")
    group.add_argument("--dino_feature_dynamic_from_iter", type=int, default=0,
                       help="Euler iteration where dynamic DINO transport supervision starts.")
    group.add_argument("--dino_feature_field_dim", type=int, default=32,
                       help="Number of DINO channels learned per Gaussian. <=0 uses the full extracted DINO dimension.")
    group.add_argument("--dino_feature_lr", type=float, default=1e-3,
                       help="Optimizer LR for the per-Gaussian DINO feature field.")
    group.add_argument("--dino_feature_from_iter", type=int, default=0,
                       help="Euler iteration where DINO feature-field supervision starts.")
    group.add_argument("--dino_feature_cache_size", type=int, default=16,
                       help="Number of dense DINO maps cached in RAM/CPU.")
    group.add_argument("--dino_feature_loss_type", type=str, default="cosine", choices=["cosine", "cosine_l1"],
                       help="Feature-field supervision loss. Default cosine follows normalized DINO distillation.")
    group.add_argument("--dino_feature_opacity_threshold", type=float, default=0.05,
                       help="Only supervise DINO pixels whose rendered Gaussian accumulated weight is above this threshold.")
    group.add_argument("--freeze_dino_feature_dynamic", action="store_true", default=True,
                       help="Freeze per-Gaussian DINO descriptors during dynamic training and use DINO loss to supervise transport.")
    group.add_argument("--train_dino_feature_dynamic", action="store_true", default=False,
                       help="Override --freeze_dino_feature_dynamic and continue optimizing DINO descriptors during dynamic training. Not recommended.")

    # Learnable one-channel foreground / semantic field from binary masks.
    group.add_argument("--use_semantic_feature_field", action="store_true", default=True,
                       help="Learn a per-Gaussian foreground/semantic field rendered as a 1-channel splat image.")
    group.add_argument("--disable_semantic_feature_field", action="store_true", default=False,
                       help="Disable rendered semantic-mask field supervision.")
    group.add_argument("--lambda_semantic_feature_field", type=float, default=0.05,
                       help="Weight for rendered semantic/foreground mask supervision.")
    group.add_argument("--semantic_feature_lr", type=float, default=1e-3,
                       help="Optimizer LR for per-Gaussian semantic logits.")
    group.add_argument("--semantic_feature_from_iter", type=int, default=0,
                       help="Euler iteration where semantic feature-field supervision starts.")
    group.add_argument("--semantic_mask_dir", type=str, default="mask",
                       help="Mask root relative to --root_scene. Expected mask/cam_XXX/frame_YYYYYY.png.")
    group.add_argument("--semantic_loss_type", type=str, default="bce", choices=["bce", "dice", "bce_dice"],
                       help="Loss for rendered foreground mask vs GT binary mask. Default BCE for binary semantic/foreground logits.")
    # Dynamic-stage semantic foreground gating.
    group.add_argument("--filter_sem", action="store_true", default=False,
                       help="Restrict dynamic training/transport/losses to Gaussians whose learned semantic probability exceeds --semantic_dynamic_threshold.")
    group.add_argument("--use_semantic_dynamic_foreground", action="store_true", default=False,
                       help="Alias for --filter_sem.")
    group.add_argument("--disable_semantic_dynamic_foreground", action="store_true", default=False,
                       help="Compatibility no-op/override: disable semantic foreground gating.")
    group.add_argument("--semantic_dynamic_threshold", type=float, default=0.5,
                       help="Semantic probability threshold used only when --filter_sem is enabled.")
    group.add_argument("--semantic_dynamic_min_points", type=int, default=32,
                       help="Fallback minimum number of foreground Gaussians if the semantic threshold is too strict.")
    group.add_argument("--semantic_dynamic_background_logit", type=float, default=-20.0,
                       help="Initial/fixed dynamic logit for non-foreground Gaussians.")


    # Weak 3D auxiliary-field smoothness. Affinity is local in 3D and color-aware, so
    # it denoises within surfaces without forcing smoothness across visual boundaries.
    group.add_argument("--lambda_semantic_3d_smooth", type=float, default=0.02,
                       help="Inner weight for local 3D smoothness on semantic probabilities.")
    group.add_argument("--lambda_dino_3d_smooth", type=float, default=0.05,
                       help="Inner weight for local 3D smoothness on learned DINO Gaussian features.")
    group.add_argument("--smooth_3d_knn", type=int, default=8,
                       help="Number of 3D nearest neighbours used by aux-field smoothness.")
    group.add_argument("--smooth_3d_max_points", type=int, default=2048,
                       help="Maximum random Gaussians sampled per iteration for aux-field smoothness.")
    group.add_argument("--smooth_3d_sigma_x", type=float, default=-1.0,
                       help="Spatial sigma for aux-field affinity. <=0 derives it from sampled KNN distances.")
    group.add_argument("--smooth_3d_sigma_c", type=float, default=0.25,
                       help="Color sigma for aux-field affinity. <=0 disables color weighting.")

    # Adaptive motion-grid refinement. This is the dynamic-stage analogue of
    # static 3DGS densification: static training clones/splits Gaussians where
    # screen-space gradients are high; here we split motion cells where the
    # velocity field is ambiguous or render/depth gradients remain high.
    group.add_argument("--enable_adaptive_grid", action="store_true", default=True,
                       help="Enable adaptive split/merge of the Eulerian motion grid during dynamic training.")
    group.add_argument("--grid_from_iter", type=int, default=500,
                       help="Euler-stage iteration at which adaptive motion-grid updates start.")
    group.add_argument("--grid_update_interval", type=int, default=250,
                       help="Run adaptive motion-grid split/merge every N global training iterations.")
    group.add_argument("--grid_split_threshold", type=float, default=0.0,
                       help="Split active motion cells whose ambiguity score is above this threshold.")
    group.add_argument("--grid_merge_threshold", type=float, default=0.0,
                       help="Merge sibling child cells whose ambiguity score is below this threshold. 0 disables practical merging.")
    group.add_argument("--grid_max_splits_per_update", type=int, default=64,
                       help="Maximum number of parent cells split per adaptive-grid update.")
    group.add_argument("--grid_min_points", type=int, default=4,
                       help="Minimum Gaussians required in a cell before it can be split.")
    group.add_argument("--grid_lambda_vel", type=float, default=2.0,
                       help="Weight for within-cell velocity variance in the adaptive-grid score.")
    group.add_argument("--grid_lambda_disp", type=float, default=0.25,
                       help="Weight for within-cell displacement variance in the adaptive-grid score.")
    group.add_argument("--grid_lambda_dino", type=float, default=2.0,
                       help="Weight for within-cell DINO feature variance / semantic-boundary score.")
    group.add_argument("--grid_lambda_render_grad", type=float, default=2.0,
                       help="Weight for per-Gaussian screen-space gradient magnitude from RGB/SSIM/depth losses.")
    group.add_argument("--grid_lambda_dyn_entropy", type=float, default=1.0,
                       help="Weight for dynamic-label ambiguity p*(1-p) inside a cell.")
    group.add_argument("--grid_lambda_dyn_var", type=float, default=1.0,
                       help="Weight for dynamic/static mixture variance inside a cell.")
    group.add_argument("--grid_lambda_twist_residual", type=float, default=0.5,
                       help="Weight for residual to the fitted global Eulerian twist in the adaptive-grid score.")
    group.add_argument("--grid_lambda_strain", type=float, default=0.5,
                       help="Weight for local radial-rate/strain inside a motion cell in the adaptive-grid score.")
    group.add_argument("--grid_strain_max_points", type=int, default=64,
                       help="Maximum points sampled per cell to estimate local radial-rate/strain score.")
    group.add_argument("--grid_render_grad_clip_percentile", type=float, default=95.0,
                       help="Robust normalization percentile for viewspace-gradient adaptive scores.")
    group.add_argument("--grid_score_ema_decay", type=float, default=0.95,
                       help="EMA decay for persistent per-cell adaptive-grid scores. Use -1 to disable EMA and use current scores only.")
    group.add_argument("--grid_split_percentile", type=float, default=90.0,
                       help="If >=0, split cells whose EMA score is above this active-cell percentile, in addition to grid_split_threshold. Values like 90 or 95 are useful.")
    group.add_argument("--grid_print_topk", type=int, default=5,
                       help="Print top-K scored cells whenever the adaptive grid updates. 0 disables details.")

    # Positive supervision for dynamic logits. Without this, sparse+binary losses
    # can collapse all logits to static.
    group.add_argument("--lambda_dyn_motion_target", type=float, default=0.0,
                       help="Deprecated/off by default: old discovery-stage target from windowed displacement + speed. Avoid for articulated objects.")
    group.add_argument("--lambda_dyn_motion_target_motion", type=float, default=0.0,
                       help="Deprecated/off by default: old motion-stage windowed displacement target. Replaced by --lambda_model_dynamic_target.")
    group.add_argument("--lambda_model_dynamic_target", type=float, default=0.05,
                       help="Motion-stage dynamic-logit target from identity-vs-fitted-rigid velocity residuals.")
    group.add_argument("--model_dynamic_target_from_iter", type=int, default=-1,
                       help="Euler iteration where model-based dynamic target starts. -1 = global_twist_from_iter.")
    group.add_argument("--model_dynamic_target_dynamic_threshold", type=float, default=0.5,
                       help="Threshold on fitting score for selecting points used to fit moving rigid fields.")
    group.add_argument("--model_dynamic_target_min_fit_points", type=int, default=128,
                       help="Minimum points per fitted moving rigid field for model-based dynamic target.")
    group.add_argument("--model_dynamic_target_max_fit_points", type=int, default=8192,
                       help="Maximum points used to fit moving rigid fields for model-based dynamic target.")
    group.add_argument("--model_dynamic_target_max_points", type=int, default=4096,
                       help="Maximum sampled points supervised by the model-based dynamic target.")
    group.add_argument("--model_dynamic_target_temperature", type=float, default=-1.0,
                       help="Temperature for sigmoid((e_static-e_rigid)/T). <=0 uses robust auto temperature.")
    group.add_argument("--model_dynamic_target_power", type=float, default=1.0,
                       help="Optional power applied to the model-based dynamic target.")
    group.add_argument("--model_dynamic_target_loss_type", type=str, default="bce", choices=["bce", "smooth_l1", "l1", "l2"],
                       help="Loss between dynamic probability and model-based dynamic target.")
    group.add_argument("--model_dynamic_target_huber_beta", type=float, default=0.10,
                       help="SmoothL1 beta if model_dynamic_target_loss_type=smooth_l1.")
    group.add_argument("--model_dynamic_target_fit_score_mode", type=str, default="prob_or_speed", choices=["prob", "speed", "prob_or_speed"],
                       help="Score used to select points for fitting moving rigid fields.")
    group.add_argument("--dynamic_target_quantile", type=float, default=0.75,
                       help="Deprecated old windowed displacement target quantile; ignored by default.")
    group.add_argument("--dynamic_target_temperature", type=float, default=0.20,
                       help="Deprecated old windowed displacement target temperature; ignored by default.")
    group.add_argument("--dynamic_target_times", type=str, default="0,0.25,0.5,0.75,1.0",
                       help="Deprecated old windowed displacement target times; ignored by default.")
    group.add_argument("--dynamic_target_integration_steps", type=int, default=4,
                       help="Deprecated old windowed displacement target integration steps; ignored by default.")
    group.add_argument("--dynamic_target_max_points", type=int, default=4096,
                       help="Deprecated old windowed displacement target max points; ignored by default.")
    group.add_argument("--dynamic_target_speed_weight", type=float, default=0.25,
                       help="Deprecated old speed weight; ignored by default.")
    group.add_argument("--dynamic_target_displacement_weight", type=float, default=0.75,
                       help="Deprecated old displacement weight; ignored by default.")
    group.add_argument("--dyn_motion_target_until_iter", type=int, default=-1,
                       help="Last Euler iteration for dynamic target. -1 = full Eulerian stage.")

    group.add_argument("--lambda_motion_aware_dyn_coherence", type=float, default=0.02,
                       help="Discovery-stage dynamic-label coherence weighted by temporal-motion similarity, so it does not smooth across hinge/static boundaries.")
    group.add_argument("--lambda_motion_aware_dyn_coherence_motion", type=float, default=0.01,
                       help="Motion-stage motion-aware dynamic-label coherence.")
    group.add_argument("--motion_aware_coherence_max_points", type=int, default=2048,
                       help="Max Gaussians sampled for motion-aware label coherence.")
    group.add_argument("--motion_aware_coherence_knn", type=int, default=8,
                       help="KNN pairs for motion-aware label coherence.")
    group.add_argument("--motion_aware_coherence_sigma_x", type=float, default=-1.0,
                       help="Spatial sigma for motion-aware coherence. <=0 uses median KNN distance.")
    group.add_argument("--motion_aware_coherence_sigma_motion", type=float, default=0.25,
                       help="Temporal-displacement sigma for motion-aware coherence. Larger = more smoothing across different motion.")
    group.add_argument("--motion_aware_coherence_sigma_dino", type=float, default=0.25,
                       help="DINO-feature sigma for optional appearance weighting in motion-aware coherence.")
    group.add_argument("--motion_aware_coherence_boundary_weight", type=float, default=0.0,
                       help="Inner weight for the boundary-sharpening part of motion-aware dynamic-label coherence.")
    group.add_argument("--motion_aware_coherence_boundary_margin", type=float, default=0.50,
                       help="Required dynamic-probability gap for close neighbours with different temporal motion.")

    group.add_argument("--lambda_motion_contrastive_label", type=float, default=0.0,
                       help="Discovery-stage pairwise contrastive label loss: normalized temporal-motion difference should match normalized dynamic-label difference.")
    group.add_argument("--lambda_motion_contrastive_label_motion", type=float, default=0.0,
                       help="Motion-stage pairwise contrastive dynamic-label loss.")
    group.add_argument("--motion_contrastive_from_iter", type=int, default=-1,
                       help="Euler iteration where motion-stage contrastive label loss starts. -1 = motion_understanding_from_iter.")
    group.add_argument("--motion_contrastive_times", type=str, default="0,0.25,0.5,0.75,1.0",
                       help="Normalized times used to build temporal signatures for contrastive dynamic labels.")
    group.add_argument("--motion_contrastive_integration_steps", type=int, default=4,
                       help="Euler integration steps for contrastive dynamic label signatures.")
    group.add_argument("--motion_contrastive_max_points", type=int, default=2048,
                       help="Maximum sampled Gaussians for pairwise contrastive dynamic labels.")
    group.add_argument("--motion_contrastive_num_pairs", type=int, default=32768,
                       help="Number of globally sampled point pairs for contrastive dynamic labels.")
    group.add_argument("--motion_contrastive_target_quantile", type=float, default=0.95,
                       help="Robust quantile used to normalize temporal pair differences to [0,1].")
    group.add_argument("--motion_contrastive_target_gamma", type=float, default=1.0,
                       help="Optional gamma on normalized temporal pair-difference targets.")
    group.add_argument("--motion_contrastive_loss_type", type=str, default="smooth_l1", choices=["smooth_l1", "l1", "l2"],
                       help="Penalty matching dynamic-label pair differences to temporal-motion pair differences.")
    group.add_argument("--motion_contrastive_huber_beta", type=float, default=0.10,
                       help="SmoothL1 beta for pairwise contrastive dynamic-label loss.")

    group.add_argument("--lambda_rigid_model_contrastive_label", type=float, default=0.03,
                       help="Late rigid-model contrastive dynamic-label sharpening after identity/static and k rigid moving fields are fitted.")
    group.add_argument("--rigid_model_contrastive_from_iter", type=int, default=-1,
                       help="Euler iteration where rigid-model contrastive label sharpening starts. -1 = global_twist_from_iter + delay.")
    group.add_argument("--rigid_model_contrastive_delay_iters", type=int, default=1000,
                       help="Delay after global_twist_from_iter before enabling rigid-model contrastive sharpening.")
    group.add_argument("--rigid_model_contrastive_dynamic_threshold", type=float, default=0.5,
                       help="Dynamic probability threshold used to fit moving rigid fields for late sharpening.")
    group.add_argument("--rigid_model_contrastive_min_fit_points", type=int, default=128,
                       help="Minimum points per fitted rigid field for late sharpening.")
    group.add_argument("--rigid_model_contrastive_max_fit_points", type=int, default=8192,
                       help="Maximum points used to fit rigid fields for late sharpening.")
    group.add_argument("--rigid_model_contrastive_max_points", type=int, default=2048,
                       help="Maximum sampled points receiving rigid-model contrastive pair supervision.")
    group.add_argument("--rigid_model_contrastive_num_pairs", type=int, default=32768,
                       help="Number of sampled point pairs for rigid-model contrastive sharpening.")
    group.add_argument("--rigid_model_contrastive_temperature", type=float, default=-1.0,
                       help="Temperature for residual-to-dynamic target. <=0 derives a robust scale from residual differences.")
    group.add_argument("--rigid_model_contrastive_loss_type", type=str, default="smooth_l1", choices=["smooth_l1", "l1", "l2"],
                       help="Loss type for late rigid-model contrastive label sharpening.")
    group.add_argument("--rigid_model_contrastive_huber_beta", type=float, default=0.10,
                       help="SmoothL1 beta for late rigid-model contrastive label sharpening.")

    group.add_argument("--lambda_dynamic_position_rigidity", type=float, default=0.0,
                       help="Motion-phase local pairwise distance preservation on dynamic neighbouring Gaussians.")
    group.add_argument("--dynamic_position_rigidity_from_iter", type=int, default=-1,
                       help="Euler iteration where dynamic pairwise position rigidity starts. -1 = motion_understanding_from_iter.")
    group.add_argument("--dynamic_position_rigidity_times", type=str, default="0,0.25,0.5,0.75,1.0",
                       help="Normalized times used by pairwise dynamic position rigidity.")
    group.add_argument("--dynamic_position_rigidity_integration_steps", type=int, default=4,
                       help="Euler integration steps per rigidity time.")
    group.add_argument("--dynamic_position_rigidity_max_points", type=int, default=4096,
                       help="Max dynamic Gaussians sampled for pairwise position rigidity.")
    group.add_argument("--dynamic_position_rigidity_knn", type=int, default=8,
                       help="KNN neighbours per dynamic Gaussian for pairwise position rigidity.")
    group.add_argument("--dynamic_position_rigidity_dynamic_threshold", type=float, default=0.5,
                       help="Dynamic probability threshold for pairwise position rigidity.")
    group.add_argument("--dynamic_position_rigidity_min_points", type=int, default=128,
                       help="Minimum sampled dynamic points required for pairwise position rigidity.")
    group.add_argument("--dynamic_position_rigidity_robust_delta", type=float, default=0.01,
                       help="Huber/robust delta in metric distance units for pairwise position rigidity.")
    group.add_argument("--dynamic_position_rigidity_sigma_dino", type=float, default=0.25,
                       help="DINO-feature sigma used to weight pairwise rigidity neighbours.")

    return parser


def apply_dynamic_defaults(args):
    """Map the small public CLI to the internal names expected by the existing code."""
    if args.root_scene:
        root = str(Path(args.root_scene).expanduser().resolve())
    else:
        src = Path(args.source_path).expanduser().resolve()
        root = str(src.parent if src.name == "colmap" else src)
        args.root_scene = root

    # Internal dataset names used by Scene.enable_dynamic_sequence(...).
    args.dynamic_sequence_root = root
    args.dynamic_rgb_dir = "rgb"
    args.dynamic_depth_npy_dir = "depth_npy"
    args.dynamic_depth_png_dir = "depth_png"
    args.dynamic_use_depth = bool(args.use_depth) and not bool(getattr(args, "no_depth", False))
    args.dynamic_use_depth_npy = args.depth_format == "npy"
    args.dynamic_depth_is_metric = True
    args.dynamic_cameras = args.cameras
    args.dynamic_frame_start = int(args.frame_start)
    args.dynamic_frame_end = int(args.frame_end)
    args.dynamic_frame_stride = int(args.frame_stride)
    args.dynamic_test_stride = int(args.test_stride)

    # DINOv3 internal defaults.
    args.dinov3_root = "dinov3"
    args.dinov3_feature_ext = ".pt"
    args.dinov3_repo = "submodules/dinov3"
    args.dinov3_model_name = "dinov3_vitl16"
    args.dinov3_image_size = DINOv3_IMAGE_SIZE
    # The learnable DINO Gaussian field requires dense per-pixel DINO maps.
    # If the field is enabled, automatically enable DINO map handling and try
    # extraction when the maps are missing.
    if bool(getattr(args, "use_dino_feature_field", False)):
        args.use_dinov3 = True
    args.dinov3_extract_on_start = bool(args.extract_dinov3) or bool(getattr(args, "use_dino_feature_field", False))
    args.dinov3_weights = str(getattr(args, "dinov3_weights", "") or "")
    args.dinov3_required = bool(getattr(args, "dinov3_required", False))

    # W&B defaults. The API key is never stored in cfg_args by this code;
    # authenticate externally with `wandb login` or WANDB_API_KEY.
    args.no_wandb = bool(getattr(args, "no_wandb", False))
    args.wandb = bool(getattr(args, "wandb", True)) and not args.no_wandb and str(getattr(args, "wandb_mode", "online") or "online") != "disabled"
    args.wandb_project = str(getattr(args, "wandb_project", "RigEuler3DGS") or "RigEuler3DGS")
    args.wandb_entity = str(getattr(args, "wandb_entity", "") or "")
    args.wandb_name = str(getattr(args, "wandb_name", "") or "")
    args.wandb_mode = str(getattr(args, "wandb_mode", "online") or "online")
    args.wandb_tags = str(getattr(args, "wandb_tags", "") or "")
    args.wandb_log_interval = int(getattr(args, "wandb_log_interval", 10))
    args.wandb_log_artifacts = bool(getattr(args, "wandb_log_artifacts", False))
    args.wandb_log_tracks = bool(getattr(args, "wandb_log_tracks", False))

    args.dinov3_overwrite = False
    args.dinov3_feature_dim = 0
    args.dinov3_gaussian_frame_id = 0
    args.dinov3_gaussian_max_cameras = 0

    # Motion-field architecture and optimizers.
    args.motion_feature_dim = 32
    args.motion_hidden_dim = 64
    args.motion_num_layers = 3
    args.motion_time_frequencies = 8
    args.motion_max_velocity = 0.0
    args.motion_integration_steps = 4
    args.motion_voxel_size = -1.0
    args.motion_opacity_threshold = 0.005
    args.motion_max_initial_cells = 50000
    args.motion_mlp_lr = 1e-3
    args.motion_grid_lr = 1e-2
    args.dynamic_lr = 1e-3
    args.dynamic_logit_init = -2.0

    # Semantic foreground gating for dynamic stage.
    # Default is OFF. Enable only with --filter_sem (or legacy alias
    # --use_semantic_dynamic_foreground).
    args.filter_sem = bool(getattr(args, "filter_sem", False)) or bool(getattr(args, "use_semantic_dynamic_foreground", False))
    if bool(getattr(args, "disable_semantic_dynamic_foreground", False)):
        args.filter_sem = False
    args.use_semantic_dynamic_foreground = bool(args.filter_sem)
    args.semantic_dynamic_threshold = float(getattr(args, "semantic_dynamic_threshold", 0.5))
    args.semantic_dynamic_min_points = int(getattr(args, "semantic_dynamic_min_points", 32))
    args.semantic_dynamic_background_logit = float(getattr(args, "semantic_dynamic_background_logit", -20.0))

    # Two-stage resume defaults.
    args.force_static_retrain = bool(getattr(args, "force_static_retrain", False))
    args.auto_resume_static_ply = not bool(getattr(args, "no_auto_static_resume", False))

    # Canonical Gaussian handling during dynamic stage.
    args.freeze_canonical_geometry = True
    args.freeze_canonical_appearance = True
    args.allow_dynamic_gaussian_finetune = False

    # DINO / feature loss defaults.
    args.dino_features_path = ""
    args.dino_feature_dim = 0
    args.dino_sigma_f = 0.25

    # Staged dynamic losses. Removed active losses: rollout/speed/temporal/accel/local SE(3)/global SE(3).
    args.joint_single_rigid_training = bool(getattr(args, "joint_single_rigid_training", True)) and not bool(getattr(args, "disable_joint_single_rigid_training", False))
    args.joint_twist_lr = float(getattr(args, "joint_twist_lr", 5e-4))
    args.joint_twist_hidden_dim = int(getattr(args, "joint_twist_hidden_dim", 64))
    args.joint_twist_num_layers = int(getattr(args, "joint_twist_num_layers", 3))
    args.joint_twist_time_frequencies = int(getattr(args, "joint_twist_time_frequencies", 8))
    args.joint_max_omega = float(getattr(args, "joint_max_omega", 4.0))
    args.joint_max_translation_velocity = float(getattr(args, "joint_max_translation_velocity", 2.0))
    args.joint_total_max_velocity = float(getattr(args, "joint_total_max_velocity", 3.0))
    args.joint_grad_clip = float(getattr(args, "joint_grad_clip", 1.0))

    args.joint_num_moving_parts = int(getattr(args, "joint_num_moving_parts", 1))
    if args.joint_num_moving_parts != 1:
        raise ValueError("The current explicit physical model supports exactly one moving rigid part; set --joint_num_moving_parts 1.")
    args.joint_init_euclidean_weight = float(getattr(args, "joint_init_euclidean_weight", 1.0))
    args.joint_init_cosine_weight = float(getattr(args, "joint_init_cosine_weight", 1.0))
    args.joint_init_euclidean_scale = float(getattr(args, "joint_init_euclidean_scale", -1.0))
    args.joint_init_distance_block_size = int(getattr(args, "joint_init_distance_block_size", 512))
    args.joint_init_distance_device = str(getattr(args, "joint_init_distance_device", "cpu"))
    args.dynamic_camera_data_device = str(getattr(args, "dynamic_camera_data_device", "cpu"))
    args.joint_init_distance_storage = str(getattr(args, "joint_init_distance_storage", "float16"))
    args.joint_init_kmedoids_iters = int(getattr(args, "joint_init_kmedoids_iters", 12))
    args.joint_save_initial_distance_matrix = bool(getattr(args, "joint_save_initial_distance_matrix", True)) and not bool(getattr(args, "disable_joint_save_initial_distance_matrix", False))

    args.joint_part_field_dim = int(getattr(args, "joint_part_field_dim", 8))
    args.joint_part_assignment_temperature = float(getattr(args, "joint_part_assignment_temperature", 1.0))
    args.joint_part_field_init_noise = float(getattr(args, "joint_part_field_init_noise", 0.02))
    args.joint_part_field_lr = float(getattr(args, "joint_part_field_lr", 1e-3))
    args.joint_part_prototype_lr = float(getattr(args, "joint_part_prototype_lr", 5e-4))
    args.joint_field_pair_samples = int(getattr(args, "joint_field_pair_samples", 1024))
    args.joint_field_different_margin = float(getattr(args, "joint_field_different_margin", 0.20))
    args.joint_minimum_part_fraction = float(getattr(args, "joint_minimum_part_fraction", 0.02))

    args.lambda_joint_twist_fit = float(getattr(args, "lambda_joint_twist_fit", 0.10))
    args.lambda_joint_proposal_rigid = float(getattr(args, "lambda_joint_proposal_rigid", 0.05))
    args.lambda_joint_proposal_static = float(getattr(args, "lambda_joint_proposal_static", 0.02))
    args.lambda_joint_label = float(getattr(args, "lambda_joint_label", 0.05))
    args.lambda_joint_field_compact = float(getattr(args, "lambda_joint_field_compact", 0.02))
    args.lambda_joint_field_pairwise = float(getattr(args, "lambda_joint_field_pairwise", 0.01))
    args.lambda_joint_prototype_separation = float(getattr(args, "lambda_joint_prototype_separation", 0.01))
    args.lambda_joint_assignment_entropy = float(getattr(args, "lambda_joint_assignment_entropy", 0.002))
    args.lambda_joint_part_occupancy = float(getattr(args, "lambda_joint_part_occupancy", 0.02))
    args.lambda_joint_twist_smooth = float(getattr(args, "lambda_joint_twist_smooth", 0.01))

    args.joint_label_temperature = float(getattr(args, "joint_label_temperature", -1.0))
    args.joint_label_temperature_quantile = float(getattr(args, "joint_label_temperature_quantile", 0.50))
    args.joint_label_temperature_min = float(getattr(args, "joint_label_temperature_min", 1e-3))
    args.joint_label_temperature_max = float(getattr(args, "joint_label_temperature_max", 5e-2))
    args.joint_confident_static_threshold = float(getattr(args, "joint_confident_static_threshold", 0.20))
    args.joint_confident_dynamic_threshold = float(getattr(args, "joint_confident_dynamic_threshold", 0.80))
    args.joint_robust_delta = float(getattr(args, "joint_robust_delta", 0.05))
    args.joint_twist_smooth_dt = float(getattr(args, "joint_twist_smooth_dt", 0.02))
    args.joint_rigid_handover_start = int(getattr(args, "joint_rigid_handover_start", 0))
    args.joint_rigid_handover_iters = int(getattr(args, "joint_rigid_handover_iters", 3000))
    args.joint_debug_early_iter = int(getattr(args, "joint_debug_early_iter", 1000))
    args.joint_debug_mid_iter = int(getattr(args, "joint_debug_mid_iter", -1))

    args.dynamic_discovery_iters = int(getattr(args, "dynamic_discovery_iters", 3000))
    args.motion_understanding_from_iter = int(getattr(args, "motion_understanding_from_iter", -1))
    if args.motion_understanding_from_iter < 0:
        args.motion_understanding_from_iter = args.dynamic_discovery_iters
    args.lambda_static = float(getattr(args, "lambda_static", 0.01))
    args.lambda_static_motion = float(getattr(args, "lambda_static_motion", 0.001))
    args.static_confident_threshold = float(getattr(args, "static_confident_threshold", 0.05))
    args.lambda_rigid_dino = 0.0
    args.lambda_cycle = float(getattr(args, "lambda_cycle", 0.01))
    args.lambda_dyn_sparse = float(getattr(args, "lambda_dyn_sparse", 0.001))
    args.lambda_dyn_bin = float(getattr(args, "lambda_dyn_bin", 0.001))
    args.lambda_dyn_motion_target = float(getattr(args, "lambda_dyn_motion_target", 0.0))
    args.dynamic_target_quantile = float(getattr(args, "dynamic_target_quantile", 0.75))
    args.dynamic_target_temperature = float(getattr(args, "dynamic_target_temperature", 0.15))
    args.dyn_motion_target_until_iter = int(getattr(args, "dyn_motion_target_until_iter", -1))
    args.lambda_dyn_motion_target_motion = float(getattr(args, "lambda_dyn_motion_target_motion", 0.0))
    args.lambda_model_dynamic_target = float(getattr(args, "lambda_model_dynamic_target", 0.05))
    args.model_dynamic_target_from_iter = int(getattr(args, "model_dynamic_target_from_iter", -1))
    if args.model_dynamic_target_from_iter < 0:
        args.model_dynamic_target_from_iter = int(getattr(args, "global_twist_from_iter", args.motion_understanding_from_iter))
    args.model_dynamic_target_dynamic_threshold = float(getattr(args, "model_dynamic_target_dynamic_threshold", 0.5))
    args.model_dynamic_target_min_fit_points = int(getattr(args, "model_dynamic_target_min_fit_points", 128))
    args.model_dynamic_target_max_fit_points = int(getattr(args, "model_dynamic_target_max_fit_points", 8192))
    args.model_dynamic_target_max_points = int(getattr(args, "model_dynamic_target_max_points", 4096))
    args.model_dynamic_target_temperature = float(getattr(args, "model_dynamic_target_temperature", -1.0))
    args.model_dynamic_target_power = float(getattr(args, "model_dynamic_target_power", 1.0))
    args.model_dynamic_target_loss_type = str(getattr(args, "model_dynamic_target_loss_type", "bce"))
    args.model_dynamic_target_huber_beta = float(getattr(args, "model_dynamic_target_huber_beta", 0.10))
    args.model_dynamic_target_fit_score_mode = str(getattr(args, "model_dynamic_target_fit_score_mode", "prob_or_speed"))
    args.dynamic_target_times = str(getattr(args, "dynamic_target_times", "0,0.25,0.5,0.75,1.0"))
    args.dynamic_target_integration_steps = int(getattr(args, "dynamic_target_integration_steps", 4))
    args.dynamic_target_max_points = int(getattr(args, "dynamic_target_max_points", 4096))
    args.dynamic_target_speed_weight = float(getattr(args, "dynamic_target_speed_weight", 0.25))
    args.dynamic_target_displacement_weight = float(getattr(args, "dynamic_target_displacement_weight", 0.75))
    args.lambda_motion_aware_dyn_coherence = float(getattr(args, "lambda_motion_aware_dyn_coherence", 0.02))
    args.lambda_motion_aware_dyn_coherence_motion = float(getattr(args, "lambda_motion_aware_dyn_coherence_motion", 0.01))
    args.motion_aware_coherence_max_points = int(getattr(args, "motion_aware_coherence_max_points", 2048))
    args.motion_aware_coherence_knn = int(getattr(args, "motion_aware_coherence_knn", 8))
    args.motion_aware_coherence_sigma_x = float(getattr(args, "motion_aware_coherence_sigma_x", -1.0))
    args.motion_aware_coherence_sigma_motion = float(getattr(args, "motion_aware_coherence_sigma_motion", 0.25))
    args.motion_aware_coherence_sigma_dino = float(getattr(args, "motion_aware_coherence_sigma_dino", 0.25))
    args.motion_aware_coherence_boundary_weight = float(getattr(args, "motion_aware_coherence_boundary_weight", 0.0))
    args.motion_aware_coherence_boundary_margin = float(getattr(args, "motion_aware_coherence_boundary_margin", 0.50))

    args.lambda_motion_contrastive_label = float(getattr(args, "lambda_motion_contrastive_label", 0.0))
    args.lambda_motion_contrastive_label_motion = float(getattr(args, "lambda_motion_contrastive_label_motion", 0.0))
    args.motion_contrastive_from_iter = int(getattr(args, "motion_contrastive_from_iter", -1))
    if args.motion_contrastive_from_iter < 0:
        args.motion_contrastive_from_iter = args.motion_understanding_from_iter
    args.motion_contrastive_times = str(getattr(args, "motion_contrastive_times", "0,0.25,0.5,0.75,1.0"))
    args.motion_contrastive_integration_steps = int(getattr(args, "motion_contrastive_integration_steps", 4))
    args.motion_contrastive_max_points = int(getattr(args, "motion_contrastive_max_points", 2048))
    args.motion_contrastive_num_pairs = int(getattr(args, "motion_contrastive_num_pairs", 32768))
    args.motion_contrastive_target_quantile = float(getattr(args, "motion_contrastive_target_quantile", 0.95))
    args.motion_contrastive_target_gamma = float(getattr(args, "motion_contrastive_target_gamma", 1.0))
    args.motion_contrastive_loss_type = str(getattr(args, "motion_contrastive_loss_type", "smooth_l1"))
    args.motion_contrastive_huber_beta = float(getattr(args, "motion_contrastive_huber_beta", 0.10))

    args.lambda_rigid_model_contrastive_label = float(getattr(args, "lambda_rigid_model_contrastive_label", 0.03))
    args.rigid_model_contrastive_delay_iters = int(getattr(args, "rigid_model_contrastive_delay_iters", 1000))
    args.rigid_model_contrastive_from_iter = int(getattr(args, "rigid_model_contrastive_from_iter", -1))
    if args.rigid_model_contrastive_from_iter < 0:
        args.rigid_model_contrastive_from_iter = int(getattr(args, "global_twist_from_iter", args.motion_understanding_from_iter)) + args.rigid_model_contrastive_delay_iters
    args.rigid_model_contrastive_dynamic_threshold = float(getattr(args, "rigid_model_contrastive_dynamic_threshold", 0.5))
    args.rigid_model_contrastive_min_fit_points = int(getattr(args, "rigid_model_contrastive_min_fit_points", 128))
    args.rigid_model_contrastive_max_fit_points = int(getattr(args, "rigid_model_contrastive_max_fit_points", 8192))
    args.rigid_model_contrastive_max_points = int(getattr(args, "rigid_model_contrastive_max_points", 2048))
    args.rigid_model_contrastive_num_pairs = int(getattr(args, "rigid_model_contrastive_num_pairs", 32768))
    args.rigid_model_contrastive_temperature = float(getattr(args, "rigid_model_contrastive_temperature", -1.0))
    args.rigid_model_contrastive_loss_type = str(getattr(args, "rigid_model_contrastive_loss_type", "smooth_l1"))
    args.rigid_model_contrastive_huber_beta = float(getattr(args, "rigid_model_contrastive_huber_beta", 0.10))

    args.lambda_dynamic_position_rigidity = float(getattr(args, "lambda_dynamic_position_rigidity", 0.0))
    args.dynamic_position_rigidity_from_iter = int(getattr(args, "dynamic_position_rigidity_from_iter", -1))
    if args.dynamic_position_rigidity_from_iter < 0:
        args.dynamic_position_rigidity_from_iter = args.motion_understanding_from_iter
    args.dynamic_position_rigidity_times = str(getattr(args, "dynamic_position_rigidity_times", "0,0.25,0.5,0.75,1.0"))
    args.dynamic_position_rigidity_integration_steps = int(getattr(args, "dynamic_position_rigidity_integration_steps", 4))
    args.dynamic_position_rigidity_max_points = int(getattr(args, "dynamic_position_rigidity_max_points", 4096))
    args.dynamic_position_rigidity_knn = int(getattr(args, "dynamic_position_rigidity_knn", 8))
    args.dynamic_position_rigidity_dynamic_threshold = float(getattr(args, "dynamic_position_rigidity_dynamic_threshold", 0.5))
    args.dynamic_position_rigidity_min_points = int(getattr(args, "dynamic_position_rigidity_min_points", 128))
    args.dynamic_position_rigidity_robust_delta = float(getattr(args, "dynamic_position_rigidity_robust_delta", 0.01))
    args.dynamic_position_rigidity_sigma_dino = float(getattr(args, "dynamic_position_rigidity_sigma_dino", 0.25))

    if args.dyn_motion_target_until_iter < 0:
        args.dyn_motion_target_until_iter = args.euler_field_train_iters

    args.lambda_velocity_temporal = 0.0
    args.lambda_velocity_accel = 0.0
    args.lambda_velocity_rigid = float(getattr(args, "lambda_velocity_rigid", 0.02))
    args.lambda_cell_label_coherence = float(getattr(args, "lambda_cell_label_coherence", 0.0))
    args.num_moving_parts = int(getattr(args, "num_moving_parts", 1))
    args.lambda_global_se3 = 0.0
    args.lambda_global_twist = float(getattr(args, "lambda_global_twist", 0.05))
    args.global_twist_from_iter = int(getattr(args, "global_twist_from_iter", args.motion_understanding_from_iter))
    args.global_twist_dynamic_threshold = float(getattr(args, "global_twist_dynamic_threshold", 0.5))
    args.global_twist_min_points = int(getattr(args, "global_twist_min_points", 128))
    args.global_twist_max_points = int(getattr(args, "global_twist_max_points", 8192))
    args.global_twist_cluster_iters = int(getattr(args, "global_twist_cluster_iters", 8))
    args.global_twist_robust_delta = float(getattr(args, "global_twist_robust_delta", 0.03))
    args.global_twist_ridge = float(getattr(args, "global_twist_ridge", 1e-6))
    args.global_twist_detach_fit = bool(getattr(args, "global_twist_detach_fit", True))

    # DINO feature field defaults.
    args.use_dino_feature_field = bool(getattr(args, "use_dino_feature_field", True)) and not bool(getattr(args, "disable_dino_feature_field", False))
    args.lambda_dino_feature_field = float(getattr(args, "lambda_dino_feature_field", 0.02))
    args.lambda_dino_feature_field_dynamic = float(getattr(args, "lambda_dino_feature_field_dynamic", 0.003))
    args.dino_feature_dynamic_from_iter = int(getattr(args, "dino_feature_dynamic_from_iter", 0))
    args.dino_feature_field_dim = int(getattr(args, "dino_feature_field_dim", 32))
    args.dino_feature_lr = float(getattr(args, "dino_feature_lr", 1e-3))
    args.dino_feature_from_iter = int(getattr(args, "dino_feature_from_iter", 0))
    args.dino_feature_cache_size = int(getattr(args, "dino_feature_cache_size", 16))
    args.dino_feature_loss_type = getattr(args, "dino_feature_loss_type", "cosine")
    args.dino_feature_opacity_threshold = float(getattr(args, "dino_feature_opacity_threshold", 0.05))
    args.freeze_dino_feature_dynamic = bool(getattr(args, "freeze_dino_feature_dynamic", True)) and not bool(getattr(args, "train_dino_feature_dynamic", False))
    args.semantic_loss_type = getattr(args, "semantic_loss_type", "bce")
    args.lambda_semantic_3d_smooth = float(getattr(args, "lambda_semantic_3d_smooth", 0.02))
    args.lambda_dino_3d_smooth = float(getattr(args, "lambda_dino_3d_smooth", 0.05))
    args.smooth_3d_knn = int(getattr(args, "smooth_3d_knn", 8))
    args.smooth_3d_max_points = int(getattr(args, "smooth_3d_max_points", 2048))
    args.smooth_3d_sigma_x = float(getattr(args, "smooth_3d_sigma_x", -1.0))
    args.smooth_3d_sigma_c = float(getattr(args, "smooth_3d_sigma_c", 0.25))

    # Stage schedule.
    args.cycle_from_iter = int(getattr(args, "cycle_from_iter", -1))
    if args.cycle_from_iter < 0:
        args.cycle_from_iter = args.motion_understanding_from_iter
    args.cycle_se3_from_iter = args.cycle_from_iter
    args.se3_from_iter = 2000000000
    args.enable_adaptive_grid = bool(getattr(args, "enable_adaptive_grid", True))
    args.grid_from_iter = int(getattr(args, "grid_from_iter", 1000))
    args.finetune_from_iter = 2000000000

    # Track export.
    args.export_gaussian_tracks = True
    args.track_export_name = "gaussian_tracks.pt"
    args.track_export_dtype = "float32"
    args.track_frame_stride = 1

    # Phase-boundary debug snapshots.
    args.export_phase_debug_snapshots = bool(getattr(args, "export_phase_debug_snapshots", True)) and not bool(getattr(args, "disable_phase_debug_snapshots", False))
    args.phase_debug_make_html = bool(getattr(args, "phase_debug_make_html", True)) and not bool(getattr(args, "phase_debug_no_html", False))
    args.phase_debug_dir = getattr(args, "phase_debug_dir", "debug_phase_snapshots")
    args.phase_debug_html_max_points = int(getattr(args, "phase_debug_html_max_points", 30000))
    args.phase_debug_html_frame_stride = int(getattr(args, "phase_debug_html_frame_stride", 1))
    args.phase_debug_velocity_max_arrows = int(getattr(args, "phase_debug_velocity_max_arrows", 3000))
    args.phase_debug_grid_octomap_html = bool(getattr(args, "phase_debug_grid_octomap_html", True)) and not bool(getattr(args, "phase_debug_no_grid_octomap_html", False))
    args.phase_debug_grid_octomap_color_mode = str(getattr(args, "phase_debug_grid_octomap_color_mode", "feature_pca"))
    args.phase_debug_grid_octomap_max_cells = int(getattr(args, "phase_debug_grid_octomap_max_cells", 5000))
    args.phase_debug_grid_octomap_min_count = int(getattr(args, "phase_debug_grid_octomap_min_count", 1))
    args.phase_debug_grid_octomap_opacity = float(getattr(args, "phase_debug_grid_octomap_opacity", 0.24))
    args.phase_debug_grid_octomap_active_only = not bool(getattr(args, "phase_debug_grid_octomap_include_inactive", False))
    args.phase_debug_deep_diagnostics = bool(getattr(args, "phase_debug_deep_diagnostics", True)) and not bool(getattr(args, "phase_debug_no_deep_diagnostics", False))
    args.phase_debug_write_csv = bool(getattr(args, "phase_debug_write_csv", True)) and not bool(getattr(args, "phase_debug_no_csv", False))
    args.phase_debug_seed_quantile = float(getattr(args, "phase_debug_seed_quantile", 0.80))
    args.phase_debug_seed_min_points = int(getattr(args, "phase_debug_seed_min_points", 128))
    args.phase_debug_seed_max_points = int(getattr(args, "phase_debug_seed_max_points", 1024))
    args.phase_debug_nearest_chunk = int(getattr(args, "phase_debug_nearest_chunk", 2048))
    args.phase_debug_q_temperature = float(getattr(args, "phase_debug_q_temperature", -1.0))
    args.phase_debug_twist_ridge = float(getattr(args, "phase_debug_twist_ridge", getattr(args, "global_twist_ridge", 1e-6)))
    args.phase_debug_stuck_speed_threshold = float(getattr(args, "phase_debug_stuck_speed_threshold", 0.05))
    args.phase_debug_stuck_disp_threshold = float(getattr(args, "phase_debug_stuck_disp_threshold", 0.02))
    args.phase_debug_moving_speed_threshold = float(getattr(args, "phase_debug_moving_speed_threshold", 0.10))
    args.phase_debug_moving_disp_threshold = float(getattr(args, "phase_debug_moving_disp_threshold", 0.10))
    args.phase_debug_close_seed_distance = float(getattr(args, "phase_debug_close_seed_distance", 0.10))
    args.phase_debug_max_problem_cells_json = int(getattr(args, "phase_debug_max_problem_cells_json", 48))

    # Post-training rendering / HTML visualization defaults.
    # Rendering uses the same camera/frame subset by default unless explicitly overridden.
    if not getattr(args, "render_cameras", ""):
        args.render_cameras = getattr(args, "cameras", "")
    if int(getattr(args, "render_frame_start", -2)) == -2:
        args.render_frame_start = int(getattr(args, "frame_start", 0))
    if int(getattr(args, "render_frame_end", -2)) == -2:
        args.render_frame_end = int(getattr(args, "frame_end", -1))
    if int(getattr(args, "render_frame_stride", 0)) <= 0:
        args.render_frame_stride = int(getattr(args, "frame_stride", 1))
    args.render_use_tracks = not bool(getattr(args, "render_no_tracks", False))
    args.track_html_label_mode = getattr(args, "track_html_label_mode", "auto")
    args.track_html_speed_percentile = float(getattr(args, "track_html_speed_percentile", 80.0))
    args.make_velocity_html = bool(getattr(args, "make_velocity_html", False))
    args.velocity_html_name = getattr(args, "velocity_html_name", "velocity_field_visualization.html")
    args.velocity_html_max_points = int(getattr(args, "velocity_html_max_points", 30000))
    args.velocity_html_max_arrows = int(getattr(args, "velocity_html_max_arrows", 3000))
    args.velocity_html_frame_stride = int(getattr(args, "velocity_html_frame_stride", 1))
    args.velocity_html_speed_clip_percentile = float(getattr(args, "velocity_html_speed_clip_percentile", 98.0))
    args.velocity_html_arrow_scale = float(getattr(args, "velocity_html_arrow_scale", 0.0))
    args.html_frustum_scale = float(getattr(args, "html_frustum_scale", 0.12))
    args.html_aspectmode = getattr(args, "html_aspectmode", "cube")
    args.html_axis_scope = getattr(args, "html_axis_scope", "all")
    args.html_camera_deduplicate = not bool(getattr(args, "html_no_camera_deduplicate", False))

    # Rigidity / SE(3) sampling.
    args.rigid_max_cells = 512
    args.rigid_pairs_per_cell = 32
    args.se3_max_cells = 256
    args.se3_min_points = 6
    args.se3_max_points_per_cell = 64

    # Adaptive grid internal defaults / CLI values.
    args.grid_update_interval = int(getattr(args, "grid_update_interval", 250))
    args.grid_split_threshold = float(getattr(args, "grid_split_threshold", 0.0))
    args.grid_merge_threshold = float(getattr(args, "grid_merge_threshold", 0.0))
    args.grid_max_splits_per_update = int(getattr(args, "grid_max_splits_per_update", 64))
    args.grid_min_points = int(getattr(args, "grid_min_points", 4))
    args.grid_lambda_vel = float(getattr(args, "grid_lambda_vel", 2.0))
    args.grid_lambda_disp = float(getattr(args, "grid_lambda_disp", 0.5))
    args.grid_lambda_dino = float(getattr(args, "grid_lambda_dino", 2.0))
    args.grid_lambda_render_grad = float(getattr(args, "grid_lambda_render_grad", 2.0))
    args.grid_lambda_dyn_entropy = float(getattr(args, "grid_lambda_dyn_entropy", 1.0))
    args.grid_lambda_dyn_var = float(getattr(args, "grid_lambda_dyn_var", 1.0))
    args.grid_lambda_twist_residual = float(getattr(args, "grid_lambda_twist_residual", 0.5))
    args.grid_lambda_strain = float(getattr(args, "grid_lambda_strain", 0.5))
    args.grid_strain_max_points = int(getattr(args, "grid_strain_max_points", 64))
    args.grid_render_grad_clip_percentile = float(getattr(args, "grid_render_grad_clip_percentile", 95.0))
    args.grid_score_ema_decay = float(getattr(args, "grid_score_ema_decay", 0.95))
    args.grid_split_percentile = float(getattr(args, "grid_split_percentile", 90.0))
    args.grid_print_topk = int(getattr(args, "grid_print_topk", 5))

    if args.joint_single_rigid_training:
        # No hard discovery/motion split. Legacy fitted-twist/target losses are
        # disabled because the explicit bounded twist is optimized directly.
        args.motion_understanding_from_iter = 0
        args.global_twist_from_iter = 0
        args.model_dynamic_target_from_iter = 0
        args.cycle_from_iter = 0
        args.lambda_static = 0.0
        args.lambda_dyn_sparse = 0.0
        args.lambda_dyn_bin = 0.0
        args.lambda_static_motion = 0.0
        args.lambda_global_twist = 0.0
        args.lambda_model_dynamic_target = 0.0
        args.lambda_rigid_model_contrastive_label = 0.0
        args.lambda_motion_contrastive_label = 0.0
        args.lambda_motion_contrastive_label_motion = 0.0
        args.lambda_motion_aware_dyn_coherence = 0.0
        args.lambda_motion_aware_dyn_coherence_motion = 0.0
        args.lambda_cell_label_coherence = 0.0
        args.grid_from_iter = min(int(getattr(args, "grid_from_iter", 1000)), 1000)

    return args


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
