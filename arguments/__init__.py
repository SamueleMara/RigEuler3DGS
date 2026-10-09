"""Arguments for canonical 3DGS and the retained Phase-1 motion branches."""

from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path
import os
import sys


DINOv3_HF_MODEL = os.environ.get(
    "DINOV3_HF_MODEL", "facebook/dinov3-vitl16-pretrain-lvd1689m"
)
DINOv3_HF_LOCAL_DIR = os.environ.get(
    "DINOV3_HF_LOCAL_DIR",
    str(Path.home() / ".cache" / "huggingface" / "dinov3-vitl16-pretrain-lvd1689m"),
)
DINOv3_BACKEND = os.environ.get("DINOV3_BACKEND", "transformers")
DINOv3_IMAGE_SIZE = int(os.environ.get("DINOV3_IMAGE_SIZE", "512"))

RIGID_KILLING_REPRESENTATIONS = (
    "rigid_killing_1d",  # legacy name for the cell/Eulerian branch
    "rigid_killing_eulerian",
    "rigid_killing_lagrangian",
)


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none: bool = False):
        group = parser.add_argument_group(name)
        for original_key, value in vars(self).items():
            shorthand = original_key.startswith("_")
            key = original_key[1:] if shorthand else original_key
            default = None if fill_none else value
            flags = ["--" + key]
            if shorthand:
                flags.append("-" + key[0])
            if isinstance(value, bool):
                group.add_argument(*flags, default=default, action="store_true")
            else:
                group.add_argument(*flags, default=default, type=type(value))

    def extract(self, args):
        group = GroupParams()
        own = vars(self)
        for key, value in vars(args).items():
            if key in own or "_" + key in own:
                setattr(group, key, value)
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        # Depth stored as metric distance rather than as inverse depth. The
        # renderer composites 1/z, so a metric map has to be inverted before
        # it can be compared with what comes out. Without this a .npy holding
        # 3.5 metres is silently read as an inverse depth of 3.5.
        self.sapien_depth_is_metric = False
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


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


def add_dynamic_args(parser: ArgumentParser):
    """Expose only the fixed-grid Eulerian Phase-1 configuration surface."""
    group = parser.add_argument_group("Fixed-curl Eulerian dynamic pipeline")
    correspondence_toggle = group.add_mutually_exclusive_group()
    wandb_toggle = group.add_mutually_exclusive_group()

    group.add_argument("--dynamic_motion", action="store_true")
    group.add_argument("--root_scene", type=str, default="")
    group.add_argument("--static_train_iters", type=int, default=30_000)
    group.add_argument(
        "--free_field_iters",
        type=int,
        default=500,
        help="Optimizer updates per temporal window and schedule pass, using all cameras.",
    )
    group.add_argument("--num_moving_parts", type=int, default=1)
    group.add_argument("--force_static_retrain", action="store_true")
    group.add_argument("--start_ply", type=str, default="")
    group.add_argument("--start_ply_iteration", type=int, default=30_000)
    group.add_argument("--fps", type=float, default=30.0)
    group.add_argument(
        "--time_scale",
        type=float,
        default=1.0,
        help="Must remain 1.0: dynamic velocities use physical seconds.",
    )
    group.add_argument("--motion_time_mode", choices=("seconds",), default="seconds")
    group.add_argument(
        "--dynamic_timestamps_file",
        type=str,
        default="",
        help="Optional JSON/CSV/TSV acquisition timestamps; otherwise frame_id/fps is used.",
    )
    group.add_argument("--dynamic_sequence_root", type=str, default="")
    group.add_argument("--dynamic_rgb_dir", type=str, default="rgb")
    group.add_argument("--dynamic_depth_npy_dir", type=str, default="depth_npy")
    group.add_argument("--dynamic_depth_png_dir", type=str, default="depth_png")
    group.add_argument("--dynamic_use_depth", action="store_true")
    group.add_argument(
        "--dynamic_use_depth_npy", action=BooleanOptionalAction, default=True
    )
    group.add_argument(
        "--dynamic_depth_is_metric", action=BooleanOptionalAction, default=True
    )
    group.add_argument("--dynamic_cameras", type=str, default="")
    group.add_argument("--dynamic_frame_start", type=int, default=0)
    group.add_argument("--dynamic_frame_end", type=int, default=-1)
    group.add_argument("--dynamic_frame_stride", type=int, default=1)
    group.add_argument("--dynamic_test_stride", type=int, default=0)
    group.add_argument(
        "--dynamic_camera_data_device", choices=("cpu", "cuda"), default="cpu"
    )
    group.add_argument("--use_semantic_feature_field", action="store_true")
    group.add_argument("--semantic_dynamic_threshold", type=float, default=0.5)
    group.add_argument("--semantic_dynamic_min_points", type=int, default=32)
    group.add_argument(
        "--dynamic_use_all_gaussians",
        action="store_true",
        help="Debug/baseline mode: bypass semantic foreground compaction and allow every Gaussian into Phase 1.",
    )
    group.add_argument("--semantic_mask_dir", type=str, default="mask")
    group.add_argument("--lambda_semantic_feature_field", type=float, default=0.01)
    group.add_argument("--semantic_feature_lr", type=float, default=0.0025)
    group.add_argument(
        "--dynamic_prior_path",
        type=str,
        default="",
        help="Optional DynamicPrior .pt used to initialize static/dynamic material probabilities before the shared Killing solver.",
    )
    group.add_argument(
        "--freeze_dynamic_prior_labels",
        action=BooleanOptionalAction,
        default=False,
        help="When --dynamic_prior_path is supplied, keep its hard dynamic/static mask fixed during plain rigid Killing fitting.",
    )
    group.add_argument(
        "--prior_use_soft_weights",
        action=BooleanOptionalAction,
        default=True,
        help="Use DynamicPrior probabilities directly; disable to threshold them before initialization.",
    )
    group.add_argument("--prior_hard_threshold", type=float, default=0.5)
    group.add_argument("--run_dynamic_prior_comparison", action="store_true")
    group.add_argument("--comparison_output_dir", type=str, default="")
    group.add_argument("--comparison_seed", type=int, default=20260904)
    group.add_argument(
        "--comparison_methods",
        type=str,
        default="baseline,denoise,flow_matching",
        help="Comma/space separated methods: baseline all_fg gt denoise flow_matching endpoint_transport_debug",
    )
    group.add_argument("--comparison_denoise_target_frames", type=str, default="")
    group.add_argument("--comparison_denoise_prob_bias", type=float, default=0.0)
    group.add_argument("--comparison_denoise_prob_temperature", type=float, default=0.35)
    group.add_argument("--comparison_prior_smooth_knn", type=int, default=16)
    group.add_argument("--comparison_prior_smooth_iters", type=int, default=0)
    group.add_argument("--comparison_prior_smooth_alpha", type=float, default=0.7)
    group.add_argument("--comparison_kpart_k", type=int, default=1)
    group.add_argument("--comparison_fit_joints", action="store_true")
    group.add_argument("--comparison_optimize_joint_trajectory", action="store_true")
    group.add_argument("--fm_endpoint_knn", type=int, default=16)
    group.add_argument("--fm_transport_temperature", type=float, default=0.02)
    group.add_argument("--fm_transport_robust_delta", type=float, default=0.02)
    group.add_argument("--fm_motion_scale", choices=("local_knn", "global_nn", "gaussian_scale"), default="local_knn")
    group.add_argument("--fm_confidence_entropy_scale", type=float, default=1.0)
    group.add_argument("--fm_prior_prob_bias", type=float, default=1.0)
    group.add_argument("--fm_prior_prob_temperature", type=float, default=0.35)
    group.add_argument("--fm_transport_chunk_size", type=int, default=1024)
    group.add_argument("--fm_hidden_dim", type=int, default=128)
    group.add_argument("--fm_layers", type=int, default=4)
    group.add_argument("--fm_lr", type=float, default=1.0e-3)
    group.add_argument("--fm_iters", type=int, default=1000)
    group.add_argument("--fm_batch_size", type=int, default=2048)
    group.add_argument("--fm_identity_weight", type=float, default=0.01)
    group.add_argument("--fm_smooth_weight", type=float, default=0.0)
    group.add_argument("--fm_integrator", choices=("euler", "rk4"), default="euler")
    group.add_argument("--fm_integration_steps", type=int, default=16)
    group.add_argument(
        "--dynamic_discovery_mode",
        choices=("legacy", "render_denoise"),
        default="legacy",
        help="Static/dynamic discovery source. render_denoise uses RGB/depth render-guided noising only.",
    )
    group.add_argument(
        "--render_denoise_probe_frames",
        type=str,
        default="",
        help="Comma-separated target frame ids for render-denoise. Empty means symmetric sampling over the sequence.",
    )
    group.add_argument("--render_denoise_num_probe_frames", type=int, default=4)
    group.add_argument("--render_denoise_cameras_per_probe", type=int, default=4)
    group.add_argument("--render_denoise_min_support_cameras", type=int, default=2)
    group.add_argument("--render_denoise_min_informative_frames", type=int, default=2)
    group.add_argument("--render_denoise_num_seeds", type=int, default=3)
    group.add_argument("--render_denoise_steps", type=int, default=8)
    group.add_argument("--render_denoise_noise_sigma", type=float, default=0.01)
    group.add_argument("--render_denoise_lr", type=float, default=1.0e-2)
    group.add_argument("--render_denoise_max_displacement", type=float, default=0.10)
    group.add_argument("--render_denoise_support_threshold", type=float, default=0.05)
    group.add_argument("--render_denoise_seed_consistency_threshold", type=float, default=0.65)
    group.add_argument("--render_denoise_view_consistency_threshold", type=float, default=0.55)
    group.add_argument("--render_denoise_dynamic_threshold", type=float, default=0.75)
    group.add_argument("--render_denoise_static_threshold", type=float, default=0.25)
    group.add_argument("--render_denoise_confidence_threshold", type=float, default=0.60)
    group.add_argument("--lambda_render_denoise_rgb", type=float, default=1.0)
    group.add_argument("--lambda_render_denoise_ssim", type=float, default=0.2)
    group.add_argument("--lambda_render_denoise_depth", type=float, default=0.1)
    group.add_argument("--lambda_render_denoise_zero", type=float, default=0.01)
    group.add_argument("--lambda_render_denoise_neighbor", type=float, default=0.001)
    group.add_argument("--render_gate_num_seeds", type=int, default=3)
    group.add_argument("--render_gate_steps", type=int, default=8)
    group.add_argument("--render_gate_lr", type=float, default=1.0e-2)
    group.add_argument("--render_gate_temperature", type=float, default=1.0)
    group.add_argument("--render_gate_noise_sigma", type=float, default=0.5)
    group.add_argument("--lambda_render_gate_prior", type=float, default=0.05)
    group.add_argument("--lambda_render_gate_neighbor", type=float, default=0.001)
    group.add_argument("--lambda_render_gate_entropy", type=float, default=0.01)
    group.add_argument(
        "--lambda_render_label",
        type=float,
        default=1.0,
        help="Weight of render-denoise static/dynamic evidence inside K-part soft responsibility optimization.",
    )
    group.add_argument(
        "--lambda_velocity_label",
        type=float,
        default=0.1,
        help="Weight of observed velocity-field unaries when render-denoise label evidence is enabled.",
    )
    group.add_argument("--use_dinov3", action="store_true")
    group.add_argument("--use_dino_feature_field", action="store_true")
    group.add_argument("--dinov3_root", type=str, default="dinov3")
    group.add_argument("--dinov3_feature_ext", type=str, default=".pt")
    group.add_argument("--dino_feature_field_dim", type=int, default=32)
    group.add_argument("--dino_feature_lr", type=float, default=0.0025)
    group.add_argument("--lambda_dino_feature_field", type=float, default=0.01)
    group.add_argument("--lambda_dino_feature_field_dynamic", type=float, default=0.003)
    group.add_argument("--use_sam3_feature_field", action="store_true")
    group.add_argument("--sam3_feature_root", type=str, default="")
    group.add_argument("--sam3_feature_dim", type=int, default=32)
    group.add_argument("--sam3_feature_lr", type=float, default=0.0025)
    group.add_argument("--lambda_sam3_feature_field", type=float, default=0.01)
    group.add_argument("--lambda_sam3_feature_field_dynamic", type=float, default=0.003)
    group.add_argument("--sam3_projection_path", type=str, default="")
    group.add_argument("--sam3_cluster_weight", type=float, default=0.0)
    group.add_argument("--sam3_feature_opacity_threshold", type=float, default=0.05)
    group.add_argument("--use_sam3_affinity_field", action="store_true")
    group.add_argument("--sam3_mask_root", type=str, default="")
    group.add_argument("--sam3_affinity_dim", type=int, default=16)
    group.add_argument("--sam3_affinity_iters", type=int, default=5000)
    group.add_argument("--sam3_affinity_lr", type=float, default=0.0025)
    group.add_argument("--sam3_mask_min_fg_fraction", type=float, default=0.001)
    group.add_argument("--sam3_mask_max_fg_fraction", type=float, default=0.95)
    group.add_argument("--sam3_mask_use_confidence", action=BooleanOptionalAction, default=True)
    group.add_argument("--sam3_affinity_pixels_per_iter", type=int, default=768)
    group.add_argument("--sam3_affinity_positive_pairs", type=int, default=512)
    group.add_argument("--sam3_affinity_intermediate_pairs", type=int, default=256)
    group.add_argument("--sam3_affinity_negative_pairs", type=int, default=512)
    group.add_argument("--sam3_affinity_pos_threshold", type=float, default=0.75)
    group.add_argument("--sam3_affinity_neg_threshold", type=float, default=0.25)
    group.add_argument("--sam3_affinity_loss", choices=("soft_bce", "l1"), default="soft_bce")
    group.add_argument("--sam3_affinity_temperature", type=float, default=0.1)
    group.add_argument("--sam3_affinity_bias", type=float, default=0.0)
    group.add_argument("--sam3_affinity_norm_weight", type=float, default=0.1)
    group.add_argument("--sam3_affinity_hard_pair_fraction", type=float, default=0.0)
    group.add_argument("--sam3_affinity_debug_interval", type=int, default=500)
    group.add_argument("--sam3_affinity_graph_knn", type=int, default=16)
    group.add_argument("--sam3_affinity_html_max_points", type=int, default=5000)
    correspondence_toggle.add_argument(
        "--use_motion_correspondence",
        dest="use_motion_correspondence",
        action="store_true",
    )
    correspondence_toggle.add_argument(
        "--no_motion_correspondence",
        dest="use_motion_correspondence",
        action="store_false",
    )
    group.add_argument("--dynamic_flow_dir", type=str, default="flow")
    group.add_argument("--correspondence_flow_max_side", type=int, default=256)
    group.add_argument("--correspondence_cycle_threshold", type=float, default=1.5)
    group.add_argument("--correspondence_dino_min_cosine", type=float, default=0.20)
    group.add_argument("--correspondence_min_confidence", type=float, default=0.05)
    group.add_argument("--correspondence_alpha_threshold", type=float, default=0.05)
    group.add_argument(
        "--observed_flow_velocity_ridge",
        type=float,
        default=1.0e-4,
        help="Relative ridge for direct multiview observed-flow -> 3D velocity inversion.",
    )
    group.add_argument(
        "--observed_flow_velocity_min_observations",
        type=int,
        default=6,
        help="Minimum Gaussian-camera equations contributing to one observed 3D velocity target.",
    )
    group.add_argument(
        "--observed_flow_velocity_min_condition",
        type=float,
        default=1.0e-4,
        help="Minimum conditioning ratio for accepting a direct observed-flow 3D velocity target.",
    )
    group.add_argument(
        "--observed_flow_velocity_min_cameras",
        type=int,
        default=3,
        help="Minimum number of distinct physical cameras supporting one observed 3D velocity target.",
    )
    group.add_argument(
        "--observed_flow_velocity_max_reprojection_px",
        type=float,
        default=0.75,
        help="Reject observed 3D velocity hypotheses whose moving-model RMS reprojection residual exceeds this many pixels.",
    )
    group.add_argument(
        "--observed_flow_require_static_improvement",
        action=BooleanOptionalAction,
        default=None,
        help="Accept a moving 3-D target only when it beats the explicit zero-velocity model.",
    )
    group.add_argument(
        "--observed_flow_static_improvement_abs",
        type=float,
        default=0.05,
        help="Required absolute moving-vs-static RMS improvement in pixels.",
    )
    group.add_argument(
        "--observed_flow_static_improvement_rel",
        type=float,
        default=0.15,
        help="Required relative moving-vs-static RMS improvement.",
    )
    group.add_argument(
        "--multiview_flow_noise_px",
        type=float,
        default=0.25,
        help="Per-view optical-flow noise scale used by the explicit static v=0 hypothesis.",
    )
    group.add_argument(
        "--multiview_flow_model_margin_px",
        type=float,
        default=0.05,
        help="Required reprojection improvement before the moving hypothesis receives positive evidence.",
    )
    group.add_argument(
        "--multiview_flow_model_temperature_px", type=float, default=0.12
    )
    group.add_argument(
        "--eulerian_assignment_init_floor",
        type=float,
        default=0.20,
        help="Lower dynamic-probability bound for weak Eulerian initialization from the original gradient prior.",
    )
    group.add_argument(
        "--eulerian_assignment_init_ceiling",
        type=float,
        default=0.80,
        help="Upper dynamic-probability bound for weak Eulerian initialization; labels remain trainable from the first update.",
    )
    group.add_argument(
        "--eulerian_static_gate_threshold",
        type=float,
        default=0.85,
        help="Cell static probability above which the forward tracer velocity is hard-zeroed via a straight-through gate.",
    )
    group.add_argument(
        "--eulerian_static_gate_temperature",
        type=float,
        default=0.04,
        help="Backward softness of the straight-through hard static gate.",
    )
    group.add_argument(
        "--lambda_eulerian_hard_static_speed",
        type=float,
        default=250.0,
        help="Strong A-field penalty forcing currently hard-static cells to converge to v_c=0.",
    )
    group.add_argument(
        "--lambda_assignment_dino_boundary",
        type=float,
        default=0.20,
        help="DINO-aware label smoothness: similar neighbouring descriptors resist a static/dynamic boundary.",
    )
    group.add_argument(
        "--dino_boundary_rigidity_strength",
        type=float,
        default=1.0,
        help="Use DINO discontinuity only where current motion labels disagree to permit a material rigidity boundary.",
    )
    group.add_argument(
        "--observed_flow_target_refresh_iters",
        type=int,
        default=5,
        help="Recompute observed-flow 3D velocity targets every N local iterations inside a window.",
    )
    group.add_argument(
        "--lambda_observed_flow_velocity",
        type=float,
        default=20.0,
        help="Direct magnitude supervision from observed multiview flow-derived 3D velocity targets.",
    )
    group.add_argument(
        "--lambda_observed_flow_direction",
        type=float,
        default=10.0,
        help="Explicit cosine-direction supervision toward observed multiview flow-derived 3D velocities.",
    )
    group.add_argument(
        "--lambda_motion_projected_velocity_flow",
        type=float,
        default=2.0,
        help="Weight of the differentiable rasterized optical-flow loss on project_target(x + v dt) - project_source(x).",
    )
    group.add_argument(
        "--phase1_motion_representation",
        choices=("eulerian_curl_strain", *RIGID_KILLING_REPRESENTATIONS),
        default="eulerian_curl_strain",
    )
    group.add_argument("--rigid_solver_max_iters", type=int, default=20)
    group.add_argument("--rigid_solver_tol", type=float, default=1.0e-6)
    group.add_argument("--rigid_solver_irls_iters", type=int, default=5)
    group.add_argument("--rigid_solver_huber_delta", type=float, default=0.05)
    group.add_argument("--rigid_min_valid_cells", type=int, default=8)
    group.add_argument("--rigid_min_distinct_cameras", type=int, default=3)
    group.add_argument("--rigid_angular_eps", type=float, default=1.0e-4)
    group.add_argument("--rigid_speed_temporal_weight", type=float, default=0.0)
    group.add_argument("--rigid_assignment_improvement_margin", type=float, default=0.05)
    group.add_argument("--rigid_assignment_temperature", type=float, default=0.12)
    group.add_argument("--rigid_assignment_prior_weight", type=float, default=0.25)
    group.add_argument("--rigid_max_condition_number", type=float, default=1.0e10)
    group.add_argument("--rigid_static_displacement_tolerance", type=float, default=1.0e-6)
    group.add_argument("--rigid_pairwise_audit_points", type=int, default=512)
    group.add_argument(
        "--rigid_analytic_validation",
        action="store_true",
        help=(
            "After the measured-flow solve is complete, compare its frozen result "
            "with dataset rigid-motion GT. GT is never used as an observation."
        ),
    )
    group.add_argument(
        "--rigid_analytic_moving_part",
        type=int,
        default=1,
        help="Moving part label used only by post-solve analytic validation.",
    )
    group.add_argument(
        "--phase1_neutral_rigid_discovery",
        action="store_true",
        help=(
            "Initialize every semantic-foreground Gaussian at P(dynamic)=0.5, "
            "fit one ungated rigid Killing mode, then assign material labels once."
        ),
    )
    group.add_argument("--rigid_responsibility_training", action="store_true")
    group.add_argument("--rigid_responsibility_cycles", type=int, default=4)
    group.add_argument("--rigid_responsibility_temperature_start", type=float, default=0.50)
    group.add_argument("--rigid_responsibility_temperature_end", type=float, default=0.10)
    group.add_argument("--rigid_responsibility_margin", type=float, default=0.05)
    group.add_argument("--rigid_responsibility_dino", action=BooleanOptionalAction, default=True)
    group.add_argument("--rigid_responsibility_dino_sigma", type=float, default=0.15)
    group.add_argument("--rigid_responsibility_dino_strength", type=float, default=0.25)
    group.add_argument("--rigid_responsibility_dino_knn", type=int, default=8)
    group.add_argument("--rigid_responsibility_static_threshold", type=float, default=0.20)
    group.add_argument("--rigid_responsibility_rigid_threshold", type=float, default=0.80)
    group.add_argument("--kpart_rigid_discovery", action="store_true")
    group.add_argument("--kpart_support", choices=("eulerian", "lagrangian"), default=None)
    group.add_argument("--kpart_cycles", type=int, default=None)
    group.add_argument("--kpart_temperature_start", type=float, default=1.0)
    group.add_argument("--kpart_temperature_end", type=float, default=0.2)
    group.add_argument("--kpart_anneal_cycles", "--part_discovery_anneal_cycles", type=int, default=8)
    group.add_argument("--kpart_plateau_cycles", "--part_discovery_plateau_cycles", type=int, default=5)
    group.add_argument("--kpart_confidence_threshold", type=float, default=0.20)
    group.add_argument("--enable_rigid_part_discovery", action="store_true")
    group.add_argument("--rigid_part_outer_iters", type=int, default=3)
    group.add_argument("--rigid_part_local_motion_steps", type=int, default=5)
    group.add_argument("--rigid_part_label_steps", type=int, default=10)
    group.add_argument("--rigid_part_raw_lr", type=float, default=5.0e-3)
    group.add_argument("--rigid_part_render_frames", type=int, default=4)
    group.add_argument("--rigid_part_render_cameras", type=int, default=2)
    group.add_argument("--rigid_part_refine_iters", type=int, default=25)
    group.add_argument("--rigid_part_lr", type=float, default=5.0e-2)
    group.add_argument("--rigid_part_label_temperature", type=float, default=1.0)
    group.add_argument("--rigid_part_label_temperature_final", type=float, default=0.7)
    group.add_argument("--rigid_part_knn", type=int, default=16)
    group.add_argument("--rigid_part_max_frames", type=int, default=20)
    group.add_argument("--rigid_part_min_fraction", type=float, default=0.01)
    group.add_argument("--rigid_part_final_assignment_temperature", type=float, default=0.01)
    group.add_argument("--rigid_part_endpoint_chunk", type=int, default=2048)
    group.add_argument("--enable_nested_killing_part_discovery", action="store_true")
    group.add_argument("--part_embedding_dim", type=int, default=8)
    group.add_argument("--part_embedding_init_std", type=float, default=0.05)
    group.add_argument("--part_outer_iters", type=int, default=3)
    group.add_argument("--part_outer_embedding_steps", type=int, default=10)
    group.add_argument("--part_embedding_lr", type=float, default=1.0e-2)
    group.add_argument("--part_temperature", type=float, default=1.0)
    group.add_argument("--part_temperature_final", type=float, default=0.25)
    group.add_argument("--part_final_assignment_temperature", type=float, default=0.01)
    group.add_argument("--static_anchor_distance_ratio", type=float, default=0.002)
    group.add_argument("--static_anchor_use_mutual_nn", action=BooleanOptionalAction, default=True)
    group.add_argument("--static_anchor_use_neighborhood_check", action=BooleanOptionalAction, default=False)
    group.add_argument("--static_anchor_neighborhood_k", type=int, default=8)
    group.add_argument("--killing_inner_steps_per_frame", type=int, default=10)
    group.add_argument("--nested_killing_control_points", type=int, default=64)
    group.add_argument(
        "--nested_killing_kernel_scale",
        type=float,
        default=0.0,
        help="RBF length scale for the nested local differential Killing field. 0 = auto from control spacing.",
    )
    group.add_argument("--killing_lr", type=float, default=1.0e-2)
    group.add_argument("--lambda_inner_rgb", type=float, default=1.0)
    group.add_argument("--lambda_inner_ssim", type=float, default=0.0)
    group.add_argument("--lambda_inner_depth", type=float, default=0.1)
    group.add_argument("--lambda_inner_observed_velocity", type=float, default=0.0)
    group.add_argument("--lambda_inner_strain", type=float, default=10.0)
    group.add_argument("--lambda_inner_omega_consistency", type=float, default=1.0)
    group.add_argument("--lambda_inner_b_consistency", type=float, default=1.0)
    group.add_argument("--lambda_inner_twist_temporal", type=float, default=0.01)
    group.add_argument("--lambda_inner_twist_acceleration", type=float, default=0.0)
    group.add_argument("--lambda_inner_field_separation", type=float, default=0.0)
    group.add_argument("--nested_motion_separation_margin", type=float, default=0.01)
    group.add_argument("--lambda_static_anchor", type=float, default=10.0)
    group.add_argument("--lambda_outer_candidate_nn", type=float, default=1.0)
    group.add_argument("--lambda_outer_final_nn", type=float, default=1.0)
    group.add_argument("--lambda_outer_global_com", type=float, default=1.0)
    group.add_argument("--lambda_outer_global_inertia", type=float, default=1.0)
    group.add_argument("--lambda_outer_part_com", type=float, default=2.0)
    group.add_argument("--lambda_outer_part_inertia", type=float, default=2.0)
    group.add_argument("--lambda_outer_embedding_motion", type=float, default=0.0)
    group.add_argument("--lambda_outer_candidate_margin", type=float, default=0.1)
    group.add_argument("--lambda_outer_entropy", type=float, default=0.001)
    group.add_argument("--lambda_outer_occupancy", type=float, default=0.01)
    group.add_argument("--outer_candidate_margin", type=float, default=0.001)
    # ---- FIRM v3: local Killing field, confidence gating, guards ------------
    group.add_argument("--killing_jacobian_mode", type=str, default="analytic", choices=["analytic", "lsq"])
    group.add_argument("--killing_neighbor_k", type=int, default=16)
    group.add_argument("--killing_neighbor_chunk", type=int, default=4096)
    group.add_argument("--killing_camera_chunk", type=int, default=0)
    group.add_argument("--killing_field_diagnostic_points", type=int, default=512)
    group.add_argument("--observed_velocity_c_min", type=float, default=0.60)
    group.add_argument("--observed_velocity_gamma", type=float, default=2.0)
    group.add_argument("--observed_velocity_rho_max", type=float, default=0.20)
    group.add_argument("--observed_velocity_min_high_conf", type=int, default=16)
    group.add_argument("--observed_velocity_huber_delta", type=float, default=1.0)
    group.add_argument("--field_separation_alpha_v", type=float, default=1.0)
    group.add_argument("--field_separation_alpha_omega", type=float, default=1.0)
    group.add_argument("--killing_field_symmetry_break", type=float, default=1.0e-4)
    group.add_argument("--lambda_inner_rigid_speed", type=float, default=1.0)
    group.add_argument("--partwise_twist_reg", type=float, default=1.0e-8)
    # ---- Render-centric outer E-step (§43 starting weights) ----------------
    group.add_argument("--lambda_outer_video", type=float, default=1.0)
    group.add_argument("--lambda_outer_label_graph", type=float, default=0.05)
    group.add_argument("--lambda_outer_kinetic", type=float, default=0.05)
    group.add_argument("--rho_3d_gradient_cap", type=float, default=0.30)
    group.add_argument("--rho_local_gradient_cap", type=float, default=0.50)
    # ---- Pixel-mediated final-frame Gaussian matching (section 45 weights) --
    group.add_argument("--enable_pixel_matching", action="store_true", default=False)
    group.add_argument("--raster_top_contributors", type=int, default=4)
    group.add_argument("--raster_contributor_min_weight", type=float, default=1.0e-4)
    group.add_argument("--lambda_pixel_candidate", type=float, default=1.0)
    group.add_argument("--lambda_pixel_margin", type=float, default=0.1)
    group.add_argument("--pixel_candidate_margin", type=float, default=0.05)
    group.add_argument("--pixel_footprint_window", type=int, default=3)
    group.add_argument("--pixel_sigma_z_ratio", type=float, default=0.02)
    group.add_argument("--lambda_pixel_overlap", type=float, default=0.5)
    group.add_argument("--lambda_pixel_xyz", type=float, default=1.0)
    group.add_argument("--lambda_pixel_depth", type=float, default=0.5)
    group.add_argument("--lambda_pixel_coverage", type=float, default=0.02)
    group.add_argument("--graph_iterations", type=int, default=1000)
    group.add_argument("--graph_tolerance", type=float, default=1.0e-4)
    group.add_argument("--boundary_aware_graph", action="store_true", default=False)
    group.add_argument("--per_class_seeds", action="store_true", default=True)
    group.add_argument("--global_seed_threshold", dest="per_class_seeds", action="store_false")
    group.add_argument("--min_seeds_per_class", type=int, default=32)
    group.add_argument("--absolute_margin_floor", type=float, default=0.05)
    group.add_argument("--tau_render_preference", type=float, default=1.0)
    group.add_argument("--min_support_views", type=int, default=2)
    group.add_argument("--render_seed_confidence", type=float, default=0.75)
    group.add_argument("--outer_label_mode", type=str, default="learned")
    group.add_argument("--pixel_delta_z_ratio", type=float, default=0.15)
    group.add_argument("--kinetic_alpha_static", type=float, default=0.0)
    group.add_argument("--killing_warm_start", action="store_true", default=True)
    group.add_argument("--no_killing_warm_start", dest="killing_warm_start", action="store_false")
    group.add_argument("--outer_3d_cap_interval", type=int, default=1)
    group.add_argument("--outer_label_graph_sigma", type=float, default=1.0)
    group.add_argument("--outer_video_frame_block", type=int, default=0)
    group.add_argument("--outer_video_shuffle_frames", action="store_true", default=True)
    group.add_argument("--outer_video_no_shuffle", dest="outer_video_shuffle_frames", action="store_false")
    group.add_argument("--outer_video_debug_every", type=int, default=0)
    group.add_argument("--outer_top_gradient_points", type=int, default=256)
    group.add_argument("--kinetic_alpha_omega", type=float, default=1.0)
    group.add_argument("--kinetic_alpha_rigid", type=float, default=1.0)
    group.add_argument("--kinetic_alpha_strain", type=float, default=1.0)
    group.add_argument("--multiview_grad_probe_every", type=int, default=0)
    group.add_argument("--multiview_debug_every", type=int, default=0)
    group.add_argument("--multiview_visibility_floor", type=float, default=0.1)
    group.add_argument("--lambda_multiview_view_var", type=float, default=0.05)
    group.add_argument("--nested_time_unit", type=str, default="auto", choices=["auto", "frame", "seconds"])
    # ---- Endpoint-contrastive static/dynamic initialization (spec §23) -----
    group.add_argument("--init_mode", type=str, default="legacy",
                       choices=["legacy", "endpoint_contrastive", "high_precision_bootstrap"],
                       help="legacy = diffused static anchors + random moving logits; "
                            "endpoint_contrastive = learn binary static/dynamic first, "
                            "then split the dynamic support into K parts")
    group.add_argument("--init_graph_k", type=int, default=16)
    group.add_argument("--init_binary_iters", type=int, default=300)
    group.add_argument("--init_binary_lr", type=float, default=1.0e-2)
    group.add_argument("--init_static_quantile", type=float, default=0.10)
    group.add_argument("--init_dynamic_quantile", type=float, default=0.10)
    group.add_argument("--init_contrastive_margin", type=float, default=0.5)
    group.add_argument("--lambda_init_same", type=float, default=1.0)
    group.add_argument("--lambda_init_diff", type=float, default=0.5)
    group.add_argument("--lambda_init_static", type=float, default=1.0)
    group.add_argument("--lambda_init_dynamic", type=float, default=0.25)
    group.add_argument("--lambda_init_trusted_static", type=float, default=2.0)
    # ---- Soft label outer cycle (CODEX_soft_labels §46) ---------------------
    group.add_argument("--outer_soft_labels", action=BooleanOptionalAction, default=True,
                       help="soft P is the persistent label state; z=argmax(P) exists "
                            "only inside an inner Killing solve. --no-outer_soft_labels "
                            "restores the legacy hard render-graph behaviour (ablation)")
    # ---- High-precision bootstrap initialization (§22) ----------------------
    group.add_argument("--bootstrap_dynamic_quantile", type=float, default=0.98,
                       help="§4: A_D = {d >= Q_q(d)}; 0.98 means the top 2%%")
    group.add_argument("--bootstrap_static_quantile", type=float, default=0.10)
    group.add_argument("--lambda_bootstrap_knn", type=float, default=1.0)
    group.add_argument("--lambda_bootstrap_anchor", type=float, default=1.0)
    group.add_argument("--lambda_bootstrap_dynamic_core", type=float, default=1.0)
    group.add_argument("--lambda_bootstrap_geometric_static", type=float, default=0.5)
    group.add_argument("--lambda_bootstrap_polar", type=float, default=0.0,
                       help="§3: MUST stay 0 in bootstrap mode; non-zero is the legacy ablation")
    group.add_argument("--bootstrap_iterations", type=int, default=200)
    group.add_argument("--init_dynamic_anchor_quantile", type=float, default=0.95,
                       help="§5: A_D = {d >= Q_q(d)}; 0.95 means the top 5%%")
    group.add_argument("--lambda_init_pair", type=float, default=1.0,
                       help="SUPERSEDED by --lambda_init_smooth (corrections §4)")
    group.add_argument("--lambda_init_polar", type=float, default=0.5,
                       help="corrections §9: this is now lambda_polar_MAX, ramped in")
    group.add_argument("--lambda_init_anchor", type=float, default=1.0)
    group.add_argument("--init_pair_confidence_gamma", type=float, default=2.0)
    # ---- Algorithmic corrections (CODEX_soft_label_algorithmic_corrections) --
    group.add_argument("--lambda_init_smooth", type=float, default=1.0,
                       help="§5: positive-only graph smoothness, replaces the pairwise BCE")
    group.add_argument("--init_positive_similarity_gamma", type=float, default=2.0,
                       help="§5: c_ij = (s_ij^+)^gamma")
    group.add_argument("--init_endpoint_logit_bias", type=float, default=0.2,
                       help="§8: weak zero-centred endpoint asymmetry, eps_d")
    group.add_argument("--polarization_ramp_start", type=float, default=0.20)
    group.add_argument("--polarization_ramp_end", type=float, default=0.70)
    group.add_argument("--outer_confidence_mode", type=str,
                       default="candidate_validity_entropy",
                       choices=["candidate_validity_entropy", "normalized_margin"],
                       help="§15/§17: how alpha_i is chosen; normalized_margin is legacy")
    # ---- Motion-boundary 3DGS split (CODEX_motion_boundary_3dgs_split) ------
    group.add_argument("--enable_motion_boundary_split", action="store_true", default=False,
                       help="§4: ONE split event, after OUTER 0 and before INNER 1")
    group.add_argument("--boundary_split_local_class_min", type=float, default=0.15)
    group.add_argument("--boundary_split_extent_sigma", type=float, default=2.0)
    group.add_argument("--boundary_split_offset_sigma", type=float, default=0.5)
    group.add_argument("--boundary_split_max_offset_sigma", type=float, default=1.0)
    group.add_argument("--boundary_split_gradient_ridge", type=float, default=1.0e-6)
    group.add_argument("--boundary_split_label_mix", type=float, default=0.05)
    group.add_argument("--boundary_split_max_fraction", type=float, default=0.01)
    group.add_argument("--boundary_split_max_count", type=int, default=0,
                       help="0 = no absolute cap")
    group.add_argument("--boundary_split_size_threshold", type=float, default=0.0,
                       help="0 = use percent_dense * scene_extent")
    group.add_argument("--boundary_split_confident_class", type=float, default=0.6)
    group.add_argument("--boundary_split_opacity_mode", type=str, default="weighted_mass",
                       choices=["weighted_mass", "original_copy"])
    # ---- Render-response contrastive outer loss (§34) -----------------------
    group.add_argument("--enable_render_response_contrast", action="store_true", default=False)
    group.add_argument("--render_contrast_steps", type=int, default=10)
    group.add_argument("--render_contrast_lr", type=float, default=1.0e-2)
    group.add_argument("--render_contrast_positive_threshold", type=float, default=0.70)
    group.add_argument("--render_contrast_negative_threshold", type=float, default=0.70)
    group.add_argument("--render_contrast_min_shared_obs", type=int, default=2)
    group.add_argument("--lambda_render_contrast_preserve", type=float, default=1.0)
    group.add_argument("--lambda_render_contrast_agree", type=float, default=1.0)
    group.add_argument("--lambda_render_contrast_disagree", type=float, default=0.25)
    group.add_argument("--render_contrast_sign_temperature", type=float, default=0.10)
    # ---- Robust inner Killing field (§33, §34, §3, §5, §6, §7) --------------
    group.add_argument("--inner_solver", type=str, default="per_frame",
                       choices=["per_frame", "shared_scalar"],
                       help="shared_scalar = v(x,t) = s(t) Phi(x): ONE spatial Killing "
                            "field for the whole video plus a scalar speed per frame")
    group.add_argument("--inner_alternation_rounds", type=int, default=3)
    group.add_argument("--inner_min_fit_points", type=int, default=6)
    group.add_argument("--inner_projection_reg", type=float, default=1.0e-8)
    group.add_argument("--inner_render_frame_stride", type=int, default=1,
                       help="\u00a758: reference implementation is 1 (every frame); frames are mini-batched by --inner_render_frame_batch with full epoch coverage")
    group.add_argument("--lambda_rigid_proj", type=float, default=1.0)
    group.add_argument("--inner_spatial_steps", type=int, default=80)
    group.add_argument("--inner_scalar_steps", type=int, default=80)
    group.add_argument("--inner_motion_prior", type=str, default="none",
                       choices=["none", "revolute", "prismatic"],
                       help="§30-§33: OFF by default; only toggles loss weights")
    group.add_argument("--lambda_revolute_prior", type=float, default=0.0)
    group.add_argument("--lambda_prismatic_prior", type=float, default=0.0)
    group.add_argument("--inner_outlier_mad_scale", type=float, default=3.5)
    group.add_argument("--inner_outlier_local_dynamic_min", type=float, default=0.35)
    group.add_argument("--inner_fit_confidence_floor", type=float, default=0.50)
    group.add_argument("--inner_fit_confidence_gamma", type=float, default=2.0)
    group.add_argument("--inner_cluster_core_confidence", type=float, default=0.80)
    group.add_argument("--inner_cluster_knn", type=int, default=16)
    group.add_argument("--inner_cluster_confidence_gamma", type=float, default=1.0)
    group.add_argument("--lambda_exact_omega", type=float, default=1.0)
    group.add_argument("--lambda_exact_translation", type=float, default=1.0)
    group.add_argument("--lambda_rigid_field_probe", type=float, default=0.0)
    group.add_argument("--lambda_scalar_consistency", type=float, default=1.0)
    group.add_argument("--lambda_s_continuity", type=float, default=0.01)
    group.add_argument("--lambda_s_flip", type=float, default=0.01)
    group.add_argument("--lambda_s_jerk", type=float, default=0.0)
    # ---- CODEX_DECOUPLED joint geometry / scalar trajectory -----------------
    group.add_argument("--inner_decoupled_ablation", type=str, default="E",
                       choices=["A", "B", "C", "D", "E"],
                       help="§59 ladder for the Stage-B losses: A = joint geometry only, "
                            "B += velocity magnitude, C += whole-video render, "
                            "D += final PCD, E += continuity/flip/jerk")
    group.add_argument("--lambda_frame_joint", type=float, default=1.0,
                       help="§10: all-frame joint consistency under the analytical alpha*")
    group.add_argument("--lambda_final_pcd", type=float, default=1.0,
                       help="§25: trimmed-Chamfer endpoint constraint (Stage B only)")
    group.add_argument("--inner_frame_robust", type=str, default="huber",
                       choices=["huber", "charbonnier", "l2"],
                       help="§10's rho on the residual norm")
    group.add_argument("--inner_frame_robust_scale", type=float, default=1.0,
                       help="rho's delta as a multiple of the median observed speed, so the "
                            "cutoff is scale-free across datasets")
    group.add_argument("--inner_robust_tau_scale", type=float, default=2.0,
                       help="§38's IRLS tau, same units as --inner_frame_robust_scale")
    group.add_argument("--inner_irls_steps", type=int, default=2,
                       help="§38: 1 disables reweighting, 2+ reweights the joint fit")
    # ---- CODEX_MERGED robust support (Parts I-VII) --------------------------
    # ---- Sparse alpha-aware error-state IEKF (section 33) ------------------
    # Every one of these is OFF/neutral by default: with --use_sparse_alpha_iekf
    # unset the legacy render and training path is unchanged (section 0 rule 2).
    group.add_argument("--use_sparse_alpha_iekf", action="store_true", default=False,
                       help="enable the analytic-Jacobian IEKF motion estimator")
    group.add_argument("--iekf_frozen_appearance", action="store_true", default=True,
                       help="\u00a73: freeze colour, opacity and scales during motion estimation")
    group.add_argument("--iekf_strict_frozen_radiance", action="store_true", default=True,
                       help="\u00a73.1: evaluated per-Gaussian RGB is constant during one update")
    group.add_argument("--iekf_use_rgb", action="store_true", default=True)
    group.add_argument("--no_iekf_use_rgb", action="store_false", dest="iekf_use_rgb")
    group.add_argument("--iekf_use_depth", action="store_true", default=True)
    group.add_argument("--no_iekf_use_depth", action="store_false", dest="iekf_use_depth")
    group.add_argument("--iekf_max_iters", type=int, default=10)
    group.add_argument("--iekf_lm_initial", type=float, default=1.0e-3)
    group.add_argument("--iekf_lm_min", type=float, default=1.0e-8)
    group.add_argument("--iekf_lm_max", type=float, default=1.0e8)
    group.add_argument("--iekf_line_search_shrink", type=float, default=0.5)
    group.add_argument("--iekf_alpha_min", type=float, default=1.0e-3)
    group.add_argument("--iekf_pcg_max_iters", type=int, default=30)
    group.add_argument("--iekf_pcg_tol", type=float, default=1.0e-4)
    group.add_argument("--iekf_robust_measurement", action="store_true", default=True,
                       help="\u00a713.1: Student-t IRLS reweighting of the measurement precision")
    group.add_argument("--no_iekf_robust_measurement", action="store_false",
                       dest="iekf_robust_measurement")
    group.add_argument("--iekf_student_t_nu", type=float, default=5.0)
    group.add_argument("--iekf_depth_sigma_model", type=str, default="constant:0.02:0.0",
                       help="\u00a76: sigma_D(z) as kind[:a[:b]] -- constant/linear/quadratic. "
                            "Depth noise that grows with range is real, so this is a sensor "
                            "model, not a loss weight")
    group.add_argument("--iekf_rgb_sigma", type=float, default=0.1)
    group.add_argument("--iekf_process_noise_pos", type=float, default=1.0e-3)
    group.add_argument("--iekf_process_noise_rot", type=float, default=1.0e-3)
    group.add_argument("--iekf_process_noise_vel", type=float, default=1.0e-1)
    group.add_argument("--iekf_process_noise_omega", type=float, default=1.0e-1)
    # V2 section 13 Option B: --iekf_cache_derivative_basis is REMOVED from the
    # production CLI. Mode B (precomputed per-Gaussian pose basis derivatives)
    # is not implemented, and section 13 is explicit that "raises if enabled"
    # does not count as an implemented feature. Mode A -- recomputing the
    # differentials inside each JVP/VJP -- is the validated path and the only
    # one. Mode B is recorded as a future optimization, not a flag.
    group.add_argument("--iekf_debug_jacobians", action="store_true", default=False)
    group.add_argument("--iekf_debug_adjoint_test", action="store_true", default=False,
                       help="\u00a735 T8: assert <Hv,z> == <v,H^T z> each frame")
    group.add_argument("--iekf_multiview", action="store_true", default=True,
                       help="\u00a725: fuse all synchronized cameras")
    group.add_argument("--iekf_eps_depth", type=float, default=1.0e-6)
    group.add_argument("--iekf_min_alpha_for_depth", type=float, default=0.05,
                       help="\u00a727: a pixel with no coverage has no depth measurement")
    # ---- V2 section 4: explicit scaled trust region ------------------------
    group.add_argument("--iekf_trust_region", action="store_true", default=True,
                       help="V2 \u00a74: explicit scaled trust region with Steihaug PCG. ON by "
                            "default -- V2 \u00a71.3 showed the LM+line-search path accepts ZERO "
                            "steps at a 400-iteration PCG budget because early termination was "
                            "acting as the step controller")
    group.add_argument("--no_iekf_trust_region", action="store_false",
                       dest="iekf_trust_region",
                       help="V2 \u00a71.3's reproducible failure mode, as a controlled ablation")
    group.add_argument("--iekf_trust_radius_init", type=float, default=1.0,
                       help="V2 \u00a74.1: initial radius in RMS prior-sigma units (scene-size "
                            "independent)")
    group.add_argument("--iekf_trust_radius_max", type=float, default=100.0)
    group.add_argument("--iekf_trust_radius_min", type=float, default=1.0e-8)
    group.add_argument("--iekf_trust_eta_accept", type=float, default=0.1)
    group.add_argument("--iekf_trust_eta_weak", type=float, default=0.25)
    group.add_argument("--iekf_trust_eta_good", type=float, default=0.75)
    group.add_argument("--iekf_scale_sigma_mu", type=float, default=0.05,
                       help="V2 \u00a74.1: translation scale of the trust-region metric")
    group.add_argument("--iekf_scale_sigma_phi", type=float, default=0.05)
    group.add_argument("--iekf_scale_sigma_v", type=float, default=0.5)
    group.add_argument("--iekf_scale_sigma_omega", type=float, default=0.5)
    group.add_argument("--iekf_scale_from_prior", action="store_true", default=False,
                       help="V2 \u00a74.1: take the metric from the prior information diagonal "
                            "instead of fixed physical scales")
    group.add_argument("--iekf_forcing_c", type=float, default=0.5,
                       help="V2 \u00a74.3: inexact-Newton forcing constant")
    group.add_argument("--iekf_forcing_min", type=float, default=1.0e-6)
    group.add_argument("--iekf_forcing_max", type=float, default=0.01,
                       help="V2 \u00a74.3: loosest accepted relative linear residual. Tight by "
                            "default because the trust region makes an accurate solve safe")
    # ---- V2 section 3: grouped / supernode state ---------------------------
    group.add_argument("--iekf_supernodes", action="store_true", default=False,
                       help="V2 \u00a73: estimate a 6-DoF rigid tangent per canonical supernode "
                            "instead of per Gaussian")
    group.add_argument("--iekf_supernode_size", type=int, default=64,
                       help="V2 \u00a73.1: target Gaussians per supernode")
    group.add_argument("--iekf_supernode_count", type=int, default=0,
                       help="V2 \u00a73.1: exact supernode count (0 = derive from the target size)")
    group.add_argument("--iekf_supernode_knn", type=int, default=8,
                       help="V2 \u00a73.1: centroid-kNN degree of the canonical supernode graph")
    group.add_argument("--iekf_supernode_seed", type=int, default=0)
    # ---- V2 section 6: support inference -----------------------------------
    group.add_argument("--iekf_support_k_mad", type=float, default=8.0,
                       help="V2 \u00a76.3: robust median+k*MAD threshold on the "
                            "alpha-aware achievable-motion gain that separates "
                            "moving supernodes from static ones")
    group.add_argument("--iekf_support_max_moving", type=float, default=0.5,
                       help="V2 \u00a76.3: if more than this fraction is flagged "
                            "moving the threshold found no outlier population, "
                            "and only the strongest are kept")
    group.add_argument("--no_support_inference", action="store_true", default=False,
                       help="V2 \u00a76 ablation: fit the joint over every Gaussian "
                            "instead of the inferred moving support")
    # ---- V2 section 5: observability-aware update --------------------------
    group.add_argument("--iekf_observability_tau", type=float, default=1.0e-2,
                       help="V2 \u00a75.2: scaled Fisher eigenvalue below which a direction is "
                            "not reported as observed")
    group.add_argument("--iekf_project_weak", action="store_true", default=False,
                       help="V2 \u00a75.2: project the measurement correction out of "
                            "unobservable directions")
    group.add_argument("--iekf_inflate_weak", action="store_true", default=True,
                       help="V2 \u00a75.2: inflate posterior uncertainty in weak directions "
                            "rather than inventing motion")
    group.add_argument("--no_iekf_inflate_weak", action="store_false",
                       dest="iekf_inflate_weak")
    group.add_argument("--iekf_observability_report", action="store_true", default=True)
    # ---- V2 section 12: radiance mode --------------------------------------
    group.add_argument("--iekf_allow_view_dependent_radiance", action="store_true",
                       default=False,
                       help="V2 \u00a712.2: deliberately run the view-dependent radiance mode "
                            "whose analytic colour derivative is NOT in H. Only for measuring "
                            "that modelling error")
    # ---- joint reduction (sections 28-30) ----------------------------------
    group.add_argument("--joint_reduction", action="store_true", default=False)
    group.add_argument("--joint_models", type=str, default="prismatic,revolute",
                       help="\u00a70 rule 13: prismatic and revolute only; screw is out of scope")
    group.add_argument("--joint_reduce_after_convergence", action="store_true", default=False)
    # ---- force-enabled dynamics (section 32) -------------------------------
    group.add_argument("--use_wrench_dynamics", action="store_true", default=False,
                       help="\u00a732: optional; the method works from RGB-D alone")
    group.add_argument("--use_viscous_damping", action="store_true", default=True,
                       help="\u00a732.1: ON in the initial physical baseline")
    group.add_argument("--no_use_viscous_damping", action="store_false",
                       dest="use_viscous_damping")
    group.add_argument("--use_gravity", action="store_true", default=False,
                       help="\u00a732.1: implemented but OFF by default")
    group.add_argument("--use_coulomb_friction", action="store_true", default=False,
                       help="\u00a732.1: implemented but OFF by default")
    group.add_argument("--debug_gt_fitting_support", action="store_true", default=False,
                       help="CODEX_DEBUG \u00a733/\u00a735: replace the FITTING support with GT for the "
                            "storage B/D ablations. Diagnostic upper bound only, never an "
                            "unsupervised result; loudly logged when active")
    group.add_argument("--assert_soft_inner_probs", action="store_true", default=False,
                       help="CODEX_DEBUG \u00a710: warn (never crash) when the inner support "
                            "builder receives near-one-hot probabilities")
    group.add_argument("--inner_support_knn", type=int, default=16,
                       help="§5: KNN inside the moving candidate set, for local spacing")
    group.add_argument("--inner_core_dynamic_confidence", type=float, default=0.80,
                       help="§8: tau_core-dyn")
    group.add_argument("--inner_core_part_confidence", type=float, default=0.70,
                       help="§8: tau_core-part")
    group.add_argument("--inner_core_confidence_floor", type=float, default=0.60,
                       help="§10: the relaxation schedule never goes below this")
    group.add_argument("--inner_core_min_points", type=int, default=None,
                       help="§10: default max(24, 6K)")
    group.add_argument("--inner_cluster_part_gamma", type=float, default=0.5,
                       help="§11: exponent of the WEAK outer-part compatibility term")
    group.add_argument("--inner_attach_min_confidence", type=float, default=0.55)
    group.add_argument("--inner_attach_margin_ratio", type=float, default=1.5)
    group.add_argument("--inner_attach_affinity_quantile", type=float, default=0.25,
                       help="§16: the adaptive affinity cut, as a quantile of the core's "
                            "own internal affinity")
    group.add_argument("--inner_fit_part_gamma", type=float, default=1.0,
                       help="§22: exponent of P_ik in the fitting weight")
    # ---- CODEX_MERGED deterministic loss routing (Parts X-XIII) -------------
    group.add_argument("--lambda_geometry_render", type=float, default=1.0,
                       help="§38: render supervises joint geometry (mandatory)")
    group.add_argument("--lambda_geometry_depth", type=float, default=1.0,
                       help="§39")
    group.add_argument("--lambda_motion_direction", type=float, default=1.0,
                       help="§40: speed-invariant angular loss, geometry block only")
    group.add_argument("--lambda_geometry_final_pcd", type=float, default=1.0,
                       help="§41: endpoint geometry supervises the joint (mandatory)")
    group.add_argument("--lambda_scalar_render", type=float, default=1.0,
                       help="§49")
    group.add_argument("--lambda_scalar_depth", type=float, default=1.0,
                       help="§50")
    group.add_argument("--lambda_motion_magnitude", type=float, default=1.0,
                       help="§51: scalar block only")
    group.add_argument("--lambda_scalar_final_pcd", type=float, default=1.0,
                       help="§52")
    group.add_argument("--inner_render_frame_batch", type=int, default=4,
                       help="§58: frames per step, drawn from a shuffled permutation so "
                            "every frame is visited before the epoch completes")
    group.add_argument("--inner_scalar_lr", type=float, default=1.0e-2,
                       help="relative to the current |s(t)|, since Adam's step is ~lr "
                            "regardless of gradient magnitude")
    # OFF by default so L_scalar is exactly §30's plain weighted sum. Enable it
    # only to explore the weighting: the raw terms are incommensurate (measured
    # on blade, L_video ~ 1.1e-3 against L_velocity ~ 5.3e-7), so at equal
    # lambdas the render term dominates the trajectory by ~1e3.
    group.add_argument("--inner_stage_b_normalize", action="store_true", default=False,
                       help="divide each Stage-B term by its value at the start of the scalar "
                            "phase, so the lambdas express relative priority instead of raw units")
    group.add_argument("--inner_final_pcd_forward_trim", type=float, default=0.8)
    group.add_argument("--inner_final_pcd_backward_trim", type=float, default=0.8)
    group.add_argument("--inner_final_pcd_budget", type=float, default=1.5,
                       help="§25: final points considered, as a multiple of the part size")
    group.add_argument("--temporal_score_render_weight", type=float, default=1.0)
    group.add_argument("--temporal_score_match_weight", type=float, default=0.05,
                       help="§22: the score is render-dominant; 0.0 = render only")
    group.add_argument("--init_soft_cluster_iters", type=int, default=20)
    group.add_argument("--init_soft_cluster_temperature", type=float, default=0.0,
                       help="tau_K; 0 = derive from the cloud scale")
    group.add_argument("--outer_render_temperature", type=float, default=1.0,
                       help="§21: R = softmax(-E / tau_R)")
    group.add_argument("--outer_probability_smoothing", type=float, default=0.0,
                       help="convex step size on the persistent soft state; 0 = none")
    group.add_argument("--init_only", action="store_true", default=False,
                       help="spec §28: stop after initialization, before the first "
                            "inner Killing solve, and write the initialization artifacts")
    # ---- Temporal credit outer cycle (spec §41) -----------------------------
    group.add_argument("--enable_temporal_credit", action="store_true", default=False,
                       help="spec §42 arm B: judge a label change by whether the NEXT "
                            "Killing solve improved the reconstruction")
    group.add_argument("--lambda_score_render", type=float, default=1.0)
    group.add_argument("--lambda_score_match", type=float, default=1.0)
    group.add_argument("--lambda_score_final_nn", type=float, default=0.1)
    group.add_argument("--temporal_credit_graph_k", type=int, default=16)
    group.add_argument("--temporal_global_reward_weight", type=float, default=0.25)
    group.add_argument("--temporal_component_reward_weight", type=float, default=0.0)
    group.add_argument("--temporal_reward_clip", type=float, default=1.0)
    group.add_argument("--temporal_memory_decay", type=float, default=0.8)
    group.add_argument("--temporal_memory_global_decay", type=float, default=1.0)
    group.add_argument("--lambda_temporal_memory", type=float, default=0.25)
    group.add_argument("--temporal_memory_margin_cap", type=float, default=0.5)
    group.add_argument("--temporal_score_frame_stride", type=int, default=1,
                       help="spec §30 requires ONE fixed protocol for the whole run; this "
                            "is a run-level constant, never changed between outer iterations")
    group.add_argument("--temporal_score_attribute", action="store_true", default=True,
                       help="spec §13 per-Gaussian render attribution (needs one extra "
                            "contributor pass per camera)")
    group.add_argument("--no_temporal_score_attribute", dest="temporal_score_attribute",
                       action="store_false")
    group.add_argument("--part_logit_init_std", type=float, default=1.0)
    group.add_argument("--static_affinity_gain", type=float, default=4.0)
    group.add_argument("--static_affinity_iterations", type=int, default=32)
    group.add_argument("--static_affinity_alpha", type=float, default=0.85)
    group.add_argument("--anchor_hard_freeze_outer_iters", type=int, default=1)
    group.add_argument("--part_temperature_soft_floor", type=float, default=0.7)
    group.add_argument("--part_temperature_hard_floor", type=float, default=0.4)
    group.add_argument("--part_temperature_decay", type=float, default=0.85)
    group.add_argument("--part_temperature_churn_tolerance", type=float, default=0.01)
    group.add_argument("--lambda_outer_part_mass", type=float, default=2.0)
    group.add_argument("--lambda_outer_trust_region", type=float, default=1.0)
    group.add_argument("--outer_geometry_guard_eps", type=float, default=1.0e-6)
    group.add_argument("--outer_geometry_guard_backtracks", type=int, default=4)
    group.add_argument("--inner_accept_strain_threshold", type=float, default=float("inf"))
    group.add_argument("--inner_accept_require_render", action="store_true", default=True)
    group.add_argument("--inner_accept_allow_render_regression", dest="inner_accept_require_render", action="store_false")
    group.add_argument("--inner_accept_render_tolerance", type=float, default=0.0)
    group.add_argument("--inner_accept_require_endpoint", action="store_true", default=True)
    group.add_argument("--inner_accept_allow_endpoint_regression", dest="inner_accept_require_endpoint", action="store_false")
    group.add_argument("--inner_accept_endpoint_tolerance", type=float, default=0.0)
    group.add_argument("--inner_reject_observed_decay", type=float, default=0.5)
    group.add_argument("--lambda_rigid_trajectory", type=float, default=1.0)
    group.add_argument("--lambda_rigid_graph", type=float, default=1.0)
    group.add_argument("--lambda_rigid_graph_repulsive", type=float, default=0.5)
    group.add_argument("--lambda_rigid_prior", type=float, default=0.1)
    group.add_argument("--lambda_rigid_moving_complexity", type=float, default=0.01)
    group.add_argument("--lambda_rigid_endpoint", type=float, default=0.1)
    group.add_argument("--lambda_rigid_entropy", type=float, default=0.001)
    group.add_argument("--lambda_rigid_twist", type=float, default=1.0)
    group.add_argument("--lambda_rigid_pair", type=float, default=1.0)
    group.add_argument("--lambda_rigid_distance_rate", type=float, default=1.0)
    group.add_argument("--lambda_rigid_com", type=float, default=0.1)
    group.add_argument("--lambda_rigid_inertia", type=float, default=0.1)
    group.add_argument("--lambda_rigid_com_rate", type=float, default=0.05)
    group.add_argument("--lambda_rigid_inertia_rate", type=float, default=0.05)
    group.add_argument("--lambda_rigid_inertia_invariants", type=float, default=0.05)
    group.add_argument("--lambda_raw_render_rgb", type=float, default=1.0)
    group.add_argument("--lambda_raw_render_depth", type=float, default=0.1)
    group.add_argument("--lambda_raw_observed_velocity", type=float, default=1.0)
    group.add_argument("--lambda_raw_killing", type=float, default=0.1)
    group.add_argument("--lambda_raw_temporal", type=float, default=0.01)
    group.add_argument("--lambda_raw_rigid_feedback", type=float, default=0.1)
    group.add_argument("--lambda_rigid_occupancy", type=float, default=0.01)
    group.add_argument("--final_3dgs_path", type=str, default="")
    group.add_argument("--lambda_endpoint_geometry", type=float, default=0.1)
    group.add_argument("--lambda_endpoint_part_com", type=float, default=0.05)
    group.add_argument("--lambda_endpoint_part_inertia", type=float, default=0.05)
    group.add_argument("--lambda_endpoint_global_com", type=float, default=0.05)
    group.add_argument("--lambda_endpoint_global_inertia", type=float, default=0.05)
    group.add_argument("--kpart_dino", action=BooleanOptionalAction, default=False)
    group.add_argument("--no_kpart_dino", dest="kpart_dino", action="store_false")
    group.add_argument("--kpart_dino_strength", type=float, default=0.0)
    group.add_argument("--kpart_dino_sigma", type=float, default=0.15)
    group.add_argument("--kpart_dino_knn", type=int, default=8)
    group.add_argument("--kpart_graph_segmentation", action=BooleanOptionalAction, default=True)
    group.add_argument("--kpart_velocity_field_segmentation", action=BooleanOptionalAction, default=True)
    group.add_argument(
        "--kpart_initial_oversegment",
        type=int,
        nargs="?",
        const=0,
        default=0,
        help="Enable data-driven K'>K initialization; optionally provide K' explicitly.",
    )
    group.add_argument("--kpart_initial_supernode_knn", type=int, default=8)
    group.add_argument("--kpart_initial_dino_affinity", type=float, default=0.0)
    group.add_argument("--kpart_initial_dynamic_bootstrap_weight", type=float, default=1.0)
    group.add_argument(
        "--kpart_dynamic_prior_weight",
        type=float,
        default=1.0,
        help="Fuse an explicit DynamicPrior into Stage-1 K-part static/dynamic bootstrap.",
    )
    group.add_argument(
        "--kpart_prior_boundary_expansion",
        type=int,
        default=0,
        help="Optional KNN expansion passes around prior-dynamic anchors before K-part clustering.",
    )
    # The denoise/flow-matching priors rank Gaussians very well (ROC-AUC 0.89 to
    # 0.998 over the measured runs) but are poorly calibrated, so a fixed 0.5
    # split throws away F1 the ranking already earned.  "otsu" picks the split
    # from each scene's own score histogram; "fixed" keeps the old behaviour.
    group.add_argument(
        "--kpart_prior_threshold_mode", choices=("otsu", "fixed"), default="otsu"
    )
    group.add_argument(
        "--kpart_prior_anchor_threshold",
        type=float,
        default=0.75,
        help="Prior probability threshold for frozen static/dynamic K-part anchors.",
    )
    group.add_argument(
        "--kpart_prior_cleanup_min_size",
        type=int,
        default=16,
        help="Minimum graph-component size kept in prior-constrained K-part active masks.",
    )
    group.add_argument("--kpart_merge_shared_field_weight", type=float, default=1.0)
    group.add_argument("--kpart_merge_render_weight", type=float, default=0.25)
    group.add_argument("--kpart_merge_dino_weight", type=float, default=0.0)
    group.add_argument("--kpart_soft_temperature", type=float, default=0.2)
    group.add_argument("--kpart_field_fit_responsibility_power", type=float, default=2.0)
    group.add_argument("--kpart_dino_3d_weight", type=float, default=0.0)
    group.add_argument("--kpart_dino_2d_seam_weight", type=float, default=0.0)
    group.add_argument("--kpart_sam3", action=BooleanOptionalAction, default=False)
    group.add_argument("--kpart_sam3_3d_weight", type=float, default=0.0)
    group.add_argument("--kpart_sam3_2d_seam_weight", type=float, default=0.0)
    group.add_argument("--kpart_sam3_similarity_sigma", type=float, default=0.15)
    group.add_argument("--kpart_soft_logit_steps", type=int, default=4)
    group.add_argument("--kpart_soft_margin_threshold", type=float, default=0.15)
    group.add_argument("--kpart_soft_convergence_tolerance", type=float, default=0.005)
    # Eulerian support converged in 2 to 5 of 13 cycles on the measured runs and
    # then spent the rest below 1e-8, while Lagrangian support was still moving
    # at cycle 14.  Convergence now ends the loop on its own: the temperature
    # jumps to its final value the first time it fires, and the loop exits after
    # this many further converged cycles at that temperature.
    group.add_argument("--kpart_convergence_patience", type=int, default=2)
    # Observed-velocity SNR on the discovered moving set: the mean speed of
    # moving Gaussians over the mean speed of static ones.  It is always
    # measured and reported, because it is the quantity every later stage
    # inherits.  It is not gated by default: any floor would be a number picked
    # out of the air, so set one explicitly when you want the run to complain
    # or, with --kpart_require_velocity_snr, to stop.
    group.add_argument("--kpart_min_velocity_snr", type=float, default=0.0)
    group.add_argument("--kpart_require_velocity_snr", action="store_true")
    group.add_argument("--kpart_render_field_invalidation_threshold", type=float, default=0.25)
    group.add_argument("--kpart_speed_signature", action=BooleanOptionalAction, default=True)
    group.add_argument("--kpart_speed_signature_weight", type=float, default=1.0)
    group.add_argument("--kpart_speed_huber_delta", type=float, default=1.0)
    group.add_argument("--kpart_dino_local_boundary", action=BooleanOptionalAction, default=False)
    group.add_argument("--kpart_dino_boundary_gamma", type=float, default=1.0)
    group.add_argument("--kpart_graph_knn", type=int, default=8)
    group.add_argument("--kpart_boundary_weight", type=float, default=0.5)
    group.add_argument("--kpart_counterfactual_render", action=BooleanOptionalAction, default=True)
    group.add_argument("--kpart_render_weight", type=float, default=1.0)
    group.add_argument("--kpart_render_rgb_weight", type=float, default=1.0)
    group.add_argument("--kpart_render_depth_weight", type=float, default=0.1)
    group.add_argument("--kpart_render_dino_weight", type=float, default=0.0)
    group.add_argument("--kpart_render_max_components_per_cycle", "--kpart_counterfactual_max_components", type=int, default=2)
    group.add_argument("--kpart_render_frame_pairs_per_component", "--kpart_counterfactual_pairs_per_component", type=int, default=2)
    group.add_argument("--kpart_render_min_projected_displacement", type=float, default=0.5)
    group.add_argument("--kpart_render_max_projected_displacement", type=float, default=64.0)
    group.add_argument("--kpart_final_completion_passes", type=int, default=1)
    group.add_argument("--use_endpoint_static_model", action="store_true")
    group.add_argument("--endpoint_frame", type=int, default=-1)
    group.add_argument("--endpoint_model_path", type=str, default="")
    group.add_argument("--endpoint_static_iters", type=int, default=1000)
    group.add_argument("--endpoint_init", choices=("depth_fusion", "colmap", "canonical"), default="depth_fusion")
    group.add_argument("--endpoint_depth_voxel_size", type=float, default=0.01)
    group.add_argument("--endpoint_build_if_missing", action="store_true")
    group.add_argument("--kpart_endpoint_transport", action="store_true")
    group.add_argument("--endpoint_transport_mode", choices=("local_soft", "unbalanced_ot"), default="local_soft")
    group.add_argument("--endpoint_knn", type=int, default=16)
    group.add_argument("--endpoint_radius", type=float, default=0.05)
    group.add_argument("--endpoint_transport_temperature", type=float, default=0.02)
    group.add_argument("--endpoint_geometry_weight", type=float, default=1.0)
    group.add_argument("--endpoint_normal_weight", type=float, default=0.0)
    group.add_argument("--endpoint_sam3_weight", type=float, default=0.0)
    group.add_argument("--endpoint_unary_weight", type=float, default=0.0)
    group.add_argument("--endpoint_velocity_weight", type=float, default=0.0)
    group.add_argument("--endpoint_confidence_min", type=float, default=0.05)
    group.add_argument("--endpoint_robust_delta", type=float, default=0.02)
    group.add_argument("--endpoint_geometry_metric", choices=("point", "point_plane"), default="point")
    group.add_argument("--fit_joints", action="store_true")
    group.add_argument("--joint_types", choices=("auto", "revolute", "prismatic", "screw"), default="auto")
    group.add_argument("--joint_topology", choices=("auto", "known"), default="known")
    group.add_argument("--joint_compare_iters", type=int, default=12)
    group.add_argument("--joint_compare_batch_size", type=int, default=4)
    group.add_argument("--joint_compare_validation_cameras", type=int, default=2)
    group.add_argument("--joint_compare_geometry_lr", type=float, default=1.0e-3)
    group.add_argument("--joint_compare_trajectory_lr", type=float, default=1.0e-2)
    group.add_argument("--joint_compare_screw_margin", type=float, default=0.01)
    # Stage 2 is a labeller: axis/pivot/pitch arrive frozen from the Stage-1
    # rigid field and only delta-q moves while the candidates are compared.
    # Re-fitting the geometry here moved the axis by a median 0.08 degrees over
    # 18 measured runs while costing 30% of the compare budget.
    group.add_argument("--joint_compare_refit_geometry", action=BooleanOptionalAction, default=False)
    # Revolute versus prismatic is decided by fitting both hypotheses and
    # comparing how well each explains the data -- the translation-only field
    # against the full rigid field on the same observations, plus the held-out
    # render loss of the two joint candidates.  There is no threshold to set.

    # Frame subset for Stage 2 and Stage 3.  "all" keeps every complete frame
    # group, "endpoints" keeps only the first and last, "stride:N" keeps every
    # Nth frame plus the last, and an explicit comma list keeps those frame ids.
    group.add_argument("--joint_frames", type=str, default="all")
    group.add_argument("--optimize_joint_trajectory", action="store_true")
    group.add_argument(
        "--joint_trajectory_mode",
        choices=("monotonic", "profile_scale", "per_frame", "spline"),
        default="monotonic",
    )
    # Phase 1 supplies a coarse physical prior, not reliable signed metric
    # speeds.  The default Stage-3 solve therefore optimizes monotonic joint
    # displacement directly; profile_scale is kept for ablations.
    group.add_argument("--joint_trajectory_monotonic", action=BooleanOptionalAction, default=True)
    group.add_argument("--joint_trajectory_acceleration_weight", type=float, default=1.0e-5)
    group.add_argument("--joint_trajectory_jerk_weight", type=float, default=1.0e-8)
    group.add_argument("--joint_trajectory_reverse_weight", type=float, default=1.0e-2)
    group.add_argument("--joint_trajectory_release_per_frame", action=BooleanOptionalAction, default=False)
    group.add_argument("--joint_trajectory_iters", type=int, default=200)
    group.add_argument("--joint_trajectory_batch_size", type=int, default=8)
    group.add_argument("--joint_geometry_refine", action="store_true")
    group.add_argument("--eulerian_split_mixed_supports", action=BooleanOptionalAction, default=True)
    group.add_argument("--eulerian_dino_partition_mixed_supports", action=BooleanOptionalAction, default=True)
    group.add_argument("--eulerian_min_support_gaussians", type=int, default=4)
    group.add_argument("--rigid_motion_support", choices=("eulerian", "lagrangian"), default=None)
    group.add_argument("--eulerian_contrastive_dino_split", action=BooleanOptionalAction, default=True)
    group.add_argument("--eulerian_contrastive_dino_cosine_threshold", type=float, default=0.85)
    group.add_argument("--eulerian_contrastive_dino_dispersion_threshold", type=float, default=0.08)
    group.add_argument("--eulerian_contrastive_min_gaussians", type=int, default=4)
    group.add_argument("--eulerian_contrastive_responsibility_split", action=BooleanOptionalAction, default=True)
    group.add_argument("--eulerian_contrastive_static_threshold", type=float, default=0.20)
    group.add_argument("--eulerian_contrastive_rigid_threshold", type=float, default=0.80)
    group.add_argument("--eulerian_contrastive_min_bimodal_fraction", type=float, default=0.10)
    group.add_argument(
        "--rigid_reference_policy",
        choices=("max_confidence_cell",),
        default="max_confidence_cell",
    )
    group.add_argument("--rigid_use_dino_assignment_prior", action="store_true")
    group.add_argument(
        "--sequence_pretrain_assignment",
        action=BooleanOptionalAction,
        default=None,
        help="Infer persistent material cores from whole-sequence depth/flow before rigid fitting.",
    )
    group.add_argument(
        "--phase1_depth_sequence_pretrain",
        action=BooleanOptionalAction,
        default=None,
        help="Build initial Gaussian cores only from measured whole-sequence depth.",
    )
    group.add_argument("--phase1_depth_pretrain_only", action="store_true")
    group.add_argument(
        "--phase1_freeze_depth_cores",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly accept Stage-A depth cores and fit rigid motion without "
            "changing their labels."
        ),
    )
    group.add_argument("--sequence_depth_present_abs", type=float, default=0.005)
    group.add_argument("--sequence_depth_present_rel", type=float, default=0.01)
    group.add_argument("--sequence_depth_occlusion_abs", type=float, default=0.01)
    group.add_argument("--sequence_depth_vacated_abs", type=float, default=0.01)
    group.add_argument("--sequence_depth_min_observable_frames", type=int, default=8)
    group.add_argument("--sequence_vacated_min_run", type=int, default=3)
    group.add_argument("--sequence_pixel_static_present_ratio", type=float, default=0.90)
    group.add_argument("--sequence_pixel_dynamic_vacated_ratio", type=float, default=0.30)
    group.add_argument("--sequence_gaussian_min_camera_weight", type=float, default=0.05)
    group.add_argument("--sequence_gaussian_static_camera_threshold", type=float, default=0.75)
    group.add_argument("--sequence_gaussian_dynamic_camera_threshold", type=float, default=0.75)
    group.add_argument("--sequence_gaussian_min_static_cameras", type=int, default=2)
    group.add_argument("--sequence_gaussian_min_dynamic_cameras", type=int, default=2)
    group.add_argument("--sequence_gaussian_min_static_camera_fraction", type=float, default=0.50)
    group.add_argument("--sequence_gaussian_min_dynamic_camera_fraction", type=float, default=0.50)
    group.add_argument("--sequence_gaussian_max_conflicting_cameras", type=int, default=0)
    group.add_argument("--visibility_hmm", action=BooleanOptionalAction, default=True)
    group.add_argument("--visibility_depth_sigma", type=float, default=0.02)
    group.add_argument("--visibility_present_threshold", type=float, default=0.60)
    group.add_argument("--visibility_occlusion_margin", type=float, default=2.5)
    group.add_argument("--visibility_vacated_margin", type=float, default=2.5)
    group.add_argument("--visibility_min_canonical_alpha", type=float, default=0.05)
    group.add_argument("--static_flow_calibration", action=BooleanOptionalAction, default=True)
    group.add_argument("--static_flow_tile_size", type=int, default=0)
    group.add_argument("--static_flow_min_samples", type=int, default=64)
    group.add_argument("--static_flow_mad_floor", type=float, default=0.15)
    group.add_argument("--initial_dynamic_core_min_score", type=float, default=0.72)
    group.add_argument("--initial_dynamic_core_min_time_fraction", type=float, default=0.60)
    group.add_argument("--initial_dynamic_core_min_camera_fraction", type=float, default=0.40)
    group.add_argument("--initial_static_core_min_score", type=float, default=0.88)
    group.add_argument("--initial_core_min_observations", type=int, default=12)
    group.add_argument("--rigid_assignment_min_time_fraction", type=float, default=0.60)
    group.add_argument("--rigid_assignment_min_camera_fraction", type=float, default=0.40)
    group.add_argument("--rigid_assignment_min_margin", type=float, default=0.35)
    group.add_argument("--rigid_assignment_min_speed_correlation", type=float, default=0.50)
    group.add_argument("--core_promotion_min_cycles", type=int, default=2)
    group.add_argument("--core_promotion_min_margin", type=float, default=0.60)
    group.add_argument("--core_demotion_margin", type=float, default=0.70)
    group.add_argument("--assignment_graphcut", action=BooleanOptionalAction, default=False)
    group.add_argument("--assignment_graph_lambda", type=float, default=0.0)
    group.add_argument(
        "--curl_grid_target_cells",
        type=int,
        default=16384,
        help="Approximate fixed number of regular Eulerian cells; cell counts follow scene aspect ratio.",
    )
    group.add_argument(
        "--curl_grid_padding",
        type=float,
        default=0.15,
        help="World-domain padding as a fraction of the largest canonical foreground extent.",
    )
    group.add_argument("--curl_grid_min_cells_per_axis", type=int, default=2)
    group.add_argument(
        "--curl_temporal_control_knots",
        type=int,
        default=10,
        help="Cubic B-spline control tensors for the Eulerian vector potential A(t).",
    )
    group.add_argument("--curl_grid_grad_clip", type=float, default=1.0)
    group.add_argument(
        "--eulerian_velocity_gradient_min_neighbors",
        type=int,
        default=3,
        help="Minimum occupied material neighbours used to fit the local velocity-gradient tensor G=dv/dx.",
    )
    group.add_argument(
        "--eulerian_velocity_gradient_ridge",
        type=float,
        default=1.0e-4,
        help="Relative Tikhonov ridge for the local material velocity-gradient least-squares solve.",
    )
    group.add_argument(
        "--eulerian_velocity_scale",
        type=float,
        default=1.0,
        help="Physical velocity normalization for observed-flow and velocity-gradient rigidity losses.",
    )
    group.add_argument(
        "--lambda_eulerian_strain_rate",
        type=float,
        default=100.0,
        help="Strong rigidity weight on ||D||_F^2 where D=sym(grad v); D=0 is exact local non-deformation.",
    )
    group.add_argument(
        "--lambda_eulerian_velocity_spatial_coherence",
        type=float,
        default=10.0,
        help="Direct robust same-moving-part vector-velocity coherence ||v_i-v_j||. This damps random local arrows before differentiating v.",
    )
    group.add_argument(
        "--lambda_eulerian_speed_spatial_coherence",
        type=float,
        default=2.0,
        help="Direct robust same-moving-part speed-magnitude coherence | ||v_i||-||v_j|| |.",
    )
    group.add_argument(
        "--velocity_spatial_coherence_robust_delta",
        type=float,
        default=0.25,
        help="Charbonnier transition scale for direct velocity/speed spatial coherence, normalized by eulerian_velocity_scale.",
    )
    group.add_argument(
        "--lambda_eulerian_spatial_gradient_coherence",
        type=float,
        default=50.0,
        help="Mandatory axis-free same-moving-part spatial smoothness: neighbouring local velocity gradients G=dv/dx must agree.",
    )
    group.add_argument(
        "--lambda_eulerian_local_affine_coherence",
        type=float,
        default=50.0,
        help="Mandatory axis-free local velocity consistency: dv must match the neighbour-averaged generic gradient times dx; preserves arbitrary affine/rigid motion without fitting an axis.",
    )
    group.add_argument(
        "--lambda_eulerian_static_speed_strain",
        type=float,
        default=5.0,
        help="Secondary soft zero-speed prior; the v8 hard straight-through static gate is the primary static constraint.",
    )
    group.add_argument(
        "--lambda_eulerian_assignment_kinematic",
        type=float,
        default=2.0,
        help="Assignment-only loss from observed nonzero cell speed and local rigid-kinematic discontinuities.",
    )
    group.add_argument(
        "--curl_material_neighbors",
        type=int,
        default=12,
        help="Nearest occupied material control volumes used for the local velocity-gradient tensor fit.",
    )
    group.add_argument(
        "--curl_material_neighbor_radius_cells",
        type=float,
        default=3.5,
        help="Maximum occupied-cell neighbour distance in units of the largest cell side; <=0 disables radius filtering.",
    )
    group.add_argument(
        "--curl_static_cell_purity",
        type=float,
        default=0.90,
        help="Minimum fraction of static Gaussian centres required before an occupied cell receives the zero-speed static prior.",
    )
    group.add_argument("--lagrangian_knn_chunk_size", type=int, default=512)
    group.add_argument(
        "--phase1_window_frames",
        type=int,
        default=6,
        help="Number of synchronized physical frames jointly contributing before one optimizer step.",
    )
    group.add_argument(
        "--phase1_window_stride",
        type=int,
        default=5,
        help="Requested stride; clamped to window width so every target frame is supervised.",
    )
    group.add_argument("--assignment_lr", type=float, default=1.0e-2)
    group.add_argument(
        "--assignment_initial_dynamic_probability",
        type=float,
        default=0.95,
        help="Initial moving probability for gradient-attribution dynamic anchors.",
    )
    group.add_argument(
        "--assignment_initial_static_dynamic_probability",
        type=float,
        default=0.10,
        help="Initial moving probability for nominal static points; deliberately nonzero so occluded false statics can flip.",
    )
    group.add_argument("--assignment_temperature_start", type=float, default=1.0)
    group.add_argument("--assignment_temperature_end", type=float, default=0.25)
    group.add_argument(
        "--assignment_entropy_warmup_fraction",
        type=float,
        default=0.30,
        help="Do not strongly harden assignments before this fraction of window updates.",
    )
    group.add_argument("--lambda_assignment_prior", type=float, default=0.05)
    group.add_argument(
        "--assignment_prior_final_scale",
        type=float,
        default=0.25,
        help="Anneal the initialization prior so rendering/rigid context can overturn wrong labels late in training.",
    )
    group.add_argument(
        "--assignment_prior_dynamic_weight",
        type=float,
        default=1.0,
        help="Trust initial dynamic anchors more strongly because false dynamics are rarer.",
    )
    group.add_argument(
        "--assignment_prior_static_weight",
        type=float,
        default=0.25,
        help="Weak static prior: occlusion-induced false statics must be easy to overturn.",
    )
    group.add_argument(
        "--lambda_assignment_spatial",
        type=float,
        default=0.03,
        help="Canonical material-neighbour coherence of soft part probabilities.",
    )
    group.add_argument(
        "--lambda_assignment_entropy",
        type=float,
        default=0.01,
        help="Late-stage entropy penalty encouraging an eventual discrete material partition.",
    )
    group.add_argument(
        "--assignment_same_part_floor",
        type=float,
        default=0.25,
        help="Minimum Eulerian neighbour rigidity coupling across a current assignment boundary; prevents false-static cuts from deleting mechanics.",
    )
    group.add_argument(
        "--phase1_assignment_passes",
        type=int,
        default=1,
        help="Assignment-only passes with the current motion frozen.",
    )
    group.add_argument(
        "--phase1_motion_passes",
        type=int,
        default=5,
        help="Passes that jointly update the curl field and material assignments.",
    )
    group.add_argument(
        "--assignment_evidence_temperature",
        type=float,
        default=0.08,
        help="Soft calibration width mapping continuous motion evidence to initial moving probability.",
    )
    group.add_argument(
        "--assignment_same_part_floor_start",
        type=float,
        default=0.50,
        help="Early rigidity coupling floor so false-static boundaries cannot cut the material graph.",
    )
    group.add_argument(
        "--assignment_same_part_floor_end",
        type=float,
        default=0.05,
        help="Late rigidity coupling floor allowing true static/dynamic boundaries to sharpen.",
    )
    group.add_argument(
        "--lambda_curl_temporal_acceleration",
        type=float,
        default=0.10,
        help="Second-difference penalty on temporal CurlGrid B-spline controls.",
    )
    group.add_argument(
        "--eulerian_projection_steps",
        type=int,
        default=3,
        help="Pure rigid-projection steps on A after each data update.",
    )
    group.add_argument("--eulerian_projection_lr", type=float, default=5.0e-4)
    group.add_argument("--motion_lr", type=float, default=1.0e-3)
    group.add_argument("--phase1_data_gradient_scale", type=float, default=1.0)
    group.add_argument("--phase1_pair_neighbors", type=int, default=16)
    group.add_argument("--phase1_pair_max_pairs", type=int, default=262144)
    group.add_argument(
        "--motion_evidence_future_frames",
        type=int,
        default=0,
        help="Future frames per camera for attribution; 0 means all.",
    )
    group.add_argument("--motion_evidence_max_cameras", type=int, default=0)
    group.add_argument("--motion_evidence_min_dynamic_points", type=int, default=32)
    group.add_argument("--motion_evidence_html_max_points", type=int, default=50000)
    group.add_argument(
        "--static_evidence_refine_steps",
        type=int,
        default=24,
        help="Cheap per-Gaussian optimization steps for the complementary frame0/future static-consistency score.",
    )
    group.add_argument("--static_evidence_refine_lr", type=float, default=0.20)
    group.add_argument(
        "--static_evidence_refine_prior_weight",
        type=float,
        default=0.05,
        help="Retention prior in the static-consistency minimization; lower values demand tighter frame0/future agreement.",
    )
    group.add_argument(
        "--transported_html_max_points",
        type=int,
        default=1000,
        help="Deterministic Gaussian sample size for each transported-motion HTML.",
    )
    group.add_argument("--render_after_training", action="store_true")
    group.add_argument(
        "--phase1_only",
        action="store_true",
        help="Legacy dynamic-pipeline guard; staged K-part sub-stages are controlled by their own flags.",
    )
    wandb_toggle.add_argument("--wandb", dest="wandb", action="store_true")
    wandb_toggle.add_argument("--no_wandb", dest="wandb", action="store_false")
    group.add_argument("--wandb_project", type=str, default="RigEuler3DGS")
    group.add_argument("--wandb_entity", type=str, default="")
    group.add_argument("--wandb_name", type=str, default="")
    group.add_argument("--wandb_tags", type=str, default="")
    group.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default="online"
    )
    group.add_argument("--wandb_log_interval", type=int, default=10)

    group.set_defaults(use_motion_correspondence=True, wandb=True)


def apply_dynamic_defaults(args):
    """Validate and normalize the single supported dynamic pipeline."""
    args.static_train_iters = max(0, int(args.static_train_iters))
    args.free_field_iters = max(1, int(args.free_field_iters))
    args.num_moving_parts = max(1, int(args.num_moving_parts))
    args.fps = max(1.0e-6, float(args.fps))
    args.time_scale = float(args.time_scale)
    if args.time_scale != 1.0 or args.motion_time_mode != "seconds":
        raise ValueError(
            "Dynamic motion uses physical seconds: --time_scale 1 --motion_time_mode seconds"
        )
    args.dynamic_frame_start = max(0, int(args.dynamic_frame_start))
    args.dynamic_frame_stride = max(1, int(args.dynamic_frame_stride))
    args.dynamic_test_stride = max(0, int(args.dynamic_test_stride))
    args.rigid_analytic_moving_part = max(
        0, int(args.rigid_analytic_moving_part)
    )
    responsibility_training = bool(args.rigid_responsibility_training)
    neutral_discovery = bool(args.phase1_neutral_rigid_discovery)
    kpart_discovery = bool(args.kpart_rigid_discovery)
    nested_killing_discovery = bool(getattr(args, "enable_nested_killing_part_discovery", False))
    render_denoise_discovery = str(args.dynamic_discovery_mode) == "render_denoise"
    dynamic_prior_supplied = bool(str(getattr(args, "dynamic_prior_path", "") or "").strip())
    feature_boundary_available = (
        bool(args.use_dino_feature_field)
        or bool(args.use_sam3_feature_field)
        or render_denoise_discovery
    )
    if render_denoise_discovery:
        if not kpart_discovery:
            raise ValueError("--dynamic_discovery_mode render_denoise requires --kpart_rigid_discovery")
        args.phase1_depth_sequence_pretrain = False
        args.phase1_depth_pretrain_only = False
        args.phase1_freeze_depth_cores = False
        args.sequence_pretrain_assignment = False
        args.kpart_dino = False
        args.kpart_dino_3d_weight = 0.0
        args.kpart_dino_2d_seam_weight = 0.0
        args.kpart_dino_local_boundary = False
        args.kpart_initial_dino_affinity = 0.0
        args.kpart_merge_dino_weight = 0.0
    if dynamic_prior_supplied and kpart_discovery:
        args.kpart_dino = False
        args.kpart_dino_3d_weight = 0.0
        args.kpart_dino_2d_seam_weight = 0.0
        args.kpart_dino_local_boundary = False
        args.kpart_initial_dino_affinity = 0.0
        args.kpart_merge_dino_weight = 0.0
        args.kpart_render_dino_weight = 0.0
    if nested_killing_discovery:
        if args.phase1_motion_representation not in RIGID_KILLING_REPRESENTATIONS:
            raise ValueError("--enable_nested_killing_part_discovery requires a rigid Killing representation")
        if dynamic_prior_supplied:
            raise ValueError("--enable_nested_killing_part_discovery forbids --dynamic_prior_path")
        if kpart_discovery:
            raise ValueError("--enable_nested_killing_part_discovery replaces legacy --kpart_rigid_discovery")
        if bool(args.enable_rigid_part_discovery):
            raise ValueError("--enable_nested_killing_part_discovery replaces --enable_rigid_part_discovery")
        if responsibility_training or neutral_discovery:
            raise ValueError("--enable_nested_killing_part_discovery cannot be combined with single-rigid discovery controllers")
        if (
            args.phase1_depth_sequence_pretrain is True
            or args.phase1_depth_pretrain_only
            or args.phase1_freeze_depth_cores
            or args.sequence_pretrain_assignment is True
        ):
            raise ValueError("Nested Killing discovery starts from neutral foreground labels, not depth/sequence labels")
        if not str(getattr(args, "final_3dgs_path", "") or "").strip():
            raise ValueError("--enable_nested_killing_part_discovery requires --final_3dgs_path")
        if args.fit_joints or args.optimize_joint_trajectory or args.joint_geometry_refine:
            raise ValueError("Nested Killing discovery currently implements Stage 1 only; do not pass joint flags")
        args.phase1_depth_sequence_pretrain = False
        args.sequence_pretrain_assignment = False
        args.kpart_dino = False
        args.kpart_dino_3d_weight = 0.0
        args.kpart_dino_2d_seam_weight = 0.0
        args.kpart_dino_local_boundary = False
        args.kpart_initial_dino_affinity = 0.0
        args.kpart_merge_dino_weight = 0.0
        args.kpart_render_dino_weight = 0.0
    # The comparison pipeline is prior-first: diffusion/FM probabilities define
    # static/dynamic anchors, then Stage 1 clusters only the allowed moving
    # domain.  DINO stays opt-in and defaults to zero for K-part clustering.
    comparison_pipeline = bool(getattr(args, "run_dynamic_prior_comparison", False))
    if (
        args.fit_joints
        or args.optimize_joint_trajectory
        or args.joint_geometry_refine
    ) and not (kpart_discovery or comparison_pipeline):
        raise ValueError("Joint fitting/trajectory flags require --kpart_rigid_discovery")
    if bool(args.enable_rigid_part_discovery) and not kpart_discovery:
        raise ValueError("--enable_rigid_part_discovery requires --kpart_rigid_discovery")
    if kpart_discovery:
        if args.phase1_motion_representation not in RIGID_KILLING_REPRESENTATIONS:
            raise ValueError("--kpart_rigid_discovery requires a rigid Killing representation")
        if responsibility_training or neutral_discovery:
            raise ValueError("K-part discovery replaces the single-rigid responsibility/discovery controller")
        if (
            args.phase1_depth_sequence_pretrain is True
            or args.phase1_depth_pretrain_only
            or args.phase1_freeze_depth_cores
            or args.sequence_pretrain_assignment is True
        ):
            raise ValueError("K-part discovery must start from uniform Gaussian responsibilities")
        inferred_support = (
            "lagrangian"
            if args.phase1_motion_representation == "rigid_killing_lagrangian"
            else "eulerian"
        )
        if args.kpart_support is not None and args.kpart_support != inferred_support:
            raise ValueError("--kpart_support must match --phase1_motion_representation")
        args.kpart_support = inferred_support
        args.phase1_depth_sequence_pretrain = False
        args.sequence_pretrain_assignment = False
        if args.optimize_joint_trajectory:
            args.fit_joints = True
        if args.fit_joints and not (feature_boundary_available or dynamic_prior_supplied):
            raise ValueError(
                "Rendered Stage-2 joint competition requires DINO, SAM3, render-denoise evidence, or --dynamic_prior_path"
            )
        if (
            args.fit_joints
            and bool(args.use_dino_feature_field)
            and not bool(args.use_sam3_feature_field)
            and float(args.lambda_dino_feature_field_dynamic) <= 0.0
            and not render_denoise_discovery
        ):
            raise ValueError("Rendered Stage-2 joint competition requires a positive dynamic DINO loss weight")
        if (
            args.kpart_velocity_field_segmentation
            and not (
                args.kpart_sam3
                or render_denoise_discovery
                or args.kpart_endpoint_transport
                or dynamic_prior_supplied
            )
        ):
            raise ValueError("--kpart_velocity_field_segmentation requires SAM3, render-denoise, endpoint transport, or --dynamic_prior_path")
    elif responsibility_training:
        if args.phase1_motion_representation not in RIGID_KILLING_REPRESENTATIONS:
            raise ValueError(
                "--rigid_responsibility_training requires a rigid Killing representation"
            )
        if (
            neutral_discovery
            or args.phase1_depth_sequence_pretrain is True
            or args.phase1_depth_pretrain_only
            or args.phase1_freeze_depth_cores
            or args.sequence_pretrain_assignment is True
        ):
            raise ValueError(
                "Soft rigid responsibility training cannot use one-shot, depth-core, "
                "or sequence-core assignment"
            )
        inferred_support = (
            "lagrangian"
            if args.phase1_motion_representation == "rigid_killing_lagrangian"
            else "eulerian"
        )
        if (
            args.rigid_motion_support is not None
            and args.rigid_motion_support != inferred_support
        ):
            raise ValueError(
                "--rigid_motion_support must match --phase1_motion_representation"
            )
        args.rigid_motion_support = inferred_support
        args.phase1_depth_sequence_pretrain = False
        args.sequence_pretrain_assignment = False
    elif neutral_discovery:
        if args.phase1_motion_representation not in RIGID_KILLING_REPRESENTATIONS:
            raise ValueError(
                "--phase1_neutral_rigid_discovery requires a rigid Killing representation"
            )
        if (
            args.phase1_depth_sequence_pretrain is True
            or args.phase1_depth_pretrain_only
            or args.phase1_freeze_depth_cores
            or args.sequence_pretrain_assignment is True
        ):
            raise ValueError(
                "Neutral rigid discovery cannot use depth-derived or sequence-pretrained labels"
            )
        args.phase1_depth_sequence_pretrain = False
        args.sequence_pretrain_assignment = False
    else:
        if dynamic_prior_supplied:
            args.phase1_depth_sequence_pretrain = False
            args.sequence_pretrain_assignment = False
        elif args.phase1_depth_sequence_pretrain is None:
            args.phase1_depth_sequence_pretrain = (
                args.phase1_motion_representation in RIGID_KILLING_REPRESENTATIONS
            )
        if args.phase1_depth_pretrain_only:
            args.phase1_depth_sequence_pretrain = True
        if args.phase1_freeze_depth_cores and not args.phase1_depth_sequence_pretrain:
            raise ValueError(
                "--phase1_freeze_depth_cores requires depth sequence pretraining"
            )
        if (
            args.phase1_depth_sequence_pretrain
            and not args.phase1_depth_pretrain_only
            and not args.phase1_freeze_depth_cores
        ):
            raise ValueError(
                "Depth-first Stage C reassignment is disabled; use "
                "--phase1_depth_pretrain_only or --phase1_freeze_depth_cores"
            )
        if args.sequence_pretrain_assignment is None:
            args.sequence_pretrain_assignment = (
                args.phase1_motion_representation in RIGID_KILLING_REPRESENTATIONS
            )
    if args.phase1_depth_sequence_pretrain:
        args.sequence_pretrain_assignment = True
        args.visibility_hmm = False
    for name in (
        "sequence_depth_present_abs",
        "sequence_depth_present_rel",
        "sequence_depth_occlusion_abs",
        "sequence_depth_vacated_abs",
        "sequence_gaussian_min_camera_weight",
    ):
        setattr(args, name, max(0.0, float(getattr(args, name))))
    args.sequence_depth_min_observable_frames = max(
        1, int(args.sequence_depth_min_observable_frames)
    )
    args.sequence_vacated_min_run = max(1, int(args.sequence_vacated_min_run))
    for name in (
        "sequence_pixel_static_present_ratio",
        "sequence_pixel_dynamic_vacated_ratio",
        "sequence_gaussian_static_camera_threshold",
        "sequence_gaussian_dynamic_camera_threshold",
        "sequence_gaussian_min_static_camera_fraction",
        "sequence_gaussian_min_dynamic_camera_fraction",
    ):
        setattr(args, name, min(1.0, max(0.0, float(getattr(args, name)))))
    args.sequence_gaussian_min_static_cameras = max(
        1, int(args.sequence_gaussian_min_static_cameras)
    )
    args.sequence_gaussian_min_dynamic_cameras = max(
        2, int(args.sequence_gaussian_min_dynamic_cameras)
    )
    args.sequence_gaussian_max_conflicting_cameras = max(
        0, int(args.sequence_gaussian_max_conflicting_cameras)
    )
    args.visibility_depth_sigma = max(1.0e-6, float(args.visibility_depth_sigma))
    args.visibility_present_threshold = min(1.0, max(0.0, float(args.visibility_present_threshold)))
    args.visibility_occlusion_margin = max(0.0, float(args.visibility_occlusion_margin))
    args.visibility_vacated_margin = max(0.0, float(args.visibility_vacated_margin))
    args.visibility_min_canonical_alpha = min(1.0, max(0.0, float(args.visibility_min_canonical_alpha)))
    args.static_flow_tile_size = max(0, int(args.static_flow_tile_size))
    args.static_flow_min_samples = max(4, int(args.static_flow_min_samples))
    args.static_flow_mad_floor = max(1.0e-6, float(args.static_flow_mad_floor))
    args.render_denoise_num_probe_frames = max(1, int(args.render_denoise_num_probe_frames))
    args.render_denoise_cameras_per_probe = max(0, int(args.render_denoise_cameras_per_probe))
    args.render_denoise_min_support_cameras = max(1, int(args.render_denoise_min_support_cameras))
    args.render_denoise_min_informative_frames = max(1, int(args.render_denoise_min_informative_frames))
    args.render_denoise_num_seeds = max(1, int(args.render_denoise_num_seeds))
    args.render_denoise_steps = max(1, int(args.render_denoise_steps))
    args.render_denoise_noise_sigma = max(1.0e-6, float(args.render_denoise_noise_sigma))
    args.render_denoise_lr = max(1.0e-8, float(args.render_denoise_lr))
    args.render_denoise_max_displacement = max(1.0e-8, float(args.render_denoise_max_displacement))
    args.render_denoise_support_threshold = min(1.0, max(0.0, float(args.render_denoise_support_threshold)))
    args.render_denoise_seed_consistency_threshold = min(1.0, max(0.0, float(args.render_denoise_seed_consistency_threshold)))
    args.render_denoise_view_consistency_threshold = min(1.0, max(0.0, float(args.render_denoise_view_consistency_threshold)))
    args.render_denoise_dynamic_threshold = min(0.999, max(0.501, float(args.render_denoise_dynamic_threshold)))
    args.render_denoise_static_threshold = min(0.499, max(0.001, float(args.render_denoise_static_threshold)))
    args.render_denoise_confidence_threshold = min(1.0, max(0.0, float(args.render_denoise_confidence_threshold)))
    if args.render_denoise_static_threshold >= args.render_denoise_dynamic_threshold:
        raise ValueError("render denoise static threshold must be below dynamic threshold")
    args.render_gate_num_seeds = max(1, int(args.render_gate_num_seeds))
    args.render_gate_steps = max(1, int(args.render_gate_steps))
    args.render_gate_lr = max(1.0e-8, float(args.render_gate_lr))
    args.render_gate_temperature = max(1.0e-6, float(args.render_gate_temperature))
    args.render_gate_noise_sigma = max(0.0, float(args.render_gate_noise_sigma))
    for name in (
        "initial_dynamic_core_min_score",
        "initial_dynamic_core_min_time_fraction",
        "initial_dynamic_core_min_camera_fraction",
        "initial_static_core_min_score",
        "rigid_assignment_min_time_fraction",
        "rigid_assignment_min_camera_fraction",
        "rigid_assignment_min_speed_correlation",
    ):
        setattr(args, name, min(1.0, max(0.0, float(getattr(args, name)))))
    args.initial_core_min_observations = max(1, int(args.initial_core_min_observations))
    args.rigid_assignment_min_margin = max(0.0, float(args.rigid_assignment_min_margin))
    args.core_promotion_min_cycles = max(1, int(args.core_promotion_min_cycles))
    args.core_promotion_min_margin = max(0.0, float(args.core_promotion_min_margin))
    args.core_demotion_margin = max(args.core_promotion_min_margin, float(args.core_demotion_margin))
    args.assignment_graph_lambda = max(0.0, float(args.assignment_graph_lambda))
    args.dynamic_use_depth_npy = bool(args.dynamic_use_depth_npy)
    args.dynamic_depth_is_metric = bool(args.dynamic_depth_is_metric)

    args.semantic_dynamic_threshold = min(
        1.0, max(0.0, float(args.semantic_dynamic_threshold))
    )
    args.prior_hard_threshold = min(1.0, max(0.0, float(args.prior_hard_threshold)))
    args.prior_use_soft_weights = bool(args.prior_use_soft_weights)
    args.comparison_seed = int(args.comparison_seed)
    args.comparison_denoise_prob_temperature = max(
        1.0e-6, float(args.comparison_denoise_prob_temperature)
    )
    args.comparison_denoise_prob_bias = float(args.comparison_denoise_prob_bias)
    args.comparison_prior_smooth_knn = max(
        1, int(args.comparison_prior_smooth_knn)
    )
    args.comparison_prior_smooth_iters = max(
        0, int(args.comparison_prior_smooth_iters)
    )
    args.comparison_prior_smooth_alpha = min(
        1.0, max(0.0, float(args.comparison_prior_smooth_alpha))
    )
    args.comparison_kpart_k = max(1, int(args.comparison_kpart_k))
    if bool(args.comparison_optimize_joint_trajectory):
        args.comparison_fit_joints = True
    args.fm_endpoint_knn = max(1, int(args.fm_endpoint_knn))
    args.fm_transport_temperature = max(1.0e-8, float(args.fm_transport_temperature))
    args.fm_transport_robust_delta = max(1.0e-8, float(args.fm_transport_robust_delta))
    args.fm_confidence_entropy_scale = max(0.0, float(args.fm_confidence_entropy_scale))
    args.fm_prior_prob_bias = float(args.fm_prior_prob_bias)
    args.fm_prior_prob_temperature = max(1.0e-6, float(args.fm_prior_prob_temperature))
    args.fm_transport_chunk_size = max(32, int(args.fm_transport_chunk_size))
    args.fm_hidden_dim = max(8, int(args.fm_hidden_dim))
    args.fm_layers = max(1, int(args.fm_layers))
    args.fm_lr = max(1.0e-8, float(args.fm_lr))
    args.fm_iters = max(1, int(args.fm_iters))
    args.fm_batch_size = max(1, int(args.fm_batch_size))
    args.fm_identity_weight = max(0.0, float(args.fm_identity_weight))
    args.fm_smooth_weight = max(0.0, float(args.fm_smooth_weight))
    args.fm_integration_steps = max(1, int(args.fm_integration_steps))
    args.semantic_dynamic_min_points = max(1, int(args.semantic_dynamic_min_points))
    args.dino_feature_field_dim = max(1, int(args.dino_feature_field_dim))
    args.sam3_feature_dim = max(1, int(args.sam3_feature_dim))
    args.sam3_cluster_weight = max(0.0, float(args.sam3_cluster_weight))
    args.sam3_affinity_dim = max(1, int(args.sam3_affinity_dim))
    args.sam3_affinity_iters = max(0, int(args.sam3_affinity_iters))
    args.sam3_affinity_lr = max(1.0e-8, float(args.sam3_affinity_lr))
    args.sam3_mask_min_fg_fraction = min(
        1.0, max(0.0, float(args.sam3_mask_min_fg_fraction))
    )
    args.sam3_mask_max_fg_fraction = min(
        1.0, max(args.sam3_mask_min_fg_fraction, float(args.sam3_mask_max_fg_fraction))
    )
    args.sam3_affinity_pixels_per_iter = max(2, int(args.sam3_affinity_pixels_per_iter))
    args.sam3_affinity_positive_pairs = max(0, int(args.sam3_affinity_positive_pairs))
    args.sam3_affinity_intermediate_pairs = max(0, int(args.sam3_affinity_intermediate_pairs))
    args.sam3_affinity_negative_pairs = max(0, int(args.sam3_affinity_negative_pairs))
    args.sam3_affinity_pos_threshold = min(
        1.0, max(0.0, float(args.sam3_affinity_pos_threshold))
    )
    args.sam3_affinity_neg_threshold = min(
        args.sam3_affinity_pos_threshold, max(0.0, float(args.sam3_affinity_neg_threshold))
    )
    args.sam3_affinity_temperature = max(
        1.0e-6, float(args.sam3_affinity_temperature)
    )
    args.sam3_affinity_bias = float(args.sam3_affinity_bias)
    args.sam3_affinity_norm_weight = max(0.0, float(args.sam3_affinity_norm_weight))
    args.sam3_affinity_hard_pair_fraction = min(
        1.0, max(0.0, float(args.sam3_affinity_hard_pair_fraction))
    )
    args.sam3_affinity_debug_interval = max(0, int(args.sam3_affinity_debug_interval))
    args.sam3_affinity_graph_knn = max(1, int(args.sam3_affinity_graph_knn))
    args.sam3_affinity_html_max_points = max(1, int(args.sam3_affinity_html_max_points))
    args.use_dinov3 = bool(args.use_dino_feature_field)
    args.endpoint_static_iters = max(0, int(args.endpoint_static_iters))
    args.endpoint_depth_voxel_size = max(0.0, float(args.endpoint_depth_voxel_size))
    args.endpoint_knn = max(1, int(args.endpoint_knn))
    args.endpoint_radius = max(0.0, float(args.endpoint_radius))
    args.endpoint_transport_temperature = max(1.0e-6, float(args.endpoint_transport_temperature))
    args.endpoint_geometry_weight = max(0.0, float(args.endpoint_geometry_weight))
    args.endpoint_normal_weight = max(0.0, float(args.endpoint_normal_weight))
    args.endpoint_sam3_weight = max(0.0, float(args.endpoint_sam3_weight))
    args.endpoint_unary_weight = max(0.0, float(args.endpoint_unary_weight))
    args.endpoint_velocity_weight = max(0.0, float(args.endpoint_velocity_weight))
    args.endpoint_confidence_min = min(1.0, max(0.0, float(args.endpoint_confidence_min)))
    args.endpoint_robust_delta = max(1.0e-8, float(args.endpoint_robust_delta))
    args.kpart_sam3_3d_weight = max(0.0, float(args.kpart_sam3_3d_weight))
    args.kpart_sam3_2d_seam_weight = max(0.0, float(args.kpart_sam3_2d_seam_weight))
    args.kpart_sam3_similarity_sigma = max(1.0e-8, float(args.kpart_sam3_similarity_sigma))

    args.correspondence_flow_max_side = max(32, int(args.correspondence_flow_max_side))
    args.correspondence_cycle_threshold = max(
        1.0e-3, float(args.correspondence_cycle_threshold)
    )
    args.correspondence_dino_min_cosine = min(
        1.0, max(-1.0, float(args.correspondence_dino_min_cosine))
    )
    args.correspondence_min_confidence = min(
        1.0, max(0.0, float(args.correspondence_min_confidence))
    )
    args.correspondence_alpha_threshold = min(
        1.0, max(0.0, float(args.correspondence_alpha_threshold))
    )
    args.observed_flow_velocity_ridge = max(
        0.0, float(args.observed_flow_velocity_ridge)
    )
    args.observed_flow_velocity_min_observations = max(
        2, int(args.observed_flow_velocity_min_observations)
    )
    args.observed_flow_velocity_min_condition = min(
        1.0, max(0.0, float(args.observed_flow_velocity_min_condition))
    )
    args.observed_flow_velocity_min_cameras = max(
        1, int(args.observed_flow_velocity_min_cameras)
    )
    args.observed_flow_velocity_max_reprojection_px = max(
        1.0e-4, float(args.observed_flow_velocity_max_reprojection_px)
    )
    if neutral_discovery or responsibility_training or kpart_discovery:
        args.observed_flow_require_static_improvement = False
    elif args.observed_flow_require_static_improvement is None:
        args.observed_flow_require_static_improvement = (
            args.phase1_motion_representation in RIGID_KILLING_REPRESENTATIONS
        )
    else:
        args.observed_flow_require_static_improvement = bool(
            args.observed_flow_require_static_improvement
        )
    args.observed_flow_static_improvement_abs = max(
        0.0, float(args.observed_flow_static_improvement_abs)
    )
    args.observed_flow_static_improvement_rel = min(
        1.0, max(0.0, float(args.observed_flow_static_improvement_rel))
    )
    args.observed_flow_target_refresh_iters = max(
        1, int(args.observed_flow_target_refresh_iters)
    )
    args.multiview_flow_noise_px = max(1.0e-4, float(args.multiview_flow_noise_px))
    args.multiview_flow_model_margin_px = max(
        0.0, float(args.multiview_flow_model_margin_px)
    )
    args.multiview_flow_model_temperature_px = max(
        1.0e-4, float(args.multiview_flow_model_temperature_px)
    )

    args.rigid_solver_max_iters = max(1, int(args.rigid_solver_max_iters))
    args.rigid_solver_tol = max(1.0e-12, float(args.rigid_solver_tol))
    args.rigid_solver_irls_iters = max(1, int(args.rigid_solver_irls_iters))
    args.rigid_solver_huber_delta = max(0.0, float(args.rigid_solver_huber_delta))
    args.rigid_min_valid_cells = max(4, int(args.rigid_min_valid_cells))
    args.rigid_min_distinct_cameras = max(1, int(args.rigid_min_distinct_cameras))
    args.rigid_angular_eps = max(1.0e-12, float(args.rigid_angular_eps))
    args.rigid_speed_temporal_weight = max(
        0.0, float(args.rigid_speed_temporal_weight)
    )
    args.rigid_assignment_improvement_margin = max(
        0.0, float(args.rigid_assignment_improvement_margin)
    )
    args.rigid_assignment_temperature = max(
        1.0e-6, float(args.rigid_assignment_temperature)
    )
    args.rigid_assignment_prior_weight = max(
        0.0, float(args.rigid_assignment_prior_weight)
    )
    args.rigid_max_condition_number = max(
        1.0, float(args.rigid_max_condition_number)
    )
    args.rigid_static_displacement_tolerance = max(
        0.0, float(args.rigid_static_displacement_tolerance)
    )
    args.rigid_pairwise_audit_points = max(2, int(args.rigid_pairwise_audit_points))
    args.rigid_responsibility_cycles = max(1, int(args.rigid_responsibility_cycles))
    if args.kpart_cycles is not None:
        args.kpart_cycles = max(1, int(args.kpart_cycles))
    args.kpart_temperature_start = max(1.0e-6, float(args.kpart_temperature_start))
    args.kpart_temperature_end = max(1.0e-6, float(args.kpart_temperature_end))
    args.kpart_anneal_cycles = max(1, int(args.kpart_anneal_cycles))
    args.kpart_plateau_cycles = max(1, int(args.kpart_plateau_cycles))
    args.kpart_confidence_threshold = min(1.0, max(0.0, float(args.kpart_confidence_threshold)))
    args.kpart_dino_strength = max(0.0, float(args.kpart_dino_strength))
    args.kpart_dino_sigma = max(1.0e-6, float(args.kpart_dino_sigma))
    args.kpart_dino_knn = max(1, int(args.kpart_dino_knn))
    args.kpart_initial_oversegment = max(0, int(args.kpart_initial_oversegment))
    args.kpart_initial_supernode_knn = max(1, int(args.kpart_initial_supernode_knn))
    args.kpart_initial_dino_affinity = max(0.0, float(args.kpart_initial_dino_affinity))
    args.kpart_initial_dynamic_bootstrap_weight = max(0.0, float(args.kpart_initial_dynamic_bootstrap_weight))
    args.kpart_prior_boundary_expansion = max(0, int(args.kpart_prior_boundary_expansion))
    args.kpart_prior_anchor_threshold = min(1.0, max(0.5, float(args.kpart_prior_anchor_threshold)))
    args.kpart_prior_cleanup_min_size = max(1, int(args.kpart_prior_cleanup_min_size))
    args.kpart_dynamic_prior_weight = max(0.0, float(args.kpart_dynamic_prior_weight))
    args.kpart_merge_shared_field_weight = max(1.0e-6, float(args.kpart_merge_shared_field_weight))
    args.kpart_merge_render_weight = max(0.0, float(args.kpart_merge_render_weight))
    args.kpart_merge_dino_weight = max(0.0, float(args.kpart_merge_dino_weight))
    args.kpart_soft_temperature = max(1.0e-6, float(args.kpart_soft_temperature))
    args.kpart_field_fit_responsibility_power = max(1.000001, float(args.kpart_field_fit_responsibility_power))
    args.kpart_dino_3d_weight = max(0.0, float(args.kpart_dino_3d_weight))
    args.kpart_dino_2d_seam_weight = max(0.0, float(args.kpart_dino_2d_seam_weight))
    args.kpart_soft_logit_steps = max(1, int(args.kpart_soft_logit_steps))
    args.kpart_soft_margin_threshold = min(1.0, max(0.0, float(args.kpart_soft_margin_threshold)))
    args.kpart_soft_convergence_tolerance = max(1.0e-8, float(args.kpart_soft_convergence_tolerance))
    args.kpart_render_field_invalidation_threshold = max(0.0, float(args.kpart_render_field_invalidation_threshold))
    args.kpart_speed_signature_weight = max(0.0, float(args.kpart_speed_signature_weight))
    args.kpart_speed_huber_delta = max(1.0e-6, float(args.kpart_speed_huber_delta))
    args.kpart_dino_boundary_gamma = max(0.0, float(args.kpart_dino_boundary_gamma))
    args.kpart_graph_knn = max(1, int(args.kpart_graph_knn))
    args.kpart_boundary_weight = max(0.0, float(args.kpart_boundary_weight))
    args.kpart_render_weight = max(0.0, float(args.kpart_render_weight))
    args.kpart_render_rgb_weight = max(0.0, float(args.kpart_render_rgb_weight))
    args.kpart_render_depth_weight = max(0.0, float(args.kpart_render_depth_weight))
    args.kpart_render_dino_weight = max(0.0, float(args.kpart_render_dino_weight))
    if render_denoise_discovery:
        args.kpart_dino_3d_weight = 0.0
        args.kpart_dino_2d_seam_weight = 0.0
        args.kpart_initial_dino_affinity = 0.0
        args.kpart_merge_dino_weight = 0.0
        args.kpart_render_dino_weight = 0.0
    args.kpart_render_max_components_per_cycle = max(0, int(args.kpart_render_max_components_per_cycle))
    args.kpart_render_frame_pairs_per_component = max(1, int(args.kpart_render_frame_pairs_per_component))
    args.kpart_render_min_projected_displacement = max(0.0, float(args.kpart_render_min_projected_displacement))
    args.kpart_render_max_projected_displacement = max(
        args.kpart_render_min_projected_displacement,
        float(args.kpart_render_max_projected_displacement),
    )
    args.kpart_final_completion_passes = max(0, int(args.kpart_final_completion_passes))
    args.kpart_convergence_patience = max(1, int(args.kpart_convergence_patience))
    args.kpart_min_velocity_snr = max(0.0, float(args.kpart_min_velocity_snr))
    args.joint_trajectory_iters = max(1, int(args.joint_trajectory_iters))
    args.joint_compare_iters = max(3, int(args.joint_compare_iters))
    args.joint_compare_batch_size = max(1, int(args.joint_compare_batch_size))
    args.joint_compare_validation_cameras = max(1, int(args.joint_compare_validation_cameras))
    args.joint_compare_geometry_lr = max(1.0e-8, float(args.joint_compare_geometry_lr))
    args.joint_compare_trajectory_lr = max(1.0e-8, float(args.joint_compare_trajectory_lr))
    args.joint_compare_screw_margin = max(0.0, float(args.joint_compare_screw_margin))
    args.joint_trajectory_batch_size = max(1, int(args.joint_trajectory_batch_size))
    args.joint_trajectory_acceleration_weight = max(0.0, float(args.joint_trajectory_acceleration_weight))
    args.joint_trajectory_jerk_weight = max(0.0, float(args.joint_trajectory_jerk_weight))
    args.joint_trajectory_reverse_weight = max(0.0, float(args.joint_trajectory_reverse_weight))
    args.joint_frames = str(args.joint_frames or "all").strip() or "all"
    args.eulerian_min_support_gaussians = max(1, int(args.eulerian_min_support_gaussians))
    args.rigid_responsibility_temperature_start = max(
        1.0e-6, float(args.rigid_responsibility_temperature_start)
    )
    args.rigid_responsibility_temperature_end = max(
        1.0e-6, float(args.rigid_responsibility_temperature_end)
    )
    args.rigid_responsibility_margin = max(
        0.0, float(args.rigid_responsibility_margin)
    )
    args.rigid_responsibility_dino_sigma = max(
        1.0e-6, float(args.rigid_responsibility_dino_sigma)
    )
    args.rigid_responsibility_dino_strength = max(
        0.0, float(args.rigid_responsibility_dino_strength)
    )
    args.rigid_responsibility_dino_knn = max(
        1, int(args.rigid_responsibility_dino_knn)
    )
    for name in (
        "rigid_responsibility_static_threshold",
        "rigid_responsibility_rigid_threshold",
        "eulerian_contrastive_dino_cosine_threshold",
        "eulerian_contrastive_dino_dispersion_threshold",
        "eulerian_contrastive_static_threshold",
        "eulerian_contrastive_rigid_threshold",
        "eulerian_contrastive_min_bimodal_fraction",
    ):
        setattr(args, name, min(1.0, max(0.0, float(getattr(args, name)))))
    if (
        args.rigid_responsibility_static_threshold
        >= args.rigid_responsibility_rigid_threshold
    ):
        raise ValueError(
            "rigid responsibility static threshold must be below rigid threshold"
        )
    if (
        args.eulerian_contrastive_static_threshold
        >= args.eulerian_contrastive_rigid_threshold
    ):
        raise ValueError(
            "Eulerian contrastive static threshold must be below rigid threshold"
        )
    args.eulerian_contrastive_min_gaussians = max(
        1, int(args.eulerian_contrastive_min_gaussians)
    )

    args.curl_grid_target_cells = max(8, int(args.curl_grid_target_cells))
    args.curl_grid_padding = max(0.0, float(args.curl_grid_padding))
    args.curl_grid_min_cells_per_axis = max(2, int(args.curl_grid_min_cells_per_axis))
    args.curl_temporal_control_knots = max(4, int(args.curl_temporal_control_knots))
    args.curl_grid_grad_clip = max(0.0, float(args.curl_grid_grad_clip))
    args.curl_material_neighbors = max(1, int(args.curl_material_neighbors))
    args.curl_static_cell_purity = min(
        1.0, max(0.0, float(args.curl_static_cell_purity))
    )
    args.eulerian_velocity_gradient_min_neighbors = max(
        1, int(args.eulerian_velocity_gradient_min_neighbors)
    )
    args.eulerian_velocity_gradient_ridge = max(
        0.0, float(args.eulerian_velocity_gradient_ridge)
    )
    args.eulerian_velocity_scale = max(1.0e-6, float(args.eulerian_velocity_scale))
    args.eulerian_static_gate_threshold = min(
        0.999, max(0.5, float(args.eulerian_static_gate_threshold))
    )
    args.eulerian_static_gate_temperature = max(
        1.0e-4, float(args.eulerian_static_gate_temperature)
    )
    args.eulerian_assignment_init_floor = min(
        0.49, max(0.001, float(args.eulerian_assignment_init_floor))
    )
    args.eulerian_assignment_init_ceiling = min(
        0.999, max(0.51, float(args.eulerian_assignment_init_ceiling))
    )
    if args.eulerian_assignment_init_ceiling <= args.eulerian_assignment_init_floor:
        raise ValueError(
            "eulerian_assignment_init_ceiling must exceed eulerian_assignment_init_floor"
        )

    args.phase1_window_frames = max(2, int(args.phase1_window_frames))
    args.phase1_window_stride = max(1, int(args.phase1_window_stride))
    args.phase1_assignment_passes = max(0, int(args.phase1_assignment_passes))
    args.phase1_motion_passes = max(1, int(args.phase1_motion_passes))
    args.phase1_pair_neighbors = max(1, int(args.phase1_pair_neighbors))
    args.phase1_pair_max_pairs = max(1, int(args.phase1_pair_max_pairs))
    args.lagrangian_knn_chunk_size = max(32, int(args.lagrangian_knn_chunk_size))
    args.phase1_data_gradient_scale = max(0.0, float(args.phase1_data_gradient_scale))

    args.assignment_lr = max(0.0, float(args.assignment_lr))
    args.motion_lr = max(0.0, float(args.motion_lr))
    args.assignment_initial_dynamic_probability = min(
        0.999, max(0.501, float(args.assignment_initial_dynamic_probability))
    )
    args.assignment_initial_static_dynamic_probability = min(
        0.499, max(0.001, float(args.assignment_initial_static_dynamic_probability))
    )
    args.assignment_temperature_start = max(
        1.0e-3, float(args.assignment_temperature_start)
    )
    args.assignment_temperature_end = max(
        1.0e-3, float(args.assignment_temperature_end)
    )
    args.assignment_entropy_warmup_fraction = min(
        1.0, max(0.0, float(args.assignment_entropy_warmup_fraction))
    )
    args.assignment_prior_final_scale = min(
        1.0, max(0.0, float(args.assignment_prior_final_scale))
    )
    args.assignment_same_part_floor = min(
        1.0, max(0.0, float(args.assignment_same_part_floor))
    )
    args.assignment_same_part_floor_start = min(
        1.0, max(0.0, float(args.assignment_same_part_floor_start))
    )
    args.assignment_same_part_floor_end = min(
        1.0, max(0.0, float(args.assignment_same_part_floor_end))
    )
    args.assignment_evidence_temperature = max(
        1.0e-4, float(args.assignment_evidence_temperature)
    )
    args.dino_boundary_rigidity_strength = min(
        1.0, max(0.0, float(args.dino_boundary_rigidity_strength))
    )

    nonnegative = (
        "lambda_semantic_feature_field",
        "lambda_dino_feature_field",
        "lambda_dino_feature_field_dynamic",
        "lambda_motion_projected_velocity_flow",
        "lambda_observed_flow_velocity",
        "lambda_observed_flow_direction",
        "lambda_eulerian_strain_rate",
        "lambda_eulerian_velocity_spatial_coherence",
        "lambda_eulerian_speed_spatial_coherence",
        "lambda_eulerian_spatial_gradient_coherence",
        "lambda_eulerian_local_affine_coherence",
        "lambda_eulerian_static_speed_strain",
        "lambda_eulerian_hard_static_speed",
        "lambda_eulerian_assignment_kinematic",
        "lambda_assignment_prior",
        "lambda_assignment_spatial",
        "lambda_assignment_entropy",
        "lambda_assignment_dino_boundary",
        "lambda_curl_temporal_acceleration",
        "lambda_render_denoise_rgb",
        "lambda_render_denoise_ssim",
        "lambda_render_denoise_depth",
        "lambda_render_denoise_zero",
        "lambda_render_denoise_neighbor",
        "lambda_render_gate_prior",
        "lambda_render_gate_neighbor",
        "lambda_render_gate_entropy",
        "lambda_render_label",
        "lambda_velocity_label",
    )
    for name in nonnegative:
        setattr(args, name, max(0.0, float(getattr(args, name))))
    args.velocity_spatial_coherence_robust_delta = max(
        1.0e-8, float(args.velocity_spatial_coherence_robust_delta)
    )
    args.eulerian_projection_steps = max(0, int(args.eulerian_projection_steps))
    args.eulerian_projection_lr = max(0.0, float(args.eulerian_projection_lr))

    args.motion_evidence_future_frames = max(0, int(args.motion_evidence_future_frames))
    args.motion_evidence_max_cameras = max(0, int(args.motion_evidence_max_cameras))
    args.motion_evidence_min_dynamic_points = max(
        1, int(args.motion_evidence_min_dynamic_points)
    )
    args.motion_evidence_html_max_points = max(
        100, int(args.motion_evidence_html_max_points)
    )
    args.static_evidence_refine_steps = max(0, int(args.static_evidence_refine_steps))
    args.static_evidence_refine_lr = max(1.0e-6, float(args.static_evidence_refine_lr))
    args.static_evidence_refine_prior_weight = max(
        1.0e-8, float(args.static_evidence_refine_prior_weight)
    )
    args.transported_html_max_points = max(1, int(args.transported_html_max_points))
    args.wandb_log_interval = max(1, int(args.wandb_log_interval))
    args.rigid_part_outer_iters = max(1, int(args.rigid_part_outer_iters))
    args.rigid_part_local_motion_steps = max(0, int(args.rigid_part_local_motion_steps))
    args.rigid_part_label_steps = max(1, int(args.rigid_part_label_steps))
    args.rigid_part_raw_lr = max(1.0e-8, float(args.rigid_part_raw_lr))
    args.rigid_part_render_frames = max(0, int(args.rigid_part_render_frames))
    args.rigid_part_render_cameras = max(0, int(args.rigid_part_render_cameras))
    args.rigid_part_refine_iters = max(0, int(args.rigid_part_refine_iters))
    args.rigid_part_lr = max(1.0e-8, float(args.rigid_part_lr))
    args.rigid_part_label_temperature = max(1.0e-6, float(args.rigid_part_label_temperature))
    args.rigid_part_label_temperature_final = max(1.0e-6, float(args.rigid_part_label_temperature_final))
    args.rigid_part_knn = max(1, int(args.rigid_part_knn))
    args.rigid_part_max_frames = max(2, int(args.rigid_part_max_frames))
    args.rigid_part_min_fraction = min(1.0, max(0.0, float(args.rigid_part_min_fraction)))
    args.rigid_part_final_assignment_temperature = max(1.0e-8, float(args.rigid_part_final_assignment_temperature))
    args.rigid_part_endpoint_chunk = max(1, int(args.rigid_part_endpoint_chunk))
    args.part_embedding_dim = max(1, int(args.part_embedding_dim))
    args.part_embedding_init_std = max(0.0, float(args.part_embedding_init_std))
    args.part_outer_iters = max(1, int(args.part_outer_iters))
    args.part_outer_embedding_steps = max(1, int(args.part_outer_embedding_steps))
    args.part_embedding_lr = max(1.0e-8, float(args.part_embedding_lr))
    args.part_temperature = max(1.0e-6, float(args.part_temperature))
    args.part_temperature_final = max(1.0e-6, float(args.part_temperature_final))
    args.part_final_assignment_temperature = max(1.0e-8, float(args.part_final_assignment_temperature))
    args.static_anchor_distance_ratio = max(0.0, float(args.static_anchor_distance_ratio))
    args.static_anchor_neighborhood_k = max(1, int(args.static_anchor_neighborhood_k))
    args.killing_inner_steps_per_frame = max(1, int(args.killing_inner_steps_per_frame))
    args.nested_killing_control_points = max(4, int(args.nested_killing_control_points))
    args.nested_killing_kernel_scale = max(0.0, float(args.nested_killing_kernel_scale))
    args.killing_lr = max(1.0e-8, float(args.killing_lr))
    args.lambda_inner_rgb = max(0.0, float(args.lambda_inner_rgb))
    args.lambda_inner_ssim = max(0.0, float(args.lambda_inner_ssim))
    args.lambda_inner_depth = max(0.0, float(args.lambda_inner_depth))
    args.lambda_inner_observed_velocity = max(0.0, float(args.lambda_inner_observed_velocity))
    args.lambda_inner_strain = max(0.0, float(args.lambda_inner_strain))
    args.lambda_inner_omega_consistency = max(0.0, float(args.lambda_inner_omega_consistency))
    args.lambda_inner_b_consistency = max(0.0, float(args.lambda_inner_b_consistency))
    args.lambda_inner_twist_temporal = max(0.0, float(args.lambda_inner_twist_temporal))
    args.lambda_inner_twist_acceleration = max(0.0, float(args.lambda_inner_twist_acceleration))
    args.lambda_inner_field_separation = max(0.0, float(args.lambda_inner_field_separation))
    args.nested_motion_separation_margin = max(0.0, float(args.nested_motion_separation_margin))
    args.lambda_static_anchor = max(0.0, float(args.lambda_static_anchor))
    args.lambda_outer_candidate_nn = max(0.0, float(args.lambda_outer_candidate_nn))
    args.lambda_outer_final_nn = max(0.0, float(args.lambda_outer_final_nn))
    args.lambda_outer_global_com = max(0.0, float(args.lambda_outer_global_com))
    args.lambda_outer_global_inertia = max(0.0, float(args.lambda_outer_global_inertia))
    args.lambda_outer_part_com = max(0.0, float(args.lambda_outer_part_com))
    args.lambda_outer_part_inertia = max(0.0, float(args.lambda_outer_part_inertia))
    args.lambda_outer_embedding_motion = max(0.0, float(args.lambda_outer_embedding_motion))
    args.lambda_outer_candidate_margin = max(0.0, float(args.lambda_outer_candidate_margin))
    args.lambda_outer_entropy = max(0.0, float(args.lambda_outer_entropy))
    args.lambda_outer_occupancy = max(0.0, float(args.lambda_outer_occupancy))
    args.outer_candidate_margin = max(0.0, float(args.outer_candidate_margin))
    # ---- Endpoint-contrastive initialization (spec §23) ---------------------
    args.init_mode = str(args.init_mode).strip().lower()
    if args.init_mode not in ("legacy", "endpoint_contrastive", "high_precision_bootstrap"):
        raise ValueError(
            "--init_mode must be legacy, endpoint_contrastive or high_precision_bootstrap, "
            f"got {args.init_mode}")
    # ---- High-precision bootstrap (§4, §5, §10) ----------------------------
    args.bootstrap_dynamic_quantile = min(max(float(args.bootstrap_dynamic_quantile), 0.5), 1.0)
    args.bootstrap_static_quantile = min(max(float(args.bootstrap_static_quantile), 0.0), 0.5)
    args.lambda_bootstrap_knn = max(0.0, float(args.lambda_bootstrap_knn))
    args.lambda_bootstrap_anchor = max(0.0, float(args.lambda_bootstrap_anchor))
    args.lambda_bootstrap_dynamic_core = max(0.0, float(args.lambda_bootstrap_dynamic_core))
    args.lambda_bootstrap_geometric_static = max(0.0, float(args.lambda_bootstrap_geometric_static))
    args.lambda_bootstrap_polar = max(0.0, float(args.lambda_bootstrap_polar))
    args.bootstrap_iterations = max(0, int(args.bootstrap_iterations))
    if args.init_mode == "high_precision_bootstrap":
        if not bool(getattr(args, "outer_soft_labels", True)):
            raise ValueError("--init_mode high_precision_bootstrap requires --outer_soft_labels: "
                             "§14 makes the SOFT outer loop the segmentation algorithm")
        if args.lambda_bootstrap_polar != 0.0:
            # §3 is explicit: no global entropy minimization in bootstrap mode.
            print(f"[Bootstrap] WARNING: lambda_bootstrap_polar={args.lambda_bootstrap_polar} "
                  "is non-zero; §3 forbids global polarization in bootstrap mode. This is the "
                  "legacy ablation path, not the specified behaviour.")
    args.init_graph_k = max(2, int(args.init_graph_k))
    args.init_binary_iters = max(0, int(args.init_binary_iters))
    args.init_binary_lr = max(1.0e-8, float(args.init_binary_lr))
    # Both flags are tail fractions, so 0.5 each is the most either can mean.
    args.init_static_quantile = min(max(float(args.init_static_quantile), 0.0), 0.5)
    args.init_dynamic_quantile = min(max(float(args.init_dynamic_quantile), 0.0), 0.5)
    # The margin compares two probabilities, so it cannot exceed 1.
    args.init_contrastive_margin = min(max(float(args.init_contrastive_margin), 0.0), 1.0)
    args.lambda_init_same = max(0.0, float(args.lambda_init_same))
    args.lambda_init_diff = max(0.0, float(args.lambda_init_diff))
    args.lambda_init_static = max(0.0, float(args.lambda_init_static))
    args.lambda_init_dynamic = max(0.0, float(args.lambda_init_dynamic))
    args.lambda_init_trusted_static = max(0.0, float(args.lambda_init_trusted_static))
    if args.init_mode == "endpoint_contrastive":
        # §11 orders the anchor weights; a run that violates it is silently a
        # different experiment from the one the spec describes.
        if not (args.lambda_init_trusted_static >= args.lambda_init_static
                >= args.lambda_init_dynamic):
            raise ValueError(
                "spec §11 requires lambda_init_trusted_static >= lambda_init_static >= "
                f"lambda_init_dynamic, got {args.lambda_init_trusted_static} / "
                f"{args.lambda_init_static} / {args.lambda_init_dynamic}")
    # ---- Soft label outer cycle (CODEX_soft_labels §46) ---------------------
    args.init_dynamic_anchor_quantile = min(max(float(args.init_dynamic_anchor_quantile), 0.5), 1.0)
    args.lambda_init_pair = max(0.0, float(args.lambda_init_pair))
    args.lambda_init_polar = max(0.0, float(args.lambda_init_polar))
    args.lambda_init_anchor = max(0.0, float(args.lambda_init_anchor))
    args.init_pair_confidence_gamma = max(0.0, float(args.init_pair_confidence_gamma))
    args.init_soft_cluster_iters = max(1, int(args.init_soft_cluster_iters))
    args.init_soft_cluster_temperature = max(0.0, float(args.init_soft_cluster_temperature))
    args.outer_render_temperature = max(1.0e-6, float(args.outer_render_temperature))
    # ---- Algorithmic corrections -------------------------------------------
    args.lambda_init_smooth = max(0.0, float(args.lambda_init_smooth))
    args.init_positive_similarity_gamma = max(0.0, float(args.init_positive_similarity_gamma))
    args.init_endpoint_logit_bias = max(0.0, float(args.init_endpoint_logit_bias))
    args.polarization_ramp_start = min(max(float(args.polarization_ramp_start), 0.0), 1.0)
    args.polarization_ramp_end = min(max(float(args.polarization_ramp_end), 0.0), 1.0)
    if args.polarization_ramp_end < args.polarization_ramp_start:
        raise ValueError("§9: polarization_ramp_end must be >= polarization_ramp_start, got "
                         f"{args.polarization_ramp_end} < {args.polarization_ramp_start}")
    args.outer_confidence_mode = str(args.outer_confidence_mode).strip().lower()
    # ---- Robust inner Killing field ----------------------------------------
    args.inner_solver = str(args.inner_solver).strip().lower()
    args.inner_alternation_rounds = max(1, int(args.inner_alternation_rounds))
    args.inner_min_fit_points = max(6, int(args.inner_min_fit_points))
    args.inner_projection_reg = max(0.0, float(args.inner_projection_reg))
    args.inner_render_frame_stride = max(1, int(args.inner_render_frame_stride))
    args.lambda_rigid_proj = max(0.0, float(args.lambda_rigid_proj))
    args.inner_spatial_steps = max(1, int(args.inner_spatial_steps))
    args.inner_scalar_steps = max(1, int(args.inner_scalar_steps))
    # ---- CODEX_DECOUPLED ----------------------------------------------------
    args.inner_decoupled_ablation = str(args.inner_decoupled_ablation).strip().upper()
    if args.inner_decoupled_ablation not in ("A", "B", "C", "D", "E"):
        raise ValueError("--inner_decoupled_ablation must be one of A/B/C/D/E, got "
                         f"{args.inner_decoupled_ablation}")
    args.lambda_frame_joint = max(0.0, float(args.lambda_frame_joint))
    args.lambda_final_pcd = max(0.0, float(args.lambda_final_pcd))
    args.inner_frame_robust = str(args.inner_frame_robust).strip().lower()
    if args.inner_frame_robust not in ("huber", "charbonnier", "l2"):
        raise ValueError("--inner_frame_robust must be huber/charbonnier/l2, got "
                         f"{args.inner_frame_robust}")
    if float(args.inner_frame_robust_scale) <= 0.0:
        raise ValueError("--inner_frame_robust_scale must be > 0, got "
                         f"{args.inner_frame_robust_scale}")
    if float(args.inner_robust_tau_scale) <= 0.0:
        raise ValueError(f"--inner_robust_tau_scale must be > 0, got {args.inner_robust_tau_scale}")
    args.inner_irls_steps = max(1, int(args.inner_irls_steps))
    if float(args.inner_scalar_lr) <= 0.0:
        raise ValueError(f"--inner_scalar_lr must be > 0, got {args.inner_scalar_lr}")
    for _name in ("inner_final_pcd_forward_trim", "inner_final_pcd_backward_trim"):
        _value = float(getattr(args, _name))
        if not 0.0 < _value <= 1.0:
            raise ValueError(f"--{_name} must be in (0, 1], got {_value}")
    if float(args.inner_final_pcd_budget) <= 0.0:
        raise ValueError(f"--inner_final_pcd_budget must be > 0, got {args.inner_final_pcd_budget}")
    # ---- IEKF (section 33) --------------------------------------------------
    if bool(getattr(args, "use_sparse_alpha_iekf", False)):
        if not (args.iekf_use_rgb or args.iekf_use_depth):
            raise ValueError("--use_sparse_alpha_iekf needs at least one of RGB or depth")
        if int(args.iekf_max_iters) < 1:
            raise ValueError(f"--iekf_max_iters must be >= 1, got {args.iekf_max_iters}")
        if not (0.0 < float(args.iekf_line_search_shrink) < 1.0):
            raise ValueError("--iekf_line_search_shrink must lie in (0, 1), got "
                             f"{args.iekf_line_search_shrink}")
        if not (0.0 < float(args.iekf_alpha_min) <= 1.0):
            raise ValueError(f"--iekf_alpha_min must lie in (0, 1], got {args.iekf_alpha_min}")
        if float(args.iekf_lm_min) > float(args.iekf_lm_max):
            raise ValueError("--iekf_lm_min must not exceed --iekf_lm_max")
        if float(args.iekf_student_t_nu) <= 0.0:
            raise ValueError(f"--iekf_student_t_nu must be > 0, got {args.iekf_student_t_nu}")
        if float(args.iekf_rgb_sigma) <= 0.0:
            raise ValueError(f"--iekf_rgb_sigma must be > 0, got {args.iekf_rgb_sigma}")
        for _n in ("iekf_process_noise_pos", "iekf_process_noise_rot",
                   "iekf_process_noise_vel", "iekf_process_noise_omega"):
            if float(getattr(args, _n)) < 0.0:
                raise ValueError(f"--{_n} must be >= 0, got {getattr(args, _n)}")
        from estimator.measurement import make_depth_sigma_model
        make_depth_sigma_model(args.iekf_depth_sigma_model)   # validates eagerly
    args.joint_models = tuple(m.strip().lower() for m in str(args.joint_models).split(",")
                              if m.strip())
    for _m in args.joint_models:
        if _m not in ("prismatic", "revolute"):
            # Section 0 rule 13: do NOT add screw/helical to the initial estimator.
            raise ValueError(f"--joint_models supports prismatic/revolute only, got {_m!r}")
    if bool(getattr(args, "use_gravity", False)) and not bool(
            getattr(args, "use_wrench_dynamics", False)):
        raise ValueError("--use_gravity requires --use_wrench_dynamics")
    if bool(getattr(args, "use_coulomb_friction", False)) and not bool(
            getattr(args, "use_wrench_dynamics", False)):
        raise ValueError("--use_coulomb_friction requires --use_wrench_dynamics")
    args.inner_support_knn = max(2, int(args.inner_support_knn))
    for _name in ("inner_core_dynamic_confidence", "inner_core_part_confidence",
                  "inner_core_confidence_floor", "inner_attach_min_confidence",
                  "inner_attach_affinity_quantile"):
        _v = float(getattr(args, _name))
        if not 0.0 <= _v <= 1.0:
            raise ValueError(f"--{_name} must be in [0, 1], got {_v}")
    if float(args.inner_core_confidence_floor) > float(args.inner_core_dynamic_confidence):
        raise ValueError("--inner_core_confidence_floor must not exceed "
                         "--inner_core_dynamic_confidence "
                         f"({args.inner_core_confidence_floor} > {args.inner_core_dynamic_confidence})")
    if args.inner_core_min_points is not None:
        args.inner_core_min_points = max(1, int(args.inner_core_min_points))
    if float(args.inner_attach_margin_ratio) < 1.0:
        raise ValueError("--inner_attach_margin_ratio must be >= 1 (a margin below 1 would "
                         f"attach to the runner-up cluster), got {args.inner_attach_margin_ratio}")
    for _name in ("inner_cluster_part_gamma", "inner_fit_part_gamma"):
        if float(getattr(args, _name)) < 0.0:
            raise ValueError(f"--{_name} must be >= 0, got {getattr(args, _name)}")
    for _name in ("lambda_geometry_render", "lambda_geometry_depth", "lambda_motion_direction",
                  "lambda_geometry_final_pcd", "lambda_scalar_render", "lambda_scalar_depth",
                  "lambda_motion_magnitude", "lambda_scalar_final_pcd"):
        if float(getattr(args, _name)) < 0.0:
            raise ValueError(f"--{_name} must be >= 0, got {getattr(args, _name)}")
    args.inner_render_frame_batch = max(1, int(args.inner_render_frame_batch))
    args.inner_motion_prior = str(args.inner_motion_prior).strip().lower()
    if args.inner_motion_prior not in ("none", "revolute", "prismatic"):
        raise ValueError(f"--inner_motion_prior must be none/revolute/prismatic, "
                         f"got {args.inner_motion_prior}")
    args.lambda_revolute_prior = max(0.0, float(args.lambda_revolute_prior))
    args.lambda_prismatic_prior = max(0.0, float(args.lambda_prismatic_prior))
    if args.inner_motion_prior == "none" and (
            args.lambda_revolute_prior != 0.0 or args.lambda_prismatic_prior != 0.0):
        # §50 Test U / §30: "none" must mean exactly zero, with no hidden bias.
        raise ValueError("--inner_motion_prior none requires both prior weights at 0, got "
                         f"revolute={args.lambda_revolute_prior} "
                         f"prismatic={args.lambda_prismatic_prior}")
    args.inner_outlier_mad_scale = max(0.0, float(args.inner_outlier_mad_scale))
    args.inner_outlier_local_dynamic_min = min(max(
        float(args.inner_outlier_local_dynamic_min), 0.0), 1.0)
    args.inner_fit_confidence_floor = min(max(float(args.inner_fit_confidence_floor), 0.0), 0.999)
    args.inner_fit_confidence_gamma = max(0.0, float(args.inner_fit_confidence_gamma))
    args.inner_cluster_core_confidence = min(max(
        float(args.inner_cluster_core_confidence), 0.0), 0.999)
    args.inner_cluster_knn = max(2, int(args.inner_cluster_knn))
    args.inner_cluster_confidence_gamma = max(0.0, float(args.inner_cluster_confidence_gamma))
    args.lambda_exact_omega = max(0.0, float(args.lambda_exact_omega))
    args.lambda_exact_translation = max(0.0, float(args.lambda_exact_translation))
    args.lambda_rigid_field_probe = max(0.0, float(args.lambda_rigid_field_probe))
    args.lambda_scalar_consistency = max(0.0, float(args.lambda_scalar_consistency))
    args.lambda_s_continuity = max(0.0, float(args.lambda_s_continuity))
    args.lambda_s_flip = max(0.0, float(args.lambda_s_flip))
    args.lambda_s_jerk = max(0.0, float(args.lambda_s_jerk))
    # ---- Render-response contrast (§34) -------------------------------------
    args.render_contrast_steps = max(0, int(args.render_contrast_steps))
    args.render_contrast_lr = max(1.0e-8, float(args.render_contrast_lr))
    args.render_contrast_positive_threshold = min(max(
        float(args.render_contrast_positive_threshold), 0.0), 0.999)
    args.render_contrast_negative_threshold = min(max(
        float(args.render_contrast_negative_threshold), 0.0), 0.999)
    args.render_contrast_min_shared_obs = max(1, int(args.render_contrast_min_shared_obs))
    args.lambda_render_contrast_preserve = max(0.0, float(args.lambda_render_contrast_preserve))
    args.lambda_render_contrast_agree = max(0.0, float(args.lambda_render_contrast_agree))
    args.lambda_render_contrast_disagree = max(0.0, float(args.lambda_render_contrast_disagree))
    args.render_contrast_sign_temperature = max(1.0e-6, float(args.render_contrast_sign_temperature))
    if bool(getattr(args, "enable_render_response_contrast", False)):
        # §2: the contrast refines the SOFT base proposal; there is no base
        # proposal to refine in the legacy hard outer loop.
        if not bool(getattr(args, "outer_soft_labels", True)):
            raise ValueError("--enable_render_response_contrast requires --outer_soft_labels")
        if not bool(getattr(args, "enable_pixel_matching", False)):
            raise ValueError("--enable_render_response_contrast requires --enable_pixel_matching "
                             "(§4: fingerprints come from the candidate energies)")
    # ---- Motion-boundary split ---------------------------------------------
    args.boundary_split_local_class_min = min(max(float(args.boundary_split_local_class_min), 0.0), 0.5)
    args.boundary_split_extent_sigma = max(0.0, float(args.boundary_split_extent_sigma))
    args.boundary_split_offset_sigma = max(0.0, float(args.boundary_split_offset_sigma))
    args.boundary_split_max_offset_sigma = max(0.0, float(args.boundary_split_max_offset_sigma))
    args.boundary_split_gradient_ridge = max(0.0, float(args.boundary_split_gradient_ridge))
    args.boundary_split_label_mix = min(max(float(args.boundary_split_label_mix), 0.0), 0.5)
    args.boundary_split_max_fraction = min(max(float(args.boundary_split_max_fraction), 0.0), 1.0)
    args.boundary_split_max_count = max(0, int(args.boundary_split_max_count))
    args.boundary_split_size_threshold = max(0.0, float(args.boundary_split_size_threshold))
    args.boundary_split_confident_class = min(max(float(args.boundary_split_confident_class), 0.0), 1.0)
    if args.boundary_split_offset_sigma > args.boundary_split_max_offset_sigma:
        raise ValueError("§11: boundary_split_offset_sigma must not exceed "
                         "boundary_split_max_offset_sigma")
    if bool(getattr(args, "enable_motion_boundary_split", False)) and \
            not bool(getattr(args, "outer_soft_labels", True)):
        # §2/§18: the split is driven by the persistent SOFT ownership field.
        raise ValueError("--enable_motion_boundary_split requires --outer_soft_labels")
    args.temporal_score_render_weight = max(0.0, float(args.temporal_score_render_weight))
    args.temporal_score_match_weight = max(0.0, float(args.temporal_score_match_weight))
    # §21/§22: the temporal score must stay render-dominant.
    if args.temporal_score_match_weight > args.temporal_score_render_weight:
        raise ValueError(
            "§22 requires a render-dominant temporal score, but "
            f"temporal_score_match_weight={args.temporal_score_match_weight} > "
            f"temporal_score_render_weight={args.temporal_score_render_weight}")
    args.outer_probability_smoothing = min(max(float(args.outer_probability_smoothing), 0.0), 1.0)
    # Soft labels default on for the nested render-graph solver. Their
    # dependencies must not prevent static endpoint (or legacy dynamic)
    # training, which never runs that solver.
    uses_nested_outer = nested_killing_discovery or args.outer_label_mode == "render_graph"
    if uses_nested_outer and bool(getattr(args, "outer_soft_labels", True)):
        # §21/§26: the soft update consumes the pixel matcher's candidate energies
        # and the temporal score is built from the same matcher.
        if str(getattr(args, "outer_label_mode", "learned")).lower() != "render_graph":
            raise ValueError("--outer_soft_labels requires --outer_label_mode render_graph")
        if not bool(getattr(args, "enable_pixel_matching", False)):
            raise ValueError("--outer_soft_labels requires --enable_pixel_matching")
    # ---- Temporal credit outer cycle (spec §41) -----------------------------
    args.lambda_score_render = max(0.0, float(args.lambda_score_render))
    args.lambda_score_match = max(0.0, float(args.lambda_score_match))
    args.lambda_score_final_nn = max(0.0, float(args.lambda_score_final_nn))
    args.temporal_credit_graph_k = max(2, int(args.temporal_credit_graph_k))
    args.temporal_reward_clip = max(1.0e-6, float(args.temporal_reward_clip))
    args.temporal_memory_decay = min(max(float(args.temporal_memory_decay), 0.0), 1.0)
    args.temporal_memory_global_decay = min(max(float(args.temporal_memory_global_decay), 0.0), 1.0)
    args.lambda_temporal_memory = max(0.0, float(args.lambda_temporal_memory))
    args.temporal_memory_margin_cap = max(0.0, float(args.temporal_memory_margin_cap))
    args.temporal_score_frame_stride = max(1, int(args.temporal_score_frame_stride))
    args.temporal_global_reward_weight = min(max(float(args.temporal_global_reward_weight), 0.0), 1.0)
    args.temporal_component_reward_weight = min(
        max(float(args.temporal_component_reward_weight), 0.0), 1.0)
    if args.temporal_global_reward_weight + args.temporal_component_reward_weight > 1.0:
        raise ValueError(
            "spec §18/§33: temporal_global_reward_weight + temporal_component_reward_weight "
            f"must be <= 1, got {args.temporal_global_reward_weight} + "
            f"{args.temporal_component_reward_weight}")
    if bool(getattr(args, "enable_temporal_credit", False)):
        # §23/§25: the memory biases the render-graph candidate score. Without
        # that proposal mechanism there is nothing for the memory to bias.
        if str(getattr(args, "outer_label_mode", "learned")).lower() != "render_graph":
            raise ValueError("--enable_temporal_credit requires --outer_label_mode render_graph")
        if not bool(getattr(args, "enable_pixel_matching", False)):
            raise ValueError("--enable_temporal_credit requires --enable_pixel_matching "
                             "(§5: the pixel-mediated matcher supplies S_match)")
    # ---- FIRM v3 normalisation ---------------------------------------------
    args.killing_neighbor_k = max(2, int(args.killing_neighbor_k))
    args.killing_neighbor_chunk = max(64, int(args.killing_neighbor_chunk))
    args.killing_camera_chunk = max(0, int(args.killing_camera_chunk))
    args.killing_field_diagnostic_points = max(16, int(args.killing_field_diagnostic_points))
    args.observed_velocity_c_min = min(max(float(args.observed_velocity_c_min), 0.0), 0.999)
    args.observed_velocity_gamma = max(0.0, float(args.observed_velocity_gamma))
    # §13: the observed-velocity share of the inner objective is hard-capped.
    args.observed_velocity_rho_max = min(max(float(args.observed_velocity_rho_max), 0.0), 0.95)
    args.observed_velocity_min_high_conf = max(0, int(args.observed_velocity_min_high_conf))
    args.observed_velocity_huber_delta = max(1.0e-6, float(args.observed_velocity_huber_delta))
    args.field_separation_alpha_v = max(0.0, float(args.field_separation_alpha_v))
    args.field_separation_alpha_omega = max(0.0, float(args.field_separation_alpha_omega))
    args.killing_field_symmetry_break = max(0.0, float(args.killing_field_symmetry_break))
    args.lambda_inner_rigid_speed = max(0.0, float(args.lambda_inner_rigid_speed))
    args.partwise_twist_reg = max(0.0, float(args.partwise_twist_reg))
    args.lambda_outer_video = max(0.0, float(args.lambda_outer_video))
    args.lambda_outer_label_graph = max(0.0, float(args.lambda_outer_label_graph))
    args.lambda_outer_kinetic = max(0.0, float(args.lambda_outer_kinetic))
    args.rho_3d_gradient_cap = max(0.0, float(args.rho_3d_gradient_cap))
    args.rho_local_gradient_cap = max(0.0, float(args.rho_local_gradient_cap))
    args.raster_top_contributors = min(max(0, int(args.raster_top_contributors)), 8)
    args.raster_contributor_min_weight = max(0.0, float(args.raster_contributor_min_weight))
    args.lambda_pixel_candidate = max(0.0, float(args.lambda_pixel_candidate))
    args.lambda_pixel_margin = max(0.0, float(args.lambda_pixel_margin))
    args.pixel_candidate_margin = max(0.0, float(args.pixel_candidate_margin))
    args.pixel_footprint_window = min(max(0, int(args.pixel_footprint_window)), 16)
    args.pixel_sigma_z_ratio = max(1.0e-6, float(args.pixel_sigma_z_ratio))
    args.outer_3d_cap_interval = max(1, int(args.outer_3d_cap_interval))
    args.outer_label_graph_sigma = max(1.0e-6, float(args.outer_label_graph_sigma))
    args.outer_video_frame_block = max(0, int(args.outer_video_frame_block))
    args.outer_video_debug_every = max(0, int(args.outer_video_debug_every))
    args.outer_top_gradient_points = max(0, int(args.outer_top_gradient_points))
    args.lambda_multiview_view_var = max(0.0, float(args.lambda_multiview_view_var))
    args.multiview_visibility_floor = min(max(float(args.multiview_visibility_floor), 0.0), 1.0)
    args.multiview_debug_every = max(0, int(args.multiview_debug_every))
    args.multiview_grad_probe_every = max(0, int(args.multiview_grad_probe_every))
    args.anchor_hard_freeze_outer_iters = max(0, int(args.anchor_hard_freeze_outer_iters))
    # §23: tau stays soft; it may never be driven below 0.4.
    args.part_temperature_hard_floor = max(0.4, float(args.part_temperature_hard_floor))
    args.part_temperature_soft_floor = max(
        float(args.part_temperature_hard_floor), min(float(args.part_temperature_soft_floor), 1.0)
    )
    args.part_temperature_decay = min(max(float(args.part_temperature_decay), 0.1), 1.0)
    args.part_temperature_churn_tolerance = max(0.0, float(args.part_temperature_churn_tolerance))
    args.lambda_outer_part_mass = max(0.0, float(args.lambda_outer_part_mass))
    args.lambda_outer_trust_region = max(0.0, float(args.lambda_outer_trust_region))
    args.outer_geometry_guard_eps = max(0.0, float(args.outer_geometry_guard_eps))
    args.outer_geometry_guard_backtracks = max(0, int(args.outer_geometry_guard_backtracks))
    args.inner_accept_render_tolerance = abs(float(args.inner_accept_render_tolerance))
    args.inner_accept_endpoint_tolerance = abs(float(args.inner_accept_endpoint_tolerance))
    args.inner_reject_observed_decay = min(max(float(args.inner_reject_observed_decay), 0.0), 1.0)
    args.lambda_rigid_trajectory = max(0.0, float(args.lambda_rigid_trajectory))
    args.lambda_rigid_graph = max(0.0, float(args.lambda_rigid_graph))
    args.lambda_rigid_graph_repulsive = max(0.0, float(args.lambda_rigid_graph_repulsive))
    args.lambda_rigid_prior = max(0.0, float(args.lambda_rigid_prior))
    args.lambda_rigid_moving_complexity = max(0.0, float(args.lambda_rigid_moving_complexity))
    args.lambda_rigid_endpoint = max(0.0, float(args.lambda_rigid_endpoint))
    args.lambda_rigid_entropy = max(0.0, float(args.lambda_rigid_entropy))
    args.lambda_rigid_twist = max(0.0, float(args.lambda_rigid_twist))
    args.lambda_rigid_pair = max(0.0, float(args.lambda_rigid_pair))
    args.lambda_rigid_distance_rate = max(0.0, float(args.lambda_rigid_distance_rate))
    args.lambda_rigid_com = max(0.0, float(args.lambda_rigid_com))
    args.lambda_rigid_inertia = max(0.0, float(args.lambda_rigid_inertia))
    args.lambda_rigid_com_rate = max(0.0, float(args.lambda_rigid_com_rate))
    args.lambda_rigid_inertia_rate = max(0.0, float(args.lambda_rigid_inertia_rate))
    args.lambda_rigid_inertia_invariants = max(0.0, float(args.lambda_rigid_inertia_invariants))
    args.lambda_raw_render_rgb = max(0.0, float(args.lambda_raw_render_rgb))
    args.lambda_raw_render_depth = max(0.0, float(args.lambda_raw_render_depth))
    args.lambda_raw_observed_velocity = max(0.0, float(args.lambda_raw_observed_velocity))
    args.lambda_raw_killing = max(0.0, float(args.lambda_raw_killing))
    args.lambda_raw_temporal = max(0.0, float(args.lambda_raw_temporal))
    args.lambda_raw_rigid_feedback = max(0.0, float(args.lambda_raw_rigid_feedback))
    args.lambda_rigid_occupancy = max(0.0, float(args.lambda_rigid_occupancy))
    args.lambda_endpoint_geometry = max(0.0, float(args.lambda_endpoint_geometry))
    args.lambda_endpoint_part_com = max(0.0, float(args.lambda_endpoint_part_com))
    args.lambda_endpoint_part_inertia = max(0.0, float(args.lambda_endpoint_part_inertia))
    args.lambda_endpoint_global_com = max(0.0, float(args.lambda_endpoint_global_com))
    args.lambda_endpoint_global_inertia = max(0.0, float(args.lambda_endpoint_global_inertia))

    if args.dynamic_motion:
        if not args.phase1_only and not args.kpart_rigid_discovery:
            raise ValueError(
                "The retained dynamic implementation requires --phase1_only"
            )
        if not args.use_semantic_feature_field:
            raise ValueError("Dynamic training requires --use_semantic_feature_field")
        if args.phase1_motion_representation in RIGID_KILLING_REPRESENTATIONS:
            if not args.phase1_depth_pretrain_only and not args.use_motion_correspondence:
                raise ValueError("Rigid Killing representations require motion correspondence")
            if (
                args.num_moving_parts != 1
                and not args.kpart_rigid_discovery
                and not bool(getattr(args, "enable_nested_killing_part_discovery", False))
            ):
                raise ValueError("Rigid Killing representations support one moving material only outside K-part/nested discovery")
            if (
                args.phase1_motion_representation == "rigid_killing_lagrangian"
                and not args.sequence_pretrain_assignment
                and not args.phase1_neutral_rigid_discovery
                and not args.rigid_responsibility_training
                and not args.kpart_rigid_discovery
                and not bool(getattr(args, "enable_nested_killing_part_discovery", False))
                and not dynamic_prior_supplied
            ):
                raise ValueError(
                    "rigid_killing_lagrangian requires whole-sequence assignment pretraining or --dynamic_prior_path"
                )
            if args.phase1_depth_sequence_pretrain and not args.dynamic_use_depth:
                raise ValueError("depth sequence pretraining requires --dynamic_use_depth")
            if args.phase1_neutral_rigid_discovery and not args.dynamic_use_depth:
                raise ValueError(
                    "neutral post-solve assignment requires --dynamic_use_depth"
                )
            if args.rigid_responsibility_training and not args.dynamic_use_depth:
                raise ValueError(
                    "soft rigid responsibility training requires --dynamic_use_depth"
                )
            if args.rigid_responsibility_training and not args.use_dino_feature_field:
                raise ValueError(
                    "Matched rigid-responsibility training requires "
                    "--use_dino_feature_field so both supports share one DINO graph"
                )
            if args.kpart_rigid_discovery and not args.dynamic_use_depth:
                raise ValueError("K-part whole-sequence expert comparison requires --dynamic_use_depth")
            if args.kpart_sam3 and args.kpart_rigid_discovery and not args.use_sam3_feature_field:
                raise ValueError("--kpart_sam3 requires --use_sam3_feature_field")
            if args.kpart_endpoint_transport and not args.use_endpoint_static_model:
                raise ValueError("--kpart_endpoint_transport requires --use_endpoint_static_model")
            if args.endpoint_sam3_weight > 0.0 and not args.use_sam3_feature_field:
                raise ValueError("--endpoint_sam3_weight > 0 requires --use_sam3_feature_field")
            if args.endpoint_transport_mode == "unbalanced_ot":
                raise ValueError("--endpoint_transport_mode unbalanced_ot is intentionally not implemented; use local_soft")
            if (
                args.kpart_graph_segmentation
                and args.kpart_rigid_discovery
                and not (
                    args.kpart_sam3
                    or render_denoise_discovery
                    or args.kpart_endpoint_transport
                    or dynamic_prior_supplied
                )
            ):
                raise ValueError("--kpart_graph_segmentation requires SAM3, render-denoise, endpoint transport, or --dynamic_prior_path")
            if args.rigid_use_dino_assignment_prior and not args.use_dino_feature_field:
                raise ValueError(
                    "--rigid_use_dino_assignment_prior requires --use_dino_feature_field"
                )
            if args.rigid_use_dino_assignment_prior:
                raise ValueError(
                    "The rigid DINO assignment prior remains disabled until its "
                    "frame-0 cosine and attribution-gradient validation pass"
                )
            # These losses are physically incorrect for a rigid rotation and are
            # retained only by the legacy free-field branch.
            args.lambda_eulerian_strain_rate = 0.0
            args.lambda_eulerian_velocity_spatial_coherence = 0.0
            args.lambda_eulerian_speed_spatial_coherence = 0.0
            args.lambda_eulerian_spatial_gradient_coherence = 0.0
            args.lambda_eulerian_local_affine_coherence = 0.0
        args.iterations = args.static_train_iters + args.free_field_iters
    else:
        args.static_train_iters = max(0, int(args.iterations))
        args.iterations = args.static_train_iters
    return args


def get_combined_args(parser: ArgumentParser):
    command_line = sys.argv[1:]
    config = "Namespace()"
    args_command = parser.parse_args(command_line)
    try:
        config_path = os.path.join(args_command.model_path, "cfg_args")
        with open(config_path, "r", encoding="utf-8") as stream:
            config = stream.read()
    except (TypeError, FileNotFoundError):
        pass
    args_config = eval(config, {"Namespace": Namespace})
    merged = vars(args_config).copy()
    merged.update({k: v for k, v in vars(args_command).items() if v is not None})
    return Namespace(**merged)
