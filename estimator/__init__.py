"""Sparse alpha-aware error-state IEKF for dynamic 3D Gaussian Splatting.

Specification: ``CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md``.

This file intentionally re-exports only the manifold and state layer (spec
section 4).  The remaining modules named in section 23 -- ``process``,
``covariance``, ``measurement``, ``pcg``, ``sparse_iekf``, ``diagnostics`` --
are added to this same package separately; importing them from here before
they land would make ``import estimator`` fail for everyone, so new names are
appended only together with the module that defines them.
"""

from __future__ import annotations

from .manifold import (
    is_rotation,
    project_to_so3,
    rotation_angle,
    small_angle_cutoff_sq,
    so3_adjoint,
    so3_exp,
    so3_hat,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
    so3_log,
    so3_vee,
)
from .state import (
    DELTA_DIM,
    DELTA_SLICE_MU,
    DELTA_SLICE_OMEGA,
    DELTA_SLICE_PHI,
    DELTA_SLICE_V,
    STATE_DICT_VERSION,
    GaussianMotionState,
    boxminus,
    boxplus,
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
    rotation_validation_enabled,
    set_rotation_validation,
    split_delta,
    stack_delta,
)

__all__ = [
    # manifold (section 4)
    "so3_exp",
    "so3_log",
    "so3_hat",
    "so3_vee",
    "so3_left_jacobian",
    "so3_left_jacobian_inverse",
    "so3_adjoint",
    "is_rotation",
    "project_to_so3",
    "rotation_angle",
    "small_angle_cutoff_sq",
    # state (sections 3, 4, 23)
    "GaussianMotionState",
    "boxplus",
    "boxminus",
    "split_delta",
    "stack_delta",
    "quaternion_wxyz_to_matrix",
    "matrix_to_quaternion_wxyz",
    "set_rotation_validation",
    "rotation_validation_enabled",
    "DELTA_DIM",
    "DELTA_SLICE_MU",
    "DELTA_SLICE_PHI",
    "DELTA_SLICE_V",
    "DELTA_SLICE_OMEGA",
    "STATE_DICT_VERSION",
]

# ---------------------------------------------------------------------------
# Estimator runtime (spec sections 5, 6, 13-17, 19.1, 22, 24, 25). Imported
# last so a partially built package still imports the manifold/state layer.
# ---------------------------------------------------------------------------
from .convergence import (  # noqa: E402
    ConvergenceCertificate,
    IterationRecord,
    quadratic_model_decrease,
)
from .jacobian_audit import ChainAudit, audit_chain, chain_forward  # noqa: E402
from .covariance import (  # noqa: E402
    apply_prior,
    approximate_posterior_information,
    covariance_predict,
    invert_blocks,
    make_process_noise,
)
from .measurement import (  # noqa: E402
    build_precision,
    measurement_energy,
    normalized_innovation_stats,
    rgbd_residual,
)
from .pcg import block_jacobi_preconditioner, pcg  # noqa: E402
from .process import error_state_transition, process_predict  # noqa: E402
from .renderer_bridge import (  # noqa: E402
    CameraSpec,
    FrameLinearization,
    build_linearization,
    embed_pose_to_full_state,
    extract_pose,
)
from .sparse_iekf import (  # noqa: E402
    IEKFConfig,
    config_from_args,
    IEKFDiagnostics,
    evaluate_true_map_energy,
    iekf_frame_step,
)
from .pipeline import PipelineResult, run_sequence  # noqa: E402
