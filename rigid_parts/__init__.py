"""Nested part-embedding outer loop + differential Killing-field inner loop.

Two nested questions, kept strictly apart (§8, §39):

    inner: given memberships, what K rigid Killing motions explain the video?
    outer: given those motions, which memberships make the final geometry,
           COM and inertia physically correct?

Gradients never cross. The inner solve trains only Killing-field parameters
with embeddings frozen; the outer solve trains only embeddings with every
integrated transform frozen.

``rigid_fit`` is exported for DIAGNOSTICS ONLY. §40 forbids Kabsch anywhere in
training, and `tests/test_nested_killing_part_discovery.py` monkeypatches it
to raise so that the prohibition is enforced rather than merely intended.
"""

from rigid_parts.embeddings import PartEmbeddingModel
from rigid_parts.static_anchor_init import (
    StaticAnchorResult,
    extract_confident_static_anchors,
    extract_static_anchors_from_ply,
)
from rigid_parts.killing_inner_solver import (
    NestedKillingMotionState,
    solve_nested_killing_sequence,
)
from rigid_parts.outer_embedding_solver import (
    build_outer_targets,
    candidate_final_positions,
    candidate_nn_energies,
    candidates_from_local_endpoints,
    final_target_part_assignment,
    geometry_guard_value,
    outer_embedding_losses,
    soft_compose_candidates,
)
from rigid_parts.neighborhood import (
    NeighborGraph,
    build_neighbor_graph,
    edge_part_weights,
    fit_local_jacobian,
    local_coherence_loss,
)
from rigid_parts.local_killing import (
    confidence_gate,
    effective_obs_lambda,
    evaluate_local_field,
    field_separation_loss,
    integrate_local_se3,
    local_killing_quantities,
    observed_velocity_loss,
)
from rigid_parts.inner_acceptance import (
    InnerAcceptance,
    endpoint_usefulness,
    evaluate_inner_solve,
)
from rigid_parts.moments import (
    compute_global_com,
    compute_global_inertia,
    compute_soft_com,
    compute_soft_inertia,
    compute_soft_mass,
)
from rigid_parts.endpoint import load_ply_xyz
from rigid_parts.nested_pipeline import (
    adaptive_temperature,
    run_nested_part_embedding_killing_discovery,
)

__all__ = [
    "PartEmbeddingModel",
    "StaticAnchorResult",
    "extract_confident_static_anchors",
    "extract_static_anchors_from_ply",
    "NestedKillingMotionState",
    "solve_nested_killing_sequence",
    "build_outer_targets",
    "candidate_final_positions",
    "candidate_nn_energies",
    "candidates_from_local_endpoints",
    "final_target_part_assignment",
    "geometry_guard_value",
    "outer_embedding_losses",
    "soft_compose_candidates",
    "NeighborGraph",
    "build_neighbor_graph",
    "edge_part_weights",
    "fit_local_jacobian",
    "local_coherence_loss",
    "confidence_gate",
    "effective_obs_lambda",
    "evaluate_local_field",
    "field_separation_loss",
    "integrate_local_se3",
    "local_killing_quantities",
    "observed_velocity_loss",
    "InnerAcceptance",
    "endpoint_usefulness",
    "evaluate_inner_solve",
    "adaptive_temperature",
    "compute_global_com",
    "compute_global_inertia",
    "compute_soft_com",
    "compute_soft_inertia",
    "compute_soft_mass",
    "load_ply_xyz",
    "run_nested_part_embedding_killing_discovery",
]
