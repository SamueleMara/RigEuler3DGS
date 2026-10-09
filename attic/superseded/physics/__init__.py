"""Force-enabled physical prediction for identified 1-DoF joints.

Implements CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md section 32 (and the
`physics/` package of section 23):

    wrench.py          generalized-force projection of a wrench onto a joint axis
    viscous_models.py  linear-in-parameters prismatic/revolute dynamics + regressor
    rls.py             box-constrained WLS and recursive LS with conditioning report

Default model switches (spec 0.12 / 32.1): viscous damping ON, viscous b
estimated, gravity OFF, Coulomb/static friction OFF.  Disabled terms are
excluded from the regressor and Jacobian, not zeroed.

The package is standalone by design -- it takes plain tensors and floats, imports
nothing from the estimator or joint-reduction packages, and contains no autograd
or optimizer code (spec rules 0.4/0.5, section 36).
"""

from __future__ import annotations

from .rls import (
    RecursiveLeastSquares,
    RLSUpdate,
    WLSResult,
    constrained_wls,
)
from .viscous_models import (
    DEFAULT_GRAVITY,
    JOINT_TYPES,
    PRISMATIC,
    REVOLUTE,
    PhysicsConfig,
    Rates,
    StateJacobian,
    acceleration_parameter_jacobian,
    acceleration_state_jacobian,
    build_regressor,
    canonical_joint_type,
    central_difference_rates,
    coulomb_basis,
    coulomb_name,
    inertia_name,
    inverse_dynamics,
    known_load,
    parameter_bounds,
    parameter_names,
    predict_acceleration,
    regression_target,
)
from .wrench import (
    generalized_force,
    gravity_basis_prismatic,
    gravity_basis_revolute,
    prismatic_generalized_force,
    revolute_generalized_force,
    rotate_about_axis,
    unit_axis,
)

__all__ = [
    "DEFAULT_GRAVITY",
    "JOINT_TYPES",
    "PRISMATIC",
    "REVOLUTE",
    "PhysicsConfig",
    "Rates",
    "RecursiveLeastSquares",
    "RLSUpdate",
    "StateJacobian",
    "WLSResult",
    "acceleration_parameter_jacobian",
    "acceleration_state_jacobian",
    "build_regressor",
    "canonical_joint_type",
    "central_difference_rates",
    "constrained_wls",
    "coulomb_basis",
    "coulomb_name",
    "generalized_force",
    "gravity_basis_prismatic",
    "gravity_basis_revolute",
    "inertia_name",
    "inverse_dynamics",
    "known_load",
    "parameter_bounds",
    "parameter_names",
    "predict_acceleration",
    "prismatic_generalized_force",
    "regression_target",
    "revolute_generalized_force",
    "rotate_about_axis",
    "unit_axis",
]
