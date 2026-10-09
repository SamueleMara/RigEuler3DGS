"""Linear-in-parameters 1-DoF joint dynamics and its regressor.

Reference: CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md sections 32.1-32.4
(switch defaults, prismatic/revolute dynamics, parameter estimation).

Model actually implemented (all loads on the left, actuation on the right):

    prismatic   m   * qddot + b * qdot + m_g * gamma(t) + f_c   * s(qdot) = u
    revolute    J_e * qddot + b * qdot + m_g * gamma(t) + tau_c * s(qdot) = u

with `gamma` the known kinematic gravity basis (see `physics.wrench`) and
`s(qdot)` the Coulomb direction basis.  Spec 32.1 default switch positions:

    viscous damping          ON
    estimated viscous b      ON
    gravity                  OFF   (implemented, flag-gated)
    Coulomb/static friction  OFF   (implemented, flag-gated)

A disabled term is *removed from the regressor and from the Jacobian* -- it is
never a zeroed column.  That is not cosmetic: a zero column makes the normal
matrix singular, which would make every identification look rank-deficient and
would destroy the conditioning diagnostics required by spec 32.4.

This module is deliberately free of autograd and of any optimizer (spec 0.4/0.5
and section 36): everything here is closed form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

PRISMATIC = "prismatic"
REVOLUTE = "revolute"
JOINT_TYPES = (PRISMATIC, REVOLUTE)

# Standard gravity along -Y; callers in a Z-up scene pass their own vector.
DEFAULT_GRAVITY = (0.0, -9.80665, 0.0)


def canonical_joint_type(joint_type: str) -> str:
    kind = str(joint_type).lower().strip()
    if kind not in JOINT_TYPES:
        raise ValueError(
            f"unsupported joint type {joint_type!r}; spec rule 0.13 restricts the baseline to {JOINT_TYPES}"
        )
    return kind


@dataclass
class PhysicsConfig:
    """Switches for the force-enabled prediction path (spec 32.1 / section 33 flags).

    `use_wrench_dynamics` gates the whole force path and is OFF by default because
    spec 32 opens with "the force/wrench path must be optional; the method must
    work with RGB-D only".  The switches *inside* that path follow 32.1 exactly.
    """

    use_wrench_dynamics: bool = False
    use_viscous_damping: bool = True
    estimate_viscous_b: bool = True
    use_gravity: bool = False
    use_coulomb_friction: bool = False

    # Used only when damping is on but `estimate_viscous_b` is off: b is then a
    # known load and moves to the regression target instead of the regressor.
    viscous_b: float = 0.0
    gravity_vector: tuple = DEFAULT_GRAVITY
    # 0 -> exact sign(). A positive value selects tanh(qdot/eps), which is what the
    # joint filter needs when it wants a differentiable Coulomb term; the exact
    # sign is kept as the default because it is the physically correct basis.
    coulomb_smoothing_eps: float = 0.0
    # m > 0 / J_e > 0 are strict in the spec but a box QP needs a closed set, so
    # strict positivity is realised as a tiny positive floor.
    min_inertia: float = 1.0e-9

    @classmethod
    def from_namespace(cls, args) -> "PhysicsConfig":
        """Build from an argparse namespace using the section-33 flag names."""
        defaults = cls()
        return cls(
            use_wrench_dynamics=bool(getattr(args, "use_wrench_dynamics", defaults.use_wrench_dynamics)),
            use_viscous_damping=bool(getattr(args, "use_viscous_damping", defaults.use_viscous_damping)),
            estimate_viscous_b=bool(getattr(args, "estimate_viscous_b", defaults.estimate_viscous_b)),
            use_gravity=bool(getattr(args, "use_gravity", defaults.use_gravity)),
            use_coulomb_friction=bool(getattr(args, "use_coulomb_friction", defaults.use_coulomb_friction)),
            viscous_b=float(getattr(args, "viscous_b", defaults.viscous_b)),
            gravity_vector=tuple(getattr(args, "gravity_vector", defaults.gravity_vector)),
            coulomb_smoothing_eps=float(
                getattr(args, "coulomb_smoothing_eps", defaults.coulomb_smoothing_eps)
            ),
            min_inertia=float(getattr(args, "min_inertia", defaults.min_inertia)),
        )


def inertia_name(joint_type: str) -> str:
    return "m" if canonical_joint_type(joint_type) == PRISMATIC else "J_e"


def coulomb_name(joint_type: str) -> str:
    """Prismatic Coulomb term is a force, revolute is a torque -- keep them distinct."""
    return "f_c" if canonical_joint_type(joint_type) == PRISMATIC else "tau_c"


def parameter_names(config: PhysicsConfig, joint_type: str) -> list[str]:
    """Identified parameters, in regressor column order.

    Single source of truth: the regressor, the bounds, the Jacobians and
    `predict_acceleration` all index through this list, so a flag can never
    desynchronise a column from its meaning.
    """
    names = [inertia_name(joint_type)]
    if config.use_viscous_damping and config.estimate_viscous_b:
        names.append("b")
    if config.use_gravity:
        names.append("m_g")
    if config.use_coulomb_friction:
        names.append(coulomb_name(joint_type))
    return names


def parameter_bounds(
    config: PhysicsConfig,
    joint_type: str,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Box constraints of spec 32.4: m > 0, J_e > 0, b >= 0 (plus m_g, f_c >= 0)."""
    names = parameter_names(config, joint_type)
    lower, upper = [], []
    for name in names:
        if name in ("m", "J_e"):
            lower.append(config.min_inertia)  # strict positivity as a closed box
        else:
            lower.append(0.0)  # viscous b, gravitational mass and Coulomb are dissipative/positive
        upper.append(math.inf)
    kwargs = {"dtype": dtype, "device": device}
    return torch.tensor(lower, **kwargs), torch.tensor(upper, **kwargs)


def coulomb_basis(qdot: Tensor, config: PhysicsConfig) -> Tensor:
    """sign(qdot), or tanh(qdot/eps) when smoothing is requested.

    torch.sign(0) = 0 on purpose: the baseline models kinetic Coulomb friction
    only, and stiction at qdot = 0 is a set-valued force that a least-squares
    regressor cannot represent.  Spec 32.1 keeps static friction out of the
    default model, so the zero-velocity row simply carries no friction column
    contribution rather than a fabricated one.
    """
    eps = float(config.coulomb_smoothing_eps)
    if eps > 0.0:
        return torch.tanh(qdot / eps)
    return torch.sign(qdot)


def _check_rates(qddot: Tensor, qdot: Tensor) -> None:
    if qddot.shape != qdot.shape:
        raise ValueError(f"qddot {tuple(qddot.shape)} and qdot {tuple(qdot.shape)} must have the same shape")
    if not bool(torch.isfinite(qddot).all()) or not bool(torch.isfinite(qdot).all()):
        raise ValueError("non-finite qddot/qdot: refusing to build a regressor from corrupt rates")


def _gravity_column(gravity_basis, qdot: Tensor) -> Tensor:
    if gravity_basis is None:
        raise ValueError(
            "use_gravity=True requires `gravity_basis` (physics.wrench.gravity_basis_*): "
            "the gravity column is kinematics the estimator cannot invent"
        )
    column = torch.as_tensor(gravity_basis, dtype=qdot.dtype, device=qdot.device)
    return column.expand_as(qdot) if column.dim() == 0 or column.shape != qdot.shape else column


def build_regressor(
    qddot: Tensor,
    qdot: Tensor,
    config: PhysicsConfig,
    joint_type: str,
    *,
    gravity_basis: Tensor | float | None = None,
) -> tuple[Tensor, list[str]]:
    """Phi such that Phi @ theta = u - (known loads), with column names.

    Columns (spec 32.4), in this order and only when enabled:
        [qddot]                     -> m or J_e            (always)
        [qdot]                      -> b                   (viscous ON and estimated)
        [gamma]                     -> m_g                 (gravity ON)
        [sign(qdot)]                -> f_c / tau_c         (Coulomb ON)
    """
    joint_type = canonical_joint_type(joint_type)
    _check_rates(qddot, qdot)
    columns = [qddot]
    if config.use_viscous_damping and config.estimate_viscous_b:
        columns.append(qdot)
    if config.use_gravity:
        columns.append(_gravity_column(gravity_basis, qdot))
    if config.use_coulomb_friction:
        columns.append(coulomb_basis(qdot, config))
    phi = torch.stack(columns, dim=-1)
    return phi, parameter_names(config, joint_type)


def known_load(qdot: Tensor, config: PhysicsConfig, joint_type: str) -> Tensor:
    """Generalized force explained by terms whose coefficient is *not* identified.

    Today that is only a pinned viscous b (damping ON, estimation OFF).  It is
    subtracted from u before regression so that the fixed term still acts on the
    physics while staying out of the parameter vector.
    """
    canonical_joint_type(joint_type)
    load = torch.zeros_like(qdot)
    if config.use_viscous_damping and not config.estimate_viscous_b:
        load = load + float(config.viscous_b) * qdot
    return load


def regression_target(u: Tensor, qdot: Tensor, config: PhysicsConfig, joint_type: str) -> Tensor:
    """u minus the known loads -- the right-hand side that pairs with `build_regressor`."""
    u = torch.as_tensor(u, dtype=qdot.dtype, device=qdot.device)
    return u - known_load(qdot, config, joint_type)


def _unpack(params: Tensor, config: PhysicsConfig, joint_type: str) -> dict[str, Tensor]:
    names = parameter_names(config, joint_type)
    params = torch.as_tensor(params)
    if params.shape[-1] != len(names):
        raise ValueError(
            f"expected {len(names)} parameters {names} for this config, got {params.shape[-1]}"
        )
    return {name: params[..., i] for i, name in enumerate(names)}


def inverse_dynamics(
    params: Tensor,
    qddot: Tensor,
    qdot: Tensor,
    config: PhysicsConfig,
    joint_type: str,
    *,
    gravity_basis: Tensor | float | None = None,
) -> Tensor:
    """Forward dynamics: u required to produce (qddot, qdot) under `params`."""
    phi, _ = build_regressor(qddot, qdot, config, joint_type, gravity_basis=gravity_basis)
    params = torch.as_tensor(params, dtype=phi.dtype, device=phi.device)
    return (phi * params).sum(dim=-1) + known_load(qdot, config, joint_type)


def predict_acceleration(
    params: Tensor,
    qdot: Tensor,
    u: Tensor,
    config: PhysicsConfig,
    joint_type: str,
    *,
    gravity_basis: Tensor | float | None = None,
) -> Tensor:
    """qddot = (u - b qdot - m_g gamma - f_c s(qdot) - known loads) / inertia.

    This is the mechanical replacement for constant-velocity prediction (spec 32):
    the renderer still supplies the correction, this supplies the propagation.
    """
    joint_type = canonical_joint_type(joint_type)
    qdot = torch.as_tensor(qdot)
    u = torch.as_tensor(u, dtype=qdot.dtype, device=qdot.device)
    theta = _unpack(torch.as_tensor(params, dtype=qdot.dtype, device=qdot.device), config, joint_type)

    resistive = known_load(qdot, config, joint_type)
    if config.use_viscous_damping and config.estimate_viscous_b:
        resistive = resistive + theta["b"] * qdot
    if config.use_gravity:
        resistive = resistive + theta["m_g"] * _gravity_column(gravity_basis, qdot)
    if config.use_coulomb_friction:
        resistive = resistive + theta[coulomb_name(joint_type)] * coulomb_basis(qdot, config)

    # A non-positive inertia is unphysical and would flip the sign of the whole
    # prediction; the identification already clamps it, this is the last guard.
    inertia = torch.clamp(theta[inertia_name(joint_type)], min=float(config.min_inertia))
    return (u - resistive) / inertia


def acceleration_parameter_jacobian(
    params: Tensor,
    qdot: Tensor,
    u: Tensor,
    config: PhysicsConfig,
    joint_type: str,
    *,
    gravity_basis: Tensor | float | None = None,
) -> tuple[Tensor, list[str]]:
    """d qddot / d theta, columns aligned with `parameter_names`.

    Differentiating Phi(qddot, qdot) theta = u - known at fixed (u, qdot) gives
    d qddot/d theta = -Phi(qddot(theta), qdot) / inertia, i.e. the SAME column set
    as the regressor -- so a disabled term is absent from the Jacobian too, which
    is what spec 32.1 demands.
    """
    joint_type = canonical_joint_type(joint_type)
    qddot = predict_acceleration(params, qdot, u, config, joint_type, gravity_basis=gravity_basis)
    phi, names = build_regressor(qddot, qdot, config, joint_type, gravity_basis=gravity_basis)
    theta = _unpack(torch.as_tensor(params, dtype=phi.dtype, device=phi.device), config, joint_type)
    inertia = torch.clamp(theta[inertia_name(joint_type)], min=float(config.min_inertia))
    return -phi / inertia.unsqueeze(-1), names


class StateJacobian(NamedTuple):
    """Linearization of the mechanical prediction w.r.t. the joint state."""

    d_dq: Tensor
    d_dqdot: Tensor


def acceleration_state_jacobian(
    params: Tensor,
    qdot: Tensor,
    config: PhysicsConfig,
    joint_type: str,
    *,
    dgravity_dq: Tensor | float | None = None,
) -> StateJacobian:
    """(d qddot/dq, d qddot/d qdot) for the joint-filter process model.

    d qddot/dq is nonzero only through the gravity basis gamma(q) of a revolute
    joint, so the caller supplies gamma'(q); with gravity off it is exactly zero.
    With the exact sign() basis the Coulomb term contributes 0 almost everywhere
    (its jump at qdot = 0 is not a derivative); the smoothed basis contributes
    the real tanh slope.
    """
    joint_type = canonical_joint_type(joint_type)
    qdot = torch.as_tensor(qdot)
    theta = _unpack(torch.as_tensor(params, dtype=qdot.dtype, device=qdot.device), config, joint_type)
    inertia = torch.clamp(theta[inertia_name(joint_type)], min=float(config.min_inertia))

    damping = torch.zeros_like(qdot)
    if config.use_viscous_damping:
        damping = damping + (theta["b"] if config.estimate_viscous_b else float(config.viscous_b))
    if config.use_coulomb_friction and float(config.coulomb_smoothing_eps) > 0.0:
        eps = float(config.coulomb_smoothing_eps)
        damping = damping + theta[coulomb_name(joint_type)] * (1.0 - torch.tanh(qdot / eps) ** 2) / eps

    d_dq = torch.zeros_like(qdot)
    if config.use_gravity:
        if dgravity_dq is None:
            raise ValueError("use_gravity=True requires `dgravity_dq` to linearize d qddot/dq")
        d_dq = -theta["m_g"] * torch.as_tensor(
            dgravity_dq, dtype=qdot.dtype, device=qdot.device
        ).expand_as(qdot) / inertia
    return StateJacobian(d_dq=d_dq, d_dqdot=-damping / inertia)


class Rates(NamedTuple):
    """Finite-difference joint rates plus the mask of trustworthy samples."""

    qdot: Tensor
    qddot: Tensor
    valid: Tensor


def central_difference_rates(q: Tensor, t: Tensor | float) -> Rates:
    """qdot/qddot from a sampled joint trajectory on a REAL time grid.

    `t` is either a scalar frame spacing (seconds) or the per-frame timestamps,
    which may be non-uniform -- spec rule 0.11 forbids assuming unit frame
    spacing, and dropped/retimed frames make the uniform formula wrong, so the
    general unequal-spacing stencils are used:

        f'(t_i)  = -h2/(h1(h1+h2)) f_{i-1} + (h2-h1)/(h1 h2) f_i + h1/(h2(h1+h2)) f_{i+1}
        f''(t_i) = 2[ f_{i-1}/(h1(h1+h2)) - f_i/(h1 h2) + f_{i+1}/(h2(h1+h2)) ]

    Endpoints get a one-sided velocity and the neighbouring acceleration, and are
    reported as invalid so the identification can drop them instead of absorbing
    a first-order biased sample.
    """
    q = torch.as_tensor(q)
    if q.dim() != 1 or q.numel() < 3:
        raise ValueError("central differences need a 1-D trajectory with at least 3 samples")
    if isinstance(t, Tensor) and t.dim() == 1:
        times = t.to(dtype=q.dtype, device=q.device)
        if times.shape != q.shape:
            raise ValueError("timestamp vector must match the trajectory length")
    else:
        dt = float(t)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"frame spacing must be a positive finite duration, got {t!r}")
        times = torch.arange(q.numel(), dtype=q.dtype, device=q.device) * dt
    h = times[1:] - times[:-1]
    if bool(torch.any(h <= 0)):
        raise ValueError("timestamps must be strictly increasing")

    h1, h2 = h[:-1], h[1:]
    qm, q0, qp = q[:-2], q[1:-1], q[2:]
    interior_dot = (
        -h2 / (h1 * (h1 + h2)) * qm + (h2 - h1) / (h1 * h2) * q0 + h1 / (h2 * (h1 + h2)) * qp
    )
    interior_ddot = 2.0 * (qm / (h1 * (h1 + h2)) - q0 / (h1 * h2) + qp / (h2 * (h1 + h2)))

    qdot = torch.empty_like(q)
    qddot = torch.empty_like(q)
    qdot[1:-1], qddot[1:-1] = interior_dot, interior_ddot
    qdot[0], qdot[-1] = (q[1] - q[0]) / h[0], (q[-1] - q[-2]) / h[-1]
    qddot[0], qddot[-1] = interior_ddot[0], interior_ddot[-1]
    valid = torch.zeros_like(q, dtype=torch.bool)
    valid[1:-1] = True
    return Rates(qdot=qdot, qddot=qddot, valid=valid)
