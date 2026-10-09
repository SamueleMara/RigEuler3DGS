"""Generalized-force (wrench) projection for 1-DoF joints.

Reference: CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md sections 32.2 and 32.3.

A 1-DoF joint only ever "feels" the component of an applied wrench along its own
screw axis.  Everything in this module is that single projection, written so it
broadcasts over a whole frame batch at once:

    prismatic   u = e^T F
    revolute    u = e^T [ tau + (p - a) x F ]

The same projection is reused for the *gravity* basis column of the regressor
(see `viscous_models.build_regressor`): gravity is just another wrench, and the
only reason it gets its own helper is that its magnitude carries the unknown
mass, so the known kinematic part has to be factored out into a regressor column.

No autograd, no optimizer: these are closed-form algebraic maps (spec section 36).
"""

from __future__ import annotations

import torch
from torch import Tensor

# Below this the axis direction is numerically meaningless; projecting onto it
# would amplify round-off into the generalized force instead of reporting a
# degenerate joint, so we refuse rather than silently return garbage.
_AXIS_EPS = 1.0e-12


def _coerce(value, reference: Tensor) -> Tensor:
    """Bring `value` onto the reference tensor's device/dtype without copying tensors."""
    if isinstance(value, Tensor):
        if value.dtype != reference.dtype or value.device != reference.device:
            return value.to(device=reference.device, dtype=reference.dtype)
        return value
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _reference(*values) -> Tensor:
    """First tensor argument wins device/dtype, so float64 callers stay in float64."""
    for value in values:
        if isinstance(value, Tensor):
            return value
    raise TypeError("at least one argument must be a torch.Tensor to fix device/dtype")


def unit_axis(e: Tensor, *, eps: float = _AXIS_EPS) -> Tensor:
    """Normalize a joint axis (last dim = 3).

    The generalized force is only equal to the projected force magnitude when the
    axis is a unit vector, so we normalize here instead of trusting the caller:
    an axis that drifted off the unit sphere during joint fitting would otherwise
    rescale the identified mass/inertia by ||e||.
    """
    if e.shape[-1] != 3:
        raise ValueError(f"joint axis must have last dimension 3, got {tuple(e.shape)}")
    norm = torch.linalg.vector_norm(e, dim=-1, keepdim=True)
    if bool(torch.any(norm <= eps)):
        raise ValueError("degenerate joint axis (near-zero norm) cannot be projected onto")
    return e / norm


def prismatic_generalized_force(e: Tensor, F: Tensor, *, normalize: bool = True) -> Tensor:
    """u = e^T F  (spec 32.2).

    `e` is the prismatic axis, `F` the applied force in the same frame.  Shapes
    broadcast on all leading dimensions, e.g. e=(3,) with F=(T, 3) -> (T,).
    """
    ref = _reference(F, e)
    F, e = _coerce(F, ref), _coerce(e, ref)
    axis = unit_axis(e) if normalize else e
    return (axis * F).sum(dim=-1)


def revolute_generalized_force(
    e: Tensor,
    a: Tensor,
    p: Tensor,
    F: Tensor,
    tau: Tensor | None = None,
    *,
    normalize: bool = True,
) -> Tensor:
    """u = e^T [ tau + (p - a) x F ]  (spec 32.3).

    `a` is any point on the revolute axis, `p` the point where `F` is applied and
    `tau` an additional pure torque (defaults to zero, i.e. force-only actuation).
    The lever arm is measured from the axis point, so the component of (p - a)
    along `e` drops out of the projection automatically -- which is exactly why a
    revolute joint is insensitive to where along the axis `a` is chosen.
    """
    ref = _reference(F, e, p, a)
    F, e, a, p = _coerce(F, ref), _coerce(e, ref), _coerce(a, ref), _coerce(p, ref)
    axis = unit_axis(e) if normalize else e
    # torch.linalg.cross does not broadcast rank, so a static lever arm (3,) has
    # to be expanded against a per-frame force batch (T, 3) explicitly.
    lever, force = torch.broadcast_tensors(p - a, F)
    moment = torch.linalg.cross(lever, force, dim=-1)
    if tau is not None:
        moment = moment + _coerce(tau, F)
    return (axis * moment).sum(dim=-1)


def generalized_force(joint_type: str, **kwargs) -> Tensor:
    """Dispatch to the prismatic/revolute projection by joint type string."""
    kind = str(joint_type).lower()
    if kind == "prismatic":
        return prismatic_generalized_force(kwargs["e"], kwargs["F"])
    if kind == "revolute":
        return revolute_generalized_force(
            kwargs["e"], kwargs["a"], kwargs["p"], kwargs["F"], kwargs.get("tau")
        )
    raise ValueError(f"unsupported joint type {joint_type!r} (spec 0.13: prismatic|revolute only)")


def rotate_about_axis(e: Tensor, q: Tensor, v: Tensor) -> Tensor:
    """Rodrigues rotation of `v` by angle `q` about unit axis `e`.

    Used to swing a revolute lever arm from its reference configuration (q = 0) to
    the configuration at time t, so the gravity regressor column can be evaluated
    without ever building a rotation matrix per frame.
    """
    axis = unit_axis(_coerce(e, v))
    q = _coerce(q, v).unsqueeze(-1)
    cos_q, sin_q = torch.cos(q), torch.sin(q)
    dot = (axis * v).sum(dim=-1, keepdim=True)
    return v * cos_q + torch.linalg.cross(axis.expand_as(v), v, dim=-1) * sin_q + axis * dot * (1.0 - cos_q)


def gravity_basis_prismatic(e: Tensor, gravity_vector: Tensor) -> Tensor:
    """Known part of the prismatic gravity load, per unit mass.

    The generalized gravity force is u_g = m * (e^T g).  We return
    gamma = -(e^T g) so that the regressor term `m_g * gamma` sits on the SAME
    side of the equation as the inertial and damping loads:

        m qddot + b qdot + m_g * gamma + f_c sign(qdot) = u

    Sign convention matters here: get it backwards and the identified mass absorbs
    the error with the wrong sign whenever gravity is enabled.
    """
    ref = _reference(e, gravity_vector)
    e, g = _coerce(e, ref), _coerce(gravity_vector, ref)
    return -(unit_axis(e) * g).sum(dim=-1)


def gravity_basis_revolute(
    e: Tensor,
    lever0: Tensor,
    q: Tensor,
    gravity_vector: Tensor,
) -> Tensor:
    """Known part of the revolute gravity torque, per unit mass.

    `lever0` is the vector from the axis point `a` to the part's centre of mass in
    the reference configuration q = 0 (available from the rigid fit).  At angle q
    the lever is R(e, q) lever0, so the gravity torque is
    m * e^T [ (R(e,q) lever0) x g ] and we return its negation, matching the
    left-hand-side convention of `gravity_basis_prismatic`.

    This keeps ONE gravity column (one unknown: the gravitational mass).  The
    two-column cos/sin parameterisation would also identify the unknown lever
    direction, which spec 32.3 explicitly defers out of the baseline.
    """
    ref = _reference(lever0, e, q, gravity_vector)
    lever0, e = _coerce(lever0, ref), _coerce(e, ref)
    g, q = _coerce(gravity_vector, ref), _coerce(q, ref)
    lever = rotate_about_axis(e, q, lever0.expand(tuple(q.shape) + (3,)) if q.dim() > 0 else lever0)
    return -(unit_axis(e) * torch.linalg.cross(lever, g.expand_as(lever), dim=-1)).sum(dim=-1)
