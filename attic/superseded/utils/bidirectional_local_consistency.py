"""Continuous forward/backward consistency between two local motion fields.

A globally reversible transform is the wrong test for a scene with several
moving parts: one global ``T_T0 ~= T_0T^-1`` check averages incompatible
motions.  The reversibility condition here is *local* -- a forward local
transform and the backward local transform found around the position it
transports to should approximately invert each other -- which holds for any
number of rigid parts, and holds separately inside each of them.

Nothing here requires that a frame-0 Gaussian correspond to any particular
frame-T Gaussian.  The backward field is queried at a continuous transported
location, not at a matched target index.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.local_rigid_field import (
    LocalDeformationResult,
    fit_local_deformation_at,
    local_scale_from_knn,
    query_knn,
)


@dataclass
class LocalInverseConsistency:
    """Forward/backward agreement, per frame-0 Gaussian.

    ``reverse_valid`` says only whether the *reverse* query succeeded.  It is
    deliberately NOT combined with forward validity: a point whose backward
    neighbourhood is unusable has missing reverse evidence, which is not the
    same thing as being intrinsically ambiguous, and must never be allowed to
    override strong forward/local evidence.
    """

    cycle_error: torch.Tensor              # [N] ||x_hat - x|| / effective scale
    rotation_cycle_error: torch.Tensor     # [N] ||R_B R_F - I||_F^2
    jacobian_cycle_error: torch.Tensor     # [N] ||J_B J_F - I||_F^2
    #: Debug only -- badly scaled in practice (median ~6, p90 ~125 on real
    #: data) and destructive while the backward FM is unreliable.  It is kept
    #: for analysis and must not enter the label update.
    translation_cycle_error: torch.Tensor  # [N] ||R_B t_F + t_B||^2 / scale^2
    #: Is the backward field, evaluated where the forward map transports to,
    #: the inverse of the forward motion?  Perfect inversion gives ratio 1 and
    #: cosine 1; these separate "wrong magnitude" from "wrong direction".
    reverse_magnitude_ratio: torch.Tensor  # [N] ||u_B@F|| / ||u_F||
    reverse_direction_cosine: torch.Tensor # [N] cos(-u_F, u_B@F)
    transported_xyz: torch.Tensor          # [N, 3] z_hat, forward-transported
    returned_xyz: torch.Tensor             # [N, 3] x_hat, after the backward map
    reverse_valid: torch.Tensor            # [N] bool -- reverse query usable


@torch.no_grad()
def query_local_backward_field(
    query_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    target_displacement: torch.Tensor,
    *,
    knn: int = 16,
    target_confidence: torch.Tensor | None = None,
    chunk_size: int = 2048,
    **fit_kwargs,
) -> tuple[LocalDeformationResult, torch.Tensor, torch.Tensor]:
    """Fit the backward local field around arbitrary continuous locations.

    Returns the local fit plus the neighbour indices/distances used, so a
    caller can reuse them (for example to interpolate instead of fitting).
    """
    idx, dist = query_knn(query_xyz, target_xyz, int(knn), chunk_size=int(chunk_size))
    target_xyz = torch.as_tensor(target_xyz).detach().to(query_xyz)
    target_displacement = torch.as_tensor(target_displacement).detach().to(query_xyz)
    neighbor_xyz = target_xyz[idx]
    neighbor_disp = target_displacement[idx]
    neighbor_conf = None
    if target_confidence is not None:
        neighbor_conf = torch.as_tensor(target_confidence).detach().to(query_xyz)[idx]
    local = fit_local_deformation_at(
        neighbor_xyz,
        neighbor_disp,
        center_xyz=query_xyz,
        neighbor_distance=dist,
        neighbor_confidence=neighbor_conf,
        **fit_kwargs,
    )
    return local, idx, dist


@torch.no_grad()
def evaluate_local_inverse_consistency(
    xyz0: torch.Tensor,
    forward_local: LocalDeformationResult,
    xyzT: torch.Tensor,
    backward_displacement: torch.Tensor,
    *,
    local_scale: torch.Tensor | None = None,
    backward_confidence: torch.Tensor | None = None,
    knn: int = 16,
    mode: str = "transform",
    chunk_size: int = 2048,
    max_normalized_error: float = 1.0e3,
    **fit_kwargs,
) -> tuple[LocalInverseConsistency, LocalDeformationResult | None]:
    """Transport forward, come back through the backward field, compare.

    ``mode='transform'`` (option B) fits a backward local transform around
    the transported location and is preferred: it yields the rotation and
    translation inverse diagnostics as well as the positional cycle.
    ``mode='interpolate'`` (option A) only interpolates the backward
    displacement and yields the positional cycle alone.
    """
    if str(mode) not in {"transform", "interpolate"}:
        raise ValueError("mode must be 'transform' or 'interpolate'")
    xyz0 = torch.as_tensor(xyz0).detach()
    device, dtype = xyz0.device, xyz0.dtype
    n = int(xyz0.shape[0])

    # z_hat = R_F x + t_F -- the forward local map applied to the centre.
    transported = xyz0 + forward_local.predicted_disp.to(xyz0)

    if local_scale is None:
        _idx, dist = query_knn(xyz0, xyz0, min(9, max(2, n)), chunk_size=int(chunk_size))
        local_scale = local_scale_from_knn(dist[:, 1:] if dist.shape[1] > 1 else dist)
    local_scale = local_scale.to(device=device, dtype=dtype).reshape(-1).clamp_min(1.0e-6)

    backward_local: LocalDeformationResult | None = None
    if str(mode) == "transform":
        backward_local, _idx, _dist = query_local_backward_field(
            transported,
            xyzT,
            backward_displacement,
            knn=int(knn),
            target_confidence=backward_confidence,
            chunk_size=int(chunk_size),
            local_scale=local_scale,
            **fit_kwargs,
        )
        returned = torch.einsum("nab,nb->na", backward_local.R, transported) + backward_local.t
        RB_RF = torch.einsum("nab,nbc->nac", backward_local.R, forward_local.R.to(device))
        eye = torch.eye(3, device=device, dtype=dtype).expand_as(RB_RF)
        rotation_cycle = (RB_RF - eye).square().sum(dim=(-1, -2))
        trans_cycle = (
            torch.einsum("nab,nb->na", backward_local.R, forward_local.t.to(device)) + backward_local.t
        ).square().sum(dim=-1) / local_scale.square().clamp_min(1.0e-12)
        JB_JF = torch.einsum("nab,nbc->nac", backward_local.J, forward_local.J.to(device))
        jacobian_cycle = (JB_JF - eye).square().sum(dim=(-1, -2))
        backward_valid = backward_local.valid
    else:
        idx, dist = query_knn(transported, xyzT, int(knn), chunk_size=int(chunk_size))
        target_disp = torch.as_tensor(backward_displacement).detach().to(xyz0)
        sigma = (2.0 * local_scale).clamp_min(1.0e-6)
        alpha = torch.exp(-0.5 * dist.square() / sigma[:, None].square())
        if backward_confidence is not None:
            alpha = alpha * torch.as_tensor(backward_confidence).detach().to(xyz0).clamp(0.0, 1.0)[idx]
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        returned = transported + (alpha[..., None] * target_disp[idx]).sum(dim=1)
        zeros = torch.zeros(n, device=device, dtype=dtype)
        rotation_cycle = zeros.clone()
        trans_cycle = zeros.clone()
        jacobian_cycle = zeros.clone()
        backward_valid = (alpha.sum(dim=1) > 0) & torch.isfinite(returned).all(-1)

    cycle_error = torch.linalg.norm(returned - xyz0, dim=-1) / local_scale.clamp_min(1.0e-8)

    # Direction vs magnitude: a backward field that points the right way but
    # under-shoots is a calibration problem; one that points anywhere is a
    # coupling problem.  Separating them is what makes the backward branch
    # debuggable instead of just "bad".
    forward_motion = transported - xyz0
    reverse_motion = returned - transported
    forward_norm = torch.linalg.norm(forward_motion, dim=-1)
    reverse_norm = torch.linalg.norm(reverse_motion, dim=-1)
    eps = 1.0e-12
    magnitude_ratio = reverse_norm / forward_norm.clamp_min(eps)
    direction_cosine = (-forward_motion * reverse_motion).sum(dim=-1) / (
        forward_norm * reverse_norm
    ).clamp_min(eps)

    # Reverse validity ONLY. Forward validity is the caller's business.
    reverse_valid = (
        backward_valid
        & torch.isfinite(cycle_error)
        & torch.isfinite(rotation_cycle)
        & torch.isfinite(jacobian_cycle)
    )
    cap = float(max_normalized_error)
    zeros = torch.zeros(n, device=device, dtype=dtype)
    consistency = LocalInverseConsistency(
        cycle_error=torch.where(reverse_valid, cycle_error.clamp(max=cap), zeros),
        rotation_cycle_error=torch.where(reverse_valid, rotation_cycle.clamp(max=cap), zeros),
        jacobian_cycle_error=torch.where(reverse_valid, jacobian_cycle.clamp(max=cap), zeros),
        translation_cycle_error=torch.where(reverse_valid, trans_cycle.clamp(max=cap), zeros),
        reverse_magnitude_ratio=torch.where(reverse_valid, magnitude_ratio.clamp(max=cap), zeros),
        reverse_direction_cosine=torch.where(reverse_valid, direction_cosine.clamp(-1.0, 1.0), zeros),
        transported_xyz=transported,
        returned_xyz=returned,
        reverse_valid=reverse_valid,
    )
    return consistency, backward_local
