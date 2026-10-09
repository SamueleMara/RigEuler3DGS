"""R8 -- 3D motion implies 2D flow analytically, through the rasteriser.

    u_dot_ci = J_pi(X_ci) R_c v_i^3D
    v_cp     = sum_i Omega_cpi u_dot_ci

§23: this is why the learned motion object is 3D and not 2D.  A separate 2D
flow network would have to be kept consistent with the 3D field by yet another
weighted loss; going through the projection Jacobian and the rasteriser makes
the consistency structural, because the 2D flow is not a second model at all --
it is the 3D field seen through a camera.
"""

from __future__ import annotations

import torch

from utils.rasterizer_ownership import splat_gaussian_features

EPS = 1.0e-8


@torch.no_grad()
def project_gaussian_velocity_to_image_flow(
    camera,
    xyz: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    """Per-Gaussian image-space velocity in pixels, ``[N, 2]``.

    The perspective Jacobian is applied in camera coordinates:
    ``d(u, v)/dX = f/z * [[1, 0, -x/z], [0, 1, -y/z]]``.  Using the full
    Jacobian rather than a small-motion approximation matters for Gaussians
    near the image edge, where the ``-x/z`` term dominates.
    """
    xyz = torch.as_tensor(xyz).detach().float()
    velocity = torch.as_tensor(velocity).detach().float().to(xyz.device)
    view = camera.world_view_transform.to(xyz.device).float()

    rotation = view[:3, :3].T                       # world -> camera
    camera_xyz = xyz @ rotation.T + view[3, :3]
    camera_velocity = velocity @ rotation.T

    focal_x = float(camera.image_width) / (2.0 * float(torch.tan(torch.tensor(camera.FoVx * 0.5))))
    focal_y = float(camera.image_height) / (2.0 * float(torch.tan(torch.tensor(camera.FoVy * 0.5))))
    z = camera_xyz[:, 2].clamp_min(1.0e-4)

    du = focal_x * (camera_velocity[:, 0] / z - camera_xyz[:, 0] * camera_velocity[:, 2] / z.square())
    dv = focal_y * (camera_velocity[:, 1] / z - camera_xyz[:, 1] * camera_velocity[:, 2] / z.square())
    behind = camera_xyz[:, 2] <= 1.0e-4
    return torch.where(behind[:, None], torch.zeros((int(xyz.shape[0]), 2), device=xyz.device),
                       torch.stack((du, dv), dim=1))


@torch.no_grad()
def rasterize_pixel_flow_from_gaussian_velocities(
    camera,
    gaussians,
    pipe,
    xyz: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    """``[2, H, W]`` pixel flow, ownership-weighted through the rasteriser.

    Normalised by the ownership mass so the result is the expected pixel
    velocity at that pixel rather than a quantity that also encodes how much
    geometry happens to be there.
    """
    image_velocity = project_gaussian_velocity_to_image_flow(camera, xyz, velocity)
    payload = torch.cat((image_velocity, torch.ones((int(xyz.shape[0]), 1), device=xyz.device)), dim=1)
    splatted = splat_gaussian_features(camera, gaussians, pipe, payload)
    return splatted[:2] / splatted[2:3].clamp_min(EPS)


@torch.no_grad()
def compare_projected_flow_with_pixel_transport(
    projected_flow: torch.Tensor,
    transport,
    width: int,
) -> dict[str, float]:
    """Is the 3D motion's implied flow compatible with the image evidence?

    §47 clause 6: a hypothesis whose projected flow contradicts the measured
    pixel transport must not reach FM.  This is a consistency check, not an
    acceptance test -- the transport is search evidence and cannot certify a
    3D motion on its own (§15).
    """
    if int(transport.source_pixel_indices.numel()) == 0:
        return {"pairs": 0}
    source = transport.source_pixel_indices
    source_xy = torch.stack((source // width, source % width), dim=1).float()
    target_xy = torch.stack(
        (transport.target_pixel_indices // width, transport.target_pixel_indices % width), dim=2
    ).float()

    weight = transport.coupling_values
    observed = (weight[..., None] * (target_xy - source_xy[:, None, :])).sum(dim=1)
    observed = observed / weight.sum(dim=1, keepdim=True).clamp_min(EPS)

    flow = projected_flow.reshape(2, -1)[:, source].T                  # (du, dv)
    predicted = torch.stack((flow[:, 1], flow[:, 0]), dim=1)           # to (row, col)
    error = torch.linalg.norm(predicted - observed, dim=1)
    magnitude = torch.linalg.norm(observed, dim=1)
    return {
        "pairs": int(source.numel()),
        "median_flow_error_px": float(error.median()),
        "median_observed_displacement_px": float(magnitude.median()),
        "relative_flow_error": float(error.median() / magnitude.median().clamp_min(EPS)),
        "agreement_fraction": float((error < magnitude.clamp_min(1.0)).float().mean()),
    }
