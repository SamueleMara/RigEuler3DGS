"""Per-Gaussian observation adapter for soft rigid responsibility training."""

from __future__ import annotations

import torch

from utils.lagrangian_rigid_motion import compute_gaussian_speed_observations


@torch.no_grad()
def build_gaussian_motion_observations(
    correspondence_cache,
    query_xyz,
    source_cameras,
    target_cameras,
    time_start,
    time_end,
    *,
    gaussians,
    pipe,
    query_rotations=None,
    observation_transform=None,
):
    """Accumulate direct raster evidence with one persistent support per Gaussian."""
    return correspondence_cache.observed_velocity_targets(
        query_xyz,
        source_cameras,
        target_cameras,
        time_start,
        time_end,
        group_ids=None,
        num_groups=None,
        gaussians=gaussians,
        pipe=pipe,
        query_rotations=query_rotations,
        observation_transform=observation_transform,
    )


@torch.no_grad()
def compute_gaussian_observation_confidence(observation_states):
    confidence = torch.stack(
        [state["confidence"].detach().cpu().double() for state in observation_states]
    )
    valid = torch.stack(
        [state["valid"].detach().cpu().bool() for state in observation_states]
    )
    return {"confidence": confidence, "valid": valid}


@torch.no_grad()
def compute_gaussian_speed_diagnostics(
    positions,
    observed_velocity,
    observation_states,
    valid_mask,
    mode,
    interval_speeds,
):
    normal = torch.stack(
        [state["normal_matrix"].detach().cpu().double() for state in observation_states]
    )
    return compute_gaussian_speed_observations(
        positions,
        observed_velocity,
        normal,
        valid_mask,
        mode,
        interval_speeds,
    )
