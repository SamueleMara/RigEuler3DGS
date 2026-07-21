#!/usr/bin/env python3
"""CPU validation for the exact per-Gaussian joint implementation.

Checks:
1. Every Gaussian contributes to the full XYZ+DINO matrix.
2. Precomputed-distance k-medoids creates static + one moving initial cluster.
3. JointPartField contains one trainable vector per Gaussian and exactly 2 slots.
4. EM gradient routing is separated:
   - twist_fit -> twist only
   - proposal_rigid/static -> proposal only
   - label/field losses -> part field only
"""

import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.joint_part_field import (  # noqa: E402
    JointPartField,
    initialize_joint_parts_from_static_dino,
)
from utils.dynamic_loss_utils import joint_single_rigid_losses  # noqa: E402


class DummyTwist(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor([0.0, 0.0, 0.05]))
        self.translation = nn.Parameter(torch.tensor([0.01, 0.0, 0.0]))

    def forward(self, _t):
        return self.omega, self.translation


def _grad_norm(parameters):
    vals = []
    for p in parameters:
        if p.grad is not None:
            vals.append(p.grad.detach().norm())
    return float(torch.stack(vals).sum().item()) if vals else 0.0


def _clear(*modules_or_params):
    for obj in modules_or_params:
        if isinstance(obj, nn.Module):
            for p in obj.parameters():
                p.grad = None
        elif isinstance(obj, torch.Tensor):
            obj.grad = None


def main():
    torch.manual_seed(7)
    n_static, n_moving, feat_dim = 40, 24, 12
    xyz_static = 0.03 * torch.randn(n_static, 3) + torch.tensor([-0.25, 0.0, 0.0])
    xyz_moving = 0.03 * torch.randn(n_moving, 3) + torch.tensor([0.30, 0.0, 0.0])
    xyz = torch.cat([xyz_static, xyz_moving], dim=0)

    f_static = torch.randn(feat_dim)
    f_moving = torch.randn(feat_dim)
    f_static = f_static / f_static.norm()
    f_moving = f_moving / f_moving.norm()
    dino = torch.cat([
        f_static[None].repeat(n_static, 1) + 0.02 * torch.randn(n_static, feat_dim),
        f_moving[None].repeat(n_moving, 1) + 0.02 * torch.randn(n_moving, feat_dim),
    ], dim=0)

    init = initialize_joint_parts_from_static_dino(
        canonical_xyz=xyz,
        static_dino_field=dino,
        num_moving_parts=1,
        xyz_weight=1.0,
        dino_weight=1.0,
        distance_device="cpu",
        distance_storage="float32",
        kmedoids_iterations=8,
    )
    assert tuple(init.combined_distance.shape) == (xyz.shape[0], xyz.shape[0])
    assert init.gaussian_ids.numel() == xyz.shape[0]
    assert init.initial_cluster_id_fg.unique().numel() == 2

    field = JointPartField(
        num_gaussians=xyz.shape[0],
        gaussian_ids=init.gaussian_ids,
        initial_cluster_id_fg=init.initial_cluster_id_fg,
        num_moving_parts=1,
        embedding_dim=8,
        assignment_temperature=0.15,
        init_noise=0.0,
    )
    assert field.point_field.shape[0] == xyz.shape[0]
    assert field.prototypes.shape[0] == 2

    proposal = nn.Parameter(torch.zeros_like(xyz))
    with torch.no_grad():
        proposal[n_static:] = torch.tensor([0.0, 0.04, 0.0])
    twist = DummyTwist()
    center = xyz.mean(dim=0)
    rigid = torch.cross(twist.omega.expand_as(xyz), xyz - center, dim=-1) + twist.translation
    p = field.dynamic_probability_full()

    losses, info = joint_single_rigid_losses(
        total_velocity=proposal,
        rigid_velocity=rigid,
        proposal_velocity=proposal,
        dynamic_prob=p,
        part_field=field,
        rigid_twist_field=twist,
        current_time=0.4,
        confident_static_threshold=0.20,
        confident_dynamic_threshold=0.80,
        label_temperature=-1.0,
        field_pair_samples=64,
    )

    # twist_fit: proposal and part field must be detached.
    losses["twist_fit"].backward(retain_graph=True)
    twist_route = {
        "proposal_grad": 0.0 if proposal.grad is None else float(proposal.grad.norm().item()),
        "twist_grad": _grad_norm(twist.parameters()),
        "field_grad": _grad_norm(field.parameters()),
    }
    assert twist_route["proposal_grad"] == 0.0
    assert twist_route["twist_grad"] > 0.0
    assert twist_route["field_grad"] == 0.0
    _clear(proposal, twist, field)

    # proposal expert losses: twist and field must be detached.
    proposal_loss = losses["proposal_rigid"] + losses["proposal_static"]
    proposal_loss.backward(retain_graph=True)
    proposal_route = {
        "proposal_grad": 0.0 if proposal.grad is None else float(proposal.grad.norm().item()),
        "twist_grad": _grad_norm(twist.parameters()),
        "field_grad": _grad_norm(field.parameters()),
    }
    assert proposal_route["proposal_grad"] > 0.0
    assert proposal_route["twist_grad"] == 0.0
    assert proposal_route["field_grad"] == 0.0
    _clear(proposal, twist, field)

    # Label + clustering: both velocity experts must be detached.
    field_loss = (
        losses["label"]
        + losses["field_compact"]
        + losses["field_pairwise"]
        + losses["prototype_separation"]
        + losses["assignment_entropy"]
        + losses["part_occupancy"]
    )
    field_loss.backward()
    field_route = {
        "proposal_grad": 0.0 if proposal.grad is None else float(proposal.grad.norm().item()),
        "twist_grad": _grad_norm(twist.parameters()),
        "field_grad": _grad_norm(field.parameters()),
    }
    assert field_route["proposal_grad"] == 0.0
    assert field_route["twist_grad"] == 0.0
    assert field_route["field_grad"] > 0.0

    report = {
        "status": "PASS",
        "num_gaussians": int(xyz.shape[0]),
        "distance_matrix_shape": list(init.combined_distance.shape),
        "initial_cluster_sizes": init.cluster_sizes.tolist(),
        "num_trainable_point_vectors": int(field.point_field.shape[0]),
        "num_part_slots": int(field.prototypes.shape[0]),
        "twist_fit_gradient_route": twist_route,
        "proposal_gradient_route": proposal_route,
        "field_gradient_route": field_route,
        "adaptive_label_temperature": info["joint_label_temperature_used"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
