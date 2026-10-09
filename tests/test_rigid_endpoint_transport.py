from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.rigid_endpoint_transport import (
    EndpointTransportConfig,
    RigidEndpointTransport,
    integrate_endpoint_hypotheses,
)
from utils.rigid_killing_motion import RigidKillingMode


def _translation_mode(vector):
    return RigidKillingMode(
        reference_point=torch.zeros(3),
        reference_velocity=torch.as_tensor(vector).double(),
        angular_mode=torch.zeros(3, dtype=torch.float64),
        mode_type="translational",
        normalization_scale=1.0,
        valid=True,
        diagnostics={"mode_type": "translational"},
    )


def _rotation_mode(axis):
    return RigidKillingMode(
        reference_point=torch.zeros(3),
        reference_velocity=torch.zeros(3, dtype=torch.float64),
        angular_mode=torch.as_tensor(axis).double(),
        mode_type="rotational",
        normalization_scale=1.0,
        valid=True,
        diagnostics={"mode_type": "rotational"},
    )


def _transport(target, **kwargs):
    cfg = EndpointTransportConfig(
        enabled=True,
        knn=kwargs.pop("knn", 4),
        radius=kwargs.pop("radius", 10.0),
        temperature=kwargs.pop("temperature", 0.01),
        robust_delta=kwargs.pop("robust_delta", 1.0),
        **kwargs,
    )
    return RigidEndpointTransport({"xyz": torch.as_tensor(target).float()}, cfg)


def test_static_identity_wins_over_nonzero_transform():
    source = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]])
    field = {"valid": True, "mode": _translation_mode([1.0, 0.0, 0.0]), "temporal_speed": torch.tensor([1.0])}
    predicted = integrate_endpoint_hypotheses(source, [field], [0.0, 1.0])
    result = _transport(source).evaluate(source, predicted)
    assert torch.all(result.energy[:, 0] < result.energy[:, 1])


def test_pure_translation_split_prefers_correct_class():
    static = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    moving = torch.tensor([[2.0, 0.0, 0.0], [2.1, 0.0, 0.0], [2.0, 0.1, 0.0]])
    source = torch.cat((static, moving), dim=0)
    target = torch.cat((static, moving + torch.tensor([0.5, 0.0, 0.0])), dim=0)
    field = {"valid": True, "mode": _translation_mode([1.0, 0.0, 0.0]), "temporal_speed": torch.tensor([0.5])}
    predicted = integrate_endpoint_hypotheses(source, [field], [0.0, 1.0])
    result = _transport(target, radius=10.0).evaluate(source, predicted)
    assert torch.all(result.energy[: static.shape[0], 0] < result.energy[: static.shape[0], 1])
    assert torch.all(result.energy[static.shape[0] :, 1] < result.energy[static.shape[0] :, 0])


def test_pure_rotation_prefers_exact_rotational_hypothesis():
    source = torch.tensor([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 0.5, 0.0], [2.5, 1.5, 0.0]])
    angle = math.pi / 2.0
    target = torch.stack((-source[:, 1], source[:, 0], source[:, 2]), dim=-1)
    rotation = {"valid": True, "mode": _rotation_mode([0.0, 0.0, 1.0]), "temporal_speed": torch.tensor([angle])}
    wrong_translation = {"valid": True, "mode": _translation_mode([1.0, 0.0, 0.0]), "temporal_speed": torch.tensor([0.25])}
    predicted = integrate_endpoint_hypotheses(source, [rotation, wrong_translation], [0.0, 1.0])
    result = _transport(target).evaluate(source, predicted)
    assert torch.all(result.energy[:, 1] <= result.energy[:, 0])
    assert torch.all(result.energy[:, 1] <= result.energy[:, 2])


def test_missing_target_geometry_has_low_confidence():
    source = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    predicted = source[:, None, :]
    result = _transport(source[:1], radius=0.1, robust_delta=0.05).evaluate(source, predicted)
    assert result.confidence[0, 0] > 0.5
    assert result.confidence[1, 0] < 0.05
    assert not bool(result.valid[1, 0])


def test_sam3_feature_selects_discriminative_candidate():
    source = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    source_sam3 = torch.tensor([[1.0, 0.0]])
    target_sam3 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    cfg = EndpointTransportConfig(
        enabled=True,
        knn=2,
        radius=1.0,
        temperature=0.01,
        geometry_weight=0.01,
        sam3_weight=1.0,
        robust_delta=1.0,
    )
    transport = RigidEndpointTransport({"xyz": target, "sam3_features": target_sam3}, cfg)
    result = transport.evaluate(source, source[:, None, :], canonical_sam3=source_sam3)
    assert int(result.target_indices[0, 0, result.transport_weights[0, 0].argmax()]) == 1


def test_endpoint_transport_rejects_trainable_source_xyz():
    source = torch.zeros((2, 3), requires_grad=True)
    transport = _transport(torch.zeros((2, 3)))
    try:
        transport.evaluate(source, source.detach()[:, None, :])
    except RuntimeError:
        return
    raise AssertionError("trainable source XYZ was not rejected")


if __name__ == "__main__":
    test_static_identity_wins_over_nonzero_transform()
    test_pure_translation_split_prefers_correct_class()
    test_pure_rotation_prefers_exact_rotational_hypothesis()
    test_missing_target_geometry_has_low_confidence()
    test_sam3_feature_selects_discriminative_candidate()
    test_endpoint_transport_rejects_trainable_source_xyz()
