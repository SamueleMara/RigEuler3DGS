import torch

from utils.gaussian_sequence_assignment import (
    DYNAMIC_CANDIDATE,
    DYNAMIC_CORE,
    STATIC_CORE,
    compute_rigid_hypothesis_unaries,
    infer_observation_speed_scalar,
    update_candidate_states,
)


def test_exact_rigid_motion_wins_and_occlusion_has_zero_weight():
    static = torch.tensor([[[4.0, 4.0, 100.0]], [[0.1, 0.1, 100.0]]])
    rigid = torch.tensor([[[0.1, 0.1, 0.0]], [[4.0, 4.0, 0.0]]])
    visible = torch.tensor([[[True, True, False]], [[True, True, False]]])
    result = compute_rigid_hypothesis_unaries(
        static, rigid, static_visibility=visible, rigid_visibility=visible, margin=0.25
    )
    assert result["margin"][0] > 3.0
    assert result["margin"][1] < -3.0
    assert result["informative_interval_count"].tolist() == [2, 2]


def test_speed_scalar_tracks_varying_global_speed():
    speed = torch.tensor([1.0, 2.0, -1.0, 0.5])
    unit = torch.tensor([1.0, 0.25]).reshape(1, 1, 1, 2).expand(2, 3, 4, 2)
    observed = unit * speed.reshape(1, 1, 4, 1)
    result = infer_observation_speed_scalar(unit, observed, speed)
    assert torch.allclose(result["speed_relative_error"], torch.zeros(2), atol=1e-5)
    assert torch.all(result["speed_correlation"] > 0.99)
    assert torch.all(result["speed_sign_consistency"] == 1)


def test_hysteresis_keeps_core_and_admits_candidate():
    prior = torch.tensor([STATIC_CORE, DYNAMIC_CORE], dtype=torch.uint8)
    hypothesis = {
        "margin": torch.tensor([1.0, -0.5]),
        "time_persistence": torch.tensor([1.0, 0.0]),
        "camera_persistence": torch.tensor([1.0, 0.0]),
    }
    state = update_candidate_states(prior, hypothesis, demotion_margin=0.7)["state"]
    assert state.tolist() == [DYNAMIC_CANDIDATE, DYNAMIC_CORE]

