import torch

from utils.rigid_killing_motion import solve_rigid_assignment


def test_static_and_rigid_hypotheses_define_the_effective_mask():
    static = torch.tensor(
        [[0.10, 1.0, 0.8], [0.12, 0.9, 0.7]], dtype=torch.float64
    )
    rigid = torch.tensor(
        [[0.09, 0.1, 0.2], [0.11, 0.2, 0.3]], dtype=torch.float64
    )
    result = solve_rigid_assignment(
        static,
        rigid,
        improvement_margin=0.05,
        temperature=0.1,
        prior_dynamic_probability=torch.tensor([0.9, 0.1, 0.5]),
    )
    assert result["effective_dynamic_mask"].tolist() == [False, True, True]
    assert result["dynamic_probability"][1] > 0.99


def test_assignment_is_interval_order_invariant_and_rejects_static_outlier_flow():
    static = torch.tensor([[0.10, 0.8], [2.0, 0.9], [0.11, 1.0]])
    rigid = torch.tensor([[0.09, 0.2], [1.95, 0.3], [0.10, 0.25]])
    weights = torch.tensor([[1.0, 1.0], [0.01, 1.0], [1.0, 1.0]])
    forward = solve_rigid_assignment(
        static, rigid, weights=weights, improvement_margin=0.05
    )
    reverse = solve_rigid_assignment(
        static.flip(0), rigid.flip(0), weights=weights.flip(0), improvement_margin=0.05
    )
    assert torch.equal(forward["effective_dynamic_mask"], reverse["effective_dynamic_mask"])
    assert forward["effective_dynamic_mask"].tolist() == [False, True]

