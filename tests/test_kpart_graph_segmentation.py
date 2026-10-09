import torch

from utils.kpart_graph_segmentation import normalize_unary_families, solve_multilabel_potts


def test_structural_static_speed_zero_does_not_collapse_speed_scale():
    speed = torch.tensor([[0.0, 0.001], [0.0, 0.003], [0.0, 0.005], [0.0, 0.007]])
    valid = torch.isfinite(speed)
    valid[:, 0] = False
    result = normalize_unary_families({"speed": {"values": speed, "valid": valid}})
    assert result["statistics"]["speed"]["scale"] > 1.0e-3
    assert torch.equal(result["families"]["speed"][:, 0], torch.zeros(4))
    assert float(result["families"]["speed"][:, 1].abs().max()) < 3.0


def test_binary_graph_cut_completes_near_axis_gaussians_and_improves_iou():
    truth = torch.tensor([0] * 6 + [1] * 6)
    pairs = torch.tensor([[index, index + 1] for index in range(11)])
    affinity = torch.where(truth[pairs[:, 0]] == truth[pairs[:, 1]], torch.tensor(4.0), torch.tensor(0.01))
    unary = torch.full((12, 2), 2.0)
    unary[torch.arange(12), truth] = 0.0
    unary[8:10] = torch.tensor([0.7, 0.8])  # weak near-axis false-static evidence
    independent = unary.argmin(-1)
    recovered = solve_multilabel_potts(unary, pairs, affinity)
    assert int((recovered == truth).sum()) > int((independent == truth).sum())
    assert torch.equal(recovered, truth)


def test_multilabel_alpha_expansion_recovers_static_and_two_parts():
    truth = torch.arange(18) // 6
    pairs = torch.tensor([[index, index + 1] for index in range(17)])
    affinity = torch.where(truth[pairs[:, 0]] == truth[pairs[:, 1]], torch.tensor(3.0), torch.tensor(0.01))
    unary = torch.full((18, 3), 3.0)
    unary[torch.arange(18), truth] = 0.0
    unary[2] = torch.tensor([4.0, 0.0, 4.0])
    unary[8] = torch.tensor([0.0, 4.0, 4.0])
    unary[14] = torch.tensor([4.0, 0.0, 4.0])
    labels = solve_multilabel_potts(unary, pairs, affinity)
    assert torch.equal(labels, truth)
    assert set(labels.tolist()) == {0, 1, 2}
