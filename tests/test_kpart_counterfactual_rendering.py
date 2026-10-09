import torch

from utils.kpart_counterfactual_rendering import (
    evaluate_component_all_labels,
    select_informative_frame_pairs,
)


def test_long_baseline_selected_from_actual_projected_displacement():
    projected = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [2.0, 0.0]])
    pairs = select_informative_frame_pairs(projected, count=2, min_displacement=0.5, max_displacement=3.0)
    assert any(abs(item["source"] - item["target"]) == 3 for item in pairs)
    assert all(item["projected_displacement"] >= 0.5 for item in pairs)


def test_bidirectional_invalid_occlusion_comparison_does_not_vote():
    pairs = [{"source": 0, "target": 1}, {"source": 1, "target": 0}]

    def render(_, label, pair):
        return {"label": label, "pair": pair}

    def loss(rendered, _):
        if rendered["pair"]["source"] == 0:
            return {"valid": False, "total": torch.tensor(float("nan")), "rgb": torch.tensor(float("nan")), "depth": torch.tensor(float("nan")), "dino": torch.tensor(float("nan"))}
        value = torch.tensor(float(rendered["label"]))
        return {"valid": True, "total": value, "rgb": value, "depth": value, "dino": value}

    result = evaluate_component_all_labels(torch.tensor([0]), [0, 1], pairs, render_label_fn=render, loss_fn=loss)
    assert result["winner"] == 0
    assert result["labels"][0]["valid_comparisons"] == 1
