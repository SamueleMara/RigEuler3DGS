from tools.compare_eulerian_lagrangian import _winner


def test_comparison_winner_respects_metric_direction():
    eulerian = {"score": {"quality": 0.7, "error": 0.4}}
    lagrangian = {"score": {"quality": 0.8, "error": 0.2}}
    assert _winner(eulerian, lagrangian, "score.quality", True) == "lagrangian"
    assert _winner(eulerian, lagrangian, "score.error", False) == "lagrangian"

