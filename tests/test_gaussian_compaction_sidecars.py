from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from scene.gaussian_model import GaussianModel


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_semantic_compaction_prunes_gt_part_labels():
    model = GaussianModel(0)
    n = 5
    model._xyz = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
    model._features_dc = torch.zeros(n, 1, 3)
    model._features_rest = torch.zeros(n, 0, 3)
    model._scaling = torch.zeros(n, 3)
    model._rotation = torch.zeros(n, 4)
    model._opacity = torch.zeros(n, 1)
    model._semantic_logits = torch.tensor([[2.0], [-2.0], [3.0], [-3.0], [4.0]])
    model._gt_part_labels = torch.arange(n, dtype=torch.long)
    model.max_radii2D = torch.zeros(n)
    model.xyz_gradient_accum = torch.zeros(n, 1)
    model.denom = torch.zeros(n, 1)
    model.training_setup = lambda args: None

    stats = model.compact_to_semantic_foreground(SimpleNamespace(semantic_dynamic_threshold=0.5, semantic_dynamic_min_points=1))

    check(stats["foreground_gaussians"] == 3, "three foreground Gaussians should remain")
    check(model._xyz.shape[0] == 3, "xyz was not compacted")
    check(model._gt_part_labels.tolist() == [0, 2, 4], "GT sidecar must be compacted with xyz")


if __name__ == "__main__":
    test_semantic_compaction_prunes_gt_part_labels()
    print("PASS gaussian_compaction_sidecars")
