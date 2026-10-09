import torch

from utils.eulerian_contrastive_support import (
    aggregate_gaussian_responsibility_to_support,
    build_contrastive_group_ids,
    split_cells_by_dino_components,
    split_supports_by_responsibility,
)
from utils.rigid_killing_motion import (
    aggregate_points_to_rigid_cells,
    solve_rigid_killing_mode_all_intervals,
)
from utils.rigid_responsibility_assignment import (
    classify_final_gaussians,
    smooth_gaussian_responsibility_dino,
    update_gaussian_responsibility,
)


def test_gaussian_responsibility_update_and_final_polarization():
    updated = update_gaussian_responsibility(
        torch.tensor([0.0, 4.0, 1.0]),
        torch.tensor([4.0, 0.0, 1.0]),
        margin=0.0,
        temperature=0.1,
        energy_weight=torch.ones(3),
    )
    classification = classify_final_gaussians(
        updated["raw_probability"], static_threshold=0.2, rigid_threshold=0.8
    )
    assert classification["static_mask"].tolist() == [True, False, False]
    assert classification["rigid_mask"].tolist() == [False, True, False]
    assert classification["ambiguous_mask"].tolist() == [False, False, True]


def test_dino_smoothing_does_not_cross_weak_boundary():
    graph = {
        "pairs": torch.tensor([[0, 1], [1, 2]]),
        "affinity": torch.tensor([1.0, 0.001]),
        "strong_affinity": torch.tensor([True, False]),
    }
    result = smooth_gaussian_responsibility_dino(
        torch.tensor([0.0, 1.0, 1.0]), graph, strength=1.0
    )
    assert torch.allclose(result, torch.tensor([0.5, 0.5, 1.0]))


def test_mixed_cell_dino_split_prevents_raw_pooling_speed_dilution():
    count = 8
    angle = torch.arange(count, dtype=torch.float64) * (2.0 * torch.pi / count)
    base = torch.stack((torch.cos(angle), torch.sin(angle), torch.ones(count)), dim=-1)
    points = torch.stack((base, base), dim=1).reshape(-1, 3)
    raw_cells = torch.arange(count).repeat_interleave(2)
    dino = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).repeat(count, 1)
    support = split_cells_by_dino_components(
        points,
        raw_cells,
        dino,
        num_cells=count,
        enabled=True,
        cosine_threshold=0.85,
        dispersion_threshold=0.08,
        min_gaussians=1,
        local_neighbors=1,
    )
    assert support["num_supports"] == 2 * count
    assert all(reason == "dino" for reason in support["split_reason"])

    static_energy = torch.zeros(2 * count)
    rigid_energy = torch.zeros(2 * count)
    static_energy[1::2] = 4.0
    rigid_energy[0::2] = 4.0
    probability = update_gaussian_responsibility(
        static_energy,
        rigid_energy,
        margin=0.0,
        temperature=0.1,
    )["raw_probability"].double()
    assert float(probability[0::2].max()) < 0.01
    assert float(probability[1::2].min()) > 0.99

    speeds_gt = torch.tensor([0.8, 1.2], dtype=torch.float64)
    rigid_velocity = torch.cross(
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand_as(points),
        points,
        dim=-1,
    )
    gaussian_velocity = torch.zeros((2, 2 * count, 3), dtype=torch.float64)
    gaussian_velocity[:, 1::2] = speeds_gt[:, None, None] * rigid_velocity[1::2]
    positions = points[None].expand(2, -1, -1).clone()

    raw_positions = torch.stack([
        aggregate_points_to_rigid_cells(frame, raw_cells, count)
        for frame in positions
    ])
    raw_velocity = torch.stack([
        aggregate_points_to_rigid_cells(frame, raw_cells, count)
        for frame in gaussian_velocity
    ])
    raw_probability = aggregate_points_to_rigid_cells(
        probability, raw_cells, count
    )
    raw_mode, raw_speed, _ = solve_rigid_killing_mode_all_intervals(
        raw_positions,
        raw_velocity,
        weights=raw_probability[None].expand(2, -1),
        min_valid_cells=6,
    )
    assert raw_mode.valid
    assert torch.allclose(raw_speed, 0.5 * speeds_gt, atol=1.0e-5, rtol=1.0e-5)

    contrastive_ids = build_contrastive_group_ids(support)
    contrastive_positions = torch.stack([
        aggregate_points_to_rigid_cells(frame, contrastive_ids, 2 * count)
        for frame in positions
    ])
    contrastive_velocity = torch.stack([
        aggregate_points_to_rigid_cells(frame, contrastive_ids, 2 * count)
        for frame in gaussian_velocity
    ])
    support_probability = aggregate_gaussian_responsibility_to_support(
        support, probability
    )
    contrastive_mode, contrastive_speed, _ = solve_rigid_killing_mode_all_intervals(
        contrastive_positions,
        contrastive_velocity,
        weights=support_probability[None].expand(2, -1),
        min_valid_cells=6,
    )
    assert contrastive_mode.valid
    assert torch.allclose(contrastive_speed, speeds_gt, atol=1.0e-5, rtol=1.0e-5)

    lagrangian_mode, lagrangian_speed, _ = solve_rigid_killing_mode_all_intervals(
        positions,
        gaussian_velocity,
        weights=probability[None].expand(2, -1),
        min_valid_cells=6,
    )
    assert lagrangian_mode.valid
    assert torch.allclose(lagrangian_speed, speeds_gt, atol=1.0e-5, rtol=1.0e-5)


def test_responsibility_split_recovers_weak_dino_boundary_without_merging():
    points = torch.tensor(
        [[-0.3, 0.0, 0.0], [-0.2, 0.0, 0.0], [-0.1, 0.0, 0.0],
         [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    )
    state = split_cells_by_dino_components(
        points,
        torch.zeros(6, dtype=torch.long),
        torch.ones((6, 2)),
        num_cells=1,
        enabled=True,
        cosine_threshold=0.85,
        dispersion_threshold=0.08,
        min_gaussians=1,
        local_neighbors=2,
    )
    assert state["num_supports"] == 1
    probability = torch.tensor([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])
    split = split_supports_by_responsibility(
        state,
        probability,
        cycle=1,
        enabled=True,
        static_threshold=0.2,
        rigid_threshold=0.8,
        min_bimodal_fraction=0.1,
        min_gaussians=1,
    )
    assert split["num_supports"] == 2
    assert split["group_ids"][:3].unique().numel() == 1
    assert split["group_ids"][3:].unique().numel() == 1
    assert split["group_ids"][0] != split["group_ids"][3]
    retained = split_supports_by_responsibility(
        split,
        torch.full((6,), 0.5),
        cycle=2,
        enabled=True,
        static_threshold=0.2,
        rigid_threshold=0.8,
        min_bimodal_fraction=0.1,
        min_gaussians=1,
    )
    assert retained["num_supports"] == 2
    assert retained["group_ids"][0] != retained["group_ids"][3]
