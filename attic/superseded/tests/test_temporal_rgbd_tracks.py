from __future__ import annotations

import torch

from utils.rgbd_track_lifting import backproject_pixel_with_depth, propagate_pixel_track_to_world_track
from utils.temporal_pixel_fm import build_foreground_adjacent_frame_pairs


def test_adjacent_pixel_pairs_obey_foreground_and_delta_t():
    rgb0 = torch.zeros(3, 4, 4)
    rgb1 = torch.zeros(3, 4, 4)
    depth = torch.ones(4, 4)
    m0 = torch.zeros(4, 4, dtype=torch.bool)
    m1 = torch.zeros(4, 4, dtype=torch.bool)
    m0[1, 1] = True
    m1[1, 2] = True
    field = build_foreground_adjacent_frame_pairs(rgb0, rgb1, depth, depth, m0, m1, delta_t=0.5)
    assert field.source_pixel_indices.tolist() == [5]
    assert field.target_pixel_indices.tolist() == [6]
    assert torch.allclose(field.velocity[0], torch.tensor([2.0, 0.0]))


def test_rgbd_lifting_matches_known_camera_geometry():
    K = torch.eye(3)
    uv = torch.tensor([[2.0, 3.0]])
    cam = backproject_pixel_with_depth(uv, torch.tensor([4.0]), K)
    assert torch.allclose(cam[0], torch.tensor([8.0, 12.0, 4.0]))
    T = torch.eye(4)
    T[:3, 3] = torch.tensor([1.0, 0.0, 0.0])
    track = propagate_pixel_track_to_world_track(torch.tensor([[2.0, 3.0], [2.0, 3.0]]), torch.tensor([4.0, 4.0]), K, T)
    assert track.visible.tolist() == [True, True]
    assert torch.allclose(track.positions[0], torch.tensor([9.0, 12.0, 4.0]))


if __name__ == "__main__":
    test_adjacent_pixel_pairs_obey_foreground_and_delta_t()
    test_rgbd_lifting_matches_known_camera_geometry()
    print("PASS temporal_rgbd_tracks")
