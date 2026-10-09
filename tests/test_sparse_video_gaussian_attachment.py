from types import SimpleNamespace

import torch

from utils.sparse_video_gaussian_attachment import (
    attach_pixels_to_projected_gaussians,
    build_projected_video_endpoint_support,
)


def _camera(width=11, height=11):
    return SimpleNamespace(
        image_width=width,
        image_height=height,
        full_proj_transform=torch.eye(4),
        world_view_transform=torch.eye(4),
    )


def test_projected_pixels_attach_to_nearest_visible_gaussians():
    camera = _camera()
    xyz = torch.tensor([
        [0.0, 0.0, 0.5],    # projects to pixel (5, 5)
        [0.8, 0.8, 0.5],    # far from pixel (5, 5)
        [2.0, 0.0, 0.5],    # outside frustum
    ])
    pixels = torch.tensor([5 + 5 * camera.image_width])
    rows = attach_pixels_to_projected_gaussians(xyz, camera, pixels, topk=2, max_pixel_radius=1.5)
    assert rows.valid[0, 0]
    assert int(rows.ids[0, 0]) == 0
    assert float(rows.weights[0, 0]) > 0.99
    assert not rows.valid[0, 1]


def test_projected_video_support_links_source_to_target_endpoint_gaussians():
    camera = _camera()
    source_xyz = torch.tensor([[0.0, 0.0, 0.5], [0.8, 0.8, 0.5]])
    target_xyz = torch.tensor([[0.8, 0.8, 0.5], [0.0, 0.0, 0.5]])
    source_pixel = torch.tensor([5 + 5 * camera.image_width])
    target_pixel = torch.tensor([9 + 9 * camera.image_width])
    support = build_projected_video_endpoint_support(
        source_xyz,
        target_xyz,
        camera,
        camera,
        source_pixel,
        target_pixel,
        torch.ones(1),
        topk_gaussians=1,
        max_pixel_radius=1.5,
    )
    assert support.source_ids.tolist() == [0]
    assert support.target_ids.tolist() == [0]
    assert torch.allclose(support.support, torch.ones_like(support.support))


def main():
    test_projected_pixels_attach_to_nearest_visible_gaussians()
    test_projected_video_support_links_source_to_target_endpoint_gaussians()
    print("PASS sparse_video_gaussian_attachment")


if __name__ == "__main__":
    main()
