from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys

import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.run_closed_loop_endpoint_fm_probe import _camera_mask


def test_camera_mask_prefers_dataset_object_mask_over_all_image_alpha():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mask_dir = root / "mask" / "cam_000"
        mask_dir.mkdir(parents=True)
        arr = np.zeros((4, 5), dtype=np.uint8)
        arr[1, 2] = 255
        arr[3, 4] = 255
        Image.fromarray(arr).save(mask_dir / "frame_000001.png")
        cam = SimpleNamespace(
            original_image=torch.zeros(3, 4, 5),
            alpha_mask=torch.ones(4, 5, dtype=torch.bool),
            dynamic_camera_id="cam_000",
            frame_id=1,
        )
        mask, source = _camera_mask(cam, root, return_source=True)
        assert source == "dataset_mask:mask/cam_000/frame_000001.png"
        assert int(mask.sum()) == 2
        assert not bool(mask[0, 0])
        assert bool(mask[1, 2])


def main():
    test_camera_mask_prefers_dataset_object_mask_over_all_image_alpha()
    print("PASS closed_loop_probe_masks")


if __name__ == "__main__":
    main()
