"""How much Gaussian occupancy sits over background, per trained model.

The question --masks exists to answer: does the model put geometry where the
object is not? This reads that off the existing renderer without any new
output and without touching CUDA.

The rasteriser composites

    I(b) = F + (1 - A) b

so rendering one model twice, over black and over white, gives

    I(white) - I(black) = (1 - A)

and A -- the accumulated alpha at each pixel -- follows exactly. Averaged over
the mask's background pixels it is the number that should fall when object
masks are used as ground-truth alpha; averaged over its foreground pixels it
is the control that should NOT fall, because carving the background is only
worth anything if the object survives it.

    python tools/measure_background_occupancy.py \\
        --source input/USB_100109 --iteration 3000 \\
        --model /path/without/masks --model /path/with/masks

Read-only: loads checkpoints, renders, prints. Writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import inf, nan  # noqa: F401 - cfg_args is eval'd and may hold these
from argparse import ArgumentParser, Namespace  # noqa: F401 - same
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scene as _scene_package  # noqa: F401,E402 - must initialise before utils
from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402


def _mask_for(camera, mask_root: Path) -> Path | None:
    """The mask belonging to this camera, by relative path, never basename."""
    name = Path(camera.image_name)
    for stem in (name, Path(name.parent.name) / name.name):
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = mask_root / stem.with_suffix(ext)
            if candidate.is_file():
                return candidate
    return None


def measure(model_dir: str, source: Path, mask_root: Path, iteration: int) -> dict:
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    args = parser.parse_args(["-s", str(source), "-m", str(model_dir)])

    dataset = model_params.extract(args)
    stored = eval(Path(model_dir, "cfg_args").read_text())  # noqa: S307 - 3DGS format
    for key, value in vars(stored).items():
        setattr(dataset, key, value)
    dataset.model_path = str(model_dir)
    # The masks are not needed to RENDER, and loading them here would make the
    # measurement depend on the very flag under test.
    dataset.masks = ""

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    pipe = pipeline_params.extract(args)

    black = torch.zeros(3, device="cuda")
    white = torch.ones(3, device="cuda")
    background, foreground, counted, missing = 0.0, 0.0, 0, 0
    for camera in scene.getTrainCameras():
        mask_path = _mask_for(camera, mask_root)
        if mask_path is None:
            missing += 1
            continue
        with torch.no_grad():
            over_black = render(camera, gaussians, pipe, black)["render"]
            over_white = render(camera, gaussians, pipe, white)["render"]
        occupancy = (1.0 - (over_white - over_black)).mean(0).clamp(0.0, 1.0)

        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask).cuda()
        if mask.shape != occupancy.shape:
            mask = torch.nn.functional.interpolate(
                mask[None, None], size=occupancy.shape, mode="nearest")[0, 0]
        background += float((occupancy * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1))
        foreground += float((occupancy * mask).sum() / mask.sum().clamp_min(1))
        counted += 1

    if not counted:
        raise RuntimeError(f"No camera in {model_dir} had a mask under {mask_root}")
    return {
        "model": str(model_dir),
        "iteration": iteration,
        "gaussians": int(gaussians.get_xyz.shape[0]),
        "views": counted,
        "views_without_mask": missing,
        "background_occupancy": background / counted,
        "foreground_occupancy": foreground / counted,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="scene root, holding mask/")
    ap.add_argument("--model", action="append", required=True,
                    help="trained model directory; repeat to compare")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--masks", default="mask", help="mask dir under --source")
    ap.add_argument("--json", default="", help="also write the rows here")
    args = ap.parse_args()

    source = Path(args.source).expanduser().resolve()
    mask_root = Path(args.masks)
    if not mask_root.is_absolute():
        mask_root = source / mask_root
    if not mask_root.is_dir():
        raise SystemExit(f"no mask directory at {mask_root}")

    rows = [measure(m, source, mask_root, args.iteration) for m in args.model]
    print()
    print(f"{'model':54s} {'gauss':>7s} {'bg occ':>11s} {'fg occ':>8s}")
    for row in rows:
        print(f"{row['model'][-54:]:54s} {row['gaussians']:7d} "
              f"{row['background_occupancy']:11.3e} {row['foreground_occupancy']:8.4f}")
    if len(rows) == 2:
        a, b = rows[0]["background_occupancy"], rows[1]["background_occupancy"]
        if b > 0:
            print(f"\nbackground occupancy {a:.3e} -> {b:.3e}  ({a / b:.0f}x lower)")
        print(f"foreground occupancy {rows[0]['foreground_occupancy']:.4f} -> "
              f"{rows[1]['foreground_occupancy']:.4f}  (the control: it should hold)")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
