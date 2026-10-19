#!/usr/bin/env python
"""Under transparency carving the depth term supervises the object only.

Carving drives every background pixel transparent. Real RGB-D captures also
measure depth on the background (walls, table, robot: 65-78% of the depth
pixels on fridge_plastic), and a depth term there demands opaque geometry at
that depth -- the two terms then leave semi-transparent splats over the
background. The SAPIEN exports have depth on the object only, so for them the
restriction changes nothing.

  [1] carving: depth pixels = valid depth AND object mask
  [2] carving: the background's measured depth does not change the depth loss
  [3] no carving: depth is supervised on every valid pixel, as before

    python tests/test_carving_depth_object_only.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

CHECKS, FAILURES = 0, []


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    import test_transparency_carving as tc
    import train

    H = W = 48
    cam = tc.toy_camera(H, W)
    g = torch.Generator(device="cuda").manual_seed(0)
    cam.original_image = torch.rand(3, H, W, device="cuda", generator=g)
    cam.alpha_mask = None
    obj = torch.zeros(1, H, W)
    obj[..., : W // 2] = 1.0                          # object = left half
    cam.object_alpha = obj
    cam.object_rgb_premult = cam.original_image.cpu() * obj
    cam.invdepthmap = torch.full((1, H, W), 0.25)    # valid depth EVERYWHERE
    cam.depth_mask = torch.ones(1, H, W)
    cam.depth_reliable = True
    model = tc.ToyGaussians(torch.tensor([[0.0, 0.0, 0.0]]), [0.9], scale=0.5)
    dataset = SimpleNamespace(train_test_exp=False)
    args = SimpleNamespace(use_semantic_feature_field=False, use_dino_feature_field=False,
                           use_sam3_feature_field=False)
    background = torch.zeros(3, device="cuda")

    def loss(carving):
        opt = SimpleNamespace(splatfacto_carving=carving, lambda_dssim=0.2)
        gen = torch.Generator(device="cuda").manual_seed(2026)
        _t, stats, _p = train._static_visual_loss(cam, model, tc.PIPE, background, dataset, args, opt,
                                                  depth_weight=1.0, bg_generator=gen)
        return stats

    print("\n[1-2] carving")
    s = loss(True)
    check("[1] depth pixels are the object pixels only", s["depth_pixels"] == H * W // 2,
          f"{s['depth_pixels']:.0f} of {H * W}")
    cam.invdepthmap = cam.invdepthmap.clone()
    cam.invdepthmap[..., W // 2:] = 3.0               # move the background's measured depth
    s2 = loss(True)
    check("[2] the background's measured depth does not reach the depth loss",
          abs(s2["depth_l1"] - s["depth_l1"]) < 1e-9, f"{s['depth_l1']:.6f} vs {s2['depth_l1']:.6f}")
    print("\n[3] no carving")
    s3 = loss(False)
    check("[3] without carving every valid depth pixel is supervised", s3["depth_pixels"] == H * W,
          f"{s3['depth_pixels']:.0f}")
    cam.invdepthmap[..., W // 2:] = 0.25
    s4 = loss(False)
    check("[3] and the background depth does reach it there", abs(s4["depth_l1"] - s3["depth_l1"]) > 1e-6)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
