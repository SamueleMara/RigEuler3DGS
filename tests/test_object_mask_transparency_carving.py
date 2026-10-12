"""Object masks as ground-truth alpha: Splatfacto-style transparency carving.

CODEX_SURGICAL_OBJECT_RGBD_3DGS.md, tests 1-7. The claim under test is narrow:
with --masks, the RGB term composites BOTH the render and the ground truth
over one freshly sampled random background per iteration, scores the whole
image, and never multiplies either side by the object mask. Without --masks
nothing changes at all.

    python tests/test_object_mask_transparency_carving.py
"""

from __future__ import annotations

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scene as _scene_package  # noqa: F401,E402  (scene must initialise first:
#   utils.camera_utils imports scene.cameras, and scene/__init__ imports
#   utils.camera_utils back. train.py happens to import scene first, so the
#   cycle never bites there. Pre-existing; not this feature's to fix.)
from utils.camera_utils import (  # noqa: E402
    load_object_alpha_premult,
    object_mask_root,
    resolve_object_mask_path,
)

CHECKS = 0
FAILURES: list[str] = []
SKIPPED: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")


def raises(label: str, fn, needle: str = "") -> None:
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the message is the assertion
        if needle and needle.lower() not in str(exc).lower():
            FAILURES.append(f"{label}: raised, but message lacks {needle!r}: {exc}")
            print(f"  FAIL  {label}  message lacks {needle!r}")
        return
    FAILURES.append(f"{label}: no exception raised")
    print(f"  FAIL  {label}  no exception raised")


def write_scene(root: Path, *, size=(8, 8), cams=("cam_000", "cam_001")) -> Namespace:
    """Two cameras whose frames share a basename, the way the real data does."""
    w, h = size
    images = root / "colmap" / "images"
    masks = root / "mask"
    (root / "colmap" / "sparse").mkdir(parents=True, exist_ok=True)
    for index, cam in enumerate(cams):
        (images / cam).mkdir(parents=True, exist_ok=True)
        (masks / cam).mkdir(parents=True, exist_ok=True)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        # A distinct square per camera, so a cross-camera mix-up is visible.
        mask[1 + index: 3 + index, 1 + index: 3 + index] = 255
        rgb[mask == 255] = (40 + 60 * index, 90, 160)
        Image.fromarray(rgb).save(images / cam / "frame_000000.png")
        Image.fromarray(mask, mode="L").save(masks / cam / "frame_000000.png")
    return Namespace(source_path=str(root), images="images", masks="mask",
                     resolution=1, data_device="cuda", train_test_exp=False,
                     depths="")


# --- 1. legacy regression -------------------------------------------------
def test_legacy_untouched() -> None:
    print("[1] legacy path, no --masks")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        args = write_scene(root)
        args.masks = ""
        check("masks unset -> no mask root", object_mask_root(args) is None)
        args.masks = None
        check("masks None -> no mask root", object_mask_root(args) is None)

    from scene.cameras import Camera
    check("Camera declares object_alpha",
          "object_alpha" in Camera.__init__.__code__.co_names
          or hasattr(Camera, "object_alpha"))

    # A camera built without masks must present the legacy signature.
    import inspect
    src = inspect.getsource(Camera.__init__)
    check("object_alpha defaults to None", "self.object_alpha = None" in src)
    check("object_rgb_premult defaults to None",
          "self.object_rgb_premult = None" in src)
    check("alpha_mask untouched", "self.alpha_mask = None" in src)

    train_src = (REPO / "train.py").read_text()
    # Comments stripped: the object branch DOCUMENTS that it does not multiply
    # by alpha, so a raw substring scan matches its own explanation.
    code_only = "\n".join(
        line.split("#", 1)[0] for line in train_src.splitlines())
    check("legacy branch keeps alpha_mask multiply",
          "image = image * camera.alpha_mask.cuda()" in code_only)
    check("legacy branch keeps original_image target",
          "target = camera.original_image.cuda()" in code_only)
    check("object branch never multiplies the prediction by alpha",
          "image * object_alpha" not in code_only
          and "image * camera.object_alpha" not in code_only)
    check("object branch never multiplies the target by alpha",
          "target * object_alpha" not in code_only
          and "target = target *" not in code_only)
    check("one render call per training step",
          code_only.count("package = render(") == 1,
          f"found {code_only.count('package = render(')}")


# --- 2. mask association --------------------------------------------------
def test_mask_association() -> None:
    print("[2] mask association across cameras")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        args = write_scene(root)
        mask_root = object_mask_root(args)
        check("relative --masks resolves under source_path",
              mask_root == root / "mask", str(mask_root))

        for cam in ("cam_000", "cam_001"):
            image = root / "colmap" / "images" / cam / "frame_000000.png"
            got = resolve_object_mask_path(image, args, mask_root)
            check(f"{cam} maps to its own mask",
                  got == root / "mask" / cam / "frame_000000.png", str(got))

        # The decisive one: identical basenames must not cross cameras.
        a = resolve_object_mask_path(
            root / "colmap" / "images" / "cam_000" / "frame_000000.png", args, mask_root)
        b = resolve_object_mask_path(
            root / "colmap" / "images" / "cam_001" / "frame_000000.png", args, mask_root)
        check("same basename, different masks", a != b, f"{a} vs {b}")

        # Absolute --masks works too (what the endpoint trainer passes).
        args_abs = Namespace(**vars(args))
        args_abs.masks = str(root / "mask")
        check("absolute --masks resolves", object_mask_root(args_abs) == root / "mask")

        # A missing mask is an error, never an all-ones fallback.
        (root / "mask" / "cam_001" / "frame_000000.png").unlink()
        raises("missing mask fails loudly",
               lambda: resolve_object_mask_path(
                   root / "colmap" / "images" / "cam_001" / "frame_000000.png",
                   args, mask_root),
               "no mask")

        # An image outside the tree has no unambiguous mask.
        raises("image outside the root fails loudly",
               lambda: resolve_object_mask_path(
                   Path("/tmp/elsewhere/frame_000000.png"), args, mask_root),
               "image root")

        args_bad = Namespace(**vars(args))
        args_bad.masks = "does_not_exist"
        raises("missing mask dir fails loudly",
               lambda: object_mask_root(args_bad), "not a directory")

        # One file reachable under two spellings is NOT ambiguous. A
        # case-insensitive filesystem -- exFAT, which the external datasets
        # live on -- reports both frame_000000.png and frame_000000.PNG for
        # the same file, and comparing the path strings refused every scene
        # on that drive. A hard link reproduces the identity relation on a
        # case-sensitive filesystem.
        import os
        link_root = root / "mask_linked"
        (link_root / "cam_000").mkdir(parents=True)
        real = link_root / "cam_000" / "frame_000000.png"
        real.write_bytes((root / "mask" / "cam_000" / "frame_000000.png").read_bytes())
        os.link(real, link_root / "cam_000" / "frame_000000.PNG")
        got = resolve_object_mask_path(
            root / "colmap" / "images" / "cam_000" / "frame_000000.png",
            args, link_root)
        check("two spellings of one file resolve, not conflict",
              os.path.samefile(got, real), str(got))

        # Two genuinely different files under one stem stay an error.
        split_root = root / "mask_split"
        (split_root / "cam_000").mkdir(parents=True)
        (split_root / "cam_000" / "frame_000000.png").write_bytes(real.read_bytes())
        (split_root / "cam_000" / "frame_000000.jpg").write_bytes(b"different file")
        raises("two distinct masks for one image still fail",
               lambda: resolve_object_mask_path(
                   root / "colmap" / "images" / "cam_000" / "frame_000000.png",
                   args, split_root),
               "ambiguous")


# --- 3. mask validation ---------------------------------------------------
def test_mask_validation() -> None:
    print("[3] mask validation at source resolution")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rgb = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

        wrong = root / "wrong.png"
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(wrong)
        raises("dimension mismatch fails loudly",
               lambda: load_object_alpha_premult(rgb, "img.png", wrong, (8, 8)),
               "disagree on size")

        grey = root / "grey.png"
        arr = np.zeros((8, 8), dtype=np.uint8)
        arr[2:4, 2:4] = 128
        Image.fromarray(arr, mode="L").save(grey)
        raises("intermediate values fail loudly",
               lambda: load_object_alpha_premult(rgb, "img.png", grey, (8, 8)),
               "not binary")

        ok = root / "ok.png"
        arr = np.zeros((8, 8), dtype=np.uint8)
        arr[2:4, 2:4] = 255
        Image.fromarray(arr, mode="L").save(ok)
        premult, alpha = load_object_alpha_premult(rgb, "img.png", ok, (8, 8))
        check("binary mask accepted", premult.shape == (3, 8, 8) and alpha.shape == (1, 8, 8),
              f"{tuple(premult.shape)} {tuple(alpha.shape)}")
        check("alpha is exactly 0/1 at native resolution",
              bool(((alpha == 0) | (alpha == 1)).all()))


# --- 4. resize edge correctness ------------------------------------------
def test_resize_edge() -> None:
    print("[4] downscale keeps premul + (1-a)*b equal to the area average")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(0)
        h0 = w0 = 8
        rgb_u8 = rng.integers(0, 256, (h0, w0, 3), dtype=np.uint8)
        mask = np.zeros((h0, w0), dtype=np.uint8)
        # Odd offsets on purpose: a square on even boundaries downsamples to
        # all-0/all-1 and would test nothing about partial coverage.
        mask[1:6, 2:7] = 255
        rgb_u8[mask == 0] = 0          # black background, as the real renders are
        rgb_path, mask_path = root / "rgb.png", root / "mask.png"
        Image.fromarray(rgb_u8).save(rgb_path)
        Image.fromarray(mask, mode="L").save(mask_path)

        image = Image.open(rgb_path)
        premult, alpha = load_object_alpha_premult(image, rgb_path, mask_path, (4, 4))
        check("downscaled shapes", premult.shape == (3, 4, 4) and alpha.shape == (1, 4, 4))
        check("partial coverage survives the resize",
              bool(((alpha > 0) & (alpha < 1)).any()),
              "every pixel ended fully 0 or 1 -- nearest-neighbour behaviour")

        rgb0 = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        alpha0 = mask.astype(np.float32) / 255.0
        for trial in range(4):
            bg = rng.random(3).astype(np.float32)
            # Per-pixel composite at FULL resolution, then plain 2x2 area average.
            full = rgb0 * alpha0[..., None] + (1.0 - alpha0)[..., None] * bg[None, None, :]
            want = full.reshape(4, 2, 4, 2, 3).mean(axis=(1, 3))          # [4,4,3]
            got = (premult + (1.0 - alpha) * torch.from_numpy(bg)[:, None, None])
            got = got.permute(1, 2, 0).numpy()
            err = float(np.abs(got - want).max())
            check(f"composite == area average (bg {trial})", err < 1e-5, f"max |err| {err:.2e}")

        # The halo this path exists to prevent. The legacy target is the
        # blacked-out RGB resized on its own and then treated as fully opaque.
        # At a half-covered edge pixel that asks the renderer for 0.5*colour --
        # a dark fringe -- while the composite asks for 0.5*colour + 0.5*bg,
        # which is what a half-covered pixel actually looks like.
        legacy = np.asarray(Image.open(rgb_path).resize((4, 4), Image.BOX),
                            dtype=np.float32) / 255.0
        edge = (alpha[0].numpy() > 0) & (alpha[0].numpy() < 1)
        check("the test image really has partial edge pixels", bool(edge.any()))
        white = torch.ones(3)
        ours = (premult + (1.0 - alpha) * white[:, None, None]).permute(1, 2, 0).numpy()
        check("legacy target is darker at the edge than the composite",
              bool((ours[edge] > legacy[edge] + 1e-6).all()),
              f"legacy mean {legacy[edge].mean():.4f} vs ours {ours[edge].mean():.4f}")
        # Where coverage is full the two targets agree, up to PIL's uint8
        # rounding -- the legacy path averages bytes, this one averages
        # float32. The halo above is ~100x larger than this.
        full = alpha[0].numpy() == 1
        gap = float(np.abs(ours[full] - legacy[full]).max())
        check("the two agree where coverage is full", gap <= 1.5 / 255.0,
              f"max gap {gap:.5f} exceeds uint8 rounding")


# --- 5. background compositing -------------------------------------------
def test_background_compositing() -> None:
    print("[5] ground-truth compositing")
    rng = np.random.default_rng(1)
    bg = torch.rand(3)

    # M = 0 everywhere: the target IS the background.
    alpha = torch.zeros(1, 4, 4)
    premult = torch.zeros(3, 4, 4)
    gt = premult + (1.0 - alpha) * bg[:, None, None]
    check("empty mask -> target is the background",
          torch.allclose(gt, bg[:, None, None].expand(3, 4, 4)),
          f"max dev {(gt - bg[:, None, None]).abs().max():.2e}")

    # M = 1 everywhere: the target is the foreground, background irrelevant.
    fg = torch.rand(3, 4, 4)
    alpha = torch.ones(1, 4, 4)
    gt = fg * alpha + (1.0 - alpha) * bg[:, None, None]
    check("full mask -> target is the foreground", torch.allclose(gt, fg))
    gt2 = fg * alpha + (1.0 - alpha) * torch.rand(3)[:, None, None]
    check("full mask -> target independent of background", torch.allclose(gt, gt2))

    # Fractional coverage interpolates linearly.
    a = torch.full((1, 1, 1), 0.25)
    colour = torch.tensor([0.8, 0.4, 0.2]).view(3, 1, 1)
    gt = colour * a + (1.0 - a) * bg[:, None, None]
    want = 0.25 * colour + 0.75 * bg[:, None, None]
    check("fractional coverage interpolates", torch.allclose(gt, want, atol=1e-7))

    # Two different backgrounds disagree on background pixels by exactly
    # (b1 - b2) * (1 - a) -- the signal that carves transparency.
    b1, b2 = torch.rand(3), torch.rand(3)
    a = torch.rand(1, 4, 4)
    p = torch.rand(3, 4, 4) * a
    d = (p + (1 - a) * b1[:, None, None]) - (p + (1 - a) * b2[:, None, None])
    check("background difference scales with (1-alpha)",
          torch.allclose(d, (1 - a) * (b1 - b2)[:, None, None], atol=1e-6))


# --- 6/7. GPU: gradients reach opacity, and a few real training steps ------
def test_gpu_opacity_gradient() -> None:
    print("[6] gradient reaches opacity through background pixels")
    if not torch.cuda.is_available():
        SKIPPED.append("[6] no CUDA")
        print("  SKIP  no CUDA")
        return
    try:
        from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                                 GaussianRasterizer)
    except Exception as exc:  # noqa: BLE001
        SKIPPED.append(f"[6] rasterizer unavailable: {exc}")
        print(f"  SKIP  rasterizer unavailable: {exc}")
        return

    from utils.loss_utils import l1_loss

    torch.manual_seed(0)
    H = W = 64
    fov = 0.8
    tan = float(np.tan(fov * 0.5))
    view = torch.eye(4, device="cuda")
    # 3DGS keeps world_view_transform TRANSPOSED (getWorld2View2(...).transpose),
    # so the translation lives in the last ROW. Putting it in the last column
    # leaves the splat behind the camera and every gradient is exactly zero.
    view[3, 2] = 4.0                       # camera 4 units back, looking +z
    proj = torch.zeros(4, 4, device="cuda")
    proj[0, 0] = 1.0 / tan
    proj[1, 1] = 1.0 / tan
    proj[2, 2] = 100.01 / 99.99
    proj[3, 2] = 1.0
    proj[2, 3] = -2 * 100.0 * 0.01 / 99.99
    full = (view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    means = torch.zeros(1, 3, device="cuda")
    means.requires_grad_(True)
    screen = torch.zeros_like(means, requires_grad=True)
    scales = torch.full((1, 3), 0.25, device="cuda")
    rots = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    shs = torch.zeros(1, 1, 3, device="cuda")
    shs[0, 0] = 0.5
    raw_opacity = torch.tensor([[2.0]], device="cuda", requires_grad=True)

    grads = []
    for trial in range(3):
        bg = torch.rand(3, device="cuda")
        settings = GaussianRasterizationSettings(
            image_height=H, image_width=W, tanfovx=tan, tanfovy=tan, bg=bg,
            scale_modifier=1.0, viewmatrix=view, projmatrix=full,
            sh_degree=0, campos=view.inverse()[3, :3], prefiltered=False,
            debug=False, antialiasing=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)
        opacity = torch.sigmoid(raw_opacity)
        out = rasterizer(means3D=means, means2D=screen, shs=shs,
                         colors_precomp=None, opacities=opacity,
                         scales=scales, rotations=rots, cov3D_precomp=None)
        image = out[0]
        # Ground truth: an EMPTY object (alpha 0 everywhere), so the target is
        # the background alone. The only way down is to switch the splat off.
        alpha_gt = torch.zeros(1, H, W, device="cuda")
        premult = torch.zeros(3, H, W, device="cuda")
        target = premult + (1.0 - alpha_gt) * bg[:, None, None]
        loss = l1_loss(image, target)
        if raw_opacity.grad is not None:
            raw_opacity.grad = None
        loss.backward()
        g = float(raw_opacity.grad.flatten()[0])
        grads.append(g)
        check(f"opacity gradient finite (trial {trial})", np.isfinite(g), f"{g}")

    check("opacity gradient nonzero", any(abs(g) > 1e-9 for g in grads),
          f"grads {grads}")
    check("gradient pushes opacity DOWN over background",
          all(g > 0 for g in grads),
          f"expected positive d(loss)/d(raw opacity), got {grads}")

    # The control: masking the prediction the old way zeroes exactly this.
    bg = torch.rand(3, device="cuda")
    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=tan, tanfovy=tan, bg=bg,
        scale_modifier=1.0, viewmatrix=view, projmatrix=full,
        sh_degree=0, campos=view.inverse()[3, :3], prefiltered=False,
        debug=False, antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    opacity = torch.sigmoid(raw_opacity)
    image = rasterizer(means3D=means, means2D=torch.zeros_like(means, requires_grad=True),
                       shs=shs, colors_precomp=None, opacities=opacity,
                       scales=scales, rotations=rots, cov3D_precomp=None)[0]
    masked = image * torch.zeros(1, H, W, device="cuda")       # alpha_gt = 0
    raw_opacity.grad = None
    l1_loss(masked, torch.zeros_like(masked)).backward()
    g_masked = float(raw_opacity.grad.flatten()[0])
    check("masking the prediction kills the gradient (the old bug)",
          abs(g_masked) < 1e-12, f"{g_masked}")


def test_gpu_smoke() -> None:
    print("[7] a few real training steps on a masked scene")
    scene_root = REPO / "input" / "USB_100109"
    if not torch.cuda.is_available():
        SKIPPED.append("[7] no CUDA")
        print("  SKIP  no CUDA")
        return
    if not (scene_root / "mask").is_dir() or not (scene_root / "colmap" / "sparse").is_dir():
        SKIPPED.append(f"[7] dataset absent at {scene_root}")
        print(f"  SKIP  dataset absent at {scene_root}")
        return

    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, str(REPO / "train.py"),
            "-s", str(scene_root), "-m", tmp,
            "--masks", str(scene_root / "mask"),
            "--iterations", "12", "--static_train_iters", "12",
            "--disable_viewer", "--no_wandb",
        ]
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
        tail = (proc.stdout + proc.stderr)[-2500:]
        check("training ran to completion", proc.returncode == 0,
              f"rc={proc.returncode}\n{tail}")
        check("no NaN reported", "nan" not in tail.lower(), tail[-600:])


def main() -> int:
    test_legacy_untouched()
    test_mask_association()
    test_mask_validation()
    test_resize_edge()
    test_background_compositing()
    test_gpu_opacity_gradient()
    test_gpu_smoke()
    print()
    for note in SKIPPED:
        print(f"SKIPPED {note}")
    if FAILURES:
        print(f"\n{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed"
          + (f" ({len(SKIPPED)} blocked)" if SKIPPED else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
