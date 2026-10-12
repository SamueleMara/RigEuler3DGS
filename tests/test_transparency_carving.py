"""Transparency carving A/B: the objective change and its diagnostics.

CODEX_GRAPHDECO_SPLATFACTO_CARVING_ABLATION.md section 5. Two things are under
test. First, that --splatfacto_carving changes exactly one thing and that
--masks on its own changes nothing. Second, that the opacity diagnostic
measures what it claims -- because an A/B is only worth as much as its
instrument.

    python tests/test_transparency_carving.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scene as _scene_package  # noqa: F401,E402  (scene initialises first)
from utils.carving_diagnostics import (  # noqa: E402
    CarvingDebugWriter,
    anisotropy_stats,
    camera_leakage,
    initial_state_fingerprint,
    render_projected_alpha,
)

CHECKS = 0
FAILURES: list[str] = []
SKIPPED: list[str] = []
PYTHON = sys.executable


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
            FAILURES.append(f"{label}: message lacks {needle!r}: {exc}")
            print(f"  FAIL  {label}  message lacks {needle!r}")
        return
    FAILURES.append(f"{label}: no exception raised")
    print(f"  FAIL  {label}  no exception raised")


# --------------------------------------------------------------------------
# A camera and a Gaussian model small enough to reason about by hand.
# --------------------------------------------------------------------------
def toy_camera(height=48, width=48, distance=4.0):
    fov = 0.8
    tan = float(np.tan(fov * 0.5))
    view = torch.eye(4, device="cuda")
    view[3, 2] = distance                       # 3DGS stores this transposed
    projection = torch.zeros(4, 4, device="cuda")
    projection[0, 0] = 1.0 / tan
    projection[1, 1] = 1.0 / tan
    projection[2, 2] = 100.01 / 99.99
    projection[3, 2] = 1.0
    projection[2, 3] = -2 * 100.0 * 0.01 / 99.99
    full = (view.unsqueeze(0).bmm(projection.unsqueeze(0))).squeeze(0)
    return SimpleNamespace(
        image_height=height, image_width=width, FoVx=fov, FoVy=fov,
        world_view_transform=view, full_proj_transform=full,
        camera_center=view.inverse()[3, :3], image_name="toy/frame_000000",
        object_alpha=None, object_rgb_premult=None,
    )


class ToyGaussians:
    """The attributes render() and the diagnostics touch. Nothing else."""

    def __init__(self, xyz, opacities, scale=0.3, colour=0.5):
        n = xyz.shape[0]
        self._xyz = torch.nn.Parameter(xyz.cuda())
        self._opacity = torch.nn.Parameter(
            torch.log(torch.tensor(opacities).cuda().view(n, 1)
                      / (1 - torch.tensor(opacities).cuda().view(n, 1))))
        self._scaling = torch.nn.Parameter(
            torch.log(torch.full((n, 3), scale, device="cuda")))
        rotation = torch.zeros(n, 4, device="cuda")
        rotation[:, 0] = 1.0
        self._rotation = torch.nn.Parameter(rotation)
        self._features_dc = torch.nn.Parameter(
            torch.full((n, 1, 3), colour, device="cuda"))
        self._features_rest = torch.nn.Parameter(torch.zeros(n, 15, 3, device="cuda"))
        self.max_sh_degree = 3
        self.active_sh_degree = 0

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    def get_covariance(self, scaling_modifier=1):
        raise NotImplementedError


PIPE = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                       debug=False, antialiasing=False)


# --- 1. the flags do what they say ---------------------------------------
def test_flag_semantics() -> None:
    print("[1] --masks alone changes nothing; --splatfacto_carving changes one thing")
    source = (REPO / "train.py").read_text()
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())

    check("carving is gated on the flag, not on mask presence",
          'object_mode = bool(getattr(opt, "splatfacto_carving", False)) and object_alpha is not None'
          in code)
    check("legacy branch kept verbatim",
          "image = image * camera.alpha_mask.cuda()" in code
          and "target = camera.original_image.cuda()" in code)
    check("treatment never masks the prediction",
          "image * object_alpha" not in code and "image * camera.object_alpha" not in code)
    check("treatment never masks the target or the loss",
          "target * object_alpha" not in code and "loss * " not in code)
    check("one render call per training step",
          code.count("package = render(") == 1, f"{code.count('package = render(')}")
    check("background drawn from the dedicated generator",
          "generator=bg_generator" in code)
    # Inside train_static the diagnostic is a statement, never a term: its
    # value is discarded, so there is no path from it into the loss.
    import ast
    body = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == "train_static")
    evaluate_calls = [n for n in ast.walk(body) if isinstance(n, ast.Call)
                      and getattr(n.func, "id", "") == "carving_evaluate"]
    check("the diagnostic is actually called", len(evaluate_calls) == 2,
          f"{len(evaluate_calls)} call sites")
    discarded = [n.value for n in ast.walk(body) if isinstance(n, ast.Expr)]
    check("every diagnostic call discards its value",
          all(call in discarded for call in evaluate_calls),
          "a diagnostic result is bound to a name and could reach the loss")
    backwards = [n for n in ast.walk(body) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "backward"]
    check("exactly one backward per training step", len(backwards) == 1,
          f"{len(backwards)} backward calls in train_static")

    diagnostics = (REPO / "utils" / "carving_diagnostics.py").read_text()
    check("every diagnostic render is under no_grad",
          diagnostics.count("@torch.no_grad()") >= 4)
    check("the alpha render is not masked",
          "object_alpha" not in diagnostics.split("def render_projected_alpha")[1].split("def ")[0])
    # Checked on the parsed module, not on its text: the file DOCUMENTS that
    # it must not prune, so a substring scan matches its own prohibition.
    import ast
    tree = ast.parse(diagnostics)
    called = {node.func.attr if isinstance(node.func, ast.Attribute)
              else getattr(node.func, "id", "")
              for node in ast.walk(tree) if isinstance(node, ast.Call)}
    forbidden = sorted(n for n in called
                       if any(k in n.lower()
                              for k in ("prune", "densify", "backward", "step")))
    check("diagnostics call nothing that prunes, densifies or optimises",
          not forbidden, f"calls {forbidden}")
    assigned = {t.attr for node in ast.walk(tree) if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Attribute)}
    check("diagnostics assign no model attribute",
          not (assigned & {"_xyz", "_scaling", "_opacity", "_rotation",
                           "_features_dc", "_features_rest"}),
          f"assigns {sorted(assigned)}")

    # The arguments exist with the documented defaults.
    from argparse import ArgumentParser
    from arguments import ModelParams, OptimizationParams
    parser = ArgumentParser()
    ModelParams(parser)
    OptimizationParams(parser)
    defaults = parser.parse_args(["-s", "/tmp"])
    check("carving defaults ON", defaults.splatfacto_carving is True)
    off = parser.parse_args(["-s", "/tmp", "--no-splatfacto_carving"])
    check("--no-splatfacto_carving opts out", off.splatfacto_carving is False)
    check("debug defaults off", defaults.carving_debug is False)
    check("debug interval default 1000", defaults.carving_debug_interval == 1000)
    check("background seed default 2026", defaults.carving_background_seed == 2026)
    check("masks default empty", defaults.masks == "")


# --- 2. startup validation ------------------------------------------------
def test_validation() -> None:
    global CHECKS
    print("[2] option combinations that would answer the wrong question")
    import train as train_module

    dataset = SimpleNamespace(masks="")
    scene = SimpleNamespace(getTrainCameras=lambda: [])

    raises("carving EXPLICITLY requested without --masks is rejected",
           lambda: train_module._validate_carving_options(
               dataset, SimpleNamespace(splatfacto_carving=True, carving_debug=False,
                                        random_background=False), scene,
               explicitly_requested=True),
           "requires --masks")
    # The default-on case must NOT fail a dataset that has no masks, or every
    # maskless scene stops training the moment carving becomes the default.
    CHECKS += 1
    try:
        train_module._validate_carving_options(
            dataset, SimpleNamespace(splatfacto_carving=True, carving_debug=False,
                                     random_background=False), scene,
            explicitly_requested=False)
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"default-on carving must tolerate a maskless dataset: {exc}")
        print(f"  FAIL  default-on carving must tolerate a maskless dataset: {exc}")
    # ... and --random_background must still work on such a dataset.
    CHECKS += 1
    try:
        train_module._validate_carving_options(
            dataset, SimpleNamespace(splatfacto_carving=True, carving_debug=False,
                                     random_background=True), scene,
            explicitly_requested=False)
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"--random_background must still work without masks: {exc}")
        print(f"  FAIL  --random_background must still work without masks: {exc}")
    raises("debug without --masks is rejected",
           lambda: train_module._validate_carving_options(
               dataset, SimpleNamespace(splatfacto_carving=False, carving_debug=True,
                                        random_background=False), scene),
           "requires --masks")
    raises("carving with --random_background is rejected when masks exist",
           lambda: train_module._validate_carving_options(
               SimpleNamespace(masks="m"),
               SimpleNamespace(splatfacto_carving=True, carving_debug=False,
                               random_background=True), scene,
               explicitly_requested=False),
           "random_background")

    without = SimpleNamespace(image_name="cam_000/frame_000000", object_alpha=None)
    withmask = SimpleNamespace(image_name="cam_001/frame_000000",
                               object_alpha=torch.zeros(1, 2, 2))
    raises("a camera missing its mask is rejected",
           lambda: train_module._validate_carving_options(
               SimpleNamespace(masks="m"),
               SimpleNamespace(splatfacto_carving=True, carving_debug=False,
                               random_background=False),
               SimpleNamespace(getTrainCameras=lambda: [withmask, without]),
               explicitly_requested=False),
           "no mask")

    # Neither flag set: validation must not fire at all, masks or no masks.
    CHECKS += 1
    try:
        train_module._validate_carving_options(
            SimpleNamespace(masks=""),
            SimpleNamespace(splatfacto_carving=False, carving_debug=False,
                            random_background=True), scene,
            explicitly_requested=False)
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"legacy path must not be validated: {exc}")
        print(f"  FAIL  legacy path must not be validated: {exc}")


# --- 3. the opacity diagnostic measures what it claims --------------------
def test_alpha_diagnostic() -> None:
    print("[3] projected alpha from the unmodified renderer")
    if not torch.cuda.is_available():
        SKIPPED.append("[3] no CUDA")
        print("  SKIP  no CUDA")
        return
    try:
        import diff_gaussian_rasterization  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        SKIPPED.append(f"[3] rasterizer unavailable: {exc}")
        print(f"  SKIP  rasterizer unavailable: {exc}")
        return

    camera = toy_camera()

    # (1) Nothing in front of the camera: alpha is zero everywhere.
    far = ToyGaussians(torch.tensor([[0.0, 0.0, -50.0]]), [0.99], scale=0.01)
    alpha = render_projected_alpha(camera, far, PIPE)
    check("no contribution -> alpha 0", float(alpha.max()) < 1e-5, f"max {float(alpha.max()):.2e}")

    # (2) One nearly opaque splat: alpha reaches ~1 at its centre.
    solid = ToyGaussians(torch.tensor([[0.0, 0.0, 0.0]]), [0.99], scale=0.5)
    alpha = render_projected_alpha(camera, solid, PIPE)
    check("opaque splat -> alpha ~1 at centre", float(alpha.max()) > 0.95,
          f"max {float(alpha.max()):.4f}")
    check("alpha stays within [0,1]",
          float(alpha.min()) >= 0.0 and float(alpha.max()) <= 1.0)

    # (3) Known compositing: three coincident splats of opacity a each should
    #     reach 1 - (1-a)^3 where they fully overlap.
    a = 0.5
    stack = ToyGaussians(
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2]]),
        [a, a, a], scale=0.5)
    alpha = render_projected_alpha(camera, stack, PIPE)
    expected = 1.0 - (1.0 - a) ** 3
    got = float(alpha.max())
    check("stacked opacities match 1 - prod(1 - a_i)", abs(got - expected) < 0.02,
          f"expected {expected:.4f}, got {got:.4f}")

    # (4) Independent of colour: the diagnostic overrides it.
    dark = ToyGaussians(torch.tensor([[0.0, 0.0, 0.0]]), [0.9], scale=0.5, colour=-2.0)
    bright = ToyGaussians(torch.tensor([[0.0, 0.0, 0.0]]), [0.9], scale=0.5, colour=3.0)
    da, ba = render_projected_alpha(camera, dark, PIPE), render_projected_alpha(camera, bright, PIPE)
    check("alpha independent of SH coefficients",
          float((da - ba).abs().max()) < 1e-5, f"{float((da - ba).abs().max()):.2e}")

    # (5) Repeated calls agree: no dependence on any sampled background.
    first = render_projected_alpha(camera, solid, PIPE)
    torch.rand(3, device="cuda")       # perturb the global RNG in between
    second = render_projected_alpha(camera, solid, PIPE)
    check("alpha independent of the random background",
          float((first - second).abs().max()) < 1e-6)

    # (6) no_grad: no gradient is produced, and the parameters are untouched.
    solid._opacity.grad = None
    render_projected_alpha(camera, solid, PIPE)
    check("diagnostic leaves .grad untouched", solid._opacity.grad is None)


# --- 4. leakage arithmetic ------------------------------------------------
def test_leakage_metric() -> None:
    print("[4] E_background is a pixel-weighted sum, not a mean of ratios")
    if not torch.cuda.is_available():
        SKIPPED.append("[4] no CUDA")
        print("  SKIP  no CUDA")
        return
    try:
        import diff_gaussian_rasterization  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        SKIPPED.append(f"[4] rasterizer unavailable: {exc}")
        return

    camera = toy_camera()
    model = ToyGaussians(torch.tensor([[0.0, 0.0, 0.0]]), [0.9], scale=0.5)

    # All foreground: no background pixels at all, so the metric is undefined
    # for this camera rather than zero -- and nothing divides by zero.
    camera.object_alpha = torch.ones(1, camera.image_height, camera.image_width)
    stats = camera_leakage(camera, model, PIPE)
    check("all-foreground -> N/A, not 0", np.isnan(stats["E_background_camera"]),
          str(stats["E_background_camera"]))
    check("all-foreground -> zero background weight", stats["bg_denominator"] == 0.0)
    check("all-foreground -> foreground still defined",
          np.isfinite(stats["foreground_opacity_mean"]))

    # All background: leakage is the mean alpha over the whole frame.
    camera.object_alpha = torch.zeros(1, camera.image_height, camera.image_width)
    stats = camera_leakage(camera, model, PIPE)
    alpha = render_projected_alpha(camera, model, PIPE)
    check("all-background leakage equals mean alpha",
          abs(stats["E_background_camera"] - float(alpha.mean())) < 1e-5,
          f"{stats['E_background_camera']:.6f} vs {float(alpha.mean()):.6f}")

    # Pixel weighting: a view with many background pixels must dominate one
    # with few. Averaging per-view ratios would give them equal votes.
    wide = {"bg_numerator": 1.0, "bg_denominator": 1000.0}     # ratio 0.001
    narrow = {"bg_numerator": 5.0, "bg_denominator": 10.0}     # ratio 0.5
    pixel_weighted = (wide["bg_numerator"] + narrow["bg_numerator"]) / \
                     (wide["bg_denominator"] + narrow["bg_denominator"])
    mean_of_ratios = (0.001 + 0.5) / 2
    check("the two aggregations genuinely differ",
          abs(pixel_weighted - mean_of_ratios) > 0.2,
          f"{pixel_weighted:.4f} vs {mean_of_ratios:.4f}")

    diagnostics = (REPO / "utils" / "carving_diagnostics.py").read_text()
    check("global figure accumulates numerator and denominator separately",
          "bg_num += " in diagnostics and "bg_den += " in diagnostics
          and "(bg_num / bg_den)" in diagnostics)
    check("a camera with no background is excluded from the global sums",
          "cameras_without_background" in diagnostics)


# --- 5. anisotropy and fingerprints ---------------------------------------
def test_shape_and_fingerprint() -> None:
    print("[5] anisotropy statistics and initial-state fingerprints")
    if not torch.cuda.is_available():
        SKIPPED.append("[5] no CUDA")
        print("  SKIP  no CUDA")
        return

    model = ToyGaussians(torch.zeros(4, 3), [0.5] * 4, scale=0.2)
    stats = anisotropy_stats(model)
    check("isotropic splats -> ratio 1", abs(stats["anisotropy_median"] - 1.0) < 1e-5,
          str(stats["anisotropy_median"]))
    check("count reported", stats["N_gaussians"] == 4)

    # A 10:1 splat must read as 10, from activated scales, not log scales.
    scaling = model._scaling.detach().clone()
    scaling[0] = torch.log(torch.tensor([1.0, 0.1, 0.1], device="cuda"))
    model._scaling = torch.nn.Parameter(scaling)
    stats = anisotropy_stats(model)
    check("10:1 splat reads as 10", abs(stats["anisotropy_max"] - 10.0) < 1e-3,
          str(stats["anisotropy_max"]))

    # A non-finite scale is counted and reported, never silently dropped.
    scaling = model._scaling.detach().clone()
    scaling[1] = float("nan")
    model._scaling = torch.nn.Parameter(scaling)
    stats = anisotropy_stats(model)
    check("non-finite parameters counted", stats["nonfinite_parameters"] >= 1,
          str(stats["nonfinite_parameters"]))
    check("remaining statistics still finite", np.isfinite(stats["anisotropy_max"]))

    empty = ToyGaussians(torch.zeros(0, 3), [], scale=0.2)
    stats = anisotropy_stats(empty)
    check("empty model -> N/A, no crash",
          stats["N_gaussians"] == 0 and np.isnan(stats["anisotropy_median"]))

    # Fingerprints: identical tensors agree, one changed value does not.
    a = ToyGaussians(torch.zeros(3, 3), [0.5] * 3)
    b = ToyGaussians(torch.zeros(3, 3), [0.5] * 3)
    check("identical models fingerprint identically",
          initial_state_fingerprint(a)["combined"] == initial_state_fingerprint(b)["combined"])
    moved = b._xyz.detach().clone()
    moved[0, 0] += 1e-6
    b._xyz = torch.nn.Parameter(moved)
    fa, fb = initial_state_fingerprint(a), initial_state_fingerprint(b)
    check("a 1e-6 change breaks the fingerprint", fa["combined"] != fb["combined"])
    check("the differing tensor is named",
          fa["per_tensor"]["_xyz"] != fb["per_tensor"]["_xyz"]
          and fa["per_tensor"]["_opacity"] == fb["per_tensor"]["_opacity"])


# --- 6. the writer only writes when enabled -------------------------------
def test_writer() -> None:
    print("[6] debug writer")
    with tempfile.TemporaryDirectory() as tmp:
        off = CarvingDebugWriter(tmp, "legacy", False)
        off.trace(1, "cam_000/frame_000000")
        off.global_row({"iteration": 1})
        off.initial_state({"a": 1})
        off.close()
        check("disabled writer creates nothing",
              not (Path(tmp) / "carving_debug").exists())

    with tempfile.TemporaryDirectory() as tmp:
        on = CarvingDebugWriter(tmp, "carving", True)
        for i in range(1, 4):
            on.trace(i, f"cam_00{i}/frame_000000")
        on.global_row({"iteration": 3, "branch": "carving",
                       "E_background_global": 0.25, "N_gaussians": 7})
        on.initial_state({"branch": "carving"})
        on.close()
        root = Path(tmp) / "carving_debug"
        rows = list(csv.DictReader(open(root / "camera_trace.csv")))
        check("one trace row per iteration", len(rows) == 3, str(len(rows)))
        check("trace records the camera", rows[1]["image_name"] == "cam_002/frame_000000")
        rows = list(csv.DictReader(open(root / "metrics_global.csv")))
        check("global row written", rows[0]["E_background_global"] == "0.25")
        check("initial_state.json written",
              json.loads((root / "initial_state.json").read_text())["branch"] == "carving")


# --- 7. the comparison script refuses incomparable runs -------------------
def test_comparison_refuses() -> None:
    print("[7] comparison script refuses runs that are not an A/B")
    script = REPO / "scripts" / "compare_carving_ab.py"

    def build(root: Path, combined: str, cameras: list[str], values: list[float]):
        debug = root / "carving_debug"
        debug.mkdir(parents=True)
        (debug / "initial_state.json").write_text(json.dumps({
            "branch": "x", "num_initial_gaussians": 3, "num_train_cameras": 1,
            "initial_state_sha256": {"combined": combined,
                                     "per_tensor": {"_xyz": combined}}}))
        with open(debug / "camera_trace.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["iteration", "image_name"])
            writer.writeheader()
            for index, name in enumerate(cameras, start=1):
                writer.writerow({"iteration": index, "image_name": name})
        with open(debug / "metrics_global.csv", "w", newline="") as handle:
            from utils.carving_diagnostics import GLOBAL_FIELDS
            writer = csv.DictWriter(handle, fieldnames=list(GLOBAL_FIELDS))
            writer.writeheader()
            for index, value in enumerate(values):
                writer.writerow({"iteration": index, "branch": "x",
                                 "E_background_global": value, "N_gaussians": 3,
                                 "anisotropy_median": 1.0, "anisotropy_p99": 2.0,
                                 "anisotropy_max": 3.0, "nonfinite_parameters": 0})

    def run(a: Path, b: Path, out: Path):
        return subprocess.run(
            [PYTHON, str(script), "--legacy", str(a), "--carving", str(b),
             "--output", str(out)], capture_output=True, text=True, cwd=REPO)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build(root / "a", "aaa", ["cam_000/f0", "cam_001/f0"], [0.5, 0.4])
        build(root / "b", "bbb", ["cam_000/f0", "cam_001/f0"], [0.5, 0.1])
        result = run(root / "a", root / "b", root / "out")
        check("different initial state is refused", result.returncode == 2, result.stdout[-200:])
        check("the refusal names the cause",
              "initial Gaussians differ" in result.stdout, result.stdout[-200:])

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build(root / "a", "same", ["cam_000/f0", "cam_001/f0"], [0.5, 0.4])
        build(root / "b", "same", ["cam_000/f0", "cam_002/f0"], [0.5, 0.1])
        result = run(root / "a", root / "b", root / "out")
        check("different camera order is refused", result.returncode == 2, result.stdout[-200:])
        check("the refusal names the iteration",
              "camera order diverges" in result.stdout, result.stdout[-200:])

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build(root / "a", "same", ["cam_000/f0", "cam_001/f0"], [0.5, 0.4])
        build(root / "b", "same", ["cam_000/f0", "cam_001/f0"], [0.5, 0.1])
        result = run(root / "a", root / "b", root / "out")
        check("a valid pair is accepted", result.returncode == 0, result.stdout[-300:])
        summary = (root / "out" / "summary.md").read_text()
        check("summary states the verdict",
              "Verdict" in summary and ("Background fixed" in summary
                                        or "Supports transparency" in summary),
              summary[-300:])
        rows = list(csv.DictReader(open(root / "out" / "comparison.csv")))
        check("comparison matched on iteration", len(rows) == 2 and rows[0]["iteration"] == "0")
        check("delta computed", abs(float(rows[1]["delta_E_bg"]) - (0.1 - 0.4)) < 1e-9)

    # And a run that did not improve must not be reported as if it did.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build(root / "a", "same", ["cam_000/f0"], [0.5, 0.1])
        build(root / "b", "same", ["cam_000/f0"], [0.5, 0.4])
        result = run(root / "a", root / "b", root / "out")
        check("a worse treatment is reported as no improvement",
              "No improvement" in result.stdout, result.stdout[-300:])


# --- 8. a real paired run -------------------------------------------------
def test_paired_run() -> None:
    print("[8] a real 60-iteration A/B on a masked scene")
    source = REPO / "input" / "USB_100109"
    if not torch.cuda.is_available() or not (source / "mask").is_dir():
        SKIPPED.append("[8] no CUDA or dataset absent")
        print("  SKIP  no CUDA or dataset absent")
        return

    with tempfile.TemporaryDirectory() as tmp:
        outputs = {}
        for branch in ("legacy", "carving"):
            out = Path(tmp) / branch
            command = [
                PYTHON, str(REPO / "train.py"), "-s", str(source), "-m", str(out),
                "--masks", str(source / "mask"), "-r", "1",
                "--iterations", "60", "--static_train_iters", "60",
                "--depth_l1_weight_init", "0", "--depth_l1_weight_final", "0",
                "--carving_debug", "--carving_debug_interval", "30",
                "--disable_viewer", "--no_wandb",
            ]
            # Carving is the default now, so the CONTROL is the branch that
            # has to say something: --no-splatfacto_carving loads the masks
            # and keeps the legacy loss.
            command.append("--splatfacto_carving" if branch == "carving"
                           else "--no-splatfacto_carving")
            result = subprocess.run(command, cwd=REPO, capture_output=True,
                                    text=True, timeout=2400)
            check(f"{branch} run completed", result.returncode == 0,
                  (result.stdout + result.stderr)[-1500:])
            if result.returncode != 0:
                return
            outputs[branch] = out

        states = {b: json.loads((o / "carving_debug" / "initial_state.json").read_text())
                  for b, o in outputs.items()}
        check("initial state matches across branches",
              states["legacy"]["initial_state_sha256"]["combined"]
              == states["carving"]["initial_state_sha256"]["combined"])
        check("the branches are recorded distinctly",
              states["legacy"]["splatfacto_carving"] is False
              and states["carving"]["splatfacto_carving"] is True)

        traces = {b: (o / "carving_debug" / "camera_trace.csv").read_text()
                  for b, o in outputs.items()}
        check("camera traces identical", traces["legacy"] == traces["carving"])
        check("one trace row per iteration",
              len(traces["legacy"].strip().splitlines()) == 61,
              str(len(traces["legacy"].strip().splitlines())))

        rows = {b: list(csv.DictReader(open(o / "carving_debug" / "metrics_global.csv")))
                for b, o in outputs.items()}
        for branch, data in rows.items():
            check(f"{branch} has an iteration-0 row", int(data[0]["iteration"]) == 0)
            check(f"{branch} has a final row", int(data[-1]["iteration"]) == 60)
            check(f"{branch} reports finite leakage",
                  all(np.isfinite(float(r["E_background_global"])) for r in data))
        check("iteration-0 leakage agrees across branches",
              abs(float(rows["legacy"][0]["E_background_global"])
                  - float(rows["carving"][0]["E_background_global"])) < 1e-9,
              f"{rows['legacy'][0]['E_background_global']} vs "
              f"{rows['carving'][0]['E_background_global']}")
        check("no non-finite parameters in either run",
              all(int(r["nonfinite_parameters"]) == 0
                  for data in rows.values() for r in data))

        comparison = Path(tmp) / "cmp"
        result = subprocess.run(
            [PYTHON, str(REPO / "scripts" / "compare_carving_ab.py"),
             "--legacy", str(outputs["legacy"]), "--carving", str(outputs["carving"]),
             "--output", str(comparison)], cwd=REPO, capture_output=True, text=True)
        check("comparison accepts the real pair", result.returncode == 0,
              result.stdout[-400:])
        check("comparison.csv written", (comparison / "comparison.csv").is_file())
        check("summary.md written", (comparison / "summary.md").is_file())


def main() -> int:
    test_flag_semantics()
    test_validation()
    test_alpha_diagnostic()
    test_leakage_metric()
    test_shape_and_fingerprint()
    test_writer()
    test_comparison_refuses()
    test_paired_run()
    print()
    for note in SKIPPED:
        print(f"SKIPPED {note}")
    if FAILURES:
        print(f"\n{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed" + (f" ({len(SKIPPED)} blocked)" if SKIPPED else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
