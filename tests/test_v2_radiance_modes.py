"""V2 section 12: strict frozen radiance must be real, explicit and tested.

Section 12.3's contract is the point of this file:

    "iekf_strict_frozen_radiance must have a unit test that changes its
     behavior. A parser flag that is read but does not alter the actual
     measurement model is still a dead flag."

So these tests do not merely check that the flag is readable. They check that
the two modes produce different evaluated radiance, a different rendered image
and therefore a different residual, and that only the strict mode has the
delta_c = 0 property the analytic Jacobian assumes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.radiance import (  # noqa: E402
    STRICT,
    VIEW_DEPENDENT,
    RadianceModel,
    resolve_radiance_mode,
    unmodelled_color_jacobian,
)
from estimator.renderer_bridge import CameraSpec, build_linearization  # noqa: E402
from estimator.state import GaussianMotionState  # noqa: E402

DEV = "cuda"
PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def make_sh(n, deg=3, seed=0):
    """Frozen SH with a genuinely non-trivial directional part."""
    g = torch.Generator().manual_seed(seed)
    k = (deg + 1) ** 2
    sh = torch.zeros((n, k, 3))
    sh[:, 0, :] = torch.rand((n, 3), generator=g) * 1.5      # DC
    sh[:, 1:, :] = (torch.rand((n, k - 1, 3), generator=g) - 0.5) * 0.8
    return sh.to(DEV)


def camera(width=64, height=64, fov=0.8):
    tanf = math.tan(fov * 0.5)
    view = torch.eye(4, device=DEV)
    znear, zfar = 0.01, 100.0
    proj = torch.zeros((4, 4), device=DEV)
    proj[0, 0] = 1.0 / tanf
    proj[1, 1] = 1.0 / tanf
    proj[2, 2] = zfar / (zfar - znear)
    proj[3, 2] = 1.0
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    full = view.unsqueeze(0).bmm(proj.t().unsqueeze(0)).squeeze(0)
    return CameraSpec(viewmatrix=view, projmatrix=full, tanfovx=tanf, tanfovy=tanf,
                      image_height=height, image_width=width)


def scene(n, colors):
    g = torch.Generator().manual_seed(3)
    mu = torch.zeros((n, 3))
    mu[:, 0] = (torch.rand(n, generator=g) - 0.5) * 0.4
    mu[:, 1] = (torch.rand(n, generator=g) - 0.5) * 0.4
    mu[:, 2] = 2.0 + torch.rand(n, generator=g) * 0.3
    z = torch.zeros((n, 3))
    return GaussianMotionState(
        mu=mu.to(DEV), R=torch.eye(3).repeat(n, 1, 1).to(DEV),
        v=z.clone().to(DEV), omega=z.to(DEV), colors=colors,
        opacity=torch.full((n, 1), 0.8).to(DEV),
        scales=torch.full((n, 3), 0.08).to(DEV))


def test_strict_mode_has_zero_color_derivative():
    """Section 12.1: 'test that moving a Gaussian with fixed alpha geometry
    does not introduce an unmodelled SH color derivative'."""
    n = 64
    sh = make_sh(n)
    mu = scene(n, torch.zeros((n, 3), device=DEV)).mu
    campos = torch.tensor([0.0, 0.0, 0.0], device=DEV)
    torch.manual_seed(1)
    dmu = torch.randn((n, 3), device=DEV)

    strict = RadianceModel(sh=sh, active_degree=0, mode=STRICT)
    info = unmodelled_color_jacobian(strict, mu, campos, dmu)
    check("§12.1 strict mode has EXACTLY zero colour derivative",
          info["is_exactly_zero"], f"max |dc/dmu| = {info['max_abs_dc_dmu']:.3e}")

    vd = RadianceModel(sh=sh, active_degree=3, mode=VIEW_DEPENDENT)
    info_vd = unmodelled_color_jacobian(vd, mu, campos, dmu)
    check("§12.2 view-dependent mode has a NON-zero colour derivative",
          info_vd["max_abs_dc_dmu"] > 1e-3,
          f"max |dc/dmu| = {info_vd['max_abs_dc_dmu']:.4f}, "
          f"{info_vd['relative_unmodelled']:.1%} of the colour scale")
    check("§12.2 that derivative is the term missing from H",
          not info_vd["is_exactly_zero"])
    print(f"       the omitted term is {info_vd['max_abs_dc_dmu']:.4f} per unit "
          f"of motion, i.e. {info_vd['relative_unmodelled']:.1%} of the colour "
          f"scale -- this is why §12.2 forbids the mode")


def test_strict_mode_ignores_camera_and_position():
    """The delta_c = 0 guarantee is structural, not numerical: in strict mode
    the evaluated colour is not even a function of mu or campos."""
    n = 32
    sh = make_sh(n, seed=5)
    strict = RadianceModel(sh=sh, active_degree=0, mode=STRICT)
    mu1 = torch.randn((n, 3), device=DEV)
    mu2 = torch.randn((n, 3), device=DEV) * 10.0
    c1 = strict.evaluate(mu1, torch.zeros(3, device=DEV))
    c2 = strict.evaluate(mu2, torch.tensor([5.0, -3.0, 2.0], device=DEV))
    check("§12.1 strict radiance is identical at two unrelated geometries",
          torch.equal(c1, c2), f"max diff {float((c1 - c2).abs().max()):.3e}")
    check("§12.1 strict radiance can be evaluated with no camera at all",
          strict.evaluate(mu1, None) is not None)

    vd = RadianceModel(sh=sh, active_degree=3, mode=VIEW_DEPENDENT)
    d1 = vd.evaluate(mu1, torch.zeros(3, device=DEV))
    d2 = vd.evaluate(mu2, torch.tensor([5.0, -3.0, 2.0], device=DEV))
    check("§12.2 view-dependent radiance DOES change with geometry",
          float((d1 - d2).abs().max()) > 1e-3,
          f"max diff {float((d1 - d2).abs().max()):.4f}")
    try:
        vd.evaluate(mu1, None)
        raised = False
    except ValueError:
        raised = True
    check("§12.2 view-dependent radiance refuses to be evaluated without a camera",
          raised)


def test_the_flag_changes_the_rendered_image():
    """Section 12.3: the flag must alter the ACTUAL measurement model, which
    means the rendered image and therefore the residual."""
    n = 96
    sh = make_sh(n, seed=7)
    strict = RadianceModel(sh=sh, active_degree=0, mode=STRICT)
    vd = RadianceModel(sh=sh, active_degree=3, mode=VIEW_DEPENDENT)
    cams = [camera()]
    campos = torch.zeros(3, device=DEV)

    base = scene(n, torch.zeros((n, 3), device=DEV))
    c_strict = strict.evaluate(base.mu, campos)
    c_vd = vd.evaluate(base.mu, campos)
    check("§12.3 the two modes give different per-Gaussian colours",
          float((c_strict - c_vd).abs().max()) > 1e-3,
          f"max colour diff {float((c_strict - c_vd).abs().max()):.4f}")

    img_s = build_linearization(scene(n, c_strict), cams).views[0].rgb
    img_v = build_linearization(scene(n, c_vd), cams).views[0].rgb
    diff = float((img_s - img_v).abs().max())
    mean = float((img_s - img_v).abs().mean())
    check("§12.3 the RENDERED IMAGE differs between the modes -- the flag is "
          "not dead", diff > 1e-3, f"max pixel diff {diff:.4f}, mean {mean:.5f}")

    # And the residual against a fixed observation differs, which is what the
    # filter actually consumes.
    gt = img_s.clone()
    r_s = float(((gt - img_s) ** 2).sum())
    r_v = float(((gt - img_v) ** 2).sum())
    check("§12.3 the measurement residual differs between the modes",
          abs(r_s - r_v) > 1e-6, f"strict {r_s:.6e} vs view-dependent {r_v:.6e}")


def test_flag_contract_refuses_the_unvalidated_mode():
    """Section 12.2: the view-dependent mode must not run silently."""
    check("§12.3 strict=True resolves to DC-only",
          resolve_radiance_mode(True) == STRICT)
    try:
        resolve_radiance_mode(False)
        raised = False
        msg = ""
    except NotImplementedError as e:
        raised, msg = True, str(e)
    check("§12.2 strict=False is refused rather than silently run", raised)
    check("§12.2 and the refusal explains the missing colour derivative",
          "d c_i / d mu_i" in msg or "colour derivative" in msg, msg[:90])
    check("§12.2 it can still be selected deliberately to MEASURE the error",
          resolve_radiance_mode(False, allow_unvalidated_view_dependent=True)
          == VIEW_DEPENDENT)


def test_mode_is_logged():
    """Section 12.1: 'log the exact radiance mode'."""
    n = 16
    sh = make_sh(n, seed=9)
    for mode, deg in ((STRICT, 0), (VIEW_DEPENDENT, 3)):
        d = RadianceModel(sh=sh, active_degree=deg, mode=mode).describe()
        check(f"§12.1 the {mode} mode reports its name", d["radiance_mode"] == mode)
        for key in ("strict_frozen_radiance", "active_sh_degree",
                    "delta_c_is_zero_by_construction", "color_jacobian_in_H"):
            check(f"§12.1 {mode} logs '{key}'", key in d)
    strict_d = RadianceModel(sh=sh, active_degree=0, mode=STRICT).describe()
    vd_d = RadianceModel(sh=sh, active_degree=3, mode=VIEW_DEPENDENT).describe()
    check("§12.1 only strict mode claims delta_c = 0 by construction",
          strict_d["delta_c_is_zero_by_construction"]
          and not vd_d["delta_c_is_zero_by_construction"])
    check("§12.2 neither mode claims the colour Jacobian is in H",
          not strict_d["color_jacobian_in_H"] and not vd_d["color_jacobian_in_H"])


def test_degree_validation():
    sh = make_sh(8, deg=1, seed=2)          # only 4 coefficients stored
    try:
        RadianceModel(sh=sh, active_degree=3, mode=VIEW_DEPENDENT)
        raised = False
    except ValueError:
        raised = True
    check("a degree beyond the stored coefficients is rejected", raised)
    try:
        RadianceModel(sh=sh, active_degree=0, mode="whatever")
        raised2 = False
    except ValueError:
        raised2 = True
    check("an unknown radiance mode is rejected", raised2)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
