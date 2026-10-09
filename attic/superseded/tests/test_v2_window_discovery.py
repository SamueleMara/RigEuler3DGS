"""V2 section 11 and the remaining section 19 axis tests (T-J4, T-J6).

Both candidate models are fitted to the SAME robust whitened RGB-D window
objective on the real renderer, and selected on that same number -- which is
what section 11.4 demands and what the previous rotation-share-vs-pitch-share
comparison did not do.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator.measurement import build_precision  # noqa: E402
from estimator.renderer_bridge import CameraSpec, build_linearization  # noqa: E402
from estimator.state import GaussianMotionState  # noqa: E402
from joint_reduction.line_geometry import (  # noqa: E402
    PluckerLine,
    fit_plucker_from_motion,
    revolute_transport_line,
)
from joint_reduction.window_discovery import (  # noqa: E402
    UNRESOLVED,
    SelectionConfig,
    WindowFit,
    fit_window,
    initialize_prismatic,
    initialize_revolute,
    refine_line_geometry,
    select_model,
)

DEV = "cuda"
PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def camera(width=80, height=80, fov=0.9):
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


def part(n=140, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros((n, 3))
    mu[:, 0] = (torch.rand(n, generator=g) - 0.5) * 0.4
    mu[:, 1] = (torch.rand(n, generator=g) - 0.5) * 0.4
    mu[:, 2] = 2.0 + torch.rand(n, generator=g) * 0.2
    z = torch.zeros((n, 3))
    return GaussianMotionState(
        mu=mu.to(DEV), R=torch.eye(3).repeat(n, 1, 1).to(DEV),
        v=z.clone().to(DEV), omega=z.to(DEV),
        colors=torch.rand((n, 3), generator=g).to(DEV),
        opacity=torch.full((n, 1), 0.85).to(DEV),
        scales=torch.full((n, 3), 0.06).to(DEV))


def at(state, mu, R):
    return GaussianMotionState(mu=mu, R=R, v=state.v, omega=state.omega,
                               colors=state.colors, opacity=state.opacity,
                               scales=state.scales)


def render(state, mu, R, cams):
    lin = build_linearization(at(state, mu, R), cams)
    return [(v.rgb.clone(), v.depth.clone(), (v.alpha > 0.05).clone())
            for v in lin.views]


def make_energy(state, cams):
    """The robust whitened RGB-D data term -- the SAME function for both
    candidate models (section 11.4)."""
    def energy_fn(frame, mu, R):
        lin = build_linearization(at(state, mu, R), cams)
        e = 0.0
        for view, (rgb_gt, d_gt, d_valid) in zip(lin.views, frame):
            rr = rgb_gt - view.rgb
            pr = build_precision(rr, valid_mask=None, sigma_rgb=0.1, robust=True,
                                 nu=5.0, layout=("rgb", "rgb", "rgb"), channel_dim=0)
            e += float(0.5 * (pr * rr * rr).sum())
            rd = d_gt - view.depth
            pd = build_precision(rd.unsqueeze(0), valid_mask=d_valid.unsqueeze(0),
                                 sigma_rgb=(), depth_sigma_model="constant:0.02:0.0",
                                 depth_range=d_gt.unsqueeze(0), robust=True, nu=5.0,
                                 layout=("depth",), channel_dim=0).squeeze(0)
            e += float(0.5 * (pd * rd * rd).sum())
        return e
    return energy_fn


def prismatic_transport(state):
    def f(axis, q):
        return state.mu + axis.reshape(1, 3) * float(q), state.R
    return f


def revolute_transport(state):
    def f(line, q):
        mu, R = revolute_transport_line(state.mu, state.R, line, float(q))
        return mu, R
    return f


def residual_count(cams):
    return sum(3 * c.image_height * c.image_width + c.image_height * c.image_width
               for c in cams)


def run_both(state, cams, frames, *, e_init, line_init, q_init, n_res):
    """Fit prismatic and revolute to the same window objective."""
    energy_fn = make_energy(state, cams)
    pf = fit_window("prismatic", prismatic_transport(state), e_init, frames,
                    energy_fn, q_init=q_init, num_parameters=2 + len(frames),
                    num_residuals=n_res, max_outer=3)
    rf = fit_window("revolute", revolute_transport(state), line_init, frames,
                    energy_fn, q_init=q_init, num_parameters=4 + len(frames),
                    num_residuals=n_res, max_outer=3,
                    refine_geometry=lambda g, q, fr, ef, tf: refine_line_geometry(
                        g, q, fr, ef, tf))
    return pf, rf


# ===========================================================================
# section 11
# ===========================================================================
def test_section11_prismatic_window_is_selected():
    """A pure translation must be called prismatic -- on the shared objective,
    not on a model-specific score."""
    cams = [camera()]
    state = part()
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    q_true = [0.02, 0.04, 0.06, 0.08]
    frames = [render(state, state.mu + axis * q, state.R, cams) for q in q_true]
    n_res = residual_count(cams)

    # §11.3 multi-frame initializer from the velocity field, not one twist
    dt = 0.05
    vel = [(axis * (q_true[i] - (q_true[i - 1] if i else 0.0)) / dt
            ).reshape(1, 3).expand_as(state.mu) for i in range(len(q_true))]
    init_p = initialize_prismatic(state.mu, vel)
    check("§11.3 the prismatic initializer is observable", init_p["observable"])
    ang = float(torch.rad2deg(torch.arccos(
        torch.clamp((init_p["axis"] @ axis).abs(), -1.0, 1.0))))
    check("§11.3 and recovers the axis direction from the window",
          ang < 1.0, f"{ang:.4f} deg")
    init_r = initialize_revolute(state.mu, vel)
    check("§11.3 the revolute initializer correctly reports a pure translation "
          "as unobservable", not init_r["observable"], str(init_r.get("reason")))

    line0 = PluckerLine(direction=torch.tensor([0.0, 1.0, 0.0], device=DEV),
                        moment=torch.tensor([0.0, 0.0, 0.3], device=DEV))
    pf, rf = run_both(state, cams, frames, e_init=init_p["axis"],
                      line_init=line0, q_init=q_true, n_res=n_res)
    out = select_model([pf, rf], SelectionConfig())
    print(f"       prismatic: J {pf.objective:.4f}  norm {pf.normalized_data_term():.6f}  BIC {pf.bic:.2f}")
    print(f"       revolute : J {rf.objective:.4f}  norm {rf.normalized_data_term():.6f}  BIC {rf.bic:.2f}")
    print(f"       selected : {out['selected']}  margin {out['relative_margin']:.4f}")
    check("§11.4 a pure translation selects prismatic",
          out["selected"] == "prismatic", f"{out['selected']} {out['reason']}")
    check("§11.4 both models were scored on the SAME criterion",
          out["comparable"] and len({c["model"] for c in out["candidates"]}) == 2)
    # The substantive form of "same criterion": every candidate must carry the
    # identical set of score fields, so neither can be judged on a quantity the
    # other does not have.
    keys = [set(c.keys()) for c in out["candidates"]]
    check("§11.4 the criterion is the shared window objective, not a "
          "model-specific share",
          all(k == keys[0] for k in keys)
          and {"objective", "data_term", "normalized_data_term", "bic"} <= keys[0],
          f"score fields {sorted(keys[0] & {'objective', 'data_term', 'bic'})}")


def test_section11_revolute_window_is_selected():
    """An offset-axis rotation must be called revolute, with the axis refined
    jointly (section 11.2)."""
    cams = [camera()]
    state = part(seed=3)
    e = torch.tensor([0.0, 1.0, 0.1], device=DEV)
    e = e / e.norm()
    a = torch.tensor([0.0, 0.0, 2.45], device=DEV)
    a = a - (a @ e) * e
    line = PluckerLine(direction=e, moment=torch.linalg.cross(a, e))
    q_true = [0.05, 0.10, 0.15, 0.20]
    frames = [render(state, *revolute_transport_line(state.mu, state.R, line, q),
                     cams) for q in q_true]
    n_res = residual_count(cams)

    dt = 0.05
    vel = []
    for i, q in enumerate(q_true):
        prev = q_true[i - 1] if i else 0.0
        m1, _ = revolute_transport_line(state.mu, None, line, q)
        m0, _ = revolute_transport_line(state.mu, None, line, prev)
        vel.append((m1 - m0) / dt)
    init_r = initialize_revolute(state.mu, vel)
    check("§11.3 the revolute initializer is observable", init_r["observable"],
          str(init_r.get("reason")))
    d = init_r["line"].distance_to(line)
    print(f"       initializer axis: {d['angle_deg']:.4f} deg, "
          f"{d['line_distance']:.5f} line distance, "
          f"spread {init_r['axis_spread_deg']:.4f} deg over {init_r['frames_used']} frames")
    check("§11.3 the multi-frame initializer recovers the axis direction",
          d["angle_deg"] < 2.0, f"{d['angle_deg']:.4f} deg")

    # Deliberately hand the fitter a DISPLACED axis so the refinement has work.
    bad = PluckerLine(direction=init_r["line"].direction,
                      moment=init_r["line"].moment
                      + torch.tensor([0.06, 0.0, 0.0], device=DEV))
    e_axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    pf, rf = run_both(state, cams, frames, e_init=e_axis, line_init=bad,
                      q_init=q_true, n_res=n_res)
    out = select_model([pf, rf], SelectionConfig())
    print(f"       prismatic: J {pf.objective:.4f}  BIC {pf.bic:.2f}")
    print(f"       revolute : J {rf.objective:.4f}  BIC {rf.bic:.2f}")
    print(f"       selected : {out['selected']}  margin {out['relative_margin']:.4f}")
    check("§11.4 an offset rotation selects revolute",
          out["selected"] == "revolute", f"{out['selected']} {out['reason']}")

    # T-J6: the refinement must improve the axis, not merely keep it.
    refined = PluckerLine(
        direction=torch.tensor(rf.geometry["direction"], device=DEV),
        moment=torch.tensor(rf.geometry["moment"], device=DEV))
    d_bad = bad.distance_to(line)
    d_ref = refined.distance_to(line)
    print(f"       T-J6 axis distance: {d_bad['line_distance']:.5f} (seeded) "
          f"-> {d_ref['line_distance']:.5f} (refined)")
    check("T-J6 the window refinement improves the axis LOCATION",
          d_ref["line_distance"] < d_bad["line_distance"],
          f"{d_bad['line_distance']:.5f} -> {d_ref['line_distance']:.5f}")
    check("T-J6 and keeps the direction accurate",
          d_ref["angle_deg"] < 5.0, f"{d_ref['angle_deg']:.3f} deg")


def test_T_J4_pure_translation():
    """T-J4 (section 19): a pure translation must NOT produce a revolute axis."""
    torch.manual_seed(2)
    mu = torch.randn((200, 3), dtype=torch.float64) * 0.3
    v = torch.tensor([0.4, -0.1, 0.05], dtype=torch.float64).expand_as(mu)
    line, info = fit_plucker_from_motion(mu, v, min_omega=1e-4)
    check("T-J4 a pure translation yields no revolute axis", line is None)
    check("T-J4 and is reported unobservable", info["observable"] is False)
    check("T-J4 with ||omega|| ~ 0", info["omega_norm"] < 1e-6,
          f"||omega|| = {info['omega_norm']:.3e}")

    init = initialize_revolute(mu, [v])
    check("T-J4 the window initializer also declines",
          not init["observable"], str(init.get("reason")))
    initp = initialize_prismatic(mu, [v])
    check("T-J4 but the prismatic initializer succeeds", initp["observable"])
    ang = float(torch.rad2deg(torch.arccos(torch.clamp(
        initp["axis"] @ (v[0] / v[0].norm()), -1.0, 1.0))))
    check("T-J4 and recovers the translation direction", ang < 1e-6,
          f"{ang:.3e} deg")
    check("T-J4 the direction share is ~1 for a rank-1 field",
          initp["direction_share"] > 0.999,
          f"{initp['direction_share']:.6f}")


def test_section11_unresolved_is_possible():
    """Section 11.4: if neither model explains the data, return UNRESOLVED."""
    n_res = 10000
    bad_p = WindowFit(name="prismatic", data_term=5.0e4, num_parameters=6,
                      num_residuals=n_res)
    bad_r = WindowFit(name="revolute", data_term=5.1e4, num_parameters=8,
                      num_residuals=n_res)
    out = select_model([bad_p, bad_r], SelectionConfig(max_normalized_data_term=1.0))
    check("§11.4 neither model explaining the data gives UNRESOLVED",
          out["selected"] == UNRESOLVED, out["selected"])
    check("§11.4 and the absolute-quality failure is named",
          any("absolute" in r for r in out["reason"]), str(out["reason"]))

    # good absolute quality but indistinguishable -> still UNRESOLVED
    tie_p = WindowFit(name="prismatic", data_term=100.0, num_parameters=6,
                      num_residuals=n_res)
    tie_r = WindowFit(name="revolute", data_term=100.4, num_parameters=6,
                      num_residuals=n_res)
    out2 = select_model([tie_p, tie_r],
                        SelectionConfig(max_normalized_data_term=1.0,
                                        min_relative_margin=0.05))
    check("§11.4 two indistinguishable models give UNRESOLVED",
          out2["selected"] == UNRESOLVED, out2["selected"])
    check("§11.4 and the margin failure is named",
          any("distinguishable" in r for r in out2["reason"]), str(out2["reason"]))

    # a clear winner of good quality is selected
    good_p = WindowFit(name="prismatic", data_term=50.0, num_parameters=6,
                       num_residuals=n_res)
    good_r = WindowFit(name="revolute", data_term=5000.0, num_parameters=8,
                       num_residuals=n_res)
    out3 = select_model([good_p, good_r], SelectionConfig())
    check("§11.4 a clear, good-quality winner IS selected",
          out3["selected"] == "prismatic", out3["selected"])
    check("§11.4 no candidate converging gives UNRESOLVED",
          select_model([])["selected"] == UNRESOLVED)


def test_section11_bic_charges_for_parameters():
    """A revolute joint has more parameters; BIC must charge for them so a
    marginally better fit does not automatically win."""
    n = 10000
    p = WindowFit(name="prismatic", data_term=100.0, num_parameters=6,
                  num_residuals=n)
    r = WindowFit(name="revolute", data_term=99.9, num_parameters=10,
                  num_residuals=n)
    check("§11.4 revolute fits marginally better on the raw objective",
          r.objective < p.objective)
    check("§11.4 but BIC prefers the simpler model", p.bic < r.bic,
          f"prismatic {p.bic:.3f} vs revolute {r.bic:.3f}")
    out = select_model([p, r], SelectionConfig(use_bic=True,
                                               min_relative_margin=0.0))
    check("§11.4 so the selection prefers prismatic",
          out["selected"] == "prismatic", out["selected"])
    out2 = select_model([p, r], SelectionConfig(use_bic=False,
                                                min_relative_margin=0.0))
    check("§11.4 and the criterion used is reported",
          out["criterion"] == "bic" and out2["criterion"] == "objective")


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
