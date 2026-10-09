"""Stage-B acceptance tests: the two parallel physical Phase-2 models.

CODEX_VALIDATE_GENERIC_TWIST_THEN_PARALLEL_PRISMATIC_REVOLUTE_PHASE2.md
sections 34 (synthetic P), 35 (synthetic R), 36 (wrong initialization),
plus the section 20/26 derivative contracts and the section 28 "no new
optimizer" structural check.

Everything runs against the real CUDA renderer through the shared Phase-1
linearization, not a surrogate.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator import IEKFConfig, build_linearization, build_precision  # noqa: E402
from estimator import phase2_prismatic as mp  # noqa: E402
from estimator import phase2_revolute as mr  # noqa: E402
from estimator.joint_iekf import JointIEKFConfig  # noqa: E402
from estimator.phase2_model_comparison import (  # noqa: E402
    fit_model, init_prismatic, init_revolute, prismatic_state, revolute_state,
)
from estimator.manifold import so3_exp  # noqa: E402
from estimator.joint_state import JointState, initialise_from_twist  # noqa: E402
from estimator.phase2_revolute_init import (  # noqa: E402
    FITTED, UNOBSERVABLE, initialize_revolute, revolute_part_from_axis,
)
from estimator.rigid_motion_audit import _aggregate_scores, rigidity_check  # noqa: E402

sys.path.insert(0, str(REPO / "tests"))
from test_iekf_end_to_end import camera, make_state, render_gt  # noqa: E402

DEV = "cuda"
PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# A synthetic articulated scene, rendered by the production path
# ---------------------------------------------------------------------------

def _cams(n=3):
    return [camera(yaw=y, radius=0.0) for y in
            torch.linspace(-0.35, 0.35, n).tolist()]


def _scene(seed=0, n=40):
    return make_state(n=n, seed=seed, spread=0.45)


def _make_frames(truth, cams, cfg, transforms):
    """Render the ground-truth sequence under a list of (R, p) part motions."""
    frames = []
    for (R, p) in transforms:
        mu = truth.mu.double() @ R.T.to(truth.mu.device) + p.to(truth.mu.device)
        Rg = R.to(truth.mu.device).float() @ truth.R
        s = dataclasses.replace(truth, mu=mu.to(truth.mu.dtype), R=Rg)
        frames.append(render_gt(s, cams, cfg))
    return frames


def _bindings(truth, cams, cfg, frames, state):
    """The same callbacks the production runtime passes to `run_window`."""
    counter = {"frame": 0}

    def linearize(st, frame):
        counter["frame"] = frame
        mu, _ = st.gaussians(frame, truth.mu.double(), None)
        s = dataclasses.replace(truth, mu=mu.to(truth.mu.dtype))
        return build_linearization(s, cams, eps_depth=cfg.eps_depth)

    def precision_fn(lin):
        obs = frames[counter["frame"]]
        out = []
        for v, ob in zip(lin.views, obs):
            pr = build_precision(ob[0].to(v.rgb.device) - v.rgb,
                                 valid_mask=None, sigma_rgb=cfg.rgb_sigma,
                                 robust=False, layout=("rgb", "rgb", "rgb"),
                                 channel_dim=0)
            m = (v.alpha > cfg.min_alpha_for_depth) & ob[2].to(v.depth.device)
            pd = build_precision((ob[1].to(v.depth.device)
                                  - v.depth).unsqueeze(0),
                                 valid_mask=m.unsqueeze(0), sigma_rgb=(),
                                 depth_sigma_model=cfg.depth_sigma_model,
                                 robust=False, layout=("depth",),
                                 channel_dim=0, use_rgb=False).squeeze(0)
            out.append((pr, pd))
        return out

    obs = [[(o[0], o[1]) for o in f] for f in frames]
    return linearize, precision_fn, obs


def _energy_of(state, truth, cams, cfg, frames):
    """E_M on the current nominal, through the production measurement model."""
    tot = 0.0
    for t, f in enumerate(frames):
        mu, _ = state.gaussians(t, truth.mu.double(), None)
        s = dataclasses.replace(truth, mu=mu.to(truth.mu.dtype))
        lin = build_linearization(s, cams, eps_depth=cfg.eps_depth)
        for v, ob in zip(lin.views, f):
            rr = (ob[0].to(v.rgb.device) - v.rgb).double()
            dr = (ob[1].to(v.depth.device) - v.depth).double()
            m = ((v.alpha > cfg.min_alpha_for_depth)
                 & ob[2].to(v.depth.device)).double()
            tot += 0.5 * float((rr * rr).sum()) / cfg.rgb_sigma ** 2
            tot += 0.5 * float((m * dr * dr).sum()) / cfg.depth_sigma_a ** 2
    return tot


# ---------------------------------------------------------------------------
# Section 20 / 26: the derivative contracts
# ---------------------------------------------------------------------------

def test_S20_S26_analytic_jacobians():
    g = torch.Generator().manual_seed(0)
    mu0 = torch.randn((200, 3), generator=g, dtype=torch.float64)
    cases = [
        ("M_P", mp.PrismaticPart(torch.arange(200),
                                 torch.tensor([0.2, -0.5, 0.84]),
                                 torch.linspace(0, 0.7, 8)),
         mp.PRISMATIC_ERROR_DIM, mp.prismatic_jvp, mp.prismatic_vjp,
         mp.current_points, (0, 2, 3)),
        ("M_R", mr.RevolutePart(torch.arange(200), mu0.mean(0),
                                torch.tensor([0.2, -0.5, 0.84]),
                                torch.tensor([0.3, -0.2]),
                                torch.linspace(0, 0.9, 8)),
         mr.REVOLUTE_ERROR_DIM, mr.revolute_jvp, mr.revolute_vjp,
         mr.current_points, (0, 2, 3, 4, 5)),
    ]
    for name, part, dim, jvp, vjp, cur, coords in cases:
        f, h = 5, 1e-6
        mu, _, _, _ = cur(part, f, mu0, None)
        worst = 0.0
        for k in coords:
            d = torch.zeros(dim, dtype=torch.float64)
            d[k] = h
            a, _, _, _ = cur(part.retract(d, f), f, mu0, None)
            b, _, _, _ = cur(part.retract(-d, f), f, mu0, None)
            e = torch.zeros(dim, dtype=torch.float64)
            e[k] = 1.0
            an = jvp(part, f, e, mu)[:, :3]
            worst = max(worst, float((((a - b) / (2 * h)) - an).abs().max()
                                     / max(float(an.abs().max()), 1e-30)))
        check(f"S20/26 {name} JVP == finite differences", worst < 1e-6,
              f"rel {worst:.2e}")
        gg = torch.Generator().manual_seed(3)
        wadj = 0.0
        for _ in range(8):
            v = torch.randn(dim, generator=gg, dtype=torch.float64)
            z = torch.randn((200, 6), generator=gg, dtype=torch.float64)
            l = float((jvp(part, f, v, mu) * z).sum())
            r = float((v * vjp(part, f, z, mu)).sum())
            wadj = max(wadj, abs(l - r) / max(abs(l), abs(r), 1e-30))
        check(f"S20/26 {name} VJP is the exact transpose", wadj < 1e-12,
              f"rel {wadj:.2e}")


def test_S18_S22_constraints_are_exact():
    """`|e| = 1` for both models and `e^T(a - c_0) = 0` for M_R, after
    retraction -- not enforced by a penalty, true by construction."""
    g = torch.Generator().manual_seed(7)
    mu0 = torch.randn((50, 3), generator=g, dtype=torch.float64)
    p = mp.PrismaticPart(torch.arange(50), torch.tensor([0.1, 0.3, -0.9]),
                         torch.linspace(0, 0.4, 5))
    r = mr.RevolutePart(torch.arange(50), mu0.mean(0),
                        torch.tensor([0.1, 0.3, -0.9]),
                        torch.tensor([0.4, -0.3]), torch.linspace(0, 0.6, 5))
    wp = wr = wg = 0.0
    for k in range(25):
        d4 = torch.randn(4, generator=g, dtype=torch.float64) * 0.4
        d6 = torch.randn(6, generator=g, dtype=torch.float64) * 0.4
        p, r = p.retract(d4, 2), r.retract(d6, 2)
        wp = max(wp, abs(float(p.direction.norm()) - 1.0))
        wr = max(wr, abs(float(r.axis.norm()) - 1.0))
        wg = max(wg, abs(r.geometry()["gauge_e_dot_offset"]))
    check("S18 M_P keeps |e_P| = 1 under retraction", wp < 1e-12, f"{wp:.2e}")
    check("S22 M_R keeps |e_R| = 1 under retraction", wr < 1e-12, f"{wr:.2e}")
    check("S22 M_R keeps e_R^T(a_R - c_0) = 0 exactly", wg < 1e-12,
          f"{wg:.2e}")


# ---------------------------------------------------------------------------
# Section 34: synthetic P
# ---------------------------------------------------------------------------

def _run(state, truth, cams, cfg, frames, iters=6):
    lin, prec, obs = _bindings(truth, cams, cfg, frames, state)
    jcfg = JointIEKFConfig(max_iters=iters, max_cg_iters=40, verbose=False)
    return fit_model(state, truth.mu.double(), obs, lin, prec, jcfg)


def test_S34_synthetic_prismatic():
    cfg = IEKFConfig()
    truth, cams = _scene(seed=1), _cams()
    e_true = torch.tensor([0.6, 0.3, -0.742], dtype=torch.float64)
    e_true = e_true / e_true.norm()
    s_true = torch.tensor([0.0, 0.02, 0.04, 0.06, 0.08], dtype=torch.float64)
    I3 = torch.eye(3, dtype=torch.float64)
    frames = _make_frames(truth, cams, cfg,
                          [(I3, s * e_true) for s in s_true])
    pos = [truth.mu.double().cpu() + s * e_true for s in s_true]
    members = torch.arange(truth.mu.shape[0])

    # Section 29's initializer is EXACT on noiseless synthetic data (it reads
    # the direction straight off the COM displacements), so starting there
    # would make "the energy went down" vacuous: E is already 0.  Perturb it
    # so the filter has real work to do and the recovery is what is measured.
    part = init_prismatic(truth.mu.double().cpu(), members, pos)
    part = part.retract(torch.tensor([0.0, 0.0, 0.14, -0.10],
                                     dtype=torch.float64), 0)
    part.s = part.s * 0.7
    st = prismatic_state(part)
    e0 = _energy_of(st, truth, cams, cfg, frames)
    out = _run(st, truth, cams, cfg, frames)
    e1 = _energy_of(st, truth, cams, cfg, frames)
    fitted = st.parts[0]
    ang = math.degrees(math.acos(min(1.0, abs(float(
        fitted.direction @ e_true)))))
    serr = float((fitted.s.abs() - s_true.abs()).abs().max())
    check("S34 M_P recovers the prismatic direction", ang < 2.0,
          f"{ang:.3f} deg")
    check("S34 M_P recovers the travel", serr < 5e-3, f"max |ds| {serr:.2e}")
    check("S34 M_P reduces the energy", e1 < 0.05 * e0,
          f"{e0:.4e} -> {e1:.4e}")

    rp = init_revolute(truth.mu.double().cpu(), members, pos)
    rst = revolute_state(rp)
    _run(rst, truth, cams, cfg, frames)
    er = _energy_of(rst, truth, cams, cfg, frames)
    # Section 33: this is evidence, NOT a threshold-based label.  R is strictly
    # more flexible than P, so it may match E_P; what must not happen is R
    # beating P on data that is exactly prismatic by more than the renderer's
    # own reproducibility.
    check("S34 M_R does not beat M_P on exactly-prismatic data",
          er >= e1 * (1.0 - 1e-2), f"E_P {e1:.4e}  E_R {er:.4e}")


# ---------------------------------------------------------------------------
# Section 35: synthetic R
# ---------------------------------------------------------------------------

def test_S35_synthetic_revolute():
    cfg = IEKFConfig()
    truth, cams = _scene(seed=2), _cams()
    e_true = torch.tensor([0.1, 0.97, 0.22], dtype=torch.float64)
    e_true = e_true / e_true.norm()
    c0 = truth.mu.double().cpu().mean(0)
    a_true = c0 + torch.tensor([0.25, 0.0, 0.1], dtype=torch.float64)
    a_true = a_true - e_true * float(e_true @ (a_true - c0))  # on the gauge
    th_true = torch.tensor([0.0, 0.05, 0.10, 0.15, 0.20], dtype=torch.float64)
    tf, pos = [], []
    for th in th_true:
        R = so3_exp((float(th) * e_true).reshape(1, 3))[0].double()
        p = a_true - R @ a_true
        tf.append((R, p))
        pos.append(truth.mu.double().cpu() @ R.T + p)
    frames = _make_frames(truth, cams, cfg, tf)
    members = torch.arange(truth.mu.shape[0])

    rp = init_revolute(truth.mu.double().cpu(), members, pos)
    rp = rp.retract(torch.tensor([0.0, 0.0, 0.12, -0.09, 0.05, -0.04],
                                 dtype=torch.float64), 0)
    rp.theta = rp.theta * 0.7
    rst = revolute_state(rp)
    e0 = _energy_of(rst, truth, cams, cfg, frames)
    _run(rst, truth, cams, cfg, frames)
    er = _energy_of(rst, truth, cams, cfg, frames)
    fit = rst.parts[0]
    ang = math.degrees(math.acos(min(1.0, abs(float(fit.axis @ e_true)))))
    # Only the component of the axis-point error PERPENDICULAR to the axis is
    # physical; sliding along the axis moves nothing.
    d = fit.axis_point - a_true
    perp = float((d - fit.axis * float(fit.axis @ d)).norm())
    therr = float((fit.theta.abs() - th_true.abs()).abs().max())
    check("S35 M_R recovers the axis direction", ang < 3.0, f"{ang:.3f} deg")
    check("S35 M_R recovers the axis position", perp < 5e-2,
          f"perp {perp:.3e}")
    check("S35 M_R recovers the angle trajectory", therr < 2e-2,
          f"max |dtheta| {therr:.2e}")
    check("S35 M_R reduces the energy", er < 0.05 * e0,
          f"{e0:.4e} -> {er:.4e}")

    pp = init_prismatic(truth.mu.double().cpu(), members, pos)
    pst = prismatic_state(pp)
    _run(pst, truth, cams, cfg, frames)
    ep = _energy_of(pst, truth, cams, cfg, frames)
    check("S35 M_R beats M_P on exactly-revolute data", er < ep,
          f"E_R {er:.4e} < E_P {ep:.4e}")


# ---------------------------------------------------------------------------
# Section 36: wrong initialization
# ---------------------------------------------------------------------------

def test_S36_wrong_initialization():
    cfg = IEKFConfig()
    truth, cams = _scene(seed=3), _cams()
    e_true = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    s_true = torch.tensor([0.0, 0.02, 0.04, 0.06], dtype=torch.float64)
    I3 = torch.eye(3, dtype=torch.float64)
    frames = _make_frames(truth, cams, cfg, [(I3, s * e_true) for s in s_true])
    members = torch.arange(truth.mu.shape[0])

    # 25 degrees off, and the travel scaled by 0.5
    bad = torch.tensor([0.0, math.sin(0.436), math.cos(0.436)],
                       dtype=torch.float64)
    part = mp.PrismaticPart(members, bad, s_true * 0.5)
    st = prismatic_state(part)
    _run(st, truth, cams, cfg, frames, iters=10)
    ang = math.degrees(math.acos(min(1.0, abs(float(
        st.parts[0].direction @ e_true)))))
    check("S36 M_P converges from a 25 deg wrong direction", ang < 5.0,
          f"{ang:.3f} deg")


# ---------------------------------------------------------------------------
# Section 15 / 38: a REAL wrong-initialization test for R
# ---------------------------------------------------------------------------

def _revolute_sequence(truth, e_true, a_true, th_true):
    tf, pos = [], []
    for th in th_true:
        R = so3_exp((float(th) * e_true).reshape(1, 3))[0].double()
        p = a_true - R @ a_true
        tf.append((R, p))
        pos.append(truth.mu.double().cpu() @ R.T + p)
    return tf, pos


def test_S15_wrong_initialization_revolute():
    """Section 15: axis direction ~25 deg wrong, axis offset shifted by O(part
    scale), theta scaled 0.6, and a NON-CONSTANT angular speed. The same IEKF
    must recover all three."""
    cfg = IEKFConfig()
    truth, cams = _scene(seed=5, n=60), _cams(4)
    e_true = torch.tensor([0.12, 0.96, 0.25], dtype=torch.float64)
    e_true = e_true / e_true.norm()
    c0 = truth.mu.double().cpu().mean(0)
    scale = float((truth.mu.double().cpu() - c0).norm(dim=1).mean())
    a_true = c0 + torch.tensor([0.3, 0.0, 0.12], dtype=torch.float64)
    a_true = a_true - e_true * float(e_true @ (a_true - c0))
    # non-constant angular speed: accelerating, then nearly stopping
    th_true = torch.tensor([0.0, 0.03, 0.09, 0.17, 0.22, 0.24],
                           dtype=torch.float64)
    _tf, pos = _revolute_sequence(truth, e_true, a_true, th_true)
    frames = _make_frames(truth, cams, cfg,
                          [(so3_exp((float(t) * e_true).reshape(1, 3))[0].double(),
                            a_true - so3_exp((float(t) * e_true).reshape(1, 3))[0].double() @ a_true)
                           for t in th_true])
    members = torch.arange(truth.mu.shape[0])

    # ---- the deliberately wrong start ---------------------------------
    ax = torch.linalg.cross(e_true, torch.tensor([1.0, 0.0, 0.0],
                                                 dtype=torch.float64))
    ax = ax / ax.norm()
    W = so3_exp((math.radians(25.0) * ax).reshape(1, 3))[0].double()
    e_bad = W @ e_true
    a_bad = a_true + torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) * scale
    bad = revolute_part_from_axis(members, truth.mu.double().cpu(), e_bad,
                                  a_bad, th_true * 0.6)
    ang0 = math.degrees(math.acos(min(1.0, abs(float(bad.axis @ e_true)))))
    d0 = bad.axis_point - a_true
    perp0 = float((d0 - bad.axis * float(bad.axis @ d0)).norm())
    check("S15 the start really is wrong", ang0 > 20.0 and perp0 > 0.2 * scale,
          f"{ang0:.1f} deg, offset {perp0:.3f} (part scale {scale:.3f})")

    st = revolute_state(bad)
    e0 = _energy_of(st, truth, cams, cfg, frames)
    _run(st, truth, cams, cfg, frames, iters=12)
    e1 = _energy_of(st, truth, cams, cfg, frames)
    fit = st.parts[0]
    ang = math.degrees(math.acos(min(1.0, abs(float(fit.axis @ e_true)))))
    d = fit.axis_point - a_true
    perp = float((d - fit.axis * float(fit.axis @ d)).norm())
    therr = float((fit.theta.abs() - th_true.abs()).abs().max())
    check("S15 M_R recovers the axis direction from 25 deg wrong",
          ang < 6.0, f"{ang0:.1f} -> {ang:.3f} deg")
    check("S15 M_R recovers the axis position from O(scale) wrong",
          perp < 0.25 * scale, f"{perp0:.3f} -> {perp:.3f} (scale {scale:.3f})")
    check("S15 M_R recovers a non-constant angle trajectory",
          therr < 4e-2, f"max |dtheta| {therr:.2e}")
    check("S15 M_R reduces the energy", e1 < 0.5 * e0, f"{e0:.4e} -> {e1:.4e}")


# ---------------------------------------------------------------------------
# Section 38: near-P revolute -- observability, not a threshold
# ---------------------------------------------------------------------------

def test_S38_near_p_observability():
    """A far-axis / small-angle revolute sequence, and a pure-prismatic one.

    The required behaviour is NOT "detect revolute". It is that the status is
    honest: curvature is reported observable only when the data carries it,
    and `R_GEOMETRY_UNOBSERVABLE` otherwise -- with no axis-radius, omega or
    angle threshold anywhere in the decision.
    """
    members = torch.arange(40)
    mu0 = torch.randn((40, 3), generator=torch.Generator().manual_seed(11),
                      dtype=torch.float64)
    r = initialize_revolute(members, mu0, None, screw_state=None)
    check("S38 no screw and no bootstrap -> UNOBSERVABLE",
          r["status"] == UNOBSERVABLE and r["part"] is None, r["status"])

    # A screw whose angular block is exactly zero is prismatic: there is no
    # axis to extract, and the hierarchy must say so rather than invent one.
    part = initialise_from_twist(
        members, mu0, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   dtype=torch.float64),
        torch.linspace(0, 0.4, 5), None)
    st = JointState([part], torch.arange(5, dtype=torch.float64))
    r = initialize_revolute(members, mu0, None, screw_state=st,
                            screw_records=[])
    check("S38 a zero-omega screw -> UNOBSERVABLE",
          r["status"] == UNOBSERVABLE, r["status"])
    check("S38 and the reason names observability, not a threshold",
          "threshold" not in r["reason"].lower()
          and "observable" in r["reason"].lower())

    # Rank deficiency alone must also block branch A, even with omega != 0.
    part2 = initialise_from_twist(
        members, mu0, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                                   dtype=torch.float64),
        torch.linspace(0, 0.4, 5), None)
    st2 = JointState([part2], torch.arange(5, dtype=torch.float64))

    class _Rec:
        accepted = 1
        observability = {"geometry_information_eigenvalues":
                         [1.0, 1.0, 0.0, 0.0]}

    r = initialize_revolute(members, mu0, None, screw_state=st2,
                            screw_records=[_Rec()])
    check("S38 omega != 0 but rank-deficient geometry -> UNOBSERVABLE",
          r["status"] == UNOBSERVABLE, r["status"])

    class _Full(_Rec):
        observability = {"geometry_information_eigenvalues":
                         [2.0, 1.5, 1.0, 0.8]}

    r = initialize_revolute(members, mu0, None, screw_state=st2,
                            screw_records=[_Full()])
    check("S38 omega != 0 and full-rank geometry -> FITTED from branch A",
          r["status"] == FITTED
          and r["initialization_source"] == "A_generic_screw",
          f"{r['status']} / {r.get('initialization_source')}")


# ---------------------------------------------------------------------------
# Section 3: orientation and covariance transport are really exercised
# ---------------------------------------------------------------------------

def test_S3_orientation_transport_is_a_real_gate():
    g = torch.Generator().manual_seed(4)
    mu0 = torch.randn((200, 3), generator=g, dtype=torch.float64)
    R0 = torch.linalg.qr(torch.randn((200, 3, 3), generator=g,
                                     dtype=torch.float64))[0]
    sc = torch.rand((200, 3), generator=g, dtype=torch.float64) * 0.1 + 0.02
    part = initialise_from_twist(
        torch.arange(200), mu0,
        torch.tensor([0.1, 0.2, -0.05, 0.3, -0.1, 0.2], dtype=torch.float64),
        torch.linspace(0, 1, 5), None)
    out = rigidity_check(part, 3, mu0, R0, scales=sc)
    check("S3 pairwise distances preserved",
          out["max_pair_distance_drift"] < 1e-12,
          f"{out['max_pair_distance_drift']:.2e}")
    check("S3 orientation transport R_i(t) = R_t R_i^0 verified",
          out["orientation_checked"]
          and out["max_orientation_transport_error"] < 1e-12,
          f"{out['max_orientation_transport_error']:.2e}")
    check("S3 covariance transport Sigma_i(t) = R_t Sigma_i^0 R_t^T verified",
          out["covariance_checked"]
          and out["max_covariance_transport_error"] < 1e-12,
          f"{out['max_covariance_transport_error']:.2e}")
    raised = False
    try:
        rigidity_check(part, 3, mu0, None)
    except ValueError:
        raised = True
    check("S3 a missing R0 is an ERROR, not a silent skip", raised)


# ---------------------------------------------------------------------------
# Section 6: geometry scores decompose per frame AND per camera
# ---------------------------------------------------------------------------

def test_S6_per_frame_and_per_camera_scores():
    """Two cameras pulling in exactly opposite directions. The per-frame sum
    is identically zero and looks healthy; only the per-camera level shows
    the disagreement. This is the Blade failure mode."""
    cells = {(0, 0): torch.tensor([1.0, 0.0]), (0, 1): torch.tensor([-1.0, 0.0]),
             (1, 0): torch.tensor([1.0, 0.0]), (1, 1): torch.tensor([-1.0, 0.0])}
    F = {k: torch.eye(2, dtype=torch.float64) for k in cells}
    out = _aggregate_scores(cells, F, 2)
    check("S6 per-frame sums hide the disagreement",
          all(abs(r[-1]) < 1e-12 for r in out["rows_frame"]))
    check("S6 per-camera sums expose it",
          out["across_cameras"]["median_pairwise_cosine"] < -0.99
          and out["across_cameras"]["fraction_opposed"] == 1.0,
          f"cos {out['across_cameras']['median_pairwise_cosine']:.3f}, "
          f"opposed {out['across_cameras']['fraction_opposed']:.2f}")
    check("S6 all three aggregation levels are produced",
          bool(out["rows_frame_camera"]) and bool(out["rows_frame"])
          and bool(out["rows_camera"]))


# ---------------------------------------------------------------------------
# Section 28: structural -- no new optimizer
# ---------------------------------------------------------------------------

def test_S28_no_new_optimizer():
    import ast
    bad = []
    for f in ("phase2_prismatic.py", "phase2_revolute.py",
              "phase2_model_comparison.py"):
        src = (REPO / "estimator" / f).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in (
                    "backward", "Adam", "AdamW", "SGD", "LBFGS"):
                bad.append(f"{f}:{node.lineno} .{node.attr}")
            if isinstance(node, ast.Name) and node.id in ("Adam", "AdamW"):
                bad.append(f"{f}:{node.lineno} {node.id}")
        if "requires_grad" in src:
            bad.append(f"{f}: requires_grad")
    check("S28 no autograd and no external optimizer in the Stage-B models",
          not bad, "; ".join(bad))
    # The models must not re-render: only `run_window`/the bindings may.
    for f in ("phase2_prismatic.py", "phase2_revolute.py"):
        src = (REPO / "estimator" / f).read_text()
        check(f"S28 {f} contains no renderer code",
              "build_linearization" not in src and "rasteriz" not in src.lower())


def main():
    torch.manual_seed(0)
    for fn in (test_S20_S26_analytic_jacobians,
               test_S18_S22_constraints_are_exact,
               test_S3_orientation_transport_is_a_real_gate,
               test_S6_per_frame_and_per_camera_scores,
               test_S38_near_p_observability,
               test_S28_no_new_optimizer,
               test_S34_synthetic_prismatic,
               test_S35_synthetic_revolute,
               test_S36_wrong_initialization,
               test_S15_wrong_initialization_revolute):
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed")


if __name__ == "__main__":
    main()
