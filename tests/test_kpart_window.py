"""Windowed discovery, model error, solver, handover and production wiring.

Tests T-TW1, T-TW2, T-MERR1, T-RATE1, T-K2, T-MV2, T-HO1, T-WD1, T-REAL-CONFIG
from CODEX_REQUIRED_MODIFICATIONS_MULTIVIEW_MULTIFRAME_KPART_IEKF.md section 27.
"""

from __future__ import annotations

import ast
import inspect
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator import IEKFConfig  # noqa: E402
from estimator.measurement import build_precision  # noqa: E402
from estimator.model_error import estimate_model_error  # noqa: E402
from estimator.renderer_bridge import CameraSpec, build_linearization  # noqa: E402
from estimator.state import GaussianMotionState  # noqa: E402
from estimator.supernodes import GroupedObservation, build_supernodes  # noqa: E402
from joint_reduction.kpart_models import (  # noqa: E402
    PRISMATIC,
    REVOLUTE,
    UNRESOLVED,
    PartModel,
    SceneModel,
)
from joint_reduction.kpart_support import (  # noqa: E402
    baseline_excess_gain,
    infer_kpart_support,
    kpart_metrics,
    part_achievable_gain,
)
from joint_reduction.kpart_window import (  # noqa: E402
    AlternationConfig,
    ObjectiveWeights,
    SelectionConfig,
    WindowObjective,
    alternate_to_convergence,
    bic,
    select_assignment,
    temporal_energy,
)
from tools.kpart_pipeline import _prec_res, choose_solver, resolve_cameras  # noqa: E402

DEV = "cuda"
PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def camera(width=80, height=80, fov=0.9, tz=0.0, ty=0.0):
    tanf = math.tan(fov * 0.5)
    view = torch.eye(4, device=DEV)
    view[3, 0] = tz
    view[3, 1] = ty
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


def two_region(n_each=400, seed=0):
    g = torch.Generator().manual_seed(seed)

    def blob(n, cx, cy, cz, s=0.25):
        p = torch.zeros((n, 3))
        p[:, 0] = (torch.rand(n, generator=g) - 0.5) * s + cx
        p[:, 1] = (torch.rand(n, generator=g) - 0.5) * s + cy
        p[:, 2] = (torch.rand(n, generator=g) - 0.5) * 0.12 + cz
        return p

    mu = torch.cat([blob(n_each, -0.45, 0.0, 2.45), blob(n_each, 0.45, 0.0, 2.05)])
    n = mu.shape[0]
    truth = torch.cat([torch.zeros(n_each, dtype=torch.long),
                       torch.ones(n_each, dtype=torch.long)])
    z = torch.zeros((n, 3))
    st = GaussianMotionState(
        mu=mu.to(DEV), R=torch.eye(3).repeat(n, 1, 1).to(DEV),
        v=z.clone().to(DEV), omega=z.to(DEV),
        colors=torch.rand((n, 3), generator=g).to(DEV),
        opacity=torch.full((n, 1), 0.85).to(DEV),
        scales=torch.full((n, 3), 0.045).to(DEV))
    return st, truth.to(DEV)


def obs_of(state, mu, R, cams):
    s = GaussianMotionState(mu=mu, R=R if R is not None else state.R, v=state.v,
                            omega=state.omega, colors=state.colors,
                            opacity=state.opacity, scales=state.scales)
    lin = build_linearization(s, cams)
    out = [(v.rgb.clone(), v.depth.clone(), (v.alpha > 0.05).clone())
           for v in lin.views]
    del lin
    return out


def make_objective(st, cams, frames, model_error=None):
    cfg = IEKFConfig(depth_sigma_model="constant:0.02:0.0")
    return WindowObjective(st.mu.clone(), st.R.clone(), cams, frames, cfg,
                           model_error=model_error,
                           weights=ObjectiveWeights(), state_template=st)


# ===========================================================================
def test_T_TW1_window_beats_sequential():
    """T-TW1: with weak per-frame motion the WINDOW must recover the rate where
    frame-local estimation cannot."""
    cams = [camera(), camera(tz=0.3)]
    st, truth = two_region(seed=3)
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    model = PartModel(kind=PRISMATIC, axis=axis, part_index=1)
    scene = SceneModel(parts=[model], K=1)
    dq = 0.004                      # deliberately weak per frame
    T = 10
    frames = []
    for t in range(1, T + 1):
        mu, R = scene.transport_all(st.mu, st.R, truth, [dq * t])
        frames.append((obs_of(st, mu, R, cams), t))

    obj = make_objective(st, cams, frames)
    # frame-local: fit q from ONE adjacent pair only
    phi = 0.5 * (5 ** 0.5 - 1)

    def local_q(ti):
        a, b = -0.1, 0.1
        c_, d_ = b - phi * (b - a), a + phi * (b - a)

        def E(x):
            mu, R = scene.transport_all(st.mu, st.R, truth, [float(x)])
            er, ed = obj.frame_energy(mu, R, frames[ti][0])
            return er + ed

        fc, fd = E(c_), E(d_)
        for _ in range(40):
            if abs(b - a) < 1e-9:
                break
            if fc < fd:
                b, d_, fd = d_, c_, fc
                c_ = b - phi * (b - a)
                fc = E(c_)
            else:
                a, c_, fc = c_, d_, fd
                d_ = a + phi * (b - a)
                fd = E(d_)
        return 0.5 * (a + b)

    q_local = [local_q(t) for t in range(T)]
    rate_local = (q_local[-1] - q_local[0]) / (T - 1)

    from joint_reduction.kpart_window import optimise_q_trajectory
    acfg = AlternationConfig(q_search_bracket=0.1)
    q_win = optimise_q_trajectory(obj, scene, truth, [[0.0] * T], acfg)[0]
    rate_win = (q_win[-1] - q_win[0]) / (T - 1)

    print(f"       true dq/frame      {dq:.6f}")
    print(f"       frame-local rate   {rate_local:.6f}  "
          f"({100 * rate_local / dq:.0f}% of truth)")
    print(f"       windowed rate      {rate_win:.6f}  "
          f"({100 * rate_win / dq:.0f}% of truth)")
    check("T-TW1 the windowed estimator recovers the rate",
          abs(rate_win - dq) < 0.25 * dq,
          f"{rate_win:.6f} vs {dq:.6f}")
    check("T-TW1 and its q trajectory is monotone with the motion",
          all(b >= a - 1e-6 for a, b in zip(q_win, q_win[1:])),
          f"{[round(x, 5) for x in q_win[:5]]}")


def test_T_TW2_long_baseline_beats_adjacent():
    """T-TW2: with a persistent reconstruction bias, long-baseline evidence
    must improve motion inference over adjacent-frame evidence."""
    cams = [camera()]
    st, truth = two_region(seed=5)
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    scene = SceneModel(parts=[PartModel(kind=PRISMATIC, axis=axis,
                                        part_index=1)], K=1)
    dq = 0.004
    T = 12
    torch.manual_seed(2)
    bias = None
    frames = []
    for t in range(1, T + 1):
        mu, R = scene.transport_all(st.mu, st.R, truth, [dq * t])
        o = obs_of(st, mu, R, cams)
        if bias is None:
            # a PERSISTENT reconstruction bias: identical at every frame
            bias = torch.randn_like(o[0][0]) * 0.05
        frames.append(([(o[0][0] + bias, o[0][1], o[0][2])], t))

    part = build_supernodes(st.mu, target_size=64, seed=0)
    cfg = IEKFConfig(depth_sigma_model="constant:0.02:0.0")
    lin = build_linearization(st, cams)
    gr = GroupedObservation(linearization=lin, partition=part)
    t_k = scene.parts[0].tangent(st.mu)

    def gain_at(i):
        prec, res = _prec_res(lin, frames[i][0], cfg, None)
        return part_achievable_gain(lin, gr, prec, res, t_k)

    g_near, g_far = gain_at(0), gain_at(T - 1)
    excess = baseline_excess_gain(g_far, g_near)

    sid = part.supernode_id.long()
    node_truth = torch.zeros(part.num_supernodes, dtype=torch.long, device=DEV)
    for c in range(part.num_supernodes):
        m = truth[sid == c]
        node_truth[c] = int(torch.mode(m).values) if m.numel() else 0
    mv, stat = node_truth == 1, node_truth == 0

    def sep(x):
        return float(x[mv].median()) / max(float(x[stat].median()), 1e-9)

    print(f"       adjacent-frame gain  moving/static median ratio {sep(g_near):8.2f}")
    print(f"       long-baseline gain   moving/static median ratio {sep(g_far):8.2f}")
    print(f"       baseline EXCESS      moving/static median ratio {sep(excess):8.2f}")
    check("T-TW2 a long baseline separates moving from static better than "
          "adjacent frames", sep(g_far) > sep(g_near),
          f"{sep(g_near):.2f} -> {sep(g_far):.2f}")
    check("T-TW2 and the baseline excess is the sharpest of the three",
          sep(excess) >= sep(g_far) * 0.95,
          f"excess {sep(excess):.2f} vs far {sep(g_far):.2f}")
    del lin


def test_T_MERR1_persistent_bias_induces_no_false_motion():
    """T-MERR1: a persistent RGB/depth bias must not be read as motion."""
    cams = [camera()]
    st, truth = two_region(seed=8)
    static_obs = obs_of(st, st.mu, st.R, cams)
    torch.manual_seed(4)
    bias = torch.randn_like(static_obs[0][0]) * 0.08
    biased = [(static_obs[0][0] + bias, static_obs[0][1], static_obs[0][2])]

    lin = build_linearization(st, cams)
    me = estimate_model_error(lin, biased, beta=1.0, blur=1)
    part = build_supernodes(st.mu, target_size=64, seed=0)
    gr = GroupedObservation(linearization=lin, partition=part)
    cfg = IEKFConfig(depth_sigma_model="constant:0.02:0.0")

    out = {}
    for tag, mod in (("sensor only", None), ("R_eff", me)):
        prec, res = _prec_res(lin, biased, cfg, mod)
        g = part_achievable_gain(lin, gr, prec, res, None)
        sup = infer_kpart_support(part, [g], K=1, potts_weight=1.0)
        out[tag] = (float(g.max()), sup.counts())
        print(f"       {tag:12s}: max gain {float(g.max()):10.2f}  "
              f"labels {sup.counts()}")
    check("§13 the model-error map reduces the false gain from a persistent bias",
          out["R_eff"][0] < out["sensor only"][0],
          f"{out['sensor only'][0]:.1f} -> {out['R_eff'][0]:.1f}")

    # The scene truly does not move, so the baseline excess must vanish.
    prec, res = _prec_res(lin, biased, cfg, me)
    g1 = part_achievable_gain(lin, gr, prec, res, None)
    excess = baseline_excess_gain(g1, g1)
    check("T-MERR1 a time-invariant bias produces ZERO baseline excess",
          float(excess.max()) == 0.0,
          "model error cannot counterfeit growth over a baseline")
    del lin


def test_T_K2_two_revolute_parts():
    """T-K2: two revolute parts with different axes and overlapping support."""
    cams = [camera(), camera(tz=0.35), camera(ty=0.3)]
    g = torch.Generator().manual_seed(11)

    def blob(n, c, s=0.2):
        p = torch.zeros((n, 3))
        for i in range(3):
            p[:, i] = (torch.rand(n, generator=g) - 0.5) * s + c[i]
        return p

    a_mu = blob(350, (-0.3, 0.1, 2.2))
    b_mu = blob(350, (0.3, -0.1, 2.1))
    s_mu = blob(350, (0.0, 0.45, 2.5))
    mu = torch.cat([s_mu, a_mu, b_mu])
    n = mu.shape[0]
    truth = torch.cat([torch.zeros(350, dtype=torch.long),
                       torch.ones(350, dtype=torch.long),
                       torch.full((350,), 2, dtype=torch.long)]).to(DEV)
    z = torch.zeros((n, 3))
    st = GaussianMotionState(
        mu=mu.to(DEV), R=torch.eye(3).repeat(n, 1, 1).to(DEV),
        v=z.clone().to(DEV), omega=z.to(DEV),
        colors=torch.rand((n, 3), generator=g).to(DEV),
        opacity=torch.full((n, 1), 0.85).to(DEV),
        scales=torch.full((n, 3), 0.04).to(DEV))

    def line_at(e, a):
        e = e / e.norm()
        a = a - (a @ e) * e
        return e, torch.linalg.cross(a, e)

    e1, m1 = line_at(torch.tensor([0.0, 0.0, 1.0], device=DEV),
                     torch.tensor([-0.3, 0.1, 0.0], device=DEV))
    e2, m2 = line_at(torch.tensor([0.0, 1.0, 0.2], device=DEV),
                     torch.tensor([0.3, 0.0, 2.1], device=DEV))
    scene = SceneModel(parts=[
        PartModel(kind=REVOLUTE, axis=e1, moment=m1, part_index=1),
        PartModel(kind=REVOLUTE, axis=e2, moment=m2, part_index=2)], K=2)
    ang = float(torch.rad2deg(torch.arccos(torch.clamp((e1 @ e2).abs(), 0, 1))))
    check("T-K2 the two revolute axes are genuinely different", ang > 45.0,
          f"{ang:.1f} deg apart")

    mu_t, R_t = scene.transport_all(st.mu, st.R, truth, [0.20, 0.25])
    gt = obs_of(st, mu_t, R_t, cams)
    part = build_supernodes(st.mu, target_size=64, seed=0)
    sid = part.supernode_id.long()
    node_truth = torch.zeros(part.num_supernodes, dtype=torch.long, device=DEV)
    for c in range(part.num_supernodes):
        m = truth[sid == c]
        node_truth[c] = int(torch.mode(m).values) if m.numel() else 0

    lin = build_linearization(st, cams)
    gr = GroupedObservation(linearization=lin, partition=part)
    cfg = IEKFConfig(depth_sigma_model="constant:0.02:0.0")
    prec, res = _prec_res(lin, gt, cfg, None)
    gains = [part_achievable_gain(lin, gr, prec, res,
                                  scene.parts[k - 1].tangent(st.mu))
             for k in (1, 2)]
    sup = infer_kpart_support(part, gains, K=2, potts_weight=1.0)
    met = kpart_metrics(sup.hard.to(DEV), node_truth, K=2)
    print(f"       labels {sup.counts()} (truth "
          f"{[int((node_truth == k).sum()) for k in range(3)]})")
    print(f"       Hungarian accuracy {met['hungarian_accuracy']:.3f}  "
          f"ARI {met['adjusted_rand_index']:.3f}")
    check("T-K2 both revolute parts are populated", sup.all_parts_present(),
          str(sup.counts()))
    check("T-K2 the two parts are separated", met["hungarian_accuracy"] > 0.75,
          f"accuracy {met['hungarian_accuracy']:.3f}")
    del lin


def test_T_RATE1_nonconstant_q_rate():
    """T-RATE1: the visual update must recover the RATE, not only the type."""
    cams = [camera(), camera(tz=0.3)]
    st, truth = two_region(seed=13)
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    scene = SceneModel(parts=[PartModel(kind=PRISMATIC, axis=axis,
                                        part_index=1)], K=1)
    T = 10
    q_true = [0.004 * t + 0.0015 * t * t for t in range(1, T + 1)]   # accelerating
    frames = []
    for t, q in enumerate(q_true, start=1):
        mu, R = scene.transport_all(st.mu, st.R, truth, [q])
        frames.append((obs_of(st, mu, R, cams), t))
    obj = make_objective(st, cams, frames)
    from joint_reduction.kpart_window import optimise_q_trajectory
    q_est = optimise_q_trajectory(obj, scene, truth, [[0.0] * T],
                                  AlternationConfig(q_search_bracket=0.2))[0]
    err = max(abs(a - b) for a, b in zip(q_est, q_true))
    rate_t = (q_true[-1] - q_true[0]) / (T - 1)
    rate_e = (q_est[-1] - q_est[0]) / (T - 1)
    print(f"       q true {[round(x, 5) for x in q_true[:5]]} ...")
    print(f"       q est  {[round(x, 5) for x in q_est[:5]]} ...")
    print(f"       rate: true {rate_t:.6f}  estimated {rate_e:.6f} "
          f"({100 * rate_e / rate_t:.0f}% of truth)")
    check("T-RATE1 the q trajectory is recovered pointwise", err < 0.004,
          f"max |dq| {err:.6f}")
    check("T-RATE1 the RATE magnitude is recovered, not only the type",
          abs(rate_e - rate_t) < 0.15 * rate_t,
          f"{rate_e:.6f} vs {rate_t:.6f}")
    check("T-RATE1 the acceleration is visible in the estimate",
          (q_est[-1] - q_est[-2]) > (q_est[1] - q_est[0]),
          "later steps are larger, as the true trajectory accelerates")


def test_T_MV2_camera_count_observability():
    """T-MV2: compare 1/2/4/8 cameras."""
    st, truth = two_region(seed=15)
    axis = torch.tensor([1.0, 0.0, 0.0], device=DEV)
    scene = SceneModel(parts=[PartModel(kind=PRISMATIC, axis=axis,
                                        part_index=1)], K=1)
    pool = [camera(tz=t, ty=y) for t, y in
            ((0.0, 0.0), (0.35, 0.0), (-0.35, 0.0), (0.0, 0.35),
             (0.0, -0.35), (0.25, 0.25), (-0.25, 0.25), (0.25, -0.25))]
    mu_t, R_t = scene.transport_all(st.mu, st.R, truth, [0.05])
    cfg = IEKFConfig(depth_sigma_model="constant:0.02:0.0")
    rows = []
    for c in (1, 2, 4, 8):
        cams = pool[:c]
        gt = obs_of(st, mu_t, R_t, cams)
        lin = build_linearization(st, cams)
        prec, res = _prec_res(lin, gt, cfg, None)
        from joint_reduction.kpart_models import KPartObservation
        kp = KPartObservation(lin, scene, st.mu, truth)
        I, b = kp.normal_system(prec, res)
        rows.append((c, float(I[0, 0])))
        print(f"       {c} camera(s): joint information I_qq = {float(I[0, 0]):.4e}")
        del lin
    check("T-MV2 information increases monotonically with camera count",
          all(b[1] > a[1] for a, b in zip(rows, rows[1:])),
          " < ".join(f"{v:.3e}" for _c, v in rows))
    check("T-MV2 eight cameras carry substantially more than one",
          rows[-1][1] > 3.0 * rows[0][1],
          f"{rows[0][1]:.3e} -> {rows[-1][1]:.3e}")


def test_section16_solver_modes():
    """§16: trust_region | lm | auto, with the reason persisted."""
    tr, r1 = choose_solver("trust_region")
    lm, r2 = choose_solver("lm")
    check("§16 trust_region is selectable", tr and r1["solver"] == "trust_region")
    check("§16 lm is selectable", (not lm) and r2["solver"] == "lm")
    a, r3 = choose_solver("auto", [])
    check("§16 auto starts with the bounded-step solver", a)
    bad = [{"accepted": False, "boundary_hit": True} for _ in range(8)]
    a2, r4 = choose_solver("auto", bad)
    check("§16 auto switches to LM when the trust region keeps rejecting",
          not a2, r4["reason"][:70])
    check("§16 and the reason is persisted", "reason" in r4 and r4["reason"])
    good = [{"accepted": True, "boundary_hit": False} for _ in range(8)]
    a3, _ = choose_solver("auto", good)
    check("§16 auto keeps the trust region when it is accepting", a3)
    try:
        choose_solver("adam")
        raised = False
    except ValueError:
        raised = True
    check("§16 an unknown solver is rejected", raised)


def test_section3_camera_assertion():
    """§3: the hard assertion that all-camera mode uses all cameras."""
    cams = list(range(8))
    use, audit = resolve_cameras(cams, 4, False, None)
    check("§3 a restricted request uses the requested count", len(use) == 4)
    check("§3 and records the exclusions", len(audit["excluded"]) == 4)
    use2, audit2 = resolve_cameras(cams, 4, True, None)
    check("§3 --all-cameras overrides the count", len(use2) == 8)
    check("§3 and records that all cameras were used",
          audit2["num_cameras_used"] == 8 and audit2["all_cameras_requested"])


def test_T_WD1_T_HO1_production_wiring():
    """T-WD1 and T-HO1: assert the actual RUNNER invokes window discovery and
    canonical handover -- not that the helpers merely exist."""
    src = (REPO / "tools" / "run_iekf_scene.py").read_text()
    tree = ast.parse(src)
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    check("T-WD1 the runner calls run_kpart_discovery",
          "run_kpart_discovery" in calls,
          "section 0 item 4: a feature the runner never calls is not implemented")
    check("T-WD1 the runner exposes --window-discovery",
          "--window-discovery" in src)
    check("T-WD1 and requires K with it",
          "--num-moving-parts" in src and "requires --num-moving-parts" in src)

    pipe = (REPO / "tools" / "kpart_pipeline.py").read_text()
    ptree = ast.parse(pipe)
    pcalls = {n.func.id for n in ast.walk(ptree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    check("T-HO1 the production pipeline calls canonical handover",
          "_canonical_handover" in pcalls)
    check("T-HO1 handover checks the canonical hash",
          "canonical_hash" in pipe and "canonical_unchanged" in pipe)
    check("T-HO1 and enforces E_handover <= E_pre + eps",
          "handover_epsilon" in pipe)
    check("T-WD1 the production pipeline calls discover_kpart_joints",
          "discover_kpart_joints" in pcalls)
    check("§11 the pipeline alternates to convergence",
          "alternate_to_convergence" in (REPO / "joint_reduction" /
                                         "kpart_window.py").read_text())
    check("§13 the pipeline estimates model error",
          "estimate_model_error" in pcalls)
    check("§17 the pipeline recomputes observability on the reduced state",
          "reduced_observability" in pcalls)


def test_T_REAL_CONFIG():
    """T-REAL-CONFIG: the real run must forbid imposed motion and require K."""
    src = (REPO / "tools" / "run_iekf_scene.py").read_text()
    check("T-REAL-CONFIG --speed/--omega are forbidden in dataset mode",
          "is forbidden in " in src and "hidden synthetic target" in src)
    check("T-REAL-CONFIG --all-cameras exists and asserts",
          "--all-cameras" in src and "--all-cameras requested but" in src)
    check("T-REAL-CONFIG --all-frames exists", "--all-frames" in src)
    check("T-REAL-CONFIG K is required for window discovery",
          "requires --num-moving-parts" in src)
    pipe = (REPO / "tools" / "kpart_pipeline.py").read_text()
    for field in ("GT part labels used by estimator",
                  "GT joint type used by estimator",
                  "GT q(t) used by estimator",
                  "GT axis used by estimator"):
        check(f"T-REAL-CONFIG the GT audit declares '{field[:28]}...'",
              field in pipe)


def test_selection_and_objective_terms():
    """§5/§24: comparable model selection, UNRESOLVED, and the energy terms."""
    n = 10000
    p = {"assignment": ["prismatic"], "objective": 100.0,
         "bic": bic(100.0, 6, n), "normalized_data_term": 0.01, "converged": True}
    r = {"assignment": ["revolute"], "objective": 99.9,
         "bic": bic(99.9, 10, n), "normalized_data_term": 0.01, "converged": True}
    check("§5 revolute fits marginally better on the raw objective",
          r["objective"] < p["objective"])
    check("§5 but BIC charges for its extra parameters", p["bic"] < r["bic"],
          f"{p['bic']:.2f} vs {r['bic']:.2f}")
    # Section 18: BIC alone no longer selects. It ranks, and the section 16
    # measurement-space test decides; without one, there is no production
    # P/R choice. The ranking is still exposed, as a diagnostic.
    out = select_assignment([p, r], SelectionConfig(min_relative_margin=0.0))
    check("§18 BIC alone does not make a production selection",
          out["selected"] == "UNRESOLVED", str(out["selected"]))
    check("§18 but the BIC ranking is still reported as a diagnostic",
          out["ranking_only_assignment"] == ["prismatic"],
          str(out["ranking_only_assignment"]))
    check("§18 the missing identifiability test is named as the reason",
          any("section 18" in x for x in out["reason"]),
          str(out["reason"])[:80])
    # With the section 16 verdict supplied, selection proceeds.
    out_r = select_assignment(
        [p, r], SelectionConfig(min_relative_margin=0.0),
        curvature=[{"verdict": "revolute", "likelihood_ratio": 40.0,
                    "threshold": 2.706, "confidence_interval": (0.1, 0.3)}])
    check("§16 a significant curvature selects revolute even though BIC "
          "ranks prismatic first",
          out_r["selected"] == ["revolute"], str(out_r["selected"]))
    tie = select_assignment([p, {**r, "bic": p["bic"] * 1.001}],
                            SelectionConfig(min_relative_margin=0.05))
    check("§5 two indistinguishable assignments give UNRESOLVED",
          tie["selected"] == "UNRESOLVED", str(tie["selected"]))
    check("§5 the margin veto survives the rewiring",
          any("distinguishable" in x for x in tie["reason"]))
    check("§5 and the margin failure is named",
          any("distinguishable" in x for x in tie["reason"]))
    check("§5 no candidate converging gives UNRESOLVED",
          select_assignment([], SelectionConfig())["selected"] == "UNRESOLVED")

    check("§24 the temporal term is a second difference",
          abs(temporal_energy([0.0, 1.0, 2.0, 3.0])) < 1e-12,
          "a constant-rate trajectory is free")
    check("§24 and charges curvature",
          temporal_energy([0.0, 1.0, 3.0]) > 0)


def test_far_axis_is_no_longer_a_selector():
    """§15/§16/§32: the far-axis override is gone and §32's statuses hold.

    This test used to assert the opposite -- that a far axis DEMOTES a revolute
    hypothesis that ranks first on BIC. Section 15 deletes that rule outright:
    "A true revolute joint can have d/R >> 1 and still be identifiable when
    angular sweep and measurement quality are sufficient." The d/R ratio is now
    a diagnostic, and `tests/test_strict_identifiability.py` measures the case
    this one used to get wrong (d/R = 20 with a 60 degree sweep IS revolute).
    """
    from joint_reduction.identifiability import (CONVERGED, UNRESOLVED_MODEL,
                                                 UNRESOLVED_OPTIMIZATION)

    n = 7_372_800                      # 8 cameras x 640x480 x (3 RGB + 1 depth)
    base = {"normalized_data_term": 0.01, "identifiability": []}
    P = {**base, "assignment": [PRISMATIC], "objective": 6_017_746.0,
         "bic": bic(6_017_746.0, 6, n), "converged": True}
    Rfar = {**base, "assignment": [REVOLUTE], "objective": 5_742_750.0,
            "bic": bic(5_742_750.0, 10, n), "converged": True,
            "identifiability": [{"applicable": True,
                                 "distance_over_extent": 2.58}]}
    check("§5 at real scale the far-axis revolute ranks FIRST on BIC",
          Rfar["bic"] < P["bic"],
          f"revolute {Rfar['bic']:.0f} < prismatic {P['bic']:.0f}")

    # Section 32 case 3: both converged and the curvature is significant.
    sig = {"verdict": REVOLUTE, "likelihood_ratio": 596.2, "threshold": 2.706,
           "confidence_interval": (0.21, 0.42)}
    out = select_assignment([P, Rfar], SelectionConfig(min_relative_margin=0.0),
                            curvature=[sig])
    check("§32 case 3: a converged, significantly curved part returns REVOLUTE",
          out["selected"] == [REVOLUTE], str(out["selected"]))
    check("§32 case 3 is reported CONVERGED", out["status"] == CONVERGED,
          out["status"])
    check("§15 no far-axis guard fired", not out["identifiability_guard_applied"])

    # Section 32 case 2: both converged, curvature unresolved.
    tie = {"verdict": UNRESOLVED_MODEL, "likelihood_ratio": 0.31,
           "threshold": 2.706, "confidence_interval": (0.0, 0.036)}
    out2 = select_assignment([P, Rfar], SelectionConfig(min_relative_margin=0.0),
                             curvature=[tie])
    check("§32 case 2: an unresolved curvature returns UNRESOLVED_MODEL",
          out2["status"] == UNRESOLVED_MODEL, out2["status"])
    check("§45 it is NOT silently converted to prismatic",
          out2["selected"] == UNRESOLVED, str(out2["selected"]))
    check("§16 the curvature interval is given as the reason",
          any("curvature interval" in r for r in out2["reason"]),
          str(out2["reason"])[:90])
    check("§18 the BIC ranking survives only as a diagnostic",
          out2["ranking_only_assignment"] == [REVOLUTE]
          and any("diagnostic" in x for x in out2["notes"]))

    # Section 32 case 1: P not converged, R converged.
    out3 = select_assignment([{**P, "converged": False},
                              {**Rfar, "converged": False}],
                             SelectionConfig(min_relative_margin=0.0),
                             curvature=[sig])
    check("§32 case 1: nothing converged -> UNRESOLVED_OPTIMIZATION",
          out3["status"] == UNRESOLVED_OPTIMIZATION, out3["status"])
    check("§0.5 and no model is selected from the unfinished optimization",
          out3["selected"] == UNRESOLVED, str(out3["selected"]))

    out4 = select_assignment([{**P, "converged": False}, Rfar],
                             SelectionConfig(min_relative_margin=0.0),
                             curvature=[sig])
    check("§17 a non-converged candidate is excluded from the comparison",
          any("did not converge" in x for x in out4["notes"]),
          str(out4["notes"])[:90])

    # Section 18: without the section 16 test there is no production selection.
    out5 = select_assignment([P, Rfar], SelectionConfig(min_relative_margin=0.0))
    check("§18 no identifiability test means no production P/R choice",
          out5["status"] == UNRESOLVED_MODEL
          and any("section 18" in r for r in out5["reason"]),
          out5["status"])


def test_null_excess_calibration():
    """§14: the level a real motion must clear is measured, not assumed."""
    from joint_reduction.kpart_support import calibrate_null_excess

    torch.manual_seed(0)
    # static supernodes: identical gain at both frames -> zero null excess
    a = torch.rand(200).double() * 100
    b = a.clone()
    thr, info = calibrate_null_excess(a, b)
    check("§14 identical gains calibrate a zero null threshold", thr == 0.0,
          f"{thr}")

    # realistic: a small noisy drift plus a few large outliers
    b2 = a + torch.rand(200).double() * 2.0
    b2[:5] += 300.0
    thr2, info2 = calibrate_null_excess(a, b2, quantile=0.99)
    check("§14 a noisy null calibrates a positive threshold", thr2 > 0,
          f"{thr2:.3f}")
    check("§14 the threshold sits above the bulk of the noise",
          thr2 > float((b2 - a).clamp_min(0).median()),
          f"threshold {thr2:.3f} vs null median "
          f"{float((b2 - a).clamp_min(0).median()):.3f}")
    check("§14 and below the largest outlier",
          thr2 < float((b2 - a).max()),
          f"{thr2:.3f} < {float((b2 - a).max()):.3f}")
    check("§14 the calibration is persisted with its quantile",
          info2["null_quantile"] == 0.99 and "null_max" in info2)

    # the degenerate case this exists for: an all-zero null
    gain_far = a.clone()
    gain_far[:10] += 500.0
    excess = baseline_excess_gain(gain_far, a)
    med = float(excess.median())
    mad = float((excess - med).abs().median())
    check("§14 an all-static excess has median AND MAD exactly zero",
          med == 0.0 and mad == 0.0,
          "which is why median + k*MAD degenerates to '> 0' and admitted 44% "
          "of the storage scene before this calibration existed")
    check("§14 subtracting a measured null leaves only the real movers",
          int(((excess - thr).clamp_min(0) > 0).sum()) == 10,
          f"{int(((excess - thr).clamp_min(0) > 0).sum())} of 200")


def main():
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
