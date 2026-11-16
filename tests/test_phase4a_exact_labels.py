"""Gate C (directive 2026-11-11, section 8): the corrected exact Phase 4A on procedural P / R objects.

  * withheld moving-core members are recovered MOVING; no static Gaussian is rendered or decided moving; no
    moving Gaussian is decided static (R forward, P backward)
  * regression against the frozen pre-correction fixture (tests/fixtures/phase4_pre_correction): the exact
    scorer recovers at least as many withheld members and introduces no false moving
  * the exact objective never increases across commits; every batch's realised change equals its predicted
    change to float64 rounding (additivity of non-interacting / joint moves)
  * a legitimate group ON the hinge axis (identical S and M renders) stays UNRESOLVED, never STATIC
  * static controls (observations that never move; q identically 0) -> UPSTREAM_JOINT_INVALID, no labels
  * ids / counts immutable; the joint is never touched
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import recovery  # noqa: E402,F401
from recovery import phase4a_exact as X  # noqa: E402
from recovery.exact_objective import ExactLossConfig  # noqa: E402
from recovery.gaussians import GaussianTensors  # noqa: E402
from recovery.material_completion import Phase4Scene  # noqa: E402
from recovery.rgbd_loss import LossConfig  # noqa: E402
from recovery.view_cache import PreparedViews, ViewKey  # noqa: E402
from synthetic_articulated import make_scene, reobserve  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "phase4_pre_correction"
POL = X.Policy4A(sens_groups=4)
QUIET = lambda *a, **k: None


def run4a(s, model=None, pol=POL):
    model = model or s.model
    sc = Phase4Scene(model, s.joint, s.views, LossConfig(), sh_degree=0)
    keys = [ViewKey(v.cam.name, int(v.frame), v.cam.W, v.cam.H) for v in s.views]
    pv = PreparedViews.from_memory(keys, [v.get() for v in s.views], log=QUIET)
    ph = X.Phase4A(sc, pv, s.core, pol, ExactLossConfig(), log=QUIET)
    return ph, ph.run()


def _labels_case(jtype, direction, fixture):
    s = make_scene(jtype, direction, withhold_frac=0.4, seed=1)
    n0, ids0 = len(s.model), s.model.lineage["original_id"].copy()
    xyz0 = s.model._xyz.clone()
    ph, res = run4a(s)
    assert not res["aborted"] and res["complete"]
    st, lab = res["status"], res["render_label"].astype(bool)
    gt = s.gt_moving_model
    withheld = np.setdiff1d(np.nonzero(gt)[0], s.core)
    assert len(st) == n0 and np.array_equal(s.model.lineage["original_id"], ids0)
    assert torch.equal(s.model._xyz, xyz0), "4A changed canonical geometry"
    assert (st[s.core] == X.CORE).all() and lab[s.core].all()
    false_moving = int((lab & ~gt).sum()) + int(((st == X.MOVING_S) & ~gt).sum())
    wrong_static = int(((st == X.STATIC_S) & gt).sum())
    rec = float((st[withheld] == X.MOVING_S).mean())
    assert false_moving == 0, f"{jtype}: {false_moving} static Gaussians rendered / decided moving"
    assert wrong_static == 0, f"{jtype}: {wrong_static} moving Gaussians decided STATIC"
    assert rec >= 0.8, f"{jtype}: only {rec:.2f} of the withheld moving Gaussians recovered"
    # monotone, additive
    for h in res["history"][1:]:
        assert h["realised_change"] <= 1e-12, h
        assert abs(h["additivity_error"]) <= 1e-12 * max(1.0, abs(h["realised_change"])), h
    # regression vs the frozen pre-correction behaviour
    old = np.load(FIX / f"{fixture}.npz")
    rec_old = float((old["status"][withheld] == "ACCEPTED_MOVING").mean())
    assert int(((old["material"] == 1) & ~gt).sum()) == 0
    assert rec >= rec_old - 1e-9, f"recovery regressed: exact {rec:.2f} < pre-correction {rec_old:.2f}"
    E_gt = None
    print(f"  {jtype}/{direction}: withheld {len(withheld)} recovered {rec:.2f} (pre-correction {rec_old:.2f}); "
          f"false moving 0; moving->static 0; statuses "
          f"{dict(zip(*np.unique(st.astype(str), return_counts=True)))}; batches {res['counters']['batches']}, "
          f"{res['seconds']:.1f} s; E {res['history'][0]['E']:.5f} -> {res['history'][-1]['E']:.5f}")
    return s, res


def test_revolute_forward():
    _labels_case("revolute", "forward", "revolute_forward")


def test_prismatic_backward():
    _labels_case("prismatic", "backward", "prismatic_backward")


def test_on_axis_group_stays_unresolved():
    s = make_scene("revolute", "forward", withhold_frac=0.4, seed=1)
    g = s.model
    x0, z0 = -0.20, 2.0                                    # hinge line of make_scene: x = x0, z = z0, along y
    ys = np.linspace(0.22, 0.30, 6)                        # beyond the body (|y| <= 0.15): its own group
    n = len(ys)
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32, device="cuda")
    add = GaussianTensors(t(np.stack([np.full(n, x0), ys, np.full(n, z0)], 1)), t(np.tile([1.0, 0, 0, 0], (n, 1))),
                          torch.log(t(np.full((n, 3), 0.006))), torch.logit(t(np.full((n, 1), 0.9))),
                          t(np.full((n, 1, 3), 0.5)), torch.zeros(n, g._features_rest.shape[1], 3, device="cuda"),
                          g.max_sh_degree, {"material": np.zeros(n, np.int8),
                                            "gaussian_uid": np.arange(len(g), len(g) + n, dtype=np.int64),
                                            "parent_uid": np.full(n, -1, np.int64),
                                            "original_id": np.arange(len(g), len(g) + n, dtype=np.int64),
                                            "provenance": np.ones(n, np.int8)})
    m = GaussianTensors.concat(g, add)
    m.active_sh_degree = 0
    # the axis Gaussians are REAL (observed) static geometry: re-render the observations with them
    full = GaussianTensors.concat(s.full, add.subset(np.arange(n)))
    full.active_sh_degree = 0
    s.views = reobserve(s, full, np.concatenate([np.isin(s.surface_full, ["door_front", "door_back"]), np.zeros(n, bool)]))
    ph, res = run4a(s, model=m)
    on = np.arange(len(g), len(g) + n)
    st = res["status"][on]
    assert (st == X.UNOBS).all(), st
    assert not (st == X.STATIC_S).any() and not (st == X.MOVING_S).any()
    gid = res["group_of"][on[0]]
    reason = [x.reason for x in ph.groups if x.gid == gid][0]
    print(f"  on-axis group: {st[0]} ({reason}); rendered with the provisional label "
          f"{int(res['render_label'][on[0]])}, not a decision")


def test_resume_after_resource_stop_equals_uninterrupted():
    import tempfile
    s = make_scene("prismatic", "backward", withhold_frac=0.4, seed=1)
    ref_ph, ref = run4a(s)
    with tempfile.TemporaryDirectory() as tmp:
        pol1 = X.Policy4A(sens_groups=4, max_batches=1)
        sc = Phase4Scene(s.model, s.joint, s.views, LossConfig(), sh_degree=0)
        keys = [ViewKey(v.cam.name, int(v.frame), v.cam.W, v.cam.H) for v in s.views]
        pv = PreparedViews.from_memory(keys, [v.get() for v in s.views], log=QUIET)
        ph1 = X.Phase4A(sc, pv, s.core, pol1, ExactLossConfig(), log=QUIET, checkpoint_dir=tmp)
        r1 = ph1.run()
        assert not r1["complete"] and "max_batches" in r1["stop_reason"]
        ph2 = X.Phase4A(sc, pv, s.core, POL, ExactLossConfig(), log=QUIET, checkpoint_dir=tmp)
        r2 = ph2.run(resume=True)
        assert r2["complete"] and getattr(ph2, "resumed_from", None)
        assert np.array_equal(r2["render_label"], ref["render_label"])
        assert (r2["status"] == ref["status"]).all()
        print(f"  resume: stopped after batch 1 (INCOMPLETE), resumed -> identical labels / statuses to the "
              f"uninterrupted run ({ref['counters']['batches']} batches)")


def test_static_controls():
    s = make_scene("revolute", "forward", withhold_frac=0.4, seed=2, observe_q=[0.0] * 8)
    _, res = run4a(s)
    assert res["aborted"] and (res["status"] == X.JOINT_INVALID).all()
    assert res["render_label"].sum() == len(s.core)
    s = make_scene("prismatic", "forward", withhold_frac=0.4, seed=3, q_max=0.0)
    _, res = run4a(s)
    assert res["aborted"] and res["core_check"]["decision"] in (X.UNRESOLVED, X.UNOBS)
    print("  static controls: UPSTREAM_JOINT_INVALID (object never moves) and core UNRESOLVED (q = 0) -> abort")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
