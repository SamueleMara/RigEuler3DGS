"""Gate 3: Phase 4A on synthetic P and R scenes with deliberately withheld moving-core members.

  * the withheld members that the cameras see moving are recovered as ACCEPTED_MOVING
  * no static Gaussian becomes moving (no false moving static wall)
  * axis-near / never-displaced evidence stays UNRESOLVED, never silently static
  * the original count and IDs are immutable
  * the wrong-label counterfactual is visibly worse on the full-scene RGB-D-mask objective
  * static controls: observations that do not move, or a joint whose q is identically 0 -> the stage
    refuses (UPSTREAM_JOINT_INVALID) and labels nothing moving beyond the core
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
import recovery  # noqa: E402,F401
from recovery.material_completion import (ACCEPTED_MOVING, ACCEPTED_STATIC, CORE, UNRESOLVED,  # noqa: E402
                                          UPSTREAM_JOINT_INVALID, Phase4Scene, PolicyConfig, run_phase4a)
from recovery.rgbd_loss import LossConfig, view_loss  # noqa: E402
from synthetic_articulated import make_scene  # noqa: E402

POL = PolicyConfig(max_component=128, sens_max_views=8)


def scene4(s):
    return Phase4Scene(s.model, s.joint, s.views, LossConfig(), sh_degree=0)


def total_loss(sc, moving):
    return sum(float(view_loss(sc.render(v, moving), sc.view(v), sc.loss_cfg)["total"]) for v in sc.views)


def _run(jtype, direction):
    s = make_scene(jtype, direction, withhold_frac=0.4, seed=1)
    sc = scene4(s)
    n0, ids0 = len(s.model), s.model.lineage["original_id"].copy()
    res = run_phase4a(sc, s.core, POL, log=lambda *a: None)
    assert not res["aborted"], res["audit"]
    st, mat = res["status"], res["material"]
    gt = s.gt_moving_model
    withheld = np.setdiff1d(np.nonzero(gt)[0], s.core)
    assert len(mat) == n0 and np.array_equal(s.model.lineage["original_id"], ids0), "count / IDs changed"
    assert (st[s.core] == CORE).all() and (mat[s.core] == 1).all()
    false_moving = np.nonzero((mat == 1) & ~gt)[0]
    assert len(false_moving) == 0, f"{jtype}: {len(false_moving)} static Gaussians labelled moving " \
                                   f"({np.unique(s.surface_model[false_moving], return_counts=True)})"
    rec = (st[withheld] == ACCEPTED_MOVING).mean()
    unres_static = ((st == UNRESOLVED) & ~gt).sum()
    acc_static = ((st == ACCEPTED_STATIC) & ~gt).sum()
    wrong_static = ((st == ACCEPTED_STATIC) & gt).sum()
    print(f"  {jtype}/{direction}: withheld {len(withheld)}, recovered moving {rec:.2f}, "
          f"withheld UNRESOLVED {(st[withheld] == UNRESOLVED).mean():.2f}, moving called static {wrong_static}, "
          f"static: accepted {acc_static} / unresolved {unres_static} / false moving 0")
    assert rec >= 0.8, f"{jtype}: only {rec:.2f} of the withheld moving Gaussians recovered"
    assert wrong_static == 0, f"{jtype}: {wrong_static} moving Gaussians declared static"
    assert acc_static > 0.5 * (~gt).sum(), "most static Gaussians far from the axis should be decided static"
    # the wrong-label counterfactual is visibly worse
    dev = sc.device
    gt_t = torch.as_tensor(gt, device=dev)
    core_only = torch.zeros_like(gt_t)
    core_only[torch.as_tensor(s.core, device=dev)] = True
    all_static = torch.zeros_like(gt_t)
    L_gt, L_core, L_static = total_loss(sc, gt_t), total_loss(sc, core_only), total_loss(sc, all_static)
    L_res = total_loss(sc, torch.as_tensor(mat == 1, device=dev))
    print(f"    objective: GT labels {L_gt:.4f}, Phase-4A {L_res:.4f}, core only {L_core:.4f}, all static {L_static:.4f}")
    assert L_gt < L_core < L_static and L_res < L_core
    return res


def test_revolute_forward():
    _run("revolute", "forward")


def test_prismatic_backward():
    _run("prismatic", "backward")


def test_static_controls():
    # (a) the object never moves but the joint says it does
    s = make_scene("revolute", "forward", withhold_frac=0.4, seed=2, observe_q=[0.0] * 8)
    res = run_phase4a(scene4(s), s.core, POL, log=lambda *a: None)
    assert res["aborted"] and (res["status"] == UPSTREAM_JOINT_INVALID).all()
    assert (res["material"] == 1).sum() == len(s.core), "nothing beyond the core may be moving"
    # (b) a joint whose coordinate is identically 0: no informative view anywhere
    s = make_scene("prismatic", "forward", withhold_frac=0.4, seed=3, q_max=0.0)
    res = run_phase4a(scene4(s), s.core, POL, log=lambda *a: None)
    assert res["aborted"] and res["audit"]["core_check"]["decision"] == UNRESOLVED
    assert (res["material"] == 1).sum() == len(s.core)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
