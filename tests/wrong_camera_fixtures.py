"""Shared toy fixtures for the wrong-camera audit tests (CUDA)."""
from __future__ import annotations

import dataclasses

import torch

NAMES = ("cam_000", "cam_001", "cam_002")


def masked_scene(n=40, H=48, W=48, shift=(0.006, -0.004, 0.008)):
    """(cams, state, truth, prepared obs, raw gt, masks, model, cfg) on 3 cameras."""
    from estimator.manifold import so3_exp
    from estimator.masked_observation import MaskedObservationModel, prepare
    from estimator.sparse_iekf import IEKFConfig
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    from tests.test_visibility_constrained_iekf_phase1 import blob, observe
    cams = make_cameras(n_cam=3, H=H, W=W, radius=2.2)
    st = blob(n=n, seed=0)
    g = torch.Generator().manual_seed(7)
    st = dataclasses.replace(st, scales=torch.tensor([0.05, 0.03, 0.015]).repeat(n, 1).cuda(),
                             R=so3_exp(torch.randn(n, 3, generator=g)).float().cuda())
    truth = dataclasses.replace(st, mu=st.mu + torch.tensor(shift, device="cuda"))
    gt, masks = observe(truth, cams)
    obs = [prepare(a[0], a[1], a[2], m, NAMES[i], 1) for i, (a, m) in enumerate(zip(gt, masks))]
    model = MaskedObservationModel("masked-rgb-expected-quad", (0.5, 0.5, 0.5), "uniform01", 0.1)
    cfg = IEKFConfig(max_iters=4, pcg_max_iters=30, camera_names=NAMES)
    return cams, st, truth, obs, gt, masks, model, cfg


def frozen(st, cams, obs, model, cfg):
    """(lin, terms, prec, alpha_terms, rhs_pose, meas_normal) of the production operators at st."""
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import _frame_terms, measurement_operators
    lin = build_linearization(st, cams, eps_depth=cfg.eps_depth, background=model.background_tensor("cuda"))
    terms = _frame_terms(lin.views, obs, cfg, observation_model=model)
    prec = [t.as_prec4() for t in terms]
    alpha_terms = [(t.kappa, t.r_a) for t in terms]
    rhs, normal, fisher = measurement_operators(lin, prec, alpha_terms)
    return lin, terms, prec, alpha_terms, rhs, normal


def check_factory():
    state = {"n": 0, "fail": []}

    def check(label, cond, detail=""):
        state["n"] += 1
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
        if not cond:
            state["fail"].append(label)
    return check, state


def finish(state) -> int:
    print()
    if state["fail"]:
        print(f"{len(state['fail'])} of {state['n']} checks FAILED")
        return 1
    print(f"{state['n']} checks passed")
    return 0
