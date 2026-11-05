#!/usr/bin/env python
"""Gate D: first-return in the Phase-1 IEKF frame step (CUDA, toy scenes).

  [D1] mode off: identical to the legacy frame step (within the measured GPU floor)
  [D2] mode audit: the update is the legacy one; the first-return summary is recorded
  [D3] mode depth: no blended-depth energy remains; the first-return energy is in every
       evaluation; every ACCEPTED step lowers the full nonlinear objective (prior + RGB +
       alpha + first-return depth, fresh contributor order); the state moves toward the
       observation; the posterior is PSD and labelled POSTERIOR_UNCERTIFIED
  [D4] preconditions refused: no masked observation model; first-return + negative-ray
  [D5] bootstrap_positions threads the mode into the per-frame records and the manifest

    python tests/test_first_return_iekf.py
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CHECKS, FAILURES = 0, []
NAMES = ("cam_000", "cam_001", "cam_002")


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def setup():
    from estimator.manifold import so3_exp
    from estimator.masked_observation import MaskedObservationModel, prepare
    from estimator.sparse_iekf import IEKFConfig
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
    from tests.test_visibility_constrained_iekf_phase1 import blob, observe
    cams = make_cameras(n_cam=3, H=56, W=56, radius=2.2)
    st = blob(n=40, seed=0)
    g = torch.Generator().manual_seed(7)
    sc = torch.tensor([0.05, 0.03, 0.015]).repeat(40, 1)
    st = dataclasses.replace(st, scales=sc.cuda(), R=so3_exp(torch.randn(40, 3, generator=g)).float().cuda())
    truth = dataclasses.replace(st, mu=st.mu + torch.tensor([0.006, -0.004, 0.008], device="cuda"))
    gt, masks = observe(truth, cams)
    # a FIRST-RETURN-consistent depth: the centre depth of the most probable first hit
    # at the true state (the renderer's blended Z/(A+eps) is no single surface)
    from estimator import first_return_operators as fo
    from estimator.renderer_bridge import build_linearization
    lin_t = build_linearization(truth, cams)
    gt2 = []
    for (rgb, d, v), m, view in zip(gt, masks, lin_t.views):
        sd = torch.full_like(d, 0.004)
        dump = fo.fr_dump(fo.camera_args(view.cache, d, m, sd, sd, 0.0, 0.2, 6.0))
        off, z, al = dump["offsets"], dump["z"], dump["alpha"]
        D = torch.zeros_like(d).reshape(-1)
        for pix in torch.nonzero(m.reshape(-1)).reshape(-1).tolist():
            a, b = int(off[pix]), int(off[pix + 1])
            if b > a:
                aa = al[a:b].double()
                T = torch.cat([torch.ones(1, device=aa.device, dtype=aa.dtype), torch.cumprod(1 - aa, 0)[:-1]])
                D[pix] = z[a:b][int(torch.argmax(T * aa))]
        D = D.reshape(d.shape)
        gt2.append((rgb, D, m & (D > 0)))
    gt = gt2
    obs = [prepare(a[0], a[1], a[2], m, NAMES[i], 1) for i, (a, m) in enumerate(zip(gt, masks))]
    model = MaskedObservationModel("masked-rgb-expected-quad", (0.5, 0.5, 0.5), "uniform01", 0.1)
    cfg = IEKFConfig(max_iters=6, pcg_max_iters=30, camera_names=NAMES)
    return cams, st, truth, obs, model, cfg, gt, masks


def prior_file(td, lam=0.6):
    from estimator.first_return_types import load_prior, write_prior
    p = Path(td) / "prior.json"
    write_prior(p, {n: {"z_near_m": 0.2, "z_far_m": 6.0, "lambda_per_m": lam,
                        "unknown_depth_sigma_model": {"kind": "sensor"}} for n in NAMES},
                "assumed", {"note": "unit test toy scene, assumed"})
    return load_prior(p, NAMES), p


def cov(n):
    P = torch.zeros((n, 12, 12), device="cuda")
    P[:, range(12), range(12)] = torch.tensor([1e-4] * 3 + [1e-3] * 3 + [1e-2] * 3 + [1e-2] * 3, device="cuda")
    return P


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from estimator.first_return_observation import FirstReturnObservation
    from estimator.sparse_iekf import iekf_frame_step
    cams, st, truth, obs, model, cfg, gt_raw, masks = setup()
    with tempfile.TemporaryDirectory() as td:
        prior, ppath = prior_file(td)

        def step(fr):
            return iekf_frame_step(st, cov(st.mu.shape[0]), obs, cams, dt=0.04, cfg=cfg,
                                   observation_model=model, first_return=fr)
        print("\n[D1] off")
        base = [step(None) for _ in range(3)]
        floor = max(float((a[0].mu - b[0].mu).abs().max()) for a in base for b in base)
        off = step(FirstReturnObservation("off", None))
        d_off = float((off[0].mu - base[0][0].mu).abs().max())
        check("[D1] mode off == legacy (within 2x the floor of 3 control runs)", d_off <= 2 * max(floor, 1e-9),
              f"{d_off:.1e}, floor {floor:.1e}")

        print("\n[D2] audit")
        au = step(FirstReturnObservation("audit", prior))
        d_au = float((au[0].mu - base[0][0].mu).abs().max())
        row = next((r for r in au[2].rows if r.get("event") == "first_return"), None)
        check("[D2] audit mode: legacy update, first-return summary recorded",
              d_au <= 2 * max(floor, 1e-9) and row is not None and row["mode"] == "audit"
              and len(row["cameras"]) == 3, f"{d_au:.1e}")

        print("\n[D3] depth")
        x, P, diag = step(FirstReturnObservation("depth", prior))
        ev = [r for r in diag.rows if r.get("event") in ("initial", "accepted", "rejected")]
        acc = [r for r in diag.rows if r.get("event") == "accepted"]
        check("[D3] no blended-depth energy; first-return depth energy in every evaluation",
              all(r.get("depth_energy", 1.0) == 0.0 for r in ev) and all("first_return_depth_energy" in r for r in ev))
        e_seq = [ev[0]["energy"]] + [r["energy"] for r in acc]
        mono = all(b < a for a, b in zip(e_seq, e_seq[1:]))
        check("[D3] every accepted step lowers the TRUE nonlinear objective", len(acc) > 0 and mono,
              f"{len(acc)} accepted: {[round(e, 1) for e in e_seq[:5]]}")
        tr0 = float((st.mu - truth.mu).norm(dim=1).median())
        tr1 = float((x.mu - truth.mu).norm(dim=1).median())
        trl = float((base[0][0].mu - truth.mu).norm(dim=1).median())
        check("[D3] the state moves toward the (first-return-consistent) observation", tr1 < tr0,
              f"median |x - truth| {1000 * tr0:.2f} -> {1000 * tr1:.2f} mm (legacy blended-depth step: {1000 * trl:.2f} mm)")
        post = next(r for r in diag.rows if r.get("event") == "posterior")
        evs = torch.linalg.eigvalsh(P.double().cpu())
        check("[D3] posterior PSD, labelled POSTERIOR_UNCERTIFIED with its curvature",
              float(evs.min()) >= -1e-10 * float(evs.max()) and post["posterior_status"] == "POSTERIOR_UNCERTIFIED"
              and "B_vis" in post["curvature"])
        frr = next((r for r in diag.rows if r.get("event") == "first_return"), None)
        check("[D3] first-return diagnostics at the accepted state", frr is not None
              and "top_front_contradiction_gaussians" in frr)

        print("\n[D4] refusals")
        res = []
        try:
            iekf_frame_step(st, cov(40), obs, cams, dt=0.04, cfg=cfg, observation_model=None,
                            first_return=FirstReturnObservation("depth", prior))
            res.append(False)
        except (ValueError, TypeError):
            res.append(True)
        from estimator.first_return_observation import from_args
        try:
            from_args(argparse.Namespace(first_return_mode="depth", first_return_prior=None)); res.append(False)
        except SystemExit:
            res.append(True)
        try:
            from_args(argparse.Namespace(first_return_mode="depth", first_return_prior=str(ppath),
                                         observation_mode="original")); res.append(False)
        except SystemExit:
            res.append(True)
        try:
            from_args(argparse.Namespace(first_return_mode="depth", first_return_prior=str(ppath),
                                         observation_mode="masked-rgb-expected-quad"), ["cam_000", "cam_099"])
            res.append(False)
        except SystemExit:
            res.append(True)
        check("[D4] refused: no masked model / no prior / original mode / camera not in the prior", all(res), str(res))

        print("\n[D5] bootstrap_positions")
        from tools.kpart_pipeline import bootstrap_positions
        from estimator.masked_observation import MaskedSequence
        frames = [(gt_raw, {"original_frame_index": 0}), (gt_raw, {"original_frame_index": 1})]
        mseq = MaskedSequence(model, [[m.cpu() for m in masks]] * 2, list(NAMES), [0, 1])
        a = argparse.Namespace(supernode_size=8, seed=0, bootstrap_iters=3, dt=1 / 25.0, visibility_constraint=None,
                               masked_observation=mseq,
                               phase1_audit=None, phase1_partition="canonical",
                               first_return=FirstReturnObservation("depth", prior))
        pos, info = bootstrap_positions(st, cams, frames, cfg, None, a, verbose=False)
        rec = info["per_frame"][1]
        check("[D5] per-frame first-return record + manifest + posterior status",
              "first_return" in rec and info["first_return"]["mode"] == "depth"
              and rec["covariance"]["posterior_status"] == "POSTERIOR_UNCERTIFIED")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
