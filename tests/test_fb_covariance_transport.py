#!/usr/bin/env python
"""Covariance-consistent transport and observation-tested ownership for the
endpoint diagnosis (J0 track rotations, J1 part motion).

  [1] rigid invariance: moving every Gaussian by a rigid T (centres AND
      orientations, R <- R_T R) and the camera by the same T leaves the RGB-D
      render unchanged; moving the centres alone does not (anisotropic splats)
  [2] the J1 transport (j1_state) is that same covariance update on its rows
  [3] track_rotations recovers a rigid rotation from centre tracks exactly,
      flags collinear neighbourhoods, and stays near identity for a static track
  [4] covariance-consistent J0 lowers the RGB-D error of a rotated part against
      the observation compared with fixed orientations
  [5] interleaved parts: proximity ownership grabs static Gaussians next to
      the moving part; observation_ownership keeps them static and moves the part

    python tests/test_fb_covariance_transport.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.test_fb_pixel_overlap_core import cams_pair, gt_of, plate, state  # noqa: E402

CHECKS, FAILURES = 0, []


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def rot(axis, ang):
    from observation_diagnostics.joint_evidence import rodrigues
    return rodrigues(axis, ang)


def aniso_scene(seed=0, n=120):
    g = torch.Generator().manual_seed(seed)
    mu = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([0.6, 0.45, 0.3])
    from estimator.manifold import so3_exp
    R = so3_exp(0.8 * torch.randn(n, 3, generator=g).double()).float()
    sc = torch.stack([0.08 + 0.04 * torch.rand(n, generator=g), 0.012 + 0.0 * torch.rand(n, generator=g),
                      0.006 + 0.0 * torch.rand(n, generator=g)], 1)            # strongly anisotropic
    return state(mu, torch.rand(n, 3, generator=g), 0.6 + 0.35 * torch.rand(n, generator=g), sc, R=R)


def moved_camera(cam, T):
    """The camera that sees T(world) exactly as `cam` sees world (row-vector convention)."""
    Ti = torch.as_tensor(np.linalg.inv(T).T, dtype=torch.float32, device="cuda")
    return dataclasses.replace(cam, viewmatrix=(Ti @ cam.viewmatrix).contiguous(),
                               projmatrix=(Ti @ cam.projmatrix).contiguous())


class FakeSource:
    def __init__(self, states):
        self.mu = np.stack([s.mu.cpu().numpy() for s in states])
        self.base = states[0]
        self.N = self.mu.shape[1]
        self.frames = list(range(len(states)))

    def state(self, frame, mu=None, R=None):
        mu_t = torch.as_tensor(self.mu[frame] if mu is None else mu, dtype=torch.float32, device="cuda")
        R_t = self.base.R if R is None else torch.as_tensor(R, dtype=torch.float32, device="cuda")
        return dataclasses.replace(self.base, mu=mu_t.contiguous(), R=R_t.contiguous())


def t1_t2_invariance():
    from observation_diagnostics import endpoint_compare as ec
    print("\n[1, 2] rigid invariance of the covariance update")
    cam = cams_pair()[0]
    st = aniso_scene()
    T = np.eye(4)
    T[:3, :3] = rot([0.3, 1.0, 0.2], 0.7)
    T[:3, 3] = [0.15, -0.05, 0.1]
    src = FakeSource([st])
    all_rows = np.arange(st.mu.shape[0])
    moved = ec.j1_state(src, 0, all_rows, T)                      # centres + orientations
    centres_only = src.state(0, mu=moved.mu.cpu().numpy())       # centres only
    camT = moved_camera(cam, T)
    v0 = ec.render("ref", st, cam, 0, "c")
    v1 = ec.render("rigid", moved, camT, 0, "c")
    v2 = ec.render("centres", centres_only, camT, 0, "c")
    e1 = float((v1.lin.rgb - v0.lin.rgb).abs().max())
    a1 = float((v1.lin.alpha - v0.lin.alpha).abs().max())
    cov = v0.lin.alpha > 0.5
    d1 = float((v1.lin.depth - v0.lin.depth)[cov].abs().max())
    check("[1] centres + orientations: RGB, alpha and depth unchanged", e1 < 2e-3 and a1 < 2e-3 and d1 < 1e-3,
          f"max |dRGB| {e1:.1e}, |dA| {a1:.1e}, |dD| {d1:.1e} m")
    e2 = float((v2.lin.rgb - v0.lin.rgb).abs().mean())
    check("[1] centres only (trained orientation kept): the render changes", e2 > 20 * max(float((v1.lin.rgb - v0.lin.rgb).abs().mean()), 1e-7),
          f"mean |dRGB| {e2:.3e} vs {float((v1.lin.rgb - v0.lin.rgb).abs().mean()):.1e}")
    sub = all_rows[: len(all_rows) // 2]
    part = ec.j1_state(src, 0, sub, T)
    Rexp = torch.as_tensor(T[:3, :3], dtype=torch.float32, device="cuda") @ st.R[sub]
    check("[2] j1_state: owned rows R <- R_part R and X <- R_part X + t; others untouched",
          torch.allclose(part.R[sub], Rexp, atol=1e-6) and torch.equal(part.R[len(sub):], st.R[len(sub):])
          and np.allclose(part.mu.cpu().numpy()[sub], st.mu.cpu().numpy()[sub] @ T[:3, :3].T + T[:3, 3], atol=1e-5))


def t3_track_rotations():
    from observation_diagnostics import endpoint_compare as ec
    print("\n[3] rotations from centre tracks")
    g = np.random.default_rng(1)
    X = g.normal(size=(400, 3)) * [0.2, 0.1, 0.05]
    R = rot([0.1, 0.2, 1.0], 1.1)
    Y = X @ R.T + [0.3, 0.0, -0.1]
    Rt, res, deg, _nr = ec.track_rotations(X, Y)
    err = np.degrees(np.arccos(np.clip((np.einsum("nij,ij->n", Rt, R) - 1) / 2, -1, 1)))
    check("[3] a rigid track gives back its rotation for every row", float(err.max()) < 1e-4 and float(res.max()) < 1e-9,
          f"max {err.max():.1e} deg")
    Rs, res_s, _, _ = ec.track_rotations(X, X + g.normal(size=X.shape) * 1e-4)
    es = np.degrees(np.arccos(np.clip((np.trace(Rs, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    check("[3] a static (noisy) track stays near identity", float(np.median(es)) < 1.0, f"median {np.median(es):.2f} deg")
    L = np.c_[np.linspace(0, 1, 50), np.zeros(50), np.zeros(50)]
    _R, _r, dg, _ = ec.track_rotations(L, L @ R.T)
    check("[3] a collinear neighbourhood is flagged degenerate (identity kept)", bool(dg.all()))
    two = np.r_[X, X + [2.0, 0, 0]]
    two_t = np.r_[X @ R.T, X + [2.0, 0, 0]]
    R2, _r2, _d2, _ = ec.track_rotations(two, two_t)
    e_mov = np.degrees(np.arccos(np.clip((np.einsum("nij,ij->n", R2[:400], R) - 1) / 2, -1, 1)))
    e_st = np.degrees(np.arccos(np.clip((np.trace(R2[400:], axis1=1, axis2=2) - 1) / 2, -1, 1)))
    check("[3] two parts: each row gets its own part's rotation", float(np.median(e_mov)) < 1e-3 and float(np.median(e_st)) < 1e-3)
    # a torn track (neighbours scatter by more than their spread, as the real free
    # tracks do on most rows) has no rotation: the trained orientation is kept
    Xd = g.normal(size=(2000, 3)) * 0.1
    Rn, resn, dgn, nrn = ec.track_rotations(Xd, Xd @ R.T + g.normal(size=Xd.shape) * 0.05)
    turned_n = ~(dgn | nrn)
    check("[3] a torn track is flagged not locally rigid and keeps the trained orientation",
          float(nrn.mean()) > 0.95 and np.allclose(Rn[nrn], np.eye(3)), f"{nrn.mean():.1%} torn")
    _Rr, _rr, _dr, nrr = ec.track_rotations(X, Y + g.normal(size=X.shape) * 1e-4)
    check("[3] a rigid track with small noise is not flagged", float(nrr.mean()) < 0.01, f"{nrr.mean():.1%} flagged")


def t4_j0_rgbd():
    from observation_diagnostics import endpoint_compare as ec
    print("\n[4] covariance-consistent J0 transport against the observation")
    cams = cams_pair()
    st = aniso_scene(seed=3, n=300)
    T = np.eye(4)
    T[:3, :3] = rot([0.0, 1.0, 0.3], 0.9)
    src = FakeSource([st])
    truth = ec.j1_state(src, 0, np.arange(st.mu.shape[0]), T)           # the scene really turned
    track = FakeSource([st, truth])                                      # centres-only track 0 -> 1
    Rt, _res, _deg, _ = ec.track_rotations(track.mu[0], track.mu[1])
    fixed = ec.transported_state(track, 1, None)
    turned = ec.transported_state(track, 1, Rt)
    err = {}
    for name, s_ in (("fixed", fixed), ("turned", turned)):
        e_rgb, e_d = 0.0, 0.0
        for cam in cams:
            gt = gt_of(truth, cam)
            v = ec.render(name, s_, cam, 1, "c")
            m = gt["valid"]
            e_rgb += float(((v.lin.rgb - gt["rgb"]) ** 2).mean().sqrt())
            e_d += float((v.lin.depth - gt["depth"])[m].abs().mean())
        err[name] = (e_rgb / len(cams), e_d / len(cams))
    check("[4] turning splats with the track lowers the RGB error of the transported state",
          err["turned"][0] < 0.25 * err["fixed"][0], f"RGB RMSE fixed {err['fixed'][0]:.4f} -> turned {err['turned'][0]:.4f}")
    check("[4] and the depth error", err["turned"][1] < 0.5 * err["fixed"][1] + 1e-4,
          f"depth MAE fixed {err['fixed'][1]:.4f} -> turned {err['turned'][1]:.4f} m")


def t5_ownership():
    from observation_diagnostics import endpoint_compare as ec
    from observation_diagnostics.pixel_evidence import Thresholds
    print("\n[5] ownership where parts interleave")
    cams = cams_pair()
    # two plates side by side, touching at x = 0: the left one turns about the
    # seam (hinge), the right one is static -- like a scissor blade on the other
    movingP = plate(-0.16, 0.0, 0.0, 0.3, 0.36, 8, (0.85, 0.2, 0.2))
    staticP = plate(0.16, 0.0, 0.0, 0.3, 0.36, 8, (0.2, 0.3, 0.85))
    wall = plate(0.0, 0.0, -0.4, 1.4, 1.0, 12, (0.5, 0.5, 0.5))
    nm, ns = movingP[0].shape[0], staticP[0].shape[0]
    mu = torch.cat([movingP[0], staticP[0], wall[0]])
    st = state(mu, torch.cat([movingP[1], staticP[1], wall[1]]), torch.cat([movingP[2], staticP[2], wall[2]]),
               torch.cat([movingP[3], staticP[3], wall[3]]))
    src = FakeSource([st, st])
    T = np.eye(4)
    T[:3, :3] = rot([0.0, 1.0, 0.0], 0.8)
    pivot = np.array([0.0, 0.0, 0.0])
    T[:3, 3] = pivot - T[:3, :3] @ pivot
    truth = ec.j1_state(src, 0, np.arange(nm), T)
    moving = np.arange(nm)
    part_xyz = st.mu.cpu().numpy()[moving]
    prox = ec.infer_ownership(st.mu.cpu().numpy(), part_xyz, 0.12)
    wrong = np.setdiff1d(prox, moving)
    check("[5] proximity ownership grabs static rows next to the moving part", wrong.size > 0,
          f"{wrong.size} static rows owned by proximity")
    gts = {f"c{i}": gt_of(truth, c) for i, c in enumerate(cams)}
    res = ec.observation_ownership(src, 0, 1, prox, T, [f"c{i}" for i in range(len(cams))], cams,
                                   [f"c{i}" for i in range(len(cams))], lambda cam: gts[cam], Thresholds())
    own = res["owned"]
    tp = np.isin(own, moving).sum()
    fp = own.size - tp
    rec = tp / moving.size
    check("[5] observation ownership keeps the static neighbours static", fp <= max(1, 0.05 * own.size),
          f"{fp} static rows owned, reasons {res['reasons']}")
    check("[5] and still moves the part", rec >= 0.85, f"recall {rec:.2f}")
    # the estimator's support covers only half the moving plate (scissor: recall
    # 0.43-0.50 at precision 1.0): the free-space test of every row recovers the rest
    half = moving[: nm // 2]
    near = ec.infer_ownership(st.mu.cpu().numpy(), part_xyz[: nm // 2], 0.03)
    names = [f"c{i}" for i in range(len(cams))]
    tr = ec.tested_ownership(src, 0, 1, T, names, cams, names, lambda cam: gts[cam], Thresholds(), near,
                             prior_owned=half)
    # rows the motion displaces by less than the depth tolerance (the hinge column)
    # cannot be told apart by any observation; recall is scored on the others
    disp = np.linalg.norm(part_xyz @ T[:3, :3].T + T[:3, 3] - part_xyz, axis=1)
    tell = moving[disp > 2 * Thresholds().depth_tol_m]
    tp2 = np.isin(tell, tr["owned"]).sum()
    fp2 = np.setdiff1d(tr["owned"], moving).size
    missed = np.setdiff1d(moving, tr["owned"])
    check("[5] half the support: every-row testing recovers the rest of the part",
          tp2 / tell.size >= 0.95 and fp2 <= max(1, 0.05 * tr["owned"].size),
          f"recall {tp2 / tell.size:.2f} on {tell.size} distinguishable rows, {fp2} static rows owned, "
          f"missed rows max displacement {disp[np.isin(moving, missed)].max() if missed.size else 0:.3f} m, "
          f"reasons {tr['reasons']}")


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    t1_t2_invariance()
    t3_track_rotations()
    t4_j0_rgbd()
    t5_ownership()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
