#!/usr/bin/env python
"""Cross-camera depth registration on stationary frames (READ-ONLY diagnostic).

Is the cam-to-cam depth disagreement found by tools/phase0_geometry_consistency.py an
extrinsic / depth-registration error? Each camera's observed depth (per-pixel median over
the stationary frames, object mask, valid depth) is back-projected to world with the
production cameras; for every camera pair a robust point-to-plane ICP, started at the
calibrated identity, estimates the residual rigid transform T_AB that best aligns A's
points to B's surface.

    calibration error   -> a consistent T_AB (cm / degrees) that substantially raises the
                           fraction of A's points within 1 cm of B's surface
    calibration fine    -> T_AB ~ identity, the agreement barely changes

No model, no Gaussians: observations and cameras only. The 6x6 normal matrix condition
number is reported because an object with little overlap lets ICP slide.

    python tools/cross_camera_registration.py --model M --source S --sensor-model CAL \\
        --frames 60 80 --out R.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))


def backproject(cam, D, valid):
    """[M,3] world points of the valid pixels (3DGS pixel-centre convention)."""
    H, W = D.shape
    v, u = np.nonzero(valid)
    z = D[v, u]
    x = ((2 * u + 1) / W - 1) * cam.tanfovx * z
    y = ((2 * v + 1) / H - 1) * cam.tanfovy * z
    Vinv = np.linalg.inv(cam.viewmatrix.double().cpu().numpy())         # row-vector convention: p_cam = p_w @ V
    P = np.c_[x, y, z, np.ones_like(z)] @ Vinv
    return P[:, :3]


def normals(P, k=12):
    from scipy.spatial import cKDTree
    _, nn = cKDTree(P).query(P, k=k)
    Q = P[nn] - P[nn].mean(1, keepdims=True)
    w, V = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))
    return V[:, :, 0], w[:, 0] / np.maximum(w.sum(1), 1e-30)


def icp(PA, PB, NB, iters=40, max_dist=0.05):
    """Point-to-plane ICP moving A onto B; returns (R, t, info)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(PB)
    R, t = np.eye(3), np.zeros(3)
    cond = None
    for _ in range(iters):
        X = PA @ R.T + t
        d, j = tree.query(X)
        n = NB[j]
        r = ((X - PB[j]) * n).sum(1)
        keep = d < max_dist
        if keep.sum() < 100:
            break
        s = np.median(np.abs(r[keep])) * 1.4826 + 1e-6
        w = np.where(np.abs(r) < 3 * s, 1.0, 3 * s / np.abs(r).clip(1e-12)) * keep        # Huber
        J = np.c_[np.cross(X, n), n]
        A = (J * w[:, None]).T @ J
        b = -(J * w[:, None]).T @ r
        cond = float(np.linalg.cond(A))
        x = np.linalg.solve(A + 1e-9 * np.eye(6), b)
        from scipy.spatial.transform import Rotation
        dR = Rotation.from_rotvec(x[:3]).as_matrix()
        R, t = dR @ R, dR @ t + x[3:]
        if np.linalg.norm(x) < 1e-7:
            break
    return R, t, cond


def agreement(PA, PB, R=np.eye(3), t=np.zeros(3)):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(PB).query(PA @ R.T + t)
    return {"within_5mm": float((d < 0.005).mean()), "within_1cm": float((d < 0.01).mean()),
            "within_2cm": float((d < 0.02).mean()), "median_mm": 1000 * float(np.median(d))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="any endpoint of the scene (only its cameras are used)")
    ap.add_argument("--source", required=True)
    ap.add_argument("--frames", type=int, nargs=2, required=True, metavar=("START", "END"))
    ap.add_argument("--direction", choices=("forward", "backward"), default="forward")
    ap.add_argument("--max-points", type=int, default=60000)
    ap.add_argument("--out", required=True)
    from estimator.masked_observation import add_cli as mask_cli
    from estimator.run_config import add_cli as rc_cli
    mask_cli(ap)
    rc_cli(ap, phase1=True)
    a = ap.parse_args(argv)
    for k, v in {"bootstrap_iters": 12, "iekf_supernode_size": 64, "phase1_partition": "canonical",
                 "phase1_covariance_transport": False, "first_return_mode": "off", "first_return_prior": None,
                 "gt_labels": "off"}.items():
        setattr(a, k, v)
    from audit_calibrated_phase1 import _frame_obs, setup_window
    t0 = time.time()
    s0, s1 = a.frames
    ns_args = argparse.Namespace(**vars(a))
    w0, w1 = (s0, s1) if s1 > s0 else ((s0 - 1, s1) if a.direction == "backward" else (s0, s1 + 1))
    ns_args.motion_start_frame, ns_args.motion_end_frame = w0, w1
    truth, cams, window, cfg, ns, _ = setup_window(ns_args)
    names = list(cfg.camera_names)
    stacks = {nm: [] for nm in names}
    used = []
    for k in range(len(window)):
        fid = window[k][1]
        of = fid.get("original_frame_index") if isinstance(fid, dict) else fid
        if not (min(s0, s1) <= int(of) <= max(s0, s1)):
            continue
        used.append(of)
        obs = _frame_obs(ns, k, window, truth.mu.device)
        for c, ob in enumerate(obs):
            D = ob.depth_observed.float()
            ok = ob.depth_valid.bool() & torch.isfinite(D) & (D > 0) & (ob.object_coverage > 0.5)
            stacks[names[c]].append(torch.where(ok, D, torch.full_like(D, float("nan"))).cpu().numpy())
    rng = np.random.default_rng(0)
    pts = {}
    for c, nm in enumerate(names):
        Dm = np.nanmedian(np.stack(stacks[nm]), 0)
        valid = np.isfinite(Dm)
        P = backproject(cams[c], Dm, valid)
        if len(P) > a.max_points:
            P = P[rng.choice(len(P), a.max_points, replace=False)]
        pts[nm] = P
    nrm = {nm: normals(P)[0] for nm, P in pts.items()}
    pairs = {}
    for A in names:
        for B in names:
            if A == B:
                continue
            before = agreement(pts[A], pts[B])
            R, t, cond = icp(pts[A], pts[B], nrm[B])
            after = agreement(pts[A], pts[B], R, t)
            from scipy.spatial.transform import Rotation
            pairs[f"{A}->{B}"] = {"n_points": int(len(pts[A])), "before": before, "after": after,
                                  "rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))),
                                  "translation_mm": float(1000 * np.linalg.norm(t)), "t_mm": (1000 * t).tolist(),
                                  "rotvec": Rotation.from_matrix(R).as_rotvec().tolist(), "R": R.tolist(),
                                  "normal_matrix_condition": cond}
    # a real extrinsic error is ONE transform: T_BA must be ~ T_AB^-1 (cycle residual ~ 0); ICP sliding
    # on a poorly overlapping pair has no reason to be cycle consistent
    for k, v in pairs.items():
        A, B = k.split("->")
        w = pairs.get(f"{B}->{A}")
        if w is None:
            continue
        Rab, Rba = np.array(v["R"]), np.array(w["R"])
        tab, tba = np.array(v["t_mm"]) / 1000, np.array(w["t_mm"]) / 1000
        from scipy.spatial.transform import Rotation
        v["cycle_rotation_deg"] = float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rba @ Rab).as_rotvec())))
        v["cycle_translation_mm"] = float(1000 * np.linalg.norm(Rba @ tab + tba))
    rep = {"tool": "tools/cross_camera_registration.py", "source": a.source, "frames": used, "cameras": names,
           "definition": "T_AB = robust point-to-plane ICP of camera A's back-projected observed depth (median over "
                         "frames, in-mask, valid) onto camera B's, started at the calibrated cameras; agreement = "
                         "fraction of A's points within x of B's points (overlap-limited: only part of A is seen by B)",
           "pairs": pairs, "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    for k, v in pairs.items():
        print(f"{k}: rot {v['rotation_deg']:.2f} deg, t {v['translation_mm']:.1f} mm, within 1 cm "
              f"{v['before']['within_1cm']:.3f} -> {v['after']['within_1cm']:.3f}, cond {v['normal_matrix_condition']:.0f}, "
              f"cycle {v.get('cycle_rotation_deg', float('nan')):.2f} deg / {v.get('cycle_translation_mm', float('nan')):.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
