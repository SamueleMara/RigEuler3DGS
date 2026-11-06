#!/usr/bin/env python
"""tools/cross_camera_registration.py oracles (CPU).

  [X1] back-projection inverts the 3DGS projection (pixel-centre convention, row-vector view matrix)
  [X2] point-to-plane ICP recovers a known 2 deg / 20 mm transform between two samplings of one surface
  [X3] identical clouds give the identity

    python tests/test_cross_camera_registration.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from tests.wrong_camera_fixtures import check_factory, finish  # noqa: E402


def surface(n, rng):
    """A box corner + a curved patch: well-conditioned for 6-DOF registration."""
    a = rng.uniform(-0.2, 0.2, (n, 2))
    P = [np.c_[a[:, 0], a[:, 1], np.zeros(n)], np.c_[np.zeros(n), a[:, 0], a[:, 1] + 0.2],
         np.c_[a[:, 0], np.full(n, 0.2), a[:, 1] + 0.2], np.c_[a[:, 0], a[:, 1], 0.1 * np.sin(6 * a[:, 0]) + 0.4]]
    return np.concatenate(P)


def main() -> int:
    from cross_camera_registration import agreement, backproject, icp, normals
    from scipy.spatial.transform import Rotation
    check, st = check_factory()
    rng = np.random.default_rng(0)
    # [X1] project random points with a camera, back-project the depths
    H, W, tx, ty = 60, 80, 0.6, 0.45
    Rw = Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix()
    tw = np.array([0.05, -0.02, 1.5])
    V = np.eye(4)
    V[:3, :3], V[3, :3] = Rw.T, tw                    # row-vector: p_cam = p_w @ V
    cam = types.SimpleNamespace(viewmatrix=torch.tensor(V), tanfovx=tx, tanfovy=ty, image_height=H, image_width=W)
    u, v = rng.integers(0, W, 200), rng.integers(0, H, 200)
    z = rng.uniform(0.5, 2.0, 200)
    D = np.full((H, W), np.nan)
    D[v, u] = z
    valid = np.isfinite(D)
    Pw = backproject(cam, D, valid)
    Pc = np.c_[Pw, np.ones(len(Pw))] @ V
    vv, uu = np.nonzero(valid)
    ndc_x = Pc[:, 0] / Pc[:, 2] / tx
    ndc_y = Pc[:, 1] / Pc[:, 2] / ty
    pix_u = ((ndc_x + 1) * W - 1) * 0.5
    pix_v = ((ndc_y + 1) * H - 1) * 0.5
    err = max(np.abs(pix_u - uu).max(), np.abs(pix_v - vv).max(), np.abs(Pc[:, 2] - D[vv, uu]).max())
    check("[X1] back-projection re-projects to the same pixel and depth", err < 1e-9, f"max err {err:.1e}")
    # [X2] known transform
    PB = surface(4000, rng)
    PA0 = surface(4000, rng)
    Rt = Rotation.from_rotvec(np.radians(2.0) * np.array([0.3, -0.8, 0.5]) / np.linalg.norm([0.3, -0.8, 0.5])).as_matrix()
    tt = np.array([0.012, -0.010, 0.013])
    PA = (PA0 - tt) @ Rt                              # A = T^-1 (B-surface samples)
    R, t, cond = icp(PA, PB, normals(PB)[0])
    rot_err = np.degrees(np.linalg.norm(Rotation.from_matrix(R @ Rt.T).as_rotvec()))
    t_err = 1000 * np.linalg.norm(t - tt)
    check("[X2] ICP recovers a 2 deg / 20 mm transform", rot_err < 0.1 and t_err < 1.0,
          f"rot err {rot_err:.3f} deg, t err {t_err:.2f} mm, cond {cond:.0f}")
    ag0, ag1 = agreement(PA, PB), agreement(PA, PB, R, t)
    check("[X2] agreement within 5 mm rises after registration", ag1["within_5mm"] > ag0["within_5mm"] + 0.3,
          f"{ag0['within_5mm']:.3f} -> {ag1['within_5mm']:.3f}")
    # [X3] identity
    R, t, _ = icp(PB, PB, normals(PB)[0])
    check("[X3] identical clouds give the identity", np.abs(R - np.eye(3)).max() < 1e-6 and np.abs(t).max() < 1e-6)
    return finish(st)


if __name__ == "__main__":
    raise SystemExit(main())
