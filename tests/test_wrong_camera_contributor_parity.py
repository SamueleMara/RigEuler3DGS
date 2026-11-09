#!/usr/bin/env python
"""Wrong-camera audit, test 7 + trace parity: production CSR contributor traces.

Scene: one camera, a 'door' sheet (z = 2.0, opacity 0.95) in front of a 'cavity' sheet
(z = 2.3) covering the same pixels.

  [T1] traced q = alpha T, beta = q / (A + eps); over a fully traced pixel
       sum beta * z == the rendered blended depth and sum q == A (production parity)
  [T2] door observed (D = 2.0): door records MATCH_COMPATIBLE, cavity records
       BEHIND_FIRST_RETURN (a legitimate occlusion, not an error)
  [T3] cavity observed (D = 2.3): door records IN_FRONT_CONTRADICTION (false foreground),
       never 'correctly hidden'; the cavity's front contributors name the door
  [T4] the trace is read-only: the linearization renders are unchanged

    python tests/test_wrong_camera_contributor_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tests.wrong_camera_fixtures import check_factory, finish  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from estimator import wrong_camera_audit as wa
    from estimator.masked_observation import prepare
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import IEKFConfig
    from tests.test_first_return_occlusion import sheet
    from tests.test_iekf_end_to_end import camera
    from tests.test_visibility_constrained_iekf_phase1 import cat
    check, st_ = check_factory()
    cam = camera(width=64, height=64, fov=0.8)
    door, cav = sheet(2.0, 0.95), sheet(2.3, 0.95, seed=1)
    st = cat(door, cav)
    nd = door.mu.shape[0]
    lin = build_linearization(st, [cam])
    v = lin.views[0]
    covered = v.alpha > 0.9
    cfg = IEKFConfig(camera_names=("cam_000",))
    sig = lambda c, D, valid: torch.full_like(D, 0.004)                     # noqa: E731
    ids = list(range(st.mu.shape[0]))

    def obs_with(depth):
        D = torch.where(covered, torch.full_like(v.depth, depth), torch.zeros_like(v.depth))
        return [prepare(v.rgb.clone(), D, covered, covered, "cam_000", 1)]
    before = (v.rgb.clone(), v.depth.clone(), v.alpha.clone())
    rows = wa.contributor_traces(lin, obs_with(2.0), cfg, ids, ["cam_000"], sig)
    by = {}
    for r in rows:
        by.setdefault((r["u"], r["v"]), []).append(r)
    errA = errD = 0.0
    for (u, w), rr in by.items():
        A = float(v.alpha[w, u]); Dh = float(v.depth[w, u])
        errA = max(errA, abs(sum(r["q"] for r in rr) - A))
        if A > 0.5:
            errD = max(errD, abs(sum(r["beta"] * r["gaussian_depth"] for r in rr) - Dh))
    check("[T1] sum q == rendered A and sum beta z == rendered blended depth (all contributors traced)",
          errA < 1e-5 and errD < 1e-4, f"|dA| {errA:.1e}, |dD| {errD:.1e}")

    def frac(rows, lo, hi, cls, central=True):
        sel = [r for r in rows if lo <= r["gaussian_id"] < hi and r["mask"] and r["depth_edge_distance_px"] > 3
               and (r["q"] > 1e-3 or not central)]
        return sum(r["class"] == cls for r in sel) / max(len(sel), 1), len(sel)
    fm, nm = frac(rows, 0, nd, "MATCH_COMPATIBLE")
    fb, nb = frac(rows, nd, 2 * nd, "BEHIND_FIRST_RETURN")
    check("[T2] door observed: door MATCH_COMPATIBLE, hidden cavity BEHIND_FIRST_RETURN", fm > 0.95 and fb > 0.95,
          f"door match {fm:.2f} (n={nm}), cavity behind {fb:.2f} (n={nb})")
    rows3 = wa.contributor_traces(lin, obs_with(2.3), cfg, ids, ["cam_000"], sig)
    ff, nf = frac(rows3, 0, nd, "IN_FRONT_CONTRADICTION")
    cav_front = [r for r in rows3 if r["gaussian_id"] >= nd and r["mask"] and r["front_contributors"]]
    names_door = sum(any(f["gaussian_id"] < nd for f in r["front_contributors"]) for r in cav_front) / max(len(cav_front), 1)
    check("[T3] cavity observed: the door is an IN_FRONT_CONTRADICTION (false foreground)", ff > 0.95,
          f"{ff:.2f} of {nf} door records")
    check("[T3] the cavity's front contributors name the door", names_door > 0.95, f"{names_door:.2f}")
    check("[T4] tracing left the linearization untouched",
          torch.equal(before[0], v.rgb) and torch.equal(before[1], v.depth) and torch.equal(before[2], v.alpha))
    return finish(st_)


if __name__ == "__main__":
    raise SystemExit(main())
