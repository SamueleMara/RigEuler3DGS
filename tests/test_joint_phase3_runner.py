#!/usr/bin/env python
"""tools/run_joint_phase3.py plumbing, on real Phase-2 outputs (read only, CPU).

  [1] the joint identification of a backward run becomes P and R parts in
      processing order, anchored at q = 0 on the canonical (final) frame
  [2] a window whose length differs from the identification is refused
  [3] a support from the other direction is refused before anything loads

    python tests/test_joint_phase3_runner.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RES = Path(os.environ.get("RESULTS_ROOT", "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"))
SCENE = "fridge_10905"
CHECKS, FAILURES = 0, []


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from tools import run_joint_phase3 as p3
    p2 = RES / SCENE / "bidirectional" / "backward_rigid"
    if not (p2 / "joint_reduction").is_dir():
        print(f"NOT_RUN: {p2} missing")
        return 0
    sup = json.loads((p2 / "support_coherence" / "support.json").read_text())
    members = torch.as_tensor(sup["members"], dtype=torch.long)
    n = int(members.max()) + 1
    mu0 = torch.zeros((n, 3), dtype=torch.float64)
    print("\n[1] initialisation from the backward joint identification")
    init = p3.load_initialisation(p2, members, mu0, 60)
    P, R = init["parts"].get("prismatic"), init["parts"].get("revolute")
    check("[1] both hypotheses initialised", P is not None and R is not None, json.dumps(
        {k: v.get("init") is not None for k, v in init["notes"].items()}))
    check("[1] coordinates start at 0 on the canonical frame",
          float(P.s[0]) == 0.0 and float(R.theta[0]) == 0.0)
    check("[1] the revolute track moves (endpoint-relative range)",
          float(R.theta.max() - R.theta.min()) > 0.5, f"{float(R.theta.max() - R.theta.min()):.3f} rad")
    check("[1] the axis is a unit vector, members kept", abs(float(R.axis.norm()) - 1) < 1e-12
          and torch.equal(R.members, members.cpu()))
    print("\n[2] window length")
    try:
        p3.load_initialisation(p2, members, mu0, 30)
        check("[2] a 30-frame window against a 60-frame identification is refused", False)
    except SystemExit as e:
        check("[2] a 30-frame window against a 60-frame identification is refused", True, str(e)[:80])
    print("\n[3] direction")
    with tempfile.TemporaryDirectory() as td:
        try:
            p3.main(["--scene", SCENE, "--direction", "forward", "--model", str(td), "--source", str(td),
                     "--phase2-dir", str(p2), "--out", str(Path(td) / "o"),
                     # the default masked mode would stop first at the mask preflight
                     # (no masks in the temp dir); this check is about the direction
                     "--observation-mode", "original"])
            check("[3] a backward support with --direction forward is refused", False)
        except SystemExit as e:
            check("[3] a backward support with --direction forward is refused",
                  "not --direction forward" in str(e), str(e)[:100])
    print("\n[4] the joint is fixed: only (q, qdot) is estimated")
    from estimator.joint_displacement import DisplacementOnlyState, geometry_snapshot, geometry_unchanged
    from estimator.joint_process import apply_correction
    from estimator.phase2_model_comparison import prismatic_state, revolute_state
    g = torch.Generator().manual_seed(0)
    X = torch.randn((300, 3), generator=g, dtype=torch.float64) * 0.2
    mem = torch.arange(0, 300, 2)
    for name, part in (("revolute", R), ("prismatic", P)):
        part = type(part).__new__(type(part)); part.__dict__.update(
            (init["parts"][name]).__dict__)
        part.members = mem
        full = (revolute_state if name == "revolute" else prismatic_state)(part)
        st = DisplacementOnlyState(full)
        snap = geometry_snapshot(part)
        k = 7
        d2 = torch.tensor([0.013, -0.002], dtype=torch.float64)
        dfull = torch.zeros(full.error_dim, dtype=torch.float64)
        dfull[:2] = d2
        Jr, Jf = st.jvp(k, d2, X), full.jvp(k, dfull, X)
        check(f"[4] {name}: H of the reduced state is the q-column of the full joint model",
              torch.allclose(Jr, Jf, atol=0, rtol=0) and float(Jr.abs().max()) > 0)
        a = torch.randn((300, 6), generator=g, dtype=torch.float64)
        lhs = float((Jr * a).sum())
        rhs = float((st.vjp(k, a, X) * d2).sum())
        check(f"[4] {name}: reduced VJP is the transpose (<Jd,a> = <d,J^T a>)",
              abs(lhs - rhs) <= 1e-10 * max(1.0, abs(lhs)), f"{lhs:.6e} vs {rhs:.6e}")
        q_before = part.q.clone()
        apply_correction(st, k, torch.tensor([0.05, 0.01], dtype=torch.float64))
        dq = part.q - q_before
        check(f"[4] {name}: a retraction moves q[k] only and leaves the geometry bitwise unchanged",
              abs(float(dq[k]) - 0.05) < 1e-15 and float(dq.abs().sum() - abs(dq[k])) == 0.0
              and geometry_unchanged(snap, part) and st.covariance.shape == (2, 2))
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
