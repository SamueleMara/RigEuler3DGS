#!/usr/bin/env python
"""Deeper first-update analyses (estimator/wrong_camera_deep.py) on the toy CUDA scene.

  [W1] the tool with --deep --deterministic writes objective_conflict.json,
       group_mediation.json and nonlinear_attribution.json, and the deep replay
       reproduces the accepted state bitwise
  [W2] group mediation: own + groupmates + rest + curvature == the leave-one-camera
       influence (exact identity) in the grouped AND the ungrouped system
  [W3] nonlinear attribution agrees with the production objective: E(x_prior) is the
       frame's initial energy, E(x_post) + 1/2 d^T Lambda d is its accepted energy
  [W4] objective conflict: at iteration 0 the prior exerts no force, and the summed
       measurement force pulls along the accepted update for most moved Gaussians

    python tests/test_wrong_camera_deep.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from tests.wrong_camera_fixtures import NAMES, check_factory, finish, masked_scene  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    from audit_phase1_wrong_camera import main as tool_main
    from estimator.masked_observation import MaskedSequence
    check, st_ = check_factory()
    cams, st, truth, obs, gt, masks, model, cfg = masked_scene(n=40, H=48, W=48)
    window = [(gt, {"original_frame_index": k}) for k in range(4)]
    mseq = MaskedSequence(model, [[m.cpu() for m in masks]] * 4, list(NAMES), [0, 1, 2, 3])
    ns = argparse.Namespace(supernode_size=8, seed=0, bootstrap_iters=4, dt=1 / 25.0, visibility_constraint=None,
                            masked_observation=mseq, phase1_partition="canonical")
    with tempfile.TemporaryDirectory() as td:
        mdir = Path(td) / "model"
        mdir.mkdir()
        (mdir / "bootstrap_manifest.json").write_text(json.dumps({"endpoint_frame": 0}))

        def toy_setup(a):
            return st, cams, window, cfg, argparse.Namespace(**vars(ns)), \
                {"frames": [w[1] for w in window], "cameras": list(NAMES)}
        rc = tool_main(["--model", str(mdir), "--source", td, "--out", td, "--motion-start-frame", "0",
                        "--motion-end-frame", "3", "--n-stable", "8", "--n-case-studies", "4",
                        "--runaway-quantile", "0.75", "--loo-max-iters", "300", "--deep-ungrouped-max-iters", "600",
                        "--deep", "--deterministic", "--deep-top-groups", "3"], _setup=toy_setup)
        out = Path(td) / "wrong_camera_audit"
        need = ["objective_conflict.json", "group_mediation.json", "nonlinear_attribution.json"]
        missing = [f for f in need if not (out / f).is_file()]
        check("[W1] tool exit 0 and the three deep artefacts written", rc == 0 and not missing, f"rc {rc}, {missing}")
        oc = json.loads((out / "objective_conflict.json").read_text())
        gm = json.loads((out / "group_mediation.json").read_text())
        na = json.loads((out / "nonlinear_attribution.json").read_text())
        check("[W1] the deep replay reproduces the accepted state bitwise", oc["x_post_bitwise_reproduced"] is True)

        print("\n[W2] exact decomposition of the leave-one-camera influence")
        for label in ("grouped", "ungrouped"):
            errs = {c: r["decomposition_rel_err"] for c, r in gm[label]["cameras"].items()}
            conv = gm[label]["solver_all"]["converged"]
            check(f"[W2] {label}: own + groupmates + rest + curvature == I_c", conv and max(errs.values()) < 1e-3,
                  f"converged {conv}, {errs}")
        rc_ = gm["row_counts"]
        check("[W2] the three row parts partition the scene", sum(rc_.values()) == st.mu.shape[0], str(rc_))

        print("\n[W3] nonlinear attribution == the production objective")
        rows = na["production_energy_rows_of_this_frame"]
        init = next(r for r in rows if r["event"] == "initial")
        acc = [r for r in rows if r["event"] == "accepted"]
        e0 = sum(na["E_x_prior"].values())
        rel0 = abs(e0 - init["energy"]) / max(abs(init["energy"]), 1e-30)
        check("[W3] sum of per camera:channel energies at x_prior == the initial energy", rel0 < 1e-4,
              f"{e0:.6g} vs {init['energy']:.6g}")
        if acc:
            e1 = init["energy"] + na["all_moved"]["TOTAL"]
            ref = acc[-1].get("trial_energy", acc[-1]["energy"])
            rel1 = abs(e1 - ref) / max(abs(ref), 1e-30)
            check("[W3] E(x_post) + prior energy of the move == the last accepted energy", rel1 < 1e-3,
                  f"{e1:.6g} vs {ref:.6g}")
        check("[W3] subset and per-supernode rows present", "only_runaway_moved" in na and "only_stable_moved" in na
              and len(na.get("per_supernode_top_runaway", [])) >= 1)

        print("\n[W4] objective conflict")
        it0 = oc["iteration_0"]["gaussian"]["runaway"]
        check("[W4] the prior exerts no force at iteration 0 (x = x_prior)",
              abs(it0["PRIOR"]["median_projection"]) < 1e-9, str(it0["PRIOR"]))
        check("[W4] the summed measurement force pulls along the accepted update for most runaway Gaussians",
              it0["MEASUREMENT_TOTAL"]["fraction_pulling"] >= 0.5, str(it0["MEASUREMENT_TOTAL"]))
        check("[W4] the last inner iteration is recorded", "last_iteration" in oc)
    return finish(st_)


if __name__ == "__main__":
    raise SystemExit(main())
