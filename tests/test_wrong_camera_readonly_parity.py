#!/usr/bin/env python
"""Wrong-camera audit, test 1: zero-mode and debug-on parity of production Phase 1.

  [R1] iteration_observer=None (default) is the unmodified path: three control runs give
       the GPU reproducibility floor
  [R2] the full debug observer (forces every iteration, additivity checks and the grouped
       leave-one-out solves at the first update) leaves every frame's mu, v, omega, P_post
       and the accepted/rejected trial sequence within 2x that floor
  [R3] the observer actually captured the first update (checks and LOO present)
  [R4] the tool's camera-drop and channel-ablation helpers leave the baseline cfg untouched

    python tests/test_wrong_camera_readonly_parity.py
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
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
    from audit_phase1_wrong_camera import WrongCameraObserver, drop_camera, parity, replay
    from estimator.masked_observation import MaskedSequence
    check, st_ = check_factory()
    cams, st, truth, obs, gt, masks, model, cfg = masked_scene(n=40, H=48, W=48)
    window = [(gt, {"original_frame_index": k}) for k in range(4)]
    mseq = MaskedSequence(model, [[m.cpu() for m in masks]] * 4, list(NAMES), [0, 1, 2, 3])
    ns = argparse.Namespace(supernode_size=8, seed=0, bootstrap_iters=4, dt=1 / 25.0, visibility_constraint=None,
                            masked_observation=mseq, phase1_partition="canonical")
    runs = [replay(st, cams, window, cfg, ns) for _ in range(3)]
    floors = [parity(runs[0], r)["worst"] for r in runs[1:]]
    floor = {k: max(f[k] for f in floors) for k in floors[0]}
    print(f"  floor (3 control runs): {floor}")
    check("[R1] the default path is reproducible (finite floor)", all(v == v for v in floor.values()))
    obs_on = WrongCameraObserver(list(NAMES), loo_frame=1, loo_tol=1e-6, loo_max_iters=300)
    on = replay(st, cams, window, cfg, ns, observer=obs_on)
    par = parity(runs[0], on)
    ok = all(par["worst"][k] <= 2 * max(floor[k], 1e-9) for k in floor)
    check("[R2] debug-on == debug-off within 2x the floor (mu, v, omega, P_post)", ok,
          f"on {par['worst']} vs floor {floor}")
    check("[R2] same accepted / rejected trial sequence", par["all_same_accept_sequence"])
    fu = obs_on.first
    check("[R3] first update captured: checks, LOO arms, iteration rows",
          fu is not None and "rhs_additivity_rel_err" in fu["checks"] and len(fu["loo"]["arms"]) == 12
          and len(obs_on.iter_rows) > 0 and 1 in obs_on.forces0)
    names_before = cfg.camera_names
    c2, w2, cfg2, ns2 = drop_camera(cams, window, cfg, ns, 1)
    cfg3 = dataclasses.replace(cfg, debug_ablation={"cam_000": ["depth"]})
    check("[R4] ablation helpers never mutate the baseline config / observation",
          cfg.camera_names == names_before and cfg.debug_ablation is None and len(cfg2.camera_names) == 2
          and len(ns.masked_observation.masks[0]) == 3 and len(w2[0][0]) == 2 and cfg3.debug_ablation is not None)
    print("\n[R5] the audit tool end to end on the toy scene (all gates, every artefact)")
    import json, tempfile
    from audit_phase1_wrong_camera import main as tool_main
    with tempfile.TemporaryDirectory() as td:
        mdir = Path(td) / "model"; mdir.mkdir()
        (mdir / "bootstrap_manifest.json").write_text(json.dumps({"endpoint_frame": 0}))

        def toy_setup(a):
            ns2 = argparse.Namespace(**vars(ns))
            return st, cams, window, cfg, ns2, {"frames": [w[1] for w in window], "cameras": list(NAMES)}
        rc = tool_main(["--model", str(mdir), "--source", td, "--out", td, "--motion-start-frame", "0",
                        "--motion-end-frame", "3", "--n-stable", "8", "--n-case-studies", "4", "--loo-max-iters", "200",
                        "--gate-c"], _setup=toy_setup)
        out = Path(td) / "wrong_camera_audit"
        need = ["manifest.json", "baseline_parity.json", "per_frame_summary.csv", "per_camera_channel_energy.csv",
                "per_gaussian_force.npz", "per_group_force.npz", "per_camera_normal_checks.json",
                "loo_frozen_influence.npz", "selected_contributor_traces.jsonl", "first_update_case_studies.json",
                "ablation_results.csv", "nonlinear_energy.jsonl", "conclusion.md", "schemas.json",
                "plots/false_displacement.png", "plots/loo_camera_influence.png"]
        missing = [f for f in need if not (out / f).is_file()]
        check("[R5] tool exit 0 and every artefact written", rc == 0 and not missing, f"rc {rc}, missing {missing}")
        par = json.loads((out / "baseline_parity.json").read_text())
        cs = json.loads((out / "first_update_case_studies.json").read_text())
        abl = (out / "ablation_results.csv").read_text().splitlines()
        check("[R5] parity PASS, case studies carry forces + LOO + contributors, 13 ablation rows",
              par["status"] == "PASS" and cs["cases"]["runaway"] and "camera_loo_influence" in cs["cases"]["runaway"][0]
              and len(abl) == 1 + 1 + 3 + 9 + 1, f"{par['status']}, {len(abl) - 1} ablation rows")
        try:
            tool_main(["--model", str(mdir), "--source", td, "--out", td, "--motion-start-frame", "1",
                       "--motion-end-frame", "3"], _setup=toy_setup)
            check("[R5] endpoint frame != start frame is refused without --accept-endpoint-mismatch", False)
        except SystemExit:
            check("[R5] endpoint frame != start frame is refused without --accept-endpoint-mismatch", True)
    return finish(st_)


if __name__ == "__main__":
    raise SystemExit(main())
