#!/usr/bin/env python
"""Freeze the PRE-CORRECTION Phase-4 behaviour on the procedural gate scenes (directive 2026-11-11, section 1).

Run ONCE, before the corrected 4A/4B/4C replaced the scoring. Writes tests/fixtures/phase4_pre_correction/:
  <case>.npz   statuses (old 4A), provisional material, GT material, core, per-view full-render losses
               (rgbd_loss.view_loss) of four fixed labellings (GT, old result, core only, all static), and the
               old 4B residual-code maps of the old-result scene
  summary.json counts and the code version that produced them

This is an independent oracle of what the OLD code did, not ground truth: the corrected 4A uses a different
(exact, full-image-normalised) objective, so the regression test compares labels/IDs, not score values.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import recovery  # noqa: E402,F401
from recovery import material_completion as MC  # noqa: E402
from recovery import residual_discovery as RD  # noqa: E402
from recovery.rgbd_loss import LossConfig, view_loss  # noqa: E402
from synthetic_articulated import make_scene  # noqa: E402

OUT = HERE.parent / "fixtures" / "phase4_pre_correction"
CASES = {"revolute_forward": dict(jtype="revolute", direction="forward", withhold_frac=0.4, seed=1),
         "prismatic_backward": dict(jtype="prismatic", direction="backward", withhold_frac=0.4, seed=1)}


def per_view(sc, moving):
    return np.array([[float(view_loss(sc.render(v, moving), sc.view(v), sc.loss_cfg)[k])
                      for k in ("total", "rgb_l1", "rgb_ssim_loss", "depth_l1", "mask_l1")] for v in sc.views])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"frozen": time.strftime("%Y-%m-%d %H:%M:%S"), "code": "recovery/material_completion.py (pre-correction)",
               "columns": ["total", "rgb_l1", "rgb_ssim_loss", "depth_l1", "mask_l1"], "cases": {}}
    for name, kw in CASES.items():
        s = make_scene(**kw)
        sc = MC.Phase4Scene(s.model, s.joint, s.views, LossConfig(), sh_degree=0)
        t0 = time.time()
        res = MC.run_phase4a(sc, s.core, MC.PolicyConfig(max_component=128, sens_max_views=8), log=lambda *a: None)
        dt = time.time() - t0
        dev = sc.device
        gt = torch.as_tensor(s.gt_moving_model, device=dev)
        core = torch.zeros_like(gt); core[torch.as_tensor(s.core, device=dev)] = True
        old = torch.as_tensor(res["material"] == 1, device=dev)
        losses = {k: per_view(sc, m) for k, m in (("gt", gt), ("old_result", old), ("core_only", core),
                                                   ("all_static", torch.zeros_like(gt)))}
        maps, _ = RD.discovery_maps(sc, old, RD.DiscoveryConfig())
        np.savez_compressed(OUT / f"{name}.npz", status=np.asarray(res["status"], dtype="U32"),
                            material=res["material"], gt_moving=s.gt_moving_model, core=s.core,
                            residual_codes=np.stack([m.numpy() for m in maps]),
                            **{f"loss_{k}": v for k, v in losses.items()})
        st = res["status"]
        summary["cases"][name] = {"make_scene": kw, "seconds_old_4a": dt, "n": int(len(st)),
                                  "status_counts": {k: int(v) for k, v in zip(*np.unique(st, return_counts=True))},
                                  "false_moving": int(((res["material"] == 1) & ~s.gt_moving_model).sum()),
                                  "total_loss": {k: float(v[:, 0].sum()) for k, v in losses.items()}}
        print(name, json.dumps(summary["cases"][name]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
