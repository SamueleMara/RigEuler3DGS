#!/usr/bin/env python
"""Side-by-side summary of tools/audit_calibrated_phase1.py outputs (read-only).

    python tools/summarize_phase1_audit.py audit_A.json audit_B.json ...

Per arm and frame: displacement from canonical (p50 / p99 mm), the accepted update
along the most aligned camera ray, how much of the position correction became
velocity (dv / (dmu/dt), explained fraction of the pose->twist conditional),
information attribution, camera-force conflict at the PRIOR state, solver status,
and the Fisher check. A residual report (--residual-report-only) is summarized as
persistent vs temporal depth residual against the sensor sd.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def f(x, k=0, scale=1.0, fmt="{:.2f}"):
    try:
        return fmt.format(scale * x[k])
    except (TypeError, IndexError, KeyError):
        return "-"


def residual(d: dict, name: str) -> None:
    print(f"\n== {name}: residual report, window {d.get('window')}, declared model sd {d.get('declared_model_sd')}")
    for c in d["stationary_window"]["cameras"]:
        for reg in ("interior", "boundary"):
            r = c.get(reg)
            if r:
                print(f"  {c['camera']:8s} {reg:9s} px {r['pixels']:7d}  persistent rms {r['persistent_rms_mm']:7.2f} mm  "
                      f"temporal sd med {f(r['temporal_sd_mm_q'])} mm  sensor sd {r['sensor_sd_mm_median']:.2f} mm  "
                      f"temporal/sensor {r['temporal_sd_over_sensor_sd_median']:.2f}")
        rg = c.get("rgb_in_mask")
        if rg:
            print(f"  {c['camera']:8s} rgb       persistent rms {[round(x, 4) for x in rg['persistent_rms']]}  "
                  f"temporal sd {[round(x, 4) for x in rg['temporal_sd_median']]}  sensor {[round(x, 4) for x in rg['sensor_sd']]}")
    for c in d["canonical_control"]["cameras"]:
        i = c.get("interior") or {}
        print(f"  canonical control {c['camera']}: interior signed median {i.get('signed_median_mm', float('nan')):.2f} mm, "
              f"render behind {i.get('fraction_render_behind', float('nan')):.2f}")


def audit(d: dict, name: str) -> None:
    print(f"\n== {name}: partition {(d.get('phase1_partition') or {}).get('mode')}, window {d.get('window')}")
    print("  frame  |dx0| p50/p99 mm   upd p99 mm  max|cos ray|  dv/(dmu/dt)  expl   own/grp/pred      "
          "prior-conflict  stop / status            fisher")
    for r in d["rows"]:
        st = r["state"]
        vg = r.get("velocity_gain", {})
        at = r.get("attribution", {})
        ray = max((c.get("abs_cos_top1pct_q") or [0])[0] for c in r.get("ray_motion", {}).get("cameras", [{}]))
        pri = r.get("forces_at", {}).get("prior", {}).get("camera_agreement", {})
        conf = [v.get("fraction_conflicting") for v in pri.values() if v.get("fraction_conflicting") is not None]
        sv = r.get("solver", {})
        fid = r["frame"].get("original_frame_index", r["processing_index"]) if isinstance(r["frame"], dict) else r["frame"]
        print(f"  {fid:5}  {f(st['disp_from_canonical_mm_q'], 0)}/{f(st['disp_from_canonical_mm_q'], 2)}"
              f"{'':6}{f(r.get('ray_motion', {}).get('update_mm_q'), 2):>8}   {ray:8.2f}    "
              f"{f(vg.get('dv_over_dmu_per_dt_q')):>7}    {f(vg.get('explained_fraction_q')):>5}  "
              f"{at.get('own_measurement', '-')}/{at.get('no_own_information_moved_by_group', '-')}/{at.get('prediction_only', '-')}"
              f"{'':4}{(sum(conf) / len(conf)) if conf else float('nan'):6.2f}        "
              f"{sv.get('stop_reason')} / {sv.get('posterior_status')}  "
              f"{(r.get('fisher_check') or {}).get('rel_err_sum_cameras_vs_production', float('nan')):.1e}")


def main(argv) -> int:
    for p in argv:
        d = json.loads(Path(p).read_text())
        (residual if "stationary_window" in d else audit)(d, Path(p).stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
