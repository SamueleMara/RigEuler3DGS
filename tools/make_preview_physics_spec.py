#!/usr/bin/env python
"""PREVIEW physics spec for a completed Phase-5 run when no measured physics exists. Everything it writes is an
explicit, labelled assumption; an asset exported with it can only reach PHYSICS_PREVIEW_ASSUMPTIONS, never
FULL_PHYSICS_READY.

    python tools/make_preview_physics_spec.py --phase5-dir P5 --out DIR/physics_spec.yaml --density 300 \
        --velocity-limit 1.0 --effort-limit 10 --damping 0.1 --coulomb-friction 0.0

  collision   convex hull of each part's Gaussian means, after dropping the outer 2 % per axis (floaters);
              provenance `estimated`. NOTE: on real captures the static part is everything that does not move
              (object body AND the surrounding scene), so its hull is a coarse block, not the object's body.
  inertia     uniform --density over that hull (density_assumption)
  joint       limits = the observed demonstration range (limits_from_observed_range); velocity / effort / damping /
              friction = the declared values (provenance `estimated`)
  gravity     gauge W = capture world, alignment `unverified` (pass --asset-registration later to change it)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    import yaml
    from recovery.sim_asset_export.collision import convex_hull, Mesh, write_obj
    from recovery.sim_asset_export.load_phase5 import load_phase5
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase5-dir", required=True)
    ap.add_argument("--out", required=True, help="spec path; the hull OBJs are written next to it")
    for k in ("density", "velocity-limit", "effort-limit", "damping", "coulomb-friction"):
        ap.add_argument(f"--{k}", type=float, required=True)
    ap.add_argument("--trim-percent", type=float, default=2.0, help="per-axis outer percentile dropped before the hull")
    a = ap.parse_args(argv)
    inp = load_phase5(a.phase5_dir)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = {"schema_version": 1,
         "notes": "PREVIEW ONLY (tools/make_preview_physics_spec.py): convex hulls of the Gaussians, uniform declared "
                  "density, observed-range limits, declared joint dynamics, unverified gravity. Nothing is measured.",
         "registration": {"T_world_capture": None, "gauge": "capture_world", "gravity_alignment": "unverified",
                          "provenance": "preview: capture axes not verified against gravity"},
         "collision": {}, "links": {},
         "joint": {"lower": None, "upper": None, "limits_from_observed_range": True,
                   "velocity_limit": a.velocity_limit, "effort_limit": a.effort_limit, "damping": a.damping,
                   "coulomb_friction": a.coulomb_friction, "provenance": "estimated"}}
    for side, lk, pid in (("static", "static_link", 0), ("moving", "moving_link", 1)):
        p = inp.xyz[inp.part_id == pid]
        lo, hi = np.percentile(p, a.trim_percent, 0), np.percentile(p, 100 - a.trim_percent, 0)
        keep = p[((p >= lo) & (p <= hi)).all(1)]
        h = convex_hull(Mesh(keep, np.zeros((0, 3), np.int64)))
        f = out.parent / f"{side}_preview_hull.obj"
        write_obj(h, f, [f"PREVIEW collision for {lk}: convex hull of {len(keep)} of {len(p)} Gaussian means "
                         f"(outer {a.trim_percent}% per axis dropped), capture coordinates at the anchor (q = 0)"])
        d["collision"][side] = {"source": f.name, "units": "m", "T_capture_mesh": np.eye(4).tolist(),
                                "provenance": "estimated"}
        if side == "moving":
            d["collision"][side]["at_original_frame"] = int(inp.joint.anchor_frame)
        d["links"][lk] = {"density_kg_m3": a.density, "provenance": "density_assumption"}
        print(f"  {lk}: hull of {len(keep)}/{len(p)} Gaussians, {len(h.vertices)} vertices")
    out.write_text(yaml.safe_dump(d, sort_keys=False))
    print(f"preview physics spec -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
