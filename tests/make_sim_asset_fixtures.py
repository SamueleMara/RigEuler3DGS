#!/usr/bin/env python
"""Build the two runnable synthetic fixtures (revolute forward, prismatic backward) and export them completely:
a completed synthetic Phase-5 run, its physics spec, and assets/<scene>/ with both PLYs, URDF, USD, MJCF.

    RIGEULER_SIM_PYTHON=... RIGEULER_SAPIEN_PYTHON=... python tests/make_sim_asset_fixtures.py --out DIR \
        [--isaac-python ~/isaac-sim/python.sh]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "helpers"))
from recovery.sim_asset_export import export_phase5_sim_asset  # noqa: E402
from synthetic_phase5_run import make_phase5_run, make_physics_spec  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--isaac-python", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new directory")
    summary = {}
    for jt, d in (("revolute", "forward"), ("prismatic", "backward")):
        root = out / f"synthetic_{jt}_{d}"
        run = make_phase5_run(root, jt, d, sh_degree=3)
        spec = make_physics_spec(run, root / "physics_inputs")
        r = export_phase5_sim_asset(run["phase5_dir"], physics_spec=spec, mode="required")
        rep = {"status": r.status, "asset_root": str(r.asset_root), "formats": {k: r.validation_report[k]["status"]
                                                                              for k in ("urdf", "usd", "mjcf")},
               "sapien_parttwin1d": r.validation_report.get("sapien_parttwin1d", {}).get("status")}
        if a.isaac_python:
            c = subprocess.run([sys.executable, str(HERE.parent / "tools" / "check_sim_asset_isaac.py"), "--asset-dir",
                                str(r.asset_root), "--isaac-python", a.isaac_python], capture_output=True, text=True)
            rep["isaac_runtime"] = {0: "PASS", 3: "FAIL"}.get(c.returncode, "NOT_RUN")
        else:
            rep["isaac_runtime"] = "NOT_RUN"
        summary[f"{jt}_{d}"] = rep
    (out / "fixtures_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
