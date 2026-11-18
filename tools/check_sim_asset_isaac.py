#!/usr/bin/env python
"""Optional Isaac Sim PhysX runtime check of an exported asset (spec 6B / gate D), run by hand:

    python tools/check_sim_asset_isaac.py --asset-dir <P5>/assets/<scene> --isaac-python ~/isaac-sim/python.sh

Reads the asset's own asset.yaml + joint_trajectory.csv (no Phase-5 run needed), starts a headless Isaac Sim in its own
interpreter (tools/sim_runtimes/_isaac_runtime_probe.py), instantiates the articulation from isaac/model.usd, sets
every recorded q (+ the stops) and compares the PhysX link poses with T_W_B and T_W_B T_B_J0 M_z(q).
The asset directory is immutable (its manifest hashes every file): the report is written NEXT to it,
<asset-dir>.isaac_runtime_check.json. Exit 0 PASS, 3 FAIL, 2 could not run.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from recovery.sim_asset_export.frames import M_z, R_from_quat_wxyz, se3  # noqa: E402

TOL = {"pos_m": 1e-4, "rot": 1e-4, "q": 1e-4}     # float32 PhysX after one step with zero velocity / gravity off


def main(argv=None) -> int:
    import yaml
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--asset-dir", required=True)
    ap.add_argument("--isaac-python", required=True, help="Isaac Sim python.sh")
    ap.add_argument("--max-q", type=int, default=60, help="evenly subsample the recorded q (the stops are always added)")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args(argv)
    A = Path(a.asset_dir).resolve()
    y = yaml.safe_load((A / "asset.yaml").read_text())
    usd = A / "isaac" / "model.usd"
    if not usd.is_file():
        print(f"{usd} missing (asset status {y['status']})", file=sys.stderr)
        return 2
    jt = y["joint"]["type"]
    T_W_B = np.asarray(y["frames"]["T_world_root"], float)
    T_B_J0 = np.asarray(y["joint"]["T_parent_joint"], float)
    qs = [float(r["q"]) for r in csv.DictReader(open(A / "joint_trajectory.csv"))]
    if len(qs) > a.max_q:
        qs = [qs[int(i)] for i in np.linspace(0, len(qs) - 1, a.max_q).round()]
    lim = y["joint"]["physical_limits"]
    qs += [lim["lower"], lim["upper"], 0.0]
    with tempfile.TemporaryDirectory() as tmp:
        rq, oj = Path(tmp) / "req.json", Path(tmp) / "out.json"
        rq.write_text(json.dumps({"usd": str(usd), "q": qs}))
        env = {k: os.environ[k] for k in ("HOME", "DISPLAY", "XAUTHORITY", "USER") if k in os.environ}
        env["PATH"] = "/usr/bin:/bin"
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        try:
            r = subprocess.run([a.isaac_python, str(REPO / "tools" / "sim_runtimes" / "_isaac_runtime_probe.py"),
                                str(rq), str(oj)], capture_output=True, text=True, timeout=a.timeout, env=env)
        except subprocess.TimeoutExpired:
            print("Isaac Sim timed out", file=sys.stderr)
            return 2
        if r.returncode != 0 or not oj.is_file():
            print(f"Isaac Sim probe failed (rc {r.returncode}):\n{r.stderr[-4000:]}\n{r.stdout[-2000:]}",
                  file=sys.stderr)
            return 2
        out = json.loads(oj.read_text())
    errs, e_pos, e_rot, e_q = [], 0.0, 0.0, 0.0
    if out["num_dof"] != 1:
        errs.append(f"num_dof {out['num_dof']}")
    for q, p in zip(qs, out["poses"]):
        for n, Te in (("static_link", T_W_B), ("moving_link", T_W_B @ T_B_J0 @ M_z(jt, q))):
            Tg = se3(R_from_quat_wxyz(p[n]["quat_wxyz"]), p[n]["pos"])
            e_pos = max(e_pos, float(np.abs(Tg[:3, 3] - Te[:3, 3]).max()))
            e_rot = max(e_rot, float(np.abs(Tg[:3, :3] - Te[:3, :3]).max()))
        e_q = max(e_q, abs(p["q_read"] - q))
    if e_pos > TOL["pos_m"] or e_rot > TOL["rot"] or e_q > TOL["q"]:
        errs.append(f"pose error {e_pos:.3g} m / {e_rot:.3g}, q error {e_q:.3g}")
    lim_rep = None
    if "dof_lower" in out:
        k = 1.0
        lim_rep = {"isaac": [out["dof_lower"], out["dof_upper"]], "asset": [lim["lower"], lim["upper"]],
                   "units_note": "Isaac core reports revolute limits in radians" if jt == "revolute" else "m"}
        if max(abs(out["dof_lower"] - lim["lower"] * k), abs(out["dof_upper"] - lim["upper"] * k)) > 1e-4:
            errs.append(f"limits {lim_rep}")
    rep = {"status": "PASS" if not errs else "FAIL", "errors": errs, "q_evaluated": len(qs), "pose_max_error_m": e_pos,
           "rot_max_error": e_rot, "q_max_error": e_q, "limits": lim_rep, "isaac": {k: v for k, v in out.items()
                                                                                    if k != "poses"},
           "tolerance": TOL, "deg_to_rad": math.pi / 180}
    dst = A.parent / f"{A.name}.isaac_runtime_check.json"
    dst.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "isaac"}, indent=1))
    print(f"-> {dst}")
    return 0 if not errs else 3


if __name__ == "__main__":
    raise SystemExit(main())
