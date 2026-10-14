"""The fused Phase-1 wrappers do what their contract says, and nothing else.

Section 4 of CODEX_IMPLEMENT_FUSED_PHASE1_REUSE_EXISTING_FB_WASHER_THEN_OVERNIGHT_ALL.md.

  1  --help of all three wrappers
  2  neither fusion wrapper nor the driver can reach the legacy IEKF runner,
     static training, run_bidirectional_all.sh or an rm -rf
  3  the legacy wrapper fails CLOSED without a mode, and its preflight /
     dry-run point only at a new output path and create nothing
  4  real Washer preflight + dry-run: no forbidden command, every --out
     under unified_phase1_experiments/
  5  real discovery: washer eligible, USB_100109 case preserved
  6  a mocked two-scene overnight run in a results root whose path contains
     spaces: strictly serial, continues after a failure, honest summary
  7  a corrupt trajectory manifest makes a scene ineligible, with the reason
  8  --resume-from refuses a run with a different fingerprint
  9  the IEKS stop hook ends a run with its own status and no applied step

    python tests/test_fused_phase1_wrappers.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
RESULTS = Path("/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours")
DATASET = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/"
               "output/preparation_8cams/partnet_subset_fixed_render_linear")
CHECKS = 0
FAILURES: list[str] = []


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")
    else:
        print(f"  ok    {label}  {detail}")


def sh(args, env=None, timeout=600):
    e = dict(os.environ, PY=PY)
    if env:
        e.update(env)
    p = subprocess.run(args, cwd=REPO, capture_output=True, text=True,
                       env=e, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    scene_sh = ["bash", "scripts/run_fused_phase1.sh"]
    night_sh = ["bash", "scripts/run_fused_phase1_overnight.sh"]
    legacy_sh = ["bash", "scripts/run_legacy_free_phase1_export.sh"]

    print("[1] --help")
    for name, cmd in (("scene", scene_sh), ("overnight", night_sh), ("legacy", legacy_sh)):
        rc, out = sh(cmd + ["--help"])
        check(f"{name} --help exits 0 and documents itself",
              rc == 0 and "--tag" in out, f"exit {rc}")
    rc, out = sh(scene_sh + ["--help"])
    check("scene --help explains RAW_UNION_EXPERIMENT vs VALID_CANONICAL_SCENE",
          "RAW_UNION_EXPERIMENT" in out and "VALID_CANONICAL_SCENE" in out)

    print("\n[2] the fusion path cannot reach the legacy runner or training")
    forbidden = ("run_motion_coherence_support", "train.py", "run_bidirectional_all",
                 "rm -rf", "legacy_phase1_rerun", "run_full_pipeline")
    import ast

    def code_strings(path):
        """String constants that are CODE, not docstrings or table prose."""
        tree = ast.parse(path.read_text())
        docs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and \
                    node.body and isinstance(node.body[0], ast.Expr) and \
                    isinstance(getattr(node.body[0], "value", None), ast.Constant):
                docs.add(id(node.body[0].value))
        return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                and isinstance(n.value, str) and id(n) not in docs]

    for rel in ("scripts/run_fused_phase1.sh", "scripts/run_fused_phase1_overnight.sh",
                "scripts/_fused_phase1_python.sh"):
        code = "\n".join(l for l in (REPO / rel).read_text().splitlines()
                         if not l.lstrip().startswith("#"))
        hits = [f for f in forbidden if f in code]
        check(f"{rel}: no forbidden command in its executable lines", not hits, str(hits))
    drv = code_strings(REPO / "tools/fused_phase1_driver.py")
    hits = [f for f in forbidden if any(f in c for c in drv)]
    check("the driver's code (docstrings excluded) names no forbidden entry point",
          not hits, str(hits))
    rep = (REPO / "tools/write_fused_phase1_report.py").read_text()
    check("the report writer launches nothing at all",
          "subprocess" not in rep and "os.system" not in rep)
    src = (REPO / "tools/fused_phase1_driver.py").read_text()
    targets = set(re.findall(r'REPO / "tools" / "([a-z_0-9]+\.py)"', src)) | set(
        re.findall(r'"(tools/[a-z_0-9]+\.py)"', src))
    check("the driver only launches the fused CLI and the report writer",
          {"run_unified_phase1_real_bidirectional.py", "write_fused_phase1_report.py"}
          >= {Path(t).name for t in targets} or not targets,
          f"{sorted(targets)}")

    print("\n[3] the opt-in legacy wrapper fails closed")
    rc, out = sh(legacy_sh + ["--scene", "washer_103776", "--direction", "backward",
                              "--tag", "legacy_probe_x"])
    check("no mode flag -> nothing runs, exit 2", rc == 2 and "FAIL CLOSED" in out, f"exit {rc}")
    rc, out = sh(legacy_sh + ["--scene", "washer_103776", "--direction", "backward",
                              "--tag", "legacy_probe_x", "--dry-run", "--save-full-state"])
    target = RESULTS / "washer_103776" / "unified_phase1_experiments" / "legacy_probe_x"
    check("dry-run prints the SAME historical runner with a NEW output",
          rc == 0 and "run_motion_coherence_support.py" in out
          and "unified_phase1_experiments/legacy_probe_x/legacy_iekf_backward" in out
          and "--canonical-endpoint final" in out and "--full-state-out" in out,
          f"exit {rc}")
    check("and creates nothing", not target.exists(), str(target))
    rc, out = sh(legacy_sh + ["--scene", "washer_103776", "--direction", "forward",
                              "--tag", "legacy_probe_x", "--preflight"])
    j = json.loads(out[out.index("{"):out.rindex("}") + 1])
    check("preflight reports the historical settings it would reuse",
          rc == 0 and j["settings"]["frames"] == 60 and j["settings"]["cameras"] == 8
          and j["settings"]["bootstrap_iters"] == 12, json.dumps(j["settings"]))

    print("\n[4] real Washer preflight and dry-run")
    rc, out = sh(scene_sh + ["--scene", "washer_103776", "--reuse-existing-phase1",
                             "--profile", "washer-validation", "--tag", "wrapper_probe_x",
                             "--preflight"])
    j = json.loads(out[out.index("{"):out.rindex("}") + 1])
    check("preflight: no blockers, full population, all frames, TWO_CAMERAS",
          rc == 0 and not j["blockers"] and j["gaussians"]["union"] == 45920
          and j["num_frames"] == 60 and j["camera_policy"] == "TWO_CAMERAS",
          f"exit {rc}, {j['gaussians']}, {j['camera_policy']}")
    check("preflight declares the derived initialisation honestly",
          j["initialization_mode"] == "DERIVED_12D_INITIAL_GUESS_FROM_EXISTING_TRACKS")
    rc, out = sh(scene_sh + ["--scene", "washer_103776", "--reuse-existing-phase1",
                             "--profile", "washer-validation", "--tag", "wrapper_probe_x",
                             "--dry-run"])
    plan = out[out.index("=== plan"):]
    outs = re.findall(r"--out '([^']+)'", plan)
    check("dry-run plan has no forbidden command",
          rc == 0 and not [f for f in forbidden if f in plan], f"exit {rc}")
    check("every --out lies under unified_phase1_experiments/",
          outs and all("/unified_phase1_experiments/wrapper_probe_x/" in o for o in outs),
          f"{len(outs)} outputs")
    check("the optimiser call is the raw-union joint IEKS on both sources",
          "--mode raw_union_experiment" in plan and "--representation-policy diagnostic-only"
          in plan and "--camera-ids cam_000,cam_004" in plan and "--frames all" in plan)
    check("dry-run and preflight created nothing",
          not (RESULTS / "washer_103776" / "unified_phase1_experiments" /
               "wrapper_probe_x").exists())
    rc, out = sh(scene_sh + ["--scene", "washer_103776", "--profile", "washer-validation",
                             "--tag", "wrapper_probe_x", "--preflight"])
    check("--reuse-existing-phase1 is mandatory", rc == 2 and "required" in out, f"exit {rc}")

    print("\n[5] real discovery")
    rc, out = sh(night_sh + ["--all-existing-scenes", "--reuse-existing-phase1",
                             "--tag", "wrapper_probe_x", "--list-scenes"], timeout=900)
    inv = json.loads(out[out.index("{"):out.rindex("}") + 1])
    names = [q["scene"] for q in inv["eligible"]]
    check("washer_103776 is eligible", "washer_103776" in names, f"{len(names)} eligible")
    check("USB_100109 keeps its case", "USB_100109" in names and "usb_100109" not in names)

    print("\n[6] mocked two-scene overnight run, path with spaces")
    with tempfile.TemporaryDirectory(prefix="fused phase1 mock ") as tmp:
        root = Path(tmp) / "results root"
        data = Path(tmp) / "data set"
        for scene in ("AScene_1", "bScene_2", "CBroken_3"):
            d = root / scene
            for e in ("initial", "final"):
                (d / "static_bidi" / e / "point_cloud" / "iteration_30000").mkdir(parents=True)
                (d / "static_bidi" / e / "point_cloud" / "iteration_30000" /
                 "point_cloud.ply").write_bytes(b"ply\n")
                (d / "static_bidi" / e / "cameras.json").write_text(json.dumps(
                    [{"img_name": "cam_000/frame_000000.png", "position": [0, 0, 1],
                      "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]))
            for direction, endpoint in (("forward", "initial"), ("backward", "final")):
                c = d / "bidirectional" / direction / "phase1_centers"
                c.mkdir(parents=True)
                (c / "trajectory_manifest.json").write_text(json.dumps({
                    "direction": ("sideways" if scene == "CBroken_3" and direction
                                  == "forward" else direction),
                    "canonical_endpoint": endpoint,
                    "trajectory_order": "original_chronological",
                    "num_frames": 3, "num_gaussians": 5}))
                for f in range(3):
                    (c / f"centers_frame_{f:04d}.npy").write_bytes(b"x")
            (data / scene / "rgb" / "cam_000").mkdir(parents=True)
            (data / scene / "depth_npy").mkdir(parents=True)
        mock = Path(tmp) / "mock runner.py"
        events = Path(tmp) / "events.jsonl"
        mock.write_text(
            "import sys, json, time\n"
            "scene = sys.argv[sys.argv.index('--scene') + 1]\n"
            f"ev = open({str(events)!r}, 'a')\n"
            "ev.write(json.dumps({'scene': scene, 'start': time.time()}) + '\\n'); ev.flush()\n"
            "time.sleep(1.2)\n"
            "ev.write(json.dumps({'scene': scene, 'end': time.time()}) + '\\n'); ev.flush()\n"
            "sys.exit(1 if scene == 'AScene_1' else 3)\n")
        rc, out = sh(night_sh + ["--all-existing-scenes", "--reuse-existing-phase1",
                                 "--tag", "mock_tag", "--results-root", str(root),
                                 "--dataset-root", str(data), "--resume"],
                     env={"FUSED_PHASE1_SCENE_RUNNER": f"{PY} '{mock}'"})
        ev = [json.loads(l) for l in events.read_text().splitlines()]
        order = [e["scene"] for e in ev]
        check("only the two eligible scenes ran, in sorted order",
              order == ["AScene_1", "AScene_1", "bScene_2", "bScene_2"], str(order))
        a_end = next(e["end"] for e in ev if e["scene"] == "AScene_1" and "end" in e)
        b_start = next(e["start"] for e in ev if e["scene"] == "bScene_2" and "start" in e)
        check("strictly serial: the second starts after the first ends",
              b_start >= a_end, f"gap {b_start - a_end:.2f} s")
        summ = json.loads((root / "unified_phase1_overnight" / "mock_tag" /
                           "ALL_SCENES_SUMMARY.json").read_text())
        st = {r["scene"]: r["status"] for r in summ["rows"]}
        check("a failure is recorded and the next scene still runs",
              st == {"AScene_1": "FAILED", "bScene_2": "PARTIAL"}, str(st))
        check("the overnight exit code says a scene failed", rc == 1, f"exit {rc}")
        skipped = {s["scene"]: s["reasons"] for s in summ["skipped"]}
        print("\n[7] a corrupt trajectory manifest")
        check("the corrupt scene is skipped with its reason, never retrained",
              "CBroken_3" in skipped and any("BAD_FORWARD_TRACK_MANIFEST" in r
                                            for r in skipped["CBroken_3"]),
              str(skipped.get("CBroken_3")))
        check("summary files written",
              all((root / "unified_phase1_overnight" / "mock_tag" / f).is_file()
                  for f in ("ALL_SCENES_SUMMARY.md", "ALL_SCENES_SUMMARY.csv",
                            "eligible_scenes.json")))
        check("the run lock was released", not (root / "unified_phase1_overnight" /
                                                "mock_tag" / ".lock").exists())

    print("\n[8] --resume-from refuses a different posterior")
    prev = (RESULTS / "washer_103776" / "unified_phase1_experiments" /
            "fused_phase1_v1__smoke" / "optimizer" / "attempt_001")
    if (prev / "manifest.json").is_file():
        with tempfile.TemporaryDirectory(prefix="resume probe ") as tmp:
            rc, out = sh([PY, "tools/run_unified_phase1_real_bidirectional.py",
                          "--scene", "washer_103776", "--data-root",
                          str(DATASET / "washer_103776"), "--out-root",
                          str(RESULTS / "washer_103776"), "--mode", "raw_union_experiment",
                          "--representation-policy", "diagnostic-only",
                          "--reuse-existing-phase1", "--frames", "all",
                          "--camera-ids", "cam_001", "--max-gaussians-per-source", "1500",
                          "--max-outer", "1", "--pcg-iters", "2", "--no-visuals",
                          "--resume-from", str(prev), "--out", str(Path(tmp) / "x")],
                         timeout=900)
            check("a different camera set is refused, not continued",
                  rc != 0 and "differs from this run" in out, f"exit {rc}")
    else:
        print("  NOT_RUN: no smoke attempt on disk")

    print("\n[9] the stop hook")
    import torch
    if torch.cuda.is_available():
        sys.path.insert(0, str(REPO))
        from bidirectional_iekf.ieks import IEKSConfig, Trajectory, UnifiedFreeIEKS
        from bidirectional_iekf.observations import FreeRGBDObservation
        from estimator.manifold import so3_exp
        from estimator.renderer_bridge import build_linearization
        from estimator.state import GaussianMotionState
        from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras
        cams = make_cameras(n_cam=1, H=48, W=48)
        st = GaussianMotionState(
            mu=torch.tensor([[0.0, 0.0, 0.0]]).cuda(), R=so3_exp(torch.zeros(1, 3)).cuda(),
            v=torch.zeros(1, 3).cuda(), omega=torch.zeros(1, 3).cuda(),
            colors=torch.tensor([[.8, .2, .2]]).cuda(), opacity=torch.full((1, 1), .9).cuda(),
            scales=torch.full((1, 3), .1).cuda())
        obs = [[(r, d, torch.ones_like(d, dtype=torch.bool))
                for r, d in build_linearization(st, cams).predicted()]] * 2
        warm = Trajectory(states=[st, st], frame_ids=[0, 1], dt=0.1)
        res = UnifiedFreeIEKS(FreeRGBDObservation(cams, obs, frame_ids=[0, 1]),
                              torch.eye(12, dtype=torch.float64)[None],
                              config=IEKSConfig(stop_check=lambda: "STOPPED_BY_USER",
                                                verbose=False)).run(warm)
        check("a stop request ends the run with its own status",
              res["status"] == "STOPPED_BY_USER" and not res["trace"], res["status"])
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
