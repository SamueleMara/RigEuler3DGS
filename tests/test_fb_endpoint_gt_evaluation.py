#!/usr/bin/env python
"""Leakage and regression tests of the evaluation-only GT scoring
(CODEX_ADD_EVALUATION_ONLY_GT_METRICS_AND_ALL_SCENES_RUNNER.md section 5).

  [1] GT changes do not change inference: the REAL diagnosis is run on washer
      data (one camera) in sandbox roots whose GT is (A) real, (B) altered,
      (C) absent; every decision, ID, label and number of the primary outputs
      must match; the GT scores of A and B must differ; C reports GT_UNAVAILABLE
  [2] no GT, same algorithm (C above)
  [3] firewall: GT readable by the evaluator, refused in a diagnosis process;
      the diagnosis never imports the evaluator
  [4] the evaluator cannot rewrite primary results; unsealed / symlinked
      output refused
  [5] joint geometry metrics: parallel / skew / antiparallel / collinear, units
  [6] support confusion with unlabelled rows excluded; row -> label mapping
  [7] scene discovery and the all-scenes runner (paths with spaces, missing
      final PLY, missing dataset, missing joints, existing tagged output,
      failure isolation, no deletion, no rendering in --list-scenes /
      --preflight-only)
  [8] real smoke on the existing washer endpoint_evidence_v2 (dry run, then
      evaluation; every original file unchanged)

    python tests/test_fb_endpoint_gt_evaluation.py [--no-gpu]
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PY = sys.executable
RESULTS = Path(os.environ.get("RESULTS_ROOT", "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"))
DATA = Path(os.environ.get("DATASET_ROOT", "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/"
                           "output/preparation_8cams/partnet_subset_fixed_render_linear"))
SCENE = "washer_103776"
V2 = RESULTS / SCENE / "endpoint_correspondence" / "endpoint_evidence_v2"
CHECKS, FAILURES = 0, []


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def tree_hashes(root: Path, skip=("evaluation_gt",)) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(s in p.relative_to(root).parts for s in skip):
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ---------------------------------------------------------------------------

def t5_joint_geometry():
    from observation_diagnostics import gt_evaluation as G
    print("\n[5] joint geometry metrics")
    check("[5] antiparallel axes are the same line (0 deg)", G.line_angle_deg([1, 0, 0], [-1, 0, 0]) < 1e-9)
    check("[5] perpendicular axes are 90 deg", abs(G.line_angle_deg([1, 0, 0], [0, 0, 2]) - 90) < 1e-9)
    check("[5] parallel lines: distance between them", abs(G.line_distance([0, 0, 0], [1, 0, 0], [5, 0.3, 0.4], [-1, 0, 0]) - 0.5) < 1e-12)
    check("[5] collinear lines: 0", G.line_distance([0, 0, 0], [0, 1, 0], [0, 7, 0], [0, -1, 0]) < 1e-12)
    check("[5] intersecting lines: 0", G.line_distance([0, 0, 0], [1, 0, 0], [3, -1, 0], [0, 1, 0]) < 1e-12)
    check("[5] skew lines: common-perpendicular length", abs(G.line_distance([0, 0, 0], [1, 0, 0], [0, 0, 0.25], [0, 1, 0]) - 0.25) < 1e-12)
    check("[5] nearly parallel handled without blow-up",
          abs(G.line_distance([0, 0, 0], [1, 0, 0], [0, 0.2, 0], [1, 1e-12, 0]) - 0.2) < 1e-6)
    R = G.rodrigues([0, 0, 1], np.pi / 2)
    check("[5] radians in, geodesic in degrees", abs(G.geodesic_deg(np.eye(3), R) - 90) < 1e-9)
    T = G.revolute_T([0, 0, 1], [1, 0, 0], np.pi)
    check("[5] revolute about a point maps it to itself, prismatic in metres",
          np.allclose(G.apply(T, [[1, 0, 0]]), [[1, 0, 0]]) and np.allclose(G.apply(G.prismatic_T([0, 2, 0], 0.3), [[0, 0, 0]]), [[0, 0.3, 0]]))


def t6_support():
    from observation_diagnostics import gt_evaluation as G
    print("\n[6] support confusion and row -> GT mapping")
    pred = np.array([1, 1, 0, 0, 1, 0, 1], bool)
    gt = np.array([1, 0, 1, 0, -1, -1, 1])
    c = G.confusion(pred, gt)
    check("[6] TP/FP/FN/TN with unlabelled rows excluded", (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 1)
          and c["labelled_rows"] == 5 and c["unlabelled_rows_excluded"] == 2, json.dumps(c))
    check("[6] precision, recall, F1, IoU", abs(c["precision"] - 2 / 3) < 1e-12 and abs(c["recall"] - 2 / 3) < 1e-12
          and abs(c["f1"] - 2 / 3) < 1e-12 and abs(c["iou"] - 0.5) < 1e-12)
    c0 = G.confusion(np.zeros(3, bool), np.array([0, 0, 0]))
    check("[6] no positives: precision/recall undefined, not zero-filled", c0["precision"] is None and c0["recall"] is None)
    cloud = (np.array([[0, 0, 0], [1, 0, 0]], float), np.array([0, 1]))
    rows = np.array([[0.005, 0, 0], [0.99, 0, 0], [0.5, 0, 0]])
    lab = G.map_labels(rows, cloud, 0.02)
    check("[6] rows take the nearest GT label within tolerance, else -1 (by original row order)",
          lab.tolist() == [0, 1, -1])


def t3_firewall_and_imports(tmp: Path):
    print("\n[3] GT readable by the evaluator, refused in the diagnosis process")
    ds = tmp / "data root" / "scn"
    (ds / "metadata").mkdir(parents=True)
    (ds / "pointcloud").mkdir()
    (ds / "metadata" / "joint_info.json").write_text(json.dumps(
        {"selected_joint": {"name": "j", "type": "revolute", "dof": 1}}))
    (ds / "pointcloud" / "joint_metadata.json").write_text(json.dumps(
        {"name": "j", "type": "revolute", "axis_vector": [1, 0, 0], "point_of_application": [0, 0, 0],
         "coordinate_frame": "render_world"}))
    (ds / "metadata" / "frame_values.csv").write_text("frame_index,j\n0,0\n1,0.5\n")
    code_eval = ("import sys; sys.path.insert(0, %r)\n"
                 "from observation_diagnostics.gt_evaluation import GroundTruth\n"
                 "g = GroundTruth(%r); print(g.joint['type'], list(g.q))") % (str(REPO), str(ds))
    r = subprocess.run([PY, "-c", code_eval], capture_output=True, text=True)
    check("[3] the evaluator process reads metadata/ and pointcloud/ GT", r.returncode == 0 and "revolute" in r.stdout,
          r.stdout.strip() + r.stderr[-200:])
    code_diag = ("import sys, json; sys.path.insert(0, %r)\n"
                 "from observation_diagnostics.gt_firewall import Firewall, GTAccessError\n"
                 "Firewall.install([%r])\n"
                 "try:\n    json.load(open(%r)); print('READ')\n"
                 "except GTAccessError as e:\n    print('REFUSED')") % (str(REPO), str(ds.parent), str(ds / "metadata" / "joint_info.json"))
    r = subprocess.run([PY, "-c", code_diag], capture_output=True, text=True)
    check("[3] the diagnosis firewall refuses the same file", r.stdout.strip() == "REFUSED", r.stdout + r.stderr[-200:])
    code_imp = ("import sys; sys.path.insert(0, %r)\n"
                "import tools.analyze_fb_endpoint_correspondence as t\n"
                "import observation_diagnostics.endpoint_compare, observation_diagnostics.joint_evidence\n"
                "print('LEAK' if 'observation_diagnostics.gt_evaluation' in sys.modules else 'CLEAN')") % str(REPO)
    r = subprocess.run([PY, "-c", code_imp], capture_output=True, text=True)
    check("[3] importing the diagnosis does not import the evaluator", r.stdout.strip().endswith("CLEAN"),
          r.stdout[-100:] + r.stderr[-200:])
    src = "".join((REPO / f).read_text() for f in (
        "tools/analyze_fb_endpoint_correspondence.py", "observation_diagnostics/endpoint_compare.py",
        "observation_diagnostics/joint_evidence.py", "observation_diagnostics/endpoint_visuals.py",
        "observation_diagnostics/pair_evidence.py", "observation_diagnostics/pixel_evidence.py",
        "observation_diagnostics/gaussian_evidence.py", "observation_diagnostics/sources.py"))
    import re
    imp = re.findall(r"^\s*(?:from\s+\S*gt_evaluation\S*\s+import|import\s+\S*gt_evaluation)", src, re.M)
    check("[3] no diagnosis source file imports the evaluator module", not imp, str(imp))


def t4_containment(tmp: Path):
    print("\n[4] the evaluator cannot rewrite primary results")
    if not V2.is_dir():
        check("[4] washer endpoint_evidence_v2 present", False, str(V2))
        return
    d = tmp / "diag copy"
    d.mkdir(parents=True)
    for f in ("run_state.json", "provenance.json", "endpoint_summary.json", "joint_estimate_provenance.json"):
        shutil.copy(V2 / f, d / f)
    (tmp / "elsewhere").mkdir()
    os.symlink(tmp / "elsewhere", d / "evaluation_gt")
    r = subprocess.run([PY, str(REPO / "tools/evaluate_fb_endpoint_correspondence_gt.py"), "--diagnosis-dir", str(d),
                        "--dataset-dir", str(DATA / SCENE), "--results-dir", str(RESULTS / SCENE)],
                       capture_output=True, text=True)
    check("[4] a symlinked evaluation_gt is refused", r.returncode == 2 and "symlink" in r.stdout, r.stdout[-200:])
    check("[4] nothing written through the symlink", not any((tmp / "elsewhere").iterdir()))
    os.unlink(d / "evaluation_gt")
    st = json.loads((d / "run_state.json").read_text())
    st["complete"] = False
    (d / "run_state.json").write_text(json.dumps(st))
    r = subprocess.run([PY, str(REPO / "tools/evaluate_fb_endpoint_correspondence_gt.py"), "--diagnosis-dir", str(d),
                        "--dataset-dir", str(DATA / SCENE)], capture_output=True, text=True)
    check("[4] an unsealed diagnosis is refused", r.returncode == 2 and "not a sealed" in r.stdout, r.stdout[-200:])
    check("[4] and nothing was created", not (d / "evaluation_gt").exists())


def t8_real_smoke():
    print("\n[8] real smoke on the existing washer endpoint_evidence_v2")
    if not V2.is_dir():
        check("[8] washer endpoint_evidence_v2 present", False)
        return
    before = tree_hashes(V2)
    r = subprocess.run([PY, str(REPO / "tools/evaluate_fb_endpoint_correspondence_gt.py"), "--diagnosis-dir", str(V2),
                        "--dataset-dir", str(DATA / SCENE), "--dry-run"], capture_output=True, text=True)
    ok = r.returncode == 0
    rep = json.loads(r.stdout) if ok else {}
    check("[8] dry run lists the real GT files and writes nothing",
          ok and rep.get("gt_status") == "GT_AVAILABLE" and rep.get("writes", "").startswith("NONE")
          and tree_hashes(V2) == before, json.dumps(rep.get("metrics_available")))
    r = subprocess.run([PY, str(REPO / "tools/evaluate_fb_endpoint_correspondence_gt.py"), "--diagnosis-dir", str(V2),
                        "--dataset-dir", str(DATA / SCENE), "--resume"], capture_output=True, text=True)
    # an evaluation made by an older evaluator is refused by --resume (fingerprint
    # changed) and kept intact; superseding it on real results is the user's call
    refused_old = r.returncode != 0 and "evaluator changed" in r.stdout and "nothing deleted" in r.stdout
    check("[8] evaluation on the existing result completes, is reused, or (older evaluator) is refused intact",
          r.returncode == 0 or refused_old, r.stdout[-200:])
    after = tree_hashes(V2)
    check("[8] every original file of the diagnosis is unchanged", after == before, f"{len(before)} files")
    S = json.loads((V2 / "evaluation_gt" / "gt_evaluation_summary.json").read_text())
    C = json.loads((V2 / "evaluation_gt" / "evaluation_checks.json").read_text())
    check("[8] evaluator's own manifest check agrees; writes contained",
          S["primary_outputs_unchanged"] and C["all_writes_under_evaluation_gt"] and not C["evaluation_dir_is_symlink"])
    rows = list(csv.DictReader(open(V2 / "evaluation_gt" / "metrics_joint.csv")))
    hyps = {(r["direction"], r["hypothesis"]) for r in rows}
    check("[8] both hypotheses scored per direction, UNDECIDED kept, degenerate flagged",
          {("forward", "revolute"), ("forward", "prismatic"), ("backward", "revolute"), ("backward", "prismatic")} <= hyps
          and all(r["decided_type"] == "UNDECIDED" for r in rows)
          and any(r["direction"] == "forward" and r["degenerate_estimate"] == "True" for r in rows))
    for f in ("metrics_joint.csv", "metrics_support.csv", "metrics_geometry.csv", "metrics_patches.csv"):
        with open(V2 / "evaluation_gt" / f) as fh:
            hdr = fh.readline()
        check(f"[8] {f} written with a header", len(hdr.split(",")) > 3)
    prim = "".join(open(V2 / f).read() for f in ("backward_only_observed_patches.csv", "endpoint_summary.json"))
    check("[8] no GT metric name appears in a primary table", "gt_evaluation_metric" not in prim
          and "posthoc_gt_reading" not in prim)


# ---------------------------------------------------------------------------
# [1, 2] the real diagnosis under three GT variants
# ---------------------------------------------------------------------------

def sandbox(tmp: Path, variant: str):
    res = tmp / f"results {variant}" / SCENE
    res.mkdir(parents=True)
    for d in ("static_bidi", "bidirectional"):
        os.symlink(RESULTS / SCENE / d, res / d)
    ds = tmp / f"dataset {variant}" / SCENE
    ds.mkdir(parents=True)
    for d in ("rgb", "depth_npy", "colmap", "colmap_final"):
        os.symlink(DATA / SCENE / d, ds / d)
    if variant in ("A", "B"):
        shutil.copytree(DATA / SCENE / "metadata", ds / "metadata")
        shutil.copytree(DATA / SCENE / "pointcloud", ds / "pointcloud")
        os.symlink(DATA / SCENE / "mask", ds / "mask")
    if variant == "B":     # different GT: another axis, another point, scaled q
        jm = json.loads((ds / "pointcloud" / "joint_metadata.json").read_text())
        jm["axis_vector"], jm["point_of_application"] = [0.0, 1.0, 0.0], [0.3, 0.1, 0.2]
        (ds / "pointcloud" / "joint_metadata.json").write_text(json.dumps(jm))
        lines = (ds / "metadata" / "frame_values.csv").read_text().splitlines()
        (ds / "metadata" / "frame_values.csv").write_text(
            "\n".join([lines[0]] + [f"{l.split(',')[0]},{0.5 * float(l.split(',')[1])}" for l in lines[1:]]) + "\n")
    return res.parent, ds.parent


def normalise(obj):
    if isinstance(obj, dict):
        return {k: normalise(v) for k, v in obj.items()
                if k not in ("runtime_s", "peak_gpu_mib", "peak_host_rss_mib", "finished", "started")}
    if isinstance(obj, list):
        return [normalise(v) for v in obj]
    if isinstance(obj, float):
        return float(f"{obj:.6g}")
    if isinstance(obj, str):
        return obj.replace("results A", "results X").replace("results B", "results X").replace(
            "results C", "results X").replace("dataset A", "dataset X").replace("dataset B", "dataset X").replace(
            "dataset C", "dataset X")
    return obj


def csv_norm(p: Path):
    rows = list(csv.reader(open(p)))
    out = []
    for r in rows:
        rr = []
        for x in r:
            try:
                rr.append(f"{float(x):.6g}")
            except ValueError:
                rr.append(normalise(x))
        out.append(rr)
    return out


def t1_t2_leakage(tmp: Path):
    print("\n[1, 2] the real diagnosis with real, altered and absent GT")
    outs = {}
    for v in ("A", "B", "C"):
        rr, dr = sandbox(tmp, v)
        out = rr / SCENE / "endpoint_correspondence" / "t"
        cmd = [PY, "-W", "ignore", str(REPO / "tools/analyze_fb_endpoint_correspondence.py"), "--scene", SCENE,
               "--results-root", str(rr), "--dataset-root", str(dr), "--tag", "t", "--cameras", "cam_000",
               "--execute", "--gpu-memory-fraction", os.environ.get("FB_TEST_GPU_FRACTION", "0.2")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        check(f"[1] diagnosis with GT variant {v} completes", r.returncode == 0, (r.stdout + r.stderr)[-300:])
        outs[v] = (out, dr)
        if r.returncode:
            return
        fw = json.loads((out / "provenance.json").read_text())["gt_firewall"]
        check(f"[1] variant {v}: the diagnosis opened no GT file", not fw["refused"] and not any(
            ("metadata" in k or "pointcloud" in k) for k in fw["opened_by_directory"]))
    a, b, c = (outs[v][0] for v in "ABC")
    files = sorted(str(p.relative_to(a)) for p in a.rglob("*") if p.is_file() and p.suffix in (".json", ".csv")
                   and "evaluation_gt" not in p.parts and p.name not in ("log.txt",))
    diffs = {"B": [], "C": []}
    for f in files:
        for v, other in (("B", b), ("C", c)):
            q = other / f
            if not q.is_file():
                diffs[v].append(f + " (missing)")
                continue
            if f.endswith(".csv"):
                same = csv_norm(a / f) == csv_norm(q)
            elif f in ("provenance.json", "tests_and_checks.json", "preflight.json", "run_state.json"):
                continue      # paths, hashes of sandbox inputs, timing
            else:
                same = normalise(json.loads((a / f).read_text())) == normalise(json.loads(q.read_text()))
            if not same:
                diffs[v].append(f)
    check("[1] real vs altered GT: identical primary outputs (IDs, labels, J0/J1, numbers to 6 digits)",
          not diffs["B"], f"{len(files)} files compared; differing: {diffs['B'][:5]}")
    check("[2] real vs absent GT: identical primary outputs", not diffs["C"], f"differing: {diffs['C'][:5]}")
    for v in "ABC":
        out, dr = outs[v]
        subprocess.run([PY, str(REPO / "tools/evaluate_fb_endpoint_correspondence_gt.py"), "--diagnosis-dir", str(out),
                        "--dataset-dir", str(dr / SCENE)], capture_output=True, text=True)
    ja = list(csv.DictReader(open(a / "evaluation_gt" / "metrics_joint.csv")))
    jb = list(csv.DictReader(open(b / "evaluation_gt" / "metrics_joint.csv")))
    ang = lambda rows: [r.get("axis_direction_angle_deg_vs_gt_axis") for r in rows]  # noqa: E731
    check("[1] the GT evaluation scores DO change with the GT", ang(ja) != ang(jb), f"{ang(ja)[:2]} vs {ang(jb)[:2]}")
    sc = json.loads((c / "evaluation_gt" / "gt_evaluation_summary.json").read_text())
    check("[2] without GT the evaluator reports GT_UNAVAILABLE and scores nothing",
          sc["status"] == "GT_UNAVAILABLE" and sc["counts"]["joint_rows"] == 0, sc["status"])


# ---------------------------------------------------------------------------
# [7] discovery and the all-scenes runner
# ---------------------------------------------------------------------------

def t7_runner(tmp: Path):
    print("\n[7] scene discovery and the all-scenes runner")
    R = tmp / "results root with spaces"
    D = tmp / "dataset root with spaces"
    R.mkdir(parents=True)
    D.mkdir(parents=True)

    def make_scene(name, final=True, joints=True, data=True, depth=True):
        s = R / name
        (s / "static_bidi" / "initial" / "point_cloud").mkdir(parents=True)
        os.symlink(RESULTS / SCENE / "static_bidi" / "initial" / "point_cloud" / "iteration_30000",
                   s / "static_bidi" / "initial" / "point_cloud" / "iteration_30000")
        for f in ("cfg_args", "cameras.json"):
            if (RESULTS / SCENE / "static_bidi" / "initial" / f).exists():
                os.symlink(RESULTS / SCENE / "static_bidi" / "initial" / f, s / "static_bidi" / "initial" / f)
        if final:
            os.symlink(RESULTS / SCENE / "static_bidi" / "final", s / "static_bidi" / "final")
        (s / "bidirectional").mkdir()
        for d in ("forward", "backward"):
            os.symlink(RESULTS / SCENE / "bidirectional" / d, s / "bidirectional" / d)
            if joints:
                os.symlink(RESULTS / SCENE / "bidirectional" / f"{d}_rigid", s / "bidirectional" / f"{d}_rigid")
        if data:
            ds = D / name
            ds.mkdir()
            for d in ("rgb", "colmap", "colmap_final") + (("depth_npy",) if depth else ()):
                os.symlink(DATA / SCENE / d, ds / d)
    make_scene("scene_ok")
    make_scene("scene_nofinal", final=False)
    make_scene("scene_nojoints", joints=False)
    make_scene("scene_nodata", data=False)
    make_scene("scene_nodepth", depth=False)
    (R / "not_a_scene").mkdir()
    (R / "not_a_scene" / "notes.txt").write_text("x")
    (R / "a_file.txt").write_text("x")
    listing0 = sorted(str(p) for p in R.rglob("*"))
    script = REPO / "scripts" / "run_fb_endpoint_correspondence_all.sh"
    base = ["bash", str(script), "--results-root", str(R), "--dataset-root", str(D), "--estimated-joints", "auto"]
    r = subprocess.run(base + ["--list-scenes"], capture_output=True, text=True)
    out = r.stdout
    elig = out.split("eligible scenes")[1].split("skipped")[0]
    skip = out.split("skipped")[1]
    check("[7] --list-scenes finds exactly the complete scenes (spaces in roots)",
          r.returncode == 0 and all(s in elig for s in ("scene_ok", "scene_nojoints", "scene_nodepth"))
          and "scene_nofinal" not in elig and "scene_nodata" not in elig, elig.strip().replace("\n", ","))
    check("[7] and lists every skipped directory with its reason",
          "scene_nofinal: no static_bidi/final PLY" in skip and "scene_nodata: no dataset directory" in skip
          and "not_a_scene" in skip)
    check("[7] --list-scenes writes nothing", sorted(str(p) for p in R.rglob("*")) == listing0)
    r = subprocess.run(base + ["--tag", "tt", "--preflight-only"], capture_output=True, text=True)
    st = {row["scene"]: row for row in csv.DictReader(open(R / "endpoint_correspondence_batch" / "tt" / "scenes_status.tsv"),
                                                       delimiter="\t")}
    check("[7] --preflight-only: preflight per scene, failures isolated",
          st.get("scene_ok", {}).get("preflight") == "OK" and st.get("scene_nodepth", {}).get("preflight", "").startswith("BLOCKED")
          and st.get("scene_nojoints", {}).get("preflight") == "OK", json.dumps({k: v["preflight"] for k, v in st.items()}))
    check("[7] --preflight-only renders nothing (no run_state anywhere)", not list(R.glob("*/endpoint_correspondence/tt/run_state.json")))
    pj = json.loads((R / "scene_nojoints" / "endpoint_correspondence" / "tt" / "preflight.json").read_text())
    check("[7] missing joint estimates do not block J0 under auto",
          all(e["status"] == "UNAVAILABLE" for e in pj["estimated_joints"].values()) and not pj["blockers"])
    # an existing tagged experiment with another fingerprint is refused, not overwritten
    old = R / "scene_ok" / "endpoint_correspondence" / "tt"
    (old / "run_state.json").write_text(json.dumps({"fingerprint": "deadbeef" * 8, "complete": True}))
    (old / "precious.txt").write_text("keep me")
    r = subprocess.run(base + ["--tag", "tt", "--execute", "--scenes", "scene_ok"], capture_output=True, text=True)
    st = {row["scene"]: row for row in csv.DictReader(open(R / "endpoint_correspondence_batch" / "tt" / "scenes_status.tsv"),
                                                       delimiter="\t")}
    check("[7] an existing tagged experiment is refused and left intact",
          st["scene_ok"]["diagnosis"].startswith("FAILED") and (old / "precious.txt").read_text() == "keep me"
          and json.loads((old / "run_state.json").read_text())["fingerprint"] == "deadbeef" * 8,
          json.dumps(st["scene_ok"]))
    check("[7] the GT evaluation is not run on an unsealed/foreign result",
          st["scene_ok"]["gt_evaluation"].startswith("NOT_RUN") or st["scene_ok"]["gt_evaluation"] == "-")
    check("[7] no source input deleted", all(Path(p).exists() or Path(p).is_symlink() for p in listing0))
    check("[7] batch tables written outside the scenes", (R / "endpoint_correspondence_batch" / "tt" / "scenes_status.csv").is_file())


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        t5_joint_geometry()
        t6_support()
        t3_firewall_and_imports(td / "t3")
        t4_containment(td / "t4")
        t7_runner(td / "t7")
        t8_real_smoke()
        if "--no-gpu" not in sys.argv:
            t1_t2_leakage(td / "t1")
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
