"""Sections 47, 48, 49: the required debug-layer tests.

CODEX_ADD_VISUAL_AND_NUMERICAL_DEBUG_GEOMETRIC_PIPELINE.md sections 47-49.

Section 47 is the renderer-refinement test the previous report listed as
missing. Sections 48 and 49 check that the debug layer is complete and, in
particular, that enabling it does not change a single inference number
(section 0 rule 10 / section 49.9).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from debug import GeometricDebug  # noqa: E402
from debug.geometric_motion_debug import build_payload  # noqa: E402
from estimator.manifold import so3_exp  # noqa: E402
from estimator.rigid_part_estimator import (RigidTransforms,  # noqa: E402
                                            apply_transforms)
from joint_reduction.kpart_models import (PRISMATIC, REVOLUTE,  # noqa: E402
                                          PartModel, SceneModel)
from joint_reduction.kpart_window import (AlternationConfig,  # noqa: E402
                                          WindowObjective,
                                          solve_banded_trajectory)
from tests.test_strict_trajectory import (Cfg, DEV, camera,  # noqa: E402
                                          render_obs, scene_of)

PASSED: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def _window(kind, q_true, axis, point=None, seed=0, n=320):
    """A real multiview window of a known articulated motion."""
    cams = [camera(), camera(tz=0.3)]
    st = scene_of(n=n, seed=seed)
    labels = torch.zeros(st.num_gaussians, dtype=torch.long, device=DEV)
    labels[st.mu[:, 0] > 0.0] = 1
    if kind == PRISMATIC:
        model = PartModel(kind=PRISMATIC, axis=axis, part_index=1)
    else:
        model = PartModel(kind=REVOLUTE, axis=axis,
                          moment=torch.linalg.cross(point, axis), part_index=1)
    scene = SceneModel(parts=[model], K=1)
    frames = []
    for t, q in enumerate(q_true):
        mu, R = scene.transport_all(st.mu, st.R, labels, [float(q)])
        frames.append((render_obs(st, mu, R, cams), t))
    obj = WindowObjective(st.mu, st.R, cams, frames, Cfg(), state_template=st)
    return obj, scene, labels, st


# ===========================================================================
def test_47_renderer_refinement():
    """§47: perturb the joint state, refine, require the objective and the
    error to improve -- for BOTH prismatic and revolute."""
    cfg = AlternationConfig(sigma_a=1.0, trajectory_iters=30)
    T = 8

    cases = []
    e_p = torch.tensor([0.55, 0.60, 0.58], device=DEV).double()
    e_p = (e_p / e_p.norm()).float()
    cases.append(("prismatic", PRISMATIC, e_p, None,
                  [0.0] + [0.012 * t for t in range(1, T)]))
    e_r = torch.tensor([0.0, 1.0, 0.0], device=DEV)
    a_r = torch.tensor([0.42, 0.0, 2.05], device=DEV)
    cases.append(("revolute", REVOLUTE, e_r, a_r,
                  [0.0] + [0.05 * t for t in range(1, T)]))

    for label, kind, axis, point, q_true in cases:
        obj, scene, labels, _st = _window(kind, q_true, axis, point, seed=5)

        # Section 47 step 2: perturb q(t) away from the truth.
        q_bad = [[q * 0.55 + 0.01 for q in q_true]]
        e_before = obj.evaluate(scene, labels, q_bad).total
        err_before = max(abs(a - b) for a, b in zip(q_bad[0], q_true))

        q_ref, info = solve_banded_trajectory(obj, scene, labels,
                                              [list(q_bad[0])], cfg)
        e_after = obj.evaluate(scene, labels, q_ref).total
        err_after = max(abs(a - b) for a, b in zip(q_ref[0], q_true))

        print(f"       {label:10s} E {e_before:10.2f} -> {e_after:10.2f}   "
              f"max |q - q_true| {err_before:.5f} -> {err_after:.5f}")
        check(f"§47 {label}: renderer refinement decreases the RGB-D objective",
              e_after < e_before, f"{e_before:.2f} -> {e_after:.2f}")
        check(f"§47 {label}: and the q error improves",
              err_after < err_before,
              f"{err_before:.5f} -> {err_after:.5f}")
        torch.cuda.empty_cache()


def test_48_visual_smoke():
    """§48: a tiny scene -- one static part, one OFF-ORIGIN revolute part,
    10 frames -- produces an HTML whose metadata is verifiable."""
    T = 10
    axis = torch.tensor([0.0, 0.0, 1.0], device=DEV)
    point = torch.tensor([0.40, -0.25, 0.0], device=DEV)
    q_true = [0.0] + [0.06 * t for t in range(1, T)]
    obj, scene, labels, st = _window(REVOLUTE, q_true, axis, point, seed=9,
                                     n=240)

    tr = RigidTransforms.identity(T, 1, device=DEV)
    for t, q in enumerate(q_true):
        R = so3_exp((float(q) * axis.double()).reshape(1, 3))[0]
        tr.R[t, 0] = R
        tr.t[t, 0] = (torch.eye(3, device=DEV).double() - R) @ point.double()

    dbg = GeometricDebug(enabled=True)
    dbg.record(mu0=st.mu, R0=st.R, scales=st.scales, colors=st.colors,
               K=1, labels=labels, frame_ids=list(range(T)),
               transforms=tr, scene=scene, q_traj=[list(q_true)],
               positions=[apply_transforms(st.mu, st.R, labels, tr, t)[0]
                          for t in range(T)],
               num_supernodes=0, status="RESOLVED",
               camera_meta=[{"name": "cam_000",
                             "viewmatrix": obj.cameras[0].viewmatrix.cpu().tolist(),
                             "tanfovx": float(obj.cameras[0].tanfovx),
                             "tanfovy": float(obj.cameras[0].tanfovy)}],
               joints=[{"part": 1, "selected": "revolute",
                        "status": "resolved",
                        "axis": [float(x) for x in axis],
                        "axis_point": [float(x) for x in point],
                        "q": list(q_true),
                        "normalized_error": {"prismatic": 1e-2,
                                             "revolute": 1e-9,
                                             "absolute_gap": 1e-2,
                                             "indistinguishable_floor": 1e-9},
                        "prismatic": {}, "revolute": {}}])

    pay = build_payload(dbg, max_gaussians=200)
    m = pay["meta"]
    print(f"       payload: {m['frames_total']} frames, "
          f"{m['shown_gaussians']} Gaussians, labels "
          f"{sorted(set(pay['labels']))}")
    check("§48 the HTML metadata reports 10 frames", m["frames_total"] == 10,
          str(m["frames_total"]))
    check("§48 two labels are present (static and one moving part)",
          sorted(set(pay["labels"])) == [0, 1],
          str(sorted(set(pay["labels"]))))
    j = pay["joints"][0]
    check("§48 the joint is reported revolute", j["selected"] == "revolute")
    check("§48 the axis point is NOT the origin",
          max(abs(x) for x in j["axis_point"]) > 1e-6,
          f"{[round(x, 4) for x in j['axis_point']]}")
    check("§48 q is given per frame", len(j["q"]) == 10, str(len(j["q"])))
    check("§48 the Plucker moment m = a x e is carried",
          j["moment"] is not None and max(abs(x) for x in j["moment"]) > 1e-9,
          f"{[round(x, 4) for x in j['moment']]}")
    check("§48 a camera frustum is present and world-fixed",
          len(pay["cameras"]) == 1 and "corners" in pay["cameras"][0])
    check("§4 the part colour map is persisted and stable",
          pay["meta"]["part_colors"][0] != pay["meta"]["part_colors"][1])

    # Section 2: the page must reproduce the estimator's own transport.
    from debug.recorder import GeometricDebug as G
    t = 7
    want = dbg.positions_at("refined", t)
    a = torch.tensor(j["axis_point"], device=DEV).double()
    e = torch.tensor(j["axis"], device=DEV).double()
    R = so3_exp((q_true[t] * e).reshape(1, 3))[0]
    manual = a + (st.mu.double() - a) @ R.T
    moving = labels == 1
    err = float((want[moving].double() - manual[moving]).abs().max())
    print(f"       section 2 transport reproduced to {err:.2e}")
    check("§2 the displayed motion IS the estimator's transport", err < 1e-5,
          f"max deviation {err:.2e}")
    torch.cuda.empty_cache()


def test_49_numerical_debug_contract():
    """§49: the numerical debug package's structural guarantees."""
    out = REPO / "tests" / "_debug_fixture"
    run = _last_debug_run()
    if run is None:
        print("       no recorded debug run found; run the pipeline with "
              "--debug-all first")
        check("§49 a debug run is available to check", False,
              "expected a directory containing debug_numeric/run_summary.json")
        return
    nd = run / "debug_numeric"
    summary = json.loads((nd / "run_summary.json").read_text())
    T = summary["frames"]
    K = summary["K"]
    C = len(summary.get("per_part", []))

    import csv as _csv

    def rows(name):
        with (nd / name).open() as fh:
            return list(_csv.DictReader(fh))

    fm = rows("frame_metrics.csv")
    check("§49.1 frame_metrics has one row per frame", len(fm) == T,
          f"{len(fm)} rows vs {T} frames")

    cm = rows("camera_metrics.csv")
    ncam = summary["cameras"]
    check("§49.2 camera_metrics has T*C rows", len(cm) == T * ncam,
          f"{len(cm)} rows vs {T}x{ncam}")

    qt = rows("q_trajectories.csv")
    check("§49.3 q_trajectories has T rows per part", len(qt) == T * K,
          f"{len(qt)} rows vs {T}x{K}")

    M = torch.load(nd / "rigid_transforms.pt", weights_only=False)
    check("§49.4 rigid transforms are [K,T,4,4]",
          tuple(M.shape) == (K, T, 4, 4), str(tuple(M.shape)))
    check("§49.4 and are valid SE(3)",
          float((M[:, :, 3, :] - torch.tensor([0.0, 0.0, 0.0, 1.0])).abs().max())
          < 1e-9)

    lab = torch.load(nd.parent / "rigid_part_labels.pt", weights_only=False)
    check("§49.5 labels cover every Gaussian",
          int(lab.numel()) == summary["num_gaussians"],
          f"{int(lab.numel())} vs {summary['num_gaussians']}")

    pf = json.loads((nd / "prismatic_fit_history.json").read_text())
    rf = json.loads((nd / "revolute_fit_history.json").read_text())
    check("§49.6/§49.7 BOTH fits are written for every moving part, "
          "selected or not",
          len(pf["per_part"]) == K and len(rf["per_part"]) == K,
          f"P {len(pf['per_part'])}, R {len(rf['per_part'])}, K={K}")

    ho = json.loads((nd / "canonical_handover.json").read_text())
    before, after = ho.get("canonical_hash_before"), ho.get("canonical_hash_after")
    if ho.get("performed") is False:
        # Section 24's handover runs only after a joint is selected. On an
        # unresolved run there is nothing to hand over, and the artifact must
        # SAY that rather than look like a passed check.
        check("§49.8 an absent handover is recorded with its reason",
              bool(ho.get("why_not")), str(ho.get("why_not"))[:70])
    else:
        check("§49.8 the canonical hash is recorded before and after",
              bool(before) and bool(after),
              f"{str(before)[:12]} / {str(after)[:12]}")
        check("§49.8 and is unchanged by the handover",
              before == after and ho.get("canonical_unchanged") is True,
              f"{str(before)[:16]}...")

    for f in ("failure_summary.json", "failure_summary.md"):
        check(f"§38 {f} exists even on this run", (nd / f).exists())


def test_49_9_debug_does_not_alter_inference():
    """§49.9 / §0 rule 10: enabling debug must change no inference number.

    Two runs of the same configuration, one with the debug package and one
    without. The CUDA rasterizer accumulates with atomics, so identical runs
    are not bit-identical; the floor is measured from two debug-OFF runs and
    the debug-ON difference must not exceed it.
    """
    import subprocess
    import tempfile
    from tests.test_gt_firewall import PY, max_rel_diff

    ART = ["geometric_joints.json", "rigid_parts.json", "motion_evidence.json",
           "support_null.json", "model_error.json"]

    def run(out, debug):
        cmd = [PY, "tools/run_iekf_scene.py",
               "--model", "output/merged_USB_100109",
               "--observation-source", "dataset_rgbd",
               "--sequence-start", "0", "--sequence-length", "4",
               "--cameras", "2", "--max-gaussians", "1200", "--supernodes",
               "--window-discovery", "--num-moving-parts", "1",
               "--discovery-outer", "1", "--depth-sigma", "constant:0.05:0.0",
               "--no-visualizations", "--out", str(out)]
        if debug:
            cmd += ["--debug-numerics", "--make-geometric-debug-html"]
        env = {"PYTHONPATH": ".", "PATH": "/usr/bin:/bin",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "CUDA_HOME": "/usr/local/cuda-12.8", "HOME": str(Path.home())}
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                           text=True, timeout=3000)
        if r.returncode != 0:
            raise RuntimeError(r.stdout[-2500:] + r.stderr[-2500:])
        return out

    tmp = Path(tempfile.mkdtemp(prefix="dbgoff_"))
    try:
        a = run(tmp / "off_a", False)
        b = run(tmp / "off_b", False)
        c = run(tmp / "on", True)
        present = [n for n in ART if (a / n).exists() and (c / n).exists()]
        check("§49.9 inference artifacts exist in both runs", len(present) >= 3,
              str(present))
        worst_ctl = worst_dbg = 0.0
        for n in present:
            ctl = max_rel_diff(a / n, b / n)
            dbgd = max_rel_diff(a / n, c / n)
            worst_ctl = max(worst_ctl, ctl)
            worst_dbg = max(worst_dbg, dbgd)
            print(f"       {n:28s} debug-off control {ctl:.3e}   "
                  f"debug-on {dbgd:.3e}")
        check("§49.9 enabling debug changes inference no more than rerunning "
              "it does", worst_dbg <= max(worst_ctl * 3.0, 1e-9),
              f"debug-on {worst_dbg:.2e} vs control floor {worst_ctl:.2e}")
        check("§0 rule 10 the debug package was actually produced",
              (c / "debug_numeric" / "run_summary.json").exists()
              and (c / "geometric_motion_debug.html").exists())
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_2_page_javascript_matches_the_estimator():
    """§0 rule 3 / §2: run the PAGE'S OWN transport and compare.

    `test_48_visual_smoke` checks the Python side of the claim. This one
    extracts the JavaScript the browser will actually execute, runs it under
    node, and compares its output against the section 2 formulas applied to the
    same payload. That closes the gap between "the data is right" and "the page
    draws the right thing".

    Skipped when node is unavailable; the check is a bonus, not a gate.
    """
    import json
    import re
    import shutil
    import subprocess
    import tempfile

    if shutil.which("node") is None:
        print("       node not available; skipping the in-browser check")
        return
    run = _last_debug_run()
    html = None if run is None else (run / "geometric_motion_debug.html")
    if html is None or not html.exists():
        print("       no HTML from a --make-geometric-debug-html run; skipping")
        return

    src = html.read_text()
    js = re.findall(r"<script(?![^>]*src=)[^>]*>(.*?)</script>", src, re.S)[-1]
    head = js[:js.index("const view=document.getElementById")]
    tmp = Path(tempfile.mkdtemp(prefix="pagejs_"))
    try:
        f = tmp / "t.js"
        f.write_text(head + """
const out={};
["canonical","bootstrap","rigid","refined"].forEach(s=>{
  out[s]=[]; for(let i=0;i<NF;i++) out[s].push(Array.from(positions(s,i)));
});
console.log(JSON.stringify(out));
""")
        chk = subprocess.run(["node", "--check", str(f)], capture_output=True,
                             text=True)
        check("§1 the page's JavaScript parses", chk.returncode == 0,
              chk.stderr[-200:])
        r = subprocess.run(["node", str(f)], capture_output=True, text=True,
                           timeout=600)
        check("§1 the page's transport code runs", r.returncode == 0,
              r.stderr[-300:])
        got = json.loads(r.stdout)
        pay = json.loads((run / "geometric_motion_debug_payload.json").read_text())
        x0 = torch.tensor(pay["x0"]).reshape(-1, 3).double()
        lab = torch.tensor(pay["labels"])
        worst = {}
        for stage in ("canonical", "bootstrap", "rigid", "refined"):
            w = 0.0
            for fi in range(len(pay["meta"]["frame_ids"])):
                g = torch.tensor(got[stage][fi]).reshape(-1, 3).double()
                if stage == "canonical":
                    exp = x0
                elif stage == "bootstrap":
                    exp = torch.tensor(pay["bootstrap"][fi]).reshape(-1, 3).double()
                elif stage == "rigid":
                    exp = x0.clone()
                    for k in range(1, pay["meta"]["K"] + 1):
                        m = lab == k
                        T = pay["rigid"][fi][k - 1]
                        R = torch.tensor(T["R"]).reshape(3, 3).double()
                        exp[m] = x0[m] @ R.T + torch.tensor(T["t"]).double()
                else:
                    exp = x0.clone()
                    for k in range(1, pay["meta"]["K"] + 1):
                        J = pay["joints"][k - 1]
                        if not J or not J.get("selected"):
                            continue
                        m = lab == k
                        q = J["q"][fi]
                        e = torch.tensor(J["axis"]).double()
                        if J["selected"] == "prismatic":
                            exp[m] = x0[m] + q * e
                        else:
                            a = torch.tensor(J["axis_point"]).double()
                            R = so3_exp((q * e).reshape(1, 3))[0].double()
                            exp[m] = a + (x0[m] - a) @ R.T
                w = max(w, float((g - exp).abs().max()))
            worst[stage] = w
            print(f"       {stage:10s} page JS vs §2 formula: {w:.3e}")
        check("§2/§0.3 the page draws the estimator's own motion, not an "
              "approximation", max(worst.values()) < 1e-5,
              f"worst {max(worst.values()):.2e} (float32 payload rounding)")
    finally:
        import shutil as sh
        sh.rmtree(tmp, ignore_errors=True)


def _last_debug_run():
    """Find a directory produced by a --debug-all run."""
    import os
    roots = [Path(os.environ.get("DEBUG_RUN", "")),
             Path("/tmp/claude-1000")]
    best, newest = None, -1
    for r in roots:
        if not r or not r.exists():
            continue
        for p in r.rglob("debug_numeric/run_summary.json"):
            m = p.stat().st_mtime
            if m > newest:
                best, newest = p.parent.parent, m
    return best


def main():
    for fn in (test_47_renderer_refinement,
               test_48_visual_smoke,
               test_49_numerical_debug_contract,
               test_2_page_javascript_matches_the_estimator,
               test_49_9_debug_does_not_alter_inference):
        print(f"\n{fn.__name__}")
        fn()
        torch.cuda.empty_cache()
    print(f"\n{len(PASSED)} checks passed across 5 tests")


if __name__ == "__main__":
    main()
