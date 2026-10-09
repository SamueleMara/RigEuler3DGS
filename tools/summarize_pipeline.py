"""One table for the whole pipeline, across any set of scenes.

Reads only finished artifacts:

    output/motion_coherence_<scene>/numerical/gt_metrics.json   static/dynamic
    output/rigid_se3_<scene>/summary.json                       pose + joint

Scenes that have not run yet are listed as such rather than silently omitted,
because a table that quietly drops a failure reads like a clean sweep.

    python tools/summarize_pipeline.py scissor_11100 laptop_10211
    python tools/summarize_pipeline.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _coh_dir(root: Path, scene: str) -> Path:
    """The coherence output, in either layout.

    PER_SCENE gathers a run under <root>/<scene>/{coherence,rigid}; the flat
    layout the earlier runs used puts them at <root>/motion_coherence_<scene>
    and <root>/rigid_se3_<scene>. Both are checked so one summary can read a
    mixture, which is what a machine with months of runs on it actually has.
    """
    a = root / scene / "coherence"
    return a if a.is_dir() else root / f"motion_coherence_{scene}"


def _rig_dir(root: Path, scene: str) -> Path:
    a = root / scene / "rigid"
    return a if a.is_dir() else root / f"rigid_se3_{scene}"


def _scenes_on_disk(root: Path) -> list:
    out = set()
    for p in root.glob("rigid_se3_*"):
        if (p / "summary.json").exists():
            out.add(p.name[len("rigid_se3_"):])
    for p in root.iterdir() if root.is_dir() else []:
        if (p / "rigid" / "summary.json").exists():
            out.add(p.name)
    return sorted(out)


def _f(x, w=8, p=3, nan="--"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return f"{nan:>{w}}"
    if v != v:
        return f"{nan:>{w}}"
    return f"{v:>{w}.{p}f}"


def _e(x, w=10, p=3, nan="--"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return f"{nan:>{w}}"
    if v != v:
        return f"{nan:>{w}}"
    return f"{v:>{w}.{p}e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--outroot", default="output")
    ap.add_argument("--all", action="store_true",
                    help="every scene with a rigid_se3_* run on disk")
    args = ap.parse_args()
    root = Path(args.outroot)
    scenes = list(args.scenes)
    if args.all or not scenes:
        scenes = _scenes_on_disk(root)
    if not scenes:
        print("no finished runs found")
        return 0

    # ---- stage 2: static/dynamic labelling -----------------------------
    hdr = (f"{'scene':<18}{'selector':<11}{'prec':>7}{'recall':>8}{'F1':>7}"
           f"{'IoU':>7}{'pred dyn':>10}{'GT dyn':>9}")
    print("STAGE 2 - static/dynamic labelling, against the optimized 3DGS GT")
    print(hdr)
    print("-" * len(hdr))
    for s in scenes:
        p = _coh_dir(root, s) / "numerical" / "gt_metrics.json"
        if not p.exists():
            print(f"{s:<18}-- not run --")
            continue
        d = json.load(p.open())
        for name in ("current", "coherence"):
            m = d.get(name)
            if not m:
                continue
            print(f"{s:<18}{name:<11}{_f(m['dynamic_precision'],7)}"
                  f"{_f(m['dynamic_recall'],8)}{_f(m['dynamic_f1'],7)}"
                  f"{_f(m['dynamic_iou'],7)}"
                  f"{_f(100*m['predicted_dynamic_fraction'],9,1)}%"
                  f"{_f(100*m['gt_dynamic_fraction'],8,1)}%")

    # ---- stage 3: rigid pose + joint -----------------------------------
    hdr2 = (f"{'scene':<18}{'GT':<11}{'moving':>8}{'gamma':>8}{'E_P':>11}"
            f"{'E_R':>11}{'P ang':>8}{'R ang':>8}{'R line':>9}{'rad/L':>9}")
    print(f"\nSTAGE 3 - rigid SE(3) pose and the reduced joint")
    print(hdr2)
    print("-" * len(hdr2))
    picked = {}
    for s in scenes:
        p = _rig_dir(root, s) / "summary.json"
        if not p.exists():
            print(f"{s:<18}-- not run --")
            continue
        d = json.load(p.open())
        g = d.get("gt_evaluation", {})
        P, R = d["prismatic"], d["revolute"]
        ep = P["pose_residual"]["weighted_energy"]
        er = R["pose_residual"]["weighted_energy"]
        gt = g.get("gt_joint_type") if g.get("available") else None
        # A scene whose GT labels nothing as moving scores as if nothing
        # moves: precision 0, F1 undefined, and an axis error measured against
        # a joint its own labels deny. washer_103776 reads that way, and the
        # cause is upstream of the labels: summed over all eight views, only
        # 4454 depth pixels change by more than a centimetre across the whole
        # episode, against 89084 on oven. Five of its eight cameras see no
        # change at all. So COLMAP triangulated no point on the moving link,
        # `part` came out 0 everywhere, and the file is a faithful record of a
        # reconstruction with nothing on that link -- not a corrupt file.
        # The row is MARKED rather than read as a result.
        mcf = _coh_dir(root, s) / "numerical" / "gt_metrics.json"
        if mcf.exists():
            try:
                if float(json.load(mcf.open())["coherence"]
                         ["gt_dynamic_fraction"]) <= 0.0:
                    gt = "!GT-EMPTY"
            except Exception:
                pass
        picked[s] = (gt, R.get("axis_radius_over_part_scale"), ep, er,
                     g.get("prismatic", {}).get("axis_angle_deg"),
                     g.get("revolute", {}).get("axis_angle_deg"))
        print(f"{s:<18}{str(gt):<11}{d['support']['num_members']:>8}"
              f"{_f(d['one_dof']['gamma_1dof'],8,4)}{_e(ep,11)}{_e(er,11)}"
              f"{_f(g.get('prismatic',{}).get('axis_angle_deg'),8,2)}"
              f"{_f(g.get('revolute',{}).get('axis_angle_deg'),8,2)}"
              f"{_f((g.get('revolute',{}).get('axis_line_error') or {}).get('distance'),9,4)}"
              f"{_f(R.get('axis_radius_over_part_scale'),9,1)}")

    # ---- the numbers that say whether to trust the run -----------------
    hdr3 = (f"{'scene':<18}{'rigidity drift':>16}{'unexpl.':>9}"
            f"{'gamma':>8}{'Kabsch d_rot':>14}{'Kabsch d_p':>12}{'runtime':>10}")
    print(f"\nHEALTH - read these before the table above")
    print(hdr3)
    print("-" * len(hdr3))
    for s in scenes:
        p = _rig_dir(root, s) / "summary.json"
        if not p.exists():
            continue
        d = json.load(p.open())
        gate, k = d["exact_rigidity_gate"], d["phase1_kabsch_vs_rgbd"]
        sr = d.get("support_rigidity") or {}
        rv = sr.get("unexplained_fraction_final")
        body = f"{100*rv:>8.0f}%" if rv is not None and rv == rv else "       --"
        print(f"{s:<18}{_e(gate['max_pairwise_distance_drift'],16)}{body}"
              f"{_f(d['one_dof']['gamma_1dof'],8,3)}"
              f"{_f(k['max_d_rotation_rad'],14,4)}"
              f"{_f(k['max_d_translation'],12,4)}"
              f"{_f(d.get('runtime_s'),9,0)}s")

    # ---- what the evidence says, without pretending to classify --------
    print("\nREADING IT")
    print("  rigidity drift must be at machine precision; it is a property of")
    print("  the shared SE(3) parameterisation, so anything else is a defect.")
    print("  A large Kabsch d_rot means the RGB-D stage overrode Phase 1 --")
    print("  inspect the RGB-D objective before trusting the axis.")
    print("  rad/L is the far-axis tell: order 1 means the revolute axis lies")
    print("  on the part; a large value means R is approximating a translation.")
    print("  unexpl. is the share of the support's motion that one rigid body")
    print("  does not explain -- a quality number for the support and its")
    print("  Phase-1 trajectories. It is NOT a body count: the scissor, the one")
    print("  scene that really is two bodies, sits mid-pack at 24%.")
    print("  gamma is what finds that: 0.59 there, 0.92+ everywhere else.")
    print("  !GT-EMPTY means that scene's GT labels NOTHING as moving, so every")
    print("  GT number for it is vacuous. On washer that traces to the episode")
    print("  itself: 4454 depth pixels change across all 8 views, against 89084")
    print("  on oven, so the moving link was never triangulated. Re-render the")
    print("  episode with views that observe it; the files are not corrupt.")
    print("  No joint-type classifier is produced here. E_R < E_P alone is NOT")
    print("  sufficient -- on a near-prismatic part R can win on energy while")
    print("  placing its axis tens of part-radii away.")
    agree = [s for s, v in picked.items()
             if v[0] and v[4] is not None and v[5] is not None
             and ((v[0] == "prismatic" and v[4] < 10.0)
                  or (v[0] == "revolute" and v[5] < 10.0))]
    scored = [s for s, v in picked.items() if v[0] and v[4] is not None]
    if scored:
        print(f"\n  the model matching the GT type recovers its axis to within "
              f"10 deg on {len(agree)}/{len(scored)} scored scenes")
        for s in scored:
            gt, rl, ep, er, pa, ra = picked[s]
            use = pa if gt == "prismatic" else ra
            print(f"    {s:<18}{gt:<11}axis error {use:6.2f} deg"
                  f"   rad/L {rl:8.1f}" if rl is not None else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
