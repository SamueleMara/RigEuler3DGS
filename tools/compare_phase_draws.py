#!/usr/bin/env python
"""Run-to-run comparison of independent Phase 1-3 runs ("draws") of the real scenes. Read-only, CPU only.

Phase 1 is not reproducible on the real scenes (GPU atomics, amplified over the sequence), so one run says little.
For every scene and every draw this reports, and for every PAIR of draws how far they agree:

  Phase 1   track divergence between draws at fixed processing steps (median / p99 over Gaussians, mm);
            drift of the Gaussians outside the Phase-2 support (median over the sequence, mm)
  coherence status and size of the predicted support; Jaccard overlap of supports between draws;
            --replay: the extractor re-run on the saved tracks with the SAME code for every draw, giving the
            best gate score min(eta_cv relative, eta_cv world) over the qualified candidates, the score of the
            default (v6) selection, and whether the cautious selection (0.8 / 0.25 / 3) finds a body
  Phase 2/3 joint type (R/L rule), R/L, revolute sweep, prismatic travel; axis / direction angle between
            draws; error to the manual GT axes where they exist (old camera frame: only valid for draws run on
            the pre-registration poses -- orange has no GT)

  python tools/compare_phase_draws.py --draw v6=$RES/ours_v6_two_cams \
      --override v6:orange_suitecase=$RES/ours_v6_three_cams --draw d1=$RES/ours_v8_overnight_real/draw1 \
      --replay --out $RES/ours_v8_overnight_real/draw_comparison
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "results"))

SCENES = {"black_box": "backward", "black_laptop": "forward", "light_blue_box": "forward", "matchbox": "forward",
          "orange_suitecase": "forward", "red_fridge": "forward", "robot_dog": "forward", "wooden_box": "forward"}
EXPECTED = dict(black_box="revolute", black_laptop="revolute", light_blue_box="prismatic", matchbox="prismatic",
                orange_suitecase="prismatic", red_fridge="revolute", robot_dog="prismatic", wooden_box="revolute")
GT_FILE = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/"
               "gt_joint_axes_3_real_scenes/gt_joint_axes_3_real_scenes.txt")
GT_NAME = {"red_fridge": "red_fridge", "wooden_box": "wood_box", "light_blue_box": "blue_box"}
STEPS = (1, 2, 5, 10, "25%", "50%", "last")


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def line_angle(a, b):
    return float(np.degrees(np.arccos(min(1.0, abs(float(unit(a) @ unit(b)))))))


def line_distance(p1, a1, p2, a2):
    a1, a2, d = unit(a1), unit(a2), np.asarray(p2, float) - np.asarray(p1, float)
    n = np.cross(a1, a2)
    if np.linalg.norm(n) < 1e-9:
        return float(np.linalg.norm(d - (d @ a1) * a1))
    return float(abs(d @ unit(n)))


def parse_gt(p):
    out, cur = {}, None
    if not p.exists():
        return out
    for ln in open(p):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("["):
            cur = ln[1:-1]; out[cur] = {}; continue
        k, v = [x.strip() for x in ln.split("=", 1)]
        try:
            v = np.array([float(x) for x in v.split(",")])
        except ValueError:
            pass
        out[cur][k] = v
    return out


def scene_dir(root, sc):
    n = f"final_{sc}_episode_000000"
    return Path(root) / n / "unanimous" / "pipeline" / n / "bidirectional"


def load_draw(root, sc, replay):
    d = SCENES[sc]
    B = scene_dir(root, sc)
    r = {"root": str(root)}
    files = sorted(glob.glob(str(B / d / "phase1_centers" / "centers_frame_*.npy")))
    if not files:
        r["missing"] = "no Phase-1 tracks"
        return r, None
    if d == "backward":
        files = files[::-1]               # processing order: the anchor first
    tracks = [np.load(f) for f in files]
    r["frames"] = len(tracks)
    ps = json.load(open(B / d / "predicted_support.json"))
    r["phase1_status"] = ps["status"]
    r["phase1_support"] = sorted(ps["dynamic_gaussians"])
    sup = B / f"{d}_rigid" / "support_coherence" / "support.json"
    r["phase2_support"] = sorted(json.load(open(sup))["members"]) if sup.exists() else None
    if r["phase2_support"] is not None:
        m = np.zeros(len(tracks[0]), bool); m[r["phase2_support"]] = True
        disp = np.linalg.norm(tracks[-1] - tracks[0], axis=1)
        r["drift_outside_mm"] = 1e3 * float(np.median(disp[~m]))
        r["motion_support_mm"] = 1e3 * float(np.median(disp[m]))
    jt = B / f"{d}_joint"
    if (jt / "revolute/geometry.json").exists():
        import rigeuler_tables as t
        g = t.joint_geometry(B / f"{d}_rigid")
        R = json.load(open(jt / "revolute/geometry.json")); P = json.load(open(jt / "prismatic/geometry.json"))
        r.update(type=g["type_decision"], R_over_L=g["rev_axis_radius_over_part_scale"], sweep_deg=g["rev_sweep_deg"],
                 travel_cm=100 * g["pri_travel_m"], rev_axis=list(unit(R["axis"])), rev_point=list(R["axis_point"]),
                 pri_dir=list(unit(P["direction"])))
    if replay:
        r["replay"] = replay_coherence(B / d / "phase1_centers", tracks if d == "forward" else tracks[::-1])
    return r, tracks


def replay_coherence(centers, tracks_file_order):
    """The v2 extractor on the saved tracks, same graph as tools/run_motion_coherence_support.py."""
    import torch
    from estimator.coherence_extraction import extract_moving_component
    from estimator.motion_coherence_support import run_motion_coherence, supernode_trajectories
    from estimator.supernodes import build_supernodes
    pos = [torch.from_numpy(x).double() for x in tracks_file_order]
    mu0 = torch.from_numpy(np.load(centers / "canonical_xyz.npy")).double()
    sid = build_supernodes(mu0.float(), num_supernodes=min(192, mu0.shape[0]), seed=0).supernode_id.long().cpu()
    res = run_motion_coherence(pos, sid, knn=12, verbose=False)
    traj = supernode_trajectories(pos, sid)["trajectories"][:, res["node_ids"], :]
    args = (traj, res["edge_i"], res["edge_j"], res["C"], res["num_supernodes"])
    ex = extract_moving_component(*args, folds=5, verbose=False)
    cand = {c["candidate_id"]: c for c in ex["candidates"]}
    gate = lambda c: min(c["rigidity"]["eta_cv"], c["rigidity_world"]["eta_cv"])
    q = [cand[i] for i in ex["selection"].get("qualified_ids", []) if i in cand]
    out = {"default_status": ex["status"], "qualified": len(q),
           "best_gate_score": max((gate(c) for c in q), default=float("nan"))}
    sel = cand.get(ex["selection"].get("selected"))
    if sel:
        out.update(selected_eta_rel=sel["rigidity"]["eta_cv"], selected_eta_world=sel["rigidity_world"]["eta_cv"],
                   selected_nodes=int(ex["moving_nodes"].numel()))
    cx = extract_moving_component(*args, folds=5, verbose=False, min_eta_cv=0.8, recruit_ratio=0.25, trim_k=3.0)
    out["cautious_status"] = cx["status"]
    out["cautious_nodes"] = int(cx["moving_nodes"].numel()) if cx["status"] == "RESOLVED" else 0
    return out


def jaccard(a, b):
    if a is None or b is None:
        return float("nan")
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def divergence(ta, tb):
    T = min(len(ta), len(tb)); out = {}
    for s in STEPS:
        k = {"25%": T // 4, "50%": T // 2, "last": T - 1}.get(s, s)
        if not isinstance(k, int) or k >= T:
            continue
        d = np.linalg.norm(ta[k] - tb[k], axis=1)
        out[str(s)] = (1e3 * float(np.median(d)), 1e3 * float(np.percentile(d, 99)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", action="append", required=True, help="LABEL=ROOT (an ours_* tree of Phase 1-3)")
    ap.add_argument("--override", action="append", default=[], help="LABEL:SCENE=ROOT for one scene of a draw")
    ap.add_argument("--scenes", default=" ".join(SCENES))
    ap.add_argument("--replay", action="store_true", help="re-run the coherence extractor on every draw (CPU)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    draws = dict(x.split("=", 1) for x in a.draw)
    over = {}
    for x in a.override:
        k, root = x.split("=", 1); lab, sc = k.split(":", 1); over[(lab, sc)] = root
    gt = parse_gt(GT_FILE)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report, md = {}, ["# Run-to-run comparison of Phase 1-3 draws", "",
                      "draws: " + ", ".join(f"`{k}` = {v}" for k, v in draws.items()), ""]
    for sc in a.scenes.split():
        rows, tracks = {}, {}
        for lab, root in draws.items():
            root = over.get((lab, sc), root)
            try:
                rows[lab], tracks[lab] = load_draw(root, sc, a.replay)
            except Exception as e:  # a missing / partial draw is reported, not fatal
                rows[lab], tracks[lab] = {"missing": f"{type(e).__name__}: {e}"}, None
            g = gt.get(GT_NAME.get(sc, ""), {})
            r = rows[lab]
            if g and "type" in r:
                ag = unit(g["axis_direction_unit"])
                if str(g.get("joint_type")) == "revolute":
                    r["gt_rev_axis_deg"] = line_angle(r["rev_axis"], ag)
                    r["gt_rev_line_mm"] = 1e3 * line_distance(r["rev_point"], r["rev_axis"], g["axis_point"], ag)
                else:
                    r["gt_pri_dir_deg"] = line_angle(r["pri_dir"], ag)
            print(f"{sc} {lab}: {r.get('missing') or r.get('type', r.get('phase1_status'))}", flush=True)
        pairs = {}
        for x, y in combinations(rows, 2):
            if tracks[x] is None or tracks[y] is None:
                continue
            p = {"tracks_mm_median_p99": divergence(tracks[x], tracks[y]),
                 "jaccard_phase1_support": jaccard(rows[x].get("phase1_support"), rows[y].get("phase1_support")),
                 "jaccard_phase2_support": jaccard(rows[x].get("phase2_support"), rows[y].get("phase2_support"))}
            if "rev_axis" in rows[x] and "rev_axis" in rows[y]:
                p["rev_axis_angle_deg"] = line_angle(rows[x]["rev_axis"], rows[y]["rev_axis"])
                p["pri_dir_angle_deg"] = line_angle(rows[x]["pri_dir"], rows[y]["pri_dir"])
            pairs[f"{x}~{y}"] = p
        for r in rows.values():
            for k in ("phase1_support", "phase2_support"):
                if isinstance(r.get(k), list):
                    r[k] = len(r[k])
        report[sc] = {"expected": EXPECTED[sc], "draws": rows, "pairs": pairs}
        # markdown
        md += [f"## {sc} (expected {EXPECTED[sc]})", "",
               "| draw | P1 support | gate score | v6-selected eta rel/world | cautious | P2 support | type | R/L | "
               "sweep deg | travel cm | drift outside mm | GT err |", "|" + "---|" * 12]
        for lab, r in rows.items():
            if "missing" in r:
                md.append(f"| {lab} | {r['missing']} |" + " |" * 10); continue
            rp = r.get("replay", {})
            gte = (f"{r['gt_rev_axis_deg']:.1f}deg/{r['gt_rev_line_mm']:.0f}mm" if "gt_rev_axis_deg" in r else
                   f"{r['gt_pri_dir_deg']:.1f}deg" if "gt_pri_dir_deg" in r else "")
            f = lambda k, fmt: (fmt % r[k]) if isinstance(r.get(k), (int, float)) else ""
            md.append(f"| {lab} | {r['phase1_status'][:8]} {r['phase1_support']} | "
                      f"{rp.get('best_gate_score', float('nan')):.3f} | "
                      f"{rp.get('selected_eta_rel', float('nan')):.2f}/{rp.get('selected_eta_world', float('nan')):.2f} | "
                      f"{rp.get('cautious_status', '')[:8]} {rp.get('cautious_nodes', '')} | {r.get('phase2_support', '')} | "
                      f"{r.get('type', '-')} | {f('R_over_L', '%.2f')} | {f('sweep_deg', '%.1f')} | {f('travel_cm', '%.1f')} | "
                      f"{f('drift_outside_mm', '%.0f')} | {gte} |")
        md += ["", "| pair | tracks median/p99 mm at step " + ", ".join(map(str, STEPS)) +
               " | Jaccard P1 / P2 support | axis angle rev / pri deg |", "|---|---|---|---|"]
        for k, p in pairs.items():
            tr = "; ".join(f"{s}: {m:.2g}/{q:.3g}" for s, (m, q) in p["tracks_mm_median_p99"].items())
            ax = (f"{p['rev_axis_angle_deg']:.1f} / {p['pri_dir_angle_deg']:.1f}" if "rev_axis_angle_deg" in p else "")
            md.append(f"| {k} | {tr} | {p['jaccard_phase1_support']:.2f} / {p['jaccard_phase2_support']:.2f} | {ax} |")
        md.append("")
        json.dump(report, open(out / "draw_comparison.json", "w"), indent=1, default=float)
        (out / "draw_comparison.md").write_text("\n".join(md))
    print(f"-> {out / 'draw_comparison.md'}")


if __name__ == "__main__":
    main()
