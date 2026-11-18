#!/usr/bin/env python
"""Phase-1 motion of static Gaussians on ONE fixed static set, across runs. Read-only, CPU,
evaluation only (no GT; nothing here feeds back into any stage).

Static set of a scene = Gaussians outside the Phase-2 support of every run named in
--static-from (default: every run; the same set for every run, so a run is never judged on a
different set because its support differs). Name the runs explicitly when one run's support
has collapsed onto most of the model -- the set would otherwise be empty (a warning is printed
when it is below 10% of the Gaussians).
Per run: displacement from the canonical position at 1/4, 1/2 and the last frame (median,
p90, p99, count > 50 mm), static per-frame step at the end (median mm/frame, the velocity
accumulation), support size and Jaccard overlap with every other run.

Gate (declared, printed, exit status 3 when it fails): the --candidate run is no worse than
the WORST of the --reference runs on the last-frame median, p90 and count > 50 mm of every
scene. It is a regression gate against the existing pipeline's own run-to-run spread, not a
quality claim.

  python tools/compare_static_displacement.py --run v6=ROOT --run d1=ROOT --run fix=ROOT \\
      --reference v6 d1 --candidate fix --scenes "robot_dog" --out DIR
<ROOT> is laid out as <ROOT>/<name>/unanimous/pipeline/<name>/bidirectional/<dir>/.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

DIRECTION = dict(matchbox="forward", wooden_box="forward", black_laptop="forward", robot_dog="forward",
                 orange_suitecase="forward", light_blue_box="forward", black_box="backward", red_fridge="forward")


def bidi(root: Path, name: str) -> Path:
    return root / name / "unanimous" / "pipeline" / name / "bidirectional"


def support(root: Path, name: str, d: str):
    f = bidi(root, name) / f"{d}_rigid" / "support_coherence" / "support.json"
    if f.is_file():
        return set(json.load(open(f))["members"])
    f = bidi(root, name) / d / "predicted_support.json"
    if f.is_file():
        return set(json.load(open(f))["dynamic_gaussians"])
    return None


def tracks(root: Path, name: str, d: str):
    c = bidi(root, name) / d / "phase1_centers"
    fs = sorted(glob.glob(str(c / "centers_frame_*.npy")))
    if not fs:
        return None, None
    return np.load(c / "canonical_xyz.npy"), np.stack([np.load(f) for f in fs])


def stats(X, x0, st):
    T = X.shape[0]
    d = np.linalg.norm(X - x0[None], axis=2) * 1e3
    step = np.linalg.norm(X[1:] - X[:-1], axis=2) * 1e3
    out = {}
    for lab, k in (("q1", T // 4), ("half", T // 2), ("end", T - 1)):
        v = d[k, st]
        out[lab] = {"median_mm": float(np.median(v)), "p90_mm": float(np.percentile(v, 90)),
                    "p99_mm": float(np.percentile(v, 99)), "over_50mm": int((v > 50).sum())}
    tail = step[-10:, :][:, st]
    out["step_end_median_mm_per_frame"] = float(np.median(tail)) if tail.size else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=ROOT")
    ap.add_argument("--reference", nargs="+", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--static-from", nargs="*", default=None, help="runs whose supports define the static set")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [(r.split("=", 1)[0], Path(r.split("=", 1)[1])) for r in a.run]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, gate = [], {}
    for sc in a.scenes.split():
        name, d = f"final_{sc}_episode_000000", DIRECTION[sc]
        have = {}
        for rn, root in runs:
            x0, X = tracks(root, name, d)
            if X is not None:
                have[rn] = (x0, X, support(root, name, d))
        if not have:
            print(f"{sc}: no run has phase1_centers"); continue
        x0 = next(iter(have.values()))[0]
        st = np.ones(x0.shape[0], bool)
        for rn, (y0, X, s) in list(have.items()):
            if y0.shape != x0.shape or not np.allclose(y0, x0):
                print(f"{sc}/{rn}: different canonical model; skipped"); del have[rn]; continue
            if s and (a.static_from is None or rn in a.static_from):
                st[list(s)] = False
        if st.sum() < 0.1 * st.size:
            print(f"WARNING {sc}: static set is {int(st.sum())} of {st.size} Gaussians -- name --static-from runs")
        res = {}
        for rn, (_, X, s) in have.items():
            r = {"scene": sc, "run": rn, "static_gaussians": int(st.sum()), "frames": int(X.shape[0]),
                 "support_size": None if s is None else len(s), **stats(X, x0, st)}
            r["jaccard"] = {o: (len(s & t) / max(len(s | t), 1) if s is not None and t is not None else None)
                            for o, (_, _, t) in have.items() if o != rn}
            res[rn] = r
            rows.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "jaccard"}), flush=True)
        refs = [res[r]["end"] for r in a.reference if r in res]
        cand = res.get(a.candidate, {}).get("end")
        if cand is None or not refs:
            gate[sc] = {"status": "NOT_EVALUATED", "reason": "candidate or references missing"}
        else:
            band = {k: max(r[k] for r in refs) for k in ("median_mm", "p90_mm", "over_50mm")}
            ok = {k: cand[k] <= band[k] for k in band}
            gate[sc] = {"status": "PASS" if all(ok.values()) else "FAIL", "candidate": cand, "worst_reference": band,
                        "per_statistic": ok}
    (out / "static_displacement.json").write_text(json.dumps({"rows": rows, "gate": gate}, indent=1))
    f = lambda x: "-" if x is None else f"{x:.3g}"                                  # noqa: E731
    L = ["# Phase-1 displacement of static Gaussians (one fixed static set per scene)", "",
         f"Static set = outside the Phase-2 supports of {a.static_from or 'every run'}. Displacement from the canonical "
         "position, mm.",
         "", "| scene | run | static | support | end median | end p90 | end p99 | end > 50 mm | half median | "
             "step at end (mm/frame) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['scene']} | {r['run']} | {r['static_gaussians']} | {r['support_size']} | {f(r['end']['median_mm'])} | "
                 f"{f(r['end']['p90_mm'])} | {f(r['end']['p99_mm'])} | {r['end']['over_50mm']} | "
                 f"{f(r['half']['median_mm'])} | {f(r['step_end_median_mm_per_frame'])} |")
    L += ["", f"## Gate: {a.candidate} no worse than the worst of {a.reference} (end median, p90, > 50 mm)", ""]
    for sc, g in gate.items():
        L.append(f"- {sc}: **{g['status']}** {json.dumps(g.get('per_statistic', g.get('reason')))}")
    (out / "static_displacement.md").write_text("\n".join(L) + "\n")
    print(f"-> {out / 'static_displacement.md'}")
    print("gate:", {k: v["status"] for k, v in gate.items()})
    return 0 if all(v["status"] == "PASS" for v in gate.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
