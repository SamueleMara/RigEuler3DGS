#!/usr/bin/env python
"""Section 8 + 9.2: decide P/R against a floor MEASURED from camera folds.

CODEX_DIRECT_GAUSSIAN_JOINT_RGBD_RELIABILITY.md sections 8, 9.2, 12.

The decision needs to know how big a P-versus-R gain has to be before it means
anything. Section 8 is explicit that this cannot be assumed: the previous
floor divided a rotation uncertainty by sqrt(N) over supernodes, which treats
every supernode in a part as an independent look at the same rotation. They
are not independent -- they are fitted through the same images, the same
calibration and the same canonical model -- so that floor reported a
resolution the reconstruction never had.

This tool measures the floor instead. Phase 2 is refit with a different block
of cameras held out each time; the gain is recomputed on each refit; and the
spread of those gains is the floor. Produce the refits with

    for f in 0 1 2 3; do
      python tools/run_all_datasets.py --only <scene> \\
        --camera-folds 4 --camera-fold $f --out-root output/folds_f$f
    done

then point this tool at them.

Nothing here reads ground truth. The GT column, when a run carries one, is
printed after the decision and takes no part in it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from joint_reduction.continuous_joint_geometry import \
    fit_continuous_joint  # noqa: E402
from joint_reduction.gaussian_transport_metric import (  # noqa: E402
    canonical_moments, gain_attribution)
from joint_reduction.geometric_floor import (  # noqa: E402
    gain_floor_from_folds, transform_discrepancy)
from joint_reduction.geometric_joint_fit import (  # noqa: E402
    PRISMATIC, REVOLUTE, UNRESOLVED_GEOMETRY, decide_joint_type_continuous)


def load_part(root: Path, part: int = 0, canonical: Path | None = None):
    """Everything one part's joint fit needs, from a finished run.

    The canonical Gaussians are a property of the trained model, so every fold
    of the same scene has the identical set in the identical order; only the
    labels and the recovered transforms differ between folds. `canonical`
    therefore lets one exported copy serve all of them, which keeps a fold
    refit to the cheap numerics instead of re-exporting every Gaussian.
    """
    tf = root / "debug_numeric" / "rigid_transforms.pt"
    cf = root / "canonical_gaussians.pt"
    if not cf.is_file() and canonical is not None:
        cf = canonical
    lf = root / "rigid_part_labels.pt"
    missing = [str(p) for p in (tf, cf, lf) if not p.is_file()]
    if missing:
        # Section 11: "If earlier outputs lack essential fields, fail clearly
        # and provide a one-time export command; do not silently reconstruct
        # invented confidence data."
        raise FileNotFoundError(
            f"{root} is missing {missing}. Re-run that fold with debug "
            f"numerics enabled:\n"
            f"  python tools/run_all_datasets.py --only <scene> "
            f"--camera-folds <n> --camera-fold <k> --out-root {root.parent}")
    T4 = torch.load(tf, map_location="cpu", weights_only=False).double()
    can = torch.load(cf, map_location="cpu", weights_only=False)
    lab = torch.load(lf, map_location="cpu", weights_only=False)
    lab = (lab["labels"] if isinstance(lab, dict) else lab).reshape(-1)
    if can["mu"].shape[0] != lab.numel():
        raise ValueError(
            f"{root}: {lab.numel()} labels against {can['mu'].shape[0]} "
            f"canonical Gaussians from {cf}. The folds must come from the "
            f"same trained model and the same --full-gaussians setting; "
            f"re-export the canonical set for this run rather than sharing "
            f"one that does not match.")
    sel = lab == (part + 1)
    if int(sel.sum()) < 10:
        sel = lab != 0
    mu = can["mu"].double()[sel]
    w = can["opacity"].reshape(-1)[sel].double().clamp_min(1e-6)
    M4 = T4[min(part, T4.shape[0] - 1)]
    return {"T4": M4, "R": M4[:, :3, :3].contiguous(),
            "p": M4[:, :3, 3].contiguous(),
            "moments": canonical_moments(mu, w), "n_gaussians": int(sel.sum())}


def fit_pair(d, max_iters: int):
    M = d["moments"]
    fP = fit_continuous_joint(M["c"], d["R"], d["p"], M["L"],
                              max_iters=max_iters, fix_eta_zero=True,
                              moments=M)
    fR = fit_continuous_joint(M["c"], d["R"], d["p"], M["L"],
                              max_iters=max_iters, fix_eta_zero=False,
                              moments=M)
    return fP, fR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="scene directory name")
    ap.add_argument("--folds", nargs="+", required=True,
                    help="out-roots of the camera-fold refits")
    ap.add_argument("--full", default=None,
                    help="out-root of the all-camera run (reported, and the "
                         "gain it is compared against)")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--max-iters", type=int, default=600)
    ap.add_argument("--epsilon-opt", type=float, default=0.0,
                    help="solver error; measure it with "
                         "joint_reduction.geometric_floor.optimization_error")
    ap.add_argument("--allow-incomplete-folds", action="store_true",
                    help="use folds that produced fewer artifacts than their "
                         "siblings; their spread measures pipeline failure, "
                         "not camera sampling")
    ap.add_argument("--canonical-from", default=None,
                    help="run directory holding canonical_gaussians.pt to "
                         "share across folds (default: the --full run)")
    ap.add_argument("--out", default=None,
                    help="directory for the section 12 artifacts "
                         "(default: <full>/debug_numeric/joint_direct)")
    args = ap.parse_args()

    shared = None
    for cand in ([args.canonical_from] if args.canonical_from else []) + \
            ([args.full] if args.full else []) + list(args.folds):
        p = Path(cand) / args.scene / "canonical_gaussians.pt"
        if p.is_file():
            shared = p
            break
    if shared is not None:
        print(f"canonical Gaussians: {shared}")

    # --- reject folds whose runs did not get as far as their siblings -----
    # A fold that bailed out of a stage still leaves transforms behind, and
    # those transforms will happily enter the spread and inflate the floor.
    # Comparing each fold's artifacts against what the majority produced
    # catches that without hard-coding a list of required files: a fold that
    # is missing what every other fold wrote did not run the same pipeline,
    # whatever its run.log says.
    present = {}
    for fr in args.folds:
        root = Path(fr) / args.scene
        present[fr] = {p.name for p in root.glob("*.json")} | \
                      {p.name for p in root.glob("*.pt")}
    if len(present) > 1:
        from collections import Counter
        counts = Counter(n for names in present.values() for n in names)
        quorum = max(2, (len(present) + 1) // 2 + 1)
        common = {n for n, c in counts.items() if c >= quorum}
        bad = {fr: sorted(common - names) for fr, names in present.items()
               if common - names}
        for fr, miss in bad.items():
            print(f"  WARNING {fr}: missing {len(miss)} artifact(s) that "
                  f"{quorum}+ other folds wrote, e.g. {miss[:4]}")
        if bad and not args.allow_incomplete_folds:
            raise SystemExit(
                f"\nsection 8: {len(bad)} fold(s) did not complete the same "
                f"pipeline as the others, so their transforms would measure a "
                f"pipeline failure rather than camera-sampling spread. Re-run "
                f"them, drop them from --folds, or pass "
                f"--allow-incomplete-folds to override deliberately.")

    folds, gains, rows = [], [], []
    for fr in args.folds:
        root = Path(fr) / args.scene
        d = load_part(root, args.part, canonical=shared)
        fP, fR = fit_pair(d, args.max_iters)
        att = gain_attribution(d["R"], d["p"], fP, fR, d["moments"])
        gain = fP["energy"] - fR["energy"]
        folds.append((root, d, fP, fR, att))
        gains.append(gain)
        rows.append({
            "fold_root": str(root), "E_P": fP["energy"], "E_R": fR["energy"],
            "signed_gain": gain, "kappa": fR["kappa"],
            "delta_E_center": att["delta_E_center"],
            "delta_E_rotation": att["delta_E_rotation"],
            "gain_is_rotation_only": att["gain_is_rotation_only"],
            "status_P": fP["status"], "status_R": fR["status"],
            "projected_grad_rel_P": fP["projected_grad_relative"],
            "projected_grad_rel_R": fR["projected_grad_relative"],
            "n_gaussians": d["n_gaussians"], "n_frames": int(d["R"].shape[0])})
        print(f"  {root.parent.name:<22} E_P {fP['energy']:.6e}  "
              f"E_R {fR['energy']:.6e}  gain {gain:+.4e}  "
              f"kappa {fR['kappa']:.4f}  [{fP['status']}/{fR['status']}]")

    floor = gain_floor_from_folds(gains, epsilon_opt=args.epsilon_opt)
    print(f"\nsection 8 floor from {len(gains)} camera folds:")
    for k in ("mean_gain", "sd_gain", "min_gain", "max_gain",
              "folds_favouring_revolute", "folds_favouring_prismatic",
              "sign_is_unanimous", "n_folds", "rho_lag1", "n_effective",
              "epsilon_geom", "floor"):
        print(f"   {k:<26} {floor.get(k)}")

    # --- pairwise transform repeatability (section 8's D_bc) --------------
    rep = []
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            if folds[i][1]["R"].shape != folds[j][1]["R"].shape:
                continue
            dd = transform_discrepancy(folds[i][1]["T4"], folds[j][1]["T4"],
                                       folds[i][1]["moments"])
            rep.append({"fold_a": folds[i][0].parent.name,
                        "fold_b": folds[j][0].parent.name,
                        "D_bc_rms": dd["D_bc_rms"], "D_bc_max": dd["D_bc_max"],
                        "translation_rms": dd["translation_rms"],
                        "rotation_rms": dd["rotation_rms"]})
    if rep:
        print(f"\ntransform repeatability across folds (D_bc): "
              f"rms {min(r['D_bc_rms'] for r in rep):.3e} .. "
              f"{max(r['D_bc_rms'] for r in rep):.3e}")

    # --- the decision, on the all-camera run, against the measured floor --
    verdict = None
    full_root = Path(args.full) / args.scene if args.full else folds[0][0]
    d = load_part(full_root, args.part, canonical=shared)
    eps_geom = floor.get("epsilon_geom")
    if eps_geom is None:
        print("\nsection 8: no floor could be measured, so no P/R claim is "
              "made. Provide at least two camera folds.")
        dec = {"selected": None, "status": UNRESOLVED_GEOMETRY,
               "reason": floor.get("reason", floor["status"])}
    else:
        dec = decide_joint_type_continuous(
            d["R"], d["p"], d["moments"]["c"], d["moments"]["L"],
            epsilon_geom=eps_geom, epsilon_opt=args.epsilon_opt,
            max_iters=args.max_iters, moments=d["moments"])
        fP, fR = fit_pair(d, args.max_iters)
        att = gain_attribution(d["R"], d["p"], fP, fR, d["moments"])
        dec["gain_attribution"] = att
        verdict = dec["selected"] or dec["status"]
        print(f"\n--- decision on {full_root} ---")
        print(f"  E_P {dec['E_P']:.6e}   E_R {dec['E_R']:.6e}")
        print(f"  gain {dec['curvature_gain']:.4e} vs measured floor "
              f"{dec['floor']:.4e}  ->  {verdict}")
        print(f"  dE_center {att['delta_E_center']:+.4e}   "
              f"dE_rotation {att['delta_E_rotation']:+.4e}")
        if att["gain_is_rotation_only"]:
            print("  NOTE (section 9.2): the straight-line model fits the "
                  "centre path at least as well as the curved one, so this "
                  "gain is orientation wobble, not measured curvature.")
        o = dec["curvature_observability"]
        print(f"  smallest curvature this data could resolve: "
              f"{o['kappa_min_detectable']:.4g}  (fitted {dec['kappa']:.4g})")

    # --- section 12 artifacts ---------------------------------------------
    out = Path(args.out) if args.out else (full_root / "debug_numeric"
                                           / "joint_direct")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "curvature_by_fold.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    if rep:
        with open(out / "phase2_transform_repeatability.csv", "w",
                  newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rep[0].keys()))
            wr.writeheader()
            wr.writerows(rep)
    json.dump(floor, open(out / "geometric_floor_derivation.json", "w"),
              indent=1)
    json.dump({"scene": args.scene, "part": args.part,
               "verdict": verdict, "decision": dec,
               "folds": [str(f[0]) for f in folds],
               "full_run": str(full_root),
               "ground_truth_used": False,
               "note": "sections 8 and 9.2. The floor is the spread of the "
                       "signed gain across independent camera blocks, not a "
                       "1/sqrt(N) over supernodes. No ground truth entered "
                       "this decision."},
              open(out / "joint_fit_comparison.json", "w"), indent=1,
              default=str)
    print(f"\nsection 12 artifacts -> {out}")

    gt_path = full_root / "dataset_joint_ground_truth.json"
    if gt_path.is_file():
        try:
            gt = json.load(open(gt_path)).get("joint_type_raw")
            gt = {"continuous": "revolute"}.get(gt, gt)
            print(f"\n[post-inference only] ground truth {gt}, decided "
                  f"{verdict}, match {gt == verdict}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
