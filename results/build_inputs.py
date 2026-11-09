#!/usr/bin/env python
"""Assemble results/input/ from the raw sources (read-only over every source folder).

  RigEuler3DGS_synthetic_results.csv       rebuilt from the RigEuler synthetic result folders
  RigEuler3DGS_synthetic_trajectories.csv  per-frame Phase-3 q vs GT q (primary runs), validated against the saved MAE
  RigEuler3DGS_real_results.csv            rebuilt from ours_v6_two_cams / ours_v5_two_cams (slow: depth projections)
  RigEuler3DGS_columns.csv                 column dictionary
  FreeArtGS_summary.csv, FreeArtGS_report.json, FreeArtGS_per_observation_psnr.csv
                                           copied from the FreeArtGS extraction folder (renamed to the spec's names)
  freeart_rigeuler_comparison.csv          normalized 10-row join in common units (generate_paper_figures.joined_table)

    python results/build_inputs.py                      # everything
    python results/build_inputs.py --skip-real          # synthetic + FreeArt only (seconds)

Then:  python results/generate_paper_figures.py   (see README.md)
"""
import argparse
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rigeuler_tables as T  # noqa: E402

FREEART_DEFAULT = Path.home() / "FreeArtGS/outputs/synthetic_results_20261108"
FREEART_FILES = {"summary.csv": "FreeArtGS_summary.csv", "report.json": "FreeArtGS_report.json",
                 "per_observation_psnr.csv": "FreeArtGS_per_observation_psnr.csv"}
TRAJ_DOC = [
    ("trajectories: q_est / q_gt", "Phase-3 joint coordinate of the GT-type hypothesis / GT, relative to canonical_frame, "
                                   "estimate multiplied by sign(axis . gt_axis) as in tools/run_joint_phase3.evaluate_gt; deg or mm"),
    ("trajectories: abs_error", "|q_est - q_gt|; its mean and max reproduce the saved coord_mae / coord_max_abs to 1e-9"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "input"))
    ap.add_argument("--results-root", default=str(T.R))
    ap.add_argument("--real-root", default=str(T.REAL))
    ap.add_argument("--synthetic-root", default=str(T.SYN))
    ap.add_argument("--freeart-dir", default=str(FREEART_DEFAULT))
    ap.add_argument("--skip-real", action="store_true", help="keep an existing RigEuler3DGS_real_results.csv")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T.configure(a.results_root, a.real_root, a.synthetic_root)

    for src, dst in FREEART_FILES.items():
        shutil.copyfile(Path(a.freeart_dir) / src, out / dst)
        print(f"copied {Path(a.freeart_dir) / src} -> {out / dst}")

    syn = T.synthetic()
    T.write_csv(out / "RigEuler3DGS_synthetic_results.csv", syn)
    print(f"RigEuler3DGS_synthetic_results.csv: {len(syn)} rows")
    traj = T.synthetic_trajectories()
    T.write_csv(out / "RigEuler3DGS_synthetic_trajectories.csv", traj)
    print(f"RigEuler3DGS_synthetic_trajectories.csv: {len(traj)} rows, MAE/max reproduce the saved evaluation")

    if not a.skip_real:
        jobs = [(s, arm) for s in T.SCENES for arm in ("C", "majority", "unanimous")]
        with Pool(a.workers) as p:
            real = p.map(T.real_row, jobs)
        T.write_csv(out / "RigEuler3DGS_real_results.csv", real)
        print(f"RigEuler3DGS_real_results.csv: {len(real)} rows")
    elif not (out / "RigEuler3DGS_real_results.csv").exists():
        print("WARNING: --skip-real and no existing RigEuler3DGS_real_results.csv")

    import csv
    with open(out / "RigEuler3DGS_columns.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["column", "meaning"]); w.writerows(T.DOC + TRAJ_DOC)

    import generate_paper_figures as G
    fa = G.load_freeart_summary(out / "FreeArtGS_summary.csv")
    rig = G.load_rigeuler_synthetic(out / "RigEuler3DGS_synthetic_results.csv")
    joined = G.joined_table(fa, rig)
    joined.to_csv(out / "freeart_rigeuler_comparison.csv", index=False)
    print(f"freeart_rigeuler_comparison.csv: {len(joined)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
