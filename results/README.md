# Results: FreeArtGS vs RigEuler3DGS paper figures

Plotting and evaluation package implementing
[FreeArtGS_vs_RigEuler3DGS_Codex_Matplotlib_Spec.md](FreeArtGS_vs_RigEuler3DGS_Codex_Matplotlib_Spec.md).
It reads the result folders and writes nothing back into them; it doesn't touch model or training code.
Project status and the standard pipeline: [`../STATUS.md`](../STATUS.md).

Git ignores `*.png` and `input/`, so a fresh clone has the PDFs, `FIGURES.md` and
`metrics_recomputed.json`. Run `build_inputs.py` first; it reads the result folders on the external disk.

## Pipeline

```bash
PY=~/miniconda3/envs/rigeuler3dgs/bin/python     # needs numpy, pandas, matplotlib, scipy, Pillow

$PY results/build_inputs.py                       # raw result folders -> results/input/   (~10 s)
$PY results/generate_paper_figures.py             # results/input/ -> results/figures/
$PY results/test_generate_paper_figures.py        # 14 validation tests (pytest also works)
```

`generate_paper_figures.py` accepts the spec's interface (`--freeart-summary --freeart-report --freeart-psnr
--rigeuler-synthetic --outdir --formats pdf,png[,svg] --dpi 400`). Every argument defaults to a file in `results/input/`.

## Files

| file | role |
|---|---|
| `rigeuler_tables.py` | builds the RigEuler tables from the pipeline outputs (synthetic GT evaluation, per-frame Phase-3 trajectories, real-scene diagnostics) |
| `build_inputs.py` | assembles `input/`: the RigEuler tables, the three FreeArtGS files (renamed to the spec's names), the normalized join |
| `generate_paper_figures.py` | loading, validation, aggregates, figures, `FIGURES.md`, `metrics_recomputed.json` |
| `test_generate_paper_figures.py` | spec tests 1-13 plus the supplementary inputs |
| `freeart_rigeuler_comparison.csv` | the supplied normalized table; checked against the table rebuilt from the raw inputs on every run |
| `input/` | built inputs (see below) |
| `figures/` | fig01-fig09 (PDF + PNG), `FIGURES.md`, `metrics_recomputed.json` |
| `figures/supplementary/` | figS1-figS5 (see below) |

### Sources read by `build_inputs.py`

| input file | source |
|---|---|
| `FreeArtGS_summary.csv`, `FreeArtGS_report.json`, `FreeArtGS_per_observation_psnr.csv` | `~/FreeArtGS/outputs/synthetic_results_20261108/` (`summary.csv`, `report.json`, `per_observation_psnr.csv`) |
| `RigEuler3DGS_synthetic_results.csv` | `results_11_10_26/ours_v5_canonical_synthetic` (9 scenes), `ours_v5_phase1_gates/washer_103776_canonical`, `ours_v4_bootstrap_synthetic` |
| `RigEuler3DGS_synthetic_trajectories.csv` | the primary runs' `*_joint/<GT type>/trajectory.csv` + the dataset's `metadata/frame_values.csv`, aligned exactly as `tools/run_joint_phase3.evaluate_gt` does (q relative to the canonical frame, axis sign aligned). The build fails unless the recomputed MAE and max match the saved evaluation to 1e-9. |
| `RigEuler3DGS_real_results.csv` | `results_11_10_26/ours_v6_two_cams` and `ours_v5_two_cams` (8 real scenes × 3 support arms) |
| `RigEuler3DGS_columns.csv` | column dictionary |

The rebuilt `RigEuler3DGS_synthetic_results.csv` and `RigEuler3DGS_real_results.csv` are byte-identical to
`results_11_10_26/summary_2026_11_08/`. GT is read only after the runs; it never entered inference.

## Protocol (as in the spec)

- **Main comparison:** 10 synthetic scenes. RigEuler's primary run is `v5_canonical`; only `washer_103776` uses
  `v5_gates_canonical`, and the selection rejects that exception for any other scene. `v4_bootstrap` appears only in fig09.
- **Units:** degrees for revolute states, millimetres for prismatic states (RigEuler cm × 10), never averaged together
  (`mean_state_mae` raises `MixedUnitError`).
- **Alignment:** state errors are each method's native evaluation numbers. Nothing is re-aligned, rescaled or interpolated.
- **Rendering:** FreeArtGS only (fig08). The inputs contain no RigEuler PSNR/SSIM/LPIPS.

## Supplementary figures (beyond the spec's nine)

The spec allows a two-method per-frame figure once the RigEuler per-frame trajectories are located and validated, and
a separate real-scene supplement that is RigEuler-only.

- **figS1_rigeuler_trajectories:** RigEuler GT and estimate per scene, the counterpart of fig07.
- **figS2_per_frame_abs_error:** per-frame |error| of both methods, each in its native alignment. FreeArtGS is
  relative to frame 0, RigEuler to its canonical frame (59 for backward scenes).
- **figS3_type_criterion:** the geometric joint-type criterion (revolute axis radius / part scale, threshold 10):
  synthetic with GT types, and real v6 and v5 against the *visually* expected type. That is agreement with the
  visual expectation, not ground truth.
- **figS4_real_phase1_drift:** v5 against v6 Phase-1 static drift on the same Gaussians.
- **figS5_real_drift_attribution:** v6 static drift split into near the moving part / rarely seen / seen away from the part.

No numerical real-scene joint errors are reported, because the real scenes have no joint GT.
