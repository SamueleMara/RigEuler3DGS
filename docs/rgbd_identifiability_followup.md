# Automatic RGB-D root-cause investigation

This is an isolated analysis of the existing 118 real scenes (110 final episodes and eight legacy scenes), with all 71,154 camera/frame observations. It needs no manual annotations, target captures or hardware timestamp logs. It never writes to datasets, calibration, masks, trained models or the production v6 worktree.

The implementation lives in `Sapien/scripts/audits/real_rgbd/followup/`. The RigEuler launcher imports that package. Use the existing `yolo26` environment for NumPy/SciPy/OpenCV/Polars/PyArrow; compatible Phase-1 runs are inspected in separate processes using the existing `rigeuler3dgs` CUDA environment. CPU workers are bounded by scene and 16-frame chunks; GPU diagnostics run sequentially while CPU analysis proceeds.

## Run the complete investigation

```bash
AUDIT='/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/rgbd_depth_audit_all_real_2026_11_07'
PYTHON='/home/samuelemara/miniconda3/envs/yolo26/bin/python'
SCRIPT='/home/samuelemara/Joint3DGS/RigEuler3DGS/tools/run_rgbd_identifiability_followup.py'

"$PYTHON" -B "$SCRIPT" \
  --audit-root "$AUDIT" --all \
  --out "$AUDIT/followup_identifiability_20261109" \
  --workers 8
```

Add `--dry-run` to inspect the complete inventory without creating output. Add `--resume` when restarting or extending a run. Resume verifies analysis code/environment, original manifest, committed outputs and recorded input hashes; changed inputs or core analysis code are refused. It does not overwrite finished scene measurements. Do not modify the analysis code during an active run.

All stages run by default: `inventory,synthetic,holes,coverage,verified-geometry,pose-vs-bias,timing,phase1-cause`. Despite its historical name, `verified-geometry` performs automatic conditional association and requires no human review. Use `--stage` with a comma-separated subset if isolating a diagnostic. No stage applies a correction.

## Four-scene validation

Run a fresh pilot before a new method revision:

```bash
"$PYTHON" -B "$SCRIPT" --audit-root "$AUDIT" \
  --scene final_red_fridge_episode_000000 final_light_blue_box_episode_000000 \
    final_black_laptop_episode_000000 final_wooden_box_episode_000000 \
  --out "$AUDIT/followup_pilot_NEW_VERSION" --workers 4
```

The wooden-box episode supplies the outside-object-mask static-background control. Every frame and camera of each selected scene is included. Untextured/ambiguous surfaces remain explicitly unresolved.

## What is measured

- Exact native-to-registered projection replay, collision winners, source UV, calibrated ray support and target occupancy. Native zero-depth rays have no measured target location; they are not assigned invented correspondences. Conservative gap candidates, depth discontinuities and unknown holes remain separate. A diagnostic-only splat is tested against withheld native measurements; held-out values do not enter its values or continuity estimates.
- Predicted-mask depth availability, boundary bands, connected missing regions and frame-to-frame persistence. No automatic mask becomes GT.
- Connected depth patches with robust local normals, separate object/background regions, independent per-camera robust plane fits, held-out residuals, spatial extents and conditional fitting uncertainty. Normals use a four-pixel grid; fits and held-out comparisons use measured fine-resolution pixels. No temporal frame subsampling occurs.
- Cross-camera hypotheses combine projected overlap/visibility, oriented normals, disjoint RGB homography validation, competing texture matches and temporal persistence. Fixed-calibration agreement is never independent physical identity. Opposite faces, repeated logos, nonplanar patches and large layer/calibration conflicts are rejected or retained as ambiguous. Their candidate measurements are preserved. Strong matches remain **conditional automatic identities**, not externally verified surfaces.
- Pose/depth-offset/scale Jacobians, rank, null combinations and bounded fixed-population models with disjoint object-family holdouts. Numerical rank alone is insufficient: fitting noise can lift rank on one physical plane. Insufficient independent surface/family support, rank deficiency or a coefficient at a bound means UNIDENTIFIABLE.
- Existing timestamp/index schemas and exact source-frame mapping. Nominal recorder time is not hardware exposure time. Only identifiable, varying-motion, fixed-population image-content lags are tested on disjoint temporal holdouts; no synchronization shift is applied.
- The actual production CUDA renderer and grouped Phase-1 operator: RGB/depth/alpha forces, per-channel Fisher matrices, coupled leave-one-camera/channel-out responses, nonlinear objective acceptance, true residual convergence and actual composited transmittance-alpha contributor traces. Prior propagation and measurement correction are saved separately. Camera names come from each saved run; camera 000 is already excluded in the supplied two-camera v6 runs.

## Scientific and execution status are separate

`EXECUTION_COMPLETE_SCIENTIFIC_IDENTIFICATION_PARTIAL` means requested measurements finished, not that calibration, timing or absolute sensor accuracy was identified. Inspect `gate_results.json`, `execution_receipt.json`, per-scene logs and qualifications.

Phase-1 qualification first repeats an unchanged three-frame baseline. If that baseline is not repeatable, only the initial three-frame actual-force diagnostic is attempted; full counterfactual trajectory attribution is UNOBSERVABLE. Large nondeterministic differences never enlarge the parity tolerance. A full 189-frame fridge development replay has already exposed this failure; the production solver is left unchanged. Saved v6 code was subsequently committed: tracked-patch equivalence is checked, but original signatures omitted then-untracked helpers, so saved-state parity is also mandatory.

Synthetic controls include known plane bias, diverse and degenerate plane geometry, joint pose/bias/scale recovery, curved patches, registration gaps, native dropout, occlusion/layer steps, false matches from actual reciprocal SIFT on identical patterns on separated planes, and positive/negative timing controls. Actual CUDA controls separately exercise rendering, force additivity, grouped coupling and visibility contributions.

## Outputs

- `ROOT_CAUSE_IDENTIFIABILITY_REPORT.md`: quantitative coverage, hole mechanisms, conditional surfaces and interpretation.
- `execution_receipt.json`, `gate_results.json`, `inventory.json`, `manifest.json`, `.runtime/source_snapshot/`: scope, source versions, immutable hashes and qualifications.
- `registration/`, `coverage/`, `geometry/`: chunked all-frame measurements, every pair's measurable/unresolved status, all competing surface hypotheses, equal-scene/family aggregates and diagnostic projection maps.
- `calibration/jacobian_rank.json`, `heldout_model_comparisons.csv`: conditional identifiability, not corrected calibration.
- `timing/`: original schema inspection and accepted/unobservable image-content hypotheses.
- `phase1/`: renderer controls, per-run parity, actual forces, process/measurement increments, frozen systems, solver qualifications and contributor traces. Scope is available saved training runs, not 118 newly trained models.
- `annotation_review/`: automatic RGB/depth/mask/patch/native-provenance contact sheets; the directory name does not imply a manual prerequisite.

Absolute accuracy and attribution of a target hole to an invalid native ray can remain fundamentally unidentifiable. Identical logos on separate parallel faces can generate indistinguishable RGB matches; a single plane can trade depth bias against translation; constant-velocity motion can trade spatial bias against delay. The controls demonstrate these degeneracies. Optional independent target acquisition is documented in `calibration/target_capture_required.md`; it does not block analysis of existing recordings.
