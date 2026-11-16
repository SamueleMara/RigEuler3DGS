# Phase 4 real validation report

Status 2026-11-11: **the complete 378-view real Phase 4A has NOT been run.** Real data has been processed only at benchmark size (marked NOT_FINAL), because the user launches every full-size run. Ground truth was not read anywhere in inference: joint type comes from the documented R/L rule, and GT axes are used only by the separate comparison renderer.

## 1. What has run on real data (red_fridge, unanimous arm, forward, revolute, cam_001 + cam_002)

### 1.1 Exactness and caching

- `test_real_red_fridge_view` and the 50-group parity benchmark: exact (see `PHASE4_EXACTNESS_REPORT.md`).
- 378-view decode-once preparation: Gate A (see `PHASE4_PERFORMANCE_REPORT.md`).

### 1.2 CLI `--stage all --view-subset 4` (8 views: frames 100, 163, 225, 288 per camera) — NOT_FINAL

**4A**

- **Core check:** MOVING. Δ = 0.637 (195,817 px-equivalent) over 6 informative views; consistency 1.00. Channels: depth +0.528, mask +0.066, L1 +0.029, SSIM +0.014.
- **Groups:** 288. Batches: 4, with every realised change equal to the predicted one (additivity error ≤ 3e-16).
- **Statuses (Gaussians):**

| status | count |
|---|---|
| CORE_MOVING | 3,474 |
| MOVING | 2,224 |
| STATIC | 17,946 |
| UNRESOLVED | 4,416 |
| SUSPECT_ORIGINAL | 29 |
| UNTESTED_RESOURCE_LIMIT | 0 |

- **Scene objective:** E 0.8003 → 0.5849. Per view (channels L1, SSIM, mask, depth):

| view | before | after |
|---|---|---|
| cam_001 f100 (anchor) | .0069 .0128 .0126 .0056 | unchanged (identity transform) |
| cam_001 f163 | .0094 .0144 .0502 .0491 | .0086 .0134 .0374 .0312 |
| cam_001 f225 | .0107 .0162 .0395 .0636 | .0100 .0152 .0254 .0404 |
| cam_001 f288 | .0111 .0171 .0309 .0662 | .0104 .0161 .0166 .0434 |
| cam_002 f100 (anchor) | .0056 .0141 .0180 .0039 | unchanged |
| cam_002 f163 | .0127 .0183 .0493 .0357 | .0107 .0166 .0261 .0273 |
| cam_002 f225 | .0113 .0170 .0521 .0332 | .0102 .0155 .0294 .0236 |
| cam_002 f288 | .0121 .0177 .0497 .0334 | .0111 .0163 .0269 .0236 |

- **Joint sensitivity** (whole joint, scene objective, E − E_nominal):

| perturbation | ΔE |
|---|---|
| axis ±u | +0.0030 / −0.0012 |
| axis ±w | +0.0029 / −0.0012 |
| pivot ±u (5 mm) | +0.0149 / −0.0086 |
| pivot ±w (5 mm) | +0.0083 / −0.0045 |
| q·1.02 / q·0.98 | **−0.0063** / +0.0093 |

Several perturbations lower the 4A objective, so the fixed Phase-3 joint is not a local optimum of the Phase-4 RGB-D-mask objective on these views. It stays fixed (Phase 3 is authoritative) and is reported as JOINT_CONFIDENCE_UNCALIBRATED. This is input for a later, explicitly enabled joint review, not something Phase 4 acts on.

**4B**

- Sensor: ASSUMED (no calibrated model for this capture).
- Roles in view 0: GUARD 34 %, PROPOSAL 22 %, FIT 22 %, VALIDATION 22 %.
- Pixel counts: 6,975 birth-source pixels → 3,193 on existing render support (dropped), 40 on undecided-original alternates (dropped), 3,742 back-projected.
- 23 source-equivalent candidates.

**4C**

| decision | candidates |
|---|---|
| ACCEPTED_MOVING | 4 (38 Gaussians) |
| REJECTED_OCCLUSION_CONTRADICTION | 6 |
| AMBIGUOUS_MATERIAL | 1 |
| REJECTED_DUPLICATE | 2 |
| REJECTED_TOO_SMALL | 10 |

- Leave-one-out withdrew 0.
- Final audit: E 0.5849 → 0.5781, with one view slightly worse.

These numbers come from 8 of 378 views and are not results. They show that the stages, markers, checkpoints and outputs work on the real tree.

## 2. Gate status relevant to real use

- 4A: Gates A, B and C (synthetic) pass. **The complete real 4A is the next step (command below).**
- 4B/4C: Gate D has one known failing criterion (material purity of births; see `PHASE4_CORRECTION_AUDIT.md` §4), and the full-scale 4C runtime is not measured. **Do not use 4B/4C outputs as results yet.**

## 3. Command: one real scene, 4A only, unbuffered log, no downstream stage

```bash
cd ~/Joint3DGS/RigEuler3DGS-p45
SCENES=red_fridge ARMS=unanimous STAGE=4a bash run_phase45_real.sh
```

Equivalent explicit call (writes `.../ours_phase4_corrected_real/red_fridge/unanimous/phase4/<run_id>/`):

```bash
cd ~/Joint3DGS/RigEuler3DGS-p45
RES="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26"
REAL="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real"
n=final_red_fridge_episode_000000; R="$RES/ours_v6_two_cams/$n/unanimous/pipeline/$n"
O="$RES/ours_phase4_corrected_real/red_fridge/unanimous/phase4"; mkdir -p "$O"
~/miniconda3/envs/rigeuler3dgs/bin/python -u tools/run_phase4_recovery.py --source "$REAL/$n" \
  --direction forward --model "$R/static_bidi/initial" --phase2-dir "$R/bidirectional/forward_rigid" \
  --phase3-dir "$R/bidirectional/forward_joint" --joint-type revolute --cameras cam_001,cam_002 \
  --stage 4a --run-id fridge_4a --output "$O" 2>&1 | tee -a "$O/fridge_4a_console.log"
```

- **Duration:** expected about 50 min (extrapolated; see the performance report).
- **Progress:** `tail -f "$O/fridge_4a/phase4.log"`.
- **Resume after an interruption:** the same command with `--resume "$O/fridge_4a"` in place of `--run-id fridge_4a`.

Outputs to review:

| file | contents |
|---|---|
| `support_completion/score_manifest.json` | core check, batch history with additivity, counts, counters |
| `support_completion/full_scene_audit.json` | per-view, per-channel loss before / after 4A |
| `support_completion/groups.json` | every group: status, reason, Δ per channel, per camera, time-block agreement |
| `support_completion/group_evidence.npz` | exact per-view evidence |
| `support_completion/joint_sensitivity.json` | whole-joint perturbation ranges |
| `stage4a_complete.json`, `cache_manifest.json` | resources, decode counters |
