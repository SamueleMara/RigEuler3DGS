# Phase 4: unified specification (2026-11-11), implementation report

**Spec:** `RigEuler3DGS_UNIFIED_PHASE4_PHASE5_CODEX_SPEC.md`.
**Worktree:** `~/Joint3DGS/RigEuler3DGS-p45`, branch `phase45`, uncommitted.

**Scope.** Part I (Phase 4: 4A, then 4B, then 4C) is implemented. Part II (Phase 5) was not implemented and not run, because the spec forbids it until Phase 4 is validated (Part II §9). The existing Phase-5 code (`recovery/training.py`, `tools/train_phase5_rgbd_gaussians.py`) is untouched. `--stage all` runs 4A → 4B → 4C and never Phase 5.

**Not touched:** the main checkout, the datasets, Phase 0–3 sources and outputs, the installed legacy CUDA extension, and the joint-refinement tool. No joint or camera tensor is optimised anywhere.

**Runs.** All results below come from procedural scenes and unit tests. No full-size real run was made: the user launches those (commands in §6).

## 1. Requirement → implementation → test

| spec | requirement | implementation | test |
|---|---|---|---|
| §2 | one direction, fixed joint `T_t`, metric depth `Z/(A+ε)`, valid = `M ∧ finite ∧ D>0` | `articulated_scene.py`, `exact_objective.ViewTargets` (unchanged) | `test_phase45_direction_anchor.py`, `test_phase4_exact_roi_parity.py` |
| §2.1 | one frozen objective, fixed denominators, SSIM halo | `exact_objective.py` (unchanged) | ROI parity, 4C parity |
| §3.1 | 4A changes labels only: N0 rows, bytes identical | `phase4_stages.stage_4a`: canonical tensor sha256 before and after; snapshot npz | `test_4a_artifact_contract` |
| §3.2 | S / M / OMIT counterfactuals per group; raw `E_S`, `E_M`, `E_OMIT` | `phase4a_exact.ExactEvaluator.sweep` (adds the current-labelling ROI loss); `counterfactual_evidence.jsonl` (scene and ROI totals per group) | `test_4a_artifact_contract` |
| §3.3 | 8 statuses; evidence status and render material kept separate | `phase4a_exact`: `CORE_MOVING_ANCHOR` … `UNTESTED_RESOURCE_LIMIT`; `UNRESOLVED_UNOBSERVABLE` when S and M renders differ by less than the declared magnitude in every view; `render_label` separate | `test_phase4a_exact_labels` (hinge → UNOBSERVABLE) |
| §3.1 / 3.3 | core audited, never silently relabelled | `Phase4A._core_audit`: core split into groups, re-scored on all views; a decisive STATIC → `UPSTREAM_JOINT_OR_SUPPORT_CONFLICT`, integrity gate fails | `test_4a_core_conflict_fails_gate` |
| §3.4 | decode once; view-major; exact affected-region scoring; interaction-aware batches; resumable | unchanged from the 2026-11-11 correction (`view_cache.py`, `footprint.py`, `phase4a_exact`) | `test_phase4_view_cache.py`, ROI parity, `test_resume_…` |
| §3.5 | `4a_original_membership/` (8 files); `complete.json` only when every UID is accounted for | `phase4_stages.stage_4a`, `phase4_artifacts.write_complete` | `test_4a_artifact_contract` |
| §3.6 | forward / backward same outcome | (contract) | `test_4a_forward_backward_same_outcome` |
| §4 | 4B / 4C cannot mutate 4A; hash-verified | `phase4_artifacts.require_complete`: sha256 of every artifact file, re-checked before and after each later stage | `test_4b_refuses_without_valid_4a`, `test_4b_zero_births_and_no_mutation`, `test_4c_drawer_floor` |
| §5.1 | moving proposals only; unresolved originals never seed | `phase4b_surface`: trusted = core + MOVING_CONFIRMED (+ accepted births) | `test_4c_unresolved_original_creates_conflict_not_duplicate` |
| §5.2 | one view-major scan; moving responsibility `A_M` with all occluders | `phase4b_surface.scan`: one render plus two auxiliary indicator-colour passes of the same compositing (A_M, A_trusted, rendered-depth spread) | `test_4b_zero_births_and_no_mutation` |
| §5.3 | residual taxonomy before any birth; mixed depth never back-projected; no erosion bands | `phase4b_surface.CODES` (9 spec codes + `ORIGINAL_MEMBERSHIP_CONFLICT`); `mixed_depth_map` uses only the measured depth (two-sided line fits) | `test_4b_mixed_edge_failures_traced` |
| §5.4 | frontier from trusted support; disconnected search is a separate, off-by-default mode | `propose`: a cell within `frontier_reach` of a trusted Gaussian's 3-σ support; `disconnected_search` | 4C scene tests |
| §5.5 | `x_M = Q_tᵀ(y_W − p_t)`, propagated measurement covariance, shared joint not iid | `scan` (`point_covariance`); joint recorded as conditional | — |
| §5.6 | layer identity without a global min-distance; sidedness; corners as a separate graph | `fragments` (per-view two-sided line test), `build_cells` (robust same-side inlier plane, propagated normal sd), `link_cells` (cell-precision offset test, reliable normals only; corner graph) | `test_4b_thin_layers` |
| §5.7 | proposals only, full provenance, 4B output files | `phase4_stages.write_4b` (11 files); `births: 0` | `test_4b_zero_births_and_no_mutation` |
| §6.1 | ABSENT / MOVING / STATIC; static never committed; complexity penalty; identical objective | `phase4c_validate.validate_cycle`; `NOT_MOVING_STATIC_ALTERNATIVE` exported to `static_alternatives_for_review.json` | `test_4c_static_shelf_never_moving` |
| §6.2 | per-view visibility classes from the renderer's compositing | `evaluate`: candidate responsibility `A_P` from an indicator pass; full composite with the candidate vs the measurement; contradictions must be new | `test_4c_door_back_and_wrong_side` |
| §6.3 | independence (roles), existing-geometry correction shadow, joint sensitivity, statuses | `data_roles` (unchanged); `shadow_gain` (bounded: ±½ voxel, ×1.5 scale); `joint_sensitivity` (declared whole-joint perturbations) | 4C scene tests |
| §6.4 | iterative growth; commit only accepted non-interacting patches; return to 4A on conflicts | `run_4c` cycles; `ORIGINAL_MEMBERSHIP_CONFLICT` → `return_to_4a_requests.json`; leave-one-out against the final scene | `test_4c_drawer_floor` |
| §6.5 | 4C view-major exact scoring, all views, only proven-zero views skipped | `evaluate` (one baseline per view; footprint boxes; OUT_OF_FRUSTUM = empty kernel footprint) | `test_4c_roi_parity` |
| §6.6 | `4c_accepted_moving/` files; N_final = N0 + births; no static births | `phase4_stages.write_4c` | `test_4c_drawer_floor` |
| §7–8 | stage isolation; CLI `--stage 4a / 4b --phase4a / 4c --phase4a --phase4b / all`; never Phase 5 | `tools/run_phase4_recovery.py`, `run_phase45_{real,synthetic}.sh` | `test_optional_final_joint_disabled.py` |

## 2. What changed, by stage

**4A**
- Status names as in the spec.
- `UNRESOLVED` is split into UNOBSERVABLE and AMBIGUOUS.
- Absolute `E_S`, `E_M`, `E_OMIT` are recorded for the whole scene and for the affected region.
- The core is audited group by group.
- New immutable artifact directory; `complete.json` only when the gate passes, otherwise `incomplete.json` with the reasons.
- Scoring, scheduling and exactness are unchanged from the morning correction.

**4B (new: `recovery/phase4b_surface.py`)**
- *Mixed-pixel detection.* A pixel is MIXED_EDGE_AMBIGUOUS when, along some direction, the lines fitted on its two sides disagree and its depth fits neither. The tolerance is k·σ with the prediction variance propagated.
- *Model error.* σ_model is AUTO: the declared floor, or the robust spread of the 4A scene's depth residuals if larger.
- *Layers:*
  - per-view fragments (both one-sided 5-sample lines must agree);
  - cells, whose normal is a plane fit over the pooled same-side inliers, with an angular sd;
  - co-layer links at cell precision, only between cells with reliable normals.
- Candidates must touch the trusted frontier. Each carries its moving version and a static competitor built from the same samples.

**4C (new: `recovery/phase4c_validate.py`)**
- View-major exact evaluation.
- Visibility classes taken from the renderer's own compositing.
- Contradiction-driven pruning.
- The bounded existing-geometry shadow and joint sensitivity.
- Growth cycles with re-scans, then leave-one-out against the final scene.
- Every decision status in the spec, plus `REJECTED_NO_IMPROVEMENT` and `REJECTED_EXISTING_GEOMETRY_CORRECTION`, which the spec's decision criteria need but its status list lacks.

## 3. Test results

Executed 2026-11-11 under `ulimit -n 1024`, on procedural scenes (256 × 192, 3 cameras, 8 frames) unless marked real.

| file | result |
|---|---|
| `test_phase4_view_cache.py` | PASS |
| `test_phase4_exact_roi_parity.py` (4, including one real view) | PASS, max error 1.7e-18 |
| `test_phase4a_exact_labels.py` (5) | PASS; withheld recovery 1.00 / 1.00; hinge → UNRESOLVED_UNOBSERVABLE |
| `test_phase4_unified_4a.py` (3) | PASS |
| `test_phase4_unified_4b.py` (5) | PASS |
| `test_phase4_unified_4c.py` (7) | **6 PASS, 1 FAIL** (door-back recovery) |
| `test_rgbd_cuda_forward_backward`, `_legacy_compat`, `test_phase45_contracts`, `_direction_anchor`, `test_phase5_training`, `test_optional_final_joint_disabled` | PASS (unchanged code, compatibility) |

### Measured

**4A**
- Every original UID gets a status, N0 = N_final, the canonical bytes are identical, and re-invocation is skipped.
- A modified `membership.npz` is refused, by 4A itself and by 4B.
- Forward and backward give identical statuses for all 1,106 originals.
- A core seeded with 96 static back-wall Gaussians: all 96 are flagged UPSTREAM_JOINT_OR_SUPPORT_CONFLICT, no `complete.json` is written, and the core labels are untouched.

**The 19 / 80 mixed-edge failures (4B)**
- 121 source pixels of the 19 old STATIC births.
- Ground-truth trace: 98.3% blend at least two surfaces and lie more than 5 mm from every surface that contributes to the pixel.
- New codes: 118 MIXED_EDGE_AMBIGUOUS, 3 FRONT_SURFACE. None reaches a candidate.
- Single-surface pixels falsely flagged MIXED: 2.7%.

**Thin layers (4B, ray-cast)**
- Opposite faces never share a patch: 3–20 mm, 0 / 35°, 1 / 3 mm noise, 2 mm bias.
- Same-facing layers are separated at 12 and 20 mm with 1 mm noise (also at 8 mm head-on), and at 20 mm with 3 mm noise (head-on, no bias).
- They merge below that, consistent with the declared resolvability 2k σ √(Var+1): 8.7 mm at 1 mm noise, 26 mm at 3 mm.

**Drawer floor (4C)**

| | pre-unified | unified |
|---|---|---|
| births | 61 moving + 19 static | 112 moving, 0 static |
| plane offset (median) | 6.1 mm | 8.6 mm |
| coverage within 3 cm | 0.35 (whole floor) | 0.42 of the observed area (98 / 140 GT Gaussians observed); 0.07 of the never-observed area |
| E | – | 1.049 → 0.837 |

- Growth: 2 patches in cycle 0, 1 in cycle 1, none in cycle 2.
- 2 patches are NOT_MOVING_STATIC_ALTERNATIVE (exported for review).
- 6 are REJECTED_EXISTING_GEOMETRY_CORRECTION.

**Other 4C scenes**
- *Static panel* revealed behind the drawer, observed in 18 views: 0 moving births; 1 patch exported as a static alternative.
- *Corner* (drawer side wall missing): 24 births, 20 on the side wall (coverage 0.57); 0 on the static body wall 2 cm away.
- *Coupling* (undecided door back rendered in the wrong pose): 749 ORIGINAL_MEMBERSHIP_CONFLICT pixels; 4C {DUPLICATE 1, ORIGINAL_MEMBERSHIP_CONFLICT 1}; no moving duplicate.
- *Parity:* 72 item-views (moving, static, combined two-patch); region score = full frame within 3.5e-15 relative; views outside the footprint have a full-frame delta of exactly 0.

## 4. Failing or unmet criteria (not hidden)

1. **Door rear face (spec 6.7) is not recovered: FAIL.** The scene is opened to q_max 1.6 rad so the back face is seen by two cameras over several time blocks.
   - The multi-group back-face patches lose to the existing-geometry correction. With location, scale and opacity only (no recolouring, shift bounded by the measurement tolerance), the 31 door-front Gaussians gain 92 px-equivalent, against the patch's 70.
   - The other back-face patches are INSUFFICIENT_INDEPENDENT_SUPPORT: one camera × time block.
   - At 1.1 rad the back face is effectively seen by one camera in one time block, and the honest answer is INSUFFICIENT_INDEPENDENT_SUPPORT.
   - The wrong-side copy is never accepted.
   - Not tuned: the shadow rule (`shadow_frac` 0.8) and the bounds are declared policy. Whether a refit of existing Gaussians should be allowed to explain a thin-panel back face is a decision for review.
2. **Short-lived surface (spec 6.7) is not recovered.**
   - The floor is exposed in two rows only. Its observed part is about 10 cm from the trusted drawer walls, separated by the junction band of mixed pixels, so the strict frontier does not reach it.
   - In the separately labelled `disconnected_search` mode it does not reach the extra support group either.
   - The floor fragments that are proposed become DUPLICATE or NOT_MOVING_STATIC_ALTERNATIVE (static 3.2 vs moving 1.3 px-equivalent).
   - No candidate is rejected for missing frames (the spec's literal requirement), but nothing is recovered.
3. **Same-facing layers closer than the per-view resolvability merge.** That is 3–8 mm at 1 mm noise; 3–12 mm, and 20 mm with obliquity plus bias, at 3 mm noise. The spec asks for separation at 3 / 5 / 8 mm. A single view cannot detect such a step at that noise, and the cross-view plane test does not undo an in-view merge.
4. **Real data, 8-view subset (red_fridge, unanimous arm, `--view-subset 4`, NOT_FOR_RESULTS): the 4A integrity gate FAILS.**
   - The core audit finds 13 of 65 core groups (52 of 3,474 core Gaussians, 1.5%; 12 singletons and one group of 29) decisively preferring STATIC, with consistency 0.95–1.00 over 6 informative views.
   - Per spec 3.1, 4A writes `incomplete.json`, flags them UPSTREAM_JOINT_OR_SUPPORT_CONFLICT, and 4B / 4C refuse.
   - Whether a small conflicting fraction of the Phase-2 core should be a flag rather than a stop is a policy decision for review. It is not changed here.
   - The other 4A numbers on this subset match the 2026-11-11 report (CORE 3,422, MOVING 2,224, STATIC 17,946, UNRESOLVED 4,416, SUSPECT 29).
5. **Full-scale 4B / 4C performance is not measured.**
   - 4B scan: 3 renders per view.
   - 4C: per candidate-view, 2 renders plus the fit and shadow (bounded steps); joint sensitivity re-evaluates the nominal footprint views once per declared perturbation.
   - On the procedural drawer (24 views, about 14 candidates per cycle, 3 cycles), 4C takes about 40 s.
   - Real cost depends on the candidate count after a passing 4A.
6. **Calibration.** On the real data σ_D is ASSUMED (5 mm); the mixed-pixel detector is correspondingly less sensitive there (prototype on the drawer: it flags 32 of 121 failure pixels at an assumed 5 mm, against 118 at 1 mm). σ_model and σ_layer are AUTO, estimated from the data with 1 mm floors. All decision margins are declared and uncalibrated, and no score is a probability.

## 5. Files

**New**
- `recovery/phase4_artifacts.py`, `recovery/phase4b_surface.py`, `recovery/phase4c_validate.py`
- `recovery/phase4_stages.py` (rewritten; the old driver is kept as `recovery/phase4_stages_legacy.py`)
- `tests/test_phase4_unified_4a.py`, `_4b.py`, `_4c.py`, `tests/helpers/phase4_ctx.py`
- `tests/fixtures/phase4_pre_unified_drawer_births.npz` (the frozen 19 / 80 failure)
- this report

**Modified**
- `recovery/phase4a_exact.py`: statuses, UNOBSERVABLE, ROI absolute loss, core audit
- `recovery/view_cache.py`: spill uses 4 files, not 4 per view
- `tools/run_phase4_recovery.py`: staged CLI
- `run_phase45_real.sh`, `run_phase45_synthetic.sh`: `--output <out>/<RUN_ID>`; `STAGE=4b|4c` need `RUN_ID`
- `tests/helpers/synthetic_articulated.py`: `extra` surfaces, `q_profile`
- `tests/test_phase4a_exact_labels.py`: UNOBSERVABLE expectation
- `tests/test_phase4_view_cache.py`

**Retired**
- `tests/test_phase4bc_exact.py` → `tests/legacy_phase4bc_pre_unified.py`
- `recovery/phase4b_exact.py`, `phase4c_exact.py` are kept as the pre-unified reference and are no longer called by the CLI.

**Untouched:** Phase 5 code, the joint-refinement tool, and the CUDA sources (this session).

## 6. Commands

Tests:

```bash
cd ~/Joint3DGS/RigEuler3DGS-p45
for t in test_phase4_view_cache test_phase4_exact_roi_parity test_phase4a_exact_labels test_phase4_unified_4a \
         test_phase4_unified_4b test_phase4_unified_4c; do ~/miniconda3/envs/rigeuler3dgs/bin/python tests/$t.py; done
```

One real scene, 4A only (the spec's Gate A):

```bash
cd ~/Joint3DGS/RigEuler3DGS-p45
SCENES=red_fridge ARMS=unanimous STAGE=4a RUN_ID=unified_4a bash run_phase45_real.sh
```

- Writes `<RES>/ours_phase4_corrected_real/red_fridge/unanimous/phase4/unified_4a/4a_original_membership/` and `console_4a.log`.
- Expected about 50 min (extrapolated).
- Expect `incomplete.json` if the core conflict seen on the subset persists over 378 views.

Only after a 4A `complete.json`:

```bash
SCENES=red_fridge ARMS=unanimous STAGE=4b RUN_ID=unified_4a bash run_phase45_real.sh
SCENES=red_fridge ARMS=unanimous STAGE=4c RUN_ID=unified_4a bash run_phase45_real.sh
```

## Addendum 2026-11-11 evening: recovery frame stride and Phase 5 entry point (user-authorized)

**fridge_10905 `subset15` (120 views), 4A checked against GT part labels (evaluation only):**
- MOVING precision is 1.000 and recall is 0.935 (9895 true positives, 0 false positives, 683 false negatives).
- Among STATIC_CONFIRMED rows, 2.0 % are GT-moving.
- UNRESOLVED_AMBIGUOUS (337 rows) is 90 % GT-moving.

**4B on the same run:**
- 6181 birth-source pixels. Of these, 5319 lie on existing support and 381 are membership conflicts.
- 468 samples were back-projected into 398 cells, every one a singleton, so there were 0 candidates.

**`--recovery-frame-stride K`** (`tools/run_phase4_recovery.py`; `RECOVERY_STRIDE` in `run_phase45_synthetic.sh`):
- 4A always uses all views. 4B and 4C use every K-th Phase-3 frame of every camera.
- 4A now stores `view_hashes.json` inside its hashed artifact. Each strided recovery view must be byte-identical to 4A's copy.
- 4B and 4C record `recovery_views` (with `full_sequence: false`). 4C refuses a stride that differs from 4B's.

**Phase 5** (`tools/train_phase5_rgbd_gaussians.py`, now `--phase4-dir RUN --output DIR`):
- Preflight requires a hash-verified 4A→4C chain. The upstream files are read from the 4A manifest and checked by hash and joint fingerprint.
- It trains on all views.
- RGB loss: `utils/loss_utils.l1_loss` plus `fused_ssim`, the 3DGS `train.py` call, with no silent fallback. `--loss-ssim-impl 3dgs_python` selects the reference Python SSIM instead.
- It writes `input_proof.json`, `unresolved_originals.npz`, the metrics files and `complete.json`.

**Scripts:** `run_phase5_synthetic.sh` runs Phase 5 alone; `run_phase45_then_5_synthetic.sh` chains Phase 4 and then Phase 5, scene by scene.

**Test:** `tests/test_phase4_recovery_stride_phase5_preflight.py`.
