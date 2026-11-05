# Calibrated bootstrap: surgical corrections and evidence gates (2026-11-05)

Specification: `RigEuler3DGS_Codex_Surgical_Fixes_4c88725.md`. Baseline `4c88725`, branch `calibrated-bootstrap`.

**Gate 0.**
- HEAD was `4c887257ef81` with no diff against `4c88725`.
- The working tree was clean, apart from the untracked spec and handoff files and the `submodules/sam3` pointer, none of which were touched.
- The overnight batch (`~/run_calibrated_bootstrap.sh`) had never been started (no log, no `ours_v4*` folders), so no running experiment was disturbed.
- Every run in this report is a small pilot in a scratch directory. **No full-scale run, A/B arm or batch was started.**

## 1. Silhouette loss (P0, confirmed defect)

**Files and functions:**
- `estimator/surface_bootstrap.py` (+251–303): new `pixel_partition`, new `bootstrap_loss_terms`;
- `tools/train_tracking_bootstrap.py` (+221–275): the partition, `losses` and `evaluate`.

**The defect.** `held`/`train` were drawn inside the object mask (`& msk`). The silhouette weight was therefore zero wherever M = 0, so opacity spilled onto the background cost nothing.

**Now:**
- **The split.** One seeded, disjoint train/held split of the **whole image** per camera, `pixel_partition(H, W, f, seed, camera)`, drawn once and never resampled.
- **Per-term supports on a partition S:**
  - alpha: all of S, foreground and background, E_α = Σ_{p∈S}(A − M)² / (2σ_A²|S|). This is a soft loss on the hard binary mask; it never turns background into a depth measurement.
  - RGB: S ∧ M, ½ Σ W_rgb |C − I|² / |S ∧ M|.
  - depth: S ∧ M ∧ observed-valid ∧ A > 0.5 (the existing gate is kept), as a Student-t NLL of inverse depth with Var(1/z) = Var(z)/z⁴.
- **Empty supports** give exactly 0 with count 0 (finite).
- **Reporting:** train and held-out scores use identical definitions. Each checkpoint reports foreground/background/depth counts, foreground miss and off-mask opacity per camera.

## 2. Content-safe reuse (P0, confirmed defect)

**Files:**
- `tools/stage_cache.py` (new);
- `tools/run_noise_aware_rigeuler.py` (`cached_stage`, the calibration and bootstrap sections, +178–350);
- `tools/train_tracking_bootstrap.py` (+167–170: a directory holding only `stage_signature.json` is accepted).

**What the signatures cover.** All hashes are of file content, never names or mtimes:
- **Calibration:** source, ordered cameras, the inclusive interval, sha256 of every rgb/depth/mask file in it, the estimator options (`bins`, `min_bin_samples`), and an implementation fingerprint.
- **Bootstrap:** endpoint and frame, sha256 of the COLMAP `cameras.bin`/`images.bin`, sha256 of that frame's rgb/depth/mask files, sensor policy and `sensor_model_id`, and every setting:
  - Gaussian count, kNN, refinement steps and checkpoints;
  - held-out fraction, silhouette σ and thickness weight;
  - both model-mismatch σs and both P0 prior σs;
  - learning-rate scale and seed;
  - plus an implementation fingerprint.

The implementation fingerprint is the content hash of the source files that stage executes, not the git commit. An unrelated commit therefore doesn't invalidate a result, while any change to the code that produced it does.

**Decision rules.** `stage_signature.json` (schema 1) records the signature, its hash, the status and, on completion, the sha256 of every mandatory output.

| situation | decision |
|---|---|
| matching signature, status complete, outputs unchanged and structurally valid | reuse |
| signature differs | refuse, with the differing fields |
| legacy directory without a signature | refuse |
| outputs changed or missing since completion | refuse |
| structural problems | refuse |
| same signature, status not complete | moved aside to `<dir>.incomplete-<UTC>` (never deleted), then redone |

The structural checks:
- calibration: `sensor_model.json` passes its content-id check and the camera set matches;
- bootstrap: the requested checkpoint is the top-level PLY, P0 is [N,6,6] with N from the PLY, and the manifest count, sensor id and endpoint/frame all match.

The pipeline copy of each endpoint must be byte-identical to its bootstrap, or the run is refused. Downstream Phase 1/2/3 reuse is already keyed on the endpoint PLY sha, the P0 sha and the sensor-model sha in the driver signatures. `--dry-run` is read-only: verified by an identical file listing, sizes and mtimes before and after.

## 3. Physical-time exports (P1)

**Files and functions:**
- `utils/bidirectional_time.py: chronological_twists` (+64–89);
- `tools/run_rigid_se3_pipeline.py` (+444–468, and the summary at 635);
- `tools/run_joint_phase3.py` (+413);
- `tools/run_motion_coherence_support.py` (+400–416);
- `bidirectional_iekf/free_state.py: FullStateRecorder.write` (+113–221).

**Changes:**
- **Twists from measured intervals.** ξ[t] = Log(T[t+1] T[t]⁻¹) / (time[t+1] − time[t]), with `dt` either a scalar (unchanged behaviour) or the [T−1] positive chronological durations. The left/spatial convention and last-row padding are preserved.
  - Phase 2 takes the durations from the measured time base, never `ns.dt`. For backward runs, the chronological intervals are the processing intervals reversed.
  - `chronological_twists.npy` is now written in both directions, in original chronological order.
- **Honest time base.** `time_base.json` adds `uniform`, `intervals_s_processing_order` and `intervals_s_chronological`. `dt_s` is null unless the sampling is uniform, and the Phase 2/3 summaries follow the same rule.
- **Honest recorder metadata.** The Phase 1 full-state recorder reports the measured chronological intervals and `dt_is_measured = true` when the loader read timestamps. The IEKF transition is unchanged.

## 4. Read-only Phase-1 audit (P1, diagnosis)

**Files:**
- `tools/kpart_pipeline.py: bootstrap_positions` (+1840–1857): an optional `args.phase1_audit` callback, called under `no_grad` with *clones* of P_prev/P_post; its return value is ignored;
- `tools/audit_calibrated_phase1.py` (new).

**What it reports per frame, using the production measurement terms:**
- **Covariance:**
  - P_prev, F P_prev Fᵀ, Q(dt), P_pred and P_post, as per-Gaussian position/rotation/velocity SD quantiles, pose-velocity cross-norms and PSD minima;
  - the pose variance contributed by the velocity block alone, F_pv P_vv F_pvᵀ.
- **Per camera:**
  - support counts and energies;
  - residuals standardised by the *unweighted* sd: sensor σ plus declared mismatch, each labelled. The IRLS precision would cap |z| at √(ν+1), which the first pilot exposed.
  - |z| quantiles and the coverage within 1/2/3 σ;
  - mask IoU, off-mask opacity, and per-camera Fisher trace and weak directions;
  - the per-camera gradient J_cᵀ W_c r_c, with its cosine against the accepted update and against every other camera.
- **State:** accepted/rejected steps, stop reason, and displacement quantiles.
- **GT (evaluation only):** static drift and moving displacement reported separately.

No χ² NIS claim is made (Student-t weights, correlated pixels).

## 5. Bootstrap geometry evaluation (P2)

`tests/test_bootstrap_geometry_eval.py` measures `fit_surface` on synthetic geometry with known truth. The tolerances are stated test tolerances, not pipeline gates.

| scene | measured | test tolerance | result |
|---|---|---|---|
| tilted plane; 3 cameras at 0.5/2/6 mm, different ray angles | centre error median 0.10, p90 0.25, max 0.39 mm | median < 0.5 mm | PASS |
| (same) | normal error median 2.9°, p90 6.2°, max 11.5° | median < 2° | **FAIL** |
| fit's own normal variance vs Monte-Carlo scatter (30 draws) | empirical/predicted sd: median 1.00, p90 1.21 | not overconfident by > 2× | PASS |
| 5 cm sharp step, seeds 6 mm from the edge | centre error median 0.05, p90 0.12 mm | p90 < 2 mm | PASS |
| 4 mm thin panel, faces seen from opposite sides | 71% of seeds (89% with k = 48) pulled > 1 mm off their own face; error median 1.3 mm | < 10% | **FAIL** |
| plane partly behind an occluder 3 cm in front | background seeds: centre error p90 0.23 mm | p90 < 2 mm | PASS |

**Thin panel: reproducible failure.**
- The first, unweighted PCA fits a plane between the two layers. Both layers then sit symmetrically at about 4σ, so the Student-t IRLS has no basis to pick a side.
- Follow-up (not implemented here): restrict a seed's samples to cameras that see the seed's side, nᵀ(camera − y) > 0. That is a visibility-consistent association rather than a new threshold.
- The fitting was not changed in this patch.

**Not evaluated:** P0 (the renderer-Fisher block approximation) against actual estimation error. It remains labelled as a block approximation, not an exact marginal.

## 6. Pilots (red_fridge episode 0; SAPIEN fridge_10905)

**Setup:**
- **Calibration:** frames 0–89, `sensor_model_id aaa04e18c2af`. cam_000 now selects a+bz², with depth σ 0.70, 0.95 and 0.88 mm for cam_000, cam_001 and cam_002.
- **Bootstraps:** 8,000 Gaussians, 200 steps, checkpoints 0/100/200, seed 0, at frames 90 and 100 (real) and frame 0 (synthetic, ideal sensor).
- **Responsibility check:** sum error ≈ 4e-7 in every bootstrap, with no unseen Gaussian.
- **Fixed population:** counts are identical at every checkpoint (8,094 / 8,020 / 8,068); per-index movement over the refinement has median 7 mm, i.e. no reordering.

| pilot | window | static drift (median / p90 / max) | standardised depth residual (median, within 3σ) | camera gradients |
|---|---|---|---|---|
| real **static control**, calibrated, bootstrap P0 | 90–93 (onset ~105) | 6.4/19/73 mm after 1 frame; 13/34/129 mm after 2; **20/50/191 mm after 3** | 26–36σ; 3–8% within 3σ | ≈ 50% of camera pairs disagree per Gaussian |
| real motion window | 100–110 | median displacement grows 4 → 77 mm | 81–123σ; 1–2% within 3σ | ≈ 50% disagree |
| synthetic, ideal, GT labels (evaluation only) | 0–5 | GT-static drift 3.2 → 8.3 mm (p90 22 mm); GT-moving 11 → 54 mm | 0.3–0.4σ median; 81–93% within 1σ | ≈ 50% disagree |

**What the audit shows.** These are measured mechanisms; their cause is not proven, and no fix was attempted.
1. **The first prediction is dominated by the declared velocity prior.** P_prev position SD is 0.19 mm (bootstrap P0), but P_pred is **40.0 mm**. That is entirely the velocity contribution: 1 m/s × 0.04 s, matching the oracle. From the second frame on, the velocity-origin pose SD is about 0.6 mm.
2. **The static drift grows linearly:** about 6.5 mm per frame, consistent with a spurious velocity learned in the first step and then extrapolated.
3. **With the calibrated σ, the real residuals are 26–123 times the sensor noise.** Sensor noise is therefore not the dominant error on real data; model mismatch is. With the declared default σ on synthetic data, residuals are consistent with the noise model.
4. **Per-camera gradients agree with each other and with the accepted update only at chance level (≈ 50%).** The per-Gaussian evidence is noise-dominated.

The ablations that would separate these mechanisms (velocity prior, declared mismatch, endpoint model) are the arms of §7. They are not run here; no constant was tuned.

## 7. Controlled A/B (§6 of the spec): NOT RUN

**Blocker for arms A and C:** there is no standard-3DGS endpoint at the real motion start (frame 100). `colmap/` holds frame 0 only, and the canonical-endpoint check rightly refuses a frame-0 model for a window starting at 100. They need a standard 3DGS trained at frames 100 and 288 (e.g. via `tools/make_endpoint_colmap.py`), which is a training run for the user.

Arms B, D and E are runnable with the current tools. The commands use **new** output folders:

```bash
RES="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ab_calibrated_2026_11_05"
EP="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/final_red_fridge_episode_000000"
COMMON=(--source "$EP" --dataset-kind real --motion-start-frame 100 --motion-end-frame 288 --num-gaussians 30000 --refine-iters 1000 --seed 0)
# B: new bootstrap, legacy assumed sensor, legacy P0 -- the orchestrator always passes the bootstrap P0;
#    for the legacy-P0 arm run the driver directly without --bootstrap-p0 on B's pipeline/<scene> folder
python tools/run_noise_aware_rigeuler.py "${COMMON[@]}" --sensor-noise-model assumed --out "$RES/B"
# D: new bootstrap, calibrated sensor (frames 0-95), bootstrap P0
python tools/run_noise_aware_rigeuler.py "${COMMON[@]}" --rgbd-noise-calibration --calibration-start-frame 0 \
  --calibration-end-frame 95 --sensor-noise-model calibrated --out "$RES/D"
# E: as D plus a declared model mismatch (an ABLATION, not a default)
python tools/run_noise_aware_rigeuler.py "${COMMON[@]}" --rgbd-noise-calibration --calibration-start-frame 0 \
  --calibration-end-frame 95 --sensor-noise-model calibrated --rgb-model-sigma 0.05 --depth-model-sigma 0.005 --out "$RES/E"
# the read-only audit on any arm's endpoint (static control shown):
python tools/audit_calibrated_phase1.py --model "$RES/D/bootstrap/initial" --source "$EP" --motion-start-frame 100 \
  --motion-end-frame 103 --frame-timestamps auto --sensor-model "$RES/D/calibration/sensor_model.json" \
  --p0-geometry "$RES/D/bootstrap/initial/geometry_information.npz" --gt-labels off --out "$RES/D/audit_static.json"
```

## 8. PASS / FAIL / NOT RUN

| test | status |
|---|---|
| §1 split deterministic, disjoint, whole image; background in both partitions | PASS |
| §1 negative-ray oracle (CPU), old objective zero | PASS |
| §1 positive-ray oracle (CPU), sign unchanged | PASS |
| §1 RGB/depth never on background or invalid pixels; alpha on both; empty support finite | PASS |
| §1 held-out bytes do not change a training step | PASS |
| §1 renderer oracle (CUDA, real rasterizer): off-mask Gaussians get opacity gradient now, exactly 0 before | PASS (18 Gaussians) |
| §1 fixed population at all checkpoints | PASS (pilots) |
| §2 same → reuse; rgb σ / depth σ / count / refine iters / prior σ / silhouette σ / sensor id / same-size bytes → refuse | PASS |
| §2 calibration interval / one mask file / edited sensor model → refuse | PASS |
| §2 missing P0 file / corrupted manifest id / swapped checkpoint / legacy dir → refuse; nothing deleted; interrupted → redo | PASS |
| §2 real orchestrator: reuse, refusal on rgb σ, dry run read-only | PASS |
| §3 25 Hz constant twist, nonuniform intervals, scalar compatibility, forward = reconstructed backward, non-positive durations refused | PASS |
| §3 real-data metadata uses timestamp intervals | PASS (code path; pilot loaders recorded 0.04 s) |
| §4 linear-Gaussian predict decomposition; velocity prior 0.04 m / 0.01265 rad | PASS |
| §4 tracking with audit equals tracking without | PASS within the GPU nondeterminism floor (two runs without the audit differ by 1.5e-8; with it by 7.5e-9). **Not bitwise.** |
| §4 camera identity swap / missing calibration refused | PASS |
| §4 pilots finite, missing depth handled | PASS |
| §5 plane centre / step / occluder / normal-variance calibration | PASS |
| §5 plane normal error < 2° | **FAIL** (median 2.9°) |
| §5 thin panel layers not merged | **FAIL** (71%) |
| §5 P0 vs actual estimation error | NOT RUN |
| §6 static control (real) | RUN: drift reported (§6), no pass criterion |
| §6 motion control (real, synthetic with GT) | RUN: reported |
| §7 A/B arms A–E | NOT RUN (A/C blocked: no baseline endpoint at frame 100) |
| Regression: surgical_review 46, calibrated_bootstrap 49, bidirectional_sequence_direction 54, rigid_se3_phase2 47, joint_phase3_runner 12, masked_default_repair 76, iekf_end_to_end, kpart_window 74, masked_rgb_observation 46, motion_coherence_support 41, strict_trajectory 25, fb_pixel_overlap_core 58 | PASS |
| Full-scale accuracy | NOT RUN |

## 9. Remaining assumptions and limitations

- **Sensor model:**
  - persistent sensor bias is not identifiable from static frames;
  - the RGB full covariance is reduced to its diagonal;
  - temporal and shared-camera correlations are not modelled.
- **Covariance:** P0 and P_post are per-Gaussian blocks, so inter-Gaussian Fisher coupling is dropped and they are likely overconfident.
- **Observability:** pose observability is weak for many surface Gaussians.
- **Thin structures:** the surface fit merges them (§5).
- **Model mismatch:** on real data it dominates the calibrated sensor noise by one to two orders of magnitude (§6).
- **Accuracy:** full-scale accuracy is unvalidated, and no improvement is claimed.
