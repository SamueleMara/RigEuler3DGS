# Standard pipeline vs. the five-stage framework: conformance and failure points

*2026-10-16*

**Theory.** `assets/Bidirectional_Articulated_3DGS_IEKF_Final_Framework.pdf` (16 Oct 2026).

**Implementation.** The standard bidirectional pipeline, `run_bidirectional_all.sh`. The code that
produced `results_11_10_26/ours/<scene>/bidirectional` is preserved as tag
`bidirectional-batch-2026-10-12`; the stage code audited here is unchanged since then, apart from
additive options.

**Scope.**

* The audited stages are Phase 1F/1B, support, Phase 2F/2B and joint identification, plus the new
  Phase 3 runner.
* The fused forward/backward Phase-1 merge (IEKS) is not part of the standard pipeline and is not
  audited.

**Evidence.**

* Read-only code audit, with file:line references below.
* The results of the 10 scenes × 2 directions run, scored afterwards against the dataset GT, which is
  never read during inference: `endpoint_eval_v3/evaluation_gt`.
* Two smoke runs of the new Phase-3 runner.

Items marked **(verified)** were re-read in the code for this report.

## 1. Bottom line

1. **The pipeline works where its Phase-1 support is right.**
   * 16 of 20 scene×direction runs recover the joint: axis ≤ 0.27°, angle MAE ≤ 0.02 rad, slide MAE
     ≤ 1.4 cm.
   * In the 4 failures, the first thing that goes wrong is Phase 1 or the support: oven F, storage F,
     washer F and laptop B.
   * Phase 2 and the joint fit never turned a good support into a bad joint.
2. **The measurement side is faithful to the theory.** This covers the shared renderer and analytic
   Jacobians, exact rigid transport including orientations, static Gaussians kept in the same
   renderer, one shared pose across cameras, and no Adam or Procrustes substitution.
3. **The estimation side deviates in ways the theory explicitly forbids or requires.**
   * Phase 1 is a sequence of per-frame warm-started MAP solves in a per-frame-rebuilt rigid-group
     subspace. No covariance carries across frames, and only centres are exported.
   * The support has no uncertainty weighting, no AMBIGUOUS class downstream and no calibrated null.
   * Phase 2's process prior is numerically inert.
   * Joint identification produces no P/R/unresolved decision.
   * Phase 3 was never run on bidirectional outputs, and could not be until today.
4. **Phase 3 now runs, forward and backward** (`tools/run_joint_phase3.py`, §6).
   * As specified for this project, it keeps the joint type and geometry from joint identification
     **fixed** and re-estimates only the per-frame displacement q_k (and q̇_k) on the RGB-D.
   * On fridge it reproduces the Phase-2 track to within 0.006 rad. It does not improve it: angle MAE
     0.0026 vs 0.0022 rad forward, 0.0073 vs 0.0071 rad backward.

## 2. Stage-by-stage conformance

Verdicts: ✅ match, 🟡 partial, ❌ deviates or missing.

### Shared measurement model and IEKF (Sec. 2–3, 10.1)

| Contract | Verdict | Where / note |
|---|---|---|
| Alpha-compositing renderer, normalized depth, analytic JVP/VJP | ✅ | Tests: `test_iekf_cuda_operators.py` T6–T10, `test_unified_cuda_renderer_jvp_vjp.py` (all synthetic, small; the H^T contributor-cutoff bug was only visible at full scale) |
| Information-form iterated update (eq. 9), matrix-free Hv / H^T z | ✅ | Trust-region Steihaug-CG, not LM; equivalent role |
| Accept on the true nonlinear MAP objective (eq. 10) | 🟡 | Phase 1: yes (`sparse_iekf.py:606-617`). Phase 2/3 (`joint_iekf.py`): `_energy` is likelihood-only while the predicted reduction includes the prior **(verified, `joint_iekf.py:149-162, 258`)** |
| R and Q in physical units | 🟡 | Depth σ 0.02 m, RGB σ 0.1. dt is assumed (1/30 s in Phase 1; **1 frame in Phase 2/3**, verified `rigid_part_runtime.py:340`) |
| GT only in post-run evaluation | 🟡 | No GT enters any estimate. But `load_scene()` without `inference_only=True` attaches GT part labels (`points3D_parts.ply`) to the model in the Phase-1/2 processes (`run_iekf_scene.py:118`, `scene/dataset_readers.py:189`). They are read only by GT-scoring code (`estimator/optimized_gt.py`). Phase 2 also scores GT inline. The new Phase-3 runner loads in inference-only mode |

### Phase 1F/1B: free-motion IEKF (Sec. 4, 10.2)

Entry point: `tools/run_motion_coherence_support.py` → `tools/kpart_pipeline.bootstrap_positions` →
`estimator/sparse_iekf.iekf_frame_step`.

| Contract | Verdict | Where / note |
|---|---|---|
| 12D state (μ, R, v, ω), constant-twist transition, left J_l | ✅ | `process.py:113-128, 188`; `test_iekf_process_covariance.py` |
| P⁻ = F P F^T + Q carried frame to frame | ❌ | **(verified)** `kpart_pipeline.py:1749-1752` rebuilds P = diag(1e-2, 1e-2, 1, 0.1) every frame, and P_post is discarded. Velocity information never accumulates; the "filter" is a chain of warm-started per-frame solves |
| Per-Gaussian 12N state | ❌ | **(verified)** `partition=None` (`run_motion_coherence_support.py:276`) makes `iekf_frame_step` rebuild 64-Gaussian groups from the predicted means **every frame** (`sparse_iekf.py:425`). Each frame's correction lives in a 12-DoF rigid motion per group. The module's own §3.1 requires the partition to be immutable |
| F anchored at frame 0, B at frame T, processed in reverse | ✅ | `kpart_pipeline.py:1740-1746`, strict `verify_endpoint_provenance` |
| Appearance, opacity and shape frozen; own population only | ✅ | Radiance is the DC term only, not the trained SH |
| Required exports: μ, R, v, ω, covariances, Q, innovations/NIS, rate convention | ❌ | Only `phase1_centers/centers_frame_*.npy` and a manifest. `--full-state-out` exists but the standard driver does not pass it. **Downstream stages must invent orientations and rates** |
| Backward rates converted to physical time | ❌ | Rates are not exported, so this is moot |

### Rigid-coherence support (Sec. 5, 10.3)

Entry point: `estimator/motion_coherence_support.py` (strain, Ncut) followed by
`estimator/coherence_extraction.py` (hierarchy, held-out Kabsch CV).

| Contract | Verdict | Where / note |
|---|---|---|
| Strain z_ij = log((ℓ+ε)/(ℓ̄+ε)), median rest length | ✅ | `motion_coherence_support.py:139-140`. Computed between the 192 supernode centroids, not between Gaussian pairs |
| Weights from visibility and Phase-1 uncertainty | ❌ | Uniform weights (`w_t` not passed, `:374`). Phase-1 covariance is used nowhere, and the independence approximation is not disclosed |
| Labels STATIC / MOVING_RIGID / AMBIGUOUS with posterior margins | 🟡 | The extractor assigns 3 classes (`coherence_extraction.py:544-573`), but there is no eq.-20 posterior. INCOHERENT is merged into static on export (`run_motion_coherence_support.py:410-411`), so **unresolved geometry is rendered as static in Phase 2** |
| Motion vs. static from displacement and its uncertainty; a null hypothesis | 🟡 | `NO_COHERENT_MOVING_COMPONENT` exists, but the gate is η_cv > 0 / ΔRSS > 0, with no noise calibration. All 20 runs report `RESOLVED`, including storage F and washer F, where the support moves 2 mm and 0.7 mm |
| Supernodes only within reliable neighbourhoods | ❌ | k-means on canonical xyz; whole-node label propagation (about 142 Gaussians per node). Node count 192 was chosen on GT F1 (`run_motion_coherence_support.py:110-119`), against Sec. 10.7 |
| High-precision moving core frozen for Phase 2 | 🟡 | Phase 2 takes all `dynamic_gaussians` (seed + recruited) with no precision gate |
| Backward reference geometry | ❌ | `canonical_knn_graph(traj[0])` uses chronological frame 0. For a backward run that is the filter's last, most drifted estimate, not its anchor |

### Phase 2F/2B: rigid SE(3) IEKF (Sec. 6, 10.4)

Entry point: `tools/run_rigid_se3_pipeline.py` → `estimator/rigid_part_runtime.run_rigid_window`.

| Contract | Verdict | Where / note |
|---|---|---|
| One (T_k, ξ_k) per part; μ = Qμ̄ + b, R = QR̄, Σ = QΣQ^T in the render | ✅ | `rigid_part_se3.py:44-50, 185-202`; orientations reach the rasterizer |
| Static Gaussians in the same renderer; J_iT; H2 = H^G J [I6 0] | ✅ | `rigid_part_se3.py:194-264`; `test_rigid_se3_phase2.py` RSE3_1-8 |
| One pose across all cameras; no Adam, no Procrustes | ✅ | Kabsch is initialisation and diagnostics only |
| Constant-twist process with Ad / J_l blocks | ✅ | `rigid_part_runtime.py:121-138` |
| Q in physical units, mapped by Ad / J_l | ❌ | **(verified)** Diagonal, σ_v̇ = σ_ω̇ = 1 per frame², dt = 1 frame (`:140-157, 340`). Prior pose σ ≥ 0.58 against a posterior σ of about 1e-4: **the process prior is inert, so Phase 2 is per-frame registration from a warm start.** 3–11 frames per run accept no step |
| Information transport under the chart change | ❌ | **(verified)** `joint_iekf.py:275-277` computes G⁻¹ΛG⁻^T; information needs G⁻^T Λ G⁻¹. Hidden today by the inert prior; it becomes live once Q is fixed |
| Anchor T_s = I with conditioned covariance | 🟡 | The mean is pinned, but the covariance is σ = 1e-2, not zero |
| Outputs {T, ξ, P}, support, NIS, gauge | 🟡 | No NIS and no direction in `summary.json`. `velocities.npy` is per frame in processing order, while `chronological_twists.npy` is per second |

### Joint identification (Sec. 7, 9, 10.5)

Entry point: `joint_reduction/rigid_trajectory_reduction.reduce_trajectory`, called inside Phase 2.

| Contract | Verdict | Where / note |
|---|---|---|
| T_k\|s, prismatic (e, s), revolute (e, a) with axial gauge | ✅ | `rigid_trajectory_reduction.py:204-353, 511-580` |
| Uncertainty-weighted pose residual (Ω₂) | 🟡 | Per-frame P_t⁻¹ only; cross-time terms dropped (disclosed only for P_ζ) |
| Manifold iterated update over (q, q̇, ϑ), posterior geometry information | ❌ | Batch LM with finite-difference Jacobians; no q̇, no F_dyn, no geometry covariance |
| Report P, R **and unresolved**, in comparable units | ❌ | Both fits are written; `decided_type` is UNDECIDED in all 20 runs. The only discriminator is raw weighted energy (E_P/E_R ratios of 1.15–2.4e5) with no dof correction. The unresolved-capable geometric fit (`joint_reduction/geometric_joint_fit.py`) is **not on this path** |
| No double counting of the same pixels | ✅ | The reduction never renders |
| Direction and endpoint-relative bookkeeping | 🟡 | θ is endpoint-relative (correct), but `trajectory.csv` labels processing indices as "frame", and Phase 2's inline GT check pairs processing-order θ with chronological GT. A sign-by-correlation step hides this |

### Phase 3F/3B: per-frame joint displacement with the joint fixed (Sec. 8, 10.6)

**Project choice.** Phase 3 must not change the joint type or geometry. That differs from the PDF's
Sec. 8, where the geometry (e, a) is part of the Phase-3 state and is "corrected directly by the
original RGB-D innovation". Here the state is (q_k, q̇_k) only, and the joint comes from
identification.

| Contract | Verdict | Where / note |
|---|---|---|
| Joint type and geometry held exactly | ✅ **new** | `estimator/joint_displacement.DisplacementOnlyState`. The runner refuses to write unless the geometry is bitwise unchanged |
| Moving Gaussians transported by the fixed joint, including orientation; statics rendered | ✅ | P: μ̄ + s e_P. R: a + Q(θ)(μ̄ − a), R = Q(θ)R̄ (`phase2_revolute.py:152-171`) |
| H = q-column of H^G J_G; VJP the exact transpose | ✅ **new** | `tests/test_joint_phase3_runner.py` [4], on both models |
| Process [[1,h],[0,1]], white acceleration on q̇ | 🟡 | `joint_process.py:42-83`; h = 1 frame, σ_q̈ = 1 by default (`--sigma-qddot`) |
| Same iterated information update | 🟡 | `joint_iekf.run_window`; the acceptance and transport defects of §2 apply. Transport is trivial here (G = I) |
| Endpoint-relative anchor (backward at T, final model) | ✅ **new** | Frames in Phase-2 processing order; q(anchor) = 0 |
| Initialised from joint identification and the frozen Phase-2 support | ✅ **new** | Before this, only `tools/run_complete_joint_debug.py` existed. It is forward-only, re-runs Phase 1 and refits the geometry |

## 3. The deviations that matter most, ranked

1. **Phase 1 is not a filter over a per-Gaussian state.**
   * Covariance is reset every frame, and the rigid-group subspace is rebuilt every frame.
   * Group-rigid errors are spatially correlated rigid errors. That is exactly what the coherence
     selector reads as "a coherent moving body". This is the documented mechanism behind the
     storage/washer forward false supports and the selector's lack of a null.
   * It also makes the free tracks tear: neighbourhood residual is 2.8× the extent on scissor, so
     orientations cannot be recovered from them.
2. **The support stage has no calibrated null and no AMBIGUOUS output.**
   * A static or barely moving forward source still yields `RESOLVED` with a "moving" component:
     storage F (precision 0.045), washer F (100% of rows).
   * The theory's AMBIGUOUS class is merged into static.
3. **No type decision and no "unresolved".**
   * Raw energies cannot separate prismatic from far-axis revolute: blade F has E_P/E_R = 2.17 in
     favour of R, with its axis 245 part radii away.
   * The geometric fit that can say "unresolved" exists but is not wired into the standard path.
4. **Inert process priors in Phase 2 (and Phase 3).**
   * ξ and P_ξ carry no information, and the documented MAP-acceptance and information-transport
     defects are latent.
   * Phase 2 still works because each frame is strongly observed. Fixing Q will make the latent
     defects live, so they must be fixed together.
5. **Centre-only Phase-1 export.**
   * Phase 2 needs only a Kabsch initial twist, so it copes.
   * The endpoint diagnostics and any warm start above about 0.8 rad cannot.
6. **Phase 3 freely refitting the geometry was harmful.** A first version left the axis free with a
   diag(1) prior in mixed units. It moved the fridge axis from 0.06° to 1.27° (axis point by 8.6 cm),
   with the geometry covariance collapsing to about 1e-10. Holding the joint fixed, as the project
   specifies, removes that failure mode.

## 4. Where our datasets fail

GT is used for scoring only. Sources:

* support precision/recall at 2 cm (as-given mapping): `metrics_support.csv`;
* joint errors (for the GT-type hypothesis) and Phase-2 pose errors: `metrics_joint.csv`;
* γ and the unexplained share: `bidirectional/<dir>_rigid/summary.json`.

| scene | dir | GT | support P / R | axis err | coord MAE | pose err mean / max | **failure point** |
|---|---|---|---|---|---|---|---|
| USB | F / B | rev 1.00 rad | 0.97/0.92, 0.99/0.94 | 0.03°, 0.02° | 0.001, 0.009 rad | ≤ 0.65 / 0.86° | OK |
| stapler | F / B | rev 1.91 | 1.00/0.83, 1.00/0.86 | 0.01°, 0.02° | ≤ 0.003 | ≤ 0.17 / 0.35° | OK |
| fridge | F / B | rev 1.77 | 1.00/0.94, 0.96/0.89 | 0.06°, 0.10° | 0.002, 0.005 | ≤ 0.51 / 0.96° | OK |
| foldchair | F / B | rev 1.87 | 1.00/0.92, 1.00/0.90 | 0.27°, 0.10° | 0.007, 0.020 | ≤ 0.38 / 0.90° | OK |
| scissor | F / B | rev 0.78 | 1.00/**0.43**, 1.00/**0.50** | 0.19°, 0.23° | 0.001, 0.009 | ≤ 0.72 / 1.77° | OK; the low recall did no harm to the joint (it did to the J1 endpoint reading) |
| blade | F | pris 0.835 m | 0.96/0.93 | 0.12° | 0.5 mm | 0.74 / 1.65° | **Joint ID:** raw energy prefers a far-axis revolute (E_P/E_R 2.17, rad/L 245) |
| blade | B | 〃 | 1.00/0.74 | 0.14° | 1.4 cm | 0.61 / 1.12° | **Joint ID:** type barely separable (ratio 1.15) |
| storage | F | pris 0.532 m | **0.045/0.16** | 31° | 0.265 m | (support never moved) | **Phase 1 / support:** no measurable motion; the owned "part" moves 2 mm; still `RESOLVED` |
| storage | B | 〃 | 0.99/0.99 | 0.04° | 4 mm | 0.13 / 0.46° | OK (weak type margin, ratio 2.79) |
| laptop | F | rev 1.72 | 1.00/0.92 | 0.08° | 0.004 | 0.27 / 0.80° | OK |
| laptop | B | 〃 | **0.87/0.64** | 2.38° | 0.117 rad | **3.95 / 10.8°** | **Support** (γ_1DOF 0.747) → Phase-2 pose error |
| oven | F | rev 1.49 | **0.49/0.19** | **24.8°** | 0.70 rad | **40.9 / 90.6°** | **Phase 1:** tracks 0.14 of 1.49 rad, so the support is wrong |
| oven | B | 〃 | 1.00/0.93 | 0.04° | 0.006 | 0.97 / 4.71° | OK |
| washer | F | rev 1.33 | vacuous (frame-0 GT has no moving samples) | 12.8° | 0.67 rad | **38.2 / 76.3°** | **Dataset → Phase 1:** the moving link has 0 sparse points at frame 0, so Phase 1 sees no motion (≤ 0.7 mm) and the support takes 100% of rows |
| washer | B | 〃 | 1.00/0.99 | 0.04° | 0.014 | 0.64 / 4.38° | OK (line 3.6 cm) |

### Patterns

* **All four failures start at Phase 1 or the support, and three of them are forward runs of
  closed-start enclosures** (oven door, storage drawer, washer door, all at q = 0 at frame 0).
  * The backward run, anchored on the open state, recovers each of them (axis 0.04°).
  * Fridge F and laptop F also start closed but succeed, so "closed start" is a risk factor, not a
    rule.
  * The common mechanism: at the closed endpoint, the moving link is barely triangulated or
    textured. In washer there are 0 of 4,594 sparse points; at frame 59 there are 489. Phase 1
    cannot lock onto it, and the support stage has no null to say so.
* **Prismatic joints are where identification is weak.**
  * The geometry of the correct type is excellent (0.04–0.14°).
  * The raw-energy margins are 1.15–2.79, against 95–2.4e5 for revolute.
* **Low support precision hurts the joint; low recall does not.**
  * Scissor and blade B keep good joints at recall 0.43–0.74.
  * Oven F (0.49), storage F (0.045) and laptop B (0.87) do not.
* **Health flags that already detect the failures without GT.**
  * The unexplained share is 0.79–1.12 for oven F, storage F and washer F, against 0.14–0.36
    elsewhere.
  * γ_1DOF < 0.91 in exactly the four failures. Laptop B is caught only by γ (its unexplained share
    is a normal 0.22).
  * These two flags could gate an "unresolved / abstain" output today.
* **Kabsch-divergence is not a failure flag.** Its largest values (stapler F 3.10, blade 2.55–3.14
  rad) occur on runs whose GT pose errors are < 0.75°. The Phase-1 Kabsch track was wrong there, and
  the RGB-D stage corrected it.

## 5. How the deviations connect to the failures

* **Oven F, storage F, washer F.**
  * Phase-1 deviations 1 and 2 (no carried covariance, rebuilt rigid groups) plus the uncalibrated
    support null (deviation 2) turn "no motion seen" into a confident wrong moving body. A filter that
    accumulates velocity information, plus a noise-calibrated null, would instead return "no
    measurable motion / unresolved".
  * This is what the theory requires. Phase 2 cannot recover, by design: it is conditional on the
    support.
* **Laptop B.**
  * The support is mixed (P 0.87) and Phase 2 fits one rigid body to it.
  * An AMBIGUOUS class kept out of the frozen core (Sec. 5: "do not silently convert a broad binary
    partition into the rigid core") would have removed the false-positive rows.
* **Blade.** No P/R/unresolved decision exists (deviation 3). The geometry is fine; the type is not
  defensibly chosen.
* **Scissor.** Low recall is harmless for the joint, but endpoint and J1 analyses must not treat the
  support as part ownership (see `docs/FB_ENDPOINT_COVARIANCE_AND_OWNERSHIP.md`).

## 6. Phase 3 status (new runner)

`tools/run_joint_phase3.py`, and the opt-in stage `phase3` of `run_bidirectional_all.sh`.

**What it does:**

* Loads the direction's endpoint model and the RGB-D in Phase-2 processing order, so a backward run
  is anchored at original frame 59 with the final model.
* Moves the frozen Phase-2 support by the **fixed** joint from `joint_reduction/<type>/geometry.json`.
* Starts q from joint identification's track.
* Runs the RGB-D IEKF on (q_k, q̇_k) only, for the type(s) given by `--hypotheses`.
* Writes `bidirectional/<dir>_joint/`. With `--evaluate-gt`, scores afterwards into `evaluation_gt/`.

Tests: `tests/test_joint_phase3_runner.py`, 12 checks:

* initialisation;
* window-length and direction refusals;
* reduced H equals the q-column of the full model, and VJP is its transpose, for P and R;
* retraction moves only q[k], and the geometry stays bitwise equal.

**Smoke on fridge** (revolute = GT type, 8 cameras, 2 iterations per frame, about 0.9 min per
direction; scored afterwards):

| | forward | backward |
|---|---|---|
| geometry (fixed) axis / line error | 0.062° / 1.3 mm | 0.104° / 3.6 mm |
| angle MAE: Phase-2 track → Phase 3 | 0.0022 → 0.0026 rad | 0.0071 → 0.0073 rad |
| angle range: Phase 2 → Phase 3 (GT 1.772) | 1.779 → 1.778 | 1.760 → 1.759 |
| largest change of q vs Phase 2 | 0.004 rad | 0.006 rad |
| image energy | 7.225e6 → 7.140e6 | 9.774e6 → 9.713e6 |

**Reading.** With the joint fixed, Phase 3 is stable and endpoint-anchored in both directions. On a
scene where Phase 2 is already accurate, re-fitting q on the RGB-D changes it by less than 0.006 rad
and does not lower the GT error. It lowers the image energy, so the residual it removes is not joint
error. Whether it helps where the Phase-2 track is noisier (laptop B, blade B) is untested.

A first version that refitted the geometry is recorded in §3 item 6.

## 7. Recommendations, in order

1. **Abstain instead of guessing (cheap, no re-run).**
   * Gate each run on γ_1DOF ≥ 0.91 (`one_dof_diagnostics.json`) and
     `support_rigidity.unexplained_fraction_final` ≤ 0.5 (`summary.json`).
   * Report failing runs as `UNRESOLVED` and fall back to the other direction's estimate.
   * Checked on all 20 runs: the gate flags exactly oven F, storage F, washer F and laptop B, and
     passes the other 16.
   * The thresholds were read off these same runs, so they are post-hoc and need a held-out check. The
     closest pass is scissor F, with γ = 0.924.
2. **Wire the type decision and "unresolved".** Put `geometric_joint_fit` (or a dof-corrected,
   held-out comparison) on the standard path, and stop reading raw E_P/E_R.
3. **Make Phase 1 a filter.**
   * Carry P_post → P⁻.
   * Pass a canonical, immutable partition, or none at all.
   * Export the full 12N state (`--full-state-out` exists).
   * Weight the support by Phase-1 uncertainty, keep AMBIGUOUS out of the frozen core, and add a
     noise-calibrated null.
   * Re-run Phase 1 + support on oven, storage and washer forward first.
4. **Fix Phase 2/3 process noise and priors together with the two `joint_iekf.py` defects.** Use
   physical dt, Q mapped through Ad/J_l, MAP-consistent acceptance, and the G⁻^T Λ G⁻¹ transport.
   Then run the fixed-joint Phase 3 on the scenes with noisier Phase-2 tracks (laptop B, blade B,
   oven B) to see whether it earns its place.
5. **Separate the GT-scoring process** from Phases 1–2: `inference_only=True` and evaluation
   afterwards, as the endpoint diagnosis already does.
6. **Data.** For closed-start scenes, add views or frames that observe the moving link at the closed
   endpoint, or treat that direction as unidentifiable and use the backward estimate.

## 8. Corrections to earlier documents

* `results_11_10_26/ours/README.txt` says washer changes only 4,454 depth pixels, with 5 of 8 cameras
  blind. On this dataset, frame 0 vs 59 changes 34,071 pixels by more than 1 cm, and all 8 cameras
  see change. Washer is still the lowest share of foreground (10.7%). Its forward failure is the
  missing moving-link sparse points at frame 0.
* `endpoint_eval_v3/ANALYSIS.md` §4 used the *kinematic* GT mapping for some forward support rows
  (USB F, oven F). The table above uses the as-given mapping throughout.
