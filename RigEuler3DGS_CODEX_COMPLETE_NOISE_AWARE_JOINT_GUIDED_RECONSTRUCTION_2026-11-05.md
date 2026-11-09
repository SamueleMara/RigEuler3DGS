# CODEX IMPLEMENTATION CONTRACT — RigEuler3DGS Phase 4 + Phase 5 (REVISED)

**Subject:** conservative joint-conditioned completion of existing Gaussians, late-visible / hidden-at-anchor surface recovery, and final **full-sequence, full-iteration** articulated 3D Gaussian Splatting.

**Prepared:** 2026-11-05. **Status:** implementation instructions and acceptance gates, **not** an implementation report. **Do not claim that Phase 4/5 is coded or validated.**

**Source specification reviewed:** `RigEuler3DGS_CODEX_COMPLETE_NOISE_AWARE_JOINT_GUIDED_RECONSTRUCTION_2026-11-05(1).md` (533 lines). This document replaces its outdated HEAD assumptions and tightens its stages D–H, especially the FreeArtGS handoff and the scientific evidence required for hidden surfaces.

**Verified repository baselines (at authoring):**

- `SamueleMara/RigEuler3DGS` main `90dbf58bc94015167e074c83a67806c923d06552` (direction `auto|forward|backward|both`). The source document's `4c88725` HEAD is **outdated**. Phase 4/5 interfaces need inspection against *current local HEAD* before editing.
- `SamueleMara/FreeArtGS` main `dffcf46bddedd6cc39e053a10e2185a1f49e6528`; its Nerfstudio implementation lives on this repository's `nerfstudio-local` branch, pinned in the working repository via `reconstruction/nerfstudio`. Check the actual **gitlink SHA** in the user's checkout, not simply branch HEAD.
- The upstream `FreeArtGS/FreeArtGS` and the user's `SamueleMara/FreeArtGS` fork are **not interchangeable** for this integration.

**Separate prerequisite document:** `RigEuler3DGS_Phase1_Covariance_Velocity_Corrections_Codex.md`. Phase 4 must not silently change the Phase 1 solver to hide the existing divergence.

---

## 0. Nonnegotiable scientific target

**Existing endpoint Gaussians:** A Phase-1 moving core is *deliberately conservative*. The rest of the original endpoint population is **not** implicitly static. Use the accurate Phase-2/3 estimated joint to recover missed-moving original Gaussians, retaining an unresolved class whenever evidence is insufficient.

**Previously unseen-at-anchor, later-visible surfaces:** Reconstruct only *observed* new surface patches from multiview RGB-D. Transform any accepted moving patch to a single canonical moving frame. It must follow the same joint for **every original frame** — before the first observation, during occlusion, and after reappearance — and must be checked against complete nonlinear joint-scene renders. Never synthesize a measured surface from geometry that is invisible in all frames.

**Final appearance:** Train a single physical, canonical two-body Gaussian model (static body and one joint-driven moving body) on the **entire chosen RGB-D motion interval** using the adapted FreeArtGS/Nerfstudio training backend for the declared **full run of iterations**. A short Phase-0 200/1000-iteration refinement is *not* final 3DGS training. The existing FreeArtGS two-half-opacity rendering and independent motion-estimation pipeline are **not** the desired final physical model.

```
Phase 0  Calibrated sensor + endpoint 3DGS                  EXISTING
Phase 1  High-confidence motion evidence                   EXISTING; currently unreliable on real long run
Phase 2  Rigid motion + revolute/prismatic hypotheses      EXISTING
Phase 3  Fixed-axis joint displacement q_t                 EXISTING

GATE Q   Qualify actual Phase-3 trajectory and support     NEW prerequisite
Phase 4A Existing-Gaussian material assignment under J     NEW
Phase 4B Genuine missing-surface proposal/fusion          NEW
Phase 4C All-frame, all-camera nonlinear acceptance        NEW
Phase 5  Full-interval articulated 3DGS, 30k-style train   NEW
```

### 0.1 Boundary conditions

1. **Do not edit Phase 1/2/3** as part of this project. Their input/output SHA and baseline results remain immutable. Fixes to their covariance/velocity issues belong to the separate correction branch/document, then requalify Phase 3.
2. **Do not accept a bad Phase-3 joint** just because it has syntactically valid `geometry.json` and `trajectory.csv`. The observed real `red_fridge` Phase-1 run diverged markedly (frame 101 p50 2.7 mm from anchor; frame 140 p50 ~308 mm; frame 288 p50 ~2.63 m; extreme outliers reached hundreds of metres). These are displacements, not necessarily GT errors, but are a serious upstream failure. See Gate Q below.
3. **All cameras** contribute to acceptance and final photometric reconstruction. `--direction auto` from `metadata/direction/summary.json` selects the **initial canonical anchor**, not a subset of the original motion frames to train on. The recommendation must match the exact frames, cameras and masks used; otherwise explicitly override `--direction` and record it.
4. Real/SAPIEN sensor models remain separate; preserve frozen calibration and zero use of synthetic GT in inference. No hard arbitrary speed caps, per-frame independent moving trajectories, or manually assigned static/moving GT labels.
5. **Binary object masks are hard measurement-validity constraints:** RGB/depth used as object evidence only on valid foreground support. Object silhouette loss may use observed foreground/background masks, but a background depth pixel is never a measurement of the foreground object. All occlusion decisions use the **single complete** physical renderer.
6. Part assignment is **material-persistent**. Every *accepted* Gaussian is physically either static or moving throughout the interval; label uncertainty is a separate inference variable, not a second ghost Gaussian.
7. **Only one estimated moving joint** in this implementation, in addition to an identity static part. Multiple joints are out of scope and must not be silently approximated.
8. `Sigma_render` (splat extent), geometry covariance, calibrated RGB-D measurement covariance, joint-axis covariance, q covariance, and uncertain membership are distinct. No covariance double counting.
9. Full nonlinear production 3DGS renderer for every *final* material reassignment and birth acceptance. IEKF JVP/VJP linearization and isolated patch render are allowed as approximations **only for proposal ranking**, never as the final acceptance score.
10. Never overwrite the raw dataset, previous stage outputs, or source FreeArtGS checkpoints. New artifacts must use content-keyed, stage-specific directories and resumable manifests.

---

## 1. Immediate source and training-backend findings (verified)

| Location | What the current code does | Implication for Phase 4/5 |
|---|---|---|
| `RigEuler3DGS/tools/run_noise_aware_rigeuler.py` | `--direction auto` chooses `recommended_direction` and creates only requested endpoint(s); passes selected direction to the driver. | Read source metadata, record its hashes and selection provenance, process original interval regardless of direction. |
| `RigEuler3DGS/tools/run_joint_phase3.py` | Runs P/R joint-coordinate filters and stores trajectories relative to the chosen endpoint and **processing order**. | Need exact original-frame mapping, anchor gauge, and parity checks before inverse-transforming any depth pixel. |
| `RigEuler3DGS/estimator/sparse_iekf.py` | Per-Gaussian block Fisher covariance and constant-velocity recursion; current real long run diverged. | Phase 4 **gated on qualified J/q**, not unconditional. |
| `FreeArtGS/pipelines/freeartgs.sh` | Standard reconstruction performs part training, two-part blend, articulation estimation, and `splatfacto-art`. | **Do not invoke the standard pipeline** for RigEuler handoff; it reruns the front-end that this project is replacing. |
| `FreeArtGS/pipelines/train_parts.sh` | Invokes `generate_ply.py`, then trains `splatfacto-depth` for both parts. | Its part masks and point generation assume upstream part segmentation; not a safe default for incomplete/unresolved materials. |
| `FreeArtGS/pipelines/splatfacto_art.sh` | Calls `preprocessing/estimate_joint.py`, then `ns-train splatfacto-art`. | **Bypass** `estimate_joint.py`: the RigEuler Phase-3 joint must be imported, and initially frozen. |
| `FreeArtGS/configs/freeart-21.yaml` | `train_parts`, `twopart_blend`, `splatfacto_art` each have `max_num_iterations: 30000`. Art stage enables axis/origin/q optimization. | Full train means explicitly configured optimizer steps; **all relevant joint parameters must be fixed by default**. |
| `FreeArtGS/nerfstudio-local:nerfstudio/models/splatfacto_art.py:_init_gaussians` | **Recomputes isotropic scales using 4-nearest neighbors and randomly initializes quaternions** instead of keeping imported anisotropic geometry. | Must implement a native import that preserves recovered means, scales, quaternions, DC/SH, and opacities. **This is a critical defect for direct handoff.** |
| `.../splatfacto_art.py:get_outputs` | Concatenates static/moving copies of **every** Gaussian, scaling their alphas by `1-w` and `w`. | A half-confidence label produces two physical-looking surfaces. Add exact **hard-part one-copy-per-Gaussian** rendering. |
| `.../splatfacto_art.py:_init_joint_parameters` | By default `optimize_joint_axis`, `optimize_joint_origin`, `optimize_joint_params`, `optimize_rotation_offset` can be trainable. | All are disabled in the baseline RigEuler handoff. Preserve `J`/`q_t` across 30k steps and every checkpoint. |
| `.../splatfacto_art.py:get_outputs` | Uses `frame_index` from camera metadata; gsplat path returns depth normalized by alpha where enabled. | Map global temporal indices (not each camera-view index) to full original frames. Validate depth units and alpha normalization. |
| `.../splatfacto_art.py` | `sh_degree == 0` uses `sigmoid(features_dc)`; 3DGS SH representations use different DC encoding. | Convert DC features correctly for the selected rendering branch, **render-parity test required**. |
| `.../splatfacto_art_dataparser.py` | Expects `twopart_blend/object_3dgs.ply`, `3dgs_part_weight.txt`, `transforms_train.json`, `estimated_joint.json`; defaults orient/center/scale the scene. | Add a **new RigEuler-specific dataparser**. Do not fake old files or silently apply similarity transforms without transforming J/geometry. |
| `.../splatfacto_art_dataparser.py` | Uses `time_index` for multiview `frame_index` where present; normalizes cameras, Gaussians, joint geometry and P prismatic q scale. | Reuse its convention conversions only after an independent world/camera/landmark parity test; prefer identity transform when possible. |
| `.../method_configs.py` | `splatfacto-art` trainer's full schedule is **30,000 iterations**, saves every 2,000, and invokes single-image training via `FullImageDatamanager`. | Create a distinct method/config with full-interval sampling, measurable coverage and immutable joint, preserving the standard baseline. |

**Codex:** verify the line/function content against checked-out commits before copying. The fork's `nerfstudio-local` is a **git submodule**, not a normal directory tracked by the FreeArtGS main commit; implement in the correct repository/branch and update pin only with explicit verification. No accidental change to third-party upstream.

---

## 2. Gate Q: qualify Phase-3 joint before any Phase 4 reconstruction

**Reason:** An inaccurate trajectory creates *coherent but false* moving-surface explanations. Full-sequence 3DGS refinement could conceal those errors by deforming geometry. Phase 4 cannot manufacture trustworthy geometry from unqualified joint motion.

### Required `recovery/joint_quality.py`

- Load Phase 2 support and P/R hypothesis results, Phase 3 original camera/time manifests and completed stage signatures. Confirm exact endpoint PLY and sensor/mask/source data hashes.
- Verify numerical finiteness, positive physical frame durations, alignment to `S..E` original dataset indices (inclusive), canonical q(anchor)=0 in the correct gauge, consistent joint type/axis conventions, monotonic **timestamps** (not necessarily monotonic q).
- Independently render only the **high-confidence moving core + current static-compatible scene** under Phase-3 J and q; compare with actual foreground RGB-D in every available camera on a short diagnostically informative window and an all-frame summary. Use full rasterization and per-camera residuals.
- Check static-region consistency and known occlusion behavior; don't let a low average RGB loss hide a catastrophic negative depth contradiction or out-of-frame Gaussian explosion.
- Report `JOINT_QUALIFIED`, `JOINT_UNRESOLVED` or `JOINT_REJECTED` with **observed evidence and uncertainty**, not fabricated numerical model confidence. Phase-3 q posterior SD does not imply calibrated axis/type probability.
- Where P and R remain indistinguishable, maintain **two separate physical hypotheses** through recovery; never blend them into one fabricated averaged transform. If neither is qualified, **stop Phase 4**.
- Use synthetic GT only in a separate scoring/evaluation command, never to qualify an inference-time hypothesis.

**Required tests:** correct nonzero-start forward/backward anchor; phase3 processing->chronological permutation; revolute axis gauge and prismatic direction units; artificially incorrect axis/q makes held-out images worse; a verified static scene rejects spurious movement; broken run rejected. Report data provenance and independent support.

**Gate Q deliverable:** `recovery/joint_quality.json`, `recovery/inputs_manifest.json`, a small set of true full-render RGB/metric-depth/alpha comparisons, and a report with `PASS/FAIL/NOT RUN`. Do **not** use the divergent full red_fridge run as a passing gate.

---

## 3. Canonical world/joint coordinate contract (Phase 4 foundation)

Use original frame IDs `t in [S,E]`, physical timestamps, camera index `c`, pinhole intrinsics `K_c`, and the measured static `T_WC,c`.

Let the joint-driven moving body have world transform `T_WM(t; J, q_t)`; the endpoint/reference anchor is `a = S` for forward, `a = E` for backward.

For a point world-observed at frame t:

```math
\mathbf y_W(c,t,u,v,d)=T_{WC,c}[d K_c^{-1}(u,v,1)^T;1].
```

Define the **relative world motion** for a point stored in anchor-world coordinates:

```math
\Delta T(t\leftarrow a) = T_{WM}(t)\,T_{WM}(a)^{-1}.
```

- Original static Gaussian: world mean `mu_i(t)=mu_i(a)`, orientation/covariance unchanged.
- Original moving Gaussian: `mu_i(t)=DeltaT(t<-a) mu_i(a)` and `R_i(t)=DeltaR(t<-a) R_i(a)`.
- Newly observed moving surface point y at birth time b: anchor-world `mu_i(a)=DeltaT(a<-b)y_W`; then `mu_i(t)=DeltaT(t<-a)mu_i(a)` for **all t**, including before b.
- Static birth candidate: `mu_i(a)=y_W`.
- Rotate **anisotropic render covariance** as `Sigma_i(t)=DeltaR Sigma_i(a) DeltaR^T`, or rotate quaternion consistently. `Sigma_render` is not the posterior position covariance.
- Same applies to Gaussian opacity/color features in their canonical material frame; no automatic per-frame appearance change or per-frame independent deformation.

**No double transform.** If importing into FreeArtGS `apply_joint_transform()`, either store the appropriate canonical coordinates and exact declared `rotation_offset`/joint parameter gauge, or use the verified `DeltaT` formula directly. Setting a nonidentity `rotation_offset` *and* pretransforming to that offset is wrong.

**Hard adapter parity gate:** use Phase-3's own kinematic functions to reproduce exported moving-core means/quaternions at 3+ frames (including the anchor). Compare rendered RGB/metric depth/alpha under both renderers, not just matrix elements. FreeArtGS uses internal quaternion order (xyzw in gsplat path) and camera CV/GL conversions; explicitly test against a non-symmetric landmark and an anisotropic splat.

Proposed files: `recovery/phase3_adapter.py`, `recovery/frame_index.py`, `recovery/articulated_kinematics.py`, `tests/test_recovery_gauge.py`.

---

## 4. Gate R: a *single physical* articulated renderer with appendable candidates

**Implement before membership or surface birth.** Reuse production `gaussian_renderer` for RigEuler acceptance (one alpha/depth-sorted pass), and implement a matched hard-material gsplat path for Phase 5. A faster approximation is allowed **only** for ranking, and must not be confused with the acceptance renderer.

### Required primitive

```python
render_articulated(scene, camera_id, original_frame_id,
                  joint_hypothesis, hard_part_ids,
                  candidate_gaussians=None, return_contributions=True)
# -> rgb [3,H,W], metric_depth [H,W], alpha [H,W]
#    per-material visible contribution [2,H,W]
#    per-candidate/group visible contribution (optional)
#    eligibility/validity and provenance
```

**Physical rendering:** concatenate each real Gaussian **once** after its assigned rigid transport. All static + moving + provisional Gaussians participate in **one** depth-sorted 3DGS alpha composite. Never create static and moving opacity copies to express uncertainty during acceptance or final default training.

**GPU support:** appending candidates must append *all* model tensors consistently (`means`, `scales`, `rotations`, `opacity`, SH/DC and stable ID/membership maps). An override of means alone may keep original fixed N and is not an append API. Provenance ID != mutable tensor row index after later densification.

**Correct depth:** the regular 3DGS rendering code and IEKF depth may use different normalizations/representations. Define exactly whether `rendered_depth_raw` is accumulated metric z, normalized metric z, or inverse-depth. Convert with alpha and valid rays as appropriate; compare against axial metric camera Z, not Euclidean ray length, unless calibration explicitly says otherwise. For a genuine inverse-depth measurement, propagate variance through `d(1/z)/dz=-1/z^2`; never compare it directly to metric z.

**Visibility:** use per-pixel visible weights `w_i=T_i alpha_i` from the same rasterizer. A Gaussian center in front of an object does not prove its **splat footprint** visibly contributes; an occluded splat may have zero effective weight. Record contributions/occlusion history separately from image residual validity.

**Renderer-parity tests (blocking):**

1. Empty candidate population = identical baseline image for same physical model/camera/anchor under chosen production renderer; float tolerances measured, not invented.
2. Identity and nonidentity P/R motion transport exact for means, anisotropic covariances and quaternions.
3. Per-part contributions sum to one-pass total alpha, including overlapping static/moving depth layers.
4. Added candidate behind an occluder leaves foreground evidence unaffected; same candidate in front changes RGB/depth/alpha.
5. True analytic depth in a tiny two-layer scene; normalized metric versus inverse-depth semantics checked.
6. Differentiability / finite differences for candidate mean, quaternion, scale, opacity and DC; gradient correctness near occlusion boundaries interpreted with rasterizer nonsmoothness in mind.
7. Correct RGB DC encoding for `sh_degree=0` sigmoid path **versus** Gaussian SH DC path, and correct opacity-logit conversion.
8. CUDA production and CPU analytic/oracle tests; source and backend parity must be measured independently.

Outputs `recovery/renderer_parity.json`, per-camera PNG RGB/depth/alpha, and canonical/physical PLY snapshots. Stop if failed.

---

## 5. Phase 4A — missed-moving original-Gaussian recovery (must precede births)

The initial moving core `C_move` is a high-confidence **seed**, not a complete moving mask. Every original Gaussian outside it has hypotheses `S` (static), `M` (moving under J), or `U` (unresolved). `U` is not a third physical motion.

### 5.1 Joint reliability must affect assignment evidence, not physical motion amplitude

Do not multiply q by a heuristic `trust` to make a mildly uncertain joint move less: such scaling produces a **new physical trajectory** unsupported by the Phase-3 estimate. Do not assume `P(J correct)=0.99` from small `posterior_sd_q`.

Instead, for a candidate component B with current other labels z_-B, compute conditional full-render evidence for **two physical hypotheses**:

```math
E_S(B;J) = \mathcal L_{\rm val}(Y\mid z_B=S,z_{-B},J),
\qquad
E_M(B;J) = \mathcal L_{\rm val}(Y\mid z_B=M,z_{-B},J).
```

If a valid global joint uncertainty representation is available, marginalize it **jointly across all Gaussians affected by it**:

```math
L_M(B)=\int p(Y_{\rm val}\mid z_B=M,J,z_{-B})\,p(J\mid Y_{\rm fit})\,dJ.
```

If not, report scores under nominal J plus **declared** sensitivity perturbations/alternative P-R models, mark `JOINT_CONFIDENCE_UNCALIBRATED`, and defer fragile decisions. A Gaussian's apparent membership confidence should weaken when moving likelihood superiority disappears under plausible joint perturbations. A fixed moving-core anchor may be retained provisionally, but inconsistent core points must be flagged for upstream audit.

### 5.2 The evaluation must compare complete scenes

- Source data: the original endpoint Gaussian population, canonical geometry and existing moving core; do **not** treat non-core = static truth.
- Deterministic connected candidate groups: use spatial/render overlap and proposal evidence to reduce compute. Avoid smoothing labels across articulation boundaries. Preserve stable original Gaussian IDs.
- For each group B, keep the rest of the physical scene fixed and render `S` versus `M` hypotheses on informative observations, then evaluate all camera/frame residuals before committing.
- Preserve actual self-occlusion, depth ordering, mask validity, RGB/appearance consistency and candidate-caused violations of existing foreground rays.
- A hinge-axis Gaussian may be physically moving yet have indistinguishable image motion; it must remain `U` if data cannot decide.
- Camera contributions and joint uncertainty are correlated. A sum over millions of pixels with only `sigma_sensor` overstates certainty under model discrepancy; use calibrated/model-aware likelihood and effective information accounting from separate Phase-1 corrections.
- Reassign **existing** Gaussian geometry before adding new Gaussians; otherwise large apparent gains from births may just duplicate a mislabeled source Gaussian.

### 5.3 Posterior/score honesty and data leakage

The original draft writes a posterior sigmoid based on `pi_i L_i(M)` etc. This is a valid conditional posterior only if `pi`, `L`, J distribution and observation dependencies are actual probabilities. Otherwise export `score_static`, `score_moving`, `sensitivity`, `status`, not a falsely calibrated `p_moving`.

Allocate validation observations **before membership searching or birth proposal**. If Phase 3 already used every pixel, no evaluation subset is strictly independent of the joint; mark `conditional_on_fitted_J`. Use spatially separated validation patches and time/camera cross-fitting to reduce reuse, and external synthetic or new captures for truly independent evaluation. Do not claim pixels from the same residual map were held out if they influenced candidate discovery.

### 5.4 Scalable but exact commitment

Use inexpensive ranking from existing `utils/gaussian_sequence_assignment.py`, `utils/kpart_counterfactual_rendering.py`, `rigid_parts/outer_render.py` after code/convention audit. Do not import their fixed thresholds, DINO/flow dependencies, soft-half-opacity renders as a physical likelihood, or approximate crop scores as the final answer.

Order: rank components -> optional coarse frames -> **exact full physical render all frames/cameras for finalist changes** -> commit one group -> update full-scene baseline -> reconsider interacting groups -> full-scene final audit. Use a stop report with rejected/ambiguous cases, not forced binary completion.

Outputs:

```
recovery/support_completion/
  original_ids.npz              # exactly one ID per endpoint Gaussian
  material_status.npz           # STATIC / MOVING / UNRESOLVED
  per_component_evidence.json   # fit+validation scores, camera/time evidence
  joint_sensitivity.json        # J alternatives and uncertainty qualification
  visibility_summary.npz        # actual visible weights by time/camera
  scene_state.json              # hard physical label configuration
  unresolved.npz
  renders/                     # before/after static and moving alternatives
```

**Gate A:** synthetic revolute and prismatic scenes recover initially missed moving Gaussians without forcing near-axis or unobserved ones; true static wall stays static; wrong label hurts held-out full render; initial core unchanged unless explicitly flagged; IDs stable; all frames checked. Real static control must not acquire widespread false moving labels.

---

## 6. Phase 4B — recover *observed* geometry hidden at the anchor

### 6.1 What can and cannot be recovered

A drawer side wall invisible when closed but later recorded with valid depth can be **measured** and placed into a canonical moving frame. It should later render behind the front when the drawer is closed. A wall invisible in every camera at every frame cannot be recovered or accuracy-certified from this RGB-D sequence alone. The latter requires a separate explicit prior/model-completion method and belongs to a different task.

New candidates come **only after** the original-Gaussian material reassignment. Render the newly completed original model over **all** frames/cameras. Extract foreground discrepancies and true coverage gaps.

### 6.2 Pixel eligibility and explanation taxonomy

At every original frame t, camera c and pixel p, keep `RGB`, `D`, `M`, valid-depth, predicted `alpha`, RGB/metric depth, per-material visible weights, and sensor + model discrepancy metadata.

Potential classes:

- `EXPLAINED`: current scene adequately accounts for valid foreground.
- `COVERAGE_GAP`: valid observed object depth and insufficient predicted occupancy.
- `FRONT_SURFACE`: observed surface nearer than current prediction; could be new surface **or** current pose/calibration/model error, not automatic birth.
- `BEHIND_PREDICTION`: observed depth deeper than a currently rendered surface; occlusion / model issue, no automatic birth.
- `MIXED_EDGE`: likely mixed pixels/silhouette discontinuity; don't fit a naïve 3D point to a blended depth.
- `DEPTH_INVALID`: no depth geometry proposal, even when RGB/mask valid.
- `OUT_OF_VIEW`: neither supporting nor contradictory.

Use the calibrated noise and **independent rendering/model discrepancy** to score candidate eligibility; no universal fixed `|depth_diff| > 5mm` rule. Persistent model error near depth edges cannot be repaired simply by adding splats.

### 6.3 Canonical fusion under two material hypotheses

For each valid 3D observation `y_W(c,t,p)`, propose:

```math
x_{S}=y_W, \qquad x_{M}=\Delta T(a\leftarrow t)\,y_W.
```

The hypothesized joint is fixed for the first implementation; transform depth ray covariance into the canonical frame. A depth pixel has a **ray-directional** covariance (plus documented camera geometric uncertainty), not arbitrary isotropic 3D uncertainty. Uncertainty in q/J is a **shared latent variable**, so repeating correlated frame samples cannot shrink it as if every sample had independent q error.

Build coherent locally connected patches across cameras and times; associate by canonical geometry and visibility under the candidate label. Avoid mixing front/back surfaces of thin structures. The known `surface_bootstrap.fit_surface` 4-mm two-face test failed badly (71% of samples moved >1 mm toward other face in one reported setup); **do not reuse its unmodified kNN across opposite depth layers**. Require layer-aware, viewpoint/visibility-consistent associations and a synthetic thin-panel oracle before adopting it for births.

Candidate patch parameters: canonical `mu`, anisotropic covariance/scale, quaternion, opacity, DC color, optional higher-order appearance only during later reconstruction, label `S` or `M`, sources `(c,t,p)`, provisional score/uncertainty and original observed support. No one-Gaussian-per-pixel explosion; no permanent birth yet.

### 6.4 Late-birth material consistency and early occlusion

A birth at frame b is **not** a physical creation event. For moving geometry, its canonical mean is computed at b and its predicted pose exists at all t in [S,E]:

```math
\mu_B(t) = \Delta T(t\leftarrow b)\,\mu_B(b).
```

This must be tested for all frames. At early frames the new patch can be fully occluded by known geometry, so early missing visibility is **not** negative evidence. Conversely, if the candidate appears in front of measured foreground, it is potentially contradictory even when it beautifully fits frame b. Use one depth-sorted combined render, not Gaussian-centre depth tests.

Export:

```
recovery/births/
  unexplained_observations.npz    # c,t,p,d, RGB, status, sensor uncertainty
  proposed_patches.npz           # canonical geometry, label alternatives
  birth_provenance.json          # source frames/cameras/pixels + hashes
  layer_association.json
  unresolved_patches.npz
```

**Gate B:** synthetic closed->open drawer: a late-visible side wall is recovered in a common canonical moving frame from multiple views, hidden when closed, visible when opened, with no duplicates or boundary layer merge; a never-seen patch remains unmeasured. Include late-visible revolute rear-side test and wrong-joint fixture.

---

## 7. Phase 4C — stringent full-interval nonlinear candidate acceptance

For candidate component B and hard physical material label H in {S,M}, compare **complete** model images:

```math
\mathcal L_{\rm seq}(G,H) =
  \sum_{t=S}^{E}\sum_{c=1}^{C}
  \mathcal L_{ct}\big(\widehat Y_{ct}(G,H),Y_{ct}\big)
  +\mathcal R_{\rm shape}(G)+\mathcal R_{\rm complexity}(G).
```

All renders use a common depth-sorted Gaussian population, with calibrated RGB/depth channels and an explicitly specified mask/alpha objective. `M==0` never creates a depth observation. `D` invalid never creates depth information. Off-mask object opacity and in-mask missing rendered coverage are tracked as silhouette evidence. **Do not count the silhouette objective as a second independent mask likelihood when it is mathematically induced by the same RGB expected-background objective.** Keep its separate interpretation and weight consistent with the corrected measurement-model policy.

### 7.1 Train/validation discipline and acceptance criterion

Before candidate extraction, declare data roles, e.g. disjoint spatial validation patches / cameras / time blocks. Where candidate discovery has already considered all data, say **validation is conditional**, not independent. Keep a separate untouched evaluation set when scientifically feasible. A patch seen only briefly should be evaluated using held-out pixels/camera views **within its informative frames**; it must not be rejected for having zero evidence in fully occluded frames.

Score both `G_without_B` and `G_with_B(H)` with exact renderer on **all** original frames/cameras. Record train NLL, held-out NLL, model complexity change, joint sensitivity, and source sample overlap. A lower training loss alone is insufficient for acceptance. New opacity can trivially reduce image error by duplicating existing geometry, so include redundancy and held-out coverage tests.

Use conditional acceptance decisions with explicit evidence and uncertainty, not arbitrary thresholds masquerading as calibrated probability. Allowed statuses:

- `ACCEPTED_MOVING`, `ACCEPTED_STATIC`
- `AMBIGUOUS_MATERIAL`, `PROVISIONAL_INSUFFICIENT_EVIDENCE`
- `REJECTED_DUPLICATE`, `REJECTED_VISIBLE_CONTRADICTION`
- `JOINT_UNRESOLVED`, `INVALID_OBSERVATION`.

### 7.2 Full-sequence validation is mandatory for each accepted finalist

For every accepted candidate/group report **for every frame**: out-of-frustum, occluded, partially visible, visibly supported, visibility contradiction, and invalid-depth pixels. Always run full nonlinear, single-pass compositing; the IEKF derivative cannot decide occlusion ordering across a large motion. A candidate may be visible for only 5% of frames and accepted if those independent measurements support it and all the others do not contradict it.

When proposals interact via occlusion, refit/evaluate as groups. After any accepted group modifies the scene, re-evaluate previously provisional/ambiguous neighbors; finally rerender the **joint accepted scene** across all cameras/times (not just individual marginal gains). A final multi-candidate model can violate observations even if each candidate separately looked good.

**Performance engineering:** broad-phase conservative projected splat footprints and affected-pixel lists may rank candidates cheaply, but final scoring uses full physical render on every frame/camera. Process in GPU memory-bounded chunks, stream frames on CPU, cache only invariant metadata, and record peak VRAM/runtime. Do not store T×N×3 states persistently if avoidable. Do not silently reduce interval, camera count, or render resolution to claim Gate C passed.

### 7.3 Frozen-joint canonical optimization

Optimize provisional patches only in canonical coordinates (mu, orientation, scale, opacity, DC), transported under fixed J/q and static identity. This is joint-conditioned **geometry reconstruction**, not free 6D tracking. Full-scene losses include RGB, sensor/model-aware metric depth, silhouette and physically motivated geometry regularization. Explicit `0.5 log det R` terms if covariance itself is trained; never let model variance inflate arbitrarily to explain errors. No learned arbitrary per-frame motion.

**Gate C:** wrong-side-of-door candidate visibly occludes earlier measured RGB-D and is rejected; properly hidden late wall is accepted; full render and linear approximation give demonstrably different results at occlusion boundaries; existing original Gaussian count remains unchanged; final accepted scene has no net worsening of held-out normalized likelihood/geometry coverage. Report true number of evaluated full-frame renders.

---

## 8. Phase 5 architecture — true full-iteration articulated Gaussian training

**This is the main addition to the uploaded plan.** Do not simply invoke `FreeArtGS/pipelines/freeartgs.sh`, `train_parts.sh` or `splatfacto_art.sh` with the recovery directory: those scripts run their own tracking/part masks/`estimate_joint.py` and rely on a model with duplicated soft-opacity surfaces. The desired baseline is a *new explicit* RigEuler-to-FreeArtGS training path with one physical Gaussian per accepted surface.

### 8.1 What is being trained

Canonical parameters:

```math
\Theta=\{G_S^0,G_M^0\},\quad
G_t=G_S^0\cup\Delta T(t\leftarrow a;J,q_t)\,G_M^0.
```

`G_S^0`: accepted static originals + accepted static births. `G_M^0`: accepted moving originals + accepted moving births. Unresolved originals/births are **excluded or preserved as explicitly unresolved artifacts**, not silently force-assigned; coverage limitations must be reported.

All training frames sample the **same canonical** Gaussian parameters; there is no per-frame Gaussian pose parameter. World poses follow fixed joint motion. The physical hard-part IDs are persistent. Trainable initial parameters: canonical means, **existing anisotropic scales**, existing quaternions, opacity, DC appearance, optionally SH appearance; each new Gaussian is initialized from its Phase-4 accepted canonical estimate, not a random re-fit.

**Required initial freeze:** joint type, axis, origin, all q_t, any rotation offset, and calibrated extrinsics/intrinsics are non-trainable. Optional q-only and camera pose refinement are later **separate A/B experiments**, never on by default. This prevents the backend silently absorbing wrong births by changing joint geometry.

### 8.2 Correct FreeArtGS integration — new model+dataparser, no fake files

Implement in the *pinned `SamueleMara/FreeArtGS` Nerfstudio fork* new opt-in classes, for example:

```
nerfstudio/data/dataparsers/rigeuler_articulated_dataparser.py
nerfstudio/models/rigeuler_articulated.py
nerfstudio/configs/method_configs.py              # register new method only
```

Keep the existing `splatfacto-art` baseline untouched. The new parser reads the **RigEuler recovery manifest directly**: canonical PLYs/parameters, per-Gaussian hard part IDs, fixed J and full original q_t, calibrated camera intrinsics/extrinsics, color/depth/masks and original frame indices, without relying on `twopart_blend`, `3dgs_part_weight.txt`, `estimated_joint.json`, or synthetic per-part RGB-D masks. It may reuse the existing parser's CV/GL mapping and scale conversion **only after test**. Prefer an identity world normalization for the first version; if normalizing, apply one invertible similarity consistently to every Gaussian, camera, axis, origin, prismatic distance, and covariance, and export the inverse transform.

New model `get_outputs`:

1. `frame_idx` is derived by an explicit mapping `original_frame_id -> temporal_index`, **not** by enumerating `view_index`. Three cameras at one time share the same q index.
2. Select static Gaussians once and moving Gaussians once from hard part IDs. Transport only moving Gaussians using the exact Phase-3 relative world transform at that time.
3. Concatenate them exactly **once**; no `alpha_S=(1-w)alpha`, `alpha_M=w alpha` duplicated splats.
4. Render in one depth-sorted gsplat call. Preserve grad flow to canonical Gaussian means, orientation, anisotropic scale, opacity and color.
5. Respect hard object foreground validity for RGB/depth and separately score silhouette against mask. Do not use background depth to move foreground Gaussians. Handle invalid depth and measured metric z consistently.
6. Preserve original Gaussian count and IDs at step 0. Later densification/pruning is explicitly the **Phase-5-only** lifecycle, with stable ancestry and inherited material part IDs. Never reintroduce a second motion trajectory.
7. Export canonical and per-frame PLYs, not only a normalized GS checkpoint that cannot be mapped back to RigEuler world coordinates.

**Critical initialization fixes:** Existing `SplatfactoArtModel._init_gaussians` recomputes isotropic scales from neighbors and draws random quaternions. This must **not** happen for RigEuler input. The new model must ingest source `means/scales/quats/opacities/features_dc/features_rest` with explicit conversions. `sh_degree==0` may use `sigmoid(features_dc)` rather than SH DC `(rgb-.5)/C0`; prove identical initial rendered colors. Standard 3DGS opacity is often stored as a logit: decode/re-encode only once. Test quaternion order conversion `wxyz<->xyzw` and covariance rotation.

**Densification:** Use the existing Nerfstudio optimizer state migration/callback contract, but ensure splitting a moving Gaussian produces **moving** offspring in the canonical frame (same rigid body), preserving material ID, ancestry, and SH/opacity convention. Culling must not delete correct late-visible surfaces simply because they are absent/occluded in a currently sampled early frame. Test with a wall visible in only a small part of the interval. Track counts by part and source class; do not densify originals during Phase 4.

### 8.3 Full sequence means all original motion frames, not just sampled discovery frames

Declare `frames_train = [S,S+1,...,E]` with full original timestamps and all selected synchronized cameras. Every view `(t,c)` has its own RGB, depth and whole-object foreground mask; each view maps to a unique camera and shared temporal q index. Record exact number of time steps, cameras, views, valid foreground pixels, and image dimensions. Fail on gaps unless explicitly documented by an observation manifest; don't reinterpret absent frame indices as a different time sequence.

A **training iteration** normally samples one or a few views; **30,000 iterations != 30,000 full sequence epochs**. The final training runner must log how often each `(time, camera)` and motion/visibility regime was sampled. Use deterministic balanced view scheduling (or a declared probabilistic scheduler with verified empirical coverage) so rare late-visible surfaces are not starved and no moving state is ignored. Do not apply arbitrary speed/motion thresholds; choose balance strata from observed visibility/information and record the sampling probabilities for unbiased weighted losses if necessary.

For validation, reserve disjoint observed patches or held-out camera/time observations **before hyperparameter tuning**. A released final all-view fit may retrain on all data using fixed chosen settings, but must report that it is no longer independently validated on those views. Preserve a separate untouched evaluation set or synthetic GT controls for final claims.

### 8.4 Full optimization budget and controlled curriculum

Declare an explicit `--max-num-iterations` (default **30000** to match the current FreeArtGS `splatfacto-art` reference schedule), an actual optimizer step counter, checkpoint cadence, warmup, geometry refinement enable/disable and total number of views. Do not replace 30k with 200/1000 bootstrap iterations or stop early and label the checkpoint "full training".

Recommended *configurable* curriculum within the **same declared total step budget**:

- **Initial identity/parity gate (no gradient):** exact inherited canonical model reproduces every test render under fixed joint; skip training entirely if it fails.
- **Early stable initialization:** train appearance/opacity and modest canonical geometry with frozen joint, no densification at first. Do not arbitrarily reset imported geometry.
- **Articulated joint-consistent refinement:** train canonical means, rotations, scales, opacity, DC then optional SH, over balanced full-interval views. Full renderer mandatory; gradients aggregate to canonical parameters. No per-frame spatial free corrections.
- **Explicit Phase-5 densification/culling:** enabled only by separate flag, maintains hard part ID and ancestry; evaluate rare-visible patches before pruning. Do not run it as an undocumented default.
- **Finish/fidelity pass:** freeze topology if appropriate, optimize canonical appearance/geometry under the same fixed J. Retain state at last iteration, selected best validation checkpoint, and initial step-0 model separately.

These are **training phases within a 30k (or user-declared) run**, not extra 30k per part plus 30k final unless explicitly requested and costed. Record actual transition steps and configs. Do not invent a schedule (e.g. 5k/15k/10k) and claim it is validated; experiment on short controls before selecting a production schedule.

The existing FreeArtGS code's baseline `max_num_iterations=30000`, `steps_per_save=2000`, `steps_per_eval_all_images=1000` provides a useful reference. An eval interval of 1000 **does not prove** all original cameras/times are included: the new parser and evaluator must enforce coverage.

### 8.5 Loss definition and measurement fidelity

Use a declared objective with normalization by valid observations/sampling probabilities and clear per-sensor channels:

```math
\mathcal L_{train}(\Theta)=\sum_{(t,c)\in\mathcal B}
 w_{tc}\Big[\mathcal L_{RGB}^{\rm fg}
 +\mathcal L_{D}^{\rm valid\,fg}
 +\mathcal L_{M}^{\rm silhouette}\Big]
 +\mathcal R_{geometry}(\Theta)+\mathcal R_{complexity}(\Theta).
```

- RGB only on valid object foreground for the object appearance objective. If retaining the existing artificial-background expected-RGB objective, do not separately count its *derived* silhouette term twice.
- Metric depth only on finite positive foreground depth with predicted valid visibility; independently test rasterizer depth/alpha normalization, inverse depth versus metric depth and image-space mixed-depth edges.
- Silhouette loss on whole binary object mask (including off-object negative rays) with a declared mask-model interpretation, not an unjustified copy of tiny calibrated RGB sigma into an independent fake sensor.
- Calibrated per-camera noise used where meaningful, with a separately declared and tested renderer/model discrepancy term. Do not interpret 0.7-mm temporal depth sensor variance as proof that a 20-mm 3DGS reconstruction mismatch is a 28-sigma Gaussian motion measurement.
- If using a robust Student-t likelihood, use its actual NLL and correct score/curvature; mark Gauss-Newton/Fisher approximations as such.
- Regularize anisotropy/thickness where justified by surface evidence. Keep `Sigma_render` distinct from motion/geometry uncertainty.
- Do not let learned per-frame camera offsets or per-frame color changes explain dynamic part motion in the default baseline.

Save per-channel train/val **NLL or declared surrogate**, RGB PSNR/SSIM, valid-depth MAE/RMSE, silhouette IoU/offmask opacity, invisible/contradictory rendered regions, and performance per camera and time region. Global mean PSNR alone does not establish correct joint or recovered hidden geometry.

### 8.6 Resume, completeness, and result integrity

The Phase-5 train output must contain:

```
training/
  input_manifest.json          # source files/cameras/masks/J/q/artifacts/submodule SHAs
  config_resolved.yaml         # every option, step schedule, random seed
  train_view_manifest.json     # ORIGINAL (t,c) list and t->q mapping
  initial_model/               # exact accepted canonical geometry, step 0
  checkpoints/                # iterations + model + optimizer/scheduler/RNG state
  best_validation/             # selection rule and metric; no overwrite of last
  final_iteration/             # checkpoint at explicitly requested full step
  full_sequence_eval/          # all frames/cameras RGB/depth/alpha plus metrics
  canonical_static.ply
  canonical_moving.ply
  material_ancestry.npz         # old ID -> split descendants, part labels, birth source
  joint_fixed.json             # original J and q; checksum pre/post training
  completion_report.md         # step counts, failures, evidence, runtime/VRAM
```

Exactly reproducible resume should restore all trainable parameters, optimizer moments, scheduler position, dataloader sampler/RNG, SH schedule, topology/densification state and original-world transform. Merely loading last PLY then starting a new optimizer is a **warm start**, not true resume.

Stage signatures must include input content hashes, **selected direction metadata**, endpoint PLY, Phase-3 J/q trajectories, sensor model, source image/masks/depth, camera model, FreeArtGS fork commit and pinned Nerfstudio gitlink SHA, chosen loss/scheduler and total iteration count. Refuse stale or partial reuse; do not mutate prior run manifests on refusal. For large data, a written reusable hash manifest is acceptable if every input has actually been content-verified.

**Gate 5:** a tiny 2-frame/2-camera fixture trains to the declared step count, survives interrupted resume with parity to uninterrupted run within documented nondeterminism, preserves fixed joint, and handles one late-visible moving Gaussian through densification. Then a short multi-frame synthetic run and one real pilot. Only after these pass launch the **complete 30,000-iteration all-frame** run; measure true elapsed steps, view exposure and final full-sequence performance.

---

## 9. Optional refinements and ablations — never silently default

1. **Phase-3 q-only refinement** with fixed joint type and axis/origin; use original physical timestamps and separate validation. Must test whether q changes compensate model mismatch, and retain the original-q result as baseline.
2. **Joint axis/type correction** is a separate downstream experiment, *not* a Phase-5 reconstruction default. Competing P/R hypotheses train as separate physical models until independent evidence distinguishes them.
3. **Camera pose refinement** is an ablation. Multi-camera static calibrated rigs have known extrinsics; free per-view camera SE(3) can absorb the moving object's motion and fabricate improved photometric fit.
4. **Soft material assignment** is an optimization surrogate only, with physical one-copy hard-render evaluation. Default final scene uses accepted hard S/M, unresolved separate.
5. **Geometrically never-seen surface completion** is out of scope. A prior-generated guess must be labelled as hypothetical, not RGB-D recovered.
6. **Post-training reassignment or new births** may be studied separately; they invalidate Phase-4 acceptance and require a new full-sequence physical consistency gate.

---

## 10. Codex incremental gates and exact ownership

| Gate | Minimal edited/additional code | Required independent evidence | Stop if |
|---|---|---|---|
| `Q` | Phase3 adapter, quality audit, manifests | Original index/gauge/parity; static/moving full-render evidence; synthetic J oracle | joint not qualified or wrong provenance |
| `R` | New appended physical RigEuler renderer, alpha/depth/visibility adapter | Identity + P/R + overlapping depth-layer + gradient + DC parity | duplicate/incorrect contribution or depth convention |
| `A` | Persistent original-Gaussian material assignment and counterfactual scorer | Moving missed support recovered; true static protected; ambiguous retained | GT leaks into inference or likelihood uses linearization |
| `B` | Foreground candidate extraction, layer-aware canonical fusion | Late drawer wall and revolute back surface; zero invalid births; thin-panel no merge | geometry hallucinated/duplicate or mixed layers |
| `C` | All-frame candidate acceptance, full physical renders | Earlier-occlusion/no-visibility is neutral; wrong visible occluder rejected; global consistency | finalist accepted without full original interval |
| `5a` | New RigEuler dataparser, hard single-copy gsplat renderer, source Gaussian initialization | Same frame zero render + anisotropic PLY, correct time map, no joint drift | random quats, scale reset, ghost surfaces or camera normalization mismatch |
| `5b` | Fixed-joint full training scheduler, losses, checkpoint/resume | Tiny deterministic interrupted resume and balanced view coverage | 30k mistaken for full-frame epochs, missing views, incomplete resume |
| `5c` | Full 30k-style training + evaluation | All frames/cameras trained/evaluated, metrics/VRAM/time, checkpoints and baselines | no upstream quality gate, stage mismatch, unverified claims |

**Suggested opt-in new command surfaces** (proposed, NOT currently implemented; Codex must publish `--help` from actual code):

```
python tools/run_joint_surface_recovery.py \
  --source <scene> --endpoint-model <endpoint> --phase2-dir <phase2> \
  --phase3-dir <phase3> --direction forward|backward \
  --motion-start-frame S --motion-end-frame E \
  --hypothesis prismatic|revolute \
  --stages quality renderer support birth acceptance \
  --out <new_recovery_dir>

python tools/export_rigeuler_articulated_dataset.py \
  --recovery <accepted_recovery_dir> --out <new_export_dir>

# Proposed new wrapper around the *pinned* FreeArtGS fork's registered method:
python tools/train_rigeuler_articulated.py \
  --input <new_export_dir> --max-num-iterations 30000 \
  --freeze-joint --hard-material --full-interval \
  --out <new_training_dir>
```

The wrapper may live in FreeArtGS, RigEuler, or an integration package; it **must** call the actual pinned `nerfstudio-local` training API rather than fabricate an `ns-train` invocation that has not been registered/tested. Never add a false `--freeze-joint` flag without implementing it. A production long run starts only after verified `--help`, a 2-view end-to-end fixture, and at least one short synthetic multi-frame success.

### Implementation tests (minimum fixture set)

- Nonzero-start frames, forward/backward, 3 synchronized cameras, original IDs and physical timestamps.
- Prismatic drawer closed->open with side wall invisible at anchor; moving wall transforms back under joint and remains behind drawer at early times.
- Revolute blade/door with near-hinge ambiguous samples, late-visible back side, and static nearby wall.
- Exact static null; wrong axis/q model; two-layer 4-mm sheet; depth holes and mixed pixels.
- Candidate wrong-side occlusion; candidate duplicating existing Gaussian; silhouette leakage.
- Synthetic known uncertainty with shared q nuisance; independent sampled sensor noise versus persistent model bias.
- Renderer parity for anisotropic covariance, quaternion, RGB SH/DC, opacity, alpha and metric/inverse depth.
- Training on 2 cameras × 3 times (6 views) with 30 optimizer steps (small fixture **only**), stable q and part labels and view exposure; later full 30k actual run.
- FreeArtGS original backend versus new hard-part backend with **the same canonical input geometry and J**, to isolate renderer/initialization differences.
- Baselines: RigEuler core only; core+support completion; +accepted births; +short Phase5; +full Phase5; FreeArtGS matched-backend standard baseline when feasible.

### Report every gate

`PASS | FAIL | NOT RUN`, command, actual checkout SHA, generated artifacts/figures, random seed, frame/camera set, input hashes, quantitative metrics, source edits, runtime, peak VRAM, independent synthetic GT scoring after inference, unresolved hypotheses, and theory/implementation deviations. Unit tests pass ≠ scientific accuracy pass. No overnight full run until gates Q/R/A/B/C/5a/5b succeed.

---

## 11. Deviations, branch discipline, and definition of completion

Codex must create/update:

- `docs/PHASE4_REUSE_COMPATIBILITY.md`: current checkout interface and FreeArtGS fork/submodule parity (including no masked part GT inference).
- `docs/PHASE4_THEORY_CODE_PARITY.md`: each exact/approximate equation and its code path; declaration of correlated-pixel, joint and Gaussian covariance approximations.
- `docs/PHASE5_NERFSTUDIO_INTEGRATION.md`: exact parser/model files, color/opacity/quaternion/depth conversion, scene normalization and full-iteration trainer config.
- `reports/phase4/deviation_register.json` with categories `EXACT`, `NUMERIC_EQUIVALENT`, `MODEL_APPROX`, `THEORY_CHANGE`, `NOT_IMPLEMENTED`.
- `reports/phase5/completion_report.md`: training step totals, exposed view IDs, errors, rendered video/figures, baseline comparisons, hardware/runtime, export correctness, all unresolved items.

**Code changes are opt-in and isolated.** Do not edit Phase 1/2/3 internals, original FreeArtGS `splatfacto-art` model, or existing baseline result folders in this scope. If editing the FreeArtGS fork's Nerfstudio submodule, commit there, verify upstream and checked-out submodule SHAs, then deliberately update the parent gitlink. No imaginary push/test successes. A changed pinned submodule must have a documented model/data parser parity test.

**Done means all three—not just code existence:**

1. Initially missed moving original Gaussians have been conservatively reclassified under an independently qualified joint, without forcing unidentifiable hinge/static points to a false label.
2. Late-observed surfaces have been reconstructed in a canonical frame and pass full nonlinear **all-frame/all-camera** rendering, including correct earlier occlusion; genuinely never-seen geometry is not claimed as recovered.
3. One physical canonical articulated Gaussian model has been initialized **without losing recovered anisotropy/rotations**, trained for the **declared complete iteration budget** on the complete motion interval, with fixed-joint world transforms, genuinely recorded view coverage, valid full-sequence evaluation and reproducible checkpoint/resume.

**If the Phase-3 joint gate fails, do not proceed to births or long full-sequence training.** Report the failure and return to the separately scoped Phase-1 covariance/velocity fixes. This is a scientific blocker, not a reason to weaken the acceptance requirements.
