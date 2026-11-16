# RigEuler3DGS — Unified Codex implementation specification: Phase 4 and Phase 5

**ONE Markdown document; SEPARATE phases.** Target worktree: `~/Joint3DGS/RigEuler3DGS-p45` (`phase45`). This file contains the full mathematical and implementation contract for the Phase-4 recovery pipeline and the later Phase-5 training pipeline. Combining specifications into this one file does **not** authorize merging stage logic, optimizers, or outputs.

## Non-negotiable stage boundaries

| Stage | What it may change | What it must not change | Required standalone output |
|---|---|---|---|
| **4A — original-Gaussian material recovery** | Static/moving/unresolved labels for the existing `N0` Gaussians | Canonical means, scale, orientation, opacity, SH, Gaussian count, joint or trajectory | Immutable `4a_original_membership/` result with original-row provenance and completed full-sequence material audit |
| **4B — new-surface discovery** | Provisional observation classifications, canonical surface patches and their provenance | Original labels/geometry, accepted population, joint or trajectory | Separate `4b_new_surface_proposals/` result; **zero permanent births** |
| **4C — candidate validation and committed births** | Only proven new moving-patch Gaussian parameters, lineage and accepted count, through local candidate-only refinement | Original Gaussian geometry/labels, joint or trajectory; no global 3DGS training | Separate `4c_accepted_moving/` complete canonical result and rejection/visibility audit |
| **5 — photorealistic training** | Gaussian parameters under the explicitly selected training mode; densification only as allowed by that mode | Material assignments, joint type/geometry/trajectory, camera calibration; no renewed Phase-4 discovery | New `phase5/` training outputs plus `joint_unchanged_proof.json` |

**Execution sequence**: `4A -> validate/checkpoint -> 4B -> validate/checkpoint -> 4C -> validate/checkpoint -> (STOP)`. Phase 5 is a **separate invocation only after explicit authorization**. In particular, `tools/run_phase4_recovery.py --stage all` means `4A -> 4B -> 4C`, **never Phase 5**. No 4B birth inside 4A, no automatic original-label changes inside 4B/4C, no global appearance training during 4A/4B/4C, and no joint correction in any stage.

**Source of truth:** the actual worktree takes precedence over file names listed as planned APIs; audit it first. Reuse working CUDA rendering and tests. The previously reported procedural results and failures remain reports, not independent proof of real-scene correctness. Do not launch full overnight jobs before stage gates pass.

---

# PART I — PHASE 4: GEOMETRY RECOVERY (4A, THEN 4B, THEN 4C)

**Within Phase 4, each substage is independent. The complete original-material reassignment of 4A must finish before surface discovery begins.**

**Target worktree:** `~/Joint3DGS/RigEuler3DGS-p45` (branch `phase45`; confirm actual state and diff before edits).  
**Upstream base in supplied audit:** `6807d834c0e10ca6eff7cbb2d6255307ff3b1dca`.  
**Scope of this section:** Phase **4A**, then **4B**, then **4C**, each with a distinct program contract, immutable artifact, and validation gate.  
**Out of scope for Phase-4 execution:** Phase 5 photorealistic training (specified later in THIS document); adjustment of Phase-3 joint type, axis, origin, trajectory, camera calibration; additional forward/backward bootstrap; unrestricted per-point motion; reimplementation in Nerfstudio/FreeArt.  
**Non-negotiable separation:** Phase 4A modifies **only material labels of ORIGINAL Gaussians**, Phase 4B proposes **only NEW geometry**, and Phase 4C validates/commits new geometry. Do not jointly optimize these, run births during 4A, or silently change original Gaussian poses during 4B/4C. Explicit targeted reconsideration of *unresolved* original labels is a separately logged **return to 4A**, followed by a new immutable 4A snapshot, before resuming births.

## 0. Why this is required

The source 3DGS has incomplete moving support. Phases 1–3 intentionally use conservative moving anchors; original unselected Gaussians are **not proven static**. Phase 4 first recovers all justifiable moving membership among existing endpoint Gaussians. Only then does it search all original RGB-D observations for actual missing **moving surfaces**, beginning at the geometric frontier of the recovered moving support.

The previous implementation accepted depth-edge mixture artifacts as static geometry (19/80 false additions on a synthetic drawer), merged distinct layers, and had limited surface completion. Its old 4A also attempted thousands of per-component all-view renders and repeatedly decoded 378 images with a 64-view LRU, leading to ~13 hours with no complete red-fridge artifact. Existing 8-view red-fridge birth results (4 patches, 38 moving Gaussians) are **diagnostic only**, not a validated full-sequence result. These are reported findings, not independent verification in this document.

**The inference order is contractual:**

```
Read Phase-3 inputs (fixed, single direction)
  |
  v
[PHASE 4A: ORIGINAL GAUSSIANS ONLY]
  Compare STATIC vs JOINT-MOVING (and diagnose OMIT)
  Include every original Gaussian; retain undecidable cases
  Emit immutable original-material snapshot and complete audit
  |
  v             4A completion/checkpoint gate
[PHASE 4B: NEW MOVING SURFACE PROPOSALS ONLY]
  Scan original RGB-D sequence, start at trusted moving frontier
  Separate ordinary residuals / depth mixtures / missing surfaces
  Joint-inverse transport measured surface patches to canonical frame
  Preserve depth layers and provenance; NO accepted births yet
  |
  v
[PHASE 4C: NEW PATCH ACCEPTANCE ONLY]
  Compare ABSENT, MOVING, STATIC *as explanations* using real renders
  Geometric evidence + all-frame visible consistency + validation
  Commit supported MOVING births; do not auto-accept STATIC births
  Update moving frontier; rescan; re-audit joint accepted scene
  |
  v
Validated Phase-4 canonical scene + complete reports

[PHASE 5] SEPARATE PROGRAM / LATER SECTION OF THIS DOCUMENT; NOT CALLED BY PHASE 4.
```

## 1. Repository audit before editing

Inspect current uncommitted files, not merely the earlier report. At minimum:

- `recovery/contracts.py`, `recovery/views.py`, `recovery/io_setup.py`, `recovery/articulated_scene.py`, `recovery/gaussians.py`, `recovery/rgbd_loss.py`, `recovery/phase4.py`, `recovery/topology.py`.
- Phase-4A implementation currently present (`recovery/material_completion.py` and/or more recent files).
- **Current** Phase-4B/4C entry points, including reported `phase4b_exact.py` / `phase4c_exact.py` (search the tree; do not assume the old `residual_discovery.py`, `canonical_fusion.py`, `candidate_acceptance.py` are the invoked code).
- `tools/run_phase4_recovery.py`, `run_phase45_real.sh`, any new runners, test fixtures and compiled worktree CUDA extension.
- Existing native helpers: `utils/gaussian_sequence_assignment.py`, `utils/kpart_counterfactual_rendering.py`, `estimator/surface_bootstrap.py`, `observation_diagnostics/renderer_access.py`; reuse only after checking tensor, projection, opacity, and frame conventions.
- Reports: `PHASE45_IMPLEMENTATION_REPORT.md`, `phase45_repo_audit.md`; document actual changed implementation versus claims.

Publish an audit mapping each new requirement to **existing function / required change / test**. Preserve every passing CUDA gradient and Phase 0–3 compatibility test. Never modify the main checkout, its environment's installed legacy extension, the input dataset, or upstream Phase 0–3 artifacts. The worktree's new `rasterize_gaussians_rgbd` must remain opt-in; AA must stay disabled for this pipeline.

## 2. Immutable common physical and observation contract

Use **one** automatically recommended direction already selected upstream, one matching endpoint Gaussian population, and its corresponding Phase-2 moving-core indices and Phase-3 trajectory. Do not require two endpoint models, two reconstructions or opposite-direction reconstruction. All original frames/cameras in the declared reconstruction set are available for Phase 4; do not imply that the chosen *processing direction* restricts which frames can be inspected.

Observed data: registered undistorted pinhole `I_ct ∈ [0,1]^(H×W×3)`, camera-Z metric `D_ct` in meters, strict complete-object mask `M_ct ∈ {0,1}`, calibrated camera intrinsics/extrinsics, original-frame IDs and physical timestamps. Valid depth is `V_ct(p)=M_ct(p) ∧ finite(D_ct(p)) ∧ D_ct(p)>0`. Never use outside-mask depth as object data; invalid depth cannot create a 3D birth. Mask is full **object**, not a static/moving part segmentation.

Canonical original Gaussian `G_i^0=(μ_i^0, R_i^0, s_i, α_i, SH_i, uid_i)`. Preserve original row indices and immutable stable `uid_i` even if moving support expands. `Σ_i^0=R_i^0 diag(s_i²) R_i^{0T}` is the **rendering** covariance, not depth sensor noise or a point-location confidence covariance.

The fixed Phase-3 joint yields a **relative-to-canonical** transform `T_t=[Q_t,p_t]`, anchor `T_a=I`:

- Prismatic: `Q_t=I`, `p_t=(q_t-q_a)e` with unit `e`, `q` in metres.
- Revolute: `Q_t=Exp((q_t-q_a)[e]_×)`, `p_t=(I-Q_t)a_J`, `q` in radians.
- Moving `μ_i(t)=Q_t μ_i^0+p_t`, `R_i(t)=Q_t R_i^0`, `Σ_i(t)=Q_t Σ_i^0 Q_t^T`.
- Static `μ_i(t)=μ_i^0`, `R_i(t)=R_i^0`.

Never optimize or scale `q`, `e`, `a_J` or camera extrinsics in Phase 4. Keep Phase-3 joint exact byte/hash proofs in each output. Ensure Phase-3 `original_frames_processing_order` is used directly; do not confuse original-frame and processing indices or assign backward q to wrong frames.

All counterfactuals render one combined depth-sorted scene with full static and moving occlusion. Pixel splat weights are `w_i(p)=a_i(p) Π_{j<i}(1-a_j(p))`. Rendered alpha `A=Σ_i w_i=1−T_final`. The new metric depth channel is `Dhat=Σ_i w_i z_i/(A+ε)`; **expected depth is not necessarily a first physical surface** when layers overlap. The legacy inverse-depth output must not be mistaken for metric depth. SH convention must be identical under competing hypotheses (start 4A at the upstream validated degree-0 appearance; any later SH evaluation must be consistent with material orientation).

### 2.1 Loss model (declare, audit, and freeze per run)

Use one fixed objective and pixel supports for **all competing hypotheses**:

```
L_ct(G) = λ_RGB L_RGB(Ihat,I,M)
        + λ_D L_D(Dhat,D,V; A and depth-support diagnostics)
        + λ_M L_M(A,M)
```

Store per-term loss, pixel denominators and support counts. `V` depends on **observed** valid foreground, never on a candidate's predicted visibility. If excluding predictions with low alpha, log exclusions and enforce coverage/mask penalty; do not allow a candidate to lower apparent depth loss simply by changing the denominator. Depth uncertainty from `utils/noise/sensor_model.py` is used where calibrated, with explicit independent rendering/model mismatch; assumed values are labeled uncalibrated. Distinguish scores from calibrated probabilities. Whole-image SSIM has a finite neighborhood: exact locality needs a halo and correct original global normalization, not a naïve average over crops.

## 3. PHASE 4A — exhaustive ORIGINAL-Gaussian material recovery

### 3.1 Scope and freeze rules

Input `N0` canonical Gaussians. Output must contain **exactly N0 original Gaussians** with **identical canonical means, rotations, scales, opacity and color**, in the same original row order. No births, clones, splits, pruning, densification or geometry/appearance optimization. Only per-original evidence and provisional render material may change. Phase-4A and 4B/4C have **different output directories, CLI entry points and checkpoints**.

Inspect **every original Gaussian**, including core Gaussian consistency. Trusted core cannot be silently relabeled; if a core group is inconsistent, flag `UPSTREAM_JOINT_OR_SUPPORT_CONFLICT` and stop/fail the integrity gate. Noncore material inference may be componentwise for efficiency, but all original UIDs must have a recorded evaluation outcome. Do not treat `UNTESTED` as `STATIC`.

### 3.2 Counterfactuals

For original candidate `B` (singleton or spatially coherent group), hold all other Gaussians fixed under the current hard rendering assignments:

```
G_S(B)    = G_{−B} ∪ B_STATIC
G_M(B)    = G_{−B} ∪ T_t(B)_MOVING
G_OMIT(B) = G_{−B}                     # diagnostic, no automatic deletion
E_H(B)    = Σ_{c,t} L_ct(G_H(B)), H∈{S,M,OMIT}
Δ_B       = E_S(B) − E_M(B)
```

Positive Δ supports moving **conditionally**, not categorically. Check absolute residual quality, material alternatives, observable motion, per-camera and per-frame contradictions and sensitivity to declared joint uncertainty. A large `E_S` and equally large `E_M` is `SUSPECT_GEOMETRY_OR_JOINT`, not necessarily a valid label. A near-axis moving Gaussian may remain `UNRESOLVED` because its motion is unobservable. Core stays trusted even if near-axis, subject to upstream audit.

Group proposals based on canonical surface connections and shared render influence, **not** broad 3D proximity across front/back layers or across a physical articulation boundary. If a group's evidence is mixed, split/retest its constituent Gaussian IDs. Report per-UID status even when hypothesis evaluated in a group.

### 3.3 Decision outcomes

Record separately: `material_evidence_status` and `render_material` (provisional static/moving assignment for unresolved entries). Statuses:

- `CORE_MOVING_ANCHOR` (must be audited; original confident moving seed)
- `MOVING_CONFIRMED`
- `STATIC_CONFIRMED`
- `UNRESOLVED_UNOBSERVABLE` (e.g. hinge)
- `UNRESOLVED_AMBIGUOUS`
- `SUSPECT_ORIGINAL_GEOMETRY` (OMIT substantially better than both)
- `UPSTREAM_JOINT_OR_SUPPORT_CONFLICT`
- `UNTESTED_RESOURCE_LIMIT` (not a completed 4A)

A score margin or ratio is **not** calibrated classification probability. Record raw `E_S,E_M,E_OMIT`, per-loss components, changed views and weighted affected pixels; margins and uncertainty perturbations are declared configuration, never hidden constants. Preserve original Gaussian IDs; labels derived from testing an interacting component must not be reported as independent single-Gaussian likelihoods.

### 3.4 Efficiency without losing exhaustive evidence

The 378-view×28k-Gaussian case must be practical. Implement:

1. **One-time observation preparation** in compact CPU arrays or memory-mapped storage, with actual memory accounting, no repeated PNG/NPY decoding across thousands of candidate tests. Do not put all 378 full-resolution views on the shared 8-GB GPU.
2. **View-major batch scheduling** and baseline-render cache versioned by the current label configuration. Candidate screening may use cheap influence maps; final accepted statuses require the declared full set of views.
3. **Exact affected-region/tile counterfactuals** (union of candidate static and moving projected influences) with **all occluding Gaussians** in those pixels; no isolated Gaussian render or soft half-opacity proxy. Include SSIM halo and fixed full-image denominators. If a real exact-tile implementation is not yet possible, retain the correct full-reference scorer for bounded tests and mark production not ready; do not claim a truncated subset as a final all-view score.
4. **Interaction-aware batches**: when multiple candidate footprints overlap, changes are not additive. Commit conflict-free candidates or retest interacting ones against updated baseline; run a full all-view model audit after a round.
5. Progress logging by tested UID/group and view sweeps; GPU/CPU time, cache hit/miss and decode count, calls and peak VRAM; atomic resumable checkpoints tied to full upstream input hashes.

**Exactness oracle:** for fixed candidate groups, compare optimized local score differences to full-scene full-frame reference for each term and S/M/OMIT, across occlusions, thin layers, empty regions and SSIM boundaries. Must pass declared numerical tolerance, and preserve the same ordering/decisions. No arbitrary view cap for claiming full validation.

### 3.5 Phase-4A artifact and stop gate

Write under `<phase4_out>/4a_original_membership/`:

```
manifest.json                      # input hashes; direction, J/q immutable fingerprint; camera/frame IDs
original_gaussians_snapshot.npz   # original UID/row map and before/after equality evidence
membership.npz                    # statuses, provisional render labels, component IDs, source core membership
counterfactual_evidence.jsonl     # per-component/per-UID conditional S/M/OMIT evidence
joint_sensitivity.json            # declared perturbations + ambiguity, no invented probabilities
all_view_model_audit.json         # RGB-D-mask residual/coverage and per-view contradiction metrics
resource_report.json              # timings, number of full/ROI renders, decodes, peak memory
complete.json                     # ONLY emitted atomically when every original UID accounted for and tests pass
```

`complete.json` asserts `N_final=N0`, canonical parameter hashes unchanged, joint hashes unchanged, no original UID untested, and genuine all-view verification. **4B cannot run without it.** Do not interpret the absence of `complete.json` as an empty scene or success.

### 3.6 Mandatory 4A tests

- Every original UID evaluated: withheld-moving R/door and P/drawer; no false static→moving; hinge entries are allowed unresolved; core never silently overturned.
- 4A reassigns a previously unselected Gaussian without adding any Gaussian; `N0` and canonical bytes identical.
- Same outcome forward/backward when paired to correct anchors and q index mapping; no opposite-direction computation requirement.
- ROI delta score vs whole-render oracle for RGB, depth, mask, SSIM, depth-layer occlusion and overlapping candidates.
- 378-view repeated passes decode each source view no more than documented once per preparation (or documented view-major schedule), with bounded CPU/GPU memory; a measured full-size cost estimate before real production.
- Complete full-scene all-view validation on **one** real red-fridge 4A run before moving to 4B.

---

## 4. HARD BOUNDARY — 4A FINISHES BEFORE 4B STARTS

A 4B process **loads the read-only `4a_original_membership/complete.json` and canonical scene**. It never changes 4A results in place. Any unresolved-label revisit requires: (1) flag conflict and stop/refer candidate, (2) resume 4A on the ORIGINAL population using new evidence with an explicitly versioned reclassification pass, (3) emit new immutable `4a` snapshot, (4) restart/revalidate downstream proposals whose baselines changed. No hidden co-optimization and no retroactive tampering with accepted births.

This is the most important implementation constraint. Unit tests must assert 4B and 4C cannot mutate the original-membership artifact, original UID mapping, or fixed joint.

## 5. PHASE 4B — discover *new moving* surfaces from the recovered moving support

### 5.1 Scope

Phase 4B creates **candidate surface patches**, not accepted Gaussians. Start with the original trusted moving core plus `MOVING_CONFIRMED` assignments from 4A; unresolved original Gaussians may be rendered under their declared provisional material, but **cannot seed frontier growth**. Do not run general global static geometry completion inside this moving completion stage. Static hypothesis remains a **competing explanation** assessed in 4C; a static winner gets `NOT_MOVING_CANDIDATE / REVIEW_STATIC_SEPARATELY`, not automatically committed as a static Gaussian. This is a deliberate task boundary, not a trick to pass purity tests.

### 5.2 Single view-major sequence scan

For every **original frame and every declared camera**, render complete scene once and save/cache compact per-view diagnostics:

- observed `(I,D,M,V)`, per-camera calibrated σ_D(z) and model discrepancy if available;
- rendered RGB `Ihat`, alpha `Ahat`, weighted metric depth `Dhat` **and, when needed, contributing front-layer depth/weights** (avoid treating mixed expected depth as measured surface);
- **visible moving responsibility** `A_M(p)=Σ_{i∈moving} w_i(p)` with `w_i` including **all** static and moving occluders; and the first/strong contributors/visible layer provenance;
- foreground coverage deficit, depth/appearance residuals, local depth discontinuity/mixed-pixel diagnostic, offmask opacity, actual visibility/occlusion.

The scan is `O(number_of_views)` decodes/renders per scene version, not `O(candidates × views)` image I/O. Store surface evidence references to `(camera, original_frame, pixel, measured depth and RGB, flags)`.

### 5.3 Residual explanation taxonomy — BEFORE proposing births

For observed foreground pixels, classify (possibly with uncertainty) into:

| Code | Physical reading | Birth handling |
|---|---|---|
| `EXPLAINED` | Existing front surface and coverage consistent within declared uncertainty | no birth |
| `APPEARANCE_ONLY` | depth/coverage plausible but RGB differs | no geometry birth |
| `ORDINARY_GEOMETRY_RESIDUAL` | deviation explained by sensor+model noise or modest mis-shape of existing Gaussian(s) | no automatic birth |
| `COVERAGE_GAP` | valid observed surface; insufficient predicted object/moving support | candidate investigation |
| `FRONT_SURFACE` | reliable measured object surface is nearer than model's predicted visible layer | candidate investigation or mis-modeled existing surface |
| `BEHIND_PREDICTION` | measurement lies behind rendered foreground layer | occlusion/model-label issue; **not** automatic back-surface birth |
| `MIXED_EDGE_AMBIGUOUS` | depth jump, RGB/depth misregistration, inconsistent local support, or sensor mixed return | do not directly turn into a 3D point |
| `DEPTH_INVALID` | no valid masked metric depth | silhouette/RGB only, no 3D backprojection |
| `OUTSIDE_OBJECT_MASK` | M=0 | no object depth birth; check silhouette contradiction |

The taxonomy separates *model inadequacy* from *missing material*. Do not infer a birth solely from `|D−Dhat|>constant` or from a mask boundary; calibrate depth noise by camera/range and independently assess renderer/model error. Existing Gaussian location/scale/opacity problems should be logged as an **existing-geometry correction alternative**, not automatically repaired by creating a duplicate. Any bounded corrective fit is a **shadow diagnostic**, frozen/discarded unless a separate explicitly authorized refinement stage exists; Phase 4A's canonical original geometry remains unchanged.

### 5.4 Frontier: where are we allowed to search first?

Find **canonical 3D moving-surface frontier patches** from trusted moving Gaussians using local tangent coverage, surface footprints, visibility and layer identity. Project those patches under the fixed joint into all cameras/times and identify adjacent unexplained measured foreground regions. There are three different boundaries:

1. **True geometric frontier:** known moving surface ends and a measured compatible surface continues.
2. **Occlusion frontier:** a moving surface was hidden by another surface at canonical endpoint and is exposed later.
3. **Depth-image edge:** possibly a physical edge, but also mixed foreground/background sensor depth; *not* enough evidence by itself.

Use canonical reachability as *a proposal prior*, not as proof of material membership. A static cabinet wall may touch the drawer in 2D/3D near an articulation boundary. Enforce coherent normal-layer association, temporal joint transport, non-crossing visibility and candidate material comparison. Allow real right-angle corners by connecting adjacent **surface patches** across a supported edge; do not enforce one global coplanarity threshold.

For the first validated implementation, search only trusted-frontier-connected **moving candidates**. A **separate optional second search mode** may examine strongly joint-coherent disconnected unrepresented regions *after* the frontier converges; label this mode independently, require stronger evidence and never silently mix it into the frontier benchmark.

### 5.5 Canonical accumulation with known joint

For valid measured depth `d` at pixel center `(u+½,v+½)`:

```
r = K_c^(−1) [u+½, v+½, 1]^T
  # normalized to r_z = 1 for axial camera-Z depth

y_C = d r
y_W = R_WC y_C + t_WC
x_M^0 = Q_t^T (y_W − p_t)
x_S^0 = y_W                     # diagnostic STATIC competitor only
```

Propagate camera ray/sensor covariance to world and canonical space using projection Jacobian and `Q_t`. Joint uncertainty is **shared across all affected observations**, not iid per point; record nominal results plus sensitivity to declared plausible joint perturbations, without actually changing the joint.

Collect candidate measured samples into **physical surface patches** across views/times, not one Gaussian per pixel. Use the same-object/same-layer reprojection and inverse-transport consistency with a robust point-to-tangent-plane residual:

`r_j=n_Pᵀ(x_{Mj}^0−μ_P^0)` and `σ²_{⊥,j}=n_PᵀΣ_j n_P + σ²_model,j`.

Near a depth edge, sensor mixing makes the assumed single-surface depth likelihood invalid: flag ambiguous samples; seek corroborating independent pixels/rays/articulation states rather than using their back-projection as truth. **Do not** use fixed mask erosion bands, blanket reject pixels near a border, or assert a mixed pixel is a new surface. Preserve raw flagged observations for audits.

### 5.6 Layer identity and corner connectivity

A local kNN/voxel neighborhood alone must **never** merge two front/back sheets, even if 3, 5, 8, 12 or 20 mm apart and their normals are parallel. Require measured depth ordering, visibility/reprojection, patch-sidedness and canonical layer IDs; build separate neighbor graphs for co-layer continuity vs supported surface corner edges. Do not impose `min_layer_distance=20mm` as a replacement global threshold. Test real depth uncertainty and oblique views. Reuse `estimator/surface_bootstrap.fit_surface` only if a thin-panel test proves that cross-layer samples cannot contaminate its unconstrained kNN fit; otherwise add the needed layer-aware association before fitting.

### 5.7 Proposal representation and output (NO births yet)

Each candidate patch has:

```
patch_uid
canonical surface geometry (μ, tangent frame, normal, shape and uncertainty)
proposed set of one or several anisotropic Gaussian means, rotations, scales,
  opacities, DC color; all in canonical coordinates
hypothesis=MOVING (primary); S and OMIT reserved for 4C comparison
parent frontier UID(s), patch connectivity and layer UID
source observations (camera, original_frame, pixel, measured depth, RGB, noise flags)
independent support groups and diagnostic confidence (not invented probability)
prior original-Gaussian overlaps, unresolved material conflicts
proposal/generation view IDs; candidate source/holdout partition ID
```

Save under `<phase4_out>/4b_new_surface_proposals/`: `scan_manifest.json`, `residual_taxonomy.npz` (or compressed tiled representation), `moving_frontier.npz`, `source_observations.*`, `layer_graph.json`, `candidates.npz`, `candidate_provenance.jsonl`, `ambiguous_measurements.json`, `unexplained_coverage_report.json`, `complete.json`. `complete.json` means a **complete scan and proposals**, not accepted geometry. It must reference 4A's immutable complete hash. **No original or new Gaussian is added to the authoritative scene in Phase 4B.**

### 5.8 Mandatory 4B tests

- Synthetic drawer: original floor not visible at anchor, revealed later; moving inverse-transport aligns and preserves real plane; mixed drawer/background pixels approximately 18 mm off the true physical surfaces cannot become trusted surface evidence from one contaminated neighborhood.
- A genuine 90° moving corner can grow; a static cabinet immediately adjacent to the moving drawer cannot be swallowed into the moving frontier.
- Thin two-face panels 3/5/8/12/20 mm with same-facing parallel sheets, different obliquity and noise: layer-separated candidate patches.
- Short-lived genuine surface (one frame or a few cameras): cannot be falsely rejected purely for missing frames; evidence strength must be reported honestly.
- Never-observed geometry remains absent/unresolved and is never inferred from a geometric prior alone.
- 4B refuses to run without valid 4A complete hash; original Gaussian count, membership, joint geometry and trajectory remain unchanged.

---

## 6. PHASE 4C — validate NEW patches before any permanent birth

4C consumes **read-only 4B candidate patches** and the same immutable 4A scene. It must not redo material reassignment, nor generate candidates ad hoc from residual pixels. It *temporarily* inserts candidate moving geometry for fitting/evaluation, and commits only after the declared physical validation.

### 6.1 Three hypotheses, but moving-birth scope

For each coherent candidate patch `P`:

```
H_0: G                                      # ABSENT, no birth
H_M: G ∪ T_t(P)                            # MOVING, same fixed joint for ALL frames
H_S: G ∪ P                                # STATIC, comparison/diagnostic only
```

Using disjoint fitting and validation **data roles assigned before 4B proposal discovery where feasible**, fit a bounded canonical patch under fixed J/q, optimizing provisional Gaussian geometry only. Evaluate geometric support on measurements not used for fitting where possible and full-render validation evidence for all camera/frame views. `H_S` can demonstrate a candidate is not moving; **do not commit it to the moving reconstruction**, do not classify it as automatically acceptable static geometry, and never claim high moving purity by simply deleting all difficult candidates. Export residual static geometry for a separately approved static-completion task.

The chosen render objective must use identical background, RGB/depth/mask definitions and global pixel normalization under `H_0,H_M,H_S`, with an explicit complexity penalty for new geometry (rather than counting training loss gain as sufficient). Use a physically valid surface support check **independently of image appearance reduction**.

### 6.2 Visibility evidence (complete temporal interval)

For each proposed Gaussian at each original frame, it **exists** under `T_t` whether it was first seen at the last frame or the first. No creation-time motion; record observation source times as *measurement provenance* only.

Classify each candidate-view contribution as:

- **VISIBLE_MATCH:** predicted front surface compatible with observed object RGB-D;
- **VISIBLE_DEPTH_CONTRADICTION:** candidate produces foreground surface at impossible depth relative to reliable measurement;
- **VISIBLE_BACKGROUND_CONTRADICTION:** candidate yields visible alpha in `M=0` region;
- **OBSERVED_NEARER_OCCLUDER:** measured nearer surface hides candidate — neutral, **not** penalty;
- **OUT_OF_FRUSTUM:** neutral;
- **INVALID_OR_MIXED_DEPTH:** depth evidence inconclusive; RGB/mask and independent observations still available.

Use the renderer's true depth-ordered compositing and per-layer contributor evidence; comparing just candidate center-Z to expected `Dhat` is insufficient. Under actual opacity blending, quantify effect on existing scene RGB/depth/alpha (especially earlier frames). Occlusion by static and moving geometry happens in one combined scene.

### 6.3 Independence, model complexity, and decision

Select and store proposal/fitting/validation roles **before** searching observations if possible (spatially separated patches, camera/time cross-fit). The Phase-3 joint already used the sequence, so this validation is conditional on fitted J; never claim fully independent ground-truth confirmation. For short-visibility surfaces, ensure at least some informative validation evidence is possible; when none exists, return `INSUFFICIENT_INDEPENDENT_SUPPORT` rather than accepting a self-fit as proof.

Use these components in a decision:

1. A **real measured surface** exists: reliable non-mixed geometry, layer-consistent canonical patch, explicit source/held-out evidence and propagated uncertainties.
2. The **MOVING** transport matches all supported views better than alternatives to a defensible margin, or competing STATIC explanation is flagged as ambiguous; not merely one temporal edge pixel.
3. Compared to ABSENT and a bounded **existing-geometry correction shadow hypothesis**, the new patch provides justified reduction in held-out error without duplicating existing original/unresolved Gaussians.
4. No physical contradiction in any original view in which the candidate would be visibly exposed; neutral on genuine occlusion/out-of-view.
5. Complexity and joint-sensitivity are declared, not hidden thresholds or fake posterior scores.

Statuses: `ACCEPTED_MOVING`, `AMBIGUOUS_MATERIAL`, `INSUFFICIENT_INDEPENDENT_SUPPORT`, `REJECTED_MIXED_MEASUREMENT`, `REJECTED_DUPLICATE`, `REJECTED_VISIBLE_CONTRADICTION`, `NOT_MOVING_STATIC_ALTERNATIVE`, `JOINT_SENSITIVE`, `UNTESTED_RESOURCE_LIMIT`. Only `ACCEPTED_MOVING` creates authoritative Gaussian births.

### 6.4 Iterative expansion, no simultaneous 4A/4B training

At frontier-growth cycle `k`:

```
1. Freeze authoritative G^k and original 4A membership artifact.
2. Scan/select missing measured surface near current accepted-moving frontier.
3. 4B emits new proposals P^k (no mutation).
4. 4C fits and validates P^k against G^k, using full-sequence evidence.
5. Commit only ACCEPTED_MOVING P^k atomically to make G^{k+1}.
6. Update cached affected observations + frontier; reconsider interactions.
7. Repeat only if genuinely new, supported surface frontier remains.
```

Accepted patches may seed next growth only after verified independent support and complete-sequence consistency; preserve trusted original core as root of each frontier chain. A weak provisional patch may never seed another weak patch. If a proposed region overlaps unresolved ORIGINAL Gaussians, flag `ORIGINAL_MEMBERSHIP_CONFLICT` and return to a separately executed/versioned 4A re-evaluation **before** accepting duplicate births; do not silently reassign original labels inside 4C.

The final accepted scene must be rerendered **jointly** across all original views: independent per-patch gains are not additive when Gaussian footprints and occlusion interact. Re-evaluate interacting groups and detect ghost surfaces, duplicated front/back layers, offmask opacity and early-frame conflicts.

### 6.5 4C performance is an algorithmic requirement

The old reported ~100 s per patch×378 views is not acceptable. Reuse 4A's validated exact ROI/tile scorer but evaluate candidates **view-major**, with only affected projected contributions recomposited and all occluding splats included. Cache baseline (and target) channels and full-image-normalized local loss. Include SSIM halo. Candidate screening may be coarse; **final commit uses ALL views**, with sufficient computed influence bounds to skip only **proven zero-effect views/tiles**, not an arbitrary 8-view cap. Keep only small GPU buffers; stream CPU-cached source views. Record exact full reference versus ROI score for representative P/R scenes and real views; report wall-time, peak VRAM, number of views/tiles touched, full-scene validation passes. No output qualifies as production when final all-view evaluation was truncated.

### 6.6 Phase-4C outputs

Under `<phase4_out>/4c_accepted_moving/`:

```
accepted_patches.npz                # new canonical Gaussians only; persistent UIDs
per_patch_evidence.jsonl           # ABSENT/MOVING/STATIC and geometry-correction diagnostics
per_patch_visibility.jsonl         # every original frame/camera: match/contradiction/neutral
birth_provenance.jsonl             # parent frontier, source camera/time/pixels, layer UID
candidate_statuses.jsonl           # accepted/rejected/ambiguous; no missing candidates
interaction_audit.json
final_canonical/point_cloud.ply     # original N0 unchanged canonical geometry + accepted moving births
final_canonical/material_labels.npz
full_sequence_audit.json           # metrics, original-frame mapping, coverage, contradictions
joint_unchanged_proof.json
resource_report.json
complete.json                      # atomic; ONLY when all required gates and full-view audit pass
```

Check `N_final=N0+N_accepted_births`; all new UIDs distinct with immutable original UID lineage; all new Gaussians have exactly one physical moving label and obey the single fixed `T_t` for every frame. No accepted static births in this default moving-only completion contract.

### 6.7 Required 4C tests (do not tune to one purity target)

- Drawer mixed-edge fixture: trace the 19/80 failures to original pixels and their 3D surfaces; demonstrate why each is invalid physically, not merely that it was relabeled or omitted. The ≥90% moving-purity benchmark is **one** regression metric, not the objective being hacked.
- Real static surface visible briefly behind moving drawer: detected as stationary alternative but **not** falsely declared moving; lack of later visibility cannot alone reject it.
- Late-visible valid moving drawer floor: measured area and surface completeness increase with iterative growth; report observed-vs-never-observed GT area separately (previous ~65% completeness was incomplete).
- Revolute door rear-side surface recovered in canonical coordinates, remains behind front when closed; wrong-side-to-door patch rejected due to early visible contradiction.
- Thin parallel same-facing layers at multiple thicknesses remain separated; 90° corner grows without eating adjacent static body.
- True moving surface visible briefly passes when physically supported; nonexistent surface not hallucinated.
- 4C exact optimized ROI versus full-frame scene delta parity, including SSIM, depth invalid/alpha, combined patches, depth ordering; all-view final scan.
- 4C leaves original canonical Gaussian bytes and original 4A labels unchanged; Phase-3 hashes unchanged. No downstream Phase-5 call.

---

## 7. Explicitly do not mix Phase 4 and Phase 5

Phase 4 can briefly refine only **candidate NEW patch geometry** inside 4C so that competing hypotheses are fairly evaluated, always against the frozen original scene and frozen joint. No global photorealistic Gaussian training, no SH progression, no original-scene densification/pruning and no joint refinement in Phase 4. Such operations would invalidate material counterfactuals and blur the distinction between original-support recall and new-surface discovery.

A separately controlled **Phase 5** consumes only a completed 4C canonical snapshot, with frozen material labels and fixed joint. It is specified later **in this same Markdown file** under **PART II — PHASE 5**. **Do not implement or run Phase 5 as part of this Phase-4 repair task.** `--stage all` in the Phase-4 runner means **4A→4B→4C ONLY**, with hard stage output checks; it does *not* invoke Phase 5.

## 8. CLI contracts and stage isolation

Prefer explicit commands (adapt exact arguments to currently implemented code; show `--help` after edits):

```bash
python -u tools/run_phase4_recovery.py --stage 4a \
  --source "$SOURCE" --model "$ENDPOINT_MODEL" --phase2-dir "$PHASE2_DIR" \
  --phase3-dir "$PHASE3_DIR" --direction "$DIRECTION" \
  --joint-type "$JOINT_TYPE" --output "$PHASE4_OUT"

# Must REFUSE unless the immutable completed 4A snapshot exists:
python -u tools/run_phase4_recovery.py --stage 4b \
  --phase4a "$PHASE4_OUT/4a_original_membership" --output "$PHASE4_OUT"

# Must REFUSE unless valid 4A and fully scanned/validated 4B proposal artifacts exist:
python -u tools/run_phase4_recovery.py --stage 4c \
  --phase4a "$PHASE4_OUT/4a_original_membership" \
  --phase4b "$PHASE4_OUT/4b_new_surface_proposals" --output "$PHASE4_OUT"
```

The above is **desired CLI contract**, not a claim that all options already exist; preserve existing essential args and avoid breaking compatible code. `--stage all` is only an orchestrator calling the three distinct stage functions one after another, with completion/hash validation between stages. Any failure stops immediately without writing downstream `complete.json`. Store resolved config, source hashes, deterministic seed, calibration/uncertainty policy, branch/commit/diff state, CUDA build hash and status flags in every stage manifest. Warn when all-camera or all-frame coverage is incomplete; mark smoke runs `NOT_FOR_RESULTS`.

## 9. Implementation order and immutable test gates

**Gate 0 — audit and freeze baseline:** inspect actual worktree (new exact files may supersede old), preserve reference numerical outputs, input hashes and current test scores. Never kill unrelated processes; report any active old Phase-4 process PID and command before requesting termination. Use `python -u ... 2>&1 | tee logfile`, never a sole buffered `tail` pipe.

**Gate A — exhaustive original 4A:** implement/verifiably retain all original S/M/OMIT counterfactuals, explicit trusted and unresolved status, exact ROI parity, no image decoding thrash, final full-sequence audit; complete 4A on one real scene **alone**. No births attempted yet. Freeze artifact.

**Gate B — physically grounded 4B:** original image scan once per scene version, missing-vs-normal-vs-mixed residual taxonomy, visible moving responsibility, canonical 3D frontier, uncertainty/layer-aware accumulation. Produce provisional candidates only. Test mixed depth, near-static boundary, corners, back-layer and rare visibility.

**Gate C — exact 4C:** test geometrically plausible candidates vs ABSENT/MOVING/STATIC and existing-geometry alternative; full-sequence physical visibility and ROI parity; accepted moving only; iterative frontier update; final jointly rerendered 4C scene. Validate on procedural fixtures first, then one real scene only after all prior gates pass.

**Gate D — measured resources and report:** measure actual 28k-Gaussian/378-view real-stage performance, CPU decode calls, GPU render counts, RAM and VRAM. Never launch 13 overnight jobs on an unmeasured loop. A deliberately stopped test must be explicitly marked incomplete; no fabricated expected total runtime or completion claims.

Report: files touched, tests run, measured results, failures with line-level root causes, precise model selection/geometry evidence, stage outputs, input-output hashes and remaining blockers. Never declare Phase 4 production-ready on 8-view smoke evidence.

## 10. FINAL CODEX TASK (verbatim operating instructions)

> Implement a mathematically grounded **two-pass Phase 4** in the EXISTING `RigEuler3DGS-p45` worktree. **Pass ONE (4A):** test every original Gaussian under static and exact fixed-joint-moving hypotheses, keep the original Gaussian population and all canonical attributes unchanged, produce a complete immutable original-material snapshot. **Pass TWO (4B then 4C):** scan the complete original RGB-D-mask sequence around the recovered confident moving frontier; distinguish ordinary RGB-D errors, mixed depth, mislabeled existing geometry, and actually missing physical surfaces; inverse-transport reliable measurements into the fixed-joint canonical moving frame, fit layer-aware surface patches, then validate and accept new MOVING Gaussians under full-sequence nonlinear visibility. Do not birth in 4A or reassign original labels inside 4B/4C. Prove exact optimized local evaluation against full-scene reference; avoid per-candidate per-view image decodes and full re-renders. Phase 5 is documented in PART II below but must not be executed during this Phase-4 task; joint refinement is forbidden. Do not call an 8-view smoke run a validated result. Preserve all upstream files, original Gaussian lineage and CUDA compatibility; show executed tests and measured real performance, not aspirations.

---

# PART II — PHASE 5: PHOTOREALISTIC 3DGS REFINEMENT (SEPARATE EXECUTION)

**This section is in the same Markdown document for handoff convenience. It is NOT part of the Phase-4 implementation or its `--stage all` runner and must not run now.**

**Target:** `~/Joint3DGS/RigEuler3DGS-p45` (`phase45` worktree, inspect actual state).  
**Precondition:** a **complete and validated Phase-4C** artifact from **PART I of this same document**.  
**This is a SEPARATE execution phase/program/output/checkpoint, documented in the SAME Markdown file.** Do not call this trainer from any Phase-4A/4B/4C implementation.  
**Joint:** frozen Phase-3 joint type, geometry and entire trajectory. **No joint optimizer, no axis adjustment, no per-frame q correction, no camera pose refinement.** Any future joint-only refinement is a separate opt-in command, explicitly OUT OF SCOPE and not to be run now.

## 0. Hard separation and current implementation status

The previous Codex report states there are already `recovery/training.py`, `tools/train_phase5_rgbd_gaussians.py` and a `rasterize_gaussians_rgbd` entry point with reported CUDA/finite-difference tests. **Audit and reuse these; do not rewrite successful code**. The reported procedural tests demonstrate expected loss reduction and a `joint_unchanged_proof.json`, but the report explicitly did not demonstrate completed real-sequence or full-size synthetic training. Do not claim those results until measured.

**Phase 4** discovers and verifies physical geometry; **Phase 5** refines the representation/appearance of that **already accepted** static+moving canonical model. It does not create a new physical material assignment from rendering residuals and does not rerun surface discovery. Re-enter Phase 4 with explicit new versioned artifacts if material completion is found inadequate; never silently run 4A/4B/4C inside the Phase-5 loss loop.

The input's original Gaussian labels, accepted new Gaussian labels and joint transform are **fixed** in this entire phase. Newly densified children inherit their parent's original material labels and lineage; no static/moving interpolation.

## 1. Inputs (hard preflight)

Load:

- `<phase4_out>/4a_original_membership/complete.json` and `<phase4_out>/4c_accepted_moving/complete.json`, with matching upstream hashes.
- The Phase-4C canonical static+moving Gaussian PLY and material-label/UID arrays; count and lineage cross-check (`N_total=N_original+N_accepted_births`, subject only to explicit later Phase-5 densification/pruning).
- The **same** fixed Phase-3 joint type, `geometry_fixed`, full `trajectory.csv`, original-frame mapping, direction/endpoint, camera calibration and image registration used by Phase 4.
- Original RGB images, positive finite foreground camera-Z depth, full-object binary masks, and declared sensor/model uncertainty metadata where available.
- The worktree's native CUDA `rasterize_gaussians_rgbd` implementation; **no Nerfstudio training backend** and no alternate camera/coordinate convention.

Reject missing/incomplete 4C, wrong canonical endpoint, incorrect trajectory mapping, out-of-range/duplicate Gaussian IDs, missing binary masks, 3DGS orientation-override bugs, unavailable RGBD extension, antialiasing enabled, and any joint/trajectory parameter with `requires_grad=True`.

`UNRESOLVED` original material status is epistemic rather than a third physical transform. For Phase 5 use a **declared, frozen physical rendering label** for unresolved originals, and export the list so reconstruction metrics do not pretend all membership is known. No automatic conversion to STATIC or MOVING inside the trainer.

## 2. The exact physical forward model

For each original frame `t` (mapped using Phase-3 original-frame indices) and camera `c`:

```
G_t = G_S(Θ) ∪ T_J(t) G_M(Θ)
Yhat_ct = R_native_cuda(G_t, camera_c) = (Ihat_ct, Ahat_ct, Zhat_ct)
Dhat_ct = Zhat_ct / (Ahat_ct + eps)
```

For the fixed joint `T_J(t)=[Q_t,p_t]`, moving canonical means `μ_i^0` become `μ_i(t)=Q_t μ_i^0+p_t`; orientations `R_i(t)=Q_t R_i^0`, covariances `Σ_i(t)=Q_t Σ_i^0 Q_t^T`. Static Gaussians keep their canonical pose. These tensors may have gradients with respect to **Gaussian canonical parameters**, but **not joint or camera parameters**. Use a single depth-sorted render with true static/moving self-occlusion.

For SH material color, distinguish world-space SH parity with upstream Phase 1–3 and local/object-coordinate SH. In `local_SH` mode, the viewing direction for a moving Gaussian is inverse-rotated by `Q_t^T` into the Gaussian's canonical material frame, so surface appearance follows the part. At anchor `Q=I`, parity must hold. Record appearance-mode choice and initial render gap; do not change SH convention mid-run silently.

## 3. Two training profiles — make them explicit

### `gaussian_rgbd` — full Gaussian optimization, fixed joint

Optimized **canonical** parameters: Gaussian means `μ^0`, quaternions/rotation, scales, opacity, SH/DC. Joint and membership frozen. Use complete RGB-D-mask loss and declared geometry/complexity regularization. The newly completed model can be refined, split, cloned, and pruned **only** under canonical-space/topology rules preserving material labels and UID lineage. This mode can improve geometry, opacity, depth and silhouette. It is not merely color training.

### `sh_only` — literal appearance-only optimization, fixed *all* geometry

Only SH/DC coefficients are trainable. **Bitwise freeze** means, rotations, scales, opacities, number of Gaussians, material assignments, joint and trajectory. **No** densification/pruning, no opacity reset. Consequently metric-depth and alpha predictions should remain identical before/after (barring numerical nondeterminism). Depth and mask may be monitored but are not trainable channels in this mode; use RGB objective for optimization. Do not claim depth/geometry improvement from SH-only training.

Record selected profile in manifest and test it explicitly. Default `gaussian_rgbd` only when a validated Phase-4C artifact is present and a user intentionally invokes Phase 5; **never auto-invoke from Phase 4**.

## 4. Loss functions in original CUDA output convention

Let `M_ct` be strict whole-object binary mask and `V_ct=M_ct ∧ finite(D_ct) ∧ (D_ct>0)`. Implement reproducible FreeArtGS-style baseline:

```
I^M    = M * I
Ihat^M = M * Ihat
L_rgb  = (1−λ_ssim) * mean_abs(Ihat^M−I^M)
       + λ_ssim * [1−SSIM(Ihat^M,I^M)]
L_mask = mean_abs(Ahat−M)                          # entire image, including background
L_depth= Σ_{p∈V} |Dhat(p)−D(p)| / max(1, |V|)      # declared masked-valid reduction
L_base = λ_RGB L_rgb + λ_M L_mask + λ_D L_depth
L_full = L_base + R_surface + R_complexity          # full-geometry profile only
```

The FreeArtGS implementation masks RGB and depth and trains alpha with a separate mask term. The baseline here **explicitly differs** in depth normalization: only observed valid foreground depth contributes, rather than letting invalid/zero depth change the image-average denominator. Make any other deviation explicit and include a strict parity experiment for comparable inputs.

**Depth low-alpha pitfall:** `Dhat=Z/(A+eps)` is unstable for `A≈0`. If depth loss ignores low-alpha predictions, log excluded-valid-foreground fraction and retain an explicit coverage penalty; do not let the predictor eliminate depth error by setting opacity to zero. Use a *fixed observed-valid* denominator for numerical comparisons and monitor invalid predicted support. Do not treat expected depth under overlapping semi-transparent layers as a first-return surface. In an optional depth model ablation, compare with first-return depth, but never silently replace the baseline's definition.

A robust depth loss or per-camera noise model may be added **only as a separately named experiment**, not conflated with the strict FreeArt-inspired baseline. Do not quietly apply RGB/mask/depth rescaling based on image size or valid-pixel count differently between train and evaluation.

## 5. Training loop, data coverage, regularization

1. Load the exact completed Phase-4 model, material labels, and joint/trajectory fingerprint; save initial all-view metrics and representative renders.
2. Use deterministic seeded view sampling over the complete original frame×declared-camera set, balancing rare-visible surfaces and articulation positions; guarantee every view visited within a documented epoch. No arbitrary global 8-view training when claiming full-sequence output.
3. For each view, transport Gaussians using fixed `T_J(t)`; render with original CUDA RGBD-alpha; optimize only the permitted Gaussian tensors.
4. For `gaussian_rgbd`, use established 3DGS optimizer and gradual SH progression, optionally densification/cloning/splitting/pruning under strict UID/material ancestry. Splits occur **in canonical space**, never at a transported frame; all children inherit parent material. Keep topology rules/thresholds declared and benchmark false duplication on thin surfaces.
5. `sh_only` uses DC/SH optimizer only. Gradient masks and optimizer parameter groups must prove all other tensors unchanged. Turn off any dynamic Gaussian model optimizer or camera/joint optimizer implicitly enabled by original RigEuler training code.
6. At regular intervals save frame/camera loss terms, coverage, depth MAE/RMSE on observed-valid pixels, offmask alpha, PSNR/SSIM, low-alpha-depth exclusion, numerical stability and peak VRAM. Recompute an **all-view** evaluation at initial, milestone and final checkpoints.
7. Use a frozen evaluation partition to detect overfitting, marking it conditional on a Phase-3 joint fitted using the same sequence. Never assert fully independent generalization from the same capture.
8. Stop on NaN, joint fingerprint discrepancy, material-label mismatch, canonical-view inconsistency, invalid depth scale, CUDA/autograd anomaly or an untracked change of Phase-4 source hash.

## 6. Joint immutability: proof rather than trust

At startup and after every checkpoint, compare:

- file hashes of Phase-3 selected joint `geometry.json`, `trajectory.csv`, `summary.json` and phase-4 copied joint artifacts;
- tensors representing unit axis, axis line/point (where applicable), per-frame q and original-frame mapping;
- absence of joint/camera learnable parameters in all optimizer groups;
- fixed transform `Q_t,p_t` for selected canary frames on GPU and CPU;
- material labels for all original and accepted new UID rows (accounting for inherited children in full Gaussian mode).

Emit `joint_unchanged_proof.json` with `joint_optimized:false` and exact before/after evidence. Failure must abort run. Do not call, connect, import-as-side-effect or execute the optional `tools/refine_joint_geometry_final.py`. Its present CLI/dry-run guard remains **untouched**; the missing optimizer loop is irrelevant and not to be implemented here.

## 7. Separate outputs and staging

Use `<phase5_out>/` distinct from the `<phase4_out>/` directory:

```
manifest.json                     # mode, full upstream Phase-4C hash, joint/frozen labels
config_resolved.json
initial_all_view_metrics.json
checkpoint_XXXX/                 # model, optimizer, material/UID lineage, hashes
loss_per_camera_frame.csv
surface_and_visibility_metrics.json
renderer_parity.json             # legacy RGB vs new RGBD and SH convention
final_canonical/point_cloud.ply
final_canonical/material_labels.npz
final_all_view_metrics.json
joint_unchanged_proof.json
resource_report.json
complete.json                     # only if all tests, all-view audit and proof pass
```

**Never overwrite Phase-4 artifacts or alter their acceptance provenance.** Phase 5 may expand the Gaussian count in `gaussian_rgbd` but must separately track all changes with stable parent lineage; `sh_only` count remains exactly fixed.

## 8. Required tests and execution gate

- **Input contract:** refuse missing Phase-4C `complete.json`, mismatched hashes, wrong direction, masked/depth registration issue and active joint optimizer.
- **CUDA:** existing RGBD-alpha forward/backward test suite and bit-identical legacy renderer outputs; AA disabled; gradients in canonical transported Gaussian parameters, including scale/rotation/opacity and SH.
- **Revolute/prismatic:** frozen joint correctly transports all labeled moving Gaussians and no static Gaussian; same poses at every frame across phases; `joint_unchanged_proof.json` always valid.
- **`sh_only`:** bitwise exact geometry/opacity/material/trajectory before/after, identical depth/alpha, decreased or stable RGB loss in small deterministic fixture.
- **`gaussian_rgbd`:** full objective decreases on known synthetic R/P scenes, gradients reach geometry, labels inherited on split/prune, surface precision/coverage and background alpha do not silently deteriorate.
- **Real pilot:** only after Phase-4C tests pass, run **one** complete real-scene sequence with staged checkpoints, all-view metrics and measured RAM/VRAM/wall time. A 4-frame smoke or 8-view subset is strictly `NOT_FOR_RESULTS`.

## 9. CODEx RUN POLICY — important

**Do not run Phase 5 now** while Phase 4A/4B/4C scientific validation is pending. This section is an **independent later implementation and test contract**. If inspecting current Phase-5 code is useful to audit shared native renderer behavior, inspect read-only; do not invoke training or launch long jobs.

When explicitly authorized after Phase 4 completion, the desired invocation is a separate command, e.g.:

```bash
python -u tools/train_phase5_rgbd_gaussians.py \
  --phase4-dir "$VALIDATED_PHASE4_OUT" --source "$SOURCE" \
  --direction "$DIRECTION" --joint-type "$JOINT_TYPE" \
  --mode gaussian_rgbd --output "$PHASE5_OUT"
```

Adapt new CLI flags to actual installed commands; do not assume the example has already been implemented. For color-only mode, use `--mode sh_only` and verify literal frozen-geometry behavior. Never start this command from Phase 4's `--stage all`.

### Final instructions for Codex

> Keep Phase 5 independent of Phase 4. Accept only a complete immutable Phase-4C scene; optimize its canonical Gaussian representation under exactly the fixed Phase-3 joint and fixed material assignments using the native RigEuler CUDA RGBD-alpha renderer. Provide distinct `gaussian_rgbd` and literal `sh_only` modes with exact optimizer freezes and joint-unchanged proof. Do not modify the joint, camera poses, Phase-4 material inference or earlier outputs. **Do not implement or execute Phase 5 until the user explicitly authorizes it after Phase 4 validation**.
