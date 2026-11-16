# CODEX DIRECTIVE — REPAIR PHASE 4, NOT JUST ITS RUNTIME

**Target checkout:** `~/Joint3DGS/RigEuler3DGS-p45` (`phase45` worktree).  
**Upstream baseline:** `6807d83`; never modify the main checkout, datasets, Phase 0–3 sources, outputs, or installed legacy CUDA extension.  
**Scope:** Correct Phase 4A/4B/4C, particularly exact candidate evidence, visibility and scaling. Do **not** modify/run Phase 5 or connect/run optional joint refinement.  
**Evidence:** `PHASE45_IMPLEMENTATION_REPORT.md`, `phase45_repo_audit.md`, and the November 11 overnight diagnostic. The worktree is uncommitted; re-audit the actual current local diff first. Do not assume this directive itself proves the implementation.

## 0. The immediate failure and rules of engagement

On `red_fridge`, current run has **28,089 original Gaussians, 378 views, 640×480**. At ~13 hours, only `manifest.json` and `config_resolved.json` existed. Phase-4 process was using ~1167% CPU and ~282 MiB VRAM; `/proc/<pid>/io` reported ~3 TB `rchar` versus ~44 MB physical disk reads. Logs were buffered by `tail`. Source inspection shows `ViewStore(cache=64)` cycling through 378 views, and `material_completion.py` multiplying 48-view screens + halves + OMIT + 378-view rescoring over many components, eight rounds and three verifications. The cached-byte read count strongly suggests thrashing; **instrument actual decode counters** to prove it.

**Do not relaunch the 13-job overnight runner. Do not run Phase 5.** First verify that the runaway job is no longer active. If still active, show its command line and request termination of **only its exact PIDs**, not unrelated `track_2part.py`/FreeArt work. Example, only after checking `ps -p 589317,589256,589252 -o pid,etime,args` and confirming those PIDs still belong to this job: `kill -TERM 589317 589256 589252`. PID reuse is possible, so never blindly execute this historical command. Avoid broad `pkill -f` patterns.

**Success means**: a mathematically justified Phase-4 physical inference algorithm that scales to the actual sequence, conservatively reports uncertainty and preserves every required observation. A time cap, coarse view subset, or cheap approximate score is **not** proof of exact acceptance.

## 1. Required preflight: audit and measure before editing

Inspect these files/functions on the current worktree, showing relevant call paths and current data semantics:

- `recovery/material_completion.py`: `build_components`, `informative_views`, `BaselineCache`, `flip_scores`, `omit_check`, `run_phase4a`, sensitivity, verification.
- `recovery/views.py`: `ViewStore.get`; `recovery/io_setup.py` cache construction and view order.
- `recovery/articulated_scene.py`: transport and `render`; `recovery/rgbd_loss.py`: full-image reductions, SSIM window, depth validity, background and support denominator.
- `recovery/residual_discovery.py`, `recovery/canonical_fusion.py`, `recovery/candidate_acceptance.py`: dependencies on 4A and all pixel thresholds; `recovery/phase4.py` stage control.
- `recovery/gaussians.py`, `recovery/topology.py`: persistent IDs, provisional material states and original lineage.
- `gaussian_renderer/__init__.py` and opt-in `rasterize_gaussians_rgbd`: projection, conic footprint, alpha cutoff, tile overlap, front-to-back order, transmittance termination, gradient behavior; AA must remain disabled.
- Existing utilities: `utils/gaussian_sequence_assignment.py`, `utils/kpart_counterfactual_rendering.py`, `rigid_parts/outer_render.py`, and `estimator/surface_bootstrap.py` — reuse only after verifying conventions and actual mathematical equivalence.

First add deterministic timing/counter logging around all code paths, with `flush=True` or unbuffered `tee`, never `| tail` as the sole stdout consumer. Emit counts of original/core/noncore Gaussians, candidate groups, views, decoded views, cache hits/misses, full renders, ROI renders, candidate score calls, CPU/GPU transfers, bytes copied, loss evaluations, accepted/rejected/ambiguous/untested groups, per-stage elapsed CPU/wall times, peak CPU RSS and CUDA memory. **No future run may spend an hour without a measurable progress record.**

Freeze a regression fixture from the current small procedural scenes **before** changing their scoring. Save full-render per-hypothesis reference losses, accepted IDs, final statuses and full-frame residual maps as an independent correctness oracle.

## 2. Nonnegotiable physical and observation contracts

Input: **one selected direction**, one canonical endpoint, Phase-2 confident moving core, explicitly chosen Phase-3 joint type and its fixed geometry and trajectory; original-frame ID map. Ensure `q_anchor=0`, correct m/rad units, valid cameras and masks, calibrated RGB/depth registration, no GT access during inference. The Phase-3 joint, all `q_t`, and moving-core membership are fixed through all of Phase 4 unless a preflight invalidity is reported. **No rerun in the opposite direction**.

Canonical Gaussians are `G_i^0=(mu_i,R_i,scale_i,opacity_i,SH_i,uid_i,material_i)`; preserve stable IDs and source provenance. Let `z_i∈{STATIC,MOVING,UNRESOLVED}` be epistemic status. `UNRESOLVED` is **not a third physical transform**; its provisional rendering pose/label must be explicitly tracked separately. Moving Gaussians obey only the Phase-3 relative rigid joint motion, for the *entire* sequence; static Gaussians are exactly stationary.

For each frame `t`, the prismatic transform is `Q_t=I`, `p_t=(s_t-s_anchor)e`. Revolute is `Q_t=Exp((theta_t-theta_anchor)[e]_x)`, `p_t=(I-Q_t)a`. For moving Gaussians `mu_i(t)=Q_t mu_i^0+p_t`, `Sigma_i(t)=Q_t Sigma_i^0 Q_t^T`; rotations and canonical SH/view-direction conventions must be consistent with the existing renderer. Always render **one full physical static+moving scene**, never independent opacity-weighted overlays.

Images `I_ct`, metric camera-Z depth `D_ct`, binary complete-object mask `M_ct`, camera K/T are immutable. Foreground depth valid iff `M=1`, positive, finite and registered. `M=0` creates NO depth observation. Invalid depth creates NO 3D birth. The complete silhouette still contributes to alpha/mask loss including background pixels. Original pixel-center `(u+0.5,v+0.5)` and projection must match the CUDA renderer.

## 3. Make the observation path O(number of views), not O(number of candidates × views) decodes

`ViewStore(cache=64)` is catastrophic with a 378-view round-robin sweep. Implement **one decode/preparation per source view** for this phase, with explicit measured memory limits:

1. Decode RGB as compact CPU `uint8[H,W,3]`, depth as `float32[H,W]` **in meters** (do not reduce to float16 silently; it loses useful precision), mask as `uint8/bool[H,W]`; compute valid-depth bitmap once. Camera matrices, pixel registration and frame IDs are immutable.
2. If enough host memory, preload the actual sequence: approximate raw data at 378×640×480×(3 RGB bytes+4 depth bytes+1 mask byte) ≈ **0.87 GiB**, excluding metadata, validity and overhead. Confirm real RAM usage. Otherwise build a hash-checked memory-mapped unpacked view store, then **process candidates in view-major batches** so each source PNG is decoded at most once per stage. Do not let cache capacity silently force cyclic re-decode.
3. Never preload 378 full-resolution float32 RGB/depth/mask tensors on an 8-GB GPU, especially while `track_2part.py` occupies VRAM. Stream one or a few views and needed tile/ROI buffers to the GPU; optionally pin host memory. Use an explicit GPU memory target and report actual peak.
4. Cache baseline rendered/target **per-pixel loss contributions and/or compact image channels** in CPU RAM or a memory map as appropriate. Keep on GPU only the active view's baselines and ROI data. No full H×W loss maps copied GPU→CPU per candidate; reduce to compact scalar/channel diagnostics on GPU, transfer those only.
5. Assert source view-decode counts and bytes read. Add a 378-view sequential repeated-sweep test demonstrating no re-decoding after first preparation.

Do not claim that a cache-size increase alone fixes the algorithm. It only removes a multiplier.

## 4. Phase 4A: exact conditional static-vs-moving model selection

### 4.1 Score the correct **complete-scene** physical hypotheses

Let `G_-B` be the current scene excluding candidate group `B`. Define `G_S(B)=G_-B∪B_static`, `G_M(B)=G_-B∪B_moving`, `G_omit(B)=G_-B`. Hold all other original Gaussians, SH, opacity, shapes, joint geometry and `q` fixed. Under the full native compositing renderer, for camera/frame `v=(c,t)`:

`Yhat_v(G)=(Ihat_v, Ahat_v, Zhat_v / max(Ahat_v,eps))` with `Zhat_v=Σ_i w_i z_i`, `w_i=alpha_i(p)∏_{j<i}(1-alpha_j(p))`, and `Ahat_v=Σ_i w_i`.

Define *additive, full-image-normalized* physical image loss, with observed-mask validity fixed independently of the candidate:

```
L_v(G) = λrgb L1[M⊙Ihat, M⊙I]/(3HW)
       + λssim (1-SSIM(M⊙Ihat, M⊙I))
       + λmask Σ_p |Ahat-M|/(HW)
       + λdepth Σ_{p∈V} depth_penalty(p,G) / max(1, |V|)
```

Careful: `V` = valid observed foreground depth, never predicted alpha support. To prevent avoiding depth error by making `Ahat` small, either (a) mask depth by a documented fixed observed-valid support and use a stabilized physically meaningful depth + explicit low-alpha coverage penalty, or (b) retain the existing low-alpha exclusion **only** with an explicit missing-coverage term and count; make the depth support denominator fixed for exact incremental differences. Report all channel terms separately. Low-alpha regions cannot be silently called depth-matched.

`E_S(B)=Σ_v L_v(G_S(B))`, `E_M(B)=Σ_v L_v(G_M(B))`, `E_omit(B)=Σ_v L_v(G_omit(B))` using the same set of cameras/frames and same loss definition. `Δ_B=E_S−E_M` positive favors moving. The count/complexity is the same for 4A S versus M. OMIT is only a **diagnostic** to flag a bad existing original Gaussian, not permission to prune the original without an explicit upstream review.

**Do not divide the candidate score by a hypothesis-specific affected-pixel count.** This would bias a small footprint versus a large footprint. All loss denominators are fixed full-view observation counts. For reporting, also log affected-pixel mean as a diagnostic, never replace the objective.

### 4.2 Exact impact region instead of full-image full-scene renders per group

For fixed view `v`, candidate `B` can change the image only on pixels receiving any nonzero renderer contribution from `B` in either `S` or `M` hypothesis. Construct a **conservative** union:

`P_v(B) = footprint_v(B_static) ∪ footprint_v(B_moving)`.

Compute footprints using the *same projected conics, front-to-back visibility and rasterizer alpha support conventions as the production CUDA kernel*: Gaussian extents, near/far clipping, tile bounds and alpha threshold (`alpha<1/255` skipped; renderer AA off). **Do not use projected means plus a fixed-radius crop or just the Gaussian center.** Include every tile intersected by either hypothesis and its entire required contributor list. If conservative footprint computation cannot be proven yet, fall back to full frame for that candidate, not a guessed crop.

Per-pixel RGB/alpha/depth error outside `P_v(B)` is invariant. SSIM is local but has a kernel footprint; changed SSIM centers lie within the Minkowski expansion of `P_v(B)` by the SSIM radius. Compute SSIM using the original **full-image pixel coordinates and unchanged neighborhood/context pixels**. Output-comparison region and source sampling halo are different sets. Track the spatial support exactly, especially image edges and large anisotropic splats.

Therefore the candidate difference can be evaluated exactly by recompositing only potentially affected tiles/pixels **with all scene Gaussians that overlap them**, not rendering `B` alone. A new GPU `scissor`/tile-mask mode inside the opt-in renderer is acceptable if the original viewport/projection, tile lists, depth sorting, compositing and stopping rules are identical. Alternatively use a verified cropped render whose full-image projection and boundary neighborhoods are analytically identical. **An approximate crop is forbidden for final decisions.**

With fixed per-view normalizations and finite-support SSIM:

`L_v(G_M)-L_v(G_S) = Σ_{p∈affected} [ell_v(p,G_M)-ell_v(p,G_S)]`,

where the SSIM-dependent affected set includes its output halo, and per-pixel `ell_v` already incorporates the original global denominator. Each candidate should return one small score vector and visibility/contradiction counts, **not a full-image tensor moved to CPU**.

Cache initial complete-scene baseline once per view. After committing a candidate, update/recompute only changed tiles in impacted views, invalidate affected precomputed scores, and increment a scene-version counter. Fully re-render the final accepted scene across **all** views once for a global audit. If exact incremental baseline maintenance is not available, perform a limited batch of independent candidate proposals against a frozen scene, commit a non-overlapping set, then rerender the whole baseline once per batch. Do not use stale baselines as though they described the newly committed scene.

### 4.3 MANDATORY exactness oracle before using ROI results

Implement independent full-scene references `E_S_full`, `E_M_full`, and exact cropped/scissor counterparts `E_S_roi`, `E_M_roi` on deterministic fixtures. Assert:

- Pixelwise RGB/A/Z/metric-depth in changed region agree to the renderer's existing float32/atomic tolerance.
- Every unchanged pixel outside the conservative region is truly equal in both full hypotheses (not just visually similar).
- All-channel **global normalized** loss differences agree: `abs((E_M_roi-E_S_roi)-(E_M_full-E_S_full)) ≤ tolerance` using both absolute and scale-relative tolerances declared from existing CUDA numerical floors, not arbitrary `1e-2`.
- Forward and backward direction; occluded / visible late geometry; image boundary; same-tile competing surfaces; thin two-sided layers; very large projected Gaussian; low alpha cutoff; moving/static tie; SSIM halo; background, invalid depth, alpha threshold crossing, sorted-depth change.
- For 4C fitting, gradients through ROI must match full-frame gradients on candidate parameters (finite differences where appropriate), or use full-frame optimization for 4C until proven.

Create `tests/test_phase4_exact_roi_parity.py` and retain the all-frame full oracle for regression tests. **No performance optimization is acceptable if it changes classification or loss beyond measured numeric tolerance.**

### 4.4 Design candidate scheduling to avoid combinatorial work

Build physically coherent connected groups from canonical proximity, surface normal compatibility and **same depth layer**, without crossing drawer/door articulation boundaries. Do not group opposite faces by naïve Euclidean kNN. Core points are fixed moving seeds; original noncore labels are unknown, not certified static.

**Screening** (safe, cheap): determine projected motion magnitude and any overlapping observed foreground under the fixed Phase-3 joint; estimate whether `B` could be distinguishable. Rank with existing `utils/gaussian_sequence_assignment.py` or residual/visibility summaries only after auditing geometry conventions. Use a deterministic, view-balanced shortlist for prioritization; sample motion-extreme and canonical views, plus cameras with informative visibility. A group that has no discriminative observations is `UNRESOLVED`, not `STATIC`. A weak preliminary score cannot irreversibly reject moving or static.

**Final assessment of each potentially decidable group:** accumulate exact incremental loss differences over **all 378 relevant original views**, with early skipping only when *provably zero-impact* (offscreen, zero support, or identical compositing), NOT via arbitrary view caps. Log numbers of actually affected views/tiles and the reason for each skipped view. Expensive full-scene renders must not be repeated for each group.

Compare S/M with robust stability diagnostics across time blocks/cameras, realistic observation noise/model error, and independent views when available. No fake calibrated `p_moving` from arbitrary margins. Retain `UNRESOLVED` for a near-axis or unobservable group. Log `E_S`, `E_M`, `E_omit` (if tested), affected pixels per view, evidence signs over time/cameras, current baseline state and threshold versions. Use a clear, conservative *provisional* policy for uncalibrated margins rather than conflating decisiveness with probability.

**Updates:** prefer one deterministic global candidate ranking and a monotonic priority queue of improvements; after a committed group changes the scene, invalidate/re-evaluate only spatial/render-overlapping candidate neighbors. Avoid the existing 8 unconditional rounds × 3 full verification passes × multiple bisection renders. Split a group only if it is demonstrably mixed and splitting creates separable evidence; never recursively split every ambiguous group by default. Terminate on no further supported improvements or explicit declared resource stop, with unprocessed groups `UNTESTED_RESOURCE_LIMIT` (not silently static) and unresolved count visible. A final **global scene** audit must still evaluate every original camera/frame.

**Joint sensitivity:** perturbations of J/q affect **all moving Gaussians simultaneously**; evaluate globally coherent perturbations, not an incorrect single-component-only perturbation. Do not execute 12×24×components expensive sensitivity sweeps. Perform joint-validity check on the confident core and one small, declared representative set of fragile/accepted groups, with batched/shared perturbation caches; if evidence is uncalibrated, export `JOINT_CONFIDENCE_UNCALIBRATED`. This check cannot silently change the fixed Phase-3 joint.

## 5. Phase 4B: missing geometry must be genuinely *observed*

Only start 4B after 4A finishes its exact-scoring and final scene audit. Render the accepted original-Gaussian scene over all cameras/frames once, producing `RGB`, `A`, normalized `D`, and optional contributor/layer maps. Compute residual/visibility classes **once per source view**. Proposal eligibility must respect strict `M`, depth validity, calibrated camera geometry, and real depth uncertainty. Use `utils/noise/sensor_model.py` where available; the current config's `sigma_a=sigma_b=0`, fixed `depth_tol=0.01`, `sigma_depth=0.005`, `layer_tol=0.008` and `depth_jump=0.03` are *assumptions*, **not sensor calibration**. Do not claim accurate millimetre surface reconstruction from these values.

Pixel classes should include: already explained; genuine uncovered foreground; nearer observed surface requiring scrutiny; observed depth behind current model (do not automatically birth); mixed-depth/silhouette edge; invalid depth; background. Do not infer missing geometry solely from `expected_depth` mixing front and back layers. Foreground edge bias in real RGB-D is a first-class error mode: avoid interpolating across edges, use original depth samples, and keep persistent residuals near discontinuities separate until verified by multiple valid observations.

**Important 4A→4B coupling:** unresolved original Gaussians have alternate plausible S/M labels. Before declaring a pixel `COVERAGE_GAP`, consider whether one of those alternate assignments could explain it. A gap attributable to an unresolved original should be marked `ORIGINAL_MATERIAL_AMBIGUOUS`, not used to create a duplicate new moving splat. Do not hard-code all unresolved as truth-static and then create moving duplicates elsewhere.

Back-project a measured axial depth pixel `d` in camera `(c,t)`:

`y_C=d K_c^-1 [u+0.5,v+0.5,1]^T`, `y_W=R_WC y_C+t_WC`.

Candidate canonical coordinates: `x_S=y_W`, `x_M=Q_t^T(y_W-p_t)`. Both are *hypotheses* until selected; the moving one must exist at every frame, including when initially occluded. Propagate the camera-depth covariance through the back-projection and inverse rigid transform; keep separate **measurement uncertainty**, **shared joint uncertainty**, and **rendering covariance**. **Do not** use the 3DGS footprint covariance as a sensor-error Mahalanobis covariance. The existing `explained_mahal=3` must be audited and relabeled or replaced accordingly.

Build distinct front/back oriented surface layers and robust patches across cameras and separated times; use canonical geometry alignment, compatible normals/appearance, visibility and same-layer mutual consistency. The fitted normal scale of an anisotropic Gaussian is a *surface thickness/render footprint*, not a free uncertainty knob. Generate a compact patch population rather than a Gaussian per depth pixel. Record exact source camera/frame/pixel lineage and derived noise model.

Never claim recovery of a surface not observed in any RGB-D frame. Do not mix surfaces of different rigid bodies or opposite sides of a thin panel. Emit ambiguous/rejected observations with reasons rather than hallucinating a label.

## 6. Phase 4C: birth acceptance must be precise, independent where feasible, and complete

Before 4B proposal generation, assign immutable view/pixel role masks: **PROPOSAL**, **FIT**, **VALIDATION**. Where data permit, separate by non-overlapping spatial blocks, times and cameras, retaining informative views of rare-visible surfaces. Spatial separation must exceed the local fitting and SSIM receptive fields; rows-with-even/odd parity are not independent. The Phase-3 joint is already data-fitted: label validation `conditional_on_phase3_joint` truthfully. No candidate may be generated from its own supposed held-out pixels.

For each proposed physical patch `B`, pair the source-equivalent hypotheses `H∈{OMIT,STATIC,MOVING}`. Optimize **canonical Gaussian parameters only**, for the same declared limited refinement schedule under fixed J/q; no per-frame positions. For S and M, fit separately on FIT views (identical initialization budget). Score against OMIT and each other on held-out VALIDATION pixels with fixed full-image denominators, complexity/redundancy penalty and physical visibility contradictions. If independent validation support is missing, mark `PROVISIONAL_INSUFFICIENT_EVIDENCE` rather than pretending to accept based on the fitting views.

**Run exact physically composited candidate evaluation over every original frame/camera** for each finalist, using the proven ROI method and complete occluder list. Fully occluded or outside-frustum frames are neutral, not negative observations. A candidate that materially violates a real earlier visible surface must be rejected even if its discovery frame improves. Accept only if held-out evidence supports it and the all-view audit has no unacceptable contradiction; record why under declared, validated error scales. Reconsider interacting candidates together and rerender the **combined final accepted model** on all views to catch collective occlusion/duplicate errors.

A new Gaussian is a physical material element, **not born at a later timestamp**. Its anchor coordinate and material state are permanent and its world pose at every timestamp is fixed by the joint if moving.

Prevent duplicate births by testing overlap against (a) original material alternatives, (b) already accepted new surfaces, and (c) existing visible depth layers under both hypotheses. The crude current `dup_tol=0.003`, `dup_frac=0.5`, `contradiction_frac=0.34` must be reported as uncalibrated until tested under real noise and varied panel thickness. No arbitrary 34%-contradiction allowance should be interpreted as physical permission to occlude real measurements.

## 7. Engineering and logging requirements

- Keep `recovery/*` and opt-in worktree-only CUDA build; old entrypoint bitwise compatible and original Phase 0–3 extension untouched. No new Nerfstudio/FreeArtGS backend.
- Add a stage CLI: `--stage 4a|4b|4c|all`, with independent resumeable checkpoints and immutable upstream hash proof. `4a` is the default for verification until Gate 4A passes; `all` must explicitly be requested. Never fall through into Phase 5. Support `--dry-run` showing exact planned resources and view count.
- Persist `cache_manifest.json` (decoded/prepared view counts, checksums, memory), `score_manifest.json` (scoring exactness, masks/denominators, accepted mode), and an atomic `checkpoint.json` for resumability. Hash input PLY, support IDs, Phase-3 geometry/trajectory, camera metadata, selected original frames, source masks and relevant loss/policy config; refuse stale cache reuse.
- Log line-buffered to `tee` with complete stdout + stderr, deterministic config + random seed, phase/round/group count, accepted/ambiguous status, cache hit ratio, valid views, ROI pixels, render counts, frames checked, per-stage seconds, RAM/GPU memory every configurable number of components or wall-seconds.
- Do not overwrite partial prior results. Use distinct `run_id` and output directory; checkpoints must be versioned and atomic, restore moving-label/scene-version consistency, and make a stopped run visibly incomplete.
- Retain original Gaussian IDs and canonical geometry for unresolved; only mark outcome when actually evaluated. `status` values must distinguish `STATIC`, `MOVING`, `UNRESOLVED`, `UNTESTED_RESOURCE_LIMIT`, `SUSPECT_ORIGINAL`, `UPSTREAM_JOINT_INVALID`, and rejected/ambiguous births. Do not conflate omitted with static.
- If projected ROI optimization is not yet correct, implement and measure a slower but correct bounded **benchmark of 50 fixed components**. Do not implement a misleading production `all` mode with truncated scoring.

## 8. Test gates (run in order; stop on failure)

### Gate A — instrumentation and caching
- On an actual 378-view sequence, repeated access after a preparation sweep shows no repeated PNG decode (or, for view-major streaming, bounded documented per-stage loads). Count `ViewStore.get`, cache misses, `Image.open`, `np.load` and IO bytes; demonstrate the 64-view thrash is gone.
- Demonstrate bounded GPU memory with the actual 8-GB machine and no assumption that the other GPU process is idle. Report RSS, peak allocated/reserved VRAM, time per 100 candidates and estimated total score operations from measured counts, **not guessed wall-time**.

### Gate B — exact ROI renderer and objective
- `tests/test_phase4_exact_roi_parity.py`: synthetic and one real image; S/M/OMIT, all 4 losses and SSIM; compare ROI versus full-render score delta and candidate geometry gradients if applicable. Every pixel outside the projected impact+SSIM halo is unchanged.
- Run a deterministic 50-group reference: old full-scene scorer and new exact ROI scorer must agree on scores/statuses to declared numerical tolerance. Include occlusions, cross-layer changes and near-axis unresolved.

### Gate C — physics and Phase 4A labels
- Existing procedural R/P withheld-group tests still pass with exact local evidence; no false static→moving; legitimate near-axis remains `UNRESOLVED`.
- 4A on red-fridge **small fixed representative view subset only as a benchmark**, fully marked `NOT_FINAL`, to obtain measured per-group cost before its full 378-view production run. Then one complete 378-view 4A; export all-view evidence and final scene audit before 4B.
- Test provisional unresolved Gaussian producing spurious 4B gap. Must not create its moving duplicate.

### Gate D — real depth and 4B/4C
- Synthetic two-sided panels at 3, 5, 8, 12, 20 mm and oblique angles/noise, plus calibrated and systematically biased depth; inspect false birth and layer merge rates.
- Drawer floor and door back recover without front/back collapse; wrong-side geometry rejected by earlier frames; never-seen surface never invented.
- Candidate PROPOSAL/FIT/VALIDATION provenance checked by code; even/odd rows alone forbidden as independence claim; independent support reported per candidate.
- One synthetic full-sequence complete Phase 4, then **one** selected real sequence complete Phase 4 before any multi-scene batch. Per-camera/frame depth residual, silhouette, coverage, accepted/rejected/unresolved counts, occlusion contradiction map, runtime, VRAM, CPU RSS, source hashes.

### Gate E — compatibility
- Existing 8 native CUDA test suites and opt-in RGB-D renderer tests pass. Legacy rendering outputs remain unchanged within their documented baseline and main extension is not touched. Phase-3 joint hashes and q arrays unchanged byte-for-byte. Phases 4B/C and Phase 5 must NOT start if Gate B fails.

## 9. Codex execution order and deliverables

**Implement in this order only:**

1. Inspect worktree + reproduce/measure a tiny representative 4A slice; collect a profiler/counter report and reference numeric scores.
2. Correct caching, data residency, GPU/CPU transfer, progress logging; write Gate-A tests.
3. Implement exact ROI material-hypothesis rendering and additive loss-delta equivalence, including SSIM halo. Write and pass Gate B. No unexplained approximation in final scoring.
4. Replace the exhaustive 4A component × views × full-frame loop with screening + exact ROI all-view scoring, targeted invalidation and deterministic scheduling. Maintain conservative unresolved states and global final verification. Pass Gate C on synthetic; benchmark measured per-group cost on real.
5. Correct 4B residual eligibility, potential duplication from unresolved originals, sensor covariance semantics and canonical layer fusion. Pass synthetic noise/thin-panel tests.
6. Correct 4C proposal/fit/validation split, exact nonlinear all-view acceptance, global final scene audit. Pass Gate D synthetic.
7. Run **one** real scene Phase 4 end-to-end; report physics, resources and any failure before proceeding. Do not launch the 13-scene overnight batch and do not run Phase 5. Ask for review of Phase-4 results.

Deliver in the worktree:

- `PHASE4_CORRECTION_AUDIT.md`: root cause with measured counts, math, specific code diffs and previous incorrect assumptions.
- `PHASE4_EXACTNESS_REPORT.md`: full-vs-ROI per-channel loss parity and Gaussian gradient tests; any unsupported cases.
- `PHASE4_PERFORMANCE_REPORT.md`: decoded views, render calls, ROI pixels, throughput, peak memory, ETA *from a measured sample only*, not from guesswork.
- `PHASE4_REAL_VALIDATION_REPORT.md`: per-view residuals and evidence with all unresolved/rejected status and no GT used for inference.
- A minimal single-scene command for the user's existing scene paths, with unbuffered logging and no implicit downstream stages.
- A final file/commit list and full test commands/results. Leave changes **uncommitted until explicitly instructed**; do not touch main or running environments.

**Hard stop:** If you cannot prove local scoring equivalence or if the real sequence remains computationally intractable, report the precise failing gate and measured bottleneck. Do not substitute a cropped heuristic, early-exit cap or 48-view summary and label it complete. The purpose of Phase 4 is to recover physically justified geometry, not to make the overnight script terminate.
