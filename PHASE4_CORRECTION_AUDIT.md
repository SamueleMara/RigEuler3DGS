# Phase 4 correction audit

Date 2026-11-11. Directive: `RigEuler3DGS_PHASE4_CORRECTIVE_CODEX_DIRECTIVE_2026-11-11.md`.
Worktree `~/Joint3DGS/RigEuler3DGS-p45` (branch `phase45`, base `6807d83`, uncommitted). Phase 5 was neither changed nor run. The optional joint refinement was not connected. The main checkout, the datasets, the Phase 0–3 outputs and the CUDA extension were not touched in this correction; the rasterizer sources are unchanged since the 2026-11-10 build.

## 1. The failure, measured

### 1.1 The overnight run

| quantity | value | source |
|---|---|---|
| scene | red_fridge, 28,089 Gaussians (3,474 core), 2 cameras × 189 frames = 378 views, 640×480 | upstream contract |
| state after 13 h | only `manifest.json` and `config_resolved.json` | output directory |
| process | 1167 % CPU, 282 MiB VRAM, `rchar` 3.0 TB vs `read_bytes` 44 MB | `/proc/589317` |
| log | empty, because Python's buffered stdout went to `\| tail -25` | `~/phase45_real.log` |
| state on re-audit | the runaway job was no longer running, so nothing was killed | `pgrep` |

### 1.2 Per-operation costs (red_fridge, `tools/benchmark_phase4a.py`)

| operation | cost |
|---|---|
| view decode (PNG + `.npy` + mask) | 11.4 ms |
| full-scene RGB-D render | 0.95 ms |
| per-pixel loss map | 0.5–0.8 ms |
| full view loss with SSIM | 2.9 ms |

### 1.3 Decode thrash

`ViewStore(cache=64)` against sequential sweeps over 378 views:

| store | work over two sweeps | time |
|---|---|---|
| old `ViewStore(cache=64)` | 756 gets, **0 hits, 756 decodes** | 7.8 s |
| `PreparedViews` | 378 decodes (557 MB read) once, **0** afterwards | 3.4 s once |

### 1.4 Work per component in the old 4A

Each component paid for:
- 48 screening views;
- 2 × 48 views for the halves check;
- 48 views for the OMIT check;
- **378** views for the acceptance re-score.

Each of those was a full render plus full-image loss maps copied to the CPU, plus a decode on every miss. On top of that:
- up to 8 rounds and 3 verification passes;
- 12 × 24 sensitivity renders per decided component.

This cost about 6.6 s per decided component per round on 1,968 components, which is days of work. Even on the 1,100–1,250-Gaussian, 24-view procedural gate scenes the old 4A took 110–146 s (frozen fixture `tests/fixtures/phase4_pre_correction/summary.json`).

### 1.5 Grouping

With `link_factor = 1` the component linker left 1,487 of 1,968 components as singletons.

## 2. Pre-correction assumptions that were wrong

1. **A cache-size problem.** It was a structural multiplier. The fix is one decode per view, with work organised view by view.
2. **The decision statistic was the objective.** It was not:
   - `delta_aff` summed `pixel_loss_map` without SSIM;
   - its depth term was normalised by the hypothesis-dependent support `A ≥ α_min`, so a hypothesis could lower its depth error by leaving pixels uncovered;
   - decisions used a 48-view screening subset.
3. **Joint sensitivity per component.** It perturbed the joint for one component at a time and re-rendered 12 × 24 views each. That is neither globally coherent nor affordable.
4. **`explained_mahal` (4B)** used the 3DGS footprint covariance `R S² Rᵀ + σ² I` as if it were sensor error.
5. **Independence.** 4C called even/odd processing rows "validation", but they are not independent.
6. **Contradiction allowance (4C).** A 34 % fraction of views was allowed to get worse.
7. **Found during this correction:**
   - **Missing geometry absorbed by relabelling.** With the drawer floor removed, the old (and the first exact) 4A labelled the static back wall MOVING. Only the depth channel favoured it (+0.085): moving the wall forward fills the floor hole. Both appearance channels opposed it (L1 −0.007, SSIM −0.004).
   - **4C duplicate test** compared only against Gaussians of the same material. Canonical coordinates are label-independent: at the anchor frame both materials share one world frame. As a result a STATIC copy of the core-moving drawer front was accepted.
   - **One-to-one S/M patch pairing** left static fragments of a moving surface unpaired, so they were tested alone.
   - **The 20-step 4C fit** moved patch positions about 2 cm off the measured surface.

## 3. What replaced it

### 3.1 Observation path (`recovery/view_cache.py`)

- Every source file is read once (`bytes → sha256 → decode from memory`) into compact host arrays: uint8 RGB, float32 metric depth, bool mask and validity.
- That is 1.04 GiB for red_fridge. Beyond `--max-host-gb` the arrays go to a hash-named memory-mapped spill.
- Views go to the GPU one at a time. Counters are written to `cache_manifest.json`.
- Access before `prepare()` is refused.
- Phase 5 still uses `ViewStore` (unchanged except for diagnostic counters).

### 3.2 Exact objective (`recovery/exact_objective.py`)

```
L_v = Σ_p  λrgb(1−λs)·Σ_c|M(Î_c−I_c)|/(3HW)  +  λrgb·λs·(1 − SSIM_p)/(HW)
         + λmask·|Â−M|/(HW)  +  λdepth·[p∈V]·pen(p)/max(1,|V|)
pen = ρ(Z/(A+1e−6) − D) if A ≥ α_min, else depth_miss_penalty   (declared 0.05, uncalibrated; counted)
```

- `V` is the observed valid foreground depth, fixed for the view.
- Every denominator is fixed, so `L_v(a) − L_v(b)` is a plain sum of per-pixel differences.
- SSIM uses fixed-order float64 shifted sums, so a crop and the full image produce bitwise-identical values.
- Phase 5's `rgbd_loss.py` is unchanged; it keeps its own support-normalised depth term.

### 3.3 Exact footprints (`recovery/footprint.py`)

- The kernel lists Gaussian i only in tiles inside `rect_i`, computed from the CUDA radius and the projected mean, so pixels outside `rect_i` are bitwise invariant to i's pose or presence.
- The radius comes from the kernel. The mean is re-projected in torch, and every rect is padded by one tile to absorb float rounding.
- The ROI is the union of both poses, dilated by the SSIM radius, so the per-view delta is exact (`PHASE4_EXACTNESS_REPORT.md`).

### 3.4 4A (`recovery/phase4a_exact.py`)

- **Core check:** the Phase-2 core as one exact all-view group; it must be MOVING, otherwise UPSTREAM_JOINT_INVALID.
- **Groups:** layer-aware linking at the kernel's 3σ extents (`link_factor 3`), `max_component 128`. On red_fridge this gives 288 groups instead of 1,968.
- **Scoring:** view-major sweeps. Each view is uploaded and its baseline rendered once per sweep; every group's flip is then rendered in full and scored exactly on its ROI. Views are skipped only when provably zero-impact (culled in both poses, or identity transform).
- **Decision** (declared provisional policy, not probabilities):
  - |Δ| in pixel-equivalent units ≥ `decisive_abs_px`, and consistency |ΣΔ_v|/Σ|Δ_v| ≥ 0.25;
  - sign agreement ≥ 0.6 among the informative views **and** among camera × 16-row time blocks;
  - **new channel-agreement rule:** if appearance and geometry disagree in sign and the weaker is ≥ 10 % of the stronger, the group stays UNRESOLVED.
- **Monotone schedule:**
  1. Proposals go into render-interaction components (footprints dilated by one tile in any view).
  2. A singleton commits its own exact flip.
  3. A larger component is scored as one joint move; it commits if that exact move lowers E by ≥ the threshold, otherwise only its best proposal commits.
  4. Components do not interact, so the gains add. The realised change is checked against the prediction, and the run aborts if E ever rises.
  5. Only groups whose footprint meets a commit, plus uncommitted proposals, are re-scored.
- **Splits:** only for mixed-evidence groups, and only kept if the halves are separable (opposite decisions, or one decisive while the other is still mixed); depth ≤ 5.
- **Final verification:** every leaf group on every view with the OMIT diagnostic. This sweep is also the global scene audit.
- **Joint sensitivity:** whole-joint perturbations (axis ±1°, pivot ±5 mm, q ±2 %) on the scene objective and on the 16 least decisive groups. Ranges only: JOINT_CONFIDENCE_UNCALIBRATED.
- **Status bookkeeping:** statuses are CORE_MOVING, MOVING, STATIC, UNRESOLVED, UNTESTED_RESOURCE_LIMIT, SUSPECT_ORIGINAL and UPSTREAM_JOINT_INVALID. The render label of an undecided group is stored separately (`render_label.npz`).

### 3.5 4B (`recovery/phase4b_exact.py`)

- **Roles first (`recovery/data_roles.py`).** PROPOSAL / FIT / VALIDATION are assigned from 64-px spatial blocks × 8-row time blocks, with guard bands of `SSIM_R + 1` px so that no SSIM window crosses roles. Adjacent blocks differ, and every block cycles through all three roles. The roles are labelled `spatially_and_temporally_separated_conditional_on_phase3_joint`.
- **New class ORIGINAL_MATERIAL_AMBIGUOUS.** Gap or front pixels explained by an undecided original in its alternate pose are not birth sources.
- **Birth sources** are COVERAGE_GAP and FRONT_SURFACE pixels in the PROPOSAL role only.
- **Three uncertainties are kept separate:**
  - measurement, `P_y` propagated per point;
  - shared joint, not propagated and recorded as conditional;
  - rendering footprint, used only for a render-support containment test `(y−μ)ᵀ(Σ_g+P_y)⁻¹(y−μ) ≤ 3²`.
- **Sensor tolerances** are ASSUMED unless a calibrated sensor model is given.
- **Fusion.** Cells are keyed by (voxel, signed normal bin) and split along the normal at gaps > 8 mm. Patches link cells whose offsets agree within `max(layer_gap, 2·sqrt(t_i²+t_j²))`.
- **Source-equivalent candidates.** Every patch's STATIC and MOVING versions are fused from exactly the same source pixels.

### 3.6 4C (`recovery/phase4c_exact.py`)

- **Duplicates first,** in canonical coordinates and label-independent: an undecided original gives ORIGINAL_MATERIAL_AMBIGUOUS; a decided original or earlier birth gives REJECTED_DUPLICATE.
- **Fit:** FIT-role pixels only, full-frame rendering. Positions stay at the measured cell medians; appearance, opacity, scale and rotation are fitted.
- **Exact evaluation** over every view the patch touches: all-pixel delta, VALIDATION-pixel delta and validation support.
- **Occlusion contradictions:** valid observed-depth pixels, outside the observed depth-edge band, that the patch newly covers in front of the measurement. Offending cells are pruned, then the patch is refitted and re-scored (≤ 2 rounds).
- **Decisions:** PROVISIONAL_INSUFFICIENT_EVIDENCE, REJECTED_OCCLUSION_CONTRADICTION, REJECTED_NO_IMPROVEMENT, REJECTED_DUPLICATE, REJECTED_SOURCES_ALREADY_EXPLAINED, ORIGINAL_MATERIAL_AMBIGUOUS, AMBIGUOUS_MATERIAL and ACCEPTED_*. AMBIGUOUS_MATERIAL is also used when a one-sided test's sources do not separate S from M by at least 2 voxels, and when the better-supported material was stopped only by contradictions.
- **After the sequential pass,** each birth gets a leave-one-out re-check against the final scene, followed by a final all-view audit.

### 3.7 Stage control (`recovery/phase4_stages.py`, `tools/run_phase4_recovery.py`)

- `--stage 4a|4b|4c|all`, with 4a as the default. Each run gets its own `--run-id` directory, and existing runs are never overwritten.
- Atomic `stage4a/checkpoint.json` and `state.npz`. Resume is checked against the input proof (upstream sha256 plus the combined view checksum) and a config hash that excludes resource limits.
- Stage markers: 4b and 4c refuse to start unless the previous stage is complete. A run stopped by a resource limit is marked INCOMPLETE and stays resumable.
- `--dry-run` writes nothing. `--view-subset` produces NOT_FINAL benchmarks.
- Logging is line-buffered with resource stamps; sweeps report progress every 60 s.
- The runner scripts are Phase 4 only (`tee`, no `tail`). The multi-scene wrapper refuses to start without `CONFIRM_BATCH=1`.

## 4. Gate status

| gate | status | evidence |
|---|---|---|
| A caching / instrumentation | **PASS** | `tests/test_phase4_view_cache.py`; real 378 views: 378 decodes, 0 repeats; old cache: 756 / 756 misses |
| B exact ROI | **PASS** | `tests/test_phase4_exact_roi_parity.py` (4 tests incl. one real view); 50-group real parity 8.7e-19, decisions 50/50 |
| C physics / 4A labels (synthetic) | **PASS** | `tests/test_phase4a_exact_labels.py` (5): withheld recovery 1.00 / 1.00 (pre-correction 0.88 / 0.92), 0 false moving, on-axis UNRESOLVED, static controls abort, resume ≡ uninterrupted |
| C real 4A | **NOT RUN** (user's run) | 4-frame subset end to end only (NOT_FINAL); command in `PHASE4_REAL_VALIDATION_REPORT.md` |
| D 4B / 4C synthetic | **PARTIAL: one known failure** | `tests/test_phase4bc_exact.py`: 8 pass, `KNOWN_FAILING_drawer_birth_material_purity` fails (0.76 < 0.90) |
| E compatibility | **PASS** | rasterizer suites (9 + 1 legacy bitwise), contracts, transport, Phase 5 (5), disabled refinement (3) |

### 4.1 The failing part of Gate D

In the drawer scene, world-static mixed-depth fragments are accepted as STATIC births: 19 of 80 births, two candidates. Their sources come from frames 4 and 5, their static-hypothesis centroids coincide (z 2.095 / 2.092 m), and they sit about 1.8 cm off the true floor plane. These are alpha-blended expected-depth observations at the drawer edge. The 2–3 px edge band does not remove them.

### 4.2 Other weaknesses (reported, not hidden)

- **Same-facing parallel layers** closer than 20 mm are merged into one patch, because the noise-aware link tolerance absorbs them. Opposite faces never merge.
- **Coverage is partial:** 35 % of the drawer floor and 9 % of the door back face are within 3 cm of a birth.
- **4C runtime** at full scale has not been measured; the extrapolation is in `PHASE4_PERFORMANCE_REPORT.md`.

Per the directive's hard stop, 4B/4C should not be used for results until Gate D passes. Running 4A on one real scene is the next step.
