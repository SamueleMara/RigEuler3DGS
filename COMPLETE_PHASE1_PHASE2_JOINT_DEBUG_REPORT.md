# Complete Phase-1 / Phase-2 joint debug

Implements `CODEX_COMPLETE_PHASE2_DEBUG_AND_PARALLEL_PR_MODELS.md`, §45's
twelve steps in order. Steps 1–8 and 12 are done; steps 9–11 are the three
full real runs, which are handed over as a script (see §42 status at the end).

**No classifier was added** (§32). This produces evidence and stops.

---

## 1. Common dataset / configuration

One shared budget for every scene (§33 — a per-scene budget tuned until the
numbers look good is not evidence):

| | |
|---|---|
| frames | 60 — **every** dataset frame |
| cameras | 8 — **every** synchronized view |
| Gaussians | **all of them** (7 206 – 39 405 depending on scene; no subsampling) |
| rendered camera | 1, **every** frame (§23) |
| held-out folds | 2 (8 cameras → a clean 4 train / 4 held-out split) |
| iteration caps | Phase 1 12, screw 10, P/R 5 |
| frame staging | host RAM (60×8 RGB-D pins ~2.4 GiB; the card has 8 GiB) |

### Why those iteration caps — the measured budget

Measured on this box at the real scale (storage, 39 405 Gaussians × 8 cameras):

| | |
|---|---|
| Phase 1 | 6.0 s / frame |
| screw | **3.35 s per (frame × iteration)** |

Cost is ~linear in Gaussian count, so summed over the five scenes one
frame-iteration costs **8.07 s** (storage 3.35, fridge 2.38, USB 0.96,
stapler 0.77, blade 0.61). At 60 frames the fit cost for all five scenes is

    484 s × (screw_iters + 2 × folds × pr_iters)

| config | fit (upper bound) | + ~0.9 h overhead |
|---|---:|---:|
| 3 folds, screw 20, P/R 20 | 18.8 h | **19.8 h** |
| 3 folds, screw 12, P/R 8 | 8.1 h | 9.0 h |
| 2 folds, screw 12, P/R 8 | 5.9 h | 6.8 h |
| **2 folds, screw 10, P/R 5** | **4.0 h** | **5.0 h** |

Full Gaussians × 8 cameras × 60 frames is ~45× the per-iteration cost of the
development configuration, so "lots of iterations" and a 5-hour budget are not
simultaneously achievable: 20/20 with 3 folds is a 20-hour run. The caps above
keep every one of the user's stated priorities — **no Gaussian, camera or
frame is dropped** — and spend the budget on the only axis that was free.

These are caps, not counts: the trust region stops a frame as soon as it
converges (the storage certificate accepted 6/6 and stopped), so the real run
should land at 50–70% of the upper bound. Scenes run cheapest-first, so an
overrun costs the last scene rather than all of them, and a `DEADLINE_H` guard
**skips** any scene it cannot finish rather than leaving a half-written tree
that looks like a completed run (§42).

Ground truth, for reference only — it reaches no estimator:

| scene | GT joint | Gaussians |
|---|---|---:|
| blade_103706 | prismatic | 7 206 |
| stapler_103111 | revolute | 9 077 |
| USB_100109 | revolute (continuous, unlimited) | 11 259 |
| fridge_10905 | revolute | 28 021 |
| storage_45135 | prismatic | 39 405 |

§21's `shared_inputs_manifest.json` records SHA-256 digests of the canonical
`mu`/`R`/`scales`/`colors`/`opacity`, the frozen support membership, every
camera's view and projection matrix, and the per-frame RGB and depth of the
chosen camera — so "all four models saw the same inputs" is checkable, not
asserted.

## 1b. The moving support was a median split — and it dominated everything

The first full five-scene run exposed a defect upstream of every Phase-2 model.
`_frozen_moving_support` selected

    moving = displacement > median(displacement)

which is a **median split**: it returns exactly half the scene by
construction, whatever the data says. Measured on all five datasets it returned
**50.0%, 50.0%, 50.0%, 50.0%, 50.0%**.

The joint is fitted to whatever this set contains, so for a drawer — where
roughly a tenth of the Gaussians move — Phase 2 was handed a "rigid body" that
was ~80% static, and the best single rigid screw through it is almost no
motion:

| scene | GT travel | COM displacement recovered |
|---|---:|---:|
| storage | 0.56 | **0.00022** |
| blade | 0.88 | **0.00186** |

Replaced by `estimator/moving_support.py`: Otsu's parameter-free two-class
threshold on each Gaussian's trajectory excursion `max_t ||x_t − x_0||`. No
tunable parameter, no world unit, and the kept fraction is whatever the scene
supports rather than a fixed share. The validation is that the selected set's
**median excursion lands on the ground-truth joint travel**:

| scene | was | now | selected median excursion | GT travel | separability |
|---|---:|---:|---:|---:|---:|
| blade | 50.0% | 15.5% | 0.82 | 0.88 | 0.86 |
| stapler | 50.0% | 43.7% | 2.14 | 2.01 | 0.85 |
| USB | 50.0% | 40.8% | 1.04 | — | 0.81 |
| fridge | 50.0% | 26.2% | 1.01 | 1.05 | 0.85 |
| storage | 50.0% | 9.2% | 0.47 | 0.56 | 0.56 |

Two secondary changes: the excursion is `max_t`, not last-minus-first (a joint
that reverses or returns to its start has a small end-to-end displacement and
would be called static), and the selection persists its evidence — threshold,
separability, kept fraction, selected-vs-rejected contrast — so it is
checkable rather than implicit. Storage's separability of 0.56 is the weakest
of the five and is worth watching.

**Every number in §4 and §10–§12 of the first full run was produced with the
median split and is void.**

## 2. Phase-1 numerical behaviour

Phase 1 is now **inside** the unified tree (§9), not a prerequisite run
elsewhere: `phase1/state_trace.csv`, `per_frame_camera.csv`,
`centers/centers_frame_XXXX.npy` for every frame, `support.json`, and its own
renders of the chosen camera. Its free centers feed the 3D page directly
(§29) — they are never faked by applying one rigid transform.

## 3. Corrected generic screw audit

Three corrections, all of which changed what the audit actually tests.

**§2 — orientations now reach the rasterizer.** The Stage-A runner called the
audit with `R0 = None`, *and* its `build_lin(mu, Rg)` accepted `Rg` and
dropped it. So even passing orientations would have changed nothing: every
state was rendered with canonical covariances, and a rotation-only state was
indistinguishable from a translation of the same centers. Both are fixed;
`estimator/phase2_shared.render_state` is the single place orientations enter
the renderer.

**§4 — one `R_eff`, not two.** The audit rebuilt sensor precision with
`build_precision` while production used `R_eff = R_sensor + R_model`. The
formula now lives once in `estimator/phase2_shared.effective_precision` and is
used by Phase-1 energies, the Stage-A audit, the generic screw, exact P, exact
R and the held-out evaluation. `joint_runtime._measurement_precision` delegates
to it and owns no formula.

**§5 — weights frozen at the full-rigid state.** Already correct; now recorded
(`reference_precision_source: full_rigid`, `robust_weights_frozen: true`).

## 4. Corrected static / trans / rot / full table

Per frame **and** per camera in
`phase2_screw/static_trans_rot_full/per_frame_camera.csv`, with
`delta_rot_given_trans`, `delta_trans_given_rot` and the contribution shares.

§3's transport gates are now real gates:

| check | status |
|---|---|
| `max_pair_distance_drift` | verified |
| `max_orientation_transport_error` — `R_i(t) = R_t R_i^0` | verified, 1.1e-16 |
| `max_covariance_transport_error` — `Σ_i(t) = R_t Σ_i^0 R_tᵀ` | verified, 6.9e-18 |

The covariance check previously computed `R₀R₀ᵀ = I` and compared it with the
identity, which tests nothing; it now uses the real per-Gaussian scales. And a
missing `R0` **raises** instead of skipping — a gate that quietly does nothing
when its input is `None` is how the old audit reported "rigidity verified"
while never touching an orientation.

## 5. Exact P behaviour

State and equations unchanged (§10). Diagnostics completed: `state_trace.csv`
carries `s`, `sdot`, `e_P`, `direction_update_norm`, RGB/depth/total energy,
accepted, trust radius, PCG iterations, posterior and geometry-information
eigenvalues; plus per-frame-per-camera energies and all three geometry-score
levels.

## 6. Exact R behaviour

Finite-axis physical state kept (§11): `μ_i(t) = a_R + R(e_R, θ_t)(μ_i^0 − a_R)`
with constant `e_R, a_R`. The final reported joint is never replaced by a
generic screw. Same diagnostic set as P plus `axis_direction_update_norm`,
`axis_position_update_norm` and `R_fit_status` on every row.

## 7. R small-angle observability

This is the substantive fix. Previously M_R froze at `θ = 0` and its
initial-axis energy was reported as a fitted model.

`estimator/phase2_revolute_init.py` implements §13's hierarchy:

- **A** — finite-axis initialization from the generic screw, taken only when
  its angular block is nonzero **and** the accumulated geometry information is
  numerically full rank **and** the solve accepted a decrease. `θ_t = q_t|ω|`,
  because the chart's `q` is dimensionless and reading it as the angle would
  scale the whole trajectory by `|ω|`.
- **B** — otherwise the continuous zero-pitch chart as a *bootstrap*, where
  `η = 0` is exactly prismatic and a regular interior point, so "does the
  residual want curvature?" can be asked without dividing by a vanishing
  angle. Its result is never reported as the final R joint.
- **C** — otherwise `R_GEOMETRY_UNOBSERVABLE`.

No axis-radius, omega or angle threshold anywhere. The rank tolerance is the
standard `max(eig)·n·eps`, which scales with the problem.

§14's validity block is saved per run: `initialization_source`,
`max_abs_theta`, `accepted_iterations`, `geometry_information_eigenvalues`,
`geometry_information_rank`, `axis_update_norm`, `axis_offset_update_norm`,
`R_fit_valid`, `R_fit_status`.

**Effect:** on storage, R now reports `FITTED` via branch A, where the previous
implementation left it frozen at `θ = 0` with 0 accepted iterations.

## 8. Wrong-initialization P

25° wrong direction, travel halved → recovers to **0.001°** (§36).

## 9. Wrong-initialization R — the new P0 gate

§15 asked for substantially wrong axis direction, an offset error of O(part
scale), a wrong θ scale **and** a non-constant angular speed, with no loosened
tolerance:

| quantity | start | recovered |
|---|---|---|
| axis direction | 25.0° wrong | **0.913°** |
| axis offset (transverse) | 0.211 (part scale 0.216) | **0.008** |
| θ trajectory | scaled ×0.6, accelerating then stalling | max \|Δθ\| **1.85e-03** |
| energy | 1.9865e+04 | **3.6622e+02** |

## 10–12. USB / Storage / Blade

**Not yet run at the agreed budget.** §42 is explicit that a smoke test proves
plumbing only, so these sections are deliberately empty rather than filled
with reduced-size numbers. The command is in §15 below.

What the reduced-size validation runs *do* establish is that the pipeline
executes end to end on real data for storage and blade, with all nine stages,
the complete artifact tree, both HTML pages and nonzero CUDA per stage.

**The old Stage-A table (USB trans 63.8%, storage 86.9%, blade 75.5%) is
superseded and must not be quoted** (§34). It predates all three corrections
above. The authoritative numbers come from the full runs.

## 13. Per-camera geometry consistency

§6 was the other substantive fix. The old `geometry_score_per_frame`
concatenated all cameras and produced only `g_{θ,t}`. Now:

    g_{θ,t,c} = J_{θ,t,c}ᵀ R_eff,t,cᐨ¹ r_{t,c}
    F_{θ,t,c} = J_{θ,t,c}ᵀ R_eff,t,cᐨ¹ J_{θ,t,c}

with aggregation **afterwards** to `g_{θ,t}`, `g_{θ,c}` and `g_θ`. All levels
persist (`geometry_score_per_frame_camera.csv`, `..._per_frame.csv`,
`..._per_camera.csv`, `geometry_information_per_frame_camera.npz`), for the
screw, P and R alike — the geometry block is read from `state.geometry_slice`,
so one implementation serves all three.

Why this is not a refinement but the diagnostic, as a unit test
(`test_S6_per_frame_and_per_camera_scores`): two cameras pulling in exactly
opposite directions give a per-frame sum that is **identically zero** and looks
perfectly healthy, while the per-camera level shows cosine −1.00 and 100%
opposed. §7's alignment numbers are reported at both levels, and the
interpretation is recorded with them — disagreement *across cameras* at one
frame is model mismatch, disagreement *across frames* is the
constant-geometry assumption.

On a 5-frame blade validation the screw's across-camera cosine is **−0.504**,
which is the §37 signature appearing immediately.

**§8 is respected:** nothing here concludes that USB invalidates the constant
joint. The axis is never relaxed to be frame-varying, and the per-camera level
exists precisely so that opposition can be attributed to specific cameras or
time segments before any such conclusion is reachable.

## 14. Held-out comparison

Fixed camera folds, with §31's validity rule enforced: a fold counts only when
`R_fit_status == FITTED`. `comparison/heldout_metrics.csv` carries train and
held-out camera IDs, P and R train and held-out energies, the R status and
`comparison_interpretable` per fold. An unobservable R is **not** silently
included via its initial-axis energy.

No fold win is converted into a label, and no majority vote exists (§31, §32).

## 15. Chosen-camera render paths

`<out>/html/chosen_camera_all_frames.html` — frame slider, five methods (GT,
Phase 1, screw, P, R) × RGB, depth, RGB residual, depth residual, with the
current frame's `screw q/qdot`, `P s`, `R theta`, energies and R status shown
above the grid.

PNGs live under each model's own `renders/camera_<id>/` (§22). §23's scaling
rule is enforced by construction: a first pass computes **one** depth range
over the whole sequence and all methods, and **one** symmetric residual range,
and a second pass writes every image against them. No method is normalized on
its own.

## 16. Definitive 3D HTML paths

`<out>/html/joint_motion_debug.html` — the full debugger
(`debug/joint_motion_debug.py`), built on the shell of the earlier
`geometric_motion_debug` page: stage selector, play/loop/speed, HUD,
dashboard, camera frustums. Its stages are the four motions this pipeline
produces — Phase 1 free, generic screw, exact P, exact R — in one world frame,
with an **overlay** mode that shows all four at once, the **moving support
drawn against the static remainder** with its selection evidence in a side
panel, and joint overlays carrying their own numbers (`e_P` and travel, screw
`|ω|` and pitch, `e_R`/`a_R` with sweep and fit status).

`<out>/html/definitive_joint_3d.html` — one shared world frame, a frame slider,
and the actual Gaussian centers at the selected frame for canonical, Phase 1,
screw, P and R, each toggleable. Overlays: Phase-1 COM trail, `e_P` direction,
generic screw axis, revolute axis `(e_R, a_R)`, camera frustums.

§26 is enforced: one deterministic index subset is chosen once, saved to
`html/debug_gaussian_indices.npy`, and used by every layer. No layer resamples.

## 17. CUDA usage per phase

§40: `reset_peak_memory_stats()` before each stage, recorded after. From the
8-camera validation run (1500 Gaussians, 6 frames):

| stage | peak CUDA | seconds |
|---|---:|---:|
| phase1 | 1.45 GiB | 0.9 |
| phase2_screw | 0.77 GiB | 5.6 |
| stage_a_audit | 0.67 GiB | 0.2 |
| phase2_prismatic | 0.74 GiB | 0.6 |
| phase2_revolute | 0.69 GiB | 0.9 |
| renders | 0.36 GiB | 1.7 |

At the real scale (storage, all 39 405 Gaussians, 8 cameras) Phase 1 peaks at
**2.15 GiB** — comfortable headroom on the 8 GiB card once the RGB-D window is
staged in host RAM.

Every stage is nonzero, so the renderer/JVP/VJP path is genuinely on CUDA. The
joint state itself stays CPU float64, which is the validated design.

## 18. Unresolved problems

1. **The three full runs have not been executed** (§42). Steps 9–11 of §45 are
   open. Nothing in §10–§12 above may be quoted as a result until they finish.
2. **The old Stage-A numbers are void** and no corrected replacement exists
   yet, for the same reason.
3. **R's `θ → 0` degeneracy is mitigated, not removed.** The hierarchy stops
   M_R from stalling at a singular start and reports honestly when curvature
   is unobservable, but the underlying fact — that `δφ = θ J_l(θe)E δη`
   vanishes at `θ = 0` — is a property of the finite-axis chart. A
   reparameterization that stays observable as `θ → 0` is not specified.
4. **§38's fuller synthetic battery is partial.** Wrong direction, wrong
   travel, wrong axis, wrong offset, wrong θ scale, variable angular speed and
   the near-P observability ladder are covered. Stop/restart/reversal and
   partial visibility are not.
5. **The screw under-fits the trajectory.** With the corrected support on an
   8-frame storage slice the filter reached `q = 0.0094` at 25 iterations —
   roughly an eighth of the travel the data contains — with the trust radius
   pinned to its floor (1e-08) on five of eight frames. `q(t)` started at zero
   on every frame, so the filter had to discover the whole trajectory
   incrementally against a collapsing trust region. §29's Phase-1 COM
   projection is now used to initialize `q(t)` (`initial_q_from_phase1`), with
   `q(0) = 0` kept as the canonical gauge; whether that is sufficient is still
   being measured.
6. **K > 1 remains deferred**, as the spec requires.

### Regression gates (§39)

| suite | result |
|---|---|
| `test_phase2_parallel_pr` (P/R JVP+VJP, wrong-init P and R, near-P) | 37 passed |
| `test_joint_screw_iekf` | 37 passed |
| `test_phase2_screw_runtime` | 17 passed |
| `test_efficient_p_null` | 20 passed |
| `test_iekf_end_to_end` (Phase 1) | 26 passed |
| CUDA peak nonzero on real runs | yes, per stage |

**137 checks, all passing.**
