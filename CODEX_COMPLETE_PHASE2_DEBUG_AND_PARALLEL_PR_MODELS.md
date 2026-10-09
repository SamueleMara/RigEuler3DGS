# CODEX — COMPLETE THE PHASE-2 DEBUG PROPERLY
## Fix the current audit, finish the real P/R runs, and build one definitive per-dataset debug pipeline

# 0. Objective

Do not add another classifier yet.

The next task is to make the current pipeline scientifically inspectable and to finish the missing implementation/debug work.

For every dataset, the code must produce and compare:

\[
\boxed{\text{Phase 1 free motion}}
\]

\[
\boxed{\text{Phase 2 generic rigid screw}}
\]

\[
\boxed{\text{Phase 2 exact prismatic}}
\]

\[
\boxed{\text{Phase 2 exact revolute}}
\]

using the **same RGB-D observations, same cameras, same frozen moving support, same renderer, and same Phase-1 measurement model**.

For Phase 2 P and Phase 2 R:

\[
\boxed{\text{joint geometry and joint displacement trajectory must be optimized simultaneously}}
\]

inside the same IEKF.

The deliverable is not another scalar P/R label. The deliverable is a complete numerical + rendering + 3D debug system from which we can see exactly:

- what Phase 1 wants to move;
- what rigid motion the generic screw recovers;
- how much of that motion is translation vs internal rotation;
- what the directly optimized P model does;
- what the directly optimized R model does;
- which cameras and frames support each motion;
- whether one constant joint geometry is really supported;
- whether the R fit is observable or degenerate;
- whether P/R differences generalize to held-out views.

---

# 1. Preserve the working architecture

Do not modify the mathematical behavior of Phase 1.

Do not replace the current generic rigid-screw Phase 2.

Preserve the already validated fixes:

- Phase 1 untouched;
- rigid-screw branch bypasses old free-SE(3) Phase 2;
- constrained-Newton screw retraction;
- exact generic P point;
- dense K=1 small-state preconditioner;
- full q/qdot/geometry cross-covariance;
- prior-derived trust-region scaling;
- frame-0 gauge;
- \(R_{\rm eff}=R_{\rm sensor}+R_{\rm model}\);
- sign-gauge continuity;
- analytic JVP/VJP;
- Gaussian orientation transport;
- CUDA rasterizer;
- direct RGB-D IEKF update.

The CUDA `0.00 GiB` issue was only a bookkeeping bug and is already fixed.

Do not revisit it unless the new unified runner reports inconsistent CUDA use.

---

# 2. First correction: Stage-A audit must move full Gaussian geometry, not only centers

The current Stage-A runner calls the audit with `R0 = None`, so the position decomposition is tested but Gaussian orientation transport is not actually exercised.

This must be fixed.

The debug must use the canonical Gaussian orientations/covariances from the actual 3DGS state.

For a rigid transform \(R_t\):

\[
\boxed{R_i(t)=R_tR_i^0}
\]

and equivalently:

\[
\boxed{\Sigma_i(t)=R_t\Sigma_i^0R_t^\top.}
\]

The four Stage-A render states must therefore be:

## Static

\[
\mu_i^S=\mu_i^0,\qquad R_i^S=R_i^0.
\]

## Translation only

\[
\mu_i^T=\mu_i^0+(c_t-c_0),\qquad R_i^T=R_i^0.
\]

## Rotation only

\[
\mu_i^R=c_0+R_t(\mu_i^0-c_0),
\]

\[
\boxed{R_i^R=R_tR_i^0.}
\]

## Full rigid

\[
\mu_i^F=c_t+R_t(\mu_i^0-c_0),
\]

\[
\boxed{R_i^F=R_tR_i^0.}
\]

Static Gaussians remain unchanged in every case.

The audit must use the actual orientation representation used by the renderer.

Do not create an approximate orientation path just for debug.

---

# 3. Verify orientation/covariance transport numerically

The current pairwise-distance rigidity test is good but insufficient.

Add mandatory checks:

\[
\max_i\|R_i^F-R_tR_i^0\|_\infty
\]

and, if the renderer stores covariance/scaling explicitly:

\[
\max_i\|\Sigma_i^F-R_t\Sigma_i^0R_t^\top\|.
\]

These checks must actually execute on real non-null orientation tensors.

A test that silently skips because `R0 is None` is not a valid gate.

Persist:

```text
max_pair_distance_drift
max_orientation_transport_error
max_covariance_transport_error
```

---

# 4. Second correction: Stage-A must use the same R_eff as the real IEKF

The current Stage-A audit rebuilds sensor precision directly with `build_precision`.

That is not enough.

The production Phase-2 runtime uses:

\[
\boxed{R_{\rm eff}=R_{\rm sensor}+R_{\rm model}.}
\]

The Stage-A audit must use the exact same code path as production:

```text
_measurement_precision(...)
```

or a common shared helper extracted from the production runtime.

Do not duplicate the formula.

The same helper must be used by:

- generic Phase 2;
- Stage-A static/trans/rot/full audit;
- exact P;
- exact R;
- held-out evaluation.

The weighted energies are otherwise not comparable.

This correction is especially important for Storage, where model-error covariance previously changed the apparent evidence by orders of magnitude.

---

# 5. Freeze robust precision consistently when comparing the four Stage-A states

For one frame/camera, the four Stage-A renders must be scored with a common weighting policy.

Use one documented reference precision:

\[
\boxed{R_{\rm eff}^{-1}\text{ evaluated at the fitted full-rigid state}}
\]

then reuse that exact precision for:

```text
static
translation-only
rotation-only
full-rigid
```

for that frame/camera.

Reason: if robust weights are recomputed independently for each state, a bad model can partially hide its residual by changing its own robust weights.

Persist:

```text
reference_precision_source = full_rigid
robust_weights_frozen = true
```

in the audit metadata.

---

# 6. Third correction: geometry score must really be per frame AND per camera

The current `geometry_score_per_frame()` concatenates all cameras and produces only \(g_{\theta,t}\).

That is not sufficient.

Implement:

\[
\boxed{g_{\theta,t,c}=J_{\theta,t,c}^TR_{{\rm eff},t,c}^{-1}r_{t,c}}
\]

and:

\[
\boxed{F_{\theta,t,c}=J_{\theta,t,c}^TR_{{\rm eff},t,c}^{-1}J_{\theta,t,c}.}
\]

Then aggregate only afterward:

\[
g_{\theta,t}=\sum_cg_{\theta,t,c},
\]

\[
g_{\theta,c}=\sum_tg_{\theta,t,c},
\]

\[
g_\theta=\sum_{t,c}g_{\theta,t,c}.
\]

Persist all levels.

Required files:

```text
geometry_score_per_frame_camera.csv
geometry_score_per_frame.csv
geometry_score_per_camera.csv
geometry_information_per_frame_camera.npz
```

---

# 7. Geometry-consistency diagnostics

For each Phase-2 model compute frame/camera score alignment.

For two blocks \(b_1,b_2\):

\[
\operatorname{cos}(g_{\theta,b_1},g_{\theta,b_2})=
\frac{g_{\theta,b_1}^Tg_{\theta,b_2}}
{\|g_{\theta,b_1}\|\|g_{\theta,b_2}\|}.
\]

Report:

```text
median cosine across frames
median cosine across cameras
fraction opposed across frames
fraction opposed across cameras
score norm per frame
score norm per camera
```

A constant joint geometry should not merely be imposed by the model. The debug must show whether the observations consistently support it.

---

# 8. Do not conclude that USB invalidates the constant-joint theory

USB is supposed to be a revolute joint.

Therefore, if the corrected score audit still shows that different frames oppose one another, do not immediately conclude:

```text
one constant screw is the wrong physical model
```

Instead diagnose which assumption is failing:

- wrong moving support;
- poor initial q trajectory;
- insufficient IEKF convergence;
- appearance/model mismatch;
- camera-specific calibration/depth mismatch;
- local minimum;
- geometry-score implementation;
- too short/incorrect temporal integration;
- incorrect orientation transport;
- incorrect weighting.

The physical theory remains:

\[
\boxed{e_R,a_R\text{ constant over the sequence}}
\]

for a single revolute joint.

Do not relax the axis to be frame-varying as a workaround.

---

# 9. Phase-1 debug must be included in the unified dataset output

The new unified script must not start at Phase 2.

For every dataset it must save Phase-1 outputs in the same artifact tree.

At minimum:

```text
phase1/
    moving_support.json
    state_trace.csv
    per_frame_camera_energy.csv
    centers.npy / centers_frame_*.npy
    velocities.npy
    covariance_or_information_summary.json
    renders/
```

For the chosen diagnostic camera, render **all frames**.

The Phase-1 rendered center positions must also be exported to the final 3D HTML.

The final debug should allow direct comparison:

\[
\text{Phase 1 free centers}
\]

versus:

\[
\text{generic rigid centers}
\]

versus:

\[
\text{P centers}
\]

versus:

\[
\text{R centers}.
\]

---

# 10. Exact P model: keep current implementation, but complete the diagnostics

The exact P model is:

\[
\boxed{\mu_i(t)=\mu_i^0+s_te_P}
\]

with:

\[
\boxed{x_t^P=(s_t,\dot s_t,e_P).}
\]

Joint geometry and travel remain jointly estimated.

Do not rewrite this state unless a failing test proves an issue.

Add complete per-frame/per-camera diagnostics:

```text
s
sdot
e_P
direction update
posterior covariance
RGB energy
depth energy
total energy
geometry score
geometry information eigs
accepted/rejected IEKF steps
trust radius
PCG iterations
```

---

# 11. Exact R model: keep the finite-axis physical state

The physical R output remains:

\[
\boxed{\mu_i(t)=a_R+R(e_R,\theta_t)(\mu_i^0-a_R).}
\]

with constant:

\[
\boxed{e_R,a_R.}
\]

State:

\[
x_t^R=(\theta_t,\dot\theta_t,e_R,\rho).
\]

Keep the exact physical finite-axis representation for the final R estimate.

Do not replace the final reported joint by a generic screw.

---

# 12. Fix the near-zero-angle R initialization/observability problem

At:

\[
\theta_t=0
\]

the physical axis is inherently unobservable.

This is a real property of the model, not a derivative bug.

However, the current implementation also allows the optimizer to stall at exactly zero before it has a chance to test whether the data wants rotation.

The corrected behavior must distinguish:

```text
R truly unobservable
```

from:

```text
R optimizer never left a singular initialization
```

---

# 13. R initialization hierarchy

Use this deterministic hierarchy.

## A. Generic-screw initialization

If the generic screw gives an observable nonzero curvature/rotation estimate, convert it to:

\[
e_R^{init},\quad a_R^{init},\quad \theta_t^{init}.
\]

Use the whole sequence, not one frame.

## B. Continuous near-P curvature bootstrap

If the finite-axis initialization is singular or numerically rank-deficient, do **not** inject an arbitrary axis threshold.

Use the existing continuous zero-pitch screw chart as a temporary bootstrap:

\[
T_t=\Exp\left(s_t\begin{bmatrix}d\\\eta\end{bmatrix}^{\wedge}\right),
\]

with:

\[
d^T\eta=0.
\]

At:

\[
\eta=0
\]

this is exactly P.

For:

\[
\eta\neq0
\]

it defines a zero-pitch revolute candidate.

The purpose of this bootstrap is only:

```text
find whether the RGB-D residual produces an observable nonzero curvature direction
```

and, if so, convert that result to a finite-axis R initialization.

Do not report the bootstrap as the final R joint.

Do not use a hand-tuned curvature threshold.

Use numerical observability/rank and accepted nonlinear decrease.

## C. If the bootstrap remains curvature-unobservable

Return:

```text
R_GEOMETRY_UNOBSERVABLE
```

for the R branch.

Do not pretend that the energy of the initial arbitrary axis is a fitted R model.

---

# 14. R branch comparison validity

For every R run save:

```text
initialization_source
max_abs_theta
accepted_iterations
geometry_information_eigenvalues
geometry_information_rank
axis_update_norm
axis_offset_update_norm
R_fit_valid
R_fit_status
```

Allowed statuses:

```text
FITTED
R_GEOMETRY_UNOBSERVABLE
OPTIMIZATION_FAILED
```

Held-out P-vs-R comparison is valid only when:

```text
R_fit_status == FITTED
```

Do not compare P to an unfitted frozen R hypothesis.

---

# 15. Add a real wrong-initialization test for R

The current wrong-initialization test is strong for P but incomplete for R.

Add a synthetic revolute test with:

- axis direction rotated substantially;
- axis offset shifted in the transverse plane;
- angular trajectory scaled incorrectly;
- nonconstant angular speed.

Example initial errors:

```text
axis direction: 20–30 degrees wrong
axis offset: O(part scale)
theta trajectory scale: 0.5–0.7
```

The exact same IEKF must reduce the RGB-D objective and recover:

\[
e_R,\quad a_R,\quad \theta_t.
\]

This is a mandatory P0 gate.

Do not loosen the synthetic tolerance simply to pass.

---

# 16. Re-check the generic screw with the corrected Stage-A audit

After fixing:

- Gaussian orientations;
- \(R_{\rm eff}\);
- per-camera geometry scores;

rerun the generic audit on:

1. USB;
2. Storage;
3. Blade.

The old Stage-A numbers must be treated as provisional until regenerated.

Persist both old and corrected numbers if useful, but mark which one is authoritative.

---

# 17. Required generic-screw numerical debug

For every frame save:

```text
frame
time
q
qdot
com_x com_y com_z
com_dx com_dy com_dz
rotvec_x rotvec_y rotvec_z
theta
rgb_energy
depth_energy
total_energy
accepted
trust_radius
pcg_iters
geometry_update_norm
geometry_info_min_eig
geometry_info_max_eig
```

Per camera/frame additionally save:

```text
E_static
E_translation_only
E_rotation_only
E_full
delta_rot_given_trans
delta_trans_given_rot
geometry_score components
```

---

# 18. Required exact P numerical debug

For every frame:

```text
frame
time
s
sdot
ePx ePy ePz
direction_update_norm
rgb_energy
depth_energy
total_energy
accepted
trust_radius
pcg_iters
posterior_eigs
geometry_info_eigs
```

Per camera/frame:

```text
RGB energy
depth energy
geometry score
geometry information
```

---

# 19. Required exact R numerical debug

For every frame:

```text
frame
time
theta
thetadot
eRx eRy eRz
aRx aRy aRz
axis_direction_update_norm
axis_position_update_norm
rgb_energy
depth_energy
total_energy
accepted
trust_radius
pcg_iters
posterior_eigs
geometry_info_eigs
R_fit_status
```

Per camera/frame:

```text
RGB energy
depth energy
axis/geometry score
axis/geometry information
```

---

# 20. Build one unified dataset runner

Create a new self-contained tool:

```text
tools/run_complete_joint_debug.py
```

This must be the authoritative debug entry point.

Example:

```bash
python tools/run_complete_joint_debug.py \
    --scene storage_45135 \
    --model output/dec_A_storage_45135 \
    --source input/storage_45135 \
    --camera 1 \
    --all-frames \
    --max-gaussians 6000 \
    --out output/complete_joint_debug_storage_45135
```

The script must execute, in order:

```text
1. Phase 1
2. freeze Phase-1 moving support
3. generic rigid-screw Phase 2
4. exact P Phase 2
5. exact R Phase 2
6. all numerical audits
7. all-frame chosen-camera renders
8. definitive 3D HTML
9. final comparison summary
```

No manual intermediate commands should be required.

---

# 21. Shared inputs must be proven identical

The unified runner must save a manifest proving that all Phase-2 models used:

```text
same canonical 3DGS
same moving support
same Gaussian indices
same camera calibration
same frame IDs
same timestamps / dt
same RGB observations
same depth observations
same R_eff model
same appearance
same renderer
```

Write:

```text
shared_inputs_manifest.json
```

with hashes/IDs where practical.

---

# 22. Output tree

Required structure:

```text
<out>/
    shared_inputs_manifest.json
    run_summary.json

    phase1/
        support.json
        numerical/
            state_trace.csv
            per_frame_camera.csv
        centers/
            centers_frame_0000.npy
            ...
        renders/
            camera_<id>/
                frame_0000_rgb.png
                frame_0000_depth.png
                frame_0000_rgb_residual.png
                frame_0000_depth_residual.png
                ...

    phase2_screw/
        numerical/
            state_trace.csv
            per_frame_camera.csv
            geometry_score_per_frame_camera.csv
            geometry_score_per_frame.csv
            geometry_score_per_camera.csv
        centers/
        renders/
            camera_<id>/
                ...
        static_trans_rot_full/
            per_frame_camera.csv

    phase2_prismatic/
        numerical/
            state_trace.csv
            per_frame_camera.csv
            geometry_score_per_frame_camera.csv
        centers/
        renders/
            camera_<id>/
                ...

    phase2_revolute/
        numerical/
            state_trace.csv
            per_frame_camera.csv
            geometry_score_per_frame_camera.csv
            observability.json
        centers/
        renders/
            camera_<id>/
                ...

    comparison/
        train_metrics.csv
        heldout_metrics.csv
        summary.json

    html/
        definitive_joint_3d.html
        chosen_camera_all_frames.html
```

---

# 23. Render one chosen camera for ALL frames

The user must be able to choose:

```text
--camera <id>
```

The selected camera must be rendered for every available frame, not only a stride.

For each frame create rows:

```text
GT RGB
GT depth

Phase1 RGB
Phase1 depth
Phase1 RGB residual
Phase1 depth residual

Phase2 screw RGB
Phase2 screw depth
Phase2 screw RGB residual
Phase2 screw depth residual

Phase2 P RGB
Phase2 P depth
Phase2 P RGB residual
Phase2 P depth residual

Phase2 R RGB
Phase2 R depth
Phase2 R RGB residual
Phase2 R depth residual
```

Use identical image scaling across methods.

Depth visualization must use one common depth range for the whole sequence or a clearly documented global normalization.

Do not normalize each method independently.

---

# 24. Chosen-camera HTML

Create:

```text
html/chosen_camera_all_frames.html
```

with a frame slider or frame-indexed table.

For each frame the page must show side-by-side:

```text
GT
Phase1
Phase2 screw
Phase2 P
Phase2 R
```

for both RGB and depth.

Also show residuals.

At the top display current frame numerical values:

```text
Phase1 motion magnitude
screw q/qdot
P s/sdot
R theta/thetadot
current RGB/depth energies
R observability status
```

---

# 25. Definitive 3D HTML

Create:

```text
html/definitive_joint_3d.html
```

This is mandatory.

Use a single shared world coordinate frame.

The HTML must contain a frame slider over all frames.

At every frame show the centers after:

```text
canonical / frame 0
Phase1
Phase2 generic screw
Phase2 P
Phase2 R
```

Use separate toggles.

Do not export only trajectories.

The user must see the actual Gaussian centers at the selected frame.

---

# 26. Use the same Gaussian subset in all 3D layers

For visualization select one deterministic subset of the frozen moving support.

Example:

```text
max_debug_points = 5000
```

with deterministic sampling.

Save the selected indices:

```text
html/debug_gaussian_indices.npy
```

Use the same indices for:

- canonical;
- Phase1;
- screw;
- P;
- R.

Do not independently resample each model.

---

# 27. 3D HTML layers

Mandatory layers:

```text
canonical moving centers
static-context centers
Phase1 centers
screw centers
P centers
R centers
```

Mandatory geometry overlays:

```text
canonical COM
Phase1 COM
screw COM
P COM
R COM

generic screw axis/direction
P translation direction
R revolute axis
camera frustums
```

Optional but recommended:

```text
velocity vectors
center trails
COM trails
```

---

# 28. 3D HTML frame slider

For frame \(t\), all methods must update synchronously.

The slider label must include:

```text
frame id
time
```

and model coordinates:

```text
screw q
P s
R theta
```

The axis overlays must remain in the correct world coordinates.

For P, \(e_P\) is constant.

For R, \(e_R,a_R\) are constant physical geometry.

For generic screw, \(\xi_J\) is constant.

The display must make this visibly obvious.

---

# 29. Phase-1 3D centers

Do not fake Phase-1 centers by applying one rigid transform.

Use the actual free-motion Phase-1 centers/supernode-transported Gaussian centers.

The whole point is to visually compare:

\[
\boxed{\text{free Phase 1 motion}}
\]

to:

\[
\boxed{\text{one rigid screw}}
\]

to:

\[
\boxed{P}
\]

to:

\[
\boxed{R}.
\]

---

# 30. Numeric comparison summary

Create:

```text
comparison/summary.json
```

and a readable table.

For every model:

```text
train RGB
train depth
train total
chosen-camera RGB
chosen-camera depth
heldout RGB
heldout depth
motion magnitude
geometry observability
geometry consistency
```

For generic screw additionally:

```text
translation-only share
rotation-after-translation share
camera spread
```

For P:

```text
direction
total travel
```

For R:

```text
axis direction
axis point
total angle sweep
R_fit_status
```

---

# 31. Held-out comparison remains mandatory

For P and R use the fixed camera-fold protocol.

But only count an R fold as a valid fitted hypothesis when:

```text
R_fit_status == FITTED
```

If R is unobservable:

```text
heldout comparison = NOT_INTERPRETABLE
```

Do not silently include its initial-axis energy.

Persist per fold:

```text
train camera IDs
heldout camera IDs
P train energy
R train energy
P heldout energy
R heldout energy
R fit status
comparison interpretable
```

---

# 32. Do not create a P/R classifier in this task

This task ends with evidence.

Do not implement:

```python
if E_R < E_P:
    REVOLUTE
```

Do not implement majority vote across folds.

Do not implement axis-radius, omega, or theta thresholds.

The purpose is to complete the debug first.

---

# 33. Full real runs are mandatory

Do not stop at 3-frame/2-camera smoke tests.

Run the complete debug pipeline on:

```text
USB_100109
storage_45135
blade_103706
```

using the intended real evaluation configuration.

At minimum use the same agreed frame/camera budget for all three scenes.

If full-all-frame/all-camera is computationally feasible, do it.

If not, document the exact common budget.

Do not use different budgets per scene to make results look better.

---

# 34. Authoritative Stage-A numbers must be regenerated

The old numbers:

```text
USB:     trans 63.8%, rot|trans 36.2%
Storage: trans 86.9%, rot|trans 13.1%
Blade:   trans 75.5%, rot|trans 24.5%
```

were useful but were generated before the audit corrections.

After fixing:

- Gaussian orientation transport;
- \(R_{\rm eff}\);
- per-camera score decomposition;

regenerate all three.

The new numbers become authoritative.

Do not quote the old table as final.

---

# 35. USB-specific debug

For USB, explicitly inspect:

1. whether corrected screw audit still shows large rotational benefit;
2. whether P clearly underfits;
3. whether R is fully observable;
4. whether exact R recovers a stable axis;
5. whether frame/camera geometry scores still oppose each other;
6. whether opposition comes from specific cameras or specific time segments;
7. whether the chosen-camera renders visibly support rotation;
8. whether Phase1 free centers are compatible with one revolute motion.

Do not relax the constant axis.

---

# 36. Storage-specific debug

For Storage inspect:

1. corrected translation-only share;
2. actual magnitude of generic screw rotation;
3. whether P reproduces the generic screw render nearly completely;
4. whether R is `FITTED` or `R_GEOMETRY_UNOBSERVABLE`;
5. if R is fitted, whether its gain generalizes to held-out cameras;
6. whether residual rotation is supported consistently across cameras;
7. whether Phase1 free centers show a coherent translational field.

This is the primary near-P stress test.

---

# 37. Blade-specific debug

For Blade inspect:

1. Phase1 center motion;
2. generic screw center motion;
3. P center motion;
4. R center motion;
5. camera-specific rotational gain;
6. per-camera geometry score direction;
7. chosen-camera residuals;
8. whether the fitted rotation improves one view while hurting others.

The output must make model mismatch visible, not hide it in one scalar objective.

---

# 38. P/R synthetic tests after corrections

Re-run all existing tests.

Additionally require:

## P synthetic

- wrong direction;
- wrong travel;
- stop/restart/reversal;
- partial visibility;
- P recovers geometry and trajectory.

## R synthetic

- wrong axis direction;
- wrong axis point;
- wrong theta scale;
- variable angular speed;
- stop/restart/reversal;
- partial visibility;
- R recovers geometry and trajectory.

## Near-P synthetic R

Create a far-axis / small-angle R sequence.

Check:

- finite-axis initialization may be weak;
- continuous curvature bootstrap can detect observable curvature when present;
- exact R is initialized and fitted if the data contains enough information;
- otherwise status is `R_GEOMETRY_UNOBSERVABLE`.

No arbitrary physical threshold.

---

# 39. Regression gates

Before claiming completion:

```text
generic screw tests all pass
Phase1 tests all pass
P JVP FD passes
P VJP adjoint passes
R JVP FD passes
R VJP adjoint passes
wrong-init P passes
wrong-init R passes
near-P R observability test passes
CUDA peak is nonzero on real CUDA runs
```

---

# 40. Verify actual CUDA usage in the unified runner

The unified runner must save:

```text
cuda_available
cuda_device_name
peak_cuda_gib
```

for:

```text
Phase1
generic screw
P
R
```

Do not infer GPU use from one global number.

Reset peak memory before each major stage:

```python
torch.cuda.reset_peak_memory_stats()
```

and record after each stage.

This is debug metadata only.

The tiny joint state may remain CPU float64 if that is the current validated design.

The renderer/JVP/VJP path must remain CUDA.

---

# 41. Definitive report

Create:

```text
COMPLETE_PHASE1_PHASE2_JOINT_DEBUG_REPORT.md
```

Required sections:

1. common dataset/configuration;
2. Phase1 numerical behavior;
3. corrected generic screw audit;
4. corrected static/trans/rot/full table;
5. exact P behavior;
6. exact R behavior;
7. R small-angle observability;
8. wrong-initialization P;
9. wrong-initialization R;
10. USB;
11. Storage;
12. Blade;
13. per-camera geometry consistency;
14. held-out comparison;
15. chosen-camera render paths;
16. definitive 3D HTML paths;
17. CUDA usage per phase;
18. unresolved problems.

For every real scene include one compact table:

| model | RGB | depth | total | heldout | geometry status |
|---|---:|---:|---:|---:|---|
| Phase1 | | | | n/a | free |
| Screw | | | | | |
| P | | | | | |
| R | | | | | |

---

# 42. Do not claim completion unless the real outputs exist

A smoke test proves plumbing only.

The task is not complete until:

```text
USB complete run exists
Storage complete run exists
Blade complete run exists
```

with:

```text
all-frame selected-camera renders
numerical CSV/JSON diagnostics
definitive 3D HTML
```

If one run fails, report the failure and leave the task open.

Do not substitute old outputs.

---

# 43. Final expected command

Create a helper script:

```text
run_complete_joint_debug_all.sh
```

that runs:

```bash
python tools/run_complete_joint_debug.py --scene USB_100109 ...
python tools/run_complete_joint_debug.py --scene storage_45135 ...
python tools/run_complete_joint_debug.py --scene blade_103706 ...
```

using one shared configuration.

Allow:

```text
CAMERA=<id>
FRAMES=all
CAMS=all
NG=<number>
```

or equivalent CLI options.

At the end print the output directories and the two HTML files for each scene.

---

# 44. Final methodological requirement

The final debug must let a human inspect, for any frame:

\[
\boxed{\text{Phase1 centers}}
\]

\[
\boxed{\text{generic screw centers}}
\]

\[
\boxed{\text{P centers}}
\]

\[
\boxed{\text{R centers}}
\]

in the **same 3D world frame**.

And for the chosen camera, for every frame:

\[
\boxed{\text{GT, Phase1, Screw, P, R}}
\]

must be directly comparable in RGB and depth.

Only after this debug is complete should we decide how to turn the P/R evidence into the final joint classifier.

---

# 45. Implementation order

Do this exactly in order:

```text
1. Fix Stage-A orientation transport.
2. Fix Stage-A R_eff weighting.
3. Implement true per-frame/per-camera geometry scores.
4. Add real R wrong-initialization test.
5. Add R near-zero-angle bootstrap + observability status.
6. Build unified per-dataset runner.
7. Add chosen-camera all-frame rendering.
8. Add definitive 3D HTML with synchronized Phase1/Screw/P/R centers.
9. Run full USB.
10. Run full Storage.
11. Run full Blade.
12. Write the definitive report.
```

Do not work on K>1 and do not add another P/R classifier before these twelve steps are complete.
