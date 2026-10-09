# Rigid SE(3) Phase 2 and post-hoc joint reduction

Implementation report for
`CODEX_IMPLEMENT_RIGID_SE3_PHASE2_AND_POSTHOC_JOINT_REDUCTION.md`.

The production estimator no longer asks the renderer where the revolute axis
is. It asks only which rigid pose `T_t` best explains the moving part at each
frame. The joint is inferred afterwards, from the recovered finite rotations.

```
Phase 1 free motion
  -> coherence MOVING_RIGID support
  -> rigid SE(3) RGB-D Phase 2          <- six pose numbers, no joint geometry
  -> joint reduction P + R              <- no renderer
  -> pose-space model comparison
```

**Status.** Complete and measured. 222 checks pass across seven suites, 47 of
them new; all 21 P0 gates in section 46 hold, including the five-dataset
reruns of sections 37–38.

Headline: against the dataset's own joint geometry, the axis error on the
stapler went from **89.22 degrees to 0.23 degrees**, and on storage from
**72.91 to 0.23**. The new pipeline is better on all five scenes. It is now
the default (`JOINT_PIPELINE.md`); the direct-axis estimators remain as
ablations and announce themselves as such.

---

## 1. Why direct P/R image optimization was removed from production

The old Phase 2 handed the image optimiser an axis direction, an axis point
and an angle, and let it move them to reduce the RGB-D residual. That
parameterisation has a degenerate valley:

```
far-away axis + small angle   ~=   translation-like image motion
```

A revolute joint whose axis sits far from the part produces, over a short
sweep, almost exactly the pixels a translation produces. The optimiser is
free to slide along that valley, and the axis it stops at is set by whatever
breaks the tie — initialisation, noise, the prior — not by the data. That is
the shape of the stapler failure.

The new architecture has no such valley, because the rigid pose has no axis
ambiguity at all. `(R_t, p_t)` is six numbers with a unique minimiser. Only
afterwards is the joint read off:

```
Log(R_t)^v = theta_t e_R        pins the direction
p_t = (I - R_t) a               pins the point
```

Both are linear-algebra problems on the recovered finite transforms, and
neither touches a pixel. §13 shows that the second one stays conditioned at
`cond = 1.0` even when the axis is 100 part-radii away and the total rotation
is 0.003 rad — precisely the regime where the image optimiser was blind.

**Legacy branches are kept** (section 43). `tools/run_complete_joint_debug.py`
still runs the generic screw, direct-P and direct-R estimators for regression
and paper ablations. They no longer define production joint geometry.

---

## 2. The coherence support used

`--phase2-support-source coherence` is now the production default, in the new
pipeline and in `run_complete_joint_debug.py`. Only the final high-confidence
`MOVING_RIGID` class reaches Phase 2; `INCOHERENT_OR_AMBIGUOUS` nodes are
excluded, and there is **no silent fallback** — a missing or `NO_MOVING`
support stops the run with a message naming the file.

The policy is deliberately precision-first, and the asymmetry is measurable.
RSE3-8/9 construct both error modes on the same scene:

| support | result |
|---|---|
| pure but **incomplete** (12 of 24 real members) | pose error `8.7e-16` |
| complete plus **6 injected static points** | pose error `7.9e-02` |

A false negative removes evidence. A false positive forces a genuinely static
Gaussian through the estimated transform and biases it directly — here by
eleven orders of magnitude more. The contaminated fit also shows it in its own
Kabsch residual (`1.1e-16 -> 3.1e-02`), so the condition is detectable
without ground truth.

Static Gaussians are still rendered. They carry visibility, alpha
compositing, occlusion, RGB and depth; the rigid state simply does not move
them, which a bitwise test confirms.

---

## 3. The rigid SE(3) state

`estimator/rigid_part_se3.py`. Per frame: a pose `T_t in SE(3)` and a spatial
velocity `xi_t = [v_t; omega_t]`. The local error state is 12-dimensional,

```
delta_x = [ delta_tau (6) | delta_xi (6) ]
```

with `delta_tau = [delta_v; delta_omega]` the LEFT perturbation of the pose,
matching Phase 1's `boxplus` (`R+ = Exp(hat(dphi)) R`) and the existing screw
runtime. Left and right perturbations are never mixed.

Frame 0 is the canonical gauge, `T_0 = I` — a definition, not an estimate.
`run_rigid_window` pins it and never updates on it: `T_0 = I` whatever the
motion is, so frame 0 carries no information about any coordinate, and
updating there would spend the residual that belongs to model error on a
motion that did not happen.

There is no prismatic direction, revolute axis, axis point or joint type in
this state. A structural test reads the runtime's AST and fails if it imports
or names any axis fitter.

---

## 4. Exact Gaussian transport

For `i` in the frozen support:

```
mu_{i,t} = R_t mu_i^0 + p_t        R_{i,t} = R_t R_i^0        Sigma_{i,t} = R_t Sigma_i^0 R_t^T
```

and for every other `i`, nothing changes. Because all members share one
transform,

```
||mu_i(t) - mu_j(t)||^2 = (mu_i^0 - mu_j^0)^T R_t^T R_t (mu_i^0 - mu_j^0) = ||mu_i^0 - mu_j^0||^2
```

exactly. **No rigidity regulariser exists anywhere in the implementation**, and
none is needed; rigidity is a property of the parameterisation.

Centers and orientations move together. The renderer takes `(scales,
rotation)` rather than a covariance, and `R_{i,t} = R_t R_i^0` with unchanged
scales *is* the covariance transport — `transported_covariance` exists so the
test can check that claim rather than assert it. Updating centers without
orientations is what previously made a rotation-only state indistinguishable
from a translation in the image.

Measured (section 20 gate), on synthetic data and on the real blade support:

| quantity | synthetic | real runs, 60 frames, all five scenes |
|---|---|---|
| `max_{i,j,t} \| d_ij(t) - d_ij(0) \|` | `1.78e-15` | `2.78e-16` … `1.33e-15` |
| `max \| R_i(t) - R_t R_i^0 \|` | `0` bitwise | `0` bitwise |
| static centers / orientations moved | `0` bitwise | — |
| `max \| Sigma_i(t) - R_t Sigma_i^0 R_t^T \|` | `0` bitwise | — |

The gate runs inside the pipeline on every scene and writes
`phase2_rigid/exact_rigidity_gate.json`.

---

## 5. JVP and VJP

A left perturbation moves the already-transformed center `mu_i`:

```
d mu_i  = delta_v + delta_omega x mu_i = [ I , -hat(mu_i) ] delta_tau
d phi_i = delta_omega
H_xi    = 0
```

and the transpose, from `<bar_mu, delta_omega x mu> = <mu x bar_mu, delta_omega>`:

```
bar_v     = sum_i bar_mu_i
bar_omega = sum_i ( mu_i x bar_mu_i + bar_phi_i )
```

Both analytic; no autograd. Composition with the renderer is `H_T = H_G
J_{G,T}` through the existing `joint_apply_H` / `joint_apply_HT` bridge — the
Phase-1 linearization, the CUDA contributor traversal and the alpha-clamp
derivative are untouched.

Measured:

| contract | result |
|---|---|
| JVP vs central finite differences, all 12 coordinates | `2.3e-10` (centers), `1.1e-10` (orientations) — the FD floor |
| `H_xi = 0`, all six velocity columns | bitwise zero |
| adjoint identity `<Hv, z> = <v, H^T z>`, 8 random probes | `1.7e-15` |

`H_xi = 0` is not an approximation. A single image has no derivative with
respect to rigid velocity, and pretending otherwise would count the process
prior as an extra measurement.

---

## 6. The IEKF process

`T_{t+1}^- = Exp(dt xi_t^) T_t^+`, `xi_{t+1}^- = xi_t^+ + w`. The first-order
error propagation is

```
delta_tau_{t+1} = Ad_E delta_tau_t + dt J_l(dt xi) delta_xi_t
delta_xi_{t+1}  = delta_xi_t
```

with `E = Exp(dt xi^)`. The off-diagonal block is the only route by which
image evidence reaches the velocity, since `H_xi = 0`; it is not optional and
is tested against numerical propagation (`1.4e-10`). At `xi = 0` it degenerates
exactly to `[[I, dt I], [0, I]]` (`0.0e+00` difference).

`Q` is white acceleration on the twist, integrated once, with **separate**
`sigma_vdot` and `sigma_wdot`: a linear and an angular acceleration do not
share units, and one scalar for both would make the process prior depend on
whether the scene is measured in metres.

The error-state reset transports the pose block by the SE(3) left Jacobian at
the accepted step, `G_tau = J_l(d)`, verified against the numerical Jacobian
of the exact error remap (`1.7e-11`). Using `I` instead would rotate the pose
uncertainty away from the pose it describes, and the first casualty would be
the pose/velocity cross-covariance.

Everything downstream of the state-to-Gaussian map is Phase 1's: `R_eff =
R_sensor + R_model`, robust weighting, visibility, the trust region, the
acceptance test and the Steihaug-CG solve. The shared `frame_step` is reused
unmodified.

---

## 7. Phase-1 Kabsch initialization

Per frame, weighted Kabsch on the Phase-1 free trajectories restricted to the
frozen support, then the twist that connects consecutive poses:

```
xi_t^init = (1/dt) Log( T_{t+1} T_t^{-1} )^v
```

**Weighting policy, stated once:** every member of the frozen support is
weighted equally. The support is already precision-first — built to be pure
rather than complete — so a second reweighting here would be tuning on top of
a selector designed not to need it. `weights` exists for the RSE3-9 ablation
and the production path does not use it.

The reflection guard (`det(U V^T)` folded into the SVD) is not cosmetic: a
nearly planar support — a drawer front, a blade face — produces a mirror
whenever the smallest singular value is at noise level. Tested on a
deliberately planar support: `det = 1.000000`.

Frame 0 is forced to identity rather than taken from Kabsch, so the gauge
holds exactly.

**Section 17 is enforced structurally.** No screw, prismatic or revolute model
is fitted to produce this initialization, and the test fails the build if the
runtime imports `phase2_revolute`, `phase2_prismatic`, `joint_state`,
`phase2_revolute_init`, `curvature_score` or `prismatic_null`, or names
`init_revolute`, `initialize_revolute`, `run_phase2_rigid_screw`,
`revolute_axis` or `prismatic_direction`.

Measured: Kabsch recovers an exact rigid trajectory to `2.2e-16` (rotation)
and `1.4e-16` (translation); the initial twist reproduces the pose sequence to
`1.4e-15`.

---

## 8. Rigid Phase-2 numerical tests

`tests/test_rigid_se3_phase2.py`, 47 checks. The renderer-coupled cases run
against the real CUDA rasterizer through the production linearization.

| test | result |
|---|---|
| RSE3-1 arbitrary rigid trajectory | rigidity `1.8e-15`, JVP `2.3e-10`, VJP `1.7e-15` |
| RSE3-2 pure translation | pose recovered to `2.7e-15`; spurious rotation `1.5e-15` rad |
| RSE3-3 finite revolute | rotations to `1.9e-15` rad, translation to `3.5e-15` |
| RSE3-4 variable speed, stop, reversal | `gamma_1DOF = 1.000000`, axis `1.5e-06` deg, theta `2.4e-08` |
| RSE3-5 wrong Kabsch seed (0.21 rad, 40% short) | corrected to `0.000` rad / `2e-4` |
| RSE3-6 weak late frames | axis error `25.04 deg` uniform -> `0.000 deg` weighted |
| RSE3-7 far-axis trap (radius 1, 10, 100) | axis `0.00e+00` deg, relative point error `4.2e-14` |
| RSE3-8 pure incomplete support | `8.7e-16` |
| RSE3-9 static contamination | `7.9e-02` — eleven orders worse |

RSE3-5 is the one that matters for trusting the stage at all: RGB-D rigid
Phase 2 is not a cosmetic refinement of the Kabsch seed. Given a seed rotated
12 degrees off and 40% short, it recovers the true pose.

---

## 9. Phase-1 Kabsch versus RGB-D rigid

Every run writes `phase2_rigid/phase1_kabsch_vs_rgbd.csv` and `.json` with,
per frame,

```
d_R(t) = || Log( (R_t^P1)^T R_t^RGBD )^v ||        d_p(t) = || p_t^P1 - p_t^RGBD ||
```

This is the diagnostic to read **before** any joint reduction. Phase 1 on a
coherent support is already a rigid trajectory; if the RGB-D stage moves a long
way from it, the thing to inspect is the RGB-D objective, not the axis fit.
Section 22's render page puts the Kabsch and Phase-2 rows adjacent for exactly
this comparison, with residual images.

Measured on all five: the diagnostic fires on stapler (3.1382 rad, 33/60
frames) and blade (2.8724 rad, 19/60) and is quiet on the other three
(< 0.25 rad, 0 frames). On both scenes where it fired, the RGB-D answer is
the one that matches GT. Full table in §15–19.

---

## 10. Pose covariance

Persisted per frame as `phase2_rigid/pose_covariances.npy` (the `6x6`
`delta_tau` block) and `state_covariances.npy` (the full `12x12`). The
posterior is the exact one: the 12-dimensional operator is probed with its own
basis vectors, so no pixel Jacobian is ever materialised.

`state_trace.csv` carries the section 19 fields per frame, including
`pose_information_eigs` and `velocity_information_eigs`.

---

## 11. One-DOF rank analysis

Relative increments use **one** convention throughout, `Delta T_t = T_{t+1}
T_t^{-1}` (world/spatial), documented in the module and recorded in the output.

Translation and rotation do not share units, so the generator stack is made
dimensionless with the part scale `L^2 = (1/N) sum_i ||mu_i^0 - c_0||^2`:

```
zeta_bar_t = [ v_t / L ; omega_t ]        gamma_1DOF = sigma_1^2 / sum_k sigma_k^2
```

Without `L` the statistic would change if the scene were measured in
centimetres. Section 26 is explicit that this is a **structural diagnostic**,
not a classifier: no threshold is applied and no joint type is returned from
it.

The SVD is weighted by the relative-twist covariance propagated from the pose
marginals (§27). **The approximation is named**: the inter-frame
cross-covariance is not available from a filter that stores marginals, so it
is dropped. Consecutive frames are positively correlated in practice, so the
dropped term would *reduce* `P_zeta` — the weights are conservative, never
over-confident.

Measured: `gamma = 1.000000` on exact prismatic, exact revolute, and on a
trajectory with two stops and a reversal.

---

## 12. Prismatic reduction

`e_P` by weighted PCA of the translations, `s_t = e_P^T p_t`. The scatter is
**not** mean-centred: under `p_t = s_t e_P` the trajectory passes through the
origin because `T_0 = I` is the gauge, so centring would discard the one point
the gauge fixes.

The observed rotation is reported alongside (`residual_rotation_angle_max` and
`_mean`) — a P model cannot explain rotation at all, so that number *is* the
part of the motion it must leave on the table.

Measured on an exact prismatic trajectory: direction error `0.000 deg`, `s`
to `2.8e-08`, pose residual `5.0e-15` against `3.6e-04` for the revolute fit
of the same data.

---

## 13. Revolute finite-axis reduction

Two separate problems, both on finite transforms.

**Direction** (§29): `e_R = argmax_{||e||=1} sum_t w_t (e^T phi_t)^2` with
`phi_t = Log(R_t)^v`, i.e. the principal eigenvector of `sum w_t phi_t
phi_t^T`. Finite rotations, not angular velocities — a finite rotation
sequence about a fixed axis is exactly rank 1, while differentiating first
throws away the signal on every frame where the part is momentarily still.

**Point** (§30): `a* = argmin_a sum_t w_t || (I - R_t) a - p_t ||^2`, with the
gauge `e_R^T (a - c_0) = 0` imposed by construction — solving for `a = c_0 + E
y` on the plane perpendicular to the axis makes the system `2x2` instead of
rank-deficient `3x3`, and the returned point is the one on the axis nearest
the canonical centroid. The solve is by SVD, not normal equations, so the
condition number is observable rather than squared away.

`a = omega x v / ||omega||^2` is **not** used. It is the instantaneous screw
centre, it divides by an angular speed that is near zero exactly when the part
is slow, and it is the route by which the old pipeline produced axes at
infinity.

**Angles** (§31): `Log` returns an angle in `[0, pi]`, so a part turning past
`pi` folds back and `e_R^T phi_t` reverses sign. The continuous series is
accumulated from the relative rotations instead; the raw projection and the
largest gap between the two are both reported, so a wrap is visible rather
than silent. On a 4-radian sweep the accumulated series tracks truth to
`1.1e-07` where the raw projection is off by `6.28`.

This is the section 36 claim, measured:

| axis radius | total rotation | axis direction error | relative point error | solver cond |
|---|---|---|---|---|
| 1.0 | 0.300 rad | `0.00e+00` deg | `0` | `1.00` |
| 10.0 | 0.030 rad | `0.00e+00` deg | `5.1e-16` | `1.00` |
| 100.0 | 0.003 rad | `0.00e+00` deg | `4.2e-14` | `1.00` |

Same arc length in all three — i.e. the same image motion. The image optimiser
could not tell these apart; the finite solve does so exactly, and stays
perfectly conditioned.

---

## 14. Pose-space P/R refinement

Both models are refined against `T_hat_t` with `W_t = P_{T,t}^{-1}`, by
Levenberg-damped Gauss-Newton. No renderer is called; a structural test
enforces that `joint_reduction` imports only `torch`, `math` and the two
manifold modules.

`e_P` and `e_R` move on the sphere through their tangent planes, so they stay
unit vectors exactly rather than being renormalised after each step; `a_R`
moves only in the plane perpendicular to the axis, because sliding it *along*
the axis changes nothing and would leave the problem rank-deficient.

Jacobians are finite-difference. The repository's no-autograd rule is about
the estimator and the image Jacobians; this problem has at most `T + 3`
parameters and `6T` residuals, never touches a pixel, and runs in float64.
Hand-deriving four more Jacobians would buy nothing and risk a sign. The cost
is a visible floor: the refinement converges to ~`1e-7` relative rather than
machine precision, which is why §8's axis tolerances are `1e-4` deg and not
`1e-12`.

### One deliberate deviation from section 32

Section 32 writes the pose residual as `Log(T_M^{-1} T_hat)` — a **right**
perturbation — and weights it by `P_T^{-1}`. But `P_T` is the covariance of a
**left** perturbation, because that is the rigid Phase-2 error state. The two
charts differ by `Ad_{T_M}`, whose translation block grows with `|p|`. Pairing
them as written would mis-weight the fit by a factor that scales with the
quantity being estimated — worst exactly on the parts that travel furthest.

The primary residual is therefore the left-chart one, `Log(T_hat T_M^{-1})`,
which lives in the same chart as the covariance. The right-chart residual is
computed too and written to `pose_residuals.csv` alongside, so the two can be
compared directly and the deviation checked rather than taken on trust.

---

## 15-19. Results, all five datasets

Run: 60 frames, 8 cameras, every Gaussian, coherence `MOVING_RIGID` support,
8 rigid iterations. 321-710 s per scene.

| scene | GT type | moving | gamma | E_P | E_R | P ang | R ang | R line | cond | rad/L |
|---|---|---|---|---|---|---|---|---|---|---|
| stapler_103111 | revolute | 4451 | 0.9954 | 2.82e+10 | 6.82e+05 | 89.77 | **0.23** | **0.0023** | 1.00 | 2.8 |
| USB_100109 | revolute | 6930 | 0.9964 | 1.90e+09 | 3.38e+05 | 89.47 | **0.37** | **0.0041** | 1.00 | 1.7 |
| fridge_10905 | revolute | 8058 | 0.9223 | 1.55e+09 | 2.80e+06 | 89.16 | **0.07** | **0.0069** | 1.00 | 1.7 |
| blade_103706 | prismatic | 1078 | 0.9934 | 1.12e+04 | 3.99e+04 | **0.08** | 89.93 | 40.14 | 1.14 | 289.9 |
| storage_45135 | prismatic | 2897 | 0.9643 | 1.07e+06 | 9.23e+05 | **0.23** | 89.88 | 32.72 | 1.00 | 62.2 |

Angles in degrees against `pointcloud/joint_metadata.json`. The bolded column
is the one that matches the GT type; the other is the wrong model fitted to
the same trajectory and is expected to be near 90 degrees.

### 15. Stapler — the forensic case

Section 37's six questions, answered.

1. **Does rigid Phase 2 recover sensible `R_t, p_t`?** Yes. Maximum rotation
   1.9099 rad (109 degrees), maximum translation 1.3939, both monotone in the
   opening direction.
2. **How close is it to Phase-1 Kabsch?** Not close, and that is the correct
   answer: `max d_rot = 3.1382 rad`, with 33 of 60 frames diverging by more
   than 0.5 rad, first at frame 27. Phase-1 free motion stops being rigid on
   this scene past the midpoint (Kabsch mean residual reaches 0.3153). The
   RGB-D stage overrode it, and the GT comparison says the override was right.
   This is the single strongest piece of evidence that rigid RGB-D Phase 2 is
   not a cosmetic refinement of the initializer.
3. **Are the finite rotations rank-1?** `gamma_1DOF = 0.9954`.
4. **Does R reduction recover the correct axis direction?** **0.23 degrees.**
5. **Does the axis-point solve stay finite and well conditioned?**
   `cond = 1.00`, line error **0.0023**, axis radius 1.1167 against a part
   scale of 0.4042 — the axis sits on the stapler.
6. **Is the far-axis degeneracy gone?** It cannot occur: no axis is a
   parameter of the image optimiser. The legacy direct-R estimator on the same
   scene and the same frozen support gives **89.22 degrees**.

The stapler axis error went from 89.22 degrees to 0.23 degrees — a 388-fold
reduction, and the difference between an axis perpendicular to the truth and
one essentially on it.

### 16-17. Blade and storage — the prismatic controls

Both behave as section 38 predicts. Blade's rigid trajectory is almost pure
translation (max rotation 0.0198 rad over 60 frames); storage's is 0.0282 rad.
P recovers the direction to **0.08** and **0.23 degrees**.

**Storage is the informative failure of the pose energy as a classifier.**
`E_R/E_P = 0.863`: the revolute model fits the trajectory marginally *better*
than the prismatic one, on a scene whose GT joint is prismatic. It does so by
putting its axis 32.7 away — **62 part-radii** — where a circular arc is
straight to within the residual. Blade is the same story at 290 part-radii.

So the energies alone would call storage revolute. The ratio
`axis_radius_over_part_scale` separates the two classes without ambiguity:

| | stapler | USB | fridge | blade | storage |
|---|---|---|---|---|---|
| GT | revolute | revolute | revolute | prismatic | prismatic |
| rad/L | 2.8 | 1.7 | 1.7 | 289.9 | 62.2 |

It is now computed and persisted as `revolute.axis_radius_over_part_scale`.
It is **reported, not thresholded** — section 26 forbids classifying from a
single statistic, and this report does not claim a classifier. But any
classifier built later should use this and not `E_R/E_P` alone.

### 18-19. USB and fridge — the revolute controls

Both recover coherent finite rotation before any axis is computed: maximum
rotation 1.0021 rad (USB) and 1.5919 rad (fridge). Axis errors **0.37** and
**0.07 degrees**, line errors **0.0041** and **0.0069**, both `cond = 1.00`,
both with the axis on the object (rad/L = 1.7).

Fridge has the lowest one-DOF statistic of the five, `gamma = 0.9223`. It also
has the largest support (8058 Gaussians) and the longest runtime (710 s). The
reduction is unaffected — 0.07 degrees is the best axis of the five — so the
rank-1 shortfall is in the increments, not in the finite rotations they
integrate to. Worth a look, not a defect.

### The exact-rigidity gate, on real data

Section 20 is a P0 gate and it holds on every scene, measured over 4000
sampled member pairs across all 60 frames:

| scene | max pairwise distance drift | max orientation defect |
|---|---|---|
| stapler_103111 | 8.88e-16 | 0 bitwise |
| blade_103706 | 2.78e-16 | 0 bitwise |
| USB_100109 | 1.33e-15 | 0 bitwise |
| fridge_10905 | 5.55e-16 | 0 bitwise |
| storage_45135 | 1.11e-15 | 0 bitwise |

No rigidity regulariser exists in the implementation. This is the
parameterisation, not a penalty that happens to be well tuned.

### Phase-1 Kabsch versus RGB-D rigid (section 21 diagnostic)

| scene | max d_rot (rad) | max d_p | frames > 0.5 rad | max Kabsch residual |
|---|---|---|---|---|
| stapler_103111 | 3.1382 | 0.8634 | 33 / 60 | 0.3153 |
| blade_103706 | 2.8724 | 0.7698 | 19 / 60 | 0.1843 |
| USB_100109 | 0.1202 | 0.0475 | 0 / 60 | 0.1852 |
| fridge_10905 | 0.2498 | 0.1401 | 0 / 60 | 0.2266 |
| storage_45135 | 0.1372 | 0.2492 | 0 / 60 | 0.2085 |

The diagnostic fires on exactly two scenes, and on both the RGB-D answer is
the one that matches GT. Three scenes agree with their initializer to within
0.25 rad. Had this gone the other way — a large divergence plus a bad axis —
section 21 says to inspect the RGB-D objective before the reduction, and that
is still the right reading if it happens on new data.

---

## 20. GT axis evaluation

`joint_reduction/gt_evaluation.py`, read **after** rigid Phase 2 and the
reduction have both finished. GT never enters either.

The axis geometry comes from `<source>/pointcloud/joint_metadata.json`
(`axis_vector`, `point_of_application`, frame `render_world`), which exists
for all five scenes. The type and `q(t)` come from `metadata/joint_info.json`
and `metadata/frame_values.csv`.

Reported: `alpha = arccos(|e . e_gt|)` in degrees for both models, and the
shortest line-to-line distance for R — with the parallel case handled
separately, since the skew formula divides by `|e1 x e2|`, which is zero
exactly when the two axes agree. No axis-point error is reported for P: a
prismatic joint has no distinguished point on its axis, and reporting one
would be a fabricated accuracy metric.

`q` is compared as **increments**. The dataset's `q` is absolute with a joint
offset while the reduction's `s_t`/`theta_t` start at zero at the canonical
frame, so only increments are comparable; the sign gauge is resolved by
correlation.

---

## 21. Legacy direct-P/R comparison

Same five scenes, same frozen coherence supports. GT axis angle error, in
degrees, for whichever model matches the GT type:

| scene | GT type | legacy direct-axis RGB-D | **rigid + reduction** | factor |
|---|---|---|---|---|
| stapler_103111 | revolute | 89.22 | **0.23** | 388x |
| USB_100109 | revolute | 1.48 | **0.37** | 4x |
| fridge_10905 | revolute | 6.84 | **0.07** | 98x |
| storage_45135 | prismatic | 72.91 | **0.23** | 317x |
| blade_103706 | prismatic | 0.14 | **0.08** | 1.8x |

Revolute line error, where both produce an axis point:

| scene | legacy | **rigid + reduction** |
|---|---|---|
| stapler_103111 | 0.1695 | **0.0023** |
| USB_100109 | 0.3444 | **0.0041** |
| fridge_10905 | 0.0094 | **0.0069** |

The new pipeline is better on all five, decisively on four. The two
catastrophic legacy failures — stapler at 89 degrees and storage at 73 — are
both the same mechanism: the image optimiser was free to trade axis position
against angle, and nothing in the pixels stopped it. The new pipeline removes
that freedom rather than regularising it.

Both legacy entry points now announce themselves as the ablation branch on
every run, and `tools/run_complete_joint_debug.py --phase2-motion-model
rigid-se3` refuses with a pointer rather than silently running something else.
The production path is documented in `JOINT_PIPELINE.md`.

---

## 22. Unresolved issues

1. **`gamma_1DOF = 0.9223` on fridge** is the lowest of the five, on the
   largest support, while that scene produces the *best* axis of the five
   (0.07 degrees). So the rank-1 shortfall is in the per-frame increments and
   not in the finite rotations they integrate to — most likely the
   increment-level noise that the dropped cross-covariance (item 2) would
   otherwise absorb. Worth understanding, not currently costing accuracy.

2. **Inter-frame pose cross-covariance is not available.** `P_zeta` uses
   adjacent marginals only. The direction of the error is known and
   conservative (§11), but a filter that retained the full trajectory
   covariance would weight the reduction better. Section 27 permits the
   approximation and requires it be documented; it is.

3. **The refinement's finite-difference Jacobians put a `~1e-7` floor** on the
   recovered geometry. Harmless against real data, visible in the synthetic
   tests. Analytic Jacobians would remove it if it ever matters.

4. **The left/right chart deviation in §14** is a deliberate departure from
   the spec's literal formula. Both residuals are written out so the choice
   can be audited; if the spec's pairing is wanted instead, it is one argument.

5. **No joint-type classifier exists**, by design (§26, §35). The results
   make the shape of one clear, though: `E_R/E_P` alone calls storage
   revolute (ratio 0.863 on a prismatic joint), while
   `axis_radius_over_part_scale` separates the five cleanly (1.7–2.8 versus
   62–290). Any classifier built later should use the radius ratio and not
   the energy ratio alone. Deciding the threshold is still open and is not
   decided here.

6. **Storage's coherence recall (0.532)** is unchanged by this work and still
   open. A support that misses half the drawer is a Phase-2 input problem, and
   §2's asymmetry says it costs evidence rather than correctness — but it
   costs evidence.

---

## Files

| path | what |
|---|---|
| `estimator/rigid_part_se3.py` | the state, exact transport, analytic JVP/VJP |
| `estimator/rigid_part_runtime.py` | process model, Kabsch init, the window |
| `estimator/joint_manifold.py` | `se3_log`, `se3_adjoint` added |
| `joint_reduction/rigid_trajectory_reduction.py` | sections 24–35 |
| `joint_reduction/gt_evaluation.py` | section 39 |
| `debug/rigid_debug.py` | sections 22, 23, 40, 41 |
| `tools/run_rigid_se3_pipeline.py` | the section 44 pipeline, section 45 tree |
| `tests/test_rigid_se3_phase2.py` | RSE3-1…9, section 20, the P0 gates |
| `run_rigid_se3_all.sh` | all five datasets |
