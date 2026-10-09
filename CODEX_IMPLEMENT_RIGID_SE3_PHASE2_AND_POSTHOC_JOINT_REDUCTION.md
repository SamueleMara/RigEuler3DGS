# CODEX — REPLACE DIRECT P/R RENDERER TRAINING WITH EXACT RIGID-SE(3) PHASE 2 + POST-HOC JOINT REDUCTION

## 0. Goal

Simplify the articulated-motion pipeline.

The production architecture should become:

\[
\boxed{
\text{Phase 1 free motion}
\rightarrow
\text{coherence static/dynamic support}
\rightarrow
\text{exact rigid }SE(3)\text{ Phase 2}
\rightarrow
\text{joint reduction}
}
\]

The key design change is:

> **Phase 2 must estimate only the rigid motion of the selected moving part.**
>
> It must NOT estimate a prismatic direction, revolute axis, revolute center, screw axis, or joint type while fitting RGB-D.

The physical joint is inferred **after** the best rigid-body trajectory has been recovered.

This removes the current coupling:

\[
(e_R,a_R,\theta_t)
\longleftrightarrow
\text{render loss}
\]

that allows far-axis revolute solutions to mimic translations and gives physically wrong axes such as the current Stapler result.

---

# 1. Production static/dynamic support

The motion-coherence selector is now the production default.

Use only the final high-confidence:

```text
MOVING_RIGID
```

support.

Do not pass:

```text
INCOHERENT_OR_AMBIGUOUS
```

to Phase 2.

False-positive dynamic points are more dangerous than false negatives:

- a false negative only removes some useful moving evidence;
- a false positive forces a truly static Gaussian to undergo the estimated rigid transform and directly biases the rigid motion.

Therefore the production policy is deliberately precision-first.

The old Otsu/current selector remains available only as an explicit ablation:

```text
--phase2-support-source current
```

The default is:

```text
--phase2-support-source coherence
```

No silent fallback.

---

# 2. Freeze the support before rigid Phase 2

Let:

\[
\mathcal D
\]

be the coherence-selected moving Gaussian indices.

This support is frozen for the whole rigid Phase 2.

All static Gaussians:

\[
i\notin\mathcal D
\]

remain canonical and are still rendered because they contribute to:

- visibility;
- alpha compositing;
- occlusion;
- RGB;
- depth.

But their geometry is not moved by the rigid state.

---

# 3. Do not use pairwise rigidity as a penalty

The desired physical constraint is:

\[
\|x_i(t)-x_j(t)\|
=
\|x_i(0)-x_j(0)\|
\]

for all selected moving Gaussians.

Do NOT implement this as thousands of pairwise soft penalties.

Instead parameterize the entire moving part by one exact rigid transform:

\[
\boxed{
T_t=
\begin{bmatrix}
R_t&p_t\\
0&1
\end{bmatrix}
\in SE(3).
}
\]

Then every moving Gaussian shares exactly the same transform.

This makes rigidity exact by construction.

---

# 4. Exact state-to-Gaussian map

For every:

\[
i\in\mathcal D
\]

and frame \(t\):

\[
\boxed{
\mu_{i,t}=R_t\mu_i^0+p_t.
}
\]

Gaussian orientation must also move rigidly:

\[
\boxed{
R_{i,t}=R_tR_i^0.
}
\]

If the renderer uses covariance:

\[
\boxed{
\Sigma_{i,t}
=
R_t\Sigma_i^0R_t^\top.
}
\]

Static Gaussians remain:

\[
\mu_{i,t}=\mu_i^0,
\qquad
R_{i,t}=R_i^0
\]

for:

\[
i\notin\mathcal D.
\]

This is non-negotiable.

Do not update Gaussian centers without updating Gaussian orientation/covariance.

---

# 5. Exact rigidity proof

For any two selected moving Gaussians:

\[
\mu_{i,t}-\mu_{j,t}
=
R_t(\mu_i^0-\mu_j^0).
\]

Therefore:

\[
\begin{aligned}
\|\mu_{i,t}-\mu_{j,t}\|^2
&=
(\mu_i^0-\mu_j^0)^T
R_t^TR_t
(\mu_i^0-\mu_j^0)
\\
&=
\|\mu_i^0-\mu_j^0\|^2.
\end{aligned}
\]

Hence:

\[
\boxed{
\text{shared }SE(3)
\Rightarrow
\text{exact rigid-body motion}.
}
\]

No rigidity regularizer is required.

---

# 6. Mathematical connection to the coherence filter

The coherence filter uses:

\[
r_{ij}^T(v_j-v_i)=0.
\]

A rigid-body velocity field is:

\[
\boxed{
v_i=v+\omega\times x_i.
}
\]

Then:

\[
v_j-v_i
=
\omega\times(x_j-x_i)
=
\omega\times r_{ij},
\]

so:

\[
r_{ij}^T(v_j-v_i)
=
r_{ij}^T(\omega\times r_{ij})
=
0.
\]

Therefore the method has one continuous physical principle:

```text
Phase 1:
    detect points whose free trajectories approximately satisfy rigid metric coherence

Phase 2:
    constrain those points exactly to the rigid-motion nullspace

Joint reduction:
    determine whether the recovered rigid trajectory lies on a 1-DOF joint subgroup
```

This continuity is intentional.

---

# 7. New production Phase 2 state

Create a new module:

```text
estimator/rigid_part_se3.py
```

and runtime:

```text
estimator/rigid_part_runtime.py
```

For K=1 use one rigid part.

Recommended nominal state per current frame:

\[
\boxed{
x_t=(T_t,\xi_t)
}
\]

where:

\[
T_t\in SE(3)
\]

and:

\[
\xi_t=
\begin{bmatrix}
v_t\\
\omega_t
\end{bmatrix}
\in\mathbb R^6
\]

is the spatial/world rigid velocity.

Local error state:

\[
\boxed{
\delta x_t=
[
\delta\tau_t,\delta\xi_t
]
\in\mathbb R^{12}.
}
\]

Here:

\[
\delta\tau_t=
\begin{bmatrix}
\delta v\\
\delta\omega
\end{bmatrix}.
\]

If the existing IEKF architecture has an already validated equivalent rigid-pose + rigid-velocity state, reuse that implementation rather than duplicating it.

Do not reintroduce joint geometry into this state.

---

# 8. Gauge

Frame 0 is canonical:

\[
\boxed{
T_0=I.
}
\]

This is a gauge, not an estimate.

The whole trajectory is relative to the canonical static 3DGS at \(t=0\).

---

# 9. Rigid process model

Use the true timestamp interval:

\[
\Delta t_t.
\]

Prediction:

\[
\boxed{
T_{t+1}^{-}
=
\Exp(\Delta t_t\,\xi_t^\wedge)\,T_t^{+}.
}
\]

Velocity:

\[
\boxed{
\xi_{t+1}^{-}
=
\xi_t^{+}+w_{\xi,t}.
}
\]

Use the same process/covariance philosophy already validated in Phase 1.

Do not invent a joint-specific process model here.

The state is a free rigid body.

---

# 10. Pose retraction

Use left/spatial perturbation consistently with the existing renderer and Phase-1 conventions:

\[
\boxed{
T^+
=
\Exp(\delta\tau^\wedge)\,T.
}
\]

If current manifold utilities already implement this convention, reuse them.

Do not mix left and right perturbations.

---

# 11. State-to-Gaussian JVP

At the current transformed Gaussian center:

\[
\mu_i
\]

a left perturbation gives:

\[
\boxed{
\delta\mu_i
=
\delta v+\delta\omega\times\mu_i.
}
\]

Therefore:

\[
\boxed{
\delta\mu_i
=
\begin{bmatrix}
I&-[\mu_i]_\times
\end{bmatrix}
\delta\tau.
}
\]

Gaussian orientation tangent:

\[
\boxed{
\delta\phi_i=\delta\omega.
}
\]

Hence the 6D pose tangent per moving Gaussian is:

\[
\boxed{
\begin{bmatrix}
\delta\mu_i\\
\delta\phi_i
\end{bmatrix}
=
\begin{bmatrix}
I&-[\mu_i]_\times\\
0&I
\end{bmatrix}
\delta\tau.
}
\]

The instantaneous image has no direct derivative with respect to rigid velocity:

\[
\boxed{
H_{\xi}=0.
}
\]

Velocity becomes observable through process cross-covariance.

---

# 12. State-to-Gaussian VJP

Given renderer pose adjoints per moving Gaussian:

\[
\bar\mu_i,
\qquad
\bar\phi_i,
\]

the rigid pose adjoint is:

\[
\boxed{
\bar v
=
\sum_i\bar\mu_i
}
\]

and:

\[
\boxed{
\bar\omega
=
\sum_i
\left(
\mu_i\times\bar\mu_i
+
\bar\phi_i
\right).
}
\]

Thus:

\[
\boxed{
J_{G,T}^T
\{\bar\mu_i,\bar\phi_i\}
=
\begin{bmatrix}
\bar v\\
\bar\omega
\end{bmatrix}.
}
\]

Implement analytically.

Add adjoint-identity tests.

---

# 13. Renderer composition

Do not change the working Phase-1 renderer linearization.

Let:

\[
H_G
\]

be the existing Gaussian-pose RGB-D Jacobian operator.

Then:

\[
\boxed{
H_T
=
H_GJ_{G,T}.
}
\]

JVP:

```text
rigid tangent
→ moving-Gaussian pose tangents
→ existing renderer JVP
```

VJP:

```text
renderer VJP
→ moving-Gaussian pose adjoints
→ rigid-pose adjoint
```

Everything after the state-to-Gaussian map is reused from Phase 1.

---

# 14. Measurement model

Use exactly the current production RGB-D model:

\[
\boxed{
R_{\rm eff}
=
R_{\rm sensor}
+
R_{\rm model}.
}
\]

Reuse:
- RGB residual;
- depth residual;
- visibility;
- alpha compositing;
- robust weighting;
- camera pooling;
- nonlinear rerender acceptance;
- trust region;
- PCG / dense small-state solve where appropriate.

Do not build a separate rigid-specific image loss.

---

# 15. Information-form IEKF

At each update solve the same form as Phase 1:

\[
\boxed{
(\Lambda^-+H_T^TR^{-1}H_T)\delta x
=
H_T^TR^{-1}r
}
\]

with the existing prior-offset/trust-region semantics.

The only new component is the exact rigid state-to-Gaussian map.

---

# 16. Initialization from Phase-1 coherent trajectories

The coherence support already gives actual free Phase-1 trajectories.

Use them to initialize the rigid pose trajectory.

For every frame fit weighted Kabsch:

\[
\boxed{
(R_t^{P1},p_t^{P1})
=
\arg\min_{R,p}
\sum_{i\in\mathcal D}
w_i
\|
x_i^{P1}(t)-(Rx_i^0+p)
\|^2.
}
\]

Prefer coherence supernode trajectories / stable representatives if they are cleaner than raw Gaussian tracks.

Use one documented weighting policy.

Frame 0 must be identity.

Initialize rigid velocity from:

\[
\boxed{
\xi_t^{init}
=
\frac{1}{\Delta t_t}
\Log
\left(
T_{t+1}^{P1}
(T_t^{P1})^{-1}
\right)^\vee.
}
\]

This is initialization only.

RGB-D Phase 2 must be able to correct it.

---

# 17. Do not initialize through an axis or joint model

Forbidden for the new production rigid Phase 2:

```text
fit screw axis
→ initialize T_t
```

or:

```text
fit P/R
→ initialize T_t
```

Initialization comes directly from Phase-1 rigid Kabsch on the coherence support.

There is no joint geometry yet.

---

# 18. Phase-2 output

The primary output of rigid Phase 2 is:

\[
\boxed{
\hat T_0,\hat T_1,\ldots,\hat T_{T-1}
}
\]

plus uncertainty:

\[
\boxed{
P_{T,t}
}
\]

and rigid velocity:

\[
\hat\xi_t.
\]

Persist:

```text
phase2_rigid/
    poses.npy
    velocities.npy
    pose_covariances.npy
    state_trace.csv
    per_frame_camera_energy.csv
    renders/
```

Do not report any P/R axis from this stage.

---

# 19. Numerical rigid Phase-2 debug

Per frame report:

```text
frame
time

tx ty tz
rotvec_x rotvec_y rotvec_z
rotation_angle

vx vy vz
wx wy wz

rgb_energy
depth_energy
total_energy

accepted_steps
rejected_steps
trust_radius
pcg_iterations

pose_information_eigs
velocity_information_eigs
```

---

# 20. Exact rigidity regression

For sampled moving Gaussian pairs verify:

\[
\boxed{
\max_{i,j,t}
\left|
\|\mu_i(t)-\mu_j(t)\|
-
\|\mu_i(0)-\mu_j(0)\|
\right|
}
\]

is at numerical precision.

Also verify:

\[
R_i(t)=R_tR_i^0
\]

and covariance transport.

This is a P0 gate.

---

# 21. Compare rigid RGB-D refinement with Phase-1 Kabsch

For every frame compare:

\[
T_t^{P1\text{-Kabsch}}
\]

with:

\[
T_t^{RGBD\text{-rigid}}.
\]

Rotation difference:

\[
\boxed{
d_R(t)
=
\left\|
\Log
\left(
(R_t^{P1})^TR_t^{RGBD}
\right)^\vee
\right\|.
}
\]

Translation difference:

\[
\boxed{
d_p(t)
=
\|
p_t^{P1}-p_t^{RGBD}
\|.
}
\]

This is a crucial diagnostic.

If rigid Phase 2 diverges strongly from a highly coherent Phase-1 rigid trajectory, inspect the RGB-D objective before doing any joint reduction.

---

# 22. Fixed-camera rendering

For one selected camera, render every frame:

```text
GT
Phase1
Phase1-Kabsch rigid
Phase2 rigid
```

for:
- RGB;
- depth;
- residuals.

This tells us whether direct rigid RGB-D refinement actually improves the free-motion rigid projection.

---

# 23. Definitive rigid 3D HTML

Create:

```text
html/rigid_phase2_debug.html
```

Frame slider over all frames.

Layers:

```text
canonical static scene
coherence moving support
Phase1 free moving centers
Phase1 Kabsch rigid centers
Phase2 RGB-D rigid centers
```

Also show:
- COM trajectories;
- rigid velocity vectors;
- camera frustums.

Do NOT show a joint axis in this Phase-2 view.

The point is to inspect rigid-body reconstruction independently from joint inference.

---

# PART II — JOINT REDUCTION AFTER RIGID MOTION

# 24. Joint reduction is a separate module

Create:

```text
joint_reduction/rigid_trajectory_reduction.py
```

Input:

```text
T_t
P_T_t
timestamps
canonical moving-part geometry
```

Output:

```text
one-DOF consistency
P model
R model
joint uncertainties
diagnostics
```

This module must not call the RGB-D renderer.

It operates only on the recovered rigid trajectory and its uncertainty.

---

# 25. Relative rigid increments

For the world/spatial convention use:

\[
\boxed{
\Delta T_t
=
T_{t+1}T_t^{-1}.
}
\]

Then:

\[
\boxed{
\zeta_t
=
\frac{1}{\Delta t_t}
\Log(\Delta T_t)^\vee
\in\mathbb R^6.
}
\]

Use one convention everywhere and document it.

Do not mix:

\[
T_{t+1}T_t^{-1}
\]

with:

\[
T_t^{-1}T_{t+1}.
\]

---

# 26. Dimensionless 1-DOF trajectory test

Translation and rotation have different units.

Let \(L\) be canonical part scale:

\[
L^2
=
\frac1N
\sum_i
\|\mu_i^0-c_0\|^2.
\]

Define:

\[
\boxed{
\bar\zeta_t
=
\begin{bmatrix}
v_t/L\\
\omega_t
\end{bmatrix}.
}
\]

Build:

\[
Z=
[
\bar\zeta_0,
\ldots,
\bar\zeta_{T-2}
].
\]

A fixed one-DOF joint implies:

\[
\boxed{
\zeta_t
=
\dot q_t\xi_J
}
\]

and therefore:

\[
\boxed{
\operatorname{rank}(Z)\simeq1.
}
\]

Compute weighted SVD.

Report:

\[
\boxed{
\gamma_{1DOF}
=
\frac{\sigma_1^2}{\sum_k\sigma_k^2}.
}
\]

Do not classify solely from a threshold on this value yet.

Use it as a structural diagnostic.

---

# 27. Weight the reduction by Phase-2 uncertainty

A poorly observed frame must contribute less.

Propagate pose covariance into relative-twist covariance:

\[
P_{\zeta,t}.
\]

If full trajectory cross-covariance is available, use it.

Otherwise use a first-order approximation from the adjacent pose marginals and clearly document the approximation.

All reduced-model fits should be Mahalanobis weighted whenever possible.

---

# 28. Fit the prismatic model from the rigid trajectory

For a perfect prismatic joint:

\[
R_t=I
\]

and:

\[
p_t=s_t e_P.
\]

Fit:

\[
\boxed{
e_P
=
\arg\min_{\|e\|=1,\{s_t\}}
\sum_t
\|
p_t-s_te
\|_{W_t}^2
}
\]

while also accounting for the observed rotational component.

Equivalent initialization:
- weighted PCA of translations \(p_t\);
- \(s_t=e_P^Tp_t\).

Then optionally refine:

\[
e_P,\{s_t\}
\]

against the recovered rigid poses only.

No renderer.

P model:

\[
\boxed{
T_P(s_t)
=
\begin{bmatrix}
I&s_te_P\\
0&1
\end{bmatrix}.
}
\]

---

# 29. Fit the revolute axis direction from finite rotations

For the rigid pose trajectory:

\[
R_t.
\]

Compute:

\[
\boxed{
\phi_t
=
\Log(R_t)^\vee.
}
\]

For a true revolute joint:

\[
\boxed{
\phi_t
=
\theta_t e_R.
}
\]

Thus the matrix:

\[
\Phi
=
[
\phi_1,\ldots,\phi_{T-1}
]
\]

is rank 1.

Estimate:

\[
\boxed{
e_R
=
\arg\max_{\|e\|=1}
\sum_t
w_t
(e^T\phi_t)^2.
}
\]

This is a weighted principal eigenvector/SVD problem.

Resolve the sign gauge consistently through time.

Report:
- singular values;
- rank-1 explained fraction;
- per-frame axis alignment.

---

# 30. Fit the revolute axis point from finite transforms

Do NOT use:

\[
a
=
\frac{\omega\times v}{\|\omega\|^2}
\]

as the primary finite-axis estimator.

For a true rotation around fixed axis point \(a\):

\[
\boxed{
p_t
=
(I-R_t)a.
}
\]

Therefore solve:

\[
\boxed{
a^\star
=
\arg\min_a
\sum_t
w_t
\|
(I-R_t)a-p_t
\|^2.
}
\]

Apply the line gauge:

\[
\boxed{
e_R^T(a-c_0)=0.
}
\]

This gives the point on the recovered axis closest to the canonical moving-part COM.

Solve the constrained small linear system by SVD/QR.

Persist:
- singular values;
- condition number;
- finite-axis residual;
- axis radius from COM.

---

# 31. Revolute angular trajectory

Given \(e_R\):

\[
\boxed{
\theta_t
=
e_R^T\phi_t.
}
\]

Unwrap consistently in time.

Estimate:

\[
\dot\theta_t
\]

using true timestamps.

Use sign convention:

\[
(e_R,\theta_t)
\equiv
(-e_R,-\theta_t)
\]

and choose one continuous gauge.

---

# 32. Refine P and R in pose space, not image space

After initialization, refine both physical models against:

\[
\hat T_t
\]

from rigid Phase 2.

Define the pose residual:

\[
\boxed{
r_t^M
=
\Log
\left(
T_M(q_t;\theta_M)^{-1}
\hat T_t
\right)^\vee.
}
\]

For:

\[
M\in\{P,R\}.
\]

Optimize:

\[
\boxed{
E_M
=
\sum_t
(r_t^M)^T
W_t
r_t^M
}
\]

where:

\[
W_t
\approx
P_{T,t}^{-1}.
\]

If full trajectory covariance is available, use the joint quadratic form instead of independent frame weights.

This optimization is tiny and must not call the renderer.

---

# 33. Physical reduced models

Prismatic:

\[
\boxed{
T_P(s)
=
\begin{bmatrix}
I&s e_P\\
0&1
\end{bmatrix}.
}
\]

Revolute:

\[
\boxed{
T_R(\theta)
=
\begin{bmatrix}
R(e_R,\theta)&a_R-R(e_R,\theta)a_R\\
0&1
\end{bmatrix}.
}
\]

The fitted geometry is constant across all frames.

Only \(s_t\) or \(\theta_t\) varies.

---

# 34. Do not let joint reduction modify the rigid trajectory

The recovered:

\[
\hat T_t
\]

is the measurement/output of rigid Phase 2.

P/R reduction fits to it.

Do not send the reduced P/R motion back into the renderer in the first production version.

Do not modify:

\[
\hat T_t
\]

to make the joint fit look better.

This separation is central to the new design.

---

# 35. Model comparison happens in pose space

Report:

```text
P weighted residual
R weighted residual
1-DOF rank statistic
P geometry uncertainty
R geometry uncertainty
```

Do not use direct P/R RGB-D training loss anymore in the production branch.

Old direct P/R RGB-D estimators may remain as explicit ablations only.

---

# 36. Why this removes the current Stapler failure mode

Current architecture:

```text
pixels
→ directly optimize axis direction + axis position + theta
```

allows:

```text
far-away axis + small/moderate theta
≈ translation-like image motion
```

and can converge to a physically meaningless axis.

New architecture:

```text
pixels
→ directly estimate R_t and p_t
```

first.

The rigid pose has no axis ambiguity.

Only afterward do we solve:

\[
p_t=(I-R_t)a
\]

and:

\[
\Log(R_t)^\vee=\theta_t e_R.
\]

Thus the axis is inferred from the already recovered finite rotation sequence rather than used as the coordinate system of the image optimizer.

---

# 37. Required Stapler forensic rerun

Run:

```text
stapler_103111
```

through the new production pipeline.

Save:

```text
Phase1 free trajectories
coherence support
Phase1 Kabsch rigid trajectory
RGB-D rigid Phase2 trajectory
P reduction
R reduction
GT joint geometry when available
```

The main questions:

1. Does rigid Phase 2 recover sensible \(R_t,p_t\)?
2. How close is it to Phase1 Kabsch?
3. Are the finite rotations rank-1?
4. Does finite R reduction recover the correct axis direction?
5. Does axis point solve remain finite and well-conditioned?
6. Is the old far-axis degeneracy gone from the renderer optimization?

---

# 38. Negative controls

Use:

## Blade
Expected primarily prismatic.

The rigid Phase-2 trajectory should contain:
- strong translation;
- negligible coherent rotation.

P reduction should explain it well.

## Storage
Another near-P control.

## USB / Fridge / Stapler
Revolute controls.

The rigid trajectory should reveal coherent finite rotation before any R axis is computed.

---

# 39. GT joint evaluation

When true joint metadata is available, evaluate after reduction only.

For R:

\[
\boxed{
\alpha_R
=
\cos^{-1}
(|e_R^Te_R^{GT}|)
}
\]

and shortest line-to-line axis error.

For P:

\[
\boxed{
\alpha_P
=
\cos^{-1}
(|e_P^Te_P^{GT}|)
}
\]

if GT exists.

Also compare q trajectory after sign/gauge alignment when available.

GT must never enter rigid Phase-2 optimization.

---

# 40. Debug HTML architecture

Create one definitive:

```text
html/rigid_to_joint_debug.html
```

Two tabs/stages.

## Stage A — Rigid reconstruction

Show:
```text
Phase1 free
Phase1 Kabsch
RGB-D rigid Phase2
```

No joint axis required.

## Stage B — Joint reduction

Show the same rigid Phase-2 centers plus overlays:
```text
P direction
R axis
GT axis when available
```

Also show P/R reconstructed pose trajectories as optional overlays, but never replace the actual rigid Phase-2 centers.

---

# 41. Required timelines

Plot:

```text
translation p_x,p_y,p_z
rotation vector phi_x,phi_y,phi_z
rotation angle
rigid velocity v
rigid angular velocity omega

1-DOF singular values / explained fraction

P s(t)
R theta(t)

P pose residual
R pose residual

pose uncertainty
```

For Stapler, the user must be able to see whether axis failure originates from:
- wrong rigid rotations;
- or wrong reduction of correct rotations.

---

# 42. Synthetic tests

Add:

## RSE3-1 — arbitrary rigid trajectory
Verify exact Gaussian rigidity and JVP/VJP.

## RSE3-2 — pure translation
Rigid Phase 2 recovers translation; joint reduction gives P.

## RSE3-3 — finite revolute
Rigid Phase 2 recovers finite poses; reduction recovers axis direction/line.

## RSE3-4 — variable speed, stop, reversal
Rigid trajectory remains valid; rank-1 joint generator remains consistent.

## RSE3-5 — wrong Phase1 Kabsch initializer
RGB-D rigid Phase 2 corrects the pose.

## RSE3-6 — noisy/weak late frames
Pose covariance grows; joint reduction downweights them.

## RSE3-7 — far-axis direct-R trap
Construct data where a direct R pixel optimizer can fit a far axis.
Show that rigid-first estimation recovers the actual finite pose and reduction recovers the true axis.

## RSE3-8 — coherence false negatives
Remove part of the moving support but keep it pure.
Rigid pose/axis should remain stable.

## RSE3-9 — static false-positive contamination
Inject static points into support and demonstrate quantitatively why the precision-first coherence policy is important.

---

# 43. Legacy branches

Keep the current:

```text
generic screw RGB-D
direct P RGB-D
direct R RGB-D
```

only under explicit legacy/ablation flags.

For example:

```text
--phase2-motion-model rigid-se3       # production default
--phase2-motion-model legacy-screw
--phase2-motion-model legacy-direct-p
--phase2-motion-model legacy-direct-r
```

Do not remove them yet; they remain valuable for regression and paper ablations.

But they must no longer define the production joint geometry.

---

# 44. New default pipeline

The intended default command path is:

```text
Phase1
  ↓
coherence MOVING_RIGID support
  ↓
rigid SE(3) RGB-D Phase2
  ↓
joint reduction P + R
  ↓
pose-space model comparison
```

No direct axis parameter appears inside the RGB-D optimizer.

---

# 45. Required output tree

```text
<scene_output>/
    support_coherence/

    phase1/
        ...

    phase2_rigid/
        poses.npy
        velocities.npy
        pose_covariances.npy
        state_trace.csv
        per_frame_camera_energy.csv
        phase1_kabsch_poses.npy
        renders/

    joint_reduction/
        one_dof_diagnostics.json

        prismatic/
            geometry.json
            trajectory.csv
            pose_residuals.csv

        revolute/
            geometry.json
            trajectory.csv
            pose_residuals.csv
            finite_axis_solver.json

    html/
        rigid_phase2_debug.html
        rigid_to_joint_debug.html
```

---

# 46. P0 acceptance gates

Do not claim completion unless:

1. coherence support is production default;
2. only `MOVING_RIGID` enters Phase 2;
3. exact shared \(SE(3)\) state-to-Gaussian map is implemented;
4. Gaussian centers and orientations both move rigidly;
5. pairwise distances are preserved numerically;
6. rigid JVP passes finite differences;
7. rigid VJP passes adjoint identity;
8. frame 0 remains identity;
9. Phase1 Kabsch initialization works;
10. RGB-D rigid Phase2 renders all frames;
11. pose covariance/information is persisted;
12. joint reduction does not call renderer;
13. 1-DOF rank diagnostic is implemented;
14. P reduction is implemented;
15. finite-motion R axis direction fit is implemented;
16. constrained finite axis-point solve is implemented;
17. uncertainty-weighted pose-space refinement is implemented;
18. Stapler rerun completed;
19. Blade and Storage negative controls completed;
20. USB/Fridge R controls completed;
21. definitive HTML exists.

---

# 47. Required report

Create:

```text
RIGID_SE3_PHASE2_AND_JOINT_REDUCTION_REPORT.md
```

Required sections:

1. why direct P/R image optimization was removed from production;
2. coherence support used;
3. rigid \(SE(3)\) state;
4. exact Gaussian transport;
5. JVP/VJP derivation;
6. IEKF process;
7. Phase1 Kabsch initialization;
8. rigid Phase2 numerical tests;
9. Phase1-Kabsch vs RGB-D rigid comparison;
10. pose covariance;
11. 1-DOF rank analysis;
12. prismatic reduction;
13. revolute finite-axis reduction;
14. pose-space P/R refinement;
15. Stapler results;
16. Blade results;
17. Storage results;
18. USB results;
19. Fridge results;
20. GT axis evaluation;
21. legacy direct-P/R comparison;
22. unresolved issues.

---

# 48. Implementation order

Do exactly:

```text
1. Make coherence MOVING_RIGID the production default support.
2. Add rigid_part_se3.py.
3. Implement exact rigid center/orientation/covariance transport.
4. Implement analytic rigid JVP/VJP.
5. Add FD + adjoint tests.
6. Implement rigid pose/velocity process.
7. Add Phase1 Kabsch initializer.
8. Wire rigid state into existing RGB-D IEKF.
9. Produce rigid Phase2 poses + uncertainty.
10. Build rigid Phase2 numerical/visual debugger.
11. Implement rigid_trajectory_reduction.py.
12. Implement dimensionless one-DOF rank test.
13. Implement P finite-trajectory fit.
14. Implement R finite-rotation axis-direction fit.
15. Implement constrained finite axis-point solve.
16. Implement uncertainty-weighted pose-space P/R refinement.
17. Add synthetic RSE3-1 ... RSE3-9.
18. Rerun Stapler.
19. Rerun Blade + Storage.
20. Rerun USB + Fridge.
21. Build rigid_to_joint_debug.html.
22. Write the final report.
```

---

# 49. Final methodological statement

The production estimator should no longer ask the renderer:

\[
\text{“where is the revolute axis?”}
\]

The renderer should answer only:

\[
\boxed{
\text{“what rigid pose }T_t\text{ best explains the moving part at this frame?”}
}
\]

The physical joint is then inferred from the recovered rigid trajectory:

\[
\boxed{
\{T_t,P_t\}
\longrightarrow
\text{1-DOF generator}
\longrightarrow
P\text{ or }R.
}
\]

This keeps the high-dimensional image estimation problem and the low-dimensional joint-geometry problem separate.

The underlying physics is continuous from Phase 1 to the end:

\[
\boxed{
\text{coherence identifies the rigid/Killing field}
}
\]

\[
\boxed{
\text{rigid Phase 2 estimates the finite }SE(3)\text{ motion}
}
\]

\[
\boxed{
\text{joint reduction extracts the one-DOF generator}.
}
\]
