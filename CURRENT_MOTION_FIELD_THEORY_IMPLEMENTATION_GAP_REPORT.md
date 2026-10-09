# Dynamic 3DGS Motion Field — Theory, Current Implementation, and Theory–Implementation Gap

> **Historical report (v11.1, 2026-08-12).** This document analyzes the
> pre-pruning multi-representation repository and is not a map of the retained
> executable code. See [`PIPELINE.md`](PIPELINE.md) for the current fixed-curl
> pipeline, revised command, optical-flow rasterizers, and live loss graph.

**Current implementation basis:** `generic_spatial_coherence_v11_1_20260812`  
**Main branches compared:** fixed Eulerian CurlGrid, per-Gaussian Lagrangian point-strain, adaptive Eulerian CurlOctree  
**Primary supervision:** RGB/depth/DINO rendering + observed 2-D optical flow + multiview 2-D→3-D velocity inversion  
**Primary objective of Phase 1:** recover a physically meaningful motion field and a static/moving material partition **without assuming a joint axis, pivot, screw pitch, or rigid-body twist parameters**.

---

## Executive summary

The current system is attempting to solve two coupled problems:

1. **Material partitioning:** determine which Gaussians are static and which belong to one or more moving parts.
2. **Motion reconstruction:** infer the time-dependent 3-D velocity field that transports the moving Gaussians while remaining physically coherent.

The central theoretical idea is sound:

- use the video observations to infer motion;
- use multiview optical flow to provide direct 3-D velocity evidence;
- use DINO only as a **material-boundary cue**, not as a motion estimator;
- keep the first motion stage axis-free;
- enforce local rigidity from continuum kinematics through
  \[
  D = \frac12\left(\nabla v + \nabla v^T\right) \rightarrow 0;
  \]
- in the Eulerian branches, parameterize the raw velocity as a curl field
  \[
  v = \nabla\times A,
  \]
  so incompressibility is architectural rather than learned;
- compare that Eulerian representation against a per-Gaussian Lagrangian representation using as much of the same supervision and physics as possible.

However, the current implementation still differs from this theory in several important ways.

The most important current mismatch is that **local non-deformation is implemented, but direct spatial coherence of the velocity field is not yet enforced in a robust first-order way**. The current v11 implementation adds spatial coherence through the fitted velocity gradient \(G\) and a local affine consistency residual, but these are derivative-based quantities. The complete debug shows that the measured multiview 3-D velocities are often very coherent while the optimized field becomes incoherent, especially at later frames and at weakly observed locations. Therefore the missing mechanism is not simply “more rigidity”; it is a correctly normalized, robust way to propagate coherent velocity information spatially through same-part material without overwhelming the observational data.

A second major gap is **loss scaling**. The physics losses, direct observed-velocity losses, dense 2-D flow losses, RGB/depth/DINO rendering losses, and static constraints do not currently share a principled dimensionless normalization. This makes it possible for the optimizer to prefer a small or nearly zero field even when multiview optical flow clearly indicates nonzero motion.

A third major gap is the role of DINO. In theory DINO should answer:

> “Is a static/dynamic or part-to-part discontinuity plausible here?”

It should not answer:

> “How much should two points already assigned to the same moving part obey the same physics?”

The current implementation still multiplies same-moving-part mechanical coupling by DINO similarity. That is too aggressive: once two neighbors are confidently assigned to the same moving part, their motion coherence should be a physical requirement independent of appearance variation inside the part.

Finally, the adaptive CurlOctree still has an important causal-state issue: after a previously refined branch is coarsened and later reactivated, the old dormant fine-level vector-potential residual can become active again. The intended theory says reactivation must inherit the current parent field **without injecting stale motion**, meaning the fine residual for the current time state must be reset or reinitialized consistently.

The available debug therefore points toward a simpler next step rather than another large redesign:

\[
\boxed{
\text{multiview observed motion}
\;\rightarrow\;
\text{robust direct spatial velocity coherence}
\;\rightarrow\;
\text{axis-free local rigidity }D\to0
\;\rightarrow\;
\text{representation-specific transport}
}
\]

with DINO acting on the **material boundary problem**, not as a substitute for motion physics.

---

# 1. Research objective

The target problem is dynamic 3D Gaussian Splatting for an articulated object observed by multiple synchronized cameras. The static 3DGS reconstructed at the canonical frame provides Gaussian centers, appearance, opacity, covariance, semantic foreground information, and DINO descriptors. The video then provides multiple views of the object's motion.

The goal of Phase 1 is **not yet to estimate the joint axis**. The goal is to recover the most accurate generic motion description possible from the visual evidence.

The desired sequence is:

\[
\boxed{
\text{visual observations}
\rightarrow
\text{3-D motion field}
\rightarrow
\text{static/moving-part structure}
\rightarrow
\text{later rigid/joint interpretation}
}
\]

The first motion stage must therefore remain independent of explicit parameters such as:

- joint axis;
- pivot / rotation center;
- angular velocity parameter \(\omega\) as a trainable rigid-body variable;
- translational twist \(b\);
- screw pitch;
- revolute/prismatic/screw class.

Those quantities may be **derived later** once a coherent motion field has been recovered.

---

# 2. Common notation

Let

\[
X_i^k \in \mathbb R^3
\]

be the position of Gaussian \(i\) at video frame \(k\), with timestamp \(t_k\).

The true physical timestep is

\[
\Delta t_k=t_{k+1}-t_k.
\]

A velocity field is

\[
v(X,t)\in\mathbb R^3.
\]

For a Lagrangian Gaussian representation, each material Gaussian can own or sample

\[
v_i^k.
\]

For an Eulerian cell representation, each spatial control volume owns or samples

\[
v_c(t).
\]

The first-order transport currently used is

\[
X_i^{k+1}
=
X_i^k+
\Delta t_k\,v_i^k
\]

or the analogous cell-sampled velocity.

For the fixed CurlGrid, the implementation samples the field at the interval midpoint and applies it to the source cell of the Gaussian over that physical interval.

---

# 3. Theoretical kinematics of a rigid motion field

## 3.1 Local velocity gradient

For two nearby material points,

\[
X_j=X_i+\delta X,
\]

a smooth velocity field admits the local expansion

\[
v(X_j)
\approx
v(X_i)+G_i\delta X,
\]

where

\[
\boxed{G_i=\nabla v(X_i)}.
\]

The gradient decomposes uniquely into symmetric and antisymmetric parts:

\[
G=D+\Omega,
\]

with

\[
\boxed{D=\frac12(G+G^T)}
\]

and

\[
\boxed{\Omega=\frac12(G-G^T)}.
\]

The symmetric tensor \(D\) is the **rate-of-deformation tensor**. It contains local extension and shear.

The antisymmetric tensor \(\Omega\) contains local rigid rotation.

---

## 3.2 Why rigid motion implies \(D=0\)

For two nearby points, let

\[
r=X_j-X_i.
\]

The rate of change of squared distance is

\[
\frac{d}{dt}\|r\|^2
=
2r^T(v_j-v_i).
\]

Using the first-order expansion,

\[
v_j-v_i\approx Gr,
\]

so

\[
\frac{d}{dt}\|r\|^2
\approx
2r^TGr.
\]

Because

\[
r^T\Omega r=0
\]

for any skew-symmetric \(\Omega\), only the symmetric part contributes:

\[
\frac{d}{dt}\|r\|^2
\approx
2r^TDr.
\]

Therefore preservation of local distances for all directions requires

\[
\boxed{D=0}.
\]

This is the correct local continuum condition for a rigid velocity field.

Importantly, **zero divergence is not enough**. Since

\[
\nabla\cdot v=\operatorname{tr}(G)=\operatorname{tr}(D),
\]

incompressibility only constrains the trace of \(D\). A pure shear can have zero divergence while still deforming the object.

---

## 3.3 Rigid motion can contain nonzero curl

A general instantaneous rigid-body velocity has the form

\[
\boxed{v(X)=b+\omega\times X}
\]

for some translation component \(b\) and angular velocity \(\omega\).

Its gradient is skew-symmetric, therefore

\[
D=0,
\]

but its curl is generally nonzero:

\[
\boxed{\nabla\times v=2\omega}.
\]

Therefore **curl magnitude must never be minimized as a rigidity loss**. Doing so would suppress revolute motion.

The current implementation correctly derives

\[
\omega=\frac12\nabla\times v
\]

for diagnostics and Gaussian covariance rotation, while not optimizing \(\omega\) as an explicit Phase-1 rigid-body parameter.

---

# 4. Why the first stage should remain axis-free

If the joint axis were already known, one could directly constrain

\[
v_i=b+\omega\times X_i
\]

for one shared \((b,\omega)\) per part.

But the goal of Phase 1 is precisely to discover motion before estimating the axis. Therefore explicit optimization of \(b\), \(\omega\), axis position, or pivot would prematurely assume the structure that the method is supposed to infer.

The appropriate first-stage theory is therefore:

1. estimate a generic velocity field from observations;
2. impose non-deformation through \(D\rightarrow0\);
3. impose spatial coherence between neighboring points/cells in the same material part;
4. identify motion discontinuities / part boundaries;
5. only afterward fit a global rigid/joint model.

This means the first stage should recover a **smooth, locally rigid velocity field**, not an explicit rigid-body model.

---

# 5. Eulerian incompressibility through a vector potential

## 5.1 Continuous theory

Any sufficiently smooth vector field written as

\[
\boxed{v=\nabla\times A}
\]

satisfies

\[
\boxed{\nabla\cdot v=\nabla\cdot(\nabla\times A)=0}.
\]

This is useful because the continuity constraint does not have to be learned from a penalty.

However:

\[
\boxed{\nabla\cdot v=0\not\Rightarrow D=0}.
\]

The curl architecture guarantees incompressibility, not rigidity.

Thus the intended Eulerian physics is

\[
A
\rightarrow
v=\nabla\times A
\rightarrow
G=\nabla v
\rightarrow
D=\operatorname{sym}(G)
\rightarrow0.
\]

---

## 5.2 Gauge freedom

The vector potential is not unique:

\[
A' = A+\nabla\phi
\]

produces the same velocity because

\[
\nabla\times\nabla\phi=0.
\]

Therefore the optimization contains a **gauge null space** unless a gauge convention is imposed.

The current implementation does not explicitly impose a gauge such as

\[
\nabla\cdot A=0.
\]

This does not invalidate the velocity field, but it creates theoretically unnecessary degrees of freedom. It is especially relevant because temporal regularization is applied to \(A\) in the adaptive CurlOctree: two gauge-equivalent vector potentials can represent the same physical velocity while having different temporal penalties.

This is a current theory–implementation gap.

---

# 6. Fixed Eulerian CurlGrid — current implementation

The fixed Eulerian branch uses the public option

```text
--phase1_motion_representation eulerian_curl_grid
```

which routes to `eulerian_curl_strain`.

## 6.1 Spatial discretization

The foreground canonical bounding box is padded and divided into a regular Cartesian grid. The default target is approximately 16,384 cells, adjusted for scene aspect ratio.

The Gaussian centers are **material tracers**. The cells remain fixed in world coordinates.

At every interval, the source position determines the source cell:

\[
c_i^k=\operatorname{cell}(X_i^k).
\]

All Gaussians in the same source cell receive the same velocity for that interval.

This is the fundamental Eulerian comparison point: motion is indexed by **space**, not by persistent Gaussian identity.

---

## 6.2 Trainable vector potential

The code stores staggered edge potentials

\[
A_x,\;A_y,\;A_z.
\]

Time is represented by a cubic B-spline with a small number of control tensors. Thus the actual trainable motion variables are B-spline control values of \(A\), not one independent velocity at every frame.

Face velocities are obtained from the discrete curl. Cell-center velocity is the average of the corresponding two face velocities.

The discrete staggered construction satisfies

\[
\boxed{\operatorname{div}_h(\operatorname{curl}_hA)\approx0}
\]

up to floating-point precision.

---

## 6.3 Time integration

For interval \(k\to k+1\), the current fixed-grid branch evaluates the cell velocity at approximately the midpoint time

\[
t_{k+1/2}=\frac12(t_k+t_{k+1})
\]

and transports

\[
\boxed{X_i^{k+1}=X_i^k+\Delta t_k\,v_{c_i^k}(t_{k+1/2}).}
\]

The use of the actual \(\Delta t_k\) is physically required: otherwise learned “velocity” and displacement become arbitrarily coupled to frame rate.

The source trajectory used inside a target-frame optimization is detached from previous intervals, so the current optimization is not full backpropagation through the entire video trajectory.

---

# 7. Per-Gaussian Lagrangian point-strain — current implementation

The public command

```text
--phase1_motion_representation lagrangian_per_frame
```

routes to `lagrangian_point_strain`.

Its purpose is to provide a clean spatial-representation ablation against the Eulerian cells.

Instead of a shared cell velocity, each Gaussian has a trainable velocity at each video frame:

\[
\boxed{v_i^k.}
\]

The transport is

\[
X_i^{k+1}=X_i^k+\Delta t_kv_i^k.
\]

The material-neighbor graph is constructed from persistent Gaussian identities and current metric distances are refreshed as the Gaussians move.

The intended physics is otherwise shared with the Eulerian formulation:

- same static/moving material probabilities;
- same observed multiview 3-D velocity targets;
- same local velocity-gradient fit;
- same \(D\to0\) non-deformation constraint;
- same DINO boundary logic;
- same RGB/depth/DINO rendering;
- same dense 2-D optical-flow endpoint loss.

This makes the comparison scientifically meaningful:

\[
\boxed{
\text{cell-shared velocity}
\quad\text{vs}\quad
\text{per-Gaussian velocity}
}
\]

without changing the underlying local continuum theory.

Unlike the CurlGrid, the point representation does **not** enforce exact incompressibility architecturally. If the estimated \(D\) were exactly zero, its trace would also be zero, but numerically this is only a learned local condition.

---

# 8. Adaptive Eulerian CurlOctree — current implementation

The adaptive branch now routes

```text
--phase1_motion_representation eulerian_octree_strain
```

to `eulerian_octree_curl_strain`.

Its objective is to retain the CurlGrid motion parameterization while adapting spatial resolution to the transported material.

## 8.1 Hierarchical vector potential

Each dyadic level \(\ell\) owns sparse staggered edge potentials

\[
A_\ell.
\]

The raw velocity sampled at a leaf is

\[
\boxed{
v_c^{raw}(t)
=
\sum_{\ell\le L_c}
\operatorname{curl}_{h_\ell}A_\ell(X_c,t)
}
\]

where \(L_c\) is the active leaf depth.

Thus refinement introduces **additional residual vector-potential capacity**, not a free child velocity.

---

## 8.2 Intended refinement principle

A split should answer:

> “Can one current cell adequately explain the motion of the material inside it?”

Observed per-Gaussian multiview 3-D velocities are used to compare one parent velocity model against separate prospective octant models using BIC.

Observed velocity is therefore intended to affect **topology**, not directly initialize motion DOFs.

When a new child level is first created, its residual potential is initialized to zero, giving

\[
\operatorname{curl}A_{fine}=0
\]

and therefore preserving the parent field at the instant of refinement.

---

## 8.3 Recycling

The v11 line added coarsening/recycling so historical refined leaves do not permanently consume the active-cell budget. Empty sibling groups can become dormant, and velocity-compatible siblings can be merged according to the inverse BIC criterion.

The v11.1 hotfix vectorized candidate discovery to eliminate repeated GPU→CPU scalar synchronization during recycling.

This is a performance correction, not a physics change.

---

# 9. Static and moving material labels

Each retained foreground Gaussian has trainable probabilities over

\[
\{\text{static},\text{moving part 1},\ldots,\text{moving part K}\}.
\]

The labels are **persistent material variables**, shared across frames.

The current corrected initialization is intentionally weak. The original gradient-motion evidence is mapped only into a bounded probability interval rather than treated as ground truth.

The labels are then optimized from visual and kinematic evidence during Phase 1.

---

## 9.1 Exact static forward behavior

The theoretical static condition is simply

\[
\boxed{v=0.}
\]

The implementation uses a straight-through hard gate.

For a cell or point whose static probability exceeds the configured threshold, the forward velocity is exactly zero.

For fixed/adaptive Eulerian cells, an additional purity requirement prevents a mixed static/dynamic cell from being hard-frozen.

The backward pass uses a smooth sigmoid surrogate so contradictory RGB/depth/flow observations can move the label away from static.

This gives the desired semantic interpretation:

> static does not mean “low velocity penalty”; static means **zero physical transport in the forward model**.

---

## 9.2 Static raw-field penalty

Because a hard gate could hide a large latent raw velocity, the code also applies a strong raw-field penalty inside hard-static locations.

Thus the intention is:

1. immediate physical behavior: \(v=0\);
2. underlying motion representation: gradually converge toward zero there as well.

---

## 9.3 Theoretical limitation of the static gate

The raw CurlGrid satisfies discrete divergence-free structure. The **gated** field does not necessarily do so across a static/moving interface because a learned binary material gate multiplies or replaces the raw field discontinuously.

Therefore the correct statement is:

\[
\boxed{
\text{raw CurlGrid is divergence-free; the hard material-gated transport field is not globally guaranteed divergence-free.}
}
\]

This distinction is currently handled correctly in the audit, but it is an important theoretical limitation of combining an incompressible Eulerian field with discontinuous learned material gating.

---

# 10. 2-D optical flow as the primary motion measurement

The optical-flow pipeline is central to the method and should be treated as the main direct motion observation.

DINO does **not** provide the displacement vector.

For camera \(c\), source pixel \(u\), and consecutive frames,

\[
f_c^{obs}(u)
\]

is the observed 2-D flow displacement in pixels.

For a 3-D point \(X\), camera projection is

\[
u=\pi_c(X).
\]

Differentiating gives

\[
\dot u
=
J_{\pi,c}(X)v,
\]

where

\[
J_{\pi,c}(X)
=
\frac{\partial \pi_c}{\partial X}.
\]

For a short interval,

\[
\dot u
\approx
\frac{f_c^{obs}}{\Delta t}.
\]

Thus each visible camera contributes

\[
\boxed{
J_{\pi,c}(X)v
\approx
\frac{f_c^{obs}}{\Delta t}.
}
\]

---

# 11. Multiview 2-D flow → 3-D velocity inversion

For a cell or Gaussian group \(g\), the code accumulates all valid synchronized-camera equations into the normal system

\[
H_g
=
\sum_m w_mJ_m^TJ_m,
\]

\[
g_g
=
\sum_m w_mJ_m^Ty_m,
\]

with

\[
y_m=\frac{f_m^{obs}}{\Delta t}.
\]

The observed 3-D velocity is obtained from

\[
\boxed{
(H_g+\lambda I)v_g^{obs}=g_g.
}
\]

The implementation requires support from multiple **distinct physical cameras**, not simply many samples from one camera.

It also evaluates:

- observation count;
- distinct-camera count;
- conditioning;
- moving-model reprojection RMS;
- static \(v=0\) RMS;
- maximum reliable per-camera static RMS.

Accepted observed targets therefore contain meaningful multiview geometric checks.

The current default acceptance parameters include:

- minimum observations: 6;
- minimum distinct cameras: 3;
- minimum conditioning ratio: \(10^{-4}\);
- maximum moving-model reprojection RMS: 0.75 px.

These are implementation hyperparameters, not physical constants.

---

# 12. Why the multiview nature matters

A moving 3-D point may have almost zero optical flow in one view when its motion is approximately along that camera's viewing direction.

Therefore a single-view rule such as

\[
\|f_c\|\approx0
\Rightarrow
\text{static}
\]

is invalid.

The correct static hypothesis must be compatible with **all informative views**.

Conversely, different cameras constrain different components of the 3-D velocity. The normal matrix

\[
H=\sum_cJ_c^TJ_c
\]

encodes the observability of the velocity direction.

This is why the multiview inversion is more valuable than thresholding optical-flow magnitude in individual images.

---

# 13. Dense rasterized 2-D flow supervision

The method also contains a second, complementary optical-flow loss that does not first invert to a 3-D target.

At the source camera, the renderer alpha-composites the transported world-space Gaussian centers into a dense homogeneous world-position map.

Those predicted target-time world positions are projected into the target camera, producing a dense predicted flow

\[
f^{pred}(u).
\]

The loss directly compares

\[
\boxed{f^{pred}(u)-f^{obs}(u)}
\]

at confident flow-grid pixels.

This is important because it preserves the original observed 2-D measurement instead of forcing every pixel through a potentially ill-conditioned 3-D inversion.

The dense loss also retains confident near-zero optical flow, which is important for static regions.

Thus the current design contains **two complementary optical-flow pathways**:

1. multiview flow → explicit 3-D velocity target;
2. dense 2-D observed flow → differentiable endpoint reprojection loss.

This is theoretically desirable.

---

# 14. Current limitation of the 2-D→3-D velocity model

The normal-system inversion uses the differential relation

\[
J_\pi v
\approx
f/\Delta t.
\]

But optical flow is a finite image displacement. The more exact relation is

\[
\boxed{
f^{obs}
\approx
\pi(X+\Delta t\,v)-\pi(X).
}
\]

For small inter-frame motion the Jacobian approximation is appropriate. For larger displacements, depth changes, or perspective changes, the linearization can be biased.

The dense raster endpoint loss already evaluates a finite projected displacement, but the explicit 3-D velocity inversion remains first-order.

A theoretically cleaner future version could solve the multiview 3-D velocity using a small nonlinear Gauss–Newton problem around the finite endpoint relation while retaining the current normal-system result as initialization.

---

# 15. Optical-flow confidence and DINO validation

The motion-correspondence cache currently computes forward and backward optical flow and evaluates forward/backward cycle consistency.

DINO maps are then sampled at the flow-warped target location. Their cosine similarity is converted into a confidence that multiplies the cycle confidence and semantic mask.

Therefore:

\[
\boxed{
\text{optical flow provides the vector; DINO only validates whether the proposed correspondence looks semantically/materially consistent.}
}
\]

This role is appropriate.

However, the correspondence code still uses a configured minimum DINO cosine (`correspondence_dino_min_cosine`, default 0.20) to remap correspondence confidence. This is separate from the DINO **part-boundary** logic, which now uses raw cosine distance. The correspondence threshold remains a heuristic measurement-confidence parameter and should eventually be calibrated or replaced by a robust data-driven confidence model.

---

# 16. DINO for material / part boundaries

For Gaussian or cell descriptors \(F_i,F_j\), the current boundary quantity is

\[
\boxed{
d_{ij}^{DINO}
=
1-\cos(F_i,F_j)
\in[0,2].
}
\]

This is the correct raw cosine distance. No sigmoid or arbitrary “DINO boundary threshold” is theoretically necessary.

When a bounded same-material affinity is required, the implementation uses the linear conversion

\[
\boxed{
a_{ij}^{DINO}
=
1-\frac12d_{ij}^{DINO}
=
\frac{1+\cos(F_i,F_j)}{2}.
}
\]

Thus:

- identical features → affinity 1;
- orthogonal features → affinity 0.5;
- opposite features → affinity 0.

---

## 16.1 Correct conceptual role of DINO

DINO should answer:

> “Do these neighboring material samples look like they could belong to the same physical part?”

It should not answer:

> “Are they moving?”

and it should not produce the motion vector.

The desired causal structure is

\[
\boxed{
\text{DINO}
\rightarrow
\text{material boundary plausibility}
}
\]

while

\[
\boxed{
\text{optical flow / rendering}
\rightarrow
\text{motion}
}
\]

and the optimized material labels combine both sources.

---

## 16.2 Current DINO assignment loss

The current label smoothness penalizes a probability discontinuity strongly when neighboring DINO descriptors are similar.

This is conceptually reasonable:

\[
F_i\approx F_j
\quad\Rightarrow\quad
\text{resist arbitrary part-label discontinuity}.
\]

A strong DINO discontinuity permits a motion/material boundary to form more easily.

Independent DINO k-means remains a diagnostic only, which is appropriate because cluster IDs have no static/dynamic semantic meaning.

---

## 16.3 Current DINO/physics mismatch

The current physical rigidity weights also multiply same-moving-part probability by DINO material affinity.

Conceptually this means that two neighboring points already assigned to the same moving part can have their mechanical coupling weakened if their DINO appearance differs.

That is not the clean theoretical separation we want.

A preferable design is:

\[
\boxed{
DINO
\rightarrow
\text{whether labels should split}
}
\]

followed by

\[
\boxed{
\text{same confident moving-part label}
\rightarrow
\text{mandatory physical spatial coherence}.
}
\]

Once two points are confidently in the same moving part, local texture or appearance variation should not allow them to violate the part's motion physics.

This is a current theory–implementation gap.

---

# 17. Current local velocity-gradient implementation

The local velocity gradient is estimated on a graph of neighboring occupied cells or neighboring Gaussians.

For node \(i\), the implementation solves

\[
\boxed{
G_i
=
\arg\min_G
\sum_{j\in\mathcal N(i)}
a_{ij}
\|v_j-v_i-G(X_j-X_i)\|^2
+
\lambda_G\|G\|^2.
}
\]

This is a differentiable weighted least-squares fit.

It is useful because the occupied material lies primarily on an object surface, not throughout a volumetric continuum. The ridge selects a minimum-norm extension in poorly observed normal directions.

From \(G_i\), the implementation computes

\[
D_i,
\quad
\Omega_i,
\quad
\omega_i=\frac12\nabla\times v_i.
\]

Only \(v\) or \(A\) is trainable in the corrected generic branches.

---

# 18. Current axis-free physics losses

The v11 implementation contains three main local terms.

## 18.1 Non-deformation

\[
\boxed{
L_D
=
\operatorname{mean}_i\|D_i\|_F^2.
}
\]

This is the most directly physically justified rigidity constraint.

---

## 18.2 Spatial gradient coherence

For neighboring same-moving-part nodes,

\[
\boxed{
L_G
=
\frac{\sum_{ij}w_{ij}\|G_i-G_j\|_F^2}
{\sum_{ij}w_{ij}}.
}
\]

This is a second-order smoothness penalty on the velocity field.

For an exact global affine field,

\[
G_i=G_j,
\]

so the loss is zero.

It therefore preserves both rigid translation and rigid rotation, but also preserves smooth affine shear; the \(D\) term is responsible for rejecting deformation.

---

## 18.3 Local affine consistency

For pair \((i,j)\),

\[
\Delta X_{ij}=X_j-X_i,
\]

\[
\Delta v_{ij}=v_j-v_i.
\]

The implementation uses

\[
\bar G_{ij}=\frac12(G_i+G_j)
\]

and penalizes

\[
\boxed{
\frac{\|\Delta v_{ij}-\bar G_{ij}\Delta X_{ij}\|^2}
{\|\Delta X_{ij}\|^2}.
}
\]

This is also axis-free and is exactly zero for a perfectly affine field.

---

# 19. Why these terms are theoretically attractive but practically incomplete

Mathematically, if a smooth velocity field over a connected 3-D region satisfies

\[
D=0
\]

everywhere, it is an infinitesimal rigid motion.

The implementation, however, does **not** operate on a dense smooth continuum. It operates on:

- sparse occupied cell centers or surface Gaussians;
- a finite noisy neighbor graph;
- locally fitted \(G_i\) tensors;
- incomplete multiview observations;
- changing Eulerian occupancy;
- learned material boundaries.

Therefore the continuous theorem does not automatically transfer numerically.

Each local \(G_i\) can be poorly conditioned, especially in surface-normal directions. Differentiating a noisy velocity field to estimate \(G\) amplifies noise. Differencing those fitted gradients again in \(L_G\) is effectively a second-order operation and can amplify noise further.

Thus the current implementation has a theoretically elegant but numerically fragile hierarchy:

\[
v
\rightarrow
G
\rightarrow
D
\]

plus

\[
G_i-G_j.
\]

What is still missing is a robust **first-order spatial coherence prior directly on the velocity field**.

---

# 20. Missing direct spatial velocity coherence

The debug strongly indicates that neighboring observed velocities are much more coherent than the optimized velocities, especially at later frames and in weakly observed regions.

The missing generic prior should act directly on

\[
v_i-v_j
\]

for neighboring points/cells in the same moving part.

A generic form is

\[
\boxed{
L_{v,sp}
=
\frac{\sum_{ij}w_{ij}\,\rho\!\left(\|v_i-v_j\|/s_{ij}\right)}
{\sum_{ij}w_{ij}}
}
\]

where:

- \(\rho\) is robust, e.g. Charbonnier/Huber;
- \(w_{ij}\) expresses same-moving-part support and spatial adjacency;
- \(s_{ij}\) should be derived from physical/observed velocity scale rather than hardcoded.

This must be a **smoothness prior**, not a hard equality constraint, because under rotation neighboring points legitimately have different velocities.

The purpose is to stop the field from producing independent arbitrary arrows where the data indicates one coherent moving material region.

This term is currently missing.

---

# 21. Why direct velocity smoothness should be weaker than rigidity

A rigid rotation satisfies

\[
v_j-v_i
=
\omega\times(X_j-X_i),
\]

so \(v_j\ne v_i\) in general.

Therefore a very strong loss

\[
\|v_i-v_j\|^2
\rightarrow0
\]

would bias the model toward translation and suppress rotation.

The direct velocity term should instead be viewed as a stabilizing graph prior that propagates motion through weakly observed regions.

The actual physical discrimination between rigid rotation and shear remains

\[
D\rightarrow0.
\]

A sensible theoretical hierarchy is therefore:

1. direct observed velocity — strongest data evidence where available;
2. robust local velocity smoothness — propagate evidence spatially;
3. \(D\to0\) — reject deformation;
4. higher-order \(G\)-coherence / affine consistency — weaker integrability regularization.

The current implementation effectively places steps 3–4 before step 2.

---

# 22. Loss-scale problem

The current default coefficients are approximately:

```text
lambda_eulerian_strain_rate                = 100
lambda_eulerian_spatial_gradient_coherence = 50
lambda_eulerian_local_affine_coherence     = 50
lambda_observed_flow_velocity              = 20
lambda_observed_flow_direction             = 10
```

These coefficients multiply quantities with different physical units and different natural numerical scales.

For example:

- velocity residual has units of velocity;
- \(G\) and \(D\) have units of inverse time;
- \(G_i-G_j\) is also inverse time;
- the local affine residual divided by edge length again behaves like inverse time;
- 2-D flow residual is normalized by image dimension;
- RGB/SSIM/DINO losses are dimensionless but have unrelated scales.

Using a single fixed `eulerian_velocity_scale=1.0` does not make these terms commensurate.

This is a major theory–implementation gap.

---

# 23. Required dimensionless normalization

A more principled implementation should compute robust scales from the current observed motion.

For each frame/window, define for example

\[
s_v
=
\operatorname{median}_{valid}\|v^{obs}\|,
\]

and a material spatial scale

\[
s_x
=
\operatorname{median}_{(i,j)}\|X_j-X_i\|.
\]

Then the natural gradient scale is

\[
\boxed{s_G=s_v/s_x.}
\]

Losses can then be normalized as

\[
\frac{\|v-v^{obs}\|}{s_v},
\]

\[
\frac{\|D\|}{s_G},
\]

\[
\frac{\|G_i-G_j\|}{s_G},
\]

and so forth.

This would make the relative weights interpretable instead of strongly dependent on the object's metric scale, frame rate, grid resolution, or particular optical-flow magnitude.

The code does not yet do this.

---

# 24. Gradient-balance diagnostics are missing

Even if scalar loss values are logged, what matters to the optimizer is the gradient each term sends to the motion parameters.

A required diagnostic should report, for each major term,

\[
\left\|\frac{\partial L_j}{\partial \theta_{motion}}\right\|,
\]

for:

- observed 3-D velocity magnitude;
- observed direction;
- dense 2-D flow;
- RGB/depth/DINO rendering;
- strain;
- direct velocity spatial smoothness;
- gradient coherence;
- affine consistency;
- hard-static raw-field loss.

Without this, a visually small physics loss can still dominate because its gradient is large, or vice versa.

The current implementation does not provide a complete per-term gradient-balance audit.

---

# 25. Debug evidence: multiview observations are often better than the learned field

The available completed per-Gaussian point run provides direct evidence.

At frame 30, approximately 64.2% of Gaussians have accepted multiview 3-D velocity observations. At frame 59 this is approximately 43.7%.

For final-dynamic Gaussians with valid observations:

### Frame 30

Observed median speed:

\[
\approx0.0998
\]

Learned median speed:

\[
\approx0.00134.
\]

The learned/observed median speed ratio is only

\[
\approx0.0128.
\]

Yet the median direction cosine is

\[
\approx0.995.
\]

This is a very important signature: the optimization has roughly found the observed direction, but has heavily suppressed the magnitude.

### Frame 59

Observed median speed:

\[
\approx0.0742.
\]

Learned median speed:

\[
\approx0.00957.
\]

The field is still much too slow and the directional distribution has degraded.

---

# 26. Debug evidence: neighboring observed velocities are highly coherent

For neighboring final-dynamic Gaussian pairs with valid multiview observations:

### Frame 30

Observed neighbor-angle median:

\[
\boxed{1.44^\circ}
\]

Observed p90:

\[
\boxed{5.63^\circ}.
\]

The learned field on the same observed subset is still relatively coherent, but already worse.

### Frame 59

Observed neighbor-angle median:

\[
\boxed{2.12^\circ}
\]

Observed p90:

\[
\boxed{8.45^\circ}.
\]

The learned field on that same subset reaches approximately

\[
21.7^\circ
\]

median and

\[
95.1^\circ
\]

p90.

Across all dynamic neighbor pairs, including weakly/unobserved points, the frame-59 learned median neighbor angle is approximately

\[
53^\circ
\]

with p90 around

\[
134^\circ.
\]

This demonstrates that the main failure is not that optical flow itself is random. The current field fails to preserve/propagate the observed spatial coherence.

---

# 27. Debug evidence: fixed CurlGrid has the same late-time pathology

For neighboring dynamic fixed-grid cells, the learned velocity direction becomes progressively incoherent:

| target frame | median neighbor angle | p90 | fraction > 60° |
|---:|---:|---:|---:|
| 1 | 24.9° | 58.3° | 9.8% |
| 30 | 29.7° | 85.7° | 18.4% |
| 40 | 31.8° | 91.7° | 20.9% |
| 50 | 46.9° | 120.5° | 38.9% |
| 59 | **62.5°** | **138.1°** | **51.8%** |

At frame 30, on dynamic neighboring cells for which both multiview observed 3-D velocities are valid:

- observed median neighbor angle ≈ **4.67°**;
- observed p90 ≈ **15.97°**;
- learned median ≈ **22.18°**;
- learned p90 ≈ **49.20°**.

Again, the observation field is significantly more coherent than the optimized field.

This strongly supports adding direct spatial velocity coherence and correcting the loss balance.

---

# 28. What \(D=0\) guarantees in theory but not in the current discrete implementation

In the continuous theorem, \(D=0\) is evaluated everywhere in a connected domain.

In the current implementation:

- \(G_i\) is fitted independently at sparse nodes;
- many nodes can be invalid because fewer than the minimum supported neighbors exist;
- occupied material is surface-like, so local geometry is rank-deficient in the normal direction;
- same-part/DINO weighting can remove edges;
- Eulerian occupancy changes over time;
- hard static gating introduces discontinuities;
- the field is only sampled at cell centers or Gaussian centers.

Therefore small average \(\|D_i\|\) does not guarantee a globally integrated rigid field in practice.

This is the central distinction between **continuum rigidity theory** and **discrete learned rigidity**.

---

# 29. Temporal optimization gap

The physical trajectory is sequential, but current optimization detaches committed source positions before optimizing the next interval.

Therefore a late-frame error cannot backpropagate through all preceding motion decisions.

The current behavior is closer to

\[
\text{optimize interval }k
\rightarrow
\text{commit}
\rightarrow
\text{detach}
\rightarrow
\text{optimize interval }k+1
\]

than to a global trajectory optimization.

This is computationally practical, but drift can accumulate.

The full theory would minimize a trajectory-wide objective

\[
L(X_0,\ldots,X_T,v_0,\ldots,v_{T-1})
\]

subject to all integration constraints jointly.

A compromise could use short overlapping windows with truncated backpropagation through several consecutive intervals, rather than a fully detached one-step source.

---

# 30. Adaptive CurlOctree: theory versus old debug

The old direct-velocity adaptive-octree run saturated its active-cell budget extremely early.

With a maximum around 12,288 active cells:

| frame | active cells | split parents | BIC candidates |
|---:|---:|---:|---:|
| 1 | 4,425 | 173 | 936 |
| 5 | 7,141 | 97 | 574 |
| 10 | 10,508 | 96 | 281 |
| 12 | 11,852 | 96 | 148 |
| 13 | 12,286 | 62 | 107 |
| 14 | 12,286 | 0 | 64 |
| 30 | 12,286 | 0 | 116 |
| 55 | 12,286 | 0 | 136 |
| 59 | 12,286 | 0 | 143 |

Thus the topology still wanted to split, but the global active-cell budget prevented it.

The v11 recycler was introduced specifically to address this failure mode.

---

# 31. Adaptive CurlOctree: current stale-reactivation gap

The intended refinement invariant is:

\[
\boxed{
\text{refinement/re-activation must not inject motion.}
}
\]

The first creation of a fine residual satisfies this because the new fine-level \(A\) is initialized to zero.

However, the current v11.1 causal knot copy loops over **all allocated curl levels**, including dormant fine levels, and copies their previous temporal state forward.

If a region was previously refined, learned a nonzero fine residual, then was coarsened, that residual can remain dormant but continue being copied temporally.

If the old children are later reactivated, the current topology code reuses those child cells. Their old fine residual can therefore become active again.

The resulting field can be

\[
v_{child}
=
v_{parent}+v_{fine,stale}
\]

instead of the intended inherited state

\[
\boxed{v_{child}=v_{parent}}
\]

at the instant of reactivation.

This is a real theory–implementation mismatch and should be corrected before interpreting late-frame adaptive results.

The correct behavior is to reset or rebase the currently reactivated fine residual so the field is continuous with the active parent solution at the reactivation time.

---

# 32. Adaptive CurlOctree and coarse–fine conservation

The implementation verifies \(\operatorname{div}_h(\operatorname{curl}_hA)=0\) independently on each dyadic level.

This is a useful algebraic property.

However, a fully conservative adaptive finite-volume method normally also requires coarse–fine flux consistency at hanging interfaces.

The current motion transport samples the hierarchical field at active leaf centers rather than explicitly constructing one globally reconciled coarse/fine face-flux field.

Therefore the strongest mathematically safe statement is:

> each sparse uniform level uses a curl construction whose native discrete divergence is zero; the hierarchy is a sum of such fields at sampled leaf centers.

A future exact adaptive finite-volume formulation could explicitly enforce coarse/fine face-flux matching if strict global conservation at adaptive interfaces becomes important.

---

# 33. Cell mixing remains an Eulerian representation limitation

If a fixed Eulerian cell contains both static and dynamic Gaussians, one cell velocity cannot satisfy both simultaneously.

Likewise, if a cell overlaps two different moving parts, one velocity cannot represent a true motion discontinuity.

The current purity gate prevents a mixed cell from being hard-frozen, but that does not solve the representational conflict; it only avoids one particularly damaging decision.

Adaptive refinement is intended to reduce this problem by introducing smaller cells where the current material/motion distribution requires additional spatial capacity.

The per-Gaussian Lagrangian branch provides the control experiment where this spatial sharing is removed completely.

---

# 34. Current training objective — conceptual form

For the generic corrected branches, the motion loss can be summarized as

\[
L
=
L_{visual}
+
L_{2Dflow}
+
L_{3Dflow}
+
L_{static}
+
L_D
+
L_G
+
L_{affine}
+
L_{temporal}
+
L_{assignment}.
\]

Where:

### Visual

\[
L_{visual}
=
L_{RGB/SSIM}
+
L_{depth}
+
L_{DINO-render}.
\]

### Dense flow

\[
L_{2Dflow}
\sim
\sum_u
w(u)\rho( f^{pred}(u)-f^{obs}(u)).
\]

### Explicit multiview velocity

\[
L_{3Dflow}
=
\lambda_vL_{mag}
+
\lambda_{dir}L_{dir}.
\]

### Generic continuum rigidity

\[
L_D
=
\|\operatorname{sym}(\nabla v)\|^2.
\]

### Current higher-order spatial terms

\[
L_G
\sim
\|G_i-G_j\|^2,
\]

\[
L_{affine}
\sim
\|\Delta v-\bar G\Delta X\|^2.
\]

### Missing direct velocity coherence

\[
\boxed{L_{v,sp}\quad\text{not yet present}.}
\]

---

# 35. What the theoretical objective should become

A more robust ordering is

\[
\boxed{
L_{motion}
=
\lambda_{obs}L_{obs}
+
\lambda_{vsp}L_{v,sp}
+
\lambda_D L_D
+
\lambda_G L_G
+
\lambda_A L_{affine}
+
\lambda_TL_T
}
\]

with every term dimensionlessly normalized using data-derived motion and spatial scales.

The hierarchy of influence should be:

1. **observation fidelity** where multiview evidence is strong;
2. **direct velocity spatial propagation** into weakly observed same-part neighbors;
3. **non-deformation** to remove shear/extension;
4. **higher-order integrability** as a weaker stabilizer;
5. **temporal coherence** to suppress framewise oscillation.

This is closer to the information hierarchy actually visible in the debug.

---

# 36. Static/dynamic and part-boundary theory

The material labeling problem should combine three logically different sources:

### Motion evidence

Does the observed image motion require nonzero 3-D motion?

Provided mainly by optical flow and rendering.

### Static evidence

Can zero motion explain the observations?

The current implementation handles this contextually through near-zero observed motion, rendering losses, and the hard static model rather than a separate precomputed static classifier.

### Boundary evidence

If neighboring material seems to require different motion states, is a physical/material boundary plausible?

Provided mainly by DINO descriptor discontinuity.

Thus the intended factorization is

\[
\boxed{
\text{flow/rendering}\rightarrow\text{motion state}
}
\]

\[
\boxed{
\text{DINO}\rightarrow\text{boundary plausibility}
}
\]

\[
\boxed{
\text{same optimized moving part}\rightarrow\text{motion coherence}
}
\]

The last arrow is the one that still needs to become stronger and less appearance-dependent.

---

# 37. What is already correct and should be preserved

Several parts of the current design should **not** be discarded.

1. **Real video \(\Delta t\)** is used in motion supervision and integration.
2. **Optical flow supplies the displacement vector.** DINO is not used as a pseudo-motion vector.
3. **Multiview 3-D inversion requires distinct cameras.**
4. **Dense 2-D flow supervision remains in image space** and complements the 3-D inverse solve.
5. **DINO part-boundary distance is raw cosine distance** rather than a hand-designed sigmoid boundary threshold.
6. **Static forward motion is exactly zero** when confidently static.
7. **Curl is not minimized.** Rotation is allowed.
8. **The first stage does not optimize a joint axis, pivot, \(b\), or rigid \(\omega\).**
9. **The Eulerian raw field is represented through a vector potential** rather than free cell velocities.
10. **The Lagrangian point branch exists as a true representation ablation.**
11. **The adaptive branch adds residual curl capacity rather than free child velocities.**

These are the correct conceptual foundations.

---

# 38. What is missing — priority order

## Priority 1 — direct same-part velocity spatial coherence

Add a robust first-order graph loss on neighboring velocities.

It must:

- operate on current positions;
- use same-moving-part support;
- not depend on DINO once the same part is confidently established;
- be normalized by observed velocity/spatial scale;
- remain weak enough not to suppress valid rigid rotation.

---

## Priority 2 — physically meaningful loss normalization

Replace arbitrary unit-dependent loss balance with per-frame/window robust scales:

\[
s_v,
\quad
s_x,
\quad
s_G=s_v/s_x.
\]

Log both scalar losses and gradient norms.

Without this correction, any new spatial term can simply push the solution further toward zero motion.

---

## Priority 3 — separate DINO boundary inference from physical coupling

DINO should regularize labels / boundary likelihood.

Once two nodes are confidently in the same moving part, their physical coherence should not be weakened simply because their DINO descriptors differ.

---

## Priority 4 — fix adaptive reactivation inheritance

On reactivating a previously dormant branch, ensure the newly active fine residual contributes zero at the current reactivation state unless the optimizer subsequently changes it.

Do not resurrect historical fine motion automatically.

---

## Priority 5 — sufficient local convergence before committing a frame

A sequential adaptive run with only four local updates per physical interval is too weak to simultaneously optimize:

- motion;
- rendering;
- optical flow;
- labels;
- strain;
- spatial coherence;
- topology.

The correct criterion should be based on convergence/stability or a substantially larger local iteration budget, not merely a very small fixed count used for a smoke test.

---

## Priority 6 — uncertainty-aware multiview propagation

The 3-D velocity normal matrix naturally contains anisotropic uncertainty.

Rather than compressing this to a single confidence scalar, future spatial propagation could use

\[
\Sigma_v\propto(H+\lambda I)^{-1}
\]

so motion is propagated strongly along well-constrained directions and cautiously along poorly observed ones.

This would make better use of the multiview geometry.

---

## Priority 7 — gauge handling for \(A\)

Introduce a consistent vector-potential gauge or ensure all regularization is expressed in gauge-invariant quantities such as \(v\), \(D\), and curl-derived fields.

Temporal smoothing directly on \(A\) is otherwise partly gauge-dependent.

---

## Priority 8 — improve temporal optimization / truncated trajectory gradients

Use overlapping windows with gradients through several consecutive intervals so future-frame evidence can correct preceding motion without the cost of full-video BPTT.

---

# 39. Recommended target Phase-1 architecture

The intended data flow should be:

```text
STATIC 3DGS + synchronized multiview video
                  │
                  ├──────── RGB / depth / DINO images
                  │
                  └──────── observed 2-D optical flow
                                      │
                       cycle + visibility + DINO confidence
                                      │
                     ┌────────────────┴─────────────────┐
                     │                                  │
                     ▼                                  ▼
          dense 2-D endpoint loss            multiview 3-D velocity solve
                                                        │
                                                        ▼
                                              observed v_obs + uncertainty
                                                        │
                                                        ▼
                                     trainable static / moving-part labels
                                                        │
                         DINO cosine-distance boundaries│
                                      ──────────────────┤
                                                        ▼
                                  SAME-PART MOTION GRAPH
                                                        │
                      ┌─────────────────────────────────┼───────────────────────┐
                      ▼                                 ▼                       ▼
            direct robust v smoothness        D = sym(grad v) -> 0     weak G/affine coherence
                      │                                 │                       │
                      └─────────────────────────────────┴───────────────────────┘
                                                        │
                                     representation-specific velocity
                           ┌────────────────────────────┼──────────────────────────┐
                           ▼                            ▼                          ▼
                    fixed CurlGrid             Lagrangian points          adaptive CurlOctree
                           │                            │                          │
                           └────────────────────────────┴──────────────────────────┘
                                                        │
                                                        ▼
                                          integrate with real Δt
                                                        │
                                                        ▼
                                          rasterize all cameras
                                                        │
                                                        └── repeat / refine
```

The central principle is that **the visual/multiview measurement establishes the motion**, while physics and spatial coherence make that motion interpretable and propagate it through weakly observed regions.

Physics must not overpower the observation to the point where the trivial zero field becomes preferable.

---

# 40. Clean comparison protocol: cell vs Gaussian vs adaptive cells

To answer the research question scientifically, the three representations should share:

- identical foreground Gaussian set;
- identical timestamps and \(\Delta t\);
- identical RGB/depth/DINO targets;
- identical optical-flow correspondences;
- identical multiview 3-D velocity inversion;
- identical static/moving label optimizer;
- identical DINO boundary loss;
- identical direct velocity spatial coherence rule;
- identical \(D\to0\) continuum loss;
- identical robust normalization scales;
- identical temporal data coverage;
- identical number of effective visual observations.

Only the motion representation should differ:

| Property | Fixed CurlGrid | Lagrangian point | Adaptive CurlOctree |
|---|---|---|---|
| Motion coordinate | world-space cells | material Gaussian identity | adaptive world-space leaves |
| Trainable motion | B-spline edge potential \(A\) | \(v_i^k\) | hierarchical edge potentials \(A_\ell^k\) |
| Raw incompressibility | exact discrete curl | not architectural | per-level curl construction |
| Spatial sharing | one velocity per cell | none before graph regularization | one velocity per active leaf |
| Boundary resolution | fixed | Gaussian-level | adaptive |
| Main hypothesis tested | effect of fixed Eulerian discretization | behavior without cell sharing | whether adaptive cells solve mixing while retaining curl |

This is the comparison that can meaningfully tell us whether the problem comes from the Eulerian coordinate system, cell sharing, adaptive topology, or optimization itself.

---

# 41. Required debug outputs before trusting a new run

Every selected target frame should report:

## Observed motion

- valid multiview 3-D target fraction;
- distinct-camera count distribution;
- conditioning distribution;
- 3-D observed speed median/p90;
- observed neighbor velocity angle median/p90;
- observed neighbor speed-difference distribution.

## Learned motion

- learned speed median/p90;
- learned-vs-observed cosine;
- learned/observed speed ratio;
- same-part neighbor velocity angle median/p90;
- direct spatial velocity-coherence residual.

## Continuum physics

- \(\|D\|_F\) median/p90;
- \(\|G_i-G_j\|\) median/p90;
- affine consistency residual;
- valid-gradient fraction;
- neighbor-count / local conditioning statistics.

## Loss balance

- scalar value of every loss;
- weighted scalar contribution;
- gradient norm sent by every loss to the motion parameters.

## Material partition

- static/dynamic probability histogram;
- number of hard-static points/cells;
- static observed speed distribution;
- dynamic observed speed distribution;
- DINO boundary strength on learned part boundaries vs interiors.

## Eulerian grid / octree

- occupied cells;
- mixed static/dynamic cells;
- active levels;
- split / merge / reactivation events;
- reactivated-child residual norm before optimization;
- raw `div(curl A)` audit;
- points outside the domain.

## 2-D cameras

For every camera used in the selected frame:

- observed optical-flow arrows;
- predicted rasterized optical flow;
- residual magnitude/direction;
- DINO correspondence confidence;
- current material boundary;
- DINO cosine-distance boundary.

This is necessary to determine whether a failure originates in observation, 3-D inversion, spatial propagation, material labeling, or the motion representation.

---

# 42. Final theoretical picture

The desired first-stage model can be summarized by four equations.

## Observation

\[
\boxed{
J_{\pi,c}(X)v
\approx
\frac{f_c^{obs}}{\Delta t}
}
\]

across synchronized cameras, complemented by dense nonlinear endpoint reprojection.

## Eulerian continuity

\[
\boxed{v=\nabla\times A}
\]

for the CurlGrid representations.

## Local rigidity

\[
\boxed{
D
=
\frac12(\nabla v+\nabla v^T)
\rightarrow0.
}
\]

## Spatial material coherence

\[
\boxed{
\text{same moving material}
\Rightarrow
\text{robustly coherent neighboring velocity field}
}
\]

with DINO used only to decide where that **same-material assumption may break**.

The first stage should not know the joint axis. It should produce a field from which the axis becomes obvious later.

The current implementation already contains most of the required observation and continuum machinery. The major missing bridge is the robust, correctly normalized propagation from **good multiview local velocity observations** to a **globally coherent same-part velocity field**.

That bridge must be fixed before adding more sophisticated joint models.

---

# 43. Implementation/source map

This report is based on the exact v11.1 package and the available completed debug tensors.

Key source files/functions:

- `scene/fixed_curl_grid.py`
  - `FixedCurlGrid`
  - staggered vector potential and discrete curl
  - cubic B-spline temporal potential
- `scene/lagrangian_frame_velocity.py`
  - per-Gaussian per-frame velocity table
- `scene/adaptive_curl_octree_grid.py`
  - hierarchical sparse curl levels
  - split/reactivate/recycle logic
- `utils/dynamic_loss_utils.py`
  - `fit_local_velocity_gradient`
  - `decompose_velocity_gradient`
  - `eulerian_velocity_gradient_rigidity_loss`
  - `assignment_dino_boundary_coherence_loss`
  - `dino_cosine_distance_pair_values`
- `utils/motion_correspondence.py`
  - forward/backward optical flow
  - DINO-warp correspondence confidence
  - `observed_velocity_targets`
  - `trajectory_correspondence_loss`
- `utils/unified_dynamic_training.py`
  - fixed CurlGrid Phase 1
  - Lagrangian point-strain Phase 1
  - adaptive CurlOctree Phase 1
  - straight-through static gates
  - frame/window scheduling

Available completed debug evidence used in the diagnosis:

- `phase1_eulerian_velocity_gradient_audit(1).pt`
- `phase1_observed_3d_flow_cells_000030(1).pt`
- `phase1_lagrangian_point_velocity_gradient_audit.pt`
- `phase1_observed_3d_flow_points_000030.pt`
- `phase1_observed_3d_flow_points_000059.pt`
- `phase1_lagrangian_point_velocity_final.pt`
- `phase1_adaptive_octree_strain_audit.pt`
- `phase1_adaptive_octree_velocity_final.pt`

**Important provenance note:** the completed fixed/point/octree debug tensors above were produced before the final v11.1 recycler hotfix. The v11.1 run shown most recently was manually interrupted before completion, so no claim is made here that v11.1 has already resolved the numerical motion-quality problems. The report distinguishes current source-code intent from behavior demonstrated by the most recent completed runs.
