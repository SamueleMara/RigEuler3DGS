# CurlField: Continuous Divergence-Free Eulerian Motion for Dynamic 3D Gaussian Centers

> **Historical design exploration.** The implemented repository now uses the
> finite-volume fixed CurlGrid documented in [`PIPELINE.md`](PIPELINE.md), not
> the alternative neural/Lagrangian comparison proposed below.

## 1. Purpose of this document

This document defines a simplified research direction for Phase 1 dynamic motion estimation in RigEuler3DGS.

The objective is to deliberately remove the mechanisms that have made the previous Eulerian formulation difficult to interpret:

- no cell-wise trainable velocity parameters;
- no adaptive octree velocity field;
- no split/merge-driven motion optimization;
- no direct injection of optical-flow-derived 3D velocities into the physical field;
- no learned static/dynamic membership feedback loop during the first motion-estimation experiment;
- no use of Gaussian covariance, scale, or opacity to define material support in the first implementation.

For the first implementation, every foreground Gaussian is treated only as a **3D point located at its center**.

The two motion representations to compare are:

1. **Lagrangian baseline**  
   A shared MLP predicts a time-varying velocity for each Gaussian/material point from its identity/canonical location and time.

2. **Continuous Eulerian CurlField**  
   A shared neural field predicts a vector potential
   \[
   \mathbf A_\theta(\mathbf x,t)\in\mathbb R^3,
   \]
   and the physical velocity is defined by
   \[
   \boxed{
   \mathbf v_\theta(\mathbf x,t)
   =
   \nabla_{\mathbf x}\times
   \mathbf A_\theta(\mathbf x,t)
   }.
   \]

The CurlField formulation is attractive because it enforces incompressibility exactly by construction while keeping the velocity field continuous and independent of an explicit spatial velocity grid.

The spatial grid is retained only as a **material-support and indexing device**, constructed from the current transported point centers.

---

# 2. Why simplify the representation now

The previous adaptive-cell formulation mixed several conceptually different tasks:

- representing the velocity field;
- deciding where the scene is occupied;
- allocating additional spatial degrees of freedom;
- deciding which Gaussians are static or dynamic;
- detecting whether a cell contains incompatible motions;
- initializing newly created velocity parameters;
- determining candidate rigid parts.

This made debugging extremely difficult because an error in one mechanism changed the behavior of all the others.

The new decomposition is:

\[
\boxed{
\text{motion representation}
\neq
\text{material support}
\neq
\text{rigidity regularization}
\neq
\text{joint-boundary inference}
}
\]

Each component has a single role.

### Motion representation

The continuous CurlField answers:

> Given a world-space location \(\mathbf x\) and physical time \(t\), what is the instantaneous Eulerian velocity?

### Material support

A simple voxelization of transported Gaussian centers answers:

> At the current time, which regions of space contain observed object points?

### Physics

The curl construction and local strain regularization answer:

> Is the learned motion compatible with constant-density undeformable material?

### Joint inference

Spatial regions where the field cannot remain locally rigid while satisfying the visual data become candidates for articulation boundaries.

This separation should make every failure mode directly measurable.

---

# 3. Point-cloud abstraction: the first implementation must use Gaussian centers only

For the first CurlField experiment, Gaussian \(i\) is represented only by

\[
\mathbf x_i(t)\in\mathbb R^3.
\]

We intentionally ignore:

- covariance \(\Sigma_i\);
- Gaussian scale;
- Gaussian rotation;
- opacity;
- 2-sigma or 3-sigma ellipsoids;
- ellipsoid/voxel intersection;
- continuous Gaussian occupancy;
- Gaussian volume;
- physical mass assigned from Gaussian scale.

This is deliberate.

The purpose of the first experiment is to determine whether a continuous physically constrained Eulerian field can recover the motion of the reconstructed object points.

Therefore the working abstraction is simply

\[
\boxed{
\{\text{foreground Gaussians}\}
\longrightarrow
\{\text{3D material tracer points}\}.
}
\]

The renderer can still use the full Gaussian attributes for RGB, depth, and feature rendering.  
However, all motion-field reasoning described in this document initially uses only the Gaussian centers.

This distinction is important:

- **rendering representation:** full 3D Gaussians;
- **motion/physics representation:** point centers only.

Only after the center-based formulation is understood should finite Gaussian support be considered.

---

# 4. Lagrangian baseline

Before judging any Eulerian representation, we need a representation with no spatial-discretization problem.

Let Gaussian \(i\) have canonical center

\[
\mathbf X_i=\mathbf x_i(t_0).
\]

A shared MLP predicts

\[
\mathbf v_i(t)
=
f_\phi(\mathbf X_i,\mathbf e_i,t),
\]

where \(\mathbf e_i\) can be a small learnable identity embedding.

This is Lagrangian because the velocity query follows the material identity.

For physical interval

\[
[t_k,t_{k+1}],
\qquad
\Delta t_k=t_{k+1}-t_k,
\]

the point is transported by

\[
\mathbf x_i(t_{k+1})
=
\mathbf x_i(t_k)
+
\int_{t_k}^{t_{k+1}}
\mathbf v_i(t)\,dt.
\]

The key contract is:

\[
\boxed{
\text{optimizer iterations are not physical integration steps}.
}
\]

For every frame interval, all local optimization iterations restart from the same committed previous-frame positions \(\mathbf x_i(t_k)\).  
Only after the final local optimization update do we integrate once with the final field and commit

\[
\mathbf x_i(t_{k+1}).
\]

The Lagrangian baseline answers a crucial question:

> Can the existing RGB, depth, DINO/feature, flow/rasterization, and rigidity supervision recover sensible motion if there is no Eulerian discretization at all?

If the Lagrangian baseline fails, the problem is not the spatial grid.

If it succeeds while CurlField fails, the Eulerian parameterization or its physical constraints are responsible.

---

# 5. Eulerian CurlField

## 5.1 Eulerian definition

In an Eulerian representation, velocity belongs to a point in world space, not to a material identity.

The field is

\[
\mathbf v(\mathbf x,t).
\]

A material point follows

\[
\boxed{
\frac{d\mathbf x_i}{dt}
=
\mathbf v(\mathbf x_i(t),t)
}.
\]

Thus the same Gaussian can query different spatial portions of the field as it moves.

This is the fundamental property that must be preserved.

---

## 5.2 Intermediate vector potential

Instead of predicting velocity directly, use a neural vector potential

\[
\boxed{
\mathbf A_\theta:
(\mathbf x,t)\mapsto
\mathbb R^3
}.
\]

The physical velocity is

\[
\boxed{
\mathbf v_\theta(\mathbf x,t)
=
\nabla_{\mathbf x}
\times
\mathbf A_\theta(\mathbf x,t)
}.
\]

Writing

\[
\mathbf A=(A_x,A_y,A_z),
\]

the velocity is

\[
v_x
=
\frac{\partial A_z}{\partial y}
-
\frac{\partial A_y}{\partial z},
\]

\[
v_y
=
\frac{\partial A_x}{\partial z}
-
\frac{\partial A_z}{\partial x},
\]

\[
v_z
=
\frac{\partial A_y}{\partial x}
-
\frac{\partial A_x}{\partial y}.
\]

The network does **not** directly optimize one velocity per Gaussian or one velocity per cell.

It learns one continuous potential over space-time.

---

# 6. Constant density and continuity

The general continuity equation is

\[
\frac{\partial \rho}{\partial t}
+
\nabla\cdot(\rho\mathbf v)
=
0.
\]

Expanding,

\[
\frac{\partial \rho}{\partial t}
+
\mathbf v\cdot\nabla\rho
+
\rho\nabla\cdot\mathbf v
=
0.
\]

For a homogeneous undeformable material, inside the material domain,

\[
\rho=\rho_0=\text{constant}.
\]

Therefore

\[
\frac{\partial\rho}{\partial t}=0,
\]

and

\[
\nabla\rho=0.
\]

The continuity equation becomes

\[
\rho_0\nabla\cdot\mathbf v=0.
\]

Since \(\rho_0>0\),

\[
\boxed{
\nabla\cdot\mathbf v=0.
}
\]

Therefore density completely disappears from the interior equation.

This is a major simplification.

We do not need:

- a learned density field;
- Gaussian masses;
- mass per voxel;
- density transport;
- finite-volume mass flux during the first implementation.

---

# 7. Why the curl representation is especially useful

For every sufficiently smooth vector field \(\mathbf A\),

\[
\nabla\cdot
(\nabla\times\mathbf A)
=
0.
\]

Therefore, if

\[
\mathbf v=\nabla\times\mathbf A_\theta,
\]

then

\[
\boxed{
\nabla\cdot\mathbf v=0
}
\]

is satisfied by construction.

There is no need to balance a continuity loss against RGB or depth loss.

This avoids the common failure mode

\[
\lambda_{\rm div}
\text{ too small}
\Rightarrow
\text{physics ignored},
\]

or

\[
\lambda_{\rm div}
\text{ too large}
\Rightarrow
\text{visual reconstruction blocked}.
\]

The architecture itself enforces the conservation law.

---

# 8. Gauge freedom of the vector potential

The vector potential is not unique.

For any scalar field

\[
\psi(\mathbf x,t),
\]

the transformation

\[
\mathbf A'
=
\mathbf A+\nabla\psi
\]

does not alter the velocity because

\[
\nabla\times\nabla\psi=0.
\]

Therefore many \(\mathbf A\) fields correspond to the same physical \(\mathbf v\).

This is called gauge freedom.

For our application, this is not intrinsically a problem because only \(\mathbf v\) enters:

- trajectory integration;
- rendering;
- rigidity losses;
- joint inference.

However, gauge freedom can make the latent potential unnecessarily large or poorly conditioned.

Possible later regularizers include

\[
\mathcal L_A
=
\|\mathbf A\|^2,
\]

or a Coulomb-gauge penalty

\[
\mathcal L_{\rm gauge}
=
(\nabla\cdot\mathbf A)^2.
\]

These should **not** be added to the first experiment unless optimization shows a real problem.

The first implementation should remain minimal.

---

# 9. Undeformability is stronger than incompressibility

A divergence-free field preserves local volume but can still shear.

Define the velocity Jacobian

\[
J_v
=
\nabla\mathbf v.
\]

The rate-of-deformation tensor is

\[
\boxed{
D(\mathbf x,t)
=
\frac{1}{2}
\left(
J_v+J_v^T
\right)
}.
\]

For a locally rigid motion,

\[
\boxed{
D=0.
}
\]

The trace is

\[
\operatorname{tr}(D)
=
\nabla\cdot\mathbf v.
\]

Therefore

\[
D=0
\Rightarrow
\nabla\cdot\mathbf v=0.
\]

The curl construction enforces the zero-trace part exactly.  
The remaining strain loss suppresses shear and anisotropic extension.

A natural local physics loss is

\[
\boxed{
\mathcal L_{\rm strain}
=
\frac{1}{N}
\sum_i
\left\|
D(\mathbf x_i(t),t)
\right\|_F^2.
}
\]

Because \(\mathbf v\) is already divergence-free, this primarily penalizes non-rigid shear/stretch components.

---

# 10. Rigid-body velocity is compatible with CurlField

For a rigid body,

\[
\mathbf v(\mathbf x,t)
=
\mathbf V(t)
+
\boldsymbol\omega(t)
\times
(\mathbf x-\mathbf c(t)).
\]

Its spatial Jacobian is skew-symmetric:

\[
\nabla\mathbf v
=
[\boldsymbol\omega]_\times.
\]

Thus

\[
D
=
\frac12
([\omega]_\times+[\omega]_\times^T)
=
0.
\]

Also,

\[
\nabla\cdot\mathbf v=0.
\]

So the desired rigid-link motion lies exactly inside the admissible class of the CurlField.

The physical prior does not force zero motion.  
It permits translation and rotation while rejecting deformation.

---

# 11. The discontinuity problem

An articulated object is not globally one rigid body.

Different links may have different instantaneous rigid motions.

For link \(a\),

\[
\mathbf v_a
=
\mathbf V_a+
\boldsymbol\omega_a\times(\mathbf x-\mathbf c_a).
\]

For link \(b\),

\[
\mathbf v_b
=
\mathbf V_b+
\boldsymbol\omega_b\times(\mathbf x-\mathbf c_b).
\]

At or around the joint, these fields may differ strongly.

A standard smooth MLP cannot represent an exact mathematical jump without passing through a finite transition layer.

The proposed interpretation is:

\[
\boxed{
\text{do not require the neural field to be rigid in empty space or at the articulation transition}.
}
\]

A single global CurlField can approximate two different rigid motions by forming a thin divergence-free transition/shear layer.

This preserves incompressibility while allowing local strain near the articulation.

---

# 12. Material support from point centers

The central simplification for the first implementation is:

\[
\boxed{
\text{a cell is occupied if and only if it contains at least one Gaussian center}.
}
\]

No Gaussian extent is considered.

At time \(t\), transported centers are

\[
\mathcal P(t)
=
\{
\mathbf x_i(t)
\}_{i=1}^N.
\]

Choose a fixed world-space voxel size

\[
h>0.
\]

For point

\[
\mathbf x_i=(x_i,y_i,z_i),
\]

define integer voxel coordinates

\[
\mathbf q_i
=
\left\lfloor
\frac{
\mathbf x_i-\mathbf x_{\min}
}{h}
\right\rfloor.
\]

A voxel \(c\) is occupied when

\[
\boxed{
O_c(t)
=
\mathbf 1
\left[
\exists i:
\mathbf q_i(t)=c
\right].
}
\]

Equivalent implementation:

```python
voxel_ids = floor((xyz - grid_origin) / voxel_size)
occupied_voxels = unique(voxel_ids)
```

The occupancy computation can be completely non-differentiable because it is not the motion representation.

It is only used to decide where physical constraints should be evaluated and to obtain a coarse material-support topology.

---

# 13. The grid no longer represents velocity

This is a fundamental architectural decision.

The old approach was

\[
c
\longrightarrow
\mathbf v_c(t).
\]

The new approach is

\[
(\mathbf x,t)
\longrightarrow
\mathbf A_\theta
\longrightarrow
\nabla\times\mathbf A_\theta
\longrightarrow
\mathbf v.
\]

The voxel grid stores only information such as

```text
voxel coordinate
occupied / empty
Gaussian center IDs currently in voxel
possibly point count
possibly neighborhood connectivity
```

There is:

- no `velocity[cell]`;
- no velocity interpolation across cells;
- no cell optimizer;
- no cell-specific Adam state;
- no split-child velocity initialization;
- no velocity-dependent octree topology.

The grid is now purely a support/index structure.

---

# 14. Recompute occupancy after transport

Because this is Eulerian motion, the material points move through world space.

At every committed frame,

\[
\mathbf x_i^k
\rightarrow
\mathbf x_i^{k+1}.
\]

After committing the new positions, recompute

\[
O_c(t_{k+1}).
\]

Therefore the support map automatically follows the transported point cloud.

The grid itself does not move.

Conceptually:

```text
world-space voxelization

frame k:

 . . X X . .
 . X X X . .
 . . . . . .

frame k+1:

 . . . X X .
 . . X X X .
 . . . . . .
```

The occupied set changes because the points move.

This is consistent with Eulerian coordinates.

---

# 15. Point-center support is not the true volumetric material domain

An important limitation must be stated explicitly.

3DGS centers generally sample visible object geometry, often concentrated around surfaces.

Therefore

\[
O_c=1
\]

means

> this voxel contains at least one reconstructed material tracer point,

not necessarily

> the entire voxel is physically filled with solid material.

At the first stage we should therefore use the term

\[
\boxed{\text{point-supported material region}}
\]

rather than claiming exact volumetric occupancy.

This matters because a sparse surface point cloud may produce a shell-like occupancy pattern.

We should not fill interior voxels or infer an SDF in the initial implementation.

---

# 16. Exterior support boundary

Using point-center occupancy, define a neighborhood

\[
\mathcal N(c).
\]

For a regular 3D grid, possible neighborhoods are:

- 6-connected: face-sharing voxels;
- 18-connected: face + edge sharing;
- 26-connected: face + edge + corner sharing.

For the first implementation, use the **6-connected neighborhood** because it has the clearest geometric meaning and is least likely to create diagonal connectivity artifacts.

Define an occupied support-boundary voxel as

\[
\boxed{
B_c(t)=1
}
\]

if

\[
O_c(t)=1
\]

and at least one face neighbor is empty:

\[
\exists n\in\mathcal N_6(c)
:
O_n(t)=0.
\]

Thus

\[
B_c
=
O_c
\land
\left[
\sum_{n\in\mathcal N_6(c)}O_n<6
\right].
\]

This gives a discrete estimate of the boundary of the observed point support.

---

# 17. Sparse point clouds may make most occupied cells boundary cells

Because we initially use only centers, if the point cloud is sparse relative to voxel size, nearly every occupied voxel can have an empty neighbor.

Therefore the distinction

\[
\text{interior voxel}
\quad\text{vs}\quad
\text{boundary voxel}
\]

may be weak.

This is not a failure of CurlField.

It simply means the point-center occupancy is a sampling support, not a watertight solid reconstruction.

The voxel size \(h\) controls this strongly:

### Very small \(h\)

- many isolated occupied voxels;
- fragmented support;
- almost everything appears to be boundary;
- poor neighborhood statistics.

### Very large \(h\)

- distinct links or surfaces may be artificially connected;
- joint gaps can disappear;
- support geometry becomes overly coarse.

The initial experiment should therefore sweep a small number of voxel sizes and report:

- number of occupied voxels;
- average points per occupied voxel;
- number of connected components;
- fraction of occupied voxels marked as boundary.

This grid-size sensitivity should be treated as a diagnostic, not hidden.

---

# 18. Where to evaluate physical losses

The safest first implementation is to evaluate the physical field primarily at material tracer points.

At frame \(t_k\),

\[
\mathbf x_i^k
\]

is a point where we know reconstructed material exists.

Evaluate

\[
\mathbf v_i
=
\mathbf v_\theta(\mathbf x_i^k,t_k),
\]

and

\[
D_i
=
D(\mathbf x_i^k,t_k).
\]

Then compute the local strain loss over those points.

This avoids enforcing rigid material physics in arbitrary empty world space.

The occupancy grid can additionally be used to subsample or group the points spatially.

For example:

```text
for every occupied voxel:
    collect point IDs
    sample at most K points
    evaluate strain there
```

This prevents dense parts of the Gaussian point cloud from dominating the physics loss simply because they contain many more splats.

---

# 19. Why not evaluate strain uniformly in the whole bounding box

Suppose two rigid links are separated by a small empty gap.

Their velocities may be very different.

A continuous MLP must transition from one velocity to the other somewhere.

If we enforce

\[
D=0
\]

at random points throughout that empty gap, we incorrectly demand that empty space itself behave as rigid material.

Instead, if physics is evaluated only at point-supported locations,

```text
link A points       empty gap       link B points

o o o o o                            o o o o o
D≈0                                  D≈0

          network is free to
          transition here
```

the velocity field can smoothly connect the two rigid motions in unsampled space.

This is a major mechanism for handling apparent discontinuities without requiring an explicit discontinuous neural network.

---

# 20. What if two articulated links touch and there is no empty gap?

Then the field cannot hide the transition entirely in empty space.

Some material-supported region will exhibit nonzero strain.

This is useful.

Instead of considering it a failure, interpret it as an articulation cue.

For each point,

\[
s_i(t)
=
\left\|
D(\mathbf x_i(t),t)
\right\|_F.
\]

Inside a rigid link,

\[
s_i\approx0.
\]

Near a transition between different rigid motions,

\[
s_i
\]

should rise.

Thus the strain field can become a candidate joint-boundary signal.

---

# 21. Do not globally enforce zero strain with an ordinary L2 loss initially

If the entire foreground is articulated, forcing

\[
D=0
\]

equally everywhere can make the optimizer prefer one global rigid motion.

Instead, use a robust penalty

\[
\boxed{
\mathcal L_{\rm piecewise-rigid}
=
\frac1N
\sum_i
\rho_{\rm robust}
\left(
\|D_i\|_F
\right)
}.
\]

Possible choices:

### Charbonnier

\[
\rho(r)
=
\sqrt{r^2+\epsilon^2}.
\]

### Huber

Quadratic near zero and linear for large residuals.

### Geman-McClure

\[
\rho(r)
=
\frac{r^2}{r^2+\sigma^2}.
\]

The desired behavior is:

- make the large majority of the material highly rigid;
- tolerate a sparse set of high-strain points if required by image evidence.

That sparse set can later be analyzed as a joint/interface candidate.

---

# 22. Optional future explicit rigidity-confidence field

Only after the simple robust-strain formulation is understood, consider learning

\[
r_\phi(\mathbf x,t)\in[0,1],
\]

where

- \(r\approx1\): rigid material;
- \(r\approx0\): allowed articulation transition.

Then

\[
\mathcal L_{\rm strain}
=
\sum_i
r_i
\|D_i\|_F^2.
\]

A separate sparsity penalty would prevent the trivial solution \(r=0\) everywhere.

For example,

\[
\mathcal L_{\rm nonrigid-volume}
=
\sum_i(1-r_i).
\]

This is **not part of the first implementation**.

The first version should use robust strain only.

---

# 23. Internal articulation boundary is not the same as exterior material boundary

The point-occupancy grid can identify

\[
\boxed{
\text{point support}
\leftrightarrow
\text{empty space}
}
\]

but it does not automatically identify

\[
\boxed{
\text{rigid link A}
\leftrightarrow
\text{rigid link B}.
}
\]

If two links touch, both sides are occupied.

Therefore use different signals for different boundaries.

### Exterior/support boundary

Derived from occupancy:

\[
O_c=1,
\qquad
\exists n: O_n=0.
\]

### Internal kinematic boundary

Derived from motion:

- elevated strain;
- abrupt change in local angular velocity;
- abrupt change in local rigid-motion fit;
- persistent incompatibility between neighboring material points.

Do not overload the occupancy grid with the task of discovering the joint.

---

# 24. Possible local rigid-motion analysis from the CurlField

Once a good velocity field exists, local rigid motion can be extracted directly.

For a rigid field,

\[
\nabla\mathbf v
=
[\boldsymbol\omega]_\times.
\]

Therefore the antisymmetric component

\[
W
=
\frac12
\left(
\nabla v-\nabla v^T
\right)
\]

encodes angular velocity.

Specifically,

\[
W
=
\begin{bmatrix}
0 & -\omega_z & \omega_y\\
\omega_z & 0 & -\omega_x\\
-\omega_y & \omega_x & 0
\end{bmatrix}.
\]

Thus

\[
\boldsymbol\omega(\mathbf x,t)
\]

can be recovered locally from derivatives of the learned Eulerian field.

This suggests a much cleaner later joint-estimation route:

1. learn the continuous field;
2. compute local strain \(D\);
3. identify mostly rigid material regions;
4. compute local angular velocity from \(W\);
5. cluster regions with coherent rigid motion;
6. infer revolute/prismatic joint geometry from relative rigid motions.

This avoids using arbitrary cell velocity clusters.

---

# 25. Visual supervision remains responsible for choosing the actual motion

The physical constraints do not determine the motion by themselves.

For example, all of the following satisfy incompressibility:

- zero velocity;
- translation;
- rotation;
- many non-rigid divergence-free flows.

Therefore the actual field must still be driven by visual evidence.

The primary supervision remains:

\[
\mathcal L_{\rm RGB},
\]

\[
\mathcal L_{\rm SSIM},
\]

\[
\mathcal L_{\rm depth},
\]

\[
\mathcal L_{\rm DINO/feature},
\]

and, where trustworthy,

\[
\mathcal L_{\rm rasterized\ flow}.
\]

The physical model restricts the set of plausible motions.

The images select which motion from that admissible set explains the observations.

---

# 26. Differential-gradient evidence in the new formulation

The previously computed differential-gradient signal remains useful, but its role should be limited and explicit.

It can indicate which canonical Gaussian centers are likely to belong to dynamically affected regions.

It should **not**:

- initialize a physical velocity;
- generate pseudo 3D velocity vectors;
- modify \(\mathbf A_\theta\) directly;
- create independent velocity parameters.

Possible uses are:

1. frozen initial static/dynamic prior for the Lagrangian baseline;
2. weighting of visual motion losses;
3. selecting points where stronger CurlField optimization is needed;
4. diagnostics comparing expected dynamic support with learned speed.

The CurlField itself should begin from a near-zero velocity state.

---

# 27. Initialization of the CurlField

A useful physical default is

\[
\mathbf v(\mathbf x,t)\approx0.
\]

Therefore initialize the final layer of the \(A_\theta\) MLP near zero.

Because velocity is obtained from spatial derivatives of \(\mathbf A\), care is required:

- simply setting a constant output gives zero curl;
- the network should start close to a spatially constant/very-low-gradient potential.

The first field should therefore represent approximately static motion.

Visual gradients then create the required spatial derivatives.

This parallels 4DGen's useful principle of initializing the space-time representation so static content does not move unless evidence requires it.

---

# 28. Neural architecture

A minimal first architecture can be:

```text
input:
    normalized x, y, z
    normalized physical time t

optional:
    Fourier encoding of x
    Fourier encoding of t

MLP:
    3 hidden layers
    width 64 or 128
    smooth activation

output:
    A_x, A_y, A_z
```

Because spatial derivatives and second derivatives are required, activation smoothness matters.

ReLU produces piecewise-constant first derivatives and undefined second derivatives at activation boundaries.

Prefer:

- Softplus;
- SiLU;
- tanh;
- sine/SIREN if later needed.

A good first choice is **Softplus** or **SiLU**.

---

# 29. Derivative order and computational implications

Velocity is already a first spatial derivative of \(\mathbf A\):

\[
v=\nabla\times A.
\]

Strain requires

\[
\nabla v,
\]

which is a second spatial derivative of \(A\).

Therefore training with strain requires second-order automatic differentiation.

This has real cost.

We should not evaluate strain at every Gaussian on every camera render.

Instead:

1. rendering losses use all relevant Gaussians/cameras;
2. physics loss samples a limited subset of material points once per optimizer iteration;
3. those points are spatially balanced using occupied voxels.

Example:

```text
occupied voxels
    ↓
sample ≤ K voxels
    ↓
sample one/few Gaussian centers per voxel
    ↓
compute curl and second derivatives
    ↓
strain loss
```

This keeps the physics cost controlled.

---

# 30. Spatially balanced physics sampling using the occupancy grid

If one part of the object has 3000 Gaussians and another has 300, a naive average over Gaussian centers weights the dense part ten times more strongly.

The occupancy grid provides a simple correction.

Let occupied voxel \(c\) contain point set

\[
\mathcal G_c.
\]

A balanced loss is

\[
\mathcal L_{\rm strain}
=
\frac{1}{|\mathcal C_{\rm occ}|}
\sum_{c\in\mathcal C_{\rm occ}}
\frac{1}{|\mathcal S_c|}
\sum_{i\in\mathcal S_c}
\rho(\|D_i\|),
\]

where \(\mathcal S_c\) is a small sample from that voxel.

Then every occupied spatial region contributes approximately equally.

This is another useful role for the grid without making it part of the velocity representation.

---

# 31. Temporal regularity

The field should not change arbitrarily between neighboring physical times.

Possible temporal regularization is

\[
\mathcal L_{\partial_t v}
=
\left\|
\frac{\partial v}{\partial t}
\right\|^2,
\]

or acceleration along transported particles,

\[
\mathbf a_i
=
\frac{d}{dt}
\mathbf v(\mathbf x_i(t),t).
\]

The material derivative is

\[
\boxed{
\frac{D\mathbf v}{Dt}
=
\frac{\partial\mathbf v}{\partial t}
+
(\mathbf v\cdot\nabla)\mathbf v.
}
\]

However, this should not initially be over-constrained.

Articulated objects can accelerate significantly.

A small smoothness prior is reasonable; a strong zero-acceleration prior is not.

The 4DGen paper supports the general value of temporal smoothness by penalizing acceleration of point trajectories over time.

---

# 32. Integration must use physical video time

If the video timestamp difference is

\[
\Delta t_k,
\]

then transport must be

\[
\frac{d\mathbf x}{dt}
=
\mathbf v(\mathbf x,t)
\]

integrated over exactly

\[
[t_k,t_{k+1}].
\]

Speed must therefore have physical units

\[
[\mathbf v]
=
\text{world distance}/\text{second}.
\]

Changing the number of optimizer iterations must never change the physical displacement represented by a frame.

This remains a non-negotiable contract for both the Lagrangian and Eulerian experiments.

---

# 33. Frame-wise optimization contract

For frame interval \(k\to k+1\):

```text
committed x_k
    │
    ├── optimizer iteration 1:
    │      integrate x_k with current field
    │      render all cameras
    │      compute losses
    │      update network
    │
    ├── optimizer iteration 2:
    │      integrate SAME x_k with updated field
    │      ...
    │
    └── optimizer iteration N:
           integrate SAME x_k
           update network

after final update:
    integrate x_k once with final field
    commit x_{k+1}
```

Never accumulate the local optimizer iterations as physical time.

---

# 34. How the occupancy grid is updated during local optimization

There are two possible occupancy states:

### Committed occupancy

Built from the previous committed frame positions

\[
\mathbf x_i^k.
\]

This is stable during all local optimizer iterations.

### Candidate endpoint occupancy

Built from the current predicted

\[
\hat{\mathbf x}_i^{k+1}.
\]

For the first implementation, use committed occupancy for selecting physics samples during optimization.

After the interval is fully optimized and

\[
\mathbf x_i^{k+1}
\]

is committed, rebuild the occupancy grid.

This avoids having the support mask flicker during every optimizer update.

Later, candidate endpoint occupancy can be added if needed.

---

# 35. Boundary handling in the first implementation

For the first point-center version:

- determine occupied cells from centers;
- determine occupied/empty boundary using 6-neighbors;
- do **not** explicitly impose a boundary condition such as \(v\cdot n=0\).

Why?

The object boundary itself moves with the material.

The surface velocity is not generally tangent to a fixed world-space boundary.

A rigid translating object has nonzero normal velocity relative to its current surface location.

Therefore a fixed-wall no-penetration condition would be wrong.

The role of the occupancy boundary is initially only:

- identifying where support ends;
- preventing physics sampling far into empty space;
- visualization/debugging;
- possible future local neighborhood handling.

---

# 36. What incompressibility means at the moving object boundary

Inside the material,

\[
\nabla\cdot v=0.
\]

Outside the object, the neural field is merely a mathematical extension.

We do not assign physical density or material interpretation to it.

At the moving interface, the actual material boundary is advected by the material velocity.

Thus we do not need to explicitly model the air/material density jump in the first center-based CurlField.

This is the precise sense in which density can disappear from our learning problem while the physical object still moves through Eulerian space.

---

# 37. Handling discontinuity between disconnected supports

Suppose the point-support voxel graph has two disconnected components.

Then there is no reason to impose rigidity between them.

The shared CurlField remains globally continuous, but the physical strain loss is sampled only on each material-supported component.

The network may transition freely in the empty region between them.

This is one of the cleanest ways to reconcile a smooth neural field with different rigid motions.

---

# 38. Handling discontinuity between touching supports

If two links touch, occupancy may form one connected component.

Then the visual data and robust strain loss must determine where the local rigid-motion change occurs.

A useful diagnostic is to render/visualize:

\[
\|D(x_i,t)\|_F
\]

on the Gaussian centers.

Expected behavior:

```text
link interior      joint region       link interior
low strain         high strain        low strain
```

If high strain appears randomly across an entire link, the velocity field is not learning a piecewise-rigid solution.

If it concentrates near the true articulation, that is strong evidence that the formulation is working.

---

# 39. Why not immediately use multiple CurlFields

One possibility would be

\[
v
=
\sum_k p_k v_k,
\qquad
v_k=\nabla\times A_k.
\]

But even if every expert is divergence-free,

\[
\nabla\cdot v_k=0,
\]

the mixture generally is not:

\[
\nabla\cdot
\left(
\sum_kp_kv_k
\right)
=
\sum_k
\nabla p_k\cdot v_k.
\]

Therefore mixtures introduce a new conservation problem at the gating boundaries.

The first implementation should use one global vector potential.

Only if a single continuous CurlField demonstrably cannot represent the articulated motion should multi-field formulations be reconsidered.

---

# 40. Relation to 4D Scaffold Gaussian Splatting

The 4D Scaffold paper provides several useful conceptual lessons.

## 40.1 Sparse grid as representation support, not hard physical velocity bins

The paper uses sparse 4D anchors with learned feature vectors.

Shared MLPs decode the anchor features into local neural Gaussian attributes, including velocity.

The important transferable principle is:

\[
\boxed{
\text{grid/anchors allocate representation capacity}
}
\]

rather than

\[
\boxed{
\text{every grid cell is an independent physical velocity}.
}
\]

This supports abandoning the hard cell-velocity interpretation.

## 40.2 Simple motion parametrization can outperform overcomplicated trajectories

The paper models local Gaussian motion using linear velocity over the Gaussian's temporal support.

This is evidence that a compact motion parameterization can be sufficient when the spatial representation is well organized.

## 40.3 Dynamic-aware capacity is allocated from accumulated rendering gradients

Their anchor-growing mechanism accumulates 2D position gradients and compensates for temporal coverage before voxelizing the evidence and creating new anchors.

The transferable lesson is:

\[
\boxed{
\text{visual gradient tells us where representation is insufficient}.
}
\]

It should not be interpreted as a direct physical velocity measurement.

## 40.4 For CurlField

If a simple global MLP lacks capacity later, we can borrow the scaffold principle by introducing:

- sparse learnable support features;
- interpolation of nearby support features;
- one shared decoder producing \(\mathbf A\).

Even then, the decoded velocity remains continuous and curl-based.

The support should refine capacity, not instantiate independent velocity vectors.

---

# 41. Relation to 4DGen / HexPlane

4DGen uses a multi-resolution HexPlane representation over

\[
xy,\ xz,\ yz,\ xt,\ yt,\ zt
\]

and decodes interpolated features with an MLP into dynamic deformation.

Transferable insights:

1. use continuous interpolation rather than hard nearest-cell velocity;
2. use multiple spatial/temporal resolutions;
3. initialize the dynamic representation close to static;
4. apply temporal smoothness explicitly;
5. separate coarse capacity from fine refinement.

If the minimal CurlField MLP is underexpressive, a future representation can be

\[
\Phi(x,t)
=
\text{multi-resolution interpolated features},
\]

\[
A_\theta
=
f_\theta(\Phi(x,t)),
\]

\[
v=\nabla\times A_\theta.
\]

This would preserve the exact divergence-free construction while adding local spatial capacity.

---

# 42. First CurlField experiment: keep it intentionally minimal

The first experiment should contain only:

### Inputs

- current transported center \(\mathbf x_i(t)\);
- physical time \(t\).

### Motion model

\[
A_\theta(x,t)
\]

and

\[
v=\nabla\times A.
\]

### Integration

real \(\Delta t\).

### Visual losses

- RGB/SSIM;
- depth;
- DINO/feature;
- rasterized flow if already validated.

### Physics

- exact divergence-free field by construction;
- robust local strain penalty.

### Support

- hard voxel occupancy from center positions only;
- no adaptive topology;
- no Gaussian extent.

### Dynamic prior

- fixed differential-gradient evidence for diagnostics/optional weighting;
- no learned membership in the first experiment.

This minimality is essential.

---

# 43. Metrics that must be logged

## 43.1 Motion scale

At every frame:

- speed median;
- speed p90;
- speed p99;
- speed maximum;
- displacement median;
- displacement p90;
- displacement maximum.

## 43.2 Physics

- divergence numerical residual;
- strain median;
- strain p90;
- strain p99;
- antisymmetric Jacobian magnitude;
- estimated local angular-speed magnitude.

The divergence residual should be near numerical precision if curl is implemented correctly.

## 43.3 Support grid

- occupied voxel count;
- boundary voxel count;
- connected components;
- average points per occupied voxel;
- median points per occupied voxel;
- max points per occupied voxel.

## 43.4 Visual

- RGB L1;
- SSIM;
- depth loss;
- DINO loss;
- observed vs predicted rasterized optical-flow magnitude.

## 43.5 Stability

- maximum network gradient norm;
- maximum point speed;
- any NaN/Inf;
- points leaving the predefined normalization domain.

---

# 44. Essential debug visualizations

At every selected frame generate:

1. transported Gaussian center point cloud;
2. velocity arrows on a subsample;
3. occupied voxel centers;
4. support-boundary voxels;
5. strain magnitude on Gaussian centers;
6. local angular velocity on Gaussian centers;
7. differential-gradient prior compared with learned speed;
8. per-camera predicted 2D motion versus observed flow/DINO tracks.

This is necessary so that numerical loss reduction cannot hide physically nonsensical motion.

---

# 45. Lagrangian vs CurlField comparison protocol

Use identical:

- foreground Gaussian set;
- frame timestamps;
- RGB/depth/DINO targets;
- local iteration count;
- renderer;
- cameras;
- losses wherever representation-independent;
- rigid-neighborhood construction where applicable.

Compare:

| Property | Lagrangian MLP | CurlField |
|---|---|---|
| Query | canonical identity + time | current world position + time |
| Motion coordinate | material | spatial |
| Incompressibility | not automatic | exact via curl |
| Spatial continuity | indirect | intrinsic |
| Point identity | explicit | not needed for velocity |
| Joint discontinuity | easy per identity | thin spatial transition |
| Grid needed | no | support/index only |
| Hard cell boundaries | no | no |
| Cell splitting | no | no |

The goal is not necessarily for CurlField to achieve lower image loss.

The goal is to determine whether it recovers a more physically meaningful and interpretable motion field.

---

# 46. Failure interpretations

## Lagrangian succeeds, CurlField fails

Likely issues:

- CurlField network capacity too low;
- derivative optimization too difficult;
- smooth spatial field cannot express articulation sharply enough;
- strain loss too strong;
- normalization/encoding inadequate.

Then consider HexPlane/scaffold features while retaining curl.

## Both fail

Likely issues:

- visual supervision is insufficient or incorrectly routed;
- DINO/flow tracks are bad;
- rendering gradient does not recover motion;
- depth supervision is inconsistent;
- framewise optimization strategy is wrong.

Do not blame the Eulerian representation first.

## Both succeed

Then CurlField becomes attractive because it gives:

- a continuous Eulerian field;
- exact incompressibility;
- local strain;
- local vorticity/angular velocity;
- natural joint-boundary cues.

---

# 47. Implementation order

## Step 1 — Lagrangian baseline

Finish and validate the simple shared per-point velocity MLP.

No grid-related mechanisms.

## Step 2 — Direct continuous Eulerian MLP baseline

Before curl, optionally test

\[
v_\theta(x,t)=\mathrm{MLP}(x,t)
\]

to isolate whether Eulerian coordinates themselves are sufficient.

## Step 3 — CurlField

Implement

\[
A_\theta(x,t),
\qquad
v=\nabla\times A.
\]

Verify numerically that

\[
\nabla\cdot v\approx0.
\]

## Step 4 — point-center occupancy

Implement a fixed-size hash/sparse voxelization of transported centers.

No velocities stored in voxels.

## Step 5 — robust strain physics

Compute

\[
D=\operatorname{sym}(\nabla v)
\]

at spatially balanced material points.

## Step 6 — diagnostics

Generate support, velocity, strain, vorticity, and 2D track visualizations.

## Step 7 — only if needed, add representation capacity

Possible upgrades:

- Fourier encoding;
- multi-resolution hash features;
- HexPlane features;
- sparse scaffold features.

Always decode a continuous vector potential rather than returning to independent cell velocities.

---

# 48. Explicit list of mechanisms that should NOT be part of CurlField v1

Do not include:

- adaptive octree velocity cells;
- cell velocity trajectories;
- grid split/merge;
- flow-triangulated 3D velocity seeding;
- per-cell static/dynamic label;
- dynamic membership optimization;
- cell disagreement thresholds;
- cell velocity inheritance;
- cell root expansion;
- Gaussian ellipsoid occupancy;
- Gaussian opacity-based occupancy;
- learned interface field;
- multiple velocity experts;
- explicit density network;
- finite-volume mass conservation.

Each can be reconsidered later only if a clear deficiency requires it.

---

# 49. Core conceptual summary

The proposed representation is:

\[
\boxed{
\text{Gaussian centers}
=
\text{material tracer points}
}
\]

\[
\boxed{
\text{voxel grid}
=
\text{where tracer points currently exist}
}
\]

\[
\boxed{
A_\theta(x,t)
=
\text{continuous latent vector potential}
}
\]

\[
\boxed{
v(x,t)
=
\nabla\times A_\theta(x,t)
}
\]

\[
\boxed{
\nabla\cdot v=0
\quad
\text{exactly}
}
\]

\[
\boxed{
\operatorname{sym}(\nabla v)\approx0
\quad
\text{through robust local rigidity}
}
\]

\[
\boxed{
\text{visual losses}
\rightarrow
\text{select the actual motion}
}
\]

\[
\boxed{
\text{high local strain}
\rightarrow
\text{candidate articulation transition}
}
\]

\[
\boxed{
\text{grid occupancy}
\rightarrow
\text{material support, not motion}
}
\]

The first implementation should remain a point-cloud experiment.  
Only after this formulation is understood should Gaussian finite extent, continuous occupancy, or more sophisticated spatial-support representations be introduced.
