# CODEX — IMPLEMENT A SEPARATE MOTION-COHERENCE STATIC/DYNAMIC FILTER
## Phase 1 free trajectories → rigid-coherence segmentation → frozen Phase-2 support

# 0. Scope

Implement a **new, separate experimental path** for static/dynamic selection between Phase 1 and Phase 2.

Do **not** modify or reuse the current Otsu selector.
Do **not** combine Otsu displacement scores with the new method.
Do **not** average, vote, union, intersect, or fuse the results of the two methods.

The new branch must infer static/dynamic membership **only from the geometric coherence of the free 3D trajectories produced by Phase 1**.

The algorithm must not use:
- Otsu;
- median displacement;
- top-k displacement;
- hand-picked displacement thresholds;
- RGB residual magnitude;
- DINO/SAM/semantic features;
- P/R type;
- joint axis;
- GT labels;
- Phase-2 render losses.

GT static/dynamic labels are used **only after the prediction is finished**, for evaluation and debugging.

The new branch is:

\[
\boxed{
\text{Phase 1 free trajectories}
\longrightarrow
\text{trajectory-coherence analysis}
\longrightarrow
\text{static/dynamic support}
\longrightarrow
\text{Phase 2}
}
\]

The old branch remains available for A/B comparison, but its output must never enter the new branch.

---

# 1. Scientific principle

The visual debug shows two qualitatively different kinds of free motion.

True dynamic points may move a lot, but move **coherently as a rigid body**. For nearby points \(i,j\),

\[
\|x_i(t)-x_j(t)\|
\]

remains essentially constant through the sequence.

False dynamic points are free-motion DOFs used by Phase 1 to compensate changes such as illumination, appearance mismatch, alpha-ordering effects, depth/model mismatch, or camera-specific residual structure. Their trajectories can have non-negligible amplitude, but neighboring points do not preserve their relative geometry. They stretch, compress and jitter with respect to one another.

The new filter must exploit exactly this distinction.

---

# 2. Do not infer the joint yet

This filter must be joint-type agnostic.

It must not assume:
- prismatic;
- revolute;
- axis;
- COM;
- \(q(t)\);
- \(\dot q(t)\).

It asks only:

\[
\boxed{
\text{Do these free trajectories preserve a common rigid metric over time?}
}
\]

This property is valid for every rigid body.

The output is only:
```text
STATIC
DYNAMIC
```
plus optionally:
```text
AMBIGUOUS
```
if the coherence partition is not identifiable.

Phase 2 receives the final DYNAMIC support.

---

# 3. Work at supernode level first

The primary selection unit must be the same supernode structure used by Phase 1.

Let
\[
s=1,\ldots,N_S
\]
index supernodes.

For each supernode obtain its Phase-1 free trajectory:
\[
x_s(t)\in\mathbb R^3.
\]

If Phase 1 directly stores supernode states, use them.

If only Gaussian trajectories are persisted, reconstruct a supernode trajectory using fixed Phase-1 supernode membership:
\[
x_s(t)
=
\frac{
\sum_{i\in s}w_i x_i(t)
}{
\sum_{i\in s}w_i
}.
\]

Use one documented fixed weight convention.

Do not use GT to construct these trajectories.

After final supernode labels are inferred, propagate labels to member Gaussians.

---

# 4. Canonical spatial graph

Construct one fixed graph in canonical frame \(t=0\).

Prefer reusing the exact canonical kNN/supernode graph already maintained by the repository.

If a new graph is required, expose one global parameter:
```text
--coherence-knn
```
with one fixed default for all datasets.

Do not tune it per dataset.

The graph is:
\[
G=(V,E).
\]

Every edge
\[
(i,j)\in E
\]
must refer to the same two supernodes for the entire sequence.

Do not rebuild neighbors at every frame.

---

# 5. Core invariant: pairwise distance preservation

For every edge \((i,j)\), define:
\[
r_{ij}(t)=x_j(t)-x_i(t)
\]
and
\[
\ell_{ij}(t)=\|r_{ij}(t)\|.
\]

For any rigid-body motion:
\[
\boxed{
\ell_{ij}(t)=\text{constant}
}
\]
for all points belonging to the same rigid component.

This remains true for:
- static bodies;
- translation;
- rotation;
- arbitrary rigid \(SE(3)\);
- variable speed;
- stop/restart;
- reversal.

Pairwise metric preservation is the primary signal.

---

# 6. Whole-window finite strain

Do not use only frame 0 versus frame T.

Define the robust reference length:
\[
\boxed{
\bar \ell_{ij}
=
\operatorname{median}_t \ell_{ij}(t).
}
\]

Then define the dimensionless finite log-strain trajectory:
\[
\boxed{
z_{ij}(t)
=
\log
\frac{
\ell_{ij}(t)+\epsilon_\ell
}{
\bar\ell_{ij}+\epsilon_\ell
}.
}
\]

For a perfectly rigid edge:
\[
z_{ij}(t)=0
\quad\forall t.
\]

Use \(\epsilon_\ell\) only as a numerical guard based on machine/data scale, never as a motion threshold.

---

# 7. Primary edge coherence score

The selection algorithm must use **one primary coherence score**, not a hand-designed mixture of unrelated features.

Use:
\[
\boxed{
C_{ij}
=
\sqrt{
\frac{
\sum_t w_{ij}(t)z_{ij}(t)^2
}{
\sum_t w_{ij}(t)+\epsilon
}
}.
}
\]

Here \(w_{ij}(t)\) is a validity/observability weight.

If reliable Phase-1 visibility/confidence exists, use it. Otherwise use:
\[
w_{ij}(t)=1
\]
for frames in which both trajectories are valid.

Do not use RGB residual magnitude as this weight.

Interpretation:
\[
C_{ij}\approx0
\]
means the edge preserves its metric over the entire sequence.

Large \(C_{ij}\) means the trajectories are not mutually rigid.

This is the main quantity used for segmentation.

---

# 8. Differential symmetric-strain diagnostic

Also compute the differential version for theory/debug.

Using the real time step:
\[
\Delta t_t=t_{t+1}-t_t,
\]
define:
\[
\boxed{
\epsilon_{ij}(t)
=
\frac{
z_{ij}(t+1)-z_{ij}(t)
}{
\Delta t_t
}.
}
\]

Equivalently:
\[
\epsilon_{ij}(t)
=
\frac{
r_{ij}(t)^T(v_j(t)-v_i(t))
}{
\|r_{ij}(t)\|^2
}.
\]

Because:
\[
v_j-v_i\simeq\nabla v\,r_{ij},
\]
and
\[
\nabla v=D+\Omega,
\]
with
\[
D=\frac12(\nabla v+\nabla v^T),
\qquad
\Omega=\frac12(\nabla v-\nabla v^T),
\]
we have:
\[
r^T\Omega r=0
\]
and therefore:
\[
\boxed{
r_{ij}^T(v_j-v_i)
=
r_{ij}^TDr_{ij}.
}
\]

For rigid motion:
\[
\boxed{D=0}
\]
so:
\[
\boxed{\epsilon_{ij}(t)=0.}
\]

This is a diagnostic of the same metric-coherence principle.

Do not combine it with \(C_{ij}\) through a weighted production score.

---

# 9. Jitter diagnostic

For visualization/debug save:
\[
\boxed{
J_{ij}
=
\sqrt{
\frac1{T-1}
\sum_t
\left[z_{ij}(t+1)-z_{ij}(t)\right]^2
}.
}
\]

Interpretation:
- low \(C_{ij}\), low \(J_{ij}\): rigid-compatible edge;
- high \(C_{ij}\), moderate \(J_{ij}\): smooth non-rigid drift;
- high \(C_{ij}\), high \(J_{ij}\): chaotic renderer-driven motion.

Again \(J_{ij}\) is diagnostic only.

---

# 10. Persist the whole time signal

Save the full temporal signal for edge debugging:
```text
edge_strain_timeseries.npz
```

with:
```text
edge_i
edge_j
z[edge, frame]
epsilon[edge, frame-1]
C_edge
J_edge
```

For very large graphs, save all scalar scores and a deterministic subset of full time series.

The debug must expose whether an edge:
- stays flat;
- slowly drifts;
- oscillates;
- spikes at isolated frames.

---

# 11. Convert coherence into graph affinity

Use only \(C_{ij}\) to build affinity.

A self-scaled form is recommended:
\[
\boxed{
W_{ij}
=
\exp\left(
-\frac{C_{ij}^2}{\sigma_i\sigma_j+\epsilon}
\right).
}
\]

Use:
\[
\boxed{
\sigma_i
=
\operatorname{median}_{j\in\mathcal N_i}C_{ij}.
}
\]

Use one documented global fallback only when the local median is numerically zero.

Do not use displacement amplitude in \(W\).

---

# 12. Binary coherence partition for K=1

For the current K=1 articulated experiments, infer exactly two coherent components:
\[
\boxed{
\mathcal C_0,\mathcal C_1.
}
\]

Use normalized spectral cut / normalized cut on \(W\).

The partition must be determined only by edge coherence.

Do not:
- seed with Otsu;
- seed with displacement magnitude;
- initialize from current support;
- initialize from GT;
- force a predetermined dynamic fraction.

---

# 13. Why two coherent components are expected

For K=1 the ideal geometry contains:
1. a static rigid component;
2. one moving rigid component.

Static-static edges:
\[
C_{ij}\approx0.
\]

Moving-moving edges:
\[
C_{ij}\approx0.
\]

Static-moving boundary edges:
\[
C_{ij}>0
\]
because their relative distance changes.

False dynamic trajectories generated independently by the renderer should have poor coherence with both rigid components.

---

# 14. Ambiguity must be visible

After partitioning, compute:
\[
\boxed{
C_i^{in}
=
\operatorname{median}_{j\in\mathcal N_i,l_j=l_i}C_{ij}
}
\]
and:
\[
\boxed{
C_i^{out}
=
\operatorname{median}_{j\in\mathcal N_i,l_j\neq l_i}C_{ij}.
}
\]

Persist these as confidence/debug values.

Do not repair ambiguous nodes with Otsu.

---

# 15. Whole-window rigid validation of each component

After the coherence-only partition has been produced, validate each component independently.

For each component \(\mathcal C_k\) and frame \(t\), solve weighted Kabsch:
\[
\boxed{
(R_{k,t},p_{k,t})
=
\arg\min_{R\in SO(3),p}
\sum_{i\in\mathcal C_k}
w_i
\|x_i(t)-(Rx_i(0)+p)\|^2.
}
\]

This fit does **not** initialize or construct the graph partition.

It validates the result.

Compute:
\[
\boxed{
E_k^{rigid}
=
\frac{
\sum_{t,i\in\mathcal C_k}
w_i\|x_i(t)-(R_{k,t}x_i(0)+p_{k,t})\|^2
}{
\sum_{t,i\in\mathcal C_k}w_i+\epsilon
}.
}
\]

Save per-frame residuals too.

---

# 16. Identify static versus dynamic only after partition

Once the two coherent components exist, determine which is static.

For component \(k\), compute identity residual:
\[
\boxed{
E_k^I
=
\frac{
\sum_{t,i\in\mathcal C_k}
w_i\|x_i(t)-x_i(0)\|^2
}{
\sum_{t,i\in\mathcal C_k}w_i+\epsilon
}.
}
\]

Let \(E_k^R\) be the fitted rigid residual.

Define coherent motion gain:
\[
\boxed{
G_k=E_k^I-E_k^R.
}
\]

Then:
- component with the smaller normalized rigid-motion gain → STATIC;
- component with the larger rigid-motion gain → DYNAMIC.

This labeling happens only **after** coherence partitioning.

There is no per-point displacement threshold.

If the components cannot be distinguished reliably, return:
```text
UNRESOLVED_COHERENCE_PARTITION
```
and keep all debug outputs.

---

# 17. Keep component motion magnitude as debug only

Also report:
\[
M_k(t)
=
\sqrt{
\frac{\|p_{k,t}\|^2}{L_k^2}
+
\|\Log(R_{k,t})\|^2
}.
\]

This is interpretation/debug only.

It must not be fed into the graph partition.

---

# 18. Strict separation from Otsu/current selector

Create:
```text
estimator/motion_coherence_support.py
```
and:
```text
tools/run_motion_coherence_support.py
```

The new module must not import:
```text
estimator.moving_support
otsu_threshold
select_moving_support
```

It must not call:
```text
median(displacement)
topk(displacement)
```
for support selection.

Add an AST/import regression test enforcing this.

Keep the old selector unchanged as a baseline.

---

# 19. Separate pipeline option

Add:
```text
--phase2-support-source coherence
```

Keep existing behavior under an explicit separate option such as:
```text
--phase2-support-source current
```

For the new branch:
```text
Phase 1
  ↓
motion_coherence_support
  ↓
freeze predicted dynamic support
  ↓
Phase 2
```

Do not silently replace current behavior.

---

# 20. Never mix branch outputs

Keep:
```text
support_baselines/current/
support_coherence/
```
separate.

Do not:
- union;
- intersect;
- average;
- vote;
- use current support to seed coherence;
- use coherence to alter baseline outputs.

Comparison is evaluation only.

---

# 21. GT is evaluation-only

The user explicitly wants comparison against GT static vs dynamic.

GT must be loaded only after the prediction has been written to disk.

Create:
```python
gt = load_static_dynamic_gt(...)
```

only in the evaluation stage.

Add a no-leakage assertion/order guard.

GT must never enter:
- graph construction;
- \(C_{ij}\);
- affinity;
- spectral cut;
- static/dynamic component identification;
- Kabsch;
- ambiguity handling.

---

# 22. GT mapping must be documented

Determine the real GT representation in each dataset; do not guess.

For every scene write:
```text
gt_mapping.json
```
with:
```text
source
mapping_method
num_static
num_dynamic
num_unmapped
coverage
```

Unmapped points are excluded from metrics.

---

# 23. GT metrics

Compute supernode-level and Gaussian-level metrics when possible.

For the dynamic class:
\[
Precision_D=\frac{TP}{TP+FP},
\]
\[
Recall_D=\frac{TP}{TP+FN},
\]
\[
F1_D=\frac{2PR}{P+R},
\]
\[
IoU_D=\frac{TP}{TP+FP+FN}.
\]

Also compute:
```text
static precision
static recall
static F1
balanced accuracy
Matthews correlation coefficient
confusion matrix
predicted dynamic fraction
GT dynamic fraction
```

Do not report raw accuracy alone.

---

# 24. Compare baseline and coherence only in evaluation

Create:
| scene | selector | Dyn Precision | Dyn Recall | Dyn F1 | Dyn IoU | Balanced Acc |
|---|---|---:|---:|---:|---:|---:|
| ... | current | | | | | |
| ... | coherence | | | | | |

This is evaluation only.

No hybrid support is allowed.

---

# 25. Per-edge debug

Create:
```text
support_coherence/numerical/edges.csv
```
with:
```text
edge_id
node_i
node_j
canonical_distance
C_rigid
J_jitter
same_predicted_component
same_gt_component
```

Save sampled full edge time series too.

---

# 26. Per-node debug

Create:
```text
support_coherence/numerical/nodes.csv
```
with:
```text
supernode_id
predicted_label
gt_label
C_in
C_out
degree
component_id
component_rigid_residual
component_identity_residual
component_motion_gain
num_gaussians
```

Do not use Otsu or displacement as production features.

---

# 27. Finite-strain heatmap

Create:
```text
support_coherence/debug/finite_strain_heatmap.html
```

Visualize:
\[
Z_{e,t}=z_{ij}(t).
\]

Group rows by:
```text
predicted static-static
predicted dynamic-dynamic
predicted cross-component
ambiguous/outlier
```

When GT exists, also create a GT-ordered version.

The expected visual pattern is:
- flat low-strain blocks inside true rigid components;
- strong temporal variation/spikes in false dynamic regions;
- cross-component edges changing as the moving body moves relative to static geometry.

---

# 28. Differential-strain heatmap

Create:
```text
support_coherence/debug/differential_strain_heatmap.html
```
from:
\[
\epsilon_{ij}(t).
\]

Use a symmetric display around zero.

Display scaling must not affect selection.

---

# 29. Representative temporal plots

Create deterministic examples of:
```text
lowest C
median C
highest C
TP dynamic edges
FP dynamic edges
FN dynamic edges
static edges
cross-boundary edges
```
when GT exists.

Plot:
\[
\ell_{ij}(t)
\]
and:
\[
z_{ij}(t).
\]

---

# 30. 3D static/dynamic debugger

Create:
```text
support_coherence/debug/motion_coherence_support_3d.html
```

Reuse the good UX/style of `geometric_motion_debug.html`.

Required:
```text
frame slider
play/pause
speed
loop
camera frustums
show all
show predicted static
show predicted dynamic
show GT static
show GT dynamic
show errors
```

Error layers:
```text
TP dynamic
FP dynamic
FN dynamic
TN static
```

The page must animate the **actual Phase-1 free positions** \(x_i(t)\), not a rigid reconstruction.

---

# 31. Graph-edge visualization

Add toggles for:
```text
canonical kNN edges
low-strain edges
high-strain edges
cross-component edges
```

Display a deterministic manageable subset.

Display settings never affect algorithmic weights.

---

# 32. Node dashboard

Selecting a supernode should show:
```text
predicted label
GT label
degree
C_in
C_out
component
incident-edge median C
incident-edge max C
```

Also show incident edge strain trajectories.

---

# 33. Component debug

Create:
```text
support_coherence/numerical/components.json
```
with:
```text
component_id
num_supernodes
num_gaussians
E_identity
E_rigid
motion_gain
per_frame_kabsch_residual
per_frame_R
per_frame_p
```

---

# 34. Structural coherence summary

For predicted components report:
\[
\operatorname{median}_{(i,j)\in E_S}C_{ij},
\]
\[
\operatorname{median}_{(i,j)\in E_D}C_{ij},
\]
and:
\[
\operatorname{median}_{(i,j)\in E_{S,D}}C_{ij}.
\]

Desired structural relation:
\[
\boxed{
C_{within-static},C_{within-dynamic}
\ll
C_{cross}
}
\]
when the partition is identifiable.

Do not tune per-dataset thresholds from this.

---

# 35. Evaluate the coherence of GT itself

Using GT only after prediction, calculate:
```text
GT static within-edge median C
GT dynamic within-edge median C
GT cross-edge median C
```

This is critical.

If GT dynamic Phase-1 trajectories are already incoherent, then the failure is upstream in Phase 1, not in the segmentation.

The report must explicitly distinguish these two cases.

---

# 36. Synthetic tests

Implement renderer-independent synthetic free-trajectory tests.

## S1 — static + translated rigid body
Expected clean binary recovery.

## S2 — static + revolute rigid body
Use an offset revolute axis.
Points near the axis have tiny displacement but must still remain dynamic through coherence connectivity.

## S3 — speed changes / stop / restart / reversal
Metric preservation must remain valid.

## S4 — renderer-like chaotic false dynamics
Apply independent temporally varying perturbations to a static region, even with amplitudes comparable to the true moving body.
It must not become a coherent dynamic component.

## S5 — smooth non-rigid shear
Low temporal jitter but finite metric strain.
Must fail rigid coherence.

## S6 — move and return to start
Must still be recovered because all frames are used.

## S7 — tiny-motion revolute-axis points
Must not become static merely because center displacement is small.

---

# 37. Verify differential identity numerically

For synthetic trajectories verify:
\[
\epsilon_{ij}(t)
\]
from finite log-length differences matches:
\[
\frac{
r_{ij}^T(v_j-v_i)
}{
\|r_{ij}\|^2
}.
\]

For exact rigid motion both must converge toward zero with decreasing \(\Delta t\).

---

# 38. Real datasets

Run the coherence-only branch at minimum on:
```text
USB_100109
storage_45135
blade_103706
stapler_103111
fridge_10905
```

Use one common configuration.

Do not tune graph/spectral parameters by scene.

---

# 39. Output tree

For every scene:
```text
output/motion_coherence_<scene>/
    config.json
    predicted_support.json

    numerical/
        edges.csv
        nodes.csv
        components.json
        edge_strain_timeseries.npz
        gt_metrics.json
        gt_mapping.json

    debug/
        finite_strain_heatmap.html
        differential_strain_heatmap.html
        representative_edges.html
        motion_coherence_support_3d.html

    evaluation/
        confusion_matrix.json
        current_vs_coherence.json
```

---

# 40. Predicted support artifact

Write:
```text
predicted_support.json
```
with:
```text
method = "trajectory_metric_coherence"
num_supernodes
num_gaussians
static_supernodes
dynamic_supernodes
static_gaussians
dynamic_gaussians
graph_config
component_stats
```

Do not include Otsu results.

---

# 41. Plug into Phase 2 only after freezing

When:
```text
--phase2-support-source coherence
```
is used, all:
```text
Phase2 screw
Phase2 P
Phase2 R
```
must receive exactly the same:
\[
\boxed{
\mathcal D_{coherence}
}
\]
Gaussian indices.

Do not let any joint fit feed back into this support in this task.

---

# 42. No support/joint iteration yet

Do not implement:
```text
coherence support
→ joint fit
→ support relabel
→ joint fit
```

First test whether trajectory coherence alone recovers GT.

---

# 43. Downstream Phase-2 A/B comparison

For evaluation only, run Phase 2 independently with:
```text
current support
coherence support
```

Compare:
```text
joint motion magnitude
render RGB/depth energy
geometry consistency
P/R fit status
held-out energy
```

Do not combine supports.

---

# 44. Required final tables

Static/dynamic GT table:

| scene | method | GT dyn % | Pred dyn % | Dyn Precision | Dyn Recall | Dyn F1 | Dyn IoU | Bal. Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| USB | current | | | | | | | |
| USB | coherence | | | | | | | |
| Storage | current | | | | | | | |
| Storage | coherence | | | | | | | |
| Blade | current | | | | | | | |
| Blade | coherence | | | | | | | |
| Stapler | current | | | | | | | |
| Stapler | coherence | | | | | | | |
| Fridge | current | | | | | | | |
| Fridge | coherence | | | | | | | |

Coherence table:

| scene | GT static within C | GT dynamic within C | GT cross C | predicted static within C | predicted dynamic within C | predicted cross C |
|---|---:|---:|---:|---:|---:|---:|
| ... | | | | | | |

---

# 45. Failure behavior

If the coherence partition is not identifiable:
- do not fall back to Otsu;
- do not force top 10%;
- do not force 50%;
- do not silently choose a partition.

Return:
```text
UNRESOLVED_COHERENCE_PARTITION
```

Still write all heatmaps, numerical outputs, and 3D debug.

---

# 46. Leakage regression tests

Add tests that fail if:
- GT is loaded before prediction is saved;
- Otsu selector is imported;
- displacement threshold seeds spectral labels;
- current-support membership initializes coherence labels;
- GT labels influence graph weights or Kabsch.

The coherence branch must stand on its own.

---

# 47. Required report

Create:
```text
MOTION_COHERENCE_STATIC_DYNAMIC_REPORT.md
```

Required sections:
1. exact Phase-1 trajectory source;
2. supernode graph construction;
3. finite metric-strain derivation;
4. differential symmetric-strain derivation;
5. edge coherence definition;
6. binary graph partition;
7. static/dynamic component identification;
8. whole-window Kabsch validation;
9. synthetic tests S1–S7;
10. GT mapping per dataset;
11. GT metrics;
12. current-vs-coherence comparison;
13. heatmap results;
14. 3D debug paths;
15. GT coherence analysis;
16. downstream Phase-2 comparison;
17. unresolved cases.

---

# 48. Implementation order

Do exactly:

```text
1. Add motion_coherence_support.py as an independent module.
2. Extract Phase-1 supernode trajectories.
3. Build the fixed canonical graph.
4. Compute l_ij(t), z_ij(t), C_ij.
5. Add epsilon_ij and J_ij diagnostics.
6. Implement coherence-only normalized spectral binary partition.
7. Fit whole-window rigid transforms per component.
8. Identify static versus dynamic after partition.
9. Freeze predicted Gaussian support.
10. Add strict GT evaluation with no leakage.
11. Build finite/differential strain heatmaps.
12. Build representative edge temporal plots.
13. Build 3D Phase-1 support debugger.
14. Run synthetic S1–S7.
15. Run USB, Storage, Blade, Stapler, Fridge.
16. Compare prediction against GT.
17. Run Phase 2 current-support vs coherence-support as separate experiments.
18. Write the final report.
```

---

# 49. Final methodological statement

This method intentionally does **not** ask:

\[
\text{Which points moved far?}
\]

It asks:

\[
\boxed{
\text{Which groups of Phase-1 free trajectories preserve their internal metric over the whole sequence?}
}
\]

The invariant is:
\[
\boxed{
\|x_i(t)-x_j(t)\|
=
\text{constant}
}
\]
for points on the same rigid body.

Equivalently:
\[
\boxed{
r_{ij}^T(v_j-v_i)=0.
}
\]

This is the only physical principle used for static/dynamic segmentation in this branch.

Do not mix this result with Otsu, displacement thresholds, current support, semantics, or joint type.

The purpose of the experiment is to test cleanly and independently whether **whole-window rigid trajectory coherence** is sufficient to recover the true static/dynamic partition visible in the Phase-1 motion debug.
