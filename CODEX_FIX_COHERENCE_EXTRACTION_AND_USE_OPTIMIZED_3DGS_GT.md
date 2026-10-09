# CODEX — FIX COHERENCE SUPPORT EXTRACTION + USE OPTIMIZED LABELED 3DGS AS GT

## Objective

Continue from `MOTION_COHERENCE_STATIC_DYNAMIC_REPORT.md`.

Make two independent corrections:

1. **Replace the approximate sparse-PC→Gaussian NN evaluation GT with the optimized labeled 3DGS GT that already exists in this project.**
2. **Keep the current trajectory-coherence metric \(C_{ij}\) unchanged, but replace the two-way normalized cut with an outlier-aware coherent-moving-component extractor that has no balance prior and does not force every node into one of two bodies.**

Then rerun the five real datasets and, once CUDA works, execute the missing downstream Phase-2 A/B:

```text
current support      → Phase2 Screw / P / R
coherence-v2 support → Phase2 Screw / P / R
```

Never combine the two supports.

---

# 1. Keep the existing coherence measurement exactly as-is

Do not redesign \(C_{ij}\).

Keep:

\[
\ell_{ij}(t)=\|x_j(t)-x_i(t)\|
\]

\[
\bar\ell_{ij}=\operatorname{median}_t\ell_{ij}(t)
\]

\[
z_{ij}(t)
=
\log
\frac{\ell_{ij}(t)+\epsilon_\ell}
{\bar\ell_{ij}+\epsilon_\ell}
\]

and:

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

Keep differential strain and jitter as diagnostics only.

Do not reintroduce:
- Otsu;
- median displacement;
- top-k displacement;
- target dynamic fraction;
- GT priors;
- P/R information;
- semantic features.

The current evidence already says the useful quantity is the coherence metric; the main failure is how the graph is being partitioned.

---

# 2. Find the old optimized labeled 3DGS GT before changing evaluation

We previously implemented a modified 3DGS training path that propagated the original sparse static/dynamic / part label into the **optimized Gaussian model**.

Find that code and those optimized labeled PLYs before writing any new GT propagation.

Search repository, git history and output trees.

Start with:

```bash
git grep -n -E \
'points3D_parts|part_label|part_id|gt_part|gt_label|dynamic_gt|static_dynamic|dynamic_foreground_mask|dynamic_logit|dynamic_prob|save_ply|load_ply|densif|clone|split|prun'
```

Inspect at least:

```text
train.py
scene/gaussian_model.py
utils/dynamic_loss_utils.py
utils/static_endpoint_scene.py
```

and any historical/alternate 3DGS training scripts.

Also search artifacts such as:

```text
output/.../point_cloud/iteration_*/point_cloud.ply
gaussian_tracks.pt
```

Historical names such as:

```text
dynamic_foreground_mask
dynamic_prob
dynamic_logit
```

are search hints only.

---

# 3. Prove label provenance

The GT field must be an **immutable lineage from the labeled sparse input**, propagated through:

```text
sparse points3D_parts.ply
    ↓
Gaussian initialization
    ↓
densification / clone / split
    ↓
pruning
    ↓
optimized point_cloud.ply
```

A learned field such as:

```text
dynamic_prob
dynamic_logit
```

is NOT automatically GT.

For every candidate field determine:
- where it is initialized;
- whether it is optimized or frozen;
- how clone/split inherits it;
- how pruning removes it;
- how it is serialized.

The desired field may be named `part`, `part_id`, `gt_part`, `gt_dynamic`, `static_dynamic_gt`, `dynamic_foreground_mask`, etc., but do not guess. Trace the code.

---

# 4. Identify the labeled optimized t0 model corresponding to Phase 1

For every scene:

```text
USB_100109
storage_45135
blade_103706
stapler_103111
fridge_10905
```

identify the exact static \(t=0\) 3DGS model used to generate:

```text
phase1/centers/centers_frame_0000.npy
```

and find its labeled optimized equivalent.

Preferred case:

\[
N_{GT}=N_{Phase1}
\]

and the Gaussian ordering is identical.

Verify:

\[
\max_i\|\mu_i^{GT}-\mu_i^{Phase1}(0)\|
\]

and median alignment error.

If alignment is at serialization tolerance, use **direct index labels**. No nearest-neighbour label transfer is needed.

If ordering differs but the point set is the same, establish one-to-one mutual-NN correspondence under numerical tolerance and report unmatched/duplicates.

Do **not** use the previous coarse `4 × sparse-cloud median spacing` NN mapping as the primary GT.

---

# 5. If the optimized labeled GT cannot be found

Do not silently fall back and call sparse-NN labels “GT”.

Return:

```text
OPTIMIZED_LABELED_3DGS_GT_NOT_FOUND
```

and report:
- searched files;
- candidate PLYs;
- candidate fields;
- why each was rejected;
- exact regeneration command if the historical training code still exists.

---

# 6. Primary GT artifacts

For each scene create:

```text
evaluation/optimized_3dgs_gt.json
evaluation/optimized_3dgs_gt_labels.npy
```

with:

```text
scene
source_sparse_labeled_pcd
optimized_labeled_3dgs_path
iteration
label_field
label_provenance
num_gaussians
num_static
num_dynamic
num_other
index_aligned_with_phase1_t0
max_position_alignment_error
median_position_alignment_error
mapping_method
```

Primary metrics must be computed directly at **Gaussian level**.

---

# 7. Quantify the old sparse-NN pseudo-GT error

Keep the old sparse-PC NN labels only as a diagnostic.

Compare:

```text
old sparse-PC NN pseudo-GT
vs
optimized labeled 3DGS GT
```

and save:

```text
agreement fraction
static→dynamic disagreement
dynamic→static disagreement
old unmapped count
3D disagreement visualization
```

The optimized labeled 3DGS GT becomes the primary evaluation reference.

---

# 8. Keep GT Gaussian-level; use supernode GT only as a diagnostic

For every supernode \(s\), using the optimized Gaussian GT, compute:

\[
p_s
=
\frac{N^{GT}_{dynamic}(s)}{N^{GT}_{total}(s)}.
\]

Save:

```text
supernode_id
num_members
gt_dynamic_fraction
gt_static_fraction
gt_purity = max(p_s,1-p_s)
```

Do not use this during prediction.

---

# 9. Reuse the exact Phase-1 supernode partition if available

The current coherence runner rebuilds supernodes from \(t=0\).

Search Phase-1 state/artifacts for the actual:

```text
SupernodePartition
supernode_id
partition hash
```

used by Phase 1.

If it exists, reuse it exactly and record:

```text
phase1_supernode_source
partition_hash
num_supernodes
target_size
```

Only rebuild if no original partition exists.

---

# 10. Audit Gaussian vs supernode coherence

On at least:

```text
storage_45135
blade_103706
fridge_10905
```

compare:
1. Gaussian-level local-edge \(C_{ij}\);
2. current supernode-centroid \(C_{ij}\);
3. exact original-Phase1-supernode \(C_{ij}\), if different.

Using optimized 3DGS GT, report:

\[
C_{within-static},\quad C_{within-dynamic},\quad C_{cross}.
\]

This tests whether centroid averaging is cancelling the chaotic motion we want to detect.

Do not change \(C_{ij}\) itself.

---

# 11. Remove two-way Ncut from the production coherence branch

Do not do:

```text
Fiedler vector
→ balanced two-way Ncut
→ every node assigned to component 0 or 1
```

The corrected method must support:

```text
MOVING_RIGID
STATIC_COMPATIBLE
INCOHERENT_OR_AMBIGUOUS
```

Only `MOVING_RIGID` is passed to Phase 2.

False dynamic trajectories are allowed to stay `INCOHERENT_OR_AMBIGUOUS`.

---

# 12. Build a persistent coherent-component hierarchy directly from C_ij

Keep the canonical graph \(G=(V,E)\).

Sort edges by increasing \(C_{ij}\).

Use union-find to progressively add edges.

Each merge event defines a candidate connected component.

For each candidate record:

```text
candidate_id
members
num_nodes
num_internal_edges
birth_edge_cost
death_merge_edge_cost
```

This is a graph-constrained single-linkage hierarchy driven only by \(C_{ij}\).

There is:
- no target fraction;
- no balance term;
- no GT;
- no Otsu;
- no displacement threshold.

---

# 13. Validate the full internal and boundary edge distributions

Do not trust a component merely because one low-\(C\) path connects it.

For every candidate \(\mathcal C\), compute:

\[
C_{in}=\{C_{ij}: i,j\in\mathcal C\}
\]

and:

\[
C_{boundary}=\{C_{ij}: i\in\mathcal C,j\notin\mathcal C\}.
\]

Persist:

```text
internal q50/q75/q90
boundary q10/q25/q50
```

This prevents a single accidental low-\(C\) cross edge from silently chaining two bodies together.

---

# 14. Whole-window rigid validation of every candidate

For every non-degenerate candidate fit, at every frame:

\[
(R_t,p_t)
=
\arg\min_{R_t\in SO(3),p_t}
\sum_{i\in\mathcal C}
\|x_i(t)-(R_t x_i(0)+p_t)\|^2.
\]

This remains joint-type agnostic.

Compute:

\[
RSS_R(\mathcal C)
=
\sum_{t,i\in\mathcal C}
\|x_i(t)-(R_t x_i(0)+p_t)\|^2
\]

and:

\[
RSS_I(\mathcal C)
=
\sum_{t,i\in\mathcal C}
\|x_i(t)-x_i(0)\|^2.
\]

---

# 15. Use node-wise cross-validation

Do not select a component by in-sample Kabsch residual.

Use deterministic node-wise folds.

For each candidate:
- fit \(R_t,p_t\) on training nodes;
- evaluate held-out nodes;
- aggregate all folds.

Save:

\[
RSS_R^{cv},\qquad RSS_I^{cv}.
\]

Define:

\[
\boxed{
\eta_{\mathcal C}^{cv}
=
1-
\frac{RSS_R^{cv}}{RSS_I^{cv}+\epsilon}.
}
\]

True coherent body:

\[
\eta^{cv}\rightarrow1.
\]

Chaotic renderer-driven cloud:

\[
\eta^{cv}\approx0.
\]

Also save:

\[
\boxed{
\Delta RSS_{\mathcal C}^{cv}
=
RSS_I^{cv}-RSS_R^{cv}.
}
\]

Use one global fold count for all scenes.

---

# 16. Add complexity-aware identity-vs-rigid evidence

Also compute a BIC-style diagnostic:

\[
BIC_I=N\log(RSS_I/N)
\]

\[
BIC_R=N\log(RSS_R/N)+k\log N
\]

and:

\[
\boxed{\Delta BIC=BIC_I-BIC_R.}
\]

Persist this, but keep raw residuals visible.

The purpose is to prevent the current behavior where the component with the slightly larger \(E_I-E_R\) is called dynamic even when both components are mostly non-rigid.

---

# 17. Candidate selection: sequential evidence, not score soup

Keep the roles separate:

```text
C_ij
→ generates coherent candidate components

held-out whole-window rigid prediction
→ decides whether a candidate is a physical moving body
```

Do not build a weighted sum such as:

```text
0.3 persistence + 0.4 eta + 0.3 motion
```

Expose the separate evidence for every candidate.

Select the candidate with strongest held-out non-identity rigid evidence among graph-persistent candidates.

If no candidate is convincingly explained by a common non-identity rigid trajectory, return:

```text
NO_COHERENT_MOVING_COMPONENT
```

Do not fabricate a half-scene dynamic component.

---

# 18. Moving support size is an outcome

The selected component may contain:

```text
5%
8%
15%
30%
60%
```

of the scene.

Do not reward 50/50.
Do not penalize small components.
Do not use GT fraction.
Do not use current/Otsu fraction.

---

# 19. Optional refinement after a moving core is found

After selecting a robust moving core, fit its whole-window rigid trajectory \(T_t\).

For every neighboring/unassigned node compute:

\[
E_i^M
=
\sum_t\|x_i(t)-T_t x_i(0)\|^2.
\]

This may recruit near-axis / low-displacement points if they are consistent with the same rigid body.

If implemented:
- fit \(T_t\) from the core only;
- estimate residual scale from core residuals;
- use one global statistical rule;
- leave uncertain nodes ambiguous;
- never use displacement amplitude.

Do not refit a separate transform for each candidate node.

---

# 20. Synthetic regressions

Keep S1–S7.

Add:

## S8 — 8% translating moving body
Must recover the small moving component without a near-balanced split.

## S9 — 15% revolute moving body
Same.

## S10 — 25% chaotic renderer-driven outliers
They must remain mostly `INCOHERENT_OR_AMBIGUOUS`, not be forced into moving.

## S11 — small hinge adjacent to a large static body
Cross-body kNN edges exist; moving body must still be recovered.

## S12 — one accidental low-C cross edge
Full internal-edge validation must prevent chaining.

## S13 — static body with tiny coherent global drift
Must not become the moving body merely because Kabsch gives a tiny improvement.

No scene-specific parameters.

---

# 21. New candidate-tree debug

Create:

```text
debug/coherence_component_tree.html
```

For every candidate show:

```text
num_nodes
birth C
death C
internal C q50/q75/q90
boundary C q10/q25/q50
RSS_I_cv
RSS_R_cv
eta_cv
delta_RSS_cv
delta_BIC
selected/not selected
```

Plot:

```text
candidate size vs eta_cv
candidate size vs delta_RSS_cv
candidate size vs persistence
internal vs boundary coherence
```

Storage and Blade must be understandable from this page.

---

# 22. Update the 3D debugger

`motion_coherence_support_3d.html` must show:

```text
MOVING_RIGID
STATIC_COMPATIBLE
INCOHERENT_OR_AMBIGUOUS

optimized 3DGS GT static
optimized 3DGS GT dynamic

TP dynamic
FP dynamic
FN dynamic
TN static
```

Animate the actual Phase-1 free trajectories.

Also add toggles for:

```text
selected moving core
candidate component
low-C edges
high-C edges
moving-core boundary edges
mixed-GT supernodes
```

---

# 23. Re-evaluate all five scenes with optimized GT

Run:

```text
USB_100109
storage_45135
blade_103706
stapler_103111
fridge_10905
```

with one common configuration.

Primary table:

| scene | selector | GT dyn % | Pred dyn % | Dyn P | Dyn R | Dyn F1 | Dyn IoU | Bal Acc | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| USB | current | | | | | | | | |
| USB | coherence-v2 | | | | | | | | |
| Storage | current | | | | | | | | |
| Storage | coherence-v2 | | | | | | | | |
| Blade | current | | | | | | | | |
| Blade | coherence-v2 | | | | | | | | |
| Stapler | current | | | | | | | | |
| Stapler | coherence-v2 | | | | | | | | |
| Fridge | current | | | | | | | | |
| Fridge | coherence-v2 | | | | | | | | |

No hybrid selector.

---

# 24. Recompute GT coherence under the optimized labels

At Gaussian and supernode level report:

\[
C_{within-static},\quad C_{within-dynamic},\quad C_{cross}
\]

plus edge counts.

The optimized Gaussian GT should remove the biggest weakness of the previous sparse-NN evaluation.

---

# 25. Deep-debug Storage and Blade

The previous Ncut produced approximately:

```text
Storage: 46% dynamic
Blade:   71% dynamic
```

The corrected method must explain why those near-balanced components are rejected.

Save:
- old Ncut partition;
- new moving core;
- optimized GT;
- candidate hierarchy;
- held-out Kabsch evidence;
- ambiguous/outlier trajectories.

Do not merely report final F1.

---

# 26. No GT leakage

Optimized GT is evaluation only.

Prediction must finish and be written before GT is loaded.

Add/keep tests ensuring:
- no GT imports in coherence predictor;
- no optimized label field in candidate selection;
- no GT dynamic fraction;
- no supernode GT purity in graph construction;
- no current/Otsu support initialization.

---

# 27. Output structure

For every scene:

```text
output/motion_coherence_v2_<scene>/
    config.json
    predicted_support.json

    numerical/
        edges.csv
        nodes.csv
        candidate_components.csv
        selected_component.json
        supernode_partition_provenance.json

    evaluation/
        optimized_3dgs_gt.json
        optimized_3dgs_gt_labels.npy
        old_nn_gt_vs_optimized_gt.json
        gt_metrics.json
        gt_coherence_gaussian.json
        gt_coherence_supernode.json
        supernode_gt_purity.json

    debug/
        coherence_component_tree.html
        finite_strain_heatmap.html
        differential_strain_heatmap.html
        representative_edges.html
        motion_coherence_support_3d.html
```

---

# 28. predicted_support.json

Required:

```text
method = trajectory_metric_coherence_persistent_rigid_component
status
num_gaussians
num_supernodes

moving_rigid_supernodes
moving_rigid_gaussians

static_compatible_supernodes
ambiguous_supernodes

selected_candidate_id
selected_candidate:
    num_nodes
    internal_C_quantiles
    boundary_C_quantiles
    RSS_identity_cv
    RSS_rigid_cv
    eta_rigid_cv
    delta_RSS_cv
    delta_BIC

supernode_partition_provenance
```

No GT fields inside the prediction artifact.

---

# 29. Run the missing downstream Phase-2 A/B once CUDA works

First check:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is unavailable, report the block honestly.

If available, run:

```text
A. current support      → Screw / P / R
B. coherence-v2 support → Screw / P / R
```

Within each branch all three models receive exactly the same frozen moving Gaussian indices.

Do not mix supports.

---

# 30. Required Phase-2 outputs

For every:

```text
scene × support source × {Screw,P,R}
```

save:
- optimized joint geometry;
- \(q(t)\);
- \(\dot q(t)\);
- accepted/rejected IEKF steps;
- RGB energy;
- depth energy;
- total \(R_{eff}\) energy;
- held-out energy if available;
- covariance/information;
- fixed-camera renders for all frames;
- 3D animated centers;
- Gaussian orientations.

Remember:

For Screw and R:

\[
R_i(t)=R_{joint}(t)R_i^0
\]

and:

\[
\Sigma_i(t)=R_{joint}(t)\Sigma_i^0R_{joint}(t)^T.
\]

For P:

\[
R_i(t)=R_i^0.
\]

---

# 31. Final Phase-2 table

Create:

| scene | support | dyn % | Screw energy | P energy | R energy | best model | joint motion magnitude | geometry status |
|---|---|---:|---:|---:|---:|---|---:|---|
| ... | current | | | | | | | |
| ... | coherence-v2 | | | | | | | |

Compare against known dataset joint type only after fitting; never use it during optimization.

---

# 32. Final report

Create:

```text
MOTION_COHERENCE_V2_OPTIMIZED_GT_AND_COMPONENT_EXTRACTION_REPORT.md
```

Required sections:

1. old sparse-NN pseudo-GT problem;
2. historical labeled-3DGS training code found;
3. optimized labeled GT artifact per scene;
4. proof of label propagation through densification/pruning;
5. Phase1-t0 alignment;
6. old pseudo-GT vs optimized-GT disagreement;
7. exact Phase1 supernode provenance;
8. Gaussian-vs-supernode coherence audit;
9. why Ncut is removed;
10. persistent component hierarchy;
11. full internal/boundary validation;
12. node-wise CV rigid evidence;
13. S8–S13;
14. five-scene S/D results;
15. Storage deep debug;
16. Blade deep debug;
17. optimized-GT confusion matrices;
18. ambiguous/outlier analysis;
19. downstream current-vs-coherence-v2 Phase2 Screw/P/R;
20. final conclusions.

---

# 33. Implementation order

Do exactly:

```text
1. Find historical optimized labeled 3DGS training implementation.
2. Find optimized labeled t0 PLY for every real scene.
3. Prove label provenance through initialization/densification/pruning/save.
4. Align labels exactly to Phase1 Gaussian indices.
5. Quantify old sparse-NN pseudo-GT disagreement.
6. Find/reuse exact Phase1 supernode partition.
7. Audit Gaussian-level vs supernode-level C_ij.
8. Keep C_ij unchanged.
9. Remove production Ncut.
10. Build C_ij persistent component hierarchy.
11. Add full internal/boundary edge validation.
12. Add node-wise CV Kabsch rigid evidence.
13. Select one physically supported MOVING_RIGID component.
14. Leave unrelated chaotic trajectories ambiguous/outlier.
15. Run S1–S13.
16. Run all five real scenes.
17. Evaluate with optimized labeled 3DGS GT.
18. Deep-debug Storage and Blade.
19. Check CUDA.
20. If CUDA works, run current-vs-coherence-v2 Phase2 Screw/P/R A/B.
21. Write final report.
```

---

# 34. Final principle

Do **not** change the physical invariant:

\[
\boxed{\|x_i(t)-x_j(t)\|=\text{constant}}
\]

for points on the same rigid body.

Change only how that evidence becomes a moving support.

Old:

\[
C_{ij}
\rightarrow
\text{balanced 2-way Ncut}
\rightarrow
\text{force every node into static/dynamic}.
\]

New:

\[
\boxed{
C_{ij}
\rightarrow
\text{persistent coherent graph candidates}
\rightarrow
\text{held-out whole-window rigid validation}
\rightarrow
\text{one physically supported moving rigid component}
}
\]

with renderer-driven chaotic trajectories allowed to remain outside the moving body.

And evaluate against the **already existing optimized labeled 3DGS**, not against a new sparse-PC nearest-neighbour approximation.
