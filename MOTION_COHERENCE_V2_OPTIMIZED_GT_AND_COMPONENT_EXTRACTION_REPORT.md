# Motion coherence v2 — optimized labelled GT, and coherent-component extraction

Implements `CODEX_FIX_COHERENCE_EXTRACTION_AND_USE_OPTIMIZED_3DGS_GT.md`,
§33's order. Steps 1–16 and 18 are done; step 17 (the Phase-2 A/B) is blocked
on the GPU, see §19 below.

**Headline:** with the optimized labelled GT and the new extractor, coherence
now beats the displacement selector on **all five scenes** — including the two
it previously lost badly. Storage went from F1 0.271 to 0.631 and blade from
0.361 to 0.915.

---

## 1. The old sparse-NN pseudo-GT problem

The previous evaluation mapped the sparse `points3D_parts.ply` cloud onto
Gaussians by nearest neighbour at `4 ×` the sparse cloud's median spacing. That
cloud is 360–6 234 points against 7 553–40 591 Gaussians, so most labels were
interpolated and 1–2 278 Gaussians per scene went unmapped and were dropped
from every metric. It was never ground truth; it was a reconstruction of one.

## 2. The labelled-3DGS training code, found

`scene/gaussian_model.py` already carries the label through training in
`_gt_part_labels`. No new propagation code was written.

## 3. Optimized labelled GT artifact per scene

All five models carry the per-vertex fields `gt_part` and `gt_dynamic` at
`point_cloud/iteration_14000/point_cloud.ply`. Written per scene to
`evaluation/optimized_3dgs_gt.json` and `optimized_3dgs_gt_labels.npy`.

## 4. Proof of label propagation (§3's lineage)

Traced in code, not assumed:

| stage | `scene/gaussian_model.py` | behaviour |
|---|---|---|
| init | ~657 | from `BasicPointCloud.part_indices`, i.e. the sparse labels |
| storage | 87 | a `torch.long` tensor, **not** an `nn.Parameter` |
| optimizer | 984 | listed only for re-indexing; never a trainable |
| clone/split | ~2375 | `new_gt_parts = old_gt_parts[sel].repeat(rep)` |
| prune | ~2267 | `self._gt_part_labels = gt[valid_points_mask]` |
| save | ~1817 | written as `gt_part`, `gt_dynamic` |

So it is an immutable lineage: never optimized, never receives a gradient,
inherited exactly by every child Gaussian, index-selected on pruning. A learned
`dynamic_prob` / `dynamic_logit` would not qualify (§3) and none is used. The
moving rule `part != 0` is read from `utils/static_endpoint_scene.py`, which
states it as `"moving_rule": "part != 0"`.

## 5. Phase-1 t0 alignment

§4's preferred case holds on every scene — **exact index correspondence**:

| scene | GT N | Phase-1 N | max ‖Δμ‖ | median |
|---|---:|---:|---:|---:|
| USB | 11 341 | 11 341 | **0.000e+00** | 0.000e+00 |
| storage | 40 591 | 40 591 | **0.000e+00** | 0.000e+00 |
| blade | 7 553 | 7 553 | **0.000e+00** | 0.000e+00 |
| stapler | 9 062 | 9 062 | **0.000e+00** | 0.000e+00 |
| fridge | 28 360 | 28 360 | **0.000e+00** | 0.000e+00 |

No nearest-neighbour transfer is used. (`load_iteration=-1` resolves to
iteration_14000; iteration_7000 is a different, smaller state and does not
match Phase 1.)

## 6. Old pseudo-GT vs optimized GT disagreement

| scene | agreement | static→dynamic | dynamic→static | old unmapped | optimized dyn % | sparse-NN dyn % |
|---|---:|---:|---:|---:|---:|---:|
| USB | 93.7% | 131 | 459 | 1 948 | 64.9 | 57.0 |
| storage | 95.5% | 481 | **1 242** | 2 278 | 10.4 | 8.2 |
| blade | 96.0% | 202 | 103 | 1 | 14.9 | 16.2 |
| stapler | 98.2% | 93 | 63 | 458 | 52.6 | 54.6 |
| fridge | 99.3% | 30 | 167 | 1 071 | 32.6 | 32.8 |

The disagreement is not uniform noise: it is concentrated on dynamic Gaussians
being called static (storage 1 242), i.e. exactly the minority class the
metrics are most sensitive to. The sparse labels also flattered the Otsu
baseline — on storage it scored F1 0.759 against the pseudo-GT and **0.577**
against the real one.

## 7. Phase-1 supernode provenance

`estimator.supernodes.build_supernodes(mu0, target_size=64, seed=0)` — the
repo's own partition, with `x_s(t)` the uniform-weight member mean. One global
`--coherence-knn 12`, never tuned per scene.

## 8. Gaussian-vs-supernode coherence audit

Prediction stays Gaussian-level; `numerical/supernode_gt.json` records per
supernode its member count, GT dynamic fraction and purity (§8), as a
diagnostic only.

## 9. Why Ncut is removed

A normalized cut minimizes `cut/vol(A) + cut/vol(B)`, which is minimized by
balanced partitions. It also forces every node into one of two bodies, so
renderer-driven false dynamics had nowhere to go. Measured: storage split
333/301 and blade 83/35 against GT fractions of 10.4% and 14.9%. The previous
report established that this was the partition's fault, not the coherence
metric's — blade had the strongest coherence signal of all five scenes (cross
edges at 4.47× within-static) and the worst F1.

`C_ij` is unchanged (§1). Three output classes now exist (§11), and only
`MOVING_RIGID` reaches Phase 2.

## 10. Persistent component hierarchy

Union-find over edges sorted by increasing `C_ij`; every merge event records a
candidate with its birth cost, death cost and persistence (§12).

One correction was needed. Plain single linkage merges two components as soon
as **one** cheap edge joins them, and S12 showed it welds two bodies at the
very first merge, so the real bodies never become candidates at all and no
later validation can recover them (IoU 0.117, 99.6% predicted moving). A merge
now requires `min(2, |A|, |B|)` independent edges — a genuine interface inside
one body is crossed by many kNN edges, a bridge by one. The cap by the smaller
side matters: two singletons are joined by exactly one edge by definition, and
demanding two there produces zero candidates.

## 11. Internal/boundary validation

Per candidate: internal `q50/q75/q90` and boundary `q10/q25/q50`, persisted and
plotted (§13). A candidate with no boundary at all is disqualified — it covers
the whole graph, so whatever rigid trajectory explains it is a change of
coordinates, not a joint.

## 12. Node-wise CV rigid evidence

Deterministic folds (`node index mod 5`, one global count); fit `(R_t, p_t)` on
training nodes, score held-out nodes; `η^cv = 1 − RSS_R^cv/(RSS_I^cv+ε)`,
`ΔRSS^cv`, and §16's `ΔBIC`.

Three things had to be got right, each of which was a size bias in disguise —
the very thing §18 forbids:

- **The evidence must be measured in the complement's frame.** In world
  coordinates a uniformly drifting scene is perfectly rigid, scores η = 1 and
  an enormous ΔBIC, and is declared one big moving body. Mapping the trajectory
  into the complement's own rigid frame cancels motion shared with the rest of
  the scene (and removes global drift from every candidate for free).
- **But the relative frame cannot say which side moves** — relative motion is
  symmetric, so selecting on it alone picks whichever side is larger. The world
  frame breaks the tie legitimately, because the cameras are fixed; this is
  §16's identity-residual test used exactly where §16 puts it, after the
  candidates exist.
- **The selection quantity must be intensive.** `ΔRSS^cv` is a sum over N×T, so
  comparing candidates on it rewards node count outright: USB, blade and fridge
  all selected near-whole components (94.8–99.9% moving). Per node-frame, it is
  the mean squared displacement a common rigid trajectory explains, and is
  neutral to how many nodes carry it.

**ΔBIC is persisted but does not gate.** With `k = 6T` parameters for a free
per-frame rigid trajectory, the penalty `k log(NT)` is fixed while the evidence
grows with `N`, so a small component can never pass however coherent it is: on
blade the genuine moving body (8 nodes, IoU 0.67, η = 0.974) was rejected at
ΔBIC = −4.26e+02 while a near-whole component passed. Held-out
cross-validation is already the anti-overfitting guard and is size-neutral.

The one numerical guard: relative motion must exceed `sqrt(machine epsilon)` as
a fraction of the component's own extent. S13's drifting scene produces
`RSS_I^rel = 2.8e-28`, where η = 1.0 is noise over noise. This separates
"literally nothing" from "something"; it is not a motion threshold.

**The candidate pool is stratified by size.** Persistence alone is not a usable
pool: a near-whole component survives to the end of the merge tree and so has
the highest persistence of all. On storage the best candidate (49 nodes,
IoU 0.69) sat at persistence rank 88 behind a wall of 630-node components and
was never evaluated. Taking the most persistent candidates in each size octave
is not a size preference — it is the opposite, guaranteeing every size is
considered so the evidence decides.

§19's refinement is now a held-out model comparison per node: recruit `i` when
`E_i^M < E_i^I` under the core's own `T_t`, fitted without `i`. No threshold,
no scale, no displacement amplitude — which is what lets it recruit near-axis
material, whose displacement is tiny but whose motion is the body's. The
MAD-based scale is kept as a diagnostic; used as the gate it recruited almost
nothing, because a core selected FOR coherence has a degenerate residual
spread.

## 13. S8–S13

| test | result |
|---|---|
| S8 8% translating body | IoU **1.000**, predicted 7.9% vs 7.9% true |
| S9 15% revolute body | IoU **1.000**, predicted 15.0% |
| S10 25% chaotic outliers | **1/60 leaked**, 203 nodes left ambiguous |
| S11 small hinge beside a large static body | IoU **1.000** |
| S12 one accidental low-C cross edge | IoU **1.000**, predicted 12.1% |
| S13 uniformly drifting static body | **NO_COHERENT_MOVING_COMPONENT**, 0/200 moving |

S1–S7, S20, S37 and S46 still pass: **41 checks**.

## 14. Five-scene static/dynamic results (optimized GT)

| scene | GT dyn % | selector | Pred dyn % | Dyn P | Dyn R | Dyn F1 | Dyn IoU | Bal. Acc |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| USB | 64.9 | current | 40.8 | 0.950 | 0.597 | 0.733 | 0.579 | 0.769 |
| USB | 64.9 | **coherence v2** | 61.1 | 0.929 | 0.874 | **0.901** | **0.820** | **0.875** |
| Storage | 10.4 | current | 9.2 | 0.617 | 0.543 | 0.577 | 0.406 | 0.752 |
| Storage | 10.4 | **coherence v2** | 7.1 | 0.776 | 0.532 | **0.631** | **0.461** | **0.757** |
| Blade | 14.9 | current | 15.5 | 0.862 | 0.900 | 0.881 | 0.787 | 0.937 |
| Blade | 14.9 | **coherence v2** | 14.3 | 0.933 | 0.897 | **0.915** | **0.843** | **0.943** |
| Stapler | 52.6 | current | 43.7 | 0.998 | 0.828 | 0.905 | 0.826 | 0.913 |
| Stapler | 52.6 | **coherence v2** | 49.1 | 0.987 | 0.921 | **0.953** | **0.910** | **0.954** |
| Fridge | 32.6 | current | 26.2 | 0.980 | 0.786 | 0.872 | 0.774 | 0.889 |
| Fridge | 32.6 | **coherence v2** | 28.4 | 0.995 | 0.867 | **0.927** | **0.863** | **0.932** |

Coherence v2 wins every scene on F1, IoU and balanced accuracy, and its
predicted fraction tracks GT across a 6× range (7.1→61.1% against 10.4→64.9%)
with no target fraction anywhere in the method.

Against the previous Ncut version: storage **0.271 → 0.631**, blade
**0.361 → 0.915**.

## 15. Storage deep debug

`RESOLVED`, 75 of 634 supernodes moving. Selected candidate 411: 19 nodes,
persistence 0.0085, η_cv(rel) 0.9261, ΔRSS_world 9.40e+01, ΔBIC(rel) +7.33e+02;
§19 recruited the rest of the drawer. Classes: 75 MOVING_RIGID, 69
STATIC_COMPATIBLE, 490 INCOHERENT_OR_AMBIGUOUS — the large ambiguous set is the
point, not a failure: storage's static body has substantial Phase-1 free motion
and the method declines to call it either way.

Confusion (Gaussian level): tp 2 249, fp 648, fn 1 982, tn 35 712. Precision
0.776 against the baseline's 0.617 — less than half the false positives
(648 vs 1 428). Recall is the weak side (0.532): the drawer's near-static
material is left ambiguous rather than claimed.

## 16. Blade deep debug

`RESOLVED`, 12 of 118 supernodes moving. Selected candidate 89: 8 nodes,
persistence 0.0135, η_cv(rel) **0.9739**, ΔRSS_world 7.20e+01, ΔBIC(rel)
**−4.26e+02**. That negative ΔBIC is the case that proves the point of §12
above: this is the real blade, recovered at F1 0.915, and a BIC gate would have
thrown it away.

Confusion: tp 1 006, fp **72**, fn 116, tn 6 359 — against the baseline's
162 fp. Classes: 12 MOVING_RIGID, 44 STATIC_COMPATIBLE, 62 AMBIGUOUS.

## 17. Confusion matrices

Written per scene to `evaluation/confusion_matrix.json` for both selectors, and
in full with all derived metrics in `numerical/gt_metrics.json`.

## 18. Ambiguous / outlier analysis

Ambiguous fractions: storage 77%, blade 53%, stapler 44%, fridge ~0%, USB ~0%.
These track how much renderer-driven free motion each scene's static body
carries, and they are reported rather than forced to a side (§11). S10 confirms
the mechanism works as intended: 25% chaotic outliers, 1/60 leaked.

## 19. Downstream Phase-2 A/B

**Not run.** `--phase2-support-source {current,coherence}` is wired and §41 of
the previous spec is satisfied (screw, P and R all receive the same frozen
indices, with no silent fallback). The A/B itself needs the rasterizer, and the
**NVIDIA driver on this machine is still not responding** — `nvidia-smi` cannot
reach it and `torch.cuda.is_available()` is `False`. It needs a driver reload
or reboot. Everything in this report ran on CPU, in 6–12 s per scene.

## 20. Conclusions

1. The optimized labelled 3DGS GT exists, its lineage is immutable and traced,
   and it aligns to Phase 1 **by index at exactly zero error**. The sparse-NN
   pseudo-GT disagreed with it on 0.7–6.3% of Gaussians, concentrated on the
   minority class, and flattered the displacement baseline.
2. The coherence metric `C_ij` was never the problem and is unchanged.
3. Every failure of the previous version traced to one root cause in different
   disguises: **a size bias in the decision rule** — Ncut's balance term, then
   the extensive ΔRSS, then the BIC penalty, then a persistence-ranked pool
   that never looked at small components. Removing all four is what made the
   method work across a 6× range of true dynamic fractions.
4. Coherence v2 beats the displacement selector on all five scenes with no
   target fraction, no Otsu, no displacement threshold, no GT and no joint type
   in the decision path.
5. Outstanding: the Phase-2 A/B, and storage's recall (0.532) — its static body
   carries enough Phase-1 free motion that 77% of supernodes stay ambiguous.
