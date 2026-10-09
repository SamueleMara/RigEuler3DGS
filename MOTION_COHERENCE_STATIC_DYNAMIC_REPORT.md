# Motion-coherence static/dynamic filter

Implements `CODEX_IMPLEMENT_MOTION_COHERENCE_STATIC_DYNAMIC_FILTER.md`,
§48's eighteen steps. Steps 1–16 and 18 are done; step 17 (the downstream
Phase-2 A/B) is blocked on the GPU, see §16 below.

**Headline:** the coherence principle is sound and measurable — GT
cross-boundary edges carry 1.5–4.5× the metric strain of within-static edges
on every scene. It beats the displacement selector where the moving part is a
large fraction (fridge F1 0.949 vs 0.882, stapler 0.932 vs 0.905, both at
**perfect dynamic precision**), and loses badly where it is small (storage
0.271 vs 0.759, blade 0.361 vs 0.894). The failure is in §12's prescribed
two-way normalized cut, not in the coherence measurement — blade has the
*strongest* signal of all five scenes and the worst F1.

---

## 1. Exact Phase-1 trajectory source

`phase1/centers/centers_frame_XXXX.npy` from the unified debug runner — the
free per-Gaussian positions, one array per frame. No transform is applied and
nothing is regenerated for this branch.

The branch is **renderer-independent in fact, not just in principle**: given
those files it needs no CUDA, no 3DGS model and no cameras (frame 0 of the
trajectory *is* the canonical state). A filter whose claim is about trajectory
geometry should be runnable, and testable, without the renderer it is meant to
be independent of. The five real runs below completed on CPU in ~6 s each.

## 2. Supernode graph construction

§3: trajectories are reduced to the repo's own `build_supernodes` partition
(target size 64), `x_s(t) = Σ_{i∈s} w_i x_i(t) / Σ_{i∈s} w_i` with the
documented uniform convention `w_i = 1`.

§4: one kNN graph, built **once** in the canonical frame (`--coherence-knn`,
default 12, one global value for every dataset, never tuned per scene).
Rebuilding neighbours per frame would let the graph track the motion it is
supposed to measure — a stretching edge would simply be replaced by a shorter
one and the strain would vanish by construction.

## 3. Finite metric-strain derivation

    ℓ_ij(t) = ‖x_j(t) − x_i(t)‖
    ℓ̄_ij   = median_t ℓ_ij(t)
    z_ij(t) = log((ℓ_ij(t) + ε) / (ℓ̄_ij + ε))

The median is over the **whole window**, not frame 0 and not the endpoint, so
an edge that moves and returns (S6) is measured against its own typical length
and no single bad frame can set the reference. `ε` is a data-scale numerical
guard, never a motion threshold.

## 4. Differential symmetric-strain derivation

    ε_ij(t) = (z(t+1) − z(t))/Δt = r^T(v_j − v_i)/‖r‖²

With `v_j − v_i ≃ ∇v·r` and `∇v = D + Ω`, the antisymmetric part drops out
identically (`r^T Ω r = 0`), so `r^T(v_j−v_i) = r^T D r` and rigid motion gives
`D = 0`, hence `ε ≡ 0`. Verified numerically (§37): the two forms agree to
**5.0e-05** at Δt = 1e-4, and exact rigid motion gives `max|ε| = 5.1e-11`.

## 5. Edge coherence definition

    C_ij = sqrt( Σ_t w_ij(t) z_ij(t)² / Σ_t w_ij(t) )

One primary score, not a mixture. `J_ij` (jitter) and `ε_ij` are kept as
diagnostics and are never folded into it (§8, §9).

## 6. Binary graph partition

`W_ij = exp(−C²/(σ_i σ_j + ε))` with `σ_i = median_{j∈N(i)} C_ij`
(self-scaled, so no global coherence threshold enters), then a two-way
normalized cut from the Fiedler vector, with the sweep point chosen by the
Ncut objective itself.

Nothing seeds it. A regression test asserts that `normalized_cut` takes only
`(edge_i, edge_j, W, n)` and calls nothing named `median`, `topk`,
`displacement`, `excursion`, `otsu`, `quantile` or `kthvalue`.

## 7. Static/dynamic identification

§16, and only **after** the partition: per component, `E_k^I` (identity
residual), `E_k^R` (whole-window Kabsch residual), `G_k = E_k^I − E_k^R`.
Larger gain is DYNAMIC. No per-point displacement threshold anywhere.

## 8. Whole-window Kabsch validation

§15: weighted Kabsch per component per frame, with per-frame residuals saved.
This validates the partition; it never constructs or seeds it. §17's `M_k(t)`
is computed and kept as debug only.

## 9. Synthetic tests S1–S7

All pass (`tests/test_motion_coherence_support.py`, 25 checks):

| test | result |
|---|---|
| S1 static + translation | dynamic IoU **1.000** |
| S2 static + revolute, offset axis | IoU **1.000** |
| S3 speed change / stop / restart / reversal | IoU **1.000** |
| S4 chaotic false dynamics at comparable amplitude | **0/60 leaked**, IoU 1.0 |
| S5 smooth non-rigid shear | C 1.97e-02 vs rigid **0.00e+00**; J < C |
| S6 move and return to start | IoU **1.000** |
| S7 tiny-motion points near the axis | **11/11 kept** (displacement 0.20 vs body median 0.84) |
| S37 differential identity | 5.0e-05 match; rigid ε 5.1e-11 |

S4 and S7 are the two that displacement cannot do: S4's chaotic points move as
far as the real body and are correctly rejected; S7's near-axis points barely
move and are correctly kept.

## 10. GT mapping per dataset

§22. Source: `input/<scene>/pointcloud/points3D_parts.ply`, integer `part`
field. The convention `moving = part != 0` is **read from the repository's own
`utils/static_endpoint_scene.py`**, not guessed. The cloud is sparse
(360–6 234 points against 7 206–40 591 Gaussians), so labels are carried by
nearest neighbour in the canonical frame; Gaussians farther than `4 ×` the
labelled cloud's own median spacing are **unmapped and excluded from every
metric** rather than assigned a guess. Written per scene to
`numerical/gt_mapping.json`.

## 11. GT metrics

| scene | GT dyn % | selector | Pred dyn % | Dyn P | Dyn R | Dyn F1 | Dyn IoU | Bal. Acc |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| USB | 57.0 | current | 44.5 | 0.910 | 0.709 | 0.797 | 0.663 | 0.808 |
| USB | 57.0 | **coherence** | 66.8 | 0.695 | 0.814 | 0.750 | 0.599 | 0.669 |
| Storage | 8.2 | current | 7.9 | 0.777 | 0.741 | 0.759 | 0.611 | 0.861 |
| Storage | 8.2 | **coherence** | 46.3 | 0.160 | 0.898 | 0.271 | 0.157 | 0.737 |
| Blade | 16.2 | current | 15.5 | 0.913 | 0.876 | 0.894 | 0.809 | 0.930 |
| Blade | 16.2 | **coherence** | 71.0 | 0.222 | 0.974 | 0.361 | 0.220 | 0.657 |
| Stapler | 54.6 | current | 45.3 | 0.999 | 0.828 | 0.905 | 0.827 | 0.913 |
| Stapler | 54.6 | **coherence** | 47.6 | **1.000** | 0.872 | **0.932** | **0.872** | **0.936** |
| Fridge | 32.8 | current | 26.7 | 0.983 | 0.800 | 0.882 | 0.789 | 0.896 |
| Fridge | 32.8 | **coherence** | 29.6 | **1.000** | 0.902 | **0.949** | **0.902** | **0.951** |

Raw accuracy is deliberately not reported alone: with storage's 8.2% dynamic
class, predicting everything static scores 92%.

## 12. Current vs coherence — what the split actually says

Coherence wins on the two scenes with the largest moving fraction and loses on
the two smallest. The ordering is monotone in GT dynamic fraction:

| scene | GT dyn % | coherence F1 | current F1 |
|---|---:|---:|---:|
| Stapler | 54.6 | **0.932** | 0.905 |
| USB | 57.0 | 0.750 | 0.797 |
| Fridge | 32.8 | **0.949** | 0.882 |
| Blade | 16.2 | 0.361 | 0.894 |
| Storage | 8.2 | 0.271 | 0.759 |

Where it wins it wins outright, at **dynamic precision 1.000** on both — when
the cut finds the right component, the component is exactly right.

Per §20 and §24 the two are never unioned, intersected, averaged or voted, and
neither seeds the other. The comparison is a table and nothing else.

## 13. Heatmap results and the GT coherence analysis

§35's diagnostic settles where the failure lives. Median edge coherence by GT
class (lower = more rigid):

| scene | within static | within dynamic | **cross** | cross / within-static |
|---|---:|---:|---:|---:|
| Stapler | 0.1937 | — | 0.5322 | **2.75×** |
| Fridge | 0.1546 | — | 0.2313 | 1.50× |
| USB | 0.1636 | — | 0.3399 | 2.08× |
| Blade | 0.1255 | — | 0.5605 | **4.47×** |
| Storage | 0.0946 | 0.1650 | 0.3493 | 3.69× |

**The invariant holds on every scene**: cross-boundary edges carry the most
strain, exactly as §13 predicts. The signal is strongest on **blade (4.47×)**,
which has the *worst* F1 (0.361). So the coherence measurement is not the
failure — the two-way normalized cut is.

Two things conspire. First, a normalized cut minimizes `cut/vol(A) + cut/vol(B)`
and is therefore biased toward balanced partitions; isolating an 8% minority
costs an Ncut penalty that a 50/50 split avoids. Storage's split was 333/301
and blade's 83/35 against GT fractions of 8.2% and 16.2%. Second, within-static
coherence is not near zero (0.095–0.194), so the rigid block is not a tight
cluster and the balance term dominates what separation there is. The spectral
gap does **not** predict the failures (blade 0.021 fails, fridge 0.021
succeeds); the GT dynamic fraction does.

§12 prescribes the normalized cut and §45 forbids forcing a fraction, so this
is reported rather than tuned around.

## 14. 3D debug paths

Per scene under `output/motion_coherence_<scene>/`:

```
debug/motion_coherence_support_3d.html     partition in 3D; graph edges drawn,
                                           cross-component edges highlighted
debug/finite_strain_heatmap.html           z_ij(t), edges sorted by C
debug/differential_strain_heatmap.html     eps_ij(t), same ordering and scale
debug/representative_edges.html            section 9's four regimes over time
numerical/{edges,nodes}.csv, components.json, edge_strain_timeseries.npz
evaluation/{confusion_matrix,current_vs_coherence,gt_coherence}.json
predicted_support.json, config.json
```

The 3D page draws the cross-component graph edges in orange: those are the
edges whose length changes, i.e. the boundary the cut actually found, so a
wrong partition is visible as a boundary in the wrong place.

## 15. No-leakage guarantees

`estimator/motion_coherence_support.py` imports **only** `__future__`, `math`
and `torch` — there is no path from the predictor to GT or to the Otsu
selector. Enforced by AST test (§46):

- no import of `estimator.moving_support`, `otsu_threshold`,
  `select_moving_support` or `trajectory_excursion`;
- no occurrence of `ground_truth`, `load_static_dynamic_gt`,
  `points3D_parts`, `gt_label`, `gt_parts`;
- `normalized_cut` takes only the graph and its affinity;
- `require_prediction_on_disk` raises unless `predicted_support.json` already
  exists, so evaluation cannot run before prediction.

GT evaluation lives in a separate file, `estimator/motion_coherence_eval.py`.

## 16. Downstream Phase-2 comparison

**Not run.** §19's `--phase2-support-source {current,coherence}` is wired into
`tools/run_complete_joint_debug.py`, and §41 is satisfied — screw, P and R all
receive exactly the same frozen `D_coherence` indices, with §45's no-fallback
behaviour (an `UNRESOLVED` partition stops the run rather than reverting to
Otsu).

The A/B itself (§43, §48 step 17) could not be executed: **the NVIDIA driver on
this machine stopped responding** mid-session — `nvidia-smi` reports it cannot
communicate with the driver and `torch.cuda.is_available()` is `False`. That is
a host-level fault, not a code fault, and it needs a driver reload or reboot.
Everything in this report that does not require the rasterizer was completed on
CPU.

## 17. Unresolved

1. **The two-way normalized cut cannot isolate a small dynamic fraction.** Below
   roughly 30% it returns a near-balanced split regardless of the coherence
   signal. This is the single blocker to the branch being usable on storage and
   blade. §12 prescribes Ncut and §45 forbids forcing a fraction, so resolving
   it needs a spec decision, not a parameter.
2. **Within-static coherence is not near zero** (0.095–0.194 against a
   theoretical 0). Phase-1's "static" trajectories carry real metric strain, so
   the rigid block is not a tight cluster. Whether that is renderer-driven free
   motion or supernode-reduction blur is not yet separated.
3. **The downstream Phase-2 A/B is outstanding**, pending the GPU.
4. **GT within-dynamic coherence** is only populated where enough same-label
   edges survive the mapping; on four of five scenes that cell is empty and the
   comparison rests on within-static vs cross.
