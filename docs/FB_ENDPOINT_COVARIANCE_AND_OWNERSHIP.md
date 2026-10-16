# Endpoint correspondence: covariance-consistent transport and tested J1 ownership

Two changes to the endpoint diagnosis (`tools/analyze_fb_endpoint_correspondence.py`) made before the
J1 failures of `endpoint_eval_v3` (scissor, blade, laptop) are read as missing geometry.

## 1. Orientations move with the rigid motion

The 3DGS covariance is Σ = R S Sᵀ Rᵀ. Moving a centre by (R_m, t) without R ← R_m R leaves every
anisotropic footprint pointing the old way.

* **J1** (estimated part motion) already turned owned rows. `j1_state` sets R ← R_part R, and test [2]
  checks it.
* **J0** (free Phase-1 transport) used the trained orientation at every frame, because the historical
  tracks saved centres only. It now recovers a rotation per row (`ec.track_rotations`) with a Kabsch fit
  of the row's 16 nearest neighbours' track. It applies the rotation **only where that neighbourhood
  moved rigidly**: rms residual ≤ 0.25 × the neighbourhood's rms extent. Torn and collinear
  neighbourhoods keep the trained orientation and are counted. The flag is
  `--transport-orientation {track_kabsch,fixed}` (default `track_kabsch`, part of the fingerprint).
* **Why the gate is needed.** The real free tracks tear most neighbourhoods. On scissor, neighbours
  separate by 2.8× their spread at every k from 16 to 256. Unconditioned Kabsch rotations there had a
  median of 64–72° and a 90th percentile of 162°, and tripled the depth error of FT and B0 (1.3 → 4.5 cm
  MAE).
* **Effect, with the gate** (`transport_orientation_effect.csv`, turned vs fixed, recorded RGB-D):

  | scene | rows turned | depth MAE change | RGB RMSE change |
  |---|---|---|---|
  | scissor | 19% | −0.2 mm | ≤ −0.0015 |
  | blade | 24–27% | ≈ 0 | ≈ 0 |
  | laptop | 9–14% | −0.7 mm | ≈ 0 |

  The J0 transport mismatch is dominated by torn centre tracks, not by orientation.

## 2. J1 ownership tested on the recorded RGB-D

`endpoint_eval_v3` moved the estimator's support (owner source) and, in the other source, the rows
within 2 × median splat scale of the moved part. The evaluator shows that the support itself was the
problem (GT, `metrics_support.csv`, 2 cm, evaluation only):

| scene, source | support precision | support recall |
|---|---|---|
| scissor F | 1.00 | 0.43 |
| scissor B | 1.00 | 0.50 |
| laptop B | 0.87 | 0.64 |
| blade B | 1.00 | 0.74 |
| fridge | 0.96–1.0 | 0.89–0.94 (the scene where J1 helped) |

Half of scissor's moving blade was left at its reference pose, and that is what J1 "failed" on.

`ec.tested_ownership` now decides **every row of both sources**, GT-free, at the other endpoint:

1. **Free-space test** (`ec.freespace_ownership`). The row's centre is projected, static and moved,
   into every camera.
   * At the recorded depth: support.
   * The recorded surface lies behind the centre, so the centre floats in observed free space:
     contradiction.
   * Occluded, invalid or out of view: no information.

   The row is decided when the net difference reaches `MIN_WITNESS_CAMERAS` (2).
2. **Render test** (`ec.observation_ownership`) for open rows near the part. Two renders per camera
   compare the depth-compatible supported mass of the moved and static hypotheses.
3. **Still open:** the estimator support (owner source), or the free track's endpoint position (other
   source), else static.

The owned sets are saved (`j1_ownership.npz`). The evaluator scores them as
`j1_tested_ownership_<dir>_joint` beside the support. Provenance records every stage's counts
(`joint_estimate_provenance.json → j1_runs.<dir>.ownership`).

### Effect (J1 misses, % of valid recorded-depth pixels; ownership F1 from GT at 2 cm)

| scene | contrast | v3 | now |
|---|---|---|---|
| scissor | frame T, F unexplained (bwd / fwd joint) | 15.35 / 21.45 | 3.05 / 4.36 |
| scissor | frame 0, B unexplained (bwd / fwd) | 18.98 / 22.49 | 4.98 / 3.82 |
| blade | frame T (bwd / fwd) | 12.62 / 11.09 | 4.95 / 7.42 |
| blade | frame 0 (bwd / fwd) | 7.75 / 5.41 | 3.57 / 4.16 |
| laptop | frame T (bwd / fwd) | 30.10 / 11.19 | 23.32 / 15.38 |
| laptop | frame 0 (bwd / fwd) | 37.77 / 18.42 | 21.00 / 21.43 |

J0 at T for comparison: scissor 4.23, blade 7.74, laptop 22.64.

Ownership F1 against GT:

| scene, source | support F1 | tested ownership F1 |
|---|---|---|
| scissor F | 0.60 | 0.97–0.98 |
| scissor B | 0.67 | 0.96–0.98 |
| laptop B | 0.74 | 0.90–0.92 |
| blade B | 0.85 | 0.90 |
| blade F | 0.95 | 0.87–0.89 (more false positives) |

Reading:

* **Scissor and blade.** The J1 failures were ownership, not missing geometry. With tested ownership,
  J1 explains the endpoints as well as J0 or better.
* **Laptop.** The backward joint is the poorest estimate: 2.4° axis, 6.7° coordinate MAE, Phase-2 pose
  3.9° mean. The forward frame-T miss is ownership-ambiguous. Over the variants below it ranges from 8%
  to 23%, depending on how a few dozen huge opaque splats (σ 13–17 cm, opacity ≈ 1) are assigned. Their
  centre says little about where they render.
* **Caveat.** The ownership is fitted to the recorded RGB-D at the very endpoint that the J1 contrast is
  scored on. The estimated motion is fixed and each row only chooses static or moved, but the
  frame-T/frame-0 J1 numbers are in-sample for ownership.

### Variants tried (scissor + blade + laptop, sum of the 12 J1 miss percentages)

| variant | sum | note |
|---|---|---|
| v3: support + proximity | 210.4 | |
| **every row by centre free-space, render test near part (default)** | **116.7** | |
| free-space only for compact splats (2σ ≤ depth tol) | 149.6 | best laptop fwd at T (8.0) |
| + render test of every open row in context | 158.7 | |
| + neighbour vote for wide splats | 152.8 | best scissor |
| wide owner splats keep the support label | 143.4 | |

Every variant beats v3. None wins on every scene. The default is the simplest rule with the lowest
total.

## Tests

`tests/test_fb_covariance_transport.py` (15 checks, real CUDA renderer):

* [1] Moving the scene and the camera together leaves the render unchanged; moving centres only does
  not.
* [2] `j1_state` turns the covariance.
* [3] `track_rotations`:
  * a rigid track gives back its rotation exactly;
  * static, collinear and two-part tracks;
  * a torn track is flagged and keeps its orientation.
* [4] Turning splats lowers the RGB/depth error of a transported state.
* [5] Ownership with interleaved parts:
  * proximity grabs static rows;
  * observation testing does not;
  * with half the support given, every distinguishable row of the part is recovered and no static row
    is taken.

The endpoint (49) and GT-evaluation (51) suites pass. Ownership timings are kept out of the provenance
so that runs with real, altered and absent GT stay identical.
