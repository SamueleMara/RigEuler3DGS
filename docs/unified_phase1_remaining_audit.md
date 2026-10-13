# Unified Phase 1 — what exists, what is reused, what is missing

Audit of the checkout at `9b78efa`, written before any edit, as
`CODEX_COMPLETE_MISSING_UNIFIED_BIDI_PHASE1_AFTER_9B78EFA.md` section 1
requires. Everything below was read from the files, not recalled.

## 1. Checkout state

    git log --oneline -3
      9b78efa T9 PASS: one posterior, two initializations, no double-counted evidence
      8655eb5 G1A, G1B and G2: nonzero prior gradient, analytic Taylor, eigen/Schur evidence
      0aadc1e Fix the predicted-reduction formula; verify each gradient term separately

    git status --short
      m submodules/sam3          <- dirty BEFORE this work, never touched

`git diff --cached --name-only` is empty. No `git add -A`, `reset` or `clean`
is used anywhere in this work.

## 2. APIs to reuse, with their real signatures

### 2.1 The accepted trajectory (`bidirectional_iekf/ieks.py`)

    Trajectory(states: list[GaussianMotionState], frame_ids: list[int], dt: float)
      .T -> int                 number of frames
      .N -> int                 number of candidates
      .centers()    -> [T,N,3]  stacked mu
      .rotations()  -> [T,N,3,3]
      .velocities() -> [T,N,3]  per SECOND, physical time
      .angular_velocities() -> [T,N,3]

    UnifiedFreeIEKS.run(trajectory) -> {
        "status": CONVERGED | STALLED | MAX_ITERS | LINE_SEARCH_FAILED
                  | PCG_FAILED | NONFINITE,
        "trajectory": Trajectory,      # the ACCEPTED nominal
        "energy": {...}, "trace": [...], "factor_inventory": {...},
        "covariance_contract": {...}, "wall_time_s": float}

    save_run(out_dir, result) writes out_dir/unified_free/*.npy and
    out_dir/run_status.json.

`states[t]` holds EVERY candidate at physical frame `frame_ids[t]`. Centers
taken from one `t` are therefore simultaneous by construction; this is the
only place in the repository where that is true for a mixed-source
population.

### 2.2 Candidate provenance (`bidirectional_iekf/provenance.py`)

    CandidatePopulation(source_endpoint[K], source_gaussian_index[K],
                        anchor_xyz[K,3], anchor_original_frame[K], anchors)
      .num_candidates, .mask_for(e), .rows_for(e), .index_hash()
    build_candidate_population(anchors) -> CandidatePopulation
    load_source_anchor(endpoint, model_dir, iteration=30000,
                       anchor_original_frame=int) -> SourceAnchor
    candidate_manifest(population, total_frames=, source_pose_treatment=,
                       representation_gate="NOT_RUN") -> dict

**`anchor_xyz` is NOT a common-time array.** The initial anchor is frame 0 and
the final anchor is frame T-1, so `anchor_original_frame` takes two distinct
values. Using it to build the coherence graph is exactly the mistake section
3.3 forbids. The adapter refuses it; see the negative control in T10.

### 2.3 The coherence branch — unchanged, called as production calls it

`estimator/motion_coherence_support.py`

    supernode_trajectories(positions, supernode_id, weights=None)
        -> {"trajectories": [T,n_s,3], "valid": [n_s] bool,
            "num_supernodes": int, "members_per_supernode": [n_s]}
    canonical_knn_graph(x0, k=12)      # graph built at traj[0]
    run_motion_coherence(positions, supernode_id, knn=12, times=None,
                         weights=None, verbose=True)
        -> {"node_ids", "edge_i", "edge_j", "C", "J", "z", "epsilon", "W",
            "labels", "components", "identification", "num_supernodes", ...}
    propagate_to_gaussians(dynamic_nodes, supernode_id) -> member row indices

`estimator/coherence_extraction.py`

    extract_moving_component(traj, edge_i, edge_j, C, n, min_nodes=3,
                             folds=5, max_candidates=48,
                             min_merge_support=2, refine=True, verbose=True)
        -> {"status", "labels": list[str] length n, "counts", "moving_nodes",
            "candidates", "candidate_members", "selection", "refinement"}

    label literals: MOVING_RIGID, STATIC_COMPATIBLE,
                    INCOHERENT_OR_AMBIGUOUS, and the status
                    NO_COHERENT_MOVING_COMPONENT

`estimator/supernodes.py`

    build_supernodes(means, target_size=64, num_supernodes=None, knn=8,
                     iterations=25, seed=0) -> SupernodePartition(.supernode_id)

The production runner `tools/run_motion_coherence_support.py` chains these in
exactly this order (lines 337-380): supernodes from the canonical geometry,
`run_motion_coherence`, then `supernode_trajectories` + `extract_moving_component`
on `res["node_ids"]`, then `propagate_to_gaussians`. The unified adapter
reproduces that chain verbatim and changes only where the centers come from
and what the partition is built on.

### 2.4 Index spaces, which are three and are not interchangeable

    candidate index   k in [0, K)     row of the disjoint union
    supernode id      s in [0, n_s)   build_supernodes output
    node index        i in [0, n)     position within res["node_ids"]

`extract_moving_component` returns labels in NODE index space.
`res["node_ids"][ex["moving_nodes"]]` converts to supernode ids, and
`propagate_to_gaussians` converts those to candidate indices. Skipping the
first conversion is an off-by-many, not an off-by-one, whenever any supernode
came out empty.

## 3. Real Washer data on this machine — verified, not assumed

    OUT=/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours/washer_103776

| path | state |
|---|---|
| `static_bidi/initial`, `static_bidi/final` | present |
| `bidirectional/backward/phase1_centers` | 60 frames, 24,582 Gaussians |
| `bidirectional/forward/phase1_centers` | 60 frames, 21,338 Gaussians |
| `bidirectional/{forward,backward}_rigid` | present, untouched |

**The forward free Phase 1 is NOT missing.** It was produced on 2026-10-13 at
10:19 and its manifest reads `direction: forward`, `canonical_endpoint:
initial`, `trajectory_order: original_chronological`, 60 frames. Section 4.3
of the directive assumes it absent; that premise is stale, and nothing needs
to be re-run. Measured from the files:

| | forward (initial) | backward (final) |
|---|---|---|
| Gaussians | 21,338 | 24,582 |
| `max abs(centers_frame_0000 - canonical_xyz)` | 0.0 | 0.830 |
| max displacement over the window | 1.477 | 1.014 |
| coherence status | RESOLVED | RESOLVED |
| dynamic Gaussians | **21,338 of 21,338** | 5,131 of 24,582 |

The zero and the 0.830 are both correct and are the direction convention
working: for a forward run frame 0 IS the canonical geometry, for a backward
run frame 0 is the final model transported to the start of the sequence.

The forward coherence result is degenerate: every Gaussian is in the moving
core, so there is no static reference left to be moving relative to. That is a
real result about the forward free pass on this scene, not a file problem, and
it is why the backward 13-supernode / 5,131-Gaussian support is the one the
rigid stage consumed. Both are historical regression references here.

`21,338 + 24,582 = 45,920` — the candidate count the directive quotes.

## 4. Currently passing suites (counts as executed, not as acceptance)

    test_unified_source_index.py              22
    test_unified_process_jacobians.py         32
    test_unified_full_window_map.py           18
    test_unified_cuda_renderer_jvp_vjp.py     22
    test_unified_nonlinear_ieks.py            32
    test_unified_ieks_convergence.py          10
    test_unified_ieks_gradient_components.py  24
    test_unified_ieks_observability.py         8
    test_unified_direction_invariance.py      22

## 5. Blockers, honestly

* **Photometric capture radius.** Measured on the T8 fixture: a 5.8 px
  projected radius against an 8.3 px displacement leaves the image gradient
  pointing nowhere useful. Every synthetic IEKS fixture must warm-start inside
  that radius or it reports STALLED, correctly.
* **Representation.** The two endpoint mixtures were trained independently and
  overlap in space. Until the gate measures otherwise their union is a
  candidate state space, not one scene.
* **Memory.** 60 x 45,920 x 12 = 33,062,400 tangent scalars. Nothing dense of
  that size may be allocated; the adapter keeps centers on the CPU in float64
  and never holds more than one frame of CUDA state.

---

# Results of this pass

Written after the work, with the numbers the runs produced.

## 6. The gate matrix, as it now stands

| Gate | Status | Evidence |
|---|---|---|
| G1A | PASS, retained | nonzero prior gradient incl SO(3), rel 1.30e-05 |
| G1B | **corrected** | `tests/test_unified_curvature_and_observability.py` |
| G2 | **corrected** | same file; image rate blocks are identically zero |
| T5 | PASS (scoped) | unchanged |
| T6 synthetic | bounded diagnostic | unchanged |
| T8 | PASS local; large-displacement stress fails | unchanged |
| T9 | PASS on nonoverlap | unchanged |
| T9 source-T | **PASS** | terminal anchor eliminated exactly, both ends at once |
| T10 | **PASS** | `tests/test_unified_coherence_t10.py`, 42 checks |
| T11-source-only | **RUNS** | real 60x8 window at 2265 MiB; command below |
| Washer forward | **present, verified** | not missing; see section 3 |
| T11-merged | **UNRESOLVED_SCENE_REPRESENTATION** | 86-95% doubly covered pixels |
| Rigid P2 | not demonstrated | unchanged |
| Joint P3 | not demonstrated | unchanged |
| Joint P4 | not demonstrated | unchanged |

Full suite, all passing, every one executed:

    test_unified_source_index                 22
    test_unified_process_jacobians            32
    test_unified_full_window_map              18
    test_unified_cuda_renderer_jvp_vjp        22   real CUDA
    test_unified_nonlinear_ieks               32   real CUDA
    test_unified_ieks_convergence             10   real CUDA
    test_unified_ieks_gradient_components     24   real CUDA
    test_unified_ieks_observability            8   real CUDA
    test_unified_direction_invariance         22   real CUDA
    test_unified_coherence_t10                42   real CUDA, real extractor
    test_unified_streaming_window             23   real CUDA
    test_unified_curvature_and_observability  24   real CUDA
    test_bidirectional_sequence_direction     54
    test_bidirectional_time_and_source        63
    test_motion_coherence_support             41

## 7. Memory on the real window, measured

`washer_103776` / final, 24,582 Gaussians, 640x480, RTX 4070 Laptop 8 GiB.
Peak CUDA at the system build, one outer iteration, two PCG iterations.

| frames x cameras | factors | resident | + stream_frames | + host process factors | + diagonal preconditioner |
|---|---|---|---|---|---|
| 4 x 1 | 4 | 878 MiB | | | |
| 8 x 2 | 16 | 1901 MiB | | | |
| 16 x 4 | 64 | **OOM** 4147 MiB | 3611 MiB | | |
| 30 x 8 | 240 | | **OOM** 3950 MiB | 4504 MiB, 22.2 s | |
| 60 x 8 | 480 | | | **OOM** 4155 MiB | **2265 MiB, 43.6 s** |

The resident path costs about 85 MiB per `(frame, camera)` factor, so the real
window would need of the order of 41 GiB. The three options are bounded
streaming, not reduced states: the arithmetic is identical and
`tests/test_unified_streaming_window.py` checks it against a measured
float32 reproducibility floor rather than bitwise.

## 8. Commands for the runs that belong to the user

    REPO=/home/samuelemara/Joint3DGS/RigEuler3DGS
    PY=/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python
    SRC='/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear/washer_103776'
    OUT='/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours/washer_103776'

SINGLE_SOURCE on the full real window, source final (hours, not minutes):

    cd $REPO && $PY tools/run_unified_phase1_single_source.py \
      --endpoint final --frame-start 0 --frames 60 --cameras 8 \
      --stream-frames --frames-on-cpu --process-factors-on-cpu \
      --diagonal-preconditioner \
      --max-outer 8 --pcg-iters 40 \
      --source "$SRC" --out-root "$OUT"

The same for source initial, which anchors at original frame 0:

    cd $REPO && $PY tools/run_unified_phase1_single_source.py \
      --endpoint initial --frame-start 0 --frames 60 --cameras 8 \
      --stream-frames --frames-on-cpu --process-factors-on-cpu \
      --diagonal-preconditioner \
      --max-outer 8 --pcg-iters 40 \
      --source "$SRC" --out-root "$OUT"

A short rung first, if preferred (minutes):

    cd $REPO && $PY tools/run_unified_phase1_single_source.py \
      --endpoint final --frame-start 52 --frames 8 --cameras 2 \
      --max-outer 8 --pcg-iters 40 \
      --source "$SRC" --out-root "$OUT"

The representation gate, already run and recorded under
`$OUT/unified_phase1_tests/representation_gate/`:

    cd $REPO && $PY tools/run_unified_representation_gate.py \
      --source "$SRC" --out-root "$OUT" --frames 0 15 30 45 59 \
      --cameras 4 --save-images

## 9. What is still blocked, and by what

* **T11-merged.** 86-95% of covered pixels receive coverage from both endpoint
  models. The union is two reconstructions of the same surfaces, so a merged
  IEKS would count those surfaces twice and nothing in the theory says which
  copy is physical. Resolving it needs a representation decision that the
  directive forbids being made silently -- not more compute.
* **Unified coherence on real data.** T10 runs on accepted states and is
  validated synthetically. On Washer it needs an accepted two-source
  trajectory, which the gate blocks. A SINGLE_SOURCE accepted trajectory can
  be fed to it today, and that is a one-source result, not the unified core.
* **P2, P3, P4.** Untouched. Each needs its own state space and its own
  acceptance gate; none is demonstrated and none is claimed.
