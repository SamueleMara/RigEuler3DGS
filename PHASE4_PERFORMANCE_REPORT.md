# Phase 4 performance report

Every number below is measured unless it says **extrapolated**. Where a figure is extrapolated, its basis is given.

Machine: RTX 4070 Laptop (8 GB), shared with the user's `track_2part.py` (3.7 GB VRAM, 96 % utilisation during these measurements); 22 CPUs, 62 GB RAM.

## 1. Observation path (red_fridge, 378 views, 640×480)

From `reports/phase4_2026-11-11/benchmark_red_fridge_378views_10groups.json`:

| | pre-correction `ViewStore(cache=64)` | `PreparedViews` |
|---|---|---|
| decodes for two sequential sweeps | **756** (0 hits) | **378** once, **0** afterwards |
| bytes read from disk | ≈ 2 × 557 MB | 556,704,150 once |
| time | 7.8 s per two sweeps | 3.4 s decode once; 0.59 s per two sweeps (GPU uploads only) |
| host memory | per-view tensors, re-decoded | 0.97 GiB unpacked (uint8 RGB, float32 depth, bool mask / valid); RSS 1.9 GiB after preparation |
| GPU | – | one view at a time (≈ 2.8 MB per upload) |

## 2. Exact 4A scoring throughput

| measurement | value |
|---|---|
| 378 views, 10 fixed groups, red_fridge | **198 group-views / s** (3,760 scored, 20 identity views skipped); 19.0 s |
| 24 views, 50 fixed groups, red_fridge | 312 group-views / s (more groups share each view's baseline) |
| full-image reference on the same 24 × 50 | 46 group-views / s (ROI is 6.8× faster) |
| ROI pixels per group-view | 21,700 average (378-view run: 81.5 M pixels / 3,760); full frame is 307,200 |
| peak CUDA (allocated / reserved) | 284 MiB / 376 MiB (378-view benchmark); 210 MiB / 296 MiB (CLI subset run) |
| peak RSS | 2.7 GiB (benchmark), 1.7 GiB (CLI subset run) |
| GPU → CPU copies | 5 per sweep (the score arrays); no per-candidate image copy |
| procedural gate scenes (≈1,100–1,250 Gaussians, 24 views) | 10–13 s for a complete 4A including sensitivity, vs 110–146 s pre-correction |

## 3. Complete-run counts (CLI, red_fridge, 4-frame subset = 8 views, NOT_FINAL)

From `reports/phase4_2026-11-11/subset4_*`:

| quantity | value |
|---|---|
| groups | 288 (`max_component 128`, `link_factor 3`) |
| sweeps | 23 (core check, initial, 4 batches with joint moves and re-scores, split probes, final verification with OMIT, 10 sensitivity perturbations) |
| hypothesis renders | 10,344 = **1,293 per view** |
| OMIT renders | 1,824 = 228 per view |
| identity skips | 3,448 (the anchor frame of each camera) |
| 4A wall time | 48 s (CPU 54 s) |
| 4B, 8 views | 0.7 s |
| 4C, 23 candidates over 8 views | 49 s, i.e. ≈ 0.27 s per candidate-view including the fit and pruning |

## 4. Extrapolation to the complete 378-view red_fridge run

These estimates scale the subset's per-view counts to 376 non-identity views at the 378-view measured rate. They assume the batch, split and sensitivity structure of the subset carries over, which it may not: more views can change how many groups are decisive.

| stage | basis | estimate |
|---|---|---|
| 4A | (1,293 + 228) item-views per view × 376 views / 198 per s | **≈ 48 min** (+ about 3 min footprints and preparation) |
| 4B | linear in views | ≈ 0.5 min |
| 4C | 0.27 s per candidate-view × 378 views ≈ 100 s per candidate (both hypotheses) | **≈ 100 s × number of candidates** (23 on 8 views; likely more on 378) |

4C is unoptimised: it renders a baseline per view per candidate, with no view-major batching. Even 100 candidates means about 3 h. That is the measured bottleneck to fix before 4C is used at scale.

## 5. Progress records

- Every message carries elapsed minutes, RSS and CUDA allocated / peak.
- Each 4A sweep logs view position, rate and the measured remaining time of that sweep every 60 s (`--p4a-progress-every-s`).
- Every batch logs proposals, commits, the predicted vs realised objective change, and the re-score count.
- `stage4a/checkpoint.json` is rewritten atomically after every batch, so no run can spend an hour without a progress record on disk.

## 6. Fix after review (2026-11-11): spill store file descriptors

The memory-mapped fallback used to write 4 `.npy` maps per view. Python's `mmap` keeps a duplicated descriptor per map, so 378 views needed about 1,500 descriptors and failed with `OSError: [Errno 24] Too many open files` under the default `ulimit -n 1024`. It now writes one flat file per array kind (4 descriptors in total). `test_phase4_view_cache.py` checks every view and every array against the in-RAM store, asserts exactly 4 spill files, and passes under `ulimit -n 1024`. The real red_fridge run (0.97 GiB, below the default `max_host_gb` of 8) keeps its views in RAM and never took this path.
