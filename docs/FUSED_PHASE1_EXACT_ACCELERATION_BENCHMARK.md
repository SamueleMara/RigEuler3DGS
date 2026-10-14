# Exact acceleration of the fused Phase-1 IEKS — benchmark

Raw data: `docs/FUSED_PHASE1_EXACT_ACCELERATION_BENCHMARK.json`, and per run
`.../washer_103776/unified_phase1_experiments/acceleration_benchmark_v1/`.
Tool: `tools/profile_fused_ieks_backend.py`.

**Every number here is CONTENDED.** The GPU was shared with a FreeArt tracker
(3.1 GB) and a Part2GS training (1.9 GB), utilisation 100%. Nothing was paused.
Uncontended timings are `UNCONTENDED_REQUIRED` — not measured.

Fixed state: the real washer union (21,338 + 24,582 = 45,920 Gaussians,
all of both sources), derived 12-D initial guess from the historical tracks,
lambda = 1e-3, both source anchors in the window.

## Fixed linear system, 60 frames x 2 cameras (120 factors)

Same system for all four rows; the target is the true residual the legacy
production path reaches in 20 iterations.

| backend + preconditioner | A u | M build | M apply | 5 iterations | to the same true residual |
|---|---|---|---|---|---|
| legacy + diagonal | 6.86 s | 3.4 s | 0.11 s | 44.3 s | 20 its, **154.4 s**, 2.39e-2 |
| cached + diagonal | 5.16 s | 0.5 s | 0.09 s | 32.9 s | 19 its, **109.6 s**, 2.34e-2 |
| cached + block12 | 5.07 s | 18.0 s | 1.32 s | 41.6 s | 20 its, 142.3 s, 2.30e-2 |
| cached + temporal | 4.91 s | 20.9 s | 2.26 s | 46.2 s | **7 its, 59.2 s, 2.08e-2** |

Measured: the operator application is **1.33x faster** (6.86 -> 5.16 s); the
solve to the same certified true residual is **1.41x faster** with the same
diagonal preconditioner and **2.61x faster** with the temporal one (which
also lands at a lower residual). The temporal preconditioner's build (20.9 s)
happens once per outer step and is included in neither column; per step it
is ~1/3 of one legacy 20-iteration solve. block12 is NOT faster: its apply
cost eats the iteration saving, and it stays optional.

Host cache: 2.57 GiB for 120 views. Peak GPU 0.6–1.3 GiB.

## 8 frames x 8 cameras (64 factors) — the all-camera regime

| backend + preconditioner | A u | to the same true residual |
|---|---|---|
| legacy + diagonal | 3.17 s | 8 its, 30.1 s, 6.51e-2 |
| cached + diagonal | 2.35 s | 9 its, 25.4 s, 5.66e-2 |
| cached + temporal | 2.35 s | 6 its, 20.2 s, 3.75e-2 |

Per-frame A u at 8 cameras: 396 ms legacy, 294 ms cached. Host cache **21.6 MB
per (frame, camera) view, so the full washer 60 x 8 = 480 views take ~10.1 GiB**,
within the default 24 GiB budget and the 41 GB that were available.
Extrapolated (not measured) for 60 x 8: ~17.6 s per A u cached vs ~23.8 s
legacy, plus 2.3 s per temporal apply.

## One real nonlinear step (60 frames, 2 cameras, 60 PCG, diagonal)

Same solve settings as the legacy production run's first step, so the
accepted step is directly comparable:

| | E_total | E_rgb | E_depth | E_process | true PCG residual | wall |
|---|---|---|---|---|---|---|
| legacy_reference | 2.797073e7 | 9.416356e6 | 1.757705e7 | 9.773242e5 | 5.111e-3 | 513 s |
| exact_cached_structured | 2.797810e7 | 9.418346e6 | 1.758327e7 | 9.764789e5 | 5.115e-3 | 382 s |

Relative difference in E_total 2.6e-4 at the same true residual. A
legacy-vs-legacy repeat of this full-scale step was NOT run (the directive
allows one real step), so that floor is not measured here; at the converged
synthetic scale the legacy-vs-legacy and cached-vs-legacy objective
differences were both ~4e-6. Wall time **1.34x** faster for an identical
solve.

## Why the legacy production run was slow

The 2-camera legacy run took 5,168 s for 10 steps and ended MAX_ITERS. After
its first step, 60 diagonal-PCG iterations reached only 4.3–5.6e-2 true
residual per step: the Newton steps were very inexact, so the objective fell
slowly. The temporal preconditioner reaches 2.1e-2 in 7 iterations on the
same system.

## Not measured

Uncontended timings; a full 60 x 8 operator application; double-buffered vs
synchronous wall time at full scale (equivalence is tested, the overlap gain
is not quantified); exact pose/rate Schur (not implemented).
