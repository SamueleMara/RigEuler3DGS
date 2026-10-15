# Forward/backward pixel overlap and complementary coverage — implementation

From `CODEX_PHASE1_FORWARD_BACKWARD_PIXEL_OVERLAP_COVERAGE_DIAGNOSTICS.md`.
A read-only measurement of the EXISTING Phase-1 sources. Nothing is trained,
merged, pruned or optimised; the raw F+B union is never rendered.

| | |
|---|---|
| F | `static_bidi/initial` + `bidirectional/forward/phase1_centers` |
| B | `static_bidi/final` + `bidirectional/backward/phase1_centers` |
| motion state | `CENTER_ONLY_FIXED_ORIENTATION` (centres from the tracks, PLY orientation) unless a `--full-state-out` export exists |
| renderer | production `iekf_build_cache` + `iekf_render` (unmodified), DC colour, black background |
| output | `<results>/<scene>/observation_overlap_coverage/<tag>/` only |

The code that produced `<scene>/bidirectional/` (Phase 1, coherence, rigid
Phase 2) is preserved as git tag `bidirectional-batch-2026-10-12` (`b99ee40`).

## Code

| file | role |
|---|---|
| `observation_diagnostics/renderer_access.py` | standalone diagnostic CUDA kernel: the production render loop, emitting one record (Gaussian row, alpha, T before) per composited contributor; tile-aware float64 reference compositor for tests |
| `observation_diagnostics/sources.py` | read-only loading and verification of the two sources, the calibration and the GT RGB-D; protected-file hashing |
| `observation_diagnostics/pixel_evidence.py` | per-pixel coverage / explanation / depth-layer categories, metrics, threshold sweeps |
| `observation_diagnostics/gaussian_evidence.py` | per-Gaussian, per-frame accounting of exact contributions (mass, supported mass, witness U, ...) |
| `observation_diagnostics/pair_evidence.py` | cross-source co-contribution sums as exact sparse products, pruned storage with accounted omissions, sequence merge |
| `observation_diagnostics/aggregate.py` | tables, labels, certification, patches, occlusion test, case studies, galleries |
| `observation_diagnostics/report.py`, `visualize.py` | `report/*.md`, maps, panels, `visuals/index.html` |
| `tools/analyze_fb_pixel_overlap.py` | CLI: preflight, coverage pass, pair pass, aggregation; resumable per frame |
| `scripts/run_fb_pixel_overlap.sh` | one scene |
| `scripts/run_fb_pixel_overlap_all.sh` | every eligible scene, serially |

## Exactness

* **Contributions** q = T·alpha are the renderer's own: the diagnostic kernel
  copies `renderFromCacheCUDA` (same tiles, sort, `evalAlpha`, `__expf`,
  0.99 clamp, 1/255 skip, T_STOP early termination). On every processed view
  the run checks sum q = A, sum q z/(A+eps) = D and final T bitwise.
* **Pair sums** O, O_surf (2/4/8 cm), O_incompat, pixel counts per view are
  exact sparse products Q_F^T Q_B (cuSPARSE SpGEMM, float64), using that the
  same-surface kernel factors into per-contributor depth tests and a per-pair
  |z_F - z_B| test. Checked against a CPU enumeration of every pair (T9).
* **Storage is bounded**: a view's pair is stored if its per-view O >= 1e-4
  (`--pair-prune`). The pruned mass is accumulated per Gaussian, so every
  stored value is a lower bound, and a best partner is CERTIFIED when its lead
  exceeds the row's omitted same-surface mass. One frame (two cameras) is also
  stored unpruned and compared. Status in reports: `APPROXIMATE_BOUNDED`.

Measured on that reference: storage 0.086 %, washer 0.065 %, foldchair 1.9 %
of same-surface mass dropped; best partners identical wherever the best
same-surface overlap is >= 1e-2 (100 %, also 100 % mass-weighted); the
disagreements (3–14 % of rows) are all Gaussians with negligible evidence.

## Tests

| suite | checks | covers |
|---|---|---|
| `tests/test_fb_pixel_overlap_core.py` | 58 pass | T1 conservation + independent compositor, T3 identical Gaussian, T4 same pixel / different depth, T5 occluded then revealed (incl. OCCLUSION_HYPOTHESIS), T6 wrong depth despite coverage, T7 hidden behind opaque foreground, T8 fail closed, T9 sparse products = CPU exhaustive pairs, pruning accounting, T11 sweeps monotone, T14 GT-invalid never missing |
| `tests/test_fb_pixel_overlap_real.py` | 34 pass | T2 endpoints / chronological order / row identity (storage, washer), T8 one calibration, T10 parity (records rebuild production RGB/alpha/depth; production = training rasterizer except <= 1 boundary pixel per view), T12 read-only + resume refusal + resume reuse, T13 motion-state labelling |

## Measured costs (shared GPU, 8 GB, other jobs running)

| scene, views | coverage | pairs | pair records | stored pairs / frame | host peak | GPU peak |
|---|---|---|---|---|---|---|
| washer, 1 frame x 8 cams | 13.1 s | 12.8 s | 2.48e9 | 6.6e6 | 6.5 GB | 1.7 GB |
| foldchair, 1 frame x 8 cams | 13.3 s | 22.1 s | 1.84e9 | 2.9e7 | 12 GB | 1.7 GB |

Not measured: a full 60-frame scene. Extrapolated: 30–60 min per scene
(coverage ~13 min, pairs 13–25 min, aggregation and case studies ~10 min),
so all ten scenes in roughly 6–9 h. The sequence pair table grows with new
pairs per frame; for foldchair it can reach several GB. The pair pass
checkpoints every 10 frames and stops cleanly (`HOST_RAM_LOW`, exit 3) if
available RAM falls under `--min-free-ram-gib` (3 GiB); `--resume` continues.

## Commands

```bash
cd /home/samuelemara/Joint3DGS/RigEuler3DGS
bash scripts/run_fb_pixel_overlap.sh --scene washer_103776 --frames all --cameras all --tag overlap_coverage_v1 --preflight
bash scripts/run_fb_pixel_overlap.sh --scene washer_103776 --frames 0,10,20,30,40,50,59 --cameras cam_000,cam_004 --mode all --tag overlap_coverage_pilot_v1
bash scripts/run_fb_pixel_overlap_all.sh --tag overlap_coverage_v1 --resume
```

Start with `<out>/report/RESULTS.md`, `report/TOP_MISSING_GEOMETRY.md` and
`visuals/index.html`.
