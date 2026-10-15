# Exact acceleration — equivalence to the reference system

Tests: `tests/test_exact_cached_structured_backend.py` (51 checks, real
CUDA), machine-readable rows in
`docs/FUSED_PHASE1_EXACT_ACCELERATION_EQUIVALENCE_tests.json`; the real
full-scale step is in `docs/FUSED_PHASE1_EXACT_ACCELERATION_BENCHMARK.md`.

Fixture: two sources (4 + 4 Gaussians) overlapping in every view, a rotating
body with per-Gaussian rotation and rate errors (so the SO(3) process blocks
vary per Gaussian and the rotation residual is nonzero), a correlated Q with
pose-rate cross terms, 4 frames, 2 cameras, initial anchored at frame 0 and
final at frame 3, lambda = 3e-3.

## Tolerances come from a measured floor

The CUDA image path is float32 with atomic accumulation, so two legacy builds
at one state differ. Three legacy-vs-legacy rebuilds give the floor: `A u`
9.0e-8, `b` 6.6e-8, `D` 9.5e-8 (relative). Image-path checks allow 4x that;
float64 process checks are held to 1e-12.

| quantity | reference | optimised | measured | tolerance | pass |
|---|---|---|---|---|---|
| frozen cache round trip | linearize_frame | pack -> host -> device | bitwise, all tensors; shared tensors shared; point_list, ranges identical | exact | yes |
| `A u` | legacy | cached | 1.3e-7 | 3.6e-7 | yes |
| `b = -g` | legacy | cached | 7.0e-8 | 2.6e-7 | yes |
| LM metric D | legacy | cached | 1.2e-7 | 3.8e-7 | yes |
| symmetry `u'Av - v'Au` | legacy 1.41e-7 | cached 1.63e-7 | | 4x legacy | yes |
| anchors in `A u`, `b` | | | own rows zero, other source free | exact | yes |
| process stencil | dense validated A_proc, B_proc | 3x3 block stencil | 1.6e-16 | 1e-12 | yes |
| omitted pose-rate block of Q | | | caught, 3.4e-3 | > 1e-3 | yes |
| cache rebuilt inside PCG | | | 0 builds in 15 iterations | 0 | yes |
| cache rebuilt on lambda change | | | 0 | 0 | yes |
| cache rebuilt on new state | | | T builds | T | yes |
| stale state / scrambled point_list | | | detected (8.9e-3 / 1.6e-1) | > 100x floor | yes |
| double-buffered vs synchronous `A u` | | | 7.3e-8 | 3.6e-7 | yes |
| diagonal M^-1 | legacy | cached | 1.6e-10 | 3.8e-7 | yes |
| block12 M^-1, unanchored frames | legacy block-Jacobi | cached | 3.5e-7 (legacy adds a 1e-9 ridge) | 1e-6 | yes |
| block12 M^-1, anchored blocks | legacy inverts full block | cached restricts to free coords | differ by design (4.1e-1) | — | reported |
| temporal M vs dense oracle | A with cross-Gaussian blocks removed | inverse of the factored M | 2.0e-7 | 1e-5 | yes |
| temporal M SPD | | | min eig of M^-1 5.5e-6 > 0 | > 0 | yes |
| M^-1 on eliminated coordinates | | | exactly 0 | 0 | yes |
| linear solve, same true residual | legacy diagonal, 160 its | diagonal 160 / block12 168 / temporal 36 its | solutions agree to 2.7e-7 – 7.4e-7 | 1e-6 | yes |
| `A x` reuse for predicted reduction | fresh application | PCG's true-residual product | within floor | | yes |
| converged nonlinear run | legacy E 32.93270 (legacy-vs-legacy 1.4e-4) | cached diagonal 32.93260, cached temporal 32.93276 | states within 1.7e-5 / 1.8e-5 (legacy-vs-legacy 1.9e-5) | 10x legacy spread | yes |
| anchors after a nonlinear run | | | bitwise unchanged | exact | yes |
| host-resident trajectory | GPU-resident | trajectory in host RAM, one frame on the GPU | A u, b, D within floor; same objective; converged states 4.9e-6 from legacy (legacy-vs-legacy 4.4e-6); anchors bitwise | floor / 10x legacy spread | yes |

## Theory mapping

Nothing in the posterior changes: the same image factors (one per original
frame and camera, both sources rendered together), the same SO(3) process
residual and Jacobians, the same Q, the same Gaussian RGB-D precisions, the
same anchor elimination, the same LM metric and acceptance rule. The frozen
store holds the linearisation at the CURRENT accepted state only and is
discarded on acceptance; trial states are always re-rendered. The
preconditioners are solver components; none is exported as information or
covariance.

The raw two-source union is still a diagnostic representation; faster
inference over it does not make it a validated canonical scene.
