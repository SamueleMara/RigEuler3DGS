# Exact acceleration of the fused Phase-1 IEKS — design

`--linear-backend exact_cached_structured` (code: `bidirectional_iekf/accelerated.py`,
`bidirectional_iekf/ieks.py::_build_cached`, `_preconditioner`). The legacy
backend (`legacy_reference`) is unchanged and remains the oracle.

## The system is the same

At accepted iterate X^k both backends apply

    A_k = H^T W H + J_proc^T Q^-1 J_proc + A_prior + lambda D,   b_k = -g_k

on the free-coordinate subspace (initial-source pose eliminated at frame 0,
final-source pose at the last frame, rates free). `H` is the production
CUDA JVP/VJP over ALL Gaussians of BOTH sources in every view; nothing in A is
dropped, reduced or approximated. Only the preconditioner M may omit
cross-Gaussian image blocks, and M never enters the objective, the gradient,
the step acceptance or any reported uncertainty.

## Data lifetimes and placement

| object | built | lives in | invalidated by |
|---|---|---|---|
| per-(frame, camera) `IEKFLinearizationCache` + rendered depth/alpha/T/n_contrib + precisions | once per accepted state, on the GPU | ONE flat pageable host buffer per frame (`PackedFrame`), structure in a skeleton | a new accepted state (the store is keyed by the trajectory object, which changes only on acceptance) |
| per-frame gradient `H^T W r` | same pass | host `[T][N,12]` float64 | same |
| per-frame Fisher diagonal (and 6x6 pose blocks if a block preconditioner is selected) | same pass | host | same |
| process coefficients (nonzero 3x3 blocks of the validated `A_proc`, `B_proc`; Q once if uniform) | once per accepted state | GPU if within `--gpu-workspace-budget-mib` AND 15% of the GPU memory free at build time, else host and staged per pair | same |
| process RHS and LM diagonal contributions | same pass | host | same |
| LM metric D (`curvature_scale`) | per operator | host | state; NOT lambda (lambda multiplies it) |
| preconditioner factors | per outer iteration | host, GPU chunk by chunk | state OR lambda |
| PCG vectors (x, r, z, p, Ap) | per solve | host (`solve_on_cpu`) | — |
| the trajectory (mu, R, v, omega, appearance; every frame) | per accepted or trial state | host; one frame visits the GPU while it is rendered or differentiated | — |

A rejected trial changes only lambda, so `build_system` reuses the frozen
store and process coefficients and builds a new operator (new damping) and a
new preconditioner. The line search always re-renders the TRIAL state through
`FreeRGBDObservation.nonlinear_cost`; it never reads the frozen store.

Residual images are dropped from the host copy after the gradient is taken:
the PCG operators (JVP, precision, VJP) never read them. Everything they do
read is copied bit for bit (`tests/test_exact_cached_structured_backend.py
[2]`), including the integer `ranges` and the sorted `point_list`, with shared
tensors (one `means3D` per frame for all cameras) kept shared.

## Running beside other GPU jobs

The overnight run failed on every scene with CUDA OOM after its first
accepted step: the process coefficients became per-Gaussian (585 MB on
washer) and were placed by a fixed budget, the previous operator was still
alive during the rebuild, and the temporal preconditioner's GPU chunk was a
fixed 2,048 Gaussians. Now placement and chunk size are taken from the memory
actually free, the old operator is released before the new one is built, Q
is one 12x12 broadcast, and the trajectory itself lives in host RAM (at
foldchair's 135,694 Gaussians, 60 resident frames plus the line-search trial
were 1.6 GB of the 2.35 GB left beside the other jobs). When the temporal
factors do not fit the host budget next to the image cache, that step uses
the diagonal preconditioner and records it in the trace. None of this
changes A, b or the objective (test [11]).

## One operator application

For t = 0..T-1, in time order:

1. copy `u_t` to the GPU once (4.4 MB at 45,920 Gaussians), eliminate anchors;
2. copy frame t's packed cache to the GPU — ONE H2D of the whole frame —
   and unpack it to views of that buffer;
3. production `information_apply` (JVP, precision, VJP) over all cameras of
   the frame, accumulated on the GPU;
4. add the prior (frame 0) and `lambda D_t u_t`;
5. process pair (t-1, t) on the GPU in float64 from the kept `u_{t-1}`:
   `w = Q (A u_{t-1} + B u_t)`, `out_{t-1} += A^T w`, `out_t += B^T w`;
6. frame t-1 is now complete: eliminate anchors, ONE D2H.

The legacy streamed path re-rendered and re-linearised every frame inside
every application (`linearize_frame`, 220 ms per frame at 8 cameras measured).

## Transfer policies

`synchronous`: pageable flat buffer -> one device buffer, on the compute
stream; ordered after the previous frame's kernels by stream order.
`double_buffered`: two pinned slots and two device slots; frame t+1 is packed
into a pinned slot (CPU) and copied on a side stream while frame t computes.
The side stream waits for the compute stream before overwriting a device slot,
and the compute stream waits on the copy's event before reading it. The
pinned budget (`--pinned-staging-budget-mib`) bounds pinned memory to two
frames; the whole cache is never pinned.

## Process stencil

`_BlockMatrix` reads the sparsity of the validated dense Jacobians: a 3x3
block that is zero for every Gaussian is dropped, one identical for every
Gaussian is stored once. On washer-like data A keeps 6 of 16 blocks (2
varying: the SO(3) rows), B 4 (1 varying). Products are float64; they equal
the dense ones to 1e-16 relative (test [4]).

## Preconditioners (solver components, not models)

* `diagonal` — identical to the legacy `_diagonal_jacobi_preconditioner`.
* `block12` — per-(frame, Gaussian) 12x12 blocks with the legacy definition
  (identity added). On anchored blocks it restricts to the free coordinates
  BEFORE factorising; the legacy one inverts the full block and zeroes the
  pose afterwards (a Schur complement). Both are valid; they differ only
  there.
* `temporal_block_tridiagonal` — per Gaussian, over its whole trajectory:
  same-time blocks = the Gaussian's own image Fisher pose block + its process
  terms + prior + lambda D; adjacent-time blocks = `A_t^T Q B_t`. Factorised by
  the block Cholesky recurrence `S_t = D_t - W_{t-1}^T W_{t-1}`,
  `W = L^-1 E`, without inverting a block; chunks of Gaussians on the GPU,
  factors in host RAM. A non-SPD block raises with its frame, Gaussian and
  minimum eigenvalue; no jitter is added. The dense oracle (test [8]) checks
  that M equals A with every cross-Gaussian block removed, to 2e-7.

Host memory of the temporal factors: `2 x T x N x 144 x 8` bytes = 6.3 GB at
washer size (60 x 45,920).

## Arithmetic reuse

`pcg(..., true_residual=True)` keeps `A x` from the true-residual check, and
the solver uses it for the predicted reduction (`A0 d = A_lambda d - lambda D d`)
instead of a second application.

## What was not done

Stage F (exact pose/rate Schur elimination) is not implemented: the
temporal preconditioner already removes most of the iteration count, and the
directive makes Schur conditional on a benchmark proving it worth adding.
