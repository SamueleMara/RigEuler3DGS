"""Matrix-free preconditioned conjugate gradient and block-Jacobi/Fisher preconditioner.

Implements specification sections 15 and 16.

Section 15 forbids ever forming ``H`` or ``H^T R^-1 H``.  The solver below only
ever sees the normal operator through a callable

    A_mv(v) = Lambda^- v + H^T R^-1 (H v) + lambda D v,

so the whole CUDA JVP/VJP pipeline plugs in without materializing anything.
State vectors are plain tensors of shape ``[N, D]`` (N Gaussians, D = 12 error
dimensions of section 4) and are treated as one flat vector; the code is shape
agnostic, so ``[N, 6]`` pose-only or reduced joint states work unchanged.

Section 16 supplies the preconditioner

    M_i = Lambda^-_ii + F_i + lambda D_i,     M^-1 = blockdiag(M_1^-1, ..., M_N^-1)

applied through a per-block Cholesky.  Fisher blocks accumulated from a tile
rasterizer are only positive *semi*definite - a Gaussian that is fully occluded,
degenerate or outside every tile contributes nothing - so non-PD blocks are
expected, repaired with escalating jitter and *counted*, because the count is a
direct observability diagnostic (section 34: "fraction of weakly observable
Gaussian pose blocks").

Honesty requirements taken literally from the spec:

  * a zero right-hand side returns exactly zeros and performs no iteration;
  * NaN/Inf anywhere aborts with a reason and a finite iterate, never a silent
    NaN state update;
  * negative curvature (p^T A p <= 0) stops the recursion and is reported;
  * the reported relative residual is recomputed from ``b - A x`` at the end, so
    convergence is never claimed on a drifted recursive residual.

No autograd, no optimizer: every tensor is detached on entry (spec 0.4/0.5/36).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

MatVec = Callable[[torch.Tensor], torch.Tensor]

# Reasons reported in ``info['reason']``.
REASON_ZERO_RHS = "zero_rhs"
REASON_CONVERGED = "converged"
REASON_MAX_ITERS = "max_iterations"
REASON_NEGATIVE_CURVATURE = "negative_curvature"
REASON_NONFINITE_RHS = "nonfinite_rhs"
REASON_NONFINITE_X0 = "nonfinite_initial_guess"
REASON_NONFINITE_OPERATOR = "nonfinite_operator"
REASON_NONFINITE_PRECONDITIONER = "nonfinite_preconditioner"
REASON_NONFINITE_SCALAR = "nonfinite_scalar"
REASON_NONFINITE_ITERATE = "nonfinite_iterate"
REASON_NON_PD_PRECONDITIONER = "non_pd_preconditioner"
REASON_BREAKDOWN = "breakdown"
REASON_RESIDUAL_DRIFT = "recursive_residual_drift"


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flat inner product accumulated in float64.

    alpha and beta are ratios of sums over millions of entries; float32
    accumulation stalls the recursion orders of magnitude above the requested
    tolerance, so the reduction (not the vectors) is promoted.
    """
    return float(torch.sum(a * b, dtype=torch.float64))


def _norm(a: torch.Tensor) -> float:
    return math.sqrt(max(_dot(a, a), 0.0))


# ---------------------------------------------------------------------------
# Section 16: packed symmetric storage for the CUDA Fisher kernel
# ---------------------------------------------------------------------------
def symmetric_packed_length(dim: int) -> int:
    """Number of independent entries of a symmetric ``dim x dim`` block.

    Section 16: "only 21 independent values are needed for symmetric 6x6
    storage" - kernel E writes that packed form, halving both the atomic
    traffic and the per-Gaussian footprint.
    """
    if int(dim) < 1:
        raise ValueError("dim must be >= 1")
    return int(dim) * (int(dim) + 1) // 2


def _tril_index(dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.tril_indices(dim, dim, device=device)
    return idx[0], idx[1]


def unpack_symmetric(packed: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """``[..., L] -> [..., D, D]``, row-major lower triangle including diagonal.

    Order is (0,0), (1,0), (1,1), (2,0), ... which is the order a kernel
    accumulating a symmetric outer product naturally produces.
    """
    length = int(packed.shape[-1])
    if dim is None:
        dim = int((math.isqrt(8 * length + 1) - 1) // 2)
    if symmetric_packed_length(dim) != length:
        raise ValueError(f"{length} packed entries do not describe a symmetric {dim}x{dim} block")
    rows, cols = _tril_index(dim, packed.device)
    out = torch.zeros(packed.shape[:-1] + (dim, dim), dtype=packed.dtype, device=packed.device)
    out[..., rows, cols] = packed
    out[..., cols, rows] = packed  # diagonal is written twice with the same value
    return out


def pack_symmetric(blocks: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`unpack_symmetric` (symmetrizes first)."""
    if blocks.shape[-1] != blocks.shape[-2]:
        raise ValueError(f"expected square blocks, got {tuple(blocks.shape)}")
    sym = 0.5 * (blocks + blocks.transpose(-1, -2))
    rows, cols = _tril_index(int(blocks.shape[-1]), blocks.device)
    return sym[..., rows, cols]


def assemble_preconditioner_blocks(
    fisher_blocks: torch.Tensor | None = None,
    *,
    prior_blocks: torch.Tensor | None = None,
    lam: float = 0.0,
    damping: torch.Tensor | str | None = None,
    block_dim: int | None = None,
    fisher_offset: int = 0,
    num_blocks: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    marquardt_floor: float = 1.0e-12,
) -> torch.Tensor:
    """Build ``M_i = Lambda^-_ii + F_i + lambda D_i`` of section 16.

    ``fisher_blocks`` may be dense ``[N, f, f]`` or packed ``[N, f(f+1)/2]``.
    A pose-only 6x6 measurement Fisher is *embedded* at ``fisher_offset`` inside
    the 12D block rather than padded into a 12x12 measurement block: the
    velocity rows then keep exactly the prior's pose-velocity coupling, which is
    the only thing making a velocity observable at all through one frame.

    ``damping`` selects D: ``None``/``"identity"`` is plain Levenberg,
    ``"marquardt"`` scales by the diagonal of ``Lambda^- + F`` (unit invariant,
    which matters because metres and radians share the block).
    """
    if fisher_blocks is None and prior_blocks is None:
        raise ValueError("assemble_preconditioner_blocks needs a prior and/or a Fisher term")

    fisher = fisher_blocks
    if fisher is not None:
        if fisher.dim() == 2:
            fisher = unpack_symmetric(fisher)
        elif fisher.dim() != 3 or fisher.shape[-1] != fisher.shape[-2]:
            raise ValueError(
                f"fisher_blocks must be [N, f, f] or packed [N, f(f+1)/2], got {tuple(fisher_blocks.shape)}")

    reference = prior_blocks if prior_blocks is not None else fisher
    dtype = dtype if dtype is not None else reference.dtype
    device = device if device is not None else reference.device
    n = int(num_blocks) if num_blocks is not None else int(reference.shape[0])
    if block_dim is None:
        block_dim = int(prior_blocks.shape[-1]) if prior_blocks is not None else int(fisher.shape[-1])
    d = int(block_dim)

    m = torch.zeros((n, d, d), dtype=dtype, device=device)
    if prior_blocks is not None:
        if tuple(prior_blocks.shape) != (n, d, d):
            raise ValueError(
                f"prior_blocks must be [{n}, {d}, {d}], got {tuple(prior_blocks.shape)}")
        m += prior_blocks.to(dtype=dtype, device=device)
    if fisher is not None:
        f = int(fisher.shape[-1])
        off = int(fisher_offset)
        if off < 0 or off + f > d:
            raise ValueError(f"a {f}x{f} Fisher block does not fit at offset {off} in a {d}D block")
        if int(fisher.shape[0]) != n:
            raise ValueError(f"fisher_blocks has {fisher.shape[0]} blocks, expected {n}")
        sub = m[:, off:off + f, off:off + f]
        m[:, off:off + f, off:off + f] = sub + fisher.to(dtype=dtype, device=device)

    lam = float(lam)
    if lam != 0.0:
        eye = torch.eye(d, dtype=dtype, device=device)
        if damping is None or damping == "identity":
            m = m + lam * eye
        elif damping == "marquardt":
            scale = torch.diagonal(m, dim1=-2, dim2=-1).clamp_min(float(marquardt_floor))
            m = m + lam * torch.diag_embed(scale)
        elif torch.is_tensor(damping):
            dm = damping.to(dtype=dtype, device=device)
            if dm.shape == (n, d):
                m = m + lam * torch.diag_embed(dm)
            elif dm.shape == (n, d, d):
                m = m + lam * dm
            else:
                raise ValueError(f"damping tensor must be [{n}, {d}] or [{n}, {d}, {d}]")
        else:
            raise ValueError(f"unknown damping specification {damping!r}")
    return 0.5 * (m + m.transpose(-1, -2))


# ---------------------------------------------------------------------------
# Section 16: block-Jacobi / block-Fisher preconditioner
# ---------------------------------------------------------------------------
@dataclass
class BlockJacobiPreconditioner:
    """Applies ``M^-1 v`` blockwise through a cached batched Cholesky factor.

    Build it with :func:`block_jacobi_preconditioner`; call it like a function.
    ``cholesky`` holds the lower factors of the *repaired* blocks, so one
    ``torch.cholesky_solve`` serves every PCG iteration of one linearization -
    the factorization is computed once per relinearization, as section 20.3
    requires.
    """

    cholesky: torch.Tensor
    jitter: torch.Tensor
    jittered_mask: torch.Tensor
    eigenvalue_floored_mask: torch.Tensor
    identity_fallback_mask: torch.Tensor
    tries: int = 0

    @property
    def num_blocks(self) -> int:
        return int(self.cholesky.shape[0])

    @property
    def block_dim(self) -> int:
        return int(self.cholesky.shape[-1])

    @property
    def num_jittered(self) -> int:
        """Blocks that were not positive definite as given (section 34)."""
        return int(self.jittered_mask.sum().item())

    @property
    def num_eigenvalue_floored(self) -> int:
        """Blocks that jitter alone could not repair; eigenvalues were clamped."""
        return int(self.eigenvalue_floored_mask.sum().item())

    @property
    def num_identity_fallback(self) -> int:
        """Blocks replaced by a scaled identity - genuinely unobservable poses."""
        return int(self.identity_fallback_mask.sum().item())

    @property
    def repaired_mask(self) -> torch.Tensor:
        """Any block that was not usable exactly as given."""
        return self.jittered_mask | self.eigenvalue_floored_mask | self.identity_fallback_mask

    @property
    def num_repaired(self) -> int:
        return int(self.repaired_mask.sum().item())

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return self.apply(v)

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        """``M^-1 v`` for ``v`` of shape ``[N, D]`` (or anything with N*D entries)."""
        if v.numel() != self.num_blocks * self.block_dim:
            raise ValueError(
                f"preconditioner expects {self.num_blocks * self.block_dim} entries, got {v.numel()}")
        shape = v.shape
        rhs = v.reshape(self.num_blocks, self.block_dim, 1).to(
            dtype=self.cholesky.dtype, device=self.cholesky.device)
        out = torch.cholesky_solve(rhs, self.cholesky)
        return out.reshape(shape).to(dtype=v.dtype, device=v.device)

    def inverse_blocks(self) -> torch.Tensor:
        """Explicit ``[N, D, D]`` inverse blocks - diagnostics and tests only.

        Never call this on a full scene inside the estimator: section 0.7 bans
        materializing dense covariance, and this is exactly that for N blocks.
        """
        eye = torch.eye(self.block_dim, dtype=self.cholesky.dtype, device=self.cholesky.device)
        return torch.cholesky_solve(eye.expand(self.num_blocks, -1, -1), self.cholesky)

    def log_det(self) -> torch.Tensor:
        """``sum_i log det M_i`` from the cached factors (cheap evidence term)."""
        return 2.0 * torch.log(torch.diagonal(self.cholesky, dim1=-2, dim2=-1)).sum()

    def report(self) -> Dict[str, Any]:
        """Section 34 logging payload."""
        return {
            "num_blocks": self.num_blocks,
            "block_dim": self.block_dim,
            "num_jittered": self.num_jittered,
            "num_eigenvalue_floored": self.num_eigenvalue_floored,
            "num_identity_fallback": self.num_identity_fallback,
            "num_repaired": self.num_repaired,
            "jitter_max": float(self.jitter.max()) if self.jitter.numel() else 0.0,
            "jitter_mean": float(self.jitter.mean()) if self.jitter.numel() else 0.0,
            "cholesky_tries": int(self.tries),
        }


def block_jacobi_preconditioner(
    blocks: torch.Tensor,
    *,
    jitter: float = 1.0e-10,
    growth: float = 1.0e2,
    max_tries: int = 6,
    eigenvalue_floor: float = 1.0e-8,
    degenerate_fraction: float = 1.0e-12,
    symmetrize: bool = True,
) -> BlockJacobiPreconditioner:
    """Factorize ``blocks[N, D, D]`` into a block-Jacobi inverse operator.

    ``jitter`` is *relative*: the absolute amount added to a failing block is
    ``jitter * scale_i`` with ``scale_i`` the mean absolute diagonal of that
    block, so the repair is invariant to the units of the state (metres vs
    radians vs metres/second all coexist in the 12D error state of section 4).

    Repair ladder, each step counted separately:
      0. blocks with essentially no curvature at all (an occluded Gaussian with
         neither prior nor damping) are replaced by ``typical * I``, where
         ``typical`` is the scene's mean block scale.  Flooring them relatively
         instead would hand PCG a preconditioner that *amplifies* the one
         direction carrying no information;
      1. escalating jitter (``growth`` per try, at most ``max_tries``);
      2. symmetric eigenvalue flooring for whatever still fails;
      3. ``typical * I`` for the pathological remainder.

    In the IEKF this ladder should stay idle: ``M_i = Lambda^-_ii + F_i +
    lambda D_i`` is PD by construction through the prior and the LM damping, so
    a nonzero repair count is a diagnostic that something upstream is wrong.
    """
    if blocks.dim() != 3 or blocks.shape[-1] != blocks.shape[-2]:
        raise ValueError(f"blocks must have shape [N, D, D], got {tuple(blocks.shape)}")
    if blocks.shape[0] == 0:
        raise ValueError("block_jacobi_preconditioner needs at least one block")
    if not (jitter > 0.0 and growth > 1.0 and max_tries >= 1):
        raise ValueError("jitter must be > 0, growth > 1 and max_tries >= 1")

    # Detached and promoted to at least float32: a Cholesky of a near-singular
    # Fisher block in float16 is noise.
    mats = blocks.detach().clone()
    if mats.dtype not in (torch.float32, torch.float64):
        mats = mats.to(torch.float32)
    if not bool(torch.isfinite(mats).all()):
        raise ValueError("block_jacobi_preconditioner received non-finite blocks")
    if symmetrize:
        # Fisher blocks accumulated atomically over tiles are only symmetric up
        # to summation order; Cholesky silently reads the lower triangle, so the
        # asymmetry would become a hidden, order-dependent preconditioner.
        mats = 0.5 * (mats + mats.transpose(-1, -2))

    n, d = mats.shape[0], mats.shape[-1]
    device, dtype = mats.device, mats.dtype
    eye = torch.eye(d, dtype=dtype, device=device)

    total_jitter = torch.zeros(n, dtype=dtype, device=device)
    jittered = torch.zeros(n, dtype=torch.bool, device=device)
    floored = torch.zeros(n, dtype=torch.bool, device=device)
    fallback = torch.zeros(n, dtype=torch.bool, device=device)

    raw_scale = torch.diagonal(mats, dim1=-2, dim2=-1).abs().mean(dim=-1)
    positive = raw_scale[raw_scale > 0]
    typical = float(positive.mean()) if positive.numel() else 1.0
    if not math.isfinite(typical) or typical <= 0.0:
        typical = 1.0

    degenerate = raw_scale <= typical * float(degenerate_fraction)
    if bool(degenerate.any()):
        idx0 = degenerate.nonzero(as_tuple=True)[0]
        mats[idx0] = typical * eye
        fallback[idx0] = True
    scale = torch.diagonal(mats, dim1=-2, dim2=-1).abs().mean(dim=-1).clamp_min(
        typical * float(degenerate_fraction))

    chol, info = torch.linalg.cholesky_ex(mats)
    bad = info > 0
    tries = 0
    current = float(jitter)
    while bool(bad.any()) and tries < int(max_tries):
        idx = bad.nonzero(as_tuple=True)[0]
        add = current * scale[idx]
        mats[idx] = mats[idx] + add.view(-1, 1, 1) * eye
        total_jitter[idx] = total_jitter[idx] + add
        jittered[idx] = True
        sub_chol, sub_info = torch.linalg.cholesky_ex(mats[idx])
        chol[idx] = sub_chol
        bad = torch.zeros_like(bad)
        bad[idx] = sub_info > 0
        current *= float(growth)
        tries += 1

    if bool(bad.any()):
        # Diagonal jitter cannot fix a block that is indefinite rather than
        # merely ill-conditioned; clamp the spectrum instead.  The floor is tied
        # to max(own spectrum, scene typical) so a null direction yields a large
        # but bounded - never explosive - preconditioner entry.
        idx = bad.nonzero(as_tuple=True)[0]
        evals, evecs = torch.linalg.eigh(mats[idx])
        reference = torch.maximum(
            evals.abs().amax(dim=-1),
            torch.full_like(scale[idx], typical))
        floor = float(eigenvalue_floor) * reference
        evals = torch.maximum(evals, floor.view(-1, 1))
        fixed = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
        fixed = 0.5 * (fixed + fixed.transpose(-1, -2))
        sub_chol, sub_info = torch.linalg.cholesky_ex(fixed)
        chol[idx] = sub_chol
        floored[idx] = True
        still_bad = sub_info > 0
        if bool(still_bad.any()):
            fb = idx[still_bad.nonzero(as_tuple=True)[0]]
            chol[fb] = math.sqrt(typical) * eye
            fallback[fb] = True
            floored[fb] = False

    return BlockJacobiPreconditioner(
        cholesky=chol,
        jitter=total_jitter,
        jittered_mask=jittered,
        eigenvalue_floored_mask=floored,
        identity_fallback_mask=fallback,
        tries=tries,
    )


# ---------------------------------------------------------------------------
# Section 15: preconditioned conjugate gradient
# ---------------------------------------------------------------------------
def _info(
    *,
    iterations: int,
    relative_residual: float,
    converged: bool,
    reason: str,
    history: List[float],
    rhs_norm: float,
    matvecs: int,
    recursive_relative_residual: float,
    negative_curvature: bool = False,
) -> Dict[str, Any]:
    return {
        "iterations": int(iterations),
        "relative_residual": float(relative_residual),
        "recursive_relative_residual": float(recursive_relative_residual),
        "converged": bool(converged),
        "reason": str(reason),
        "residual_history": list(history),
        "rhs_norm": float(rhs_norm),
        "matvecs": int(matvecs),
        "negative_curvature": bool(negative_curvature),
    }


def pcg(
    A_mv: MatVec,
    b: torch.Tensor,
    M_inv: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    max_iters: int = 30,
    tol: float = 1.0e-6,
    x0: Optional[torch.Tensor] = None,
    *,
    check_finite: bool = True,
    verify_final: bool = True,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Solve ``A x = b`` with preconditioned CG, matrix-free (section 15).

    ``A_mv`` is any callable ``v -> A v``; ``A`` is never formed.  ``M_inv`` is
    a callable ``r -> M^-1 r`` (typically a
    :class:`BlockJacobiPreconditioner`); ``None`` means no preconditioning.

    Stops on ``||b - A x|| / ||b|| < tol``.  Returns ``(x, info)`` where ``x``
    is always finite and ``info`` carries the diagnostics of section 34:
    ``iterations``, ``relative_residual``, ``converged``, ``reason`` and the
    per-iteration ``residual_history``.
    """
    if not callable(A_mv):
        raise TypeError("A_mv must be callable")
    if M_inv is not None and not callable(M_inv):
        raise TypeError("M_inv must be callable or None")
    if int(max_iters) < 0:
        raise ValueError("max_iters must be non-negative")
    if not (float(tol) > 0.0):
        raise ValueError("tol must be strictly positive")
    if not torch.is_tensor(b):
        raise TypeError("b must be a torch.Tensor")

    max_iters = int(max_iters)
    tol = float(tol)
    rhs = b.detach()
    history: List[float] = []
    matvecs = 0

    if not bool(torch.isfinite(rhs).all()):
        return torch.zeros_like(rhs), _info(
            iterations=0, relative_residual=float("inf"), converged=False,
            reason=REASON_NONFINITE_RHS, history=history, rhs_norm=float("nan"),
            matvecs=0, recursive_relative_residual=float("inf"))

    # Exact zero RHS: the solution is exactly zero for any SPD A, so iterating
    # would only divide by ||b|| = 0.  Checked on the values, not on the norm,
    # because a norm can underflow to 0 for a tiny but nonzero b.
    if not bool(rhs.any()):
        return torch.zeros_like(rhs), _info(
            iterations=0, relative_residual=0.0, converged=True,
            reason=REASON_ZERO_RHS, history=[0.0], rhs_norm=0.0, matvecs=0,
            recursive_relative_residual=0.0)

    b_norm = _norm(rhs)
    if not math.isfinite(b_norm) or b_norm <= 0.0:
        return torch.zeros_like(rhs), _info(
            iterations=0, relative_residual=float("inf"), converged=False,
            reason=REASON_NONFINITE_RHS, history=history, rhs_norm=b_norm,
            matvecs=0, recursive_relative_residual=float("inf"))

    def _apply(op: Callable[[torch.Tensor], torch.Tensor], v: torch.Tensor, what: str) -> torch.Tensor:
        out = op(v)
        if not torch.is_tensor(out):
            raise TypeError(f"{what} must return a torch.Tensor")
        if out.shape != v.shape:
            raise ValueError(
                f"{what} returned shape {tuple(out.shape)}, expected {tuple(v.shape)}")
        return out.detach()

    if x0 is None:
        x = torch.zeros_like(rhs)
        r = rhs.clone()
    else:
        if x0.shape != rhs.shape:
            raise ValueError(
                f"x0 shape {tuple(x0.shape)} does not match b shape {tuple(rhs.shape)}")
        if not bool(torch.isfinite(x0).all()):
            return torch.zeros_like(rhs), _info(
                iterations=0, relative_residual=float("inf"), converged=False,
                reason=REASON_NONFINITE_X0, history=history, rhs_norm=b_norm,
                matvecs=0, recursive_relative_residual=float("inf"))
        x = x0.detach().clone().to(dtype=rhs.dtype, device=rhs.device)
        ax = _apply(A_mv, x, "A_mv")
        matvecs += 1
        if check_finite and not bool(torch.isfinite(ax).all()):
            return torch.zeros_like(rhs), _info(
                iterations=0, relative_residual=float("inf"), converged=False,
                reason=REASON_NONFINITE_OPERATOR, history=history, rhs_norm=b_norm,
                matvecs=matvecs, recursive_relative_residual=float("inf"))
        r = rhs - ax

    rel = _norm(r) / b_norm
    history.append(rel)
    if not math.isfinite(rel):
        return torch.zeros_like(rhs), _info(
            iterations=0, relative_residual=float("inf"), converged=False,
            reason=REASON_NONFINITE_OPERATOR, history=history, rhs_norm=b_norm,
            matvecs=matvecs, recursive_relative_residual=float("inf"))

    def _finalize(iters: int, reason: str, converged: bool, recursive_rel: float,
                  neg_curv: bool = False) -> tuple[torch.Tensor, Dict[str, Any]]:
        """One extra matvec buys an honest residual (never claim a drifted one)."""
        nonlocal matvecs
        true_rel = recursive_rel
        if verify_final and reason not in (REASON_NONFINITE_OPERATOR,):
            ax_end = _apply(A_mv, x, "A_mv")
            matvecs += 1
            if bool(torch.isfinite(ax_end).all()):
                true_rel = _norm(rhs - ax_end) / b_norm
        final_converged = bool(converged)
        final_reason = reason
        if converged and math.isfinite(true_rel) and true_rel >= tol:
            final_converged = False
            final_reason = REASON_RESIDUAL_DRIFT
        return x, _info(
            iterations=iters, relative_residual=true_rel, converged=final_converged,
            reason=final_reason, history=history, rhs_norm=b_norm, matvecs=matvecs,
            recursive_relative_residual=recursive_rel, negative_curvature=neg_curv)

    if rel < tol:
        return _finalize(0, REASON_CONVERGED, True, rel)

    z = _apply(M_inv, r, "M_inv") if M_inv is not None else r.clone()
    if check_finite and not bool(torch.isfinite(z).all()):
        return _finalize(0, REASON_NONFINITE_PRECONDITIONER, False, rel)
    rz = _dot(r, z)
    if not math.isfinite(rz):
        return _finalize(0, REASON_NONFINITE_SCALAR, False, rel)
    if rz <= 0.0:
        # <r, M^-1 r> <= 0 means the preconditioner is not positive definite;
        # continuing would search along a non-descent direction.
        return _finalize(0, REASON_NON_PD_PRECONDITIONER, False, rel)
    p = z.clone()

    iters = 0
    reason = REASON_MAX_ITERS
    converged = False
    neg_curv = False
    for iters in range(1, max_iters + 1):
        ap = _apply(A_mv, p, "A_mv")
        matvecs += 1
        if check_finite and not bool(torch.isfinite(ap).all()):
            return _finalize(iters - 1, REASON_NONFINITE_OPERATOR, False, rel)
        p_ap = _dot(p, ap)
        if not math.isfinite(p_ap):
            return _finalize(iters - 1, REASON_NONFINITE_SCALAR, False, rel)
        if p_ap <= 0.0:
            # Negative/zero curvature: A is not positive definite along p.  The
            # iterate so far is still a valid descent step, so it is returned
            # rather than discarded (and the LM damping of section 14 is what
            # should grow in response).
            neg_curv = True
            return _finalize(iters - 1, REASON_NEGATIVE_CURVATURE, False, rel, neg_curv=True)

        alpha = rz / p_ap
        if not math.isfinite(alpha):
            return _finalize(iters - 1, REASON_NONFINITE_SCALAR, False, rel)

        x_prev = x
        x = x + alpha * p
        if check_finite and not bool(torch.isfinite(x).all()):
            x = x_prev  # never hand a NaN state update back to the IEKF
            return _finalize(iters - 1, REASON_NONFINITE_ITERATE, False, rel)
        r = r - alpha * ap

        rel = _norm(r) / b_norm
        history.append(rel)
        if not math.isfinite(rel):
            return _finalize(iters, REASON_NONFINITE_SCALAR, False, rel)
        if rel < tol:
            converged = True
            reason = REASON_CONVERGED
            break

        z = _apply(M_inv, r, "M_inv") if M_inv is not None else r
        if check_finite and not bool(torch.isfinite(z).all()):
            return _finalize(iters, REASON_NONFINITE_PRECONDITIONER, False, rel)
        rz_new = _dot(r, z)
        if not math.isfinite(rz_new):
            return _finalize(iters, REASON_NONFINITE_SCALAR, False, rel)
        if rz_new <= 0.0:
            # Either M^-1 lost positive definiteness or r underflowed to zero;
            # in both cases the recursion has broken down and must not continue.
            return _finalize(iters, REASON_BREAKDOWN, False, rel)
        beta = rz_new / rz
        rz = rz_new
        p = z + beta * p

    return _finalize(iters, reason, converged, rel, neg_curv=neg_curv)


__all__ = [
    "BlockJacobiPreconditioner",
    "block_jacobi_preconditioner",
    "assemble_preconditioner_blocks",
    "pack_symmetric",
    "unpack_symmetric",
    "symmetric_packed_length",
    "pcg",
    "REASON_ZERO_RHS",
    "REASON_CONVERGED",
    "REASON_MAX_ITERS",
    "REASON_NEGATIVE_CURVATURE",
    "REASON_NONFINITE_RHS",
    "REASON_NONFINITE_X0",
    "REASON_NONFINITE_OPERATOR",
    "REASON_NONFINITE_PRECONDITIONER",
    "REASON_NONFINITE_SCALAR",
    "REASON_NONFINITE_ITERATE",
    "REASON_NON_PD_PRECONDITIONER",
    "REASON_BREAKDOWN",
    "REASON_RESIDUAL_DRIFT",
]
