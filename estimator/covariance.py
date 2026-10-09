"""Block-marginal covariance / information propagation for the sparse IEKF.

Specification sections 5 and 17.  The exact posterior covariance becomes dense
as soon as two Gaussians are co-visible in one pixel, so between frames we only
ever store, propagate and invert PER-GAUSSIAN 12x12 BLOCKS.  Nothing in this
module ever allocates a (12N x 12N) object: every routine is batched over the
leading Gaussian axis.

Error-state ordering (spec section 4):

    delta = [ delta_mu (0:3) | delta_phi (3:6) | delta_v (6:9) | delta_omega (9:12) ]

so indices 0..5 are the POSE sub-block and 6..11 the velocity sub-block.

No autograd, no optimizer: closed-form linear algebra only (spec 0.4/0.5/36).
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple, Union

import torch

ERROR_DIM = 12
POSE_DIM = 6

ScalarLike = Union[float, int, torch.Tensor]


@dataclasses.dataclass
class BlockInverseDiagnostics:
    """What ``invert_blocks`` had to do to make every block invertible.

    A non-zero ``n_eigen_clamped`` means the SPD assumption was violated for
    that many blocks and the result is a regularized pseudo-inverse, not the
    true inverse.  ``n_non_finite`` counts blocks that arrived carrying NaN/Inf
    and could not be inverted at all; those come back as exact zeros so the
    corruption cannot propagate through ``apply_prior``.  Callers must surface
    all of this (spec section 34 diagnostics) rather than treat the output as
    exact.
    """

    n_blocks: int = 0
    n_jittered: int = 0
    n_eigen_clamped: int = 0
    n_non_finite: int = 0
    max_relative_jitter: float = 0.0

    @property
    def n_fallback(self) -> int:
        """Number of blocks that left the Cholesky path entirely."""
        return self.n_eigen_clamped + self.n_non_finite

    @property
    def ok(self) -> bool:
        return (
            self.n_jittered == 0
            and self.n_eigen_clamped == 0
            and self.n_non_finite == 0
        )


def _symmetrize(A: torch.Tensor) -> torch.Tensor:
    """Average with the transpose.

    Covariance/information blocks are symmetric by construction, but
    ``F P F^T`` accumulates asymmetric round-off that compounds over hundreds
    of frames and eventually breaks Cholesky; this is the cheap standard guard.
    """
    return 0.5 * (A + A.transpose(-1, -2))


def _check_blocks(A: torch.Tensor, dim: Optional[int], name: str) -> int:
    if A.dim() != 3 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"{name} must be [N, d, d], got {tuple(A.shape)}")
    if dim is not None and A.shape[-1] != dim:
        raise ValueError(f"{name} must be [N, {dim}, {dim}], got {tuple(A.shape)}")
    return A.shape[0]


def _as_per_gaussian(
    value: ScalarLike, n: int, device: torch.device, dtype: torch.dtype, name: str
) -> torch.Tensor:
    """Broadcast a scalar or ``[N]`` tensor to ``[N]`` on the target device."""
    if isinstance(value, torch.Tensor):
        out = value.to(device=device, dtype=dtype).reshape(-1)
        if out.numel() == 1:
            out = out.expand(n)
        elif out.numel() != n:
            raise ValueError(f"{name} must be scalar or have {n} entries, got {tuple(value.shape)}")
    else:
        out = torch.full((n,), float(value), device=device, dtype=dtype)
    return out


def covariance_predict(P: torch.Tensor, F: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Block covariance prediction ``P^- = F P^+ F^T + Q`` (spec section 5).

    Operates on the full 12x12 block so the pose/velocity cross-covariances
    that make the constant-twist prior informative are preserved; a
    block-diagonal (pose-only + velocity-only) shortcut would throw away
    exactly the correlation the twist prior is there to express.
    """
    n = _check_blocks(P, ERROR_DIM, "P")
    if _check_blocks(F, ERROR_DIM, "F") != n or _check_blocks(Q, ERROR_DIM, "Q") != n:
        raise ValueError("P, F and Q must share the same leading Gaussian count")
    propagated = F @ _symmetrize(P) @ F.transpose(-1, -2)
    return _symmetrize(propagated + Q)


def make_process_noise(
    n: int,
    dt: ScalarLike,
    sigma_pos: ScalarLike,
    sigma_rot: ScalarLike,
    sigma_vel: ScalarLike,
    sigma_omega: ScalarLike,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Per-Gaussian process noise ``Q`` for a step of real duration ``dt``.

    dt SCALING CHOICE (documented as required by the contract): all four sigmas
    are continuous-time noise DENSITIES in units of [value] / sqrt(second), and
    the block is the exact discretization of a continuous white-noise model
    over the interval ``dt``:

      * ``sigma_vel`` / ``sigma_omega`` are white ACCELERATION densities acting
        on v / omega.  Integrating them over the step gives the classical
        white-noise-acceleration triple
            Q_vv = q dt,   Q_mu_v = q dt^2 / 2,   Q_mu_mu = q dt^3 / 3,
        which is why the pose/velocity CROSS terms below are non-zero.  Keeping
        them is mandated by spec section 5 ("keep pose-velocity
        cross-covariances inside every 12x12 block"); dropping them would make
        the prior claim position and velocity errors are independent, which is
        false for an integrated random walk and makes the filter overconfident.
      * ``sigma_pos`` / ``sigma_rot`` are direct random-walk densities on the
        pose itself, modelling non-rigid jitter that no velocity explains, and
        contribute ``sigma^2 dt`` on the diagonal.

    Consequence of the density convention: Q is additive over sub-intervals in
    its random-walk part, so halving the frame rate does not change the implied
    continuous-time model.  Never pass per-frame variances here; pass densities.

    Rotational note: the same scalar density is applied to all three components
    of ``delta_phi``.  That is isotropic in the tangent space at the current
    nominal rotation, which is the correct place to inject SO(3) noise under
    the left retraction.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    device = torch.device("cpu") if device is None else torch.device(device)
    dtype = torch.get_default_dtype() if dtype is None else dtype

    step = _as_per_gaussian(dt, n, device, dtype, "dt")
    if bool(torch.any(step < 0)):
        raise ValueError("dt must be non-negative")
    s_pos = _as_per_gaussian(sigma_pos, n, device, dtype, "sigma_pos")
    s_rot = _as_per_gaussian(sigma_rot, n, device, dtype, "sigma_rot")
    s_vel = _as_per_gaussian(sigma_vel, n, device, dtype, "sigma_vel")
    s_omg = _as_per_gaussian(sigma_omega, n, device, dtype, "sigma_omega")
    for name, sigma in (
        ("sigma_pos", s_pos),
        ("sigma_rot", s_rot),
        ("sigma_vel", s_vel),
        ("sigma_omega", s_omg),
    ):
        if bool(torch.any(sigma < 0)):
            raise ValueError(f"{name} must be non-negative")

    q_vel = s_vel * s_vel
    q_omg = s_omg * s_omg
    dt1 = step
    dt2 = step * step
    dt3 = dt2 * step

    mu_mu = s_pos * s_pos * dt1 + q_vel * dt3 / 3.0
    mu_v = q_vel * dt2 / 2.0
    v_v = q_vel * dt1
    phi_phi = s_rot * s_rot * dt1 + q_omg * dt3 / 3.0
    phi_w = q_omg * dt2 / 2.0
    w_w = q_omg * dt1

    eye3 = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3)
    Q = torch.zeros(n, ERROR_DIM, ERROR_DIM, device=device, dtype=dtype)

    def _set(row: int, col: int, scale: torch.Tensor) -> None:
        Q[:, row : row + 3, col : col + 3] = scale.view(n, 1, 1) * eye3

    _set(0, 0, mu_mu)
    _set(3, 3, phi_phi)
    _set(6, 6, v_v)
    _set(9, 9, w_w)
    _set(0, 6, mu_v)
    _set(6, 0, mu_v)
    _set(3, 9, phi_w)
    _set(9, 3, phi_w)
    return Q


def invert_blocks(
    P: torch.Tensor,
    jitter_steps: int = 6,
    base_relative_jitter: float = 1e-12,
    eigenvalue_floor_ratio: float = 1e-12,
) -> Tuple[torch.Tensor, BlockInverseDiagnostics]:
    """Batched SPD inverse of per-Gaussian blocks, covariance <-> information.

    The map is an involution: ``P -> Lambda`` and ``Lambda -> P`` are the same
    routine, which is why there is no separate ``invert_information``.

    Path: symmetrize -> batched Cholesky (``cholesky_ex``, which reports failure
    through ``info`` instead of raising so one bad Gaussian cannot abort a
    million-block batch) -> escalating RELATIVE jitter on the still-failing
    blocks -> eigenvalue clamping for whatever is left.  Jitter is relative to
    each block's mean diagonal because the 12x12 block mixes metres, radians,
    m/s and rad/s, so an absolute epsilon would be meaningless in some rows.

    The eigenvalue fallback produces a regularized pseudo-inverse, NOT the true
    inverse, so its use is counted in the returned diagnostics rather than
    hidden.  Blocks carrying NaN/Inf are quarantined before any factorization
    (LAPACK's batched eigensolver RAISES on one, which would abort the whole
    batch) and returned as zeros, counted in ``n_non_finite``.
    """
    n = _check_blocks(P, None, "P")
    dim = P.shape[-1]
    A = _symmetrize(P)
    out = torch.empty_like(A)
    eye = torch.eye(dim, device=A.device, dtype=A.dtype)
    diagnostics = BlockInverseDiagnostics(n_blocks=n)

    # Quarantine non-finite blocks first.  torch.linalg.eigh raises on a NaN
    # input, so leaving one in the batch would take down every other Gaussian
    # with it -- precisely the failure this routine is built to avoid.  A zero
    # inverse is returned so the corruption cannot spread through apply_prior.
    finite = torch.isfinite(A).all(dim=-1).all(dim=-1)
    if not bool(finite.all()):
        bad = torch.nonzero(~finite, as_tuple=False).squeeze(-1)
        diagnostics.n_non_finite = int(bad.numel())
        out[bad] = 0.0
        A[bad] = eye  # keep the batched kernels below well defined

    # Per-block scale for the relative jitter.  A block with no scale at all
    # (an all-zero information block, i.e. a Gaussian carrying no information)
    # gets an ABSOLUTE unit scale instead: a relative jitter would underflow to
    # a denormal, Cholesky would still report success on it, and the solve
    # would silently return NaN flagged as a mere jitter.  Unit scale yields a
    # finite, enormous covariance, which is the right reading of zero
    # information.
    tiny = torch.finfo(A.dtype).tiny
    scale = A.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1)
    scale = torch.where(scale > tiny, scale, torch.ones_like(scale))

    pending = finite.clone()

    for attempt in range(jitter_steps + 1):
        idx = torch.nonzero(pending, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            break
        block = A.index_select(0, idx)
        rel = 0.0
        if attempt > 0:
            rel = base_relative_jitter * (10.0 ** (attempt - 1))
            block = block + (rel * scale.index_select(0, idx)).view(-1, 1, 1) * eye
        chol, info = torch.linalg.cholesky_ex(block)
        ok = info == 0
        if bool(ok.any()):
            good = idx[ok]
            rhs = eye.expand(int(good.numel()), dim, dim)
            solved = torch.cholesky_solve(rhs, chol[ok])
            # A factorization that "succeeds" on a numerically degenerate block
            # can still overflow in the triangular solve; such a block has NOT
            # been inverted and must keep escalating instead of being accepted.
            usable = torch.isfinite(solved).all(dim=-1).all(dim=-1)
            if bool(usable.any()):
                accepted = good[usable]
                out[accepted] = solved[usable]
                pending[accepted] = False
                if attempt > 0:
                    diagnostics.n_jittered += int(usable.sum())
                    diagnostics.max_relative_jitter = max(diagnostics.max_relative_jitter, rel)

    remaining = torch.nonzero(pending, as_tuple=False).squeeze(-1)
    if remaining.numel() > 0:
        evals, evecs = torch.linalg.eigh(A.index_select(0, remaining))
        # Same absolute floor argument as above: an all-zero block has no
        # eigenvalue scale to be relative to, so 1/clamped would be inf.
        amax = evals.abs().amax(dim=-1, keepdim=True)
        floor = eigenvalue_floor_ratio * torch.where(amax > tiny, amax, torch.ones_like(amax))
        clamped = evals.clamp_min(floor)
        out[remaining] = evecs @ torch.diag_embed(1.0 / clamped) @ evecs.transpose(-1, -2)
        diagnostics.n_eigen_clamped = int(remaining.numel())

    return _symmetrize(out), diagnostics


@torch.no_grad()
def project_psd(P: torch.Tensor, floor_ratio: float | None = None,
                chunk: int = 2048,
                storage_dtype: torch.dtype | None = None) -> Tuple[torch.Tensor, dict]:
    """Force each block to be symmetric positive definite.

    Why this is needed rather than merely tidy. The posterior information is
    ``Lambda^+ = Lambda^- + F``, and the measurement Fisher F is strongly
    anisotropic on a real scene: a well-seen translation direction gets ~1e6
    while an unobserved one gets ~1e0. The block condition number then exceeds
    what float32 can invert, and the returned "covariance" acquires a NEGATIVE
    eigenvalue. Measured on a real USB sequence, the posterior minimum
    eigenvalue decayed 3.2e-9 -> 2.8e-9 -> -3.8e-4 over three frames; inverting
    that gave a prior information with a -9.5e4 eigenvalue, so the prior ENERGY
    went negative and the filter happily "reduced" the MAP objective to
    -15949 by moving along the indefinite direction.

    A covariance with a negative eigenvalue asserts negative variance, which is
    not a small numerical blemish -- it inverts the sign of the prior's opinion.
    So the eigenvalues are floored at a relative level and the block rebuilt.
    The number of blocks touched is returned, not hidden.

    The default floor is tied to the STORAGE dtype's epsilon, because a PSD
    matrix whose eigenvalue spread exceeds 1/eps cannot survive being stored at
    all: rounding a block with eigenvalues {1.0, 2.8e-9} to float32 (eps 1.2e-7)
    produces an indefinite matrix with no arithmetic involved. Claiming a
    variance that many orders below the largest is false precision, so the floor
    rounds that claim up rather than letting the next frame inherit a negative
    eigenvalue. This inflates uncertainty, which is the safe direction and the
    one section 5.2 asks for.
    """
    target = storage_dtype if storage_dtype is not None else P.dtype
    if floor_ratio is None:
        eps = float(torch.finfo(target).eps) if target.is_floating_point else 1e-9
        floor_ratio = max(32.0 * eps, 1e-12)
    A = _symmetrize(P).double()
    # Work in CORRELATION form. The 12x12 block mixes metres^2, radians^2,
    # (m/s)^2 and (rad/s)^2, so a floor relative to the block's largest
    # eigenvalue is a floor in whichever unit happens to be largest. With a
    # velocity variance of 1.0 and a well-measured position variance of 1e-6,
    # such a floor inflated the position uncertainty roughly fourfold and the
    # filter visibly lagged a constant-velocity target. Normalizing by the
    # diagonal first makes the floor dimensionless: it removes only genuinely
    # degenerate CORRELATION directions and leaves every variance scale intact.
    d = A.diagonal(dim1=-2, dim2=-1).clamp_min(0.0).sqrt()
    # A zero diagonal means that coordinate carries no variance at all; using 1
    # keeps the normalization well defined and leaves the row/column at zero.
    dz = d.clamp_min(1e-300)
    C = A / (dz.unsqueeze(2) * dz.unsqueeze(1))
    lam_list, vec_list = [], []
    for s in range(0, C.shape[0], chunk):
        l_, v_ = torch.linalg.eigh(C[s:s + chunk])
        lam_list.append(l_)
        vec_list.append(v_)
    lam = torch.cat(lam_list)
    vec = torch.cat(vec_list)
    # In correlation form the eigenvalues are O(1) by construction, so the
    # floor is simply the rounding bound of the storage dtype.
    floor = float(floor_ratio)
    bad = lam < floor
    n_bad = int(bad.any(dim=1).sum())
    n_negative = int((lam < 0).any(dim=1).sum())
    lam_fixed = lam.clamp_min(floor)
    Cf = torch.einsum("bik,bk,bjk->bij", vec, lam_fixed, vec)
    out = Cf * (dz.unsqueeze(2) * dz.unsqueeze(1))
    out = _symmetrize(out).to(P.dtype)
    return out, {"blocks_floored": n_bad,
                 "blocks_with_negative_eigenvalue": n_negative,
                 "min_correlation_eigenvalue_before": float(lam.min()),
                 "min_correlation_eigenvalue_after": float(lam_fixed.min()),
                 "floor_ratio": floor}


def approximate_posterior_information(
    Lambda_prior: torch.Tensor, fisher_pose: torch.Tensor
) -> torch.Tensor:
    """Spec section 17 approximate posterior information ``Lambda^+ ~= Lambda^- + F``.

    ``fisher_pose`` is the per-Gaussian measurement Fisher block
    ``F_i = sum_p H_{p,i}^T R_p^{-1} H_{p,i}`` (spec section 16).  It is a POSE
    block (indices 0..5 = [delta_mu, delta_phi]) because the RGB-D observation
    depends on the state only through pose: velocity is unobserved within a
    single frame and keeps whatever information the constant-twist prior gave
    it.  Passing a full 12x12 Fisher is also accepted for callers that solve
    directly in 12D.

    EXPLICIT APPROXIMATION: within one frame solve the Gaussians ARE coupled
    (co-visible contributors share a pixel), and the exact posterior
    information therefore has off-diagonal Gaussian-to-Gaussian blocks.  Adding
    only each Gaussian's own Fisher block DROPS all cross-Gaussian covariance
    when the state is carried to the next frame.  That is the deliberate
    scalable baseline of spec section 17: exact coupling is retained inside the
    frame through the matrix-free JVP/VJP solve, and only the 12x12 marginals
    survive between frames.  The consequence is a mildly OPTIMISTIC (too
    confident) marginal, since the discarded cross terms can only reduce the
    information attributable to an individual Gaussian.
    """
    n = _check_blocks(Lambda_prior, ERROR_DIM, "Lambda_prior")
    if fisher_pose.dim() != 3 or fisher_pose.shape[0] != n:
        raise ValueError(
            f"fisher_pose must be [{n}, d, d], got {tuple(fisher_pose.shape)}"
        )
    dim = fisher_pose.shape[-1]
    if fisher_pose.shape[-2] != dim or dim not in (POSE_DIM, ERROR_DIM):
        raise ValueError(
            f"fisher_pose must be [N, 6, 6] or [N, 12, 12], got {tuple(fisher_pose.shape)}"
        )

    Lambda_post = _symmetrize(Lambda_prior).clone()
    fisher = _symmetrize(fisher_pose).to(device=Lambda_post.device, dtype=Lambda_post.dtype)
    Lambda_post[:, :dim, :dim] = Lambda_post[:, :dim, :dim] + fisher
    return Lambda_post


def apply_prior(Lambda: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Block mat-vec ``(Lambda v)_i = Lambda_i v_i`` for the PCG normal operator.

    The prior term of the normal equations is block diagonal over Gaussians, so
    this is an einsum over ``[N, 12, 12] x [N, 12]`` and never forms the
    ``12N x 12N`` operator (spec rule 7).
    """
    n = _check_blocks(Lambda, None, "Lambda")
    dim = Lambda.shape[-1]
    if v.shape != (n, dim):
        raise ValueError(f"v must be [{n}, {dim}], got {tuple(v.shape)}")
    return torch.einsum("nij,nj->ni", Lambda, v)
