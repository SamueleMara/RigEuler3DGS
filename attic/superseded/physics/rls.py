"""Constrained weighted / recursive least squares for joint parameter ID.

Reference: CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md section 32.4.

Spec 32.4 asks for "constrained weighted least squares / recursive least
squares, not Adam", with m > 0, J_e > 0, b >= 0, and it requires the regressor
conditioning / persistent excitation to be reported and confident parameters to
be withheld when the regressor is rank deficient.

Method used here -- stated explicitly because "constrained" is often faked with a
penalty or a clamp after an unconstrained fit:

    The problem is the convex box-constrained QP

        min_theta  0.5 || W^{1/2} (Phi theta - u) ||^2     s.t.  l <= theta <= h

    solved by a PROJECTED-NEWTON ACTIVE-SET iteration on the (at most 4x4)
    system: variables sitting at a bound whose gradient points out of the box are
    pinned, the remaining free block is solved EXACTLY by an SVD least-squares
    solve, and the step is accepted along the projected arc with a backtracking
    line search; a projected-gradient step with the exact quadratic step length
    is the fallback that guarantees progress.  Termination is the exact KKT test
    (zero gradient on free variables, non-negative/non-positive multipliers at
    the lower/upper bounds).  No gradient-descent optimizer, no Adam, no
    autograd anywhere -- consistent with spec rules 0.4/0.5 and section 36.

Everything is tiny (p <= 4), so the dense p x p algebra here is unrelated to the
"never materialize a dense H" rule, which is about the per-Gaussian measurement
Jacobian.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import torch
from torch import Tensor

# A parameter the active set pushed onto its bound must be REPORTED at the bound:
# handing -1e-18 back for b would read as negative damping downstream.
_SNAP_REL = 1.0e-12


class WLSResult(NamedTuple):
    """Constrained WLS solution plus the conditioning report demanded by 32.4."""

    params: Tensor
    confident: bool
    rank_deficient: bool
    rank: int
    condition_number: float
    excitation: float
    singular_values: Tensor
    active_lower: Tensor
    active_upper: Tensor
    kkt_residual: float
    residual_rms: float
    iterations: int
    converged: bool


def _solve_dtype(dtype: torch.dtype) -> torch.dtype:
    """Run the tiny solve in float64 even for float32 inputs.

    The active-set decisions are sign tests on a gradient; in float32 the ~7
    significant digits of a squared-conditioning problem can flip those signs and
    return a different active set. Upcasting p<=4 columns costs nothing.
    """
    return torch.float64 if dtype in (torch.float16, torch.bfloat16, torch.float32) else dtype


def _lstsq_svd(A: Tensor, b: Tensor, rcond: float) -> Tensor:
    """Minimum-norm least-squares solve via SVD.

    torch.linalg.lstsq's CUDA driver requires full rank; the SVD path behaves
    identically on CPU and GPU and gives the min-norm answer for the rank
    deficient case instead of NaNs.
    """
    if A.shape[1] == 0:
        return torch.zeros(0, dtype=A.dtype, device=A.device)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    cutoff = rcond * float(S[0]) if S.numel() and float(S[0]) > 0.0 else 0.0
    inv = torch.where(S > cutoff, 1.0 / S, torch.zeros_like(S))
    return Vh.transpose(-2, -1) @ (inv * (U.transpose(-2, -1) @ b))


def _box_qp(
    design: Tensor,
    target: Tensor,
    lower: Tensor,
    upper: Tensor,
    *,
    rcond: float,
    max_iter: int = 64,
) -> tuple[Tensor, int, float, bool]:
    """Projected-Newton active set on 0.5||design theta - target||^2 over a box."""
    eps = torch.finfo(design.dtype).eps
    theta = torch.clamp(_lstsq_svd(design, target, rcond), lower, upper)
    scale = float(torch.linalg.vector_norm(design.transpose(-2, -1) @ target, ord=float("inf")))
    kkt_tol = (eps ** 0.75) * max(1.0, scale)

    iterations, kkt = 0, math.inf
    converged = False
    for iterations in range(1, max_iter + 1):
        residual = design @ theta - target
        gradient = design.transpose(-2, -1) @ residual
        at_lo, at_hi = theta <= lower, theta >= upper
        free_mask = ~((at_lo & (gradient > 0)) | (at_hi & (gradient < 0)))
        # KKT: stationarity on free variables, correct multiplier sign on bounds.
        violation = torch.where(free_mask, gradient.abs(), torch.zeros_like(gradient))
        violation = torch.maximum(violation, torch.where(at_lo, (-gradient).clamp(min=0), torch.zeros_like(gradient)))
        violation = torch.maximum(violation, torch.where(at_hi, gradient.clamp(min=0), torch.zeros_like(gradient)))
        kkt = float(violation.max()) if violation.numel() else 0.0
        if kkt <= kkt_tol or not bool(free_mask.any()):
            converged = True
            break

        f0 = 0.5 * float(residual @ residual)
        fixed_mask = ~free_mask
        rhs = target - design[:, fixed_mask] @ theta[fixed_mask]
        newton = theta.clone()
        newton[free_mask] = _lstsq_svd(design[:, free_mask], rhs, rcond)
        theta, improved = _project_line_search(design, target, theta, newton - theta, lower, upper, f0)
        if not improved:
            # Exact-step projected gradient: guarantees descent whenever the KKT
            # test still fails, so the active set cannot stall on a bad guess.
            direction = -gradient
            direction = torch.where(at_lo & (direction < 0), torch.zeros_like(direction), direction)
            direction = torch.where(at_hi & (direction > 0), torch.zeros_like(direction), direction)
            curvature = float((design @ direction) @ (design @ direction))
            if curvature <= 0.0:
                break
            step = -float(gradient @ direction) / curvature
            theta, improved = _project_line_search(
                design, target, theta, step * direction, lower, upper, f0
            )
            if not improved:
                break

    snap = _SNAP_REL * max(1.0, float(theta.abs().max()) if theta.numel() else 1.0)
    theta = torch.where((theta - lower).abs() <= snap, lower, theta)
    theta = torch.where((upper - theta).abs() <= snap, upper, theta)
    return torch.clamp(theta, lower, upper), iterations, kkt, converged


def _project_line_search(
    design: Tensor,
    target: Tensor,
    theta: Tensor,
    direction: Tensor,
    lower: Tensor,
    upper: Tensor,
    f0: float,
    *,
    max_halvings: int = 50,
) -> tuple[Tensor, bool]:
    """Backtrack along the projected arc P(theta + alpha d) until the QP decreases."""
    alpha = 1.0
    for _ in range(max_halvings):
        trial = torch.clamp(theta + alpha * direction, lower, upper)
        residual = design @ trial - target
        if 0.5 * float(residual @ residual) < f0:
            return trial, True
        alpha *= 0.5
    return theta, False


def _diagnostics(singular: Tensor, n_rows: int, n_params: int, weight_sum: float) -> tuple[int, float, float, float]:
    """(rank, condition number, persistent-excitation level, rank tolerance)."""
    s_max = float(singular[0]) if singular.numel() else 0.0
    tol = max(n_rows, n_params) * torch.finfo(singular.dtype).eps * s_max
    rank = int((singular > tol).sum())
    s_min = float(singular[-1]) if singular.numel() else 0.0
    cond = math.inf if s_min <= tol or s_max == 0.0 else s_max / s_min
    # lambda_min of the time-averaged information matrix Phi^T W Phi / sum(w):
    # the standard persistent-excitation level of the regressor.
    excitation = 0.0 if weight_sum <= 0.0 else (s_min ** 2) / weight_sum
    return rank, cond, excitation, tol


def constrained_wls(
    Phi: Tensor,
    u: Tensor,
    lower_bounds: Tensor | Sequence[float] | None = None,
    upper_bounds: Tensor | Sequence[float] | None = None,
    weights: Tensor | Sequence[float] | None = None,
    *,
    cond_max: float = 1.0e8,
    excitation_min: float = 0.0,
    max_iter: int = 64,
) -> WLSResult:
    """Box-constrained weighted least squares for spec 32.4.

    `weights` are inverse-variance row weights (W = diag(weights)), e.g. the
    robust IRLS weights of the joint filter.  `lower_bounds`/`upper_bounds`
    default to the unconstrained box; callers should pass
    `viscous_models.parameter_bounds(...)` to get m > 0, J_e > 0, b >= 0.

    `confident` is False when the regressor is rank deficient, ill-conditioned
    beyond `cond_max`, or under-excited -- spec 32.4 forbids reporting confident
    physical parameters in those cases.  The parameters are still returned (the
    minimum-norm, box-feasible answer) because prediction needs *something*, but
    the flag tells the caller not to trust them.
    """
    if Phi.dim() != 2:
        raise ValueError(f"regressor must be (n_samples, n_params), got {tuple(Phi.shape)}")
    u = torch.as_tensor(u, dtype=Phi.dtype, device=Phi.device).reshape(-1)
    if u.shape[0] != Phi.shape[0]:
        raise ValueError(f"target length {u.shape[0]} does not match regressor rows {Phi.shape[0]}")
    if not bool(torch.isfinite(Phi).all()) or not bool(torch.isfinite(u).all()):
        raise ValueError("non-finite regressor/target: refusing to identify physical parameters")

    out_dtype, device = Phi.dtype, Phi.device
    dtype = _solve_dtype(out_dtype)
    phi64 = Phi.to(dtype)
    u64 = u.to(dtype)
    n_rows, n_params = phi64.shape

    if weights is None:
        w = torch.ones(n_rows, dtype=dtype, device=device)
    else:
        w = torch.as_tensor(weights, dtype=dtype, device=device).reshape(-1)
        if w.shape[0] != n_rows:
            raise ValueError("weights must have one entry per sample")
        if bool((w < 0).any()):
            raise ValueError("weights must be non-negative (they are inverse variances)")
    sqrt_w = torch.sqrt(w)
    design = sqrt_w.unsqueeze(-1) * phi64
    target = sqrt_w * u64

    lower = (
        torch.full((n_params,), -math.inf, dtype=dtype, device=device)
        if lower_bounds is None
        else torch.as_tensor(lower_bounds, dtype=dtype, device=device).reshape(-1)
    )
    upper = (
        torch.full((n_params,), math.inf, dtype=dtype, device=device)
        if upper_bounds is None
        else torch.as_tensor(upper_bounds, dtype=dtype, device=device).reshape(-1)
    )
    if lower.shape[0] != n_params or upper.shape[0] != n_params:
        raise ValueError("bounds must have one entry per parameter")
    if bool((lower > upper).any()):
        raise ValueError("empty box: some lower bound exceeds its upper bound")

    singular = torch.linalg.svdvals(design) if n_rows and n_params else torch.zeros(0, dtype=dtype, device=device)
    rcond = max(n_rows, n_params) * torch.finfo(dtype).eps
    rank, cond, excitation, _ = _diagnostics(singular, n_rows, n_params, float(w.sum()))

    theta, iterations, kkt, converged = _box_qp(
        design, target, lower, upper, rcond=rcond, max_iter=max_iter
    )
    residual = design @ theta - target
    residual_rms = float(torch.sqrt((residual @ residual) / max(n_rows, 1)))

    rank_deficient = rank < n_params
    confident = bool(
        (not rank_deficient)
        and converged
        and cond <= cond_max
        and excitation >= excitation_min
        and bool(torch.isfinite(theta).all())
    )
    return WLSResult(
        params=theta.to(out_dtype),
        confident=confident,
        rank_deficient=rank_deficient,
        rank=rank,
        condition_number=cond,
        excitation=excitation,
        singular_values=singular.to(out_dtype),
        active_lower=(theta <= lower),
        active_upper=(theta >= upper),
        kkt_residual=kkt,
        residual_rms=residual_rms,
        iterations=iterations,
        converged=converged,
    )


class RLSUpdate(NamedTuple):
    """What one recursive update produced (for per-frame diagnostics logging)."""

    params: Tensor
    innovation: float
    gain: Tensor
    rank_deficient: bool
    condition_number: float
    excitation: float
    confident: bool


class RecursiveLeastSquares:
    """Exponentially-weighted RLS with conditioning / excitation reporting.

    Recursion (lambda = forgetting factor, w = per-sample inverse variance):

        K     = P phi / (lambda / w + phi^T P phi)
        theta = theta + K (u - phi^T theta)
        P     = (P - K phi^T P) / lambda

    which is exactly Sherman-Morrison on S <- lambda S + w phi phi^T, so the
    recursive estimate equals the batch WLS solution up to the ridge introduced
    by P0 = initial_variance * I.  The *data* information S and right-hand side
    are accumulated separately from P so that (a) the conditioning report is not
    polluted by the ridge and (b) `constrained_solution()` can reproduce the
    batch box-constrained answer exactly at any time.

    Box constraints are applied as a projection of the reported parameters;
    `constrained_solution()` is the exact constrained optimum of the accumulated
    normal equations and is what a caller should log as the identified physics.
    """

    def __init__(
        self,
        n_params: int,
        *,
        forgetting: float = 1.0,
        initial_variance: float = 1.0e10,
        lower_bounds: Tensor | Sequence[float] | None = None,
        upper_bounds: Tensor | Sequence[float] | None = None,
        param_names: Sequence[str] | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        cond_max: float = 1.0e8,
        excitation_min: float = 0.0,
    ) -> None:
        if n_params <= 0:
            raise ValueError("RLS needs at least one parameter")
        if not (0.0 < float(forgetting) <= 1.0):
            raise ValueError("forgetting factor must lie in (0, 1]")
        if float(initial_variance) <= 0.0:
            raise ValueError("initial_variance must be positive")
        self.n_params = int(n_params)
        self.forgetting = float(forgetting)
        self.initial_variance = float(initial_variance)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.param_names = list(param_names) if param_names is not None else [f"theta_{i}" for i in range(n_params)]
        if len(self.param_names) != self.n_params:
            raise ValueError("param_names length must match n_params")
        self.cond_max = float(cond_max)
        self.excitation_min = float(excitation_min)
        self.lower = (
            torch.full((n_params,), -math.inf, dtype=dtype, device=self.device)
            if lower_bounds is None
            else torch.as_tensor(lower_bounds, dtype=dtype, device=self.device).reshape(-1)
        )
        self.upper = (
            torch.full((n_params,), math.inf, dtype=dtype, device=self.device)
            if upper_bounds is None
            else torch.as_tensor(upper_bounds, dtype=dtype, device=self.device).reshape(-1)
        )
        if self.lower.shape[0] != n_params or self.upper.shape[0] != n_params:
            raise ValueError("bounds must have one entry per parameter")
        self.reset()

    def reset(self, theta0: Tensor | None = None) -> None:
        kw = {"dtype": self.dtype, "device": self.device}
        self._theta = (
            torch.zeros(self.n_params, **kw)
            if theta0 is None
            else torch.as_tensor(theta0, **kw).reshape(-1).clone()
        )
        self._P = self.initial_variance * torch.eye(self.n_params, **kw)
        self._S = torch.zeros(self.n_params, self.n_params, **kw)
        self._rhs = torch.zeros(self.n_params, **kw)
        self._weight_sum = 0.0
        self._n_samples = 0

    # -- accessors -------------------------------------------------------
    @property
    def params(self) -> Tensor:
        """Raw recursive estimate (no projection); see `projected_params`."""
        return self._theta.clone()

    @property
    def projected_params(self) -> Tensor:
        return torch.clamp(self._theta, self.lower, self.upper)

    @property
    def covariance(self) -> Tensor:
        """P = (lambda-weighted information + ridge)^{-1}; scale by sigma^2 for a real covariance."""
        return self._P.clone()

    @property
    def information(self) -> Tensor:
        """Accumulated data information S = sum lambda^{t-i} w_i phi_i phi_i^T (no ridge)."""
        return self._S.clone()

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def effective_samples(self) -> float:
        """Forgetting-discounted sum of weights -- the sample count the estimate really has."""
        return self._weight_sum

    def _spectrum(self) -> Tensor:
        # S is symmetric PSD; eigenvalues are the squared singular values of the
        # weighted design, so the diagnostics match `constrained_wls` exactly.
        evals = torch.linalg.eigvalsh(0.5 * (self._S + self._S.transpose(-2, -1)))
        return torch.sqrt(torch.clamp(evals.flip(0), min=0.0))

    def diagnostics(self) -> tuple[int, float, float]:
        """(rank, condition number, persistent-excitation level) of the regressor."""
        singular = self._spectrum()
        rank, cond, excitation, _ = _diagnostics(singular, max(self._n_samples, 1), self.n_params, self._weight_sum)
        return rank, cond, excitation

    @property
    def rank_deficient(self) -> bool:
        rank, _, _ = self.diagnostics()
        return rank < self.n_params

    @property
    def condition_number(self) -> float:
        return self.diagnostics()[1]

    @property
    def excitation(self) -> float:
        return self.diagnostics()[2]

    @property
    def confident(self) -> bool:
        rank, cond, excitation = self.diagnostics()
        return bool(
            rank == self.n_params
            and cond <= self.cond_max
            and excitation >= self.excitation_min
            and bool(torch.isfinite(self._theta).all())
        )

    # -- recursion -------------------------------------------------------
    def update(self, phi_row: Tensor, u: float | Tensor, weight: float | Tensor = 1.0) -> RLSUpdate:
        """Fold one sample (phi_row, u) with inverse-variance `weight` into the estimate."""
        phi = torch.as_tensor(phi_row, dtype=self.dtype, device=self.device).reshape(-1)
        if phi.shape[0] != self.n_params:
            raise ValueError(f"regressor row must have {self.n_params} entries, got {phi.shape[0]}")
        target = float(u)
        w = float(weight)
        if w < 0.0:
            raise ValueError("weight must be non-negative (it is an inverse variance)")
        if not math.isfinite(target) or not bool(torch.isfinite(phi).all()):
            raise ValueError("non-finite RLS sample rejected")

        lam = self.forgetting
        if w == 0.0:
            # A zero-weight sample carries no information but still ages the filter.
            self._P = self._P / lam
            self._S = lam * self._S
            self._rhs = lam * self._rhs
            self._weight_sum *= lam
            gain = torch.zeros_like(phi)
            innovation = target - float(phi @ self._theta)
        else:
            Pphi = self._P @ phi
            denom = lam / w + float(phi @ Pphi)
            gain = Pphi / denom
            innovation = target - float(phi @ self._theta)
            self._theta = self._theta + gain * innovation
            self._P = (self._P - torch.outer(gain, Pphi)) / lam
            self._P = 0.5 * (self._P + self._P.transpose(-2, -1))  # kill round-off asymmetry
            self._S = lam * self._S + w * torch.outer(phi, phi)
            self._rhs = lam * self._rhs + w * phi * target
            self._weight_sum = lam * self._weight_sum + w
        self._n_samples += 1

        rank, cond, excitation = self.diagnostics()
        return RLSUpdate(
            params=self._theta.clone(),
            innovation=innovation,
            gain=gain,
            rank_deficient=rank < self.n_params,
            condition_number=cond,
            excitation=excitation,
            confident=self.confident,
        )

    def update_batch(self, Phi: Tensor, u: Tensor, weights: Tensor | None = None) -> RLSUpdate:
        """Stream a whole frame window through `update`, returning the last report."""
        Phi = torch.as_tensor(Phi, dtype=self.dtype, device=self.device)
        u = torch.as_tensor(u, dtype=self.dtype, device=self.device).reshape(-1)
        if Phi.dim() != 2 or Phi.shape[0] != u.shape[0]:
            raise ValueError("Phi must be (n_samples, n_params) matching u")
        if weights is None:
            weights = torch.ones_like(u)
        else:
            weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device).reshape(-1)
        report: RLSUpdate | None = None
        for row, target, w in zip(Phi, u, weights):
            report = self.update(row, float(target), float(w))
        if report is None:
            raise ValueError("update_batch requires at least one sample")
        return report

    def constrained_solution(self, *, max_iter: int = 64) -> WLSResult:
        """Exact box-constrained optimum of the accumulated normal equations.

        Factorizing S = V diag(d) V^T gives an equivalent design matrix
        diag(sqrt(d)) V^T with right-hand side diag(1/sqrt(d)) V^T rhs, which has
        the same quadratic as the streamed data, so this reproduces
        `constrained_wls` on the same samples without storing them.
        """
        S = 0.5 * (self._S + self._S.transpose(-2, -1))
        evals, evecs = torch.linalg.eigh(S)
        evals = torch.clamp(evals, min=0.0)
        root = torch.sqrt(evals)
        design = root.unsqueeze(-1) * evecs.transpose(-2, -1)
        safe = torch.where(root > 0, 1.0 / root, torch.zeros_like(root))
        target = safe * (evecs.transpose(-2, -1) @ self._rhs)

        singular = root.flip(0)
        rank, cond, excitation, _ = _diagnostics(
            singular, max(self._n_samples, 1), self.n_params, self._weight_sum
        )
        rcond = max(self._n_samples, self.n_params) * torch.finfo(self.dtype).eps
        theta, iterations, kkt, converged = _box_qp(
            design, target, self.lower, self.upper, rcond=rcond, max_iter=max_iter
        )
        residual = design @ theta - target
        rank_deficient = rank < self.n_params
        return WLSResult(
            params=theta,
            confident=bool(
                (not rank_deficient)
                and converged
                and cond <= self.cond_max
                and excitation >= self.excitation_min
                and bool(torch.isfinite(theta).all())
            ),
            rank_deficient=rank_deficient,
            rank=rank,
            condition_number=cond,
            excitation=excitation,
            singular_values=singular,
            active_lower=(theta <= self.lower),
            active_upper=(theta >= self.upper),
            kkt_residual=kkt,
            residual_rms=float(torch.sqrt((residual @ residual).clamp(min=0.0) / max(self._n_samples, 1))),
            iterations=iterations,
            converged=converged,
        )
