"""Observability-aware state update.

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md section 5.

Section 1.2 measured that 44% of pose blocks are weakly observable at full
scene scale. Section 5's instruction about those directions is blunt: "Do not
pretend they are measured."

Concretely, for each block (preferably a supernode -- section 5.2 is explicit
that per-Gaussian freezing is the wrong granularity when the supernode is
jointly observable):

    F_c = sum_p H_{p,c}^T R_p^-1 H_{p,c} = U_c diag(lambda_{c,k}) U_c^T

Eigenvalues of a raw F mix units: a translation direction is measured in
pixels^2 per metre^2, a rotation direction in pixels^2 per radian^2. Comparing
them against one threshold is meaningless, so section 5.1 requires whitening
first. We use the same scaling metric as the trust region, which makes an
eigenvalue of 1 mean "one prior standard deviation of motion in this direction
produces one unit of weighted image residual" in every direction alike.

The weak directions are then handled by section 5.2's policy: the measurement
correction may be projected out of them, the update stays prior-dominated, and
the posterior uncertainty is INFLATED rather than motion being invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

EPS = 1e-12


@dataclass
class ObservabilityReport:
    """Section 5.3's required persistence, per block and aggregated."""

    eigenvalues: torch.Tensor          # [B,6] ascending, in the scaled metric
    eigenvectors: torch.Tensor         # [B,6,6] columns are directions
    rank: torch.Tensor                 # [B] count of lambda >= tau
    weak_mask: torch.Tensor            # [B,6] bool, True = weak
    tau: float
    scaled: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def num_blocks(self) -> int:
        return int(self.eigenvalues.shape[0])

    @property
    def fraction_weak(self) -> float:
        return float(self.weak_mask.float().mean())

    @property
    def active_dimension(self) -> int:
        """Section 5.3's 'global/active state dimension': the number of state
        directions the measurement actually constrains."""
        return int(self.rank.sum())

    @property
    def global_dimension(self) -> int:
        return int(self.eigenvalues.numel())

    def to_dict(self) -> dict:
        ev = self.eigenvalues
        strong = ev[~self.weak_mask]
        d = {
            "num_blocks": self.num_blocks,
            "tau_obs": self.tau,
            "scaled": self.scaled,
            "min_eigenvalue": float(ev.min()),
            "median_eigenvalue": float(ev.median()),
            "max_eigenvalue": float(ev.max()),
            "mean_rank": float(self.rank.float().mean()),
            "min_rank": int(self.rank.min()),
            "max_rank": int(self.rank.max()),
            "fraction_weak": self.fraction_weak,
            "global_dimension": self.global_dimension,
            "active_dimension": self.active_dimension,
            "blocks_fully_observable": int((self.rank == ev.shape[1]).sum()),
            "blocks_fully_unobservable": int((self.rank == 0).sum()),
            "min_strong_eigenvalue": float(strong.min()) if strong.numel() else None,
        }
        d.update(self.extra)
        return d


def _chunked_eigh(A: torch.Tensor, chunk: int = 1024
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched symmetric eigendecomposition with a bounded workspace."""
    if A.shape[0] <= chunk:
        return torch.linalg.eigh(A)
    lam, U = [], []
    for s in range(0, A.shape[0], chunk):
        l_, u_ = torch.linalg.eigh(A[s:s + chunk])
        lam.append(l_)
        U.append(u_)
    return torch.cat(lam), torch.cat(U)


@torch.no_grad()
def analyze_blocks(fisher: torch.Tensor, *, tau: float = 1e-2,
                   scale: torch.Tensor | None = None,
                   metric=None) -> ObservabilityReport:
    """Section 5.1: scaled eigendecomposition of the per-block Fisher.

    ``fisher`` is [B,6,6]. ``scale`` is a [6] or [B,6] vector of physical units
    (the prior sigma in each direction); the whitened block is

        F~ = S F S,    S = diag(scale)

    so lambda(F~) is dimensionless and comparable across translation and
    rotation. Passing a :class:`~estimator.trust_region.ScalingMetric` as
    ``metric`` uses exactly the metric the trust region steps in, which is what
    keeps "weak" meaning the same thing in both places.
    """
    F = fisher.double()
    B, n, _ = F.shape
    if metric is not None and scale is None:
        w = getattr(metric, "weight", None)
        if w is not None:
            w = torch.as_tensor(w).double().reshape(-1, n)
            # ScalingMetric stores 1/sigma^2 weights; the whitening scale is sigma.
            scale = (1.0 / w.clamp_min(EPS)).sqrt()
            if scale.shape[0] == 1:
                scale = scale.expand(B, n)
    if scale is None:
        S = torch.ones((B, n), device=F.device, dtype=F.dtype)
        scaled = False
    else:
        S = torch.as_tensor(scale).double().to(F.device).reshape(-1, n)
        if S.shape[0] == 1:
            S = S.expand(B, n)
        scaled = True
    Fw = S.unsqueeze(2) * F * S.unsqueeze(1)
    # Symmetrize: atomicAdd accumulation leaves the two triangles differing at
    # round-off, and eigvalsh would silently read only one of them.
    Fw = 0.5 * (Fw + Fw.transpose(1, 2))
    # Chunked: cuSOLVER's batched syevj asks for a workspace proportional to
    # the batch size, which at full scene scale (11320 blocks) is several GiB
    # for 3 MB of actual data. The decomposition is per-block independent, so
    # chunking changes nothing but the peak allocation.
    lam, U = _chunked_eigh(Fw)
    lam = lam.clamp_min(0.0)                     # a Fisher block is PSD
    weak = lam < tau
    rank = (~weak).sum(dim=1)
    return ObservabilityReport(
        eigenvalues=lam, eigenvectors=U, rank=rank, weak_mask=weak,
        tau=float(tau), scaled=scaled,
        extra={"whitening": "prior_sigma" if scaled else "none"})


@torch.no_grad()
def observable_projector(report: ObservabilityReport) -> torch.Tensor:
    """P_c = U_s U_s^T over the STRONG eigenvectors only. [B,6,6]."""
    U = report.eigenvectors
    keep = (~report.weak_mask).to(U.dtype).unsqueeze(1)    # [B,1,6]
    Us = U * keep
    return Us @ Us.transpose(1, 2)


@torch.no_grad()
def project_to_observable(delta: torch.Tensor,
                          report: ObservabilityReport) -> torch.Tensor:
    """Section 5.2's optional projection: drop the measurement correction's
    component along unobservable directions instead of letting the solver put
    arbitrary motion there."""
    P = observable_projector(report).to(delta.dtype)
    return torch.einsum("bij,bj->bi", P, delta)


@torch.no_grad()
def split_observable_nullspace(delta: torch.Tensor,
                               report: ObservabilityReport
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Section 5.3: a state error must be reported in the two subspaces
    separately, because a large error in an unmeasured direction is not an
    estimator failure -- claiming it was measured would be."""
    obs = project_to_observable(delta, report)
    return obs, delta - obs


@torch.no_grad()
def observable_error_metrics(error: torch.Tensor, report: ObservabilityReport,
                             prefix: str = "") -> dict:
    """Section 5.3's 'observable-subspace state error'."""
    obs, null = split_observable_nullspace(error, report)
    total = float(error.norm())
    return {
        f"{prefix}error_total": total,
        f"{prefix}error_observable": float(obs.norm()),
        f"{prefix}error_nullspace": float(null.norm()),
        f"{prefix}error_observable_rms": float(obs.norm() / max(obs.shape[0], 1) ** 0.5),
        f"{prefix}error_nullspace_fraction": float(null.norm() / max(total, EPS)),
        f"{prefix}active_dimension": report.active_dimension,
        f"{prefix}global_dimension": report.global_dimension,
    }


@torch.no_grad()
def inflate_weak_covariance(posterior: torch.Tensor, report: ObservabilityReport,
                            prior: torch.Tensor | None = None,
                            floor_scale: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Section 5.2: 'inflate posterior uncertainty rather than inventing motion'.

    Along a weak eigenvector the measurement said nothing, so the posterior
    variance in that direction must not be smaller than the prior's. Rebuilding
    the block as U diag(sigma^2) U^T with the weak entries raised to the prior
    floor does exactly that, and leaves the strong directions untouched.
    """
    U = report.eigenvectors
    P = posterior.double()
    B, n, _ = P.shape
    # variance along each eigendirection
    var = torch.einsum("bki,bkl,bli->bi", U, P, U)
    if prior is not None:
        floor = torch.einsum("bki,bkl,bli->bi", U, prior.double(), U)
    else:
        floor = var.max(dim=1, keepdim=True).values.expand_as(var)
    floor = floor * float(floor_scale)
    weak = report.weak_mask
    raised = torch.where(weak & (var < floor), floor, var)
    n_raised = int((weak & (var < floor)).sum())
    out = torch.einsum("bik,bk,bjk->bij", U, raised, U)
    out = 0.5 * (out + out.transpose(1, 2))
    return out.to(posterior.dtype), {
        "weak_directions_inflated": n_raised,
        "weak_directions_total": int(weak.sum()),
        "mean_inflation_factor": float((raised / var.clamp_min(EPS))[weak].mean())
        if int(weak.sum()) else 1.0,
    }


@torch.no_grad()
def weak_direction_policy(step: torch.Tensor, report: ObservabilityReport,
                          *, project: bool = True) -> tuple[torch.Tensor, dict]:
    """Apply section 5.2's policy to a computed measurement correction.

    Returns the (possibly projected) step and the diagnostics that let a reader
    see how much of the proposed motion lived in directions nothing measured.
    """
    obs, null = split_observable_nullspace(step, report)
    info = {
        "step_norm": float(step.norm()),
        "step_observable_norm": float(obs.norm()),
        "step_nullspace_norm": float(null.norm()),
        "step_nullspace_fraction": float(null.norm() / max(float(step.norm()), EPS)),
        "fraction_weak_directions": report.fraction_weak,
        "projected": bool(project),
    }
    return (obs if project else step), info
