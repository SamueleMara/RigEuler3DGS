"""Observation-supported grouped update for the Phase-1 IEKF (opt-in).

RigEuler3DGS_PHASE1_COHERENCE_CODEX_IMPLEMENTATION_SPEC_2026-11-16.md section I.D.

Legacy grouped solve (`--phase1-group-update legacy`, DEFAULT): 12 DoF per supernode,
dx = G du, every member receives the group's rigid twist whether or not ANY camera
measured it, which is how a hidden Gaussian is dragged by its visible neighbours.

Observable map (`--phase1-group-update observable`). Per Gaussian i, at the CURRENT
linearization (frozen for that iteration, rebuilt for each new accepted iterate):

  P_i^- = [[P_ee, P_ev], [P_ve, P_vv]]        predicted 12x12 block (pose e, velocity v)
  S_i   = chol(P_ee)                           prior scale of the pose tangent
  F~_i  = S_i^T F_i S_i = U diag(lam) U^T      measurement Fisher (6x6 pose block of the
                                               SAME GN operator the solve uses), whitened
  Q~_i  = U diag(q(lam)) U^T                   q = lam/(1+lam)   ("smooth", default)
                                               q = 1[lam > rtol lam_max_i]   ("rank")
  Q_i   = S_i Q~_i S_i^{-1}                    exactly 0 on the nullspace of F_i
  K_i   = P_ve P_ee^{-1}                        conditional-mean velocity of the prior
  B_i   = [I ; K_i] Q_i G_i^pose                (12 x 6C)

The solve is A_B du = b_B with A_B = B^T (Lambda^- + H^T W H + lm I) B, b_B = B^T b,
dx = B du, and the trial re-renders boxplus(x, B du). Measurement-free directions get
exactly no update relative to the prediction x^- (the process prediction is unchanged).

What this map is NOT: q = lam/(1+lam) is the per-Gaussian, per-direction Kalman
variance-reduction factor in the prior-whitened tangent, used as a declared restriction
of the solution space; it is not the unconstrained posterior. Velocity enters only
through the prior's conditional mean K_i (a single RGB-D frame has no direct velocity
Jacobian), so the velocity correction of member i is K_i times its pose correction.
The posterior covariance stays the block approximation (Lambda^- + F_ii)^-1 (which
already leaves a zero-information member at P^-); it is NOT the Joseph covariance of
this restricted update -- tests/test_phase1_observable_group.py measures the gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

POSE = 6
STATE = 12
MAPS = ("smooth", "rank")


@dataclass
class ObservableMap:
    lift: torch.Tensor        # [N,12,6] = [Q_i ; K_i Q_i]
    q: torch.Tensor           # [N,6] the per-direction factors q(lam) (eigen order)
    lam: torch.Tensor         # [N,6] whitened Fisher eigenvalues
    jittered: int             # prior pose blocks that needed a Cholesky jitter
    kind: str

    def stats(self) -> dict:
        qmax = self.q.max(dim=1).values
        return {"group_update_map": self.kind, "q_max_mean": float(qmax.mean()),
                "members_q_max_below_1e-3": int((qmax < 1e-3).sum()),
                "members_zero_information": int((self.lam.max(dim=1).values <= 0).sum()),
                "prior_pose_blocks_jittered": self.jittered}


def _sym(a):
    return 0.5 * (a + a.transpose(-1, -2))


def build_map(P_prior: torch.Tensor, fisher_pose: torch.Tensor, kind: str = "smooth",
              rank_rtol: float = 1e-6) -> ObservableMap:
    """Q_i and K_i from the predicted 12x12 covariance and the 6x6 measurement Fisher."""
    if kind not in MAPS:
        raise ValueError(f"group update map must be one of {MAPS}")
    P = P_prior.double()
    Pee = _sym(P[:, :POSE, :POSE])
    Pve = P[:, POSE:, :POSE]
    n = Pee.shape[0]
    eye = torch.eye(POSE, dtype=P.dtype, device=P.device)
    L, info = torch.linalg.cholesky_ex(Pee)
    bad = info != 0
    jittered = int(bad.sum())
    if jittered:
        scale = Pee.diagonal(dim1=1, dim2=2).abs().mean(dim=1).clamp_min(1e-30)
        fixed = Pee[bad] + (1e-10 * scale[bad])[:, None, None] * eye
        Lb, info_b = torch.linalg.cholesky_ex(fixed)
        if bool((info_b != 0).any()):
            raise ValueError("predicted pose covariance is not positive definite even after jitter")
        L = L.clone()
        L[bad] = Lb
    F = _sym(fisher_pose.double())
    Ft = _sym(L.transpose(1, 2) @ F @ L)
    # 6x6 eigendecompositions on the CPU: the batched CUDA solver (cusolver syevBatched)
    # requests a workspace of ~0.5 MB per matrix (14 GiB for 27k Gaussians -> OOM on the
    # real scenes), while the CPU does 27k 6x6 problems in milliseconds and independently
    # of GPU scheduling.
    lam, U = torch.linalg.eigh(Ft.cpu())
    lam, U = lam.to(Ft.device), U.to(Ft.device)
    lam = lam.clamp_min(0.0)
    if kind == "smooth":
        q = lam / (1.0 + lam)
    else:
        top = lam.max(dim=1, keepdim=True).values
        q = ((lam > rank_rtol * top) & (top > 0)).to(lam.dtype)
    Qt = U @ torch.diag_embed(q) @ U.transpose(1, 2)
    Linv = torch.linalg.solve_triangular(L, eye.expand(n, POSE, POSE), upper=False)
    Q = L @ Qt @ Linv
    # K = P_ve P_ee^-1  ->  K^T = P_ee^-1 P_ev
    K = torch.cholesky_solve(Pve.transpose(1, 2), L).transpose(1, 2)
    lift = torch.cat([Q, K @ Q], dim=1)
    return ObservableMap(lift.to(P_prior.dtype), q.float(), lam.float(), jittered, kind)


def operators(omap: ObservableMap, grouped=None):
    """(B, B^T): du [C,6] (or [N,6] without groups) <-> dx [N,12]."""
    lift = omap.lift
    liftT = lift.transpose(1, 2).contiguous()

    if grouped is None:
        def B(du):
            return torch.einsum("nij,nj->ni", lift, du.to(lift.dtype)).to(du.dtype)

        def BT(y):
            return torch.einsum("nij,nj->ni", liftT, y.to(lift.dtype)).to(y.dtype)
    else:
        def B(du):
            g = grouped.expand(du[:, :POSE].contiguous())
            return torch.einsum("nij,nj->ni", lift, g.to(lift.dtype)).to(du.dtype)

        def BT(y):
            z = torch.einsum("nij,nj->ni", liftT, y.to(lift.dtype))
            return grouped.reduce(z.to(y.dtype).contiguous()).to(y.dtype)
    return B, BT


@torch.no_grad()
def reduced_preconditioner(M_blocks: torch.Tensor, omap: ObservableMap, grouped, B, BT):
    """Block-Jacobi preconditioner of B^T M B with M the per-Gaussian 12x12 normal
    blocks (preconditioner only; the exact operator keeps every coupling)."""
    from .pcg import block_jacobi_preconditioner
    if grouped is None:
        lift = omap.lift.to(M_blocks.dtype)
        blocks = lift.transpose(1, 2) @ M_blocks @ lift
    else:
        c = grouped.num_supernodes
        cols = []
        for k in range(POSE):
            e = torch.zeros((c, POSE), device=M_blocks.device, dtype=M_blocks.dtype)
            e[:, k] = 1.0
            cols.append(BT(torch.einsum("nij,nj->ni", M_blocks, B(e))))
        blocks = torch.stack(cols, dim=2)
    blocks = _sym(blocks)
    return block_jacobi_preconditioner(blocks)


def whitened_information(P_prior: torch.Tensor, fisher_pose: torch.Tensor) -> torch.Tensor:
    """[N,6] eigenvalues of S^T F S (S = chol of the predicted pose covariance); export only."""
    return build_map(P_prior, fisher_pose, "smooth").lam
