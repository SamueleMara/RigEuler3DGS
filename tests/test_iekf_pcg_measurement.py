"""Tests for CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF sections 6, 13.1, 15, 16.

Covers the RGB-D observation model (section 6: physical channel precision,
range-dependent sigma_D, exactly-zero precision on depth holes), the robust
Student-t IRLS reweighting (section 13.1), the matrix-free PCG of section 15
(including mandatory test T11, PCG versus dense Cholesky) and the block-Jacobi
/ block-Fisher preconditioner of section 16.

Dense matrices appear ONLY inside this file, as oracles.  The solver itself is
handed nothing but a closure ``v -> A v`` (spec 0.6).  No autograd anywhere:
spec 36 allows it as an oracle only, and none is needed here.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch

from estimator import measurement as meas
from estimator import pcg as pcg_mod
from estimator.measurement import (
    DepthSigmaModel,
    build_precision,
    channel_sigma,
    depth_validity,
    make_depth_sigma_model,
    measurement_energy,
    normalized_innovation_stats,
    rgbd_residual,
    student_t_weight,
)
from estimator.pcg import (
    REASON_CONVERGED,
    REASON_MAX_ITERS,
    REASON_NEGATIVE_CURVATURE,
    REASON_NONFINITE_RHS,
    REASON_ZERO_RHS,
    assemble_preconditioner_blocks,
    block_jacobi_preconditioner,
    pack_symmetric,
    pcg,
    symmetric_packed_length,
    unpack_symmetric,
)

torch.manual_seed(20261004)
DT = torch.float64

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}" + (f"   [{detail}]" if detail else ""))


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def spd(m: int, seed: int, ridge: float = 0.5) -> torch.Tensor:
    """Dense SPD oracle matrix (test-only; the solver never sees it)."""
    b = torch.randn((m, m), generator=gen(seed), dtype=DT)
    return b @ b.T / m + ridge * torch.eye(m, dtype=DT)


class CountingOperator:
    """Wraps a dense matrix as a matrix-free callable and counts invocations."""

    def __init__(self, matrix: torch.Tensor, shape: tuple[int, ...]) -> None:
        self.matrix = matrix
        self.shape = shape
        self.calls = 0

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return (self.matrix @ v.reshape(-1)).reshape(self.shape)


# ---------------------------------------------------------------------------
# Section 15: PCG
# ---------------------------------------------------------------------------
def test_T11_pcg_versus_dense_cholesky():
    """T11: same SPD system solved by dense Cholesky and by matrix-free PCG."""
    n, d = 9, 12
    m = n * d
    a = spd(m, seed=11)
    b = torch.randn((n, d), generator=gen(12), dtype=DT)
    op = CountingOperator(a, (n, d))

    x, info = pcg(op, b, None, max_iters=4 * m, tol=1.0e-12)
    reference = torch.linalg.solve(a, b.reshape(-1)).reshape(n, d)

    err = float((x - reference).abs().max())
    check("T11 pcg solution matches dense solve", err < 1.0e-9, f"max|dx|={err:.3e}")
    check("T11 cholesky oracle agrees too",
          float((torch.cholesky_solve(b.reshape(-1, 1), torch.linalg.cholesky(a)).reshape(n, d)
                 - x).abs().max()) < 1.0e-9)
    check("T11 reports convergence", info["converged"] and info["reason"] == REASON_CONVERGED,
          info["reason"])
    check("T11 verified relative residual below tol",
          info["relative_residual"] < 1.0e-12, f"rel={info['relative_residual']:.3e}")
    check("T11 history has one entry per iteration plus the initial one",
          len(info["residual_history"]) == info["iterations"] + 1,
          f"{len(info['residual_history'])} vs {info['iterations']}")
    check("T11 CG terminates in at most m iterations", info["iterations"] <= m,
          f"iters={info['iterations']}")
    check("T11 A was only ever touched through the callable",
          op.calls == info["matvecs"], f"{op.calls} vs {info['matvecs']}")


def test_pcg_exact_inverse_preconditioner_converges_in_one_iteration():
    n, d = 5, 6
    m = n * d
    a = spd(m, seed=21)
    a_inv = torch.linalg.inv(a)
    b = torch.randn((n, d), generator=gen(22), dtype=DT)
    op = CountingOperator(a, (n, d))

    x, info = pcg(op, b, lambda v: (a_inv @ v.reshape(-1)).reshape(n, d),
                  max_iters=50, tol=1.0e-10)
    reference = torch.linalg.solve(a, b.reshape(-1)).reshape(n, d)
    check("M^-1 = A^-1 converges in exactly one iteration", info["iterations"] == 1,
          f"iters={info['iterations']}")
    check("M^-1 = A^-1 lands on the exact solution",
          float((x - reference).abs().max()) < 1.0e-12)
    check("M^-1 = A^-1 is reported as converged", info["converged"])


def test_pcg_zero_rhs_returns_exact_zeros_without_iterating():
    n, d = 4, 12
    a = spd(n * d, seed=31)
    op = CountingOperator(a, (n, d))
    b = torch.zeros((n, d), dtype=DT)

    x, info = pcg(op, b, None, max_iters=25, tol=1.0e-8)
    check("b = 0 gives bitwise zeros", bool((x == 0).all()) and x.shape == b.shape)
    check("b = 0 performs no matrix-vector product", op.calls == 0, f"calls={op.calls}")
    check("b = 0 reports zero iterations", info["iterations"] == 0)
    check("b = 0 says so explicitly", info["reason"] == REASON_ZERO_RHS, info["reason"])
    check("b = 0 is trivially converged", info["converged"])
    check("b = 0 reports a zero relative residual", info["relative_residual"] == 0.0)


def test_pcg_reports_non_convergence_honestly():
    n, d = 8, 12
    m = n * d
    # Deliberately ill conditioned: a tiny ridge leaves a wide spectrum, so two
    # iterations cannot possibly reach 1e-14.
    a = spd(m, seed=41, ridge=1.0e-4)
    b = torch.randn((n, d), generator=gen(42), dtype=DT)
    op = CountingOperator(a, (n, d))

    x, info = pcg(op, b, None, max_iters=2, tol=1.0e-14)
    check("too few iterations -> converged is False", not info["converged"])
    check("too few iterations -> reason is max_iterations",
          info["reason"] == REASON_MAX_ITERS, info["reason"])
    check("too few iterations -> residual is above tol",
          info["relative_residual"] > 1.0e-14, f"rel={info['relative_residual']:.3e}")
    check("non-convergent iterate is still finite", bool(torch.isfinite(x).all()))
    check("exactly max_iters iterations were run", info["iterations"] == 2)
    check("the reported residual is the verified one, not the recursive one",
          abs(info["relative_residual"]
              - float((b - (a @ x.reshape(-1)).reshape(n, d)).norm() / b.norm())) < 1.0e-12)


def test_pcg_aborts_on_nan_and_inf():
    n, d = 3, 4
    a = spd(n * d, seed=51)
    poisoned = a.clone()
    poisoned[2, 5] = float("nan")
    b = torch.randn((n, d), generator=gen(52), dtype=DT)

    x, info = pcg(CountingOperator(poisoned, (n, d)), b, None, max_iters=10, tol=1.0e-10)
    check("NaN operator aborts", not info["converged"])
    check("NaN operator gives a reason", "nonfinite" in info["reason"], info["reason"])
    check("NaN operator never returns NaN", bool(torch.isfinite(x).all()))

    inf_matrix = a.clone()
    inf_matrix[0, 0] = float("inf")
    x_inf, info_inf = pcg(CountingOperator(inf_matrix, (n, d)), b, None, max_iters=10, tol=1.0e-10)
    check("Inf operator aborts", not info_inf["converged"])
    check("Inf operator never returns non-finite values", bool(torch.isfinite(x_inf).all()))

    bad_rhs = b.clone()
    bad_rhs[1, 1] = float("nan")
    x_b, info_b = pcg(CountingOperator(a, (n, d)), bad_rhs, None, max_iters=10, tol=1.0e-10)
    check("NaN right-hand side is rejected before iterating",
          info_b["reason"] == REASON_NONFINITE_RHS and info_b["iterations"] == 0, info_b["reason"])
    check("NaN right-hand side returns finite zeros",
          bool(torch.isfinite(x_b).all()) and bool((x_b == 0).all()))

    # check_finite=False must still catch it through the scalar guards.
    x_s, info_s = pcg(CountingOperator(poisoned, (n, d)), b, None, max_iters=10, tol=1.0e-10,
                      check_finite=False)
    check("scalar guards catch NaN even without vector scans",
          not info_s["converged"] and bool(torch.isfinite(x_s).all()), info_s["reason"])


def test_pcg_detects_negative_curvature():
    n, d = 4, 3
    diag = torch.ones(n * d, dtype=DT)
    diag[-1] = -2.0  # one indefinite direction
    a = torch.diag(diag)
    b = torch.zeros((n, d), dtype=DT)
    b[-1, -1] = 1.0  # points straight down the negative-curvature direction

    x, info = pcg(CountingOperator(a, (n, d)), b, None, max_iters=10, tol=1.0e-12)
    check("negative curvature is detected", info["reason"] == REASON_NEGATIVE_CURVATURE,
          info["reason"])
    check("negative curvature sets the flag", info["negative_curvature"])
    check("negative curvature does not claim convergence", not info["converged"])
    check("negative curvature stops before corrupting the iterate",
          bool((x == 0).all()) and info["iterations"] == 0)


def test_pcg_warm_start_and_shape_handling():
    n, d = 6, 12
    m = n * d
    a = spd(m, seed=61)
    b = torch.randn((n, d), generator=gen(62), dtype=DT)
    reference = torch.linalg.solve(a, b.reshape(-1)).reshape(n, d)
    op = CountingOperator(a, (n, d))

    x_warm, info_warm = pcg(op, b, None, max_iters=4 * m, tol=1.0e-12,
                            x0=reference + 1.0e-6 * torch.randn((n, d), generator=gen(63), dtype=DT))
    check("warm start still converges", info_warm["converged"], info_warm["reason"])
    check("warm start reaches the same solution",
          float((x_warm - reference).abs().max()) < 1.0e-9)
    check("warm start needs fewer iterations than a cold start",
          info_warm["iterations"] < pcg(op, b, None, max_iters=4 * m, tol=1.0e-12)[1]["iterations"],
          f"warm={info_warm['iterations']}")

    x0_exact = reference.clone()
    _, info_exact = pcg(op, b, None, max_iters=10, tol=1.0e-10, x0=x0_exact)
    check("an exact warm start converges with zero iterations",
          info_exact["iterations"] == 0 and info_exact["converged"])

    # float32 on the same system: the solver must preserve dtype and device.
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    x32, info32 = pcg(CountingOperator(a32, (n, d)), b32, None, max_iters=4 * m, tol=1.0e-5)
    check("float32 is preserved", x32.dtype == torch.float32 and x32.device == b32.device)
    check("float32 still converges", info32["converged"], info32["reason"])
    check("float32 solution is close to the float64 one",
          float((x32.double() - reference).abs().max()) < 1.0e-3,
          f"max|dx|={float((x32.double() - reference).abs().max()):.3e}")


# ---------------------------------------------------------------------------
# Section 16: block-Jacobi / block-Fisher preconditioner
# ---------------------------------------------------------------------------
def test_block_jacobi_reproduces_blockwise_inverse():
    n, d = 7, 12
    factors = torch.randn((n, d, d), generator=gen(71), dtype=DT)
    blocks = factors @ factors.transpose(-1, -2) + 0.75 * torch.eye(d, dtype=DT)

    pre = block_jacobi_preconditioner(blocks)
    check("a healthy PD input needs no repair at all",
          pre.num_repaired == 0 and pre.num_jittered == 0, str(pre.report()))

    v = torch.randn((n, d), generator=gen(72), dtype=DT)
    blockwise = torch.linalg.solve(blocks, v.unsqueeze(-1)).squeeze(-1)
    check("M^-1 v matches the per-block solve",
          float((pre(v) - blockwise).abs().max()) < 1.0e-10)

    dense = torch.block_diag(*[blocks[i] for i in range(n)])
    dense_ref = torch.linalg.solve(dense, v.reshape(-1)).reshape(n, d)
    check("M^-1 v matches the dense block-diagonal solve",
          float((pre(v) - dense_ref).abs().max()) < 1.0e-10)

    check("inverse_blocks equals the explicit block inverse",
          float((pre.inverse_blocks() - torch.linalg.inv(blocks)).abs().max()) < 1.0e-10)
    check("log_det equals sum_i log det M_i",
          abs(float(pre.log_det()) - float(torch.linalg.slogdet(blocks)[1].sum())) < 1.0e-9)
    check("the preconditioner is symmetric: <u, M^-1 v> == <M^-1 u, v>",
          abs(float((v * pre(blockwise)).sum()) - float((pre(v) * blockwise).sum())) < 1.0e-10)


def test_block_jacobi_repairs_and_reports_non_pd_blocks():
    n, d = 6, 6
    factors = torch.randn((n, d, d), generator=gen(81), dtype=DT)
    blocks = factors @ factors.transpose(-1, -2) + 0.5 * torch.eye(d, dtype=DT)
    blocks[1] = blocks[1] - (torch.linalg.eigvalsh(blocks[1])[0] + 1.0e-3) * torch.eye(d, dtype=DT)
    blocks[3] = torch.zeros((d, d), dtype=DT)  # fully occluded Gaussian: no information at all

    pre = block_jacobi_preconditioner(blocks)
    report = pre.report()
    check("non-PD blocks are repaired and counted", pre.num_repaired >= 2, str(report))
    check("the indefinite block is reported", bool(pre.repaired_mask[1]))
    check("the zero block falls back to the scene-typical identity",
          bool(pre.identity_fallback_mask[3]))
    check("healthy blocks are untouched",
          not bool(pre.repaired_mask[[0, 2, 4, 5]].any()))
    check("the report carries the jitter counters",
          {"num_jittered", "num_eigenvalue_floored", "num_identity_fallback", "num_repaired"}
          <= set(report))

    v = torch.randn((n, d), generator=gen(82), dtype=DT)
    out = pre(v)
    check("the repaired preconditioner stays finite", bool(torch.isfinite(out).all()))
    check("the repaired preconditioner does not explode on the null block",
          float(out[3].abs().max()) < 1.0e3, f"max={float(out[3].abs().max()):.3e}")
    check("untouched blocks keep their exact inverse",
          float((out[0] - torch.linalg.solve(blocks[0], v[0])).abs().max()) < 1.0e-10)

    asym = blocks.clone()
    asym[0, 0, 1] += 3.0  # atomic tile accumulation never produces an exactly symmetric block
    pre_sym = block_jacobi_preconditioner(asym)
    expect = torch.linalg.solve(0.5 * (asym[0] + asym[0].T), v[0])
    check("blocks are symmetrized before factorization",
          float((pre_sym(v)[0] - expect).abs().max()) < 1.0e-10)


def test_packed_symmetric_storage_and_block_assembly():
    """Section 16: 21 values per symmetric 6x6 and M_i = Lambda^- + F + lambda D."""
    check("a symmetric 6x6 needs exactly 21 values", symmetric_packed_length(6) == 21)
    check("a symmetric 12x12 needs exactly 78 values", symmetric_packed_length(12) == 78)

    n, f, d = 5, 6, 12
    raw = torch.randn((n, f, f), generator=gen(101), dtype=DT)
    fisher = raw @ raw.transpose(-1, -2)  # PSD, as H^T R^-1 H always is
    packed = pack_symmetric(fisher)
    check("packing yields 21 entries per block", packed.shape == (n, 21))
    check("unpack(pack(F)) == F", float((unpack_symmetric(packed) - fisher).abs().max()) < 1.0e-12)
    check("packing order is row-major lower triangle",
          float(packed[0, 0] - fisher[0, 0, 0]) == 0.0
          and float(packed[0, 1] - fisher[0, 1, 0]) == 0.0
          and float(packed[0, 2] - fisher[0, 1, 1]) == 0.0)
    check("the dimension is inferred from the packed length",
          unpack_symmetric(packed).shape == (n, f, f))
    check("a bad packed length is rejected",
          _raises(lambda: unpack_symmetric(torch.zeros((n, 20), dtype=DT))))

    prior_raw = torch.randn((n, d, d), generator=gen(102), dtype=DT)
    prior = prior_raw @ prior_raw.transpose(-1, -2) + torch.eye(d, dtype=DT)
    lam = 0.25
    m = assemble_preconditioner_blocks(packed, prior_blocks=prior, lam=lam, damping="identity")
    expected = prior.clone()
    expected[:, :f, :f] += fisher
    expected = expected + lam * torch.eye(d, dtype=DT)
    check("M = Lambda^- + embed(F) + lambda I",
          float((m - expected).abs().max()) < 1.0e-12)
    check("the pose-only Fisher does not touch the velocity block",
          float((m[:, f:, f:] - (prior[:, f:, f:] + lam * torch.eye(f, dtype=DT))).abs().max())
          < 1.0e-12)
    check("the prior's pose-velocity coupling survives embedding",
          float((m[:, :f, f:] - prior[:, :f, f:]).abs().max()) < 1.0e-12)
    check("the assembled blocks are exactly symmetric",
          float((m - m.transpose(-1, -2)).abs().max()) == 0.0)

    marq = assemble_preconditioner_blocks(fisher, prior_blocks=prior, lam=lam, damping="marquardt")
    base = prior.clone()
    base[:, :f, :f] += fisher
    check("marquardt damping scales by the diagonal of Lambda^- + F",
          float((marq - (base + lam * torch.diag_embed(
              torch.diagonal(base, dim1=-2, dim2=-1)))).abs().max()) < 1.0e-12)
    check("a dense Fisher block is accepted as well as a packed one",
          float((assemble_preconditioner_blocks(fisher, prior_blocks=prior, lam=lam) - m).abs().max())
          < 1.0e-12)
    check("a Fisher block that does not fit is rejected",
          _raises(lambda: assemble_preconditioner_blocks(fisher, prior_blocks=prior,
                                                         fisher_offset=d)))
    check("assembling nothing at all is an error",
          _raises(lambda: assemble_preconditioner_blocks(None)))

    pre = block_jacobi_preconditioner(m)
    v = torch.randn((n, d), generator=gen(103), dtype=DT)
    check("the assembled blocks invert exactly through the preconditioner",
          float((pre(v) - torch.linalg.solve(m, v.unsqueeze(-1)).squeeze(-1)).abs().max()) < 1.0e-10)
    check("a Fisher-plus-prior block never needs repair", pre.num_repaired == 0)

    # A Gaussian seen by no pixel contributes F_i = 0; the prior alone must keep
    # its block positive definite (section 27, low alpha / full occlusion).
    occluded = packed.clone()
    occluded[2] = 0.0
    m_occ = assemble_preconditioner_blocks(occluded, prior_blocks=prior, lam=lam)
    check("an unobserved Gaussian keeps the prior block",
          float((m_occ[2] - (prior[2] + lam * torch.eye(d, dtype=DT))).abs().max()) < 1.0e-12)
    check("an unobserved Gaussian is still invertible",
          block_jacobi_preconditioner(m_occ).num_repaired == 0)


def test_pcg_with_block_jacobi_beats_unpreconditioned():
    """End-to-end section 15 + 16: block-diagonal-dominant normal operator."""
    n, d = 10, 12
    m = n * d
    scales = torch.logspace(-2, 2, n, dtype=DT)  # wildly different per-Gaussian curvature
    factors = torch.randn((n, d, d), generator=gen(91), dtype=DT)
    blocks = (factors @ factors.transpose(-1, -2) + 2.0 * torch.eye(d, dtype=DT)) \
        * scales.view(-1, 1, 1)
    dense = torch.block_diag(*[blocks[i] for i in range(n)])
    coupling = torch.randn((m, m), generator=gen(92), dtype=DT) * 1.0e-3
    a = dense + 0.5 * (coupling + coupling.T)
    a = a + (abs(float(torch.linalg.eigvalsh(a)[0])) + 1.0e-3) * torch.eye(m, dtype=DT)

    b = torch.randn((n, d), generator=gen(93), dtype=DT)
    reference = torch.linalg.solve(a, b.reshape(-1)).reshape(n, d)
    pre = block_jacobi_preconditioner(blocks)

    _, plain = pcg(CountingOperator(a, (n, d)), b, None, max_iters=4 * m, tol=1.0e-10)
    x_pre, info_pre = pcg(CountingOperator(a, (n, d)), b, pre, max_iters=4 * m, tol=1.0e-10)

    check("preconditioned PCG converges", info_pre["converged"], info_pre["reason"])
    check("preconditioned PCG hits the same solution",
          float((x_pre - reference).abs().max()) < 1.0e-8,
          f"max|dx|={float((x_pre - reference).abs().max()):.3e}")
    check("the block-Jacobi preconditioner cuts the iteration count",
          info_pre["iterations"] < plain["iterations"],
          f"{info_pre['iterations']} vs {plain['iterations']}")
    check("the residual history is monotone enough to be usable",
          info_pre["residual_history"][-1] < info_pre["residual_history"][0])


# ---------------------------------------------------------------------------
# Section 6: RGB-D observation model
# ---------------------------------------------------------------------------
def make_rgbd(h: int = 5, w: int = 7, seed: int = 101):
    y_hat = torch.rand((4, h, w), generator=gen(seed), dtype=DT)
    y_gt = y_hat + 0.01 * torch.randn((4, h, w), generator=gen(seed + 1), dtype=DT)
    y_hat[3] = 1.0 + 3.0 * torch.rand((h, w), generator=gen(seed + 2), dtype=DT)
    y_gt[3] = y_hat[3] + 0.02
    return y_gt, y_hat


def test_rgbd_residual_and_channel_precision():
    y_gt, y_hat = make_rgbd()
    r = rgbd_residual(y_gt, y_hat)
    check("residual is y_gt - h(x_hat)", torch.equal(r, y_gt - y_hat))
    check("residual keeps shape/dtype/device",
          r.shape == y_gt.shape and r.dtype == DT and r.device == y_gt.device)

    prec = build_precision(r, None, sigma_rgb=(0.02, 0.04, 0.08),
                           depth_sigma_model=DepthSigmaModel("constant", a=0.05))
    check("precision is shaped like the residual", prec.shape == r.shape)
    check("R channel precision is 1/sigma_R^2",
          abs(float(prec[0].mean()) - 1.0 / 0.02 ** 2) < 1.0e-9)
    check("G channel precision is 1/sigma_G^2",
          abs(float(prec[1].mean()) - 1.0 / 0.04 ** 2) < 1.0e-9)
    check("B channel precision is 1/sigma_B^2",
          abs(float(prec[2].mean()) - 1.0 / 0.08 ** 2) < 1.0e-9)
    check("D channel precision is 1/sigma_D^2",
          abs(float(prec[3].mean()) - 1.0 / 0.05 ** 2) < 1.0e-9)
    check("channel precisions genuinely differ (physical, not a shared weight)",
          len({round(float(prec[c].mean()), 6) for c in range(4)}) == 4)

    energy = measurement_energy(r, prec)
    manual = 0.5 * float((prec * r.square()).sum())
    check("energy is 0.5 * sum(precision * residual^2)",
          abs(float(energy) - manual) < 1.0e-9, f"{float(energy):.6f}")

    rgb_only = build_precision(r, None, sigma_rgb=0.02, use_depth=False)
    check("--iekf_use_depth off zeroes the depth channel exactly",
          bool((rgb_only[3] == 0).all()) and bool((rgb_only[:3] > 0).all()))
    depth_only = build_precision(r, None, sigma_rgb=0.02, use_rgb=False)
    check("--iekf_use_rgb off zeroes the colour channels exactly",
          bool((depth_only[:3] == 0).all()) and bool((depth_only[3] > 0).all()))

    flat = r.permute(1, 2, 0).contiguous()  # [H, W, 4] contributor-style layout
    prec_flat = build_precision(flat, None, sigma_rgb=(0.02, 0.04, 0.08),
                                depth_sigma_model=DepthSigmaModel("constant", a=0.05),
                                channel_dim=-1)
    check("channel_dim=-1 gives the same precision field",
          float((prec_flat.permute(2, 0, 1) - prec).abs().max()) < 1.0e-12)


def test_invalid_depth_pixels_get_exactly_zero_precision():
    y_gt, y_hat = make_rgbd(seed=201)
    # Realistic hole encodings: 0.0 (no return) and NaN (rejected return).
    y_gt[3, 0, 0] = 0.0
    y_gt[3, 1, 2] = float("nan")
    valid_depth = depth_validity(y_gt[3])
    check("zero-depth pixel is flagged invalid", not bool(valid_depth[0, 0]))
    check("NaN-depth pixel is flagged invalid", not bool(valid_depth[1, 2]))
    check("every other depth pixel stays valid", int((~valid_depth).sum()) == 2)

    mask = torch.ones_like(y_gt, dtype=torch.bool)
    mask[3] = valid_depth
    r = rgbd_residual(y_gt, y_hat, valid_mask=mask)
    check("a NaN hole never reaches the residual", bool(torch.isfinite(r).all()))

    prec = build_precision(r, mask, sigma_rgb=0.02,
                           depth_sigma_model=DepthSigmaModel("constant", a=0.05))
    check("invalid depth precision is bitwise zero",
          float(prec[3, 0, 0]) == 0.0 and float(prec[3, 1, 2]) == 0.0)
    check("no fake depth value was substituted",
          bool((prec[3][~valid_depth] == 0).all()) and bool((prec[3][valid_depth] > 0).all()))
    check("colour channels at a depth hole are untouched",
          bool((prec[:3, 0, 0] > 0).all()))

    # Energy contribution of the holes is exactly zero: perturb them arbitrarily.
    energy = measurement_energy(r, prec)
    r_perturbed = r.clone()
    r_perturbed[3, 0, 0] = 1.0e6
    r_perturbed[3, 1, 2] = -5.0e5
    check("holes contribute exactly zero energy whatever the residual is",
          float(measurement_energy(r_perturbed, prec)) == float(energy),
          f"{float(energy):.12f}")

    # And a raw NaN residual on an invalid pixel must not poison the precision.
    r_nan = r.clone()
    r_nan[3, 0, 0] = float("nan")
    prec_nan = build_precision(r_nan, mask, sigma_rgb=0.02, robust=True, nu=5.0,
                               depth_sigma_model=DepthSigmaModel("constant", a=0.05))
    check("a NaN residual on an invalid pixel yields zero, not NaN, precision",
          bool(torch.isfinite(prec_nan).all()) and float(prec_nan[3, 0, 0]) == 0.0)
    check("a NaN residual on a *valid* pixel is auto-invalidated",
          float(build_precision(r_nan, None, sigma_rgb=0.02)[3, 0, 0]) == 0.0)

    stats = normalized_innovation_stats(r, prec, channel_dim=0)
    check("innovation stats count only active measurements",
          stats["active"] == r.numel() - 2, f"{stats['active']} of {r.numel()}")
    check("chi2 equals twice the measurement energy",
          abs(stats["chi2"] - 2.0 * float(energy)) < 1.0e-9)
    check("per-channel breakdown is present", set(stats["channels"]) == {"rgb", "depth"})
    check("the depth channel reports exactly the two holes as inactive",
          stats["channels"]["depth"]["active"] == y_gt[3].numel() - 2)


def test_normalized_innovation_statistics():
    r = torch.tensor([[1.0, -2.0, 3.0, 0.0]], dtype=DT).reshape(4, 1, 1)
    prec = torch.tensor([[1.0, 0.25, 4.0, 0.0]], dtype=DT).reshape(4, 1, 1)
    stats = normalized_innovation_stats(r, prec)
    whitened = torch.tensor([1.0, 1.0, 6.0], dtype=DT)
    check("whitened mean", abs(stats["mean"] - float(whitened.mean())) < 1.0e-12)
    check("whitened median", abs(stats["median"] - float(whitened.median())) < 1.0e-12)
    check("whitened max", abs(stats["max"] - float(whitened.max())) < 1.0e-12)
    check("chi2 is the sum of squared whitened innovations",
          abs(stats["chi2"] - float(whitened.square().sum())) < 1.0e-12)
    check("chi2 per dof divides by the active count only",
          abs(stats["chi2_per_dof"] - float(whitened.square().sum()) / 3.0) < 1.0e-12)
    check("rms is sqrt(chi2/dof)",
          abs(stats["rms"] - math.sqrt(float(whitened.square().sum()) / 3.0)) < 1.0e-12)
    check("active/total are reported", stats["active"] == 3 and stats["total"] == 4)
    check("nll includes the log-determinant term",
          abs(stats["nll"] - (0.5 * float(whitened.square().sum())
                              - 0.5 * math.log(1.0 * 0.25 * 4.0)
                              + 1.5 * math.log(2.0 * math.pi))) < 1.0e-12)

    empty = normalized_innovation_stats(r, torch.zeros_like(prec))
    check("a fully masked frame yields finite zeros, not NaN",
          empty["active"] == 0 and empty["chi2"] == 0.0 and empty["mean"] == 0.0)


# ---------------------------------------------------------------------------
# Section 6: range-dependent depth sigma
# ---------------------------------------------------------------------------
def test_quadratic_depth_sigma_lowers_precision_at_range():
    h, w = 1, 4
    r = torch.zeros((4, h, w), dtype=DT)
    z = torch.tensor([[0.5, 1.0, 2.0, 4.0]], dtype=DT)

    model = DepthSigmaModel("quadratic", a=0.002, b=0.003)
    sigma = channel_sigma(r, sigma_rgb=0.02, depth_sigma_model=model, depth_range=z)
    expected = 0.002 + 0.003 * z.square()
    check("sigma_D(z) = a + b z^2", float((sigma[3] - expected).abs().max()) < 1.0e-12)
    check("the RGB channels ignore the depth model",
          float((sigma[:3] - 0.02).abs().max()) < 1.0e-12)

    prec = build_precision(r, None, sigma_rgb=0.02, depth_sigma_model=model, depth_range=z)
    dprec = prec[3, 0]
    check("quadratic model: precision strictly decreases with range",
          bool((dprec[1:] < dprec[:-1]).all()), str([round(float(p), 3) for p in dprec]))
    check("quadratic precision equals 1/(a+b z^2)^2",
          float((dprec - expected[0].square().reciprocal()).abs().max()) < 1.0e-9)
    check("far pixels are far less informative than near ones",
          float(dprec[0] / dprec[-1]) > 100.0, f"ratio={float(dprec[0] / dprec[-1]):.1f}")

    const = build_precision(r, None, sigma_rgb=0.02,
                            depth_sigma_model=DepthSigmaModel("constant", a=0.05),
                            depth_range=z)
    check("constant model is flat in range", float(const[3].std()) == 0.0)
    lin = channel_sigma(r, 0.02, DepthSigmaModel("linear", a=0.002, b=0.003), depth_range=z)
    check("linear model is a + b z", float((lin[3] - (0.002 + 0.003 * z)).abs().max()) < 1.0e-12)

    check("a range-dependent model refuses to run without a range",
          _raises(lambda: build_precision(r, None, 0.02, model)))
    check("model names are parsed from the CLI-style string",
          make_depth_sigma_model("quadratic:0.002:0.003") == model)
    check("unknown model names are rejected",
          _raises(lambda: make_depth_sigma_model("cubic")))
    check("sigma never underflows to zero",
          float(DepthSigmaModel("quadratic", a=0.0, b=1.0, sigma_min=1e-4)
                .sigma(torch.zeros(1, dtype=DT))[0]) == 1.0e-4)


def _raises(fn) -> bool:
    try:
        fn()
    except (ValueError, TypeError):
        return True
    return False


# ---------------------------------------------------------------------------
# Section 13.1: robust Student-t IRLS
# ---------------------------------------------------------------------------
def test_student_t_weights():
    w = torch.tensor([0.0, 1.0, 2.0, 10.0], dtype=DT)
    check("nu -> inf gives exactly one",
          bool((student_t_weight(w, float("inf")) == 1.0).all()))
    nu = 5.0
    expect = (nu + 1.0) / (nu + w.square())
    check("w_j = (nu+1)/(nu + (r/sigma)^2)",
          float((student_t_weight(w, nu) - expect).abs().max()) < 1.0e-14)
    check("a zero residual is upweighted by (nu+1)/nu",
          abs(float(student_t_weight(w, nu)[0]) - 1.2) < 1.0e-14)
    check("a 10-sigma outlier is strongly downweighted",
          float(student_t_weight(w, nu)[3]) < 0.06,
          f"w={float(student_t_weight(w, nu)[3]):.4f}")
    check("weights decrease monotonically with the residual",
          bool((student_t_weight(w, nu)[1:] < student_t_weight(w, nu)[:-1]).all()))
    check("non-positive nu is rejected", _raises(lambda: student_t_weight(w, 0.0)))

    h, w_px = 1, 3
    r = torch.zeros((4, h, w_px), dtype=DT)
    r[0, 0, 1] = 0.5  # a 25-sigma colour outlier
    sigma_rgb = 0.02
    plain = build_precision(r, None, sigma_rgb=sigma_rgb, robust=False)
    robust = build_precision(r, None, sigma_rgb=sigma_rgb, robust=True, nu=nu)
    gaussian_limit = build_precision(r, None, sigma_rgb=sigma_rgb, robust=True, nu=float("inf"))
    check("robust precision equals the sensor precision at zero residual",
          abs(float(robust[1, 0, 0]) - float(plain[1, 0, 0]) * (nu + 1.0) / nu) < 1.0e-6)
    check("the 25-sigma outlier's precision is cut by two orders of magnitude",
          float(robust[0, 0, 1]) < 1.0e-2 * float(plain[0, 0, 1]),
          f"{float(robust[0, 0, 1]):.3e} vs {float(plain[0, 0, 1]):.3e}")
    check("nu = inf reproduces the plain Gaussian precision bit for bit",
          torch.equal(gaussian_limit, plain))
    check("robust reweighting lowers the measurement energy of the outlier",
          float(measurement_energy(r, robust)) < float(measurement_energy(r, plain)))

    mask = torch.ones_like(r, dtype=torch.bool)
    mask[3] = False
    check("robust mode keeps invalid entries at exactly zero",
          bool((build_precision(r, mask, sigma_rgb, robust=True, nu=nu)[3] == 0).all()))

    _, parts = build_precision(r, None, sigma_rgb=sigma_rgb, robust=True, nu=nu, return_parts=True)
    check("return_parts exposes sigma/valid/weight for logging",
          set(parts) == {"sigma", "valid", "weight"}
          and parts["weight"].shape == r.shape)


# ---------------------------------------------------------------------------
# Spec 0 / 36: the production path must contain no autograd and no optimizer
# ---------------------------------------------------------------------------
def test_estimator_modules_are_autograd_free():
    banned_attrs = {"backward", "grad", "requires_grad_", "retain_grad"}
    banned_names = {"Adam", "AdamW", "SGD", "LBFGS"}
    # `estimator/__init__` now re-exports the `pcg` FUNCTION, which shadows the
    # module of the same name for a plain attribute lookup; import the modules
    # explicitly so this keeps testing the source files.
    import importlib

    for module in (importlib.import_module("estimator.measurement"),
                   importlib.import_module("estimator.pcg")):
        src = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in banned_attrs:
                hits.append(node.attr)
            if isinstance(node, ast.Name) and node.id in banned_names:
                hits.append(node.id)
            if isinstance(node, ast.Attribute) and node.attr == "autograd":
                hits.append("autograd")
        check(f"{Path(module.__file__).name} has no autograd/optimizer calls",
              not hits, str(sorted(set(hits))))
        stubs = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Pass) or (isinstance(n, ast.Expr)
                                                and isinstance(n.value, ast.Constant)
                                                and n.value.value is Ellipsis)]
        check(f"{Path(module.__file__).name} contains no stub bodies", not stubs)
        check(f"{Path(module.__file__).name} contains no TODO markers",
              "TODO" not in src and "FIXME" not in src)

    # The estimator path never builds a graph even if handed graph-carrying input.
    r = torch.randn((4, 2, 2), dtype=DT, requires_grad=True)
    prec = build_precision(r, None, sigma_rgb=0.02, robust=True, nu=4.0)
    check("build_precision returns a detached tensor", not prec.requires_grad)
    check("measurement_energy returns a detached scalar",
          not measurement_energy(r, prec).requires_grad)
    b = torch.randn((3, 4), dtype=DT, requires_grad=True)
    a = spd(12, seed=131)
    x, _ = pcg(CountingOperator(a, (3, 4)), b, None, max_iters=50, tol=1.0e-10)
    check("pcg returns a detached solution", not x.requires_grad)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
