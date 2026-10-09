"""Tests for the IEKF discrete process model (spec 5) and block covariance /
information propagation (spec 17).

Run with:
    cd /home/samuelemara/Joint3DGS/RigEuler3DGS && PYTHONPATH=. python tests/test_iekf_process_covariance.py

Everything defaults to CPU float64.  The finite-difference oracle below is a
plain numerical difference of the production ``process_predict`` -- no autograd
is used anywhere (spec section 36), so these tests exercise the same code path
production does.
"""

import math

import torch

from estimator.covariance import (
    BlockInverseDiagnostics,
    apply_prior,
    approximate_posterior_information,
    covariance_predict,
    invert_blocks,
    make_process_noise,
)
from estimator.manifold import so3_exp
from estimator.process import (
    FIRST_ORDER,
    SO3,
    angular_increment,
    error_state_transition,
    process_predict,
)
from estimator.state import GaussianMotionState

DTYPE = torch.float64
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# helpers / independent oracles
# ---------------------------------------------------------------------------


def _so3_log(R):
    """Independent SO(3) logarithm oracle (deliberately not the repo's own).

    Using a second implementation here means a sign/convention bug in
    ``manifold.so3_exp`` cannot cancel itself out in the boxminus check.
    """
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    skew = 0.5 * (R - R.transpose(-1, -2))
    axis = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)
    small = theta < 1e-7
    # (R - R^T)/2 = sin(theta) [u]_x, so scale by theta / sin(theta).
    factor = torch.where(
        small, 1.0 + theta * theta / 6.0, theta / torch.sin(theta).clamp_min(1e-300)
    )
    return axis * factor.unsqueeze(-1)


def _boxminus(a, b):
    """delta such that ``b boxplus delta == a`` under the left retraction."""
    return torch.cat(
        [
            a.mu - b.mu,
            _so3_log(a.R @ b.R.transpose(-1, -2)),
            a.v - b.v,
            a.omega - b.omega,
        ],
        dim=-1,
    )


def _random_rotation(n, generator):
    phi = torch.randn(n, 3, generator=generator, dtype=DTYPE, device=DEVICE) * 0.7
    return so3_exp(phi)


def _make_state(mu, R, v, omega):
    """Build a state with arbitrary but valid FROZEN appearance (spec 3.1).

    Nothing in process.py / covariance.py may read colors/opacity/scales; they
    are supplied only because the state container validates their presence, and
    the tests below check they survive a prediction untouched.
    """
    n = mu.shape[0]
    return GaussianMotionState(
        mu=mu,
        R=R,
        v=v,
        omega=omega,
        colors=torch.full((n, 3), 0.5, dtype=mu.dtype, device=mu.device),
        opacity=torch.full((n, 1), 0.8, dtype=mu.dtype, device=mu.device),
        scales=torch.full((n, 3), 0.05, dtype=mu.dtype, device=mu.device),
    )


def _random_state(n, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    mu = torch.randn(n, 3, generator=g, dtype=DTYPE, device=DEVICE)
    R = _random_rotation(n, g)
    v = torch.randn(n, 3, generator=g, dtype=DTYPE, device=DEVICE) * 0.5
    omega = torch.randn(n, 3, generator=g, dtype=DTYPE, device=DEVICE) * 0.4
    return _make_state(mu, R, v, omega)


def _random_spd(n, dim, seed, scale=1.0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    A = torch.randn(n, dim, dim, generator=g, dtype=DTYPE, device=DEVICE)
    M = A @ A.transpose(-1, -2)
    # 0.5 * (M + M^T) is EXACTLY symmetric in IEEE arithmetic, whereas A @ A^T
    # alone is only symmetric up to round-off; the structural assertions below
    # need the exact version so they test the code, not the generator.
    M = 0.5 * (M + M.transpose(-1, -2))
    return scale * (M + dim * torch.eye(dim, dtype=DTYPE, device=DEVICE))


def _min_eigenvalue(A):
    return torch.linalg.eigvalsh(0.5 * (A + A.transpose(-1, -2))).min().item()


def _assert_exactly_symmetric(A, what):
    """Bitwise symmetry check.

    ``torch.allclose(A, A.T, atol=1e-18)`` LOOKS exact but still carries the
    default ``rtol=1e-5``, so it passes on a matrix that is 0.001% asymmetric
    and cannot tell whether the module symmetrized at all.  ``F P F^T`` is only
    symmetric up to round-off, so the explicit symmetrization is load bearing
    and the test has to be able to see it.
    """
    assert torch.equal(A, A.transpose(-1, -2)), f"{what} is not bitwise symmetric"


# ---------------------------------------------------------------------------
# process model
# ---------------------------------------------------------------------------


def test_boxplus_uses_left_so3_retraction():
    """Guard: F is derived for ``R+ = Exp(dphi) R``; fail loudly if state disagrees."""
    state = _random_state(4, seed=11)
    delta = torch.zeros(4, 12, dtype=DTYPE, device=DEVICE)
    delta[:, 0:3] = torch.tensor([0.1, -0.2, 0.3], dtype=DTYPE)
    delta[:, 3:6] = torch.tensor([0.02, 0.05, -0.03], dtype=DTYPE)
    delta[:, 6:9] = torch.tensor([-0.4, 0.1, 0.2], dtype=DTYPE)
    delta[:, 9:12] = torch.tensor([0.07, -0.01, 0.009], dtype=DTYPE)
    moved = state.boxplus(delta)
    assert torch.allclose(moved.mu, state.mu + delta[:, 0:3], atol=1e-12)
    assert torch.allclose(moved.R, so3_exp(delta[:, 3:6]) @ state.R, atol=1e-12)
    assert torch.allclose(moved.v, state.v + delta[:, 6:9], atol=1e-12)
    assert torch.allclose(moved.omega, state.omega + delta[:, 9:12], atol=1e-12)
    # and the boxminus oracle must invert it, agreeing with the state module's own
    assert torch.allclose(_boxminus(moved, state), delta, atol=1e-12)
    assert torch.allclose(moved.boxminus(state), delta, atol=1e-12)


def test_process_predict_zero_dt_is_identity():
    state = _random_state(32, seed=1)
    pred = process_predict(state, 0.0)
    assert torch.allclose(pred.mu, state.mu, atol=1e-14)
    assert torch.allclose(pred.R, state.R, atol=1e-12)
    assert torch.allclose(pred.v, state.v, atol=1e-14)
    assert torch.allclose(pred.omega, state.omega, atol=1e-14)


def test_process_predict_zero_twist_is_identity():
    state = _random_state(16, seed=2)
    still = _make_state(state.mu, state.R, torch.zeros_like(state.v), torch.zeros_like(state.omega))
    pred = process_predict(still, 0.0413)
    assert torch.allclose(pred.mu, still.mu, atol=1e-14)
    assert torch.allclose(pred.R, still.R, atol=1e-12)
    assert torch.allclose(pred.v, still.v, atol=1e-14)
    assert torch.allclose(pred.omega, still.omega, atol=1e-14)


def test_process_predict_does_not_alias_input():
    state = _random_state(5, seed=3)
    pred = process_predict(state, 0.1)
    pred.v.add_(1.0)
    pred.omega.add_(1.0)
    assert not torch.allclose(pred.v, state.v)
    assert not torch.allclose(pred.omega, state.omega)
    # frozen appearance rides along untouched (spec section 3)
    assert torch.equal(pred.colors, state.colors)
    assert torch.equal(pred.opacity, state.opacity)
    assert torch.equal(pred.scales, state.scales)


def test_constant_twist_composition_over_ten_steps():
    """10 x dt=0.1 must equal 1 x dt=1.0 for translation and for rotation."""
    state = _random_state(8, seed=4)
    rolled = state
    for _ in range(10):
        rolled = process_predict(rolled, 0.1)
    one_shot = process_predict(state, 1.0)

    assert torch.allclose(rolled.mu, state.mu + state.v * 1.0, atol=1e-12)
    assert torch.allclose(rolled.mu, one_shot.mu, atol=1e-12)
    # Exp is a one-parameter subgroup for a constant axis, so composing ten
    # small increments reproduces Exp(omega * 1.0) exactly (up to round-off).
    expected_R = so3_exp(state.omega * 1.0) @ state.R
    assert torch.allclose(rolled.R, expected_R, atol=1e-11)
    assert torch.allclose(rolled.R, one_shot.R, atol=1e-11)
    assert torch.allclose(rolled.v, state.v, atol=1e-14)
    assert torch.allclose(rolled.omega, state.omega, atol=1e-14)


def test_dt_is_real_seconds_not_unit_frame_spacing():
    state = _random_state(6, seed=5)
    dt = 1.0 / 29.97  # real NTSC frame spacing
    pred = process_predict(state, dt)
    assert torch.allclose(pred.mu, state.mu + state.v * dt, atol=1e-14)
    assert not torch.allclose(pred.mu, state.mu + state.v, atol=1e-6)
    # per-Gaussian dt must also be honoured
    per_gauss = torch.full((6,), dt, dtype=DTYPE, device=DEVICE)
    per_gauss[0] = 2.0 * dt
    pred2 = process_predict(state, per_gauss)
    assert torch.allclose(pred2.mu[0], state.mu[0] + state.v[0] * 2.0 * dt, atol=1e-14)
    assert torch.allclose(pred2.mu[1:], pred.mu[1:], atol=1e-14)


def test_angular_increment_consistency():
    state = _random_state(7, seed=6)
    dt = 0.0317
    Omega, Omega_hat, exp_omega = angular_increment(state.omega, dt)
    assert torch.allclose(Omega, state.omega * dt, atol=1e-15)
    assert torch.allclose(Omega_hat, -Omega_hat.transpose(-1, -2), atol=1e-15)
    assert torch.allclose(Omega_hat @ Omega.unsqueeze(-1), torch.zeros(7, 3, 1, dtype=DTYPE), atol=1e-14)
    assert torch.allclose(exp_omega, so3_exp(Omega), atol=1e-14)


def test_first_order_transition_matches_spec_block_form():
    state = _random_state(3, seed=7)
    dt = 0.045
    F = error_state_transition(state, dt, mode=FIRST_ORDER)
    eye = torch.eye(3, dtype=DTYPE, device=DEVICE).expand(3, 3, 3)
    zero = torch.zeros(3, 3, 3, dtype=DTYPE, device=DEVICE)
    assert F.shape == (3, 12, 12)
    assert torch.allclose(F[:, 0:3, 0:3], eye, atol=1e-15)
    assert torch.allclose(F[:, 3:6, 3:6], eye, atol=1e-15)
    assert torch.allclose(F[:, 6:9, 6:9], eye, atol=1e-15)
    assert torch.allclose(F[:, 9:12, 9:12], eye, atol=1e-15)
    assert torch.allclose(F[:, 0:3, 6:9], dt * eye, atol=1e-15)
    assert torch.allclose(F[:, 3:6, 9:12], dt * eye, atol=1e-15)
    assert torch.allclose(F[:, 0:3, 3:6], zero, atol=1e-15)
    assert torch.allclose(F[:, 6:9, 0:6], torch.zeros(3, 3, 6, dtype=DTYPE), atol=1e-15)
    assert torch.allclose(F[:, 9:12, 0:9], torch.zeros(3, 3, 9, dtype=DTYPE), atol=1e-15)


def test_so3_transition_matches_finite_difference():
    """The real test of F: numerical directional derivatives of the predictor."""
    torch.manual_seed(1234)
    state = _make_state(
        torch.tensor([[0.3, -1.1, 2.0]], dtype=DTYPE),
        so3_exp(torch.tensor([[0.4, -0.25, 0.9]], dtype=DTYPE)),
        torch.tensor([[-0.7, 0.2, 1.3]], dtype=DTYPE),
        torch.tensor([[0.8, -1.4, 0.55]], dtype=DTYPE),
    )
    dt = 0.0431
    F = error_state_transition(state, dt, mode=SO3)
    base = process_predict(state, dt)

    def _forward_difference(eps):
        numeric = torch.zeros(1, 12, 12, dtype=DTYPE)
        for j in range(12):
            delta = torch.zeros(1, 12, dtype=DTYPE)
            delta[0, j] = eps
            perturbed = process_predict(state.boxplus(delta), dt)
            numeric[:, :, j] = _boxminus(perturbed, base) / eps
        return numeric

    # Truncation-dominated regime: forward differences are first order, so the
    # error must fall linearly with eps.  Below ~1e-5 cancellation round-off
    # takes over, which is why the sweep stops there.
    errors = []
    for eps in (1e-3, 1e-4, 1e-5):
        err = (_forward_difference(eps) - F).abs().max().item()
        errors.append(err)
        assert err < 1e-3 * eps + 1e-12, f"so3 F mismatch at eps={eps}: {err}"
    for coarse, fine in zip(errors, errors[1:]):
        # a systematic Jacobian bug would show up as a floor that does not move
        assert fine < 0.2 * coarse, f"error did not shrink with eps: {errors}"

    # Round-off-dominated regime: still small, just no longer O(eps).
    for eps in (1e-6, 1e-7):
        assert (_forward_difference(eps) - F).abs().max().item() < 1e-7

    # central differences: second order accurate, so a far tighter bound holds
    eps = 1e-5
    central = torch.zeros(1, 12, 12, dtype=DTYPE)
    for j in range(12):
        delta = torch.zeros(1, 12, dtype=DTYPE)
        delta[0, j] = eps
        plus = process_predict(state.boxplus(delta), dt)
        minus = process_predict(state.boxplus(-delta), dt)
        central[:, :, j] = _boxminus(plus, minus) / (2.0 * eps)
    assert (central - F).abs().max().item() < 1e-9


def test_so3_transition_finite_difference_large_rotation():
    """Same check at a large per-step rotation, where Exp/J_l matter most."""
    state = _make_state(
        torch.tensor([[1.0, 2.0, -0.5]], dtype=DTYPE),
        so3_exp(torch.tensor([[-1.2, 0.3, 0.6]], dtype=DTYPE)),
        torch.tensor([[0.1, -0.2, 0.3]], dtype=DTYPE),
        torch.tensor([[2.1, -1.7, 0.9]], dtype=DTYPE),
    )
    dt = 0.5  # |omega * dt| ~ 1.5 rad, far outside the small-angle regime
    F = error_state_transition(state, dt, mode=SO3)
    base = process_predict(state, dt)
    eps = 1e-5
    central = torch.zeros(1, 12, 12, dtype=DTYPE)
    for j in range(12):
        delta = torch.zeros(1, 12, dtype=DTYPE)
        delta[0, j] = eps
        plus = process_predict(state.boxplus(delta), dt)
        minus = process_predict(state.boxplus(-delta), dt)
        central[:, :, j] = _boxminus(plus, minus) / (2.0 * eps)
    assert (central - F).abs().max().item() < 1e-8

    # the first-order baseline is visibly WRONG at this rotation magnitude,
    # which is exactly why the so3 mode exists
    F_first = error_state_transition(state, dt, mode=FIRST_ORDER)
    assert (central - F_first).abs().max().item() > 1e-2


def test_so3_reduces_to_first_order_as_dt_shrinks():
    """Convergence rates of the two modes, block by block.

    The velocity-coupling block ``dt * J_l(Omega)`` agrees with ``dt * I`` to
    O(dt^2).  The adjoint block ``Exp(Omega)`` agrees with ``I`` only to O(dt),
    because the baseline drops the frame rotation of the attitude error
    altogether -- that missing term is the whole point of the 'so3' mode, so
    the test pins the two different rates rather than pretending both are
    second order.
    """
    state = _random_state(4, seed=8)
    full_gaps = []
    cross_gaps = []
    deltas = (0.1, 0.05, 0.025, 0.0125)
    for dt in deltas:
        F_fo = error_state_transition(state, dt, mode=FIRST_ORDER)
        F_so3 = error_state_transition(state, dt, mode=SO3)
        gap = (F_so3 - F_fo).abs()
        full_gaps.append(gap.max().item())
        cross_gaps.append(gap[:, 3:6, 9:12].max().item())
        # only the rotational row can differ at all
        assert gap[:, 0:3, :].max().item() == 0.0
        assert gap[:, 6:12, :].max().item() == 0.0

    for coarse, fine in zip(full_gaps, full_gaps[1:]):
        assert 0.4 * coarse < fine < 0.6 * coarse, f"adjoint gap is not O(dt): {full_gaps}"
    for coarse, fine in zip(cross_gaps, cross_gaps[1:]):
        assert 0.15 * coarse < fine < 0.35 * coarse, f"J_l gap is not O(dt^2): {cross_gaps}"
    assert full_gaps[-1] < full_gaps[0]
    assert cross_gaps[-1] < cross_gaps[0]

    # and the limit is exact at dt = 0
    F_fo0 = error_state_transition(state, 0.0, mode=FIRST_ORDER)
    F_so30 = error_state_transition(state, 0.0, mode=SO3)
    assert torch.allclose(F_fo0, F_so30, atol=1e-14)
    assert torch.allclose(F_so30, torch.eye(12, dtype=DTYPE).expand(4, 12, 12), atol=1e-14)


def test_error_state_transition_rejects_unknown_mode():
    state = _random_state(2, seed=9)
    try:
        error_state_transition(state, 0.1, mode="euler")
    except ValueError:
        return
    raise AssertionError("unknown transition mode must raise ValueError")


# ---------------------------------------------------------------------------
# covariance / information
# ---------------------------------------------------------------------------


def test_make_process_noise_is_spd_and_scales_with_dt():
    n = 5
    dt = 1.0 / 24.0
    Q = make_process_noise(n, dt, 1e-3, 1e-3, 0.4, 0.3, device=DEVICE, dtype=DTYPE)
    assert Q.shape == (n, 12, 12)
    _assert_exactly_symmetric(Q, "Q")
    assert _min_eigenvalue(Q) > 0.0
    # documented white-noise-acceleration structure
    assert abs(Q[0, 6, 6].item() - 0.4 ** 2 * dt) < 1e-15
    assert abs(Q[0, 0, 6].item() - 0.4 ** 2 * dt ** 2 / 2.0) < 1e-15
    assert abs(Q[0, 0, 0].item() - (1e-3 ** 2 * dt + 0.4 ** 2 * dt ** 3 / 3.0)) < 1e-15
    assert abs(Q[0, 9, 9].item() - 0.3 ** 2 * dt) < 1e-15
    assert abs(Q[0, 3, 9].item() - 0.3 ** 2 * dt ** 2 / 2.0) < 1e-15
    # zero dt injects no noise at all
    Q0 = make_process_noise(n, 0.0, 1e-3, 1e-3, 0.4, 0.3, device=DEVICE, dtype=DTYPE)
    assert torch.allclose(Q0, torch.zeros_like(Q0), atol=1e-18)
    # densities: the velocity random walk is additive over sub-intervals
    Qa = make_process_noise(1, 0.3, 0.0, 0.0, 0.7, 0.5, device=DEVICE, dtype=DTYPE)
    Qb = make_process_noise(1, 0.2, 0.0, 0.0, 0.7, 0.5, device=DEVICE, dtype=DTYPE)
    Qc = make_process_noise(1, 0.5, 0.0, 0.0, 0.7, 0.5, device=DEVICE, dtype=DTYPE)
    assert abs((Qa[0, 6, 6] + Qb[0, 6, 6] - Qc[0, 6, 6]).item()) < 1e-15
    # per-Gaussian sigmas
    sig = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=DTYPE)
    Qp = make_process_noise(n, dt, 0.0, 0.0, sig, 0.3, device=DEVICE, dtype=DTYPE)
    assert torch.allclose(Qp[:, 6, 6], sig * sig * dt, atol=1e-15)


def test_covariance_predict_preserves_symmetry_and_positive_definiteness():
    n = 24
    state = _random_state(n, seed=10)
    dt = 1.0 / 30.0
    P = _random_spd(n, 12, seed=20, scale=1e-2)
    Q = make_process_noise(n, dt, 1e-4, 1e-4, 0.2, 0.2, device=DEVICE, dtype=DTYPE)
    F = error_state_transition(state, dt, mode=SO3)
    P_next = covariance_predict(P, F, Q)
    assert P_next.shape == (n, 12, 12)
    _assert_exactly_symmetric(P_next, "P_next")
    assert _min_eigenvalue(P_next) > 0.0
    # matches the textbook formula
    reference = F @ P @ F.transpose(-1, -2) + Q
    assert (P_next - 0.5 * (reference + reference.transpose(-1, -2))).abs().max().item() < 1e-14
    # uncertainty never decreases through a pure prediction
    assert _min_eigenvalue(P_next) > 0.0
    for i in range(n):
        growth = P_next[i].trace().item() - (F[i] @ P[i] @ F[i].T).trace().item()
        assert growth > 0.0

    # 200 chained predictions stay symmetric and PD (round-off accumulation)
    rolled = P
    rolling_state = state
    for _ in range(200):
        Fk = error_state_transition(rolling_state, dt, mode=SO3)
        rolled = covariance_predict(rolled, Fk, Q)
        rolling_state = process_predict(rolling_state, dt)
    assert torch.isfinite(rolled).all()
    _assert_exactly_symmetric(rolled, "rolled P after 200 predictions")
    assert _min_eigenvalue(rolled) > 0.0


def test_invert_blocks_round_trip():
    n = 40
    P = _random_spd(n, 12, seed=30, scale=3.0)
    Lambda, diag_a = invert_blocks(P)
    assert isinstance(diag_a, BlockInverseDiagnostics)
    assert diag_a.n_blocks == n and diag_a.ok
    eye = torch.eye(12, dtype=DTYPE, device=DEVICE).expand(n, 12, 12)
    assert (P @ Lambda - eye).abs().max().item() < 1e-9
    P_again, diag_b = invert_blocks(Lambda)
    assert diag_b.ok
    rel = (P_again - P).abs().max().item() / P.abs().max().item()
    assert rel < 1e-9, f"round trip relative error {rel}"
    _assert_exactly_symmetric(Lambda, "Lambda")
    # works for the 6x6 pose blocks too (same routine, any square size)
    S = _random_spd(n, 6, seed=31)
    S_inv, diag_c = invert_blocks(S)
    assert diag_c.ok
    eye6 = torch.eye(6, dtype=DTYPE, device=DEVICE).expand(n, 6, 6)
    assert (S @ S_inv - eye6).abs().max().item() < 1e-9


def test_invert_blocks_reports_eigenvalue_fallback():
    """A rank-deficient block must be reported, never silently returned as garbage."""
    P = _random_spd(3, 12, seed=32)
    # kill one direction entirely -> singular, Cholesky must fail even jittered
    u = torch.zeros(12, dtype=DTYPE, device=DEVICE)
    u[4] = 1.0
    P[1] = P[1] - torch.outer(P[1] @ u, P[1] @ u) / (u @ P[1] @ u)
    P[2] = -_random_spd(1, 12, seed=33)[0]  # negative definite: not a covariance
    Lambda, diag = invert_blocks(P, jitter_steps=2, base_relative_jitter=1e-14)
    assert diag.n_blocks == 3
    assert diag.n_eigen_clamped >= 1, "singular/indefinite blocks must be reported"
    assert diag.n_fallback == diag.n_eigen_clamped
    assert not diag.ok
    assert torch.isfinite(Lambda).all()
    # the healthy block is still inverted exactly
    assert (P[0] @ Lambda[0] - torch.eye(12, dtype=DTYPE)).abs().max().item() < 1e-9


def test_invert_blocks_zero_information_block_stays_finite():
    """A Gaussian carrying NO information must not come back as NaN.

    Lambda == 0 is reachable (a freshly spawned Gaussian that nothing has
    observed yet).  A purely RELATIVE jitter underflows to a denormal there,
    Cholesky still reports success, and the triangular solve overflows -- so
    the block would be returned as NaN while the diagnostics claimed it had
    merely been jittered.
    """
    Lambda = _random_spd(4, 12, seed=34)
    Lambda[2] = 0.0  # no information at all
    P, diag = invert_blocks(Lambda)
    assert torch.isfinite(P).all(), "zero-information block produced a non-finite inverse"
    assert not diag.ok, "regularizing a zero block must be reported"
    # zero information => enormous but finite covariance, and far larger than
    # the covariance of the well-informed neighbours
    assert P[2].diagonal().min().item() > 0.0
    assert P[2].abs().max().item() > 1e3 * P[0].abs().max().item()
    # the healthy blocks are untouched by their degenerate neighbour
    eye = torch.eye(12, dtype=DTYPE, device=DEVICE)
    for i in (0, 1, 3):
        assert (Lambda[i] @ P[i] - eye).abs().max().item() < 1e-9


def test_invert_blocks_quarantines_non_finite_blocks():
    """One NaN Gaussian must not abort the batch (torch.linalg.eigh raises)."""
    P = _random_spd(5, 12, seed=35)
    P[1, 0, 0] = float("nan")
    P[3, 4, 7] = float("inf")
    Lambda, diag = invert_blocks(P)
    assert diag.n_blocks == 5
    assert diag.n_non_finite == 2, "non-finite blocks must be counted"
    assert diag.n_fallback >= 2 and not diag.ok
    # the corruption is contained: nothing non-finite leaks into the output,
    # so a downstream apply_prior cannot spread it across the state
    assert torch.isfinite(Lambda).all()
    assert torch.equal(Lambda[1], torch.zeros(12, 12, dtype=DTYPE, device=DEVICE))
    assert torch.equal(Lambda[3], torch.zeros(12, 12, dtype=DTYPE, device=DEVICE))
    eye = torch.eye(12, dtype=DTYPE, device=DEVICE)
    for i in (0, 2, 4):
        assert (P[i] @ Lambda[i] - eye).abs().max().item() < 1e-9


def test_empty_gaussian_batch_is_supported_everywhere():
    """N == 0 (every Gaussian culled) must flow through, not crash one routine."""
    state = _make_state(
        torch.zeros(0, 3, dtype=DTYPE),
        torch.zeros(0, 3, 3, dtype=DTYPE),
        torch.zeros(0, 3, dtype=DTYPE),
        torch.zeros(0, 3, dtype=DTYPE),
    )
    dt = 0.04
    pred = process_predict(state, dt)
    assert pred.mu.shape == (0, 3)
    F = error_state_transition(state, dt, mode=SO3)
    assert F.shape == (0, 12, 12)
    Q = make_process_noise(0, dt, 1e-3, 1e-3, 0.1, 0.1, device=DEVICE, dtype=DTYPE)
    assert Q.shape == (0, 12, 12)
    P = covariance_predict(torch.zeros(0, 12, 12, dtype=DTYPE), F, Q)
    assert P.shape == (0, 12, 12)
    Lambda, diag = invert_blocks(P)
    assert Lambda.shape == (0, 12, 12) and diag.n_blocks == 0 and diag.ok
    assert apply_prior(Lambda, torch.zeros(0, 12, dtype=DTYPE)).shape == (0, 12)
    # a negative count is still a programming error
    try:
        make_process_noise(-1, dt, 1e-3, 1e-3, 0.1, 0.1, device=DEVICE, dtype=DTYPE)
    except ValueError:
        return
    raise AssertionError("negative n must raise ValueError")


def test_approximate_posterior_information_increases_information():
    n = 16
    Lambda_prior = _random_spd(n, 12, seed=40, scale=0.5)
    fisher = _random_spd(n, 6, seed=41, scale=2.0)
    Lambda_post = approximate_posterior_information(Lambda_prior, fisher)
    assert Lambda_post.shape == (n, 12, 12)
    gain = Lambda_post - Lambda_prior
    _assert_exactly_symmetric(gain, "information gain")
    assert torch.linalg.eigvalsh(gain).min().item() > -1e-12  # PSD
    # the gain lives exactly in the pose sub-block; velocity keeps the prior
    assert torch.allclose(gain[:, :6, :6], fisher, atol=1e-14)
    assert torch.allclose(gain[:, 6:, :], torch.zeros(n, 6, 12, dtype=DTYPE), atol=1e-18)
    assert torch.allclose(gain[:, :, 6:], torch.zeros(n, 12, 6, dtype=DTYPE), atol=1e-18)
    # information up => covariance down in the Loewner order
    P_prior, d0 = invert_blocks(Lambda_prior)
    P_post, d1 = invert_blocks(Lambda_post)
    assert d0.ok and d1.ok
    assert torch.linalg.eigvalsh(P_prior - P_post).min().item() > -1e-10
    # a 12x12 Fisher (direct 12D solve) is accepted as well
    fisher12 = _random_spd(n, 12, seed=42, scale=0.1)
    post12 = approximate_posterior_information(Lambda_prior, fisher12)
    assert torch.allclose(post12 - Lambda_prior, fisher12, atol=1e-14)


def test_apply_prior_matches_explicit_block_matmul():
    n = 11
    g = torch.Generator(device=DEVICE).manual_seed(50)
    Lambda = _random_spd(n, 12, seed=51)
    v = torch.randn(n, 12, generator=g, dtype=DTYPE, device=DEVICE)
    out = apply_prior(Lambda, v)
    assert out.shape == (n, 12)
    for i in range(n):
        assert torch.allclose(out[i], Lambda[i] @ v[i], atol=1e-13)
    # A deliberately NON-symmetric block: with a symmetric Lambda the einsum
    # cannot be told apart from its transpose, so the orientation of the
    # contraction would go unchecked.
    g2 = torch.Generator(device=DEVICE).manual_seed(53)
    A = torch.randn(n, 12, 12, generator=g2, dtype=DTYPE, device=DEVICE)
    outa = apply_prior(A, v)
    for i in range(n):
        assert torch.allclose(outa[i], A[i] @ v[i], atol=1e-12)
        assert not torch.allclose(outa[i], A[i].T @ v[i], atol=1e-6)
    # linearity (the PCG operator relies on it)
    w = torch.randn(n, 12, generator=g, dtype=DTYPE, device=DEVICE)
    assert torch.allclose(apply_prior(Lambda, v + 2.0 * w), out + 2.0 * apply_prior(Lambda, w), atol=1e-12)
    # 6x6 pose-only blocks work as well
    S = _random_spd(n, 6, seed=52)
    vs = torch.randn(n, 6, generator=g, dtype=DTYPE, device=DEVICE)
    outs = apply_prior(S, vs)
    for i in range(n):
        assert torch.allclose(outs[i], S[i] @ vs[i], atol=1e-13)


def test_information_round_trip_through_a_full_predict_update_cycle():
    """End-to-end spec 5 + 17 cycle on block marginals only (never dense)."""
    n = 12
    dt = 1.0 / 25.0
    state = _random_state(n, seed=60)
    P = _random_spd(n, 12, seed=61, scale=1e-2)
    Q = make_process_noise(n, dt, 1e-4, 1e-4, 0.3, 0.3, device=DEVICE, dtype=DTYPE)
    fisher = _random_spd(n, 6, seed=62, scale=50.0)

    for _ in range(25):
        F = error_state_transition(state, dt, mode=SO3)
        P = covariance_predict(P, F, Q)
        Lambda_prior, d0 = invert_blocks(P)
        assert d0.ok
        Lambda_post = approximate_posterior_information(Lambda_prior, fisher)
        P, d1 = invert_blocks(Lambda_post)
        assert d1.ok
        state = process_predict(state, dt)

    assert torch.isfinite(P).all()
    assert _min_eigenvalue(P) > 0.0
    # measurements must keep the pose marginal bounded despite process noise
    assert P[:, :6, :6].diagonal(dim1=-2, dim2=-1).max().item() < 1.0
    # rotations stay on SO(3) after 25 compositions
    eye = torch.eye(3, dtype=DTYPE, device=DEVICE).expand(n, 3, 3)
    assert (state.R @ state.R.transpose(-1, -2) - eye).abs().max().item() < 1e-12
    assert abs(torch.linalg.det(state.R).min().item() - 1.0) < 1e-12


def test_shape_validation_rejects_malformed_blocks():
    n = 4
    P = _random_spd(n, 12, seed=70)
    F = torch.eye(12, dtype=DTYPE).expand(n, 12, 12)
    Q = make_process_noise(n, 0.1, 1e-3, 1e-3, 0.1, 0.1, device=DEVICE, dtype=DTYPE)
    failures = 0
    for call in (
        lambda: covariance_predict(P[:, :6, :6], F, Q),
        lambda: covariance_predict(P, F[:2], Q),
        lambda: approximate_posterior_information(P, torch.zeros(n, 7, 7, dtype=DTYPE)),
        lambda: apply_prior(P, torch.zeros(n, 6, dtype=DTYPE)),
        lambda: make_process_noise(n, -1.0, 1e-3, 1e-3, 0.1, 0.1, device=DEVICE, dtype=DTYPE),
        lambda: make_process_noise(n, 0.1, -1.0, 1e-3, 0.1, 0.1, device=DEVICE, dtype=DTYPE),
    ):
        try:
            call()
        except ValueError:
            failures += 1
    assert failures == 6, f"expected 6 ValueErrors, got {failures}"


def test_float32_and_device_dtype_are_preserved():
    """The estimator must not silently promote; CUDA tensors must stay on CUDA."""
    n = 6
    g = torch.Generator().manual_seed(80)
    state = _make_state(
        torch.randn(n, 3, generator=g).float(),
        so3_exp(torch.randn(n, 3, generator=g).double() * 0.3).float(),
        torch.randn(n, 3, generator=g).float(),
        torch.randn(n, 3, generator=g).float(),
    )
    pred = process_predict(state, 0.04)
    assert pred.mu.dtype == torch.float32 and pred.R.dtype == torch.float32
    F = error_state_transition(state, 0.04, mode=SO3)
    assert F.dtype == torch.float32 and F.device == state.mu.device
    Q = make_process_noise(n, 0.04, 1e-3, 1e-3, 0.1, 0.1, device=state.mu.device, dtype=torch.float32)
    P = covariance_predict(torch.eye(12).expand(n, 12, 12).contiguous(), F, Q)
    assert P.dtype == torch.float32
    Lambda, diag = invert_blocks(P)
    assert Lambda.dtype == torch.float32 and diag.n_eigen_clamped == 0


def _collect():
    return [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]


def main():
    tests = _collect()
    passed = 0
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # report every failure, do not stop at the first
            failed.append((name, exc))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed}/{len(tests)} checks passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
