"""T2, T3, T4 — reverse time, the transition F, and the process Jacobians.

CODEX_IMPLEMENT_EXACT_UNIFIED_PHASE1_IEKF_IEKS.md sections 3, 3.1 and 6.

T3 settles a convention the specification warns about: the PDF writes J_r
beside a left rotation perturbation, and the repository's
`error_state_transition(mode='so3')` uses J_l. Rather than change working
code on the strength of a letter, every column of F is compared with central
differences of the code's OWN retraction.

T4 is the one that matters for correctness of the smoother. At zero process
innovation any implementation returning [-F, I] passes, including a wrong
one, so the Jacobians are tested at a genuinely NONZERO rotational innovation
where the J_l^-1(e_R) and J_r^-1(e_R) factors are the whole difference.

    python tests/test_unified_process_jacobians.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bidirectional_iekf.free_dynamics import (  # noqa: E402
    physical_timestamps,
    reverse_to_physical_covariance,
    reverse_to_physical_state,
    verify_transition_against_finite_differences,
)
from bidirectional_iekf.process_factors import (  # noqa: E402
    process_jacobians_finite_difference,
    process_residual,
    process_residual_jacobians,
    so3_right_jacobian_inverse,
    zero_innovation_reduces_to_minus_F,
)
from estimator.manifold import so3_exp, so3_left_jacobian_inverse  # noqa: E402
from estimator.process import error_state_transition, process_predict  # noqa: E402
from estimator.state import GaussianMotionState, boxplus  # noqa: E402

CHECKS = 0
FAILURES: list[str] = []
DT = 1.0 / 30.0


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")


def raises(label: str, fn, needle: str = "") -> None:
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        if needle and needle.lower() not in str(exc).lower():
            FAILURES.append(f"{label}: message lacks {needle!r}: {exc}")
            print(f"  FAIL  {label}  message lacks {needle!r}")
        return
    FAILURES.append(f"{label}: no exception raised")
    print(f"  FAIL  {label}  no exception raised")


def state(n=4, omega_scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return GaussianMotionState(
        mu=torch.randn(n, 3, generator=g, dtype=torch.float64),
        R=so3_exp(torch.randn(n, 3, generator=g, dtype=torch.float64) * 0.4),
        v=torch.randn(n, 3, generator=g, dtype=torch.float64) * 0.1,
        omega=torch.randn(n, 3, generator=g, dtype=torch.float64) * omega_scale,
        colors=torch.rand(n, 3, generator=g, dtype=torch.float64),
        opacity=torch.rand(n, 1, generator=g, dtype=torch.float64),
        scales=torch.rand(n, 3, generator=g, dtype=torch.float64) * 0.05 + 0.01)


# --- T2 reverse time ------------------------------------------------------
def test_reverse_time() -> None:
    print("[T2] reverse processing order -> physical chronological time")
    T, n = 6, 3
    # A chronological truth: constant +x translation, constant +z rotation.
    mu = torch.stack([torch.full((n, 3), 0.0, dtype=torch.float64) for _ in range(T)])
    for t in range(T):
        mu[t, :, 0] = 0.05 * t
    v = torch.full((T, n, 3), 0.0, dtype=torch.float64)
    v[..., 0] = 0.05 / DT
    omega = torch.full((T, n, 3), 0.0, dtype=torch.float64)
    omega[..., 2] = 0.03 / DT
    R = torch.stack([so3_exp(omega[t] * DT * t) for t in range(T)])

    # What a backward run stores: processing order, reverse-time rates.
    mu_r, R_r, v_r, omega_r = mu.flip(0), R.flip(0), (-v).flip(0), (-omega).flip(0)
    mu_p, R_p, v_p, omega_p = reverse_to_physical_state(mu_r, R_r, v_r, omega_r)

    check("positions restored chronologically",
          torch.allclose(mu_p, mu, atol=1e-12))
    check("rotations restored chronologically",
          torch.allclose(R_p, R, atol=1e-12))
    check("translational rate sign restored",
          torch.allclose(v_p, v, atol=1e-12) and bool((v_p[..., 0] > 0).all()),
          f"v_x = {float(v_p[0, 0, 0]):.4f}, expected +{float(v[0, 0, 0]):.4f}")
    check("angular rate sign restored",
          torch.allclose(omega_p, omega, atol=1e-12)
          and bool((omega_p[..., 2] > 0).all()))

    # Reversing WITHOUT negating is the bug this exists to prevent.
    naive = v_r.flip(0)
    check("mere array reversal gives the wrong sign",
          bool((naive[..., 0] < 0).all()),
          "if this passed, reversal alone would look like a valid conversion")

    P = torch.eye(12, dtype=torch.float64).expand(T, 12, 12).clone()
    P[:, 0, 6] = P[:, 6, 0] = 0.3          # a mu-v cross term
    P[:, 3, 9] = P[:, 9, 3] = 0.2          # a phi-omega cross term
    P_phys = reverse_to_physical_covariance(P)
    check("covariance reordered", P_phys.shape == P.shape)
    check("mu-v cross-covariance flips sign",
          torch.allclose(P_phys[:, 0, 6], torch.full((T,), -0.3, dtype=torch.float64)),
          f"{float(P_phys[0, 0, 6]):.3f}")
    check("phi-omega cross-covariance flips sign",
          torch.allclose(P_phys[:, 3, 9], torch.full((T,), -0.2, dtype=torch.float64)))
    check("v-v block keeps its sign",
          torch.allclose(P_phys[:, 6, 6], torch.ones(T, dtype=torch.float64)),
          "two sign flips cancel on a diagonal rate block")
    raises("a non-12x12 covariance is refused",
           lambda: reverse_to_physical_covariance(torch.eye(6)), "12,12")

    check("timestamps strictly increase",
          bool((torch.diff(torch.from_numpy(physical_timestamps(T, DT))) > 0).all()))
    raises("a non-positive dt is refused",
           lambda: physical_timestamps(T, -DT), "positive")


# --- T3 the transition ----------------------------------------------------
def test_transition_columns() -> None:
    print("[T3] every column of F vs central differences of the real retraction")
    for label, scale, tol in (("omega = 0", 0.0, 1e-7),
                              ("small rotation", 0.5, 1e-7),
                              ("large rotation", 8.0, 1e-6),
                              ("near pi per step", 3.14159 / DT, 1e-4)):
        report = verify_transition_against_finite_differences(state(omega_scale=scale), DT)
        check(f"F matches finite differences ({label})",
              report["max_abs_error"] < tol,
              f"max |dF| = {report['max_abs_error']:.3e}")

    # The J_l-vs-J_r question, settled numerically rather than by name.
    s = state(omega_scale=6.0)
    F = error_state_transition(s, DT, mode="so3").double()
    from estimator.manifold import so3_left_jacobian
    omega_dt = s.omega.double() * DT
    check("the (phi, omega) block is dt * J_l(dt omega), not J_r",
          torch.allclose(F[:, 3:6, 9:12], DT * so3_left_jacobian(omega_dt), atol=1e-12),
          "the repository's left perturbation pairs with J_l; the PDF's "
          "letter name alone does not settle it")
    check("the (phi, phi) block is Exp(dt omega)",
          torch.allclose(F[:, 3:6, 3:6], so3_exp(omega_dt), atol=1e-12))


# --- T4 the process residual Jacobians ------------------------------------
def test_process_jacobians() -> None:
    print("[T4] A_proc / B_proc at NONZERO process innovation")
    for label, scale in (("small rotation", 0.5), ("large rotation", 8.0)):
        s = state(omega_scale=scale, seed=1)
        predicted = process_predict(s, DT)
        kick = torch.zeros(s.mu.shape[0], 12, dtype=torch.float64)
        kick[:, 0:3] = 0.10          # translation innovation
        kick[:, 3:6] = 0.25          # rotation innovation -- the point
        nxt = boxplus(predicted, kick)

        e = process_residual(s, nxt, DT)
        check(f"the innovation really is nonzero ({label})",
              float(e[:, 3:6].abs().max()) > 1e-3,
              f"max |e_R| = {float(e[:, 3:6].abs().max()):.3e}")

        fd = process_jacobians_finite_difference(s, nxt, DT)
        check(f"A_proc matches finite differences ({label})",
              fd["max_abs_error_A"] < 1e-7, f"{fd['max_abs_error_A']:.3e}")
        check(f"B_proc matches finite differences ({label})",
              fd["max_abs_error_B"] < 1e-7, f"{fd['max_abs_error_B']:.3e}")

        # And the shortcut is genuinely WRONG here, which is why it is tested.
        A, B = process_residual_jacobians(s, nxt, DT)
        F = error_state_transition(s, DT, mode="so3").double()
        eye = torch.eye(12, dtype=torch.float64).expand_as(B)
        check(f"[-F, I] would be wrong at this innovation ({label})",
              float((A + F).abs().max()) > 1e-4 or float((B - eye).abs().max()) > 1e-4,
              f"|A+F| = {float((A + F).abs().max()):.3e}, "
              f"|B-I| = {float((B - eye).abs().max()):.3e}")

    print("     and the zero-innovation limit")
    for scale in (0.0, 0.5, 8.0):
        report = zero_innovation_reduces_to_minus_F(state(omega_scale=scale), DT)
        check(f"A = -F and B = I at zero innovation (omega scale {scale})",
              report["passes"],
              f"|A+F| = {report['max_abs_A_plus_F']:.2e}, "
              f"|B-I| = {report['max_abs_B_minus_I']:.2e}")

    # J_r^-1(x) = J_l^-1(-x), the identity the rotation row rests on.
    x = torch.randn(10, 3, dtype=torch.float64) * 1.3
    check("J_r^-1 is J_l^-1 of the negated vector",
          torch.allclose(so3_right_jacobian_inverse(x),
                         so3_left_jacobian_inverse(-x), atol=1e-12))
    check("both reduce to I at zero",
          torch.allclose(so3_right_jacobian_inverse(torch.zeros(1, 3, dtype=torch.float64)),
                         torch.eye(3, dtype=torch.float64).unsqueeze(0), atol=1e-12))

    raises("a non-positive dt is refused",
           lambda: process_residual(state(), state(), 0.0), "positive")


def main() -> int:
    test_reverse_time()
    test_transition_columns()
    test_process_jacobians()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
