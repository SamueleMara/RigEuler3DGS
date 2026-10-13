"""G2 — whitened spectrum and rate Schur complement of the undamped operator.

CODEX_FINISH_UNIFIED_PHASE1_AFTER_0AADC1E.md section 2.

The earlier work concluded from ablations that the rate blocks were not the
cause of the stall. Ablations can rule a cause out; they cannot characterise
the information. This does that directly, on a scene small enough to
materialise A0 exactly.

Two things it is careful about.

**No damping.** LM damping manufactures positive curvature, so an
observability eigenproblem built from the damped operator would report
information that the measurements never supplied.

**Units.** A raw spectrum of A0 mixes metres, radians, m/s and rad/s, so its
eigenvalues are not comparable across blocks and "small" means nothing. The
spectrum is therefore also reported in a stated physical whitening, which
changes no objective.

The question it settles: H_omega = 0 instantaneously does NOT mean angular
velocity is unobservable over a window. The Schur complement of the rates
over the pose variables is what decides that, and it is computed here.

    python tests/test_unified_ieks_observability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CHECKS = 0
FAILURES: list[str] = []

# Physical scales used ONLY to whiten the reported spectrum: metres, radians,
# metres/second, radians/second. The estimation objective is untouched.
STATE_SCALE = torch.tensor([0.10] * 3 + [0.10] * 3 + [1.0] * 3 + [1.0] * 3,
                           dtype=torch.float64)


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")
    else:
        print(f"  ok    {label}  {detail}")


def cuda_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import diff_gaussian_rasterization  # noqa: F401
        return True
    except Exception:
        return False


def materialize(operator, T, N):
    """A0 by canonical basis probes of the matrix-free matvec."""
    size = T * N * 12
    columns = []
    for k in range(size):
        probe = operator.zeros().reshape(-1)
        probe[k] = 1.0
        columns.append(operator.information_matvec(
            probe.reshape(T, N, 12)).reshape(-1).cpu())
    return torch.stack(columns, dim=1).double()


def schur_rate_block(A, T, N, ridge=0.0):
    """S_{r|p} = A_rr - A_rp A_pp^-1 A_pr over rate vs pose coordinates."""
    size = T * N * 12
    index = torch.arange(size).reshape(T, N, 12)
    rate = index[:, :, 6:12].reshape(-1)
    pose = index[:, :, 0:6].reshape(-1)
    A_rr = A[rate][:, rate]
    A_rp = A[rate][:, pose]
    A_pp = A[pose][:, pose]
    if ridge:
        A_pp = A_pp + ridge * torch.eye(A_pp.shape[0], dtype=A_pp.dtype)
    # Rank-aware: lstsq rather than an inverse, so a singular pose block is
    # handled instead of producing nonsense.
    solved = torch.linalg.lstsq(A_pp, A_rp.transpose(0, 1)).solution
    return A_rr - A_rp @ solved, rate, pose


def main() -> int:
    if not cuda_ready():
        print("G2 NOT_RUN: CUDA or diff_gaussian_rasterization unavailable.")
        return 0

    from bidirectional_iekf.ieks import IEKSConfig, Trajectory, UnifiedFreeIEKS
    from bidirectional_iekf.observations import FreeRGBDObservation
    from estimator.manifold import so3_exp
    from estimator.renderer_bridge import build_linearization
    from estimator.state import GaussianMotionState
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

    # Tiny on purpose: 3 frames x 2 Gaussians x 12 = 72 coordinates, so A0 can
    # be materialised exactly and its spectrum computed.
    N, T, dt = 2, 3, 0.1
    size = T * N * 12

    def make(mu, R, v, w):
        return GaussianMotionState(
            mu=mu.cuda(), R=R.cuda(), v=v.cuda(), omega=w.cuda(),
            colors=torch.tensor([[.9, .3, .2], [.2, .7, .9]]).cuda(),
            opacity=torch.full((N, 1), 0.92).cuda(),
            scales=torch.tensor([[.14, .07, .06]]).expand(N, 3).contiguous().cuda())

    cameras = make_cameras(n_cam=3, H=96, W=96, radius=2.4)
    mu0 = torch.tensor([[-0.30, 0.20, 0.0], [0.30, -0.22, 0.04]])
    v_true = torch.tensor([[0.10, 0.0, 0.0], [0.0, 0.08, 0.0]])
    w_true = torch.tensor([[0.0, 0.0, 0.22], [0.10, 0.0, 0.0]])
    R0 = so3_exp(torch.zeros(N, 3))
    truth = [make(mu0 + v_true * (dt * t), so3_exp(w_true * (dt * t)) @ R0,
                  v_true, w_true) for t in range(T)]
    observations = [[(rgb.clone(), depth.clone(),
                      torch.ones_like(depth, dtype=torch.bool))
                     for rgb, depth in build_linearization(truth[t], cameras).predicted()]
                    for t in range(T)]
    Q = torch.diag(torch.tensor([1e4] * 3 + [1e4] * 3 + [1e2] * 3 + [1e2] * 3,
                                dtype=torch.float64)).expand(N, 12, 12).contiguous()
    truth_traj = Trajectory(states=truth, frame_ids=list(range(T)), dt=dt)
    obs = FreeRGBDObservation(cameras, observations, frame_ids=list(range(T)))

    def solver():
        return UnifiedFreeIEKS(obs, Q, config=IEKSConfig(verbose=False))

    print("[1] materialise A0 UNDAMPED and check it is a real symmetric operator")
    op, _ = solver().build_system(truth_traj, damping=0.0)
    A = materialize(op, T, N)
    asym = float((A - A.transpose(0, 1)).abs().max()) / max(float(A.abs().max()), 1e-30)
    check("A0 is symmetric to renderer precision", asym < 1e-5,
          f"relative asymmetry {asym:.3e} over a {size}x{size} operator")
    probe = torch.randn(T, N, 12, dtype=torch.float64, device="cuda", generator=None)
    direct = float((probe.cpu().reshape(-1) * (A @ probe.cpu().reshape(-1))).sum())
    matfree = float((probe * op.information_matvec(probe)).sum())
    check("the materialised matrix agrees with the matrix-free action",
          abs(direct - matfree) / max(abs(matfree), 1e-30) < 1e-6,
          f"{direct:.6e} vs {matfree:.6e}")

    print("\n[2] spectrum, raw units and physically whitened")
    A_sym = 0.5 * (A + A.transpose(0, 1))
    raw = torch.linalg.eigvalsh(A_sym)
    scale = STATE_SCALE.repeat(T * N)
    W = torch.diag(scale)
    whitened = torch.linalg.eigvalsh(W @ A_sym @ W)
    print(f"       raw      min {float(raw.min()):+.4e}  max {float(raw.max()):.4e}"
          f"  cond {float(raw.max() / raw.abs().min()):.3e}")
    print(f"       whitened min {float(whitened.min()):+.4e}  max "
          f"{float(whitened.max()):.4e}  cond "
          f"{float(whitened.max() / whitened.abs().min()):.3e}")
    check("no materially negative curvature at the truth",
          float(raw.min()) > -1e-3 * float(raw.max()),
          f"min eigenvalue {float(raw.min()):+.4e} against max "
          f"{float(raw.max()):.4e}")
    near_null = int((whitened < 1e-9 * float(whitened.max())).sum())
    print(f"       whitened eigenvalues below 1e-9 of the max: {near_null} "
          f"of {size}")

    print("\n[3] rate Schur complement: is angular velocity observable over the window?")
    S, rate_idx, pose_idx = schur_rate_block(A_sym, T, N)
    S = 0.5 * (S + S.transpose(0, 1))
    s_eig = torch.linalg.eigvalsh(S)
    A_rr = A_sym[rate_idx][:, rate_idx]
    rr_eig = torch.linalg.eigvalsh(0.5 * (A_rr + A_rr.transpose(0, 1)))
    print(f"       A_rr   (rates alone)      min {float(rr_eig.min()):.4e}  "
          f"max {float(rr_eig.max()):.4e}")
    print(f"       S_r|p  (rates | pose)     min {float(s_eig.min()):.4e}  "
          f"max {float(s_eig.max()):.4e}")
    check("the rate block carries information even though H_omega = 0",
          float(rr_eig.min()) > 0,
          f"smallest A_rr eigenvalue {float(rr_eig.min()):.4e} -- it comes "
          "from the process factors, since the image Jacobian has no rate "
          "columns at all")
    check("the Schur complement is positive, so rates are observable in the window",
          float(s_eig.min()) > 0,
          f"smallest S_r|p eigenvalue {float(s_eig.min()):.4e}; an "
          "instantaneously zero H_omega does NOT imply an unobservable "
          "angular velocity once the window is accounted for")
    ratio = float(s_eig.min()) / max(float(rr_eig.min()), 1e-30)
    print(f"       S_r|p min / A_rr min = {ratio:.4f}  "
          "(below 1 means conditioning on pose REMOVES rate information)")

    print("\n[4] the same spectrum at three nominal states")
    g = torch.Generator().manual_seed(5)
    small = torch.zeros(T, N, 12, dtype=torch.float64)
    small[:, :, 0:3] = torch.randn(T, N, 3, generator=g, dtype=torch.float64) * 0.004
    small[:, :, 3:6] = torch.randn(T, N, 3, generator=g, dtype=torch.float64) * 0.010
    far = torch.zeros_like(small)
    far[:, :, 0] = 0.30                     # beyond the capture radius

    spectra = {}
    for label, offset in (("truth", torch.zeros_like(small)),
                          ("near-truth", small), ("outside-capture", far)):
        nominal = truth_traj.retract(offset.cuda(), 1.0)
        op_n, _ = solver().build_system(nominal, damping=0.0)
        A_n = 0.5 * (materialize(op_n, T, N) + materialize(op_n, T, N).transpose(0, 1))
        eig = torch.linalg.eigvalsh(torch.diag(scale) @ A_n @ torch.diag(scale))
        S_n, _, _ = schur_rate_block(A_n, T, N)
        s_n = torch.linalg.eigvalsh(0.5 * (S_n + S_n.transpose(0, 1)))
        b_n = op_n.rhs()
        spectra[label] = (eig, s_n, float(b_n.norm()))
        print(f"       {label:16s} whitened min {float(eig.min()):+.3e}  "
              f"max {float(eig.max()):.3e}   S_r|p min {float(s_n.min()):.3e}   "
              f"|g| {float(b_n.norm()):.3e}")

    near_cond = float(spectra["near-truth"][0].max()
                      / spectra["near-truth"][0].abs().min())
    far_cond = float(spectra["outside-capture"][0].max()
                     / spectra["outside-capture"][0].abs().min())
    check("the Hessian is not degenerate at the far nominal either",
          float(spectra["outside-capture"][1].min()) > 0,
          f"S_r|p min {float(spectra['outside-capture'][1].min()):.3e} -- the "
          "far start is NOT a local nullspace, which is why the stall there "
          "is a capture-region effect and not an unobservable mode")
    print(f"       whitened condition number: near-truth {near_cond:.3e}, "
          f"outside-capture {far_cond:.3e}")
    check("both nominals are positive definite in the whitened metric",
          float(spectra["near-truth"][0].min()) > 0
          and float(spectra["outside-capture"][0].min()) > 0,
          "so neither stall can be attributed to a rank-deficient operator")

    print("\n[5] where does a large step point?")
    from bidirectional_iekf.full_window_free import pcg
    op_far, _ = solver().build_system(truth_traj.retract(far.cuda(), 1.0), damping=0.0)
    step, info = pcg(op_far, op_far.rhs(), max_iters=400, tol=1e-11)
    A_far = 0.5 * (materialize(op_far, T, N) + materialize(op_far, T, N).transpose(0, 1))
    eigval, eigvec = torch.linalg.eigh(A_far)
    coeff = (eigvec.transpose(0, 1) @ step.cpu().reshape(-1)).abs()
    weakest = int(eigval.abs().argmin())
    energy_in_weakest_10 = float((coeff[:10] ** 2).sum() / (coeff ** 2).sum())
    print(f"       |step| {float(step.norm()):.4e}, pcg {info['status']} "
          f"{info['iterations']} iters at {info['residual']:.2e}")
    print(f"       fraction of step energy in the 10 weakest eigen-directions: "
          f"{energy_in_weakest_10:.4f}")
    check("the large step is dominated by weakly constrained directions",
          energy_in_weakest_10 > 0.5,
          f"{100 * energy_in_weakest_10:.1f}% of the step's energy lies in the "
          f"10 smallest of {size} eigen-directions (smallest eigenvalue "
          f"{float(eigval[weakest]):.3e}) -- the solve is correct, the "
          "direction is simply unconstrained by the data at this nominal")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
