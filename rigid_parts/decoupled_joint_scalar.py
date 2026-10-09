"""CODEX_DECOUPLED_JOINT_KILLING_AND_SCALAR_TRAJECTORY primitives.

Two strictly separated subproblems per moving part k (§4, §64):

  A. JOINT / KILLING GEOMETRY -- optimize theta_k only. Speed is eliminated
     ANALYTICALLY per frame with the nuisance scalar alpha_k,t* (§8), so a
     wrong speed profile can never distort the recovered joint (§9).

  B. SCALAR TRAJECTORY -- freeze theta_k / xi_k*, optimize the whole vector
     s_k(0:T-1) against render + observed magnitude + final PCD + temporal
     regularization (§19-§30).

The functions here are the pieces Stage A and Stage B need that did not exist
for the monolithic shared-scalar solver. Everything that was already correct in
``robust_inner_killing`` (rigid projection, SE(3) exp, scalar regularizers) is
imported rather than reimplemented.

Why alpha* must stay differentiable: §47/§48 require L_frame-joint to be
invariant under Phi -> a Phi. That holds only because alpha*(a Phi) =
alpha*(Phi)/a exactly cancels the rescaling, which in turn requires alpha* to be
computed from the field in the graph rather than detached from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .robust_inner_killing import robust_scalar_weight

EPS = 1.0e-8


# ---------------------------------------------------------------------------
# §10: robust residual
# ---------------------------------------------------------------------------
def charbonnier(residual: torch.Tensor, eps: float = 1.0e-3) -> torch.Tensor:
    """rho(r) = sqrt(r^2 + eps^2) - eps -- smooth, and rho(0) = 0 exactly."""
    e = float(eps)
    return (residual.square() + e * e).sqrt() - e


def robust_residual(norm: torch.Tensor, *, mode: str = "huber", delta: float = 1.0) -> torch.Tensor:
    """§10's rho applied to the NORM of the 3-vector residual.

    Applying rho to the norm (not componentwise) keeps the loss rotationally
    symmetric, which matters because the residual lives in world coordinates and
    no axis is privileged.
    """
    m = str(mode).lower()
    if m == "l2":
        return norm.square()
    if m == "charbonnier":
        return charbonnier(norm, eps=max(float(delta), EPS))
    if m == "huber":
        d = float(delta)
        return torch.where(norm <= d, 0.5 * norm.square(), d * (norm - 0.5 * d))
    raise ValueError(f"unknown robust residual mode {mode!r}")


# ---------------------------------------------------------------------------
# §8: analytical nuisance scalar
# ---------------------------------------------------------------------------
def analytical_frame_scalar(
    psi: torch.Tensor,
    observed_velocity: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """alpha_k,t* = sum_i w psi^T v_hat / (sum_i w ||psi||^2 + eps).   [§8]

    ``psi``                [M, 3]      Phi_k^rigid(x_i), fixed across frames
    ``observed_velocity``  [T, M, 3]   v_hat_k,i,t
    ``weights``            [T, M]      w_k,i,t

    Returns [T]. ONE scalar per (part, frame) -- never per camera (§12, §36).

    Derived, temporary, frame-specific, not a Parameter, not the trajectory
    (§8, §17). It is the exact weighted least-squares optimum of
    ||v_hat - alpha psi||^2_w, so §63's inner ``min_alpha`` is solved in closed
    form rather than by descent.
    """
    p = torch.as_tensor(psi)
    v = torch.as_tensor(observed_velocity)
    w = torch.as_tensor(weights).to(p)
    if int(p.shape[0]) == 0 or int(v.shape[0]) == 0:
        return v.new_zeros(int(v.shape[0]))
    psi_sq = p.square().sum(dim=-1)                                  # [M]
    num = (w * (v * p[None]).sum(dim=-1)).sum(dim=1)                 # [T]
    raw = (w * psi_sq[None]).sum(dim=1)                              # [T]
    # §48 demands EXACT homogeneity: alpha*(a Phi) = alpha*(Phi)/a. Adding an
    # absolute epsilon to the denominator breaks it at small ||Phi||, because the
    # guard stops shrinking while the signal keeps shrinking. So eps decides only
    # WHETHER the frame is supported, against a threshold that is degree-2 in Phi
    # exactly like ``raw`` -- and where it is supported the ratio is exact.
    floor = float(eps) * psi_sq.mean() * w.sum(dim=1).clamp_min(1.0)
    supported = raw > floor
    safe = torch.where(supported, raw, torch.ones_like(raw))         # no 0/0 in backward
    return torch.where(supported, num / safe, torch.zeros_like(num))


@dataclass
class FrameJointFit:
    """Everything Stage A produces for one part over the WHOLE sequence."""

    alpha: torch.Tensor                       # [T]        §17 diagnostic
    loss: torch.Tensor                        # scalar     §10
    per_frame_loss: torch.Tensor              # [T]        §40
    residual: torch.Tensor                    # [T, M]     ||v_hat - alpha* psi||
    robust_weight: torch.Tensor               # [T, M]     §38
    point_residual: torch.Tensor              # [M]        §39
    diagnostics: dict[str, Any] = field(default_factory=dict)


def frame_joint_consistency(
    psi: torch.Tensor,
    observed_velocity: torch.Tensor,
    weights: torch.Tensor,
    *,
    mode: str = "huber",
    delta: float = 1.0,
    robust_tau: float = 1.0,
    irls_steps: int = 1,
    eps: float = EPS,
) -> FrameJointFit:
    """§10's whole-sequence joint consistency loss.

        L_frame-joint = sum_t [ sum_i w rho(v_hat - alpha_t* Phi^rigid) / sum_i w ]

    §38: the robust weight w^robust = rho_w(r) is recomputed from the residual
    (IRLS) and DETACHED, so it reweights the fit without letting the optimizer
    reduce the loss by driving its own weights to zero.

    §39: the per-point median residual over time is returned as a diagnostic. It
    is never written back into the persistent labels inside Stage A.
    """
    p = torch.as_tensor(psi)
    v = torch.as_tensor(observed_velocity)
    w_base = torch.as_tensor(weights).to(p).clamp_min(0.0)
    frames = int(v.shape[0])
    if int(p.shape[0]) == 0 or frames == 0:
        zero = p.sum() * 0.0
        return FrameJointFit(
            alpha=v.new_zeros(frames), loss=zero, per_frame_loss=v.new_zeros(frames),
            residual=v.new_zeros((frames, int(p.shape[0]))),
            robust_weight=v.new_zeros((frames, int(p.shape[0]))),
            point_residual=p.new_zeros(int(p.shape[0])),
            diagnostics={"empty": True})

    w = w_base
    alpha = analytical_frame_scalar(p, v, w, eps=eps)
    residual = v - alpha[:, None, None] * p[None]
    norm = residual.norm(dim=-1)
    for _ in range(max(int(irls_steps) - 1, 0)):
        # §38: protect the joint fit from surviving false dynamic points.
        w = (w_base * robust_scalar_weight(norm.detach(), tau=float(robust_tau))).clamp_min(0.0)
        alpha = analytical_frame_scalar(p, v, w, eps=eps)
        residual = v - alpha[:, None, None] * p[None]
        norm = residual.norm(dim=-1)
    w_robust = robust_scalar_weight(norm.detach(), tau=float(robust_tau))
    w_final = (w_base * w_robust).clamp_min(0.0)

    rho = robust_residual(norm, mode=mode, delta=delta)              # [T, M]
    per_frame = (w_final * rho).sum(dim=1) / (w_final.sum(dim=1) + float(eps))
    loss = per_frame.sum()

    with torch.no_grad():
        point_residual = norm.detach().median(dim=0).values          # §39
        supported = (w_final.sum(dim=1) > 0)
        diagnostics = {
            "frames_supported": int(supported.sum()),
            "alpha_mean": float(alpha.detach()[supported].mean()) if bool(supported.any()) else 0.0,
            "alpha_std": float(alpha.detach()[supported].std(unbiased=False))
            if int(supported.sum()) > 1 else 0.0,
            "alpha_abs_max": float(alpha.detach().abs().max()),
            "alpha_sign_changes": int(
                (alpha.detach()[:-1] * alpha.detach()[1:] < 0).sum()) if frames > 1 else 0,
            "median_point_residual": float(point_residual.median()) if point_residual.numel() else 0.0,
        }
    return FrameJointFit(alpha=alpha, loss=loss, per_frame_loss=per_frame, residual=norm,
                         robust_weight=w_final, point_residual=point_residual,
                         diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# §25: robust final-PCD endpoint geometry
# ---------------------------------------------------------------------------
def _nearest_distance(
    source: torch.Tensor, target: torch.Tensor, *, chunk: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared distance to the nearest target point, chunked over the source."""
    if int(source.shape[0]) == 0 or int(target.shape[0]) == 0:
        return source.new_zeros(int(source.shape[0])), source.new_zeros(
            int(source.shape[0]), dtype=torch.long)
    dists, index = [], []
    for start in range(0, int(source.shape[0]), int(chunk)):
        block = source[start : start + int(chunk)]
        d = torch.cdist(block, target).square()
        best, arg = d.min(dim=1)
        dists.append(best)
        index.append(arg)
    return torch.cat(dists), torch.cat(index)


def trimmed_chamfer(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    forward_trim: float = 0.8,
    backward_trim: float = 0.8,
    backward_budget: float = 1.5,
    chunk: int = 4096,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """§25: bidirectional TRIMMED Chamfer, never one-to-one Gaussian identity.

    Forward  -- every predicted part point must land on some final geometry;
                the worst ``1 - forward_trim`` fraction is dropped so a few
                mislabelled Gaussians cannot dominate the endpoint.

    Backward -- the final cloud contains the static scene and the other parts,
                which this part must NOT be pulled onto. Only the
                ``backward_budget * |predicted|`` final points closest to the
                prediction are considered candidates at all (a part should
                explain roughly as many final points as it owns), and that set
                is then trimmed again. This replaces per-part labels on the
                final cloud, which do not exist.
    """
    pred = torch.as_tensor(predicted)
    tgt = torch.as_tensor(target).to(pred)
    zero = pred.sum() * 0.0
    out = {"forward": zero, "backward": zero, "loss": zero,
           "forward_count": torch.zeros((), dtype=torch.long),
           "backward_count": torch.zeros((), dtype=torch.long)}
    if int(pred.shape[0]) == 0 or int(tgt.shape[0]) == 0:
        return out

    d_fwd, _ = _nearest_distance(pred, tgt, chunk=chunk)
    keep = max(int(round(float(forward_trim) * int(d_fwd.numel()))), 1)
    fwd = torch.topk(d_fwd, keep, largest=False).values.mean()

    with torch.no_grad():
        d_t2p, _ = _nearest_distance(tgt, pred.detach(), chunk=chunk)
        budget = min(max(int(round(float(backward_budget) * int(pred.shape[0]))), 1),
                     int(tgt.shape[0]))
        cand = torch.topk(d_t2p, budget, largest=False).indices
    d_bwd, _ = _nearest_distance(tgt.index_select(0, cand), pred, chunk=chunk)
    keep_b = max(int(round(float(backward_trim) * int(d_bwd.numel()))), 1)
    bwd = torch.topk(d_bwd, keep_b, largest=False).values.mean()

    out.update({"forward": fwd, "backward": bwd, "loss": 0.5 * (fwd + bwd),
                "forward_count": torch.tensor(keep), "backward_count": torch.tensor(keep_b)})
    return out


# ---------------------------------------------------------------------------
# §24: observed magnitude with the LEARNED scalar (speed NOT eliminated)
# ---------------------------------------------------------------------------
def velocity_magnitude_loss(
    psi: torch.Tensor,
    observed_velocity: torch.Tensor,
    weights: torch.Tensor,
    s_mid: torch.Tensor,
    *,
    mode: str = "huber",
    delta: float = 1.0,
    eps: float = EPS,
) -> torch.Tensor:
    """§24: sum_{t,i} w rho(v_hat - s_bar(t) psi).

    Unlike Stage A this does NOT analytically eliminate the speed -- it is the
    term that actually determines the temporal trajectory.
    """
    p = torch.as_tensor(psi)
    v = torch.as_tensor(observed_velocity)
    w = torch.as_tensor(weights).to(p).clamp_min(0.0)
    if int(p.shape[0]) == 0 or int(v.shape[0]) == 0:
        return p.sum() * 0.0 + s_mid.sum() * 0.0
    predicted = s_mid[:, None, None] * p[None]
    norm = (v - predicted).norm(dim=-1)
    rho = robust_residual(norm, mode=mode, delta=delta)
    return (w * rho).sum() / (w.sum() + float(eps))


# ---------------------------------------------------------------------------
# CODEX_MERGED §40: motion DIRECTION (geometry block, speed-invariant)
# ---------------------------------------------------------------------------
def motion_direction_loss(
    psi: torch.Tensor,
    observed_velocity: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """CODEX_MERGED §40's explicit angular loss.

        L = sum_{k,t} [ sum_i w (1 - <v_hat, psi> / (|v_hat| |psi|)) ] / sum_i w

    Strictly a DIRECTION loss: it is invariant to the magnitude of both psi and
    v_hat, so it can inform the joint axis without ever acting as a temporal
    magnitude optimizer (§40's closing requirement). Magnitude is the scalar
    block's job, via ``velocity_magnitude_loss``.

    ``psi`` [M, 3], ``observed_velocity`` [T, M, 3], ``weights`` [T, M].
    """
    p = torch.as_tensor(psi)
    v = torch.as_tensor(observed_velocity)
    w = torch.as_tensor(weights).to(p).clamp_min(0.0)
    if int(p.shape[0]) == 0 or int(v.shape[0]) == 0:
        return p.sum() * 0.0
    # Guard both norms before dividing: a zero observed velocity (an unobserved
    # point) must contribute nothing, not a NaN.
    pn = p.norm(dim=-1).clamp_min(eps)                      # [M]
    vn = v.norm(dim=-1)                                     # [T, M]
    cos = (v * p[None]).sum(dim=-1) / (vn.clamp_min(eps) * pn[None])
    valid = (vn > eps).to(w.dtype)
    ww = w * valid
    per_frame = (ww * (1.0 - cos)).sum(dim=1) / (ww.sum(dim=1) + float(eps))
    return per_frame.sum()


def frame_epoch_schedule(steps: int, frames: list[int], batch: int, *, seed: int = 0) -> list[list[int]]:
    """§58: stochastic frame mini-batches whose epochs cover EVERY frame.

    §58 asks for stride 1 -- every frame and every camera contributing -- and
    explicitly permits mini-batching provided "every frame is visited before the
    epoch completes". Frames are drawn from a shuffled permutation that is only
    reshuffled once exhausted, so coverage is guaranteed rather than expected.
    """
    if not frames or int(batch) <= 0:
        return [[] for _ in range(max(int(steps), 0))]
    generator = torch.Generator().manual_seed(int(seed))
    pool: list[int] = []
    out: list[list[int]] = []
    for _ in range(max(int(steps), 0)):
        picked: list[int] = []
        while len(picked) < min(int(batch), len(frames)):
            if not pool:
                order = torch.randperm(len(frames), generator=generator).tolist()
                pool = [frames[i] for i in order]
            picked.append(pool.pop())
        out.append(picked)
    return out


# ---------------------------------------------------------------------------
# §34: alpha* vs learned s_mid agreement
# ---------------------------------------------------------------------------
@torch.no_grad()
def alpha_scalar_agreement(alpha: torch.Tensor, s_mid: torch.Tensor) -> dict[str, float]:
    """§34/§41: corr, sign agreement and scale ratio. Diagnostic only -- the two
    are deliberately NOT forced equal, because render and final-PCD losses are
    allowed to move the magnitude."""
    a = torch.as_tensor(alpha).reshape(-1).double()
    s = torch.as_tensor(s_mid).reshape(-1).double()
    m = min(int(a.numel()), int(s.numel()))
    a, s = a[:m], s[:m]
    if m < 2:
        return {"corr": float("nan"), "sign_agreement": float("nan"), "scale_ratio": float("nan")}
    ac, sc = a - a.mean(), s - s.mean()
    denom = (ac.norm() * sc.norm()).clamp_min(1.0e-12)
    corr = float((ac * sc).sum() / denom)
    sign = float(((a * s) > 0).double().mean())
    ratio = float(s.abs().mean() / a.abs().mean().clamp_min(1.0e-12))
    return {"corr": corr, "sign_agreement": sign, "scale_ratio": ratio}
