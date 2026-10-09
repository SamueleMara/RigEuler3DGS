"""Local differential Killing field quantities (FIRM v3 §4-§9, §12, §13, §18).

Everything here is per-Gaussian.  There is deliberately no global (omega_k, b_k):
v3 §8 forbids imposing ``omega_ik == omega_k_bar`` before the support is known.
Rigidity is instead expressed as *local zero strain* plus *local twist
coherence* over the membership-weighted neighbourhood graph.
"""

from __future__ import annotations

import torch

from .neighborhood import NeighborGraph, fit_local_jacobian, local_coherence_loss


def vee(skew: torch.Tensor) -> torch.Tensor:
    return torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1)


def evaluate_local_field(
    points_by_part: torch.Tensor,
    centers: torch.Tensor,
    control_velocity: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The original local Gaussian/cell velocity field, evaluated per Gaussian.

    ``points_by_part`` [K,N,3]; ``centers``/``control_velocity`` [K,C,3].
    Returns (velocity [K,N,3], analytic Jacobian [K,N,3,3], kernel weight [K,N,C])
    with ``J[k,n,m,d] = d v_m / d x_d``.
    """
    x = points_by_part
    c = centers[:, None, :, :]
    diff = x[:, :, None, :] - c
    sigma2 = sigma.reshape(-1, 1, 1).square().clamp_min(1.0e-10)
    logits = -0.5 * diff.square().sum(dim=-1) / sigma2
    weight = torch.softmax(logits, dim=-1)
    velocity = torch.einsum("knc,kcm->knm", weight, control_velocity)
    grad_logits = -diff / sigma2[..., None]
    mean_grad = torch.einsum("knc,kncd->knd", weight, grad_logits)
    grad_weight = weight[..., None] * (grad_logits - mean_grad[:, :, None, :])
    jacobian = torch.einsum("kncd,kcm->knmd", grad_weight, control_velocity)
    return velocity, jacobian, weight


def local_killing_quantities(
    points_by_part: torch.Tensor,
    centers: torch.Tensor,
    control_velocity: torch.Tensor,
    sigma: torch.Tensor,
    part_prob_moving: torch.Tensor,
    graph: NeighborGraph | None = None,
    edge_weight: torch.Tensor | None = None,
    *,
    jacobian_mode: str = "analytic",
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    """Per-Gaussian velocity, Jacobian, strain, omega, b and the v3 local losses.

    ``part_prob_moving`` is [N,K] for the moving classes only.
    """
    velocity, jac_analytic, kernel_weight = evaluate_local_field(
        points_by_part, centers, control_velocity, sigma
    )
    if str(jacobian_mode) == "lsq":
        if graph is None or edge_weight is None:
            raise ValueError("jacobian_mode='lsq' requires a neighbour graph and edge weights")
        # §4: G_ik = argmin_G sum_j w_ij^(k) ||(v_jk - v_ik) - G (x_j - x_i)||^2
        jac = fit_local_jacobian(points_by_part, velocity, graph, edge_weight)
    else:
        jac = jac_analytic

    # §5 strain / §6 rotation tensor.
    strain = 0.5 * (jac + jac.transpose(-1, -2))
    skew = 0.5 * (jac - jac.transpose(-1, -2))
    omega_local = vee(skew)                                            # §6
    b_local = velocity - torch.cross(omega_local, points_by_part, dim=-1)  # §7

    prob = part_prob_moving.transpose(0, 1).to(points_by_part)  # [K,N]
    mass = prob.sum(dim=1).clamp_min(float(eps))

    # §5: per-part normalised strain, summed over k.
    strain_loss = ((strain.square().sum(dim=(-2, -1)) * prob).sum(dim=1) / mass).sum()

    # §8: local coherence, NOT a hard global equality.
    if graph is not None and edge_weight is not None:
        omega_coherence = local_coherence_loss(omega_local, graph, edge_weight, eps=eps)
        b_coherence = local_coherence_loss(b_local, graph, edge_weight, eps=eps)
    else:
        zero = velocity.sum() * 0.0
        omega_coherence, b_coherence = zero, zero

    # Diagnostics only: the membership-weighted mean twist. Never used to drive
    # integration (that would be the forbidden global hard twist of §8).
    omega_mean = (prob[:, :, None] * omega_local).sum(dim=1) / mass[:, None]
    b_mean = (prob[:, :, None] * b_local).sum(dim=1) / mass[:, None]
    center = (prob[:, :, None] * points_by_part).sum(dim=1) / mass[:, None]
    center_velocity = b_mean + torch.cross(omega_mean, center, dim=-1)

    return {
        "velocity": velocity,
        "jacobian": jac,
        "kernel_weight": kernel_weight,
        "strain": strain,
        "omega_local": omega_local,
        "b_local": b_local,
        "strain_loss": strain_loss,
        "omega_coherence": omega_coherence,
        "b_coherence": b_coherence,
        "omega_mean": omega_mean,
        "b_mean": b_mean,
        "center": center,
        "center_velocity": center_velocity,
    }


def _solve_symmetric_3x3(h: torch.Tensor, g: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    """Closed-form solve of ``H omega = g`` for symmetric 3x3 ``H``.

    Deliberately avoids ``torch.linalg.solve``/cuSOLVER: this repository has
    already been bitten by cuSOLVER on small batched systems, and a 3x3 system
    has an exact adjugate solution that is cheaper, fully differentiable, and
    free of any host synchronisation.
    """
    a, b, c = h[..., 0, 0], h[..., 0, 1], h[..., 0, 2]
    d, e = h[..., 1, 1], h[..., 1, 2]
    f = h[..., 2, 2]

    # Adjugate of a symmetric 3x3 (itself symmetric).
    adj_aa = d * f - e * e
    adj_ab = c * e - b * f
    adj_ac = b * e - c * d
    adj_bb = a * f - c * c
    adj_bc = b * c - a * e
    adj_cc = a * d - b * b

    det = a * adj_aa + b * adj_ab + c * adj_ac
    safe_det = torch.where(det.abs() < float(eps), torch.full_like(det, float(eps)), det)

    gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
    x = (adj_aa * gx + adj_ab * gy + adj_ac * gz) / safe_det
    y = (adj_ab * gx + adj_bb * gy + adj_bc * gz) / safe_det
    z = (adj_ac * gx + adj_bc * gy + adj_cc * gz) / safe_det
    return torch.stack((x, y, z), dim=-1)


def _symmetric_3x3_eigenvalues(h: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of a symmetric 3x3, computed on CPU for diagnostics only.

    Kept off the GPU/cuSOLVER path on purpose; the tensor is [K,3,3].
    """
    with torch.no_grad():
        return torch.linalg.eigvalsh(h.detach().cpu()).to(h.device)


def analytic_partwise_twist(
    xyz: torch.Tensor,
    velocity: torch.Tensor,
    part_weight: torch.Tensor,
    update_weight: torch.Tensor | None = None,
    hard_label: torch.Tensor | None = None,
    *,
    eps: float = 1.0e-8,
    reg: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    """FIRM v4 §6-§9, §20: ONE instantaneous rigid twist per part, fitted
    analytically from ALL weighted Gaussian velocities of that part.

    This is the projection of the learned local velocity field onto the rigid
    Killing subspace -- NOT Kabsch, and not a per-Gaussian transform:

        a_ik = p_ik u_i,                 A_k = sum_i a_ik
        c_k  = sum a x / A,              u_k = sum a v / A
        r_ik = x_i - c_k,                dv  = v_i - u_k
        H_k  = sum a (||r||^2 I - r r^T)
        g_k  = sum a (r x dv)
        H_k omega_k = g_k,               b_k = u_k - omega_k x c_k

    Fully differentiable: nothing on the path render -> T_k -> omega_k,b_k ->
    v_i -> field parameters is detached (§13).

    ``xyz``/``velocity`` are [K,N,3] (per-part current positions and the local
    field sampled there); ``part_weight`` is [N,K]; ``update_weight`` is [N] or
    [K,N].  Returns per-part center, center velocity, omega, b, the rigid-speed
    residual and §21 conditioning diagnostics.
    """
    pos = torch.as_tensor(xyz)
    vel = torch.as_tensor(velocity, dtype=pos.dtype, device=pos.device)
    w = torch.as_tensor(part_weight, dtype=pos.dtype, device=pos.device).transpose(0, 1)  # [K,N]
    if update_weight is None:
        a = w
    else:
        u_w = torch.as_tensor(update_weight, dtype=pos.dtype, device=pos.device)
        a = w * (u_w[None] if u_w.ndim == 1 else u_w)

    # FINAL spec §5/§17: a_ik = 1[z_i=k] p_ik u_i. Only the Gaussians whose
    # CURRENT winning class is k may shape part k's twist. Using p_ik for every
    # Gaussian regardless of its winning class is what let nearly the whole
    # object contaminate the moving-part twist.
    if hard_label is not None:
        z = torch.as_tensor(hard_label, device=pos.device).reshape(-1)
        classes = torch.arange(1, int(w.shape[0]) + 1, device=pos.device)
        a = a * (z[None, :] == classes[:, None]).to(a.dtype)

    total = a.sum(dim=1).clamp_min(float(eps))                       # A_k
    center = (a[:, :, None] * pos).sum(dim=1) / total[:, None]       # c_k
    center_velocity = (a[:, :, None] * vel).sum(dim=1) / total[:, None]  # u_k

    r = pos - center[:, None, :]
    dv = vel - center_velocity[:, None, :]

    eye = torch.eye(3, dtype=pos.dtype, device=pos.device)
    r2 = r.square().sum(dim=-1)                                       # [K,N]
    outer = r[:, :, :, None] * r[:, :, None, :]                       # [K,N,3,3]
    h = (a[:, :, None, None] * (r2[:, :, None, None] * eye - outer)).sum(dim=1)   # H_k
    g = (a[:, :, None] * torch.cross(r, dv, dim=-1)).sum(dim=1)                   # g_k

    # §21: never let a degenerate support produce a giant angular velocity.
    scale = torch.diagonal(h, dim1=-2, dim2=-1).sum(-1).clamp_min(0.0) / 3.0
    damping = (float(reg) * (scale + 1.0))[:, None, None]
    h_reg = h + damping * eye
    omega = _solve_symmetric_3x3(h_reg, g, eps=eps)
    omega = torch.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)

    b = center_velocity - torch.cross(omega, center, dim=-1)          # §8

    rigid_velocity = torch.cross(omega[:, None, :].expand_as(pos), pos, dim=-1) + b[:, None, :]
    residual = ((a * (vel - rigid_velocity).square().sum(dim=-1)).sum(dim=1) / total).sum()  # §9

    with torch.no_grad():
        eigenvalues = _symmetric_3x3_eigenvalues(h_reg)
        lo = eigenvalues[:, 0].abs().clamp_min(1.0e-30)
        condition = eigenvalues[:, -1].abs() / lo
        effective_points = (a.sum(dim=1).square() / a.square().sum(dim=1).clamp_min(float(eps)))

    return {
        "center": center,
        "center_velocity": center_velocity,
        "omega": omega,
        "b": b,
        "rigid_velocity": rigid_velocity,
        "rigid_speed_residual": residual,
        "H": h,
        "g": g,
        "effective_weight": total.detach(),
        "effective_num_points": effective_points,
        "eigenvalues": eigenvalues,
        "condition_number": condition,
    }


def partwise_motion_separation(
    omega: torch.Tensor,
    center_velocity: torch.Tensor,
    *,
    margin: float,
    alpha_omega: float = 1.0,
    alpha_u: float = 1.0,
    embedding_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FIRM v4 §16: separation acts on the PARTWISE twists.

        D_kl = alpha_w ||w_k - w_l||^2 + alpha_u ||u_k - u_l||^2
        L    = sum_{k<l} w_kl max(0, m - D_kl)^2

    ``omega``/``center_velocity`` are [F,K,3] trajectories (or [K,3] for one
    frame); D is averaged over frames.  Structurally zero at K=1.
    """
    w = torch.as_tensor(omega)
    u = torch.as_tensor(center_velocity, dtype=w.dtype, device=w.device)
    if w.ndim == 2:
        w, u = w[None], u[None]
    k_parts = int(w.shape[1])
    dist = torch.zeros((k_parts, k_parts), dtype=w.dtype, device=w.device)
    if k_parts < 2:
        return w.sum() * 0.0, dist
    loss = w.sum() * 0.0
    for a in range(k_parts):
        for c in range(a + 1, k_parts):
            d = (
                float(alpha_omega) * (w[:, a] - w[:, c]).square().sum(-1)
                + float(alpha_u) * (u[:, a] - u[:, c]).square().sum(-1)
            ).mean()
            weight = 1.0 if embedding_weight is None else embedding_weight[a, c]
            loss = loss + weight * torch.relu(float(margin) - d).square()
            dist[a, c] = d.detach()
            dist[c, a] = d.detach()
    return loss, dist


def confidence_gate(
    valid: torch.Tensor,
    confidence: torch.Tensor,
    *,
    c_min: float = 0.60,
    gamma: float = 2.0,
) -> torch.Tensor:
    """§12: g_i = m_i * [clamp((c_i - c_min)/(1 - c_min), 0, 1)]^gamma."""
    m = torch.as_tensor(valid).to(confidence.dtype)
    cmin = min(max(float(c_min), 0.0), 0.999999)
    scaled = ((confidence - cmin) / (1.0 - cmin)).clamp(0.0, 1.0)
    return m * scaled.pow(float(gamma))


def huber(residual_norm: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    d = float(delta)
    small = residual_norm.abs() <= d
    return torch.where(small, 0.5 * residual_norm.square(), d * (residual_norm.abs() - 0.5 * d))


def observed_velocity_loss(
    predicted_velocity_by_part: torch.Tensor,
    observed_state: dict | None,
    part_prob_moving: torch.Tensor,
    *,
    velocity_scale: float = 1.0,
    c_min: float = 0.60,
    gamma: float = 2.0,
    huber_delta: float = 1.0,
    min_high_conf: int = 16,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """§12: confidence-gated, robust, per-part normalised observed-velocity loss.

        L_obs = sum_k [ sum_i p_ik g_i rho_Huber(r_ik / s_v) ] / [ sum_i p_ik g_i + eps ]

    If a part/frame has too few high-confidence observations the term is zero.
    """
    zero = predicted_velocity_by_part.sum() * 0.0
    stats = {
        "num_valid": 0,
        "coverage_fraction": 0.0,
        "mean_confidence": 0.0,
        "median_confidence": 0.0,
        "num_above_cmin": 0,
        "high_conf_coverage": 0.0,
    }
    if observed_state is None:
        return zero, stats

    target = observed_state["velocity"].detach().to(predicted_velocity_by_part)
    valid = observed_state["valid"].detach().to(device=predicted_velocity_by_part.device).bool()
    confidence = observed_state["confidence"].detach().to(predicted_velocity_by_part).clamp(0.0, 1.0)
    n = int(target.shape[0])
    valid = valid & torch.isfinite(target).all(dim=-1)

    g = confidence_gate(valid, confidence, c_min=c_min, gamma=gamma)
    above = (confidence >= float(c_min)) & valid
    stats = {
        "num_valid": int(valid.sum()),
        "coverage_fraction": float(valid.float().mean()) if n else 0.0,
        "mean_confidence": float(confidence[valid].mean()) if bool(valid.any()) else 0.0,
        "median_confidence": float(confidence[valid].median()) if bool(valid.any()) else 0.0,
        "num_above_cmin": int(above.sum()),
        "high_conf_coverage": float(above.float().mean()) if n else 0.0,
    }
    if int(above.sum()) < int(min_high_conf):
        # Too few trustworthy observations to say anything about this frame.
        return zero, stats

    scale = max(float(velocity_scale), 1.0e-6)
    prob = part_prob_moving.transpose(0, 1).to(predicted_velocity_by_part)  # [K,N]
    residual = (predicted_velocity_by_part - target[None]) / scale
    rho = huber(residual.norm(dim=-1), delta=float(huber_delta))  # [K,N]
    w = prob * g[None]
    loss = ((w * rho).sum(dim=1) / w.sum(dim=1).clamp_min(float(eps))).sum()
    return loss, stats


def effective_obs_lambda(
    base_lambda: float,
    primary_loss: torch.Tensor,
    obs_loss: torch.Tensor,
    *,
    rho_max: float = 0.20,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """§13: cap the observed-velocity share of the inner objective.

        lambda_eff = min(lambda_base, rho/(1-rho) * L_primary / (L_obs + eps))

    Detached: never backpropagate through the coefficient.
    """
    base = torch.as_tensor(float(base_lambda), dtype=obs_loss.dtype, device=obs_loss.device)
    rho = min(max(float(rho_max), 0.0), 0.999999)
    with torch.no_grad():
        cap = (rho / (1.0 - rho)) * primary_loss.detach() / (obs_loss.detach() + float(eps))
        return torch.minimum(base, cap.clamp_min(0.0)).detach()


def field_separation_loss(
    centers: torch.Tensor,
    control_velocity: torch.Tensor,
    sigma: torch.Tensor,
    diagnostic_points: torch.Tensor,
    *,
    margin: float,
    alpha_v: float = 1.0,
    alpha_omega: float = 1.0,
    embedding_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """§18: separate *local fields* on a shared diagnostic point set.

        D_kl = mean_x [ alpha_v ||v_k(x) - v_l(x)||^2 + alpha_w ||w_k(x) - w_l(x)||^2 ]
        L    = sum_{k<l} w_kl max(0, m - D_kl)^2

    Returns (loss, D matrix).  Structurally inactive at K=1.
    """
    k_parts = int(centers.shape[0])
    dev, dt = centers.device, centers.dtype
    dist = torch.zeros((k_parts, k_parts), dtype=dt, device=dev)
    if k_parts < 2:
        return control_velocity.sum() * 0.0, dist

    x = diagnostic_points.to(device=dev, dtype=dt)[None].expand(k_parts, -1, -1)
    velocity, jac, _ = evaluate_local_field(x, centers, control_velocity, sigma)
    omega = vee(0.5 * (jac - jac.transpose(-1, -2)))

    loss = control_velocity.sum() * 0.0
    for a in range(k_parts):
        for b in range(a + 1, k_parts):
            d = (
                float(alpha_v) * (velocity[a] - velocity[b]).square().sum(-1)
                + float(alpha_omega) * (omega[a] - omega[b]).square().sum(-1)
            ).mean()
            w = 1.0 if embedding_weight is None else embedding_weight[a, b]
            loss = loss + w * torch.relu(float(margin) - d).square()
            dist[a, b] = d.detach()
            dist[b, a] = d.detach()
    return loss, dist


def integrate_local_se3(
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    omega_local: torch.Tensor,
    b_local: torch.Tensor,
    delta_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """§9: per-Gaussian exact SE(3) transport.

        Delta T_ik = exp(dt * xi_ik),  x_ik(t+1) = Delta T_ik x_ik(t),
        q_ik(t+1)  = q_Delta,ik (x) q_ik(t)

    No Euler integration; uses the already-tested exact spatial helper.
    """
    from utils.rigid_transform_utils import (
        normalize_quaternion,
        quaternion_multiply,
        se3_exp_spatial,
    )

    r_inc, p_inc, q_inc = se3_exp_spatial(omega_local, b_local, delta_t)
    new_xyz = (r_inc @ positions[..., None])[..., 0] + p_inc
    new_quat = normalize_quaternion(quaternion_multiply(q_inc, quaternions))
    return new_xyz, new_quat
