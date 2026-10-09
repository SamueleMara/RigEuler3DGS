"""P6 -- "do neighbouring source patches agree on the same motion?"

A single valid local rigid transform is not evidence of a moving part.  Noise
produces valid-looking local transforms all the time; what noise does not
produce is a spatially coherent field of transforms that agree with each
other.  Agreement is the check that turns local hypotheses into a part.

Comparing two transforms requires trading rotation against translation, and
that trade has no natural constant -- radians and metres are not
commensurable, and any chosen ratio is a tuned weight of exactly the kind
this architecture forbids.  So the metric is learned instead: the covariance
of the observed transform differences supplies the scaling (§25).  If this
scene's transforms vary by a lot in translation and little in rotation, that
is what "a large discrepancy" means here.

Connected components of the agreement graph are candidate motion-consistent
regions.  ``K`` is never used (§26): the number of parts is whatever the
graph turns out to have.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Any

import torch

EPS = 1.0e-12


@dataclass
class TransformComponents:
    """Agreement graph over accepted local transforms and its components."""

    label: torch.Tensor          # [A] component id, -1 for isolated / rejected
    component_sizes: torch.Tensor
    edges: torch.Tensor          # [E, 2] accepted agreement edges
    distance: torch.Tensor       # [E] Mahalanobis discrepancy of each edge
    tangent: torch.Tensor        # [A, 6] se(3) coordinates
    covariance: torch.Tensor     # [6, 6] fitted tangent covariance
    stats: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def se3_log(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """``xi = log(T)`` in ``se(3)``, returned as ``[B, 6] = (omega, v)``.

    The full logarithm, including the ``V^-1`` factor, rather than just
    stacking the rotation vector next to the raw translation.  Those differ
    whenever the rotation is not small, and it is exactly the large-rotation
    case where an approximation would make two identical motions look
    different and split one part in two.
    """
    rotation = torch.as_tensor(rotation).detach().double()
    translation = torch.as_tensor(translation).detach().double().to(rotation.device)
    b = int(rotation.shape[0])

    trace = rotation.diagonal(dim1=1, dim2=2).sum(dim=1)
    cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle = torch.acos(cos)                                     # [B]

    skew = (rotation - rotation.transpose(1, 2)) / 2.0
    axis = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=1)
    small = angle < 1.0e-6
    sin = torch.sin(angle)
    factor = torch.where(small, torch.ones_like(angle), angle / sin.clamp_min(1.0e-12))
    omega = axis * factor[:, None]

    # Near theta = pi the antisymmetric part vanishes and axis/sin(theta)
    # diverges, so the axis has to come from the SYMMETRIC part instead:
    # R + I = 2 n n^T there.  A half-turn is not an exotic case here -- two
    # parts of a manipulated object routinely differ by close to one -- and
    # without this branch those transforms come back as inf and poison every
    # covariance and consensus they touch.
    near_pi = angle > (math.pi - 1.0e-3)
    if bool(near_pi.any()):
        symmetric = (rotation + torch.eye(3, dtype=rotation.dtype, device=rotation.device)[None]) / 2.0
        column = int(torch.argmax(symmetric.diagonal(dim1=1, dim2=2).sum(0)))
        candidate = symmetric[:, :, column]
        candidate = candidate / torch.linalg.norm(candidate, dim=1, keepdim=True).clamp_min(1.0e-12)
        # n and -n both square to the same R; pick the sign that agrees with
        # whatever little antisymmetric signal survives.
        sign = torch.sign((candidate * axis).sum(dim=1))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        omega = torch.where(near_pi[:, None], candidate * (angle * sign)[:, None], omega)

    # V^-1 = I - (1/2) W + (1/theta^2)(1 - theta sin/(2(1-cos))) W^2
    w = torch.zeros((b, 3, 3), dtype=rotation.dtype, device=rotation.device)
    w[:, 0, 1], w[:, 0, 2] = -omega[:, 2], omega[:, 1]
    w[:, 1, 0], w[:, 1, 2] = omega[:, 2], -omega[:, 0]
    w[:, 2, 0], w[:, 2, 1] = -omega[:, 1], omega[:, 0]

    theta2 = (angle ** 2).clamp_min(1.0e-12)
    coefficient = torch.where(
        small,
        torch.full_like(angle, 1.0 / 12.0),
        (1.0 - angle * sin / (2.0 * (1.0 - cos).clamp_min(1.0e-12))) / theta2,
    )
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)[None].expand(b, 3, 3)
    v_inverse = identity - 0.5 * w + coefficient[:, None, None] * (w @ w)
    v = torch.einsum("bij,bj->bi", v_inverse, translation)
    return torch.cat((omega, v), dim=1).float()


@torch.no_grad()
def se3_exp(tangent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(R, t) = exp(xi)``, the inverse of :func:`se3_log`."""
    tangent = torch.as_tensor(tangent).detach().double()
    omega, v = tangent[:, :3], tangent[:, 3:]
    b = int(tangent.shape[0])
    angle = torch.linalg.norm(omega, dim=1)
    small = angle < 1.0e-6

    w = torch.zeros((b, 3, 3), dtype=tangent.dtype, device=tangent.device)
    w[:, 0, 1], w[:, 0, 2] = -omega[:, 2], omega[:, 1]
    w[:, 1, 0], w[:, 1, 2] = omega[:, 2], -omega[:, 0]
    w[:, 2, 0], w[:, 2, 1] = -omega[:, 1], omega[:, 0]

    theta = angle.clamp_min(1.0e-12)
    a = torch.where(small, 1.0 - angle ** 2 / 6.0, torch.sin(theta) / theta)
    c = torch.where(small, torch.full_like(angle, 0.5), (1.0 - torch.cos(theta)) / theta ** 2)
    d = torch.where(small, torch.full_like(angle, 1.0 / 6.0), (theta - torch.sin(theta)) / theta ** 3)
    identity = torch.eye(3, dtype=tangent.dtype, device=tangent.device)[None].expand(b, 3, 3)

    rotation = identity + a[:, None, None] * w + c[:, None, None] * (w @ w)
    v_matrix = identity + c[:, None, None] * w + d[:, None, None] * (w @ w)
    return rotation.float(), torch.einsum("bij,bj->bi", v_matrix, v).float()


@torch.no_grad()
def estimate_robust_tangent_covariance(
    tangent: torch.Tensor,
    neighbour_pairs: torch.Tensor,
    *,
    shrinkage: float = 1.0e-6,
) -> torch.Tensor:
    """Covariance of the DIFFERENCES between neighbouring transforms.

    Differences, not the transforms themselves.  The spread of the transforms
    is dominated by the fact that different parts move differently, which is
    the signal; the spread of neighbour differences is the noise, which is
    what the metric has to be calibrated against.  Using the former would
    make two genuinely different motions look like one.

    Robustified by a median-absolute-deviation rescaling so that the
    disagreeing pairs -- the very ones this is meant to detect -- do not set
    the scale they are then judged against.
    """
    tangent = torch.as_tensor(tangent).detach().float()
    pairs = torch.as_tensor(neighbour_pairs).detach().long()
    if int(pairs.shape[0]) < 8:
        return torch.eye(6, device=tangent.device)

    difference = tangent[pairs[:, 0]] - tangent[pairs[:, 1]]         # [E, 6]
    centre = difference.median(dim=0).values
    deviation = difference - centre
    mad = deviation.abs().median(dim=0).values * 1.4826
    mad = torch.where(mad > 0, mad, torch.ones_like(mad))
    # Keep the inlier bulk, then take an ordinary covariance of it.
    keep = (deviation.abs() <= 3.0 * mad).all(dim=1)
    if int(keep.sum()) < 8:
        keep = torch.ones(int(difference.shape[0]), dtype=torch.bool, device=difference.device)
    inliers = deviation[keep]
    covariance = inliers.T @ inliers / float(max(1, int(inliers.shape[0]) - 1))
    return covariance + float(shrinkage) * torch.eye(6, device=covariance.device)


@torch.no_grad()
def compute_neighbor_mahalanobis_distance(
    tangent: torch.Tensor,
    neighbour_pairs: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    """``d = (xi_i - xi_k)^T Sigma^-1 (xi_i - xi_k)`` for each pair."""
    difference = (tangent[neighbour_pairs[:, 0]] - tangent[neighbour_pairs[:, 1]]).double()
    # solve, never an explicit inverse: the tangent covariance is routinely
    # ill-conditioned when one part dominates the scene.
    solved = torch.linalg.solve(covariance.double(), difference.T).T
    return (difference * solved).sum(dim=1).float().clamp_min(0.0)


@torch.no_grad()
def build_consistent_transform_components(
    tangent: torch.Tensor,
    neighbour_pairs: torch.Tensor,
    *,
    accepted: torch.Tensor | None = None,
    alpha: float = 0.05,
    min_component: int = 3,
) -> TransformComponents:
    """Connected components of the SE(3) agreement graph.

    An edge survives when the discrepancy between its two transforms is
    consistent with the fitted transform-noise model -- a chi-square test on
    six degrees of freedom at the globally fixed significance level, so the
    cut is a statistical convention rather than a radius someone picked.
    """
    tangent = torch.as_tensor(tangent).detach().float()
    pairs = torch.as_tensor(neighbour_pairs).detach().long()
    anchors = int(tangent.shape[0])
    if accepted is not None:
        keep = accepted.to(device=pairs.device).bool()
        pairs = pairs[keep[pairs[:, 0]] & keep[pairs[:, 1]]]

    if int(pairs.shape[0]) == 0:
        return TransformComponents(
            label=torch.full((anchors,), -1, dtype=torch.long),
            component_sizes=torch.zeros(0, dtype=torch.long),
            edges=pairs, distance=torch.zeros(0),
            tangent=tangent, covariance=torch.eye(6),
            stats={"components": 0, "edge_fraction": 0.0, "chi2_limit": 0.0, "assigned_fraction": 0.0},
        )

    covariance = estimate_robust_tangent_covariance(tangent, pairs)
    distance = compute_neighbor_mahalanobis_distance(tangent, pairs, covariance)
    # chi2(6) upper tail at alpha. Evaluated via the regularised gamma
    # function so the convention is exact rather than a looked-up constant.
    limit = _chi2_quantile(6, 1.0 - float(alpha))
    consistent = distance <= limit
    edges = pairs[consistent]

    label = _connected_components(edges, anchors)
    counts = torch.bincount(label[label >= 0], minlength=max(1, int(label.max()) + 1)) if bool((label >= 0).any()) else torch.zeros(0, dtype=torch.long)
    # A component of one or two anchors is not agreement, it is coincidence.
    for component in range(int(counts.numel())):
        if int(counts[component]) < int(min_component):
            label[label == component] = -1
    return TransformComponents(
        label=label,
        component_sizes=counts,
        edges=edges,
        distance=distance[consistent],
        tangent=tangent,
        covariance=covariance,
        stats={
            "components": int((counts >= int(min_component)).sum()),
            "edge_fraction": float(consistent.float().mean()),
            "chi2_limit": float(limit),
            "assigned_fraction": float((label >= 0).float().mean()),
        },
    )


def _chi2_quantile(dof: int, probability: float, *, steps: int = 4096, upper: float = 200.0) -> float:
    """Quantile of chi-square by monotone bisection on its CDF.

    Deterministic and dependency-free; ``steps``/``upper`` are numerical
    controls whose effect is below the tolerance of the test they serve.
    """
    import math

    def cdf(x: float) -> float:
        # Regularised lower incomplete gamma P(k/2, x/2) by series expansion.
        k = dof / 2.0
        z = x / 2.0
        if z <= 0.0:
            return 0.0
        term = 1.0 / k
        total = term
        for n in range(1, steps):
            term *= z / (k + n)
            total += term
            if term < 1.0e-14 * total:
                break
        return total * math.exp(-z + k * math.log(z) - math.lgamma(k))

    low, high = 0.0, float(upper)
    for _ in range(200):
        mid = 0.5 * (low + high)
        if cdf(mid) < probability:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _connected_components(edges: torch.Tensor, count: int) -> torch.Tensor:
    """Union-find over the agreement edges; isolated anchors get ``-1``."""
    parent = list(range(count))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges.tolist():
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv

    seen: dict[int, int] = {}
    label = torch.full((count,), -1, dtype=torch.long)
    touched = set(int(x) for x in edges.reshape(-1).tolist())
    for node in sorted(touched):
        root = find(node)
        if root not in seen:
            seen[root] = len(seen)
        label[node] = seen[root]
    return label


@torch.no_grad()
def robust_se3_consensus(tangent: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Robust Fréchet mean in the tangent space, returned as ``[6]``.

    Iteratively reweighted about the running mean, which is the tangent-space
    analogue of a geometric median: a single badly-fitted patch inside an
    otherwise coherent component must not drag the part's motion with it.
    """
    tangent = torch.as_tensor(tangent).detach().float()
    if int(tangent.shape[0]) == 0:
        return torch.zeros(6)
    weight = torch.ones(int(tangent.shape[0]), device=tangent.device) if weights is None else weights.float()
    centre = (tangent * weight[:, None]).sum(0) / weight.sum().clamp_min(EPS)
    for _ in range(20):
        residual = torch.linalg.norm(tangent - centre, dim=1)
        scale = residual.median().clamp_min(1.0e-6)
        robust = weight / (1.0 + (residual / scale) ** 2)
        updated = (tangent * robust[:, None]).sum(0) / robust.sum().clamp_min(EPS)
        if float(torch.linalg.norm(updated - centre)) < 1.0e-7:
            centre = updated
            break
        centre = updated
    return centre
