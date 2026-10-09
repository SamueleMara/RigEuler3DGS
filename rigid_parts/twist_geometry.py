"""Twist-level gradients and screw-geometry diagnostics.

CODEX_DEBUG_SOFTP_GRADIENTS_STORAGE_DIAGNOSIS, Parts IV-V (§21-§30).

Why this exists: storage's failure is not visible in the control gradients the
geometry optimizer actually steps. It is visible physically -- too much rotation
for a prismatic GT joint. So the gradients are also measured at the DERIVED
rigid twist (omega, u), and the recovered twist is decomposed into the screw
quantities that would distinguish a genuine rotation from a far-away rotation
axis that merely approximates a translation over a small support (§30).

Convention, fixed by this repository and verified in
tests/test_debug_handoff_and_twist.py::test_45_screw_axis_convention:

    v(x) = u + omega x x

so a rotation about a point c has u = c x omega, and the minimum-norm point on
the axis is

    c_perp = (omega x u) / |omega|^2.

§28 explicitly forbids hard-coding an untested sign; that formula is pinned by
the test, which reconstructs c from a constructed rotation and also checks that
v vanishes on the returned point.
"""

from __future__ import annotations

from typing import Any

import torch

EPS = 1.0e-12


@torch.no_grad()
def _norm(t: torch.Tensor | None) -> float:
    return 0.0 if t is None else float(t.detach().norm())


def twist_gradient_diagnostics(
    losses: dict[str, torch.Tensor],
    omega: torch.Tensor,
    u: torch.Tensor,
) -> dict[str, Any]:
    """§22/§23/§24: per-loss dL/domega and dL/du, plus pairwise cosines.

    Uses ``torch.autograd.grad`` so the optimizer's ``.grad`` is never touched
    (§22). The cosines are the point of §24: two losses can both have large
    gradient norms and still cancel, which a norm alone cannot show.
    """
    grad_omega: dict[str, float] = {}
    grad_u: dict[str, float] = {}
    vec_omega: dict[str, torch.Tensor] = {}
    vec_u: dict[str, torch.Tensor] = {}

    for name, value in losses.items():
        if not torch.is_tensor(value) or not value.requires_grad:
            grad_omega[name] = 0.0
            grad_u[name] = 0.0
            continue
        try:
            g_w, g_u = torch.autograd.grad(value, (omega, u), retain_graph=True,
                                           allow_unused=True)
        except RuntimeError:
            grad_omega[name] = float("nan")
            grad_u[name] = float("nan")
            continue
        grad_omega[name] = _norm(g_w)
        grad_u[name] = _norm(g_u)
        if g_w is not None:
            vec_omega[name] = g_w.detach().reshape(-1)
        if g_u is not None:
            vec_u[name] = g_u.detach().reshape(-1)

    def cosines(vectors: dict[str, torch.Tensor]) -> dict[str, float]:
        out: dict[str, float] = {}
        names = sorted(vectors)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                va, vb = vectors[a], vectors[b]
                denom = (va.norm() * vb.norm()).clamp_min(EPS)
                out[f"{a}|{b}"] = float((va * vb).sum() / denom)
        return out

    return {"grad_omega_norm": grad_omega, "grad_u_norm": grad_u,
            "cosine_omega": cosines(vec_omega), "cosine_u": cosines(vec_u)}


@torch.no_grad()
def screw_decomposition(omega: torch.Tensor, u: torch.Tensor, *, eps: float = EPS) -> dict[str, Any]:
    """§26/§27/§28: |omega|, |u|, their ratio, pitch, axis direction and point."""
    w = torch.as_tensor(omega).detach().double().reshape(3)
    v = torch.as_tensor(u).detach().double().reshape(3)
    w2 = float(w.square().sum())
    w_norm, u_norm = float(w.norm()), float(v.norm())

    out: dict[str, Any] = {
        "omega_norm": w_norm,
        "u_norm": u_norm,
        "omega_over_u": w_norm / (u_norm + eps),
        "pitch": (float((w * v).sum() / (w2 + eps)) if w2 > eps else float("nan")),
        "axis_direction": None,
        "axis_point": None,
        "is_pure_translation": bool(w_norm <= 1e-9),
    }
    if w_norm > 1e-9:
        out["axis_direction"] = (w / w_norm).tolist()
        # v(x) = u + omega x x vanishes (up to the along-axis slide) at
        # c_perp = (omega x u) / |omega|^2. Sign verified by test §45.
        out["axis_point"] = (torch.linalg.cross(w, v) / (w2 + eps)).tolist()
    return out


@torch.no_grad()
def axis_distance_statistics(
    points: torch.Tensor,
    weights: torch.Tensor,
    decomposition: dict[str, Any],
    *,
    pseudo_translation_ratio: float = 5.0,
    eps: float = EPS,
) -> dict[str, Any]:
    """§29: how far the recovered rotation axis is from the support that fitted it.

    Normalised by the moving-part bbox diagonal, because "far" only means
    anything relative to the extent of the points that produced it. A very
    distant axis can approximate a translation locally, which is the failure
    mode §46 asks us to detect.
    """
    x = torch.as_tensor(points).detach().double()
    out: dict[str, Any] = {"min": float("nan"), "median": float("nan"),
                           "mean": float("nan"), "p90": float("nan"),
                           "median_over_bbox": float("nan"),
                           "bbox_diagonal": float("nan"),
                           "pseudo_translation": False}
    if int(x.shape[0]) == 0 or decomposition.get("axis_point") is None:
        return out
    direction = torch.tensor(decomposition["axis_direction"], dtype=torch.float64, device=x.device)
    point = torch.tensor(decomposition["axis_point"], dtype=torch.float64, device=x.device)
    delta = x - point
    perpendicular = delta - (delta @ direction)[:, None] * direction[None]
    d = perpendicular.norm(dim=1)
    bbox = float((x.max(dim=0).values - x.min(dim=0).values).norm().clamp_min(eps))
    out.update({
        "min": float(d.min()), "median": float(d.median()), "mean": float(d.mean()),
        "p90": float(torch.quantile(d, 0.90)), "bbox_diagonal": bbox,
        "median_over_bbox": float(d.median()) / bbox,
    })
    out["pseudo_translation"] = bool(out["median_over_bbox"] > float(pseudo_translation_ratio))
    return out


@torch.no_grad()
def rotational_field_variation(
    points: torch.Tensor,
    weights: torch.Tensor,
    omega: torch.Tensor,
    u: torch.Tensor,
    *,
    eps: float = EPS,
) -> dict[str, Any]:
    """§30: how much the rotational part of the field actually VARIES over the
    support, relative to the translation it rides on.

        r_i = omega x (x_i - x_bar),   sigma_rot = sqrt(sum w |r|^2 / sum w)
        R = sigma_rot / (|u| + eps)

    A tiny R with a non-zero |omega| means the recovered screw behaves like a
    pseudo-translation across the observed points: the rotation is real in the
    parameters but indistinguishable from a translation in the data.
    """
    x = torch.as_tensor(points).detach().double()
    w = torch.as_tensor(weights).detach().double().reshape(-1).clamp_min(0.0)
    om = torch.as_tensor(omega).detach().double().reshape(3)
    uu = torch.as_tensor(u).detach().double().reshape(3)
    if int(x.shape[0]) == 0 or float(w.sum()) <= 0:
        return {"sigma_rot": float("nan"), "rot_over_trans": float("nan"),
                "u_norm": float(uu.norm())}
    centroid = (w[:, None] * x).sum(dim=0) / (w.sum() + eps)
    r = torch.linalg.cross(om[None].expand_as(x), x - centroid[None])
    sigma = float(((w * r.square().sum(dim=1)).sum() / (w.sum() + eps)).clamp_min(0.0).sqrt())
    return {"sigma_rot": sigma,
            "rot_over_trans": sigma / (float(uu.norm()) + eps),
            "u_norm": float(uu.norm()),
            "centroid": centroid.tolist()}
