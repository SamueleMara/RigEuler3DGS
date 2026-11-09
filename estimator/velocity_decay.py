"""Mean-reverting velocity for the constant-velocity process models (Phases 1, 2, 3).

The constant-velocity model carries a velocity unchanged until an image contradicts
it. Along a direction the RGB-D cannot see -- tangential sliding on a uniform
surface, a Gaussian hidden behind an occluder, a rigid part whose support left the
view -- nothing ever does, so a velocity picked up from one noisy update is
integrated for the rest of the sequence. An Ornstein-Uhlenbeck velocity

    dv = -v / tau dt + sigma dW,        dx = v dt

decays to zero without evidence instead. Over a step dt, with phi = exp(-dt/tau):

    v+ = phi v,      x+ = x + h v,      h = tau (1 - phi)
    Q_vv = s2 tau / 2 (1 - phi^2)
    Q_xv = s2 tau^2 / 2 (1 - phi)^2
    Q_xx = s2 tau^3 / 2 (2 dt/tau - 3 + 4 phi - phi^2)

(the exact discretization; s2 = sigma^2 is the same white-acceleration density the
constant-velocity model uses). tau <= 0 means tau = infinity: phi = 1, h = dt and
the classical triple dt^3/3, dt^2/2, dt -- bitwise the constant-velocity model.

For a rotation under a world-frame twist the same h replaces dt in Exp(h omega):
the decaying omega keeps its direction, so the increments commute and the
integral is exact.
"""
from __future__ import annotations

import math


def terms(dt: float, tau: float = 0.0) -> tuple[float, float, float, float, float]:
    """(phi, h, c_xx, c_xv, c_vv): Q_ab = s2 * c_ab per unit acceleration density."""
    dt = float(dt)
    if dt < 0:
        raise ValueError(f"dt must be non-negative, got {dt}")
    if not enabled(tau):
        return 1.0, dt, dt ** 3 / 3.0, dt ** 2 / 2.0, dt
    tau = float(tau)
    x = dt / tau
    em1, em2 = math.expm1(-x), math.expm1(-2.0 * x)      # phi - 1, phi^2 - 1, without cancellation
    phi = 1.0 + em1
    h = -tau * em1
    # 2x - 3 + 4 phi - phi^2 = 2x + 4 (phi - 1) - (phi^2 - 1) ~ 2x^3/3: series below 1e-2
    poly = (2 * x ** 3 / 3 - x ** 4 / 2 + 7 * x ** 5 / 30) if x < 1e-2 else (2 * x + 4 * em1 - em2)
    return phi, h, tau ** 3 / 2.0 * poly, tau ** 2 / 2.0 * em1 * em1, -tau / 2.0 * em2


def enabled(tau) -> bool:
    return tau is not None and float(tau) > 0 and not math.isinf(float(tau))


def terms_tensor(dt, tau: float):
    """`terms` for a tensor of step lengths (per-Gaussian dt); tau must be enabled."""
    import torch
    if not enabled(tau):
        raise ValueError("terms_tensor is the decaying branch only; keep the constant-velocity code for tau off")
    x = dt / float(tau)
    em1, em2 = torch.expm1(-x), torch.expm1(-2.0 * x)
    series = 2 * x ** 3 / 3 - x ** 4 / 2 + 7 * x ** 5 / 30
    poly = torch.where(x < 1e-2, series, 2 * x + 4 * em1 - em2)
    tau = float(tau)
    return 1.0 + em1, -tau * em1, tau ** 3 / 2.0 * poly, tau ** 2 / 2.0 * em1 * em1, -tau / 2.0 * em2


def describe(tau: float) -> dict:
    """What a stage records about its velocity model."""
    on = enabled(tau)
    return {"model": "ornstein_uhlenbeck" if on else "constant_velocity",
            "tau_s": float(tau) if on else None,
            "note": ("velocity decays as exp(-dt/tau) without evidence; drift along an unobserved "
                     "direction is bounded by v tau" if on else
                     "velocity carried unchanged; drift along an unobserved direction is unbounded")}
