"""RGB-D observation model, sensor precision and robust IRLS weights.

Implements specification sections 6 and 13.1 plus the normalized-innovation
diagnostics demanded by section 34.

Section 6 fixes the measurement covariance as a *physical sensor* quantity

    R_p^-1 = diag(sigma_R^-2, sigma_G^-2, sigma_B^-2, sigma_D^-2),

NOT a bag of hand-tuned loss weights.  Two consequences drive this module:

  * ``sigma_D`` may grow with range (active stereo / ToF noise scales with z),
    so the depth precision is a per-pixel field, not a constant;
  * depth holes / invalid returns carry *zero information*.  They are encoded
    as exactly-zero precision, never as a fake depth value.  Anything else
    injects a fictitious measurement into the normal equations of section 13
    and biases the state.

Section 13.1 adds the Student-t IRLS reweighting

    w_j = (nu + 1) / (nu + (r_j / sigma_j)^2),      R_eff^-1 = W_r R^-1,

which multiplies - and never replaces - the sensor precision.  The renderer
Jacobian is untouched: sensor geometry and robust statistics stay separate.

Autograd policy (spec 0.5 / 36): nothing here differentiates anything.  The
residual is detached before the precision is built, which also makes the IRLS
weights constants of the current linearization - exactly the IRLS fixed point.

Channel layout
--------------
Residual tensors carry a channel axis (``channel_dim``, default 0 to match the
repository's CHW render convention).  Four channels are read as (R, G, B, D),
three as RGB-only and one as depth-only, which is what the
``--iekf_use_rgb`` / ``--iekf_use_depth`` flags of section 33 produce.  An
explicit ``layout`` overrides the inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import torch

RGB = "rgb"
DEPTH = "depth"
CHANNEL_KINDS = (RGB, DEPTH)

DEPTH_SIGMA_MODELS = ("constant", "linear", "quadratic")

# Floor on any standard deviation.  A zero sigma would make the precision
# infinite and turn one pixel into a hard constraint on the whole state.
SIGMA_FLOOR = 1.0e-9


# ---------------------------------------------------------------------------
# Section 6: depth noise model sigma_D(z)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DepthSigmaModel:
    """Range-dependent depth standard deviation ``sigma_D(z)``.

    ``constant``   sigma = a                  (ideal / pre-rectified sensor)
    ``linear``     sigma = a + b*z            (rolling-shutter-ish, lidar)
    ``quadratic``  sigma = a + b*z^2          (triangulation: disparity noise
                                               maps to depth noise as z^2/(f*B))

    ``a`` keeps the model finite at z = 0 so that a near-field pixel never gets
    an unbounded precision; it is the quantization/electronic noise floor.
    """

    kind: str = "constant"
    a: float = 0.01
    b: float = 0.0
    sigma_min: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.kind not in DEPTH_SIGMA_MODELS:
            raise ValueError(
                f"unknown depth sigma model {self.kind!r}; expected one of {DEPTH_SIGMA_MODELS}")
        if not (math.isfinite(self.a) and math.isfinite(self.b)):
            raise ValueError("depth sigma coefficients must be finite")
        if self.a < 0.0 or self.b < 0.0:
            raise ValueError("depth sigma coefficients must be non-negative (sigma must grow with range)")
        if not (self.sigma_min > 0.0):
            raise ValueError("sigma_min must be strictly positive")

    @property
    def range_dependent(self) -> bool:
        """True when ``sigma`` actually needs a range tensor to be evaluated."""
        return self.kind != "constant" and self.b != 0.0

    def sigma(self, depth: torch.Tensor | None = None) -> torch.Tensor | float:
        """Evaluate sigma_D.  Returns a tensor iff ``depth`` is a tensor.

        A range-dependent model with no range is an error, not a fallback: the
        silent substitution of a constant sigma is precisely the "silently
        simplify" failure mode forbidden by spec rule 0.14.
        """
        if depth is None:
            if self.range_dependent:
                raise ValueError(
                    f"depth sigma model {self.kind!r} is range dependent; pass depth_range")
            return max(float(self.a), float(self.sigma_min))
        if not torch.is_tensor(depth):
            raise TypeError("depth_range must be a torch.Tensor")
        # Range magnitude: some conventions store camera-forward depth with a
        # negative sign, and sensor noise depends on |z|, not on its sign.
        z = depth.abs()
        if self.kind == "constant":
            s = torch.full_like(z, float(self.a))
        elif self.kind == "linear":
            s = float(self.a) + float(self.b) * z
        else:  # quadratic
            s = float(self.a) + float(self.b) * z * z
        return s.clamp_min(float(self.sigma_min))


def make_depth_sigma_model(
    spec: DepthSigmaModel | str | Mapping[str, Any] | None,
) -> DepthSigmaModel:
    """Normalize ``--iekf_depth_sigma_model`` style input to a model object.

    Accepts the object itself, a bare name, ``"quadratic:0.004:0.0015"`` or a
    mapping of the dataclass fields.
    """
    if spec is None:
        return DepthSigmaModel()
    if isinstance(spec, DepthSigmaModel):
        return spec
    if isinstance(spec, Mapping):
        return DepthSigmaModel(**spec)
    if isinstance(spec, str):
        parts = [p for p in spec.strip().split(":") if p != ""]
        if not parts:
            raise ValueError("empty depth sigma model specification")
        kind = parts[0].lower()
        kwargs: Dict[str, float] = {}
        if len(parts) > 1:
            kwargs["a"] = float(parts[1])
        if len(parts) > 2:
            kwargs["b"] = float(parts[2])
        if len(parts) > 3:
            kwargs["sigma_min"] = float(parts[3])
        return DepthSigmaModel(kind=kind, **kwargs)
    raise TypeError(f"cannot interpret depth sigma model specification {spec!r}")


# ---------------------------------------------------------------------------
# Channel bookkeeping
# ---------------------------------------------------------------------------
def channel_kinds(n_channels: int, layout: Sequence[str] | None = None) -> tuple[str, ...]:
    """Map a channel count to (R, G, B, D) semantics; ``layout`` overrides it."""
    if layout is not None:
        kinds = tuple(str(k).lower() for k in layout)
        if len(kinds) != int(n_channels):
            raise ValueError(f"layout has {len(kinds)} entries but residual has {n_channels} channels")
        bad = [k for k in kinds if k not in CHANNEL_KINDS]
        if bad:
            raise ValueError(f"unknown channel kinds {bad}; expected {CHANNEL_KINDS}")
        return kinds
    if n_channels == 4:
        return (RGB, RGB, RGB, DEPTH)
    if n_channels == 3:
        return (RGB, RGB, RGB)
    if n_channels == 1:
        return (DEPTH,)
    raise ValueError(
        f"cannot infer RGB-D layout for {n_channels} channels; pass layout=(...) explicitly")


def _resolve_dim(dim: int, ndim: int) -> int:
    if ndim == 0:
        raise ValueError("residual must have at least one dimension (the channel axis)")
    d = int(dim)
    if d < 0:
        d += ndim
    if not (0 <= d < ndim):
        raise ValueError(f"channel_dim {dim} is out of range for a {ndim}-D residual")
    return d


def _slice_shape(shape: torch.Size, cd: int) -> torch.Size:
    return torch.Size(tuple(shape[:cd]) + tuple(shape[cd + 1:]))


def _align_to_residual(
    value: torch.Tensor,
    residual: torch.Tensor,
    cd: int,
    name: str,
) -> torch.Tensor:
    """Broadcast a per-pixel field to the full residual shape.

    A field given without a channel axis is understood as "same for every
    channel"; explicitly inserting the axis here (instead of relying on trailing
    broadcast) keeps the semantics identical for channel-first images and for
    channel-last contributor lists.
    """
    v = value
    if v.shape == _slice_shape(residual.shape, cd) and v.dim() == residual.dim() - 1:
        v = v.unsqueeze(cd)
    try:
        return torch.broadcast_to(v, residual.shape)
    except RuntimeError as exc:  # pragma: no cover - message path
        raise ValueError(
            f"{name} of shape {tuple(value.shape)} is not broadcastable to the "
            f"residual shape {tuple(residual.shape)} (channel_dim={cd})") from exc


def _channel_selector(
    residual: torch.Tensor, cd: int, kinds: Sequence[str], wanted: str
) -> torch.Tensor:
    """Boolean mask, residual-shaped, selecting every channel of kind ``wanted``."""
    flags = torch.tensor(
        [k == wanted for k in kinds], dtype=torch.bool, device=residual.device)
    view = [1] * residual.dim()
    view[cd] = len(kinds)
    return flags.view(view).expand(residual.shape)


# ---------------------------------------------------------------------------
# Section 6: sensor sigma and precision
# ---------------------------------------------------------------------------
def channel_sigma(
    residual: torch.Tensor,
    sigma_rgb: float | Sequence[float] | torch.Tensor = 1.0,
    depth_sigma_model: DepthSigmaModel | str | Mapping[str, Any] | None = "constant",
    *,
    depth_range: torch.Tensor | None = None,
    channel_dim: int = 0,
    layout: Sequence[str] | None = None,
) -> torch.Tensor:
    """Per-pixel-channel sensor standard deviation, shaped like ``residual``.

    ``sigma_rgb`` is a scalar or a per-channel triple (sigma_R, sigma_G,
    sigma_B); the depth channel uses ``sigma_D(z)`` from the depth model.
    """
    cd = _resolve_dim(channel_dim, residual.dim())
    kinds = channel_kinds(residual.shape[cd], layout)
    dtype, device = residual.dtype, residual.device
    slice_shape = _slice_shape(residual.shape, cd)

    n_rgb = sum(1 for k in kinds if k == RGB)
    rgb = torch.as_tensor(sigma_rgb, dtype=dtype, device=device).reshape(-1)
    if rgb.numel() == 1 and n_rgb > 0:
        rgb = rgb.expand(n_rgb)
    if rgb.numel() != n_rgb:
        raise ValueError(f"sigma_rgb has {rgb.numel()} entries but the layout has {n_rgb} RGB channels")
    if n_rgb and not bool((rgb > 0).all()):
        raise ValueError("sigma_rgb must be strictly positive")

    model = make_depth_sigma_model(depth_sigma_model)
    z = None
    if depth_range is not None:
        # Broadcast first, then drop the channel axis: sigma_D(z) is one field
        # per pixel, and the caller may hand it over with or without that axis.
        z = _align_to_residual(
            depth_range.to(dtype=dtype, device=device), residual, cd, "depth_range").select(cd, 0)
    depth_sigma = model.sigma(z)

    channels = []
    i_rgb = 0
    for kind in kinds:
        if kind == RGB:
            s = rgb[i_rgb].clamp_min(SIGMA_FLOOR).expand(slice_shape)
            i_rgb += 1
        else:
            if torch.is_tensor(depth_sigma):
                s = torch.broadcast_to(depth_sigma.to(dtype=dtype, device=device), slice_shape)
            else:
                s = torch.full(slice_shape, float(depth_sigma), dtype=dtype, device=device)
            s = s.clamp_min(SIGMA_FLOOR)
        channels.append(s)
    return torch.stack(channels, dim=cd)


def student_t_weight(whitened: torch.Tensor, nu: float) -> torch.Tensor:
    """Section 13.1 IRLS weight ``w = (nu + 1) / (nu + whitened^2)``.

    ``nu = inf`` is the Gaussian limit and returns exactly ones; evaluating the
    formula there would be inf/inf = NaN, so it is short-circuited rather than
    approximated by a large finite nu.
    """
    nu_f = float(nu)
    if math.isnan(nu_f) or nu_f <= 0.0:
        raise ValueError(f"student-t degrees of freedom must be positive, got {nu}")
    if math.isinf(nu_f):
        return torch.ones_like(whitened)
    return (nu_f + 1.0) / (nu_f + whitened.square())


def student_t_nll(whitened: torch.Tensor, nu: float) -> torch.Tensor:
    """The per-sample robust objective whose derivative IS the IRLS score.

        rho(z) = (nu + 1)/2 * log1p(z^2 / nu)      (nu finite)
        rho(z) = z^2 / 2                            (nu = inf)

    d rho / d r = w(z) r / sigma^2 with w = student_t_weight(z, nu), so
    Gauss-Newton with IRLS precision w/sigma^2 is the local model of THIS
    function. Summing 0.5 * w(z) z^2 with weights re-evaluated at every trial
    state is not a fixed objective; this is.
    """
    nu_f = float(nu)
    if math.isnan(nu_f) or nu_f <= 0.0:
        raise ValueError(f"student-t degrees of freedom must be positive, got {nu}")
    z2 = whitened.square()
    if math.isinf(nu_f):
        return 0.5 * z2
    return 0.5 * (nu_f + 1.0) * torch.log1p(z2 / nu_f)


def robust_term(residual: torch.Tensor, sigma: torch.Tensor, valid: torch.Tensor,
                robust: bool, nu: float) -> tuple[torch.Tensor, float]:
    """(GN precision, objective) for one channel with a FIXED per-sample sigma.

    precision = valid * w(z) / sigma^2 (w = 1 when not robust) -- the IRLS local
    model; objective = sum over valid samples of student_t_nll(z, nu) when
    robust, 0.5 z^2 otherwise. Invalid samples are excluded by indexing before
    any arithmetic, so a NaN / inf / garbage value stored there can never reach
    the objective, the precision or the gradient. float64 accumulation.
    """
    v = valid.bool() & torch.isfinite(residual) & torch.isfinite(sigma) & (sigma > 0)
    zero = torch.zeros((), dtype=residual.dtype, device=residual.device)
    one = torch.ones((), dtype=residual.dtype, device=residual.device)
    r = torch.where(v, residual, zero)
    s = torch.where(v, sigma.to(residual.dtype), one)
    z = r / s
    w = student_t_weight(z, nu) if robust else torch.ones_like(z)
    prec = torch.where(v, w / (s * s), zero)
    rho = student_t_nll(z, nu if robust else float("inf"))
    energy = float(torch.where(v, rho, zero).sum(dtype=torch.float64))
    return prec, energy


def depth_validity(
    depth_gt: torch.Tensor,
    *,
    min_depth: float = 1.0e-6,
    max_depth: float | None = None,
    hole_value: float | None = 0.0,
) -> torch.Tensor:
    """Boolean validity for a raw depth map (section 6 "holes/invalid").

    Sensors signal "no return" with 0, NaN or a saturation value; all three must
    end up as zero precision, so they are detected here instead of being
    silently replaced by a plausible number.
    """
    valid = torch.isfinite(depth_gt)
    valid &= depth_gt.abs() >= float(min_depth)
    if max_depth is not None:
        valid &= depth_gt.abs() <= float(max_depth)
    if hole_value is not None:
        valid &= depth_gt != float(hole_value)
    return valid


def build_precision(
    residual: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    sigma_rgb: float | Sequence[float] | torch.Tensor = 1.0,
    depth_sigma_model: DepthSigmaModel | str | Mapping[str, Any] | None = "constant",
    robust: bool = False,
    nu: float = 5.0,
    *,
    depth_range: torch.Tensor | None = None,
    channel_dim: int = 0,
    layout: Sequence[str] | None = None,
    use_rgb: bool = True,
    use_depth: bool = True,
    invalidate_nonfinite: bool = True,
    return_parts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Effective measurement precision ``R_eff^-1 = W_r R^-1``, residual-shaped.

    Parameters mirror spec sections 6 and 13.1: ``sigma_rgb`` /
    ``depth_sigma_model`` are the sensor model, ``valid_mask`` marks usable
    measurements, and ``robust`` / ``nu`` switch on the Student-t IRLS
    reweighting.  ``depth_range`` supplies z for a range-dependent sigma_D.

    Invalid entries get exactly 0.0 - bit-for-bit, not a small number - so they
    contribute exactly nothing to the energy, the gradient ``H^T R^-1 r`` and
    the Fisher blocks.

    With ``return_parts`` the sigma, validity and robust-weight fields are
    returned alongside for section 34 logging.
    """
    if not torch.is_tensor(residual):
        raise TypeError("residual must be a torch.Tensor")
    # Detached: the estimator is autograd-free (spec 36) and IRLS weights are
    # constants of the current linearization by definition.
    r = residual.detach()
    cd = _resolve_dim(channel_dim, r.dim())
    kinds = channel_kinds(r.shape[cd], layout)

    sigma = channel_sigma(
        r,
        sigma_rgb=sigma_rgb,
        depth_sigma_model=depth_sigma_model,
        depth_range=depth_range,
        channel_dim=cd,
        layout=kinds,
    )

    if valid_mask is None:
        mask = torch.ones_like(r, dtype=torch.bool)
    else:
        if not torch.is_tensor(valid_mask):
            raise TypeError("valid_mask must be a torch.Tensor or None")
        mask = _align_to_residual(valid_mask.to(device=r.device).bool(), r, cd, "valid_mask")
    if invalidate_nonfinite:
        # A NaN/Inf residual is an unusable measurement, not an infinitely
        # informative one; it is dropped here so it can never reach a dot
        # product further down the pipeline.
        mask = mask & torch.isfinite(r)
    if not use_rgb and RGB in kinds:
        mask = mask & ~_channel_selector(r, cd, kinds, RGB)
    if not use_depth and DEPTH in kinds:
        mask = mask & ~_channel_selector(r, cd, kinds, DEPTH)

    zero = torch.zeros((), dtype=r.dtype, device=r.device)
    precision = torch.where(mask, sigma.square().reciprocal(), zero)

    if robust:
        # Whiten with the residual already zeroed on invalid entries: a hole may
        # literally hold NaN, and NaN would survive the multiplication by a zero
        # precision and poison the whole update.
        r_safe = torch.where(mask, r, zero)
        weights = student_t_weight(r_safe / sigma, nu)
        precision = precision * weights
    else:
        weights = torch.ones_like(precision)

    if return_parts:
        return precision, {"sigma": sigma, "valid": mask, "weight": weights}
    return precision


# ---------------------------------------------------------------------------
# Residual and energies
# ---------------------------------------------------------------------------
def rgbd_residual(
    y_gt: torch.Tensor,
    y_hat: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    channel_dim: int = 0,
    zero_invalid: bool = True,
) -> torch.Tensor:
    """Innovation ``r = y_GT - h(x_hat)`` of section 13, RGB-D stacked.

    Invalid entries are zeroed by default so that a hole encoded as NaN in the
    ground-truth map cannot propagate even if a caller forgets to pass the same
    mask to :func:`build_precision`.
    """
    if y_gt.shape != y_hat.shape:
        raise ValueError(
            f"ground truth {tuple(y_gt.shape)} and prediction {tuple(y_hat.shape)} must have equal shape")
    r = y_gt.to(dtype=y_hat.dtype, device=y_hat.device) - y_hat
    if valid_mask is not None and zero_invalid:
        cd = _resolve_dim(channel_dim, r.dim())
        mask = _align_to_residual(valid_mask.to(device=r.device).bool(), r, cd, "valid_mask")
        r = torch.where(mask, r, torch.zeros((), dtype=r.dtype, device=r.device))
    return r


def measurement_energy(
    residual: torch.Tensor,
    precision: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """``0.5 * sum(precision * residual^2)`` - the measurement half of J (13).

    Accumulated in float64 by default because the line search of section 14
    accepts a step on a *relative* energy decrease that can sit in the seventh
    significant digit of a sum over millions of pixels.
    """
    if residual.shape != precision.shape:
        raise ValueError(
            f"residual {tuple(residual.shape)} and precision {tuple(precision.shape)} must have equal shape")
    r = residual.detach()
    return 0.5 * torch.sum(precision * r.square(), dtype=dtype)


def normalized_innovation_stats(
    residual: torch.Tensor,
    precision: torch.Tensor,
    *,
    channel_dim: int | None = None,
    layout: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Whitened-innovation diagnostics required by section 34.

    The whitened innovation is ``r_j * sqrt(R^-1_jj)``; a correctly tuned sensor
    model makes its chi-square statistic approach the number of *active*
    measurements, so everything is reduced over entries with nonzero precision
    only - averaging in the zeros of masked-out pixels would make any model look
    good.
    """
    if residual.shape != precision.shape:
        raise ValueError(
            f"residual {tuple(residual.shape)} and precision {tuple(precision.shape)} must have equal shape")
    r = residual.detach()
    stats = _innovation_reduce(r, precision)
    stats["total"] = int(r.numel())
    stats["active_fraction"] = float(stats["active"]) / float(max(r.numel(), 1))

    if channel_dim is not None:
        cd = _resolve_dim(channel_dim, r.dim())
        kinds = channel_kinds(r.shape[cd], layout)
        per_kind: Dict[str, Dict[str, Any]] = {}
        for kind in sorted(set(kinds)):
            sel = _channel_selector(r, cd, kinds, kind)
            per_kind[kind] = _innovation_reduce(r[sel], precision[sel])
        stats["channels"] = per_kind
    return stats


def _innovation_reduce(residual: torch.Tensor, precision: torch.Tensor) -> Dict[str, Any]:
    active = precision > 0
    n_active = int(active.sum().item())
    chi2 = float(torch.sum(precision * residual.square(), dtype=torch.float64))
    out: Dict[str, Any] = {
        "active": n_active,
        "chi2": chi2,
        "chi2_per_dof": chi2 / n_active if n_active else 0.0,
        "energy": 0.5 * chi2,
    }
    if n_active == 0:
        out.update({"mean": 0.0, "median": 0.0, "max": 0.0, "rms": 0.0, "nll": 0.0})
        return out
    w = (residual[active].double() * precision[active].double().sqrt()).abs()
    out["mean"] = float(w.mean())
    out["median"] = float(w.median())
    out["max"] = float(w.max())
    out["rms"] = math.sqrt(chi2 / n_active)
    # Gaussian negative log likelihood, constants included so that models with
    # different sigma_D(z) are actually comparable (section 34).
    log_det = float(torch.sum(torch.log(precision[active].double())))
    out["nll"] = 0.5 * chi2 - 0.5 * log_det + 0.5 * n_active * math.log(2.0 * math.pi)
    return out


__all__ = [
    "RGB",
    "DEPTH",
    "DEPTH_SIGMA_MODELS",
    "DepthSigmaModel",
    "make_depth_sigma_model",
    "channel_kinds",
    "channel_sigma",
    "student_t_weight",
    "depth_validity",
    "build_precision",
    "rgbd_residual",
    "measurement_energy",
    "normalized_innovation_stats",
]
