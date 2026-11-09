"""Small differentiable CPU/PyTorch oracle for a depth-only first-return ray.

REFERENCE / UNIT-TEST ORACLE, NOT THE PRODUCTION CUDA IMPLEMENTATION.
All distances are axial camera-Z metres, lambda_unknown is per metre,
and sigmas are standard deviations in metres. A Gaussian contributor is
modelled as a discrete first-hit candidate at its predicted centre depth.
A Poisson unknown-surface hazard supplies missing geometry along the ray.

Inputs must already be filtered to one mask-positive, valid-depth camera pixel.
Multi-camera likelihoods add their log probabilities; do NOT multiply/normalize
measurements by a foreground count unless that scaling is explicitly fixed.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch


def _logdiffcdf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log[Phi(b)-Phi(a)] for a < b, differentiable for finite nondegenerate spans.

    Use the survival-function identity when a >= 0 to avoid subtracting two
    near-one CDFs. Extreme-tail accuracy still needs high-precision tests.
    """
    if bool((b <= a).item()):
        raise ValueError("non-positive CDF interval")
    if bool((a >= 0).item()):
        hi = torch.special.log_ndtr(-a)
        lo = torch.special.log_ndtr(-b)
    else:
        hi = torch.special.log_ndtr(b)
        lo = torch.special.log_ndtr(a)
    ratio = torch.exp(lo - hi).clamp(max=1 - 2e-15)
    return hi + torch.log1p(-ratio)


def _normal_logpdf(value: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor):
    return -0.5 * ((value - mean) / sigma).square() - torch.log(sigma) - 0.5 * math.log(2 * math.pi)


def _log_unknown_interval(D, a, b, logS_a, lam, sigma):
    """log integral_a^b S(a)*lambda*exp(-lambda*(z-a))*N(D;z,sigma^2) dz."""
    if float(lam.detach()) == 0 or bool((b <= a).detach().item()):
        # Coincident modeled depths produce a zero-length unknown interval.
        # Ordering within that tie is stable; differentiability is piecewise.
        return D.new_full((), float('-inf'))
    h = lam * sigma
    zl = (a - D + lam * sigma * sigma) / sigma
    zh = (b - D + lam * sigma * sigma) / sigma
    return logS_a + torch.log(lam) - lam * (D - a) + 0.5 * h * h + _logdiffcdf(zl, zh)


@dataclass
class RayResult:
    log_valid_depth_likelihood: torch.Tensor
    log_no_return_probability: torch.Tensor
    gaussian_first_hit_mass: torch.Tensor   # original Gaussian order
    unknown_first_hit_mass: torch.Tensor
    no_return_mass: torch.Tensor
    gamma_gaussian: torch.Tensor           # original Gaussian order
    gamma_unknown: torch.Tensor
    log_unknown_evidence: torch.Tensor


def first_return_ray(
    depth_observed: torch.Tensor,
    contributor_depths: torch.Tensor,
    contributor_opacities: torch.Tensor,
    sigma_contributors: torch.Tensor,
    *,
    z_near: float,
    z_far: float,
    lambda_unknown: float,
    sigma_unknown: float,
) -> RayResult:
    """Evaluate a normalized first-hit prior and valid-depth density on ONE ray.

    The unknown depth is analytically marginalized over every interval between
    sorted modeled contributor depths. RGB and depth-return detection are not
    included in this minimal reference. For an invalid depth, with perfect
    detection, use log_no_return_probability instead.

    All contributor effective opacities must be in (0,1), with the same
    compositing semantics as the physical renderer. Sorting is piecewise
    differentiable and is recomputed for every nonlinear trial state.
    """
    D = depth_observed
    if D.ndim != 0 or D.dtype != torch.float64:
        raise ValueError("depth_observed must be a scalar float64 torch tensor")
    z = contributor_depths
    a = contributor_opacities
    sig = sigma_contributors
    if z.ndim != 1 or z.shape != a.shape or z.shape != sig.shape:
        raise ValueError("contributor arrays must all be shape [N]")
    if any(t.dtype != D.dtype or t.device != D.device for t in (z,a,sig)):
        raise ValueError("same dtype/device as observed depth required")
    if not (0 <= lambda_unknown and z_near < z_far and sigma_unknown > 0):
        raise ValueError("invalid ray support or unknown prior")
    if bool(((a <= 0) | (a >= 1) | (sig <= 0) | (z < z_near) | (z > z_far)).any().item()):
        raise ValueError("invalid contributor inputs")
    order = torch.argsort(z, stable=True)
    zo, ao, so = z[order], a[order], sig[order]
    lam = D.new_tensor(float(lambda_unknown))
    su = D.new_tensor(float(sigma_unknown))
    last = D.new_tensor(float(z_near))
    far = D.new_tensor(float(z_far))
    logS = D.new_zeros(())
    model_logq = []
    model_loge = []
    unknown_loge = []
    unknown_mass = D.new_zeros(())
    for zi, ai, si in zip(zo.unbind(), ao.unbind(), so.unbind()):
        before_interval = logS
        unknown_loge.append(_log_unknown_interval(D, last, zi, before_interval, lam, su))
        unknown_mass = unknown_mass + torch.exp(before_interval) * (-torch.expm1(-lam * (zi-last)))
        logS = logS - lam * (zi-last)
        qi = logS + torch.log(ai)
        model_logq.append(qi)
        model_loge.append(qi + _normal_logpdf(D, zi, si))
        logS = logS + torch.log1p(-ai)
        last = zi
    unknown_loge.append(_log_unknown_interval(D, last, far, logS, lam, su))
    unknown_mass = unknown_mass + torch.exp(logS) * (-torch.expm1(-lam * (far-last)))
    log_no_return = logS - lam * (far-last)
    if model_logq:
        mlogq = torch.stack(model_logq)
        mloge = torch.stack(model_loge)
        gaussian_first_hit_mass = torch.empty_like(a).scatter(0, order, torch.exp(mlogq))
    else:
        mloge = D.new_empty((0,))
        gaussian_first_hit_mass = a.new_empty((0,))
    u_loge = torch.logsumexp(torch.stack(unknown_loge), 0)
    loge = torch.logsumexp(torch.cat([mloge, u_loge.reshape(1)]), 0)
    if bool(torch.isneginf(loge)):
        gamma = a.new_zeros(a.shape)
        gu = D.new_zeros(())
    else:
        gm = torch.exp(mloge - loge)
        gamma = torch.zeros_like(a).scatter(0, order, gm)
        gu = torch.exp(u_loge - loge)
    return RayResult(
        log_valid_depth_likelihood=loge,
        log_no_return_probability=log_no_return,
        gaussian_first_hit_mass=gaussian_first_hit_mass,
        unknown_first_hit_mass=unknown_mass,
        no_return_mass=torch.exp(log_no_return),
        gamma_gaussian=gamma,
        gamma_unknown=gu,
        log_unknown_evidence=u_loge,
    )
