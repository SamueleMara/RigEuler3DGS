"""Bayesian static/dynamic posterior update without lambda soup."""

from __future__ import annotations

import torch

EPS = 1.0e-8
PROB_EPS = 1.0e-6


def compute_static_hypothesis_likelihood(observed_displacement: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    disp = torch.as_tensor(observed_displacement).detach().float()
    cov = torch.as_tensor(covariance).detach().float()
    if cov.ndim == 2:
        cov = cov.expand(disp.shape[0], -1, -1)
    inv = torch.linalg.pinv(cov + torch.eye(cov.shape[-1], device=cov.device)[None] * EPS)
    maha = torch.einsum("ni,nij,nj->n", disp, inv.to(disp), disp)
    return torch.exp(-0.5 * maha).clamp_min(EPS)


def compute_dynamic_hypothesis_likelihood(observed_displacement: torch.Tensor, predicted_displacement: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    return compute_static_hypothesis_likelihood(torch.as_tensor(observed_displacement) - torch.as_tensor(predicted_displacement), covariance)


def update_dynamic_posterior(prior_dynamic: torch.Tensor, static_likelihood: torch.Tensor, dynamic_likelihood: torch.Tensor) -> torch.Tensor:
    prior = torch.as_tensor(prior_dynamic).detach().float().clamp(PROB_EPS, 1.0 - PROB_EPS)
    ls = torch.as_tensor(static_likelihood).detach().float().clamp_min(EPS)
    ld = torch.as_tensor(dynamic_likelihood).detach().float().clamp_min(EPS)
    # Work in log-odds.  The old odds-domain update produced NaNs for exact
    # p=1 priors in float32: 1 - 1e-8 rounds back to 1, odds=inf, inf/(1+inf)=NaN.
    posterior_logit = torch.logit(prior) + torch.log(ld) - torch.log(ls)
    return torch.sigmoid(posterior_logit).clamp(PROB_EPS, 1.0 - PROB_EPS)
