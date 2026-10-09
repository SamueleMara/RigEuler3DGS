from __future__ import annotations

import torch

from utils.semantic_render_likelihood import compute_foreground_depth_likelihood, compute_foreground_rgb_likelihood
from utils.static_dynamic_bayesian_update import compute_dynamic_hypothesis_likelihood, compute_static_hypothesis_likelihood, update_dynamic_posterior


def test_dynamic_posterior_moves_with_likelihood_ratio():
    prior = torch.tensor([0.2, 0.8])
    cov = torch.eye(3).expand(2, 3, 3) * 0.01
    observed = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    pred = torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    ps = compute_static_hypothesis_likelihood(observed, cov)
    pd = compute_dynamic_hypothesis_likelihood(observed, pred, cov)
    post = update_dynamic_posterior(prior, ps, pd)
    assert post[0] > prior[0]
    assert post[1] < prior[1]


def test_dynamic_posterior_is_finite_for_saturated_priors():
    prior = torch.tensor([0.0, 1.0, 0.999999, 1.0], dtype=torch.float32)
    static_l = torch.tensor([1.0, 1.0e-8, 0.25, 1.0])
    dynamic_l = torch.tensor([1.0e-8, 1.0, 0.75, 1.0])
    post = update_dynamic_posterior(prior, static_l, dynamic_l)
    assert torch.isfinite(post).all()
    assert (post > 0.0).all() and (post < 1.0).all()


def test_render_likelihood_ignores_background():
    obs = torch.ones(3, 2, 2)
    pred = obs.clone()
    pred[:, 0, 1] = 100.0
    mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    rgb = compute_foreground_rgb_likelihood(obs, pred, mask, scale=1.0)
    depth = compute_foreground_depth_likelihood(torch.ones(2, 2), torch.ones(2, 2) * 2.0, mask, scale=1.0)
    assert rgb["mean_abs_residual"] == 0.0
    assert depth["pixels"] == 3


if __name__ == "__main__":
    test_dynamic_posterior_moves_with_likelihood_ratio()
    test_dynamic_posterior_is_finite_for_saturated_priors()
    test_render_likelihood_ignores_background()
    print("PASS closed_loop_static_dynamic_update")
