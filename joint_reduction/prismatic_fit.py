"""Section 15: the geometric prismatic fit.

CODEX_IMPLEMENT_GEOMETRIC_RIGID_MOTION_JOINT_DISCOVERY.md section 15, 20, 22.

A prismatic joint means the body does not turn and its translations are
rank 1:

    R_{k,t} ~= I,        t_{k,t} = q_{k,t} e_k,     ||e_k|| = 1.

so the fit is

    E_P = min_{e,q} sum_t [ w^p ||t_t - q_t e||^2 + w^R ||log R_t||^2 ].

This has a CLOSED FORM, which matters: section 19 requires both fits to be
converged before they are compared, and a closed form cannot fail to converge.
For a fixed e the inner minimiser is q_t = e^T t_t, leaving

    sum_t ||t_t||^2 - (e^T t_t)^2,

so the optimal e maximises e^T (sum_t t_t t_t^T) e -- the leading eigenvector
of the translation scatter. That is exactly the "PCA/SVD of the recovered
COM/translation trajectory" section 15 asks for, and here it is the optimum
rather than only an initializer.

The rotation term carries no free parameter. It is the price a prismatic model
pays for any rotation the data actually contains, and dropping it would let a
revolute motion masquerade as prismatic by ignoring its own turning.
"""

from __future__ import annotations

import torch

from estimator.manifold import project_to_so3, so3_log

from .joint_geometry import normalized_geometric_error

EPS = 1e-12


def fit_prismatic(R_obs: torch.Tensor, t_obs: torch.Tensor, length: float,
                  *, weights: torch.Tensor | None = None) -> dict:
    """Section 15's E_P, in closed form.

    `R_obs`/`t_obs` are the recovered {T_{k,t}} of ONE part, `length` is
    section 20's L_k.

    Returns the axis, the trajectory q(t), the predicted transforms and the
    section 20 normalized error.
    """
    R_obs = R_obs.double()
    t_obs = t_obs.double()
    T = int(t_obs.shape[0])
    dev = t_obs.device
    if T == 0:
        raise ValueError("a prismatic fit needs at least one frame")
    w = (torch.ones(T, dtype=torch.float64, device=dev)
         if weights is None else weights.double().clamp_min(0.0))

    # Section 22's gauge q_0 = 0 is imposed by measuring every translation
    # RELATIVE to the first frame, so the fit never has to know where the
    # sequence started.
    t_rel = t_obs - t_obs[0:1]
    scatter = (w.unsqueeze(1) * t_rel).T @ t_rel          # [3,3]
    evals, evecs = torch.linalg.eigh(scatter)
    axis = evecs[:, -1]
    axis = axis / axis.norm().clamp_min(EPS)
    q = t_rel @ axis                                      # q_t = e^T t_t

    # Orient the axis so the motion is non-negative on average: e and -e
    # describe the same joint, and a sign convention keeps q(t) comparable
    # across runs without ever consulting ground truth.
    if float(q.sum()) < 0:
        axis, q = -axis, -q

    t_hat = t_obs[0:1] + q.unsqueeze(1) * axis.unsqueeze(0)
    R_hat = torch.eye(3, dtype=torch.float64, device=dev).expand(T, 3, 3)
    err = normalized_geometric_error(R_obs, t_obs, R_hat, t_hat, length)

    rank_ratio = float((evals[-1] / evals.sum().clamp_min(EPS)))
    body_rotation = [float(x) for x in
                     so3_log(project_to_so3(R_obs)).norm(dim=1)]
    return {"kind": "prismatic", "axis": axis, "q": [float(x) for x in q],
            "R_hat": R_hat, "t_hat": t_hat, "origin": t_obs[0].clone(),
            "error": err["error"], "error_terms": err,
            "num_parameters": 2 + T,       # 2 for e on S^2, one q per frame
            "converged": True,
            "convergence": {"closed_form": True,
                            "note": ("section 15 has a closed-form optimum, so "
                                     "section 28's joint-fit convergence test "
                                     "is satisfied exactly rather than "
                                     "iteratively")},
            "diagnostics": {
                "translation_rank1_ratio": rank_ratio,
                "translation_eigenvalues": [float(x) for x in evals],
                "body_rotation_angle_rad": body_rotation,
                "max_body_rotation_rad": max(body_rotation) if body_rotation
                else 0.0}}
