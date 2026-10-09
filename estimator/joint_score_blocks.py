"""Sections 7, 9, 13: whitened scores and information, by camera/time block.

CODEX_NEXT_EFFICIENT_P_NULL_AND_PHASE2_DEBUG.md.

Everything is built from the EXISTING analytic chain (section 9):

    joint tangent -> joint kinematic JVP -> renderer JVP -> R_eff whitening

Nothing is finite-differenced in production.

The nuisance vector is section 7's

    alpha = [gamma_1, gamma_2, q_1, ..., q_{T-1}],      q_0 = 0,

with `qdot` deliberately absent: the image has no `qdot` derivative at all, so
including it would count the process prior as extra observations. The
hypothesis directions are section 8's two curvature coordinates `beta`.

A `q_t` coordinate moves only frame `t`, so its whitened column is zero
everywhere else -- which is what keeps the cost at five operator applications
per frame rather than `2 + T`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .joint_kinematics import current_points
from .joint_renderer_bridge import joint_apply_H
from .prismatic_null import BETA_DIM, GAMMA_DIM, null_tangent_basis

DEFAULT_BLOCK_FRAMES = 5


def alpha_dim(num_frames: int) -> int:
    """gamma (2) + q_1..q_{T-1}; q_0 is the gauge and is not a parameter."""
    return GAMMA_DIM + max(num_frames - 1, 0)


def _whiten(prec, tensors):
    """R_eff^{-1/2} applied to an RGB-D pair; `prec` is the precision."""
    out = []
    for (pc, pd), (a, b) in zip(prec, tensors):
        out.append((pc.clamp_min(0.0).sqrt() * a,
                    pd.clamp_min(0.0).sqrt() * b))
    return out


def _flat(pairs) -> Tensor:
    """Flatten an RGB-D pair to one float64 vector on the HOST.

    The scores and information blocks are tiny and are accumulated in float64
    (section 9); keeping the reduction on the host avoids mixing devices when
    the explicit-projection validation materialises a Jacobian.
    """
    return torch.cat([t.reshape(-1).double().cpu() for p in pairs for t in p])


def accumulate_scores(state, mu0, R0, observations, linearize, precision_fn,
                      *, block_frames: int = DEFAULT_BLOCK_FRAMES,
                      part: int = 0) -> dict:
    """Sections 9, 13: F blocks, total scores, and per-block efficient inputs.

    Blocks are `(camera, contiguous time window)` with one globally fixed
    window length -- section 13 forbids tuning block construction per scene.
    """
    p = state.parts[part]
    T = len(observations)
    na, nb = alpha_dim(T), BETA_DIM
    F_aa = torch.zeros((na, na), dtype=torch.float64)
    F_ab = torch.zeros((na, nb), dtype=torch.float64)
    F_bb = torch.zeros((nb, nb), dtype=torch.float64)
    g_a = torch.zeros(na, dtype=torch.float64)
    g_b = torch.zeros(nb, dtype=torch.float64)
    blocks: dict = {}

    for t in range(T):
        lin = linearize(state, t)
        prec = precision_fn(lin)
        obs = observations[t]
        mu_now, _rg, _R, _tt = current_points(p, t, mu0, R0)

        # whitened residual, per camera
        res = []
        for v, ob, (pc, pd) in zip(lin.views, obs, prec):
            if ob is None:
                res.append((torch.zeros_like(v.rgb), torch.zeros_like(v.depth)))
                continue
            res.append((ob[0].to(v.rgb.device) - v.rgb,
                        ob[1].to(v.depth.device) - v.depth))
        rw = _whiten(prec, res)

        # the columns this frame contributes to
        cols: list = []           # (index_into_alpha_or_None, beta_index, vec)
        full = torch.zeros(6, dtype=torch.float64)

        def column(vec6):
            tang = joint_apply_H(lin, state, t, _embed(state, part, vec6),
                                 mu0, R0)
            return _whiten(prec, tang)

        B, _bg, _bb = null_tangent_basis(p.zeta)
        # gamma_1, gamma_2 -> alpha slots 0, 1
        for k in range(GAMMA_DIM):
            v6 = torch.zeros(6, dtype=torch.float64)
            v6[2 + k] = 1.0
            cols.append(("a", k, column(v6)))
        # q_t -> alpha slot 2 + (t - 1); frame 0 is the gauge and has none
        if t >= 1:
            v6 = torch.zeros(6, dtype=torch.float64)
            v6[0] = 1.0
            cols.append(("a", GAMMA_DIM + (t - 1), column(v6)))
        # beta_1, beta_2
        for k in range(BETA_DIM):
            v6 = torch.zeros(6, dtype=torch.float64)
            v6[4 + k] = 1.0
            cols.append(("b", k, column(v6)))

        ncam = len(lin.views)
        win = max(1, int(block_frames))
        for ci in range(ncam):
            key = (ci, t // win)
            rec = blocks.setdefault(
                key, {"camera": ci, "window": t // win, "frames": [],
                      "s_alpha": torch.zeros(na, dtype=torch.float64),
                      "s_beta": torch.zeros(nb, dtype=torch.float64)})
            rec["frames"].append(t)
            r_c = _flat([rw[ci]])
            for kind, idx, colv in cols:
                j_c = _flat([colv[ci]])
                s = float(j_c @ r_c)
                if kind == "a":
                    rec["s_alpha"][idx] += s
                    g_a[idx] += s
                else:
                    rec["s_beta"][idx] += s
                    g_b[idx] += s
            # information blocks, accumulated in float64
            for kind_i, i, ci_v in cols:
                ji = _flat([ci_v[ci]])
                for kind_j, j, cj_v in cols:
                    if (kind_i, i) > (kind_j, j):
                        continue
                    jj = _flat([cj_v[ci]])
                    val = float(ji @ jj)
                    if kind_i == "a" and kind_j == "a":
                        F_aa[i, j] += val
                        if i != j:
                            F_aa[j, i] += val
                    elif kind_i == "a" and kind_j == "b":
                        F_ab[i, j] += val
                    elif kind_i == "b" and kind_j == "a":
                        F_ab[j, i] += val
                    else:
                        F_bb[i, j] += val
                        if i != j:
                            F_bb[j, i] += val
        del lin

    order = sorted(blocks)
    return {"F_aa": F_aa, "F_ab": F_ab, "F_bb": F_bb,
            "g_alpha": g_a, "g_beta": g_b,
            "blocks": [blocks[k] for k in order],
            "alpha_dim": na, "beta_dim": nb,
            "block_frames": block_frames,
            "num_blocks": len(order),
            "layout": ("alpha = [gamma_1, gamma_2, q_1..q_{T-1}]; "
                       "beta = [beta_1, beta_2]; q_0 is the gauge and qdot "
                       "is excluded because dh/dqdot = 0 (section 7)")}


def _embed(state, part: int, vec6: Tensor) -> Tensor:
    """Place one part's 6-vector into the full joint tangent."""
    out = torch.zeros(state.dim, dtype=torch.float64)
    out[state.slice_of(part)] = vec6
    return out


def explicit_projection_check(state, mu0, R0, observations, linearize,
                              precision_fn, part: int = 0) -> dict:
    """Section 11: materialise the whitened Jacobians and verify the Schur form.

    Only for validation, and only on a tiny problem: this builds `J_alpha` and
    `J_beta` explicitly so that

        F_{beta|alpha} == J_{beta|alpha}^T J_{beta|alpha},
        J_{beta|alpha} = (I - J_alpha J_alpha^+) J_beta

    can be checked against the Schur complement production uses.
    """
    p = state.parts[part]
    T = len(observations)
    na, nb = alpha_dim(T), BETA_DIM
    Ja_rows, Jb_rows, r_rows = [], [], []

    for t in range(T):
        lin = linearize(state, t)
        prec = precision_fn(lin)
        obs = observations[t]
        res = []
        for v, ob in zip(lin.views, obs):
            if ob is None:
                res.append((torch.zeros_like(v.rgb), torch.zeros_like(v.depth)))
            else:
                res.append((ob[0].to(v.rgb.device) - v.rgb,
                            ob[1].to(v.depth.device) - v.depth))
        r_rows.append(_flat(_whiten(prec, res)))

        def col(v6):
            return _flat(_whiten(prec, joint_apply_H(
                lin, state, t, _embed(state, part, v6), mu0, R0)))

        ja = torch.zeros((r_rows[-1].numel(), na), dtype=torch.float64)
        for k in range(GAMMA_DIM):
            v6 = torch.zeros(6, dtype=torch.float64)
            v6[2 + k] = 1.0
            ja[:, k] = col(v6)
        if t >= 1:
            v6 = torch.zeros(6, dtype=torch.float64)
            v6[0] = 1.0
            ja[:, GAMMA_DIM + (t - 1)] = col(v6)
        jb = torch.zeros((r_rows[-1].numel(), nb), dtype=torch.float64)
        for k in range(BETA_DIM):
            v6 = torch.zeros(6, dtype=torch.float64)
            v6[4 + k] = 1.0
            jb[:, k] = col(v6)
        Ja_rows.append(ja)
        Jb_rows.append(jb)
        del lin

    Ja = torch.cat(Ja_rows, dim=0)
    Jb = torch.cat(Jb_rows, dim=0)
    r = torch.cat(r_rows, dim=0)
    Pperp = torch.eye(Ja.shape[0], dtype=torch.float64) \
        - Ja @ torch.linalg.pinv(Ja)
    Jba = Pperp @ Jb
    return {"J_alpha": Ja, "J_beta": Jb, "residual": r,
            "F_beta_given_alpha_explicit": Jba.T @ Jba,
            "g_beta_given_alpha_explicit": Jba.T @ r,
            "F_aa": Ja.T @ Ja, "F_ab": Ja.T @ Jb, "F_bb": Jb.T @ Jb,
            "g_alpha": Ja.T @ r, "g_beta": Jb.T @ r}
