"""Stage A (sections 4-12): what is the fitted generic screw actually doing?

CODEX_VALIDATE_GENERIC_TWIST_THEN_PARALLEL_PRISMATIC_REVOLUTE_PHASE2.md.

The generic Phase 2 fits ONE screw and reports a single energy. That number
cannot say whether the motion it found is mostly a slide with a cosmetic
rotation, or a genuine hinge. Stage A takes the SAME fitted solution and
renders four states through the SAME measurement model:

    static | translation-only | rotation-only | full rigid

and reports the energy of each, per frame and per camera.

Section 4 fixes the decomposition. The split is taken about the part's
canonical COM,

    mu_i(t) = c_t + R_t r_i^0,     c_t = R_t c_0 + p_t,   r_i^0 = mu_i^0 - c_0,

NOT by separating the world-origin screw components `v` and `omega x x`. That
latter split moves when the world origin moves, so "how much of this is
translation" would depend on where the scene happens to be centred.

Nothing here is thresholded (section 7). These are diagnostics.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor

from .joint_kinematics import part_transform
from .manifold import so3_log

STATES = ("static", "translation", "rotation", "full")


def com_decomposition(part, frame: int, mu0: Tensor) -> dict:
    """Section 4: c_t, R_t and the canonical COM-referenced quantities."""
    R, p = part_transform(float(part.q[frame]), part.zeta, part.length)
    c0 = part.centroid.double()
    c_t = R @ c0 + p
    phi = so3_log(R.reshape(1, 3, 3))[0]
    return {"R": R, "p": p, "c0": c0, "c_t": c_t,
            "com_displacement": c_t - c0,
            "phi": phi, "theta": float(phi.norm())}


def build_state(kind: str, part, frame: int, mu0: Tensor,
                R0: Tensor | None, dec: dict | None = None):
    """Section 5: one of the four render states, as (mu, R_gaussian).

    Static Gaussians are left canonical in every one of them, so the only
    thing that differs between the four images is the moving part's motion
    model.
    """
    dec = dec or com_decomposition(part, frame, mu0)
    mu = mu0.double().clone()
    Rg = None if R0 is None else R0.double().clone()
    idx = part.members.to(mu.device)
    sub0 = mu0.double()[idx]
    R = dec["R"].to(mu.device)
    c0 = dec["c0"].to(mu.device)
    dcom = dec["com_displacement"].to(mu.device)

    if kind == "static":
        return mu, Rg
    if kind == "translation":
        mu[idx] = sub0 + dcom                      # orientation unchanged
        return mu, Rg
    if kind == "rotation":
        mu[idx] = c0 + (sub0 - c0) @ R.T           # COM held at c0
        if Rg is not None:
            Rg[idx] = R @ Rg[idx]
        return mu, Rg
    if kind == "full":
        mu[idx] = dec["c_t"].to(mu.device) + (sub0 - c0) @ R.T
        if Rg is not None:
            Rg[idx] = R @ Rg[idx]
        return mu, Rg
    raise ValueError(f"unknown state {kind!r}")


def _energies(lin, obs, prec, min_alpha: float) -> dict:
    """The SAME weighted RGB-D objective the IEKF uses (section 6)."""
    out = []
    for v, ob, (pc, pd) in zip(lin.views, obs, prec):
        if ob is None:
            out.append({"rgb": 0.0, "depth": 0.0, "total": 0.0,
                        "visible": 0})
            continue
        rr = (ob[0].to(v.rgb.device) - v.rgb).double()
        rd = (ob[1].to(v.depth.device) - v.depth).double()
        e_rgb = float((pc.double() * rr * rr).sum())
        e_d = float((pd.double() * rd * rd).sum())
        out.append({"rgb": 0.5 * e_rgb, "depth": 0.5 * e_d,
                    "total": 0.5 * (e_rgb + e_d),
                    "visible": int((v.alpha > min_alpha).sum())})
    return out


def audit_frame(part, frame: int, mu0, R0, obs, truth, cameras,
                build_lin, precision_fn, min_alpha: float) -> dict:
    """Sections 5-7: render all four states for one frame, same everything."""
    dec = com_decomposition(part, frame, mu0)
    per_state = {}
    for kind in STATES:
        mu, Rg = build_state(kind, part, frame, mu0, R0, dec)
        lin = build_lin(mu, Rg)
        # The precision is evaluated ONCE, on the full-rigid state, and reused
        # for all four. Recomputing it per state would let a robust weight
        # re-weight the pixels differently for each, and the four energies
        # would no longer be comparable -- section 6 requires the identical
        # measurement model, which includes its weights.
        if kind == "full":
            per_state["_prec"] = precision_fn(lin)
        per_state[kind] = lin
    prec = per_state["_prec"]
    res = {}
    for kind in STATES:
        res[kind] = _energies(per_state[kind], obs, prec, min_alpha)
        del per_state[kind]
    return {"decomposition": dec, "per_camera": res}


def rigidity_check(part, frame: int, mu0, R0, num_pairs: int = 500,
                   seed: int = 0, scales: Tensor | None = None,
                   require_orientation: bool = True) -> dict:
    """Sections 3, 10: distances, ORIENTATIONS and covariances are preserved.

    Pairwise distance drift alone is not enough (section 3): it is a test on
    centers, and it passes identically whether or not the Gaussian
    orientations were transported at all. The orientation and covariance
    checks are the ones that actually exercise `R_i(t) = R_t R_i^0` and
    `Sigma_i(t) = R_t Sigma_i^0 R_t^T`.

    `require_orientation` makes a missing `R0` an ERROR rather than a silent
    skip. A gate that quietly does nothing when its input is `None` is not a
    gate, and that is exactly how the previous audit reported "rigidity
    verified" while never touching an orientation.
    """
    dec = com_decomposition(part, frame, mu0)
    mu, Rg = build_state("full", part, frame, mu0, R0, dec)
    idx = part.members.to(mu.device)
    a0, at = mu0.double()[idx], mu[idx]
    n = int(idx.numel())
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, n, (num_pairs,), generator=g).to(mu.device)
    j = torch.randint(0, n, (num_pairs,), generator=g).to(mu.device)
    keep = i != j
    d0 = (a0[i[keep]] - a0[j[keep]]).norm(dim=1)
    dt = (at[i[keep]] - at[j[keep]]).norm(dim=1)
    out = {"num_pairs": int(keep.sum()),
           "max_pair_distance_drift": float((dt - d0).abs().max()),
           "max_abs_drift": float((dt - d0).abs().max()),
           "max_rel_drift": float(((dt - d0).abs()
                                   / d0.clamp_min(1e-300)).max())}
    if R0 is None or Rg is None:
        if require_orientation:
            raise ValueError(
                "section 3: rigidity_check needs the real canonical Gaussian "
                "orientations. R0 is None, so orientation and covariance "
                "transport would be silently skipped and the gate would pass "
                "without testing anything.")
        out["max_orientation_transport_error"] = None
        out["max_covariance_transport_error"] = None
        out["orientation_checked"] = False
        return out

    R = dec["R"].to(mu.device)
    R0d = R0.double()[idx]
    want = torch.einsum("ij,njk->nik", R, R0d)
    out["max_orientation_transport_error"] = float((Rg[idx] - want).abs().max())
    out["orientation_checked"] = True

    # Sigma = R diag(s^2) R^T with the REAL per-Gaussian scales. Using
    # R R^T = I instead would compare the identity with the identity.
    if scales is not None:
        s2 = (torch.as_tensor(scales).double().to(mu.device)[idx] ** 2)
        S0 = torch.einsum("nij,nj,nkj->nik", R0d, s2, R0d)
        St = torch.einsum("nij,nj,nkj->nik", Rg[idx], s2, Rg[idx])
        want_S = torch.einsum("ij,njk,lk->nil", R, S0, R)
        out["max_covariance_transport_error"] = float((St - want_S).abs().max())
        out["covariance_checked"] = True
    else:
        out["max_covariance_transport_error"] = None
        out["covariance_checked"] = False
    return out


PER_FRAME_CAMERA_HEADER = [
    "frame", "camera",
    "E_static_rgb", "E_trans_rgb", "E_rot_rgb", "E_full_rgb",
    "E_static_depth", "E_trans_depth", "E_rot_depth", "E_full_depth",
    "E_static_total", "E_trans_total", "E_rot_total", "E_full_total",
    "delta_rot_given_trans", "delta_trans_given_rot", "delta_full",
    "delta_trans", "delta_rot", "visible_moving_gaussians"]


def per_frame_camera_rows(audits: list) -> list:
    """Section 8's table, with the section 7 contribution diagnostics."""
    rows = []
    for t, a in enumerate(audits):
        pc = a["per_camera"]
        for ci in range(len(pc["full"])):
            S, Tr = pc["static"][ci], pc["translation"][ci]
            Ro, F = pc["rotation"][ci], pc["full"][ci]
            rows.append([
                t, ci,
                S["rgb"], Tr["rgb"], Ro["rgb"], F["rgb"],
                S["depth"], Tr["depth"], Ro["depth"], F["depth"],
                S["total"], Tr["total"], Ro["total"], F["total"],
                Tr["total"] - F["total"],          # rotation benefit after trans
                Ro["total"] - F["total"],          # translation benefit after rot
                S["total"] - F["total"],           # full gain over static
                S["total"] - Tr["total"],          # translation-only gain
                S["total"] - Ro["total"],          # rotation-only gain
                F["visible"]])
    return rows


def summarise(audits: list, rows: list) -> dict:
    """Section 14's questions, answered as numbers rather than a verdict."""
    import statistics as st

    def col(name):
        k = PER_FRAME_CAMERA_HEADER.index(name)
        return [r[k] for r in rows]

    full = sum(col("delta_full"))
    trans = sum(col("delta_trans"))
    rot = sum(col("delta_rot"))
    rot_given = sum(col("delta_rot_given_trans"))
    trans_given = sum(col("delta_trans_given_rot"))
    ncam = max(r[1] for r in rows) + 1 if rows else 0
    per_cam = {}
    for ci in range(ncam):
        sel = [r for r in rows if r[1] == ci]
        k = PER_FRAME_CAMERA_HEADER.index("delta_rot_given_trans")
        kf = PER_FRAME_CAMERA_HEADER.index("delta_full")
        tot = sum(r[kf] for r in sel)
        per_cam[ci] = {"delta_rot_given_trans": sum(r[k] for r in sel),
                       "delta_full": tot,
                       "rotation_share": (sum(r[k] for r in sel) / tot
                                          if tot else float("nan"))}
    shares = [v["rotation_share"] for v in per_cam.values()
              if not math.isnan(v["rotation_share"])]
    return {
        "delta_full": full, "delta_trans": trans, "delta_rot": rot,
        "delta_rot_given_trans": rot_given,
        "delta_trans_given_rot": trans_given,
        # "How much of what the full rigid model buys is the rotation?"
        "rotation_share_of_full_gain": (rot_given / full) if full else
        float("nan"),
        "translation_share_of_full_gain": (trans_given / full) if full else
        float("nan"),
        "translation_only_share": (trans / full) if full else float("nan"),
        "per_camera": per_cam,
        "rotation_share_spread_across_cameras":
            (max(shares) - min(shares)) if len(shares) > 1 else 0.0,
        "max_theta_rad": max(float(a["decomposition"]["theta"])
                             for a in audits) if audits else 0.0,
        "max_com_displacement": max(
            float(a["decomposition"]["com_displacement"].norm())
            for a in audits) if audits else 0.0,
        "note": ("section 7: these are diagnostics and are not thresholded. "
                 "A rotation share near zero means the fitted rotation is "
                 "doing no useful work once the COM displacement is allowed")}


# ---------------------------------------------------------------------------
# section 11: is ONE constant geometry actually supported?
# ---------------------------------------------------------------------------
def geometry_scores(state, mu0, R0, observations, linearize, precision_fn,
                    part: int = 0, camera_ids=None) -> dict:
    """Sections 6, 7, 11: geometry score and information PER FRAME AND CAMERA.

        g_theta,t,c = J_theta,t,c^T R_eff,t,c^-1 r_t,c
        F_theta,t,c = J_theta,t,c^T R_eff,t,c^-1 J_theta,t,c

    and only THEN the aggregates `g_theta,t = sum_c`, `g_theta,c = sum_t`,
    `g_theta = sum_{t,c}`.

    Concatenating the cameras first -- which is what the previous version did
    -- throws away the one decomposition that matters here. A fitted rotation
    that improves one view while hurting the others produces per-camera scores
    that point in opposite directions and a per-frame sum that looks small and
    well behaved. Blade is exactly that case, so the per-camera level is not a
    refinement, it is the diagnostic.

    Works for any Phase-2 model: the geometry block is read from
    `state.geometry_slice`, so the generic screw's 4 coordinates, M_P's 2 and
    M_R's 4 all go through this same code.
    """
    from utils.progress import pbar

    from .joint_renderer_bridge import joint_apply_H

    gslice = getattr(state, "geometry_slice", slice(2, 6))
    base = state.slice_of(part).start if hasattr(state, "slice_of") else 0
    g_dim = gslice.stop - gslice.start
    T = len(observations)
    cells, F_cells = {}, {}

    for t in pbar(range(T), "geometry scores", total=T):
        lin = linearize(state, t)
        prec = precision_fn(lin)
        obs = observations[t]
        cams = (list(range(len(lin.views))) if camera_ids is None
                else list(camera_ids))
        # whitened residual, kept PER CAMERA
        rw = {}
        for j, v in enumerate(lin.views):
            ob = obs[cams[j]] if camera_ids is not None else obs[j]
            if ob is None:
                rw[j] = torch.zeros(0, dtype=torch.float64)
                continue
            pc, pd = prec[j]
            rr = (ob[0].to(v.rgb.device) - v.rgb).double()
            rd = (ob[1].to(v.depth.device) - v.depth).double()
            rw[j] = torch.cat([
                (pc.double().clamp_min(0).sqrt() * rr).reshape(-1),
                (pd.double().clamp_min(0).sqrt() * rd).reshape(-1)]).cpu()
        # whitened geometry columns, also kept PER CAMERA
        Jc = {j: torch.zeros((rw[j].numel(), g_dim), dtype=torch.float64)
              for j in rw}
        for k in range(g_dim):
            e = torch.zeros(state.dim, dtype=torch.float64)
            e[base + gslice.start + k] = 1.0
            tang = joint_apply_H(lin, state, t, e, mu0, R0)
            for j, ((pc, pd), (dc, dd)) in enumerate(zip(prec, tang)):
                if rw[j].numel() == 0:
                    continue
                Jc[j][:, k] = torch.cat([
                    (pc.double().clamp_min(0).sqrt() * dc.double()).reshape(-1),
                    (pd.double().clamp_min(0).sqrt() * dd.double()).reshape(-1)
                ]).cpu()
        for j in rw:
            c = cams[j]
            cells[(t, c)] = Jc[j].T @ rw[j]
            F_cells[(t, c)] = Jc[j].T @ Jc[j]
        del lin

    return _aggregate_scores(cells, F_cells, g_dim)


def _alignment(vectors) -> dict:
    """Median pairwise cosine and the fraction of OPPOSED pairs."""
    V = [v for v in vectors if float(v.norm()) > 0]
    if len(V) < 2:
        return {"median_pairwise_cosine": float("nan"),
                "fraction_opposed": float("nan"), "num_blocks": len(V)}
    M = torch.stack(V)
    n = M / M.norm(dim=1, keepdim=True).clamp_min(1e-300)
    C = n @ n.T
    off = C[~torch.eye(C.shape[0], dtype=torch.bool)]
    return {"median_pairwise_cosine": float(off.median()),
            "fraction_opposed": float((off < 0).double().mean()),
            "num_blocks": len(V)}


def _aggregate_scores(cells: dict, F_cells: dict, g_dim: int) -> dict:
    """Section 6: build every aggregation level from the (t, c) cells."""
    frames = sorted({t for (t, _c) in cells})
    cams = sorted({c for (_t, c) in cells})
    z = lambda: torch.zeros(g_dim, dtype=torch.float64)          # noqa: E731
    per_frame = {t: z() for t in frames}
    per_cam = {c: z() for c in cams}
    F_total = torch.zeros((g_dim, g_dim), dtype=torch.float64)
    rows_fc = []
    for (t, c), g in sorted(cells.items()):
        F = F_cells[(t, c)]
        per_frame[t] = per_frame[t] + g
        per_cam[c] = per_cam[c] + g
        F_total = F_total + F
        rows_fc.append([t, c] + g.tolist()
                       + torch.linalg.eigvalsh(F).tolist() + [float(g.norm())])
    g_total = sum(per_frame.values(), z())
    rows_f = [[t] + per_frame[t].tolist() + [float(per_frame[t].norm())]
              for t in frames]
    rows_c = [[c] + per_cam[c].tolist() + [float(per_cam[c].norm())]
              for c in cams]
    af = _alignment([per_frame[t] for t in frames])
    ac = _alignment([per_cam[c] for c in cams])
    return {
        "g_dim": g_dim,
        "rows_frame_camera": rows_fc, "rows_frame": rows_f, "rows_camera": rows_c,
        "g_per_frame": torch.stack([per_frame[t] for t in frames])
        if frames else torch.zeros((0, g_dim), dtype=torch.float64),
        "g_per_camera": torch.stack([per_cam[c] for c in cams])
        if cams else torch.zeros((0, g_dim), dtype=torch.float64),
        "F_total": F_total,
        "F_total_eigenvalues": torch.linalg.eigvalsh(F_total).tolist(),
        "sum_g_norm": float(g_total.norm()),
        "mean_abs_g_norm": float(
            torch.stack(list(cells.values())).norm(dim=1).mean())
        if cells else 0.0,
        "score_norm_per_frame": {int(t): float(per_frame[t].norm())
                                 for t in frames},
        "score_norm_per_camera": {int(c): float(per_cam[c].norm())
                                  for c in cams},
        "across_frames": af, "across_cameras": ac,
        # kept flat for the callers and reports that read these two names
        "median_pairwise_cosine": af["median_pairwise_cosine"],
        "fraction_opposed": af["fraction_opposed"],
        "interpretation": (
            "sections 6, 7: a median pairwise cosine near 1 means the blocks "
            "pull the geometry the same way; near 0 or negative means the one "
            "constant geometry is an average compromise between blocks that "
            "disagree. ACROSS CAMERAS disagreeing at one frame is model "
            "mismatch; across frames is the constant-geometry assumption"),
    }


def geometry_score_per_frame(state, mu0, R0, observations, linearize,
                             precision_fn, part: int = 0) -> dict:
    """Section 11: per-frame geometry score and information.

        g_theta,t = J_theta^T R_eff^-1 r_t,   F_theta,t = J_theta^T R_eff^-1 J_theta

    The generic Phase 2 imposes one constant screw. If every frame pulls the
    geometry the same way these scores point the same way; if the constant
    geometry is only an average compromise, they cancel. The sum is near zero
    at the fitted optimum BY CONSTRUCTION, so what matters is whether the
    individual terms are small or large-and-opposed -- which is what the
    alignment numbers below report.

    Nothing is relaxed here (section 11): this measures the fit, it does not
    change it.
    """
    out = geometry_scores(state, mu0, R0, observations, linearize,
                          precision_fn, part)
    out["rows"] = out["rows_frame"]
    return out


def geometry_score_headers(g_dim: int = 4):
    """Section 6's three CSV headers, for a model with `g_dim` geometry DOF."""
    g = [f"g_theta_{k}" for k in range(g_dim)]
    f = [f"F_theta_eig_{k}" for k in range(g_dim)]
    return {"frame_camera": ["frame", "camera"] + g + f + ["g_norm"],
            "frame": ["frame"] + g + ["g_norm"],
            "camera": ["camera"] + g + ["g_norm"]}


GEOMETRY_SCORE_HEADER = (["frame"] + [f"g_theta_{k}" for k in range(4)]
                         + ["g_norm"])
