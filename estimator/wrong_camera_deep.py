"""Deeper READ-ONLY analyses of the frozen 80->81 update (wrong-camera audit, follow-up).

objective_conflict     per camera x channel force b[c,ch] (production H^T W r, the descent
                       direction of that term) projected on each selected Gaussian's ACCEPTED
                       6D pose change d (left tangent, x_post boxminus x_prior): s > 0 the term
                       pulls along the update, s < 0 it opposes it. Also the prior force
                       -Lambda_prior (x_j boxminus x_prior), and the same at the supernode level
                       (G^T b against the group's least-squares rigid increment). Evaluated at
                       inner iteration 0 (x = x_prior, the prior exerts no force yet) and at the
                       LAST inner iteration of the frame (the balance at the accepted iterate).
group_mediation        the exact split of a camera's frozen leave-one-out influence
                           I_c = A^-1 b_c[own] + A^-1 b_c[groupmates] + A^-1 b_c[rest] - A^-1 A_c(d_-c)
                       own = the runaway Gaussians, groupmates = the other members of their
                       supernodes, rest = everything else. In the canonical GROUPED system and in
                       the per-Gaussian (UNGROUPED) system with the same frozen measurement
                       factors, prior and damping -- only the state map differs.
nonlinear_attribution  per camera x channel energy of the full production objective (true
                       render) at x_prior, at the accepted x_post, and with ONLY a subset (the
                       runaway set, the stable set, one supernode) moved to x_post; plus the
                       exact prior energy 1/2 d^T Lambda_prior d of that move. Actual losses,
                       not linearizations.
"""
from __future__ import annotations

import dataclasses
import time

import torch

from . import wrong_camera_audit as wa


def _q(x, qs=(0.1, 0.5, 0.9)):
    x = torch.as_tensor(x).double().reshape(-1)
    x = x[torch.isfinite(x)]
    return None if x.numel() == 0 else [float(v) for v in torch.quantile(x.cpu(), torch.tensor(qs, dtype=torch.float64))]


def group_increment(grouped, dpose):
    """Least-squares rigid increment per supernode: argmin_z ||G z - dpose||^2 (pose part)."""
    C = grouped.num_supernodes
    dev = dpose.device
    GtG = torch.zeros((C, 6, 6), dtype=torch.float64, device=dev)
    for k in range(6):
        e = torch.zeros((C, 6), device=dev)
        e[:, k] = 1.0
        GtG[:, :, k] = grouped.reduce(grouped.expand(e).contiguous()).double()
    rhs = grouped.reduce(dpose.float().contiguous()).double()
    GtG = 0.5 * (GtG + GtG.transpose(1, 2)) + 1e-12 * torch.eye(6, dtype=torch.float64, device=dev)
    return torch.linalg.solve(GtG, rhs.unsqueeze(-1)).squeeze(-1)


def _proj_rows(f, unit, ids_t):
    s = (f[ids_t] * unit[ids_t]).sum(1)
    return {"median_projection": float(s.median()), "q10_q50_q90": _q(s),
            "fraction_pulling": float((s > 0).double().mean()), "fraction_opposing": float((s < 0).double().mean())}


@torch.no_grad()
def objective_conflict(lin, terms, dpose, sets: dict, grouped, names, sid=None, prior_force=None) -> dict:
    """dpose [N,6] accepted pose change; prior_force [N,6] (pose part of -Lambda (x - x_prior)) or None."""
    b = wa.camera_channel_rhs(lin, terms)
    keys = [(c, ch) for c in range(len(names)) for ch in wa.CHANNELS]
    dp = dpose.double().to(b[keys[0]].device)
    unit = dp / dp.norm(dim=1, keepdim=True).clamp_min(1e-30)
    forces = {f"{names[c]}:{ch}": b[(c, ch)] for c, ch in keys}
    forces["MEASUREMENT_TOTAL"] = sum(b[k] for k in keys)
    for c, nm in enumerate(names):
        forces[f"{nm}:ALL"] = sum(b[(c, ch)] for ch in wa.CHANNELS)
    if prior_force is not None:
        forces["PRIOR"] = prior_force.double().to(dp.device)
    out = {"gaussian": {}, "supernode": {}}
    for name, ids in sets.items():
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=dp.device)
        out["gaussian"][name] = {k: _proj_rows(f, unit, ids_t) for k, f in forces.items()}
    if grouped is not None and sid is not None:
        z = group_increment(grouped, dp)
        zu = z / z.norm(dim=1, keepdim=True).clamp_min(1e-30)
        for name, ids in sets.items():
            groups = torch.unique(torch.as_tensor(sid)[torch.as_tensor(ids, dtype=torch.long)]).to(dp.device)
            rows = {k: _proj_rows(grouped.reduce(f.float().contiguous()).double(), zu, groups) for k, f in forces.items()}
            out["supernode"][name] = {"num_supernodes": int(groups.numel()), "terms": rows}
    return out


@torch.no_grad()
def group_mediation(systems: dict, lin, terms, names, row_parts: dict, sets: dict, tol=1e-6, max_iters=None) -> dict:
    """systems {label: (FrozenSystem, max_iters)}; row_parts {name: bool[N]} partition of the rows."""
    from .renderer_bridge import embed_pose_to_full_state, extract_pose
    b = wa.camera_channel_rhs(lin, terms)
    out = {}
    for label, (sys_, mi) in systems.items():
        t0 = time.time()
        d_all, _, inf_all = sys_.solve((), tol=tol, max_iters=mi)
        res = {"solver_all": inf_all, "cameras": {}}
        for c, nm in enumerate(names):
            keys = [(c, ch) for ch in wa.CHANNELS]
            d_k, _, inf_k = sys_.solve(keys, tol=tol, max_iters=mi)
            I = d_all - d_k
            bc = sum(b[k] for k in keys)
            Ak = sum(wa.camera_channel_normal(lin, terms, extract_pose(d_k), keys=set(keys)).values())
            rhs = {f"force_{pn}": torch.where(m.to(bc.device)[:, None], bc, torch.zeros_like(bc))
                   for pn, m in row_parts.items()}
            rhs["curvature"] = -Ak
            parts = {}
            for pn, r in rhs.items():
                y, _, inf_y = sys_.solve((), rhs12=embed_pose_to_full_state(r), tol=tol, max_iters=mi)
                parts[pn] = (y, inf_y)
            recon = sum(p[0] for p in parts.values())
            rec = {"solver": inf_k, "part_solvers": {pn: p[1] for pn, p in parts.items()},
                   "decomposition_rel_err": float((recon - I).norm() / I.norm().clamp_min(1e-30)), "sets": {}}
            for sn, ids in sets.items():
                ids_t = torch.as_tensor(ids, dtype=torch.long, device=I.device)
                It = I[ids_t, :3]
                nI2 = It.square().sum(1).clamp_min(1e-60)
                row = {"influence_translation_mm_q": [1000 * v for v in _q(It.norm(dim=1))],
                       "delta_all_translation_mm_q": [1000 * v for v in _q(d_all[ids_t, :3].norm(dim=1))],
                       "influence_fraction_q": _q(It.norm(dim=1) / d_all[ids_t, :3].norm(dim=1).clamp_min(1e-12))}
                for pn, (y, _) in parts.items():
                    yt = y[ids_t, :3]
                    row[f"{pn}_share_along_I_q"] = _q((yt * It).sum(1) / nI2)
                    row[f"{pn}_translation_mm_q"] = [1000 * v for v in _q(yt.norm(dim=1))]
                rec["sets"][sn] = row
            res["cameras"][nm] = rec
        res["seconds"] = round(time.time() - t0, 1)
        out[label] = res
    out["note"] = ("share_along_I = <part, I> / |I|^2 per Gaussian, translation only; the parts sum to I exactly "
                   "(decomposition_rel_err), so the shares sum to 1. grouped vs ungrouped use the SAME frozen "
                   "measurement factors, prior and damping; only the state map (G vs identity) differs. Frozen "
                   "unconstrained quadratic, NOT the production trust-region step.")
    return out


def _energies(terms, names):
    return {f"{names[c]}:{ch}": float(t.e_rgb if ch == "rgb" else t.e_d if ch == "depth" else (t.e_a or 0.0))
            for c, t in enumerate(terms) for ch in wa.CHANNELS}


@torch.no_grad()
def nonlinear_attribution(x_prior, x_post, Lambda_prior, render_terms, names, sid, sets: dict, top_groups=20) -> dict:
    """render_terms(state) -> per-camera production CameraTerms on THIS frame's data."""
    from .covariance import apply_prior
    from .state import boxminus
    delta = boxminus(x_post, x_prior).double()                        # [N,12]

    def moved(mask):
        m = mask.to(x_prior.mu.device)
        pick = lambda a, b: torch.where(m.reshape(-1, *([1] * (a.dim() - 1))), b, a)  # noqa: E731
        return dataclasses.replace(x_prior, mu=pick(x_prior.mu, x_post.mu), R=pick(x_prior.R, x_post.R),
                                   v=pick(x_prior.v, x_post.v), omega=pick(x_prior.omega, x_post.omega))

    def prior_energy(mask):
        d = torch.where(mask.to(delta.device)[:, None], delta, torch.zeros_like(delta))
        return 0.5 * float((d * apply_prior(Lambda_prior.double(), d)).sum())

    n = x_prior.num_gaussians
    E0 = _energies(render_terms(x_prior), names)

    def row(mask):
        E = _energies(render_terms(moved(mask)), names)
        dE = {k: E[k] - E0[k] for k in E0}
        dE["MEASUREMENT_TOTAL"] = sum(dE.values())
        dE["PRIOR"] = prior_energy(mask)
        dE["TOTAL"] = dE["MEASUREMENT_TOTAL"] + dE["PRIOR"]
        return dE

    out = {"definition": "dE = E(subset moved to x_post) - E(x_prior) per camera:channel on the full nonlinear "
                         "production objective; dE < 0: that term's loss decreased, i.e. the term favours the move. "
                         "PRIOR = 1/2 d^T Lambda_prior d of the same move (>= 0, always opposes).",
           "E_x_prior": E0, "all_moved": row(torch.ones(n, dtype=torch.bool))}
    sid_t = torch.as_tensor(sid) if sid is not None else None
    for name, ids in sets.items():
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.as_tensor(ids, dtype=torch.long)] = True
        out[f"only_{name}_moved"] = row(m)
        if sid_t is not None:
            out[f"only_{name}_supernodes_moved"] = row(torch.isin(sid_t, torch.unique(sid_t[m])))
    run = torch.as_tensor(sets.get("runaway", []), dtype=torch.long)
    if run.numel() and sid_t is not None:
        g, cnt = torch.unique(sid_t[run], return_counts=True)
        rows = []
        for k in torch.argsort(cnt, descending=True)[:top_groups].tolist():
            m = sid_t == int(g[k])
            rows.append({"supernode": int(g[k]), "members": int(m.sum()), "runaway_members": int(cnt[k]),
                         "max_translation_mm": 1000 * float(delta[m.to(delta.device), :3].norm(dim=1).max()),
                         "dE": row(m)})
        out["per_supernode_top_runaway"] = rows
    return out


class DeepObserver:
    """iteration_observer for ONE frame (processing index `frame`): objective conflict at
    iteration 0 and at the last inner iteration, group mediation and the nonlinear
    attribution at iteration 0. `x_post` is the accepted state of that frame (a capture of
    the same replay; bitwise identical under --deterministic)."""

    def __init__(self, camera_names, sets: dict, x_post: dict, frame=1, tol=1e-6, max_iters=400,
                 max_iters_ungrouped=1500, top_groups=20, mediation=True, nonlinear=True):
        self.names, self.sets, self.x_post, self.frame = list(camera_names), sets, x_post, frame
        self.tol, self.max_iters, self.max_iters_ungrouped = tol, max_iters, max_iters_ungrouped
        self.top_groups, self.mediation, self.nonlinear = top_groups, mediation, nonlinear
        self.ti, self.conflict, self.group_mediation, self.attribution = None, {}, None, None
        self._post = None

    def begin_frame(self, ti, fid):
        self.ti = ti

    def _post_state(self, x):
        if self._post is None:
            cv = lambda t: torch.as_tensor(t).to(device=x.mu.device, dtype=x.mu.dtype)  # noqa: E731
            self._post = dataclasses.replace(x, mu=cv(self.x_post["mu"]), R=cv(self.x_post["R"]),
                                             v=cv(self.x_post["v"]), omega=cv(self.x_post["omega"]))
        return self._post

    @torch.no_grad()
    def __call__(self, *, iteration, x, x_prior, Lambda_prior, lin, terms, b, meas_normal, grouped, lm, M_blocks,
                 rgbd_gt, cameras, cfg, background=None, visibility=None, observation_model=None, **kw):
        if self.ti != self.frame:
            return
        from .covariance import apply_prior
        from .pcg import block_jacobi_preconditioner
        from .sparse_iekf import _frame_terms, group_precond
        from .state import boxminus
        from .renderer_bridge import build_linearization
        post = self._post_state(x)
        dpose = boxminus(post, x_prior)[:, :6]
        sid = grouped.part.supernode_id.long().cpu() if grouped is not None else None
        prior_f = -apply_prior(Lambda_prior.double(), boxminus(x, x_prior).double())[:, :6]
        rec = objective_conflict(lin, terms, dpose, self.sets, grouped, self.names, sid, prior_force=prior_f)
        rec["iteration"] = int(iteration)
        rec["iterate_minus_prior_translation_mm_median"] = {
            k: 1000 * float(boxminus(x, x_prior)[torch.as_tensor(v, dtype=torch.long, device=x.mu.device), :3]
                            .norm(dim=1).median()) for k, v in self.sets.items()}
        self.conflict["iteration_0" if iteration == 0 else "last_iteration"] = rec
        if iteration != 0:
            return
        if self.mediation:
            G = (lambda v: torch.cat([grouped.expand(v[:, :6].float()), grouped.expand(v[:, 6:].float())], 1)) \
                if grouped is not None else None
            GT = (lambda v: torch.cat([grouped.reduce(v[:, :6].float().contiguous()),
                                       grouped.reduce(v[:, 6:].float().contiguous())], 1)) if grouped is not None else None
            systems = {}
            if grouped is not None:
                systems["grouped"] = (wa.FrozenSystem(lin, terms, Lambda_prior, b, meas_normal, float(lm), grouped,
                                                      group_precond(M_blocks, grouped, G, GT)), self.max_iters)
            systems["ungrouped"] = (wa.FrozenSystem(lin, terms, Lambda_prior, b, meas_normal, float(lm), None,
                                                    block_jacobi_preconditioner(M_blocks)), self.max_iters_ungrouped)
            n = lin.num_gaussians
            own = torch.zeros(n, dtype=torch.bool)
            own[torch.as_tensor(self.sets["runaway"], dtype=torch.long)] = True
            mates = (torch.isin(sid, torch.unique(sid[own])) & ~own) if sid is not None else torch.zeros_like(own)
            parts = {"own_runaway_rows": own, "groupmates": mates, "rest_of_scene": ~(own | mates)}
            self.group_mediation = group_mediation(systems, lin, terms, self.names, parts, self.sets, tol=self.tol)
            self.group_mediation["row_counts"] = {k: int(v.sum()) for k, v in parts.items()}
        if self.nonlinear:
            def render_terms(state):
                L = build_linearization(state, cameras, background=background, eps_depth=cfg.eps_depth,
                                        antialiasing=cfg.antialiasing, debug=cfg.debug)
                return _frame_terms(L.views, rgbd_gt, cfg, visibility, observation_model)
            self.attribution = nonlinear_attribution(x_prior, post, Lambda_prior, render_terms, self.names, sid,
                                                     self.sets, top_groups=self.top_groups)
