"""Phase 4A: classify the ORIGINAL Gaussian population under the fixed Phase-3 joint (spec section 4).

The Phase-2 core is a guaranteed moving seed; its complement is NOT static truth. Every other original
Gaussian is grouped into deterministic spatial components, and each component B is tested by two
full-scene counterfactuals with everything else in place:

    E_h(B) = sum_{c,f} L_cf[ R(G_-B  U  B_h) ],   h in {S, M},   Delta_B = E_S(B) - E_M(B)

with the complete physical joint transform (positions AND orientations), full RGB-D-alpha compositing and
the rgbd_loss objective. Two statistics per view, both kept:
  * affected-pixel score: sum of the per-pixel loss difference over the pixels the flip changes
    (rgbd_loss.pixel_loss_map) -- the informative-view evidence a small surface needs
  * full-view score: the FreeArt view loss (with SSIM) difference -- the all-view contradiction audit
Screening uses the views where B's S and M placements differ by >= min_disp_px; ACCEPTANCE re-scores every
camera and every original frame of the Phase-3 window.

Decisions (declared policy, PolicyConfig; nothing here is a calibrated probability):
  ACCEPTED_MOVING / ACCEPTED_STATIC  decisive, sign-consistent comparative evidence on screening views, AND
                                     the all-view re-score agrees in sign
  UNRESOLVED                         no informative view (axis-near, never moved in view, never visible)
                                     or indecisive / inconsistent evidence; keeps the provisional baseline
  SUSPECT_ORIGINAL                   undecided AND omitting B fits better than both hypotheses (flag only)
  UPSTREAM_JOINT_INVALID             the core itself is not better explained moving than static: Phase 4
                                     aborts (the joint is never softened by scaling q)
Undecided components do not become static by default: their provisional label is the upstream baseline and
their status says UNRESOLVED. Large ambiguous components are bisected (longest axis, median) down to
min_component before they are declared UNRESOLVED, so a component that straddles a hinge splits instead
of being smoothed across it.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .articulated_scene import JointTransport, canonical_tensors, posed, render_articulated
from .contracts import rot_exp
from .gaussians import MOVING, STATIC
from .rgbd_loss import LossConfig, pixel_loss_map, view_loss

ACCEPTED_MOVING, ACCEPTED_STATIC, UNRESOLVED = "ACCEPTED_MOVING", "ACCEPTED_STATIC", "UNRESOLVED"
SUSPECT_ORIGINAL, UPSTREAM_JOINT_INVALID, CORE = "SUSPECT_ORIGINAL", "UPSTREAM_JOINT_INVALID", "CORE_MOVING"
STATUS_CODES = {CORE: 0, ACCEPTED_MOVING: 1, ACCEPTED_STATIC: 2, UNRESOLVED: 3, SUSPECT_ORIGINAL: 4,
                UPSTREAM_JOINT_INVALID: 5}


@dataclass
class PolicyConfig:
    component_voxel: float = 0.05          # m; hard cap on the link distance between two Gaussians
    link_factor: float = 1.0               # link i, j when |x_i - x_j| <= link_factor (s_max,i + s_max,j)
    max_component: int = 256               # larger components are bisected before testing
    min_component: int = 4                 # ambiguous components are not split below this
    normal_cos: float = 0.8                # surface elements link only if |n_i . n_j| >= this
    layer_tol: float = 0.008               # ... and each within this of the other tangent plane (m)
    flat_max: float = 0.5                  # smallest / middle scale below this = surface element
    min_disp_px: float = 0.5               # a view is informative for B if S and M placements differ this much
    min_affected_pixels: int = 4           # ... and the flip changes at least this many pixels
    decisive_rel: float = 0.25             # |sum Delta_aff| / sum |Delta_aff_v| needed (sign consistency)
    decisive_abs: float = 1e-3             # minimal |sum Delta_aff| (loss units x pixels)
    decisive_per_px: float = 0.0           # minimal |sum Delta_aff| per affected pixel-view (weak, neighbour-
                                           # dependent evidence stays UNRESOLVED instead of becoming a decision)
    favour_frac: float = 0.6               # fraction of informative views that must agree in sign
    omit_margin: float = 0.05              # relative margin for the SUSPECT_ORIGINAL flag
    max_rounds: int = 8
    verify_passes: int = 3                 # final all-decided re-test passes against the final labelling
    max_flips: int = 3                     # a component flipped this often is declared UNRESOLVED (oscillation)
    max_tests_per_round: int = 4000        # cap, logged when it binds
    interaction_radius: float = 0.05       # m; re-test components near a flipped one
    core_check_views: int = 64             # views used for the upstream-joint validity check
    screen_max_views: int = 48             # screening uses at most this many informative views (evenly spaced
                                           # in time, largest displacement kept); acceptance re-scores ALL views
    sensitivity: bool = True
    sens_axis_deg: float = 1.0
    sens_point_m: float = 0.005
    sens_q_rel: float = 0.02
    sens_max_views: int = 24

    def to_dict(self):
        return asdict(self)


@dataclass
class ViewRef:
    cam: object            # views.RenderCamera
    frame: int
    get: object            # callable -> views.View (CPU)


@dataclass
class Component:
    cid: int
    members: np.ndarray
    parent: int = -1
    status: str = UNRESOLVED
    reason: str = "not tested"
    target: bool = False
    scores: dict = field(default_factory=dict)


# ------------------------------------------------------------------ components
def surface_normals(rot: np.ndarray, scale: np.ndarray):
    """Normal = the axis of the smallest scale; `flat` = the Gaussian is a surface element
    (smallest scale <= flat_ratio x middle scale), otherwise it has no usable normal."""
    from .raster_oracle import quat_to_rot
    R = quat_to_rot(torch.as_tensor(rot, dtype=torch.float64)).numpy()     # columns are the local axes
    k = np.argmin(scale, 1)
    n = R[np.arange(len(R)), :, k]
    s = np.sort(scale, 1)
    return n / np.linalg.norm(n, axis=1, keepdims=True), s[:, 0] / np.maximum(s[:, 1], 1e-12)


def build_components(xyz: np.ndarray, candidates: np.ndarray, radius: float, max_size: int, normals=None,
                     flat_ratio=None, normal_cos: float = 0.8, flat_max: float = 0.5, extent=None,
                     link_factor: float = 1.0, layer_tol: float = 0.008) -> list:
    """Connected components of the overlap graph over the candidate Gaussians, bisected until each has
    <= max_size members. i and j are linked when |x_i - x_j| <= link_factor (extent_i + extent_j) (their
    footprints touch; extent = largest scale) and <= `radius` (a hard cap). Layer-aware: two surface
    elements (both flat) are linked only if their normals agree (|n_i . n_j| >= normal_cos) AND each lies
    within layer_tol of the other's tangent plane, so a component never runs across a hinge or a part edge
    where the surfaces turn, nor across two parallel layers (a drawer side 2 cm inside the body wall, the
    two faces of a thin panel). Deterministic."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pts = xyz[candidates]
    pairs = np.array(sorted(cKDTree(pts).query_pairs(radius)), dtype=np.int64).reshape(-1, 2)
    if extent is not None and len(pairs):
        e = extent[candidates]
        d = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
        pairs = pairs[d <= link_factor * (e[pairs[:, 0]] + e[pairs[:, 1]])]
    if normals is not None and len(pairs):
        n, fr = normals[candidates], flat_ratio[candidates]
        both_flat = (fr[pairs[:, 0]] <= flat_max) & (fr[pairs[:, 1]] <= flat_max)
        agree = np.abs((n[pairs[:, 0]] * n[pairs[:, 1]]).sum(1)) >= normal_cos
        dv = pts[pairs[:, 1]] - pts[pairs[:, 0]]
        same_layer = (np.abs((n[pairs[:, 0]] * dv).sum(1)) <= layer_tol) & \
                     (np.abs((n[pairs[:, 1]] * dv).sum(1)) <= layer_tol)
        pairs = pairs[~both_flat | (agree & same_layer)]
    m = len(pts)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m)) if len(pairs) else \
        coo_matrix((m, m))
    _, lab = connected_components(g, directed=False)
    comps = []
    for r in np.unique(lab):
        comps.extend(_bisect(xyz, candidates[lab == r], max_size))
    return comps


def _bisect(xyz, members, max_size):
    if len(members) <= max_size:
        return [np.sort(members)]
    p = xyz[members]
    ax = int(np.argmax(p.max(0) - p.min(0)))
    order = members[np.argsort(p[:, ax], kind="stable")]
    h = len(order) // 2
    return _bisect(xyz, order[:h], max_size) + _bisect(xyz, order[h:], max_size)


def split_component(xyz, members):
    p = xyz[members]
    ax = int(np.argmax(p.max(0) - p.min(0)))
    order = members[np.argsort(p[:, ax], kind="stable")]
    h = len(order) // 2
    return [np.sort(order[:h]), np.sort(order[h:])]


# ------------------------------------------------------------------ rendering helpers
class Phase4Scene:
    """Canonical model + fixed joint + views; renders the full scene under a material assignment."""

    def __init__(self, gaussians, joint, views, loss_cfg: LossConfig, bg=(1.0, 1.0, 1.0), sh_mode="local_SH",
                 sh_degree=0, device="cuda"):
        self.g = gaussians
        self.joint = joint
        self.transport = JointTransport(joint, device)
        self.views = views
        self.loss_cfg = loss_cfg
        self.bg = torch.tensor(bg, device=device, dtype=torch.float32)
        self.sh_mode, self.sh_degree, self.device = sh_mode, int(sh_degree), device
        self._tensors = None

    def tensors(self):
        if self._tensors is None:
            self._tensors = {k: v.detach() for k, v in canonical_tensors(self.g).items()}
        return self._tensors

    def invalidate(self):
        self._tensors = None

    def render(self, vr: ViewRef, moving: torch.Tensor, keep: torch.Tensor | None = None, joint_transport=None):
        t = self.tensors()
        tr = joint_transport or self.transport
        if keep is not None:
            t = {k: v[keep] for k, v in t.items()}
            moving = moving[keep]
        with torch.no_grad():
            return render_articulated(vr.cam, tr.at(vr.frame), None, moving, self.bg, self.sh_mode, self.sh_degree,
                                      tensors=t)

    def view(self, vr: ViewRef):
        return vr.get().to(self.device)


def project_px(cam, X: torch.Tensor):
    """Pixel coordinates and camera depth of world points (3DGS conventions)."""
    h = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    z = (h @ cam.V.to(X.device))[:, 2]
    ph = h @ cam.P.to(X.device)
    ndc = ph[:, :2] / (ph[:, 3:] + 1e-7)
    u = ((ndc[:, 0] + 1) * cam.W - 1) * 0.5
    v = ((ndc[:, 1] + 1) * cam.H - 1) * 0.5
    return u, v, z


def screening(iv, cap):
    """Deterministic subset of informative views for screening: evenly spaced in view order, always keeping
    the view of largest displacement. Acceptance never uses this subset alone."""
    if len(iv) <= cap:
        return iv
    keep = {int(np.argmax([d for _, d in iv]))}
    keep |= set(np.linspace(0, len(iv) - 1, cap - 1).round().astype(int).tolist())
    return [iv[k] for k in sorted(keep)]


def informative_views(scene: Phase4Scene, members: np.ndarray, min_disp_px: float):
    """Views where B's static and moving placements differ by >= min_disp_px and B is in front of the camera
    and inside the image in at least one hypothesis. Returns [(view index, max displacement px)]."""
    t = scene.tensors()
    idx = torch.as_tensor(members, device=scene.device)
    X = t["xyz"][idx]
    out = []
    for i, vr in enumerate(scene.views):
        ft = scene.transport.at(vr.frame)
        Xm = X @ ft.Q.T + ft.p
        u0, v0, z0 = project_px(vr.cam, X)
        u1, v1, z1 = project_px(vr.cam, Xm)
        ok0 = (z0 > 0.2) & (u0 > -0.5) & (u0 < vr.cam.W) & (v0 > -0.5) & (v0 < vr.cam.H)
        ok1 = (z1 > 0.2) & (u1 > -0.5) & (u1 < vr.cam.W) & (v1 > -0.5) & (v1 < vr.cam.H)
        if not bool((ok0 | ok1).any()):
            continue
        d = torch.sqrt((u1 - u0) ** 2 + (v1 - v0) ** 2)
        dm = float(d[ok0 | ok1].max())
        if dm >= min_disp_px:
            out.append((i, dm))
    return out


# ------------------------------------------------------------------ the counterfactual test
class BaselineCache:
    """Per-view baseline render statistics under the current labelling (CPU)."""

    def __init__(self, scene: Phase4Scene, moving: torch.Tensor):
        self.scene, self.moving = scene, moving.clone()
        self.pix, self.full = {}, {}

    def get(self, i):
        if i not in self.pix:
            vr = self.scene.views[i]
            v = self.scene.view(vr)
            out = self.scene.render(vr, self.moving)
            self.pix[i] = pixel_loss_map(out, v, self.scene.loss_cfg).cpu()
            self.full[i] = float(view_loss(out, v, self.scene.loss_cfg)["total"])
        return self.pix[i], self.full[i]


def omit_check(scene, base, members, view_ids, dec, st, pol):
    """The decided hypothesis must also beat OMIT (B removed). Evidence that only says 'B is better
    elsewhere' is removal evidence, not motion evidence (e.g. a splat on a binary silhouette edge)."""
    o = flip_scores(scene, base, members, view_ids, omit=True)
    e_omit = sum(r["delta_aff"] for r in o)                      # E_OMIT - E_baseline, affected pixels
    base_moving = bool(base.moving[int(members[0])])
    s = st.get("sum_delta_aff", 0.0)                             # E_S - E_M
    e_S, e_M = (s, 0.0) if base_moving else (0.0, -s)            # each relative to the baseline
    e_dec = e_M if dec == ACCEPTED_MOVING else e_S
    st.update(omit_minus_baseline_aff=e_omit, static_minus_baseline_aff=e_S, moving_minus_baseline_aff=e_M)
    margin = pol.omit_margin * max(1e-12, st.get("sum_abs_delta_aff", 0.0))
    return e_dec < e_omit - margin, e_omit, min(e_S, e_M)


def flip_scores(scene: Phase4Scene, base: BaselineCache, members: np.ndarray, view_ids, omit=False, full=False):
    """Per-view (Delta_aff, Delta_full, n_affected) for flipping B's label (S <-> M) relative to the baseline,
    signed so that POSITIVE favours MOVING. With omit=True also E_OMIT - E_baseline."""
    idx = torch.as_tensor(members, device=scene.device)
    moving_b = bool(base.moving[idx[0]])
    alt = base.moving.clone()
    alt[idx] = not moving_b
    keep = None
    if omit:
        keep = torch.ones_like(base.moving)
        keep[idx] = False
    rows = []
    for i in view_ids:
        vr = scene.views[i]
        v = scene.view(vr)
        pb, fb = base.get(i)
        out = scene.render(vr, alt) if not omit else scene.render(vr, base.moving, keep=keep)
        pa = pixel_loss_map(out, v, scene.loss_cfg).cpu()
        dpix = pa - pb
        aff = dpix.abs() > 1e-7
        d_aff = float(dpix[aff].sum())
        d_full = (float(view_loss(out, v, scene.loss_cfg)["total"]) - fb) if full else 0.0
        if not omit and not moving_b:            # baseline static, alternative moving: E_S - E_M = base - alt
            d_aff, d_full = -d_aff, -d_full
        rows.append({"view": int(i), "camera": vr.cam.name, "frame": int(vr.frame), "delta_aff": d_aff,
                     "delta_full": d_full, "affected_px": int(aff.sum())})
    return rows


def decide(rows, pol: PolicyConfig):
    inf = [r for r in rows if r["affected_px"] >= pol.min_affected_pixels]
    if not inf:
        return UNRESOLVED, "no informative view (flip changes no pixel)", {}
    s = sum(r["delta_aff"] for r in inf)
    a = sum(abs(r["delta_aff"]) for r in inf)
    fav_m = sum(r["delta_aff"] > 0 for r in inf) / len(inf)
    npx = sum(r["affected_px"] for r in inf)
    stats = {"sum_delta_aff": s, "sum_abs_delta_aff": a, "favour_moving_frac": fav_m, "informative_views": len(inf),
             "sum_delta_full": sum(r["delta_full"] for r in inf), "affected_pixel_views": npx}
    if abs(s) < pol.decisive_abs or abs(s) < pol.decisive_rel * a or abs(s) < pol.decisive_per_px * npx:
        return UNRESOLVED, "indecisive comparative evidence", stats
    if s > 0 and fav_m >= pol.favour_frac:
        return ACCEPTED_MOVING, "moving explains the informative views", stats
    if s < 0 and (1 - fav_m) >= pol.favour_frac:
        return ACCEPTED_STATIC, "static explains the informative views", stats
    return UNRESOLVED, "views disagree in sign", stats


# ------------------------------------------------------------------ joint sensitivity
def perturbed_joints(joint, pol: PolicyConfig):
    """Declared shared perturbations of the WHOLE joint (one joint per draw, never per Gaussian)."""
    from dataclasses import replace
    e = np.asarray(joint.axis)
    u = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
    w = np.cross(e, u)
    out = []
    for name, b in (("axis+u", u), ("axis-u", -u), ("axis+w", w), ("axis-w", -w)):
        e2 = rot_exp(np.cross(e, b) / np.linalg.norm(np.cross(e, b)), np.radians(pol.sens_axis_deg)) @ e
        out.append((name, replace(joint, axis=tuple(e2 / np.linalg.norm(e2)))))
    if joint.type == "revolute":
        a = np.asarray(joint.point)
        for name, b in (("point+u", u), ("point-u", -u), ("point+w", w), ("point-w", -w)):
            out.append((name, replace(joint, point=tuple(a + pol.sens_point_m * b))))
    for name, s in (("q*(1+r)", 1 + pol.sens_q_rel), ("q*(1-r)", 1 - pol.sens_q_rel)):
        out.append((name, replace(joint, q=tuple(float(x) * s for x in joint.q))))
    return out


def sensitivity(scene: Phase4Scene, moving: torch.Tensor, members, view_ids, pol: PolicyConfig):
    view_ids = list(view_ids)[: pol.sens_max_views]
    res = {}
    orig = scene.transport
    for name, j in perturbed_joints(scene.joint, pol):
        scene.transport = JointTransport(j, scene.device)
        try:
            rows = flip_scores(scene, BaselineCache(scene, moving), members, view_ids)
        finally:
            scene.transport = orig
        res[name] = sum(r["delta_aff"] for r in rows)
    return res


# ------------------------------------------------------------------ the stage
def run_phase4a(scene: Phase4Scene, core: np.ndarray, pol: PolicyConfig, log=print) -> dict:
    t0 = time.time()
    N = len(scene.g)
    xyz = scene.tensors()["xyz"].cpu().numpy()
    moving = torch.zeros(N, dtype=torch.bool, device=scene.device)
    moving[torch.as_tensor(core, device=scene.device)] = True
    status = np.full(N, UNRESOLVED, dtype=object)
    status[core] = CORE
    provisional = moving.cpu().numpy().astype(np.int8)          # upstream baseline: core moving, rest static
    # 1. the upstream joint must explain its own core better moving than static
    core_views = [i for i, _ in sorted(informative_views(scene, core, pol.min_disp_px), key=lambda x: -x[1])]
    core_views = core_views[: pol.core_check_views]
    base = BaselineCache(scene, moving)
    core_rows = flip_scores(scene, base, core, core_views)
    core_dec, core_why, core_stats = decide(core_rows, pol)
    audit = {"core_check": {"decision": core_dec, "reason": core_why, **core_stats, "views": len(core_views)}}
    log(f"  4A core check: {core_dec} ({core_why}); sum Delta_aff {core_stats.get('sum_delta_aff', 0):.4g} "
        f"over {len(core_views)} views")
    if core_dec != ACCEPTED_MOVING:
        status[:] = UPSTREAM_JOINT_INVALID
        return dict(status=status, material=provisional, provisional=provisional, components=[], audit=audit,
                    aborted=True, sensitivity={}, seconds=time.time() - t0)
    # 2. components of the non-core originals
    cand = np.setdiff1d(np.arange(N), core)
    tt = scene.tensors()
    normals, flat = surface_normals(tt["rot"].cpu().numpy(), tt["scale"].cpu().numpy())
    comps = [Component(i, m) for i, m in enumerate(build_components(
        xyz, cand, pol.component_voxel, pol.max_component, normals, flat, pol.normal_cos, pol.flat_max,
        extent=tt["scale"].cpu().numpy().max(1), link_factor=pol.link_factor, layer_tol=pol.layer_tol))]
    log(f"  4A {len(cand)} candidate Gaussians in {len(comps)} components")
    pending = list(range(len(comps)))
    next_id = len(comps)
    flip_count, flips = {}, []
    for rnd in range(pol.max_rounds):
        base = BaselineCache(scene, moving)
        flips, tested = [], 0
        queue = []
        for ci in pending:
            c = comps[ci]
            iv = informative_views(scene, c.members, pol.min_disp_px)
            if not iv:
                c.status, c.reason = UNRESOLVED, "no informative view (no visible displacement between S and M)"
                continue
            queue.append((ci, iv))
        queue.sort(key=lambda x: -max(d for _, d in x[1]))
        if len(queue) > pol.max_tests_per_round:
            log(f"  4A round {rnd}: {len(queue) - pol.max_tests_per_round} components over the per-round cap "
                f"stay UNRESOLVED this round")
            for ci, _ in queue[pol.max_tests_per_round:]:
                comps[ci].status, comps[ci].reason = UNRESOLVED, "over the per-round test cap"
            queue = queue[: pol.max_tests_per_round]
        new_pending = []
        while queue:
            ci, iv = queue.pop(0)
            c = comps[ci]
            tested += 1
            iv = screening(iv, pol.screen_max_views)
            rows = flip_scores(scene, base, c.members, [i for i, _ in iv])
            dec, why, st = decide(rows, pol)
            if dec == UNRESOLVED and len(c.members) > pol.min_component and st:
                for half in split_component(xyz, c.members):          # bisect ambiguous components
                    comps.append(Component(next_id, half, parent=c.cid))
                    hv = informative_views(scene, half, pol.min_disp_px)
                    if hv:
                        queue.append((next_id, hv))
                    else:
                        comps[-1].status, comps[-1].reason = UNRESOLVED, "no informative view"
                    next_id += 1
                c.status, c.reason, c.scores = "SPLIT", f"bisected: {why}", st
                continue
            if dec in (ACCEPTED_MOVING, ACCEPTED_STATIC) and len(c.members) > pol.min_component:
                # both halves must agree on their own; otherwise the component spans two materials
                halves = split_component(xyz, c.members)
                hdec = []
                for half in halves:
                    hv = screening(informative_views(scene, half, pol.min_disp_px), pol.screen_max_views)
                    hdec.append(decide(flip_scores(scene, base, half, [i for i, _ in hv]), pol)[0] if hv else UNRESOLVED)
                if any(h != dec for h in hdec):
                    for half in halves:
                        comps.append(Component(next_id, half, parent=c.cid))
                        hv = informative_views(scene, half, pol.min_disp_px)
                        if hv:
                            queue.append((next_id, hv))
                        else:
                            comps[-1].status, comps[-1].reason = UNRESOLVED, "no informative view"
                        next_id += 1
                    c.status, c.reason, c.scores = "SPLIT", f"halves disagree with {dec}: {hdec}", st
                    continue
            if dec in (ACCEPTED_MOVING, ACCEPTED_STATIC):
                ok, e_omit, e_best = omit_check(scene, base, c.members, [i for i, _ in iv], dec, st, pol)
                if not ok:
                    dec, why = (SUSPECT_ORIGINAL, "omitting B fits at least as well as the decided hypothesis") \
                        if e_omit < e_best else (UNRESOLVED, "decided hypothesis does not beat OMIT by the margin")
            if dec in (ACCEPTED_MOVING, ACCEPTED_STATIC):
                # acceptance: re-score EVERY camera and original frame (all-view contradiction audit)
                all_rows = flip_scores(scene, base, c.members, range(len(scene.views)), full=True)
                tot_aff = sum(r["delta_aff"] for r in all_rows)
                tot_full = sum(r["delta_full"] for r in all_rows)
                st.update(all_view_delta_aff=tot_aff, all_view_delta_full=tot_full, all_views=len(all_rows))
                if (dec == ACCEPTED_MOVING) != (tot_aff > 0):
                    dec, why = UNRESOLVED, "all-view re-score contradicts the informative views"
            if dec == UNRESOLVED and st and "omit_minus_baseline_aff" not in st:
                o = flip_scores(scene, base, c.members, [i for i, _ in iv], omit=True)
                e_omit = sum(r["delta_aff"] for r in o)           # E_OMIT - E_baseline (affected pixels)
                st["omit_minus_baseline_aff"] = e_omit
                e_alt = -st["sum_delta_aff"] if not bool(base.moving[c.members[0]]) else st["sum_delta_aff"]
                if e_omit < -pol.omit_margin * max(1e-12, st["sum_abs_delta_aff"]) and e_omit < e_alt:
                    dec, why = SUSPECT_ORIGINAL, "omitting B fits better than both S and M"
            c.status, c.reason, c.scores = dec, why, st
            if dec in (ACCEPTED_MOVING, ACCEPTED_STATIC):
                want_moving = dec == ACCEPTED_MOVING
            else:                      # undecided: back to the provisional upstream baseline (never kept flipped)
                want_moving = bool(provisional[c.members[0]])
            if bool(moving[c.members[0]]) != want_moving:
                c.target = want_moving
                flips.append(ci)
        for ci in flips:
            c = comps[ci]
            moving[torch.as_tensor(c.members, device=scene.device)] = c.target
            flip_count[ci] = flip_count.get(ci, 0) + 1
        log(f"  4A round {rnd}: tested {tested}, flipped {len(flips)}")
        if not flips:
            break
        # Every decision above was taken against the round's baseline. Re-test, against the new baseline,
        # every flipped component and every leaf component near a flip (decided or not): a component that
        # only "helped" because a neighbour was still mislabelled loses that support now.
        flipped_pts = np.concatenate([xyz[comps[ci].members] for ci in flips])
        from scipy.spatial import cKDTree
        tree = cKDTree(flipped_pts)
        for c in comps:
            if c.status in ("SPLIT", CORE) or c.reason.startswith("no informative view"):
                continue
            if flip_count.get(c.cid, 0) >= pol.max_flips:
                c.status, c.reason = UNRESOLVED, f"label oscillated {flip_count[c.cid]} times"
                moving[torch.as_tensor(c.members, device=scene.device)] = bool(provisional[c.members[0]])
                continue
            if c.cid in flips or np.isfinite(tree.query(xyz[c.members],
                                                        distance_upper_bound=pol.interaction_radius)[0]).any():
                new_pending.append(c.cid)
        pending = new_pending
        if not pending:
            break
    else:
        # rounds exhausted with flips still happening: the last flips were never re-verified
        for ci in flips:
            c = comps[ci]
            c.status, c.reason = UNRESOLVED, "flipped in the last round, never re-verified (max_rounds)"
            moving[torch.as_tensor(c.members, device=scene.device)] = bool(provisional[c.members[0]])
    # 2b. final verification: every decided leaf against the FINAL labelling (decisions taken earlier may
    #     have relied on neighbours that were still mislabelled); repeat while labels change, bounded
    for vpass in range(pol.verify_passes):
        base = BaselineCache(scene, moving)
        changed = []
        for c in comps:
            if c.status not in (ACCEPTED_MOVING, ACCEPTED_STATIC, UNRESOLVED) or c.reason.startswith("no informative") \
                    or c.reason.startswith("label oscillated"):
                continue
            iv = screening(informative_views(scene, c.members, pol.min_disp_px), pol.screen_max_views)
            if not iv:
                continue
            dec, why, st = decide(flip_scores(scene, base, c.members, [i for i, _ in iv]), pol)
            if dec in (ACCEPTED_MOVING, ACCEPTED_STATIC):
                ok, e_omit, e_best = omit_check(scene, base, c.members, [i for i, _ in iv], dec, st, pol)
                if not ok:
                    dec, why = UNRESOLVED, "verification: does not beat OMIT"
            want = (dec == ACCEPTED_MOVING) if dec in (ACCEPTED_MOVING, ACCEPTED_STATIC) else bool(provisional[c.members[0]])
            if dec != c.status:
                c.status, c.reason, c.scores = dec, f"final verification: {why}", st
            if bool(moving[c.members[0]]) != want:
                changed.append((c, want))
        for c, want in changed:
            moving[torch.as_tensor(c.members, device=scene.device)] = want
        log(f"  4A verification pass {vpass}: {len(changed)} label changes")
        if not changed:
            break
    # 3. statuses, provisional labels, sensitivity of the decided components
    sens = {}
    leaf = [c for c in comps if c.status != "SPLIT"]
    for c in leaf:
        status[c.members] = c.status
    material = moving.cpu().numpy().astype(np.int8)
    provisional = material.copy()
    if pol.sensitivity:
        for c in leaf:
            if c.status in (ACCEPTED_MOVING, ACCEPTED_STATIC):
                iv = [i for i, _ in sorted(informative_views(scene, c.members, pol.min_disp_px), key=lambda x: -x[1])]
                s = sensitivity(scene, moving, c.members, iv, pol)
                sens[str(c.cid)] = {"status": c.status, "delta_aff_nominal": c.scores.get("sum_delta_aff"),
                                    "delta_aff_perturbed": s, "min": min(s.values()), "max": max(s.values()),
                                    "sign_stable": bool(all(np.sign(v) == np.sign(c.scores.get("sum_delta_aff", 0))
                                                            for v in s.values()))}
    return dict(status=status, material=material, provisional=provisional, components=comps, audit=audit,
                aborted=False, sensitivity=sens, seconds=time.time() - t0)


def visibility_counts(scene: Phase4Scene, moving: torch.Tensor):
    """Per Gaussian: number of views in which it is rasterized (radii > 0) under the final labelling."""
    cnt = torch.zeros(len(scene.g), dtype=torch.int32, device=scene.device)
    for vr in scene.views:
        cnt += (scene.render(vr, moving)["radii"] > 0).int()
    return cnt.cpu().numpy()


def write_outputs(out_dir: Path, scene: Phase4Scene, res: dict, pol: PolicyConfig, upstream_manifest: dict):
    d = Path(out_dir) / "support_completion"
    d.mkdir(parents=True, exist_ok=True)
    N = len(scene.g)
    np.savez(d / "original_ids.npz", original_id=np.arange(N, dtype=np.int64))
    codes = np.array([STATUS_CODES[s] for s in res["status"]], np.int8)
    np.savez(d / "material_status.npz", status_code=codes, legend=json.dumps(STATUS_CODES))
    np.savez(d / "material_provisional.npz", material=res["provisional"],
             legend=json.dumps({"0": "STATIC", "1": "MOVING"}),
             note="UNRESOLVED rows keep the upstream baseline here and stay UNRESOLVED in material_status")
    comps = [{"cid": c.cid, "parent": c.parent, "size": int(len(c.members)), "status": c.status, "reason": c.reason,
              "members": c.members.tolist(), "scores": c.scores} for c in res["components"]]
    (d / "components.json").write_text(json.dumps(comps))
    (d / "full_scene_scores.json").write_text(json.dumps(
        {"policy": pol.to_dict(), "loss": scene.loss_cfg.to_dict(), "core_check": res["audit"]["core_check"],
         "counts": {s: int((res["status"] == s).sum()) for s in STATUS_CODES},
         "statistic": "Delta = E_S - E_M (positive favours MOVING); affected-pixel and full-view, per component",
         "validation": "no parameters are fitted in 4A; all views compare two fixed hypotheses, conditional on "
                       "the upstream joint (Phase 3 used the same frames)"}, indent=1))
    (d / "joint_sensitivity.json").write_text(json.dumps(
        {"joint_confidence": "JOINT_CONFIDENCE_UNCALIBRATED",
         "note": "Phase-3 posterior_sd_q is about q only; declared shared perturbations, ranges only, no prob_moving",
         "perturbations": {"axis_deg": pol.sens_axis_deg, "point_m": pol.sens_point_m, "q_rel": pol.sens_q_rel},
         "components": res["sensitivity"]}, indent=1))
    mv = torch.as_tensor(res["material"] == MOVING, device=scene.device)
    np.savez(d / "visibility.npz", views_rasterized=visibility_counts(scene, mv))
    (d / "full_scene_audit.json").write_text(json.dumps(
        {"upstream": upstream_manifest, "aborted": res["aborted"], "seconds": res["seconds"],
         "views": [{"camera": v.cam.name, "frame": int(v.frame)} for v in scene.views],
         "num_gaussians": N, "num_moving_final": int((res["material"] == MOVING).sum()),
         "num_moving_core": int((res["status"] == CORE).sum())}, indent=1, default=str))
    return d
