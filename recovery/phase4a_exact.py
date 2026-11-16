"""Phase 4A, corrected: exact conditional static-vs-moving model selection of the ORIGINAL Gaussians
(directive 2026-11-11, section 4).

Hypotheses for a group B of original Gaussians, everything else fixed (other labels, shapes, SH, opacity,
the Phase-3 joint and every q_t):

    G_S(B) = G_-B u B_static      G_M(B) = G_-B u B_moving      G_omit(B) = G_-B   (diagnostic only)
    E_h(B) = sum over ALL (camera, original frame) views of L_v(G_h(B))        (exact_objective.L_v)
    Delta_B = E_S(B) - E_M(B)          positive favours MOVING

Exactness (no approximation in any decision): for each view the flipped scene is rendered in full by the
production RGB-D kernel; L_v(alt) - L_v(base) is accumulated only over the conservative tile footprint of B
in both poses dilated by the SSIM radius (footprint.py, exact_objective.box_delta). Pixels outside are
bitwise identical (asserted by tests/test_phase4_exact_roi_parity.py), so the sum is the full-image
difference to float64 rounding. Views are skipped ONLY when provably zero-impact: B culled / off-screen in
both poses (kernel radius 0 everywhere) or an identity joint transform at that frame (S and M coincide).

Work is VIEW-MAJOR: each view is uploaded once per sweep, its baseline rendered once, and every group in the
sweep is scored against it. Per-group results are small score vectors kept on the GPU and copied once per
sweep.

Schedule (deterministic, monotone):
  1. core check -- the Phase-2 core as one group; it must be decided MOVING, else UPSTREAM_JOINT_INVALID
  2. layer-aware groups of the non-core originals; one sweep scores every group on every view
  3. proposals = decisive groups whose decision differs from their current render label, ranked by |Delta|.
     A batch commits the largest NON-INTERACTING subset (footprints, dilated one tile, disjoint in every
     view), so their exact gains add. The objective drops by exactly sum |Delta| (checked against the next
     sweep's re-rendered baseline). Only groups whose footprint meets a committed footprint, plus
     uncommitted proposals, are re-scored (all other scores are unchanged by construction).
  4. groups left UNRESOLVED with sign-inconsistent evidence are bisected once per level into probe halves;
     a split is kept only when both halves are decisive with OPPOSITE labels (separable evidence)
  5. final verification: every leaf group re-scored on every view against the final labelling, with the
     OMIT diagnostic; final statuses come from this sweep; its baselines are the global scene audit
  6. joint sensitivity: globally coherent perturbations of the whole joint (every moving Gaussian moves),
     on the scene objective and a small declared set of the least decisive groups -> ranges only
  7. core audit -- the Phase-3 support, split into the same layer-aware groups, re-scored on every view.
     core_policy "revisable" (default, user decision 2026-11-12): the support is a hypothesis Phase 4 exists to
     correct. If any support group decisively prefers STATIC, the audit groups become ordinary leaves, the same
     exact monotone descent commits the flips, and a second final verification re-decides every group against
     the corrected labelling. Relabelled support rows end STATIC_CONFIRMED (recorded in core_audit with
     relabelled_static); the rest keep CORE_MOVING_ANCHOR. The joint stays gated by step 1 (the support as a
     whole must be MOVING). core_policy "gate": the old rule (decisive STATIC = UPSTREAM_JOINT_OR_SUPPORT_CONFLICT,
     4A incomplete).
Statuses (unified spec 3.3): CORE_MOVING_ANCHOR, MOVING_CONFIRMED, STATIC_CONFIRMED, UNRESOLVED_UNOBSERVABLE (S and M
renders differ by less than the declared magnitude in every view, e.g. on the hinge), UNRESOLVED_AMBIGUOUS,
SUSPECT_ORIGINAL_GEOMETRY, UPSTREAM_JOINT_OR_SUPPORT_CONFLICT, UNTESTED_RESOURCE_LIMIT. The render label of an UNRESOLVED group (its provisional pose) is stored separately
(`render_label`) and is never reported as a decision. Decision thresholds are a DECLARED provisional policy,
not calibrated probabilities.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import footprint as FP
from .articulated_scene import JointTransport
from .contracts import rot_exp
from .exact_objective import CHANNELS, SSIM_R, ExactLossConfig, ViewTargets, box_delta
from .material_completion import build_components, split_component, surface_normals
from .runlog import atomic_json

# statuses of the unified spec (section 3.3); evidence status and render material are stored separately
CORE, MOVING_S, STATIC_S = "CORE_MOVING_ANCHOR", "MOVING_CONFIRMED", "STATIC_CONFIRMED"
UNRESOLVED, UNOBS = "UNRESOLVED_AMBIGUOUS", "UNRESOLVED_UNOBSERVABLE"
UNTESTED, SUSPECT, JOINT_INVALID, SPLIT = ("UNTESTED_RESOURCE_LIMIT", "SUSPECT_ORIGINAL_GEOMETRY",
                                           "UPSTREAM_JOINT_OR_SUPPORT_CONFLICT", "SPLIT")
STATUS_CODES = {CORE: 0, MOVING_S: 1, STATIC_S: 2, UNRESOLVED: 3, SUSPECT: 4, JOINT_INVALID: 5, UNTESTED: 6, UNOBS: 7}
UNDECIDED = (UNRESOLVED, UNOBS, SUSPECT, UNTESTED)     # may be rendered provisionally; never seed / never "known"
SKIP_NONE, SKIP_OFFSCREEN, SKIP_IDENTITY, SKIP_NOT_EVALUATED = 0, 1, 2, 3
POLICY_VERSION = "4A-unified-2026-11-12-core-revisable"
RESOURCE_FIELDS = ("max_batches", "max_hours", "progress_every_s")    # not part of the scientific identity


@dataclass
class Policy4A:
    # grouping (layer-aware, physically linked footprints)
    component_voxel: float = 0.05          # m, hard cap on a link
    link_factor: float = 3.0               # link if |x_i - x_j| <= link_factor (s_max,i + s_max,j): the kernel's
                                           # 3-sigma extents touch (was 1.0 -> 75 % singletons on red_fridge)
    max_component: int = 128               # larger connected components are bisected before testing
    min_component: int = 4
    normal_cos: float = 0.8
    layer_tol: float = 0.008               # m
    flat_max: float = 0.5
    # decision (declared provisional policy; nothing here is a probability)
    min_changed_px: int = 1                # a view is informative if the flip changes >= this many pixels
    decisive_abs_px: float = 1e-3          # minimal |Delta| in pixel-equivalent loss units (sum_v delta_v H_v W_v)
    decisive_rel: float = 0.25             # |sum_v delta_v| / sum_v |delta_v|
    favour_frac: float = 0.6               # sign agreement among informative views AND among time blocks
    block_frames: int = 16                 # processing rows per stability block (per camera)
    channel_conflict_frac: float = 0.1     # appearance (L1 + SSIM) and geometry (depth + mask) must not oppose each
                                           # other: opposite signs with the weaker >= this x the stronger -> UNRESOLVED
                                           # (missing geometry otherwise gets absorbed by relabelling originals)
    omit_margin: float = 0.05              # SUSPECT_ORIGINAL: E_omit < min(E_S, E_M) - margin * sum|delta_v|
    # scheduling / resources (0 = unlimited; a limit makes the run visibly incomplete)
    max_batches: int = 0
    max_hours: float = 0.0
    max_split_depth: int = 5                # 192-member groups -> >= 6 members; only separable splits are kept
    # joint sensitivity
    sensitivity: bool = True
    # Phase-3 support: "revisable" (decisive-STATIC support groups are relabelled) | "gate" (they fail 4A)
    core_policy: str = "revisable"
    sens_axis_deg: float = 1.0
    sens_point_m: float = 0.005
    sens_q_rel: float = 0.02
    sens_groups: int = 16
    # logging
    progress_every_s: float = 60.0

    def to_dict(self):
        return asdict(self) | {"policy_version": POLICY_VERSION}


@dataclass
class Group:
    gid: int
    members: np.ndarray
    parent: int = -1
    depth: int = 0
    kind: str = "leaf"                     # leaf | probe | split | core
    status: str = UNTESTED
    reason: str = "not evaluated"
    stats: dict = field(default_factory=dict)
    version: int = -1                      # scene version of its current evidence
    mixed: bool = False
    origin: str = "candidate"              # candidate | core (a Phase-3 support group revised by the core audit)


@dataclass
class Counters:
    sweeps: int = 0
    views_uploaded: int = 0
    baseline_renders: int = 0
    footprint_renders: int = 0
    hypothesis_renders: int = 0
    omit_renders: int = 0
    group_view_scored: int = 0
    group_view_skipped_offscreen: int = 0
    group_view_skipped_identity: int = 0
    roi_boxes: int = 0
    roi_pixels: int = 0
    gpu_to_cpu_copies: int = 0
    gpu_to_cpu_bytes: int = 0
    commits: int = 0
    batches: int = 0
    seconds_footprints: float = 0.0
    seconds_sweeps: float = 0.0

    def to_dict(self):
        return asdict(self)


# ====================================================================== evaluator
class PosedPair:
    """Both physical poses of every Gaussian at one view (static: canonical; moving: Phase-3 transport) and
    the matching colours, computed exactly as articulated_scene.render_articulated computes them, so a
    hypothesis render is a per-row selection between the two (bitwise the same kernel inputs)."""

    def __init__(self, scene, vr, tr):
        from .articulated_scene import quat_mul
        from .raster_oracle import sh_to_rgb
        t = scene.tensors()
        ft = tr.at(vr.frame)
        self.t, self.cam = t, vr.cam
        self.muS, self.qS = t["xyz"], t["rot"]
        self.muM = t["xyz"] @ ft.Q.T + ft.p
        self.qM = quat_mul(ft.qQ.expand_as(t["rot"]), t["rot"])
        deg = scene.sh_degree
        sh = torch.cat([t["dc"], t["rest"]], 1)[:, :max((deg + 1) ** 2, 1)].contiguous()
        self.world_sh = scene.sh_mode == "world_SH"
        if self.world_sh:
            self.sh, self.deg = sh, deg
        else:
            C = vr.cam.campos.to(t["xyz"].device)
            self.cS = sh_to_rgb(t["xyz"], C[None, :].expand_as(t["xyz"]), sh, deg)
            self.cM = sh_to_rgb(t["xyz"], ((C - ft.p) @ ft.Q)[None, :].expand_as(t["xyz"]), sh, deg)
            self.deg = deg
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        dev = t["xyz"].device
        self.R = GaussianRasterizer(GaussianRasterizationSettings(
            image_height=vr.cam.H, image_width=vr.cam.W, tanfovx=vr.cam.tanfovx, tanfovy=vr.cam.tanfovy,
            bg=scene.bg.to(dev).float(), scale_modifier=1.0, viewmatrix=vr.cam.V.to(dev), projmatrix=vr.cam.P.to(dev),
            sh_degree=deg, campos=vr.cam.campos.to(dev), prefiltered=False, debug=False, antialiasing=False))

    def render(self, labels: torch.Tensor, keep=None) -> dict:
        from diff_gaussian_rasterization import metric_depth_from
        m = labels[:, None]
        mu = torch.where(m, self.muM, self.muS)
        q = torch.where(m, self.qM, self.qS)
        op, sc = self.t["opacity"], self.t["scale"]
        col = None if self.world_sh else torch.where(m, self.cM, self.cS)
        sh = self.sh if self.world_sh else None
        if keep is not None:
            mu, q, op, sc = mu[keep], q[keep], op[keep], sc[keep]
            col = None if col is None else col[keep]
            sh = None if sh is None else sh[keep]
        means2D = torch.zeros(mu.shape[0], 3, device=mu.device, dtype=mu.dtype)
        if self.world_sh:
            color, radii, invd, alpha, zd = self.R.forward_rgbd(mu, means2D, op, shs=sh, scales=sc, rotations=q)
        else:
            color, radii, invd, alpha, zd = self.R.forward_rgbd(mu, means2D, op, colors_precomp=col, scales=sc, rotations=q)
        return dict(color=color, alpha=alpha, zdepth=zd, depth=metric_depth_from(zd, alpha, 1e-6), radii=radii, means=mu)


class ExactEvaluator:
    """Exact all-view S/M (and OMIT) scoring of groups against a frozen labelling."""

    def __init__(self, scene, pviews, cfg: ExactLossConfig, log=print, progress_every_s=60.0):
        self.scene, self.pviews, self.cfg, self.log = scene, pviews, cfg, log
        self.views = scene.views
        self.V = len(self.views)
        self.cams = sorted({vr.cam.name for vr in self.views}, key=lambda n: [vr.cam.name for vr in self.views].index(n))
        self.cam_of_view = np.array([self.cams.index(vr.cam.name) for vr in self.views])
        self.HW = np.array([vr.cam.W * vr.cam.H for vr in self.views], float)
        self.counters = Counters()
        self.progress_every_s = progress_every_s
        self._fp = {}
        self.grid = (max(FP.tile_grid(vr.cam.W, vr.cam.H)[1] for vr in self.views),
                     max(FP.tile_grid(vr.cam.W, vr.cam.H)[0] for vr in self.views))

    # ------------------------------------------------------------------ footprints
    def footprints(self, transport=None, key="nominal"):
        """Per-Gaussian padded tile rects: static pose per camera [C,N,4], moving pose per view [V,N,4] (GPU)."""
        if key in self._fp:
            return self._fp[key]
        t0 = time.time()
        sc = self.scene
        N = len(sc.g)
        tr = transport or sc.transport
        allS = torch.zeros(N, dtype=torch.bool, device=sc.device)
        allM = torch.ones(N, dtype=torch.bool, device=sc.device)
        rS = torch.zeros(len(self.cams), N, 4, dtype=torch.int16, device=sc.device)
        rM = torch.zeros(self.V, N, 4, dtype=torch.int16, device=sc.device)
        ident = np.zeros(self.V, bool)
        done = set()
        for vi, vr in enumerate(self.views):
            ci = self.cam_of_view[vi]
            if ci not in done:
                o = sc.render(vr, allS, joint_transport=tr)
                rS[ci] = FP.tile_rects(vr.cam, o["means"], o["radii"])
                done.add(ci)
                self.counters.footprint_renders += 1
            o = sc.render(vr, allM, joint_transport=tr)
            rM[vi] = FP.tile_rects(vr.cam, o["means"], o["radii"])
            self.counters.footprint_renders += 1
            ft = tr.at(vr.frame)
            ident[vi] = bool(torch.equal(ft.Q, torch.eye(3, device=ft.Q.device, dtype=ft.Q.dtype))
                             and not bool(ft.p.abs().max() > 0))
        self.counters.seconds_footprints += time.time() - t0
        self._fp[key] = (rS, rM, ident)
        return self._fp[key]

    def group_rects(self, groups, key="nominal", transport=None):
        """[G,C,4] static and [G,V,4] moving union rects (numpy int16)."""
        rS, rM, _ = self.footprints(transport, key)
        dev = rS.device
        mem = np.concatenate([g.members for g in groups])
        gof = np.concatenate([np.full(len(g.members), k) for k, g in enumerate(groups)])
        mi, go = torch.as_tensor(mem, device=dev), torch.as_tensor(gof, device=dev)
        gS = FP.aggregate(rS, mi, go, len(groups)).permute(1, 0, 2).cpu().numpy()
        out = []
        for a in range(0, self.V, 64):                                 # bounded GPU temporaries
            out.append(FP.aggregate(rM[a:a + 64], mi, go, len(groups)).cpu())
        gM = torch.cat(out, 0).permute(1, 0, 2).numpy()
        return gS, gM

    # ------------------------------------------------------------------ one view-major sweep
    def sweep(self, items, labels: torch.Tensor, rects, omit=False, transport=None, key="nominal", tag="sweep"):
        """Score every item on every view against the frozen `labels`. An item is a list of Groups flipped
        TOGETHER (one group = the S/M test of that group; several = one joint move). rects = (S [I,C,4],
        M [I,V,4]) union rects of each item.

        Returns numpy arrays: d_alt [I,V,4] (L_v(alt) - L_v(base) per channel), changed [I,V], skip [I,V],
        unc_new [I,V], omit [I,V] (E_omit - E_current; NaN unless omit, single-group items only),
        base_totals [V,4]."""
        sc, dev = self.scene, self.scene.device
        tr = transport or sc.transport
        _, _, ident = self.footprints(transport, key)
        gS, gM = rects
        n = len(items)
        D = torch.zeros(n, self.V, 4, dtype=torch.float64, device=dev)
        CH = torch.zeros(n, self.V, dtype=torch.int64, device=dev)
        UN = torch.zeros(n, self.V, dtype=torch.int64, device=dev)
        OM = torch.full((n, self.V), float("nan"), dtype=torch.float64, device=dev)
        BR = torch.zeros(n, self.V, 4, dtype=torch.float64, device=dev)       # current-labelling loss inside the ROI
        SK = np.full((n, self.V), SKIP_NONE, np.int8)
        BT = torch.zeros(self.V, 4, dtype=torch.float64, device=dev)
        flips = []
        for it in items:
            parts = []
            for g in it:
                ix = torch.as_tensor(g.members, device=dev)
                c = bool(labels[int(g.members[0])])
                if bool((labels[ix] != c).any()):          # invariant: a group renders with one label
                    raise RuntimeError(f"group {g.gid} has mixed render labels")
                parts.append((ix, c))
            flips.append(parts)
        t0 = time.time()
        last = t0
        n_scored = 0
        for vi, vr in enumerate(self.views):
            tgt = ViewTargets(self.pviews.gpu_view(vi, dev), self.cfg)
            self.counters.views_uploaded += 1
            pp = PosedPair(sc, vr, tr)
            base = pp.render(labels)
            self.counters.baseline_renders += 1
            bm = tgt.full_maps(base)
            BT[vi] = tgt.totals(bm)
            ci = self.cam_of_view[vi]
            for k in range(n):
                rs, rm = gS[k, ci], gM[k, vi]
                if not (FP.rect_nonempty(rs) or FP.rect_nonempty(rm)):
                    SK[k, vi] = SKIP_OFFSCREEN
                    self.counters.group_view_skipped_offscreen += 1
                    continue
                if ident[vi]:
                    SK[k, vi] = SKIP_IDENTITY
                    self.counters.group_view_skipped_identity += 1
                    continue
                alt = labels.clone()
                for ix, c in flips[k]:
                    alt[ix] = not c
                a = pp.render(alt)
                self.counters.hypothesis_renders += 1
                boxes = FP.boxes_for(rs, rm, tgt.W, tgt.H, SSIM_R)
                d, ch, un = box_delta(a, base, bm, tgt, boxes)
                self.counters.roi_boxes += len(boxes)
                self.counters.roi_pixels += sum((b[1] - b[0]) * (b[3] - b[2]) for b in boxes)
                D[k, vi], CH[k, vi], UN[k, vi] = d, ch, un
                for (y0, y1, x0, x1) in boxes:
                    BR[k, vi] += torch.stack([bm[c][y0:y1, x0:x1].sum() for c in CHANNELS])
                if omit and len(flips[k]) == 1:
                    ix, c = flips[k][0]
                    keep = torch.ones_like(labels)
                    keep[ix] = False
                    o = pp.render(labels, keep=keep)
                    self.counters.omit_renders += 1
                    ob = FP.boxes_for(rm if c else rs, np.zeros(4, np.int16), tgt.W, tgt.H, SSIM_R)
                    od, _, _ = box_delta(o, base, bm, tgt, ob)
                    OM[k, vi] = od.sum()
                n_scored += 1
                self.counters.group_view_scored += 1
            if time.time() - last > self.progress_every_s:
                last = time.time()
                rate = n_scored / max(1e-9, last - t0)
                left = (self.V - vi - 1) / max(1, vi + 1) * (last - t0)
                self.log(f"    {tag}: view {vi + 1}/{self.V}, {n_scored} item-views scored "
                         f"({rate:.0f}/s), ETA this sweep {left / 60:.1f} min (measured)")
        out = {"d_alt": D.cpu().numpy(), "changed": CH.cpu().numpy(), "unc_new": UN.cpu().numpy(),
               "omit": OM.cpu().numpy(), "skip": SK, "base_totals": BT.cpu().numpy(), "base_roi": BR.cpu().numpy()}
        self.counters.gpu_to_cpu_copies += 6
        self.counters.gpu_to_cpu_bytes += sum(v.nbytes for k, v in out.items() if k != "skip")
        self.counters.sweeps += 1
        self.counters.seconds_sweeps += time.time() - t0
        return out


def union_rects(r: np.ndarray) -> np.ndarray:
    """Union bounding rect over axis 0 of r [k,...,4] (empty rects ignored)."""
    r = np.asarray(r, np.int32)
    ne = FP.rect_nonempty(r)
    big = 1 << 20
    lo = np.where(ne[..., None], r[..., :2], big).min(0)
    hi = np.where(ne[..., None], r[..., 2:], -1).max(0)
    out = np.concatenate([lo, hi], -1)
    out[(out[..., 2] <= out[..., 0]) | (out[..., 3] <= out[..., 1])] = 0
    return out.astype(np.int16)


# ====================================================================== decision
def blocks_of(views, block_frames):
    """Stability block id per view: (camera, processing-row // block_frames)."""
    cams, rows, out = {}, {}, []
    for vr in views:
        rows.setdefault(vr.cam.name, 0)
        k = rows[vr.cam.name]
        rows[vr.cam.name] += 1
        out.append(cams.setdefault((vr.cam.name, k // block_frames), len(cams)))
    return np.asarray(out)


def decide(delta_v4: np.ndarray, changed: np.ndarray, HW: np.ndarray, blocks: np.ndarray, cam_of_view, pol: Policy4A):
    """delta_v4 [V,4] (E_S - E_M per view and channel). Returns (decision, reason, stats, mixed)."""
    dv = delta_v4.sum(1)
    inf = changed >= pol.min_changed_px
    stats = {"Delta": float(dv.sum()), "Delta_px": float((dv * HW).sum()), "abs_sum": float(np.abs(dv).sum()),
             "channels": {c: float(delta_v4[:, i].sum()) for i, c in enumerate(CHANNELS)},
             "informative_views": int(inf.sum()), "changed_pixels": int(changed.sum())}
    stats["abs_sum_px"] = float((np.abs(dv) * HW).sum())
    if not inf.any():
        return UNOBS, "S and M renders are identical in every view (motion unobservable)", stats, False
    if stats["abs_sum_px"] < pol.decisive_abs_px:
        return UNOBS, "S and M differ by less than the declared magnitude in every view (motion not observable)", \
            stats, False
    D = stats["Delta"]
    stats["consistency"] = abs(D) / max(stats["abs_sum"], 1e-300)
    fav_m = float((dv[inf] > 0).mean())
    stats["favour_moving_views"] = fav_m
    ub = np.unique(blocks[inf])
    bs = np.array([dv[(blocks == b) & inf].sum() for b in ub])
    stats["blocks"] = int(len(ub))
    stats["block_favour_moving"] = float((bs > 0).mean())
    stats["per_camera"] = {int(c): float(dv[cam_of_view == c].sum()) for c in np.unique(cam_of_view)}
    if abs(stats["Delta_px"]) < pol.decisive_abs_px or abs(D) < pol.decisive_rel * stats["abs_sum"]:
        return UNRESOLVED, "indecisive evidence (below the declared magnitude / consistency thresholds)", stats, False
    app = float(delta_v4[:, 0].sum() + delta_v4[:, 1].sum())
    geo = float(delta_v4[:, 2].sum() + delta_v4[:, 3].sum())
    stats["appearance"], stats["geometry"] = app, geo
    if app * geo < 0 and min(abs(app), abs(geo)) >= pol.channel_conflict_frac * max(abs(app), abs(geo)):
        return UNRESOLVED, "appearance and geometry channels disagree (possible missing geometry absorbed by a " \
                           "relabelling)", stats, False
    side = MOVING_S if D > 0 else STATIC_S
    vf = fav_m if D > 0 else 1 - fav_m
    bf = stats["block_favour_moving"] if D > 0 else 1 - stats["block_favour_moving"]
    if vf >= pol.favour_frac and bf >= pol.favour_frac:
        return side, f"{side.lower()} explains the informative views and time blocks", stats, False
    return UNRESOLVED, "views / time blocks disagree in sign (mixed evidence)", stats, True


# ====================================================================== scheduling helpers
def paint(rect_sets, cam_of_view, grid, dil=0):
    """Tile occupancy [V, gy, gx] of a set of groups: rect_sets = list of (S rects [C,4], M rects [V,4])."""
    V = len(cam_of_view)
    gy, gx = grid
    occ = np.zeros((V, gy, gx), bool)
    ys, xs = np.arange(gy)[None, :, None], np.arange(gx)[None, None, :]
    for rs, rm in rect_sets:
        for r in (rs[cam_of_view], rm):
            r = r.astype(np.int32)
            ne = FP.rect_nonempty(r)
            if not ne.any():
                continue
            x0, y0, x1, y1 = (r[:, i][:, None, None] for i in range(4))
            m = (xs >= x0 - dil) & (xs < x1 + dil) & (ys >= y0 - dil) & (ys < y1 + dil) & ne[:, None, None]
            occ |= m
    return occ


def touches(occ, rS_g, rM_g, cam_of_view) -> np.ndarray:
    """Per group: does any of its rects meet the occupancy in any view? rS_g [G,C,4], rM_g [G,V,4] -> [G] bool."""
    V, gy, gx = occ.shape
    I = np.zeros((V, gy + 1, gx + 1), np.int32)
    I[:, 1:, 1:] = occ.cumsum(1).cumsum(2)
    vv = np.arange(V)[None, :]
    hit = np.zeros(rS_g.shape[0], bool)
    for r in (rS_g[:, cam_of_view], rM_g):                     # [G,V,4]
        r = r.astype(np.int64)
        x0, y0 = np.clip(r[..., 0], 0, gx), np.clip(r[..., 1], 0, gy)
        x1, y1 = np.clip(r[..., 2], 0, gx), np.clip(r[..., 3], 0, gy)
        s = I[vv, y1, x1] - I[vv, y0, x1] - I[vv, y1, x0] + I[vv, y0, x0]
        hit |= ((s > 0) & FP.rect_nonempty(r)).any(1)
    return hit


def perturbed_joints(joint, pol: Policy4A):
    """Declared shared perturbations of the WHOLE joint (every moving Gaussian moves with it)."""
    from dataclasses import replace
    e = np.asarray(joint.axis)
    u = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
    w = np.cross(e, u)
    out = []
    for name, b in (("axis+u", u), ("axis-u", -u), ("axis+w", w), ("axis-w", -w)):
        k = np.cross(e, b) / np.linalg.norm(np.cross(e, b))
        e2 = rot_exp(k, np.radians(pol.sens_axis_deg)) @ e
        out.append((name, replace(joint, axis=tuple(e2 / np.linalg.norm(e2)))))
    if joint.type == "revolute":
        a = np.asarray(joint.point)
        for name, b in (("point+u", u), ("point-u", -u), ("point+w", w), ("point-w", -w)):
            out.append((name, replace(joint, point=tuple(a + pol.sens_point_m * b))))
    for name, s in (("q*(1+r)", 1 + pol.sens_q_rel), ("q*(1-r)", 1 - pol.sens_q_rel)):
        out.append((name, replace(joint, q=tuple(float(x) * s for x in joint.q))))
    return out


# ====================================================================== the stage
class Phase4A:
    def __init__(self, scene, pviews, core: np.ndarray, pol: Policy4A, cfg: ExactLossConfig, log=print,
                 checkpoint_dir=None):
        self.scene, self.pviews, self.pol, self.cfg, self.log = scene, pviews, pol, cfg, log
        self.core = np.asarray(core, np.int64)
        self.N = len(scene.g)
        self.ev = ExactEvaluator(scene, pviews, cfg, log, pol.progress_every_s)
        self.blocks = blocks_of(scene.views, pol.block_frames)
        self.labels = torch.zeros(self.N, dtype=torch.bool, device=scene.device)
        self.labels[torch.as_tensor(self.core, device=scene.device)] = True
        self.provisional_upstream = self.labels.clone()
        self.groups: list[Group] = []
        self.rects = (np.zeros((0, len(self.ev.cams), 4), np.int16), np.zeros((0, self.ev.V, 4), np.int16))
        self.delta = np.zeros((0, self.ev.V, 4)); self.changed = np.zeros((0, self.ev.V), np.int64)
        self.skip = np.zeros((0, self.ev.V), np.int8); self.unc_new = np.zeros((0, self.ev.V), np.int64)
        self.omit = np.zeros((0, self.ev.V))
        self.base_roi = {}                                  # gid -> [V,4] current-labelling ROI loss (final sweep)
        self.core_audit, self.core_conflict = [], False
        self.core_revised = None
        self.version = 0
        self.history = []
        self.ckpt = Path(checkpoint_dir) if checkpoint_dir else None
        self.t0 = time.time()
        self.complete = True
        self.stop_reason = None

    # ------------------------------------------------------------------ bookkeeping
    def _add_groups(self, new):
        gS, gM = self.ev.group_rects(new)
        self.groups.extend(new)
        self.rects = (np.concatenate([self.rects[0], gS]), np.concatenate([self.rects[1], gM]))
        G = len(new)
        z = lambda *s, dt=float: np.zeros(s, dt)
        self.delta = np.concatenate([self.delta, z(G, self.ev.V, 4)])
        self.changed = np.concatenate([self.changed, z(G, self.ev.V, dt=np.int64)])
        self.skip = np.concatenate([self.skip, np.full((G, self.ev.V), SKIP_NOT_EVALUATED, np.int8)])
        self.unc_new = np.concatenate([self.unc_new, z(G, self.ev.V, dt=np.int64)])
        self.omit = np.concatenate([self.omit, np.full((G, self.ev.V), np.nan)])

    def _score(self, gids, omit=False, tag="sweep"):
        gs = [self.groups[i] for i in gids]
        rects = (self.rects[0][gids], self.rects[1][gids])
        r = self.ev.sweep([[g] for g in gs], self.labels, rects, omit=omit, tag=tag)
        for k, i in enumerate(gids):
            sgn = 1.0 if self._label(i) else -1.0                 # E_S - E_M from L(alt) - L(base)
            self.delta[i], self.changed[i], self.skip[i] = sgn * r["d_alt"][k], r["changed"][k], r["skip"][k]
            self.unc_new[i] = r["unc_new"][k]
            if omit:
                self.omit[i] = r["omit"][k]
                self.base_roi[i] = r["base_roi"][k]
            dec, why, st, mixed = decide(self.delta[i], self.changed[i], self.ev.HW, self.blocks, self.ev.cam_of_view, self.pol)
            g = self.groups[i]
            g.status, g.reason, g.stats, g.mixed, g.version = dec, why, st, mixed, self.version
        E = float(r["base_totals"].sum())
        return r, E

    def _budget_exceeded(self):
        p = self.pol
        if p.max_hours and (time.time() - self.t0) / 3600 > p.max_hours:
            return f"max_hours={p.max_hours} reached"
        if p.max_batches and self.ev.counters.batches >= p.max_batches:
            return f"max_batches={p.max_batches} reached"
        return None

    def _label(self, i) -> bool:
        return bool(self.labels[int(self.groups[i].members[0])])

    def _flip(self, i):
        g = self.groups[i]
        self.labels[torch.as_tensor(g.members, device=self.scene.device)] = not self._label(i)

    def _leaves(self):
        return [i for i, g in enumerate(self.groups) if g.kind == "leaf"]

    # ------------------------------------------------------------------ checkpoint
    def checkpoint(self, phase: str):
        if self.ckpt is None:
            return
        self.ckpt.mkdir(parents=True, exist_ok=True)
        tmp = self.ckpt / "state.npz.tmp.npz"
        np.savez_compressed(tmp, labels=self.labels.cpu().numpy(), delta=self.delta, changed=self.changed,
                            E_before=np.asarray(getattr(self, "E_before", np.zeros((0, 4)))),
                            skip=self.skip, unc_new=self.unc_new, omit=self.omit,
                            members=np.concatenate([g.members for g in self.groups]) if self.groups else np.zeros(0, int),
                            sizes=np.array([len(g.members) for g in self.groups], np.int64))
        tmp.replace(self.ckpt / "state.npz")
        atomic_json(self.ckpt / "checkpoint.json", {
            "phase": phase, "version": self.version, "complete": self.complete and phase == "done",
            "stop_reason": self.stop_reason, "policy": self.pol.to_dict(), "history": self.history,
            "groups": [{"gid": g.gid, "parent": g.parent, "depth": g.depth, "kind": g.kind, "status": g.status,
                        "reason": g.reason, "version": g.version, "mixed": g.mixed, "origin": g.origin}
                       for g in self.groups],
            "counters": self.ev.counters.to_dict(), "seconds": time.time() - self.t0})

    def load_checkpoint(self) -> bool:
        """Restore groups, labels, evidence and history from <checkpoint_dir> (phase 'batches'). Decisions are
        recomputed from the stored exact evidence (deterministic). Returns False when there is nothing to resume."""
        if self.ckpt is None or not (self.ckpt / "checkpoint.json").is_file():
            return False
        meta = json.loads((self.ckpt / "checkpoint.json").read_text())
        if meta["phase"] not in ("batches", "stopped"):
            return False
        ident = lambda d: {k: v for k, v in d.items() if k not in RESOURCE_FIELDS}
        if ident(meta["policy"]) != ident(self.pol.to_dict()):
            raise RuntimeError("4A checkpoint was written with a different policy; refusing to resume")
        z = np.load(self.ckpt / "state.npz")
        self.labels = torch.as_tensor(z["labels"], device=self.scene.device)
        cuts = np.cumsum(z["sizes"])[:-1]
        mems = np.split(z["members"], cuts) if len(z["sizes"]) else []
        groups = []
        for gm, m in zip(meta["groups"], mems):
            groups.append(Group(gm["gid"], m.astype(np.int64), parent=gm["parent"], depth=gm["depth"], kind=gm["kind"],
                                status=gm["status"], reason=gm["reason"], version=gm["version"], mixed=gm["mixed"],
                                origin=gm.get("origin", "candidate")))
        self.groups = []
        self.rects = (np.zeros((0, len(self.ev.cams), 4), np.int16), np.zeros((0, self.ev.V, 4), np.int16))
        self.delta = np.zeros((0, self.ev.V, 4)); self.changed = np.zeros((0, self.ev.V), np.int64)
        self.skip = np.zeros((0, self.ev.V), np.int8); self.unc_new = np.zeros((0, self.ev.V), np.int64)
        self.omit = np.zeros((0, self.ev.V))
        self._add_groups(groups)
        self.delta, self.changed, self.skip = z["delta"], z["changed"], z["skip"]
        self.unc_new, self.omit = z["unc_new"], z["omit"]
        for i, g in enumerate(self.groups):
            if g.kind in ("leaf", "probe") and (self.skip[i] != SKIP_NOT_EVALUATED).any():
                dec, why, st, mixed = decide(self.delta[i], self.changed[i], self.ev.HW, self.blocks, self.ev.cam_of_view,
                                             self.pol)
                g.status, g.reason, g.stats, g.mixed = dec, why, st, mixed
        self.history, self.version = meta["history"], meta["version"]
        self.ev.counters.batches = meta["counters"]["batches"]
        self.resumed_from = {"version": self.version, "batches": self.ev.counters.batches, "seconds_before": meta["seconds"]}
        self.log(f"4A: RESUMED from checkpoint (version {self.version}, batch {self.ev.counters.batches}, "
                 f"{len(self.groups)} groups)")
        return True

    # ------------------------------------------------------------------ run
    def run(self, resume: bool = False) -> dict:
        pol, log = self.pol, self.log
        ev = self.ev
        log(f"4A: {self.N} original Gaussians ({len(self.core)} core), {ev.V} views, cameras {ev.cams}")
        log("4A: conservative footprints (kernel radii, static per camera + moving per view)")
        ev.footprints()
        log(f"4A: footprints ready ({ev.counters.footprint_renders} renders, {ev.counters.seconds_footprints:.1f} s)")
        # 1. core check
        core_g = Group(-1, self.core, kind="core")
        self.core_rects = ev.group_rects([core_g])
        r = ev.sweep([[core_g]], self.labels, self.core_rects, tag="core check")
        dec, why, st, _ = decide(r["d_alt"][0], r["changed"][0], ev.HW, self.blocks, ev.cam_of_view, pol)
        self.E_before = r["base_totals"]
        self.core_check = {"decision": dec, "reason": why, **st,
                           "views_scored": int((r["skip"][0] == SKIP_NONE).sum())}
        log(f"4A core check: {dec} ({why}); Delta {st['Delta']:.6g} ({st['Delta_px']:.4g} px-equiv) over "
            f"{st['informative_views']} informative views")
        if dec != MOVING_S:
            self.aborted = True
            return self.result()
        self.aborted = False
        # 2. groups
        tt = self.scene.tensors()
        xyz = tt["xyz"].cpu().numpy()
        self._xyz = xyz
        if resume and self.load_checkpoint():
            self._converge(self.history[-1]["E"])
            return self._finish()
        cand = np.setdiff1d(np.arange(self.N), self.core)
        normals, flat = surface_normals(tt["rot"].cpu().numpy(), tt["scale"].cpu().numpy())
        comps = build_components(xyz, cand, pol.component_voxel, pol.max_component, normals, flat, pol.normal_cos,
                                 pol.flat_max, extent=tt["scale"].cpu().numpy().max(1), link_factor=pol.link_factor,
                                 layer_tol=pol.layer_tol)
        self._add_groups([Group(k, m) for k, m in enumerate(comps)])
        sz = np.array([len(m) for m in comps])
        log(f"4A: {len(cand)} candidate Gaussians in {len(comps)} groups (sizes: median {np.median(sz):.0f}, "
            f"max {sz.max()}, singletons {(sz == 1).sum()})")
        # 3. initial sweep + monotone batches
        _, E = self._score(self._leaves(), tag="initial sweep")
        self.history.append({"batch": 0, "E": E, "committed": 0, "predicted_drop": 0.0})
        log(f"4A initial sweep: E = {E:.8g}; {self._status_counts()}")
        self.checkpoint("batches")
        self._converge(E)
        return self._finish()

    def _finish(self) -> dict:
        pol, log = self.pol, self.log
        # 4. split probes for mixed evidence
        for depth in range(pol.max_split_depth):
            if self.stop_reason:
                break
            mixed = [i for i in self._leaves() if self.groups[i].status == UNRESOLVED and self.groups[i].mixed
                     and len(self.groups[i].members) >= 2 * pol.min_component and self.groups[i].depth == depth]
            if not mixed:
                break
            probes, parent_of = [], {}
            for i in mixed:
                for h in split_component(self._xyz, self.groups[i].members):
                    gid = len(self.groups) + len(probes)
                    probes.append(Group(gid, h, parent=self.groups[i].gid, depth=depth + 1, kind="probe"))
                    parent_of[gid] = i
            first = len(self.groups)
            self._add_groups(probes)
            pids = list(range(first, len(self.groups)))
            self._score(pids, tag=f"split probes (depth {depth + 1})")
            kept = 0
            for i in mixed:
                h = [j for j in pids if parent_of[j] == i]
                s = [self.groups[j].status for j in h]
                mx = [self.groups[j].mixed for j in h]
                separable = len(h) == 2 and set(s) == {MOVING_S, STATIC_S}
                partial = len(h) == 2 and sum(x in (MOVING_S, STATIC_S) for x in s) == 1 and any(mx)
                if separable or partial:
                    # demonstrably mixed parent; the halves separate the evidence (fully, or one half decisive and
                    # the other still mixed -> that half is probed again at the next depth)
                    self.groups[i].kind, self.groups[i].status = "split", SPLIT
                    self.groups[i].reason = f"halves {'separable' if separable else 'partly separable'}: {s}"
                    for j in h:
                        self.groups[j].kind = "leaf"
                    kept += 1
                else:
                    for j in h:
                        self.groups[j].kind, self.groups[j].reason = "probe_discarded", f"halves {s}: not separable"
            log(f"4A split probes depth {depth + 1}: {len(mixed)} mixed groups, {kept} split (separable halves)")
            if not kept:
                break
            E = self.history[-1]["E"]
            self._converge(E)
        # 5. final verification sweep (all leaves, all views, OMIT diagnostic) = global scene audit
        E = self._final_verification("final verification")
        log(f"4A final verification: E = {E:.8g}; {self._status_counts()}")
        self._core_audit()
        if pol.core_policy == "revisable" and self.core_conflict and not self.stop_reason:
            self._revise_core(E)
        # 6. joint sensitivity
        self.sens = self._sensitivity() if pol.sensitivity else {}
        self.checkpoint("done" if self.complete else "stopped")       # a stopped run stays resumable
        return self.result()

    def _final_verification(self, tag) -> float:
        pol = self.pol
        leaves = self._leaves()
        self.version += 1
        r, E = self._score(leaves, omit=True, tag=tag)
        self.E_after = r["base_totals"]
        for i in leaves:
            g = self.groups[i]
            lab = self._label(i)
            if g.status in (MOVING_S, STATIC_S) and (g.status == MOVING_S) != lab:
                g.status, g.reason = UNRESOLVED, f"final evidence favours {g.status} but the label was not " \
                                                 f"committed ({self.stop_reason or 'interaction'})"
            om = self.omit[i]
            if np.isfinite(om).any():
                e_omit = float(np.nansum(om))
                d = g.stats.get("Delta", 0.0)
                e_alt = -d if lab else d                  # E_alt - E_current
                g.stats.update(omit_minus_current=e_omit, alt_minus_current=e_alt)
                margin = pol.omit_margin * g.stats.get("abs_sum", 0.0)
                if e_omit < min(0.0, e_alt) - margin:
                    g.stats["suspect_original"] = True
                    if g.status in (UNRESOLVED, UNOBS):
                        g.status, g.reason = SUSPECT, "omitting the group fits better than both S and M (flag only)"
        return E

    def _revise_core(self, E):
        """core_policy 'revisable': the support groups join the leaves; exact monotone descent commits the decisive
        STATIC flips (and anything they interact with); a second final verification re-decides every group."""
        ids = [i for i, g in enumerate(self.groups) if g.kind == "core_audit"]
        n_static = [i for i in ids if self.groups[i].status == STATIC_S]
        self.log(f"4A core revision: {len(n_static)} Phase-3 support groups "
                 f"({sum(len(self.groups[i].members) for i in n_static)} Gaussians) decisively prefer STATIC -> "
                 f"relabelled by the exact descent, then every group is re-verified")
        for i in ids:
            self.groups[i].kind, self.groups[i].origin = "leaf", "core"
        self._converge(E)
        E = self._final_verification("final verification (after the support revision)")
        self._record_core_outcome(ids)
        self.log(f"4A final verification (after the support revision): E = {E:.8g}; {self._status_counts()}; "
                 f"support relabelled STATIC: {self.core_revised['groups']} groups, "
                 f"{self.core_revised['gaussians']} Gaussians")

    def _record_core_outcome(self, ids):
        self.core_audit, n_g, n_x = [], 0, 0
        for i in ids:
            g = self.groups[i]
            moved = not self._label(i)
            n_g += moved
            n_x += len(g.members) if moved else 0
            self.core_audit.append({"gid": g.gid, "size": int(len(g.members)), "decision": g.status,
                                    "reason": g.reason, "relabelled_static": bool(moved)})
        self.core_conflict = False
        self.core_revised = {"groups": int(n_g), "gaussians": int(n_x), "policy": "revisable"}

    def _components(self, props):
        """Connected components of the render-interaction graph among proposals (footprints dilated by one
        tile, any view). Components never interact with each other, so their moves add exactly."""
        ev = self.ev
        lab = np.full((ev.V,) + ev.grid, -1, np.int64)
        parent = {}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for i in props:
            parent[i] = i
            m = paint([(self.rects[0][i], self.rects[1][i])], ev.cam_of_view, ev.grid, dil=1)
            for o in np.unique(lab[m]):
                if o >= 0:
                    a, b = find(int(o)), find(i)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
            lab[m] = i
        comps = {}
        for i in props:
            comps.setdefault(find(i), []).append(i)
        return list(comps.values())

    def _converge(self, E):
        """Monotone descent on the exact objective. Each batch: proposals -> interaction components;
        a singleton commits its own (exact, favourable) flip; a larger component is scored as ONE joint move
        (all its proposals flipped together) and commits if that exact move lowers E by >= decisive_abs_px,
        otherwise only its best single proposal commits. Components do not interact, so gains add."""
        pol, log, ev = self.pol, self.log, self.ev
        thr = pol.decisive_abs_px
        while True:
            stop = self._budget_exceeded()
            self.stop_reason, self.complete = None, True
            if stop:
                self.stop_reason, self.complete = stop, False
                log(f"4A: RESOURCE STOP ({stop}); the run is INCOMPLETE")
                return
            props = [i for i in self._leaves() if self.groups[i].status in (MOVING_S, STATIC_S)
                     and (self.groups[i].status == MOVING_S) != self._label(i)]
            if not props:
                return
            props.sort(key=lambda i: (-abs(self.groups[i].stats["Delta"]), self.groups[i].gid))
            comps = self._components(props)
            multi = [c for c in comps if len(c) > 1]
            chosen, pred, joint_ok = [], 0.0, 0
            if multi:
                items = [[self.groups[i] for i in c] for c in multi]
                rects = (np.stack([union_rects(self.rects[0][c]) for c in multi]),
                         np.stack([union_rects(self.rects[1][c]) for c in multi]))
                r = ev.sweep(items, self.labels, rects, tag=f"batch {ev.counters.batches + 1} joint moves")
                for k, c in enumerate(multi):
                    dE = float(r["d_alt"][k].sum())
                    if (r["d_alt"][k].sum(1) * ev.HW).sum() <= -thr:
                        chosen += c
                        pred += dE
                        joint_ok += 1
                    else:
                        best = min(c, key=lambda i: (-abs(self.groups[i].stats["Delta"]), self.groups[i].gid))
                        chosen.append(best)
                        pred -= abs(self.groups[best].stats["Delta"])
            for c in comps:
                if len(c) == 1:
                    chosen.append(c[0])
                    pred -= abs(self.groups[c[0]].stats["Delta"])
            occ = paint([(self.rects[0][i], self.rects[1][i]) for i in chosen], ev.cam_of_view, ev.grid, dil=1)
            for i in chosen:
                self._flip(i)
            self.version += 1
            ev.counters.batches += 1
            ev.counters.commits += len(chosen)
            # re-score: uncommitted proposals + every leaf whose footprint meets a committed footprint
            leaves = np.array(self._leaves())
            hit = touches(occ, self.rects[0][leaves], self.rects[1][leaves], ev.cam_of_view)
            dirty = sorted(set(leaves[hit].tolist()) | (set(props) - set(chosen)))
            E_prev = E
            _, E = self._score(dirty, tag=f"batch {ev.counters.batches} re-score")
            err = (E - E_prev) - pred
            self.history.append({"batch": ev.counters.batches, "E": E, "proposals": len(props),
                                 "components": len(comps), "joint_moves": len(multi), "joint_moves_committed": joint_ok,
                                 "committed_groups": len(chosen), "rescored": len(dirty), "predicted_change": pred,
                                 "realised_change": E - E_prev, "additivity_error": err})
            log(f"4A batch {ev.counters.batches}: {len(props)} proposals in {len(comps)} interaction components "
                f"({len(multi)} joint moves, {joint_ok} committed whole); {len(chosen)} groups flipped; "
                f"E {E_prev:.8g} -> {E:.8g} (predicted {pred:+.4g}, additivity error {err:+.2e}); "
                f"re-scored {len(dirty)}; {self._status_counts()}")
            if E - E_prev > 1e-9 * max(1.0, abs(E_prev)):
                raise RuntimeError(f"4A objective increased after a commit ({E_prev} -> {E}): exactness violated")
            self.checkpoint("batches")

    def _core_audit(self):
        """7. Every core Gaussian is inspected: the core, split into the same layer-aware groups, is re-scored
        (S vs M, OMIT) on every view against the final labelling. Never relabelled; a decisive STATIC preference
        is UPSTREAM_JOINT_OR_SUPPORT_CONFLICT and fails the 4A integrity gate."""
        pol = self.pol
        self.core_audit, self.core_conflict = [], False
        n_keep = sum(1 for g in self.groups if g.kind != "core_audit")
        if n_keep < len(self.groups):                     # a resumed run re-audits; the old audit groups are the tail
            assert all(g.kind == "core_audit" for g in self.groups[n_keep:])
            self.groups = self.groups[:n_keep]
            self.rects = (self.rects[0][:n_keep], self.rects[1][:n_keep])
            for k in ("delta", "changed", "skip", "unc_new", "omit"):
                setattr(self, k, getattr(self, k)[:n_keep])
            self.base_roi = {i: v for i, v in self.base_roi.items() if i < n_keep}
        if not len(self.core):
            return
        revised = [i for i, g in enumerate(self.groups) if g.origin == "core"]
        if revised:                                       # resumed after a support revision: already re-verified
            self._record_core_outcome(revised)
            return
        tt = self.scene.tensors()
        normals, flat = surface_normals(tt["rot"].cpu().numpy(), tt["scale"].cpu().numpy())
        comps = build_components(self._xyz, self.core, pol.component_voxel, pol.max_component, normals, flat,
                                 pol.normal_cos, pol.flat_max, extent=tt["scale"].cpu().numpy().max(1),
                                 link_factor=pol.link_factor, layer_tol=pol.layer_tol)
        first = len(self.groups)
        self._add_groups([Group(first + k, np.asarray(m, np.int64), kind="core_audit") for k, m in enumerate(comps)])
        ids = list(range(first, len(self.groups)))
        self._score(ids, omit=True, tag="core audit")
        conflict = 0
        for i in ids:
            g = self.groups[i]
            om = self.omit[i]
            if np.isfinite(om).any():
                g.stats.update(omit_minus_current=float(np.nansum(om)), alt_minus_current=float(g.stats.get("Delta", 0.0)))
            if g.status == STATIC_S:
                conflict += 1
                g.reason = "core group: exact all-view evidence decisively prefers STATIC" + (
                    " (relabelled by the support revision)" if pol.core_policy == "revisable" else
                    " (upstream joint or support conflict; not relabelled)")
            self.core_audit.append({"gid": g.gid, "size": int(len(g.members)), "decision": g.status, "reason": g.reason})
        self.core_conflict = conflict > 0
        self.log(f"4A core audit: {len(ids)} core groups; decisions "
                 f"{ {s: sum(1 for c in self.core_audit if c['decision'] == s) for s in {c['decision'] for c in self.core_audit}} }"
                 + (f"; {conflict} groups prefer STATIC -> " + ("support revision" if pol.core_policy == "revisable"
                                                                else "integrity gate fails") if conflict else ""))

    def _status_counts(self):
        c = {}
        for i in self._leaves():
            c[self.groups[i].status] = c.get(self.groups[i].status, 0) + 1
        return c

    def _sensitivity(self):
        """Globally coherent joint perturbations: scene objective and the least decisive decided groups."""
        pol, log, ev = self.pol, self.log, self.ev
        dec = [i for i in self._leaves() if self.groups[i].status in (MOVING_S, STATIC_S)]
        dec.sort(key=lambda i: (self.groups[i].stats.get("consistency", 0), self.groups[i].gid))
        rep = dec[: pol.sens_groups]
        log(f"4A joint sensitivity: {len(perturbed_joints(self.scene.joint, pol))} shared perturbations x "
            f"{ev.V} views, scene objective + {len(rep)} least-decisive groups (declared set)")
        out = {"joint_confidence": "JOINT_CONFIDENCE_UNCALIBRATED", "perturbations": {}, "groups": [int(self.groups[i].gid) for i in rep],
               "E_nominal": float(self.E_after.sum())}
        for name, j in perturbed_joints(self.scene.joint, pol):
            tr = JointTransport(j, self.scene.device)
            ev._fp.pop("sens", None)
            gS, gM = ev.group_rects([self.groups[i] for i in rep], key="sens", transport=tr) if rep else (None, None)
            if rep:
                r = ev.sweep([[self.groups[i]] for i in rep], self.labels, (gS, gM), transport=tr, key="sens", tag=f"sens {name}")
            else:
                r = ev.sweep([], self.labels, (np.zeros((0, len(ev.cams), 4)), np.zeros((0, ev.V, 4))), transport=tr,
                             key="sens", tag=f"sens {name}")
            out["perturbations"][name] = {
                "E": float(r["base_totals"].sum()), "E_minus_nominal": float(r["base_totals"].sum() - out["E_nominal"]),
                "group_Delta": {int(self.groups[i].gid): float((1.0 if self._label(i) else -1.0) * r["d_alt"][k].sum())
                                for k, i in enumerate(rep)}}
        ev._fp.pop("sens", None)
        for k, i in enumerate(rep):
            g = self.groups[i]
            vals = [p["group_Delta"][int(g.gid)] for p in out["perturbations"].values()]
            g.stats["sensitivity"] = {"min": min(vals), "max": max(vals),
                                      "sign_stable": bool(all(np.sign(v) == np.sign(g.stats["Delta"]) for v in vals))}
        return out

    # ------------------------------------------------------------------ result
    def result(self) -> dict:
        N = self.N
        status = np.full(N, UNTESTED, dtype=object)
        status[self.core] = CORE if not getattr(self, "aborted", False) else JOINT_INVALID
        if getattr(self, "aborted", False):
            status[:] = JOINT_INVALID
        group_of = np.full(N, -1, np.int64)
        for i, g in enumerate(self.groups):
            if g.kind == "leaf" and g.origin == "core":
                status[g.members] = CORE if self._label(i) else STATIC_S
                group_of[g.members] = g.gid
            elif g.kind == "leaf":
                status[g.members] = g.status
                group_of[g.members] = g.gid
            elif g.kind == "core_audit":
                group_of[g.members] = g.gid
                if g.status == STATIC_S:
                    status[g.members] = JOINT_INVALID
        return {"aborted": getattr(self, "aborted", False), "complete": self.complete, "stop_reason": self.stop_reason,
                "status": status, "render_label": self.labels.cpu().numpy().astype(np.int8),
                "upstream_provisional": self.provisional_upstream.cpu().numpy().astype(np.int8),
                "group_of": group_of, "groups": self.groups, "core_check": self.core_check,
                "history": self.history, "counters": self.ev.counters.to_dict(),
                "E_before": getattr(self, "E_before", None), "E_after": getattr(self, "E_after", None),
                "sensitivity": getattr(self, "sens", {}), "seconds": time.time() - self.t0,
                "core_audit": self.core_audit, "core_conflict": self.core_conflict,
                "core_revised": self.core_revised}


def write_outputs(out_dir, phase: Phase4A, res: dict, upstream_manifest: dict):
    d = Path(out_dir) / "support_completion"
    d.mkdir(parents=True, exist_ok=True)
    N = phase.N
    views = [{"camera": vr.cam.name, "frame": int(vr.frame)} for vr in phase.scene.views]
    np.savez_compressed(d / "material_status.npz", status_code=np.array([STATUS_CODES.get(s, 3) for s in res["status"]], np.int8),
                        legend=json.dumps(STATUS_CODES), original_id=np.arange(N, dtype=np.int64), group_of=res["group_of"])
    np.savez_compressed(d / "render_label.npz", render_label=res["render_label"], upstream_provisional=res["upstream_provisional"],
                        legend=json.dumps({"0": "STATIC pose", "1": "MOVING pose"}),
                        note="the pose each Gaussian is RENDERED with after 4A; for UNRESOLVED / UNTESTED / SUSPECT rows "
                             "this is a provisional rendering choice, not a decision (see material_status)")
    np.savez_compressed(d / "group_evidence.npz", delta=phase.delta, changed=phase.changed, skip=phase.skip,
                        unc_new=phase.unc_new, omit=phase.omit, channels=json.dumps(CHANNELS),
                        skip_legend=json.dumps({"0": "scored", "1": "offscreen/culled in both poses",
                                                "2": "identity joint transform (S == M)", "3": "not evaluated"}),
                        views=json.dumps(views), gid=np.array([g.gid for g in phase.groups]))
    (d / "groups.json").write_text(json.dumps([{"gid": g.gid, "parent": g.parent, "depth": g.depth, "kind": g.kind,
                                                "size": int(len(g.members)), "status": g.status, "reason": g.reason,
                                                "render_label": int(phase._label(i)) if len(g.members) else None,
                                                "stats": g.stats, "members": g.members.tolist()}
                                               for i, g in enumerate(phase.groups)], default=float))
    counts = {s: int((res["status"] == s).sum()) for s in STATUS_CODES}
    atomic_json(d / "score_manifest.json", {
        "scoring": "exact: full native RGB-D render per hypothesis, loss difference accumulated over the conservative "
                   "kernel-tile footprint of the group in both poses dilated by the SSIM radius; pixels outside are "
                   "bitwise identical (tests/test_phase4_exact_roi_parity.py)",
        "objective": "recovery/exact_objective.py L_v; fixed full-view denominators 3HW / HW / |V_observed|",
        "loss": phase.cfg.to_dict(), "policy": phase.pol.to_dict(), "skips_allowed": ["offscreen/culled in both poses",
                                                                                   "identity joint transform"],
        "statistic": "Delta = E_S - E_M summed over ALL views (positive favours MOVING)",
        "decision_policy": "DECLARED provisional thresholds (decisive_abs_px, decisive_rel, favour_frac over views "
                           "and time blocks); not calibrated probabilities",
        "validation": "conditional_on_phase3_joint: Phase 3 used the same frames; no parameter is fitted in 4A",
        "complete": res["complete"], "stop_reason": res["stop_reason"], "counts": counts,
        "core_check": res["core_check"], "history": res["history"], "counters": res["counters"]})
    if res["E_before"] is not None:
        atomic_json(d / "full_scene_audit.json", {
            "views": views, "channels": list(CHANNELS), "upstream": upstream_manifest,
            "before_4A": np.asarray(res["E_before"]).tolist(),
            "after_4A": None if res["E_after"] is None else np.asarray(res["E_after"]).tolist(),
            "E_before": float(np.sum(res["E_before"])),
            "E_after": None if res["E_after"] is None else float(np.sum(res["E_after"]))})
    atomic_json(d / "joint_sensitivity.json", res["sensitivity"] or {"joint_confidence": "JOINT_CONFIDENCE_UNCALIBRATED",
                                                                     "note": "sensitivity disabled"})
    return d
