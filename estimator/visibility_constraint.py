"""Negative-mask-ray visible-opacity constraint for the RGB-D IEKFs (opt-in).

Theory: Bidirectional_Articulated_3DGS_IEKF_Background_Visibility_Updated
(19 Oct addendum), sections "Distinguish object appearance, free rays and
unknown/occluded rays", "Visible object alpha", "The principled constraint".
Spec: CODEX_IMPLEMENT_ANALYTIC_BACKGROUND_VISIBILITY_CONSTRAINED_IEKF_2026-10-19.md.

Ray classes, fixed per frame from the recorded mask M and depth only:
  * M = 1                     object evidence: RGB-D likelihood (object-valid rays)
  * negative, trusted-full-ray        constraint on the full accumulated alpha
  * negative, known-nonobject-depth   constraint on the alpha of contributors in
                                      front of the measured surface, z_obs - delta_z
  * negative, unknown                 no claim (counted)
"negative" excludes a declared boundary band around the mask (mask/rasterizer
fidelity). The classification never looks at the model: no normalized
composited depth, no current-render gate.

Hard constraint on every trusted negative ray p:  A_vis_p(x) <= epsilon.
Rays already violated at the frame's starting state are not grandfathered
silently: they may never get worse, and are restored only when restoring them
is consistent with the RGB-D (constrained_qp.data_conflict_test); otherwise
the frame reports MODEL_OWNERSHIP_INFEASIBLE for them. Rays no state variable
can move (zero Jacobian rows) are reported INDISTINGUISHABLE. Every accepted
step passes a full nonlinear audit of all trusted negative rays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

SEMANTICS = ("trusted-full-ray", "known-nonobject-depth", "conservative")
MODE_NONE, MODE_FULL, MODE_FRONT = 0, 1, 2


@dataclass
class MaskPolicy:
    semantics: str
    epsilon: float                 # tolerated visible object opacity on a negative ray
    boundary_px: int               # negative pixels within this distance of the mask carry no claim
    depth_noise_m: float | None    # declared sensor depth sigma (required for known-nonobject-depth)
    delta_z_sigmas: float = 3.0    # delta_z = delta_z_sigmas * depth_noise_m
    audit_tol: float = 1e-3        # nonlinear audit tolerance on A_vis (float32 render)

    def __post_init__(self):
        if self.semantics not in SEMANTICS:
            raise ValueError(f"mask-ray semantics {self.semantics!r} not in {SEMANTICS}")
        if self.semantics == "known-nonobject-depth" and (self.depth_noise_m is None or self.depth_noise_m <= 0):
            raise SystemExit("--mask-ray-semantics known-nonobject-depth needs a declared --mask-depth-noise-m "
                             "(the sensor depth sigma); it is not invented here")
        if not (0.0 <= self.epsilon < 1.0):
            raise ValueError("epsilon must be in [0, 1)")

    @property
    def delta_z(self) -> float:
        return float(self.delta_z_sigmas * self.depth_noise_m) if self.semantics == "known-nonobject-depth" else 0.0

    def to_dict(self) -> dict:
        return {"semantics": self.semantics, "epsilon": self.epsilon, "boundary_px": self.boundary_px,
                "depth_noise_m": self.depth_noise_m, "delta_z_m": self.delta_z, "audit_tol": self.audit_tol,
                "epsilon_meaning": ("visible object opacity tolerated on a trusted negative ray beyond the "
                                    "boundary band: Gaussian tails below this are compatible with 'no object'"),
                "boundary_meaning": ("negative pixels within boundary_px of the binary mask carry no claim: "
                                     "binarization and splat antialiasing make the edge uncertain")}


def load_masks(mask_root: Path, cam_names, original_frames, height: int, width: int):
    """<mask_root>/<cam>/frame_NNNNNN.{png,jpg} as bool [H,W] CPU tensors, per (processing-order)
    frame and camera. mask_root is the mask directory itself (default <source>/mask)."""
    from PIL import Image
    out = []
    for f in original_frames:
        row = []
        for c in cam_names:
            cands = [Path(mask_root) / c / f"frame_{int(f):06d}{e}" for e in (".png", ".jpg", ".jpeg")]
            p = next((q for q in cands if q.is_file()), None)
            if p is None:
                raise SystemExit(f"object-mask constraint: no mask for {c} frame {int(f)} under {mask_root}")
            im = Image.open(p).convert("L")
            if im.size != (width, height):
                raise SystemExit(f"{p}: mask {im.size} but the camera is {(width, height)}")
            row.append(torch.from_numpy(np.asarray(im) > 127))
        out.append(row)
    return out


def ray_modes(mask: torch.Tensor, depth_obs: torch.Tensor, depth_valid, policy: MaskPolicy):
    """(ray_mode int32 [H,W], z_obs float [H,W], counts) on the mask's device."""
    dev = mask.device
    M = mask.bool()
    k = int(policy.boundary_px)
    near = (torch.nn.functional.max_pool2d(M[None, None].float(), 2 * k + 1, stride=1, padding=k)[0, 0] > 0
            if k > 0 else M)
    neg = ~near
    valid = depth_valid.to(dev).bool() if depth_valid is not None else depth_obs.to(dev) > 0
    mode = torch.zeros(M.shape, dtype=torch.int32, device=dev)
    z = torch.zeros(M.shape, dtype=torch.float32, device=dev)
    if policy.semantics == "trusted-full-ray":
        mode[neg] = MODE_FULL
    elif policy.semantics == "known-nonobject-depth":
        sel = neg & valid
        mode[sel] = MODE_FRONT
        z[sel] = depth_obs.to(dev).float()[sel]
    counts = {"object_pixels": int(M.sum()), "boundary_band_pixels": int((near & ~M).sum()),
              "negative_pixels": int(neg.sum()), "constrained_full_ray": int((mode == MODE_FULL).sum()),
              "constrained_front_of_depth": int((mode == MODE_FRONT).sum()),
              "negative_unknown": int((neg & (mode == MODE_NONE)).sum())}
    return mode, z, counts


class FrameVisibility:
    """The fixed negative rays of ONE frame, all cameras, and their operators."""

    def __init__(self, masks, depths, valids, policy: MaskPolicy, device="cuda"):
        if not (len(masks) == len(depths) == len(valids)):
            raise AssertionError(f"one mask / depth / validity per camera: {len(masks)}, {len(depths)}, {len(valids)}")
        self.policy = policy
        self.masks = [m.to(device) for m in masks]
        self.modes, self.zobs, self.counts = [], [], []
        for m, d, v in zip(self.masks, depths, valids):
            if tuple(m.shape) != tuple(d.shape[-2:]):
                raise AssertionError(f"mask {tuple(m.shape)} vs depth {tuple(d.shape)}")
            md, z, c = ray_modes(m, d.to(device), None if v is None else v.to(device), policy)
            self.modes.append(md)
            self.zobs.append(z)
            self.counts.append(c)
        self.rows = [torch.nonzero(md.reshape(-1) > 0).reshape(-1) for md in self.modes]
        self.ncam = len(self.masks)
        # rays released from the hard set in this frame because holding them
        # contradicted the RGB-D (constrained_qp.data_conflict_test)
        self.released = None
        self.aggregate_released = False      # same, for the pre-existing-total row
        self._qp_cache = None                # (lin, A, {ray -> (c_i, A^-1 c_i)}) for one linearization

    def qp_cache(self, lin, A) -> dict:
        """Rows (C^T e_i, A^-1 C^T e_i) already solved at THIS linearization and THIS
        A, keyed by global ray index; any other (lin, A) starts a new cache."""
        c = self._qp_cache
        if c is None or c[0] is not lin or c[1] is not A:
            self._qp_cache = (lin, A, {})
        return self._qp_cache[2]

    def check_views(self, lin) -> int:
        n = len(lin.views)
        if n != self.ncam:
            raise AssertionError(f"{n} rendered views but {self.ncam} cameras of masks: refusing a silent zip truncation")
        for v, m in zip(lin.views, self.masks):
            if tuple(v.alpha.shape) != tuple(m.shape):
                raise AssertionError(f"view {tuple(v.alpha.shape)} vs mask {tuple(m.shape)}")
        return n

    # -- nonlinear values -----------------------------------------------------
    def evaluate(self, lin):
        """Per camera (A_vis, T_front, n_contrib, n_front, n_behind) at this linearization."""
        from diff_gaussian_rasterization import iekf_visible_alpha
        self.check_views(lin)
        return [iekf_visible_alpha(v.cache, md, z, self.policy.delta_z)
                for v, md, z in zip(lin.views, self.modes, self.zobs)]

    def row_values(self, vals) -> torch.Tensor:
        return torch.cat([va[0].reshape(-1)[r] for va, r in zip(vals, self.rows)]) if self.ncam else torch.zeros(0)

    # -- linear operators on the stacked row vector ---------------------------
    def operators(self, lin, vals):
        """(C_pose, CT_pose): [N,6] pose tangent <-> stacked constrained rows."""
        from diff_gaussian_rasterization import iekf_vis_apply_H, iekf_vis_apply_HT
        shapes = [md.shape for md in self.modes]
        sizes = [int(r.numel()) for r in self.rows]

        def C_pose(dpose):
            out = []
            for v, md, z, r in zip(lin.views, self.modes, self.zobs, self.rows):
                out.append(iekf_vis_apply_H(v.cache, md, z, self.policy.delta_z, dpose.float().contiguous()).reshape(-1)[r])
            return torch.cat(out).to(dpose.dtype)

        def CT_pose(nu):
            tot = None
            off = 0
            for v, md, z, r, va, sh, k in zip(lin.views, self.modes, self.zobs, self.rows, vals, shapes, sizes):
                za = torch.zeros(sh[0] * sh[1], device=nu.device, dtype=torch.float32)
                za[r] = nu[off:off + k].float()
                off += k
                g = iekf_vis_apply_HT(v.cache, md, z, self.policy.delta_z, za.reshape(sh), va[1], va[2])
                tot = g if tot is None else tot + g
            return tot
        return C_pose, CT_pose

    # -- classification at the frame's starting state and the audit ----------
    def bounds(self, a_ref: torch.Tensor, a_cur: torch.Tensor) -> torch.Tensor:
        """Upper bounds u on the linearized change C dx <= u for the HARD rows:
        rays feasible at the frame's reference stay A_vis <= eps."""
        return self.policy.epsilon - a_cur

    def hard_rows(self, a_ref: torch.Tensor) -> torch.Tensor:
        h = a_ref <= self.policy.epsilon
        return h if self.released is None else h & ~self.released

    def audit(self, a_ref: torch.Tensor, a_cur: torch.Tensor, a_new: torch.Tensor) -> dict:
        """Full nonlinear audit of every trusted negative ray at a trial state.

        (i) per ray: no ray feasible at the frame's reference may exceed eps
        (+ audit_tol), and (ii) the total violation over the rays already
        violated at the reference, h = sum (A_vis - eps)_+, may not increase
        relative to the current accepted state (restoration filter)."""
        eps, tol = self.policy.epsilon, self.policy.audit_tol
        feasible_ref = a_ref <= eps
        hard = self.hard_rows(a_ref)
        bad_new = hard & (a_new > eps + tol)
        rel = self.released if self.released is not None else torch.zeros_like(feasible_ref)
        h = lambda a: float((a[~feasible_ref] - eps).clamp_min(0).double().sum())
        h_cur, h_new = h(a_cur), h(a_new)
        h_ok = self.aggregate_released or h_new <= h_cur * (1.0 + 1e-6) + tol
        return {"rays": int(a_ref.numel()), "violating_now": int((a_new > eps + tol).sum()),
                "new_violations": int(bad_new.sum()),
                "released_rays": int(rel.sum()), "released_rays_violating": int((rel & (a_new > eps + tol)).sum()),
                "aggregate_released": bool(self.aggregate_released),
                "preexisting_total_violation_before": h_cur, "preexisting_total_violation_after": h_new,
                "max_violation": float((a_new - eps).clamp_min(0).max()) if a_new.numel() else 0.0,
                "mean_violation": float((a_new - eps).clamp_min(0).mean()) if a_new.numel() else 0.0,
                "ok": bool(int(bad_new.sum()) == 0 and h_ok)}

    def coverage(self) -> dict:
        tot = {}
        for c in self.counts:
            for k, v in c.items():
                tot[k] = tot.get(k, 0) + v
        return tot


@dataclass
class SequenceVisibility:
    """Per-frame FrameVisibility in processing order, from the recorded masks and depth."""
    policy: MaskPolicy
    masks: list                 # [frame][camera] bool CPU
    depths: list                # [frame][camera] float CPU
    valids: list                # [frame][camera] bool CPU or None
    report: list = field(default_factory=list)

    def for_frame(self, k: int, device="cuda") -> FrameVisibility:
        return FrameVisibility(self.masks[k], self.depths[k], self.valids[k], self.policy, device)

    @classmethod
    def from_frames(cls, policy, mask_root, frames_raw, frame_ids):
        """masks from mask_root (<source>/mask by default), depth/validity from the SAME
        loaded RGB-D frames."""
        cams = list(frame_ids.get("cameras") or [])
        orig = [int(f["original_frame_index"]) if isinstance(f, dict) and "original_frame_index" in f
                else int(f["dataset_frame_index"]) if isinstance(f, dict) else int(f) for f in frame_ids["frames"]]
        ncam = len(frames_raw[0][0])
        if len(cams) < ncam:
            raise AssertionError(f"{len(cams)} camera names for {ncam} observed cameras")
        rgb0 = frames_raw[0][0][0][0]
        H, W = int(rgb0.shape[-2]), int(rgb0.shape[-1])
        if len(orig) != len(frames_raw):
            raise AssertionError(f"{len(orig)} frame ids for {len(frames_raw)} frames")
        masks = load_masks(Path(mask_root), cams[:ncam], orig, H, W)
        depths = [[o[1].cpu() for o in ob] for (ob, _s, _d) in frames_raw]
        valids = [[(o[2].cpu() if len(o) > 2 and o[2] is not None else None) for o in ob] for (ob, _s, _d) in frames_raw]
        return cls(policy, masks, depths, valids)


def add_cli(ap) -> None:
    ap.add_argument("--object-mask-constraint", choices=("off", "visible-negative-hard"), default="off",
                    help="hard visible-opacity constraint on trusted negative mask rays (default off)")
    ap.add_argument("--mask-ray-semantics", choices=SEMANTICS, default="trusted-full-ray",
                    help="trusted-full-ray: no object anywhere on a negative ray (synthetic, no occluder); "
                         "known-nonobject-depth: no object in front of the measured surface (needs "
                         "--mask-depth-noise-m); conservative: count negative rays, claim nothing")
    ap.add_argument("--mask-source", default=None, help="mask root (default <source>/mask)")
    ap.add_argument("--mask-depth-noise-m", type=float, default=None,
                    help="declared sensor depth sigma [m]; delta_z = 3 sigma (known-nonobject-depth only)")
    ap.add_argument("--mask-boundary-px", type=int, default=2,
                    help="negative pixels within this many pixels of the mask carry no claim")
    ap.add_argument("--mask-alpha-epsilon", type=float, default=0.05,
                    help="tolerated visible object opacity on a trusted negative ray")


def from_args(args, source, frames_raw, frame_ids):
    if getattr(args, "object_mask_constraint", "off") == "off":
        return None
    pol = MaskPolicy(args.mask_ray_semantics, args.mask_alpha_epsilon, args.mask_boundary_px,
                     args.mask_depth_noise_m)
    root = Path(args.mask_source) if getattr(args, "mask_source", None) else Path(source) / "mask"
    seq = SequenceVisibility.from_frames(pol, root, frames_raw, frame_ids)
    seq.mask_root = str(root)
    print(f"  object-mask constraint: masks {root}")
    print(f"  object-mask constraint: {pol.semantics}, eps {pol.epsilon}, boundary {pol.boundary_px} px, "
          f"delta_z {pol.delta_z:.3f} m, {len(seq.masks)} frames x {len(seq.masks[0])} cameras")
    return seq


def _lifted_rows(vis, lin, vals, a_cur, lift, liftT, like, rows=None):
    """C and C^T restricted to `rows` (bool over all rays) that the state can move."""
    from .constrained_qp import zero_rows
    C_pose, CT_pose = vis.operators(lin, vals)
    m = int(a_cur.numel())

    def C_all(v):
        return C_pose(lift(v))
    zr = zero_rows(C_all, like)
    sel = ~zr if rows is None else (~zr & rows)
    idx = torch.nonzero(sel).reshape(-1)

    def C_k(v):
        return C_all(v)[idx]

    def CT_k(nu):
        full = torch.zeros(m, device=nu.device, dtype=nu.dtype)
        full[idx] = nu
        return liftT(CT_pose(full))
    return C_k, CT_k, idx, zr


def constrained_step(vis: FrameVisibility, lin, vals, a_ref, a_cur, A_solve, b, lift, liftT,
                     step_u, radius=None, metric=None, dim_state=None, M_inv=None, margin=None, _depth=0,
                     release_aggregate=True):
    """The constrained IEKF correction at one linearization (any stage).

    lift: solved-state tangent -> per-Gaussian pose tangent [N,6] (the stage's
    J_g); liftT its transpose. step_u is the unconstrained trust-region step the
    stage already computed. The returned step is its A-norm projection onto the
    linearized constraints (constrained_qp.solve_qp_alm with b = A step_u):
        min 1/2 (d - d_tr)^T A (d - d_tr)   s.t.
            C_p d <= eps - a_p - margin_p     for every ray feasible at the frame's
                                              reference (HARD rows, C = J_A J_g)
            sum_{p violated now} C_p d <= 0   first-order: the total of the
                                              pre-existing violations may not grow
    The projection stays next to the step the trust region already trusts; with
    d_tr inside the radius and the feasible set convex around d = 0 (u >= 0 at
    an accepted state) a final scaling into the radius keeps it feasible.
    `margin` (second-order correction) tightens rows whose nonlinear value
    overshot the linear prediction at a rejected trial.
    release_aggregate: whether a data conflict may also release the aggregate
    row. True for the grouped stages (Phase 2/3), where one misowned Gaussian on
    a pre-existing violation can hold a whole part back. False for the
    free-Gaussian Phase 1: there the unconstrained step that the conflict test
    compares against is itself the background fit (object opacity spread onto
    negative rays), so a large excess is the case the constraint is meant to stop,
    and releasing the row lets the pre-existing violation grow.

    Returns (step, info, predicted_change) with predicted_change the linear
    model's change of every ray (NaN on rows the state cannot move)."""
    from .constrained_qp import solve_qp_projection
    eps = vis.policy.epsilon
    hard = vis.hard_rows(a_ref)
    C_k, CT_k, idx, zr = _lifted_rows(vis, lin, vals, a_cur, lift, liftT, b, rows=hard)
    C_pose, CT_pose = vis.operators(lin, vals)
    m = int(a_cur.numel())
    viol_now = (~hard) & (a_cur > eps)
    c_agg = None
    if bool(viol_now.any()) and not vis.aggregate_released:
        c_agg = liftT(CT_pose(viol_now.to(a_cur.dtype)))
        if float(c_agg.norm()) == 0.0:
            c_agg = None
    info = {"rays": m, "hard_rows": int(hard.sum()), "hard_rows_movable": int(idx.numel()),
            "zero_jacobian_rows": int(zr.sum()),
            "indistinguishable_violations": int((zr & (a_cur > eps)).sum()),
            "violated_at_reference": int((~hard).sum()), "aggregate_row": c_agg is not None}
    u = (eps - a_cur)[idx]
    if margin is not None:
        u = u - margin[idx]

    def C_ext(v):
        out = C_k(v)
        if c_agg is None:
            return out
        return torch.cat([out, (c_agg * v).sum().reshape(1).to(out.device, out.dtype)])

    def CT_ext(nu):
        if c_agg is None:
            return CT_k(nu)
        return CT_k(nu[:-1]) + float(nu[-1]) * c_agg
    u_ext = torch.cat([u, torch.zeros(1, device=u.device, dtype=u.dtype)]) if c_agg is not None else u

    def predicted(d):
        full = torch.full((m,), float("nan"), device=a_cur.device, dtype=a_cur.dtype)
        full[~zr] = C_pose(lift(d))[~zr]
        return full
    if bool((C_ext(step_u) <= u_ext + 1e-7).all()):
        info.update({"status": "INACTIVE", "active": 0})
        return step_u, info, predicted(step_u)
    # each A^-1 c_i to the accuracy of the trust-region step it projects (pcg_tol 1e-4)
    res = solve_qp_projection(A_solve, step_u, C_ext, CT_ext, u_ext, M_inv=M_inv, keys=idx,
                              tail_key=("aggregate", int(viol_now.sum())) if c_agg is not None else None,
                              cache=vis.qp_cache(lin, A_solve), z_cg_tol=1e-4, z_cg_iters=60)
    # (state-space quantities stay on the stage's device/dtype; ray-space ones on the render's)
    # Consistency with the RGB-D: the constrained step's model excess over the
    # trust-region step, against 1/2 chi2_99(k). A larger excess means holding the
    # active rays contradicts the data (wrong mask or wrong ownership, e.g. a
    # rigid part held back to hide one misowned Gaussian): release them for the
    # rest of the frame, report them, and do not move the state to hide them.
    from .constrained_qp import data_conflict_test
    q = (lambda d: 0.5 * float((d.double() * A_solve(d).double()).sum()) - float((b.double() * d.double()).sum()))
    nu_rows = res.nu[:-1] if c_agg is not None else res.nu
    test = data_conflict_test(q(res.x), q(step_u), int((nu_rows > 0).sum()), int(dim_state or b.numel()))
    info["data_conflict"] = test
    first_solve = margin is None or not bool((margin > 0).any())
    agg_active = release_aggregate and c_agg is not None and float(res.nu[-1]) > 0
    if test["conflict"] and (bool((nu_rows > 0).any()) or agg_active) and first_solve and _depth < 3:
        act = idx[(nu_rows > 0).to(idx.device)]
        if vis.released is None:
            vis.released = torch.zeros_like(hard)
        vis.released[act] = True
        if agg_active:
            vis.aggregate_released = True
        step2, info2, pred2 = constrained_step(vis, lin, vals, a_ref, a_cur, A_solve, b, lift, liftT,
                                               step_u, radius, metric, dim_state, M_inv, margin=margin,
                                               _depth=_depth + 1, release_aggregate=release_aggregate)
        info2["status"] = "MODEL_OWNERSHIP_INFEASIBLE"
        info2["released_now"] = int(act.numel())
        info2["aggregate_released_now"] = bool(agg_active)
        info2["data_conflict_released"] = test
        return step2, info2, pred2
    step = res.x
    scale = 1.0
    if radius is not None and metric is not None:
        n = metric.norm(step)
        if n > radius:
            scale = radius / n
            step = step * scale
    info.update({"status": res.status, "active": res.active, "qp_rounds": res.outer_iterations,
                 "qp_rows": res.inner_iterations, "qp_cg": res.cg_iterations,
                 "qp_cg_seconds": res.seconds_in_A_solves,
                 "qp_max_violation_linear": res.max_violation, "trust_scale": scale,
                 "projection_distance_A": float(((step - step_u) * A_solve(step - step_u)).sum()) ** 0.5})
    return step, info, predicted(step)


def classify_persistent(vis: FrameVisibility, lin, vals, a_ref, a_cur, A_solve, b, lift, liftT,
                        dim_state=None, M_inv=None) -> dict:
    """Once per frame, for violations still present at the end: would restoring
    them (A_vis <= eps) be consistent with the RGB-D? One restoration QP at the
    last linearization and constrained_qp.data_conflict_test.
      MODEL_OWNERSHIP_INFEASIBLE  restoring contradicts the data (e.g. a rigid
                                  part would have to move to hide one Gaussian)
      UNRESOLVED_INFEASIBLE       a restoring correction exists locally but the
                                  data did not produce it; kept, not invented
      INDISTINGUISHABLE           no state variable moves the violating rays"""
    from .constrained_qp import cg, data_conflict_test, solve_qp_projection
    eps = vis.policy.epsilon
    C_k, CT_k, idx, zr = _lifted_rows(vis, lin, vals, a_cur, lift, liftT, b)
    viol = a_cur > eps + vis.policy.audit_tol
    movable = viol.clone(); movable[zr] = False
    out = {"violating": int(viol.sum()), "indistinguishable": int((viol & zr).sum()),
           "movable_violating": int(movable.sum())}
    if int(movable.sum()) == 0:
        out["status"] = "INDISTINGUISHABLE" if int(viol.sum()) else "FEASIBLE"
        return out
    q = (lambda d: 0.5 * float((d.double() * A_solve(d).double()).sum()) - float((b.double() * d.double()).sum()))
    u = (vis.policy.epsilon - a_cur)[idx]          # every movable ray restored to eps
    x_u, _it, _r = cg(A_solve, b, tol=1e-4, max_iters=60, M_inv=M_inv)   # the trust-region PCG accuracy
    res = solve_qp_projection(A_solve, x_u, C_k, CT_k, u, M_inv=M_inv, keys=idx, cache=vis.qp_cache(lin, A_solve),
                              z_cg_tol=1e-4, z_cg_iters=60)
    test = data_conflict_test(q(res.x), q(x_u), res.active, int(dim_state or b.numel()))
    out.update({"restoration_qp": res.status, "data_conflict": test,
                "status": ("MODEL_OWNERSHIP_INFEASIBLE" if (test["conflict"] or res.status != "OPTIMAL")
                           else "UNRESOLVED_INFEASIBLE")})
    return out
