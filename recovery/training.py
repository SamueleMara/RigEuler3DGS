"""Phase 5: Gaussian-only RGB-D-mask training with the Phase-3 joint FROZEN (spec section 7).

Modes
  gaussian_rgbd  optimise canonical _xyz, _rotation, _scaling, _opacity, _features_dc, _features_rest with the
                 full objective (masked RGB L1 + SSIM, valid-foreground metric depth L1, whole-mask alpha L1)
                 plus the FreeArt scale regulariser; stages A (geometry + opacity + DC warm-up at SH degree 0),
                 B (everything, SH progression, controlled densification / pruning), C (topology frozen,
                 colour polish; geometry also frozen with freeze_geometry_in_C)
  sh_only        ONLY _features_dc / _features_rest (stage C alone). Geometry and opacity are not registered
                 with any optimiser and are checked bitwise unchanged at the end. Depth and mask are computed for
                 every view as MONITORS: with fixed geometry and opacity they have no parameter to act on, and
                 they are not part of the optimised loss.

The joint is never a parameter: no joint tensor is registered with an optimiser (checked), the per-frame
(Q_f, p_f) are constant buffers built once from contracts.Joint, the Phase-3 geometry / trajectory files are
hashed at start, at every checkpoint and at the end, and copied byte for byte into final/joint/.

Sampling: a seeded permutation of ALL (camera, original frame) views per epoch (every view once before any
repeats). Validation sweeps cover every view (never a random subset).
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import topology as TP
from .articulated_scene import JointTransport, canonical_tensors, render_articulated
from .contracts import sha256
from .gaussians import PROV_NAMES, GaussianTensors
from .rgbd_loss import LossConfig, view_loss


@dataclass
class Phase5Config:
    mode: str = "gaussian_rgbd"              # gaussian_rgbd | sh_only
    iters_A: int = 500
    iters_B: int = 6000
    iters_C: int = 1000
    sh_degree_max: int = 3
    sh_interval: int = 1000                  # +1 SH degree every this many stage-B iterations
    sh_mode: str = "local_SH"
    position_lr_init: float = 1.6e-4         # x scene extent (3DGS defaults)
    position_lr_final: float = 1.6e-6
    feature_lr: float = 2.5e-3
    opacity_lr: float = 2.5e-2
    scaling_lr: float = 5e-3
    rotation_lr: float = 1e-3
    densify: bool = True
    densify_every: int = 100
    densify_from: int = 500                  # stage-B iteration
    densify_until: int = 4000
    densify_grad: float = 2e-4
    percent_dense: float = 0.01
    min_opacity: float = 0.005
    big_ws: float = 0.1
    max_gaussians: int = 400000
    scale_reg_ratio: float = 10.0            # FreeArt max_gauss_ratio
    scale_reg_weight: float = 1.0
    scale_reg_every: int = 10
    freeze_geometry_in_C: bool = False
    bg: tuple = (1.0, 1.0, 1.0)
    val_every: int = 0                       # 0 = only the initial and final full sweeps
    render_frames: int = 3                   # visual comparison: this many evenly spaced frames x every camera
    checkpoint_every: int = 0
    seed: int = 0
    loss: LossConfig = field(default_factory=LossConfig)

    def to_dict(self):
        d = asdict(self)
        d["loss"] = self.loss.to_dict()
        return d


def expon_lr(step, init, final, max_steps):
    if max_steps <= 0:
        return init
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(init) * (1 - t) + math.log(final) * t)


def scene_extent(cameras) -> float:
    C = np.stack([c.campos.cpu().numpy() for c in cameras])
    return float(np.linalg.norm(C - C.mean(0), axis=1).max() * 1.1) or 1.0


class ViewSampler:
    def __init__(self, n, seed):
        self.n, self.rng, self.order, self.k, self.epoch = n, np.random.default_rng(seed), [], 0, -1

    def next(self):
        if self.k >= len(self.order):
            self.order, self.k, self.epoch = self.rng.permutation(self.n), 0, self.epoch + 1
        i = int(self.order[self.k])
        self.k += 1
        return i


def joint_fingerprint(joint, transport: JointTransport) -> dict:
    files = {}
    for k in ("geometry_file", "trajectory_file"):
        p = Path(getattr(joint, k))
        files[k] = sha256(p) if p.is_file() else None
    h = __import__("hashlib").sha256()
    for f in joint.frames:
        ft = transport.at(f)
        for t in (ft.Q, ft.p, ft.qQ):
            h.update(t.detach().cpu().numpy().tobytes())
    return {"files": files, "transport_buffers_sha256": h.hexdigest(),
            "joint": {"type": joint.type, "axis": list(joint.axis), "point": joint.point, "q": list(joint.q)}}


def make_trainable(g: GaussianTensors, mode: str):
    """Leaves become nn.Parameters; only the mode's parameters require grad."""
    train = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"} if mode == "gaussian_rgbd" else {"f_dc", "f_rest"}
    for name, attr in TP.ATTR.items():
        t = getattr(g, attr).detach().clone()
        setattr(g, attr, nn.Parameter(t, requires_grad=name in train))
    return train


def build_optimizer(g, train, cfg: Phase5Config, extent):
    lr = {"xyz": cfg.position_lr_init * extent, "f_dc": cfg.feature_lr, "f_rest": cfg.feature_lr / 20.0,
          "opacity": cfg.opacity_lr, "scaling": cfg.scaling_lr, "rotation": cfg.rotation_lr}
    groups = [{"params": [getattr(g, TP.ATTR[n])], "lr": lr[n], "name": n} for n in TP.PARAMS if n in train]
    return torch.optim.Adam(groups, lr=0.0, eps=1e-15)


def set_trainable(g, opt, names):
    for gr in opt.param_groups:
        on = gr["name"] in names
        gr["params"][0].requires_grad_(on)
        gr["lr_scale"] = 1.0 if on else 0.0


def sweep(views, g, transport, moving, cfg, deg, get_view, device):
    """Every view once: metrics per (camera, frame) and the per-Gaussian 'seen' flag."""
    rows = []
    seen = torch.zeros(len(g), dtype=torch.bool, device=device)
    bg = torch.tensor(cfg.bg, device=device)
    with torch.no_grad():
        for vr in views:
            out = render_articulated(vr.cam, transport.at(vr.frame), g, moving, bg, cfg.sh_mode, deg)
            v = get_view(vr)
            t = view_loss(out, v, cfg.loss)
            vis = out["radii"] > 0
            seen |= vis
            rows.append({"camera": vr.cam.name, "frame": int(vr.frame),
                         **{k: (float(x) if torch.is_tensor(x) else x) for k, x in t.items()},
                         "visible_gaussians": int(vis.sum()),
                         "visible_moving_fraction": float((vis & moving).sum()) / max(1, int(moving.sum()))})
    return rows, seen


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def comparison_views(views, n_frames):
    """Every camera at n_frames evenly spaced frames (first / middle / last for 3)."""
    frames = sorted({int(v.frame) for v in views})
    if n_frames <= 0 or not frames:
        return []
    keep = {frames[k] for k in np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).round().astype(int)}
    return [i for i, v in enumerate(views) if int(v.frame) in keep]


def render_views(idx, views, g, transport, moving, cfg, deg, device):
    bg = torch.tensor(cfg.bg, device=device)
    res = {}
    with torch.no_grad():
        for i in idx:
            vr = views[i]
            o = render_articulated(vr.cam, transport.at(vr.frame), g, moving, bg, cfg.sh_mode, deg)
            res[i] = {"rgb": o["color"].clamp(0, 1).cpu(), "depth": o["depth"][0].cpu(), "alpha": o["alpha"][0].cpu()}
    return res


def _depth_rgb(d, valid, lo, hi):
    import matplotlib.cm as cm
    x = ((d - lo) / max(hi - lo, 1e-9)).clamp(0, 1).numpy()
    im = cm.turbo(x)[..., :3]
    im[~valid.numpy()] = 0.0
    return torch.as_tensor(im, dtype=torch.float32).permute(2, 0, 1)


def _err_rgb(e, vmax):
    import matplotlib.cm as cm
    return torch.as_tensor(cm.inferno((e / vmax).clamp(0, 1).numpy())[..., :3], dtype=torch.float32).permute(2, 0, 1)


def write_comparisons(out_dir, idx, views, init, final, get_view):
    """renders/<camera>_f<frame>.png, 2 rows x 4 columns:
         GT RGB | initial render | final render | |final - GT| RGB (0..0.25)
         GT depth | final depth | |final - GT| depth (0..5 cm, valid GT only) | mask (red) vs final alpha (green)
    plus renders/overview_<camera>.png stacking that camera's frames, and renders/index.json."""
    from PIL import Image, ImageDraw
    out = Path(out_dir) / "renders"
    out.mkdir(parents=True, exist_ok=True)
    index, rows_by_cam = [], {}
    for i in idx:
        vr = views[i]
        v = get_view(vr)
        gt, M = v.rgb.cpu().clamp(0, 1), v.mask.cpu()
        dv = v.depth_valid.cpu()
        D = v.depth.cpu()
        a, b = init[i], final[i]
        lo, hi = (float(D[dv].min()), float(D[dv].max())) if dv.any() else (0.0, 1.0)
        mf = M.float()
        err_rgb = ((b["rgb"] - gt).abs() * mf).mean(0)
        derr = torch.where(dv, (b["depth"] - D).abs(), torch.zeros_like(D))
        cover = b["alpha"] > 0.01
        mask_img = torch.stack([mf, b["alpha"].clamp(0, 1), torch.zeros_like(mf)])
        tiles = [[gt, a["rgb"], b["rgb"], _err_rgb(err_rgb, 0.25)],
                 [_depth_rgb(D, dv, lo, hi), _depth_rgb(b["depth"], cover, lo, hi), _err_rgb(derr, 0.05), mask_img]]
        grid = torch.cat([torch.cat(r, 2) for r in tiles], 1)
        im = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        dr = ImageDraw.Draw(im)
        W, H = gt.shape[2], gt.shape[1]
        labels = ["GT RGB", "initial", "final", "|err| RGB 0-0.25", "GT depth", "final depth", "|err| depth 0-5cm",
                  "mask R / alpha G"]
        for k, t in enumerate(labels):
            dr.text((k % 4 * W + 4, k // 4 * H + 4), t, fill=(255, 0, 255))
        dr.text((4, H - 14), f"{vr.cam.name} frame {int(vr.frame)}", fill=(255, 0, 255))
        f = out / f"{vr.cam.name}_f{int(vr.frame):04d}.png"
        im.save(f)
        rows_by_cam.setdefault(vr.cam.name, []).append(im)
        mse_i = float((((a["rgb"] - gt) ** 2) * mf).sum() / max(1, 3 * int(M.sum())))
        mse_f = float((((b["rgb"] - gt) ** 2) * mf).sum() / max(1, 3 * int(M.sum())))
        index.append({"camera": vr.cam.name, "frame": int(vr.frame), "file": f.name,
                      "psnr_fg_initial": -10 * math.log10(max(mse_i, 1e-12)),
                      "psnr_fg_final": -10 * math.log10(max(mse_f, 1e-12)),
                      "depth_mae_final_m": float(derr[dv].mean()) if dv.any() else None})
    for cam, ims in rows_by_cam.items():
        sheet = Image.new("RGB", (ims[0].width, sum(x.height for x in ims)))
        y = 0
        for x in ims:
            sheet.paste(x, (0, y))
            y += x.height
        sheet.save(out / f"overview_{cam}.png")
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


def train_phase5(g: GaussianTensors, joint, views, cameras, cfg: Phase5Config, out_dir=None, device="cuda",
                 log=print, manifest=None) -> dict:
    t0 = time.time()
    if cfg.mode not in ("gaussian_rgbd", "sh_only"):
        raise ValueError(cfg.mode)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    out = Path(out_dir) if out_dir else None
    if out:
        for d in ("checkpoints", "final", "renders"):
            (out / d).mkdir(parents=True, exist_ok=True)
    transport = JointTransport(joint, device)
    fp0 = joint_fingerprint(joint, transport)
    get_view = lambda vr: vr.get().to(device)
    moving_of = lambda: torch.as_tensor(g.lineage["material"] == 1, device=device)
    loss_cfg_opt = LossConfig(**cfg.loss.to_dict())
    if cfg.mode == "sh_only":                    # depth / mask: monitors only, never optimised terms
        loss_cfg_opt.use_depth = loss_cfg_opt.use_mask = False
    frozen0 = None
    train = make_trainable(g, cfg.mode)
    if cfg.mode == "sh_only":
        frozen0 = {n: getattr(g, TP.ATTR[n]).detach().clone() for n in ("xyz", "rotation", "scaling", "opacity")}
    extent = scene_extent(cameras)
    opt = build_optimizer(g, train, cfg, extent)
    joint_ids = {id(t) for f in joint.frames for t in (transport.at(f).Q, transport.at(f).p, transport.at(f).qQ)}
    assert not any(id(p) in joint_ids for gr in opt.param_groups for p in gr["params"]), "joint tensor in optimiser"
    deg_final = min(cfg.sh_degree_max, g.max_sh_degree)
    # renderer parity at the anchor: world_SH vs local_SH (identical by construction there) and per-frame gap
    parity = []
    bg = torch.tensor(cfg.bg, device=device)
    with torch.no_grad():
        for vr in views[:: max(1, len(views) // 8)]:
            a = render_articulated(vr.cam, transport.at(vr.frame), g, moving_of(), bg, "world_SH", deg_final)
            b = render_articulated(vr.cam, transport.at(vr.frame), g, moving_of(), bg, "local_SH", deg_final)
            parity.append({"camera": vr.cam.name, "frame": int(vr.frame), "anchor": vr.frame == joint.anchor_frame,
                           "world_vs_local_SH_mean_abs_rgb": float((a["color"] - b["color"]).abs().mean())})
    init_rows, _ = sweep(views, g, transport, moving_of(), cfg, deg_final if cfg.mode == "sh_only" else 0, get_view, device)
    cmp_idx = comparison_views(views, cfg.render_frames) if out else []
    init_renders = render_views(cmp_idx, views, g, transport, moving_of(), cfg,
                                deg_final if cfg.mode == "sh_only" else 0, device)
    log(f"Phase 5 [{cfg.mode}] {len(g)} Gaussians ({int(moving_of().sum())} moving), {len(views)} views, "
        f"initial mean loss {np.mean([r['total'] for r in init_rows]):.5f}")
    stages = [("C", cfg.iters_C)] if cfg.mode == "sh_only" else [("A", cfg.iters_A), ("B", cfg.iters_B), ("C", cfg.iters_C)]
    sampler = ViewSampler(len(views), cfg.seed)
    loss_rows, topo_log, it = [], [], 0
    grad_acc = torch.zeros(len(g), device=device)
    denom = torch.zeros(len(g), device=device)
    # 'seen' over COMPLETE sweeps, tracked by persistent uid so densify / prune cannot misalign it
    seen_cur = torch.zeros(len(g), dtype=torch.bool, device=device)
    seen_last_uids = None                         # uids rasterized in the last complete epoch
    last_epoch = 0

    def remap(mask_uids_src, mask):
        u = g.lineage["gaussian_uid"]
        return torch.as_tensor(np.isin(u, mask_uids_src[mask.cpu().numpy()]), device=device)
    for stage, n_it in stages:
        if cfg.mode == "gaussian_rgbd":
            names = {"A": {"xyz", "rotation", "scaling", "opacity", "f_dc"},
                     "B": set(TP.PARAMS),
                     "C": {"f_dc", "f_rest"} | (set() if cfg.freeze_geometry_in_C else {"xyz", "rotation", "scaling", "opacity"})}[stage]
        else:
            names = {"f_dc", "f_rest"}
        set_trainable(g, opt, names)
        for k in range(n_it):
            deg = 0 if stage == "A" else (min(deg_final, k // max(1, cfg.sh_interval)) if stage == "B" else deg_final)
            if cfg.mode == "sh_only":
                deg = deg_final
            for gr in opt.param_groups:
                base = {"xyz": expon_lr(it, cfg.position_lr_init * extent, cfg.position_lr_final * extent,
                                        cfg.iters_A + cfg.iters_B + cfg.iters_C),
                        "f_dc": cfg.feature_lr, "f_rest": cfg.feature_lr / 20.0, "opacity": cfg.opacity_lr,
                        "scaling": cfg.scaling_lr, "rotation": cfg.rotation_lr}[gr["name"]]
                gr["lr"] = base * gr.get("lr_scale", 1.0)
            i = sampler.next()
            if sampler.epoch > last_epoch:            # a full sweep of every view just completed
                seen_last_uids = g.lineage["gaussian_uid"][seen_cur.cpu().numpy()]
                seen_cur = torch.zeros(len(g), dtype=torch.bool, device=device)
                last_epoch = sampler.epoch
            vr = views[i]
            mv = moving_of()
            o = render_articulated(vr.cam, transport.at(vr.frame), g, mv, bg, cfg.sh_mode, deg)
            terms = view_loss(o, get_view(vr), loss_cfg_opt)
            loss = terms["total"]
            if cfg.mode == "gaussian_rgbd" and cfg.scale_reg_weight > 0 and it % cfg.scale_reg_every == 0:
                s = torch.exp(g._scaling)
                reg = (torch.clamp_min(s.max(1).values / s.min(1).values, cfg.scale_reg_ratio) - cfg.scale_reg_ratio).mean()
                loss = loss + cfg.scale_reg_weight * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                vis = o["radii"] > 0
                seen_cur |= vis
                if o["viewspace"].grad is not None:
                    grad_acc[vis] += o["viewspace"].grad[vis, :2].norm(dim=-1)
                    denom[vis] += 1
            loss_rows.append({"iteration": it, "stage": stage, "camera": vr.cam.name, "frame": int(vr.frame),
                              "sh_degree": deg, "gaussians": len(g),
                              **{kk: (float(x) if torch.is_tensor(x) else x) for kk, x in terms.items()}})
            it += 1
            if stage == "B" and cfg.densify and cfg.densify_from <= k <= cfg.densify_until and k % cfg.densify_every == 0 \
                    and k > 0:
                gn = grad_acc / denom.clamp_min(1)
                rec = {"iteration": it}
                uids_cur = g.lineage["gaussian_uid"].copy()
                if len(g) < cfg.max_gaussians:
                    rec.update(TP.densify(g, opt, gn, cfg.densify_grad, extent, cfg.percent_dense, generator=gen))
                if seen_last_uids is not None:     # prune only on the evidence of a COMPLETE sweep
                    seen = torch.as_tensor(np.isin(g.lineage["gaussian_uid"], seen_last_uids), device=device)
                    rec.update(TP.prune(g, opt, seen, cfg.min_opacity, cfg.big_ws, extent))
                seen_cur = remap(uids_cur, seen_cur)
                topo_log.append(rec)
                grad_acc = torch.zeros(len(g), device=device)
                denom = torch.zeros(len(g), device=device)
            if cfg.checkpoint_every and it % cfg.checkpoint_every == 0 and out:
                assert joint_fingerprint(joint, transport) == fp0, "joint changed"
                g.save_ply(out / "checkpoints" / f"iter_{it:06d}" / "point_cloud.ply")
                g.save_lineage(out / "checkpoints" / f"iter_{it:06d}")
        log(f"  stage {stage}: {n_it} iterations, {len(g)} Gaussians, last-100 mean loss "
            f"{np.mean([r['total'] for r in loss_rows[-100:]]) if loss_rows else float('nan'):.5f}")
    final_rows, _ = sweep(views, g, transport, moving_of(), cfg, deg_final, get_view, device)
    if cmp_idx:
        comparisons = write_comparisons(out, cmp_idx, views, init_renders,
                                        render_views(cmp_idx, views, g, transport, moving_of(), cfg, deg_final, device),
                                        get_view)
        log(f"  visual comparison: {len(comparisons)} panels + per-camera overviews -> {out / 'renders'}")
    fp1 = joint_fingerprint(joint, transport)
    if fp1 != fp0:
        raise RuntimeError("the joint changed during Phase 5")
    unchanged = None
    if frozen0 is not None:
        unchanged = {n: bool(torch.equal(frozen0[n], getattr(g, TP.ATTR[n]).detach())) for n in frozen0}
        if not all(unchanged.values()):
            raise RuntimeError(f"sh_only changed frozen tensors: {unchanged}")
    res = {"initial": init_rows, "final": final_rows, "loss_log": loss_rows, "topology": topo_log,
           "parity": parity, "joint_fingerprint": fp0, "sh_only_frozen_unchanged": unchanged,
           "seconds": time.time() - t0,
           "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None}
    if out:
        write_csv(out / "loss_log.csv", loss_rows)
        write_csv(out / "frame_camera_metrics.csv",
                  [{"sweep": "initial", **r} for r in init_rows] + [{"sweep": "final", **r} for r in final_rows])
        (out / "renderer_parity.json").write_text(json.dumps({"world_vs_local_SH": parity,
            "note": "equal at the anchor by construction; the gap elsewhere is the initial colour discrepancy "
                    "of world-frame SH on moving Gaussians"}, indent=1))
        (out / "config_resolved.json").write_text(json.dumps(cfg.to_dict(), indent=1))
        g.save_ply(out / "final" / "point_cloud.ply")
        g.save_lineage(out / "final")
        (out / "final" / "appearance_settings.json").write_text(json.dumps(
            {"sh_mode": cfg.sh_mode, "active_sh_degree": deg_final, "background": cfg.bg,
             "antialiasing": False}, indent=1))
        jd = out / "final" / "joint"
        jd.mkdir(parents=True, exist_ok=True)
        copies = {}
        for k in ("geometry_file", "trajectory_file"):
            p = Path(getattr(joint, k))
            if p.is_file():
                shutil.copy2(p, jd / p.name)
                copies[k] = {"source": str(p), "copy": str(jd / p.name), "sha256_source": sha256(p),
                             "sha256_copy": sha256(jd / p.name)}
        prov = g.lineage["provenance"]
        (out / "final" / "joint_unchanged_proof.json").write_text(json.dumps({
            "joint_optimized": False, "joint_parameters_registered_with_optimizer": 0,
            "fingerprint_before": fp0, "fingerprint_after": fp1, "identical": fp0 == fp1, "copies": copies,
            "optional_final_joint_refinement": "NOT_RUN_BY_DESIGN"}, indent=1, default=str))
        (out / "manifest.json").write_text(json.dumps({
            "mode": cfg.mode, "upstream": manifest or {}, "final_gaussians": len(g),
            "by_provenance": {PROV_NAMES[int(k)]: int((prov == k).sum()) for k in np.unique(prov)},
            "moving": int(g.lineage["material"].sum()), "seconds": res["seconds"],
            "peak_gpu_memory_mb": res["peak_gpu_memory_mb"], "topology_events": topo_log,
            "mean_initial": {k: float(np.mean([r[k] for r in init_rows])) for k in
                             ("total", "psnr_fg", "depth_l1", "mask_l1", "alpha_gap_fg", "false_alpha_bg")},
            "mean_final": {k: float(np.mean([r[k] for r in final_rows])) for k in
                           ("total", "psnr_fg", "depth_l1", "mask_l1", "alpha_gap_fg", "false_alpha_bg")}},
            indent=1, default=str))
    return res
