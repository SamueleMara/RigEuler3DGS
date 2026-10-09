"""Section 12: per-frame, per-camera image debug panels.

CODEX_ADD_VISUAL_AND_NUMERICAL_DEBUG_GEOMETRIC_PIPELINE.md section 12.

One PNG per (frame, camera) holding, side by side:

    GT RGB | render before motion | after rigid | after joint refinement
    RGB residual | GT depth | rendered depth | depth residual | motion mask

Section 12 allows these to live outside the HTML when they are heavy, so they
are written to `render_debug/` and linked from the page. Every render here is
produced from the estimator's own state through the SAME rasterizer the
estimator used -- nothing is re-simulated.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def write_render_panels(dbg, objective, writer, *, frame_stride: int = 1,
                        verbose: bool = True) -> list:
    """Write section 12's panels. Returns the written relative paths."""
    if not dbg.enabled or objective is None:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = writer.dir / "render_debug"
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def img(t):
        """[H,W,3] uint8-ready array from a [3,H,W] float tensor."""
        return t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    stages = [("canonical", "before motion"), ("rigid", "after rigid"),
              ("refined", "after joint refinement")]
    todo = [(ti, f) for ti, f in enumerate(objective.frames)
            if ti % max(1, frame_stride) == 0]
    from tools.kpart_pipeline import _progress
    for ti, (obs, fid) in _progress(todo, "  §12 render panels", verbose):
        obs_d = objective._to_device(obs, dbg.mu0.device)
        rendered = {}
        for key, _label in stages:
            mu = dbg.positions_at(key, ti)
            if mu is None:
                continue
            R = dbg.orientations_at(key, ti)
            lin, _p, _r = objective.linearize_frame(mu, R, obs_d)
            rendered[key] = [(v.rgb.clone(), v.depth.clone(), v.alpha.clone())
                             for v in lin.views]
            del lin
        if not rendered:
            break
        for ci, ob in enumerate(obs_d):
            if ob is None:
                continue
            gt_rgb, gt_d, d_valid = ob
            panels = [("GT RGB", img(gt_rgb), None)]
            for key, label in stages:
                if key in rendered:
                    panels.append((f"RGB {label}", img(rendered[key][ci][0]),
                                   None))
            last = rendered.get("refined") or rendered.get("rigid") \
                or rendered.get("canonical")
            res = (gt_rgb - last[ci][0]).abs().mean(0)
            panels.append(("RGB residual |.|", res.cpu().numpy(), "magma"))
            panels.append(("GT depth", gt_d.cpu().numpy(), "viridis"))
            panels.append(("rendered depth", last[ci][1].cpu().numpy(),
                           "viridis"))
            dres = (gt_d - last[ci][1]).abs()
            dres = dres * (d_valid & (last[ci][2] > objective.cfg
                                      .min_alpha_for_depth)).to(dres.dtype)
            panels.append(("depth residual", dres.cpu().numpy(), "magma"))
            # Section 12's motion-sensitive mask: pixels the moving set covers.
            mask = (last[ci][2] > objective.cfg.min_alpha_for_depth)
            panels.append(("motion-sensitive mask",
                           mask.float().cpu().numpy(), "gray"))
            # Size the grid to the panels actually collected, so none is
            # silently dropped (the earlier fixed 2x4 lost the motion mask).
            ncol = (len(panels) + 1) // 2
            f, ax = plt.subplots(2, ncol, figsize=(3.0 * ncol, 5.2))
            ax = ax.reshape(-1)
            for a in ax:
                a.axis("off")
            for a, (title, data, cmap) in zip(ax, panels):
                a.imshow(data, cmap=cmap)
                a.set_title(title, fontsize=7)
            f.suptitle(f"frame {fid}  camera {ci}", fontsize=9)
            f.tight_layout()
            name = f"frame_{ti:04d}_cam_{ci:02d}.png"
            f.savefig(out / name, dpi=95)
            plt.close(f)
            written.append(f"render_debug/{name}")
    if verbose:
        print(f"  debug: {len(written)} render panels in render_debug/")
    writer.json("render_debug_index", {"note": "section 12", "files": written})
    return written
