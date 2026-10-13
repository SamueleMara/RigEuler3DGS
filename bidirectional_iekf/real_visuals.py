"""Visual and numerical comparison of real forward/backward/union Phase-1 states.

Section 6 of CODEX_PRIORITY_REAL_BIDIRECTIONAL_PHASE1_MERGE_AND_VISUALIZE.md.

Everything here RENDERS states that already exist; nothing is optimised. Each
variant is a callable `frame -> GaussianMotionState | None`, and `None` is
drawn as a labelled "NOT AVAILABLE" panel -- a missing state is never filled
in with a neighbouring frame or a different variant.

Conventions, fixed for every panel so panels are comparable:

  depth   the renderer's normalised depth D = Z/(A+eps) against the dataset's
          metric Z, on ONE metric colour scale for the whole report (2nd-98th
          percentile of valid real depth over all rendered frames)
  common  the comparison domain for every error statistic: real depth valid
          AND alpha > 0.05 (the production coverage gate) for EVERY variant
          rendered at that frame/camera. Own-support numbers are reported too
  alpha   rendered accumulated opacity, 0..1
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ALPHA_GATE = 0.05
DEPTH_AGREE_M = 0.06            # 3 x the production depth sigma a = 0.02 m
SOURCE_COLORS = {0: "#1f77b4", 1: "#ff7f0e"}    # initial blue, final orange


def _slug(label: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render(state, cameras):
    from estimator.renderer_bridge import build_linearization
    with torch.no_grad():
        lin = build_linearization(state, cameras)
    return [{"rgb": v.rgb.detach().float().cpu().clamp(0, 1).numpy(),
             "depth": v.depth.detach().float().cpu().numpy(),
             "alpha": v.alpha.detach().float().cpu().numpy()} for v in lin.views]


def subset(state, rows):
    import dataclasses
    rows = torch.as_tensor(rows, dtype=torch.long, device=state.mu.device)
    return dataclasses.replace(
        state, mu=state.mu[rows], R=state.R[rows], v=state.v[rows],
        omega=state.omega[rows], colors=state.colors[rows],
        opacity=state.opacity[rows], scales=state.scales[rows])


def _metrics(view, real, common):
    rgb_gt, depth_gt, valid = real
    out = {}
    err = np.abs(view["rgb"] - rgb_gt).mean(0)
    out["rgb_rmse_all"] = float(np.sqrt(((view["rgb"] - rgb_gt) ** 2).mean()))
    own = valid & (view["alpha"] > ALPHA_GATE)
    out["alpha_support_px"] = int((view["alpha"] > ALPHA_GATE).sum())
    out["depth_support_px"] = int(own.sum())
    for name, mask in (("common", common), ("own", own)):
        if mask.any():
            d = view["depth"][mask] - depth_gt[mask]
            out[f"depth_mae_{name}_m"] = float(np.abs(d).mean())
            out[f"depth_rmse_{name}_m"] = float(np.sqrt((d ** 2).mean()))
            out[f"rgb_rmse_{name}"] = float(np.sqrt(
                ((view["rgb"][:, mask] - rgb_gt[:, mask]) ** 2).mean()))
        else:
            out[f"depth_mae_{name}_m"] = out[f"depth_rmse_{name}_m"] = None
            out[f"rgb_rmse_{name}"] = None
    out["rgb_abs_err_map_mean"] = float(err.mean())
    return out


def build_report(out_dir, *, scene, real_frames, cameras, camera_names,
                 variants, render_frames, source_rows=None, traces=None,
                 tracks=None, title="", notes=None):
    """Render every variant at every (frame, camera) and write the report.

    real_frames   {original_frame: [(rgb[3,H,W], depth[H,W], valid[H,W]) per cam]}
    variants      ordered {label: callable(frame) -> state | None}
    source_rows   {0: rows, 1: rows} for union states, for per-source alpha
    traces        {label: list of IEKS trace records} for optimisation curves
    tracks        {label: {"frames": [...], "mu": [T,N,3] np, "v": ..., "omega":
                  ..., "source": [N], "row": [N]}} for the 3D HTML
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    vis = out / "visuals"
    vis.mkdir(parents=True, exist_ok=True)
    labels = list(variants)

    # one metric depth scale for the whole report
    vals = np.concatenate([d[v].ravel() for f in render_frames
                           for (_r, d, v) in [(a, b.numpy() if torch.is_tensor(b) else b,
                                               c.numpy() if torch.is_tensor(c) else c)
                                              for a, b, c in real_frames[f]]])
    dmin, dmax = (float(np.percentile(vals, 2)), float(np.percentile(vals, 98))) \
        if vals.size else (0.0, 1.0)

    # One crop per camera, from where the REAL object is across every rendered
    # frame, padded so a moving part that leaves that box is still in view.
    # The same crop is used for every variant, so panels stay comparable.
    crops = {}
    for ci, cam in enumerate(camera_names):
        mask = None
        for f in render_frames:
            v = real_frames[f][ci][2]
            v = v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
            mask = v.astype(bool) if mask is None else (mask | v.astype(bool))
        ys, xs = np.nonzero(mask)
        H, W = mask.shape
        if ys.size:
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            py = int(0.35 * (y1 - y0 + 1)) + 8
            px = int(0.35 * (x1 - x0 + 1)) + 8
            crops[cam] = (max(0, y0 - py), min(H, y1 + py + 1),
                          max(0, x0 - px), min(W, x1 + px + 1))
        else:
            crops[cam] = (0, H, 0, W)

    def cut(img, cam):
        y0, y1, x0, x1 = crops[cam]
        return img[..., y0:y1, x0:x1]

    rows, images = [], []
    for f in render_frames:
        real = [(np.asarray(r.cpu() if torch.is_tensor(r) else r, dtype=np.float32),
                 np.asarray(d.cpu() if torch.is_tensor(d) else d, dtype=np.float32),
                 np.asarray(v.cpu() if torch.is_tensor(v) else v, dtype=bool))
                for r, d, v in real_frames[f]]
        rendered = {}
        per_source = {}
        for lab in labels:
            st = variants[lab](f)
            rendered[lab] = None if st is None else render(st, cameras)
            if st is not None and source_rows and st.mu.shape[0] == sum(
                    len(r) for r in source_rows.values()):
                per_source[lab] = {k: render(subset(st, r), cameras)
                                   for k, r in source_rows.items()}
        for ci, cam in enumerate(camera_names):
            rgb_gt, depth_gt, valid = real[ci]
            common = valid.copy()
            for lab in labels:
                if rendered[lab] is not None:
                    common &= rendered[lab][ci]["alpha"] > ALPHA_GATE
            for lab in labels:
                row = {"scene": scene, "original_frame": f, "camera": cam,
                       "variant": lab,
                       "available": rendered[lab] is not None,
                       "common_px": int(common.sum())}
                if rendered[lab] is not None:
                    row.update(_metrics(rendered[lab][ci], real[ci], common))
                    if lab in per_source:
                        a0 = per_source[lab][0][ci]
                        a1 = per_source[lab][1][ci]
                        both = (a0["alpha"] > 1e-3) & (a1["alpha"] > 1e-3)
                        row["alpha_initial_rows_mean"] = float(a0["alpha"].mean())
                        row["alpha_final_rows_mean"] = float(a1["alpha"].mean())
                        row["doubly_covered_px"] = int(both.sum())
                        row["depth_agree_frac_of_doubly_covered"] = (
                            float((np.abs(a0["depth"] - a1["depth"])[both]
                                   <= DEPTH_AGREE_M).mean()) if both.any() else None)
                rows.append(row)

            # ---- contact sheet: real | variants ------------------------------
            ncol = 1 + len(labels)
            fig, ax = plt.subplots(5, ncol, figsize=(3.2 * ncol, 13.5))
            ax = np.atleast_2d(ax)
            ax[0, 0].imshow(cut(rgb_gt, cam).transpose(1, 2, 0))
            ax[0, 0].set_title("REAL RGB")
            dshow = cut(np.where(valid, depth_gt, np.nan), cam)
            im_d = ax[1, 0].imshow(dshow, cmap="turbo", vmin=dmin, vmax=dmax)
            ax[1, 0].set_title("REAL depth (m)")
            for r in (2, 3, 4):
                ax[r, 0].axis("off")
            ax[2, 0].text(0.0, 0.5, "row 3: |RGB error|\nrow 4: |depth error| (m)\n"
                          "row 5: alpha\n\ndepth scale fixed\n"
                          f"[{dmin:.2f}, {dmax:.2f}] m\ncommon px {int(common.sum())}",
                          fontsize=8, va="center")
            for j, lab in enumerate(labels, start=1):
                v = rendered[lab]
                if v is None:
                    for r in range(5):
                        ax[r, j].text(0.5, 0.5, "NOT AVAILABLE\n(not run /\noutside window)",
                                      ha="center", va="center", color="red", fontsize=9)
                    ax[0, j].set_title(lab, fontsize=8)
                    continue
                vi = v[ci]
                m = next(r for r in rows if r["original_frame"] == f
                         and r["camera"] == cam and r["variant"] == lab)
                ax[0, j].imshow(cut(vi["rgb"], cam).transpose(1, 2, 0))
                ax[0, j].set_title(f"{lab}\nRGB rmse(common) "
                                   f"{m['rgb_rmse_common'] if m['rgb_rmse_common'] is None else round(m['rgb_rmse_common'], 4)}",
                                   fontsize=8)
                ax[1, j].imshow(cut(np.where(vi["alpha"] > ALPHA_GATE, vi["depth"],
                                             np.nan), cam),
                                cmap="turbo", vmin=dmin, vmax=dmax)
                ax[1, j].set_title(f"depth mae(common) "
                                   f"{m['depth_mae_common_m'] if m['depth_mae_common_m'] is None else round(m['depth_mae_common_m'], 4)} m",
                                   fontsize=8)
                ax[2, j].imshow(cut(np.abs(vi["rgb"] - rgb_gt).mean(0), cam),
                                cmap="magma", vmin=0, vmax=0.5)
                derr = np.where(valid & (vi["alpha"] > ALPHA_GATE),
                                np.abs(vi["depth"] - depth_gt), np.nan)
                ax[3, j].imshow(cut(derr, cam), cmap="magma", vmin=0, vmax=0.3)
                ax[4, j].imshow(cut(vi["alpha"], cam), cmap="gray", vmin=0, vmax=1)
            for a in ax.ravel():
                a.set_xticks([]); a.set_yticks([])
            fig.colorbar(im_d, ax=ax[1, :].tolist(), fraction=0.02, label="depth (m)")
            fig.suptitle(f"{scene}  original frame {f}  camera {cam}  {title}\n"
                         f"(cropped to the real object, rows {crops[cam][0]}:"
                         f"{crops[cam][1]} cols {crops[cam][2]}:{crops[cam][3]}; "
                         "metrics are full-frame)", fontsize=9)
            path = vis / f"contact_sheet_{cam}_frame{f:04d}.png"
            fig.savefig(path, dpi=80)
            plt.close(fig)
            images.append(path)

            # ---- the three named comparisons: real | each variant -----------
            for kind in ("rgb", "depth", "alpha"):
                fig, ax = plt.subplots(2, ncol, figsize=(3.2 * ncol, 6.4))
                ax = np.atleast_2d(ax)
                if kind == "rgb":
                    ax[0, 0].imshow(cut(rgb_gt, cam).transpose(1, 2, 0))
                    ax[0, 0].set_title("REAL RGB", fontsize=9)
                    ax[1, 0].axis("off")
                    ax[1, 0].text(0, 0.5, "row 2: mean |RGB error|\nscale 0..0.5",
                                  fontsize=8)
                elif kind == "depth":
                    im = ax[0, 0].imshow(cut(np.where(valid, depth_gt, np.nan), cam),
                                         cmap="turbo", vmin=dmin, vmax=dmax)
                    ax[0, 0].set_title("REAL depth (m)", fontsize=9)
                    ax[1, 0].axis("off")
                    ax[1, 0].text(0, 0.5, "row 2: |depth error| (m)\non real-valid AND\n"
                                  f"alpha>{ALPHA_GATE}; scale 0..0.3 m", fontsize=8)
                else:
                    ax[0, 0].imshow(cut(valid.astype(float), cam), cmap="gray",
                                    vmin=0, vmax=1)
                    ax[0, 0].set_title("REAL valid-depth mask", fontsize=9)
                    ax[1, 0].axis("off")
                    ax[1, 0].text(0, 0.5, "row 1: rendered alpha\nrow 2: depth "
                                  f"support\n(real valid & alpha>{ALPHA_GATE})",
                                  fontsize=8)
                for j, lab in enumerate(labels, start=1):
                    v = rendered[lab]
                    if v is None:
                        for r in range(2):
                            ax[r, j].text(0.5, 0.5, "NOT AVAILABLE\n(not run /\n"
                                          "outside window)", ha="center",
                                          va="center", color="red", fontsize=9)
                        ax[0, j].set_title(lab, fontsize=8)
                        continue
                    vi = v[ci]
                    m = next(r for r in rows if r["original_frame"] == f
                             and r["camera"] == cam and r["variant"] == lab)
                    if kind == "rgb":
                        ax[0, j].imshow(cut(vi["rgb"], cam).transpose(1, 2, 0))
                        ax[1, j].imshow(cut(np.abs(vi["rgb"] - rgb_gt).mean(0), cam),
                                        cmap="magma", vmin=0, vmax=0.5)
                        val = m["rgb_rmse_common"]
                        ax[0, j].set_title(f"{lab}\nRGB rmse(common) "
                                           f"{'n/a' if val is None else f'{val:.4f}'}",
                                           fontsize=8)
                    elif kind == "depth":
                        ax[0, j].imshow(cut(np.where(vi["alpha"] > ALPHA_GATE,
                                                     vi["depth"], np.nan), cam),
                                        cmap="turbo", vmin=dmin, vmax=dmax)
                        ax[1, j].imshow(cut(np.where(valid & (vi["alpha"] > ALPHA_GATE),
                                                     np.abs(vi["depth"] - depth_gt),
                                                     np.nan), cam),
                                        cmap="magma", vmin=0, vmax=0.3)
                        val = m["depth_mae_common_m"]
                        ax[0, j].set_title(f"{lab}\ndepth MAE(common) "
                                           f"{'n/a' if val is None else f'{val:.4f} m'}",
                                           fontsize=8)
                    else:
                        ax[0, j].imshow(cut(vi["alpha"], cam), cmap="gray", vmin=0, vmax=1)
                        ax[1, j].imshow(cut((valid & (vi["alpha"] > ALPHA_GATE))
                                            .astype(float), cam), cmap="gray",
                                        vmin=0, vmax=1)
                        ax[0, j].set_title(f"{lab}\nalpha support "
                                           f"{m['alpha_support_px']} px, depth "
                                           f"{m['depth_support_px']} px", fontsize=8)
                for a in ax.ravel():
                    a.set_xticks([]); a.set_yticks([])
                if kind == "depth":
                    fig.colorbar(im, ax=ax[0, :].tolist(), fraction=0.02,
                                 label="metric depth (m), fixed scale")
                fig.suptitle(f"{scene}  {kind}  original frame {f}  {cam}  {title}",
                             fontsize=9)
                p_ = vis / f"{kind}_comparison_{cam}_{f:04d}.png"
                fig.savefig(p_, dpi=80)
                plt.close(fig)
                images.append(p_)

            # ---- per-source alpha / depth agreement for union variants -------
            for lab, ps in per_source.items():
                a0, a1 = ps[0][ci], ps[1][ci]
                both = (a0["alpha"] > 1e-3) & (a1["alpha"] > 1e-3)
                agree = np.where(both, (np.abs(a0["depth"] - a1["depth"])
                                        <= DEPTH_AGREE_M).astype(float), np.nan)
                fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
                ax[0].imshow(cut(a0["alpha"], cam), cmap="Blues", vmin=0, vmax=1)
                ax[0].set_title("alpha, initial-source rows")
                ax[1].imshow(cut(a1["alpha"], cam), cmap="Oranges", vmin=0, vmax=1)
                ax[1].set_title("alpha, final-source rows")
                ax[2].imshow(cut(rendered[lab][ci]["alpha"], cam), cmap="gray",
                             vmin=0, vmax=1)
                ax[2].set_title("alpha, union (one composite)")
                ax[3].imshow(cut(agree, cam), cmap="RdYlGn", vmin=0, vmax=1)
                ax[3].set_title(f"doubly covered: green = depths within "
                                f"{DEPTH_AGREE_M} m", fontsize=8)
                for a in ax:
                    a.set_xticks([]); a.set_yticks([])
                fig.suptitle(f"{lab}  frame {f}  {cam}", fontsize=9)
                p = vis / f"alpha_sources_{_slug(lab)}_{cam}_{f:04d}.png"
                fig.savefig(p, dpi=80)
                plt.close(fig)
                images.append(p)

    # ---- CSV --------------------------------------------------------------
    keys = sorted({k for r in rows for k in r})
    front = ["scene", "original_frame", "camera", "variant", "available"]
    keys = front + [k for k in keys if k not in front]
    with open(out / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- per-camera sequences (only the frames that were rendered) --------
    try:
        import imageio.v2 as imageio
        for cam in camera_names:
            for stem in ("rgb_comparison", "depth_comparison", "contact_sheet"):
                pat = ("{stem}_{cam}_{f:04d}.png" if stem != "contact_sheet"
                       else "{stem}_{cam}_frame{f:04d}.png")
                seq = [vis / pat.format(stem=stem, cam=cam, f=f) for f in render_frames]
                frames = [imageio.imread(p) for p in seq if p.is_file()]
                if len(frames) > 1:
                    h = min(fr.shape[0] for fr in frames)
                    w_ = min(fr.shape[1] for fr in frames)
                    frames = [fr[:h, :w_] for fr in frames]
                    # only the observed frames, one per second: nothing is
                    # interpolated between them
                    imageio.mimsave(vis / f"{stem}_{cam}.mp4", frames, fps=1,
                                    macro_block_size=1)
    except Exception as exc:                         # a missing codec is not the result
        (vis / "video_error.txt").write_text(f"{type(exc).__name__}: {exc}")

    # ---- interactive HTML -------------------------------------------------
    if tracks:
        write_timeline_html(vis / "gaussians_unified_timeline.html", tracks, scene)
        write_tracks_html(vis / "motion_tracks_source0_sourceT.html", tracks, scene)
    if traces:
        write_curves_html(vis / "optimization_curves.html", traces, scene)

    summary = _summarise(rows, labels)
    manifest = {
        "scene": scene, "title": title, "render_frames": list(render_frames),
        "cameras": list(camera_names), "variants": labels,
        "depth_colour_scale_m": [dmin, dmax], "alpha_gate": ALPHA_GATE,
        "display_crops_rows_cols": {c: list(map(int, v)) for c, v in crops.items()},
        "depth_agree_m": DEPTH_AGREE_M,
        "common_domain": ("real depth valid AND alpha > 0.05 for EVERY variant "
                          "rendered at that frame/camera"),
        "summary_mean_over_frames_and_cameras": summary,
        "notes": notes or [],
        "images": {str(p.relative_to(out)): sha256_file(p) for p in images},
    }
    (out / "visual_manifest.json").write_text(json.dumps(manifest, indent=1))
    return {"rows": rows, "summary": summary, "images": images,
            "manifest": manifest}


def _summarise(rows, labels):
    out = {}
    for lab in labels:
        rs = [r for r in rows if r["variant"] == lab and r["available"]]
        if not rs:
            out[lab] = None
            continue

        def mean(k):
            xs = [r[k] for r in rs if r.get(k) is not None]
            return float(np.mean(xs)) if xs else None
        out[lab] = {"n": len(rs), "rgb_rmse_common": mean("rgb_rmse_common"),
                    "depth_mae_common_m": mean("depth_mae_common_m"),
                    "depth_rmse_common_m": mean("depth_rmse_common_m"),
                    "rgb_rmse_own": mean("rgb_rmse_own"),
                    "depth_mae_own_m": mean("depth_mae_own_m"),
                    "alpha_support_px": mean("alpha_support_px"),
                    "depth_support_px": mean("depth_support_px"),
                    "depth_agree_frac_of_doubly_covered":
                        mean("depth_agree_frac_of_doubly_covered")}
    return out


def _sample_rows(mu0, cap, seed=0):
    """A deterministic spatial stratification for DISPLAY only."""
    n = mu0.shape[0]
    if n <= cap:
        return np.arange(n)
    lo, hi = mu0.min(0), mu0.max(0)
    cells = np.floor((mu0 - lo) / np.maximum(hi - lo, 1e-9) * 12).astype(int)
    key = cells[:, 0] * 169 + cells[:, 1] * 13 + cells[:, 2]
    order = np.lexsort((np.arange(n), key))
    picked, seen = [], {}
    for i in order:                       # round-robin over occupied cells
        seen.setdefault(key[i], []).append(i)
    buckets = list(seen.values())
    k = 0
    while len(picked) < cap:
        progressed = False
        for b in buckets:
            if k < len(b):
                picked.append(b[k]); progressed = True
                if len(picked) >= cap:
                    break
        if not progressed:
            break
        k += 1
    return np.sort(np.asarray(picked))


def write_timeline_html(path, tracks, scene, cap=1200):
    """Frame slider over original frames; one trace per (variant, source)."""
    import plotly.graph_objects as go

    labels = list(tracks)
    frames_all = sorted({f for t in tracks.values() for f in t["frames"]})
    samples = {}
    for lab, t in tracks.items():
        samples[lab] = {}
        for s in (0, 1):
            rows = np.flatnonzero(np.asarray(t["source"]) == s)
            if rows.size:
                pick = _sample_rows(t["mu"][0][rows], cap)
                samples[lab][s] = rows[pick]

    def traces_at(f):
        out = []
        for lab, t in tracks.items():
            for s, rows in samples[lab].items():
                if f in t["frames"]:
                    k = t["frames"].index(f)
                    p = np.round(t["mu"][k][rows], 4)
                else:
                    p = np.zeros((0, 3))
                out.append(go.Scatter3d(
                    x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                    marker=dict(size=1.6, color=SOURCE_COLORS[s],
                                opacity=0.75 if "after" in lab else 0.45,
                                symbol="circle" if s == 0 else "diamond"),
                    name=f"{lab} / {'initial' if s == 0 else 'final'} source "
                         f"({len(t['source']) if False else int((np.asarray(t['source']) == s).sum())} total, {len(rows)} shown)",
                    legendgroup=lab))
        return out

    fig = go.Figure(data=traces_at(frames_all[0]),
                    frames=[go.Frame(data=traces_at(f), name=str(f))
                            for f in frames_all])
    fig.update_layout(
        title=(f"{scene}: Gaussian centres per ORIGINAL frame. blue = initial "
               "source, orange = final source. Points are a fixed spatial "
               "sample for display only; the optimiser used every Gaussian. "
               "Click legend entries to toggle variants."),
        scene=dict(aspectmode="data"), height=850,
        sliders=[dict(steps=[dict(method="animate", label=str(f),
                                  args=[[str(f)], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=True))])
                             for f in frames_all],
                      currentvalue=dict(prefix="original frame "))],
        updatemenus=[dict(type="buttons", buttons=[dict(
            label="play", method="animate",
            args=[None, dict(frame=dict(duration=150, redraw=True))])])])
    fig.write_html(str(path), include_plotlyjs="cdn")


def write_tracks_html(path, tracks, scene, per_source=30):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene", "rowspan": 2},
                                                {"type": "xy"}],
                                               [None, {"type": "xy"}]],
                        subplot_titles=("tracks (original frames)", "|v| per declared second",
                                        "|omega| (rad per declared second)"))
    for lab, t in tracks.items():
        for s in (0, 1):
            rows = np.flatnonzero(np.asarray(t["source"]) == s)
            if not rows.size:
                continue
            pick = rows[_sample_rows(t["mu"][0][rows], per_source)]
            for i in pick:
                p = t["mu"][:, i]
                tag = (f"{lab} {'initial' if s == 0 else 'final'} "
                       f"row {int(t['row'][i])}")
                fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2],
                                           mode="lines", line=dict(
                                               color=SOURCE_COLORS[s], width=2),
                                           name=tag, legendgroup=lab,
                                           showlegend=False), row=1, col=1)
                if t.get("v") is not None:
                    fig.add_trace(go.Scatter(x=t["frames"],
                                             y=np.linalg.norm(t["v"][:, i], axis=1),
                                             mode="lines", line=dict(
                                                 color=SOURCE_COLORS[s], width=1),
                                             name=tag, showlegend=False),
                                  row=1, col=2)
                if t.get("omega") is not None:
                    fig.add_trace(go.Scatter(x=t["frames"],
                                             y=np.linalg.norm(t["omega"][:, i], axis=1),
                                             mode="lines", line=dict(
                                                 color=SOURCE_COLORS[s], width=1),
                                             name=tag, showlegend=False),
                                  row=2, col=2)
    fig.update_layout(title=(f"{scene}: a FIXED spatially stratified sample of "
                             f"up to {per_source} Gaussians per source "
                             "(chosen from frame-0 positions; no labels used). "
                             "Hover shows source and original PLY row."),
                      height=850)
    fig.write_html(str(path), include_plotlyjs="cdn")


def write_curves_html(path, traces, scene):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("objective terms (accepted = filled)",
                                        "true PCG residual ||b - A d|| / ||b||",
                                        "LM damping", "decrement ratio (certificate)"))
    for lab, tr in traces.items():
        rows = [r for r in tr if "E_total" in r]
        x = [r["outer"] for r in rows]
        for key in ("E_total", "E_rgb", "E_depth", "E_process", "E_prior"):
            fig.add_trace(go.Scatter(
                x=x, y=[r.get(key) for r in rows], mode="lines+markers",
                name=f"{lab} {key}",
                marker=dict(symbol=["circle" if r.get("accepted") else "circle-open"
                                    for r in rows])), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=[r.get("pcg_true_residual") or r.get("pcg_residual")
                                         for r in rows], name=f"{lab} PCG residual"), 2, 1)
        fig.add_trace(go.Scatter(x=x, y=[r.get("damping") for r in rows],
                                 name=f"{lab} damping"), 3, 1)
        fig.add_trace(go.Scatter(x=x, y=[r.get("decrement_ratio") for r in rows],
                                 name=f"{lab} decrement"), 4, 1)
    fig.update_yaxes(type="log")
    fig.update_layout(title=f"{scene}: nonlinear IEKS traces", height=1100)
    fig.write_html(str(path), include_plotlyjs="cdn")
