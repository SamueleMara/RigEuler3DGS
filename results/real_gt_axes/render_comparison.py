#!/usr/bin/env python
"""Recorded RGB vs RigEuler3DGS render, with the manual GT axis and the estimated axis, per frame.

The RigEuler counterpart of FreeArtGS's evaluate/visualize_local_comparison.py (same layout: one
recorded / render pair per camera, 640 px per pair, header with frame and joint state; outputs
comparison_<t>.png at the first / middle / last timestamp, motion_vs_gt.mp4, motion_vs_gt.gif and
comparison_info.json).

For each real scene with a manual GT axis (red_fridge, wooden_box, light_blue_box) and each v6 arm:
  * model    = the arm's endpoint model (forward_joint/summary.json "canonical_model")
  * moving   = the frozen Phase-2 support Phase 3 used (<dir>_rigid/support_coherence/support.json)
  * joint    = the type chosen by the evaluation rule (results/rigeuler_tables.joint_geometry, R/L > 10
               -> prismatic) with Phase 3's FIXED geometry and its per-frame q (<dir>_joint/<type>/)
  * posing   = Phase 3's transport: revolute mu = a + Q(q)(mu_bar - a), R = Q(q) R_bar;
               prismatic mu = mu_bar + q e. The static Gaussians stay in place.
  * render   = the pipeline's own renderer (estimator.phase2_shared.render_state, frozen DC colour),
               on white, with the cameras the run used
GT is read only to draw the green axis and the header's error; it never enters the render.
Overlays ignore occlusion. Writes only under --out (default on the SSD results root).

    PY=~/miniconda3/envs/rigeuler3dgs/bin/python
    $PY results/real_gt_axes/render_comparison.py                          # 3 scenes x 3 arms
    $PY results/real_gt_axes/render_comparison.py --scenes red_fridge --arms majority --max-frames 5

Only the cameras the run itself used are rendered (the joint stage's stage_manifest.json); there is no
option to add a camera the estimation never saw. Output: <results root>/real_gt_axes_comparison/<version>/
<scene>/<arm>/, never inside the repository or the run folders.
"""
import argparse, csv, json, math, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (HERE, HERE.parent, REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from gt_common import GT, REAL, SCENE, line_angle_deg, line_distance, pt_line, unit  # noqa: E402
import rigeuler_tables as rt  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
GREEN, RED = "#008833", "#dd2222"
PW, PH = 320, 240                       # one panel; a camera pair is 640 px wide as in FreeArtGS


def rot(axis, th):
    a = unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def load_run(gt_key, arm, version):
    scene = SCENE[gt_key]
    _, rigid, joint, _ = rt.pipe(version, f"final_{scene}_episode_000000", arm)
    jt = rt.joint_geometry(rigid)["type_decision"]
    summ = json.loads((joint / "summary.json").read_text())
    geo = json.loads((joint / jt / "geometry.json").read_text())["geometry_fixed"]
    rows = list(csv.DictReader(open(joint / jt / "trajectory.csv")))
    sup = json.loads((rigid / "support_coherence" / "support.json").read_text())
    return dict(scene=scene, gt_key=gt_key, arm=arm, version=version, joint_type=jt, geometry=geo,
                frames=[int(r["original_frame"]) for r in rows], q=np.array([float(r["q"]) for r in rows]),
                members=np.asarray(sup["members"], int), model=Path(summ["canonical_model"]),
                direction=summ["direction"], joint_dir=joint, support_file=rigid / "support_coherence/support.json",
                cameras=json.loads((joint / "stage_manifest.json").read_text())["signature"]["source"]["cameras"])


def pose(run, mu0, R0, q):
    """Phase 3's fixed-joint transport of the support (static Gaussians unchanged)."""
    mu, Rg, m = mu0.copy(), R0.copy(), run["members"]
    g = run["geometry"]
    if run["joint_type"] == "prismatic":
        mu[m] = mu0[m] + q * unit(g["direction"])
    else:
        Q, a = rot(g["axis"], q), np.asarray(g["axis_point"], float)
        mu[m] = a + (mu0[m] - a) @ Q.T
        Rg[m] = Q @ R0[m]
    return mu, Rg


def axis_segments(run):
    """Display segments: the GT segment exactly as annotated (green); the estimated axis with the same
    length, centred on its point closest to the GT midpoint (revolute) or on the GT midpoint (prismatic:
    only the direction is intrinsic, so both share the display anchor, as in FreeArtGS)."""
    gt = GT[run["gt_key"]]
    e1, e2 = np.asarray(gt["endpoint_1"]), np.asarray(gt["endpoint_2"])
    mid, L = (e1 + e2) / 2, float(np.linalg.norm(e2 - e1))
    g = run["geometry"]
    d = unit(g["direction"] if run["joint_type"] == "prismatic" else g["axis"])
    d = d if d @ (e2 - e1) >= 0 else -d               # same arrow sense as the GT segment
    if run["joint_type"] == "prismatic":
        c = mid
    else:
        a = np.asarray(g["axis_point"], float)
        c = a + ((mid - a) @ d) * d
    return np.stack([e1, e2]), np.stack([c - L / 2 * d, c + L / 2 * d])


def errors(run):
    gt = GT[run["gt_key"]]
    g, gd = run["geometry"], np.asarray(gt["axis_direction_unit"])
    gp = np.asarray(gt["axis_point"])
    out = dict(gt_type=gt["joint_type"], type_correct=run["joint_type"] == gt["joint_type"])
    if run["joint_type"] == "prismatic":
        out["axis_angle_deg"] = line_angle_deg(np.asarray(g["direction"]), gd)
    else:
        a, e = np.asarray(g["axis_point"]), np.asarray(g["axis"])
        out["axis_angle_deg"] = line_angle_deg(e, gd)
        if gt["joint_type"] == "revolute":
            out["line_distance_m"] = line_distance(a, e, gp, gd)
            out["distance_at_gt_midpoint_m"] = pt_line(gp, a, e)
    return out


def project(spec, X):
    """World points -> pixels with the render's own matrices (3DGS row-vector convention)."""
    Xh = np.c_[X, np.ones(len(X))]
    z = (Xh @ spec.viewmatrix.double().cpu().numpy())[:, 2]
    p = Xh @ spec.projmatrix.double().cpu().numpy()
    ndc = p[:, :2] / p[:, 3:]
    px = np.c_[((ndc[:, 0] + 1) * spec.image_width - 1) / 2, ((ndc[:, 1] + 1) * spec.image_height - 1) / 2]
    return None if np.any(z <= 1e-4) or not np.isfinite(px).all() else px


def draw_axis(image, pixels, color):
    """FreeArtGS's draw_axis: white halo, coloured line, arrow head at the second end."""
    if pixels is None:
        return
    draw = ImageDraw.Draw(image)
    a, b = pixels
    v = b - a
    n = np.linalg.norm(v)
    if n < 3:
        x, y = (a + b) / 2
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
        return
    v /= n
    nrm = np.array([-v[1], v[0]])
    draw.line([tuple(a), tuple(b)], fill="white", width=7)
    draw.line([tuple(a), tuple(b)], fill=color, width=3)
    draw.polygon([tuple(b), tuple(b - 8 * v + 4 * nrm), tuple(b - 8 * v - 4 * nrm)], fill=color)


def to_u8(rgb):
    x = rgb.detach().float().cpu()
    if x.dim() == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    return (x.clamp(0, 1).numpy() * 255).astype(np.uint8)


def render_run(run, out, fps, max_frames=None):
    import torch
    from estimator.phase2_shared import render_state
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import IEKFConfig
    from tools.run_iekf_scene import camera_specs, load_scene, state_from_gaussians
    from utils.camera_selection import camera_name, select_cameras

    src = REAL / f"final_{run['scene']}_episode_000000"
    cam_sel = ",".join(run["cameras"])
    _, _, gaussians, sc = load_scene(str(run["model"]), str(src), inference_only=True, camera_device="cpu")
    cams = select_cameras(sc.getTrainCameras(), cam_sel)
    names = [camera_name(c.image_name) for c in cams]
    if names != list(run["cameras"]):
        raise SystemExit(f"selected {names}, the run used {run['cameras']}")
    specs = camera_specs(cams, len(cams))
    truth = state_from_gaussians(gaussians)
    cfg = IEKFConfig()
    mu0 = truth.mu.double().cpu().numpy()
    R0 = truth.R.double().cpu().numpy()
    if run["members"].max() >= len(mu0):
        raise SystemExit(f"support indexes row {run['members'].max()}, the model has {len(mu0)}")
    bg = torch.ones(3, device="cuda", dtype=truth.mu.dtype)
    gt_seg, est_seg = axis_segments(run)
    err = errors(run)
    order = np.argsort(run["frames"])                  # display in time order (backward runs are reversed)
    if max_frames:
        order = order[np.linspace(0, len(order) - 1, min(max_frames, len(order))).round().astype(int)]
    revolute = run["joint_type"] == "revolute"
    scale, unit_s = (180 / math.pi, "deg") if revolute else (100.0, "cm")
    font = ImageFont.truetype(FONT, 16)
    cols = min(2, len(names))
    W, H = cols * 2 * PW, 64 + math.ceil(len(names) / cols) * (PH + 40)
    err_s = f"axis error {err['axis_angle_deg']:.2f} deg"
    if "line_distance_m" in err:
        err_s += f", line distance {err['line_distance_m'] * 1000:.0f} mm"
    out.mkdir(parents=True, exist_ok=True)
    keep = {0, len(order) // 2, len(order) - 1}
    with tempfile.TemporaryDirectory(prefix="rigeuler-comparison-") as tmp:
        for t, k in enumerate(order):
            f, q = run["frames"][k], float(run["q"][k])
            mu, Rg = pose(run, mu0, R0, q)
            mu_t = torch.as_tensor(mu, device=truth.mu.device)
            R_t = torch.as_tensor(Rg, device=truth.mu.device)
            canvas = Image.new("RGB", (W, H), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 5), f"{src.name} | v6 arm {run['arm']} | Frame {f} | Estimated {run['joint_type']} "
                      f"{q * scale:.2f} {unit_s} | {err_s}", font=font, fill="black")
            draw.text((10, 28), "Training views; overlays ignore occlusion. GT axis (manual annotation) green; "
                      "estimated axis red." + (" Prismatic anchor is arbitrary." if not revolute else ""),
                      font=font, fill="black")
            for c, (name, spec) in enumerate(zip(names, specs)):
                lin = render_state(truth, mu_t, R_t, [spec], cfg, build_linearization, background=bg)
                rend = Image.fromarray(to_u8(lin.views[0].rgb))
                del lin
                rec = Image.open(src / "rgb" / name / f"frame_{f:06d}.png").convert("RGB")
                if rec.size != rend.size:
                    raise ValueError(f"{name} frame {f}: recorded {rec.size} vs render {rend.size}")
                sx, sy = PW / rend.size[0], PH / rend.size[1]
                pg, pe = project(spec, gt_seg), project(spec, est_seg)
                x, y = (c % cols) * 2 * PW, 64 + (c // cols) * (PH + 40)
                for j, img in enumerate((rec, rend)):
                    img = img.resize((PW, PH), Image.Resampling.LANCZOS)
                    if j == 0:
                        draw_axis(img, None if pg is None else pg * [sx, sy], GREEN)
                    draw_axis(img, None if pe is None else pe * [sx, sy], RED)
                    label = f"{name} | " + ("Recorded / GT + estimated" if j == 0 else "Render / estimated axis")
                    draw.text((x + j * PW + 8, y), label, font=font, fill=GREEN if j == 0 else RED)
                    canvas.paste(img, (x + j * PW, y + 24))
            canvas.save(Path(tmp) / f"{t:05d}.png")
            if t in keep:
                canvas.save(out / f"comparison_{t:05d}.png")
        ff = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(Path(tmp) / "%05d.png")]
        subprocess.run(ff + ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out / "motion_vs_gt.mp4")], check=True)
        subprocess.run(ff + ["-filter_complex", "scale=960:-1,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                             "-loop", "0", str(out / "motion_vs_gt.gif")], check=True)
    info = dict(scene=src.name, version=run["version"], arm=run["arm"], direction=run["direction"],
                predicted_joint_type=run["joint_type"], type_rule=f"revolute axis radius / part scale > {rt.RL_PRISMATIC} -> prismatic",
                geometry=run["geometry"], joint_state_unit="m" if not revolute else "rad",
                joint_state="Phase-3 q relative to the canonical frame (first processed frame)",
                frames=[run["frames"][k] for k in order], q=[float(run["q"][k]) for k in order],
                gt_axis={"axis_point": GT[run["gt_key"]]["axis_point"].tolist(),
                         "axis_direction_unit": GT[run["gt_key"]]["axis_direction_unit"].tolist(),
                         "endpoints": gt_seg.tolist(), "type": GT[run["gt_key"]]["joint_type"],
                         "note": "manual CloudCompare annotation, not independently verified"},
                errors=err, estimated_display_segment=est_seg.tolist(), cameras=names,
                model=str(run["model"]), support=str(run["support_file"]), support_members=int(len(run["members"])),
                trajectory=str(run["joint_dir"] / run["joint_type"] / "trajectory.csv"),
                render="pipeline renderer (estimator.phase2_shared.render_state), frozen DC colour, white background",
                views="the run's own cameras (training views); not novel-view evaluation",
                fps=fps, timestamps=len(order), video="motion_vs_gt.mp4", animation="motion_vs_gt.gif")
    (out / "comparison_info.json").write_text(json.dumps(info, indent=2))
    print(f"  {src.name} {run['arm']}: {len(order)} frames, {run['joint_type']}, {err_s} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenes", nargs="+", default=list(SCENE.values()), choices=list(SCENE.values()))
    ap.add_argument("--arms", nargs="+", default=["C", "majority", "unanimous"])
    ap.add_argument("--version", default="v6", choices=list(rt.REAL_RUNS))
    ap.add_argument("--out", type=Path, default=rt.R / "real_gt_axes_comparison")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None, help="evenly spaced subset (smoke tests)")
    a = ap.parse_args()
    key = {v: k for k, v in SCENE.items()}
    for s in a.scenes:
        for arm in a.arms:
            run = load_run(key[s], arm, a.version)
            render_run(run, a.out / a.version / s / arm, a.fps, a.max_frames)


if __name__ == "__main__":
    main()
