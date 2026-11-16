#!/usr/bin/env python
"""FreeArtGS-style final comparison for a finished Phase-5 run (same layout as FreeArtGS
evaluate/visualize_local_comparison.py): per timestamp, every camera as [reference | final render], each with the
joint axis drawn on it; mp4 + gif over all timestamps, three stills and comparison_info.json.

    python tools/phase5_comparison.py --phase5-dir OUT/<scene>/phase5/<run>_gaussian_rgbd [--phase5-dir ...]

With dataset joint GT (<source>/pointcloud/joint_metadata.json, metadata/frame_values.csv): reference images are
composited on white with the object mask, the GT axis is green on the reference, the estimated axis red on the
render, and the title shows the estimated and the GT joint state relative to the anchor frame (motion_vs_gt.*).
Without GT: raw reference, estimated axis red on both (motion_axis_debug.*). GT is read for display / evaluation
only, after Phase 5 finished. Output: <phase5-dir>/comparison/ (or --output-dir).
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def project_axis(axis, origin, V_rowvec, K, length):
    """Segment origin +- length/2 * axis -> two pixels, or None if behind the camera. V: [4,4] row-vector world->view."""
    pts = np.asarray(origin, float)[None, :] + np.array([[-.5], [.5]]) * length * np.asarray(axis, float)
    pv = np.c_[pts, np.ones(2)] @ np.asarray(V_rowvec, float)
    pv = pv[:, :3]
    if not np.isfinite(pv).all() or np.any(pv[:, 2] <= 1e-4):
        return None
    px = pv @ np.asarray(K, float).T
    return px[:, :2] / px[:, 2:]


def draw_axis(image, pixels, color):
    if pixels is None:
        return
    draw = ImageDraw.Draw(image)
    a, b = pixels
    direction = b - a
    size = np.linalg.norm(direction)
    if size < 3:
        x, y = (a + b) / 2
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
        return
    direction = direction / size
    normal = np.array([-direction[1], direction[0]])
    draw.line([tuple(a), tuple(b)], fill="white", width=7)
    draw.line([tuple(a), tuple(b)], fill=color, width=3)
    draw.polygon([tuple(b), tuple(b - 8 * direction + 4 * normal), tuple(b - 8 * direction - 4 * normal)], fill=color)


def closest_point_on_line(point, line_point, axis):
    e = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.asarray(line_point, float)
    return a + ((np.asarray(point, float) - a) @ e) * e


def gt_relative_states(gt, joint, axis_est):
    """GT q relative to the anchor frame, sign-aligned to the estimated axis (q about -e == -q about e)."""
    q = gt.get("gt_q") or []
    frames = sorted(joint.frames)
    if not q or max(frames) >= len(q) or joint.anchor_frame >= len(q):
        return None
    s = 1.0 if float(np.dot(axis_est, gt["axis"])) >= 0 else -1.0
    return {f: s * (q[f] - q[joint.anchor_frame]) for f in frames}, s


def joint_report(joint, gt, moving_centroid):
    from joint_reduction.gt_evaluation import axis_angle_error, line_to_line_distance
    axis = np.asarray(joint.axis, float)
    rev = joint.type == "revolute"
    rep = {"predicted_joint_type": joint.type, "joint_state_unit": "rad" if rev else "m", "predicted_axis": axis.tolist(),
           "anchor_frame": joint.anchor_frame, "direction": joint.direction, "gt_available": bool(gt.get("available"))}
    if rev:
        rep["predicted_axis_point"] = list(joint.point)
    fr, q = joint.chronological()
    est = {int(f): float(x) for f, x in zip(fr, q)}
    if gt.get("available"):
        e_gt, a_gt = np.asarray(gt["axis"], float), np.asarray(gt["axis_point"], float)
        rep.update(gt_axis=e_gt.tolist(), gt_origin=a_gt.tolist(), gt_joint_type=gt.get("joint_type"),
                   gt_coordinate_frame=gt.get("coordinate_frame"),
                   joint_axis_error_deg=axis_angle_error(axis, e_gt))
        # display anchors only (any point of an axis line is equivalent): each line is drawn through its point
        # nearest the moving-part centroid; prismatic has no axis point, so both use the centroid (common anchor)
        if rev:
            rep["joint_axis_position_error_m"] = line_to_line_distance(joint.point, axis, a_gt, e_gt)["distance"]
            rep["predicted_origin"] = closest_point_on_line(moving_centroid, joint.point, axis).tolist()
            rep["gt_display_anchor"] = closest_point_on_line(moving_centroid, a_gt, e_gt).tolist()
        else:
            rep["predicted_origin"] = rep["gt_display_anchor"] = np.asarray(moving_centroid, float).tolist()
        g = gt_relative_states(gt, joint, axis)
        if g is not None:
            gq, sign = g
            err = np.array([abs(est[f] - gq[f]) for f in est])
            rep.update(joint_state_evaluation="relative_to_anchor_frame", gt_state_sign_alignment=sign,
                       joint_state_mae=float(err.mean()), joint_state_max_error=float(err.max()))
            rep["joint_state_samples"] = [dict(time_index=t, source_frame_index=f, predicted_relative_state=est[f],
                                               gt_relative_state=gq[f], absolute_error=abs(est[f] - gq[f]))
                                          for t, f in enumerate(sorted(est))]
    else:
        rep["predicted_origin"] = (closest_point_on_line(moving_centroid, joint.point, axis) if rev
                                   else np.asarray(moving_centroid, float)).tolist()
    if "joint_state_samples" not in rep:
        rep["joint_state_samples"] = [dict(time_index=t, source_frame_index=f, predicted_relative_state=est[f])
                                      for t, f in enumerate(sorted(est))]
    return rep


def compose_frame(scene_name, sample, rep, cams, refs, renders, axis_length, font):
    """One canvas: 2 camera pairs per row, each pair = reference | render (FreeArtGS layout)."""
    has_gt, pris = rep["gt_available"], rep["predicted_joint_type"] == "prismatic"
    unit = rep["joint_state_unit"]
    scale, du = (180 / math.pi, "deg") if unit == "rad" else (1, "m")
    W0, H0 = refs[0].size
    tw, th = 320, int(round(320 * H0 / W0))
    columns = min(2, len(cams))
    width, height = columns * 2 * tw, 64 + math.ceil(len(cams) / columns) * (th + 40)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label = f"Estimated {sample['predicted_relative_state'] * scale:.2f} {du}"
    if "gt_relative_state" in sample:
        label += f" | GT {sample['gt_relative_state'] * scale:.2f} {du}"
    draw.text((10, 5), f"{scene_name} | Frame {sample['source_frame_index']} | {label}", font=font, fill="black")
    draw.text((10, 28), "Training views; axis overlays ignore occlusion. " + (
        "Prismatic axis anchor is arbitrary." if pris else
        "GT axis green; estimated axis red." if has_gt else "No joint GT; estimated axis red in both images."),
        font=font, fill="black")
    axes = [rep.get("gt_axis", rep["predicted_axis"]), rep["predicted_axis"]]
    ref_anchor = rep.get("gt_display_anchor", rep["predicted_origin"])
    anchors = [ref_anchor, ref_anchor if pris else rep["predicted_origin"]]
    colors = ["#008833" if has_gt else "#dd2222", "#dd2222"]
    for c, cam in enumerate(cams):
        x, y = (c % columns) * 2 * tw, 64 + (c // columns) * (th + 40)
        for j, image in enumerate((refs[c], renders[c])):
            if image.size != (cam.W, cam.H):
                raise ValueError(f"{cam.name}: image {image.size} vs camera {cam.W}x{cam.H}")
            image = image.resize((tw, th), Image.Resampling.LANCZOS)
            px = project_axis(axes[j], anchors[j], cam.V.cpu().numpy(), cam.K, axis_length)
            if px is not None:
                px = px * np.array([tw / cam.W, th / cam.H])
            draw_axis(image, px, colors[j])
            draw.text((x + j * tw + 8, y), f"{cam.name} | " + (
                ("Reference / GT axis" if has_gt else "Recorded / estimated axis") if j == 0
                else "Render / estimated axis"), font=font, fill=colors[j])
            canvas.paste(image, (x + j * tw, y + 24))
    return canvas


def to_pil(t):
    a = t.detach().float().clamp(0, 1).cpu().numpy()
    if a.ndim == 3 and a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


def write_videos(frames_dir, out, basename, fps):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                    "-i", str(Path(frames_dir) / "%05d.png"), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out / f"{basename}.mp4")], check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                    "-i", str(Path(frames_dir) / "%05d.png"), "-filter_complex",
                    "scale=960:-1,split[a][b];[a]palettegen[p];[b][p]paletteuse", "-loop", "0",
                    str(out / f"{basename}.gif")], check=True)


def run(d: Path, out: Path | None, fps: int, axis_length: float, cameras: str | None, max_frames: int, device: str):
    import torch
    from joint_reduction.gt_evaluation import load_joint_gt
    from recovery import phase4_artifacts as AR
    from recovery.articulated_scene import JointTransport, render_articulated
    from recovery.gaussians import GaussianTensors
    from recovery.io_setup import setup
    from recovery.fair_psnr import load_reference, quantize_like_splatfacto, score_view, summarize
    m = json.loads((d / "manifest.json").read_text())
    st = m.get("setup") or m.get("upstream", {}).get("setup")
    if not st:
        raise SystemExit(f"{d}/manifest.json: no setup")
    ns = argparse.Namespace(source=st["source"], model=st["model"], phase2_dir=st["phase2_dir"],
                            phase3_dir=st["phase3_dir"], joint_type=st["joint_type"], direction=st["direction"],
                            cameras=st.get("cameras"), frame_window=st.get("frame_window"),
                            sensor_model=st.get("sensor_model"), dry_run=False, view_cache=4, smoke_frames=0)
    up, cams, views, _ = setup(ns, log=lambda *a, **k: None)
    proof = json.loads((d / "input_proof.json").read_text())
    if AR.joint_fingerprint(up.joint) != proof["joint_fingerprint"]:
        raise SystemExit(f"{d}: the Phase-3 joint on disk differs from the one Phase 5 used")
    joint = up.joint
    if cameras:
        want = cameras.split(",")
        missing = [w for w in want if w not in {c.name for c in cams}]
        if missing:
            raise SystemExit(f"--cameras: {missing} not among the Phase-3 cameras {[c.name for c in cams]}")
        cams = [c for c in cams if c.name in want]
    fin = d / "final"
    g = GaussianTensors.load_ply(fin / "point_cloud.ply", device=device)
    g.lineage = GaussianTensors.load_lineage(fin)
    app = json.loads((fin / "appearance_settings.json").read_text())
    moving = torch.as_tensor(g.lineage["material"] == 1, device=device)
    xyz = g._xyz.detach().cpu().numpy()
    centroid = xyz[g.lineage["material"] == 1].mean(0) if moving.any() else xyz.mean(0)
    gt = load_joint_gt(up.source)
    rep = joint_report(joint, gt, centroid)
    pris = joint.type == "prismatic"
    if axis_length <= 0:      # auto: revolute 1.2 x the moving part's extent along the axis; prismatic 1.2 x the travel
        if pris:
            axis_length = 1.2 * float(np.ptp(joint.q))
        else:
            sel = xyz[g.lineage["material"] == 1] if moving.any() else xyz
            lo, hi = np.percentile(sel @ (np.asarray(joint.axis, float) / np.linalg.norm(joint.axis)), [1, 99])
            axis_length = 1.2 * float(hi - lo)
    has_gt = rep["gt_available"]
    basename = "motion_vs_gt" if has_gt else "motion_axis_debug"
    samples = rep["joint_state_samples"]
    if max_frames:
        samples = samples[:max_frames]
    by_key = {(v.cam.name, int(v.frame)): v for v in views}
    transport = JointTransport(joint, device)
    bg = torch.tensor(app["background"], device=device, dtype=torch.float32)
    font = ImageFont.truetype(FONT, 16)
    out = out or d / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    scene_name = Path(up.source).name
    T = len(samples)
    stills = {0, T // 2, T - 1}
    psnr_rows = []
    with tempfile.TemporaryDirectory(prefix="phase5-comparison-") as tmp, torch.no_grad():
        for s in samples:
            f = s["source_frame_index"]
            refs, rends = [], []
            for c in cams:
                v = by_key[(c.name, f)].get()
                rgb = v.rgb
                if has_gt:
                    rgb = torch.where(v.mask[None].bool(), rgb, torch.ones_like(rgb))
                refs.append(to_pil(rgb))
                o = render_articulated(c, transport.at(f), g, moving, bg, app["sh_mode"], app["active_sh_degree"])
                rends.append(to_pil(o["color"]))
                # PSNR exactly as FreeArtGS (recovery/fair_psnr.py), against the recorded files themselves
                ref_u8, ref_mask, ref_sha = load_reference(up.source, c.name, f"frame_{f:06d}")
                pred_u8 = quantize_like_splatfacto(o["color"].permute(1, 2, 0).float().cpu().numpy())
                psnr_rows.append(dict(camera=c.name, frame=int(f), **score_view(pred_u8, ref_u8, ref_mask),
                                      reference_sha256=ref_sha))
            rf = rep
            if pris:          # the prismatic arrow rides with the moving part: centroid posed at this frame
                a = (np.asarray(centroid, float) + np.asarray(joint.transform(f)[1], float)).tolist()
                rf = dict(rep, predicted_origin=a, gt_display_anchor=a)
            canvas = compose_frame(scene_name, s, rf, cams, refs, rends, axis_length, font)
            canvas.save(Path(tmp) / f"{s['time_index']:05d}.png")
            if s["time_index"] in stills:
                canvas.save(out / f"comparison_{s['time_index']:05d}.png")
        write_videos(tmp, out, basename, fps)
    with open(out / "psnr_views.csv", "w") as fh:
        fh.write("camera,frame,psnr,mask_empty,reference_sha256\n")
        for r in psnr_rows:
            fh.write(f"{r['camera']},{r['frame']},{r['psnr']:.6f},{int(r['mask_empty'])},{r['reference_sha256']}\n")
    ps = summarize(psnr_rows)
    ps.update(complete=not max_frames and not cameras, reference_source=str(up.source),
              quantization="(clip(rgb,0,1)*255).astype(uint8), as splatfacto-art eval_results",
              background=app["background"], per_view_csv="psnr_views.csv")
    info = dict(rep, phase5_dir=str(d), scene=scene_name, psnr_masked_freeart=ps,
                source="Phase-5 final renders versus recorded RGB" + (" composited on white using object masks" if has_gt else ""),
                cameras=[c.name for c in cams], timestamps=T, fps=fps, axis_display_length_m=axis_length,
                views="training views; not novel-view evaluation",
                prismatic_common_display_anchor=rep["predicted_joint_type"] == "prismatic",
                predicted_origin_note="display anchors only: each axis line is drawn through its point nearest the "
                                      "moving-part centroid (prismatic: that centroid posed at each frame, so the "
                                      "arrow moves with the part); gt_origin is the raw "
                                      "dataset point_of_application",
                axis_sign_note="an axis sign is a convention: GT and estimated arrows may point opposite ways; GT "
                               "states are sign-aligned to the estimated axis (gt_state_sign_alignment)",
                gt_origin_available="gt_origin" in rep, video=f"{basename}.mp4", animation=f"{basename}.gif")
    (out / "comparison_info.json").write_text(json.dumps(info, indent=2))
    msg = (f"{scene_name}: {T} timestamps x {len(cams)} cameras -> {out / (basename + '.mp4')}; masked PSNR "
           f"(FreeArtGS definition) {ps['psnr']:.2f} dB over {ps['frames']} views")
    if has_gt:
        msg += f"; axis error {rep['joint_axis_error_deg']:.3f} deg"
        if "joint_axis_position_error_m" in rep:
            msg += f", axis line distance {rep['joint_axis_position_error_m'] * 1000:.1f} mm"
        if "joint_state_mae" in rep:
            k = 180 / math.pi if rep["joint_state_unit"] == "rad" else 1000
            msg += f", state MAE {rep['joint_state_mae'] * k:.2f} {'deg' if k != 1000 else 'mm'}"
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase5-dir", action="append", required=True)
    ap.add_argument("--output-dir", default=None, help="only with one --phase5-dir (default <phase5-dir>/comparison)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--axis-length", type=float, default=0.0,
                    help="drawn axis segment in metres; 0 = 1.2 x the moving part extent along the axis (FreeArtGS: fixed 0.4)")
    ap.add_argument("--cameras", default=None, help="comma list (default: all Phase-3 cameras)")
    ap.add_argument("--max-frames", type=int, default=0, help="DEBUG ONLY: first N timestamps")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.fps < 1:
        ap.error("fps must be positive")
    if a.output_dir and len(a.phase5_dir) > 1:
        ap.error("--output-dir needs a single --phase5-dir")
    for d in a.phase5_dir:
        run(Path(d), Path(a.output_dir) if a.output_dir else None, a.fps, a.axis_length, a.cameras, a.max_frames,
            a.device)


if __name__ == "__main__":
    main()
