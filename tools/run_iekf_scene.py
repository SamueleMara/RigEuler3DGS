"""Run one sparse alpha-aware IEKF frame step on a REAL trained scene.

CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md sections 24, 25, 34.

This loads an actual trained Gaussian model and its real cameras, builds the
frozen-appearance motion state from them, synthesizes the RGB-D measurement by
perturbing the pose of a chosen subset of Gaussians, and runs the fixed-prior
IEKF against it. It exercises exactly the production estimator path: analytic
CUDA Jacobians, matrix-free PCG, LM + nonlinear line search. No Adam, no
autograd (section 36).

Why a synthesized measurement rather than the dataset's own next frame: the
estimator needs y = h(x_true) for a KNOWN x_true in order to report pose error.
Pointing it at a real captured frame is a one-line change (pass --real-frame),
but then only the energy and the diagnostics are meaningful, not a pose error.

    python tools/run_iekf_scene.py --model output/merged_USB_100109 \
        --source input/USB_100109 --cameras 2 --iters 6
"""

from __future__ import annotations

import argparse
import math
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from estimator import (  # noqa: E402
    CameraSpec,
    GaussianMotionState,
    IEKFConfig,
    audit_chain,
    boxplus,
    build_linearization,
    iekf_frame_step,
    run_sequence,
)
from estimator.state import quaternion_wxyz_to_matrix  # noqa: E402
from tools.iekf_artifacts import (  # noqa: E402
    DATASET_RGBD,
    SYNTHETIC,
    VALID_SOURCES,
    ArtifactWriter,
    GroundTruthMotion,
    ObservationSource,
    axis_error_from_artifact,
    axis_polyline,
    ground_truth_round_trip,
    save_plot,
    save_residual_png,
    support_colors,
    write_ply,
    write_scene_html,
)


def load_scene(model_path: str, source_path: str | None, sh_degree: int = 3,
               *, inference_only: bool = False):
    """Load the trained Gaussians and the real cameras of a scene."""
    from argparse import Namespace

    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import GaussianModel
    from scene import Scene

    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipe = PipelineParams(parser)
    args = parser.parse_args([])
    args.model_path = model_path
    if source_path is not None:
        args.source_path = str(Path(source_path).resolve())
    args.sh_degree = sh_degree
    # ModelParams is built with sentinel=True, so anything not named here stays
    # a sentinel. Rather than guess, take the dataset layout from the trained
    # model's own cfg_args: the estimator must see exactly the scene the model
    # was fitted to, or the frozen appearance would not match the geometry.
    args.eval = False
    args.images = "images"
    args.depths = ""
    args.resolution = -1
    args.white_background = False
    args.train_test_exp = False
    # sentinel=True leaves every unnamed ModelParams field as None, and
    # `data_device` is not in cfg_args either. Camera.__init__ then calls
    # torch.device(None), which raises, prints the exception plus a "[Warning]
    # Custom device None failed" line once PER CAMERA, and falls back to cuda.
    # Setting it here produces exactly that fallback without the eight
    # exception traces at startup.
    args.data_device = "cuda"
    cfg = Path(model_path) / "cfg_args"
    if cfg.exists():
        saved = eval(cfg.read_text(),
                     {"Namespace": Namespace, "inf": float("inf"),
                      "nan": float("nan"), "true": True, "false": False})
        # `sapien_depth_is_metric` travels WITH `depths`. It is not a
        # preference, it is how the bytes behind `depths` are to be read: the
        # same .npy is a distance or its reciprocal depending on it. Restoring
        # one without the other loads the file and misinterprets it -- which
        # is what happened the first time, and only showed up because the
        # mis-read is now checked rather than silently trained on.
        for name in ("images", "depths", "sapien_depth_is_metric",
                     "resolution", "white_background",
                     "train_test_exp", "sh_degree", "eval"):
            if hasattr(saved, name):
                setattr(args, name, getattr(saved, name))
        if getattr(saved, "source_path", None) and source_path is None:
            args.source_path = saved.source_path
    dataset = model.extract(args)
    dataset.load_gt_parts = not inference_only
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    return dataset, pipe.extract(args), gaussians, scene


def split_camera_fold(cameras, frames, train_ids):
    """Filter camera specifications and their observations with identical IDs."""
    held_ids = [i for i in range(len(cameras)) if i not in train_ids]
    heldout = {'camera_ids':held_ids, 'train_camera_ids':list(train_ids),
        'cameras':[cameras[i] for i in held_ids],
        'frames':[([obs[i] for i in held_ids],fid) for obs,fid in frames]}
    return ([cameras[i] for i in train_ids],
            [([obs[i] for i in train_ids],fid) for obs,fid in frames],heldout)


def camera_specs(cams, limit: int) -> list[CameraSpec]:
    out = []
    for cam in cams[:limit]:
        out.append(CameraSpec(
            viewmatrix=cam.world_view_transform.cuda().contiguous(),
            projmatrix=cam.full_proj_transform.cuda().contiguous(),
            tanfovx=float(torch.tan(torch.tensor(cam.FoVx * 0.5))),
            tanfovy=float(torch.tan(torch.tensor(cam.FoVy * 0.5))),
            image_height=int(cam.image_height),
            image_width=int(cam.image_width)))
    return out


def state_from_gaussians(gaussians, subset=None) -> GaussianMotionState:
    """Frozen appearance (section 3): colour, opacity and scales are carried but
    never updated. DC-only radiance is used, which is section 3.1's strict
    frozen-radiance mode."""
    xyz = gaussians.get_xyz.detach()
    rot = torch.nn.functional.normalize(gaussians.get_rotation.detach(), dim=1)
    scales = gaussians.get_scaling.detach()
    opacity = gaussians.get_opacity.detach()
    # DC term of the SH -> constant colour, so delta c_i = 0 (section 3.1).
    dc = gaussians.get_features_dc.detach().reshape(xyz.shape[0], -1)[:, :3]
    colors = (dc * 0.28209479177387814 + 0.5).clamp(0.0, 1.0)
    if subset is not None:
        xyz, rot, scales = xyz[subset], rot[subset], scales[subset]
        opacity, colors = opacity[subset], colors[subset]
    n = xyz.shape[0]
    zeros = torch.zeros((n, 3), device=xyz.device, dtype=xyz.dtype)
    return GaussianMotionState(
        mu=xyz.contiguous(), R=quaternion_wxyz_to_matrix(rot).contiguous(),
        v=zeros.clone(), omega=zeros.clone(),
        colors=colors.contiguous(), opacity=opacity.contiguous(),
        scales=scales.contiguous())


def dataset_joint_ground_truth(source: Path) -> dict | None:
    """The dataset's own joint ground truth (V2 section 16).

    `metadata/joint_info.json` names the joint and its type; `frame_values.csv`
    carries the true per-frame joint coordinate. This is loaded for REPORTING and
    for scoring the estimate afterwards -- it is never fed to the estimator,
    which sees only RGB and depth.
    """
    info_path = source / "metadata" / "joint_info.json"
    values_path = source / "metadata" / "frame_values.csv"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    sel = info.get("selected_joint") or {}
    out = {"joint_name": sel.get("name"), "joint_type_raw": sel.get("type"),
           "dof": sel.get("dof"), "limits": sel.get("limits"),
           "source": str(info_path)}
    # SAPIEN calls an unlimited 1-DoF rotational joint "continuous"; the
    # estimator's vocabulary is prismatic/revolute.
    out["joint_type"] = {"continuous": "revolute", "revolute": "revolute",
                         "prismatic": "prismatic"}.get(sel.get("type"))
    if values_path.exists():
        import csv as _csv
        with values_path.open() as fh:
            rows = list(_csv.reader(fh))
        header, body = rows[0], rows[1:]
        col = header.index(sel.get("name")) if sel.get("name") in header else 1
        out["gt_q"] = [float(r[col]) for r in body if len(r) > col]
        out["gt_q_source"] = str(values_path)
        if len(out["gt_q"]) >= 2:
            d = [b - a for a, b in zip(out["gt_q"], out["gt_q"][1:])]
            out["gt_dq_per_frame"] = sum(d) / len(d)
    return out


# Datasets in this repository store RGB either as PNG (the SAPIEN exports) or
# as JPG (the captured fridge episodes). The extension is a storage detail, so
# it is resolved here rather than hard-coded at each use.
RGB_EXTENSIONS = (".png", ".jpg", ".jpeg")


def rgb_frame_path(source: Path, cam_name: str, t: int) -> Path | None:
    """<source>/rgb/<cam>/frame_%06d.<ext> for whichever extension exists."""
    base = source / "rgb" / cam_name / f"frame_{t:06d}"
    for ext in RGB_EXTENSIONS:
        p = base.with_suffix(ext)
        if p.exists():
            return p
    return None


def _count_dataset_frames(source: Path, cams_all) -> int:
    """How many time steps the dataset actually has (section 4.1's T)."""
    name = str(getattr(cams_all[0], "image_name", "cam_000")).split("/")[0]
    d = source / "rgb" / name
    if not d.exists():
        return 0
    return max(len(sorted(d.glob(f"frame_*{ext}"))) for ext in RGB_EXTENSIONS)


def dataset_rgbd_frames(cams_all, cfg, args, truth, source: Path, *,
                        direction: str = "forward"):
    """V2 section 16: observations straight from the dataset's own RGB-D files.

    No synthetic motion is applied anywhere in this function -- that is the
    whole point. The dataset is 8 fixed viewpoints observed over 60 time steps,
    so a SEQUENCE steps time and fuses the cameras at each step; the camera
    poses come from the trained model and never move.

    RGB  : <source>/rgb/<cam>/frame_%06d.{png,jpg,jpeg}
    depth: <source>/depth_npy/<cam>/frame_%06d.npy   metric Z in metres,
           0 = no return (the renderer's D = Z/(A+eps) is the same quantity
           wherever alpha is high, so no unit conversion is applied; a zero is
           marked INVALID rather than filled in, which would be the hidden
           synthetic target section 16 forbids).

    Returns the frames plus the per-frame RGB/depth file IDs and the dt that
    was used, both of which section 16 requires to be logged.
    """
    import numpy as np
    from PIL import Image

    start = args.sequence_start
    # Section 4.1: full-sequence mode uses every frame the dataset provides.
    available = _count_dataset_frames(source, cams_all)
    if args.all_frames:
        length = available - start
    elif args.window_length > 0:
        length = min(args.window_length, available - start)
    else:
        length = args.sequence_length
    if length <= 0 or start + length > available:
        raise SystemExit(
            f"section 4: requested frames [{start}, {start + length}) but the "
            f"dataset provides {available}")
    # Section 3: the whole camera set unless explicitly restricted.
    n_cam = len(cams_all) if args.all_cameras else args.cameras
    if args.all_cameras and n_cam != len(cams_all):
        raise AssertionError(
            f"section 3: --all-cameras requested but {n_cam} of "
            f"{len(cams_all)} would be used")
    specs = camera_specs(cams_all, n_cam)
    # cam_000/frame_000000.png -> cam_000
    names = [str(getattr(c, "image_name", f"cam_{i:03d}")).split("/")[0]
             for i, c in enumerate(cams_all[:n_cam])]

    frames, ids = [], []
    for k in range(length):
        t = start + k
        per_cam, cam_ids = [], []
        for ci, cam_name in enumerate(names):
            rgb_path = rgb_frame_path(source, cam_name, t)
            dep_path = source / "depth_npy" / cam_name / f"frame_{t:06d}.npy"
            if rgb_path is None:
                raise SystemExit(
                    f"missing dataset RGB frame "
                    f"{source / 'rgb' / cam_name / f'frame_{t:06d}'}"
                    f"[{'|'.join(RGB_EXTENSIONS)}]")
            dest = "cpu" if args.frames_on_cpu else "cuda"
            im = torch.from_numpy(
                np.asarray(Image.open(rgb_path).convert("RGB"),
                           dtype=np.float32) / 255.0).permute(2, 0, 1).to(dest)
            h, w = int(im.shape[1]), int(im.shape[2])
            if dep_path.exists():
                dz = torch.from_numpy(
                    np.load(dep_path).astype(np.float32)).to(dest)
                valid = dz > 0
            else:
                dz = torch.zeros((h, w), device=im.device)
                valid = torch.zeros((h, w), dtype=torch.bool, device=im.device)
            per_cam.append((im.contiguous(), dz.contiguous(), valid))
            cam_ids.append({"camera": cam_name,
                            "rgb_frame_id": rgb_path.name,
                            "depth_frame_id": (dep_path.name if dep_path.exists()
                                               else None),
                            "depth_valid_pixels": int(valid.sum()),
                            "depth_min_m": float(dz[valid].min()) if bool(valid.any()) else None,
                            "depth_max_m": float(dz[valid].max()) if bool(valid.any()) else None})
        frames.append((per_cam, specs, float(args.dt)))
        ids.append({"frame": k, "dataset_frame_index": t,
                    "dt_used_s": float(args.dt), "cameras": cam_ids})

    any_depth = any(c["depth_frame_id"] for f in ids for c in f["cameras"])
    provenance = {
        "frames": ids, "num_frames": len(frames), "num_cameras": len(names),
        "cameras": names,
        "sequence_start": start, "sequence_length": length,
        "any_depth_available": any_depth,
        "rgb_root": str(source / "rgb"),
        "depth_root": str(source / "depth_npy"),
        "depth_units": "metres, metric Z; 0 means no return and is marked invalid",
        "dt_source": ("metadata/job_config_resolved.yaml records fps: null and "
                      f"duration_sec: null, so --dt was used verbatim "
                      f"({args.dt} s) and is persisted here"),
        "synthetic_motion_applied": False,
    }
    # Sections 4-6: the backward pass reads the SAME files through the SAME
    # loader and only reverses the order afterwards. Loading the dataset twice
    # with two index conventions is how the two directions drift apart.
    from utils.sequence_direction import (BACKWARD, annotate_forward,
                                          normalise, reverse_sequence)
    if normalise(direction) == BACKWARD:
        return reverse_sequence(frames, provenance)
    return frames, annotate_forward(provenance)


def _jsonable(value):
    """Non-finite floats become null: a NaN in a debug file is indistinguishable
    from a parse error downstream."""
    import math

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def save_artifacts(out_dir: Path, payload: dict, tensors: dict | None = None,
                   quiet: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in payload.items():
        (out_dir / f"{name}.json").write_text(json.dumps(_jsonable(obj), indent=2))
    if tensors:
        torch.save(tensors, out_dir / "iekf_state.pt")
    if quiet:
        return
    print(f"\n  artifacts written to {out_dir}")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"    {f.name}  ({f.stat().st_size} bytes)")


def score_against_dataset_joint_gt(gt: dict, selected, res, args) -> dict:
    """Score the estimate against the dataset's own joint ground truth.

    What this dataset actually provides is the joint TYPE and the per-frame
    joint coordinate q(t). It does NOT provide the axis in world coordinates,
    so no axis-angle error is computed: section 16 forbids fabricating accuracy
    metrics the data cannot support.

    q is compared as a RATE. The dataset's q is an absolute joint angle
    (starting at -0.5 here) while the reduced filter's q starts at zero at the
    reduction frame, so only their increments are comparable.
    """
    out = {"gt_joint_type": gt.get("joint_type"),
           "gt_joint_type_raw": gt.get("joint_type_raw"),
           "estimated_joint_type": (selected.kind if selected else None),
           "joint_type_correct": bool(selected is not None
                                      and selected.kind == gt.get("joint_type")),
           "estimated_axis_direction": (selected.axis.tolist()
                                        if selected is not None else None),
           "axis_error_reported": False,
           "axis_error_note": ("the dataset provides joint type and q(t) but no "
                               "world-frame axis, so an axis error cannot be "
                               "computed from it"),
           "dq_ground_truth": gt.get("gt_dq_per_frame"),
           "computed_from": gt.get("source")}
    qs = [f.q for f in res.frames if f.q is not None]
    if len(qs) >= 2 and gt.get("gt_dq_per_frame"):
        steps = [float(b) - float(a) for a, b in zip(qs, qs[1:])]
        dq_est = sum(steps) / len(steps)
        out["estimated_q_trace"] = [float(q) for q in qs]
        out["dq_estimated"] = dq_est
        out["dq_ratio"] = dq_est / gt["gt_dq_per_frame"]
        out["gt_q_window"] = gt["gt_q"][args.sequence_start:
                                        args.sequence_start + args.sequence_length]
    return out


def _report_kpart(kres, writer, args, out_dir, truth, cams, window, cfg):
    """Print and persist the K-part run (section 26's required debug outputs)."""
    print()
    if kres.get("status") == "geometric":
        # CODEX_IMPLEMENT_GEOMETRIC_RIGID_MOTION_JOINT_DISCOVERY sections 29-30.
        rp = kres.get("rigid_parts", {})
        print(f"  === RESULT: geometric joint discovery ===")
        print(f"      rigid parts: counts {rp.get('counts')}, "
              f"{'converged' if rp.get('converged') else 'NOT converged'}, "
              f"E = {rp.get('energy', float('nan')):.2f}")
        for j in kres.get("geometric_joints", []) or []:
            if j is None:
                continue
            ne = j["normalized_error"]
            verdict = j["selected"] or j["status"]
            print(f"      part {j['part']}: {str(verdict).upper()}")
            print(f"          normalized geometric error  "
                  f"E_P = {ne['prismatic']:.6e}   E_R = {ne['revolute']:.6e}")
            print(f"          gap {ne['absolute_gap']:.3e} vs "
                  f"indistinguishable floor {ne['indistinguishable_floor']:.3e}")
            if j["axis"] is not None:
                print(f"          axis {[round(x, 5) for x in j['axis']]}")
            if j["axis_point"] is not None:
                print(f"          axis point "
                      f"{[round(x, 5) for x in j['axis_point']]}")
            if j["q"] is not None:
                q = j["q"]
                print(f"          q(t) {[round(x, 5) for x in q[:8]]}"
                      f"{' ...' if len(q) > 8 else ''}")
                print(f"          total sweep {max(q) - min(q):.6f}")
        print(f"      runtime {kres.get('runtime_s', 0):.1f} s, peak CUDA "
              f"{kres.get('peak_cuda_gib', 0):.2f} GiB")
        return
    if kres.get("status") != "RESOLVED":
        print(f"  === RESULT: {kres.get('status')} ===")
        for r in (kres.get("discovery") or {}).get("reason", []):
            print(f"      - {r}")
    else:
        scene = kres["scene"]
        print(f"  === RESULT: K={kres['K']} parts discovered ===")
        for p_ in scene["parts"]:
            if p_.get("kind") == "prismatic":
                print(f"      part {p_['part_index']}: PRISMATIC  axis "
                      f"{[round(x, 4) for x in p_['axis']]}")
            elif p_.get("kind") == "revolute":
                print(f"      part {p_['part_index']}: REVOLUTE   axis "
                      f"{[round(x, 4) for x in p_['axis']]}  "
                      f"closest point {[round(x, 4) for x in p_['closest_point_to_origin']]}")
            else:
                print(f"      part: UNRESOLVED")
        for k, q in enumerate(kres["q_traj"], start=1):
            dq = [b - a for a, b in zip(q, q[1:])]
            mean_dq = sum(dq) / len(dq) if dq else 0.0
            print(f"      q_{k}: {[round(x, 5) for x in q[:6]]}"
                  f"{' ...' if len(q) > 6 else ''}   mean dq/frame {mean_dq:+.6f}")
        ro = kres["reduced_observability"]
        print(f"      §17 reduced-state rank {ro['observable_rank']}/{ro['dimension']}")
    print(f"  runtime {kres.get('runtime_s', 0):.1f} s, peak CUDA "
          f"{kres.get('peak_cuda_gib', 0):.2f} GiB")
    writer.json("kpart_summary", {k: v for k, v in kres.items()
                                  if k not in ("labels", "partition", "result")})
    if kres.get("partition") is not None:
        writer.json("supernodes", kres["partition"].to_dict())
    if kres.get("labels") is not None:
        writer.tensor("support_gaussian_labels", kres["labels"].cpu())
    man = writer.write_manifest(False)
    print(f"\n  §26 artifacts: {len(man['present'])} written to {out_dir}")


def _gate_config(args):
    from joint_reduction.coherence_gate import GateConfig
    return GateConfig(window=args.gate_window,
                      max_omega_turn_deg=args.gate_max_turn_deg,
                      max_speed_change=args.gate_max_speed_change,
                      max_normalized_residual=args.gate_max_norm_residual,
                      min_support_jaccard=args.gate_min_support_jaccard,
                      min_observable_rank=3, require_certified=True,
                      require_stable_model=True)


def _emit_v2_artifacts(writer, res, truth, cams, cfg, args, out_dir: Path, *,
                       gt_path, axis_metrics, synthetic: bool) -> None:
    """V2 sections 17, 22 and 23's mandatory outputs.

    Section 23: "No critical result may exist only in terminal scrollback."
    Everything printed above is therefore also written here, in the names the
    spec fixes, plus the manifest that says what is present and what is not.
    """
    from estimator.certificate_v2 import assess
    from estimator.observability import analyze_blocks
    from estimator.supernodes import build_supernodes

    n = truth.num_gaussians

    # --- §4.5 trust region + linear solver histories -------------------
    tr_rows, ls_rows = [], []
    for f in res.frames:
        d = getattr(f, "diagnostics", None)
        rows = getattr(d, "rows", []) if d is not None else []
        hist = getattr(d, "trust_region", None) if d is not None else None
        if hist is not None:
            for r in hist.records:
                tr_rows.append({"frame": f.index, **r.to_dict()})
        for row in rows:
            if row.get("event") in ("accepted", "rejected"):
                ls_rows.append({
                    "frame": f.index, "iteration": row.get("iteration"),
                    "pcg_iterations": row.get("pcg_iterations"),
                    "pcg_rel_residual": row.get("pcg_rel_residual"),
                    "pcg_reason": row.get("pcg_reason"),
                    "forcing_eta": row.get("forcing_eta"),
                    "boundary_hit": row.get("boundary_hit"),
                    "gradient_norm": row.get("gradient_norm")})
    writer.json("trust_region_history", {"records": tr_rows,
                                         "num_records": len(tr_rows)})
    writer.json("linear_solver_history", {"records": ls_rows,
                                          "num_records": len(ls_rows)})

    # --- §5.3 observability --------------------------------------------
    obs_payload = {"per_frame": []}
    for f in res.frames:
        d = getattr(f, "diagnostics", None)
        for row in (getattr(d, "rows", []) if d is not None else []):
            if row.get("event") in ("observability", "observability_final"):
                obs_payload["per_frame"].append({"frame": f.index, **row})
    lin = build_linearization(truth, cams, eps_depth=cfg.eps_depth)
    prec = [(torch.ones_like(v.rgb),
             (v.alpha > cfg.min_alpha_for_depth).to(v.depth.dtype))
            for v in lin.views]
    fisher = lin.build_block_fisher(prec)
    sigma = torch.tensor([cfg.scale_sigma_mu] * 3 + [cfg.scale_sigma_phi] * 3,
                         device=fisher.device, dtype=torch.float64)
    report = analyze_blocks(fisher, tau=args.observability_tau, scale=sigma)
    obs_payload["final_state"] = report.to_dict()
    writer.json("observability", obs_payload)
    print(f"  §5 observability: {report.fraction_weak:.1%} weak directions, "
          f"{report.active_dimension}/{report.global_dimension} active state "
          f"dimensions")

    # --- §3 supernodes ---------------------------------------------------
    part = build_supernodes(truth.mu, target_size=args.supernode_size,
                            seed=args.seed)
    writer.json("supernodes", {
        **part.to_dict(),
        "free_state_dim": 6 * n, "grouped_state_dim": 6 * part.num_supernodes,
        "reduction_factor": n / max(part.num_supernodes, 1)})
    print(f"  §3 supernodes: {part.num_supernodes} groups, state "
          f"{6 * n} -> {6 * part.num_supernodes} DoF "
          f"({n / max(part.num_supernodes, 1):.0f}x)")

    # --- §6 support ------------------------------------------------------
    # The free stage here moves every Gaussian, so the support that was
    # actually USED is all-moving. Recorded as what it is; the full alternating
    # inference is exercised by the T-S1..T-S4 suite.
    sup = getattr(res, "support", None)
    if sup is not None:
        hard = torch.tensor(sup["support_hard"], dtype=torch.long)
        gain = torch.tensor(sup["supernode_gain"], dtype=torch.float32)
        writer.tensor("support_hard", hard)
        # A calibrated read of the same evidence, not a second estimate.
        writer.tensor("support_probability",
                      torch.sigmoid((gain - sup["threshold"])
                                    / max(sup["mad"], 1e-12)))
        writer.json("support_iteration_history", {
            "method": ("V2 §6.3 step 1: robust threshold on the alpha-aware "
                       "achievable-motion gain 1/2 b^T F^-1 b, from the exact "
                       "analytic renderer sensitivity"),
            "iterations": [{k: v for k, v in sup.items()
                            if k not in ("supernode_gain", "support_hard")}]})
    else:
        writer.tensor("support_hard", torch.ones(part.num_supernodes, dtype=torch.long))
        writer.tensor("support_probability", torch.ones(part.num_supernodes))
        writer.json("support_iteration_history", {
            "iterations": [],
            "note": ("support inference did not run on this frame set (no "
                     "reduction attempt reached it, or --no-support-inference "
                     "was passed), so every supernode is recorded as moving")})

    # --- §11 joint hypotheses / selection / geometry ---------------------
    red = res.reduction
    hyps = []
    if red is not None:
        hyps = [{"kind": h.kind, "score": h.score, "axis": h.axis.tolist(),
                 "point": (h.point.tolist() if h.point is not None else None),
                 "pitch": h.pitch, "omega_norm": h.omega_norm,
                 "u_norm": h.u_norm, "note": h.note} for h in red.hypotheses]
    writer.json("joint_hypotheses", {"hypotheses": hyps})
    sel = red.selected if red is not None else None
    writer.json("joint_selected", {
        "selected": (sel.kind if sel else None),
        "accepted": (red.confident if red is not None else False),
        "reasons": (red.reasons if red is not None else []),
        "reduced_at_frame": res.reduced_at_frame,
        "axis_metrics_vs_ground_truth": axis_metrics})

    axis_geom = {"has_joint": sel is not None}
    if sel is not None:
        from joint_reduction.line_geometry import PluckerLine
        e = sel.axis / sel.axis.norm()
        axis_geom["direction"] = e.tolist()
        axis_geom["kind"] = sel.kind
        if getattr(sel, "point", None) is not None:
            line = PluckerLine.from_point_direction(sel.point, e)
            axis_geom.update({
                "plucker_moment": line.moment.tolist(),
                "closest_point_to_origin": line.closest_point.tolist(),
                "gauge_residual_e_dot_m": float(line.direction @ line.moment),
                "passes_through_origin": bool(float(line.moment.norm()) < 1e-6)})
        else:
            axis_geom["note"] = ("a prismatic joint is a direction, not a line: "
                                 "it has no axis point and no Plucker moment")
    writer.json("joint_axis_geometry", axis_geom)
    writer.json("joint_window_refinement", {
        "performed": False,
        "note": ("the sequence runner selects the joint from the filtered "
                 "multi-frame twist (section 11.3's initializer). The full "
                 "window objective with joint axis refinement is implemented in "
                 "joint_reduction/window_discovery.py and exercised by "
                 "tests/test_v2_window_discovery.py.")})

    # --- §14 handover ----------------------------------------------------
    writer.json("handover_refit", {
        "performed": False,
        "reduced_at_frame": res.reduced_at_frame,
        "note": ("canonical_handover() in joint_reduction/handover.py performs "
                 "the section 14 refit and is exercised by "
                 "tests/test_v2_reduced_and_handover.py; this runner reduces "
                 "directly from canonical geometry, so no deformed reference "
                 "is ever used.")})

    # --- §15 / §23 q trace ----------------------------------------------
    rows = [[f.index, f.mode, f.dt,
             "" if f.q is None else f"{float(f.q):.9f}",
             "" if f.qdot is None else f"{float(f.qdot):.9f}",
             f"{f.energy_initial:.9f}", f"{f.energy_final:.9f}"]
            for f in res.frames]
    writer.csv("q_trace", ["frame", "mode", "dt", "q", "qdot",
                           "energy_initial", "energy_final"], rows)

    # --- §17 convergence status ------------------------------------------
    statuses = []
    for f in res.frames:
        d = getattr(f, "diagnostics", None)
        hist = getattr(d, "trust_region", None) if d is not None else None
        if hist is None:
            statuses.append({"frame": f.index, "status": "NOT_APPLICABLE",
                             "note": "reduced-stage frame or LM ablation"})
            continue
        c = assess(hist, getattr(d, "observability", None),
                   innovation_stats={"final_energy": f.energy_final},
                   max_iters_reached=(len(hist.records) >= cfg.max_iters))
        statuses.append({"frame": f.index, **c.to_dict()})
    writer.json("convergence_status", {
        "per_frame": statuses,
        "statuses_present": sorted({s["status"] for s in statuses}),
        "note": ("V2 section 17: a frame is never CERTIFIED from a monotone "
                 "objective alone; see the ten conditions in each entry")})
    for s in statuses:
        if s.get("status") != "NOT_APPLICABLE":
            print(f"  §17 frame {s['frame']}: {s['status']}")

    # --- §23 final state --------------------------------------------------
    writer.tensor("state_final", {
        "mu": res.state.mu.cpu(), "R": res.state.R.cpu(),
        "v": res.state.v.cpu(), "omega": res.state.omega.cpu(),
        "q": (res.joint_state.q.cpu() if res.joint_state is not None else None),
        "qdot": (res.joint_state.qdot.cpu() if res.joint_state is not None else None),
        "supernode_id": part.supernode_id.cpu(),
        "canonical_hash": part.canonical_hash()})

    if synthetic:
        writer.json("metrics", {
            "axis": axis_metrics,
            "frames_certified": sum(
                1 for s in statuses if s.get("status") == "CERTIFIED_LOCAL_MAP"),
            "frames_total": len(statuses),
            "energy_first_frame": (res.frames[0].energy_initial
                                   if res.frames else None),
            "energy_last_frame": (res.frames[-1].energy_final
                                  if res.frames else None),
            "observability": report.to_dict(),
            "note": ("computed from the reloaded ground_truth_motion.json "
                     "artifact (section 2.2 step 5)")})

    # --- §22 debug visualizations ----------------------------------------
    if not args.no_visualizations:
        _write_visualizations(writer, res, truth, cams, cfg, lin, part, report,
                              sel, tr_rows, out_dir)

    man = writer.write_manifest(synthetic)
    print(f"\n  §23 artifacts: {len(man['present'])} written to {out_dir}")
    if man["missing"]:
        print(f"  MISSING (section 23): {', '.join(man['missing'])}")
    else:
        print("  every section 23 artifact is present")


def _write_visualizations(writer, res, truth, cams, cfg, lin, part, report,
                          sel, tr_rows, out_dir: Path) -> None:
    """V2 section 22's twelve required views."""
    viz = out_dir / "visualizations"
    viz.mkdir(parents=True, exist_ok=True)
    made = []

    # 1. canonical cloud coloured by support probability
    prob = torch.ones(truth.num_gaussians)
    made.append(str(write_ply(viz / "support_probability.ply", truth.mu,
                              support_colors(prob))))

    # 9. observability / Fisher score mapped to supernodes
    sid = part.supernode_id.long().clamp_min(0)
    rank = report.rank.float().cpu()
    per_g = rank[sid.cpu().clamp(max=rank.shape[0] - 1)] if rank.numel() else \
        torch.zeros(truth.num_gaussians)
    score = (per_g / 6.0).clamp(0, 1).reshape(-1, 1)
    made.append(str(write_ply(viz / "observability.ply", truth.mu,
                              torch.cat([1 - score, score,
                                         0.2 * torch.ones_like(score)], dim=1))))

    # 2. group motion vectors + 3/4. fitted axis, 5. camera frustums
    motion = None
    if float(res.state.v.norm()) > 0 or float((res.state.mu - truth.mu).norm()) > 0:
        motion = (truth.mu, res.state.mu)
    axis_pts = None
    extra = {"gaussians": truth.num_gaussians,
             "supernodes": part.num_supernodes,
             "fraction_weak": report.fraction_weak}
    if sel is not None:
        length = float((truth.mu - truth.mu.mean(dim=0)).norm(dim=1).max()) * 1.2
        axis_pts = axis_polyline(sel.axis, getattr(sel, "point", None),
                                 length=max(length, 0.5))
        extra["joint"] = sel.kind
    cam_pos = []
    for c in cams:
        try:
            inv = torch.linalg.inv(c.viewmatrix.T.double())
            cam_pos.append(inv[:3, 3].tolist())
        except Exception:
            pass
    made.append(str(write_scene_html(
        viz / "scene.html", truth.mu, support_colors(prob),
        axis_points=axis_pts, motion_vectors=motion, cameras=cam_pos,
        title="IEKF canonical scene", extra=extra)))

    # 6/7/8. per-camera RGB residual, depth residual, robust precision
    from estimator.measurement import build_precision
    for i, view in enumerate(lin.views):
        made.append(str(save_residual_png(viz / f"cam{i}_rgb_render.png", view.rgb)))
        made.append(str(save_residual_png(viz / f"cam{i}_depth_render.png", view.depth)))
        made.append(str(save_residual_png(viz / f"cam{i}_alpha.png", view.alpha)))
        zero = torch.zeros_like(view.rgb)
        p = build_precision(zero, valid_mask=None, sigma_rgb=cfg.rgb_sigma,
                            robust=cfg.robust_measurement, nu=cfg.student_t_nu,
                            layout=("rgb", "rgb", "rgb"), channel_dim=0)
        made.append(str(save_residual_png(viz / f"cam{i}_rgb_precision.png", p)))

    # 11. trust radius / rho / PCG residual
    if tr_rows:
        made.append(str(save_plot(
            viz / "trust_region.png",
            {"trust radius": [r["trust_radius_after"] for r in tr_rows],
             "rho": [r["rho"] for r in tr_rows]},
            "V2 §4.5 trust radius and gain ratio")))
        made.append(str(save_plot(
            viz / "linear_residual.png",
            {"PCG relative residual": [r["pcg_rel_residual"] for r in tr_rows],
             "forcing eta": [r["forcing_eta"] for r in tr_rows]},
            "V2 §4.3 inexact-Newton forcing vs achieved residual", logy=True)))

    # 10. q / qdot trace
    qs = [(f.index, f.q, f.qdot) for f in res.frames if f.q is not None]
    if qs:
        made.append(str(save_plot(
            viz / "q_trace.png",
            {"q": [float(q) for _, q, _ in qs],
             "qdot": [float(qd) if qd is not None else float("nan")
                      for _, _, qd in qs]},
            "V2 §15 reduced joint coordinate", xlabel="reduced frame")))

    # 12. support change over outer iterations
    made.append(str(save_plot(
        viz / "energy.png",
        {"energy_initial": [f.energy_initial for f in res.frames],
         "energy_final": [f.energy_final for f in res.frames]},
        "MAP energy per frame", xlabel="frame", logy=True)))

    writer.json("visualizations", {"directory": str(viz), "files": made,
                                   "count": len(made)})
    print(f"  §22 visualizations: {len(made)} files in {viz}")


def run_full_technique(truth, cams, cfg, args, out_dir: Path,
                       artifacts: dict | None = None,
                       cams_all_for_dataset=None,
                       dataset_source_path: str | None = None) -> None:
    """Section 2's ladder on real scene geometry with a known rigid motion.

    The motion is imposed rather than taken from the dataset so the recovered
    joint can be checked against a known axis; the GEOMETRY, cameras, colours
    and opacities are the real trained scene.
    """
    n = truth.num_gaussians
    dev, dt_t = truth.mu.device, truth.mu.dtype
    dt = args.dt
    axis = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=dt_t)
    centroid = truth.mu.mean(dim=0)
    writer = ArtifactWriter(out_dir)

    # ---- V2 section 16: real dataset RGB-D, no synthetic motion ----------
    if args.observation_source == DATASET_RGBD:
        source = Path(dataset_source_path)
        frames, frame_ids = dataset_rgbd_frames(
            cams_all_for_dataset, cfg, args, truth, source)
        # The dataset loader builds its OWN camera specs (section 3 may ask for
        # all 8 while main() defaulted to --cameras 2). Everything downstream
        # must use those, or zip() silently truncates the per-camera
        # observations to the shorter list and the run quietly fuses fewer
        # cameras than it claims.
        cams = frames[0][1]
        joint_gt = (None if args.joint_direct_rgbd or args.joint_direct_gaussian
                    else dataset_joint_ground_truth(source))
        frame_ids["joint_ground_truth"] = joint_gt
        print(f"\n  === REAL RGB-D SEQUENCE: {len(frames)} dataset frames "
              f"[{args.sequence_start}, {args.sequence_start + len(frames)}) ===")
        if not frame_ids["any_depth_available"]:
            print(f"  note: {frame_ids['depth_note']}")
        writer.json("observation_source",
                    ObservationSource(DATASET_RGBD, frame_ids).to_dict())
        if joint_gt:
            q = joint_gt.get("gt_q") or []
            lo = args.sequence_start
            # --all-frames leaves sequence_length at 0; the window actually used
            # is what the loader reported, not the flag.
            n_used = frame_ids.get("num_frames") or args.sequence_length
            hi = min(lo + n_used, len(q))
            print(f"  dataset joint GT: {joint_gt['joint_name']} "
                  f"type={joint_gt['joint_type_raw']} -> {joint_gt['joint_type']}, "
                  f"q[{lo}:{hi}] = {[round(x, 5) for x in q[lo:hi]][:6]}"
                  f"{' ...' if hi - lo > 6 else ''}")
            print(f"  dq per frame = {joint_gt.get('gt_dq_per_frame'):.6f} "
                  f"(the estimator never sees this; it is used only to score "
                  f"the result)")
            writer.json("dataset_joint_ground_truth", joint_gt)
        else:
            print("  this dataset carries no joint ground truth; only estimated "
                  "support/joint/trajectory and internal diagnostics are reported")
        gt_path = None
        kind = None
        dataset_joint_gt = joint_gt
    else:
        kind = "revolute" if abs(args.omega) > 1e-9 else "prismatic"
        print(f"\n  === FULL TECHNIQUE: {args.sequence} frames of {kind} motion "
              f"(speed {args.speed}, omega {args.omega}) ===")
        print(f"  observation source: {SYNTHETIC} -- synthetic imposed motion "
              f"on real reconstructed geometry.")
        print(f"  This is NOT joint discovery from a real RGB-D articulated "
              f"sequence (V2 section 2.1).")

        # Section 2.2's immutable gt_ record, written BEFORE the frames are
        # generated from it.
        gt_axis = ([1.0, 0.0, 0.0] if kind == "prismatic" else axis.tolist())
        rate = args.speed if kind == "prismatic" else args.omega
        gt = GroundTruthMotion(
            gt_joint_type=kind, gt_axis_direction=gt_axis,
            gt_axis_point=(None if kind == "prismatic" else centroid.tolist()),
            gt_axis_moment=None,
            gt_q=[rate * dt * (k + 1) for k in range(args.sequence)],
            gt_qdot=[rate] * args.sequence, gt_dt=dt,
            gt_centroid=centroid.tolist(), gt_seed=args.seed,
            gt_num_gaussians=n)
        if kind == "revolute":
            from joint_reduction.line_geometry import PluckerLine
            gl = PluckerLine.from_point_direction(centroid, axis)
            gt.gt_axis_moment = gl.moment.tolist()

        frames, generated = [], []
        for k in range(args.sequence):
            mu_k, R_k = gt.transform(truth.mu, truth.R, k)
            generated.append((mu_k, R_k))
            moved = GaussianMotionState(mu=mu_k, R=R_k, v=truth.v,
                                        omega=truth.omega, colors=truth.colors,
                                        opacity=truth.opacity, scales=truth.scales)
            lin = build_linearization(moved, cams, eps_depth=cfg.eps_depth)
            frames.append(([(v.rgb.clone(), v.depth.clone(),
                             (v.alpha > cfg.min_alpha_for_depth).clone())
                            for v in lin.views], cams, dt))

        # Section 2.2's mandatory round-trip: save, reload, regenerate, compare.
        gt_path = out_dir / "ground_truth_motion.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        rt = ground_truth_round_trip(gt, gt_path, truth.mu, truth.R, generated)
        writer.written.append(gt_path.name)
        print(f"  §2.2 ground-truth round-trip: "
              f"{'OK' if rt['ok'] else 'MISMATCH -- metric invalid'} "
              f"(max |dmu| {rt['max_mu_mismatch']:.3e}, "
              f"max |dR| {rt['max_R_mismatch']:.3e})")
        if not rt["ok"]:
            raise SystemExit(
                "section 2.2: the ground-truth artifact does not regenerate the "
                "state that produced the observations, so every synthetic "
                "metric from this run would be invalid")
        dataset_joint_gt = None
        writer.json("observation_source", ObservationSource(
            SYNTHETIC, {"imposed_kind": kind, "speed": args.speed,
                        "omega": args.omega, "dt": dt,
                        "ground_truth_round_trip": rt}).to_dict())
        # §6.4 / §23: GT support. The imposed motion moves every Gaussian here,
        # so the honest GT label is "all moving" -- recorded, not inferred.
        writer.tensor("ground_truth_support",
                      torch.ones(n, dtype=torch.long))

    # ======================================================================
    # Section 5 + 31 priority 1-2: the windowed K-part discovery IS the
    # production path when --window-discovery is given. The sequential IEKF
    # selector below remains only as the section 28 ablation.
    # ======================================================================
    if args.window_discovery:
        from tools.kpart_pipeline import (
            per_camera_diagnostics,
            resolve_cameras,
            run_kpart_discovery,
        )

        window = [(obs, fid) for (obs, _specs, _dt), fid in
                  zip(frames, [f["dataset_frame_index"]
                               for f in frame_ids["frames"]]
                      if args.observation_source == DATASET_RGBD
                      else list(range(len(frames))))]
        used, cam_audit = resolve_cameras(
            cams_all_for_dataset if cams_all_for_dataset is not None else cams,
            len(cams), args.all_cameras, None,
            fold=(args.camera_fold if args.camera_fold >= 0 else None),
            num_folds=args.camera_folds)
        heldout_data = None
        if args.camera_folds > 1 and args.camera_fold >= 0:
            # Section 8: the held-out block must actually be dropped before the
            # estimator sees it, not merely recorded as excluded.
            train_ids = cam_audit['cameras_used']
            cams,window,heldout_data = split_camera_fold(cams,window,train_ids)
        if len(cams) != cam_audit["num_cameras_used"]:
            raise AssertionError(
                f"section 3: the audit says {cam_audit['num_cameras_used']} "
                f"cameras but {len(cams)} specs reach the estimator")
        percam, cam_total = per_camera_diagnostics(truth, cams, window[0][0], cfg)
        writer.json("camera_usage", {**cam_audit, "per_camera": percam,
                                     **cam_total})
        writer.json("frame_usage", {
            "num_frames": len(window),
            "frame_ids": [f for _o, f in window],
            "sequence_start": args.sequence_start,
            "all_frames_requested": bool(args.all_frames),
            "window_length": args.window_length or len(window),
            "mode": ("full_sequence" if args.all_frames else "explicit_window")})
        print(f"\n  §3 cameras: {len(cams)} used "
              f"{'(ALL dataset cameras)' if args.all_cameras else ''}")
        print(f"  §4 frames: {len(window)} "
              f"({'full sequence' if args.all_frames else 'explicit window'})")

        reference = window[0][0] if args.observation_source == DATASET_RGBD else None
        kres = run_kpart_discovery(truth, cams, window, cfg, args, writer,
                                   reference_obs=reference, gt_labels=None,
                                   verbose=True, heldout_data=heldout_data)
        if args.observation_source == DATASET_RGBD and (args.joint_direct_rgbd or args.joint_direct_gaussian):
            joint_gt = dataset_joint_ground_truth(Path(dataset_source_path))
            if joint_gt:
                writer.json('dataset_joint_ground_truth', joint_gt)
        _report_kpart(kres, writer, args, out_dir, truth, cams, window, cfg)
        return

    P0 = torch.zeros((n, 12, 12), device=dev, dtype=dt_t)
    P0[:, range(12), range(12)] = torch.tensor(
        [1e-2]*3 + [1e-2]*3 + [1.0]*3 + [1e-1]*3, device=dev, dtype=dt_t)
    res = run_sequence(truth, P0, frames, cfg=cfg,
                       reduce_after=max(2, args.sequence // 2),
                       reduce_only_after_convergence=args.reduce_after_convergence,
                       reduction_kwargs={"max_relative_residual": args.coherence_residual},
                       infer_support_before_reduction=not args.no_support_inference,
                       use_multi_criterion_gate=not args.single_residual_gate,
                       gate_config=_gate_config(args),
                       verbose=True)
    print()
    print(res.format())
    axis_metrics = None
    if res.reduction is not None and res.reduction.selected is not None:
        sel = res.reduction.selected
        print(f"\n  recovered joint: {sel.kind}  axis {[round(float(x),4) for x in sel.axis]}")
        if gt_path is not None:
            # Section 2.2 step 5: errors from the RELOADED artifact, never from
            # the in-memory variables that generated the observations.
            axis_metrics = axis_error_from_artifact(
                gt_path, sel.axis, sel.kind,
                getattr(sel, "point", None))
            print(f"  axis error vs the imposed motion (from "
                  f"ground_truth_motion.json): "
                  f"{axis_metrics['axis_angle_error_deg']:.3f} deg "
                  f"(type {'CORRECT' if axis_metrics['joint_type_correct'] else 'WRONG'})")
        elif dataset_joint_gt:
            # Section 16: this dataset DOES carry joint ground truth, so it is
            # scored. The estimator never saw it.
            axis_metrics = score_against_dataset_joint_gt(
                dataset_joint_gt, sel, res, args)
            print(f"  joint type: GT {axis_metrics['gt_joint_type']} vs "
                  f"estimated {axis_metrics['estimated_joint_type']} -> "
                  f"{'CORRECT' if axis_metrics['joint_type_correct'] else 'WRONG'}")
            if axis_metrics.get("dq_estimated") is not None:
                print(f"  joint rate: GT dq/frame "
                      f"{axis_metrics['dq_ground_truth']:.6f} vs estimated "
                      f"{axis_metrics['dq_estimated']:.6f} "
                      f"({axis_metrics['dq_ratio']:.0%} of truth)")
            print("  NOTE: the dataset gives the joint TYPE and q(t) but not an "
                  "axis in world coordinates, so no axis-angle error is "
                  "reported -- only what the data actually supports")
        else:
            print("  no joint ground truth exists for this dataset, so no "
                  "accuracy metric is reported (V2 section 16)")
    certs = [f.certificate for f in res.frames if f.certificate is not None]
    if certs:
        print(f"  free-stage frames certified: "
              f"{sum(1 for c in certs if c.certified)}/{len(certs)}")


    art = artifacts if artifacts is not None else {}
    # §23 requires these by name whether or not the optional passes ran, so a
    # reader can tell "not run" from "lost".
    writer.json("config", art.get("config", {}))
    writer.json("jacobian_chain_audit", art.get("jacobian_chain_audit", {
        "performed": False,
        "note": "pass --audit-chain to validate every analytic Jacobian link "
                "against central differences on this scene's geometry"}))
    adj = [{"frame": f.index, **row}
           for f in res.frames
           for row in (getattr(getattr(f, "diagnostics", None), "rows", []) or [])
           if row.get("event") == "adjoint_test"]
    writer.json("adjoint_test", {
        "performed": bool(adj), "records": adj,
        "note": (None if adj else
                 "pass --adjoint-test to assert <Hv,z> == <v,H^T z> on every "
                 "frame's live linearization")})
    art["sequence"] = {
        "imposed_motion": {"kind": kind, "speed": args.speed, "omega": args.omega,
                           "axis": axis.tolist(), "dt": dt},
        "reduced_at_frame": res.reduced_at_frame,
        "frames": [{"index": f.index, "mode": f.mode, "dt": f.dt,
                    "energy_initial": f.energy_initial,
                    "energy_final": f.energy_final, "q": f.q, "qdot": f.qdot,
                    "certified": (f.certificate.certified if f.certificate else None),
                    "median_gain_ratio": (f.certificate.median_gain_ratio
                                          if f.certificate else None),
                    "monotone": (f.certificate.monotone if f.certificate else None),
                    "gradient_first": (f.certificate.records[0].gradient_norm
                                       if f.certificate and f.certificate.records else None),
                    "gradient_last": (min(r.gradient_norm for r in f.certificate.records)
                                      if f.certificate and f.certificate.records else None)}
                   for f in res.frames],
    }
    if res.reduction is not None:
        c = res.reduction.coherence
        art["reduction"] = {
            "accepted": res.reduction.confident,
            "reasons": res.reduction.reasons,
            "coherence": {"v0": c.v0.tolist(), "omega": c.omega.tolist(),
                          "center": c.center.tolist(), "residual": c.residual,
                          "relative_residual": c.relative_residual,
                          "rank": c.rank, "condition_number": c.condition_number,
                          "is_static": c.is_static, "is_coherent": c.is_coherent,
                          "n_points": c.n_points},
            "hypotheses": [{"kind": h.kind, "score": h.score,
                            "axis": h.axis.tolist(),
                            "point": (h.point.tolist() if h.point is not None else None),
                            "pitch": h.pitch, "omega_norm": h.omega_norm,
                            "u_norm": h.u_norm, "note": h.note}
                           for h in res.reduction.hypotheses],
            "selected": (res.reduction.selected.kind
                         if res.reduction.selected else None)}
    # per-iteration certificate rows, so the convergence evidence is auditable
    art["certificates"] = [
        {"frame": f.index,
         "rows": [{"iteration": r.iteration, "accepted": r.accepted,
                   "energy_before": r.energy_before, "energy_after": r.energy_after,
                   "predicted_decrease": r.predicted_decrease,
                   "actual_decrease": r.actual_decrease,
                   "gain_ratio": r.gain_ratio,
                   "significant": r.significant(f.certificate.significance_floor),
                   "gradient_norm": r.gradient_norm, "alpha": r.alpha, "lm": r.lm,
                   "pcg_iterations": r.pcg_iterations,
                   "pcg_rel_residual": r.pcg_rel_residual}
                  for r in f.certificate.records],
         "certified": f.certificate.certified,
         "reasons": f.certificate.reasons()}
        for f in res.frames if f.certificate is not None]
    save_artifacts(out_dir, art, quiet=True,
                   tensors={"mu": res.state.mu.cpu(), "R": res.state.R.cpu(),
                            "v": res.state.v.cpu(), "omega": res.state.omega.cpu(),
                            "q": (res.joint_state.q.cpu()
                                  if res.joint_state is not None else None)})

    # ---- V2 sections 17, 22, 23 ----------------------------------------
    # Last, so the manifest sees everything this run produced.
    _emit_v2_artifacts(writer, res, truth, cams, cfg, args, out_dir,
                       gt_path=gt_path, axis_metrics=axis_metrics,
                       synthetic=(args.observation_source == SYNTHETIC))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="trained model directory")
    ap.add_argument("--source", default=None,
                    help="dataset source directory; defaults to the one the model was trained on")
    ap.add_argument("--cameras", type=int, default=2, help="synchronized cameras to fuse")
    ap.add_argument("--max-gaussians", type=int, default=4000,
                    help="cap the state size; the free per-Gaussian stage is a "
                         "bootstrap (section 28), not the final representation")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--pcg-iters", type=int, default=25)
    ap.add_argument("--perturb-mu", type=float, default=0.004)
    ap.add_argument("--perturb-phi", type=float, default=0.01)
    ap.add_argument("--dt", type=float, default=1.0 / 30.0)
    ap.add_argument("--no-rgb", action="store_true")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sequence", type=int, default=0,
                    help="run the FULL technique over N frames of synthetic rigid "
                         "motion: free IEKF -> rigid fit -> P/R joint -> reduced filter")
    ap.add_argument("--speed", type=float, default=0.3,
                    help="translation speed (world units/s) for --sequence")
    ap.add_argument("--omega", type=float, default=0.0,
                    help="rotation rate (rad/s) about +y through the centroid for "
                         "--sequence; non-zero makes the joint revolute")
    ap.add_argument("--coherence-residual", type=float, default=0.35,
                    help="section 28 gate: the largest share of the per-Gaussian "
                         "velocity field the fitted rigid twist may leave "
                         "unexplained before reduction is declined")
    ap.add_argument("--out", default=None,
                    help="directory for the debug artifacts; defaults to "
                         "<model>/iekf_debug")
    ap.add_argument("--adjoint-test", action="store_true",
                    help="section 35 T8: assert <Hv,z> == <v,H^T z> on every "
                         "frame's live linearization")
    ap.add_argument("--no-multiview", action="store_true",
                    help="section 25 ablation: use only the first camera")
    ap.add_argument("--reduce-after-convergence", action="store_true",
                    help="only attempt joint reduction on a frame whose IEKF "
                         "solve was certified")
    ap.add_argument("--audit-chain", action="store_true",
                    help="section 40: validate every analytic Jacobian link against "
                         "central differences, on this scene's own geometry")
    # ---- V2 section 2.1 / 16 -------------------------------------------
    ap.add_argument("--observation-source", default=SYNTHETIC,
                    choices=list(VALID_SOURCES),
                    help="V2 §2.1: where the observations come from. "
                         f"'{SYNTHETIC}' imposes a known motion on the real "
                         f"reconstructed geometry; '{DATASET_RGBD}' consumes the "
                         f"dataset's own synchronized frames and forbids --speed "
                         f"and --omega")
    ap.add_argument("--sequence-start", type=int, default=0,
                    help=f"V2 §16: first dataset frame index in "
                         f"{DATASET_RGBD} mode")
    ap.add_argument("--sequence-length", type=int, default=0,
                    help=f"V2 §16: number of dataset frames in {DATASET_RGBD} "
                         f"mode")
    ap.add_argument("--full-gaussians", action="store_true",
                    help="use every Gaussian in the model, ignoring --max-gaussians")
    # ---- V2 sections 3, 4, 5 --------------------------------------------
    ap.add_argument("--supernodes", action="store_true",
                    help="V2 §3: build the canonical supernode partition and "
                         "report the grouped state dimension")
    ap.add_argument("--supernode-size", type=int, default=64)
    ap.add_argument("--no-trust-region", action="store_true",
                    help="V2 §1.3 ablation: revert to LM + line search, which "
                         "accepts zero steps at a large PCG budget")
    ap.add_argument("--trust-radius", type=float, default=1.0,
                    help="V2 §4.1: initial trust radius, in RMS prior-sigma units")
    ap.add_argument("--observability-tau", type=float, default=1e-2,
                    help="V2 §5.2: scaled Fisher eigenvalue below which a "
                         "direction is not reported as observed")
    # ---- K-part multiview multi-frame contract (sections 3,4,16,18,22,23) ----
    ap.add_argument("--phase2-rigid-screw-iekf", action="store_true",
                    help="section 28: run the NEW Phase-2 rigid-screw IEKF "
                         "instead of the old free-SE(3) Phase 2. The frozen "
                         "Phase-1 support is kept; no free rigid fit runs")
    ap.add_argument("--phase2-screw-iters", type=int, default=8,
                    help="nonlinear IEKF iterations per frame")
    ap.add_argument("--phase2-parallel-pr", action="store_true",
                    help="Stage B: also fit the exact prismatic and exact "
                         "revolute Phase-2 models on the same frozen support "
                         "and compare them on held-out cameras (sections "
                         "30, 31). Produces evidence, not a label.")
    ap.add_argument("--phase2-pr-folds", type=int, default=3,
                    help="number of camera folds for the section 31 held-out "
                         "comparison")
    ap.add_argument("--phase2-block-frames", type=int, default=1,
                    help="section 13: frames per camera/time block for the "
                         "cluster-robust score (1 = finest partition)")
    ap.add_argument("--phase2-null-passes", type=int, default=2,
                    help="section 6: P-null refit passes before the test")
    ap.add_argument("--phase2-screw-cg-iters", type=int, default=40,
                    help="truncated-CG iterations inside each step")
    ap.add_argument("--camera-folds", type=int, default=0,
                    help="section 8: split the cameras into this many blocks "
                         "for repeatability refits (0 disables folding)")
    ap.add_argument("--camera-fold", type=int, default=-1,
                    help="section 8: which camera block to HOLD OUT of this "
                         "run; requires --camera-folds")
    ap.add_argument("--all-cameras", action="store_true",
                    help="\u00a73: use every valid dataset camera (8 for USB). "
                         "Asserts that all of them are actually used")
    ap.add_argument("--all-frames", action="store_true",
                    help="\u00a74.1: full-sequence mode, t = 0..T-1 (60 for USB)")
    ap.add_argument("--window-length", type=int, default=0,
                    help="\u00a74.2: explicit window L; 0 means the whole sequence")
    ap.add_argument("--num-moving-parts", type=int, default=0,
                    help="\u00a77/\u00a723: K, the known number of moving rigid "
                         "parts. A CARDINALITY prior only -- never labels")
    ap.add_argument("--window-discovery", action="store_true",
                    help="\u00a75: windowed multiview K-part discovery performs "
                         "the final P/R selection, not the bootstrap twist")
    ap.add_argument('--joint-direct-gaussian', action='store_true',
                    help='information-weighted joint fit directly from Phase-1 Gaussian tracks')
    ap.add_argument('--joint-direct-rgbd', action='store_true',
                    help='analytic direct RGB-D joint estimation; unresolved C never falls back to A/B')
    ap.add_argument('--joint-heldout-validation', action='store_true',
                    help='fast training-camera joint refits and frozen held-out prediction; external camera folds rerun Phases 1/2')
    ap.add_argument('--joint-confidence', type=float, default=.95,
                    help='global statistical confidence for curvature and predictive evidence (default 0.95)')
    ap.add_argument('--joint-block-frames', type=int, default=10,
                    help='contiguous frames per camera/time uncertainty block')
    ap.add_argument('--joint-iters', type=int, default=30,
                    help='GN/LM iteration budget for each direct joint fit')
    ap.add_argument("--canonical-handover", action="store_true",
                    help="\u00a76: refit from immutable canonical geometry at "
                         "handover")
    ap.add_argument("--solver", type=str, default="trust_region",
                    choices=["trust_region", "lm", "auto"],
                    help="\u00a716: step-control solver")
    ap.add_argument("--alpha-threshold", type=float, default=0.05,
                    help="\u00a718 Option B: alpha-normalised depth is used only "
                         "where coverage exceeds this")
    ap.add_argument("--bootstrap-iters", type=int, default=6)
    ap.add_argument("--discovery-outer", type=int, default=6,
                    help="\u00a711: outer alternation iterations")
    ap.add_argument("--potts-weight", type=float, default=1.0)
    ap.add_argument("--lambda-support", type=float, default=1.0)
    ap.add_argument("--lambda-temporal", type=float, default=1.0)
    ap.add_argument("--sigma-q", type=float, default=1.0)
    ap.add_argument("--tau-energy", type=float, default=1e-4)
    ap.add_argument("--q-bracket", type=float, default=0.5)
    ap.add_argument("--support-alpha", type=float, default=1e-3,
                    help="\u00a76: significance of the support test. The null is "
                         "chi-square from R_eff, Bonferroni-corrected over the "
                         "supernodes; dataset-independent by construction")
    ap.add_argument("--geometric-joints", action="store_true", default=True,
                    help="CODEX_IMPLEMENT_GEOMETRIC_RIGID_MOTION_JOINT_DISCOVERY "
                         "sections 1/25: decide the joint type from the geometric "
                         "fit of the recovered SE(3) part trajectories. This is "
                         "the production path.")
    ap.add_argument("--profile-likelihood-joints", action="store_false",
                    dest="geometric_joints",
                    help="section 1: fall back to the superseded "
                         "profile-likelihood/BIC classifier. Diagnostic only.")
    # ---- CODEX_ADD_VISUAL_AND_NUMERICAL_DEBUG_GEOMETRIC_PIPELINE §50 -----
    # A read-only debug layer. Section 0 rule 10: none of these may change an
    # inference decision, and `tests/test_debug_layer.py` asserts that the
    # estimator's outputs are identical with and without them.
    ap.add_argument("--make-geometric-debug-html", action="store_true",
                    help="section 1: write geometric_motion_debug.html")
    ap.add_argument("--debug-numerics", action="store_true",
                    help="section 13: write the debug_numeric/ directory")
    ap.add_argument("--debug-render-panels", action="store_true",
                    help="section 12: per-frame/per-camera image panels under "
                         "render_debug/")
    ap.add_argument("--export-debug-gaussian-frames", action="store_true",
                    help="section 41: export per-frame PLYs with part colours")
    ap.add_argument("--debug-frame-stride", type=int, default=1,
                    help="section 50: keep every Nth frame in the HTML")
    ap.add_argument("--debug-gaussian-subsample", type=int, default=4000,
                    help="section 50: Gaussians drawn in the HTML (0 = all)")
    ap.add_argument("--debug-ply-color", choices=("part", "appearance"),
                    default="part",
                    help="section 41: colour the exported PLYs by part label "
                         "(default, section 4's palette) or by the Gaussians' "
                         "own appearance")
    ap.add_argument("--debug-all", action="store_true",
                    help="section 50: one flag that enables the complete "
                         "debug package")
    ap.add_argument("--reopen-support-rounds", type=int, default=2,
                    help="section 26: rounds of z <-> T <-> joint geometry "
                         "after the joint type is known")
    ap.add_argument("--rigid-refine-iters", type=int, default=12,
                    help="section 9: Levenberg iterations per frame when "
                         "fitting T_{k,t} against the RGB-D window")
    ap.add_argument("--keep-null-diagnostic", action="store_true", default=False,
                    help="\u00a75: also compute the frame-0/1 contrast as a "
                         "DIAGNOSTIC. It is never used to decide support")
    ap.add_argument("--null-quantile", type=float, default=0.99,
                    help="\u00a714: quantile of the no-motion excess that real "
                         "motion must clear to count as support evidence")
    ap.add_argument("--q-prior-scale", type=float, default=1.0,
                    help="\u00a712: Gaussian prior scale on the joint step used "
                         "to regularise the achievable-gain ratio")
    ap.add_argument("--model-margin", type=float, default=0.02,
                    help="\u00a75: relative evidence margin below which the P/R "
                         "call is UNRESOLVED")
    ap.add_argument("--enumerate-limit", type=int, default=3,
                    help="\u00a720: enumerate {P,R}^K exactly up to this K")
    ap.add_argument("--baselines", type=int, nargs="+",
                    default=[1, 2, 3, 5, 10, 15],
                    help="\u00a714: temporal separations used for initialisation")
    ap.add_argument("--no-model-error", action="store_true",
                    help="\u00a713 ablation: disable R_model and use sensor noise "
                         "only")
    ap.add_argument("--model-error-beta", type=float, default=1.0)
    ap.add_argument("--model-error-blur", type=int, default=1)
    ap.add_argument("--handover-epsilon", type=float, default=1e-6)
    ap.add_argument("--frames-on-cpu", action="store_true", default=True,
                    help="section 22: stage the window observations in host "
                         "memory and move each frame to the device on demand. "
                         "60 frames x 8 cameras pin about 2.4 GiB of RGB-D on "
                         "the GPU otherwise, which does not fit alongside the "
                         "renders on an 8 GiB card")
    ap.add_argument("--frames-on-gpu", action="store_false", dest="frames_on_cpu")
    ap.add_argument("--reduced-accel-psd", type=float, default=10.0,
                    help="\u00a71.7/\u00a719: process noise for the reduced "
                         "(q, qdot) filter; the joint acceleration is unmodelled")
    ap.add_argument("--split-supernodes", action="store_true",
                    help="\u00a715: refine partition topology by splitting "
                         "supernodes that straddle a part boundary")
    ap.add_argument("--tau-split", type=float, default=0.25)
    ap.add_argument("--depth-sigma", type=str, default="constant:0.02:0.0",
                    help="\u00a76 depth sensor model, kind[:a[:b]]. On real "
                         "dataset depth the RECONSTRUCTION error usually "
                         "dominates the sensor: measured 0.039 m model-vs-sensor "
                         "on USB frame 0, so 0.02 understates it")
    ap.add_argument("--single-residual-gate", action="store_true",
                    help="V2 \u00a718 ablation: use the old one-number coherence "
                         "residual threshold instead of the windowed "
                         "multi-criterion gate")
    ap.add_argument("--gate-window", type=int, default=3,
                    help="V2 \u00a718: frames the reduction gate must agree over")
    ap.add_argument("--gate-max-turn-deg", type=float, default=12.0)
    ap.add_argument("--gate-max-speed-change", type=float, default=0.35)
    ap.add_argument("--gate-max-norm-residual", type=float, default=3.0,
                    help="V2 \u00a718: largest twist residual in units of the "
                         "filter's own predicted velocity uncertainty")
    ap.add_argument("--gate-min-support-jaccard", type=float, default=0.5)
    ap.add_argument("--no-support-inference", action="store_true",
                    help="V2 \u00a76 ablation: fit the joint over every Gaussian "
                         "instead of the inferred moving support")
    ap.add_argument("--no-visualizations", action="store_true",
                    help="skip V2 §22's debug visualizations")
    args = ap.parse_args()
    if getattr(args, "debug_all", False):
        # Section 50: "one command must enable the complete package".
        args.make_geometric_debug_html = True
        args.debug_numerics = True
        args.debug_render_panels = True
        args.export_debug_gaussian_frames = True

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(args.seed)

    # ---- V2 section 16's rules, enforced before anything is loaded -------
    explicit = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
    if args.observation_source == DATASET_RGBD:
        forbidden = {"--speed", "--omega"} & explicit
        if forbidden:
            raise SystemExit(
                f"V2 section 16: {', '.join(sorted(forbidden))} is forbidden in "
                f"--observation-source {DATASET_RGBD}. That mode consumes the "
                f"dataset's own RGB-D frames; applying synthetic motion to them "
                f"would be hidden synthetic target generation, which section 16 "
                f"explicitly prohibits.")
        if args.sequence_length <= 0 and not args.all_frames:
            raise SystemExit(
                f"--observation-source {DATASET_RGBD} needs --sequence-length "
                f"or --all-frames (how many dataset frames to consume)")
    if args.window_discovery and args.num_moving_parts <= 0:
        raise SystemExit(
            "section 23: --window-discovery requires --num-moving-parts K. K is "
            "the one permitted GT structural prior and must be stated "
            "explicitly; it is a cardinality constraint, not a label source.")
    if args.num_moving_parts < 0:
        raise SystemExit("--num-moving-parts must be non-negative")

    if args.joint_heldout_validation:
        args.joint_direct_rgbd = True
    if args.joint_direct_rgbd:
        args.joint_direct_gaussian = True
    if not 0 < args.joint_confidence < 1 or args.joint_block_frames < 1 or args.joint_iters < 1:
        raise SystemExit('joint confidence must be in (0,1); block width and iteration budget must be positive')
    if (args.joint_direct_rgbd or args.joint_direct_gaussian) and not args.window_discovery:
        raise SystemExit('direct joint estimation requires --window-discovery')
    print(f"loading {args.model}")
    dataset, pipe, gaussians, scene = load_scene(args.model, args.source,
        inference_only=args.joint_direct_rgbd or args.joint_direct_gaussian)
    cams_all = scene.getTrainCameras()
    print(f"  {gaussians.get_xyz.shape[0]} Gaussians, {len(cams_all)} cameras")

    # Keep the densest sub-cloud so the run is representative but bounded.
    total = gaussians.get_xyz.shape[0]
    subset = None
    if args.full_gaussians:
        print(f"  --full-gaussians: estimating all {total} Gaussians "
              f"(V2 §1.2's full-scene regime)")
    elif total > args.max_gaussians:
        subset = torch.randperm(total, device=gaussians.get_xyz.device)[:args.max_gaussians]
    truth = state_from_gaussians(gaussians, subset)
    cams = camera_specs(cams_all, args.cameras)
    n = truth.num_gaussians
    if args.observation_source == DATASET_RGBD:
        print(f"  estimating {n} Gaussians at "
              f"{cams[0].image_width}x{cams[0].image_height}; the camera set is "
              f"resolved by the dataset loader below")
    else:
        print(f"  estimating {n} Gaussians against {len(cams)} camera(s) "
              f"at {cams[0].image_width}x{cams[0].image_height}")

    cfg = IEKFConfig(max_iters=args.iters, pcg_max_iters=args.pcg_iters,
                     use_rgb=not args.no_rgb, use_depth=not args.no_depth,
                     robust_measurement=True,
                     multiview=not args.no_multiview,
                     debug_adjoint_test=args.adjoint_test,
                     # ---- V2 sections 3, 4, 5 ----
                     use_trust_region=not args.no_trust_region,
                     trust_radius_init=args.trust_radius,
                     use_supernodes=args.supernodes,
                     supernode_target_size=args.supernode_size,
                     observability_tau=args.observability_tau,
                     depth_sigma_model=args.depth_sigma,
                     supernode_seed=args.seed)
    print(f"  step control: "
          f"{'V2 §4 scaled trust region (Steihaug)' if cfg.use_trust_region else 'LM + line search (V2 §1.3 ablation)'}")

    # y = h(x_true): the measurement this frame is supposed to explain.
    t0 = time.perf_counter()
    lin_true = build_linearization(truth, cams, eps_depth=cfg.eps_depth)
    rgbd_gt = [(v.rgb.clone(), v.depth.clone(),
                (v.alpha > cfg.min_alpha_for_depth).clone()) for v in lin_true.views]
    torch.cuda.synchronize()
    print(f"  measurement rendered in {(time.perf_counter()-t0)*1e3:.0f} ms "
          f"({lin_true.views[0].cache.num_rendered} Gaussian-tile instances)")

    # Start from a perturbed pose, which is what the filter has to undo.
    delta = torch.zeros((n, 12), device=truth.mu.device, dtype=truth.mu.dtype)
    delta[:, :3] = torch.randn((n, 3), device=delta.device) * args.perturb_mu
    delta[:, 3:6] = torch.randn((n, 3), device=delta.device) * args.perturb_phi
    start = boxplus(truth, delta)

    P0 = torch.zeros((n, 12, 12), device=delta.device, dtype=delta.dtype)
    diag_vals = torch.tensor([1e-2] * 3 + [1e-2] * 3 + [1e-1] * 3 + [1e-1] * 3,
                             device=delta.device, dtype=delta.dtype)
    P0[:, range(12), range(12)] = diag_vals

    def pose_err(a, b):
        dmu = (a.mu - b.mu).norm(dim=1)
        dR = a.R @ b.R.transpose(1, 2)
        cos = ((dR.diagonal(dim1=1, dim2=2).sum(-1) - 1) * 0.5).clamp(-1, 1)
        return float(dmu.mean()), float(torch.rad2deg(torch.arccos(cos)).mean())

    out_dir = Path(args.out) if args.out else Path(args.model) / "iekf_debug"
    artifacts: dict = {"config": {
        "model": args.model, "cameras": len(cams), "gaussians": n,
        "image": [cams[0].image_width, cams[0].image_height],
        "iters": args.iters, "pcg_iters": args.pcg_iters, "dt": args.dt,
        "sequence": args.sequence, "speed": args.speed, "omega": args.omega,
        "coherence_residual": args.coherence_residual,
        "perturb_mu": args.perturb_mu, "perturb_phi": args.perturb_phi,
        "use_rgb": cfg.use_rgb, "use_depth": cfg.use_depth}}

    # ---- section 40: is the derivative chain itself correct, link by link? --
    if args.audit_chain:
        print("\n  Jacobian chain audit (section 40), on this scene's geometry:")
        k = min(64, n)
        audit = audit_chain(
            truth.mu[:k].double(), truth.R[:k].double(), truth.scales[:k].double(),
            truth.opacity[:k].double(),
            cams[0].viewmatrix.double(), cams[0].projmatrix.double(),
            cams[0].image_width, cams[0].image_height,
            cams[0].tanfovx, cams[0].tanfovy,
            torch.tensor([cams[0].image_width / 2.0, cams[0].image_height / 2.0],
                         device=truth.mu.device, dtype=torch.float64))
        print(audit.format())
        artifacts["jacobian_chain_audit"] = {
            "ok": audit.ok, "broken_links": audit.broken(),
            "links": [{"name": l.name, "rel_error": l.rel_error,
                       "abs_error": l.abs_error, "magnitude": l.magnitude,
                       "eps_used": l.eps_used, "n_samples": l.n_samples,
                       "ok": l.ok, "note": l.note} for l in audit.links]}

    if args.sequence > 0 or args.observation_source == DATASET_RGBD:
        run_full_technique(truth, cams, cfg, args, out_dir, artifacts,
                           cams_all_for_dataset=cams_all,
                           dataset_source_path=dataset.source_path)
        return

    e0, r0 = pose_err(start, truth)
    print(f"\n  start error: mean |dmu| {e0:.6f}   mean |dphi| {r0:.4f} deg")

    t0 = time.perf_counter()
    out, P_post, diag = iekf_frame_step(start, P0, rgbd_gt, cams, dt=args.dt, cfg=cfg)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    e1, r1 = pose_err(out, truth)

    # Section 34's per-iteration log.
    print(f"\n  {'it':>3} {'event':>9} {'energy':>14} {'meas':>12} {'prior':>10} "
          f"{'lm':>9} {'alpha':>6} {'pcg':>4} {'pcg_rel':>9} {'max|dmu|':>9}")
    for row in diag.rows:
        if row.get("event") == "posterior":
            continue
        print(f"  {row.get('iteration', -1):>3} {str(row.get('event')):>9} "
              f"{row.get('energy', float('nan')):>14.5f} "
              f"{row.get('measurement', float('nan')):>12.5f} "
              f"{row.get('prior', float('nan')):>10.5f} "
              f"{row.get('lm', float('nan')):>9.2e} "
              f"{row.get('alpha', 0.0):>6.3f} "
              f"{str(row.get('pcg_iterations', '-')):>4} "
              f"{row.get('pcg_rel_residual', float('nan')) or float('nan'):>9.2e} "
              f"{row.get('max_dmu', float('nan')):>9.2e}")

    # ---- section 14/34: the convergence certificate -----------------------
    cert = getattr(diag, "certificate", None)
    if cert is not None:
        print("\n  convergence certificate (sections 14, 34):")
        print(cert.format())

    energies = [r["energy"] for r in diag.rows if r.get("event") in ("initial", "accepted")]
    monotone = all(b <= a + 1e-6 for a, b in zip(energies, energies[1:]))
    accepted = sum(1 for r in diag.rows if r.get("event") == "accepted")

    print(f"\n  final error: mean |dmu| {e1:.6f} ({100*(1-e1/max(e0,1e-30)):+.1f}%)   "
          f"mean |dphi| {r1:.4f} deg ({100*(1-r1/max(r0,1e-30)):+.1f}%)")
    print(f"  MAP energy {energies[0]:.4f} -> {energies[-1]:.4f}  "
          f"monotone={monotone}  accepted={accepted}/{args.iters}")
    print(f"  wall clock {elapsed:.2f} s  ({elapsed/max(args.iters,1):.2f} s/iteration)")

    # Section 27: report weak observability instead of pretending to have fixed it.
    prec = [(torch.ones_like(v.rgb), (v.alpha > cfg.min_alpha_for_depth).to(v.depth.dtype))
            for v in lin_true.views]
    fisher = lin_true.build_block_fisher(prec)
    ev = torch.linalg.eigvalsh(fisher.double())
    weak = int((ev[:, 0] < 1e-6).sum())
    print(f"  observability: median min Fisher eigenvalue {float(ev[:, 0].median()):.3e}, "
          f"{weak}/{n} ({100*weak/n:.1f}%) weakly observable pose blocks")
    print(f"  posterior finite={bool(torch.isfinite(P_post).all())}")

    artifacts["single_frame"] = {
        "start_error_mu": e0, "start_error_phi_deg": r0,
        "final_error_mu": e1, "final_error_phi_deg": r1,
        "energy_initial": energies[0], "energy_final": energies[-1],
        "monotone": monotone, "accepted": accepted, "max_iters": args.iters,
        "wall_clock_s": elapsed,
        "observability": {
            "median_min_fisher_eigenvalue": float(ev[:, 0].median()),
            "weak_blocks": weak, "n": n, "weak_fraction": weak / n},
        "posterior_finite": bool(torch.isfinite(P_post).all()),
        "diagnostics": diag.rows}
    if cert is not None:
        artifacts["certificate"] = {
            "certified": cert.certified, "monotone": cert.monotone,
            "median_gain_ratio": cert.median_gain_ratio,
            "n_significant": len(cert.significant),
            "n_at_noise_floor": cert.n_at_noise_floor,
            "acceptance_rate": cert.acceptance_rate,
            "energy_reduction": cert.energy_reduction,
            "reasons": cert.reasons(),
            "rows": [{"iteration": r.iteration, "accepted": r.accepted,
                      "energy_after": r.energy_after,
                      "predicted_decrease": r.predicted_decrease,
                      "actual_decrease": r.actual_decrease,
                      "gain_ratio": r.gain_ratio,
                      "significant": r.significant(cert.significance_floor),
                      "gradient_norm": r.gradient_norm, "alpha": r.alpha,
                      "lm": r.lm, "pcg_iterations": r.pcg_iterations,
                      "pcg_rel_residual": r.pcg_rel_residual}
                     for r in cert.records]}
    save_artifacts(out_dir, artifacts,
                   tensors={"mu": out.mu.cpu(), "R": out.R.cpu(),
                            "v": out.v.cpu(), "omega": out.omega.cpu(),
                            "P_post": P_post.cpu()})


if __name__ == "__main__":
    main()
