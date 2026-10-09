#!/usr/bin/env python3
"""Strict endpoint-geometry closed-loop FM smoke probe.

This uses only G0 and G_T endpoint geometry, computes P0 spatial mismatch, trains
independent endpoint FM directions, and performs a Bayesian posterior update.
Intermediate frames, when used by the companion video probe, are observations
only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.bidirectional_endpoint_fm import compute_endpoint_cycle_consistency, train_backward_endpoint_fm, train_forward_endpoint_fm
from utils.bidirectional_fm import FlowTrainConfig, local_spacing
from utils.bidirectional_spatial_mismatch import compute_bidirectional_spatial_mismatch
from utils.endpoint_video_candidate_search import rank_target_gaussians_from_video_support
from utils.dynamic_prior import DynamicPrior, save_dynamic_prior
from utils.matching_debug import binary_metrics, tensor_summary, write_json
from utils.motion_correspondence import pair_fixed_cameras
from utils.sparse_video_gaussian_attachment import build_projected_video_endpoint_support
from utils.static_dynamic_bayesian_update import update_dynamic_posterior
from utils.temporal_pixel_fm import match_foreground_pixels_to_next
from utils.video_endpoint_transport import build_sparse_video_endpoint_gaussian_support


def _resolve_ply(path: str, iteration: int) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return p
    q = p / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
    if q.is_file():
        return q
    raise FileNotFoundError(q)


def _xyz(path: Path) -> torch.Tensor:
    v = PlyData.read(str(path))["vertex"].data
    return torch.from_numpy(np.stack((v["x"], v["y"], v["z"]), axis=1).astype("float32"))


def _read_vertex(path: Path):
    return PlyData.read(str(path))["vertex"].data


def _maybe_gt_dynamic(vertex) -> torch.Tensor | None:
    names = vertex.dtype.names or ()
    if "gt_dynamic" not in names:
        return None
    return torch.from_numpy(np.asarray(vertex["gt_dynamic"]).astype(bool, copy=False))


def _logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(torch.as_tensor(value).float().clamp(1.0e-6, 1.0 - 1.0e-6))


def _label_from_probability(probability: torch.Tensor, static_threshold: float, dynamic_threshold: float) -> torch.Tensor:
    p = torch.as_tensor(probability).float()
    label = torch.full_like(p, 0.5)
    label[p <= float(static_threshold)] = 0.0
    label[p >= float(dynamic_threshold)] = 1.0
    return label


def _append_ply_fields(vertex, fields: dict[str, torch.Tensor]) -> np.ndarray:
    names = vertex.dtype.names or ()
    arrays = {name: vertex[name] for name in names}
    dtype = [(name, vertex.dtype[name]) for name in names]
    n = len(vertex)
    for name, value in fields.items():
        tensor = torch.as_tensor(value).detach().cpu().float().reshape(-1)
        if tensor.numel() != n or name in arrays:
            continue
        arrays[name] = tensor.numpy().astype(np.float32, copy=False)
        dtype.append((name, "f4"))
    out = np.empty(n, dtype=dtype)
    for name in out.dtype.names or ():
        out[name] = arrays[name]
    return out


def _write_debug_ply(input_ply: Path, output_ply: Path, fields: dict[str, torch.Tensor]) -> None:
    data = PlyData.read(str(input_ply))
    vertex = data["vertex"].data
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    enriched = _append_ply_fields(vertex, fields)
    PlyData([PlyElement.describe(enriched, "vertex")], text=data.text).write(str(output_ply))


def _sample_ids(n: int, max_points: int, seed: int) -> torch.Tensor:
    if n <= int(max_points):
        return torch.arange(n)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(n, generator=gen)[: int(max_points)]


def _write_static_dynamic_html(
    path: Path,
    xyz: torch.Tensor,
    probability: torch.Tensor,
    confidence: torch.Tensor,
    displacement: torch.Tensor,
    label: torch.Tensor,
    *,
    max_points: int,
    seed: int,
) -> None:
    ids = _sample_ids(int(xyz.shape[0]), int(max_points), int(seed))
    x = xyz.index_select(0, ids).detach().cpu().float()
    p = probability.index_select(0, ids).detach().cpu().float()
    c = confidence.index_select(0, ids).detach().cpu().float()
    d = displacement.index_select(0, ids).detach().cpu().float()
    l = label.index_select(0, ids).detach().cpu().float()
    payload = {
        "xyz": x.tolist(),
        "scalars": {
            "dynamic_probability": p.tolist(),
            "confidence": c.tolist(),
            "endpoint_displacement": d.tolist(),
            "static_dynamic_label": l.tolist(),
        },
        "first": "dynamic_probability",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<!doctype html><meta charset="utf-8"><title>Closed-loop static/dynamic prior</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<select id="scalar"></select><div id="plot" style="width:100vw;height:94vh"></div>
<script>
const payload = """
        + json.dumps(payload)
        + """;
const sel = document.getElementById('scalar');
for (const key of Object.keys(payload.scalars)) { const o=document.createElement('option'); o.value=key; o.textContent=key; sel.appendChild(o); }
function draw(key) {
  const xyz = payload.xyz;
  Plotly.react('plot', [{type:'scatter3d', mode:'markers',
    x:xyz.map(p=>p[0]), y:xyz.map(p=>p[1]), z:xyz.map(p=>p[2]),
    marker:{size:2, color:payload.scalars[key], colorscale:'Viridis', cmin:0, cmax:key === 'endpoint_displacement' ? null : 1, showscale:true}}],
    {title:key, scene:{aspectmode:'data'}, margin:{l:0,r:0,t:35,b:0}});
}
sel.onchange = () => draw(sel.value); sel.value = payload.first; draw(sel.value);
</script>""",
        encoding="utf-8",
    )


def _counts(label: torch.Tensor) -> dict[str, int]:
    l = torch.as_tensor(label).float()
    return {
        "static": int((l == 0.0).sum()),
        "uncertain": int((l == 0.5).sum()),
        "dynamic": int((l == 1.0).sum()),
    }


def _camera_ids(camera) -> tuple[str | None, int | None]:
    cam_id = getattr(camera, "dynamic_camera_id", None)
    frame_id = getattr(camera, "frame_id", None)
    if frame_id is not None:
        frame_id = int(frame_id)
    for text in (getattr(camera, "dynamic_image_path", ""), getattr(camera, "image_name", "")):
        text = str(text)
        if cam_id is None:
            m = re.search(r"(cam_\d+)", text)
            if m:
                cam_id = m.group(1)
        if frame_id is None:
            m = re.search(r"frame_(\d+)", text)
            if m:
                frame_id = int(m.group(1))
    return (str(cam_id) if cam_id is not None else None), frame_id


def _read_dataset_object_mask(root_scene: str | Path | None, camera, shape: tuple[int, int]) -> tuple[torch.Tensor, str] | None:
    if root_scene is None or not str(root_scene).strip():
        return None
    cam_id, frame_id = _camera_ids(camera)
    if cam_id is None or frame_id is None:
        return None
    path = Path(root_scene).expanduser() / "mask" / cam_id / f"frame_{frame_id:06d}.png"
    if not path.is_file():
        return None
    arr = np.asarray(Image.open(path).convert("L"))
    h, w = shape
    if tuple(arr.shape[:2]) != (h, w):
        nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
        arr = np.asarray(Image.fromarray(arr).resize((w, h), nearest))
    return torch.from_numpy(arr > 0), f"dataset_mask:{path.relative_to(Path(root_scene).expanduser())}"


def _camera_mask(camera, root_scene: str | Path | None = None, *, return_source: bool = False):
    image = camera.original_image.detach()
    h, w = int(image.shape[-2]), int(image.shape[-1])
    dataset = _read_dataset_object_mask(root_scene, camera, (h, w))
    if dataset is not None:
        mask, source = dataset
        return (mask.bool(), source) if return_source else mask.bool()
    for name in ("alpha_mask", "gt_alpha_mask", "depth_mask"):
        value = getattr(camera, name, None)
        if value is None:
            continue
        mask = torch.as_tensor(value).detach().cpu()
        if mask.ndim == 3:
            mask = mask[0]
        if tuple(mask.shape[-2:]) == (h, w):
            mask = mask.bool()
            source = name
            if float(mask.float().mean()) > 0.95:
                source = f"{name}:all_or_nearly_all"
            return (mask, source) if return_source else mask
    mask = torch.ones((h, w), dtype=torch.bool)
    source = "fallback_all_image"
    return (mask, source) if return_source else mask


def _camera_depth(camera) -> torch.Tensor | None:
    value = getattr(camera, "invdepthmap", None)
    if value is None:
        return None
    depth = torch.as_tensor(value).detach().cpu().float()
    if depth.ndim == 3:
        depth = depth[0]
    return depth


def _frame_time(frame: int, fps: float) -> float:
    return float(frame) / max(float(fps), 1.0e-8)


def _camera_rgb(camera) -> torch.Tensor:
    return camera.original_image.detach().cpu().float().clamp(0.0, 1.0)


def _build_video_rgbd_support(args, x0: torch.Tensor, yT: torch.Tensor, output: Path) -> tuple[object | None, dict]:
    if not bool(args.use_video_rgbd):
        return None, {"enabled": False, "reason": "disabled"}
    if not str(args.root_scene or "").strip():
        raise RuntimeError("--use-video-rgbd requires --root-scene")
    from tools.test_render_denoise_discovery import _ordered_complete_frame_groups, build_scene

    scene_args = SimpleNamespace(
        root_scene=args.root_scene,
        static_model_dir=args.static_model_dir,
        iteration=int(args.static_iteration),
        sh_degree=None,
        resolution=None,
        white_background=None,
        camera_data_device="cpu",
        use_depth=True,
        cameras="",
        fps=float(args.fps),
    )
    scene, _gaussians, _pipe, _white_background = build_scene(scene_args)
    groups = _ordered_complete_frame_groups(scene.getTrainCameras())
    all_groups = groups
    if len(groups) < 2:
        return None, {"enabled": True, "reason": "not_enough_frames", "complete_frames": len(groups)}
    if int(args.video_frame_stride) > 1:
        stride = max(1, int(args.video_frame_stride))
        groups = groups[::stride]
        if groups[-1][0] != all_groups[-1][0]:
            groups.append(all_groups[-1])
    source_group = groups[0][1]
    target_group = groups[-1][1]
    camera_pairs = pair_fixed_cameras(source_group, target_group)[: max(1, int(args.video_max_cameras))]
    print(
        "[closed-loop RGB-D] sparse video support: "
        f"frames={len(groups)} cameras={len(camera_pairs)} "
        f"max_pixels/camera={int(args.video_max_pixels)} "
        f"local_offsets={int(args.video_pixel_candidates)} "
        f"radius={int(args.video_pixel_search_radius)}"
    )
    supports = []
    reports = []
    mask_cache: dict[tuple[str, int], tuple[torch.Tensor, str]] = {}

    def mask_for(camera) -> tuple[torch.Tensor, str]:
        cam_id, frame_id = _camera_ids(camera)
        key = (str(cam_id), int(frame_id) if frame_id is not None else -1)
        if key not in mask_cache:
            mask_cache[key] = _camera_mask(camera, args.root_scene, return_source=True)
        return mask_cache[key]

    for source_cam, target_cam in tqdm(camera_pairs, desc="RGB-D video cameras", unit="camera"):
        cam_id = str(getattr(source_cam, "dynamic_camera_id", getattr(source_cam, "image_name", "camera")))
        camera_sequence = []
        for frame, cams in groups:
            matches = [cam for cam in cams if str(getattr(cam, "dynamic_camera_id", getattr(cam, "image_name", ""))) == cam_id]
            if matches:
                camera_sequence.append((frame, matches[0]))
        if len(camera_sequence) < 2:
            continue
        first_frame, first_cam = camera_sequence[0]
        source_mask, source_mask_source = mask_for(first_cam)
        source_pixels = torch.nonzero(source_mask.reshape(-1), as_tuple=False).flatten()
        initial_object_pixels = int(source_pixels.numel())
        if source_pixels.numel() > int(args.video_max_pixels):
            gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
            source_pixels = source_pixels[torch.randperm(source_pixels.numel(), generator=gen)[: int(args.video_max_pixels)]]
        current = source_pixels.clone()
        confidence = torch.ones_like(current, dtype=torch.float32)
        alive_source = source_pixels.clone()
        step_reports = []
        frame_iter = tqdm(
            list(zip(camera_sequence[:-1], camera_sequence[1:])),
            desc=f"RGB-D tracks {cam_id}",
            unit="pair",
            leave=False,
        )
        for (frame_a, cam_a), (frame_b, cam_b) in frame_iter:
            if current.numel() == 0:
                break
            frame_iter.set_postfix(tracks=int(current.numel()))
            dt = _frame_time(frame_b, float(args.fps)) - _frame_time(frame_a, float(args.fps))
            mask_a, mask_a_source = mask_for(cam_a)
            mask_b, mask_b_source = mask_for(cam_b)
            depth_a = _camera_depth(cam_a)
            depth_b = _camera_depth(cam_b)
            field = match_foreground_pixels_to_next(
                _camera_rgb(cam_a),
                _camera_rgb(cam_b),
                depth_a if depth_a is not None else torch.ones_like(mask_a, dtype=torch.float32),
                depth_b if depth_b is not None else torch.ones_like(mask_b, dtype=torch.float32),
                mask_a,
                mask_b,
                source_pixel_indices=current,
                delta_t=max(dt, 1.0e-8),
                frame_pair=(int(frame_a), int(frame_b)),
                camera_id=cam_id,
                max_source_pixels=int(args.video_max_pixels),
                candidates=int(args.video_pixel_candidates),
                search_radius=int(args.video_pixel_search_radius),
            )
            if field.source_pixel_indices.numel() == 0:
                current = field.target_pixel_indices
                confidence = torch.empty(0)
                alive_source = torch.empty(0, dtype=torch.long)
                break
            # Preserve only tracks that survived foreground masking in this step.
            lookup = {int(p): i for i, p in enumerate(current.tolist())}
            parent = torch.tensor([lookup[int(p)] for p in field.source_pixel_indices.tolist()], dtype=torch.long)
            alive_source = alive_source.index_select(0, parent)
            confidence = confidence.index_select(0, parent) * field.confidence.detach().cpu().float()
            current = field.target_pixel_indices.detach().cpu().long()
            step_reports.append({
                "pair": [int(frame_a), int(frame_b)],
                "tracks": int(current.numel()),
                "median_confidence": float(confidence.median()) if confidence.numel() else 0.0,
                "source_mask_pixels": int(mask_a.sum()),
                "target_mask_pixels": int(mask_b.sum()),
                "source_mask": mask_a_source,
                "target_mask": mask_b_source,
                "matched_median_pixel_displacement": float(field.stats.get("median_pixel_displacement", 0.0)),
                "matched_p90_pixel_displacement": float(field.stats.get("p90_pixel_displacement", 0.0)),
            })
        if current.numel() == 0:
            continue
        target_mask, target_mask_source = mask_for(camera_sequence[-1][1])
        print(f"[closed-loop RGB-D] attaching {int(current.numel())} tracks for {cam_id} to endpoint Gaussians")
        support = build_projected_video_endpoint_support(
            x0,
            yT,
            first_cam,
            camera_sequence[-1][1],
            alive_source,
            current,
            confidence,
            source_depth=_camera_depth(first_cam),
            target_depth=_camera_depth(camera_sequence[-1][1]),
            topk_gaussians=int(args.video_topk_gaussians),
            max_pixel_radius=float(args.video_max_pixel_radius),
            depth_sigma=float(args.video_depth_sigma),
        )
        if support.source_ids.numel():
            supports.append(support)
        reports.append({
            "camera": cam_id,
            "initial_object_pixels": initial_object_pixels,
            "initial_object_fraction": float(source_mask.float().mean()),
            "target_object_pixels": int(target_mask.sum()),
            "target_object_fraction": float(target_mask.float().mean()),
            "source_mask": source_mask_source,
            "target_mask": target_mask_source,
            "initial_tracks": int(source_pixels.numel()),
            "surviving_tracks": int(current.numel()),
            "support_pairs": int(support.source_ids.numel()),
            "median_track_confidence": float(confidence.median()) if confidence.numel() else 0.0,
            "steps": step_reports,
        })
        tqdm.write(
            f"[closed-loop RGB-D] {cam_id}: tracks={int(current.numel())} "
            f"support_pairs={int(support.source_ids.numel())} "
            f"median_conf={float(confidence.median()) if confidence.numel() else 0.0:.4g}"
        )
    if not supports:
        return None, {"enabled": True, "reason": "no_sparse_support", "reports": reports}
    source_ids = torch.cat([s.source_ids for s in supports])
    target_ids = torch.cat([s.target_ids for s in supports])
    values = torch.cat([s.support for s in supports])
    from utils.video_endpoint_transport import SparseVideoEndpointSupport
    support = SparseVideoEndpointSupport(source_ids, target_ids, values, "forward", {"cameras": len(supports)}).coalesce()
    torch.save({"support": support, "reports": reports}, output / "video_rgbd_endpoint_support.pt")
    return support, {
        "enabled": True,
        "support_pairs": int(support.source_ids.numel()),
        "sources_with_support": int(torch.unique(support.source_ids).numel()),
        "target_with_support": int(torch.unique(support.target_ids).numel()),
        "reports": reports,
    }


def _video_likelihoods_from_support(
    support,
    x0: torch.Tensor,
    yT: torch.Tensor,
    fm_endpoint: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(x0.shape[0])
    static_l = torch.ones(n, dtype=torch.float32)
    dynamic_l = torch.ones(n, dtype=torch.float32)
    support_mass = torch.zeros(n, dtype=torch.float32)
    dynamic_energy = torch.zeros(n, dtype=torch.float32)
    if support is None or int(support.source_ids.numel()) == 0:
        return static_l, dynamic_l, support_mass, dynamic_energy
    sid = support.source_ids.detach().cpu().long()
    tid = support.target_ids.detach().cpu().long()
    w = support.support.detach().cpu().float().clamp_min(0.0)
    valid = (sid >= 0) & (sid < n) & (tid >= 0) & (tid < int(yT.shape[0])) & torch.isfinite(w) & (w > 0)
    sid, tid, w = sid[valid], tid[valid], w[valid]
    if sid.numel() == 0:
        return static_l, dynamic_l, support_mass, dynamic_energy
    expected = torch.zeros((n, 3), dtype=torch.float32)
    support_mass.scatter_add_(0, sid, w)
    expected.index_add_(0, sid, yT.index_select(0, tid).float() * w[:, None])
    has = support_mass > 0
    expected[has] = expected[has] / support_mass[has, None].clamp_min(1.0e-8)
    s = scale.detach().cpu().float().reshape(-1).clamp_min(1.0e-6)
    static_energy = torch.linalg.norm(x0.float() - expected, dim=-1).square() / s.square()
    dynamic_energy = torch.linalg.norm(fm_endpoint.detach().cpu().float() - expected, dim=-1).square() / s.square()
    static_l[has] = torch.exp(-0.5 * static_energy[has]).clamp_min(1.0e-8)
    dynamic_l[has] = torch.exp(-0.5 * dynamic_energy[has]).clamp_min(1.0e-8)
    return static_l, dynamic_l, support_mass, dynamic_energy


def _merge_supports(items: list[object]):
    items = [item for item in items if item is not None and int(item.source_ids.numel()) > 0]
    if not items:
        return None
    from utils.video_endpoint_transport import SparseVideoEndpointSupport
    return SparseVideoEndpointSupport(
        torch.cat([item.source_ids.detach().cpu().long() for item in items]),
        torch.cat([item.target_ids.detach().cpu().long() for item in items]),
        torch.cat([item.support.detach().cpu().float() for item in items]),
        "forward",
        {"sources": len(items)},
    ).coalesce()


def _robust_mismatch_probability(energy: torch.Tensor) -> torch.Tensor:
    value = torch.log1p(torch.as_tensor(energy).detach().cpu().float().clamp_min(0.0))
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return torch.full_like(value, 0.5)
    q25, q50, q75 = torch.quantile(finite, torch.tensor([0.25, 0.5, 0.75]))
    scale = (q75 - q25).clamp_min(1.0e-6)
    return torch.sigmoid((value - q50) / scale).clamp(1.0e-6, 1.0 - 1.0e-6)


@torch.no_grad()
def _nearest_squared_distance_and_index(query: torch.Tensor, reference: torch.Tensor, *, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.as_tensor(query).detach().cpu().float()
    r = torch.as_tensor(reference).detach().cpu().float()
    values: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for start in range(0, int(q.shape[0]), max(1, int(chunk_size))):
        d2 = torch.cdist(q[start:start + int(chunk_size)], r).square()
        best = d2.min(dim=1)
        values.append(best.values)
        indices.append(best.indices)
    return torch.cat(values, 0).clamp_min(0.0), torch.cat(indices, 0)


def run(args) -> dict:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[closed-loop endpoint FM] CUDA requested but unavailable; falling back to CPU")
        device = "cpu"
    static_ply = _resolve_ply(args.static_model_dir, int(args.static_iteration))
    endpoint_ply = _resolve_ply(args.endpoint_static_model_dir, int(args.endpoint_static_iteration))
    x0 = _xyz(static_ply)
    yT = _xyz(endpoint_ply)
    sx = local_spacing(x0, k=8, chunk_size=int(args.chunk_size))
    sy = local_spacing(yT, k=8, chunk_size=int(args.chunk_size))
    mismatch = compute_bidirectional_spatial_mismatch(x0, yT, sx, sy, chunk_size=int(args.chunk_size))
    q0 = _robust_mismatch_probability(mismatch.forward)
    qT = _robust_mismatch_probability(mismatch.backward)
    cfg = FlowTrainConfig(iters=int(args.fm_iters), batch_size=int(args.fm_batch_size), knn=int(args.fm_knn), chunk_size=int(args.chunk_size), seed=int(args.seed))
    fwd = train_forward_endpoint_fm(x0, yT, cfg, device=device)
    bwd = train_backward_endpoint_fm(x0, yT, FlowTrainConfig(**{**cfg.__dict__, "seed": int(args.seed) + 1}), device=device)
    cycle = compute_endpoint_cycle_consistency(x0, yT, fwd, bwd, cfg, cfg, device=device)
    n = min(int(x0.shape[0]), int(fwd.displacement.shape[0]))
    endpoint_d2, endpoint_idx = _nearest_squared_distance_and_index(fwd.endpoint[:n], yT, chunk_size=int(args.chunk_size))
    endpoint_scale = sy.index_select(0, endpoint_idx).clamp_min(1.0e-6)
    endpoint_energy = endpoint_d2 / endpoint_scale.square()
    static_l = torch.exp(-0.5 * mismatch.forward[:n].detach().cpu().float()).clamp_min(1.0e-8)
    endpoint_l = torch.exp(-0.5 * endpoint_energy.detach().cpu().float()).clamp_min(1.0e-8)
    support_items = []
    legacy_video_summary = {"enabled": False}
    if args.track_ownership:
        payload = torch.load(args.track_ownership, map_location="cpu")
        support = build_sparse_video_endpoint_gaussian_support(
            payload["source_ownership"].float(),
            payload["target_ownership"].float(),
            payload["track_source_pixels"].long(),
            payload["track_target_pixels"].long(),
            torch.ones_like(payload["track_source_pixels"], dtype=torch.float32),
            topk_gaussians=int(args.video_topk_gaussians),
        )
        support_items.append(support)
        ranked, score = rank_target_gaussians_from_video_support(support, num_sources=int(x0.shape[0]), k=int(args.video_candidates))
        torch.save({"support": support, "ranked_targets": ranked, "scores": score}, output / "video_guided_endpoint_candidates.pt")
        legacy_video_summary = {
            "enabled": True,
            "source": "track_ownership",
            "support_pairs": int(support.source_ids.numel()),
            "sources_with_video_candidates": int((ranked[:, 0] >= 0).sum()),
            "top_score_median": float(score[:, 0].median()) if score.numel() else 0.0,
        }
    rgbd_support, rgbd_video_summary = _build_video_rgbd_support(args, x0[:n], yT, output)
    support_items.append(rgbd_support)
    video_support = _merge_supports(support_items)
    video_static_l, video_dynamic_l, video_support_mass, video_dynamic_energy = _video_likelihoods_from_support(
        video_support,
        x0[:n],
        yT,
        fwd.endpoint[:n],
        sx[:n],
    )
    if video_support is not None:
        weight = float(args.video_likelihood_weight)
        static_l = (static_l * video_static_l.pow(weight)).clamp_min(1.0e-8)
        endpoint_l = (endpoint_l * video_dynamic_l.pow(weight)).clamp_min(1.0e-8)
        torch.save(
            {
                "support": video_support,
                "video_static_likelihood": video_static_l,
                "video_dynamic_likelihood": video_dynamic_l,
                "video_support_mass": video_support_mass,
                "video_dynamic_energy": video_dynamic_energy,
            },
            output / "closed_loop_video_rgbd_fusion.pt",
        )
    video_summary = {
        "enabled": video_support is not None,
        "legacy_track_ownership": legacy_video_summary,
        "rgbd_tracks": rgbd_video_summary,
        "support_pairs": 0 if video_support is None else int(video_support.source_ids.numel()),
        "sources_with_support": 0 if video_support is None else int(torch.unique(video_support.source_ids).numel()),
        "likelihood_weight": float(args.video_likelihood_weight),
    }
    post = update_dynamic_posterior(q0[:n], static_l, endpoint_l)
    if not bool(torch.isfinite(post).all()):
        raise RuntimeError("Closed-loop posterior contains non-finite values after Bayesian update")
    cycle_err = cycle["forward_cycle_error"][:n].detach().cpu().float()
    spacing = sx[:n].detach().cpu().float().clamp_min(1.0e-6)
    cycle_conf = torch.exp(-cycle_err / (3.0 * spacing)).clamp(0.0, 1.0)
    confidence = (2.0 * (post.detach().cpu().float() - 0.5).abs()).clamp(0.0, 1.0) * cycle_conf * endpoint_l.sqrt().clamp(0.0, 1.0)
    valid = torch.isfinite(post.detach().cpu()) & torch.isfinite(confidence) & torch.isfinite(cycle_err)
    label = _label_from_probability(post, float(args.static_threshold), float(args.dynamic_threshold))
    gaussian_ids = torch.arange(n, dtype=torch.long)
    prior = DynamicPrior(
        probability=post.detach().cpu(),
        raw_score=_logit(post).detach().cpu(),
        valid=valid.detach().cpu(),
        confidence=confidence.detach().cpu(),
        method="strict_closed_loop_rgbd_endpoint_fm" if video_support is not None else "strict_closed_loop_endpoint_fm",
        gaussian_ids=gaussian_ids,
        metadata={
            "static_model_dir": str(Path(args.static_model_dir).expanduser()),
            "endpoint_static_model_dir": str(Path(args.endpoint_static_model_dir).expanduser()),
            "static_iteration": int(args.static_iteration),
            "endpoint_static_iteration": int(args.endpoint_static_iteration),
            "fm_iters": int(args.fm_iters),
            "fm_knn": int(args.fm_knn),
            "observation_sigma": float(args.observation_sigma),
            "device": device,
            "video_endpoint_support": video_summary,
            "static_threshold_for_debug_label": float(args.static_threshold),
            "dynamic_threshold_for_debug_label": float(args.dynamic_threshold),
            "ground_truth_used_as_algorithm_input": False,
        },
    ).validate()
    prior_path = save_dynamic_prior(prior, output / "dynamic_prior.pt")
    summary = {
        "source_points": int(x0.shape[0]),
        "target_points": int(yT.shape[0]),
        "p0_dynamic_mean": float(q0.mean()),
        "pt_dynamic_mean": float(qT.mean()),
        "posterior_dynamic_mean": float(post.mean()),
        "static_likelihood": tensor_summary(static_l),
        "endpoint_dynamic_likelihood": tensor_summary(endpoint_l),
        "video_static_likelihood": tensor_summary(video_static_l),
        "video_dynamic_likelihood": tensor_summary(video_dynamic_l),
        "video_support_mass": tensor_summary(video_support_mass),
        "endpoint_match_energy": tensor_summary(endpoint_energy),
        "posterior_dynamic_probability": tensor_summary(post),
        "dynamic_prior_confidence": tensor_summary(confidence),
        "debug_label_counts": _counts(label),
        "dynamic_prior_path": str(prior_path),
        "forward_cycle_error_median": float(cycle["forward_cycle_error"].median()),
        "backward_cycle_error_median": float(cycle["backward_cycle_error"].median()),
        "video_endpoint_support": video_summary,
        "ground_truth_used_as_algorithm_input": False,
    }
    static_vertex = _read_vertex(static_ply)
    gt_dynamic = _maybe_gt_dynamic(static_vertex)
    if gt_dynamic is not None and int(gt_dynamic.numel()) >= n:
        gt_debug = {
            "argmax_0p5": binary_metrics(post.detach().cpu(), gt_dynamic[:n], threshold=0.5),
            "trusted_dynamic_threshold": binary_metrics(post.detach().cpu(), gt_dynamic[:n], threshold=float(args.dynamic_threshold)),
            "ground_truth_used_as_algorithm_input": False,
        }
        summary["gt_debug_only"] = gt_debug
        prior.metadata["gt_debug_only"] = gt_debug
        save_dynamic_prior(prior, prior_path)

    displacement = fwd.displacement[:n].detach().cpu().float().norm(dim=-1)
    full_probability = torch.full((int(x0.shape[0]),), 0.5, dtype=torch.float32)
    full_confidence = torch.zeros_like(full_probability)
    full_label = torch.full_like(full_probability, 0.5)
    full_p0 = torch.full_like(full_probability, 0.5)
    full_cycle = torch.full_like(full_probability, -1.0)
    full_disp = torch.zeros_like(full_probability)
    full_endpoint_energy = torch.zeros_like(full_probability)
    full_video_support_mass = torch.zeros_like(full_probability)
    full_video_dynamic_energy = torch.zeros_like(full_probability)
    full_video_static_likelihood = torch.ones_like(full_probability)
    full_video_dynamic_likelihood = torch.ones_like(full_probability)
    full_probability[:n] = post.detach().cpu().float()
    full_confidence[:n] = confidence.detach().cpu().float()
    full_label[:n] = label.detach().cpu().float()
    full_p0[:n] = q0[:n].detach().cpu().float()
    full_cycle[:n] = cycle_err
    full_disp[:n] = displacement
    full_endpoint_energy[:n] = endpoint_energy.detach().cpu().float()
    full_video_support_mass[:n] = video_support_mass.detach().cpu().float()
    full_video_dynamic_energy[:n] = video_dynamic_energy.detach().cpu().float()
    full_video_static_likelihood[:n] = video_static_l.detach().cpu().float()
    full_video_dynamic_likelihood[:n] = video_dynamic_l.detach().cpu().float()
    debug_ply = output / "closed_loop_static_dynamic_point_cloud.ply"
    _write_debug_ply(
        static_ply,
        debug_ply,
        {
            "closed_loop_dynamic_probability": full_probability,
            "closed_loop_dynamic_confidence": full_confidence,
            "closed_loop_static_dynamic_label": full_label,
            "closed_loop_p0_dynamic_probability": full_p0,
            "closed_loop_endpoint_displacement": full_disp,
            "closed_loop_cycle_error": full_cycle,
            "closed_loop_endpoint_match_energy": full_endpoint_energy,
            "closed_loop_video_support_mass": full_video_support_mass,
            "closed_loop_video_dynamic_energy": full_video_dynamic_energy,
            "closed_loop_video_static_likelihood": full_video_static_likelihood,
            "closed_loop_video_dynamic_likelihood": full_video_dynamic_likelihood,
        },
    )
    debug_html = output / "closed_loop_static_dynamic_debug.html"
    _write_static_dynamic_html(
        debug_html,
        x0.detach().cpu().float(),
        full_probability,
        full_confidence,
        full_disp,
        full_label,
        max_points=int(args.max_html_points),
        seed=int(args.seed),
    )
    debug = {
        "posterior_dynamic_probability": tensor_summary(post),
        "dynamic_prior_confidence": tensor_summary(confidence),
        "p0_dynamic_probability": tensor_summary(q0[:n]),
        "endpoint_displacement": tensor_summary(displacement),
        "endpoint_match_energy": tensor_summary(endpoint_energy),
        "video_support_mass": tensor_summary(video_support_mass),
        "video_dynamic_energy": tensor_summary(video_dynamic_energy),
        "video_static_likelihood": tensor_summary(video_static_l),
        "video_dynamic_likelihood": tensor_summary(video_dynamic_l),
        "static_likelihood": tensor_summary(static_l),
        "endpoint_dynamic_likelihood": tensor_summary(endpoint_l),
        "video_endpoint_support": video_summary,
        "forward_cycle_error": tensor_summary(cycle_err),
        "debug_label_counts": _counts(label),
        "dynamic_prior_path": str(prior_path),
        "debug_ply": str(debug_ply),
        "debug_html": str(debug_html),
        "ground_truth_used_as_algorithm_input": False,
    }
    if "gt_debug_only" in summary:
        debug["gt_debug_only"] = summary["gt_debug_only"]
    summary["debug_ply"] = str(debug_ply)
    summary["debug_html"] = str(debug_html)
    torch.save(
        {
            "q0": q0,
            "qT": qT,
            "posterior": post,
            "confidence": confidence,
            "valid": valid,
            "static_dynamic_label": label,
            "forward_endpoint": fwd.endpoint,
            "backward_endpoint": bwd.endpoint,
            "forward_cycle_error": cycle["forward_cycle_error"],
            "backward_cycle_error": cycle["backward_cycle_error"],
            "endpoint_match_energy": endpoint_energy.detach().cpu(),
            "static_likelihood": static_l.detach().cpu(),
            "endpoint_dynamic_likelihood": endpoint_l.detach().cpu(),
            "video_static_likelihood": video_static_l.detach().cpu(),
            "video_dynamic_likelihood": video_dynamic_l.detach().cpu(),
            "video_support_mass": video_support_mass.detach().cpu(),
            "video_dynamic_energy": video_dynamic_energy.detach().cpu(),
        },
        output / "closed_loop_endpoint_fm_probe.pt",
    )
    torch.save(post.detach().cpu(), output / "posterior_dynamic_probability.pt")
    torch.save(confidence.detach().cpu(), output / "dynamic_prior_confidence.pt")
    torch.save(label.detach().cpu(), output / "static_dynamic_label.pt")
    write_json(output / "closed_loop_static_dynamic_debug.json", debug)
    write_json(output / "closed_loop_endpoint_fm_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root-scene", default="", help="Dataset root. Required only when --use-video-rgbd is enabled.")
    p.add_argument("--static-model-dir", required=True)
    p.add_argument("--endpoint-static-model-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--static-iteration", type=int, default=14000)
    p.add_argument("--endpoint-static-iteration", type=int, default=14000)
    p.add_argument("--fm-iters", type=int, default=100)
    p.add_argument("--fm-batch-size", type=int, default=2048)
    p.add_argument("--fm-knn", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--observation-sigma", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--device", default="cuda")
    p.add_argument("--track-ownership", default="", help="Optional torch payload with source/target ownership and video track endpoint pixels.")
    p.add_argument("--use-video-rgbd", action="store_true", help="Propagate sparse foreground RGB-D tracks through the full video and fuse them with endpoint FM evidence.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--video-max-cameras", type=int, default=8)
    p.add_argument("--video-max-pixels", type=int, default=4096)
    p.add_argument("--video-frame-stride", type=int, default=1, help="Track every Nth complete frame, always keeping the last frame.")
    p.add_argument("--video-pixel-candidates", type=int, default=169)
    p.add_argument("--video-pixel-search-radius", type=int, default=8)
    p.add_argument("--video-topk-gaussians", type=int, default=8)
    p.add_argument("--video-candidates", type=int, default=16)
    p.add_argument("--video-max-pixel-radius", type=float, default=6.0)
    p.add_argument("--video-depth-sigma", type=float, default=0.03)
    p.add_argument("--video-likelihood-weight", type=float, default=1.0)
    p.add_argument("--static-threshold", type=float, default=0.35)
    p.add_argument("--dynamic-threshold", type=float, default=0.65)
    p.add_argument("--max-html-points", type=int, default=12000)
    run(p.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
