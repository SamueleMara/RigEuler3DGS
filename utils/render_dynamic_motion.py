"""Render the final hard dynamic model or its exported Gaussian tracks."""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional

import torch
import torchvision
from tqdm import tqdm

from gaussian_renderer import render
from utils.dynamic_loss_utils import get_camera_timestamp


def _camera_id(camera):
    value = getattr(camera, "dynamic_camera_id", None)
    if value is not None:
        return str(value)
    match = re.search(r"cam_\d+", str(getattr(camera, "image_name", "")))
    return match.group(0) if match is not None else f"cam_{int(camera.uid):03d}"


def _frame_id(camera):
    if hasattr(camera, "frame_id"):
        return int(camera.frame_id)
    match = re.search(r"frame[_-]?(\d+)", str(getattr(camera, "image_name", "")))
    return int(match.group(1)) if match is not None else 0


def _selected_views(scene, split, cameras, frame_start, frame_end, frame_stride):
    if split == "train":
        views = list(scene.getTrainCameras())
    elif split == "test":
        views = list(scene.getTestCameras())
    else:
        views = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    allowed = None if not cameras or cameras == "all" else set(cameras.split(","))
    selected, seen = [], set()
    for camera in views:
        camera_id, frame_id = _camera_id(camera), _frame_id(camera)
        if allowed is not None and camera_id not in allowed:
            continue
        if frame_id < frame_start or (frame_end >= 0 and frame_id > frame_end):
            continue
        if (frame_id - frame_start) % max(1, frame_stride) != 0:
            continue
        key = (camera_id, frame_id)
        if key not in seen:
            selected.append(camera)
            seen.add(key)
    return sorted(selected, key=lambda camera: (_camera_id(camera), _frame_id(camera)))


def _load_tracks(path: Optional[str]):
    if not path:
        return None
    tracks = torch.load(path, map_location="cpu", weights_only=False)
    required = ("times", "positions", "rotations", "velocities")
    if not isinstance(tracks, dict) or any(key not in tracks for key in required):
        raise ValueError(f"{path} is not a unified Gaussian-track checkpoint")
    return tracks


def _inject_nearest_track(gaussians, tracks, time):
    index = int(torch.argmin(torch.abs(tracks["times"].float() - float(time))))
    gaussians._current_xyz = tracks["positions"][index].cuda().float()
    gaussians._current_rotation = tracks["rotations"][index].cuda().float()
    gaussians._current_velocity = tracks["velocities"][index].cuda().float()
    gaussians._current_time = float(tracks["times"][index])


def _write_video(frame_dir: Path, output: Path, fps: float):
    try:
        import imageio.v2 as imageio
        frames = [imageio.imread(path) for path in sorted(frame_dir.glob("frame_*.png"))]
        if frames:
            imageio.mimsave(output, frames, fps=fps, quality=8)
    except Exception as error:
        print(f"[render] MP4 export skipped: {error}")


@torch.no_grad()
def render_dynamic_sequence(
    scene,
    gaussians,
    pipeline,
    background,
    args,
    iteration,
    tracks_path: Optional[str] = None,
    output_root: Optional[str] = None,
    split: str = "all",
    cameras: str = "",
    frame_start: int = 0,
    frame_end: int = -1,
    frame_stride: int = 1,
    use_tracks: bool = False,
    save_video: bool = False,
    separate_sh: bool = False,
) -> Dict:
    """Render selected dynamic views.

    The default evaluates the in-memory exact joint model at each timestamp.
    ``use_tracks`` is a deterministic debug replay of exported states, not an
    alternative trainable motion representation.
    """
    tracks = _load_tracks(tracks_path) if use_tracks else None
    views = _selected_views(scene, split, cameras, frame_start, frame_end, frame_stride)
    output = Path(output_root or (Path(scene.model_path) / "dynamic_renders")) / f"ours_{iteration}"
    output.mkdir(parents=True, exist_ok=True)
    counts, rows = {}, []
    for camera in tqdm(views, desc="Rendering exact dynamic model"):
        camera_id, frame_id = _camera_id(camera), _frame_id(camera)
        time = get_camera_timestamp(camera, args.fps, args.time_scale)
        if tracks is None:
            gaussians.set_time(time, detach_canonical=True)
        else:
            _inject_nearest_track(gaussians, tracks, time)
        package = render(
            camera, gaussians, pipeline, background,
            use_trained_exp=scene._scene_args.train_test_exp,
            separate_sh=separate_sh,
        )
        render_dir = output / camera_id / "renders"
        target_dir = output / camera_id / "gt"
        render_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        render_path = render_dir / f"frame_{frame_id:06d}.png"
        torchvision.utils.save_image(package["render"].clamp(0, 1), render_path)
        torchvision.utils.save_image(camera.original_image[:3].clamp(0, 1), target_dir / render_path.name)
        gaussians.clear_time()
        counts[camera_id] = counts.get(camera_id, 0) + 1
        rows.append({"camera": camera_id, "frame": frame_id, "time": time, "render": str(render_path)})
    if save_video:
        for camera_id in counts:
            _write_video(
                output / camera_id / "renders",
                output / camera_id / f"{camera_id}.mp4",
                args.fps,
            )
    summary = {
        "schema_version": 1,
        "iteration": int(iteration),
        "hard_kinematic_render": tracks is None,
        "views_per_camera": counts,
        "views": rows,
    }
    (output / "render_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    from arguments import (
        ModelParams, PipelineParams, add_dynamic_args,
        apply_dynamic_defaults, get_combined_args,
    )
    from scene import GaussianModel, Scene

    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    add_dynamic_args(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--render_cameras", type=str, default="")
    parser.add_argument("--render_frame_start", type=int, default=0)
    parser.add_argument("--render_frame_end", type=int, default=-1)
    parser.add_argument("--render_frame_stride", type=int, default=1)
    parser.add_argument("--tracks", type=str, default="")
    parser.add_argument("--use_tracks", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    args = apply_dynamic_defaults(get_combined_args(parser))
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, dynamic_args=args)
    iteration = scene.loaded_iter if scene.loaded_iter is not None else args.iteration
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        device="cuda",
    )
    render_dynamic_sequence(
        scene, gaussians, pipeline.extract(args), background, args, iteration,
        tracks_path=args.tracks,
        split=args.split,
        cameras=args.render_cameras,
        frame_start=args.render_frame_start,
        frame_end=args.render_frame_end,
        frame_stride=args.render_frame_stride,
        use_tracks=args.use_tracks,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
