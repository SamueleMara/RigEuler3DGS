"""Utilities to render Dynamic Eulerian/RigEuler 3DGS trajectories.

This module is intentionally importable from train.py and also runnable as:

    python -m utils.render_dynamic_motion ...

The core function is render_dynamic_sequence(...). It accepts an already-loaded
Scene/GaussianModel, so post-training rendering does not need to reload the
whole checkpoint. Rendering can be driven either by gaussian_tracks.pt or by the
in-memory Eulerian motion field via gaussians.set_time(...).
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchvision
from tqdm import tqdm

from gaussian_renderer import render


def _split_csv(value: str) -> Optional[set]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() == "all":
        return None
    return {v.strip() for v in value.split(",") if v.strip()}


def camera_id(cam) -> str:
    """Best-effort dynamic camera id, compatible with patched Camera clones."""
    for attr in ("dynamic_camera_id", "camera_id", "cam_id"):
        val = getattr(cam, attr, None)
        if val is not None:
            return str(val)
    name = str(getattr(cam, "image_name", ""))
    m = re.search(r"cam_\d+", name)
    if m:
        return m.group(0)
    return f"cam_{int(getattr(cam, 'uid', 0)):03d}"


def frame_id(cam, fps: float = 30.0) -> int:
    """Best-effort frame id from Camera attributes or image name."""
    if hasattr(cam, "frame_id"):
        return int(getattr(cam, "frame_id"))
    name = str(getattr(cam, "image_name", ""))
    m = re.search(r"frame[_-]?(\d+)", name)
    if m:
        return int(m.group(1))
    return int(round(float(getattr(cam, "timestamp", 0.0)) * float(fps)))


def timestamp(cam, fps: float = 30.0, time_scale: float = 1.0) -> float:
    if hasattr(cam, "timestamp"):
        return float(getattr(cam, "timestamp")) * float(time_scale)
    return float(frame_id(cam, fps=fps)) / max(float(fps), 1e-8) * float(time_scale)


def load_tracks(tracks_path: str) -> Dict[str, torch.Tensor]:
    path = Path(tracks_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Could not find gaussian track file: {path}")
    tracks = torch.load(str(path), map_location="cpu")
    if not isinstance(tracks, dict):
        raise ValueError(f"{path} must contain a dict saved with torch.save(...).")
    if "positions" not in tracks:
        raise ValueError(f"{path} must contain tracks['positions'] with shape [T, N, 3].")
    positions = tracks["positions"]
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"tracks['positions'] must have shape [T, N, 3], got {tuple(positions.shape)}")
    return tracks


def build_track_index(tracks: Dict[str, torch.Tensor]) -> Tuple[Dict[int, int], torch.Tensor]:
    positions = tracks["positions"]
    if "frame_ids" in tracks and tracks["frame_ids"] is not None:
        frame_ids = tracks["frame_ids"].long().cpu()
    else:
        frame_ids = torch.arange(positions.shape[0], dtype=torch.long)
    return {int(fid): idx for idx, fid in enumerate(frame_ids.tolist())}, frame_ids


def nearest_track_index(fid: int, frame_ids: torch.Tensor) -> int:
    diffs = torch.abs(frame_ids.long() - int(fid))
    return int(torch.argmin(diffs).item())


def set_gaussians_from_tracks(gaussians, tracks: Dict[str, torch.Tensor], track_idx: int, device: str = "cuda") -> None:
    """Inject per-frame xyz/velocity into GaussianModel without changing canonical xyz."""
    xyz = tracks["positions"][track_idx].to(device=device, dtype=torch.float32, non_blocking=True)
    if "velocities" in tracks and tracks["velocities"] is not None:
        vel = tracks["velocities"][track_idx].to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        vel = torch.zeros_like(xyz)

    # The patched GaussianModel.get_xyz reads _current_xyz when set_time/track
    # injection is active. We keep rotations/scales/features canonical for the
    # first version, exactly as requested: simple positions + velocities.
    gaussians._current_xyz = xyz
    gaussians._current_velocity = vel
    if hasattr(gaussians, "_rotation"):
        gaussians._current_rotation = gaussians._rotation.detach()
    if "times" in tracks and tracks["times"] is not None:
        gaussians._current_time = float(tracks["times"][track_idx])


def clear_dynamic_state(gaussians) -> None:
    if hasattr(gaussians, "clear_time"):
        gaussians.clear_time()
        return
    for attr in ("_current_xyz", "_current_velocity", "_current_rotation", "_current_time"):
        if hasattr(gaussians, attr):
            setattr(gaussians, attr, None)


def collect_dynamic_views(scene, split: str = "all", cameras: str = "", frame_start: int = 0,
                          frame_end: int = -1, frame_stride: int = 1, fps: float = 30.0) -> List:
    """Collect and sort dynamic Camera clones by camera id and frame id."""
    if split == "train":
        views = list(scene.getTrainCameras())
    elif split == "test":
        views = list(scene.getTestCameras())
    else:
        views = list(scene.getTrainCameras()) + list(scene.getTestCameras())

    allowed = _split_csv(cameras)
    out = []
    seen = set()
    stride = max(1, int(frame_stride))
    for cam in views:
        cid = camera_id(cam)
        fid = frame_id(cam, fps=fps)
        if allowed is not None and cid not in allowed:
            continue
        if fid < int(frame_start):
            continue
        if int(frame_end) >= 0 and fid > int(frame_end):
            continue
        if stride > 1 and ((fid - int(frame_start)) % stride != 0):
            continue
        key = (cid, fid)
        if key in seen:
            # Avoid duplicate train/test entries for the same dynamic view.
            continue
        seen.add(key)
        out.append(cam)
    out.sort(key=lambda c: (camera_id(c), frame_id(c, fps=fps)))
    return out


def save_video_from_pngs(render_dir: Path, video_path: Path, fps: float) -> None:
    frames = sorted(render_dir.glob("frame_*.png"))
    if not frames:
        return
    try:
        import imageio.v2 as imageio
        imgs = [imageio.imread(str(p)) for p in frames]
        imageio.mimsave(str(video_path), imgs, fps=float(fps), quality=8)
        print(f"[render_dynamic_motion] Wrote video {video_path}")
    except Exception as exc:
        print(f"[render_dynamic_motion] Could not write video {video_path}: {exc}")


def render_dynamic_sequence(
    scene,
    gaussians,
    pipeline,
    background: torch.Tensor,
    dyn,
    iteration: int,
    tracks_path: Optional[str] = None,
    out_root: Optional[Path] = None,
    cameras: str = "",
    frame_start: int = 0,
    frame_end: int = -1,
    frame_stride: int = 1,
    split: str = "all",
    use_tracks: bool = True,
    save_video: bool = False,
    train_test_exp: bool = False,
    separate_sh: bool = False,
) -> Dict:
    """Render all selected dynamic views from either gaussian_tracks.pt or field rollout.

    Args:
        scene: already-loaded Scene with dynamic cameras enabled.
        gaussians: trained GaussianModel.
        pipeline: PipelineParams object passed to gaussian_renderer.render.
        background: CUDA background color tensor.
        dyn: argparse namespace containing fps/time_scale/motion_integration_steps.
        iteration: iteration tag used in output path.
        tracks_path: gaussian_tracks.pt path. If None, uses scene.model_path/track_export_name.
        out_root: output root. If None, scene.model_path/dynamic_renders is used.
        cameras: comma-separated camera ids. Empty/all = all cameras.
        frame_start/frame_end/frame_stride: selected frame range.
        split: train/test/all.
        use_tracks: inject exported per-frame positions when True. Otherwise use gaussians.set_time.
        save_video: create one MP4 per camera.
    """
    fps = float(getattr(dyn, "fps", 30.0))
    time_scale = float(getattr(dyn, "time_scale", 1.0))
    model_path = Path(scene.model_path)
    out_root = Path(out_root) if out_root is not None else model_path / "dynamic_renders"
    out_base = out_root / f"ours_{int(iteration)}"
    out_base.mkdir(parents=True, exist_ok=True)

    tracks = None
    track_index = None
    track_frame_ids = None
    if use_tracks:
        if tracks_path is None:
            tracks_path = model_path / getattr(dyn, "track_export_name", "gaussian_tracks.pt")
        tracks = load_tracks(str(tracks_path))
        track_index, track_frame_ids = build_track_index(tracks)
        print(f"[render_dynamic_motion] Loaded tracks {tracks_path} with positions {tuple(tracks['positions'].shape)}")

    views = collect_dynamic_views(
        scene=scene,
        split=split,
        cameras=cameras,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_stride=frame_stride,
        fps=fps,
    )
    if not views:
        print("[render_dynamic_motion] No views matched the requested camera/frame filters.")
        return {"num_views": 0, "out_dir": str(out_base)}

    rendered = []
    by_camera: Dict[str, int] = {}
    for view in tqdm(views, desc="Rendering dynamic motion"):
        cid = camera_id(view)
        fid = frame_id(view, fps=fps)
        cam_dir = out_base / cid
        render_dir = cam_dir / "renders"
        gt_dir = cam_dir / "gt"
        render_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        if tracks is not None:
            tidx = track_index.get(fid, None)
            if tidx is None:
                tidx = nearest_track_index(fid, track_frame_ids)
            set_gaussians_from_tracks(gaussians, tracks, tidx, device="cuda")
        else:
            t = timestamp(view, fps=fps, time_scale=time_scale)
            gaussians.set_time(
                t,
                detach_canonical=True,
                steps=int(getattr(dyn, "motion_integration_steps", 4)),
            )

        pkg = render(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=train_test_exp,
            separate_sh=separate_sh,
        )
        image = pkg["render"]
        gt = getattr(view, "original_image", None)
        if gt is not None:
            gt = gt[0:3, :, :]

        if train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            if gt is not None:
                gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(torch.clamp(image, 0.0, 1.0), render_dir / f"frame_{fid:06d}.png")
        if gt is not None:
            torchvision.utils.save_image(torch.clamp(gt, 0.0, 1.0), gt_dir / f"frame_{fid:06d}.png")

        clear_dynamic_state(gaussians)
        by_camera[cid] = by_camera.get(cid, 0) + 1
        rendered.append({"camera": cid, "frame_id": int(fid), "image": str(render_dir / f"frame_{fid:06d}.png")})

    if save_video:
        for cid in sorted(by_camera):
            render_dir = out_base / cid / "renders"
            save_video_from_pngs(render_dir, out_base / cid / f"{cid}_renders.mp4", fps=fps)

    summary = {
        "iteration": int(iteration),
        "num_views": int(len(rendered)),
        "num_cameras": int(len(by_camera)),
        "views_per_camera": by_camera,
        "used_tracks": bool(tracks is not None),
        "tracks_path": str(tracks_path) if tracks_path is not None else "",
        "out_dir": str(out_base),
        "split": split,
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "frame_stride": int(frame_stride),
    }
    with open(out_base / "render_dynamic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[render_dynamic_motion] Rendered {len(rendered)} views to {out_base}")
    return summary


# Optional standalone entry point. This is intentionally small: for full training
# integration, train.py calls render_dynamic_sequence(...) directly.
def main() -> None:
    parser = ArgumentParser(description="Render dynamic 3DGS tracks from an already trained model.")
    parser.add_argument("--tracks_path", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--note", default="This module is mainly designed to be called from train.py after training.")
    args = parser.parse_args()
    print(args.note)
    print("For checkpoint-based standalone rendering, prefer calling render_dynamic_sequence from a small project script after loading Scene/GaussianModel.")


if __name__ == "__main__":
    main()
