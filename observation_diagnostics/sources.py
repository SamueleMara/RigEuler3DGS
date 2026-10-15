"""Read-only access to one scene: the two endpoint models, their historical
Phase-1 tracks, the calibrated cameras and the dataset's own RGB-D frames.

Every file is opened read-only. Paths are discovered and verified, not assumed:
the PLY iteration is read from the directory, the track manifests must say
`original_chronological`, the endpoint must match the direction, and the row
count of every centres file must equal the PLY's (the identity of
`original_gaussian_indices.npy` is checked, never assumed).

Motion state. The historical runs saved centres only (phase1_centers/). The
Phase-1 IEKF estimated orientations internally but did not export them, so a
source at frame t is rendered with its Phase-1 centre at t and its PLY
orientation, scales, opacity and DC colour. That is labelled
CENTER_ONLY_FIXED_ORIENTATION everywhere it is used. A full 12-D export
(`--full-state-out`, `full_state_manifest.json`) is used instead when present.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

CENTER_ONLY = "CENTER_ONLY_FIXED_ORIENTATION"
FULL_STATE = "RECORDED_FULL_12D_STATE"
DIRECTIONS = {"F": ("forward", "initial"), "B": ("backward", "final")}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_ply(model_dir: Path) -> Path:
    """The highest `iteration_N` checkpoint of a trained model."""
    its = sorted((int(p.name.split("_")[-1]), p) for p in
                 (model_dir / "point_cloud").glob("iteration_*")
                 if (p / "point_cloud.ply").is_file())
    if not its:
        raise FileNotFoundError(f"no point_cloud/iteration_*/point_cloud.ply in {model_dir}")
    return its[-1][1] / "point_cloud.ply"


def scene_paths(results_root: Path, dataset_root: Path, scene: str) -> dict:
    res, data = Path(results_root) / scene, Path(dataset_root) / scene
    out = {"scene": scene, "results_dir": res, "dataset_dir": data, "sources": {}}
    for key, (direction, endpoint) in DIRECTIONS.items():
        model = res / "static_bidi" / endpoint
        bidi = res / "bidirectional" / direction
        out["sources"][key] = {
            "direction": direction, "endpoint": endpoint, "model": model,
            "ply": find_ply(model) if (model / "point_cloud").is_dir() else None,
            "centers": bidi / "phase1_centers",
            "manifest": bidi / "phase1_centers" / "trajectory_manifest.json",
            "support": bidi / "predicted_support.json",
            "full_state": bidi / "full_state"}
    return out


def protected_files(paths: dict) -> list[Path]:
    """Everything the diagnostic reads and must leave untouched (T12)."""
    files = []
    for s in paths["sources"].values():
        for k in ("ply", "manifest", "support"):
            if s[k] is not None and Path(s[k]).is_file():
                files.append(Path(s[k]))
        c = Path(s["centers"])
        if c.is_dir():
            files += sorted(c.glob("*.npy"))
    repo = Path(__file__).resolve().parents[1]
    for rel in ("submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf.cu",
                "submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py",
                "estimator/renderer_bridge.py", "estimator/state.py",
                "tools/run_iekf_scene.py"):
        if (repo / rel).is_file():
            files.append(repo / rel)
    return files


def hash_files(files: list[Path]) -> dict:
    return {str(f): sha256_file(f) for f in files}


class SourceTrack:
    """One endpoint model plus its chronological Phase-1 track (read only)."""

    def __init__(self, key: str, spec: dict, dataset_dir: Path, *, device="cuda"):
        from tools.run_iekf_scene import load_scene, state_from_gaussians

        self.key = key
        self.direction, self.endpoint = spec["direction"], spec["endpoint"]
        self.model = Path(spec["model"])
        self.ply = Path(spec["ply"])
        self.ply_sha256 = sha256_file(self.ply)
        _d, _p, gaussians, scene = load_scene(str(self.model), str(dataset_dir))
        self.cams_all = scene.getTrainCameras()
        self.base = state_from_gaussians(gaussians)
        self.N = int(self.base.mu.shape[0])
        del gaussians, scene

        cdir = Path(spec["centers"])
        self.manifest = json.loads(Path(spec["manifest"]).read_text())
        m = self.manifest
        if m.get("trajectory_order") != "original_chronological":
            raise SystemExit(f"{cdir}: trajectory_order is {m.get('trajectory_order')!r}")
        if m.get("canonical_endpoint") != self.endpoint:
            raise SystemExit(f"{cdir}: endpoint {m.get('canonical_endpoint')!r}, "
                             f"expected {self.endpoint!r}")
        if m.get("direction") != self.direction:
            raise SystemExit(f"{cdir}: direction {m.get('direction')!r}")
        if int(m.get("num_gaussians", self.N)) != self.N:
            raise SystemExit(f"{cdir}: manifest has {m['num_gaussians']} Gaussians, "
                             f"PLY has {self.N}")
        files = sorted(cdir.glob("centers_frame_*.npy"))
        self.frames = [int(f.stem.split("_")[-1]) for f in files]
        if self.frames != list(range(len(files))):
            raise SystemExit(f"{cdir}: centre files are not frames 0..{len(files) - 1}")
        self.mu = np.stack([np.load(f).astype(np.float32) for f in files])
        if self.mu.shape[1] != self.N:
            raise SystemExit(f"{cdir}: {self.mu.shape[1]} rows, PLY has {self.N}")
        idx_path = cdir / "original_gaussian_indices.npy"
        self.original_indices = None
        self.index_note = "original_gaussian_indices.npy absent; rows are PLY rows"
        if idx_path.is_file():
            idx = np.load(idx_path)
            if idx.shape != (self.N,):
                raise SystemExit(f"{idx_path}: shape {idx.shape}, expected ({self.N},)")
            self.original_indices = idx
            self.index_note = ("identity: row r is PLY row r"
                               if np.array_equal(idx, np.arange(self.N))
                               else "NOT identity: rows map through original_gaussian_indices")
            if not np.array_equal(idx, np.arange(self.N)):
                raise SystemExit(f"{idx_path} is not the identity; this diagnostic "
                                 "refuses to reorder rows silently")
        canon = cdir / "canonical_xyz.npy"
        self.canonical_check = None
        if canon.is_file():
            cx = np.load(canon)
            ply_xyz = self.base.mu.detach().cpu().numpy()
            self.canonical_check = {
                "canonical_xyz_vs_ply_max_abs": float(np.abs(cx - ply_xyz).max()),
                "anchor_frame": int(m.get("canonical_original_frame", -1)),
                "centres_at_anchor_vs_ply_max_abs": float(np.abs(
                    self.mu[int(m.get("canonical_original_frame", 0))] - ply_xyz).max())}
        self.state_mode = CENTER_ONLY
        self.R_track = None
        fs = Path(spec["full_state"])
        if (fs / "full_state_manifest.json").is_file():
            from bidirectional_iekf.free_state import load_full_state
            d = load_full_state(str(fs), device="cpu", dtype=torch.float32)
            if d["mu"].shape[1] == self.N and list(d["original_frame_ids"]) == self.frames:
                self.state_mode = FULL_STATE
                self.mu = d["mu"].numpy()
                self.R_track = d["R"]
        self.device = device
        self.base = dataclasses.replace(
            self.base, **{f: getattr(self.base, f).to(device)
                          for f in ("mu", "R", "v", "omega", "colors", "opacity", "scales")})
        self.track_sha256 = hashlib.sha256(self.mu.tobytes()).hexdigest()

    def state(self, frame: int):
        """The source at original frame `frame` (see the module docstring)."""
        k = self.frames.index(frame)
        mu = torch.from_numpy(self.mu[k]).to(self.device)
        R = self.base.R if self.R_track is None else self.R_track[k].to(self.device)
        return dataclasses.replace(self.base, mu=mu.contiguous(), R=R.contiguous())

    def describe(self) -> dict:
        return {"key": self.key, "direction": self.direction, "endpoint": self.endpoint,
                "model": str(self.model), "ply": str(self.ply), "ply_sha256": self.ply_sha256,
                "num_gaussians": self.N, "frames": [self.frames[0], self.frames[-1]],
                "num_frames": len(self.frames), "state_mode": self.state_mode,
                "colour": "DC-only SH term (state_from_gaussians), as the Phase-1 IEKF rendered",
                "row_mapping": self.index_note, "canonical_check": self.canonical_check,
                "processing_order_first_last": [
                    self.manifest["processing_order_original_indices"][0],
                    self.manifest["processing_order_original_indices"][-1]],
                "track_sha256": self.track_sha256}


def cam_name(c, i):
    return str(getattr(c, "image_name", f"cam_{i:03d}")).split("/")[0]


def check_gauge(F: SourceTrack, B: SourceTrack) -> dict:
    """T8: one calibration for both models, or refuse."""
    out = {"F_cameras": [cam_name(c, i) for i, c in enumerate(F.cams_all)],
           "B_cameras": [cam_name(c, i) for i, c in enumerate(B.cams_all)]}
    if out["F_cameras"] != out["B_cameras"]:
        raise SystemExit(f"camera names differ: {out}")
    worst = 0.0
    for a, b in zip(F.cams_all, B.cams_all):
        if (a.image_width, a.image_height) != (b.image_width, b.image_height):
            raise SystemExit(f"resolution differs for {a.image_name}")
        for x, y in ((a.world_view_transform, b.world_view_transform),
                     (a.full_proj_transform, b.full_proj_transform)):
            worst = max(worst, float((x - y).abs().max()))
        if abs(a.FoVx - b.FoVx) > 1e-9 or abs(a.FoVy - b.FoVy) > 1e-9:
            raise SystemExit(f"intrinsics differ for {a.image_name}")
    out["max_abs_extrinsic_or_projection_difference"] = worst
    if worst > 1e-6:
        raise SystemExit(f"the two models' cameras differ by {worst}; one gauge is required")
    out["resolution"] = [int(F.cams_all[0].image_width), int(F.cams_all[0].image_height)]
    return out


def load_gt(dataset_dir: Path, cam: str, frame: int, H: int, W: int, device="cuda"):
    """The dataset's own observation: RGB in [0,1] and metric depth (0 = no return).

    Same files and decoding as tools/run_iekf_scene.dataset_rgbd_frames, which
    the Phase-1 IEKF consumed. Fails closed on a missing file or a resolution
    mismatch (T8).
    """
    from PIL import Image
    from tools.run_iekf_scene import rgb_frame_path

    rgb_path = rgb_frame_path(Path(dataset_dir), cam, frame)
    dep_path = Path(dataset_dir) / "depth_npy" / cam / f"frame_{frame:06d}.npy"
    if rgb_path is None or not dep_path.is_file():
        raise SystemExit(f"missing GT for {cam} frame {frame}: {rgb_path} / {dep_path}")
    im = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    dz = np.load(dep_path).astype(np.float32)
    if im.shape[:2] != (H, W) or dz.shape != (H, W):
        raise SystemExit(f"{cam} frame {frame}: GT {im.shape[:2]}/{dz.shape}, "
                         f"camera {H}x{W}")
    if not np.isfinite(dz).all() or dz.min() < 0:
        raise SystemExit(f"{dep_path}: non-finite or negative depth")
    rgb = torch.from_numpy(im).permute(2, 0, 1).contiguous().to(device)
    depth = torch.from_numpy(dz).to(device)
    return rgb, depth, depth > 0, {"rgb": str(rgb_path), "depth": str(dep_path)}
