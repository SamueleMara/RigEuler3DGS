#!/usr/bin/env python3
"""Train frame-0/final static 3DGS, including explicitly imported real RGB-D scenes."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.static_endpoint_scene import (  # noqa: E402
    build_labeled_frame_static_scene,
    has_inherited_gt_labels,
    has_sparse_gt_parts,
    model_ply_path,
    scene_last_frame,
)


def _iter_scenes(input_root: Path, only: set[str] | None) -> list[Path]:
    scenes: list[Path] = []
    for scene in sorted(input_root.iterdir()):
        if not scene.is_dir():
            continue
        if only is not None and scene.name not in only:
            continue
        if not (scene / "colmap").is_dir() or not (scene / "rgb").is_dir():
            continue
        scenes.append(scene)
    return scenes


def _has_frame_cache(root: Path, frame_idx: int) -> bool:
    if not root.is_dir():
        return False
    return any((root / cam / f"frame_{int(frame_idx):06d}.pt").is_file() for cam in os.listdir(root))


def _train_command(args, source_scene: Path, model_dir: Path, scene: Path, frame_idx: int) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO / "train.py"),
        "-s",
        str(source_scene),
        "-m",
        str(model_dir),
        "--iterations",
        str(int(args.iterations)),
        "--static_train_iters",
        str(int(args.iterations)),
        "--use_semantic_feature_field",
        "--disable_viewer",
        "--no_wandb",
    ]
    # Depth supervision, when the scene actually carries depth. These exports
    # store float32 METRES in depth_npy/; the .png beside it is a normalised
    # visualisation and must not be used as a measurement. The source scene is
    # what the COLMAP reader opens, so the directory is looked for there --
    # the synthesised frame_t scene does not always have one.
    depth_dir = Path(source_scene) / "depth_npy"
    if depth_dir.is_dir() and any(depth_dir.iterdir()):
        cmd.extend(["--depths", "depth_npy", "--sapien_depth_is_metric"])
    # Object masks as ground-truth alpha. Absolute, and taken from the ORIGINAL
    # scene rather than source_scene: the synthesised frame_t scene has its own
    # colmap/ but no mask/, and its images are symlinks back into <scene>/rgb/,
    # so one mask tree serves both endpoints. Without this the RGB term scores
    # background pixels against nothing and the silhouette grows black splats.
    mask_dir = Path(scene) / "mask"
    if mask_dir.is_dir() and any(mask_dir.iterdir()):
        cmd.extend(["--masks", str(mask_dir.resolve())])
    dino_root = scene / "dinov3"
    if bool(args.use_dino) and _has_frame_cache(dino_root, int(frame_idx)):
        cmd.extend(["--use_dino_feature_field", "--dinov3_root", str(dino_root)])
    sam3_root = scene / "sam3"
    if bool(args.use_sam3) and _has_frame_cache(sam3_root, int(frame_idx)):
        cmd.extend(
            [
                "--use_sam3_feature_field",
                "--sam3_feature_root",
                str(sam3_root),
                "--sam3_feature_dim",
                str(int(args.sam3_feature_dim)),
            ]
        )
    return cmd


def _run_train(args, cmd: list[str], log: Path) -> int:
    if args.dry_run:
        tqdm.write(" ".join(cmd))
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_HUB_OFFLINE", "1")
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return int(result.returncode)


@contextlib.contextmanager
def _single_batch_lock(lock_path: Path, *, dry_run: bool):
    """Prevent two endpoint batches from training into the same outputs.

    The previous failure mode was two identical nohup jobs running at once.  They
    used the same GPU, same model directories, and same log files; one process
    can then be killed with SIGKILL while the other overwrites the log with a
    successful-looking run.  A non-blocking flock makes that state impossible.
    """
    if dry_run:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(
                f"another static-endpoint batch is already running for this output; "
                f"lock={lock_path}"
            ) from exc
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()}\n")
        lock.write(f"cwd={REPO}\n")
        lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _ensure_model(args, *, phase: str, source_scene: Path, model_dir: Path, scene: Path, frame_idx: int, log: Path) -> Path:
    ply = model_ply_path(model_dir, int(args.iterations))
    requires_gt = has_sparse_gt_parts(scene)
    cmd = _train_command(args, source_scene, model_dir, scene, int(frame_idx))
    # Reuse only if the existing model was trained the way this run would
    # train it. A checkpoint at the right iteration is not enough: a model
    # fitted before depth supervision worked sits at the same path under the
    # same name, and silently reusing it is how one endpoint ends up with
    # depth and the other without.
    wants_depth = "--depths" in cmd
    has_depth = False
    cfg = model_dir / "cfg_args"
    if cfg.is_file():
        text = cfg.read_text(errors="replace")
        has_depth = ("sapien_depth_is_metric=True" in text
                     and "depths=''" not in text)
    if (ply.is_file() and (not requires_gt or has_inherited_gt_labels(ply))
            and not bool(args.force)):
        if wants_depth and not has_depth:
            tqdm.write(f"{scene.name} {phase}: retraining -- the existing "
                       f"model was fitted without depth")
        else:
            tqdm.write(f"{scene.name} {phase}: reuse {ply}")
            return ply
    code = _run_train(args, cmd, log)
    if code != 0:
        raise RuntimeError(f"{scene.name} {phase} train failed exit={code}; log={log}")
    if args.dry_run:
        return ply
    if not ply.is_file():
        raise FileNotFoundError(f"{scene.name} {phase} training did not write {ply}")
    if requires_gt and not has_inherited_gt_labels(ply):
        raise RuntimeError(f"{scene.name} {phase} model lacks inherited GT labels: {ply}")
    return ply


def _imported_endpoint(scene: Path, last: int) -> Path:
    """Only accept a prepared real endpoint, never reuse frame-0 moving geometry."""
    manifest_path = scene / 'metadata/real_rgbd_import.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Unlabeled scene needs a calibrated endpoint import: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    if int(manifest['endpoint_frame']) != last or manifest['units'] != 'metres':
        raise ValueError(f'Invalid real endpoint manifest: {manifest_path}')
    endpoint = (scene / manifest['endpoint_scene']).resolve()
    if not endpoint.is_relative_to(scene.resolve()):
        raise ValueError('Prepared endpoint must be inside its input scene')
    for filename in ('cameras.bin', 'images.bin', 'points3D.bin', 'points3D.ply'):
        if not (endpoint / 'colmap/sparse/0' / filename).is_file():
            raise FileNotFoundError(endpoint / 'colmap/sparse/0' / filename)
    return endpoint


def _merge_summary(summary_path: Path, *, scenes_run: set[str], manifests: list[dict], failures: list[str]) -> dict:
    """Merge a partial ``--only`` repair into the existing batch summary."""
    existing = {"schema_version": 1, "manifests": [], "failures": []}
    if summary_path.is_file():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {"schema_version": 1, "manifests": [], "failures": []}

    merged_manifests = {
        item.get("scene"): item
        for item in existing.get("manifests", [])
        if item.get("scene") and item.get("scene") not in scenes_run
    }
    for item in manifests:
        if item.get("scene"):
            merged_manifests[item["scene"]] = item

    def failure_scene(line: str) -> str:
        return str(line).split(":", 1)[0].strip()

    merged_failures = [
        line for line in existing.get("failures", [])
        if failure_scene(line) not in scenes_run
    ]
    merged_failures.extend(failures)
    return {
        "schema_version": 1,
        "manifests": [merged_manifests[name] for name in sorted(merged_manifests)],
        "failures": merged_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(REPO / "input"))
    parser.add_argument("--output-root", default=str(REPO / "output"))
    parser.add_argument("--log-dir", default=str(REPO / "output" / "static_endpoint_logs"))
    parser.add_argument("--tag", default="static_endpoints_30k")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--only", default="", help="Comma-separated scene names.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-missing-gt", action="store_true")
    parser.add_argument("--allow-unlabeled", action="store_true",
                        help="Train imported real RGB-D scenes using their calibrated endpoint models; no GT labels required.")
    parser.add_argument("--use-dino", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-sam3", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam3-feature-dim", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.allow_unlabeled and args.fail_missing_gt:
        parser.error('--allow-unlabeled conflicts with --fail-missing-gt')

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    only = {x.strip() for x in args.only.replace(",", " ").split() if x.strip()} or None

    failures: list[str] = []
    manifests: list[dict] = []
    scenes = _iter_scenes(input_root, only)
    lock_path = log_dir / f"{args.tag}_{int(args.iterations)}.lock"
    with _single_batch_lock(lock_path, dry_run=bool(args.dry_run)):
        for scene in tqdm(scenes, desc="static endpoints", unit="scene"):
            last = scene_last_frame(scene)
            if last is None:
                tqdm.write(f"SKIP {scene.name}: no rgb frames")
                continue
            if not has_sparse_gt_parts(scene) and not args.allow_unlabeled:
                msg = f"{scene.name}: missing pointcloud/points3D_parts.ply"
                if args.fail_missing_gt:
                    failures.append(msg)
                tqdm.write(f"SKIP {msg}")
                continue
            out = output_root / f"{scene.name}_{args.tag}" / "shared"
            frame0_dir = out / "frame0_static"
            frame_t_dir = out / "frame_t_static"
            frame_t_scene = out / "frame_t_scene"
            tqdm.write(f"START {scene.name}: frame0 + frame{last}")
            try:
                if not has_sparse_gt_parts(scene):
                    frame_t_scene = _imported_endpoint(scene, int(last))
                frame0_ply = _ensure_model(
                    args,
                    phase="frame0",
                    source_scene=scene,
                    model_dir=frame0_dir,
                    scene=scene,
                    frame_idx=0,
                    log=log_dir / f"{scene.name}_frame0_{int(args.iterations)}.log",
                )
                if not args.dry_run and has_sparse_gt_parts(scene):
                    build_labeled_frame_static_scene(scene, frame_t_scene, int(last), force=bool(args.force))
                frame_t_ply = _ensure_model(
                    args,
                    phase=f"frame{last}",
                    source_scene=frame_t_scene,
                    model_dir=frame_t_dir,
                    scene=scene,
                    frame_idx=int(last),
                    log=log_dir / f"{scene.name}_frame{int(last)}_{int(args.iterations)}.log",
                )
                manifest = {
                    "scene": scene.name,
                    "root_scene": str(scene),
                    "frame0_static_dir": str(frame0_dir),
                    "frame0_ply": str(frame0_ply),
                    "endpoint_frame": int(last),
                    "frame_t_scene": str(frame_t_scene),
                    "frame_t_static_dir": str(frame_t_dir),
                    "frame_t_ply": str(frame_t_ply),
                    "iterations": int(args.iterations),
                    "gt_available": has_sparse_gt_parts(scene),
                }
                manifests.append(manifest)
                if not args.dry_run:
                    (out / "static_endpoints_manifest.json").write_text(
                        json.dumps(manifest, indent=2), encoding="utf-8"
                    )
            except Exception as exc:
                failures.append(f"{scene.name}: {exc}")
                tqdm.write(f"FAIL {scene.name}: {exc}")

    if not args.dry_run:
        summary_path = output_root / f"{args.tag}_summary.json"
        summary = _merge_summary(
            summary_path,
            scenes_run={scene.name for scene in scenes},
            manifests=manifests,
            failures=failures,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        print("FAILED")
        for failure in failures:
            print(failure)
        return 1
    print("COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
