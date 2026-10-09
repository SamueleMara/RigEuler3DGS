#!/usr/bin/env python3
"""Import calibrated Orbbec episodes as metre-scale, rectified 3DGS scenes.

Uses the supplied global-rig COLMAP endpoints; never estimates new camera poses
or invents part labels. Native depth is registered, not resized into RGB.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.read_write_model import read_model, write_model, qvec2rotmat


def intrinsic(cal):
    return np.array([[cal['fx'], 0, cal['cx']], [0, cal['fy'], cal['cy']], [0, 0, 1.]], dtype=np.float64)


def calibration_maps(cal):
    color, depth = cal['color'], cal['depth']
    size = (color['width'], color['height'])
    target_k = intrinsic(color)
    # Camera/getProjectionMatrix uses centred intrinsics. Rectify the data to
    # that convention, including COLMAP track coordinates, without changing pose.
    target_k[:2, 2] = np.array(size) / 2
    maps = cv2.initUndistortRectifyMap(intrinsic(color), np.array(color['distortion']),
                                    None, target_k, size, cv2.CV_32FC1)
    yy, xx = np.indices((depth['height'], depth['width']))
    uv = np.stack((xx, yy), -1).reshape(-1, 1, 2).astype(np.float64)
    xy = cv2.undistortPoints(uv, intrinsic(depth), np.array(depth['distortion'])).reshape(-1, 2)
    rays = np.column_stack((xy, np.ones(len(xy))))
    return target_k, maps, rays


def register_depth(raw, cal, rays, target_k):
    """Native Z(mm) -> undistorted color optical Z(m), nearest-surface z-buffer."""
    depth, color = cal['depth'], cal['color']
    if raw.dtype != np.uint16 or raw.shape != (depth['height'], depth['width']):
        raise ValueError(f'Unexpected native depth: {raw.dtype} {raw.shape}')
    transform = cal['T_color_from_depth']
    if transform['translation_unit'] != 'mm':
        raise ValueError('Expected millimetre factory extrinsics')
    valid = raw.reshape(-1) > 0
    xyz = rays[valid] * raw.reshape(-1)[valid, None]
    xyz = xyz @ np.array(transform['rotation']).reshape(3, 3).T + np.array(transform['translation'])
    xyz = xyz[np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)]
    uv = np.rint((xyz @ target_k.T)[:, :2] / xyz[:, 2:]).astype(np.int64)
    w, h = color['width'], color['height']
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    out = np.full(h * w, np.inf, dtype=np.float32)
    np.minimum.at(out, uv[inside, 1] * w + uv[inside, 0], xyz[inside, 2] * 0.001)
    out[~np.isfinite(out)] = 0
    return out.reshape(h, w)


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.symlink_to(os.path.relpath(source, target.parent))


def write_ply(path, points):
    data = np.empty(len(points), dtype=[(n, 'f4') for n in ('x', 'y', 'z', 'nx', 'ny', 'nz')]
                    + [(n, 'u1') for n in ('red', 'green', 'blue')])
    xyz = np.array([p.xyz for p in points.values()])
    rgb = np.array([p.rgb for p in points.values()])
    for i, n in enumerate(('x', 'y', 'z')):
        data[n] = xyz[:, i]
    for n in ('nx', 'ny', 'nz'):
        data[n] = 0
    for i, n in enumerate(('red', 'green', 'blue')):
        data[n] = rgb[:, i]
    PlyData([PlyElement.describe(data, 'vertex')]).write(str(path))


def import_endpoint(source, dest, scene, calibrations):
    summary = json.loads((source / 'model_summary.json').read_text())
    if summary['model_geometry'] != 'depth_projected_multiview_global_metric_rig':
        raise ValueError(f'Endpoint is not in the calibrated common rig: {source}')
    if any(v['depth_units_from_meta'] != 'uint16_mm' for v in summary['views'].values()):
        raise ValueError('Unknown endpoint coordinate scale')
    cameras, images, points = read_model(str(source / 'sparse/0'), ext='.bin')
    converted_cameras, converted_images = {}, {}
    reprojection = []
    for iid, image in images.items():
        name = Path(image.name).parts[0]
        camera = cameras[image.camera_id]
        k, _, _ = calibrations[name]
        if camera.model != 'PINHOLE':
            raise ValueError(f'Expected undistorted PINHOLE model: {source}')
        delta = k[:2, 2] - camera.params[2:4]
        converted_cameras[camera.id] = camera._replace(params=np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]]))
        image = image._replace(tvec=image.tvec * .001, xys=image.xys + delta)
        converted_images[iid] = image
        link(scene / 'rgb' / image.name, dest / 'images' / image.name)
        good = image.point3D_ids >= 0
        if good.any():
            xyz = np.array([points[int(pid)].xyz for pid in image.point3D_ids[good]]) * .001
            xyz = xyz @ qvec2rotmat(image.qvec).T + image.tvec
            projected = (xyz @ k.T)[:, :2] / xyz[:, 2:]
            reprojection.extend(np.linalg.norm(projected - image.xys[good], axis=1).tolist())
    points = {pid: p._replace(xyz=p.xyz * .001) for pid, p in points.items()}
    sparse = dest / 'sparse/0'
    sparse.mkdir(parents=True, exist_ok=True)
    for ext in ('.bin', '.txt'):
        write_model(converted_cameras, converted_images, points, str(sparse), ext=ext)
    write_ply(sparse / 'points3D.ply', points)
    stats = dict(frame=summary['frame_index'], points=len(points), cameras=len(images),
                 source=str(source), units='metres',
                 reprojection_median_px=float(np.median(reprojection)),
                 reprojection_p95_px=float(np.quantile(reprojection, .95)))
    (dest / 'import_summary.json').write_text(json.dumps(stats, indent=2))
    return converted_images, stats


def import_episode(source, target, calibration, fps):
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite existing input: {target}')
    cams = sorted(p.name for p in (source / 'rgb').iterdir() if p.is_dir())
    frames = {cam: sorted((source / 'rgb' / cam).glob('frame_*.jpg')) for cam in cams}
    ids = [[int(p.stem.split('_')[-1]) for p in frames[cam]] for cam in cams]
    if not ids or not ids[0] or any(x != ids[0] for x in ids) or ids[0] != list(range(ids[0][-1] + 1)):
        raise ValueError('RGB frame IDs must be synchronized and contiguous from zero')
    last = ids[0][-1]
    for f in (0, last):
        if not (source / f'colmap/frame_{f:06d}/sparse/0/points3D.bin').is_file():
            raise FileNotFoundError(f'Missing calibrated endpoint frame {f} in {source}')
    maps = {cam: calibration_maps(calibration[cam]) for cam in cams}
    for cam in cams:
        for kind in ('rgb', 'mask', 'depth_npy'):
            (target / kind / cam).mkdir(parents=True, exist_ok=True)
        cal = calibration[cam]
        k, remap, rays = maps[cam]
        for path in tqdm(frames[cam], desc=f'{source.name} {cam}', unit='frame'):
            rgb = cv2.imread(str(path))
            mask = cv2.imread(str(source / 'mask' / cam / (path.stem + '.png')), cv2.IMREAD_GRAYSCALE)
            raw = cv2.imread(str(source / 'depth_png' / cam / (path.stem + '.png')), cv2.IMREAD_UNCHANGED)
            shape = (cal['color']['height'], cal['color']['width'])
            if rgb is None or mask is None or raw is None or rgb.shape[:2] != shape or mask.shape != shape:
                raise ValueError(f'Missing/misaligned RGB, mask or native depth: {path}')
            rgb = cv2.remap(rgb, *remap, cv2.INTER_LINEAR)
            mask = cv2.remap(mask, *remap, cv2.INTER_NEAREST)
            for dst, array in ((target / 'rgb' / cam / path.name, rgb),
                               (target / 'mask' / cam / (path.stem + '.png'), (mask > 0).astype(np.uint8) * 255)):
                if not cv2.imwrite(str(dst), array):
                    raise OSError(f'Image write failed: {dst}')
            np.save(target / 'depth_npy' / cam / (path.stem + '.npy'), register_depth(raw, cal, rays, k))
    first_images, first_stats = import_endpoint(source / 'colmap/frame_000000', target / 'colmap', target, maps)
    last_images, last_stats = import_endpoint(source / f'colmap/frame_{last:06d}', target / f'colmap_frame{last}', target, maps)
    for iid, image in first_images.items():
        if not np.allclose(qvec2rotmat(image.qvec), qvec2rotmat(last_images[iid].qvec), atol=1e-7) or not np.allclose(image.tvec, last_images[iid].tvec, atol=1e-7):
            raise ValueError('Endpoint camera poses disagree: cannot use a fixed camera rig')
    endpoint_scene = target / 'static_frames' / f'frame_{last:06d}'
    for kind in ('rgb', 'mask', 'depth_npy'):
        link(target / kind, endpoint_scene / kind)
    link(target / f'colmap_frame{last}', endpoint_scene / 'colmap')
    shutil.copytree(source / 'metadata', target / 'metadata/source')
    meta = target / 'metadata'
    (meta / 'factory_calibration.json').write_text(json.dumps(calibration, indent=2))
    (meta / 'timestamps.json').write_text(json.dumps({str(i): i / fps for i in ids[0]}))
    manifest = dict(source=str(source), units='metres', camera_names=cams, frame_count=len(ids[0]),
                    endpoint_frame=last, endpoint_scene=str(endpoint_scene.relative_to(target)),
                    fps=fps, timestamp_source='frame_index / source metadata fps (not sensor acquisition timestamps)',
                    depth='float32 metre color-optical Z, registered sparse z-buffer; zero = unobserved',
                    rgb='factory undistorted, principal point recentered for GraphDECO camera',
                    mask='binary foreground only, not static/dynamic ground truth', gt_available=False,
                    endpoint_models=[first_stats, last_stats])
    (meta / 'real_rgbd_import.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, default=REPO / 'input')
    args = parser.parse_args()
    source = args.source.resolve()
    calibration = json.loads((source / 'meta/orbbec_calibration_factory_exact.json').read_text())['cameras']
    fps = float(json.loads((source / 'meta/info.json').read_text())['fps'])
    if fps <= 0:
        raise ValueError('Invalid source fps')
    episodes = sorted((source / 'output').glob('episode_*'))
    if not episodes:
        raise FileNotFoundError(f'No processed episodes in {source}')
    for episode in episodes:
        import_episode(episode, args.input_root.resolve() / f'{source.name}_{episode.name}', calibration, fps)


if __name__ == '__main__':
    main()
