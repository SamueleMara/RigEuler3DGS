"""CPU regression for metric registration and real endpoint selection.

Run: python tests/test_fixedk_rgbd_import.py
"""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.import_fixedk_rgbd import calibration_maps, register_depth
from tools.train_all_static_endpoints import _imported_endpoint


def main():
    sensor = dict(width=4, height=4, fx=2., fy=2., cx=2., cy=2., distortion=[0.] * 8)
    cal = dict(color=sensor.copy(), depth=sensor.copy(), T_color_from_depth=dict(
        rotation=np.eye(3).reshape(-1).tolist(), translation=[0., 0., 0.], translation_unit='mm'))
    k, _, rays = calibration_maps(cal)
    raw = np.full((4, 4), 1000, dtype=np.uint16)
    raw[0, 0] = 0
    result = register_depth(raw, cal, rays, k)
    np.testing.assert_allclose(result, raw * .001)
    # Real registration translates points; it must not just resize the depth.
    cal['T_color_from_depth']['translation'] = [500., 0., 0.]
    shifted = register_depth(raw, cal, rays, k)
    np.testing.assert_allclose(shifted[:, 1:], result[:, :-1])
    assert not shifted[:, 0].any()
    # Two depth rays projected to one color pixel keep the front surface.
    cal['T_color_from_depth']['translation'] = [0., 0., 0.]
    raw[:] = 2000
    raw[0, 0] = 500
    collided = register_depth(raw, cal, np.tile([0., 0., 1.], (16, 1)), k)
    assert collided[2, 2] == .5 and np.count_nonzero(collided) == 1
    with tempfile.TemporaryDirectory() as tmp:
        scene = Path(tmp)
        (scene / 'metadata').mkdir()
        sparse = scene / 'static_frames/frame_000010/colmap/sparse/0'
        sparse.mkdir(parents=True)
        for name in ('cameras.bin', 'images.bin', 'points3D.bin', 'points3D.ply'):
            (sparse / name).touch()
        (scene / 'metadata/real_rgbd_import.json').write_text(json.dumps(dict(
            endpoint_frame=10, units='metres', endpoint_scene='static_frames/frame_000010')))
        assert _imported_endpoint(scene, 10) == scene / 'static_frames/frame_000010'
        try:
            _imported_endpoint(scene, 9)
        except ValueError:
            pass
        else:
            raise AssertionError('An incorrect endpoint frame was accepted')
    print('PASS metric depth, calibration transform, z-buffer, prepared endpoint validation')


if __name__ == '__main__':
    main()
