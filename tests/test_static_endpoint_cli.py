"""Check real endpoint CLI startup without requiring a CUDA device.

Run: python tests/test_static_endpoint_cli.py
"""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train
from tools.train_all_static_endpoints import _train_command


def check_startup(command):
    # Exercise train.main's real parser, defaults and dispatch. Only the CUDA
    # seeding and training body are replaced; this is not a render smoke test.
    with patch.object(sys, 'argv', command[1:]), \
         patch.object(train, 'safe_state'), \
         patch.object(train, 'training') as training:
        train.main()
        training.assert_called_once()
        return training.call_args.args[3]


def main():
    config = SimpleNamespace(iterations=14000, use_dino=False, use_sam3=False)
    scene = Path('/unused/real_rgbd_scene')
    for frame in (0, 484):
        source = scene if frame == 0 else scene / 'static_frames/frame_000484'
        command = _train_command(config, source, Path('/unused/model'), scene, frame)
        args = check_startup(command)
        assert not args.dynamic_motion
        assert args.iterations == args.static_train_iters == 14000
        assert args.outer_soft_labels  # Preserve the nested solver's default.
        assert not args.enable_pixel_matching

    base = _train_command(config, scene, Path('/unused/model'), scene, 0)
    nested = base + ['--dynamic_motion', '--phase1_only', '--enable_nested_killing_part_discovery',
                     '--phase1_motion_representation', 'rigid_killing_eulerian',
                     '--final_3dgs_path', '/unused/endpoint']
    for extra, error in (([], 'outer_label_mode render_graph'),
                         (['--outer_label_mode', 'render_graph'], 'enable_pixel_matching')):
        try:
            check_startup(nested + extra)
        except ValueError as exc:
            assert error in str(exc), str(exc)
        else:
            raise AssertionError(f'Nested solver accepted missing dependency: {error}')
    args = check_startup(nested + ['--outer_label_mode', 'render_graph', '--enable_pixel_matching'])
    assert args.outer_soft_labels and args.enable_pixel_matching
    print('PASS both static endpoint commands reach training; nested dependency guards remain active')


if __name__ == '__main__':
    main()
