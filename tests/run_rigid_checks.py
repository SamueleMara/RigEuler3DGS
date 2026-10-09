"""Run the rigid checks without requiring pytest in the training environment."""

from pathlib import Path
import runpy
import sys


def main():
    count = 0
    root = Path(__file__).parent
    sys.path.insert(0, str(root.parent))
    paths = (
        sorted(root.glob("test_rigid_*.py"))
        + sorted(root.glob("test_lagrangian_*.py"))
        + sorted(root.glob("test_eulerian_lagrangian_*.py"))
        + sorted(root.glob("test_depth_*.py"))
        + sorted(root.glob("test_kpart_*.py"))
        + [
            root / "test_killing_gaussian_rotation_replay.py",
            root / "test_joint_model_rendering_competition.py",
            root / "test_motion_correspondence_flow_stats.py",
            root / "test_temporal_visibility_pretraining.py",
        ]
    )
    for path in dict.fromkeys(paths):
        namespace = runpy.run_path(str(path))
        for name, function in namespace.items():
            if name.startswith("test_") and callable(function):
                function()
                count += 1
    print(f"{count} rigid checks passed")


if __name__ == "__main__":
    main()
