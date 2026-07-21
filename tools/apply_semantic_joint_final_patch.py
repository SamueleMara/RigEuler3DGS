#!/usr/bin/env python3
"""Install final joint estimation into the current semantic-isolated joint branch.

This patcher is intentionally strict. It refuses to integrate into the obsolete
all-Gaussian joint-understanding branch, because doing so would allow background
proposal motion to contaminate the rigid joint fit.
"""
from pathlib import Path
import argparse
import shutil
import re
import sys

PATCH_ROOT = Path(__file__).resolve().parents[1]


SEMANTIC_MARKERS = (
    "Semantic background uses only the unrestricted proposal field",
    "grid_mask = torch.ones_like(joint_fg_mask",
    "canonical_xyz=self._xyz.detach()[fg_ids].cpu()",
    "static_dino_field=dino.detach()[fg_ids].cpu()",
)


def require_semantic_contract(repo: Path):
    gm = repo / "scene" / "gaussian_model.py"
    text = gm.read_text()
    missing = [m for m in SEMANTIC_MARKERS if m not in text]
    if missing:
        raise RuntimeError(
            "scene/gaussian_model.py is not the corrected semantic-isolated joint branch. "
            "Missing markers:\n  - " + "\n  - ".join(missing) +
            "\nExpected behavior: global proposal/grid for all Gaussians; foreground-only "
            "XYZ+DINO clustering, assignments and rigid fitting. Refusing a silent partial patch."
        )


def copy_files(repo: Path):
    mapping = {
        PATCH_ROOT / "utils" / "final_joint_estimation.py": repo / "utils" / "final_joint_estimation.py",
        PATCH_ROOT / "tools" / "estimate_final_joint.py": repo / "tools" / "estimate_final_joint.py",
        PATCH_ROOT / "tools" / "validate_final_joint_estimation.py": repo / "tools" / "validate_final_joint_estimation.py",
        PATCH_ROOT / "tools" / "verify_semantic_joint_contract.py": repo / "tools" / "verify_semantic_joint_contract.py",
        PATCH_ROOT / "RIGEULER3DGS_FULL_CONTEXT.md": repo / "RIGEULER3DGS_FULL_CONTEXT.md",
    }
    for src, dst in mapping.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {src.name} -> {dst.relative_to(repo)}")


def patch_arguments(repo: Path):
    path = repo / "arguments" / "__init__.py"
    text = path.read_text()
    if "--disable_final_joint_estimation" not in text:
        anchor_candidates = [
            "    # Staged dynamic losses.",
            "    # Adaptive motion-grid refinement.",
            "def apply_dynamic_defaults",
        ]
        anchor = next((a for a in anchor_candidates if a in text), None)
        if anchor is None:
            raise RuntimeError("Could not locate insertion point in arguments/__init__.py")
        block = '''    # Final foreground moving-part joint estimation.\n    group.add_argument("--disable_final_joint_estimation", action="store_true", default=False,\n                       help="Disable final prismatic/revolute/screw selection and axis visualization.")\n    group.add_argument("--final_joint_dynamic_threshold", type=float, default=0.80,\n                       help="Minimum optimized foreground moving probability used by final joint estimation.")\n    group.add_argument("--final_joint_min_moving_points", type=int, default=64,\n                       help="Minimum semantic-foreground moving Gaussians required for final joint estimation.")\n    group.add_argument("--final_joint_max_fit_points", type=int, default=4096,\n                       help="Maximum moving foreground Gaussians used for joint model residual fitting.")\n    group.add_argument("--final_joint_max_html_points", type=int, default=20000,\n                       help="Maximum Gaussians shown in final_joint_estimation.html.")\n    group.add_argument("--final_joint_axis_length_scale", type=float, default=1.25,\n                       help="Displayed joint-axis length relative to moving-part extent.")\n    group.add_argument("--final_joint_omega_activity_threshold", type=float, default=1e-3,\n                       help="Angular-velocity activity threshold for rotational model fitting.")\n    group.add_argument("--final_joint_prismatic_rotation_ratio", type=float, default=0.10,\n                       help="Select prismatic when rotational linear activity is below this fraction of translation.")\n    group.add_argument("--final_joint_screw_min_pitch", type=float, default=1e-3,\n                       help="Minimum absolute fitted pitch in scene units per radian for screw selection.")\n    group.add_argument("--final_joint_screw_min_axial_ratio", type=float, default=0.03,\n                       help="Minimum axial-to-rotational velocity ratio for screw selection.")\n    group.add_argument("--final_joint_screw_min_improvement", type=float, default=0.05,\n                       help="Minimum residual improvement over revolute required to select screw.")\n\n'''
        text = text.replace(anchor, block + anchor, 1)

    # Correct misleading legacy help while preserving the CLI name.
    text = re.sub(
        r'(group\.add_argument\("--filter_sem"[^\n]*\n\s*help=")[^"]*("\))',
        r'\1Use the learned semantic field only to isolate joint understanding. Non-foreground Gaussians retain unrestricted proposal velocity; they are excluded from clustering, rigid fitting and final joint estimation.\2',
        text,
        count=1,
    )
    path.write_text(text)
    print("patched arguments/__init__.py")


HELPER = r'''

def _final_joint_config_from_dyn(dyn):
    from utils.final_joint_estimation import JointEstimationConfig
    return JointEstimationConfig(
        dynamic_threshold=float(getattr(dyn, "final_joint_dynamic_threshold", 0.80)),
        min_moving_points=int(getattr(dyn, "final_joint_min_moving_points", 64)),
        max_fit_points=int(getattr(dyn, "final_joint_max_fit_points", 4096)),
        max_html_points=int(getattr(dyn, "final_joint_max_html_points", 20000)),
        axis_length_scale=float(getattr(dyn, "final_joint_axis_length_scale", 1.25)),
        omega_activity_threshold=float(getattr(dyn, "final_joint_omega_activity_threshold", 1e-3)),
        prismatic_rotation_ratio=float(getattr(dyn, "final_joint_prismatic_rotation_ratio", 0.10)),
        screw_min_pitch=float(getattr(dyn, "final_joint_screw_min_pitch", 1e-3)),
        screw_min_axial_ratio=float(getattr(dyn, "final_joint_screw_min_axial_ratio", 0.03)),
        screw_min_relative_improvement=float(getattr(dyn, "final_joint_screw_min_improvement", 0.05)),
    )


def _maybe_write_final_joint(tracks_or_path, output_dir, dyn, tag="final"):
    if bool(getattr(dyn, "disable_final_joint_estimation", False)):
        return None
    if not bool(getattr(dyn, "joint_single_rigid_training", False)):
        return None
    try:
        from utils.final_joint_estimation import (
            estimate_and_write_final_joint,
            estimate_and_write_final_joint_from_file,
        )
        cfg = _final_joint_config_from_dyn(dyn)
        if isinstance(tracks_or_path, dict):
            result = estimate_and_write_final_joint(tracks_or_path, output_dir, cfg)
        else:
            result = estimate_and_write_final_joint_from_file(tracks_or_path, output_dir, cfg)
        print(
            f"[Final joint] {tag}: type={result['joint_type']}, "
            f"axis={result['axis_direction']}, anchor={result['axis_anchor']}, "
            f"moving={result['moving_part_count']}"
        )
        return result
    except Exception as exc:
        print(f"[Final joint] {tag} estimation failed: {exc}")
        return None
'''


def patch_train(repo: Path):
    path = repo / "train.py"
    text = path.read_text()
    if "def _maybe_write_final_joint(" not in text:
        anchor = "def export_gaussian_tracks(scene, gaussians, dyn):"
        if anchor not in text:
            raise RuntimeError("Could not locate export_gaussian_tracks in train.py")
        text = text.replace(anchor, HELPER + "\n\n" + anchor, 1)

    # Integrate into the joint_final phase snapshot using the already exported in-memory tracks.
    if "tag=\"joint_final snapshot\"" not in text:
        phase_pattern = re.compile(
            r"^(?P<indent>[ \t]*)_write_track_visual_debug\(scene\.model_path, tracks_path, dyn, safe_tag, out_dir\)[ \t]*$",
            re.MULTILINE,
        )
        match = phase_pattern.search(text)
        if match is None:
            raise RuntimeError("Could not locate phase debug track visualization call in train.py")
        indent = match.group("indent")
        phase_call = (
            match.group(0)
            + "\n" + indent + 'if safe_tag == "joint_final":'
            + "\n" + indent + '    _maybe_write_final_joint(tracks, out_dir, dyn, tag="joint_final snapshot")'
        )
        text = text[:match.start()] + phase_call + text[match.end():]

    # Integrate at model root after the final gaussian_tracks.pt export.
    final_marker = '''            else:\n                print("[Dynamic 3DGS] WARNING: GaussianModel.export_time_tracks missing; skipping track export.")\n        post_training_render_and_visualize('''
    final_repl = '''            else:\n                print("[Dynamic 3DGS] WARNING: GaussianModel.export_time_tracks missing; skipping track export.")\n        if tracks_path is not None:\n            _maybe_write_final_joint(tracks_path, scene.model_path, dyn, tag="post-training")\n        post_training_render_and_visualize('''
    if "tag=\"post-training\"" not in text:
        if final_marker in text:
            text = text.replace(final_marker, final_repl, 1)
        else:
            # Compatibility with versions without the export_time_tracks guard.
            simple = '''        if getattr(dyn, "export_gaussian_tracks", True):\n            tracks_path = export_gaussian_tracks(scene, gaussians, dyn)\n        post_training_render_and_visualize('''
            simple_repl = '''        if getattr(dyn, "export_gaussian_tracks", True):\n            tracks_path = export_gaussian_tracks(scene, gaussians, dyn)\n        if tracks_path is not None:\n            _maybe_write_final_joint(tracks_path, scene.model_path, dyn, tag="post-training")\n        post_training_render_and_visualize('''
            if simple not in text:
                raise RuntimeError("Could not locate final track-export block in train.py")
            text = text.replace(simple, simple_repl, 1)

    path.write_text(text)
    print("patched train.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="RigEuler3DGS repository root")
    args = ap.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    require_semantic_contract(repo)
    copy_files(repo)
    patch_arguments(repo)
    patch_train(repo)
    print("Semantic-isolated final joint patch applied successfully.")


if __name__ == "__main__":
    main()
