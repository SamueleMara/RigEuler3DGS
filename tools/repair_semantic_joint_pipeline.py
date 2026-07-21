#!/usr/bin/env python3
"""Repair the exact semantic-joint pipeline in an existing RigEuler3DGS repo.

This repair is intentionally strict and idempotent. It replaces complete
GaussianModel methods instead of trying to insert a single argument into an
unknown call layout.

Canonical contract
------------------
* The proposal velocity field and adaptive grid remain global.
* The frozen semantic field selects the subset used for joint understanding.
* Exact XYZ+DINO initialization receives full-scene tensors plus the full-scene
  semantic mask. It performs its own foreground sub-selection exactly once.
* JointPartField owns one trainable vector per semantic-foreground Gaussian.
* Background never enters clustering, twist fitting, type selection, axis, or
  pitch, but it keeps proposal velocity.
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
import time
from pathlib import Path


GET_MASK_METHOD = '''    @property
    def get_dynamic_foreground_mask(self):
        """Frozen full-scene mask used only for joint understanding."""
        mask = getattr(self, "_dynamic_foreground_mask", None)
        if torch.is_tensor(mask) and mask.numel() == self._xyz.shape[0]:
            return mask.reshape(-1).bool()
        if bool(getattr(self, "joint_single_rigid_enabled", False)):
            raise RuntimeError(
                "Joint semantic foreground has not been initialized. Refusing an "
                "all-Gaussian fallback because it would contaminate rigid fitting."
            )
        return torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
'''

BUILD_MASK_METHOD = '''    @torch.no_grad()
    def build_dynamic_foreground_mask(self, args):
        """Freeze the semantic subset used by clustering and rigid fitting.

        This mask does not disable proposal motion for non-foreground points.
        """
        n = int(self._xyz.shape[0])
        use_semantic = (
            bool(getattr(args, "filter_sem", False))
            or bool(getattr(args, "use_semantic_dynamic_foreground", False))
        )
        if bool(getattr(args, "disable_semantic_dynamic_foreground", False)):
            use_semantic = False
        if bool(getattr(self, "joint_single_rigid_enabled", False)) and not use_semantic:
            raise RuntimeError(
                "Exact one-moving-part training requires --filter_sem. The semantic "
                "field isolates joint understanding; proposal velocity remains global."
            )

        threshold = float(getattr(args, "semantic_dynamic_threshold", 0.5))
        minimum = int(getattr(args, "semantic_dynamic_min_points", 32))
        self.dynamic_foreground_threshold = threshold
        sem_prob = self.get_semantic_probability
        if sem_prob is None or sem_prob.numel() != n:
            raise RuntimeError(
                "--filter_sem was requested, but the loaded static PLY does not contain "
                "one semantic value for every Gaussian."
            )
        probability = sem_prob.detach().reshape(-1)
        mask = probability >= threshold
        count = int(mask.sum().item())
        if count < max(2, minimum):
            raise RuntimeError(
                f"Semantic foreground has {count}/{n} Gaussians at threshold "
                f"{threshold}, below semantic_dynamic_min_points={minimum}. "
                "Refusing top-k/background fallback."
            )
        self._dynamic_foreground_mask = mask.detach().bool()
        print(
            "[Semantic joint] Frozen foreground mask: "
            f"foreground={count}, background={n-count}, full={n}, threshold={threshold}"
        )
        return self._dynamic_foreground_mask
'''

ASSERT_METHOD = '''    @torch.no_grad()
    def assert_joint_foreground_isolation(self):
        """Fail if JointPartField IDs differ from the frozen semantic mask."""
        part_field = getattr(self, "joint_part_field", None)
        if part_field is None:
            raise RuntimeError("joint_part_field is not initialized")
        mask = self.get_dynamic_foreground_mask.detach().reshape(-1).bool()
        expected = torch.where(mask)[0]
        actual = part_field.foreground_ids.reshape(-1).long().to(expected.device)
        if expected.numel() != actual.numel() or not torch.equal(expected, actual):
            extra = actual[~torch.isin(actual, expected)] if actual.numel() else actual
            missing = expected[~torch.isin(expected, actual)] if expected.numel() else expected
            raise RuntimeError(
                "Semantic joint isolation failed: "
                f"expected={expected.numel()}, actual={actual.numel()}, "
                f"background_leaks={extra.numel()}, missing_foreground={missing.numel()}"
            )
        return {
            "num_gaussians_full": int(mask.numel()),
            "num_joint_foreground": int(expected.numel()),
            "num_proposal_only_background": int((~mask).sum().item()),
        }
'''

INITIALIZE_METHOD = '''    @torch.no_grad()
    def initialize_joint_part_field(self, args):
        """Initialize foreground-static + one moving slot from exact XYZ+DINO.

        IMPORTANT: canonical XYZ and DINO are passed at full-scene length. The
        full-scene semantic mask is passed separately. The initializer performs
        the foreground index selection exactly once, preserving original IDs.
        """
        dino = self._dino_feature_field
        n_full = int(self._xyz.shape[0])
        if (
            dino is None
            or not torch.is_tensor(dino)
            or dino.ndim != 2
            or dino.shape[0] != n_full
            or dino.shape[1] <= 0
        ):
            raise RuntimeError(
                "Exact joint initialization requires the frozen static DINO field "
                "for every loaded Gaussian."
            )

        semantic_mask = self.get_dynamic_foreground_mask.detach().reshape(-1).bool()
        if semantic_mask.numel() != n_full:
            raise RuntimeError(
                f"Semantic mask length {semantic_mask.numel()} != Gaussian count {n_full}"
            )
        n_fg = int(semantic_mask.sum().item())
        if n_fg < 2:
            raise RuntimeError("Semantic foreground is too small for static/moving initialization")

        result = initialize_joint_parts_from_static_dino(
            canonical_xyz=self._xyz.detach().cpu(),
            static_dino_field=dino.detach().cpu(),
            foreground_mask=semantic_mask.detach().cpu(),
            num_moving_parts=int(getattr(args, "joint_num_moving_parts", 1)),
            xyz_weight=float(getattr(args, "joint_init_euclidean_weight", 1.0)),
            dino_weight=float(getattr(args, "joint_init_cosine_weight", 1.0)),
            xyz_scale=float(getattr(args, "joint_init_euclidean_scale", -1.0)),
            distance_block_size=int(getattr(args, "joint_init_distance_block_size", 512)),
            distance_device="cpu",
            distance_storage=str(getattr(args, "joint_init_distance_storage", "float16")),
            kmedoids_iterations=int(getattr(args, "joint_init_kmedoids_iters", 12)),
        )
        self._joint_initialization_result = result
        self._joint_initialization_config = {
            "xyz_weight": float(getattr(args, "joint_init_euclidean_weight", 1.0)),
            "dino_weight": float(getattr(args, "joint_init_cosine_weight", 1.0)),
            "save_distance_matrix": bool(getattr(args, "joint_save_initial_distance_matrix", True)),
        }
        self.joint_part_field = JointPartField(
            num_gaussians=n_full,
            foreground_ids=result.foreground_ids,
            initial_cluster_id_fg=result.initial_cluster_id_fg,
            num_moving_parts=int(getattr(args, "joint_num_moving_parts", 1)),
            embedding_dim=int(getattr(args, "joint_part_field_dim", 8)),
            assignment_temperature=float(getattr(args, "joint_part_assignment_temperature", 1.0)),
            init_noise=float(getattr(args, "joint_part_field_init_noise", 0.02)),
        ).to(self._xyz.device)
        self._joint_cluster_id = self.joint_part_field.initial_cluster_id_full().detach()
        isolation = self.assert_joint_foreground_isolation()
        print(
            "[Joint init] Semantic isolation OK: "
            f"joint_foreground={isolation['num_joint_foreground']}, "
            f"proposal_only_background={isolation['num_proposal_only_background']}, "
            f"full={isolation['num_gaussians_full']}, matrix={tuple(result.combined_distance.shape)}"
        )
        return self._joint_cluster_id
'''


def _class_node(tree: ast.AST, class_name: str) -> ast.ClassDef:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise RuntimeError(f"Class {class_name} not found")


def _method_span(source: str, class_name: str, method_name: str):
    tree = ast.parse(source)
    cls = _class_node(tree, class_name)
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            end = node.end_lineno
            return start, end
    return None


def _replace_method(source: str, class_name: str, method_name: str, replacement: str) -> str:
    span = _method_span(source, class_name, method_name)
    if span is None:
        raise RuntimeError(f"Required method {class_name}.{method_name} not found")
    lines = source.splitlines(keepends=True)
    start, end = span
    repl = replacement.rstrip() + "\n\n"
    return "".join(lines[:start]) + repl + "".join(lines[end:])


def _insert_method_before(source: str, class_name: str, before_name: str, method_source: str) -> str:
    span = _method_span(source, class_name, before_name)
    if span is None:
        raise RuntimeError(f"Insertion anchor {class_name}.{before_name} not found")
    lines = source.splitlines(keepends=True)
    start, _ = span
    return "".join(lines[:start]) + method_source.rstrip() + "\n\n" + "".join(lines[start:])


def _patch_init_dynamic_motion(source: str) -> str:
    span = _method_span(source, "GaussianModel", "init_dynamic_motion")
    if span is None:
        raise RuntimeError("GaussianModel.init_dynamic_motion not found")
    lines = source.splitlines(keepends=True)
    start, end = span
    block = "".join(lines[start:end])

    # Remove the confirmed bad all-Gaussian joint mask while retaining a global grid.
    block = re.sub(
        r"(?m)^(\s*)fg_mask\s*=\s*torch\.ones_like\(semantic_mask,\s*dtype=torch\.bool\)\s*$",
        r"\1joint_fg_mask = semantic_mask.detach().bool().clone()\n"
        r"\1self._dynamic_foreground_mask = joint_fg_mask\n"
        r"\1grid_mask = torch.ones_like(joint_fg_mask, dtype=torch.bool)",
        block,
    )
    block = re.sub(
        r"(?m)^(\s*)fg_mask\s*=\s*semantic_mask[^\n]*$",
        r"\1joint_fg_mask = semantic_mask.detach().bool().clone()\n"
        r"\1self._dynamic_foreground_mask = joint_fg_mask\n"
        r"\1grid_mask = torch.ones_like(joint_fg_mask, dtype=torch.bool)",
        block,
    )

    if "semantic_mask = self.build_dynamic_foreground_mask(args)" not in block:
        marker = "self.joint_single_rigid_enabled = bool(getattr(args, \"joint_single_rigid_training\", False))"
        if marker not in block:
            raise RuntimeError("Cannot locate joint_single_rigid_enabled assignment")
        block = block.replace(
            marker,
            marker + "\n        semantic_mask = self.build_dynamic_foreground_mask(args)\n"
            "        joint_fg_mask = semantic_mask.detach().bool().clone()\n"
            "        self._dynamic_foreground_mask = joint_fg_mask\n"
            "        grid_mask = torch.ones_like(joint_fg_mask, dtype=torch.bool)",
            1,
        )

    if "joint_fg_mask" not in block:
        block = block.replace(
            "semantic_mask = self.build_dynamic_foreground_mask(args)",
            "semantic_mask = self.build_dynamic_foreground_mask(args)\n"
            "        joint_fg_mask = semantic_mask.detach().bool().clone()\n"
            "        self._dynamic_foreground_mask = joint_fg_mask\n"
            "        grid_mask = torch.ones_like(joint_fg_mask, dtype=torch.bool)",
            1,
        )
    if "grid_mask" not in block:
        block = block.replace(
            "joint_fg_mask = semantic_mask",
            "joint_fg_mask = semantic_mask",
            1,
        )
        insert = "\n        grid_mask = torch.ones_like(joint_fg_mask, dtype=torch.bool)"
        pos = block.find("self._dynamic_foreground_mask", block.find("joint_fg_mask"))
        pos = block.find("\n", pos)
        block = block[:pos] + insert + block[pos:]

    block = block.replace("foreground_mask=fg_mask", "foreground_mask=grid_mask")
    # Some intermediate branches incorrectly pass the joint mask to the global grid.
    block = block.replace("foreground_mask=joint_fg_mask", "foreground_mask=grid_mask")
    block = block.replace("self._xyz.detach().mean(dim=0).detach()", "self._xyz.detach()[joint_fg_mask].mean(dim=0).detach()")

    if "self.initialize_joint_part_field(args)" not in block:
        raise RuntimeError("init_dynamic_motion no longer calls initialize_joint_part_field")

    return "".join(lines[:start]) + block + "".join(lines[end:])


def _patch_checkpoint_compat(source: str) -> str:
    source = source.replace(
        '"gaussian_ids": self.joint_part_field.gaussian_ids.detach(),',
        '"foreground_ids": self.joint_part_field.foreground_ids.detach(),',
    )
    source = source.replace(
        'saved_ids = part_cfg.get("gaussian_ids", part_cfg.get("foreground_ids", None))',
        'saved_ids = part_cfg.get("foreground_ids", part_cfg.get("gaussian_ids", None))',
    )
    source = source.replace("gaussian_ids=saved_ids.long(),", "foreground_ids=saved_ids.long(),")
    source = source.replace(
        'if "foreground_ids" in part_state and "gaussian_ids" not in part_state:\n'
        '                    part_state = dict(part_state)\n'
        '                    part_state["gaussian_ids"] = part_state.pop("foreground_ids")',
        'if "gaussian_ids" in part_state and "foreground_ids" not in part_state:\n'
        '                    part_state = dict(part_state)\n'
        '                    part_state["foreground_ids"] = part_state.pop("gaussian_ids")',
    )
    source = source.replace(
        "self.joint_part_field.foreground_mask_full().detach()",
        "self.joint_part_field.foreground_mask_full.detach()",
    )
    return source


def patch_gaussian_model(path: Path) -> None:
    source = path.read_text()
    source = _replace_method(source, "GaussianModel", "get_dynamic_foreground_mask", GET_MASK_METHOD)
    source = _replace_method(source, "GaussianModel", "build_dynamic_foreground_mask", BUILD_MASK_METHOD)
    source = _replace_method(source, "GaussianModel", "initialize_joint_part_field", INITIALIZE_METHOD)
    if _method_span(source, "GaussianModel", "assert_joint_foreground_isolation") is None:
        source = _insert_method_before(source, "GaussianModel", "initialize_joint_part_field", ASSERT_METHOD)
    else:
        source = _replace_method(source, "GaussianModel", "assert_joint_foreground_isolation", ASSERT_METHOD)
    source = _patch_init_dynamic_motion(source)
    source = _patch_checkpoint_compat(source)
    ast.parse(source)
    path.write_text(source)


def patch_loss_wrapper(path: Path) -> None:
    source = path.read_text()
    if "_joint_single_rigid_losses_semantic_unmasked" in source:
        return
    marker = "def joint_single_rigid_losses("
    if marker not in source:
        raise RuntimeError("joint_single_rigid_losses not found")
    source = source.replace(marker, "def _joint_single_rigid_losses_semantic_unmasked(", 1)
    wrapper = r'''


def joint_single_rigid_losses(*args, **kwargs):
    """Run every joint-specific loss on semantic foreground only."""
    import inspect as _inspect
    target = _joint_single_rigid_losses_semantic_unmasked
    bound = _inspect.signature(target).bind_partial(*args, **kwargs)
    part_field = bound.arguments.get("part_field", None)
    if part_field is None or not hasattr(part_field, "foreground_ids"):
        raise RuntimeError("Foreground-only JointPartField is required")
    ids = part_field.foreground_ids.reshape(-1).long()
    n_full = int(part_field.num_gaussians)
    for name in ("total_velocity", "rigid_velocity", "proposal_velocity"):
        value = bound.arguments.get(name, None)
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == n_full:
            bound.arguments[name] = value.index_select(0, ids.to(value.device))
    value = bound.arguments.get("dynamic_prob", None)
    if torch.is_tensor(value) and value.shape[0] == n_full:
        bound.arguments["dynamic_prob"] = value.index_select(0, ids.to(value.device))
    return target(*bound.args, **bound.kwargs)
'''
    source = source.rstrip() + wrapper + "\n"
    ast.parse(source)
    path.write_text(source)


def patch_train_preflight(path: Path) -> None:
    source = path.read_text()
    if "[Joint preflight] semantic foreground only" in source:
        return
    needle = "gaussians.init_dynamic_motion(dyn, scene.cameras_extent, cameras=scene.getTrainCameras())"
    if needle not in source:
        raise RuntimeError("init_dynamic_motion call not found in train.py")
    pos = source.find(needle)
    line_start = source.rfind("\n", 0, pos) + 1
    indent = source[line_start:pos]
    replacement = needle + (
        f"\n{indent}if bool(getattr(dyn, \"joint_single_rigid_training\", False)):"
        f"\n{indent}    _iso = gaussians.assert_joint_foreground_isolation()"
        f"\n{indent}    print("
        f"\n{indent}        \"[Joint preflight] semantic foreground only: \""
        f"\n{indent}        f\"joint={{{{_iso['num_joint_foreground']}}}}, \""
        f"\n{indent}        f\"proposal_only_background={{{{_iso['num_proposal_only_background']}}}}, \""
        f"\n{indent}        f\"full={{{{_iso['num_gaussians_full']}}}}\""
        f"\n{indent}    )"
    )
    replacement = replacement.replace("{{", "{").replace("}}", "}")
    source = source.replace(needle, replacement, 1)
    ast.parse(source)
    path.write_text(source)


def backup(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = path.with_suffix(path.suffix + f".before_semantic_joint_repair_{stamp}")
    shutil.copy2(path, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    repo = args.repo.resolve()
    package = args.package_root.resolve()
    required = [
        repo / "scene/gaussian_model.py",
        repo / "scene/joint_part_field.py",
        repo / "utils/dynamic_loss_utils.py",
        repo / "train.py",
    ]
    for p in required:
        if not p.exists():
            raise SystemExit(f"Missing {p}")

    backups = [(p, backup(p)) for p in required]
    try:
        shutil.copy2(package / "scene/joint_part_field.py", repo / "scene/joint_part_field.py")
        patch_gaussian_model(repo / "scene/gaussian_model.py")
        patch_loss_wrapper(repo / "utils/dynamic_loss_utils.py")
        patch_train_preflight(repo / "train.py")
        shutil.copy2(package / "utils/final_joint_estimation.py", repo / "utils/final_joint_estimation.py")
        (repo / "tools").mkdir(exist_ok=True)
        for name in ("verify_semantic_joint_pipeline.py", "inspect_joint_output.py"):
            shutil.copy2(package / "tools" / name, repo / "tools" / name)

        print("Semantic joint pipeline repaired successfully.")
        print("Backups:")
        for _, b in backups:
            print(f"  {b}")
        print("Run:")
        print("  python -m py_compile train.py scene/gaussian_model.py scene/joint_part_field.py utils/dynamic_loss_utils.py")
        print("  python tools/verify_semantic_joint_pipeline.py --repo .")
    except Exception:
        for original, saved in backups:
            shutil.copy2(saved, original)
        raise


if __name__ == "__main__":
    main()
