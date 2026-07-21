#!/usr/bin/env python3
"""Apply the semantic-joint isolation repair to the exact joint branch.

The script is intentionally strict. It backs up every changed file and aborts
when it cannot recognize the expected all-Gaussian implementation rather than
silently producing a partial repair.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime


STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(path: Path) -> Path:
    dst = path.with_suffix(path.suffix + f".before_semantic_isolation_{STAMP}.bak")
    shutil.copy2(path, dst)
    return dst


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}; found {count}")
    return text.replace(old, new, 1)


def find_call_end(text: str, open_pos: int) -> int:
    depth = 0
    quote = None
    escape = False
    for i in range(open_pos, len(text)):
        ch = text[i]
        if quote is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    raise RuntimeError("Unbalanced call parentheses")


def function_span(text: str, function_name: str) -> tuple[int, int]:
    match = re.search(rf"(?m)^    def {re.escape(function_name)}\s*\(", text)
    if not match:
        raise RuntimeError(f"Could not find method {function_name}")
    start = match.start()
    next_match = re.search(r"(?m)^    (?:@[^\n]+\n    )?def \w+\s*\(", text[match.end():])
    end = len(text) if next_match is None else match.end() + next_match.start()
    return start, end


def patch_gaussian_model(path: Path) -> list[str]:
    text = path.read_text()
    changes = []

    broken_line = "fg_mask = torch.ones_like(semantic_mask, dtype=torch.bool)"
    if broken_line in text:
        text = text.replace(
            broken_line,
            "fg_mask = semantic_mask.to(dtype=torch.bool).clone()",
            1,
        )
        changes.append("replaced hardcoded all-Gaussian joint mask with semantic mask")
    elif "fg_mask = semantic_mask.to(dtype=torch.bool).clone()" not in text:
        raise RuntimeError(
            "Could not locate the confirmed all-Gaussian fg_mask assignment. "
            "No source changes were written."
        )

    text = text.replace(
        "# Exact joint branch assigns every starting Gaussian. Semantic",
        "# Exact joint branch assigns every semantic-foreground Gaussian. Semantic",
    )
    text = text.replace(
        "# filtering is retained only",
        "# filtering isolates joint understanding while proposal motion remains global",
    )

    # The exact initializer must receive the semantic mask explicitly.
    marker = "initialize_joint_parts_from_static_dino("
    pos = text.find(marker)
    if pos < 0:
        raise RuntimeError("Could not find initialize_joint_parts_from_static_dino call")
    open_pos = pos + len("initialize_joint_parts_from_static_dino")
    end = find_call_end(text, open_pos)
    call = text[pos:end + 1]
    if "foreground_mask=" not in call and "foreground_ids=" not in call:
        # Insert after the DINO argument when possible, otherwise immediately
        # after the opening parenthesis.
        m = re.search(r"(?m)^(\s*)static_dino_field\s*=.*?,\s*$", call)
        if m:
            indent = m.group(1)
            insert_at = m.end()
            call = call[:insert_at] + f"\n{indent}foreground_mask=self.dynamic_foreground_mask.detach().cpu()," + call[insert_at:]
        else:
            first_newline = call.find("\n")
            if first_newline < 0:
                raise RuntimeError("Initializer call is not multiline; refusing unsafe edit")
            indent_match = re.search(r"\n(\s+)\S", call)
            indent = indent_match.group(1) if indent_match else "            "
            call = call[:first_newline + 1] + indent + "foreground_mask=self.dynamic_foreground_mask.detach().cpu(),\n" + call[first_newline + 1:]
        text = text[:pos] + call + text[end + 1:]
        changes.append("passed semantic foreground mask to exact XYZ+DINO initializer")

    # Constructor compatibility: make the foreground nature explicit.
    text = text.replace(
        "gaussian_ids=result.gaussian_ids,",
        "foreground_ids=result.foreground_ids,",
    )

    # Add a strict runtime invariant to GaussianModel.
    if "def assert_joint_foreground_isolation" not in text:
        init_method = re.search(r"(?m)^    (?:@torch\.no_grad\(\)\n    )?def initialize_joint_part_field\s*\(", text)
        if not init_method:
            raise RuntimeError("Could not find initialize_joint_part_field method")
        method = '''    @torch.no_grad()\n    def assert_joint_foreground_isolation(self):\n        """Fail immediately if joint IDs differ from the frozen semantic mask."""\n        part_field = getattr(self, "joint_part_field", None)\n        mask = getattr(self, "dynamic_foreground_mask", None)\n        if part_field is None:\n            raise RuntimeError("joint_part_field is not initialized")\n        if mask is None or not torch.is_tensor(mask):\n            raise RuntimeError("dynamic_foreground_mask is missing")\n        mask = mask.reshape(-1).bool().to(device=part_field.foreground_ids.device)\n        if mask.numel() != int(self._xyz.shape[0]):\n            raise RuntimeError(\n                f"semantic mask has {mask.numel()} entries but model has {self._xyz.shape[0]} Gaussians"\n            )\n        expected = torch.where(mask)[0]\n        actual = part_field.foreground_ids.reshape(-1).long()\n        if expected.numel() != actual.numel() or not torch.equal(expected, actual):\n            extra = actual[~torch.isin(actual, expected)] if actual.numel() else actual\n            missing = expected[~torch.isin(expected, actual)] if expected.numel() else expected\n            raise RuntimeError(\n                "Semantic joint isolation failed: "\n                f"expected {expected.numel()} foreground IDs, got {actual.numel()}; "\n                f"extra_background={extra.numel()}, missing_foreground={missing.numel()}"\n            )\n        return {\n            "num_gaussians_full": int(mask.numel()),\n            "num_joint_foreground": int(expected.numel()),\n            "num_proposal_only_background": int((~mask).sum().item()),\n        }\n\n'''
        text = text[:init_method.start()] + method + text[init_method.start():]
        changes.append("added strict semantic-joint isolation assertion")

    # Call the invariant at the end of initialize_joint_part_field.
    start, end = function_span(text, "initialize_joint_part_field")
    block = text[start:end]
    if "self.assert_joint_foreground_isolation()" not in block:
        # Insert before the method ends, after its final nonblank line.
        stripped_end = len(block.rstrip())
        addition = '''\n        isolation = self.assert_joint_foreground_isolation()\n        print(\n            "[Joint init] Semantic isolation OK: "\n            f"joint_foreground={isolation['num_joint_foreground']}, "\n            f"proposal_only_background={isolation['num_proposal_only_background']}, "\n            f"full={isolation['num_gaussians_full']}"\n        )\n'''
        block = block[:stripped_end] + addition + block[stripped_end:]
        text = text[:start] + block + text[end:]
        changes.append("asserted semantic isolation immediately after joint initialization")

    # Background must bypass static/rigid gating and retain proposal velocity.
    if "proposal-only semantic background" not in text:
        candidate_methods = ["query_joint_velocity", "query_velocity_components", "query_dynamic_velocity"]
        patched = False
        for name in candidate_methods:
            try:
                start, end = function_span(text, name)
            except RuntimeError:
                continue
            block = text[start:end]
            if "moving_velocity" not in block or "total" not in block or "residual" not in block:
                continue
            # Insert immediately before a total-velocity clamp or return.
            insert_match = re.search(r"(?m)^(\s*)(?:if\s+self\.joint_total_max_velocity|return\s+total)", block)
            if not insert_match:
                continue
            indent = insert_match.group(1)
            snippet = f'''{indent}# Semantic non-foreground is proposal-only. It is excluded from joint\n{indent}# understanding but its Eulerian velocity remains trainable/rendered.\n{indent}_joint_fg = getattr(self, "dynamic_foreground_mask", None)\n{indent}if _joint_fg is not None and torch.is_tensor(_joint_fg) and _joint_fg.numel() == total.shape[0]:\n{indent}    _joint_fg = _joint_fg.reshape(-1, 1).to(device=total.device, dtype=torch.bool)\n{indent}    total = torch.where(_joint_fg, total, residual)\n{indent}    rigid = torch.where(_joint_fg, rigid, torch.zeros_like(rigid))\n'''
            block = block[:insert_match.start()] + snippet + block[insert_match.start():]
            text = text[:start] + block + text[end:]
            changes.append(f"made semantic background proposal-only in {name}")
            patched = True
            break
        if not patched:
            raise RuntimeError(
                "Could not recognize the joint transport method. The mask and initializer "
                "repairs were prepared, but no files were written because background "
                "proposal-only transport could not be guaranteed."
            )

    path.write_text(text)
    return changes


def patch_dynamic_loss_utils(path: Path) -> list[str]:
    text = path.read_text()
    if "_joint_single_rigid_losses_semantic_unmasked" in text:
        return ["joint loss semantic wrapper already present"]

    pattern = re.compile(r"(?m)^def joint_single_rigid_losses\s*\(")
    match = pattern.search(text)
    if not match:
        raise RuntimeError("Could not find joint_single_rigid_losses")
    text = text[:match.start()] + text[match.start():].replace(
        "def joint_single_rigid_losses(",
        "def _joint_single_rigid_losses_semantic_unmasked(",
        1,
    )
    wrapper = r'''


def joint_single_rigid_losses(*args, **kwargs):
    """Foreground-only wrapper around the exact joint objective.

    The global proposal field may contain all Gaussians. Every tensor entering
    static/moving assignment, twist fitting, or rigid consistency is sliced to
    ``part_field.foreground_ids`` before the original objective is evaluated.
    """
    import inspect as _inspect

    target = _joint_single_rigid_losses_semantic_unmasked
    signature = _inspect.signature(target)
    bound = signature.bind_partial(*args, **kwargs)
    part_field = bound.arguments.get("part_field", None)
    if part_field is None or not hasattr(part_field, "foreground_ids"):
        raise RuntimeError("joint_single_rigid_losses requires a semantic-foreground JointPartField")
    ids = part_field.foreground_ids.reshape(-1).long()
    n_full = int(getattr(part_field, "num_gaussians"))
    if ids.numel() == 0:
        raise RuntimeError("semantic joint foreground is empty")

    for name in ("total_velocity", "rigid_velocity", "proposal_velocity"):
        value = bound.arguments.get(name, None)
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == n_full:
            bound.arguments[name] = value.index_select(0, ids.to(value.device))
    value = bound.arguments.get("dynamic_prob", None)
    if torch.is_tensor(value) and value.numel() == n_full:
        shape_tail = value.shape[1:]
        value = value.reshape(n_full, *shape_tail)
        bound.arguments["dynamic_prob"] = value.index_select(0, ids.to(value.device))

    return target(*bound.args, **bound.kwargs)
'''
    text = text.rstrip() + wrapper + "\n"
    path.write_text(text)
    return ["wrapped all joint losses with strict semantic-foreground slicing"]


def patch_train(path: Path) -> list[str]:
    text = path.read_text()
    changes = []

    # Add a preflight immediately after dynamic initialization if recognizable.
    if "expected semantic foreground only" not in text:
        needle = "gaussians.init_dynamic_motion(dyn, scene.cameras_extent, cameras=scene.getTrainCameras())"
        if needle in text:
            line_start = text.rfind("\n", 0, text.find(needle)) + 1
            indent = text[line_start:text.find(needle)]
            replacement = needle + (
                f"\n{indent}if bool(getattr(dyn, \"joint_single_rigid_training\", False)):"
                f"\n{indent}    _isolation = gaussians.assert_joint_foreground_isolation()"
                f"\n{indent}    print("
                f"\n{indent}        \"[Joint preflight] expected semantic foreground only: \""
                f"\n{indent}        f\"joint={{_isolation['num_joint_foreground']}}, \""
                f"\n{indent}        f\"proposal-only background={{_isolation['num_proposal_only_background']}}, \""
                f"\n{indent}        f\"full={{_isolation['num_gaussians_full']}}\""
                f"\n{indent}    )"
            )
            text = text.replace(needle, replacement, 1)
            changes.append("added post-initialization semantic preflight")
        else:
            raise RuntimeError("Could not find gaussians.init_dynamic_motion call in train.py")

    # Wire final joint estimation after post-training track export.
    if "estimate_and_write_final_joint_from_file" not in text:
        call_marker = "post_training_render_and_visualize("
        pos = text.rfind(call_marker)
        if pos < 0:
            raise RuntimeError("Could not find post_training_render_and_visualize call")
        open_pos = pos + len("post_training_render_and_visualize")
        end = find_call_end(text, open_pos)
        line_start = text.rfind("\n", 0, pos) + 1
        indent = text[line_start:pos]
        insertion = (
            f"\n{indent}if tracks_path is not None:"
            f"\n{indent}    try:"
            f"\n{indent}        from utils.final_joint_estimation import ("
            f"\n{indent}            FinalJointEstimationConfig,"
            f"\n{indent}            estimate_and_write_final_joint_from_file,"
            f"\n{indent}        )"
            f"\n{indent}        _joint_result = estimate_and_write_final_joint_from_file("
            f"\n{indent}            tracks_path,"
            f"\n{indent}            Path(scene.model_path),"
            f"\n{indent}            FinalJointEstimationConfig(),"
            f"\n{indent}        )"
            f"\n{indent}        print("
            f"\n{indent}            \"[Final joint] Wrote final_joint_estimation.html/json and \""
            f"\n{indent}            f\"model scores; selected={{_joint_result.get('joint_type', 'unknown')}}\""
            f"\n{indent}        )"
            f"\n{indent}    except Exception as exc:"
            f"\n{indent}        print(f\"[Final joint] Automatic estimation failed: {{exc}}\")"
        )
        text = text[:end + 1] + insertion + text[end + 1:]
        changes.append("wired automatic final joint estimation after track export")

    path.write_text(text)
    return changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    package = args.package_root.resolve()

    required = [
        repo / "scene/gaussian_model.py",
        repo / "scene/joint_part_field.py",
        repo / "utils/dynamic_loss_utils.py",
        repo / "train.py",
    ]
    for path in required:
        if not path.exists():
            raise SystemExit(f"Missing required repo file: {path}")

    backups = []
    try:
        for path in required:
            backups.append((path, backup(path)))

        # Replace the known broken all-Gaussian field implementation completely.
        shutil.copy2(package / "scene/joint_part_field.py", repo / "scene/joint_part_field.py")
        changes = {"scene/joint_part_field.py": ["replaced with semantic-foreground exact implementation"]}
        changes["scene/gaussian_model.py"] = patch_gaussian_model(repo / "scene/gaussian_model.py")
        changes["utils/dynamic_loss_utils.py"] = patch_dynamic_loss_utils(repo / "utils/dynamic_loss_utils.py")
        changes["train.py"] = patch_train(repo / "train.py")

        # Install estimator and verification tools.
        shutil.copy2(package / "utils/final_joint_estimation.py", repo / "utils/final_joint_estimation.py")
        (repo / "tools").mkdir(exist_ok=True)
        for name in ("verify_semantic_joint_contract.py", "inspect_joint_output.py"):
            shutil.copy2(package / "tools" / name, repo / "tools" / name)

        print("Semantic joint isolation repair applied:\n")
        for filename, items in changes.items():
            print(filename)
            for item in items:
                print(f"  - {item}")
        print("\nBackups:")
        for _, bak in backups:
            print(f"  {bak}")
        print("\nRun next:")
        print("  python -m py_compile train.py scene/gaussian_model.py scene/joint_part_field.py utils/dynamic_loss_utils.py utils/final_joint_estimation.py")
        print("  python tools/verify_semantic_joint_contract.py --repo .")
    except Exception:
        # Restore every source backup on any failure.
        for original, bak in backups:
            if bak.exists():
                shutil.copy2(bak, original)
        raise


if __name__ == "__main__":
    main()
