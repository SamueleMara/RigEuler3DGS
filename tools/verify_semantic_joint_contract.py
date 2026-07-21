#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[ OK ] {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify semantic isolation in source and optional run output")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()

    gm = (repo / "scene/gaussian_model.py").read_text()
    jp = (repo / "scene/joint_part_field.py").read_text()
    dl = (repo / "utils/dynamic_loss_utils.py").read_text()
    train = (repo / "train.py").read_text()

    if "fg_mask = torch.ones_like(semantic_mask, dtype=torch.bool)" in gm:
        fail("gaussian_model.py still hardcodes all Gaussians as joint foreground")
    if "fg_mask = semantic_mask.to(dtype=torch.bool).clone()" not in gm:
        fail("gaussian_model.py does not explicitly copy semantic_mask into fg_mask")
    ok("joint foreground is created from semantic_mask")

    for token, label in [
        ("assert_joint_foreground_isolation", "runtime isolation assertion"),
        ("foreground_mask=self.dynamic_foreground_mask", "semantic mask passed to exact initializer"),
        ("Semantic non-foreground is proposal-only", "background proposal-only transport"),
    ]:
        if token not in gm:
            fail(f"missing {label} in gaussian_model.py")
        ok(label)

    for token, label in [
        ("foreground_mask_full", "full-scene semantic mask persisted by JointPartField"),
        ("Semantic foreground selection is mandatory", "all-Gaussian initializer fallback rejected"),
        ("full[self.foreground_ids]", "foreground assignment mapped into full scene"),
    ]:
        if token not in jp:
            fail(f"missing {label} in joint_part_field.py")
        ok(label)

    if "_joint_single_rigid_losses_semantic_unmasked" not in dl:
        fail("joint loss foreground-slicing wrapper is missing")
    ok("joint losses are sliced to JointPartField.foreground_ids")

    if "expected semantic foreground only" not in train:
        fail("training preflight assertion is missing")
    ok("training preflight prints expected semantic foreground only")

    if "estimate_and_write_final_joint_from_file" not in train:
        fail("automatic final joint estimation is not wired into train.py")
    ok("automatic final joint estimation is wired")

    if args.output is None:
        print("\nSource contract passed. Add --output output/RUN_NAME after initialization to validate artifacts.")
        return

    out = args.output.resolve()
    summary_path = out / "joint_initialization/joint_initialization_summary.json"
    if not summary_path.exists():
        fail(
            f"{summary_path} does not exist. This is expected before the new run reaches dynamic initialization; "
            "it cannot be used to validate a folder that has not been created yet."
        )
    data = json.loads(summary_path.read_text())
    n_full = int(data.get("num_gaussians_full", -1))
    n_fg = int(data.get("num_foreground_gaussians", data.get("num_gaussians", -1)))
    shape = data.get("distance_matrix_shape", [])
    if n_full <= 0 or n_fg <= 0:
        fail(f"invalid counts in {summary_path}: full={n_full}, foreground={n_fg}")
    if shape != [n_fg, n_fg]:
        fail(f"distance matrix must be N_fg x N_fg; got {shape}, N_fg={n_fg}")
    if n_fg > n_full:
        fail(f"foreground count {n_fg} exceeds full count {n_full}")
    if data.get("non_foreground_used_for_joint_initialization", True):
        fail("summary does not confirm that non-foreground was excluded")
    ok(f"artifact isolation: full={n_full}, foreground={n_fg}, background={n_full-n_fg}, matrix={shape}")


if __name__ == "__main__":
    main()
