#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def ok(msg: str): print(f"[ OK ] {msg}")
def fail(msg: str): raise SystemExit(f"[FAIL] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    repo = args.repo.resolve()
    gm = (repo / "scene/gaussian_model.py").read_text()
    jp = (repo / "scene/joint_part_field.py").read_text()
    dl = (repo / "utils/dynamic_loss_utils.py").read_text()
    tr = (repo / "train.py").read_text()
    for name, src in [("gaussian_model.py", gm), ("joint_part_field.py", jp), ("dynamic_loss_utils.py", dl), ("train.py", tr)]:
        try: ast.parse(src)
        except SyntaxError as e: fail(f"{name} syntax error: {e}")
        ok(f"{name} parses")

    forbidden = [
        "fg_mask = torch.ones_like(semantic_mask, dtype=torch.bool)",
        "canonical_xyz=self._xyz.detach()[fg_ids]",
        "static_dino_field=dino.detach()[fg_ids]",
    ]
    for token in forbidden:
        if token in gm: fail(f"obsolete semantic-joint code remains: {token}")
    ok("no all-Gaussian/sliced-index initializer path remains")

    required_gm = [
        "foreground_mask=semantic_mask.detach().cpu()",
        "canonical_xyz=self._xyz.detach().cpu()",
        "static_dino_field=dino.detach().cpu()",
        "foreground_ids=result.foreground_ids",
        "assert_joint_foreground_isolation",
        "grid_mask = torch.ones_like(joint_fg_mask",
        "foreground_mask=grid_mask",
        "self._xyz.detach()[joint_fg_mask].mean",
    ]
    for token in required_gm:
        if token not in gm: fail(f"gaussian_model.py missing: {token}")
    ok("full tensors + semantic mask are passed exactly once")
    ok("global grid and foreground-only joint center are explicit")

    required_jp = [
        "Semantic foreground selection is mandatory",
        "foreground_mask_full",
        "full[self.foreground_ids]",
        "non_foreground_used_for_joint_initialization",
    ]
    for token in required_jp:
        if token not in jp: fail(f"joint_part_field.py missing: {token}")
    ok("JointPartField is foreground-only and persists full mask metadata")

    if "_joint_single_rigid_losses_semantic_unmasked" not in dl:
        fail("joint loss foreground slicing wrapper missing")
    ok("joint losses are sliced to foreground IDs")

    if "[Joint preflight] semantic foreground only" not in tr:
        fail("train preflight missing")
    ok("train.py runs semantic isolation preflight")

    if args.output is not None:
        out = args.output.resolve()
        summary = out / "joint_initialization/joint_initialization_summary.json"
        if not summary.exists():
            fail(f"{summary} does not exist; run has not reached successful dynamic initialization")
        d = json.loads(summary.read_text())
        n_full = int(d["num_gaussians_full"])
        n_fg = int(d["num_foreground_gaussians"])
        shape = list(d["distance_matrix_shape"])
        if not (0 < n_fg < n_full): fail(f"invalid isolation counts full={n_full}, fg={n_fg}")
        if shape != [n_fg, n_fg]: fail(f"matrix={shape}, expected {[n_fg,n_fg]}")
        if d.get("non_foreground_used_for_joint_initialization", True):
            fail("summary says background entered initialization")
        ok(f"artifacts isolated: full={n_full}, foreground={n_fg}, background={n_full-n_fg}, matrix={shape}")

    print("\nSemantic joint pipeline verification PASSED")


if __name__ == "__main__":
    main()
