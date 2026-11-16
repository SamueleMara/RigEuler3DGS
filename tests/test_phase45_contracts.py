"""Gate 0/2 contracts: Phase-3 ingestion accepts real upstream runs and refuses every mismatch.

Uses real result trees (read only): the v6 red_fridge (forward, revolute), and the v5 synthetic blade_103706
(backward, prismatic) and USB_100109 (backward, revolute). Skipped when the result disk is absent.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery.contracts import ContractError, load_upstream  # noqa: E402

RES = Path("/media/samuelemara/Extreme SSD/Samuele/results_11_10_26")
REAL = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real")
SYN = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/"
           "partnet_subset_fixed_render_linear")


def run_paths(kind, scene, arm=None):
    if kind == "real":
        name = f"final_{scene}_episode_000000"
        root = RES / "ours_v6_two_cams" / name / arm / "pipeline" / name
        src = REAL / name
    else:
        root = RES / "ours_v5_canonical_synthetic" / scene / "pipeline" / scene
        src = SYN / scene
    bd = root / "bidirectional"
    d = "forward" if (bd / "forward_joint").is_dir() else "backward"
    end = "initial" if d == "forward" else "final"
    return dict(source=src, model=root / "static_bidi" / end, phase2_dir=bd / f"{d}_rigid",
                phase3_dir=bd / f"{d}_joint"), d


CASES = [("real", "red_fridge", "majority", "revolute"), ("synthetic", "blade_103706", None, "prismatic"),
         ("synthetic", "USB_100109", None, "revolute")]


def _available():
    return RES.is_dir() and REAL.is_dir()


def test_accepts_real_upstream_runs():
    if not _available():
        print("SKIP: result disk absent")
        return
    for kind, scene, arm, jt in CASES:
        p, d = run_paths(kind, scene, arm)
        up = load_upstream(**p, joint_type=jt, direction="auto")
        assert up.direction == d
        j = up.joint
        assert j.q[0] == 0.0 and j.anchor_frame == j.frames[0]
        steps = np.diff(j.frames)
        assert np.all(steps == (1 if d == "forward" else -1))
        # backward: processing order runs from the last original frame down; q is indexed by ORIGINAL frame
        if d == "backward":
            assert j.anchor_frame == max(j.frames)
            f_mid = j.frames[len(j.frames) // 2]
            assert j.q_of(f_mid) == j.q[len(j.frames) // 2]
            fr, q = j.chronological()
            assert fr[0] == min(j.frames) and q[-1] == 0.0
        Q, p_ = j.transform(j.anchor_frame)
        assert np.array_equal(Q, np.eye(3)) and np.array_equal(p_, np.zeros(3))
        assert up.members.max() < up.num_gaussians
        assert set(up.hashes) >= {"ply", "support", "geometry", "trajectory", "phase3_summary"}


def _expect_refusal(fn, why):
    try:
        fn()
    except ContractError:
        return
    raise AssertionError(f"accepted: {why}")


def test_refuses_mismatches():
    if not _available():
        print("SKIP: result disk absent")
        return
    p, d = run_paths("real", "red_fridge", "majority")
    other = "backward" if d == "forward" else "forward"
    _expect_refusal(lambda: load_upstream(**p, joint_type="revolute", direction=other), "opposite direction")
    _expect_refusal(lambda: load_upstream(**p, joint_type="screw"), "unknown joint type")
    _expect_refusal(lambda: load_upstream(**dict(p, model=p["model"].parent / "final"), joint_type="revolute"),
                    "opposite endpoint model")
    _expect_refusal(lambda: load_upstream(**p, joint_type="revolute", cameras="cam_000,cam_001,cam_002"),
                    "camera set differing from Phase 3")
    _expect_refusal(lambda: load_upstream(**p, joint_type="revolute", frame_window="0:288"),
                    "frame window differing from Phase 3")
    p5, _ = run_paths("real", "red_fridge", "C")
    _expect_refusal(lambda: load_upstream(**dict(p, model=p5["model"]), joint_type="revolute"),
                    "endpoint model of another run")
    # a tampered copy: anchor q != 0, non-unit axis, support row out of range
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        p3 = t / p["phase3_dir"].name
        shutil.copytree(p["phase3_dir"], p3, ignore=shutil.ignore_patterns("*.npz", "*.npy"))
        tr = p3 / "revolute" / "trajectory.csv"
        lines = tr.read_text().splitlines()
        cells = lines[1].split(",")
        cells[2] = "1e-9"
        tr.write_text("\n".join([lines[0], ",".join(cells)] + lines[2:]) + "\n")
        _expect_refusal(lambda: load_upstream(**dict(p, phase3_dir=p3), joint_type="revolute"), "anchor q != 0")
        shutil.copy(p["phase3_dir"] / "revolute" / "trajectory.csv", tr)
        g = json.loads((p3 / "revolute" / "geometry.json").read_text())
        g["geometry_fixed"]["axis"] = [1.0, 1.0, 0.0]
        (p3 / "revolute" / "geometry.json").write_text(json.dumps(g))
        _expect_refusal(lambda: load_upstream(**dict(p, phase3_dir=p3), joint_type="revolute"), "non-unit axis")
        p2 = t / p["phase2_dir"].name
        (p2 / "support_coherence").mkdir(parents=True)
        s = json.loads((p["phase2_dir"] / "support_coherence" / "support.json").read_text())
        s["members"] = s["members"] + [10 ** 7]
        (p2 / "support_coherence" / "support.json").write_text(json.dumps(s))
        _expect_refusal(lambda: load_upstream(**dict(p, phase2_dir=p2), joint_type="revolute"),
                        "support row beyond the PLY")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
