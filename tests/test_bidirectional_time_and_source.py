"""Chronological twists, direction context, and read-only source staging.

CODEX_BIDIRECTIONAL_IEKF_AND_ARTICULATED_3DGS_MERGE.md sections E and S,
steps 2, 3 and 6 of the updated execution order.

Section E names four tests outright, and they are the point of this file:

    constant positive translation forward
    constant positive translation processed backward
    restored chronological pose increments have the same sign
    restored chronological twist has the same sign as forward

and the same for a constant positive rotation. A backward run that merely
reverses its velocity array fails these by a sign, silently, and every
downstream fusion then averages two opposite answers.

    python tests/test_bidirectional_time_and_source.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.bidirectional_time import (  # noqa: E402
    chronological_twists,
    relative_increments,
    restore_chronological,
    twist_sign_agreement,
)
from utils.readonly_source import (  # noqa: E402
    images_argument,
    source_untouched,
    stage_colmap,
)
from utils.sequence_direction import (  # noqa: E402
    BACKWARD,
    FORWARD,
    DirectionContext,
    context_from_frame_ids,
)

CHECKS = 0
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")


def raises(label: str, fn, needle: str = "") -> None:
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        if needle and needle.lower() not in str(exc).lower():
            FAILURES.append(f"{label}: message lacks {needle!r}: {exc}")
            print(f"  FAIL  {label}  message lacks {needle!r}")
        return
    FAILURES.append(f"{label}: no exception raised")
    print(f"  FAIL  {label}  no exception raised")


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def constant_translation(total=8, step=(0.05, 0.0, 0.0)):
    """T_t = pure translation, t * step. Chronological order."""
    poses = np.tile(np.eye(4), (total, 1, 1))
    for t in range(total):
        poses[t, :3, 3] = np.array(step) * t
    return torch.from_numpy(poses)


def constant_rotation(total=8, step=0.05):
    """T_t = rotation about +z by t * step, about the origin."""
    poses = np.tile(np.eye(4), (total, 1, 1))
    for t in range(total):
        poses[t, :3, :3] = rotation_z(step * t)
    return torch.from_numpy(poses)


# --- 1. section E: the four mandated sign tests, translation --------------
def test_translation_signs() -> None:
    print("[1] constant positive translation, forward vs processed backward")
    dt = 0.1
    forward_poses = constant_translation(total=8, step=(0.05, 0.0, 0.0))

    # The backward filter would store its poses in PROCESSING order: last
    # frame first. Restoring must put them back chronologically.
    backward_processing_order = forward_poses.flip(0)
    restored = restore_chronological(backward_processing_order, BACKWARD)
    check("restoring returns the chronological poses",
          torch.allclose(restored, forward_poses),
          f"max dev {float((restored - forward_poses).abs().max()):.2e}")

    eta_f, _, _ = relative_increments(forward_poses)
    eta_b, _, _ = relative_increments(restored)
    check("restored pose increments have the same sign",
          bool((eta_f[:, 0] > 0).all() and (eta_b[:, 0] > 0).all()),
          f"forward {eta_f[:, 0].tolist()[:3]}, restored {eta_b[:, 0].tolist()[:3]}")
    check("restored increments match forward exactly",
          torch.allclose(eta_f, eta_b, atol=1e-12))

    xi_f = chronological_twists(forward_poses, dt)
    xi_b = chronological_twists(restored, dt)
    check("chronological twist is positive forward", bool((xi_f[:, 0] > 0).all()))
    check("chronological twist is positive after restore", bool((xi_b[:, 0] > 0).all()))
    check("twist magnitude is step/dt",
          abs(float(xi_f[0, 0]) - 0.05 / dt) < 1e-9, f"{float(xi_f[0, 0]):.6f}")

    agree = twist_sign_agreement(xi_f, xi_b)
    check("translation signs agree", agree["translation_same_sign_fraction"] == 1.0,
          json.dumps(agree))

    # The failure this guards: reversing the stored twist array instead of
    # recomputing it from restored poses flips the sign.
    naive = xi_f.flip(0) * -1.0
    bad = twist_sign_agreement(xi_f, naive)
    check("a sign-flipped twist IS detected",
          bad["translation_same_sign_fraction"] == 0.0, json.dumps(bad))


# --- 2. section E: the same for rotation ----------------------------------
def test_rotation_signs() -> None:
    print("[2] constant positive rotation, forward vs processed backward")
    dt = 0.1
    forward_poses = constant_rotation(total=8, step=0.05)
    restored = restore_chronological(forward_poses.flip(0), BACKWARD)
    check("rotation restored chronologically",
          torch.allclose(restored, forward_poses, atol=1e-12))

    eta_f, _, _ = relative_increments(forward_poses)
    eta_b, _, _ = relative_increments(restored)
    check("restored rotation increments have the same sign",
          bool((eta_f[:, 5] > 0).all() and (eta_b[:, 5] > 0).all()),
          f"forward {eta_f[:, 5].tolist()[:3]}")
    check("rotation increment is the step angle",
          abs(float(eta_f[0, 5]) - 0.05) < 1e-9, f"{float(eta_f[0, 5]):.6f}")

    xi_f = chronological_twists(forward_poses, dt)
    xi_b = chronological_twists(restored, dt)
    agree = twist_sign_agreement(xi_f, xi_b)
    check("rotation signs agree", agree["rotation_same_sign_fraction"] == 1.0,
          json.dumps(agree))
    check("rotation rate is angle/dt",
          abs(float(xi_f[0, 5]) - 0.05 / dt) < 1e-9, f"{float(xi_f[0, 5]):.6f}")

    naive = xi_f.flip(0) * -1.0
    bad = twist_sign_agreement(xi_f, naive)
    check("a sign-flipped rotation IS detected",
          bad["rotation_same_sign_fraction"] == 0.0, json.dumps(bad))


# --- 3. twist reconstruction mechanics ------------------------------------
def test_twist_mechanics() -> None:
    print("[3] increments, padding and refusals")
    poses = constant_translation(total=5)
    eta, dR, dp = relative_increments(poses)
    check("one increment fewer than poses", eta.shape == (4, 6), str(tuple(eta.shape)))
    check("increment rotation is identity for pure translation",
          torch.allclose(dR, torch.eye(3, dtype=torch.float64).expand(4, 3, 3), atol=1e-12))

    xi = chronological_twists(poses, 0.1)
    check("twist array matches pose count", xi.shape == (5, 6), str(tuple(xi.shape)))
    check("last entry repeats the previous", torch.allclose(xi[-1], xi[-2]))

    raises("a non-positive dt is refused",
           lambda: chronological_twists(poses, 0.0), "positive")
    raises("a negative dt is refused",
           lambda: chronological_twists(poses, -0.1), "positive")
    raises("a single pose cannot form an increment",
           lambda: relative_increments(poses[:1]), "at least two")

    # (R, p) pairs are accepted as well as [T,4,4].
    R, p = poses[:, :3, :3], poses[:, :3, 3]
    eta_pair, _, _ = relative_increments((R, p))
    check("accepts an (R, p) pair", torch.allclose(eta_pair, eta, atol=1e-12))
    raises("a malformed pose array is refused",
           lambda: relative_increments(torch.zeros(4, 3)), "expected")

    check("forward restore is a no-op",
          torch.allclose(restore_chronological(poses, FORWARD), poses))


# --- 4. the shared direction context --------------------------------------
def test_direction_context() -> None:
    print("[4] one direction context for every stage")
    fwd = DirectionContext(FORWARD, total_frames=60)
    bwd = DirectionContext(BACKWARD, total_frames=60)

    check("forward step 0 is frame 0", fwd.original_index(0) == 0)
    check("forward step 59 is frame 59", fwd.original_index(59) == 59)
    check("backward step 0 is frame 59", bwd.original_index(0) == 59)
    check("backward step 59 is frame 0", bwd.original_index(59) == 0)
    check("backward is flagged", bwd.is_backward and not fwd.is_backward)

    for ctx in (fwd, bwd):
        roundtrip = all(ctx.processing_index(ctx.original_index(k)) == k
                        for k in range(ctx.total_frames))
        check(f"{ctx.direction} index mapping round-trips", roundtrip)

    check("backward order is 59..0", bwd.order()[:3] == [59, 58, 57])
    check("forward order is 0..59", fwd.order()[:3] == [0, 1, 2])

    # A window that does not start at zero keeps dataset numbering.
    win = DirectionContext(BACKWARD, total_frames=4, sequence_start=10)
    check("windowed backward starts at the window's last frame",
          win.original_index(0) == 13, str(win.original_index(0)))
    check("windowed order is 13..10", win.order() == [13, 12, 11, 10])
    raises("an out-of-window frame is refused",
           lambda: win.processing_index(99), "outside")

    check("context serialises", DirectionContext(BACKWARD, 4).as_dict()["canonical_endpoint"] == "final")
    rebuilt = context_from_frame_ids({
        "sequence_direction": BACKWARD, "num_frames": 6,
        "canonical_endpoint": "final", "sequence_start": 0, "frames": []})
    check("context rebuilds from frame ids",
          rebuilt.is_backward and rebuilt.total_frames == 6)
    raises("backward with the initial endpoint is refused",
           lambda: DirectionContext(BACKWARD, 4, canonical_endpoint_name="initial"),
           "final canonical model")


# --- 5. section S: the dataset is never written to ------------------------
def test_readonly_source() -> None:
    print("[5] staging a COLMAP model without touching the source")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scene = root / "scene"
        sparse = scene / "colmap_final" / "sparse" / "0"
        sparse.mkdir(parents=True)
        (sparse / "cameras.bin").write_bytes(b"cameras")
        (sparse / "images.bin").write_bytes(b"images")
        (sparse / "points3D.bin").write_bytes(b"points")
        (scene / "rgb" / "cam_000").mkdir(parents=True)
        (scene / "rgb" / "cam_000" / "frame_000059.png").write_bytes(b"png")

        before = source_untouched(scene / "colmap_final")["listing"]
        manifest = stage_colmap(scene / "colmap_final", root / "cache", name="final")
        staged = Path(manifest["staged_colmap"])

        check("staged model has sparse/0", (staged / "sparse" / "0").is_dir())
        check("cameras copied", (staged / "sparse" / "0" / "cameras.bin").is_file())
        check("points copied", (staged / "sparse" / "0" / "points3D.bin").is_file())
        check("manifest records the source",
              manifest["source_colmap"].endswith("colmap_final"))
        check("manifest written beside the staged model",
              (staged / "stage_manifest.json").is_file())

        # The loader would write points3D.ply beside the model it opened.
        (staged / "sparse" / "0" / "points3D.ply").write_bytes(b"derived")
        after = source_untouched(scene / "colmap_final", before)
        check("the source gained nothing", after["untouched"], json.dumps(after))
        check("the derived file landed in the cache",
              (staged / "sparse" / "0" / "points3D.ply").is_file())

        # And the check must notice when something IS added to the source.
        (sparse / "points3D.ply").write_bytes(b"leaked")
        leaked = source_untouched(scene / "colmap_final", before)
        check("a write into the source IS detected",
              not leaked["untouched"] and "points3D.ply" in leaked["added"],
              json.dumps(leaked))

        images = images_argument(staged, scene, "rgb")
        resolved = (staged / images).resolve()
        check("--images reaches the real frames from the cache",
              resolved == (scene / "rgb").resolve(), f"{images} -> {resolved}")

        raises("a directory with no sparse/0 is refused",
               lambda: stage_colmap(scene / "rgb", root / "cache2", name="x"),
               "no sparse/0")
        empty = root / "empty" / "sparse" / "0"
        empty.mkdir(parents=True)
        raises("a model with no cameras is refused",
               lambda: stage_colmap(root / "empty", root / "cache3", name="y"),
               "cameras")


def main() -> int:
    test_translation_signs()
    test_rotation_signs()
    test_twist_mechanics()
    test_direction_context()
    test_readonly_source()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
