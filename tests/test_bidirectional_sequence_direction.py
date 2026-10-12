"""Sequence direction: the backward walk and its bookkeeping.

CODEX_BIDIRECTIONAL_IEKF_AND_ARTICULATED_3DGS_MERGE.md sections 3-6, steps
2-4 of Part XIII. The claims under test:

  - a backward run processes T-1..0 and a forward run 0..T-1;
  - dt stays POSITIVE -- the order is reversed, time is not negated;
  - the original frame index survives the reversal and is never overwritten;
  - the backward run is refused the initial canonical model;
  - the forward path is byte-identical to what it was.

    python tests/test_bidirectional_sequence_direction.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from debug.sequence_direction_debug import (  # noqa: E402
    check_frame_order,
    write_phase_debug,
)
from utils.sequence_direction import (  # noqa: E402
    BACKWARD,
    FORWARD,
    annotate_forward,
    canonical_endpoint,
    frame_order,
    normalise,
    original_frame_index,
    reverse_sequence,
    to_original_order,
    validate_endpoint,
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
    except Exception as exc:  # noqa: BLE001 - the message is the assertion
        if needle and needle.lower() not in str(exc).lower():
            FAILURES.append(f"{label}: message lacks {needle!r}: {exc}")
            print(f"  FAIL  {label}  message lacks {needle!r}")
        return
    FAILURES.append(f"{label}: no exception raised")
    print(f"  FAIL  {label}  no exception raised")


def fake_sequence(total=6, start=0, dt=0.1):
    """What the forward loader returns, in miniature."""
    frames = [(f"observations-{start + k}", "camera-specs", dt) for k in range(total)]
    ids = {
        "frames": [{
            "frame": k,
            "dataset_frame_index": start + k,
            "dt_used_s": dt,
            "cameras": [{"camera": "cam_000",
                         "rgb_frame_id": f"frame_{start + k:06d}.png",
                         "depth_frame_id": f"frame_{start + k:06d}.npy",
                         "depth_valid_pixels": 100 + k}],
        } for k in range(total)],
        "num_frames": total, "num_cameras": 1, "cameras": ["cam_000"],
        "sequence_start": start, "sequence_length": total,
    }
    return frames, ids


# --- 1. the direction vocabulary -----------------------------------------
def test_vocabulary() -> None:
    print("[1] direction, endpoint and index arithmetic")
    check("forward normalises", normalise("Forward") == FORWARD)
    check("backward normalises", normalise(" BACKWARD ") == BACKWARD)
    check("none defaults forward", normalise(None) == FORWARD)
    raises("an unknown direction is refused", lambda: normalise("sideways"), "one of")

    check("forward starts at the initial model", canonical_endpoint(FORWARD) == "initial")
    check("backward starts at the final model", canonical_endpoint(BACKWARD) == "final")

    # Section 3: this is the combination that produces a meaningless run.
    raises("backward off the initial model is refused",
           lambda: validate_endpoint(BACKWARD, "initial"), "final canonical model")
    raises("forward off the final model is refused",
           lambda: validate_endpoint(FORWARD, "final"), "initial canonical model")
    check("backward + final accepted", validate_endpoint(BACKWARD, "final") == "final")
    check("defaulted endpoint accepted", validate_endpoint(BACKWARD, None) == "final")

    # Section 4: original_frame = T - 1 - k
    check("k=0 maps to the last frame", original_frame_index(0, 60) == 59)
    check("k=59 maps to the first frame", original_frame_index(59, 60) == 0)
    raises("an out-of-range step is refused",
           lambda: original_frame_index(60, 60), "outside")

    check("forward order is 0..T-1", frame_order(4, FORWARD) == [0, 1, 2, 3])
    check("backward order is T-1..0", frame_order(4, BACKWARD) == [3, 2, 1, 0])
    raises("an empty sequence is refused", lambda: frame_order(0), "at least one")


# --- 2. reversing a loaded sequence --------------------------------------
def test_reverse() -> None:
    print("[2] reversing frames and provenance together")
    frames, ids = fake_sequence(total=6)
    rev_frames, rev_ids = reverse_sequence(frames, ids)

    check("frame count preserved", len(rev_frames) == 6)
    check("pixels reversed with the records",
          rev_frames[0] == frames[-1] and rev_frames[-1] == frames[0],
          f"{rev_frames[0]} vs {frames[-1]}")

    entries = rev_ids["frames"]
    check("processing index is 0..T-1",
          [e["processing_index"] for e in entries] == list(range(6)))
    check("original index counts down",
          [e["original_frame_index"] for e in entries] == [5, 4, 3, 2, 1, 0])
    check("step 0 is the last original frame", entries[0]["original_frame_index"] == 5)
    check("the rgb file follows its record",
          entries[0]["cameras"][0]["rgb_frame_id"] == "frame_000005.png")
    check("dt stays positive", all(e["dt_used_s"] > 0 for e in entries))
    check("direction recorded", rev_ids["sequence_direction"] == BACKWARD)
    check("canonical endpoint recorded", rev_ids["canonical_endpoint"] == "final")
    check("processing order published",
          rev_ids["processing_order_original_indices"] == [5, 4, 3, 2, 1, 0])

    # A non-zero --sequence-start must survive: the original index is the
    # dataset's, not a position in the window.
    frames, ids = fake_sequence(total=4, start=10)
    _, rev_ids = reverse_sequence(frames, ids)
    check("sequence_start survives the reversal",
          [e["original_frame_index"] for e in rev_ids["frames"]] == [13, 12, 11, 10],
          str([e["original_frame_index"] for e in rev_ids["frames"]]))

    # Mismatched lengths must fail rather than silently truncate.
    frames, ids = fake_sequence(total=4)
    ids["frames"] = ids["frames"][:3]
    raises("frames and ids out of step are refused",
           lambda: reverse_sequence(frames, ids), "one to one")

    # A negative dt is the mistake this design exists to avoid.
    frames, ids = fake_sequence(total=3)
    ids["frames"][1]["dt_used_s"] = -0.1
    raises("a negative dt is refused",
           lambda: reverse_sequence(frames, ids), "positive")


# --- 3. the forward path is untouched ------------------------------------
def test_forward_unchanged() -> None:
    print("[3] forward annotation adds fields and changes nothing else")
    frames, ids = fake_sequence(total=5)
    annotated = annotate_forward(ids)
    check("order unchanged",
          [e["original_frame_index"] for e in annotated["frames"]] == [0, 1, 2, 3, 4])
    check("processing index equals original index forward",
          all(e["processing_index"] == e["original_frame_index"]
              for e in annotated["frames"]))
    check("direction recorded forward", annotated["sequence_direction"] == FORWARD)
    for key in ("num_frames", "cameras", "sequence_start", "sequence_length"):
        check(f"{key} preserved", annotated[key] == ids[key])
    check("per-frame payload preserved",
          all(a["cameras"] == o["cameras"]
              for a, o in zip(annotated["frames"], ids["frames"])))

    # to_original_order is identity forward, a reversal backward.
    values = [10, 20, 30, 40]
    check("forward needs no reordering", to_original_order(values, FORWARD) == values)
    check("backward reorders to chronological",
          to_original_order(values, BACKWARD) == [40, 30, 20, 10])


# --- 4. the phase's debug output -----------------------------------------
def test_debug_artifacts() -> None:
    print("[4] frame-order debug output")
    frames, ids = fake_sequence(total=6)
    rev_frames, rev_ids = reverse_sequence(frames, ids)

    checks = check_frame_order(rev_ids)
    check("backward order verified", checks["passes"], json.dumps(checks))
    check("first processed is the last frame", checks["first_original_frame"] == 5)
    check("last processed is frame 0", checks["last_original_frame"] == 0)
    check("dt all positive", checks["all_dt_positive"])

    forward_checks = check_frame_order(annotate_forward(ids))
    check("forward order verified", forward_checks["passes"])
    check("forward starts at 0", forward_checks["first_original_frame"] == 0)

    # The checker must actually catch a broken mapping, or it is decoration.
    broken = dict(rev_ids)
    broken["frames"] = [dict(e) for e in rev_ids["frames"]]
    broken["frames"][2]["original_frame_index"] = 5     # duplicate
    bad = check_frame_order(broken)
    check("a duplicated original index is caught",
          not bad["original_indices_unique"] and not bad["passes"])

    scrambled = dict(rev_ids)
    scrambled["frames"] = [dict(e) for e in rev_ids["frames"]]
    scrambled["frames"][0]["original_frame_index"] = 1
    scrambled["frames"][1]["original_frame_index"] = 5
    bad = check_frame_order(scrambled)
    check("an out-of-order walk is caught",
          not bad["original_order_matches_direction"] and not bad["passes"])

    with tempfile.TemporaryDirectory() as tmp:
        out = write_phase_debug(Path(tmp), rev_ids, scene="unit")
        rows = list(csv.DictReader(open(out["csv"])))
        check("csv has one row per frame", len(rows) == 6)
        check("csv carries both indices",
              rows[0]["processing_index"] == "0" and rows[0]["original_frame_index"] == "5")
        page = Path(out["html"]).read_text()
        check("page names the direction", "backward" in page)
        check("page reports the verdict", "PASS" in page)
        check("page is themed for dark mode", "prefers-color-scheme" in page)
        saved = json.loads(Path(out["json"]).read_text())
        check("verdict persisted", saved["passes"] is True)
        check("endpoint persisted", saved["canonical_endpoint"] == "final")


def main() -> int:
    test_vocabulary()
    test_reverse()
    test_forward_unchanged()
    test_debug_artifacts()
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
