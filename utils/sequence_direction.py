"""Sequence direction for the bidirectional pipeline.

CODEX_BIDIRECTIONAL_IEKF_AND_ARTICULATED_3DGS_MERGE.md sections 4-6.

A forward run starts from the initial canonical model and walks t = 0..T-1.
A backward run starts from the FINAL canonical model and walks T-1..0. The
point is observability: if the moving part is occluded early, the forward
filter has H_t ~ 0 for its state and no later measurement can repair an
estimate that has already passed. The backward run meets that part while it
is still visible.

Nothing here is a second estimator. The frames are the same files, the
cameras are the same calibration, the process model is the same code. Only
the ORDER changes, and the bookkeeping that keeps the original frame numbers
attached to it.

Two rules this module exists to enforce:

- **dt stays positive.** The process model integrates Exp(dt * xi); feeding
  it a negative dt would ask it to represent the inverse motion through code
  that was never written for it. Reversing the sequence and keeping the step
  duration positive gets the same physics through the same code.
- **The original frame index is never overwritten.** Every artifact carries
  both `processing_index` (where it sat in the reversed walk) and
  `original_frame_index` (what the dataset calls it), because every
  downstream comparison between the two directions is done in original
  chronological order.
"""

from __future__ import annotations

FORWARD = "forward"
BACKWARD = "backward"
DIRECTIONS = (FORWARD, BACKWARD)

# Which endpoint model each direction starts from. Section 3: the backward
# run must NOT load the frame-0 PLY as its canonical geometry.
CANONICAL_ENDPOINT = {FORWARD: "initial", BACKWARD: "final"}


def normalise(direction) -> str:
    value = str(direction or FORWARD).strip().lower()
    if value not in DIRECTIONS:
        raise ValueError(
            f"direction must be one of {DIRECTIONS}, got {direction!r}")
    return value


def canonical_endpoint(direction) -> str:
    """'initial' for forward, 'final' for backward."""
    return CANONICAL_ENDPOINT[normalise(direction)]


def validate_endpoint(direction, endpoint) -> str:
    """Reject the combination that silently produces a meaningless run.

    Running backward off the initial model is not a backward pass: it walks
    the frames in reverse while the geometry still sits in the frame-0 gauge,
    so the first thing the filter sees is the largest displacement in the
    sequence and the result is neither direction's answer.
    """
    direction = normalise(direction)
    endpoint = str(endpoint or canonical_endpoint(direction)).strip().lower()
    if endpoint not in ("initial", "final"):
        raise ValueError(f"canonical endpoint must be initial|final, got {endpoint!r}")
    expected = canonical_endpoint(direction)
    if endpoint != expected:
        raise ValueError(
            f"--direction {direction} requires the {expected} canonical model, "
            f"not the {endpoint} one. Section 3: the backward run starts from "
            "the final endpoint; pointing it at frame 0 walks the frames "
            "backwards against geometry that never moved, which is not a "
            "backward pass.")
    return endpoint


def original_frame_index(processing_index: int, total: int) -> int:
    """Section 4: original_frame = T - 1 - k for the backward walk."""
    if not 0 <= processing_index < total:
        raise IndexError(f"processing index {processing_index} outside [0, {total})")
    return total - 1 - processing_index


def frame_order(total: int, direction=FORWARD) -> list[int]:
    """Original frame indices in the order this direction processes them."""
    if total <= 0:
        raise ValueError(f"a sequence needs at least one frame, got {total}")
    if normalise(direction) == FORWARD:
        return list(range(total))
    return list(range(total - 1, -1, -1))


def reverse_sequence(frames, frame_ids):
    """Reverse a loaded forward sequence in place of re-reading the dataset.

    `frames` is the estimator's own input -- (per-camera observations, camera
    specs, dt) per step -- and `frame_ids` is the provenance record the
    forward loader already builds. Both are reversed together so they cannot
    drift apart, and every id gains the two indices section 4 requires.

    dt is left exactly as it was: these sequences are uniformly sampled, so
    the interval between consecutive reverse steps is the same positive
    number as between consecutive forward ones.
    """
    total = len(frames)
    if total != len(frame_ids.get("frames", [])):
        raise ValueError(
            f"{total} frames but {len(frame_ids.get('frames', []))} id records; "
            "they are reversed together and must correspond one to one")

    reversed_frames = list(frames)[::-1]
    records = list(frame_ids["frames"])[::-1]
    annotated = []
    for processing_index, record in enumerate(records):
        entry = dict(record)
        # `dataset_frame_index` is what the forward loader already recorded;
        # it is the original frame number and is never recomputed here, so a
        # non-zero --sequence-start survives the reversal.
        original = int(entry.get("dataset_frame_index", entry.get("frame", 0)))
        entry["processing_index"] = processing_index
        entry["original_frame_index"] = original
        entry["frame"] = processing_index
        dt = entry.get("dt_used_s")
        if dt is not None and float(dt) <= 0.0:
            raise ValueError(
                f"frame {original} carries dt={dt}; the backward pass reverses "
                "the ORDER and keeps dt positive, it does not negate time")
        annotated.append(entry)

    out = dict(frame_ids)
    out["frames"] = annotated
    out["sequence_direction"] = BACKWARD
    out["canonical_endpoint"] = "final"
    out["processing_order_original_indices"] = [
        e["original_frame_index"] for e in annotated]
    out["note"] = (
        "Processed last to first. dt is positive throughout: the sequence is "
        "reversed, time is not negated. original_frame_index is the dataset "
        "frame; processing_index is its position in this walk.")
    return reversed_frames, out


def annotate_forward(frame_ids):
    """Give a forward run the same two index fields, so artifacts match."""
    out = dict(frame_ids)
    annotated = []
    for processing_index, record in enumerate(out.get("frames", [])):
        entry = dict(record)
        entry["processing_index"] = processing_index
        entry["original_frame_index"] = int(
            entry.get("dataset_frame_index", entry.get("frame", processing_index)))
        annotated.append(entry)
    out["frames"] = annotated
    out["sequence_direction"] = FORWARD
    out["canonical_endpoint"] = "initial"
    out["processing_order_original_indices"] = [
        e["original_frame_index"] for e in annotated]
    return out


def to_original_order(values, direction):
    """Reorder a per-processing-step sequence into original chronological order.

    Section 11: the backward trajectory is PERSISTED in original order, so
    that every forward/backward comparison downstream indexes one array with
    one meaning. Works for a list or anything indexable along axis 0.
    """
    if normalise(direction) == FORWARD:
        return values
    try:
        return values[::-1]
    except TypeError as exc:
        raise TypeError(
            f"cannot reverse {type(values).__name__} into original order") from exc


# ---------------------------------------------------------------------------
# Updated execution order, step 2: one direction context every stage consumes.
#
# Phase 1, coherence, rigid Phase 2 and the reduction each need the same three
# facts -- which way time runs, which endpoint the geometry came from, and what
# the original frame numbers are. Letting each stage re-derive them is how two
# of them end up disagreeing about frame 0, so they are derived once here and
# passed down.
# ---------------------------------------------------------------------------
class DirectionContext:
    """Direction, endpoint and index mapping, resolved once per run."""

    def __init__(self, direction=FORWARD, total_frames: int = 0,
                 canonical_endpoint_name=None, sequence_start: int = 0):
        self.direction = normalise(direction)
        self.endpoint = validate_endpoint(self.direction, canonical_endpoint_name)
        self.total_frames = int(total_frames)
        self.sequence_start = int(sequence_start)

    @property
    def is_backward(self) -> bool:
        return self.direction == BACKWARD

    def original_index(self, processing_index: int) -> int:
        """Dataset frame number for a given step of this walk."""
        if self.direction == FORWARD:
            if not 0 <= processing_index < self.total_frames:
                raise IndexError(
                    f"processing index {processing_index} outside "
                    f"[0, {self.total_frames})")
            return self.sequence_start + processing_index
        return self.sequence_start + original_frame_index(
            processing_index, self.total_frames)

    def processing_index(self, original: int) -> int:
        """Inverse of `original_index`."""
        local = int(original) - self.sequence_start
        if not 0 <= local < self.total_frames:
            raise IndexError(
                f"original frame {original} outside the window "
                f"[{self.sequence_start}, {self.sequence_start + self.total_frames})")
        return local if self.direction == FORWARD else self.total_frames - 1 - local

    def order(self) -> list[int]:
        """Original frame numbers in the order this walk processes them."""
        return [self.sequence_start + i
                for i in frame_order(self.total_frames, self.direction)]

    def to_original(self, values):
        """Reorder per-step values into original chronological order."""
        return to_original_order(values, self.direction)

    def as_dict(self) -> dict:
        return {
            "direction": self.direction,
            "canonical_endpoint": self.endpoint,
            "total_frames": self.total_frames,
            "sequence_start": self.sequence_start,
            "processing_order_original_indices": self.order(),
        }

    def __repr__(self) -> str:
        return (f"DirectionContext({self.direction}, endpoint={self.endpoint}, "
                f"frames={self.total_frames}, start={self.sequence_start})")


def context_from_frame_ids(frame_ids) -> DirectionContext:
    """Rebuild the context from what a loaded sequence already recorded."""
    return DirectionContext(
        direction=frame_ids.get("sequence_direction", FORWARD),
        total_frames=int(frame_ids.get("num_frames", len(frame_ids.get("frames", [])))),
        canonical_endpoint_name=frame_ids.get("canonical_endpoint"),
        sequence_start=int(frame_ids.get("sequence_start", 0)),
    )
