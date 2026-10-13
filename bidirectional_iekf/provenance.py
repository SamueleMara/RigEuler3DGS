"""Two-source candidate indexing with immutable provenance.

Section 2 of CODEX_IMPLEMENT_EXACT_UNIFIED_PHASE1_IEKF_IEKS.md.

The two endpoint models are separate trainings of separate COLMAP
reconstructions. On washer they hold 21,338 and 24,582 Gaussians. They share
no rows, no ordering and no correspondence, and nothing here tries to find
one: no ICP, no nearest neighbour, no radius, no displacement test, no mask,
no ground truth. Identity is `(source_endpoint, source_gaussian_index)` and
nothing else.

A disjoint union of two trained populations is a CANDIDATE STATE SPACE. It is
not a demonstration that the union renders as one physical scene -- two
independently trained mixtures can represent the same surface twice. That
question belongs to the representation gate, and until the gate passes this
module's output must not be described as a completed scene.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INITIAL = "initial"
FINAL = "final"
ENDPOINTS = (INITIAL, FINAL)

# Endpoint -> the original frame its geometry is anchored at. The initial
# model is trained on frame 0, the final model on the last frame; a backward
# run's canonical geometry is therefore the FINAL anchor, which is why the
# anchor frame is carried per candidate rather than assumed to be zero.
ANCHOR_FRAME = {INITIAL: 0, FINAL: None}  # None -> resolved from T at build time


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(f"{contiguous.shape}:{contiguous.dtype.str}".encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceAnchor:
    """One endpoint's immutable reference data.

    `xyz` and `rotation` are REFERENCE values. Whether they are also held
    fixed during estimation is a separate modelling decision recorded in the
    manifest as `source_pose_treatment`; section 2 is explicit that
    `u_i(s_i)=0` is an enforced conditional constraint, not a free gauge, and
    must not be imposed silently.
    """

    endpoint: str
    model_path: str
    ply_path: str
    ply_sha256: str
    anchor_original_frame: int
    xyz: np.ndarray                      # [N,3] float64, never modified
    num_gaussians: int

    def __post_init__(self) -> None:
        if self.endpoint not in ENDPOINTS:
            raise ValueError(f"endpoint must be one of {ENDPOINTS}, got {self.endpoint!r}")
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must be [N,3], got {self.xyz.shape}")
        if self.xyz.shape[0] != self.num_gaussians:
            raise ValueError(
                f"{self.endpoint}: {self.xyz.shape[0]} rows but num_gaussians="
                f"{self.num_gaussians}")


@dataclass(frozen=True)
class CandidatePopulation:
    """The disjoint union, with every row traceable to its source.

    `source_endpoint[k]` and `source_gaussian_index[k]` identify candidate k
    exactly. The ordering is initial-block then final-block, which is an
    implementation choice recorded in the manifest -- downstream code must
    read provenance, never assume a block layout.
    """

    source_endpoint: np.ndarray          # [K] of 0 (initial) / 1 (final)
    source_gaussian_index: np.ndarray    # [K] row within that source
    anchor_xyz: np.ndarray               # [K,3] source anchor positions
    anchor_original_frame: np.ndarray    # [K] frame each anchor refers to
    anchors: dict

    @property
    def num_candidates(self) -> int:
        return int(self.source_endpoint.shape[0])

    def mask_for(self, endpoint: str) -> np.ndarray:
        if endpoint not in ENDPOINTS:
            raise ValueError(f"unknown endpoint {endpoint!r}")
        return self.source_endpoint == ENDPOINTS.index(endpoint)

    def rows_for(self, endpoint: str) -> np.ndarray:
        """Candidate indices belonging to one source, in source-row order."""
        where = np.flatnonzero(self.mask_for(endpoint))
        order = np.argsort(self.source_gaussian_index[where], kind="stable")
        return where[order]

    def index_hash(self) -> str:
        """Identity of the whole population, for manifest validation."""
        return _sha256_array(np.stack([
            self.source_endpoint.astype(np.int64),
            self.source_gaussian_index.astype(np.int64)], axis=1))


def build_candidate_population(anchors) -> CandidatePopulation:
    """Disjoint union of source anchors. No correspondence is sought."""
    anchors = list(anchors)
    seen = [a.endpoint for a in anchors]
    if len(set(seen)) != len(seen):
        raise ValueError(f"each endpoint may appear once, got {seen}")

    endpoint_ids, rows, xyz, frames = [], [], [], []
    for anchor in sorted(anchors, key=lambda a: ENDPOINTS.index(a.endpoint)):
        n = anchor.num_gaussians
        endpoint_ids.append(np.full(n, ENDPOINTS.index(anchor.endpoint), dtype=np.int64))
        rows.append(np.arange(n, dtype=np.int64))
        xyz.append(np.asarray(anchor.xyz, dtype=np.float64))
        frames.append(np.full(n, anchor.anchor_original_frame, dtype=np.int64))

    return CandidatePopulation(
        source_endpoint=np.concatenate(endpoint_ids),
        source_gaussian_index=np.concatenate(rows),
        anchor_xyz=np.concatenate(xyz, axis=0),
        anchor_original_frame=np.concatenate(frames),
        anchors={a.endpoint: a for a in anchors},
    )


def load_source_anchor(endpoint: str, model_dir, *, iteration: int = 30000,
                       anchor_original_frame: int) -> SourceAnchor:
    """Read one trained endpoint PLY. Attributes are copied, never altered."""
    from plyfile import PlyData

    model_dir = Path(model_dir)
    ply = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.is_file():
        raise FileNotFoundError(f"no trained PLY at {ply}")
    data = PlyData.read(str(ply))["vertex"]
    xyz = np.stack([np.asarray(data["x"]), np.asarray(data["y"]),
                    np.asarray(data["z"])], axis=1).astype(np.float64)
    return SourceAnchor(
        endpoint=endpoint, model_path=str(model_dir), ply_path=str(ply),
        ply_sha256=_sha256_file(ply),
        anchor_original_frame=int(anchor_original_frame),
        xyz=xyz, num_gaussians=int(xyz.shape[0]))


def candidate_manifest(population: CandidatePopulation, *, total_frames: int,
                       source_pose_treatment: str,
                       representation_gate: str = "NOT_RUN") -> dict:
    """What a later stage must validate before trusting this population."""
    if source_pose_treatment not in ("conditional_fixed_anchor", "optimized_source_pose"):
        raise ValueError(
            "source_pose_treatment must say which model is in force: "
            "'conditional_fixed_anchor' enforces u_i(s_i)=0 as a constraint "
            "and makes every uncertainty conditional on the source geometry; "
            "'optimized_source_pose' keeps the anchor as reference data and "
            "lets the source-frame pose move, in which case u(s) is NOT zero. "
            f"got {source_pose_treatment!r}")
    return {
        "num_candidates": population.num_candidates,
        "per_endpoint": {e: int(population.mask_for(e).sum()) for e in ENDPOINTS
                         if e in population.anchors},
        "candidate_index_sha256": population.index_hash(),
        "sources": {
            e: {"model_path": a.model_path, "ply_path": a.ply_path,
                "ply_sha256": a.ply_sha256,
                "anchor_original_frame": a.anchor_original_frame,
                "num_gaussians": a.num_gaussians}
            for e, a in population.anchors.items()},
        "total_frames": int(total_frames),
        "identity": "(source_endpoint, source_gaussian_index)",
        "correspondence_assumed": False,
        "source_pose_treatment": source_pose_treatment,
        "representation_gate": representation_gate,
        "note": ("A disjoint union of two independently trained mixtures is a "
                 "candidate state space, not a demonstrated single scene. "
                 "Until representation_gate is PASSED this population must "
                 "not be described as a completed scene."),
    }


def write_population(out_dir, population: CandidatePopulation, manifest: dict) -> dict:
    """Atomic-ish write of the arrays section 11 requires."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "source_endpoint.npy", population.source_endpoint)
    np.save(out_dir / "source_gaussian_index.npy", population.source_gaussian_index)
    np.save(out_dir / "anchor_original_frame.npy", population.anchor_original_frame)
    tmp = out_dir / "candidate_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(out_dir / "candidate_manifest.json")
    return manifest
