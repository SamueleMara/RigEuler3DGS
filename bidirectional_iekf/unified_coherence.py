"""T10 — coherent support from ACCEPTED unified smoothed trajectories.

Section 3 of CODEX_COMPLETE_MISSING_UNIFIED_BIDI_PHASE1_AFTER_9B78EFA.md.

This module is an ADAPTER and nothing else. The segmentation stays in
`estimator/motion_coherence_support.py` and `estimator/coherence_extraction.py`,
is called here in exactly the order `tools/run_motion_coherence_support.py`
calls it, and is not reimplemented, reparameterised or weakened. What the
adapter owns is the three things that are genuinely different once the
trajectory comes from a two-source `UnifiedFreeIEKS` instead of a one-source
Phase-1 bootstrap:

1.  **The input must be an accepted posterior.** Warm starts, a trajectory
    from a run that stopped at MAX_ITERS or STALLED, and above all an old
    `predicted_support.json` are all refused by name. A support file is a
    CONCLUSION of a previous run; feeding one back in and re-exporting it
    would reproduce the old answer under a new filename and look like a
    result.

2.  **The graph must be built at ONE physical time.** This is where a
    two-source population can go wrong in a way a one-source population
    cannot. `CandidatePopulation.anchor_xyz` holds the initial model at
    original frame 0 and the final model at original frame T-1, so its rows
    are NOT simultaneous. Building the kNN graph or the supernode partition on
    it compares a pose at t=0 with a pose at t=T and calls the difference
    geometry. The adapter refuses any reference array whose per-candidate
    frame index is not constant, and builds the partition from the accepted
    trajectory's own frame-0 centers, which are simultaneous by construction.

3.  **Source identity must survive the three index spaces.** candidate index
    -> supernode id -> node index and back, with `(source_endpoint,
    source_gaussian_index)` carried through to the exported core. Nothing here
    merges, dedups or corresponds rows across sources.

No ICP, no nearest neighbour, no radius, no displacement threshold, no Otsu,
no mask, no semantics, no joint type, no ground truth.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from bidirectional_iekf.provenance import ENDPOINTS

# Status/label vocabulary. The three label literals are the EXTRACTOR's own
# (estimator/coherence_extraction.py); they are re-exported rather than
# renamed, because a silent rename would make two incomparable things look
# like one. The integer contract is this module's, and is written into the
# manifest so a reader never has to guess it.
MOVING_RIGID = "MOVING_RIGID"
STATIC_COMPATIBLE = "STATIC_COMPATIBLE"
INCOHERENT = "INCOHERENT_OR_AMBIGUOUS"
NO_MOVING = "NO_COHERENT_MOVING_COMPONENT"
UNASSIGNED = "UNASSIGNED_NO_SUPERNODE"

LABEL_INDEX = {MOVING_RIGID: 0, STATIC_COMPATIBLE: 1, INCOHERENT: 2,
               UNASSIGNED: -1}
LABEL_NAME = {v: k for k, v in LABEL_INDEX.items()}


class StaleSupportRejected(Exception):
    """A previous run's conclusion was offered as this run's input."""


class MixedEndpointTimeGraph(Exception):
    """Centers from two different physical times were offered as simultaneous."""


class TrajectoryNotAccepted(Exception):
    """The IEKS did not converge, so there is no accepted posterior to read."""


# Names that only ever appear in a finished support export. Any of them in the
# object handed to this module means a conclusion is being recycled as
# evidence.
_SUPPORT_KEYS = ("predicted_support", "dynamic_gaussians", "static_gaussians",
                 "dynamic_supernodes", "static_supernodes",
                 "predicted_dynamic_fraction", "labels", "node_class",
                 "moving_nodes", "core_candidate_indices")
_SUPPORT_FILES = ("predicted_support.json", "support_manifest.json",
                  "labels.npy", "core_candidate_indices.npy")


@dataclass(frozen=True)
class UnifiedCoherenceConfig:
    """Everything the chain needs, with production's own defaults."""

    knn: int = 12
    # 0 keeps one supernode per candidate, which is the right default for a
    # small validated scene; on the real population the count is what makes
    # the graph the same shape whatever the Gaussian density.
    supernode_count: int = 0
    supernode_size: int = 0
    cv_folds: int = 5
    max_candidates: int = 48
    min_nodes: int = 3
    seed: int = 0
    # The extractor builds its kNN graph, its canonical distances and its
    # Kabsch reference at traj[0], and `edge_strain` differentiates z along
    # the stored order. Rolling the list to move the reference would silently
    # reorder time, so the reference frame IS trajectory index 0 and a
    # different choice is refused rather than approximated.
    graph_reference_frame_index: int = 0


@dataclass
class UnifiedCoherenceResult:
    status: str
    label_index: np.ndarray                 # [K] int8, LABEL_INDEX contract
    core_candidate_indices: np.ndarray      # [M] int64, sorted
    core_source_ids: np.ndarray             # [M,2] (endpoint_id, source_row)
    supernode_id: np.ndarray                # [K] int64
    source_endpoint: np.ndarray             # [K] int64, 0=initial 1=final
    source_gaussian_index: np.ndarray       # [K] int64, row within that source
    node_ids: np.ndarray                    # [n] supernode id per node index
    node_labels: list                       # [n] extractor literals
    graph_reference_original_frame: int
    counts: dict
    per_source_counts: dict
    extraction: dict = field(default_factory=dict)
    coherence: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def reject_stale_support(obj, *, what: str = "input") -> None:
    """Refuse a previous run's conclusion wherever it is offered.

    Accepts a mapping, a path or a directory. A directory is refused when it
    holds a finished support export, because pointing the unified adapter at
    `bidirectional/backward/` and re-exporting its `predicted_support.json`
    would reproduce the single-source answer and look like a unified result.
    """
    if obj is None:
        return
    if isinstance(obj, dict):
        found = [k for k in _SUPPORT_KEYS if k in obj]
        if found:
            raise StaleSupportRejected(
                f"{what} carries {found}, which only a finished support export "
                "has. The unified coherence input is an ACCEPTED IEKS "
                "trajectory; a previous run's labels are a conclusion, not "
                "evidence, and re-exporting them would reproduce that run's "
                "answer under a new filename.")
        return
    path = Path(obj)
    if path.is_dir():
        found = [f for f in _SUPPORT_FILES if (path / f).is_file()]
        if found:
            raise StaleSupportRejected(
                f"{path} already holds {found}: it is a finished support "
                "export, not a source of accepted states. Point this at the "
                "IEKS run directory (the one with unified_free/ and "
                "run_status.json) instead.")
        return
    if path.is_file() and path.name in _SUPPORT_FILES:
        raise StaleSupportRejected(
            f"{path.name} is a previous run's conclusion and cannot be this "
            "run's input.")


def assert_common_time(frame_index_per_candidate) -> int:
    """Every candidate's reference center must come from ONE physical frame."""
    frames = np.asarray(frame_index_per_candidate).reshape(-1)
    unique = np.unique(frames)
    if unique.size != 1:
        raise MixedEndpointTimeGraph(
            f"reference centers span physical frames {unique.tolist()}. A "
            "spatial graph needs simultaneous positions: comparing the "
            "source-initial pose at frame 0 with the source-final pose at "
            "frame T-1 measures the sequence's motion as if it were geometry, "
            "and every distance on the moving part is then wrong by the whole "
            "displacement. Use the accepted trajectory's centers at one "
            "frame, which are simultaneous by construction.")
    return int(unique[0])


def accepted_trajectory(result) -> tuple:
    """The accepted nominal from a finished run, or a refusal with the reason."""
    from bidirectional_iekf.ieks import CONVERGED

    reject_stale_support(result, what="the IEKS result")
    status = result.get("status")
    if status != CONVERGED:
        raise TrajectoryNotAccepted(
            f"the run ended {status!r}, not {CONVERGED!r}. MAX_ITERS, STALLED "
            "and LINE_SEARCH_FAILED leave a nominal that is not a posterior "
            "mode, and coherence computed on one would describe the "
            "optimiser's last position rather than the scene.")
    trajectory = result["trajectory"]
    if trajectory.T < 2:
        raise TrajectoryNotAccepted(
            f"{trajectory.T} frame(s): a strain measurement needs a window.")
    frame_ids = [int(f) for f in trajectory.frame_ids]
    if any(b <= a for a, b in zip(frame_ids, frame_ids[1:])):
        raise TrajectoryNotAccepted(
            f"frame ids {frame_ids} are not strictly increasing; the unified "
            "window carries ONE chronological time coordinate, and a "
            "backward run's states must be converted to it before they get "
            "here (bidirectional_iekf.free_dynamics.reverse_to_physical_state).")
    return trajectory, frame_ids


def streamed_centers(trajectory) -> list:
    """`[T]` of `[K,3]` float64 CPU centers, one frame resident at a time.

    33 million tangent scalars is the real population's size; the centers
    alone are 33 MB but the CUDA solver caches around them are not, so the
    trajectory is never stacked on the device here.
    """
    out = []
    for state in trajectory.states:
        out.append(state.mu.detach().to("cpu", torch.float64).clone())
    return out


def array_sha256(array) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(f"{contiguous.shape}:{contiguous.dtype.str}".encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------

def unified_coherence_support(result, population, *,
                              config: UnifiedCoherenceConfig | None = None,
                              reference_centers=None,
                              reference_center_frames=None,
                              prior_support=None,
                              verbose: bool = True) -> UnifiedCoherenceResult:
    """Accepted two-source trajectory -> the production coherence extractor.

    `reference_centers` exists only so a caller can be explicit about the
    common-time array; it must come with `reference_center_frames` and those
    must be constant. Omitting both uses the accepted trajectory's own frame
    `config.graph_reference_frame_index`, which is the correct default.
    """
    from estimator.coherence_extraction import extract_moving_component
    from estimator.motion_coherence_support import (propagate_to_gaussians,
                                                    run_motion_coherence,
                                                    supernode_trajectories)
    from estimator.supernodes import build_supernodes

    cfg = config or UnifiedCoherenceConfig()
    reject_stale_support(prior_support, what="prior_support")
    if prior_support is not None:
        raise StaleSupportRejected(
            "prior_support was supplied. The unified core is derived from the "
            "accepted states alone; seeding it with an earlier support set "
            "would carry that run's decisions into this one.")
    if cfg.graph_reference_frame_index != 0:
        raise ValueError(
            "graph_reference_frame_index must be 0: the extractor builds its "
            "kNN graph, canonical distances and Kabsch reference at traj[0], "
            "and edge_strain differentiates z along the stored order, so "
            "moving the reference by rolling the list would reorder time "
            "while every array kept its shape.")

    trajectory, frame_ids = accepted_trajectory(result)
    K = trajectory.N
    if population.num_candidates != K:
        raise ValueError(
            f"{population.num_candidates} candidates in the population but "
            f"{K} in the trajectory; the state and its provenance must be the "
            "same disjoint union, in the same row order.")

    positions = streamed_centers(trajectory)
    reference_frame = frame_ids[cfg.graph_reference_frame_index]
    if reference_centers is None:
        reference = positions[cfg.graph_reference_frame_index]
        reference_frame = assert_common_time(np.full(K, reference_frame))
    else:
        if reference_center_frames is None:
            raise MixedEndpointTimeGraph(
                "reference_centers was given without reference_center_frames. "
                "An array of centers carries no evidence that its rows are "
                "simultaneous, and for a two-source population they usually "
                "are not.")
        reference_frame = assert_common_time(reference_center_frames)
        reference = torch.as_tensor(np.asarray(reference_centers)).double().cpu()
        if reference.shape != (K, 3):
            raise ValueError(f"reference_centers must be [{K},3], got "
                             f"{tuple(reference.shape)}")

    # ---- the partition, on simultaneous centers --------------------------
    if cfg.supernode_size:
        part = build_supernodes(reference.float(), target_size=cfg.supernode_size,
                                seed=cfg.seed)
        how = f"target size {cfg.supernode_size}"
    elif cfg.supernode_count:
        want = min(int(cfg.supernode_count), K)
        part = build_supernodes(reference.float(), num_supernodes=want,
                                seed=cfg.seed)
        how = f"target count {want}"
    else:
        part = build_supernodes(reference.float(), num_supernodes=K,
                                seed=cfg.seed)
        how = "one node per candidate"
    sid = part.supernode_id.long().cpu()
    if verbose:
        print(f"    {int(sid.max()) + 1} supernodes ({how}) at original frame "
              f"{reference_frame}")

    # ---- production chain, in production's order -------------------------
    coherence = run_motion_coherence(positions, sid, knn=cfg.knn, verbose=verbose)
    node_ids = coherence["node_ids"]
    sn = supernode_trajectories(positions, sid)
    traj_nodes = sn["trajectories"][:, node_ids, :]
    extraction = extract_moving_component(
        traj_nodes, coherence["edge_i"], coherence["edge_j"], coherence["C"],
        coherence["num_supernodes"], min_nodes=cfg.min_nodes,
        folds=cfg.cv_folds, max_candidates=cfg.max_candidates, verbose=verbose)

    # ---- node index -> supernode id -> candidate row ---------------------
    node_labels = list(extraction["labels"])
    moving_supernodes = (node_ids[extraction["moving_nodes"]]
                         if extraction["moving_nodes"].numel()
                         else torch.zeros(0, dtype=torch.long))
    core = propagate_to_gaussians(moving_supernodes, sid)

    label_index = np.full(K, LABEL_INDEX[UNASSIGNED], dtype=np.int8)
    sid_np = sid.numpy()
    node_of_supernode = np.full(int(sid_np.max()) + 1, -1, dtype=np.int64)
    node_of_supernode[node_ids.numpy()] = np.arange(len(node_labels))
    assigned = node_of_supernode[sid_np]
    for node, literal in enumerate(node_labels):
        label_index[assigned == node] = LABEL_INDEX[literal]

    core_np = np.sort(core.numpy().astype(np.int64))
    # The core is the extractor's MOVING_RIGID set propagated to candidates;
    # the two must agree, and a mismatch would mean the three index spaces
    # came apart somewhere above.
    from_labels = np.flatnonzero(label_index == LABEL_INDEX[MOVING_RIGID])
    if not np.array_equal(core_np, from_labels):
        raise RuntimeError(
            f"{core_np.size} core candidates by propagation but "
            f"{from_labels.size} by label; the candidate/supernode/node index "
            "spaces disagree.")

    core_source_ids = np.stack([
        population.source_endpoint[core_np].astype(np.int64),
        population.source_gaussian_index[core_np].astype(np.int64)], axis=1)

    counts = {literal: int(node_labels.count(literal))
              for literal in (MOVING_RIGID, STATIC_COMPATIBLE, INCOHERENT)}
    per_source = {}
    for endpoint in ENDPOINTS:
        if endpoint not in population.anchors:
            continue
        mask = population.mask_for(endpoint)
        per_source[endpoint] = {
            "candidates": int(mask.sum()),
            "core": int(mask[core_np].sum()) if core_np.size else 0,
            **{literal: int((label_index[mask] == LABEL_INDEX[literal]).sum())
               for literal in (MOVING_RIGID, STATIC_COMPATIBLE, INCOHERENT)}}

    return UnifiedCoherenceResult(
        status=extraction["status"], label_index=label_index,
        core_candidate_indices=core_np, core_source_ids=core_source_ids,
        supernode_id=sid_np.astype(np.int64),
        source_endpoint=population.source_endpoint.astype(np.int64),
        source_gaussian_index=population.source_gaussian_index.astype(np.int64),
        node_ids=node_ids.numpy().astype(np.int64), node_labels=node_labels,
        graph_reference_original_frame=int(reference_frame),
        counts=counts, per_source_counts=per_source,
        extraction=extraction, coherence=coherence,
        manifest={
            "graph_reference_original_frame": int(reference_frame),
            "original_frame_ids": frame_ids,
            "dt_seconds": float(trajectory.dt),
            "num_frames": int(trajectory.T),
            "num_candidates": int(K),
            "supernode_partition": how,
            "num_supernodes_built": int(sid_np.max()) + 1,
            "num_nodes_evaluated": int(coherence["num_supernodes"]),
            "coherence_knn": int(cfg.knn),
            "cv_folds": int(cfg.cv_folds),
            "seed": int(cfg.seed),
            "candidate_index_sha256": population.index_hash(),
            "trajectory_centers_sha256": array_sha256(
                torch.stack(positions).numpy()),
            "sources": {e: {"model_path": a.model_path,
                            "ply_sha256": a.ply_sha256,
                            "anchor_original_frame": a.anchor_original_frame,
                            "num_gaussians": a.num_gaussians}
                        for e, a in population.anchors.items()},
        })


def write_unified_coherence(out_dir, result: UnifiedCoherenceResult, *,
                            scene: str = "", write_debug: bool = True) -> dict:
    """The opt-in export. Never writes into a historical output directory."""
    out_dir = Path(out_dir)
    reject_stale_support(out_dir, what="the output directory")
    unified = out_dir / "unified_coherence"
    unified.mkdir(parents=True, exist_ok=True)

    np.save(unified / "labels.npy", result.label_index)
    np.save(unified / "core_candidate_indices.npy", result.core_candidate_indices)
    np.save(unified / "core_source_ids.npy", result.core_source_ids)
    np.save(unified / "supernode_id.npy", result.supernode_id)

    members = {}
    for node, supernode in enumerate(result.node_ids.tolist()):
        rows = np.flatnonzero(result.supernode_id == supernode).astype(np.int64)
        members[str(supernode)] = {
            "node_index": node,
            "label": result.node_labels[node],
            "num_members": int(rows.size),
            "candidate_indices": rows.tolist(),
            # Lossless source map for every member, so a reader never has to
            # re-derive provenance from a block layout.
            "source_ids": [[int(result.source_endpoint[r]),
                            int(result.source_gaussian_index[r])]
                           for r in rows],
        }
    # One flat array per candidate, for readers that would rather not parse
    # the membership JSON. Same contract, same order, no derived quantity.
    np.savez_compressed(
        unified / "candidate_source_map.npz",
        candidate_index=np.arange(result.supernode_id.size, dtype=np.int64),
        supernode_id=result.supernode_id,
        source_endpoint=result.source_endpoint,
        source_gaussian_index=result.source_gaussian_index,
        label_index=result.label_index)
    (unified / "supernode_memberships.json").write_text(json.dumps({
        "graph_reference_original_frame": result.graph_reference_original_frame,
        "label_index_contract": {str(v): k for k, v in LABEL_INDEX.items()},
        "node_index_is_position_within_node_ids": True,
        "supernodes": members,
    }, indent=1))

    manifest = dict(result.manifest)
    manifest.update({
        "scene": scene,
        "status": result.status,
        "node_counts": result.counts,
        "per_source_counts": result.per_source_counts,
        "num_core_candidates": int(result.core_candidate_indices.size),
        "label_index_contract": {str(v): k for k, v in LABEL_INDEX.items()},
        "category_semantics": {
            MOVING_RIGID: "the selected high-confidence core, Phase 2's input",
            STATIC_COMPATIBLE: "stationary-compatible support",
            INCOHERENT: ("neither; false dynamics are allowed to stay here "
                         "rather than be forced into a body"),
            UNASSIGNED: "candidate whose supernode was dropped as empty",
        },
        "outputs": {
            "labels.npy": array_sha256(result.label_index),
            "core_candidate_indices.npy": array_sha256(result.core_candidate_indices),
            "core_source_ids.npy": array_sha256(result.core_source_ids),
            "supernode_id.npy": array_sha256(result.supernode_id),
        },
        "files": ["labels.npy", "core_candidate_indices.npy",
                  "core_source_ids.npy", "supernode_id.npy",
                  "candidate_source_map.npz", "supernode_memberships.json",
                  "support_manifest.json"],
        "note": ("Derived from ACCEPTED UnifiedFreeIEKS states at one "
                 "chronological time coordinate. No predicted_support.json "
                 "was read, no labels were copied, no correspondence between "
                 "the two sources was assumed or sought."),
    })
    (unified / "support_manifest.json").write_text(json.dumps(manifest, indent=1,
                                                              default=str))

    if write_debug:
        try:
            from debug.motion_coherence_debug import (
                write_candidate_tree, write_differential_strain_heatmap,
                write_finite_strain_heatmap, write_representative_edges)
            dbg = unified / "debug"
            dbg.mkdir(parents=True, exist_ok=True)
            res = dict(result.coherence)
            res["labels"] = torch.tensor(
                [LABEL_INDEX[literal] for literal in result.node_labels])
            res["status"] = result.status
            write_finite_strain_heatmap(dbg / "finite_strain_heatmap.html", res)
            write_differential_strain_heatmap(
                dbg / "differential_strain_heatmap.html", res)
            write_representative_edges(dbg / "representative_edges.html", res)
            write_candidate_tree(dbg / "coherence_component_tree.html",
                                 result.extraction,
                                 f"{scene} — unified coherence candidate tree")
            manifest["debug"] = str(dbg)
        except Exception as exc:                 # diagnostics are not the result
            manifest["debug_error"] = f"{type(exc).__name__}: {exc}"
        (unified / "support_manifest.json").write_text(
            json.dumps(manifest, indent=1, default=str))

    return {"out_dir": str(unified), "manifest": manifest}
