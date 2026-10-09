"""Stage-A probe: does relational graph coupling beat XYZ endpoint coupling?

Compares three ways of deciding which frame-T region a frame-0 Gaussian
corresponds to:

    A0  current raw XYZ coupling      -- nearest target in absolute position
    A1  descriptor-only coupling      -- nearest in rigid-invariant descriptor
    A2  descriptor + sparse graph     -- A1, refined by relational consistency

No flow matching is trained here and no labels are produced.  Per the spec
the new coupling must not reach FM until this probe validates it, so this
tool exists to decide whether it has.

Ground truth is a firewall: it is read only after all three couplings exist,
and only to report how each one behaves on the moving versus static parts.

Run:
    PYTHONPATH=. python tools/run_relational_graph_probe.py \
        --frame0-model-dir <frame0_static> --frameT-model-dir <frame_t_static> \
        --iteration 14000 --output-dir <out> --root-scene <scene> --gt-debug
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.relational_coupling import (  # noqa: E402
    SparseRelationalCoupling,
    build_relational_coupling,
    compute_sparse_relational_cost,
    compute_transport_coverage,
    initialize_soft_partial_transport,
)
from utils.relational_graph import build_relational_graph  # noqa: E402


def _json_safe(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu()) if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def _find_model_ply(root: str | Path, iteration: int | None) -> Path | None:
    path = Path(root).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".ply":
        return path
    candidates = sorted(path.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        candidates = sorted(path.glob("**/point_cloud.ply"))
    if not candidates:
        return None
    if iteration is not None:
        exact = [p for p in candidates if f"iteration_{int(iteration)}" in str(p)]
        if exact:
            return exact[0]
    return candidates[-1]


def _load_cloud(model_dir: str, iteration: int, sh_degree: int = 3):
    from scene.gaussian_model import GaussianModel

    ply = _find_model_ply(model_dir, iteration)
    if ply is None:
        raise SystemExit(f"No point_cloud.ply found under {model_dir}")
    model = GaussianModel(int(sh_degree), optimizer_type="default")
    model.load_ply(str(ply))
    return model.get_canonical_xyz.detach().float(), model.get_scaling.detach().float(), ply


def build_xyz_coupling(
    source_graph,
    target_graph,
    *,
    max_candidates: int,
    column_capacity: float,
    chunk_size: int = 512,
) -> SparseRelationalCoupling:
    """A0: the current behaviour -- nearest target in absolute XYZ.

    Kept as the baseline precisely because it is what the relational coupling
    is meant to replace: it can only find a target that has not moved far.
    """
    source_xyz = source_graph.xyz
    target_xyz = target_graph.xyz.to(source_xyz)
    costs: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    k = min(int(max_candidates), int(target_xyz.shape[0]))
    for start in range(0, int(source_xyz.shape[0]), max(1, int(chunk_size))):
        d2 = torch.cdist(source_xyz[start:start + int(chunk_size)], target_xyz).square()
        value, index = torch.topk(d2, k=k, dim=1, largest=False, sorted=True)
        costs.append(value)
        indices.append(index)
    unary = torch.cat(costs, 0)
    idx = torch.cat(indices, 0)
    # Normalised by the local scale so the cost is comparable with descriptor space.
    unary = unary / source_graph.effective_scale[:, None].square().clamp_min(1.0e-12)
    valid = torch.ones_like(idx, dtype=torch.bool)
    num_targets = int(target_xyz.shape[0])
    mass, unmatched = initialize_soft_partial_transport(
        unary, valid, num_targets=num_targets, target_idx=idx, column_capacity=float(column_capacity)
    )
    col_mass = torch.zeros(num_targets, device=mass.device, dtype=mass.dtype)
    col_mass.scatter_add_(0, idx[valid].reshape(-1), mass[valid].reshape(-1))
    return SparseRelationalCoupling(
        src_idx=torch.arange(int(source_xyz.shape[0]), device=idx.device)[:, None].expand_as(idx),
        tgt_idx=idx,
        unary_cost=unary,
        relational_cost=compute_sparse_relational_cost(source_graph, target_graph, idx, valid, mass=mass),
        total_cost=unary,
        mass=mass,
        valid=valid,
        unmatched_src=unmatched,
        row_mass=mass.sum(dim=1),
        col_mass=col_mass,
        num_targets=num_targets,
    )


def coupling_displacement(coupling: SparseRelationalCoupling, target_xyz: torch.Tensor, source_xyz: torch.Tensor) -> torch.Tensor:
    """Expected displacement implied by a coupling, ``[N, 3]``."""
    return coupling.expected_target(target_xyz) - source_xyz


def score_coupling_against_gt(
    source_xyz: torch.Tensor,
    coupling: SparseRelationalCoupling,
    target_xyz: torch.Tensor,
    root_scene: Path,
    effective_scale: torch.Tensor,
) -> dict[str, Any] | None:
    """Post-hoc only. Reports how the coupling behaves on moving vs static GT.

    This stage produces no labels, so there is no classification score to
    give.  What matters is whether the coupling assigns the moving part a
    displacement at all -- an XYZ coupling systematically under-reports it,
    because the nearest target to a moved Gaussian is whatever stayed put.
    """
    parts_ply = root_scene / "pointcloud" / "points3D_parts.ply"
    if not parts_ply.is_file():
        return None
    from tools.evaluate_closed_loop_motion import _ground_truth_moving

    gt = _ground_truth_moving(source_xyz.detach().cpu().double(), parts_ply)
    displacement = coupling_displacement(coupling, target_xyz, source_xyz).detach().cpu()
    normalised = torch.linalg.norm(displacement, dim=-1) / effective_scale.detach().cpu().clamp_min(1.0e-8)
    moving, static = normalised[gt], normalised[~gt]

    def _stats(v: torch.Tensor) -> dict[str, float]:
        if v.numel() == 0:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0}
        return {
            "mean": float(v.mean()),
            "median": float(v.median()),
            "p90": float(torch.quantile(v, 0.90)) if v.numel() > 1 else float(v.mean()),
        }

    # Rank separability of "does this Gaussian appear to move", which is the
    # only thing the downstream label stage can use from a coupling.
    order = torch.argsort(normalised)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(int(normalised.numel()), dtype=torch.float64)
    n1, n0 = float(gt.sum()), float((~gt).sum())
    auc = float((ranks[gt].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1.0))
    return {
        "ground_truth_used_as_algorithm_input": False,
        "gt_moving": int(gt.sum()),
        "moving_normalised_displacement": _stats(moving),
        "static_normalised_displacement": _stats(static),
        "moving_vs_static_median_ratio": float(
            moving.median() / static.median().clamp_min(1.0e-8)
        ) if moving.numel() and static.numel() else 0.0,
        "displacement_separability_auc": auc,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame0-model-dir", required=True)
    parser.add_argument("--frameT-model-dir", required=True)
    parser.add_argument("--iteration", type=int, default=14000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-scene", default="", help="only used for post-hoc GT reporting")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--edge-chunk", type=int, default=2048)

    parser.add_argument("--knn-rig", type=int, default=16)
    parser.add_argument("--knn-ctx", type=int, default=64)
    parser.add_argument("--descriptor-candidates", type=int, default=32)
    parser.add_argument("--lambda-distance-descriptor", type=float, default=1.0)
    parser.add_argument("--lambda-covariance-descriptor", type=float, default=0.2)
    parser.add_argument("--lambda-appearance-descriptor", type=float, default=0.0)
    parser.add_argument("--lambda-relational", type=float, default=1.0)
    parser.add_argument("--relational-sweeps", type=int, default=2)
    parser.add_argument("--relational-bandwidth", type=float, default=1.0)
    parser.add_argument("--unmatched-quantile", type=float, default=0.90)
    parser.add_argument("--column-capacity", type=float, default=4.0)
    parser.add_argument("--no-bidirectional-union", action="store_true")
    parser.add_argument("--gt-debug", action="store_true")
    return parser


def run(args) -> dict[str, Any]:
    device = torch.device(args.device)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz0, scale0, ply0 = _load_cloud(args.frame0_model_dir, int(args.iteration), int(args.sh_degree))
    xyzT, scaleT, plyT = _load_cloud(args.frameT_model_dir, int(args.iteration), int(args.sh_degree))
    xyz0, scale0 = xyz0.to(device), scale0.to(device)
    xyzT, scaleT = xyzT.to(device), scaleT.to(device)
    print(f"[relational] frame0={int(xyz0.shape[0])} ({ply0})")
    print(f"[relational] frameT={int(xyzT.shape[0])} ({plyT})")

    source = build_relational_graph(
        xyz0, scaling=scale0, knn_rig=int(args.knn_rig), knn_ctx=int(args.knn_ctx),
        chunk_size=max(512, int(args.chunk_size)),
    )
    target = build_relational_graph(
        xyzT, scaling=scaleT, knn_rig=int(args.knn_rig), knn_ctx=int(args.knn_ctx),
        chunk_size=max(512, int(args.chunk_size)),
    )
    print(
        f"[relational] descriptors built: rig-KNN={int(args.knn_rig)} ctx-KNN={int(args.knn_ctx)} "
        f"| frame0 geometry-valid {float(source.geometry_valid.float().mean()):.4f}"
    )

    variants: dict[str, SparseRelationalCoupling] = {}
    variants["A0_xyz"] = build_xyz_coupling(
        source, target, max_candidates=int(args.descriptor_candidates),
        column_capacity=float(args.column_capacity), chunk_size=int(args.chunk_size),
    )
    variants["A1_descriptor"] = build_relational_coupling(
        source, target, max_candidates=int(args.descriptor_candidates),
        lambda_distance=float(args.lambda_distance_descriptor),
        lambda_covariance=float(args.lambda_covariance_descriptor),
        lambda_appearance=float(args.lambda_appearance_descriptor),
        relational_sweeps=0,
        unmatched_quantile=float(args.unmatched_quantile),
        column_capacity=float(args.column_capacity),
        bidirectional_union=not bool(args.no_bidirectional_union),
        chunk_size=int(args.chunk_size), edge_chunk=int(args.edge_chunk),
    )
    variants["A2_relational"] = build_relational_coupling(
        source, target, max_candidates=int(args.descriptor_candidates),
        lambda_distance=float(args.lambda_distance_descriptor),
        lambda_covariance=float(args.lambda_covariance_descriptor),
        lambda_appearance=float(args.lambda_appearance_descriptor),
        lambda_relational=float(args.lambda_relational),
        relational_sweeps=int(args.relational_sweeps),
        huber_delta=float(args.relational_bandwidth),
        unmatched_quantile=float(args.unmatched_quantile),
        column_capacity=float(args.column_capacity),
        bidirectional_union=not bool(args.no_bidirectional_union),
        chunk_size=int(args.chunk_size), edge_chunk=int(args.edge_chunk),
    )

    root_scene = Path(args.root_scene).expanduser().resolve() if args.root_scene else None
    results: dict[str, Any] = {}
    for name, coupling in variants.items():
        stats = compute_transport_coverage(coupling)
        entry: dict[str, Any] = {"coupling": stats}
        displacement = coupling_displacement(coupling, xyzT, xyz0)
        normalised = torch.linalg.norm(displacement, dim=-1) / source.effective_scale.clamp_min(1.0e-8)
        entry["displacement"] = {
            "mean": float(normalised.mean()),
            "median": float(normalised.median()),
            "p90": float(torch.quantile(normalised, 0.90)),
        }
        print(
            f"[relational] {name:16s} src_cov={stats['source_coverage']:.4f} "
            f"tgt_cov={stats['target_coverage']:.4f} peak_mass={stats['peak_mass_mean']:.3f} "
            f"entropy={stats['transport_entropy_mean']:.3f} "
            f"unmatched={stats['mean_unmatched_mass']:.3f} "
            f"distortion p50={stats['relational_distortion_p50']:.3f} p90={stats['relational_distortion_p90']:.3f}"
        )
        if bool(args.gt_debug) and root_scene is not None:
            # GT firewall: every coupling already exists before this runs.
            entry["gt"] = score_coupling_against_gt(xyz0, coupling, xyzT, root_scene, source.effective_scale)
            g = entry.get("gt")
            if g:
                print(
                    f"[relational] {name:16s} GT  moving |d|/s median={g['moving_normalised_displacement']['median']:.3f} "
                    f"static median={g['static_normalised_displacement']['median']:.3f} "
                    f"ratio={g['moving_vs_static_median_ratio']:.2f} "
                    f"separability AUC={g['displacement_separability_auc']:.4f}"
                )
        results[name] = entry
        torch.save(
            {
                "tgt_idx": coupling.tgt_idx.detach().cpu(),
                "mass": coupling.mass.detach().cpu(),
                "valid": coupling.valid.detach().cpu(),
                "unmatched_src": coupling.unmatched_src.detach().cpu(),
                "expected_displacement": displacement.detach().cpu(),
            },
            out_dir / f"coupling_{name}.pt",
        )

    summary = {
        "schema_version": 1,
        "pipeline": "relational_graph_coupling_probe",
        "stage": "A",
        "frame0_ply": str(ply0),
        "frameT_ply": str(plyT),
        "frame0_count": int(xyz0.shape[0]),
        "frameT_count": int(xyzT.shape[0]),
        "config": _json_safe(vars(args)),
        "variants": results,
        "uses_exact_gaussian_correspondence": False,
        "uses_global_rigid_transform": False,
        "ground_truth_used_as_algorithm_input": False,
    }
    _write_json(out_dir / "relational_graph_probe_summary.json", summary)
    print(f"[relational] complete: {out_dir}")
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
