#!/usr/bin/env python3
"""P0-P9 with the post-audit corrections. P10 stays behind an explicit gate.

What differs from the earlier probe, and why each difference matters:

  * every statistically valid hypothesis per source anchor is KEPT, instead
    of collapsing to one best target before the graph is built.  That
    collapse is what left P6, P7 and P8 with nothing to resolve;
  * the null reproduces the SEARCH (min over M candidates), not a single
    random pair, so widening the search no longer manufactures significance;
  * each transform carries its own bootstrap covariance, so P6 consistency
    cannot be certified by the scene-wide scatter it was meant to reject;
  * P8 calibrates on frame-T cameras, which the frame-0 model never saw, and
    contrasts the moved Gaussians against those same held-out images;
  * coverage is reported against the physical target domain, not against the
    candidate slots that happened to be filled.

Ground truth is a firewall: loaded only after every decision, used only to
report, and reported as TWO separate things -- dynamic-region classification
and transform accuracy -- because a point can be correctly called dynamic
while its motion is completely wrong (§35).

Run:
    PYTHONPATH=. python tools/run_rigid_patch_hypothesis_probe.py \\
        --frame0-model-dir <frame0> --frameT-model-dir <frameT> \\
        --root-scene <scene> --output-dir <out>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils import gpu_memory, progress  # noqa: E402
from utils.dynamic_prior import _binary_auc  # noqa: E402
from utils.local_rigid_hypothesis import solve_isometry_with_rigid_selection  # noqa: E402
from utils.metric_patch_graph import (  # noqa: E402
    build_full_geometry_knn,
    build_nested_metric_patches,
    compute_pairwise_metric_matrix,
    gather_patches,
    normalise_metric_matrix,
)
from utils.physical_appearance_disambiguation import (  # noqa: E402
    build_patch_sh_dc_descriptor,
    rank_physically_valid_candidates,
    sh_dc_to_rgb,
)
from utils.rigid_hypothesis_graph import (  # noqa: E402
    build_hypothesis_graph,
    hypotheses_per_source,
)
from utils.search_aware_null import (  # noqa: E402
    benjamini_hochberg,
    build_coverage_report,
    build_search_aware_null,
    search_aware_pvalue,
)
from utils.spatial_motion_posterior import compute_motion_posterior  # noqa: E402
from utils.transform_uncertainty import estimate_transform_uncertainty  # noqa: E402
from tools.run_relational_graph_probe import _find_model_ply, _json_safe, _write_json  # noqa: E402


def load_cloud(model_dir: str, iteration: int, sh_degree: int = 3):
    from scene.gaussian_model import GaussianModel

    ply = _find_model_ply(model_dir, iteration)
    if ply is None:
        raise SystemExit(f"No point_cloud.ply found under {model_dir}")
    model = GaussianModel(int(sh_degree), optimizer_type="default")
    model.load_ply(str(ply))
    xyz = model.get_canonical_xyz.detach().float()
    scaling = model.get_scaling.detach().float()
    rgb = sh_dc_to_rgb(model.get_features.detach().float())
    # The GaussianModel holds optimiser state and SH bands this stage never
    # needs; dropping it here is most of the difference on a 40k scene.
    del model
    gpu_memory.release()
    return xyz, scaling, rgb, ply


def json_finite(value):
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(v) for v in value]
    return value


def stats(value: torch.Tensor | None) -> dict[str, float]:
    if value is None:
        return {}
    v = torch.as_tensor(value).detach().float().cpu().reshape(-1)
    v = v[torch.isfinite(v)]
    if int(v.numel()) == 0:
        return {}
    return {
        "median": float(v.median()), "mean": float(v.mean()),
        "p10": float(torch.quantile(v, 0.10)), "p90": float(torch.quantile(v, 0.90)),
    }


def local_scale_from_knn(distances: torch.Tensor) -> torch.Tensor:
    return distances[:, 1:].median(dim=1).values.clamp_min(1.0e-8)


# --------------------------------------------------------------------------
# P3/P4/P5 with multi-hypothesis retention and search-aware nulls
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_source_block(
    source_xyz, target_xyz, d0, source_idx, source_valid, source_scale,
    dt_pool, target_idx_pool, target_valid_pool, target_anchors,
    *, candidates, null_replicates, generator, device, block_label,
):
    """All candidates for one block of source anchors, plus its search-aware null.

    Returns every candidate's statistics -- not just the best -- because §14
    requires all significant hypotheses to survive into P6.
    """
    batch = int(d0.shape[0])
    pool = int(target_anchors.numel())
    picks = torch.randint(0, pool, (batch, int(candidates)), generator=generator).to(device)

    def evaluate(index: torch.Tensor):
        coupling, distortion, hypothesis = solve_isometry_with_rigid_selection(
            source_xyz[source_idx], target_xyz[target_idx_pool[index]], d0, dt_pool[index],
            source_valid, target_valid_pool[index], patch_scale=source_scale,
        )
        residual = torch.where(
            hypothesis.degenerate, torch.full_like(hypothesis.residual, float("inf")), hypothesis.residual
        )
        return coupling, distortion, residual, hypothesis

    records: list[dict[str, torch.Tensor]] = []
    inner = progress.bar(range(int(candidates)), desc=f"  {block_label} candidates", unit="cand")
    for slot in inner:
        index = picks[:, slot]
        coupling, distortion, residual, hypothesis = evaluate(index)
        records.append({
            "target_slot": index,
            "distortion": distortion,
            "residual": residual,
            "rotation": hypothesis.rotation,
            "translation": hypothesis.translation,
            "degenerate": hypothesis.degenerate,
            "matched_mass": coupling.sum(dim=(1, 2)),
            "coupling": coupling,
        })
        inner.set_postfix({
            "E_met": f"{float(distortion.median()):.4f}",
            "E_SE3": f"{float(residual[torch.isfinite(residual)].median()) if bool(torch.isfinite(residual).any()) else float('nan'):.4f}",
        })
    inner.close()

    # §11: the null runs the SAME search -- M candidates, keep the minimum.
    null = build_search_aware_null(
        lambda index: (lambda c, d, r, h: (d, r))(*evaluate(index)),
        batch=batch, pool_size=pool, candidates_per_search=int(candidates),
        replicates=int(null_replicates), generator=generator, device=device,
        progress_desc=f"  {block_label} search-aware null",
    )
    return records, null, picks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame0-model-dir", required=True)
    parser.add_argument("--frameT-model-dir", required=True)
    parser.add_argument("--root-scene", default="")
    parser.add_argument("--iteration", type=int, default=14000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--knn", type=int, default=48)
    parser.add_argument("--patch-level", type=int, default=1)
    parser.add_argument("--source-anchors", type=int, default=96)
    parser.add_argument("--target-anchors", type=int, default=96)
    parser.add_argument("--candidates", type=int, default=24,
                        help="target candidates evaluated per source anchor (M)")
    parser.add_argument("--null-replicates", type=int, default=12,
                        help="search-aware null replicates (B); each repeats the full M-candidate search")
    parser.add_argument("--bootstrap-draws", type=int, default=16)
    parser.add_argument("--uncertainty", default="bootstrap", choices=("bootstrap", "hessian"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--graph-neighbours", type=int, default=8)
    parser.add_argument("--min-anchors-per-component", type=int, default=3)

    parser.add_argument("--render-cameras", type=int, default=6)
    parser.add_argument("--render-hypotheses", type=int, default=12)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--save-tensors", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--report-memory", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    progress.set_enabled(not bool(args.no_progress))
    device = torch.device(args.device)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(int(args.seed))
    results: dict[str, Any] = {"config": {k: _json_safe(v) for k, v in vars(args).items()}}

    # ---------------- P0 ---------------------------------------------------
    with gpu_memory.scope("P0", report=bool(args.report_memory)), progress.stage("P0 mismatch + posterior"):
        xyz0, scale0, rgb0, ply0 = load_cloud(args.frame0_model_dir, int(args.iteration), int(args.sh_degree))
        xyzT, scaleT, rgbT, plyT = load_cloud(args.frameT_model_dir, int(args.iteration), int(args.sh_degree))
        xyz0, xyzT = xyz0.to(device), xyzT.to(device)
        knn0, dist0 = build_full_geometry_knn(xyz0, int(args.knn), chunk_size=int(args.chunk_size), desc="  knn frame0")
        knnT, distT = build_full_geometry_knn(xyzT, int(args.knn), chunk_size=int(args.chunk_size), desc="  knn frameT")
        posterior = compute_motion_posterior(
            xyz0, xyzT, local_scale_from_knn(dist0), local_scale_from_knn(distT), chunk_size=int(args.chunk_size)
        )
        gpu_memory.release(dist0, distT)

    fit = posterior.source_mixture
    progress.write(
        f"[P0] frame0={int(xyz0.shape[0])} frameT={int(xyzT.shape[0])} | "
        f"separation src {fit.separation:.2f} tgt {posterior.target_mixture.separation:.2f} | "
        f"bimodal {fit.bimodal}/{posterior.target_mixture.bimodal} | "
        f"MAP dynamic src {posterior.stats['source_map_dynamic_fraction']:.3f} "
        f"tgt {posterior.stats['target_map_dynamic_fraction']:.3f}"
    )
    results["P0"] = {"source_mixture": vars(fit), "target_mixture": vars(posterior.target_mixture),
                     **posterior.stats, "frame0_ply": str(ply0), "frameT_ply": str(plyT)}

    # ---------------- P1/P2 ------------------------------------------------
    with gpu_memory.scope("P2", report=bool(args.report_memory)), progress.stage("P2 patches"):
        source_patches = build_nested_metric_patches(xyz0, knn0, desc="  rank sweep frame0")
        target_patches = build_nested_metric_patches(xyzT, knnT, desc="  rank sweep frameT")
    source_pool = (posterior.source_map_dynamic & source_patches.full_rank).nonzero(as_tuple=True)[0]
    target_pool = (posterior.target_map_dynamic & target_patches.full_rank).nonzero(as_tuple=True)[0]
    progress.write(f"[P2] dynamic full-rank anchors: source {int(source_pool.numel())} target {int(target_pool.numel())}")

    if int(source_pool.numel()) < 8 or int(target_pool.numel()) < 8:
        results["verdict"] = {"result": "NO PHYSICALLY VALID TRANSPORT FOUND", "reason": "empty dynamic domain"}
        _write_json(out / "rigid_patch_hypothesis_probe.json", json_finite(_json_safe(results)))
        return 0

    def subsample(pool, count):
        if int(pool.numel()) <= int(count):
            return pool
        pick = torch.randperm(int(pool.numel()), generator=generator)[: int(count)]
        return pool[pick.sort().values.to(pool.device)]

    source_anchors = subsample(source_pool, int(args.source_anchors))
    target_anchors = subsample(target_pool, int(args.target_anchors))

    target_idx_pool, target_valid_pool = gather_patches(target_patches, target_anchors, level=int(args.patch_level))
    dt_pool, _scale_t = normalise_metric_matrix(
        compute_pairwise_metric_matrix(xyzT, target_idx_pool), valid=target_valid_pool
    )

    # ---------------- P3/P4/P5 ---------------------------------------------
    columns: dict[str, list[torch.Tensor]] = {}
    null_metric_all, null_se3_all, picks_all = [], [], []
    with progress.stage("P3-P5 multi-hypothesis + search-aware nulls"):
        for start in range(0, int(source_anchors.numel()), int(args.batch_size)):
            block = source_anchors[start:start + int(args.batch_size)]
            label = f"anchors {start}-{start + int(block.numel())}"
            with gpu_memory.scope(label):
                source_idx, source_valid = gather_patches(source_patches, block, level=int(args.patch_level))
                d0, source_scale = normalise_metric_matrix(
                    compute_pairwise_metric_matrix(xyz0, source_idx), valid=source_valid
                )
                records, null, picks = evaluate_source_block(
                    xyz0, xyzT, d0, source_idx, source_valid, source_scale,
                    dt_pool, target_idx_pool, target_valid_pool, target_anchors,
                    candidates=int(args.candidates), null_replicates=int(args.null_replicates),
                    generator=generator, device=device, block_label=label,
                )
                for record in records:
                    uncertainty = estimate_transform_uncertainty(
                        xyz0[source_idx], xyzT[target_idx_pool[record["target_slot"]]],
                        source_valid, record["coupling"], record["rotation"], record["translation"],
                        method=str(args.uncertainty), draws=int(args.bootstrap_draws), generator=generator,
                    )
                    columns.setdefault("source_anchor", []).append(block.to(device))
                    columns.setdefault("target_anchor", []).append(target_anchors.to(device)[record["target_slot"]])
                    columns.setdefault("distortion", []).append(record["distortion"])
                    columns.setdefault("residual", []).append(record["residual"])
                    columns.setdefault("rotation", []).append(record["rotation"])
                    columns.setdefault("translation", []).append(record["translation"])
                    columns.setdefault("degenerate", []).append(record["degenerate"])
                    columns.setdefault("matched_mass", []).append(record["matched_mass"])
                    columns.setdefault("covariance", []).append(uncertainty.covariance)
                    columns.setdefault("observability_rank", []).append(uncertainty.observability_rank)
                    columns.setdefault("chirality", []).append(uncertainty.signed_volume)
                    columns.setdefault("patch_index", []).append(source_idx)
                    columns.setdefault("patch_valid", []).append(source_valid)
                null_metric_all.append(null.metric_minima)
                null_se3_all.append(null.se3_minima)
                picks_all.append(picks)
                del records, d0, source_idx, source_valid, source_scale

    # Patch width is decided per anchor by the local rank test, so different
    # blocks legitimately produce different widths. Pad to the widest with the
    # anchor's own index and a false validity flag, which keeps every gathered
    # index legal while excluding the padding from every computation.
    width = max(int(t.shape[1]) for t in columns["patch_index"])
    for key in ("patch_index", "patch_valid"):
        padded = []
        for tensor in columns[key]:
            missing = width - int(tensor.shape[1])
            if missing > 0:
                if key == "patch_index":
                    tensor = torch.cat((tensor, tensor[:, :1].expand(-1, missing)), dim=1)
                else:
                    tensor = torch.cat((tensor, torch.zeros_like(tensor[:, :1].expand(-1, missing))), dim=1)
            padded.append(tensor)
        columns[key] = padded
    hypothesis = {k: torch.cat(v, 0) for k, v in columns.items()}
    null_metric = torch.cat(null_metric_all, 0)
    null_se3 = torch.cat(null_se3_all, 0)
    # Every hypothesis is compared against the null of ITS OWN anchor, which
    # is the search that anchor actually ran. Hypotheses are appended
    # candidate-major within each block, so the anchor row is rebuilt from the
    # stored anchor ids rather than inferred from ordering.
    anchor_lookup = {int(a): i for i, a in enumerate(source_anchors.tolist())}
    rows = torch.tensor([anchor_lookup[int(a)] for a in hypothesis["source_anchor"].tolist()], device=device)
    metric_p = search_aware_pvalue(hypothesis["distortion"], null_metric[rows])
    se3_p = search_aware_pvalue(hypothesis["residual"], null_se3[rows])
    metric_pass, metric_cut = benjamini_hochberg(metric_p, alpha=float(args.alpha))
    se3_pass, se3_cut = benjamini_hochberg(se3_p, alpha=float(args.alpha))
    physically_valid = metric_pass.to(device) & se3_pass.to(device) & ~hypothesis["degenerate"]

    coverage = build_coverage_report(
        int(target_pool.numel()), torch.cat(picks_all, 0), int(args.candidates)
    )
    ambiguity = hypotheses_per_source(hypothesis["source_anchor"], physically_valid)
    progress.write(
        f"[P5] hypotheses {int(physically_valid.numel())} | metric-significant {int(metric_pass.sum())} "
        f"| SE3-significant {int(se3_pass.sum())} | both {int(physically_valid.sum())} "
        f"| anchors with >=1: {ambiguity['anchors_with_hypothesis']} "
        f"(mean {ambiguity['mean_per_anchor']:.2f}, max {ambiguity['max_per_anchor']}, "
        f"ambiguous {ambiguity['anchors_ambiguous']})"
    )
    progress.write(
        f"[P5] coverage: {coverage.target_anchor_search_fraction:.3f} of the {int(target_pool.numel())}-anchor "
        f"dynamic target domain actually tested"
    )
    results["P3_P5"] = {
        "hypotheses_tested": int(physically_valid.numel()),
        "metric_significant": int(metric_pass.sum()), "metric_fdr_cut": metric_cut,
        "se3_significant": int(se3_pass.sum()), "se3_fdr_cut": se3_cut,
        "both_significant": int(physically_valid.sum()),
        "selected_candidate_metric_distortion": stats(hypothesis["distortion"]),
        "selected_candidate_se3_residual": stats(hypothesis["residual"]),
        "null_metric_minimum": stats(null_metric), "null_se3_minimum": stats(null_se3),
        "search_aware": True, "null_replicates": int(args.null_replicates),
        "candidates_per_source": int(args.candidates),
        "coverage": coverage.summary(), "ambiguity": ambiguity,
        "matched_mass": stats(hypothesis["matched_mass"]),
        "observability_rank": stats(hypothesis["observability_rank"].float()),
        "chirality": stats(hypothesis["chirality"]),
    }

    # ---------------- P6 ---------------------------------------------------
    with progress.stage("P6 uncertainty-aware hypothesis graph"):
        graph = build_hypothesis_graph(
            hypothesis["source_anchor"], hypothesis["target_anchor"],
            hypothesis["rotation"], hypothesis["translation"], hypothesis["covariance"],
            xyz0, accepted=physically_valid, neighbours=int(args.graph_neighbours),
            alpha=float(args.alpha), min_anchors_per_component=int(args.min_anchors_per_component),
        )
    progress.write(
        f"[P6] nodes {graph.stats['nodes']} | candidate edges {graph.stats['candidate_edges']} "
        f"| consistent {graph.stats['consistent_edges']} ({graph.stats['edge_acceptance_rate']:.3f}) "
        f"| components {graph.stats['components']} | median observable rank {graph.stats['median_observable_rank']}"
    )
    results["P6"] = dict(graph.stats)

    # ---------------- P7: only where P6 left real ambiguity ----------------
    in_component = graph.component >= 0
    component_ambiguity = hypotheses_per_source(hypothesis["source_anchor"], in_component)
    appearance: dict[str, Any] = {"run": False, "anchors_with_multiple": component_ambiguity["anchors_ambiguous"]}
    if component_ambiguity["anchors_ambiguous"] > 0:
        with progress.stage("P7 appearance on surviving ambiguity"):
            source_descriptor = build_patch_sh_dc_descriptor(
                rgb0.to(device), hypothesis["patch_index"], hypothesis["patch_valid"]
            )
            target_idx, target_valid = gather_patches(
                target_patches, hypothesis["target_anchor"], level=int(args.patch_level)
            )
            target_descriptor = build_patch_sh_dc_descriptor(rgbT.to(device), target_idx, target_valid)
            distance = torch.linalg.norm(source_descriptor - target_descriptor, dim=-1)
            resolved = 0
            for anchor in torch.unique(hypothesis["source_anchor"][in_component]).tolist():
                members = ((hypothesis["source_anchor"] == anchor) & in_component).nonzero(as_tuple=True)[0]
                if int(members.numel()) < 2:
                    continue
                ranking = rank_physically_valid_candidates(
                    source_descriptor[members[:1]],
                    target_descriptor[members][None, :, :],
                    torch.ones((1, int(members.numel())), dtype=torch.bool, device=device),
                )
                resolved += int(not bool(ranking.ambiguous[0]))
            appearance = {
                "run": True,
                "anchors_with_multiple": component_ambiguity["anchors_ambiguous"],
                "anchors_resolved": resolved,
                "anchors_still_ambiguous": component_ambiguity["anchors_ambiguous"] - resolved,
                "appearance_distance": stats(distance),
            }
            progress.write(
                f"[P7] {appearance['anchors_with_multiple']} ambiguous anchors -> "
                f"{resolved} resolved by appearance, {appearance['anchors_still_ambiguous']} still ambiguous"
            )
    else:
        progress.write("[P7] inert: no source anchor carries more than one P6-valid hypothesis (§24)")
    results["P7"] = appearance

    # ---------------- P8: contrastive, predictively calibrated -------------
    render: dict[str, Any] = {"run": False, "reason": "--skip-render" if args.skip_render else "no P6 component"}
    if not bool(args.skip_render) and int(graph.stats["components"]) > 0:
        with progress.stage("P8 contrastive render on held-out frame-T views"):
            render = run_contrastive_render(args, xyz0, hypothesis, in_component, posterior, device=device)
    results["P8"] = render
    if not render.get("run"):
        progress.write(f"[P8] not run: {render.get('reason')}")

    # ---------------- P9: gated on ALL upstream stages ---------------------
    gate = {
        "P5_physical_hypotheses": int(physically_valid.sum()) > 0,
        "P6_coherent_component": int(graph.stats["components"]) > 0,
        "P7_ambiguity_resolved": appearance.get("anchors_still_ambiguous", 0) == 0,
        "P8_render_favours_dynamic": bool(render.get("prefers_dynamic_count", 0) > 0),
    }
    results["P9"] = {
        "gate": gate,
        "supervision_generated": all(gate.values()),
        "reason": None if all(gate.values()) else
                  "§33 forbids generating supervision until every upstream stage passes",
    }
    progress.write(f"[P9] gate {gate} -> supervision {'generated' if all(gate.values()) else 'WITHHELD'}")

    results["verdict"] = {
        "gate": gate,
        "proceed_to_fm": all(gate.values()),
        "result": ("READY FOR FM SUPERVISION" if all(gate.values())
                   else "NO VALIDATED TRANSPORT -- " + next(k for k, v in gate.items() if not v) + " failed"),
    }

    if bool(args.save_tensors):
        tensors = out / "tensors"
        tensors.mkdir(parents=True, exist_ok=True)
        for name in ("source_anchor", "target_anchor", "rotation", "translation",
                     "covariance", "observability_rank", "distortion", "residual", "chirality"):
            torch.save(hypothesis[name].cpu(), tensors / f"hypothesis_{name}.pt")
        torch.save(metric_p.cpu(), tensors / "hypothesis_metric_pvalue.pt")
        torch.save(se3_p.cpu(), tensors / "hypothesis_se3_pvalue.pt")
        torch.save(physically_valid.cpu(), tensors / "hypothesis_physically_valid.pt")
        torch.save(graph.edges.cpu(), tensors / "hypothesis_graph_edges.pt")
        torch.save(graph.edge_pvalue.cpu(), tensors / "hypothesis_graph_edge_pvalue.pt")
        torch.save(graph.component.cpu(), tensors / "hypothesis_component_id.pt")
        torch.save(graph.component_transform.cpu(), tensors / "component_consensus_transform.pt")
        torch.save(graph.component_covariance.cpu(), tensors / "component_consensus_covariance.pt")
        torch.save(posterior.source_dynamic_posterior.cpu(), tensors / "source_dynamic_posterior.pt")

    # ---------------- GT firewall ------------------------------------------
    root_scene = Path(args.root_scene).expanduser().resolve() if args.root_scene else None
    gt = None
    if root_scene is not None and (root_scene / "pointcloud" / "points3D_parts.ply").is_file():
        from tools.evaluate_closed_loop_motion import _ground_truth_moving

        gt = _ground_truth_moving(xyz0.detach().cpu().double(), root_scene / "pointcloud" / "points3D_parts.ply").bool()

    if gt is not None:
        q0 = posterior.source_dynamic_posterior.cpu()
        predicted = q0 > 0.5
        tp = int((predicted & gt).sum()); fp = int((predicted & ~gt).sum()); fn = int((~predicted & gt).sum())
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
        anchor_gt = gt[hypothesis["source_anchor"].cpu()]
        valid_cpu = physically_valid.cpu()
        results["ground_truth"] = {
            "ground_truth_used_as_algorithm_input": False,
            # §35: classification and motion accuracy are DIFFERENT things.
            "classification": {
                "P0_auc": _binary_auc(q0, gt), "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / max(1e-12, precision + recall),
                "gt_dynamic_fraction": float(gt.float().mean()),
                "hypothesis_anchors_gt_dynamic": float(anchor_gt.float().mean()),
                "accepted_anchors_gt_dynamic": float(anchor_gt[valid_cpu].float().mean()) if bool(valid_cpu.any()) else None,
            },
            "motion_accuracy": {
                "note": ("no per-part GT transform is available for this scene, so transform accuracy "
                         "is NOT established; high dynamic precision is not evidence of correct motion (§35)"),
                "rotation_error_deg": None, "translation_error": None, "endpoint_error": None,
            },
        }
        c = results["ground_truth"]["classification"]
        progress.write(
            f"\n[GT] classification: AUC {c['P0_auc']:.4f} F1 {c['f1']:.4f} | "
            f"accepted anchors GT-dynamic {c['accepted_anchors_gt_dynamic']}"
        )
        progress.write("[GT] motion accuracy: NOT ESTABLISHED (no GT transform for this scene)")
    else:
        results["ground_truth"] = None

    _write_json(out / "rigid_patch_hypothesis_probe.json", json_finite(_json_safe(results)))
    if bool(args.report_memory):
        results["memory"] = gpu_memory.summarise()
        progress.write(f"[mem] {gpu_memory.summarise()}")
    print(f"\n=== {results['verdict']['result']} ===")
    for name, passed in gate.items():
        print(f"  {'PASS' if passed else 'fail'}  {name}")
    print(f"\n[probe] wrote {out / 'rigid_patch_hypothesis_probe.json'}")
    return 0


@torch.no_grad()
def run_contrastive_render(args, xyz0, hypothesis, in_component, posterior, *, device):
    """P8 against frame-T images, which the frame-0 model never saw."""
    from types import SimpleNamespace

    from tools.test_render_denoise_discovery import build_scene
    from utils.render_hypothesis_likelihood import (
        compare_contrastive_hypotheses,
        fit_predictive_rgb_residual_model,
    )

    scene, gaussians, pipe, white_background = build_scene(SimpleNamespace(
        static_model_dir=args.frame0_model_dir, root_scene=args.root_scene,
        iteration=int(args.iteration), sh_degree=None, resolution=None, white_background=None,
        camera_data_device="cpu", use_depth=True, cameras="", fps=30.0,
    ))
    cameras = scene.getTrainCameras()
    background = torch.tensor(
        [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0], device=gaussians.get_xyz.device
    )
    count = int(gaussians.get_xyz.shape[0])
    if count != int(xyz0.shape[0]):
        return {"run": False, "reason": f"render model has {count} Gaussians, probe has {int(xyz0.shape[0])}"}

    static_mask = (posterior.source_dynamic_posterior <= 0.5).to(background.device)
    model = fit_predictive_rgb_residual_model(
        cameras, gaussians, pipe, background,
        static_gaussian_mask=static_mask, max_cameras=int(args.render_cameras),
    )
    progress.write(
        f"[P8] predictive residual model: scale {model.scale:.5g} from {model.samples} pixels "
        f"on static regions of held-out views"
    )

    members = in_component.cpu().nonzero(as_tuple=True)[0]
    order = torch.argsort(hypothesis["residual"].cpu()[members])
    chosen = members[order][: int(args.render_hypotheses)]

    prefers, ratios, rows = 0, [], []
    for index in progress.bar(chosen.tolist(), desc="P8 hypotheses", unit="patch", leave=True):
        mask = torch.zeros(count, dtype=torch.bool)
        mask[hypothesis["patch_index"][index].cpu()[hypothesis["patch_valid"][index].cpu()]] = True
        comparison = compare_contrastive_hypotheses(
            cameras, gaussians, pipe, background, mask.to(background.device),
            hypothesis["rotation"][index].cpu(), hypothesis["translation"][index].cpu(), model,
            max_cameras=int(args.render_cameras),
        )
        prefers += int(comparison.prefers_dynamic)
        ratios.append(comparison.log_likelihood_ratio)
        rows.append({"hypothesis": int(index), "log_likelihood_ratio": comparison.log_likelihood_ratio,
                     "prefers_dynamic": comparison.prefers_dynamic,
                     "indistinguishable": comparison.indistinguishable, "pixels": comparison.pixels})
        progress.write(f"    hypothesis {int(index):4d}  LLR {comparison.log_likelihood_ratio:+.4g} "
                       f"over {comparison.pixels} px -> "
                       f"{'DYNAMIC' if comparison.prefers_dynamic else ('ambiguous' if comparison.indistinguishable else 'static')}")
        gpu_memory.release()

    del scene, gaussians
    gpu_memory.release()
    return {
        "run": True, "calibration": "held-out frame-T views, static regions only",
        "residual_scale": model.scale, "residual_samples": model.samples,
        "hypotheses_tested": int(chosen.numel()), "prefers_dynamic_count": prefers,
        "log_likelihood_ratio": stats(torch.tensor(ratios)) if ratios else {},
        "per_hypothesis": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
