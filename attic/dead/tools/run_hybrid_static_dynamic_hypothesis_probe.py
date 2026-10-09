"""Stage A.5 probe: does a dynamic hypothesis add anything to raw XYZ?

The previous probe answered a narrower question and answered it negatively:
rigid-invariant descriptors do NOT beat raw XYZ proximity at deciding which
frame-0 Gaussians moved (AUC 0.568 vs 0.845 on USB_100109), because a smooth
3DGS surface has no local geometric signature to retrieve with.

So this stage does not try to replace XYZ.  It keeps XYZ as the STATIC
hypothesis and asks the relational branch only the question it can answer:
where the static explanation has already failed, is there a confident,
structurally corroborated region the part could have moved to?

    A0  static only          -- E_S, nearest target in absolute position
    A1  dynamic geometry     -- relational coupling, geometry descriptors
    A2  A1 + SH-DC appearance
    A3  A1 + renderer-aggregated appearance   (skipped unless supplied)
    A4  combined static-vs-dynamic score

The success criterion is NOT "A2 beats A0".  It is that the COMBINED score
beats raw E_S -- and specifically that it does so in the high-E_S bins,
because that is the only regime where a second hypothesis can contribute.
A combined score that merely ties A0 overall while losing the low-E_S bins
is a failure, and §33's binned table is what makes that visible.

Ground truth is a firewall.  It is loaded only after every score exists, and
it never touches a KNN graph, a descriptor, a coupling, a weight, a
threshold, or the combination.  No labels are written and no FM is trained.

Run:
    PYTHONPATH=. python tools/run_hybrid_static_dynamic_hypothesis_probe.py \
        --frame0-model-dir <frame0_static> --frameT-model-dir <frameT_static> \
        --iteration 14000 --output-dir <out> --root-scene <scene>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.dynamic_prior import _binary_auc  # noqa: E402
from utils.hybrid_motion_hypotheses import (  # noqa: E402
    HybridWeights,
    build_hybrid_evidence,
    combine_hypothesis_evidence,
    compute_peak_candidate,
    compute_static_mismatch,
)
from utils.relational_coupling import build_relational_coupling  # noqa: E402
from utils.relational_graph import build_relational_graph  # noqa: E402
from tools.run_relational_graph_probe import (  # noqa: E402
    _find_model_ply,
    _json_safe,
    _write_json,
)

SH_C0 = 0.28209479177387814


def load_cloud(model_dir: str, iteration: int, sh_degree: int = 3):
    """Positions, scales and the view-independent colour of one endpoint."""
    from scene.gaussian_model import GaussianModel

    ply = _find_model_ply(model_dir, iteration)
    if ply is None:
        raise SystemExit(f"No point_cloud.ply found under {model_dir}")
    model = GaussianModel(int(sh_degree), optimizer_type="default")
    model.load_ply(str(ply))
    # The DC band of the SH is the view-independent colour.  Only this band
    # is usable here: the higher bands encode view-dependent appearance, so
    # they change with the camera rather than with the Gaussian's identity,
    # and two independently trained endpoint models will not agree on them.
    dc = model.get_features.detach().float()[:, 0, :]
    rgb = (SH_C0 * dc + 0.5).clamp(0.0, 1.0)
    return (
        model.get_canonical_xyz.detach().float(),
        model.get_scaling.detach().float(),
        rgb,
        ply,
    )


def load_render_appearance(path: str | Path | None, count: int) -> torch.Tensor | None:
    """A3's renderer-aggregated appearance, if a previous stage produced it."""
    if not path:
        return None
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        print(f"[hybrid] A3 skipped: no renderer appearance at {file}")
        return None
    value = torch.load(str(file), map_location="cpu")
    if torch.is_tensor(value) and int(value.shape[0]) == int(count):
        return value.float().reshape(int(count), -1)
    print(f"[hybrid] A3 skipped: {file} does not hold one row per frame-0 Gaussian")
    return None


def evidence_for_variant(
    source_graph,
    target_graph,
    *,
    static_mismatch: torch.Tensor,
    lambda_appearance: float,
    args,
):
    """One relational coupling and the hybrid evidence it supports."""
    coupling = build_relational_coupling(
        source_graph, target_graph,
        max_candidates=int(args.descriptor_candidates),
        lambda_distance=float(args.lambda_distance_descriptor),
        lambda_covariance=float(args.lambda_covariance_descriptor),
        lambda_appearance=float(lambda_appearance),
        lambda_relational=float(args.lambda_relational),
        relational_sweeps=int(args.relational_sweeps),
        huber_delta=float(args.relational_bandwidth),
        unmatched_quantile=float(args.unmatched_quantile),
        column_capacity=float(args.column_capacity),
        chunk_size=int(args.chunk_size),
        edge_chunk=int(args.edge_chunk),
    )
    evidence = build_hybrid_evidence(
        coupling, source_graph.xyz, target_graph.xyz, source_graph.effective_scale,
        static_mismatch=static_mismatch,
        knn_indices=source_graph.knn_rig,
        locality_radius=float(args.locality_radius),
        max_anchors=int(args.context_anchors),
    )
    return coupling, evidence


def per_point_fields(coupling, evidence) -> dict[str, torch.Tensor]:
    """§32's per-point log. Every field is GT-free by construction."""
    peak = compute_peak_candidate(coupling)
    normalised = coupling.mass / coupling.mass.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    entropy = -(normalised * torch.log(normalised.clamp_min(1.0e-12))).sum(dim=1)
    return {
        "static_mismatch": evidence.static_mismatch,
        "dynamic_gate": evidence.dynamic_gate,
        "dynamic_candidate_entropy": entropy,
        "dynamic_peak_mass": evidence.relational_peak_mass,
        "dynamic_relational_confidence": evidence.relational_confidence,
        "best_dynamic_descriptor_cost": coupling.unary_cost.gather(1, peak[:, None]).reshape(-1),
        "best_dynamic_graph_support": -coupling.relational_cost.gather(1, peak[:, None]).reshape(-1),
        "dynamic_candidate_displacement": evidence.candidate_displacement,
        "dynamic_internal_distance_change": evidence.internal_change,
        "dynamic_context_distance_change": evidence.context_change,
        "dynamic_relational_evidence": evidence.relational_evidence,
        "dynamic_context_evidence": evidence.context_evidence,
    }


def summarise(value: torch.Tensor | None) -> dict[str, float]:
    if value is None or int(value.numel()) == 0:
        return {}
    v = value.detach().float().cpu()
    return {
        "mean": float(v.mean()),
        "median": float(v.median()),
        "p10": float(torch.quantile(v, 0.10)),
        "p90": float(torch.quantile(v, 0.90)),
    }


# --------------------------------------------------------------------------
# Everything below this line reads ground truth.  Nothing above it may.
# --------------------------------------------------------------------------
def load_ground_truth(source_xyz: torch.Tensor, root_scene: Path | None) -> torch.Tensor | None:
    if root_scene is None:
        return None
    parts_ply = root_scene / "pointcloud" / "points3D_parts.ply"
    if not parts_ply.is_file():
        print(f"[hybrid] no ground truth at {parts_ply}; scores will be omitted")
        return None
    from tools.evaluate_closed_loop_motion import _ground_truth_moving

    return _ground_truth_moving(source_xyz.detach().cpu().double(), parts_ply)


def classification_scores(score: torch.Tensor, gt: torch.Tensor, *, quantile: float) -> dict[str, float]:
    """Precision/recall/F1 at a fixed operating rate, plus AUC.

    The threshold is a quantile of the SCORE, never of the labels: picking a
    cut that maximises F1 against the ground truth would report the best of
    many decisions rather than the one the pipeline would actually make.
    """
    score = score.detach().float().cpu().reshape(-1)
    gt = gt.detach().cpu().bool().reshape(-1)
    cut = float(torch.quantile(score, 1.0 - float(quantile)))
    predicted = score >= cut
    tp = int((predicted & gt).sum())
    fp = int((predicted & ~gt).sum())
    fn = int((~predicted & gt).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "auc": _binary_auc(score, gt),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1.0e-12, precision + recall),
        "threshold_quantile": float(quantile),
        "predicted_dynamic": int(predicted.sum()),
    }


def binned_evaluation(
    static_mismatch: torch.Tensor,
    fields: dict[str, torch.Tensor],
    combined: torch.Tensor,
    gt: torch.Tensor,
    *,
    bins: int = 5,
) -> list[dict[str, Any]]:
    """§33: everything sliced by E_S quantile.

    The headline number hides the only thing that matters here.  A combined
    score can match raw E_S overall while contributing nothing where E_S
    fails, and that is indistinguishable from success without this table.
    """
    es = static_mismatch.detach().float().cpu().reshape(-1)
    gt = gt.detach().cpu().bool().reshape(-1)
    order = torch.argsort(es)
    chunks = torch.chunk(order, int(bins))
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        low = index * 100 // int(bins)
        high = (index + 1) * 100 // int(bins)
        truth = gt[chunk]
        row: dict[str, Any] = {
            "bin": f"{low}-{high}%",
            "count": int(chunk.numel()),
            "static_mismatch_range": [float(es[chunk].min()), float(es[chunk].max())],
            "gt_dynamic_fraction": float(truth.float().mean()),
            "relational_confidence_median": float(
                fields["dynamic_relational_confidence"].detach().float().cpu()[chunk].median()
            ),
            "candidate_displacement_median": float(
                fields["dynamic_candidate_displacement"].detach().float().cpu()[chunk].median()
            ),
        }
        # AUC needs both classes present inside the bin.
        if bool(truth.any()) and bool((~truth).any()):
            row["static_only_auc"] = _binary_auc(es[chunk], truth)
            row["combined_auc"] = _binary_auc(combined.detach().float().cpu()[chunk], truth)
            row["combined_minus_static_auc"] = row["combined_auc"] - row["static_only_auc"]
        else:
            row["static_only_auc"] = None
            row["combined_auc"] = None
            row["combined_minus_static_auc"] = None
        rows.append(row)
    return rows


def complementarity_table(
    static_mismatch: torch.Tensor,
    fields: dict[str, torch.Tensor],
    gt: torch.Tensor,
    *,
    bins: int = 5,
) -> dict[str, Any]:
    """Does each dynamic field say anything E_S has not already said?

    A field can score well overall purely by correlating with E_S, and then
    adding it to E_S can only dilute.  What decides whether a cue is worth
    combining is its AUC WITHIN an E_S bin, where E_S itself is nearly
    constant -- and whether that within-bin signal keeps the same sign across
    bins, because a cue that flips sign cannot be used with a single weight
    no matter how strong it looks in aggregate.
    """
    es = static_mismatch.detach().float().cpu().reshape(-1)
    gt = gt.detach().cpu().bool().reshape(-1)
    chunks = torch.chunk(torch.argsort(es), int(bins))
    es_rank = torch.argsort(torch.argsort(es)).float()

    table: dict[str, Any] = {}
    for name, value in fields.items():
        if value is None:
            continue
        v = value.detach().float().cpu().reshape(-1)
        per_bin = []
        for chunk in chunks:
            truth = gt[chunk]
            per_bin.append(_binary_auc(v[chunk], truth) if bool(truth.any()) and bool((~truth).any()) else None)
        present = [a for a in per_bin if a is not None]
        signs = {a > 0.5 for a in present}
        table[name] = {
            "overall_auc": _binary_auc(v, gt),
            "within_bin_auc": per_bin,
            "rank_correlation_with_static_mismatch": float(
                torch.corrcoef(torch.stack([es_rank, torch.argsort(torch.argsort(v)).float()]))[0, 1]
            ),
            "consistently_signed": len(signs) == 1,
            "min_abs_within_bin_lift": min((abs(a - 0.5) for a in present), default=0.0),
        }
    return table


def recovery_counts(static_score: torch.Tensor, combined: torch.Tensor, gt: torch.Tensor, *, quantile: float) -> dict[str, int]:
    """How many of the static branch's mistakes the combination actually fixes."""
    gt = gt.detach().cpu().bool().reshape(-1)
    static_score = static_score.detach().float().cpu().reshape(-1)
    combined = combined.detach().float().cpu().reshape(-1)
    static_predicted = static_score >= float(torch.quantile(static_score, 1.0 - float(quantile)))
    combined_predicted = combined >= float(torch.quantile(combined, 1.0 - float(quantile)))
    return {
        "false_static_recovered": int((~static_predicted & gt & combined_predicted).sum()),
        "false_static_remaining": int((~static_predicted & gt & ~combined_predicted).sum()),
        "false_dynamic_rejected": int((static_predicted & ~gt & ~combined_predicted).sum()),
        "false_dynamic_remaining": int((static_predicted & ~gt & combined_predicted).sum()),
        "newly_broken_static": int((static_predicted & gt & ~combined_predicted).sum()),
        "newly_broken_dynamic": int((~static_predicted & ~gt & combined_predicted).sum()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame0-model-dir", required=True)
    parser.add_argument("--frameT-model-dir", required=True)
    parser.add_argument("--iteration", type=int, default=14000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-scene", default="", help="only used for post-hoc GT reporting")
    parser.add_argument("--render-appearance", default="", help="A3: [N, C] tensor of renderer-aggregated appearance")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--edge-chunk", type=int, default=2048)

    parser.add_argument("--knn-rig", type=int, default=16)
    parser.add_argument("--knn-ctx", type=int, default=64)
    parser.add_argument("--descriptor-candidates", type=int, default=32)
    parser.add_argument("--lambda-distance-descriptor", type=float, default=1.0)
    parser.add_argument("--lambda-covariance-descriptor", type=float, default=0.2)
    parser.add_argument("--lambda-appearance-descriptor", type=float, default=1.0)
    parser.add_argument("--lambda-relational", type=float, default=1.0)
    parser.add_argument("--relational-sweeps", type=int, default=2)
    parser.add_argument("--relational-bandwidth", type=float, default=1.0)
    parser.add_argument("--unmatched-quantile", type=float, default=0.90)
    parser.add_argument("--column-capacity", type=float, default=4.0)
    parser.add_argument("--locality-radius", type=float, default=3.0)
    parser.add_argument("--context-anchors", type=int, default=64)

    parser.add_argument("--weight-static", type=float, default=2.0)
    parser.add_argument("--weight-relational", type=float, default=2.0)
    parser.add_argument("--weight-context", type=float, default=2.0)
    parser.add_argument("--eval-quantile", type=float, default=0.25,
                        help="fraction of points called dynamic when scoring precision/recall")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--save-tensors", action="store_true", help="write §50 per-point .pt debug tensors")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    xyz0, scale0, rgb0, ply0 = load_cloud(args.frame0_model_dir, int(args.iteration), int(args.sh_degree))
    xyzT, scaleT, rgbT, plyT = load_cloud(args.frameT_model_dir, int(args.iteration), int(args.sh_degree))
    xyz0, scale0, rgb0 = xyz0.to(device), scale0.to(device), rgb0.to(device)
    xyzT, scaleT, rgbT = xyzT.to(device), scaleT.to(device), rgbT.to(device)
    print(f"[hybrid] frame0={int(xyz0.shape[0])} ({ply0})")
    print(f"[hybrid] frameT={int(xyzT.shape[0])} ({plyT})")

    render0 = load_render_appearance(args.render_appearance, int(xyz0.shape[0]))
    renderT = load_render_appearance(args.render_appearance, int(xyzT.shape[0])) if render0 is not None else None

    graph_kwargs = dict(
        knn_rig=int(args.knn_rig), knn_ctx=int(args.knn_ctx), chunk_size=max(512, int(args.chunk_size)),
    )
    source_geom = build_relational_graph(xyz0, scaling=scale0, **graph_kwargs)
    target_geom = build_relational_graph(xyzT, scaling=scaleT, **graph_kwargs)
    # The appearance variants differ ONLY in the colour attached to the same
    # geometry, so they must reuse the same graph rather than rebuild it: a
    # rebuilt KNN would make A1 and A2 differ by more than the cue under test,
    # and four full builds do not fit in GPU memory anyway.
    source_sh = replace(source_geom, appearance=rgb0)
    target_sh = replace(target_geom, appearance=rgbT)
    print(f"[hybrid] graphs built | frame0 geometry-valid {float(source_geom.geometry_valid.float().mean()):.4f}")

    # A0 -- the static hypothesis, on its own.
    static_mismatch = compute_static_mismatch(xyz0, xyzT, source_geom.effective_scale, chunk_size=int(args.chunk_size))
    print(
        f"[hybrid] E_S median {float(static_mismatch.median()):.4f} "
        f"p90 {float(torch.quantile(static_mismatch, 0.90)):.4f}"
    )

    variants: dict[str, tuple[Any, Any]] = {}
    variants["A1_dynamic_geometry"] = evidence_for_variant(
        source_geom, target_geom, static_mismatch=static_mismatch, lambda_appearance=0.0, args=args,
    )
    variants["A2_dynamic_sh_appearance"] = evidence_for_variant(
        source_sh, target_sh, static_mismatch=static_mismatch,
        lambda_appearance=float(args.lambda_appearance_descriptor), args=args,
    )
    if render0 is not None and renderT is not None:
        variants["A3_dynamic_render_appearance"] = evidence_for_variant(
            replace(source_geom, appearance=render0.to(device)),
            replace(target_geom, appearance=renderT.to(device)),
            static_mismatch=static_mismatch,
            lambda_appearance=float(args.lambda_appearance_descriptor), args=args,
        )
    else:
        print("[hybrid] A3 not run: pass --render-appearance to include it")

    weights = HybridWeights(
        static_mismatch=float(args.weight_static),
        relational=float(args.weight_relational),
        context=float(args.weight_context),
    )

    results: dict[str, Any] = {
        "config": {k: _json_safe(v) for k, v in vars(args).items()},
        "frame0_count": int(xyz0.shape[0]),
        "frameT_count": int(xyzT.shape[0]),
        "A0_static_only": {"static_mismatch": summarise(static_mismatch)},
        "variants": {},
    }

    scores: dict[str, torch.Tensor] = {"A0_static_only": static_mismatch}
    logged: dict[str, dict[str, torch.Tensor]] = {}
    for name, (coupling, evidence) in variants.items():
        fields = per_point_fields(coupling, evidence)
        combined, terms = combine_hypothesis_evidence(evidence, weights=weights)
        logged[name] = dict(fields, combined_static_vs_dynamic_score=combined)
        scores[name] = evidence.relational_evidence
        scores[f"A4_combined_{name}"] = combined
        results["variants"][name] = {
            "evidence_stats": evidence.stats,
            "fields": {key: summarise(value) for key, value in fields.items()},
            "term_stats": {key: summarise(value) for key, value in terms.items()},
        }
        print(
            f"[hybrid] {name}: confidence {float(evidence.relational_confidence.median()):.4f} "
            f"| displacement {float(evidence.candidate_displacement.median()):.4f} scales"
        )

    if bool(args.save_tensors):
        tensors = out / "tensors"
        tensors.mkdir(parents=True, exist_ok=True)
        torch.save(static_mismatch.cpu(), tensors / "static_mismatch.pt")
        for name, fields in logged.items():
            for key, value in fields.items():
                if value is not None:
                    torch.save(value.detach().cpu(), tensors / f"{name}__{key}.pt")
        print(f"[hybrid] per-point tensors written to {tensors}")

    # ---------------- GT firewall: nothing above this line read labels ------
    root_scene = Path(args.root_scene).expanduser().resolve() if args.root_scene else None
    gt = load_ground_truth(xyz0, root_scene)
    if gt is None:
        results["ground_truth"] = None
        _write_json(out / "hybrid_hypothesis_probe.json", results)
        print(f"[hybrid] wrote {out / 'hybrid_hypothesis_probe.json'} (no GT available)")
        return 0

    gt = gt.to(torch.bool)
    results["ground_truth"] = {
        "ground_truth_used_as_algorithm_input": False,
        "gt_dynamic": int(gt.sum()),
        "gt_dynamic_fraction": float(gt.float().mean()),
    }
    results["scores"] = {
        name: classification_scores(value, gt, quantile=float(args.eval_quantile))
        for name, value in scores.items()
    }

    baseline = results["scores"]["A0_static_only"]["auc"]
    best_name, best = max(
        ((k, v) for k, v in results["scores"].items() if k.startswith("A4_combined")),
        key=lambda item: item[1]["auc"],
    )
    combined_score = scores[best_name]
    source_variant = best_name.replace("A4_combined_", "")
    results["verdict"] = {
        "baseline_static_auc": baseline,
        "best_combined": best_name,
        "best_combined_auc": best["auc"],
        "combined_minus_static_auc": best["auc"] - baseline,
        # §31: the criterion is the combination beating raw E_S, not the
        # relational branch beating it alone.
        "criterion_combined_beats_static": bool(best["auc"] > baseline),
        "recovery": recovery_counts(static_mismatch, combined_score, gt, quantile=float(args.eval_quantile)),
    }
    results["binned_by_static_mismatch"] = binned_evaluation(
        static_mismatch, logged[source_variant], combined_score, gt, bins=int(args.bins),
    )
    results["complementarity"] = {
        name: complementarity_table(static_mismatch, fields, gt, bins=int(args.bins))
        for name, fields in logged.items()
    }

    _write_json(out / "hybrid_hypothesis_probe.json", results)

    print("\n=== Stage A.5 ===")
    for name, score in results["scores"].items():
        print(f"  {name:38s} AUC {score['auc']:.4f}  F1 {score['f1']:.4f}")
    print(f"\n  criterion (combined > raw static): {results['verdict']['criterion_combined_beats_static']}")
    print(f"  {baseline:.4f} -> {best['auc']:.4f} ({best['auc'] - baseline:+.4f}) via {best_name}")
    print("\n  E_S bin      GT dyn   c_rel   disp   static AUC  combined AUC  delta")
    for row in results["binned_by_static_mismatch"]:
        static_auc = "  n/a " if row["static_only_auc"] is None else f"{row['static_only_auc']:.4f}"
        combined_auc = "  n/a " if row["combined_auc"] is None else f"{row['combined_auc']:.4f}"
        delta = "  n/a " if row["combined_minus_static_auc"] is None else f"{row['combined_minus_static_auc']:+.4f}"
        print(
            f"  {row['bin']:>10s}  {row['gt_dynamic_fraction']:.3f}  "
            f"{row['relational_confidence_median']:.3f}  {row['candidate_displacement_median']:6.2f}  "
            f"    {static_auc}       {combined_auc}  {delta}"
        )
    print("\n  complementarity with E_S (within-bin AUC is what a combination can use)")
    print(f"  {'field':42s} overall  corr(E_S)  consistent  min lift")
    for name, fields in results["complementarity"].items():
        print(f"  [{name}]")
        for field_name, row in sorted(fields.items(), key=lambda kv: -kv[1]["min_abs_within_bin_lift"]):
            print(
                f"    {field_name:40s} {row['overall_auc']:.4f}   {row['rank_correlation_with_static_mismatch']:+.3f}"
                f"      {str(row['consistently_signed']):5s}     {row['min_abs_within_bin_lift']:.3f}"
            )

    recovery = results["verdict"]["recovery"]
    print(
        f"\n  false-static recovered {recovery['false_static_recovered']} "
        f"(remaining {recovery['false_static_remaining']}) | "
        f"false-dynamic rejected {recovery['false_dynamic_rejected']} "
        f"(remaining {recovery['false_dynamic_remaining']})"
    )
    print(
        f"  newly broken: static {recovery['newly_broken_static']} "
        f"dynamic {recovery['newly_broken_dynamic']}"
    )
    print(f"\n[hybrid] wrote {out / 'hybrid_hypothesis_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
