"""Standalone probe: local rigid bidirectional dynamic-label refinement.

Refines ONLY the frame-0 static/dynamic probabilities.  It does not produce
part labels, a joint axis, a trajectory, or any global SE(3) -- those belong
to the later K-part phase, which consumes the local descriptors saved here.

What it does, in order:

    frame-0 3DGS  --forward FM-->  displacement  --denoise-->  local field
    frame-T 3DGS  --backward FM--> displacement  --denoise-->  local field
                            |
                            +-- continuous forward/backward cycle
                            +-- local inverse rotation/translation consistency
                            v
              conservative frame-0 probability update, repeated

Flow matching is only ever evaluated on the two ORIGINAL endpoint clouds
(X0 -> XT and XT -> X0).  No transformed cloud is ever fed back into it, no
exact Gaussian correspondence is assumed, and no single global rigid
transform is fitted anywhere.

Ground truth is a firewall: it is loaded only after the refined labels
exist, and only to score them.

Run:
    PYTHONPATH=. python tools/run_local_rigid_dynamic_refinement_probe.py \
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

from utils.bidirectional_local_consistency import evaluate_local_inverse_consistency  # noqa: E402
from utils.dynamic_label_refinement import (  # noqa: E402
    LABEL_AMBIGUOUS,
    LABEL_DYNAMIC,
    LABEL_STATIC,
    DynamicRefinementState,
    RefinementWeights,
    assign_discrete_labels,
    compute_classification_metrics,
    compute_prior_logit,
    fit_confidence_weights,
    refine_dynamic_probabilities,
)
from utils.local_rigid_field import (  # noqa: E402
    build_knn_graph,
    compute_effective_local_scale,
    compute_motion_gated_rigid_evidence,
    denoise_displacement_field,
    fit_local_deformation,
    local_scale_from_knn,
)

#: Ablation ladder: each stage adds exactly one mechanism.  The reverse
#: stages (F/G/H) only do something when their CLI weight is explicitly
#: raised above zero -- the backward flow-matching branch has to pass its own
#: diagnostics before its evidence is worth acting on, and turning these up
#: to compensate for a bad backward field would be fitting the label rule to
#: an upstream defect.
ABLATION_STAGES = {
    "A": {"label": "raw forward FM", "effective_scale": False, "refine": False,
          "leave_center_out": True, "irls_iters": 1, "cycle": False, "rotation": False, "jacobian": False},
    "B": {"label": "+ robust effective scale", "effective_scale": True, "refine": False,
          "leave_center_out": True, "irls_iters": 1, "cycle": False, "rotation": False, "jacobian": False},
    "C": {"label": "+ motion-gated rigid evidence", "effective_scale": True, "refine": True,
          "leave_center_out": False, "irls_iters": 1, "cycle": False, "rotation": False, "jacobian": False},
    "D": {"label": "+ leave-one-centre-out recovery", "effective_scale": True, "refine": True,
          "leave_center_out": True, "irls_iters": 1, "cycle": False, "rotation": False, "jacobian": False},
    "E": {"label": "+ adaptive robust IRLS", "effective_scale": True, "refine": True,
          "leave_center_out": True, "irls_iters": 2, "cycle": False, "rotation": False, "jacobian": False},
    "F": {"label": "+ backward cycle (needs backward FM)", "effective_scale": True, "refine": True,
          "leave_center_out": True, "irls_iters": 2, "cycle": True, "rotation": False, "jacobian": False},
    "G": {"label": "+ rotation inverse consistency", "effective_scale": True, "refine": True,
          "leave_center_out": True, "irls_iters": 2, "cycle": True, "rotation": True, "jacobian": False},
    "H": {"label": "+ Jacobian inverse consistency", "effective_scale": True, "refine": True,
          "leave_center_out": True, "irls_iters": 2, "cycle": True, "rotation": True, "jacobian": True},
}


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


def _load_cloud(model_dir: str, iteration: int, sh_degree: int = 3) -> tuple[torch.Tensor, Path]:
    from scene.gaussian_model import GaussianModel

    ply = _find_model_ply(model_dir, iteration)
    if ply is None:
        raise SystemExit(f"No point_cloud.ply found under {model_dir}")
    model = GaussianModel(int(sh_degree), optimizer_type="default")
    model.load_ply(str(ply))
    return model.get_canonical_xyz.detach().float(), ply


def displacement_statistics(displacement: torch.Tensor) -> dict[str, Any]:
    """Magnitude spread plus directional collapse for one displacement field.

    ``unit_direction_resultant_length`` is the norm of the mean unit vector:
    ~0 for a physically diverse motion field, ~1 when every point is being
    pushed the same way, which is what a collapsed flow-matching branch looks
    like regardless of how plausible its magnitudes are.
    """
    d = torch.as_tensor(displacement).detach().float().reshape(-1, 3)
    norm = torch.linalg.norm(d, dim=-1)
    finite = torch.isfinite(norm)
    norm = norm[finite]
    if norm.numel() == 0:
        return {"count": 0}
    unit = d[finite] / norm.clamp_min(1.0e-12)[:, None]
    mean_unit = unit.mean(dim=0)
    return {
        "count": int(norm.numel()),
        "mean": float(norm.mean()),
        "median": float(norm.median()),
        "p90": float(torch.quantile(norm, 0.90)),
        "p95": float(torch.quantile(norm, 0.95)),
        "p99": float(torch.quantile(norm, 0.99)),
        "max": float(norm.max()),
        "mean_direction": [float(v) for v in mean_unit],
        "unit_direction_resultant_length": float(torch.linalg.norm(mean_unit)),
    }


def _sample_training_target_displacement(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    transport_result,
    *,
    seed: int,
    samples: int = 8192,
) -> torch.Tensor | None:
    """Offline replica of the couplings FM actually trains on (stage 2).

    Reproduces the sampling ``train_endpoint_flow_matching`` does internally
    so the training targets can be inspected without modifying the FM module:
    if the endpoint transport looks fine but these do not, the fault is in
    candidate sampling rather than in the coupling or the network.
    """
    indices = transport_result.target_indices
    weights = transport_result.transport_weights
    if indices is None or weights is None:
        return None
    generator = torch.Generator().manual_seed(int(seed))
    n = int(source_xyz.shape[0])
    take = min(int(samples), n)
    source_ids = torch.randperm(n, generator=generator)[:take]
    w = weights[source_ids].clamp_min(0.0).float()
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    picked = torch.multinomial(w, num_samples=1, replacement=True, generator=generator).reshape(-1)
    target_ids = indices[source_ids, picked].long()
    return target_xyz[target_ids] - source_xyz[source_ids]


def _flow_matching_config(args, seed_offset: int):
    from utils.endpoint_flow_matching import FlowMatchingConfig

    return FlowMatchingConfig(
        hidden_dim=int(args.fm_hidden_dim),
        layers=int(args.fm_layers),
        lr=float(args.fm_lr),
        iters=int(args.fm_iters),
        batch_size=int(args.fm_batch_size),
        identity_weight=float(args.fm_identity_weight),
        integration_steps=int(args.fm_integration_steps),
        integrator=str(args.fm_integrator),
        prior_prob_bias=float(args.fm_prior_prob_bias),
        prior_prob_temperature=float(args.fm_prior_prob_temperature),
        seed=int(args.seed) + int(seed_offset),
    )


def run_flow_matching(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    args,
    *,
    tag: str,
    cache_dir: Path,
    device: torch.device,
    seed_offset: int,
) -> dict[str, torch.Tensor]:
    """One directional FM pass on ORIGINAL clouds only: source -> target.

    ``tag`` is 'forward' (X0 -> XT) or 'backward' (XT -> X0).  Neither ever
    receives a transformed cloud; the local-rigidity feedback is applied
    downstream of this function, never inside it.
    """
    from utils.endpoint_flow_matching import (
        build_dynamic_prior_from_flow,
        integrate_flow,
        train_endpoint_flow_matching,
    )
    from utils.endpoint_transport_prior import EndpointTransportConfig, EndpointTransportPrior

    cfg = _flow_matching_config(args, seed_offset)
    transport_cfg = EndpointTransportConfig(
        knn=int(args.endpoint_transport_knn),
        temperature=float(args.endpoint_transport_temperature),
        robust_delta=float(args.endpoint_transport_robust_delta),
        chunk_size=int(args.chunk_size),
        prior_prob_bias=float(args.fm_prior_prob_bias),
        prior_prob_temperature=float(args.fm_prior_prob_temperature),
    )
    signature = {
        "tag": tag,
        "source_count": int(source_xyz.shape[0]),
        "target_count": int(target_xyz.shape[0]),
        "fm": {k: v for k, v in vars(cfg).items()},
        "transport_knn": int(transport_cfg.knn),
        "transport_temperature": float(transport_cfg.temperature),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cache_dir / f"{tag}_fm_config.json"
    payload_path = cache_dir / f"{tag}_fm_payload.pt"
    if (
        not bool(args.force_fm)
        and cfg_path.is_file()
        and payload_path.is_file()
        and json.loads(cfg_path.read_text(encoding="utf-8")) == _json_safe(signature)
    ):
        print(f"[local-rigid] {tag} FM cached -- skipping training")
        cached = torch.load(payload_path, map_location="cpu", weights_only=False)
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in cached.items()}

    print(f"[local-rigid] training {tag} FM: {int(source_xyz.shape[0])} -> {int(target_xyz.shape[0])} Gaussians")
    transport = EndpointTransportPrior(target_xyz.cpu(), {}, transport_cfg).compute(
        source_xyz.cpu(), torch.arange(int(source_xyz.shape[0]))
    )
    # Stage 1 of the failure-localisation instrumentation: the endpoint
    # transport coupling FM is about to be trained against.
    endpoint_transport_disp = transport.displacement.detach().cpu().float()
    # Stage 2: the couplings actually sampled during training.
    sampled_target_disp = _sample_training_target_displacement(
        source_xyz.cpu(), target_xyz.cpu(), transport, seed=int(cfg.seed)
    )
    model, _logs = train_endpoint_flow_matching(
        source_xyz.cpu(), target_xyz.cpu(), transport, cfg, device=device
    )
    integrated = integrate_flow(model, source_xyz.cpu(), cfg)
    # Stage 3: what the trained field integrates to.
    integrated_disp = (integrated - source_xyz.cpu()).float()
    prior = build_dynamic_prior_from_flow(
        source_xyz.cpu(), integrated, torch.arange(int(source_xyz.shape[0])), cfg,
        confidence=transport.prior.confidence, valid=transport.prior.valid,
    )
    stage_diagnostics = {
        "endpoint_transport": displacement_statistics(endpoint_transport_disp),
        "sampled_training_target": displacement_statistics(sampled_target_disp) if sampled_target_disp is not None else None,
        "integrated_fm": displacement_statistics(integrated_disp),
    }
    torch.save(endpoint_transport_disp, cache_dir / f"{tag}_endpoint_transport_disp.pt")
    if sampled_target_disp is not None:
        torch.save(sampled_target_disp, cache_dir / f"{tag}_sampled_training_target_disp.pt")
    torch.save(integrated_disp, cache_dir / f"{tag}_integrated_fm_disp.pt")
    _write_json(cache_dir / f"{tag}_fm_stage_diagnostics.json", stage_diagnostics)
    for name, stats in stage_diagnostics.items():
        if stats:
            print(
                f"[local-rigid] {tag}/{name}: mean={stats['mean']:.5f} median={stats['median']:.5f} "
                f"p99={stats['p99']:.5f} max={stats['max']:.5f} "
                f"direction_resultant={stats['unit_direction_resultant_length']:.4f}"
            )
    payload = {
        "displacement": integrated_disp,
        "probability": prior.probability.float(),
        "confidence": prior.confidence.float(),
        "valid": prior.valid.bool(),
        "stage_diagnostics": stage_diagnostics,
    }
    torch.save(payload, payload_path)
    _write_json(cfg_path, signature)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in payload.items()}


def _refinement_weights(args, stage: dict[str, Any]) -> RefinementWeights:
    return RefinementWeights(
        motion=float(args.local_rigid_lambda_motion),
        rigid=float(args.local_rigid_lambda_rigid),
        neighbor=float(args.local_rigid_lambda_neighbor) if bool(stage["leave_center_out"]) else 0.0,
        strain=float(args.local_rigid_lambda_strain),
        cycle=float(args.local_rigid_lambda_cycle) if bool(stage["cycle"]) else 0.0,
        rotation_cycle=float(args.local_rigid_lambda_rot_cycle) if bool(stage["rotation"]) else 0.0,
        jacobian_cycle=float(args.local_rigid_lambda_jac_cycle) if bool(stage["jacobian"]) else 0.0,
        uncertainty=float(args.local_rigid_lambda_uncertainty),
    )


def _motion_gate_settings(args) -> tuple[float, float]:
    """Gate threshold/temperature, derived from FM's own motion boundary.

    The flow-matching prior already calls a point moving at normalised
    motion ``fm_prior_prob_bias``; reusing that boundary keeps the gate from
    being an independent hand-tuned constant.  Explicit CLI values override.
    """
    threshold = float(args.rigid_motion_gate_threshold)
    temperature = float(args.rigid_motion_gate_temperature)
    if threshold < 0.0:
        threshold = float(args.fm_prior_prob_bias)
    if temperature <= 0.0:
        temperature = max(float(args.fm_prior_prob_temperature), 1.0e-6)
    return threshold, temperature


def _fm_probability_from_displacement(displacement: torch.Tensor, scale: torch.Tensor, args) -> torch.Tensor:
    raw = torch.linalg.norm(displacement, dim=-1) / scale.clamp_min(1.0e-6)
    return torch.sigmoid(
        (raw - float(args.fm_prior_prob_bias)) / max(float(args.fm_prior_prob_temperature), 1.0e-6)
    )


def refine(
    *,
    xyz0: torch.Tensor,
    xyzT: torch.Tensor,
    forward: dict[str, torch.Tensor],
    backward: dict[str, torch.Tensor],
    args,
    stage: dict[str, Any],
    knn0: tuple[torch.Tensor, torch.Tensor],
    knnT: tuple[torch.Tensor, torch.Tensor],
    raw_scale0: torch.Tensor,
    effective_scale0: torch.Tensor,
    effective_scaleT: torch.Tensor,
) -> dict[str, Any]:
    """The outer refinement loop for one ablation configuration."""
    idx0, dist0 = knn0
    idxT, distT = knnT

    forward_disp = forward["displacement"]
    backward_disp = backward["displacement"]
    forward_conf = forward["confidence"].clamp(0.0, 1.0)
    forward_valid = forward["valid"].bool()
    uncertainty = None

    # The length unit every normalised quantity uses -- including the flow
    # matching prior itself, which is a normalised displacement magnitude and
    # therefore inherits whatever density bias the raw scale carries.
    scale0 = effective_scale0 if bool(stage["effective_scale"]) else raw_scale0
    if bool(stage["effective_scale"]):
        forward_prob = _fm_probability_from_displacement(forward_disp, scale0, args)
    else:
        forward_prob = forward["probability"].clamp(0.0, 1.0)

    if str(args.displacement_denoiser) == "local":
        forward_disp, uncertainty = denoise_displacement_field(
            forward_disp, idx0, dist0, confidence=forward_conf, local_scale=scale0,
            huber_delta=float(args.local_rigid_huber_delta), iters=int(args.denoiser_iters),
        )
        backward_disp, _ = denoise_displacement_field(
            backward_disp, idxT, distT, confidence=backward["confidence"].clamp(0.0, 1.0),
            local_scale=effective_scaleT,
            huber_delta=float(args.local_rigid_huber_delta), iters=int(args.denoiser_iters),
        )
        forward_prob = _fm_probability_from_displacement(forward_disp, scale0, args)

    state = DynamicRefinementState(
        fm_prior_prob=forward_prob,
        fm_prior_logit=compute_prior_logit(forward_prob, cap=float(args.refine_prior_logit_cap)),
        refined_prob=forward_prob.clone(),
    )
    weights = _refinement_weights(args, stage)
    gate_threshold, gate_temperature = _motion_gate_settings(args)

    label, ambiguous = assign_discrete_labels(
        state.refined_prob,
        ambiguous_low=float(args.local_rigid_ambiguous_low),
        ambiguous_high=float(args.local_rigid_ambiguous_high),
    )
    forward_local = None
    consistency = None
    history: list[dict[str, Any]] = []
    previous_evidence = None
    previous_proposal = None

    iterations = int(args.local_rigid_refine_iters) if bool(stage["refine"]) else 0
    for iteration in range(max(0, iterations)):
        fit_weight = fit_confidence_weights(
            state.refined_prob, forward_conf, forward_valid, alpha=float(args.local_rigid_support_alpha)
        )
        forward_local = fit_local_deformation(
            xyz0, forward_disp, idx0, fit_weight, scale0, knn_distance=dist0,
            jacobian_reg=float(args.local_rigid_jacobian_reg),
            irls_iters=int(stage["irls_iters"]),
            huber_delta=float(args.local_rigid_huber_delta),
            robust_factor=float(args.local_rigid_robust_factor),
            huber_delta_min=float(args.local_rigid_huber_delta_min),
            leave_center_out=bool(stage["leave_center_out"]),
            spatial_sigma_scale=float(args.local_rigid_spatial_sigma_scale),
            min_valid_neighbors=int(args.local_rigid_min_valid_neighbors),
            degeneracy_ratio=float(args.local_rigid_degeneracy_ratio),
        )
        # Reverse diagnostics are always computed when a backward field exists;
        # whether they carry any label weight is a separate decision.
        consistency, _backward_local = evaluate_local_inverse_consistency(
            xyz0, forward_local, xyzT, backward_disp,
            local_scale=scale0,
            backward_confidence=backward["confidence"].clamp(0.0, 1.0),
            knn=int(args.local_rigid_knn),
            mode=str(args.local_rigid_cycle_mode),
            chunk_size=int(args.chunk_size),
            jacobian_reg=float(args.local_rigid_jacobian_reg),
            irls_iters=int(stage["irls_iters"]),
            huber_delta=float(args.local_rigid_huber_delta),
            robust_factor=float(args.local_rigid_robust_factor),
            huber_delta_min=float(args.local_rigid_huber_delta_min),
            spatial_sigma_scale=float(args.local_rigid_spatial_sigma_scale),
            min_valid_neighbors=int(args.local_rigid_min_valid_neighbors),
            degeneracy_ratio=float(args.local_rigid_degeneracy_ratio),
        )
        report = refine_dynamic_probabilities(
            state, forward_local, consistency, uncertainty,
            center_displacement=forward_disp,
            fm_confidence=forward_conf,
            ema=float(args.local_rigid_prob_ema),
            weights=weights,
            motion_gate_threshold=gate_threshold,
            motion_gate_temperature=gate_temperature,
            ambiguous_low=float(args.local_rigid_ambiguous_low),
            ambiguous_high=float(args.local_rigid_ambiguous_high),
        )
        state.refined_prob = report.probability
        label = report.label
        ambiguous = report.ambiguous

        row = {"iteration": int(iteration), **report.stats}
        # Distinguishes changing physics from plain EMA convergence: if the
        # evidence is unchanged the proposal must be unchanged too, and only
        # the refined probability may still move.
        row["mean_abs_evidence_change"] = (
            0.0 if previous_evidence is None else float((report.evidence - previous_evidence).abs().mean())
        )
        row["mean_abs_proposal_change"] = (
            0.0 if previous_proposal is None
            else float((report.proposal_probability - previous_proposal).abs().mean())
        )
        previous_evidence = report.evidence.clone()
        previous_proposal = report.proposal_probability.clone()
        history.append(row)
        print(
            f"[local-rigid iter {iteration:02d}] "
            f"mean_p={row['mean_p_dynamic']:.4f} dyn={row['dynamic_fraction']:.4f} "
            f"amb={row['ambiguous_fraction']:.4f} "
            f"d_evidence={row['mean_abs_evidence_change']:.2e} "
            f"d_proposal={row['mean_abs_proposal_change']:.2e} "
            f"d_prob={row['mean_abs_refined_prob_change']:.2e} "
            f"inlier={row['mean_inlier_fraction']:.3f}"
        )

    return {
        "probability": state.refined_prob,
        "proposal_probability": previous_proposal if previous_proposal is not None else state.refined_prob,
        "label": label,
        "ambiguous": ambiguous,
        "forward_disp": forward_disp,
        "backward_disp": backward_disp,
        "forward_prob": forward_prob,
        "prior_logit": state.fm_prior_logit,
        "effective_scale": scale0,
        "uncertainty": uncertainty,
        "forward_local": forward_local,
        "consistency": consistency,
        "history": history,
    }


def score_against_gt(
    xyz0: torch.Tensor,
    label: torch.Tensor,
    probability: torch.Tensor,
    root_scene: Path,
) -> dict[str, Any] | None:
    """Post-hoc only. Never called before the refined labels exist."""
    parts_ply = root_scene / "pointcloud" / "points3D_parts.ply"
    if not parts_ply.is_file():
        return None
    from tools.evaluate_closed_loop_motion import _ground_truth_moving

    gt = _ground_truth_moving(xyz0.detach().cpu().double(), parts_ply)
    # Every metric family is reported separately, each tied to an explicit
    # decision rule, so the headline number is never computed from a rule the
    # saved labels do not use.
    return compute_classification_metrics(probability, label, gt)


def write_debug_html(path: Path, xyz: torch.Tensor, scalars: dict[str, torch.Tensor], vectors: dict[str, torch.Tensor], *, max_points: int = 20000, max_vectors: int = 1500) -> None:
    """Frame-0 point cloud with switchable scalar overlays and vector fields.

    Deliberately shows no global transform: every overlay is per-Gaussian or
    a local vector, because a single scene-level arrow would misrepresent
    what this stage computes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(xyz.shape[0])
    ids = torch.arange(n)
    if n > int(max_points):
        ids = ids[:: max(1, n // int(max_points))][: int(max_points)]
    points = xyz[ids].detach().cpu()
    scalar_data = {k: v.detach().cpu().float()[ids].tolist() for k, v in scalars.items() if v is not None}

    vector_ids = ids[:: max(1, int(ids.numel()) // max(1, int(max_vectors)))][: int(max_vectors)]
    vector_data: dict[str, list[list[float]]] = {}
    for name, field in vectors.items():
        if field is None:
            continue
        tail = xyz[vector_ids].detach().cpu()
        head = tail + field[vector_ids].detach().cpu()
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for a, b in zip(tail.tolist(), head.tolist()):
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
        vector_data[name] = [xs, ys, zs]

    payload = {
        "x": points[:, 0].tolist(),
        "y": points[:, 1].tolist(),
        "z": points[:, 2].tolist(),
        "scalars": scalar_data,
        "vectors": vector_data,
    }
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>local rigid dynamic-label refinement</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:sans-serif;margin:0}} #controls{{padding:8px 12px}} #plot{{height:86vh}}</style>
</head><body>
<div id="controls">
  overlay:
  <select id="scalar"></select>
  &nbsp; vectors:
  <select id="vector"><option value="">none</option></select>
</div>
<div id="plot"></div>
<script>
const d = {json.dumps(payload)};
const scalarSelect = document.getElementById('scalar');
Object.keys(d.scalars).forEach(name => {{
  const option = document.createElement('option');
  option.value = name; option.textContent = name;
  scalarSelect.appendChild(option);
}});
const vectorSelect = document.getElementById('vector');
Object.keys(d.vectors).forEach(name => {{
  const option = document.createElement('option');
  option.value = name; option.textContent = name;
  vectorSelect.appendChild(option);
}});

function draw() {{
  const scalarName = scalarSelect.value;
  const traces = [{{
    type: 'scatter3d', mode: 'markers', name: scalarName,
    x: d.x, y: d.y, z: d.z,
    marker: {{ size: 2, color: d.scalars[scalarName], colorscale: 'Turbo',
               colorbar: {{ title: scalarName }} }},
  }}];
  const vectorName = vectorSelect.value;
  if (vectorName && d.vectors[vectorName]) {{
    const v = d.vectors[vectorName];
    traces.push({{ type: 'scatter3d', mode: 'lines', name: vectorName,
                   x: v[0], y: v[1], z: v[2], line: {{ width: 2, color: '#111' }} }});
  }}
  Plotly.react('plot', traces, {{ scene: {{ aspectmode: 'data' }}, margin: {{ l: 0, r: 0, b: 0, t: 10 }} }});
}}
scalarSelect.onchange = draw;
vectorSelect.onchange = draw;
if (scalarSelect.options.length) {{ scalarSelect.selectedIndex = 0; }}
draw();
</script></body></html>"""
    path.write_text(html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame0-model-dir", required=True)
    parser.add_argument("--frameT-model-dir", required=True)
    parser.add_argument("--iteration", type=int, default=14000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-scene", default="", help="only used for post-hoc GT scoring")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--chunk-size", type=int, default=2048)

    parser.add_argument("--use-local-rigid-label-refinement", action="store_true", default=True)
    parser.add_argument("--local-rigid-knn", type=int, default=16)
    parser.add_argument("--local-rigid-refine-iters", type=int, default=3)
    parser.add_argument(
        "--local-rigid-irls-iters", type=int, default=-1,
        help="override the ablation stage's IRLS iterations (-1 keeps the stage value)",
    )
    parser.add_argument("--local-rigid-huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--local-rigid-robust-factor", type=float, default=2.5,
        help="adaptive Huber threshold = factor * 1.4826 * MAD of the normalised residuals",
    )
    parser.add_argument("--local-rigid-huber-delta-min", type=float, default=0.05)
    parser.add_argument("--local-rigid-jacobian-reg", type=float, default=1.0e-5)
    parser.add_argument("--local-rigid-prob-ema", type=float, default=0.15)
    parser.add_argument("--local-rigid-spatial-sigma-scale", type=float, default=2.0)
    parser.add_argument("--local-rigid-min-valid-neighbors", type=int, default=6)
    parser.add_argument("--local-rigid-degeneracy-ratio", type=float, default=1.0e-3)
    parser.add_argument("--local-rigid-support-alpha", type=float, default=0.25)
    parser.add_argument("--local-rigid-cycle-mode", choices=("transform", "interpolate"), default="transform")

    parser.add_argument(
        "--refine-prior-logit-cap", type=float, default=4.0,
        help="clip |logit(p_fm)| so a saturated prior stays correctable (<=0 disables)",
    )
    parser.add_argument(
        "--local-scale-mode", choices=("raw", "neighborhood_median", "neighborhood_mean"),
        default="neighborhood_median",
    )
    parser.add_argument("--local-scale-clip-low-q", type=float, default=0.10)
    parser.add_argument("--local-scale-clip-high-q", type=float, default=0.90)
    parser.add_argument(
        "--rigid-motion-gate-threshold", type=float, default=-1.0,
        help="normalised motion at which the rigid-evidence gate opens (<0 derives it from the FM prior)",
    )
    parser.add_argument("--rigid-motion-gate-temperature", type=float, default=-1.0)

    parser.add_argument("--local-rigid-lambda-motion", type=float, default=2.0)
    parser.add_argument("--local-rigid-lambda-rigid", type=float, default=2.0)
    parser.add_argument("--local-rigid-lambda-neighbor", type=float, default=1.0)
    #: Off by default: empirically larger in high-motion regions, so as signed
    #: evidence it suppresses the moving points it should validate.
    parser.add_argument("--local-rigid-lambda-strain", type=float, default=0.0)
    #: Reverse terms stay off until the backward FM passes its diagnostics.
    parser.add_argument("--local-rigid-lambda-cycle", type=float, default=0.0)
    parser.add_argument("--local-rigid-lambda-rot-cycle", type=float, default=0.0)
    parser.add_argument("--local-rigid-lambda-jac-cycle", type=float, default=0.0)
    parser.add_argument("--local-rigid-lambda-uncertainty", type=float, default=0.0)

    parser.add_argument("--local-rigid-ambiguous-low", type=float, default=0.30)
    parser.add_argument("--local-rigid-ambiguous-high", type=float, default=0.70)
    parser.add_argument("--save-local-rigid-debug", action="store_true")

    #: Measured to cost ~0.02 F1 on USB_100109; kept for ablation, off by default.
    parser.add_argument("--displacement-denoiser", choices=("none", "local"), default="none")
    parser.add_argument("--denoiser-iters", type=int, default=2)

    parser.add_argument("--fm-hidden-dim", type=int, default=128)
    parser.add_argument("--fm-layers", type=int, default=4)
    parser.add_argument("--fm-lr", type=float, default=1.0e-3)
    parser.add_argument("--fm-iters", type=int, default=1000)
    parser.add_argument("--fm-batch-size", type=int, default=2048)
    parser.add_argument("--fm-identity-weight", type=float, default=0.01)
    parser.add_argument("--fm-integration-steps", type=int, default=16)
    parser.add_argument("--fm-integrator", default="euler")
    parser.add_argument("--fm-prior-prob-bias", type=float, default=1.0)
    parser.add_argument("--fm-prior-prob-temperature", type=float, default=0.35)
    parser.add_argument("--force-fm", action="store_true")
    parser.add_argument("--endpoint-transport-knn", type=int, default=16)
    parser.add_argument("--endpoint-transport-temperature", type=float, default=0.02)
    parser.add_argument("--endpoint-transport-robust-delta", type=float, default=0.02)

    parser.add_argument("--ablation", default="E", help="one of A..H, or 'all' for the full ladder")
    parser.add_argument("--gt-debug", action="store_true", help="score refined labels against GT, after refinement")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="rigeuler3dgs-local-rigid")
    parser.add_argument("--wandb-run-name", default="")
    return parser


def run(args) -> dict[str, Any]:
    device = torch.device(args.device)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz0, ply0 = _load_cloud(args.frame0_model_dir, int(args.iteration), int(args.sh_degree))
    xyzT, plyT = _load_cloud(args.frameT_model_dir, int(args.iteration), int(args.sh_degree))
    xyz0 = xyz0.to(device)
    xyzT = xyzT.to(device)
    print(f"[local-rigid] frame0={int(xyz0.shape[0])} Gaussians ({ply0})")
    print(f"[local-rigid] frameT={int(xyzT.shape[0])} Gaussians ({plyT})")

    forward = run_flow_matching(xyz0, xyzT, args, tag="forward", cache_dir=out_dir / "fm", device=device, seed_offset=0)
    backward = run_flow_matching(xyzT, xyz0, args, tag="backward", cache_dir=out_dir / "fm", device=device, seed_offset=1)

    idx0, dist0 = build_knn_graph(xyz0, int(args.local_rigid_knn), chunk_size=int(args.chunk_size))
    idxT, distT = build_knn_graph(xyzT, int(args.local_rigid_knn), chunk_size=int(args.chunk_size))
    raw_scale0 = local_scale_from_knn(dist0)
    raw_scaleT = local_scale_from_knn(distT)
    effective_scale0 = compute_effective_local_scale(
        raw_scale0, idx0, mode=str(args.local_scale_mode),
        clip_low_quantile=float(args.local_scale_clip_low_q),
        clip_high_quantile=float(args.local_scale_clip_high_q),
    )
    effective_scaleT = compute_effective_local_scale(
        raw_scaleT, idxT, mode=str(args.local_scale_mode),
        clip_low_quantile=float(args.local_scale_clip_low_q),
        clip_high_quantile=float(args.local_scale_clip_high_q),
    )

    def _scale_stats(name: str, scale: torch.Tensor) -> dict[str, float]:
        return {
            "p10": float(torch.quantile(scale, 0.10)),
            "p50": float(torch.quantile(scale, 0.50)),
            "p90": float(torch.quantile(scale, 0.90)),
            "min": float(scale.min()),
            "max": float(scale.max()),
            "name": name,
        }

    scale_stats = {
        "raw_frame0": _scale_stats("raw", raw_scale0),
        "effective_frame0": _scale_stats(str(args.local_scale_mode), effective_scale0),
        "max_normalised_motion_raw": float(
            (torch.linalg.norm(forward["displacement"], dim=-1) / raw_scale0.clamp_min(1.0e-6)).max()
        ),
        "max_normalised_motion_effective": float(
            (torch.linalg.norm(forward["displacement"], dim=-1) / effective_scale0.clamp_min(1.0e-6)).max()
        ),
    }
    print(
        f"[local-rigid] scale p10/p50/p90 raw={scale_stats['raw_frame0']['p10']:.5f}/"
        f"{scale_stats['raw_frame0']['p50']:.5f}/{scale_stats['raw_frame0']['p90']:.5f} "
        f"effective={scale_stats['effective_frame0']['p10']:.5f}/"
        f"{scale_stats['effective_frame0']['p50']:.5f}/{scale_stats['effective_frame0']['p90']:.5f} | "
        f"max normalised motion {scale_stats['max_normalised_motion_raw']:.2f} -> "
        f"{scale_stats['max_normalised_motion_effective']:.2f}"
    )

    requested = str(args.ablation).upper()
    stages = list(ABLATION_STAGES) if requested == "ALL" else [requested]
    for name in stages:
        if name not in ABLATION_STAGES:
            raise SystemExit(f"--ablation must be one of {sorted(ABLATION_STAGES)} or 'all'")

    wandb_run = None
    if bool(args.wandb):
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(args.wandb_project),
                name=str(args.wandb_run_name) or out_dir.name,
                config=_json_safe(vars(args)),
            )
        except Exception as error:  # noqa: BLE001 - logging must never break the probe
            print(f"[local-rigid] W&B disabled: {error}")

    root_scene = Path(args.root_scene).expanduser().resolve() if args.root_scene else None
    results: dict[str, Any] = {}
    last: dict[str, Any] | None = None
    for name in stages:
        stage = dict(ABLATION_STAGES[name])
        if int(args.local_rigid_irls_iters) > 0:
            stage["irls_iters"] = int(args.local_rigid_irls_iters)
        weights = _refinement_weights(args, stage)
        inactive = [
            key for key, enabled, weight in (
                ("cycle", stage["cycle"], weights.cycle),
                ("rotation_cycle", stage["rotation"], weights.rotation_cycle),
                ("jacobian_cycle", stage["jacobian"], weights.jacobian_cycle),
            ) if enabled and weight == 0.0
        ]
        print(f"\n[local-rigid] === ablation {name}: {stage['label']} ===")
        if inactive:
            print(
                f"[local-rigid] note: {', '.join(inactive)} enabled by the stage but weighted 0 "
                "(reverse evidence stays off until the backward FM passes its diagnostics)"
            )
        outcome = refine(
            xyz0=xyz0, xyzT=xyzT, forward=forward, backward=backward, args=args, stage=stage,
            knn0=(idx0, dist0), knnT=(idxT, distT),
            raw_scale0=raw_scale0, effective_scale0=effective_scale0, effective_scaleT=effective_scaleT,
        )
        entry: dict[str, Any] = {
            "stage": name,
            "description": stage["label"],
            "config": {k: v for k, v in stage.items() if k != "label"},
            "weights": _json_safe(vars(weights)),
            "inactive_terms": inactive,
            "convergence": {
                "mean_abs_evidence_change": [row.get("mean_abs_evidence_change", 0.0) for row in outcome["history"]],
                "mean_abs_proposal_change": [row.get("mean_abs_proposal_change", 0.0) for row in outcome["history"]],
                "mean_abs_probability_change": [row.get("mean_abs_refined_prob_change", 0.0) for row in outcome["history"]],
            },
            "history": outcome["history"],
        }
        if bool(args.gt_debug) and root_scene is not None:
            # GT firewall: only reached once the refined labels already exist.
            entry["gt"] = score_against_gt(xyz0, outcome["label"], outcome["probability"], root_scene)
            g = entry.get("gt")
            if g:
                prob_m = g["probability_metrics"]
                sel = g["selective_metrics"]
                print(
                    f"[local-rigid] {name} GT  prob-threshold: P={prob_m['precision']:.4f} "
                    f"R={prob_m['recall']:.4f} F1={prob_m['f1']:.4f} acc={prob_m['accuracy']:.4f} "
                    f"tp={prob_m['true_positive']} fp={prob_m['false_positive']} fn={prob_m['false_negative']}"
                )
                print(
                    f"[local-rigid] {name} GT  selective(cov={sel['coverage']:.3f}): F1={sel['f1']:.4f} | "
                    f"amb-as-static F1={g['f1_ambiguous_as_static']['f1']:.4f} | "
                    f"amb-as-dynamic F1={g['f1_ambiguous_as_dynamic']['f1']:.4f} | "
                    f"ambiguous={g['ambiguity_metrics']['ambiguous_fraction']:.4f} "
                    f"(gt-dynamic share {g['ambiguity_metrics']['ambiguous_gt_dynamic_fraction']:.3f})"
                )
        results[name] = entry
        last = outcome
        if wandb_run is not None:
            for row in outcome["history"]:
                wandb_run.log(
                    {f"local_refine/{k}": v for k, v in row.items() if k != "iteration"}
                    | {"local_refine/iteration": row["iteration"], "local_refine/stage": name}
                )
            if entry.get("gt"):
                flat = {
                    f"local_refine/gt_{family}_{k}": v
                    for family in ("probability_metrics", "selective_metrics")
                    for k, v in entry["gt"][family].items()
                    if isinstance(v, (int, float))
                }
                wandb_run.log(flat | {"local_refine/stage": name})

    assert last is not None
    torch.save(last["probability"].detach().cpu(), out_dir / "refined_dynamic_probability_frame0.pt")
    torch.save(last["probability"].detach().cpu(), out_dir / "refined_dynamic_prob_frame0.pt")
    torch.save(last["label"].detach().cpu(), out_dir / "refined_dynamic_label_frame0.pt")
    torch.save(last["ambiguous"].detach().cpu(), out_dir / "ambiguity_mask_frame0.pt")

    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, torch.Tensor] = {
        "fm_prior_prob": last["forward_prob"],
        "fm_prior_logit": last["prior_logit"],
        "proposal_prob": last["proposal_probability"],
        "refined_prob": last["probability"],
        "delta_prob": last["probability"] - last["forward_prob"],
        "raw_local_scale": raw_scale0,
        "effective_local_scale": last["effective_scale"],
        "forward_denoised_disp": last["forward_disp"],
        "backward_denoised_disp": last["backward_disp"],
        "ambiguity_mask": last["ambiguous"],
        "discrete_label": last["label"],
    }
    local = last["forward_local"]
    if local is not None:
        saved.update({
            "local_motion_magnitude": local.motion_magnitude,
            "local_static_error": local.static_fit_error,
            "local_rigid_error": local.rigid_fit_error,
            "local_rigid_failure_fraction": local.rigid_failure_fraction,
            "local_rigid_gain": local.rigid_gain,
            "neighbor_predicted_displacement": local.predicted_disp,
            "neighbor_motion_magnitude": local.motion_magnitude,
            "strain_error": local.strain_error,
            "local_cov_eigenvalues": local.cov_eigenvalues,
            "local_cov_condition": local.cov_condition,
            "local_inlier_fraction": local.inlier_fraction,
            "local_robust_delta": local.robust_delta,
            "forward_local_J": local.J,
            "forward_local_R": local.R,
            "forward_local_t": local.t,
        })
    consistency = last["consistency"]
    if consistency is not None:
        saved.update({
            "reverse_query_valid": consistency.reverse_valid,
            "cycle_error": consistency.cycle_error,
            "rotation_inverse_error": consistency.rotation_cycle_error,
            "jacobian_inverse_error": consistency.jacobian_cycle_error,
            # Debug only -- never enters the label update.
            "translation_inverse_error": consistency.translation_cycle_error,
            "reverse_magnitude_ratio": consistency.reverse_magnitude_ratio,
            "reverse_direction_cosine": consistency.reverse_direction_cosine,
        })
    for name, tensor in saved.items():
        torch.save(tensor.detach().cpu(), diagnostics_dir / f"{name}.pt")

    reverse_summary: dict[str, Any] = {
        "forward_fm_stage_diagnostics": forward.get("stage_diagnostics"),
        "backward_fm_stage_diagnostics": backward.get("stage_diagnostics"),
    }
    if consistency is not None:
        reverse_valid = consistency.reverse_valid
        reverse_summary["reverse_valid_fraction"] = float(reverse_valid.float().mean())
        if bool(reverse_valid.any()):
            ratio = consistency.reverse_magnitude_ratio[reverse_valid]
            cosine = consistency.reverse_direction_cosine[reverse_valid]
            cycle = consistency.cycle_error[reverse_valid]
            reverse_summary.update({
                "cycle_stats": {"mean": float(cycle.mean()), "median": float(cycle.median())},
                "reverse_magnitude_ratio_stats": {"mean": float(ratio.mean()), "median": float(ratio.median())},
                "reverse_direction_cosine_stats": {"mean": float(cosine.mean()), "median": float(cosine.median())},
            })
            print(
                f"[local-rigid] reverse: valid={reverse_summary['reverse_valid_fraction']:.3f} "
                f"magnitude_ratio median={float(ratio.median()):.3f} "
                f"direction_cosine median={float(cosine.median()):.3f} "
                f"(inversion would give 1.0 and 1.0)"
            )

    geometry_summary: dict[str, Any] = {"scale_stats": scale_stats}
    if local is not None:
        valid = local.valid
        geometry_summary.update({
            "rigid_stats": {
                "mean_rigid_failure_fraction": float(local.rigid_failure_fraction[valid].mean()) if bool(valid.any()) else 0.0,
                "mean_rigid_gain": float(local.rigid_gain[valid].mean()) if bool(valid.any()) else 0.0,
            },
            "strain_stats": {"mean": float(local.strain_error[valid].mean()) if bool(valid.any()) else 0.0},
            "inlier_stats": {
                "mean": float(local.inlier_fraction.mean()),
                "median": float(local.inlier_fraction.median()),
                "fraction_saturated": float((local.inlier_fraction >= 0.999).float().mean()),
                "mean_robust_delta": float(local.robust_delta.mean()),
            },
            "degeneracy_stats": {
                "valid_fraction": float(valid.float().mean()),
                "mean_cov_condition": float(local.cov_condition[valid].mean()) if bool(valid.any()) else 0.0,
            },
        })

    summary = {
        "schema_version": 2,
        "pipeline": "local_rigid_bidirectional_dynamic_labeling",
        "frame0_ply": str(ply0),
        "frameT_ply": str(plyT),
        "frame0_count": int(xyz0.shape[0]),
        "frameT_count": int(xyzT.shape[0]),
        "config": _json_safe(vars(args)),
        "tensor_to_metric_mapping": {
            "refined_dynamic_probability_frame0.pt": "probability_metrics (p_refined >= 0.5)",
            "refined_dynamic_label_frame0.pt": "selective_metrics / f1_ambiguous_as_static / f1_ambiguous_as_dynamic",
            "ambiguity_mask_frame0.pt": "ambiguity_metrics",
        },
        "ablation": results,
        "geometry": geometry_summary,
        "reverse_fm": reverse_summary,
        "uses_global_rigid_transform": False,
        "uses_exact_gaussian_correspondence": False,
        "ground_truth_used_as_algorithm_input": False,
    }
    _write_json(out_dir / "local_rigid_summary.json", summary)

    if bool(args.save_local_rigid_debug):
        scalars = {
            "fm_prior_probability": last["forward_prob"],
            "refined_probability": last["probability"],
            "probability_delta": last["probability"] - last["forward_prob"],
            "discrete_label": last["label"].float(),
            "ambiguous_mask": last["ambiguous"].float(),
            "effective_local_scale": last["effective_scale"],
            "raw_local_scale": raw_scale0,
            "fm_displacement_magnitude": torch.linalg.norm(last["forward_disp"], dim=-1),
        }
        if local is not None:
            scalars.update({
                "local_motion_magnitude": local.motion_magnitude,
                "static_fit_error": local.static_fit_error,
                "local_rigid_fit_error": local.rigid_fit_error,
                "rigid_failure_fraction": local.rigid_failure_fraction,
                "motion_gated_rigid_evidence": compute_motion_gated_rigid_evidence(
                    local.motion_magnitude, local.rigid_failure_fraction,
                    threshold=_motion_gate_settings(args)[0], temperature=_motion_gate_settings(args)[1],
                ),
                "neighbor_predicted_motion": torch.linalg.norm(local.predicted_disp, dim=-1),
                "local_strain": local.strain_error,
                "covariance_degeneracy": local.cov_condition,
                "local_inlier_fraction": local.inlier_fraction,
            })
        if consistency is not None:
            scalars.update({
                "reverse_valid_mask": consistency.reverse_valid.float(),
                "cycle_error": consistency.cycle_error,
                "reverse_magnitude_ratio": consistency.reverse_magnitude_ratio,
                "reverse_direction_cosine": consistency.reverse_direction_cosine,
                "rotation_inverse_error": consistency.rotation_cycle_error,
                "jacobian_inverse_error": consistency.jacobian_cycle_error,
            })
        vectors = {
            "raw_fm_displacement": forward["displacement"],
            "denoised_displacement": last["forward_disp"],
        }
        if local is not None:
            vectors["neighbor_predicted_displacement"] = local.predicted_disp
        write_debug_html(out_dir / "local_rigid_debug.html", xyz0, scalars, vectors)
        print(f"[local-rigid] debug HTML: {out_dir / 'local_rigid_debug.html'}")

    if wandb_run is not None:
        wandb_run.finish()
    print(f"[local-rigid] complete: {out_dir}")
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
