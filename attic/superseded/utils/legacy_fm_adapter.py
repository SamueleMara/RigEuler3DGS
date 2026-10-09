"""Adapter for the historical endpoint flow-matching initialization.

The progressive pipeline treats this output as measurement data.  The endpoint
xyz, displacement, probability and base confidence are immutable after this
module writes ``fm_base.pt``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from utils.dynamic_prior import DynamicPrior, prior_metrics


def _stats(name: str, value: torch.Tensor) -> dict[str, float]:
    v = torch.as_tensor(value).detach().cpu().float().reshape(-1)
    finite = v[torch.isfinite(v)]
    if finite.numel() == 0:
        return {f"{name}_{key}": float("nan") for key in ("mean", "median", "p10", "p25", "p50", "p75", "p90")}
    q = torch.quantile(finite, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    return {
        f"{name}_mean": float(finite.mean()),
        f"{name}_median": float(finite.median()),
        f"{name}_p10": float(q[0]),
        f"{name}_p25": float(q[1]),
        f"{name}_p50": float(q[2]),
        f"{name}_p75": float(q[3]),
        f"{name}_p90": float(q[4]),
    }


def _json_finite(value):
    if isinstance(value, dict):
        return {k: _json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_finite(v) for v in value]
    if isinstance(value, tuple):
        return [_json_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def legacy_fm_base_from_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    endpoint = inputs.get("endpoint_transport")
    if endpoint is None:
        raise RuntimeError(
            "progressive_rigid requires the restored legacy endpoint Flow Matching base; "
            "provide a final-frame static 3DGS and do not pass --no-endpoint-fm"
        )
    confidence = torch.as_tensor(inputs["fm_base_confidence"]).detach()
    if confidence.ndim == 2:
        confidence = confidence.mean(0)
    result = {
        "endpoint_xyz": torch.as_tensor(inputs["fm_base_displacement"]).to(inputs["canonical_fg"]) + inputs["canonical_fg"],
        "displacement": torch.as_tensor(inputs["fm_base_displacement"]).to(inputs["canonical_fg"]),
        "probability": torch.as_tensor(inputs["fm_base_probability"]).to(inputs["canonical_fg"]).reshape(-1).clamp(0.0, 1.0),
        "confidence": confidence.to(inputs["canonical_fg"]).reshape(-1).clamp(0.0, 1.0),
        "valid": torch.as_tensor(inputs["fm_base_valid"], device=inputs["canonical_fg"].device).bool().reshape(-1),
        "candidate_matches": endpoint.get("candidate_matches") if isinstance(endpoint, dict) else None,
        "candidate_costs": endpoint.get("candidate_costs") if isinstance(endpoint, dict) else None,
        "target_xyz": endpoint.get("target_xyz") if isinstance(endpoint, dict) else None,
        "summary": endpoint.get("summary", {}) if isinstance(endpoint, dict) else {},
    }
    n = int(inputs["canonical_fg"].shape[0])
    for key in ("endpoint_xyz", "displacement"):
        if tuple(result[key].shape) != (n, 3):
            raise RuntimeError(f"legacy FM {key} must be [{n},3], got {tuple(result[key].shape)}")
    for key in ("probability", "confidence", "valid"):
        if int(result[key].numel()) != n:
            raise RuntimeError(f"legacy FM {key} must have {n} rows")
    return result


def load_legacy_fm_base(path: str | Path, *, device=None, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "fm_base.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported legacy FM payload: {type(payload).__name__}")
    base = dict(fallback or {})
    for key in ("endpoint_xyz", "displacement", "probability", "confidence", "valid"):
        if key not in payload:
            raise RuntimeError(f"{source} is missing required FM field {key!r}")
        value = payload[key]
        base[key] = value.to(device=device) if torch.is_tensor(value) and device is not None else value
    for key in ("candidate_matches", "candidate_costs", "target_xyz", "summary"):
        if key in payload and payload[key] is not None:
            value = payload[key]
            base[key] = value.to(device=device) if torch.is_tensor(value) and device is not None else value
    base.setdefault("summary", {})
    base["summary"] = {**dict(base["summary"]), "loaded_legacy_fm_path": str(source)}
    return base


def write_fm_base_artifacts(
    fm: dict[str, Any],
    *,
    output_dir: str | Path,
    foreground_idx: torch.Tensor,
    root_scene: str | Path,
    min_valid_fraction: float = 0.05,
    min_gt_recall: float = 0.0,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "endpoint_xyz": fm["endpoint_xyz"].detach().cpu(),
        "displacement": fm["displacement"].detach().cpu(),
        "probability": fm["probability"].detach().cpu(),
        "confidence": fm["confidence"].detach().cpu(),
        "valid": fm["valid"].detach().cpu(),
        "candidate_matches": None if fm.get("candidate_matches") is None else fm["candidate_matches"].detach().cpu(),
        "candidate_costs": None if fm.get("candidate_costs") is None else fm["candidate_costs"].detach().cpu(),
        "target_xyz": None if fm.get("target_xyz") is None else fm["target_xyz"].detach().cpu(),
        "summary": dict(fm.get("summary", {})),
        "ground_truth_used_as_algorithm_input": False,
    }
    torch.save(payload, output / "fm_base.pt")

    disp_norm = torch.linalg.norm(payload["displacement"], dim=-1)
    prob = payload["probability"]
    conf = payload["confidence"]
    valid = payload["valid"]
    regression = {
        "schema_version": 1,
        "N": int(prob.numel()),
        **_stats("displacement_norm", disp_norm),
        **_stats("probability", prob),
        "probability_fraction_gt_0_5": float((prob > 0.5).float().mean()) if prob.numel() else 0.0,
        **_stats("confidence", conf),
        "valid_fraction": float(valid.float().mean()) if valid.numel() else 0.0,
        "ground_truth_used_as_algorithm_input": False,
        "fm_summary": dict(fm.get("summary", {})),
    }
    parts_ply = Path(root_scene).expanduser().resolve() / "pointcloud" / "points3D_parts.ply"
    if parts_ply.is_file():
        from tools.evaluate_closed_loop_motion import _ground_truth_moving

        gt = _ground_truth_moving(
            fm["endpoint_xyz"].detach().cpu().double() - payload["displacement"].double(),
            parts_ply,
        )
        prior = DynamicPrior(
            probability=prob,
            raw_score=disp_norm,
            valid=valid,
            confidence=conf,
            method="legacy_flow_matching_base",
            gaussian_ids=torch.as_tensor(foreground_idx).detach().cpu().long(),
            metadata={"ground_truth_used_as_algorithm_input": False},
        ).validate()
        full_gt = torch.zeros(max(int(prior.gaussian_ids.max()) + 1, gt.numel()), dtype=torch.bool)
        full_gt[prior.gaussian_ids] = gt
        regression["gt_metrics"] = prior_metrics(prior, full_gt, threshold=0.5, high_confidence_threshold=0.5)
    if regression["valid_fraction"] < float(min_valid_fraction):
        raise RuntimeError(
            f"legacy FM regression failed: valid_fraction={regression['valid_fraction']:.4f} "
            f"< {float(min_valid_fraction):.4f}"
        )
    gt_metrics = regression.get("gt_metrics") or {}
    recall = gt_metrics.get("all_recall")
    if recall is not None and float(recall) < float(min_gt_recall):
        raise RuntimeError(
            f"legacy FM regression failed: GT recall={float(recall):.4f} "
            f"< {float(min_gt_recall):.4f}"
        )
    regression = _json_finite(regression)
    (output / "fm_base_regression.json").write_text(
        json.dumps(regression, indent=2, allow_nan=False), encoding="utf-8"
    )
    return regression
