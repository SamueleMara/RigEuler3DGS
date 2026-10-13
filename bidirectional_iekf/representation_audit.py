"""Does a two-source candidate population render as ONE scene?

T6 of CODEX_NEXT_STEPS_UNIFIED_PHASE1_AFTER_T5.md, section 2.

This is a DIAGNOSTIC. It measures what the unchanged renderer does with a
concatenated candidate population and reports a status. It is not a fusion
algorithm, and there is deliberately nothing here that could become one: no
`eta`, no optical thickness, no `alpha_eff`, no ownership logit, no blending
weight, no overlap threshold, no nearest-neighbour deduplication. Source
colour, opacity and scales are passed through untouched.

The identity it rests on is the compositor's own. Two Gaussians covering one
pixel accumulate

    A = 1 - (1 - a_1)(1 - a_2)

so duplicating a surface raises its alpha and shifts the normalised depth
D = Z/(A+eps) toward the duplicate. That is a fact about alpha compositing,
not permission to introduce a parameter that cancels it.

Possible statuses:

    PASSED                          disjoint sources compose as one scene
    UNRESOLVED_SCENE_REPRESENTATION sources multiply-represent a surface
    FAILED                          the diagnostic itself could not run
    NOT_RUN

PASSED on a synthetic disjoint fixture and UNRESOLVED on real endpoints are
compatible results, and the first does not license claiming the second.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from estimator.renderer_bridge import build_linearization
from estimator.state import GaussianMotionState

PASSED = "PASSED"
UNRESOLVED = "UNRESOLVED_SCENE_REPRESENTATION"
FAILED = "FAILED"
NOT_RUN = "NOT_RUN"


def concatenate_states(a: GaussianMotionState, b: GaussianMotionState) -> GaussianMotionState:
    """Disjoint union of two populations. Attributes are copied, not merged."""
    return GaussianMotionState(
        mu=torch.cat([a.mu, b.mu], 0), R=torch.cat([a.R, b.R], 0),
        v=torch.cat([a.v, b.v], 0), omega=torch.cat([a.omega, b.omega], 0),
        colors=torch.cat([a.colors, b.colors], 0),
        opacity=torch.cat([a.opacity, b.opacity], 0),
        scales=torch.cat([a.scales, b.scales], 0))


def _render(state, cameras):
    lin = build_linearization(state, cameras)
    return [{"rgb": v.rgb, "depth": v.depth, "alpha": v.alpha,
             "n_contrib": v.n_contrib} for v in lin.views]


def alpha_composition_defect(alpha_a, alpha_b, alpha_union) -> dict:
    """How far the union's alpha is from 1 - (1-a)(1-b), per pixel."""
    predicted = 1.0 - (1.0 - alpha_a) * (1.0 - alpha_b)
    defect = (alpha_union - predicted).abs()
    return {"max_abs": float(defect.max()), "mean_abs": float(defect.mean()),
            "predicted_max": float(predicted.max())}


def audit_pair(state_a, state_b, cameras, *, label: str,
               overlap_alpha_tol: float = 1e-3) -> dict:
    """Render A, B and A+B, and report whether the union is one scene."""
    views_a = _render(state_a, cameras)
    views_b = _render(state_b, cameras)
    views_u = _render(concatenate_states(state_a, state_b), cameras)

    per_camera = []
    overlap_pixels_total = 0
    worst_alpha_gain = 0.0
    worst_depth_shift = 0.0
    for va, vb, vu in zip(views_a, views_b, views_u):
        # A pixel is multiply represented when BOTH sources put real
        # coverage there. That is measured from the renderer's own alpha,
        # not from a distance between Gaussians.
        both = (va["alpha"] > overlap_alpha_tol) & (vb["alpha"] > overlap_alpha_tol)
        overlap = int(both.sum())
        overlap_pixels_total += overlap
        composition = alpha_composition_defect(va["alpha"], vb["alpha"], vu["alpha"])

        if overlap:
            gain = (vu["alpha"][both] - torch.maximum(va["alpha"][both],
                                                      vb["alpha"][both]))
            worst_alpha_gain = max(worst_alpha_gain, float(gain.max()))
            # Where only A is present the union's depth must equal A's.
            only_a = (va["alpha"] > 0.5) & (vb["alpha"] <= overlap_alpha_tol)
            if bool(only_a.any()):
                worst_depth_shift = max(worst_depth_shift, float(
                    (vu["depth"][only_a] - va["depth"][only_a]).abs().max()))
        per_camera.append({
            "overlap_pixels": overlap,
            "covered_pixels_a": int((va["alpha"] > overlap_alpha_tol).sum()),
            "covered_pixels_b": int((vb["alpha"] > overlap_alpha_tol).sum()),
            "alpha_composition_defect": composition,
            "union_contributors_max": int(vu["n_contrib"].max()),
        })

    disjoint = overlap_pixels_total == 0
    status = PASSED if disjoint else UNRESOLVED
    return {
        "label": label,
        "status": status,
        "num_gaussians_a": int(state_a.mu.shape[0]),
        "num_gaussians_b": int(state_b.mu.shape[0]),
        "overlap_pixels_total": overlap_pixels_total,
        "max_alpha_gain_over_single_source": worst_alpha_gain,
        "max_depth_shift_where_only_a_is_present": worst_depth_shift,
        "per_camera": per_camera,
        "reason": (
            "No pixel receives coverage from both sources, so concatenating "
            "the populations composites exactly as one scene."
            if disjoint else
            f"{overlap_pixels_total} pixels receive coverage from BOTH "
            "sources. Alpha compositing gives 1-(1-a_1)(1-a_2) there, so a "
            "surface represented twice reads as more opaque and its "
            "normalised depth shifts toward the duplicate. Nothing in the "
            "available theory says which of the two representations is the "
            "physical surface, so this is reported rather than resolved."),
    }


def write_audit(out_dir, entries, *, sources: dict | None = None) -> dict:
    """Persist representation_audit.json with an overall status."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    statuses = [e["status"] for e in entries]
    if not entries:
        overall = NOT_RUN
    elif FAILED in statuses:
        overall = FAILED
    elif UNRESOLVED in statuses:
        overall = UNRESOLVED
    else:
        overall = PASSED
    payload = {
        "status": overall,
        "entries": entries,
        "sources": sources or {},
        "method": "unchanged CUDA renderer, unchanged source attributes",
        "forbidden_and_absent": [
            "eta", "optical thickness", "alpha_eff", "ownership logits",
            "blending weights", "overlap thresholds",
            "nearest-neighbour deduplication", "opacity rescaling"],
        "note": ("A diagnostic, not a fusion method. PASSED on a disjoint "
                 "synthetic fixture does not license describing a real "
                 "overlapping candidate population as a solved scene."),
    }
    tmp = out_dir / "representation_audit.json.tmp"
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out_dir / "representation_audit.json")
    return payload
