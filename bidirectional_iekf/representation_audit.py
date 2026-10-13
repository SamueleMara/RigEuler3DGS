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


DISJOINT = "DISJOINT"
LAYERED = "DEPTH_SEPARATED_LAYERS"
DUPLICATES = "DEPTH_AGREEING_DUPLICATES"
MIXED = "MIXED"


def depth_character(view_a, view_b, both, *, depth_sigma: float,
                    k_sigma: float) -> dict:
    """At doubly covered pixels, are the two sources at the SAME depth?

    This is the distinction a pixel-intersection count cannot make, and
    without it the gate overstates its case. Two independently trained
    mixtures can cover one pixel for two completely different reasons:

      the same surface represented twice          |D_a - D_b| <~ sigma_D
      two surfaces at different depths, one
      behind the other, which is an ordinary
      and entirely valid scene                    |D_a - D_b| >> sigma_D

    `sigma_D` is the declared sensor noise scale, not a tuned cutoff: it is
    the production depth model's own constant (a = 0.02 m), and `k_sigma`
    says how many of them count as "the same place". Both are recorded.

    D here is the renderer's normalised depth Z/(A+eps), which is the
    quantity the measurement model compares against the dataset's metric Z,
    so a difference in it is a difference in metres wherever alpha is high.
    """
    import torch as _torch

    count = int(both.sum())
    if not count:
        return {"overlap_pixels": 0, "character": DISJOINT}
    gap = (view_a["depth"][both] - view_b["depth"][both]).abs()
    threshold = k_sigma * depth_sigma
    agreeing = int((gap <= threshold).sum())
    separated = count - agreeing
    quantiles = _torch.tensor([0.5, 0.9, 0.99], device=gap.device,
                              dtype=gap.dtype)
    q = _torch.quantile(gap.double(), quantiles.double()).tolist()
    fraction = agreeing / count
    if fraction >= 0.9:
        character = DUPLICATES
    elif fraction <= 0.1:
        character = LAYERED
    else:
        character = MIXED
    return {
        "overlap_pixels": count,
        "depth_agreeing_pixels": agreeing,
        "depth_separated_pixels": separated,
        "depth_agreeing_fraction": fraction,
        "depth_gap_median_m": q[0], "depth_gap_p90_m": q[1],
        "depth_gap_p99_m": q[2],
        "depth_gap_max_m": float(gap.max()),
        "depth_sigma_m": depth_sigma, "k_sigma": k_sigma,
        "threshold_m": threshold,
        "character": character,
        "what_it_means": {
            DUPLICATES: ("the two sources put a surface at the same depth at "
                         "these pixels, so the union represents it twice"),
            LAYERED: ("the two sources are at clearly different depths here, "
                      "which is an ordinary occluded layer and not a "
                      "duplicate"),
            MIXED: "both kinds of pixel are present in quantity",
        }[character],
    }


def observation_residuals(views, observed, *, eps: float = 1e-6) -> dict:
    """RMS image and depth residual against the real calibrated frame.

    Reported for A alone, B alone and A+B so they can be compared. A LOW
    union residual is NOT evidence that the union is one scene: a mixture
    with twice the Gaussians can fit an image better while representing the
    same surface twice, and section 5.4 is explicit that this must not be
    promoted to proof.
    """
    import torch as _torch

    if observed is None:
        return {}
    rgb_gt, depth_gt, valid = observed
    rgb = views["rgb"]
    depth = views["depth"]
    valid = valid.to(depth.device)
    covered = valid & (views["alpha"] > 0.05)
    out = {"rgb_rmse": float(((rgb - rgb_gt.to(rgb)) ** 2).mean().sqrt()),
           "covered_pixels": int(covered.sum())}
    if int(covered.sum()):
        d = depth[covered] - depth_gt.to(depth)[covered]
        out["depth_rmse_m"] = float((d ** 2).mean().sqrt())
        out["depth_mae_m"] = float(d.abs().mean())
    return out


def audit_pair(state_a, state_b, cameras, *, label: str,
               overlap_alpha_tol: float = 1e-3,
               depth_sigma: float = 0.02, k_sigma: float = 3.0,
               observations=None) -> dict:
    """Render A, B and A+B, and report whether the union is one scene.

    The pixel intersection alone answers a narrower question than it looks:
    it says the two sources are projected onto the same pixels, which two
    valid depth-separated parts also do. `depth_character` is what separates
    a duplicate from a layer, and the status below now depends on it.
    """
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
        row = {
            "overlap_pixels": overlap,
            "covered_pixels_a": int((va["alpha"] > overlap_alpha_tol).sum()),
            "covered_pixels_b": int((vb["alpha"] > overlap_alpha_tol).sum()),
            "alpha_composition_defect": composition,
            "union_contributors_max": int(vu["n_contrib"].max()),
            "depth": depth_character(va, vb, both, depth_sigma=depth_sigma,
                                     k_sigma=k_sigma),
        }
        if observations is not None:
            index = len(per_camera)
            observed = (observations[index]
                        if index < len(observations) else None)
            row["residual_vs_observation"] = {
                "initial_only": observation_residuals(va, observed),
                "final_only": observation_residuals(vb, observed),
                "union": observation_residuals(vu, observed),
                "caveat": ("a lower union residual is not evidence of a "
                           "unique representation; more Gaussians fit an "
                           "image better whether or not they duplicate a "
                           "surface"),
            }
        per_camera.append(row)

    disjoint = overlap_pixels_total == 0
    agreeing = sum(c["depth"].get("depth_agreeing_pixels", 0)
                   for c in per_camera)
    separated = sum(c["depth"].get("depth_separated_pixels", 0)
                    for c in per_camera)
    agreeing_fraction = agreeing / max(overlap_pixels_total, 1)
    if disjoint:
        character, status = DISJOINT, PASSED
    elif agreeing_fraction >= 0.9:
        character, status = DUPLICATES, UNRESOLVED
    elif agreeing_fraction <= 0.1:
        # Projected overlap that is entirely depth-separated is an ordinary
        # occluded layer. It still is not a PASS: nothing here shows the two
        # mixtures do not duplicate a surface somewhere no camera resolves,
        # and the audit refuses to turn the absence of one kind of evidence
        # into the presence of another.
        character, status = LAYERED, UNRESOLVED
    else:
        character, status = MIXED, UNRESOLVED
    return {
        "label": label,
        "status": status,
        "overlap_character": character,
        "depth_agreeing_pixels": agreeing,
        "depth_separated_pixels": separated,
        "depth_agreeing_fraction": agreeing_fraction,
        "depth_sigma_m": depth_sigma, "k_sigma": k_sigma,
        "approximation": ("CENTER_ONLY_FIXED_ORIENTATION when the caller "
                          "transported centres only; a splat's projected "
                          "footprint depends on its covariance orientation "
                          "as well as its centre"),
        "num_gaussians_a": int(state_a.mu.shape[0]),
        "num_gaussians_b": int(state_b.mu.shape[0]),
        "overlap_pixels_total": overlap_pixels_total,
        "max_alpha_gain_over_single_source": worst_alpha_gain,
        "max_depth_shift_where_only_a_is_present": worst_depth_shift,
        "per_camera": per_camera,
        "reason": (
            "No pixel receives coverage from both sources in the sampled "
            "views, so concatenating the populations composites as one scene "
            "THERE. Absence of projected overlap in a sampled set of cameras "
            "is not a proof of global nonredundancy."
            if disjoint else
            f"{overlap_pixels_total} pixels receive coverage from BOTH "
            f"sources, and {100 * agreeing_fraction:.1f}% of them have the "
            f"two normalised depths within {k_sigma:g} x {depth_sigma:g} m of "
            "each other. Projected overlap alone would not settle this -- two "
            "valid depth-separated parts also share pixels -- but agreement "
            "IN DEPTH is the two sources placing a surface at the same place. "
            "Alpha compositing then gives 1-(1-a_1)(1-a_2) and the normalised "
            "depth shifts toward the duplicate. Nothing in the available "
            "theory says which representation is the physical surface, so "
            "this is reported rather than resolved."),
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
