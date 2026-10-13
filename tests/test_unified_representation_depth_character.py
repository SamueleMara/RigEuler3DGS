"""Projected overlap is not duplication: the gate has to tell them apart.

Section 5 of CODEX_NEXT_AFTER_F80E3E1_COMPLETE_ARTICULATED_PIPELINE.md.

The earlier gate counted pixels where both sources have alpha above a floor
and called any nonzero intersection unresolved. That overstates its case, and
the directive is right to say so: two perfectly valid parts at different
depths -- one simply behind the other -- cover the same pixels too. A count
of shared pixels cannot distinguish

    the same surface represented twice        |D_a - D_b| <~ sigma_D
    two surfaces, one occluding the other     |D_a - D_b| >> sigma_D

and only the second of those is an ordinary scene.

`depth_character` makes the distinction, and this file makes it falsifiable:
a deliberately LAYERED fixture, where the two sources are 0.6 m apart along
the view ray, must come back DEPTH_SEPARATED_LAYERS. If it did not, the
DEPTH_AGREEING_DUPLICATES verdict on the real Washer scene would be
unfalsifiable and worth nothing.

Both still return UNRESOLVED_SCENE_REPRESENTATION, and deliberately: a
layered overlap is not evidence that the two mixtures agree everywhere a
camera does not look. The audit refuses to turn the absence of one kind of
evidence into the presence of another.

    python tests/test_unified_representation_depth_character.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CHECKS = 0
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}: {detail or 'failed'}")
        print(f"  FAIL  {label}  {detail}")
    else:
        print(f"  ok    {label}  {detail}")


def cuda_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import diff_gaussian_rasterization  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    if not cuda_ready():
        print("depth character NOT_RUN: CUDA or diff_gaussian_rasterization "
              "unavailable.")
        return 0

    from bidirectional_iekf.representation_audit import (
        DISJOINT, DUPLICATES, LAYERED, MIXED, PASSED, UNRESOLVED, audit_pair)
    from estimator.manifold import so3_exp
    from estimator.state import GaussianMotionState
    from tests.test_unified_cuda_renderer_jvp_vjp import make_cameras

    # One camera, on the +z axis looking toward -z, so "behind" is simply a
    # more negative z and the geometry needs no interpretation.
    cameras = make_cameras(n_cam=1, H=96, W=96, radius=2.2)
    SIGMA, K = 0.02, 3.0

    def cloud(xy, z, n=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        offsets = 0.14 * (torch.rand(n, 2, generator=g) - 0.5)
        mu = torch.zeros(n, 3)
        mu[:, 0] = xy[0] + offsets[:, 0]
        mu[:, 1] = xy[1] + offsets[:, 1]
        mu[:, 2] = z
        return GaussianMotionState(
            mu=mu.cuda(), R=so3_exp(torch.zeros(n, 3)).cuda(),
            v=torch.zeros(n, 3).cuda(), omega=torch.zeros(n, 3).cuda(),
            colors=torch.rand(n, 3, generator=g).cuda(),
            opacity=torch.full((n, 1), 0.95).cuda(),
            scales=torch.tensor([[0.09, 0.06, 0.05]]).expand(n, 3)
            .contiguous().cuda())

    print("[1] genuinely disjoint: no shared pixel at all")
    left, right = cloud((-0.45, 0.0), 0.0, seed=1), cloud((0.45, 0.0), 0.0, seed=2)
    e = audit_pair(left, right, cameras, label="disjoint",
                   depth_sigma=SIGMA, k_sigma=K)
    check("PASSED and DISJOINT",
          e["status"] == PASSED and e["overlap_character"] == DISJOINT,
          f"{e['status']} / {e['overlap_character']}, "
          f"{e['overlap_pixels_total']} shared pixels")
    check("but the reason says what it does NOT prove",
          "not a proof of global nonredundancy" in e["reason"],
          "no overlap in the sampled cameras is not nonredundancy")

    print("\n[2] the same surface twice: a duplicate")
    e = audit_pair(left, left, cameras, label="identical",
                   depth_sigma=SIGMA, k_sigma=K)
    check("UNRESOLVED and DEPTH_AGREEING_DUPLICATES",
          e["status"] == UNRESOLVED and e["overlap_character"] == DUPLICATES,
          f"{e['status']} / {e['overlap_character']}, "
          f"{100 * e['depth_agreeing_fraction']:.1f}% of "
          f"{e['overlap_pixels_total']} shared pixels agree in depth")
    check("and the depth gap really is at zero",
          e["per_camera"][0]["depth"]["depth_gap_p99_m"] < SIGMA,
          f"p99 gap {e['per_camera'][0]['depth']['depth_gap_p99_m']:.3e} m "
          f"against a {K:g} x {SIGMA:g} m threshold")

    print("\n[3] THE NEGATIVE CONTROL: two layers, one behind the other")
    front = cloud((0.0, 0.0), 0.0, seed=3)
    back = cloud((0.0, 0.0), -0.6, seed=4)
    e = audit_pair(front, back, cameras, label="layered",
                   depth_sigma=SIGMA, k_sigma=K)
    d = e["per_camera"][0]["depth"]
    print(f"       {e['overlap_pixels_total']} shared pixels; depth gap "
          f"median {d['depth_gap_median_m']:.4f} m, p99 "
          f"{d['depth_gap_p99_m']:.4f} m, max {d['depth_gap_max_m']:.4f} m")
    check("the same pixels are shared, so a pixel count cannot tell them apart",
          e["overlap_pixels_total"] > 0,
          f"{e['overlap_pixels_total']} pixels -- a count alone would have "
          "called this a duplicate")
    check("but the character is DEPTH_SEPARATED_LAYERS",
          e["overlap_character"] == LAYERED,
          f"{e['overlap_character']}, only "
          f"{100 * e['depth_agreeing_fraction']:.1f}% agree in depth. This is "
          "the check that makes the real scene's verdict falsifiable")
    check("the measured gap is the geometry we built",
          abs(d["depth_gap_median_m"] - 0.6) < 0.05,
          f"{d['depth_gap_median_m']:.4f} m against the 0.600 m we placed")
    check("and it is STILL not a PASS",
          e["status"] == UNRESOLVED,
          "a layered overlap says nothing about surfaces no camera resolves; "
          "the audit will not read the absence of one kind of evidence as "
          "the presence of another")

    print("\n[4] both kinds at once")
    mixed_b = GaussianMotionState(
        mu=torch.cat([back.mu, front.mu], 0),
        R=torch.cat([back.R, front.R], 0),
        v=torch.cat([back.v, front.v], 0),
        omega=torch.cat([back.omega, front.omega], 0),
        colors=torch.cat([back.colors, front.colors], 0),
        opacity=torch.cat([back.opacity, front.opacity], 0),
        scales=torch.cat([back.scales, front.scales], 0))
    e = audit_pair(front, mixed_b, cameras, label="mixed",
                   depth_sigma=SIGMA, k_sigma=K)
    print(f"       character {e['overlap_character']}, "
          f"{100 * e['depth_agreeing_fraction']:.1f}% agreeing")
    check("a population holding a duplicate AND a layer is not read as a layer",
          e["overlap_character"] in (DUPLICATES, MIXED)
          and e["status"] == UNRESOLVED,
          f"{e['overlap_character']}: the duplicate in it dominates the "
          "nearest-surface depth, which is what the renderer composites")

    print("\n[5] the threshold is the declared sensor scale, and it is recorded")
    tight = audit_pair(front, back, cameras, label="layered",
                       depth_sigma=SIGMA, k_sigma=K)
    loose = audit_pair(front, back, cameras, label="layered",
                       depth_sigma=1.0, k_sigma=K)
    check("widening sigma_D to 1 m turns the same layer into a 'duplicate'",
          tight["overlap_character"] == LAYERED
          and loose["overlap_character"] == DUPLICATES,
          f"{tight['overlap_character']} at sigma 0.02 m, "
          f"{loose['overlap_character']} at sigma 1.0 m -- which is why the "
          "value used is the production depth model's own a = 0.02 and is "
          "written into every report, not chosen per scene")
    check("and every entry carries the sigma it was judged at",
          tight["depth_sigma_m"] == SIGMA and tight["k_sigma"] == K
          and "CENTER_ONLY_FIXED_ORIENTATION" in tight["approximation"],
          "including the fact that a centres-only transport is an "
          "approximation, since a splat's footprint depends on its "
          "covariance orientation too")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"{CHECKS} checks passed (real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
