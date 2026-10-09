#!/usr/bin/env python
"""Sections 39, 63-67: run the dual differential/finite chain on a real scene.

CODEX_IMPLEMENT_DUAL_DIFFERENTIAL_FINITE_JOINT_DISCOVERY.md.

The question this exists to answer is section 39's, and it needs both halves
of a finished run to answer it:

    Does Phase 1 contain coherent angular motion, or does it appear only
    after Phase 2?

Phase-1 supernode tracks come from `bootstrap_motion.csv` -- the displacement
field the multiview gradients asked for BEFORE any rigid projection -- and the
Phase-2 finite rotations come from `rigid_transforms.pt`. Comparing the
angular velocity implied by each is the only instrument that distinguishes a
rotation that was measured from one that was manufactured downstream.

Nothing here is scene-specific (section 62) and no ground truth enters the
inference (section 61); the GT line at the end is printed after the verdict
and plays no part in it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from debug.dual_joint_artifacts import (  # noqa: E402
    joint_type_evidence_payload, write_dual_joint_artifacts)
from estimator.manifold import project_to_so3  # noqa: E402
from joint_reduction.differential_finite_consistency import (  # noqa: E402
    attribute_angular_origin, compare_angular_velocity,
    incremental_spatial_omega)
from joint_reduction.direct_gaussian_joint import (  # noqa: E402
    dual_differential_finite_evidence, geometric_joint_candidate)
from tools.iekf_artifacts import ArtifactWriter  # noqa: E402

DEFAULT_DT = 1.0 / 30.0


def load_phase1_tracks(root: Path, part: int = 1):
    """Phase-1 supernode tracks: x_s(t) = c_s(0) + d_s(t).

    `bootstrap_motion.csv` stores the displacement of each supernode at each
    frame relative to frame 0, which is Phase-1's own motion evidence with no
    rigid model imposed on it. `supernode_motion.csv` supplies the frame-0
    centres and the part label each supernode was finally assigned.
    """
    sm = root / "debug_numeric" / "supernode_motion.csv"
    bm = root / "debug_numeric" / "bootstrap_motion.csv"
    for p in (sm, bm):
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} is missing. Re-run that scene with --debug-numerics so "
                f"the Phase-1 motion evidence is persisted:\n"
                f"  python tools/run_all_datasets.py --only <scene> "
                f"--debug-numerics --out-root <root>")

    centres, labels = {}, {}
    for r in csv.DictReader(sm.open()):
        s = int(r["supernode"])
        centres[s] = [float(r["cx0"]), float(r["cy0"]), float(r["cz0"])]
        labels[s] = int(r["label"])

    frames: dict[int, dict[int, list]] = {}
    stored: dict[int, dict[int, list]] = {}
    for r in csv.DictReader(bm.open()):
        t, s = int(r["frame"]), int(r["supernode"])
        frames.setdefault(t, {})[s] = [float(r["dx"]), float(r["dy"]),
                                       float(r["dz"])]
        try:
            stored.setdefault(t, {})[s] = [float(r["vx"]), float(r["vy"]),
                                           float(r["vz"])]
        except (KeyError, ValueError):
            pass

    members = sorted(s for s, lab in labels.items()
                     if (lab == part if part >= 0 else lab != 0))
    if len(members) < 6:
        raise ValueError(f"{root}: part {part} has only {len(members)} "
                         f"supernodes; the affine velocity field of section 10 "
                         f"needs enough spread to determine G")
    T = max(frames) + 1
    x0 = torch.tensor([centres[s] for s in members], dtype=torch.float64)
    tracks = torch.stack([
        x0 + torch.tensor([frames[t].get(s, [0.0, 0.0, 0.0]) for s in members],
                          dtype=torch.float64)
        for t in range(T)])
    vel = None
    if stored and all(len(stored.get(t, {})) >= len(members) for t in range(T)):
        vel = torch.stack([
            torch.tensor([stored[t][s] for s in members], dtype=torch.float64)
            for t in range(T)])
    return {"x0": x0, "tracks": tracks, "stored_velocity": vel,
            "members": members, "num_frames": T,
            "labels": {s: labels[s] for s in members}}


def phase2_finite_omega(root: Path, part: int, dt: float):
    """omega^finite from the Phase-2 rigid trajectory, for section 39."""
    tf = root / "debug_numeric" / "rigid_transforms.pt"
    if not tf.is_file():
        return None
    T4 = torch.load(tf, map_location="cpu", weights_only=False).double()
    idx = min(max(part - 1, 0), T4.shape[0] - 1)
    R = project_to_so3(T4[idx, :, :3, :3].contiguous()).detach()
    return incremental_spatial_omega(R, dt=dt)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a finished run directory")
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)
    ap.add_argument("--confidence", type=float, default=0.95,
                    help="section 6: one confidence for the whole run")
    ap.add_argument("--velocity-source", choices=("central", "stored"),
                    default="central",
                    help="section 8: 'central' applies the second-order "
                         "operator to the stored positions; 'stored' uses the "
                         "velocity column, which is a FIRST-order backward "
                         "difference of those same positions and so carries "
                         "no extra information")
    ap.add_argument("--out", default=None,
                    help="artifact directory (default <run>/debug_numeric/"
                         "dual_joint)")
    args = ap.parse_args()

    root = Path(args.run)
    ph1 = load_phase1_tracks(root, args.part)
    print(f"Phase-1 tracks: {len(ph1['members'])} supernodes x "
          f"{ph1['num_frames']} frames from {root}")

    vel = ph1["stored_velocity"] if args.velocity_source == "stored" else None
    ev = dual_differential_finite_evidence(ph1["x0"], ph1["tracks"],
                                           dt=args.dt, velocity=vel)
    cand = geometric_joint_candidate(ev, confidence=args.confidence,
                                     x0=ph1["x0"], dt=args.dt)

    a, b, c = cand["stage_a_rigid"], cand["stage_b_one_dof"], cand["stage_c_angular"]
    print(f"\n  A rigid compatibility  : {'pass' if a['passes'] else 'FAIL'}"
          f"   median chi2(rigid vs affine) "
          f"{a['median_chi2_rigid_vs_affine']:.3e} vs cut {a['cut']:.2f}"
          f"   (median Q_D {a['median_Q_D']:.3e})")
    print(f"  B one-DoF compatibility: {'pass' if b['passes'] else 'FAIL'}"
          f"   chi2_rank1 {b['chi2_rank1']:.3e} over dof {b['dof']} "
          f"-> {b['per_dof']:.3e} per dof")
    print(f"  C angular evidence     : omega!=0 {c['angular_present']}"
          f"   chi2(omega=0) {c['chi2_omega_zero']:.3e} vs cut {c['cut']:.2f}"
          f"   one axis {c['angular_shares_one_axis']}")

    # ---- section 39: Phase-1 versus Phase-2 angular motion ---------------
    p2 = phase2_finite_omega(root, args.part, args.dt)
    section39 = None
    if p2 is not None and p2["omega"].numel():
        om_diff_mid = 0.5 * (ev["twist"]["omega"][1:] + ev["twist"]["omega"][:-1])
        om_cov = torch.stack([f["omega_covariance"]
                              for f in ev["twist"]["fits"]])
        # The Phase-2 transforms carry no covariance of their own here, so the
        # comparison is normalised by the Phase-1 uncertainty alone. That is
        # the conservative direction: it can only make the two look MORE
        # discrepant than a full treatment would.
        cmp2 = compare_angular_velocity(
            om_diff_mid, p2["omega"],
            covariance_diff=0.25 * (om_cov[1:] + om_cov[:-1]))
        info_mid = 0.5 * (ev["omega_information"][1:]
                          + ev["omega_information"][:-1])
        section39 = attribute_angular_origin(
            cmp2, strain_statistic=ev["strain_statistic"],
            omega_information=info_mid)
        print(f"\n  section 39 -- Phase-1 velocities vs Phase-2 transforms")
        print(f"    median |omega^diff|   (Phase 1) "
              f"{section39['median_omega_diff']:.4e} rad/s")
        print(f"    median |omega^finite| (Phase 2) "
              f"{section39['median_omega_finite']:.4e} rad/s")
        print(f"    median axis cosine {section39['median_axis_cosine']:.4f}"
              f"   median Mahalanobis {cmp2['median_mahalanobis']:.3e}"
              f" vs chi2(3) cut {section39['agreement_cut']:.2f}")
        print(f"    axes agree: {section39['axes_agree']}  "
              f"({section39['agreement_basis']})")
        print(f"    -> {section39['attribution']}")
        print(f"       {section39['reason']}")

    print(f"\n  geometric candidate: {cand['status']}  "
          f"(candidate={cand['candidate']})")
    print(f"    {cand['reason']}")

    out = Path(args.out) if args.out else (root / "debug_numeric" / "dual_joint")
    out.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(out)
    write_dual_joint_artifacts(writer, ev, cand)
    payload = joint_type_evidence_payload(ev, cand)
    payload["section_39_phase1_vs_phase2"] = (
        None if section39 is None else
        {k: (v if not torch.is_tensor(v) else None)
         for k, v in section39.items() if k != "chi2_omega_diff"})
    payload["velocity_source"] = args.velocity_source
    payload["velocity_source_note"] = (
        "section 8: the stored velocity column is a first-order backward "
        "difference of the stored positions, so the second-order central "
        "operator on those positions is used by default")
    payload["num_supernodes"] = len(ph1["members"])
    payload["num_frames"] = ph1["num_frames"]
    payload["dt"] = args.dt
    json.dump(payload, (out / "joint_type_evidence.json").open("w"), indent=1,
              default=str)
    print(f"\n  sections 43-48 artifacts -> {out}")

    gt = root / "dataset_joint_ground_truth.json"
    if gt.is_file():
        try:
            t = json.load(gt.open()).get("joint_type_raw")
            t = {"continuous": "revolute"}.get(t, t)
            print(f"\n  [post-inference only] ground truth {t}, "
                  f"geometric candidate {cand['candidate']}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
