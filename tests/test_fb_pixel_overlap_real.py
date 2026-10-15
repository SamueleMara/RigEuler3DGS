#!/usr/bin/env python
"""Real-data checks of the F/B overlap diagnostic (storage_45135, washer_103776).

T2  stable identities and time mapping: endpoints, chronological order, row
    identity, canonical geometry, no reversal of the backward track.
T8  one camera gauge for both models (real calibrations).
T10 rendering parity at anchor / mid / end frames: the contributor records
    rebuild the production IEKF render's RGB, alpha and depth, and the
    production render matches the independent training rasterizer's RGB.
T12 a real run writes only to its own output directory and leaves every
    protected input byte-identical.
T13 the renders are labelled CENTER_ONLY_FIXED_ORIENTATION when only centres
    exist, and the report says so.

    python tests/test_fb_pixel_overlap_real.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RESULTS = Path(os.environ.get("RESULTS_ROOT",
                              "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"))
DATA = Path(os.environ.get("DATASET_ROOT",
                           "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/"
                           "preparation_8cams/partnet_subset_fixed_render_linear"))
SCENES = ("storage_45135", "washer_103776")

CHECKS, FAILURES = 0, []


def check(label, cond, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        FAILURES.append(label)


def t2_t8_t10(scene):
    from estimator.renderer_bridge import build_linearization
    from estimator.state import matrix_to_quaternion_wxyz
    from observation_diagnostics.renderer_access import pixel_contributions
    from observation_diagnostics.sources import SourceTrack, check_gauge, scene_paths
    from tools.run_iekf_scene import camera_specs

    print(f"\n[T2/T8/T10] {scene}")
    paths = scene_paths(RESULTS, DATA, scene)
    F = SourceTrack("F", paths["sources"]["F"], DATA / scene)
    B = SourceTrack("B", paths["sources"]["B"], DATA / scene)
    mF, mB = F.manifest, B.manifest
    check("T2 forward = initial endpoint at original frame 0",
          mF["canonical_endpoint"] == "initial" and mF["canonical_original_frame"] == 0)
    check("T2 backward = final endpoint at original frame 59",
          mB["canonical_endpoint"] == "final" and mB["canonical_original_frame"] == 59)
    po_F, po_B = mF["processing_order_original_indices"], mB["processing_order_original_indices"]
    check("T2 forward processed 0..59, backward processed 59..0, both SAVED chronologically",
          po_F == list(range(60)) and po_B == list(range(59, -1, -1))
          and F.frames == B.frames == list(range(60)))
    cF, cB = F.canonical_check, B.canonical_check
    check("T2 forward centres at frame 0 are the initial PLY", cF["centres_at_anchor_vs_ply_max_abs"] < 1e-5,
          f"{cF['centres_at_anchor_vs_ply_max_abs']:.1e}")
    check("T2 backward centres at frame 59 are the final PLY", cB["centres_at_anchor_vs_ply_max_abs"] < 1e-5,
          f"{cB['centres_at_anchor_vs_ply_max_abs']:.1e}")
    check("T2 canonical_xyz.npy is each PLY's geometry",
          cF["canonical_xyz_vs_ply_max_abs"] < 1e-5 and cB["canonical_xyz_vs_ply_max_abs"] < 1e-5)
    ply_B = B.base.mu.cpu().numpy()
    d0 = float(np.abs(B.mu[0] - ply_B).max())
    check("T2 backward frame 0 is NOT its endpoint geometry (no reversal of the time axis)",
          d0 > 1e-3 and float(np.abs(B.mu[59] - ply_B).max()) < 1e-5, f"|B(0) - PLY_final| max {d0:.3f}")
    check("T2 rows are PLY rows (original_gaussian_indices identity, no reorder)",
          F.index_note.startswith("identity") and B.index_note.startswith("identity"))
    g = check_gauge(F, B)
    check("T8 one calibration for both models", g["max_abs_extrinsic_or_projection_difference"] == 0.0,
          f"{g['max_abs_extrinsic_or_projection_difference']}")
    specs = camera_specs(F.cams_all, len(F.cams_all))
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    worst = {"A": 0.0, "D": 0.0, "C": 0.0, "train_rgb_max": 0.0, "train_rgb_mean": 0.0,
             "train_px_gt_1e-4": 0}
    for src in (F, B):
        for f in (0, 30, 59):
            for ci in (0, 4):
                st = src.state(f)
                v = build_linearization(st, [specs[ci]]).views[0]
                c = pixel_contributions(v.cache)
                pix, q = c.pixel, c.q.double()
                HW = c.H * c.W
                A = torch.zeros(HW, dtype=torch.float64, device="cuda").index_add_(0, pix, q)
                Z = torch.zeros_like(A).index_add_(0, pix, q * v.cache.depth[c.gid.long()].double())
                C = torch.zeros(3, HW, dtype=torch.float64, device="cuda")
                for ch in range(3):
                    C[ch].index_add_(0, pix, q * st.colors[c.gid.long(), ch].double())
                cov = v.alpha.reshape(-1) > 0.05
                worst["A"] = max(worst["A"], float((A - v.alpha.reshape(-1).double()).abs().max()))
                worst["D"] = max(worst["D"], float((Z / (A + v.cache.eps_depth)
                                                    - v.depth.reshape(-1).double())[cov].abs().max()))
                worst["C"] = max(worst["C"], float((C - v.rgb.reshape(3, -1).double()).abs().max()))
                cam = F.cams_all[ci]
                rs = GaussianRasterizationSettings(
                    image_height=specs[ci].image_height, image_width=specs[ci].image_width,
                    tanfovx=specs[ci].tanfovx, tanfovy=specs[ci].tanfovy,
                    bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
                    viewmatrix=specs[ci].viewmatrix, projmatrix=specs[ci].projmatrix, sh_degree=0,
                    campos=cam.camera_center.cuda(), prefiltered=False, debug=False,
                    antialiasing=False)
                means2D = torch.zeros_like(st.mu)
                with torch.no_grad():
                    rgb_t = GaussianRasterizer(rs)(
                        means3D=st.mu, means2D=means2D, opacities=st.opacity,
                        colors_precomp=st.colors, scales=st.scales,
                        rotations=matrix_to_quaternion_wxyz(st.R).contiguous())[0]
                diff = (rgb_t - v.rgb).abs()
                worst["train_px_gt_1e-4"] = max(worst["train_px_gt_1e-4"],
                                                int((diff.amax(0) > 1e-4).sum()))
                worst["train_rgb_max"] = max(worst["train_rgb_max"], float(diff.max()))
                worst["train_rgb_mean"] = max(worst["train_rgb_mean"], float(diff.mean()))
    check("T10 records rebuild the production alpha (12 views x 2 sources)", worst["A"] < 5e-6,
          f"max {worst['A']:.1e}")
    check("T10 records rebuild the production depth", worst["D"] < 5e-5, f"max {worst['D']:.1e} m")
    check("T10 records rebuild the production RGB", worst["C"] < 5e-6, f"max {worst['C']:.1e}")
    # Two kernels (__expf vs expf) can disagree on a contributor whose alpha sits
    # exactly at 1/255 or whose T sits at T_STOP; that flips single pixels, not images.
    check("T10 production IEKF render = independent training rasterizer (RGB): at most 3 "
          "boundary-flip pixels per 640x480 view above 1e-4",
          worst["train_px_gt_1e-4"] <= 3 and worst["train_rgb_mean"] < 1e-6,
          f"worst view {worst['train_px_gt_1e-4']} px > 1e-4, max {worst['train_rgb_max']:.1e}, "
          f"worst mean {worst['train_rgb_mean']:.1e}")
    return F, B


def t12_t13():
    from observation_diagnostics.sources import hash_files, protected_files, scene_paths
    print("\n[T12/T13] a real run is read-only and labels its motion state")
    scene = "storage_45135"
    paths = scene_paths(RESULTS, DATA, scene)
    prot = protected_files(paths)
    before = hash_files(prot)
    res_dir = RESULTS / scene

    def listing():
        return sorted((str(p.relative_to(res_dir)), p.stat().st_size, p.stat().st_mtime_ns)
                      for d in ("static_bidi", "bidirectional") for p in (res_dir / d).rglob("*")
                      if p.is_file())
    lst0 = listing()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run"
        cmd = [sys.executable, str(REPO / "tools" / "analyze_fb_pixel_overlap.py"), "--scene", scene,
               "--frames", "0,59", "--cameras", "cam_000", "--tag", "t12", "--mode", "all",
               "--out-dir", str(out), "--results-root", str(RESULTS), "--dataset-root", str(DATA)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        check("T12 the run completes", r.returncode == 0, r.stdout[-300:] + r.stderr[-300:])
        prov = json.loads((out / "provenance.json").read_text())
        check("T12 the run's own before/after hashes agree", prov.get("protected_files_changed") == [])
        r2 = subprocess.run(cmd, capture_output=True, text=True)
        check("T12 rerunning into the same tag without --resume is refused",
              r2.returncode != 0 and "--resume" in (r2.stdout + r2.stderr))
        r3 = subprocess.run(cmd + ["--resume"], capture_output=True, text=True)
        check("T12 --resume reuses every frame", r3.returncode == 0 and "2 reused" in r3.stdout
              and "already complete" in r3.stdout, r3.stdout[-200:])
        modes = {s["state_mode"] for s in prov["sources"].values()}
        check("T13 motion state labelled CENTER_ONLY_FIXED_ORIENTATION",
              modes == {"CENTER_ONLY_FIXED_ORIENTATION"}, str(modes))
        rep = (out / "report" / "RESULTS.md").read_text()
        check("T13 the report states it and its consequence",
              "CENTER_ONLY_FIXED_ORIENTATION" in rep and "orientation" in rep)
    after = hash_files(prot)
    check("T12 every protected file byte-identical (independent hash)",
          before == after, f"{len(prot)} files")
    check("T12 nothing written under static_bidi/ or bidirectional/", listing() == lst0)


def main() -> int:
    if not torch.cuda.is_available():
        print("NOT_RUN: CUDA unavailable")
        return 0
    for s in SCENES:
        if not (RESULTS / s / "bidirectional").is_dir():
            print(f"NOT_RUN: {RESULTS / s / 'bidirectional'} missing")
            return 0
    for s in SCENES:
        t2_t8_t10(s)
        torch.cuda.empty_cache()
    t12_t13()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"{CHECKS} checks passed (real data, real CUDA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
