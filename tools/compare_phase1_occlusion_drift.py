"""Do Phase-1 Gaussians move while something that is not the object covers them?
Read-only, CPU, evaluation only (no GT; nothing here feeds back into any stage).

For each scene, ONE fixed set of "static" Gaussians is used for every run compared: the
Gaussians outside the Phase-2 support of EVERY run given (so a run is not judged on a
different set because its support differs). For each such Gaussian and frame, its
CANONICAL position x0 is classified per camera with the observed mask M and depth d
(tau = 2 cm, evaluation threshold only):
    SEEN      M = 1 and |d - z| <= tau             on the observed object surface
    OCCLUDED  M = 0, d valid and d < z - tau       something that is not the object is in front
A Gaussian-frame is "occluded" when it is OCCLUDED in every camera where it projects
inside the image and SEEN in none; "seen" when SEEN in at least one camera.
Reported per run: per-step Phase-1 motion |x_t - x_{t-1}| when seen vs occluded, drift
gained over each occluded interval, final drift |x_T - x_0| of static Gaussians that were
ever occluded vs never, and (runs with phase1_observability/) the fraction of the
occluded Gaussian-frames the filter itself marked prediction-only.

  python tools/compare_phase1_occlusion_drift.py --run draw1=<root> --run occ=<root> \
      --scenes "robot_dog wooden_box" --out <dir>
<root> is a run root laid out as <root>/<name>/unanimous/pipeline/<name>/bidirectional/<dir>/.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary  # noqa: E402

REAL = Path("/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real")
SHADOW = Path.home() / "real_poses_before_20261112"
DIRECTION = dict(matchbox="forward", wooden_box="forward", black_laptop="forward", robot_dog="forward",
                 orange_suitecase="forward", light_blue_box="forward", black_box="backward", red_fridge="forward")
TAU = 0.02


def bidi(root: Path, name: str) -> Path:
    return root / name / "unanimous" / "pipeline" / name / "bidirectional"


def support(root: Path, name: str, d: str):
    for f in (bidi(root, name) / f"{d}_rigid" / "support_coherence" / "support.json",):
        if f.is_file():
            return set(json.load(open(f))["members"])
    f = bidi(root, name) / d / "predicted_support.json"
    if f.is_file():
        return set(json.load(open(f))["dynamic_gaussians"])
    return None


def classify(sc, x0, fids):
    name = f"final_{sc}_episode_000000"
    src = (REAL if sc == "orange_suitecase" else SHADOW) / name
    cams = ["cam_000", "cam_001", "cam_002"] if sc == "orange_suitecase" else ["cam_001", "cam_002"]
    ext = {v.name.split("/")[0]: v for v in read_extrinsics_binary(str(src / "colmap/sparse/0/images.bin")).values()}
    intr = read_intrinsics_binary(str(src / "colmap/sparse/0/cameras.bin"))
    T, N = len(fids), x0.shape[0]
    seen = np.zeros((T, N), bool); occ_any = np.zeros((T, N), bool); notocc = np.zeros((T, N), bool)
    for c in cams:
        e = ext[c]; K = intr[e.camera_id]; fx, fy, cx, cy = K.params[:4]; W, H = K.width, K.height
        R = qvec2rotmat(e.qvec); t = e.tvec
        pc = x0 @ R.T + t; z = pc[:, 2]
        u = np.round(fx * pc[:, 0] / z + cx).astype(int); v = np.round(fy * pc[:, 1] / z + cy).astype(int)
        ins = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        uu, vv = np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
        for k, f in enumerate(fids):
            M = np.array(Image.open(src / "mask" / c / f"frame_{f:06d}.png")) > 0
            if M.ndim == 3:
                M = M[..., 0]
            D = np.load(src / "depth_npy" / c / f"frame_{f:06d}.npy")
            m, dd = M[vv, uu], D[vv, uu]
            val = np.isfinite(dd) & (dd > 0)
            seen[k] |= ins & m & val & (np.abs(dd - z) <= TAU)
            o = ins & ~m & val & (dd < z - TAU)
            occ_any[k] |= o; notocc[k] |= ins & ~o
    return seen, occ_any & ~notocc & ~seen


def stats(X, x0, static, seen, occluded, obs_dir, fids):
    T = X.shape[0]
    step = np.zeros(X.shape[:2]); step[1:] = np.linalg.norm(X[1:] - X[:-1], axis=2) * 1e3
    drift = np.linalg.norm(X - x0[None], axis=2) * 1e3
    st = np.repeat(static[None], T, 0); st[0] = False
    a, b = step[st & seen], step[st & occluded]
    ever = (occluded & st).any(0)
    gain = []
    for i in np.where(ever)[0]:
        o = occluded[:, i]; k = 1
        while k < T:
            if o[k]:
                j = k
                while j < T and o[j]:
                    j += 1
                gain.append(drift[min(j, T - 1), i] - drift[k - 1, i]); k = j
            else:
                k += 1
    gain = np.array(gain) if gain else np.zeros(1)
    q = lambda x, p: float(np.percentile(x, p)) if x.size else None          # noqa: E731
    r = {"step_mm_seen_median": q(a, 50), "step_mm_seen_p90": q(a, 90),
         "step_mm_occluded_median": q(b, 50), "step_mm_occluded_p90": q(b, 90),
         "drift_gain_per_occlusion_mm_median": q(gain, 50), "drift_gain_p90": q(gain, 90),
         "final_drift_mm_ever_occluded": q(drift[-1, ever & static], 50) if (ever & static).any() else None,
         "final_drift_mm_never_occluded": q(drift[-1, static & ~ever], 50),
         "static_ever_occluded": int((ever & static).sum()), "occluded_gaussian_frames": int((st & occluded).sum())}
    if obs_dir is not None and Path(obs_dir).is_dir():
        po = np.zeros(X.shape[:2], bool); have = np.zeros(T, bool)
        for k, f in enumerate(fids):
            p = Path(obs_dir) / f"obs_{f:06d}.npz"
            if p.is_file():
                po[k] = np.load(p)["q_max"] < 1e-3; have[k] = True
        m = st & occluded & have[:, None]
        r["occluded_frames_marked_prediction_only"] = float(po[m].mean()) if m.any() else None
        m2 = st & seen & have[:, None]
        r["seen_frames_marked_prediction_only"] = float(po[m2].mean()) if m2.any() else None
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=ROOT")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [(r.split("=", 1)[0], Path(r.split("=", 1)[1])) for r in a.run]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sc in a.scenes.split():
        name, d = f"final_{sc}_episode_000000", DIRECTION[sc]
        have = [(rn, root) for rn, root in runs if (bidi(root, name) / d / "phase1_centers").is_dir()]
        if not have:
            print(f"{sc}: no run has phase1_centers; skipped"); continue
        cdir = bidi(have[0][1], name) / d / "phase1_centers"
        x0 = np.load(cdir / "canonical_xyz.npy")
        sups = [support(root, name, d) for _, root in have]
        static = np.ones(x0.shape[0], bool)
        for s in sups:
            if s is not None:
                static[list(s)] = False
        files = sorted(glob.glob(str(cdir / "centers_frame_*.npy")))
        fids = [int(Path(f).stem.split("_")[-1]) for f in files]
        seen, occluded = classify(sc, x0, fids)
        for rn, root in have:
            c = bidi(root, name) / d / "phase1_centers"
            fs = sorted(glob.glob(str(c / "centers_frame_*.npy")))
            if [int(Path(f).stem.split("_")[-1]) for f in fs] != fids:
                print(f"{sc}/{rn}: different frame set; skipped"); continue
            if not np.allclose(np.load(c / "canonical_xyz.npy"), x0):
                print(f"{sc}/{rn}: different canonical model; skipped"); continue
            X = np.stack([np.load(f) for f in fs])
            r = {"scene": sc, "run": rn, "static_gaussians": int(static.sum()),
                 **stats(X, x0, static, seen, occluded, bidi(root, name) / d / "phase1_observability", fids)}
            rows.append(r); print(json.dumps(r), flush=True)
    (out / "occlusion_drift.json").write_text(json.dumps(rows, indent=1))
    f = lambda x: "-" if x is None else f"{x:.3g}"                                # noqa: E731
    lines = ["# Phase-1 motion of static Gaussians while covered by a non-object", "",
             "Static set = outside the Phase-2 support of every run compared. Per-step motion and drift in mm.", "",
             "| scene | run | static ever covered | step seen med/p90 | step covered med/p90 | drift gained per cover med/p90 "
             "| final drift: ever covered / never | covered frames marked prediction-only |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scene']} | {r['run']} | {r['static_ever_occluded']} | {f(r['step_mm_seen_median'])} / "
                     f"{f(r['step_mm_seen_p90'])} | {f(r['step_mm_occluded_median'])} / {f(r['step_mm_occluded_p90'])} | "
                     f"{f(r['drift_gain_per_occlusion_mm_median'])} / {f(r['drift_gain_p90'])} | "
                     f"{f(r['final_drift_mm_ever_occluded'])} / {f(r['final_drift_mm_never_occluded'])} | "
                     f"{f(r.get('occluded_frames_marked_prediction_only'))} |")
    (out / "occlusion_drift.md").write_text("\n".join(lines) + "\n")
    print(f"-> {out / 'occlusion_drift.md'}")


if __name__ == "__main__":
    main()
