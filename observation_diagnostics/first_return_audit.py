#!/usr/bin/env python
"""READ-ONLY first-return audit (Gate B) and stationary-window hazard calibration.

RigEuler3DGS_CODEX_Phase1_FirstReturn_Implementation_2026-11-05.md sections 2, 6 (Gate B).

  audit        for each requested frame: the model state (canonical endpoint, or the
               Phase-1 tracked centres/rotations of that frame) against that frame's
               masked RGB-D. Per camera: valid foreground pixels, modelled-hit /
               unknown / front-contradiction fractions, PNG maps (log L, gamma_U,
               front contradiction, best matched gamma), and the Gaussian ids with
               the largest summed front contradiction (full-ray, not a centre check).
               Parity: (i) the full contributor dump, truncated at T_STOP exactly as
               the production renderer, reproduces its (RGB, A, D); (ii) CUDA log L
               equals the float64 CPU oracle (estimator/first_return_reference.py)
               on sampled pixels.
  calibrate    the unknown-surface hazard lambda_U per camera by maximum likelihood of
               the FIXED canonical render against a window the caller asserts is
               stationary and that is NOT used by any evaluated motion frame. Writes
               a prior JSON (estimator/first_return_types.py) with status
               "calibrated" and the full likelihood curve as provenance.

    python observation_diagnostics/first_return_audit.py audit --model M --source S \\
        --sensor-model CAL --first-return-prior P --frames 90 119 133 145 160 190 --out DIR
    python observation_diagnostics/first_return_audit.py calibrate --model M --source S \\
        --sensor-model CAL --frames 60 79 --z-far 2.0 --out prior.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

RENDERER_NEAR_M = 0.2     # auxiliary.h in_frustum: p_view.z <= 0.2 is culled by the renderer


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _setup(args, start, end):
    """The production Phase-1 setup (tools/audit_calibrated_phase1.setup_window) on [start, end]."""
    from audit_calibrated_phase1 import setup_window
    ns = argparse.Namespace(**vars(args))
    ns.motion_start_frame, ns.motion_end_frame = int(start), int(end)
    for k, v in {"direction": "forward", "bootstrap_iters": 12, "iekf_supernode_size": 64,
                 "p0_geometry": None, "p0_velocity_sigma": (1.0, 0.316227766),
                 "observation_mode": "masked-rgb-expected-quad", "rgb_background_constant": None,
                 "rgb_background_distribution": None, "alpha_precision": "coupled", "alpha_sd": None,
                 "rgb_support": "full", "mask_source": None, "frames": None, "cameras": None,
                 "object_mask_constraint": "off", "first_return_mode": "off", "first_return_prior": None,
                 "phase1_partition": "canonical"}.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    ns.first_return_mode, ns.first_return_prior = "off", None     # the audit builds its own
    return setup_window(ns)


def _prep(ns, window, k, dev):
    from audit_calibrated_phase1 import _frame_obs
    return _frame_obs(ns, k, window, dev)


def _state_at(truth, phase1_dir, frame):
    if phase1_dir is None:
        return truth, "canonical endpoint"
    import dataclasses
    d = Path(phase1_dir)
    mu = torch.from_numpy(np.load(d / f"centers_frame_{frame:04d}.npy")).to(truth.mu)
    R = torch.from_numpy(np.load(d / f"rotations_frame_{frame:04d}.npy")).to(truth.R)
    return dataclasses.replace(truth, mu=mu, R=R), f"Phase-1 state {d}"


def _png(path, img, lo, hi, cmap="viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(str(path), np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1), cmap=cmap)


def renderer_parity(view, dump, colors, bg, eps_depth):
    """Composite the full dump with the production T_STOP rule; compare with the render."""
    H, W = view.depth.shape
    off = dump["offsets"].cpu().numpy(); gid = dump["gid"].cpu().numpy()
    al = dump["alpha"].double().cpu().numpy(); z = dump["z"].double().cpu().numpy()
    col = colors.double().cpu().numpy()
    A = np.zeros(H * W); Z = np.zeros(H * W); C = np.zeros((H * W, 3)); Tend = np.ones(H * W)
    for p in range(H * W):
        T = 1.0
        for e in range(off[p], off[p + 1]):
            a = al[e]
            if T * (1 - a) < 1e-4:
                break
            w = a * T
            A[p] += w; Z[p] += w * z[e]; C[p] += w * col[gid[e]]
            T *= (1 - a)
        Tend[p] = T
    C = C + Tend[:, None] * bg.double().cpu().numpy()[None]
    Dr = Z / (A + eps_depth)
    ref = {"alpha": view.alpha.double().cpu().numpy().reshape(-1), "depth": view.depth.double().cpu().numpy().reshape(-1),
           "rgb": view.rgb.double().cpu().numpy().reshape(3, -1).T}
    return {"alpha_max_abs": float(np.abs(A - ref["alpha"]).max()),
            "depth_max_abs_where_A>0.5": float(np.abs(Dr - ref["depth"])[A > 0.5].max()) if (A > 0.5).any() else 0.0,
            "rgb_max_abs": float(np.abs(C - ref["rgb"]).max())}


def oracle_parity(dump, D, valid, sd, su, logL, cam_prior, n=200, seed=0):
    """CUDA log L vs the float64 CPU oracle on sampled valid pixels."""
    from estimator.first_return_reference import first_return_ray
    ids = torch.nonzero(valid.reshape(-1)).reshape(-1).cpu()
    if ids.numel() == 0:
        return {"pixels": 0}
    g = torch.Generator().manual_seed(seed)
    ids = ids[torch.randperm(ids.numel(), generator=g)[:n]]
    off = dump["offsets"].cpu(); al = dump["alpha"].double().cpu(); z = dump["z"].double().cpu()
    Dv = D.reshape(-1).double().cpu(); sdv = sd.reshape(-1).double().cpu(); suv = su.reshape(-1).double().cpu()
    errs = []
    for p in ids.tolist():
        zz, aa = z[off[p]:off[p + 1]], al[off[p]:off[p + 1]]
        keep = (zz > cam_prior.z_near) & (zz <= cam_prior.z_far)
        zz, aa = zz[keep], aa[keep]
        r = first_return_ray(Dv[p], zz, aa, torch.full_like(zz, float(sdv[p])), z_near=cam_prior.z_near,
                             z_far=cam_prior.z_far, lambda_unknown=cam_prior.lam, sigma_unknown=float(suv[p]))
        errs.append(abs(float(r.log_valid_depth_likelihood) - float(logL.reshape(-1)[p])))
    e = np.array(errs)
    return {"pixels": int(e.size), "max_abs_logL_err": float(e.max()), "median_abs_logL_err": float(np.median(e))}


@torch.no_grad()
def run_audit(a) -> int:
    from estimator.first_return_observation import FirstReturnObservation, sigma_maps
    from estimator.first_return_types import load_prior
    from estimator import first_return_operators as fo
    from estimator.renderer_bridge import build_linearization
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in a.frames:
        truth, cameras, window, cfg, ns, _ = _setup(a, f, f)
        prior = load_prior(a.first_return_prior, list(cfg.camera_names))
        obs_model = FirstReturnObservation("audit", prior)
        state, what = _state_at(truth, a.phase1_dir, f)
        dev = state.mu.device
        bg = ns.masked_observation.model.background_tensor(dev)
        lin = build_linearization(state, cameras, eps_depth=cfg.eps_depth, background=bg)
        obs = _prep(ns, window, 0, dev)
        fr = obs_model.evaluate(lin, obs, cfg, diagnostics=True)
        row = {"frame": f, "state": what, **fr.summary(top=a.top)}
        par = []
        for c, (cam, view, ob) in enumerate(zip(fr.cams, lin.views, obs)):
            maps = cam.diag["_maps"]
            for k, (lo, hi, cm) in {"logL": (-10.0, 6.0, "viridis"), "gamma_unknown": (0, 1, "magma"),
                                    "front": (0, 3.0, "inferno"), "gamma_max": (0, 1, "viridis")}.items():
                m = maps[k].double().cpu().numpy()
                m = np.where(cam.valid.cpu().numpy(), m, np.nan)
                _png(out / f"frame{f:04d}_{cam.name}_{k}.png", np.nan_to_num(m, nan=lo), lo, hi, cm)
            if a.parity:
                cp = prior.camera(cam.name)
                D = ob.depth_observed.to(dev).float()
                sd, su = sigma_maps(cfg, cam.name, D, cam.valid, cp)
                dump = fo.fr_dump(cam.args)
                par.append({"camera": cam.name,
                            "renderer": renderer_parity(view, dump, view.cache.colors, bg, cfg.eps_depth),
                            "oracle": oracle_parity(dump, D, cam.valid, sd, su, cam.logL, cp, n=a.parity_pixels)})
        if par:
            row["parity"] = par
        rows.append(row)
        print(f"  frame {f}: " + "; ".join(
            f"{c['camera']} unknown-dominant {c['unknown_dominant_fraction']:.2f} front {c['front_contradiction_mean']:.2f}"
            for c in row["cameras"] if c["valid_pixels"]))
    rep = {"tool": "observation_diagnostics/first_return_audit.py audit", "read_only": True, "model": a.model,
           "source": a.source, "prior": prior.to_manifest(), "kernel_source_sha256": fo.source_sha(),
           "note": "alpha is the renderer's effective opacity, NOT a calibrated physical hit probability",
           "frames": rows}
    (out / "first_return_audit.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"wrote {out / 'first_return_audit.json'}")
    return 0


@torch.no_grad()
def run_calibrate(a) -> int:
    from estimator.first_return_observation import sigma_maps
    from estimator.first_return_types import CameraPrior, write_prior
    from estimator import first_return_operators as fo
    from estimator.renderer_bridge import build_linearization
    t0 = time.time()
    s, e = a.frames
    truth, cameras, window, cfg, ns, ids = _setup(a, s, e)
    dev = truth.mu.device
    bg = ns.masked_observation.model.background_tensor(dev)
    lin = build_linearization(truth, cameras, eps_depth=cfg.eps_depth, background=bg)
    grid = [0.0] + [float(x) for x in np.logspace(math.log10(a.lambda_min), math.log10(a.lambda_max), a.lambda_steps)]
    sig_kind = "sensor" if a.unknown_sigma == "sensor" else "constant"
    sig_m = None if sig_kind == "sensor" else float(a.unknown_sigma)
    zn = RENDERER_NEAR_M
    curves, out_cams, support = {}, {}, {}
    for c, (name, view) in enumerate(zip(cfg.camera_names, lin.views)):
        tot = np.zeros(len(grid)); npix = 0; dmax = 0.0; excluded = 0
        for k in range(len(window)):
            ob = _prep(ns, window, k, dev)[c]
            D = ob.depth_observed.to(dev).float()
            v0 = ob.depth_valid.to(dev).bool() & torch.isfinite(D) & (D > 0)
            if bool(v0.any()):
                dmax = max(dmax, float(D[v0].max()))
            valid = v0 & (D >= zn) & (D <= a.z_far)
            excluded += int((v0 & ~valid).sum())
            npix += int(valid.sum())
            for gi, lam in enumerate(grid):
                cp = CameraPrior(name, zn, a.z_far, lam, sig_kind, sig_m)
                sd, su = sigma_maps(cfg, name, D, valid, cp)
                args = fo.camera_args(view.cache, torch.where(valid, D, torch.zeros_like(D)), valid, sd, su,
                                      lam, zn, a.z_far)
                tot[gi] += float(fo.fr_eval(args, diagnostics=False)["logL"][valid].sum())
        best = int(np.argmax(tot))
        curves[name] = {"lambda_per_m": grid, "total_logL": tot.tolist(), "valid_pixels": npix,
                        "excluded_out_of_support": excluded, "observed_max_depth_m": dmax}
        support[name] = dmax
        out_cams[name] = {"z_near_m": zn, "z_far_m": a.z_far, "lambda_per_m": grid[best],
                          "unknown_depth_sigma_model": ({"kind": "sensor"} if sig_kind == "sensor"
                                                        else {"kind": "constant", "sigma_m": sig_m})}
        edge = " (AT THE GRID EDGE: widen --lambda-min/--lambda-max)" if best in (1, len(grid) - 1) else ""
        print(f"  {name}: lambda_U MLE {grid[best]:.4g} 1/m over {npix} pixels x frames{edge}; "
              f"observed max depth {dmax:.3f} m, {excluded} pixels outside [{zn}, {a.z_far}]")
    ply = Path(a.model) / "point_cloud.ply"
    prov = {"method": "maximum likelihood of the fixed canonical render over a stationary window",
            "stationary_window_asserted_by_caller": [s, e],
            "model": a.model, "model_ply_sha256": _sha(ply) if ply.is_file() else None,
            "source": a.source, "sensor_model": a.sensor_model,
            "sensor_model_id": (cfg.sensor_model.model_id if cfg.sensor_model is not None else None),
            "declared_depth_model_sigma_m": float(cfg.depth_model_sigma),
            "z_near_m": f"{zn} = renderer near-plane cull (auxiliary.h in_frustum)",
            "z_far_m": f"{a.z_far} declared by the caller (--z-far)",
            "grid": {"lambda_min": a.lambda_min, "lambda_max": a.lambda_max, "steps": a.lambda_steps, "includes_zero": True},
            "curves": curves, "kernel_source_sha256": fo.source_sha(),
            "warning": "never evaluate a motion arm on the calibration frames",
            "runtime_s": round(time.time() - t0, 1)}
    raw = write_prior(a.out, out_cams, "calibrated", prov)
    print(f"wrote {a.out} (model_id {raw['model_id'][:12]})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    from estimator.run_config import add_cli as rc_cli
    for name in ("audit", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--source", required=True)
        p.add_argument("--out", required=True)
        rc_cli(p, phase1=True)
        p.add_argument("--gt-labels", default="off")
    pa = sub.choices["audit"]
    pa.add_argument("--frames", type=int, nargs="+", required=True)
    pa.add_argument("--first-return-prior", required=True)
    pa.add_argument("--phase1-dir", default=None, help="phase1_centers/ of a run: audit the TRACKED state")
    pa.add_argument("--top", type=int, default=20)
    pa.add_argument("--parity", action="store_true", help="renderer + CPU-oracle parity (slow, Python loops)")
    pa.add_argument("--parity-pixels", type=int, default=200)
    pc = sub.choices["calibrate"]
    pc.add_argument("--frames", type=int, nargs=2, required=True, metavar=("START", "END"),
                    help="a STATIONARY window, held out from every evaluated motion frame")
    pc.add_argument("--z-far", type=float, required=True, help="declared far support of every camera [m]")
    pc.add_argument("--unknown-sigma", default="sensor", help="'sensor' or a constant sd in metres")
    pc.add_argument("--lambda-min", type=float, default=1e-3)
    pc.add_argument("--lambda-max", type=float, default=1e2)
    pc.add_argument("--lambda-steps", type=int, default=31)
    a = ap.parse_args(argv)
    return run_audit(a) if a.cmd == "audit" else run_calibrate(a)


if __name__ == "__main__":
    raise SystemExit(main())
