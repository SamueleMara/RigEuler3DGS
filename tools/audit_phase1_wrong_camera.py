#!/usr/bin/env python
"""READ-ONLY Phase-1 wrong-camera force provenance audit (real fridge 80-99).

CODEX_DEBUG_RigEuler3DGS_REAL_FRIDGE_WRONG_CAMERA_GAUSSIAN_DRIFT.md. Nothing here changes
the estimator: the production Phase 1 (tools/kpart_pipeline.bootstrap_positions ->
estimator.sparse_iekf.iekf_frame_step) runs unchanged; a read-only iteration observer
reads its frozen operator assembly (estimator/wrong_camera_audit.py).

  Gate B  replay of the stationary window twice with the debug observer OFF (the GPU
          reproducibility floor) and once ON; every frame's mu, v, omega, P_post and the
          accepted/rejected trial energies are compared (baseline_parity.json)
  Gate A  at the first update (processing index 1, inner iteration 0 = the pre-drift
          frozen state): per-camera x per-channel forces (additivity, symmetry, PSD
          checked against the production operators), grouped leave-one-camera /
          leave-one-channel frozen solves with the exact identity check, and CUDA
          contributor traces of the top-runaway and stable Gaussians (selected AFTER
          the replay, forces taken from the pre-drift state)
  Gate C  (--gate-c) separate, explicitly labelled nonlinear ablation arms from the same
          endpoint: minus a camera, minus one channel of a camera, plus the static
          reference; every arm scored on ALL cameras with the UNCHANGED observation model

Outputs under <out>/wrong_camera_audit/ (npz + JSON schema when pyarrow is absent).

    python tools/audit_phase1_wrong_camera.py --model M --source S --sensor-model CAL \\
        --p0-geometry M/geometry_information.npz --frame-timestamps auto \\
        --rgb-model-sigma 0.04 --depth-model-sigma 0.003 \\
        --motion-start-frame 80 --motion-end-frame 99 --out OUT [--gate-c]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
SCHEMA = 1


def _sha(p: Path) -> str | None:
    p = Path(p)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(*a):
    try:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:                                                           # noqa: BLE001
        return None


def _q(x, qs=(0.5, 0.9, 0.99)):
    x = torch.as_tensor(x).double().reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return None
    return [float(v) for v in torch.quantile(x.cpu(), torch.tensor(qs, dtype=torch.float64))]


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------
class StateCapture:
    """phase1_audit callback (identical in every arm): per frame mu, v, omega, P_post,
    the predicted state, and the inner-iteration energies. Read-only."""

    def __init__(self, keep_P=True):
        self.frames, self.keep_P = [], keep_P

    @torch.no_grad()
    def __call__(self, *, ti, fid, prior_state, P_prev, state, P_post, diag, obs, cfg, dt, cameras,
                 observation_model, canonical):
        from estimator.process import process_predict
        pred = process_predict(prior_state, dt)
        rows = [{k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                 for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None}
                for r in diag.rows if r.get("event") in ("initial", "accepted", "rejected", "stop", "posterior")]
        f32 = lambda t: t.detach().float().cpu()                               # noqa: E731
        self.frames.append({"ti": ti, "fid": fid, "dt": float(dt), "mu": f32(state.mu), "R": f32(state.R),
                            "v": f32(state.v), "omega": f32(state.omega), "mu_pred": f32(pred.mu), "v_pred": f32(pred.v),
                            "P_post": f32(P_post) if self.keep_P else None, "rows": rows})


class WrongCameraObserver:
    """iteration_observer: force decomposition every inner iteration; full per-Gaussian
    forces at iteration 0 of every frame; checks + leave-one-out at (loo_frame, 0)."""

    def __init__(self, camera_names, loo_frame=1, loo_tol=1e-6, loo_max_iters=400, n_probe=3):
        self.names, self.loo_frame = list(camera_names), loo_frame
        self.loo_tol, self.loo_max_iters, self.n_probe = loo_tol, loo_max_iters, n_probe
        self.ti, self.fid = None, None
        self.iter_rows, self.forces0, self.group0, self.first = [], {}, {}, None

    def begin_frame(self, ti, fid):
        self.ti, self.fid = ti, fid

    @torch.no_grad()
    def __call__(self, *, iteration, x, x_prior, P_prior, Lambda_prior, lin, prec, alpha_terms, terms,
                 rhs_pose, meas_normal, b, grouped, lm, radius, use_trust_region, dt, cfg, M_blocks,
                 first_return_depth, rgbd_gt, cameras, **kw):
        from estimator import wrong_camera_audit as wa
        if first_return_depth:
            raise SystemExit("the wrong-camera audit decomposes the LEGACY operator; first-return depth mode is "
                             "a different measurement model")
        orig = self.fid.get("original_frame_index") if isinstance(self.fid, dict) else self.fid
        bc = wa.camera_channel_rhs(lin, terms)
        tot = sum(bc.values())
        rel = float((tot - rhs_pose.double()).norm() / rhs_pose.double().norm().clamp_min(1e-30))
        en = wa.camera_channel_energy(terms)
        for r in en:
            r["camera"] = self.names[r["camera_index"]]
        self.iter_rows.append({
            "processing_index": self.ti, "original_frame": orig, "iteration": iteration,
            "rhs_additivity_rel_err": rel, "trust_radius": float(radius), "lm": float(lm),
            "energy": en,
            "force_norm_q": {f"{self.names[c]}:{ch}": _q(v.norm(dim=1)) for (c, ch), v in bc.items()}})
        if iteration == 0:
            stack = torch.stack([bc[(c, ch)] for c in range(len(self.names)) for ch in wa.CHANNELS]).float().cpu()
            self.forces0[self.ti] = stack                                     # [3*C, N, 6]
            if grouped is not None:
                g = torch.stack([grouped.reduce(bc[(c, ch)].float().contiguous())
                                 for c in range(len(self.names)) for ch in wa.CHANNELS]).float().cpu()
                self.group0[self.ti] = g                                      # [3*C, C_groups, 6]
        if self.ti == self.loo_frame and iteration == 0:
            t0 = time.time()
            checks, _ = wa.additivity_and_symmetry(lin, terms, rhs_pose, meas_normal, n_probe=self.n_probe)
            from estimator.sparse_iekf import group_precond
            G = (lambda v: torch.cat([grouped.expand(v[:, :6].float()), grouped.expand(v[:, 6:].float())], 1)) \
                if grouped is not None else None
            GT = (lambda v: torch.cat([grouped.reduce(v[:, :6].float().contiguous()),
                                       grouped.reduce(v[:, 6:].float().contiguous())], 1)) if grouped is not None else None
            # the production M_blocks already hold Lambda_prior + Fisher + lm I
            pre = group_precond(M_blocks, grouped, G, GT) if grouped is not None else None
            sys_ = wa.FrozenSystem(lin, terms, Lambda_prior, b, meas_normal, float(lm), grouped, pre)
            loo = wa.leave_one_out(sys_, len(self.names), tol=self.loo_tol, max_iters=self.loo_max_iters)
            C = wa.camera_centres(lin).to(x.mu.device)
            self.first = {
                "processing_index": self.ti, "original_frame": orig, "checks": checks, "loo": loo,
                "x_prior_mu": x_prior.mu.detach().float().cpu(), "x_mu": x.mu.detach().float().cpu(),
                "P_prior_pv": P_prior[:, :6, 6:].detach().float().cpu(),
                "camera_centres": C.cpu(), "lm": float(lm), "radius": float(radius),
                "use_trust_region": bool(use_trust_region), "dt": float(dt),
                "supernode_id": (grouped.part.supernode_id.long().cpu() if grouped is not None else None),
                "seconds": round(time.time() - t0, 1)}


class TraceObserver:
    """Contributor traces of selected Gaussians at (frame, iteration 0); read-only."""

    def __init__(self, camera_names, gaussian_ids, frame=1, sigma_fn=None, curvature_dirs=None):
        self.names, self.ids, self.frame, self.sigma_fn = list(camera_names), gaussian_ids, frame, sigma_fn
        self.curvature_dirs = curvature_dirs or {}
        self.ti, self.rows, self.rayleigh = None, None, {}

    def begin_frame(self, ti, fid):
        self.ti = ti

    @torch.no_grad()
    def __call__(self, *, iteration, lin, terms, rgbd_gt, cfg, **kw):
        if self.ti != self.frame or iteration != 0 or self.rows is not None:
            return
        from estimator import wrong_camera_audit as wa
        self.rows = wa.contributor_traces(lin, rgbd_gt, cfg, self.ids, self.names,
                                          lambda c, D, v: self.sigma_fn(c, D, v, cfg))
        # camera x channel curvature along each selected Gaussian's own accepted translation
        n = lin.num_gaussians
        for g, u3 in self.curvature_dirs.items():
            u = torch.zeros((n, 6), device=lin.views[0].depth.device)
            u[int(g), :3] = torch.as_tensor(u3, device=u.device)
            if float(u.norm()) == 0:
                continue
            u = u / u.norm()
            Au = wa.camera_channel_normal(lin, terms, u)
            self.rayleigh[int(g)] = {f"{self.names[c]}:{ch}": float((u.double() * a).sum()) for (c, ch), a in Au.items()}


def make_sigma_fn():
    def fn(c, D, valid, cfg):
        from estimator.observation_terms import calibrated_camera, camera_name, depth_sensor_sigma
        cal = calibrated_camera(cfg, camera_name(cfg, c))
        z = torch.where(valid, D, torch.ones_like(D))
        sd = cal.depth_variance(z.double()).to(D.dtype).sqrt() if cal is not None else depth_sensor_sigma(D, valid, cfg)
        dms = float(cfg.depth_model_sigma or 0.0)
        return (sd * sd + dms * dms).sqrt() if dms > 0 else sd
    return fn


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
def setup(args):
    from audit_calibrated_phase1 import setup_window
    return setup_window(args)


def drop_camera(cameras, window, cfg, ns, idx):
    """Diagnostic camera removal: RHS, curvature and energy of that camera all vanish."""
    keep = [i for i in range(len(cameras)) if i != idx]
    cams = [cameras[i] for i in keep]
    win = [([ob[i] for i in keep], fid) for ob, fid in window]
    cfg2 = dataclasses.replace(cfg, camera_names=tuple(cfg.camera_names[i] for i in keep))
    ms = ns.masked_observation
    if ms is not None:
        ms2 = dataclasses.replace(ms, masks=[[m[i] for i in keep] for m in ms.masks],
                                  cameras=[ms.cameras[i] for i in keep], report=[])
        ns = argparse.Namespace(**{**vars(ns), "masked_observation": ms2})
    return cams, win, cfg2, ns


def replay(truth, cameras, window, cfg, ns, observer=None, keep_P=True):
    from tools.kpart_pipeline import bootstrap_positions
    cap = StateCapture(keep_P=keep_P)
    ns2 = argparse.Namespace(**vars(ns))
    ns2.phase1_audit = cap
    ns2.phase1_iteration_observer = observer
    if ns2.masked_observation is not None:
        ns2.masked_observation = dataclasses.replace(ns2.masked_observation, report=[])
    torch.manual_seed(0)
    t0 = time.time()
    pos, info = bootstrap_positions(truth, cameras, window, cfg, None, ns2, verbose=False)
    return {"positions": [p.detach().float().cpu() for p in pos], "rotations": info["rotations"],
            "info": info, "capture": cap, "seconds": time.time() - t0,
            "report": (ns2.masked_observation.report if ns2.masked_observation is not None else [])}


def parity(a, b) -> dict:
    out = []
    for fa, fb in zip(a["capture"].frames, b["capture"].frames):
        r = {"processing_index": fa["ti"], "original_frame": fa["fid"].get("original_frame_index") if isinstance(fa["fid"], dict) else fa["fid"]}
        for k in ("mu", "v", "omega", "P_post"):
            if fa[k] is not None and fb[k] is not None:
                r[f"{k}_max_abs"] = float((fa[k] - fb[k]).abs().max())
        ea = [x.get("energy") for x in fa["rows"] if x.get("event") in ("accepted", "rejected")]
        eb = [x.get("energy") for x in fb["rows"] if x.get("event") in ("accepted", "rejected")]
        acc_a = [x.get("event") for x in fa["rows"] if x.get("event") in ("accepted", "rejected")]
        acc_b = [x.get("event") for x in fb["rows"] if x.get("event") in ("accepted", "rejected")]
        r["same_accept_sequence"] = acc_a == acc_b
        r["trial_energy_max_rel"] = (max(abs(x - y) / max(abs(x), 1e-30) for x, y in zip(ea, eb))
                                     if ea and len(ea) == len(eb) else None)
        out.append(r)
    worst = {k: max((r.get(k) or 0.0) for r in out) for k in ("mu_max_abs", "v_max_abs", "omega_max_abs", "P_post_max_abs")}
    return {"frames": out, "worst": worst, "all_same_accept_sequence": all(r["same_accept_sequence"] for r in out)}


@torch.no_grad()
def heldout_scores(states, cameras, window, cfg, ns) -> list:
    """Every camera, UNCHANGED observation model: per frame energies, mask IoU, depth MAE."""
    import dataclasses as dc
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import _frame_terms
    from audit_calibrated_phase1 import _frame_obs
    model = None if ns.masked_observation is None else ns.masked_observation.model
    rows = []
    for k, (mu, R) in enumerate(states):
        st = dc.replace(ns._truth, mu=mu.to(ns._truth.mu), R=R.to(ns._truth.R))
        obs = _frame_obs(ns, k, window, st.mu.device)
        lin = build_linearization(st, cameras, eps_depth=cfg.eps_depth,
                                  background=None if model is None else model.background_tensor(st.mu.device))
        terms = _frame_terms(lin.views, obs, cfg, observation_model=model)
        for c, (v, t, ob) in enumerate(zip(lin.views, terms, obs)):
            M = ob.object_coverage.to(v.alpha.device) > 0.5
            Ab = v.alpha > 0.5
            used = t.p_d > 0
            rows.append({"processing_index": k, "camera": cfg.camera_names[c], "rgb_energy": t.e_rgb,
                         "depth_energy": t.e_d, "alpha_energy": t.e_a,
                         "mask_iou": float((M & Ab).sum() / (M | Ab).sum().clamp_min(1)),
                         "depth_mae_used_mm": 1000 * float(t.r_d[used].abs().mean()) if bool(used.any()) else None})
    return rows


def disp_summary(run, ref0) -> list:
    rows = []
    for k, p in enumerate(run["positions"]):
        d = (p - ref0).norm(dim=1)
        fr = run["capture"].frames[k - 1] if k > 0 else None
        speed = fr["v"].norm(dim=1) if fr is not None else torch.zeros(1)
        rows.append({"processing_index": k, "median_mm": 1000 * float(d.median()), "p90_mm": 1000 * float(torch.quantile(d, 0.9)),
                     "p99_mm": 1000 * float(torch.quantile(d, 0.99)), "max_mm": 1000 * float(d.max()),
                     "median_speed_mps": float(speed.median()),
                     "stop_reason": next((r.get("reason") for r in (fr["rows"] if fr else []) if r.get("event") == "stop"), None)})
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


# ---------------------------------------------------------------------------
def main(argv=None, _setup=None) -> int:
    """_setup: tests only -- a callable(args) replacing the production dataset setup."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deterministic", action="store_true", help="bitwise reproducible reductions")
    ap.add_argument("--direction", choices=("forward",), default="forward")
    ap.add_argument("--bootstrap-iters", type=int, default=12)
    ap.add_argument("--iekf-supernode-size", type=int, default=64)
    ap.add_argument("--phase1-partition", choices=("canonical", "per-frame"), default="canonical")
    ap.add_argument("--accept-endpoint-mismatch", default=None, metavar="REASON",
                    help="run although the model's endpoint frame != --motion-start-frame; REASON is recorded")
    ap.add_argument("--runaway-quantile", type=float, default=0.99)
    ap.add_argument("--n-stable", type=int, default=300)
    ap.add_argument("--n-case-studies", type=int, default=40)
    ap.add_argument("--loo-tol", type=float, default=1e-6)
    ap.add_argument("--loo-max-iters", type=int, default=400)
    ap.add_argument("--gate-c", action="store_true", help="also run the nonlinear ablation arms")
    ap.add_argument("--deep", action="store_true",
                    help="also the objective-conflict, group-mediation and nonlinear-attribution analyses of the "
                         "first update (estimator/wrong_camera_deep.py)")
    ap.add_argument("--deep-ungrouped-max-iters", type=int, default=1500)
    ap.add_argument("--deep-top-groups", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    from estimator.masked_observation import add_cli as mask_cli
    from estimator.run_config import add_cli as rc_cli
    mask_cli(ap)
    rc_cli(ap, phase1=True)
    a = ap.parse_args(argv)
    if a.deterministic:
        from estimator.deterministic_ops import enable as _det_enable
        print(f"deterministic reductions: {_det_enable()}")
    a.first_return_mode, a.first_return_prior, a.phase1_covariance_transport, a.gt_labels = "off", None, False, "off"
    out = Path(a.out) / "wrong_camera_audit"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    status = {}

    # ---- manifest + endpoint check ------------------------------------------------
    bm = Path(a.model) / "bootstrap_manifest.json"
    endpoint = json.loads(bm.read_text()).get("endpoint_frame") if bm.is_file() else None
    mismatch = endpoint is not None and int(endpoint) != int(a.motion_start_frame)
    if endpoint is None or mismatch:
        if not a.accept_endpoint_mismatch:
            raise SystemExit(f"the model's endpoint frame is {endpoint}, the replay starts at {a.motion_start_frame}: "
                             "refusing (train an endpoint at the start frame, or pass --accept-endpoint-mismatch "
                             "REASON to record a deliberate deviation)")
    truth, cameras, window, cfg, ns, frame_ids = (_setup or setup)(a)
    ns._truth = truth
    names = list(cfg.camera_names)
    man = {"schema": SCHEMA, "tool": "tools/audit_phase1_wrong_camera.py", "read_only": True,
           "git": {"sha": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                   "dirty": _git("status", "--short", "--untracked-files=no")},
           "model": a.model, "model_ply_sha256": _sha(Path(a.model) / "point_cloud.ply"),
           "model_endpoint_frame": endpoint,
           "endpoint_deviation": ({"accepted_by_user": a.accept_endpoint_mismatch} if mismatch else None),
           "source": a.source, "frames": frame_ids["frames"], "cameras": names,
           "sensor_model_id": cfg.sensor_model.model_id if cfg.sensor_model is not None else None,
           "declared": {"rgb_model_sigma": cfg.rgb_model_sigma, "depth_model_sigma": cfg.depth_model_sigma},
           "observation_mode": a.observation_mode, "phase1_partition": a.phase1_partition,
           "deterministic_reductions": bool(a.deterministic),
           "p0_geometry": a.p0_geometry, "p0_velocity_sigma": list(a.p0_velocity_sigma),
           "num_gaussians": int(truth.mu.shape[0]),
           "note": "forces / LOO decompose the LEGACY production operator at a frozen iterate; the LOO solves are the "
                   "unconstrained frozen quadratic, NOT the production trust-region step"}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))

    # ---- Gate B: replay off, off, on ----------------------------------------------------
    print("Gate B: replay x3 (off, off, on)")
    off1 = replay(truth, cameras, window, cfg, ns)
    off2 = replay(truth, cameras, window, cfg, ns)
    obs_on = WrongCameraObserver(names, loo_frame=1, loo_tol=a.loo_tol, loo_max_iters=a.loo_max_iters)
    on = replay(truth, cameras, window, cfg, ns, observer=obs_on)
    floor = parity(off1, off2)
    par = parity(off1, on)
    tol = {k: 2 * max(v, 1e-9) for k, v in floor["worst"].items()}
    ok = all(par["worst"][k] <= tol[k] for k in tol) and par["all_same_accept_sequence"] == floor["all_same_accept_sequence"]
    status["gate_B_parity"] = "PASS" if ok else "FAIL"
    (out / "baseline_parity.json").write_text(json.dumps(
        {"floor_off_vs_off": floor["worst"], "on_vs_off": par["worst"], "tolerance_2x_floor": tol,
         "accept_sequence_identical": par["all_same_accept_sequence"], "status": status["gate_B_parity"],
         "frames": par["frames"], "seconds": {"off1": off1["seconds"], "off2": off2["seconds"], "on": on["seconds"]}},
        indent=1, default=float))
    ref0 = on["positions"][0]
    per_frame = disp_summary(on, ref0)
    _write_csv(out / "per_frame_summary.csv", per_frame)
    erows = []
    for r in obs_on.iter_rows:
        for e in r["energy"]:
            erows.append({"processing_index": r["processing_index"], "original_frame": r["original_frame"],
                          "iteration": r["iteration"], **e, "rhs_additivity_rel_err": r["rhs_additivity_rel_err"]})
    _write_csv(out / "per_camera_channel_energy.csv", erows)
    with open(out / "nonlinear_energy.jsonl", "w") as f:
        for fr in on["capture"].frames:
            for r in fr["rows"]:
                f.write(json.dumps({"processing_index": fr["ti"], "frame": fr["fid"], **r}, default=str) + "\n")
    chan_keys = [f"{c}:{ch}" for c in names for ch in ("rgb", "depth", "alpha")]
    np.savez_compressed(out / "per_gaussian_force.npz",
                        processing_index=np.array(sorted(obs_on.forces0)),
                        forces=np.stack([obs_on.forces0[k].numpy() for k in sorted(obs_on.forces0)]),
                        camera_channel=np.array(chan_keys))
    if obs_on.group0:
        np.savez_compressed(out / "per_group_force.npz", processing_index=np.array(sorted(obs_on.group0)),
                            forces=np.stack([obs_on.group0[k].numpy() for k in sorted(obs_on.group0)]),
                            camera_channel=np.array(chan_keys))
    (out / "schemas.json").write_text(json.dumps({
        "per_gaussian_force.npz": "forces[frame, camera_channel, gaussian_row, 6] float32: production H^T W r of that "
                                  "camera/channel at inner iteration 0 (pose tangent: dmu xyz, dphi xyz)",
        "per_group_force.npz": "forces[frame, camera_channel, supernode, 6] = G^T of the above",
        "loo_frozen_influence.npz": "delta_all[gaussian, 12] and influence[arm, gaussian, 12] (I_k = delta_all - "
                                    "delta_-k) at processing index 1, iteration 0; arms listed in arm_names",
        "selected_contributor_traces.jsonl": "one record per (camera, pixel, selected Gaussian): CSR alpha, T, q, "
                                             "beta = q/(A+eps), depths, d/sigma, class (REPORTING ONLY)"}, indent=1))

    # ---- Gate A: first update ------------------------------------------------------------
    fu = obs_on.first
    if fu is None:
        status["gate_A"] = "NOT_RUN (no first update captured)"
    else:
        ck = fu["checks"]
        chk_ok = (ck["rhs_additivity_rel_err"] < 1e-4 and ck["normal_additivity_rel_err_max"] < 1e-4
                  and max(ck["symmetry_rel_err"].values()) < 1e-3 and min(ck["min_vAv_over_norms"].values()) > -1e-5)
        loo = fu["loo"]
        ident = {k: v.get("identity_rel_err") for k, v in loo["arms"].items() if "identity_rel_err" in v}
        conv = loo["solver_all"]["converged"] and all(v["solver"]["converged"] for v in loo["arms"].values())
        status["gate_A_operator_checks"] = "PASS" if chk_ok else "FAIL"
        status["gate_A_loo_identity"] = ("PASS" if conv and all(e < 1e-3 for e in ident.values())
                                         else "INDETERMINATE (solver not converged)" if not conv else "FAIL")
        (out / "per_camera_normal_checks.json").write_text(json.dumps(
            {**ck, "status": status["gate_A_operator_checks"], "camera_names": names}, indent=1))
        arm_names = list(loo["arms"])
        np.savez_compressed(out / "loo_frozen_influence.npz", delta_all=loo["delta_all"].float().cpu().numpy(),
                            influence=np.stack([loo["arms"][k]["influence"].float().cpu().numpy() for k in arm_names]),
                            arm_names=np.array(arm_names))
        (out / "loo_solver.json").write_text(json.dumps(
            {"all": loo["solver_all"], "arms": {k: {"keys": v["keys"], "solver": v["solver"],
                                                    "identity_rel_err": v.get("identity_rel_err"),
                                                    "identity_solver": v.get("identity_solver")}
                                                for k, v in loo["arms"].items()},
             "status": status["gate_A_loo_identity"], "seconds": fu["seconds"]}, indent=1, default=float))

        # selection AFTER the replay, forces from the pre-drift state
        fin = (on["positions"][-1] - ref0).norm(dim=1)
        n = fin.numel()
        thr = torch.quantile(fin, a.runaway_quantile)
        runaway = torch.nonzero(fin >= thr).reshape(-1)
        g = torch.Generator().manual_seed(a.seed)
        low = torch.nonzero(fin <= torch.quantile(fin, 0.5)).reshape(-1)
        stable = low[torch.randperm(low.numel(), generator=g)[:a.n_stable]]
        sel = torch.cat([runaway, stable])
        fr1 = on["capture"].frames[0]
        dmu_acc = (fr1["mu"] - fr1["mu_pred"])
        curv = {int(i): dmu_acc[i].tolist() for i in torch.cat([runaway[:a.n_case_studies], stable[:a.n_case_studies]])}
        tr = TraceObserver(names, sel.tolist(), frame=1, sigma_fn=make_sigma_fn(), curvature_dirs=curv)
        print("Gate A: contributor traces at the pre-drift frozen state")
        replay(truth, cameras, window[:2], cfg, ns, observer=tr, keep_P=False)
        from estimator import wrong_camera_audit as wa
        with open(out / "selected_contributor_traces.jsonl", "w") as f:
            for r in tr.rows or []:
                r["set"] = "runaway" if r["gaussian_id"] in set(runaway.tolist()) else "stable"
                f.write(json.dumps(r) + "\n")
        tsum = wa.summarize_traces(tr.rows or [])
        # per-Gaussian evidence rows + aggregates
        F0 = obs_on.forces0[1]                                             # [9, N, 6]
        I = {k: loo["arms"][k]["influence"].float().cpu() for k in arm_names}
        dall = loo["delta_all"].float().cpu()
        Cc = fu["camera_centres"].float()
        mu0 = fu["x_prior_mu"]
        sid = fu["supernode_id"]
        dv_acc = fr1["v"] - fr1["v_pred"]
        dt = fu["dt"]

        def evidence(i):
            i = int(i)
            ray = {}
            for c, nm in enumerate(names):
                r = mu0[i] - Cc[c]
                r = r / r.norm()
                ray[nm] = {"cos_accepted_dmu": float(abs((dmu_acc[i] * r).sum()) / dmu_acc[i].norm().clamp_min(1e-30)),
                           "cos_loo_translation": float(abs((I[f"minus_cam{c}"][i, :3] * r).sum())
                                                        / I[f"minus_cam{c}"][i, :3].norm().clamp_min(1e-30))}
            force = {chan_keys[j]: F0[j, i].tolist() for j in range(len(chan_keys))}
            align = {chan_keys[j]: float((F0[j, i, :3] * dmu_acc[i]).sum() / (F0[j, i, :3].norm() * dmu_acc[i].norm()).clamp_min(1e-30))
                     for j in range(len(chan_keys))}
            loo_i = {k: {"translation_norm": float(I[k][i, :3].norm()), "velocity_norm": float(I[k][i, 6:9].norm()),
                         "influence_fraction": float(I[k][i, :3].norm() / max(float(dall[i, :3].norm()), 1e-12)),
                         "cos_with_accepted_dmu": float((I[k][i, :3] * dmu_acc[i]).sum()
                                                        / (I[k][i, :3].norm() * dmu_acc[i].norm()).clamp_min(1e-30)),
                         "vector12": I[k][i].tolist()} for k in arm_names}
            return {"gaussian_id": i, "supernode_id": int(sid[i]) if sid is not None else None,
                    "source_camera_id": None,
                    "position_start": mu0[i].tolist(), "position_predicted": fr1["mu_pred"][i].tolist(),
                    "position_accepted": fr1["mu"][i].tolist(), "delta_mu": dmu_acc[i].tolist(),
                    "delta_v": dv_acc[i].tolist(),
                    "velocity_booking": float(dv_acc[i].norm() / max(float(dmu_acc[i].norm()) / dt, 1e-30)),
                    "P_prior_pose_velocity_fro": float(fu["P_prior_pv"][i].norm()),
                    "final_displacement_mm": 1000 * float(fin[i]),
                    "camera_channel_force": force, "force_alignment_with_accepted_dmu": align,
                    "camera_channel_curvature_rayleigh_along_accepted_dmu": tr.rayleigh.get(i),
                    "camera_loo_influence": loo_i, "contributors": tsum.get(str(i)),
                    "ray_alignment_by_camera": ray,
                    "frozen_unconstrained_delta_all_translation": dall[i, :3].tolist()}

        cases = {"runaway": [evidence(i) for i in runaway[:a.n_case_studies]],
                 "stable": [evidence(i) for i in stable[:a.n_case_studies]]}

        def agg(ids):
            ids = torch.as_tensor(ids)
            res = {}
            for k in arm_names:
                nrm = I[k][ids, :3].norm(dim=1)
                frac = nrm / dall[ids, :3].norm(dim=1).clamp_min(1e-12)
                res[k] = {"influence_translation_mm_q": [1000 * x for x in _q(nrm)], "influence_fraction_q": _q(frac)}
            for j, key in enumerate(chan_keys):
                res[f"force:{key}"] = {"translation_norm_q": _q(F0[j, ids, :3].norm(dim=1))}
            cls = {}
            for gid in ids.tolist():
                for cam, e in (tsum.get(str(gid)) or {}).items():
                    for cn, fr in e["class_fraction_beta_weighted"].items():
                        cls.setdefault(cam, {}).setdefault(cn, []).append(fr)
            res["contributor_class_fraction_median"] = {cam: {cn: float(np.median(v)) for cn, v in d.items()}
                                                        for cam, d in cls.items()}
            res["velocity_booking_q"] = _q(dv_acc[ids].norm(dim=1) / (dmu_acc[ids].norm(dim=1) / dt).clamp_min(1e-30))
            res["final_displacement_mm_q"] = [1000 * x for x in _q(fin[ids])]
            return res
        studies = {"selection": {"runaway_quantile": a.runaway_quantile, "num_runaway": int(runaway.numel()),
                                 "num_stable": int(stable.numel()), "stable_rule": "random sample of the lower half "
                                 "of final displacement (seeded)"},
                   "aggregate": {"runaway": agg(runaway), "stable": agg(stable)}, "cases": cases}
        (out / "first_update_case_studies.json").write_text(json.dumps(studies, indent=1, default=float))
        _plots(out / "plots", per_frame, studies, arm_names)
        if a.deep:
            status.update(_deep(a, out, truth, cameras, window, cfg, ns, names, on, runaway, stable))

    # ---- Gate C: nonlinear ablations ---------------------------------------------------------
    abl_rows = []
    if a.gate_c:
        print("Gate C: ablation arms")
        arms = [("all", None, None)] + [(f"minus_{nm}", i, None) for i, nm in enumerate(names)] + \
               [(f"minus_{nm}_{ch}", None, (nm, ch)) for nm in names for ch in ("depth", "rgb", "alpha")]
        for name, drop, chan in arms:
            c2, w2, cfg2, ns2 = cameras, window, cfg, ns
            if drop is not None:
                c2, w2, cfg2, ns2 = drop_camera(cameras, window, cfg, ns, drop)
            if chan is not None:
                cfg2 = dataclasses.replace(cfg, debug_ablation={chan[0]: [chan[1]]})
            r = replay(truth, c2, w2, cfg2, ns2, keep_P=False)
            states = [(p, rot) for p, rot in zip(r["positions"], r["rotations"])]
            ho = heldout_scores(states, cameras, window, cfg, ns)
            d = (r["positions"][-1] - r["positions"][0]).norm(dim=1)
            row = {"arm": name, "label": "DIAGNOSTIC ABLATION (not a method)",
                   "final_median_mm": 1000 * float(d.median()), "final_p99_mm": 1000 * float(torch.quantile(d, 0.99)),
                   "final_max_mm": 1000 * float(d.max()),
                   "median_speed_mps_last": float(r["capture"].frames[-1]["v"].norm(dim=1).median()),
                   "converged_frames": sum(1 for f in r["capture"].frames
                                           if any(x.get("reason") == "converged" for x in f["rows"])),
                   "seconds": round(r["seconds"], 1)}
            for nm in names:
                sel = [h for h in ho if h["camera"] == nm and h["processing_index"] > 0]
                row[f"heldout_{nm}_depth_energy_mean"] = float(np.mean([h["depth_energy"] for h in sel]))
                row[f"heldout_{nm}_rgb_energy_mean"] = float(np.mean([h["rgb_energy"] for h in sel]))
                row[f"heldout_{nm}_mask_iou_mean"] = float(np.mean([h["mask_iou"] for h in sel]))
            abl_rows.append(row)
            print(f"  {name}: median {row['final_median_mm']:.1f} mm, p99 {row['final_p99_mm']:.1f} mm")
        static = [(truth.mu.detach(), truth.R.detach())] * len(window)
        ho = heldout_scores(static, cameras, window, cfg, ns)
        row = {"arm": "reference_static_control", "label": "canonical geometry held still (evaluation only)",
               "final_median_mm": 0.0, "final_p99_mm": 0.0, "final_max_mm": 0.0}
        for nm in names:
            sel = [h for h in ho if h["camera"] == nm and h["processing_index"] > 0]
            row[f"heldout_{nm}_depth_energy_mean"] = float(np.mean([h["depth_energy"] for h in sel]))
            row[f"heldout_{nm}_rgb_energy_mean"] = float(np.mean([h["rgb_energy"] for h in sel]))
            row[f"heldout_{nm}_mask_iou_mean"] = float(np.mean([h["mask_iou"] for h in sel]))
        abl_rows.append(row)
        _write_csv(out / "ablation_results.csv", abl_rows)
        status["gate_C"] = "RUN"
    else:
        status["gate_C"] = "NOT_RUN (pass --gate-c)"
    status["gate_D"] = "NOT_RUN (wooden box / synthetic controls: run this tool on those endpoints)"
    _conclusion(out, man, status, per_frame, abl_rows, names)
    print(f"wrote {out} in {time.time() - t_start:.0f} s; status {status}")
    return 0 if all(not str(v).startswith("FAIL") for v in status.values()) else 1


def _deep(a, out, truth, cameras, window, cfg, ns, names, on, runaway, stable) -> dict:
    """Objective conflict, group mediation and nonlinear attribution at processing index 1."""
    from estimator.wrong_camera_deep import DeepObserver
    fr1 = on["capture"].frames[0]
    sets = {"runaway": runaway.tolist(), "stable": stable.tolist()}
    deep = DeepObserver(names, sets, {k: fr1[k] for k in ("mu", "R", "v", "omega")}, frame=fr1["ti"],
                        tol=a.loo_tol, max_iters=a.loo_max_iters, max_iters_ungrouped=a.deep_ungrouped_max_iters,
                        top_groups=a.deep_top_groups)
    print("deep: objective conflict, group mediation, nonlinear attribution at the first update")
    t0 = time.time()
    chk = replay(truth, cameras, window[:2], cfg, ns, observer=deep, keep_P=False)
    same = bool(torch.equal(chk["capture"].frames[0]["mu"], fr1["mu"]))
    meta = {"processing_index": fr1["ti"], "frame": fr1["fid"], "x_post_source": "the Gate-B 'on' replay",
            "x_post_bitwise_reproduced": same, "deterministic_reductions": bool(a.deterministic),
            "num_runaway": len(sets["runaway"]), "num_stable": len(sets["stable"]), "seconds": round(time.time() - t0, 1)}
    (out / "objective_conflict.json").write_text(json.dumps(
        {**meta, "definition": "s = b . d / |d| per Gaussian (or supernode), d = accepted 6D pose change of the frame; "
                               "s > 0 the term pulls along the update, s < 0 it opposes", **deep.conflict},
        indent=1, default=float))
    if deep.group_mediation is not None:
        (out / "group_mediation.json").write_text(json.dumps({**meta, **deep.group_mediation}, indent=1, default=float))
    if deep.attribution is not None:
        acc = [r for r in fr1["rows"] if r.get("event") in ("initial", "posterior", "accepted")]
        (out / "nonlinear_attribution.json").write_text(json.dumps(
            {**meta, **deep.attribution, "production_energy_rows_of_this_frame": acc}, indent=1, default=float))
    st = {"deep_x_post_reproduced": "PASS" if same else "NOT_BITWISE (run with --deterministic)"}
    gm = deep.group_mediation or {}
    errs = [c["decomposition_rel_err"] for k in ("grouped", "ungrouped") for c in gm.get(k, {}).get("cameras", {}).values()]
    if errs:
        st["deep_mediation_decomposition"] = "PASS" if max(errs) < 1e-3 else f"FAIL (max rel err {max(errs):.1e})"
    return st


def _plots(pdir, per_frame, studies, arm_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = [r["processing_index"] for r in per_frame]
    plt.figure(figsize=(6, 3.5))
    for k in ("median_mm", "p90_mm", "p99_mm"):
        plt.plot(fr, [r[k] for r in per_frame], label=k)
    plt.yscale("symlog"); plt.xlabel("processing index"); plt.ylabel("false displacement [mm]"); plt.legend()
    plt.tight_layout(); plt.savefig(pdir / "false_displacement.png", dpi=120); plt.close()
    ag = studies["aggregate"]
    cams = [k for k in arm_names if k.count("_") == 1]
    chans = [k for k in arm_names if k.count("_") == 2]
    for nm, keys in (("loo_camera", cams), ("loo_channel", chans)):
        plt.figure(figsize=(max(6, len(keys) * 0.7), 3.5))
        x = np.arange(len(keys))
        for off, s in ((-0.2, "runaway"), (0.2, "stable")):
            plt.bar(x + off, [ag[s][k]["influence_translation_mm_q"][0] for k in keys], width=0.4, label=f"{s} median")
        plt.xticks(x, keys, rotation=45, ha="right"); plt.ylabel("|I_k| translation [mm]"); plt.legend()
        plt.tight_layout(); plt.savefig(pdir / f"{nm}_influence.png", dpi=120); plt.close()


def _conclusion(out, man, status, per_frame, abl_rows, names):
    L = ["# Wrong-camera force audit: evidence summary (auto-generated)", "",
         f"git {man['git']['sha']} ({man['git']['branch']}), dirty: {bool(man['git']['dirty'])}; "
         f"model endpoint frame {man['model_endpoint_frame']}; deviation: {man['endpoint_deviation']}", "",
         "| gate | status |", "|---|---|"] + [f"| {k} | {v} |" for k, v in status.items()] + [""]
    if per_frame:
        last = per_frame[-1]
        L += [f"Stationary replay, last frame: median {last['median_mm']:.1f} mm, p99 {last['p99_mm']:.1f} mm.", ""]
    cs = out / "first_update_case_studies.json"
    if cs.is_file():
        s = json.loads(cs.read_text())["aggregate"]
        L += ["## Q1 camera / channel influence at the pre-drift frozen state (median |I_k| translation, mm)", "",
              "| arm | runaway | stable |", "|---|---|---|"]
        for k in s["runaway"]:
            if k.startswith("minus_"):
                L.append(f"| {k} | {s['runaway'][k]['influence_translation_mm_q'][0]:.3f} | "
                         f"{s['stable'][k]['influence_translation_mm_q'][0]:.3f} |")
        L += ["", "## Q2 contributor classes (beta-weighted, median over selected Gaussians; reporting only)", "",
              "```", json.dumps(s["runaway"]["contributor_class_fraction_median"], indent=1), "```", "",
              f"## Q4 velocity booking |dv| / (|dmu|/dt): runaway {s['runaway']['velocity_booking_q']}, "
              f"stable {s['stable']['velocity_booking_q']}", ""]
    if abl_rows:
        L += ["## Q5 nonlinear ablations (diagnostic; see ablation_results.csv for held-out scores)", "",
              "| arm | median mm | p99 mm |", "|---|---|---|"] + \
             [f"| {r['arm']} | {r['final_median_mm']:.1f} | {r['final_p99_mm']:.1f} |" for r in abl_rows] + [""]
    L += ["## Hypothesis status", "",
          "The wrong-camera hypothesis needs all four criteria of section 7 (production force, contributor "
          "incompatibility under uncertainty, nonlinear ablation reducing false motion without destroying a moving "
          "control, repetition). This file reports the evidence; the status of each hypothesis is assigned by "
          "reading it -- it is not computed here.", "",
          "Answer to Q8 if supported: the blended-depth path lives in "
          "estimator/observation_terms.py::camera_terms (depth residual against Z/(A+eps)), "
          "estimator/renderer_bridge.py::FrameLinearization.apply_H/apply_HT and the iekf.cu vjpCUDA/jvpCUDA kernels; "
          "a contributor-level association would replace those paths (not done here)."]
    (out / "conclusion.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
