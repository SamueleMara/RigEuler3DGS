#!/usr/bin/env python
"""READ-ONLY Phase-1 ablation of the occlusion-consistent likelihood (robot_dog regression).

The production Phase 1 (tools/kpart_pipeline.bootstrap_positions -> iekf_frame_step) runs
unchanged in every arm; a read-only iteration observer decomposes the frozen RHS.

Arms (likelihood / group update), every one from the SAME endpoint, frames, cameras,
sensor model, P0 and deterministic reductions:
    legacy/legacy        v6
    legacy/observable    observable group update only
    certified/legacy     occlusion-consistent likelihood with q = 1 only (first implementation)
    certified/observable the regressed run (ours_v9_phase1_occlusion_real3)
    calibrated/legacy    calibrated contamination likelihood only
    calibrated/observable the correction

Part S (sequential): each arm runs the first --max-frames frames. Per frame and stratum:
    forces at inner iteration 0 and at the last iteration, per camera and per channel
    (rgb / depth / alpha on M = 1 and on M = 0, negative ray), and the prior gradient
    -Lambda^- (x - x^-); grouped RHS of each component (G^T or B^T, static supernodes);
    accepted pose correction x+ - x^-, velocity correction v+ - v^-, prediction-only
    displacement x^- - x_{k-1}; all translations projected on every camera's optical axis
    (+ = away from that camera) and as norms.
Part F (frozen): at --frozen-frames the reference arm's exact (state, P) before the update is
    replayed through ONE frame step of every arm: same decomposition, per-Gaussian arrays.
Strata (evaluation only; never fed back): the static set = Gaussians outside every support
    given with --static-exclude-support; per frame, from a certified negative-ray evaluation
    of the arm's own linearization (iteration 0): exposed = static with front weight > 0.01
    (footprint on M = 0 pixels in front of the measured depth), interior = static with front
    weight < 1e-3 and supported weight > 0.

    python tools/diagnose_phase1_negative_ray.py --model M --source S --sensor-model CAL \\
        --p0-geometry M/geometry_information.npz --frame-timestamps auto --rgb-model-sigma 0.04 \\
        --depth-model-sigma 0.003 --velocity-tau 0.5 --motion-start-frame 36 --motion-end-frame 186 \\
        --cameras cam_001,cam_002 --max-frames 40 --frozen-frames 5 15 30 \\
        --static-exclude-support A/support.json B/support.json --out OUT --deterministic
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

ARMS = {"legacy/legacy": (None, "legacy"), "legacy/observable": (None, "observable"),
        "certified/legacy": ("certified", "legacy"), "certified/observable": ("certified", "observable"),
        "calibrated/legacy": ("calibrated", "legacy"), "calibrated/observable": ("calibrated", "observable")}


def _q(x):
    x = torch.as_tensor(x).double().reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return None
    return [float(v) for v in torch.quantile(x.cpu(), torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64))]


def camera_axes(lin) -> torch.Tensor:
    """[C,3] world direction of each camera's +z (depth increases along it)."""
    out = []
    for v in lin.views:
        a = v.camera.viewmatrix.double()[:3, 2]
        out.append(a / a.norm())
    return torch.stack(out)


def load_support(path) -> set:
    d = json.loads(Path(path).read_text())
    return set(d.get("members", d.get("dynamic_gaussians", [])))


# ---------------------------------------------------------------------------
# read-only observer
# ---------------------------------------------------------------------------
class ForceLog:
    def __init__(self, names, static: torch.Tensor, full_at=()):
        self.names, self.static, self.full_at = list(names), static, set(full_at)
        self.ti = self.fid = None
        self.rows, self.full, self.strata, self._last = [], {}, {}, None

    def begin_frame(self, ti, fid):
        if self._last is not None:
            self.rows.append(self._last)
        self.ti, self.fid, self._last = ti, fid, None

    def flush(self):
        if self._last is not None:
            self.rows.append(self._last)
            self._last = None

    @torch.no_grad()
    def components(self, lin, terms, rgbd_gt, x, x_prior, Lambda_prior):
        from estimator.sparse_iekf import apply_prior, boxminus
        from estimator.wrong_camera_audit import view_HT, view_HT_alpha
        comp = {}
        for c, (v, t) in enumerate(zip(lin.views, terms)):
            M = (rgbd_gt[c].object_coverage.to(v.rgb.device) > 0.5).to(t.r_d.dtype)
            zr, zd = torch.zeros_like(t.r_rgb), torch.zeros_like(t.r_d)
            nm = self.names[c]
            for side, w in (("on", M), ("off", 1.0 - M)):
                comp[f"{nm}:rgb:{side}"] = view_HT(v, t.p_rgb * t.r_rgb * w[None], zd).double()
                comp[f"{nm}:depth:{side}"] = view_HT(v, zr, t.p_d * t.r_d * w).double()
                comp[f"{nm}:alpha:{side}"] = (view_HT_alpha(v, t.kappa * t.r_a * w).double() if t.kappa is not None
                                              else torch.zeros_like(comp[f"{nm}:rgb:{side}"]))
        nr = getattr(lin, "negative_ray_evaluation", None)
        if nr is not None:
            for nm, s in nr.score_pose_by_camera():
                comp[f"{nm}:negray"] = s.double()
        pr = -apply_prior(Lambda_prior, boxminus(x, x_prior)).double()
        comp["prior"] = pr[:, :6]
        return comp, pr

    @torch.no_grad()
    def strata_of(self, lin, rgbd_gt, cfg):
        from estimator.occlusion_likelihood import OcclusionLikelihood, evaluate
        pg = evaluate(OcclusionLikelihood("certified"), lin, rgbd_gt, cfg).per_gaussian()
        st = self.static.to(pg["front_weight"].device)
        return {"static_exposed": st & (pg["front_weight"] > 0.01),
                "static_interior": st & (pg["front_weight"] < 1e-3) & (pg["supported_weight"] > 0),
                "static_all": st, "support": ~st}

    @torch.no_grad()
    def summarize(self, comp, strata, axes):
        out = {}
        for sname, m in strata.items():
            n = int(m.sum())
            if n == 0:
                continue
            row = {"n": n}
            for k, F in comp.items():
                t = F[m.to(F.device), :3]
                row[k] = {"norm_mean": float(t.norm(dim=1).mean()),
                          "along_z": [float((t @ axes[c].to(t)).mean()) for c in range(axes.shape[0])]}
            out[sname] = row
        return out

    @torch.no_grad()
    def __call__(self, *, iteration, x, x_prior, Lambda_prior, lin, terms, rhs_pose, b, grouped, rgbd_gt, cfg,
                 solve_b=None, observable_map=None, **kw):
        comp, pr = self.components(lin, terms, rgbd_gt, x, x_prior, Lambda_prior)
        meas = sum(v for k, v in comp.items() if k != "prior")
        add_err = float((meas - rhs_pose.double()).norm() / rhs_pose.double().norm().clamp_min(1e-30))
        axes = camera_axes(lin)
        if iteration == 0:
            self.strata[self.ti] = self.strata_of(lin, rgbd_gt, cfg)
        strata = self.strata[self.ti]
        row = {"processing_index": self.ti, "iteration": iteration, "rhs_additivity_rel_err": add_err,
               "forces": self.summarize(comp, strata, axes)}
        # grouped RHS of each component (the map the solve actually uses)
        if grouped is not None:
            from estimator.observable_group import operators
            if observable_map is not None:
                _, red = operators(observable_map, grouped)
            else:
                def red(f):
                    return torch.cat([grouped.reduce(f[:, :6].float().contiguous()),
                                      grouped.reduce(f[:, 6:].float().contiguous())], 1)
            sid = grouped.part.supernode_id.long().to(self.static.device)
            frac_static = torch.zeros(grouped.num_supernodes, device=sid.device).index_add_(
                0, sid, self.static.float()) / torch.bincount(sid, minlength=grouped.num_supernodes).clamp_min(1)
            sn_static = frac_static >= 0.9
            gr = {}
            for k, F in comp.items():
                full = torch.zeros((F.shape[0], 12), dtype=torch.float32, device=F.device)
                full[:, :6] = F.float()
                if k == "prior":
                    full[:, 6:] = pr[:, 6:].float()
                r = red(full)
                gr[k] = float(r[sn_static.to(r.device), :3].norm(dim=1).mean()) if bool(sn_static.any()) else None
            tot = solve_b[sn_static.to(solve_b.device), :3].norm(dim=1).mean() if bool(sn_static.any()) else None
            row["grouped_rhs_static_supernodes"] = {"n_supernodes": int(sn_static.sum()),
                                                    "total_translation_norm_mean": None if tot is None else float(tot),
                                                    "per_component_translation_norm_mean": gr}
        if iteration == 0:
            self._first = row
            if self.ti in self.full_at:
                self.full[self.ti] = {k: v[:, :3].float().cpu() for k, v in comp.items()}
        self._last = {"first": getattr(self, "_first", None), "last": row}


class FrameCapture:
    """phase1_audit callback: corrections and positions; optionally the exact pre-update
    (state, P, obs, cfg, dt) at chosen processing indices (reference arm only)."""

    def __init__(self, static, keep_at=()):
        self.static, self.keep_at = static, set(keep_at)
        self.rows, self.kept, self.positions, self.velocity = [], {}, [], []

    @torch.no_grad()
    def __call__(self, *, ti, fid, prior_state, P_prev, state, P_post, diag, obs, cfg, dt, cameras,
                 observation_model, canonical):
        from estimator.process import process_predict
        pred = process_predict(prior_state, dt)
        self.positions.append(state.mu.detach().float().cpu())
        self.velocity.append(state.v.detach().float().cpu())
        self.rows.append({"processing_index": ti, "dt": float(dt),
                          "_pred_disp": (pred.mu - prior_state.mu).float().cpu(),
                          "_corr": (state.mu - pred.mu).float().cpu(), "_dv": (state.v - pred.v).float().cpu(),
                          "accepted": sum(1 for r in diag.rows if r.get("event") == "accepted"),
                          "stop": next((r.get("reason") for r in diag.rows if r.get("event") == "stop"), None)})
        if ti in self.keep_at:
            clone = dataclasses.replace(prior_state, **{f: getattr(prior_state, f).detach().clone()
                                                       for f in ("mu", "R", "v", "omega")})
            self.kept[ti] = {"seed": clone, "P": P_prev.detach().clone(), "obs": obs, "cfg": cfg, "dt": dt,
                             "cameras": cameras, "observation_model": observation_model}


def correction_summary(cap: FrameCapture, flog: ForceLog, axes) -> list:
    out = []
    for r in cap.rows:
        st = flog.strata.get(r["processing_index"], {})
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        for sname, m in st.items():
            m = m.cpu()
            if int(m.sum()) == 0:
                continue
            e = {}
            for key in ("_pred_disp", "_corr", "_dv"):
                t = r[key][m].double()
                e[key[1:]] = {"norm_mean_mm" if key != "_dv" else "norm_mean_mm_per_s": 1e3 * float(t.norm(dim=1).mean()),
                              "along_z_mm": [1e3 * float((t @ axes[c].cpu()).mean()) for c in range(axes.shape[0])]}
            row[sname] = e
        out.append(row)
    return out


# ---------------------------------------------------------------------------
def make_occlusion(kind, cal, deterministic):
    if kind is None:
        return None
    from estimator.occlusion_likelihood import OcclusionLikelihood
    return OcclusionLikelihood(kind, "unknown", deterministic, calibration=cal if kind == "calibrated" else None)


def run_arm(arm, truth, cameras, window, cfg, ns, static, cal, a, keep_at=(), full_at=()):
    from tools.kpart_pipeline import bootstrap_positions
    lik, grp = ARMS[arm]
    ns2 = argparse.Namespace(**vars(ns))
    ns2.occlusion_likelihood = make_occlusion(lik, cal, a.deterministic)
    flog = ForceLog(cfg.camera_names, static, full_at)
    cap = FrameCapture(static, keep_at)
    ns2.phase1_iteration_observer = flog
    ns2.phase1_audit = cap
    ns2.phase1_observability_dir = None
    if ns2.masked_observation is not None:
        ns2.masked_observation = dataclasses.replace(ns2.masked_observation, report=[])
    cfg2 = dataclasses.replace(cfg, group_update=grp)
    torch.manual_seed(0)
    t0 = time.time()
    pos, info = bootstrap_positions(truth, cameras, window, cfg2, None, ns2, verbose=False)
    flog.flush()
    return {"positions": [p.detach().float().cpu() for p in pos], "flog": flog, "cap": cap,
            "seconds": time.time() - t0}


def frozen_step(arm, kept, cfg, static, cal, a, names):
    from estimator.sparse_iekf import iekf_frame_step
    from estimator.supernodes import build_supernodes
    lik, grp = ARMS[arm]
    bcfg = dataclasses.replace(kept["cfg"], group_update=grp)
    part = build_supernodes(kept["canonical_mu"], target_size=bcfg.supernode_target_size,
                            num_supernodes=bcfg.supernode_count, knn=bcfg.supernode_knn, seed=bcfg.supernode_seed)
    flog = ForceLog(names, static, full_at={kept["ti"]})
    flog.begin_frame(kept["ti"], None)
    from estimator.process import process_predict
    seed = kept["seed"]
    x, P, d = iekf_frame_step(seed, kept["P"].clone(), kept["obs"], kept["cameras"], dt=kept["dt"], cfg=bcfg,
                              partition=part, group_geometry="current", iteration_observer=flog,
                              observation_model=kept["observation_model"],
                              occlusion=make_occlusion(lik, cal, a.deterministic))
    flog.flush()
    pred = process_predict(seed, kept["dt"])
    return {"flog": flog, "corr": (x.mu - pred.mu).float().cpu(), "dv": (x.v - pred.v).float().cpu(),
            "pred_disp": (pred.mu - seed.mu).float().cpu(),
            "accepted": sum(1 for r in d.rows if r.get("event") == "accepted"),
            "stop": next((r.get("reason") for r in d.rows if r.get("event") == "stop"), None)}


def disp_rows(pos, static, ref0):
    out = []
    for k, p in enumerate(pos):
        d = (p - ref0).norm(dim=1)[static.cpu()] * 1e3
        out.append({"processing_index": k, "static_median_mm": float(d.median()),
                    "static_p90_mm": float(torch.quantile(d, 0.9)), "static_p99_mm": float(torch.quantile(d, 0.99)),
                    "static_over_50mm": int((d > 50).sum())})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", default=None)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--direction", choices=("forward", "backward"), default="forward")
    ap.add_argument("--bootstrap-iters", type=int, default=12)
    ap.add_argument("--iekf-supernode-size", type=int, default=64)
    ap.add_argument("--phase1-partition", choices=("canonical",), default="canonical")
    ap.add_argument("--max-frames", type=int, default=40, help="processing frames per sequential arm (incl. frame 0)")
    ap.add_argument("--frozen-frames", type=int, nargs="*", default=[5, 15, 30])
    ap.add_argument("--reference-arm", default="legacy/legacy", choices=list(ARMS))
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--static-exclude-support", nargs="+", required=True,
                    help="support JSONs (members / dynamic_gaussians); static = outside all of them (evaluation only)")
    from estimator.masked_observation import add_cli as mask_cli
    from estimator.run_config import add_cli as rc_cli
    mask_cli(ap)
    rc_cli(ap, phase1=True)
    a = ap.parse_args(argv)
    if a.deterministic:
        from estimator.deterministic_ops import enable
        print(f"deterministic reductions: {enable()}")
    a.first_return_mode, a.first_return_prior, a.phase1_covariance_transport, a.gt_labels = "off", None, False, "off"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from audit_calibrated_phase1 import setup_window
    truth, cameras, window, cfg, ns, frame_ids = setup_window(a)
    names = list(cfg.camera_names)
    n = truth.mu.shape[0]
    excl = set()
    for p in a.static_exclude_support:
        excl |= load_support(p)
    static = torch.ones(n, dtype=torch.bool, device=truth.mu.device)
    static[torch.as_tensor(sorted(excl), dtype=torch.long, device=static.device)] = False
    # calibration from the FULL window (as production), before truncating the run
    cal = None
    if any(ARMS[k][0] == "calibrated" for k in a.arms):
        from estimator.occlusion_likelihood import OcclusionLikelihood, calibrate_reliability
        t0 = time.time()
        cal = calibrate_reliability(OcclusionLikelihood("calibrated"), cfg, names, [ob for ob, _ in window],
                                    ns.masked_observation.masks)
        (out / "reliability_calibration.json").write_text(json.dumps({k: v.to_manifest() for k, v in cal.items()},
                                                                     indent=1, default=float))
        print(f"calibration {time.time() - t0:.0f} s")
    win = window[:a.max_frames]
    if ns.masked_observation is not None:
        ms = ns.masked_observation
        ns.masked_observation = dataclasses.replace(ms, masks=ms.masks[:a.max_frames],
                                                    original_frames=ms.original_frames[:a.max_frames], report=[])
    frozen = [f for f in a.frozen_frames if 0 < f < len(win)]
    man = {"tool": "tools/diagnose_phase1_negative_ray.py", "read_only": True, "arms": a.arms,
           "reference_arm": a.reference_arm, "frames": [fid for _, fid in win], "frozen_processing_indices": frozen,
           "cameras": names, "num_gaussians": n, "static_gaussians": int(static.sum()),
           "static_exclude_support": a.static_exclude_support, "deterministic": bool(a.deterministic),
           "sensor_model_id": cfg.sensor_model.model_id if cfg.sensor_model is not None else None,
           "declared": {"rgb_model_sigma": cfg.rgb_model_sigma, "depth_model_sigma": cfg.depth_model_sigma,
                        "velocity_tau": getattr(cfg, "velocity_tau", None)}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    ref0 = truth.mu.detach().float().cpu()
    results, kept = {}, None
    order = [a.reference_arm] + [k for k in a.arms if k != a.reference_arm]
    for arm in order:
        if arm not in a.arms and arm != a.reference_arm:
            continue
        print(f"== arm {arm}")
        r = run_arm(arm, truth, cameras, win, cfg, ns, static, cal, a,
                    keep_at=frozen if arm == a.reference_arm else (), full_at=frozen)
        d = out / "arms" / arm.replace("/", "__")
        d.mkdir(parents=True, exist_ok=True)
        axes = None
        with open(d / "forces.jsonl", "w") as f:
            for row in r["flog"].rows:
                f.write(json.dumps(row, default=float) + "\n")
        # camera axes do not change over the run (fixed rig)
        from estimator.renderer_bridge import build_linearization
        axes = camera_axes(build_linearization(truth, cameras))
        with open(d / "corrections.jsonl", "w") as f:
            for row in correction_summary(r["cap"], r["flog"], axes):
                f.write(json.dumps(row, default=float) + "\n")
        dr = disp_rows(r["positions"], static, ref0)
        (d / "static_displacement.json").write_text(json.dumps(dr, indent=1))
        vel = torch.stack(r["cap"].velocity)
        np.savez_compressed(d / "tracks.npz", positions=torch.stack(r["positions"]).numpy().astype(np.float32),
                            velocity=vel.numpy().astype(np.float32), static=static.cpu().numpy())
        if r["flog"].full:
            np.savez_compressed(d / "forces_full.npz", **{f"{ti}|{k}": v.numpy() for ti, comp in r["flog"].full.items()
                                                          for k, v in comp.items()})
        results[arm] = {"final": dr[-1], "seconds": round(r["seconds"], 1),
                        "static_speed_median_mm_s": [1e3 * float(v[static.cpu()].norm(dim=1).median()) for v in vel]}
        print(f"   {r['seconds']:.0f} s; final static median {dr[-1]['static_median_mm']:.1f} mm, "
              f"p90 {dr[-1]['static_p90_mm']:.1f}, >50mm {dr[-1]['static_over_50mm']}")
        if arm == a.reference_arm:
            kept = r["cap"].kept
            for ti, k in kept.items():
                k["ti"], k["canonical_mu"] = ti, truth.mu
        del r
        torch.cuda.empty_cache()
    # ---- Part F: frozen states of the reference arm through every arm ---------------
    fro = {}
    for ti in sorted(kept or {}):
        fro[ti] = {}
        for arm in a.arms:
            r = frozen_step(arm, kept[ti], cfg, static, cal, a, names)
            axes = camera_axes(__import__("estimator.renderer_bridge", fromlist=["x"]).build_linearization(
                kept[ti]["seed"], kept[ti]["cameras"]))
            st = r["flog"].strata[ti]
            e = {"accepted": r["accepted"], "stop": r["stop"], "forces": r["flog"].rows[0] if r["flog"].rows else None}
            for sname, m in st.items():
                m = m.cpu()
                if int(m.sum()) == 0:
                    continue
                e[sname] = {k: {"norm_mean_mm": 1e3 * float(r[k][m].double().norm(dim=1).mean()),
                                "along_z_mm": [1e3 * float((r[k][m].double() @ axes[c].cpu()).mean())
                                               for c in range(axes.shape[0])]} for k in ("corr", "dv", "pred_disp")}
            fro[ti][arm] = e
            d = out / "frozen" / f"pi_{ti:03d}"
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{arm.replace('/', '__')}.npz", corr=r["corr"].numpy(), dv=r["dv"].numpy(),
                                **{k: v.numpy() for k, v in r["flog"].full.get(ti, {}).items()})
            print(f"   frozen {ti} {arm}: exposed-static correction along z "
                  f"{e.get('static_exposed', {}).get('corr', {}).get('along_z_mm')}")
    (out / "frozen.json").write_text(json.dumps(fro, indent=1, default=float))
    (out / "summary.json").write_text(json.dumps(results, indent=1, default=float))
    write_report(out, man, results, fro, names)
    return 0


def write_report(out, man, results, fro, names):
    L = ["# Phase-1 negative-ray ablation (auto-generated, read-only diagnostic)", "",
         f"{man['static_gaussians']} static Gaussians (outside {len(man['static_exclude_support'])} supports), "
         f"{len(man['frames'])} frames, cameras {names}, deterministic {man['deterministic']}.", "",
         "## Sequential arms: static displacement at the last frame", "",
         "| arm | median mm | p90 mm | p99 mm | > 50 mm | static speed median at end mm/s |", "|---|---|---|---|---|---|"]
    for arm, r in results.items():
        f = r["final"]
        L.append(f"| {arm} | {f['static_median_mm']:.1f} | {f['static_p90_mm']:.1f} | {f['static_p99_mm']:.1f} | "
                 f"{f['static_over_50mm']} | {r['static_speed_median_mm_s'][-1]:.1f} |")
    L += ["", "## Frozen states (reference arm's exact pre-update state; one step per arm)", "",
          "Mean accepted correction of exposed static Gaussians along each camera axis (mm; + = away from camera), "
          "and the mean negative-ray / off-mask force along the axes at iteration 0.", ""]
    for ti, arms in fro.items():
        L += [f"### processing index {ti}", "", "| arm | n exposed | correction along z (mm) | negray force along z "
              "| off-mask legacy force along z | prior force along z (last iter) |", "|---|---|---|---|---|---|"]
        for arm, e in arms.items():
            ex = e.get("static_exposed")
            fr = (e.get("forces") or {}).get("first") or {}
            fs = (fr.get("forces") or {}).get("static_exposed", {})
            neg = [fs.get(f"{c}:negray", {}).get("along_z") for c in names]
            off = [[sum(fs.get(f"{c}:{ch}:off", {}).get("along_z", [0] * len(names))[j] for ch in ("rgb", "depth", "alpha"))
                    for j in range(len(names))] for c in names]
            la = ((e.get("forces") or {}).get("last") or {}).get("forces", {}).get("static_exposed", {})
            L.append(f"| {arm} | {fs.get('n')} | {None if ex is None else [round(v, 3) for v in ex['corr']['along_z_mm']]} "
                     f"| {neg} | {off} | {la.get('prior', {}).get('along_z')} |")
        L.append("")
    L += ["Interpretation is NOT computed here; read frozen.json / forces.jsonl. Forces are -dL/dx (pose tangent "
          "translation part, units of the objective per metre)."]
    (out / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
