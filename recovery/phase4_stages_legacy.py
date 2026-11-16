"""SUPERSEDED (kept for the pre-unified regression only; see recovery/phase4_stages.py). Phase 4 stage control: 4a | 4b | 4c | all, one run directory per run id, atomic checkpoints, resume with
input-hash proof (directive 2026-11-11, section 7). Never falls through into Phase 5.

<output>/<run_id>/
  manifest.json            upstream contract + sha256 of every input, code state, argv, stage status
  config_resolved.json     every config field (hashed: a resume with a different config is refused)
  cache_manifest.json      one-decode view store: counts, bytes, checksums, memory
  phase4.log               line-buffered log (stdout + resource stamps)
  stage4a/                 4A checkpoint (state.npz + checkpoint.json, atomic)
  support_completion/      4A outputs (statuses, render labels, groups, exact evidence, audits, sensitivity)
  stage4a_complete.json    written last; `complete: false` when a declared resource limit stopped the stage
  discovery/ stage4b_complete.json, acceptance/ final_canonical/ stage4c_complete.json   (4B / 4C)
A stage refuses to start unless the previous stage's marker says complete (no stale or partial inputs).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .runlog import RunLog, atomic_json


def config_hash(cfgs: dict) -> str:
    """Hash of the scientific configuration (resource limits excluded: a resumed run may get more budget)."""
    from .phase4a_exact import RESOURCE_FIELDS
    c = json.loads(json.dumps(cfgs, default=str))
    for k in RESOURCE_FIELDS:
        c.get("policy4a", {}).pop(k, None)
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def input_proof(up, pv_manifest: dict) -> dict:
    return {"upstream_sha256": dict(up.hashes), "views_combined_sha256": pv_manifest.get("combined_sha256"),
            "views": pv_manifest.get("views"), "joint": up.joint.to_dict()}


class Run:
    def __init__(self, run_dir: Path, log: RunLog, up, cams, views, pv, scene, cfgs: dict, manifest: dict):
        self.dir, self.log, self.up, self.cams, self.views, self.pv = run_dir, log, up, cams, views, pv
        self.scene, self.cfgs, self.manifest = scene, cfgs, manifest

    def marker(self, stage):
        p = self.dir / f"stage{stage}_complete.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def write_manifest(self):
        atomic_json(self.dir / "manifest.json", self.manifest)


def open_run(args, cfgs: dict, argv, repo: Path):
    """Create or resume a run directory; verify the input proof and config hash on resume."""
    from .gaussians import GaussianTensors
    from .io_setup import git_state, setup_phase4
    from .material_completion import Phase4Scene
    from .rgbd_loss import LossConfig
    from .view_cache import write_cache_manifest
    if args.resume:
        run_dir = Path(args.resume)
        if not (run_dir / "manifest.json").is_file():
            raise SystemExit(f"--resume {run_dir}: no manifest.json")
    else:
        run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
        run_dir = Path(args.output) / run_id
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"{run_dir} exists and is not empty: choose another --run-id or --resume it "
                             "(partial results are never overwritten)")
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(None if args.dry_run else run_dir / "phase4.log")
    log(f"Phase 4 run directory {run_dir} ({'RESUME' if args.resume else 'new'})")
    up, cams, views, pv, extra = setup_phase4(args, log=log, spill_dir=run_dir / "view_spill")
    chash = config_hash(cfgs)
    if args.dry_run:
        return run_dir, log, up, cams, pv, extra, chash
    cm = write_cache_manifest(run_dir / "cache_manifest.json", pv)
    proof = input_proof(up, cm)
    if args.resume:
        old = json.loads((run_dir / "manifest.json").read_text())
        if old.get("input_proof") != json.loads(json.dumps(proof, default=str)):
            raise SystemExit("resume refused: the inputs (upstream hashes / decoded views) differ from the run's")
        if old.get("config_sha256") != chash:
            raise SystemExit("resume refused: the configuration differs from the run's config_resolved.json")
        manifest = old
        manifest.setdefault("resumes", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": argv})
    else:
        manifest = {"stage_runner": "phase4", "run_dir": str(run_dir), "upstream": up.manifest(), "input_proof": proof,
                    "config_sha256": chash, "code": git_state(repo), "argv": argv, **extra,
                    "views": len(views), "NOT_FINAL": bool(extra.get("view_subset")),
                    "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        atomic_json(run_dir / "config_resolved.json", cfgs)
    g = GaussianTensors.load_ply(up.ply)
    if len(g) != up.num_gaussians:
        raise SystemExit(f"PLY has {len(g)} rows, header says {up.num_gaussians}")
    g.active_sh_degree = min(cfgs["sh_degree"], g.max_sh_degree)
    scene = Phase4Scene(g, up.joint, views, LossConfig(), sh_mode=cfgs["sh_mode"], sh_degree=g.active_sh_degree)
    run = Run(run_dir, log, up, cams, views, pv, scene, cfgs, manifest)
    run.write_manifest()
    return run


def joint_files_proof(up) -> dict:
    from .contracts import sha256
    return {"geometry": sha256(up.joint.geometry_file), "trajectory": sha256(up.joint.trajectory_file)}


def stage_4a(run: Run, pol, cfg, resume: bool) -> dict:
    from . import phase4a_exact as X
    log = run.log
    j0 = joint_files_proof(run.up)
    ph = X.Phase4A(run.scene, run.pv, run.up.members, pol, cfg, log=log, checkpoint_dir=run.dir / "stage4a")
    res = ph.run(resume=resume)
    X.write_outputs(run.dir, ph, res, run.up.manifest())
    if joint_files_proof(run.up) != j0:
        raise RuntimeError("Phase-3 joint files changed during 4A")
    counts = {s: int((res["status"] == s).sum()) for s in X.STATUS_CODES}
    marker = {"stage": "4a", "complete": bool(res["complete"]) and not res["aborted"], "aborted": res["aborted"],
              "stop_reason": res["stop_reason"], "counts": counts, "seconds": res["seconds"],
              "counters": res["counters"], "resources": log.resources(), "joint_files_sha256": j0,
              "NOT_FINAL": bool(run.manifest.get("NOT_FINAL")),
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_json(run.dir / "stage4a_complete.json", marker)
    run.manifest.setdefault("stages", {})["4a"] = {k: marker[k] for k in ("complete", "aborted", "stop_reason", "counts")}
    run.write_manifest()
    log(f"4A {'COMPLETE' if marker['complete'] else 'INCOMPLETE'}: {counts}; {res['seconds'] / 60:.1f} min")
    return marker


def load_4a(run: Run):
    """Render labels and statuses of a complete 4A stage."""
    m = run.marker("4a")
    if not m or not m["complete"]:
        raise SystemExit("stage 4a is not complete in this run directory; 4b/4c refuse to start")
    d = run.dir / "support_completion"
    st = np.load(d / "material_status.npz")
    rl = np.load(d / "render_label.npz")
    legend = {v: k for k, v in json.loads(str(st["legend"])).items()}
    status = np.array([legend[int(c)] for c in st["status_code"]], dtype=object)
    return status, rl["render_label"].astype(bool)


def run_all_in_memory(scene, pv, core, pol, loss_cfg, cfg4b, cfg4c, log=print, noise_path=None):
    """4A -> 4B -> 4C without a run directory (procedural gate tests). Same functions as the CLI stages."""
    from . import phase4a_exact as X
    from . import phase4b_exact as B
    from . import phase4c_exact as C
    ph = X.Phase4A(scene, pv, core, pol, loss_cfg, log=log)
    r4a = ph.run()
    out = {"phase4a": r4a, "phase": ph}
    if r4a["aborted"] or not r4a["complete"]:
        return out
    labels = torch.as_tensor(r4a["render_label"].astype(bool), device=scene.device)
    noise = B.DepthNoise(noise_path, cfg4b)
    codes, stats, obs, cnt, roles = B.run_4b(scene, pv, labels, r4a["status"], cfg4b, noise, log=log)
    cells, lab, cands = B.build_candidates(obs, cfg4b)
    out.update(codes=codes, view_stats=stats, obs=obs, counters_4b=cnt, roles=roles, cells=cells, candidates=cands)
    undecided = torch.as_tensor(np.isin(r4a["status"], [X.UNRESOLVED, X.SUSPECT, X.UNTESTED]), device=scene.device)
    n_orig = len(scene.g)
    scene.g.lineage["material"] = r4a["render_label"].astype(np.int8)
    s4 = C.Scene4C(scene, labels)
    E_before = C.scene_audit(s4, pv, loss_cfg)
    decisions, accepted = C.run_4c(s4, pv, roles, cands, undecided, cfg4c, loss_cfg, log=log)
    C.leave_one_out(s4, pv, roles, accepted, decisions, cfg4c, loss_cfg, log=log)
    E_after = C.scene_audit(s4, pv, loss_cfg)
    scene.g.lineage["material"] = s4.labels.cpu().numpy().astype(np.int8)
    out.update(decisions=decisions, accepted=accepted, n_original=n_orig, n_final=len(scene.g), labels_final=s4.labels,
               E_before_4c=E_before, E_after_4c=E_after)
    return out
