#!/usr/bin/env python
"""Phase 4 (unified spec 2026-11-11): geometry recovery under the FIXED Phase-3 joint, as three separate stages.

  --stage 4a   original-Gaussian material recovery -> <OUT>/4a_original_membership/
               python -u tools/run_phase4_recovery.py --stage 4a --source SRC --model MODEL --phase2-dir P2 \\
                   --phase3-dir P3 --direction forward --joint-type revolute --cameras cam_001,cam_002 --output OUT
  --stage 4b   new moving-surface proposals (zero births); REFUSES unless <phase4a>/complete.json is valid
               python -u tools/run_phase4_recovery.py --stage 4b --phase4a OUT/4a_original_membership [--output OUT]
  --stage 4c   validation + committed moving births; REFUSES unless valid 4A and 4B artifacts exist
               python -u tools/run_phase4_recovery.py --stage 4c --phase4a OUT/4a_original_membership \\
                   --phase4b OUT/4b_new_surface_proposals [--output OUT]
  --stage all  4a -> 4b -> 4c with hash checks between stages; stops at the first incomplete stage

4B and 4C rebuild exactly the 4A inputs from <phase4a>/manifest.json and verify them by hash. Completed stage
directories are immutable (re-invoking a completed stage only verifies it). --resume continues an interrupted 4A
from its checkpoint. --view-subset K (benchmark only): every artifact is marked NOT_FOR_RESULTS.
--recovery-frame-stride K: 4A on all views, 4B / 4C on every K-th frame (declared in their complete.json).
Phase 5 is NEVER started by this program.
"""
import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import recovery  # noqa: E402,F401  (pins this checkout's RGB-D rasterizer before anything else imports it)


def _cfg_cli(ap, cls, prefix):
    for f in fields(cls):
        t = {"int": int, "float": float, "bool": bool, "str": str}.get(f.type, f.type)
        if t not in (int, float, bool, str):
            continue
        name = f"--{prefix}-{f.name.replace('_', '-')}"
        if t is bool:
            ap.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            ap.add_argument(name, type=t, default=f.default)


def _cfg_from(args, cls, prefix):
    return cls(**{f.name: getattr(args, f"{prefix}_{f.name}") for f in fields(cls) if hasattr(args, f"{prefix}_{f.name}")})


def main(argv=None):
    from recovery import phase4_artifacts as AR
    from recovery import phase4_stages as S
    from recovery import phase4a_exact as X
    from recovery import phase4b_surface as B
    from recovery import phase4c_validate as C
    from recovery.exact_objective import ExactLossConfig
    from recovery.runlog import RunLog, stdout_line_buffered
    stdout_line_buffered()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="4a", choices=("4a", "4b", "4c", "all"))
    ap.add_argument("--output", default=None, help="the Phase-4 output directory (holds the three stage directories)")
    ap.add_argument("--run-id", default=None, help="optional sub-directory of --output")
    ap.add_argument("--phase4a", default=None, help="4B / 4C: a completed 4a_original_membership directory")
    ap.add_argument("--phase4b", default=None, help="4C: a completed 4b_new_surface_proposals directory")
    ap.add_argument("--resume", action="store_true", help="4A: continue from <output>/4a_original_membership/checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="contract + planned resources, nothing rendered")
    # upstream inputs (4A / all only; 4B / 4C read them from the 4A manifest)
    ap.add_argument("--source")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--direction", default="auto", choices=("auto", "forward", "backward"))
    ap.add_argument("--model")
    ap.add_argument("--phase2-dir")
    ap.add_argument("--phase3-dir")
    ap.add_argument("--joint-type", choices=("prismatic", "revolute"))
    ap.add_argument("--cameras", default=None)
    ap.add_argument("--frame-window", default=None)
    ap.add_argument("--sensor-model", default=None)
    ap.add_argument("--view-subset", type=int, default=0, help="benchmark only: frames per camera; NOT_FOR_RESULTS")
    ap.add_argument("--recovery-frame-stride", type=int, default=1,
                    help="4B / 4C only: scan and validate every K-th Phase-3 frame of every camera (4A always uses all "
                         "views); recorded as recovery_views in the 4B / 4C complete.json")
    ap.add_argument("--max-host-gb", type=float, default=8.0)
    ap.add_argument("--sh-degree", type=int, default=0)
    ap.add_argument("--sh-mode", default="local_SH", choices=("local_SH", "world_SH"))
    ap.add_argument("--seed", type=int, default=0)
    _cfg_cli(ap, X.Policy4A, "p4a")
    _cfg_cli(ap, ExactLossConfig, "loss")
    _cfg_cli(ap, B.Surface4B, "p4b")
    _cfg_cli(ap, C.Validate4C, "p4c")
    a = ap.parse_args(argv)
    import torch
    torch.manual_seed(a.seed)
    pol = _cfg_from(a, X.Policy4A, "p4a")
    loss = _cfg_from(a, ExactLossConfig, "loss")
    cfg4b = _cfg_from(a, B.Surface4B, "p4b")
    cfg4c = _cfg_from(a, C.Validate4C, "p4c")

    if a.stage in ("4a", "all"):
        missing = [k for k in ("source", "model", "phase2_dir", "phase3_dir", "joint_type", "output") if not getattr(a, k)]
        if missing:
            ap.error(f"--stage {a.stage} needs --{', --'.join(m.replace('_', '-') for m in missing)}")
        out = Path(a.output) / a.run_id if a.run_id else Path(a.output)
        setup = {k: getattr(a, k) for k in S.SETUP_KEYS}
    else:
        if not a.phase4a:
            ap.error(f"--stage {a.stage} needs --phase4a")
        if a.stage == "4c" and not a.phase4b:
            ap.error("--stage 4c needs --phase4b")
        ca = AR.require_complete(a.phase4a, AR.STAGE_A)                # refuse early, before any decoding
        if a.stage == "4c":
            AR.require_complete(a.phase4b, AR.STAGE_B)
        setup = json.loads((Path(a.phase4a) / "manifest.json").read_text())["setup"]
        if setup != ca["setup"]:
            raise SystemExit("4A manifest.json setup differs from its complete.json")
        out = Path(a.output) if a.output else Path(a.phase4a).parent
    if a.dry_run:
        log = RunLog(None)
        up, cams, pv, extra = S.context_from_setup(setup, out, log=log, dry_run=True)
        free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        print(json.dumps({"DRY_RUN": True, "output": str(out), "stage": a.stage, "views": len(pv),
                          "cameras": [c.name for c in cams], "decoded_bytes": pv.bytes_required(),
                          "gaussians": up.num_gaussians, "core": int(len(up.members)), "joint": up.joint.to_dict(),
                          "gpu_free_mb": free / 2 ** 20, "NOT_FOR_RESULTS": bool(a.view_subset),
                          "upstream_sha256": up.hashes}, indent=1, default=str))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    log = RunLog(out / f"phase4_{a.stage}.log")
    log(f"Phase 4 stage {a.stage} -> {out}  (argv: {' '.join(sys.argv[1:]) if argv is None else argv})")
    t0 = time.time()
    K = max(1, a.recovery_frame_stride)
    ctx = S.context_from_setup(setup, out, log=log, repo=REPO, recovery_stride=K if a.stage in ("4b", "4c") else 1)
    if a.stage == "4a":
        r = S.stage_4a(ctx, out, pol, loss, resume=a.resume, log=log)
        rc = 0 if r["complete"] else 4
    elif a.stage == "4b":
        r = S.stage_4b(ctx, out, Path(a.phase4a), cfg4b, log=log)
        rc = 0 if r["complete"] else 5
    elif a.stage == "4c":
        r = S.stage_4c(ctx, out, Path(a.phase4a), Path(a.phase4b), cfg4c, loss, log=log)
        rc = 0 if r["complete"] else 6
    else:
        rec = None
        if K > 1:
            def rec():
                log(f"4B / 4C on every {K}-th frame (4A used all {len(ctx.scene.views)} views)")
                return S.context_from_setup(setup, out, log=log, repo=REPO, recovery_stride=K)
        r = S.run_all(ctx, out, pol, loss, cfg4b, cfg4c, resume=a.resume, log=log, recovery_ctx=rec)
        rc = {None: 0, "4a": 4, "4b": 5, "4c": 6}[r["stopped_after"]]
    log(f"Phase 4 stage {a.stage} done in {(time.time() - t0) / 60:.1f} min, rc {rc}; decode counters "
        f"{ctx.pv.counters.to_dict()}; Phase 5 NOT started")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
