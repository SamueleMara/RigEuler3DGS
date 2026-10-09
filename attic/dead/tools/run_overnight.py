"""Overnight sweep: the current pipeline across every scene in input/.

Configuration under test (everything implemented so far, nothing mixed in that a
spec told us to isolate):

    high-precision bootstrap initialization   (seed, do not segment)
    soft persistent labels + validity-aware alpha
    render-dominant temporal score, memory ON
    NEW render-response contrastive refinement
    motion-boundary split OFF                 (kept separate per its own §47/§48)

Plus the §45 A/B on the two scenes where we have a baseline to compare against:
contrast OFF vs contrast ON, identical otherwise.

Resumable: a run whose artifacts already look complete is skipped, so an
interrupted sweep can simply be relaunched. Failures are logged and the sweep
continues to the next run rather than losing the whole night.

    python tools/run_overnight.py                 # launch
    python tools/run_overnight.py --report        # score whatever finished
    python tools/run_overnight.py --dry-run       # list the plan
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCENES = (
    "USB_100109",
    "storage_45135",
    "blade_103706",
    "fridge_10905",
    "stapler_103111",
    "dataset_samuele_fridge_fixedK_episode_000000",
    "dataset_samuele_fridge_fixedK_episode_000001",
    "dataset_samuele_fridge_fixedK_episode_000002",
)

# §45's ablation only where a meaningful baseline exists.
AB_SCENES = ("storage_45135", "USB_100109")

BASE_OVERRIDES = {
    "init_mode": "high_precision_bootstrap",
    "outer_soft_labels": True,
    "enable_temporal_credit": True,
    "enable_motion_boundary_split": False,
    "part_outer_iters": 6,
}


def plan(outer: int | None = None) -> list[dict]:
    """Every run this sweep will attempt, in priority order."""
    runs: list[dict] = []
    # The A/B pairs first: they answer the specific question the last spec asked.
    for scene in AB_SCENES:
        for arm, contrast in (("contrastON", True), ("contrastOFF", False)):
            runs.append({"scene": scene, "out": f"night_{scene}_{arm}",
                         "contrast": contrast, "tag": f"{scene}/{arm}"})
    # Then broad coverage with the full configuration.
    for scene in SCENES:
        if scene in AB_SCENES:
            continue
        runs.append({"scene": scene, "out": f"night_{scene}_contrastON",
                     "contrast": True, "tag": f"{scene}/contrastON"})
    if outer is not None:
        for r in runs:
            r["outer"] = int(outer)
    return runs


def is_complete(out: str, expected_outers: int) -> bool:
    debug = REPO / "output" / out / "dynamic_debug" / "nested_part_embedding_killing"
    if not debug.is_dir():
        return False
    soft = len(list(debug.glob("soft_outer_*.pt")))
    ply = list((REPO / "output" / out).glob("point_cloud/iteration_*/point_cloud.ply"))
    return soft >= expected_outers and bool(ply)


def execute(runs: list[dict], *, skip_complete: bool = True) -> dict:
    from tools.run_render_graph import run as run_one

    results: dict[str, dict] = {}
    status_path = REPO / "output" / "overnight_status.json"
    started = time.time()

    for index, spec in enumerate(runs, 1):
        tag = spec["tag"]
        out = spec["out"]
        outers = int(spec.get("outer", BASE_OVERRIDES["part_outer_iters"]))
        header = f"[{index}/{len(runs)}] {tag} -> output/{out}"

        if skip_complete and is_complete(out, outers):
            print(f"\n{'='*78}\n{header}   SKIPPED (already complete)\n{'='*78}", flush=True)
            results[tag] = {"status": "skipped", "out": out}
            continue

        overrides = dict(BASE_OVERRIDES)
        overrides["part_outer_iters"] = outers
        overrides["enable_render_response_contrast"] = bool(spec["contrast"])
        print(f"\n{'='*78}\n{header}\n   {overrides}\n{'='*78}", flush=True)

        t0 = time.time()
        try:
            run_one(spec["scene"], out, overrides=overrides, evaluate=False)
            elapsed = time.time() - t0
            results[tag] = {"status": "ok", "out": out, "seconds": round(elapsed, 1)}
            print(f"--- {tag} finished in {elapsed/60:.1f} min", flush=True)
        except Exception as error:                       # keep the sweep alive
            elapsed = time.time() - t0
            results[tag] = {"status": "failed", "out": out, "seconds": round(elapsed, 1),
                            "error": f"{type(error).__name__}: {error}"}
            print(f"--- {tag} FAILED after {elapsed/60:.1f} min: {error}", flush=True)
            traceback.print_exc()

        status_path.write_text(json.dumps(
            {"elapsed_minutes": round((time.time() - started) / 60, 1),
             "completed": index, "total": len(runs), "results": results}, indent=2))

    print(f"\nsweep finished in {(time.time() - started)/60:.1f} min")
    return results


def report(runs: list[dict]) -> None:
    from tools.run_soft_ablation import score_run

    print(f"\n{'run':<52s} {'per-outer F1':<34s} {'best':>6s} {'final':>6s} {'prec':>6s} {'rec':>6s}")
    print("-" * 118)
    for spec in runs:
        out = spec["out"]
        try:
            r = score_run(out)
        except Exception as error:
            print(f"{spec['tag']:<52s} score failed: {error}")
            continue
        if not r.get("available"):
            print(f"{spec['tag']:<52s} {r.get('reason', 'incomplete')}")
            continue
        if not r.get("gt"):
            n = r.get("num_outer", 0)
            print(f"{spec['tag']:<52s} {n} outers, no per-Gaussian gt_part")
            continue
        f = " ".join(f"{v:.2f}" for v in r["per_outer_f1"])
        print(f"{spec['tag']:<52s} {f:<34s} {r['best']['f1']:>6.2f} {r['final']['f1']:>6.2f} "
              f"{r['final']['precision']:>6.2f} {r['final']['recall']:>6.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="score finished runs, launch nothing")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--outer", type=int, default=None, help="override outer iterations")
    ap.add_argument("--scenes", nargs="+", default=None, help="restrict to these scenes")
    ap.add_argument("--no-skip", action="store_true", help="re-run even if complete")
    args = ap.parse_args()

    runs = plan(args.outer)
    if args.scenes:
        keep = set(args.scenes)
        runs = [r for r in runs if r["scene"] in keep]

    if args.dry_run:
        print(f"{len(runs)} runs planned:")
        for index, r in enumerate(runs, 1):
            done = " (complete)" if is_complete(
                r["out"], int(r.get("outer", BASE_OVERRIDES["part_outer_iters"]))) else ""
            print(f"  {index:2d}. {r['tag']:<52s} -> output/{r['out']}{done}")
        return
    if args.report:
        report(runs)
        return

    execute(runs, skip_complete=not args.no_skip)
    report(runs)


if __name__ == "__main__":
    main()
