#!/usr/bin/env python3
"""Run the physics-first patch-isometry pipeline over every prepared scene.

Each scene is an independent subprocess, so one dataset crashing or running
out of GPU memory cannot take the sweep down with it, and a partial sweep
still leaves usable results on disk.

The comparison across datasets is the point.  A single scene's verdict says
little -- the pipeline has a lot of stages and any of them can starve -- but
the same stages measured on five objects show WHICH stage is the binding
constraint, and whether it is the same one everywhere.  So the summary table
reports the funnel per scene, not just the final answer.

Ground truth is post-hoc in every child run, exactly as in the single-scene
probe; this tool never sees a label itself.

Run:
    PYTHONPATH=. python tools/run_physics_first_all_inputs.py \\
        --output-root output/physics_first_sweep

    # quick pass over everything
    PYTHONPATH=. python tools/run_physics_first_all_inputs.py \\
        --output-root output/physics_first_sweep --preset smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROBES = {
    # The post-audit probe: multi-hypothesis retention, search-aware nulls,
    # per-hypothesis SE(3) uncertainty, contrastive predictive rendering.
    "corrected": REPO / "tools" / "run_rigid_patch_hypothesis_probe.py",
    # The rasterizer-coupled probe: does image support fix candidate recall?
    "rasterizer": REPO / "tools" / "run_rasterizer_coupled_transport_probe.py",
    # The pre-audit probe, kept so its numbers stay reproducible.
    "legacy": REPO / "tools" / "run_patch_isometry_probe.py",
}
PROBE = PROBES["legacy"]

# Search width only -- these bound cost, never a physical decision (§45).
PRESETS: dict[str, dict[str, Any]] = {
    "smoke": dict(source_anchors=32, target_anchors=32, null_samples=32,
                  candidates=8, null_replicates=4, bootstrap_draws=8,
                  rasterizer_sources=24, rasterizer_targets=48,
                  render_cameras=2, render_hypotheses=4, fm_iters=300),
    "standard": dict(source_anchors=96, target_anchors=128, null_samples=128,
                     candidates=16, null_replicates=8, bootstrap_draws=12,
                     rasterizer_sources=48, rasterizer_targets=128,
                     render_cameras=4, render_hypotheses=12, fm_iters=1000),
    "full": dict(source_anchors=192, target_anchors=256, null_samples=256,
                 candidates=24, null_replicates=12, bootstrap_draws=16,
                 rasterizer_sources=96, rasterizer_targets=256,
                 render_cameras=6, render_hypotheses=24, fm_iters=2000),
}


def discover_scenes(
    input_root: Path,
    output_root: Path,
    endpoint_suffix: str,
    iteration: int,
    only: set[str] | None,
) -> list[dict[str, Any]]:
    """Scenes that have both endpoint models at the requested iteration.

    A scene missing an endpoint is reported as skipped rather than silently
    dropped: "ran on four datasets" and "ran on four of five datasets" are
    different claims, and only one of them is honest.
    """
    scenes: list[dict[str, Any]] = []
    for scene in sorted(p for p in input_root.iterdir() if p.is_dir()):
        if only is not None and scene.name not in only:
            continue
        endpoints = output_root / f"{scene.name}{endpoint_suffix}" / "shared"
        frame0 = endpoints / "frame0_static"
        frame_t = endpoints / "frame_t_static"
        missing = [
            name for name, path in (("frame0", frame0), ("frameT", frame_t))
            if not (path / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply").is_file()
        ]
        scenes.append({
            "scene": scene.name,
            "root_scene": scene,
            "frame0_model_dir": frame0,
            "frameT_model_dir": frame_t,
            "has_ground_truth": (scene / "pointcloud" / "points3D_parts.ply").is_file(),
            "skip_reason": f"no endpoint model at iteration {iteration}: {', '.join(missing)}" if missing else None,
        })
    return scenes


def build_corrected_command(scene: dict[str, Any], out_dir: Path, args, preset: dict[str, Any]) -> list[str]:
    """The post-audit physics probe (P0-P9)."""
    command = [
        sys.executable, str(PROBES["corrected"]),
        "--frame0-model-dir", str(scene["frame0_model_dir"]),
        "--frameT-model-dir", str(scene["frameT_model_dir"]),
        "--root-scene", str(scene["root_scene"]),
        "--iteration", str(int(args.iteration)),
        "--output-dir", str(out_dir / "corrected"),
        "--knn", str(int(args.knn)),
        "--patch-level", str(int(args.patch_level)),
        "--alpha", str(float(args.alpha)),
        "--batch-size", str(int(args.batch_size)),
        "--seed", str(int(args.seed)),
        "--source-anchors", str(int(preset["source_anchors"])),
        "--target-anchors", str(int(preset["target_anchors"])),
        "--candidates", str(int(preset.get("candidates", 24))),
        "--null-replicates", str(int(preset.get("null_replicates", 12))),
        "--bootstrap-draws", str(int(preset.get("bootstrap_draws", 16))),
        "--render-cameras", str(int(preset["render_cameras"])),
        "--render-hypotheses", str(int(preset["render_hypotheses"])),
    ]
    if args.skip_render:
        command.append("--skip-render")
    if args.save_tensors:
        command.append("--save-tensors")
    if args.no_progress:
        command.append("--no-progress")
    command.append("--report-memory")
    return command


def build_rasterizer_command(scene: dict[str, Any], out_dir: Path, args, preset: dict[str, Any]) -> list[str]:
    """The rasterizer-coupled candidate-recall probe (R0-R5)."""
    command = [
        sys.executable, str(PROBES["rasterizer"]),
        "--frame0-model-dir", str(scene["frame0_model_dir"]),
        "--frameT-model-dir", str(scene["frameT_model_dir"]),
        "--root-scene", str(scene["root_scene"]),
        "--iteration", str(int(args.iteration)),
        "--output-dir", str(out_dir / "rasterizer"),
        "--knn", str(int(args.knn)),
        "--patch-level", str(int(args.patch_level)),
        "--seed", str(int(args.seed)),
        "--cameras", str(int(preset.get("render_cameras", 6))),
        "--source-patches", str(int(preset.get("rasterizer_sources", 48))),
        "--target-patches", str(int(preset.get("rasterizer_targets", 128))),
        "--candidates", str(int(preset.get("candidates", 16))),
    ]
    if args.save_tensors:
        command.append("--save-tensors")
    if args.no_progress:
        command.append("--no-progress")
    return command


def build_command(scene: dict[str, Any], out_dir: Path, args, preset: dict[str, Any]) -> list[str]:
    command = [
        sys.executable, str(PROBE),
        "--frame0-model-dir", str(scene["frame0_model_dir"]),
        "--frameT-model-dir", str(scene["frameT_model_dir"]),
        "--root-scene", str(scene["root_scene"]),
        "--iteration", str(int(args.iteration)),
        "--output-dir", str(out_dir),
        "--knn", str(int(args.knn)),
        "--patch-level", str(int(args.patch_level)),
        "--alpha", str(float(args.alpha)),
        "--batch-size", str(int(args.batch_size)),
        "--seed", str(int(args.seed)),
    ]
    for flag, key in (
        ("--source-anchors", "source_anchors"), ("--target-anchors", "target_anchors"),
        ("--null-samples", "null_samples"), ("--render-cameras", "render_cameras"),
        ("--render-hypotheses", "render_hypotheses"), ("--fm-iters", "fm_iters"),
    ):
        command += [flag, str(int(preset[key]))]
    if args.skip_render:
        command.append("--skip-render")
    if args.skip_fm:
        command.append("--skip-fm")
    if args.no_html:
        command.append("--no-html")
    if args.save_tensors:
        command.append("--save-tensors")
    if args.no_progress:
        command.append("--no-progress")
    return command


def summarise(report: dict[str, Any]) -> dict[str, Any]:
    """Pull the funnel out of one scene's JSON, tolerating missing stages."""
    spatial = report.get("P0_spatial", {})
    tests = report.get("P3_P5_patch_tests", {})
    components = report.get("P6_se3_components", {})
    render = report.get("P8_render", {})
    supervision = report.get("P9_supervision", {})
    flow = report.get("P10_flow_matching", {})
    verdict = report.get("verdict", {})
    truth = report.get("ground_truth") or {}
    return {
        "P0_bimodal": spatial.get("source_bimodal"),
        "P0_separation": spatial.get("source_separation"),
        "P0_map_dynamic": spatial.get("source_map_dynamic_fraction"),
        "P0_auc": truth.get("P0_auc"),
        "P0_f1": truth.get("P0_f1"),
        "P3_tested": tests.get("source_patches_tested"),
        "P5_metric_ok": tests.get("metric_accepted"),
        "P5_rigid_ok": tests.get("rigid_accepted"),
        "P5_both_ok": tests.get("physically_valid"),
        "P6_components": components.get("components"),
        "P8_tested": render.get("hypotheses_tested"),
        "P8_prefers_dynamic": render.get("prefers_dynamic_count"),
        "P8_residual_scale": render.get("residual_scale"),
        "P9_fm_pairs": supervision.get("fm_pairs"),
        "P10_trained": flow.get("run"),
        "P10_relative_error": (flow.get("evaluation") or {}).get("relative_endpoint_error"),
        "result": verdict.get("result"),
    }


def first_blocking_stage(row: dict[str, Any]) -> str:
    """Which stage actually stopped this scene.

    Reported because the final verdict is the same string whatever went
    wrong, and "no components formed" and "the renderer said no" call for
    completely different next steps.
    """
    if not row.get("P0_bimodal"):
        return "P0: mismatch not bimodal"
    if not row.get("P5_metric_ok"):
        return "P5: no significant metric isometry"
    if not row.get("P5_rigid_ok"):
        return "P5: no significant rigid transform"
    if not row.get("P5_both_ok"):
        return "P5: nothing passed both tests"
    if not row.get("P6_components"):
        return "P6: no transform-consistent component"
    if row.get("P8_tested") and not row.get("P8_prefers_dynamic"):
        return "P8: renderer prefers static"
    if not row.get("P9_fm_pairs"):
        return "P9: no supervised pairs"
    if not row.get("P10_trained"):
        return "P10: flow matching not trained"
    return "-- reached FM"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", default="input")
    parser.add_argument("--output-root", default="output/physics_first_sweep")
    parser.add_argument("--endpoint-root", default="output")
    parser.add_argument("--endpoint-suffix", default="_static_endpoints_14k")
    parser.add_argument("--iteration", type=int, default=14000)
    parser.add_argument("--scenes", default="", help="comma-separated subset; default every prepared scene")
    parser.add_argument("--preset", default="standard", choices=sorted(PRESETS))
    parser.add_argument("--probes", default="corrected,rasterizer",
                        help="comma-separated: corrected, rasterizer, legacy")

    parser.add_argument("--knn", type=int, default=48)
    parser.add_argument("--patch-level", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-fm", action="store_true")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--save-tensors", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200, help="seconds per scene")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--continue-on-failure", action="store_true", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    endpoint_root = Path(args.endpoint_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.scenes.split(",") if s.strip()} or None
    preset = PRESETS[args.preset]

    scenes = discover_scenes(input_root, endpoint_root, args.endpoint_suffix, int(args.iteration), only)
    runnable = [s for s in scenes if s["skip_reason"] is None]
    skipped = [s for s in scenes if s["skip_reason"] is not None]

    print(f"[sweep] {len(runnable)} runnable scene(s), {len(skipped)} skipped, preset '{args.preset}'")
    for scene in runnable:
        print(f"  + {scene['scene']:<18} gt={'yes' if scene['has_ground_truth'] else 'no'}")
    for scene in skipped:
        print(f"  - {scene['scene']:<18} {scene['skip_reason']}")
    if not runnable:
        print("[sweep] nothing to run")
        return 1

    if args.dry_run:
        preview = {"corrected": build_corrected_command, "rasterizer": build_rasterizer_command,
                   "legacy": build_command}
        for scene in runnable:
            for probe in [p.strip() for p in args.probes.split(",") if p.strip()]:
                print(f"\n# {scene['scene']} / {probe}")
                print(" ".join(preview[probe](scene, output_root / scene["scene"], args, preset)))
        return 0

    results: dict[str, Any] = {
        "preset": args.preset,
        "preset_values": preset,
        "iteration": int(args.iteration),
        "skipped": [{"scene": s["scene"], "reason": s["skip_reason"]} for s in skipped],
        "scenes": {},
    }

    builders = {
        "corrected": (build_corrected_command, "corrected/rigid_patch_hypothesis_probe.json"),
        "rasterizer": (build_rasterizer_command, "rasterizer/rasterizer_coupled_probe.json"),
        "legacy": (build_command, "patch_isometry_probe.json"),
    }
    wanted = [p.strip() for p in args.probes.split(",") if p.strip()]
    for probe in wanted:
        if probe not in builders:
            raise SystemExit(f"unknown probe {probe!r}; choose from {sorted(builders)}")

    for index, scene in enumerate(runnable, start=1):
        name = scene["scene"]
        out_dir = output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 78}\n[sweep {index}/{len(runnable)}] {name}\n{'=' * 78}")
        entry: dict[str, Any] = {
            "output_dir": str(out_dir), "has_ground_truth": scene["has_ground_truth"], "probes": {}
        }

        for probe in wanted:
            builder, report_name = builders[probe]
            command = builder(scene, out_dir, args, preset)
            log_path = out_dir / f"{probe}.log"
            started = time.perf_counter()
            print(f"  -- {probe} --")
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command, cwd=str(REPO), timeout=int(args.timeout),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    )
                    log.write(completed.stdout or "")
                probe_status = "ok" if completed.returncode == 0 else f"exit {completed.returncode}"
                for line in (completed.stdout or "").strip().splitlines()[-8:]:
                    print("     " + line)
            except subprocess.TimeoutExpired:
                probe_status = f"timeout after {args.timeout}s"
                print(f"     TIMEOUT after {args.timeout}s")
            probe_seconds = time.perf_counter() - started
            probe_entry: dict[str, Any] = {
                "status": probe_status, "seconds": probe_seconds, "log": str(log_path)
            }
            report_path = out_dir / report_name
            if report_path.is_file():
                try:
                    probe_entry["report"] = json.loads(report_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as error:
                    probe_entry["parse_error"] = str(error)
            entry["probes"][probe] = probe_entry
            print(f"     -> {probe_status} in {probe_seconds / 60:.1f} min")

        entry["status"] = (
            "ok" if all(p["status"] == "ok" for p in entry["probes"].values()) else "partial"
        )
        entry["seconds"] = sum(p["seconds"] for p in entry["probes"].values())
        results["scenes"][name] = entry
        _summarise_entry(entry)
        print(f"  => {entry['status']} in {entry['seconds'] / 60:.1f} min | "
              f"{entry.get('blocking_stage', '?')}")
        if entry["status"] != "ok" and not args.continue_on_failure:
            break

    _write_summary(results, output_root, runnable, args)
    return 0


def _summarise_entry(entry: dict[str, Any]) -> None:
    """Pull the headline numbers out of whichever probes ran."""
    corrected = entry["probes"].get("corrected", {}).get("report") or {}
    raster = entry["probes"].get("rasterizer", {}).get("report") or {}

    p0 = corrected.get("P0") or {}
    truth = (corrected.get("ground_truth") or {}).get("classification") or {}
    tests = corrected.get("P3_P5") or {}
    graph = corrected.get("P6") or {}
    gate = (corrected.get("verdict") or {}).get("gate") or {}
    entry.update({
        "P0_auc": truth.get("P0_auc"),
        "P0_bimodal": p0.get("source_bimodal"),
        "P5_hypotheses": tests.get("hypotheses_tested"),
        "P5_metric": tests.get("metric_significant"),
        "P5_se3": tests.get("se3_significant"),
        "P5_both": tests.get("both_significant"),
        "P5_search_fraction": ((tests.get("coverage") or {}).get("target_anchor_search_fraction")),
        "P6_components": graph.get("components"),
        "P6_edges": graph.get("consistent_edges"),
        "recall_image": ((raster.get("R5_recall_audit") or {}).get("image_supported") or {}).get("recall_at_8"),
        "recall_random": ((raster.get("R5_recall_audit") or {}).get("random") or {}).get("recall_at_8"),
        "image_support_helps": (raster.get("verdict") or {}).get("image_support_improves_recall"),
    })
    if not gate:
        entry["blocking_stage"] = "corrected probe produced no verdict"
    else:
        failed = [k for k, v in gate.items() if not v]
        entry["blocking_stage"] = failed[0] if failed else "-- reached FM gate"


def _write_summary(results, output_root, runnable, args) -> None:
    summary_path = output_root / "physics_first_sweep.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(f"\n{'=' * 124}\nNIGHT SWEEP -- preset '{args.preset}', probes {args.probes}\n{'=' * 124}")
    print(f"  {'scene':<16}{'P0 AUC':>8}{'hyp':>6}{'metric':>7}{'se3':>5}{'both':>6}{'srchFrac':>9}"
          f"{'comp':>6}{'rec@8 img':>10}{'rec@8 rnd':>10}  blocking stage")
    print("  " + "-" * 120)
    for name, entry in results["scenes"].items():
        def cell(key, fmt="{}"):
            value = entry.get(key)
            return "-" if value is None else fmt.format(value)
        print(
            f"  {name:<16}{cell('P0_auc', '{:.3f}'):>8}{cell('P5_hypotheses'):>6}"
            f"{cell('P5_metric'):>7}{cell('P5_se3'):>5}{cell('P5_both'):>6}"
            f"{cell('P5_search_fraction', '{:.3f}'):>9}{cell('P6_components'):>6}"
            f"{cell('recall_image', '{:.3f}'):>10}{cell('recall_random', '{:.3f}'):>10}"
            f"  {entry.get('blocking_stage', '?')}"
        )
    helped = [n for n, e in results["scenes"].items() if e.get("image_support_helps")]
    print(f"\n  image support improved candidate recall on {len(helped)}/{len(runnable)}"
          + (f": {', '.join(helped)}" if helped else ""))
    blocking: dict[str, list[str]] = {}
    for name, entry in results["scenes"].items():
        blocking.setdefault(entry.get("blocking_stage", "?"), []).append(name)
    print("\n  where the pipeline stops:")
    for stage, names in sorted(blocking.items(), key=lambda kv: -len(kv[1])):
        print(f"    {len(names)}x  {stage:<44} {', '.join(names)}")
    print(f"\n[sweep] wrote {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
