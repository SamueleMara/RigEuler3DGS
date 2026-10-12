"""Compare a legacy run against a carving run, and refuse if they are not comparable.

    python scripts/compare_carving_ab.py \
        --legacy /abs/OUT_A --carving /abs/OUT_B --output /abs/COMPARISON

Two runs are an A/B of one intervention only if they started from the same
Gaussians and saw the same cameras in the same order. Both are checked before
any metric is read, and a mismatch is an error rather than a caveat in the
summary: a comparison of two runs that diverged for an unrelated reason looks
exactly like a comparison of two runs that diverged because of the treatment.

Reads the CSVs the training run wrote. No torch, no CUDA, no training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# The intended differences. Anything else differing in initial_state.json is
# a confound and is reported.
EXPECTED_DIFFERENCES = {"branch", "splatfacto_carving", "initial_state_sha256",
                        "carving_background_seed"}

TRACKED = ("E_background_global", "foreground_opacity_global", "N_gaussians",
           "anisotropy_median", "anisotropy_p99", "anisotropy_max",
           "max_scale", "max_scale_over_bbox_diag", "nonfinite_parameters")


class NotComparable(RuntimeError):
    pass


def _debug_dir(run: Path) -> Path:
    directory = run / "carving_debug"
    if not directory.is_dir():
        raise NotComparable(
            f"{run} has no carving_debug/. Re-run it with --carving_debug; "
            "without it there is nothing to compare.")
    return directory


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise NotComparable(f"missing {path}")
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    if value is None or value == "" or value == "N/A":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_initial_state(a: Path, b: Path) -> dict:
    """Same starting Gaussians, bit for bit, or this is not an A/B."""
    state_a = json.loads((a / "initial_state.json").read_text())
    state_b = json.loads((b / "initial_state.json").read_text())
    hash_a = state_a["initial_state_sha256"]["combined"]
    hash_b = state_b["initial_state_sha256"]["combined"]
    if hash_a != hash_b:
        differing = [name for name in state_a["initial_state_sha256"]["per_tensor"]
                     if state_a["initial_state_sha256"]["per_tensor"].get(name)
                     != state_b["initial_state_sha256"]["per_tensor"].get(name)]
        raise NotComparable(
            "initial Gaussians differ, so this is not a controlled A/B.\n"
            f"  legacy  {hash_a}\n  carving {hash_b}\n"
            f"  tensors that differ: {', '.join(differing) or 'combined digest only'}")

    confounds = {}
    for key in set(state_a) | set(state_b):
        if key in EXPECTED_DIFFERENCES or key == "cameras":
            continue
        if state_a.get(key) != state_b.get(key):
            confounds[key] = (state_a.get(key), state_b.get(key))
    return {"combined_sha256": hash_a, "legacy": state_a, "carving": state_b,
            "confounds": confounds}


def check_camera_trace(a: Path, b: Path) -> dict:
    """Same camera at every iteration, or the runs saw different data."""
    trace_a = _read_csv(a / "camera_trace.csv")
    trace_b = _read_csv(b / "camera_trace.csv")
    if len(trace_a) != len(trace_b):
        raise NotComparable(
            f"camera traces differ in length: {len(trace_a)} vs {len(trace_b)} "
            "iterations. The runs did not train for the same number of steps.")
    for row_a, row_b in zip(trace_a, trace_b):
        if row_a["iteration"] != row_b["iteration"] or row_a["image_name"] != row_b["image_name"]:
            raise NotComparable(
                f"camera order diverges at iteration {row_a['iteration']}: "
                f"legacy saw {row_a['image_name']}, carving saw {row_b['image_name']}. "
                "The background RNG is not isolated from the camera RNG.")
    return {"iterations": len(trace_a)}


def merge(a_rows: list[dict], b_rows: list[dict]) -> list[dict]:
    """Join on iteration, never on row order."""
    by_iteration_a = {int(r["iteration"]): r for r in a_rows}
    by_iteration_b = {int(r["iteration"]): r for r in b_rows}
    shared = sorted(set(by_iteration_a) & set(by_iteration_b))
    merged = []
    for iteration in shared:
        row_a, row_b = by_iteration_a[iteration], by_iteration_b[iteration]
        row = {"iteration": iteration}
        for field in TRACKED:
            row[f"{field}_A"] = _number(row_a.get(field))
            row[f"{field}_B"] = _number(row_b.get(field))
        bg_a, bg_b = row["E_background_global_A"], row["E_background_global_B"]
        row["delta_E_bg"] = (bg_b - bg_a) if (bg_a is not None and bg_b is not None) else None
        row["ratio_E_bg"] = (bg_a / bg_b) if (bg_a is not None and bg_b
                                              not in (None, 0.0)) else None
        merged.append(row)
    if not merged:
        raise NotComparable("no iteration appears in both runs' metrics_global.csv")
    return merged


def plot(merged: list[dict], output: Path) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return f"not produced ({exc})"
    iterations = [r["iteration"] for r in merged]
    panels = [
        ("E_background_global", "background opacity leakage", True),
        ("anisotropy_median", "anisotropy, median", False),
        ("anisotropy_p99", "anisotropy, 99th percentile", False),
        ("anisotropy_max", "anisotropy, max", False),
        ("N_gaussians", "Gaussians", False),
        ("max_scale_over_bbox_diag", "max scale / bbox diagonal", False),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (field, title, log) in zip(axes.flat, panels):
        for suffix, label, colour in (("_A", "legacy", "tab:gray"),
                                      ("_B", "carving", "tab:blue")):
            values = [r[field + suffix] for r in merged]
            if any(v is not None for v in values):
                axis.plot(iterations, values, label=label, color=colour)
        axis.set_title(title)
        axis.set_xlabel("iteration")
        if log:
            axis.set_yscale("log")
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=120)
    plt.close(figure)
    return "comparison.png"


def verdict(merged: list[dict]) -> tuple[str, str]:
    """One of the four classifications, with the reason it was chosen."""
    final = merged[-1]
    bg_a, bg_b = final["E_background_global_A"], final["E_background_global_B"]
    if bg_a is None or bg_b is None:
        return ("Inconclusive",
                "E_background is undefined in at least one run at the final "
                "checkpoint, so there is nothing to compare.")
    if any((r["nonfinite_parameters_A"] or 0) or (r["nonfinite_parameters_B"] or 0)
           for r in merged):
        return ("Inconclusive",
                "a run holds non-finite parameters; its metrics describe a "
                "broken model, not a trained one.")

    comparable = [r for r in merged
                  if r["E_background_global_A"] is not None
                  and r["E_background_global_B"] is not None and r["iteration"] > 0]
    lower = sum(1 for r in comparable
                if r["E_background_global_B"] < r["E_background_global_A"])
    consistent = comparable and lower >= 0.8 * len(comparable)
    improved = bg_b < bg_a

    p99_a, p99_b = final["anisotropy_p99_A"], final["anisotropy_p99_B"]
    max_a, max_b = final["anisotropy_max_A"], final["anisotropy_max_B"]
    shape_better = (p99_a is not None and p99_b is not None and p99_b < p99_a
                    and max_a is not None and max_b is not None and max_b < max_a)

    if improved and consistent and shape_better:
        return ("Supports transparency-supervision hypothesis",
                f"carving lowers E_background at {lower} of {len(comparable)} "
                f"checkpoints and at the final one ({bg_a:.3e} -> {bg_b:.3e}), "
                f"and both anisotropy p99 ({p99_a:.2f} -> {p99_b:.2f}) and max "
                f"({max_a:.2f} -> {max_b:.2f}) are lower.")
    if improved and consistent:
        return ("Background fixed, anisotropy not fixed",
                f"E_background falls {bg_a:.3e} -> {bg_b:.3e} and is lower at "
                f"{lower} of {len(comparable)} checkpoints, but anisotropy does "
                f"not improve (p99 {p99_a} -> {p99_b}, max {max_a} -> {max_b}). "
                "Transparency supervision addresses background occupancy; "
                "primitive shape needs a separate experiment.")
    if improved:
        return ("Background fixed, anisotropy not fixed",
                f"E_background is lower at the final checkpoint "
                f"({bg_a:.3e} -> {bg_b:.3e}) but only at {lower} of "
                f"{len(comparable)} checkpoints, so the effect is not steady "
                "over training. Treat as weak evidence and run longer.")
    return ("No improvement",
            f"E_background did not fall ({bg_a:.3e} -> {bg_b:.3e}). Before "
            "looking further afield, check that the masks align with the "
            "images, that their values are 0/255, and that the composited "
            "target is what the renderer is being asked to match. This says "
            "nothing about the rasteriser being defective.")


def write_summary(output: Path, args, initial, trace, merged, label, reason, plot_note):
    first, final = merged[0], merged[-1]

    def row(field, record):
        a, b = record[f"{field}_A"], record[f"{field}_B"]
        fmt = (lambda v: "N/A" if v is None
               else (f"{v:.4e}" if abs(v) < 1e-2 and v != 0 else f"{v:.4f}"))
        return f"| {field} | {fmt(a)} | {fmt(b)} |"

    lines = [
        "# Transparency carving: legacy vs carving",
        "",
        f"- legacy run: `{args.legacy}`",
        f"- carving run: `{args.carving}`",
        f"- iterations traced: {trace['iterations']}",
        f"- initial Gaussians: {initial['legacy'].get('num_initial_gaussians')}",
        f"- training cameras: {initial['legacy'].get('num_train_cameras')}"
        f" (masks: {initial['legacy'].get('num_masks')})",
        f"- git commit: {initial['legacy'].get('git_commit')}",
        "",
        "## Is this a controlled A/B?",
        "",
        f"- initial-state SHA256 **matches**: `{initial['combined_sha256']}`",
        f"- camera traces **identical** across {trace['iterations']} iterations",
    ]
    if initial["confounds"]:
        lines.append("- **configuration differs beyond the intended flags:**")
        for key, (left, right) in sorted(initial["confounds"].items()):
            lines.append(f"  - `{key}`: legacy `{left}` vs carving `{right}`")
    else:
        lines.append("- configuration otherwise identical")
    lines += [
        "",
        "Bitwise GPU reproducibility is not claimed: the rasteriser uses "
        "non-deterministic atomics. What is guaranteed is identical initial "
        "tensors, an identical view schedule, and no code difference outside "
        "the treatment branch. Splitting decisions that diverge *after* the "
        "intervention are an outcome, not a confound.",
        "",
        f"## At initialization (iteration {first['iteration']})",
        "",
        "| metric | legacy | carving |", "| --- | --- | --- |",
    ]
    lines += [row(f, first) for f in TRACKED]
    lines += [
        "",
        "These must agree: same model, same measurement, independent of branch.",
        "",
        f"## At the final checkpoint (iteration {final['iteration']})",
        "",
        "| metric | legacy | carving |", "| --- | --- | --- |",
    ]
    lines += [row(f, final) for f in TRACKED]

    bg_a, bg_b = final["E_background_global_A"], final["E_background_global_B"]
    if bg_a is not None and bg_b not in (None, 0.0):
        lines += ["", f"Background leakage ratio legacy/carving: **{bg_a / bg_b:.1f}x**"]
    lines += [
        "",
        "## Verdict",
        "",
        f"**{label}**",
        "",
        reason,
        "",
        "The training RGB losses of the two branches are not compared: they "
        "score against different composites and the numbers are not "
        "commensurable. E_background and the shape statistics are the "
        "comparable outcomes.",
        "",
        "This measures an association under one controlled intervention: "
        "random-background compositing. It does not implicate the CUDA "
        "rasteriser and says nothing about Splatfacto's other defaults.",
        "",
        f"Artifacts: `comparison.csv`, `{plot_note}`",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy", required=True)
    ap.add_argument("--carving", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    legacy = _debug_dir(Path(args.legacy).expanduser().resolve())
    carving = _debug_dir(Path(args.carving).expanduser().resolve())
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    try:
        initial = check_initial_state(legacy, carving)
        trace = check_camera_trace(legacy, carving)
    except NotComparable as exc:
        print(f"REFUSED: these runs are not a controlled A/B.\n\n{exc}")
        (output / "summary.md").write_text(
            f"# Comparison refused\n\nThese runs are not a controlled A/B.\n\n```\n{exc}\n```\n")
        return 2

    merged = merge(_read_csv(legacy / "metrics_global.csv"),
                   _read_csv(carving / "metrics_global.csv"))

    columns = ["iteration", "E_background_global_A", "E_background_global_B",
               "delta_E_bg", "ratio_E_bg"]
    columns += [f"{f}_{s}" for f in TRACKED if f != "E_background_global" for s in "AB"]
    with open(output / "comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in merged:
            writer.writerow({k: record.get(k, "") for k in columns})

    plot_note = plot(merged, output)
    label, reason = verdict(merged)
    write_summary(output, args, initial, trace, merged, label, reason, plot_note)

    final = merged[-1]
    print(f"initial state matches: {initial['combined_sha256'][:16]}...")
    print(f"camera traces identical over {trace['iterations']} iterations")
    print(f"\nfinal iteration {final['iteration']}:")
    print(f"  E_background   legacy {final['E_background_global_A']:.4e}"
          f"   carving {final['E_background_global_B']:.4e}")
    print(f"  anisotropy p99 legacy {final['anisotropy_p99_A']}"
          f"   carving {final['anisotropy_p99_B']}")
    print(f"\nverdict: {label}\n{reason}")
    print(f"\nwritten to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
