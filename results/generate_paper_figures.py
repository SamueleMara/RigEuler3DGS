#!/usr/bin/env python
"""FreeArtGS vs RigEuler3DGS: paper figures (Matplotlib only), per FreeArtGS_vs_RigEuler3DGS_Codex_Matplotlib_Spec.md.

Required figures (figures/):
  fig01_axis_direction          fig02_revolute_state_mae      fig03_prismatic_state_mae
  fig04_revolute_axis_line      fig05_motion_range            fig06_gaussian_counts
  fig07_freeart_trajectories    fig08_freeart_rendering       fig09_rigeuler_v4_v5_ablation
plus FIGURES.md (captions, plotted fields, units) and metrics_recomputed.json (Table 5).

Supplementary, RigEuler-only or validated per-frame data (figures/supplementary/), written only when the
corresponding input is given:
  figS1_rigeuler_trajectories   figS2_per_frame_abs_error     (--rigeuler-trajectories, validated against saved MAE)
  figS3_type_criterion          figS4_real_phase1_drift       figS5_real_drift_attribution   (--rigeuler-real)

    python results/generate_paper_figures.py                       # defaults read results/input/
    python results/generate_paper_figures.py --formats pdf,png,svg --dpi 400
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import ticker  # noqa: E402

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input"

ORDER = ["foldchair_102255", "laptop_10211", "storage_45135", "oven_101917", "scissor_11100",
         "washer_103776", "blade_103706", "fridge_10905", "stapler_103111", "USB_100109"]
WASHER = "washer_103776"
FA, RIG = "FreeArtGS", "RigEuler3DGS"
C_FA, C_RIG, C_V4 = "#285780", "#A33B3B", "#6E6E6E"
M_FA, M_RIG, M_V4 = "o", "s", "^"
W2, W1 = 7.15, 3.45                    # two-column / single-column width, inches
DEG = 180.0 / math.pi
REQUIRED = ["fig01_axis_direction", "fig02_revolute_state_mae", "fig03_prismatic_state_mae",
            "fig04_revolute_axis_line", "fig05_motion_range", "fig06_gaussian_counts",
            "fig07_freeart_trajectories", "fig08_freeart_rendering", "fig09_rigeuler_v4_v5_ablation"]
COMPARISON_COLUMNS = [
    "scene", "joint_type", "rig_version", "freeart_joint_type_correct", "rig_joint_type_correct",
    "freeart_axis_error_deg", "rig_axis_error_deg", "freeart_axis_line_distance_mm", "rig_axis_line_distance_mm",
    "freeart_state_mae", "rig_state_mae", "freeart_state_max_abs_error", "rig_state_max_abs_error", "state_unit",
    "gt_motion_range", "freeart_pred_motion_range", "rig_pred_motion_range", "freeart_gaussians", "rig_gaussians",
    "rig_support", "rig_support_fraction", "freeart_logged_psnr_db", "freeart_logged_ssim", "freeart_logged_lpips",
    "freeart_final_foreground_psnr_db", "rig_v4_axis_error_deg", "rig_v4_state_mae", "rig_v4_axis_line_distance_mm"]
FA_COLUMNS = ["scene", "cameras", "timestamps", "joint_type", "joint_type_correct", "axis_error_deg",
              "axis_line_error_mm", "state_mae", "state_max_error", "state_unit", "predicted_motion_range",
              "gt_motion_range", "final_foreground_psnr_db", "logged_psnr_db", "logged_ssim", "logged_lpips", "gaussians"]
RIG_COLUMNS = ["object", "run", "gt_type", "type_decision", "type_decision_correct", "gt_hyp_axis_err_deg",
               "revolute_line_dist_cm", "revolute_q_mae_deg_p3", "revolute_q_max_abs_deg_p3",
               "prismatic_q_mae_cm_p3", "prismatic_q_max_abs_cm_p3", "revolute_range_est_deg", "revolute_range_gt_deg",
               "prismatic_range_est_cm", "prismatic_range_gt_cm", "gaussians", "support", "support_fraction"]


class InputError(ValueError):
    """Schema / selection / protocol violation in the inputs."""


class MixedUnitError(ValueError):
    """An aggregate was requested over joint states in different units (degrees and millimetres)."""


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
        "axes.edgecolor": "#222222", "lines.linewidth": 1.1, "lines.markersize": 4.0,
        "axes.spines.top": False, "axes.spines.right": False})


def short(scene):
    return scene.rsplit("_", 1)[0]


# ------------------------------------------------------------------------------------------------ loading
def _need(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise InputError(f"{name}: missing columns {miss}")


def load_freeart_summary(path):
    df = pd.read_csv(path)
    _need(df, FA_COLUMNS, Path(path).name)
    if df["scene"].duplicated().any():
        raise InputError(f"{path}: duplicated scenes {df.loc[df.scene.duplicated(), 'scene'].tolist()}")
    return df.set_index("scene", drop=False)


def load_freeart_report(path):
    """{scene: DataFrame(source_frame_index, predicted, gt, unit)} in deg / mm (JSON holds rad / m)."""
    rep = json.load(open(path)); out = {}
    for s in rep["scenes"]:
        name = s["summary"]["scene"]; ev = s["saved_joint_evaluation"]
        unit = ev["joint_state_unit"]
        conv, u = {"rad": (DEG, "deg"), "m": (1000.0, "mm")}[unit]
        smp = pd.DataFrame(ev["joint_state_samples"])
        out[name] = pd.DataFrame({"source_frame_index": smp["source_frame_index"].astype(int),
                                  "predicted": conv * smp["predicted_relative_state"],
                                  "gt": conv * smp["gt_relative_state"], "unit": u})
        out[name].attrs["saved_mae_native"] = ev["joint_state_mae"]; out[name].attrs["conv"] = conv
    return out


def load_freeart_psnr(path):
    df = pd.read_csv(path)
    _need(df, ["scene", "camera", "time_index", "foreground_psnr_db"], Path(path).name)
    df["scene"] = df["scene"].str.replace("_multiview", "", regex=False)
    return df


def load_rigeuler_synthetic(path):
    df = pd.read_csv(path)
    _need(df, RIG_COLUMNS, Path(path).name)
    if df.duplicated(["object", "run"]).any():
        raise InputError(f"{path}: duplicated object/run rows")
    return df


def select_rigeuler(rig, run="primary"):
    """One row per scene. primary = v5_canonical; v5_gates_canonical only where no v5_canonical row exists,
    which must be washer_103776 alone. v4_bootstrap rows never enter the primary selection."""
    if run == "primary":
        v5 = rig[rig.run == "v5_canonical"].set_index("object", drop=False)
        gates = rig[rig.run == "v5_gates_canonical"].set_index("object", drop=False)
        rows = []
        for s in ORDER:
            if s in v5.index:
                rows.append(v5.loc[s])
            elif s in gates.index:
                if s != WASHER:
                    raise InputError(f"{s}: only a v5_gates_canonical run; the gates exception is for {WASHER} only")
                rows.append(gates.loc[s])
            else:
                raise InputError(f"{s}: no primary RigEuler run")
        sel = pd.DataFrame(rows)
    else:
        sel = rig[rig.run == run].set_index("object", drop=False)
        missing = [s for s in ORDER if s not in sel.index]
        if missing:
            raise InputError(f"run {run}: missing scenes {missing}")
        sel = sel.loc[ORDER]
    if (sel["run"] == "v4_bootstrap").any() and run == "primary":
        raise InputError("v4_bootstrap leaked into the primary selection")
    return sel.set_index("object", drop=False)


def rig_metrics(r):
    """Native RigEuler fields in common units: deg / mm (prismatic cm x 10, axis line cm x 10)."""
    rev = r["gt_type"] == "revolute"
    f = lambda c: float(r[c]) if pd.notna(r[c]) and r[c] != "" else np.nan  # noqa: E731
    return dict(
        type_correct=bool(r["type_decision_correct"]) if not isinstance(r["type_decision_correct"], str)
        else r["type_decision_correct"] == "True",
        axis_deg=f("gt_hyp_axis_err_deg"),
        line_mm=10 * f("revolute_line_dist_cm") if rev else np.nan,
        mae=f("revolute_q_mae_deg_p3") if rev else 10 * f("prismatic_q_mae_cm_p3"),
        max_abs=f("revolute_q_max_abs_deg_p3") if rev else 10 * f("prismatic_q_max_abs_cm_p3"),
        unit="deg" if rev else "mm",
        range_est=f("revolute_range_est_deg") if rev else 10 * f("prismatic_range_est_cm"),
        range_gt=f("revolute_range_gt_deg") if rev else 10 * f("prismatic_range_gt_cm"))


def joined_table(fa, rig):
    """Normalized 10-row table (freeart_rigeuler_comparison.csv schema). FreeArt values are passed through."""
    prim, v4 = select_rigeuler(rig, "primary"), select_rigeuler(rig, "v4_bootstrap")
    missing = [s for s in ORDER if s not in fa.index]
    if missing:
        raise InputError(f"FreeArtGS summary: missing scenes {missing}")
    rows = []
    for s in ORDER:
        a, r, r4 = fa.loc[s], prim.loc[s], v4.loc[s]
        m, m4 = rig_metrics(r), rig_metrics(r4)
        if a["joint_type"] != r["gt_type"]:
            raise InputError(f"{s}: FreeArt GT type {a['joint_type']} != RigEuler GT type {r['gt_type']}")
        if a["state_unit"] != m["unit"]:
            raise InputError(f"{s}: state units differ ({a['state_unit']} vs {m['unit']})")
        rev = a["joint_type"] == "revolute"
        rows.append({
            "scene": s, "joint_type": a["joint_type"], "rig_version": r["run"],
            "freeart_joint_type_correct": bool(a["joint_type_correct"]), "rig_joint_type_correct": m["type_correct"],
            "freeart_axis_error_deg": a["axis_error_deg"], "rig_axis_error_deg": m["axis_deg"],
            "freeart_axis_line_distance_mm": a["axis_line_error_mm"] if rev else np.nan, "rig_axis_line_distance_mm": m["line_mm"],
            "freeart_state_mae": a["state_mae"], "rig_state_mae": m["mae"],
            "freeart_state_max_abs_error": a["state_max_error"], "rig_state_max_abs_error": m["max_abs"],
            "state_unit": a["state_unit"], "gt_motion_range": a["gt_motion_range"],
            "freeart_pred_motion_range": a["predicted_motion_range"], "rig_pred_motion_range": m["range_est"],
            "freeart_gaussians": int(a["gaussians"]), "rig_gaussians": int(r["gaussians"]),
            "rig_support": int(r["support"]), "rig_support_fraction": float(r["support_fraction"]),
            "freeart_logged_psnr_db": a["logged_psnr_db"], "freeart_logged_ssim": a["logged_ssim"],
            "freeart_logged_lpips": a["logged_lpips"], "freeart_final_foreground_psnr_db": a["final_foreground_psnr_db"],
            "rig_v4_axis_error_deg": m4["axis_deg"], "rig_v4_state_mae": m4["mae"], "rig_v4_axis_line_distance_mm": m4["line_mm"],
            "_rig_range_gt": m["range_gt"], "_rig_v4_max_abs": m4["max_abs"], "_rig_v4_type_correct": m4["type_correct"]})
    j = pd.DataFrame(rows)
    if len(j) != 10 or (j.joint_type == "revolute").sum() != 8 or (j.joint_type == "prismatic").sum() != 2:
        raise InputError("expected 10 matched scenes, 8 revolute and 2 prismatic")
    bad = j[~np.isclose(j["_rig_range_gt"], j["gt_motion_range"], rtol=0, atol=2e-3)]
    if len(bad):
        raise InputError(f"GT motion ranges disagree between the reports: {bad[['scene', 'gt_motion_range', '_rig_range_gt']].values.tolist()}")
    return j.set_index("scene", drop=False)[COMPARISON_COLUMNS + ["_rig_range_gt", "_rig_v4_max_abs", "_rig_v4_type_correct"]]


def public(j):
    return j[COMPARISON_COLUMNS].reset_index(drop=True)


def validate_against_comparison(j, path, rtol=1e-6):
    """The supplied normalized CSV must equal the table rebuilt from the raw inputs (to its serialization)."""
    ref = pd.read_csv(path)
    _need(ref, COMPARISON_COLUMNS, Path(path).name)
    ref = ref.set_index("scene")
    problems = []
    for s in ORDER:
        for c in COMPARISON_COLUMNS[1:]:
            a, b = j.loc[s, c], ref.loc[s, c]
            if isinstance(a, (bool, np.bool_)) or c in ("joint_type", "rig_version", "state_unit"):
                ok = str(a) == str(b)
            elif pd.isna(a) or pd.isna(b):
                ok = pd.isna(a) and pd.isna(b)
            else:
                ok = math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=1e-9)
            if not ok:
                problems.append(f"{s}.{c}: rebuilt {a!r} vs supplied {b!r}")
    if problems:
        raise InputError("comparison CSV disagrees with the raw inputs:\n  " + "\n  ".join(problems))


# ------------------------------------------------------------------------------------------------ aggregates
def mean_state_mae(values, units):
    """Equal-weight mean of joint-state errors that share ONE unit; mixing deg and mm raises."""
    u = set(units)
    if len(u) != 1:
        raise MixedUnitError(f"refusing to average joint-state errors in different units: {sorted(u)}")
    return float(np.mean(values)), u.pop()


def aggregates(j):
    rev, pri = j[j.joint_type == "revolute"], j[j.joint_type == "prismatic"]
    nw = rev[rev.scene != WASHER]
    out = {"n_scenes": int(len(j)), "n_revolute": int(len(rev)), "n_prismatic": int(len(pri)),
           "rigeuler_runs": {s: j.loc[s, "rig_version"] for s in ORDER}, "methods": {}}
    for key, pre in ((FA, "freeart_"), (RIG, "rig_")):
        m = {
            "correct_joint_types": f"{int(j[pre + 'joint_type_correct'].astype(bool).sum())}/10",
            "mean_axis_direction_error_deg": float(j[pre + "axis_error_deg"].mean()),
            "median_axis_direction_error_deg": float(j[pre + "axis_error_deg"].median()),
            "mean_revolute_state_mae_deg": mean_state_mae(rev[pre + "state_mae"], rev.state_unit)[0],
            "median_revolute_state_mae_deg": float(rev[pre + "state_mae"].median()),
            "mean_revolute_state_mae_excl_washer_deg": mean_state_mae(nw[pre + "state_mae"], nw.state_unit)[0],
            "mean_prismatic_state_mae_mm": mean_state_mae(pri[pre + "state_mae"], pri.state_unit)[0],
            "mean_revolute_axis_line_distance_mm": float(rev[pre + "axis_line_distance_mm"].mean()),
            "median_revolute_axis_line_distance_mm": float(rev[pre + "axis_line_distance_mm"].median()),
            "mean_gaussian_count": float(j[pre + "gaussians"].mean())}
        out["methods"][key] = m
    out["n_contributing"] = {"axis": 10, "revolute_mae": 8, "revolute_mae_excl_washer": 7, "prismatic_mae": 2,
                             "revolute_axis_line": 8, "gaussians": 10}
    out["aggregation"] = "arithmetic mean / median over scenes, equal weight; no unit mixing"
    return out


# ------------------------------------------------------------------------------------------------ plotting helpers
FIGS = {}          # stem -> caption metadata, filled by every figure function


def save(fig, outdir, stem, formats, dpi, meta):
    outdir.mkdir(parents=True, exist_ok=True)
    for f in formats:
        fig.savefig(outdir / f"{stem}.{f}", bbox_inches="tight", dpi=dpi if f == "png" else None)
    plt.close(fig)
    FIGS[stem] = dict(meta, dir=outdir.name)


def grid(ax, axis="x"):
    ax.grid(True, axis=axis, which="major", color="#d4d4d4", linewidth=0.4, zorder=0)
    ax.grid(True, axis=axis, which="minor", color="#ececec", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)


def plain_log(ax, axis="x"):
    fmt = ticker.FuncFormatter(lambda v, _: (f"{v:g}" if v >= 1e-3 else f"{v:.0e}"))
    loc = ticker.LogLocator(base=10, subs=(1.0,))
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(loc); a.set_major_formatter(fmt)
    a.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    a.set_minor_formatter(ticker.NullFormatter())


def log_limits(vals, pad=2.0):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if (v <= 0).any():
        raise ValueError("log axis requested for non-positive data")
    return 10 ** math.floor(math.log10(v.min() / pad * 1.0001)), 10 ** math.ceil(math.log10(v.max() * pad / 1.0001))


def paired(ax, scenes, series, log=False, xlabel="", zero=True, connect=True):
    """Horizontal paired dot plot. series = [(label, values, color, marker)], offsets keep markers apart."""
    n = len(scenes); k = len(series); offs = np.linspace(-0.14, 0.14, k) if k > 1 else [0.0]
    y = np.arange(n)
    for i in range(n) if connect and k > 1 else []:
        xs = [s[1][i] for s in series]
        if all(np.isfinite(xs)):
            ax.plot(xs, y[i] + np.asarray(offs), color="#9a9a9a", linewidth=0.5, zorder=1)
    for (lab, vals, col, mk), o in zip(series, offs):
        ax.plot(vals, y + o, linestyle="none", marker=mk, color=col, markersize=4.0, label=lab, zorder=3,
                markeredgewidth=0.6, markeredgecolor=col)
    ax.set_yticks(y); ax.set_yticklabels([short(s) for s in scenes]); ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel(xlabel)
    if log:
        ax.set_xscale("log"); ax.set_xlim(*log_limits(np.concatenate([np.asarray(s[1], float) for s in series])))
        plain_log(ax)
    elif zero:
        hi = np.nanmax(np.concatenate([np.asarray(s[1], float) for s in series]))
        ax.set_xlim(0, hi * 1.12)
    grid(ax, "x")
    ax.tick_params(axis="y", length=0)


def legend_top(ax_or_fig, handles=None, labels=None, ncol=2, y=1.0):
    kw = dict(loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, handletextpad=0.4,
              columnspacing=1.2, borderaxespad=0.2)
    if handles is None:
        ax_or_fig.legend(**kw)
    else:
        ax_or_fig.legend(handles, labels, **kw)


def two(j, fa_col, rig_col, scenes):
    return [(FA, j.loc[scenes, fa_col].astype(float).values, C_FA, M_FA),
            (RIG, j.loc[scenes, rig_col].astype(float).values, C_RIG, M_RIG)]


# ------------------------------------------------------------------------------------------------ figures
def fig01(j, out, fmt, dpi):
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    paired(ax, ORDER, two(j, "freeart_axis_error_deg", "rig_axis_error_deg", ORDER), xlabel="Axis-direction error (°)")
    legend_top(ax)
    save(fig, out, "fig01_axis_direction", fmt, dpi, dict(
        caption="Axis-direction error per synthetic scene (angle between estimated and GT joint axis, undirected). "
                "Linear axis from zero; markers offset vertically.",
        fields="freeart_axis_error_deg (FreeArtGS_summary.axis_error_deg); rig_axis_error_deg "
               "(RigEuler3DGS_synthetic_results.gt_hyp_axis_err_deg)", units="degrees", n="10 scenes"))


def fig02(j, out, fmt, dpi):
    sc = [s for s in ORDER if j.loc[s, "joint_type"] == "revolute"]
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    paired(ax, sc, two(j, "freeart_state_mae", "rig_state_mae", sc), log=True, xlabel="Revolute joint-state MAE (°, log scale)")
    legend_top(ax)
    save(fig, out, "fig02_revolute_state_mae", fmt, dpi, dict(
        caption="Revolute joint-state mean absolute error over each 60-frame sequence, in each method's native "
                "evaluation alignment (FreeArtGS: relative to the first timestamp; RigEuler3DGS: relative to its "
                "canonical frame, axis sign aligned, no scale fit). Logarithmic axis; no value clipped "
                f"(FreeArtGS {WASHER}: {j.loc[WASHER, 'freeart_state_mae']:.4f}°).",
        fields="freeart_state_mae (state_mae, deg); rig_state_mae (revolute_q_mae_deg_p3)", units="degrees", n="8 revolute scenes"))


def fig03(j, out, fmt, dpi):
    sc = [s for s in ORDER if j.loc[s, "joint_type"] == "prismatic"]
    fig, ax = plt.subplots(figsize=(W1, 1.35))
    paired(ax, sc, two(j, "freeart_state_mae", "rig_state_mae", sc), xlabel="Prismatic joint-state MAE (mm)")
    legend_top(ax)
    save(fig, out, "fig03_prismatic_state_mae", fmt, dpi, dict(
        caption="Prismatic joint-state mean absolute error, linear axis from zero. RigEuler3DGS values converted "
                "from centimetres (prismatic_q_mae_cm_p3 × 10).",
        fields="freeart_state_mae (state_mae, mm); rig_state_mae (prismatic_q_mae_cm_p3 × 10)", units="millimetres",
        n="2 prismatic scenes (storage_45135, blade_103706)"))


def fig04(j, out, fmt, dpi):
    sc = [s for s in ORDER if j.loc[s, "joint_type"] == "revolute"]
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    paired(ax, sc, two(j, "freeart_axis_line_distance_mm", "rig_axis_line_distance_mm", sc), log=True,
           xlabel="Axis-line distance (mm, log scale)")
    legend_top(ax)
    save(fig, out, "fig04_revolute_axis_line", fmt, dpi, dict(
        caption="Shortest distance between the estimated and GT infinite revolute axis lines (not a pivot-point "
                "error). Prismatic joints have no axis-line position and are excluded. Logarithmic axis; no value "
                "clipped.", fields="freeart_axis_line_distance_mm (axis_line_error_mm); rig_axis_line_distance_mm "
                                   "(revolute_line_dist_cm × 10)", units="millimetres", n="8 revolute scenes"))


def fig05(j, out, fmt, dpi):
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(wspace=0.32))
    for ax, jt, unit, tag in ((axs[0], "revolute", "°", "(a) Revolute"), (axs[1], "prismatic", "mm", "(b) Prismatic")):
        sc = [s for s in ORDER if j.loc[s, "joint_type"] == jt]
        gt = j.loc[sc, "gt_motion_range"].astype(float)
        vals = np.concatenate([gt, j.loc[sc, "freeart_pred_motion_range"], j.loc[sc, "rig_pred_motion_range"]]).astype(float)
        lo, hi = vals.min(), vals.max(); pad = 0.08 * (hi - lo)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, linestyle=":", color="#8c8c8c", linewidth=0.8, zorder=1, label="$y = x$")
        ax.plot(gt, j.loc[sc, "freeart_pred_motion_range"], linestyle="none", marker=M_FA, color=C_FA, label=FA, zorder=3)
        ax.plot(gt, j.loc[sc, "rig_pred_motion_range"], linestyle="none", marker=M_RIG, color=C_RIG, label=RIG, zorder=3,
                fillstyle="none", markeredgewidth=0.9)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"GT motion range ({unit})"); ax.set_ylabel(f"Estimated motion range ({unit})")
        ax.set_title(tag, loc="left"); grid(ax, "both")
        if jt == "revolute":
            x, yv = j.loc[WASHER, "gt_motion_range"], j.loc[WASHER, "freeart_pred_motion_range"]
            ax.annotate(f"{WASHER}\n{FA}: {yv:.1f}° (GT {x:.1f}°)", xy=(x, yv), xytext=(x + 6, yv - 14), fontsize=6.5,
                        arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.5))
        else:
            lo_s = min(sc, key=lambda s: j.loc[s, "gt_motion_range"])
            for s in sc:     # lowest point: label below-right, otherwise above-left, inside the panel
                off, ha = ((7, -9), "left") if s == lo_s else ((-6, 7), "right")
                ax.annotate(short(s), xy=(j.loc[s, "gt_motion_range"], j.loc[s, "gt_motion_range"]),
                            xytext=off, textcoords="offset points", fontsize=6.5, ha=ha)
    h, l = axs[0].get_legend_handles_labels()
    legend_top(fig, h, l, ncol=3, y=0.93)
    save(fig, out, "fig05_motion_range", fmt, dpi, dict(
        caption="Estimated versus GT motion range per scene; dotted line y = x. (a) revolute, degrees; (b) prismatic, "
                "millimetres (RigEuler3DGS converted from cm). Motion range is not an error measure.",
        fields="gt_motion_range (FreeArtGS gt_motion_range); freeart_pred_motion_range (predicted_motion_range); "
               "rig_pred_motion_range (revolute_range_est_deg | prismatic_range_est_cm × 10)",
        units="degrees (a), millimetres (b)", n="8 revolute + 2 prismatic scenes"))


def fig06(j, out, fmt, dpi):
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    paired(ax, ORDER, [(FA, j.loc[ORDER, "freeart_gaussians"].values / 1e3, C_FA, M_FA),
                       (RIG, j.loc[ORDER, "rig_gaussians"].values / 1e3, C_RIG, M_RIG)],
           xlabel="Gaussian count (thousands; counts only)")
    legend_top(ax)
    save(fig, out, "fig06_gaussian_counts", fmt, dpi, dict(
        caption="Number of Gaussians in each method's final model. Counts come from different training methods and "
                "stages; they are descriptive and do not measure efficiency, speed, completeness or memory.",
        fields="freeart_gaussians (FreeArtGS_summary.gaussians); rig_gaussians (RigEuler3DGS_synthetic_results.gaussians)",
        units="thousands of Gaussians", n="10 scenes"))


def small_multiples(curves, out, stem, fmt, dpi, meta, est_color, est_label, gt_label="GT"):
    """curves: {scene: (x, gt, est, unit)} in ORDER, 2 x 5 panels."""
    fig, axs = plt.subplots(2, 5, figsize=(W2, 3.5), gridspec_kw=dict(wspace=0.48, hspace=0.62))
    for ax, s in zip(axs.flat, ORDER):
        x, g, e, unit = curves[s]
        ax.plot(x, g, color="black", linewidth=1.0, label=gt_label, zorder=2)
        ax.plot(x, e, color=est_color, linewidth=1.1, linestyle=(0, (3.5, 1.8)), label=est_label, zorder=3)
        ax.set_title(short(s), fontsize=7.5, pad=3)
        ax.set_ylabel(f"q ({'°' if unit == 'deg' else 'mm'})", labelpad=1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(3, integer=True)); ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        grid(ax, "both")
    for ax in axs[1]:
        ax.set_xlabel("Source frame index")
    h, l = axs.flat[0].get_legend_handles_labels()
    legend_top(fig, h, l, ncol=2, y=0.94)
    save(fig, out, stem, fmt, dpi, meta)


def fig07(rep, out, fmt, dpi):
    curves = {s: (rep[s].source_frame_index.values, rep[s]["gt"].values, rep[s]["predicted"].values, rep[s].unit.iloc[0]) for s in ORDER}
    small_multiples(curves, out, "fig07_freeart_trajectories", fmt, dpi, dict(
        caption="FreeArtGS joint-state trajectories: GT (black, solid) and FreeArtGS estimate (dashed), relative to the "
                "first timestamp, 60 samples per scene. Single-method figure.",
        fields="FreeArtGS_report.json scenes[].saved_joint_evaluation.joint_state_samples: source_frame_index, "
               "gt_relative_state, predicted_relative_state (rad × 180/π, m × 1000)", units="degrees or millimetres per panel",
        n="10 scenes × 60 samples"), C_FA, f"{FA} estimate", "GT")


def fig08(j, psnr, out, fmt, dpi):
    fig, axs = plt.subplots(1, 4, figsize=(W2, 2.7), sharey=True, gridspec_kw=dict(wspace=0.22))
    y = np.arange(len(ORDER))
    panels = [("freeart_logged_psnr_db", "Logged PSNR (dB)"), ("freeart_logged_ssim", "Logged SSIM"),
              ("freeart_logged_lpips", "Logged LPIPS"), ("freeart_final_foreground_psnr_db", "Final FG PSNR (dB)")]
    for ax, (col, lab) in zip(axs, panels):
        v = j.loc[ORDER, col].astype(float).values
        ax.plot(v, y, linestyle="none", marker=M_FA, color=C_FA, zorder=3)
        ok = np.isfinite(v); span = v[ok].max() - v[ok].min()
        ax.set_xlim(v[ok].min() - 0.12 * span, v[ok].max() + 0.12 * span)
        for i in np.where(~ok)[0]:
            ax.text(np.mean(ax.get_xlim()), y[i], "N/A", ha="center", va="center", fontsize=6.5, color="#555555")
        ax.set_xlabel(lab); grid(ax, "x"); ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(y); axs[0].set_yticklabels([short(s) for s in ORDER]); axs[0].set_ylim(len(ORDER) - 0.5, -0.5)
    axs[3].set_title("7 scenes", fontsize=7, loc="right", pad=2)
    n_obs = int(len(psnr))
    save(fig, out, "fig08_freeart_rendering", fmt, dpi, dict(
        caption="FreeArtGS rendering metrics only (no RigEuler3DGS rendering metrics exist in the inputs). Logged "
                "PSNR / SSIM / LPIPS: latest all-evaluation-image TensorBoard record (step 29000; SSIM and LPIPS include "
                "background). Final FG PSNR: independent recomputation from rendered PNGs against RGB and exact "
                f"foreground masks, available for 7 scenes ({n_obs} camera/frame observations); N/A elsewhere. The 48 "
                "evaluation views are also training views, so these are not held-out novel-view scores.",
        fields="freeart_logged_psnr_db, freeart_logged_ssim, freeart_logged_lpips, freeart_final_foreground_psnr_db "
               "(FreeArtGS_summary: logged_psnr_db, logged_ssim, logged_lpips, final_foreground_psnr_db)",
        units="dB, unitless, unitless, dB", n="10 scenes (final FG PSNR: 7)"))


def fig09(j, out, fmt, dpi):
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.75), gridspec_kw=dict(wspace=0.55, width_ratios=[1.0, 1.0, 1.0]))
    lab5, lab4 = "RigEuler3DGS v5 (primary)", "RigEuler3DGS v4_bootstrap"
    paired(axs[0], ORDER, [(lab5, j.loc[ORDER, "rig_axis_error_deg"].values, C_RIG, M_RIG),
                           (lab4, j.loc[ORDER, "rig_v4_axis_error_deg"].values, C_V4, M_V4)],
           xlabel="Axis-direction error (°)")
    rev = [s for s in ORDER if j.loc[s, "joint_type"] == "revolute"]
    pri = [s for s in ORDER if j.loc[s, "joint_type"] == "prismatic"]
    paired(axs[1], rev, [(lab5, j.loc[rev, "rig_state_mae"].values, C_RIG, M_RIG),
                         (lab4, j.loc[rev, "rig_v4_state_mae"].values, C_V4, M_V4)], log=True,
           xlabel="Revolute state MAE (°, log)")
    paired(axs[2], pri, [(lab5, j.loc[pri, "rig_state_mae"].values, C_RIG, M_RIG),
                         (lab4, j.loc[pri, "rig_v4_state_mae"].values, C_V4, M_V4)], xlabel="Prismatic state MAE (mm)")
    axs[2].set_ylim(1.6, -0.6)
    for ax, t in zip(axs, ("(a)", "(b)", "(c)")):
        ax.set_title(t, loc="left")
    h, l = axs[0].get_legend_handles_labels()
    legend_top(fig, h, l, ncol=2, y=0.97)
    save(fig, out, "fig09_rigeuler_v4_v5_ablation", fmt, dpi, dict(
        caption="Supplementary RigEuler3DGS version comparison (not part of the FreeArtGS comparison). Primary = "
                f"v5_canonical, except {WASHER} = v5_gates_canonical. (a) axis-direction error, (b) revolute state MAE, "
                f"log axis, v4 {WASHER} {j.loc[WASHER, 'rig_v4_state_mae']:.4f}° shown unclipped, (c) prismatic state MAE.",
        fields="rig_axis_error_deg / rig_v4_axis_error_deg (gt_hyp_axis_err_deg); rig_state_mae / rig_v4_state_mae "
               "(revolute_q_mae_deg_p3 | prismatic_q_mae_cm_p3 × 10)", units="degrees; degrees; millimetres",
        n="10 scenes per version"))


# ------------------------------------------------------------------------------------------------ supplementary
def rigeuler_curves(traj, j):
    """Validated per-frame RigEuler trajectories {scene: frame-sorted DataFrame}; raises if the MAE disagrees."""
    out = {}
    for s in ORDER:
        t = traj[traj.scene == s].sort_values("source_frame_index")
        if len(t) == 0:
            raise InputError(f"RigEuler trajectories: no rows for {s}")
        if t.run.iloc[0] != j.loc[s, "rig_version"]:
            raise InputError(f"{s}: trajectory run {t.run.iloc[0]} != primary run {j.loc[s, 'rig_version']}")
        mae = float(np.abs(t.q_est - t.q_gt).mean())
        if not math.isclose(mae, float(j.loc[s, "rig_state_mae"]), rel_tol=2e-3, abs_tol=2e-4):
            raise InputError(f"{s}: trajectory MAE {mae} != reported {j.loc[s, 'rig_state_mae']}")
        out[s] = t
    return out


def figS1_S2(j, rep, rig_tr, out, fmt, dpi):
    curves = {s: (t.source_frame_index.values, t.q_gt.values, t.q_est.values, t.unit.iloc[0]) for s, t in rig_tr.items()}
    small_multiples(curves, out, "figS1_rigeuler_trajectories", fmt, dpi, dict(
        caption="RigEuler3DGS Phase-3 joint-state trajectories of the GT-type hypothesis: GT (black, solid) and estimate "
                "(dashed), relative to the canonical frame (frame 0 for forward scenes, frame 59 for backward scenes), "
                "axis sign aligned, no scale fit. Primary runs. The per-frame mean reproduces the reported MAE.",
        fields="RigEuler3DGS_synthetic_trajectories.csv: source_frame_index, q_gt, q_est (from */*_joint/<type>/"
               "trajectory.csv and metadata/frame_values.csv)", units="degrees or millimetres per panel",
        n="10 scenes × 60 frames"), C_RIG, f"{RIG} estimate", "GT")
    fig, axs = plt.subplots(2, 5, figsize=(W2, 3.5), gridspec_kw=dict(wspace=0.5, hspace=0.62))
    for ax, s in zip(axs.flat, ORDER):
        f = rep[s]; t = rig_tr[s]
        ax.plot(f.source_frame_index, np.abs(f["predicted"] - f["gt"]), color=C_FA, linewidth=1.0, label=FA)
        ax.plot(t.source_frame_index, t.abs_error, color=C_RIG, linewidth=1.0, linestyle=(0, (3.5, 1.8)), label=RIG)
        ax.set_title(short(s), fontsize=7.5, pad=3); ax.set_ylim(bottom=0)
        ax.set_ylabel(f"|error| ({'°' if f.unit.iloc[0] == 'deg' else 'mm'})", labelpad=1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(3, integer=True)); ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        grid(ax, "both")
    for ax in axs[1]:
        ax.set_xlabel("Source frame index")
    h, l = axs.flat[0].get_legend_handles_labels()
    legend_top(fig, h, l, ncol=2, y=0.94)
    save(fig, out, "figS2_per_frame_abs_error", fmt, dpi, dict(
        caption="Per-frame absolute joint-state error of each method in its native evaluation alignment (FreeArtGS: "
                "relative to the first timestamp; RigEuler3DGS: relative to its canonical frame). The reference frames "
                "differ for backward scenes; no re-alignment was applied. Linear axis from zero per panel.",
        fields="FreeArtGS joint_state_samples |predicted - gt|; RigEuler3DGS_synthetic_trajectories.abs_error",
        units="degrees or millimetres per panel", n="10 scenes × 60 frames per method"))


ARMS = [("C", "#1f1f1f", "o"), ("majority", "#6a6a6a", "s"), ("unanimous", "#a8a8a8", "^")]


def figS3_S5(rig_syn, real, out, fmt, dpi, rl_threshold):
    scenes = list(dict.fromkeys(real.scene))
    exp = {s: real[real.scene == s].expected_type.iloc[0] for s in scenes}
    prim = select_rigeuler(rig_syn, "primary")
    # S3: the geometric type criterion
    fig, axs = plt.subplots(1, 3, figsize=(W2, 3.0), gridspec_kw=dict(wspace=0.65))
    ax = axs[0]; y = np.arange(len(ORDER))
    v = prim.loc[ORDER, "rev_axis_radius_over_part_scale"].astype(float).values
    ax.plot(v, y, linestyle="none", marker=M_RIG, color=C_RIG, label="v5 primary")
    ax.set_yticks(y); ax.set_yticklabels([f"{short(s)} ({prim.loc[s, 'gt_type'][0].upper()})" for s in ORDER])
    ax.set_ylim(len(ORDER) - 0.5, -0.5); ax.set_title("(a) Synthetic, GT type", loc="left")
    allv = list(v)
    for ax, ver, tag in ((axs[1], "v6", "(b) Real v6"), (axs[2], "v5", "(c) Real v5")):
        y = np.arange(len(scenes))
        for k, (arm, col, mk) in enumerate(ARMS):
            vv = np.array([float(real[(real.scene == s) & (real.arm == arm)][f"{ver}_rev_axis_radius_over_part_scale"].iloc[0]) for s in scenes])
            ax.plot(vv, y + (k - 1) * 0.16, linestyle="none", marker=mk, color=col, label=arm, markersize=3.8)
            allv += list(vv)
        ax.set_yticks(y); ax.set_yticklabels([f"{s} ({exp[s][0].upper()})" for s in scenes])
        ax.set_ylim(len(scenes) - 0.5, -0.5); ax.set_title(tag + ", visual type", loc="left")
    lo, hi = log_limits(allv)
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlim(lo, hi); plain_log(ax); grid(ax, "x"); ax.tick_params(axis="y", length=0)
        ax.axvline(rl_threshold, color="#444444", linestyle="--", linewidth=0.7, zorder=1)
        ax.set_xlabel("Revolute axis radius / part scale")
    h, l = axs[1].get_legend_handles_labels()
    legend_top(fig, h, l, ncol=3, y=0.96)
    agree = {ver: {arm: int(real[(real.arm == arm)][f"{ver}_type_decision_matches_expected"].astype(str).eq("True").sum())
                   for arm, _, _ in ARMS} for ver in ("v6", "v5")}
    save(fig, out, "figS3_type_criterion", fmt, dpi, dict(
        caption=f"Geometric joint-type criterion: distance of the fitted revolute axis from the support centroid divided "
                f"by the part scale (log axis); dashed line at {rl_threshold:g}, above which the revolute fit is "
                "degenerate and the joint is reported prismatic. Labels give the GT type (a) or the visually expected "
                f"type (b, c; R/P). Agreement with the visual expectation (not GT accuracy), out of {len(scenes)}: "
                + "; ".join(f"{ver} " + ", ".join(f"{a} {n}" for a, n in agree[ver].items()) for ver in agree) + ".",
        fields="rev_axis_radius_over_part_scale (synthetic); v6_/v5_rev_axis_radius_over_part_scale, expected_type "
               "(real)", units="dimensionless", n=f"10 synthetic; {len(scenes)} real scenes × 3 arms × 2 versions"))
    # S4: Phase-1 drift v5 vs v6 on the same Gaussians
    rows = [(s, a) for s in scenes for a, _, _ in ARMS]
    lab = [f"{s} · {a}" for s, a in rows]
    get = lambda c: np.array([float(real[(real.scene == s) & (real.arm == a)][c].iloc[0]) for s, a in rows])  # noqa: E731
    v5, v6 = get("v5_p1_confirmed_static_drift_end_cm"), get("v6_p1_confirmed_static_drift_end_cm")
    ok = np.isfinite(v5) & np.isfinite(v6)
    fig, ax = plt.subplots(figsize=(W1, 4.6))
    sc_ok = [lab[i] for i in np.where(ok)[0]]
    paired(ax, sc_ok, [("v5", v5[ok], C_V4, M_V4), ("v6", v6[ok], C_RIG, M_RIG)], log=True,
           xlabel="Static drift at last frame (cm, log)")
    ax.set_yticklabels(sc_ok, fontsize=6.3)
    legend_top(ax)
    save(fig, out, "figS4_real_phase1_drift", fmt, dpi, dict(
        caption="Real scenes, RigEuler3DGS only: median Phase-1 displacement at the last frame of Gaussians seen at "
                "their canonical place at least 20 times and never refuted by depth (same Gaussians in v5 and v6, "
                f"which share the endpoint). Log axis. Rows without such Gaussians omitted ({int((~ok).sum())}).",
        fields="v5_p1_confirmed_static_drift_end_cm, v6_p1_confirmed_static_drift_end_cm", units="centimetres",
        n=f"{int(ok.sum())} scene × arm rows"))
    # S5: drift attribution (v6)
    groups = [("near_part", "within 2 cm of the moving support", "#2b2b2b"), ("unseen", "rarely in mask / depth-consistent", "#7a7a7a"),
              ("seen_away", "seen, away from the part", "#c4c4c4")]
    okr = np.isfinite(get("drift_all_median_cm"))
    lab2 = [lab[i] for i in np.where(okr)[0]]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 4.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    ser = [(g[1], get(f"drift_{g[0]}_median_cm")[okr], g[2], m) for g, m in zip(groups, ("o", "s", "^"))]
    paired(axs[0], lab2, ser, log=True, xlabel="Median static drift at last frame (cm, log)", connect=False)
    axs[0].set_yticklabels(lab2, fontsize=6.3); axs[0].set_title("(a)", loc="left")
    left = np.zeros(int(okr.sum()))
    yy = np.arange(len(lab2))
    for g in groups:
        w = get(f"drift_{g[0]}_share_pct")[okr]
        axs[1].barh(yy, w, left=left, height=0.62, color=g[2], edgecolor="white", linewidth=0.3, label=g[1])
        left += w
    axs[1].set_xlim(0, 100); axs[1].set_xlabel("Share of summed static drift (%)"); grid(axs[1], "x")
    axs[1].set_title("(b)", loc="left"); axs[1].tick_params(axis="y", length=0)
    h, l = axs[0].get_legend_handles_labels()
    legend_top(fig, h, l, ncol=3, y=0.93)
    save(fig, out, "figS5_real_drift_attribution", fmt, dpi, dict(
        caption="Real scenes, RigEuler3DGS v6 only: Phase-1 drift of the static set (outside both supports, never "
                "refuted by depth) split into Gaussians within 2 cm of the moving support, Gaussians in the object mask "
                "and depth-consistent in fewer than half of 12 sampled frames, and the rest. (a) group medians, log "
                "axis; (b) each group's share of the summed displacement.",
        fields="drift_<group>_median_cm, drift_<group>_share_pct", units="centimetres; percent",
        n=f"{int(okr.sum())} scene × arm rows"))
    return agree


# ------------------------------------------------------------------------------------------------ outputs
def write_figures_md(outdir, j, extra_note=""):
    lines = ["# Figures", "",
             "Generated by `generate_paper_figures.py` (Matplotlib, Agg). Synthetic evaluation: 10 scenes, 8 cameras, "
             "60 timestamps, local calibrated multiview setting (not the official FreeArt-21 monocular benchmark). "
             "RigEuler3DGS primary run: `v5_canonical`, except `washer_103776` = `v5_gates_canonical`; `v4_bootstrap` "
             "appears only in fig09. State errors are reported in each method's native evaluation alignment; degrees "
             "and millimetres are never combined.", "",
             "Scene IDs (tick labels drop the numeric suffix): " + ", ".join(f"`{s}`" for s in ORDER) + ".", ""]
    if extra_note:
        lines += [extra_note, ""]
    for stem, m in FIGS.items():
        path = stem if m["dir"] == outdir.name else f"{m['dir']}/{stem}"
        lines += [f"## {stem}", "", f"Files: `{path}.*`", "", m["caption"], "",
                  f"- Plotted fields: {m['fields']}", f"- Units: {m['units']}", f"- N: {m['n']}", ""]
    (outdir / "FIGURES.md").write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeart-summary", default=str(INPUT / "FreeArtGS_summary.csv"))
    ap.add_argument("--freeart-report", default=str(INPUT / "FreeArtGS_report.json"))
    ap.add_argument("--freeart-psnr", default=str(INPUT / "FreeArtGS_per_observation_psnr.csv"))
    ap.add_argument("--rigeuler-synthetic", default=str(INPUT / "RigEuler3DGS_synthetic_results.csv"))
    ap.add_argument("--rigeuler-trajectories", default=str(INPUT / "RigEuler3DGS_synthetic_trajectories.csv"),
                    help="per-frame RigEuler Phase-3 q vs GT (supplementary figS1/figS2); '' to skip")
    ap.add_argument("--rigeuler-real", default=str(INPUT / "RigEuler3DGS_real_results.csv"),
                    help="real-scene RigEuler table (supplementary figS3-S5); '' to skip")
    ap.add_argument("--comparison-csv", default=str(HERE / "freeart_rigeuler_comparison.csv"),
                    help="supplied normalized table, validated against the raw inputs; '' to skip")
    ap.add_argument("--rigeuler-run", default="primary", choices=["primary"],
                    help="RigEuler selection for the comparison figures (v4_bootstrap is shown only in fig09)")
    ap.add_argument("--rl-threshold", type=float, default=10.0)
    ap.add_argument("--outdir", default=str(HERE / "figures"))
    ap.add_argument("--formats", default="pdf,png")
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args(argv)
    fmt = [f.strip() for f in a.formats.split(",") if f.strip()]
    if not {"pdf", "png"} <= set(fmt):
        raise SystemExit("--formats must include pdf and png")
    out = Path(a.outdir); sup = out / "supplementary"
    FIGS.clear(); style()

    fa = load_freeart_summary(a.freeart_summary)
    if not ((fa.cameras == 8).all() and (fa.timestamps == 60).all()):
        raise InputError("FreeArtGS: every scene must have 8 cameras and 60 timestamps")
    rep = load_freeart_report(a.freeart_report)
    psnr = load_freeart_psnr(a.freeart_psnr)
    rig = load_rigeuler_synthetic(a.rigeuler_synthetic)
    j = joined_table(fa, rig)
    if a.comparison_csv and Path(a.comparison_csv).exists():
        validate_against_comparison(j, a.comparison_csv)
        print(f"validated {a.comparison_csv} against the raw inputs")
    for s in ORDER:
        if len(rep[s]) != 60:
            raise InputError(f"FreeArtGS report: {s} has {len(rep[s])} samples, expected 60")

    fig01(j, out, fmt, a.dpi); fig02(j, out, fmt, a.dpi); fig03(j, out, fmt, a.dpi); fig04(j, out, fmt, a.dpi)
    fig05(j, out, fmt, a.dpi); fig06(j, out, fmt, a.dpi); fig07(rep, out, fmt, a.dpi); fig08(j, psnr, out, fmt, a.dpi)
    fig09(j, out, fmt, a.dpi)

    note = ""
    if a.rigeuler_trajectories and Path(a.rigeuler_trajectories).exists():
        rig_tr = rigeuler_curves(pd.read_csv(a.rigeuler_trajectories), j)
        figS1_S2(j, rep, rig_tr, sup, fmt, a.dpi)
    else:
        print("SKIPPED: RigEuler per-frame trajectories not supplied")
        note = "SKIPPED: RigEuler per-frame trajectories not supplied."
    agree = None
    if a.rigeuler_real and Path(a.rigeuler_real).exists():
        agree = figS3_S5(rig, pd.read_csv(a.rigeuler_real), sup, fmt, a.dpi, a.rl_threshold)

    made = [p for p in out.glob("fig*.png")]
    if sorted(p.stem for p in made) != sorted(REQUIRED):
        raise RuntimeError(f"required figures mismatch: {sorted(p.stem for p in made)}")
    metrics = aggregates(j)
    if agree is not None:
        metrics["real_rigeuler_only"] = {"type_agreement_with_visual_expectation_out_of_8": agree,
                                         "rl_threshold": a.rl_threshold,
                                         "note": "visual expectation, not ground truth; no numerical real-scene joint errors"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics_recomputed.json").write_text(json.dumps(metrics, indent=2))
    write_figures_md(out, j, note)
    print(f"wrote {len(FIGS)} figures to {out} ({', '.join(fmt)}), FIGURES.md, metrics_recomputed.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
