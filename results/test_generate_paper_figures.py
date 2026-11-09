#!/usr/bin/env python
"""Validation tests for generate_paper_figures.py (spec section 9, tests 1-13, plus the supplementary inputs).

    python results/test_generate_paper_figures.py        # plain runner (pytest not required)
    python -m pytest results/test_generate_paper_figures.py
Needs results/input/ (python results/build_inputs.py).
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_paper_figures as G  # noqa: E402

INP = HERE / "input"


def data():
    fa = G.load_freeart_summary(INP / "FreeArtGS_summary.csv")
    rig = G.load_rigeuler_synthetic(INP / "RigEuler3DGS_synthetic_results.csv")
    return fa, rig, G.joined_table(fa, rig)


def close(a, b, rel=1e-6, abs_=1e-9):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


# 1 ------------------------------------------------------------------------------------------------------------
def test_01_schema_and_overlap():
    fa, rig, j = data()
    assert set(G.ORDER) <= set(fa.index), "FreeArt summary misses scenes"
    assert not rig.duplicated(["object", "run"]).any()
    assert len(j) == 10 and list(j.scene) == G.ORDER
    assert (j.joint_type == "revolute").sum() == 8 and (j.joint_type == "prismatic").sum() == 2
    broken = rig.drop(columns=["gt_hyp_axis_err_deg"])
    try:
        G.load_rigeuler_synthetic(_tmp_csv(broken))
    except G.InputError as e:
        assert "gt_hyp_axis_err_deg" in str(e)
    else:
        raise AssertionError("schema check must name the missing column")


def _tmp_csv(df):
    p = Path(tempfile.mkdtemp()) / "x.csv"; df.to_csv(p, index=False); return p


# 2 ------------------------------------------------------------------------------------------------------------
def test_02_freeart_protocol():
    fa, _, j = data()
    f = fa.loc[G.ORDER]
    assert (f.cameras == 8).all() and (f.timestamps == 60).all()
    assert f.joint_type_correct.astype(bool).all()
    assert (j.freeart_joint_type_correct & j.rig_joint_type_correct).all()


# 3 ------------------------------------------------------------------------------------------------------------
def test_03_rigeuler_selection():
    _, rig, j = data()
    sel = G.select_rigeuler(rig, "primary")
    assert (sel.run == "v5_canonical").sum() == 9
    assert sel.loc[G.WASHER, "run"] == "v5_gates_canonical"
    assert not (sel.run == "v4_bootstrap").any()
    assert j.loc[G.WASHER, "rig_version"] == "v5_gates_canonical"
    # the gates exception must not apply to any other scene
    bad = rig[~((rig.object == "USB_100109") & (rig.run == "v5_canonical"))].copy()
    bad.loc[(bad.object == G.WASHER) & (bad.run == "v5_gates_canonical"), "object"] = "USB_100109"
    try:
        G.select_rigeuler(bad, "primary")
    except G.InputError:
        pass
    else:
        raise AssertionError("v5_gates_canonical outside the washer must be rejected")


# 4 ------------------------------------------------------------------------------------------------------------
def test_04_conversion_oracles():
    _, _, j = data()
    assert close(j.loc["blade_103706", "rig_state_mae"], 0.496)
    assert close(j.loc["storage_45135", "rig_state_mae"], 0.182)
    assert close(j.loc["oven_101917", "rig_axis_line_distance_mm"], 1.487)
    assert np.isnan(j.loc["blade_103706", "rig_axis_line_distance_mm"])
    assert np.isnan(j.loc["storage_45135", "freeart_axis_line_distance_mm"])


# 5-7 ----------------------------------------------------------------------------------------------------------
def test_05_07_aggregates():
    _, _, j = data()
    m = G.aggregates(j)["methods"]
    fa, rg = m[G.FA], m[G.RIG]
    assert close(fa["mean_axis_direction_error_deg"], 0.0487402673, rel=1e-9)
    assert close(rg["mean_axis_direction_error_deg"], 0.05101, rel=1e-6)
    assert close(fa["mean_revolute_state_mae_deg"], 2.9441095411, rel=1e-9)
    assert close(rg["mean_revolute_state_mae_deg"], 0.1302125, rel=1e-6)
    assert close(fa["mean_prismatic_state_mae_mm"], 0.4933094810, rel=1e-9)
    assert close(rg["mean_prismatic_state_mae_mm"], 0.3390, rel=1e-6)
    assert close(fa["mean_revolute_axis_line_distance_mm"], 7.959081881, rel=1e-9)
    assert close(rg["mean_revolute_axis_line_distance_mm"], 1.77025, rel=1e-6)
    # remaining Table 5 rows, at the table's precision
    for key, a, b in (("median_axis_direction_error_deg", 0.03046, 0.03650), ("median_revolute_state_mae_deg", 0.1909, 0.1501),
                      ("mean_revolute_state_mae_excl_washer_deg", 0.4226, 0.1273),
                      ("median_revolute_axis_line_distance_mm", 0.8666, 1.6625), ("mean_gaussian_count", 73064.1, 29977.4)):
        assert abs(fa[key] - a) <= 5e-5 * max(1, abs(a)) * 10, (key, fa[key], a)
        assert abs(rg[key] - b) <= 5e-5 * max(1, abs(b)) * 10, (key, rg[key], b)
    assert fa["correct_joint_types"] == rg["correct_joint_types"] == "10/10"


# 8 ------------------------------------------------------------------------------------------------------------
def test_08_washer_and_log_range():
    _, _, j = data()
    assert close(j.loc[G.WASHER, "freeart_state_mae"], 20.5944087472, rel=1e-9)
    assert close(j.loc[G.WASHER, "rig_state_mae"], 0.1506)
    assert close(j.loc[G.WASHER, "rig_v4_state_mae"], 38.1001)
    rev = j[j.joint_type == "revolute"]
    lo, hi = G.log_limits(np.r_[rev.freeart_state_mae, rev.rig_state_mae])
    assert lo <= rev[["freeart_state_mae", "rig_state_mae"]].min().min() and hi >= 20.5944087472
    lo, hi = G.log_limits(np.r_[rev.rig_state_mae, rev.rig_v4_state_mae])
    assert hi >= 38.1001
    lo, hi = G.log_limits(np.r_[rev.freeart_axis_line_distance_mm, rev.rig_axis_line_distance_mm])
    assert hi >= 55.486


# 9 ------------------------------------------------------------------------------------------------------------
def test_09_gt_ranges_agree():
    _, _, j = data()
    assert np.allclose(j["_rig_range_gt"], j["gt_motion_range"], atol=2e-3)
    assert abs(j.loc[G.WASHER, "gt_motion_range"] - 76.266) < 1e-3
    assert abs(j.loc["blade_103706", "gt_motion_range"] - 835.2) < 1e-3


# 10 -----------------------------------------------------------------------------------------------------------
def test_10_freeart_trajectories():
    fa, _, _ = data()
    rep = G.load_freeart_report(INP / "FreeArtGS_report.json")
    for s in G.ORDER:
        t = rep[s]
        assert len(t) == 60, s
        assert t.unit.iloc[0] == fa.loc[s, "state_unit"]
        mae = float(np.abs(t["predicted"] - t["gt"]).mean())
        assert close(mae, fa.loc[s, "state_mae"], rel=1e-9), (s, mae, fa.loc[s, "state_mae"])
        assert close(t.attrs["conv"] * t.attrs["saved_mae_native"], fa.loc[s, "state_mae"], rel=1e-9)


# 11 -----------------------------------------------------------------------------------------------------------
def test_11_rendering_metrics():
    fa, _, j = data()
    psnr = G.load_freeart_psnr(INP / "FreeArtGS_per_observation_psnr.csv")
    have = sorted(psnr.scene.unique())
    assert len(have) == 7 and len(psnr) == 3360
    for s in ("foldchair_102255", "laptop_10211", "storage_45135"):
        assert s not in have and pd.isna(j.loc[s, "freeart_final_foreground_psnr_db"])
    for s in have:
        assert close(psnr[psnr.scene == s].foreground_psnr_db.mean(), fa.loc[s, "final_foreground_psnr_db"], rel=1e-9)
    assert not any(("rig" in c and any(k in c for k in ("psnr", "ssim", "lpips"))) for c in j.columns)


# 12 -----------------------------------------------------------------------------------------------------------
def test_12_generates_required_outputs():
    out = Path(tempfile.mkdtemp()) / "figs"
    assert G.main(["--outdir", str(out), "--dpi", "150"]) == 0
    stems = sorted({p.stem for p in out.glob("fig*.pdf")})
    assert stems == sorted(G.REQUIRED), stems
    for s in G.REQUIRED:
        assert (out / f"{s}.png").stat().st_size > 10_000 and (out / f"{s}.pdf").stat().st_size > 5_000
    md = (out / "FIGURES.md").read_text()
    for s in G.REQUIRED:
        assert f"## {s}" in md
    assert md.count("- Plotted fields:") >= 9 and md.count("- Units:") >= 9
    m = json.loads((out / "metrics_recomputed.json").read_text())
    assert m["n_scenes"] == 10 and m["n_contributing"]["revolute_mae"] == 8
    assert sorted(p.stem for p in (out / "supplementary").glob("*.pdf")) == [
        "figS1_rigeuler_trajectories", "figS2_per_frame_abs_error", "figS3_type_criterion",
        "figS4_real_phase1_drift", "figS5_real_drift_attribution"]


def test_12b_supplied_comparison_csv_matches_raw_inputs():
    _, _, j = data()
    G.validate_against_comparison(j, HERE / "freeart_rigeuler_comparison.csv")
    G.validate_against_comparison(j, INP / "freeart_rigeuler_comparison.csv")
    ref = pd.read_csv(HERE / "freeart_rigeuler_comparison.csv")
    tampered = ref.copy(); tampered.loc[0, "rig_axis_error_deg"] *= 1.01
    try:
        G.validate_against_comparison(j, _tmp_csv(tampered))
    except G.InputError:
        pass
    else:
        raise AssertionError("a disagreeing comparison CSV must be rejected")


# 13 -----------------------------------------------------------------------------------------------------------
def test_13_rejects_mixed_unit_mean():
    _, _, j = data()
    try:
        G.mean_state_mae(j.freeart_state_mae, j.state_unit)
    except G.MixedUnitError:
        pass
    else:
        raise AssertionError("a deg + mm joint-state mean must raise MixedUnitError")
    v, u = G.mean_state_mae(j[j.state_unit == "mm"].rig_state_mae, j[j.state_unit == "mm"].state_unit)
    assert u == "mm" and close(v, 0.339)


# supplementary ------------------------------------------------------------------------------------------------
def test_s1_rigeuler_trajectories_reproduce_reported_errors():
    fa, rig, j = data()
    tr = pd.read_csv(INP / "RigEuler3DGS_synthetic_trajectories.csv")
    curves = G.rigeuler_curves(tr, j)
    for s, t in curves.items():
        assert len(t) == 60, s
        assert close(t.abs_error.max(), j.loc[s, "rig_state_max_abs_error"], rel=2e-3, abs_=2e-4), s
        assert t.source_frame_index.iloc[0] == 0 and t.source_frame_index.iloc[-1] == 59
        canon = int(t.canonical_frame.iloc[0])
        assert canon in (0, 59) and abs(t[t.source_frame_index == canon].q_gt.iloc[0]) < 1e-12


def test_s2_real_table_is_categorical_only():
    real = pd.read_csv(INP / "RigEuler3DGS_real_results.csv")
    assert len(real) == 24 and real.scene.nunique() == 8
    agree = {a: int(real[real.arm == a].v6_type_decision_matches_expected.astype(str).eq("True").sum())
             for a in ("C", "majority", "unanimous")}
    assert agree == {"C": 5, "majority": 6, "unanimous": 5}, agree
    assert not any(c.endswith(("_axis_err_deg", "_q_mae_deg_p3")) for c in real.columns)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
