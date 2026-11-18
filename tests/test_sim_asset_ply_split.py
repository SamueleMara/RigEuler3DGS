"""Gate A (CPU): lossless split of the Phase-5 articulated PLY + Phase-5 input preflight errors.

  * SH degree 0 and 3, non-unit stored rot, log-scale / opacity logits, unique per-row values: every 3DGS field of
    both outputs is bit-identical to the source rows (matched by UID, not by the exporter's own row map), same order,
    same dtype; no extra fields / elements; part membership correct; no duplicated / missing UIDs; source hash unchanged
  * unresolved rows and their descendants survive into lineage.npz (membership marked uncertain)
  * impossible labels, an empty part, a missing joint element, q_canonical != 0, a corrupted joint copy, a missing
    complete.json and an unknown vertex field are explicit errors; NOT_FOR_RESULTS is an honest warning
  * both PLYs load with the repository's Graphdeco-convention reader and (if available) PArtTwin1D's load_gaussian_ply
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from recovery.articulated_ply import descendants_of  # noqa: E402
from recovery.sim_asset_export import ExportError, export_phase5_sim_asset  # noqa: E402
from recovery.sim_asset_export.load_phase5 import Phase5InputError, load_phase5  # noqa: E402
from recovery.sim_asset_export.ply_split import PlySplitError, sha256_file, split_articulated_ply  # noqa: E402
from sim_runtimes import report_not_run, run_all, runtime  # noqa: E402
from synthetic_phase5_run import make_phase5_run  # noqa: E402


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view(f"u{a.dtype.itemsize}")


def test_split_is_bit_identical_for_sh0_and_sh3():
    for deg in (0, 3):
        with tempfile.TemporaryDirectory() as tmp:
            run = make_phase5_run(tmp, "revolute", "forward", sh_degree=deg, seed=deg)
            src = run["phase5_dir"] / "final" / "point_cloud.ply"
            h0 = sha256_file(src)
            r = export_phase5_sim_asset(run["phase5_dir"], mode="visual_only", log=lambda *a: None)
            assert r.status == "VISUAL_ONLY" and sha256_file(src) == h0
            sv = PlyData.read(str(src))["vertex"].data
            gs_fields = [n for n in sv.dtype.names if n not in
                         ("part_id", "gaussian_uid", "original_id", "parent_uid", "provenance", "unresolved")]
            assert len([f for f in gs_fields if f.startswith("f_rest_")]) == 3 * ((deg + 1) ** 2 - 1)
            by_uid = {int(u): i for i, u in enumerate(sv["gaussian_uid"])}
            lin = np.load(r.asset_root / "lineage.npz")
            seen = []
            for lk, pid in (("static", 0), ("moving", 1)):
                p = PlyData.read(str(r.asset_root / "visual" / f"{lk}.ply"))
                assert [e.name for e in p.elements] == ["vertex"], "only the vertex element"
                v = p["vertex"].data
                assert list(v.dtype.names) == gs_fields, "same 3DGS fields, same order, nothing extra"
                assert [v.dtype[f].str for f in gs_fields] == [sv.dtype[f].str for f in gs_fields]
                uids = lin[f"{lk}_gaussian_uid"]
                rows = np.array([by_uid[int(u)] for u in uids])
                assert (sv["part_id"][rows] == pid).all() and len(rows) == int((sv["part_id"] == pid).sum())
                for f in gs_fields:
                    assert np.array_equal(_bits(v[f]), _bits(sv[f][rows])), f
                seen.append(uids)
                assert np.array_equal(lin[f"{lk}_source_row"], rows)
            allu = np.concatenate(seen)
            assert len(np.unique(allu)) == len(allu) == len(sv), "no duplicated / missing UIDs"
            rot = np.stack([PlyData.read(str(r.asset_root / "visual" / "moving.ply"))["vertex"].data[f"rot_{i}"]
                            for i in range(4)], 1)
            assert np.abs(np.linalg.norm(rot, axis=1) - 1).max() > 0.1, "stored quaternions stay unnormalized"


def test_unresolved_and_descendants_reach_the_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        run = make_phase5_run(tmp, "prismatic", "backward", sh_degree=1)
        r = export_phase5_sim_asset(run["phase5_dir"], mode="visual_only", log=lambda *a: None)
        lin = np.load(r.asset_root / "lineage.npz")
        sv = PlyData.read(str(run["phase5_dir"] / "final" / "point_cloud.ply"))["vertex"].data
        seeds = lin["phase5_unresolved_originals_gaussian_uid"]
        want = descendants_of(np.asarray(sv["gaussian_uid"], np.int64), np.asarray(sv["parent_uid"], np.int64), seeds)
        got = np.zeros(len(sv), bool)
        for lk in ("static", "moving"):
            got[lin[f"{lk}_source_row"]] = lin[f"{lk}_unresolved"].astype(bool)
        assert np.array_equal(got, want) and want.sum() > len(seeds) - 1
        assert (np.asarray(sv["parent_uid"])[want] >= 0).any() or want.sum() == len(seeds)
        assert "UNCERTAIN" in str(lin["note"])


def _rewrite(ply_path, vertex=None, joint=None, drop=()):
    p = PlyData.read(str(ply_path))
    els = []
    for e in p.elements:
        if e.name in drop:
            continue
        d = e.data.copy()
        if e.name == "vertex" and vertex:
            vertex(d)
        if e.name == "joint" and joint:
            joint(d)
        els.append(PlyElement.describe(d, e.name))
    PlyData(els, comments=p.comments).write(str(ply_path))


def _expect(exc, fn, needle):
    try:
        fn()
    except exc as e:
        assert needle in str(e), f"{needle!r} not in {e}"
        return str(e)
    raise AssertionError(f"expected {exc.__name__} ({needle})")


def test_input_errors_are_explicit():
    with tempfile.TemporaryDirectory() as tmp:
        base = make_phase5_run(Path(tmp) / "base", "revolute", "forward")
        p5 = base["phase5_dir"]

        def fresh(name):
            d = Path(tmp) / name
            shutil.copytree(base["phase5_dir"].parents[1], d)
            return d / "phase5" / "run"
        q = fresh("label")
        _rewrite(q / "final" / "point_cloud.ply", vertex=lambda d: d["part_id"].__setitem__(0, 2))
        _expect(Phase5InputError, lambda: load_phase5(q), "outside")
        q = fresh("empty")
        _rewrite(q / "final" / "point_cloud.ply", vertex=lambda d: d["part_id"].__setitem__(slice(None), 0))
        _expect(Phase5InputError, lambda: load_phase5(q), "zero moving")
        q = fresh("nojoint")
        _rewrite(q / "final" / "point_cloud.ply", drop=("joint",))
        _expect(Phase5InputError, lambda: load_phase5(q), "'joint'")
        q = fresh("anchor")
        _rewrite(q / "final" / "point_cloud.ply", joint=lambda d: d["q_canonical"].__setitem__(0, 0.1))
        _expect(Phase5InputError, lambda: load_phase5(q), "q_canonical")
        q = fresh("hash")
        t = q / "final" / "joint" / "trajectory.csv"
        t.write_text(t.read_text() + "\n")
        _expect(Phase5InputError, lambda: load_phase5(q), "fingerprint")
        q = fresh("incomplete")
        (q / "complete.json").unlink()
        _expect(Phase5InputError, lambda: load_phase5(q), "incomplete")
        _expect(ExportError, lambda: export_phase5_sim_asset(q, log=lambda *a: None), "incomplete")
        q = fresh("unknown")
        p = PlyData.read(str(q / "final" / "point_cloud.ply"))
        v = p["vertex"].data
        nd = np.empty(len(v), dtype=v.dtype.descr + [("mystery", "<f4")])
        for k in v.dtype.names:
            nd[k] = v[k]
        nd["mystery"] = 1.0
        PlyData([PlyElement.describe(nd, "vertex"), p["joint"], p["joint_state"]]).write(str(q / "final" / "pc2.ply"))
        _expect(PlySplitError, lambda: split_articulated_ply(q / "final" / "pc2.ply", {
            "static_link": Path(tmp) / "s.ply", "moving_link": Path(tmp) / "m.ply"}), "mystery")
        q = fresh("nfr")
        c = json.loads((q / "complete.json").read_text())
        (q / "complete.json").write_text(json.dumps({**c, "NOT_FOR_RESULTS": True}))
        r = export_phase5_sim_asset(q, mode="visual_only", log=lambda *a: None)
        assert r.NOT_FOR_RESULTS and any("NOT_FOR_RESULTS" in w for w in r.warnings)
        assert p5.is_dir()


def test_standard_readers_load_both_plys():
    from recovery.gaussians import GaussianTensors
    with tempfile.TemporaryDirectory() as tmp:
        run = make_phase5_run(tmp, "revolute", "forward", sh_degree=3)
        r = export_phase5_sim_asset(run["phase5_dir"], mode="visual_only", log=lambda *a: None)
        for lk in ("static", "moving"):
            g = GaussianTensors.load_ply(r.asset_root / "visual" / f"{lk}.ply", device="cpu")
            assert len(g) == int(np.load(r.asset_root / "lineage.npz")[f"{lk}_source_row"].size)
        py = runtime("sapien")
        if not py:
            report_not_run("PArtTwin1D load_gaussian_ply", "sapien")
            return
        code = ("import sys;from parttwin1d.assets.gaussian_ply import load_gaussian_ply as L;"
                "print(*[L(p).count for p in sys.argv[1:]], *[L(p).sh_degree for p in sys.argv[1:]])")
        out = subprocess.run([py, "-c", code, str(r.asset_root / "visual" / "static.ply"),
                              str(r.asset_root / "visual" / "moving.ply")], capture_output=True, text=True)
        if out.returncode and "No module named 'parttwin1d'" in out.stderr:
            report_not_run("PArtTwin1D load_gaussian_ply", "sapien (+parttwin1d)")
            return
        assert out.returncode == 0, out.stderr
        ns, nm, ds, dm = map(int, out.stdout.split())
        assert (ds, dm) == (3, 3) and ns + nm == len(run["part"])
        print(f"  PArtTwin1D load_gaussian_ply: {ns} + {nm} Gaussians, SH degree 3")


if __name__ == "__main__":
    run_all(dict(globals()))
