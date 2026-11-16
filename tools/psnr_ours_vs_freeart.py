#!/usr/bin/env python
"""Masked PSNR, FreeArtGS definition (recovery/fair_psnr.py), ours vs FreeArtGS on the same views. CPU only.

    python tools/psnr_ours_vs_freeart.py [--scenes oven_101917 storage_45135 ...] [--output-dir DIR]

FreeArtGS: its saved final renders <freeart-root>/<scene>_multiview/splatfacto-art/eval_results/rgb_<stem>.png
(every training view, white background, 8-bit), mapped to (camera, frame) through the run's multiview.json.
Ours: <ours-root>/<scene>/phase5/<run>/comparison/psnr_views.csv, written by tools/phase5_comparison.py from the
Phase-5 final model (newest complete run per scene unless --ours SCENE=PHASE5_DIR).
Both are scored against the SAME recorded files: the reference RGB + mask bytes FreeArtGS's multiview.json points
to must hash equal to the ones our run used, view by view, or the scene is refused. The headline numbers use only the
views both methods rendered.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from recovery.fair_psnr import DEFINITION, load_reference, score_view, summarize  # noqa: E402

RES = Path("/media/samuelemara/Extreme SSD/Samuele/results_11_10_26")
SCENES = ("foldchair_102255 laptop_10211 storage_45135 oven_101917 scissor_11100 washer_103776 blade_103706 "
          "fridge_10905 stapler_103111 USB_100109").split()


def ours_run(scene_dir: Path, override: str | None):
    if override:
        d = Path(override)
    else:
        cands = [p.parent.parent for p in scene_dir.glob("phase5/*/comparison/comparison_info.json")]
        cands = [c for c in cands if (c / "comparison" / "psnr_views.csv").is_file() and json.loads(
            (c / "comparison" / "comparison_info.json").read_text()).get("psnr_masked_freeart", {}).get("complete")]
        if not cands:
            return None
        d = max(cands, key=lambda c: (c / "comparison" / "psnr_views.csv").stat().st_mtime)
    info = json.loads((d / "comparison" / "comparison_info.json").read_text())
    rows = [dict(camera=r["camera"], frame=int(r["frame"]), psnr=float(r["psnr"]), mask_empty=r["mask_empty"] == "1",
                 reference_sha256=r["reference_sha256"])
            for r in csv.DictReader(open(d / "comparison" / "psnr_views.csv"))]
    return d, info, rows


class NotComparable(Exception):
    pass


def freeart_rows(run: Path, ours_source: str):
    tt = json.loads((run / "transforms_train.json").read_text())
    frames_tt = tt["frames"] if isinstance(tt, dict) else tt
    dataset = Path(frames_tt[0]["file_path"]).parent.parent
    if not (dataset / "multiview.json").is_file():
        # 2026-11-12: foldchair / laptop / storage FreeArtGS runs (2026-10-11) were trained on a deleted
        # ~/FreeArtGS/datasets/<scene> copy whose renders match the ease-in-out / trapezoid render sets
        # (silhouette IoU 0.85-0.98), not the linear set ours uses (0.25-0.39): different images, no fair PSNR
        raise NotComparable(f"FreeArtGS dataset {dataset} (multiview.json) no longer exists, so its reference "
                            f"images cannot be checked against ours")
    mv = json.loads((dataset / "multiview.json").read_text())
    rows, bad_views = [], []
    for v in mv["frames"]:
        stem = Path(v["file_path"]).stem
        pred = np.asarray(Image.open(run / "eval_results" / f"rgb_{stem}.png").convert("RGB"))
        gt, mask, sha = load_reference(ours_source, v["camera_name"], v["source_stem"])
        # the images FreeArtGS ACTUALLY trained on: its prepared copy, not the source folder (which can change later;
        # 2026-11-12: oven's source was re-rendered mid-run, so cameras 0-4 trained on the old render)
        ori = [p for p in (dataset / "images_ori").glob(stem + ".*")]
        if not ori:
            raise NotComparable(f"{dataset}/images_ori/{stem}.* (FreeArtGS's training copy) is missing")
        trained = np.asarray(Image.open(ori[0]).convert("RGB"))
        if trained.shape != gt.shape or not np.array_equal(trained, gt):
            bad_views.append(f"{v['camera_name']}/{v['source_stem']}")
            continue
        rows.append(dict(camera=v["camera_name"], frame=int(v["source_stem"].rsplit("_", 1)[-1]),
                         **score_view(pred, gt, mask), reference_sha256=sha))
    if bad_views:
        cams = sorted({b.split("/")[0] for b in bad_views})
        raise NotComparable(f"FreeArtGS trained on different images than ours in {len(bad_views)} of "
                            f"{len(mv['frames'])} views (cameras {', '.join(cams)}); e.g. {bad_views[0]}")
    return rows, {"freeart_run": str(run), "freeart_dataset": str(dataset), "freeart_source_scene": mv["source_scene"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--freeart-root", action="append", default=None,
                    help="FreeArtGS output roots, searched in order; the first holding a comparable "
                         "<scene>_multiview run wins (default: freeart_linear_rerun, then freeart)")
    ap.add_argument("--ours-root", default=str(RES / "ours_phase4_corrected_synthetic"))
    ap.add_argument("--ours", action="append", default=[], help="SCENE=PHASE5_DIR (default: newest complete run)")
    ap.add_argument("--output-dir", default=None, help="default <ours-root>/psnr_ours_vs_freeart")
    a = ap.parse_args()
    over = dict(x.split("=", 1) for x in a.ours)
    out = Path(a.output_dir or Path(a.ours_root) / "psnr_ours_vs_freeart")
    out.mkdir(parents=True, exist_ok=True)
    table, skipped = [], []
    for s in a.scenes:
        roots = a.freeart_root or [str(RES / "freeart_linear_rerun"), str(RES / "freeart")]
        runs = [Path(r) / f"{s}_multiview" / "splatfacto-art" for r in roots]
        runs = [r for r in runs if (r / "eval_results").is_dir() and (r / "transforms_train.json").is_file()]
        fa_run = runs[0] if runs else Path(roots[-1]) / f"{s}_multiview" / "splatfacto-art"
        o = ours_run(Path(a.ours_root) / s, over.get(s))
        if o is None or not (fa_run / "eval_results").is_dir():
            print(f"{s}: skipped ({'no complete ours comparison/psnr_views.csv' if o is None else 'no FreeArtGS eval_results'})")
            continue
        od, oinfo, orows = o
        try:
            frows, fmeta = freeart_rows(fa_run, oinfo["psnr_masked_freeart"]["reference_source"])
        except NotComparable as e:
            print(f"{s}: NOT COMPARABLE -- {e}", flush=True)
            skipped.append({"scene": s, "reason": str(e)})
            continue
        ok, fk = {(r["camera"], r["frame"]): r for r in orows}, {(r["camera"], r["frame"]): r for r in frows}
        common = sorted(set(ok) & set(fk))
        bad = [k for k in common if ok[k]["reference_sha256"] != fk[k]["reference_sha256"]]
        if bad:
            raise SystemExit(f"{s}: {len(bad)} common views scored against different reference bytes, e.g. {bad[0]}")
        so, sf = summarize([ok[k] for k in common]), summarize([fk[k] for k in common])
        rec = {"scene": s, "views_common": len(common), "views_ours": len(ok), "views_freeart": len(fk),
               "psnr_ours": so["psnr"], "psnr_freeart": sf["psnr"], "delta_db": so["psnr"] - sf["psnr"],
               "ours_phase5_dir": str(od), **fmeta, "ours": so, "freeart": sf}
        (out / f"{s}.json").write_text(json.dumps(rec, indent=2))
        with open(out / f"{s}_freeart_psnr_views.csv", "w") as fh:
            fh.write("camera,frame,psnr,mask_empty,reference_sha256\n")
            for r in frows:
                fh.write(f"{r['camera']},{r['frame']},{r['psnr']:.6f},{int(r['mask_empty'])},{r['reference_sha256']}\n")
        table.append(rec)
        print(f"{s:18s} ours {so['psnr']:6.2f} dB | FreeArtGS {sf['psnr']:6.2f} dB | delta {so['psnr'] - sf['psnr']:+6.2f} "
              f"| {len(common)} common views ({od.name})", flush=True)
    if not table:
        return
    keys = ("scene", "psnr_ours", "psnr_freeart", "delta_db", "views_common", "views_ours", "views_freeart",
            "ours_phase5_dir", "freeart_run")
    with open(out / "table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in table)
    mo, mf = float(np.mean([r["psnr_ours"] for r in table])), float(np.mean([r["psnr_freeart"] for r in table]))
    (out / "summary.json").write_text(json.dumps({"scenes": len(table), "mean_psnr_ours": mo, "mean_psnr_freeart": mf,
                                                  "not_comparable": skipped, "definition": DEFINITION}, indent=2))
    print(f"{'mean (' + str(len(table)) + ' scenes)':18s} ours {mo:6.2f} dB | FreeArtGS {mf:6.2f} dB | delta {mo - mf:+6.2f}"
          f"  -> {out}/table.csv")


if __name__ == "__main__":
    main()
