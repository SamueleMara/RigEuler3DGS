#!/usr/bin/env python
"""Add the missing per-frame node trajectories to existing
debug/motion_coherence_support_3d.html pages (the temporal slider was disabled
because the runner passed no trajectory), without rerunning Phase 1.

For every Phase-1 output directory found under the given paths it
  * reads the page's embedded payload (graph, labels, classes, components),
  * rebuilds the coherence supernode partition from canonical_xyz.npy with the
    run's own settings (config.json seed; --supernode-count, the driver's 192),
  * VERIFIES that the rebuilt node centroids equal the page's node positions
    (the page is left untouched if they do not),
  * averages the saved phase1_centers/centers_frame_NNNN.npy (original
    chronological frames) over each node's members, and re-renders the page.

    python tools/repair_support_3d_slider.py <results root or scene or phase1 dir> [...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PAGE = Path("debug") / "motion_coherence_support_3d.html"


def phase1_dirs(paths):
    for p in paths:
        p = Path(p)
        if (p / PAGE).is_file():
            yield p
        else:
            yield from sorted(q.parent.parent for q in p.rglob(str(PAGE)))


def repair(d: Path, supernode_count: int, force: bool) -> str:
    from debug.motion_coherence_debug import read_support_payload, write_support_page
    from estimator.motion_coherence_support import supernode_trajectories
    from estimator.supernodes import build_supernodes
    page = d / PAGE
    payload = read_support_payload(page)
    if payload.get("frames") and not force:
        return "already has frames"
    cdir = d / "phase1_centers"
    mu0 = torch.from_numpy(np.load(cdir / "canonical_xyz.npy")).double()
    files = sorted(cdir.glob("centers_frame_*.npy"))
    if not files:
        return "no phase1_centers"
    frame_ids = [int(f.stem.split("_")[-1]) for f in files]
    positions = [torch.from_numpy(np.load(f)).double() for f in files]
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").is_file() else {}
    with open(d / "numerical" / "nodes.csv") as fh:
        node_ids = [int(r["supernode"]) for r in csv.DictReader(fh)]
    size = int(cfg.get("supernode_target_size") or 0)
    seed = int(cfg.get("seed", 0))
    if size:
        part = build_supernodes(mu0.float(), target_size=size, seed=seed)
    else:
        part = build_supernodes(mu0.float(), num_supernodes=min(supernode_count, mu0.shape[0]), seed=seed)
    sid = part.supernode_id.long().cpu()
    xyz = torch.tensor(payload["xyz"], dtype=torch.float64).reshape(-1, 3)
    if xyz.shape[0] != len(node_ids):
        return f"REFUSED: page has {xyz.shape[0]} nodes, nodes.csv {len(node_ids)}"
    cent = torch.stack([mu0[sid == k].mean(0) for k in node_ids])
    err = float((cent - xyz).abs().max())
    if not err < 2e-5:                      # the page stores 5 decimals
        return f"REFUSED: rebuilt partition does not match the page (max centroid error {err:.2e})"
    traj = supernode_trajectories(positions, sid)["trajectories"][:, node_ids, :]
    payload["frames"] = [traj[t].numpy().round(5).reshape(-1).tolist() for t in range(traj.shape[0])]
    payload["frame_ids"] = frame_ids
    title = page.read_text().split("<title>", 1)[1].split("</title>", 1)[0]
    write_support_page(page, payload, title)
    return f"repaired: {len(frame_ids)} frames x {len(node_ids)} nodes (partition check {err:.1e})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--supernode-count", type=int, default=192,
                    help="the coherence graph node count of the run (driver SNCOUNT, default 192)")
    ap.add_argument("--force", action="store_true", help="rewrite pages that already have frames")
    args = ap.parse_args(argv)
    n = 0
    for d in phase1_dirs(args.paths):
        try:
            msg = repair(d, args.supernode_count, args.force)
        except Exception as ex:                 # one bad directory must not stop the rest
            msg = f"FAILED: {type(ex).__name__}: {ex}"
        print(f"{d}: {msg}")
        n += 1
    if not n:
        print("no motion_coherence_support_3d.html found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
