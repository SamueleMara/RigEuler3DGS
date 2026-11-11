#!/usr/bin/env python
"""Which cameras a run uses: `--cameras all | N | cam_001,cam_002`.

    all, auto   every camera of the scene (the default)
    N           the first N cameras in sorted order, for the motion stages only
    NAME,NAME   exactly these cameras, for EVERY stage including the endpoint model

A camera is named by its rgb/ folder. cam_002, cam002 and cam_2 are the same
camera; the order given does not matter, the sorted order is what runs, so one
camera set always means one observation order and one stage signature. An
unknown name is refused, never substituted.

Commands used by run_bidirectional_all.sh (no GPU, no writes):

    resolve  --source S --cameras SPEC              print the comma list; exit 2 if refused
    endpoint --model M --source S --cameras SPEC    exit 1 if the model was built from
                                                    other cameras than SPEC needs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ALL = ("", "all", "auto")


def available_cameras(source) -> list:
    d = Path(source) / "rgb"
    return sorted(p.name for p in d.glob("cam_*") if p.is_dir()) if d.is_dir() else []


def camera_name(image_name) -> str:
    """cam_000/frame_000000.png -> cam_000"""
    m = re.search(r"cam_\d+", str(image_name))
    if not m:
        raise ValueError(f"no cam_NNN in the image name {image_name!r}")
    return m.group(0)


def is_name_list(spec) -> bool:
    s = "" if spec is None else str(spec).strip().lower()
    return s not in ALL and not s.isdigit()


def resolve(spec, available) -> list:
    """The selected cameras, in the order of `available`. ValueError if refused."""
    available = list(available)
    s = "" if spec is None else str(spec).strip()
    if s.lower() in ALL:
        return available
    if s.isdigit():
        if int(s) < 1:
            raise ValueError(f"--cameras {s}: at least one camera is needed")
        return available[:int(s)]
    index = {int(c.rsplit("_", 1)[1]): c for c in available}
    chosen = set()
    for tok in re.split(r"[,\s]+", s):
        m = re.fullmatch(r"cam_?(\d+)", tok)
        if not tok or not m or int(m.group(1)) not in index:
            raise ValueError(f"--cameras {s}: unknown camera {tok!r} (available: {','.join(available)})")
        chosen.add(index[int(m.group(1))])
    return [c for c in available if c in chosen]


def select_cameras(cams, spec) -> list:
    """The selected Camera objects of a loaded scene, in their given order."""
    names = [camera_name(c.image_name) for c in cams]
    try:
        chosen = set(resolve(spec, names))
    except ValueError as ex:
        raise SystemExit(str(ex))
    return [c for c, n in zip(cams, names) if n in chosen]


def endpoint_cameras(model):
    """The cameras an endpoint model was built from; None = every camera of its
    COLMAP model (static training without --cameras, and every older model)."""
    model = Path(model)
    man = model / "bootstrap_manifest.json"
    if man.is_file():
        names = json.loads(man.read_text()).get("camera_names")
        if names:
            return list(names)
    cfg = model / "cfg_args"
    if cfg.is_file():
        m = re.search(r"\bcameras='([^']*)'", cfg.read_text())
        if m and m.group(1):
            return m.group(1).split(",")
    return None


def endpoint_problem(model, spec, available):
    """Why this model cannot serve a run on `spec`, or None. A name list must match the
    model exactly: an endpoint fitted to another camera carries that camera's geometry
    into a run that is meant not to see it. A count or `all` needs the model to cover
    the selected cameras (the long-standing 64-camera model, 8-camera motion setting)."""
    want = resolve(spec, available)
    have = endpoint_cameras(model)
    have = list(available) if have is None else sorted(have)
    if is_name_list(spec):
        if have != want:
            return f"the endpoint model was built from {','.join(have)}, this run selects {','.join(want)}"
    elif set(want) - set(have):
        return f"the endpoint model was built from {','.join(have)} only, this run selects {','.join(want)}"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("resolve", "endpoint"))
    ap.add_argument("--source", required=True)
    ap.add_argument("--cameras", default="all")
    ap.add_argument("--model", default=None)
    a = ap.parse_args(argv)
    available = available_cameras(a.source)
    try:
        if a.command == "resolve":
            print(",".join(resolve(a.cameras, available)))
            return 0
        why = endpoint_problem(a.model, a.cameras, available)
    except ValueError as ex:
        print(ex)
        return 2
    if why:
        print(why)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
