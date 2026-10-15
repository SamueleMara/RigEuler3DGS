"""GT access firewall for the endpoint diagnostics.

Installed as a Python audit hook (sys.addaudithook): every file the process
opens through Python (open, io.open, os.open, numpy, PIL, plyfile, json ...)
under the guarded roots (the results root and the dataset root) is checked. A forbidden path raises GTAccessError before a byte is read, and
every opened path under the data/results roots is recorded for provenance.

Forbidden (evaluation-only or ground-truth content):
  * joint_reduction/gt_evaluation.json and any *gt_evaluation* file
  * <*_rigid>/summary.json and results-root SUMMARY.txt (they embed GT metrics)
  * dataset metadata/ (joint_info.json, frame_values.csv: GT joint type, axis, q)
  * points3D_parts*.ply and the dataset pointcloud/ directory (GT part labels)
  * the dataset mask/ directory (part / object masks are not sensor data here)
  * mesh / articulation descriptions (.urdf .obj .stl .glb .gltf .dae .mtl)
  * any file whose name contains a gt token (gt_*, *_gt.*, *_gt_*)

Allowed explicitly: dataset rgb/ and depth_npy/ (the RGB-D sensor stream),
colmap*/sparse/ camera calibration (cameras.bin, images.bin; the scene loader
also parses points3D.ply there, whose contents this analysis never uses), the
trained endpoint PLYs and cfg_args, Phase-1 tracks, Phase-2 rigid poses and
joint_reduction estimates (not their GT evaluation).
"""

from __future__ import annotations

import os
import re
import sys

FORBIDDEN = [
    (re.compile(r"gt_evaluation", re.I), "GT joint evaluation"),
    (re.compile(r"_rigid/summary\.json$"), "rigid summary embeds gt_evaluation"),
    (re.compile(r"/SUMMARY\.txt$"), "results summary embeds GT scores"),
    (re.compile(r"/metadata/"), "dataset metadata: GT joint type/axis/values"),
    (re.compile(r"joint_info\.json$|frame_values\.csv$"), "GT joint description/values"),
    (re.compile(r"points3D_parts[^/]*\.ply$"), "GT part-labelled point cloud"),
    (re.compile(r"/pointcloud/"), "dataset GT point cloud directory"),
    (re.compile(r"/mask/"), "dataset masks"),
    (re.compile(r"\.(urdf|obj|stl|glb|gltf|dae|mtl)$", re.I), "mesh / articulation model"),
    (re.compile(r"(^|/)gt_[^/]*$|_gt(\.|_)[^/]*$", re.I), "gt-named file"),
]


class GTAccessError(PermissionError):
    pass


def violation(path) -> str | None:
    p = os.fspath(path) if not isinstance(path, int) else ""
    p = str(p).replace("\\", "/")
    for rx, why in FORBIDDEN:
        if rx.search(p):
            return why
    return None


def assert_allowed(path):
    why = violation(path)
    if why:
        raise GTAccessError(f"GT firewall: refused to open {path} ({why})")


class Firewall:
    """Process-wide; the audit hook cannot be removed once installed, so it is
    installed once and switched on/off through `active`."""

    _instance = None

    def __init__(self, watch_roots=()):
        self.active = False
        self.watch = [str(r) for r in watch_roots]
        self.opened: list[str] = []
        self.refused: list[str] = []

    @classmethod
    def install(cls, watch_roots=()):
        if cls._instance is None:
            cls._instance = cls(watch_roots)
            sys.addaudithook(cls._instance._hook)
        else:
            cls._instance.watch = sorted(set(cls._instance.watch) | {str(r) for r in watch_roots})
        cls._instance.active = True
        return cls._instance

    def _hook(self, event, args):
        if not self.active or event != "open" or not args:
            return
        path = args[0]
        if isinstance(path, int) or path is None:
            return
        p = os.fspath(path) if not isinstance(path, str) else path
        if isinstance(p, bytes):
            p = p.decode(errors="replace")
        # only the data and results trees are guarded: library files are not data
        if not any(p.startswith(w) for w in self.watch):
            return
        why = violation(p)
        if why:
            self.refused.append(p)
            raise GTAccessError(f"GT firewall: refused to open {p} ({why})")
        self.opened.append(p)

    def report(self) -> dict:
        import collections
        dirs = collections.Counter(os.path.dirname(p) for p in self.opened)
        return {"active": self.active, "rules": [why for _rx, why in FORBIDDEN],
                "refused": self.refused, "opened_files": len(self.opened),
                "opened_by_directory": dict(dirs)}
