"""Atomic staging, hashes, manifest, completion and idempotence (spec 7, transaction semantics).

The asset is built in <out_root>/.staging-<dir>-<random>/ (same filesystem), validated, and promoted with ONE
os.rename to <out_root>/<dir>/. export_complete.json is the last file written before the rename. On any error the
staging directory is removed and <out_root>/<dir>.export_failed.json records why; a directory that looks complete
is never left behind. An existing asset is reused only when its input signature matches and every file hash
verifies; a different signature is refused unless an explicit new revision is requested.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .ply_split import sha256_file

COMPLETE = "export_complete.json"
MANIFEST = "export_manifest.json"


class ArtifactError(RuntimeError):
    pass


def code_sha256() -> str:
    """Hash of the exporter's own source (part of the input signature)."""
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def signature(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_hashes(d: Path, exclude=(MANIFEST, COMPLETE)) -> dict:
    return {str(p.relative_to(d)): sha256_file(p) for p in sorted(d.rglob("*")) if p.is_file()
            and str(p.relative_to(d)) not in exclude}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")


def asset_dirname(scene: str, revision: str | None) -> str:
    if revision is None:
        return scene
    if not revision or any(c in revision for c in "/\\ ") or revision.startswith("."):
        raise ArtifactError(f"invalid revision name {revision!r}")
    return f"{scene}.rev-{revision}"


def verify_existing(d: Path) -> dict:
    """export_complete.json + manifest of an existing asset; every listed file must hash identically."""
    c = d / COMPLETE
    if not c.is_file():
        raise ArtifactError(f"{d} exists without {COMPLETE} (an interrupted or foreign directory); remove it or "
                            f"choose another --asset-revision")
    comp = json.loads(c.read_text())
    man_p = d / MANIFEST
    if sha256_file(man_p) != comp.get("manifest_sha256"):
        raise ArtifactError(f"{man_p} was modified after export")
    man = json.loads(man_p.read_text())
    now = file_hashes(d)
    if now != man["files"]:
        diff = sorted(set(now) ^ set(man["files"]) | {k for k in now if man["files"].get(k) != now[k]})
        raise ArtifactError(f"{d}: files changed after export: {diff[:10]}")
    return {"complete": comp, "manifest": man}


class Staging:
    def __init__(self, out_root: Path, dirname: str):
        self.out_root = Path(out_root)
        self.final = self.out_root / dirname
        self.dirname = dirname
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.path = self.out_root / f".staging-{dirname}-{os.urandom(6).hex()}"
        self.path.mkdir()

    def promote(self, complete: dict) -> Path:
        man = self.path / MANIFEST
        if not man.is_file():
            raise ArtifactError("manifest missing at promotion")
        complete = {**complete, "manifest_sha256": sha256_file(man)}
        write_json(self.path / COMPLETE, complete)
        if self.final.exists():
            raise ArtifactError(f"{self.final} appeared during the export; not overwritten")
        os.rename(self.path, self.final)
        failed = self.out_root / f"{self.dirname}.export_failed.json"
        if failed.is_file():
            failed.unlink()
        return self.final

    def abort(self, error: BaseException) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
        try:
            write_json(self.out_root / f"{self.dirname}.export_failed.json",
                       {"error": f"{type(error).__name__}: {error}", "asset_dir_written": False})
        except OSError:
            pass
