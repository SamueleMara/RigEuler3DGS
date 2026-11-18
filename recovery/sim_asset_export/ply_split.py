"""Lossless split of the Phase-5 articulated PLY into two ordinary Graphdeco 3DGS PLYs (spec section 4).

Only row selection: the output vertex dtype is built from the SOURCE dtype descriptors of the 3DGS columns, in the
source order, and every column is copied as stored (no decode / activation / re-encode). The articulated extras
(part_id, gaussian_uid, original_id, parent_uid, provenance, unresolved) and the joint / joint_state elements are not
written to either PLY; they go to lineage.npz. A field that is neither a known 3DGS column nor a known articulated
extra is an error -- never silently dropped. The source file is read only.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ARTICULATED_EXTRA = ("part_id", "gaussian_uid", "original_id", "parent_uid", "provenance", "unresolved")
ARTICULATED_ELEMENTS = ("vertex", "joint", "joint_state")
REQUIRED_3DGS = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                 "rot_0", "rot_1", "rot_2", "rot_3")
_PATTERNS = (re.compile(r"^(x|y|z|nx|ny|nz|opacity)$"), re.compile(r"^f_dc_[0-2]$"), re.compile(r"^f_rest_\d+$"),
             re.compile(r"^scale_\d+$"), re.compile(r"^rot_\d+$"))
LINKS = (("static_link", 0), ("moving_link", 1))


class PlySplitError(ValueError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _numbered(names, prefix):
    idx = sorted(int(n[len(prefix):]) for n in names if n.startswith(prefix))
    if idx != list(range(len(idx))):
        raise PlySplitError(f"{prefix}* columns are not numbered 0..{len(idx) - 1}: {idx[:8]}...")
    return len(idx)


def classify_fields(names) -> list[str]:
    """The 3DGS columns of a vertex dtype, in source order. Raises on unknown fields / malformed SH."""
    gs, unknown = [], []
    for n in names:
        if n in ARTICULATED_EXTRA:
            continue
        if any(p.match(n) for p in _PATTERNS):
            gs.append(n)
        else:
            unknown.append(n)
    if unknown:
        raise PlySplitError(f"vertex fields that are neither 3DGS nor articulated metadata: {unknown} -- refusing to "
                            f"drop data silently")
    missing = [f for f in REQUIRED_3DGS if f not in gs]
    if missing:
        raise PlySplitError(f"not a Graphdeco 3DGS vertex element; missing {missing}")
    if _numbered(gs, "scale_") != 3 or _numbered(gs, "rot_") != 4:
        raise PlySplitError("expected scale_0..2 and rot_0..3")
    n_rest = _numbered(gs, "f_rest_")
    if n_rest % 3 or (n_rest // 3 + 1) not in (1, 4, 9, 16):
        raise PlySplitError(f"{n_rest} f_rest_* columns: not 3 x (K - 1) with K in 1, 4, 9, 16")
    if any(n in gs for n in ("nx", "ny", "nz")) and not all(n in gs for n in ("nx", "ny", "nz")):
        raise PlySplitError("partial normal columns")
    return gs


@dataclass
class SplitResult:
    fields: list
    dtypes: list
    rows: dict              # link -> source row indices (int64, ascending)
    counts: dict
    source_sha256: str
    sha256: dict            # link -> sha256 of the written PLY
    sh_degree: int
    verified: dict


def _bits(a: np.ndarray) -> np.ndarray:
    """Bit pattern view (so NaN payloads / -0.0 compare exactly)."""
    a = np.ascontiguousarray(a)
    return a.view(f"u{a.dtype.itemsize}") if a.dtype.kind == "f" else a


def split_articulated_ply(src_ply, out_paths: dict, comments=()) -> SplitResult:
    """out_paths: {'static_link': path, 'moving_link': path}. Writes binary little-endian PLYs with one vertex
    element holding exactly the source 3DGS columns of the rows with part_id 0 / 1."""
    src_ply = Path(src_ply)
    h0 = sha256_file(src_ply)
    ply = PlyData.read(str(src_ply))
    names = [e.name for e in ply.elements]
    if "vertex" not in names:
        raise PlySplitError(f"{src_ply}: no vertex element")
    for need in ("joint", "joint_state"):
        if need not in names:
            raise PlySplitError(f"{src_ply}: no {need!r} element -- not a Phase-5 articulated PLY")
    extra_el = [n for n in names if n not in ARTICULATED_ELEMENTS]
    if extra_el:
        raise PlySplitError(f"{src_ply}: unexpected PLY elements {extra_el}")
    v = ply["vertex"].data
    if "part_id" not in v.dtype.names:
        raise PlySplitError(f"{src_ply}: vertex has no part_id")
    fields = classify_fields(v.dtype.names)
    part = np.asarray(v["part_id"])
    if not np.isin(part, [0, 1]).all():
        bad = np.unique(part[~np.isin(part, [0, 1])])
        raise PlySplitError(f"part_id values {bad.tolist()} outside {{0 static, 1 moving}}")
    rows = {lk: np.nonzero(part == pid)[0].astype(np.int64) for lk, pid in LINKS}
    for lk, r in rows.items():
        if len(r) == 0:
            raise PlySplitError(f"{lk}: zero Gaussians (part_id {dict(LINKS)[lk]})")
    descr = [(f, v.dtype[f].str) for f in fields]
    out_sha, verified = {}, {}
    for lk, r in rows.items():
        sub = np.empty(len(r), dtype=np.dtype(descr))
        for f in fields:
            sub[f] = v[f][r]
        p = Path(out_paths[lk])
        p.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(sub, "vertex")], text=False, byte_order="<",
                comments=[f"rigeuler3dgs sim asset visual {lk}: {len(r)} Gaussians, rows of part_id {dict(LINKS)[lk]}",
                          f"source_sha256 {h0}", *comments]).write(str(p))
        out_sha[lk] = sha256_file(p)
        # read back with plyfile: same names, order, dtypes, bit-identical values, nothing else
        back = PlyData.read(str(p))
        if [e.name for e in back.elements] != ["vertex"]:
            raise PlySplitError(f"{p}: elements {[e.name for e in back.elements]}")
        bv = back["vertex"].data
        if list(bv.dtype.names) != fields or [bv.dtype[f].str for f in fields] != [d for _, d in descr]:
            raise PlySplitError(f"{p}: written schema differs from the source 3DGS schema")
        for f in fields:
            if not np.array_equal(_bits(bv[f]), _bits(v[f][r])):
                raise PlySplitError(f"{p}: field {f} differs from the source rows")
        verified[lk] = {"fields": len(fields), "rows": len(r), "bit_identical": True}
    # partition / reassembly: every source row exactly once, all fields
    both = np.concatenate([rows["static_link"], rows["moving_link"]])
    if len(both) != len(v) or not np.array_equal(np.sort(both), np.arange(len(v))):
        raise PlySplitError("static + moving rows are not a partition of the source rows")
    re_v = {f: np.empty(len(v), dtype=v.dtype[f]) for f in fields}
    for lk, r in rows.items():
        bv = PlyData.read(str(out_paths[lk]))["vertex"].data
        for f in fields:
            re_v[f][r] = bv[f]
    for f in fields:
        if not np.array_equal(_bits(re_v[f]), _bits(v[f])):
            raise PlySplitError(f"reassembly differs from the source in field {f}")
    verified["reassembly"] = {"rows": len(v), "fields": len(fields), "bit_identical": True}
    if sha256_file(src_ply) != h0:
        raise PlySplitError(f"{src_ply} changed while being split")
    verified["source_unchanged"] = True
    n_rest = sum(1 for f in fields if f.startswith("f_rest_"))
    return SplitResult(fields=fields, dtypes=[d for _, d in descr], rows=rows,
                       counts={lk: int(len(r)) for lk, r in rows.items()}, source_sha256=h0, sha256=out_sha,
                       sh_degree={0: 0, 9: 1, 24: 2, 45: 3}[n_rest], verified=verified)
