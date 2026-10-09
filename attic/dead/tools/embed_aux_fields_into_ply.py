#!/usr/bin/env python3
"""Embed saved auxiliary Gaussian sidecars into an existing PLY."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _load_sidecar(path: Path):
    if not path.is_file():
        return None
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value.get("features", value.get(path.stem))
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().cpu().float()
    if tensor.ndim == 1:
        tensor = tensor[:, None]
    return tensor if tensor.ndim == 2 and tensor.numel() > 0 else None


def _append_fields(vertex, fields: dict[str, torch.Tensor]):
    old_names = vertex.data.dtype.names or ()
    new_specs = [(name, vertex.data.dtype[name]) for name in old_names]
    arrays: dict[str, np.ndarray] = {name: vertex.data[name] for name in old_names}
    n = len(vertex.data)
    for prefix, tensor in fields.items():
        if tensor is None or int(tensor.shape[0]) != n:
            continue
        for channel in range(int(tensor.shape[1])):
            name = f"{prefix}_{channel}"
            if name in arrays:
                continue
            new_specs.append((name, "f4"))
            arrays[name] = tensor[:, channel].numpy().astype(np.float32, copy=False)
    out = np.empty(n, dtype=new_specs)
    for name in out.dtype.names or ():
        out[name] = arrays[name]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply")
    parser.add_argument("--sam3-feature", default="")
    parser.add_argument("--sam3-affinity", default="")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    ply = Path(args.ply).expanduser().resolve()
    root = ply.parent
    feature_path = Path(args.sam3_feature).expanduser() if args.sam3_feature else root / "sam3_feature_field.pt"
    affinity_path = Path(args.sam3_affinity).expanduser() if args.sam3_affinity else root / "sam3_affinity_field.pt"

    data = PlyData.read(str(ply))
    vertex = data["vertex"]
    fields = {
        "sam3_feature": _load_sidecar(feature_path),
        "sam3_affinity": _load_sidecar(affinity_path),
    }
    before = set(vertex.data.dtype.names or ())
    new_vertex = _append_fields(vertex, fields)
    added = [name for name in (new_vertex.dtype.names or ()) if name not in before]
    if not added:
        print(f"[embed_aux_fields_into_ply] no fields added: {ply}")
        return
    if not args.no_backup:
        backup = ply.with_suffix(ply.suffix + ".before_aux_embed")
        if not backup.exists():
            shutil.copy2(ply, backup)
            print(f"[embed_aux_fields_into_ply] backup: {backup}")
    PlyData([PlyElement.describe(new_vertex, "vertex")], text=data.text).write(str(ply))
    print(f"[embed_aux_fields_into_ply] added {len(added)} fields to {ply}: {added[:4]}{' ...' if len(added) > 4 else ''}")


if __name__ == "__main__":
    main()
