#!/usr/bin/env python3
"""Inspect a saved SAM3 affinity sidecar and optionally export PCA HTML."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.sam3_gaussian_affinity import build_local_affinity_graph, _write_gaussian_html


def _load_xyz(ply_path: Path):
    ply = PlyData.read(str(ply_path))
    vertex = ply.elements[0]
    return torch.tensor(
        [[float(x), float(y), float(z)] for x, y, z in zip(vertex["x"], vertex["y"], vertex["z"])],
        dtype=torch.float32,
    )


def main():
    parser = argparse.ArgumentParser(description="Debug saved SAM3 affinity field")
    parser.add_argument("--point-cloud", required=True, help="Path to point_cloud.ply")
    parser.add_argument("--affinity-field", default="", help="Defaults to point cloud directory/sam3_affinity_field.pt")
    parser.add_argument("--output", default="", help="Defaults to point cloud directory/sam3_affinity_debug")
    parser.add_argument("--knn", type=int, default=16)
    parser.add_argument("--max-html-points", type=int, default=5000)
    args = parser.parse_args()

    ply_path = Path(args.point_cloud).expanduser().resolve()
    field_path = Path(args.affinity_field).expanduser().resolve() if args.affinity_field else ply_path.parent / "sam3_affinity_field.pt"
    output = Path(args.output).expanduser().resolve() if args.output else ply_path.parent / "sam3_affinity_debug"
    payload = torch.load(field_path, map_location="cpu", weights_only=False)
    features = payload.get("features", payload) if isinstance(payload, dict) else payload
    features = torch.as_tensor(features).float()
    xyz = _load_xyz(ply_path)
    if xyz.shape[0] != features.shape[0]:
        raise RuntimeError(f"PLY rows {xyz.shape[0]} != affinity rows {features.shape[0]}")
    output.mkdir(parents=True, exist_ok=True)
    graph = build_local_affinity_graph(xyz, features, int(args.knn))
    torch.save(graph, output / "sam3_affinity_graph.pt")
    _write_gaussian_html(output / "sam3_affinity_gaussians.html", xyz, features, int(args.max_html_points))
    summary = {
        "point_cloud": str(ply_path),
        "affinity_field": str(field_path),
        "gaussians": int(features.shape[0]),
        "dim": int(features.shape[1]),
        "feature_norm_mean": float(torch.linalg.norm(features, dim=-1).mean()),
        "graph_edges": int(graph["edge_index"].shape[1]),
        "graph_affinity_mean": float(graph["affinity"].mean()) if graph["affinity"].numel() else 0.0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
