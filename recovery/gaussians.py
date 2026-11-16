"""Canonical Gaussians with hard material labels and lineage (spec 2.2, 6, 7.3).

Row-aligned arrays (always updated together by recovery.topology):
  material      int8   0 = STATIC, 1 = MOVING            -- the physical assignment used for rendering
  gaussian_uid  int64  unique, persistent across densification / pruning
  parent_uid    int64  uid of the parent (-1 for an original or a Phase-4 birth)
  original_id   int64  row in the Phase-0 endpoint PLY (-1 for Gaussians that did not exist there)
  provenance    int8   PROV_* code below

Inference status (UNRESOLVED, ...) is stored separately by Phase 4 and never changes the transport.
PLY I/O follows the standard 3DGS layout (x y z nx ny nz f_dc_* f_rest_* opacity scale_* rot_*), readable by
scene.gaussian_model.GaussianModel.load_ply.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

STATIC, MOVING = 0, 1
PROV_ORIGINAL_CORE = 0          # Phase-2 core (moving seed)
PROV_ORIGINAL = 1               # other Phase-0 Gaussian, label from Phase 4A
PROV_PHASE4_BIRTH = 2           # Phase-4B/4C accepted patch (observed surface, back-transported)
PROV_DENSIFY_CLONE = 3
PROV_DENSIFY_SPLIT = 4
PROV_NAMES = {0: "original_core", 1: "original", 2: "phase4_patch", 3: "densify_clone", 4: "densify_split"}


class GaussianTensors:
    """Plain container with GaussianModel's attribute names (_xyz, _rotation, _scaling, _opacity,
    _features_dc [N,1,3], _features_rest [N,K,3]) plus lineage."""

    def __init__(self, xyz, rotation, scaling, opacity, features_dc, features_rest, max_sh_degree, lineage=None):
        self._xyz, self._rotation, self._scaling, self._opacity = xyz, rotation, scaling, opacity
        self._features_dc, self._features_rest = features_dc, features_rest
        self.max_sh_degree = int(max_sh_degree)
        self.active_sh_degree = int(max_sh_degree)
        n = xyz.shape[0]
        self.lineage = lineage or {
            "material": np.zeros(n, np.int8), "gaussian_uid": np.arange(n, dtype=np.int64),
            "parent_uid": np.full(n, -1, np.int64), "original_id": np.arange(n, dtype=np.int64),
            "provenance": np.full(n, PROV_ORIGINAL, np.int8)}
        self.check()

    def __len__(self):
        return int(self._xyz.shape[0])

    def check(self):
        n = len(self)
        for k in ("_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest"):
            if getattr(self, k).shape[0] != n:
                raise RuntimeError(f"{k} has {getattr(self, k).shape[0]} rows, xyz has {n}")
        for k, v in self.lineage.items():
            if len(v) != n:
                raise RuntimeError(f"lineage {k} has {len(v)} rows, xyz has {n}")
        if len(np.unique(self.lineage["gaussian_uid"])) != n:
            raise RuntimeError("gaussian_uid is not unique")
        if not np.isin(self.lineage["material"], (STATIC, MOVING)).all():
            raise RuntimeError("material must be STATIC (0) or MOVING (1)")

    @property
    def moving(self) -> torch.Tensor:
        return torch.as_tensor(self.lineage["material"] == MOVING, device=self._xyz.device)

    def to(self, device):
        for k in ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def subset(self, idx):
        idx_t = torch.as_tensor(idx, device=self._xyz.device)
        idx_n = np.asarray(torch.as_tensor(idx).cpu())
        g = GaussianTensors(self._xyz[idx_t], self._rotation[idx_t], self._scaling[idx_t], self._opacity[idx_t],
                            self._features_dc[idx_t], self._features_rest[idx_t], self.max_sh_degree,
                            {k: v[idx_n].copy() for k, v in self.lineage.items()})
        g.active_sh_degree = self.active_sh_degree
        return g

    @staticmethod
    def concat(a: "GaussianTensors", b: "GaussianTensors") -> "GaussianTensors":
        cat = lambda k: torch.cat([getattr(a, k), getattr(b, k).to(getattr(a, k).device)], 0)
        g = GaussianTensors(cat("_xyz"), cat("_rotation"), cat("_scaling"), cat("_opacity"), cat("_features_dc"),
                            cat("_features_rest"), a.max_sh_degree,
                            {k: np.concatenate([a.lineage[k], b.lineage[k]]) for k in a.lineage})
        g.active_sh_degree = a.active_sh_degree
        return g

    # ---------------------------------------------------------------- I/O
    @classmethod
    def load_ply(cls, path, device="cuda") -> "GaussianTensors":
        v = PlyData.read(str(path))["vertex"]
        names = [p.name for p in v.properties]
        xyz = np.stack([v["x"], v["y"], v["z"]], 1)
        dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)[:, None, :]
        rest_names = sorted([n for n in names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
        K = len(rest_names) // 3
        deg = int(round(np.sqrt(K + 1))) - 1
        if 3 * ((deg + 1) ** 2 - 1) != len(rest_names):
            raise ValueError(f"{path}: {len(rest_names)} f_rest fields is not a full SH degree")
        rest = np.stack([v[n] for n in rest_names], 1).reshape(len(xyz), 3, K).transpose(0, 2, 1) if K else \
            np.zeros((len(xyz), 0, 3))
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
        g = cls(t(xyz), t(np.stack([v[f"rot_{i}"] for i in range(4)], 1)), t(np.stack([v[f"scale_{i}"] for i in range(3)], 1)),
                t(np.asarray(v["opacity"])[:, None]), t(dc), t(rest), deg)
        return g

    def save_ply(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self)
        xyz = self._xyz.detach().cpu().numpy()
        dc = self._features_dc.detach().transpose(1, 2).flatten(1).cpu().numpy()
        rest = self._features_rest.detach().transpose(1, 2).flatten(1).cpu().numpy()
        attrs = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(dc.shape[1])] + \
                [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity"] + [f"scale_{i}" for i in range(3)] + \
                [f"rot_{i}" for i in range(4)]
        data = np.concatenate([xyz, np.zeros_like(xyz), dc, rest, self._opacity.detach().cpu().numpy(),
                               self._scaling.detach().cpu().numpy(), self._rotation.detach().cpu().numpy()], 1)
        el = np.empty(n, dtype=[(a, "f4") for a in attrs])
        el[:] = list(map(tuple, data.astype(np.float32)))
        PlyData([PlyElement.describe(el, "vertex")]).write(str(path))

    def save_lineage(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "material_ids.npz", material=self.lineage["material"],
                 legend=json.dumps({"0": "STATIC", "1": "MOVING"}))
        np.savez(d / "gaussian_uid.npz", gaussian_uid=self.lineage["gaussian_uid"],
                 original_id=self.lineage["original_id"])
        np.savez(d / "parent_uid.npz", parent_uid=self.lineage["parent_uid"])
        np.savez(d / "provenance.npz", provenance=self.lineage["provenance"], legend=json.dumps(PROV_NAMES))

    @staticmethod
    def load_lineage(directory) -> dict:
        d = Path(directory)
        out = {"material": np.load(d / "material_ids.npz")["material"]}
        u = np.load(d / "gaussian_uid.npz")
        out["gaussian_uid"], out["original_id"] = u["gaussian_uid"], u["original_id"]
        out["parent_uid"] = np.load(d / "parent_uid.npz")["parent_uid"]
        out["provenance"] = np.load(d / "provenance.npz")["provenance"]
        return out
