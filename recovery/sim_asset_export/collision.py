"""Collision geometry ingest / validation (spec 5). numpy + scipy only; no geometry is invented here.

Supported sources (declared in the physics spec, never chosen silently):
  * a supplied surface mesh (OBJ) in capture coordinates + its rigid T_capture_mesh + explicit units
  * optionally a supplied convex decomposition (OBJ parts) of that surface -- what the engines collide with
  * opt-in approximation `convex_hull`: ONE convex hull of the supplied surface (marked estimated, never measured)
  * `reconstructed_rgbd` is accepted only with declared end-effector exclusion evidence; the reconstruction
    algorithm itself is not implemented here (COLLISION_UNRESOLVED otherwise)
3DGS Gaussians are never used as collision bodies.

Mass properties of a closed, outward-oriented triangle mesh are the exact polyhedral integrals (divergence theorem;
D. Eberly, "Polyhedral Mass Properties (Revisited)", 2002).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001}
CONVEX_VOLUME_RTOL = 1e-6          # |V_hull - V_mesh| / V_hull for "this mesh is convex" (numerical, not a tolerance on
                                   # the physics: an exactly convex mesh gives ~1e-12)


class CollisionError(ValueError):
    pass


@dataclass
class Mesh:
    vertices: np.ndarray      # [V,3] float64
    faces: np.ndarray         # [F,3] int64, 0-based, counter-clockwise seen from outside

    def transformed(self, T) -> "Mesh":
        T = np.asarray(T, np.float64)
        return Mesh(self.vertices @ T[:3, :3].T + T[:3, 3], self.faces.copy())

    def scaled(self, s: float) -> "Mesh":
        return Mesh(self.vertices * float(s), self.faces.copy())


def read_obj(path) -> Mesh:
    path = Path(path)
    if not path.is_file():
        raise CollisionError(f"{path}: mesh file not found")
    vs, fs = [], []
    for ln, line in enumerate(path.read_text(errors="strict").splitlines(), 1):
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        if t[0] == "v":
            if len(t) < 4:
                raise CollisionError(f"{path}:{ln}: vertex needs 3 coordinates")
            vs.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            idx = []
            for tok in t[1:]:
                i = int(tok.split("/")[0])
                idx.append(i - 1 if i > 0 else len(vs) + i)
            if len(idx) < 3:
                raise CollisionError(f"{path}:{ln}: face with {len(idx)} vertices")
            for k in range(1, len(idx) - 1):          # fan triangulation of convex polygons
                fs.append([idx[0], idx[k], idx[k + 1]])
    if not vs or not fs:
        raise CollisionError(f"{path}: empty mesh ({len(vs)} vertices, {len(fs)} faces)")
    m = Mesh(np.asarray(vs, np.float64), np.asarray(fs, np.int64))
    if (m.faces < 0).any() or (m.faces >= len(m.vertices)).any():
        raise CollisionError(f"{path}: face indices out of range")
    if not np.isfinite(m.vertices).all():
        raise CollisionError(f"{path}: non-finite vertex coordinates")
    return m


def write_obj(mesh: Mesh, path, header=()) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {h}" for h in header]
    lines += [f"v {x!r} {y!r} {z!r}" for x, y, z in mesh.vertices.tolist()]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces.tolist()]
    path.write_text("\n".join(lines) + "\n")


def topology(m: Mesh) -> dict:
    """Edge-manifold / orientation audit. Watertight = every directed edge appears once and its reverse once."""
    f = m.faces
    tri = m.vertices[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    nv = len(m.vertices)
    key = e[:, 0] * nv + e[:, 1]
    rkey = e[:, 1] * nv + e[:, 0]
    uk, cnt = np.unique(key, return_counts=True)
    dup_directed = int((cnt > 1).sum())
    has_rev = np.isin(rkey, uk)
    und = np.unique(np.sort(e, 1)[:, 0] * nv + np.sort(e, 1)[:, 1])
    used = np.unique(f)
    return {"vertices": nv, "faces": int(len(f)), "degenerate_faces": int((area <= 0).sum()),
            "unused_vertices": int(nv - len(used)), "directed_edges_duplicated": dup_directed,
            "edges_without_opposite": int((~has_rev).sum()), "undirected_edges": int(len(und)),
            "watertight": bool(dup_directed == 0 and has_rev.all() and (area > 0).all()),
            "surface_area": float(area.sum())}


def mass_properties(m: Mesh, density: float = 1.0) -> dict:
    """Exact volume, COM and inertia about the COM (density x geometry) of a closed outward triangle mesh."""
    v = m.vertices[m.faces]                      # [F,3,3]
    x0, x1, x2 = v[:, 0], v[:, 1], v[:, 2]
    d = np.cross(x1 - x0, x2 - x0)               # 2 x area-weighted outward normal
    f1 = x0 + x1 + x2
    f2 = x0 * x0 + x1 * (x0 + x1) + x2 * f1
    f3 = x0 * x0 * x0 + x1 * (x0 * x0 + x0 * x1 + x1 * x1) + x2 * f2
    g0 = f2 + x0 * (f1 + x0)
    g1 = f2 + x1 * (f1 + x1)
    g2 = f2 + x2 * (f1 + x2)
    integ = np.zeros(10)
    integ[0] = (d[:, 0] * f1[:, 0]).sum()
    integ[1:4] = (d * f2).sum(0)
    integ[4:7] = (d * f3).sum(0)
    X, Y, Z = 0, 1, 2
    integ[7] = (d[:, X] * (x0[:, Y] * g0[:, X] + x1[:, Y] * g1[:, X] + x2[:, Y] * g2[:, X])).sum()
    integ[8] = (d[:, Y] * (x0[:, Z] * g0[:, Y] + x1[:, Z] * g1[:, Y] + x2[:, Z] * g2[:, Y])).sum()
    integ[9] = (d[:, Z] * (x0[:, X] * g0[:, Z] + x1[:, X] * g1[:, Z] + x2[:, X] * g2[:, Z])).sum()
    integ *= np.array([1 / 6, 1 / 24, 1 / 24, 1 / 24, 1 / 60, 1 / 60, 1 / 60, 1 / 120, 1 / 120, 1 / 120])
    vol = integ[0]
    if not vol > 0:
        raise CollisionError(f"mesh volume {vol:.6g} <= 0 (open, inside-out or degenerate surface)")
    com = integ[1:4] / vol
    xx = integ[5] + integ[6] - vol * (com[1] ** 2 + com[2] ** 2)
    yy = integ[4] + integ[6] - vol * (com[2] ** 2 + com[0] ** 2)
    zz = integ[4] + integ[5] - vol * (com[0] ** 2 + com[1] ** 2)
    xy = -(integ[7] - vol * com[0] * com[1])
    yz = -(integ[8] - vol * com[1] * com[2])
    xz = -(integ[9] - vol * com[2] * com[0])
    I = density * np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return {"volume": float(vol), "mass": float(density * vol), "com": com, "inertia_com": I}


def convex_hull(m: Mesh) -> Mesh:
    from scipy.spatial import ConvexHull
    h = ConvexHull(m.vertices)
    used = np.unique(h.simplices)
    remap = -np.ones(len(m.vertices), np.int64)
    remap[used] = np.arange(len(used))
    faces = remap[h.simplices]
    verts = m.vertices[used]
    # orient every facet outward (scipy's simplices are unordered)
    c = verts.mean(0)
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = ((tri[:, 0] - c) * n).sum(1) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return Mesh(verts, faces)


def is_convex(m: Mesh) -> tuple[bool, dict]:
    t = topology(m)
    if not t["watertight"]:
        return False, {"reason": "not watertight", **t}
    v_m = mass_properties(m)["volume"]
    v_h = mass_properties(convex_hull(m))["volume"]
    rel = abs(v_h - v_m) / v_h
    return bool(rel <= CONVEX_VOLUME_RTOL), {"mesh_volume": v_m, "hull_volume": v_h, "relative_gap": rel,
                                             "rtol": CONVEX_VOLUME_RTOL}


def distance_to_hulls(points: np.ndarray, hulls: list) -> np.ndarray:
    """Distance from each point to the union of convex parts (0 inside), via the facet planes of each hull
    (max over planes of signed distance is exact inside, a lower bound of the Euclidean distance outside)."""
    best = np.full(len(points), np.inf)
    for h in hulls:
        tri = h.vertices[h.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        off = (n * tri[:, 0]).sum(1)
        sd = points @ n.T - off[None]
        best = np.minimum(best, np.maximum(sd.max(1), 0.0))
    return best


@dataclass
class LinkCollision:
    link: str
    source_kind: str                 # supplied_mesh | reconstructed_rgbd
    provenance: str                  # measured | estimated
    approximation: str               # source_is_convex | supplied_convex_decomposition | convex_hull
    surface: Mesh                    # supplied surface, in the LINK frame
    parts: list                      # convex proxies, in the LINK frame
    report: dict = field(default_factory=dict)


def gaussian_extent(xyz: np.ndarray, lo=2.0, hi=98.0) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(xyz, lo, axis=0), np.percentile(xyz, hi, axis=0)
