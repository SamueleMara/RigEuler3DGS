"""Resolve a validated PhysicsSpec into link-frame quantities (spec 5): collision surfaces + convex proxies, mass /
COM / inertia, joint stops, contact, registration. Uses only the frozen Phase-3 joint to bring a moving-part input
posed at original frame f back to the anchor (x0 = inv(T_f) x_f); nothing is fitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .collision import (CollisionError, LinkCollision, Mesh, convex_hull, distance_to_hulls, gaussian_extent,
                        is_convex, mass_properties, topology)
from .frames import inv_se3, principal_axes, se3, transform_inertial
from .schema import SIDE, UNVERIFIED_GAUGE, PhysicsSpec, PhysicsSpecError, Registration


@dataclass
class LinkInertial:
    link: str
    mass: float
    com: np.ndarray                 # link frame
    inertia: np.ndarray             # about the COM, link-frame axes
    provenance: str
    principal_moments: np.ndarray
    principal_axes: np.ndarray      # columns, link frame, det +1
    report: dict = field(default_factory=dict)


@dataclass
class JointLimits:
    lower: float
    upper: float
    velocity: float
    effort: float
    damping: float
    coulomb_friction: float
    armature: float | None
    limits_provenance: str          # physical_stops_<provenance> | observed_range_preview
    provenance: str
    observed_range: tuple
    report: dict = field(default_factory=dict)


@dataclass
class ResolvedPhysics:
    registration: Registration
    collision: dict                 # link -> LinkCollision
    inertial: dict                  # link -> LinkInertial
    joint: JointLimits
    contact: dict | None
    assumptions: list               # opt-in assumptions that keep the asset from FULL_PHYSICS_READY
    provenance_summary: dict
    warnings: list
    spec_path: str


def joint_T(joint, frame: int) -> np.ndarray:
    if int(frame) not in joint.frames:
        raise PhysicsSpecError(f"at_original_frame {frame} is not a Phase-3 frame [{min(joint.frames)}, "
                               f"{max(joint.frames)}]")
    Q, p = joint.transform(int(frame))
    return se3(Q, p)


def T_link_capture(link, joint, T_B_J0, frame):
    """Link frame <- capture world (at original frame `frame` for the moving link)."""
    if link == "static_link":
        return np.eye(4)
    return inv_se3(T_B_J0) @ inv_se3(joint_T(joint, frame))


def _plausibility(mesh_L: Mesh, xyz_L: np.ndarray, val: dict, where: str) -> dict:
    lo, hi = gaussian_extent(xyz_L)
    g_diag = float(np.linalg.norm(hi - lo))
    m_lo, m_hi = mesh_L.vertices.min(0), mesh_L.vertices.max(0)
    m_diag = float(np.linalg.norm(m_hi - m_lo))
    ratio = m_diag / g_diag if g_diag > 0 else float("inf")
    off = float(np.linalg.norm(0.5 * (m_lo + m_hi) - 0.5 * (lo + hi)))
    rep = {"mesh_aabb_diag_m": m_diag, "gaussian_p2_p98_aabb_diag_m": g_diag, "extent_ratio": ratio,
           "extent_ratio_range": list(val["extent_ratio_range"]), "aabb_centre_offset_m": off,
           "centroid_offset_max_fraction": val["centroid_offset_max"]}
    r0, r1 = val["extent_ratio_range"]
    if not r0 <= ratio <= r1:
        raise PhysicsSpecError(f"{where}: collision mesh extent is {ratio:.3g} x the link's Gaussian extent (allowed "
                               f"[{r0}, {r1}]) -- wrong `units`, wrong mesh, or a scale inside T_capture_mesh")
    if off > val["centroid_offset_max"] * g_diag:
        raise PhysicsSpecError(f"{where}: collision mesh centre is {off:.3g} m from the link's Gaussians "
                               f"({off / g_diag:.2f} x their extent; allowed {val['centroid_offset_max']}) -- mismatched "
                               f"collision link frames (swapped links, wrong T_capture_mesh or wrong at_original_frame)")
    return rep


def _containment(parts, xyz_L: np.ndarray, val: dict, where: str) -> dict:
    """A correct collision geometry contains (or touches) most of its own link's Gaussian means."""
    lo, hi = gaussian_extent(xyz_L)
    tol = val["gaussian_inside_tolerance_fraction"] * float(np.linalg.norm(hi - lo))
    d = distance_to_hulls(xyz_L, parts)
    frac = float((d <= tol).mean())
    rep = {"gaussians": int(len(xyz_L)), "tolerance_m": tol, "fraction_within_tolerance": frac,
           "min_fraction": val["gaussian_inside_min_fraction"], "median_distance_m": float(np.median(d))}
    if frac < val["gaussian_inside_min_fraction"]:
        raise PhysicsSpecError(f"{where}: only {frac:.1%} of the link's Gaussian means lie within {tol:.3g} m of its "
                               f"collision geometry (need {val['gaussian_inside_min_fraction']:.0%}) -- mismatched "
                               f"collision link frames (swapped links, wrong T_capture_mesh or wrong at_original_frame)")
    return rep


def resolve_physics(spec: PhysicsSpec, inp, T_B_J0: np.ndarray, registration: Registration | None = None
                    ) -> ResolvedPhysics:
    joint = inp.joint
    assumptions, warnings = [], []
    reg = registration or spec.registration or UNVERIFIED_GAUGE
    if reg.gravity_alignment == "unverified":
        assumptions.append("gravity alignment unverified (gauge W = Wc)")
    xyz_L = {"static_link": inp.xyz[inp.part_id == 0], "moving_link": (inv_se3(T_B_J0)[:3, :3] @
                                                                     inp.xyz[inp.part_id == 1].T).T
                                                                    + inv_se3(T_B_J0)[:3, 3]}
    collisions = {}
    for side, cs in spec.collision.items():
        lk = SIDE[side]
        where = f"collision.{side}"
        T = T_link_capture(lk, joint, T_B_J0, cs.at_original_frame)
        surf = cs.mesh.transformed(T)
        topo = topology(surf)
        rep = {"source_file": cs.source.name, "source_kind": cs.source_kind, "units": cs.units,
               "unit_scale_to_m": {"m": 1.0, "cm": 0.01, "mm": 0.001}[cs.units],
               "T_capture_mesh": cs.T_capture_mesh.tolist(), "at_original_frame": cs.at_original_frame,
               "T_link_capture": T.tolist(), "surface_topology": topo, "provenance": cs.provenance,
               "plausibility": _plausibility(surf, xyz_L[lk], spec.validation, where)}
        if topo["degenerate_faces"]:
            warnings.append(f"{where}: {topo['degenerate_faces']} zero-area faces in the supplied surface")
        try:
            if cs.part_meshes:
                parts = [p.transformed(T) for p in cs.part_meshes]
                prep = []
                for i, p in enumerate(parts):
                    ok, info = is_convex(p)
                    if not ok:
                        raise PhysicsSpecError(f"{where}.convex_parts[{i}] ({cs.convex_parts[i].name}) is not a closed "
                                               f"convex mesh: {info}")
                    prep.append({"file": cs.convex_parts[i].name, **info})
                d = distance_to_hulls(surf.vertices, parts)
                cov = {"max_surface_vertex_distance_m": float(d.max()), "tolerance_m": cs.coverage_tolerance_m,
                       "fraction_within_tolerance": float((d <= cs.coverage_tolerance_m).mean())}
                if d.max() > cs.coverage_tolerance_m:
                    raise PhysicsSpecError(f"{where}: convex parts leave surface vertices {d.max():.4g} m outside "
                                           f"their union (tolerance {cs.coverage_tolerance_m} m)")
                appr = "supplied_convex_decomposition"
                rep.update(convex_parts=prep, coverage=cov)
            elif cs.approximation == "convex_hull":
                parts, appr = [convex_hull(surf)], "convex_hull"
                assumptions.append(f"{lk}: collision = ONE convex hull of the supplied surface (opt-in approximation)")
                rep["hull"] = {"volume": mass_properties(parts[0])["volume"]}
            else:
                ok, info = is_convex(surf)
                if not ok:
                    raise PhysicsSpecError(f"{where}: the supplied surface is not convex ({info.get('reason') or info}); "
                                           f"MuJoCo / PhysX collide with convex geometry -- supply convex_parts (a "
                                           f"validated decomposition) or opt in `approximation: convex_hull`")
                parts, appr = [surf], "source_is_convex"
                rep["convexity"] = info
        except CollisionError as e:
            raise PhysicsSpecError(f"{where}: {e}") from None
        if cs.source_kind == "reconstructed_rgbd":
            assumptions.append(f"{lk}: collision surface reconstructed from RGB-D (ee exclusion: "
                               f"{cs.ee_exclusion.get('method')})")
        rep["containment"] = _containment(parts, xyz_L[lk], spec.validation, where)
        rep.update(approximation=appr, hull_count=len(parts),
                   proxy_volumes=[mass_properties(p)["volume"] for p in parts])
        collisions[lk] = LinkCollision(lk, cs.source_kind, "estimated" if appr == "convex_hull" else cs.provenance,
                                       appr, surf, parts, rep)

    inertials = {}
    for lk, s in spec.inertial.items():
        where = f"links.{lk}"
        if s.density_kg_m3 is not None:
            surf = collisions[lk].surface
            t = topology(surf)
            if not t["watertight"]:
                raise PhysicsSpecError(f"{where}: density_kg_m3 needs a watertight collision surface "
                                       f"({t['edges_without_opposite']} open edges)")
            mp = mass_properties(surf, s.density_kg_m3)
            m, com, I = mp["mass"], mp["com"], mp["inertia_com"]
            rep = {"method": "uniform density x exact polyhedral integrals of the supplied surface",
                   "density_kg_m3": s.density_kg_m3, "volume_m3": mp["volume"]}
        else:
            T = T_link_capture(lk, joint, T_B_J0, s.at_original_frame)
            m, com, I = transform_inertial(T, s.mass_kg, s.com_capture_m, s.inertia_com_capture_kg_m2)
            rep = {"method": "supplied; rotated capture -> link frame (I_L = R I R^T)", "at_original_frame":
                   s.at_original_frame, "T_link_capture": T.tolist()}
        I = 0.5 * (I + I.T)              # R I R^T is symmetric only up to rounding; every format gets this tensor
        d, V = principal_axes(I)
        lo = np.min([p.vertices.min(0) for p in collisions[lk].parts], 0)
        hi = np.max([p.vertices.max(0) for p in collisions[lk].parts], 0)
        inside = bool(((com >= lo) & (com <= hi)).all())
        rep.update(principal_moments=d.tolist(), com_inside_collision_aabb=inside)
        if not inside:
            warnings.append(f"{where}: COM {com.tolist()} lies outside the collision AABB")
        inertials[lk] = LinkInertial(lk, float(m), com, I, s.provenance, d, V, rep)

    js = spec.joint
    fr, q = inp.trajectory["frame"], inp.trajectory["q"]
    obs = (float(q.min()), float(q.max()))
    if js.limits_from_observed_range:
        lo, hi, lprov = obs[0], obs[1], "observed_range_preview"
        assumptions.append("joint limits = observed demonstration range (opt-in preview, NOT mechanical stops)")
        if not lo < hi:
            raise PhysicsSpecError("observed q range is empty; cannot form preview limits")
    else:
        lo, hi, lprov = js.lower, js.upper, f"physical_stops_{js.provenance}"
    viol = max(0.0, lo - obs[0], obs[1] - hi)
    jrep = {"observed_range": list(obs), "observed_exceeds_stops_by": viol}
    if viol > 0 and not js.limits_from_observed_range:
        warnings.append(f"joint: the demonstration q range {obs} exceeds the declared stops [{lo}, {hi}] by {viol:.3g} "
                        f"({'rad' if joint.type == 'revolute' else 'm'})")
    jl = JointLimits(lo, hi, js.velocity_limit, js.effort_limit, js.damping, js.coulomb_friction, js.armature, lprov,
                     js.provenance, obs, jrep)
    contact = None if spec.contact is None else {"static_friction": spec.contact.static_friction,
                                                 "dynamic_friction": spec.contact.dynamic_friction,
                                                 "restitution": spec.contact.restitution,
                                                 "provenance": spec.contact.provenance}
    prov = {"collision": {lk: c.provenance for lk, c in collisions.items()},
            "inertial": {lk: i.provenance for lk, i in inertials.items()},
            "joint_limits": lprov, "joint_dynamics": js.provenance,
            "contact": None if contact is None else contact["provenance"],
            "registration": reg.gauge, "gravity_alignment": reg.gravity_alignment}
    prov["all_measured"] = bool(all(v == "measured" for v in prov["collision"].values()) and
                                all(v == "measured" for v in prov["inertial"].values()) and
                                lprov == "physical_stops_measured" and js.provenance == "measured" and
                                reg.gravity_alignment == "measured")
    return ResolvedPhysics(reg, collisions, inertials, jl, contact, assumptions, prov, warnings, str(spec.path))
