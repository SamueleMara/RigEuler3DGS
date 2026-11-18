"""Validated, typed physics input + the canonical asset model (spec 5). schema_version 1.

Nothing here invents a physical value. Every field is either supplied (with provenance) or derived by a declared,
opt-in rule whose provenance says so (density_assumption, observed_range_preview, convex_hull). Unknown keys are
errors (typos must not silently drop physics). See docs/PHASE5_FINAL_SIM_ASSET_FORMAT.md for the full schema.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .collision import UNITS, CollisionError, Mesh, read_obj
from .frames import FrameError, check_se3

SCHEMA_VERSION = 1
LINK_NAMES = ("static_link", "moving_link")
SIDE = {"static": "static_link", "moving": "moving_link"}
JOINT_NAME = "articulation_joint"

# overall export status
FULL_PHYSICS_READY = "FULL_PHYSICS_READY"
PHYSICS_PREVIEW_ASSUMPTIONS = "PHYSICS_PREVIEW_ASSUMPTIONS"
PHYSICS_INCOMPLETE_RUNTIME = "PHYSICS_INCOMPLETE_RUNTIME"
BLOCKED_MISSING_PHYSICS = "BLOCKED_MISSING_PHYSICS"
PHYSICS_REJECTED = "PHYSICS_REJECTED"
VISUAL_ONLY = "VISUAL_ONLY"
STATUSES = (FULL_PHYSICS_READY, PHYSICS_PREVIEW_ASSUMPTIONS, PHYSICS_INCOMPLETE_RUNTIME, BLOCKED_MISSING_PHYSICS,
            PHYSICS_REJECTED, VISUAL_ONLY)

PROVENANCE_MEASURED = ("measured", "estimated")
INERTIAL_PROVENANCE = ("measured", "estimated", "density_assumption")
GRAVITY = ("measured", "declared", "unverified")
INERTIA_SYM_RTOL = 1e-9

# plausibility gates between a supplied collision mesh and the link's own Gaussians (declared defaults, overridable
# in the spec's `validation:` block): catches mm meshes declared as m, swapped links, a moving mesh posed at the
# wrong frame. They do not measure anything.
DEFAULT_VALIDATION = {"extent_ratio_range": [0.2, 5.0], "centroid_offset_max": 1.0,
                      "gaussian_inside_tolerance_fraction": 0.05, "gaussian_inside_min_fraction": 0.5}


class PhysicsSpecError(ValueError):
    """Actionable error in the physics specification."""


def _keys(d, allowed, where, required=()):
    if not isinstance(d, dict):
        raise PhysicsSpecError(f"{where}: expected a mapping, got {type(d).__name__}")
    bad = sorted(set(d) - set(allowed))
    if bad:
        raise PhysicsSpecError(f"{where}: unknown keys {bad}; allowed {sorted(allowed)}")
    miss = [k for k in required if d.get(k) is None]
    if miss:
        raise PhysicsSpecError(f"{where}: required {miss} (null is not a value)")


def _num(v, where, lo=None, lo_open=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise PhysicsSpecError(f"{where}: expected a number, got {v!r}")
    v = float(v)
    if not math.isfinite(v):
        raise PhysicsSpecError(f"{where}: not finite ({v})")
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise PhysicsSpecError(f"{where}: must be {'>' if lo_open else '>='} {lo}, got {v}")
    return v


def _vec(v, where, n=3):
    a = np.asarray(v, dtype=object)
    if a.shape != (n,):
        raise PhysicsSpecError(f"{where}: expected {n} numbers, got {v!r}")
    return np.array([_num(x, f"{where}[{i}]") for i, x in enumerate(a)])


def _mat(v, where, shape):
    try:
        a = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError):
        raise PhysicsSpecError(f"{where}: expected a {shape[0]}x{shape[1]} numeric matrix") from None
    if a.shape != shape or not np.isfinite(a).all():
        raise PhysicsSpecError(f"{where}: expected a finite {shape[0]}x{shape[1]} matrix, got shape {a.shape}")
    return a


def _se3(v, where):
    try:
        return check_se3(_mat(v, where, (4, 4)), where)
    except FrameError as e:
        raise PhysicsSpecError(str(e)) from None


def _choice(v, where, allowed):
    if v not in allowed:
        raise PhysicsSpecError(f"{where}: {v!r} not in {list(allowed)}")
    return v


def _path(base: Path, v, where) -> Path:
    if not isinstance(v, str) or not v:
        raise PhysicsSpecError(f"{where}: expected a file path")
    p = Path(v).expanduser()
    p = p if p.is_absolute() else (base / p)
    if not p.is_file():
        raise PhysicsSpecError(f"{where}: file not found: {p}")
    return p.resolve()


def _frame(v, where, required: bool):
    if v is None:
        if required:
            raise PhysicsSpecError(f"{where}: required for the moving link -- the original frame id at which the "
                                   f"moving-part geometry / inertia was posed (usually the anchor frame)")
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise PhysicsSpecError(f"{where}: expected an integer original frame id, got {v!r}")
    return int(v)


# ------------------------------------------------------------------------------------------------ registration
@dataclass(frozen=True)
class Registration:
    T_world_capture: np.ndarray       # T_W_Wc (= T_W_B)
    gauge: str                        # "measured_registration" | "capture_world"
    gravity_alignment: str            # measured | declared | unverified
    provenance: str

    def to_dict(self):
        return {"T_world_capture": self.T_world_capture.tolist(), "gauge": self.gauge,
                "gravity_alignment": self.gravity_alignment, "provenance": self.provenance}


UNVERIFIED_GAUGE = Registration(np.eye(4), "capture_world", "unverified",
                                "no registration supplied: gauge W = Wc selected; gravity alignment UNVERIFIED "
                                "(COLMAP / capture axes are arbitrary)")


def parse_registration(d, where="registration") -> Registration:
    _keys(d, ("T_world_capture", "gauge", "gravity_alignment", "provenance"), where, required=("gravity_alignment",))
    g = _choice(d["gravity_alignment"], f"{where}.gravity_alignment", GRAVITY)
    prov = str(d.get("provenance") or "")
    if d.get("T_world_capture") is None:
        if d.get("gauge") != "capture_world":
            raise PhysicsSpecError(f"{where}: T_world_capture is null -- set `gauge: capture_world` to select W = Wc "
                                   f"explicitly, or give a measured 4x4 SE(3)")
        return Registration(np.eye(4), "capture_world", g, prov)
    if d.get("gauge") not in (None, "measured_registration"):
        raise PhysicsSpecError(f"{where}.gauge: with a T_world_capture matrix the gauge is measured_registration")
    if g == "unverified":
        raise PhysicsSpecError(f"{where}: a supplied T_world_capture must state gravity_alignment measured or declared")
    return Registration(_se3(d["T_world_capture"], f"{where}.T_world_capture"), "measured_registration", g, prov)


# ------------------------------------------------------------------------------------------------ physics spec
@dataclass
class CollisionSpec:
    side: str
    source_kind: str
    source: Path | None
    units: str
    T_capture_mesh: np.ndarray
    at_original_frame: int | None
    provenance: str
    convex_parts: list
    coverage_tolerance_m: float | None
    approximation: str | None
    ee_exclusion: dict | None
    mesh: Mesh | None = None               # capture coordinates, metres
    part_meshes: list = field(default_factory=list)


@dataclass
class InertialSpec:
    link: str
    provenance: str
    at_original_frame: int | None
    mass_kg: float | None = None
    com_capture_m: np.ndarray | None = None
    inertia_com_capture_kg_m2: np.ndarray | None = None
    density_kg_m3: float | None = None


@dataclass
class JointPhysicsSpec:
    lower: float | None
    upper: float | None
    velocity_limit: float
    effort_limit: float
    damping: float
    coulomb_friction: float
    armature: float | None
    actuation: None
    limits_from_observed_range: bool
    provenance: str


@dataclass
class ContactSpec:
    static_friction: float
    dynamic_friction: float
    restitution: float
    provenance: str


@dataclass
class PhysicsSpec:
    path: Path
    registration: Registration | None
    collision: dict                 # side -> CollisionSpec
    inertial: dict                  # link -> InertialSpec
    joint: JointPhysicsSpec
    contact: ContactSpec | None
    validation: dict
    raw: dict


def check_inertia(I, where) -> np.ndarray:
    I = np.asarray(I, np.float64)
    s = np.abs(I).max()
    if s == 0 or np.abs(I - I.T).max() > INERTIA_SYM_RTOL * s:
        raise PhysicsSpecError(f"{where}: inertia must be a non-zero symmetric 3x3 (|I - I^T| = "
                               f"{np.abs(I - I.T).max():.3g})")
    ev = np.linalg.eigvalsh(0.5 * (I + I.T))
    if ev.min() <= 0:
        raise PhysicsSpecError(f"{where}: inertia is not positive definite (eigenvalues {ev.tolist()})")
    a, b, c = ev
    if c > (a + b) * (1 + 1e-9):
        raise PhysicsSpecError(f"{where}: principal moments {ev.tolist()} violate the triangle inequality "
                               f"I3 <= I1 + I2 (no physical mass distribution has them)")
    return 0.5 * (I + I.T)


def _parse_collision(d, side, base) -> CollisionSpec:
    where = f"collision.{side}"
    _keys(d, ("source_kind", "source", "units", "T_capture_mesh", "at_original_frame", "provenance", "convex_parts",
              "convex_coverage_tolerance_m", "approximation", "ee_exclusion"), where,
          required=("source", "units", "T_capture_mesh", "provenance"))
    kind = _choice(d.get("source_kind", "supplied_mesh"), f"{where}.source_kind", ("supplied_mesh", "reconstructed_rgbd"))
    ee = d.get("ee_exclusion")
    if kind == "reconstructed_rgbd":
        if not isinstance(ee, dict) or not ee.get("method") or not ee.get("evidence"):
            raise PhysicsSpecError(f"{where}: COLLISION_UNRESOLVED -- a surface reconstructed from RGB-D needs declared "
                                   f"end-effector exclusion (ee_exclusion: {{method, evidence}}); a binary object mask "
                                   f"cannot separate the gripper from the object")
    elif ee is not None:
        raise PhysicsSpecError(f"{where}.ee_exclusion only applies to source_kind reconstructed_rgbd")
    units = _choice(d["units"], f"{where}.units", tuple(UNITS))
    prov = _choice(d["provenance"], f"{where}.provenance", PROVENANCE_MEASURED)
    if kind == "reconstructed_rgbd" and prov == "measured":
        raise PhysicsSpecError(f"{where}: a reconstructed surface is provenance `estimated`, not `measured`")
    src = _path(base, d["source"], f"{where}.source")
    parts = [_path(base, p, f"{where}.convex_parts[{i}]") for i, p in enumerate(d.get("convex_parts") or [])]
    tol = d.get("convex_coverage_tolerance_m")
    if parts and tol is None:
        raise PhysicsSpecError(f"{where}: convex_parts need convex_coverage_tolerance_m (max allowed distance of the "
                               f"surface vertices to the union of parts, metres)")
    tol = None if tol is None else _num(tol, f"{where}.convex_coverage_tolerance_m", 0.0)
    appr = d.get("approximation")
    if appr is not None:
        _choice(appr, f"{where}.approximation", ("convex_hull",))
        if parts:
            raise PhysicsSpecError(f"{where}: give convex_parts OR approximation: convex_hull, not both")
    T = _se3(d["T_capture_mesh"], f"{where}.T_capture_mesh")
    fr = _frame(d.get("at_original_frame"), f"{where}.at_original_frame", side == "moving")
    if side == "static" and fr is not None:
        raise PhysicsSpecError(f"{where}.at_original_frame: the static link never moves; omit it")
    try:
        s = UNITS[units]
        mesh = read_obj(src).scaled(s).transformed(T)
        pms = [read_obj(p).scaled(s).transformed(T) for p in parts]
    except CollisionError as e:
        raise PhysicsSpecError(f"{where}: {e}") from None
    return CollisionSpec(side, kind, src, units, T, fr, prov, parts, tol, appr, ee, mesh, pms)


def _parse_inertial(d, link) -> InertialSpec:
    where = f"links.{link}"
    _keys(d, ("mass_kg", "com_capture_m", "inertia_com_capture_kg_m2", "density_kg_m3", "provenance",
              "at_original_frame"), where, required=("provenance",))
    prov = _choice(d["provenance"], f"{where}.provenance", INERTIAL_PROVENANCE)
    dens = d.get("density_kg_m3") is not None
    fr = _frame(d.get("at_original_frame"), f"{where}.at_original_frame", link == "moving_link" and not dens)
    if link == "static_link" and fr is not None:
        raise PhysicsSpecError(f"{where}.at_original_frame: the static link never moves; omit it")
    if dens and fr is not None:
        raise PhysicsSpecError(f"{where}.at_original_frame: density inertia is integrated over the collision surface "
                               f"(already posed by collision.*.at_original_frame); omit it here")
    if dens:
        if prov != "density_assumption":
            raise PhysicsSpecError(f"{where}: density_kg_m3 requires provenance density_assumption")
        if any(d.get(k) is not None for k in ("mass_kg", "com_capture_m", "inertia_com_capture_kg_m2")):
            raise PhysicsSpecError(f"{where}: give density_kg_m3 OR mass/com/inertia, not both")
        return InertialSpec(link, prov, fr, density_kg_m3=_num(d["density_kg_m3"], f"{where}.density_kg_m3", 0, True))
    if prov == "density_assumption":
        raise PhysicsSpecError(f"{where}: provenance density_assumption requires density_kg_m3")
    miss = [k for k in ("mass_kg", "com_capture_m", "inertia_com_capture_kg_m2") if d.get(k) is None]
    if miss:
        raise PhysicsSpecError(f"{where}: missing {miss} (or give density_kg_m3 with a watertight collision mesh)")
    m = _num(d["mass_kg"], f"{where}.mass_kg", 0, True)
    com = _vec(d["com_capture_m"], f"{where}.com_capture_m")
    I = check_inertia(_mat(d["inertia_com_capture_kg_m2"], f"{where}.inertia_com_capture_kg_m2", (3, 3)),
                      f"{where}.inertia_com_capture_kg_m2")
    return InertialSpec(link, prov, fr, mass_kg=m, com_capture_m=com, inertia_com_capture_kg_m2=I)


def _parse_joint(d) -> JointPhysicsSpec:
    where = "joint"
    _keys(d, ("lower", "upper", "velocity_limit", "effort_limit", "damping", "coulomb_friction", "armature",
              "actuation", "limits_from_observed_range", "provenance"), where,
          required=("velocity_limit", "effort_limit", "damping", "coulomb_friction", "provenance"))
    prev = d.get("limits_from_observed_range", False)
    if not isinstance(prev, bool):
        raise PhysicsSpecError(f"{where}.limits_from_observed_range: true / false")
    lo, hi = d.get("lower"), d.get("upper")
    if prev:
        if lo is not None or hi is not None:
            raise PhysicsSpecError(f"{where}: limits_from_observed_range: true and explicit lower/upper are exclusive")
    else:
        if lo is None or hi is None:
            raise PhysicsSpecError(f"{where}: physical stops lower/upper are required (the demonstration's q range is "
                                   f"NOT a mechanical limit; opt in with limits_from_observed_range: true for a "
                                   f"labelled preview)")
        lo, hi = _num(lo, f"{where}.lower"), _num(hi, f"{where}.upper")
        if not lo < hi:
            raise PhysicsSpecError(f"{where}: lower {lo} must be < upper {hi}")
        if not lo <= 0.0 <= hi:
            raise PhysicsSpecError(f"{where}: the canonical anchor q = 0 lies outside the stops [{lo}, {hi}]")
    if d.get("actuation") not in (None, "none"):
        raise PhysicsSpecError(f"{where}.actuation: only null / none is implemented (a passive joint); an actuator "
                               f"mapping per simulator is not")
    arm = d.get("armature")
    return JointPhysicsSpec(lower=lo, upper=hi,
                            velocity_limit=_num(d["velocity_limit"], f"{where}.velocity_limit", 0, True),
                            effort_limit=_num(d["effort_limit"], f"{where}.effort_limit", 0, True),
                            damping=_num(d["damping"], f"{where}.damping", 0),
                            coulomb_friction=_num(d["coulomb_friction"], f"{where}.coulomb_friction", 0),
                            armature=None if arm is None else _num(arm, f"{where}.armature", 0),
                            actuation=None, limits_from_observed_range=prev,
                            provenance=_choice(d["provenance"], f"{where}.provenance", PROVENANCE_MEASURED))


def parse_physics_spec(path) -> PhysicsSpec:
    import yaml
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise PhysicsSpecError(f"physics spec not found: {path}")
    try:
        d = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise PhysicsSpecError(f"{path}: YAML error: {e}") from None
    _keys(d, ("schema_version", "registration", "collision", "links", "joint", "contact", "validation", "notes"),
          str(path), required=("schema_version", "collision", "links", "joint"))
    if d["schema_version"] != SCHEMA_VERSION:
        raise PhysicsSpecError(f"schema_version: expected {SCHEMA_VERSION}, got {d['schema_version']!r}")
    base = path.parent
    reg = None if d.get("registration") is None else parse_registration(d["registration"])
    _keys(d["collision"], ("static", "moving"), "collision", required=("static", "moving"))
    col = {s: _parse_collision(d["collision"][s], s, base) for s in ("static", "moving")}
    _keys(d["links"], LINK_NAMES, "links", required=LINK_NAMES)
    ine = {lk: _parse_inertial(d["links"][lk], lk) for lk in LINK_NAMES}
    jp = _parse_joint(d["joint"])
    con = None
    if d.get("contact") is not None:
        c = d["contact"]
        _keys(c, ("static_friction", "dynamic_friction", "restitution", "provenance"), "contact",
              required=("static_friction", "dynamic_friction", "restitution", "provenance"))
        con = ContactSpec(_num(c["static_friction"], "contact.static_friction", 0),
                          _num(c["dynamic_friction"], "contact.dynamic_friction", 0),
                          _num(c["restitution"], "contact.restitution", 0),
                          _choice(c["provenance"], "contact.provenance", PROVENANCE_MEASURED))
        if con.restitution > 1:
            raise PhysicsSpecError("contact.restitution must be in [0, 1]")
    val = dict(DEFAULT_VALIDATION)
    if d.get("validation") is not None:
        _keys(d["validation"], tuple(DEFAULT_VALIDATION), "validation")
        val.update(d["validation"])
        r = val["extent_ratio_range"]
        if not (isinstance(r, list) and len(r) == 2 and 0 < float(r[0]) < 1 < float(r[1])):
            raise PhysicsSpecError("validation.extent_ratio_range: [lo < 1 < hi], both > 0")
        _num(val["centroid_offset_max"], "validation.centroid_offset_max", 0, True)
        _num(val["gaussian_inside_tolerance_fraction"], "validation.gaussian_inside_tolerance_fraction", 0)
        f = _num(val["gaussian_inside_min_fraction"], "validation.gaussian_inside_min_fraction", 0)
        if f > 1:
            raise PhysicsSpecError("validation.gaussian_inside_min_fraction must be in [0, 1]")
    return PhysicsSpec(path, reg, col, ine, jp, con, val, d)
