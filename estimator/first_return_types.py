"""First-return prior schema: validated BEFORE any GPU work, fails closed.

RigEuler3DGS_CODEX_Phase1_FirstReturn_Implementation_2026-11-05.md section 4.2.

    {
      "schema": 1,
      "model_id": "<sha256 of the canonical JSON without model_id>",
      "calibration_or_assumed": "calibrated" | "assumed",
      "units": {"depth": "m", "hazard": "1/m"},
      "provenance": {... free-form, required ...},
      "cameras": {
        "cam_000": {
          "z_near_m": 0.2, "z_far_m": 2.5,
          "unknown_hazard_segments": [{"z0_m": 0.2, "z1_m": 2.5, "lambda_per_m": 0.8}],
          "unknown_depth_sigma_model": {"kind": "sensor"} | {"kind": "constant", "sigma_m": 0.003}
        }, ...
      }
    }

There are NO default numbers: a missing prior, camera, field or unit is an error.
The CUDA operators of this release implement a constant hazard per camera; a prior
with more than one segment is refused rather than approximated.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

SCHEMA = 1
STATUSES = ("calibrated", "assumed")


@dataclass(frozen=True)
class CameraPrior:
    camera: str
    z_near: float
    z_far: float
    lam: float                    # constant unknown-surface hazard [1/m]
    sigma_kind: str               # "sensor" | "constant"
    sigma_m: float | None         # constant unknown-depth sd [m] when sigma_kind == "constant"


@dataclass(frozen=True)
class FirstReturnPrior:
    path: str
    model_id: str
    status: str
    cameras: dict
    provenance: dict
    raw: dict

    def camera(self, name: str) -> CameraPrior:
        if name not in self.cameras:
            raise KeyError(f"first-return prior {self.path} has no camera {name!r}")
        return self.cameras[name]

    def to_manifest(self) -> dict:
        return {"path": self.path, "model_id": self.model_id, "calibration_or_assumed": self.status,
                "cameras": {k: {"z_near_m": v.z_near, "z_far_m": v.z_far, "lambda_per_m": v.lam,
                                "unknown_depth_sigma": ({"kind": "sensor"} if v.sigma_kind == "sensor"
                                                        else {"kind": "constant", "sigma_m": v.sigma_m})}
                            for k, v in self.cameras.items()},
                "provenance": self.provenance}


def content_id(raw: dict) -> str:
    body = {k: v for k, v in raw.items() if k != "model_id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _num(x, what) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise ValueError(f"{what} must be a finite number, got {x!r}")
    return float(x)


def load_prior(path, camera_names=None) -> FirstReturnPrior:
    """Parse and validate; `camera_names` (if given) must all be covered."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"first-return prior {p} does not exist")
    raw = json.loads(p.read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{p}: schema {raw.get('schema')!r} != {SCHEMA}")
    if raw.get("calibration_or_assumed") not in STATUSES:
        raise ValueError(f"{p}: calibration_or_assumed must be one of {STATUSES}")
    if raw.get("units") != {"depth": "m", "hazard": "1/m"}:
        raise ValueError(f"{p}: units must be exactly {{'depth': 'm', 'hazard': '1/m'}}")
    if not isinstance(raw.get("provenance"), dict) or not raw["provenance"]:
        raise ValueError(f"{p}: a non-empty provenance object is required")
    if raw.get("model_id") != content_id(raw):
        raise ValueError(f"{p}: model_id does not match the content (edited by hand?)")
    cams = raw.get("cameras")
    if not isinstance(cams, dict) or not cams:
        raise ValueError(f"{p}: no cameras")
    out = {}
    for name, c in cams.items():
        zn, zf = _num(c.get("z_near_m"), f"{name}.z_near_m"), _num(c.get("z_far_m"), f"{name}.z_far_m")
        if not (0.0 < zn < zf):
            raise ValueError(f"{p}: {name} needs 0 < z_near_m < z_far_m, got {zn}, {zf}")
        seg = c.get("unknown_hazard_segments")
        if not isinstance(seg, list) or not seg:
            raise ValueError(f"{p}: {name}.unknown_hazard_segments must be a non-empty list")
        prev = zn
        for k, s in enumerate(seg):
            z0, z1 = _num(s.get("z0_m"), f"{name}.seg{k}.z0_m"), _num(s.get("z1_m"), f"{name}.seg{k}.z1_m")
            lam = _num(s.get("lambda_per_m"), f"{name}.seg{k}.lambda_per_m")
            if abs(z0 - prev) > 1e-12 or not z1 > z0 or lam < 0:
                raise ValueError(f"{p}: {name} hazard segments must tile [z_near, z_far] with lambda >= 0")
            prev = z1
        if abs(prev - zf) > 1e-12:
            raise ValueError(f"{p}: {name} hazard segments end at {prev}, not z_far {zf}")
        if len(seg) != 1:
            raise ValueError(f"{p}: {name} has a piecewise hazard ({len(seg)} segments); this release "
                             "implements a constant hazard per camera only -- refusing to approximate")
        sm = c.get("unknown_depth_sigma_model")
        if not isinstance(sm, dict) or sm.get("kind") not in ("sensor", "constant"):
            raise ValueError(f"{p}: {name}.unknown_depth_sigma_model must be {{'kind': 'sensor'|'constant'}}")
        sig = None
        if sm["kind"] == "constant":
            sig = _num(sm.get("sigma_m"), f"{name}.unknown_depth_sigma_model.sigma_m")
            if not sig > 0:
                raise ValueError(f"{p}: {name} unknown sigma must be positive")
        out[name] = CameraPrior(name, zn, zf, float(seg[0]["lambda_per_m"]), sm["kind"], sig)
    if camera_names is not None:
        missing = [c for c in camera_names if c not in out]
        if missing:
            raise ValueError(f"{p}: no first-return prior for cameras {missing}")
    return FirstReturnPrior(str(p), raw["model_id"], raw["calibration_or_assumed"], out,
                            raw["provenance"], raw)


def write_prior(path, cameras: dict, status: str, provenance: dict) -> dict:
    """cameras: {name: {"z_near_m", "z_far_m", "lambda_per_m", "unknown_depth_sigma_model"}}."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    raw = {"schema": SCHEMA, "calibration_or_assumed": status, "units": {"depth": "m", "hazard": "1/m"},
           "provenance": provenance,
           "cameras": {n: {"z_near_m": float(c["z_near_m"]), "z_far_m": float(c["z_far_m"]),
                           "unknown_hazard_segments": [{"z0_m": float(c["z_near_m"]), "z1_m": float(c["z_far_m"]),
                                                        "lambda_per_m": float(c["lambda_per_m"])}],
                           "unknown_depth_sigma_model": c["unknown_depth_sigma_model"]}
                       for n, c in cameras.items()}}
    raw["model_id"] = content_id(raw)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(raw, indent=1, sort_keys=True))
    load_prior(path)            # round-trip validation
    return raw
