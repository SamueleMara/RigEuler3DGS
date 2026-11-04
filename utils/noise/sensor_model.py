"""Frozen, provenance-tracked RGB-D sensor model shared by every IEKF phase.

RigEuler3DGS_Calibrated_Bootstrap_IEKF_Implementation_Spec.md sections 1-2.

Policies (always explicit, recorded in every manifest):

  calibrated  estimated from a user-supplied STATIC interval of the same capture
              (tools/calibrate_static_rgbd.py); per camera, never pooled
  ideal       synthetic data: no physical sensor noise exists. No sensor model is
              applied; the filter's configured sigmas stay what they were (a
              declared numerical / model-mismatch scale, not a sensor statistic)
  assumed     a real capture run WITHOUT calibration: the configured sigmas are
              used and the run is labelled UNCALIBRATED

What the static interval identifies (section 2.1): for D = d* + b + e it gives
the distribution and the time correlation of the fluctuating part e. It does
NOT identify the persistent bias b separately from the true depth d*; that
limitation is stored in the model itself (`identifiability`).

Depth variance model per camera, in metres^2 at observed camera-Z z:

    sigma_D^2(z) = max(a + b z^2 + d z^4, q^2 / 12)

with a, b, d >= 0 and q the measured depth quantization step (an instrument
floor, measured, never invented). RGB: a per-camera 3x3 channel covariance of the
temporal fluctuation (values in [0, 1] units).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

POLICIES = ("calibrated", "ideal", "assumed")
SCHEMA = 1


@dataclass
class CameraNoise:
    camera: str
    depth_coeffs: tuple = (0.0, 0.0, 0.0)        # (a, b, d): a + b z^2 + d z^4  [m^2]
    depth_model: str = "a+bz2"                    # the selected form
    depth_quantization_m: float = 0.0             # measured step q; floor q^2/12
    depth_range_m: tuple = (0.0, 0.0)             # range the fit is supported on
    rgb_covariance: tuple = ((0.0,) * 3,) * 3     # 3x3, [0,1] units
    invalid_fraction: float = 0.0                 # missing returns inside the mask
    lag1_autocorrelation: float = 0.0             # temporal diagnostic (NOT a model)
    autocorrelation_ci95: tuple = (0.0, 0.0)
    global_offset_std_m: float = 0.0              # frame-to-frame common-mode depth
    num_pixels: int = 0
    num_frames: int = 0

    def depth_variance(self, z):
        """sigma_D^2 at observed depth z (numpy or torch, metres)."""
        a, b, d = (float(c) for c in self.depth_coeffs)
        floor = float(self.depth_quantization_m) ** 2 / 12.0
        z2 = z * z
        v = a + b * z2 + d * z2 * z2
        try:                                       # torch
            import torch
            if isinstance(z, torch.Tensor):
                return torch.clamp(v, min=floor) if floor > 0 else v
        except ImportError:                        # pragma: no cover
            pass
        return np.maximum(v, floor)

    def rgb_variance(self):
        """Per-channel variance (diagonal of the channel covariance)."""
        C = np.asarray(self.rgb_covariance, float)
        return np.diag(C).copy()


@dataclass
class SceneNoiseModel:
    policy: str
    cameras: dict = field(default_factory=dict)          # name -> CameraNoise
    provenance: dict = field(default_factory=dict)
    identifiability: str = ("static frames identify the fluctuating noise e (variance, "
                            "temporal correlation); they do NOT identify the persistent "
                            "bias b separately from the true depth d*")
    schema: int = SCHEMA

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"sensor policy must be one of {POLICIES}, got {self.policy!r}")
        if self.policy == "calibrated" and not self.cameras:
            raise ValueError("a calibrated sensor model needs per-camera noise")
        if self.policy != "calibrated" and self.cameras:
            raise ValueError(f"policy {self.policy!r} carries no physical per-camera noise")

    # -- queries ------------------------------------------------------------
    @property
    def is_physical(self) -> bool:
        return self.policy == "calibrated"

    def camera(self, camera_id: str) -> CameraNoise:
        if camera_id not in self.cameras:
            raise KeyError(f"no calibrated noise for camera {camera_id!r}; known: {sorted(self.cameras)} "
                           "(cameras are never pooled or substituted)")
        return self.cameras[camera_id]

    def depth_variance(self, camera_id: str, z, u=None, v=None, context=None):
        return self.camera(camera_id).depth_variance(z)

    def rgb_covariance(self, camera_id: str):
        return np.asarray(self.camera(camera_id).rgb_covariance, float)

    def temporal_model(self, camera_id: str) -> dict:
        c = self.camera(camera_id)
        return {"lag1_autocorrelation": c.lag1_autocorrelation, "ci95": list(c.autocorrelation_ci95),
                "use": "diagnostic only (section 4.2): not an AR(1) model of motion residuals"}

    # -- identity / io --------------------------------------------------------
    def to_dict(self) -> dict:
        return {"schema": self.schema, "policy": self.policy,
                "cameras": {k: asdict(v) for k, v in sorted(self.cameras.items())},
                "provenance": self.provenance, "identifiability": self.identifiability}

    @property
    def model_id(self) -> str:
        """sha256 of the numerical content (cameras + policy + schema), not of the
        free-form provenance -- the id every stage manifest must carry."""
        core = {"schema": self.schema, "policy": self.policy,
                "cameras": {k: asdict(v) for k, v in sorted(self.cameras.items())}}
        return hashlib.sha256(json.dumps(core, sort_keys=True, default=float).encode()).hexdigest()

    def to_json(self, path) -> str:
        d = self.to_dict()
        d["sensor_model_id"] = self.model_id
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1, default=float))
        return self.model_id

    @classmethod
    def from_dict(cls, d: dict) -> "SceneNoiseModel":
        cams = {}
        for k, v in (d.get("cameras") or {}).items():
            v = dict(v)
            for key in ("depth_coeffs", "depth_range_m", "autocorrelation_ci95"):
                if key in v:
                    v[key] = tuple(v[key])
            if "rgb_covariance" in v:
                v["rgb_covariance"] = tuple(tuple(r) for r in v["rgb_covariance"])
            cams[k] = CameraNoise(**v)
        m = cls(policy=d["policy"], cameras=cams, provenance=d.get("provenance", {}),
                identifiability=d.get("identifiability", cls.identifiability), schema=d.get("schema", SCHEMA))
        want = d.get("sensor_model_id")
        if want is not None and want != m.model_id:
            raise ValueError(f"sensor model id mismatch: file says {want[:12]}, content gives "
                             f"{m.model_id[:12]} -- the file was edited or truncated")
        return m

    @classmethod
    def from_json(cls, path) -> "SceneNoiseModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


def declared_model(policy: str, provenance: dict | None = None) -> SceneNoiseModel:
    """The `ideal` (synthetic) or `assumed` (real, uncalibrated) placeholder."""
    if policy not in ("ideal", "assumed"):
        raise ValueError("declared_model is for 'ideal' or 'assumed'")
    return SceneNoiseModel(policy=policy, provenance=dict(provenance or {},
                           status="UNCALIBRATED" if policy == "assumed" else "SYNTHETIC_NO_SENSOR_NOISE"))


def load_sensor_model(path) -> SceneNoiseModel | None:
    return None if path in (None, "", "none") else SceneNoiseModel.from_json(path)
