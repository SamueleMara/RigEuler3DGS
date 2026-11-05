"""Frame-level first-return depth observation for the Phase-1 IEKF (opt-in).

RigEuler3DGS_CODEX_Phase1_FirstReturn_Implementation_2026-11-05.md sections 2-5.

Modes (`--first-return-mode`):
  off     the legacy observation, untouched (default)
  audit   the legacy update; the first-return likelihood is evaluated at every
          accepted state for diagnostics only
  depth   the blended depth Z/(A+eps) is REMOVED from the objective and replaced by
          the per-contributor first-return depth likelihood; RGB and the alpha term
          stay exactly the legacy ones (the A/B isolates the depth model)
  rgbd    joint RGB-D association (Gate E): NOT IMPLEMENTED -- refused

Observation eligibility: the masked mode's PreparedRGBD.depth_valid (= V_orig AND M,
finite, > 0) AND D in [z_near, z_far]. No background depth is ever read.
sigma_D: the configured / calibrated sensor sd at the OBSERVED depth plus the declared
depth model-mismatch sd (in variance), fixed per pixel -- the declared model has no
variance derivative. The unknown-surface depth sd follows the prior ("sensor" = the
same sigma_D map, or a declared constant).

Local curvature (declared PSD approximation, not the observed mixture Hessian):
  B = B_match + B_vis,
  B_match = sum_{c,p,i} gamma_i / sigma^2 J_z^T J_z      (block-diagonal, translation)
  B_vis   = sum_{c,p} g_p g_p^T, g_p = visibility score of pixel p (empirical Fisher)
The same B gives the normal operator (B_vis through JVP then VJP, so cross-Gaussian
ray coupling is kept) and the per-Gaussian Fisher blocks of the preconditioner and the
posterior. The posterior built from it is labelled POSTERIOR_UNCERTIFIED: the missing-
association information is not subtracted, and no coverage test has been passed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import first_return_operators as fo
from .first_return_types import FirstReturnPrior, load_prior

MODES = ("off", "audit", "depth", "rgbd")
POSTERIOR_UNCERTIFIED = "POSTERIOR_UNCERTIFIED"


@dataclass
class FirstReturnObservation:
    mode: str
    prior: FirstReturnPrior | None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"first-return mode must be one of {MODES}")
        if self.mode == "rgbd":
            raise ValueError("first-return mode 'rgbd' (joint RGB-D association, Gate E) is not implemented")
        if self.mode != "off" and self.prior is None:
            raise ValueError(f"first-return mode {self.mode!r} needs a validated prior (--first-return-prior)")

    @property
    def replaces_depth(self) -> bool:
        return self.mode == "depth"

    @property
    def enabled(self) -> bool:
        return self.mode in ("audit", "depth")

    def to_manifest(self) -> dict:
        return {"mode": self.mode, "prior": None if self.prior is None else self.prior.to_manifest(),
                "kernel_source_sha256": fo.source_sha(),
                "depth_eligibility": "PreparedRGBD.depth_valid (V_orig AND M) AND D in [z_near, z_far]",
                "sigma_D": "sensor sd at the OBSERVED depth (+ declared depth_model_sigma in variance), fixed per pixel",
                "contributors": "production evalAlpha (skip < 1/255, clamp 0.99), FULL ray (no T_STOP), "
                                "z = camera centre depth, only z in (z_near, z_far]",
                "local_curvature": "B_match (gamma-weighted J_z^T J_z / sigma^2) + B_vis (sum_p g_p g_p^T)",
                "posterior": POSTERIOR_UNCERTIFIED if self.mode == "depth" else "legacy (first return not used)",
                "rgb_alpha_terms": "legacy, unchanged" if self.mode == "depth" else "legacy",
                "not_implemented": ["rgbd joint association (Gate E)", "piecewise hazard", "invalid-depth branch",
                                    "missing-association information correction"]}

    def evaluate(self, lin, rgbd_gt, cfg, diagnostics=True) -> "FirstReturnEvaluation":
        return FirstReturnEvaluation.build(self, lin, rgbd_gt, cfg, diagnostics)


def sigma_maps(cfg, camera: str | None, D: torch.Tensor, valid: torch.Tensor, cam_prior):
    """(sigma_D, sigma_U) at the observed depth, fixed per pixel."""
    from .observation_terms import calibrated_camera, depth_sensor_sigma
    cal = calibrated_camera(cfg, camera)
    z = torch.where(valid, D, torch.ones_like(D))
    if cal is not None:
        sd = cal.depth_variance(z.double()).to(D.dtype).sqrt()
    else:
        sd = depth_sensor_sigma(D, valid, cfg)
    dms = float(getattr(cfg, "depth_model_sigma", 0.0) or 0.0)
    if dms > 0:
        sd = (sd * sd + dms * dms).sqrt()
    sd = torch.where(valid, sd, torch.ones_like(sd))
    su = sd.clone() if cam_prior.sigma_kind == "sensor" else torch.full_like(sd, float(cam_prior.sigma_m))
    if bool(valid.any()) and not bool((sd[valid] > 0).all()):
        raise ValueError(f"{camera}: non-positive depth sd on a valid pixel")
    return sd, su


@dataclass
class _Cam:
    name: str
    args: tuple
    logL: torch.Tensor
    valid: torch.Tensor
    w_match: torch.Tensor
    axis: torch.Tensor
    diag: dict = field(default_factory=dict)


@dataclass
class FirstReturnEvaluation:
    measurement_energy: float
    cams: list
    posterior_status: str
    diagnostics: dict
    _score: torch.Tensor | None = None
    _fisher: torch.Tensor | None = None

    @classmethod
    def build(cls, obs: FirstReturnObservation, lin, rgbd_gt, cfg, diagnostics=True):
        from .observation_terms import camera_name
        cams, energy, diag = [], 0.0, {"cameras": []}
        for k, (view, ob) in enumerate(zip(lin.views, rgbd_gt, strict=True)):
            if not hasattr(ob, "depth_valid"):
                raise TypeError("first-return needs the masked observation (PreparedRGBD): M = 1 is mandatory")
            name = camera_name(cfg, k)
            if name is None:
                raise ValueError("first-return needs the camera identity of every observation (cfg.camera_names)")
            cp = obs.prior.camera(name)
            dev = view.depth.device
            D = ob.depth_observed.to(dev).float()
            valid0 = ob.depth_valid.to(dev).bool() & torch.isfinite(D) & (D > 0)
            valid = valid0 & (D >= cp.z_near) & (D <= cp.z_far)
            sd, su = sigma_maps(cfg, name, D, valid, cp)
            args = fo.camera_args(view.cache, torch.where(valid, D, torch.zeros_like(D)), valid, sd, su,
                                  cp.lam, cp.z_near, cp.z_far)
            r = fo.fr_eval(args, diagnostics=diagnostics)
            logL = r["logL"]
            e = -float(logL[valid].sum())
            energy += e
            c = _Cam(name, args, logL, valid, r["w_match"], fo.depth_axis(view.cache).to(dev))
            if diagnostics:
                v = valid
                n = int(v.sum())
                c.diag = {"camera": name, "valid_pixels": n,
                          "excluded_out_of_support": int((valid0 & ~valid).sum()),
                          "energy": e, "mean_logL": (-e / n) if n else None,
                          "gamma_unknown_mean": float(r["gamma_unknown"][v].mean()) if n else None,
                          "unknown_dominant_fraction": float((r["gamma_unknown"][v] > 0.5).float().mean()) if n else None,
                          "matched_gamma_max_mean": float(r["gamma_max"][v].mean()) if n else None,
                          "front_contradiction_mean": float(r["front"][v].mean()) if n else None,
                          "n_support_mean": float(r["n_support"][v].float().mean()) if n else None}
                c.diag["_maps"] = {k_: r[k_] for k_ in ("logL", "gamma_unknown", "gamma_max", "front", "n_support")}
                c.diag["_per_gaussian"] = {k_: r[k_] for k_ in ("gamma_sum", "front_sum", "pixel_count")}
                diag["cameras"].append({k_: v_ for k_, v_ in c.diag.items() if not k_.startswith("_")})
            cams.append(c)
        status = POSTERIOR_UNCERTIFIED
        return cls(energy, cams, status, diag)

    # ---- solver interface (pose tangent, [N,6]) -------------------------------
    def score_pose(self) -> torch.Tensor:
        """b_meas = + sum grad log L (ascent direction), [N,6]."""
        if self._score is None:
            s = None
            for c in self.cams:
                g = fo.fr_vjp(c.args, c.logL, c.valid.float(), fo.PART_ALL)
                s = g if s is None else s + g
            self._score = s
        return self._score

    def normal_pose(self, v: torch.Tensor) -> torch.Tensor:
        """(B_match + B_vis) v, PSD."""
        out = torch.zeros_like(v, dtype=torch.float32)
        vf = v.float().contiguous()
        for c in self.cams:
            t = vf[:, :3] @ c.axis                                    # J_z v per Gaussian
            out[:, :3] += (c.w_match * t)[:, None] * c.axis[None]
            s = fo.fr_jvp(c.args, c.logL, vf, fo.PART_VIS)            # g_p . v per pixel
            out += fo.fr_vjp(c.args, c.logL, s * c.valid.float(), fo.PART_VIS)
        return out.to(v.dtype)

    def fisher_pose(self) -> torch.Tensor:
        """Per-Gaussian 6x6 diagonal blocks of the SAME B, [N,6,6]."""
        if self._fisher is None:
            F = None
            for c in self.cams:
                Fc = fo.sym21_to_dense(fo.fr_fisher_vis(c.args, c.logL))
                Fc[:, :3, :3] += c.w_match[:, None, None] * (c.axis[:, None] * c.axis[None, :])[None]
                F = Fc if F is None else F + Fc
            self._fisher = F
        return self._fisher

    def per_gaussian(self) -> dict:
        """Summed over cameras: posterior hit mass, front contradiction, pixel count."""
        out = {}
        for c in self.cams:
            for k, t in c.diag.get("_per_gaussian", {}).items():
                out[k] = t.clone() if k not in out else out[k] + t
        return out

    def summary(self, top=20) -> dict:
        pg = self.per_gaussian()
        s = {"energy": self.measurement_energy, "posterior_status": self.posterior_status,
             "cameras": self.diagnostics["cameras"]}
        if "front_sum" in pg:
            fs = pg["front_sum"]
            k = min(top, int(fs.numel()))
            val, idx = torch.topk(fs, k)
            s["top_front_contradiction_gaussians"] = [{"gid": int(i), "front_sum": float(x),
                                                       "pixels": float(pg["pixel_count"][i]),
                                                       "gamma_sum": float(pg["gamma_sum"][i])}
                                                      for x, i in zip(val.tolist(), idx.tolist())]
        return s


def from_args(args, camera_names=None) -> FirstReturnObservation | None:
    mode = getattr(args, "first_return_mode", "off") or "off"
    if mode == "off":
        if getattr(args, "first_return_prior", None):
            raise SystemExit("--first-return-prior given with --first-return-mode off")
        return None
    path = getattr(args, "first_return_prior", None)
    if not path:
        raise SystemExit(f"--first-return-mode {mode} needs --first-return-prior PATH (a calibrated or "
                         "explicitly assumed prior; there is no default hazard)")
    if getattr(args, "observation_mode", "masked-rgb-expected-quad") == "original":
        raise SystemExit("--first-return-mode needs a masked --observation-mode (M = 1 eligibility)")
    try:
        return FirstReturnObservation(mode, load_prior(path, camera_names))
    except ValueError as ex:
        raise SystemExit(f"first-return prior: {ex}")


def add_cli(ap) -> None:
    ap.add_argument("--first-return-mode", choices=MODES, default="off",
                    help="off (default, legacy) | audit (legacy update + first-return diagnostics) | depth "
                         "(first-return depth likelihood replaces the blended depth; RGB/alpha unchanged) | "
                         "rgbd (not implemented, refused)")
    ap.add_argument("--first-return-prior", default=None,
                    help="validated first-return prior JSON (estimator/first_return_types.py)")
