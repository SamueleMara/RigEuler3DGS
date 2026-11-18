"""Occlusion-consistent Phase-1 observation likelihood (opt-in).

RigEuler3DGS_PHASE1_COHERENCE_CODEX_IMPLEMENTATION_SPEC_2026-11-16.md section I.A-I.C.

`--phase1-likelihood legacy-masked` (DEFAULT) keeps the masked expected-quad objective
of estimator/masked_observation.py untouched. `occlusion-consistent` replaces it by ONE
ray likelihood in which every pixel belongs to exactly one status:

  OBJECT_SUPPORTED        M = 1. The legacy positive model, restricted to M = 1:
                          RGB fixed quadratic on I - C(x; b_bar), alpha
                          1/2 kappa (1 - A)^2 with kappa = trace(W)/12 (the exact
                          Uniform[0,1]^3-background expectation of THAT RGB term),
                          blended-depth Student-t on V_raw AND M with the pre-existing
                          alpha gate. NOT IMPLEMENTED: the joint RGB-D ownership mixture
                          pi p_obj + (1 - pi) p_unknown of spec I.A (no calibrated
                          p_unknown / ownership prior exists); an end-effector inside
                          M = 1 is therefore still explained as object, as in v6.
  KNOWN_NONOBJECT_DEPTH   M = 0 with valid raw sensor depth D, not on the mask edge:
                          L_p = -log(1 - pi A_p),
                          A_p = sum_k w_k Phi((D - z_k)/sigma_p) = 1 - E_{Z~N(D,sigma_p^2)}[T_front(Z)]
                          (iekf_negative_ray.cu). An object Gaussian far BEHIND the
                          observed surface (an occluder in front of it) has s_k -> 0:
                          no score, no curvature. One in front of it has s_k -> 1 and is
                          pushed out exactly as a free-space ray says.
                          sigma_p = the calibrated sensor sd at the OBSERVED depth
                          (+ the declared depth_model_sigma in variance), fixed per pixel.
                          pi = the declared negative-mask reliability (default 1).
  UNKNOWN                 M = 0 with invalid / missing raw depth, or an M = 0 pixel with
                          an M = 1 pixel in its 3x3 neighbourhood (mask-edge mixed
                          returns: the sensor model is not calibrated there). No term.
  TRUSTED_EMPTY           never assigned: no independently established empty-ray
                          measurement exists in these datasets.

No pixel carries both a legacy and a negative-ray term: RGB and alpha precisions are
zero off M = 1, depth is V_raw AND M as before, and the negative-ray term lives only on
KNOWN_NONOBJECT pixels. The trial objective (every trust-region trial), the RHS (exact
score), the Gauss-Newton normal operator and the per-Gaussian Fisher blocks are all
derived from the same L_p:
    rhs     = -sum_p (pi / E_p) dA_p/dx                      (exact -grad L)
    normal  = sum_p (pi / E_p)^2 J_p^T J_p                   (GN: drops (pi/E) d2A)
    fisher  = per-Gaussian diagonal blocks of `normal`
with E_p = 1 - pi A_p >= T_STOP by the renderer's termination rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import negative_ray_operators as no

LIKELIHOODS = ("legacy-masked", "occlusion-consistent")
MASK_EDGE = ("unknown", "nonobject")


@dataclass(frozen=True)
class OcclusionLikelihood:
    """The declared negative-ray model. Immutable for a run."""
    negative_reliability: float = 1.0
    mask_edge: str = "unknown"
    deterministic: bool = False

    def __post_init__(self):
        if not (0.0 < float(self.negative_reliability) <= 1.0):
            raise ValueError("negative-ray reliability must be in (0, 1]")
        if self.mask_edge not in MASK_EDGE:
            raise ValueError(f"mask_edge must be one of {MASK_EDGE}")

    def to_manifest(self) -> dict:
        return {
            "phase1_likelihood": "occlusion-consistent",
            "kernel_source_sha256": no.source_sha(),
            "statuses": {"OBJECT_SUPPORTED": "M = 1", "KNOWN_NONOBJECT_DEPTH":
                         "M = 0, V_raw (finite, > 0), no M = 1 pixel in the 3x3 neighbourhood"
                         + (" (mask-edge pixels included)" if self.mask_edge == "nonobject" else ""),
                         "UNKNOWN": "M = 0 with invalid depth" + (", or on the mask edge" if self.mask_edge == "unknown" else ""),
                         "TRUSTED_EMPTY": "never assigned (no independent empty-ray measurement)"},
            "negative_ray": "L = -log(1 - pi A), A = sum_k w_k Phi((D - z_k)/sigma), exact Gaussian depth marginal",
            "negative_reliability_pi": float(self.negative_reliability),
            "negative_reliability_source": ("declared: every KNOWN_NONOBJECT ray certified (pi = 1)"
                                            if float(self.negative_reliability) == 1.0
                                            else "declared mask-reliability prior (not calibrated)"),
            "sigma": "calibrated sensor sd at the OBSERVED depth (+ declared depth_model_sigma in variance)",
            "contributors": "production evalAlpha (skip < 1/255, clamp 0.99) and T_STOP termination; "
                            "z = camera centre depth (the renderer's sort key)",
            "positive_model": "legacy expected-quad RGB + alpha restricted to M = 1, blended-depth Student-t "
                              "on V_raw AND M (UNCHANGED from v6 on those pixels)",
            "curvature": "Gauss-Newton (pi/E)^2 J^T J; the (pi/E) d2A term is dropped (PSD)",
            "deterministic_reductions": bool(self.deterministic),
            "not_implemented": ["positive-ray RGB-D ownership mixture (end-effector inside M = 1)",
                                "first-return association on M = 1 (blended depth kept)",
                                "TRUSTED_EMPTY rays", "calibrated mask false-negative rate"],
        }


def ray_status(ob, mask_edge: str = "unknown") -> torch.Tensor:
    """[H,W] uint8 status of one prepared camera (negative_ray_operators.ST_*)."""
    if getattr(ob, "depth_valid_raw", None) is None:
        raise ValueError("the occlusion-consistent likelihood needs the raw depth validity "
                         "(PreparedRGBD.depth_valid_raw); refusing to treat masked validity as raw")
    M = ob.object_coverage > 0.5
    raw = ob.depth_valid_raw.bool() & torch.isfinite(ob.depth_observed) & (ob.depth_observed > 0)
    st = torch.full(M.shape, no.ST_UNKNOWN, dtype=torch.uint8, device=M.device)
    st[M] = no.ST_SUPPORTED
    neg = (~M) & raw
    if mask_edge == "unknown":
        near = torch.nn.functional.max_pool2d(M.float()[None, None], 3, stride=1, padding=1)[0, 0] > 0
        neg = neg & ~near
    st[neg] = no.ST_NONOBJECT
    return st


def depth_sigma_map(cfg, camera, D: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """sigma at the OBSERVED depth: calibrated per camera when available, else the
    configured depth model; + declared depth_model_sigma in variance. 1 where invalid."""
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
    if bool(valid.any()) and not bool((sd[valid] > 0).all()):
        raise ValueError(f"{camera}: non-positive depth sd on a valid pixel")
    return sd


def positive_camera_terms(views, prepared, cfg, model):
    """Per-camera CameraTerms of the OBJECT_SUPPORTED pixels: the legacy masked
    expected-quad terms with RGB and alpha restricted to M = 1 (precision AND kappa
    exactly zero elsewhere); depth on V_raw AND M as before."""
    from .masked_observation import PreparedRGBD
    from .observation_terms import camera_name, camera_terms
    if model is None or not model.alpha_term:
        raise ValueError("the occlusion-consistent likelihood is defined on masked-rgb-expected-quad")
    if model.rgb_support != "full":
        raise ValueError("the occlusion-consistent likelihood sets the RGB support itself; "
                         "--rgb-support must stay 'full'")
    if getattr(cfg, "debug_ablation", None):
        raise ValueError("debug_ablation is not supported with the occlusion-consistent likelihood")
    if len(views) != len(prepared):
        raise AssertionError(f"{len(views)} views but {len(prepared)} prepared observations")
    calibrated = getattr(getattr(cfg, "sensor_model", None), "is_physical", False)
    if not calibrated and model.rgb_sigma is not None and abs(float(model.rgb_sigma) - float(cfg.rgb_sigma)) > 0:
        raise AssertionError(f"observation model sigma {model.rgb_sigma} != IEKF rgb_sigma {cfg.rgb_sigma}")
    out = []
    for j, (v, ob) in enumerate(zip(views, prepared)):
        if not isinstance(ob, PreparedRGBD):
            raise TypeError("the occlusion-consistent likelihood needs PreparedRGBD observations")
        ob = ob.to(v.rgb.device)
        M = ob.object_coverage > 0.5
        Mf = M.to(v.rgb.dtype)
        out.append(camera_terms(v, ob.target(model.background_tensor(v.rgb.device)), ob.depth_observed,
                                ob.depth_valid, cfg, rgb_robust=False, coverage=ob.object_coverage,
                                kappa_fn=lambda p, _m=Mf: model.kappa(p) * _m, rgb_valid=M,
                                camera=camera_name(cfg, j)))
    return out


@dataclass
class _Cam:
    name: str
    args: tuple
    A: torch.Tensor
    c1: torch.Tensor          # pi / E   (score weight)
    c2: torch.Tensor          # (pi / E)^2 (GN weight)
    status: torch.Tensor
    energy: float
    floor_hits: int


@dataclass
class NegativeRayEvaluation:
    """The KNOWN_NONOBJECT term at ONE linearization (trial or accepted state)."""
    energy: float
    cams: list
    deterministic: bool
    _score: torch.Tensor | None = None
    _fisher: torch.Tensor | None = None
    diagnostics: dict = field(default_factory=dict)

    @classmethod
    def build(cls, model: OcclusionLikelihood, lin, prepared, cfg):
        from .observation_terms import camera_name
        cams, energy = [], 0.0
        pi = float(model.negative_reliability)
        for k, (view, ob) in enumerate(zip(lin.views, prepared, strict=True)):
            dev = view.depth.device
            ob = ob.to(dev)
            name = camera_name(cfg, k)
            st = ray_status(ob, model.mask_edge)
            D = ob.depth_observed.to(dev).float()
            neg = st == no.ST_NONOBJECT
            sd = depth_sigma_map(cfg, name, D, neg)
            args = no.camera_args(view.cache, torch.where(neg, D, torch.zeros_like(D)), sd, st, pi)
            r = no.nr_eval(args, diagnostics=False)
            A, L = r["A"], r["L"]
            E = (1.0 - pi * A).clamp_min(1e-12)
            c1 = torch.where(neg, pi / E, torch.zeros_like(E)).float()
            c2 = torch.where(neg, (pi / E) ** 2, torch.zeros_like(E)).float()
            e = float(L[neg].sum())
            energy += e
            cams.append(_Cam(name, args, A, c1, c2, st, e, r["floor_hits"]))
        return cls(energy, cams, bool(model.deterministic))

    # ---- solver interface (pose tangent, [N,6]) ------------------------------
    def score_pose(self) -> torch.Tensor:
        """-grad L, [N,6] (added to the RHS H^T W r)."""
        if self._score is None:
            s = None
            for c in self.cams:
                g = no.nr_vjp(c.args, c.A, c.c1, self.deterministic)
                s = g if s is None else s + g
            self._score = -s
        return self._score

    def normal_pose(self, v: torch.Tensor) -> torch.Tensor:
        """sum_p (pi/E_p)^2 J_p^T J_p v (PSD), [N,6]."""
        vf = v.float().contiguous()
        out = None
        for c in self.cams:
            jv = no.nr_jvp(c.args, c.A, vf)
            g = no.nr_vjp(c.args, c.A, c.c2 * jv, self.deterministic)
            out = g if out is None else out + g
        return out.to(v.dtype)

    def fisher_pose(self) -> torch.Tensor:
        """Per-Gaussian 6x6 diagonal blocks of the SAME GN operator, [N,6,6]."""
        if self._fisher is None:
            F = None
            for c in self.cams:
                Fc = no.sym21_to_dense(no.nr_fisher(c.args, c.A, c.c2, self.deterministic))
                F = Fc if F is None else F + Fc
            self._fisher = F
        return self._fisher

    # ---- diagnostics ---------------------------------------------------------
    def status_counts(self) -> list:
        out = []
        for c in self.cams:
            n = {nm: int((c.status == code).sum()) for code, nm in no.STATUS_NAMES.items() if code != no.ST_NONE}
            out.append({"camera": c.name, **n, "negative_energy": c.energy, "log_floor_hits": c.floor_hits})
        return out

    def per_gaussian(self) -> dict:
        """Summed over cameras at THIS linearization: supported / front / occluded / unknown
        blending weights and pixel counts (evaluation only, float atomics)."""
        out = {}
        for c in self.cams:
            r = no.nr_eval(c.args, diagnostics=True)
            for k in ("supported_weight", "front_weight", "occluded_weight", "unknown_weight",
                      "supported_pixels", "nonobject_pixels", "unknown_pixels"):
                t = r[k].float()
                out[k] = t.clone() if k not in out else out[k] + t
        return out


def evaluate(model: OcclusionLikelihood, lin, prepared, cfg) -> NegativeRayEvaluation:
    return NegativeRayEvaluation.build(model, lin, prepared, cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_cli(ap) -> None:
    ap.add_argument("--phase1-likelihood", choices=LIKELIHOODS, default="legacy-masked",
                    help="legacy-masked (DEFAULT, v6) | occlusion-consistent: M = 0 pixels with valid "
                         "raw depth carry -log E[T_front(D)] instead of the grey-background RGB and "
                         "alpha -> 0 pushes; M = 0 without depth carries nothing "
                         "(estimator/occlusion_likelihood.py)")
    ap.add_argument("--negative-ray-reliability", type=float, default=1.0,
                    help="declared probability that a KNOWN_NONOBJECT ray is truly non-object (0, 1]; "
                         "1 = certified (default)")
    ap.add_argument("--negative-ray-mask-edge", choices=MASK_EDGE, default="unknown",
                    help="M = 0 pixels touching M = 1 (3x3): unknown (default; mixed returns, no term) "
                         "| nonobject")


def from_args(args) -> OcclusionLikelihood | None:
    mode = getattr(args, "phase1_likelihood", "legacy-masked") or "legacy-masked"
    if mode == "legacy-masked":
        if float(getattr(args, "negative_ray_reliability", 1.0)) != 1.0 or \
                getattr(args, "negative_ray_mask_edge", "unknown") != "unknown":
            raise SystemExit("--negative-ray-* given with --phase1-likelihood legacy-masked")
        return None
    if getattr(args, "observation_mode", "masked-rgb-expected-quad") != "masked-rgb-expected-quad":
        raise SystemExit("--phase1-likelihood occlusion-consistent needs --observation-mode masked-rgb-expected-quad")
    if getattr(args, "object_mask_constraint", "off") != "off":
        raise SystemExit("--phase1-likelihood occlusion-consistent and --object-mask-constraint are "
                         "different negative-ray models; choose one")
    if (getattr(args, "first_return_mode", "off") or "off") != "off":
        raise SystemExit("--phase1-likelihood occlusion-consistent with --first-return-mode: no joint "
                         "model has been derived; refusing to stack them")
    if (getattr(args, "rgb_support", "full") or "full") != "full":
        raise SystemExit("--phase1-likelihood occlusion-consistent sets the RGB support itself (M = 1)")
    try:
        return OcclusionLikelihood(float(getattr(args, "negative_ray_reliability", 1.0)),
                                   getattr(args, "negative_ray_mask_edge", "unknown"),
                                   bool(getattr(args, "deterministic", False)))
    except ValueError as ex:
        raise SystemExit(f"occlusion likelihood: {ex}")
