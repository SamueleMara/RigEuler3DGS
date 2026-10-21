"""Masked-RGB / foreground-valid-depth observation for the IEKFs (20 Oct revision).

CODEX_IMPLEMENT_EXACT_MASKED_RGB_VALID_DEPTH_IEKF_2026-10-20.md; theory in
Bidirectional_Articulated_3DGS_IEKF_Masked_RGB_Valid_Depth_2026-10-20.pdf,
section "Exact masked RGB-background compositing and foreground-valid depth".

For camera c, frame t, pixel p (native resolution):

    P      = M * I_obs                    premultiplied observed object colour
    Y(b)   = P + (1 - M) * b              observation composited on a KNOWN RGB background b
    C(x;b) = F(x) + (1 - A(x)) * b        the renderer's own prediction on the same b
    V_used = V_orig AND M                 depth: original valid depth on the object only

Modes (--observation-mode):
  original                  the old full-frame observation, unchanged (default)
  masked-rgb-constant       one known constant b, quadratic RGB error ||Y(b) - C(x;b)||^2_W
  masked-rgb-expected-quad  b ~ Uniform[0,1]^3 (independent channels), and the EXACT
                            expectation of the quadratic RGB error:
        E_b[1/2 e^T W e] = 1/2 e_bar^T W e_bar + 1/2 kappa (A - M)^2
        e_bar = F - P + (M - A) b_bar,  b_bar = (1/2,1/2,1/2),
        kappa = trace(W Sigma_b) = trace(W)/12
    -> one render at b_bar plus an analytic accumulated-alpha term.

RGB precision W is FIXED in these modes (never residual-reweighted): Phase 1 uses
the configured IEKFConfig.rgb_sigma (W = I / sigma^2, so kappa = 1/(4 sigma^2));
Phase 2/3 use their existing fixed sensor + model-error variance without the
Student-t weight. The Student-t IRLS weight of the original mode cannot be
carried through the exact quadratic expectation, so it is not applied to RGB
here. Depth keeps its existing robust precision, on V_used only.

Nothing here touches training, opacity, scales, colours or the camera model, and
nothing calls the hard negative-ray QP (visibility_constraint.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

MODES = ("original", "masked-rgb-constant", "masked-rgb-expected-quad")
DEFAULT_MODE = "masked-rgb-expected-quad"      # the motion pipeline's default (2026-10-21)
DISTRIBUTIONS = ("uniform01",)
UNIFORM01_MEAN = 0.5
UNIFORM01_VAR = 1.0 / 12.0


@dataclass(frozen=True)
class MaskedObservationModel:
    mode: str
    background: tuple                    # the render background: b (constant) or b_bar (expected)
    distribution: str | None = None      # expected-quad only
    rgb_sigma: float | None = None       # the declared fixed RGB sigma (Phase 1)

    def __post_init__(self):
        if self.mode not in MODES[1:]:
            raise ValueError(f"MaskedObservationModel is for {MODES[1:]}, not {self.mode!r}")
        if len(self.background) != 3 or not all(0.0 <= float(c) <= 1.0 for c in self.background):
            raise ValueError(f"background must be 3 values in [0,1], got {self.background}")
        if self.mode == "masked-rgb-expected-quad" and self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"expected-quad needs --rgb-background-distribution in {DISTRIBUTIONS}")
        if self.rgb_sigma is not None and not float(self.rgb_sigma) > 0:
            raise ValueError("rgb_sigma must be positive")

    @property
    def alpha_term(self) -> bool:
        return self.mode == "masked-rgb-expected-quad"

    @property
    def background_variance(self) -> float:
        """Sigma_b = var * I (independent channels)."""
        return UNIFORM01_VAR if self.alpha_term else 0.0

    def background_tensor(self, device) -> torch.Tensor:
        return torch.tensor([float(c) for c in self.background], device=device, dtype=torch.float32)

    def kappa(self, p_rgb: torch.Tensor) -> torch.Tensor:
        """kappa[p] = trace(W[p] Sigma_b) for a diagonal per-channel W ([3,H,W] or broadcastable)."""
        if not self.alpha_term:
            return torch.zeros_like(p_rgb[0] if p_rgb.dim() == 3 else p_rgb)
        return p_rgb.sum(0) * self.background_variance if p_rgb.dim() == 3 else 3.0 * p_rgb * self.background_variance

    def to_dict(self) -> dict:
        return {"observation_mode": self.mode, "render_background": [float(c) for c in self.background],
                "background_distribution": self.distribution,
                "background_covariance": (f"{self.background_variance:.10g} * I3" if self.alpha_term else None),
                "rgb_precision": "fixed W (no residual reweighting of RGB)",
                "alpha_term": ("0.5 * kappa * (A - M)^2, kappa = trace(W Sigma_b)" if self.alpha_term else None),
                "depth": "original metric depth on V_orig AND M only; zero precision elsewhere",
                "note": ("the artificial background is a designed contrast cost, not an independently "
                         "sensed observation; information/covariance is conditional on this objective")}


@dataclass
class PreparedRGBD:
    """One camera at one frame. Raw observations are not modified."""
    rgb_foreground_premult: torch.Tensor   # [3,H,W] P = M * I
    object_coverage: torch.Tensor          # [H,W] float, M (binary at native resolution)
    depth_observed: torch.Tensor           # [H,W] original units; NEVER read where not depth_valid
    depth_valid: torch.Tensor              # [H,W] bool, V_orig AND M
    camera: str = ""
    original_frame: int = -1

    def target(self, b: torch.Tensor) -> torch.Tensor:
        """Y(b) = P + (1 - M) b."""
        return self.rgb_foreground_premult + (1.0 - self.object_coverage)[None] * b.reshape(3, 1, 1)

    def to(self, device) -> "PreparedRGBD":
        return PreparedRGBD(self.rgb_foreground_premult.to(device), self.object_coverage.to(device),
                            self.depth_observed.to(device), self.depth_valid.to(device),
                            self.camera, self.original_frame)


def prepare(rgb: torch.Tensor, depth: torch.Tensor, depth_valid: torch.Tensor, mask: torch.Tensor,
            camera: str = "", original_frame: int = -1) -> PreparedRGBD:
    """P = M I, V_used = V_orig AND M AND finite(depth) AND depth > 0 (positive camera Z
    only: zero / negative / NaN are never a measurement). Binary mask, matching sizes."""
    if rgb.dim() != 3 or rgb.shape[0] != 3:
        raise ValueError(f"{camera} frame {original_frame}: rgb must be [3,H,W], got {tuple(rgb.shape)}")
    H, W = rgb.shape[1], rgb.shape[2]
    for name, t in (("depth", depth), ("depth_valid", depth_valid), ("mask", mask)):
        if tuple(t.shape) != (H, W):
            raise ValueError(f"{camera} frame {original_frame}: {name} {tuple(t.shape)} vs rgb {(H, W)}")
    if mask.dtype != torch.bool:
        raise TypeError(f"{camera} frame {original_frame}: the native mask must be bool (binary)")
    if not bool(torch.isfinite(rgb).all()):
        raise ValueError(f"{camera} frame {original_frame}: non-finite RGB")
    dev = rgb.device
    M = mask.to(dev)
    Mf = M.to(rgb.dtype)
    d = depth.to(dev)
    valid = depth_valid.to(dev).bool() & M & torch.isfinite(d) & (d > 0)
    return PreparedRGBD(rgb * Mf[None], Mf, depth.to(dev), valid, camera, int(original_frame))


def area_downsample(rgb: torch.Tensor, mask: torch.Tensor, factor: int):
    """Coverage-correct resize: (R(M I), R(M)) with R the area mean over factor x factor
    blocks. Y_resized(b) = R(M I) + (1 - R(M)) b is then exactly the area mean of the
    native composite for every constant b (no dark edge halo). Resizing I and M separately
    and multiplying afterwards is NOT equivalent and is not offered."""
    Mf = mask.to(rgb.dtype)
    pool = torch.nn.functional.avg_pool2d
    P = pool((rgb * Mf[None]).unsqueeze(0), factor).squeeze(0)
    Mr = pool(Mf[None, None], factor)[0, 0]
    return P, Mr


def load_object_mask(path: Path, height: int, width: int) -> torch.Tensor:
    """Strict: the file must be binary (0/255 or 0/1, identical channels) at native size."""
    from PIL import Image
    im = Image.open(path)
    if im.size != (width, height):
        raise SystemExit(f"{path}: mask {im.size} but the camera image is {(width, height)}")
    a = np.asarray(im)
    if a.ndim == 3:
        if not all(np.array_equal(a[..., 0], a[..., k]) for k in range(1, min(a.shape[2], 3))):
            raise SystemExit(f"{path}: colour mask with differing channels is not a binary mask")
        a = a[..., 0]
    vals = np.unique(a)
    if not (set(vals.tolist()) <= {0, 255} or set(vals.tolist()) <= {0, 1}):
        raise SystemExit(f"{path}: non-binary mask values {vals[:8].tolist()}... -- refusing to threshold")
    return torch.from_numpy(a > 0)


def load_object_masks(mask_root: Path, cam_names, original_frames, height: int, width: int):
    """[frame][camera] bool masks, matched by camera name and ORIGINAL chronological frame."""
    out = []
    for f in original_frames:
        row = []
        for c in cam_names:
            cands = [Path(mask_root) / c / f"frame_{int(f):06d}{e}" for e in (".png", ".jpg", ".jpeg")]
            hits = [q for q in cands if q.is_file()]
            if not hits:
                raise SystemExit(f"masked observation: no mask for {c} frame {int(f)} under {mask_root}")
            if len(hits) > 1:
                raise SystemExit(f"masked observation: ambiguous masks {[h.name for h in hits]} for {c} frame {int(f)}")
            row.append(load_object_mask(hits[0], height, width))
        out.append(row)
    return out


@dataclass
class MaskedSequence:
    """The model plus every frame's masks, in PROCESSING order."""
    model: MaskedObservationModel
    masks: list                      # [frame][camera] bool CPU
    cameras: list
    original_frames: list
    mask_root: str = ""
    report: list = field(default_factory=list)

    def prepare_frame(self, k: int, obs, device=None) -> list:
        """obs: the loader's per-camera (rgb, depth, valid) tuples for processing index k."""
        if len(obs) != len(self.masks[k]):
            raise AssertionError(f"frame {k}: {len(obs)} observations but {len(self.masks[k])} masks")
        out = []
        for ci, (o, m) in enumerate(zip(obs, self.masks[k])):
            rgb, d = o[0], o[1]
            v = o[2] if len(o) > 2 and o[2] is not None else (d > 0)
            dev = device if device is not None else rgb.device
            out.append(prepare(rgb.to(dev), d.to(dev), v.to(dev), m.to(dev),
                               self.cameras[ci], self.original_frames[k]))
        return out

    def counts(self) -> dict:
        tot = sum(int(m.numel()) for row in self.masks for m in row)
        fg = sum(int(m.sum()) for row in self.masks for m in row)
        return {"frames": len(self.masks), "cameras": len(self.cameras), "pixels": tot,
                "foreground_pixels": fg, "foreground_fraction": fg / max(tot, 1)}


# ---------------------------------------------------------------------------
# the measurement terms on one linearization
# ---------------------------------------------------------------------------
def residuals(view, ob: PreparedRGBD, b: torch.Tensor):
    """r_C = Y(b) - C(x;b) on the whole image; r_D = d - D on V_used (exactly 0 and never
    computed from the stored value elsewhere); r_A = M - A."""
    r_rgb = ob.target(b) - view.rgb
    r_d = torch.where(ob.depth_valid, ob.depth_observed - view.depth, torch.zeros_like(view.depth))
    r_a = ob.object_coverage - view.alpha
    return r_rgb, r_d, r_a


def depth_precision(view, ob: PreparedRGBD, r_d: torch.Tensor, cfg):
    """The existing depth precision (sigma model, Student-t IRLS) on V_used AND the
    pre-existing coverage gate alpha > min_alpha_for_depth (unchanged, documented)."""
    from .measurement import build_precision
    valid = ob.depth_valid & (view.alpha > cfg.min_alpha_for_depth)
    if not cfg.use_depth:
        return torch.zeros_like(r_d), valid
    rng = torch.where(valid, ob.depth_observed, torch.zeros_like(ob.depth_observed))
    p_d = build_precision(
        r_d.unsqueeze(0), valid_mask=valid.unsqueeze(0), sigma_rgb=(),
        depth_sigma_model=(cfg.depth_sigma_model if ":" in str(cfg.depth_sigma_model)
                           else {"kind": cfg.depth_sigma_model, "a": cfg.depth_sigma_a,
                                 "b": cfg.depth_sigma_b}),
        depth_range=rng.unsqueeze(0), robust=cfg.robust_measurement, nu=cfg.student_t_nu,
        layout=("depth",), channel_dim=0).squeeze(0)
    return p_d, valid


def phase1_camera_terms(views, prepared, cfg, model: MaskedObservationModel, model_error=None,
                        cam_ids=None):
    """Per camera `CameraTerms` (estimator/observation_terms.py) of the masked modes:

        RGB    fixed quadratic on Y(b) - C(x; b) over the whole image,
               W = 1/(sigma_rgb^2 [+ fixed model-error variance])
        depth  robust (Student-t NLL) on V_orig AND M, with the pre-existing alpha gate
        alpha  1/2 kappa (M - A)^2, kappa = trace(W)/12   (expected-quad only)
    """
    from .observation_terms import camera_terms
    if len(views) != len(prepared):
        raise AssertionError(f"{len(views)} views but {len(prepared)} prepared observations")
    if model.rgb_sigma is not None and abs(float(model.rgb_sigma) - float(cfg.rgb_sigma)) > 0:
        raise AssertionError(f"observation model sigma {model.rgb_sigma} != IEKF rgb_sigma {cfg.rgb_sigma}")
    out = []
    for j, (v, ob) in enumerate(zip(views, prepared)):
        if not isinstance(ob, PreparedRGBD):
            raise TypeError("the masked observation mode needs PreparedRGBD observations")
        ob = ob.to(v.rgb.device)
        c = j if cam_ids is None else cam_ids[j]
        mv_rgb = None if model_error is None else model_error.rgb_var[c]
        mv_d = None if model_error is None else model_error.depth_var[c]
        out.append(camera_terms(v, ob.target(model.background_tensor(v.rgb.device)), ob.depth_observed,
                                ob.depth_valid, cfg, rgb_robust=False, rgb_model_var=mv_rgb,
                                depth_model_var=mv_d, coverage=ob.object_coverage,
                                kappa_fn=model.kappa if model.alpha_term else None))
    return out


def phase1_terms(views, prepared, cfg, model: MaskedObservationModel):
    """Per camera ((p_rgb, p_d, r_rgb, r_d), (kappa, r_A)) -- the solver's view of
    phase1_camera_terms (kappa is zero in the constant mode)."""
    terms = phase1_camera_terms(views, prepared, cfg, model)
    alpha = [(t.kappa if t.kappa is not None else torch.zeros_like(t.r_d),
              t.r_a if t.r_a is not None else torch.zeros_like(t.r_d)) for t in terms]
    return [t.as_prec4() for t in terms], alpha


def phase1_energy(views, prepared, cfg, model: MaskedObservationModel) -> dict:
    """The masked objective's measurement part (RGB, robust depth NLL, alpha) and depth coverage."""
    from .observation_terms import summarize
    return summarize(phase1_camera_terms(views, prepared, cfg, model))


def energy_terms(prec, alpha) -> dict:
    """0.5 sum W r^2 (RGB), 0.5 sum W_D r_D^2 (depth), 0.5 sum kappa r_A^2 (alpha), float64."""
    rgb = dep = alp = 0.0
    for (p_rgb, p_d, r_rgb, r_d), (kap, r_a) in zip(prec, alpha):
        rgb += 0.5 * float((p_rgb.double() * r_rgb.double().square()).sum())
        dep += 0.5 * float((p_d.double() * r_d.double().square()).sum())
        alp += 0.5 * float((kap.double() * r_a.double().square()).sum())
    return {"rgb": rgb, "depth": dep, "alpha": alp}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_cli(ap) -> None:
    ap.add_argument("--observation-mode", choices=MODES, default=DEFAULT_MODE,
                    help="masked-rgb-expected-quad (DEFAULT: masked object RGB on a Uniform[0,1]^3 "
                         "background, exact expected quadratic cost, depth on the object only; needs "
                         "<source>/mask) | masked-rgb-constant (ablation, needs --rgb-background-constant) "
                         "| original (explicit control: full-frame RGB-D, no masks)")
    ap.add_argument("--rgb-background-constant", type=float, nargs=3, default=None, metavar=("R", "G", "B"),
                    help="the known background colour of masked-rgb-constant (required there, refused elsewhere)")
    ap.add_argument("--rgb-background-distribution", choices=DISTRIBUTIONS, default=None,
                    help="masked-rgb-expected-quad: the background distribution (uniform01 = independent "
                         "Uniform[0,1] channels; the default when that mode is chosen)")


def model_from_args(args, cfg=None) -> MaskedObservationModel | None:
    mode = getattr(args, "observation_mode", DEFAULT_MODE)
    const = getattr(args, "rgb_background_constant", None)
    dist = getattr(args, "rgb_background_distribution", None)
    if mode == "original":
        if const is not None or dist is not None:
            raise SystemExit("--rgb-background-* given without a masked --observation-mode")
        return None
    if getattr(args, "object_mask_constraint", "off") != "off":
        raise SystemExit(f"--observation-mode {mode} and --object-mask-constraint "
                         f"{args.object_mask_constraint} are different measurement models; "
                         "choose one")
    sigma = None if cfg is None else float(cfg.rgb_sigma)
    if mode == "masked-rgb-constant":
        if const is None:
            raise SystemExit("masked-rgb-constant needs --rgb-background-constant R G B")
        if dist is not None:
            raise SystemExit("--rgb-background-distribution is for masked-rgb-expected-quad")
        return MaskedObservationModel(mode, tuple(float(c) for c in const), None, sigma)
    if const is not None:
        raise SystemExit("--rgb-background-constant is for masked-rgb-constant")
    return MaskedObservationModel(mode, (UNIFORM01_MEAN,) * 3, dist or "uniform01", sigma)


def preflight(args, source) -> None:
    """Fail BEFORE any model load / GPU work when the requested observation mode
    needs masks that are missing, ambiguous, mis-sized or non-binary (P0-1).
    The explicit original mode needs no masks and is never blocked here."""
    mode = getattr(args, "observation_mode", DEFAULT_MODE)
    if mode == "original":
        return
    if getattr(args, "object_mask_constraint", "off") != "off":
        raise SystemExit(f"--observation-mode {mode} and --object-mask-constraint "
                         f"{args.object_mask_constraint} are different measurement models; pass "
                         "--observation-mode original explicitly to use the hard constraint")
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from check_masks import check
    src = Path(source)
    root = Path(args.mask_source) if getattr(args, "mask_source", None) else src / "mask"
    if not root.is_dir():
        raise SystemExit(f"observation mode {mode}: mask root {root} does not exist. The default "
                         "masked-rgb-expected-quad mode needs whole-object masks; pass "
                         "--observation-mode original explicitly to run the unmasked control.")
    cams = getattr(args, "cameras", None)
    problems, info = check(src, root, int(cams) if isinstance(cams, int) or str(cams).isdigit() else None,
                           int(args.frames) if getattr(args, "frames", None) else None)
    if problems:
        raise SystemExit(f"observation mode {mode}: masks not ready under {root} "
                         f"({len(problems)} problem(s)), e.g. " + "; ".join(problems[:4]))


def observation_manifest(args, seq=None, cfg=None) -> dict:
    """What every stage writes next to its outputs: the effective observation model."""
    mode = getattr(args, "observation_mode", DEFAULT_MODE)
    hard = getattr(args, "object_mask_constraint", "off")
    out = {"observation_mode": mode, "default_mode": DEFAULT_MODE,
           "hard_negative_ray_qp": hard if hard != "off" else "off",
           "rgb_sigma": None if cfg is None else float(cfg.rgb_sigma)}
    if seq is None:
        out.update({"masks": None,
                    "rgb": "full-frame recorded RGB vs the object rendered on black (original control)",
                    "rgb_precision": "Student-t IRLS (GN) / Student-t NLL (objective), one sigma",
                    "depth_valid": "V_orig AND alpha > min_alpha_for_depth"})
        return out
    m = seq.model
    out.update({"background_distribution": m.distribution, "render_background_bbar": list(m.background),
                "background_covariance": (f"{m.background_variance:.10g} * I3" if m.alpha_term else None),
                "rgb": "Y(b) = M I + (1 - M) b over the whole image, rendered on the same b",
                "rgb_precision": "fixed quadratic W (no residual reweighting); kappa = trace(W)/12"
                                 if m.alpha_term else "fixed quadratic W (no residual reweighting)",
                "alpha_term": m.alpha_term,
                "depth_valid": "V_orig AND M AND finite(D), then the pre-existing alpha > min_alpha_for_depth gate",
                "depth_objective": "Student-t NLL on used samples (IRLS score in the GN system)",
                "masks": {"root": seq.mask_root, "cameras": seq.cameras,
                          "original_frames": seq.original_frames, **seq.counts()},
                "note": ("the artificial background is a designed contrast cost; the information it "
                         "adds is conditional on that objective, not a calibrated sensor covariance")})
    return out


def from_args(args, source, frames_raw, frame_ids, cfg=None) -> MaskedSequence | None:
    """The runner hook: None in the original mode, else the model and every frame's masks."""
    model = model_from_args(args, cfg)
    if model is None:
        return None
    cams = list(frame_ids.get("cameras") or [])
    ncam = len(frames_raw[0][0])
    if len(cams) < ncam:
        raise AssertionError(f"{len(cams)} camera names for {ncam} observed cameras")
    orig = [int(f["original_frame_index"]) if isinstance(f, dict) and "original_frame_index" in f
            else int(f["dataset_frame_index"]) if isinstance(f, dict) else int(f) for f in frame_ids["frames"]]
    if len(orig) != len(frames_raw):
        raise AssertionError(f"{len(orig)} frame ids for {len(frames_raw)} frames")
    rgb0 = frames_raw[0][0][0][0]
    H, W = int(rgb0.shape[-2]), int(rgb0.shape[-1])
    root = Path(args.mask_source) if getattr(args, "mask_source", None) else Path(source) / "mask"
    masks = load_object_masks(root, cams[:ncam], orig, H, W)
    seq = MaskedSequence(model, masks, cams[:ncam], orig, str(root))
    c = seq.counts()
    print(f"  observation mode {model.mode}: render background {list(model.background)}, masks {root}")
    print(f"    {c['frames']} frames x {c['cameras']} cameras, foreground {100 * c['foreground_fraction']:.1f}% "
          f"of pixels; RGB sigma {model.rgb_sigma} (fixed W)"
          + (f", kappa = 1/(4 sigma^2) = {0.25 / model.rgb_sigma ** 2:.4g}" if model.alpha_term and model.rgb_sigma else ""))
    return seq
