"""Explicit radiance modes for the measurement model.

CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md section 12.

The whole analytic measurement chain in ``iekf.cu`` assumes

    delta c_i = 0

i.e. a Gaussian's emitted colour does not change when the Gaussian moves. That
assumption is NOT automatically true just because the SH coefficients are
frozen: SH radiance is a function of the camera-to-Gaussian DIRECTION, so a
moving Gaussian changes its own view direction and therefore its evaluated
colour, with a derivative d c_i / d mu_i that the Jacobian does not contain.

Section 12 therefore demands two *explicit* modes, and -- section 12.3 -- that
the flag genuinely change the measurement model rather than merely being read:

  STRICT ("dc_only")
      Radiance is the SH DC term alone. It has no directional dependence at
      all, so delta c_i = 0 holds BY CONSTRUCTION and the analytic Jacobian is
      exact in this respect. This is the required default (section 12.1).

  VIEW_DEPENDENT ("view_dependent")
      Full SH evaluated at the current geometry, re-evaluated at every
      linearization. The rendered image, and hence the residual, differs from
      strict mode -- so the flag demonstrably changes the measurement model.
      But d c_i / d mu_i is NOT in H, so the linearization is inconsistent by
      exactly the amount this module measures. Section 12.2 forbids using this
      mode until that derivative is implemented and finite-difference
      validated, so it is refused unless the caller explicitly acknowledges it.

``unmodelled_color_jacobian`` is the quantitative evidence section 12.1 asks
for: a finite-difference measurement of the term the Jacobian omits. In strict
mode it must be exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.sh_utils import C0, eval_sh

STRICT = "dc_only"
VIEW_DEPENDENT = "view_dependent"
EPS = 1e-12


@dataclass
class RadianceModel:
    """Frozen SH coefficients plus the mode that says how they are evaluated.

    The coefficients themselves are frozen in EVERY mode (section 3): what the
    mode selects is which part of them reaches the renderer, and whether the
    result depends on the Gaussian's position.
    """

    sh: torch.Tensor                 # [N, (deg+1)^2, 3] frozen SH coefficients
    active_degree: int               # the degree actually evaluated
    mode: str = STRICT

    def __post_init__(self) -> None:
        if self.mode not in (STRICT, VIEW_DEPENDENT):
            raise ValueError(f"unknown radiance mode {self.mode!r}; "
                             f"expected {STRICT!r} or {VIEW_DEPENDENT!r}")
        sh = torch.as_tensor(self.sh)
        if sh.dim() != 3 or sh.shape[-1] != 3:
            raise ValueError(f"SH coefficients must be [N,K,3], got {tuple(sh.shape)}")
        max_deg = int(round(sh.shape[1] ** 0.5)) - 1
        if self.active_degree > max_deg:
            raise ValueError(
                f"active_degree {self.active_degree} needs {(self.active_degree + 1) ** 2} "
                f"coefficients but only {sh.shape[1]} are stored")
        object.__setattr__(self, "sh", sh)

    @property
    def is_strict(self) -> bool:
        return self.mode == STRICT

    @property
    def num_gaussians(self) -> int:
        return int(self.sh.shape[0])

    @staticmethod
    def from_gaussian_model(model, mode: str = STRICT,
                            active_degree: int | None = None) -> "RadianceModel":
        """Read the frozen SH block straight off a 3DGS ``GaussianModel``."""
        sh = torch.cat([model.get_features_dc, model.get_features_rest], dim=1) \
            if hasattr(model, "get_features_dc") else model.get_features
        sh = sh.detach()
        max_deg = int(round(sh.shape[1] ** 0.5)) - 1
        if mode == STRICT:
            deg = 0
        elif active_degree is None:
            deg = int(getattr(model, "active_sh_degree", max_deg))
        else:
            deg = int(active_degree)
        return RadianceModel(sh=sh, active_degree=deg, mode=mode)

    def evaluate(self, mu: torch.Tensor, campos: torch.Tensor | None) -> torch.Tensor:
        """The [N,3] RGB that enters the renderer for this state and camera.

        Strict mode ignores ``mu`` and ``campos`` entirely -- that independence
        IS the delta c = 0 guarantee, not a shortcut.
        """
        if self.is_strict:
            # SH DC term: c = C0 * sh_0 + 0.5, with no directional argument.
            rgb = C0 * self.sh[:, 0, :] + 0.5
            return rgb.clamp_min(0.0)
        if campos is None:
            raise ValueError(
                "view_dependent radiance needs a camera position; strict mode "
                "is the one that may be evaluated without a camera")
        dirs = mu - campos.reshape(1, 3).to(device=mu.device, dtype=mu.dtype)
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(EPS)
        sh = self.sh.transpose(1, 2)                       # [N,3,K] for eval_sh
        rgb = eval_sh(self.active_degree, sh, dirs) + 0.5
        return rgb.clamp_min(0.0)

    def describe(self) -> dict:
        """Section 12.1's 'log the exact radiance mode'."""
        return {
            "radiance_mode": self.mode,
            "strict_frozen_radiance": self.is_strict,
            "active_sh_degree": int(self.active_degree),
            "stored_sh_coefficients": int(self.sh.shape[1]),
            "delta_c_is_zero_by_construction": self.is_strict,
            "color_jacobian_in_H": False,
            "note": (
                "strict mode evaluates the DC term only, so radiance has no "
                "view-direction dependence and the analytic Jacobian's "
                "delta_c = 0 assumption is exact"
                if self.is_strict else
                "view-dependent mode re-evaluates SH at the current geometry; "
                "d c / d mu is NOT present in H, so the linearization is "
                "inconsistent by that term (section 12.2)"),
        }


@torch.no_grad()
def unmodelled_color_jacobian(radiance: RadianceModel, mu: torch.Tensor,
                              campos: torch.Tensor, delta_mu: torch.Tensor,
                              eps: float = 1e-4) -> dict:
    """Measure the colour derivative the Jacobian leaves out (section 12.1).

    Central finite difference of the evaluated radiance along ``delta_mu``:

        dc ~= [ c(mu + eps*dmu) - c(mu - eps*dmu) ] / (2 eps)

    In strict mode this must be EXACTLY zero -- that is the test section 12.1
    requires ("moving a Gaussian with fixed alpha geometry does not introduce
    an unmodelled SH color derivative"). In view-dependent mode it is the size
    of the modelling error that section 12.2 refuses to tolerate silently.
    """
    d = delta_mu / delta_mu.norm(dim=1, keepdim=True).clamp_min(EPS)
    c_plus = radiance.evaluate(mu + eps * d, campos)
    c_minus = radiance.evaluate(mu - eps * d, campos)
    dc = (c_plus - c_minus) / (2 * eps)
    base = radiance.evaluate(mu, campos)
    per = dc.norm(dim=1)
    return {
        "radiance_mode": radiance.mode,
        "max_abs_dc_dmu": float(per.max()),
        "mean_abs_dc_dmu": float(per.mean()),
        "max_color": float(base.abs().max()),
        # The honest relative figure: how big is the omitted colour change per
        # unit of motion compared with the colour itself.
        "relative_unmodelled": float(per.max() / max(float(base.abs().max()), EPS)),
        "is_exactly_zero": bool(float(per.max()) == 0.0),
    }


def resolve_radiance_mode(strict_frozen_radiance: bool,
                          allow_unvalidated_view_dependent: bool = False) -> str:
    """Section 12.3's flag contract.

    ``iekf_strict_frozen_radiance=True`` -> DC-only, the required default.
    ``False`` -> view-dependent, which section 12.2 forbids until the analytic
    colour derivative exists. It is refused with an explanation rather than
    silently run with an inconsistent Jacobian.
    """
    if strict_frozen_radiance:
        return STRICT
    if not allow_unvalidated_view_dependent:
        raise NotImplementedError(
            "iekf_strict_frozen_radiance=False selects section 12.2's "
            "view-dependent radiance mode. That mode requires the analytic "
            "colour derivative d c_i / d mu_i in the measurement Jacobian, "
            "which is not implemented, so the linearization would be "
            "inconsistent by the amount unmodelled_color_jacobian() reports. "
            "Section 12.2: 'Do not use this mode until the analytic color "
            "derivative is implemented and finite-difference validated.' "
            "Pass allow_unvalidated_view_dependent=True only to measure that "
            "error deliberately.")
    return VIEW_DEPENDENT
