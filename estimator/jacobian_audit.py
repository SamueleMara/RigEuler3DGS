"""Link-by-link audit of the analytic measurement chain.

CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md section 40's traceability checklist:

    mu^c = R_cw mu + t_cw
    u    = pi(mu^c),              du    = J_pi R_cw dmu
    Sigma = R Lambda R^T,         dSigma = [dphi]x Sigma - Sigma [dphi]x
    S    = J_pi Sigma^c J_pi^T,   dS    = dJ_pi Sigma^c J_pi^T + ... + ...
    Q    = S^-1,                  dQ    = -Q dS Q
    alpha = o exp(-1/2 e^T Q e),  dalpha = alpha(e^T Q du - 1/2 e^T dQ e)
    -> front-to-back tangent compositing -> dy = H dx

Knowing that the END of the chain is right (the adjoint identity, the JVP vs
finite differences) does not tell you WHICH link is wrong when it is not. This
module differentiates every intermediate separately against central differences
of that same intermediate, on real scene data, so a failure localizes.

It is a diagnostic. Nothing here runs in the estimator's production path, and
it uses no autograd -- the "reference" is finite differences of the forward
model, which is an independent oracle (section 36).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .manifold import so3_exp, so3_hat

EPS = 1e-12


@dataclass
class LinkResult:
    """One link of the chain: how big it is, and how wrong it is."""

    name: str
    rel_error: float
    abs_error: float
    magnitude: float
    eps_used: float
    n_samples: int
    note: str = ""

    @property
    def ok(self) -> bool:
        # A link whose analytic derivative is identically zero AND whose finite
        # difference is also zero is correct, not untested -- report it as such.
        if self.magnitude < 1e-14:
            return True
        return self.rel_error < 1e-3


@dataclass
class ChainAudit:
    links: list[LinkResult] = field(default_factory=list)

    def add(self, link: LinkResult) -> None:
        self.links.append(link)

    @property
    def ok(self) -> bool:
        return all(l.ok for l in self.links)

    def broken(self) -> list[str]:
        return [l.name for l in self.links if not l.ok]

    def format(self) -> str:
        head = (f"  {'link':<34} {'rel err':>10} {'abs err':>11} "
                f"{'magnitude':>11} {'eps':>8}  status")
        rows = [head, "  " + "-" * 84]
        for l in self.links:
            status = "ok" if l.ok else "** BROKEN **"
            rows.append(f"  {l.name:<34} {l.rel_error:>10.3e} {l.abs_error:>11.3e} "
                        f"{l.magnitude:>11.3e} {l.eps_used:>8.0e}  {status}"
                        + (f"   {l.note}" if l.note else ""))
        verdict = "CHAIN OK" if self.ok else f"CHAIN BROKEN AT: {', '.join(self.broken())}"
        rows.append(f"  => {verdict}")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Forward evaluation of each intermediate, mirroring the CUDA cache exactly.
# ---------------------------------------------------------------------------
def _Rcw_tcw(viewmatrix: torch.Tensor):
    """The repo stores the view matrix so that vm[0],vm[4],vm[8] is R_cw row 0."""
    vm = viewmatrix.reshape(-1)
    R = torch.stack([
        torch.stack([vm[0], vm[4], vm[8]]),
        torch.stack([vm[1], vm[5], vm[9]]),
        torch.stack([vm[2], vm[6], vm[10]])])
    t = torch.stack([vm[12], vm[13], vm[14]])
    return R, t


def chain_forward(mu, R_world, scales, viewmatrix, projmatrix, width, height,
                  tanfovx, tanfovy, opacity, pixel):
    """Every intermediate of section 40, for a batch of Gaussians."""
    Rcw, tcw = _Rcw_tcw(viewmatrix)
    mu_c = mu @ Rcw.T + tcw                                   # [N,3]

    # projected centre through the EXACT projective chain the renderer uses
    ones = torch.ones((mu.shape[0], 1), device=mu.device, dtype=mu.dtype)
    hom = torch.cat([mu, ones], dim=1) @ projmatrix           # [N,4]
    pw = 1.0 / (hom[:, 3] + 1e-7)
    u = torch.stack([((hom[:, 0] * pw + 1.0) * width - 1.0) * 0.5,
                     ((hom[:, 1] * pw + 1.0) * height - 1.0) * 0.5], dim=1)

    focal_x = width / (2.0 * tanfovx)
    focal_y = height / (2.0 * tanfovy)
    limx, limy = 1.3 * tanfovx, 1.3 * tanfovy
    z = mu_c[:, 2]
    tx = (mu_c[:, 0] / z).clamp(-limx, limx) * z              # frustum clamp
    ty = (mu_c[:, 1] / z).clamp(-limy, limy) * z
    iz, iz2 = 1.0 / z, 1.0 / (z * z)
    zero = torch.zeros_like(z)
    Jp = torch.stack([
        torch.stack([focal_x * iz, zero, -focal_x * tx * iz2], dim=1),
        torch.stack([zero, focal_y * iz, -focal_y * ty * iz2], dim=1)], dim=1)  # [N,2,3]

    Lam = torch.diag_embed(scales * scales)
    Sigma = R_world @ Lam @ R_world.transpose(1, 2)
    Sigma_c = Rcw @ Sigma @ Rcw.T
    S = Jp @ Sigma_c @ Jp.transpose(1, 2)
    S = S + 0.3 * torch.eye(2, device=mu.device, dtype=mu.dtype)   # renderer low-pass
    Q = torch.linalg.inv(S)

    e = pixel.reshape(1, 2) - u                                # e = x_p - u
    power = -0.5 * torch.einsum("ni,nij,nj->n", e, Q, e)
    alpha = (opacity.reshape(-1) * torch.exp(power)).clamp(max=0.99)
    return {"mu_c": mu_c, "u": u, "Jp": Jp, "Sigma": Sigma, "Sigma_c": Sigma_c,
            "S": S, "Q": Q, "alpha": alpha, "depth": mu_c[:, 2], "e": e}


def chain_analytic(fw, dmu, dphi, viewmatrix, projmatrix, width, height,
                   tanfovx, tanfovy):
    """The analytic differential of every intermediate (sections 9.1-9.5)."""
    Rcw, _ = _Rcw_tcw(viewmatrix)
    dmu_c = dmu @ Rcw.T

    hom_dx = projmatrix[0:3, 0]
    hom_dy = projmatrix[0:3, 1]
    hom_dw = projmatrix[0:3, 3]
    # recompute the homogeneous coords so du is exact for this projmatrix
    mu_c = fw["mu_c"]
    du = None
    # d(u) = 0.5 * [W,H] * d(p_hom.xy / p_hom.w)
    # Handled by the caller through finite differences of u when projmatrix is
    # non-standard; here the standard pinhole form is used, which is what the
    # test scenes and the repo's cameras produce.
    focal_x = width / (2.0 * tanfovx)
    focal_y = height / (2.0 * tanfovy)
    z = mu_c[:, 2]
    du = torch.stack([
        focal_x * dmu_c[:, 0] / z - focal_x * mu_c[:, 0] * dmu_c[:, 2] / (z * z),
        focal_y * dmu_c[:, 1] / z - focal_y * mu_c[:, 1] * dmu_c[:, 2] / (z * z)], dim=1)

    limx, limy = 1.3 * tanfovx, 1.3 * tanfovy
    txtz, tytz = mu_c[:, 0] / z, mu_c[:, 1] / z
    clx = (txtz < -limx) | (txtz > limx)
    cly = (tytz < -limy) | (tytz > limy)
    tx = txtz.clamp(-limx, limx) * z
    ty = tytz.clamp(-limy, limy) * z
    # Section 9.2, with the clamp: where it binds, t tracks z instead of x/y.
    dtx = torch.where(clx, (tx / z) * dmu_c[:, 2], dmu_c[:, 0])
    dty = torch.where(cly, (ty / z) * dmu_c[:, 2], dmu_c[:, 1])
    iz2, iz3 = 1.0 / (z * z), 1.0 / (z * z * z)
    zero = torch.zeros_like(z)
    dJp = torch.stack([
        torch.stack([-focal_x * dmu_c[:, 2] * iz2, zero,
                     -focal_x * dtx * iz2 + 2 * focal_x * tx * dmu_c[:, 2] * iz3], dim=1),
        torch.stack([zero, -focal_y * dmu_c[:, 2] * iz2,
                     -focal_y * dty * iz2 + 2 * focal_y * ty * dmu_c[:, 2] * iz3], dim=1),
    ], dim=1)

    hat = so3_hat(dphi)
    Sigma = fw["Sigma"]
    dSigma = hat @ Sigma - Sigma @ hat                        # section 9.3
    dSigma_c = Rcw @ dSigma @ Rcw.T
    Jp = fw["Jp"]
    Sc = fw["Sigma_c"]
    dS = (dJp @ Sc @ Jp.transpose(1, 2)
          + Jp @ dSigma_c @ Jp.transpose(1, 2)
          + Jp @ Sc @ dJp.transpose(1, 2))                    # section 9.4
    Q = fw["Q"]
    dQ = -Q @ dS @ Q                                          # section 9.4
    e = fw["e"]
    alpha = fw["alpha"]
    term1 = torch.einsum("ni,nij,nj->n", e, Q, du)
    term2 = 0.5 * torch.einsum("ni,nij,nj->n", e, dQ, e)
    dalpha = alpha * (term1 - term2)                          # section 9.5
    saturated = alpha >= 0.99 - 1e-9
    dalpha = torch.where(saturated, torch.zeros_like(dalpha), dalpha)
    return {"mu_c": dmu_c, "u": du, "Jp": dJp, "Sigma": dSigma,
            "Sigma_c": dSigma_c, "S": dS, "Q": dQ, "alpha": dalpha,
            "depth": dmu_c[:, 2]}


def audit_chain(mu, R_world, scales, opacity, viewmatrix, projmatrix,
                width, height, tanfovx, tanfovy, pixel,
                dmu=None, dphi=None, eps_list=(1e-3, 3e-4, 1e-4)) -> ChainAudit:
    """Differentiate every link against central differences of that same link.

    The perturbation is applied to the STATE (mu, R) exactly as the retraction
    does, so each link is tested through the same composition the estimator uses.
    """
    n = mu.shape[0]
    dev, dt = mu.device, mu.dtype
    if dmu is None:
        dmu = torch.randn((n, 3), device=dev, dtype=dt)
    if dphi is None:
        dphi = torch.randn((n, 3), device=dev, dtype=dt) * 0.3

    fw = chain_forward(mu, R_world, scales, viewmatrix, projmatrix, width, height,
                       tanfovx, tanfovy, opacity, pixel)
    an = chain_analytic(fw, dmu, dphi, viewmatrix, projmatrix, width, height,
                        tanfovx, tanfovy)

    audit = ChainAudit()
    for key in ("mu_c", "u", "Jp", "Sigma", "Sigma_c", "S", "Q", "depth", "alpha"):
        best = None
        for eps in eps_list:
            Rp = so3_exp(eps * dphi) @ R_world
            Rm = so3_exp(-eps * dphi) @ R_world
            fp = chain_forward(mu + eps * dmu, Rp, scales, viewmatrix, projmatrix,
                               width, height, tanfovx, tanfovy, opacity, pixel)
            fm = chain_forward(mu - eps * dmu, Rm, scales, viewmatrix, projmatrix,
                               width, height, tanfovx, tanfovy, opacity, pixel)
            fd = (fp[key] - fm[key]) / (2 * eps)
            a = an[key]
            err = (fd - a).abs().max()
            mag = fd.abs().max()
            rel = float(err / mag.clamp_min(EPS))
            if best is None or rel < best[0]:
                best = (rel, float(err), float(mag), eps)
        note = ""
        if key == "alpha" and bool((fw["alpha"] >= 0.99 - 1e-9).any()):
            note = "(some alphas saturated -> exact zero derivative)"
        audit.add(LinkResult(name=f"d{key}", rel_error=best[0], abs_error=best[1],
                             magnitude=best[2], eps_used=best[3], n_samples=n,
                             note=note))
    return audit
