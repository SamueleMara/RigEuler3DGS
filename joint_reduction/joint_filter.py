"""Sections 29.3 and 30 -- reduced uncertainty and the reduced joint-state filter.

Once a part has been identified as prismatic or revolute, its Gaussians stop
being free states (section 30: "switch permanently away from free Gaussian state
updates for that part").  What survives is

    x^J_k = [q_k; qdot_k],      x^J_{k+1} = [[1, dt],[0, 1]] x^J_k + w_k,

and the dense RGB-D information collapses onto that 2-vector through the pose
tangents ``G`` of section 29:

    I_q = P_q^-1 + sum_p H_{p,q}^T R_p^-1 H_{p,q},
    b_q = sum_p H_{p,q}^T R_p^-1 r_p,
    dq  = I_q^-1 b_q.

Only 1x1 and 2x2 systems are ever inverted here; the dense ``H`` of rule 6 and
the dense full covariance of rule 7 are never formed.  No Adam, no autograd
(section 0 rules 4/5, section 36).

``dt`` is always the real inter-frame time (rule 11).  Nothing in this module
defaults it to 1.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from .prismatic import _as_scalar, fit_prismatic_axis, prismatic_pose_tangent, prismatic_transport
from .revolute import fit_revolute_axis, revolute_pose_tangent, revolute_transport
from .rigid_fit import _sym

EPS = 1.0e-12
JOINT_TYPES = ("prismatic", "revolute")


# ---------------------------------------------------------------------------
# Section 29.3 -- reduced uncertainty
# ---------------------------------------------------------------------------


@torch.no_grad()
def reduced_information(
    G: torch.Tensor, Lambda_pose: torch.Tensor | None = None
) -> torch.Tensor:
    """``Lambda_theta = sum_i G_i^T Lambda_i G_i`` (section 29.3).

    ``Lambda_pose`` is the *pose* information of each Gaussian (the 6x6 block
    marginal of the 12D error state kept by the IEKF).  It is contracted in full
    -- never replaced by its diagonal and never dropped -- because 29.3's whole
    point is that model reduction must not throw uncertainty away: a Gaussian
    that the renderer constrained only along the view ray must contribute only
    along the view ray.

    Accepts ``[N,6,6]``, a shared ``[6,6]``, or ``None`` (identity information,
    i.e. every Gaussian equally and isotropically trusted).  Returns ``[m,m]``,
    symmetrised because the einsum's round-off asymmetry would otherwise break
    the Cholesky factorisations downstream.
    """
    G = torch.as_tensor(G)
    if G.dim() != 3 or G.shape[1] != 6:
        raise ValueError(f"G must be [N,6,m], got {tuple(G.shape)}")
    n = G.shape[0]
    if Lambda_pose is None:
        out = torch.einsum("nia,nib->ab", G, G)
    else:
        lam = torch.as_tensor(Lambda_pose, device=G.device, dtype=G.dtype)
        if lam.dim() == 2 and lam.shape == (6, 6):
            lam = lam.expand(n, 6, 6)
        if lam.shape != (n, 6, 6):
            raise ValueError(f"Lambda_pose must be [{n},6,6] or [6,6], got {tuple(lam.shape)}")
        out = torch.einsum("nia,nij,njb->ab", G, _sym(lam), G)
    return _sym(out)


@torch.no_grad()
def reduced_covariance(Lambda_theta: torch.Tensor, *, rank_tol: float = 1.0e-12) -> torch.Tensor:
    """``P_theta ~= Lambda_theta^-1`` (section 29.3), via a truncated eigen-pinv.

    An unobservable joint direction (a part that did not move during the window)
    gives a singular ``Lambda_theta``; returning the pseudo-inverse keeps the
    reduction honest -- zero information maps to zero *confidence*, not to an
    infinite covariance that would poison the filter.
    """
    lam = _sym(torch.as_tensor(Lambda_theta))
    evals, evecs = torch.linalg.eigh(lam)
    tol = rank_tol * evals.max().clamp_min(0.0)
    keep = evals > tol
    inv = torch.where(keep, 1.0 / torch.where(keep, evals, torch.ones_like(evals)),
                      torch.zeros_like(evals))
    return _sym(evecs @ torch.diag(inv) @ evecs.transpose(-1, -2))


@torch.no_grad()
def reduce_pixel_jacobian(H_pose: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """``H_{p,q} = sum_{i in S_p} H^pose_{p,i} G_i`` -- the boxes of 29.1/29.2.

    ``H_pose`` is ``[P,N,6]`` = ``[H^mu | H^phi]`` for the Gaussians contributing
    to each pixel (``S_p`` is encoded as zero rows for non-contributors), ``G`` is
    ``[N,6,m]``.  With ``G^P = [e;0]`` this is ``sum_i H^mu_{p,i} e`` and with
    ``G^R = [e x (mu_i - a); e]`` it is
    ``sum_i [H^mu_{p,i}(e x (mu_i - a)) + H^phi_{p,i} e]``, verbatim.
    """
    H_pose = torch.as_tensor(H_pose)
    G = torch.as_tensor(G, device=H_pose.device, dtype=H_pose.dtype)
    if H_pose.dim() != 3 or H_pose.shape[2] != 6:
        raise ValueError(f"H_pose must be [P,N,6], got {tuple(H_pose.shape)}")
    if G.shape[0] != H_pose.shape[1] or G.shape[1] != 6:
        raise ValueError(f"G must be [{H_pose.shape[1]},6,m], got {tuple(G.shape)}")
    return torch.einsum("pni,nim->pm", H_pose, G)


# ---------------------------------------------------------------------------
# Section 30 -- the joint model that generates all Gaussian poses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointModel:
    """A prismatic or revolute joint: the only thing allowed to move a reduced part.

    After reduction the renderer may no longer deform Gaussians independently
    (section 30), so every centre/rotation it sees must come out of
    :meth:`transport` and every Jacobian out of :meth:`pose_tangent`.
    """

    joint_type: str
    axis: torch.Tensor                      # unit direction e
    point: torch.Tensor | None = None       # a point on the axis (revolute only)

    def __post_init__(self) -> None:
        if self.joint_type not in JOINT_TYPES:
            raise ValueError(f"joint_type must be one of {JOINT_TYPES}, got {self.joint_type!r}")
        if self.joint_type == "revolute" and self.point is None:
            raise ValueError("a revolute joint needs a point on its axis")

    def pose_tangent(self, mu: torch.Tensor) -> torch.Tensor:
        """``G`` at the current centres ``mu`` -- ``[N,6,1]`` (29.1 / 29.2)."""
        if self.joint_type == "prismatic":
            return prismatic_pose_tangent(self.axis, int(torch.as_tensor(mu).shape[0]))
        return revolute_pose_tangent(self.axis, self.point, mu)

    def transport(
        self, mu0: torch.Tensor, R0: torch.Tensor | None, q: torch.Tensor | float
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Rigidly generate ``(mu, R)`` at joint coordinate ``q`` from the rest pose."""
        if self.joint_type == "prismatic":
            return prismatic_transport(mu0, R0, self.axis, q)
        return revolute_transport(mu0, R0, self.axis, self.point, q)

    @staticmethod
    def fit(
        joint_type: str,
        mu0: torch.Tensor,
        mu_t: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple["JointModel", dict[str, Any]]:
        """Identify the joint geometry from two poses of the part (29.1 / 29.2)."""
        if joint_type == "prismatic":
            fit = fit_prismatic_axis(mu0, mu_t, weights)
            return JointModel("prismatic", fit["axis"]), fit
        if joint_type == "revolute":
            fit = fit_revolute_axis(mu0, mu_t, weights)
            return JointModel("revolute", fit["axis"], fit["point"]), fit
        raise ValueError(f"joint_type must be one of {JOINT_TYPES}, got {joint_type!r}")


# ---------------------------------------------------------------------------
# Section 30 -- the reduced filter
# ---------------------------------------------------------------------------


def _embed_information(information: torch.Tensor, device, dtype) -> torch.Tensor:
    """Lift a 1x1 (position-only) or 2x2 measurement information into the 2-state."""
    info = torch.as_tensor(information, device=device, dtype=dtype)
    if info.numel() == 1:
        out = torch.zeros(2, 2, device=device, dtype=dtype)
        out[0, 0] = info.reshape(())
        return out
    if info.shape == (2, 2):
        return _sym(info)
    raise ValueError(f"information must be scalar, [1,1] or [2,2], got {tuple(info.shape)}")


def _embed_vector(b: torch.Tensor, device, dtype) -> torch.Tensor:
    vec = torch.as_tensor(b, device=device, dtype=dtype).reshape(-1)
    if vec.numel() == 1:
        return torch.stack((vec[0], torch.zeros((), device=device, dtype=dtype)))
    if vec.numel() == 2:
        return vec
    raise ValueError(f"b must have 1 or 2 entries, got {vec.numel()}")


@torch.no_grad()
def solve_reduced_normal(
    information: torch.Tensor,
    b: torch.Tensor,
    prior_information: torch.Tensor | None = None,
    prior_offset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Section 30's normal system: ``(P^-1 + sum H^T R^-1 H) dq = b_q``.

    ``prior_offset`` is ``x_prior - x_current``; it is zero on the first inner
    iteration and non-zero afterwards, which is what keeps the IEKF prior *fixed*
    across inner iterations instead of being re-counted (rule 8).  Returns
    ``(delta, total_information)``.
    """
    info = _sym(torch.as_tensor(information))
    rhs = torch.as_tensor(b, device=info.device, dtype=info.dtype).reshape(info.shape[0])
    total = info.clone()
    if prior_information is not None:
        prior = _sym(torch.as_tensor(prior_information, device=info.device, dtype=info.dtype))
        total = total + prior
        if prior_offset is not None:
            rhs = rhs + prior @ torch.as_tensor(
                prior_offset, device=info.device, dtype=info.dtype
            ).reshape(info.shape[0])
    chol, status = torch.linalg.cholesky_ex(total)
    if int(status.item()) != 0:
        raise ValueError(
            "reduced normal matrix is not positive definite -- the joint coordinate "
            "is unobservable from this frame; keep the prior instead of updating"
        )
    delta = torch.cholesky_solve(rhs.reshape(-1, 1), chol).reshape(-1)
    return delta, total


@dataclass(frozen=True)
class JointState:
    """Section 30's reduced state ``[q; qdot]`` with its 2x2 covariance.

    Frozen: :meth:`predict` and :meth:`update` return new states, so a rejected
    IEKF step is discarded by simply dropping the result instead of having to
    roll a mutated filter back.
    """

    q: torch.Tensor
    qdot: torch.Tensor
    cov: torch.Tensor

    def __post_init__(self) -> None:
        cov = torch.as_tensor(self.cov)
        if cov.shape != (2, 2):
            raise ValueError(f"cov must be [2,2], got {tuple(cov.shape)}")
        object.__setattr__(self, "cov", _sym(cov))
        object.__setattr__(self, "q", torch.as_tensor(self.q, device=cov.device, dtype=cov.dtype).reshape(()))
        object.__setattr__(self, "qdot", torch.as_tensor(self.qdot, device=cov.device, dtype=cov.dtype).reshape(()))

    # -- constructors -------------------------------------------------------
    @classmethod
    def initial(
        cls,
        q: float | torch.Tensor = 0.0,
        qdot: float | torch.Tensor = 0.0,
        *,
        sigma_q: float = 1.0,
        sigma_qdot: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> "JointState":
        cov = torch.diag(
            torch.tensor([sigma_q ** 2, sigma_qdot ** 2], dtype=dtype, device=device)
        )
        return cls(torch.as_tensor(q, dtype=dtype, device=device),
                   torch.as_tensor(qdot, dtype=dtype, device=device), cov)

    @property
    def vector(self) -> torch.Tensor:
        """``[q; qdot]`` as a single 2-vector."""
        return torch.stack((self.q, self.qdot))

    # -- section 30 transition ---------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        dt: float | torch.Tensor,
        *,
        accel_psd: float = 0.0,
        process_noise: torch.Tensor | None = None,
    ) -> "JointState":
        """``x^- = [[1, dt],[0, 1]] x^+``, ``P^- = F P F^T + Q`` (section 30).

        ``dt`` is the *real* inter-frame time of the video (rule 11); it is never
        defaulted and never assumed to be 1.  ``Q`` defaults to the exactly
        discretised continuous white-noise-acceleration model

            Q = accel_psd * [[dt^3/3, dt^2/2], [dt^2/2, dt]],

        which is the right ``dt`` dependence for a joint whose acceleration is
        unmodelled (the no-force baseline of section 30, before the force-enabled
        dynamics of section 32).  Because every term carries a positive power of
        ``dt``, ``predict(0)`` is exactly the identity.  An explicit
        ``process_noise`` overrides it and is used verbatim -- the caller then
        owns the ``dt`` dependence.
        """
        dtv = _as_scalar(dt, self.cov.device, self.cov.dtype)
        if not bool(torch.isfinite(dtv).item()) or float(dtv.item()) < 0.0:
            raise ValueError(f"dt must be finite and non-negative, got {float(dtv.item())}")
        one = torch.ones((), device=self.cov.device, dtype=self.cov.dtype)
        zero = torch.zeros((), device=self.cov.device, dtype=self.cov.dtype)
        F = torch.stack((torch.stack((one, dtv)), torch.stack((zero, one))))
        if process_noise is None:
            if accel_psd < 0.0:
                raise ValueError("accel_psd must be non-negative")
            dt2, dt3 = dtv * dtv, dtv * dtv * dtv
            Q = accel_psd * torch.stack((
                torch.stack((dt3 / 3.0, dt2 / 2.0)),
                torch.stack((dt2 / 2.0, dtv)),
            ))
        else:
            Q = _sym(torch.as_tensor(process_noise, device=self.cov.device, dtype=self.cov.dtype))
            if Q.shape != (2, 2):
                raise ValueError(f"process_noise must be [2,2], got {tuple(Q.shape)}")
        x = F @ self.vector
        cov = _sym(F @ self.cov @ F.transpose(0, 1) + Q)
        return JointState(x[0], x[1], cov)

    # -- section 30 measurement update -------------------------------------
    @torch.no_grad()
    def update(
        self,
        information: torch.Tensor,
        b: torch.Tensor,
        *,
        prior: "JointState | None" = None,
    ) -> "JointState":
        """One information-form IEKF update of the reduced state (section 30).

        ``information = sum_p H_{p,q}^T R_p^-1 H_{p,q}`` and
        ``b = sum_p H_{p,q}^T R_p^-1 r_p``, both already reduced by
        :func:`reduce_pixel_jacobian`.  A scalar/1x1 argument means the frame
        constrains ``q`` only, which is the correct default: the renderer sees
        positions, not speeds, and the speed factor of section 31 is explicitly
        disabled in this implementation.

        ``I_q = P^-1 + sum H^T R^-1 H`` and ``P^+ = I_q^-1`` -- the 2x2 inverse is
        the whole covariance of the reduced part, so rule 7 (no dense covariance)
        is satisfied by construction.

        ``prior`` is the *frozen* predicted state of the current frame.  Leave it
        at ``None`` on the first inner iteration (the prior is then ``self``); on
        every later iteration of the same frame pass the same predicted state
        back in, so the prior contributes ``P^-1 (x^- - x_current)`` instead of
        the frame being counted as a fresh independent measurement (rule 8).
        """
        device, dtype = self.cov.device, self.cov.dtype
        meas = _embed_information(information, device, dtype)
        rhs = _embed_vector(b, device, dtype)
        source = self if prior is None else prior
        chol, status = torch.linalg.cholesky_ex(source.cov)
        if int(status.item()) != 0:
            raise ValueError("prior covariance is not positive definite")
        prior_info = _sym(torch.cholesky_inverse(chol))
        offset = None if prior is None else (prior.vector - self.vector)
        delta, total = solve_reduced_normal(meas, rhs, prior_info, offset)
        x = self.vector + delta
        return JointState(x[0], x[1], reduced_covariance(total))

    def replaced(self, **kwargs: Any) -> "JointState":
        """Copy with fields overridden (checkpoint/restore helper)."""
        return replace(self, **kwargs)
