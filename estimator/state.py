"""Per-Gaussian nominal state and 12D manifold error state.

Specification: ``CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md`` sections 3, 4 and
the ``state.py`` entry of section 23.

Nominal state of Gaussian ``i`` at frame ``k``:

    x_i = (mu_i in R^3,  R_i in SO(3),  v_i in R^3,  omega_i in R^3)

Local error state (the only thing the filter ever solves for):

    dx_i = [dmu_i, dphi_i, dv_i, domega_i] in R^12

Retraction (spec section 4 -- LEFT multiplication on the rotation):

    mu+     = mu  + dmu
    R+      = Exp(hat(dphi)) @ R
    v+      = v   + dv
    omega+  = omega + domega

The left convention is load bearing.  ``boxminus`` therefore uses
``Log(R_a @ R_b^T)``, the process model uses ``Ad_Exp(Omega)`` on the dphi
block, and every rotation block of P and of H is expressed in the same spatial
frame.  Mixing in a body-frame (right) update would not raise anything -- it
would just make the covariance wrong -- so this module never offers one.

Frozen appearance (spec section 3): ``colors`` (DC/frozen radiance),
``opacity`` and ``scales`` ride along with the state because the measurement
operator needs them, but the estimator must never write them.  They are shared
by reference across ``boxplus``/``boxminus`` results, both because copying them
every inner iteration is pure waste and because sharing makes an accidental
in-place write visible to :meth:`GaussianMotionState.assert_appearance_unchanged`.

Autograd policy (spec section 36): nothing here calls ``backward`` or builds a
graph on purpose.  :meth:`GaussianMotionState.detach` is the sanctioned
boundary between the autograd-based legacy trainer and the autograd-free
estimator.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import torch
from torch import Tensor

from .manifold import is_rotation, project_to_so3, so3_exp, so3_log

__all__ = [
    "GaussianMotionState",
    "DELTA_DIM",
    "DELTA_SLICE_MU",
    "DELTA_SLICE_PHI",
    "DELTA_SLICE_V",
    "DELTA_SLICE_OMEGA",
    "boxplus",
    "boxminus",
    "split_delta",
    "stack_delta",
    "quaternion_wxyz_to_matrix",
    "matrix_to_quaternion_wxyz",
    "set_rotation_validation",
    "rotation_validation_enabled",
    "STATE_DICT_VERSION",
]


# ---------------------------------------------------------------------------
# error-state layout -- every estimator module must index with these
# ---------------------------------------------------------------------------
DELTA_DIM = 12
DELTA_SLICE_MU = slice(0, 3)
DELTA_SLICE_PHI = slice(3, 6)
DELTA_SLICE_V = slice(6, 9)
DELTA_SLICE_OMEGA = slice(9, 12)

STATE_DICT_VERSION = 1

_MOTION_FIELDS = ("mu", "R", "v", "omega")
_FROZEN_FIELDS = ("colors", "opacity", "scales")
_ALL_FIELDS = _MOTION_FIELDS + _FROZEN_FIELDS

# Orthonormality budget for R.  Tight enough that an elementwise rotation
# update (which breaks R R^T = I at O(|delta|), i.e. ~1e-3 for a realistic
# step) is caught immediately; loose enough to tolerate the ~sqrt(#updates)*eps
# drift of a long chain of Exp(dphi) @ R products before reorthonormalisation.
_ROTATION_TOL_SCALE = 1.0e4

_VALIDATE_ROTATIONS = True


def set_rotation_validation(enabled: bool) -> bool:
    """Toggle the SO(3) guard on ``R`` assignment; returns the previous value.

    The check costs one O(N) reduction plus a host sync per constructed state.
    That is negligible next to a render, but it is on the critical path of a
    tight synthetic benchmark, hence the escape hatch.
    """
    global _VALIDATE_ROTATIONS
    previous = _VALIDATE_ROTATIONS
    _VALIDATE_ROTATIONS = bool(enabled)
    return previous


def rotation_validation_enabled() -> bool:
    return _VALIDATE_ROTATIONS


# ---------------------------------------------------------------------------
# delta helpers
# ---------------------------------------------------------------------------
def split_delta(delta: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``[N, 12] -> (dmu, dphi, dv, domega)``, each ``[N, 3]`` (views, no copy)."""
    delta = _check_delta(delta)
    return (
        delta[:, DELTA_SLICE_MU],
        delta[:, DELTA_SLICE_PHI],
        delta[:, DELTA_SLICE_V],
        delta[:, DELTA_SLICE_OMEGA],
    )


def stack_delta(dmu: Tensor, dphi: Tensor, dv: Tensor, domega: Tensor) -> Tensor:
    """``(dmu, dphi, dv, domega) -> [N, 12]`` in the canonical block order."""
    blocks = (dmu, dphi, dv, domega)
    names = ("dmu", "dphi", "dv", "domega")
    for block, name in zip(blocks, names):
        if block.ndim != 2 or block.shape[-1] != 3:
            raise ValueError(f"stack_delta: {name} must be [N, 3], got {tuple(block.shape)}")
    return torch.cat(blocks, dim=1)


def _check_delta(delta: Tensor, num_gaussians: int | None = None) -> Tensor:
    delta = torch.as_tensor(delta)
    if not delta.is_floating_point():
        raise TypeError(f"error-state delta must be floating point, got {delta.dtype}")
    if delta.ndim == 3 and delta.shape[-2:] == (3, 3):
        raise ValueError(
            "error-state delta has shape [N, 3, 3]: rotations are NOT updated by adding "
            "matrices. Pass a [N, 12] tangent vector; the dphi block is retracted with "
            "R+ = Exp(hat(dphi)) @ R (spec section 4)."
        )
    if delta.ndim != 2 or delta.shape[1] != DELTA_DIM:
        raise ValueError(
            f"error-state delta must be [N, {DELTA_DIM}] "
            f"(dmu | dphi | dv | domega), got {tuple(delta.shape)}"
        )
    if num_gaussians is not None and delta.shape[0] != num_gaussians:
        raise ValueError(
            f"error-state delta has {delta.shape[0]} rows but the state holds "
            f"{num_gaussians} Gaussians"
        )
    return delta


# ---------------------------------------------------------------------------
# quaternion bridge to the 3DGS model / CUDA rasterizer
# ---------------------------------------------------------------------------
def quaternion_wxyz_to_matrix(quaternion: Tensor) -> Tensor:
    """``[N, 4] wxyz -> [N, 3, 3]``, matching ``utils.general_utils.build_rotation``.

    The rasterizer stores orientation as an unnormalised local-to-world wxyz
    quaternion; the estimator needs a matrix because ``Sigma = R diag(s^2) R^T``
    and all the analytic differentials of section 9 are written in matrices.
    """
    quaternion = torch.as_tensor(quaternion)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"quaternion must be [N, 4] wxyz, got {tuple(quaternion.shape)}")
    norm = torch.linalg.norm(quaternion, dim=1, keepdim=True)
    q = quaternion / norm.clamp_min(torch.finfo(quaternion.dtype).eps)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[0], 3, 3)


def matrix_to_quaternion_wxyz(rotation: Tensor) -> Tensor:
    """``[N, 3, 3] -> [N, 4]`` unit wxyz, inverse of :func:`quaternion_wxyz_to_matrix`.

    Shepperd's method: the branch with the largest of ``{tr, R00, R11, R22}``
    is taken so the divisor is never small.  The single-formula version that
    always divides by ``sqrt(1 + tr)`` loses all precision for rotations near
    pi, which is exactly where an articulated joint ends up after a few dozen
    frames.  The sign is canonicalised to ``w >= 0`` so that the mapping is
    deterministic (q and -q are the same rotation).
    """
    rotation = torch.as_tensor(rotation)
    if rotation.ndim != 3 or rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must be [N, 3, 3], got {tuple(rotation.shape)}")
    r00, r01, r02 = rotation[:, 0, 0], rotation[:, 0, 1], rotation[:, 0, 2]
    r10, r11, r12 = rotation[:, 1, 0], rotation[:, 1, 1], rotation[:, 1, 2]
    r20, r21, r22 = rotation[:, 2, 0], rotation[:, 2, 1], rotation[:, 2, 2]
    trace = r00 + r11 + r22
    eps = torch.finfo(rotation.dtype).eps

    # Four algebraically identical parameterisations that differ only in
    # conditioning; all are evaluated and one is selected per row.
    diag_w, diag_x = trace, 2.0 * r00 - trace
    diag_y, diag_z = 2.0 * r11 - trace, 2.0 * r22 - trace

    t_w = (1.0 + diag_w).clamp_min(0.0).sqrt()
    t_x = (1.0 + diag_x).clamp_min(0.0).sqrt()
    t_y = (1.0 + diag_y).clamp_min(0.0).sqrt()
    t_z = (1.0 + diag_z).clamp_min(0.0).sqrt()
    h_w, h_x = 0.5 / t_w.clamp_min(eps), 0.5 / t_x.clamp_min(eps)
    h_y, h_z = 0.5 / t_y.clamp_min(eps), 0.5 / t_z.clamp_min(eps)

    cand_w = torch.stack((0.5 * t_w, (r21 - r12) * h_w, (r02 - r20) * h_w, (r10 - r01) * h_w), dim=-1)
    cand_x = torch.stack(((r21 - r12) * h_x, 0.5 * t_x, (r01 + r10) * h_x, (r02 + r20) * h_x), dim=-1)
    cand_y = torch.stack(((r02 - r20) * h_y, (r01 + r10) * h_y, 0.5 * t_y, (r12 + r21) * h_y), dim=-1)
    cand_z = torch.stack(((r10 - r01) * h_z, (r02 + r20) * h_z, (r12 + r21) * h_z, 0.5 * t_z), dim=-1)

    scores = torch.stack((diag_w, diag_x, diag_y, diag_z), dim=-1)
    pick = scores.argmax(dim=-1)
    stacked = torch.stack((cand_w, cand_x, cand_y, cand_z), dim=1)  # [N, 4 branches, 4]
    q = torch.gather(stacked, 1, pick[:, None, None].expand(-1, 1, 4)).squeeze(1)
    q = q / torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(eps)
    sign = torch.where(q[:, :1] < 0.0, -torch.ones_like(q[:, :1]), torch.ones_like(q[:, :1]))
    return q * sign


# ---------------------------------------------------------------------------
# the state
# ---------------------------------------------------------------------------
@dataclasses.dataclass(eq=False)
class GaussianMotionState:
    """Nominal per-Gaussian motion state plus its frozen appearance payload.

    Shapes (``N`` = number of Gaussians)::

        mu      [N, 3]      world-frame centers
        R       [N, 3, 3]   world-from-local rotations, Sigma = R diag(s^2) R^T
        v       [N, 3]      world-frame linear velocity   (per second)
        omega   [N, 3]      world-frame angular velocity  (per second)

        colors  [N, 3]      FROZEN DC / evaluated radiance, spec section 3.1
        opacity [N, 1]      FROZEN intrinsic opacity in [0, 1]
        scales  [N, 3]      FROZEN principal standard deviations

    Velocities are per *second*, not per frame: spec section 5 integrates with
    the real video ``dt``, so storing per-frame velocities would silently
    break the moment the sequence is not 1 fps.
    """

    mu: Tensor
    R: Tensor
    v: Tensor
    omega: Tensor
    colors: Tensor
    opacity: Tensor
    scales: Tensor

    # -- construction / validation -----------------------------------------
    def __setattr__(self, name: str, value: Any) -> None:
        # The guard lives here rather than in __post_init__ so that it also
        # fires on `state.R = state.R + dR`, which is the realistic way the
        # left-multiplied retraction gets bypassed.
        if name == "R":
            value = _validate_rotation_field(value)
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "GaussianMotionState":
        """Shape / device / dtype consistency. Raises on the first problem."""
        for name in _ALL_FIELDS:
            tensor = getattr(self, name)
            if not isinstance(tensor, Tensor):
                raise TypeError(f"GaussianMotionState.{name} must be a torch.Tensor, got {type(tensor)!r}")
            if not tensor.is_floating_point():
                raise TypeError(f"GaussianMotionState.{name} must be floating point, got {tensor.dtype}")
        n = self.mu.shape[0]
        expected = {
            "mu": (n, 3), "R": (n, 3, 3), "v": (n, 3), "omega": (n, 3),
            "colors": (n, 3), "opacity": (n, 1), "scales": (n, 3),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(self, name).shape)
            if actual != shape:
                raise ValueError(f"GaussianMotionState.{name} must be {shape}, got {actual}")
        device = self.mu.device
        for name in _ALL_FIELDS:
            if getattr(self, name).device != device:
                raise ValueError(
                    f"GaussianMotionState.{name} is on {getattr(self, name).device} but mu is on "
                    f"{device}; the measurement operator indexes them together"
                )
        dtype = self.mu.dtype
        for name in _MOTION_FIELDS:
            if getattr(self, name).dtype != dtype:
                raise TypeError(
                    f"GaussianMotionState.{name} has dtype {getattr(self, name).dtype} but mu has "
                    f"{dtype}; the 12x12 covariance blocks mix the motion fields, so they must agree"
                )
        # The frozen appearance block is deliberately NOT forced to the motion
        # dtype: a float64 filter on top of a float32 3DGS model is the normal
        # configuration, and up-casting the appearance would double its memory
        # for no benefit since it is never part of the state vector.
        return self

    # -- introspection ------------------------------------------------------
    @property
    def num_gaussians(self) -> int:
        return int(self.mu.shape[0])

    def __len__(self) -> int:
        return self.num_gaussians

    @property
    def device(self) -> torch.device:
        return self.mu.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mu.dtype

    @property
    def state_dim(self) -> int:
        """Total error-state dimension ``12 N``."""
        return DELTA_DIM * self.num_gaussians

    def covariance(self) -> Tensor:
        """``[N, 3, 3]`` world covariances ``Sigma_i = R_i diag(s_i^2) R_i^T``.

        Built here rather than cached because ``R`` changes on every accepted
        IEKF step while ``scales`` never does.
        """
        lam = self.scales.to(self.R.dtype).square()
        return (self.R * lam[:, None, :]) @ self.R.transpose(1, 2)

    # -- manifold operations ------------------------------------------------
    def boxplus(self, delta: Tensor) -> "GaussianMotionState":
        return boxplus(self, delta)

    def boxminus(self, other: "GaussianMotionState") -> Tensor:
        return boxminus(self, other)

    def reorthonormalize(self) -> "GaussianMotionState":
        """Project ``R`` back onto SO(3); see :func:`manifold.project_to_so3`."""
        return dataclasses.replace(self, R=project_to_so3(self.R))

    # -- forbidden operations ----------------------------------------------
    def __add__(self, other: Any) -> "GaussianMotionState":
        raise TypeError(_ELEMENTWISE_MESSAGE.format(op="+"))

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other: Any) -> "GaussianMotionState":
        raise TypeError(_ELEMENTWISE_MESSAGE.format(op="-"))

    __rsub__ = __sub__
    __isub__ = __sub__

    def __mul__(self, other: Any) -> "GaussianMotionState":
        raise TypeError(_ELEMENTWISE_MESSAGE.format(op="*"))

    __rmul__ = __mul__
    __imul__ = __mul__

    # -- device / dtype / graph --------------------------------------------
    def to(self, device: Any = None, dtype: torch.dtype | None = None) -> "GaussianMotionState":
        """Move/cast every field. Returns ``self`` when nothing would change."""
        if isinstance(device, torch.dtype):
            if dtype is not None:
                raise TypeError("GaussianMotionState.to: dtype given twice")
            device, dtype = None, device
        if device is None and dtype is None:
            return self
        moved = {
            name: getattr(self, name).to(device=device, dtype=dtype)
            for name in _ALL_FIELDS
        }
        return GaussianMotionState(**moved)

    def detach(self) -> "GaussianMotionState":
        """Cut every field out of the autograd graph.

        This is the enforced boundary of spec section 36: the estimator must
        never be reachable from a ``loss.backward()``, and states handed over
        from the legacy Adam trainer are still attached to its graph.
        """
        return GaussianMotionState(**{name: getattr(self, name).detach() for name in _ALL_FIELDS})

    def clone(self) -> "GaussianMotionState":
        """Deep copy, including the frozen appearance block."""
        return GaussianMotionState(**{name: getattr(self, name).clone() for name in _ALL_FIELDS})

    # -- checkpointing ------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Serialisable snapshot. Tensors are detached clones, so a later
        in-place estimator update cannot retro-actively corrupt a checkpoint."""
        payload: dict[str, Any] = {
            "version": STATE_DICT_VERSION,
            "num_gaussians": self.num_gaussians,
            "delta_dim": DELTA_DIM,
        }
        for name in _ALL_FIELDS:
            payload[name] = getattr(self, name).detach().clone()
        return payload

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> "GaussianMotionState":
        """In-place load. Bitwise exact when dtypes match; raises when they do not."""
        _check_state_dict(state_dict, strict)
        n = int(state_dict["num_gaussians"])
        if n != self.num_gaussians:
            raise ValueError(
                f"checkpoint holds {n} Gaussians but this state holds {self.num_gaussians}; "
                "use GaussianMotionState.from_state_dict for a differently sized scene"
            )
        for name in _ALL_FIELDS:
            source = torch.as_tensor(state_dict[name])
            target = getattr(self, name)
            if strict and source.dtype != target.dtype:
                raise TypeError(
                    f"checkpoint field {name} has dtype {source.dtype} but the state has "
                    f"{target.dtype}; copying would silently round the values"
                )
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"checkpoint field {name} has shape {tuple(source.shape)}, expected "
                    f"{tuple(target.shape)}"
                )
            target.copy_(source)
        # R was written through copy_, which bypasses __setattr__, so re-run the
        # SO(3) guard explicitly on whatever the checkpoint contained.
        self.R = self.R
        return self

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Any],
        device: Any = None,
        dtype: torch.dtype | None = None,
    ) -> "GaussianMotionState":
        """Rebuild a state from :meth:`state_dict`, optionally moving/casting."""
        _check_state_dict(state_dict, strict=True)
        fields = {name: torch.as_tensor(state_dict[name]) for name in _ALL_FIELDS}
        if device is not None or dtype is not None:
            fields = {name: t.to(device=device, dtype=dtype) for name, t in fields.items()}
        return cls(**fields)

    # -- construction helpers ----------------------------------------------
    @classmethod
    def identity(
        cls,
        num_gaussians: int,
        device: Any = None,
        dtype: torch.dtype = torch.float64,
    ) -> "GaussianMotionState":
        """``N`` unit-scale Gaussians at the origin, at rest. Test/bootstrap aid."""
        if num_gaussians < 0:
            raise ValueError("num_gaussians must be non-negative")
        zeros = torch.zeros((num_gaussians, 3), device=device, dtype=dtype)
        return cls(
            mu=zeros.clone(),
            R=torch.eye(3, device=device, dtype=dtype).expand(num_gaussians, 3, 3).contiguous(),
            v=zeros.clone(),
            omega=zeros.clone(),
            colors=torch.full((num_gaussians, 3), 0.5, device=device, dtype=dtype),
            opacity=torch.ones((num_gaussians, 1), device=device, dtype=dtype),
            scales=torch.ones((num_gaussians, 3), device=device, dtype=dtype),
        )

    @classmethod
    def from_gaussian_arrays(
        cls,
        xyz: Tensor,
        rotation_wxyz: Tensor,
        scales: Tensor,
        opacity: Tensor,
        colors: Tensor,
        v: Tensor | None = None,
        omega: Tensor | None = None,
        dtype: torch.dtype | None = None,
    ) -> "GaussianMotionState":
        """Build a state from the 3DGS model's *activated* arrays.

        The caller applies the model's activations (``exp`` on log-scales,
        ``sigmoid`` on the opacity logit, SH evaluation or DC readout for the
        colour).  Keeping the activations outside this module is deliberate:
        ``scene/gaussian_model.py`` owns them and the estimator must not fork a
        second copy of that convention.
        """
        # Everything is detached on the way in: the source arrays are usually
        # live nn.Parameters of the legacy Adam trainer, and the estimator must
        # not be reachable from its graph (spec section 36).
        xyz = torch.as_tensor(xyz).detach()
        target_dtype = dtype if dtype is not None else xyz.dtype
        device = xyz.device
        n = xyz.shape[0]
        rotation = quaternion_wxyz_to_matrix(
            torch.as_tensor(rotation_wxyz).detach().to(device=device, dtype=target_dtype)
        )
        zero = torch.zeros((n, 3), device=device, dtype=target_dtype)
        return cls(
            mu=xyz.to(dtype=target_dtype),
            R=rotation,
            v=zero.clone() if v is None else torch.as_tensor(v).detach().to(device=device, dtype=target_dtype),
            omega=zero.clone() if omega is None else torch.as_tensor(omega).detach().to(device=device, dtype=target_dtype),
            colors=torch.as_tensor(colors).detach().to(device=device),
            opacity=torch.as_tensor(opacity).detach().reshape(n, 1).to(device=device),
            scales=torch.as_tensor(scales).detach().to(device=device),
        )

    def to_gaussian_arrays(self, dtype: torch.dtype | None = None) -> dict[str, Tensor]:
        """Inverse of :meth:`from_gaussian_arrays`, for feeding the rasterizer."""
        cast = (lambda t: t) if dtype is None else (lambda t: t.to(dtype))
        return {
            "xyz": cast(self.mu),
            "rotation_wxyz": cast(matrix_to_quaternion_wxyz(self.R)),
            "scales": cast(self.scales),
            "opacity": cast(self.opacity),
            "colors": cast(self.colors),
        }

    # -- frozen-appearance contract ----------------------------------------
    def assert_appearance_unchanged(self, other: "GaussianMotionState") -> None:
        """Raise unless the frozen block of ``other`` is bitwise identical.

        Spec section 3 forbids the estimator from touching appearance; this is
        the assertion that turns that prose into a runtime contract.
        """
        for name in _FROZEN_FIELDS:
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine.shape != theirs.shape or not torch.equal(mine, theirs):
                raise AssertionError(
                    f"frozen appearance field '{name}' changed: the estimator updates motion "
                    "only (spec section 3)"
                )


_ELEMENTWISE_MESSAGE = (
    "GaussianMotionState does not support '{op}': R lives on SO(3) and adding rotation "
    "matrices (or quaternions) elementwise leaves the manifold. Use "
    "boxplus(state, delta[N, 12]) / boxminus(a, b) from estimator.state instead "
    "(spec section 4)."
)


def _validate_rotation_field(value: Any) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"GaussianMotionState.R must be a torch.Tensor, got {type(value)!r}")
    if not value.is_floating_point():
        raise TypeError(f"GaussianMotionState.R must be floating point, got {value.dtype}")
    if value.ndim != 3 or value.shape[-2:] != (3, 3):
        raise ValueError(
            f"GaussianMotionState.R must be [N, 3, 3] rotation matrices, got {tuple(value.shape)}"
        )
    if _VALIDATE_ROTATIONS and value.shape[0] > 0:
        tol = _ROTATION_TOL_SCALE * float(torch.finfo(value.dtype).eps)
        ok = is_rotation(value, tol=tol)
        if not bool(ok.all()):
            bad = int((~ok).sum())
            raise ValueError(
                f"GaussianMotionState.R: {bad}/{value.shape[0]} matrices are not in SO(3) "
                f"(R R^T != I or det != +1 beyond {tol:.3g}). The usual cause is an "
                "ELEMENTWISE rotation update such as `R = R + dR` or `q = q + dq`; the "
                "retraction is R+ = Exp(hat(dphi)) @ R (spec section 4). Call "
                "state.reorthonormalize() if this is accumulated round-off instead."
            )
    return value


def _check_state_dict(state_dict: Mapping[str, Any], strict: bool) -> None:
    missing = [k for k in _ALL_FIELDS if k not in state_dict]
    if missing:
        raise KeyError(f"checkpoint is missing GaussianMotionState fields: {missing}")
    if "num_gaussians" not in state_dict:
        raise KeyError("checkpoint is missing 'num_gaussians'")
    version = state_dict.get("version")
    if strict and version != STATE_DICT_VERSION:
        raise ValueError(
            f"checkpoint version {version!r} != {STATE_DICT_VERSION}; refusing to guess the layout"
        )


# ---------------------------------------------------------------------------
# boxplus / boxminus
# ---------------------------------------------------------------------------
def boxplus(state: GaussianMotionState, delta: Tensor) -> GaussianMotionState:
    """Retraction of spec section 4. Returns a NEW state; ``state`` is untouched.

        mu+ = mu + dmu
        R+  = Exp(hat(dphi)) @ R      <-- LEFT multiplication
        v+  = v + dv
        omega+ = omega + domega

    The left action is what makes ``dphi`` a *spatial* (world-frame) rotation
    error, which is the frame the measurement Jacobians of section 9 and the
    covariance blocks of section 17 are written in.
    """
    if not isinstance(state, GaussianMotionState):
        raise TypeError(f"boxplus expects a GaussianMotionState, got {type(state)!r}")
    delta = _check_delta(delta, state.num_gaussians)
    if delta.device != state.device:
        raise ValueError(f"delta is on {delta.device} but the state is on {state.device}")
    delta = delta.to(state.dtype)

    dmu, dphi, dv, domega = split_delta(delta)
    # Exp(0) == I bitwise (Rodrigues' Taylor branch gives a == 1, b == 0.5 and
    # hat(0) == 0), so a zero delta is an exact no-op -- the IEKF line search
    # depends on that to detect a rejected step.
    rotation = so3_exp(dphi) @ state.R
    return GaussianMotionState(
        mu=state.mu + dmu,
        R=rotation,
        v=state.v + dv,
        omega=state.omega + domega,
        # Shared by reference: frozen appearance is never written (section 3).
        colors=state.colors,
        opacity=state.opacity,
        scales=state.scales,
    )


def boxminus(state_a: GaussianMotionState, state_b: GaussianMotionState) -> Tensor:
    """``a boxminus b -> [N, 12]`` such that ``boxplus(b, a boxminus b) == a``.

    The rotation block is ``Log(R_a @ R_b^T)``, the left-invariant difference
    matching :func:`boxplus`.  It is only the exact inverse while the relative
    angle stays below pi -- past that the principal logarithm wraps, which is a
    property of SO(3) and not a bug; the IEKF never takes steps that large.
    """
    for name, obj in (("state_a", state_a), ("state_b", state_b)):
        if not isinstance(obj, GaussianMotionState):
            raise TypeError(f"boxminus expects GaussianMotionState for {name}, got {type(obj)!r}")
    if state_a.num_gaussians != state_b.num_gaussians:
        raise ValueError(
            f"boxminus size mismatch: {state_a.num_gaussians} vs {state_b.num_gaussians} Gaussians"
        )
    if state_a.device != state_b.device:
        raise ValueError(f"boxminus device mismatch: {state_a.device} vs {state_b.device}")
    other = state_b.to(dtype=state_a.dtype) if state_b.dtype != state_a.dtype else state_b
    return stack_delta(
        state_a.mu - other.mu,
        so3_log(state_a.R @ other.R.transpose(1, 2)),
        state_a.v - other.v,
        state_a.omega - other.omega,
    )
