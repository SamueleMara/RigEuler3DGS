"""`exact_cached_structured`: the SAME full-window IEKS system, computed faster.

CODEX_EXACT_FUSED_IEKS_ACCELERATION_SPARSE_DATA_LAYOUT_NO_APPROXIMATIONS.md.

Nothing here changes the problem. The operator applied by PCG is

    A = H^T W H + J_proc^T Q^-1 J_proc + A_prior + lambda D

on the free-coordinate subspace, exactly as `ieks._CudaWindowOperator` applies
it; the right-hand side, the LM metric D and the anchor elimination are the
same quantities. What changes is WHERE things live and HOW OFTEN they are
computed:

  FrozenFrameStore    every (frame, camera) linearisation is built ONCE per
                      accepted state, its gradient and Fisher are taken while
                      it is on the GPU, and the frozen cache is packed into ONE
                      flat host buffer per frame. Each PCG application copies a
                      frame's buffer back (one H2D) and runs the unchanged
                      production JVP/VJP on it. The legacy streamed path
                      re-rendered every frame on every application.
  StructuredProcess   the validated dense [N,12,12] process Jacobians are
                      compressed to the 3x3 blocks that are actually nonzero
                      (read off the real matrices, never assumed), uniform
                      blocks stored once; the pair stencil runs on the GPU in
                      float64.
  temporal M          an optional per-Gaussian block-tridiagonal preconditioner
                      over the whole trajectory. It omits cross-Gaussian image
                      couplings INSIDE M ONLY; A keeps every one of them.

The legacy operator remains callable and is the oracle in every test.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any

import torch

ERROR_DIM = 12
POSE_DIM = 6
_ALIGN = 256


# ---------------------------------------------------------------------------
# packing a FrameCache into one flat buffer and back, structure preserved
# ---------------------------------------------------------------------------

class _Slot:
    """Placeholder for a tensor inside a packed skeleton."""

    __slots__ = ("index",)

    def __init__(self, index):
        self.index = index


def _pack(obj, tensors: list, memo: dict):
    """Replace every tensor by a _Slot, keeping shared tensors shared."""
    if isinstance(obj, torch.Tensor):
        key = (obj.data_ptr(), tuple(obj.shape), tuple(obj.stride()), obj.dtype,
               obj.device)
        if key not in memo:
            memo[key] = len(tensors)
            tensors.append(obj)
        return _Slot(memo[key])
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return (type(obj), "dc", {f.name: _pack(getattr(obj, f.name), tensors, memo)
                                  for f in dataclasses.fields(obj)})
    if isinstance(obj, list):
        return [_pack(o, tensors, memo) for o in obj]
    if isinstance(obj, tuple):
        return ("__tuple__", [_pack(o, tensors, memo) for o in obj])
    if hasattr(obj, "__dict__") and type(obj).__module__.startswith(
            ("estimator", "bidirectional_iekf")):
        return (type(obj), "obj", {k: _pack(v, tensors, memo)
                                   for k, v in vars(obj).items()})
    return obj


def _unpack(skel, tensors: list):
    if isinstance(skel, _Slot):
        return tensors[skel.index]
    if isinstance(skel, list):
        return [_unpack(s, tensors) for s in skel]
    if isinstance(skel, tuple) and len(skel) == 2 and skel[0] == "__tuple__":
        return tuple(_unpack(s, tensors) for s in skel[1])
    if isinstance(skel, tuple) and len(skel) == 3 and skel[1] in ("dc", "obj"):
        cls, kind, fields = skel
        values = {k: _unpack(v, tensors) for k, v in fields.items()}
        if kind == "dc":
            obj = object.__new__(cls)
            for k, v in values.items():
                object.__setattr__(obj, k, v)
            return obj
        obj = object.__new__(cls)
        obj.__dict__.update(values)
        return obj
    return skel


@dataclasses.dataclass
class PackedFrame:
    skeleton: Any
    layout: list            # (offset, nbytes, dtype, shape) per tensor
    flat: torch.Tensor      # uint8 host buffer, pageable
    nbytes: int


def pack_frame(cache) -> PackedFrame:
    tensors: list = []
    skel = _pack(cache, tensors, {})
    layout, off = [], 0
    for t in tensors:
        t = t.contiguous()
        nb = t.numel() * t.element_size()
        layout.append((off, nb, t.dtype, tuple(t.shape)))
        off += (nb + _ALIGN - 1) // _ALIGN * _ALIGN
    flat = torch.empty(max(off, 1), dtype=torch.uint8)
    for (o, nb, _dt, _sh), t in zip(layout, tensors):
        if nb:
            src = t.contiguous().reshape(-1).view(torch.uint8)
            flat[o:o + nb].copy_(src, non_blocking=False)
    return PackedFrame(skel, layout, flat, off)


def unpack_frame(packed: PackedFrame, device_flat: torch.Tensor):
    tensors = []
    for o, nb, dt, sh in packed.layout:
        if nb == 0:
            tensors.append(torch.empty(sh, dtype=dt, device=device_flat.device))
            continue
        tensors.append(device_flat[o:o + nb].view(dt).view(sh))
    return _unpack(packed.skeleton, tensors)


# ---------------------------------------------------------------------------
# the frozen image linearisation store
# ---------------------------------------------------------------------------

class FrozenFrameStore:
    """All frames' linearisations at ONE accepted state, host-resident.

    Built once per accepted nonlinear state; reused across PCG iterations and
    across LM damping changes at that state. A trial state is NEVER served
    from here: the line search re-renders through `obs.nonlinear_cost`.
    """

    def __init__(self, observation, trajectory, state_token, *, device=None,
                 transfer_policy="synchronous", pinned_budget_mib=1024,
                 keep_pose_fisher=False, host_budget_gib=None, verbose=False):
        if transfer_policy not in ("synchronous", "double_buffered"):
            raise ValueError(transfer_policy)
        self.obs = observation
        self.state_token = state_token
        self.T = trajectory.T
        self.N = trajectory.N
        self.transfer_policy = transfer_policy
        from bidirectional_iekf.ieks import _on
        self.device = torch.device(device) if device is not None else \
            trajectory.states[0].mu.device
        self.frames: list[PackedFrame] = []
        self.gradient: list[torch.Tensor] = []         # [N,12] float64 host
        self.meas_diag: list[torch.Tensor] = []        # [N,12] float64 host
        self.pose_fisher: list[torch.Tensor] | None = [] if keep_pose_fisher else None
        self.build_calls = 0
        self.stage_calls = 0
        self.timing = {"build_s": 0.0, "offload_s": 0.0, "stage_s": 0.0}
        t0 = time.perf_counter()
        for t, state in enumerate(trajectory.states):
            tb = time.perf_counter()
            cache = observation.linearize_frame(_on(state, self.device), t)
            self.build_calls += 1
            g = observation.gradient(cache).double()
            F = observation.block_fisher(cache).double()
            torch.cuda.synchronize(self.device)
            self.timing["build_s"] += time.perf_counter() - tb
            to = time.perf_counter()
            self.gradient.append(g.cpu())
            self.meas_diag.append(F.diagonal(dim1=-2, dim2=-1).cpu())
            if self.pose_fisher is not None:
                self.pose_fisher.append(F[:, :POSE_DIM, :POSE_DIM].contiguous().cpu())
            # The residuals are only needed for the gradient, taken above; the
            # PCG operators (JVP, precision, VJP) never read them.
            cache.residuals = [None] * len(cache.residuals)
            self.frames.append(pack_frame(cache))
            del cache, g, F
            if host_budget_gib is not None:
                used = sum(p.nbytes for p in self.frames)
                projected = used / (t + 1) * self.T
                if projected > host_budget_gib * 2 ** 30:
                    raise MemoryError(
                        f"CACHE_HOST_BUDGET_EXCEEDED: {t + 1} of {self.T} frames "
                        f"take {used / 2**30:.2f} GiB, projecting "
                        f"{projected / 2**30:.2f} GiB against a "
                        f"{host_budget_gib} GiB budget. Raise "
                        "--cache-host-budget-gib if the RAM is really free, or "
                        "use --linear-backend legacy_reference; no frame or "
                        "camera is dropped to make it fit")
            self.timing["offload_s"] += time.perf_counter() - to
        self.host_bytes = sum(p.nbytes for p in self.frames)
        self.max_frame_bytes = max(p.nbytes for p in self.frames)
        self._dev = [torch.empty(self.max_frame_bytes, dtype=torch.uint8,
                                 device=self.device)]
        self._pinned = []
        self._events = []
        self._side = None
        if transfer_policy == "double_buffered":
            budget = pinned_budget_mib * 2 ** 20
            if 2 * self.max_frame_bytes > budget:
                raise MemoryError(
                    f"double buffering needs 2 x {self.max_frame_bytes / 2**20:.0f} "
                    f"MiB pinned, budget {pinned_budget_mib} MiB")
            self._dev.append(torch.empty(self.max_frame_bytes, dtype=torch.uint8,
                                         device=self.device))
            self._pinned = [torch.empty(self.max_frame_bytes, dtype=torch.uint8,
                                        pin_memory=True) for _ in range(2)]
            self._events = [None, None]
            self._side = torch.cuda.Stream(device=self.device)
        self._prefetched = None
        self.timing["total_build_s"] = time.perf_counter() - t0
        if verbose:
            print(f"     frozen linearisation: {self.T} frames, "
                  f"{self.host_bytes / 2**30:.2f} GiB host, built in "
                  f"{self.timing['total_build_s']:.1f} s", flush=True)

    # -- staging --------------------------------------------------------------
    def _issue(self, t, slot):
        """double_buffered: pageable -> pinned (CPU) then async H2D (side)."""
        p = self.frames[t]
        if self._events[slot] is not None:
            self._events[slot].synchronize()        # pinned slot free again
        self._pinned[slot][:p.nbytes].copy_(p.flat[:p.nbytes])
        # the device slot last held frame t-2 or t-1: its kernels must finish
        # on the compute stream before the side stream overwrites it
        self._side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._side):
            self._dev[slot][:p.nbytes].copy_(self._pinned[slot][:p.nbytes],
                                             non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self._side)
        self._events[slot] = ev
        self._prefetched = (t, slot)

    def stage(self, t):
        """The frozen FrameCache of frame t, on the device."""
        ts = time.perf_counter()
        self.stage_calls += 1
        p = self.frames[t]
        if self.transfer_policy == "synchronous":
            dev = self._dev[0]
            dev[:p.nbytes].copy_(p.flat[:p.nbytes])
            out = unpack_frame(p, dev)
        else:
            if self._prefetched is None or self._prefetched[0] != t:
                self._issue(t, t % 2)
            _t, slot = self._prefetched
            torch.cuda.current_stream().wait_event(self._events[slot])
            out = unpack_frame(p, self._dev[slot])
            if t + 1 < self.T:
                self._issue(t + 1, (t + 1) % 2)
            else:
                self._prefetched = None
        self.timing["stage_s"] += time.perf_counter() - ts
        return out


# ---------------------------------------------------------------------------
# structured process factors
# ---------------------------------------------------------------------------

class _BlockMatrix:
    """A [N,12,12] matrix kept as its nonzero 3x3 blocks.

    The pattern is READ OFF the matrix: a block that is exactly zero for every
    Gaussian is dropped, a block identical for every Gaussian is stored once.
    Nothing about the process model is assumed, so the product equals the
    dense one up to float64 summation order.
    """

    def __init__(self, M: torch.Tensor):
        self.blocks = {}
        for r in range(4):
            for c in range(4):
                b = M[:, 3 * r:3 * r + 3, 3 * c:3 * c + 3]
                if not bool((b != 0).any()):
                    continue
                if bool((b == b[:1]).all()):
                    self.blocks[(r, c)] = b[:1].contiguous()
                else:
                    self.blocks[(r, c)] = b.contiguous()

    def to(self, device):
        out = object.__new__(_BlockMatrix)
        out.blocks = {k: v.to(device, non_blocking=True) for k, v in self.blocks.items()}
        return out

    def nbytes(self):
        return sum(v.numel() * 8 for v in self.blocks.values())

    def matvec(self, u):            # (M u), u [N,12]
        out = torch.zeros_like(u)
        for (r, c), b in self.blocks.items():
            out[:, 3 * r:3 * r + 3] += torch.einsum("nij,nj->ni", b.expand(
                u.shape[0], 3, 3), u[:, 3 * c:3 * c + 3])
        return out

    def rmatvec(self, w):           # (M^T w)
        out = torch.zeros_like(w)
        for (r, c), b in self.blocks.items():
            out[:, 3 * c:3 * c + 3] += torch.einsum("nji,nj->ni", b.expand(
                w.shape[0], 3, 3), w[:, 3 * r:3 * r + 3])
        return out


class StructuredProcess:
    """J_proc^T Q^-1 J_proc as a GPU pair stencil, built once per state."""

    def __init__(self, trajectory, Q_inv, *, device, gpu_budget_mib=1024):
        from bidirectional_iekf.process_factors import (process_residual,
                                                        process_residual_jacobians)
        self.T, self.N = trajectory.T, trajectory.N
        Qd = torch.as_tensor(Q_inv).double()
        uniform = (Qd.ndim == 3 and Qd.stride(0) == 0) or bool((Qd == Qd[:1]).all())
        self.Q_uniform = uniform
        self.Q = (Qd[:1] if uniform else Qd).to(device)
        self.A, self.B = [], []
        rhs = torch.zeros(self.T, self.N, ERROR_DIM, dtype=torch.float64)
        diag = torch.zeros(self.T, self.N, ERROR_DIM, dtype=torch.float64)
        self.on_gpu = True
        total = 0
        from bidirectional_iekf.ieks import _on
        st = trajectory.states
        for t in range(self.T - 1):
            s0, s1 = _on(st[t], device), _on(st[t + 1], device)
            A, B = process_residual_jacobians(s0, s1, trajectory.dt)
            e = process_residual(s0, s1, trajectory.dt).double()
            Q = self.Q.expand(self.N, 12, 12) if uniform else self.Q
            A, B, Q = A.to(device), B.to(device), Q.to(device)
            Qe = torch.einsum("nij,nj->ni", Q, e.to(device))
            rhs[t] -= torch.einsum("nji,nj->ni", A, Qe).cpu()
            rhs[t + 1] -= torch.einsum("nji,nj->ni", B, Qe).cpu()
            diag[t] += torch.einsum("nji,njk,nki->ni", A, Q, A).cpu()
            diag[t + 1] += torch.einsum("nji,njk,nki->ni", B, Q, B).cpu()
            a, b = _BlockMatrix(A), _BlockMatrix(B)
            total += a.nbytes() + b.nbytes()
            self.A.append(a)
            self.B.append(b)
            del A, B, Q, e, Qe
        self.rhs, self.diag = rhs, diag
        self.coeff_bytes = total
        # Placement follows the MEASURED free GPU memory, not only a fixed
        # budget: the card is shared, and after the first accepted step the
        # SO(3) blocks become per-Gaussian (585 MB on washer), which is what
        # ran the first overnight run out of memory. Host coefficients are
        # staged one pair at a time (~10 MB) inside each application.
        free, _total = torch.cuda.mem_get_info(device)
        limit = min(gpu_budget_mib * 2 ** 20, 0.15 * free)
        if total > limit:
            self.on_gpu = False
            self.A = [a.to("cpu") for a in self.A]
            self.B = [b.to("cpu") for b in self.B]
        self.placement = {"coeff_mib": total / 2 ** 20, "on_gpu": self.on_gpu,
                          "gpu_free_mib_at_build": free / 2 ** 20}
        torch.cuda.empty_cache()
        self.device = device
        self.patterns = {"A": sorted(self.A[0].blocks) if self.A else [],
                         "B": sorted(self.B[0].blocks) if self.B else [],
                         "A_varying": [k for k, v in (self.A[0].blocks.items() if self.A else [])
                                       if v.shape[0] > 1],
                         "B_varying": [k for k, v in (self.B[0].blocks.items() if self.B else [])
                                       if v.shape[0] > 1],
                         "Q_uniform": uniform}

    def pair(self, t, u_t, u_t1):
        """(A_t^T w, B_t^T w) with w = Q (A_t u_t + B_t u_{t+1})."""
        a, b = self.A[t], self.B[t]
        if not self.on_gpu:
            a, b = a.to(self.device), b.to(self.device)
        z = a.matvec(u_t) + b.matvec(u_t1)
        if self.Q_uniform:
            w = z @ self.Q[0].T
        else:
            w = torch.einsum("nij,nj->ni", self.Q, z)
        return a.rmatvec(w), b.rmatvec(w)

    def blocks(self, t, rows):
        """Dense (A^T Q A, B^T Q B, A^T Q B) of pair t for Gaussians `rows`."""
        a = {k: v if v.shape[0] == 1 else v[rows.to(v.device)]
             for k, v in self.A[t].blocks.items()}
        b = {k: v if v.shape[0] == 1 else v[rows.to(v.device)]
             for k, v in self.B[t].blocks.items()}
        n = rows.numel()

        def dense(blocks):
            M = torch.zeros(n, 12, 12, dtype=torch.float64, device=self.device)
            for (r, c), v in blocks.items():
                M[:, 3 * r:3 * r + 3, 3 * c:3 * c + 3] = v.to(self.device).expand(n, 3, 3)
            return M
        A, B = dense(a), dense(b)
        Q = self.Q.expand(n, 12, 12) if self.Q_uniform else self.Q[rows]
        QA, QB = Q @ A, Q @ B
        return A.transpose(1, 2) @ QA, B.transpose(1, 2) @ QB, A.transpose(1, 2) @ QB


# ---------------------------------------------------------------------------
# the operator
# ---------------------------------------------------------------------------

class CachedStructuredOperator:
    """The same A, b, D as `_CudaWindowOperator`, from frozen data."""

    def __init__(self, *, store: FrozenFrameStore, process: StructuredProcess,
                 observation, damping, frozen_groups=(), frozen_frames=(),
                 prior_information=None, prior_residual=None, work_device="cpu"):
        self.store, self.proc, self.obs = store, process, observation
        self.T, self.N = store.T, store.N
        self.damping = float(damping)
        self.work_device = torch.device(work_device)
        self.gpu = store.device
        self.prior_information = prior_information
        self.prior_residual = prior_residual
        self.process = []                 # legacy preconditioners must not run on this
        self.frozen_frames = tuple(int(t) for t in frozen_frames)
        self.frozen_groups = tuple((int(t), torch.as_tensor(i, dtype=torch.long))
                                   for t, i in frozen_groups)
        self._frozen_by_frame = {}
        for t in self.frozen_frames:
            self._frozen_by_frame.setdefault(t, []).append(None)
        for t, idx in self.frozen_groups:
            self._frozen_by_frame.setdefault(t, []).append(idx)
        self._curv = None
        self.matvecs = 0
        self.timing = {"image_s": 0.0, "process_s": 0.0, "transfer_s": 0.0,
                       "stage_s": 0.0}

    # -- elimination ----------------------------------------------------------
    def _freeze_frame(self, t, x):
        for idx in self._frozen_by_frame.get(t, ()):
            if idx is None:
                x[:, 0:6] = 0.0
            else:
                x[idx.to(x.device), 0:6] = 0.0
        return x

    def _freeze(self, u):
        out = u.clone()
        for t in range(self.T):
            self._freeze_frame(t, out[t])
        return out

    def zeros(self):
        return torch.zeros(self.T, self.N, ERROR_DIM, dtype=torch.float64,
                           device=self.work_device)

    # -- the quantities the solver reads ----------------------------------------
    def measurement_diagonal(self):
        return torch.stack(self.store.meas_diag).to(self.work_device)

    def curvature_scale(self):
        if self._curv is None:
            d = self.measurement_diagonal().clone() + self.proc.diag.to(self.work_device)
            if self.prior_information is not None:
                d[0] += self.prior_information.double().to(d.device).diagonal(
                    dim1=-2, dim2=-1)
            self._curv = d.abs().clamp_min(1e-12)
        return self._curv

    def rhs(self):
        b = torch.stack(self.store.gradient).to(self.work_device) + \
            self.proc.rhs.to(self.work_device)
        if self.prior_information is not None and self.prior_residual is not None:
            P0 = self.prior_information.double().to(b.device)
            b[0] -= torch.einsum("nij,nj->ni", P0,
                                 self.prior_residual.double().to(b.device))
        return self._freeze(b)

    def information_matvec(self, u):
        """A u, frame by frame: one H2D of u_t, one staged cache, one D2H."""
        self.matvecs += 1
        u = torch.as_tensor(u).double()
        if u.device != self.work_device:
            u = u.to(self.work_device)
        out = torch.empty_like(u)
        curv = self.curvature_scale() if self.damping else None
        prev_u = prev_acc = None
        P0 = (self.prior_information.double().to(self.gpu)
              if self.prior_information is not None else None)
        for t in range(self.T):
            tt = time.perf_counter()
            ut = self._freeze_frame(t, u[t].to(self.gpu, non_blocking=True).clone())
            self.timing["transfer_s"] += time.perf_counter() - tt
            ts = time.perf_counter()
            cache = self.store.stage(t)
            self.timing["stage_s"] += time.perf_counter() - ts
            ti = time.perf_counter()
            ref = cache.linearization.views[0].rgb
            acc = self.obs.information_apply(cache, ut.to(ref.dtype)).double()
            self.timing["image_s"] += time.perf_counter() - ti
            if P0 is not None and t == 0:
                acc = acc + torch.einsum("nij,nj->ni", P0, ut)
            if self.damping:
                acc = acc + self.damping * curv[t].to(self.gpu, non_blocking=True) * ut
            if prev_u is not None:
                tp = time.perf_counter()
                l, r = self.proc.pair(t - 1, prev_u, ut)
                prev_acc = prev_acc + l
                acc = acc + r
                self.timing["process_s"] += time.perf_counter() - tp
                tt = time.perf_counter()
                out[t - 1] = self._freeze_frame(t - 1, prev_acc).to(self.work_device)
                self.timing["transfer_s"] += time.perf_counter() - tt
            prev_u, prev_acc = ut, acc
        out[self.T - 1] = self._freeze_frame(self.T - 1, prev_acc).to(self.work_device)
        return out

    # -- preconditioners ----------------------------------------------------------
    def make_preconditioner(self, kind, damping, chunk=2048, verbose=False,
                            host_budget_bytes=None):
        if kind == "diagonal":
            return self._diagonal(damping)
        if kind in ("block12", "temporal_block_tridiagonal"):
            return TemporalBlockPreconditioner(self, damping, coupled=(
                kind == "temporal_block_tridiagonal"), chunk=chunk,
                host_budget_bytes=host_budget_bytes, verbose=verbose)
        raise ValueError(kind)

    def _diagonal(self, damping):
        """Exactly full_window_free._diagonal_jacobi_preconditioner's M."""
        d = torch.ones(self.T, self.N, ERROR_DIM, dtype=torch.float64,
                       device=self.work_device)
        if self.prior_information is not None:
            d[0] += self.prior_information.double().to(d.device).diagonal(dim1=-2, dim2=-1)
        d += self.measurement_diagonal()
        if damping:
            d = d + damping * self.curvature_scale()
        d += self.proc.diag.to(d.device)
        floor = 1e-9 * d.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        inv = 1.0 / d.clamp_min(floor)

        def apply(z):
            return inv * torch.as_tensor(z).double().to(inv.device)
        apply.kind = "diagonal"
        return apply


class TemporalBlockPreconditioner:
    """Per-Gaussian block-tridiagonal M over the whole trajectory.

    M_i has, at each frame, the image Fisher's same-Gaussian pose block, the
    process term J^T Q^-1 J of that Gaussian, the prior and the LM term; and
    between adjacent frames the process coupling A_t^T Q B_t. Cross-Gaussian
    image couplings are OMITTED HERE ONLY -- A keeps them. Eliminated anchor
    coordinates are identity rows/columns, so M^-1 never re-introduces them.

    `coupled=False` drops the temporal blocks: per-frame 12x12 block-Jacobi
    with the legacy definition (identity added, as
    full_window_free.block_jacobi_preconditioner does).

    Factorisation: block Cholesky recurrence S_0 = D_0,
    S_t = D_t - W_{t-1}^T W_{t-1} with W_{t-1} = L_{t-1}^{-1} E_{t-1},
    L_t = chol(S_t); no block is ever inverted. Factors of each chunk of
    Gaussians are built on the GPU and kept in host RAM.
    """

    def __init__(self, op: CachedStructuredOperator, damping, *, coupled=True,
                 chunk=2048, host_budget_bytes=None, verbose=False):
        if op.store.pose_fisher is None:
            raise ValueError("block preconditioners need the store built with "
                             "keep_pose_fisher=True")
        self.op, self.coupled = op, coupled
        self.kind = "temporal_block_tridiagonal" if coupled else "block12"
        T, N, dev = op.T, op.N, op.gpu
        # host RAM for the factors, checked BEFORE allocating anything
        factor_bytes = (2 if coupled else 1) * T * N * 144 * 8
        if host_budget_bytes is not None and \
                op.store.host_bytes + factor_bytes > host_budget_bytes:
            raise PreconditionerBudgetError(
                f"{self.kind}: factors need {factor_bytes / 2**30:.1f} GiB on top of "
                f"the {op.store.host_bytes / 2**30:.1f} GiB image cache, over the "
                f"{host_budget_bytes / 2**30:.1f} GiB host budget")
        # GPU chunk from the MEASURED free memory: D, E, L, W per chunk, each
        # [T, n, 12, 12] float64, plus process blocks and scratch (x2)
        free, _ = torch.cuda.mem_get_info(dev)
        per_gaussian = 8 * T * 144 * 8
        chunk = int(max(64, min(chunk, 0.3 * free // per_gaussian)))
        self.chunk = chunk
        t0 = time.perf_counter()
        self.L, self.W, self.ranges = [], [], []
        curv = op.curvature_scale()
        free = torch.ones(T, N, ERROR_DIM, dtype=torch.bool)
        for t, idxs in op._frozen_by_frame.items():
            for idx in idxs:
                if idx is None:
                    free[t, :, :6] = False
                else:
                    free[t, idx, :6] = False
        eye = torch.eye(ERROR_DIM, dtype=torch.float64, device=dev)
        for s in range(0, N, chunk):
            rows = torch.arange(s, min(s + chunk, N), device=dev)
            n = rows.numel()
            D = torch.zeros(T, n, 12, 12, dtype=torch.float64, device=dev)
            E = torch.zeros(max(T - 1, 1), n, 12, 12, dtype=torch.float64, device=dev)
            rc = rows.cpu()
            for t in range(T):
                D[t, :, :6, :6] += op.store.pose_fisher[t][rc].to(dev)
                if damping:
                    D[t] += torch.diag_embed(damping * curv[t][rc].to(dev))
            if op.prior_information is not None:
                D[0] += op.prior_information.double().to(dev)[rows]
            for t in range(T - 1):
                AQA, BQB, AQB = op.proc.blocks(t, rows)
                D[t] += AQA
                D[t + 1] += BQB
                if coupled:
                    E[t] = AQB
            if not coupled:
                D += eye                          # the legacy block-Jacobi definition
            fr = free[:, rc].to(dev)              # [T,n,12]
            mask = (fr.unsqueeze(-1) & fr.unsqueeze(-2)).double()
            D = D * mask + torch.diag_embed((~fr).double())
            if coupled:
                E = E * (fr[:-1].unsqueeze(-1) & fr[1:].unsqueeze(-2)).double()
            L = torch.empty_like(D)
            W = torch.empty_like(E) if coupled else None
            for t in range(T):
                S = D[t] if (t == 0 or not coupled) else D[t] - W[t - 1].transpose(1, 2) @ W[t - 1]
                S = 0.5 * (S + S.transpose(1, 2))
                Lt, info = torch.linalg.cholesky_ex(S)
                if bool((info != 0).any()):
                    bad = int(torch.nonzero(info)[0])
                    ev = torch.linalg.eigvalsh(S[bad]).min()
                    raise np_linalg_error(
                        f"{self.kind}: block at frame {t}, Gaussian {s + bad} is not "
                        f"positive definite (min eigenvalue {float(ev):.3e}); no jitter "
                        "is added -- fall back to another preconditioner")
                L[t] = Lt
                if coupled and t < T - 1:
                    W[t] = torch.linalg.solve_triangular(Lt, E[t], upper=False)
            self.L.append(L.cpu())
            if coupled:
                self.W.append(W.cpu())
            self.ranges.append((s, s + n))
            del D, E, L, W
        torch.cuda.empty_cache()
        self.build_s = time.perf_counter() - t0
        self.host_bytes = sum(x.numel() * 8 for x in self.L + self.W)
        if verbose:
            print(f"     {self.kind} preconditioner: {self.build_s:.1f} s, "
                  f"{self.host_bytes / 2**30:.2f} GiB host", flush=True)

    def __call__(self, r):
        op, dev = self.op, self.op.gpu
        r = torch.as_tensor(r).double()
        out = torch.empty_like(r, device=op.work_device)
        for k, (s, e) in enumerate(self.ranges):
            L = self.L[k].to(dev, non_blocking=True)
            rc = r[:, s:e].to(dev).unsqueeze(-1)            # [T,n,12,1]
            if not self.coupled:
                y = torch.linalg.solve_triangular(L, rc, upper=False)
                x = torch.linalg.solve_triangular(L.transpose(-1, -2), y, upper=True)
            else:
                W = self.W[k].to(dev, non_blocking=True)
                T = rc.shape[0]
                y = torch.empty_like(rc)
                for t in range(T):
                    rhs = rc[t] if t == 0 else rc[t] - W[t - 1].transpose(1, 2) @ y[t - 1]
                    y[t] = torch.linalg.solve_triangular(L[t], rhs, upper=False)
                x = torch.empty_like(rc)
                for t in range(T - 1, -1, -1):
                    rhs = y[t] if t == T - 1 else y[t] - W[t] @ x[t + 1]
                    x[t] = torch.linalg.solve_triangular(L[t].transpose(1, 2), rhs, upper=True)
            out[:, s:e] = x.squeeze(-1).to(op.work_device)
        return op._freeze(out)


class PreconditionerBudgetError(MemoryError):
    """The block preconditioner does not fit; the solver falls back honestly."""


class np_linalg_error(RuntimeError):
    """A preconditioner block that is not SPD; reported, never jittered."""
