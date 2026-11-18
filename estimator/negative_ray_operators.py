"""Matrix-free negative-ray (known non-object depth) operators on a production
IEKFLinearizationCache.

The CUDA source is submodules/diff-gaussian-rasterization/cuda_rasterizer/
iekf_negative_ray.cu, compiled ONCE as a separate JIT extension (named by its source
hash) next to this checkout, exactly like estimator/first_return_operators.py. The
production `_C` / `_C_vis` modules are never rebuilt. Every call reads the frozen cache
of ONE linearization.

Per camera, given the observation maps (raw sensor depth D, sigma at the observed
depth, the per-pixel status of estimator/occlusion_likelihood.py) and the per-pixel
reliability q_p in [0, 1] (estimator/negative_ray_reliability.py):
    nr_eval    A_p = sum_k w_k Phi((D - z_k)/sigma), L_p = -log(1 - q_p A_p) (float64)
               on KNOWN_NONOBJECT pixels; diagnostics: per-Gaussian weights / counts
    nr_vjp     sum_p c_p dA_p / d pose            [P,6]
    nr_jvp     dA_p / d pose . v                   [H,W]
    nr_fisher  per-Gaussian 6x6 blocks of sum_p c_p g g^T, g = dA_p/dx_i   [P,21]
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
RAST = REPO / "submodules" / "diff-gaussian-rasterization"
SOURCE = RAST / "cuda_rasterizer" / "iekf_negative_ray.cu"
ST_NONE, ST_SUPPORTED, ST_NONOBJECT, ST_UNKNOWN = 0, 1, 2, 3
STATUS_NAMES = {ST_NONE: "NONE", ST_SUPPORTED: "OBJECT_SUPPORTED", ST_NONOBJECT: "KNOWN_NONOBJECT_DEPTH",
                ST_UNKNOWN: "UNKNOWN"}
_EXT = None


def source_sha() -> str:
    h = hashlib.sha256()
    for f in (SOURCE, RAST / "cuda_rasterizer" / "iekf_device.cuh", RAST / "cuda_rasterizer" / "auxiliary.h"):
        h.update(f.read_bytes())
    return h.hexdigest()


def ext():
    """Build (once per source hash) and return the negative-ray extension."""
    global _EXT
    if _EXT is None:
        import torch.utils.cpp_extension as cpp_ext
        want = f"/usr/local/cuda-{torch.version.cuda}"
        if Path(want, "bin", "nvcc").exists():
            os.environ["CUDA_HOME"] = want
            cpp_ext.CUDA_HOME = want
        if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        sha = source_sha()[:12]
        build = RAST / ".build_negative_ray" / sha
        build.mkdir(parents=True, exist_ok=True)
        _EXT = cpp_ext.load(name=f"rigeuler_negative_ray_{sha}", sources=[str(SOURCE)],
                            extra_include_paths=[str(RAST / "cuda_rasterizer")],
                            extra_cuda_cflags=["-O3"], build_directory=str(build), verbose=False)
    return _EXT


def camera_args(cache, D, sig, status, q):
    """q: [H,W] per-pixel reliability in [0, 1] (a python float is broadcast; tests only)."""
    f = lambda t: t.contiguous().float()                                       # noqa: E731
    if not torch.is_tensor(q):
        q = torch.full_like(D, float(q))
    if tuple(q.shape) != tuple(D.shape):
        raise ValueError(f"reliability map {tuple(q.shape)} vs depth {tuple(D.shape)}")
    return (cache.ranges.contiguous().int(), cache.point_list.contiguous().int(),
            int(cache.image_width), int(cache.image_height),
            f(cache.means3D), f(cache.mu_c), f(cache.u), f(cache.Jp), f(cache.sigma_c), f(cache.conic),
            f(cache.opacity.reshape(-1)), f(cache.depth), cache.clamp_flags.contiguous().int(),
            f(cache.viewmatrix), f(cache.projmatrix), float(cache.tanfovx), float(cache.tanfovy),
            f(D), f(sig), status.contiguous().to(torch.uint8), f(q.to(D.device)))


def nr_eval(args, diagnostics=False):
    A, L, fh, pw, fw, ow, uw, pn, nn, un = ext().nr_eval(*args, 1 if diagnostics else 0)
    out = {"A": A, "L": L, "floor_hits": int(fh.item())}
    if diagnostics:
        out.update({"supported_weight": pw, "front_weight": fw, "occluded_weight": ow, "unknown_weight": uw,
                    "supported_pixels": pn, "nonobject_pixels": nn, "unknown_pixels": un})
    return out


def nr_vjp(args, A, wpix, deterministic=False):
    return ext().nr_vjp(*args, A.contiguous(), wpix.contiguous().float(), bool(deterministic))


def nr_jvp(args, A, dpose):
    return ext().nr_jvp(*args, A.contiguous(), dpose.contiguous().float())


def nr_fisher(args, A, wpix, deterministic=False):
    return ext().nr_fisher(*args, A.contiguous(), wpix.contiguous().float(), bool(deterministic))


def sym21_to_dense(f21: torch.Tensor) -> torch.Tensor:
    out = f21.new_zeros((f21.shape[0], 6, 6))
    t = 0
    for a in range(6):
        for b in range(a, 6):
            out[:, a, b] = f21[:, t]
            out[:, b, a] = f21[:, t]
            t += 1
    return out
