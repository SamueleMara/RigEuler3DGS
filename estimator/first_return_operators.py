"""Matrix-free first-return depth operators on a production IEKFLinearizationCache.

The CUDA source is submodules/diff-gaussian-rasterization/cuda_rasterizer/
iekf_first_return.cu, compiled ONCE as a separate JIT extension (named by its
source hash) next to this checkout. The production `_C` / `_C_vis` modules are
never rebuilt. Every call reads the frozen cache of ONE linearization; nothing
is stored between linearizations.

Per camera, given the observation maps (D, valid = M AND V AND finite, sigma_D at
the observed depth, unknown sd) and the camera's prior (lambda, z_near, z_far):
    fr_eval       log L per pixel (float64) + diagnostics + sum_p gamma/sigma^2 per Gaussian
    fr_vjp        sum_p w_p d logL_p / d pose            [P,6]   (parts: 1 match, 2 visibility)
    fr_jvp        d logL_p / d pose . v                   [H,W]
    fr_fisher_vis per-Gaussian 6x6 blocks of sum_p g_p g_p^T, g_p = visibility score   [P,21]
    fr_dump       the full contributor list (CSR), tests / audits only
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
RAST = REPO / "submodules" / "diff-gaussian-rasterization"
SOURCE = RAST / "cuda_rasterizer" / "iekf_first_return.cu"
PART_MATCH, PART_VIS, PART_ALL = 1, 2, 3
_EXT = None


def source_sha() -> str:
    h = hashlib.sha256()
    for f in (SOURCE, RAST / "cuda_rasterizer" / "iekf_device.cuh", RAST / "cuda_rasterizer" / "auxiliary.h"):
        h.update(f.read_bytes())
    return h.hexdigest()


def ext():
    """Build (once per source hash) and return the first-return extension."""
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
        build = RAST / ".build_first_return" / sha
        build.mkdir(parents=True, exist_ok=True)
        _EXT = cpp_ext.load(name=f"rigeuler_first_return_{sha}", sources=[str(SOURCE)],
                            extra_include_paths=[str(RAST / "cuda_rasterizer")],
                            extra_cuda_cflags=["-O3"], build_directory=str(build), verbose=False)
    return _EXT


def camera_args(cache, D, valid, sig_d, sig_u, lam, z_near, z_far):
    f = lambda t: t.contiguous().float()                                       # noqa: E731
    return (cache.ranges.contiguous().int(), cache.point_list.contiguous().int(),
            int(cache.image_width), int(cache.image_height),
            f(cache.means3D), f(cache.mu_c), f(cache.u), f(cache.Jp), f(cache.sigma_c), f(cache.conic),
            f(cache.opacity.reshape(-1)), f(cache.depth), cache.clamp_flags.contiguous().int(),
            f(cache.viewmatrix), f(cache.projmatrix), float(cache.tanfovx), float(cache.tanfovy),
            f(D), valid.contiguous().to(torch.uint8), f(sig_d), f(sig_u),
            float(lam), float(z_near), float(z_far))


def depth_axis(cache) -> torch.Tensor:
    """e_z^T R_cw: d z / d mu for this camera (row-vector viewmatrix convention)."""
    vm = cache.viewmatrix.reshape(-1).float()
    return torch.stack([vm[2], vm[6], vm[10]])


def fr_eval(args, diagnostics=True):
    logL, gU, gmax, front, nsupp, wm, gs, fs, pc = ext().fr_eval(*args, 1 if diagnostics else 0)
    return {"logL": logL, "gamma_unknown": gU, "gamma_max": gmax, "front": front, "n_support": nsupp,
            "w_match": wm, "gamma_sum": gs, "front_sum": fs, "pixel_count": pc}


def fr_vjp(args, logL, wpix, parts=PART_ALL):
    return ext().fr_vjp(*args, logL.contiguous(), wpix.contiguous().float(), int(parts))


def fr_jvp(args, logL, dpose, parts=PART_ALL):
    return ext().fr_jvp(*args, logL.contiguous(), dpose.contiguous().float(), int(parts))


def fr_fisher_vis(args, logL):
    return ext().fr_fisher_vis(*args, logL.contiguous())


def fr_dump(args):
    offsets, gids, alphas, zs = ext().fr_dump(*args)
    return {"offsets": offsets, "gid": gids, "alpha": alphas, "z": zs}


def sym21_to_dense(f21: torch.Tensor) -> torch.Tensor:
    out = f21.new_zeros((f21.shape[0], 6, 6))
    t = 0
    for a in range(6):
        for b in range(a, 6):
            out[:, a, b] = f21[:, t]
            out[:, b, a] = f21[:, t]
            t += 1
    return out
