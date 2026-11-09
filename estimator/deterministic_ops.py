"""Deterministic (order-independent) IEKF reductions, opt-in.

The production IEKF kernels accumulate per-Gaussian adjoints, Fisher blocks and group
reductions with float `atomicAdd`, whose result depends on thread scheduling. On the real
fridge the 80->81 render is bit-identical across runs but the RHS / normal / Fisher are
not (~1e-7 relative), and 12 unconverged inner iterations x 20 frames amplify that to
metre-scale differences in the tail. This module builds a SEPARATE JIT copy of the three
kernel files in which each of those float atomics becomes

    pass 0:  global max |v| (atomicMax on the float bits) and the number of terms
    pass 1:  int64 fixed-point accumulation with scale 2^floor(62 - log2(count * max|v|))

Integer addition is associative, so the result is bitwise reproducible, and the scale
bounds every partial sum (no overflow) with an absolute resolution of
count * max|v| / 2^62 -- far finer than float32 accumulation. The copies are generated
from the CURRENT production sources at build time (a regex over `atomicAdd(&buf[i], v)`;
the expected site count is asserted), so they cannot drift from production.

`enable()` swaps the six reduction entry points of diff_gaussian_rasterization in THIS
process only (the production build is never modified), and turns on torch's
deterministic algorithms. Renders, JVPs and every other operator are unchanged.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
RAST = REPO / "submodules" / "diff-gaussian-rasterization"
GLM_CANDIDATES = [RAST / "third_party" / "glm",
                  Path("/home/samuelemara/Joint3DGS/RigEuler3DGS/submodules/diff-gaussian-rasterization/third_party/glm")]
KERNELS = {"iekf": ("cuda_rasterizer/iekf.cu", 5), "group": ("cuda_rasterizer/iekf_group.cu", 7),
           "vis": ("cuda_rasterizer/iekf_visibility.cu", 2)}
ATOMIC = re.compile(r"atomicAdd\(\s*&\s*(\w+)\s*\[([^\]]+)\]\s*,", re.S)
_EXT = None
_ENABLED = False

HEADER = r'''
#pragma once
#include <cuda_runtime.h>
#include <cstring>
// Deterministic two-pass fixed-point accumulation (estimator/deterministic_ops.py).
static __device__ int det_pass;
static __device__ unsigned int det_maxbits;
static __device__ unsigned long long det_count;
static __device__ long long* det_acc;
static __device__ double det_scale;
static __device__ __forceinline__ void det_add_impl(float* buf, long long idx, float v)
{
	if (det_pass == 0) {
		atomicMax(&det_maxbits, __float_as_uint(fabsf(v)));
		atomicAdd(&det_count, 1ULL);
	} else {
		const long long q = llrint((double)v * det_scale);
		atomicAdd((unsigned long long*)&det_acc[idx], (unsigned long long)q);
	}
}
static inline float __uint_as_float_host(unsigned int b) { float f; memcpy(&f, &b, 4); return f; }
#define DET_ADD(buf, idx, v) det_add_impl((buf), (long long)(idx), (float)(v))
#define DET_API(SUFFIX)                                                                          \
	void det_begin_##SUFFIX(int pass, long long* acc, double scale) {                            \
		unsigned int z = 0; unsigned long long zc = 0;                                           \
		cudaMemcpyToSymbol(det_pass, &pass, sizeof(int));                                        \
		cudaMemcpyToSymbol(det_acc, &acc, sizeof(long long*));                                   \
		cudaMemcpyToSymbol(det_scale, &scale, sizeof(double));                                   \
		if (pass == 0) { cudaMemcpyToSymbol(det_maxbits, &z, sizeof(z));                         \
		                 cudaMemcpyToSymbol(det_count, &zc, sizeof(zc)); }                       \
	}                                                                                            \
	void det_bound_##SUFFIX(float* maxabs, unsigned long long* count) {                          \
		unsigned int b; cudaMemcpyFromSymbol(&b, det_maxbits, sizeof(b));                        \
		cudaMemcpyFromSymbol(count, det_count, sizeof(*count));                                  \
		*maxabs = __uint_as_float_host(b);                                                       \
	}
'''

WRAP = r'''
#include <torch/extension.h>
#include <tuple>
#include "iekf_points.h"
void det_begin_iekf(int, long long*, double); void det_bound_iekf(float*, unsigned long long*);
void det_begin_group(int, long long*, double); void det_bound_group(float*, unsigned long long*);
void det_begin_vis(int, long long*, double); void det_bound_vis(float*, unsigned long long*);
torch::Tensor IEKFVisApplyHTCUDA(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                   const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                   const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                   const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const float, const float,
                   const int, const int, const torch::Tensor&, const torch::Tensor&, const float,
                   const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const bool);
torch::Tensor IEKFVisBuildFisherCUDA(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                       const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                       const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                       const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const float, const float,
                       const int, const int, const torch::Tensor&, const torch::Tensor&, const float,
                       const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const bool);
static void begin(const std::string& k, int pass, torch::Tensor acc, double scale) {
	long long* p = acc.defined() && acc.numel() ? reinterpret_cast<long long*>(acc.data_ptr<int64_t>()) : nullptr;
	if (k == "iekf") det_begin_iekf(pass, p, scale);
	else if (k == "group") det_begin_group(pass, p, scale);
	else det_begin_vis(pass, p, scale);
}
static std::tuple<double, double> bound(const std::string& k) {
	float m = 0.f; unsigned long long c = 0;
	if (k == "iekf") det_bound_iekf(&m, &c); else if (k == "group") det_bound_group(&m, &c); else det_bound_vis(&m, &c);
	return {(double)m, (double)c};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("det_begin", &begin);
	m.def("det_bound", &bound);
	m.def("iekf_apply_HT", &IEKFApplyHTCUDA);
	m.def("iekf_build_fisher", &IEKFBuildFisherCUDA);
	m.def("iekf_reduce_group_adjoint", &IEKFReduceGroupAdjointCUDA);
	m.def("iekf_reduce_group_fisher", &IEKFReduceGroupFisherCUDA);
	m.def("iekf_vis_apply_HT", &IEKFVisApplyHTCUDA);
	m.def("iekf_vis_build_fisher", &IEKFVisBuildFisherCUDA);
}
'''


def _sources() -> dict:
    out = {}
    for key, (rel, expected) in KERNELS.items():
        src = (RAST / rel).read_text()
        new, n = ATOMIC.subn(lambda m: f"DET_ADD({m.group(1)}, {m.group(2)},", src)
        if n != expected:
            raise RuntimeError(f"{rel}: {n} atomicAdd sites, expected {expected} -- the production kernel changed")
        out[key] = '#include "det_accum.cuh"\n' + new + f"\nDET_API({key})\n"
    return out


def ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    import torch.utils.cpp_extension as cpp_ext
    want = f"/usr/local/cuda-{torch.version.cuda}"
    if Path(want, "bin", "nvcc").exists():
        os.environ["CUDA_HOME"] = want
        cpp_ext.CUDA_HOME = want
    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    srcs = _sources()
    h = hashlib.sha256((HEADER + WRAP + "".join(srcs.values())
                        + (RAST / "iekf_points.cu").read_text() + (RAST / "iekf_visibility_points.cu").read_text()
                        ).encode()).hexdigest()[:12]
    build = RAST / ".build_deterministic" / h
    (RAST / ".build_deterministic").mkdir(parents=True, exist_ok=True)
    (RAST / ".build_deterministic" / ".gitignore").write_text("*\n")     # build artefacts, never tracked
    gen = build / "src"
    (gen / "cuda_rasterizer").mkdir(parents=True, exist_ok=True)
    (gen / "cuda_rasterizer" / "det_accum.cuh").write_text(HEADER)
    files = []
    for key, (rel, _n) in KERNELS.items():
        p = gen / rel
        p.write_text(srcs[key])
        files.append(str(p))
    wrap = gen / "det_wrap.cu"
    wrap.write_text(WRAP)
    files += [str(RAST / "iekf_points.cu"), str(RAST / "iekf_visibility_points.cu"), str(wrap)]
    glm = next((g for g in GLM_CANDIDATES if g.is_dir()), None)
    inc = [str(gen / "cuda_rasterizer"), str(RAST / "cuda_rasterizer"), str(RAST)] + ([str(glm)] if glm else [])
    _EXT = cpp_ext.load(name=f"rigeuler_det_{h}", sources=files, extra_include_paths=inc,
                        extra_cuda_cflags=["-O3"], build_directory=str(build), verbose=False)
    return _EXT


def _two_pass(kernel_key, fn, out_dtype=torch.float32):
    """Wrap an extension function whose float output is accumulated with DET_ADD."""
    def run(*args):
        e = ext()
        e.det_begin(kernel_key, 0, torch.empty(0, dtype=torch.int64), 0.0)
        shape_probe = fn(*args)                         # pass 0: bounds only, output stays zero
        mx, cnt = e.det_bound(kernel_key)
        if not (mx > 0 and cnt > 0) or mx != mx or mx == float("inf"):
            if mx != mx or mx == float("inf"):
                raise FloatingPointError(f"{kernel_key}: non-finite term in a deterministic reduction")
            return torch.zeros_like(shape_probe)
        import math
        scale = 2.0 ** math.floor(62 - math.log2(cnt * mx) - 1)
        acc = torch.zeros(shape_probe.numel(), dtype=torch.int64, device=shape_probe.device)
        e.det_begin(kernel_key, 1, acc, scale)
        fn(*args)
        torch.cuda.synchronize()
        e.det_begin(kernel_key, 0, torch.empty(0, dtype=torch.int64), 0.0)
        return (acc.double() / scale).to(out_dtype).reshape(shape_probe.shape)
    return run


def enable() -> dict:
    """Swap the six atomic reductions for their deterministic versions in this process."""
    global _ENABLED
    if _ENABLED:
        return {"deterministic": True}
    import diff_gaussian_rasterization as dgr
    from diff_gaussian_rasterization import _C
    e = ext()
    _C_vis = dgr._vis()
    for name, key, mod in (("iekf_apply_HT", "iekf", _C), ("iekf_build_fisher", "iekf", _C),
                           ("iekf_reduce_group_adjoint", "group", _C), ("iekf_reduce_group_fisher", "group", _C),
                           ("iekf_vis_apply_HT", "vis", _C_vis), ("iekf_vis_build_fisher", "vis", _C_vis)):
        setattr(mod, name, _two_pass(key, getattr(e, name)))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    _ENABLED = True
    return {"deterministic": True, "extension": e.__name__, "torch_deterministic_algorithms": True}


def is_enabled() -> bool:
    return _ENABLED
