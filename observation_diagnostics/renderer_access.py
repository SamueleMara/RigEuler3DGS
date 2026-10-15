"""Exact per-pixel, per-Gaussian contributions from the production renderer.

The production IEKF render kernel (`renderFromCacheCUDA` in
submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf.cu) composites
front to back and returns only per-pixel totals. `ranges`/`point_list` in its
cache are the tile candidate lists and the depth sort; they are NOT the set of
Gaussians that actually contribute to a pixel. This module adds a strictly
opt-in, standalone diagnostic kernel that walks the SAME frozen cache with the
SAME loop -- same tile, same sorted order, same `evalAlpha` (power > 0 skipped,
alpha = opacity * __expf(power), clamped to 0.99, skipped below 1/255), same
early termination (the contributor that would take T below 1e-4 is NOT
composited) -- and writes one record per composited contributor:

    gid      source Gaussian row
    alpha    the effective opacity alpha[p, i]
    T        the transmittance immediately BEFORE it
    q        = alpha * T, the exact weight it carries in RGB, A and Z

The production kernels are not modified and not rebuilt. Equivalence is
tested (tests/test_fb_pixel_overlap_core.py T1, T10): sum_i q equals the
renderer's A, sum_i q z_i / (A + eps) its depth, and sum_i q c_i + T bg its RGB.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch

_KERNEL = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

// Verbatim from iekf.cu: the constants and evalAlpha.
__device__ const float ALPHA_MAX = 0.99f;
__device__ const float ALPHA_MIN = 1.0f / 255.0f;
__device__ const float T_STOP = 1e-4f;

__forceinline__ __device__ bool evalAlpha(
	const float ux, const float uy, const float qxx, const float qxy, const float qyy,
	const float opacity, const float px, const float py,
	float& alpha, float2& d, bool& clamped)
{
	d.x = ux - px; d.y = uy - py;
	const float power = -0.5f * (qxx * d.x * d.x + qyy * d.y * d.y) - qxy * d.x * d.y;
	if (power > 0.0f) return false;
	float a = opacity * __expf(power);
	clamped = false;
	if (a > ALPHA_MAX) { a = ALPHA_MAX; clamped = true; }
	if (a < ALPHA_MIN) return false;
	alpha = a;
	return true;
}

// The loop of renderFromCacheCUDA, with its outputs replaced by contributor
// records. emit == 0: count composited contributors per pixel. emit == 1:
// write them at offsets[pix] in front-to-back order.
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
contribCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ u,
	const float* __restrict__ conic,
	const float* __restrict__ opacity,
	int emit,
	int* __restrict__ counts,
	const long* __restrict__ offsets,
	int* __restrict__ out_gid,
	float* __restrict__ out_alpha,
	float* __restrict__ out_T,
	float* __restrict__ out_finalT,
	int* __restrict__ out_term)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const bool inside = pix.x < (unsigned)W && pix.y < (unsigned)H;
	bool done = !inside;

	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_u[BLOCK_SIZE];
	__shared__ float4 collected_co[BLOCK_SIZE];

	float T = 1.0f;
	int n = 0;
	int term = 0;
	long base = (emit && inside) ? offsets[pix_id] : 0;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == BLOCK_SIZE) break;
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int gid = point_list[range.x + progress];
			collected_id[block.thread_rank()] = gid;
			collected_u[block.thread_rank()] = {u[2 * gid], u[2 * gid + 1]};
			collected_co[block.thread_rank()] = {conic[3 * gid], conic[3 * gid + 1],
			                                     conic[3 * gid + 2], opacity[gid]};
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			const float2 cu = collected_u[j];
			const float4 co = collected_co[j];
			float alpha; float2 d; bool cl;
			if (!evalAlpha(cu.x, cu.y, co.x, co.y, co.z, co.w, pixf.x, pixf.y, alpha, d, cl))
				continue;
			const float test_T = T * (1.f - alpha);
			if (test_T < T_STOP) { done = true; term = 1; continue; }
			if (emit) {
				out_gid[base + n] = collected_id[j];
				out_alpha[base + n] = alpha;
				out_T[base + n] = T;
			}
			n++;
			T = test_T;
		}
	}
	if (inside) {
		if (!emit) counts[pix_id] = n;
		else { out_finalT[pix_id] = T; out_term[pix_id] = term; }
	}
}

void contrib(torch::Tensor ranges, torch::Tensor point_list, int W, int H,
             torch::Tensor u, torch::Tensor conic, torch::Tensor opacity,
             int emit, torch::Tensor counts, torch::Tensor offsets,
             torch::Tensor out_gid, torch::Tensor out_alpha, torch::Tensor out_T,
             torch::Tensor out_finalT, torch::Tensor out_term)
{
	const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	contribCUDA<<<tile_grid, block>>>(
		(const uint2*)ranges.data_ptr<int>(), (const uint32_t*)point_list.data_ptr<int>(),
		W, H, u.data_ptr<float>(), conic.data_ptr<float>(), opacity.data_ptr<float>(),
		emit, counts.data_ptr<int>(),
		emit ? offsets.data_ptr<long>() : nullptr,
		emit ? out_gid.data_ptr<int>() : nullptr,
		emit ? out_alpha.data_ptr<float>() : nullptr,
		emit ? out_T.data_ptr<float>() : nullptr,
		emit ? out_finalT.data_ptr<float>() : nullptr,
		emit ? out_term.data_ptr<int>() : nullptr);
	C10_CUDA_CHECK(cudaGetLastError());
}
"""

_CPP = "void contrib(torch::Tensor ranges, torch::Tensor point_list, int W, int H, " \
       "torch::Tensor u, torch::Tensor conic, torch::Tensor opacity, int emit, " \
       "torch::Tensor counts, torch::Tensor offsets, torch::Tensor out_gid, " \
       "torch::Tensor out_alpha, torch::Tensor out_T, torch::Tensor out_finalT, " \
       "torch::Tensor out_term);"

KERNEL_SHA256 = hashlib.sha256(_KERNEL.encode()).hexdigest()
_EXT = None


def _ext():
    """Build (once, cached by torch) and return the diagnostic extension."""
    global _EXT
    if _EXT is None:
        import torch.utils.cpp_extension as cpp_ext
        # The toolkit must match the one torch was built with (the production
        # rasterizer was built the same way); /usr/local/cuda may point at an
        # older one whose nvcc refuses this gcc. torch reads CUDA_HOME once at
        # import, so the module attribute is set, not only the environment.
        want = f"/usr/local/cuda-{torch.version.cuda}"
        if Path(want, "bin", "nvcc").exists():
            os.environ["CUDA_HOME"] = want
            cpp_ext.CUDA_HOME = want
        if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(
            name=f"fb_contrib_diag_{KERNEL_SHA256[:12]}",
            cpp_sources=_CPP, cuda_sources=_KERNEL,
            functions=["contrib"], verbose=False,
            extra_cuda_cflags=["-O3"])
    return _EXT


@dataclass
class Contributions:
    """One view's composited contributors, CSR over pixels (row-major y*W+x)."""

    H: int
    W: int
    offsets: torch.Tensor    # [H*W+1] int64
    gid: torch.Tensor        # [M] int32 source Gaussian row
    alpha: torch.Tensor      # [M] float32
    T: torch.Tensor          # [M] float32, transmittance before the contributor
    final_T: torch.Tensor    # [H*W] float32, transmittance after the last one
    terminated: torch.Tensor  # [H*W] int32, 1 = stopped at T_STOP (early termination)

    @property
    def q(self) -> torch.Tensor:
        # The kernel's own product: w = alpha * T (iekf.cu line `w = alpha * T`).
        return self.alpha * self.T

    @property
    def counts(self) -> torch.Tensor:
        return (self.offsets[1:] - self.offsets[:-1])

    @property
    def pixel(self) -> torch.Tensor:
        """[M] int64 pixel id of every record."""
        return torch.repeat_interleave(
            torch.arange(self.H * self.W, device=self.offsets.device), self.counts)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for t in (self.offsets, self.gid, self.alpha, self.T, self.final_T))


def pixel_contributions(cache) -> Contributions:
    """Exact contributor records for one production `IEKFLinearizationCache`."""
    ext = _ext()
    H, W = int(cache.image_height), int(cache.image_width)
    dev = cache.u.device
    opacity = cache.opacity.reshape(-1).contiguous().float()
    args = (cache.ranges.contiguous(), cache.point_list.contiguous(), W, H,
            cache.u.contiguous().float(), cache.conic.contiguous().float(), opacity)
    counts = torch.zeros(H * W, dtype=torch.int32, device=dev)
    empty_l = torch.empty(0, dtype=torch.long, device=dev)
    empty_i = torch.empty(0, dtype=torch.int32, device=dev)
    empty_f = torch.empty(0, dtype=torch.float32, device=dev)
    if cache.point_list.numel() == 0:
        return Contributions(H, W, torch.zeros(H * W + 1, dtype=torch.long, device=dev),
                             empty_i, empty_f, empty_f,
                             torch.ones(H * W, device=dev),
                             torch.zeros(H * W, dtype=torch.int32, device=dev))
    ext.contrib(*args, 0, counts, empty_l, empty_i, empty_f, empty_f, empty_f, empty_i)
    offsets = torch.zeros(H * W + 1, dtype=torch.long, device=dev)
    torch.cumsum(counts.long(), 0, out=offsets[1:])
    M = int(offsets[-1])
    gid = torch.empty(M, dtype=torch.int32, device=dev)
    alpha = torch.empty(M, dtype=torch.float32, device=dev)
    T = torch.empty(M, dtype=torch.float32, device=dev)
    final_T = torch.empty(H * W, dtype=torch.float32, device=dev)
    term = torch.empty(H * W, dtype=torch.int32, device=dev)
    ext.contrib(*args, 1, counts, offsets, gid, alpha, T, final_T, term)
    return Contributions(H, W, offsets, gid, alpha, T, final_T, term)


def reference_composite(cache, W: int, H: int) -> dict:
    """Independent front-to-back compositor in float64 Python, for tests.

    It reads only the TILE CANDIDATE LISTS from the cache (ranges/point_list,
    i.e. which Gaussians the binning assigned to a tile and their depth order)
    plus each Gaussian's projected mean, conic and opacity, and re-derives
    every pixel's contributors from the rasterizer's documented rules with its
    own arithmetic (math.exp in float64). Returns {pixel: [(gid, alpha, T)]}.
    """
    import math
    ranges = cache.ranges.cpu().numpy()
    plist = cache.point_list.cpu().numpy()
    u = cache.u.double().cpu().numpy()
    conic = cache.conic.double().cpu().numpy()
    opac = cache.opacity.reshape(-1).double().cpu().numpy()
    tiles_x = (W + 15) // 16
    out = {}
    for y in range(H):
        for x in range(W):
            tile = (y // 16) * tiles_x + (x // 16)
            a0, a1 = ranges[tile]
            T, recs = 1.0, []
            for g in plist[a0:a1]:
                dx, dy = u[g, 0] - x, u[g, 1] - y
                qxx, qxy, qyy = conic[g]
                power = -0.5 * (qxx * dx * dx + qyy * dy * dy) - qxy * dx * dy
                if power > 0:
                    continue
                a = min(opac[g] * math.exp(power), 0.99)
                if a < 1.0 / 255.0:
                    continue
                if T * (1 - a) < 1e-4:
                    break
                recs.append((int(g), a, T))
                T *= 1 - a
            out[y * W + x] = recs
    return out
