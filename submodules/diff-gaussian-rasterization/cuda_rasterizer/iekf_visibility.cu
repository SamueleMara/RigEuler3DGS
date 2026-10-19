/*
 * Visible-object-alpha operators for negative mask rays (19 Oct addendum,
 * Bidirectional_Articulated_3DGS_IEKF_Background_Visibility_Updated, sec.
 * "Visible object alpha").
 *
 * For a pixel p with ray mode m_p:
 *   m_p = 0  no negative-ray claim: A_vis = 0, no derivative
 *   m_p = 1  trusted full-empty ray: g_i = 1 for every contributor, A_vis = A
 *   m_p = 2  measured non-object surface at fixed z_obs_p:
 *            g_i = 1[d_i < z_obs_p - delta_z], d_i the contributor's OWN
 *            camera depth (never the composited normalized depth)
 *   beta_i = g_i alpha_i,  T^f_1 = 1,  T^f_{i+1} = T^f_i (1 - beta_i),
 *   A_vis = 1 - T^f_end.
 * The contributor set and its order are the renderer's: the walk terminates
 * on the FULL transmittance exactly as renderFromCacheCUDA does, so A_vis is a
 * function of the same composited contributors as the RGB-D prediction. The
 * hard gate is frozen within a branch (dg = 0); a transition is a branch
 * change, re-evaluated at the next nonlinear render.
 *
 *   dT^f_{i+1} = (1 - beta_i) dT^f_i - T^f_i g_i dalpha_i,   dA_vis = -dT^f_end
 *   dA_vis/dalpha_i = g_i T^f_i prod_{j>i}(1 - beta_j)        (VJP suffix)
 *
 * dalpha_i, the clamp and the per-Gaussian pose tangent come from the shared
 * iekf_device.cuh, so these operators differentiate exactly what iekf.cu does.
 */
#include "iekf_visibility.h"

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdexcept>

#include "auxiliary.h"
#include "config.h"
#include "iekf_device.cuh"

namespace cg = cooperative_groups;

#define VIS_SYNC(debug)                                                          \
	if (debug) {                                                                 \
		auto ret = cudaDeviceSynchronize();                                      \
		if (ret != cudaSuccess) {                                                \
			printf("\n[IEKF-VIS CUDA ERROR] %s in %s:%d\n",                      \
			       cudaGetErrorString(ret), __FILE__, __LINE__);                 \
			throw std::runtime_error(cudaGetErrorString(ret));                   \
		}                                                                        \
	}

namespace
{

__forceinline__ __device__ float gateOf(int mode, float d_i, float z_obs, float delta_z)
{
	if (mode == 1) return 1.f;
	if (mode == 2) return (d_i < z_obs - delta_z) ? 1.f : 0.f;
	return 0.f;
}

// ---------------------------------------------------------------------------
// forward: A_vis, T^f_end and the renderer's own last-contributor index
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
visAlphaCUDA(
	const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ u, const float* __restrict__ conic,
	const float* __restrict__ opacity, const float* __restrict__ depths,
	const int* __restrict__ ray_mode, const float* __restrict__ z_obs, float delta_z,
	float* __restrict__ out_avis, float* __restrict__ out_Tf, uint32_t* __restrict__ out_n_contrib,
	int* __restrict__ out_front_count, int* __restrict__ out_behind_count)
{
	auto block = cg::this_thread_block();
	const uint32_t hb = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const bool inside = pix.x < (unsigned)W && pix.y < (unsigned)H;
	bool done = !inside;
	const int mode = inside ? ray_mode[pix_id] : 0;
	const float zo = inside ? z_obs[pix_id] : 0.f;

	const uint2 range = ranges[block.group_index().y * hb + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;
	__shared__ int sh_id[BLOCK_SIZE];

	float T = 1.f, Tf = 1.f;
	uint32_t contributor = 0, last_contributor = 0;
	int n_front = 0, n_behind = 0;
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == BLOCK_SIZE) break;
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) sh_id[block.thread_rank()] = point_list[range.x + progress];
		block.sync();
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			++contributor;
			const int gid = sh_id[j];
			float alpha; float2 d; bool cl;
			if (!evalAlpha(u[2 * gid], u[2 * gid + 1], conic[3 * gid], conic[3 * gid + 1], conic[3 * gid + 2],
			               opacity[gid], pixf.x, pixf.y, alpha, d, cl))
				continue;
			const float test_T = T * (1.f - alpha);
			if (test_T < T_STOP) { done = true; continue; }
			last_contributor = contributor;
			const float g = gateOf(mode, depths[gid], zo, delta_z);
			if (mode == 2) { if (g > 0.f) ++n_front; else ++n_behind; }
			Tf *= (1.f - g * alpha);
			T = test_T;
		}
	}
	if (inside)
	{
		out_avis[pix_id] = (mode > 0) ? 1.f - Tf : 0.f;
		out_Tf[pix_id] = Tf;
		out_n_contrib[pix_id] = last_contributor;
		out_front_count[pix_id] = n_front;
		out_behind_count[pix_id] = n_behind;
	}
}

// ---------------------------------------------------------------------------
// JVP: dA_vis for a [P,6] pose tangent
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
visJvpCUDA(
	const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ means3D, const float* __restrict__ mu_c_a,
	const float* __restrict__ u_a, const float* __restrict__ Jp_a,
	const float* __restrict__ sigma_a, const float* __restrict__ conic_a,
	const float* __restrict__ opacity_a, const float* __restrict__ depths,
	const int* __restrict__ clamp_a,
	const float* __restrict__ viewmatrix, const float* __restrict__ projmatrix,
	float focal_x, float focal_y,
	const int* __restrict__ ray_mode, const float* __restrict__ z_obs, float delta_z,
	const float* __restrict__ pose_delta, float* __restrict__ out_davis)
{
	auto block = cg::this_thread_block();
	const uint32_t hb = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const bool inside = pix.x < (unsigned)W && pix.y < (unsigned)H;
	const int mode = inside ? ray_mode[pix_id] : 0;
	// rays with no claim still walk (block-wide sync) but accumulate nothing
	bool done = !inside;
	const float zo = inside ? z_obs[pix_id] : 0.f;

	const uint2 range = ranges[block.group_index().y * hb + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;
	__shared__ int sh_id[BLOCK_SIZE];
	float R[9]; load_Rcw(viewmatrix, R);

	float T = 1.f, Tf = 1.f, dTf = 0.f;
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == BLOCK_SIZE) break;
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) sh_id[block.thread_rank()] = point_list[range.x + progress];
		block.sync();
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			const int gid = sh_id[j];
			const float2 cu = {u_a[2 * gid], u_a[2 * gid + 1]};
			const float3 co = {conic_a[3 * gid], conic_a[3 * gid + 1], conic_a[3 * gid + 2]};
			float alpha; float2 d; bool cl;
			if (!evalAlpha(cu.x, cu.y, co.x, co.y, co.z, opacity_a[gid], pixf.x, pixf.y, alpha, d, cl))
				continue;
			const float test_T = T * (1.f - alpha);
			if (test_T < T_STOP) { done = true; continue; }
			const float g = gateOf(mode, depths[gid], zo, delta_z);
			if (g > 0.f)
			{
				float da = 0.f;
				if (!cl)
				{
					float3 mu_c, conic, p_orig; float Jp[6], sigma_c[6];
					loadGaussian(gid, mu_c_a, Jp_a, sigma_a, conic_a, means3D, mu_c, Jp, sigma_c, conic, p_orig);
					const float3 dmu = {pose_delta[6 * gid + 0], pose_delta[6 * gid + 1], pose_delta[6 * gid + 2]};
					const float3 dphi = {pose_delta[6 * gid + 3], pose_delta[6 * gid + 4], pose_delta[6 * gid + 5]};
					float2 du; float3 dQ; float dd;
					poseTangent(dmu, dphi, mu_c, Jp, sigma_c, conic.x, conic.y, conic.z,
					            R, projmatrix, p_orig, W, H, focal_x, focal_y, clamp_a[gid], du, dQ, dd);
					da = dAlphaOf(alpha, cl, d, conic.x, conic.y, conic.z, du, dQ);
				}
				dTf = dTf * (1.f - alpha) - Tf * da;
				Tf *= (1.f - alpha);
			}
			T = test_T;
		}
	}
	if (inside) out_davis[pix_id] = (mode > 0) ? -dTf : 0.f;
}

// ---------------------------------------------------------------------------
// VJP: H_A^T z_alpha -> [P,6]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
visVjpCUDA(
	const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ means3D, const float* __restrict__ mu_c_a,
	const float* __restrict__ u_a, const float* __restrict__ Jp_a,
	const float* __restrict__ sigma_a, const float* __restrict__ conic_a,
	const float* __restrict__ opacity_a, const float* __restrict__ depths,
	const int* __restrict__ clamp_a,
	const float* __restrict__ viewmatrix, const float* __restrict__ projmatrix,
	float focal_x, float focal_y,
	const int* __restrict__ ray_mode, const float* __restrict__ z_obs, float delta_z,
	const float* __restrict__ z_alpha, const float* __restrict__ final_Tf,
	const uint32_t* __restrict__ n_contrib, float* __restrict__ out_adjoint)
{
	auto block = cg::this_thread_block();
	const uint32_t hb = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const bool inside = pix.x < (unsigned)W && pix.y < (unsigned)H;
	const int mode = inside ? ray_mode[pix_id] : 0;
	const float za = (inside && mode > 0) ? z_alpha[pix_id] : 0.f;
	const bool active = inside && mode > 0 && za != 0.f;
	const float zo = inside ? z_obs[pix_id] : 0.f;

	const uint2 range = ranges[block.group_index().y * hb + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	__shared__ int sh_id[BLOCK_SIZE];
	float R[9]; load_Rcw(viewmatrix, R);

	float Tf = active ? final_Tf[pix_id] : 1.f;   // T^f after the current contributor
	float S = 1.f;                                 // prod_{j>i} (1 - beta_j)
	const uint32_t last_contributor = inside ? n_contrib[pix_id] : 0;

	for (int i = rounds - 1; i >= 0; i--)
	{
		block.sync();
		const int base = i * BLOCK_SIZE;
		int progress = base + block.thread_rank();
		if (range.x + progress < range.y) sh_id[block.thread_rank()] = point_list[range.x + progress];
		block.sync();
		if (!active) continue;
		const int count = min(BLOCK_SIZE, (int)(range.y - range.x) - base);
		for (int j = count - 1; j >= 0; j--)
		{
			if ((uint32_t)(base + j + 1) > last_contributor) continue;
			const int gid = sh_id[j];
			const float2 cu = {u_a[2 * gid], u_a[2 * gid + 1]};
			const float3 co = {conic_a[3 * gid], conic_a[3 * gid + 1], conic_a[3 * gid + 2]};
			float alpha; float2 d; bool cl;
			if (!evalAlpha(cu.x, cu.y, co.x, co.y, co.z, opacity_a[gid], pixf.x, pixf.y, alpha, d, cl))
				continue;
			const float g = gateOf(mode, depths[gid], zo, delta_z);
			if (g == 0.f) continue;                    // behind the measured surface: no claim
			Tf = Tf / (1.f - alpha);                   // T^f BEFORE this contributor (alpha <= 0.99)
			if (Tf > 1.f) Tf = 1.f;
			const float alpha_adj = za * Tf * S;       // dA_vis / dalpha_i
			if (!cl && alpha_adj != 0.f)
			{
				float3 mu_c, conic, p_orig; float Jp[6], sigma_c[6];
				loadGaussian(gid, mu_c_a, Jp_a, sigma_a, conic_a, means3D, mu_c, Jp, sigma_c, conic, p_orig);
				for (int k = 0; k < 6; ++k)
				{
					const float3 dmu = {k == 0 ? 1.f : 0.f, k == 1 ? 1.f : 0.f, k == 2 ? 1.f : 0.f};
					const float3 dphi = {k == 3 ? 1.f : 0.f, k == 4 ? 1.f : 0.f, k == 5 ? 1.f : 0.f};
					float2 du; float3 dQ; float dd;
					poseTangent(dmu, dphi, mu_c, Jp, sigma_c, conic.x, conic.y, conic.z,
					            R, projmatrix, p_orig, W, H, focal_x, focal_y, clamp_a[gid], du, dQ, dd);
					atomicAdd(&out_adjoint[6 * gid + k],
					          alpha_adj * dAlphaOf(alpha, cl, d, conic.x, conic.y, conic.z, du, dQ));
				}
			}
			S *= (1.f - alpha);
		}
	}
}

} // anonymous namespace

namespace CudaRasterizer { namespace IEKFVis {

void visibleAlpha(int P, int W, int H, const IEKF::CachePointers cache,
                  const uint2* ranges, const uint32_t* point_list,
                  const int* ray_mode, const float* z_obs, float delta_z,
                  float* out_avis, float* out_Tf, uint32_t* out_n_contrib,
                  int* out_front, int* out_behind, bool debug)
{
	dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	visAlphaCUDA<<<grid, block>>>(ranges, point_list, W, H, cache.u, cache.conic, cache.opacity,
		cache.depth, ray_mode, z_obs, delta_z, out_avis, out_Tf, out_n_contrib, out_front, out_behind);
	VIS_SYNC(debug);
}

void applyJVP(int P, int W, int H, const IEKF::CachePointers cache,
              const uint2* ranges, const uint32_t* point_list,
              const float* means3D, const float* viewmatrix, const float* projmatrix,
              float tan_fovx, float tan_fovy,
              const int* ray_mode, const float* z_obs, float delta_z,
              const float* pose_delta, float* out_davis, bool debug)
{
	const float focal_y = H / (2.0f * tan_fovy);
	const float focal_x = W / (2.0f * tan_fovx);
	dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	visJvpCUDA<<<grid, block>>>(ranges, point_list, W, H, means3D, cache.mu_c, cache.u, cache.Jp,
		cache.sigma_c, cache.conic, cache.opacity, cache.depth, cache.clamp_flags,
		viewmatrix, projmatrix, focal_x, focal_y, ray_mode, z_obs, delta_z, pose_delta, out_davis);
	VIS_SYNC(debug);
}

void applyVJP(int P, int W, int H, const IEKF::CachePointers cache,
              const uint2* ranges, const uint32_t* point_list,
              const float* means3D, const float* viewmatrix, const float* projmatrix,
              float tan_fovx, float tan_fovy,
              const int* ray_mode, const float* z_obs, float delta_z,
              const float* z_alpha, const float* final_Tf, const uint32_t* n_contrib,
              float* out_adjoint, bool debug)
{
	const float focal_y = H / (2.0f * tan_fovy);
	const float focal_x = W / (2.0f * tan_fovx);
	dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	visVjpCUDA<<<grid, block>>>(ranges, point_list, W, H, means3D, cache.mu_c, cache.u, cache.Jp,
		cache.sigma_c, cache.conic, cache.opacity, cache.depth, cache.clamp_flags,
		viewmatrix, projmatrix, focal_x, focal_y, ray_mode, z_obs, delta_z, z_alpha, final_Tf,
		n_contrib, out_adjoint);
	VIS_SYNC(debug);
}

}} // namespace CudaRasterizer::IEKFVis
