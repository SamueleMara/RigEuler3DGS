/*
 * Sparse alpha-aware error-state IEKF observation operators.
 * CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md sections 9-11, 19-22.
 *
 * See iekf.h for the conventions this file is pinned to. Everything here is
 * additive; the legacy forward/backward kernels are untouched.
 */

#include "iekf.h"

#include <iostream>
#include <algorithm>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <stdio.h>

#include "auxiliary.h"
#include "config.h"

namespace cg = cooperative_groups;

// A kernel launch contains commas inside <<<grid, block>>>, which the
// preprocessor would split into extra macro arguments, so the sync is a
// separate statement rather than a wrapper (the legacy CHECK_CUDA has the same
// constraint and solves it the same way).
#define IEKF_SYNC(debug)                                                         \
	if (debug) {                                                                 \
		auto ret = cudaDeviceSynchronize();                                      \
		if (ret != cudaSuccess) {                                                \
			printf("\n[IEKF CUDA ERROR] %s in %s:%d\n",                          \
			       cudaGetErrorString(ret), __FILE__, __LINE__);                 \
			throw std::runtime_error(cudaGetErrorString(ret));                   \
		}                                                                        \
	}

namespace
{

// The renderer's own guards, repeated here so the linearization is built on the
// exact same contributor set the residual came from.
__device__ const float ALPHA_MAX = 0.99f;
__device__ const float ALPHA_MIN = 1.0f / 255.0f;
__device__ const float T_STOP = 1e-4f;
__device__ const float LOWPASS = 0.3f;


// forward.cu's computeCov3D has internal linkage, so the identical computation
// is repeated here. It must stay identical: the estimator's Sigma has to be the
// same Sigma the production renderer splats, or the Jacobian describes a
// different scene than the residual does.
__device__ void iekfComputeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x; S[1][1] = mod * scale.y; S[2][2] = mod * scale.z;
	glm::vec4 q = rot;
	float r = q.x, x = q.y, y = q.z, z = q.w;
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;
	cov3D[0] = Sigma[0][0]; cov3D[1] = Sigma[0][1]; cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1]; cov3D[4] = Sigma[1][2]; cov3D[5] = Sigma[2][2];
}

// Same as rasterizer_impl.cu's helper (also internal linkage): number of bits
// needed for the tile index, so the radix sort only sorts the depth bits plus
// the tile bits.
static int iekfHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4, step = msb;
	while (step > 1) { step /= 2; if (n >> msb) msb += step; else msb -= step; }
	if (n >> msb) msb++;
	return msb;
}

struct Sym2 { float xx, xy, yy; };

__forceinline__ __device__ Sym2 inverse2(const Sym2 m, float& det_out)
{
	const float det = m.xx * m.yy - m.xy * m.xy;
	det_out = det;
	const float inv = 1.0f / (det + 1e-20f);
	Sym2 r;
	r.xx = m.yy * inv;
	r.xy = -m.xy * inv;
	r.yy = m.xx * inv;
	return r;
}

// Row-major 3x3 from the repo's column-major 4x4 view matrix. transformPoint4x3
// uses vm[0],vm[4],vm[8] as the first ROW acting on a point, so R_cw row r is
// (vm[r], vm[4+r], vm[8+r]).
__forceinline__ __device__ void load_Rcw(const float* vm, float R[9])
{
	R[0] = vm[0]; R[1] = vm[4]; R[2] = vm[8];
	R[3] = vm[1]; R[4] = vm[5]; R[5] = vm[9];
	R[6] = vm[2]; R[7] = vm[6]; R[8] = vm[10];
}

__forceinline__ __device__ float3 mat3_mul_vec(const float R[9], const float3 v)
{
	return {
		R[0] * v.x + R[1] * v.y + R[2] * v.z,
		R[3] * v.x + R[4] * v.y + R[5] * v.z,
		R[6] * v.x + R[7] * v.y + R[8] * v.z};
}

// Sigma stored as upper triangle (xx, xy, xz, yy, yz, zz).
__forceinline__ __device__ void sym3_congruence(const float R[9], const float* S, float* out)
{
	const float M[9] = {S[0], S[1], S[2], S[1], S[3], S[4], S[2], S[4], S[5]};
	float tmp[9];
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
		{
			float acc = 0.f;
			for (int k = 0; k < 3; ++k) acc += R[r * 3 + k] * M[k * 3 + c];
			tmp[r * 3 + c] = acc;
		}
	float res[9];
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
		{
			float acc = 0.f;
			for (int k = 0; k < 3; ++k) acc += tmp[r * 3 + k] * R[c * 3 + k];
			res[r * 3 + c] = acc;
		}
	out[0] = res[0]; out[1] = res[1]; out[2] = res[2];
	out[3] = res[4]; out[4] = res[5]; out[5] = res[8];
}

// Full (non-symmetric) congruence R * M * R^T for a general 3x3 M, used for the
// covariance DIFFERENTIAL, which is symmetric but built from an antisymmetric
// commutator so it is assembled densely first.
__forceinline__ __device__ void mat3_congruence_full(const float R[9], const float M[9], float* out_sym)
{
	float tmp[9];
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
		{
			float acc = 0.f;
			for (int k = 0; k < 3; ++k) acc += R[r * 3 + k] * M[k * 3 + c];
			tmp[r * 3 + c] = acc;
		}
	float res[9];
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
		{
			float acc = 0.f;
			for (int k = 0; k < 3; ++k) acc += tmp[r * 3 + k] * R[c * 3 + k];
			res[r * 3 + c] = acc;
		}
	out_sym[0] = res[0]; out_sym[1] = res[1]; out_sym[2] = res[2];
	out_sym[3] = res[4]; out_sym[4] = res[5]; out_sym[5] = res[8];
}

// ---------------------------------------------------------------------------
// Kernel A: preprocess / linearization cache (section 21)
// ---------------------------------------------------------------------------
__global__ void buildCacheCUDA(
	int P, int W, int H,
	const float* __restrict__ means3D,
	const glm::vec3* __restrict__ scales,
	float scale_modifier,
	const glm::vec4* __restrict__ rotations,
	const float* __restrict__ opacities,
	const float* __restrict__ cov3D_precomp,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ projmatrix,
	float tan_fovx, float tan_fovy,
	float focal_x, float focal_y,
	bool antialiasing,
	float* __restrict__ cov3D_scratch,
	float* __restrict__ mu_c_out,
	float* __restrict__ depth_out,
	float* __restrict__ u_out,
	float* __restrict__ Jp_out,
	float* __restrict__ sigma_c_out,
	float* __restrict__ conic_out,
	float* __restrict__ opacity_out,
	int* __restrict__ radii,
	uint32_t* __restrict__ tiles_touched,
	int* __restrict__ clamp_flags,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= (unsigned)P) return;

	radii[idx] = 0;
	tiles_touched[idx] = 0;
	clamp_flags[idx] = 0;

	float3 p_view;
	if (!in_frustum(idx, means3D, viewmatrix, projmatrix, false, p_view)) return;

	const float3 p_orig = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};
	const float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	const float p_w = 1.0f / (p_hom.w + 1e-7f);
	const float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

	// Frozen 3D covariance. Scales and rotations are frozen appearance for the
	// estimator (section 3), so this is computed once per linearization.
	const float* cov3D;
	if (cov3D_precomp != nullptr) cov3D = cov3D_precomp + idx * 6;
	else { iekfComputeCov3D(scales[idx], scale_modifier, rotations[idx], cov3D_scratch + idx * 6);
	       cov3D = cov3D_scratch + idx * 6; }

	float R[9]; load_Rcw(viewmatrix, R);
	float sigma_c[6]; sym3_congruence(R, cov3D, sigma_c);

	// Frustum clamp, exactly as computeCov2D does it. The clamp is not cosmetic:
	// where it binds, Jp stops depending on tx/ty and depends on tz instead, and
	// the analytic dJp below must follow that or the Jacobian is wrong at the
	// image border.
	const float limx = 1.3f * tan_fovx, limy = 1.3f * tan_fovy;
	const float txtz = p_view.x / p_view.z, tytz = p_view.y / p_view.z;
	int flags = 0;
	float tx = txtz, ty = tytz;
	if (txtz < -limx) { tx = -limx; flags |= 1; } else if (txtz > limx) { tx = limx; flags |= 1; }
	if (tytz < -limy) { ty = -limy; flags |= 2; } else if (tytz > limy) { ty = limy; flags |= 2; }
	tx *= p_view.z; ty *= p_view.z;

	const float iz = 1.0f / p_view.z, iz2 = iz * iz;
	float Jp[6];
	Jp[0] = focal_x * iz; Jp[1] = 0.f; Jp[2] = -focal_x * tx * iz2;
	Jp[3] = 0.f; Jp[4] = focal_y * iz; Jp[5] = -focal_y * ty * iz2;

	// cov2D = Jp * Sigma_c * Jp^T
	const float Sc[9] = {sigma_c[0], sigma_c[1], sigma_c[2],
	                     sigma_c[1], sigma_c[3], sigma_c[4],
	                     sigma_c[2], sigma_c[4], sigma_c[5]};
	float JS[6];
	for (int r = 0; r < 2; ++r)
		for (int c = 0; c < 3; ++c)
		{
			float acc = 0.f;
			for (int k = 0; k < 3; ++k) acc += Jp[r * 3 + k] * Sc[k * 3 + c];
			JS[r * 3 + c] = acc;
		}
	Sym2 cov;
	cov.xx = JS[0] * Jp[0] + JS[1] * Jp[1] + JS[2] * Jp[2];
	cov.xy = JS[0] * Jp[3] + JS[1] * Jp[4] + JS[2] * Jp[5];
	cov.yy = JS[3] * Jp[3] + JS[4] * Jp[4] + JS[5] * Jp[5];

	const float det_cov = cov.xx * cov.yy - cov.xy * cov.xy;
	Sym2 S = cov; S.xx += LOWPASS; S.yy += LOWPASS;
	const float det_S = S.xx * S.yy - S.xy * S.xy;
	if (det_S == 0.0f) return;
	float h_scale = 1.0f;
	if (antialiasing) h_scale = sqrtf(max(2.5e-5f, det_cov / det_S));

	float det_unused;
	const Sym2 Q = inverse2(S, det_unused);

	const float mid = 0.5f * (S.xx + S.yy);
	const float l1 = mid + sqrtf(max(0.1f, mid * mid - det_S));
	const float l2 = mid - sqrtf(max(0.1f, mid * mid - det_S));
	const int my_radius = (int)ceilf(3.f * sqrtf(max(l1, l2)));
	const float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) return;

	mu_c_out[3 * idx + 0] = p_view.x; mu_c_out[3 * idx + 1] = p_view.y; mu_c_out[3 * idx + 2] = p_view.z;
	depth_out[idx] = p_view.z;
	u_out[2 * idx + 0] = point_image.x; u_out[2 * idx + 1] = point_image.y;
	for (int k = 0; k < 6; ++k) Jp_out[6 * idx + k] = Jp[k];
	for (int k = 0; k < 6; ++k) sigma_c_out[6 * idx + k] = sigma_c[k];
	conic_out[3 * idx + 0] = Q.xx; conic_out[3 * idx + 1] = Q.xy; conic_out[3 * idx + 2] = Q.yy;
	opacity_out[idx] = opacities[idx] * h_scale;
	radii[idx] = my_radius;
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	clamp_flags[idx] = flags;
}

__global__ void duplicateKeysCUDA(
	int P, const float* __restrict__ u, const float* __restrict__ depths,
	const uint32_t* __restrict__ offsets, uint64_t* keys, uint32_t* values,
	const int* __restrict__ radii, dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= (unsigned)P) return;
	if (radii[idx] <= 0) return;
	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	uint2 rect_min, rect_max;
	const float2 p = {u[2 * idx], u[2 * idx + 1]};
	getRect(p, radii[idx], rect_min, rect_max, grid);
	for (uint32_t y = rect_min.y; y < rect_max.y; y++)
		for (uint32_t x = rect_min.x; x < rect_max.x; x++)
		{
			uint64_t key = y * grid.x + x;
			key <<= 32;
			key |= *((uint32_t*)&depths[idx]);
			keys[off] = key;
			values[off] = idx;
			off++;
		}
}

__global__ void identifyRangesCUDA(int L, const uint64_t* __restrict__ keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= (unsigned)L) return;
	uint32_t tile = keys[idx] >> 32;
	if (idx == 0) ranges[tile].x = 0;
	else
	{
		uint32_t prev = keys[idx - 1] >> 32;
		if (tile != prev) { ranges[prev].y = idx; ranges[tile].x = idx; }
	}
	if (idx == L - 1) ranges[tile].y = L;
}

// ---------------------------------------------------------------------------
// Shared per-contributor evaluation, so forward / JVP / VJP / Fisher cannot
// drift apart. Returns false when the renderer would have skipped this
// contributor -- the contributor set is part of the frozen linearization.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Kernel B: nonlinear RGB-D render from the frozen cache (section 21)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
renderFromCacheCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ u,
	const float* __restrict__ conic,
	const float* __restrict__ opacity,
	const float* __restrict__ depths,
	const float* __restrict__ colors,
	const float* __restrict__ bg,
	float eps_depth,
	float* __restrict__ out_color,
	float* __restrict__ out_depth,
	float* __restrict__ out_alpha,
	float* __restrict__ out_T,
	uint32_t* __restrict__ out_n_contrib)
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
	__shared__ float collected_depth[BLOCK_SIZE];

	float T = 1.0f, A = 0.0f, Z = 0.0f;
	float C[3] = {0.f, 0.f, 0.f};
	// Index (within this tile's range) one past the last contributor the
	// front-to-back loop actually accumulated. The VJP and Fisher passes walk
	// the same list BACKWARD and must stop at exactly the same place, or they
	// would transpose an operator the forward pass never applied.
	uint32_t contributor = 0, last_contributor = 0;

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
			collected_depth[block.thread_rank()] = depths[gid];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			++contributor;
			const float2 cu = collected_u[j];
			const float4 co = collected_co[j];
			float alpha; float2 d; bool cl;
			if (!evalAlpha(cu.x, cu.y, co.x, co.y, co.z, co.w, pixf.x, pixf.y, alpha, d, cl))
				continue;
			const float test_T = T * (1.f - alpha);
			if (test_T < T_STOP) { done = true; continue; }

			const int gid = collected_id[j];
			last_contributor = contributor;
			const float w = alpha * T;
			for (int ch = 0; ch < 3; ch++) C[ch] += colors[gid * 3 + ch] * w;
			A += w;
			Z += w * collected_depth[j];
			T = test_T;
		}
	}

	if (inside)
	{
		for (int ch = 0; ch < 3; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg[ch];
		out_alpha[pix_id] = A;
		out_T[pix_id] = T;
		out_depth[pix_id] = Z / (A + eps_depth);
		out_n_contrib[pix_id] = last_contributor;
	}
}

// ---------------------------------------------------------------------------
// Per-Gaussian pose tangent (section 9.1-9.4). Shared by JVP, VJP and Fisher so
// there is exactly one implementation of the analytic differentials.
//
// Given the 6D pose direction (dmu, dphi) this returns du, dQ and dd.
// ---------------------------------------------------------------------------
__forceinline__ __device__ void poseTangent(
	const float3 dmu, const float3 dphi,
	const float3 mu_c, const float* Jp, const float* sigma_c,
	const float qxx, const float qxy, const float qyy,
	const float R[9], const float* projmatrix, const float3 p_orig,
	int W, int H, float focal_x, float focal_y, int clamp_flags,
	float2& du, float3& dQ, float& dd)
{
	// --- du: differentiate the EXACT projective chain the renderer uses, so
	// the tangent matches the pixel positions that produced the residual.
	const float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	const float pw = 1.0f / (p_hom.w + 1e-7f);
	// d(p_hom)/d(mu) rows: projmatrix is column-major, transformPoint4x4 uses
	// x*m[0]+y*m[4]+z*m[8]+m[12] for the first component.
	const float3 dh_x = {projmatrix[0], projmatrix[4], projmatrix[8]};
	const float3 dh_y = {projmatrix[1], projmatrix[5], projmatrix[9]};
	const float3 dh_w = {projmatrix[3], projmatrix[7], projmatrix[11]};
	const float dhx = dh_x.x * dmu.x + dh_x.y * dmu.y + dh_x.z * dmu.z;
	const float dhy = dh_y.x * dmu.x + dh_y.y * dmu.y + dh_y.z * dmu.z;
	const float dhw = dh_w.x * dmu.x + dh_w.y * dmu.y + dh_w.z * dmu.z;
	// d(p_hom.x / p_hom.w) = (dhx - p_proj.x * dhw) / p_hom.w
	const float dpx = (dhx - (p_hom.x * pw) * dhw) * pw;
	const float dpy = (dhy - (p_hom.y * pw) * dhw) * pw;
	du.x = dpx * 0.5f * W;   // ndc2Pix(v,S) = ((v+1)S - 1)*0.5
	du.y = dpy * 0.5f * H;

	// --- camera-space center tangent and center-depth tangent
	const float3 dmu_c = mat3_mul_vec(R, dmu);
	dd = dmu_c.z;                       // d(d_i)/d(dmu) = e_z^T R_cw ; no dphi term

	// --- dJp (section 9.2), with the frustum clamp from computeCov2D.
	// Where the clamp binds, tx is pinned to +-limx*tz so it tracks tz instead
	// of tx; omitting this makes the border Jacobian wrong.
	const float iz = 1.0f / mu_c.z, iz2 = iz * iz, iz3 = iz2 * iz;
	const float tx = -Jp[2] / (focal_x * iz2);   // recover clamped tx from Jp
	const float ty = -Jp[5] / (focal_y * iz2);
	const float dtx = (clamp_flags & 1) ? (tx / mu_c.z) * dmu_c.z : dmu_c.x;
	const float dty = (clamp_flags & 2) ? (ty / mu_c.z) * dmu_c.z : dmu_c.y;
	float dJp[6];
	dJp[0] = -focal_x * dmu_c.z * iz2;
	dJp[1] = 0.f;
	dJp[2] = -focal_x * dtx * iz2 + 2.f * focal_x * tx * dmu_c.z * iz3;
	dJp[3] = 0.f;
	dJp[4] = -focal_y * dmu_c.z * iz2;
	dJp[5] = -focal_y * dty * iz2 + 2.f * focal_y * ty * dmu_c.z * iz3;

	// --- dSigma_c (section 9.3): dSigma_world = hat(dphi) Sigma - Sigma hat(dphi),
	// then rotated into the camera. Sigma_c is cached, and the commutator is
	// equivariant under the congruence, so it can be formed directly on Sigma_c:
	//     R (hat(p) S - S hat(p)) R^T = hat(R p) (R S R^T) - (R S R^T) hat(R p)
	// which saves carrying the world covariance in the cache.
	const float3 w = mat3_mul_vec(R, dphi);
	const float Sc[9] = {sigma_c[0], sigma_c[1], sigma_c[2],
	                     sigma_c[1], sigma_c[3], sigma_c[4],
	                     sigma_c[2], sigma_c[4], sigma_c[5]};
	const float hat[9] = {0.f, -w.z, w.y, w.z, 0.f, -w.x, -w.y, w.x, 0.f};
	float dSc[9];
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
		{
			float a = 0.f, b = 0.f;
			for (int k = 0; k < 3; ++k) { a += hat[r * 3 + k] * Sc[k * 3 + c];
			                              b += Sc[r * 3 + k] * hat[k * 3 + c]; }
			dSc[r * 3 + c] = a - b;
		}

	// --- dS = dJp Sc Jp^T + Jp dSc Jp^T + Jp Sc dJp^T   (section 9.4)
	float A1[6], A2[6], A3[6];
	for (int r = 0; r < 2; ++r)
		for (int c = 0; c < 3; ++c)
		{
			float a = 0.f, b = 0.f, e = 0.f;
			for (int k = 0; k < 3; ++k)
			{
				a += dJp[r * 3 + k] * Sc[k * 3 + c];
				b += Jp[r * 3 + k] * dSc[k * 3 + c];
				e += Jp[r * 3 + k] * Sc[k * 3 + c];
			}
			A1[r * 3 + c] = a; A2[r * 3 + c] = b; A3[r * 3 + c] = e;
		}
	float dSxx = 0.f, dSxy = 0.f, dSyy = 0.f;
	for (int k = 0; k < 3; ++k)
	{
		dSxx += A1[k] * Jp[k] + A2[k] * Jp[k] + A3[k] * dJp[k];
		dSxy += A1[k] * Jp[3 + k] + A2[k] * Jp[3 + k] + A3[k] * dJp[3 + k];
		dSyy += A1[3 + k] * Jp[3 + k] + A2[3 + k] * Jp[3 + k] + A3[3 + k] * dJp[3 + k];
	}

	// --- dQ = -Q dS Q  (section 9.4), symmetric 2x2, three coefficients.
	const float m00 = qxx * dSxx + qxy * dSxy;
	const float m01 = qxx * dSxy + qxy * dSyy;
	const float m10 = qxy * dSxx + qyy * dSxy;
	const float m11 = qxy * dSxy + qyy * dSyy;
	dQ.x = -(m00 * qxx + m01 * qxy);
	dQ.y = -(m00 * qxy + m01 * qyy);
	dQ.z = -(m10 * qxy + m11 * qyy);
}

// delta alpha, section 9.5. e = x_p - u = -d.
__forceinline__ __device__ float dAlphaOf(
	float alpha, bool clamped, float2 d, float qxx, float qxy, float qyy,
	float2 du, float3 dQ)
{
	if (clamped) return 0.0f;   // the 0.99 clamp is active: derivative is exactly 0
	const float ex = -d.x, ey = -d.y;
	const float qe_x = qxx * ex + qxy * ey;
	const float qe_y = qxy * ex + qyy * ey;
	const float term1 = qe_x * du.x + qe_y * du.y;
	const float term2 = 0.5f * (dQ.x * ex * ex + 2.f * dQ.y * ex * ey + dQ.z * ey * ey);
	return alpha * (term1 - term2);
}

// Loads everything the tangent needs for one Gaussian.
__forceinline__ __device__ void loadGaussian(
	int gid, const float* mu_c_a, const float* Jp_a, const float* sigma_a,
	const float* conic_a, const float* means3D,
	float3& mu_c, float* Jp, float* sigma_c, float3& conic, float3& p_orig)
{
	mu_c = {mu_c_a[3 * gid], mu_c_a[3 * gid + 1], mu_c_a[3 * gid + 2]};
	for (int k = 0; k < 6; ++k) Jp[k] = Jp_a[6 * gid + k];
	for (int k = 0; k < 6; ++k) sigma_c[k] = sigma_a[6 * gid + k];
	conic = {conic_a[3 * gid], conic_a[3 * gid + 1], conic_a[3 * gid + 2]};
	p_orig = {means3D[3 * gid], means3D[3 * gid + 1], means3D[3 * gid + 2]};
}

// ---------------------------------------------------------------------------
// Kernel C: JVP tile pass, H v  (section 10 / section 21 Kernel C)
//
// Walks the SAME contributor range in the SAME order as the forward pass and
// applies the tangent recurrence of section 10. No atomics are needed: each
// pixel owns its own output.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
jvpCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ means3D,
	const float* __restrict__ mu_c_a,
	const float* __restrict__ u_a,
	const float* __restrict__ Jp_a,
	const float* __restrict__ sigma_a,
	const float* __restrict__ conic_a,
	const float* __restrict__ opacity_a,
	const float* __restrict__ depths,
	const int* __restrict__ clamp_a,
	const float* __restrict__ colors,
	const float* __restrict__ bg,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ projmatrix,
	float focal_x, float focal_y,
	float eps_depth,
	const float* __restrict__ pose_delta,
	float* __restrict__ out_dcolor,
	float* __restrict__ out_ddepth)
{
	auto block = cg::this_thread_block();
	const uint32_t hb = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const bool inside = pix.x < (unsigned)W && pix.y < (unsigned)H;
	bool done = !inside;

	const uint2 range = ranges[block.group_index().y * hb + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int sh_id[BLOCK_SIZE];
	float R[9]; load_Rcw(viewmatrix, R);

	// primal and tangent accumulators, section 10
	float T = 1.f, dT = 0.f;
	float A = 0.f, dA = 0.f;
	float Z = 0.f, dZ = 0.f;
	float C[3] = {0.f, 0.f, 0.f}, dC[3] = {0.f, 0.f, 0.f};

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == BLOCK_SIZE) break;
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
			sh_id[block.thread_rank()] = point_list[range.x + progress];
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

			float3 mu_c, conic, p_orig; float Jp[6], sigma_c[6];
			loadGaussian(gid, mu_c_a, Jp_a, sigma_a, conic_a, means3D,
			             mu_c, Jp, sigma_c, conic, p_orig);
			const float3 dmu = {pose_delta[6 * gid + 0], pose_delta[6 * gid + 1], pose_delta[6 * gid + 2]};
			const float3 dphi = {pose_delta[6 * gid + 3], pose_delta[6 * gid + 4], pose_delta[6 * gid + 5]};
			float2 du; float3 dQ; float dd;
			poseTangent(dmu, dphi, mu_c, Jp, sigma_c, conic.x, conic.y, conic.z,
			            R, projmatrix, p_orig, W, H, focal_x, focal_y, clamp_a[gid],
			            du, dQ, dd);
			const float da = dAlphaOf(alpha, cl, d, conic.x, conic.y, conic.z, du, dQ);

			// section 10, in this exact order: the weight tangent uses T BEFORE
			// the update, which is why dT is advanced last.
			const float dw = dT * alpha + T * da;
			for (int ch = 0; ch < 3; ch++) dC[ch] += dw * colors[gid * 3 + ch];
			for (int ch = 0; ch < 3; ch++) C[ch] += (T * alpha) * colors[gid * 3 + ch];
			dA += dw;  A += T * alpha;
			dZ += dw * depths[gid] + (T * alpha) * dd;
			Z += (T * alpha) * depths[gid];
			dT = dT * (1.f - alpha) - T * da;
			T = test_T;
		}
	}

	if (inside)
	{
		// background contributes through dT (section 10's closing note)
		for (int ch = 0; ch < 3; ch++)
			out_dcolor[ch * H * W + pix_id] = dC[ch] + dT * bg[ch];
		const float denom = A + eps_depth;
		const float D = Z / denom;
		out_ddepth[pix_id] = (dZ - D * dA) / denom;
	}
}

// ---------------------------------------------------------------------------
// Kernel D: VJP tile pass, H^T z  (section 11 / section 21 Kernel D)
//
// Reverse traversal maintaining the suffix accumulators of sections 11.1-11.2:
//   B  = behind-colour,  Ab = behind-alpha,  Zb = behind-depth.
// The prefix transmittance is walked backwards as T_i = T_{i+1}/(1-alpha_i).
// Section 27 warns this is unstable "near alpha = 1" -- it cannot be here,
// because the production renderer clamps alpha to 0.99, so (1-alpha) >= 0.01
// by construction. The clamp is asserted by ALPHA_MAX above and the exact
// recomputation path is validated against this one in the test suite.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
vjpCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ means3D,
	const float* __restrict__ mu_c_a,
	const float* __restrict__ u_a,
	const float* __restrict__ Jp_a,
	const float* __restrict__ sigma_a,
	const float* __restrict__ conic_a,
	const float* __restrict__ opacity_a,
	const float* __restrict__ depths,
	const int* __restrict__ clamp_a,
	const float* __restrict__ colors,
	const float* __restrict__ bg,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ projmatrix,
	float focal_x, float focal_y,
	float eps_depth,
	const float* __restrict__ z_color,
	const float* __restrict__ z_depth,
	const float* __restrict__ final_T,
	const float* __restrict__ final_A,
	const float* __restrict__ final_D,
	const uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_adjoint)
{
	auto block = cg::this_thread_block();
	const uint32_t hb = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const bool inside = pix.x < (unsigned)W && pix.y < (unsigned)H;

	const uint2 range = ranges[block.group_index().y * hb + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int sh_id[BLOCK_SIZE];
	float R[9]; load_Rcw(viewmatrix, R);

	float zc[3] = {0.f, 0.f, 0.f}, zd = 0.f, T = 0.f, A = 0.f, D = 0.f;
	if (inside)
	{
		for (int ch = 0; ch < 3; ch++) zc[ch] = z_color[ch * H * W + pix_id];
		zd = z_depth[pix_id];
		T = final_T[pix_id];
		A = final_A[pix_id];
		D = final_D[pix_id];
	}
	const float denom = A + eps_depth;

	// Suffix accumulators start at the background / empty tail.
	float B[3] = {bg[0], bg[1], bg[2]};
	float Ab = 0.f, Zb = 0.f;
	const uint32_t last_contributor = inside ? n_contrib[pix_id] : 0;
	// Index of the entry about to be visited, counted from the FRONT of the
	// tile range exactly as the forward pass counts it. The forward loop
	// increments once per list entry (before the alpha test), so the backward
	// walk must mirror that convention entry for entry.

	for (int i = rounds - 1; i >= 0; i--)
	{
		block.sync();
		const int base = i * BLOCK_SIZE;
		int progress = base + block.thread_rank();
		if (range.x + progress < range.y)
			sh_id[block.thread_rank()] = point_list[range.x + progress];
		block.sync();
		if (!inside) continue;

		const int count = min(BLOCK_SIZE, (int)(range.y - range.x) - base);
		for (int j = count - 1; j >= 0; j--)
		{
			// Everything deeper than the forward pass's stopping point was
			// never composited, so it carries no adjoint. Without this the
			// reverse walk transposes contributors H never applied, and the
			// adjoint identity fails by O(1) on any scene dense enough to
			// saturate transmittance.
			if ((uint32_t)(base + j + 1) > last_contributor) continue;
			const int gid = sh_id[j];
			const float2 cu = {u_a[2 * gid], u_a[2 * gid + 1]};
			const float3 co = {conic_a[3 * gid], conic_a[3 * gid + 1], conic_a[3 * gid + 2]};
			float alpha; float2 d; bool cl;
			if (!evalAlpha(cu.x, cu.y, co.x, co.y, co.z, opacity_a[gid], pixf.x, pixf.y, alpha, d, cl))
				continue;
			// Reconstruct the transmittance BEFORE this contributor. Safe because
			// alpha <= 0.99 so the divisor is >= 0.01.
			T = T / (1.f - alpha);
			if (T > 1.0f) T = 1.0f;

			const float w = T * alpha;
			const float di = depths[gid];

			// section 11.1 / 11.2 sensitivities at the frozen linearization
			float dCda = 0.f;
			for (int ch = 0; ch < 3; ch++) dCda += zc[ch] * T * (colors[gid * 3 + ch] - B[ch]);
			const float dDda = zd * T * ((di - Zb) - D * (1.f - Ab)) / denom;
			const float alpha_adj = dCda + dDda;
			const float depth_adj = zd * w / denom;         // section 11.2 dD/dd

			if (!cl && (alpha_adj != 0.f || depth_adj != 0.f))
			{
				// Chain alpha_adj through dalpha/d(du,dQ) and depth_adj through
				// dd/dmu, analytically. Section 9.5 gives
				//   dalpha = alpha (e^T Q du - 0.5 e^T dQ e)
				// so the adjoint on the 6D pose is obtained by evaluating the
				// tangent on each basis direction. Six evaluations of the shared
				// poseTangent keeps one implementation of the differentials and
				// guarantees JVP/VJP agree (the adjoint identity test).
				float3 mu_c, conic, p_orig; float Jp[6], sigma_c[6];
				loadGaussian(gid, mu_c_a, Jp_a, sigma_a, conic_a, means3D,
				             mu_c, Jp, sigma_c, conic, p_orig);
				float grad[6];
				for (int k = 0; k < 6; ++k)
				{
					const float3 dmu = {k == 0 ? 1.f : 0.f, k == 1 ? 1.f : 0.f, k == 2 ? 1.f : 0.f};
					const float3 dphi = {k == 3 ? 1.f : 0.f, k == 4 ? 1.f : 0.f, k == 5 ? 1.f : 0.f};
					float2 du; float3 dQ; float dd;
					poseTangent(dmu, dphi, mu_c, Jp, sigma_c, conic.x, conic.y, conic.z,
					            R, projmatrix, p_orig, W, H, focal_x, focal_y, clamp_a[gid],
					            du, dQ, dd);
					const float da = dAlphaOf(alpha, cl, d, conic.x, conic.y, conic.z, du, dQ);
					grad[k] = alpha_adj * da + depth_adj * dd;
				}
				for (int k = 0; k < 6; ++k)
					atomicAdd(&out_adjoint[6 * gid + k], grad[k]);
			}
			else if (depth_adj != 0.f)
			{
				// alpha is clamped so dalpha = 0, but the direct center-depth
				// path still carries information.
				const float3 e_z = {R[6], R[7], R[8]};
				atomicAdd(&out_adjoint[6 * gid + 0], depth_adj * e_z.x);
				atomicAdd(&out_adjoint[6 * gid + 1], depth_adj * e_z.y);
				atomicAdd(&out_adjoint[6 * gid + 2], depth_adj * e_z.z);
			}

			// advance the suffixes to "behind contributor i-1"
			for (int ch = 0; ch < 3; ch++)
				B[ch] = alpha * colors[gid * 3 + ch] + (1.f - alpha) * B[ch];
			Zb = alpha * di + (1.f - alpha) * Zb;
			Ab = alpha + (1.f - alpha) * Ab;
		}
	}
}

// ---------------------------------------------------------------------------
// Kernel E: per-Gaussian 6x6 Fisher blocks (section 16 / 21 Kernel E)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
fisherCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ means3D,
	const float* __restrict__ mu_c_a,
	const float* __restrict__ u_a,
	const float* __restrict__ Jp_a,
	const float* __restrict__ sigma_a,
	const float* __restrict__ conic_a,
	const float* __restrict__ opacity_a,
	const float* __restrict__ depths,
	const int* __restrict__ clamp_a,
	const float* __restrict__ colors,
	const float* __restrict__ bg,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ projmatrix,
	float focal_x, float focal_y,
	float eps_depth,
	const float* __restrict__ prec_color,
	const float* __restrict__ prec_depth,
	const float* __restrict__ final_T,
	const float* __restrict__ final_A,
	const float* __restrict__ final_D,
	const uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_fisher)
{
	auto block = cg::this_thread_block();
	const uint32_t hb = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const bool inside = pix.x < (unsigned)W && pix.y < (unsigned)H;

	const uint2 range = ranges[block.group_index().y * hb + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	__shared__ int sh_id[BLOCK_SIZE];
	float R[9]; load_Rcw(viewmatrix, R);

	float pc[3] = {0.f, 0.f, 0.f}, pd = 0.f, T = 0.f, A = 0.f, D = 0.f;
	if (inside)
	{
		for (int ch = 0; ch < 3; ch++) pc[ch] = prec_color[ch * H * W + pix_id];
		pd = prec_depth[pix_id];
		T = final_T[pix_id]; A = final_A[pix_id]; D = final_D[pix_id];
	}
	const float denom = A + eps_depth;

	float B[3] = {bg[0], bg[1], bg[2]};
	float Ab = 0.f, Zb = 0.f;
	const uint32_t last_contributor = inside ? n_contrib[pix_id] : 0;
	// Index of the entry about to be visited, counted from the FRONT of the
	// tile range exactly as the forward pass counts it. The forward loop
	// increments once per list entry (before the alpha test), so the backward
	// walk must mirror that convention entry for entry.

	for (int i = rounds - 1; i >= 0; i--)
	{
		block.sync();
		const int base = i * BLOCK_SIZE;
		int progress = base + block.thread_rank();
		if (range.x + progress < range.y)
			sh_id[block.thread_rank()] = point_list[range.x + progress];
		block.sync();
		if (!inside) continue;

		const int count = min(BLOCK_SIZE, (int)(range.y - range.x) - base);
		for (int j = count - 1; j >= 0; j--)
		{
			// Everything deeper than the forward pass's stopping point was
			// never composited, so it carries no adjoint. Without this the
			// reverse walk transposes contributors H never applied, and the
			// adjoint identity fails by O(1) on any scene dense enough to
			// saturate transmittance.
			if ((uint32_t)(base + j + 1) > last_contributor) continue;
			const int gid = sh_id[j];
			const float2 cu = {u_a[2 * gid], u_a[2 * gid + 1]};
			const float3 co = {conic_a[3 * gid], conic_a[3 * gid + 1], conic_a[3 * gid + 2]};
			float alpha; float2 d; bool cl;
			if (!evalAlpha(cu.x, cu.y, co.x, co.y, co.z, opacity_a[gid], pixf.x, pixf.y, alpha, d, cl))
				continue;
			T = T / (1.f - alpha);
			if (T > 1.0f) T = 1.0f;
			const float w = T * alpha;
			const float di = depths[gid];

			// Build the 4x6 block H_{p,i} of section 11.3, then accumulate
			// H^T R^-1 H into the symmetric 6x6.
			float3 mu_c, conic, p_orig; float Jp[6], sigma_c[6];
			loadGaussian(gid, mu_c_a, Jp_a, sigma_a, conic_a, means3D,
			             mu_c, Jp, sigma_c, conic, p_orig);
			float Jalpha[6], Jd[6];
			for (int k = 0; k < 6; ++k)
			{
				const float3 dmu = {k == 0 ? 1.f : 0.f, k == 1 ? 1.f : 0.f, k == 2 ? 1.f : 0.f};
				const float3 dphi = {k == 3 ? 1.f : 0.f, k == 4 ? 1.f : 0.f, k == 5 ? 1.f : 0.f};
				float2 du; float3 dQ; float dd;
				poseTangent(dmu, dphi, mu_c, Jp, sigma_c, conic.x, conic.y, conic.z,
				            R, projmatrix, p_orig, W, H, focal_x, focal_y, clamp_a[gid],
				            du, dQ, dd);
				Jalpha[k] = dAlphaOf(alpha, cl, d, conic.x, conic.y, conic.z, du, dQ);
				Jd[k] = dd;
			}
			const float dDda = T * ((di - Zb) - D * (1.f - Ab)) / denom;
			const float dDdd = w / denom;

			float acc[21]; for (int k = 0; k < 21; ++k) acc[k] = 0.f;
			// RGB rows: H^RGB_row = T (c - B) * Jalpha
			for (int ch = 0; ch < 3; ch++)
			{
				const float s = T * (colors[gid * 3 + ch] - B[ch]);
				const float pw = pc[ch];
				if (pw == 0.f) continue;
				int t = 0;
				for (int a = 0; a < 6; ++a)
					for (int b2 = a; b2 < 6; ++b2, ++t)
						acc[t] += pw * (s * Jalpha[a]) * (s * Jalpha[b2]);
			}
			// Depth row: H^D = dD/dalpha * Jalpha + dD/dd * Jd
			if (pd != 0.f)
			{
				float hd[6];
				for (int k = 0; k < 6; ++k) hd[k] = dDda * Jalpha[k] + dDdd * Jd[k];
				int t = 0;
				for (int a = 0; a < 6; ++a)
					for (int b2 = a; b2 < 6; ++b2, ++t)
						acc[t] += pd * hd[a] * hd[b2];
			}
			for (int k = 0; k < 21; ++k)
				if (acc[k] != 0.f) atomicAdd(&out_fisher[21 * gid + k], acc[k]);

			for (int ch = 0; ch < 3; ch++)
				B[ch] = alpha * colors[gid * 3 + ch] + (1.f - alpha) * B[ch];
			Zb = alpha * di + (1.f - alpha) * Zb;
			Ab = alpha + (1.f - alpha) * Ab;
		}
	}
}

} // anonymous namespace

// ===========================================================================
// Host entry points
// ===========================================================================
namespace CudaRasterizer { namespace IEKF {

int buildCache(
	const int P, const int width, const int height,
	const float* means3D, const float* scales, const float scale_modifier,
	const float* rotations, const float* opacities, const float* cov3D_precomp,
	const float* viewmatrix, const float* projmatrix,
	const float tan_fovx, const float tan_fovy, const bool antialiasing,
	CachePointers cache, float* cov3D_scratch,
	uint32_t* point_offsets, char*& list_sorting_space, size_t& sorting_size,
	uint64_t* keys_unsorted, uint32_t* values_unsorted, uint64_t* keys_sorted,
	uint32_t* point_list, uint2* ranges, int* num_rendered_out,
	bool probe_only, bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	buildCacheCUDA<<<(P + 255) / 256, 256>>>(
		P, width, height, means3D, (const glm::vec3*)scales, scale_modifier,
		(const glm::vec4*)rotations, opacities, cov3D_precomp,
		viewmatrix, projmatrix, tan_fovx, tan_fovy, focal_x, focal_y, antialiasing,
		cov3D_scratch, cache.mu_c, cache.depth, cache.u, cache.Jp, cache.sigma_c,
		cache.conic, cache.opacity, cache.radii, cache.tiles_touched,
		cache.clamp_flags, grid);
	IEKF_SYNC(debug);

	// Inclusive scan of tiles_touched -> per-Gaussian write offsets.
	size_t scan_bytes = 0;
	cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, cache.tiles_touched, point_offsets, P);
	char* scan_space = nullptr;
	cudaMalloc(&scan_space, scan_bytes);
	cub::DeviceScan::InclusiveSum(scan_space, scan_bytes,
		cache.tiles_touched, point_offsets, P);
	IEKF_SYNC(debug);
	int num_rendered = 0;
	cudaMemcpy(&num_rendered, point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(scan_space);
	*num_rendered_out = num_rendered;
	if (probe_only || num_rendered == 0) return num_rendered;

	duplicateKeysCUDA<<<(P + 255) / 256, 256>>>(
		P, cache.u, cache.depth, point_offsets, keys_unsorted, values_unsorted,
		cache.radii, grid);
	IEKF_SYNC(debug);

	int bit = iekfHigherMsb(grid.x * grid.y);
	size_t sort_bytes = 0;
	cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys_unsorted, keys_sorted,
		values_unsorted, point_list, num_rendered, 0, 32 + bit);
	char* sort_space = nullptr;
	cudaMalloc(&sort_space, sort_bytes);
	cub::DeviceRadixSort::SortPairs(sort_space, sort_bytes,
		keys_unsorted, keys_sorted, values_unsorted, point_list,
		num_rendered, 0, 32 + bit);
	IEKF_SYNC(debug);
	cudaFree(sort_space);

	cudaMemset(ranges, 0, grid.x * grid.y * sizeof(uint2));
	IEKF_SYNC(debug);
	identifyRangesCUDA<<<(num_rendered + 255) / 256, 256>>>(
		num_rendered, keys_sorted, ranges);
	IEKF_SYNC(debug);
	return num_rendered;
}

void renderRGBD(
	const int P, const int width, const int height, const CachePointers cache,
	const uint2* ranges, const uint32_t* point_list, const float* colors,
	const float* background, const float eps_depth,
	float* out_color, float* out_depth, float* out_alpha, float* out_T,
	uint32_t* out_n_contrib, bool debug)
{
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	renderFromCacheCUDA<<<grid, block>>>(
		ranges, point_list, width, height, cache.u, cache.conic, cache.opacity,
		cache.depth, colors, background, eps_depth,
		out_color, out_depth, out_alpha, out_T, out_n_contrib);
	IEKF_SYNC(debug);
}

void applyJVP(
	const int P, const int width, const int height, const CachePointers cache,
	const uint2* ranges, const uint32_t* point_list, const float* colors,
	const float* background, const float* means3D,
	const float* viewmatrix, const float* projmatrix,
	const float tan_fovx, const float tan_fovy, const float eps_depth,
	const float* pose_delta, float* out_dcolor, float* out_ddepth, bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	jvpCUDA<<<grid, block>>>(
		ranges, point_list, width, height, means3D,
		cache.mu_c, cache.u, cache.Jp, cache.sigma_c, cache.conic, cache.opacity,
		cache.depth, cache.clamp_flags, colors, background, viewmatrix, projmatrix,
		focal_x, focal_y, eps_depth, pose_delta, out_dcolor, out_ddepth);
	IEKF_SYNC(debug);
}

void applyVJP(
	const int P, const int width, const int height, const CachePointers cache,
	const uint2* ranges, const uint32_t* point_list, const float* colors,
	const float* background, const float* means3D,
	const float* viewmatrix, const float* projmatrix,
	const float tan_fovx, const float tan_fovy, const float eps_depth,
	const float* z_color, const float* z_depth,
	const float* final_T, const float* final_A, const float* final_D,
	const uint32_t* n_contrib,
	float* out_pose_adjoint, bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	vjpCUDA<<<grid, block>>>(
		ranges, point_list, width, height, means3D,
		cache.mu_c, cache.u, cache.Jp, cache.sigma_c, cache.conic, cache.opacity,
		cache.depth, cache.clamp_flags, colors, background, viewmatrix, projmatrix,
		focal_x, focal_y, eps_depth, z_color, z_depth,
		final_T, final_A, final_D, n_contrib, out_pose_adjoint);
	IEKF_SYNC(debug);
}

void buildFisher(
	const int P, const int width, const int height, const CachePointers cache,
	const uint2* ranges, const uint32_t* point_list, const float* colors,
	const float* background, const float* means3D,
	const float* viewmatrix, const float* projmatrix,
	const float tan_fovx, const float tan_fovy, const float eps_depth,
	const float* precision_color, const float* precision_depth,
	const float* final_T, const float* final_A, const float* final_D,
	const uint32_t* n_contrib,
	float* out_fisher, bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	fisherCUDA<<<grid, block>>>(
		ranges, point_list, width, height, means3D,
		cache.mu_c, cache.u, cache.Jp, cache.sigma_c, cache.conic, cache.opacity,
		cache.depth, cache.clamp_flags, colors, background, viewmatrix, projmatrix,
		focal_x, focal_y, eps_depth, precision_color, precision_depth,
		final_T, final_A, final_D, n_contrib, out_fisher);
	IEKF_SYNC(debug);
}

}} // namespace CudaRasterizer::IEKF
