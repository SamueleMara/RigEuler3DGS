/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <stdint.h>
#include <cooperative_groups.h>
#include <stdexcept>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];


	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data. Arbitrary channel counts are handled by host-side
// chunking; this kernel renders CHANNELS consecutive channels starting at
// channel_offset out of total_channels.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	int total_channels,
	int channel_offset,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth,
	int* __restrict__ out_top_ids,
	float* __restrict__ out_top_weights,
	float* __restrict__ out_top_depths,
	int top_L,
	float contributor_min_weight)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W&& pix.y < H;
	bool done = !inside;

	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Pixel-mediated matching (spec section 4/5): keep the top-L contributors
	// ranked by COMPOSITING RESPONSIBILITY rho = T_before * alpha, not raw alpha,
	// so an opaque Gaussian hidden behind an opaque one is not called a major
	// visible contributor. Selection needs no backward pass; this is detached
	// correspondence evidence and the differentiable RGB/depth path is untouched.
	const bool collect_top = (out_top_ids != nullptr) && (top_L > 0);
	int   best_ids[MAX_TOP_CONTRIBUTORS];
	float best_rho[MAX_TOP_CONTRIBUTORS];
	float best_depth[MAX_TOP_CONTRIBUTORS];
	const int L = collect_top ? min(top_L, MAX_TOP_CONTRIBUTORS) : 0;
	for (int l = 0; l < L; l++) { best_ids[l] = -1; best_rho[l] = 0.0f; best_depth[l] = 0.0f; }

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor++;

			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			const int gid = collected_id[j];
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[gid * total_channels + channel_offset + ch] * alpha * T;

			if (invdepth)
				expected_invdepth += (1 / depths[gid]) * alpha * T;

			if (collect_top)
			{
				const float rho = alpha * T;   // T here is still T_before
				if (rho > contributor_min_weight && rho > best_rho[L - 1])
				{
					int slot = L - 1;
					while (slot > 0 && best_rho[slot - 1] < rho)
					{
						best_ids[slot] = best_ids[slot - 1];
						best_rho[slot] = best_rho[slot - 1];
						best_depth[slot] = best_depth[slot - 1];
						slot--;
					}
					best_ids[slot] = gid;
					best_rho[slot] = rho;
					best_depth[slot] = depths[gid];
				}
			}

			T = test_T;
			last_contributor = contributor;
		}
	}

	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[(channel_offset + ch) * H * W + pix_id] = C[ch] + T * bg_color[channel_offset + ch];

		if (invdepth)
			invdepth[pix_id] = expected_invdepth;

		if (collect_top)
		{
			for (int l = 0; l < L; l++)
			{
				const int slot = pix_id * L + l;
				out_top_ids[slot] = best_ids[l];
				out_top_weights[slot] = best_rho[l];
				out_top_depths[slot] = best_depth[l];
			}
		}
	}
}

__forceinline__ __device__ bool computePixelJacobianWorld(
	const int gid,
	const float* __restrict__ means3D,
	const float* __restrict__ projmatrix,
	const int W,
	const int H,
	float3& du,
	float3& dv)
{
	float3 p = {
		means3D[3 * gid + 0],
		means3D[3 * gid + 1],
		means3D[3 * gid + 2]
	};
	float4 clip = transformPoint4x4(p, projmatrix);
	const float w = clip.w;
	if (!(w > 1.0e-8f) || !isfinite(w))
		return false;

	const float inv_w2 = 1.0f / (w * w);
	const float sx = 0.5f * float(max(W - 1, 1));
	const float sy = 0.5f * float(max(H - 1, 1));
	du = {
		sx * (projmatrix[0] * w - clip.x * projmatrix[3]) * inv_w2,
		sx * (projmatrix[4] * w - clip.x * projmatrix[7]) * inv_w2,
		sx * (projmatrix[8] * w - clip.x * projmatrix[11]) * inv_w2
	};
	dv = {
		sy * (projmatrix[1] * w - clip.y * projmatrix[3]) * inv_w2,
		sy * (projmatrix[5] * w - clip.y * projmatrix[7]) * inv_w2,
		sy * (projmatrix[9] * w - clip.y * projmatrix[11]) * inv_w2
	};
	return isfinite(du.x) && isfinite(du.y) && isfinite(du.z)
		&& isfinite(dv.x) && isfinite(dv.y) && isfinite(dv.z);
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
accumulateFlowStatsCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ means3D,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ observed_flow,
	const float* __restrict__ observed_confidence,
	const int64_t* __restrict__ group_ids,
	int num_groups,
	const float* __restrict__ projmatrix,
	float delta_time,
	float min_observation_weight,
	float* __restrict__ normal6,
	float* __restrict__ rhs,
	float* __restrict__ y2,
	float* __restrict__ weight,
	int* __restrict__ count)
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W && pix.y < H;
	const int image_size = W * H;
	float flow_x = 0.0f;
	float flow_y = 0.0f;
	float observation_weight = 0.0f;
	if (inside)
	{
		flow_x = observed_flow[pix_id];
		flow_y = observed_flow[image_size + pix_id];
		observation_weight = observed_confidence[pix_id];
	}
	bool done = !inside
		|| !(observation_weight > min_observation_weight)
		|| !isfinite(observation_weight)
		|| !isfinite(flow_x)
		|| !isfinite(flow_y);

	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ int collected_group[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float3 collected_du[BLOCK_SIZE];
	__shared__ float3 collected_dv[BLOCK_SIZE];

	const float safe_dt = fmaxf(fabsf(delta_time), 1.0e-8f);
	const float target_x = flow_x / safe_dt;
	const float target_y = flow_y / safe_dt;
	const float target_norm2 = target_x * target_x + target_y * target_y;

	float T = 1.0f;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			int group = group_ids != nullptr ? (int)group_ids[coll_id] : coll_id;
			float3 du, dv;
			if (group < 0 || group >= num_groups || !computePixelJacobianWorld(coll_id, means3D, projmatrix, W, H, du, dv))
				group = -1;
			collected_id[block.thread_rank()] = coll_id;
			collected_group[block.thread_rank()] = group;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_du[block.thread_rank()] = du;
			collected_dv[block.thread_rank()] = dv;
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1.0f - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			const int group = collected_group[j];
			if (group >= 0)
			{
				const float contribution = alpha * T;
				const float w = contribution * observation_weight;
				if (w > 0.0f && isfinite(w))
				{
					const float3 du = collected_du[j];
					const float3 dv = collected_dv[j];
					atomicAdd(normal6 + 6 * group + 0, w * (du.x * du.x + dv.x * dv.x));
					atomicAdd(normal6 + 6 * group + 1, w * (du.x * du.y + dv.x * dv.y));
					atomicAdd(normal6 + 6 * group + 2, w * (du.x * du.z + dv.x * dv.z));
					atomicAdd(normal6 + 6 * group + 3, w * (du.y * du.y + dv.y * dv.y));
					atomicAdd(normal6 + 6 * group + 4, w * (du.y * du.z + dv.y * dv.z));
					atomicAdd(normal6 + 6 * group + 5, w * (du.z * du.z + dv.z * dv.z));
					atomicAdd(rhs + 3 * group + 0, w * (du.x * target_x + dv.x * target_y));
					atomicAdd(rhs + 3 * group + 1, w * (du.y * target_x + dv.y * target_y));
					atomicAdd(rhs + 3 * group + 2, w * (du.z * target_x + dv.z * target_y));
					atomicAdd(y2 + group, w * target_norm2);
					atomicAdd(weight + group, w);
					atomicAdd(count + group, 1);
				}
			}

			T = test_T;
		}
	}
}


__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
accumulatePixelFeaturesCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ pixel_features,
	int feature_channels,
	const float* __restrict__ pixel_confidence,
	const int64_t* __restrict__ group_ids,
	int num_groups,
	float min_observation_weight,
	float* __restrict__ feature_sum,
	float* __restrict__ weight,
	float* __restrict__ raw_support,
	float* __restrict__ visible_mass,
	int* __restrict__ count)
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	const float confidence = inside ? pixel_confidence[pix_id] : 0.0f;
	const bool confidence_valid = inside && isfinite(confidence) && (confidence > min_observation_weight);
	// Unlike the RGB renderer, this statistics pass must keep traversing the full
	// sorted tile range after the visible transmittance becomes opaque.  Deeper
	// splats still contribute to raw_support and are therefore distinguishable
	// from Gaussians that never project into the image at all.
	bool done = !inside;
	bool visible_open = inside;
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_group[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	float T = 1.0f;
	const int image_size = W * H;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int gaussian_id = point_list[range.x + progress];
			int group = group_ids != nullptr ? (int)group_ids[gaussian_id] : gaussian_id;
			collected_group[block.thread_rank()] = (group >= 0 && group < num_groups) ? group : -1;
			collected_xy[block.thread_rank()] = points_xy_image[gaussian_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[gaussian_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;

			const int group = collected_group[j];
			if (group >= 0 && isfinite(alpha))
				atomicAdd(raw_support + group, alpha);

			if (!visible_open)
				continue;

			// Match the RGB renderer exactly: when this update would push T below
			// the renderer's early-termination threshold, that Gaussian itself is
			// not blended and neither are deeper Gaussians.  We only stop visible
			// accumulation; raw_support continues above.
			float test_T = T * (1.0f - alpha);
			if (test_T < 0.0001f)
			{
				visible_open = false;
				continue;
			}

			const float ownership = alpha * T;
			if (group >= 0 && ownership > 0.0f && isfinite(ownership))
			{
				atomicAdd(visible_mass + group, ownership);
				if (confidence_valid)
				{
					bool finite_features = true;
					for (int channel = 0; channel < feature_channels; channel++)
						finite_features = finite_features && isfinite(pixel_features[channel * image_size + pix_id]);
					if (finite_features)
					{
						const float contribution = ownership * confidence;
						for (int channel = 0; channel < feature_channels; channel++)
							atomicAdd(feature_sum + group * feature_channels + channel,
								contribution * pixel_features[channel * image_size + pix_id]);
						atomicAdd(weight + group, contribution);
						atomicAdd(count + group, 1);
					}
				}
			}
			T = test_T;
		}
	}
}


template <uint32_t CHANNELS>
static void launchRenderChunk(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	int total_channels,
	int channel_offset,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	const float* depths,
	float* depth,
	int* out_top_ids,
	float* out_top_weights,
	float* out_top_depths,
	int top_L,
	float contributor_min_weight)
{
	renderCUDA<CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		total_channels,
		channel_offset,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths,
		depth,
		out_top_ids,
		out_top_weights,
		out_top_depths,
		top_L,
		contributor_min_weight);
}

static void dispatchRenderChunk(
	int channels,
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	int total_channels,
	int channel_offset,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	const float* depths,
	float* depth,
	int* out_top_ids,
	float* out_top_weights,
	float* out_top_depths,
	int top_L,
	float contributor_min_weight)
{
	switch (channels)
	{
	case 1: launchRenderChunk<1>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 2: launchRenderChunk<2>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 3: launchRenderChunk<3>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 4: launchRenderChunk<4>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 5: launchRenderChunk<5>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 6: launchRenderChunk<6>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 7: launchRenderChunk<7>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 8: launchRenderChunk<8>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 9: launchRenderChunk<9>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 10: launchRenderChunk<10>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 11: launchRenderChunk<11>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 12: launchRenderChunk<12>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 13: launchRenderChunk<13>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 14: launchRenderChunk<14>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 15: launchRenderChunk<15>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 16: launchRenderChunk<16>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 17: launchRenderChunk<17>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 18: launchRenderChunk<18>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 19: launchRenderChunk<19>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 20: launchRenderChunk<20>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 21: launchRenderChunk<21>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 22: launchRenderChunk<22>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 23: launchRenderChunk<23>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 24: launchRenderChunk<24>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 25: launchRenderChunk<25>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 26: launchRenderChunk<26>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 27: launchRenderChunk<27>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 28: launchRenderChunk<28>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 29: launchRenderChunk<29>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 30: launchRenderChunk<30>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 31: launchRenderChunk<31>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	case 32: launchRenderChunk<32>(grid, block, ranges, point_list, W, H, means2D, colors, total_channels, channel_offset, conic_opacity, final_T, n_contrib, bg_color, out_color, depths, depth, out_top_ids, out_top_weights, out_top_depths, top_L, contributor_min_weight); break;
	default:
		throw std::runtime_error("Unsupported render chunk size. This should never happen.");
	}
}


void FORWARD::render_homogeneous_positions(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* means2D,
    const float* homogeneous_positions,
    const float4* conic_opacity,
    float* final_T,
    uint32_t* n_contrib,
    const float* bg_homogeneous,
    float* out_homogeneous,
    const float* depths,
    float* depth)
{
    // Four channels are [sum w*x_t, sum w*y_t, sum w*z_t, sum w].
    // With a zero background, the final channel is accumulated opacity.
    launchRenderChunk<4>(
        grid, block,
        ranges, point_list,
        W, H,
        means2D,
        homogeneous_positions,
        4,
        0,
        conic_opacity,
        final_T,
        n_contrib,
        bg_homogeneous,
        out_homogeneous,
        depths,
        depth,
        nullptr, nullptr, nullptr, 0, 1e-4f);   // no contributor collection here
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	int feature_channels,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	const float* depths,
	float* depth,
	int* out_top_ids,
	float* out_top_weights,
	float* out_top_depths,
	int top_L,
	float contributor_min_weight)
{
	constexpr int MAX_CHUNK = 32;
	for (int channel_offset = 0; channel_offset < feature_channels; channel_offset += MAX_CHUNK)
	{
		const int chunk = min(MAX_CHUNK, feature_channels - channel_offset);
		dispatchRenderChunk(
			chunk,
			grid,
			block,
			ranges,
			point_list,
			W,
			H,
			means2D,
			colors,
			feature_channels,
			channel_offset,
			conic_opacity,
			final_T,
			n_contrib,
			bg_color,
			out_color,
			depths,
			depth,
			// Contributors do not depend on the colour channel chunk, so collect
			// them once on the first chunk only.
			channel_offset == 0 ? out_top_ids : nullptr,
			channel_offset == 0 ? out_top_weights : nullptr,
			channel_offset == 0 ? out_top_depths : nullptr,
			channel_offset == 0 ? top_L : 0,
			contributor_min_weight);
	}
}

void FORWARD::accumulate_flow_stats(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* means3D,
	const float2* means2D,
	const float4* conic_opacity,
	const float* observed_flow,
	const float* observed_confidence,
	const int64_t* group_ids,
	int num_groups,
	const float* projmatrix,
	float delta_time,
	float min_observation_weight,
	float* normal6,
	float* rhs,
	float* y2,
	float* weight,
	int* count)
{
	accumulateFlowStatsCUDA << <grid, block >> > (
		ranges,
		point_list,
		W,
		H,
		means3D,
		means2D,
		conic_opacity,
		observed_flow,
		observed_confidence,
		group_ids,
		num_groups,
		projmatrix,
		delta_time,
		min_observation_weight,
		normal6,
		rhs,
		y2,
		weight,
		count);
}

void FORWARD::accumulate_pixel_features(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float4* conic_opacity,
	const float* pixel_features,
	int feature_channels,
	const float* pixel_confidence,
	const int64_t* group_ids,
	int num_groups,
	float min_observation_weight,
	float* feature_sum,
	float* weight,
	float* raw_support,
	float* visible_mass,
	int* count)
{
	accumulatePixelFeaturesCUDA<<<grid, block>>>(
		ranges, point_list, W, H, means2D, conic_opacity,
		pixel_features, feature_channels, pixel_confidence, group_ids,
		num_groups, min_observation_weight, feature_sum, weight,
		raw_support, visible_mass, count);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
