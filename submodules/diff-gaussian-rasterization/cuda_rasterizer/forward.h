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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
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
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool antialiasing);

	// Main rasterization method. Supports arbitrary feature-channel counts
	// by rendering them in chunks of compile-time channel count.
	// Specialized four-channel path for premultiplied homogeneous target
	// positions [x_t, y_t, z_t, 1].  The query-frame Gaussian footprints and
	// standard T*alpha weights define pixel ownership; no opacity division is
	// performed.
	void render_homogeneous_positions(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* homogeneous_positions,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_homogeneous,
		float* out_homogeneous,
		const float* depths,
		float* depth);

	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		int feature_channels,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		const float* depths,
		float* depth,
		int* out_top_ids = nullptr,
		float* out_top_weights = nullptr,
		float* out_top_depths = nullptr,
		int top_L = 0,
		float contributor_min_weight = 1e-4f);

	// Rendering-consistent optical-flow inversion primitive. Each observed
	// source pixel contributes its standard 3DGS T*alpha ownership weight to the
	// normal equations J_pi(X_i) v_g ~= flow(p) / dt, where g is either the
	// Gaussian id or a caller-provided cell/group id.
	void accumulate_flow_stats(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* means3D,
		const float2* points_xy_image,
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
		int* count);

	// Generic inverse-rendering primitive. Pixel features are accumulated into
	// Gaussians/groups with the exact front-to-back T*alpha ownership weights.
	void accumulate_pixel_features(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
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
		int* count);
}

#endif
