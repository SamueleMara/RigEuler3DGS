/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 */
#pragma once

#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansContributorsCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const int top_l,
	const float contributor_min_weight,
	const bool debug);

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool antialiasing,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_invdepth,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool antialiasing,
    const bool debug);

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussianFeaturesCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& features,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool antialiasing,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussianFeaturesBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& features,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor& dL_dout_features,
    const torch::Tensor& dL_dout_invdepth,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool antialiasing,
    const bool debug);

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussianPositionsCUDA(
    const torch::Tensor& query_means3D,
    const torch::Tensor& target_positions,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool antialiasing,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussianPositionsBackwardCUDA(
    const torch::Tensor& query_means3D,
    const torch::Tensor& radii,
    const torch::Tensor& target_positions,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor& dL_dout_homogeneous,
    const torch::Tensor& dL_dout_invdepth,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool antialiasing,
    const bool debug);

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussianFlowStatsCUDA(
    const torch::Tensor& means3D,
    const torch::Tensor& observed_flow,
    const torch::Tensor& observed_confidence,
    const torch::Tensor& group_ids,
    const int num_groups,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool antialiasing,
    const float delta_time,
    const float min_observation_weight,
    const bool debug);

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussianPixelFeatureStatsCUDA(
    const torch::Tensor& means3D,
    const torch::Tensor& pixel_features,
    const torch::Tensor& pixel_confidence,
    const torch::Tensor& group_ids,
    const int num_groups,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool antialiasing,
    const float min_observation_weight,
    const bool debug);

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix);
