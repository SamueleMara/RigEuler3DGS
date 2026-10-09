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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

static void check_feature_inputs(
    const torch::Tensor& background,
    const torch::Tensor& features,
    const torch::Tensor& means3D)
{
  if (features.ndimension() != 2) {
    AT_ERROR("features must have dimensions (num_points, num_channels)");
  }
  if (means3D.size(0) != features.size(0)) {
    AT_ERROR("features and means3D must have the same number of points");
  }
  if (background.ndimension() != 1 || background.size(0) != features.size(1)) {
    AT_ERROR("background must have shape (num_channels,) matching features.size(1)");
  }
}

// ---------------------------------------------------------------------------
// Pixel-mediated matching (spec sections 3-6): same forward pass, plus the
// per-pixel top-L contributor list ranked by compositing responsibility
// rho = T_before * alpha. Separate entry point so the differentiable training
// path is left exactly as it was.
// ---------------------------------------------------------------------------
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
	const bool debug)
{
	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;
	const int L = std::max(1, top_l);

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, int_opts);
	torch::Tensor top_ids = torch::full({H, W, L}, -1, int_opts);
	torch::Tensor top_weights = torch::full({H, W, L}, 0.0, float_opts);
	torch::Tensor top_depths = torch::full({H, W, L}, 0.0, float_opts);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	if (P != 0)
	{
		int M = 0;
		if (sh.size(0) != 0)
			M = sh.size(1);

		CudaRasterizer::Rasterizer::forward(
			geomFunc, binningFunc, imgFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().numel() ? sh.contiguous().data_ptr<float>() : nullptr,
			colors.contiguous().numel() ? colors.contiguous().data<float>() : nullptr,
			opacity.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().numel() ? cov3D_precomp.contiguous().data<float>() : nullptr,
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx, tan_fovy,
			prefiltered,
			out_color.contiguous().data<float>(),
			nullptr,
			antialiasing,
			radii.contiguous().data<int>(),
			debug,
			nullptr,
			0,
			top_ids.contiguous().data<int>(),
			top_weights.contiguous().data<float>(),
			top_depths.contiguous().data<float>(),
			L,
			contributor_min_weight);
	}
	return std::make_tuple(top_ids, top_weights, top_depths, out_color, radii);
}

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
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  float* out_invdepthptr = out_invdepth.data<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().numel() ? sh.contiguous().data_ptr<float>() : nullptr,
		colors.contiguous().numel() ? colors.contiguous().data<float>() : nullptr,
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().numel() ? cov3D_precomp.contiguous().data<float>() : nullptr,
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_invdepthptr,
		antialiasing,
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);
}

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
	const bool debug)
{
  const int P = means3D.size(0);
  int M = 0;
  if(sh.size(0) != 0)
  {
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());
  
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  if(dL_dout_invdepth.size(0) != 0)
  {
	dL_dinvdepths = torch::zeros({P, 1}, means3D.options()).contiguous();
	dL_dinvdepthsptr = dL_dinvdepths.data<float>();
	dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  if(P != 0)
  {
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  dL_dout_color.size(2), dL_dout_color.size(1), 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().numel() ? sh.contiguous().data<float>() : nullptr,
	  colors.contiguous().numel() ? colors.contiguous().data<float>() : nullptr,
	  opacities.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().numel() ? cov3D_precomp.contiguous().data<float>() : nullptr,
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_invdepthptr,
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dinvdepthsptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().numel() ? dL_dsh.contiguous().data<float>() : nullptr,
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  antialiasing,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

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
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  check_feature_inputs(background, features, means3D);

  const int P = means3D.size(0);
  const int C = features.size(1);
  const int H = image_height;
  const int W = image_width;
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_features = torch::full({C, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  auto geomFunc = resizeFunctional(geomBuffer);
  auto binningFunc = resizeFunctional(binningBuffer);
  auto imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0)
  {
    rendered = CudaRasterizer::Rasterizer::forward(
      geomFunc,
      binningFunc,
      imgFunc,
      P,
      0,
      0,
      background.contiguous().data<float>(),
      W,
      H,
      means3D.contiguous().data<float>(),
      nullptr,
      features.contiguous().data<float>(), // non-null sentinel for preprocess
      opacity.contiguous().data<float>(),
      scales.contiguous().data_ptr<float>(),
      scale_modifier,
      rotations.contiguous().data_ptr<float>(),
      cov3D_precomp.contiguous().numel() ? cov3D_precomp.contiguous().data<float>() : nullptr,
      viewmatrix.contiguous().data<float>(),
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      prefiltered,
      out_features.contiguous().data<float>(),
      out_invdepth.contiguous().data<float>(),
      antialiasing,
      radii.contiguous().data<int>(),
      debug,
      features.contiguous().data<float>(),
      C);
  }

  return std::make_tuple(rendered, out_features, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);
}

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
	const bool debug)
{
  check_feature_inputs(background, features, means3D);
  const int P = means3D.size(0);
  const int C = features.size(1);

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dfeatures = torch::zeros({P, C}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({0, 0, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());

  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  if (dL_dout_invdepth.size(0) != 0)
  {
    dL_dinvdepths = torch::zeros({P, 1}, means3D.options()).contiguous();
    dL_dinvdepthsptr = dL_dinvdepths.data<float>();
    dL_dout_invdepthptr = dL_dout_invdepth.contiguous().data<float>();
  }

  if (P != 0)
  {
    CudaRasterizer::Rasterizer::backward(
      P,
      0,
      0,
      R,
      background.contiguous().data<float>(),
      dL_dout_features.size(2),
      dL_dout_features.size(1),
      means3D.contiguous().data<float>(),
      nullptr,
      features.contiguous().data<float>(),
      opacities.contiguous().data<float>(),
      scales.contiguous().data_ptr<float>(),
      scale_modifier,
      rotations.contiguous().data_ptr<float>(),
      cov3D_precomp.contiguous().numel() ? cov3D_precomp.contiguous().data<float>() : nullptr,
      viewmatrix.contiguous().data<float>(),
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      radii.contiguous().data<int>(),
      reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
      reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
      reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
      dL_dout_features.contiguous().data<float>(),
      dL_dout_invdepthptr,
      dL_dmeans2D.contiguous().data<float>(),
      dL_dconic.contiguous().data<float>(),
      dL_dopacity.contiguous().data<float>(),
      dL_dfeatures.contiguous().data<float>(),
      dL_dinvdepthsptr,
      dL_dmeans3D.contiguous().data<float>(),
      dL_dcov3D.contiguous().data<float>(),
      nullptr,
      dL_dscales.contiguous().data<float>(),
      dL_drotations.contiguous().data<float>(),
      antialiasing,
      debug,
      features.contiguous().data<float>(),
      C);
  }

  return std::make_tuple(dL_dmeans2D, dL_dfeatures, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}


static void check_position_inputs(
    const torch::Tensor& query_means3D,
    const torch::Tensor& target_positions)
{
  if (query_means3D.ndimension() != 2 || query_means3D.size(1) != 3) {
    AT_ERROR("query means3D must have dimensions (num_points, 3)");
  }
  if (target_positions.ndimension() != 2 || target_positions.size(1) != 3) {
    AT_ERROR("target_positions must have dimensions (num_points, 3)");
  }
  if (query_means3D.size(0) != target_positions.size(0)) {
    AT_ERROR("query means3D and target_positions must have the same number of points");
  }
}

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
    const bool debug)
{
  check_position_inputs(query_means3D, target_positions);
  auto ones = torch::ones(
      {target_positions.size(0), 1}, target_positions.options());
  auto homogeneous_positions = torch::cat(
      {target_positions, ones}, 1).contiguous();
  auto background = torch::zeros({4}, target_positions.options());
  return RasterizeGaussianFeaturesCUDA(
      background,
      query_means3D,
      homogeneous_positions,
      opacity,
      scales,
      rotations,
      scale_modifier,
      cov3D_precomp,
      viewmatrix,
      projmatrix,
      tan_fovx,
      tan_fovy,
      image_height,
      image_width,
      campos,
      prefiltered,
      antialiasing,
      debug);
}

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
    const bool debug)
{
  check_position_inputs(query_means3D, target_positions);
  if (dL_dout_homogeneous.ndimension() != 3 || dL_dout_homogeneous.size(0) != 4) {
    AT_ERROR("position rasterizer output gradient must have shape (4,H,W)");
  }
  auto ones = torch::ones(
      {target_positions.size(0), 1}, target_positions.options());
  auto homogeneous_positions = torch::cat(
      {target_positions, ones}, 1).contiguous();
  auto background = torch::zeros({4}, target_positions.options());
  auto gradients = RasterizeGaussianFeaturesBackwardCUDA(
      background,
      query_means3D,
      radii,
      homogeneous_positions,
      opacities,
      scales,
      rotations,
      scale_modifier,
      cov3D_precomp,
      viewmatrix,
      projmatrix,
      tan_fovx,
      tan_fovy,
      dL_dout_homogeneous,
      dL_dout_invdepth,
      campos,
      geomBuffer,
      R,
      binningBuffer,
      imageBuffer,
      antialiasing,
      debug);

  auto dL_dhomogeneous = std::get<1>(gradients);
  auto dL_dtarget_positions = dL_dhomogeneous.slice(1, 0, 3).contiguous();
  return std::make_tuple(
      std::get<0>(gradients),
      dL_dtarget_positions,
      std::get<2>(gradients),
      std::get<3>(gradients),
      std::get<4>(gradients),
      std::get<5>(gradients),
      std::get<6>(gradients),
      std::get<7>(gradients));
}

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
    const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  if (observed_flow.ndimension() != 3 || observed_flow.size(0) != 2) {
    AT_ERROR("observed_flow must have shape (2,H,W)");
  }
  if (observed_confidence.ndimension() != 3 || observed_confidence.size(0) != 1) {
    AT_ERROR("observed_confidence must have shape (1,H,W)");
  }
  if (observed_flow.size(1) != image_height || observed_flow.size(2) != image_width) {
    AT_ERROR("observed_flow resolution must match image_height/image_width");
  }
  if (observed_confidence.size(1) != image_height || observed_confidence.size(2) != image_width) {
    AT_ERROR("observed_confidence resolution must match image_height/image_width");
  }
  if (observed_flow.scalar_type() != torch::kFloat32 || observed_confidence.scalar_type() != torch::kFloat32) {
    AT_ERROR("observed_flow and observed_confidence must be float32");
  }
  if (group_ids.numel() != 0) {
    if (group_ids.ndimension() != 1 || group_ids.size(0) != means3D.size(0)) {
      AT_ERROR("group_ids must be empty or have shape (num_points,)");
    }
    if (group_ids.scalar_type() != torch::kInt64) {
      AT_ERROR("group_ids must be int64/torch.long");
    }
    if (num_groups < 0) {
      AT_ERROR("num_groups must be non-negative when group_ids are provided");
    }
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  const int G = group_ids.numel() == 0 ? P : num_groups;
  auto float_opts = means3D.options().dtype(torch::kFloat32);
  auto int_opts = means3D.options().dtype(torch::kInt32);

  torch::Tensor normal6 = torch::zeros({G, 6}, float_opts);
  torch::Tensor rhs = torch::zeros({G, 3}, float_opts);
  torch::Tensor y2 = torch::zeros({G}, float_opts);
  torch::Tensor weight = torch::zeros({G}, float_opts);
  torch::Tensor count = torch::zeros({G}, int_opts);
  torch::Tensor radii = torch::full({P}, 0, int_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions buffer_options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, buffer_options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, buffer_options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, buffer_options.device(device));
  auto geomFunc = resizeFunctional(geomBuffer);
  auto binningFunc = resizeFunctional(binningBuffer);
  auto imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0 && G != 0)
  {
    auto means3D_c = means3D.contiguous();
    auto flow_c = observed_flow.contiguous();
    auto confidence_c = observed_confidence.contiguous();
    auto group_ids_c = group_ids.contiguous();
    auto opacity_c = opacity.contiguous();
    auto scales_c = scales.contiguous();
    auto rotations_c = rotations.contiguous();
    auto cov3D_c = cov3D_precomp.contiguous();
    auto view_c = viewmatrix.contiguous();
    auto proj_c = projmatrix.contiguous();
    auto campos_c = campos.contiguous();
    const int64_t* group_ptr = group_ids_c.numel() ? group_ids_c.data_ptr<int64_t>() : nullptr;

    rendered = CudaRasterizer::Rasterizer::accumulateFlowStats(
      geomFunc,
      binningFunc,
      imgFunc,
      P,
      W,
      H,
      means3D_c.data_ptr<float>(),
      flow_c.data_ptr<float>(),
      confidence_c.data_ptr<float>(),
      group_ptr,
      G,
      opacity_c.data_ptr<float>(),
      scales_c.numel() ? scales_c.data_ptr<float>() : nullptr,
      scale_modifier,
      rotations_c.numel() ? rotations_c.data_ptr<float>() : nullptr,
      cov3D_c.numel() ? cov3D_c.data_ptr<float>() : nullptr,
      view_c.data_ptr<float>(),
      proj_c.data_ptr<float>(),
      campos_c.data_ptr<float>(),
      tan_fovx,
      tan_fovy,
      prefiltered,
      antialiasing,
      delta_time,
      min_observation_weight,
      normal6.data_ptr<float>(),
      rhs.data_ptr<float>(),
      y2.data_ptr<float>(),
      weight.data_ptr<float>(),
      count.data_ptr<int>(),
      radii.data_ptr<int>(),
      debug);
  }

  return std::make_tuple(rendered, normal6, rhs, y2, weight, count, radii);
}

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
    const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3)
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  if (pixel_features.ndimension() != 3 || pixel_features.size(0) < 1)
    AT_ERROR("pixel_features must have shape (C,H,W) with C >= 1");
  if (pixel_confidence.ndimension() != 3 || pixel_confidence.size(0) != 1)
    AT_ERROR("pixel_confidence must have shape (1,H,W)");
  if (pixel_features.size(1) != image_height || pixel_features.size(2) != image_width ||
      pixel_confidence.size(1) != image_height || pixel_confidence.size(2) != image_width)
    AT_ERROR("pixel feature/confidence resolution must match image_height/image_width");
  if (pixel_features.scalar_type() != torch::kFloat32 || pixel_confidence.scalar_type() != torch::kFloat32)
    AT_ERROR("pixel_features and pixel_confidence must be float32");
  if (group_ids.numel() != 0) {
    if (group_ids.ndimension() != 1 || group_ids.size(0) != means3D.size(0))
      AT_ERROR("group_ids must be empty or have shape (num_points,)");
    if (group_ids.scalar_type() != torch::kInt64)
      AT_ERROR("group_ids must be int64/torch.long");
  }

  const int P = means3D.size(0);
  const int C = pixel_features.size(0);
  const int G = group_ids.numel() == 0 ? P : num_groups;
  auto float_opts = means3D.options().dtype(torch::kFloat32);
  auto int_opts = means3D.options().dtype(torch::kInt32);
  torch::Tensor feature_sum = torch::zeros({G, C}, float_opts);
  // weight is the observation-valid ownership mass: sum(confidence * T * alpha).
  torch::Tensor weight = torch::zeros({G}, float_opts);
  // raw_support ignores transmittance and therefore remains non-zero for
  // projected Gaussians hidden behind opaque foreground.
  torch::Tensor raw_support = torch::zeros({G}, float_opts);
  // visible_mass is exact renderer ownership before observation confidence:
  // sum(T * alpha), with the same early-termination semantics as RGB.
  torch::Tensor visible_mass = torch::zeros({G}, float_opts);
  torch::Tensor count = torch::zeros({G}, int_opts);
  torch::Tensor radii = torch::full({P}, 0, int_opts);
  torch::Device device(torch::kCUDA);
  torch::TensorOptions buffer_options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, buffer_options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, buffer_options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, buffer_options.device(device));
  auto geomFunc = resizeFunctional(geomBuffer);
  auto binningFunc = resizeFunctional(binningBuffer);
  auto imgFunc = resizeFunctional(imgBuffer);
  int rendered = 0;
  if (P != 0 && G != 0) {
    auto means_c = means3D.contiguous();
    auto features_c = pixel_features.contiguous();
    auto confidence_c = pixel_confidence.contiguous();
    auto groups_c = group_ids.contiguous();
    auto opacity_c = opacity.contiguous();
    auto scales_c = scales.contiguous();
    auto rotations_c = rotations.contiguous();
    auto cov_c = cov3D_precomp.contiguous();
    auto view_c = viewmatrix.contiguous();
    auto proj_c = projmatrix.contiguous();
    auto campos_c = campos.contiguous();
    const int64_t* group_ptr = groups_c.numel() ? groups_c.data_ptr<int64_t>() : nullptr;
    rendered = CudaRasterizer::Rasterizer::accumulatePixelFeatures(
      geomFunc, binningFunc, imgFunc, P, image_width, image_height,
      means_c.data_ptr<float>(), features_c.data_ptr<float>(), C,
      confidence_c.data_ptr<float>(), group_ptr, G, opacity_c.data_ptr<float>(),
      scales_c.numel() ? scales_c.data_ptr<float>() : nullptr, scale_modifier,
      rotations_c.numel() ? rotations_c.data_ptr<float>() : nullptr,
      cov_c.numel() ? cov_c.data_ptr<float>() : nullptr,
      view_c.data_ptr<float>(), proj_c.data_ptr<float>(), campos_c.data_ptr<float>(),
      tan_fovx, tan_fovy, prefiltered, antialiasing, min_observation_weight,
      feature_sum.data_ptr<float>(), weight.data_ptr<float>(),
      raw_support.data_ptr<float>(), visible_mass.data_ptr<float>(), count.data_ptr<int>(),
      radii.data_ptr<int>(), debug);
  }
  return std::make_tuple(rendered, feature_sum, weight, count, raw_support, visible_mass, radii);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
