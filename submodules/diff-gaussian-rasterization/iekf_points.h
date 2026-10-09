/*
 * Torch bindings for the sparse alpha-aware IEKF observation operators.
 * CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md section 19.1.
 *
 * Additive only: every existing binding in rasterize_points.h is untouched.
 */
#pragma once

#include <torch/extension.h>
#include <tuple>

// Preprocess + binning + sort at the current state. Returns the frozen
// linearization cache (section 20.2) that every subsequent operator consumes.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, int>
IEKFBuildCacheCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const float scale_modifier,
	const torch::Tensor& rotations,
	const torch::Tensor& opacity,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const bool antialiasing,
	const bool debug);

// h(x): RGB, normalized depth D = Z/(A+eps), accumulated alpha A, final T, and
// the per-pixel contributor count the reverse passes must stop at.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
IEKFRenderCUDA(
	const torch::Tensor& cache_mu_c, const torch::Tensor& cache_depth,
	const torch::Tensor& cache_u, const torch::Tensor& cache_Jp,
	const torch::Tensor& cache_sigma_c, const torch::Tensor& cache_conic,
	const torch::Tensor& cache_opacity, const torch::Tensor& cache_radii,
	const torch::Tensor& cache_clamp, const torch::Tensor& ranges,
	const torch::Tensor& point_list,
	const torch::Tensor& colors, const torch::Tensor& background,
	const int image_height, const int image_width,
	const float eps_depth, const bool debug);

// H v
std::tuple<torch::Tensor, torch::Tensor>
IEKFApplyHCUDA(
	const torch::Tensor& cache_mu_c, const torch::Tensor& cache_depth,
	const torch::Tensor& cache_u, const torch::Tensor& cache_Jp,
	const torch::Tensor& cache_sigma_c, const torch::Tensor& cache_conic,
	const torch::Tensor& cache_opacity, const torch::Tensor& cache_radii,
	const torch::Tensor& cache_clamp, const torch::Tensor& ranges,
	const torch::Tensor& point_list,
	const torch::Tensor& colors, const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix,
	const float tan_fovx, const float tan_fovy,
	const int image_height, const int image_width,
	const float eps_depth, const torch::Tensor& pose_delta, const bool debug);

// H^T z
torch::Tensor
IEKFApplyHTCUDA(
	const torch::Tensor& cache_mu_c, const torch::Tensor& cache_depth,
	const torch::Tensor& cache_u, const torch::Tensor& cache_Jp,
	const torch::Tensor& cache_sigma_c, const torch::Tensor& cache_conic,
	const torch::Tensor& cache_opacity, const torch::Tensor& cache_radii,
	const torch::Tensor& cache_clamp, const torch::Tensor& ranges,
	const torch::Tensor& point_list,
	const torch::Tensor& colors, const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix,
	const float tan_fovx, const float tan_fovy,
	const int image_height, const int image_width, const float eps_depth,
	const torch::Tensor& z_color, const torch::Tensor& z_depth,
	const torch::Tensor& n_contrib,
	const torch::Tensor& final_T, const torch::Tensor& final_A,
	const torch::Tensor& final_D, const bool debug);

// Per-Gaussian 6x6 Fisher blocks, upper triangle [P,21]
torch::Tensor
IEKFBuildFisherCUDA(
	const torch::Tensor& cache_mu_c, const torch::Tensor& cache_depth,
	const torch::Tensor& cache_u, const torch::Tensor& cache_Jp,
	const torch::Tensor& cache_sigma_c, const torch::Tensor& cache_conic,
	const torch::Tensor& cache_opacity, const torch::Tensor& cache_radii,
	const torch::Tensor& cache_clamp, const torch::Tensor& ranges,
	const torch::Tensor& point_list,
	const torch::Tensor& colors, const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix,
	const float tan_fovx, const float tan_fovy,
	const int image_height, const int image_width, const float eps_depth,
	const torch::Tensor& precision_color, const torch::Tensor& precision_depth,
	const torch::Tensor& n_contrib,
	const torch::Tensor& final_T, const torch::Tensor& final_A,
	const torch::Tensor& final_D, const bool debug);

// --- grouped / supernode operators (V2 section 3.3) ------------------------
torch::Tensor IEKFExpandGroupTangentCUDA(
	const torch::Tensor& supernode_id, const torch::Tensor& means3D,
	const torch::Tensor& centroids, const torch::Tensor& delta_zeta,
	const bool debug);

torch::Tensor IEKFReduceGroupAdjointCUDA(
	const torch::Tensor& supernode_id, const torch::Tensor& means3D,
	const torch::Tensor& centroids, const torch::Tensor& pose_adjoint,
	const int num_supernodes, const bool debug);

torch::Tensor IEKFReduceGroupFisherCUDA(
	const torch::Tensor& supernode_id, const torch::Tensor& means3D,
	const torch::Tensor& centroids, const torch::Tensor& fisher21,
	const int num_supernodes, const bool debug);
