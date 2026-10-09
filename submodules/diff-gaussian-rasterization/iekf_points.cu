/*
 * Torch bindings for the sparse alpha-aware IEKF observation operators.
 * CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md section 19.1.
 */
#include "iekf_points.h"
#include "cuda_rasterizer/iekf.h"
#include "cuda_rasterizer/config.h"

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace
{

void check_cuda_float(const torch::Tensor& t, const char* name)
{
	TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
	TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
	TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

CudaRasterizer::IEKF::CachePointers make_cache(
	torch::Tensor& mu_c, torch::Tensor& depth, torch::Tensor& u, torch::Tensor& Jp,
	torch::Tensor& sigma_c, torch::Tensor& conic, torch::Tensor& opacity,
	torch::Tensor& radii, torch::Tensor& clamp_flags, torch::Tensor& tiles_touched)
{
	CudaRasterizer::IEKF::CachePointers c;
	c.mu_c = mu_c.data_ptr<float>();
	c.depth = depth.data_ptr<float>();
	c.u = u.data_ptr<float>();
	c.Jp = Jp.data_ptr<float>();
	c.sigma_c = sigma_c.data_ptr<float>();
	c.conic = conic.data_ptr<float>();
	c.opacity = opacity.data_ptr<float>();
	c.radii = radii.data_ptr<int>();
	c.tiles_touched = reinterpret_cast<uint32_t*>(tiles_touched.data_ptr<int>());
	c.clamp_flags = clamp_flags.data_ptr<int>();
	return c;
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, int>
IEKFBuildCacheCUDA(
	const torch::Tensor& means3D, const torch::Tensor& scales,
	const float scale_modifier, const torch::Tensor& rotations,
	const torch::Tensor& opacity, const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix,
	const float tan_fovx, const float tan_fovy,
	const int image_height, const int image_width,
	const bool antialiasing, const bool debug)
{
	check_cuda_float(means3D, "means3D");
	TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3, "means3D must be [P,3]");
	const int P = means3D.size(0);
	const at::cuda::OptionalCUDAGuard guard(device_of(means3D));
	auto fopt = means3D.options().dtype(torch::kFloat32);
	auto iopt = means3D.options().dtype(torch::kInt32);

	torch::Tensor mu_c = torch::zeros({P, 3}, fopt);
	torch::Tensor depth = torch::zeros({P}, fopt);
	torch::Tensor u = torch::zeros({P, 2}, fopt);
	torch::Tensor Jp = torch::zeros({P, 6}, fopt);
	torch::Tensor sigma_c = torch::zeros({P, 6}, fopt);
	torch::Tensor conic = torch::zeros({P, 3}, fopt);
	torch::Tensor op = torch::zeros({P}, fopt);
	torch::Tensor radii = torch::zeros({P}, iopt);
	torch::Tensor clamp_flags = torch::zeros({P}, iopt);
	torch::Tensor tiles_touched = torch::zeros({P}, iopt);
	torch::Tensor cov3D_scratch = torch::zeros({P, 6}, fopt);
	torch::Tensor point_offsets = torch::zeros({P}, iopt);

	const int grid_x = (image_width + BLOCK_X - 1) / BLOCK_X;
	const int grid_y = (image_height + BLOCK_Y - 1) / BLOCK_Y;
	torch::Tensor ranges = torch::zeros({grid_x * grid_y, 2}, iopt);

	auto cache = make_cache(mu_c, depth, u, Jp, sigma_c, conic, op, radii,
	                        clamp_flags, tiles_touched);

	char* sorting_space = nullptr;
	size_t sorting_size = 0;
	int num_rendered = 0;

	// Probe pass: run preprocess + scan only, so the instance buffers can be
	// sized exactly before the sort. This mirrors how the production forward
	// discovers num_rendered.
	torch::Tensor empty_i64 = torch::zeros({1}, means3D.options().dtype(torch::kInt64));
	torch::Tensor empty_i32 = torch::zeros({1}, iopt);
	CudaRasterizer::IEKF::buildCache(
		P, image_width, image_height,
		means3D.contiguous().data_ptr<float>(),
		scales.contiguous().data_ptr<float>(), scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		opacity.contiguous().data_ptr<float>(),
		cov3D_precomp.numel() > 0 ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr,
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		tan_fovx, tan_fovy, antialiasing, cache,
		cov3D_scratch.data_ptr<float>(),
		reinterpret_cast<uint32_t*>(point_offsets.data_ptr<int>()),
		sorting_space, sorting_size,
		reinterpret_cast<uint64_t*>(empty_i64.data_ptr<int64_t>()),
		reinterpret_cast<uint32_t*>(empty_i32.data_ptr<int>()),
		reinterpret_cast<uint64_t*>(empty_i64.data_ptr<int64_t>()),
		reinterpret_cast<uint32_t*>(empty_i32.data_ptr<int>()),
		reinterpret_cast<uint2*>(ranges.data_ptr<int>()),
		&num_rendered, /*probe_only=*/true, debug);

	torch::Tensor point_list = torch::zeros({std::max(num_rendered, 1)}, iopt);
	if (num_rendered > 0)
	{
		auto i64opt = means3D.options().dtype(torch::kInt64);
		torch::Tensor keys_unsorted = torch::zeros({num_rendered}, i64opt);
		torch::Tensor keys_sorted = torch::zeros({num_rendered}, i64opt);
		torch::Tensor values_unsorted = torch::zeros({num_rendered}, iopt);
		CudaRasterizer::IEKF::buildCache(
			P, image_width, image_height,
			means3D.contiguous().data_ptr<float>(),
			scales.contiguous().data_ptr<float>(), scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			opacity.contiguous().data_ptr<float>(),
			cov3D_precomp.numel() > 0 ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr,
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			tan_fovx, tan_fovy, antialiasing, cache,
			cov3D_scratch.data_ptr<float>(),
			reinterpret_cast<uint32_t*>(point_offsets.data_ptr<int>()),
			sorting_space, sorting_size,
			reinterpret_cast<uint64_t*>(keys_unsorted.data_ptr<int64_t>()),
			reinterpret_cast<uint32_t*>(values_unsorted.data_ptr<int>()),
			reinterpret_cast<uint64_t*>(keys_sorted.data_ptr<int64_t>()),
			reinterpret_cast<uint32_t*>(point_list.data_ptr<int>()),
			reinterpret_cast<uint2*>(ranges.data_ptr<int>()),
			&num_rendered, /*probe_only=*/false, debug);
	}

	return std::make_tuple(mu_c, depth, u, Jp, sigma_c, conic, op, radii,
	                       clamp_flags, ranges, point_list, num_rendered);
}

#define UNPACK_CACHE()                                                          \
	auto mu_c_ = cache_mu_c.contiguous();                                       \
	auto depth_ = cache_depth.contiguous();                                     \
	auto u_ = cache_u.contiguous();                                             \
	auto Jp_ = cache_Jp.contiguous();                                           \
	auto sigma_ = cache_sigma_c.contiguous();                                   \
	auto conic_ = cache_conic.contiguous();                                     \
	auto op_ = cache_opacity.contiguous();                                      \
	auto radii_ = cache_radii.contiguous();                                     \
	auto clamp_ = cache_clamp.contiguous();                                     \
	auto tiles_ = torch::zeros_like(radii_);                                    \
	auto cache = make_cache(mu_c_, depth_, u_, Jp_, sigma_, conic_, op_,        \
	                        radii_, clamp_, tiles_);

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
	const float eps_depth, const bool debug)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(cache_mu_c));
	const int P = cache_mu_c.size(0);
	UNPACK_CACHE()
	auto fopt = cache_mu_c.options().dtype(torch::kFloat32);
	torch::Tensor out_color = torch::zeros({3, image_height, image_width}, fopt);
	torch::Tensor out_depth = torch::zeros({image_height, image_width}, fopt);
	torch::Tensor out_alpha = torch::zeros({image_height, image_width}, fopt);
	torch::Tensor out_T = torch::ones({image_height, image_width}, fopt);
	// How many contributors the front-to-back loop actually composited before
	// transmittance saturated. The reverse passes must stop at the same index
	// or they transpose an operator the forward never applied.
	torch::Tensor out_n = torch::zeros({image_height, image_width},
	                                   cache_mu_c.options().dtype(torch::kInt32));

	CudaRasterizer::IEKF::renderRGBD(
		P, image_width, image_height, cache,
		reinterpret_cast<const uint2*>(ranges.contiguous().data_ptr<int>()),
		reinterpret_cast<const uint32_t*>(point_list.contiguous().data_ptr<int>()),
		colors.contiguous().data_ptr<float>(),
		background.contiguous().data_ptr<float>(), eps_depth,
		out_color.data_ptr<float>(), out_depth.data_ptr<float>(),
		out_alpha.data_ptr<float>(), out_T.data_ptr<float>(),
		reinterpret_cast<uint32_t*>(out_n.data_ptr<int>()), debug);
	return std::make_tuple(out_color, out_depth, out_alpha, out_T, out_n);
}

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
	const float eps_depth, const torch::Tensor& pose_delta, const bool debug)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(cache_mu_c));
	const int P = cache_mu_c.size(0);
	TORCH_CHECK(pose_delta.dim() == 2 && pose_delta.size(0) == P && pose_delta.size(1) == 6,
	            "pose_delta must be [P,6]");
	UNPACK_CACHE()
	auto fopt = cache_mu_c.options().dtype(torch::kFloat32);
	torch::Tensor d_color = torch::zeros({3, image_height, image_width}, fopt);
	torch::Tensor d_depth = torch::zeros({image_height, image_width}, fopt);

	CudaRasterizer::IEKF::applyJVP(
		P, image_width, image_height, cache,
		reinterpret_cast<const uint2*>(ranges.contiguous().data_ptr<int>()),
		reinterpret_cast<const uint32_t*>(point_list.contiguous().data_ptr<int>()),
		colors.contiguous().data_ptr<float>(),
		background.contiguous().data_ptr<float>(),
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		tan_fovx, tan_fovy, eps_depth,
		pose_delta.contiguous().data_ptr<float>(),
		d_color.data_ptr<float>(), d_depth.data_ptr<float>(), debug);
	return std::make_tuple(d_color, d_depth);
}

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
	const torch::Tensor& final_D, const bool debug)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(cache_mu_c));
	const int P = cache_mu_c.size(0);
	UNPACK_CACHE()
	torch::Tensor adjoint = torch::zeros({P, 6}, cache_mu_c.options().dtype(torch::kFloat32));

	CudaRasterizer::IEKF::applyVJP(
		P, image_width, image_height, cache,
		reinterpret_cast<const uint2*>(ranges.contiguous().data_ptr<int>()),
		reinterpret_cast<const uint32_t*>(point_list.contiguous().data_ptr<int>()),
		colors.contiguous().data_ptr<float>(),
		background.contiguous().data_ptr<float>(),
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		tan_fovx, tan_fovy, eps_depth,
		z_color.contiguous().data_ptr<float>(),
		z_depth.contiguous().data_ptr<float>(),
		final_T.contiguous().data_ptr<float>(),
		final_A.contiguous().data_ptr<float>(),
		final_D.contiguous().data_ptr<float>(),
		reinterpret_cast<const uint32_t*>(n_contrib.contiguous().data_ptr<int>()),
		adjoint.data_ptr<float>(), debug);
	return adjoint;
}

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
	const torch::Tensor& final_D, const bool debug)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(cache_mu_c));
	const int P = cache_mu_c.size(0);
	UNPACK_CACHE()
	torch::Tensor fisher = torch::zeros({P, 21}, cache_mu_c.options().dtype(torch::kFloat32));

	CudaRasterizer::IEKF::buildFisher(
		P, image_width, image_height, cache,
		reinterpret_cast<const uint2*>(ranges.contiguous().data_ptr<int>()),
		reinterpret_cast<const uint32_t*>(point_list.contiguous().data_ptr<int>()),
		colors.contiguous().data_ptr<float>(),
		background.contiguous().data_ptr<float>(),
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		tan_fovx, tan_fovy, eps_depth,
		precision_color.contiguous().data_ptr<float>(),
		precision_depth.contiguous().data_ptr<float>(),
		final_T.contiguous().data_ptr<float>(),
		final_A.contiguous().data_ptr<float>(),
		final_D.contiguous().data_ptr<float>(),
		reinterpret_cast<const uint32_t*>(n_contrib.contiguous().data_ptr<int>()),
		fisher.data_ptr<float>(), debug);
	return fisher;
}

// ===========================================================================
// Grouped / supernode operators (V2 section 3.3)
// ===========================================================================
torch::Tensor IEKFExpandGroupTangentCUDA(
	const torch::Tensor& supernode_id, const torch::Tensor& means3D,
	const torch::Tensor& centroids, const torch::Tensor& delta_zeta,
	const bool debug)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(means3D));
	const int P = means3D.size(0);
	TORCH_CHECK(supernode_id.size(0) == P, "supernode_id must have one entry per Gaussian");
	TORCH_CHECK(delta_zeta.dim() == 2 && delta_zeta.size(1) == 6, "delta_zeta must be [C,6]");
	torch::Tensor pose = torch::zeros({P, 6}, means3D.options().dtype(torch::kFloat32));
	CudaRasterizer::IEKF::expandGroupTangent(
		P, supernode_id.contiguous().data_ptr<int>(),
		means3D.contiguous().data_ptr<float>(),
		centroids.contiguous().data_ptr<float>(),
		delta_zeta.contiguous().data_ptr<float>(),
		pose.data_ptr<float>(), debug);
	return pose;
}

torch::Tensor IEKFReduceGroupAdjointCUDA(
	const torch::Tensor& supernode_id, const torch::Tensor& means3D,
	const torch::Tensor& centroids, const torch::Tensor& pose_adjoint,
	const int num_supernodes, const bool debug)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(means3D));
	const int P = means3D.size(0);
	torch::Tensor out = torch::zeros({num_supernodes, 6},
	                                 means3D.options().dtype(torch::kFloat32));
	CudaRasterizer::IEKF::reduceGroupAdjoint(
		P, supernode_id.contiguous().data_ptr<int>(),
		means3D.contiguous().data_ptr<float>(),
		centroids.contiguous().data_ptr<float>(),
		pose_adjoint.contiguous().data_ptr<float>(),
		out.data_ptr<float>(), debug);
	return out;
}

torch::Tensor IEKFReduceGroupFisherCUDA(
	const torch::Tensor& supernode_id, const torch::Tensor& means3D,
	const torch::Tensor& centroids, const torch::Tensor& fisher21,
	const int num_supernodes, const bool debug)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(means3D));
	const int P = means3D.size(0);
	torch::Tensor out = torch::zeros({num_supernodes, 21},
	                                 means3D.options().dtype(torch::kFloat32));
	CudaRasterizer::IEKF::reduceGroupFisher(
		P, num_supernodes, supernode_id.contiguous().data_ptr<int>(),
		means3D.contiguous().data_ptr<float>(),
		centroids.contiguous().data_ptr<float>(),
		fisher21.contiguous().data_ptr<float>(),
		out.data_ptr<float>(), debug);
	return out;
}
