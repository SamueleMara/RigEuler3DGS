/*
 * Torch bindings for the visible-alpha operators (cuda_rasterizer/iekf_visibility.cu).
 * Cache tensors are the ones iekf_build_cache returns; nothing here builds or
 * modifies a cache.
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>
#include "cuda_rasterizer/iekf.h"
#include "cuda_rasterizer/iekf_visibility.h"

namespace
{
CudaRasterizer::IEKF::CachePointers make_cache(
	torch::Tensor& mu_c, torch::Tensor& depth, torch::Tensor& u, torch::Tensor& Jp,
	torch::Tensor& sigma_c, torch::Tensor& conic, torch::Tensor& opacity,
	torch::Tensor& radii, torch::Tensor& clamp_flags, torch::Tensor& tiles_touched)
{
	CudaRasterizer::IEKF::CachePointers c;
	c.mu_c = mu_c.data_ptr<float>(); c.depth = depth.data_ptr<float>(); c.u = u.data_ptr<float>();
	c.Jp = Jp.data_ptr<float>(); c.sigma_c = sigma_c.data_ptr<float>(); c.conic = conic.data_ptr<float>();
	c.opacity = opacity.data_ptr<float>(); c.radii = radii.data_ptr<int>();
	c.tiles_touched = reinterpret_cast<uint32_t*>(tiles_touched.data_ptr<int>());
	c.clamp_flags = clamp_flags.data_ptr<int>();
	return c;
}

void check_mode(const torch::Tensor& ray_mode, const torch::Tensor& z_obs, int H, int W)
{
	TORCH_CHECK(ray_mode.is_cuda() && ray_mode.scalar_type() == torch::kInt32, "ray_mode must be CUDA int32");
	TORCH_CHECK(z_obs.is_cuda() && z_obs.scalar_type() == torch::kFloat32, "z_obs must be CUDA float32");
	TORCH_CHECK(ray_mode.numel() == (int64_t)H * W && z_obs.numel() == (int64_t)H * W,
	            "ray_mode / z_obs must be [H,W] for this camera");
}
} // namespace

#define UNPACK_CACHE()                                                          \
	auto mu_c_ = cache_mu_c.contiguous(); auto depth_ = cache_depth.contiguous();  \
	auto u_ = cache_u.contiguous(); auto Jp_ = cache_Jp.contiguous();              \
	auto sigma_ = cache_sigma_c.contiguous(); auto conic_ = cache_conic.contiguous(); \
	auto op_ = cache_opacity.contiguous(); auto radii_ = cache_radii.contiguous(); \
	auto clamp_ = cache_clamp.contiguous(); auto tiles_ = torch::zeros_like(radii_); \
	auto cache = make_cache(mu_c_, depth_, u_, Jp_, sigma_, conic_, op_, radii_, clamp_, tiles_);

#define CACHE_ARGS                                                              \
	const torch::Tensor& cache_mu_c, const torch::Tensor& cache_depth,         \
	const torch::Tensor& cache_u, const torch::Tensor& cache_Jp,               \
	const torch::Tensor& cache_sigma_c, const torch::Tensor& cache_conic,      \
	const torch::Tensor& cache_opacity, const torch::Tensor& cache_radii,      \
	const torch::Tensor& cache_clamp, const torch::Tensor& ranges,             \
	const torch::Tensor& point_list

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
IEKFVisibleAlphaCUDA(CACHE_ARGS, const int image_height, const int image_width,
                     const torch::Tensor& ray_mode, const torch::Tensor& z_obs,
                     const float delta_z, const bool debug)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(cache_mu_c));
	const int P = cache_mu_c.size(0);
	check_mode(ray_mode, z_obs, image_height, image_width);
	UNPACK_CACHE()
	auto fopt = cache_mu_c.options().dtype(torch::kFloat32);
	auto iopt = cache_mu_c.options().dtype(torch::kInt32);
	torch::Tensor avis = torch::zeros({image_height, image_width}, fopt);
	torch::Tensor Tf = torch::ones({image_height, image_width}, fopt);
	torch::Tensor ncon = torch::zeros({image_height, image_width}, iopt);
	torch::Tensor nfront = torch::zeros({image_height, image_width}, iopt);
	torch::Tensor nbehind = torch::zeros({image_height, image_width}, iopt);
	if (P > 0 && point_list.numel() > 0)
		CudaRasterizer::IEKFVis::visibleAlpha(P, image_width, image_height, cache,
			reinterpret_cast<const uint2*>(ranges.contiguous().data_ptr<int>()),
			reinterpret_cast<const uint32_t*>(point_list.contiguous().data_ptr<int>()),
			ray_mode.contiguous().data_ptr<int>(), z_obs.contiguous().data_ptr<float>(), delta_z,
			avis.data_ptr<float>(), Tf.data_ptr<float>(),
			reinterpret_cast<uint32_t*>(ncon.data_ptr<int>()),
			nfront.data_ptr<int>(), nbehind.data_ptr<int>(), debug);
	return std::make_tuple(avis, Tf, ncon, nfront, nbehind);
}

torch::Tensor
IEKFVisApplyHCUDA(CACHE_ARGS, const torch::Tensor& means3D,
                  const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix,
                  const float tan_fovx, const float tan_fovy,
                  const int image_height, const int image_width,
                  const torch::Tensor& ray_mode, const torch::Tensor& z_obs, const float delta_z,
                  const torch::Tensor& pose_delta, const bool debug)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(cache_mu_c));
	const int P = cache_mu_c.size(0);
	check_mode(ray_mode, z_obs, image_height, image_width);
	TORCH_CHECK(pose_delta.dim() == 2 && pose_delta.size(0) == P && pose_delta.size(1) == 6, "pose_delta must be [P,6]");
	UNPACK_CACHE()
	torch::Tensor davis = torch::zeros({image_height, image_width}, cache_mu_c.options().dtype(torch::kFloat32));
	if (P > 0 && point_list.numel() > 0)
		CudaRasterizer::IEKFVis::applyJVP(P, image_width, image_height, cache,
			reinterpret_cast<const uint2*>(ranges.contiguous().data_ptr<int>()),
			reinterpret_cast<const uint32_t*>(point_list.contiguous().data_ptr<int>()),
			means3D.contiguous().data_ptr<float>(), viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(), tan_fovx, tan_fovy,
			ray_mode.contiguous().data_ptr<int>(), z_obs.contiguous().data_ptr<float>(), delta_z,
			pose_delta.contiguous().data_ptr<float>(), davis.data_ptr<float>(), debug);
	return davis;
}

torch::Tensor
IEKFVisApplyHTCUDA(CACHE_ARGS, const torch::Tensor& means3D,
                   const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix,
                   const float tan_fovx, const float tan_fovy,
                   const int image_height, const int image_width,
                   const torch::Tensor& ray_mode, const torch::Tensor& z_obs, const float delta_z,
                   const torch::Tensor& z_alpha, const torch::Tensor& final_Tf,
                   const torch::Tensor& n_contrib, const bool debug)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(cache_mu_c));
	const int P = cache_mu_c.size(0);
	check_mode(ray_mode, z_obs, image_height, image_width);
	TORCH_CHECK(z_alpha.numel() == (int64_t)image_height * image_width && final_Tf.numel() == z_alpha.numel()
	            && n_contrib.numel() == z_alpha.numel(), "z_alpha / final_Tf / n_contrib must be [H,W]");
	UNPACK_CACHE()
	torch::Tensor adj = torch::zeros({P, 6}, cache_mu_c.options().dtype(torch::kFloat32));
	if (P > 0 && point_list.numel() > 0)
		CudaRasterizer::IEKFVis::applyVJP(P, image_width, image_height, cache,
			reinterpret_cast<const uint2*>(ranges.contiguous().data_ptr<int>()),
			reinterpret_cast<const uint32_t*>(point_list.contiguous().data_ptr<int>()),
			means3D.contiguous().data_ptr<float>(), viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(), tan_fovx, tan_fovy,
			ray_mode.contiguous().data_ptr<int>(), z_obs.contiguous().data_ptr<float>(), delta_z,
			z_alpha.contiguous().data_ptr<float>(), final_Tf.contiguous().data_ptr<float>(),
			reinterpret_cast<const uint32_t*>(n_contrib.contiguous().data_ptr<int>()),
			adj.data_ptr<float>(), debug);
	return adj;
}

torch::Tensor
IEKFVisBuildFisherCUDA(CACHE_ARGS, const torch::Tensor& means3D,
                       const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix,
                       const float tan_fovx, const float tan_fovy,
                       const int image_height, const int image_width,
                       const torch::Tensor& ray_mode, const torch::Tensor& z_obs, const float delta_z,
                       const torch::Tensor& weight_alpha, const torch::Tensor& final_Tf,
                       const torch::Tensor& n_contrib, const bool debug)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(cache_mu_c));
	const int P = cache_mu_c.size(0);
	check_mode(ray_mode, z_obs, image_height, image_width);
	TORCH_CHECK(weight_alpha.is_cuda() && weight_alpha.scalar_type() == torch::kFloat32,
	            "weight_alpha must be CUDA float32");
	TORCH_CHECK(weight_alpha.numel() == (int64_t)image_height * image_width && final_Tf.numel() == weight_alpha.numel()
	            && n_contrib.numel() == weight_alpha.numel(), "weight_alpha / final_Tf / n_contrib must be [H,W]");
	UNPACK_CACHE()
	torch::Tensor fisher = torch::zeros({P, 21}, cache_mu_c.options().dtype(torch::kFloat32));
	if (P > 0 && point_list.numel() > 0)
		CudaRasterizer::IEKFVis::buildFisher(P, image_width, image_height, cache,
			reinterpret_cast<const uint2*>(ranges.contiguous().data_ptr<int>()),
			reinterpret_cast<const uint32_t*>(point_list.contiguous().data_ptr<int>()),
			means3D.contiguous().data_ptr<float>(), viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(), tan_fovx, tan_fovy,
			ray_mode.contiguous().data_ptr<int>(), z_obs.contiguous().data_ptr<float>(), delta_z,
			weight_alpha.contiguous().data_ptr<float>(), final_Tf.contiguous().data_ptr<float>(),
			reinterpret_cast<const uint32_t*>(n_contrib.contiguous().data_ptr<int>()),
			fisher.data_ptr<float>(), debug);
	return fisher;
}
