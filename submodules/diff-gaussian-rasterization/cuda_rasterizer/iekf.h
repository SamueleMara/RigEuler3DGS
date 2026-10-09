/*
 * Sparse alpha-aware error-state IEKF observation operators.
 *
 * CODEX_IMPLEMENT_SPARSE_ALPHA_AWARE_IEKF.md, sections 19-22.
 *
 * These are ADDITIVE: nothing in forward.cu / backward.cu / rasterizer_impl.cu
 * changes, so the legacy render and training path is byte-for-byte unaffected
 * when the estimator is disabled (spec section 0 rule 2).
 *
 * The estimator treats the renderer as a nonlinear measurement function h(x)
 * and needs three operators at a FROZEN linearization:
 *
 *     buildCache   -> preprocess + binning + sort, cached (section 20)
 *     renderRGBD   -> h(x), with normalized depth D = Z/(A+eps)  (section 8)
 *     applyJVP     -> H v   (section 10, tangent alpha compositor)
 *     applyVJP     -> H^T z (section 11, closed-form contributor sensitivities)
 *     buildFisher  -> per-Gaussian 6x6 sum_p H^T R^-1 H  (section 16/21 Kernel E)
 *
 * The cache is what makes the PCG legal: sorting, tile membership, alpha values
 * and every Jacobian coefficient are frozen for the whole linear solve
 * (section 20.3). Changing H inside PCG would destroy conjugacy.
 *
 * Conventions, taken from the PRODUCTION renderer in forward.cu so the
 * linearization matches the residual it explains:
 *   - cov2D = Jp * Sigma_c * Jp^T with Jp built from the frustum-CLAMPED
 *     camera coordinates (computeCov2D clamps t.x,t.y to 1.3*tan_fov*t.z).
 *   - a 0.3 low-pass is added to the 2x2 covariance before inversion, and
 *     opacity is scaled by h_convolution_scaling when antialiasing is on.
 *   - alpha = min(0.99, o_scaled * exp(-0.5 * d^T Q d)) with d = u - x_p.
 *     The 0.99 clamp has a real consequence: where it binds, d(alpha) = 0, and
 *     it also guarantees (1 - alpha) >= 0.01, which is what makes the reverse
 *     suffix traversal in the VJP numerically safe (section 27).
 */

#ifndef CUDA_RASTERIZER_IEKF_H_INCLUDED
#define CUDA_RASTERIZER_IEKF_H_INCLUDED

#include <cstdint>
#include <functional>

namespace CudaRasterizer
{
	namespace IEKF
	{
		// Per-Gaussian linearization cache (section 20.2). Stored as plain
		// device arrays owned by the caller (torch tensors on the Python side)
		// so the cache outlives a single call and can be reused by every PCG
		// matrix-vector product.
		struct CachePointers
		{
			float* mu_c;        // [P,3] camera-space center
			float* depth;       // [P]   center depth d_i = e_z^T mu_c
			float* u;           // [P,2] projected center, pixel units
			float* Jp;          // [P,6] 2x3 projection Jacobian, row major
			float* sigma_c;     // [P,6] camera covariance, upper triangle
			float* conic;       // [P,3] Q = S^-1, (xx, xy, yy)
			float* opacity;     // [P]   o_i * h_convolution_scaling
			int*   radii;       // [P]
			uint32_t* tiles_touched; // [P]
			int* clamp_flags;   // [P] bit0: tx clamped, bit1: ty clamped
		};

		// Preprocess + binning + sort. Returns the number of Gaussian-tile
		// instances; the caller sizes point_list from a first probing call or
		// from tiles_touched. Mirrors the production preprocess exactly.
		int buildCache(
			const int P,
			const int width, const int height,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* opacities,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float tan_fovx, const float tan_fovy,
			const bool antialiasing,
			CachePointers cache,
			float* cov3D_scratch,
			uint32_t* point_offsets,
			char*& list_sorting_space,
			size_t& sorting_size,
			uint64_t* keys_unsorted,
			uint32_t* values_unsorted,
			uint64_t* keys_sorted,
			uint32_t* point_list,
			uint2* ranges,
			int* num_rendered_out,
			bool probe_only,
			bool debug);

		// h(x): RGB + normalized depth + accumulated alpha + final transmittance.
		void renderRGBD(
			const int P,
			const int width, const int height,
			const CachePointers cache,
			const uint2* ranges,
			const uint32_t* point_list,
			const float* colors,        // [P,3] frozen radiance (section 3.1)
			const float* background,    // [3]
			const float eps_depth,
			float* out_color,           // [3,H,W]
			float* out_depth,           // [H,W]   D = Z/(A+eps)
			float* out_alpha,           // [H,W]   A
			float* out_transmittance,   // [H,W]   final T
			// [H,W] number of contributors the front-to-back loop actually
			// composited before transmittance saturated. The reverse passes
			// need it to stop at the same place (adjoint identity).
			uint32_t* out_n_contrib,
			bool debug);

		// H v. v is the 6D pose tangent per Gaussian, [P,6] = [dmu(3), dphi(3)].
		void applyJVP(
			const int P, const int width, const int height,
			const CachePointers cache, const uint2* ranges, const uint32_t* point_list,
			const float* colors, const float* background, const float* means3D,
			const float* viewmatrix, const float* projmatrix,
			const float tan_fovx, const float tan_fovy, const float eps_depth,
			const float* pose_delta,
			float* out_dcolor, float* out_ddepth, bool debug);

		// H^T z. z is an RGB-D pixel adjoint; the result is a 6D pose adjoint.
		// final_T/A/D come from renderRGBD at the SAME linearization and are what
		// let the reverse traversal reconstruct the prefix transmittance.
		void applyVJP(
			const int P, const int width, const int height,
			const CachePointers cache, const uint2* ranges, const uint32_t* point_list,
			const float* colors, const float* background, const float* means3D,
			const float* viewmatrix, const float* projmatrix,
			const float tan_fovx, const float tan_fovy, const float eps_depth,
			const float* z_color, const float* z_depth,
			const float* final_T, const float* final_A, const float* final_D,
			const uint32_t* n_contrib,
			float* out_pose_adjoint, bool debug);

		// Per-Gaussian 6x6 Fisher block sum_p H_{p,i}^T R_p^-1 H_{p,i}
		// (section 21 Kernel E). Upper triangle, 21 values per Gaussian.
		void buildFisher(
			const int P, const int width, const int height,
			const CachePointers cache, const uint2* ranges, const uint32_t* point_list,
			const float* colors, const float* background, const float* means3D,
			const float* viewmatrix, const float* projmatrix,
			const float tan_fovx, const float tan_fovy, const float eps_depth,
			const float* precision_color, const float* precision_depth,
			const float* final_T, const float* final_A, const float* final_D,
			const uint32_t* n_contrib,
			float* out_fisher, bool debug);

		// --- grouped / supernode state (V2 section 3.3) -------------------
		// G_{i,c}: delta_mu_i = delta_t_c + delta_theta_c x (mu_i^0 - c_c^0),
		//          delta_phi_i = delta_theta_c.
		void expandGroupTangent(
			const int P, const int* supernode_id, const float* means3D,
			const float* centroids, const float* delta_zeta,
			float* pose_delta, bool debug);

		// G^T applied to a per-Gaussian pose adjoint.
		void reduceGroupAdjoint(
			const int P, const int* supernode_id, const float* means3D,
			const float* centroids, const float* pose_adjoint,
			float* group_adjoint, bool debug);

		// F_c = sum_i G_{i,c}^T F_i G_{i,c}, preconditioner only.
		void reduceGroupFisher(
			const int P, const int C, const int* supernode_id, const float* means3D,
			const float* centroids, const float* fisher21,
			float* group_fisher, bool debug);
	}
}

#endif
