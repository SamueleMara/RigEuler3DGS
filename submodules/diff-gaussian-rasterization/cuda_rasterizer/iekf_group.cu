/*
 * Grouped (supernode) observation operators.
 * CODEX_STRICT_CORRECTIONS_IEKF_JOINT_DISCOVERY_V2.md section 3.3.
 *
 * The full-scale free-Gaussian state is structurally ill-conditioned (section
 * 1.2): 44% of pose blocks are weakly observable at 11320 Gaussians. The fix is
 * not a better solver, it is a lower-dimensional state. Each Gaussian is bound
 * to a supernode c, and a 6D supernode perturbation maps to its members by
 *
 *     delta_mu_i  = delta_t_c + delta_theta_c x (mu_i^0 - c_c^0)
 *     delta_phi_i = delta_theta_c                                 (section 3.2)
 *
 * so  H_{p,c} = sum_{i in S_p and c} H_{p,i} G_{i,c}  without ever materialising
 * G or H (section 3.2's closing line).
 *
 * These kernels REUSE the per-Gaussian analytic chain in iekf.cu rather than
 * duplicating it -- the group tangent is expanded to a Gaussian tangent, the
 * existing projection/conic/alpha differentials run unchanged, and the adjoint
 * is reduced back to the group. Section 24 requires exactly that: add grouped
 * state, do not replace the validated primitives.
 */

#include "iekf.h"

#include <iostream>
#include <algorithm>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <stdio.h>

#include "auxiliary.h"
#include "config.h"

namespace cg = cooperative_groups;

#define IEKF_GROUP_SYNC(debug)                                                   \
	if (debug) {                                                                 \
		auto ret = cudaDeviceSynchronize();                                      \
		if (ret != cudaSuccess) {                                                \
			printf("\n[IEKF GROUP CUDA ERROR] %s in %s:%d\n",                     \
			       cudaGetErrorString(ret), __FILE__, __LINE__);                 \
			throw std::runtime_error(cudaGetErrorString(ret));                   \
		}                                                                        \
	}

namespace
{

// Section 3.2's G_{i,c} applied in the forward direction:
//   delta_mu_i = delta_t_c + delta_theta_c x (mu_i^0 - c_c^0),  delta_phi_i = delta_theta_c
__global__ void expandGroupTangentCUDA(
	int P,
	const int* __restrict__ supernode_id,
	const float* __restrict__ means3D,          // canonical mu_i^0
	const float* __restrict__ centroids,        // [C,3] canonical c_c^0
	const float* __restrict__ delta_zeta,       // [C,6]
	float* __restrict__ pose_delta)             // [P,6]
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= (unsigned)P) return;
	const int c = supernode_id[idx];
	if (c < 0)
	{
		// Unassigned Gaussian: a supernode tangent says nothing about it, so its
		// tangent is exactly zero rather than an arbitrary small number.
		for (int k = 0; k < 6; ++k) pose_delta[6 * idx + k] = 0.f;
		return;
	}
	const float3 t = {delta_zeta[6 * c + 0], delta_zeta[6 * c + 1], delta_zeta[6 * c + 2]};
	const float3 th = {delta_zeta[6 * c + 3], delta_zeta[6 * c + 4], delta_zeta[6 * c + 5]};
	const float3 d = {means3D[3 * idx + 0] - centroids[3 * c + 0],
	                  means3D[3 * idx + 1] - centroids[3 * c + 1],
	                  means3D[3 * idx + 2] - centroids[3 * c + 2]};
	// theta x d
	pose_delta[6 * idx + 0] = t.x + th.y * d.z - th.z * d.y;
	pose_delta[6 * idx + 1] = t.y + th.z * d.x - th.x * d.z;
	pose_delta[6 * idx + 2] = t.z + th.x * d.y - th.y * d.x;
	pose_delta[6 * idx + 3] = th.x;
	pose_delta[6 * idx + 4] = th.y;
	pose_delta[6 * idx + 5] = th.z;
}

// Section 3.3's group VJP reduction:
//   t_bar_c     += mu_bar_i
//   theta_bar_c += (mu_i^0 - c_c^0) x mu_bar_i + phi_bar_i
__global__ void reduceGroupAdjointCUDA(
	int P,
	const int* __restrict__ supernode_id,
	const float* __restrict__ means3D,
	const float* __restrict__ centroids,
	const float* __restrict__ pose_adjoint,     // [P,6]
	float* __restrict__ group_adjoint)          // [C,6]
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= (unsigned)P) return;
	const int c = supernode_id[idx];
	if (c < 0) return;
	const float3 mb = {pose_adjoint[6 * idx + 0], pose_adjoint[6 * idx + 1],
	                   pose_adjoint[6 * idx + 2]};
	const float3 pb = {pose_adjoint[6 * idx + 3], pose_adjoint[6 * idx + 4],
	                   pose_adjoint[6 * idx + 5]};
	if (mb.x == 0.f && mb.y == 0.f && mb.z == 0.f &&
	    pb.x == 0.f && pb.y == 0.f && pb.z == 0.f) return;
	const float3 d = {means3D[3 * idx + 0] - centroids[3 * c + 0],
	                  means3D[3 * idx + 1] - centroids[3 * c + 1],
	                  means3D[3 * idx + 2] - centroids[3 * c + 2]};
	atomicAdd(&group_adjoint[6 * c + 0], mb.x);
	atomicAdd(&group_adjoint[6 * c + 1], mb.y);
	atomicAdd(&group_adjoint[6 * c + 2], mb.z);
	// d x mu_bar + phi_bar
	atomicAdd(&group_adjoint[6 * c + 3], d.y * mb.z - d.z * mb.y + pb.x);
	atomicAdd(&group_adjoint[6 * c + 4], d.z * mb.x - d.x * mb.z + pb.y);
	atomicAdd(&group_adjoint[6 * c + 5], d.x * mb.y - d.y * mb.x + pb.z);
}

// Section 3.3's group Fisher. The per-Gaussian 6x6 blocks F_i are congruence-
// transformed by G_{i,c} and summed into the supernode block:
//     F_c += G_{i,c}^T F_i G_{i,c}
// This is the PRECONDITIONER only; the exact operator still runs the full
// JVP/VJP so every cross-supernode co-visibility coupling survives.
__global__ void reduceGroupFisherCUDA(
	int P, int C,
	const int* __restrict__ supernode_id,
	const float* __restrict__ means3D,
	const float* __restrict__ centroids,
	const float* __restrict__ fisher21,         // [P,21] upper triangle
	float* __restrict__ group_fisher)           // [C,21] upper triangle
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= (unsigned)P) return;
	const int c = supernode_id[idx];
	if (c < 0) return;

	// unpack the symmetric 6x6
	float F[36];
	int t = 0;
	for (int a = 0; a < 6; ++a)
		for (int b = a; b < 6; ++b, ++t)
		{
			F[a * 6 + b] = fisher21[21 * idx + t];
			F[b * 6 + a] = fisher21[21 * idx + t];
		}
	bool any = false;
	for (int k = 0; k < 36 && !any; ++k) any = (F[k] != 0.f);
	if (!any) return;

	const float3 d = {means3D[3 * idx + 0] - centroids[3 * c + 0],
	                  means3D[3 * idx + 1] - centroids[3 * c + 1],
	                  means3D[3 * idx + 2] - centroids[3 * c + 2]};
	// G is 6x6: rows = Gaussian tangent (dmu, dphi), cols = group tangent (dt, dtheta)
	//   dmu   = I * dt + [-d]_x * dtheta      (since theta x d = -d x theta)
	//   dphi  = 0  * dt + I * dtheta
	float G[36];
	for (int k = 0; k < 36; ++k) G[k] = 0.f;
	for (int k = 0; k < 3; ++k) { G[k * 6 + k] = 1.f; G[(3 + k) * 6 + (3 + k)] = 1.f; }
	G[0 * 6 + 4] = d.z;  G[0 * 6 + 5] = -d.y;
	G[1 * 6 + 3] = -d.z; G[1 * 6 + 5] = d.x;
	G[2 * 6 + 3] = d.y;  G[2 * 6 + 4] = -d.x;

	// GtFG = G^T F G
	float FG[36];
	for (int r = 0; r < 6; ++r)
		for (int cc = 0; cc < 6; ++cc)
		{
			float acc = 0.f;
			for (int k = 0; k < 6; ++k) acc += F[r * 6 + k] * G[k * 6 + cc];
			FG[r * 6 + cc] = acc;
		}
	t = 0;
	for (int a = 0; a < 6; ++a)
		for (int b = a; b < 6; ++b, ++t)
		{
			float acc = 0.f;
			for (int k = 0; k < 6; ++k) acc += G[k * 6 + a] * FG[k * 6 + b];
			if (acc != 0.f) atomicAdd(&group_fisher[21 * c + t], acc);
		}
}

} // anonymous namespace

namespace CudaRasterizer { namespace IEKF {

void expandGroupTangent(
	const int P, const int* supernode_id, const float* means3D,
	const float* centroids, const float* delta_zeta, float* pose_delta, bool debug)
{
	expandGroupTangentCUDA<<<(P + 255) / 256, 256>>>(
		P, supernode_id, means3D, centroids, delta_zeta, pose_delta);
	IEKF_GROUP_SYNC(debug);
}

void reduceGroupAdjoint(
	const int P, const int* supernode_id, const float* means3D,
	const float* centroids, const float* pose_adjoint, float* group_adjoint,
	bool debug)
{
	reduceGroupAdjointCUDA<<<(P + 255) / 256, 256>>>(
		P, supernode_id, means3D, centroids, pose_adjoint, group_adjoint);
	IEKF_GROUP_SYNC(debug);
}

void reduceGroupFisher(
	const int P, const int C, const int* supernode_id, const float* means3D,
	const float* centroids, const float* fisher21, float* group_fisher, bool debug)
{
	reduceGroupFisherCUDA<<<(P + 255) / 256, 256>>>(
		P, C, supernode_id, means3D, centroids, fisher21, group_fisher);
	IEKF_GROUP_SYNC(debug);
}

}} // namespace CudaRasterizer::IEKF
