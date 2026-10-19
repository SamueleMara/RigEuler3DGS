/*
 * Device helpers shared by the IEKF observation operators (iekf.cu) and the
 * visible-alpha operators (iekf_visibility.cu). Moved here VERBATIM from
 * iekf.cu so both walk the same contributor set with the same alpha,
 * clamp, termination and pose-tangent code -- one implementation, no drift.
 */
#pragma once

#include "auxiliary.h"
#include "config.h"

namespace
{

// The renderer's own guards, repeated here so the linearization is built on the
// exact same contributor set the residual came from.
__device__ const float ALPHA_MAX = 0.99f;
__device__ const float ALPHA_MIN = 1.0f / 255.0f;
__device__ const float T_STOP = 1e-4f;
__device__ const float LOWPASS = 0.3f;

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


} // anonymous namespace
