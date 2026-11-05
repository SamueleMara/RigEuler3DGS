/*
 * First-return RGB-D depth likelihood for the Phase-1 IEKF (opt-in).
 * RigEuler3DGS_CODEX_Phase1_FirstReturn_Implementation_2026-11-05.md sections 2-4.
 *
 * Built as a SEPARATE JIT extension (estimator/first_return_operators.py) that only
 * READS a production IEKFLinearizationCache; the production _C / _C_vis modules are
 * never rebuilt or modified.
 *
 * One valid foreground pixel p (M = 1, V = 1, D in [z_near, z_far]) of one camera:
 *   modeled contributors i = 1..K in the tile's depth order, effective alpha_i from the
 *   production evalAlpha (same skip and 0.99 clamp), centre depth z_i = cache.depth;
 *   contributors with z outside [z_near, z_far] are not first-hit candidates.
 *   The FULL list is traversed (no T_STOP): a deeper contributor is never silently
 *   reclassified as unknown geometry.
 *
 *   log S_i^-  = -lambda (z_i - z_near) + sum_{k<i} log(1 - alpha_k)
 *   a_i        = log S_i^- + log alpha_i + log N(D; z_i, sigma_D^2)
 *   u_j        = sum_{k<j} log(1 - alpha_k) + log lambda - lambda (D - z_near)
 *                + lambda^2 s_U^2 / 2 + log[Phi(zh_j) - Phi(zl_j)]     (interval [b_{j-1}, b_j])
 *                zl = (b_{j-1} - D + lambda s_U^2) / s_U, zh = (b_j - D + lambda s_U^2) / s_U
 *   log L_p    = logsumexp(a_1..a_K, u_1..u_{K+1})
 *
 * sigma_D and s_U are evaluated at the OBSERVED depth D (fixed per pixel), so the
 * declared model has no variance derivative. Exact partials at frozen order:
 *   d logL / d alpha_k = gamma_k / alpha_k - (1 - prefix_k) / (1 - alpha_k)
 *   d logL / d z_k     = gamma_k (D - z_k) / sigma^2                      [match]
 *                        - lambda gamma_k + gU_k phi(zh_k)/(s dPhi_k)
 *                        - gU_{k+1} phi(zl_{k+1})/(s dPhi_{k+1})         [visibility]
 * prefix_k = posterior mass of every component up to and including contributor k.
 * The visibility part contains every alpha derivative; the match part only the
 * Gaussian depth density score.
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>

#include "auxiliary.h"
#include "config.h"
#include "iekf_device.cuh"

namespace cg = cooperative_groups;

namespace
{

struct FRCamera
{
	const uint2* ranges;
	const uint32_t* point_list;
	int W, H;
	const float* means3D;
	const float* mu_c;
	const float* u;
	const float* Jp;
	const float* sigma_c;
	const float* conic;
	const float* opacity;
	const float* depth;
	const int* clamp;
	const float* viewmatrix;
	const float* projmatrix;
	float focal_x, focal_y;
	// observation
	const float* D;
	const uint8_t* valid;
	const float* sig_d;
	const float* sig_u;
	float lam, z_near, z_far;
};

constexpr double LOG_SQRT_2PI = 0.91893853320467274178;

__device__ __forceinline__ double logPhi(double x)
{
	// log of the standard normal CDF, stable in both tails
	if (x < -1.0)
	{
		const double t = -x * M_SQRT1_2;
		return log(0.5 * erfcx(t)) - t * t;
	}
	return log1p(-0.5 * erfc(x * M_SQRT1_2));
}

__device__ __forceinline__ double logNormPdf(double x) { return -0.5 * x * x - LOG_SQRT_2PI; }

// log[Phi(b) - Phi(a)], a < b; reflected into the upper tail when a >= 0.
__device__ __forceinline__ double logDiffPhi(double a, double b)
{
	double hi, lo;
	if (a >= 0.0) { hi = logPhi(-a); lo = logPhi(-b); }
	else          { hi = logPhi(b);  lo = logPhi(a); }
	const double r = exp(lo - hi);
	return hi + log1p(-(r < 1.0 - 2e-16 ? r : 1.0 - 2e-16));
}

__device__ __forceinline__ void lseAdd(double& m, double& s, double v)
{
	if (v == -INFINITY) return;
	if (v <= m) { s += exp(v - m); return; }
	s = s * exp(m - v) + 1.0;
	m = v;
}

__device__ __forceinline__ bool pixelValid(const FRCamera& c, uint32_t pix, double& D)
{
	if (!c.valid[pix]) return false;
	D = (double)c.D[pix];
	return isfinite(D) && D >= c.z_near && D <= c.z_far && D > 0.0;
}

// Pass 1: log L of one pixel. Walks the tile list through shared ids.
template <typename LoadId>
__device__ double rayLogL(const FRCamera& c, float2 pixf, double D, double sd, double su,
                          int count, LoadId id_at)
{
	const double lam = c.lam, zn = c.z_near, zf = c.z_far;
	const double C0 = (lam > 0.0) ? (log(lam) - lam * (D - zn) + 0.5 * lam * lam * su * su) : -INFINITY;
	const double shift = lam * su * su;
	double logS = 0.0, prev = zn, m = -INFINITY, s = 0.0;
	for (int j = 0; j < count; ++j)
	{
		const int gid = id_at(j);
		if (gid < 0) continue;
		const double z = (double)c.depth[gid];
		if (!(z > zn && z <= zf)) continue;
		float alpha; float2 d; bool cl;
		if (!evalAlpha(c.u[2 * gid], c.u[2 * gid + 1], c.conic[3 * gid], c.conic[3 * gid + 1],
		               c.conic[3 * gid + 2], c.opacity[gid], pixf.x, pixf.y, alpha, d, cl))
			continue;
		if (lam > 0.0 && z > prev)
			lseAdd(m, s, logS + C0 + logDiffPhi((prev - D + shift) / su, (z - D + shift) / su));
		const double a = -lam * (z - zn) + logS + log((double)alpha) + logNormPdf((D - z) / sd) - log(sd);
		lseAdd(m, s, a);
		logS += log1p(-(double)alpha);
		prev = z;
	}
	if (lam > 0.0 && zf > prev)
		lseAdd(m, s, logS + C0 + logDiffPhi((prev - D + shift) / su, (zf - D + shift) / su));
	return (s > 0.0) ? m + log(s) : -INFINITY;
}

struct Contrib
{
	int gid;
	float alpha;
	float2 d;
	bool cl;
	double z;
	double pa;       // d logL / d alpha          (visibility)
	double pz_vis;   // d logL / d z, visibility part
	double pz_match; // d logL / d z, matched-depth part
	double gamma;    // posterior of this modeled hit
	double front;    // diagnostic: -log(1 - alpha Phi((D - z)/sigma_D))
};

// Pass 2: exact partials at frozen order. `emit(const Contrib&)` is called once per
// supported contributor, after its last partial is known. Returns sum of gamma_U.
template <typename LoadId, typename Emit>
__device__ double rayPartials(const FRCamera& c, float2 pixf, double D, double sd, double su,
                              double logL, int count, LoadId id_at, Emit emit)
{
	const double lam = c.lam, zn = c.z_near, zf = c.z_far;
	const double C0 = (lam > 0.0) ? (log(lam) - lam * (D - zn) + 0.5 * lam * lam * su * su) : -INFINITY;
	const double shift = lam * su * su;
	double logS = 0.0, prev = zn, prefix = 0.0, gU_tot = 0.0;
	bool has_prev = false;
	Contrib pend;
	for (int j = 0; j < count; ++j)
	{
		const int gid = id_at(j);
		if (gid < 0) continue;
		const double z = (double)c.depth[gid];
		if (!(z > zn && z <= zf)) continue;
		float alpha; float2 d; bool cl;
		if (!evalAlpha(c.u[2 * gid], c.u[2 * gid + 1], c.conic[3 * gid], c.conic[3 * gid + 1],
		               c.conic[3 * gid + 2], c.opacity[gid], pixf.x, pixf.y, alpha, d, cl))
			continue;
		double dstart = 0.0, dend = 0.0;
		if (lam > 0.0 && z > prev)
		{
			const double zl = (prev - D + shift) / su, zh = (z - D + shift) / su;
			const double ld = logDiffPhi(zl, zh);
			const double gU = exp(logS + C0 + ld - logL);
			dend = gU * exp(logNormPdf(zh) - ld) / su;
			dstart = -gU * exp(logNormPdf(zl) - ld) / su;
			prefix += gU;
			gU_tot += gU;
		}
		if (has_prev) { pend.pz_vis += dstart; emit(pend); }
		const double r = D - z;
		const double a = -lam * (z - zn) + logS + log((double)alpha) + logNormPdf(r / sd) - log(sd);
		const double g = exp(a - logL);
		prefix += g;
		double tail = 1.0 - prefix;
		if (tail < 0.0) tail = 0.0;
		pend.gid = gid; pend.alpha = alpha; pend.d = d; pend.cl = cl; pend.z = z;
		pend.gamma = g;
		pend.pa = g / (double)alpha - tail / (1.0 - (double)alpha);
		pend.pz_match = g * r / (sd * sd);
		pend.pz_vis = -lam * g + dend;
		{
			const double pf = 0.5 * erfc(-(r / sd) * M_SQRT1_2);       // Phi((D - z)/sigma)
			double t = 1.0 - (double)alpha * pf;
			pend.front = -log(t > 1e-300 ? t : 1e-300);
		}
		has_prev = true;
		logS += log1p(-(double)alpha);
		prev = z;
	}
	if (lam > 0.0 && zf > prev)
	{
		const double zl = (prev - D + shift) / su, zh = (zf - D + shift) / su;
		const double ld = logDiffPhi(zl, zh);
		const double gU = exp(logS + C0 + ld - logL);
		gU_tot += gU;
		if (has_prev) pend.pz_vis += -gU * exp(logNormPdf(zl) - ld) / su;
	}
	if (has_prev) emit(pend);
	return gU_tot;
}

// d alpha and d z of one contributor along a pose direction.
__device__ __forceinline__ void contribTangent(const FRCamera& c, const float R[9], const Contrib& k,
                                               float3 dmu, float3 dphi, float& dalpha, float& dz)
{
	float3 mu_c, conic, p_orig; float Jp[6], sigma_c[6];
	loadGaussian(k.gid, c.mu_c, c.Jp, c.sigma_c, c.conic, c.means3D, mu_c, Jp, sigma_c, conic, p_orig);
	float2 du; float3 dQ; float dd;
	poseTangent(dmu, dphi, mu_c, Jp, sigma_c, conic.x, conic.y, conic.z, R, c.projmatrix, p_orig,
	            c.W, c.H, c.focal_x, c.focal_y, c.clamp[k.gid], du, dQ, dd);
	dalpha = dAlphaOf(k.alpha, k.cl, k.d, conic.x, conic.y, conic.z, du, dQ);
	dz = dd;
}

// The 6 basis-direction tangents (shared by VJP and Fisher).
__device__ __forceinline__ void contribBasis(const FRCamera& c, const float R[9], const Contrib& k,
                                             float da[6], float dz[6])
{
	float3 mu_c, conic, p_orig; float Jp[6], sigma_c[6];
	loadGaussian(k.gid, c.mu_c, c.Jp, c.sigma_c, c.conic, c.means3D, mu_c, Jp, sigma_c, conic, p_orig);
	for (int b = 0; b < 6; ++b)
	{
		const float3 dmu = {b == 0 ? 1.f : 0.f, b == 1 ? 1.f : 0.f, b == 2 ? 1.f : 0.f};
		const float3 dphi = {b == 3 ? 1.f : 0.f, b == 4 ? 1.f : 0.f, b == 5 ? 1.f : 0.f};
		float2 du; float3 dQ; float dd;
		poseTangent(dmu, dphi, mu_c, Jp, sigma_c, conic.x, conic.y, conic.z, R, c.projmatrix, p_orig,
		            c.W, c.H, c.focal_x, c.focal_y, c.clamp[k.gid], du, dQ, dd);
		da[b] = dAlphaOf(k.alpha, k.cl, k.d, conic.x, conic.y, conic.z, du, dQ);
		dz[b] = dd;
	}
}

// ---------------------------------------------------------------------------
// Kernel: forward evaluation + per-pixel / per-Gaussian diagnostics
// mode 0: pass 1 only (log L); mode 1: pass 1 + pass 2 diagnostics
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
frEvalCUDA(FRCamera c, int mode,
           double* __restrict__ out_logL, float* __restrict__ out_gU, float* __restrict__ out_gmax,
           float* __restrict__ out_front, int* __restrict__ out_nsupp,
           float* __restrict__ w_match, float* __restrict__ gamma_sum, float* __restrict__ front_sum,
           float* __restrict__ pix_count)
{
	auto block = cg::this_thread_block();
	const uint32_t hb = (c.W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix = {block.group_index().x * BLOCK_X + block.thread_index().x,
	                   block.group_index().y * BLOCK_Y + block.thread_index().y};
	const bool inside = pix.x < (unsigned)c.W && pix.y < (unsigned)c.H;
	const uint32_t pix_id = c.W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const uint2 range = c.ranges[block.group_index().y * hb + block.group_index().x];
	const int total = (int)(range.y - range.x);
	double D = 0.0;
	const bool ok = inside && pixelValid(c, pix_id, D);
	const double sd = ok ? (double)c.sig_d[pix_id] : 1.0, su = ok ? (double)c.sig_u[pix_id] : 1.0;
	auto id_at = [&](int j) -> int { return (int)c.point_list[range.x + j]; };
	double logL = -INFINITY;
	if (ok) logL = rayLogL(c, pixf, D, sd, su, total, id_at);
	if (inside) out_logL[pix_id] = ok ? logL : 0.0;
	if (mode == 0 || !inside) return;
	if (!ok || !isfinite(logL))
	{
		out_gU[pix_id] = 0.f; out_gmax[pix_id] = 0.f; out_front[pix_id] = 0.f; out_nsupp[pix_id] = 0;
		return;
	}
	double gmax = 0.0, front = 0.0; int n = 0;
	auto emit = [&](const Contrib& k) {
		++n;
		if (k.gamma > gmax) gmax = k.gamma;
		front += k.front;
		atomicAdd(&w_match[k.gid], (float)(k.gamma / (sd * sd)));
		atomicAdd(&gamma_sum[k.gid], (float)k.gamma);
		atomicAdd(&front_sum[k.gid], (float)k.front);
		atomicAdd(&pix_count[k.gid], 1.0f);
	};
	const double gU = rayPartials(c, pixf, D, sd, su, logL, total, id_at, emit);
	out_gU[pix_id] = (float)gU;
	out_gmax[pix_id] = (float)gmax;
	out_front[pix_id] = (float)front;
	out_nsupp[pix_id] = n;
}

// ---------------------------------------------------------------------------
// Kernel: weighted VJP  sum_p w_p d logL_p / d pose   -> [P,6]
// parts: bit 1 = matched-depth part, bit 2 = visibility part
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
frVJPCUDA(FRCamera c, const double* __restrict__ logL_a, const float* __restrict__ wpix, int parts,
          float* __restrict__ out_adj)
{
	auto block = cg::this_thread_block();
	const uint32_t hb = (c.W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix = {block.group_index().x * BLOCK_X + block.thread_index().x,
	                   block.group_index().y * BLOCK_Y + block.thread_index().y};
	const bool inside = pix.x < (unsigned)c.W && pix.y < (unsigned)c.H;
	const uint32_t pix_id = c.W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const uint2 range = c.ranges[block.group_index().y * hb + block.group_index().x];
	const int total = (int)(range.y - range.x);
	double D = 0.0;
	if (!(inside && pixelValid(c, pix_id, D))) return;
	const double logL = logL_a[pix_id];
	const double w = (double)wpix[pix_id];
	if (!isfinite(logL) || w == 0.0) return;
	const double sd = c.sig_d[pix_id], su = c.sig_u[pix_id];
	float R[9]; load_Rcw(c.viewmatrix, R);
	const bool m = parts & 1, v = parts & 2;
	auto id_at = [&](int j) -> int { return (int)c.point_list[range.x + j]; };
	auto emit = [&](const Contrib& k) {
		const double pa = v ? k.pa : 0.0;
		const double pz = (m ? k.pz_match : 0.0) + (v ? k.pz_vis : 0.0);
		if (pa == 0.0 && pz == 0.0) return;
		float da[6], dz[6];
		contribBasis(c, R, k, da, dz);
		for (int b = 0; b < 6; ++b)
			atomicAdd(&out_adj[6 * k.gid + b], (float)(w * (pa * da[b] + pz * dz[b])));
	};
	rayPartials(c, pixf, D, sd, su, logL, total, id_at, emit);
}

// ---------------------------------------------------------------------------
// Kernel: JVP  d logL_p / d pose . dpose   -> [H,W]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
frJVPCUDA(FRCamera c, const double* __restrict__ logL_a, const float* __restrict__ dpose, int parts,
          float* __restrict__ out_jvp)
{
	auto block = cg::this_thread_block();
	const uint32_t hb = (c.W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix = {block.group_index().x * BLOCK_X + block.thread_index().x,
	                   block.group_index().y * BLOCK_Y + block.thread_index().y};
	const bool inside = pix.x < (unsigned)c.W && pix.y < (unsigned)c.H;
	const uint32_t pix_id = c.W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const uint2 range = c.ranges[block.group_index().y * hb + block.group_index().x];
	const int total = (int)(range.y - range.x);
	if (!inside) return;
	double D = 0.0;
	if (!pixelValid(c, pix_id, D)) { out_jvp[pix_id] = 0.f; return; }
	const double logL = logL_a[pix_id];
	if (!isfinite(logL)) { out_jvp[pix_id] = 0.f; return; }
	const double sd = c.sig_d[pix_id], su = c.sig_u[pix_id];
	float R[9]; load_Rcw(c.viewmatrix, R);
	const bool m = parts & 1, v = parts & 2;
	double acc = 0.0;
	auto id_at = [&](int j) -> int { return (int)c.point_list[range.x + j]; };
	auto emit = [&](const Contrib& k) {
		const double pa = v ? k.pa : 0.0;
		const double pz = (m ? k.pz_match : 0.0) + (v ? k.pz_vis : 0.0);
		if (pa == 0.0 && pz == 0.0) return;
		const float* q = dpose + 6 * k.gid;
		float da, dz;
		contribTangent(c, R, k, {q[0], q[1], q[2]}, {q[3], q[4], q[5]}, da, dz);
		acc += pa * da + pz * dz;
	};
	rayPartials(c, pixf, D, sd, su, logL, total, id_at, emit);
	out_jvp[pix_id] = (float)acc;
}

// ---------------------------------------------------------------------------
// Kernel: per-Gaussian 6x6 blocks of sum_p g_p g_p^T restricted to each Gaussian,
// g_p = visibility score of pixel p (empirical-Fisher PSD approximation). [P,21]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
frFisherVisCUDA(FRCamera c, const double* __restrict__ logL_a, float* __restrict__ out21)
{
	auto block = cg::this_thread_block();
	const uint32_t hb = (c.W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix = {block.group_index().x * BLOCK_X + block.thread_index().x,
	                   block.group_index().y * BLOCK_Y + block.thread_index().y};
	const bool inside = pix.x < (unsigned)c.W && pix.y < (unsigned)c.H;
	const uint32_t pix_id = c.W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const uint2 range = c.ranges[block.group_index().y * hb + block.group_index().x];
	const int total = (int)(range.y - range.x);
	double D = 0.0;
	if (!(inside && pixelValid(c, pix_id, D))) return;
	const double logL = logL_a[pix_id];
	if (!isfinite(logL)) return;
	const double sd = c.sig_d[pix_id], su = c.sig_u[pix_id];
	float R[9]; load_Rcw(c.viewmatrix, R);
	auto id_at = [&](int j) -> int { return (int)c.point_list[range.x + j]; };
	auto emit = [&](const Contrib& k) {
		if (k.pa == 0.0 && k.pz_vis == 0.0) return;
		float da[6], dz[6], g[6];
		contribBasis(c, R, k, da, dz);
		for (int b = 0; b < 6; ++b) g[b] = (float)(k.pa * da[b] + k.pz_vis * dz[b]);
		int t = 0;
		for (int a = 0; a < 6; ++a)
			for (int b = a; b < 6; ++b)
				atomicAdd(&out21[21 * k.gid + (t++)], g[a] * g[b]);
	};
	rayPartials(c, pixf, D, sd, su, logL, total, id_at, emit);
}

// ---------------------------------------------------------------------------
// Debug kernel: the FULL contributor list (no T_STOP) of every pixel, CSR.
// pass 0 counts, pass 1 fills. For tests / audits only.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
frDumpCUDA(FRCamera c, int pass, int* __restrict__ counts, const long* __restrict__ offsets,
           int* __restrict__ gids, float* __restrict__ alphas, float* __restrict__ zs)
{
	auto block = cg::this_thread_block();
	const uint32_t hb = (c.W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix = {block.group_index().x * BLOCK_X + block.thread_index().x,
	                   block.group_index().y * BLOCK_Y + block.thread_index().y};
	if (!(pix.x < (unsigned)c.W && pix.y < (unsigned)c.H)) return;
	const uint32_t pix_id = c.W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const uint2 range = c.ranges[block.group_index().y * hb + block.group_index().x];
	int n = 0;
	for (uint32_t j = range.x; j < range.y; ++j)
	{
		const int gid = (int)c.point_list[j];
		float alpha; float2 d; bool cl;
		if (!evalAlpha(c.u[2 * gid], c.u[2 * gid + 1], c.conic[3 * gid], c.conic[3 * gid + 1],
		               c.conic[3 * gid + 2], c.opacity[gid], pixf.x, pixf.y, alpha, d, cl))
			continue;
		if (pass == 1)
		{
			const long o = offsets[pix_id] + n;
			gids[o] = gid; alphas[o] = alpha; zs[o] = c.depth[gid];
		}
		++n;
	}
	if (pass == 0) counts[pix_id] = n;
}

void check(const torch::Tensor& t, torch::ScalarType st, const char* name)
{
	TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
	TORCH_CHECK(t.scalar_type() == st, name, " has the wrong dtype");
	TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

FRCamera makeCamera(
	const torch::Tensor& ranges, const torch::Tensor& point_list, int W, int H,
	const torch::Tensor& means3D, const torch::Tensor& mu_c, const torch::Tensor& u,
	const torch::Tensor& Jp, const torch::Tensor& sigma_c, const torch::Tensor& conic,
	const torch::Tensor& opacity, const torch::Tensor& depth, const torch::Tensor& clamp,
	const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix, float tanfovx, float tanfovy,
	const torch::Tensor& D, const torch::Tensor& valid, const torch::Tensor& sig_d, const torch::Tensor& sig_u,
	float lam, float z_near, float z_far)
{
	check(ranges, torch::kInt32, "ranges"); check(point_list, torch::kInt32, "point_list");
	for (auto* p : {&means3D, &mu_c, &u, &Jp, &sigma_c, &conic, &opacity, &depth, &viewmatrix, &projmatrix,
	                &D, &sig_d, &sig_u})
		check(*p, torch::kFloat32, "float input");
	check(clamp, torch::kInt32, "clamp_flags"); check(valid, torch::kUInt8, "valid");
	TORCH_CHECK(D.numel() == (int64_t)W * H && valid.numel() == D.numel() && sig_d.numel() == D.numel()
	            && sig_u.numel() == D.numel(), "observation maps must be [H,W]");
	TORCH_CHECK(lam >= 0.f && z_far > z_near && z_near > 0.f, "invalid first-return prior");
	FRCamera c;
	c.ranges = reinterpret_cast<const uint2*>(ranges.data_ptr<int>());
	c.point_list = reinterpret_cast<const uint32_t*>(point_list.data_ptr<int>());
	c.W = W; c.H = H;
	c.means3D = means3D.data_ptr<float>(); c.mu_c = mu_c.data_ptr<float>(); c.u = u.data_ptr<float>();
	c.Jp = Jp.data_ptr<float>(); c.sigma_c = sigma_c.data_ptr<float>(); c.conic = conic.data_ptr<float>();
	c.opacity = opacity.data_ptr<float>(); c.depth = depth.data_ptr<float>(); c.clamp = clamp.data_ptr<int>();
	c.viewmatrix = viewmatrix.data_ptr<float>(); c.projmatrix = projmatrix.data_ptr<float>();
	c.focal_x = W / (2.0f * tanfovx); c.focal_y = H / (2.0f * tanfovy);
	c.D = D.data_ptr<float>(); c.valid = valid.data_ptr<uint8_t>();
	c.sig_d = sig_d.data_ptr<float>(); c.sig_u = sig_u.data_ptr<float>();
	c.lam = lam; c.z_near = z_near; c.z_far = z_far;
	return c;
}

dim3 tiles(int W, int H) { return dim3((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1); }

} // namespace

#define FR_ARGS                                                                                          \
	const torch::Tensor& ranges, const torch::Tensor& point_list, int W, int H,                          \
	const torch::Tensor& means3D, const torch::Tensor& mu_c, const torch::Tensor& u,                     \
	const torch::Tensor& Jp, const torch::Tensor& sigma_c, const torch::Tensor& conic,                   \
	const torch::Tensor& opacity, const torch::Tensor& depth, const torch::Tensor& clamp,                \
	const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix, float tanfovx, float tanfovy,       \
	const torch::Tensor& D, const torch::Tensor& valid, const torch::Tensor& sig_d,                      \
	const torch::Tensor& sig_u, float lam, float z_near, float z_far
#define FR_CAM makeCamera(ranges, point_list, W, H, means3D, mu_c, u, Jp, sigma_c, conic, opacity, depth,  \
	                      clamp, viewmatrix, projmatrix, tanfovx, tanfovy, D, valid, sig_d, sig_u, lam,      \
	                      z_near, z_far)

std::vector<torch::Tensor> frEval(FR_ARGS, int mode)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(mu_c));
	FRCamera c = FR_CAM;
	const int P = mu_c.size(0);
	auto f = D.options();
	auto logL = torch::zeros({H, W}, f.dtype(torch::kFloat64));
	auto gU = torch::zeros({H, W}, f), gmax = torch::zeros({H, W}, f), front = torch::zeros({H, W}, f);
	auto nsupp = torch::zeros({H, W}, f.dtype(torch::kInt32));
	auto wm = torch::zeros({P}, f), gs = torch::zeros({P}, f), fs = torch::zeros({P}, f), pc = torch::zeros({P}, f);
	if (point_list.numel() > 0 || true)
		frEvalCUDA<<<tiles(W, H), dim3(BLOCK_X, BLOCK_Y, 1)>>>(
			c, mode, logL.data_ptr<double>(), gU.data_ptr<float>(), gmax.data_ptr<float>(),
			front.data_ptr<float>(), nsupp.data_ptr<int>(), wm.data_ptr<float>(), gs.data_ptr<float>(),
			fs.data_ptr<float>(), pc.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {logL, gU, gmax, front, nsupp, wm, gs, fs, pc};
}

torch::Tensor frVJP(FR_ARGS, const torch::Tensor& logL, const torch::Tensor& wpix, int parts)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(mu_c));
	FRCamera c = FR_CAM;
	check(logL, torch::kFloat64, "logL"); check(wpix, torch::kFloat32, "pixel weights");
	auto adj = torch::zeros({mu_c.size(0), 6}, D.options());
	frVJPCUDA<<<tiles(W, H), dim3(BLOCK_X, BLOCK_Y, 1)>>>(c, logL.data_ptr<double>(), wpix.data_ptr<float>(),
	                                                     parts, adj.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return adj;
}

torch::Tensor frJVP(FR_ARGS, const torch::Tensor& logL, const torch::Tensor& dpose, int parts)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(mu_c));
	FRCamera c = FR_CAM;
	check(logL, torch::kFloat64, "logL"); check(dpose, torch::kFloat32, "dpose");
	TORCH_CHECK(dpose.size(0) == mu_c.size(0) && dpose.size(1) == 6, "dpose must be [P,6]");
	auto out = torch::zeros({H, W}, D.options());
	frJVPCUDA<<<tiles(W, H), dim3(BLOCK_X, BLOCK_Y, 1)>>>(c, logL.data_ptr<double>(), dpose.data_ptr<float>(),
	                                                     parts, out.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return out;
}

torch::Tensor frFisherVis(FR_ARGS, const torch::Tensor& logL)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(mu_c));
	FRCamera c = FR_CAM;
	check(logL, torch::kFloat64, "logL");
	auto out = torch::zeros({mu_c.size(0), 21}, D.options());
	frFisherVisCUDA<<<tiles(W, H), dim3(BLOCK_X, BLOCK_Y, 1)>>>(c, logL.data_ptr<double>(), out.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return out;
}

std::vector<torch::Tensor> frDump(FR_ARGS)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(mu_c));
	FRCamera c = FR_CAM;
	auto counts = torch::zeros({H * W}, D.options().dtype(torch::kInt32));
	frDumpCUDA<<<tiles(W, H), dim3(BLOCK_X, BLOCK_Y, 1)>>>(c, 0, counts.data_ptr<int>(), nullptr, nullptr,
	                                                      nullptr, nullptr);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	auto offsets = torch::zeros({H * W + 1}, D.options().dtype(torch::kInt64));
	offsets.slice(0, 1).copy_(torch::cumsum(counts.to(torch::kInt64), 0));
	const int64_t M = offsets[H * W].item<int64_t>();
	auto gids = torch::zeros({M}, counts.options()), alphas = torch::zeros({M}, D.options()),
	     zs = torch::zeros({M}, D.options());
	frDumpCUDA<<<tiles(W, H), dim3(BLOCK_X, BLOCK_Y, 1)>>>(c, 1, counts.data_ptr<int>(),
	                                                      offsets.data_ptr<long>(), gids.data_ptr<int>(),
	                                                      alphas.data_ptr<float>(), zs.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {offsets, gids, alphas, zs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("fr_eval", &frEval);
	m.def("fr_vjp", &frVJP);
	m.def("fr_jvp", &frJVP);
	m.def("fr_fisher_vis", &frFisherVis);
	m.def("fr_dump", &frDump);
}
