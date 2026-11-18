/*
 * Negative-ray (known non-object depth) likelihood for the Phase-1 IEKF (opt-in).
 * RigEuler3DGS_PHASE1_COHERENCE_CODEX_IMPLEMENTATION_SPEC_2026-11-16.md section I.A.
 *
 * Built as a SEPARATE JIT extension (estimator/negative_ray_operators.py) that only
 * READS a production IEKFLinearizationCache; the production _C / _C_vis modules are
 * never rebuilt or modified.
 *
 * A pixel p with status KNOWN_NONOBJECT_DEPTH (mask M = 0, valid raw sensor depth D)
 * says: no modeled object surface lies in front of the observed surface. With the
 * renderer's own contributors k = 1..n of p (tile depth order, production evalAlpha:
 * skip < 1/255, clamp 0.99, termination when T would fall below T_STOP = 1e-4, exactly
 * as renderFromCacheCUDA), centre depth z_k = cache.depth, blending weight
 * w_k = alpha_k prod_{j<k} (1 - alpha_j), and the true occluder depth Z ~ N(D, sigma_p^2)
 * (sigma_p = the calibrated sensor sd at the OBSERVED depth, fixed per pixel):
 *
 *   T_front(z)   = prod_{k : z_k < z} (1 - alpha_k)
 *   E_q[T_front] = 1 - A_p,     A_p = sum_k w_k s_k,     s_k = Phi((D - z_k) / sigma_p)
 *
 * (Abel summation of sum_k T_k [F(z_{k+1}) - F(z_k)] with the contributors in depth
 * order; exact for the Gaussian depth marginal, no quadrature.)  Each pixel carries a
 * reliability q_p in [0, 1] (estimator/negative_ray_reliability.py): the probability that
 * the ray is a clean non-object measurement; with probability 1 - q_p it is contaminated
 * (mask false negative / indistinguishable surface) and carries no information. The
 * per-pixel negative log-likelihood of that two-component observation model is
 *
 *   L_p = -log((1 - q_p) + q_p E[T_front]) = -log(1 - q_p A_p).
 *
 * q_p = 0 gives L_p = 0 and zero derivatives (UNKNOWN). Positivity: A_p <= sum_k w_k =
 * 1 - T_end <= 1 - T_STOP, so 1 - q_p A_p >= T_STOP > 0 by the renderer's own termination
 * rule; a 1e-12 floor guards round-off only and every use is counted.
 *
 * Exact partials at frozen contributor order (the 0.99 clamp has d alpha = 0, as in
 * dAlphaOf):
 *   dA/d alpha_k = T_{k-1} s_k - (sum_{m>k} w_m s_m) / (1 - alpha_k)
 *   dA/d z_k     = -w_k phi((D - z_k)/sigma_p) / sigma_p
 * chained to the per-Gaussian 6D pose through the shared iekf_device.cuh poseTangent /
 * dAlphaOf (the same derivative code as every other IEKF operator).
 *
 * Operators (per camera):
 *   nr_eval    A_p, L_p (float64) on status-2 pixels; optional per-Gaussian diagnostics
 *              over every status (supported / front / occluded / unknown weights)
 *   nr_vjp     sum_p c_p dA_p/d pose                                   [P,6]
 *   nr_jvp     dA_p/d pose . v                                         [H,W]
 *   nr_fisher  per-Gaussian 6x6 blocks of sum_p c_p g_{p,i} g_{p,i}^T, g = dA/dx_i  [P,21]
 * Accumulation is float atomicAdd, or (det = 1) two-pass int64 fixed point with scale
 * 2^floor(62 - log2(count * max|v|)) -- the scheme of estimator/deterministic_ops.py --
 * which is bitwise reproducible.
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cmath>
#include <cstring>

#include "auxiliary.h"
#include "config.h"
#include "iekf_device.cuh"

namespace cg = cooperative_groups;

namespace
{

constexpr uint8_t ST_SUPPORTED = 1;   // M = 1
constexpr uint8_t ST_NONOBJECT = 2;   // M = 0, valid raw depth, not on the mask edge
constexpr uint8_t ST_UNKNOWN = 3;     // M = 0, invalid depth or mask edge
constexpr double E_FLOOR = 1e-12;
constexpr double INV_SQRT_2PI = 0.39894228040143267794;

struct NRCamera
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
	const float* sig;
	const uint8_t* status;
	const float* q;      // per-pixel reliability q_p in [0, 1]
};

struct DetAcc
{
	int on;                     // 0: float atomics; 1: fixed point
	int pass;                   // 0: bound pass, 1: accumulate pass
	unsigned int* maxbits;
	unsigned long long* count;
	long long* acc;
	double scale;
};

__device__ __forceinline__ void accAdd(const DetAcc& det, float* buf, long long idx, double v)
{
	if (!det.on) { atomicAdd(&buf[idx], (float)v); return; }
	if (det.pass == 0)
	{
		atomicMax(det.maxbits, __float_as_uint(fabsf((float)v)));
		atomicAdd(det.count, 1ULL);
	}
	else
	{
		const long long q = llrint(v * det.scale);
		atomicAdd((unsigned long long*)&det.acc[idx], (unsigned long long)q);
	}
}

__device__ __forceinline__ double sOf(double D, double z, double sd) { return 0.5 * erfc(-((D - z) / sd) * M_SQRT1_2); }
__device__ __forceinline__ double phiOf(double u) { return INV_SQRT_2PI * exp(-0.5 * u * u); }

struct Contrib
{
	int gid;
	float alpha;
	float2 d;
	bool cl;
	double w;      // blending weight
	double s;      // Phi((D - z)/sigma)  (1 for the full-ray walk)
	double pa;     // dA/d alpha
	double pz;     // dA/d z
};

// The production contributor walk: same evalAlpha, same float transmittance test.
// visit(gid, alpha, d, cl, Td) for every composited contributor; Td is the float64
// transmittance BEFORE this contributor.
template <typename LoadId, typename Visit>
__device__ void walk(const NRCamera& c, float2 pixf, int count, LoadId id_at, Visit visit)
{
	float Tf = 1.0f;
	double Td = 1.0;
	for (int j = 0; j < count; ++j)
	{
		const int gid = id_at(j);
		float alpha; float2 d; bool cl;
		if (!evalAlpha(c.u[2 * gid], c.u[2 * gid + 1], c.conic[3 * gid], c.conic[3 * gid + 1],
		               c.conic[3 * gid + 2], c.opacity[gid], pixf.x, pixf.y, alpha, d, cl))
			continue;
		const float test_T = Tf * (1.f - alpha);
		if (test_T < T_STOP) break;
		visit(gid, alpha, d, cl, Td);
		Tf = test_T;
		Td *= (1.0 - (double)alpha);
	}
}

// Pass 1: A_p (soft front alpha).
template <typename LoadId>
__device__ double softFrontAlpha(const NRCamera& c, float2 pixf, double D, double sd, int count, LoadId id_at)
{
	double A = 0.0;
	walk(c, pixf, count, id_at, [&](int gid, float alpha, float2, bool, double Td) {
		A += Td * (double)alpha * sOf(D, (double)c.depth[gid], sd);
	});
	return A;
}

// Pass 2: exact partials of A_p at frozen order; emit(const Contrib&) per contributor.
template <typename LoadId, typename Emit>
__device__ void softFrontPartials(const NRCamera& c, float2 pixf, double D, double sd, double A, int count,
                                  LoadId id_at, Emit emit)
{
	double prefix = 0.0;
	walk(c, pixf, count, id_at, [&](int gid, float alpha, float2 d, bool cl, double Td) {
		const double z = (double)c.depth[gid];
		const double s = sOf(D, z, sd);
		const double w = Td * (double)alpha;
		prefix += w * s;
		double after = A - prefix;
		if (after < 0.0) after = 0.0;
		Contrib k;
		k.gid = gid; k.alpha = alpha; k.d = d; k.cl = cl; k.w = w; k.s = s;
		k.pa = Td * s - after / (1.0 - (double)alpha);
		k.pz = -w * phiOf((D - z) / sd) / sd;
		emit(k);
	});
}

__device__ __forceinline__ void contribTangent(const NRCamera& c, const float R[9], const Contrib& k,
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

__device__ __forceinline__ void contribGrad(const NRCamera& c, const float R[9], const Contrib& k, double g[6])
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
		const float da = dAlphaOf(k.alpha, k.cl, k.d, conic.x, conic.y, conic.z, du, dQ);
		g[b] = k.pa * (double)da + k.pz * (double)dd;
	}
}

#define NR_PIXEL_SETUP                                                                                   \
	auto block = cg::this_thread_block();                                                                \
	const uint32_t hb = (c.W + BLOCK_X - 1) / BLOCK_X;                                                   \
	const uint2 pix = {block.group_index().x * BLOCK_X + block.thread_index().x,                         \
	                   block.group_index().y * BLOCK_Y + block.thread_index().y};                        \
	const bool inside = pix.x < (unsigned)c.W && pix.y < (unsigned)c.H;                                  \
	const uint32_t pix_id = c.W * pix.y + pix.x;                                                         \
	const float2 pixf = {(float)pix.x, (float)pix.y};                                                    \
	const uint2 range = c.ranges[block.group_index().y * hb + block.group_index().x];                    \
	const int total = (int)(range.y - range.x);                                                          \
	auto id_at = [&](int j) -> int { return (int)c.point_list[range.x + j]; };

// ---------------------------------------------------------------------------
// nr_eval: A_p, L_p on status-2 pixels; diag = 1 adds per-Gaussian weights/counts
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
nrEvalCUDA(NRCamera c, int diag, double* __restrict__ out_A, double* __restrict__ out_L,
           int* __restrict__ floor_hits, float* __restrict__ pos_w, float* __restrict__ front_w,
           float* __restrict__ occl_w, float* __restrict__ unk_w, int* __restrict__ pos_n,
           int* __restrict__ neg_n, int* __restrict__ unk_n)
{
	NR_PIXEL_SETUP
	if (!inside) return;
	const uint8_t st = c.status[pix_id];
	out_A[pix_id] = 0.0;
	out_L[pix_id] = 0.0;
	if (st == ST_NONOBJECT)
	{
		const double D = (double)c.D[pix_id], sd = (double)c.sig[pix_id];
		const double A = softFrontAlpha(c, pixf, D, sd, total, id_at);
		double E = 1.0 - (double)c.q[pix_id] * A;
		if (E < E_FLOOR) { E = E_FLOOR; atomicAdd(floor_hits, 1); }
		out_A[pix_id] = A;
		out_L[pix_id] = -log(E);
		if (diag)
			walk(c, pixf, total, id_at, [&](int gid, float alpha, float2, bool, double Td) {
				const double w = Td * (double)alpha, s = sOf(D, (double)c.depth[gid], sd);
				atomicAdd(&front_w[gid], (float)(w * s));
				atomicAdd(&occl_w[gid], (float)(w * (1.0 - s)));
				atomicAdd(&neg_n[gid], 1);
			});
	}
	else if (diag && (st == ST_SUPPORTED || st == ST_UNKNOWN))
	{
		walk(c, pixf, total, id_at, [&](int gid, float alpha, float2, bool, double Td) {
			const double w = Td * (double)alpha;
			if (st == ST_SUPPORTED) { atomicAdd(&pos_w[gid], (float)w); atomicAdd(&pos_n[gid], 1); }
			else { atomicAdd(&unk_w[gid], (float)w); atomicAdd(&unk_n[gid], 1); }
		});
	}
}

// ---------------------------------------------------------------------------
// nr_vjp: sum_p c_p dA_p/d pose  -> [P,6]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
nrVJPCUDA(NRCamera c, const double* __restrict__ A_map, const float* __restrict__ wpix, DetAcc det,
          float* __restrict__ out)
{
	NR_PIXEL_SETUP
	if (!inside || c.status[pix_id] != ST_NONOBJECT) return;
	const double w = (double)wpix[pix_id];
	if (w == 0.0) return;
	const double D = (double)c.D[pix_id], sd = (double)c.sig[pix_id];
	float R[9]; load_Rcw(c.viewmatrix, R);
	softFrontPartials(c, pixf, D, sd, A_map[pix_id], total, id_at, [&](const Contrib& k) {
		if (k.pa == 0.0 && k.pz == 0.0) return;
		double g[6];
		contribGrad(c, R, k, g);
		for (int b = 0; b < 6; ++b) accAdd(det, out, 6LL * k.gid + b, w * g[b]);
	});
}

// ---------------------------------------------------------------------------
// nr_jvp: dA_p/d pose . v  -> [H,W]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
nrJVPCUDA(NRCamera c, const double* __restrict__ A_map, const float* __restrict__ dpose, float* __restrict__ out)
{
	NR_PIXEL_SETUP
	if (!inside) return;
	if (c.status[pix_id] != ST_NONOBJECT) { out[pix_id] = 0.f; return; }
	const double D = (double)c.D[pix_id], sd = (double)c.sig[pix_id];
	float R[9]; load_Rcw(c.viewmatrix, R);
	double acc = 0.0;
	softFrontPartials(c, pixf, D, sd, A_map[pix_id], total, id_at, [&](const Contrib& k) {
		if (k.pa == 0.0 && k.pz == 0.0) return;
		const float* q = dpose + 6 * k.gid;
		float da, dz;
		contribTangent(c, R, k, {q[0], q[1], q[2]}, {q[3], q[4], q[5]}, da, dz);
		acc += k.pa * (double)da + k.pz * (double)dz;
	});
	out[pix_id] = (float)acc;
}

// ---------------------------------------------------------------------------
// nr_fisher: per-Gaussian 6x6 blocks of sum_p c_p g g^T  -> [P,21]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
nrFisherCUDA(NRCamera c, const double* __restrict__ A_map, const float* __restrict__ wpix, DetAcc det,
             float* __restrict__ out21)
{
	NR_PIXEL_SETUP
	if (!inside || c.status[pix_id] != ST_NONOBJECT) return;
	const double w = (double)wpix[pix_id];
	if (w == 0.0) return;
	const double D = (double)c.D[pix_id], sd = (double)c.sig[pix_id];
	float R[9]; load_Rcw(c.viewmatrix, R);
	softFrontPartials(c, pixf, D, sd, A_map[pix_id], total, id_at, [&](const Contrib& k) {
		if (k.pa == 0.0 && k.pz == 0.0) return;
		double g[6];
		contribGrad(c, R, k, g);
		int t = 0;
		for (int a = 0; a < 6; ++a)
			for (int b = a; b < 6; ++b)
				accAdd(det, out21, 21LL * k.gid + (t++), w * g[a] * g[b]);
	});
}

void check(const torch::Tensor& t, torch::ScalarType st, const char* name)
{
	TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
	TORCH_CHECK(t.scalar_type() == st, name, " has the wrong dtype");
	TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

NRCamera makeCamera(
	const torch::Tensor& ranges, const torch::Tensor& point_list, int W, int H,
	const torch::Tensor& means3D, const torch::Tensor& mu_c, const torch::Tensor& u,
	const torch::Tensor& Jp, const torch::Tensor& sigma_c, const torch::Tensor& conic,
	const torch::Tensor& opacity, const torch::Tensor& depth, const torch::Tensor& clamp,
	const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix, float tanfovx, float tanfovy,
	const torch::Tensor& D, const torch::Tensor& sig, const torch::Tensor& status, const torch::Tensor& q)
{
	check(ranges, torch::kInt32, "ranges"); check(point_list, torch::kInt32, "point_list");
	for (auto* p : {&means3D, &mu_c, &u, &Jp, &sigma_c, &conic, &opacity, &depth, &viewmatrix, &projmatrix, &D, &sig})
		check(*p, torch::kFloat32, "float input");
	check(clamp, torch::kInt32, "clamp_flags"); check(status, torch::kUInt8, "status");
	check(q, torch::kFloat32, "reliability q");
	TORCH_CHECK(D.numel() == (int64_t)W * H && sig.numel() == D.numel() && status.numel() == D.numel()
	            && q.numel() == D.numel(), "observation maps must be [H,W]");
	TORCH_CHECK(q.min().item<float>() >= 0.f && q.max().item<float>() <= 1.f, "reliability q must be in [0, 1]");
	NRCamera c;
	c.ranges = reinterpret_cast<const uint2*>(ranges.data_ptr<int>());
	c.point_list = reinterpret_cast<const uint32_t*>(point_list.data_ptr<int>());
	c.W = W; c.H = H;
	c.means3D = means3D.data_ptr<float>(); c.mu_c = mu_c.data_ptr<float>(); c.u = u.data_ptr<float>();
	c.Jp = Jp.data_ptr<float>(); c.sigma_c = sigma_c.data_ptr<float>(); c.conic = conic.data_ptr<float>();
	c.opacity = opacity.data_ptr<float>(); c.depth = depth.data_ptr<float>(); c.clamp = clamp.data_ptr<int>();
	c.viewmatrix = viewmatrix.data_ptr<float>(); c.projmatrix = projmatrix.data_ptr<float>();
	c.focal_x = W / (2.0f * tanfovx); c.focal_y = H / (2.0f * tanfovy);
	c.D = D.data_ptr<float>(); c.sig = sig.data_ptr<float>(); c.status = status.data_ptr<uint8_t>();
	c.q = q.data_ptr<float>();
	return c;
}

dim3 tiles(int W, int H) { return dim3((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1); }

// Runs `launch(det)` once (float) or twice (fixed point) and returns the [n] float result.
template <typename Launch>
torch::Tensor accumulate(int64_t n, bool deterministic, const torch::TensorOptions& f, Launch launch)
{
	auto out = torch::zeros({n}, f);
	DetAcc det{0, 0, nullptr, nullptr, nullptr, 0.0};
	if (!deterministic)
	{
		launch(det, out.data_ptr<float>());
		C10_CUDA_KERNEL_LAUNCH_CHECK();
		return out;
	}
	auto bounds = torch::zeros({2}, f.dtype(torch::kInt64));            // [maxbits, count]
	auto acc = torch::zeros({n}, f.dtype(torch::kInt64));
	det.on = 1; det.pass = 0;
	det.maxbits = reinterpret_cast<unsigned int*>(bounds.data_ptr<int64_t>());
	det.count = reinterpret_cast<unsigned long long*>(bounds.data_ptr<int64_t>() + 1);
	det.acc = reinterpret_cast<long long*>(acc.data_ptr<int64_t>());
	launch(det, out.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	auto hb = bounds.cpu();
	const int64_t bits = hb[0].item<int64_t>() & 0xffffffffLL;
	const uint64_t cnt = (uint64_t)hb[1].item<int64_t>();
	float mx; const uint32_t b32 = (uint32_t)bits; memcpy(&mx, &b32, 4);
	if (cnt == 0 || !(mx > 0.f)) return out;                            // nothing to add
	TORCH_CHECK(std::isfinite(mx), "non-finite term in a deterministic negative-ray reduction");
	const double bound = (double)cnt * (double)mx;
	det.scale = std::ldexp(1.0, (int)std::floor(62.0 - std::log2(bound)));
	det.pass = 1;
	launch(det, out.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return (acc.to(torch::kFloat64) / det.scale).to(torch::kFloat32);
}

} // namespace

#define NR_ARGS                                                                                          \
	const torch::Tensor& ranges, const torch::Tensor& point_list, int W, int H,                          \
	const torch::Tensor& means3D, const torch::Tensor& mu_c, const torch::Tensor& u,                     \
	const torch::Tensor& Jp, const torch::Tensor& sigma_c, const torch::Tensor& conic,                   \
	const torch::Tensor& opacity, const torch::Tensor& depth, const torch::Tensor& clamp,                \
	const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix, float tanfovx, float tanfovy,       \
	const torch::Tensor& D, const torch::Tensor& sig, const torch::Tensor& status, const torch::Tensor& q
#define NR_CAM makeCamera(ranges, point_list, W, H, means3D, mu_c, u, Jp, sigma_c, conic, opacity, depth,  \
	                      clamp, viewmatrix, projmatrix, tanfovx, tanfovy, D, sig, status, q)

std::vector<torch::Tensor> nrEval(NR_ARGS, int diag)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(mu_c));
	NRCamera c = NR_CAM;
	const int P = mu_c.size(0);
	auto f = D.options();
	auto A = torch::zeros({H, W}, f.dtype(torch::kFloat64)), L = torch::zeros({H, W}, f.dtype(torch::kFloat64));
	auto fh = torch::zeros({1}, f.dtype(torch::kInt32));
	auto pw = torch::zeros({diag ? P : 1}, f), fw = torch::zeros({diag ? P : 1}, f),
	     ow = torch::zeros({diag ? P : 1}, f), uw = torch::zeros({diag ? P : 1}, f);
	auto pn = torch::zeros({diag ? P : 1}, f.dtype(torch::kInt32)), nn = torch::zeros({diag ? P : 1}, f.dtype(torch::kInt32)),
	     un = torch::zeros({diag ? P : 1}, f.dtype(torch::kInt32));
	nrEvalCUDA<<<tiles(W, H), dim3(BLOCK_X, BLOCK_Y, 1)>>>(
		c, diag, A.data_ptr<double>(), L.data_ptr<double>(), fh.data_ptr<int>(), pw.data_ptr<float>(),
		fw.data_ptr<float>(), ow.data_ptr<float>(), uw.data_ptr<float>(), pn.data_ptr<int>(), nn.data_ptr<int>(),
		un.data_ptr<int>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {A, L, fh, pw, fw, ow, uw, pn, nn, un};
}

torch::Tensor nrVJP(NR_ARGS, const torch::Tensor& A_map, const torch::Tensor& wpix, bool deterministic)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(mu_c));
	NRCamera c = NR_CAM;
	check(A_map, torch::kFloat64, "A"); check(wpix, torch::kFloat32, "pixel weights");
	const int64_t P = mu_c.size(0);
	auto out = accumulate(6 * P, deterministic, D.options(), [&](DetAcc det, float* buf) {
		nrVJPCUDA<<<tiles(W, H), dim3(BLOCK_X, BLOCK_Y, 1)>>>(c, A_map.data_ptr<double>(), wpix.data_ptr<float>(),
		                                                     det, buf);
	});
	return out.reshape({P, 6});
}

torch::Tensor nrJVP(NR_ARGS, const torch::Tensor& A_map, const torch::Tensor& dpose)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(mu_c));
	NRCamera c = NR_CAM;
	check(A_map, torch::kFloat64, "A"); check(dpose, torch::kFloat32, "dpose");
	TORCH_CHECK(dpose.size(0) == mu_c.size(0) && dpose.size(1) == 6, "dpose must be [P,6]");
	auto out = torch::zeros({H, W}, D.options());
	nrJVPCUDA<<<tiles(W, H), dim3(BLOCK_X, BLOCK_Y, 1)>>>(c, A_map.data_ptr<double>(), dpose.data_ptr<float>(),
	                                                     out.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return out;
}

torch::Tensor nrFisher(NR_ARGS, const torch::Tensor& A_map, const torch::Tensor& wpix, bool deterministic)
{
	const at::cuda::OptionalCUDAGuard guard(device_of(mu_c));
	NRCamera c = NR_CAM;
	check(A_map, torch::kFloat64, "A"); check(wpix, torch::kFloat32, "pixel weights");
	const int64_t P = mu_c.size(0);
	auto out = accumulate(21 * P, deterministic, D.options(), [&](DetAcc det, float* buf) {
		nrFisherCUDA<<<tiles(W, H), dim3(BLOCK_X, BLOCK_Y, 1)>>>(c, A_map.data_ptr<double>(), wpix.data_ptr<float>(),
		                                                        det, buf);
	});
	return out.reshape({P, 21});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("nr_eval", &nrEval);
	m.def("nr_vjp", &nrVJP);
	m.def("nr_jvp", &nrJVP);
	m.def("nr_fisher", &nrFisher);
}
