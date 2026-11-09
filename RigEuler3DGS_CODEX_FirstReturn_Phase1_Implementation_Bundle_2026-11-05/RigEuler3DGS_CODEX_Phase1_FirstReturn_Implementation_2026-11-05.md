# CODEX IMPLEMENTATION CONTRACT — RigEuler3DGS Phase 1, first-return RGB-D measurements

**Date:** 2026-11-05  
**Repository:** <https://github.com/SamueleMara/RigEuler3DGS>  
**Inspected committed baseline:** `90dbf58bc94015167e074c83a67806c923d06552` (`main`)  
**Status:** implementation instructions, **not** a claim that the feature is already committed.  
**Companion tested CPU oracle:** `RigEuler3DGS_first_return_reference.py`; tests `test_RigEuler3DGS_first_return_reference.py` (6/6 pass locally with torch CPU).

---

## 0. Instruction to Codex — execute in this order

Implement an **opt-in, mathematically consistent, first-return depth measurement likelihood** for the existing Phase-1 iterated EKF. Finish it with a controlled real/synthetic A/B. Do **not** change the 12D per-Gaussian state, process model, fixed/canonical supernode construction, calibrated sensors, camera geometry, masks, point-cloud bootstrap, Phase 2 or Phase 3. Make changes surgical and commit one gate at a time. Always disclose the actual checkout hash, and check the current checkout against the interfaces below before editing.

### Nonnegotiable behavior

1. A *matched visible* moving Gaussian receives depth information from its own surface.
2. A *correctly occluded* Gaussian receives **no** fake depth observation from the occluder.
3. A valid *unknown/new* surface can explain a measurement without moving unrelated Gaussians.
4. A *false modeled foreground occluder* is **penalized**, rather than hidden by the unknown branch. Do not automatically interpret this penalty as a velocity error: geometry, opacity and pose are competing explanations.
5. RGB/depth sensor variance is the **variance** derived from calibrated standard deviation, e.g. `sigma_D^2(z)`; never replace it with depth median, arbitrary image-global noise inflation, or Gaussian rendering scale.
6. `M=1` is mandatory for observed RGB-D object measurements. **No background depth pulls on foreground Gaussians**. Negative silhouette evidence may use `M=0` without background depth.
7. All cameras in one frame share the **same** predicted state. All original frame timestamps, world/CV conventions, source-camera IDs and physical units remain unchanged.
8. Predicted Gaussians keep **one physical copy** and persistent identities. Unseen does not mean static; unexplained does not mean moving.
9. Trial-step acceptance uses the **full nonlinear first-return objective** recomputed at the *trial state*, with fresh contributor ordering, not the local quadratic or frozen association objective.
10. Baseline with feature disabled must reproduce the existing outputs/tests; no fake success on nonconvergence, and no silent covariance fallback.

The implementation is **not complete** after only changing `p_d` in `camera_terms`: a new per-contributor likelihood requires corresponding new gradient, Jacobian/vector products, local curvature, Fisher/preconditioner and posterior handling.

## 1. Repository locations verified against baseline

| Current component | Real committed function/interface | Action |
|---|---|---|
| `estimator/state.py` | `GaussianMotionState`: `mu,R,v,omega`, `boxplus` and 12D error | **Leave unchanged** |
| `estimator/process.py`, `estimator/covariance.py` | frame prediction, `P^- = F P^+ F^T + Q` | **Leave unchanged** |
| `estimator/renderer_bridge.py` | `build_linearization`, `CameraLinearization.cache`, `FrameLinearization` | Add opt-in new observation operator facade |
| `estimator/sparse_iekf.py` | `iekf_frame_step` at ~437; `_frame_terms` at ~247; `measurement_operators` at ~369; `evaluate_true_map_energy` at ~400 | Branch **only** new observation mode into identical outer iteration; keep old branch intact |
| `estimator/masked_observation.py` | `PreparedRGBD`, `phase1_camera_terms`, `add_cli`, `from_args` | Preserve hard-mask source; add separately selected mode |
| `estimator/observation_terms.py` | `camera_terms`, `robust_term`, `sensor_model.depth_variance` | Reuse noise values, not blended-depth JVP |
| `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py` | `IEKFLinearizationCache`, `iekf_build_linearization_cache`, `iekf_render`, `iekf_apply_H`, `iekf_apply_HT` | Add **new** opt-in first-return bindings, leave existing ones untouched |
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf.cu` | `renderFromCacheCUDA`, `poseTangent`, `evalAlpha` via `iekf_device.cuh` | Copy/reuse exact contributor and pose-derivative semantics; avoid editing legacy kernels |
| `submodules/diff-gaussian-rasterization/iekf_points.h`, `iekf_points.cu`, `ext.cpp`, `setup.py` | Existing bindings and CUDA build | Add new bindings/source without altering old APIs |
| `observation_diagnostics/renderer_access.py` | `pixel_contributions(cache)` CSR; has `gid,alpha,T,offsets` | Initial diagnostics/oracle only; don't allocate full CSR inside production PCG |
| `estimator/visibility_constraint.py` | Existing negative-ray constraint is **mutually exclusive** with masked observation | Do **not** combine it with the new masked model by accident |
| `tools/run_iekf_scene.py`, `tools/kpart_pipeline.py`, orchestrator | Phase-1 cfg/runners, external partition | Thread selection/config/manifests through **all** production Phase-1 entry points |

**Exact existing render:** `iekf_render(cache)` returns `(RGB, depth, alpha, transmittance, n_contrib)`. Its depth is **`Z/(A+eps_depth)`**, with `Z = sum_i T_i alpha_i z_i`. The forward loop has exact `evalAlpha` skip/clamp and **`T_STOP=1e-4`**; its backprop uses `n_contrib`. New kernels must be consistent about their own active list and stop semantics and must not silently treat omitted deeper Gaussians as if the physical renderer modeled them.

## 2. Core probabilistic model and units — implement exactly

At camera `c`, pixel `p` inside the **foreground mask** with valid metric axial depth `D`, build a ray from `z_near` to `z_far`. Let the active modeled Gaussian contributors in increasing camera-Z order be `i=1...K`. Each has

- `z_i(x)` [m]: camera axial centre depth from cached `cache.depth[gid]` at current state;
- `alpha_i(x)` in `(0,1)`: the **pixel effective opacity**, including projected Gaussian footprint, exactly as production `evalAlpha` (not intrinsic opacity alone);
- `sigma_i^2 = sigma_D,c^2(z_i) + sigma_repr,i,p^2` [m²], with `sigma_D,c^2` from `sensor_model.depth_variance(c,z)`.

`Sigma_render = R diag(scales²) Rᵀ` is the Gaussian *spatial footprint*. It is **not** `sigma_repr²` or the posterior `P`. Do not infer `sigma_repr²` from the cloud's median depth or silently copy an anisotropic scale into depth noise. Set the model-discrepancy term to zero until independently calibrated, or read an explicit source-backed model with a manifest.

Use a nonnegative, **independently declared/calibrated** prior hazard `lambda_U(z)` [m⁻¹] for unknown *object* surfaces. It cannot adapt per-pixel to fit current IEKF residuals. Start with piecewise-constant values; all constants must be stored with units, provenance and `ASSUMED` vs `CALIBRATED` status. **Never invent a default number of centimeters or an undocumented lambda.** Without a supplied/calibrated `lambda_U`, the new mode fails preflight with a clear error.

### 2.1 Ray survival and prior hit probabilities

For a modeled Gaussian at `z_i`, define survival before it:

\[
S(z)=\exp\!\left[-\int_{z_{near}}^z\lambda_U(u)du\right]\prod_{j:z_j<z}(1-\alpha_j).
\]

Then:

\[
q_i=S(z_i^-)\alpha_i,\qquad
f_U(z)=S(z)\lambda_U(z),\qquad
q_\varnothing=S(z_{far}).
\]

`q_i` is a **probability mass**; `f_U` is a **probability density per metre**; `q_empty` is no-hit probability. They must normalize:

\[
\boxed{\sum_{i=1}^K q_i+\int_{z_{near}}^{z_{far}}f_U(z)\,dz+q_\varnothing=1.}
\]

For a constant hazard within each interval, compute the probabilities by streaming front to back. If interval `[a,b]` begins with survival `S_a`, then

\[
m_U[a,b]=S_a\,\big(1-e^{-\lambda(b-a)}\big),\quad
S_{b^-}=S_a e^{-\lambda(b-a)},\quad
q_i=S_{z_i^-}\alpha_i,\quad
S_{z_i^+}=S_{z_i^-}(1-\alpha_i).
\]

The final survival at the far plane is `q_empty`. This O(K) recurrence is the **probability-normalization oracle**, independent of any depth image.

### 2.2 Valid observed depth likelihood

For a modeled hit:

\[
\phi_i(D)=\mathcal N(D;z_i,\sigma_i^2).
\]

For an unknown hit, integrate its depth with an independently specified normalized measurement density `phi_U(D|z)`. Start with Gaussian sensor depth noise with a specified `sigma_U` (or calibrated depth dependent function), then

\[
\boxed{L_p(D\mid x,V=1)=p_{det}\left[\sum_iq_i(x)\phi_i(D)+\int_{z_{near}}^{z_{far}}f_U(z\mid x)\phi_U(D\mid z)\,dz\right].}
\]

`p_det` can initially be fixed to 1 **for valid-return pose estimation**; its omission then adds only a constant on valid pixels. Do not pretend the valid-return branch integrates to one when `q_empty > 0`. If you later model invalid depth (`V=0`), supply separately calibrated dropout `p_drop`; with no spurious return and depth-independent detection:

\[
P(V=0\mid x)=q_\varnothing+p_{drop}(1-q_\varnothing),\qquad p_{det}=1-p_{drop}.
\]

**Phase-1 first release:** ignore invalid depth in the geometry likelihood (zero information) unless a calibrated detection model exists. Still compute `q_empty` for normalization diagnostics. Never substitute depth zero or read non-object/background depths.

### 2.3 Closed-form unknown-depth integral

For interval `[a,b]` with constant `lambda`, sensor SD `s_U`, input survival `S_a`, define

\[
I_U[a,b]=\int_a^b S_a\lambda e^{-\lambda(z-a)}\mathcal N(D;z,s_U^2)\,dz.
\]

For `lambda>0`:

\[
\boxed{
I_U[a,b]=S_a\lambda e^{-\lambda(D-a)+\frac12\lambda^2s_U^2}
\left[\Phi\!\left(\frac{b-D+\lambda s_U^2}{s_U}\right)
-\Phi\!\left(\frac{a-D+\lambda s_U^2}{s_U}\right)\right].
}
\]

For `lambda=0` this term is exactly 0. Sum interval integrals to obtain the unknown evidence. Implement the **log-domain** `logcdf_difference` with normal-tail reflection or `log_ndtr`, `log1p` and `logsumexp` to avoid cancellation. This is already implemented and numerically tested in the attached PyTorch reference. Provide a high-precision numerical quadrature oracle for extreme tails.

### 2.4 Posteriors for measured surface identity

Let `u_p` denote the integrated unknown-surface evidence, and `A_i=q_i phi_i`. Then

\[
L_p=\sum_iA_i+u_p,\quad
\boxed{\gamma_i=\frac{A_i}{L_p},\quad\gamma_U=\frac{u_p}{L_p},\quad\sum_i\gamma_i+\gamma_U=1.}
\]

**Do not equate** the 3DGS opacity `alpha` with a calibrated physical depth-hit probability without tests. This generative model is a proposed sensor/render coupling and needs held-out calibration. Record that limitation in the manifest and report posterior assignment calibration separately.

### 2.5 Crucial distinction: correct occlusion vs false occluder

- Observed door `D=0.75m`, modeled door at `0.75m`, cavity behind at `0.95m`: door has high posterior, cavity's *direct depth gradient* vanishes.
- Observed cavity `D=0.95m`, **wrong** modeled opaque Gaussian at `0.75m`: modeled-hit density does not explain `D`, and reaching a deeper unknown surface has survival factor `(1-alpha_front)`. Its negative-log contradiction is approximately `-log(1-alpha_front)` when modeled and unknown depths are well separated. This contradiction is **not discarded**.
- The contradiction alone cannot distinguish incorrect pose from incorrectly initialized canonical geometry/opacity. Log it per source Gaussian, camera and frame. **Never book it directly as a velocity pseudo-observation.**

No arbitrary `|D-z| > 2cm` eligibility gates. Near a depth discontinuity, if measured pixel registration/exposure noise is known, later convolve the whole likelihood with an independently calibrated pixel/exposure-footprint distribution. Do not simulate edge uncertainty by uniformly inflating `sigma_D`.

## 3. Exact objective and derivatives

At each frame `t`, for a candidate state `x` and fixed predicted state/prior `(x^-,P^-)`:

\[
\boxed{E_t(x)=\tfrac12\|x\boxminus x^-\|^2_{(P^-)^{-1}}-\sum_{c,p:M=1,V=1}\log L_{c,p}(D_{c,p}\mid x)+E_{mask}(x).}
\]

**Stage 1:** `E_mask` and old RGB/alpha stay *identical across the two A/B arms* to isolate the depth likelihood; run both the `depth-only` ablation and the same common RGB/alpha baseline. **Final integrated mode:** use one common latent surface identity for RGB+D; turn off any duplicate legacy foreground RGB/alpha term. Add negative silhouette rays using only mask evidence (never background depth), with explicit conditional accounting. The current expected-quad alpha cost is derived from legacy RGB precision; don't blindly retain that coupling when RGB changes meaning.

### 3.1 Exact score of valid depth likelihood

For component `i`, let `a_i=log q_i+log phi_i`; unknown evidence `u=∫f_U phi_U`. Then

\[
\nabla_x\log L_p=\sum_i\gamma_i\,\nabla_x a_i+\gamma_U\nabla_x\log u.
\]

With fixed contributor ordering, for `lambda_U=lambda` constant on the ray:

\[
\nabla\log q_i=\nabla\log\alpha_i-\sum_{j<i}\frac{\nabla\alpha_j}{1-\alpha_j}-\lambda\nabla z_i.
\]

For `R_i=\sigma_i^2` and Gaussian depth density:

\[
\nabla\log\phi_i
=\frac{D-z_i}{R_i}\nabla z_i+
\frac12\left(\frac{(D-z_i)^2}{R_i^2}-\frac1{R_i}\right)\nabla R_i.
\]

If `R_i` is explicitly held fixed in a local solve, omit its derivative **only from the local approximation**, then evaluate full variance-dependent `log L` at the nonlinear trial. Do not claim this local gradient is exact in the overall iterate unless the fixed-variance assumption is part of the declared model.

For an unknown interval, differentiate the **actual closed-form integral** (including interval boundaries `z_i`, front survival, and opacity). A constant `log_unknown` or stop-gradient unknown term would destroy the false-occluder gradient. Implement and test its derivative independently.

`z_i= e_3^T R_cw(mu_i-t_wc)` initially matches the currently committed `cache.depth[gid]`, with

\[
\frac{\partial z_i}{\partial\delta\mu_i}=e_3^TR_{cw}.
\]

Gaussian orientation influences pixel footprint/opacity through `poseTangent`. For current centre-depth model it does *not* directly affect `z_i`. Later ray-intersection surface depth is a **separate model change** requiring independent tests at oblique surfaces; don't mix it into this patch silently.

### 3.2 Information-vector sign and grouped solve

For `E_meas=-sum log L`, define the positive ascent score `b_meas=+sum grad(log L)`. At current iterate:

\[
 b=\operatorname{embedPose}(b_{meas})-\Lambda_{prior}(x\boxminus x^-),
\]

and solve approximately

\[
 (\Lambda_{prior}+B_{PSD}+\lambda_{LM}I)\delta x=b.
\]

`B_PSD` is a *declared positive-semidefinite local curvature approximation*, not the exact Hessian of a mixture. Retain the existing trust region/line search and **exact** trial-energy evaluation. Applying the canonical group map `G`:

\[
\boxed{G^T(\Lambda_{prior}+B_{PSD}+\lambda_{LM}I)G\,\delta g=G^Tb,\quad \delta x=G\delta g.}
\]

For matched depth, a justified local positive curvature is

\[
 B_{match}=\sum_{c,p,i}\gamma_{i,p}\,J_{z_i}^T R_i^{-1}J_{z_i}.
\]

For visibility, a simple **diagnostic/solver-only PSD** term is `B_vis= sum_p g_vis,p g_vis,pᵀ`, where `g_vis,p` is the score contribution from visibility/unknown depth (full `grad log L_p` minus the matched Gaussian depth scores). It is an empirical Fisher-like curvature **approximation**, not the observed Hessian or a calibrated posterior information matrix. This construction also preserves cross-Gaussian ray coupling when applied via JVP/VJP. Do not make `B_vis` an arbitrarily weighted penalty; test its numerical conditioning and effect through true-energy acceptance. If unsuitable, implement a tested PSD projection of the observed Hessian-vector product for the local solve; report the choice.

**Fatal implementation error:** calculate `b` from a per-contributor likelihood but `B` and `P_post` from the old blended-depth `iekf_build_fisher`. The estimator would be internally inconsistent.

### 3.3 Mixture covariance / missing association information

For component log scores `a_k` and posterior `gamma_k`, including the continuous unknown branch,

\[
 -\nabla^2\log L_p = -\mathbb E_\gamma[\nabla^2 a_k]-\operatorname{Cov}_\gamma(\nabla a_k).
\]

Thus complete-data Fisher `sum gamma Jᵀ R⁻¹ J` alone can **overclaim certainty**. The exact observed curvature may be indefinite. Before `first_return` mode is allowed as a production output, implement a tested, explicit posterior approximation based on the new likelihood, including a missing-association correction or conservatively justified PSD information estimate; retain the repository's stated per-Gaussian 12×12 marginal approximation, and verify PSD plus uncertainty coverage. **If not available in a stage, mark `POSTERIOR_UNCERTIFIED` and do not advertise a valid covariance to Phase 2.** Never double-count predicted state uncertainty by adding `J P^- Jᵀ` to both the correspondence likelihood and the IEKF measurement covariance.

### 3.4 No false velocities from persistent geometry errors

`E_free=-log T(D^-)` is **already** represented in the `unknown` survival term. Do not add it a second time to `E_t` unless using a separately derived, nonduplicated observation channel. Track the score of false front contributors separately; a Gaussian persistently in front of measured surfaces can cause a major likelihood contradiction with almost zero *local depth-translation* derivative if it stays in front. The response should be `GEOMETRY_OR_POSE_UNRESOLVED`, counterfactual full-renders, and a separate Phase-0 geometry repair if supported by stationary/multiview data; not a forced large Phase-1 motion or velocity tweak. Unsupported front contributions cannot be silently deleted in Phase 1 either.

## 4. Concrete file additions and API contracts

### 4.1 New files

```text
estimator/first_return_types.py
estimator/first_return_reference.py           # copy/adapt the delivered tested oracle
estimator/first_return_likelihood.py          # validation, priors, CPU/GPU adapter
estimator/first_return_association.py         # posterior + diagnostic aggregations
estimator/first_return_operators.py           # matrix-free score/local curvature API
estimator/first_return_observation.py         # mode selection and frame-level facade
observation_diagnostics/first_return_audit.py # read-only per-camera/point/group evidence
submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf_first_return.cu
submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf_first_return.h
# add C++ front wrappers to existing binding translation unit(s), extend setup.py

tests/test_first_return_probability.py
tests/test_first_return_gradients.py
tests/test_first_return_cuda_parity.py
tests/test_first_return_iekf.py
tests/test_first_return_occlusion.py
```

Keep `first_return_reference.py` independent of CUDA and restricted to small rays/scenes. Import it as the executable oracle in tests, not the production per-frame implementation.

### 4.2 Explicit configuration, fails closed

Implement the following opt-in configuration fields in `IEKFConfig`, the actual CLI parser(s), and saved frame/stage manifests:

```python
first_return_mode: str = "off"                  # off | audit | depth | rgbd
first_return_prior_path: str | None = None     # mandatory for audit/depth/rgbd
first_return_sigma_model_path: str | None = None
first_return_registration_path: str | None = None  # optional, independently calibrated
first_return_posterior: str = "unavailable"    # unavailable | observed_psd_validated
```

`--first-return-mode off|audit|depth|rgbd` is the intended user-facing option; add equivalent flags to all production pathways with exact pass-through. `off` must be 100% baseline. `audit` runs full first-return observation but does not change the motion update. `depth` changes depth likelihood, preserving old RGB/mask terms across the *comparative* A/B. `rgbd` eventually uses a jointly associated RGB-D observation with no duplicate RGB evidence. **Do not enable `rgbd` before source RGB/radiance and covariance models are verified**.

The source-backed prior JSON must include `schema`, `model_id`, `camera_names`, `z_near_m`, `z_far_m`, `unknown_hazard_segments_per_camera`, `unknown_depth_sigma_model_per_camera`, `provenance`, `calibration_or_assumed` and `units`. Reject impossible depth supports, negative hazards, missing cameras or schema mismatch *before running GPU*.

Example *schema illustration only*, **not valid numeric calibration**:

```json
{
  "schema": 1,
  "model_id": "REPLACE_WITH_HASH_OF_REAL_CALIBRATION",
  "calibration_or_assumed": "calibrated",
  "cameras": {
    "cam_000": {
      "z_near_m": "REPLACE_WITH_CAMERA_NEAR",
      "z_far_m": "REPLACE_WITH_CAMERA_FAR",
      "unknown_hazard_segments": "REPLACE_WITH_FITTED_SEGMENTS_IN_M_INV",
      "unknown_depth_sigma_model": "REPLACE_WITH_CALIBRATED_MODEL"
    }
  }
}
```

Implement an actual strict numeric schema, not those placeholders. Different cameras may have different priors. Keep camera-specific terms out of pooled mixture estimates unless physically justified.

### 4.3 Dataclasses / ownership

```python
@dataclass(frozen=True)
class FirstReturnObservation:
    camera_name: str
    depth_observed: torch.Tensor  # [H,W] m, valid values only
    depth_valid: torch.Tensor     # [H,W] bool AND foreground
    foreground: torch.Tensor      # [H,W] bool
    rgb_observed: torch.Tensor    # [3,H,W], original foreground RGB

@dataclass
class FirstReturnEvaluation:
    measurement_energy: float
    score_pose: torch.Tensor                 # [N,6], d(log L)/d(pose)
    normal_pose: Callable[[torch.Tensor],torch.Tensor] # [N,6] -> [N,6], PSD local approx
    fisher_pose: Callable[[],torch.Tensor]  # [N,6,6], diagonal blocks of same local approx
    diagnostics: dict
    posterior_status: str
    # cache must be tied to *exact* frame/state linearization, no stale contributors
```

Use `CameraSpec` and `GaussianMotionState` existing contracts; don't invent a new transform convention or independent `K` reconstruction. `FrameLinearization` can own an optional `first_return_evaluation`, created once for exactly that state and frame and re-created on every nonlinear trial.

### 4.4 CPU oracle — already written and tested

Use the attached `RigEuler3DGS_first_return_reference.py` as the baseline for one ray:

```python
first_return_ray(
    depth_observed: torch.Tensor,         # float64 scalar
    contributor_depths: torch.Tensor,    # [K], axial metres
    contributor_opacities: torch.Tensor, # [K], effective alpha
    sigma_contributors: torch.Tensor,    # [K], SD in metres
    *, z_near: float, z_far: float,
    lambda_unknown: float,               # m^-1
    sigma_unknown: float                 # metres
) -> RayResult
```

Returns `log_valid_depth_likelihood`, `gaussian_first_hit_mass`, `unknown_first_hit_mass`, `no_return_mass`, `gamma_gaussian`, `gamma_unknown`, and `log_unknown_evidence`. It supports gradients through `z`/`alpha` with Torch autograd. **It is an analytical single-ray oracle, not ready-to-run full scene code; do not replace CUDA or use Python loops per full-frame pixel in production.** Generalize it to piecewise hazard and per-camera noise only after the constant-hazard tests pass.

### 4.5 Additive CUDA API

Choose **exact C++/Python signatures** in the checkout; the following shapes and semantics are binding:

```python
# Frozen at *one* render/IEKF iteration; recomputed at every new nonlinear state.
fr_eval(cache: IEKFLinearizationCache,
        obs_depth: torch.Tensor,       # [H,W] float32 metric
        obs_valid: torch.Tensor,       # [H,W] bool (M AND raw depth valid)
        mask: torch.Tensor,            # [H,W] bool
        calibrated_sigma: torch.Tensor,# [H,W] or lambda-consistent per-hit sensor model
        prior: FirstReturnCameraPrior
       ) -> (logL: torch.Tensor,        # [H,W], valid foreground only
             gamma_U: torch.Tensor,    # [H,W] diagnostics
             front_contradiction: torch.Tensor, # [H,W] diagnostics
             aux: object)

fr_score_vjp(cache, obs, prior, frozen_aux) -> torch.Tensor      # [N,6] exact gradient logL
fr_score_jvp(cache, obs, prior, frozen_aux,
             dpose: torch.Tensor) -> torch.Tensor              # [H,W] d(logL)/dx dpose
fr_score_vjp_weighted(cache, obs, prior, frozen_aux,
                      pixel_adjoint: torch.Tensor) -> torch.Tensor # [N,6], adjoint
fr_local_normal(cache, obs, prior, frozen_aux,
                dpose: torch.Tensor) -> torch.Tensor            # [N,6] PSD local curvature
fr_local_fisher_blocks(cache, obs, prior, frozen_aux) -> torch.Tensor # [N,6,6]
```

`fr_score_vjp` may be `fr_score_vjp_weighted(..., ones_valid)`. The local curvature's `matched_depth` part requires per-contributor `dz_i/dpose`; the `visibility` part requires differentiating the modeled + unknown evidence through all preceding alpha/T and unknown integral terms. All JVP/VJP must be exact transposes of the **same frozen, piecewise-sorted** local operator, even where nonlinear ordering changes at a trial step.

**CUDA implementation strategy:**

1. Build on the actual `IEKFLinearizationCache`: `ranges`, `point_list`, `u`, `conic`, `opacity`, `depth`, `clamp_flags`, `mu_c`, `Jp`, `sigma_c`, `means3D`, view/projection, `image_height/width`.
2. For a pixel, stream contributors in the same front-to-back order, re-evaluate `evalAlpha`, do stable hazard/normal-integral recurrence, and compute log evidence. Track exact `gid`, camera axial `z_i`, projected effective alpha, and `T` in the oracle.
3. Implement per-ray reverse accumulation of `∂logL/∂alpha_i` and `∂logL/∂z_i`, including every unknown interval and its moving endpoints. Chain them with shared **`poseTangent` / `dAlphaOf`** on six pose basis directions (same wxyz/left-rotation semantics as existing `iekf.cu::vjpCUDA`). Add `atomicAdd` to `[N,6]` pose adjoint. No direct velocity derivative: the current frame has zero 6D velocity Jacobian; velocity changes through the 12D prior.
4. For JVP, compute per-Gaussian `dalpha_i, dz_i` from `poseTangent` and return the exact directional derivative of `logL` at frozen order. Compare with direct finite differences of the full objective away from ordering changes.
5. Implement PSD matrix-vector without a dense Jacobian: stream matched per-contributor depth terms plus the pixel visibility score outer-product term; apply `G`/`Gᵀ` in Python exactly as baseline. Compute per-Gaussian 6×6 local blocks with the **same approximation** as the matrix-vector for the preconditioner.
6. Do not allocate full `[H,W,K]` or all-camera dense Jacobian. The read-only CSR oracle may be heavy, but production PCG must stream/recompute frozen contributor information from the cache. **Measure peak GPU VRAM** and compare with baseline.
7. Respect the production rasterizer's precise `ALPHA_MIN`, `ALPHA_MAX`, clamping and skip tests. Do not call a JVP/VJP on contributions omitted from the nonlinear likelihood. For high opacity the legacy renderer can stop at `T_STOP`. For the new physical ray likelihood, either traverse the full modeled ray or implement an explicitly bounded tail approximation and prove its effect on `logL`, not only RGB alpha. A missing deep contributor must not silently be classified as unknown by a truncated list.
8. Save normalizer/aux buffers only while the corresponding linearization exists. On accepted and rejected trial renders, invalidate/recompute the correct caches. Enforce fixed order within PCG solve; redo sorting at next outer iteration.
9. Use stable accumulation (`float64` for probabilities / log-likelihood reductions; carefully justified float32 CUDA ops with parity checks). Avoid invalid `log(0)`, spurious NaNs, and gradient through invalid pixels or absent contributors.

## 5. Exactly how to integrate with `sparse_iekf.py`

### 5.1 New mode branch in nonlinear energy (`evaluate_true_map_energy` ~line 400)

**Existing baseline:** `build_linearization`, `_frame_terms`, `summarize(terms)`, legacy RGB+depth+alpha energy.

**Add first branch after new `build_linearization`:**

```python
if first_return_mode_is_enabled(observation_model):
    fr = observation_model.evaluate(lin, rgbd_gt, cfg)
    lin.first_return_evaluation = fr
    delta = boxminus(state, x_prior)
    prior = 0.5 * float((delta * apply_prior(Lambda_prior, delta)).sum())
    parts = {"prior": prior, "first_return_depth": fr.measurement_energy,
             "posterior_status": fr.posterior_status, **fr.diagnostics}
    return prior + fr.measurement_energy + fr.other_nonduplicated_terms, parts, lin
```

This code is *pseudocode* for the branch: implement actual types and return paths. Note `fr.other_nonduplicated_terms` includes only explicitly selected RGB/mask components. Do not call `_frame_terms` to calculate a second blended-depth objective on these pixels.

### 5.2 Mode branch in `_terms` / measurement operator (~lines 534–543 and 657)

Introduce `_measurement_ops(lin,...)` (or equivalent):

```python
if first_return_mode_is_enabled(observation_model):
    fr = lin.first_return_evaluation
    rhs_pose = fr.score_pose              # d(sum logL)/d(pose)
    normal_pose = fr.normal_pose          # PSD local approximation
    fisher_pose = fr.fisher_pose          # same local approximation
else:
    prec, alpha_terms = _terms(lin)
    rhs_pose, normal_pose, fisher_pose = measurement_operators(lin,prec,alpha_terms)
```

Existing lines 658–729 then continue with `rhs=embed_pose_to_full_state(rhs_pose)`, prior subtraction, preconditioner, optional grouped `G/Gᵀ`, PCG and true-energy trial evaluation **unchanged**. Cache and group frames must match the accepted state exactly.

### 5.3 Posterior at ~lines 993–1043

Stop the old `measurement_operators(lin,prec,alpha_terms)[2]()` posterior path from running in first-return mode. Instead:

- until tested new posterior is available, mark the new run `POSTERIOR_UNCERTIFIED`, and either run it in an explicit diagnostic/non-production mode or abort before producing misleading Phase-1 covariance artifacts;
- implement observed-information-aware, PSD-tested 6×6 marginal blocks and map them to the existing 12×12 covariance structure. Test stationary/hidden-direction coverage and cross-time position/velocity correlation;
- preserve existing Joseph/PSD and weak-direction policies without pretending `B_match` equals the true observed mixture information;
- dump actual curvature type and known block-independence approximation in the manifest.

**No silent use of old blended-render Fisher or the prior itself as if it were the posterior.**

### 5.4 Scene driver and checkpoints

Thread mode + prior hash across `tools/run_iekf_scene.py`, the Phase-1 runner in `tools/kpart_pipeline.py`, and any bidirectional/orchestrator scripts after locating the exact calls with `rg -n "iekf_frame_step\(|IEKFConfig\(|phase1_camera_terms\("`. Store in every run's manifest:

```text
scene, direction, original frame ids/times, selected endpoint,
GH commit + diff status, Gaussian model hash, camera calibration hash,
mask hash, noise-model hash, lambda_U model hash, first_return_mode,
representation-model origin, registration-model origin,
normalized ray count, unknown posterior stats, false-front contradictions,
per-camera/gaussian/group score norms, convergence state, posterior status
```

**Do not create or edit Phase-2/3 files** except a strict preflight that refuses to treat `POSTERIOR_UNCERTIFIED` as a calibrated covariance.

## 6. Staged commits / tests — STOP on failure

### Gate A — CPU probability oracle (no production edit)

- Copy `RigEuler3DGS_first_return_reference.py` to `estimator/first_return_reference.py` and `test_RigEuler3DGS_first_return_reference.py` to new tests.
- Extend tests with random rays, no contributors, coincident depths, a dense high-alpha ray, `lambda_U=0`, `lambda_U>0`, no return and per-camera near/far.
- Require `|sum(q_model)+mass_unknown+q_empty-1| < 1e-10` in float64 random tests; RGB-D valid likelihood density has units `m^-1` (and `RGB_density` when enabled).
- Independently compare closed-form unknown integral against SciPy high-precision quadrature. Check finite/autodiff gradients of each `z_i` and `alpha_i` and a multi-ray sum.
- Avoid fixed cm thresholds in inference; test numerical tolerances are allowed.
- **Commit A** only when all oracle tests pass.

### Gate B — renderer audit and camera parity (NO filter update change)

- Reuse `pixel_contributions(cache)` on 3 cameras; add `depth=cache.depth[gid]` and record `sum q`, `sum q*z`, RGB, `T_end`.
- Require parity with the actual renderer `(RGB,A,D)` within declared FP32 error tolerance for every tested pixel, including alpha saturated and overlapping Gaussian cases.
- Compare read-only first-return posteriors on fridge frames `119,133,145,160,190`, and a stationary frame within `80–99`. Report false-front and unknown explanations, their Gaussian IDs, per-camera distribution and depth-edge location.
- **Do not infer from centre-only mask checks that geometry is wrong**; require actual full-ray contributions.
- **Commit B.**

### Gate C — differentiable reference and CUDA derivative parity

- Implement analytic/auto-diff small-ray per-contributor score; explicitly include derivatives of unknown intervals and visibility through `T` and alpha.
- Add CUDA forward, JVP, VJP, PSD local normal and Fisher blocks; no dense Jacobian.
- Test `<Jv,w> = <v,Jᵀw>` with relative error < 1e-3 for FP32 CUDA (stricter CPU float64). Compare the total **true** negative-log-likelihood gradient to finite differences for translation, left rotation, overlapping same-depth and front/back cases away from sorting swaps.
- Check exact **zero effect** from mask-zero and invalid-depth pixels in the new observation. Verify original camera IDs/poses.
- Test `vᵀBv >= -epsilon` for random vectors and `sum_i score_i` parity between CPU oracle and GPU.
- If the new CUDA cache doesn't include a physical contributor that was in nonlinear evaluation, **FAIL**.
- **Commit C.**

### Gate D — IEKF update and covariance correctness

- Implement `off` and `audit` first; compare off's outputs bitwise/deterministically where possible with baseline, including all existing tests.
- Integrate `depth` opt-in into the true objective and RHS/normal/Fisher. Preserve original predicted prior, grouped motion and fixed camera synching.
- Test accepted trial steps only when actual full objective decreases. Test changes of contributor order between accepted iterations; no use of stale caches.
- Synthetic oracle: door 0.75 m, cavity 0.95 m. Matched door moves correctly, hidden cavity has no direct depth information; a false alpha=0.9 modeled foreground at 0.75 m reduces unknown-depth likelihood near 0.95 m by ≈0.1, **without** a compulsory invented velocity; unknown newly visible surface does not pull unrelated Gaussian.
- Test the covariance separately with association entropy high/low: must not report more precise than the justified observed information; per-Gaussian 12×12 PSD and no duplicate prior covariance in R. If posterior unavailable, **stop at diagnostic mode** and report blocker instead of claiming fully integrated production feature.
- **Commit D.**

### Gate E — calibrated joint RGB-D and mask evidence

- Add RGB conditional likelihood `p_I(I|hit_i)` with a separately calibrated/declared radiance-model covariance and same latent hit `s` as depth, rather than old fixed RGB quadratic on the conflicting surface.
- Unknown RGB likelihood must be proper; validate image color units `[0,1]` and sensor/appearance discrepancy separately.
- Remove duplicate foreground RGB/alpha evidence. Retain hard mask foreground eligibility and nonduplicated negative silhouette constraints. No `M=0` depth measurements.
- On simulated depth-ambiguous but color-distinguishable surfaces, association should improve and retain true door information; on unreliable/textureless RGB, should not force a wrong match.
- **Commit E.**

### Gate F — controlled full comparison, no unverified victory claim

Run same data, model initialization, calibrated cameras, mask, noise policy, frame timing, canonical partition, optimizer/IEKF settings and GPU budgets:

| Dataset/window | Mandatory output |
|---|---|
| Real fridge **80–99** static | median, p90, p99 displacement from verified static anchor; source-camera force decomposition; cumulative spurious velocity |
| Real fridge **100–105** | first-frame `dv/(dmu/dt)`, false-occluder IDs and free-space evidence, median/p99 drift |
| Real fridge closing **119–190** | per-camera visible/matched/occluded/unknown depth fractions, influence on moving-door Gaussians, full sequence motion and convergence |
| Synthetic `USB_100109`, 60 frames | moving-support precision/recall/F1, coherent support F1, selected joint type, axis angle/axis-line errors, true temporal q increment error |
| Synthetic two-layer + missing surface oracle | depth-match posterior calibration and zero spurious hidden-point update |

Arms: `legacy`, `audit-only`, `first-return depth + same RGB/alpha as legacy`, `first-return joint RGB-D` (only when Gate E passes). Also compare with the **fixed canonical partition**, not the older per-frame partition. Every arm must record convergence statuses; current baseline had `0/59` converged frames in USB tests. More improvement in median but an exploding p99 is **not** an acceptable win. Report GPU memory, runtime/frame and total iteration count.

Output an `AB_FIRST_RETURN_REPORT.md` with objective decomposition, held-out metrics, parity tests, failure cases, hash of every input/model and the exact branch commit. Do not change default mode unless the new method demonstrably improves both static and dynamic behavior and its covariance diagnostics are valid.

## 7. Implementation pitfalls Codex must explicitly guard

- **Wrong association:** comparing observed depth to `Z/(A+eps)` instead of contributors loses the entire improvement.
- **Wrong event model:** giving `unknown` arbitrary likelihood independent of preceding opacity lets false front surfaces survive cost-free.
- **Double counting:** using both per-layer depth and legacy depth, or both joint RGBD hit likelihood and old mask-background RGB quadratic, creates inconsistent information.
- **Wrong derivative:** detaching `q_i`, `T`, `lambda` interval endpoints or `gamma_U` from the gradient eliminates free-space contradiction.
- **Wrong posterior:** using EM complete-data information for posterior while latent surface identity remains ambiguous produces overconfidence.
- **Wrong layer scope:** `alpha_i` belongs to pixel `p`, not intrinsic opacity; camera axial `z` is not Euclidean range; hazard units follow the chosen depth coordinate.
- **Wrong cache:** drawing latent hits from CSR diagnostics but differentiating the old depth render gives a false adjoint identity.
- **Wrong order:** freezing contributor depth-order for an entire video instead of one local solve, or accepting trial from frozen order, suppresses occlusion transitions.
- **Wrong ray stop:** model omissions behind `T_STOP` are an approximation requiring a log-likelihood error bound, not exact unknown geometry.
- **Wrong fallback:** inability to explain an image pixel must *not* make a zero gradient against a strong false foreground occluder by construction.
- **Wrong motion semantics:** persistent foreground contradictions in a static interval should be investigated as canonical geometry defects; do not force a fake velocity by changing the process prior.
- **Wrong mask semantics:** original foreground masks are hard observation eligibility; background depth is never a source of object motion.
- **Wrong experimental comparison:** no switching canonical partition, camera calibration, sigma or direction between A/B arms. No tuning lambda_U based on the evaluated motion frames.

## 8. Deliverables expected from Codex

1. One clean feature branch with gated commits A–F; old baseline still operational.
2. Tested independent CPU reference and documented calibration/prior schema.
3. New CUDA and Torch wrappers with per-layer scoring, JVP/VJP, local curvature and Fisher parity.
4. Wired Phase-1 opt-in modes and complete provenance manifests; no stale cache or duplicate objective.
5. Numerically checked covariance logic or an explicit blocking `POSTERIOR_UNCERTIFIED` state.
6. Per-camera and per-Gaussian contradiction/association maps for real fridge, including frame 133.
7. Full reproducible A/B report, not only unit-test status; explicit conclusion on true motion retention, false static drift, tails, posterior coverage and runtime.

**Stop after each failing gate and fix the specific numerical/physical violation. Never hide missing implementation behind a fallback and never silently redesign the solved Phase 2/3 joint pipeline.**
