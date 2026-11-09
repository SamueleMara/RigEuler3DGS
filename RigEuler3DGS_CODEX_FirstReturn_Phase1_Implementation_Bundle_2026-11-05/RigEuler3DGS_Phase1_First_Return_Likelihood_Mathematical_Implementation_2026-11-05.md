# RigEuler3DGS Phase 1 — mathematically specified first-return, visibility-aware RGB-D IEKF

Date: 2026-11-05. Target: `SamueleMara/RigEuler3DGS`, main commit `90dbf58bc94015167e074c83a67806c923d06552`. This is a proposed design, **not** a statement that the repository already implements it.

## 0. Core requirements and non-goals

The proposal replaces Phase-1's *measurement likelihood and its matrix-free derivatives*, not its 12D Gaussian motion state, motion prior, canonical supernode partition, renderer's physical Gaussian population, or Phase 2–3 motion/joint stages. No scene-specific rules, centimeter gates, motion-specific hyperparameter tuning, global sensor-SD inflation, invented background-depth pulls, loss-only modifications, or resetting/overwriting frozen geometry in the IEKF. The binary foreground mask is a hard observation-eligibility constraint. All three synchronized cameras observe the same world state. The genuine nonlinear likelihood—not the local IRLS surrogate—must be evaluated on trial steps. Keep current behavior selectable as an A/B baseline.

The problem: depth and RGB at a fixed pixel may switch from a cavity surface at `t` to an occluding door surface at `t+1`. Current alpha-weighted *single* rendered depth can lie between physical surfaces; its residual is then falsely interpreted as a motion correction. Conversely, when the model predicts a wrong opaque foreground Gaussian but observation sees a deeper surface, simply declaring `unknown` should NOT make the contradiction cost-free.

## 1. State and prediction: retain exact current contracts

For each Gaussian `i`, at frame `t`:

\[
 x_{i,t}=(\mu_{i,t}\in\mathbb R^3,\ R_{i,t}\in SO(3),\ v_{i,t}\in\mathbb R^3,\ \omega_{i,t}\in\mathbb R^3),\qquad \delta x_i=(\delta\mu_i,\delta\phi_i,\delta v_i,\delta\omega_i)\in\mathbb R^{12}.
\]

Frozen during Phase 1: color/radiance `c_i`, intrinsic opacity `o_i`, principal rendering scales `s_i`. Render covariance `Sigma_render,i = R_i diag(s_i^2) R_i^T` is the *surface footprint*, NOT the uncertainty of the estimated Gaussian centre or the sensor variance. Predict once per frame with the current `estimator/process.py`; covariance: `P_t^- = F_t P_{t-1}^+ F_t^T + Q_t`, respecting the repository's per-Gaussian block approximation. Retain the fixed prior inside every iterated update. Left retraction as in `estimator/state.py`:

\[
 \mu^+=\mu+\delta\mu,\quad R^+=\mathrm{Exp}(\widehat{\delta\phi})R,\quad v^+=v+\delta v,\quad \omega^+=\omega+\delta\omega.
\]

The fixed canonical grouping is a change of variables `delta x = G delta z`; all camera and pixel Jacobians accumulate in the *same* global normal system as `G^T H^T W H G`. No averaging different cameras' depths and no independent separate group corrections.

## 2. Per-pixel depth-ordered renderer contributors

For camera `c`, pixel `p` with calibrated ray and one predicted state `x`, let the exact rasterizer's effective contributor opacity be `alpha_{i,c,p}(x)`, with projected depth `z_{i,c,p}(x)`. Maintain the *same ordering, alpha clamp, skip and termination* as the production rasterizer. The depth of a Gaussian centre can initially be `z_i=e_3^T R_cw (mu_i-t_wc)`, but this centre-depth approximation must be validated at oblique depth boundaries; do not silently identify the whole 3D Gaussian surface with a point.

The existing diagnostic kernel `observation_diagnostics/renderer_access.py::pixel_contributions(cache)` already provides exactly the composited source Gaussian IDs, `alpha`, `T_before`, `q=T_before alpha`; extend it to expose depths and optionally other differentiable properties, and test parity with production `A`/`Z`/`RGB`. For efficient production measurement derivatives, make a fused cache-aware CUDA JVP/VJP; do not allocate all CSR contributions for every full image at every IEKF iteration.

The conventional blended depth is:

\[
 \hat D_p=\frac{\sum_i q_{i,p}z_{i,p}}{A_p+\epsilon},\qquad A_p=\sum_i q_{i,p}, \qquad q_{i,p}=\alpha_{i,p}\prod_{j<i}(1-\alpha_{j,p}).
\]

**This is not the depth residual to use for physical correspondence**, because a 0.75 m door and a 0.95 m cavity can blend to a fictitious 0.81 m depth.

## 3. Proper normalized generative first-return model

Consider metric camera axial depth `z in [z_near,z_far]` along one valid foreground ray. Introduce:

- **Modeled termination candidates**: contributor `i` terminates the ray with probability `alpha_i` if the ray survives to its depth `z_i`.
- **Unmodeled-surface hazard**: fixed nonnegative `lambda_U(z|c,p,scene_prior)` (units `m^-1`), with cumulative hazard `Lambda_U(z)=int_{z_near}^{z} lambda_U(s)ds`. This encodes plausible unmodeled *object* geometry, is not a free likelihood per pixel, and is not tuned using the target motion trajectory. A constant or piecewise-constant calibrated prior is sufficient to start; later use temporally persistent 3D unknown geometry. 
- **No return**: possibility that no surface terminates the ray before `z_far`; when actual depth is invalid, use a calibrated missing-return/detection model rather than fabricating a depth.

Define survival before depth `z`:

\[
 S_p(z|x)=e^{-\Lambda_U(z)}\prod_{i:z_i<z}(1-\alpha_{i,p}(x)).
\]

Then the probability **mass** for a termination at modeled Gaussian `i`, unknown termination **density**, and no-return mass are

\[
 \boxed{q_{i,p}(x)=S_p(z_i^-|x)\alpha_{i,p}(x)},\qquad
 \boxed{f_{U,p}(z|x)=S_p(z|x)\lambda_U(z)},\qquad
 \boxed{q_{\emptyset,p}(x)=S_p(z_{far}|x)}.
\]

Exactly (disregarding physically coincident depths, which need a consistent order):

\[
 \sum_i q_{i,p}+\int_{z_{near}}^{z_{far}} f_{U,p}(z)\,dz+q_{\emptyset,p}=1.
\]

This is an actual first-hit random-variable distribution, unlike an unnormalized ad-hoc mixture of per-Gaussian depth residuals plus an arbitrary `unknown` term. If appearance and depth sensor detection need more fidelity, include detection probability `p(V|s,z)`. **Do not optimize the unknown hazard separately on the pixels used to evaluate it**: it could otherwise explain every discrepancy.

## 4. Calibrated measurement channel and full RGB-D likelihood

Let `V_cp` denote valid measured depth and `M_cp` the binary object mask; only `M_cp=1` pixels enter observed foreground RGB/depth channel. The source measurement at `t+1` is `D_cp>0` if valid and observed foreground RGB `I_cp`.

For a modeled hit `i`, use normalized likelihood

\[
 \ell_{i,cp}(D,I\mid x)=\mathcal N(D;z_{i,cp}(x),R_{D,i,cp})\,p_I(I\mid i,c,p,x),
\]

where `R_D= sigma_{sensor,c}^2(z) + sigma_{representation,i,cp}^2`. The sensor term comes from `sensor_model.depth_variance`; model mismatch is independently estimated and held fixed during a trial. A per-surface likelihood `p_I` may be a calibrated broad Gaussian/Student-t around the Gaussian's view-dependent radiance, but must use **original raw foreground RGB** with an appropriate radiance-model discrepancy; do not treat pixel RGB as independent evidence if also applying the legacy full-frame RGB quadratic at the same pixels. A depth-only A/B is advisable before joint RGB-D.

For unknown hit at depth `z`, let `ell_U(D,I|z)` be a normalized sensor likelihood at depth `z` times a fixed broad RGB/background-of-unknown-object likelihood. Then

\[
 \boxed{L_{cp}(x)=p(D,I,V=1\mid x,M=1)
 =\sum_i q_{i,cp}(x)\ell_{i,cp}(D,I\mid x)
  +\int_{z_{near}}^{z_{far}}f_{U,cp}(z\mid x)\ell_U(D,I\mid z)\,dz.}
\]

Detection factors enter each summand if applicable. For a missing depth observation, define a separate `p(V=0,I|x,M=1)` using hit-dependent sensor dropout and `q_empty`, never use `D=0`. The likelihood can be a joint density over depth/colour and return status; it is *not* necessary that its `V=1` branch alone integrates to one if the `V=0` branch carries remaining probability.

For foreground RGB+depth, use the same latent hit identity for both channels so that RGB cannot continue to pull the door towards cavity pixels once depth association rejects that correspondence. Preserve silhouette/foreground occupancy evidence as a distinct channel, guarding against applying the same pixel information twice. Pixels outside the hard foreground mask MUST NOT contribute observed background RGB-D gradients to the object; they can provide the separately declared silhouette negative-ray constraint.

Optional **measured** spatial/time-footprint uncertainty near transitions: convolve `L_{cp}` with a calibrated registration/exposure distribution:

\[
 \bar L_{cp}(x)=\int L_c(D_{cp},I_{cp};\,p+\delta p,\,t+\delta t,x)\,q_c(\delta p,\delta t)\,d\delta p\,d\delta t.
\]

This prevents sharp label changes at an occlusion boundary from being mistaken for centimetre-scale depth noise. `q_c` needs an independent calibration or explicitly labeled assumed model. Distinguish shared camera misregistration (correlated across pixels) from independent pixel noise.

## 5. The free-space contradiction is contained automatically

If the model contains an opaque Gaussian `i` at `z_i=0.75 m` but observation is a reliable deeper surface `D=0.95 m`, then the modeled-hit Gaussian-depth likelihood is negligible. The *unknown* explanation at 0.95 m contains

\[
 f_U(0.95)\propto \prod_{i:z_i<0.95}(1-\alpha_i).
\]

For a single wrong front alpha `0.9`, the likelihood of an otherwise identical deeper unknown surface is reduced by factor `0.1`, hence a `-log(0.1)≈2.30` penalty. For alpha `0.99` it is `≈4.61`. This is **not** a separate manually weighted loss; it follows from first-return visibility. Conversely, when the observed door is in front of a modeled cavity point, the door's matching likelihood dominates and the cavity point is not given an erroneous residual.

Diagnostic of front-of-observation occupancy with a calibrated smooth crossing probability:

\[
 P_{front,i}(D)=\Phi\!\left(\frac{D-z_i}{s_{vis,i}}\right),\quad
 T_{front}(D)=\prod_i[1-\alpha_i P_{front,i}(D)],\quad
 E_{front}= -\log T_{front}(D).
\]

`s_vis` represents calibrated depth/registration uncertainty—not a user-defined 2 cm cutoff. The diagnostic may supplement evidence **only if not duplicating the survival factor already present in the joint likelihood**. A spurious Gaussian far in front of `D` creates a strong *violation* but may have almost zero local derivative with respect to depth while remaining in front; do not force the IEKF into an invented large velocity. Record and resolve it by full nonlinear counterfactual geometric repair / supported coherent pose candidate.

## 6. E-step: latent surface posterior

For each valid foreground pixel at current IEKF iterate `x^k`, define component evidence:

\[
 A_{i,p}=q_{i,p}(x^k)\ell_{i,p}(D,I\mid x^k),\quad
 B_p(z)=f_{U,p}(z|x^k)\ell_U(D,I\mid z),\quad L_p=\sum_i A_{i,p}+\int B_p(z)dz.
\]

Responsibilities:

\[
 \boxed{\gamma_{i,p}=A_{i,p}/L_p},\qquad
 \boxed{\gamma_{U,p}(z)=B_p(z)/L_p},\quad
 \gamma_{U,p}=\int\gamma_{U,p}(z)dz,\quad \sum_i\gamma_{i,p}+\gamma_{U,p}=1.
\]

Calculate safely in log-space (`logsumexp`), with a finite normalized unknown prior. For initial *association proposals only*, it is valid to widen the predictive likelihood with `J_i P_t^- J_i^T` to reflect uncertainty about where the moving surface is; DO NOT then also add this prior covariance to `R_D` inside the IEKF update, because that double-counts the predicted state uncertainty. Better: evaluate posterior association by integrating over the full state prior when tractable or use explicit and labeled Gaussian approximation.

The unknown-surface posterior must be logged. Unmodeled geometry over multiple cameras/frames should be carried forward as potential Phase-4 births, not silently converted to wrong movements or free per-frame fitted hazard.

## 7. M-step / iterated IEKF: exact objective, locally fixed associations

The *true* per-frame MAP objective is

\[
 \boxed{E(x)=\tfrac12\|x\boxminus x_t^-\|_{(P_t^-)^{-1}}^2-\sum_{c,p: M_{cp}=1,V_{cp}=1}\log L_{cp}(x)+E_{validity}+E_{silhouette}.}
\]

`E_validity` accounts for missing depth when explicitly modeled; `E_silhouette` uses the existing foreground-mask occupancy formulation, but avoid duplicate RGB/alpha evidence and clearly define correlations. The manifold prior uses the committed left retraction and fixed predicted state. On each nonlinear IEKF iteration:

1. Render *all* cameras with the same current state, fresh depth sorting, all contributors.
2. Evaluate `L_p(x^k)`, `gamma_i,p`, `gamma_U,p(z)` on the same render.
3. Create a fixed-responsibility EM lower-bound / IRLS surrogate (not a new true objective):

\[
 Q(x\mid x^k)=E_{prior}(x)-\sum_{p}\left[\sum_i\gamma_{i,p}^k\log(q_{i,p}(x)\ell_{i,p}(x))
 +\int\gamma_{U,p}^k(z)\log(f_{U,p}(z\mid x)\ell_U(z))dz\right]+E_{silhouette}.
\]

4. Approximate the local motion gradient/information in the existing matrix-free grouped normal system, including **visibility/opacity** derivatives, not only depth errors. With fixed `gamma`, the matched-depth contribution is:

\[
 r_{i,p}=D_p-z_i(x),\quad J_{i,p}=\frac{\partial z_i}{\partial\delta x},\quad
 b_{match}\approx\sum_{p,i}\gamma_{i,p}J_{i,p}^TR_{D,i,p}^{-1}r_{i,p},\quad
 \Lambda_{match}\approx\sum_{p,i}\gamma_{i,p}J_{i,p}^TR_{D,i,p}^{-1}J_{i,p}.
\]

For centre-depth `z_i=e_3^T R_cw(mu_i-t_wc)`, the centre translation Jacobian is `J_mu=e_3^T R_cw`; orientation affects the footprint opacity and potentially the ray-conditional depth (if used). The visibility score adds terms from `log q_i = log alpha_i + sum_{j<i}log(1-alpha_j) - Lambda_U(z_i)`. For a model-front contributor `j`, `partial log T / partial alpha_j = -1/(1-alpha_j)`. These derivatives are **coupled across Gaussians** on each ray; retain exact JVP/VJP and the canonical grouping map `G` rather than estimating independent camera/group corrections. The unknown terms supply the free-space contradiction derivatives through `f_U`.

5. Solve the trust-region / LM step using existing `x.boxplus` and fixed prior. **Accept a trial only using the full actual nonlinear `E(x_trial)` with fresh contributor ordering and association re-evaluation**, plus existing convergence certificate; never compare only a frozen-gamma surrogate. Record all accepted/rejected step diagnostics and per-channel gradients.

If cross-layer ambiguity is high or Gaussians are fully occluded, their direct measurement Fisher should be small: propagate by process prior and preserve/increase uncertainty. If the same rigid part is observed by other Gaussians/cameras, the *explicit canonical supernode/group coherence* transfers that information; do not invent a hidden point's individual depth observation.

## 8. Exact mixture information and posterior covariance

For finite mixture term log-component scores `a_k(x)=log A_k(x)` and responsibilities `gamma_k=exp(a_k)/sum exp(a)`, define `s_k=grad a_k`. Then

\[
 \nabla\log L=\sum_k\gamma_k s_k,\quad
 \nabla^2\log L=\sum_k\gamma_k\nabla^2 a_k+\operatorname{Cov}_{\gamma}(s_k),
\]

hence the observed information is

\[
 \boxed{-\nabla^2\log L=\underbrace{-\sum_k\gamma_k\nabla^2 a_k}_{\text{complete-data surrogate information}}-\underbrace{\operatorname{Cov}_{\gamma}(s_k)}_{\text{missing association information}}.}
\]

The continuous unknown contribution has the same integral interpretation. **Using the EM complete-data Fisher alone for the Phase-1 posterior can be overconfident**. The exact observed Hessian can be indefinite; use stable PSD information approximations / regularized curvature only with explicit diagnostics and validation. The existing `approximate_posterior_information()` takes per-Gaussian 12x12 blocks and already discards inter-Gaussian covariance; preserve this approximation label, test directional coverage and credible intervals. Do not insert `J P^- J^T` in both the association proposal and the measurement covariance. Weakly observed directions must not become falsely more certain.

## 9. Persistently wrong canonical geometry is not automatically moving

A false foreground contributor can be strongly inconsistent (`-log T` large) and still have negligible **local depth translation derivative** while far in front of the observed return. The correct explanation may be a wrong Phase-0 surface, wrong opacity, wrong pose, or mixed/invalid sensor depth. One ray cannot identify them. During Phase 1, colors, opacities, scales and canonical geometry remain frozen. Record per-Gaussian/per-source-camera **foreground contradiction** evidence. Test competing explanations over **multiple cameras and times**, using actual nonlinear renders:

- geometry/opacity correction candidate at fixed motion;
- coherent motion candidate at frozen geometry;
- unresolved observation / unknown-surface candidate;
- baseline.

Use the same held-out evaluation to judge candidates, and do not count a pixel both as fit and independent validation. If the new model avoids a false foreground contradiction by giving `unknown` all the probability, this is a failure: fix the unknown hazard prior and validate log-likelihood on held-out data. Only validated persistent geometry repairs belong in a separate Phase-0/geometry stage, never disguised as unexplained Phase-1 velocity.

## 10. Computation and implementation contract

**Exact current code interfaces** (verified on main `90dbf58`):

- `estimator/state.py::GaussianMotionState` has `[mu,R,v,omega]` and 12D `boxplus` left rotation.
- `estimator/sparse_iekf.py::iekf_frame_step` handles once-per-frame process/covariance prediction, fixed MAP prior, iteration, trust region, grouped motion solves and approximate 12x12 posterior.
- `estimator/sparse_iekf.py::measurement_operators` currently implements `H^TWr`, `H^TWH` and Fisher for a single per-pixel rendered depth and RGB plus alpha.
- `estimator/renderer_bridge.py::build_linearization` and its CUDA JVP/VJP use `Z/(A+epsilon)` for depth, plus RGB/alpha, with cached contributor ordering.
- `estimator/observation_terms.py::camera_terms` holds calibrated depth sigma, depth-model-variance add-in, robust Student-t, and existing alpha gate; `estimator/masked_observation.py::phase1_camera_terms` handles mask and rendered background assumptions.
- `observation_diagnostics/renderer_access.py::pixel_contributions` reports exact per-pixel composited alpha/T/Gaussian ID, with tests verifying `sum alpha_i*T_i` equals rendered alpha.
- `estimator/visibility_constraint.py` has separate negative-ray semantics but is currently mutually exclusive with the masked observation mode; do not silently turn on both or add background depth as object measurement.

**Proposed new units (feature-flagged):**

1. `estimator/first_return_components.py`: per-ray z-sorted contributor contract, alpha/transmittance parity, unknown hazard, no-return probability, depth/colour likelihood, batched per-pixel logsumexp and unknown integral.
2. `estimator/first_return_association.py`: responsibilities and diagnostics, with a pixel-footprint model *only when its uncertainty is independently supplied*; preserve original camera identity.
3. `estimator/first_return_operators.py` + CUDA kernels: per-layer JVP, VJP, visibility scores and Fisher/preconditioner approximations. Avoid a per-Gaussian python loop or full Jacobian construction.
4. `estimator/masked_observation.py`: opt-in first-return mode with hard foreground pixels for measured RGB-D; retain existing silhouette treatment or explicitly replace without double counting; propagate exactly one likelihood/score/normal mapping into the solver.
5. `estimator/sparse_iekf.py`: select first-return measurement callback through `iekf_frame_step`; local EM/IRLS plus whole-state nonlinear trial energy; fixed canonical group mapping and all-camera reduction unchanged.
6. Diagnostics: posterior assignment heatmaps; modeled hit / unknown / no return; front contradiction per pixel; forces on each Gaussian and canonical group by camera and channel; temporal surface support; model changes to q and log-likelihood; mean/p99 motion and covariance consistency.

**Numerical feasibility:** start with depth-only CPU/PyTorch reference on a few pixels, then fused CUDA full images. For piecewise constant `lambda_U` and constant noise `sigma` over each ray interval, unknown integrals can be evaluated analytically with Gaussian CDFs, or using bounded quadrature checked against high-precision reference. Check units (`lambda` in 1/m, `f_U` per m, Gaussian depth density per m). Cache all current per-pixel contributions for the *same* state/iteration and never claim exactness if a rasterizer's early-stop contributors are missing from the new likelihood. Stable `log1p(-alpha)` and `logsumexp`; temperature/sensor blur only if physically justified. Handle finite depth near/far and ties carefully. Keep contributions across cameras independent conditionally on the scene only when measured camera noises are independent; record shared calibration uncertainties separately.

## 11. Non-negotiable tests / rollout gates

**Gate 0 — probabilities and source geometry:** `sum q_i + integral f_U + q_empty = 1` for 0/1/many Gaussians with arbitrary opacities; simulated samples recover first-return frequencies; `sum T_i alpha_i` and `RGB/Z` parity with production rasterizer; exact hard foreground eligibility; no invalid depth gradients; normalized density units.

**Gate 1 — gradient correctness:** analytical JVP/VJP adjoint identity and finite differences for depth position, alpha through projected conic, orientation/scale rotation and group-induced motion; finite-difference of the actual negative log likelihood including unknown survival and depth sorting where differentiable; at nonlinear reorder points, compare true energy rather than enforcing invalid derivatives.

**Gate 2 — toy physics:** (a) measured door 0.75 m and hidden cavity 0.95 m: high door association, cavity no spurious update; (b) a false opaque Gaussian 0.75 m, reliably measured 0.95 m: high front contradiction and likelihood penalty `-log(1-alpha)` without inventing velocity; (c) reliable new surface not represented: unknown posterior, no spurious motion; (d) ambiguous registered boundary: uncertainty of assignment, not a fake blended 0.81 m surface; (e) two modeled Gaussians representing same physical surface: one measurement, no double count; (f) true moving object with occlusion and noisy views: moving points retain enough information to track.

**Gate 3 — covariance:** predictive coverage for association, no prior double count, missing-information variance sanity; static/hidden Gaussians retain realistic posterior uncertainty. Check inter-Gaussian approximation and frame 0–1 velocity cross-covariance.

**Gate 4 — real comparisons:** same original scene, camera calibrations, Gaussian initialization, canonical partition, process prior and iteration budget. Baseline old single-blended depth vs proposed depth-only first-return vs joint RGB-D. Fridge 80–99 (verified static), 100–105, and closing frames 119–190; synthetic USB ground truth. Compare per-Gaussian update-forces by source camera and group, stationary median/p99 displacement, velocity carry, converged-frame fraction, coherent moving support F1 on synthetic, Phase-2/3 joint geometry, held-out RGB/depth log likelihood, missing/false geometry contradiction evidence. An improvement in median while tails still explode is not passing.

**Gate 5 — avoid regressions:** hard foreground masks; real source-camera calibration and timestamp parity; no background depth pushes, no motion when hidden, no artificial surface merging, no persistent unknown explaining everything, no late-training geometry changes in IEKF, no replacement of full nonlinear trial evaluation by a linearized acceptance score.

## 12. Research caveats and recommended initial deliverable

This is a physically motivated probabilistic *proposed* model, not proven calibration. The current 3DGS alpha is not known to equal the physical RGB-D sensor's depth-hit probability, and frozen centre-depth is a footprint approximation. Quantify first-hit calibration and false-foreground detection on a synthetic physics oracle first. A missing geometry/pose ambiguity is not identifiable from one ray. Do not claim covariance calibrated until coverage tests pass. Report the actual implementation branch/commit and local modifications; `main` does not currently contain this new likelihood.

**Recommended first deliverable:** read-only per-ray contributors, modeled/unknown posterior and free-space contradictions on the fridge around frames 119–190, followed by a few-pixel differentiable reference, then a fused first-return operator and real A/B. Maintain the baseline untouched until validated.
