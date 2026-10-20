# Masked RGB and foreground-valid depth: pre-implementation audit (2026-10-20)

This audit was done on `main` at `d2343f7`, the hard negative-ray constraint commit, before
the new observation mode was written. It records what the code and the data actually do.
Line numbers refer to the files after the patch where a function moved; the behaviour
described is the pre-patch one.

## Observations the dynamic stages see

- **RGB.** The loader is `tools/run_iekf_scene.py:236` (`dataset_rgbd_frames`).
  - It reads `<source>/rgb/<cam>/frame_%06d.png` as 8-bit PNG/255, in float32 [0,1].
  - It applies no gamma linearisation. That is the same convention as static training and
    the rasterizer, and this patch adds no conversion.
  - Native resolution, no resizing anywhere in the motion stages: the camera specs come
    from the model loaded at `-r 1`.
- **Depth.**
  - Source: `<source>/depth_npy/<cam>/frame_%06d.npy`, metric camera-Z in metres.
  - Validity is `valid = dz > 0` (`run_iekf_scene.py:302`). A zero is a missing return,
    never a target.
  - SAPIEN (`fridge_10905`, cam_000, frame 0): depth is positive on exactly the mask
    (100% inside, 0% outside), and the RGB outside the mask is exactly (0,0,0).
  - Real `fridge_plastic`: depth is valid on 54–67% of the off-object pixels, and the
    background RGB averages 0.45 with standard deviation 0.14–0.20.
- **Masks.** `<source>/mask/<cam>/frame_%06d.png`, one per camera and frame, binary
  {0,255} at native size (640×480).
  - They are whole-object silhouettes, not moving-part labels.
  - SAPIEN: rendered.
  - Real: Grounded-SAM2 "red fridge" (`metadata/mask_generation.json`).
  - Before this patch the motion stages read masks only under the opt-in hard constraint
    (`--object-mask-constraint`).

## Measurement model before the patch

- **Render background.** `build_linearization` composites over a constant (3,) `B`, which
  defaults to black (`estimator/renderer_bridge.py:204`). No dynamic runner passes another
  value. `B·δT_end` is correct in the JVP, VJP and Fisher; that was proven on 19 Oct
  (`docs/BACKGROUND_MASK_IEKF_AUDIT_2026_10_19.md`) and is not rewritten here.
- **Phase 1 precision** (`estimator/sparse_iekf.py:235` `_precisions`):
  - RGB on every pixel with σ = `cfg.rgb_sigma` (0.1), multiplied by a **Student-t IRLS
    weight recomputed from the current residual** (`estimator/measurement.py:389`).
  - Depth on `depth_valid & alpha > min_alpha_for_depth` (`sparse_iekf.py:260`), robust in
    the same way.
- **Acceptance energy** (`measurement_energy`, `measurement.py:426`) is ½Σ w(r)·r²/σ².
  - Because the weights are recomputed at every trial state, the accepted objective is
    not one fixed function. This is pre-existing and documented, not changed.
  - A NaN depth would reach it, since 0·NaN² = NaN, wherever a stored depth is NaN. The
    loader never stores NaN, so in practice this never happened.
- **Phase 2/3 precision** (`estimator/phase2_shared.py:29` `effective_precision`):
  - 1/(σ² + σ²_model) per pixel and channel, times the Student-t weight.
  - σ²_model is estimated once, from the frame-0 residual (`estimator/model_error.py:79`).
  - Depth as in Phase 1, plus the model-error variance.
- **The predicted-alpha depth gate** `alpha > min_alpha_for_depth`, 0.05, is
  model-dependent: it moves with the state. It is documented here and kept unchanged in
  the new mode, in both Phase 1 and Phase 2/3.

## Consequence on real data (measured 2026-10-19, `fridge_plastic`, Phase 1, frame 1)

- 97.7% of the measurement energy comes from RGB pixels outside the mask, the grey
  background against the object rendered over black.
- That term pulls the edge Gaussians outward: its gradient is 46% of the in-mask gradient
  and acts on 28% of the Gaussians.
- Over 328 frames the 0.6 m model dispersed, to a median displacement of 0.5 m.
- A second, unrelated cause is untouched by this patch (see "Not addressed" below):
  poorly observed Gaussians drift at constant velocity. The motion model resets the
  velocity variance to 1 (m/s)² every frame (`tools/kpart_pipeline.py:1751`).

## Operators available

- **Accumulated alpha.** `_C_vis` (19 Oct) gives `A_vis`, its JVP and its VJP. In full-ray
  mode (`g_i = 1`), A_vis equals the renderer's `A = 1 − T_end` to 3e-7, with the same
  last contributor (`tests/test_visible_alpha_operators.py`).
- **Missing before this patch:** an alpha Fisher, which a consistent preconditioner and
  posterior need.
- **Stage lifts.** All three map to the renderer's [P,6] pose tangent:
  - Phase 1: `expand_step`/`extract_pose` (supernodes);
  - Phase 2: `RigidState.jvp/vjp`;
  - Phase 3: `DisplacementOnlyState.jvp/vjp`.

## What the new mode must change (and nothing else)

1. **Observation.** Y(b) = M·I + (1 − M)·b over the whole image. The render uses the same b.
2. **Depth.** Precision only on V_orig ∧ M, and the residual is computed only there. The
   stored value elsewhere is never read.
3. **Expected-quad mode.** A fixed quadratic RGB cost W, plus ½κ(A − M)² with
   κ = trace(W·Σ_b). This term must enter:
   - the energy;
   - the RHS;
   - the normal operator;
   - the Fisher preconditioner;
   - the posterior.

   The Student-t RGB weight cannot be carried through that expectation, so the new mode
   uses the fixed W:
   - Phase 1: 1/σ²;
   - Phase 2/3: 1/(σ² + σ²_model), without the robust weight.

   Depth keeps its robust precision.
4. **Untouched:** static training, `_C`, the hard-constraint path, the coherence and
   support logic, the GT evaluators.
