# Surgical fixes A–C (2026-10-22)

Specification: `CODEX_SURGICAL_CONFIRMED_BUG_FIXES_RIGEULER3DGS_2026-10-22.md`. Base: `10b29be`.
Tests: `tests/test_surgical_fixes_2026_10_22.py`.

## A. Frame-time units in Phase 2 and Phase 3

**Bug.** Production callers omitted `times`, so every window solver fell back to `torch.arange(T)`. That gave dt = 1 s,
although the loader records `dt_used_s` = 1/30 s. The Kabsch twists and the Phase 3 initial `qdot` were per frame.

**Fix.** Production callers now pass elapsed seconds; the solvers' process equations and default APIs are unchanged.

| file | change |
|---|---|
| `utils/sequence_direction.py` | New `validate_times`, `times_from_intervals`, `processing_time_s`. Elapsed seconds in processing order, starting at 0, built from the loader's own interval `frames[k][2]`. It is cross-checked against `frame_ids['frames'][k]['dt_used_s']`; a contradiction fails. Non-finite, zero or negative intervals and non-increasing times are refused. A backward pass gets the same increasing elapsed time. |
| `tools/run_rigid_se3_pipeline.py` | `time_s` is passed to `kabsch_initialize(..., times=)` and `run_rigid_window(..., times=)`. The trace CSV `time` column is now seconds. Writes `phase2_rigid/time_base.json` (dt_s, time_s, units) and `summary.json["time_base"]`. `--sigma-vdot` / `--sigma-wdot` help now states units. |
| `tools/run_joint_phase3.py` | `time_s` is passed to `fit_model(..., times=)`. `_finite_rate(q, time_s)` is `np.gradient(q, time_s)`, and `load_initialisation(..., time_s=)` gives prismatic and revolute the same vector. `summary.json` config holds `dt_s`, `time_s_processing_order` and rate units; `"dt": "1 frame"` and `sigma_qddot_per_frame2` were removed. Help text for `--sigma-qddot` and `--prior-qdot` now states units. |
| `estimator/phase2_model_comparison.py` | `fit_model(..., times=None)` forwards to `run_window`. Other callers are unchanged. |
| `estimator/joint_iekf.py`, `estimator/rigid_part_runtime.py` | Comments and docstrings only, now stating the units. |

- **Units:** v m/s, ω rad/s, qdot m/s or rad/s. The σ values are white-acceleration densities, in m/s^(3/2) or
  rad/s^(3/2).
- **Not changed:** numerical values (σ, priors), process equations, pose ordering and joint geometry.
- **Other production call sites:** the active pipeline has none. The joint reduction works on poses only, and the
  pose-space P/R fitting through `fit_model`/`run_window` is used only by Phase 3. Legacy routines were not touched.

## B. Stage signatures hash file contents

**Bug.** `tools/stage_signature.py` hashed only RGB and depth file names and sizes.

**Fix.**
- `_content_digest` is a streaming SHA-256 over the relative path, byte length and bytes of the first `frames` files
  of every used camera, read in 1 MiB chunks.
- The new fields are `rgb_content_sha256` and `depth_content_sha256`, with `schema: 2`. A schema-1 manifest never
  equals a schema-2 signature, so it is refused, never silently reused.
- Mask content, endpoint, code, upstream and the reuse/refuse logic are unchanged.
- Cost: about 1.2 s for a 60-frame, 8-camera scene.

## C. Driver exit status

**Bug.** `run_bidirectional_all.sh` recorded `FAILED`, `REFUSED` and `SKIPPED` but still exited 0.

**Fix.** After the unchanged summary, the driver counts `FAILED(`, `REFUSED(` and `SKIPPED(` records:
- any such record gives exit 1, with a one-line message;
- otherwise it exits 0;
- a later successful scene never resets an earlier failure;
- a dry run uses the same rule. A normal dry run records nothing, so it still exits 0; a dry run of a scene that
  would be skipped (for example, missing masks) now exits 1.

The stage logic, watchdog, retries, messages and outputs are unchanged.

## Verification

| test | status |
|---|---|
| A: time_s = [0, 1/30, 2/30] forward and backward; contradictory metadata refused; zero, negative, NaN and inf intervals refused; nonuniform intervals | PASS |
| A: 0.3 m/s gives 0.01 m per 30 Hz step; Kabsch twist 0.3 m/s and 0.6 rad/s; Phase 2 prediction 0.01 m; Phase 3 qdot 0.3 m/s and thetadot 0.6 rad/s; backward sign | PASS |
| A: Q_qq = σ²dt³/3, Q_q,qdot = σ²dt²/2, Q_qdot,qdot = σ²dt at dt = 1/30 | PASS |
| A: Phase 2 and Phase 3 windows predict with dt = 1/30 on the real renderer (tiny scene); production call sites pass the loader times | PASS |
| A: Phase 2 `poses.npy` stays original-chronological; Phase 3 geometry bitwise unchanged | PASS via the existing tests listed below (unchanged code paths: `_chrono`, `geometry_unchanged`) |
| B: same-size RGB and depth byte changes refused; identical inputs reused; mask change refused; schema-1 manifest refused | PASS |
| C (stub stages): all success exits 0; a failure, a refusal or missing masks exits non-zero; a later success keeps the failure; dry run unchanged | PASS |
| Directly affected existing tests | see the commit message |
| Full scenes, production A/B, 64 cameras | NOT RUN (out of scope) |

**Left unchanged:**
- masked RGB/alpha, foreground-depth validity and the alpha coverage gate;
- model-error estimation and the covariance approximation;
- static training;
- support and joint logic, and the single-part structure;
- the Phase 1/2/3 MAP acceptance.

## Separate note (no code change)

- **Process prior strength.** With dt = 1/30 s instead of 1, the unchanged σ values give a process covariance per step
  about 27,000× smaller in position (dt³/3) and 30× smaller in rate (dt) than the earlier, unintended runs did. The
  initial velocity priors (`sigma_xi0` = 1, `--prior-qdot` = 1) are now per second. Results from before this fix are
  therefore not directly comparable. Whether the existing σ values suit physical units is an open modelling question
  and was not retuned here.
