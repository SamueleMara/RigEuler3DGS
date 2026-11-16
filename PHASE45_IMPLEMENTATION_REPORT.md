# Phase 4 / Phase 5 implementation report

> **Superseded for Phase 4 (2026-11-11).** The Phase-4 implementation described below did not scale: on red_fridge it
> ran 13 h without output, because of decode thrash and a full-frame scoring loop. Its decision statistic was not the
> objective either. See `PHASE4_CORRECTION_AUDIT.md`, `PHASE4_EXACTNESS_REPORT.md`, `PHASE4_PERFORMANCE_REPORT.md` and
> `PHASE4_REAL_VALIDATION_REPORT.md`. `tools/run_phase4_recovery.py` now runs the corrected, staged Phase 4.
> The old modules (`material_completion.py` scoring, `residual_discovery.py`, `canonical_fusion.py` fusion,
> `candidate_acceptance.py`, `phase4.py`) remain only as the pre-correction reference for the frozen regression fixture.
> Phase 5 is unchanged.

Date 2026-11-10. Specification: `RigEuler3DGS_PHASE4_PHASE5_CODEX_IMPLEMENTATION_2026-11-10.md`.
Checkout: worktree `~/Joint3DGS/RigEuler3DGS-p45`, branch `phase45`, from `main` @ `6807d83` (uncommitted).
Gate 0 audit: `phase45_repo_audit.md`. **Optional final joint refinement ran: NO.**

## 1. What was built

| file | role |
|---|---|
| `submodules/diff-gaussian-rasterization/` (10 files modified) | opt-in `rasterize_gaussians_rgbd`: one front-to-back pass returns RGB, radii, legacy inverse depth, alpha = 1 − T_final, Z = Σ w·z (camera axial). Backward: exact compositing adjoints for Z and A (incl. the effect on later splats); dL/dz reaches the means through the existing depth path. `metric_depth_from(Z, A) = Z/(A+ε)` is torch autograd. Every new pointer defaults to `nullptr`; legacy entry points unchanged |
| `recovery/__init__.py` | pins this checkout's extension (the conda env's editable install points at main; main's `_C` is untouched) |
| `recovery/raster_oracle.py` | float64 torch re-implementation of the kernel (projection, EWA, tiles, depth order, clipping, early stop, SH) — the Gate 1 reference |
| `recovery/contracts.py` | Phase-3 ingestion + preflight (one direction, same endpoint, explicit joint type, original-frame order, q anchor = 0, unit axis, support rows vs PLY, camera set, frame window, SHA-256 of every input) |
| `recovery/views.py` | cameras (renderer matrices + K/c2w, pixel-centre convention), check against `cameras.json` (real and synthetic formats), lazy RGB / metric depth / strict binary mask views |
| `recovery/articulated_scene.py` | fixed-joint transport of positions **and** orientations, `world_SH` (legacy parity) / `local_SH` (canonical-frame view direction), native RGB-D-alpha render; tensor-only P/R transform for gradient tests and the optional refinement |
| `recovery/rgbd_loss.py` | FreeArt objective (λ_ssim 0.2, λ_mask 1, λ_depth 1, white bg — read from the FreeArtGS real config) + per-pixel map; declared differences: depth only on valid foreground normalised by its support, low-alpha depth pixels excluded and counted, optional Student-t |
| `recovery/gaussians.py` | canonical container (GaussianModel attribute names, standard 3DGS PLY) + row-aligned lineage: material, gaussian_uid, parent_uid, original_id, provenance |
| `recovery/material_completion.py` | **4A**: layer-aware components (footprint overlap, normal agreement, same tangent layer), full-scene S/M counterfactuals (affected-pixel + full-view), screening on informative views, acceptance re-scored on ALL views, halves consistency, OMIT check (`SUSPECT_ORIGINAL`), rounds with re-tests of flipped/neighbouring components, final verification sweep, oscillation guard, core joint-validity check (`UPSTREAM_JOINT_INVALID`), shared-joint sensitivity (`JOINT_CONFIDENCE_UNCALIBRATED`) |
| `recovery/residual_discovery.py` | **4B** pixel codes at every view: EXPLAINED / COVERAGE_GAP / FRONT_SURFACE / BEHIND_PREDICTION / DEPTH_INVALID / MIXED_EDGE / BACKGROUND |
| `recovery/canonical_fusion.py` | **4B** back-projection `K⁻¹[u+½, v+½, 1]`, inverse joint transport x_M = Q_fᵀ(y−p_f), measurement covariance P_y = R J Σ Jᵀ Rᵀ, explained-geometry filter (Mahalanobis to posed Gaussians), layer-aware fusion (voxel × signed normal bin), patches, coherence ranking |
| `recovery/candidate_acceptance.py` | **4C** S/M patches paired by source pixels; hard-label local refinement on fit views (J, q fixed); validation gain vs OMIT on held-out rows (parity, fixed in advance); all-view contradiction audit; anisotropic duplicate test; 8 statuses |
| `recovery/phase4.py` | 4A → 4B → 4C driver and all artifacts (`support_completion/`, `discovery/`, `final_canonical/`) |
| `recovery/topology.py` | clone / split / prune with GaussianModel's rules on canonical tensors, Adam moments and lineage updated atomically; pruning only on evidence from a complete sweep |
| `recovery/training.py` | **Phase 5** `gaussian_rgbd` (stages A/B/C, SH progression, densification) and strict `sh_only`; seeded all-view sampler; initial/final all-view sweeps; joint fingerprint (file hashes + transport buffers) asserted at checkpoints and end; `joint_unchanged_proof.json` (`joint_optimized=false`), byte copy of the joint files |
| `recovery/joint_refine_final.py`, `tools/refine_joint_geometry_final.py` | **optional** minimal joint parameterisation (S² tangent retraction; revolute point ⟂ axis; anchor q fixed); CLI is a dry run without `--enable`, refuses without a valid Phase-5 proof; the optimisation loop is deliberately not wired |
| `tools/run_phase4_recovery.py`, `tools/train_phase5_rgbd_gaussians.py` | the spec's CLIs (`--source --direction --model --phase2-dir --phase3-dir --joint-type --cameras --frame-window --sensor-model --output --seed --dry-run`, every config field exposed) |
| `run_phase45_synthetic.sh` | Phases 4 + 5 on the 10 v5 synthetic runs; type from the documented R/L rule; writes only under `$OUT` |

## 2. Build

```bash
cd submodules/diff-gaussian-rasterization
CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9 \
  ~/miniconda3/envs/rigeuler3dgs/bin/python setup.py build_ext --inplace        # 47 s, in this worktree only
```

## 3. Tests run (all executed here, RTX 4070 Laptop)

| gate | test file | result |
|---|---|---|
| 1 | `test_rgbd_cuda_forward_backward.py` (9) | PASS. Forward = oracle (RGB/A/Z/invdepth < 2e-5, radii exact), A = 1 − T_final, empty view A = Z = 0, legacy vs RGB-D entry bitwise equal, Z/A ≠ 1/invdepth ≠ first return, CUDA grads = exact oracle grads to ~1e-6 (limit 1e-4) for means / scales / rotations / opacities / SH / colors and raw params, oracle = central FD (1e-5), each term reaches only its path |
| 1 | `test_rgbd_legacy_compat.py` | PASS. Legacy forward (colour, radii, invdepth, features, positions) **bitwise** equal to main's installed `_C`; gradients within main's own run-to-run atomicAdd floor (3.8e-6 vs 3.8e-6) |
| 1 | existing suites with the new build: `test_iekf_cuda_operators` (41), `test_iekf_joint_reduction_cuda` (15), `test_joint_renderer_composition` (11), `test_unified_cuda_renderer_jvp_vjp` (22), `test_v2_group_cuda` (26), `test_background_compositing_audit` (11), `test_first_return_cuda_parity` (14), `test_render_response_contrast` (20) | all PASS |
| 0/2 | `test_phase45_contracts.py` (2) | PASS on real v6 red_fridge (forward R), v5 blade (backward P), USB (backward R); refuses opposite direction, wrong endpoint / run, foreign camera set, wrong window, anchor q ≠ 0, non-unit axis, out-of-range support |
| 2 | `test_phase45_direction_anchor.py` (5) | PASS. Anchor identity (bitwise, incl. image), static invariance, rigidity, Σ → QΣQᵀ, pose = pipeline's Phase-3 transport < 1e-8 (fridge, blue box, USB, blade), backward mapping, joint gradients = FD, rendered joint gradient CUDA = oracle, no joint tensor requires grad, local_SH = world_SH at anchor |
| 3 | `test_phase4_material_counterfactual.py` (3) | PASS. R/forward: 88 % of withheld moving recovered, rest UNRESOLVED at the hinge; P/backward: 92 %; 0 static labelled moving; 0 moving called static; wrong-label counterfactual worse; both no-motion controls → UPSTREAM_JOINT_INVALID |
| 4/5 | `test_phase4_discovery_acceptance.py` (3) | PASS. Drawer floor (backward, visible only when out) → 414 MOVING Gaussians on the true plane, loss 0.0431 → 0.0155, first-visible frame kept as provenance; door back face (12 mm behind the front) → 97 MOVING on the back layer, no collapse; planted front-occluding patch rejected; N_final = N_orig + N_acc |
| 6 | `test_phase5_training.py` (5) | PASS. gaussian_rgbd R/fwd 0.038 → 0.012, P/bwd 0.032 → 0.009, joint bytes + buffers identical, proof written; sh_only geometry/opacity bitwise unchanged, depth/mask monitors identical; term switches gate exactly; canonical split, inherited material/parent, Adam aligned; sampler covers every view per epoch |
| 7 | `test_optional_final_joint_disabled.py` (3) | PASS. Not referenced by any stage or runner; dry run writes nothing; `--enable` without proof refused; parameterisation exact at zero, unit axis, ⟂ point, anchor fixed, FD gradients |

Run them: `for t in tests/test_rgbd_*.py tests/test_phase45_*.py tests/test_phase4_*.py tests/test_phase5_*.py tests/test_optional_*.py; do python $t; done` (no pytest in the env).

## 4. Known caveats

* **Antialiasing**: the upstream covariance backward is wrong with antialiasing on (audit §2); Phase 4/5 refuse it.
* **Expected depth is not first return**: semi-transparent splats mix what lies behind them; 4B drops back-projected points on existing posed geometry (Mahalanobis ≤ 3) for that reason.
* **Thresholds are declared, not calibrated** (component linking, layer tolerance 8 mm, decisiveness 25 %, residual tolerance 1 cm, acceptance margins). They were tuned only on the procedural gate scenes.
* **Validation is conditional on the upstream joint** (Phase 3 used every frame).
* **Joint ambiguity**: sensitivity is reported as ranges under declared shared perturbations; no probabilities.
* **SH convention**: Phase 4 compares S/M at SH degree 0 (the appearance Phases 1–3 validated); Phase 5 uses `local_SH` and records the initial world/local gap in `renderer_parity.json`.
* **Optional refinement**: parameterisation and CLI guard only; the loop is not wired and was never run.
* **Not done**: hooks in `run_bidirectional_all.sh` / stage cache; real-scene Phase 4/5 runs; synthetic full-length runs (handed to the user).
* **CLI end-to-end on a full-size synthetic scene: NOT verified.** `DRY=1 bash run_phase45_synthetic.sh` passed the
  upstream contract for all 10 scenes (types: storage, blade prismatic; the rest revolute). A 4-frame smoke run on
  USB_100109 was started and stopped before finishing (the user runs long jobs); only its config / manifest were written.
  Runtime and peak memory on 30k-Gaussian, 480-view scenes are therefore unmeasured.
