# RigEuler3DGS: full pipeline as implemented, 2026-11-18

This describes the current end-to-end pipeline (Phases 0–5 plus the simulation-asset export), where its code lives,
what is validated and what is not.

**Detailed reports:**
- `docs/PHASE1_NEGATIVE_RAY_CONTAMINATION_REPORT.md`: the Phase-1 change and its ablation;
- `docs/PHASE5_FINAL_SIM_ASSET_EXPORT_REPORT.md` and `docs/PHASE5_FINAL_SIM_ASSET_FORMAT.md`: the export;
- `STATUS.md` (2026-11-09): the v5 / v6 baseline.

## 1. Code layout

| branch | content | role |
|---|---|---|
| `main` (1f61c47) | v6 pipeline + Phase 4/5 (3bacfe5) + specs / GT axis tools | baseline |
| `phase1-occlusion` | first occlusion attempt: negative-ray likelihood −log(1 − A) with q = 1, observable group update | **attempt, superseded** (made robot_dog static drift 7.8 → 35 mm) |
| `phase1-occlusion-contam` | on top of the above: calibrated contamination q, ablation / gate tools, the full-pipeline runner | **current Phase 1** |
| `phase5-sim-asset-export` | Phase-5 last step: 3DGS PLY split + URDF / USD / MJCF; preview-physics tool | **current Phase 5 / export** |
| `full-pipeline-2026-11-18` | merge of the two above, plus the specs | **the full pipeline in one tree** |

**One command, all 8 real captures, Phases 1–5 plus export:**

```bash
bash run_full_real_negray.sh 2>&1 | tee -a ~/full_real_negray.log
```

The env is listed in the script header. In the merged tree, Phases 4–5 run from the same tree.

## 2. The pipeline, stage by stage (real captures)

| stage | what it does | settings in the current runner |
|---|---|---|
| Phase 0 | calibration (sensor model from the static window), covariance-surface bootstrap of the endpoint (30k Gaussians), Phase-0 geometry-consistency repair | **not re-run**; endpoints from `ours_v6_two_cams` (orange_suitecase: `ours_v6_three_cams`), sensor models from `ours_v5_two_cams`, arm `unanimous`. Two-camera scenes read the **pre-registration** poses (shadow datasets in `~/real_poses_before_20261112`), because their endpoints were trained before the 2026-11-12 robot-base registration. |
| Phase 1 | per-frame grouped IEKF tracking of every Gaussian (masked-RGB expected-quad observation + RGB-D), then motion-coherence support extraction | `--phase1-likelihood occlusion-consistent --negative-ray-reliability calibrated --phase1-group-update legacy --phase1-deterministic`, `--velocity-tau 0.5`, canonical partition, mismatch sds rgb 0.04 / depth 0.003, P0 velocity sd 1.0 / 0.316, `--bootstrap-p0`; coherence selection with default (non-cautious) settings |
| Phase 2 | rigid SE(3) trajectory of the moving support | free-space support refinement, 1 round, `replace` |
| Phase 3 | joint identification with geometry fixed; both prismatic and revolute fitted | type = R/L rule (R/L > 10 → prismatic), applied when Phase 4 starts |
| Phase 4 | 4A original membership → 4B new-surface proposals → 4C accepted moving (hash-chained stages) | `--recovery-frame-stride 2` |
| Phase 5 | RGB-D Gaussian training on every view with the joint frozen; articulated PLY | `gaussian_rgbd`, defaults |
| export (last Phase-5 step) | `assets/<scene>/`: two Graphdeco PLYs (bit-identical rows), `asset.yaml`, `object.yaml` (PArtTwin1D), trajectory, lineage | `--asset-export-mode auto`; no physics spec → `BLOCKED_MISSING_PHYSICS` |
| preview URDF / USD / MJCF | `tools/make_preview_physics_spec.py` + `tools/export_phase5_sim_asset.py --asset-revision preview` | convex hulls of the Gaussians, declared density 300 kg/m³, observed-range limits, declared dynamics, gravity unverified → `PHYSICS_PREVIEW_ASSUMPTIONS` |

### Phase-1 observation model (new)

On mask-0 pixels, the old off-mask RGB / alpha terms are replaced by a negative-ray contamination likelihood:

    L_p = −log(1 − q_p A_p),   A_p = Σ_k w_k Φ((D_p − z_k)/σ_p)

**What q is.** `q_p` is the probability that the pixel is a reliable free-space measurement. It is calibrated per camera from the run's own masks and depth (no GT):
- contamination falls with distance to the mask;
- depth edges are UNKNOWN (zero evidence).

**Where it runs.** Energy, gradient, the Gauss–Newton (GN) normal operator, trial acceptance and posterior all use the same model, in a CUDA kernel that keeps the renderer's contributor ordering.

**Group update.** The `observable` group update (`[I;K]·Q·G`) is still available. It is **off**: the ablation showed it caused the static drift.

## 3. Evidence for the current Phase 1 (robot_dog ablation, deterministic, same start state, 40 frames)

| likelihood / group | static median | > 50 mm |
|---|---|---|
| legacy / legacy | 3.5 mm | 1340 |
| legacy / observable | 22.8 | 5245 |
| q = 1 / legacy | 2.3 | 785 |
| **calibrated q / legacy (current)** | **2.3** | **620** |
| calibrated q / observable | 15.1 | 3409 |

In the frozen-state replays, the observable group brakes static velocity at 40–60 % of the legacy rate. The mechanism inside it is not established.

**Full 151 / 174-frame runs** (`ours_v10b_negative_ray_legacy_group`). The gate requires the result to be no worse than the worst of v6 / draw1 / draw2:

| scene | end median | p90 | > 50 mm | gate | joint type |
|---|---|---|---|---|---|
| robot_dog | **3.3 mm** (refs 6.3–7.3) | 59.6 (56–74) | **2633** (2774–3721) | PASS | prismatic (correct) |
| black_laptop | 76.1 (77–80) | **277** (393–472) | 8600 (≤ 8524) | **FAIL** by 76 Gaussians | prismatic (expected revolute) |

The earlier q = 1 / observable run had 43.9 mm / 5426 on black_laptop. **That improvement is not kept** by the current configuration.

## 4. Export validation

On synthetic revolute and prismatic fixtures, the transport matches `Joint.transform` at every recorded frame:

| check | error |
|---|---|
| analytic | ≤ 1e-15 |
| URDF | ≤ 1e-15 |
| MuJoCo 3.3.5 | 4e-16 |
| OpenUSD (float32 joint frames) | 1e-7 m |
| SAPIEN 3.0.3 | 4e-7 m |
| Isaac Sim 5.0 PhysX | 5e-7 m |
| PArtTwin1D adapter (unmodified) | 5e-7 m |

On real runs:
- **Visual and kinematic:** red_fridge, light_blue_box and the other 3 `cautious_real_v2` scenes export bit-identical PLYs, with FK at 1e-16.
- **Preview physics:** checked on red_fridge and black_box; URDF / USD / MJCF all pass validation.
- **Tests:** 23 new export tests and the 12 existing Phase-4/5 tests pass; 30 Phase-1 negative-ray tests pass.

## 5. Batch running now (started 2026-11-18 13:42)

The runner writes to `results_11_10_26/ours_v11_negray_real` (Phases 1–3) and `results_11_10_26/ours_phase45_v11_negray_real` (Phases 4–5 + assets, run id `negray_rs2`). It processes the scenes one at a time. Progress when this was written:
- robot_dog Phases 1–3 finished in 32 min, with the same 2493-Gaussian support as v10b (deterministic);
- robot_dog Phase 4 had started (4A core check MOVING_CONFIRMED);
- the remaining 7 scenes were still pending.

## 6. Open problems / not established

1. **black_laptop:** the static drift is back to v6 level, and the joint type is prismatic where revolute is expected.
   - The observable group improved black_laptop but broke robot_dog.
   - Nothing yet improves both.
2. **Why the observable group brakes less.** It is unexplained. Its drift grows with measurement force, so it is not confined to unmeasured Gaussians.
3. **The positive RGB-D term with an imperfect mask** still moves a static object by about 1.1 mm (synthetic test).
4. **Calibration assumption A2.** It is violated near surfaces continuous with the object (tables, floors). The calibration then discards valid free-space evidence there (conservative).
5. **No measured physics for any real scene.**
   - The URDF / USD / MJCF are previews.
   - On real captures the "static link" contains the whole non-moving scene, so its hull is a block, not the object.
   - Gravity alignment is unverified (two-camera scenes are in the pre-registration frame).
6. **Not run:** ManiSkill's own URDF wrapper, a gsplat render through PArtTwin1D, Isaac contact dynamics, and Isaac on real assets.
7. **The joint type rule** (R/L > 10) is still an evaluation-time rule, applied by the Phase-4 runner.
