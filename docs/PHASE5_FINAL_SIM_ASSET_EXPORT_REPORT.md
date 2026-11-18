# Phase-5 final simulation-asset export: implementation report

- **Date:** 2026-11-18.
- **Spec:** `RigEuler3DGS_PHASE5_FINAL_SIM_ASSET_EXPORT_CODEX.md`.
- **Format reference:** `docs/PHASE5_FINAL_SIM_ASSET_FORMAT.md`.

## 0. Status

| item | status |
|---|---|
| exporter inside RigEuler3DGS, called as the last Phase-5 step | **done** |
| two original-parameter Graphdeco PLYs, bit-identical | **done** (synthetic SH 0/1/3 and real red_fridge / light_blue_box) |
| one canonical model -> URDF / USD / MJCF | **done**; all three validated on synthetic revolute + prismatic |
| exact kinematics vs `Joint.transform` | **done**; every recorded frame, ≤ 1e-15 analytic, ≤ 5e-7 m in every engine |
| `auto` / `required` / `visual_only` / `off`, preflight before training, atomic promotion, idempotence | **done** |
| ManiSkill/SAPIEN runtime, MuJoCo compile + `mj_forward`, OpenUSD read-back, Isaac PhysX articulation | **PASS** on both synthetic fixtures |
| PArtTwin1D consumer (unmodified) | **PASS**: `load_config`, `load_gaussian_ply`, `LinkAdapter` driven by SAPIEN link poses, `parttwin1d.cli.inspect` |
| real-scene **physics** acceptance (red_fridge, light_blue_box) | **NOT RUN**: no measured collision / mass / inertia / joint-stop inputs exist. Only the visual + kinematic asset was exported (`BLOCKED_MISSING_PHYSICS`). |

No real asset is claimed simulation-ready.

## 1. Working tree and reference

- **Worktree:** `~/Joint3DGS/RigEuler3DGS-p45`.
  - It was on branch `phase45` at `3bacfe5`, clean (`git status` empty), so nothing local needed preserving.
  - The work is on the new branch `phase5-sim-asset-export` (commit in section 9).
- **Reference:** the inspected reference is `3bacfe5` itself, so there are **no local-vs-reference differences**.
- **Insertion point** (`tools/train_phase5_rgbd_gaussians.py`): the articulated PLY write was at lines 164–168 and `complete.json` at line 169, as the spec states.
- **Main:** `main` (`1f61c47`) is one commit ahead with docs and GT tools only; nothing it changes is under `recovery/` or `tools/train_phase5*`.
- **PArtTwin1D:** `~/PArtTwin1D` at `713ffe2`, clean before and after. It is used read-only, from its own `parttwin_sim` environment.

## 2. What changed

**Modified:** `tools/train_phase5_rgbd_gaussians.py` (+72 / −5 lines)

- **`add_asset_args`** (new): the `--asset-*` options.
  - `--asset-export-mode` takes `auto|required|visual_only|off`, default `auto`.
  - The others are `--asset-physics-spec`, `--asset-registration`, `--asset-{usd,mujoco,sapien}-python`, `--asset-scene`, `--asset-revision` and `--asset-verify`.
- **`asset_preflight`** (new): `required` only. It validates the spec, the registration (not an unverified gauge) and the USD + MuJoCo runtimes, **after the Phase-4/upstream preflight and before training or creating `--output`**. It also runs in `--dry-run`.
- **`final_asset_export`** (new): called after `write_articulated_ply` and before `complete.json`.
  - `required` turns any failure into a `SystemExit`, so `complete.json` is not written.
  - `auto` / `visual_only` record `EXPORT_FAILED` and keep the trained result.
  - `off` records `asset_export_status: OFF`.
- **`main`:** the completion payload is built first. `complete.json` = payload + `asset_export_status`, `asset_export_mode`, `asset_export_dir` (relative), `asset_export_manifest_sha256`, `asset_readiness`, `asset_missing_inputs`, `asset_export_reused` + `seconds`.
  - The old keys and their meaning are unchanged.
  - No training, checkpoint, topology, loss or joint code was touched.

**New package `recovery/sim_asset_export/`:**

| module | role |
|---|---|
| `load_phase5.py` | Phase-5 preflight. The joint is rebuilt from `final/joint/` with `contracts.load_joint` and checked against the completion fingerprint (file hashes + per-frame transform hash), `joint_unchanged_proof.json`, and the PLY `joint` / `joint_state` elements (exact equality). |
| `ply_split.py` | lossless split by dtype-descriptor row selection; unknown fields are refused; bitwise read-back + reassembly; source hash checked before / after |
| `frames.py` | `R_B_J`, `T_B_J0`, `T_link_ply`, URDF RPY, quaternions, inertia transforms (float64) |
| `schema.py` | typed, strict physics / registration schema; statuses |
| `collision.py` | OBJ I/O, watertightness, exact polyhedral mass properties, convexity, hulls, coverage |
| `physics.py` | link-frame resolution (frame-f inputs moved to the anchor with the frozen joint), plausibility + containment gates, limit provenance |
| `model.py` | the single `CanonicalAsset`; `asset.yaml`, `dynamics.yaml`, `object.yaml` |
| `urdf_export.py`, `mjcf_export.py`, `usd_export.py` + `_usd_writer.py` | the three generators (USD written by a separate pxr interpreter) |
| `validate.py` + `_mujoco_check.py`, `_sapien_check.py` | analytic FK, URDF XML read-back (independent scipy RPY), MuJoCo / SAPIEN / PArtTwin1D in their own interpreters, portability |
| `artifacts.py`, `export.py` | staging, signature, atomic rename, `export_complete.json`, idempotence / revision / tamper detection; `export_phase5_sim_asset` |

**New tools and tests:**

- `tools/export_phase5_sim_asset.py`: post-hoc CLI. It hash-guards the Phase-5 files before and after.
- `tools/check_sim_asset_isaac.py` + `tools/sim_runtimes/_isaac_runtime_probe.py`: optional Isaac PhysX check.
- `tests/test_sim_asset_{ply_split,kinematics,dynamics_schema,urdf,usd,mjcf,atomic_export}.py` and `tests/test_phase5_final_asset_hook.py`.
- Helpers `tests/helpers/{synthetic_phase5_run,sim_runtimes}.py` and the fixture builder `tests/make_sim_asset_fixtures.py`.

**Exporter source hash:** `cf7d198d4d831b49b00cc5e040f63acd7dcc7f9e0879e4757fdc4e6ea77cab52` (`artifacts.code_sha256`). It is part of every asset's input signature.

## 3. Physics: provided versus inferred

**Nothing physical is inferred.** Every mass, COM, inertia, collision surface, joint stop, damping and friction value comes from the physics spec, with its provenance. The only derived values are declared, opt-in rules, each with its own provenance label:

- `density_assumption`: exact polyhedral integrals of a watertight supplied surface;
- `observed_range_preview` limits;
- `convex_hull` collision.

**Test inputs.** The synthetic fixtures and the stapler driver run use **declared fixture physics**: boxes matching the parts. For the stapler they are labelled `estimated`, with gravity alignment `unverified`, which is why that run's status is `PHYSICS_PREVIEW_ASSUMPTIONS`. Not one of these values is a measurement of a real object.

**Coordinate gauge.**

- `B = Wc`.
- `J0` has +Z = axis, origin = the Phase-3 hinge point (revolute) or the moving canonical centroid (prismatic; a gauge that cancels).
- `W = Wc` with gravity **unverified** unless a registration is supplied.

The real exports therefore carry `world_gauge: capture_world`, `gravity_alignment: unverified`.

## 4. Results

### Synthetic fixtures

Built by `tests/make_sim_asset_fixtures.py`, 400 + 250 Gaussians, SH degree 3. They are kept at `results_11_10_26/phase5_sim_asset_export_20261118/synthetic_fixtures/`.

| check | revolute (forward, axis (0.15, 0.95, −0.25), hinge (−0.2, 0.01, 1.98), q ∈ [−0.008, 0.9]) | prismatic (backward anchor, axis (0.10, −0.20, −0.97)) |
|---|---|---|
| status | FULL_PHYSICS_READY | FULL_PHYSICS_READY |
| analytic FK, every frame (15) | 8.9e-16 m, 2.2e-16 rot | ≤ 1e-15 |
| URDF read-back FK | 6.7e-16 | ≤ 1e-15 |
| MuJoCo 3.3.5 `mj_forward` (20 q incl. stops and ±edges) | 4.4e-16; mesh vertices 4.8e-8 m; inertia rel 5.5e-10 | 4.4e-16; 4.8e-8 m; 8.2e-8 |
| OpenUSD 25.5 joint-frame FK | 1.9e-8 m (float32 local frames) | 1.1e-7 m |
| SAPIEN 3.0.3 PhysX link poses (20 q) | 3.8e-7 m/rad | 3.6e-7 |
| PArtTwin1D `LinkAdapter` from SAPIEN poses | means 5.1e-7 m, quats 2.4e-7 | means 4.0e-7 m |
| Isaac Sim 5.0 PhysX articulation (18 q, 1 DOF, limits read back in rad) | 4.7e-7 m, 2.1e-7 rot | 3.5e-7 m, 5.1e-8 rot |
| contact smoke test (probe sphere on the static top) | MuJoCo rests, 0.37 mm penetration; SAPIEN rests, 2e-7 m | same |
| `parttwin1d.cli.inspect` | all OK (rc 0) | all OK (rc 0) |

### Driver end to end

The real `tools/train_phase5_rgbd_gaussians.py` was run on the synthetic stapler Phase-4 chain (`ours_phase4_corrected_synthetic/stapler_103111/phase4/full_rs2_rev`). It was a smoke run: `--smoke-frames 4`, tiny iterations, `NOT_FOR_RESULTS`.

- **Training:** PSNR_fg 26.8 → 30.8 dB, joint unchanged.
- **Export:** `assets/stapler_103111` written as the last step, with `PHYSICS_PREVIEW_ASSUMPTIONS` and URDF / USD / MJCF all PASS; `complete.json` carries the `asset_*` fields.
- **Re-export:** a post-hoc export afterwards was **reused** (identical signature).

The output is at `results_11_10_26/phase5_sim_asset_export_20261118/driver_e2e_stapler/`.

### Real runs

Post-hoc, visual + kinematic only. The output was written to `results_11_10_26/phase5_sim_asset_export_20261118/<scene>/` with `--out-root`, so the historical Phase-5 folders were not touched.

| scene | joint, frames | static / moving Gaussians (unresolved) | split | FK vs `Joint.transform` | time, RSS | status |
|---|---|---|---|---|---|---|
| red_fridge (`ours_phase45_cautious_real_v2`) | revolute, 189 | 38,096 (4,132) / 43,780 (92) | 62 fields bit-identical, SH 3 | 6.7e-16 m | 3.6 s, 0.68 GB | BLOCKED_MISSING_PHYSICS |
| light_blue_box (same) | prismatic, 166 | 91,315 (5,503) / 38,059 (342) | bit-identical, SH 3 | 2.2e-16 m | 3.9 s, 0.71 GB | BLOCKED_MISSING_PHYSICS |

PArtTwin1D's `load_gaussian_ply` reads all four real PLYs. Its `load_config(check_files=False)` parses both `object.yaml` files. With file checks on, it correctly refuses them, because there is no URDF without physics.

**A real-scene observation:** red_fridge's static link holds 4,132 of the 4,224 unresolved rows. `lineage.npz` marks them as uncertain membership.

## 5. Tests

Run with `RIGEULER_SIM_PYTHON=<venv with mujoco + usd-core>` and `RIGEULER_SAPIEN_PYTHON=<parttwin_sim python>`. Each file is a script: `python tests/<file>.py`.

| file | result | covers |
|---|---|---|
| test_sim_asset_ply_split | 4/4 | gate A: bit identity (SH 0 / 3, matched by UID), order / dtype / no extras, partition, unresolved descendants, 7 input errors, NOT_FOR_RESULTS warning, repo reader + PArtTwin1D reader |
| test_sim_asset_kinematics | 5/5 | gate B: R_B_J determinism / handedness (208 axes); every frame of revolute + prismatic, forward + backward, negative q, non-origin hinge; q0 identity; covariance / quaternion / SH invariance; URDF RPY and quaternion conventions vs scipy (including gimbal lock); USD degrees; observed range ≠ stops |
| test_sim_asset_dynamics_schema | 3/3 | gate C: 22 rejections (mass ×2, COM, inertia ×3, stops ×3, effort limit, typo, RGB-D without EE exclusion, mm-as-m, missing / wrong frame, scaled / reflecting SE(3), swapped links, wrong pose, empty / corrupt mesh, non-convex without parts); scale / placement / transport / COM / inertia / density on known boxes; decomposition coverage; hull = 7/8 of the L's box |
| test_sim_asset_urdf | 2/2 | gate D: URDF text; SAPIEN runtime; PArtTwin1D adapter; moved folder (run deleted) still loads |
| test_sim_asset_usd | 2/2 | gate D: crate magic, schema, units, FK, re-open from a moved folder; no USD runtime -> INCOMPLETE_RUNTIME, `required` fails |
| test_sim_asset_mjcf | 1/1 | gate D: compile, 1 DOF, hinge / slide, inertia, mesh vertices, contact, moved folder |
| test_sim_asset_atomic_export | 4/4 | failure injection (no asset dir, no staging, no `export_complete.json`); reuse; refusal; revision; tamper; every status; post-hoc CLI non-mutating |
| test_phase5_final_asset_hook | 2/2 | gate E: real CUDA Phase-5 training -> driver hook -> FULL_PHYSICS_READY; Phase-5 files and joint fingerprint unchanged; required / auto failure injection; `required` stops `main()` before training |
| test_phase5_training, test_phase45_contracts, test_phase45_direction_anchor | 5/5, 2/2, 5/5 | existing regressions, unchanged |

## 6. Runtimes used

| runtime | version | how |
|---|---|---|
| training env | Python 3.11.15, plyfile, PyYAML 6.0.3, numpy / scipy 1.17.1 | exporter main process (no simulator import) |
| MuJoCo | 3.3.5 | new isolated venv `~/.venvs/rigeuler-sim-export` (created for this work; numpy 2.2.6) |
| OpenUSD | usd-core 25.5.1 (pxr 0.25.5) | same venv. Isaac's own pxr 0.24.5 + PhysxSchema also works through its kit python with its library paths. |
| SAPIEN / ManiSkill env | sapien 3.0.3 (mani_skill 3.0.1 installed) | existing `parttwin_sim` env, read-only |
| PArtTwin1D | 0.4.0 @ 713ffe2 | same env, read-only, unmodified |
| Isaac Sim | 5.0.0-rc.45 | `~/isaac-sim/python.sh`, headless, only through `tools/check_sim_asset_isaac.py` |

## 7. Not run, or not established

- **Real-scene physics acceptance:** not run, because no measured inputs exist. It needs a physics spec per scene: collision surfaces in the capture frame, masses / COM / inertia, physical stops, joint dynamics and a gravity registration.
- **ManiSkill's own loader:** the URDF went through **SAPIEN's** URDF loader, which ManiSkill builds on, and PArtTwin1D's loaders / adapter / `inspect`. ManiSkill's own `URDFLoader` wrapper, a ManiSkill env and a gsplat GPU render through PArtTwin1D were **not** run.
- **Isaac:** PhysX articulation kinematics were checked on the two fixtures. Isaac contact dynamics, and Isaac on any real asset, were not run. The exporter itself always reports `isaac_runtime: NOT_RUN`; the check is a separate, manual tool.
- **`collision.*.source_kind: reconstructed_rgbd`:** only the declaration gate exists. It refuses without end-effector-exclusion evidence. The RGB-D surface reconstruction algorithm (spec 5, optional) is **not implemented**.
- **Actuation:** only a passive joint is implemented. An actuator mapping is refused with an explicit error.

**Engine differences** are quantified, not equated:

- SAPIEN 3.0.3 ignores the URDF velocity limit (loads 100) and treats URDF `friction` as a PhysX joint-friction coefficient.
- MuJoCo has no velocity limit, a single sliding friction coefficient and soft contacts (0.37 mm penetration vs 2e-7 m in PhysX).
- USD does not map Coulomb joint friction, and stores joint local frames in float32 (~1e-7 m).

**Behaviour change.** The default `--asset-export-mode auto`, as the spec requires, means **every new Phase-5 run now also writes `assets/<scene>/`** (seconds; no model files without physics). This departs from this repository's usual opt-in convention for new behaviour. `--asset-export-mode off` restores the old output exactly, apart from the recorded `asset_export_status: OFF`.

**Plausibility gates.** The extent ratio, centre offset and containment of the link's own Gaussians (≥ 50 % within 5 % of the extent) are declared defaults, overridable in the spec's `validation:` block. They reject inconsistent inputs; they do not certify that a mesh is accurate.

**Bugs found while validating the validators:**

- the MuJoCo parent–child contact between a world-welded parent and its child (now excluded explicitly in all three formats);
- symmetric rounding of rotated inertia (now symmetrised once for every format);
- `.usd` reports format id `usd` (crate detected by its magic bytes instead);
- MjSpec leaving a probe's `ipos` at the body position (test harness only).

## 8. Commands

```bash
cd ~/Joint3DGS/RigEuler3DGS-p45
export RIGEULER_SIM_PYTHON=~/.venvs/rigeuler-sim-export/bin/python          # USD + MuJoCo
export RIGEULER_SAPIEN_PYTHON=~/miniconda3/envs/parttwin_sim/bin/python     # optional engine / consumer check

# a normal Phase-5 run with the asset as its last operation
python tools/train_phase5_rgbd_gaussians.py --phase4-dir P4 --output P5 --train-mode gaussian_rgbd \
    --asset-export-mode required --asset-physics-spec /abs/physics_spec.yaml [--asset-registration /abs/reg.yaml]

# re-export a completed run without training (never modifies it)
python tools/export_phase5_sim_asset.py --phase5-dir P5 --asset-physics-spec /abs/physics_spec.yaml \
    --asset-export-mode required [--asset-revision NAME]

# synthetic fixtures + optional Isaac check
python tests/make_sim_asset_fixtures.py --out NEW_DIR --isaac-python ~/isaac-sim/python.sh
python tools/check_sim_asset_isaac.py --asset-dir P5/assets/<scene> --isaac-python ~/isaac-sim/python.sh
```

## 9. Commit

Branch `phase5-sim-asset-export` in `~/Joint3DGS/RigEuler3DGS-p45`; the commit id is reported in the hand-over message.
