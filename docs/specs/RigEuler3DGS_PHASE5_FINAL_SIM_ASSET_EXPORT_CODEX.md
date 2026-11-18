# CODEX IMPLEMENTATION DIRECTIVE — Final Phase-5 simulation-asset export INSIDE RigEuler3DGS

**Goal:** When RigEuler3DGS finishes Phase 5, it must save a self-contained articulated digital-twin asset: two **standard Graphdeco 3DGS PLYs**, a single canonical physics/kinematics manifest, and simulator exports **URDF (ManiSkill/SAPIEN), USD (Isaac Sim/Isaac Lab), and MJCF (MuJoCo)**. This is **RigEuler3DGS's own final output-saving step**, not a task to implement in PArtTwin1D. PArtTwin1D is only a downstream consumer and must require **zero modifications** for the exporter to be considered implemented.

**Working repository:** `~/Joint3DGS/RigEuler3DGS-p45`, or whichever working tree contains the current Phase-4/5 training code. **Read `git status`, identify the active worktree/branch and preserve uncommitted changes.** Inspected reference: `SamueleMara/RigEuler3DGS`, `phase45`, commit `3bacfe51771efccf37228ed2f27a8b30bb509710` (2026-11-16). Source interfaces and line numbers below are grounded in that commit; if the local tree differs, trace the changed call graph and document the exact deltas before editing.

## 0. NON-NEGOTIABLE SCOPE

1. **Implement entirely in RigEuler3DGS**: add `recovery/sim_asset_export/`, `tools/export_phase5_sim_asset.py`, tests, and a call from the **end of** `tools/train_phase5_rgbd_gaussians.py`. Export must work without importing PArtTwin1D, ManiSkill, Isaac Sim or MuJoCo during Phase-5 training.
2. **Do not modify, resample, train, reorient, recenter, normalize, or rewrite the numerical Gaussian attributes.** Only losslessly select rows into two ordinary 3DGS PLY files; the original Phase-5 articulated PLY stays intact. Preserve `x,y,z,nx,ny,nz,f_dc_*,f_rest_*,opacity,scale_*,rot_*` if present, with unchanged values/order and quaternion convention. Do not output point clouds, meshes or `.splat` files in place of either 3DGS PLY.
3. **Two rigid links, one DOF**: `static_link` (root) and `moving_link` (child), connected by Phase-3's frozen `revolute` or `prismatic` joint. Don't estimate a new hinge, axis, direction, q trajectory or motion limits during export.
4. **One canonical exporter data model** generates URDF, USD and MJCF; no independent hand-edited model descriptions.
5. **Simulator physics drives the visuals**: each renderer must read live simulator link poses and apply `T_link_ply`; never animate PLY Gaussians with a second q interpolator when physics is running.
6. **Phase 5 does not measure rigid-body masses, inertias, collision surfaces or friction**. Obtain them from a user-provided physics description or an explicitly selected and validated approximation. Never silently manufacture physically measured values. The export reports `FULL_PHYSICS_READY` only when mandatory inputs and validation are present.
7. **Do not implement Phase 4, Phase 3, new joint fitting, new Phase-5 training, or changes to PArtTwin1D**. This change is a final artifact-writing stage after successful Phase-5 optimization.

## 1. Code-grounded input contracts — inspect/reuse, do not duplicate

### 1.1 Phase-5 driver: exact insertion point

`tools/train_phase5_rgbd_gaussians.py`, inspected commit:

- `preflight(phase4_dir)` verifies complete 4A and 4C artifacts, input hashes, frozen joint.
- `train_phase5(g, up.joint, views, cams, cfg, out, manifest=manifest)` performs training and saves `out/final/point_cloud.ply`, lineage NPZs, joint proof, and joint copies.
- **Lines 164–168** then call `recovery.articulated_ply.write_articulated_ply(...)`, adding per-Gaussian `part_id`, UID/lineage, frozen-joint metadata and joint_state rows to `out/final/point_cloud.ply`.
- **Line 169** writes `out/complete.json`.

**Insert `export_phase5_sim_asset(...)` after the articulated PLY is successfully written and validated, but before `out/complete.json` is committed.** Do not change numerical Phase-5 output or the existing joint fingerprint. Record the asset export status and manifest hash in Phase-5 `complete.json`. On an existing completed Phase-5 run, provide a separate **idempotent, non-mutating export CLI**; it must never rewrite `complete.json`, `final/point_cloud.ply`, saved joint files or trained tensors.

### 1.2 Articulated PLY: single source of truth for labels and motion

`recovery/articulated_ply.py`:

- `read_articulated_ply(path)` returns `part_id`, `gaussian_uid`, `unresolved`, `joint_type`, `axis`, `origin`, `q_range`, `anchor_frame`, original `frames`, `q`.
- The full PLY has a `vertex` element with original standard 3DGS fields **plus** `part_id`, `gaussian_uid`, `original_id`, `parent_uid`, `provenance`, `unresolved`; one `joint` element; and a `joint_state` element.
- `part_id=0` = static; `part_id=1` = moving; canonical geometry corresponds to **anchor frame q=0**. `q_min/q_max` in the PLY are the **observed demonstration range**, **not certified mechanical joint stops**.
- `recovery.gaussians.GaussianTensors.load_lineage(out/final)` supplies the original aligned labels and UIDs. Preserve unresolved flags in a separate provenance file. The frozen render assignment of unresolved rows may be exported, but **must be marked uncertain**; never misrepresent it as confidently identified object membership.

Use `plyfile.PlyData` to split the structured vertex array **without decoding and re-encoding** opacity, log-scales, quaternion components or SH. Do **not** use `GaussianTensors.save_ply()` for the split unless field-by-field bit-exactness of all original stored fields is proven. Exclude the extra articulated metadata elements/fields from each output PLY; put UID/lineage/membership mapping in sidecars instead. Keep `final/point_cloud.ply` unchanged.

### 1.3 Fixed joint model: exact kinematic oracle

`recovery/contracts.py:Joint` defines:

```python
joint.type          # 'revolute' or 'prismatic'
joint.axis          # unit 3-vector in capture/canonical world Wc
joint.point         # point on hinge line for revolute, None for prismatic
joint.frames        # original frame ids, processing order
joint.q             # q[0] == 0 at anchor; radians or metres
joint.transform(f)  # returns (Q_f,p_f): x(f) = Q_f @ x0 + p_f
joint.chronological()
```

For revolute: `Q_f = Exp([axis]x q_f)`, `p_f = (I-Q_f) point`; for prismatic: `Q_f=I`, `p_f=q_f axis`. Use `Joint.transform(frame)` or the exact articulated PLY joint state as the reference in tests. Verify `joint_unchanged_proof.json`, `input_proof.json`, source hashes and `complete.json` as available. Never infer joint geometry from Gaussian centroids when a revolute `joint.point` is provided.

### 1.4 Visual convention in the destination ecosystem

PArtTwin1D already uses `T_W_P = T_W_L @ T_L_P` and evaluates SH in the PLY's **canonical local frame** (`src/parttwin1d/rendering/link_adapter.py`); its consumer config requires explicit `T_world_root` and per-link `T_link_ply`. Generate a compatible `object.yaml` automatically, but **do not edit that repository**. For optional consumer tests, invoke its existing CLI in a separate environment.

## 2. REQUIRED final output directory

For a Phase-5 run `P5=<output passed to tools/train_phase5_rgbd_gaussians.py>` and scene name `<scene>` taken from its verified upstream `setup.source`, write:

```text
<P5>/
├── final/
│   ├── point_cloud.ply                 # existing single articulated Phase-5 PLY: DO NOT CHANGE
│   ├── joint_unchanged_proof.json       # existing
│   └── ...                              # existing lineage and joint files
├── assets/
│   └── <scene>/
│       ├── asset.yaml                  # SINGLE canonical shared asset specification
│       ├── object.yaml                 # PArtTwin1D loader-compatible consumer config
│       ├── joint_trajectory.csv        # original chronological frame, q, posterior_sd_q
│       ├── export_manifest.json        # hashes, sources, flags, evidence/status
│       ├── lineage.npz                 # source row + UID + part_id + unresolved per PLY row
│       ├── visual/
│       │   ├── static.ply              # ordinary Graphdeco 3DGS PLY, original stored attributes
│       │   └── moving.ply              # ordinary Graphdeco 3DGS PLY, original stored attributes
│       ├── physics/
│       │   ├── collision/
│       │   │   ├── static.obj          # collision mesh in static_link frame
│       │   │   └── moving.obj          # collision mesh in moving_link frame
│       │   ├── dynamics.yaml           # provided / declared estimated physics + provenance
│       │   └── collision_report.json
│       ├── maniskill/model.urdf
│       ├── isaac/model.usd
│       └── mujoco/model.xml
└── complete.json                       # existing Phase-5 completion extended only in new runs
```

**Important:** The `physics/` mesh and dynamics files, and therefore the three **physics-ready** model files, require validated physics input. If unavailable, export the two PLYs, canonical joint/trajectory, `asset.yaml`, compatible `object.yaml` (with readiness clearly marked), and a `BLOCKED_MISSING_PHYSICS` export status. Do **not** write fake collision meshes, masses or a deceptive `FULL_PHYSICS_READY` flag. `--asset-export-mode required` must fail Phase-5 completion *at preflight* if these mandatory inputs are missing; `auto` must preserve successfully trained Phase-5 results and report the independent exporter limitation.

All paths within exported files must be relative to the asset directory or file in which they occur, and work after moving the entire `<scene>/` folder. No absolute workstation paths in runtime model references.

## 3. ONE mathematically exact frame convention (do not guess)

Define column-vector transforms `T_A_B` mapping coordinates in frame B to frame A. `Wc` is the Phase-5 canonical capture world (the XYZ coordinates stored in **both** output PLYs). `B` = URDF static_link frame, chosen **equal to Wc**. `J0` = moving_link frame at the canonical Phase-3 anchor `q=0`.

Let `a = joint.axis`, `||a||=1`. Define `R_B_J` deterministically as an orthonormal, right-handed matrix whose **local +Z axis** equals `a`. Choose the auxiliary reference vector with the smallest absolute dot product with `a` to avoid degeneracy; fix handedness and validate det=+1. Let `o` be:

- revolute: **exactly** `joint.point` (a point on the measured hinge axis);
- prismatic: a **documented frame gauge**, e.g. moving canonical XYZ centroid, not a fitted hinge location and not a physical result. The choice cancels from the translation kinematics.

Then:

```text
T_B_J0 = [ R_B_J, o ; 0 0 0 1 ]
T_B_Pstatic = I
T_J0_Pmoving = inv(T_B_J0)        # CRITICAL: PLY coordinates remain in Wc/B
T_B_Lmoving(q) = T_B_J0 @ M_z(q)
T_B_Pmoving(q) = T_B_Lmoving(q) @ T_J0_Pmoving
                 = T_B_J0 @ M_z(q) @ inv(T_B_J0)
```

`M_z(q)` is pure rotation about +Z by `q` (revolute) or translation along +Z by `q` (prismatic). Hence:

```text
revolute:  x_B(q) = Exp([a]× q) (x_B(0) - o) + o
prismatic: x_B(q) = x_B(0) + q a
```

These are **identical** to `Joint.transform(f)`. At q=0, both links place their unchanged stored 3DGS PLY means at their original canonical Wc coordinates.

For the simulator world `W` choose an explicit rigid `T_W_B = T_W_Wc` supplied by calibrated capture-to-simulator registration, or explicitly select the gauge `W=Wc` and label it as an **unverified gravity alignment**. No hidden frame flips or scale inside SE(3). For a dynamics-ready asset, gravity alignment must be calibrated/declared rather than guessed from COLMAP axes. `object.yaml` must include `T_world_root=T_W_B` and the two `T_link_ply` matrices above.

Transform each Gaussian for rendering from live link pose, with **no additional q animation**:

```text
T_W_Pi = T_W_Li_physics(q) @ T_Li_Pi
mu_W = R_W_P mu_P + t_W_P
quat_W = quat(R_W_P) ⊗ quat_P    # scalar-first, Hamilton
Sigma_W = R_W_P Sigma_P R_W_P.T
SH is evaluated along view direction in original PLY frame P, not world frame
```

**Non-negotiable tests**: compare these world means and rotations, for *every original Phase-3 frame q* and random canonical points, against `recovery.contracts.Joint.transform(f)`; validate axis/rotation, both directions, q0 identity, and no residual frame rotation when the camera and link move together. The validation tolerance should reflect float64 analytic vs float32 simulation, and be declared in reports.

## 4. Split Phase-5 PLYs without changing the Gaussians

Implement `recovery/sim_asset_export/ply_split.py`:

```python
# Reference-level pseudocode; implement all schema validations.
src = PlyData.read(str(phase5_final_ply))
v = src["vertex"].data
p = np.asarray(v["part_id"], dtype=np.uint8)
assert np.isin(p, [0, 1]).all()
assert p.sum() > 0 and (p == 0).sum() > 0
# Select the ORIGINAL 3DGS stored columns and their ORIGINAL dtype, not decoded values.
# Preserve original sequence of 3DGS properties, including nx/ny/nz if present.
static_rows = v[p == 0][gaussian_fields]
moving_rows = v[p == 1][gaussian_fields]
# Write binary PLY with one vertex element and no joint/lineage extras.
# Compare every output field to v[p==k][field] with np.array_equal.
```

Technical details:

- A NumPy structured array must be constructed by selecting dtype descriptors, **not** the erroneous `v[mask][fields]` expression if unsupported by the particular structured-array API. Preserve exact numeric types and values; never roundtrip through float32-activated Gaussians.
- Include *every* original 3DGS numeric field in the output. Reject fields that cannot be classified instead of silently dropping potentially important visual data. Maintain SH `f_rest_*` ordering and `rot_0..3` exactly.
- Save index/UID/lineage mapping to `lineage.npz`, with separate arrays for static and moving rows. Include unresolved status and provenance.
- Verify `N_static+N_moving=N_phase5` and per-field partition/reassembly equality using retained original row indices. SHA-256 the source and both result files.
- Do not treat the articulated input PLY as a standard per-link PLY: the split output must be supported by ordinary Graphdeco/gsplat readers that ignore no hidden custom rows.
- Do not alter the original `final/point_cloud.ply` to split it; it is immutable.

## 5. Canonical physics manifest — no guessed physical constants

Implement strict typed dataclasses and validated YAML input in `recovery/sim_asset_export/schema.py`.

Inputs for complete physical export (explicit file `--asset-physics-spec FILE`):

```yaml
schema_version: 1
# This is a SCHEMA EXAMPLE, not measured values for red_fridge.
registration:
  T_world_capture: null   # a measured 4x4 SE(3), OR explicit capture-world gauge selection
  gravity_alignment: measured_or_declared
collision:
  static:
    source: /path/to/static_surface.obj
    T_capture_mesh: [[...], [...], [...], [...]]
    provenance: measured
  moving:
    source: /path/to/moving_surface.obj
    T_capture_mesh: [[...], [...], [...], [...]]
    provenance: measured
links:
  static_link:
    mass_kg: null
    com_capture_m: null
    inertia_com_capture_kg_m2: null  # symmetric 3x3, positive-definite
    provenance: measured_or_estimated
  moving_link:
    mass_kg: null
    com_capture_m: null
    inertia_com_capture_kg_m2: null
    provenance: measured_or_estimated
joint:
  lower: null          # PHYSICAL stops; never silently use q_demo_min
  upper: null
  velocity_limit: null
  effort_limit: null
  damping: null
  coulomb_friction: null
  actuation: null     # if known; parameter mapping documented by simulator
  provenance: measured_or_estimated
```

Do not use the `null` values literally as simulator physics. Validate each field with clear actionable errors. A *declared approximation mode* may estimate collision and inertia only if its assumptions are explicit, inputs are sufficient and its output is marked `estimated`, not `measured`:

- **Collision meshes**: prefer externally supplied, calibrated object-only meshes. Optional separate `--collision-mode reconstructed_rgbd` can fuse **calibrated original RGB-D object surface** transported into the anchor with the frozen joint, exclude unsupported/mixed-depth pixels, **exclude robot EE contamination** (cannot solve solely from the binary mask), and produce mesh/proxy coverage and uncertainty reports. This is an additional opt-in geometry algorithm and must not silently run as part of the export's default physics. If the data cannot isolate object surfaces from EE, report `COLLISION_UNRESOLVED` rather than include the gripper.
- **Inertial parameters**: if supplied explicitly, preserve them and rotate inertia from declared source frame to link frame. If an explicit material density is provided and the collision volume is verified watertight, optionally estimate mass, COM and inertia by volume integration; provenance must say `density_assumption`. No arbitrary default mass or diagonal inertia.
- **Collision approximations**: if source meshes are nonconvex, generate validated convex decomposition/convex proxies suitable for MuJoCo/PhysX, preserving separately the measured source mesh. If this cannot be done robustly, fail physics readiness. Don't use 3DGS Gaussian ellipsoids as physical collision bodies.
- **Joint limits**: the demo's `min(q),max(q)` are `observed_range` and can be offered as explicitly labeled preview limits only after an **opt-in** assumption; physical stops must come from physics input.
- All source-to-link transform, mesh watertightness, volume, COM, inertia eigenvalues, scales, and hull count must be audited.

Write canonical `asset.yaml` including exact geometry/joint binding, `source_phase5`, `anchor_original_frame`, scene/direction, `T_B_J0`, `T_static_link_ply`, `T_moving_link_ply`, canonical q and full recorded trajectory reference, provenance/hashes and readiness flags. Write `physics/dynamics.yaml` from that validated single source. All three simulator exporters **read this same canonical validated object**, never each other's output.

## 6. Simulator exports

### 6A. ManiSkill / SAPIEN: `maniskill/model.urdf`

- `<robot name=...>` with exactly two `<link>` elements (`static_link`, `moving_link`) and one `<joint>` with Phase-3 `revolute` or `prismatic` type. Root is fixed via simulator setup, not by inventing a world link unless required by the target loader.
- `<joint name="articulation_joint" type="...">`: parent static_link, child moving_link; `<origin xyz="o_B" rpy="R_B_J0">`; `<axis xyz="0 0 1">`; verified physical `<limit lower upper effort velocity>` in rad/metres and SI units. `<dynamics>` from explicit model. **Do not substitute observed min/max as mechanical limits without declared opt-in.**
- `<collision>` refer to `../physics/collision/static.obj` and `../physics/collision/moving.obj`; mesh vertices must already be in the relevant link frame, or origins must encode validated transforms. `<inertial>` mass, COM and symmetric inertia in link-local coordinates; if necessary write inertial origin rotation and transform tensor correctly.
- The PLYs are **not** URDF `<mesh>` visual geometry; do not request PhysX to interpret them as triangle meshes. Visual binding is `asset.yaml` / `object.yaml`, and PArtTwin1D's existing gsplat renderer reads `T_link_ply`.
- Generate `object.yaml` compatible with the existing PArtTwin1D `schema_version: 1` loader (`urdf`, `fixed_base`, `T_world_root`, link PLYs and explicit `T_link_ply`, optionally cameras and `initial_q`). Keep the loader's fields exact and relative paths correct.

### 6B. Isaac Sim / Isaac Lab: `isaac/model.usd`

- Use `pxr.Usd`, `UsdGeom`, `UsdPhysics`, `Sdf`, `Gf`, and optional `PhysxSchema` through a **separate export interpreter/environment**, not by installing Isaac Sim into the training environment. Generate a real parseable USD stage (`model.usd`), not an XML file or text file renamed `.usd`.
- Create explicit rigid-body prims with collision meshes, correct mass and inertia, one `UsdPhysics.RevoluteJoint` or `UsdPhysics.PrismaticJoint`, and an ArticulationRootAPI at the proper articulation prim. Write `physics:body0/body1`, verified local joint anchors `localPos0/localRot0/localPos1/localRot1`, axis `Z`, and joint limits. For revolute, apply the **USD angle-unit conversion explicitly**: Phase-3 q and URDF limits in radians, USD revolute limit attributes in degrees. Ensure initial q=0 corresponds to the Phase-5 canonical anchor.
- Preserve original canonical PLYs as external visual assets and record asset references and `T_link_ply` in documented metadata/custom attributes; do **not** claim stock USD can render Graphdeco 3DGS without a Gaussian-renderer adapter. Any converted Gaussian USD representation is an optional generated cache **outside** `visual/static.ply` and `visual/moving.ply`.
- Validate by opening the USD with OpenUSD; when Isaac is available, instantiate PhysX articulation, verify 1 DOF, link transform and collision behavior. Missing USD runtime must be reported `NOT_RUN`, not `PASS`.

### 6C. MuJoCo: `mujoco/model.xml`

- Generate proper MJCF `<mujoco>`, `<compiler angle="radian">`, `<asset><mesh .../></asset>`, nested `<worldbody><body name="static_link"><body name="moving_link"> ...` and a single `joint type="hinge"` or `joint type="slide"`, `axis="0 0 1"`, at the child frame joint location. Use `pos/quat` corresponding to `T_B_J0`; all mesh vertices must match the expected link frames. Set root static/fixed as appropriate.
- Explicit `<inertial>` with mass, COM and principal axes/full inertia representing the same physical tensor. Map viscous damping and Coulomb friction using clearly documented engine-specific semantics; do not claim identical numerical dynamics when the parameterizations differ.
- For collisions, use supplied convex decomposition/proxies and document that MuJoCo's mesh collision implementation uses convex geometry (a raw concave OBJ is **not** equivalent to exact triangle-surface collision).
- Do not use either 3DGS PLY as an MJCF visual mesh; save the unchanged PLY references/`T_link_ply` in `asset.yaml` for an **external gsplat adapter** reading MuJoCo `xpos/xquat/xmat` of the actual physics links.
- Validate by compiling `model.xml` with the actual `mujoco.MjModel.from_xml_path` and evaluating `mj_forward` at q=0 and representative demo q values.

### 6D. Document differences instead of claiming identical physics solvers

URDF, USD and MJCF must share: canonical link frames, joint type/axis/anchor, reference configuration, supplied mass/COM/inertia, collision source geometry, and reported parameter provenance. Contact friction, joint friction, damping, motor and solver conventions differ. Export transformations must be specified and read back independently, and differences must be quantified rather than silently ignored.

Official API anchors to consult during implementation: OpenUSD/Isaac USD joint local frames and limits; MuJoCo MJCF `hinge`/`slide`, inertial and collision semantics. Keep full source version and APIs in the export report.

## 7. Implemented modules and function boundaries

Create:

```text
recovery/sim_asset_export/
  __init__.py
  schema.py                # validated typed canonical asset + physics provenance
  load_phase5.py           # Phase-5 complete/joint/PLY/lineage/hash preflight
  ply_split.py             # immutable, lossless original-field row selection
  frames.py                # deterministic SE(3), R_B_J0, T_link_ply, inertia transforms
  collision.py             # provided mesh ingest/validation (+ optional separate RGB-D mode)
  urdf_export.py           # URDF generator
  usd_export.py            # USD generator via dedicated subprocess/runtime
  mjcf_export.py           # MJCF generator
  validate.py              # analytic FK, PLY parity, XML/USD/MJCF read-back
  artifacts.py             # atomic staging, hashes, manifest and completion
  export.py                # single API: export_phase5_sim_asset(...)
tools/export_phase5_sim_asset.py
```

Public API:

```python
export_phase5_sim_asset(
    phase5_dir: Path,
    scene: str | None = None,
    physics_spec: Path | None = None,
    registration: Path | None = None,
    mode: Literal["auto", "required", "visual_only"] = "auto",
    out_root: Path | None = None,
    usd_python: Path | None = None,
    verify: bool = True,
) -> ExportResult
```

`ExportResult`: `scene`, `asset_root`, `status`, `visual_complete`, `physics_complete`, `usd_complete`, `missing_inputs`, `hashes`, `validation_report`, `NOT_FOR_RESULTS`.

The CLI must support **post-hoc export of completed runs without training**:

```bash
cd ~/Joint3DGS/RigEuler3DGS-p45
conda activate rigeuler3dgs
python tools/export_phase5_sim_asset.py \
    --phase5-dir "/absolute/path/to/<scene>/phase5/<run>" \
    --asset-physics-spec "/absolute/path/to/measured_dynamics_and_collision.yaml" \
    --asset-export-mode required
```

And Phase-5's existing CLI must accept the same relevant options, e.g.:

```bash
python tools/train_phase5_rgbd_gaussians.py \
    --phase4-dir "..." --output "..." --train-mode gaussian_rgbd \
    --asset-export-mode auto \
    --asset-physics-spec "/absolute/path/to/physics_spec.yaml"
```

`auto` should be default so the visual+kinematic asset is always written as a last step; when valid physics is supplied it upgrades to all three physics formats. `required` preflights full-physics availability **before expensive training** and fails without complete URDF/USD/MJCF. `visual_only` intentionally exports no physics and records why. `off` may be retained as an explicit compatibility escape hatch for legacy runners, but must never be silently selected. Document exactly how the options propagate through `run_phase45_*` shell runners without changing earlier phase behavior.

### Safe training integration / transaction semantics

1. **Early preflight**: if `required`, check physics spec and required exporter runtimes *before* expensive training. Never train for hours before discovering a missing mandatory physics file.
2. Train and save everything as before. Do not alter Phase-5 checkpointing, topology, joint freeze, hashes, or loss.
3. After `write_articulated_ply(...)`, call exporter. Construct exports in a temporary sibling directory, validate, then atomically promote to `<P5>/assets/<scene>/`; do not leave a directory looking complete after an exception. Write `export_complete.json` inside the asset only on successful requested-level validation.
4. For `auto` with missing physics, finalize only the **visual/kinematic artifact**, with explicit `BLOCKED_MISSING_PHYSICS`, without fake model files; Phase-5 training itself may still report success, but its `complete.json` must record this export status separately.
5. Append `asset_export_status`, relative `asset_export_dir`, `asset_export_manifest_sha256`, and readiness flags to Phase-5 `complete.json` only for **new** runs. Do not alter the old completion schema's meaning (`stage=phase5`, original proof, `NOT_FOR_RESULTS`).
6. For a pre-existing completed Phase-5 run, the CLI may add a new **independent** asset directory with its own hash/manifest; **never** patch the original Phase-5 `complete.json`. Idempotence: identical inputs -> verify and reuse; changed physics -> explicit new revision or refuse overwrite, no silent replacement.
7. Never write into PArtTwin1D or symlink to mutable training results as the canonical visual asset. The `<scene>` folder must be copyable to another machine without the RigEuler checkout.

## 8. Tests — mandatory, real engine validation where available

Use repository's established `tests/test_phase5_training.py`, `tests/test_phase45_contracts.py`, `tests/test_phase45_direction_anchor.py` as regressions. Add:

```text
tests/test_sim_asset_ply_split.py
 tests/test_sim_asset_kinematics.py
 tests/test_sim_asset_dynamics_schema.py
 tests/test_sim_asset_urdf.py
 tests/test_sim_asset_usd.py
 tests/test_sim_asset_mjcf.py
 tests/test_sim_asset_atomic_export.py
 tests/test_phase5_final_asset_hook.py
```

### Test gate A — Input/PLY correctness (CPU)

- Build synthetic Phase-5 **articulated** PLY with complete SH degree 0 and degree 3, nonunit stored `rot` values, log-scale, opacity logits, per-row unique values; split into PLYs; verify every original field has **exactly equal stored values**, same order, same dtype, correct part membership, no extras and no duplicated/missing UIDs. Original file hash before/after identical.
- Verify unresolved and descendant lineage mapping survives to sidecars; impossible labels, zero Gaussians in a part, missing joint element, bad anchor q, corrupted source hash, incomplete/NOT_FOR_RESULTS status produce explicit errors or honest warnings as appropriate.
- Confirm both PLYs load in unchanged standard 3DGS reader and downstream PArtTwin `load_gaussian_ply` without requiring special metadata handling.

### Test gate B — Exact kinematics (CPU float64)

- For synthetic revolute and prismatic joints (including arbitrary hinge axis, non-origin hinge point, negative q, backward anchor frame), prove:

```text
T_B_Lmoving(q_f) @ T_Lmoving_Pmoving @ [x_original;1]
 == [Q_f @ x_original + p_f;1]
```

for every recorded Phase-3 frame and randomly generated canonical Gaussian points. At q=0, transformed means and orientation equal original canonical data. Static Gaussian world transforms invariant in B. Compare Gaussian covariance rotations and canonical SH directional invariance.
- Verify import conventions for URDF fixed-axis RPY, USD joint local frames and MuJoCo body quat; explicit degrees/radians checks, no axis sign flips, physically correct revolute point invariance and prismatic direction.
- Verify demo motion range and mechanical limits remain distinct in output.

### Test gate C — Physics inputs (CPU)

- Reject missing/negative mass, nonfinite COM, nonsymmetric/nonpositive inertia, invalid joint stops, incorrect mesh units, unregistered transforms, empty/corrupted meshes, mismatched collision link frames, fake reliance on RGB-D masks contaminated by robot EE.
- For known synthetic meshes, verify geometry scale, transformed COM and inertia, volume, convex proxy bookkeeping and independent collision mesh placement.

### Test gate D — Simulator readback/compilation (separate optional runtimes, never fake PASS)

- **URDF**: XML parsing and actual ManiSkill/SAPIEN URDF import using existing downstream integration environment when available. Confirm 2 links/1 DOF, axis/limits, `q=0` matches canonical, and live PhysX link transforms agree with analytic SE(3) over sampled q. Render unchanged PLYs through PArtTwin1D's already existing adapter if that consumer is available; DO NOT MODIFY IT.
- **USD**: open with `pxr.Usd.Stage.Open`; verify correct rigid-body/articulation/joint schemas, local anchors, limits in correct units, and PLY references. When Isaac Sim available, instantiate and evaluate PhysX joint kinematics; if unavailable report `USD_SCHEMA_PASS; ISAAC_RUNTIME_NOT_RUN` separately.
- **MJCF**: compile with `mujoco.MjModel.from_xml_path`, run `mj_forward` at q0 and at negative/positive/edge q values, compare `xpos/xmat` for links against `Joint.transform(f)`; no spuriously added DoF; validate mass and inertia readback.
- Run a simple collision/contact test in each installed engine; report dynamic deviations rather than claiming solver equality.

### Test gate E — Final-step integration (CPU + existing CUDA regression)

- Synthetic Phase 4 -> Phase 5 completed run must finish with `<P5>/assets/<scene>/visual/{static,moving}.ply`, valid manifest, and when physics input is supplied valid `model.urdf`, `model.usd`, `model.xml`.
- Preserve all previous Phase-5 training tests, original PLY hash after export, frozen joint fingerprint, samples/views and final loss values. Inject export failure: no partial `export_complete.json` and no false `FULL_PHYSICS_READY`.
- Post-hoc exporter works on a previously completed Phase-5 run and does not modify any existing Phase-5 files; repeat with identical inputs is idempotent.
- Files remain loadable when the asset folder is moved to a different absolute path.

### Real-dataset acceptance

Use a finished real Phase-5 `red_fridge` run when available, with user-supplied measured collision/dynamics inputs, and compare its frozen Phase-3 joint poses against URDF/USD/MJCF results for **all original recorded q values**. Also test one completed prismatic real scene (e.g. light-blue box). Never claim real-scene export complete if those files or physics parameters are not available.

## 9. Deliverable/report Codex must produce

1. Implement the exporter **in RigEuler3DGS**, with all changed/new code committed to the correct feature branch only after the requested test gates pass. Preserve local state; report branch and commit IDs.
2. `docs/PHASE5_FINAL_SIM_ASSET_EXPORT_REPORT.md`: exact modified functions, discovered local-vs-reference differences, source hash, inputs and provided-vs-inferred physics, chosen coordinate gauge, generated trees, duration/memory, test commands/results, actual simulator versions, unresolved/NOT_RUN gates, limitations.
3. `docs/PHASE5_FINAL_SIM_ASSET_FORMAT.md`: versioned `asset.yaml` and `dynamics.yaml` schemas, `T_link_ply` math, phase5 input contract and consumer example, USD/MJCF parameter mapping with unit conversion, portable paths.
4. One runnable synthetic revolute and one synthetic prismatic fixture with verified split PLY and complete simulation formats under a dedicated **test output directory** (not committed as large binary assets unless project convention requires).
5. A one-command normal Phase-5 invocation with physics inputs that produces the required final `assets/<scene>/` structure **as the last Phase-5 operation**, plus a separate re-export command for already completed runs.

**STOP / FAIL rules:** Never claim a complete simulation-ready asset from Phase 5 alone if collision, mass, inertia or joint limits were not established; never change Phase 3 joint geometry, Phase 5 Gaussian numerics, or the original PLY; never modify PArtTwin1D as a workaround; never report engine tests passed if the engine was not executed.

**Definition of done:** The final step of `tools/train_phase5_rgbd_gaussians.py` saves `assets/<scene>/` containing **two original-parameter standard 3DGS PLYs and jointly generated, physically validated URDF/USD/MJCF** where physical inputs exist, with exact mathematical transport and verified portability. This implementation lives and runs **entirely in RigEuler3DGS**.
