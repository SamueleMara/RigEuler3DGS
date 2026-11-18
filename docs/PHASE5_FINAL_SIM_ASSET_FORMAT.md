# Phase-5 simulation asset: format, frames and inputs (format `rigeuler3dgs-sim-asset/1`)

This is the contract for the last step of Phase 5, `assets/<scene>/`, written by `recovery/sim_asset_export`.

- **Who reads it.** Anyone who loads the asset into ManiSkill/SAPIEN, Isaac Sim/Isaac Lab or MuJoCo, or writes the physics input.
- **What `<scene>` is.** The basename of the verified upstream `setup.source`, for example `final_red_fridge_episode_000000`. Override it with `--asset-scene`.
- **Revisions.** An explicit `--asset-revision NAME` writes `assets/<scene>.rev-NAME/` instead.

## 1. Directory

```text
<P5>/assets/<scene>/
├── asset.yaml                 canonical asset specification (single source; section 3)
├── object.yaml                PArtTwin1D schema_version-1 consumer config
├── joint_trajectory.csv       original_frame, q, posterior_sd_q (chronological; rad | m)
├── lineage.npz                PLY row -> source row / UID / lineage / unresolved, per link
├── export_manifest.json       inputs + signature, runtimes, full validation report, sha256 of every file
├── export_complete.json       written last, only after validation; {status, input_signature, manifest_sha256}
├── visual/static.ply          Graphdeco 3DGS PLY, rows of part_id 0, original stored values
├── visual/moving.ply          Graphdeco 3DGS PLY, rows of part_id 1, original stored values
├── physics/                   ONLY with a validated physics spec
│   ├── collision/static.obj   supplied surface in the static_link frame (metres)
│   ├── collision/moving.obj   supplied surface in the moving_link frame (metres)
│   ├── collision/<link>_convex_NNN.obj   convex proxies, when the surface itself is not convex
│   ├── dynamics.yaml          resolved link-frame physics + provenance + audit (section 5)
│   └── collision_report.json  units, transforms, topology, plausibility, containment, convexity, coverage
├── maniskill/model.urdf       ONLY with validated physics
├── isaac/model.usd            ONLY with validated physics and a USD runtime (binary crate, PXR-USDC)
└── mujoco/model.xml           ONLY with validated physics
```

**Portability.** Every runtime reference is relative to the file it occurs in:

- URDF meshes are `../physics/collision/...`.
- The MJCF uses `meshdir="../physics/collision"`.
- USD asset paths are `../visual/*.ply` and `../asset.yaml`.
- `object.yaml` refers to `maniskill/model.urdf` and `visual/*.ply`.

The validator rejects any absolute path of the run, the asset or `$HOME` in a runtime file. `export_manifest.json` keeps the absolute source paths, as provenance only. The whole `<scene>/` folder can be moved or copied to another machine.

**Immutability.** `export_manifest.json` hashes every file. Editing any file makes a later export refuse to reuse the asset. Optional runtime reports, such as `tools/check_sim_asset_isaac.py`, are written next to the folder as `<scene>.isaac_runtime_check.json`, never inside it.

## 2. Frames (column vectors; `T_A_B` maps coordinates in B to A)

| frame | definition |
|---|---|
| `Wc` | Phase-5 canonical capture world: the XYZ stored in both PLYs (anchor configuration, q = 0) |
| `B` | static_link frame, chosen equal to `Wc` |
| `J0` | moving_link frame at q = 0; local +Z = joint axis `a` |
| `W` | simulator world, z up, gravity (0, 0, -9.81) |

**The joint frame.** `R_B_J` is right-handed with third column `a`. Its auxiliary vector is the basis vector with the smallest `|e·a|` (lowest index on ties), which makes it a deterministic function of `a`. The origin `o` is:

- **revolute:** exactly the Phase-3 `joint.point`;
- **prismatic:** the moving canonical XYZ centroid. This is a frame gauge (the same value as the articulated PLY's joint origin) and cancels from the kinematics; it is not a fitted quantity.

```text
T_B_J0          = [R_B_J, o; 0 0 0 1]
T_link_ply      = {static_link: I,  moving_link: inv(T_B_J0)}       (PLY coordinates stay in Wc = B)
T_B_Lmoving(q)  = T_B_J0 · M_z(q)          M_z = rotation about +Z by q (revolute) | translation q along +Z (prismatic)
T_B_Pmoving(q)  = T_B_J0 · M_z(q) · inv(T_B_J0)
revolute:   x(q) = Exp([a]× q)(x − o) + o          prismatic:   x(q) = x + q a
```

These are `recovery.contracts.Joint.transform(f)` exactly. The validator checks them at every recorded frame, to 1e-15 on real runs.

**`T_W_B` (`T_world_root`)** comes from one of two sources:

- a supplied, measured rigid `T_world_capture` with `gravity_alignment: measured|declared`;
- the explicit gauge `W = Wc`. Then `gravity_alignment` is `declared` (the user states that the capture axes are gravity-aligned) or `unverified` (the default when nothing is supplied).

A scale inside an SE(3) is rejected.

**Rendering rule** (every consumer). `T_W_P = T_W_L(live physics pose) · T_link_ply`, then:

- `μ_W = R μ_P + t`;
- `q_W = q(R) ⊗ q_P` (Hamilton, scalar first);
- `Σ_W = R Σ_P Rᵀ`;
- SH evaluated in the PLY frame, with the direction `R^T(μ_W − c_W)`.

Never re-animate the Gaussians with q while physics runs. PArtTwin1D's `LinkAdapter` implements exactly this rule.

## 3. `asset.yaml` (schema_version 1)

| key | content |
|---|---|
| `status` | `FULL_PHYSICS_READY`, `PHYSICS_PREVIEW_ASSUMPTIONS`, `PHYSICS_INCOMPLETE_RUNTIME`, `BLOCKED_MISSING_PHYSICS`, `PHYSICS_REJECTED` or `VISUAL_ONLY` (section 6) |
| `readiness` | `visual_complete`, `kinematics_complete`, `physics_complete`; `urdf`, `usd`, `mjcf`, `mujoco_compile`, `sapien_runtime`, `isaac_runtime` (each PASS / FAIL / NOT_RUN / NOT_WRITTEN); `physics_assumptions`; `all_physics_measured` |
| `missing_inputs` | what blocks a higher status |
| `frames` | convention text, `world_gauge`, `gravity_alignment`, `gravity_world`, `T_world_root` |
| `links.<link>` | `visual_ply`, `T_link_ply` (4x4), `n_gaussians`, `ply_sha256`, `collision` {surface, parts, approximation} |
| `visual` | format, `sh_degree`, quaternion convention, SH frame, Phase-5 `appearance_settings`, render rule |
| `joint` | `name: articulation_joint`, type, parent / child, `axis_capture`, `origin_capture`, `origin_kind`, `T_parent_joint` (= `T_B_J0`), `axis_local [0,0,1]`, `q_units`, `q_canonical: 0`, `anchor_original_frame`, `observed_demonstration_range`, `observed_range_is_not_a_mechanical_limit: true`, `physical_limits` {lower, upper, velocity, effort, provenance} or null, `trajectory` |
| `exports` | relative paths of URDF / USD / MJCF / object.yaml / dynamics / lineage |
| `source_phase5` | run name, completion source and sha256, articulated PLY sha256, joint fingerprint (file hashes + per-frame transform hash), train mode |
| `hashes` | sha256 of every file written before `asset.yaml` |

**`lineage.npz`** has the following arrays for each prefix `p` (`static` or `moving`):

- `p_source_row`, the row in `final/point_cloud.ply`;
- `p_gaussian_uid`, `p_original_id`, `p_parent_uid`, `p_provenance`;
- `p_unresolved`: 1 means 4A left the row, or an ancestor, undecided. Its link is the frozen render assignment, so its membership is **uncertain**.

It also holds `phase5_unresolved_originals_*` and a `note`.

## 4. Phase-5 input contract

The exporter reads the following, which must agree exactly:

- **`complete.json`:** `stage: phase5`, `joint_unchanged: true`, `joint_fingerprint`. Post-hoc only; in the driver, the completion payload that is about to be written is passed instead.
- **`manifest.json`:** `upstream.setup`, with `source`, `joint_type` and `direction`.
- **`final/joint/`:** the joint is rebuilt by `recovery.contracts.load_joint`. Its file hashes, frame order and per-frame transform hash must match the fingerprint.
- **`final/joint_unchanged_proof.json`:** identical, not optimized.
- **`final/point_cloud.ply`:**
  - `vertex` must hold the 3DGS fields plus `part_id`, `gaussian_uid`, `original_id`, `parent_uid`, `provenance` and `unresolved`;
  - `joint`: type, axis and origin equal the Phase-3 values exactly; `q_canonical = 0`;
  - `joint_state`: equal to the chronological Phase-3 trajectory;
  - `part_id` in {0, 1}, both parts non-empty, UIDs unique.
- **`final/appearance_settings.json`** and **`unresolved_originals.npz`**, when present.

Any mismatch is an explicit error. A `NOT_FOR_RESULTS` run is exported, with the flag carried over and a warning. The split keeps every 3DGS column bit-identical:

- the output dtype is built from the source descriptors, in source order;
- an unknown vertex field is an error, never dropped;
- reassembling the two outputs gives back the source bit for bit;
- the source file hash is unchanged.

## 5. Physics input (`--asset-physics-spec`, schema_version 1)

Every key is checked; unknown keys are errors; `null` is never a value. Relative paths resolve against the spec file.

```yaml
schema_version: 1
registration:                       # optional here, or --asset-registration FILE (same mapping, or {registration: ...})
  T_world_capture: [[...4x4...]]    # measured rigid T_W_Wc   | null together with `gauge: capture_world`
  gravity_alignment: measured       # measured | declared | unverified (unverified only with gauge capture_world)
  provenance: "how it was measured"
collision:
  static:
    source: static_surface.obj      # OBJ, closed outward triangles (polygons are fan-triangulated)
    units: m                        # m | cm | mm   (explicit scale; never inside the SE(3))
    T_capture_mesh: [[...4x4...]]   # rigid: mesh coordinates -> capture world
    provenance: measured            # measured | estimated
    source_kind: supplied_mesh      # supplied_mesh | reconstructed_rgbd (needs ee_exclusion: {method, evidence})
    convex_parts: [a.obj, b.obj]    # optional validated decomposition (each closed + convex)
    convex_coverage_tolerance_m: 0.002   # required with convex_parts
    approximation: null             # or convex_hull (opt-in; marks the asset PREVIEW)
  moving:
    ... same keys ...
    at_original_frame: 100          # REQUIRED: the Phase-3 frame at which this mesh is posed (anchor = q 0)
links:
  static_link:
    mass_kg: 12.0
    com_capture_m: [x, y, z]
    inertia_com_capture_kg_m2: [[...3x3...]]   # about the COM, capture axes; symmetric, PD, triangle inequality
    provenance: measured            # measured | estimated
  moving_link:
    ... same, + at_original_frame: 100 ...
    # OR  density_kg_m3: 600, provenance: density_assumption   (needs a watertight collision surface)
joint:
  lower: -0.1                       # PHYSICAL stops, rad | m; must contain q = 0
  upper: 1.5
  velocity_limit: 2.0               # rad/s | m/s
  effort_limit: 5.0                 # N m | N
  damping: 0.3                      # viscous, N m s/rad | N s/m
  coulomb_friction: 0.05            # N m | N
  armature: null                    # optional
  actuation: null                   # only a passive joint is implemented
  limits_from_observed_range: false # true + lower/upper null = labelled preview limits (PREVIEW status)
  provenance: measured
contact:                            # optional; otherwise each engine's default material (reported)
  static_friction: 0.8
  dynamic_friction: 0.6
  restitution: 0.0
  provenance: estimated
validation:                         # optional overrides of the declared plausibility gates
  extent_ratio_range: [0.2, 5.0]    # mesh AABB diagonal / link Gaussians p2-p98 AABB diagonal
  centroid_offset_max: 1.0          # AABB centre offset / Gaussian extent
  gaussian_inside_tolerance_fraction: 0.05
  gaussian_inside_min_fraction: 0.5 # share of the link's Gaussian means within tol of its collision geometry
```

**Moving-part inputs** posed at frame f are brought back to the anchor with the frozen joint (`x0 = inv(T_f) x_f`, then into J0). Nothing is fitted.

**Plausibility gates.** These only detect wrong units, swapped links, a wrong `T_capture_mesh` or a wrong `at_original_frame`; they measure nothing:

- extent ratio;
- centre offset;
- containment of the link's own Gaussians.

**Collision proxies.** Proxies are what the engines collide with:

- the surface itself, if it is convex (closed, and hull volume = mesh volume within 1e-6);
- otherwise the supplied `convex_parts`, which must be convex and cover every surface vertex within the declared tolerance;
- otherwise, on opt-in, `convex_hull`.

The 3DGS Gaussians are never collision bodies.

**`physics/dynamics.yaml`** holds the resolved physics, in link frames:

- per link: mass, `com_link_m`, `inertia_com_link_kg_m2`, principal moments and axes, provenance, audit;
- collision: provenance, approximation, files, audit;
- joint: stops plus `limits_provenance` (`physical_stops_<prov>` or `observed_range_preview`), velocity, effort, damping, Coulomb friction, armature;
- contact, `assumptions`, `provenance_summary` (`all_measured`), `warnings`.

## 6. Status

| status | meaning | model files |
|---|---|---|
| `FULL_PHYSICS_READY` | spec valid; physical stops; gravity measured or declared; proxies supplied or convex; URDF + USD + MJCF validated | all three |
| `PHYSICS_PREVIEW_ASSUMPTIONS` | valid but with an opt-in assumption: observed-range limits, `convex_hull`, `reconstructed_rgbd`, or unverified gravity | all three, labelled |
| `PHYSICS_INCOMPLETE_RUNTIME` | valid physics, but a USD / MuJoCo runtime was missing (that format is NOT_RUN) | those that could be validated |
| `BLOCKED_MISSING_PHYSICS` | no physics spec (auto) | none; `object.yaml` says so |
| `PHYSICS_REJECTED` | invalid spec (auto); the reason is in `missing_inputs` | none |
| `VISUAL_ONLY` | `--asset-export-mode visual_only` | none |

`required` accepts only `FULL_PHYSICS_READY` and checks the spec, registration and runtimes **before** training. Mesh-against-Gaussian gates run at export, on the trained model.

## 7. Engine parameter mapping (same canonical object, different engine semantics)

| quantity | URDF (SAPIEN / ManiSkill) | USD (Isaac) | MJCF (MuJoCo) |
|---|---|---|---|
| joint frame | `<origin xyz=o rpy=RPY(R_B_J)>` (fixed-axis, R = Rz Ry Rx), `<axis 0 0 1>` | `localPos0/localRot0 = T_B_J0` (float32), `localPos1/Rot1 = I`, axis Z; body prims at q = 0 | child `<body pos quat(wxyz) = T_B_J0>`, joint `pos 0 0 0 axis 0 0 1` |
| fixed base | simulator setup (`fix_root_link`) at `T_world_root` | `FixedJoint` world → static_link, `localPos0/Rot0 = T_W_B` | static body has no joint (welded), `pos/quat = T_W_B` |
| limits | `lower/upper` rad or m | revolute **degrees** (rad × 180/π); prismatic m | `range` rad or m (`angle="radian"`) |
| velocity limit | `velocity` (SAPIEN 3.0.3 loads 100 rad/s instead; reported) | `physxJoint:maxJointVelocity` deg/s (raw attribute, no plugin needed) | **not representable** |
| effort limit | `effort` | `DriveAPI maxForce` | `actuatorfrcrange` (bounds actuators only; joint is passive) |
| damping | `<dynamics damping>` | `DriveAPI` stiffness 0, target velocity 0, damping × π/180 (N m s/deg) for revolute | `damping` |
| Coulomb friction | `<dynamics friction>` (PhysX joint friction coefficient in SAPIEN; reported) | **not mapped** (PhysX `jointFriction` is dimensionless) | `frictionloss` |
| inertia | `<inertial>` COM origin, full tensor in link axes | `MassAPI` mass, centerOfMass, diagonalInertia + principalAxes (det +1) | `<inertial pos fullinertia>` (MuJoCo diagonalises) |
| collision | one `<collision>` mesh per convex proxy | `Mesh` + `CollisionAPI` + `MeshCollisionAPI approximation=convexHull`, purpose guide | one mesh geom per proxy (MuJoCo collides with the convex hull) |
| link-pair contact | disabled by PhysX for jointed links | `physics:collisionEnabled = false` | `<contact><exclude static_link moving_link/>` (MuJoCo does not filter a world-welded parent) |
| contact material | (engine default) | `UsdPhysics.MaterialAPI` static/dynamic/restitution, physics-purpose binding | `friction` = dynamic_friction only; restitution not mapped |
| gravity | engine default −Z | `/PhysicsScene` −Z, 9.81 | `option gravity 0 0 -9.81` |
| visual | not in URDF; `object.yaml` / `asset.yaml` | `rigeuler:visualPly` (asset path) + `rigeuler:T_link_ply` (16 doubles, row-major, column-vector convention) on each body; `rigeuler:requiresGaussianRenderer` | not in MJCF; an external gsplat adapter reads `xpos/xquat` |

USD stage: `metersPerUnit 1`, `kilogramsPerUnit 1`, `upAxis Z`, default prim `/<name>` with `ArticulationRootAPI`.

## 8. Commands

```bash
# Phase 5 with the asset as its last step (default --asset-export-mode auto)
python tools/train_phase5_rgbd_gaussians.py --phase4-dir P4 --output P5 \
    --asset-physics-spec physics.yaml [--asset-registration reg.yaml] [--asset-export-mode required] \
    --asset-usd-python PY_WITH_PXR --asset-mujoco-python PY_WITH_MUJOCO [--asset-sapien-python PY_WITH_SAPIEN]

# re-export / upgrade a completed run (never modifies the run; same inputs -> verified reuse)
python tools/export_phase5_sim_asset.py --phase5-dir P5 --asset-physics-spec physics.yaml [--asset-revision NAME]

# optional Isaac Sim PhysX runtime check (writes <asset>.isaac_runtime_check.json next to the asset)
python tools/check_sim_asset_isaac.py --asset-dir P5/assets/<scene> --isaac-python ~/isaac-sim/python.sh
```

**Runtimes.** Interpreters are found in this order:

1. an explicit `--asset-*-python`;
2. `RIGEULER_USD_PYTHON`, `RIGEULER_MUJOCO_PYTHON`, `RIGEULER_SAPIEN_PYTHON`;
3. `RIGEULER_SIM_PYTHON` (USD + MuJoCo);
4. the current interpreter.

No path is hard-coded. A missing runtime reports `NOT_RUN`, never `PASS`.

**Through the runners.** `run_phase5_synthetic.sh` passes `EXTRA_ARGS` to the driver, and `run_phase45_then_5_synthetic.sh` passes `P5_ARGS` through it. So `P5_ARGS="--asset-physics-spec ... --asset-export-mode required"` reaches Phase 5 only. Earlier phases are unchanged.
