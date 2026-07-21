# RigEuler3DGS — Full Development Context

**State represented:** 13 July 2026  
**Repository:** `~/Joint3DGS/RigEuler3DGS`

This document records the complete design evolution, implemented components, failed branches, current architecture, active objectives, debug outputs, and final joint-estimation logic. It is intended as implementation context for continuing the code without reintroducing discarded ideas or silently changing the requested formulation.

---

## 1. Project objective

The repository extends canonical 3D Gaussian Splatting with a world-fixed Eulerian motion representation for articulated scenes.

The current target scenario contains:

- a statically trained canonical 3DGS;
- a learned binary semantic foreground field;
- a frozen statically trained per-Gaussian DINOv3 field;
- one relevant semantic object;
- exactly one rigid moving part inside that object;
- a complementary static part inside the same semantic foreground;
- possible non-foreground Gaussians whose velocity may still be learned by the global Eulerian proposal field, but which must never influence joint understanding.

The final dynamic system must jointly estimate:

1. a global Eulerian proposal velocity field for all Gaussians;
2. the semantic-foreground static/moving assignment;
3. a single bounded time-varying rigid twist for the moving foreground part;
4. the final joint type: prismatic, revolute, or screw;
5. the final joint axis and physically meaningful axis anchor.

---

## 2. Static canonical training

### 2.1 Canonical Gaussian parameters

The static stage optimizes the standard 3DGS parameters:

- Gaussian centers;
- scales;
- rotations;
- opacity;
- spherical-harmonic color coefficients;
- exposure parameters.

The canonical geometry and appearance are frozen when dynamic optimization begins, unless an explicitly enabled late fine-tuning mode is used.

### 2.2 Semantic foreground field

Every Gaussian owns one semantic logit:

```text
semantic_probability_i = sigmoid(semantic_logit_i)
```

The field is splatted using the same alpha compositing as RGB. The rendered semantic probability is supervised against the binary Grounded-SAM mask with pixelwise BCE.

The semantic field is also weakly regularized in 3D with local geometry/color-aware coherence to reduce isolated noise without smearing object boundaries.

### 2.3 DINOv3 feature field

Every Gaussian owns a statically trained DINO feature vector. The CUDA/Python renderer was extended so arbitrary Gaussian feature channels are accumulated using the same alpha/transmittance weights used by color.

The rendered feature image is compared against precomputed dense DINOv3 image features using normalized cosine distillation, optionally with weak feature-field 3D coherence.

During dynamic training:

```text
per-Gaussian DINO descriptors are frozen
```

DINO rendering loss remains active as transport supervision. Its gradients move the transported Gaussians through the motion MLP/grid; they must not alter the static DINO descriptor attached to each Gaussian.

### 2.4 PLY persistence

The semantic logits and DINO field were embedded into the Gaussian PLY attribute list rather than saved only as sidecars. Loading a static PLY therefore restores the exact per-Gaussian fields required by dynamic initialization.

---

## 3. Eulerian motion representation

### 3.1 Global proposal field

The adaptive-grid-conditioned proposal is defined for **all Gaussians**:

```text
v_proposal(x,t) = motion_MLP(x, t, grid_feature[cell(x)])
```

Trainable proposal parameters:

- `motion_field.parameters()`;
- `motion_grid.features`.

The adaptive grid stores one learned latent feature per canonical spatial cell. The grid feature itself is not frame-indexed; time is supplied to the motion MLP.

### 3.2 Adaptive grid

The grid topology is updated discretely through split/merge operations. Its feature vectors are optimized continuously by gradient descent.

Grid-refinement evidence can include detached statistics such as:

- rendering-position gradient;
- velocity variance;
- displacement variance;
- DINO variance;
- assignment entropy/variance;
- rigid-twist residual;
- local strain.

The grid remains global, including semantic non-foreground Gaussians. Joint-related cell statistics must nevertheless be computed using semantic foreground only.

### 3.3 Motion-grid HTML

Every phase/debug snapshot can export an OctoMap-style Plotly visualization:

```text
motion_grid_octomap.html
```

Each active cell is displayed as a cube whose size reflects refinement level. Color modes include PCA of learned feature vectors, feature norm, dynamic probability, semantic probability, speed, displacement, score, level, and count.

---

## 4. Critical semantic-isolation contract

This distinction is mandatory.

### 4.1 All Gaussians may learn proposal velocity

Semantic non-foreground does **not** mean physically static.

Therefore:

```text
all Gaussians
    -> adaptive motion grid
    -> Eulerian proposal MLP
    -> RGB/depth/DINO/cycle supervision
```

For non-foreground Gaussians:

```text
v_transport = v_proposal
```

They remain visible in track and velocity outputs.

### 4.2 Semantic foreground alone participates in joint understanding

The frozen mask is:

```text
joint_foreground = semantic_probability >= semantic_dynamic_threshold
```

Only this subset participates in:

- exact XYZ+DINO distance-matrix initialization;
- static/moving part clustering;
- trainable part-field assignments;
- rigid-twist fitting;
- static-vs-rigid label targets;
- prismatic/revolute/screw model selection;
- axis fitting;
- pitch fitting;
- final moving-part joint visualization.

Non-foreground proposal velocities must never enter any of those computations.

### 4.3 Correct transport split

```text
semantic background:
    unrestricted proposal transport only

semantic foreground:
    gated static / proposal / rigid transport
```

Thus rendering gradients from background train the proposal field but have no path to:

- `omega(t)`;
- `u(t)`;
- foreground part assignments;
- final joint estimation.

---

## 5. Exact foreground initialization

The semantic foreground contains `N_fg` Gaussians. Every foreground Gaussian contributes:

```text
x_i = canonical center
f_i = frozen static DINO feature
```

A complete precomputed distance matrix is built:

```text
D_xyz(i,j)  = ||x_i - x_j||_2 / robust_xyz_scale
D_dino(i,j) = 1 - cosine(f_i, f_j)

D(i,j) = lambda_xyz * D_xyz(i,j)
       + lambda_dino * D_dino(i,j)
```

for every foreground pair `i,j`.

No node pooling, supernode descriptor, random fallback, or feature substitution is allowed.

The exact matrix is computed blockwise on CPU and stored in float16 or float32. K-medoids runs directly on the precomputed matrix.

For one moving part, initialization produces exactly two foreground slots:

```text
slot 0 = foreground static component
slot 1 = foreground moving component
```

The larger initial component is mapped to the static slot. This is only initialization; every foreground Gaussian remains reassignable during optimization.

---

## 6. Trainable foreground part field

Every foreground Gaussian owns a learned part-field vector:

```text
z_i in R^d
```

There are exactly two learned cosine prototypes:

```text
c_static
c_moving
```

Assignment is recomputed every iteration:

```text
A_i = softmax([
    cosine(z_i, c_static),
    cosine(z_i, c_moving)
] / temperature)

p_dynamic_i = A_i[moving]
```

After initialization, XYZ and DINO are not reused by assignment losses. Later clustering uses only the optimized part-field cosine geometry.

Part-field objectives include:

- point-to-prototype cosine compactness;
- sampled point-to-point same/different cosine loss;
- prototype separation;
- assignment-entropy sharpening;
- minimum occupancy for both known slots.

A post-step safeguard prevents a hard static or moving slot from becoming empty. It uses only current learned field affinity, not XYZ or DINO.

Non-foreground Gaussians do not own trainable joint-assignment vectors. Full-scene `dynamic_prob=0` for them is bookkeeping only and must not zero their proposal velocity.

---

## 7. Explicit single-moving-part rigid field

The moving foreground part has one time-varying bounded twist:

```text
v_rigid(x,t) = omega(t) x (x - motion_center) + u(t)
```

where:

- `omega(t)` is angular velocity;
- `u(t)` is translation velocity evaluated relative to the learned/fixed motion center;
- `motion_center` is the canonical foreground center used to improve numerical conditioning.

A temporal MLP predicts the six twist components. Its outputs are bounded with `tanh` to prevent scale explosion.

---

## 8. Joint optimization and gradient routing

Static/moving assignment, proposal motion, and rigid twist are optimized during every dynamic iteration. There is no hard discovery-versus-rigid phase split.

### 8.1 Twist fitting

Only current foreground moving assignments fit the twist:

```text
L_twist_fit = robust(
    v_rigid - stop_gradient(v_proposal)
)
```

Gradients:

```text
omega(t), u(t) only
```

The proposal must be detached here. Earlier code allowed this loss to update both experts and collapsed the faster proposal toward the zero-initialized twist.

### 8.2 Proposal-to-rigid consistency

Only confidently moving foreground points receive:

```text
L_proposal_rigid = robust(
    v_proposal - stop_gradient(v_rigid)
)
```

Gradients:

```text
motion MLP and grid only
```

### 8.3 Foreground static identity

Only confidently static foreground points receive:

```text
L_proposal_static = robust(v_proposal)
```

Uncertain points are not statically suppressed.

### 8.4 Assignment target

Using detached experts:

```text
e_static = ||v_proposal||
e_rigid  = ||v_proposal - v_rigid||

q_dynamic = sigmoid(
    (e_static - e_rigid) / adaptive_temperature
)

L_label = BCE(p_dynamic, q_dynamic)
```

The temperature is derived from the current residual-difference scale and clamped. A previous fixed temperature of `0.10` was much larger than observed residual differences and forced all targets toward `0.5`.

### 8.5 Temporal twist smoothness

First- and second-order temporal differences regularize:

```text
omega(t)
u(t)
```

This discourages framewise noise without replacing the rendering evidence.

### 8.6 Cycle and rendering supervision

RGB, depth, frozen-DINO rendering, and cycle consistency continue supervising transport.

DINO weight must remain lower than RGB/depth because dense DINO is not an exact multiview correspondence field.

---

## 9. Uncertain-point transport

With default confidence thresholds:

```text
static_threshold  = 0.2
dynamic_threshold = 0.8
```

Foreground transport is continuous:

```text
p <= 0.2:
    progressively hand proposal to identity

0.2 < p < 0.8:
    use the complete rendering-driven proposal

p >= 0.8:
    progressively hand proposal to the rigid expert
```

A separate continuous global handover ramps trust in the rigid expert while it learns.

This avoids the discarded mixture:

```text
p * rigid + (1-p) * proposal
```

which halved motion at `p=0.5` when the rigid field was initially zero.

For semantic non-foreground points, this gating is bypassed entirely:

```text
v_transport = v_proposal
```

---

## 10. Important failed branches and lessons

### 10.1 Hard separated phases

Failure:

```text
weak/no early motion seed
-> point declared static
-> excluded from later rigid fitting
-> remains permanently stuck
```

Current solution: jointly optimize assignment and twist during every dynamic iteration.

### 10.2 Detached-label gradient bug

A model-target loss created `p_dynamic` inside `torch.no_grad()`. The loss had a value but no gradient to the label field.

Lesson: only teacher targets and fitted models are detached; the predicted assignment must be read outside `no_grad()`.

### 10.3 Unbounded moving-anchor velocity propagation

A direct loss forced candidate points toward an unconstrained fitted velocity `omega x x_current + u`. As transported positions grew, target velocity grew, causing positive feedback and coordinate explosion.

This branch must not be reintroduced.

Current rigid field uses:

- bounded twist outputs;
- canonical/centered coordinates;
- separated gradients;
- velocity safety clamp.

### 10.4 Joint-collapse implementation

An earlier joint branch made `L_rigid` update both proposal and twist, applied static/residual shrinkage broadly, used a fixed oversized label temperature, and used fixed 64 supernodes instead of the requested trainable per-Gaussian clustering.

It converged to:

```text
rigid approximately zero
proposal mostly zero
p_dynamic approximately 0.5
few saturated outliers
```

All four causes were removed in the exact implementation.

### 10.5 GPU failure during exact initialization

The first exact matrix implementation defaulted the full `N^2` computation to CUDA while hundreds of dynamic-camera images were resident on GPU.

Current requirement:

- exact matrix and k-medoids run on CPU;
- dynamic sequence images remain on CPU;
- only the sampled frame is copied to CUDA.

---

## 11. Deep debugging and visualization

Joint runs export snapshots:

```text
joint_early
joint_mid
joint_final
```

Each snapshot can contain:

```text
gaussian_tracks.pt
dynamic_summary.json
gaussian_tracks_visualization.html
velocity_field_visualization.html
motion_grid_state.pt
motion_grid_summary.json
motion_grid_octomap.html
deep_diagnostics_summary.json
per_point_diagnostics.pt
per_cell_diagnostics.pt
per_point_diagnostics.csv
rigid_twist_trajectory.html
```

Track exports include:

```text
positions
velocities
canonical_positions
semantic_prob
dynamic_foreground_mask
dynamic_prob
joint_initial_cluster_id
joint_cluster_id
joint_part_assignments
joint_part_field
rigid_velocities
residual/proposal_velocities
rigid_omega
rigid_translation
rigid_center
cell_id
```

Deep diagnostics distinguish:

- never-activated foreground points;
- moving-but-labelled-static points;
- label disagreement with identity-vs-rigid target;
- proposal-vs-rigid residuals;
- assignment changes from initialization;
- dead or mixed motion-grid cells.

---

## 12. Final joint estimation

The final estimator uses only the optimized foreground moving component.

Non-foreground proposal motion is deliberately ignored for:

- joint type;
- axis direction;
- axis point;
- pitch;
- model residuals.

### 12.1 Input rigid twist

The fitted field is:

```text
v(x,t) = omega(t) x (x - c) + u(t)
```

where `c` is the motion center.

### 12.2 Axis direction

For rotational candidates, the axis direction is the sign-invariant weighted principal direction of active `omega(t)` vectors.

For a prismatic candidate, the axis is the principal direction of mean moving-part rigid velocities.

### 12.3 Axis anchor from the motion center

For each active rotational frame, split translation into components parallel and perpendicular to `omega`:

```text
u_parallel = projection of u on omega
u_perp     = u - u_parallel
```

The point on the instantaneous axis closest to the motion center is:

```text
q_t = c + (omega x u_perp) / ||omega||^2
```

The final anchor is a robust weighted geometric median of these points, projected to the canonical axis-line representation closest to `c`.

This separates:

```text
motion_center: numerical reference used by the twist
axis_anchor: physically meaningful point on the estimated joint axis
```

### 12.4 Candidate models

The estimator scores three velocity models against the learned moving-part rigid velocity.

Prismatic:

```text
v_prismatic(t) = rate(t) * axis
```

Revolute:

```text
v_revolute(x,t) = alpha(t) * axis x (x - axis_anchor)
```

Screw:

```text
v_screw(x,t) = v_revolute(x,t)
             + h * alpha(t) * axis
```

The pitch per radian is fitted from:

```text
alpha(t) = omega(t) dot axis
beta(t)  = u(t) dot axis

h = sum(w_t * alpha(t) * beta(t))
  / sum(w_t * alpha(t)^2)
```

and:

```text
pitch_per_revolution = 2*pi*h
```

Screw is selected only when its axial motion is physically meaningful and its residual improves sufficiently over revolute. Prismatic is selected when rotational linear activity is negligible relative to translation. Otherwise the estimator selects revolute.

### 12.5 Final outputs

At the model root and at `joint_final` snapshot:

```text
final_joint_estimation.json
final_joint_estimation.html
final_joint_model_scores.html
```

The 3D HTML displays:

- non-foreground points in gray;
- foreground static points in blue;
- foreground moving rigid points in red;
- estimated joint axis;
- axis anchor;
- motion center.

---

## 13. Current training parameter groups

### Static stage

Trainable:

- canonical Gaussian geometry/appearance;
- semantic logits;
- DINO Gaussian descriptors;
- exposure.

### Dynamic joint stage

Trainable on all Gaussians:

- motion MLP;
- adaptive-grid features;
- exposure.

Trainable for semantic foreground joint understanding:

- foreground point-field vectors;
- static/moving prototypes;
- time-varying rigid twist `omega(t),u(t)`.

Frozen:

- canonical Gaussian geometry/appearance;
- semantic field;
- DINO Gaussian descriptors.

---

## 14. Current invariants

Any future modification must preserve:

1. non-foreground may learn proposal velocity;
2. non-foreground never enters joint fitting or type/axis estimation;
3. exact initialization uses every semantic-foreground Gaussian and its frozen static DINO descriptor;
4. one static and exactly one moving foreground slot;
5. proposal, twist, and assignment have separated gradient paths;
6. uncertain foreground points keep full proposal motion;
7. DINO descriptors remain frozen dynamically;
8. no unbounded velocity teacher based on transported positions;
9. full pairwise initialization is CPU-only;
10. final joint estimator refuses background/top-k fallbacks.

---

## 15. Recommended run

```bash
python train.py \
  --root_scene /path/to/scene \
  -m output/run_joint_semantic_isolated \
  --dynamic_motion \
  --filter_sem \
  --semantic_dynamic_threshold 0.2 \
  --start_ply output/static_run/point_cloud/iteration_30000/point_cloud.ply \
  --start_ply_iteration 30000 \
  --joint_init_distance_device cpu \
  --dynamic_camera_data_device cpu
```

The exact joint branch should refuse to start without a valid semantic field when `--filter_sem` is required.

---

## 13. Repository structure relevant to the current implementation

The paths below are the authoritative integration points for the current branch.

```text
RigEuler3DGS/
├── train.py
│   ├── static/dynamic stage transition
│   ├── rendering and loss assembly
│   ├── dynamic optimizer stepping
│   ├── high-debug snapshots
│   ├── Gaussian-track export
│   └── final joint-estimation call
├── arguments/__init__.py
│   ├── semantic foreground threshold
│   ├── exact XYZ/DINO initialization options
│   ├── joint-field and twist hyperparameters
│   ├── confidence thresholds
│   └── debug/export options
├── scene/
│   ├── gaussian_model.py
│   │   ├── static fields and PLY loading
│   │   ├── dynamic foreground-mask construction
│   │   ├── global adaptive grid and proposal MLP
│   │   ├── JointPartField initialization
│   │   ├── bounded rigid-twist field
│   │   ├── proposal/static/rigid transport composition
│   │   └── optimizer/checkpoint state
│   ├── adaptive_motion_grid.py
│   ├── rigid_twist_field.py
│   └── joint_part_field.py
│       ├── complete foreground XYZ+DINO matrix
│       ├── precomputed-distance k-medoids
│       ├── foreground-only learned part vectors
│       ├── static/moving cosine prototypes
│       └── full-scene bookkeeping maps
├── gaussian_renderer/__init__.py
│   └── transports Gaussians before standard rasterization
├── utils/
│   ├── dynamic_loss_utils.py
│   │   ├── joint EM-style objectives
│   │   ├── foreground slicing
│   │   ├── cycle and motion diagnostics
│   │   └── adaptive-grid scores
│   ├── final_joint_estimation.py
│   │   ├── prismatic/revolute/screw scoring
│   │   ├── axis/pitch estimation
│   │   └── JSON/HTML visualization
│   ├── visualize_motion_grid_octomap_html.py
│   └── visualize_rigid_twist_html.py
└── tools/
    ├── estimate_final_joint.py
    ├── verify_semantic_joint_contract.py
    ├── inspect_joint_output.py
    ├── validate_joint_exact_training.py
    ├── analyze_phase_debug_snapshots.py
    └── visualize_motion_grid_octomap.py
```

### 13.1 Static model input

Dynamic runs commonly start from a fully trained static PLY:

```text
output/<STATIC_RUN>/point_cloud/iteration_30000/point_cloud.ply
```

The PLY must include:

- canonical Gaussian geometry and appearance;
- semantic logits/probabilities;
- statically trained per-Gaussian DINO field.

A dynamic output directory must be new. Do not resume a checkpoint whose joint field was initialized with the wrong all-scene mask.

---

## 14. Confirmed real-dataset semantic-contamination incident

### 14.1 Dataset observation

The real reconstruction contained:

```text
full Gaussians             47,792
semantic foreground        11,137  at threshold 0.2
semantic background        36,655
```

The static semantic field was visually coherent and concentrated near probability one on the object. The failure happened after static training, in dynamic joint initialization.

### 14.2 Exported contaminated state

The failed dynamic export contained:

```text
dynamic_foreground_mask true       47,792 / 47,792
joint_part_field_gaussian_ids       47,792 / 47,792
joint_part_field_foreground_ids     47,792 / 47,792
```

The learned moving component was dominated by semantic background:

```text
hard moving total                    2,249
hard moving semantic foreground         31
p_dynamic >= 0.8 total              1,938
p_dynamic >= 0.8 semantic foreground    26
mean p_dynamic foreground          0.1284
mean p_dynamic background          0.1728
```

The final estimator correctly refused to fit a joint from only nine confidently moving semantic-foreground Gaussians.

### 14.3 Confirmed source bug

The active repository printed:

```text
# Exact joint branch assigns every starting Gaussian...
fg_mask = torch.ones_like(semantic_mask, dtype=torch.bool)
```

This deliberately overwrote the valid semantic mask. The absence of the following safeguards confirmed that the semantic-isolated patch had never been installed:

```text
assert_joint_foreground_isolation
foreground_mask_full
expected semantic foreground only
```

### 14.4 Why the moving cluster became background

The initializer solved a two-cluster problem on the complete scene. Because background outnumbered the object by more than three to one, the combined XYZ+DINO geometry was dominated by environment structure. Mapping the larger cluster to static and the smaller cluster to moving did not identify the articulated part; it selected a background substructure.

The later occupancy and cosine-field losses preserved that incorrect split. The semantic object mostly remained in the foreground-static slot, leaving too few object points for joint estimation.

### 14.5 Required fix

The repaired behavior is:

```text
Global proposal field and adaptive grid:
    all 47,792 Gaussians

Joint initialization and learned assignment:
    only 11,137 semantic-foreground Gaussians

Non-foreground transport:
    proposal velocity only

Rigid twist and joint losses:
    semantic foreground only
```

The exact distance matrix must therefore be:

```text
11,137 x 11,137
```

not:

```text
47,792 x 47,792
```

---

## 15. Semantic-isolation invariants

These conditions must hold simultaneously. Satisfying only one is insufficient.

### 15.1 Mask invariant

```python
semantic_mask = semantic_probability >= semantic_dynamic_threshold
joint_foreground_mask = semantic_mask.clone()
```

The joint branch must never replace this with an all-true tensor.

### 15.2 Initializer invariant

```python
foreground_ids = where(joint_foreground_mask)
xyz_fg = canonical_xyz[foreground_ids]
dino_fg = frozen_static_dino[foreground_ids]
```

The complete pairwise matrix and k-medoids use exactly those rows.

### 15.3 Part-field invariant

```text
number of learned part-field vectors = N_foreground
```

Every foreground Gaussian is present exactly once. No semantic-background ID is present.

### 15.4 Proposal invariant

```text
number of proposal velocities = N_full
```

The proposal field remains global and is not semantically truncated.

### 15.5 Transport invariant

```python
v_transport = v_proposal.clone()
v_transport[foreground_ids] = gated_foreground_transport(...)
```

Background dynamic probability zero is bookkeeping only. It must not pass through the foreground-static gate.

### 15.6 Joint-loss invariant

Before evaluating any static/moving, twist-fitting, rigid-consistency, assignment, occupancy, or prototype loss:

```python
proposal = proposal_full[foreground_ids]
rigid = rigid_full[foreground_ids]
p_dynamic = p_dynamic_full[foreground_ids]
```

### 15.7 Export invariant

`gaussian_tracks.pt` must contain:

```text
dynamic_foreground_mask        full bool tensor, true only on semantic foreground
joint_part_field_foreground_ids exact foreground indices
joint_part_field_gaussian_ids   compatibility alias with the same foreground indices
dynamic_prob                    full tensor, zero outside foreground
proposal_velocities             full tensor, including non-foreground proposal motion
```

### 15.8 Final-estimator invariant

Final joint type, axis, pitch, and scores use:

```text
semantic foreground AND moving assignment
```

There is no top-k fallback and no background fallback.

---

## 16. Final joint-type selection correction

An earlier selector rejected prismatic based on a rotation-ratio heuristic before comparing all candidate scores. This produced a screw result even when the direct prismatic normalized residual was four times lower.

The corrected selection rule is:

1. compute the penalized scores for prismatic and revolute;
2. determine the lower-scoring simple model;
3. admit screw only if pitch and axial-motion checks pass **and** screw improves over the best simple model by the configured margin;
4. choose the lowest admissible penalized score.

A small spurious learned angular component must not override a much better direct prismatic fit.

For a pure prismatic model, the direction is identifiable but the perpendicular line location is not. Visualization therefore draws the line along the fitted translation direction through the weighted moving-part canonical centroid.

---

## 17. Correct preflight workflow

### 17.1 Source validation before running

```bash
python tools/verify_semantic_joint_contract.py --repo .
```

This must reject:

```text
fg_mask = torch.ones_like(semantic_mask, dtype=torch.bool)
```

and confirm:

```text
semantic mask passed to initializer
runtime isolation assertion
background proposal-only transport
foreground-only joint loss wrapper
automatic final joint export
```

### 17.2 Start a new run

Use a fresh output folder. Do not reference the old contaminated checkpoint.

### 17.3 Validate immediately after dynamic initialization

The output summary does not exist before the run creates it. A `FileNotFoundError` for:

```text
output/<NEW_RUN>/joint_initialization/joint_initialization_summary.json
```

means the new run has not reached dynamic initialization or the output name is wrong. It does not diagnose the old run.

Once created:

```bash
python tools/verify_semantic_joint_contract.py \
  --repo . \
  --output output/<NEW_RUN>
```

Expected real-dataset values at threshold 0.2 are approximately:

```text
num_gaussians_full        47,792
num_foreground_gaussians  11,137
num_background_gaussians  36,655
distance_matrix_shape     [11,137, 11,137]
```

### 17.4 Inspect the final track export

```bash
python tools/inspect_joint_output.py \
  output/<NEW_RUN>/gaussian_tracks.pt
```

Required:

```text
background IDs inside joint field: 0
foreground IDs missing from joint field: 0
mask disagreement: 0
```

The hard-moving foreground count must be large enough and spatially coherent before running final joint estimation.

---

## 18. Debugging interpretation rules

### 18.1 Flat `W_joint_*` curves

These values are configured loss weights, not loss values. Flat curves are expected.

Inspect instead:

```text
L_joint_*
joint_p_mean
joint_q_mean
joint_label_temperature_used
joint hard occupancy
foreground moving count
proposal/rigid speed
omega/u norms
```

### 18.2 Estimator minimum-point error

```text
Only N semantic-foreground Gaussians are assigned to the moving part
```

is not fixed by lowering `min_moving_points`. It indicates either:

- contaminated joint initialization;
- assignment collapse;
- incorrect semantic export;
- an actually empty moving component.

The estimator guard prevents a plausible-looking but meaningless axis.

### 18.3 Valid semantic field but bad joint result

A correct semantic scalar field does not guarantee correct use. Always distinguish:

```text
semantic field quality
versus
semantic mask integration into the dynamic code
```

The real-dataset incident was an integration bug, not a semantic-training failure.

---

## 19. Current non-negotiable implementation decisions

Do not reintroduce any of the following:

- hard separated label-discovery and rigid-fitting phases;
- all-scene XYZ+DINO joint clustering when semantic filtering is enabled;
- supernodes replacing requested per-Gaussian static DINO initialization;
- semantic truncation of the global proposal field;
- background points contributing to rigid twist or final joint type;
- fixed oversized label temperature;
- rigid fit backpropagating into its proposal teacher;
- uncertain-point velocity attenuation;
- unbounded `omega x x_current + u` teacher propagation;
- top-k/background fallback in final joint estimation;
- prismatic rejection before comparing candidate residual scores.

---

## 20. Recommended real-dataset acceptance criteria

Before trusting a final joint:

1. `joint_initialization_summary.json` reports an `N_fg x N_fg` matrix.
2. Source and output contract validators pass.
3. No background ID belongs to `JointPartField`.
4. Background proposal motion remains visible where rendering requires it.
5. The foreground moving cluster contains substantially more than the estimator minimum.
6. The moving cluster is spatially coherent in the final HTML.
7. The rigid residual is lower than the static residual on the moving foreground.
8. The selected joint type is the minimum admissible penalized score.
9. A rotational axis lies near the object rather than many characteristic radii away.
10. Screw pitch is only reported when screw improves over the best simple model.

---

## 21. Current repair package

The semantic-isolation repair package contains:

```text
RIGEULER3DGS_CONTEXT.md
scene/joint_part_field.py
tools/apply_semantic_joint_isolation_fix.py
tools/verify_semantic_joint_contract.py
tools/inspect_joint_output.py
utils/final_joint_estimation.py
```

The installer:

- backs up all modified source files;
- removes the all-true joint mask;
- replaces the joint field with a foreground-only implementation;
- passes the semantic mask into exact initialization;
- adds runtime ID-equality assertions;
- keeps non-foreground transport proposal-only;
- wraps all joint losses with foreground slicing;
- wires final joint estimation after track export;
- restores backups automatically if any required source pattern cannot be recognized.
