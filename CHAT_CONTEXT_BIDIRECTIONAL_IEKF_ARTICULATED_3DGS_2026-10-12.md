# Chat Context — Bidirectional IEKF + Articulated 3DGS
**Date:** 2026-10-12  
**Project repo:** `/home/samuelemara/Joint3DGS/RigEuler3DGS`

---

# 1. Main project goal

Build a robust articulated-object reconstruction pipeline from multiview RGB-D and 3D Gaussian Splatting that:

1. reconstructs static endpoint 3DGS models;
2. estimates free per-Gaussian motion with the existing IEKF;
3. extracts a high-confidence rigid moving support using trajectory coherence;
4. estimates an exact rigid-body `SE(3)` trajectory for the moving part;
5. reduces that rigid trajectory post-hoc to a physical 1-DoF joint:
   - prismatic;
   - revolute;
6. handles asymmetric visibility by running the pipeline backward from the final endpoint;
7. merges complementary initial/final endpoint geometry into a more complete articulated 3DGS;
8. eventually performs a full-window / IEKS-like joint-constrained refinement.

The user strongly prefers:
- structural/manifold formulations;
- no arbitrary thresholds unless mathematically unavoidable;
- no GT leakage into production logic;
- explicit uncertainty/provenance;
- detailed visual and numerical debug;
- building strictly on working code rather than rewriting it;
- false-positive moving Gaussians treated as more dangerous than false negatives.

---

# 2. Current desired architecture

The current preferred architecture is:

\[
\boxed{
\text{static endpoint 3DGS}
\rightarrow
\text{Phase 1 free motion}
\rightarrow
\text{coherence moving support}
\rightarrow
\text{exact rigid }SE(3)\text{ Phase 2}
\rightarrow
\text{post-hoc joint reduction}
}
\]

For the rigid moving part:

\[
T_t=
\begin{bmatrix}
R_t&p_t\\
0&1
\end{bmatrix}
\in SE(3).
\]

For moving Gaussian \(i\):

\[
\mu_{i,t}=R_t\mu_i^0+p_t,
\]

\[
R_{i,t}=R_tR_i^0,
\]

\[
\Sigma_{i,t}=R_t\Sigma_i^0R_t^\top.
\]

Static Gaussians remain canonical but are still rendered for:
- visibility;
- alpha compositing;
- RGB;
- depth;
- occlusion.

Rigidity is exact by construction, not a soft pairwise penalty.

---

# 3. IEKF / rigid-state convention

Rigid state:

\[
x_t=(T_t,\xi_t),
\qquad
\xi_t=[v_t;\omega_t]\in\mathbb R^6.
\]

Local state perturbation:

\[
\delta x_t=[\delta\tau_t,\delta\xi_t]\in\mathbb R^{12}.
\]

The implementation uses **left/spatial perturbations**:

\[
T'=\Exp(\delta\tau^\wedge)T.
\]

For a Gaussian center:

\[
\delta\mu_i
=
\delta v+\delta\omega\times\mu_i.
\]

The rigid Gaussian Jacobian block is:

\[
J_{i,T}
=
\begin{bmatrix}
I&-[\mu_i]_\times\\
0&I
\end{bmatrix}.
\]

Renderer bridge:

\[
H_T=H_GJ_{G,T}.
\]

Analytic VJP:

\[
\bar v=\sum_i\bar\mu_i,
\]

\[
\bar\omega
=
\sum_i
\left(
\mu_i\times\bar\mu_i+\bar\phi_i
\right).
\]

Keep the existing:
- robust RGB-D residual;
- alpha/visibility logic;
- effective measurement covariance;
- information-form IEKF;
- trust region;
- nonlinear rerender acceptance;
- analytic JVP/VJP.

Do not introduce a new image loss.

---

# 4. Phase-1 / coherence support policy

Production support must be the high-confidence:

```text
MOVING_RIGID
```

core from the coherence branch.

Do not send:

```text
INCOHERENT_OR_AMBIGUOUS
```

into rigid Phase 2.

The policy is precision-first:

\[
\boxed{
FP_{\rm dynamic}
\gg
FN_{\rm dynamic}
}
\]

in downstream damage.

A false positive forces a truly static Gaussian to move with the rigid transform and directly corrupts:
- pose;
- axis;
- displacement;
- render gradient;
- covariance.

A false negative only removes useful evidence.

Important distinction in the coherence outputs:

- broad binary `DYNAMIC` component;
- high-confidence `MOVING_RIGID` core.

The broad dynamic fraction is useful diagnostically but must **not** automatically become the Phase-2 frozen support.

---

# 5. Coherence model

For edge \(i,j\):

\[
r_{ij}(t)=x_j(t)-x_i(t),
\qquad
\ell_{ij}(t)=\|r_{ij}(t)\|.
\]

Reference length:

\[
\bar\ell_{ij}
=
\operatorname{median}_t\ell_{ij}(t).
\]

Normalized metric strain:

\[
z_{ij}(t)
=
\log
\frac{\ell_{ij}(t)+\epsilon_\ell}
{\bar\ell_{ij}+\epsilon_\ell}.
\]

Coherence score:

\[
C_{ij}
=
\sqrt{
\frac{
\sum_t w_{ij}(t)z_{ij}(t)^2
}{
\sum_t w_{ij}(t)+\epsilon
}
}.
\]

Differential identity:

\[
\epsilon_{ij}(t)
=
\frac{
r_{ij}^\top(v_j-v_i)
}{
\|r_{ij}\|^2
}.
\]

The current corrected coherence logic uses persistent coherent components and classes:

```text
MOVING_RIGID
STATIC_COMPATIBLE
INCOHERENT_OR_AMBIGUOUS
```

The optimized GT is diagnostics/evaluation only.

---

# 6. Static 3DGS background issue and semantic field

Earlier static 3DGS models contained black/high-opacity background floaters.

Root cause:

\[
I^{render}
=
\alpha c+(1-\alpha)b.
\]

For:
- black Gaussian color;
- black background;
- black target;

the RGB residual can be zero for any opacity.

The semantic field originally used raw alpha compositing:

\[
S(p)=\sum_iw_i(p)s_i
\]

without normalizing by accumulated alpha:

\[
A(p)=\sum_iw_i(p).
\]

Correct semantic field should be:

\[
\boxed{
\hat s(p)
=
\frac{S(p)}{A(p)+\epsilon}
}
\]

with geometry/opacity detached from semantic supervision.

Current intended semantic loss:

\[
L_{\rm sem}
=
\frac{
\sum_p A(p)\,
\mathrm{BCE}(\hat s(p),M(p))
}{
\sum_p A(p)+\epsilon
}.
\]

No arbitrary alpha threshold.

The user explicitly does not want arbitrary thresholds; at most a principled continuous depth cue if necessary.

Randomized background carving is only for static 3DGS training.  
The IEKF itself should continue to render using the actual observation background convention (black for these datasets).

---

# 7. Endpoint static models

Dataset root:

```bash
R="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
```

Static endpoint-model root:

```bash
O="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours"
```

Per scene:

```text
$O/<scene>/static_bidi/initial
$O/<scene>/static_bidi/final
```

Expected final PLY:

```text
<model>/point_cloud/iteration_30000/point_cloud.ply
```

All static endpoint trainings are now finished.

For Washer:

```text
initial: 21338 Gaussians
final:   24582 Gaussians
```

Endpoint provenance was verified:

```text
forward  + initial -> colmap       -> original frame 0
backward + final   -> colmap_final -> original frame 59
```

Do not assume equal Gaussian counts or row correspondence between initial and final models.

---

# 8. Bidirectional formulation

Forward body pose:

\[
\boxed{
T_t^F=B_tB_0^{-1}
}
\]

with:

\[
T_0^F=I.
\]

Backward body pose:

\[
\boxed{
T_t^B=B_tB_T^{-1}
}
\]

with:

\[
T_T^B=I.
\]

Endpoint transform:

\[
X=B_TB_0^{-1}.
\]

From the backward solution:

\[
\boxed{
X=(T_0^B)^{-1}.
}
\]

Backward pose converted into the initial gauge:

\[
\boxed{
\widetilde T_t^B
=
T_t^B X
=
T_t^B(T_0^B)^{-1}.
}
\]

Do **not** use:

\[
XT_t^B.
\]

Multiplication order matters.

Endpoint cycle after the completed forward pass:

\[
\boxed{
T_T^{F,2}T_0^B\approx I.
}
\]

Per-frame F/B discrepancy:

\[
\Delta_t^{FB}
=
\Log
\left[
T_t^{F,2}
(\widetilde T_t^B)^{-1}
\right]^\vee.
\]

---

# 9. Temporal ordering convention for backward runs

Backward processing order:

```text
59, 58, ..., 0
```

Original chronological order:

```text
0, 1, ..., 59
```

Critical objects must remain explicitly distinct:

```text
canonical_xyz
positions_processing_order
positions_original_order
```

For backward:

```text
canonical_xyz = final endpoint Gaussian geometry
positions_processing_order = [x_59, ..., x_0]
positions_original_order   = [x_0, ..., x_59]
```

Do not infer canonical geometry from the first chronological frame.

For backward the canonical geometry is always:

\[
\boxed{
\mathcal G_T
}
\]

even though the chronological trajectory begins at original frame 0.

---

# 10. Backward velocity convention

The backward IEKF walks in reverse processing time with positive `dt`.

The internal backward velocity is therefore a reverse-time state and must **not** simply be reversed.

For chronological comparison, reconstruct from chronological poses:

\[
\boxed{
\xi_t^{B,\mathrm{chron}}
=
\frac1{\Delta t_t}
\Log
\left(
T_{t+1}^B(T_t^B)^{-1}
\right)^\vee.
}
\]

Recommended artifact semantics:

```text
velocities_processing_order.npy
chronological_twists.npy
```

Avoid a generic `velocities.npy` whose temporal semantics differ from `poses.npy`.

---

# 11. Covariance convention

For right multiplication by a fixed gauge transform:

\[
\widetilde T=TX
\]

with left/spatial perturbations:

\[
\boxed{
P_{\widetilde T}=P_T.
}
\]

For inversion:

\[
X=T^{-1},
\]

the covariance must transform as:

\[
\boxed{
P_{T^{-1}}
=
\operatorname{Ad}_{T^{-1}}
P_T
\operatorname{Ad}_{T^{-1}}^\top.
}
\]

Do not blindly copy covariance through inversion.

Also do not call a full state covariance "chronological" simply by reversing frame order if the state contains reverse-time velocity blocks.

---

# 12. Washer asymmetric visibility case

Washer is the primary asymmetric test.

Sparse endpoint diagnostic:

```text
initial sparse endpoint:
0 / 4594 moving-labeled points

final sparse endpoint:
489 / 4975 moving-labeled points
```

Interpretation:

\[
\boxed{
\text{initial moving geometry is unreconstructed / unusable}
}
\]

while the final endpoint contains usable moving geometry.

This is diagnostics only; GT must not select production support.

The asymmetric architecture is:

\[
\boxed{
\mathcal G_T
\rightarrow
\text{backward Phase 1}
\rightarrow
\text{coherence}
\rightarrow
\text{backward rigid }SE(3)
}
\]

then:

\[
\boxed{
\mathcal M_T
\xrightarrow{T_0^B}
\mathcal M_T^{T\rightarrow0}
}
\]

then:

\[
\boxed{
\mathcal G_{0,\mathrm{complete}}
=
\mathcal G_0
\cup
\mathcal M_T^{T\rightarrow0}
}
\]

then a second forward rigid pass:

\[
\boxed{
\mathcal G_{0,\mathrm{complete}}
\rightarrow
T_t^{F,2}.
}
\]

The second forward pass should use the backward-derived/transformed moving support as a known frozen support. It should not require free Phase 1 to rediscover the missing moving part.

---

# 13. Washer backward Phase-1 / coherence result

A real backward Phase-1 + coherence run completed successfully.

Runtime:

```text
~842 s
```

The ordering fix was verified:

```text
canonical_xyz == centers_frame_0059 exactly
max diff = 0.0

canonical_xyz != centers_frame_0000
mean abs diff ≈ 0.077
max abs diff  ≈ 0.830
```

So the final model is truly the canonical geometry, while saved trajectories are chronological.

Coherence result:

```text
status: RESOLVED

high-confidence extractor:
MOVING_RIGID                13 supernodes
STATIC_COMPATIBLE           54 supernodes
INCOHERENT_OR_AMBIGUOUS    125 supernodes

broad binary components:
component 0 DYNAMIC: 35 nodes
component 1 STATIC: 157 nodes
```

Broad binary rigid statistics:

```text
DYNAMIC:
E_rigid     = 1.8350e-02
E_identity  = 4.1950e-02
gain        = 2.3600e-02

STATIC:
E_rigid     = 8.1239e-03
E_identity  = 8.2415e-03
gain        = 1.1755e-04

gain ratio ≈ 200.8
```

The broad:

```text
predicted_dynamic_fraction ≈ 0.20873
```

must not automatically be used as rigid Phase-2 support.

Production rigid support must be:

\[
\boxed{
\texttt{MOVING\_RIGID core only}
}
\]

unless the current coherence artifact explicitly proves otherwise.

---

# 14. Important predicted-support artifact issue

A diagnostic check queried:

```python
support.get("members")
support.get("member_indices")
```

and printed:

```text
members: 0
```

This does **not** mean the support is empty.

Historical support artifacts use keys such as:

```text
dynamic_gaussians
static_gaussians
```

and the corrected high-confidence artifact should preferably expose:

```text
moving_rigid_gaussians
```

Before rigid Phase 2, the code must explicitly prove which key is consumed.

Required contract:

```python
members = support["moving_rigid_gaussians"]
```

or the exact equivalent in the current artifact.

Assert:

```text
len(members) > 0
all unique
min >= 0
max < num_final_gaussians
```

Persist:
- exact member count;
- support hash;
- final-model Gaussian indices.

Do not silently feed the broad `dynamic_gaussians` set if it contains the 35-node broad DYNAMIC component rather than the 13-node `MOVING_RIGID` core.

---

# 15. Backward rigid Phase 2 status

Codex patched:

```text
tools/run_rigid_se3_pipeline.py
```

to support:

```text
--direction backward
```

Key intended semantics:
- observations processed 59 -> 0;
- canonical model = final endpoint;
- Phase-1 centers loaded from disk in chronological order and reversed internally for solve;
- rigid solve warm-starts at identity at the canonical final frame;
- saved rigid poses are restored to chronological order;
- pose covariance is restored to chronological order;
- backward chronological twists are recomputed from chronological poses.

The patch was compiled and committed.

However, at the latest point in this chat, **the real backward rigid Phase-2 result had not yet been analyzed here**.

The latest user-uploaded file is:

```text
/mnt/data/backward_rigid.zip
```

This should be treated as the newest backward rigid result artifact and analyzed next.

Do not assume success until the zip contents are inspected.

Required checks on that artifact:
- frozen support count/hash;
- whether support is `MOVING_RIGID` only;
- `T_59^B ≈ I`;
- exact pairwise rigidity drift;
- orientation rigidity defect;
- RGB/depth energy before/after;
- accepted/rejected IEKF steps;
- Phase1 Kabsch vs rigid result;
- pose covariance;
- chronological pose ordering;
- chronological twist consistency;
- any post-hoc P/R reduction outputs.

---

# 16. Known artifact-ordering issue still to check in rigid pipeline

Inside `run_rigid_se3_pipeline.py`, after loading chronological Phase-1 centers and reversing them for the backward solve, the code historically saved:

```python
for t, p in enumerate(positions):
    save(f"centers_frame_{t:04d}.npy", p)
```

If `positions` is in processing order during backward solve, this can silently produce:

```text
centers_frame_0000 = x_59
centers_frame_0001 = x_58
...
```

inside the new rigid output.

The correct pattern is to keep separate:

```text
positions_chronological
positions_processing
```

The solver uses `positions_processing`.

Any file named by original frame index must use `positions_chronological`.

This must be checked in the current code/result.

---

# 17. Known rigid-output naming issue still to check

The current rigid pipeline may contain temporally mixed artifacts:

```text
poses.npy                  chronological
pose_covariances.npy       chronological
velocities.npy             reverse-processing-time
chronological_twists.npy   chronological
state_covariances.npy      unclear
phase1_kabsch_poses.npy    needs audit
state_trace / records      needs audit
```

Preferred semantics:

```text
poses_processing_order.npy
poses_original_order.npy
pose_covariances_original_order.npy
velocities_processing_order.npy
chronological_twists.npy
state_covariances_processing_order.npy
```

Do not label a full state covariance as chronological unless the velocity block and cross-covariances are transformed consistently.

For joint reduction and F/B pose comparison, use:
- chronological poses;
- chronological pose covariance.

---

# 18. Backward post-hoc joint reduction

The same existing production joint reducer should consume:

\[
\{T_t^B,P_t^B\}
\]

in original chronological order.

Do not:
- fit directly from free Phase-1 tracks;
- invent a new backward joint fitter;
- average forward/backward axes.

Backward joint result remains endpoint-final-gauge result until gauge alignment/completion is done.

---

# 19. Post-hoc joint reduction model

For the rigid trajectory:

\[
\Delta T_t=T_{t+1}T_t^{-1}
\]

and:

\[
\zeta_t
=
\frac1{\Delta t_t}
\Log(\Delta T_t)^\vee.
\]

Dimensionless twist:

\[
\bar\zeta_t=[v_t/L;\omega_t].
\]

Weighted SVD:

\[
\gamma_{1DOF}
=
\frac{\sigma_1^2}
{\sum_k\sigma_k^2}.
\]

This is a structural diagnostic, not an arbitrary threshold classifier.

Prismatic fit:

\[
T_P(s_t)=
\begin{bmatrix}
I&s_te_P\\
0&1
\end{bmatrix}.
\]

Revolute direction from finite rotations:

\[
\phi_t=\Log(R_t)^\vee,
\qquad
\phi_t=\theta_te_R.
\]

Finite axis-point solve:

\[
p_t=(I-R_t)a.
\]

Solve:

\[
a^\star
=
\arg\min_a
\sum_t
w_t
\|(I-R_t)a-p_t\|^2
\]

with gauge:

\[
e_R^\top(a-c_0)=0.
\]

Final pose-space residual:

\[
r_t^M
=
\Log
\left[
T_M(q_t;\theta_M)^{-1}\hat T_t
\right]^\vee.
\]

Do not choose P/R simply by minimum energy because revolute contains prismatic in its far-axis closure.

---

# 20. Previous rigid-forward results

The rigid-first architecture was previously validated strongly on:

```text
Stapler  revolute
USB      revolute
Fridge   revolute
Blade    prismatic
Storage  prismatic
```

Representative results:

```text
scene           GT         gamma1DOF        E_P        E_R
stapler         revolute      0.9954   2.821e10   6.823e05
blade           prismatic     0.9934   1.120e04   3.994e04
USB             revolute      0.9964   1.898e09   3.376e05
fridge          revolute      0.9223   1.552e09   2.800e06
storage         prismatic     0.9643   1.069e06   9.227e05
```

Exact rigidity drift was around:

```text
1e-15
```

and orientation defect was zero.

This validated:
- rigid-first estimation;
- exact rigid shared transform;
- post-hoc reduction;
- Kabsch as initializer only.

Known model-selection issue:
- revolute can mimic prismatic in the far-axis limit;
- do not choose only `min(E_P,E_R)`;
- use statistically resolved rotational evidence / covariance / nested-model reasoning.

---

# 21. Moving geometry completion

After a successful backward rigid solve:

\[
T_0^B=B_0B_T^{-1}.
\]

For final moving Gaussian \(j\):

\[
\boxed{
\mu_j^{T\rightarrow0}
=
R_0^B\mu_j^T+p_0^B
}
\]

\[
\boxed{
R_j^{T\rightarrow0}
=
R_0^BR_j^T
}
\]

\[
\boxed{
\Sigma_j^{T\rightarrow0}
=
R_0^B\Sigma_j^T(R_0^B)^\top.
}
\]

No point-cloud registration should be estimated before this transform.

The rigid trajectory already gives the body alignment.

Preserve provenance for every merged Gaussian:

```text
completed_index
source_endpoint
source_model_path
source_gaussian_index
source_support_class
transformed
transform_source_frame
transform_target_frame
```

---

# 22. First asymmetric completed model

For Washer, first milestone:

\[
\boxed{
\mathcal G_{0,\mathrm{complete}}
=
\mathcal G_0
\cup
T_0^B\mathcal M_T.
}
\]

Do not block this on sophisticated static endpoint fusion.

Do not deduplicate before checking raw rendering unless duplicates make rendering impossible.

Raw-union-first.

---

# 23. Second forward rigid pass

After completion:

```text
completed initial model
    ->
known transported moving support
    ->
existing forward exact rigid-SE(3) IEKF
    ->
T_t^{F,2}, P_t^{F,2}
```

Do not require free forward Phase 1 to rediscover the moving support.

The backward-derived transported moving support should be frozen.

This second forward run is the authoritative forward comparison for asymmetric Washer.

---

# 24. Forward/backward comparison

After the completed forward run:

\[
X=(T_0^B)^{-1}.
\]

Convert:

\[
\widetilde T_t^B=T_t^BX.
\]

Compare:

\[
T_t^{F,2}
\quad\text{vs}\quad
\widetilde T_t^B.
\]

Endpoint cycle:

\[
T_T^{F,2}T_0^B\approx I.
\]

Relative increments:

\[
\Delta T_t^F
=
T_{t+1}^{F,2}(T_t^{F,2})^{-1},
\]

\[
\Delta T_t^B
=
\widetilde T_{t+1}^B(\widetilde T_t^B)^{-1}.
\]

Then compare:

\[
\zeta_t^F
=
\frac1{\Delta t_t}\Log(\Delta T_t^F)^\vee,
\]

\[
\zeta_t^B
=
\frac1{\Delta t_t}\Log(\Delta T_t^B)^\vee.
\]

---

# 25. Do not use naive F/B covariance addition as final estimator

Forward and backward estimates are correlated because they use overlapping RGB-D observations.

Therefore:

\[
(P_F^{-1}+P_B^{-1})^{-1}
\]

must not be called the final statistically correct posterior.

It may be used only as:
- diagnostic;
- initializer;
- approximate local fusion.

Final strong method should be a full-window MAP / IEKS-style solve using the original observations once.

---

# 26. Full-window final stage

After:
- backward solve;
- completion;
- completed-forward solve;
- F/B alignment;
- common joint estimation;

run one shared full-window optimizer.

Rigid objective:

\[
E
=
\sum_{t,c}
\frac12
\|y_{t,c}-h_c(T_t)\|_{R^{-1}}^2
+
\sum_t
\frac12
\|r_{\rm process}\|_{Q^{-1}}^2.
\]

Later joint-constrained:

\[
T_t=T_J(q_t;\theta_J).
\]

Use existing analytic renderer JVP/VJP.

Do not use Adam.

---

# 27. Rasterizer policy

Do not replace the current custom `diff-gaussian-rasterization` submodule while implementing the bidirectional pipeline.

Current code already depends on:
- custom feature-field rendering;
- current RGB/depth semantics;
- current semantic/DINO/SAM3 paths;
- current analytic IEKF JVP/VJP.

If accumulated alpha is needed, rendering an all-ones feature over black background is acceptable when it matches the existing compositor.

Do not equate current `invdepth` with another fork's `rendered_depth` without verifying exact semantics.

---

# 28. GT diagnostics

Initial/final sparse label files:

```text
pointcloud/points3D_parts.ply
pointcloud/points3D_parts_final.ply
```

A loader bug was found because final-endpoint GT lookup only searched the initial filename.

The loader was patched to resolve the `_final` variant for `colmap_final`.

Verified diagnostic sparse counts for Washer:

```text
colmap:
4594 labels, 0 moving

colmap_final:
4975 labels, 489 moving
```

Existing trained final PLYs do not necessarily have `gt_part` baked into them.

Do not retrain only for visualization.

GT can be shown as a separate sparse diagnostic overlay or mapped post-hoc, clearly marked as evaluation-only.

---

# 29. Git hygiene

Codex has repeatedly used:

```bash
git add -A
```

despite explicit instructions not to.

Do not do this.

The repo contains unrelated dirty state/submodules.

Before each commit:

```bash
git status --short
git diff -- <explicit files>
git add <explicit files>
```

Audit recent commit:

```bash
git show --stat --name-status a8efab2
```

and ensure no unrelated changes were committed.

Do not discard user work.

---

# 30. Latest implementation files/specs created in this chat

Key implementation specs created:

```text
/mnt/data/CODEX_BIDIRECTIONAL_IEKF_AND_ARTICULATED_3DGS_MERGE.md

/mnt/data/CODEX_IMPLEMENT_BIDIRECTIONAL_RUNTIME_WASHER_COMPLETION.md

/mnt/data/CODEX_FINISH_BIDIRECTIONAL_PIPELINE_AFTER_STATIC_TRAINING.md
```

Earlier rigid-first spec:

```text
/mnt/data/CODEX_IMPLEMENT_RIGID_SE3_PHASE2_AND_POSTHOC_JOINT_REDUCTION.md
```

---

# 31. Immediate next task after loading this context

The most recent user action before creating this context was:

```text
uploaded: /mnt/data/backward_rigid.zip
request: analyze the result
```

That archive has **not yet been analyzed in this context summary**.

The next assistant should:

1. inspect `/mnt/data/backward_rigid.zip`;
2. identify all contained artifacts;
3. determine whether backward rigid Phase 2 actually ran;
4. verify exact support used;
5. verify ordering/gauge:
   - `T_59^B ≈ I`;
   - poses chronological;
   - processing-order poses separate;
6. verify rigidity numerically;
7. inspect energies, acceptance, covariance;
8. inspect Phase1 Kabsch vs rigid result;
9. inspect any joint reduction outputs;
10. decide whether it is safe to proceed to:
   \[
   T_0^B\mathcal M_T
   \]
   model completion.

Do not assume the Phase-2 result is correct solely because Codex said the code was implemented.

---

# 32. User response style

The user prefers:
- Italian for discussion;
- precise mathematical analysis;
- direct identification of errors;
- no unnecessary simplification;
- implementation-faithful recommendations;
- explicit commands and Codex-ready specs when useful;
- no arbitrary thresholds;
- no unnecessary questions when the requested task is clear.

When evaluating results, distinguish clearly between:

```text
implemented
unit-tested
real-data tested
passed
failed
not yet implemented
```

Never claim a run succeeded unless the actual artifacts/logs support it.
