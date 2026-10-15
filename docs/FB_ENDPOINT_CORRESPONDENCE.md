# Endpoint correspondence, missing geometry and visibility — implementation

From `CODEX_DIAGNOSE_ENDPOINT_COMPLEMENTARITY_WITH_IEKF_AND_ESTIMATED_JOINTS.md`.
Read-only. Nothing is merged, trained, pruned, refitted or optimised.

## What is compared

| contrast | F side | B side | observation |
|---|---|---|---|
| A | F0 = initial PLY (anchor) | B0 = backward Phase-1 transport | frame 0 |
| B | FT = forward Phase-1 transport | BT = final PLY (anchor) | frame T |
| C | F0 | BT | frame T depth, only pixels whose MEASURED depth is unchanged since frame 0 |
| A_J1, B_J1 | F0 / FT_J | B0_J / BT | J1: the transported side replaced by the joint-informed transport |

Every state is rendered alone with the production IEKF renderer; contributors
come from the diagnostic kernel of `observation_diagnostics/renderer_access.py`
(verified on every view: sum q = A, depth rebuilt, final T bitwise).
Transported states use each Gaussian's trained orientation
(`CENTER_ONLY_FIXED_ORIENTATION`, shown on every page).

## Estimated joints (J1)

Read from `bidirectional/<d>_rigid/` only: `phase2_rigid/poses_original_order.npy`
(T_part(t), verified T_part(anchor) = I, rigid), `pose_covariances.npy`,
`support_coherence/support.json` (owned rows), `joint_reduction/{one_dof_diagnostics,
relative_increments}.json` (convention `T_{t+1} T_t^-1`, world, verified) and both
`{revolute,prismatic}/geometry.json` hypotheses. The estimator decided no type, so
both are carried; their residual ratio is reported as evidence only.

| estimate | status rule |
|---|---|
| USABLE | proper rigid track with measurable motion of the owned rows |
| NO_MEASURABLE_MOTION | owned rows move < 1 cm and < 0.02 rad over the sequence (cannot inform transport) |
| DEGENERATE | not a rigid track / convention check fails |

Measured: washer forward `NO_MEASURABLE_MOTION` (0.66 mm, 0.0006 rad; its support
owns every row), backward `USABLE` (0.80 m, 1.34 rad; revolute residual 5.6e-2 vs
prismatic 47.8). Storage forward `NO_MEASURABLE_MOTION`, backward `USABLE`
(prismatic s up to 0.54 m).

J1 transport: `X^J = T_part(t) T_part(ref)^-1 X_ref`, rotations left-multiplied,
scales/opacity/colour unchanged, unowned rows static. In the source that did not
produce the estimate, ownership is INFERRED by proximity to the estimated part at
that source's anchor frame (labelled low confidence).

## Patches and labels

Witness pixels: one source explains the observed depth (|D - D_gt| <= 4 cm,
alpha >= 0.5), the other does not. Rows whose depth-compatible contributions land
there (witness U >= 1) are grouped by 3-D proximity; components above 400 rows are
split into 10 cm cells. Four types: `B_ONLY_AT_T`, `F_ONLY_AT_0` (the failing render
is a TRANSPORT), `B_ONLY_AT_0`, `F_ONLY_AT_T` (the failing render is an ANCHOR).

Two stages: every patch is SCREENED with per-row aggregates (pixel count = witness
records, an upper bound); the top 12 by witness mass plus the top 12 proposals per
type are evaluated in DETAIL (exact witness pixels, partners, J1 recovery, occlusion
test at 2–4 frames chosen from the measured depth, sensitivity probe, case study).

J0 rules (declared order, `observation_diagnostics.endpoint_compare.j0_label`; the
estimated joint is not an input):

1. `INSUFFICIENT_EVIDENCE` — < 30 witness pixels or < 2 cameras
2. `TRANSPORT_MISMATCH` — the failing render is a transport, and the patch's rows share the observed surface with the other source at the other endpoint (>= 0.3), or are static and co-located with the other anchor in world
3. `COMMON_SURFACE` — exclusive fraction < 0.25 and >= 0.5 shared: a local fringe
4. `APPEARANCE_SHAPE_DISAGREEMENT` — depth-compatible counterpart, colour-incompatible
5. `MISSING_IN_ANCHOR_PROPOSAL` — the failing render is an anchor
6. `<s>_UNIQUE_OBSERVED_PROPOSAL` — no counterpart at either endpoint and unseen (occluded / out of view / invalid) or unsupported at the other endpoint
7. `CONFLICT_AMBIGUOUS` — otherwise

J1 (`j1_label`): share of the patch's witness pixels the other source explains
after the estimated transport: >= 0.5 `J1_TRANSPORT_EXPLAINS`, <= 0.1
`J1_DOES_NOT_RECOVER`, else `J1_PARTIAL`; `J1_NOT_APPLICABLE_ANCHOR_FAILS` when the
failing render is an anchor. A proposal with `J1_TRANSPORT_EXPLAINS` is listed in
`ambiguous_patches.csv`: J0 and J1 disagree, and neither overrides the other.

Static correspondence (contrast C, `static_candidates`): F0 rows supported on
observed-static pixels whose best BT partner has rho_surf >= 0.3 within the match
radius in WORLD coordinates. Tier `STATIC_COMMON_CANDIDATE` when both free tracks
are static, `STATIC_WORLD_MATCH_TRACK_DISAGREES` when a free track moves the row
(evidence of a transport error).

## GT firewall

`observation_diagnostics/gt_firewall.py`: a Python audit hook over the results and
dataset roots refuses `gt_evaluation*`, `<d>_rigid/summary.json` (it embeds the GT
evaluation), `SUMMARY.txt`, dataset `metadata/`, `pointcloud/`, `mask/`,
`points3D_parts*.ply`, meshes and gt-named files before a byte is read; every
opened file is listed in `provenance.json`. Sources load through
`load_scene(..., inference_only=True)` (no GT part labels).

## Tests — `tests/test_fb_endpoint_correspondence.py`

49 checks pass (36 synthetic through the real renderer, 13 on a real washer run):
renderer equivalence, frozen linearization (adjoint 2e-6; central FD 2.9 % on the
pixels whose contributor list is unchanged), two-layer overlap, static match despite
a bad track, missing surface, pure transport failure recovered by J1, initially
occluded surface revealed at the right frame, joint sign/frame conventions,
untrusted and wrong joints (J0 unchanged), GT firewall (paths, live hook, numpy),
identities, immutability, repeatability and resume, row accounting.

## Measured runs (shared GPU, `--gpu-memory-fraction 0.2`)

| scene | renders | time | GPU peak | host peak |
|---|---|---|---|---|
| washer_103776, 8 cameras | 48 endpoint/J1 + intermediate + probes | 3.8 min | 1.4 GB | 8.1 GB |
| storage_45135, 8 cameras | 48 + 66 intermediate + probes | 7.9 min | ≤ 1.6 GB | 10.5 GB |

## Commands

```bash
cd ~/Joint3DGS/RigEuler3DGS
bash scripts/run_fb_endpoint_correspondence.sh --scene washer_103776 --frames endpoints --cameras all --estimated-joints auto --tag endpoint_evidence_v1 --preflight
bash scripts/run_fb_endpoint_correspondence.sh --scene washer_103776 --frames endpoints --cameras all --estimated-joints auto --tag endpoint_evidence_v1 --dry-run
bash scripts/run_fb_endpoint_correspondence.sh --scene washer_103776 --frames endpoints --cameras all --estimated-joints auto --tag endpoint_evidence_v1 --execute
```

Output: `<results>/<scene>/endpoint_correspondence/<tag>/`; start with
`visuals/overview.html` and `endpoint_summary.md`.
