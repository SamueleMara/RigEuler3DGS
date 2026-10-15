# Endpoint correspondence: evaluation-only GT metrics and the all-scenes runner

From `CODEX_ADD_EVALUATION_ONLY_GT_METRICS_AND_ALL_SCENES_RUNNER.md`.

## Isolation boundary

```
trained PLYs + Phase-1 tracks + ESTIMATED joints + recorded RGB-D
        -> tools/analyze_fb_endpoint_correspondence.py   (GT firewall active, unchanged decisions)
        -> sealed outputs (run_state.json complete: true)
        -> tools/evaluate_fb_endpoint_correspondence_gt.py  (separate process, reads dataset GT)
        -> <diagnosis>/evaluation_gt/  only
```

* The diagnosis never imports `observation_diagnostics/gt_evaluation.py` (checked at runtime and by
  source scan) and keeps its audit-hook firewall over `metadata/`, `pointcloud/`, `mask/`, GT joint
  evaluations and meshes.
* The evaluator refuses an unsealed diagnosis or a symlinked/escaping `evaluation_gt/`, hashes every
  primary file before and after (2,854 files on washer) and fails if any changed. GT numbers are never
  written into a primary table; patch IDs and original Gaussian rows are join keys for reporting only.
* Regression (`tests/test_fb_endpoint_gt_evaluation.py`, 51 checks): the REAL diagnosis run on washer
  data with (A) real GT, (B) altered GT (another axis, point and q scale) and (C) no GT gives
  identical primary outputs (559 JSON/CSV files: IDs, labels, J0/J1, numbers to 6 digits; the only
  run-to-run difference found, a Python-set key order, was made deterministic), while the GT scores of
  A and B differ and C reports `GT_UNAVAILABLE`.

## Ground truth actually present (all ten scenes)

| file | meaning | used for |
|---|---|---|
| `metadata/joint_info.json` | selected joint name, type, limits | joint type |
| `pointcloud/joint_metadata.json` | axis_vector, point_of_application, `render_world` | axis line |
| `metadata/frame_values.csv` | q per original frame (rad / m) | coordinate + pose errors |
| `pointcloud/points3D_parts.ply` | frame-0 surface samples, part id | support, geometry, patches |
| `pointcloud/points3D_parts_final.ply` | frame-T surface samples, part id | support, geometry, patches |
| `mask/<cam>/frame_*.png` | binary OBJECT silhouette (not a part mask) | "inside object mask" only |
| `metadata/cameras.json` | render camera positions | gauge check |

Gauge: COLMAP camera centres equal `cameras.json` positions to 4e-15 m on every scene checked, so
`render_world` is the 3DGS / estimated-joint world (no alignment is fitted). GT motion convention
X_t = M(+(q_t - q_0)) about `point_of_application`, verified from GT alone where the frame-0 export
has moving-link samples (oven, storage, laptop: landing medians 1.8-3.5 cm vs 25-99 cm for the
opposite sign); washer's frame-0 export has none, so it is reported `NOT_VERIFIABLE_IN_THIS_SCENE`
and uses that convention.

## Metrics (all `gt_evaluation_metric` unless stated)

* `metrics_joint.csv` — per direction x {revolute, prismatic, phase2_rigid_track}: line angle
  `acos(|a.b|)`, line-line distance (robust when parallel), coordinate MAE/RMSE referenced to the
  direction's anchor (no fitted offset/scale; the estimator's own sign from its pose track, the axis
  sign from the GT axis -> `oracle_metric`), pose rotation geodesic and point-transfer error of the GT
  moving-link samples. `UNDECIDED` types are scored individually, never chosen.
* `metrics_support.csv` — `predicted_support_dynamic`, `rigid_phase2_support_members`,
  `endpoint_static_world_matched` scored separately; GT labels per row from the nearest GT sample
  within 1 / 2 / 5 cm of the row's anchor position, unlabelled rows excluded; frame 0 has two
  mappings (`as_given`, `kinematic`).
* `metrics_geometry.csv` — F0, B0, FT, BT centres vs GT samples (all / static link / moving link):
  one-sided precision and recall, F-score, at 1 / 2 / 5 cm.
* `metrics_patches.csv` — every frozen patch: on-GT-surface fraction, part composition, inside the
  object mask, coverage of the nearby GT samples by the opposite render, and by the opposite ANCHOR
  moved by the GT motion; a post-hoc reading at 1 / 2 / 5 cm. Patches whose rows were not saved
  (screened-only patches in outputs before the `rows_list` fix) are `ROWS_NOT_SAVED`.

## Washer (existing `endpoint_evidence_v2`)

| direction / hypothesis | line angle | line dist | coordinate error | pose error |
|---|---|---|---|---|
| backward revolute (USABLE) | 0.04 deg | 3.6 cm | 0.014 rad MAE | Phase-2 track: 0.64 deg, 3.0 cm |
| forward (NO_MEASURABLE_MOTION) | 12.8 deg | 4.5 cm | 0.67 rad (estimate never moved) | 38 deg, 23 cm |

Backward dynamic support: precision 1.000, recall 0.986 (2 cm). Static world-matched groups: their
F and B rows lie on the GT static link (median 1.000). The post-hoc patch reading is
tolerance-sensitive: at 2 cm most B-only-at-T patches read "opposite lacks it", at 5 cm almost all
read "the forward model has it once moved by the GT motion" — i.e. transport, including several that
J0 proposed as B-unique. See `evaluation_gt/gt_evaluation_report.md`.

## Commands

```bash
python tools/evaluate_fb_endpoint_correspondence_gt.py --diagnosis-dir '<results>/<scene>/endpoint_correspondence/<tag>' --dataset-dir '<dataset>/<scene>' [--dry-run] [--resume] [--replace-evaluation]

bash scripts/run_fb_endpoint_correspondence_all.sh --results-root '<results>' --dataset-root '<dataset>' --tag endpoint_eval_v3 --estimated-joints auto --evaluate-gt --list-scenes
bash scripts/run_fb_endpoint_correspondence_all.sh ... --preflight-only
bash scripts/run_fb_endpoint_correspondence_all.sh ... --execute [--resume] [--gpu-memory-fraction 0.2]
```

Resume: rerun the same `--execute` command with `--resume`; finished scenes and evaluations are
reused, a changed fingerprint is refused (new tag, or `--replace-evaluation` for an evaluation, which
keeps the old one under `evaluation_gt/superseded/`). Batch tables:
`<results>/endpoint_correspondence_batch/<tag>/`.
