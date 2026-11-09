# FreeArtGS vs. RigEuler3DGS — complete numerical comparison and Codex plotting specification

**Purpose:** instruct Codex to implement a deterministic, reproducible **Matplotlib-only scientific-figure generator** using the provided numerical reports. Produce paper-style numerical charts, *not* illustrations, infographic cards, icons, hero banners, rendered objects, or interpretive judgments.

**Evaluation scope:** 10 identical synthetic articulated scenes (8 cameras, 60 timestamps), local calibrated multiview setting. The FreeArt data are **not** the official FreeArt-21 monocular benchmark. The primary RigEuler result uses `v5_canonical` in 9 scenes and `v5_gates_canonical` for `washer_103776`; keep `v4_bootstrap` separate as an optional within-RigEuler ablation. Do not conflate the latter with the main comparison.

**Ground-truth availability:** the synthetic benchmark has GT for joint type, axis, relative joint state and motion range. The separate real-scene file has only visually inferred expected joint types, **not** numerical GT; never report numerical real-scene joint errors.

**Mandatory instruction to Codex:** follow the definitions, conversions, data splits and figure filenames in this specification without adding new metrics or claims. Any unavailable metric must be labeled `N/A`, not inferred.

## 1. Source data and identifiers

Use these source files (present among the conversation attachments and included in the companion plotting-input ZIP):

| Input name | Method / contents | Primary fields |
|:---|:---|:---|
| `FreeArtGS_summary.csv` | FreeArtGS, 10 synthetic scenes | `scene`, `joint_type`, `axis_error_deg`, `axis_line_error_mm`, `state_mae`, `state_max_error`, `state_unit`, `predicted_motion_range`, `gt_motion_range`, `gaussians`, logged rendering metrics |
| `FreeArtGS_report.json` | FreeArtGS synthetic report | `scenes[].saved_joint_evaluation.joint_state_samples` contains 60 genuine GT/predicted samples per scene |
| `FreeArtGS_per_observation_psnr.csv` | Per-camera, per-frame final foreground PSNR (7 scenes only) | `scene`, `camera`, `time_index`, `foreground_psnr_db` |
| `RigEuler3DGS_synthetic_results.csv` | RigEuler synthetic 20 rows: 10 primary + 10 v4 | `object`, `run`, `gt_type`, `type_decision_correct`, `gt_hyp_axis_err_deg`, axis-line and phase-3 state metrics |
| `RigEuler3DGS_columns.csv` | Definitions of RigEuler result fields | metric descriptions |
| `RigEuler3DGS_real_results.csv` | RigEuler real-only 8 scenes × 3 support arms | non-GT diagnostics; not a FreeArt comparison |
| `freeart_rigeuler_comparison.csv` | Normalized joined 10-row table shipped alongside this document | method-prefixed metrics in common units |

**Join key:** `FreeArtGS_summary.csv.scene == RigEuler3DGS_synthetic_results.csv.object`, case-sensitive; do not join on row order. Main RigEuler selection: `run == 'v5_canonical'`, except `washer_103776` where `run == 'v5_gates_canonical'`. Assert exactly 10 matched rows, exactly one main RigEuler row per scene, 8 `revolute` and 2 `prismatic`.

## 2. Main synthetic numerical results

### Table 1 — Joint type, axis direction and axis-line position

**Units:** axis-direction error in degrees; axis-line distance in millimeters. `—` means not geometrically defined (prismatic axis-line position). For a revolute joint, axis-line distance is the shortest distance between two infinite axis lines; it is **not** a pivot-point Euclidean position error. Lower is better for the two error metrics.

| Scene | Joint | FreeArt axis (°) | RigEuler axis (°) | FreeArt line (mm) | RigEuler line (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `foldchair_102255` | Rev | 0.00172 | 0.0265 | 0.399 | 1.838 |
| `laptop_10211` | Rev | 0.01812 | 0.0323 | 2.435 | 0.135 |
| `storage_45135` | Pris | 0.00719 | 0.0675 | — | — |
| `oven_101917` | Rev | 0.14446 | 0.0561 | 55.486 | 1.487 |
| `scissor_11100` | Rev | 0.09677 | 0.0325 | 0.335 | 4.097 |
| `washer_103776` | Rev | 0.04443 | 0.0607 | 3.161 | 2.114 |
| `blade_103706` | Pris | 0.00840 | 0.1536 | — | — |
| `fridge_10905` | Rev | 0.04280 | 0.0387 | 1.334 | 0.418 |
| `stapler_103111` | Rev | 0.00936 | 0.0079 | 0.232 | 0.215 |
| `USB_100109` | Rev | 0.11417 | 0.0343 | 0.290 | 3.858 |

### Table 2 — Joint-state trajectory errors (phase-3, where applicable)

MAE and maximum absolute error over each sequence, after each method's native evaluation alignment. **Do not mix degrees with millimeters in the same averaged score or plotting axis.** In RigEuler, use phase-3 values for the ground-truth joint-type hypothesis. Conversion: `prismatic_q_*_cm_p3 × 10 → mm`.

| Scene | Unit | FreeArt MAE | RigEuler MAE | FreeArt max abs | RigEuler max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `foldchair_102255` | ° | 0.2310 | 0.1495 | 0.3851 | 0.3490 |
| `laptop_10211` | ° | 0.3707 | 0.0791 | 0.8074 | 0.1819 |
| `storage_45135` | mm | 0.5160 | 0.1820 | 1.3364 | 0.9830 |
| `oven_101917` | ° | 2.0584 | 0.1637 | 5.1226 | 0.6172 |
| `scissor_11100` | ° | 0.0364 | 0.1781 | 0.0991 | 0.3530 |
| `washer_103776` | ° | 20.5944 | 0.1506 | 22.8716 | 0.3813 |
| `blade_103706` | mm | 0.4706 | 0.4960 | 1.6038 | 1.6660 |
| `fridge_10905` | ° | 0.1508 | 0.0267 | 0.3467 | 0.1208 |
| `stapler_103111` | ° | 0.0635 | 0.1729 | 0.1257 | 1.6414 |
| `USB_100109` | ° | 0.0477 | 0.1211 | 0.1214 | 0.2506 |

### Table 3 — Motion ranges and RigEuler run selection

The GT motion ranges agree between the reports up to their serialization/rounding. RigEuler prismatic ranges originate in centimeters and are converted to millimeters. GT range is not an error measure.

| Scene | Unit | GT range | FreeArt predicted | RigEuler predicted | RigEuler primary run |
| --- | ---: | ---: | ---: | ---: | ---: |
| `foldchair_102255` | ° | 107.388 | 107.648 | 107.600 | `v5_canonical` |
| `laptop_10211` | ° | 98.514 | 99.186 | 98.386 | `v5_canonical` |
| `storage_45135` | mm | 532.000 | 532.470 | 532.983 | `v5_canonical` |
| `oven_101917` | ° | 85.500 | 90.623 | 84.883 | `v5_canonical` |
| `scissor_11100` | ° | 44.802 | 44.834 | 45.071 | `v5_canonical` |
| `washer_103776` | ° | 76.266 | 65.999 | 76.647 | `v5_gates_canonical` |
| `blade_103706` | mm | 835.200 | 835.490 | 835.433 | `v5_canonical` |
| `fridge_10905` | ° | 101.500 | 101.248 | 101.470 | `v5_canonical` |
| `stapler_103111` | ° | 109.170 | 109.046 | 108.847 | `v5_canonical` |
| `USB_100109` | ° | 57.296 | 57.174 | 57.045 | `v5_canonical` |

### Table 4 — Gaussian and support counts (descriptive, not accuracy)

Counts come from different training methods and pipeline stages. They may be shown descriptively, but **must not** be used by themselves to claim relative efficiency, speed, reconstruction completeness, or memory consumption.

| Scene | FreeArt Gaussians | RigEuler Gaussians | RigEuler support | RigEuler support fraction |
| --- | ---: | ---: | ---: | ---: |
| `foldchair_102255` | 82,916 | 30,514 | 8,881 | 0.2910 |
| `laptop_10211` | 54,384 | 30,283 | 12,809 | 0.4230 |
| `storage_45135` | 83,165 | 29,521 | 5,664 | 0.1919 |
| `oven_101917` | 88,119 | 29,682 | 4,221 | 0.1422 |
| `scissor_11100` | 54,115 | 29,881 | 10,623 | 0.3555 |
| `washer_103776` | 103,031 | 29,989 | 2,242 | 0.0748 |
| `blade_103706` | 41,671 | 29,911 | 5,288 | 0.1768 |
| `fridge_10905` | 73,079 | 29,677 | 8,456 | 0.2849 |
| `stapler_103111` | 69,393 | 30,280 | 9,963 | 0.3290 |
| `USB_100109` | 80,768 | 30,036 | 14,591 | 0.4858 |

### Table 5 — Aggregate statistics (arithmetic mean or median across scenes)

Each scene has equal weight. Do not create a single "overall joint-state MAE" mixing degrees and millimeters. Only joint-type accuracy is a proportion. Where an `N` is given, it is the number of contributing synthetic scenes per method.

| Metric | Unit | FreeArtGS | RigEuler3DGS | N |
| --- | ---: | ---: | ---: | ---: |
| Correct joint types | count / 10 | 10/10 | 10/10 | 10 |
| Mean axis-direction error | ° | 0.04874 | 0.05101 | 10 |
| Median axis-direction error | ° | 0.03046 | 0.03650 | 10 |
| Mean revolute state MAE | ° | 2.9441 | 0.1302 | 8 |
| Median revolute state MAE | ° | 0.1909 | 0.1501 | 8 |
| Mean revolute MAE excluding washer | ° | 0.4226 | 0.1273 | 7 |
| Mean prismatic state MAE | mm | 0.4933 | 0.3390 | 2 |
| Mean revolute axis-line distance | mm | 7.9591 | 1.7702 | 8 |
| Median revolute axis-line distance | mm | 0.8666 | 1.6625 | 8 |
| Mean Gaussian count | Gaussians | 73,064.1 | 29,977.4 | 10 |

## 3. Rendering metrics — FreeArtGS only (no cross-method comparison)

**Do not draw paired FreeArt/RigEuler rendering charts:** the uploaded RigEuler synthetic table contains **no matching PSNR, SSIM or LPIPS columns**. The available FreeArt metrics are:

| Scene | Logged PSNR (dB) | Final FG PSNR (dB) | Logged SSIM | Logged LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `foldchair_102255` | 33.269 | — | 0.98960 | 0.01232 |
| `laptop_10211` | 26.556 | — | 0.99166 | 0.01121 |
| `storage_45135` | 35.050 | — | 0.98155 | 0.02887 |
| `oven_101917` | 22.900 | 23.242 | 0.90428 | 0.22261 |
| `scissor_11100` | 32.275 | 32.096 | 0.99688 | 0.00530 |
| `washer_103776` | 35.133 | 36.247 | 0.99738 | 0.00848 |
| `blade_103706` | 30.080 | 28.889 | 0.99838 | 0.00380 |
| `fridge_10905` | 33.634 | 33.284 | 0.99642 | 0.00849 |
| `stapler_103111` | 31.106 | 30.675 | 0.99440 | 0.01218 |
| `USB_100109` | 27.950 | 27.764 | 0.99628 | 0.00755 |

- **Logged PSNR/SSIM/LPIPS:** latest all-evaluation-image TensorBoard records, step `29000` in all 10 scenes. The logged SSIM and LPIPS include background. The logged PSNR is called FG PSNR in the accompanying FreeArt report; do not assume it shares identical foreground masking or output quantization with the independent final PNG analysis.
- **Final foreground PSNR:** independent recomputation from rendered PNGs, the corresponding RGB observations, and exact foreground masks. It exists only for **7 scenes** (`oven`, `scissor`, `washer`, `blade`, `fridge`, `stapler`, `USB`), totaling **3360 camera/frame observations**. Do not impute `foldchair`, `laptop`, or `storage`.
- **Train/eval overlap:** each scene has 480 training views and 48 evaluation views, with all 48 also present among the training observations. These scores are **not** held-out novel-view generalization measurements.
- FreeArt aggregate: logged PSNR **30.7955 dB** (10 scenes), SSIM **0.98468**, LPIPS **0.03208**, final foreground PSNR **30.3139 dB** (7 scenes only). Means use equal scene weights, not pooled pixels.
- No numeric geometry Chamfer distance, hidden-surface completeness, dense segmentation accuracy, runtime, VRAM, or training-time comparison is present in the supplied files. Omit those figures.

## 4. Separate RigEuler3DGS v4 baseline (supplementary only)

The following is **not** used in the primary FreeArt vs RigEuler figures. These rows are all `run == 'v4_bootstrap'`; maintain an independent legend/category if plotting a version ablation. `True` represents correct joint-type decision against synthetic GT.

| Scene | State unit | v4 axis (°) | v4 line (mm) | v4 state MAE | v4 state max abs | Type correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `foldchair_102255` | ° | 0.0709 | 3.913 | 0.1015 | 0.2165 | True |
| `laptop_10211` | ° | 0.0375 | 5.363 | 0.2441 | 0.5427 | True |
| `storage_45135` | mm | 0.0762 | — | 0.2320 | 0.9020 | True |
| `oven_101917` | ° | 0.1035 | 2.380 | 0.3316 | 0.6206 | True |
| `scissor_11100` | ° | 0.1551 | 15.435 | 0.8650 | 1.6172 | True |
| `washer_103776` | ° | 0.1564 | 663.421 | 38.1001 | 76.2860 | True |
| `blade_103706` | mm | 0.1519 | — | 0.5710 | 1.7410 | True |
| `fridge_10905` | ° | 0.0243 | 0.675 | 0.0784 | 0.2105 | True |
| `stapler_103111` | ° | 0.0110 | 4.866 | 0.1547 | 1.5189 | True |
| `USB_100109` | ° | 0.1341 | 2.375 | 0.0130 | 0.0936 | True |

Example of why version selection matters: `washer_103776` has v4 revolute MAE **38.1001°**, whereas the main RigEuler `v5_gates_canonical` MAE is **0.1506°**. Never substitute the v4 washer row into the primary 10-scene RigEuler aggregation.

## 5. Real-scene results — RigEuler only, not part of the comparison

The separate real data contain 8 scenes × 3 support-selection arms (`C`, `majority`, `unanimous`). `expected_type` is a **visual description**, not measured GT. The following **categorical predictions** may appear in a supplementary RigEuler-only diagnostic figure; they must not appear in the FreeArt-vs-RigEuler quantitative comparison charts. No numerical real-world axis or state accuracy can be computed from these inputs.

| Real scene | Visual expected | v6 C | v6 majority | v6 unanimous | v5 C | v5 majority | v5 unanimous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `black_box` | revolute | revolute | revolute | revolute | revolute | revolute | revolute |
| `black_laptop` | revolute | revolute | revolute | revolute | revolute | revolute | revolute |
| `light_blue_box` | prismatic | prismatic | prismatic | prismatic | revolute | revolute | revolute |
| `matchbox` | prismatic | revolute | prismatic | revolute | revolute | revolute | revolute |
| `orange_suitecase` | prismatic | revolute | revolute | revolute | prismatic | revolute | revolute |
| `red_fridge` | revolute | revolute | revolute | revolute | revolute | revolute | revolute |
| `robot_dog` | prismatic | revolute | revolute | revolute | revolute | revolute | revolute |
| `wooden_box` | revolute | revolute | revolute | revolute | prismatic | prismatic | prismatic |

The recorded numbers of v6 predictions agreeing with the visual expected type are **C: 5/8, majority: 6/8, unanimous: 5/8**; label as agreement with visual expectation, **not ground-truth accuracy**. Other 103-column real-scene diagnostics are retained unmodified in `RigEuler3DGS_real_results.csv`, should Codex be asked for a separate real-only supplement.

## 6. Precise metric extraction and units

### FreeArtGS synthetic

- Joint type: `joint_type_correct` (Boolean).
- Axis error: `axis_error_deg` (degrees; undirected axis orientation).
- Revolute line position: `axis_line_error_mm` (millimeters; **not** pivot-point distance).
- Joint state: `state_mae`, `state_max_error`, in the per-row `state_unit` (`deg` for revolute, `mm` for prismatic).
- Ranges: `predicted_motion_range`, `gt_motion_range`, in `state_unit`.
- Trajectory samples for figure: `report.json['scenes'][i]['saved_joint_evaluation']['joint_state_samples']`, with `source_frame_index`, `predicted_relative_state`, `gt_relative_state`. JSON values are **radians for revolute and meters for prismatic**. Multiply by `180/pi` or `1000` respectively when plotting.
- FreeArt trajectory errors are relative to the first timestamp; its evaluator removes the initial state offset and aligns axis sign, with no fitted trajectory scale.

### RigEuler3DGS synthetic (primary run)

- Joint type: `type_decision_correct`, with `gt_type` defining which ground-truth hypothesis to score.
- Axis direction: `gt_hyp_axis_err_deg`. Equivalently for the selected GT type, the corresponding `revolute_axis_err_deg_p3` or `prismatic_axis_err_deg_p3` (check equality to reported precision).
- Revolute axis-line error: `revolute_line_dist_cm * 10` (mm); no position metric for prismatic joints.
- Revolute state MAE and max: `revolute_q_mae_deg_p3` and `revolute_q_max_abs_deg_p3` (degrees).
- Prismatic state MAE and max: `prismatic_q_mae_cm_p3 * 10`, `prismatic_q_max_abs_cm_p3 * 10` (millimeters).
- Revolute predicted/GT range: `revolute_range_est_deg` and `revolute_range_gt_deg` (degrees).
- Prismatic predicted/GT range: `prismatic_range_est_cm * 10` and `prismatic_range_gt_cm * 10` (millimeters).
- `gaussians`, `support`, `support_fraction` are counts/fractions, not fit-quality metrics.

**Critical protocol caveat:** the methods' reported state metrics come from their respective evaluation implementations. Their first-frame offsets, coordinate-sign conventions, and other alignment rules must be independently matched before calling the numbers an identically implemented benchmark evaluation. Never introduce a scale fit, resampling, or other re-alignment silently. As supplied, the normalized CSV preserves each report's native numbers. GT range agreement can be used as a sanity check (each scene matches within source rounding).

**Aggregate definitions:** for a metric `e` with contributing scene set `S`, arithmetic mean `sum(e_s)/|S|` and median `median({e_s})`; no sample-count weighting. Report number of contributing scenes. Ratio-based improvements may be computed only if explicitly requested, with denominator and aggregation method displayed; do not show significance tests or error bars because no multiple random seeds/replicates are provided.

## 7. Exact required Matplotlib figures

Generate *numerical*, journal-paper-style plots only. All axes must state units, all panels have figure captions or descriptive titles, and all plotted values must be traceable to a table/CSV column. Proposed figure files below are exact output contracts; export **both PDF and PNG** per figure, with optional SVG. Default plots should favor two-column paper width **7.15 in** (181.6 mm) for multi-panel, and **3.45 in** (87.6 mm) for single-column, with appropriate readable height.

| Output stem | Actual chart type | Data shown | Required detail |
|:---|:---|:---|:---|
| `fig01_axis_direction` | Horizontal paired dot plot | 10 synthetic scenes, FreeArt/RigEuler axis-direction error in degrees | identical ordering to Tables 1–4; zero-based linear axis; two method markers offset vertically to avoid overlap |
| `fig02_revolute_state_mae` | Paired dot plot with **logarithmic error axis** | 8 revolute scenes, FreeArt/RigEuler state MAE in degrees | show full washer 20.594° without clipping; log ticks with explicit units and faint grid lines |
| `fig03_prismatic_state_mae` | Paired dot plot on **linear** axis | `storage`, `blade`; both methods in millimeters | plot two scenes only; reasonable linear x-axis, starting at 0 |
| `fig04_revolute_axis_line` | Paired dots with **logarithmic error axis** | 8 revolute scenes; shortest axis-line distances in mm | include oven 55.486 mm and all other values without clipping; exclude undefined prismatic distance |
| `fig05_motion_range` | Two side-by-side panels; estimated-vs-GT scatter with identity reference | (a) revolute range in degrees, 8 scenes; (b) prismatic range in mm, 2 scenes | draw `y=x` as thin gray dotted line; x=GT, y=predicted; separate methods by marker and annotate the washer datum legibly |
| `fig06_gaussian_counts` | Paired dot plot | per-scene FreeArt/RigEuler Gaussian counts | axis thousands of Gaussians, label as **counts only**, not efficiency or geometry quality; separate if ambiguous in paper |
| `fig07_freeart_trajectories` | Matplotlib multi-panel actual joint-state curves | 10 scenes, **FreeArt GT + FreeArt estimated** samples from JSON | 5×2 or 2×5 small multiples; black solid GT, colored dashed predicted, x=source frame, y=°/mm; NOT a two-method comparison without RigEuler per-frame outputs |
| `fig08_freeart_rendering` | 3 aligned small panels (PSNR, SSIM, LPIPS) | FreeArt logged metrics across 10 scenes | single-method figure only. If showing independent final foreground PSNR, use a separately labelled panel and plot only 7 scenes; never pair with invented RigEuler numbers |
| `fig09_rigeuler_v4_v5_ablation` | 2-panel supplemental comparison | selected v4/v5 RigEuler axis error and revolute state MAE, with separate prismatic panel if needed | separate version labels; use log scale for revolute MAE and show v4 washer 38.1001° |

Do not combine revolute and prismatic MAE in a single unnormalized ordinate. Do not create an aggregate "accuracy score" that mixes units. A tabular figure may be rendered as LaTeX or in the manuscript, not as an illustrated infographic.

**Optional fair dimensionless summary** (only if requested): per scene `relative_state_MAE_percent = 100 * MAE / GT_motion_range`, with numerator and denominator in the *same unit*, plotted as a clearly marked derived metric. Do not treat it as equal to absolute motion error or use it to replace figures `fig02` and `fig03`. GT ranges are supplied in Table 3.

**Not currently possible with supplied data:** a 10-scene **two-method** per-frame trajectory overlay. FreeArt provides all 60 samples in `report.json`; the RigEuler synthetic CSV has only aggregate metrics and ranges. To build that figure, first locate the actual per-frame RigEuler phase-3 trajectories in the RigEuler output directories and validate their alignment with GT. If unavailable, print `SKIPPED: RigEuler per-frame trajectories not supplied` and generate only `fig07_freeart_trajectories`. Do **not** interpolate a straight line between endpoints or synthesize time series from MAE/range values.

## 8. Exact plotting design rules — paper figures, not cartoons

- Use **Matplotlib** (with NumPy and pandas for data). Noninteractive `Agg` backend. **No image generation, icons, drawings, emoji, textured backgrounds, gradients, giant cards, arbitrary highlighting, or infographic layouts**.
- Background pure white, black or dark gray axes, very subtle thin grid lines on quantitative axes only, no extra decorative elements. No `seaborn` or dashboard UI.
- Suggested rcParams:

```python
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
})
```

- Use two consistent, clearly separable series: **FreeArtGS** dark blue `#285780`, circle marker, and **RigEuler3DGS** dark red `#A33B3B`, square marker. These choices are for plot differentiation only; do not encode subjective performance through colors. All plots remain legible in grayscale via marker shape/line style.
- Scene order fixed as: `foldchair_102255`, `laptop_10211`, `storage_45135`, `oven_101917`, `scissor_11100`, `washer_103776`, `blade_103706`, `fridge_10905`, `stapler_103111`, `USB_100109`. Use shorter display names on ticks only if the original scene IDs remain available in caption or data appendix.
- Use joined markers with thin optional connecting lines when they improve paired comparison. Keep legends small, inside unused whitespace or above panel in one row. No annotations expressing judgments (`best`, `failure`, `wins`, `outperform`); **numerical axis labels and concise metric names only**.
- Avoid misleading encodings: no truncated linear axes for bar charts, no clipping extreme observations without indication, log scales explicitly tick-labelled and only on strictly positive error data, no random jitter unless deterministic and documented, no interpolated trajectories.
- Target line width 1.0–1.3 pt, marker size 3.5–4.5 pt. Keep labels at >=7 pt in a normal-width 2-column rendering. For long scene IDs, use horizontal plots or wrapped labels rather than unreadably tiny text.
- Export vector PDF (`bbox_inches='tight'`) and 300–600 DPI PNG; optional SVG. Recommended default is 400 DPI. Include plot-level captions/metadata in a companion `FIGURES.md`, not large text boxes inside the images.

## 9. Codex implementation contract

### Files to write

Create the following in the repository (choose a clearly isolated folder, e.g. `analysis/benchmark_comparison/`):

```text
analysis/benchmark_comparison/
    generate_paper_figures.py
    test_generate_paper_figures.py
    README.md
    input/                         # provided source files + normalized joined CSV
    figures/
        fig01_axis_direction.{pdf,png}
        fig02_revolute_state_mae.{pdf,png}
        fig03_prismatic_state_mae.{pdf,png}
        fig04_revolute_axis_line.{pdf,png}
        fig05_motion_range.{pdf,png}
        fig06_gaussian_counts.{pdf,png}
        fig07_freeart_trajectories.{pdf,png}
        fig08_freeart_rendering.{pdf,png}
        fig09_rigeuler_v4_v5_ablation.{pdf,png}
        FIGURES.md
        metrics_recomputed.json
```

### Script interface

```bash
python analysis/benchmark_comparison/generate_paper_figures.py \
  --freeart-summary analysis/benchmark_comparison/input/FreeArtGS_summary.csv \
  --freeart-report analysis/benchmark_comparison/input/FreeArtGS_report.json \
  --freeart-psnr analysis/benchmark_comparison/input/FreeArtGS_per_observation_psnr.csv \
  --rigeuler-synthetic analysis/benchmark_comparison/input/RigEuler3DGS_synthetic_results.csv \
  --outdir analysis/benchmark_comparison/figures \
  --formats pdf,png \
  --dpi 400
```

Alternative: `--comparison-csv` may consume the supplied normalized `freeart_rigeuler_comparison.csv`, but the program must still validate units/protocol against the raw inputs when available. Use an explicit run selector (`--rigeuler-run primary|v4_bootstrap`) if implementing additional variants, with `primary` as default and the washer exception hard-coded only through validated run metadata.

### Validation tests (mandatory)

1. Input schema: expected column presence, unique method/scene/version, complete 10-scene overlap, matching GT types; fail with actionable errors otherwise.
2. All FreeArt scenes `cameras == 8`, `timestamps == 60`, joint type correct, 10 matched types; 8 revolute and 2 prismatic.
3. Main RigEuler scene selection returns 9 `v5_canonical` + 1 `v5_gates_canonical` on washer; exclude every `v4_bootstrap` from primary aggregate.
4. Conversion oracles: `blade_103706` RigEuler prismatic state MAE **0.0496 cm = 0.496 mm**; `storage_45135` **0.0182 cm = 0.182 mm**; `oven_101917` RigEuler revolute axis-line **0.1487 cm = 1.487 mm**.
5. Mean axis errors: FreeArt **0.0487402673°**, RigEuler **0.0510100000°** (approximate last digits of RigEuler due to CSV rounding).
6. Revolute MAE means: FreeArt **2.9441095411°**, RigEuler **0.1302125°**. Prismatic MAE means: FreeArt **0.4933094810 mm**, RigEuler **0.3390 mm**.
7. Revolute axis-line means: FreeArt **7.959081881 mm**, RigEuler **1.77025 mm**.
8. Washer test: FreeArt **20.5944087472°**; RigEuler v5 **0.1506°**; RigEuler v4 **38.1001°**. Log plot must contain the full numerical range.
9. Cross-report GT motion ranges match to rounding (e.g. washer **76.266°**, blade **835.2 mm**).
10. FreeArt trajectory extraction: every one of 10 scenes has exactly 60 samples; convert radians to degrees and meters to millimeters; verify computed MAE from samples against reported `state_mae` within the precision of the provided fields.
11. Missing final-foreground PSNR values remain missing for foldchair, laptop and storage; 7 valid scenes, 3360 observations. No RigEuler rendering metrics may appear.
12. Generate exactly the required numerical figures, record plotted source fields and units in `FIGURES.md`, and fail if no images are produced. Save comparison means, medians and counts as JSON for audit.
13. A test must explicitly reject attempts to calculate a mixed-unit joint-state mean (revolute degrees plus prismatic millimeters).

### Completion criteria

The implementation is complete when the script can run in a fresh noninteractive Python environment; all tests pass; the paper figures are readable at 2-column print scale; every plotted point is reproduced from the supplied numerical data; `metrics_recomputed.json` matches Table 5 at the given precision; and no required graph relies on fabricated per-frame curves or fabricated missing metrics.

**Do not modify the research methods, training pipelines, GT definitions, or existing results folders. Implement only this isolated plotting/evaluation package.**

## 10. Short Codex task statement (copy/paste)

> Implement the Matplotlib comparison package defined in `FreeArtGS_vs_RigEuler3DGS_Codex_Matplotlib_Spec.md`. Use the numerical source files in the plotting-input ZIP, not the previously generated infographic images. Match all 10 scenes by identifiers, preserve original units, use RigEuler primary v5 runs (washer `v5_gates_canonical`), keep v4 separate, and generate the 9 specified paper figures in vector PDF and high-DPI PNG. No cartoon styling, icons, comparisons without source metrics, qualitative "winner" claims, or invented RigEuler per-frame trajectories. Add complete numerical assertions/tests, `metrics_recomputed.json`, and figure captions in `FIGURES.md`. Do not change model/training code.
