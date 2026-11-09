# RigEuler3DGS status, 2026-11-09

The standard pipeline is now the code that produced the **v5 synthetic** and the **v6 real** results.
On synthetic data with ground truth, the joint axes and per-frame joint states are accurate on all 10
objects. On the 8 real captures there is no joint GT. The joint type agrees with what the images show
in 5 or 6 of 8 scenes, depending on the arm. The Phase-1 static drift is about half of what v5 had,
but it is not gone.

Figures and tables: [`results/`](results/README.md), built by `results/build_inputs.py` and
`results/generate_paper_figures.py`.

---

## 1. Standard pipeline

| | synthetic (8 cameras, GT) | real (2 cameras, no GT) |
|---|---|---|
| script | [`run_standard_synthetic.sh`](run_standard_synthetic.sh) | [`run_standard_real.sh`](run_standard_real.sh) |
| sensor model | ideal | calibrated from the static window |
| endpoint | covariance-surface bootstrap, 30000 Gaussians, 1000 steps | the same, then Phase-0 repair: arms `C` (none), `majority`, `unanimous` |
| direction | one per scene, from `metadata/direction/summary.json` | the same; motion window from `tools/estimate_motion_interval.py` |
| Phase 1 | masked-RGB expected-quad, canonical partition | the same, plus declared mismatch sds rgb 0.04 / depth 0.003 |
| v6 options | off | `--velocity-tau 0.5`, `--phase2-support-refine free-space` (1 round) |
| Phase 3 | joint fixed; both P and R hypotheses | the same |
| results | `ours_v5_canonical_synthetic` + `ours_v5_phase1_gates/washer_103776_canonical` | `ours_v6_two_cams` |

**Provenance.**
* **Synthetic.** All 10 scenes ran on commit `8273cd0` (clean tree) with one identical config. The
  washer sits in another folder only because it ran first, as a gate.
* **Real.** The v6 runs used commit `455e362` plus the v5 two-camera patch and the v6 patch.
  * The v6 code reused the v5 endpoints. Calibration, bootstrap and Phase-0 repair are byte-identical
    in v5 and v6.
  * v6 changes only Phases 1-3, and only when `--velocity-tau` / `--phase2-support-refine` are set.

On this branch both v6 options are off by default, so the defaults still run the synthetic recipe.

**Joint type.** No pipeline stage chooses the joint type. Joint identification fits both P and R.
The evaluation (`results/rigeuler_tables.py`) then decides with a geometric rule:
* it computes the revolute axis radius / part scale (R/L);
* **R/L > 10 → prismatic**, because a prismatic motion forces the revolute fit to put its axis far away.

On synthetic data, R/L is 160–353 for the prismatic objects and 0.8–4.0 for the revolute ones. The
pose-residual ratio E_P/E_R is not used: it calls blade and storage revolute.

---

## 2. Synthetic objects (10 scenes, GT read only after Phase 3)

Each result below is RigEuler's estimate under the GT joint type. The R/L rule picks that type on 10
of 10 objects (20/20 counting the v4 runs). State errors are each method's own evaluation:
* revolute in degrees, prismatic in millimetres;
* FreeArtGS measured from frame 0, RigEuler from its canonical frame.

| scene | type | GT range | R/L | axis err (°) RigEuler / FreeArtGS | axis-line dist (mm) RigEuler / FreeArtGS | state MAE RigEuler / FreeArtGS |
|---|---|---|---|---|---|---|
| foldchair_102255 | R | 107.4° | 0.83 | 0.027 / 0.002 | 1.84 / 0.40 | 0.150° / 0.231° |
| laptop_10211 | R | 98.5° | 1.05 | 0.032 / 0.018 | 0.14 / 2.44 | 0.079° / 0.371° |
| storage_45135 | P | 532 mm | 160 | 0.068 / 0.007 | – | 0.182 mm / 0.516 mm |
| oven_101917 | R | 85.5° | 0.99 | 0.056 / 0.145 | 1.49 / 55.49 | 0.164° / 2.058° |
| scissor_11100 | R | 44.8° | 2.83 | 0.033 / 0.097 | 4.10 / 0.33 | 0.178° / 0.036° |
| washer_103776 | R | 76.3° | 1.32 | 0.061 / 0.044 | 2.11 / 3.16 | 0.151° / 20.594° |
| blade_103706 | P | 835 mm | 318 | 0.154 / 0.008 | – | 0.496 mm / 0.471 mm |
| fridge_10905 | R | 101.5° | 1.42 | 0.039 / 0.043 | 0.42 / 1.33 | 0.027° / 0.151° |
| stapler_103111 | R | 109.2° | 3.19 | 0.008 / 0.009 | 0.22 / 0.23 | 0.173° / 0.064° |
| USB_100109 | R | 57.3° | 2.00 | 0.034 / 0.114 | 3.86 / 0.29 | 0.121° / 0.048° |

| aggregate (equal weight per scene) | FreeArtGS | RigEuler3DGS |
|---|---|---|
| joint type correct | 10/10 | 10/10 |
| axis direction error, mean / median | 0.049° / 0.030° | 0.051° / 0.037° |
| revolute state MAE, mean / median (8 scenes) | 2.944° / 0.191° | 0.130° / 0.150° |
| revolute state MAE without washer, mean (7) | 0.423° | 0.127° |
| prismatic state MAE, mean (2) | 0.493 mm | 0.339 mm |
| revolute axis-line distance, mean / median (8) | 7.96 mm / 0.87 mm | 1.77 mm / 1.66 mm |
| Gaussians, mean | 73064 | 29977 |

* **Washer:** the v4 bootstrap run had a 38.1° state MAE. v5 has 0.151° (fig09).
* **Rendering:** the inputs have no RigEuler PSNR/SSIM/LPIPS, so rendering is FreeArtGS only (fig08).

---

## 3. Real captures (8 scenes × 3 Phase-0 arms, no GT)

The expected type comes from reading the RGB and the depth-change maps; it is not ground truth.
* Agreement of the R/L decision with that expectation:
  * v6: arm C **5/8**, majority **6/8**, unanimous **5/8**.
  * v5: arm C 4/8, majority 3/8, unanimous 3/8.
* The free-space support refinement was accepted in 19 of 24 runs.

How to read the table:
* Triples are arms C / majority / unanimous; R = revolute, P = prismatic.
* The Phase-3 value is in ° for an R decision and cm for a P decision.
* Drift is measured on the same "confirmed static" Gaussians in v5 and v6: seen ≥ 20 times at their
  canonical place and never refuted. Median over arms; 21 of 24 runs have this set.
* The last column is the share of v6 drift that is near the moving part / rarely seen / seen away
  from the part, median over arms.

| scene | observed motion | expected | v5 type | v6 type | v6 R/L | v6 Phase-3 value | refine accepted | end drift v5 → v6 (cm) | last-15-frame drift v5 → v6 (cm) | drift share near / unseen / away (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| black_box | lid opens slightly | R | R/R/R | R/R/R | 0.2/2.1/2.1 | 12.3/36.1/2.9 | n/y/y | 7.0 → 3.4 | 1.43 → 0.46 | 37 / 22 / 41 |
| black_laptop | lid closes ~100°, glossy, depth mostly invalid | R | R/R/R | R/R/R | 1.2/8.3/2.9 | 11.0/2.6/6.9 | y/y/y | 12.5 → 6.3 | 2.89 → 0.66 | 49 / 35 / 14 |
| light_blue_box | drawer closes ~21 cm; case pushed back a few cm | P | R/R/R | P/P/P | 59.8/31.9/37.6 | 21.3/21.5/21.2 | y/y/y | 2.4 → 0.9 | 0.63 → 0.09 | 49 / 23 / 26 |
| matchbox | drawer pushed in, small | P | R/R/R | R/P/R | 3.6/18.5/0.2 | 5.2/6.6/4.4 | n/y/n | 7.6 → 3.2 | 1.98 → 0.35 | 24 / 39 / 37 |
| orange_suitecase | telescopic handle pushed in ~22 cm | P | P/R/R | R/R/R | 1.7/0.7/1.8 | 36.1/0.1/39.6 | y/n/y | 0.6 → 0.3 | 0.21 → 0.07 | 9 / 65 / 26 |
| red_fridge | door closes ~40-44°; arm retracts after | R | R/R/R | R/R/R | 0.7/0.6/1.2 | 36.3/34.1/44.1 | y/y/y | 8.4 → 2.9 | 2.23 → 0.34 | 21 / 69 / 10 |
| robot_dog | battery box pushed in, small | P | R/R/R | R/R/R | 1.1/3.4/3.3 | 0.2/31.7/21.8 | n/y/y | 2.9 → 0.7 | 1.01 → 0.18 | 19 / 29 / 53 |
| wooden_box | lid opens | R | P/P/P | R/R/R | 2.3/3.4/2.4 | 19.2/12.4/20.0 | y/y/y | 5.4 → 3.2 | 1.17 → 0.48 | 22 / 50 / 25 |

**Drift.** Across the 21 runs with the set, the median end drift fell from 5.7 cm to 2.7 cm: v6 is
0.25–0.64 of v5, median 0.44. The median last-15-frame drift fell from 1.43 cm to 0.34 cm. The
velocity damping removed the runaway Gaussians. What remains has three causes:
1. **Unseen Gaussians.** Gaussians that are rarely observed carry 20–80% of the drift; on the fridge
   it is 55–80%, mostly the interior hidden by the closing door. Their median drift is 6.4 cm.
2. **Near the moving part.** Gaussians within 2 cm of the moving support are dragged along: median
   3.9 cm, up to 16 cm. Gaussians seen away from the part drift 1.4 cm.
3. **Tangent slip.** On low-texture surfaces the drift runs along the surface: the median tangent
   component exceeds the normal one in all 23 runs that have it.

There is no global rigid shift.

---

## 4. Open problems

1. **The refinement adds Gaussians the motion never explains.**
   * Problem: the free-space test labels "moving" partly by elimination. Examples: the laptop
     keyboard (specular depth) and the light_blue_box case front (a second motion).
   * Candidate rule, not implemented: keep an added Gaussian only if Cm ≥ 6, Cm ≥ Vm and L > 4,
     with a fallback that does not keep a refuted coherence support.
2. **Small prismatic motions on real data.**
   * matchbox is correct in 1 of 3 arms; orange_suitecase and robot_dog are wrong in all 3.
   * orange_suitecase majority collapsed to a whole-object support of 23920 Gaussians (0.1 cm).
3. **The type decision is evaluation-time only.** The R/L rule lives in `results/rigeuler_tables.py`,
   not in a pipeline stage.
4. **Not yet run on synthetic:**
   * the v6 options;
   * this branch with the options off. It should match `8273cd0` up to the CUDA nondeterminism
     floor (~1e-2 between identical runs), but that has not been checked.
   * Command: `OUT=.../ours_standard_synthetic bash run_standard_synthetic.sh`.
5. **Three real runs have no confirmed-static set:** matchbox unanimous, orange_suitecase majority,
   robot_dog C. Their refinement was refused and no free-space labels were saved, so their drift is
   N/A.

---

## 5. Reproduce

```bash
PY=~/miniconda3/envs/rigeuler3dgs/bin/python
bash run_standard_synthetic.sh 2>&1 | tee ~/standard_synthetic.log    # 10 scenes, ~5.3 h on the 8 GB card (v5)
bash run_standard_real.sh      2>&1 | tee ~/standard_real.log         # 8 scenes x 3 arms
DRY=1 bash run_standard_synthetic.sh                                   # print the plan only

$PY results/build_inputs.py                 # result folders -> results/input/
$PY results/generate_paper_figures.py       # -> results/figures/ (fig01-fig09, figS1-figS5)
$PY results/test_generate_paper_figures.py  # 14 checks
```
