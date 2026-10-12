# Transparency carving: legacy vs carving

- legacy run: `/home/samuelemara/Joint3DGS/RigEuler3DGS/output/carving_ab/legacy`
- carving run: `/home/samuelemara/Joint3DGS/RigEuler3DGS/output/carving_ab/carving`
- iterations traced: 30000
- initial Gaussians: 1380
- training cameras: 8 (masks: 8)
- git commit: 75b85f751b510764f568e6ecc2c6864983d2dfa1

## Is this a controlled A/B?

- initial-state SHA256 **matches**: `a31073dee762149fd39b164e85febad122511fded7e307089781ecf7f1b5561e`
- camera traces **identical** across 30000 iterations
- configuration otherwise identical

Bitwise GPU reproducibility is not claimed: the rasteriser uses non-deterministic atomics. What is guaranteed is identical initial tensors, an identical view schedule, and no code difference outside the treatment branch. Splitting decisions that diverge *after* the intervention are an outcome, not a confound.

## At initialization (iteration 0)

| metric | legacy | carving |
| --- | --- | --- |
| E_background_global | 9.1157e-03 | 9.1157e-03 |
| foreground_opacity_global | 0.8086 | 0.8086 |
| N_gaussians | 1380.0000 | 1380.0000 |
| anisotropy_median | 1.0000 | 1.0000 |
| anisotropy_p99 | 1.0000 | 1.0000 |
| anisotropy_max | 1.0000 | 1.0000 |
| max_scale | 0.0935 | 0.0935 |
| max_scale_over_bbox_diag | 0.0457 | 0.0457 |
| nonfinite_parameters | 0.0000 | 0.0000 |

These must agree: same model, same measurement, independent of branch.

## At the final checkpoint (iteration 30000)

| metric | legacy | carving |
| --- | --- | --- |
| E_background_global | 5.6163e-04 | 1.4898e-04 |
| foreground_opacity_global | 0.9917 | 0.9971 |
| N_gaussians | 11397.0000 | 9740.0000 |
| anisotropy_median | 538.4774 | 1554.7527 |
| anisotropy_p99 | 178786.3494 | 338651.8675 |
| anisotropy_max | 1948780.8750 | 2811208.2500 |
| max_scale | 0.2615 | 0.2760 |
| max_scale_over_bbox_diag | 0.1251 | 0.1326 |
| nonfinite_parameters | 0.0000 | 0.0000 |

Background leakage ratio legacy/carving: **3.8x**

## Verdict

**Background fixed, anisotropy not fixed**

E_background falls 5.616e-04 -> 1.490e-04 and is lower at 25 of 30 checkpoints, but anisotropy does not improve (p99 178786.34937499717 -> 338651.86750000116, max 1948780.875 -> 2811208.25). Transparency supervision addresses background occupancy; primitive shape needs a separate experiment.

The training RGB losses of the two branches are not compared: they score against different composites and the numbers are not commensurable. E_background and the shape statistics are the comparable outcomes.

This measures an association under one controlled intervention: random-background compositing. It does not implicate the CUDA rasteriser and says nothing about Splatfacto's other defaults.

Artifacts: `comparison.csv`, `comparison.png`
