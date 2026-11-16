# Phase 4 exactness report: full image vs exact ROI

All numbers come from tests executed on 2026-11-11 (RTX 4070 Laptop, float32 renders, float64 losses).

## 1. Claims and the evidence for them

| claim | test | result |
|---|---|---|
| The evaluator's fast render (`PosedPair`: both poses precomputed, per-row selection) is **bitwise** the production `render_articulated` | `test_phase4_exact_roi_parity.py` (every view and case) | `torch.equal` on colour, alpha and Z for every view |
| Every pixel outside the conservative footprint is **bitwise** identical between the two full hypothesis renders. Footprint: kernel radius, projected mean, one-tile pad, both poses | same; plus 200 random flips in a 128×96 probe | 0 changed pixels outside, in every case |
| SSIM on a crop is **bitwise** the full-image SSIM at the same pixels | same | `torch.equal` |
| ROI per-channel delta = sum of the full-image per-pixel difference maps | same | max error **1.7e-18** (bound 1e-15 + 1e-12·\|Δ\|) |
| ROI delta = difference of full-image view losses | same | max relative error **8.4e-17** (bound 1e-12 relative to \|L_alt\| + \|L_base\|) |
| OMIT delta (ROI) = full-image OMIT delta | same | max error **1.7e-18** |
| A skipped view (culled in both poses / identity transform) has a full-image delta of exactly 0 | same | 0.0 |
| Phase-4 SSIM (float64, fixed-order) = `utils.loss_utils.ssim` (float32 conv) | `test_ssim_matches_reference_implementation` | 0.87070864 vs 0.87070906 (4e-7) |
| Real data: one red_fridge view pair, 3 groups (near the core, random, mid-distance), 2 frames × 2 cameras | `test_real_red_fridge_view` | ROI vs maps 8.7e-19; relative vs totals 6.4e-17; OMIT 8.7e-19 |
| 50-group real reference: exact ROI vs an independent full-image scorer, red_fridge, 12 frames × 2 cameras | `tools/benchmark_phase4a.py` → `reports/phase4_2026-11-11/benchmark_red_fridge_12frames_50groups_parity.json` | max abs diff **8.7e-19** (max \|Δ\| 6.4e-3; relative 1.3e-16); **50/50 decisions identical**; ROI 6.8× faster |
| Monotone exact descent: the realised change of every 4A batch equals the predicted (additive) change | `test_phase4a_exact_labels.py`, CLI subset run | additivity error ≤ 3e-16 on every batch |

## 2. Cases covered

- revolute forward and prismatic backward procedural objects;
- the occluded door back face behind the front;
- late-visible geometry (the drawer floor);
- a large projected Gaussian (8 cm);
- a near-threshold opacity (1.2/255);
- Gaussians at the image border and behind the camera (frustum-culled);
- three Gaussians at identical depth in one tile (sort ties);
- a thin two-sided layer (1 mm apart);
- a group crossing the depth order of static surfaces;
- background pixels;
- a block of missing observed depth inside the mask in every view;
- a joint move of three groups;
- single-group flips and OMIT.

Each case is checked in every view (24 views per procedural case, 4 for the real one).

## 3. Why it is exact

The kernel (`forward.cu`) bins Gaussian i only into tiles inside `rect_i`:
- `rect_i` comes from `ceil(3·sqrt(λ_max))` and the projected mean, clamped to the grid;
- radius 0 when frustum-culled;
- alpha below 1/255 is skipped inside the tile.

Each tile's contributor list is a stable radix sort on (tile, depth) keys. Removing i, or moving it to tiles outside a given tile, changes neither the keys nor the relative order of that tile's other entries. Compositing and early termination are per pixel within the tile. Hence pixels outside `rect_i` (in both poses) are bitwise invariant.

- **Rect computation.** We take the radius from the kernel itself (a render's `radii`) and recompute the mean in torch with the kernel's formula. Float rounding can move an edge by one tile, so each rect is padded by one tile on every side, a superset.
- **SSIM.** An output pixel reads an 11×11 neighbourhood. Changes inside the union `R` affect SSIM outputs only within `R ⊕ 5 px`. Inputs beyond the true border are zeros, which matches the reference conv padding.
- **Denominators.** All are fixed per view (3HW, HW, |V_observed|), so the view-loss difference is the plain sum over the dilated ROI.

## 4. Unsupported or not claimed

- **ROI gradients.** 4C fits candidate parameters with full-frame rendering and full-image (role-weighted) maps; ROI gradients are not used anywhere, so no gradient-parity claim is made. The opt-in RGB-D kernel's gradients were verified in Gate 1 (2026-11-10, `test_rgbd_cuda_forward_backward.py`, re-run today: PASS).
- **Antialiasing** stays refused: the upstream covariance backward is wrong with it.
- **Numerical floor.** "Exact" means equal to the full-image computation up to float64 summation order. The renders themselves are deterministic float32 (forward pass, no atomics); bitwise equality of the two full renders outside the footprint was asserted rather than assumed.
