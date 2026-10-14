# Fused forward/backward Phase 1 — commands, costs, policies

From `CODEX_IMPLEMENT_FUSED_PHASE1_REUSE_EXISTING_FB_WASHER_THEN_OVERNIGHT_ALL.md`.

## What runs

The historical forward and backward Phase 1 (the same
`tools/run_motion_coherence_support.py`, run once per direction) are reused
**read only**. The new stage is a different inference over their output:

```
static_bidi/{initial,final} PLYs + bidirectional/{forward,backward}/phase1_centers
  -> two disjoint candidate populations, one chronological time axis
  -> ONE full-window IEKS, both populations rendered TOGETHER by the original
     CUDA compositor, one RGB-D factor per (original frame, camera),
     initial pose fixed at frame 0, final pose fixed at the last frame
```

Initialisation is `DERIVED_12D_INITIAL_GUESS_FROM_EXISTING_TRACKS`: the real
centres verbatim, R = each source PLY's orientation, v = central differences,
omega = 0. That is an optimiser initial guess, not something the legacy IEKF
exported (it estimated R, v, omega internally but saved centres only). All of
mu, R, v, omega are optimised except each source's anchor pose.

## Commands

Washer first:

```bash
cd /home/samuelemara/Joint3DGS/RigEuler3DGS && bash scripts/run_fused_phase1.sh --scene washer_103776 --reuse-existing-phase1 --profile washer-validation --tag fused_phase1_allcams_v1 --resume
```

Then every eligible scene, serially, detached:

```bash
cd /home/samuelemara/Joint3DGS/RigEuler3DGS
nohup bash scripts/run_fused_phase1_overnight.sh --all-existing-scenes --reuse-existing-phase1 --profile overnight --tag fused_phase1_allcams_v1 --resume > "$HOME/fused_phase1_overnight_v1.log" 2>&1 < /dev/null &
echo "overnight PID $!"
tail -f "$HOME/fused_phase1_overnight_v1.log"
```

Clean stop (finishes the current outer iteration, keeps the best state):
`touch "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/ours/unified_phase1_overnight/fused_phase1_allcams_v1/STOP"`
(for the single-scene run: `touch <experiment>/STOP`).

## Measured costs (washer, all 45,920 Gaussians, 60 frames, 2 cameras)

| quantity | measured |
|---|---|
| peak GPU, solve on host (`--solve-on-cpu`) | 751 MiB |
| peak GPU, solve on device | OOM at 3.4 GiB while two other jobs held 4.2 GiB |
| host RSS | 14.6 GB |
| one PCG iteration | ~7.1 s (98 s at 5 iterations, 169 s at 15) |
| fixed cost per run (load, evaluate, build, line search) | ~60 s |
| first accepted step, 5 PCG | E 2.19e8 -> 3.15e7 |

Both profiles now use ALL 8 cameras and ALL 60 frames (480 factors on
washer). The 2-camera washer run measured ~8.5 min per outer step at 60 PCG;
with 4x the factors, expect roughly 30 min per step (an estimate scaled from
that run, not a measurement). One camera's linearisation of the full union
holds ~18 MiB, so 8 cameras add ~140 MiB per frame on the GPU.
washer-validation: up to 10 outer x 60 PCG, no time budget. overnight: the
same, with a 180-minute budget per scene. The
two share one experiment per scene (same posterior; PCG count and time are
not part of the fingerprint). Larger scenes cost proportionally more per
iteration; foldchair (135,694 Gaussians) is the largest.

## Resume and collision policy

An experiment directory is identified by a fingerprint of its sources
(PLY and track SHA-256), frames, cameras, population and solver code. Same
fingerprint: finished stages are reused, a CONVERGED optimiser is skipped,
otherwise one new attempt starts from the best accepted state so far.
Different fingerprint: refused (`TAG_CONFIG_MISMATCH`), never overwritten.
Without `--resume`, an existing experiment is refused (`TAG_EXISTS`).

## Optional: reproduce a legacy direction

Not a prerequisite, never called by the two commands above:

```bash
bash scripts/run_legacy_free_phase1_export.sh --scene washer_103776 --direction forward  --tag legacy_export_v1 --preflight
bash scripts/run_legacy_free_phase1_export.sh --scene washer_103776 --direction forward  --tag legacy_export_v1 --execute --save-full-state
bash scripts/run_legacy_free_phase1_export.sh --scene washer_103776 --direction backward --tag legacy_export_v1 --execute --save-full-state
```

The resulting `full_state/` can seed the fused run with
`--initial-full-state` / `--final-full-state` (a different fingerprint, so a
new `--tag`).

## Theory compliance

Per scene, `docs/PHASE1_FUSED_THEORY_COMPLIANCE.{md,json}` inside the
experiment maps F01-F21 (the equations stated in section 2 of the directive)
and the twelve bridge gates B01-B12 to code, tests and that scene's real
evidence. The revised PDF and the 24-item list it refers to are not on this
machine, so those 24 rows are not mapped and the audit says so.
