"""Read-only forward/backward observation-overlap diagnostics.

CODEX_PHASE1_FORWARD_BACKWARD_PIXEL_OVERLAP_COVERAGE_DIAGNOSTICS.md.

Nothing in this package trains, merges, prunes or optimises a Gaussian
population. It renders each existing Phase-1 source SEPARATELY with the
production IEKF renderer, extracts each Gaussian's exact per-pixel
contribution, and measures coverage, observation support and cross-source
same-surface evidence against the dataset's own RGB-D frames.
"""
