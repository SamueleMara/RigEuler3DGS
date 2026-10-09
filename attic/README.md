# attic — code the production pipeline does not reach

Nothing here is deleted. Every file kept its path, so a move back is exact:

```bash
# restore one file
cd /home/samuelemara/Joint3DGS/RigEuler3DGS
f=utils/closed_loop_assignment.py; mkdir -p "$(dirname $f)"; mv "attic/dead/$f" "$f"

# restore everything
cd /home/samuelemara/Joint3DGS/RigEuler3DGS
(cd attic/dead && find . -type f) | sed 's|^\./||' | while read f; do
  mkdir -p "$(dirname $f)"; mv "attic/dead/$f" "$f"; done
(cd attic/superseded && find . -type f) | sed 's|^\./||' | while read f; do
  mkdir -p "$(dirname $f)"; mv "attic/superseded/$f" "$f"; done

# or, once you are satisfied
rm -rf attic/
```

`MANIFEST.txt` lists every path, relative to the repo root.

## How this was decided

An import graph was walked from the production entry points, following **both**
import statements and subprocess references (`train.py` launches
`tools/extract_dinov3_features.py` by path, and a pure import graph calls that
dead). Importing `pkg.sub` also marks `pkg/__init__.py` reached, which is what
keeps `rigid_parts/` — `train.py` imports its submodules.

Roots treated as production:

```
train.py  render.py  metrics.py  full_eval.py  convert.py
tools/train_all_static_endpoints.py
tools/run_motion_coherence_support.py
tools/run_rigid_se3_pipeline.py
tools/rereduce_rigid_trajectory.py
tools/summarize_pipeline.py
tools/run_all_datasets.py
tools/run_complete_joint_debug.py      <- the legacy ablation branch, kept on purpose
```

## attic/dead — nothing reached these at all

51 Python files plus 3 superseded shell runners. No entry point, no test, no
subprocess reference. Mostly one-off probes (`run_*_probe.py`), superseded
debug pages, and ablation drivers for experiments that were replaced.

The three scripts are `run_full_geometric.sh`, `run_phase2_support_ab.sh` and
`run_stage_ab.sh` — the Stage-A/B and support-A/B runners that the rigid SE(3)
pipeline replaced.

## attic/superseded — only superseded tests reached these

73 modules and the 27 tests that exercise them, moved together so the test
suite stays consistent. These are the research branches that did not become
the pipeline: closed-loop motion, bidirectional flow matching, endpoint
transport, progressive rigid discovery, relational graphs, the physics
module, diffusion-based matching.

## Deliberately NOT moved

| kept | why |
|---|---|
| `SIBR_viewers/`, `convert.py`, `full_eval.py` | upstream 3DGS, not ours to prune |
| `utils/make_depth_scale.py` | upstream 3DGS depth-regularisation preprocessing, run by hand |
| `estimator/analytic_reference.py` | the autograd oracle the analytic derivatives are validated against; test-only by design |
| `tools/run_complete_joint_debug.py` and its tree | the legacy direct-axis ablation, kept per the rigid-SE(3) spec section 43 |
| `tools/extract_dinov3_features.py`, `extract_sam3_*.py`, `run_dynamic_prior_comparison.py` | `train.py` launches these as subprocesses |

## Verified after the move

- all 10 production entry points still start (`--help`)
- the 7 production suites still pass: 221 checks
- a static scan finds 0 files importing anything that moved
- the full three-stage pipeline still runs end to end on a real scene
