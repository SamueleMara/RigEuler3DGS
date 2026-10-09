#!/usr/bin/env bash
# Stage A (section 13 order: USB, Storage, Blade) then Stage B (section 37).
# Both stages, all three scenes, full frame count.
set -u
PY=/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python
cd /home/samuelemara/Joint3DGS/RigEuler3DGS

declare -A MODEL=(
  [USB_100109]=output/merged_USB_100109
  [storage_45135]=output/dec_A_storage_45135
  [blade_103706]=output/dec_A_blade_103706
)
FRAMES=${FRAMES:-30}
CAMS=${CAMS:-4}
NG=${NG:-6000}

for S in USB_100109 storage_45135 blade_103706; do
  echo "######## STAGE A  $S"
  $PY tools/phase2_rigid_audit.py \
      --scene "$S" --model "${MODEL[$S]}" \
      --frames "$FRAMES" --cameras "$CAMS" --max-gaussians "$NG" \
      --screw-iters 8 --bootstrap-iters 6 \
      --render-sequences --render-stride 5 \
      --out "output/stageA_$S"
done

for S in USB_100109 storage_45135 blade_103706; do
  echo "######## STAGE B  $S"
  $PY tools/run_iekf_scene.py \
      --model "${MODEL[$S]}" \
      --observation-source dataset_rgbd \
      --sequence-start 0 --sequence-length "$FRAMES" \
      --cameras "$CAMS" --max-gaussians "$NG" \
      --supernodes --window-discovery --num-moving-parts 1 \
      --phase2-rigid-screw-iekf --phase2-screw-iters 8 \
      --phase2-screw-cg-iters 40 --phase2-block-frames 1 \
      --phase2-parallel-pr --phase2-pr-folds 3 \
      --discovery-outer 1 \
      --out "output/stageB_$S"
done

echo "######## SUMMARY"
$PY - <<'PYEOF'
import json, pathlib
for s in ("USB_100109", "storage_45135", "blade_103706"):
    p = pathlib.Path(f"output/stageB_{s}/phase2_model_comparison/comparison.json")
    if not p.exists():
        print(f"{s:>14}: no comparison.json"); continue
    d = json.load(p.open())
    print(f"\n{s}  interpretable={d['comparison_is_interpretable']}  "
          f"degenerate_folds={d['revolute_degenerate_folds']}")
    for m in ("prismatic", "revolute"):
        for r in d["per_fold"].get(m, []):
            print(f"   {m:>9} fold {r['fold']}: train {r['train']['total']:.4e} "
                  f"(rgb {r['train']['rgb']:.3e} d {r['train']['depth']:.3e})  "
                  f"heldout {r['heldout']['total']:.4e} "
                  f"(rgb {r['heldout']['rgb']:.3e} d {r['heldout']['depth']:.3e})")
    print(f"   delta_heldout per fold: "
          f"{[round(x, 2) for x in d['delta_heldout_per_fold']]}")
PYEOF
