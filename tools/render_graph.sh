#!/usr/bin/env bash
# Nested part-embedding outer loop + differential Killing-field inner loop.
#
#   tools/nested_killing.sh <scene> [options] [-- extra train.py args]
#
#   tools/nested_killing.sh USB_100109
#   tools/nested_killing.sh USB_100109 --parts 2
#   tools/nested_killing.sh blade_103706 --parts 2 --out blade_k2
#   tools/nested_killing.sh USB_100109 -- --lambda_inner_strain 20.0
#
# Options (all optional):
#   --parts N          moving parts K                      (default 1)
#   --out NAME         output dir under output/            (default <scene>_nested_k<K>)
#   --outer-iters N    outer embedding iterations          (default 6)
#   --outer-steps N    outer label epochs (full video sweeps)  (default 5)
#   --inner-steps N    inner Killing steps per frame       (default 15)
#   --temp N           initial softmax temperature         (default 1.0)
#   --temp-soft N      tau soft floor while motion invalid (default 0.7)
#   --temp-hard N      tau hard floor, never below 0.4     (default 0.4)
#   --ssim N           lambda_inner_ssim (render primary)  (default 0.2)
#   --c-min N          observed-velocity confidence gate   (default 0.6)
#   --rho-obs N        max observed-velocity share         (default 0.2)
#   --trust N          lambda_outer_trust_region           (default 1.0)
#   --anchor-warmup N  outer iters of hard anchor pinning  (default 1)
#   --emb-motion N     lambda_outer_embedding_motion       (default 0.0)
#   --endpoints DIR    endpoint run dir  (default output/<scene>_static_endpoints_14k)
#   --dry-run          print the command and exit
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/samuelemara/miniconda3/envs/rigeuler3dgs/bin/python}"

die() { echo "error: $*" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: $(basename "$0") <scene> [options] [-- extra train.py args]
available scenes: $(ls "$REPO/input" 2>/dev/null | grep -v '\.zip$' | tr '\n' ' ')"

SCENE="$1"; shift

PARTS=1
OUT=""
OUTER_ITERS=6
OUTER_STEPS=5      # §9 outer_label_epochs: each is a full video sweep
INNER_STEPS=15
TEMP=1.0
TEMP_SOFT=0.7
TEMP_HARD=0.4
SSIM=0.2
C_MIN=0.6
RHO_OBS=0.2
TRUST=1.0
ANCHOR_WARMUP=1
EMB_MOTION=0.0
ENDPOINTS=""
DRY=0
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --parts)       PARTS="$2"; shift 2 ;;
    --out)         OUT="$2"; shift 2 ;;
    --outer-iters) OUTER_ITERS="$2"; shift 2 ;;
    --outer-steps) OUTER_STEPS="$2"; shift 2 ;;
    --inner-steps) INNER_STEPS="$2"; shift 2 ;;
    --temp)        TEMP="$2"; shift 2 ;;
    --temp-soft)   TEMP_SOFT="$2"; shift 2 ;;
    --temp-hard)   TEMP_HARD="$2"; shift 2 ;;
    --ssim)        SSIM="$2"; shift 2 ;;
    --c-min)       C_MIN="$2"; shift 2 ;;
    --rho-obs)     RHO_OBS="$2"; shift 2 ;;
    --trust)       TRUST="$2"; shift 2 ;;
    --anchor-warmup) ANCHOR_WARMUP="$2"; shift 2 ;;
    --emb-motion)  EMB_MOTION="$2"; shift 2 ;;
    --endpoints)   ENDPOINTS="$2"; shift 2 ;;
    --dry-run)     DRY=1; shift ;;
    --)            shift; EXTRA=("$@"); break ;;
    -h|--help)     sed -n '2,33p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)             die "unknown option: $1  (pass through to train.py after --)" ;;
  esac
done

[ -n "$OUT" ] || OUT="${SCENE}_render_graph"
[ -n "$ENDPOINTS" ] || ENDPOINTS="output/${SCENE}_static_endpoints_14k"

cd "$REPO"

SOURCE="input/${SCENE}"
START_PLY="${ENDPOINTS}/shared/frame0_static/point_cloud/iteration_14000/point_cloud.ply"
FINAL_DIR="${ENDPOINTS}/shared/frame_t_static"
MODEL="output/${OUT}"

# Fail early and clearly rather than 20 minutes into a run.
[ -d "$SOURCE" ]     || die "missing scene input: $SOURCE"
[ -f "$START_PLY" ]  || die "missing frame-0 endpoint ply: $START_PLY
run the static endpoint stage for '$SCENE' first, or pass --endpoints DIR"
[ -d "$FINAL_DIR" ]  || die "missing frame-T endpoint model: $FINAL_DIR"

ARGS=(
  -s "$SOURCE"
  -m "$MODEL"
  --iterations 14500
  --static_train_iters 14000
  --start_ply "$START_PLY"
  --start_ply_iteration 14000
  --final_3dgs_path "$FINAL_DIR"

  # Required coupling: the retained dynamic path is phase-1 only, needs the
  # semantic feature field, and the Killing representation needs correspondence.
  --dynamic_motion
  --phase1_only
  --use_semantic_feature_field
  --phase1_motion_representation rigid_killing_eulerian

  --enable_nested_killing_part_discovery
  --num_moving_parts "$PARTS"
  --part_outer_iters "$OUTER_ITERS"
  --part_outer_embedding_steps "$OUTER_STEPS"
  --killing_inner_steps_per_frame "$INNER_STEPS"
  --static_anchor_use_neighborhood_check

  # FIRM v3 §11: rendering is the PRIMARY observation.
  --lambda_inner_rgb 1.0
  --lambda_inner_ssim "$SSIM"
  --lambda_inner_depth 0.1
  --lambda_inner_strain 10.0
  --lambda_inner_omega_consistency 1.0
  --lambda_inner_b_consistency 1.0
  --lambda_inner_field_separation 1.0
  --lambda_inner_rigid_speed 1.0

  # §12/§13: tracks are a gated, capped, weak auxiliary.
  --lambda_inner_observed_velocity 1.0
  --observed_velocity_c_min "$C_MIN"
  --observed_velocity_gamma 2.0
  --observed_velocity_rho_max "$RHO_OBS"

  # §23: tau stays soft until the motion is actually correct.
  --part_temperature "$TEMP"
  --part_temperature_soft_floor "$TEMP_SOFT"
  --part_temperature_hard_floor "$TEMP_HARD"

  # Render-centric outer E-step: video is the PRIMARY label supervision (§14),
  # 3D endpoint/moments are regularizers capped at rho=0.3 of the video
  # gradient (§16), and the raw view-variance penalty is off (§11).
  # §25 first USB experiment: deterministic render-matching outer loop.
  --outer_label_mode render_graph
  --render_seed_confidence 0.75
  --min_support_views 2
  --killing_neighbor_k 16
  --graph_iterations 50
  --tau_render_preference 1.0
  --lambda_pixel_overlap 0.5
  --lambda_pixel_xyz 1.0
  --lambda_pixel_depth 0.5
  --enable_pixel_matching
  --raster_top_contributors 4
  --lambda_pixel_candidate 1.0
  --lambda_pixel_margin 0.1
  --lambda_outer_video 1.0
  --lambda_outer_label_graph 0.05
  --lambda_outer_kinetic 0.05
  --rho_3d_gradient_cap 0.30
  --lambda_multiview_view_var 0.0
  # §43 starting weights: 3D terms are deliberately small, not the old ones.
  --lambda_outer_candidate_nn 0.05
  --lambda_outer_final_nn 0.02
  --lambda_outer_part_mass 0.02
  --lambda_outer_global_com 0.01
  --lambda_outer_global_inertia 0.005
  --lambda_outer_part_com 0.02
  --lambda_outer_part_inertia 0.005
  --lambda_outer_trust_region 0.02

  # §22 finite anchor warmup, §27 support term, §30 trust region, §32 guard.
  --anchor_hard_freeze_outer_iters "$ANCHOR_WARMUP"
  --lambda_outer_part_mass 2.0
  --lambda_outer_trust_region "$TRUST"
  --outer_geometry_guard_backtracks 4
  --lambda_outer_embedding_motion "$EMB_MOTION"
)
[ ${#EXTRA[@]} -gt 0 ] && ARGS+=("${EXTRA[@]}")

if [ "$DRY" -eq 1 ]; then
  printf 'PYTHONPATH=. %s train.py' "$PYTHON"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$MODEL"
echo "scene      $SCENE"
echo "parts      K=$PARTS  (+1 static)"
echo "output     $MODEL"
echo "debug      $MODEL/dynamic_debug/nested_part_embedding_killing/"
echo

PYTHONPATH=. "$PYTHON" train.py "${ARGS[@]}" 2>&1 | tee "$MODEL/run.log"

echo
echo "done. debug artifacts:"
ls -1 "$MODEL/dynamic_debug/nested_part_embedding_killing/" 2>/dev/null || echo "  (none written)"
