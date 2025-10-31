#!/usr/bin/env bash
# Playbook: end-to-end visualizations for save_every_1 runs
#
# Usage:
#   ./all_1.sh [--base DIR] [--out DIR] [--phases PHASES] [--force]
#
# Examples:
#   ./all_1.sh --phases suite,ablation   # run selected phases only
#   FORCE=1 ./all_1.sh                   # ignore caches and re-run
#
# Notes:
# - This machine uses compact checkpoint names (no timestamps), e.g. k3_c16_gru8
# - The sister playbook all_1wd1e-3.sh targets weight-decay=1e-3 checkpoints

set -euo pipefail

BASE="checkpoints/save_every_1"
OUT="visualizations/save_every_1"
SEQ_DATA="${SEQ_DATA:-data/connect4_sequential_10k_games.pt}"
PHASES="suite,ablation,embeddings,observability,fixed,cka,metrics,trajectory"
FORCE=0
ABLA_EPOCH_STEP="${ABLA_EPOCH_STEP:-2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      BASE="$2"; shift 2;;
    --out|--output-dir)
      OUT="$2"; shift 2;;
    --phases)
      PHASES="$2"; shift 2;;
    --force)
      FORCE=1; shift;;
    *)
      echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

mkdir -p "$OUT"

has_phase() { [[ ",$PHASES," == *",$1,"* ]]; }

run_suite() {
  if [[ ! -f "configs/visualization_suite_1.json" ]]; then
    echo "Skipping suite (missing configs/visualization_suite_1.json)"; return 0
  fi
  local report="$OUT/report.md"
  if [[ $FORCE -eq 0 && -f "$report" ]]; then
    echo "Suite already has report at $report (use --force to rebuild)"; return 0
  fi
  ./wt.sh visualize --config configs/visualization_suite_1.json --report "$report"
}

# Build ablation groups automatically from $BASE/k*_c*_gru*/
build_ablation_groups() {
  mapfile -t RUNS < <(ls -d "$BASE"/k*_c*_gru*/ 2>/dev/null | sed 's:/$::' | sort)
  if (( ${#RUNS[@]} < 2 )); then
    return 1
  fi
  # Index by GRU and by C
  declare -gA BY_GRU=()
  declare -gA BY_C=()
  for d in "${RUNS[@]}"; do
    name="$(basename "$d")"
    k="$(echo "$name" | sed -E 's/^k([0-9]+)_c([0-9]+)_gru([0-9]+).*$/\1/')"
    c="$(echo "$name" | sed -E 's/^k([0-9]+)_c([0-9]+)_gru([0-9]+).*$/\2/')"
    g="$(echo "$name" | sed -E 's/^k([0-9]+)_c([0-9]+)_gru([0-9]+).*$/\3/')"
    # Append paths into delimited strings to preserve order by channels/hidden
    BY_GRU["$g"]+="$d\n"
    BY_C["$c"]+="$d\n"
  done

  # Emit sorted arrays for each key
  declare -gA GRU_GROUPS=()
  declare -gA C_GROUPS=()
  for g in "${!BY_GRU[@]}"; do
    # sort by channels numerically
    grp=$(echo -e "${BY_GRU[$g]}" | grep -v '^$' | sort -t '_' -k2,2 -k3,3)
    GRU_GROUPS["$g"]="$grp"
  done
  for c in "${!BY_C[@]}"; do
    # sort by gru numerically
    grp=$(echo -e "${BY_C[$c]}" | grep -v '^$' | sort -t '_' -k3,3)
    C_GROUPS["$c"]="$grp"
  done
  return 0
}

run_ablation() {
  build_ablation_groups || { echo "Skipping ablation (need >=2 runs under $BASE)"; return 0; }

  # CNN ablations: vary channels at fixed GRU
  for g in $(printf '%s\n' "${!GRU_GROUPS[@]}" | sort -n); do
    mapfile -t dirs < <(echo -e "${GRU_GROUPS[$g]}")
    (( ${#dirs[@]} >= 2 )) || continue
    local out_dir="$OUT/ablation_cnn_g${g}"
    if [[ $FORCE -eq 0 && -f "$out_dir/ablation_cnn_trajectories.png" ]]; then
      echo "Skip CNN ablation (gru=${g}) — output exists ($out_dir)"; continue
    fi
    python -m src.visualize_trajectories --viz-type ablation-cnn --ablation-dirs "${dirs[@]}" \
      --ablation-animate --ablation-center normalize --output-dir "$out_dir" \
      --epoch-step "$ABLA_EPOCH_STEP" --phate-n-pca 24 --phate-knn 6 --phate-t 10
  done

  # GRU ablations: vary GRU at fixed channels
  for c in $(printf '%s\n' "${!C_GROUPS[@]}" | sort -n); do
    mapfile -t dirs < <(echo -e "${C_GROUPS[$c]}")
    (( ${#dirs[@]} >= 2 )) || continue
    local out_dir="$OUT/ablation_gru_c${c}"
    if [[ $FORCE -eq 0 && -f "$out_dir/ablation_gru_trajectories.png" ]]; then
      echo "Skip GRU ablation (c=${c}) — output exists ($out_dir)"; continue
    fi
    python -m src.visualize_trajectories --viz-type ablation-gru --ablation-dirs "${dirs[@]}" \
      --ablation-animate --ablation-center normalize --output-dir "$out_dir" \
      --epoch-step "$ABLA_EPOCH_STEP" --phate-n-pca 24 --phate-knn 6 --phate-t 10
  done
}

# Embeddings and metrics/trajectory steps retained from earlier playbook
run_embeddings() {
  ./wt.sh embeddings --checkpoint-dirs "$BASE"/*/ --component cnn \
    --methods pca tsne umap phate --output-dir "$OUT/embeddings" --compare --annotate --animate

  ./wt.sh embeddings --checkpoint-dirs "$BASE"/*/ --component gru \
    --methods pca tsne umap phate --output-dir "$OUT/embeddings" --compare --annotate --animate

  ./wt.sh embeddings --checkpoint-dirs "$BASE"/*/ --component all \
    --methods pca tsne umap phate --output-dir "$OUT/embeddings" --compare --annotate --animate
}

run_metrics() {
  local metrics_dir="diagnostics/trajectory_analysis"
  if [[ $FORCE -eq 0 && -n $(ls -1 "$metrics_dir"/*_metrics.csv 2>/dev/null | head -n1) ]]; then
    echo "Metrics exist under $metrics_dir (use --force to recompute)"; return 0
  fi
  if [[ -f "$SEQ_DATA" ]]; then
    ./wt.sh metrics --checkpoint-dirs "$BASE"/*/ --component all \
      --board-source dataset --board-dataset "$SEQ_DATA" \
      --output-dir "$metrics_dir"
  else
    ./wt.sh metrics --checkpoint-dirs "$BASE"/*/ --component all \
      --output-dir "$metrics_dir"
  fi
}

run_trajectory() {
  local out_ms="$OUT/metric_space"
  mkdir -p "$out_ms"
  local html="$out_ms/trajectory_embedding_3d.html"
  if [[ $FORCE -eq 0 && -f "$out_ms/trajectory_embedding_all.png" ]]; then
    echo "Trajectory embedding outputs exist in $out_ms (use --force to rebuild)"; return 0
  fi
  ./wt.sh trajectory-embedding \
    --metrics-dir "diagnostics/trajectory_analysis" \
    --checkpoint-dir "$BASE" \
    --output-dir "$out_ms" \
    --method phate --n-neighbors 10 \
    --dims 3 --animate-3d --anim-frames 180 --anim-seconds 12 \
    --plotly-html "$html"
}

has_phase suite && run_suite
has_phase ablation && run_ablation
has_phase embeddings && run_embeddings
run_observability() {
  if [[ ! -f "$SEQ_DATA" ]]; then
    echo "Skipping observability (missing dataset at $SEQ_DATA)"; return 0
  fi
  local diag_dir="diagnostics/gru_observability"
  local out_dir="$OUT/gru_observability"
  mkdir -p "$diag_dir" "$out_dir"
  if [[ $FORCE -eq 0 && -f "$out_dir/gate_mean_trajectories.png" ]]; then
    echo "Observability visuals exist in $out_dir (use --force to rebuild)"; return 0
  fi
  ./wt.sh observability extract --checkpoint-dir "$BASE" --dataset "$SEQ_DATA" \
    --output-dir "$diag_dir" --max-games 128 --sample-hidden 1500 || true
  ./wt.sh observability analyze --analysis-dir "$diag_dir" \
    --output-dir "$out_dir" --max-hidden-samples 2000 || true
}

run_fixed() {
  if [[ ! -f "$SEQ_DATA" ]]; then
    echo "Skipping fixed points (missing dataset at $SEQ_DATA)"; return 0
  fi
  local diag_dir="diagnostics/gru_fixed_points"
  local out_dir="$OUT/gru_fixed_points"
  mkdir -p "$diag_dir" "$out_dir"
  if [[ $FORCE -eq 0 && -n $(ls -1 "$diag_dir"/*/fixed_points_summary.csv 2>/dev/null | head -n1) ]]; then
    echo "Fixed points already computed (use --force to recompute)"
  else
    ./wt.sh observability fixed --checkpoint-dir "$BASE" --dataset "$SEQ_DATA" \
      --output-dir "$diag_dir" --max-contexts 8 --restarts 4 --max-iter 200 || true
  fi
  ./wt.sh observability evolve --fixed-dir "$diag_dir" --output-dir "$out_dir" || true
}

run_cka() {
  local out_dir="$OUT/cka"
  mkdir -p "$out_dir"
  ./wt.sh cka --checkpoint-dir "$BASE" --representation gru --output-dir "$out_dir" \
    --epoch-step 10 --animate ${FORCE:+--force} || true
  ./wt.sh cka --checkpoint-dir "$BASE" --representation cnn --output-dir "$out_dir" \
    --epoch-step 10 --animate ${FORCE:+--force} || true
}

has_phase metrics && run_metrics
has_phase trajectory && run_trajectory
has_phase observability && run_observability
has_phase fixed && run_fixed
has_phase cka && run_cka

# 3) Weight embeddings (PCA/TSNE/UMAP/PHATE) + overlays + GIFs
./wt.sh embeddings --checkpoint-dirs "$BASE"/*/ --component cnn \
  --methods pca tsne umap phate --output-dir "$OUT/embeddings" --compare --annotate --animate

./wt.sh embeddings --checkpoint-dirs "$BASE"/*/ --component gru \
  --methods pca tsne umap phate --output-dir "$OUT/embeddings" --compare --annotate --animate

./wt.sh embeddings --checkpoint-dirs "$BASE"/*/ --component all \
  --methods pca tsne umap phate --output-dir "$OUT/embeddings" --compare --annotate --animate

# # 4) GRU observability (extract, analyze + MI)
# if [ -f "$SEQ_DATA" ]; then
#   ./wt.sh observability extract --checkpoint-dir "$BASE" --dataset "$SEQ_DATA" \
#     --output-dir "diagnostics/gru_observability"


#   ./wt.sh observability analyze --analysis-dir "diagnostics/gru_observability" \
#     --output-dir "$OUT/gru_observability"
# else
#   echo "Skipping observability (missing $SEQ_DATA)"
# fi

# # 5) Fixed points + evolution
# if [ -f "$SEQ_DATA" ]; then
#   ./wt.sh observability fixed --checkpoint-dir "$BASE" --dataset "$SEQ_DATA" \
#     --output-dir "diagnostics/gru_fixed_points"

#   ./wt.sh observability evolve --fixed-dir "diagnostics/gru_fixed_points" \
#     --output-dir "$OUT/gru_fixed_points"
# fi

# # 6) CKA similarity across runs (GRU and CNN), with animations
# ./wt.sh cka --checkpoint-dir "$BASE" --representation gru --output-dir "$OUT/cka" \
#   --epoch-step 10 --animate
# ./wt.sh cka --checkpoint-dir "$BASE" --representation cnn --output-dir "$OUT/cka" \
#   --epoch-step 10 --animate

# 7) Metrics CSVs feeding metric-space trajectory embeddings
#    Produces diagnostics/trajectory_analysis/<run>_metrics.csv
if [ -f "$SEQ_DATA" ]; then
  ./wt.sh metrics --checkpoint-dirs "$BASE"/*/ --component all \
    --board-source dataset --board-dataset "$SEQ_DATA" \
    --output-dir "diagnostics/trajectory_analysis"
else
  ./wt.sh metrics --checkpoint-dirs "$BASE"/*/ --component all \
    --output-dir "diagnostics/trajectory_analysis"
fi

# 8) Metric-space embedding (PHATE) with 3D rotation GIF and interactive HTML
./wt.sh trajectory-embedding \
  --metrics-dir "diagnostics/trajectory_analysis" \
  --checkpoint-dir "$BASE" \
  --output-dir "$OUT/metric_space" \
  --method phate \
  --n-neighbors 10 \
  --dims 3 \
  --animate-3d --anim-frames 180 --anim-seconds 12 \
  --plotly-html "$OUT/metric_space/trajectory_embedding_3d.html"