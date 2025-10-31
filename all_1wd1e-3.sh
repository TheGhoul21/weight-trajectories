#!/usr/bin/env bash
# Playbook: WD=1e-3 variant of save_every_1 runs

set -euo pipefail

BASE="checkpoints/save_every_1_wd1e-3"
OUT="visualizations/save_every_1_wd1e-3"
SEQ_DATA="${SEQ_DATA:-data/connect4_sequential_10k_games.pt}"
PHASES="suite,ablation,embeddings,observability,fixed,cka,metrics,trajectory"
FORCE=0
ABLA_EPOCH_STEP="${ABLA_EPOCH_STEP:-2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2;;
    --out|--output-dir) OUT="$2"; shift 2;;
    --phases) PHASES="$2"; shift 2;;
    --force) FORCE=1; shift;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

mkdir -p "$OUT"

has_phase() { [[ ",$PHASES," == *",$1,"* ]]; }

run_suite() {
  if [[ ! -f "configs/visualization_suite_1wd1e-3.json" ]]; then
    echo "Skipping suite (missing configs/visualization_suite_1wd1e-3.json)"; return 0
  fi
  local report="$OUT/report.md"
  if [[ $FORCE -eq 0 && -f "$report" ]]; then
    echo "Suite already has report at $report (use --force to rebuild)"; return 0
  fi
  ./wt.sh visualize --config configs/visualization_suite_1wd1e-3.json --report "$report"
}

run_ablation() {
  mapfile -t RUNS < <(ls -d "$BASE"/k*_c*_gru*/ 2>/dev/null | sed 's:/$::' | sort)
  (( ${#RUNS[@]} >= 2 )) || { echo "Skipping ablation (need >=2 runs under $BASE)"; return 0; }
  # Group by GRU and C
  declare -A BY_GRU=() BY_C=()
  for d in "${RUNS[@]}"; do
    name="$(basename "$d")"
    c="$(echo "$name" | sed -E 's/^k([0-9]+)_c([0-9]+)_gru([0-9]+).*$/\2/')"
    g="$(echo "$name" | sed -E 's/^k([0-9]+)_c([0-9]+)_gru([0-9]+).*$/\3/')"
    BY_GRU["$g"]+="$d\n"; BY_C["$c"]+="$d\n"
  done
  for g in $(printf '%s\n' "${!BY_GRU[@]}" | sort -n); do
    mapfile -t dirs < <(echo -e "${BY_GRU[$g]}" | grep -v '^$' | sort -t '_' -k2,2)
    (( ${#dirs[@]} >= 2 )) || continue
    local out_dir="$OUT/ablation_cnn_g${g}"
    if [[ $FORCE -eq 0 && -f "$out_dir/ablation_cnn_trajectories.png" ]]; then continue; fi
    python -m src.visualize_trajectories --viz-type ablation-cnn --ablation-dirs "${dirs[@]}" \
      --ablation-animate --ablation-center normalize --output-dir "$out_dir" \
      --epoch-step "$ABLA_EPOCH_STEP" --phate-n-pca 24 --phate-knn 6 --phate-t 10
  done
  for c in $(printf '%s\n' "${!BY_C[@]}" | sort -n); do
    mapfile -t dirs < <(echo -e "${BY_C[$c]}" | grep -v '^$' | sort -t '_' -k3,3)
    (( ${#dirs[@]} >= 2 )) || continue
    local out_dir="$OUT/ablation_gru_c${c}"
    if [[ $FORCE -eq 0 && -f "$out_dir/ablation_gru_trajectories.png" ]]; then continue; fi
    python -m src.visualize_trajectories --viz-type ablation-gru --ablation-dirs "${dirs[@]}" \
      --ablation-animate --ablation-center normalize --output-dir "$out_dir" \
      --epoch-step "$ABLA_EPOCH_STEP" --phate-n-pca 24 --phate-knn 6 --phate-t 10
  done
}

run_embeddings() {
  ./wt.sh embeddings --checkpoint-dirs "$BASE"/*/ --component cnn \
    --methods pca tsne umap phate --output-dir "$OUT/embeddings" --compare --annotate --animate
  ./wt.sh embeddings --checkpoint-dirs "$BASE"/*/ --component gru \
    --methods pca tsne umap phate --output-dir "$OUT/embeddings" --compare --annotate --animate
  ./wt.sh embeddings --checkpoint-dirs "$BASE"/*/ --component all \
    --methods pca tsne umap phate --output-dir "$OUT/embeddings" --compare --annotate --animate
}

run_observability() {
  if [[ ! -f "$SEQ_DATA" ]]; then echo "Skipping observability (missing $SEQ_DATA)"; return 0; fi
  local diag_dir="diagnostics/gru_observability_wd1e-3"
  local out_dir="$OUT/gru_observability"
  mkdir -p "$diag_dir" "$out_dir"
  if [[ $FORCE -eq 0 && -f "$out_dir/gate_mean_trajectories.png" ]]; then return 0; fi
  ./wt.sh observability extract --checkpoint-dir "$BASE" --dataset "$SEQ_DATA" \
    --output-dir "$diag_dir" --max-games 128 --sample-hidden 1500 || true
  ./wt.sh observability analyze --analysis-dir "$diag_dir" \
    --output-dir "$out_dir" --max-hidden-samples 2000 || true
}

run_fixed() {
  if [[ ! -f "$SEQ_DATA" ]]; then echo "Skipping fixed (missing $SEQ_DATA)"; return 0; fi
  local diag_dir="diagnostics/gru_fixed_points_wd1e-3"
  local out_dir="$OUT/gru_fixed_points"
  mkdir -p "$diag_dir" "$out_dir"
  if [[ $FORCE -eq 0 && -n $(ls -1 "$diag_dir"/*/fixed_points_summary.csv 2>/dev/null | head -n1) ]]; then
    echo "Fixed points already computed";
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

run_metrics() {
  local metrics_dir="diagnostics/trajectory_analysis_wd1e-3"
  if [[ $FORCE -eq 0 && -n $(ls -1 "$metrics_dir"/*_metrics.csv 2>/dev/null | head -n1) ]]; then return 0; fi
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
  local metrics_dir="diagnostics/trajectory_analysis_wd1e-3"
  local out_ms="$OUT/metric_space"
  mkdir -p "$out_ms"
  local html="$out_ms/trajectory_embedding_3d.html"
  if [[ $FORCE -eq 0 && -f "$out_ms/trajectory_embedding_all.png" ]]; then return 0; fi
  ./wt.sh trajectory-embedding \
    --metrics-dir "$metrics_dir" \
    --checkpoint-dir "$BASE" \
    --output-dir "$out_ms" \
    --method phate --n-neighbors 10 \
    --dims 3 --animate-3d --anim-frames 180 --anim-seconds 12 \
    --plotly-html "$html"
}

has_phase suite && run_suite
has_phase ablation && run_ablation
has_phase embeddings && run_embeddings
has_phase observability && run_observability
has_phase fixed && run_fixed
has_phase cka && run_cka
has_phase metrics && run_metrics
has_phase trajectory && run_trajectory