#!/usr/bin/env bash
# Unified entrypoint for weight-trajectories utilities

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${ROOT_DIR}/scripts"

detect_python() {
  # Honor explicit override first
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi

  local candidates=(
    "${ROOT_DIR}/.venv/bin/python3"
    "${ROOT_DIR}/.venv/bin/python"
    "$(command -v python3 || true)"
    "$(command -v python || true)"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      echo "${candidate}"
      return
    fi
  done
}

PYTHON_EXEC="$(detect_python)"

if [[ -z "${PYTHON_EXEC}" ]]; then
  echo "Error: unable to find a Python interpreter. Set PYTHON_BIN or create .venv." >&2
  exit 1
fi

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

run_with_python() {
  run_cmd "${PYTHON_EXEC}" "$@"
}

run_wizard() {
  local script="$1"
  shift
  PYTHON_BIN="${PYTHON_EXEC}" bash "${script}" "$@"
}

show_help() {
  cat <<'EOF'
Weight Trajectories CLI (v1.0)
Usage: ./wt.sh <command> [options]

Basics
  help                       Show this message
  python-path                Print the Python interpreter used by wt.sh

Data & Training
  dataset flat|sequential    Generate datasets (AlphaZero-based)
  train [args]               Train a single ResNet+GRU model
  train-all [args]           Train 3× GRU sizes across channels
  model                      Architecture smoke-test

Analysis & Plots
  metrics [args]             Compute checkpoint metrics (CSV)
  visualize [args]           Unified visualization suite
  trajectory-embedding       Metric-space trajectory plots (UMAP/t-SNE/PHATE)
  embeddings [args]          Weight/representation embeddings
  factorial [args]           Factorial heatmaps (3×3 sweep)

GRU Observability
  observability extract      Collect gates, timescales, hidden samples
  observability analyze      Gates, PHATE, probes; runs MI afterwards
  observability mi           Run mutual information analysis only
  observability fixed        Fixed-point finding
  observability evolve       Attractor evolution visualization

Cross-Model Similarity
  cka [args]                 CKA similarity (non-interactive)
  cka wizard                 Interactive CKA runner

Export & Reports
  report [args]              Generate markdown analysis report
  onnx [args]                Export model to ONNX

Tips
  - Run './wt.sh <command> --help' for script-specific options
  - Visualizer supports T-PHATE via '--t-phate [--t-phate-alpha A]'
  - Temporal viz: '--viz-type temporal --temporal-mode training|game'
  - Set PYTHON_BIN to override the interpreter used by all commands
  - See docs: manual/commands for details and examples
EOF
}

cmd_train_all() {
  local data_path="${WT_DATASET:-data/connect4_10k_games.pt}"
  local epochs="${WT_EPOCHS:-100}"
  local save_every="${WT_SAVE_EVERY:-10}"
  local kernel_size="${WT_KERNEL_SIZE:-3}"
  local seed="${WT_SEED:-37}"
  local extra_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --data)
        data_path="$2"
        shift 2
        ;;
      --epochs)
        epochs="$2"
        shift 2
        ;;
      --save-every)
        save_every="$2"
        shift 2
        ;;
      --kernel-size)
        kernel_size="$2"
        shift 2
        ;;
      --seed)
        seed="$2"
        shift 2
        ;;
      --)
        shift
        extra_args+=("$@")
        break
        ;;
      *)
        extra_args+=("$1")
        shift
        ;;
    esac
  done

  if [[ ! -f "${data_path}" ]]; then
    echo "Warning: dataset '${data_path}' not found. Training may fail." >&2
  fi

  local channels=(16 64 256)
  local gru_sizes=(8 32 128)

  for gru in "${gru_sizes[@]}"; do
    local batch_size=64
    if (( gru >= 128 )); then
      batch_size=32
    fi
    echo
    echo "=== Training configuration: GRU=${gru}, dataset=${data_path} ==="
    run_with_python src/train.py \
      --data "${data_path}" \
      --cnn-channels "${channels[@]}" \
      --gru-hidden "${gru}" \
      --kernel-size "${kernel_size}" \
      --epochs "${epochs}" \
      --save-every "${save_every}" \
      --batch-size "${batch_size}" \
      --seed "${seed}" \
      "${extra_args[@]}"
  done
}

cmd_dataset() {
  if [[ $# -lt 1 ]]; then
    echo "dataset command requires a subcommand: flat|sequential" >&2
    exit 1
  fi

  local sub="$1"
  shift || true

  case "${sub}" in
    flat)
      run_with_python "${SCRIPTS_DIR}/generate_connect4_dataset.py" "$@"
      ;;
    sequential|seq)
      run_with_python "${SCRIPTS_DIR}/generate_sequential_dataset.py" "$@"
      ;;
    *)
      echo "Unknown dataset subcommand '${sub}'. Use flat or sequential." >&2
      exit 1
      ;;
  esac
}

cmd_metrics() {
  run_with_python "${SCRIPTS_DIR}/compute_checkpoint_metrics.py" "$@"
}

cmd_cka() {
  if [[ "${1:-}" == "wizard" ]]; then
    shift
    run_wizard "${SCRIPTS_DIR}/run_cka_wizard.sh" "$@"
  else
    run_with_python "${SCRIPTS_DIR}/compute_cka_similarity.py" "$@"
  fi
}

cmd_embeddings() {
  if [[ "${1:-}" == "wizard" ]]; then
    shift
    run_wizard "${SCRIPTS_DIR}/run_embedding_wizard.sh" "$@"
  else
    run_with_python "${SCRIPTS_DIR}/analyze_weight_embeddings.py" "$@"
  fi
}

cmd_trajectory_embedding() {
  run_with_python "${SCRIPTS_DIR}/visualize_trajectory_embedding.py" "$@"
}

cmd_observability() {
  local sub="${1:-extract}"
  shift || true

  case "${sub}" in
    extract)
      run_with_python "${SCRIPTS_DIR}/extract_gru_dynamics.py" "$@"
      ;;
    analyze|summarize)
      run_with_python "${SCRIPTS_DIR}/analyze_gru_observability_results.py" "$@"
      run_with_python "${SCRIPTS_DIR}/compute_hidden_mutual_info.py" "$@"
      ;;
    fixed)
      run_with_python "${SCRIPTS_DIR}/find_gru_fixed_points.py" "$@"
      ;;
    evolve|evolution)
      run_with_python "${SCRIPTS_DIR}/analyze_fixed_point_evolution.py" "$@"
      ;;
    *)
      echo "Unknown observability subcommand '${sub}'. Use extract or analyze." >&2
      exit 1
      ;;
  esac
}

cmd_visualize() {
  if [[ "${1:-}" == "wizard" ]]; then
    shift
    run_wizard "${SCRIPTS_DIR}/analyze_trajectories_wizard.sh" "$@"
  else
    run_with_python "${SCRIPTS_DIR}/run_visualization_suite.py" "$@"
  fi
}

cmd_report() {
  run_with_python "${SCRIPTS_DIR}/generate_report.py" "$@"
}

cmd_onnx() {
  run_with_python "${SCRIPTS_DIR}/export_model_onnx.py" "$@"
}

cmd_factorial() {
  run_with_python "${SCRIPTS_DIR}/visualize_unified.py" "$@"
}

cmd_analyze() {
  run_wizard "${SCRIPTS_DIR}/analyze_trajectories_wizard.sh" "$@"
}

main() {
  if [[ $# -eq 0 ]]; then
    show_help
    exit 0
  fi

  local command="$1"
  shift || true

  case "${command}" in
    help|-h|--help)
      show_help
      ;;
    python-path)
      echo "${PYTHON_EXEC}"
      ;;
    dataset)
      cmd_dataset "$@"
      ;;
    train)
      run_with_python src/train.py "$@"
      ;;
    model)
      run_with_python src/model.py "$@"
      ;;
    train-all)
      cmd_train_all "$@"
      ;;
    analyze)
      cmd_analyze "$@"
      ;;
    metrics)
      cmd_metrics "$@"
      ;;
    cka)
      cmd_cka "$@"
      ;;
    embeddings)
      cmd_embeddings "$@"
      ;;
    trajectory-embedding)
      cmd_trajectory_embedding "$@"
      ;;
    observability)
      cmd_observability "$@"
      ;;
    visualize)
      cmd_visualize "$@"
      ;;
    report)
      cmd_report "$@"
      ;;
    onnx)
      cmd_onnx "$@"
      ;;
    factorial)
      cmd_factorial "$@"
      ;;
    *)
      echo "Unknown command '${command}'. Run './wt.sh help' for a list of commands." >&2
      exit 1
      ;;
  esac
}

main "$@"
