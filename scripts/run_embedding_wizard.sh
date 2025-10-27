#!/usr/bin/env bash
# Interactive helper for launching analyze_weight_embeddings.py
set -euo pipefail

BASE_DIR=${1:-checkpoints/save_every_3}
SCRIPT_PATH="scripts/analyze_weight_embeddings.py"

if [[ ! -f ${SCRIPT_PATH} ]]; then
  echo "Error: ${SCRIPT_PATH} not found." >&2
  exit 1
fi

if [[ ! -d ${BASE_DIR} ]]; then
  echo "Error: base checkpoint directory '${BASE_DIR}' not found." >&2
  exit 1
fi

ALL_RUNS=()
while IFS= read -r line; do
  ALL_RUNS+=("${line}")
done < <(find "${BASE_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'k*_c*_gru*' | sort)

if (( ${#ALL_RUNS[@]} == 0 )); then
  echo "No checkpoint runs found under ${BASE_DIR}." >&2
  exit 1
fi

read -r -p "Component to analyze (cnn/gru/all) [cnn]: " COMPONENT
COMPONENT=${COMPONENT:-cnn}
case "${COMPONENT}" in
  cnn|gru|all) ;;
  *)
    echo "Unsupported component '${COMPONENT}'. Use cnn, gru, or all." >&2
    exit 1
    ;;
esac

read -r -p "Require specific kernel size (blank for any): " REQ_KERNEL
read -r -p "Require specific CNN channel count (blank for any): " REQ_CHANNEL
read -r -p "Require specific GRU hidden size (blank for any): " REQ_GRU

FILTERED_RUNS=()
for dir in "${ALL_RUNS[@]}"; do
  bn=$(basename "${dir}")
  if [[ ${bn} =~ k([0-9]+)_c([0-9]+)_gru([0-9]+) ]]; then
    kernel=${BASH_REMATCH[1]}
    channel=${BASH_REMATCH[2]}
    gru=${BASH_REMATCH[3]}
    if [[ -n ${REQ_KERNEL} && ${kernel} != ${REQ_KERNEL} ]]; then
      continue
    fi
    if [[ -n ${REQ_CHANNEL} && ${channel} != ${REQ_CHANNEL} ]]; then
      continue
    fi
    if [[ -n ${REQ_GRU} && ${gru} != ${REQ_GRU} ]]; then
      continue
    fi
    FILTERED_RUNS+=("${dir}")
  fi
done

if (( ${#FILTERED_RUNS[@]} == 0 )); then
  echo "No runs matched the requested filters." >&2
  exit 1
fi

echo "\nMatched runs:"
index=0
for dir in "${FILTERED_RUNS[@]}"; do
  bn=$(basename "${dir}")
  if [[ ${bn} =~ k([0-9]+)_c([0-9]+)_gru([0-9]+) ]]; then
    printf "  [%d] %s (k=%s, c=%s, gru=%s)\n" "${index}" "${dir}" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
  else
    printf "  [%d] %s\n" "${index}" "${dir}"
  fi
  ((index++))
done

echo
read -r -p "Enter space-separated indices to include (blank for all): " SELECTION
SELECTED_RUNS=()
if [[ -z ${SELECTION} ]]; then
  SELECTED_RUNS=("${FILTERED_RUNS[@]}")
else
  for idx in ${SELECTION}; do
    if ! [[ ${idx} =~ ^[0-9]+$ ]] || (( idx < 0 || idx >= ${#FILTERED_RUNS[@]} )); then
      echo "Invalid index '${idx}'." >&2
      exit 1
    fi
    SELECTED_RUNS+=("${FILTERED_RUNS[idx]}")
  done
fi

if (( ${#SELECTED_RUNS[@]} < 2 )); then
  echo "Need at least two runs to compare; adjust selection." >&2
  exit 1
fi

read -r -p "Embedding methods (space separated) [pca tsne umap phate]: " METHOD_INPUT
if [[ -z ${METHOD_INPUT} ]]; then
  METHOD_LIST=(pca tsne umap phate)
else
  METHOD_LIST=(${METHOD_INPUT})
fi

read -r -p "Output directory [visualizations/simple_embeddings/custom_run]: " OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-visualizations/simple_embeddings/custom_run}

read -r -p "Enable comparison overlays? (y/N): " REPLY_COMPARE
read -r -p "Annotate key checkpoints? (y/N): " REPLY_ANNOTATE
read -r -p "Export CSV embeddings? (y/N): " REPLY_CSV

read -r -p "Create per-run animations? (Y/n): " REPLY_ANIMATE
REPLY_ANIMATE=${REPLY_ANIMATE:-Y}
read -r -p "Animation FPS [4]: " ANIMATE_FPS
ANIMATE_FPS=${ANIMATE_FPS:-4}

read -r -p "Include board representation embeddings? (y/N): " REPLY_REPR
if [[ ${REPLY_REPR} =~ ^[Yy]$ ]]; then
  read -r -p "Board source (random/dataset) [random]: " BOARD_SOURCE
  BOARD_SOURCE=${BOARD_SOURCE:-random}
  if [[ ${BOARD_SOURCE} != "random" && ${BOARD_SOURCE} != "dataset" ]]; then
    echo "Unsupported board source '${BOARD_SOURCE}'." >&2
    exit 1
  fi
  read -r -p "Number of board samples [10]: " BOARD_COUNT
  BOARD_COUNT=${BOARD_COUNT:-10}
  if [[ ! ${BOARD_COUNT} =~ ^[0-9]+$ ]] || (( BOARD_COUNT <= 0 )); then
    echo "Board count must be a positive integer." >&2
    exit 1
  fi
  if [[ ${BOARD_SOURCE} == "dataset" ]]; then
    read -r -p "Dataset path [data/connect4_sequential_10k_games.pt]: " BOARD_DATASET
    BOARD_DATASET=${BOARD_DATASET:-data/connect4_sequential_10k_games.pt}
  fi
  read -r -p "Board seed [37]: " BOARD_SEED
  BOARD_SEED=${BOARD_SEED:-37}
  read -r -p "Representation embedding methods (space separated, default = methods above): " REPR_METHOD_INPUT
  if [[ -z ${REPR_METHOD_INPUT} ]]; then
    REPR_METHOD_LIST=(${METHOD_LIST[@]})
  else
    REPR_METHOD_LIST=(${REPR_METHOD_INPUT})
  fi
fi

declare -a CMD
CMD+=(python "${SCRIPT_PATH}" --checkpoint-dirs)
for dir in "${SELECTED_RUNS[@]}"; do
  CMD+=("${dir}")
done
CMD+=(--component "${COMPONENT}" --methods)
for method in "${METHOD_LIST[@]}"; do
  CMD+=("${method}")
done
CMD+=(--output-dir "${OUTPUT_DIR}")

if [[ -n ${REQ_KERNEL} ]]; then
  CMD+=(--require-kernel "${REQ_KERNEL}")
fi
if [[ -n ${REQ_CHANNEL} ]]; then
  CMD+=(--require-channel "${REQ_CHANNEL}")
fi
if [[ -n ${REQ_GRU} ]]; then
  CMD+=(--require-gru "${REQ_GRU}")
fi
if [[ ${REPLY_COMPARE} =~ ^[Yy]$ ]]; then
  CMD+=(--compare)
fi
if [[ ${REPLY_ANNOTATE} =~ ^[Yy]$ ]]; then
  CMD+=(--annotate)
fi
if [[ ${REPLY_CSV} =~ ^[Yy]$ ]]; then
  CMD+=(--export-csv)
fi

if [[ -z ${REPLY_ANIMATE} || ${REPLY_ANIMATE} =~ ^[Yy]$ ]]; then
  CMD+=(--animate --animate-fps "${ANIMATE_FPS}")
fi

if [[ ${REPLY_REPR} =~ ^[Yy]$ ]]; then
  CMD+=(--board-representations --board-source "${BOARD_SOURCE}" --board-count "${BOARD_COUNT}" --board-seed "${BOARD_SEED}")
  if [[ ${BOARD_SOURCE} == "dataset" ]]; then
    CMD+=(--board-dataset "${BOARD_DATASET}")
  fi
  CMD+=(--representation-methods)
  for method in "${REPR_METHOD_LIST[@]}"; do
    CMD+=("${method}")
  done
fi

echo
echo "About to run:"
printf ' %q' "${CMD[@]}"
echo -e "\n"
read -r -p "Proceed? (Y/n): " CONFIRM
if [[ ${CONFIRM} =~ ^[Nn]$ ]]; then
  echo "Aborted."
  exit 0
fi

echo
"${CMD[@]}"
