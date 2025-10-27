#!/usr/bin/env bash
# Interactive helper for compute_checkpoint_metrics.py
set -euo pipefail

BASE_DIR=${1:-checkpoints/save_every_3}
SCRIPT_PATH="scripts/compute_checkpoint_metrics.py"

if [[ ! -f ${SCRIPT_PATH} ]]; then
  echo "Error: ${SCRIPT_PATH} not found." >&2
  exit 1
fi

if [[ ! -d ${BASE_DIR} ]]; then
  echo "Error: checkpoint base directory '${BASE_DIR}' not found." >&2
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

read -r -p "Component to analyze (cnn/gru/all) [gru]: " COMPONENT
COMPONENT=${COMPONENT:-gru}
case "${COMPONENT}" in
  cnn|gru|all) ;;
  *)
    echo "Unsupported component '${COMPONENT}'." >&2
    exit 1
    ;;
esac

read -r -p "Require kernel size (blank for any): " REQ_KERNEL
read -r -p "Require channel count (blank for any): " REQ_CHANNEL
read -r -p "Require GRU size (blank for any): " REQ_GRU

FILTERED_RUNS=()
for dir in "${ALL_RUNS[@]}"; do
  bn=$(basename "${dir}")
  if [[ ${bn} =~ k([0-9]+)_c([0-9]+)_gru([0-9]+) ]]; then
    k=${BASH_REMATCH[1]}
    c=${BASH_REMATCH[2]}
    g=${BASH_REMATCH[3]}
    if [[ -n ${REQ_KERNEL} && ${k} != ${REQ_KERNEL} ]]; then
      continue
    fi
    if [[ -n ${REQ_CHANNEL} && ${c} != ${REQ_CHANNEL} ]]; then
      continue
    fi
    if [[ -n ${REQ_GRU} && ${g} != ${REQ_GRU} ]]; then
      continue
    fi
    FILTERED_RUNS+=("${dir}")
  fi
done

if (( ${#FILTERED_RUNS[@]} == 0 )); then
  echo "No runs matched the requested filters." >&2
  exit 1
fi

echo
echo "Matched runs:"
for idx in "${!FILTERED_RUNS[@]}"; do
  bn=$(basename "${FILTERED_RUNS[idx]}")
  printf "  [%d] %s\n" "${idx}" "${bn}"
done

echo
read -r -p "Enter space-separated indices to analyze (blank for all): " SELECTION
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

read -r -p "Epoch minimum (blank for none): " EPOCH_MIN
read -r -p "Epoch maximum (blank for none): " EPOCH_MAX
read -r -p "Epoch stride [1]: " EPOCH_STEP
EPOCH_STEP=${EPOCH_STEP:-1}

read -r -p "Board source (none/random/dataset) [none]: " BOARD_SOURCE
BOARD_SOURCE=${BOARD_SOURCE:-none}
case "${BOARD_SOURCE}" in
  none|random|dataset) ;;
  *)
    echo "Unsupported board source '${BOARD_SOURCE}'." >&2
    exit 1
    ;;
esac

if [[ ${BOARD_SOURCE} != "none" ]]; then
  read -r -p "Board count [16]: " BOARD_COUNT
  BOARD_COUNT=${BOARD_COUNT:-16}
  read -r -p "Board seed [37]: " BOARD_SEED
  BOARD_SEED=${BOARD_SEED:-37}
  if [[ ${BOARD_SOURCE} == "dataset" ]]; then
    read -r -p "Dataset path [data/connect4_sequential_10k_games.pt]: " BOARD_DATASET
    BOARD_DATASET=${BOARD_DATASET:-data/connect4_sequential_10k_games.pt}
  fi
else
  BOARD_COUNT=
  BOARD_SEED=
  BOARD_DATASET=
fi

read -r -p "Number of leading singular values to report [4]: " TOP_SV
TOP_SV=${TOP_SV:-4}

read -r -p "Output directory [diagnostics/checkpoint_metrics]: " OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-diagnostics/checkpoint_metrics}

PYTHON_BIN=${PYTHON_BIN:-python3}

declare -a CMD
CMD+=("${PYTHON_BIN}" "${SCRIPT_PATH}" --checkpoint-dirs)
for dir in "${SELECTED_RUNS[@]}"; do
  CMD+=("${dir}")
done
CMD+=(--component "${COMPONENT}" --epoch-step "${EPOCH_STEP}" --output-dir "${OUTPUT_DIR}" --top-singular-values "${TOP_SV}")

if [[ -n ${EPOCH_MIN} ]]; then
  CMD+=(--epoch-min "${EPOCH_MIN}")
fi
if [[ -n ${EPOCH_MAX} ]]; then
  CMD+=(--epoch-max "${EPOCH_MAX}")
fi

CMD+=(--board-source "${BOARD_SOURCE}")
if [[ ${BOARD_SOURCE} != "none" ]]; then
  CMD+=(--board-count "${BOARD_COUNT}" --board-seed "${BOARD_SEED}")
  if [[ ${BOARD_SOURCE} == "dataset" ]]; then
    CMD+=(--board-dataset "${BOARD_DATASET}")
  fi
fi

echo
echo "Running:"
printf "  %q" "${CMD[@]}"
echo

"${CMD[@]}"
