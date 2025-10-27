#!/usr/bin/env bash
# Interactive wizard for CKA similarity analysis
set -euo pipefail

SCRIPT_PATH="scripts/compute_cka_similarity.py"
DEFAULT_CHECKPOINT_DIR="checkpoints/save_every_3"
DEFAULT_OUTPUT_DIR="visualizations/cka"
DEFAULT_NUM_BOARDS=64
DEFAULT_SEED=42
DEFAULT_DEVICE="cpu"
DEFAULT_EPOCH_STEP=3
DEFAULT_ANIMATE="Y"
DEFAULT_ANIM_FPS=4
DEFAULT_ANIM_FORMAT="gif"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python3}

if [[ ! -f ${SCRIPT_PATH} ]]; then
  echo "Error: ${SCRIPT_PATH} not found." >&2
  exit 1
fi

read -r -p "Checkpoint base directory [${DEFAULT_CHECKPOINT_DIR}]: " CHECKPOINT_DIR
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${DEFAULT_CHECKPOINT_DIR}}
if [[ ! -d ${CHECKPOINT_DIR} ]]; then
  echo "Error: checkpoint directory '${CHECKPOINT_DIR}' not found." >&2
  exit 1
fi

read -r -p "Epoch selection mode (step/custom) [step]: " EPOCH_MODE
EPOCH_MODE=${EPOCH_MODE:-step}

EPOCH_ARGS=()
if [[ ${EPOCH_MODE} == "step" ]]; then
  read -r -p "Epoch step [${DEFAULT_EPOCH_STEP}]: " EPOCH_STEP
  EPOCH_STEP=${EPOCH_STEP:-${DEFAULT_EPOCH_STEP}}
  if [[ ! ${EPOCH_STEP} =~ ^[0-9]+$ ]] || (( EPOCH_STEP <= 0 )); then
    echo "Epoch step must be a positive integer." >&2
    exit 1
  fi
  EPOCH_ARGS=(--epoch-step "${EPOCH_STEP}")
else
  read -r -p "Custom epochs (space-separated, e.g., '3 6 9 12 100') [3 6 9 12 100]: " EPOCH_LIST
  EPOCH_LIST=${EPOCH_LIST:-"3 6 9 12 100"}
  # Basic validation: ensure space-separated integers
  for e in ${EPOCH_LIST}; do
    if ! [[ ${e} =~ ^[0-9]+$ ]]; then
      echo "Invalid epoch '${e}'. Use integers." >&2
      exit 1
    fi
  done
  EPOCH_ARGS=(--epochs ${EPOCH_LIST})
fi

read -r -p "Number of test boards [${DEFAULT_NUM_BOARDS}]: " NUM_BOARDS
NUM_BOARDS=${NUM_BOARDS:-${DEFAULT_NUM_BOARDS}}
if ! [[ ${NUM_BOARDS} =~ ^[0-9]+$ ]] || (( NUM_BOARDS <= 0 )); then
  echo "Number of boards must be a positive integer." >&2
  exit 1
fi

read -r -p "Random seed [${DEFAULT_SEED}]: " SEED
SEED=${SEED:-${DEFAULT_SEED}}
if ! [[ ${SEED} =~ ^[0-9]+$ ]]; then
  echo "Seed must be an integer." >&2
  exit 1
fi

read -r -p "Device (cpu/cuda) [${DEFAULT_DEVICE}]: " DEVICE
DEVICE=${DEVICE:-${DEFAULT_DEVICE}}
if [[ ${DEVICE} != "cpu" && ${DEVICE} != "cuda" ]]; then
  echo "Device must be 'cpu' or 'cuda'." >&2
  exit 1
fi

read -r -p "Output directory [${DEFAULT_OUTPUT_DIR}]: " OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}

read -r -p "Animate heatmap over epochs? (Y/n) [${DEFAULT_ANIMATE}]: " REPLY_ANIM
REPLY_ANIM=${REPLY_ANIM:-${DEFAULT_ANIMATE}}
ANIMATE_FLAG=()
if [[ ${REPLY_ANIM} =~ ^[Yy]$ ]]; then
  ANIMATE_FLAG+=(--animate)
  read -r -p "Animation FPS [${DEFAULT_ANIM_FPS}]: " ANIM_FPS
  ANIM_FPS=${ANIM_FPS:-${DEFAULT_ANIM_FPS}}
  if ! [[ ${ANIM_FPS} =~ ^[0-9]+$ ]] || (( ANIM_FPS <= 0 )); then
    echo "FPS must be a positive integer." >&2
    exit 1
  fi
  ANIMATE_FLAG+=(--animate-fps "${ANIM_FPS}")
  read -r -p "Animation format (gif/mp4) [${DEFAULT_ANIM_FORMAT}]: " ANIM_FMT
  ANIM_FMT=${ANIM_FMT:-${DEFAULT_ANIM_FORMAT}}
  if [[ ${ANIM_FMT} != "gif" && ${ANIM_FMT} != "mp4" ]]; then
    echo "Animation format must be 'gif' or 'mp4'." >&2
    exit 1
  fi
  ANIMATE_FLAG+=(--animate-format "${ANIM_FMT}")
fi

# Build command
CMD=("${PYTHON_BIN}" "${SCRIPT_PATH}" "--checkpoint-dir" "${CHECKPOINT_DIR}" "--output-dir" "${OUTPUT_DIR}" "--num-boards" "${NUM_BOARDS}" "--seed" "${SEED}" "--device" "${DEVICE}")
CMD+=("${EPOCH_ARGS[@]}")
CMD+=("${ANIMATE_FLAG[@]}")

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
