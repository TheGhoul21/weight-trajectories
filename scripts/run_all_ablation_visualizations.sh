#!/usr/bin/env bash

set -euo pipefail

# ensure we are in repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

CMD="python -m src.visualize_trajectories"

ensure_pillow() {
  if ! python -c "import PIL" >/dev/null 2>&1; then
    echo "Installing Pillow for GIF export..."
    pip install pillow
  fi
}

ensure_pillow

declare -a commands=(
  "${CMD} --viz-type ablation-cnn --ablation-dirs checkpoints/k3_c16_gru8_20251024_033336 checkpoints/k3_c64_gru8_20251024_033906 checkpoints/k3_c256_gru8_20251024_073250 --output-dir visualizations/ablation_cnn_k3_gru8 --ablation-animate --ablation-center anchor"
  "${CMD} --viz-type ablation-cnn --ablation-dirs checkpoints/k3_c16_gru32_20251024_043035 checkpoints/k3_c64_gru32_20251024_043922 checkpoints/k3_c256_gru32_20251024_045446 --output-dir visualizations/ablation_cnn_k3_gru32 --ablation-animate --ablation-center anchor"
  "${CMD} --viz-type ablation-cnn --ablation-dirs checkpoints/k3_c16_gru128_20251024_065813 checkpoints/k3_c64_gru128_20251024_071751 checkpoints/k3_c256_gru128_20251024_055808 --output-dir visualizations/ablation_cnn_k3_gru128 --ablation-animate --ablation-center anchor"
  "${CMD} --viz-type ablation-cnn --ablation-dirs checkpoints/k6_c16_gru8_20251024_085919 checkpoints/k6_c64_gru8_20251024_092656 checkpoints/k6_c256_gru8_20251024_104719 --output-dir visualizations/ablation_cnn_k6_gru8 --ablation-animate --ablation-center anchor"
  "${CMD} --viz-type ablation-cnn --ablation-dirs checkpoints/k6_c16_gru32_20251024_090804 checkpoints/k6_c64_gru32_20251024_095241 checkpoints/k6_c256_gru32_20251024_150525 --output-dir visualizations/ablation_cnn_k6_gru32 --ablation-animate --ablation-center anchor"
  "${CMD} --viz-type ablation-cnn --ablation-dirs checkpoints/k6_c16_gru128_20251024_091703 checkpoints/k6_c64_gru128_20251024_101833 checkpoints/k6_c256_gru128_20251024_174300 --output-dir visualizations/ablation_cnn_k6_gru128 --ablation-animate --ablation-center anchor"
  "${CMD} --viz-type ablation-gru --ablation-dirs checkpoints/k3_c16_gru8_20251024_033336 checkpoints/k3_c16_gru32_20251024_043035 checkpoints/k3_c16_gru128_20251024_065813 --output-dir visualizations/ablation_gru_k3_c16 --ablation-animate --ablation-center anchor"
  "${CMD} --viz-type ablation-gru --ablation-dirs checkpoints/k3_c64_gru8_20251024_033906 checkpoints/k3_c64_gru32_20251024_043922 checkpoints/k3_c64_gru128_20251024_071751 --output-dir visualizations/ablation_gru_k3_c64 --ablation-animate --ablation-center anchor"
  "${CMD} --viz-type ablation-gru --ablation-dirs checkpoints/k3_c256_gru8_20251024_073250 checkpoints/k3_c256_gru32_20251024_045446 checkpoints/k3_c256_gru128_20251024_055808 --output-dir visualizations/ablation_gru_k3_c256 --ablation-animate --ablation-center anchor"
  "${CMD} --viz-type ablation-gru --ablation-dirs checkpoints/k6_c16_gru8_20251024_085919 checkpoints/k6_c16_gru32_20251024_090804 checkpoints/k6_c16_gru128_20251024_091703 --output-dir visualizations/ablation_gru_k6_c16 --ablation-animate --ablation-center anchor"
  "${CMD} --viz-type ablation-gru --ablation-dirs checkpoints/k6_c64_gru8_20251024_092656 checkpoints/k6_c64_gru32_20251024_095241 checkpoints/k6_c64_gru128_20251024_101833 --output-dir visualizations/ablation_gru_k6_c64 --ablation-animate --ablation-center anchor"
  "${CMD} --viz-type ablation-gru --ablation-dirs checkpoints/k6_c256_gru8_20251024_104719 checkpoints/k6_c256_gru32_20251024_150525 checkpoints/k6_c256_gru128_20251024_174300 --output-dir visualizations/ablation_gru_k6_c256 --ablation-animate --ablation-center anchor"
)

for run_cmd in "${commands[@]}"; do
  printf "\n========================================\n"
  printf "Running: %s\n" "${run_cmd}"
  printf "========================================\n"
  eval "${run_cmd}"
done

printf "\nAll ablation visualizations complete.\n"