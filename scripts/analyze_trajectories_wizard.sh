#!/usr/bin/env bash
# Master wizard for complete trajectory analysis
# Generates metrics, visualizations, and final report

set -euo pipefail

echo "================================================================================"
echo "Connect-4 Weight Trajectory Analysis Wizard"
echo "Analyzing Learning Trajectories in a Solved Game"
echo "================================================================================"
echo

# Configuration
CHECKPOINT_BASE=${1:-checkpoints/save_every_3}
OUTPUT_DIR=${2:-diagnostics/trajectory_analysis}
VIZ_DIR=${3:-visualizations}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python3}

# Check prerequisites
if [[ ! -f "scripts/compute_checkpoint_metrics.py" ]]; then
    echo "Error: scripts/compute_checkpoint_metrics.py not found"
    exit 1
fi

if [[ ! -d "${CHECKPOINT_BASE}" ]]; then
    echo "Error: Checkpoint directory '${CHECKPOINT_BASE}' not found"
    exit 1
fi

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${VIZ_DIR}"

echo "Configuration:"
echo "  Checkpoint base: ${CHECKPOINT_BASE}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Visualization directory: ${VIZ_DIR}"
echo "  Python binary: ${PYTHON_BIN}"
echo

# Step 1: Generate checkpoint metrics
echo "================================================================================"
echo "Step 1/4: Generating checkpoint metrics with representation analysis"
echo "================================================================================"
echo

# Find all model directories
ALL_MODELS=($(find "${CHECKPOINT_BASE}" -maxdepth 1 -mindepth 1 -type d -name 'k*_c*_gru*' | sort))

if (( ${#ALL_MODELS[@]} == 0 )); then
    echo "Error: No model directories found in ${CHECKPOINT_BASE}"
    exit 1
fi

echo "Found ${#ALL_MODELS[@]} models:"
for model in "${ALL_MODELS[@]}"; do
    echo "  - $(basename "${model}")"
done
echo

# Generate metrics for all models
"${PYTHON_BIN}" scripts/compute_checkpoint_metrics.py \
    --checkpoint-dirs "${ALL_MODELS[@]}" \
    --component all \
    --epoch-step 1 \
    --board-source random \
    --board-count 16 \
    --board-seed 37 \
    --top-singular-values 4 \
    --output-dir "${OUTPUT_DIR}"

echo
echo "✓ Checkpoint metrics generated"
echo

# Step 2: Compute advanced derived metrics
echo "================================================================================"
echo "Step 2/4: Computing advanced trajectory metrics"
echo "================================================================================"
echo

"${PYTHON_BIN}" -c "
import sys
sys.path.insert(0, 'scripts')

# Import and run advanced metrics computation inline
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import linregress

def load_all_metrics(base_dir):
    from pathlib import Path
    data = {}
    base_path = Path(base_dir)

    for csv_file in base_path.glob('k*_c*_gru*_metrics.csv'):
        model_name = csv_file.stem.replace('_metrics', '')
        data[model_name] = pd.read_csv(csv_file)

    return data

def compute_trajectory_stats(df):
    stats = {}

    if 'weight_norm' in df.columns:
        stats['initial_weight_norm'] = df['weight_norm'].iloc[0]
        stats['final_weight_norm'] = df['weight_norm'].iloc[-1]
        stats['weight_growth_factor'] = stats['final_weight_norm'] / stats['initial_weight_norm']

    if 'step_norm' in df.columns:
        valid_steps = df['step_norm'].dropna()
        if len(valid_steps) > 0:
            stats['mean_step_norm'] = valid_steps.mean()
            stats['step_norm_cv'] = valid_steps.std() / valid_steps.mean()
            stats['total_trajectory_length'] = valid_steps.sum()

    if 'step_cosine' in df.columns:
        valid_cosine = df['step_cosine'].dropna()
        if len(valid_cosine) > 0:
            stats['final_cosine'] = valid_cosine.iloc[-1]

    if 'relative_step' in df.columns:
        valid_rel = df['relative_step'].dropna()
        if len(valid_rel) > 2:
            # Exponential decay fit
            log_rel = np.log(valid_rel + 1e-10)
            epochs = df.loc[valid_rel.index, 'epoch']
            slope, _, r_value, _, _ = linregress(epochs, log_rel)
            stats['relative_step_decay_constant'] = -slope
            stats['relative_step_half_life'] = np.log(2) / max(-slope, 1e-10)

    if 'repr_total_variance' in df.columns:
        valid_repr = df['repr_total_variance'].dropna()
        if len(valid_repr) > 0:
            stats['mean_total_variance'] = valid_repr.mean()

    if 'repr_top1_ratio' in df.columns:
        valid_top1 = df['repr_top1_ratio'].dropna()
        if len(valid_top1) > 0:
            stats['mean_top1_ratio'] = valid_top1.mean()
            stats['repr_concentration_score'] = stats['mean_top1_ratio'] / (1.0/128)

    return stats

# Load and process
data = load_all_metrics('${OUTPUT_DIR}')
print(f'Loaded {len(data)} models')

all_stats = {}
for model_name, df in data.items():
    all_stats[model_name] = compute_trajectory_stats(df)

# Save summary
summary_df = pd.DataFrame(all_stats).T
summary_df.to_csv('${OUTPUT_DIR}/trajectory_summary.csv')
print(f'Saved trajectory summary to ${OUTPUT_DIR}/trajectory_summary.csv')
"

echo
echo "✓ Advanced metrics computed"
echo

# Step 3: Generate all visualizations
echo "================================================================================"
echo "Step 3/4: Generating comprehensive visualizations"
echo "================================================================================"
echo

"${PYTHON_BIN}" scripts/visualize_unified.py \
    --metrics-dir "${OUTPUT_DIR}" \
    --checkpoint-dir "${CHECKPOINT_BASE}" \
    --output-dir "${VIZ_DIR}"

echo
echo "✓ Visualizations generated"
echo

# Step 4: Generate unified report
echo "================================================================================"
echo "Step 4/4: Generating unified analysis report"
echo "================================================================================"
echo

"${PYTHON_BIN}" scripts/generate_report.py \
    --metrics-dir "${OUTPUT_DIR}" \
    --checkpoint-dir "${CHECKPOINT_BASE}" \
    --viz-dir "${VIZ_DIR}" \
    --output "${OUTPUT_DIR}/ANALYSIS_REPORT.md"

echo
echo "✓ Report generated"
echo

# Summary
echo
echo "================================================================================"
echo "Analysis Complete!"
echo "================================================================================"
echo
echo "Outputs:"
echo "  📊 Metrics:        ${OUTPUT_DIR}/*.csv"
echo "  📈 Visualizations: ${VIZ_DIR}/*.png"
echo "  📝 Report:         ${OUTPUT_DIR}/ANALYSIS_REPORT.md"
echo
echo "Quick start:"
echo "  Read:  ${OUTPUT_DIR}/ANALYSIS_REPORT.md"
echo "  View:  open ${VIZ_DIR}/*.png"
echo
echo "================================================================================"
