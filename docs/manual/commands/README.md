# Command Reference

Complete reference for all `./wt.sh` commands.

---

## Quick Navigation

### Core Workflow
1. **[dataset](dataset.md)** - Generate Connect Four training data
2. **[train](train.md)** - Train single model configuration
3. **[train-all](train-all.md)** - Train full architecture sweep (9 models)
4. **[metrics](metrics.md)** - Compute checkpoint diagnostics (weight norms, SVD)
5. **[visualize](visualize.md)** - Generate trajectory embeddings and plots

### GRU Analysis
Complete pipeline for recurrent network interpretability:
- **[observability](observability.md)** - Extract and analyze GRU dynamics
  - `extract`: Gate statistics, eigenvalues, hidden samples
  - `analyze`: PHATE embeddings, logistic probes
  - `fixed`: Fixed-point finding
  - `evolve`: Attractor evolution tracking
  - `mi`: Mutual information analysis

### Cross-Model Comparison
- **[cka](cka.md)** - Representational similarity (CKA heatmaps)
- **[trajectory-embedding](trajectory-embedding.md)** - Metric-space trajectory plots
- **[factorial](../plots/factorial_heatmaps.md)** - Factorial analysis across architecture sweep
- **[analyze](analyze.md)** - Unified analysis wizard (interactive)

### Export & Reporting
- **[onnx](onnx.md)** - Export models to ONNX format
- **[model](model.md)** - Model inspection and testing
- **[report](report.md)** - Generate analysis reports

### Utilities
- **[python-path](python-path.md)** - Print Python interpreter path

---

## Commands by Category

### Data Preparation
| Command | Purpose | Outputs |
|---------|---------|---------|
| [dataset](dataset.md) | Generate sequential Connect Four games | `data/*.pt` |

### Training
| Command | Purpose | Outputs |
|---------|---------|---------|
| [train](train.md) | Train single model | `checkpoints/<run>/weights_epoch_*.pt` |
| [train-all](train-all.md) | Train 9 architectures | `checkpoints/save_every_3/k*_c*_gru*/` |

### Analysis & Diagnostics
| Command | Purpose | Key Outputs |
|---------|---------|-------------|
| [metrics](metrics.md) | Weight/repr statistics | `diagnostics/checkpoint_metrics/*.csv` |
| [observability](observability.md) | GRU dynamics | `diagnostics/gru_observability/`, `visualizations/gru_observability/` |
| [cka](cka.md) | Repr similarity | `visualizations/cka_*/` |
| [factorial](../plots/factorial_heatmaps.md) | Factorial analysis | `visualizations/factorial/factorial_heatmaps.png` |
| [analyze](analyze.md) | Interactive wizard | User-guided workflows |

### Visualization
| Command | Purpose | Key Outputs |
|---------|---------|-------------|
| [visualize](visualize.md) | Weight trajectories | `visualizations/*/` |
| [trajectory-embedding](trajectory-embedding.md) | Metric-space plots | `trajectory_embedding_all.png`, `..._by_gru.png`, `..._3d.png`, optional `..._3d_rotate.gif`, `..._3d.html` |

### Export
| Command | Purpose | Outputs |
|---------|---------|---------|
| [onnx](onnx.md) | Export to ONNX | `*.onnx` |
| [model](model.md) | Model info | stdout |
| [report](report.md) | Analysis reports | `reports/*.md` |

---

## Command Workflow Examples

### Quick sanity check
```bash
./wt.sh dataset          # Generate data
./wt.sh train            # Train one model
./wt.sh metrics          # Check weight norms
./wt.sh visualize        # Plot trajectory
```

### Full GRU analysis pipeline
```bash
./wt.sh train-all                    # Train 9 models
./wt.sh observability extract         # Collect GRU data
./wt.sh observability analyze         # Generate plots
./wt.sh observability mi              # Mutual information
./wt.sh observability fixed           # Find fixed points
./wt.sh observability evolve          # Track evolution
```

See [GRU Interpretability Workflow](../workflows/gru_interpretability.md) for complete guide.

### Compare architectures
```bash
./wt.sh train-all                    # Train all models
./wt.sh cka --representation gru      # Similarity analysis
./wt.sh trajectory-embedding          # Metric-space view
```

---

## Getting Help

- **Command help**: `./wt.sh <command> --help`
- **General help**: `./wt.sh help`
- **Workflow guides**: [Workflows](manual/workflows/)
- **Plot interpretation**: [Plots & Outputs](manual/plots/)

---

## Caching and recomputation

Several analysis commands cache intermediate CSVs to speed up repeated plotting. Use the flags below to control recomputation:

- metrics — supports `--force` to recompute `{run}_metrics.csv` even if it exists.
- cka — supports `--force` to recompute similarity matrices and `--skip-plots` to only refresh CSVs without figures.
- observability (MI) — supports `--force` to recompute `mi_results.csv` when present.

Tip: When preparing inputs for `trajectory-embedding`, write metrics to `diagnostics/trajectory_analysis`:

```bash
./wt.sh metrics --output-dir diagnostics/trajectory_analysis ...
```

---

## Cross-References

- **Plot outputs explained**: [Plots & Outputs](manual/plots/)
- **End-to-end workflows**: [Workflows](manual/workflows/)
- **Main manual**: [User Manual Home](manual/)
