# Command Reference

Complete reference for all `./wt.sh` commands.

---

## Quick Navigation

### Core Workflow
1. **[dataset](manual/commands/dataset)** - Generate Connect Four training data
2. **[train](manual/commands/train)** - Train single model configuration
3. **[train-all](manual/commands/train-all)** - Train full architecture sweep (9 models)
4. **[metrics](manual/commands/metrics)** - Compute checkpoint diagnostics (weight norms, SVD)
5. **[visualize](manual/commands/visualize)** - Generate trajectory embeddings and plots

### GRU Analysis
Complete pipeline for recurrent network interpretability:
- **[observability](manual/commands/observability)** - Extract and analyze GRU dynamics
  - `extract`: Gate statistics, eigenvalues, hidden samples
  - `analyze`: PHATE embeddings, logistic probes
  - `fixed`: Fixed-point finding
  - `evolve`: Attractor evolution tracking
  - `mi`: Mutual information analysis

### Cross-Model Comparison
- **[cka](manual/commands/cka)** - Representational similarity (CKA heatmaps)
- **[trajectory-embedding](manual/commands/trajectory-embedding)** - Metric-space trajectory plots
- **[factorial](manual/plots/factorial_heatmaps)** - Factorial analysis across architecture sweep
- **[analyze](manual/commands/analyze)** - Unified analysis wizard (interactive)

### Export & Reporting
- **[onnx](manual/commands/onnx)** - Export models to ONNX format
- **[model](manual/commands/model)** - Model inspection and testing
- **[report](manual/commands/report)** - Generate analysis reports

### Utilities
- **[python-path](manual/commands/python-path)** - Print Python interpreter path

---

## Commands by Category

### Data Preparation
| Command | Purpose | Outputs |
|---------|---------|---------|
| [dataset](manual/commands/dataset) | Generate sequential Connect Four games | `data/*.pt` |

### Training
| Command | Purpose | Outputs |
|---------|---------|---------|
| [train](manual/commands/train) | Train single model | `checkpoints/<run>/weights_epoch_*.pt` |
| [train-all](manual/commands/train-all) | Train 9 architectures | `checkpoints/save_every_3/k*_c*_gru*/` |

### Analysis & Diagnostics
| Command | Purpose | Key Outputs |
|---------|---------|-------------|
| [metrics](manual/commands/metrics) | Weight/repr statistics | `diagnostics/checkpoint_metrics/*.csv` |
| [observability](manual/commands/observability) | GRU dynamics | `diagnostics/gru_observability/`, `visualizations/gru_observability/` |
| [cka](manual/commands/cka) | Repr similarity | `visualizations/cka_*/` |
| [factorial](manual/plots/factorial_heatmaps) | Factorial analysis | `visualizations/factorial/factorial_heatmaps.png` |
| [analyze](manual/commands/analyze) | Interactive wizard | User-guided workflows |

### Visualization
| Command | Purpose | Key Outputs |
|---------|---------|-------------|
| [visualize](manual/commands/visualize) | Weight trajectories | `visualizations/*/` |
| [trajectory-embedding](manual/commands/trajectory-embedding) | Metric-space plots | `visualizations/trajectory_embedding_*.png` |

### Export
| Command | Purpose | Outputs |
|---------|---------|---------|
| [onnx](manual/commands/onnx) | Export to ONNX | `*.onnx` |
| [model](manual/commands/model) | Model info | stdout |
| [report](manual/commands/report) | Analysis reports | `reports/*.md` |

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

See [GRU Interpretability Workflow](manual/workflows/gru_interpretability) for complete guide.

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

## Cross-References

- **Plot outputs explained**: [Plots & Outputs](manual/plots/)
- **End-to-end workflows**: [Workflows](manual/workflows/)
- **Main manual**: [User Manual Home](manual/)
