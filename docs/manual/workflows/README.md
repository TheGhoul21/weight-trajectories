# Analysis Workflows

Overview
- End‑to‑end guides for common analysis tasks, written as a practical manual to learn concepts through code.

How to Use
- Start from the workflow matching your goal below and follow the commands.
- Each workflow links to relevant command and plot pages for detail.

---

## Available Workflows

### [GRU Interpretability Pipeline](gru_interpretability.md)
Complete guide to analyzing recurrent network internals.

**What it covers**:
1. **Setup & extraction** - Collect gate statistics, eigenvalues, hidden samples
2. **Basic analysis** - Gate trajectories, timescales, PHATE embeddings
3. **Feature encoding** - Mutual information analysis (mean + per-dimension)
4. **Dynamical systems** - Fixed points, attractors, stability
5. **Diagnostic scenarios** - Debugging, architecture selection, paper figures

**Commands used**:
```bash
./wt.sh observability extract    # Data collection
./wt.sh observability analyze    # Gates + PHATE + probes
./wt.sh observability mi         # Mutual information
./wt.sh observability fixed      # Fixed-point finding
./wt.sh observability evolve     # Attractor evolution
```

**Outputs**: ~30 plots + CSVs covering gates, timescales, MI, fixed points

**When to use**:
- Understanding what GRU learns
- Debugging poor gameplay despite low loss
- Comparing GRU sizes (32 vs 64 vs 128)
- Generating paper figures

---

## Workflow Index by Goal

### "I trained a model, now what?"

**Quick diagnostics**:
1. Check training curves in `training_history.json`
2. Compute weight metrics: `./wt.sh metrics`
3. Visualize trajectory: `./wt.sh embeddings`
4. If using GRU: [GRU Interpretability Pipeline](gru_interpretability.md)

**What to look for**:
- Weight norms growing? (overfitting signal)
- Step cosines oscillating? (unstable training)
- Representation variance dropping? (collapse)

### "I want to understand GRU behavior"

**→ [GRU Interpretability Pipeline](gru_interpretability.md)**

Full 5-stage analysis covering:
- Memory mechanisms (gates, timescales)
- Feature encoding (MI, probes)
- Dynamical structure (fixed points)
- Evolution over training

### "I want to compare architectures"

**Cross-model analysis**:
1. Train all models: `./wt.sh train-all`
2. CKA similarity: `./wt.sh cka --representation gru`
3. Metric-space embedding: `./wt.sh trajectory-embedding`
4. Per-model GRU analysis: [GRU pipeline](gru_interpretability.md) for each

**Questions answered**:
- Do GRU32 and GRU128 learn similar representations?
- Which architecture uses capacity most efficiently?
- Do training dynamics differ?

### "I need publication figures"

**For weight trajectories**:
- [embeddings_weights](../plots/embeddings_weights.md) - Use PHATE with annotations

**For GRU analysis**:
- [GRU Interpretability Pipeline](gru_interpretability.md) → Scenario 3: "Preparing paper figures"
  - Gate trajectory plots
  - MI heatmaps (per-dimension)
  - Fixed-point evolution

**For CNN visualization**:
- [activations](../plots/activations.md) - Grad-CAM on critical game states

---

## Future Workflows

Planned guides:
- **Training hyperparameter tuning** - Grid search, learning rate schedules
- **Dataset balancing** - Handling imbalanced game phases
- **Transfer learning** - Using pre-trained CNN with fresh GRU
- **Ablation studies** - Systematic architecture ablations

---

## Cross-References

- **Commands**: [Commands Reference](manual/commands/)
- **Plot interpretation**: [Plots & Outputs](manual/plots/)
- **Main manual**: [User Manual Home](manual/)
- **Scientific background**: [Scientific Background](scientific/)
