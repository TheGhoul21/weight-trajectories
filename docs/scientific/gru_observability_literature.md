# GRU Observability: Intuition, Methods, and Recommendations

Imagine opening a fresh set of training checkpoints and trying to answer three questions: Is the GRU remembering what it should, has it learned the right strategic concepts, and where do those behaviours live in state space? `Observability` is the guide for that investigation. It starts with the questions, maps them to diagnostics, and anchors every recommendation in the formal results summarised in `theoretical_foundations.md`. If you prefer a condensed walkthrough, pair this document with the focused field reports in `guide_memory_diagnostics.md` and `guide_representation_diagnostics.md`.

---

## Diagnostic Modules at a Glance

| Module | Core questions | Inputs | Primary outputs | Scripts / notebooks |
| --- | --- | --- | --- | --- |
| Gates and timescales | Are we retaining memory? When do we overwrite state? What are the effective dynamical timescales? | Checkpoints, replay buffer of trajectories | Gate histograms, spectral radii, per-unit time constants | `scripts/extract_gru_dynamics.py` |
| Hidden-state geometry | How do representations organise in hidden space? Which features carve the state manifold? | Hidden-state samples with board-feature labels | PHATE/T-PHATE embeddings, feature-coloured trajectories | `src/visualize_trajectories.py --viz-type hidden` |
| Probing and mutual information | Which features are encoded, and are they linearly accessible? How specialised are neurons? | Hidden-state matrices plus labels (and shuffled controls) | Probe accuracy curves, MI heatmaps, per-dimension distributions | `scripts/run_linear_probes.py` (planned), `scripts/compute_hidden_mutual_info.py` |
| Fixed points and attractor evolution | Which stable modes exist? How do they move as training progresses? | Frozen CNN embeddings for canonical boards, optimisation loop | Fixed-point catalogue, Jacobian spectra, basin assignments | `scripts/find_fixed_points.py` (in development) |
| Training-phase synopsis | When do representations crystallise relative to validation metrics? | Aggregated outputs from the modules above | Timeline plots, regime annotations | `notebooks/observability/*.ipynb` |

---

## Measurement Playbooks

Each playbook reads like a field report: first the story you are trying to confirm, then the data you need, followed by the measurements and how to read them. Every step links back to the relevant theory section for formal support.

### 1. Gates and Timescales

**Narrative hook**: “Did the model learn when to remember versus when to overwrite?”

**Goal**: quantify memory depth and gating policy across checkpoints.

**Inputs**: checkpoint directory (`weights_epoch_*.pt`), `training_history.json`, sampled hidden states from replay.

**Procedure**:
1. Run `scripts/extract_gru_dynamics.py --analysis-dir diagnostics/...` to log gate activations, recurrent eigenvalues, and time constants per epoch.
2. Plot mean and variance of update/reset gates across epochs; overlay validation accuracy for alignment.
3. Inspect eigenvalue spectra; convert eigenvalues to time constants via `tau = -1 / ln |lambda|` (see §1.4 in `theoretical_foundations.md`).

**Interpretation heuristics**:
- Healthy runs show update gates drifting from roughly 0.3 toward 0.6 while avoiding saturation.
- Spectral radii near 1 indicate long memory; values above 1 highlight potential instability, while values far below 1 suggest memoryless behaviour.
- Compare time-constant distributions across architectures to quantify effective capacity.

**Cross-references**: `theoretical_foundations.md` §§1–2 for Jacobian derivations; `weight_embeddings_theory.md` for how these metrics align with trajectory plots.

### 2. Hidden-State Geometry

**Narrative hook**: “What shape did the representations take as the model learned strategy?”

**Goal**: reveal structure in the representation manifold and how board features partition it.

**Inputs**: hidden-state samples stratified by feature labels (e.g., immediate win, current player, move index).

**Procedure**:
1. Sample hidden states uniformly across games and timesteps; balance samples across feature values to avoid skew.
2. Fit PHATE or T-PHATE embeddings per epoch with `src/visualize_trajectories.py`; keep hyperparameters consistent to facilitate comparisons.
3. Colour embeddings by strategic features and overlay temporal trajectories to inspect flow through the manifold.

**Interpretation heuristics**:
- Distinct clusters often correspond to discrete attractors (attack, defend, neutral).
- Smooth gradients along the manifold suggest continuous accumulators (line attractors encoding evaluation).
- Loss of structure or collapsing embeddings signals representational failure or over-regularisation.

**Cross-references**: `theoretical_foundations.md` §4 for manifold learning theory; `case_studies.md` Cases 2 and 4 for empirical examples.

### 3. Probing and Mutual Information

**Narrative hook**: “Which game concepts are encoded, and are they accessible to simple decoders?”

**Goal**: quantify what information is encoded and whether it is linearly decodable.

**Inputs**: hidden-state matrices, feature label vectors, shuffled-label controls.

**Procedure**:
1. Train logistic or linear probes with stratified splits; report both accuracy and calibration metrics. Compare against shuffled-label baselines.
2. Run `scripts/compute_hidden_mutual_info.py` to compute aggregate and per-dimension MI; enable bootstrapping for confidence intervals.
3. Track probe accuracy and MI across checkpoints to study the emergence of specialised neurons.

**Interpretation heuristics**:
- Tactical features (immediate wins/blocks) should become linearly decodable early; strategic variables (center control, tempo) follow later.
- High probe accuracy with low MI implies distributed but linearly accessible codes; high MI with low probe accuracy implies nonlinear encodings.
- If shuffled controls mirror true scores, revisit sampling or label generation.

**Cross-references**: `theoretical_foundations.md` §§3 and 6 for estimator theory and learning dynamics; `case_studies.md` Case 3 for concept emergence timelines in AlphaZero.

### 4. Fixed Points and Attractor Evolution

**Narrative hook**: “Can we point to the latent states that embody tactics like attack or defend?”

**Goal**: map the computational skeleton of the GRU by cataloguing stable modes.

**Inputs**: canonical board states (attack, defence, neutral), frozen CNN feature vectors, optimisation loop for `min_h ||h - f(h)||^2`.

**Procedure**:
1. Generate and cache CNN embeddings for the chosen boards.
2. Follow the workflow in `theoretical_foundations.md` §2.5: initialise with replayed states, solve for fixed points via L-BFGS-B, deduplicate, and classify stability using the Jacobian.
3. Repeat across checkpoints to track movement, creation, or annihilation of attractors and saddles.

**Interpretation heuristics**:
- Expect a modest number of attractors (fewer than 20) corresponding to strategic archetypes.
- Stable attractors sharpening over training indicate consolidation of decision modes.
- Sudden attractor disappearance or proliferation of saddles often aligns with optimisation shocks.

**Status**: CLI support is in development (`scripts/find_fixed_points.py`). Use the pseudocode in `case_studies.md` until the script lands.

---

## Study Design Templates

These mini-guides bundle the playbooks into complete stories you can reproduce or extend.

- **Health check (approx. 2 hours)**  
  1. Compute gate/timescale statistics for best and final checkpoints.  
  2. Run MI for core tactical features.  
  3. Inspect a PHATE embedding for the final checkpoint.  
  4. Summarise findings alongside validation curves.

- **Learning-dynamics deep dive (1–2 days)**  
  1. Sample checkpoints every 5 epochs.  
  2. Run the full probe+MI suite and geometry embeddings per checkpoint.  
  3. Track gate statistics and spectral radii jointly.  
  4. (Optional) catalogue fixed points for three canonical boards.  
  5. Annotate training timelines with observed regime shifts.

- **Architecture comparison**  
  1. Repeat the deep dive for two architectures (e.g., GRU32 vs GRU128).  
  2. Use joint PHATE embeddings (`--viz-type ablation-gru`) to compare weight trajectories.  
  3. Quantify differences in timescale distributions and neuron specialisation.  
  4. Discuss trade-offs between capacity and interpretability.

---

## Failure Modes and Diagnostics

Each failure mode is phrased as a narrative symptom so you can recognise it quickly when scanning plots.

- **Memoryless behaviour**  
  - Symptoms: update gates stuck near zero, spectral radius much less than one, probes/MI near chance.  
  - Actions: increase hidden size, lengthen training sequences, adjust learning rate schedule.

- **Over-memorisation**  
  - Symptoms: update gates saturate near one, eigenvalues exceed the unit circle, validation accuracy deteriorates late.  
  - Actions: add regularisation (dropout, weight decay), clip gradients, consider smaller recurrent width.

- **Spurious probes**  
  - Symptoms: probes succeed but shuffled-label controls also perform well, MI remains low.  
  - Actions: rebalance data, verify labels, ensure temporal splits prevent leakage.

- **Representation collapse**  
  - Symptoms: PHATE embeddings contract to a point, hidden-state variance drops, MI declines.  
  - Actions: revisit optimiser hyperparameters, add auxiliary losses, diversify replay sampling.

- **Attractor instability** (when fixed-point tooling is active)  
  - Symptoms: attractors relocate dramatically, saddles proliferate, basins swap.  
  - Actions: inspect training logs for optimisation shocks, reduce learning rate during late training.

---

## Recommendations

- Monitor gate and timescale statistics throughout training; they are the quickest signal of emergent memory structure.
- Always pair probes with mutual information **and** shuffled controls to separate linear accessibility from encoding strength.
- Align geometry plots with fixed-point catalogues: PHATE clusters often indicate attractor candidates worth formal verification.
- Focus fixed-point sweeps on a curated set of board contexts and update the catalogue whenever you change architecture or training protocol.
- In Connect Four, moderate timescales with non-saturated gates consistently correlate with better generalisation than extreme memory regimes.

---

## Relationship to Other Documentation

- **Plot interpretation**: see the user manual sections on GRU observability and mutual information for figure-by-figure walkthroughs.
- **Theory**: `theoretical_foundations.md` covers Jacobians, MI estimators, manifold learning, and learning-dynamics theory.
- **Implementation**: `docs/reference/methods.md` documents CLI flags, data layouts, and library dependencies.
- **Workflows**: `docs/manual/workflows/gru_interpretability.md` chains these diagnostics into an end-to-end pipeline.

---

## Selected References

- Sussillo, D., and Barak, O. (2013). Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks. *Neural Computation*, 25(3), 626–649.
- Maheswaranathan, N., Williams, A. H., Golub, M. D., Ganguli, S., and Sussillo, D. (2019). Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics. *NeurIPS*.
- Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. (2019). Similarity of neural network representations revisited. *ICML*.
- Belinkov, Y. (2022). Probing classifiers: promises, shortcomings, and advances. *Computational Linguistics*.
