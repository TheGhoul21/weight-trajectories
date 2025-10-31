# Memory & Timescale Field Report

You just trained a fresh GRU agent and want to know whether it remembers the
right things at the right times. This guide tells the story of how memory forms
inside the network—starting with high-level questions, then walking through the
measurements, and finally tying the evidence back to the underlying theory.

---

## Questions We Answer

1. **Retention** – Are update and reset gates coordinating to hold onto useful
   context without saturating?
2. **Timescale allocation** – How long do different hidden dimensions preserve
   information, and does that match the task demands?
3. **Stability** – Do eigenvalues stay inside the unit circle, signalling
   well-behaved dynamics, or do they drift toward instability?

Each question maps directly to diagnostics implemented in
`scripts/extract_gru_dynamics.py` and grounded in §1–2 of
`theoretical_foundations.md`.

---

## Quick Recipe

1. **Collect trajectories**  
   - Sample ~10k hidden states across games and timesteps (stratify by move
     index to avoid bias toward openings).
   - Save per-epoch gate activations with `scripts/extract_gru_dynamics.py`.

2. **Plot gate stories**  
   - Generate mean/variance curves for update and reset gates.
   - Watch for the classic arc: update gates rise from ~0.3 → 0.6 while reset
     gates hover in the mid-range.

3. **Inspect eigen spectra**  
   - Use the same script to log the recurrent Jacobian spectrum per epoch.
   - Convert eigenvalues to timescales with `τ = -1 / ln |λ|` (see
     §1.4 of `theoretical_foundations.md`).

4. **Report findings**  
   - Summarise gate trends, timescale distributions, and any stability warnings.
   - Link to plots stored under `visualizations/gru_dynamics/`.

---

## Reading the Evidence

### Gate Narratives

- **Healthy retention**: update gates climb gradually while reset gates avoid
  the extremes; the model learns to hold context and selectively clear it.
- **Memoryless regime**: both gates stay low; eigenvalues shrink toward zero,
  and the agent behaves like a feed-forward model.
- **Over-memorisation**: update gates saturate near one, reset gates floor
  near zero, and eigenvalues creep toward the unit circle—expect sluggish
  adaptation and potential overfitting.

### Timescale Allocation

- Plot timescale histograms and mark tactical versus strategic thresholds
  (e.g., one-move lookahead ≈ 1–2 steps, opening preparation ≈ 5–10 steps).
- Compare architectures: larger GRUs should allocate longer tails; if they
  do not, capacity might be wasted.
- Track how timescales evolve during training—new long-range modes emerging
  alongside validation jumps is a strong sign of learning.

### Stability Checks

- Spectral radius well below 1 → fast decay; verify that validation accuracy
  is still growing before concluding the model is underpowered.
- Spectral radius above 1 → potential oscillations; inspect gradients and
  consider clipping or reducing learning rate.
- Sudden radius spikes often coincide with optimisation shocks; examine
  learning rate schedules or curriculum changes.

---

## Linking Back to Theory

- **Discrete-time dynamics (§1)**: The Jacobian expression in
  `theoretical_foundations.md` explains how gates modulate eigenvalues. Use the
  derivation to justify why gate saturation collapses timescales.
- **Fixed points (§2)**: Stable attractors require eigenvalues inside the unit
  circle. Timescale analysis acts as an early warning before explicit
  fixed-point catalogues are built.
- **Learning dynamics (§6)**: Emergent timescales mirror the formation of
  attractor basins. Document when distinct modes appear to connect training
  curves with mechanistic change.

---

## Reporting Template

Use these prompts when writing up a run:

- **Summary**: “Update gates rose from 0.28 → 0.57 over training, while reset
  gates stabilised near 0.45.”
- **Timescales**: “Effective memory spans diversified from 1.3 steps to a
  bimodal distribution centred at 2.1 and 7.4 steps.”
- **Stability**: “Spectral radius peaked at 0.96 (epoch 34) but remained below
  1.0; no evidence of runaway dynamics.”
- **Action items**: “Consider light dropout to prevent late-epoch gate
  saturation; monitor for attractor emergence in follow-up runs.”

---

## Next Experiments

- Slice timescales by board context (opening, midgame, endgame) to see if the
  model adapts its memory budget to game phase.
- Run the same diagnostics on ablation runs (e.g., smaller GRU, different
  optimiser) and compare trajectories with `--viz-type joint`.
- Combine this guide with `guide_representation_diagnostics.md` to correlate
  long timescales with specific encoded features.
