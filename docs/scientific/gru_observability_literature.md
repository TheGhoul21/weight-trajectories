# GRU Observability: Intuition, Methods, and Recommendations

Provides a top‑down scientific background for analyzing GRU dynamics in Connect Four. Starts with the motivation and questions, then introduces methods at a high level, and finally offers concrete recommendations and pitfalls. Detailed algorithms and command‑level documentation are covered in Plot Interpretation and Theoretical Foundations.

## Why Study GRU Dynamics

Training a ResNet+GRU policy/value network induces temporal computation: the GRU must decide what to remember, when to overwrite, and how to transform input into actionable state. Observability aims to answer three questions:

- Memory: how long does the network retain information and when does it choose to keep vs overwrite?
- Representation: which board variables are encoded and how disentangled are they?
- Mechanism: does learning organize into discrete modes (attractors) corresponding to strategies?

## High‑Level Intuition

- Gates as policy over memory: the update gate controls retention; the reset gate controls access to past state when forming new content. Stable training typically shows moderate increases in the update gate over epochs (more selective retention) without saturating.
- Timescales as memory depth: eigenvalue magnitudes of the recurrent transformation imply how quickly information decays. Larger hidden sizes generally allow longer memory, but “longer” is not always better for generalization.
- Geometry reveals variables: embeddings of hidden states colored by interpretable board features indicate whether learned representations align with strategic factors (threats, turn, phase).
- Probing tests accessibility: linear probes quantify whether task‑relevant variables are linearly decodable. Rising accuracy over epochs indicates progressive alignment of representation with the task.
- Attractors as computational modes: stable fixed points correspond to persistent internal states (e.g., defend, attack, neutral). Their emergence and stabilization track the formation of strategy.

## Methods (Top‑Down Summary)

1) Gate and Timescale Trajectories
- What: track mean/reset gate statistics and timescale summaries over training.
- Why: indicates if the model learns to retain context and whether memory depth matches task demands.
- Outputs: gate mean trajectories, timescale heatmaps.
- Where: Plot Interpretation → GRU Observability.

2) Hidden‑State Geometry
- What: embed sampled hidden states (per epoch) with PHATE and color by board features.
- Why: visual check that representations align with interpretable variables (threats, turn, move phase).
- Outputs: per‑epoch embedding figures by feature.
- Where: Plot Interpretation → GRU Observability.

3) Linear Probing with Controls
- What: logistic probes on hidden states for binary features with permuted‑label controls.
- Why: quantifies linear accessibility and validates signal over chance.
- Outputs: probe accuracy curves and signal‑over‑control plots.
- Where: Plot Interpretation → GRU Observability; Plot Interpretation → GRU Mutual Information for complementary analysis.

4) Fixed Points and Attractor Evolution
- What: identify fixed points for selected input contexts; classify stability via Jacobian spectrum; track over epochs.
- Why: connects learned computation to dynamical mechanisms (stable modes, decision boundaries).
- Outputs: counts by stability class, spectral radii over epochs, attractor positions.
- Where: Plot Interpretation → Fixed Points; Theoretical Foundations for derivations.

5) Mutual Information (Per‑Feature and Per‑Dimension)
- What: quantify MI between hidden units and board features over training.
- Why: reveals emergence of specialized neurons and timing of representation alignment.
- Outputs: MI heatmaps, per‑dimension MI distributions.
- Where: Plot Interpretation → GRU Mutual Information; Scientific Background → Mutual Information Theory.

## What Good Looks Like (Patterns to Expect)

- Gates: gradual increase of update gate into a mid‑high regime; reset gate stable in mid range; no pervasive saturation.
- Timescales: increase with hidden size; moderate values often generalize better than extreme long memory.
- Probing: early gains on immediate‑threat variables; later gains on strategic variables (e.g., three‑in‑a‑row, center control).
- Geometry: clearer separation by task‑relevant features as training proceeds; less structure for irrelevant ones.
- Attractors: few stable modes emerge and sharpen with training; stabilization aligns with validation improvements.

## Common Failure Modes and Diagnostics

- Memoryless behavior: low update gate, short timescales, near‑chance probes → increase GRU size or adjust training.
- Over‑memorization: saturated gates, extremely long timescales, declining probe accuracy late in training → regularize, consider early stopping, reduce capacity.
- Spurious probes: high probe accuracy but high control accuracy → data leakage or imbalance; revise sampling and splits.
- Representation collapse: decreasing representation variance with flat or worsening validation → capacity misalignment or overfitting; adjust architecture or training schedule.

## Recommendations

- Use gate/timescale trajectories for quick health checks across architectures.
- Validate representation claims with probes and controls; complement with MI for neuron‑level insights.
- Apply fixed‑point analysis on a small set of canonical board contexts and track evolution every few epochs to link mechanisms with performance.
- Prefer moderate timescales and non‑saturated gates for generalization in Connect Four; larger hidden sizes can help but may overfit without regularization.

## Relationship to the Rest of the Documentation

- Practical usage and plot details: Plot Interpretation → GRU Observability, GRU Mutual Information, Fixed Points.
- Underlying theory and derivations: Scientific Background → Theoretical Foundations, Mutual Information Theory, Weight Embeddings Theory.
- End‑to‑end workflows: User Manual → Workflows → GRU Interpretability.

## Selected References

- Sussillo, D., & Barak, O. (2013). Opening the black box: low‑dimensional dynamics in high‑dimensional recurrent neural networks. Neural Computation, 25(3):626–649.
- Maheswaranathan, N., et al. (2019). Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics. NeurIPS.
- Kornblith, S., et al. (2019). Similarity of neural network representations revisited. ICML.
- Belinkov, Y. (2022). Probing classifiers: promises, shortcomings, and advances. Computational Linguistics.
