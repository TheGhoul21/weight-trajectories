# Representation & Geometry Field Report

Once memory looks healthy, the next question is *what* the GRU encodes and
*where* those signals live in hidden space. This field report walks through the
concept-to-measurement pipeline: identify strategic concepts, probe for them,
map the geometry, and anchor every observation in the formal results covered in
`theoretical_foundations.md` §§3–4.

---

## Core Questions

1. **Concept coverage** – Which tactical and strategic features are represented
   in the hidden state?
2. **Accessibility** – Are those features linearly decodable, or do they require
   nonlinear readouts?
3. **Geometry** – Do hidden states organise into clusters, gradients, or slow
   manifolds that mirror gameplay structure?

---

## Concept Checklist

Start with the following feature categories. Add domain-specific concepts as
needed (e.g., opening templates, tempo heuristics):

- **Tactical**: immediate win/block, double threats, ladder opportunities.
- **Strategic**: centre control, piece parity, mobility.
- **Meta**: move index, current player, last-move column.
- **Outcome**: value head prediction, advantage score.

Label generation lives in `scripts/extract_gru_dynamics.py`; augment the script
if your feature set expands.

---

## Measurement Pipeline

### 1. Linear Probes with Controls

**Narrative hook**: “Which neurons light up when a forced win appears?”

- Split hidden states into train/validation folds (chronological split to avoid
  leakage). Standardise features.
- Train logistic or linear probes (`sklearn.linear_model.LogisticRegression` /
  `Ridge`) and compute accuracy, AUROC, and calibration error.
- Run the same probes on shuffled labels. If shuffled accuracy approaches the
  real one, revisit sampling or label computation.

**Interpretation**:
- High true accuracy + low shuffled accuracy → feature is linearly accessible.
- Low true accuracy + high mutual information (next step) → nonlinear code.
- High accuracy in both conditions → data leakage or unbalanced labels.

### 2. Mutual Information Sweeps

**Narrative hook**: “Is information concentrated in a few cells or distributed
across the population?”

- Use `scripts/compute_hidden_mutual_info.py` with bootstrap enabled.
- Inspect aggregate MI per feature and per-dimension MI heatmaps.
- Track MI over checkpoints to observe when concepts crystallise.

**Interpretation**:
- Sparse high-MI cells → specialised detectors (cf. §3 of
  `theoretical_foundations.md`).
- Diffuse MI → distributed representation; combine dimensions to decode.
- MI spikes aligned with validation jumps → concept emergence tied to skill.

### 3. Geometry Mapping with PHATE/T-PHATE

**Narrative hook**: “How do hidden states move as games unfold?”

- Embed hidden states per epoch using `src/visualize_trajectories.py
  --viz-type hidden`.
- Colour by feature labels (immediate win, move index, etc.).
- For temporal insight, overlay polylines for individual games or animate with
  `--viz-type animation` (optional).

**Interpretation**:
- Distinct clusters → point attractors for discrete strategies.
- Continuous ribbons → slow/line attractors encoding evaluation.
- Diffuse clouds with no structure → potential representation collapse (cross
  check with probes/MI).

---

## Formal Anchors

- **Mutual information (§3)** explains why per-dimension MI is a principled way
  to spot specialised neurons and how estimator choice affects reliability.
- **Manifolds (§4)** motivate PHATE/T-PHATE for preserving trajectory structure
  in both checkpoints and hidden states.
- **Learning dynamics (§6)** connect the timing of MI/probe improvements with
  attractor formation; watch for synchrony with gate/timescale changes from
  `guide_memory_diagnostics.md`.

---

## Reporting Template

When writing up results, answer these prompts:

- **Concept coverage**: “Immediate-win probes reached 0.94 AUROC by epoch 22;
  centre-control probes lagged until epoch 40.”
- **Accessibility**: “Mutual information for immediate-win concentrated in dims
  17 and 42 (0.82 bits each); centre control remained diffuse below 0.3 bits.”
- **Geometry**: “PHATE embeddings show three lobes aligned with attack/defend/
  neutral contexts; late epochs add a thin bridge corresponding to endgame
  scenarios.”
- **Next steps**: “Train probes on move-index-conditioned subsets to test
  whether concept encoding shifts across game phases.”

---

## Extending the Analysis

- **Nonlinear probes**: after the linear baseline, add shallow MLP probes to
  quantify the gap between linear and nonlinear accessibility.
- **Causal interventions**: zero out high-MI dimensions and re-run the policy
  to check whether gameplay changes (follow the methods in Case Study 3).
- **Phase-conditioned geometry**: embed hidden states separately for opening,
  midgame, and endgame to see how manifolds reconfigure.
- **Joint MI + geometry**: overlay per-dimension MI scores onto PHATE plots
  (colour nodes by MI) to connect information-theoretic and geometric views.
