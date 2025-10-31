# Scientific Field Guide

Picture yourself stepping into the lab after an overnight Connect Four training run. The user manual tells you which scripts to fire up; this field guide explains *why* those scripts matter, *what* to look for, and *how* the results tie back to the science of recurrent networks. Every page in `docs/scientific/` is designed to be read quickly, then revisited when you need deeper derivations.

---

## What Lives Here

- Rigorously stated concepts (fixed points, manifolds, mutual information) framed with practical motivation.
- Story-driven guides that walk from research questions → diagnostics → interpretation.
- Cross-links back to code (`scripts/…`, `src/…`) and forwards to open problems worth exploring.

The documents are intentionally short and interlinked—skim the story you need, then dive into the math section referenced in-line.

---

## Choose Your Path

1. **Conceptual grounding** → `theoretical_foundations.md`  
   Start with the short “Idea in Brief” boxes, then follow the deep dives if you need the derivations.
2. **Diagnostic playbooks** → `gru_observability_literature.md`  
   Walk through a narrative analysis of a training run, with checklists and failure-mode spotlights.
3. **Method primers** → `mutual_information_theory.md`, `weight_embeddings_theory.md`  
   Top-down: intuition first, formulas second, implementation links last.
4. **Worked stories from the literature** → `case_studies.md`  
   Each case reads like a mini documentary: task summary, what the researchers measured, and how to replicate the insight here.
5. **Citation pantry** → `references.md`  
   Grab DOIs, topic clusters, and one-line reminders of why each paper matters.

### Quick Menus

- **Just ran a model and need a sanity check?**  
  Read the first section of `gru_observability_literature.md` (“Diagnostic Modules at a Glance”), then follow the “Health Check” story.

- **Writing a talk or paper?**  
  Borrow language from §8 of `theoretical_foundations.md` and the highlighted citations at the bottom of this page.

- **Explaining the project to a newcomer?**  
  Point them to `case_studies.md` Case 1 or 2—fast narratives that show why attractors and manifolds matter.

---

## Document Index
| Document | Role in the field guide | What you gain | Key companion scripts |
| --- | --- | --- | --- |
| [`theoretical_foundations.md`](theoretical_foundations.md) | “Idea → deep dive” cards | Intuitive summaries followed by derivations for gates, attractors, MI, manifolds, neuroscience links | `src/analysis/eigs.py`, `scripts/find_fixed_points.py` (planned) |
| [`gru_observability_literature.md`](gru_observability_literature.md) | Investigation guide | Narrative walkthroughs, measurement checklists, failure diagnostics | `scripts/extract_gru_dynamics.py`, `scripts/compute_hidden_mutual_info.py` |
| [`mutual_information_theory.md`](mutual_information_theory.md) | MI primer | Conceptual overview, estimator options, bootstrapping tips | `scripts/compute_hidden_mutual_info.py` |
| [`weight_embeddings_theory.md`](weight_embeddings_theory.md) | Trajectory primer | PHATE/T-PHATE intuition, hyperparameter defaults, ablation recipes | `src/visualize_trajectories.py` |
| [`case_studies.md`](case_studies.md) | Story archive | Literature mini-narratives with implementation notes | varies; see each case for references |
| [`references.md`](references.md) | Citation pantry | Annotated bibliography by theme | — |

---

## Analysis Pillars

### Dynamical Systems & Attractors
**Story**: Watch a GRU stumble through the early epochs with no stable memories, then settle into a handful of attractors that match strategic modes.  
**Where to read**: `theoretical_foundations.md` §§1–2 for the mechanics; `gru_observability_literature.md` for the analysis playbook.

### Information Theory & Encoding
**Story**: Follow mutually-informative neurons emerging as the network learns to spot immediate wins before tackling strategy.  
**Where to read**: `theoretical_foundations.md` §3 for ideas, `mutual_information_theory.md` for estimator choices and experiment design.

### Representation Geometry
**Story**: Trace training checkpoints through PHATE projections to see phase changes and compare architectures side by side.  
**Where to read**: `theoretical_foundations.md` §4 plus `weight_embeddings_theory.md` for method specifics.

### Neuroscience Alignment & Learning Dynamics
**Story**: Compare GRU attractors to cortical motifs—point attractors for discrete strategies, slow manifolds for evaluation—and align their emergence with validation curves.  
**Where to read**: `theoretical_foundations.md` §§5–7 and Case Studies 2 & 4.

---

## Implementation Index

- `scripts/extract_gru_dynamics.py` → gate statistics, recurrent eigenvalues, MI-ready hidden state dumps.
- `scripts/compute_hidden_mutual_info.py` → aggregate/per-dimension MI, value-distribution plots, checkpoint comparisons.
- `src/visualize_trajectories.py` → PHATE/T-PHATE embeddings for weights, hidden states, and ablation comparisons.
- `scripts/run_linear_probes.py` (planned) → logistic probes with shuffled baselines for threat detection features.
- `scripts/find_fixed_points.py` (stub) → optimisation-based fixed-point finder with Jacobian classification (see §2.5 of `theoretical_foundations.md` for the algorithm sketch).

Each script deposits artefacts under `visualizations/` or `diagnostics/`; cross-reference the user manual for annotated figure walkthroughs.

---

## Research Extensions & Open Problems

- Automate fixed-point finding across canonical board contexts and track stability across epochs.
- Quantify basin volumes and align them with strategy labels extracted from gameplay analyzers.
- Extend MI analysis to multi-feature joint distributions (estimate synergy/redundancy).
- Introduce causal interventions (representation surgery) to test whether identified attractors drive policy shifts.

See `case_studies.md` for literature patterns worth replicating and `gru_observability_literature.md` for recommended next experiments.

---

## Relationship to Other Documentation

- **User Manual** (`docs/manual/`) → How to reproduce figures; best paired with the methodological background here.
- **Reference** (`docs/reference/`) → API-level details on scripts and configuration; use when wiring diagnostics into pipelines.
- **Scientific Handbook (this folder)** → Conceptual grounding, literature context, open research questions.

---

## References at a Glance

Representative sources anchoring the handbook:

- Sussillo & Barak (2013) — low-dimensional dynamics in RNNs.
- Moon et al. (2019) — PHATE for trajectory-preserving embeddings.
- Kraskov et al. (2004) — mutual information estimation.
- Khona & Fiete (2022) — attractor networks in neuroscience.
- McGrath et al. (2022) — concept emergence in AlphaZero.

Consult `references.md` for full citations, DOIs, and thematic groupings.
