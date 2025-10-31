# Neural Network Interpretability Handbook

The handbook collects foundational theory, practical techniques, and literature surveys that explain how neural networks learn and behave. Material spans introductory guides through advanced research topics while maintaining the detailed treatment of GRU dynamics that motivated the project.

---

## Quick Navigation

**New to interpretability?** Start with the [Beginner's Guide](0_start_here/for_beginners.md)

**Want practical recipes?** See the [Practitioner's Guide](0_start_here/for_practitioners.md)

**Conducting research?** Check the [Researcher's Guide](0_start_here/for_researchers.md)

**Main entry point**: [0_start_here/README.md](0_start_here/README.md)

---

## Handbook Coverage

The handbook covers:

- **All major architectures**: CNNs, RNNs, Transformers, MLPs
- **20+ interpretability techniques**: From Grad-CAM to circuit analysis
- **30+ case studies**: Vision, NLP, RL, neuroscience
- **Hands-on tutorials**: Interactive notebooks and code recipes
- **Theoretical foundations**: Information theory, dynamical systems, manifold learning
- **Tool guides**: Captum, SHAP, TransformerLens, and more

While maintaining:
- Rigorous treatment of RNN dynamics and fixed-point analysis
- Connections to neuroscience and dynamical systems theory
- Practical diagnostic workflows
- Deep literature integration

---

## Directory Structure

```
docs/scientific/
├── 0_start_here/          # Landing pages for different audiences
├── 1_foundations/         # Mathematical and conceptual foundations
├── 2_methods/             # Technique deep-dives
├── 3_architectures/       # Architecture-specific guides
├── 4_tools_and_libraries/ # Practical tool documentation
├── 5_case_studies/        # Real-world examples
├── 6_tutorials/           # Hands-on notebooks
├── 7_advanced_topics/     # Cutting-edge research
└── references/            # Bibliography and reading lists
```

---

## GRU Reference Hub

For readers concentrating on the Connect Four GRU analysis, the original deep-dive materials sit alongside the broader handbook:

1. **Conceptual grounding** → `theoretical_foundations.md` develops the dynamical-systems perspective that underpins the Connect Four study and informs [1_foundations/](1_foundations/).
2. **Diagnostic playbooks** → `gru_observability_literature.md` documents the complete Connect Four workflow with measurement checklists and failure modes.
3. **Method primers** → `mutual_information_theory.md` and `weight_embeddings_theory.md` provide mutual-information estimators and trajectory embedding guidance that complement the treatments in `2_methods/`.
4. **Case studies** → `case_studies.md` gathers narrative summaries that map directly to the domain-specific chapters in `5_case_studies/`.
5. **References** → `references.md` offers a concise bibliography; see [references/bibliography.md](references/bibliography.md) for the expanded catalog.

### Quick Menus

- **Just ran a model and need a sanity check?**  
  Read the first section of `gru_observability_literature.md` (“Diagnostic Modules at a Glance”), then follow the “Health Check” story.

- **Writing a talk or paper?**  
  Borrow language from §8 of `theoretical_foundations.md` and the highlighted citations at the bottom of this page.

- **Explaining the project to a newcomer?**  
  Point them to `case_studies.md` Case 1 or 2—fast narratives that show why attractors and manifolds matter.
- **Diagnosing memory vs overwrite behaviour?**  
  Follow `guide_memory_diagnostics.md` for a short field report with plots to reproduce.
- **Tracing how concepts surface in hidden space?**  
  Use `guide_representation_diagnostics.md` for probes, MI sweeps, and geometry maps.

---

## Document Index
| Document | Role in the field guide | What you gain | Key companion scripts |
| --- | --- | --- | --- |
| [`handbook_overview.md`](handbook_overview.md) | Structure map | Directory tour, entry points, and guiding principles | — |
| [`theoretical_foundations.md`](theoretical_foundations.md) | “Idea → deep dive” cards | Intuitive summaries followed by derivations for gates, attractors, MI, manifolds, neuroscience links | `src/analysis/eigs.py`, `scripts/find_fixed_points.py` (planned) |
| [`gru_observability_literature.md`](gru_observability_literature.md) | Investigation guide | Narrative walkthroughs, measurement checklists, failure diagnostics | `scripts/extract_gru_dynamics.py`, `scripts/compute_hidden_mutual_info.py` |
| [`guide_memory_diagnostics.md`](guide_memory_diagnostics.md) | Memory field report | Story-driven checklists for gates, eigenvalues, and timescales | `scripts/extract_gru_dynamics.py` |
| [`guide_representation_diagnostics.md`](guide_representation_diagnostics.md) | Representation field report | Concept coverage, MI sweeps, geometry interpretation | `scripts/compute_hidden_mutual_info.py`, `src/visualize_trajectories.py` |
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
