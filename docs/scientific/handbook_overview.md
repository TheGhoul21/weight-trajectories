# Scientific Handbook Overview

The scientific handbook curates theory, methodology, and case studies that explain how neural networks learn and express computation. It serves students building intuition, practitioners running analyses, and researchers digging into open problems.

---

## Handbook Structure

```
docs/scientific/
├── 0_start_here/          # Audience-specific landing pages and glossary
├── 1_foundations/         # Mathematical and conceptual primers
├── 2_methods/             # Technique guides and implementation notes
├── 3_architectures/       # Architecture-aware interpretability playbooks
├── 4_tools_and_libraries/ # External tooling and integrations
├── 5_case_studies/        # Literature surveys and applied examples
├── 6_tutorials/           # Hands-on walkthroughs and notebooks
├── 7_advanced_topics/     # Emerging research themes
└── references/            # Bibliography and quick-reference indices
```

---

## Core Entry Points

- `0_start_here/README.md` introduces the handbook, highlights quick wins, and links to audience-specific guides.
- `0_start_here/for_beginners.md`, `for_practitioners.md`, and `for_researchers.md` tailor learning paths with curated readings.
- `0_start_here/glossary.md` defines more than eighty terms with cross-links into the main text.

---

## Foundational Primers

- `1_foundations/what_is_interpretability.md` frames the field, motivations, and common question types.
- `1_foundations/information_theory_primer.md`, `dynamical_systems_primer.md`, and `linear_algebra_essentials.md` supply mathematical background for the methods chapters.
- `1_foundations/statistics_for_interpretability.md` covers experimental design, estimation, and uncertainty quantification.

---

## Method Guides

- `2_methods/README.md` acts as a selection map that compares methods and flags prerequisites.
- In-depth chapters include linear probes, mutual information estimation, trajectory analysis, dimensionality reduction, representation similarity metrics, fixed-point workflows, and mechanistic interpretability tooling.
- Each guide pairs conceptual explanations with implementation checklists and references to scripts inside `src/` and `scripts/`.

---

## Architecture and Case Studies

- `3_architectures/` outlines interpretability strategies tailored to feedforward networks, CNNs, RNNs, transformers, and specialized models.
- `5_case_studies/` documents worked examples across domains—board games, NLP, vision, reinforcement learning, neuroscience, and multimodal systems—emphasizing reproducible setups and key findings.

---

## Reference Material

- `references/README.md` and `references/bibliography.md` present curated citation lists organized by theme.
- Supporting appendices track tools (`4_tools_and_libraries/`) and hands-on resources (`6_tutorials/`).

---

## Guiding Principles

1. **Audience layering**: every topic begins with intuition and links to deeper derivations.
2. **Method-context pairing**: theoretical chapters point to scripts, diagnostics, and visualizations that operationalize the ideas.
3. **Cross-referencing**: documents interlink heavily so readers can navigate from questions to code and back.
4. **Continuity**: the GRU-focused Connect Four study remains a first-class example while coexisting with broader interpretability coverage.

