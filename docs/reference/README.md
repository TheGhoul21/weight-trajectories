# Technical Reference

Architecture specifications, internal documentation, and technical guides.

---

## Contents

### [Architecture Diagrams](./architecture_diagrams.md)
How to generate publication-ready diagrams of the ResNet+GRU model.

**Tools covered**:
1. **PlotNeuralNet** (LaTeX/TikZ)
   - 3D block diagrams common in deep learning papers
   - Requires LaTeX toolchain (texlive/mactex)
   - Full customization of layer sizes, colors, spacing

2. **Netron** (Interactive viewer)
   - Fast visualization for ONNX models
   - Good for quick architecture verification
   - Limited customization but zero setup

**Use this for**:
- Creating Figure 1 (Architecture) for papers
- Verifying model structure during development
- Presentations and documentation

---

### [Changelog](./changelog/)
Internal documentation of major feature additions and updates.

**Recent updates**:
- **[2025-10-28: Per-Dimension MI Analysis](./changelog/2025-10-28_mi_dimension_analysis.md)**
  - Added neuron-level mutual information analysis
  - New plots: `mi_per_dimension_*.png`, `mi_dimension_values_*.png`
  - Updated manual with comprehensive interpretation guides
  - 342-line workflow guide created

**Future entries**:
- Add new markdown files here when making significant documentation updates
- Format: `YYYY-MM-DD_feature_name.md`
- Include: what changed, files modified, user impact

---

## Quick Links

### For Developers
- **Model Architecture**: [`architecture_diagrams.md`](./architecture_diagrams.md)
- **Recent Changes**: [`changelog/`](./changelog/)

### For Documentation Maintainers
- **Update Workflow**:
  1. Implement feature
  2. Update user manual ([`../manual/`](../manual/))
  3. Add scientific context if novel ([`../scientific/`](../scientific/))
  4. Document changes ([`changelog/`](./changelog/))
  5. Update main README ([`../README.md`](../README.md))

---

## Internal Notes

### File Organization
```
reference/
├── README.md (this file)
├── architecture_diagrams.md    # Model visualization guides
└── changelog/                  # Major update summaries
    └── 2025-10-28_mi_dimension_analysis.md
```

### When to Add to Changelog
Document major changes that affect users:
- ✅ New analysis features (e.g., per-dimension MI)
- ✅ Breaking changes to command interfaces
- ✅ Significant documentation restructuring
- ❌ Bug fixes (use git commit messages)
- ❌ Minor doc typos (just fix directly)

---

## Cross-References

- **Main Documentation**: [`../README.md`](../README.md)
- **User Manual**: [`../manual/`](../manual/)
- **Scientific Background**: [`../scientific/`](../scientific/)
