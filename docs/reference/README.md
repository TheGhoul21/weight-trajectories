# Technical Reference

Architecture specifications, internal documentation, and technical guides.

---

## Contents

### [Methods Reference](methods.md)
Common algorithms and measures (CKA, MI, Grad‑CAM, PHATE/UMAP/t‑SNE/PCA) with links to libraries.

### [Architecture Diagrams](architecture_diagrams.md)
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

## Quick Links

### For Developers
- **Model Architecture**: [Architecture Diagrams](architecture_diagrams.md)

### For Documentation Maintainers
- **Update Workflow**:
  1. Implement feature
  2. Update user manual: [User Manual](../manual/)
  3. Add scientific context if novel: [Scientific Background](../scientific/)
  4. Reference shared methods: [Methods](methods.md)
  5. Update main README: [Documentation Home](../)

---

## Internal Notes

### File Organization
```
reference/
├── README.md
├── methods.md                  # Shared methods and libraries
└── architecture_diagrams.md    # Model visualization guides
```

## Cross-References

- **Main Documentation**: [Documentation Home](../)
- **User Manual**: [User Manual](../manual/)
- **Scientific Background**: [Scientific Background](../scientific/)
