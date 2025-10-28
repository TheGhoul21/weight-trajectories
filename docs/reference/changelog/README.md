# Documentation Changelog

Major documentation updates and feature additions.

---

## Purpose

This directory tracks significant changes to the documentation and user-facing features. Each entry summarizes:
- What changed
- Files modified
- User impact
- Examples (when applicable)

---

## Changelog Entries

### [2025-10-28: Per-Dimension MI Analysis](./2025-10-28_mi_dimension_analysis.md)
**Summary**: Added neuron-level mutual information analysis to identify specialized feature detectors

**Key changes**:
- New plots: `mi_per_dimension_<model>.png` (heatmap with ★ markers)
- New plots: `mi_dimension_values_<model>.png` (violin/scatter for encoding quality)
- Extended `compute_hidden_mutual_info.py` with per-dimension MI functions
- Expanded `gru_mutual_info.md` from 22 → 248 lines

**User impact**:
- Can now identify which individual neurons encode which features
- Enables neuron ablation studies
- Supports architecture optimization (identify under/over-capacity)

**Documentation added**:
- Complete interpretation guide with 4 examples
- 4 diagnostic scenarios (debugging, architecture selection, neuron analysis)
- Feature categorization (5 classification + 7 regression)

---

## When to Add a Changelog Entry

**Document**:
- ✅ New analysis features (e.g., per-dimension MI, fixed-point finding)
- ✅ Breaking changes to command interfaces
- ✅ Major documentation restructuring (e.g., reorganizing into manual/scientific/reference)
- ✅ New plot types or output formats
- ✅ Significant algorithm changes (e.g., switching from t-SNE to PHATE default)

**Don't document**:
- ❌ Bug fixes (use git commit messages)
- ❌ Minor documentation typos
- ❌ Code refactoring without user-visible changes
- ❌ Internal implementation details

---

## Changelog Entry Template

```markdown
# YYYY-MM-DD: Feature Name

## Summary
Brief description of what changed and why.

## Changes

### Scripts Modified
- `scripts/foo.py`: Added X, modified Y
- `scripts/bar.py`: New functions for Z

### Documentation Updated
- `docs/manual/plots/foo.md`: Expanded from X → Y lines
- Added sections: algorithm details, interpretation examples

### New Outputs
- `visualizations/foo/new_plot.png`: Description
- `diagnostics/foo/new_metric.csv`: Columns and format

## User Impact

### New Capabilities
- Can now do X
- Enables Y analysis
- Supports Z use case

### Breaking Changes (if any)
- Old command `foo --bar` is now `foo --baz`
- Migration guide: ...

## Examples

### Basic Usage
\```bash
./wt.sh command --new-flag value
\```

### Advanced Usage
\```bash
# Multi-step workflow
./wt.sh step1
./wt.sh step2 --with-new-feature
\```

## Cross-References
- **Manual**: [link to relevant command/plot docs]
- **Scientific**: [link to theory if applicable]
- **Related changes**: [link to other changelog entries]
```

---

## Changelog Workflow

When adding a major feature:

1. **Implement** the feature (code + tests)
2. **Update user manual** (`docs/manual/`)
   - Command reference (if new command/flags)
   - Plot interpretation (if new outputs)
   - Workflow guides (if new analysis type)
3. **Add scientific context** (if novel technique)
   - `docs/scientific/` for theory/literature
4. **Create changelog entry** (`docs/reference/changelog/`)
   - Format: `YYYY-MM-DD_feature_name.md`
   - Use template above
5. **Update main READMEs**
   - `docs/README.md` - Add to "Common Tasks"
   - `docs/manual/README.md` - Update navigation
   - Directory READMEs - Add links

---

## Cross-References

- **Main docs**: [../../README.md](../../README.md)
- **Reference index**: [../README.md](../README.md)
- **User manual**: [../../manual/README.md](../../manual/README.md)
