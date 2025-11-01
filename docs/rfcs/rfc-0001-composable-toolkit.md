# RFC-0001: Composable Network Surgery Toolkit

Status: Draft  
Authors: maintainers  
Target Version: 0.2  
Discussion: https://github.com/TheGhoul21/weight-trajectories/issues/1

## Summary

Generalize the current ResNet+GRU Connect4 stack into a composable toolkit that allows users to assemble, inspect, and ablate multi-part networks. The toolkit standardizes tensor contracts, exposes taps for probes, and wraps regularization so that CNN, MLP, GRU, and future components can be mixed and matched without bespoke glue code.

## Motivation

- Broaden the library beyond vision-only encoders; many tasks provide sequences of vectors or tokens where CNNs are irrelevant.
- Enable "network surgery": plug or swap encoders, temporal cores, and heads; run ablations; and capture intermediate activations.
- Make probes and visualizations modality agnostic by standardizing where we tap models and how we name those activations.
- Reduce bespoke model definitions so future experiments can rely on configuration-first pipelines.

## Goals

- Encoder abstraction with `identity`, `cnn`, `mlp`, and `embedding` variants to cover common modalities.
- Sequential pipeline MVP: `[encoder -> projector -> temporal -> head]` with automatic shape reconciliation when safe.
- Optional DAG composition for advanced fusions (concat/add/residual) with explicit shape validation.
- Unified dataset adapter returning tensors and optional masks while supporting variable-length sequences.
- Probing API targeting named node outputs and time positions with opt-in activation capture.
- Regularization wrappers (dropout, spectral norm, activation penalties, gradient clipping) tied into configuration.

## Non-goals

- Reinforcement learning loop or environment support.
- Distributed training overhaul.
- Custom GRU cell or full recurrent dropout (deferred follow-ups).

## Design Overview

### Tensor Contract and Temporalization

- All nodes consume and emit tensors annotated by a `TensorSpec` stored in the pipeline context.
- Image inputs stay as `B x C x H x W`. When a temporal node follows an image-only path and `auto_temporalize` is enabled (default), a `Temporalize` adapter lifts tensors to `B x T x ...` with `T=1`.
- Mixed pipelines (e.g., video + static image) surface explicit adapter nodes such as `FrameEncoder` or `TemporalPad`; no silent reshaping occurs.
- Variable-length sequences carry `mask: B x T`. If `mask` is `None`, downstream nodes treat all positions as valid and lazily synthesize `mask_full` when needed so legacy code keeps working.

### Component Interface

```python
class Node(nn.Module):
    id: str
    port_type_in: str   # 'image' | 'sequence' | 'vector' | 'tokens'
    port_type_out: str
    output_dim: int | None

    def forward(self, x, mask=None, ctx=None):
        ...
        return y, mask, taps  # taps is dict[str, Tensor] | None
```

Minimal node types in the registry:

- Encoders: `IdentityEncoder`, `CNNEncoder`, `MLPEncoder`, `TokenEmbedding`.
- Adapters: `Projection`, `Flatten`, `Pool`, `Permute`, `Concat`, `Add`, `Residual`, `Repeat`, `Temporalize`.
- Temporal cores: `GRU`, `LSTM`, future `TransformerEncoder`, `TCN`.
- Heads: `Classifier`, `Regressor`, step vs sequence outputs.
- Regularizers/wrappers: `FeatureDropout`, `SpectralNorm`, `ActivityPenalty`.

### Pipeline Assembly Modes

1. **Sequential (Phase 1 MVP)**: Node specs composed left-to-right. Type and dimensionality checks run at build time, inserting automatic projections or temporal adapters when allowed.
2. **DAG (Phase 2)**: `nodes` and `edges` defined explicitly. Fusion adapters (`Concat`, `Add`, `Residual`) declare their inputs. DAG mode is strict unless configs opt-in to automatic projections.

The pipeline is built from a registry so CLI tooling can enumerate available components.

### PipelineContext (`ctx`)

- `ctx` is a mutable dict-like `PipelineContext` seeded with `mode`, `global_step`, `batch_idx`, `run_id`, and other run-level knobs (teacher forcing flags, temperatures, etc.).
- Reserved namespaces live under `ctx["pipeline"]`, `ctx["heads"]`, and `ctx["taps"]`. User-defined keys live under `ctx["user"]` to avoid collisions.
- Nodes may read and extend `ctx`, enabling non-adjacent coordination in DAGs.

### Outputs, Heads, and Loss Routing

- Each head node registers under a unique `name`. During the forward pass heads stash tensors into `ctx["heads"][name]`.
- `Pipeline.forward` returns `(outputs, mask, taps)` where `outputs` is always a `dict[str, Tensor]` (single-head models yield one entry).
- Loss configuration references head names, keeping multi-head routing explicit and config-first.

### Taps and Probes

- Node `taps` are optional dictionaries keyed by tap identifier. The pipeline aggregates taps into a single dict using node-qualified names (e.g., `gru.hidden`).
- Tap capture is opt-in. Configs supply the tap id list or glob patterns (e.g., `gru.*`) plus time selectors such as `[0, -1, "mid"]`.
- A `tap_sample_rate` throttles capture for long sequences. Large tap tensors stream to disk through the experiment logger to avoid RAM spikes.

### Shape Validation and Auto-Projection

- `Pipeline.validate()` runs a dry-forward with dummy tensors derived from each node's `TensorSpec`. Validation executes automatically on the first real forward pass (once). Setting `validate_on_init=true` forces validation during construction.
- When auto-projections or temporal adapters are inserted, structured warnings note the source and destination dimensions (e.g., `Auto-inserted projection 512->128 between cnn and gru`).
- A `strict_shapes` flag escalates those warnings to errors. Sequential pipelines default to auto-insertion; DAG pipelines require explicit `projection: auto`.

### Component Registry Tooling

- CLI helpers will surface registry metadata:
  - `python -m weight_trajectories components list`
  - `python -m weight_trajectories components info <name>`
- Nodes expose docstrings and configuration schema so users can discover ports, parameters, and taps.

### Regularization and Ablation

- Regularization knobs (dropout, weight decay, activity penalties, spectral norm, gradient clipping) sit alongside node specs in configuration.
- Phase 3 introduces ablations via both config and runtime APIs:
  - Config example:
    ```yaml
    ablations:
      - {node: encoder, method: zero}
      - {node: gru, method: noise, sigma: 0.05}
    ```
  - Runtime helper: `pipeline.ablate(node="encoder", method="zero")`.
- Initial ablation methods: `zero`, `noise` (Gaussian with configurable `sigma`), `shuffle` (batch or time axis), `freeze` (stop gradients), and `dropout` (custom probability). Methods subclass `AblationStrategy` and can be user-registered.

### Dataset Adapter

- Unified adapter returns `{"x": Tensor, "y": dict[str, Tensor], "mask": Optional[Tensor], "meta": dict}`.
- Image pipelines can return `B x T? x C x H x W`; vector pipelines use `B x T x F`; token pipelines use `B x T` (handled by `TokenEmbedding`).
- Collate functions maintain variable-length sequences through `mask` and optional packing utilities.

### CLI and Config Integration

- Existing `--cnn-channels`, `--gru-hidden`, and similar flags remain for 0.2 but raise deprecation warnings. `--config` selects new pipeline definitions.
- A `wt pipelines migrate old.yaml` helper upgrades shipped presets (ResNet+GRU, GRU-only, etc.) and common Hydra overrides, emitting TODO comments where manual intervention is required.
- Pipeline presets for legacy models are provided so command-line workflows remain one-flag-compatible.

## Phased Rollout

1. MVP sequential pipeline with identity encoder, auto-projection, param groups, and integration into the current training loop. Update docs with GRU-only and CNN+GRU examples.
2. DAG composition with fusion adapters and strict validation. Add targeted tests.
3. Probing taps integrated with visualization tooling; ablation utilities per node.
4. Dataset registry and adapters that cover vision, tokens, and tabular data with masking and packing verified across modes.
5. Regularization wrappers, stability options (LayerNorm GRU optional), and extended recipes.

## Open Questions and Resolutions

- **Auto-projection policy**: Sequential pipelines insert adapters automatically (with warnings). DAG pipelines require `projection: auto` to opt in.
- **Probe scheduling**: Config-driven with glob support and time selectors. Results stream to `artifacts/<run_id>/probes/` and can be returned to training hooks.
- **Multi-head losses**: Config-first routing; CLI shorthand is optional convenience work.
- **Validation timing**: First forward by default; `validate_on_init` for eager checks.
- **Context mutation**: Reserved namespaces documented; `ctx["user"]` left free for custom components.
- **Tap memory**: Capture is lazy and can stream to disk; `tap_sample_rate` throttles long sequences.

## Risks and Mitigations

- **Complexity creep**: Maintain a minimal required node set in Phase 1 and grow registry incrementally.
- **Migration friction**: Ship presets, migration tooling, and clear deprecation cadence (warn in 0.2, remove legacy flags in 0.3).
- **Performance overhead**: Default tap capture off; streaming avoids OOM during long-rollout experiments.

## Appendix A: PipelineContext Reserved Keys

```python
ctx = {
    "pipeline": {
        "mode": Literal["train", "eval", "inference"],
        "global_step": int | None,
        "batch_idx": int | None,
        "run_id": str | None,
        "auto_temporalize": bool,
        "strict_shapes": bool,
    },
    "heads": {
        # e.g., "policy": Tensor[B, T, 7]
    },
    "taps": {
        # node-qualified tap tensors; treated as read-only by nodes
    },
    "user": {
        # free-form namespace for custom nodes or experiments
    },
}
```

Appendix A will be referenced by docs describing custom node development.

## Appendix B: Migration Playbook

### Legacy CLI

```
python -m weight_trajectories.train \
  --cnn-channels 128 \
  --gru-hidden 256 \
  --kernel-size 3
```

### Pipeline Config

```yaml
connect4_resnet_gru:
  name: ResNet18 + GRU policy/value
  data: data/connect4/train.pt
  pipeline:
    - {type: cnn, backbone: resnet18, out: 512}
    - {type: projection, out: 256}
    - {type: gru, hidden: 256, n_layers: 2, dropout: 0.2, name: gru}
    - {type: head, name: policy, head_type: classifier, classes: 7, mode: step}
    - {type: head, name: value, head_type: regressor, out: 1, act: tanh, mode: step}
  train:
    batch_size: 128
    epochs: 50
    lr: 1e-3
    weight_decay: 1e-4
    grad_clip: {type: norm, max_norm: 1.0}
    penalties:
      - {type: l2_activation, target: gru.hidden, weight: 1e-5}
```

Run with:

```
python -m weight_trajectories.train --config configs/pipelines/connect4_resnet_gru.yaml
```

`wt pipelines migrate old.yaml` emits a scaffolded version of the new config, filling in defaults and flagging TODO sections for manual review.

## Appendix C: Custom Node Development Guide

```python
# src/components/time_average.py
from weight_trajectories.components import Node, register

@register("time_average")
class TimeAverage(Node):
    port_type_in = "sequence"
    port_type_out = "vector"

    def __init__(self, keep_dims=False):
        super().__init__()
        self.keep_dims = keep_dims

    def forward(self, x, mask=None, ctx=None):
        # x: B x T x F
        if mask is None:
            weights = x.new_ones(x.size(0), x.size(1), 1)
        else:
            weights = mask.unsqueeze(-1).float()
        summed = (x * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = summed / counts
        if self.keep_dims:
            mean = mean.unsqueeze(1)
        taps = {"time_average": mean}
        return mean, None, taps
```

- Register via `@register("time_average")` so configs can reference `type: time_average`.
- Declare `port_type_in/out` to participate in validation.
- Emit taps selectively so the pipeline can capture named activations when requested.
- Custom nodes can read run-level flags from `ctx["pipeline"]` or interact with other components through the `ctx["user"]` namespace.

This guide will expand with best practices as additional nodes land.

