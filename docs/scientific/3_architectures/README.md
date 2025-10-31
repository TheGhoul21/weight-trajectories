# Architecture-Specific Guides

Interpretability considerations and best practices for different neural network architectures.

## Available Guides

- [Feedforward Networks](feedforward_networks.md) - MLPs, ResNets
- [Convolutional Networks](convolutional_networks.md) - CNNs, spatial feature hierarchies
- [Recurrent Networks](recurrent_networks.md) - RNNs, LSTMs, GRUs
- [Transformers](transformers.md) - Attention mechanisms, LLMs
- [Specialized Architectures](specialized_architectures.md) - GANs, VAEs, diffusion models

## Key Differences

Different architectures require different interpretability approaches:

**CNNs**: Spatial structure enables Grad-CAM; feature visualization reveals hierarchical detectors

**RNNs**: Temporal dynamics enable fixed-point analysis; memory mechanisms require special treatment

**Transformers**: Attention patterns are interpretable; mechanistic circuits analysis is mature

**MLPs**: Simplest case; most general techniques apply directly

Return to [main handbook](../0_start_here/README.md) | [Methods](../2_methods/)
