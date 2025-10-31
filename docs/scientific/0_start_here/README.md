# Neural Network Interpretability Handbook

Welcome to a comprehensive guide to understanding what neural networks learn and how they work. Whether you're new to interpretability or conducting cutting-edge research, this handbook provides the concepts, methods, and practical guidance you need.

---

## Choose Your Path

### 🌱 New to Interpretability?

**Start here if:**
- You're learning about neural network interpretability for the first time
- You want to understand core concepts without heavy prerequisites
- You're looking for intuitive explanations with visual examples

**→ [Beginner's Guide](for_beginners.md)** | [Glossary](glossary.md)

**What you'll learn:**
- What interpretability is and why it matters
- Main types of questions you can ask about neural networks
- Overview of techniques and when to use them
- Gentle introductions to key mathematical concepts

---

### 🔧 Want to Apply Techniques?

**Start here if:**
- You have a trained model and want to understand it
- You're looking for practical recipes and tool guides
- You need to debug model behavior or validate learning
- You want copy-paste code examples

**→ [Practitioner's Guide](for_practitioners.md)**

**What you'll find:**
- "I want to..." → technique decision flowcharts
- Tool comparison tables (Captum, SHAP, etc.)
- Quick-start code snippets for common tasks
- Integration patterns for ML pipelines
- Troubleshooting and debugging guides

**Jump directly to:**
- [Tools & Libraries](../4_tools_and_libraries/) - Practical tool guides
- [Tutorials](../6_tutorials/) - Hands-on notebooks and recipes
- [Architecture Guides](../3_architectures/) - CNN, Transformer, RNN-specific techniques

---

### 🔬 Conducting Research?

**Start here if:**
- You're developing new interpretability methods
- You're studying representation learning or training dynamics
- You want rigorous theoretical foundations
- You're exploring open research questions

**→ [Researcher's Guide](for_researchers.md)**

**What you'll find:**
- Current research landscape and open problems
- Rigorous mathematical foundations
- State-of-the-art techniques and recent advances
- Experimental design and methodological best practices
- Connections to neuroscience and learning theory

**Jump directly to:**
- [Advanced Topics](../7_advanced_topics/) - Superposition, scaling, emergence
- [Case Studies](../5_case_studies/) - 30+ detailed examples from literature
- [Methods](../2_methods/) - Deep dives into 20+ techniques

---

## What's Inside

### [1. Foundations](../1_foundations/)
Core concepts explained accessibly:
- Information theory (entropy, mutual information, KL divergence)
- Dynamical systems (fixed points, attractors, phase space)
- Linear algebra essentials (eigenvalues, SVD, projections)
- Statistical methods for interpretability

**Progressive disclosure**: Start with intuition, dive deeper as needed.

---

### [2. Methods](../2_methods/)
Comprehensive technique library organized by approach:

#### Feature Visualization
- **Gradient-based**: Saliency maps, Grad-CAM, Integrated Gradients
- **Activation maximization**: DeepDream, feature visualization
- **Attribution methods**: SHAP, LIME, Layer-wise Relevance Propagation

#### Probing & Information Theory
- **Linear probes**: Testing what's encoded and how
- **Mutual information**: Quantifying dependencies without linearity assumptions
- **Causal probing**: Intervention-based validation

#### Representation Analysis
- **Dimensionality reduction**: PCA, t-SNE, UMAP, PHATE comparison
- **Similarity metrics**: CKA, SVCCA, RSA for comparing representations
- **Geometry analysis**: Manifold structure and curvature

#### Dynamical Systems Analysis
- **Fixed points**: Finding and characterizing stable computational modes
- **Trajectory analysis**: Visualizing evolution in weight/hidden state space
- **Attractor landscapes**: Understanding computational structure

#### Mechanistic Interpretability
- **Circuits**: Decomposing networks into functional components
- **Sparse autoencoders**: Finding monosemantic features
- **Activation patching**: Causal interventions for validation

**Each method includes**: When to use it, assumptions, implementation notes, and code examples.

---

### [3. Architectures](../3_architectures/)
Architecture-specific considerations:

- **[Feedforward Networks](../3_architectures/feedforward_networks.md)**: MLPs, ResNets - layer-wise abstraction
- **[Convolutional Networks](../3_architectures/convolutional_networks.md)**: CNNs - spatial hierarchies, receptive fields
- **[Recurrent Networks](../3_architectures/recurrent_networks.md)**: RNNs, LSTMs, GRUs - memory and dynamics
- **[Transformers](../3_architectures/transformers.md)**: Attention patterns, layer specialization, mechanistic interp
- **[Specialized Architectures](../3_architectures/specialized_architectures.md)**: GANs, VAEs, diffusion models

**What you'll learn**: Which techniques work best for each architecture, common findings, and architecture-specific tools.

---

### [4. Tools & Libraries](../4_tools_and_libraries/)
Practical guides for popular interpretability tools:

- **Captum**: PyTorch interpretability suite (Grad-CAM, IG, feature ablation)
- **SHAP**: Shapley value-based explanations for any model
- **LIME**: Local interpretable model-agnostic explanations
- **TransformerLens**: Mechanistic interpretability for transformers
- **TensorBoard Projector**: Interactive embedding visualization
- **Custom tools**: This project's weight trajectory analysis toolkit

**Each guide includes**: Installation, quick start, common tasks, gotchas, and integration patterns.

---

### [5. Case Studies](../5_case_studies/)
Learn from 30+ real-world applications:

#### By Domain
- **[Computer Vision](../5_case_studies/computer_vision/)**: ImageNet features, adversarial examples, CLIP
- **[Natural Language](../5_case_studies/natural_language/)**: BERT probing, GPT-2 circuits, induction heads
- **[Reinforcement Learning](../5_case_studies/reinforcement_learning/)**: AlphaZero concepts, Atari representations
- **[Recurrent Networks](../5_case_studies/recurrent_networks/)**: Flip-flop attractors, grid cells, sentiment line attractors
- **[Board Games](../5_case_studies/board_games/)**: Connect Four GRU (this project), chess AlphaZero
- **[Neuroscience](../5_case_studies/neuroscience/)**: Motor cortex, hippocampal replay, working memory

**Each case study**: Problem context, methods applied, key findings, relevance, implementation notes.

---

### [6. Tutorials](../6_tutorials/)
Hands-on learning with code:

#### Interactive Notebooks
- Linear probes: Train, interpret, validate
- Grad-CAM: Generate and interpret heatmaps
- PHATE embeddings: Visualize training trajectories
- Mutual information: Find specialized neurons
- Fixed points: Discover attractor structure
- Attention patterns: Understand transformer heads
- Activation patching: Causal interventions

#### Recipes & Best Practices
- **[Visualization Recipes](../6_tutorials/visualization_recipes.md)**: Copy-paste plotting code
- **[Experimental Design](../6_tutorials/experimental_design.md)**: Controls, baselines, statistical testing

---

### [7. Advanced Topics](../7_advanced_topics/)
Cutting-edge research areas:

- **Scaling laws**: Interpretability at LLM scale
- **Emergence & phase transitions**: Grokking, capability jumps, double descent
- **Superposition**: Polysemantic neurons and sparse autoencoders
- **Mesa-optimization**: Inner alignment and deceptive behavior
- **Theoretical frameworks**: Information bottleneck, singular learning theory, causal abstraction

**For researchers** exploring frontiers of interpretability.

---

### [References](../references/)
Comprehensive bibliography:
- **[Bibliography](../references/bibliography.md)**: Full citations organized by topic
- **[Key Papers](../references/key_papers.md)**: Must-read papers with summaries
- **[Reading Lists](../references/reading_lists.md)**: Curated paths through the literature

---

## Quick Navigation

### By Goal

**"I want to understand why my model made a specific prediction"**
→ [Feature Visualization](../2_methods/feature_visualization/) (Grad-CAM, Integrated Gradients)

**"I want to know what patterns my model learned"**
→ [Probing](../2_methods/probing/) + [Representation Analysis](../2_methods/representation_analysis/)

**"I want to compare two models or training runs"**
→ [Similarity Metrics](../2_methods/representation_analysis/similarity_metrics.md) + [Trajectory Analysis](../2_methods/dynamical_analysis/trajectory_analysis.md)

**"I want to find interpretable features or neurons"**
→ [Mutual Information](../2_methods/probing/mutual_information.md) + [Mechanistic Interpretability](../2_methods/mechanistic_interpretability/)

**"I want to understand training dynamics"**
→ [Trajectory Analysis](../2_methods/dynamical_analysis/trajectory_analysis.md) + [Weight Embeddings Theory](../weight_embeddings_theory.md)

**"I want to understand recurrent network computation"**
→ [Fixed Points](../2_methods/dynamical_analysis/fixed_points.md) + [Attractor Landscapes](../2_methods/dynamical_analysis/attractor_landscapes.md)

### By Architecture

- **CNNs**: Start with [Grad-CAM tutorial](../6_tutorials/notebooks/) → [CNN Guide](../3_architectures/convolutional_networks.md)
- **Transformers**: Start with [Attention Viz](../6_tutorials/notebooks/) → [Transformer Guide](../3_architectures/transformers.md)
- **RNNs/LSTMs/GRUs**: Start with [Fixed Points](../2_methods/dynamical_analysis/fixed_points.md) → [RNN Guide](../3_architectures/recurrent_networks.md)

### By Experience Level

- **Beginner**: [For Beginners](for_beginners.md) → [Glossary](glossary.md) → [Foundations](../1_foundations/)
- **Intermediate**: [For Practitioners](for_practitioners.md) → [Tutorials](../6_tutorials/) → [Methods](../2_methods/)
- **Advanced**: [For Researchers](for_researchers.md) → [Advanced Topics](../7_advanced_topics/) → [Case Studies](../5_case_studies/)

---

## About This Handbook

### Philosophy

This handbook is designed with three principles:

1. **Accessibility**: Clear entry points for all experience levels
2. **Practicality**: Actionable guidance with code examples
3. **Rigor**: Maintains theoretical depth for serious research

We balance breadth (covering many architectures and methods) with depth (detailed explanations where needed) through progressive disclosure and cross-referencing.

### Origins

This handbook grew from the [Weight Trajectories](https://github.com/lucasimonetti/weight-trajectories) project, which analyzes GRU training dynamics through dynamical systems theory. The core insights about fixed points, attractors, and trajectory analysis originated from studying Connect Four-playing agents.

We've expanded to cover the broader interpretability landscape while maintaining the rigorous, theory-grounded approach.

### How to Use

- **Browse by interest**: Use the navigation above to find relevant topics
- **Follow learning paths**: Each audience guide provides a recommended sequence
- **Cross-reference freely**: Documents are densely linked - follow your curiosity
- **Start simple, go deep**: Most topics have gentle introductions linking to rigorous treatments

### Contributing

This is a living document. Areas for expansion:
- More case studies from recent papers
- Additional architecture-specific guides
- Interactive visualizations
- Tutorial notebooks for emerging techniques

---

## Getting Started

**Not sure where to start?** Try this:

1. Read the appropriate guide for your experience level ([Beginners](for_beginners.md) | [Practitioners](for_practitioners.md) | [Researchers](for_researchers.md))
2. Explore the [Glossary](glossary.md) to familiarize yourself with key terms
3. Pick a topic from [Methods](../2_methods/) or [Case Studies](../5_case_studies/) that interests you
4. Try a hands-on [Tutorial](../6_tutorials/) to solidify understanding

**Questions or suggestions?** Check the [main project README](../../../../README.md) for contact information.

---

## Table of Contents

For a complete file listing, see:
- [Foundations](../1_foundations/README.md)
- [Methods](../2_methods/README.md)
- [Architectures](../3_architectures/README.md)
- [Tools](../4_tools_and_libraries/README.md)
- [Case Studies](../5_case_studies/README.md)
- [Tutorials](../6_tutorials/README.md)
- [Advanced Topics](../7_advanced_topics/README.md)
- [References](../references/README.md)

---

**Happy exploring!** 🔍
