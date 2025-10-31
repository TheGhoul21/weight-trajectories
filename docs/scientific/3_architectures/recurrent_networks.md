# Recurrent Networks: RNNs, LSTMs, and GRUs

Architecture-specific interpretability guide for recurrent neural networks.

---

## Overview

Recurrent neural networks maintain hidden state across time steps, enabling them to process sequences and maintain memory. This temporal structure creates unique interpretability opportunities and challenges.

**Key architectures**:
- Vanilla RNNs (simple recurrence)
- LSTMs (Long Short-Term Memory)
- GRUs (Gated Recurrent Units)

---

## Architecture Fundamentals

### Vanilla RNN

**Update rule**:
```
h_t = tanh(W_h h_{t-1} + W_x x_t + b)
```

**Characteristics**:
- Simple, fully connected recurrence
- Prone to vanishing/exploding gradients
- Short effective memory

**When used**: Simple tasks, research/analysis

### LSTM

**Update rules**:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # input gate
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # output gate
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c) # candidate cell
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t        # cell state
h_t = o_t ⊙ tanh(c_t)                  # hidden state
```

**Characteristics**:
- Separate cell state and hidden state
- Three gates control information flow
- Handles long-range dependencies
- More parameters than vanilla RNN

**When used**: NLP, time series, any long sequences

### GRU

**Update rules**:
```
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)          # update gate
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)          # reset gate
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h) # candidate
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t        # hidden state
```

**Characteristics**:
- Two gates (simpler than LSTM)
- Single hidden state (no separate cell)
- Comparable performance to LSTM
- Fewer parameters

**When used**: Alternative to LSTM, faster training

---

## Interpretability Techniques for RNNs

### 1. Dynamical Systems Analysis

**Why it works**: RNNs are discrete-time dynamical systems

**Key methods**:
- [Fixed-point analysis](../2_methods/dynamical_analysis/fixed_points.md): Find stable computational modes
- [Trajectory visualization](../2_methods/dynamical_analysis/trajectory_analysis.md): Visualize evolution through phase space
- [Attractor landscapes](../2_methods/dynamical_analysis/attractor_landscapes.md): Map computational structure

**Best for**: Understanding computation, memory, state transitions

**Example workflow**:
```python
# 1. Find fixed points for key contexts
fps = find_fixed_points(gru, input_context='attack_position')

# 2. Classify stability
for fp in fps:
    jacobian = compute_jacobian(gru, fp, input_context)
    stability = classify_stability(jacobian)
    print(f"Fixed point: {stability}")

# 3. Visualize landscape
visualize_attractor_landscape(fps, trajectories)
```

**Strengths**: Reveals computational structure, grounded in theory

**Limitations**: Works best with autonomous/slowly-changing input

### 2. Gate Analysis

**Why it works**: Gates control information flow in LSTMs/GRUs

**What to analyze**:
- Gate activation distributions
- Temporal patterns (when do gates open/close?)
- Correlation with task phases

**Example**:
```python
def analyze_gates(gru, dataloader):
    """Analyze GRU gate statistics."""
    update_gates = []
    reset_gates = []

    for batch_x, _ in dataloader:
        with torch.no_grad():
            h = torch.zeros(batch_x.size(0), gru.hidden_size)

            for t in range(batch_x.size(1)):
                # Extract gates (requires modifying GRU to return them)
                h, z_t, r_t = gru.forward_with_gates(batch_x[:, t], h)

                update_gates.append(z_t.mean().item())
                reset_gates.append(r_t.mean().item())

    # Visualize
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(update_gates, bins=50)
    plt.title('Update Gate Distribution')
    plt.xlabel('Gate Activation')

    plt.subplot(1, 2, 2)
    plt.hist(reset_gates, bins=50)
    plt.title('Reset Gate Distribution')

    plt.tight_layout()
    plt.show()
```

**Interpretation**:
- **Update gate near 0**: Holding previous state (long-term memory)
- **Update gate near 1**: Accepting new input (updating)
- **Reset gate near 0**: Forgetting past
- **Reset gate near 1**: Incorporating past

**Strengths**: Direct insight into memory mechanisms

**Limitations**: LSTM/GRU specific

### 3. Hidden State Probing

**Why it works**: Hidden states encode task-relevant information

**Methods**:
- [Linear probes](../2_methods/probing/linear_probes.md): Test concept encoding
- [Mutual information](../2_methods/probing/mutual_information.md): Quantify dependencies

**Example**:
```python
# Extract hidden states from trained RNN
def extract_hidden_states(rnn, dataloader):
    """Extract hidden states at each timestep."""
    all_hidden = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            h = torch.zeros(batch_x.size(0), rnn.hidden_size)

            for t in range(batch_x.size(1)):
                h = rnn(batch_x[:, t], h)
                all_hidden.append(h.numpy())
                all_labels.append(batch_y.numpy())

    return np.vstack(all_hidden), np.hstack(all_labels)

# Probe for concept
hidden_states, labels = extract_hidden_states(rnn, val_loader)
acc, probe, weights = train_linear_probe(hidden_states, labels)

print(f"Probe accuracy: {acc:.3f}")
```

**Strengths**: Tests specific hypotheses, quantitative

**Limitations**: Correlational (use causal probing for validation)

### 4. Trajectory Embeddings

**Why it works**: Training/processing creates trajectories through state/weight space

**Methods**:
- PHATE: Trajectory-preserving dimensionality reduction
- PCA: Linear projection
- CKA: Compare representations across checkpoints

**Example**:
```python
import phate

# Collect hidden states over time
trajectory = []
h = torch.zeros(1, rnn.hidden_size)

for x_t in input_sequence:
    with torch.no_grad():
        h = rnn(x_t, h)
    trajectory.append(h.numpy())

trajectory = np.vstack(trajectory)

# Embed with PHATE
phate_op = phate.PHATE(n_components=2)
embedded = phate_op.fit_transform(trajectory)

# Visualize
plt.plot(embedded[:, 0], embedded[:, 1], '-o')
plt.title('Hidden State Trajectory')
plt.show()
```

**Strengths**: Visual, reveals global structure

**Limitations**: 2D projection loses information

### 5. Attention Mechanisms (if present)

**For architectures with attention**:
- Visualize attention weights
- Identify which parts of input matter
- Track attention patterns over time

**Not applicable** to basic RNN/LSTM/GRU (no attention)

---

## Common Analysis Workflows

### Workflow 1: Understanding a Trained RNN

**Goal**: Reverse-engineer what the network learned

**Steps**:

1. **Extract hidden states** on validation set
   ```python
   hidden_states = extract_all_hidden_states(rnn, val_loader)
   ```

2. **Visualize with PHATE**
   ```python
   embedded = phate.PHATE().fit_transform(hidden_states)
   plt.scatter(embedded[:, 0], embedded[:, 1], c=labels)
   ```

3. **Probe for concepts**
   ```python
   for concept in ['threat_detected', 'game_phase', 'position_eval']:
       acc, _, _ = train_linear_probe(hidden_states, concept_labels[concept])
       print(f"{concept}: {acc:.3f}")
   ```

4. **Find fixed points** for key contexts
   ```python
   fps = find_fixed_points(rnn, important_contexts)
   visualize_fps(fps)
   ```

5. **Analyze gates** (if GRU/LSTM)
   ```python
   analyze_gate_statistics(rnn, val_loader)
   ```

### Workflow 2: Debugging Poor Performance

**Goal**: Identify why the RNN fails

**Steps**:

1. **Check gate saturation**
   ```python
   # Gates stuck at 0 or 1 indicate problems
   gate_stats = compute_gate_statistics(rnn, train_loader)
   if gate_stats['update_gate_mean'] < 0.1:
       print("Warning: Update gates saturated at 0 (not updating)")
   ```

2. **Analyze gradient flow**
   ```python
   # Check for vanishing/exploding gradients
   grad_norms = []
   for batch in train_loader:
       loss = compute_loss(rnn, batch)
       loss.backward()
       grad_norms.append(torch.nn.utils.clip_grad_norm_(rnn.parameters(), float('inf')))

   plt.plot(grad_norms)
   plt.yscale('log')
   plt.title('Gradient Norms')
   ```

3. **Probe for expected concepts**
   ```python
   # Does it encode what it should?
   for concept in expected_concepts:
       acc = probe_accuracy(rnn, val_loader, concept)
       if acc < 0.6:
           print(f"Warning: {concept} not well encoded")
   ```

4. **Compare to baseline**
   ```python
   # Compare hidden state structure to random/untrained RNN
   cka_score = compute_cka(trained_rnn, random_rnn, val_loader)
   if cka_score > 0.5:
       print("Warning: Too similar to random initialization")
   ```

### Workflow 3: Tracking Training Dynamics

**Goal**: Understand how learning progresses

**Steps**:

1. **Save checkpoints regularly**
   ```python
   # During training
   if epoch % 5 == 0:
       torch.save(rnn.state_dict(), f'ckpt_epoch_{epoch}.pt')
   ```

2. **Track probe accuracy over epochs**
   ```python
   for ckpt in checkpoints:
       rnn.load_state_dict(torch.load(ckpt))
       acc = probe_accuracy(rnn, val_loader, concept)
       accuracies.append(acc)

   plt.plot(epochs, accuracies)
   plt.xlabel('Epoch')
   plt.ylabel('Probe Accuracy')
   ```

3. **Visualize weight trajectories**
   ```python
   # Extract weights from all checkpoints
   weight_trajectories = []
   for ckpt in checkpoints:
       state_dict = torch.load(ckpt)
       weights = flatten_weights(state_dict)
       weight_trajectories.append(weights)

   # Embed with PHATE
   embedded = phate.PHATE().fit_transform(weight_trajectories)
   plt.plot(embedded[:, 0], embedded[:, 1], '-o')
   ```

4. **Track fixed-point evolution**
   ```python
   for ckpt in checkpoints:
       rnn.load_state_dict(torch.load(ckpt))
       fps = find_fixed_points(rnn, context)
       print(f"Epoch {epoch}: {len(fps)} fixed points")
   ```

---

## Architecture-Specific Considerations

### Vanilla RNN

**Advantages for interpretability**:
- Simple structure
- Easier to analyze mathematically
- Clear dynamical system formulation

**Challenges**:
- Limited memory
- Vanishing gradients
- Less practical for real applications

**Best practices**:
- Use for research/educational purposes
- Good test bed for new methods
- Start here before analyzing complex models

### LSTM

**Advantages for interpretability**:
- Separate cell and hidden states
- Gate analysis reveals memory mechanisms
- Widely studied

**Challenges**:
- More complex (4 gates)
- Cell state less directly interpreted
- Many parameters

**Best practices**:
- Analyze both cell and hidden states
- Track gate activations over time
- Consider ablating individual gates

**LSTM-specific analysis**:
```python
def analyze_lstm_cell_state(lstm, input_sequence):
    """Analyze LSTM cell state evolution."""
    h = torch.zeros(1, lstm.hidden_size)
    c = torch.zeros(1, lstm.hidden_size)

    cell_trajectory = []
    hidden_trajectory = []

    for x_t in input_sequence:
        with torch.no_grad():
            h, c = lstm(x_t, (h, c))
        cell_trajectory.append(c.numpy())
        hidden_trajectory.append(h.numpy())

    # Compare cell vs hidden
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(np.array(cell_trajectory).mean(axis=1))
    plt.title('Cell State Evolution')

    plt.subplot(1, 2, 2)
    plt.plot(np.array(hidden_trajectory).mean(axis=1))
    plt.title('Hidden State Evolution')

    plt.tight_layout()
```

### GRU

**Advantages for interpretability**:
- Simpler than LSTM (2 gates)
- Single hidden state
- Good balance of capability and simplicity

**Challenges**:
- Less studied than LSTM
- Gates still complex

**Best practices**:
- Focus on update/reset gate patterns
- Compare to LSTM when possible
- Good for dynamical systems analysis

**GRU-specific analysis** (from this project):
- Gate statistics: [GRU Observability](../../5_case_studies/board_games/connect_four_gru.md)
- Fixed points: Works particularly well
- Trajectory analysis: Natural fit for PHATE

---

## Common Findings

### Memory Mechanisms

**Short-term memory** (vanilla RNN):
- Decays exponentially with time
- Eigenvalues of recurrent matrix control decay rate

**Long-term memory** (LSTM/GRU):
- Gates can maintain state indefinitely
- Update gate near 0 → long memory
- Forget gate (LSTM) controls retention

### Hierarchical Representations

**Typical pattern** (multi-layer RNNs):
- **Layer 1**: Low-level features (characters, phonemes)
- **Layer 2**: Mid-level patterns (words, phrases)
- **Layer 3+**: High-level semantics (sentences, meaning)

**Validation**:
```python
# Probe each layer
for layer_idx in range(num_layers):
    hidden = extract_layer_hidden_states(rnn, val_loader, layer_idx)

    for concept in ['syntax', 'semantics']:
        acc = train_linear_probe(hidden, concept_labels[concept])
        print(f"Layer {layer_idx}, {concept}: {acc:.3f}")
```

### Phase Transitions

**Observation**: RNN behavior changes qualitatively during sequence

**Example - Sentiment analysis**:
- Early timesteps: Neutral state
- Middle: Accumulating evidence
- Late: Committed to classification

**Analysis**:
```python
# Track hidden state clustering over time
for t in range(seq_length):
    hidden_at_t = hidden_states_all_sequences[:, t, :]
    n_clusters = estimate_clusters(hidden_at_t)
    print(f"Timestep {t}: {n_clusters} clusters")
```

---

## Comparison with Other Architectures

### vs Transformers

**RNN advantages**:
- Natural for sequential processing
- Maintains compact state
- Dynamical systems interpretation

**Transformer advantages**:
- Parallel processing
- Better long-range dependencies
- Attention is interpretable

**When to use RNNs**: Short sequences, real-time processing, explicit memory needs

### vs CNNs

**RNN advantages**:
- Handles variable-length sequences
- Temporal dependencies

**CNN advantages**:
- Local feature detection
- Spatial interpretation (Grad-CAM)

**When to use RNNs**: Time series, text, any sequential data

### vs Feedforward

**RNN advantages**:
- Memory across time
- Parameter sharing

**Feedforward advantages**:
- Simpler
- Easier to analyze
- No temporal complexity

**When to use RNNs**: When past context matters

---

## Best Practices

### For Analysis

1. **Start simple**: Vanilla RNN before LSTM/GRU
2. **Use multiple methods**: Combine probing, dynamics, visualization
3. **Include controls**: Random/untrained networks
4. **Track over time**: Analyze training dynamics
5. **Validate causally**: Use activation patching

### For Debugging

1. **Check gradient flow**: Vanishing/exploding gradients
2. **Analyze gates**: Saturation indicates problems
3. **Probe expectations**: Verify expected concepts are encoded
4. **Compare architectures**: Is LSTM better than GRU for this task?
5. **Visualize**: PHATE embeddings reveal issues

### For Research

1. **Document carefully**: Save checkpoints, log everything
2. **Use synthetic tasks**: Ground truth for validation
3. **Compare to neuroscience**: Biological inspiration
4. **Publish negative results**: What didn't work matters
5. **Share code/data**: Enable reproduction

---

## Further Reading

**Architecture papers**:
- Hochreiter & Schmidhuber (1997): LSTM (original)
- Cho et al. (2014): GRU
- Chung et al. (2014): Empirical evaluation

**Interpretability**:
- Sussillo & Barak (2013): Fixed-point analysis
- Karpathy et al. (2015): Interpretable neurons
- Belinkov & Glass (2019): Analysis methods

**Case studies in this handbook**:
- [Flip-Flop Attractors](../../5_case_studies/recurrent_networks/flip_flop_attractors.md)
- [Sentiment Line Attractors](../../5_case_studies/natural_language/sentiment_line_attractors.md)
- [Connect Four GRU](../../5_case_studies/board_games/connect_four_gru.md)

**Methods**:
- [Fixed Points](../2_methods/dynamical_analysis/fixed_points.md)
- [Linear Probes](../2_methods/probing/linear_probes.md)
- [Trajectory Analysis](../2_methods/dynamical_analysis/trajectory_analysis.md)

**Foundations**:
- [Dynamical Systems Primer](../../1_foundations/dynamical_systems_primer.md)

Full bibliography: [References](../../references/bibliography.md)

---

**Return to**: [Architectures](README.md) | [Main Handbook](../../0_start_here/README.md)
