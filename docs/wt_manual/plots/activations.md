# CNN activation maps (Grad-CAM)

Produced by: `./wt.sh visualize --viz-type activations --checkpoint-dir <run>`
Backed by: `src/visualize_trajectories.py`

What it shows
- For a few random board states, overlays a Grad-CAM heatmap on the 6×7 grid
- Target can be policy (per-move logit; choose a move via --activation-move or default to predicted) or value

Artifacts
- visualizations/activations/activation_000.png, activation_001.png, ...
  - Titles show target type, focus move (and its probability) and predicted value
  - Turn indicator is printed, and axes label columns 0..6 and rows 5..0

How to read
- Bright regions are most influential spatial locations for the target
- For policy target, heat aligns with plausible moves; for value target, heat may emphasize threats/opportunities

Options
- --n-boards: pool size of random boards to sample from
- --activation-target [policy|value], --activation-move index, --activation-max-examples
