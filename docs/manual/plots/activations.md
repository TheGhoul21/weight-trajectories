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
- **Color intensity**: brighter cells contributed more to the prediction (policy logit or value). Cool/neutral colours carry little weight.
- **Policy target**
  - Expect heat concentrated on winning threats, blocking moves, or long columns/diagonals tied to the selected action.
  - If the highlighted move looks unreasonable, double-check the policy probability in the title—low-confidence selections often produce diffuse maps.
- **Value target**
  - Bright streaks typically trace potential connect-fours (for/against the current player).
  - Watch for heat on opponent pieces: strong value intuition should highlight both offensive opportunities and defensive liabilities.
- **Turn indicator** reminds you whose perspective the map reflects; heat should align with the acting player’s threats.

Common diagnostics
- **Feature blindness**: flat/diffuse heatmaps across many boards —> the CNN isn’t focusing on specific patterns.
- **Over-focus**: always lighting up the last move played —> the network might be shortcutting via move history rather than global structure.
- **Inconsistent policy/value maps**: if policy heat ignores the regions that the value map cares about, the actor/critic components are disagreeing.

Options
- --n-boards: pool size of random boards to sample from
- --activation-target [policy|value], --activation-move index, --activation-max-examples
