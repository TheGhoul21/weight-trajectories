# Model Architecture Visualization Guide

Generate "paper ready" diagrams of the ResNet+GRU model using two approaches.

---

## 1. PlotNeuralNet (LaTeX/TikZ)

PlotNeuralNet renders the familiar 3D block diagrams found in many deep-learning papers.

### Install

```bash
cd docs
if [ ! -d plotneuralnet ]; then
  git clone https://github.com/HarisIqbal88/PlotNeuralNet.git plotneuralnet
fi
cd plotneuralnet
pip install -r requirements.txt
```

> PlotNeuralNet needs a LaTeX toolchain (`texlive` or `mactex`) to compile the generated `.tex` file.

### Generate LaTeX description

Inside `docs/plotneuralnet`, create a file named `resnet_gru_architecture.py` with the snippet below:

```python
from pycore.tikzeng import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    to_TikZ("ResNetGRU"),

    to_input('input_board.png', name='board'),

    to_Conv(name='conv1', s_filer=7, n_filer=16, offset="(2,0,0)", size=(25,25,4), caption='Conv+BN+ReLU'),
    to_Conv(name='conv2', s_filer=7, n_filer=16, offset="(2,0,0)", size=(25,25,4), caption='Conv+BN+ReLU'),
    to_Sum(name='residual', offset="(2,0,0)"),

    to_Flatten(name='flatten', offset="(2,0,0)", caption='Flatten'),
    to_FC(name='gru', n=8, caption='GRU (hidden=8)'),

    to_FC(name='policy', n=7, offset="(2,1,0)", caption='Policy Head'),
    to_FC(name='value', n=1, offset="(2,-1,0)", caption='Value Head'),

    to_connection("board", "conv1"),
    to_connection("conv2", "residual"),
    to_connection("residual", "flatten"),
    to_connection("flatten", "gru"),
    to_connection("gru", "policy"),
    to_connection("gru", "value"),

    to_end()
]

if __name__ == '__main__':
    to_generate(arch, "resnet_gru")
```

Adapt the block sizes or captions as needed (change the hidden/filters to match a specific configuration).

### Render the figure

```bash
cd docs/plotneuralnet
python resnet_gru_architecture.py
pdflatex resnet_gru.tex
```

The output files (`resnet_gru.pdf`, `resnet_gru.png`) live in `docs/plotneuralnet`. Adjust TikZ parameters for finer detail.

---

## 2. Net2Vis (automatic publication diagram)

Net2Vis converts a model graph into a polished architecture diagram using their web UI.

### Export the model to ONNX

```bash
uv run python scripts/export_model_onnx.py \
  --cnn-channels 256 \
  --gru-hidden 8 \
  --kernel-size 3 \
  --seq-len 30 \
  --output artifacts/model_k3_c256_gru8.onnx
```

### Generate diagram

1. Head to <https://net2vis.ulriklestelius.com/> (original Net2Vis web service). If the main site is unavailable, use the open-source fork instructions in their repository.
2. Upload the exported ONNX file.
3. Tune the layer labels/colors. Net2Vis detects convolutional/recurrent layers automatically.
4. Export the figure as PDF or SVG for inclusion in papers.

> Net2Vis may need simplified naming to group layers. Feel free to edit node names in ONNX (or the Net2Vis UI) to shorten labels.

---

## Alternative quick options

- `torchviz`: `pip install torchviz` then `torchviz.make_dot(...)` for a quick Graphviz schematic.
- `Netron`: `netron artifacts/model.onnx` for an interactive inspection.

Use PlotNeuralNet when you want full control over a TikZ diagram; use Net2Vis to rapidly produce a polished block diagram from the actual model graph.
