#!/bin/bash
# Train all three model configurations

PYTHON="dataset/Alpha-Zero-algorithm-for-Connect-4-game/.venv/bin/python3"
DATA="data/connect4_10k_games.pt"

echo "Training all model configurations"
echo "=================================="

# Small model (GRU=8)
echo -e "\n[1/3] Training SMALL model (GRU=8)..."
$PYTHON src/train.py \
  --data $DATA \
  --cnn-channels 16 64 256 \
  --gru-hidden 8 \
  --epochs 100 \
  --save-every 10 \
  --batch-size 64

# Medium model (GRU=32)
echo -e "\n[2/3] Training MEDIUM model (GRU=32)..."
$PYTHON src/train.py \
  --data $DATA \
  --cnn-channels 16 64 256 \
  --gru-hidden 32 \
  --epochs 100 \
  --save-every 10 \
  --batch-size 64

# Large model (GRU=128)
echo -e "\n[3/3] Training LARGE model (GRU=128)..."
$PYTHON src/train.py \
  --data $DATA \
  --cnn-channels 16 64 256 \
  --gru-hidden 128 \
  --epochs 100 \
  --save-every 10 \
  --batch-size 32

echo -e "\n✓ All models trained!"
echo "Check checkpoints/ directory for results"
