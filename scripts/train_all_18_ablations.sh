#!/bin/bash
# Train all 18 ablation configurations
# 3 kernel sizes × 3 channels × 2 GRU sizes = 18 models

PYTHON="dataset/Alpha-Zero-algorithm-for-Connect-4-game/.venv/bin/python3"
DATA="data/connect4_10k_games.pt"
EPOCHS=100
SAVE_EVERY=10

echo "=========================================="
echo "Training 18 Ablation Configurations"
echo "=========================================="
echo ""
echo "Ablations:"
echo "  - Kernel: 3x3, 6x6"
echo "  - Channels: 16, 64, 256"
echo "  - GRU Hidden: 8, 32, 128"
echo ""
echo "Total: 2 × 3 × 3 = 18 models"
echo "=========================================="
echo ""

COUNT=1

for KERNEL in 3 6; do
    for CHANNELS in 16 64 256; do
        for GRU in 8 32 128; do
            echo ""
            echo "[$COUNT/18] Training: Kernel=${KERNEL}x${KERNEL}, Channels=$CHANNELS, GRU=$GRU"
            echo "----------------------------------------"

            $PYTHON src/train.py \
                --data $DATA \
                --kernel-size $KERNEL \
                --cnn-channels $CHANNELS \
                --gru-hidden $GRU \
                --epochs $EPOCHS \
                --save-every $SAVE_EVERY \
                --batch-size 64

            echo ""
            echo "✓ Completed $COUNT/18"
            echo ""

            COUNT=$((COUNT + 1))
        done
    done
done

echo ""
echo "=========================================="
echo "✓ ALL 18 ABLATIONS COMPLETE!"
echo "=========================================="
echo ""
echo "Checkpoints saved in: checkpoints/"
echo ""
echo "Configuration naming: k{kernel}_c{channels}_gru{hidden}_timestamp/"
