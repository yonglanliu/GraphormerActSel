#!/bin/bash

# ==========================
#   Graphormer Evaluation
# ==========================

CHECKPOINT_PATH="checkpoints/best_model.pt"  # Need to modify: your best model checkpoint path
DATA_PATH="/data.pt"  # Need to modify: your path of the data.pt
BATCH_SIZE=16
OUT_DIR="eval_outputs" # Custom your output path

echo "Running evaluation..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data: $DATA_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Output dir: $OUT_DIR"

python src/analyze/auto_eval.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --data "$DATA_PATH" \
    --batch_size $BATCH_SIZE \
    --out_dir "$OUT_DIR"

echo "Evaluation complete."
