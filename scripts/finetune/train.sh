#!/bin/bash
# Fine-tune Gemma 3 4B for larql self-annotation
# Runs on Apple Silicon via MLX (no CUDA needed)
#
# Usage:
#   ./scripts/finetune/train.sh
#
# Prerequisites:
#   source ~/.larql/venv/bin/activate
#   pip install mlx-lm

set -e

MODEL="mlx-community/gemma-3-4b-it-4bit"
DATA_DIR="scripts/finetune/data"
OUTPUT_DIR="scripts/finetune/adapters"
ITERS=600
LR=1e-4
RANK=16
BATCH=1
LAYERS=16

echo "=== larql Foundation Fine-Tune ==="
echo "  Model:  $MODEL"
echo "  Data:   $(wc -l < $DATA_DIR/train.jsonl) train, $(wc -l < $DATA_DIR/valid.jsonl) valid"
echo "  LoRA:   rank=$RANK, layers=$LAYERS"
echo "  Iters:  $ITERS, lr=$LR, batch=$BATCH"
echo ""

python -m mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --batch-size $BATCH \
  --num-layers $LAYERS \
  --lora-rank $RANK \
  --iters $ITERS \
  --learning-rate $LR \
  --adapter-path "$OUTPUT_DIR" \
  --val-batches 5

echo ""
echo "=== Training complete ==="
echo "  Adapters saved to: $OUTPUT_DIR"
echo ""
echo "To test:"
echo "  python -m mlx_lm.generate --model $MODEL --adapter-path $OUTPUT_DIR --prompt 'My name is Alice and I work on databases.'"
echo ""
echo "To fuse + export GGUF:"
echo "  python -m mlx_lm.fuse --model $MODEL --adapter-path $OUTPUT_DIR --save-path scripts/finetune/fused --de-quantize"
echo "  python convert_hf_to_gguf.py scripts/finetune/fused --outtype f16 --outfile gemma3-4b-larql.gguf"
echo "  ./llama-quantize gemma3-4b-larql.gguf gemma3-4b-larql-q4k.gguf Q4_K_M"
