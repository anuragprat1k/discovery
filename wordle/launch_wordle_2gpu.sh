#!/bin/bash
# Launch Wordle GRPO training with TRL + colocated vLLM.
#
# vLLM inference and training run in the same process on one GPU.
# With 96GB GPUs this comfortably fits Qwen3-4B for both.
#
# Usage:
#   ./launch_wordle_2gpu.sh --reward dense                    # defaults
#   ./launch_wordle_2gpu.sh --reward sparse                   # sparse reward
#   ./launch_wordle_2gpu.sh --reward dense --max_steps 2      # smoke test

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
LORA_RANK="${LORA_RANK:-64}"

echo "=== Wordle GRPO Training (colocate mode) ==="
echo "Model:     $MODEL"
echo "LoRA rank: $LORA_RANK"
echo ""

python -m wordle.recipes.train_2gpu \
    --model "$MODEL" \
    --lora_rank "$LORA_RANK" \
    "$@"
