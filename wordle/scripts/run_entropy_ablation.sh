#!/bin/bash
# Launch the entropy ablation experiments on RunPod.
#
# Prerequisites:
#   - RunPod set up via: BRANCH=feat/entropy-metric bash scripts/setup_runpod.sh
#   - SFT checkpoint at /workspace/checkpoints/sft_merged
#
# Usage (on RunPod):
#   cd /workspace/discovery
#   bash wordle/scripts/run_entropy_ablation.sh
set -euo pipefail

cd /workspace/discovery
source /workspace/venv/bin/activate

echo "=== Entropy Ablation Runner ==="
echo "Branch: $(git branch --show-current)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Verify SFT checkpoint exists
if [ ! -d "/workspace/checkpoints/sft_merged" ]; then
    echo "ERROR: SFT checkpoint not found at /workspace/checkpoints/sft_merged"
    echo "Run setup first: bash wordle/scripts/runpod_setup.sh"
    exit 1
fi

# Use the entropy ablation queue
cp wordle/autoresearch/queue_entropy_ablation.jsonl wordle/autoresearch/queue_gpu.jsonl
echo "Queue loaded: $(wc -l < wordle/autoresearch/queue_gpu.jsonl) experiments"
cat wordle/autoresearch/queue_gpu.jsonl | python3 -c "import sys,json; [print(f'  - {json.loads(l)[\"name\"]} ({json.loads(l)[\"reward\"]}, {json.loads(l).get(\"cli_args\",\"\").split(\"--max_steps \")[1].split()[0]} steps)') for l in sys.stdin]"

# Results go to a dedicated directory
export RESULTS_DIR="wordle/autoresearch/results_entropy_ablation"
mkdir -p "$RESULTS_DIR"

echo ""
echo "Starting queue runner (nohup)..."
nohup python3 -m wordle.autoresearch.run_queue_gpu </dev/null > wordle/autoresearch/runner_entropy_ablation.log 2>&1 &
RUNNER_PID=$!
echo "Runner PID: $RUNNER_PID"
echo "$RUNNER_PID" > wordle/autoresearch/runner_entropy_ablation.pid

echo ""
echo "Monitor:"
echo "  tail -f wordle/autoresearch/runner_entropy_ablation.log"
echo "  # W&B: https://wandb.ai — project wordle-grpo"
