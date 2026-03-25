#!/bin/bash
# General RunPod setup for the discovery repo.
# Handles: deps, repo clone/checkout, .env, model caching.
#
# Usage (from local machine):
#   ssh <runpod> 'bash -s' < scripts/setup_runpod.sh
#   BRANCH=main ssh <runpod> 'bash -s' < scripts/setup_runpod.sh
#   WANDB_API_KEY=<key> ssh <runpod> 'WANDB_API_KEY='"$WANDB_API_KEY"' bash -s' < scripts/setup_runpod.sh
#
# Or from the RunPod directly:
#   BRANCH=feat/gpu-autoresearch bash scripts/setup_runpod.sh
set -euo pipefail

BRANCH="${BRANCH:-feat/gpu-autoresearch}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"

echo "=== RunPod Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch: $BRANCH"
echo "Model: $MODEL"

cd /workspace

# --- Virtual environment ---
if [ ! -d "/workspace/venv" ]; then
    echo "=== Creating virtual environment ==="
    python3 -m venv /workspace/venv --system-site-packages
fi
source /workspace/venv/bin/activate
echo "Using Python: $(which python3)"

# --- Dependencies (torch is pre-installed via system-site-packages) ---
echo "=== Installing Python dependencies ==="
pip install --quiet vllm trl peft datasets accelerate wandb python-dotenv matplotlib tqdm

# --- Clone / update repo ---
if [ ! -d "discovery" ]; then
    echo "=== Cloning repository ==="
    git clone https://github.com/anuragprat1k/discovery.git
fi
cd discovery
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH" || true

# --- .env ---
if [ ! -f ".env" ] && [ -n "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY=$WANDB_API_KEY" > .env
    echo "Created .env from WANDB_API_KEY env var."
elif [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: No .env file and WANDB_API_KEY not set."
    echo "  Option 1: export WANDB_API_KEY=<key> before running"
    echo "  Option 2: scp .env to /workspace/discovery/.env"
    echo ""
fi

# --- Cache model weights ---
echo "=== Caching model: $MODEL ==="
HF_HOME=/workspace/hf_cache python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = '$MODEL'
print(f'Downloading tokenizer for {model}...')
AutoTokenizer.from_pretrained(model)
print(f'Downloading model weights for {model}...')
AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
print('Model cached.')
"

echo ""
echo "=== Base setup complete ==="
echo "Repo: /workspace/discovery (branch: $BRANCH)"
echo "Model cached: $MODEL"
echo "HF cache: /workspace/hf_cache"
echo "Venv: source /workspace/venv/bin/activate"
echo ""
echo "Next steps (Wordle v4):"
echo "  source /workspace/venv/bin/activate"
echo "  cd /workspace/discovery"
echo "  bash wordle/scripts/runpod_setup.sh   # SFT warmup + verify"
echo ""
