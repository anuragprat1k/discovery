#!/bin/bash
# RunPod setup for Wordle RLVR autoresearch experiments.
# Base image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
#   Pre-installed: torch 2.8.0, transformers, datasets, accelerate, numpy, tqdm, matplotlib
#
# This script:
#   1. Creates a venv (system-site-packages to reuse torch/transformers)
#   2. Installs only missing pip packages
#   3. Clones repo + checks out branch
#   4. Sets up .env for W&B
#   5. Caches model weights
#   6. Generates expert replay buffer
#   7. Runs SFT warmup (LoRA → merged checkpoint)
#   8. Verifies everything works
#
# Usage (from local machine):
#   ssh <runpod> 'bash -s' < scripts/setup_runpod_wordle.sh
#
#   # With env vars:
#   BRANCH=feat/my-branch WANDB_API_KEY=<key> \
#     ssh <runpod> 'BRANCH='"$BRANCH"' WANDB_API_KEY='"$WANDB_API_KEY"' bash -s' < scripts/setup_runpod_wordle.sh
#
# Usage (from RunPod terminal):
#   BRANCH=feat/gpu-autoresearch bash scripts/setup_runpod_wordle.sh
set -euo pipefail

BRANCH="${BRANCH:-feat/gpu-autoresearch}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
EXPERT_GAMES="${EXPERT_GAMES:-1000}"
SFT_STEPS="${SFT_STEPS:-100}"

echo "========================================"
echo "  RunPod Wordle Autoresearch Setup"
echo "========================================"
echo "GPU:           $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch:        $BRANCH"
echo "Model:         $MODEL"
echo "Expert games:  $EXPERT_GAMES"
echo "SFT steps:     $SFT_STEPS"
echo "========================================"

cd /workspace

# ── 1. Virtual environment ──────────────────────────────────────────
# Reuse system-site-packages so torch/transformers/datasets/accelerate
# don't need reinstalling.
if [ ! -d "/workspace/venv" ]; then
    echo ">>> Creating virtual environment..."
    python3 -m venv /workspace/venv --system-site-packages
fi
source /workspace/venv/bin/activate
echo ">>> Python: $(which python3) ($(python3 --version))"

# ── 2. Install missing dependencies ────────────────────────────────
# Only packages NOT in the base image. torch, transformers, datasets,
# accelerate, numpy, tqdm, matplotlib are already present.
echo ">>> Installing additional pip packages..."
pip install --quiet \
    vllm \
    trl \
    peft \
    wandb \
    python-dotenv

# ── 3. Clone / update repo ─────────────────────────────────────────
if [ -d "discovery/.git" ]; then
    echo ">>> Updating existing repository..."
    cd discovery
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH" || true
else
    # Back up .env if stale directory exists
    [ -d "discovery" ] && { cp discovery/.env /tmp/discovery_env_backup 2>/dev/null || true; rm -rf discovery; }
    echo ">>> Cloning repository..."
    git clone https://github.com/anuragprat1k/discovery.git
    cd discovery
    git checkout "$BRANCH"
    [ -f /tmp/discovery_env_backup ] && { cp /tmp/discovery_env_backup .env; rm /tmp/discovery_env_backup; }
fi

# ── 4. Environment file ────────────────────────────────────────────
if [ ! -f ".env" ] && [ -n "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY=$WANDB_API_KEY" > .env
    echo ">>> Created .env from WANDB_API_KEY env var."
elif [ ! -f ".env" ]; then
    echo ""
    echo "⚠  WARNING: No .env file and WANDB_API_KEY not set."
    echo "   Option 1: re-run with WANDB_API_KEY=<key>"
    echo "   Option 2: scp .env root@<pod-ip>:/workspace/discovery/.env"
    echo ""
fi

# Symlink .env into wordle/ so train_2gpu.py can find it
[ -f ".env" ] && [ ! -e "wordle/.env" ] && ln -s "$(pwd)/.env" wordle/.env

# ── 5. Cache model weights ─────────────────────────────────────────
echo ">>> Caching model: $MODEL ..."
HF_HOME=/workspace/hf_cache python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = '$MODEL'
print(f'  Downloading tokenizer for {model}...')
AutoTokenizer.from_pretrained(model)
print(f'  Downloading model weights for {model}...')
AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
print('  Model cached.')
"

# ── 6. Generate expert replay buffer ───────────────────────────────
EXPERT_BUFFER="/workspace/expert_buffer.jsonl"
if [ -f "$EXPERT_BUFFER" ]; then
    echo ">>> Expert buffer already exists at $EXPERT_BUFFER, skipping."
else
    echo ">>> Generating expert replay buffer ($EXPERT_GAMES games)..."
    python3 -m wordle.autoresearch.expert_buffer \
        --n_games "$EXPERT_GAMES" \
        --output "$EXPERT_BUFFER"
fi

# ── 7. SFT warmup ──────────────────────────────────────────────────
SFT_DIR="/workspace/checkpoints/sft_merged"
if [ -d "$SFT_DIR" ] && [ "$(ls -A "$SFT_DIR" 2>/dev/null)" ]; then
    echo ">>> SFT checkpoint already exists at $SFT_DIR, skipping."
else
    echo ">>> Running SFT warmup ($SFT_STEPS steps)..."
    HF_HOME=/workspace/hf_cache python3 -m wordle.recipes.sft_warmup \
        --data "$EXPERT_BUFFER" \
        --output_dir "$SFT_DIR" \
        --max_steps "$SFT_STEPS" \
        --batch_size 8 \
        --model "$MODEL"
fi

# ── 8. Verify setup ────────────────────────────────────────────────
echo ">>> Verifying reward registry..."
python3 -c "
from wordle.rewards.registry import list_reward_names
names = list_reward_names()
print(f'  Registered rewards: {names}')
assert 'potential' in names, 'potential reward missing!'
assert 'reduction' in names, 'reduction reward missing!'
print('  All rewards OK.')
"

echo ">>> Verifying SFT checkpoint..."
python3 -c "
from pathlib import Path
ckpt = Path('$SFT_DIR')
assert ckpt.exists(), f'{ckpt} not found'
files = sorted(f.name for f in ckpt.iterdir())
print(f'  SFT checkpoint: {files}')
print('  SFT checkpoint OK.')
"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "Paths:"
echo "  Repo:        /workspace/discovery (branch: $BRANCH)"
echo "  Venv:        source /workspace/venv/bin/activate"
echo "  Model cache: /workspace/hf_cache"
echo "  SFT ckpt:    $SFT_DIR"
echo "  Expert data: $EXPERT_BUFFER"
echo ""
echo "Run the autoresearch queue:"
echo "  source /workspace/venv/bin/activate"
echo "  cd /workspace/discovery"
echo "  nohup python3 -m wordle.autoresearch.run_queue_gpu </dev/null > runner.log 2>&1 &"
echo ""
echo "Monitor:"
echo "  tail -f runner.log"
echo ""
