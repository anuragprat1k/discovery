#!/bin/bash
# RunPod setup for Wordle RLVR autoresearch
# Target: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)
# PyTorch 2.8.0+cu128 pre-installed
set -e

echo "=== RunPod Wordle Autoresearch Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd /workspace

# Step 1: Install dependencies (torch already present)
echo "=== Installing dependencies ==="
pip install --quiet vllm trl peft datasets accelerate wandb python-dotenv matplotlib tqdm

# Step 2: Clone repo (skip if already present)
if [ ! -d "discovery" ]; then
    echo "=== Cloning repository ==="
    git clone https://github.com/anuragprat1k/discovery.git
else
    echo "=== Updating existing repository ==="
    cd discovery && git pull && cd ..
fi
cd discovery

# Step 3: Check for .env
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: .env file not found!"
    echo "Copy your .env with API keys:"
    echo "  scp .env root@\$(hostname -I | awk '{print \$1}'):/workspace/discovery/.env"
    echo ""
fi

# Step 4: Cache model weights
echo "=== Caching Qwen3-4B model ==="
HF_HOME=/workspace/hf_cache python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
print('Downloading model...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B', torch_dtype='auto')
print('Model cached.')
"

# Step 5: Generate expert buffer
echo "=== Generating expert replay buffer (1000 games) ==="
python3 -m wordle.autoresearch.expert_buffer \
    --n_games 1000 \
    --output /workspace/expert_buffer.jsonl

# Step 6: SFT warmup (100 steps on expert data)
echo "=== Running SFT warmup ==="
HF_HOME=/workspace/hf_cache python3 -m wordle.recipes.sft_warmup \
    --data /workspace/expert_buffer.jsonl \
    --output_dir /workspace/checkpoints/sft_merged \
    --max_steps 100 \
    --batch_size 8 \
    --model Qwen/Qwen3-4B

# Step 7: Verify setup
echo "=== Verifying reward registry ==="
python3 -c "
from wordle.rewards.registry import list_reward_names
names = list_reward_names()
print(f'Registered rewards: {names}')
assert 'potential' in names, 'potential reward missing!'
assert 'reduction' in names, 'reduction reward missing!'
print('All rewards registered.')
"

echo "=== Verifying SFT checkpoint ==="
python3 -c "
from pathlib import Path
ckpt = Path('/workspace/checkpoints/sft_merged')
assert ckpt.exists(), f'{ckpt} not found'
files = list(ckpt.iterdir())
print(f'SFT checkpoint files: {[f.name for f in files]}')
print('SFT checkpoint OK.')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start the 4-condition comparison:"
echo "  cd /workspace/discovery"
echo "  nohup python3 -m wordle.autoresearch.run_queue_gpu </dev/null > runner.log 2>&1 &"
echo ""
echo "Monitor progress:"
echo "  tail -f runner.log"
echo "  # or check W&B dashboard"
echo ""
