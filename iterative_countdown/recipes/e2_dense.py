"""Experiment 2: Dense per-turn rewards.

Hypothesis: dense rewards enable discovery of new solutions by providing
gradient signal on every turn, not just at episode end. The model receives
incremental reward for making progress toward the target (reducing distance),
plus a bonus for valid expressions.
"""
from __future__ import annotations

import asyncio

from .train import CLIConfig, main

config = CLIConfig(
    reward_type="dense",
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    group_size=8,
    batch_size=16,
    learning_rate=3e-5,
    max_tokens=256,
    max_turns=5,
    temperature=1.0,
    log_path="~/countdown_logs/e2_dense",
    wandb_name="countdown-e2-dense",
    eval_every=10,
    save_every=20,
)

if __name__ == "__main__":
    asyncio.run(main(config))
