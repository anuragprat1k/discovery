"""Experiment 1: Binary reward baseline.

Outcome-only reward (1.0 if target reached, 0.0 otherwise).
No per-turn shaping signal. This serves as the control condition
to measure whether dense rewards enable discovery.
"""
from __future__ import annotations

import asyncio

from .train import CLIConfig, main

config = CLIConfig(
    reward_type="binary",
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    group_size=8,
    batch_size=16,
    learning_rate=3e-5,
    max_tokens=256,
    max_turns=5,
    temperature=1.0,
    log_path="~/countdown_logs/e1_binary",
    wandb_name="countdown-e1-binary",
    eval_every=10,
    save_every=20,
)

if __name__ == "__main__":
    asyncio.run(main(config))
