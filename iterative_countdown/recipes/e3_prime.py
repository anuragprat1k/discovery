"""Experiment 3: PRIME-style implicit process rewards.

Minimal per-step signal with holistic trajectory evaluation.
The PRIME approach uses the outcome to implicitly credit intermediate
steps, avoiding hand-crafted shaping while still providing denser
signal than pure binary rewards.
"""
from __future__ import annotations

import asyncio

from .train import CLIConfig, main

config = CLIConfig(
    reward_type="prime",
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    group_size=8,
    batch_size=16,
    learning_rate=3e-5,
    max_tokens=256,
    max_turns=5,
    temperature=1.0,
    log_path="~/countdown_logs/e3_prime",
    wandb_name="countdown-e3-prime",
    eval_every=10,
    save_every=20,
)

if __name__ == "__main__":
    asyncio.run(main(config))
