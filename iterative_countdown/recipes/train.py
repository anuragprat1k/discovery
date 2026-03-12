"""Launch script for Countdown RL training using tinker-cookbook."""
from __future__ import annotations

import asyncio
import os

import chz
from tinker_cookbook.rl import train
from tinker_cookbook.rl.train import Config

from ..environment.countdown_env import CountdownDatasetBuilder


@chz.chz
class CLIConfig:
    """CLI configuration for Countdown RL training."""
    # Model
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None
    lora_rank: int = 32
    load_checkpoint_path: str | None = None

    # Environment
    max_turns: int = 5
    reward_type: str = "dense"  # "binary", "dense", "prime"
    train_problems_path: str | None = None
    eval_problems_path: str | None = None
    n_train: int = 500
    n_eval: int = 100
    seed: int = 42
    max_trajectory_tokens: int = 4096

    # Training
    group_size: int = 8
    batch_size: int = 16
    learning_rate: float = 3e-5
    max_tokens: int = 256  # per-turn generation length
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Logging
    log_path: str = "~/countdown_logs"
    eval_every: int = 10
    save_every: int = 20
    wandb_project: str | None = None
    wandb_name: str | None = None
    base_url: str | None = None


def build_config(cli: CLIConfig) -> Config:
    """Convert CLI config to tinker-cookbook training Config."""
    from tinker_cookbook.model_info import get_recommended_renderer_name

    renderer_name = cli.renderer_name
    if renderer_name is None:
        renderer_name = get_recommended_renderer_name(cli.model_name)

    dataset_builder = CountdownDatasetBuilder(
        batch_size=cli.batch_size,
        model_name_for_tokenizer=cli.model_name,
        renderer_name=renderer_name,
        group_size=cli.group_size,
        max_turns=cli.max_turns,
        reward_type=cli.reward_type,
        n_train=cli.n_train,
        n_eval=cli.n_eval,
        seed=cli.seed,
        train_problems_path=cli.train_problems_path,
        eval_problems_path=cli.eval_problems_path,
        max_trajectory_tokens=cli.max_trajectory_tokens,
    )

    return Config(
        learning_rate=cli.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli.model_name,
        max_tokens=cli.max_tokens,
        log_path=cli.log_path,
        eval_every=cli.eval_every,
        save_every=cli.save_every,
        renderer_name=renderer_name,
        wandb_project=cli.wandb_project,
        wandb_name=cli.wandb_name,
        kl_penalty_coef=cli.kl_penalty_coef,
        lora_rank=cli.lora_rank,
        temperature=cli.temperature,
        num_substeps=cli.num_substeps,
        load_checkpoint_path=cli.load_checkpoint_path,
        base_url=cli.base_url,
        remove_constant_reward_groups=True,
    )


async def main(cli: CLIConfig) -> None:
    cfg = build_config(cli)
    await train.main(cfg)


if __name__ == "__main__":
    chz.entrypoint(main, CLIConfig)
