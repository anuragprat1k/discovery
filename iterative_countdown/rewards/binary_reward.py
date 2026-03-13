"""Binary reward: 1.0 if target reached, 0.0 otherwise.

Used as the group-level reward in compute_group_rewards.
Per-turn reward is a small formatting signal only.
"""
from __future__ import annotations

import re


def _has_expression_format(text: str) -> bool:
    """Check if model output contains 'Expression: <something>'."""
    return bool(re.search(r'[Ee]xpression:\s*.+', text))


def compute_turn_reward(
    expression_valid: bool,
    result: int | None,
    target: int,
    available_numbers: list[int],
    turn: int,
    max_turns: int,
    best_distance: float,
    initial_distance: float,
    model_text: str = "",
) -> tuple[float, dict[str, float]]:
    """Per-turn reward for binary mode.

    Returns 0.1 if model output has correct Expression: format, 0.0 otherwise.
    Main signal still comes from episode end.
    """
    has_format = _has_expression_format(model_text) if model_text else False
    reward = 0.1 if has_format else 0.0
    metrics: dict[str, float] = {
        "turn": turn,
        "valid": float(expression_valid),
        "format_reward": reward,
    }
    if result is not None:
        metrics["distance"] = abs(target - result)
    return reward, metrics


def compute_episode_reward(
    target_reached: bool,
    best_distance: float,
    initial_distance: float,
    total_turns: int,
    max_turns: int,
) -> tuple[float, dict[str, float]]:
    """Episode-level reward.  1.0 if target reached, 0.0 otherwise."""
    reward = 1.0 if target_reached else 0.0
    metrics = {
        "episode_reward": reward,
        "target_reached": float(target_reached),
        "best_distance": best_distance,
        "total_turns": float(total_turns),
    }
    return reward, metrics
