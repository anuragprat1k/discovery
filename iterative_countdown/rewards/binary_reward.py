"""Binary reward: 1.0 if target reached, 0.0 otherwise.

Used as the group-level reward in compute_group_rewards.
Per-turn reward is 0 (no shaping signal).
"""
from __future__ import annotations


def compute_turn_reward(
    expression_valid: bool,
    result: int | None,
    target: int,
    available_numbers: list[int],
    turn: int,
    max_turns: int,
    best_distance: float,
    initial_distance: float,
) -> tuple[float, dict[str, float]]:
    """Per-turn reward for binary mode.

    Always returns 0.0 reward — all signal comes from the episode end.
    Returns (reward, metrics).
    """
    metrics: dict[str, float] = {"turn": turn, "valid": float(expression_valid)}
    if result is not None:
        metrics["distance"] = abs(target - result)
    return 0.0, metrics


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
