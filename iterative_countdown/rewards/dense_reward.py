"""Dense reward: per-turn signal combining proximity, validity, constraint adherence,
and progress.

This is the key experimental condition — hypothesis is that dense rewards enable
discovery of solutions to previously unsolvable problems.
"""
from __future__ import annotations

from . import reward_utils

# Default weights for reward components
DEFAULT_WEIGHTS: dict[str, float] = {
    "proximity": 0.3,   # Getting closer to target
    "validity": 0.1,    # Valid expression syntax + valid numbers
    "progress": 0.2,    # Progress relative to turn budget
    "correctness": 0.4, # Exact match (only on turn where target is hit)
}


def compute_turn_reward(
    expression_valid: bool,
    result: int | None,
    target: int,
    available_numbers: list[int],
    numbers_used: list[int] | None,
    turn: int,
    max_turns: int,
    best_distance: float,
    initial_distance: float,
    prev_distance: float,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Per-turn reward combining multiple components.

    Returns (reward, metrics) where reward is in approximately [-0.5, 1.0].

    Components
    ----------
    proximity   : reward for getting closer to target (can be negative if further)
    validity    : small reward for valid expression
    progress    : reward based on progress toward target relative to turn budget
    correctness : big reward for hitting target exactly
    """
    w = weights or DEFAULT_WEIGHTS
    metrics: dict[str, float] = {"turn": turn, "valid": float(expression_valid)}

    if not expression_valid or result is None:
        # Penalty for invalid expression
        reward = -0.1
        metrics["reward"] = reward
        return reward, metrics

    new_distance = abs(target - result)
    metrics["distance"] = float(new_distance)
    metrics["distance_improvement"] = prev_distance - new_distance

    # Proximity improvement
    prox = reward_utils.proximity_improvement(prev_distance, new_distance, target)

    # Validity
    numbers_ok = reward_utils.constraint_score(numbers_used or [], available_numbers)
    val = reward_utils.validity_score(True, numbers_ok >= 1.0)

    # Progress
    prog = reward_utils.progress_score(turn, max_turns, new_distance, initial_distance)

    # Correctness
    correct = reward_utils.correctness_score(result, target)

    reward = (
        w["proximity"] * prox
        + w["validity"] * val
        + w["progress"] * prog
        + w["correctness"] * correct
    )

    metrics.update(
        {
            "proximity_component": prox,
            "validity_component": val,
            "progress_component": prog,
            "correctness_component": correct,
            "reward": reward,
        }
    )

    return reward, metrics


def compute_episode_reward(
    target_reached: bool,
    best_distance: float,
    initial_distance: float,
    total_turns: int,
    max_turns: int,
) -> tuple[float, dict[str, float]]:
    """Episode-level bonus reward (on top of per-turn rewards).

    Small bonus for reaching target (most signal already given per-turn).
    """
    if target_reached:
        efficiency_bonus = 0.2 * (1.0 - total_turns / max_turns)
        reward = efficiency_bonus
    else:
        # Small partial credit based on best distance achieved
        if initial_distance > 0:
            reward = 0.1 * max(0.0, 1.0 - best_distance / initial_distance)
        else:
            reward = 0.0

    metrics = {
        "episode_reward": reward,
        "target_reached": float(target_reached),
        "best_distance": best_distance,
        "total_turns": float(total_turns),
    }
    return reward, metrics
