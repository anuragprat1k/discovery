"""PRIME-style implicit process reward.

Instead of explicit per-step scoring, this evaluates the quality of the model's
reasoning process by looking at the trajectory holistically.

Key idea: reward trajectories where the model shows systematic problem-solving
(trying useful combinations, building toward the target) vs random guessing.
"""
from __future__ import annotations

import re


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
    model_text: str = "",
) -> tuple[float, dict[str, float]]:
    """PRIME per-turn reward: minimal per-step signal.

    Only rewards valid expressions (implicit: model learns what's valid through
    trial).
    """
    metrics: dict[str, float] = {"turn": turn, "valid": float(expression_valid)}

    if not expression_valid or result is None:
        return -0.05, metrics

    new_distance = abs(target - result)
    metrics["distance"] = float(new_distance)

    # Small reward just for a valid expression
    reward = 0.05

    # Exact hit gets full reward
    if result == target:
        reward = 1.0

    metrics["reward"] = reward
    return reward, metrics


def compute_episode_reward(
    target_reached: bool,
    best_distance: float,
    initial_distance: float,
    total_turns: int,
    max_turns: int,
    trajectory_texts: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """PRIME episode reward: evaluates trajectory quality holistically.

    Components
    ----------
    Outcome         : did we reach the target?
    Efficiency      : how quickly?
    Process quality : systematic exploration vs random guessing (measured by
                      diverse operations used, no repeated expressions, etc.)
    """
    metrics: dict[str, float] = {
        "target_reached": float(target_reached),
        "best_distance": best_distance,
        "total_turns": float(total_turns),
    }

    if target_reached:
        # Base reward + efficiency bonus
        reward = 1.0 + 0.3 * (1.0 - total_turns / max_turns)
    else:
        # Partial credit based on best approach
        if initial_distance > 0:
            approach_ratio = max(0.0, 1.0 - best_distance / initial_distance)
            reward = 0.3 * approach_ratio
        else:
            reward = 0.0

    # Process quality bonus from trajectory analysis
    if trajectory_texts:
        process_bonus = _evaluate_process_quality(trajectory_texts)
        reward += 0.1 * process_bonus
        metrics["process_quality"] = process_bonus

    metrics["episode_reward"] = reward
    return reward, metrics


def _evaluate_process_quality(trajectory_texts: list[str]) -> float:
    """Evaluate reasoning quality from trajectory texts.

    Heuristics
    ----------
    - Diverse operations used (not repeating same operation)
    - No repeated identical expressions (penalized)
    - More unique operations -> higher score

    Returns a value clamped to [0, 1].
    """
    if not trajectory_texts:
        return 0.0

    operations_seen: set[str] = set()
    expressions_seen: set[str] = set()
    score = 0.0

    for text in trajectory_texts:
        # Extract expression
        expr_match = re.search(r"Expression:\s*(.+?)(?:\n|$)", text)
        if not expr_match:
            continue

        expr = expr_match.group(1).strip()

        # Penalize repeated expressions
        if expr in expressions_seen:
            score -= 0.2
        expressions_seen.add(expr)

        # Reward diverse operations
        for op in ["+", "-", "*", "/"]:
            if op in expr:
                operations_seen.add(op)

    # Bonus for operation diversity
    score += len(operations_seen) * 0.15

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))
