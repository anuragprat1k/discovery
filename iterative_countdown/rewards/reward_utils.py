"""Shared reward utilities for the Countdown environment."""
from __future__ import annotations

import math


def proximity_score(current_value: int, target: int, max_distance: float | None = None) -> float:
    """Score based on how close current_value is to target.

    Returns a value in [0, 1].  1.0 means exact match.
    Uses exponential decay: exp(-|current - target| / scale)
    where scale = max(max_distance or target, 1).
    """
    distance = abs(current_value - target)
    if distance == 0:
        return 1.0
    scale = max(max_distance if max_distance is not None else abs(target), 1)
    return math.exp(-distance / scale)


def proximity_improvement(old_distance: float, new_distance: float, target: int) -> float:
    """Reward for getting closer to the target.

    Returns a positive value if improved, negative if worse, 0 if unchanged.
    Normalized by max(|target|, 1) so the magnitude is comparable across problems.
    """
    improvement = old_distance - new_distance
    normalizer = max(abs(target), 1)
    return improvement / normalizer


def validity_score(expression_valid: bool, numbers_valid: bool) -> float:
    """Score for whether the expression is syntactically valid and uses valid numbers.

    1.0 if both valid, 0.5 if syntax valid but wrong numbers, 0.0 if invalid syntax.
    """
    if not expression_valid:
        return 0.0
    return 1.0 if numbers_valid else 0.5


def constraint_score(numbers_used: list[int], available_numbers: list[int]) -> float:
    """Score for respecting number constraints.

    1.0 if every number in *numbers_used* appears in *available_numbers* (each used
    at most once).  Partial credit: fraction of used numbers that are valid.
    """
    if not numbers_used:
        return 1.0

    # Work on a mutable copy so we can "consume" available numbers one-by-one.
    remaining = list(available_numbers)
    valid_count = 0
    for n in numbers_used:
        if n in remaining:
            remaining.remove(n)
            valid_count += 1

    return valid_count / len(numbers_used)


def progress_score(
    turn: int, max_turns: int, distance_to_target: float, initial_distance: float
) -> float:
    """Score that rewards making progress toward the target relative to turn budget.

    Higher reward for reaching the target faster (fewer turns used).
    Returns a value in [0, 1].
    """
    if initial_distance <= 0:
        # Already at target from the start – full credit.
        return 1.0

    # How much of the distance has been closed?
    distance_ratio = max(0.0, 1.0 - distance_to_target / initial_distance)

    # How much budget remains?  More remaining budget → higher multiplier.
    budget_ratio = max(0.0, 1.0 - turn / max_turns) if max_turns > 0 else 0.0

    # Combine: full distance_ratio credit, with a bonus for being early.
    return distance_ratio * (0.5 + 0.5 * budget_ratio)


def correctness_score(result: int, target: int) -> float:
    """Final correctness: 1.0 if result == target, 0.0 otherwise."""
    return 1.0 if result == target else 0.0
