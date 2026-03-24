"""Greens-only + random noise reward (non-potential control).

Per-turn: +0.4 per new green tile + random bonus (0 or +0.2, uniform).
The random component is non-potential and uncorrelated with game state,
serving as a control to show that not just *any* dense signal helps —
it must be skill-aligned.

Episode terminal: +3.0 + 0.1*(6-turn) for wins, -1.0 for losses.
"""
from __future__ import annotations

import random
import re

from wordle.environment.feedback import TileColor
from wordle.rewards.reward_utils import count_new_greens, has_constraint_violation


def _has_guess_tag(completion_text: str) -> bool:
    """Check if completion contains a valid <guess>XXXXX</guess> tag."""
    return bool(re.search(r"<guess>\s*[a-zA-Z]{5}\s*</guess>", completion_text, re.IGNORECASE))


def compute_turn_reward(
    guess: str,
    feedback: list[TileColor],
    prev_feedbacks: list[list[TileColor]],
    prev_guesses: list[str],
    turn: int,
    max_turns: int,
    target_reached: bool,
    completion_text: str = "",
    completion_tokens: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Per-turn reward: greens + random noise.

    +0.4 per new green (same as greens-only)
    +0.2 or 0.0 randomly (50/50, uniform)
    """
    reward = 0.0
    metrics: dict[str, float] = {"turn": float(turn)}

    new_greens = count_new_greens(feedback, prev_feedbacks)
    reward += 0.4 * new_greens

    # Random noise: +0.2 or 0 with equal probability
    noise = 0.2 if random.random() < 0.5 else 0.0
    reward += noise

    metrics["new_greens"] = float(new_greens)
    metrics["new_yellows"] = 0.0
    metrics["noise"] = noise

    # Constraint violation tracking (metric only)
    if has_constraint_violation(guess, prev_guesses, prev_feedbacks):
        metrics["constraint_violation"] = 1.0
    else:
        metrics["constraint_violation"] = 0.0

    # Format compliance tracking
    if completion_text and _has_guess_tag(completion_text):
        metrics["format_compliance"] = 1.0
    else:
        metrics["format_compliance"] = 0.0

    return reward, metrics


def compute_episode_reward(
    target_reached: bool,
    total_turns: int,
    max_turns: int,
) -> tuple[float, dict[str, float]]:
    """Episode-level bonus: +3.0 + 0.1*turns_remaining for wins, -1.0 for losses."""
    if target_reached:
        turns_remaining = max_turns - total_turns
        reward = 3.0 + 0.1 * turns_remaining
    else:
        reward = -1.0

    return reward, {
        "episode_reward": reward,
        "target_reached": float(target_reached),
    }
