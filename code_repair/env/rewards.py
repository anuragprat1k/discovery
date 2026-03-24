"""Reward functions for iterative code repair.

Three reward conditions matching the Wordle experiment structure:
- sparse: terminal-only (win/loss)
- dense_passes: terminal + high-water-mark of passing tests (potential-based)
- dense_full: terminal + passes + partial-correctness signals (non-potential)

All share terminal rewards and format penalties.
"""
from __future__ import annotations

import re

from code_repair.env.sandbox import TestResult


# --- Shared constants ---
TERMINAL_WIN = 3.0
SPEED_BONUS = 0.1       # per remaining turn
TERMINAL_LOSS = -1.0
FORMAT_PENALTY = -1.0    # per turn with no valid <repair> tag

# --- Dense passes constants ---
PASS_REWARD_PER_TEST = 0.4  # reward per newly-HWM-passing test (normalized)

# --- Dense full constants ---
NO_CRASH_BONUS = 0.05
TYPE_MATCH_BONUS = 0.05
SHAPE_MATCH_BONUS = 0.05


def _has_repair_tag(text: str) -> bool:
    """Check if text contains a valid <repair>...</repair> tag."""
    return bool(re.search(r"<repair>.*?</repair>", text, re.DOTALL))


def _episode_reward(all_passed: bool, total_turns: int, max_turns: int) -> float:
    """Shared terminal reward."""
    if all_passed:
        turns_remaining = max_turns - total_turns
        return TERMINAL_WIN + SPEED_BONUS * turns_remaining
    return TERMINAL_LOSS


# ---------------------------------------------------------------------------
# Sparse reward
# ---------------------------------------------------------------------------

def sparse_reward(
    info: dict,
    is_terminal: bool,
    completion_text: str = "",
) -> tuple[float, dict[str, float]]:
    """Terminal-only reward. No per-turn shaping.

    Args:
        info: Dict from CodeRepairEnv.step() with test results and metrics.
        is_terminal: Whether this is the last turn.
        completion_text: Raw model output for format checking.

    Returns:
        (reward, metrics) tuple.
    """
    reward = 0.0
    metrics: dict[str, float] = {"turn": float(info.get("turn", 0))}

    # Format compliance tracking (metric only, no reward for sparse)
    if completion_text and _has_repair_tag(completion_text):
        metrics["format_compliance"] = 1.0
    else:
        metrics["format_compliance"] = 0.0

    # Format penalty (shared across all reward types)
    if info.get("format_violation"):
        reward += FORMAT_PENALTY
        metrics["format_penalty"] = FORMAT_PENALTY

    # Terminal reward
    if is_terminal:
        ep_reward = _episode_reward(
            info.get("all_passed", False),
            info.get("turn", 0),
            info.get("max_turns", 4),
        )
        reward += ep_reward
        metrics["episode_reward"] = ep_reward
        metrics["target_reached"] = float(info.get("all_passed", False))

    metrics["total_reward"] = reward
    return reward, metrics


# ---------------------------------------------------------------------------
# Dense passes reward (potential-based)
# ---------------------------------------------------------------------------

def dense_passes_reward(
    info: dict,
    is_terminal: bool,
    completion_text: str = "",
) -> tuple[float, dict[str, float]]:
    """Terminal + high-water-mark of passing tests (potential-based).

    Tracks max_passing_so_far across turns.
    Per-turn: +0.4 * (new_hwm - old_hwm) / num_tests
    This is strictly potential-based: Phi(s) = hwm(s), reward = Phi(s') - Phi(s).
    """
    reward = 0.0
    metrics: dict[str, float] = {"turn": float(info.get("turn", 0))}

    # Format compliance
    if completion_text and _has_repair_tag(completion_text):
        metrics["format_compliance"] = 1.0
    else:
        metrics["format_compliance"] = 0.0

    # Format penalty
    if info.get("format_violation"):
        reward += FORMAT_PENALTY
        metrics["format_penalty"] = FORMAT_PENALTY

    # Per-turn: high-water mark delta
    num_tests = max(info.get("num_tests", 1), 1)
    hw_passing = info.get("hw_passing", 0)
    old_hw = info.get("old_hw", 0)
    hwm_delta = hw_passing - old_hw

    if hwm_delta > 0:
        pass_reward = PASS_REWARD_PER_TEST * hwm_delta / num_tests
        reward += pass_reward
        metrics["hwm_delta"] = float(hwm_delta)
        metrics["pass_reward"] = pass_reward
    else:
        metrics["hwm_delta"] = 0.0
        metrics["pass_reward"] = 0.0

    metrics["hw_passing"] = float(hw_passing)
    metrics["curr_passing"] = float(info.get("curr_passing", 0))

    # Terminal reward
    if is_terminal:
        ep_reward = _episode_reward(
            info.get("all_passed", False),
            info.get("turn", 0),
            info.get("max_turns", 4),
        )
        reward += ep_reward
        metrics["episode_reward"] = ep_reward
        metrics["target_reached"] = float(info.get("all_passed", False))

    metrics["total_reward"] = reward
    return reward, metrics


# ---------------------------------------------------------------------------
# Dense full reward (non-potential)
# ---------------------------------------------------------------------------

def dense_full_reward(
    info: dict,
    is_terminal: bool,
    completion_text: str = "",
) -> tuple[float, dict[str, float]]:
    """Terminal + passes + partial-correctness signals (non-potential).

    Everything from dense_passes PLUS for each FAILING test:
    - +0.05 if it ran without crashing
    - +0.05 if return type matches expected type
    - +0.05 if output shape/length matches expected

    These are non-potential because a repair can go from "returns wrong value"
    to "crashes entirely" — the signals can decrease turn-over-turn.
    """
    reward = 0.0
    metrics: dict[str, float] = {"turn": float(info.get("turn", 0))}

    # Format compliance
    if completion_text and _has_repair_tag(completion_text):
        metrics["format_compliance"] = 1.0
    else:
        metrics["format_compliance"] = 0.0

    # Format penalty
    if info.get("format_violation"):
        reward += FORMAT_PENALTY
        metrics["format_penalty"] = FORMAT_PENALTY

    # Per-turn: high-water mark delta (same as dense_passes)
    num_tests = max(info.get("num_tests", 1), 1)
    hw_passing = info.get("hw_passing", 0)
    old_hw = info.get("old_hw", 0)
    hwm_delta = hw_passing - old_hw

    if hwm_delta > 0:
        pass_reward = PASS_REWARD_PER_TEST * hwm_delta / num_tests
        reward += pass_reward
        metrics["pass_reward"] = pass_reward
    else:
        metrics["pass_reward"] = 0.0

    metrics["hwm_delta"] = float(hwm_delta)
    metrics["hw_passing"] = float(hw_passing)
    metrics["curr_passing"] = float(info.get("curr_passing", 0))

    # Partial-correctness signals for FAILING tests (non-potential)
    no_crash_count = info.get("no_crash_failing", 0)
    type_match_count = info.get("type_match_failing", 0)
    shape_match_count = info.get("shape_match_failing", 0)

    # Normalize by num_tests so scale is comparable across problems
    no_crash_reward = NO_CRASH_BONUS * no_crash_count / num_tests
    type_match_reward = TYPE_MATCH_BONUS * type_match_count / num_tests
    shape_match_reward = SHAPE_MATCH_BONUS * shape_match_count / num_tests

    partial_reward = no_crash_reward + type_match_reward + shape_match_reward
    reward += partial_reward

    metrics["no_crash_reward"] = no_crash_reward
    metrics["type_match_reward"] = type_match_reward
    metrics["shape_match_reward"] = shape_match_reward
    metrics["partial_reward"] = partial_reward

    # Terminal reward
    if is_terminal:
        ep_reward = _episode_reward(
            info.get("all_passed", False),
            info.get("turn", 0),
            info.get("max_turns", 4),
        )
        reward += ep_reward
        metrics["episode_reward"] = ep_reward
        metrics["target_reached"] = float(info.get("all_passed", False))

    metrics["total_reward"] = reward
    return reward, metrics


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REWARD_FUNCTIONS = {
    "sparse": sparse_reward,
    "dense_passes": dense_passes_reward,
    "dense_full": dense_full_reward,
}


def get_reward_fn(name: str):
    """Look up a reward function by name."""
    if name not in REWARD_FUNCTIONS:
        available = ", ".join(sorted(REWARD_FUNCTIONS))
        raise KeyError(f"Unknown reward type {name!r}. Available: {available}")
    return REWARD_FUNCTIONS[name]
