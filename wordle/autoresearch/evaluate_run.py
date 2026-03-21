"""Locked evaluator for Wordle autoresearch loop.

Usage: python -m wordle.autoresearch.evaluate_run <training_log.jsonl>

Reads JSONL, computes composite score from last 5 steps, checks early-stop
conditions against every step.

Exit codes:
  0 = completed normally, score printed to stdout
  1 = early-stopped (reason on stderr)
  2 = no data / file error

Stdout (single line, TSV):
  score  win_rate  violation_rate  format_compliance  mean_completion_len  memory_peak_gb  wall_time_min  status
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Early-stop thresholds (non-negotiable)
# ---------------------------------------------------------------------------
MEMORY_PEAK_LIMIT_GB = 55.0
MAX_COMPLETION_LEN = 300        # tokens, any single step
LOSS_MAGNITUDE_LIMIT = 50.0
FORMAT_COMPLIANCE_FLOOR = 0.20  # below this for 3 consecutive steps
STEP_TIME_LIMIT_S = 900.0

# Composite score window
WINDOW = 5


def load_metrics(path: Path) -> list[dict]:
    """Load JSONL metrics, skipping SFT warmup lines."""
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Skip SFT warmup entries
            if entry.get("phase") == "sft_warmup":
                continue
            if "step" in entry:
                entries.append(entry)
    return entries


def check_early_stop(metrics: list[dict]) -> str | None:
    """Check every step for early-stop conditions. Returns reason or None."""
    low_format_streak = 0

    for m in metrics:
        # Memory
        peak = m.get("memory_peak_gb", 0)
        if peak > MEMORY_PEAK_LIMIT_GB:
            return f"memory_peak={peak:.1f}GB > {MEMORY_PEAK_LIMIT_GB}GB"

        # Completion length
        mean_len = m.get("mean_completion_len", 0)
        if mean_len > MAX_COMPLETION_LEN:
            return f"mean_completion_len={mean_len:.0f} > {MAX_COMPLETION_LEN}"

        # Loss NaN/Inf
        loss = m.get("loss", 0)
        if loss is None or math.isnan(loss) or math.isinf(loss):
            return f"loss_nan_inf={loss}"

        # Loss magnitude
        if abs(loss) > LOSS_MAGNITUDE_LIMIT:
            return f"loss_magnitude={loss:.2f} > {LOSS_MAGNITUDE_LIMIT}"

        # Format compliance streak
        fmt = m.get("format_compliance_rate", 1.0)
        if fmt < FORMAT_COMPLIANCE_FLOOR:
            low_format_streak += 1
        else:
            low_format_streak = 0
        if low_format_streak >= 3:
            return f"format_compliance<{FORMAT_COMPLIANCE_FLOOR} for 3 consecutive steps"

        # Step time
        step_time = m.get("time_total", 0)
        if step_time > STEP_TIME_LIMIT_S:
            return f"step_time={step_time:.0f}s > {STEP_TIME_LIMIT_S}s"

    return None


def compute_score(metrics: list[dict]) -> dict:
    """Compute composite wordle_score from the last WINDOW steps."""
    window = metrics[-WINDOW:] if len(metrics) >= WINDOW else metrics
    n = len(window)

    # Component averages
    win_rate = sum(m.get("win_rate", 0) for m in window) / n
    violation_rate = sum(m.get("constraint_violation_rate", 1) for m in window) / n
    format_compliance = sum(m.get("format_compliance_rate", 0) for m in window) / n
    mean_completion_len = sum(m.get("mean_completion_len", 200) for m in window) / n
    memory_peak = max(m.get("memory_peak_gb", 0) for m in window)

    # Length efficiency: 1.0 at 0 tokens, 0.0 at 200+ tokens
    length_efficiency = max(0.0, 1.0 - mean_completion_len / 200.0)

    # Reward trend: linear regression slope over window
    rewards = [m.get("mean_reward", 0) for m in window]
    if n >= 2:
        x_mean = (n - 1) / 2.0
        y_mean = sum(rewards) / n
        num = sum((i - x_mean) * (r - y_mean) for i, r in enumerate(rewards))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0
        reward_trend = max(0.0, min(1.0, slope * 10 + 0.5))
    else:
        reward_trend = 0.5

    # Wall time
    wall_time_min = sum(m.get("time_total", 0) for m in metrics) / 60.0

    # Composite score
    score = (
        0.40 * win_rate
        + 0.20 * (1.0 - violation_rate)
        + 0.15 * format_compliance
        + 0.15 * length_efficiency
        + 0.10 * reward_trend
    )

    return {
        "score": round(score, 4),
        "win_rate": round(win_rate, 4),
        "violation_rate": round(violation_rate, 4),
        "format_compliance": round(format_compliance, 4),
        "mean_completion_len": round(mean_completion_len, 1),
        "memory_peak_gb": round(memory_peak, 2),
        "wall_time_min": round(wall_time_min, 1),
        "length_efficiency": round(length_efficiency, 4),
        "reward_trend": round(reward_trend, 4),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m wordle.autoresearch.evaluate_run <training_log.jsonl>", file=sys.stderr)
        sys.exit(2)

    log_path = Path(sys.argv[1])
    metrics = load_metrics(log_path)

    if not metrics:
        print("No training data found", file=sys.stderr)
        sys.exit(2)

    # Check early-stop conditions
    early_stop_reason = check_early_stop(metrics)
    if early_stop_reason:
        # Still compute score for logging
        result = compute_score(metrics)
        status = f"EARLY_STOPPED:{early_stop_reason}"
        print(
            f"{result['score']}\t{result['win_rate']}\t{result['violation_rate']}\t"
            f"{result['format_compliance']}\t{result['mean_completion_len']}\t"
            f"{result['memory_peak_gb']}\t{result['wall_time_min']}\t{status}"
        )
        print(f"Early stopped: {early_stop_reason}", file=sys.stderr)
        sys.exit(1)

    # Normal completion
    result = compute_score(metrics)
    status = "COMPLETED"
    print(
        f"{result['score']}\t{result['win_rate']}\t{result['violation_rate']}\t"
        f"{result['format_compliance']}\t{result['mean_completion_len']}\t"
        f"{result['memory_peak_gb']}\t{result['wall_time_min']}\t{status}"
    )


if __name__ == "__main__":
    main()
