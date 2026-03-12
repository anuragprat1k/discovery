"""Pass@k evaluation for the Countdown environment.

Adapted from /workspace/discovery/eval/pass_at_k.py.
Instead of checking \\boxed{} answers, we simulate the Countdown game
and check if the model reaches the target.

Supports:
- Sampling via tinker SamplingClient
- pass@k computation using unbiased estimator from Codex paper
- Per-difficulty breakdown
- Incremental evaluation with resume support
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from ..environment.expression_parser import extract_expression, parse_expression


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased estimator of pass@k from Codex paper.

    Args:
        n: total samples generated for the problem.
        c: number of correct (target-reaching) samples.
        k: target k value.

    Returns:
        Estimated probability that at least one of k samples is correct,
        or None if n < k.
    """
    if n < k:
        return None
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def evaluate_countdown_episode(
    model_outputs: list[str],
    target: int,
    numbers: list[int],
    max_turns: int = 5,
) -> dict:
    """Simulate a Countdown episode from model outputs.

    Replays the model's proposed expressions turn-by-turn, updating the
    available number pool and checking whether the target is reached.

    Args:
        model_outputs: list of model responses (one per turn).
        target: target number to reach.
        numbers: initial available numbers.
        max_turns: maximum turns allowed.

    Returns:
        dict with keys: target_reached, best_distance, turns_used,
        expressions, results.
    """
    available = list(numbers)
    best_distance = abs(target)
    expressions: list[str] = []
    results: list[int] = []
    target_reached = False

    for output in model_outputs[:max_turns]:
        expr_str = extract_expression(output)
        if expr_str is None:
            break

        try:
            result, nums_used = parse_expression(expr_str, available)
            expressions.append(expr_str)
            results.append(result)

            # Update available numbers
            for n in nums_used:
                available.remove(n)
            available.append(result)

            distance = abs(target - result)
            best_distance = min(best_distance, distance)

            if result == target:
                target_reached = True
                break
        except (ValueError, ZeroDivisionError):
            break

    return {
        "target_reached": target_reached,
        "best_distance": best_distance,
        "turns_used": len(expressions),
        "expressions": expressions,
        "results": results,
    }


def score_problem(
    episodes: list[dict],
    k_values: list[int] | None = None,
) -> dict:
    """Score multiple episodes (samples) for one problem.

    Args:
        episodes: list of evaluate_countdown_episode results.
        k_values: k values for pass@k computation.

    Returns:
        dict with pass@k values, best_distance stats, turn stats.
    """
    if k_values is None:
        k_values = [1, 4, 16]

    n = len(episodes)
    c = sum(1 for ep in episodes if ep["target_reached"])

    result: dict = {
        "n_samples": n,
        "n_correct": c,
    }

    for k in k_values:
        pk = pass_at_k(n, c, k)
        if pk is not None:
            result[f"pass@{k}"] = pk

    # Distance statistics
    distances = [ep["best_distance"] for ep in episodes]
    result["mean_best_distance"] = float(np.mean(distances))
    result["min_best_distance"] = int(min(distances))

    # Turn statistics
    turns = [ep["turns_used"] for ep in episodes]
    result["mean_turns"] = float(np.mean(turns))

    return result


def aggregate_results(
    problem_results: list[dict],
    problems: list[dict],
    k_values: list[int] | None = None,
) -> dict:
    """Aggregate per-problem results into overall metrics.

    Args:
        problem_results: list of score_problem outputs, one per problem.
        problems: list of problem dicts (must have same length).
        k_values: k values for pass@k.

    Returns:
        dict with overall pass@k, per-difficulty pass@k, distance stats.
    """
    if k_values is None:
        k_values = [1, 4, 16]

    overall: dict = {}

    # Overall pass@k
    for k in k_values:
        key = f"pass@{k}"
        values = [r[key] for r in problem_results if key in r]
        if values:
            overall[key] = float(np.mean(values))

    # Per-difficulty breakdown
    difficulty_groups: dict[str, list[dict]] = {}
    for prob, result in zip(problems, problem_results):
        diff = prob.get("difficulty", "unknown")
        if diff not in difficulty_groups:
            difficulty_groups[diff] = []
        difficulty_groups[diff].append(result)

    by_difficulty: dict[str, dict] = {}
    for diff, results in difficulty_groups.items():
        by_difficulty[diff] = {}
        for k in k_values:
            key = f"pass@{k}"
            values = [r[key] for r in results if key in r]
            if values:
                by_difficulty[diff][key] = float(np.mean(values))
        by_difficulty[diff]["n_problems"] = len(results)

    overall["by_difficulty"] = by_difficulty
    overall["n_problems"] = len(problem_results)
    overall["mean_best_distance"] = float(
        np.mean([r["mean_best_distance"] for r in problem_results])
    )

    return overall


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Countdown model")
    parser.add_argument(
        "--problems_path",
        type=str,
        required=True,
        help="Path to problems JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=16,
        help="Number of samples per problem",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 4, 16],
        help="k values for pass@k",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Training step (for result file naming)",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=5,
        help="Max turns per episode",
    )
    # Note: actual model sampling would be added when integrating with tinker
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load problems
    with open(args.problems_path) as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems")
    print(f"Will compute pass@k for k={args.k_values}")

    # Placeholder: in real usage, this would sample from model via tinker
    # For now, just save the config
    config = {
        "problems_path": args.problems_path,
        "n_samples": args.n_samples,
        "k_values": args.k_values,
        "step": args.step,
        "max_turns": args.max_turns,
        "n_problems": len(problems),
    }

    output_path = os.path.join(args.output_dir, f"step_{args.step:04d}.json")
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {output_path}")


if __name__ == "__main__":
    main()
