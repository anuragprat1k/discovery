"""Discovery vs Sharpening evaluation metrics.

Compares pre-training and post-training performance to classify problems:
- Discovery: unsolvable before, solvable after (pass@k: 0 -> >0)
- Sharpening: solvable before, higher pass@1 after (pass@1 improved, pass@k same)
- Narrowing: solvable before, lower diversity after (pass@k decreased)
- Lost: solvable before, unsolvable after (pass@k: >0 -> 0)

Multi-turn discovery: problems only solvable in multiple turns (not in 1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def classify_problems(
    baseline_results: list[dict],
    trained_results: list[dict],
    k_discovery: int = 16,
    k_sharpening: int = 1,
    threshold: float = 0.01,
) -> dict:
    """Classify each problem into discovery/sharpening/narrowing/lost.

    Args:
        baseline_results: per-problem results from step 0 (untrained model).
        trained_results: per-problem results from the trained model.
        k_discovery: k value used for discovery detection (high k catches
            any solvability, even rare).
        k_sharpening: k value used for sharpening detection (low k measures
            reliable solvability).
        threshold: minimum change to count as meaningful.

    Returns:
        dict with counts, fractions, and problem_indices for each category.
    """
    categories: dict[str, list[int]] = {
        "discovery": [],
        "sharpening": [],
        "narrowing": [],
        "lost": [],
        "unchanged": [],
    }

    for i, (base, trained) in enumerate(zip(baseline_results, trained_results)):
        base_pass_k = base.get(f"pass@{k_discovery}", 0.0)
        trained_pass_k = trained.get(f"pass@{k_discovery}", 0.0)
        base_pass_1 = base.get(f"pass@{k_sharpening}", 0.0)
        trained_pass_1 = trained.get(f"pass@{k_sharpening}", 0.0)

        if base_pass_k < threshold and trained_pass_k >= threshold:
            categories["discovery"].append(i)
        elif base_pass_k >= threshold and trained_pass_k < threshold:
            categories["lost"].append(i)
        elif base_pass_k >= threshold and trained_pass_1 > base_pass_1 + threshold:
            categories["sharpening"].append(i)
        elif base_pass_k >= threshold and trained_pass_k < base_pass_k - threshold:
            categories["narrowing"].append(i)
        else:
            categories["unchanged"].append(i)

    n_total = len(baseline_results)
    summary = {
        "n_total": n_total,
        "counts": {k: len(v) for k, v in categories.items()},
        "fractions": {k: len(v) / max(n_total, 1) for k, v in categories.items()},
        "problem_indices": categories,
    }

    return summary


def compute_delta_metrics(
    baseline_results: list[dict],
    trained_results: list[dict],
    k_values: list[int] | None = None,
) -> dict:
    """Compute delta (change) metrics between baseline and trained.

    Args:
        baseline_results: per-problem results from untrained model.
        trained_results: per-problem results from trained model.
        k_values: k values for which to compute deltas.

    Returns:
        dict with delta_pass@k for each k, plus distance improvements.
    """
    if k_values is None:
        k_values = [1, 4, 16]

    deltas: dict[str, float] = {}

    for k in k_values:
        key = f"pass@{k}"
        base_vals = [r.get(key, 0.0) for r in baseline_results]
        trained_vals = [r.get(key, 0.0) for r in trained_results]

        base_mean = float(np.mean(base_vals))
        trained_mean = float(np.mean(trained_vals))

        deltas[f"delta_{key}"] = trained_mean - base_mean
        deltas[f"baseline_{key}"] = base_mean
        deltas[f"trained_{key}"] = trained_mean

    # Distance improvement
    base_dist = [r.get("mean_best_distance", float("inf")) for r in baseline_results]
    trained_dist = [r.get("mean_best_distance", float("inf")) for r in trained_results]
    deltas["delta_mean_distance"] = float(np.mean(trained_dist)) - float(
        np.mean(base_dist)
    )

    return deltas


def multi_turn_discovery(
    single_turn_results: list[dict],
    multi_turn_results: list[dict],
    k: int = 16,
    threshold: float = 0.01,
) -> dict:
    """Identify problems only solvable in multi-turn (not single-turn).

    This measures the value of iterative reasoning: problems where
    decomposing the solution across turns is necessary.

    Args:
        single_turn_results: results with max_turns=1.
        multi_turn_results: results with max_turns>1.
        k: k value for solvability check.
        threshold: minimum pass@k to count as solvable.

    Returns:
        dict with multi-turn discovery stats and problem indices.
    """
    multi_turn_only: list[int] = []
    both_solvable: list[int] = []
    neither: list[int] = []
    single_only: list[int] = []

    for i, (st, mt) in enumerate(zip(single_turn_results, multi_turn_results)):
        st_pass = st.get(f"pass@{k}", 0.0)
        mt_pass = mt.get(f"pass@{k}", 0.0)

        if st_pass < threshold and mt_pass >= threshold:
            multi_turn_only.append(i)
        elif st_pass >= threshold and mt_pass >= threshold:
            both_solvable.append(i)
        elif st_pass >= threshold and mt_pass < threshold:
            single_only.append(i)
        else:
            neither.append(i)

    n = len(single_turn_results)
    return {
        "multi_turn_only": len(multi_turn_only),
        "both_solvable": len(both_solvable),
        "single_only": len(single_only),
        "neither": len(neither),
        "multi_turn_discovery_rate": len(multi_turn_only) / max(n, 1),
        "problem_indices": {
            "multi_turn_only": multi_turn_only,
            "both_solvable": both_solvable,
            "single_only": single_only,
            "neither": neither,
        },
    }


def load_and_compare(
    baseline_path: str,
    trained_path: str,
    k_values: list[int] | None = None,
) -> dict:
    """Load two result files and compute all comparison metrics.

    Args:
        baseline_path: path to JSON results from the untrained model.
        trained_path: path to JSON results from the trained model.
        k_values: k values for pass@k computation.

    Returns:
        dict with classification and delta metrics.
    """
    if k_values is None:
        k_values = [1, 4, 16]

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(trained_path) as f:
        trained = json.load(f)

    baseline_results = baseline.get("per_problem", baseline.get("results", []))
    trained_results = trained.get("per_problem", trained.get("results", []))

    classification = classify_problems(baseline_results, trained_results)
    deltas = compute_delta_metrics(baseline_results, trained_results, k_values)

    return {
        "classification": classification,
        "deltas": deltas,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute discovery/sharpening metrics"
    )
    parser.add_argument(
        "--baseline", type=str, required=True, help="Baseline results JSON"
    )
    parser.add_argument(
        "--trained", type=str, required=True, help="Trained results JSON"
    )
    parser.add_argument("--output", type=str, help="Output path for comparison")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 4, 16])
    args = parser.parse_args()

    results = load_and_compare(args.baseline, args.trained, args.k_values)

    print("\n=== Discovery/Sharpening Classification ===")
    for category, count in results["classification"]["counts"].items():
        frac = results["classification"]["fractions"][category]
        print(f"  {category}: {count} ({frac:.1%})")

    print("\n=== Delta Metrics ===")
    for key, value in results["deltas"].items():
        print(f"  {key}: {value:+.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")
