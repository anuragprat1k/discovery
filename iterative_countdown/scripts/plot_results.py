"""Plot training results and comparisons across reward types.

Adapted from /workspace/discovery/analysis/plot_trajectories.py.

Generates:
1. Training curves (reward, distance over steps)
2. Delta space plot (discovery vs sharpening quadrants)
3. Per-difficulty breakdown
4. Reward type comparison
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run_data(results_dir: str) -> list[dict]:
    """Load all step_*.json files from a results directory.

    Args:
        results_dir: path to directory containing step_NNNN.json files.

    Returns:
        list of dicts sorted by step number.
    """
    results_path = Path(results_dir)
    files = sorted(results_path.glob("step_*.json"))
    data = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        # Extract step number from filename
        step = int(f.stem.split("_")[1])
        d["step"] = step
        data.append(d)
    return data


def plot_training_curves(
    runs: dict[str, list[dict]],
    output_path: str,
    k_values: list[int] | None = None,
) -> None:
    """Plot pass@k and distance over training steps for each run.

    Args:
        runs: mapping from run name to list of per-step result dicts.
        output_path: file path for the saved figure.
        k_values: k values to plot.
    """
    if k_values is None:
        k_values = [1, 4, 16]

    n_metrics = len(k_values) + 1  # +1 for distance
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    colors = {"binary": "tab:red", "dense": "tab:blue", "prime": "tab:green"}

    for name, data in runs.items():
        steps = [d["step"] for d in data]
        color = colors.get(name, "tab:gray")

        for i, k in enumerate(k_values):
            key = f"pass@{k}"
            values = [d.get(key, 0.0) for d in data]
            axes[i].plot(
                steps, values, label=name, color=color, marker="o", markersize=3
            )
            axes[i].set_title(f"pass@{k}")
            axes[i].set_xlabel("Step")
            axes[i].set_ylabel(f"pass@{k}")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        # Distance
        distances = [d.get("mean_best_distance", 0.0) for d in data]
        axes[-1].plot(
            steps, distances, label=name, color=color, marker="o", markersize=3
        )
        axes[-1].set_title("Mean Best Distance")
        axes[-1].set_xlabel("Step")
        axes[-1].set_ylabel("Distance to Target")
        axes[-1].legend()
        axes[-1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_delta_space(
    runs: dict[str, dict],
    output_path: str,
) -> None:
    """Plot in delta space (delta-pass@1 vs delta-pass@16) with quadrant labels.

    The four quadrants represent different training dynamics:
    - Top-right: Discovery (+pass@1, +pass@16)
    - Bottom-right: Sharpening (+pass@1, -pass@16)
    - Top-left: Expansion (-pass@1, +pass@16)
    - Bottom-left: Regression (-pass@1, -pass@16)

    Args:
        runs: mapping from run name to delta metrics dict.
        output_path: file path for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {"binary": "tab:red", "dense": "tab:blue", "prime": "tab:green"}

    for name, deltas in runs.items():
        dx = deltas.get("delta_pass@1", 0.0)
        dy = deltas.get("delta_pass@16", 0.0)
        color = colors.get(name, "tab:gray")
        ax.scatter(dx, dy, s=100, color=color, label=name, zorder=5)
        ax.annotate(
            name,
            (dx, dy),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=9,
        )

    # Quadrant dividers
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.text(
        xlim[1] * 0.7,
        ylim[1] * 0.7,
        "Discovery\n(+pass@1, +pass@16)",
        ha="center",
        fontsize=9,
        color="green",
        alpha=0.7,
    )
    ax.text(
        xlim[1] * 0.7,
        ylim[0] * 0.7,
        "Sharpening\n(+pass@1, -pass@16)",
        ha="center",
        fontsize=9,
        color="orange",
        alpha=0.7,
    )
    ax.text(
        xlim[0] * 0.7,
        ylim[1] * 0.7,
        "Expansion\n(-pass@1, +pass@16)",
        ha="center",
        fontsize=9,
        color="blue",
        alpha=0.7,
    )
    ax.text(
        xlim[0] * 0.7,
        ylim[0] * 0.7,
        "Regression\n(-pass@1, -pass@16)",
        ha="center",
        fontsize=9,
        color="red",
        alpha=0.7,
    )

    ax.set_xlabel("Δpass@1")
    ax.set_ylabel("Δpass@16")
    ax.set_title("Training Direction in Delta Space")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved delta space plot to {output_path}")


def plot_difficulty_breakdown(
    runs: dict[str, dict],
    output_path: str,
    k: int = 16,
) -> None:
    """Bar chart of pass@k by difficulty for each run.

    Args:
        runs: mapping from run name to aggregated results dict
            (must contain "by_difficulty" key).
        output_path: file path for the saved figure.
        k: which pass@k to plot.
    """
    difficulties = ["easy", "medium", "hard"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(difficulties))
    width = 0.25
    colors = {"binary": "tab:red", "dense": "tab:blue", "prime": "tab:green"}

    for i, (name, data) in enumerate(runs.items()):
        by_diff = data.get("by_difficulty", {})
        values = [by_diff.get(d, {}).get(f"pass@{k}", 0.0) for d in difficulties]
        color = colors.get(name, "tab:gray")
        ax.bar(x + i * width, values, width, label=name, color=color)

    ax.set_xlabel("Difficulty")
    ax.set_ylabel(f"pass@{k}")
    ax.set_title(f"pass@{k} by Difficulty Level")
    ax.set_xticks(x + width)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved difficulty breakdown to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Countdown RL results")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Base results directory (with subdirs per run)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: results_dir/plots)",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=["binary", "dense", "prime"],
        help="Run names to plot",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Load all runs
    all_runs: dict[str, list[dict]] = {}
    for run_name in args.runs:
        run_dir = os.path.join(args.results_dir, run_name)
        if os.path.exists(run_dir):
            all_runs[run_name] = load_run_data(run_dir)
            print(f"Loaded {len(all_runs[run_name])} steps for {run_name}")

    if not all_runs:
        print("No runs found!")
        return

    # Plot training curves
    plot_training_curves(
        all_runs, os.path.join(output_dir, "training_curves.png")
    )

    print("Done!")


if __name__ == "__main__":
    main()
