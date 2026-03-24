"""Plot search space entropy reduction curves.

Reads the CSV produced by entropy_reduction.py and generates publication-ready
plots showing how quickly each model type reduces search space entropy over turns.

Usage:
    python -m wordle.analysis.plot_entropy --input entropy_reduction.csv --output plots/

Generates:
    - entropy_by_turn.png: Mean entropy vs turn, averaged over all steps/episodes
    - entropy_by_step_turn.png: Faceted by training step (showing learning progression)
    - entropy_reduction_rate.png: Bits gained per turn (delta entropy)
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_COLORS = {
    "sparse": "#e74c3c",       # red
    "greens_only": "#f39c12",  # orange
    "full_dense": "#2ecc71",   # green
    "potential": "#3498db",    # blue
    "reduction": "#9b59b6",    # purple
    "dense_proxy": "#1abc9c",  # teal
}

MODEL_LABELS = {
    "sparse": "Sparse (binary)",
    "greens_only": "Greens-only (potential)",
    "full_dense": "Full Dense (non-potential)",
    "potential": "Potential (entropy)",
    "reduction": "Reduction",
    "dense_proxy": "Dense Proxy",
}


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append({
                "step": int(r["step"]),
                "turn": int(r["turn"]),
                "model_type": r["model_type"],
                "seed": r["seed"],
                "target": r["target"],
                "remaining_candidates": int(r["remaining_candidates"]),
                "remaining_entropy": float(r["remaining_entropy"]),
            })
    return rows


def plot_entropy_by_turn(rows: list[dict], output_dir: Path, step_filter: list[int] | None = None):
    """Mean entropy vs turn, averaged over all matching episodes."""
    # Group by (model_type, turn) -> list of entropies
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        if step_filter and r["step"] not in step_filter:
            continue
        data[r["model_type"]][r["turn"]].append(r["remaining_entropy"])

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_type in sorted(data.keys()):
        turns = sorted(data[model_type].keys())
        means = [np.mean(data[model_type][t]) for t in turns]
        stds = [np.std(data[model_type][t]) for t in turns]

        color = MODEL_COLORS.get(model_type, "#888888")
        label = MODEL_LABELS.get(model_type, model_type)

        ax.plot(turns, means, "-o", color=color, label=label, markersize=4, linewidth=2)
        ax.fill_between(
            turns,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=color,
        )

    h0 = math.log2(1537)
    ax.axhline(y=h0, color="gray", linestyle="--", alpha=0.5, label=f"H₀ = log₂(1537) = {h0:.2f}")

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Remaining Entropy (bits)", fontsize=12)
    title = "Search Space Entropy Reduction by Turn"
    if step_filter:
        title += f" (steps {step_filter})"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks(range(7))
    ax.set_ylim(bottom=-0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = f"_steps{'_'.join(str(s) for s in step_filter)}" if step_filter else ""
    out_path = output_dir / f"entropy_by_turn{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_entropy_reduction_rate(rows: list[dict], output_dir: Path, step_filter: list[int] | None = None):
    """Bits gained per turn (delta entropy between consecutive turns)."""
    # First, build per-episode trajectories
    episodes: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in rows:
        if step_filter and r["step"] not in step_filter:
            continue
        key = (r["model_type"], r["seed"], r["step"], r["target"])
        episodes[key][r["turn"]] = r["remaining_entropy"]

    # Compute delta per turn
    deltas: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, turn_entropy in episodes.items():
        model_type = key[0]
        turns = sorted(turn_entropy.keys())
        for i in range(1, len(turns)):
            t_prev, t_curr = turns[i - 1], turns[i]
            delta = turn_entropy[t_prev] - turn_entropy[t_curr]
            deltas[model_type][t_curr].append(delta)

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_type in sorted(deltas.keys()):
        turns = sorted(deltas[model_type].keys())
        means = [np.mean(deltas[model_type][t]) for t in turns]

        color = MODEL_COLORS.get(model_type, "#888888")
        label = MODEL_LABELS.get(model_type, model_type)

        ax.bar(
            [t + (list(sorted(deltas.keys())).index(model_type) - 1) * 0.25 for t in turns],
            means,
            width=0.25, color=color, label=label, alpha=0.8,
        )

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Entropy Reduction (bits)", fontsize=12)
    title = "Information Gain per Turn"
    if step_filter:
        title += f" (steps {step_filter})"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    suffix = f"_steps{'_'.join(str(s) for s in step_filter)}" if step_filter else ""
    out_path = output_dir / f"entropy_reduction_rate{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_entropy_over_training(rows: list[dict], output_dir: Path, target_turn: int = 3):
    """Entropy at a fixed turn over training steps (learning curve)."""
    # Group by (model_type, seed, step) -> mean entropy at target_turn
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        if r["turn"] != target_turn:
            continue
        data[r["model_type"]][r["step"]].append(r["remaining_entropy"])

    fig, ax = plt.subplots(figsize=(10, 5))

    for model_type in sorted(data.keys()):
        steps = sorted(data[model_type].keys())
        means = [np.mean(data[model_type][s]) for s in steps]

        color = MODEL_COLORS.get(model_type, "#888888")
        label = MODEL_LABELS.get(model_type, model_type)

        ax.plot(steps, means, "-", color=color, label=label, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(f"Remaining Entropy at Turn {target_turn} (bits)", fontsize=12)
    ax.set_title(f"Entropy at Turn {target_turn} Over Training", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"entropy_at_turn{target_turn}_over_training.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot entropy reduction curves.")
    parser.add_argument("--input", type=str, required=True, help="Path to entropy CSV.")
    parser.add_argument("--output", type=str, default="plots", help="Output directory for plots.")
    parser.add_argument("--steps", nargs="+", type=int, default=None,
                        help="Filter to specific training steps for by-turn plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(Path(args.input))
    print(f"Loaded {len(rows)} rows")

    # All steps combined
    print("Plotting entropy by turn (all steps)...")
    plot_entropy_by_turn(rows, output_dir)

    # Filtered steps if specified
    if args.steps:
        print(f"Plotting entropy by turn (steps {args.steps})...")
        plot_entropy_by_turn(rows, output_dir, step_filter=args.steps)

    # Entropy reduction rate
    print("Plotting entropy reduction rate...")
    plot_entropy_reduction_rate(rows, output_dir, step_filter=args.steps)

    # Entropy over training (for turns 1-4)
    for turn in [1, 2, 3, 4]:
        print(f"Plotting entropy at turn {turn} over training...")
        plot_entropy_over_training(rows, output_dir, target_turn=turn)

    print("Done.")


if __name__ == "__main__":
    main()
