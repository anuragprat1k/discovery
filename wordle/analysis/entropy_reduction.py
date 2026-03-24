"""Search Space Entropy Reduction analysis.

Computes remaining search space entropy H_t = log2(|W_t|) at each turn t,
where W_t is the set of GRPO-vocabulary words consistent with all feedback
received through turn t.

Reads trajectory JSON files saved during training (in results/<run>/trajectories/)
and outputs a CSV with columns: step, turn, model_type, seed, target,
remaining_candidates, remaining_entropy.

Usage:
    python -m wordle.analysis.entropy_reduction \
        --results_dirs results/sparse_sft100 results/potential_sft100 results/dense_proxy_sft100 \
        --model_types sparse greens_only full_dense \
        --output entropy_reduction.csv

    python -m wordle.analysis.entropy_reduction \
        --results_dir wordle/autoresearch/results \
        --auto \
        --output entropy_reduction.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

from wordle.data.split_words import get_word_splits
from wordle.environment.feedback import compute_feedback
from wordle.environment.wordle_env import _extract_guess
from wordle.rewards.reward_utils import compute_candidate_set


def get_grpo_vocab() -> list[str]:
    """Return the 1,537-word GRPO training vocabulary."""
    splits = get_word_splits(seed=42)
    return splits["grpo"]


def extract_guesses_from_messages(messages: list[dict], target: str) -> list[str]:
    """Extract valid 5-letter guesses from trajectory messages.

    Parses <guess>WORD</guess> tags from assistant messages.
    Returns list of lowercase guesses in order.
    """
    guesses = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        guess = _extract_guess(msg.get("content", ""))
        if guess is not None:
            guesses.append(guess)
    return guesses


def compute_entropy_trajectory(
    guesses: list[str],
    target: str,
    vocab: list[str],
) -> list[dict]:
    """Compute remaining entropy at each turn (0 through len(guesses)).

    Turn 0 = before any guesses (full vocab).
    Turn t = after t guesses have been made and feedback received.

    Returns list of dicts with keys: turn, remaining_candidates, remaining_entropy.
    """
    rows = []

    # Turn 0: full vocabulary
    n = len(vocab)
    rows.append({
        "turn": 0,
        "remaining_candidates": n,
        "remaining_entropy": math.log2(n) if n > 0 else 0.0,
    })

    # Accumulate guesses and feedback
    all_guesses = []
    all_feedbacks = []

    for t, guess in enumerate(guesses, start=1):
        # Skip invalid guesses (not 5 letters)
        if len(guess) != 5:
            continue
        feedback = compute_feedback(guess, target)
        all_guesses.append(guess)
        all_feedbacks.append(feedback)

        candidates = compute_candidate_set(vocab, all_guesses, all_feedbacks)
        n = len(candidates)
        rows.append({
            "turn": t,
            "remaining_candidates": n,
            "remaining_entropy": math.log2(n) if n > 0 else 0.0,
        })

    return rows


def process_trajectory_file(
    traj_path: Path,
    model_type: str,
    seed: str,
    vocab: list[str],
) -> list[dict]:
    """Process one trajectory JSON file and return entropy rows."""
    with open(traj_path) as f:
        data = json.load(f)

    step = data.get("step", 0)
    rows = []

    for episode in data.get("episodes", []):
        target = episode["target"].lower()
        messages = episode.get("messages", [])
        guesses = extract_guesses_from_messages(messages, target)

        if not guesses:
            continue

        entropy_rows = compute_entropy_trajectory(guesses, target, vocab)
        for row in entropy_rows:
            rows.append({
                "step": step,
                "turn": row["turn"],
                "model_type": model_type,
                "seed": seed,
                "target": target,
                "remaining_candidates": row["remaining_candidates"],
                "remaining_entropy": round(row["remaining_entropy"], 4),
            })

    return rows


def find_trajectory_files(results_dir: Path) -> list[Path]:
    """Find all trajectory JSON files in a results directory."""
    traj_dir = results_dir / "trajectories"
    if not traj_dir.exists():
        return []
    return sorted(traj_dir.glob("step_*.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute search space entropy reduction from Wordle trajectories.",
    )
    parser.add_argument(
        "--results_dirs", nargs="+", type=str, default=None,
        help="Paths to individual results directories (each with trajectories/ subdir).",
    )
    parser.add_argument(
        "--model_types", nargs="+", type=str, default=None,
        help="Model type labels corresponding to --results_dirs (same order).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=str, default=None,
        help="Seed labels corresponding to --results_dirs. Defaults to '42' for all.",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Parent directory containing multiple run subdirectories (use with --auto).",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-discover runs under --results_dir. Model type inferred from dir name.",
    )
    parser.add_argument(
        "--output", type=str, default="entropy_reduction.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=None,
        help="Only process these specific steps (default: all).",
    )
    return parser.parse_args()


def infer_model_type(dirname: str) -> str:
    """Infer model type from directory name."""
    name = dirname.lower()
    if "sparse" in name:
        return "sparse"
    elif "potential" in name or "entropy" in name:
        return "potential"
    elif "reduction" in name:
        return "reduction"
    elif "dense" in name or "proxy" in name:
        return "dense_proxy"
    else:
        return dirname


def infer_seed(dirname: str) -> str:
    """Try to extract seed from directory name."""
    match = re.search(r"seed[_-]?(\d+)", dirname, re.IGNORECASE)
    if match:
        return match.group(1)
    return "42"


def main():
    args = parse_args()
    vocab = get_grpo_vocab()
    print(f"GRPO vocabulary size: {len(vocab)}")
    print(f"Initial entropy H_0 = log2({len(vocab)}) = {math.log2(len(vocab)):.4f} bits")

    all_rows = []

    if args.auto and args.results_dir:
        # Auto-discover runs
        parent = Path(args.results_dir)
        run_dirs = sorted([d for d in parent.iterdir() if d.is_dir()])

        for run_dir in run_dirs:
            traj_files = find_trajectory_files(run_dir)
            if not traj_files:
                continue
            model_type = infer_model_type(run_dir.name)
            seed = infer_seed(run_dir.name)
            print(f"  {run_dir.name}: {len(traj_files)} trajectory files -> model_type={model_type}, seed={seed}")

            for traj_path in traj_files:
                if args.steps:
                    # Parse step from filename
                    step_match = re.search(r"step_(\d+)", traj_path.stem)
                    if step_match and int(step_match.group(1)) not in args.steps:
                        continue
                rows = process_trajectory_file(traj_path, model_type, seed, vocab)
                all_rows.extend(rows)

    elif args.results_dirs:
        if not args.model_types or len(args.model_types) != len(args.results_dirs):
            print("Error: --model_types must match --results_dirs in count.", file=sys.stderr)
            sys.exit(1)

        seeds = args.seeds or ["42"] * len(args.results_dirs)
        if len(seeds) != len(args.results_dirs):
            print("Error: --seeds must match --results_dirs in count.", file=sys.stderr)
            sys.exit(1)

        for results_dir, model_type, seed in zip(args.results_dirs, args.model_types, seeds):
            rd = Path(results_dir)
            traj_files = find_trajectory_files(rd)
            print(f"  {rd.name}: {len(traj_files)} trajectory files -> model_type={model_type}, seed={seed}")

            for traj_path in traj_files:
                if args.steps:
                    step_match = re.search(r"step_(\d+)", traj_path.stem)
                    if step_match and int(step_match.group(1)) not in args.steps:
                        continue
                rows = process_trajectory_file(traj_path, model_type, seed, vocab)
                all_rows.extend(rows)
    else:
        print("Error: specify --results_dirs or --results_dir with --auto.", file=sys.stderr)
        sys.exit(1)

    if not all_rows:
        print("No trajectory data found.", file=sys.stderr)
        sys.exit(1)

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["step", "turn", "model_type", "seed", "target",
                   "remaining_candidates", "remaining_entropy"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_episodes = len(set((r["step"], r["model_type"], r["seed"], r["target"]) for r in all_rows))
    print(f"\nWrote {len(all_rows)} rows ({n_episodes} episodes) to {output_path}")


if __name__ == "__main__":
    main()
