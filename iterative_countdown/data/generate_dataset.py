"""Generate Countdown numbers game problem sets for training and evaluation.

Usage as CLI:
    python -m iterative_countdown.data.generate_dataset \
        --n_train 500 --n_eval 100 --seed 42 \
        --output_dir iterative_countdown/data/

Usage as module:
    from iterative_countdown.data.generate_dataset import generate_and_save
    generate_and_save(n_train=500, n_eval=100, seed=42, output_dir="data/")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support running both as a module and as a script
try:
    from iterative_countdown.environment.problem_generator import generate_problems
except ImportError:
    # When run directly, add parent to path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from iterative_countdown.environment.problem_generator import generate_problems


def generate_and_save(
    n_train: int = 500,
    n_eval: int = 100,
    seed: int = 42,
    output_dir: str = ".",
    difficulty_tiers: dict[str, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Generate and save training and evaluation problem sets.

    Args:
        n_train: Number of training problems.
        n_eval: Number of evaluation problems.
        seed: Random seed for reproducibility.
        output_dir: Directory to save JSON files.
        difficulty_tiers: Distribution of difficulties. Defaults to
            {"easy": 0.3, "medium": 0.4, "hard": 0.3}.

    Returns:
        Tuple of (train_problems, eval_problems).
    """
    if difficulty_tiers is None:
        difficulty_tiers = {"easy": 0.3, "medium": 0.4, "hard": 0.3}

    print(f"Generating {n_train} training problems (seed={seed})...")
    train_problems = generate_problems(n_train, seed=seed, difficulty_tiers=difficulty_tiers)

    print(f"Generating {n_eval} evaluation problems (seed={seed + 1000})...")
    eval_problems = generate_problems(n_eval, seed=seed + 1000, difficulty_tiers=difficulty_tiers)

    # Print summary statistics
    for name, problems in [("Train", train_problems), ("Eval", eval_problems)]:
        difficulties = {}
        exact_count = 0
        for p in problems:
            diff = p["difficulty"]
            difficulties[diff] = difficulties.get(diff, 0) + 1
            if p["has_exact_solution"]:
                exact_count += 1
        print(f"\n{name} set ({len(problems)} problems):")
        for diff, count in sorted(difficulties.items()):
            print(f"  {diff}: {count} ({100 * count / len(problems):.0f}%)")
        print(f"  exact solutions: {exact_count} ({100 * exact_count / len(problems):.0f}%)")

    # Save to files
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = out_path / "train_problems.json"
    eval_path = out_path / "eval_problems.json"

    with open(train_path, "w") as f:
        json.dump(train_problems, f, indent=2)
    print(f"\nSaved training problems to {train_path}")

    with open(eval_path, "w") as f:
        json.dump(eval_problems, f, indent=2)
    print(f"Saved evaluation problems to {eval_path}")

    # Save metadata
    metadata = {
        "n_train": n_train,
        "n_eval": n_eval,
        "seed": seed,
        "difficulty_tiers": difficulty_tiers,
    }
    meta_path = out_path / "dataset_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    return train_problems, eval_problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Countdown numbers game problem sets"
    )
    parser.add_argument("--n_train", type=int, default=500, help="Number of training problems")
    parser.add_argument("--n_eval", type=int, default=100, help="Number of eval problems")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="iterative_countdown/data/",
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--easy_frac", type=float, default=0.3, help="Fraction of easy problems"
    )
    parser.add_argument(
        "--medium_frac", type=float, default=0.4, help="Fraction of medium problems"
    )
    parser.add_argument(
        "--hard_frac", type=float, default=0.3, help="Fraction of hard problems"
    )

    args = parser.parse_args()

    difficulty_tiers = {
        "easy": args.easy_frac,
        "medium": args.medium_frac,
        "hard": args.hard_frac,
    }

    generate_and_save(
        n_train=args.n_train,
        n_eval=args.n_eval,
        seed=args.seed,
        output_dir=args.output_dir,
        difficulty_tiers=difficulty_tiers,
    )


if __name__ == "__main__":
    main()
