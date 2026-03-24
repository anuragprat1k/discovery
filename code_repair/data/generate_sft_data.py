"""Generate expert repair trajectories for SFT warmup.

Two strategies:
1. Simple: turn 1 immediately provides the canonical solution (teaches format only)
2. Guided: show buggy code + test results, then provide fix (teaches read-test-fix cycle)

Output format: conversation turns compatible with TRL SFT format.

Usage:
    python -m code_repair.data.generate_sft_data \
        --problems_path code_repair/data/problems/sft.json \
        --output_path code_repair/data/sft_examples.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_repair.env.sandbox import run_tests
from code_repair.env.code_repair_env import (
    SYSTEM_PROMPT,
    format_initial_prompt,
    format_test_results,
)


def generate_simple_examples(problems: list[dict]) -> list[dict]:
    """Generate examples where turn 1 provides the canonical solution.

    Teaches format: model learns to output <repair>...</repair> tags.
    """
    examples = []
    for p in problems:
        # Run tests on buggy code for initial prompt
        initial_results = run_tests(
            p["buggy_code"], p["test_code"], p["entry_point"],
            timeout=5, detailed=True,
        )

        user_msg = format_initial_prompt(
            p["buggy_code"], initial_results, max_turns=4,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Expert completion: canonical solution
        completion = f"<repair>\n{p['canonical_solution']}\n</repair>"

        examples.append({
            "messages": messages,
            "completion": completion,
            "task_id": p["task_id"],
            "strategy": "simple",
        })

    return examples


def generate_guided_examples(problems: list[dict]) -> list[dict]:
    """Generate 2-turn examples: buggy code → test results → fix.

    Teaches the read-test-fix cycle: model sees test feedback, then fixes.
    """
    examples = []
    for p in problems:
        # Run tests on buggy code
        initial_results = run_tests(
            p["buggy_code"], p["test_code"], p["entry_point"],
            timeout=5, detailed=True,
        )

        user_msg = format_initial_prompt(
            p["buggy_code"], initial_results, max_turns=4,
        )

        # Turn 1: model tries (and fails with) the buggy code as-is
        # This simulates a naive first attempt
        turn1_repair = f"<repair>\n{p['buggy_code']}\n</repair>"

        # Turn 1 feedback (same test results since code unchanged)
        turn1_results_str = format_test_results(
            initial_results, turn=1, max_turns=4,
        )

        # Turn 2: model provides the fix
        turn2_completion = f"<repair>\n{p['canonical_solution']}\n</repair>"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": turn1_repair},
            {"role": "user", "content": f"{turn1_results_str}\n\n## Previous Repair\n```python\n{p['buggy_code']}\n```\n\nFix the function:"},
        ]

        examples.append({
            "messages": messages,
            "completion": turn2_completion,
            "task_id": p["task_id"],
            "strategy": "guided",
        })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate SFT warmup data for code repair.")
    parser.add_argument("--problems_path", type=str, default="code_repair/data/problems/sft.json")
    parser.add_argument("--output_path", type=str, default="code_repair/data/sft_examples.json")
    parser.add_argument("--strategy", choices=["simple", "guided", "both"], default="both")
    args = parser.parse_args()

    with open(args.problems_path) as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} SFT problems")

    examples = []
    if args.strategy in ("simple", "both"):
        simple = generate_simple_examples(problems)
        examples.extend(simple)
        print(f"Generated {len(simple)} simple examples")

    if args.strategy in ("guided", "both"):
        guided = generate_guided_examples(problems)
        examples.extend(guided)
        print(f"Generated {len(guided)} guided examples")

    with open(args.output_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Wrote {len(examples)} total examples to {args.output_path}")


if __name__ == "__main__":
    main()
