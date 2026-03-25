"""Multi-bug injection pipeline for HumanEval problems.

Injects 2-3 bugs into each working solution, verifying that different bugs
break different tests. This creates problems where partial fixes yield partial
credit — essential for dense reward to differentiate from sparse.

Also saves the dataset in a multi-turn conversation format for use as
cheap coding SFT/RLHF data.

Usage:
    python -m code_repair.data.multi_bug_injection \
        [--output_dir code_repair/data/problems_multi] \
        [--min_bugs 2] [--max_bugs 3] [--seed 42]
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import subprocess
import sys
from pathlib import Path

from code_repair.data.bug_injection import (
    BUG_INJECTORS,
    _run_code,
    _count_assertions,
    _make_runnable,
)
from code_repair.env.sandbox import run_tests


def _get_failing_tests(code: str, test_code: str, entry_point: str,
                       timeout: int = 10) -> set[str]:
    """Run code and return set of failing test names."""
    results = run_tests(code, test_code, entry_point, timeout=timeout, detailed=True)
    return {r.name for r in results if not r.passed}


def _get_passing_tests(code: str, test_code: str, entry_point: str,
                       timeout: int = 10) -> set[str]:
    """Run code and return set of passing test names."""
    results = run_tests(code, test_code, entry_point, timeout=timeout, detailed=True)
    return {r.name for r in results if r.passed}


def inject_multiple_bugs(
    prompt: str,
    canonical_solution: str,
    test_code: str,
    entry_point: str,
    rng: random.Random,
    min_bugs: int = 2,
    max_bugs: int = 3,
    max_attempts_per_bug: int = 30,
) -> dict | None:
    """Inject multiple bugs into a solution, each breaking different tests.

    Strategy:
    1. Start with canonical solution, get all-passing baseline
    2. Inject bug 1, record which tests it breaks
    3. Inject bug 2 into the already-buggy code, verify it breaks NEW tests
    4. Optionally inject bug 3 same way
    5. Require each bug breaks ≥1 test that previous bugs didn't

    Returns dict with buggy_code, bug_types list, bugs_detail list,
    or None if can't achieve min_bugs distinct failures.
    """
    full_solution = prompt + canonical_solution

    # Verify canonical passes all tests
    canonical_passing = _get_passing_tests(full_solution, test_code, entry_point)
    if not canonical_passing:
        return None

    all_test_names = canonical_passing
    n_tests = len(all_test_names)
    if n_tests < min_bugs + 1:
        # Need enough tests to have different failures per bug
        return None

    target_n_bugs = rng.randint(min_bugs, min(max_bugs, n_tests - 1))

    current_code = full_solution
    bugs_applied = []
    tests_broken_so_far: set[str] = set()
    injectors = list(BUG_INJECTORS)

    for bug_idx in range(target_n_bugs):
        # Try to find a bug that breaks NEW tests
        found = False
        rng.shuffle(injectors)

        for attempt in range(max_attempts_per_bug):
            bug_type, injector = injectors[attempt % len(injectors)]
            candidate = injector(current_code, rng)
            if candidate is None:
                continue

            # Must be syntactically valid
            try:
                ast.parse(candidate)
            except SyntaxError:
                continue

            # Check which tests fail now vs before this bug
            try:
                candidate_failing = _get_failing_tests(
                    candidate, test_code, entry_point, timeout=10,
                )
            except Exception:
                continue

            new_failures = candidate_failing - tests_broken_so_far
            if not new_failures:
                # This bug doesn't break any new tests
                continue

            # Also ensure we didn't accidentally fix a previous bug
            # (all previously broken tests should still be broken)
            if not tests_broken_so_far.issubset(candidate_failing):
                continue

            # Success: this bug breaks new tests
            bugs_applied.append({
                "bug_type": bug_type,
                "bug_index": bug_idx,
                "new_tests_broken": sorted(new_failures),
                "total_tests_broken_after": len(candidate_failing),
            })
            tests_broken_so_far = candidate_failing
            current_code = candidate
            found = True
            break

        if not found:
            break

    if len(bugs_applied) < min_bugs:
        return None

    # Final verification: buggy code fails on some tests, canonical passes all
    final_failing = _get_failing_tests(current_code, test_code, entry_point)
    final_passing = all_test_names - final_failing

    if not final_failing or not final_passing:
        # Either all tests fail (too hard) or none fail (bugs cancelled out)
        return None

    return {
        "buggy_code": current_code,
        "bug_types": [b["bug_type"] for b in bugs_applied],
        "n_bugs": len(bugs_applied),
        "bugs_detail": bugs_applied,
        "n_tests_total": n_tests,
        "n_tests_passing_buggy": len(final_passing),
        "n_tests_failing_buggy": len(final_failing),
        "tests_passing_buggy": sorted(final_passing),
        "tests_failing_buggy": sorted(final_failing),
    }


def process_humaneval_multi(
    seed: int = 42,
    min_bugs: int = 2,
    max_bugs: int = 3,
    sft_count: int = 20,
    train_count: int = 80,
) -> dict[str, list[dict]]:
    """Process HumanEval into multi-bug train/sft/eval splits."""
    from human_eval.data import read_problems

    rng = random.Random(seed)
    problems = read_problems()
    task_ids = sorted(problems.keys())
    rng.shuffle(task_ids)

    results = {"sft": [], "train": [], "eval": []}
    processed = 0
    skipped = 0

    for task_id in task_ids:
        p = problems[task_id]
        prompt = p["prompt"]
        canonical = p["canonical_solution"]
        test_code = p["test"]
        entry_point = p["entry_point"]

        result = inject_multiple_bugs(
            prompt, canonical, test_code, entry_point, rng,
            min_bugs=min_bugs, max_bugs=max_bugs,
        )
        if result is None:
            skipped += 1
            continue

        num_tests = _count_assertions(test_code)
        record = {
            "task_id": task_id,
            "prompt": prompt,
            "buggy_code": result["buggy_code"],
            "canonical_solution": prompt + canonical,
            "test_code": test_code,
            "entry_point": entry_point,
            "bug_types": result["bug_types"],
            "n_bugs": result["n_bugs"],
            "bugs_detail": result["bugs_detail"],
            "num_tests": num_tests,
            "n_tests_passing_buggy": result["n_tests_passing_buggy"],
            "n_tests_failing_buggy": result["n_tests_failing_buggy"],
        }

        # Assign to splits
        if len(results["sft"]) < sft_count:
            record["split"] = "sft"
            results["sft"].append(record)
        elif len(results["train"]) < train_count:
            record["split"] = "train"
            results["train"].append(record)
        else:
            record["split"] = "eval"
            results["eval"].append(record)

        processed += 1
        bug_str = "+".join(result["bug_types"])
        print(f"  [{processed}] {task_id}: {bug_str} "
              f"({result['n_tests_failing_buggy']}/{num_tests} failing, "
              f"{result['n_bugs']} bugs) -> {record['split']}")

    print(f"\nProcessed {processed} problems ({skipped} skipped): "
          f"{len(results['sft'])} sft, {len(results['train'])} train, "
          f"{len(results['eval'])} eval")
    return results


def _build_conversation_format(record: dict) -> dict:
    """Convert a problem record into multi-turn conversation format.

    This is useful as standalone SFT/RLHF data for code repair.
    Format: system prompt, buggy code + tests, then canonical fix.
    """
    from code_repair.env.code_repair_env import SYSTEM_PROMPT, format_initial_prompt
    from code_repair.env.sandbox import run_tests

    initial_results = run_tests(
        record["buggy_code"], record["test_code"], record["entry_point"],
        timeout=5, detailed=True,
    )
    user_msg = format_initial_prompt(
        record["buggy_code"], initial_results, max_turns=1,
    )

    return {
        "task_id": record["task_id"],
        "n_bugs": record["n_bugs"],
        "bug_types": record["bug_types"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"<repair>\n{record['canonical_solution']}\n</repair>"},
        ],
        "num_tests": record["num_tests"],
        "n_tests_failing_buggy": record["n_tests_failing_buggy"],
    }


def main():
    parser = argparse.ArgumentParser(description="Inject multiple bugs into HumanEval.")
    parser.add_argument("--output_dir", type=str, default="code_repair/data/problems_multi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_bugs", type=int, default=2)
    parser.add_argument("--max_bugs", type=int, default=3)
    parser.add_argument("--sft_count", type=int, default=20)
    parser.add_argument("--train_count", type=int, default=80)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = process_humaneval_multi(
        seed=args.seed,
        min_bugs=args.min_bugs,
        max_bugs=args.max_bugs,
        sft_count=args.sft_count,
        train_count=args.train_count,
    )

    for split_name, records in results.items():
        path = output_dir / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Wrote {len(records)} problems to {path}")

    # Write all combined
    all_records = results["sft"] + results["train"] + results["eval"]
    all_path = output_dir / "all.json"
    with open(all_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Wrote {len(all_records)} total problems to {all_path}")

    # Write conversation format (cheap multi-turn coding data)
    print("\nGenerating conversation-format dataset...")
    conversations = []
    for record in all_records:
        try:
            conv = _build_conversation_format(record)
            conversations.append(conv)
        except Exception as e:
            print(f"  Warning: skipped {record['task_id']}: {e}")

    conv_path = output_dir / "conversations.json"
    with open(conv_path, "w") as f:
        json.dump(conversations, f, indent=2)
    print(f"Wrote {len(conversations)} conversations to {conv_path}")


if __name__ == "__main__":
    main()
