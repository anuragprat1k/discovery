"""Bug injection pipeline for HumanEval problems.

Takes working solutions from HumanEval and programmatically injects single bugs.
Verifies each bug actually breaks at least one test before accepting it.

Usage:
    python -m code_repair.data.bug_injection [--output_dir code_repair/data/problems] [--seed 42]
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Bug injection transforms
# ---------------------------------------------------------------------------

def _inject_off_by_one(source: str, rng: random.Random) -> str | None:
    """range(n) -> range(n-1) or range(n+1), or ±1 to numeric literals in range."""
    # Find range(...) calls
    pattern = r'range\(([^,\)]+)\)'
    matches = list(re.finditer(pattern, source))
    if not matches:
        return None
    match = rng.choice(matches)
    arg = match.group(1).strip()
    delta = rng.choice([" - 1", " + 1"])
    new_arg = f"{arg}{delta}"
    return source[:match.start(1)] + new_arg + source[match.end(1):]


def _inject_wrong_comparison(source: str, rng: random.Random) -> str | None:
    """Swap a comparison operator: <= -> <, >= -> >, == -> !=, < -> <=, > -> >=."""
    swaps = {
        "<=": "<", ">=": ">", "==": "!=",
        "<": "<=", ">": ">=", "!=": "==",
    }
    # Find all comparison operators (longest first to avoid partial matches)
    candidates = []
    for op in ["<=", ">=", "==", "!=", "<", ">"]:
        for m in re.finditer(re.escape(op), source):
            # Skip if part of a longer operator we already matched
            start, end = m.start(), m.end()
            # Check it's not inside a string literal (rough heuristic)
            line_start = source.rfind("\n", 0, start) + 1
            line = source[line_start:source.find("\n", start)]
            if line.count('"') % 2 == 1 or line.count("'") % 2 == 1:
                continue
            # Skip -> arrows and <=>/>=  already covered
            if op in ("<", ">") and (start > 0 and source[start - 1] in "<=!>"):
                continue
            if op in ("<", ">") and (end < len(source) and source[end] == "="):
                continue
            candidates.append((start, end, op))
    if not candidates:
        return None
    start, end, op = rng.choice(candidates)
    return source[:start] + swaps[op] + source[end:]


def _inject_wrong_operator(source: str, rng: random.Random) -> str | None:
    """Swap arithmetic operator: + -> -, * -> +, // -> /, - -> +."""
    # Try specific patterns
    swaps_patterns = [
        (r'(?<!\+)\+(?!\+|=)', '-'),   # + -> - (not += or ++)
        (r'(?<!-)-(?!-|=|>)', '+'),     # - -> + (not -=, --, ->)
        (r'\*(?!\*|=)', '+'),            # * -> + (not ** or *=)
        (r'//', '/'),                     # // -> /
    ]
    candidates = []
    for pattern, replacement in swaps_patterns:
        for m in re.finditer(pattern, source):
            # Skip if in string or comment
            line_start = source.rfind("\n", 0, m.start()) + 1
            line_prefix = source[line_start:m.start()]
            if '#' in line_prefix:
                continue
            candidates.append((m.start(), m.end(), replacement))
    if not candidates:
        return None
    start, end, replacement = rng.choice(candidates)
    return source[:start] + replacement + source[end:]


def _inject_missing_return(source: str, rng: random.Random) -> str | None:
    """Delete one return statement (replace with pass)."""
    # Find return statements
    pattern = r'^(\s*)return\b.*$'
    matches = list(re.finditer(pattern, source, re.MULTILINE))
    if len(matches) < 2:
        # Only delete if there are at least 2 returns (keep at least one path)
        return None
    match = rng.choice(matches)
    indent = match.group(1)
    return source[:match.start()] + f"{indent}pass" + source[match.end():]


def _inject_wrong_init(source: str, rng: random.Random) -> str | None:
    """Change initial value: 0 -> 1, [] -> None, True -> False, "" -> " "."""
    swaps = [
        (r'= 0\b', '= 1'),
        (r'= 1\b', '= 0'),
        (r'= \[\]', '= None'),
        (r'= True\b', '= False'),
        (r'= False\b', '= True'),
        (r"= ''", "= ' '"),
        (r'= ""', '= " "'),
        (r'= None\b', '= 0'),
    ]
    candidates = []
    for pattern, replacement in swaps:
        for m in re.finditer(pattern, source):
            candidates.append((m.start(), m.end(), replacement))
    if not candidates:
        return None
    start, end, replacement = rng.choice(candidates)
    return source[:start] + replacement + source[end:]


def _inject_edge_case_removal(source: str, rng: random.Random) -> str | None:
    """Delete an if-guard / boundary check (the if line + its body)."""
    # Find simple if statements that look like guards
    lines = source.split("\n")
    guard_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for short if statements that are likely guards
        if (stripped.startswith("if ") and stripped.endswith(":")
                and len(stripped) < 80
                and any(kw in stripped for kw in ["== 0", "== 1", "is None",
                        "not ", "len(", "<= 0", "<= 1", "== []", '== ""',
                        "== ''", ">= ", "<= "])):
            guard_indices.append(i)
    if not guard_indices:
        return None
    idx = rng.choice(guard_indices)
    guard_line = lines[idx]
    indent = len(guard_line) - len(guard_line.lstrip())

    # Find the body of this if (lines with greater indentation)
    body_end = idx + 1
    while body_end < len(lines):
        line = lines[body_end]
        if line.strip() == "":
            body_end += 1
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent > indent:
            body_end += 1
        else:
            break

    # Remove the if + body
    new_lines = lines[:idx] + lines[body_end:]
    result = "\n".join(new_lines)
    # Make sure we didn't create empty function
    if result.strip() == "" or result.strip().endswith(":"):
        return None
    return result


BUG_INJECTORS: list[tuple[str, Callable]] = [
    ("off_by_one", _inject_off_by_one),
    ("wrong_comparison", _inject_wrong_comparison),
    ("wrong_operator", _inject_wrong_operator),
    ("missing_return", _inject_missing_return),
    ("wrong_init", _inject_wrong_init),
    ("edge_case_removal", _inject_edge_case_removal),
]


# ---------------------------------------------------------------------------
# Test execution helpers
# ---------------------------------------------------------------------------

def _run_code(code: str, timeout: int = 10) -> tuple[bool, str]:
    """Run code in subprocess, return (success, stderr)."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode == 0, result.stderr


def _count_assertions(test_code: str) -> int:
    """Count assert statements in test code."""
    return len(re.findall(r'\bassert\b', test_code))


def _make_runnable(prompt: str, solution: str, test_code: str, entry_point: str) -> str:
    """Combine prompt + solution + test code into a runnable script."""
    return f"{prompt}{solution}\n{test_code}\ncheck({entry_point})\n"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def inject_bug(
    prompt: str,
    canonical_solution: str,
    test_code: str,
    entry_point: str,
    rng: random.Random,
    max_attempts: int = 20,
) -> dict | None:
    """Try to inject a bug into a HumanEval solution.

    Returns dict with buggy_code and bug_type, or None if no valid bug found.
    """
    full_solution = prompt + canonical_solution

    # Verify canonical solution passes
    runnable = _make_runnable(prompt, canonical_solution, test_code, entry_point)
    try:
        ok, stderr = _run_code(runnable)
    except subprocess.TimeoutExpired:
        return None
    if not ok:
        return None

    # Shuffle bug types and try each
    injectors = list(BUG_INJECTORS)
    rng.shuffle(injectors)

    for attempt in range(max_attempts):
        bug_type, injector = injectors[attempt % len(injectors)]
        buggy_full = injector(full_solution, rng)
        if buggy_full is None:
            continue
        # Verify it's syntactically valid
        try:
            ast.parse(buggy_full)
        except SyntaxError:
            continue

        # Verify bug breaks at least one test
        buggy_runnable = f"{buggy_full}\n{test_code}\ncheck({entry_point})\n"
        try:
            ok, stderr = _run_code(buggy_runnable)
        except subprocess.TimeoutExpired:
            # Timeout counts as "breaks tests" — the bug made it hang
            return {
                "buggy_code": buggy_full,
                "bug_type": bug_type,
            }
        if not ok:
            return {
                "buggy_code": buggy_full,
                "bug_type": bug_type,
            }
        # Bug didn't break tests, try again

    return None


def process_humaneval(
    seed: int = 42,
    sft_count: int = 50,
    train_count: int = 100,
) -> dict[str, list[dict]]:
    """Process HumanEval problems into train/sft/eval splits with injected bugs.

    Returns dict with keys 'sft', 'train', 'eval'.
    """
    from human_eval.data import read_problems

    rng = random.Random(seed)
    problems = read_problems()
    task_ids = sorted(problems.keys())
    rng.shuffle(task_ids)

    results = {"sft": [], "train": [], "eval": []}
    processed = 0

    for task_id in task_ids:
        p = problems[task_id]
        prompt = p["prompt"]
        canonical = p["canonical_solution"]
        test_code = p["test"]
        entry_point = p["entry_point"]

        result = inject_bug(prompt, canonical, test_code, entry_point, rng)
        if result is None:
            continue

        num_tests = _count_assertions(test_code)
        record = {
            "task_id": task_id,
            "prompt": prompt,
            "buggy_code": result["buggy_code"],
            "canonical_solution": prompt + canonical,
            "test_code": test_code,
            "entry_point": entry_point,
            "bug_type": result["bug_type"],
            "num_tests": num_tests,
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
        print(f"  [{processed}] {task_id}: {result['bug_type']} "
              f"({num_tests} tests) -> {record['split']}")

    print(f"\nProcessed {processed} problems: "
          f"{len(results['sft'])} sft, {len(results['train'])} train, "
          f"{len(results['eval'])} eval")
    return results


def main():
    parser = argparse.ArgumentParser(description="Inject bugs into HumanEval problems.")
    parser.add_argument("--output_dir", type=str, default="code_repair/data/problems")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft_count", type=int, default=50)
    parser.add_argument("--train_count", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = process_humaneval(
        seed=args.seed,
        sft_count=args.sft_count,
        train_count=args.train_count,
    )

    for split_name, records in results.items():
        path = output_dir / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Wrote {len(records)} problems to {path}")

    # Also write all problems combined
    all_records = results["sft"] + results["train"] + results["eval"]
    all_path = output_dir / "all.json"
    with open(all_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Wrote {len(all_records)} total problems to {all_path}")


if __name__ == "__main__":
    main()
