"""Evaluate code repair model on held-out problems.

Runs single-turn code repair: model sees buggy code + test results, outputs
<repair>. Reports pass@1, pass@k, format compliance, unique problems solved.

Can be used standalone or called from the training callback.

Usage:
    python -m code_repair.eval \
        --problems_path code_repair/data/problems/eval.json \
        --model_name Qwen/Qwen3-4B-Instruct-2507 \
        --n_samples 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code_repair.env.sandbox import run_tests
from code_repair.env.code_repair_env import (
    SYSTEM_PROMPT,
    extract_repair,
    format_initial_prompt,
)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def evaluate(
    model,
    tokenizer,
    problems: list[dict],
    n_samples: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    sandbox_timeout: int = 5,
    verbose: bool = True,
) -> dict:
    """Run evaluation on a set of problems.

    Args:
        model: HF model (can be PEFT-wrapped).
        tokenizer: Matching tokenizer.
        problems: List of problem dicts with buggy_code, test_code, etc.
        n_samples: Number of generation attempts per problem.
        max_new_tokens: Max tokens per generation.
        temperature: Sampling temperature.
        sandbox_timeout: Timeout for sandbox test execution.
        verbose: Print per-problem results.

    Returns:
        Dict with aggregate metrics and per-problem results.
    """
    device = next(model.parameters()).device
    per_problem = {}
    total_solved = 0
    total_samples = 0
    total_format_ok = 0
    unique_solved = set()

    for prob_idx, p in enumerate(problems):
        task_id = p["task_id"]
        initial_results = run_tests(
            p["buggy_code"], p["test_code"], p["entry_point"],
            timeout=sandbox_timeout, detailed=True,
        )
        user_msg = format_initial_prompt(
            p["buggy_code"], initial_results, max_turns=1,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt",
        ).to(device)

        n_solved = 0
        n_format_ok = 0
        tests_passing_counts = []

        for _ in range(n_samples):
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                )
            completion = tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True,
            )

            repair = extract_repair(completion)
            if repair is None:
                tests_passing_counts.append(0)
                continue

            n_format_ok += 1
            results = run_tests(
                repair, p["test_code"], p["entry_point"],
                timeout=sandbox_timeout, detailed=True,
            )
            passed = sum(1 for r in results if r.passed)
            total_tests = len(results)
            tests_passing_counts.append(passed)

            all_passed = (
                passed == total_tests and total_tests > 0
                and not (total_tests == 1 and results[0].name in
                         ("timeout", "syntax_error", "runtime_error"))
            )
            if all_passed:
                n_solved += 1
                unique_solved.add(task_id)

        p1 = pass_at_k(n_samples, n_solved, 1)
        pk = pass_at_k(n_samples, n_solved, min(n_samples, 8))
        format_rate = n_format_ok / n_samples if n_samples > 0 else 0
        avg_tests_passing = np.mean(tests_passing_counts) if tests_passing_counts else 0

        per_problem[task_id] = {
            "n_solved": n_solved,
            "n_samples": n_samples,
            "n_format_ok": n_format_ok,
            "pass_at_1": p1,
            "pass_at_k": pk,
            "format_rate": format_rate,
            "avg_tests_passing": float(avg_tests_passing),
            "num_tests": p.get("num_tests", len(initial_results)),
            "bug_type": p.get("bug_type", "unknown"),
        }

        total_solved += n_solved
        total_samples += n_samples
        total_format_ok += n_format_ok

        if verbose:
            print(f"  [{prob_idx+1}/{len(problems)}] {task_id} ({p.get('bug_type','')}): "
                  f"{n_solved}/{n_samples} solved, format={n_format_ok}/{n_samples}, "
                  f"pass@1={p1:.3f}")

    # Aggregate
    overall_pass1 = float(np.mean([r["pass_at_1"] for r in per_problem.values()]))
    overall_passk = float(np.mean([r["pass_at_k"] for r in per_problem.values()]))
    overall_format = total_format_ok / total_samples if total_samples > 0 else 0
    n_unique_solved = len(unique_solved)

    # Per bug-type breakdown
    by_bug_type = {}
    for info in per_problem.values():
        bt = info["bug_type"]
        if bt not in by_bug_type:
            by_bug_type[bt] = {"solved": 0, "total": 0, "pass1_sum": 0, "count": 0}
        by_bug_type[bt]["solved"] += info["n_solved"]
        by_bug_type[bt]["total"] += info["n_samples"]
        by_bug_type[bt]["pass1_sum"] += info["pass_at_1"]
        by_bug_type[bt]["count"] += 1
    for bt in by_bug_type:
        by_bug_type[bt]["pass_at_1"] = by_bug_type[bt]["pass1_sum"] / by_bug_type[bt]["count"]

    return {
        "pass_at_1": overall_pass1,
        "pass_at_k": overall_passk,
        "format_rate": overall_format,
        "unique_solved": n_unique_solved,
        "unique_solved_frac": n_unique_solved / len(problems) if problems else 0,
        "total_solved": total_solved,
        "total_samples": total_samples,
        "raw_solve_rate": total_solved / total_samples if total_samples > 0 else 0,
        "n_problems": len(problems),
        "by_bug_type": by_bug_type,
        "per_problem": per_problem,
    }


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate code repair model.")
    parser.add_argument("--problems_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint directory.")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--sandbox_timeout", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/code_repair")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    with open(args.problems_path) as f:
        problems = json.load(f)
    print(f"Evaluating {len(problems)} problems, {args.n_samples} samples each")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto",
    )

    if args.checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)
        print(f"Loaded LoRA checkpoint from {args.checkpoint}")

    results = evaluate(
        model, tokenizer, problems,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        sandbox_timeout=args.sandbox_timeout,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*40}")
    print(f"pass@1:         {results['pass_at_1']:.3f}")
    print(f"pass@k:         {results['pass_at_k']:.3f}")
    print(f"format rate:    {results['format_rate']:.3f}")
    print(f"unique solved:  {results['unique_solved']}/{results['n_problems']}")
    print(f"raw solve rate: {results['total_solved']}/{results['total_samples']} "
          f"= {results['raw_solve_rate']:.3f}")
    print(f"\nBy bug type:")
    for bt, info in sorted(results["by_bug_type"].items()):
        print(f"  {bt}: pass@1={info['pass_at_1']:.3f} "
              f"({info['solved']}/{info['total']})")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
