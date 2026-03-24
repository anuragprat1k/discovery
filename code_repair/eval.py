"""Evaluate code repair checkpoints on held-out problems.

Runs multi-turn code repair episodes and reports solve rates, turn distributions,
and per-problem results.

Usage:
    python -m code_repair.eval \
        --problems_path code_repair/data/problems/eval.json \
        --n_samples 16 --max_turns 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    import tinker
    from tinker import ServiceClient
except ImportError:
    tinker = None

from code_repair.env.code_repair_env import CodeRepairEnv, extract_repair
from code_repair.env.sandbox import run_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate code repair model.")
    parser.add_argument("--problems_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=16,
                        help="Number of attempts per problem (for pass@k).")
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Tinker checkpoint path (tinker://...).")
    parser.add_argument("--output_dir", type=str, default="results/code_repair")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--sandbox_timeout", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k from n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def main():
    args = parse_args()
    random.seed(args.seed)

    with open(args.problems_path) as f:
        problems = json.load(f)
    print(f"Evaluating {len(problems)} problems, {args.n_samples} samples each")

    if tinker is None:
        print("ERROR: tinker not installed, cannot run eval", file=sys.stderr)
        sys.exit(1)

    client = ServiceClient()
    sampling_client = client.sampling_client(
        model=args.checkpoint or args.model,
    )
    tokenizer = client.tokenizer(model=args.model)

    env = CodeRepairEnv(problems, max_turns=args.max_turns, sandbox_timeout=args.sandbox_timeout)

    results_per_problem = {}
    all_solved = 0
    all_total = 0

    for prob_idx, problem in enumerate(problems):
        task_id = problem["task_id"]
        n_solved = 0

        for sample in range(args.n_samples):
            obs = env.reset(problem_idx=prob_idx)
            messages = obs["messages"]
            done = False

            while not done:
                prompt_text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                )
                prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

                result = sampling_client.sample(
                    prompt=tinker.ModelInput.from_ints(prompt_tokens),
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    ),
                ).result()

                completion_text = tokenizer.decode(
                    result.sequences[0].tokens, skip_special_tokens=True,
                )

                obs, _, done, info = env.step(completion_text)
                messages.append({"role": "assistant", "content": completion_text})
                if obs["messages"]:
                    messages.extend(obs["messages"])

            if info.get("all_passed", False):
                n_solved += 1

        p1 = pass_at_k(args.n_samples, n_solved, 1)
        pk = pass_at_k(args.n_samples, n_solved, min(args.n_samples, 8))
        results_per_problem[task_id] = {
            "n_solved": n_solved,
            "n_samples": args.n_samples,
            "pass_at_1": p1,
            "pass_at_k": pk,
        }
        all_solved += n_solved
        all_total += args.n_samples

        print(f"  [{prob_idx+1}/{len(problems)}] {task_id}: "
              f"{n_solved}/{args.n_samples} solved, pass@1={p1:.3f}")

    # Aggregate
    overall_pass1 = np.mean([r["pass_at_1"] for r in results_per_problem.values()])
    overall_passk = np.mean([r["pass_at_k"] for r in results_per_problem.values()])

    summary = {
        "overall_pass_at_1": overall_pass1,
        "overall_pass_at_k": overall_passk,
        "total_solved": all_solved,
        "total_samples": all_total,
        "raw_solve_rate": all_solved / all_total if all_total > 0 else 0,
        "per_problem": results_per_problem,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Results ===")
    print(f"pass@1: {overall_pass1:.3f}")
    print(f"pass@k: {overall_passk:.3f}")
    print(f"Raw solve rate: {all_solved}/{all_total} = {all_solved/all_total:.3f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
