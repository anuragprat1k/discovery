"""Multi-turn evaluation for code repair.

Runs multi-turn episodes: model gets buggy code + test results, outputs
<repair>, gets updated test results, repeats for up to max_turns.

Reports pass@1, pass@k, format rate, unique problems solved, avg turns.

Usage:
    python -m code_repair.eval \
        --problems_path code_repair/data/problems_combined/eval.json \
        --n_samples 8 --max_turns 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code_repair.env.sandbox import run_tests
from code_repair.env.code_repair_env import (
    CodeRepairEnv,
    SYSTEM_PROMPT,
    extract_repair,
    format_initial_prompt,
    format_feedback,
)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def run_multi_turn_episode(
    model, tokenizer, problem: dict,
    max_turns: int = 4, max_new_tokens: int = 2048,
    temperature: float = 0.6, sandbox_timeout: int = 5,
) -> dict:
    """Run one multi-turn episode, return trace dict."""
    device = next(model.parameters()).device
    env = CodeRepairEnv([problem], max_turns=max_turns, sandbox_timeout=sandbox_timeout)
    obs = env.reset(problem_idx=0)
    messages = obs["messages"]
    done = False
    turns = []

    while not done:
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        input_ids = tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True,
            )
        completion = tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True,
        )

        obs, _, done, info = env.step(completion)
        repair = extract_repair(completion)

        turns.append({
            "turn": info.get("turn", len(turns) + 1),
            "repair_found": repair is not None,
            "curr_passing": info.get("curr_passing", 0),
            "hw_passing": info.get("hw_passing", 0),
            "num_tests": info.get("num_tests", 0),
            "all_passed": info.get("all_passed", False),
            "completion": completion[:2000],
            "repair": (repair or "")[:1500],
            "feedback": obs["messages"][0]["content"][:400] if obs["messages"] else "",
        })

        messages.append({"role": "assistant", "content": completion})
        if obs["messages"]:
            messages.extend(obs["messages"])

    return {
        "solved": info.get("all_passed", False),
        "total_turns": len(turns),
        "hw_passing": info.get("hw_passing", 0),
        "num_tests": info.get("num_tests", 0),
        "turns": turns,
    }


def evaluate(
    model, tokenizer, problems: list[dict],
    n_samples: int = 8, max_turns: int = 4,
    max_new_tokens: int = 2048, temperature: float = 0.6,
    sandbox_timeout: int = 5, verbose: bool = True,
    collect_traces: bool = False, n_trace_problems: int = 5,
) -> dict:
    """Run multi-turn evaluation.

    Args:
        collect_traces: If True, also returns detailed per-turn traces
            for the first n_trace_problems (1 sample each).
    """
    per_problem = {}
    total_solved = 0
    total_samples = 0
    total_turns_sum = 0
    unique_solved = set()
    total_format_ok = 0
    traces = []

    for prob_idx, p in enumerate(problems):
        task_id = p["task_id"]
        n_solved = 0
        n_format_ok = 0
        turns_sum = 0

        for sample_idx in range(n_samples):
            ep = run_multi_turn_episode(
                model, tokenizer, p,
                max_turns=max_turns, max_new_tokens=max_new_tokens,
                temperature=temperature, sandbox_timeout=sandbox_timeout,
            )

            if ep["solved"]:
                n_solved += 1
                unique_solved.add(task_id)
            turns_sum += ep["total_turns"]

            # Count format compliance (at least one turn had repair tag)
            if any(t["repair_found"] for t in ep["turns"]):
                n_format_ok += 1

            # Collect trace for first sample of first N problems
            if collect_traces and sample_idx == 0 and prob_idx < n_trace_problems:
                traces.append({
                    "task_id": task_id,
                    "bug_types": p.get("bug_types", [p.get("bug_type", "")]),
                    "n_bugs": p.get("n_bugs", 1),
                    "solved": ep["solved"],
                    "total_turns": ep["total_turns"],
                    "hw_passing": ep["hw_passing"],
                    "num_tests": ep["num_tests"],
                    "turns": ep["turns"],
                })

        p1 = pass_at_k(n_samples, n_solved, 1)
        pk = pass_at_k(n_samples, n_solved, min(n_samples, 8))

        per_problem[task_id] = {
            "n_solved": n_solved,
            "n_samples": n_samples,
            "pass_at_1": p1,
            "pass_at_k": pk,
            "format_rate": n_format_ok / n_samples,
            "avg_turns": turns_sum / n_samples,
            "n_bugs": p.get("n_bugs", 1),
            "bug_types": p.get("bug_types", []),
        }

        total_solved += n_solved
        total_samples += n_samples
        total_turns_sum += turns_sum
        total_format_ok += n_format_ok

        if verbose:
            print(f"  [{prob_idx+1}/{len(problems)}] {task_id} "
                  f"({p.get('n_bugs',1)} bugs): "
                  f"{n_solved}/{n_samples} solved, "
                  f"avg_turns={turns_sum/n_samples:.1f}, "
                  f"pass@1={p1:.3f}")

    overall_pass1 = float(np.mean([r["pass_at_1"] for r in per_problem.values()]))
    overall_passk = float(np.mean([r["pass_at_k"] for r in per_problem.values()]))
    overall_format = total_format_ok / total_samples if total_samples else 0
    avg_turns = total_turns_sum / total_samples if total_samples else 0

    # By n_bugs breakdown
    by_n_bugs = {}
    for info in per_problem.values():
        nb = info["n_bugs"]
        if nb not in by_n_bugs:
            by_n_bugs[nb] = {"pass1_sum": 0, "count": 0, "solved": 0, "total": 0}
        by_n_bugs[nb]["pass1_sum"] += info["pass_at_1"]
        by_n_bugs[nb]["count"] += 1
        by_n_bugs[nb]["solved"] += info["n_solved"]
        by_n_bugs[nb]["total"] += info["n_samples"]
    for nb in by_n_bugs:
        by_n_bugs[nb]["pass_at_1"] = by_n_bugs[nb]["pass1_sum"] / by_n_bugs[nb]["count"]

    result = {
        "pass_at_1": overall_pass1,
        "pass_at_k": overall_passk,
        "format_rate": overall_format,
        "unique_solved": len(unique_solved),
        "unique_solved_frac": len(unique_solved) / len(problems) if problems else 0,
        "avg_turns": avg_turns,
        "raw_solve_rate": total_solved / total_samples if total_samples else 0,
        "n_problems": len(problems),
        "by_n_bugs": by_n_bugs,
        "per_problem": per_problem,
    }
    if collect_traces:
        result["traces"] = traces
    return result


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-turn code repair evaluation.")
    parser.add_argument("--problems_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--max_turns", type=int, default=4)
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

    results = evaluate(
        model, tokenizer, problems,
        n_samples=args.n_samples, max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        sandbox_timeout=args.sandbox_timeout, collect_traces=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*40}")
    print(f"pass@1:         {results['pass_at_1']:.3f}")
    print(f"pass@k:         {results['pass_at_k']:.3f}")
    print(f"format rate:    {results['format_rate']:.3f}")
    print(f"avg turns:      {results['avg_turns']:.1f}")
    print(f"unique solved:  {results['unique_solved']}/{results['n_problems']}")
    print(f"\nBy # bugs:")
    for nb, info in sorted(results["by_n_bugs"].items()):
        print(f"  {nb} bug(s): pass@1={info['pass_at_1']:.3f} "
              f"({info['solved']}/{info['total']})")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
