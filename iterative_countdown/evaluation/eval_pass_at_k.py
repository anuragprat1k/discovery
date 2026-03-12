"""Pass@k evaluation for the Countdown environment.

Adapted from /workspace/discovery/eval/pass_at_k.py.
Instead of checking \\boxed{} answers, we simulate the Countdown game
and check if the model reaches the target.

Supports:
- Multi-turn episode simulation via CountdownMessageEnv
- Sampling via tinker SamplingClient (base model or checkpoint)
- pass@k computation using unbiased estimator from Codex paper
- Per-difficulty breakdown
- Incremental evaluation with resume support (JSONL sidecar)
- Frugal sampling: n_samples defaults to max(k_values)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from ..environment.countdown_env import CountdownMessageEnv
from ..environment.expression_parser import extract_expression, parse_expression


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm

    def progress(iterable, total=None, desc="", initial=0):
        return _tqdm(iterable, total=total, desc=desc, initial=initial)
except ImportError:

    def progress(iterable, total=None, desc="", initial=0):
        for i, item in enumerate(iterable, start=initial):
            if total:
                print(f"  {desc}: {i}/{total}", flush=True)
            yield item


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased estimator of pass@k from Codex paper.

    Args:
        n: total samples generated for the problem.
        c: number of correct (target-reaching) samples.
        k: target k value.

    Returns:
        Estimated probability that at least one of k samples is correct,
        or None if n < k.
    """
    if n < k:
        return None
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def evaluate_countdown_episode(
    model_outputs: list[str],
    target: int,
    numbers: list[int],
    max_turns: int = 5,
) -> dict:
    """Simulate a Countdown episode from model outputs.

    Replays the model's proposed expressions turn-by-turn, updating the
    available number pool and checking whether the target is reached.

    Args:
        model_outputs: list of model responses (one per turn).
        target: target number to reach.
        numbers: initial available numbers.
        max_turns: maximum turns allowed.

    Returns:
        dict with keys: target_reached, best_distance, turns_used,
        expressions, results.
    """
    available = list(numbers)
    best_distance = abs(target)
    expressions: list[str] = []
    results: list[int] = []
    target_reached = False

    for output in model_outputs[:max_turns]:
        expr_str = extract_expression(output)
        if expr_str is None:
            break

        try:
            result, nums_used = parse_expression(expr_str, available)
            expressions.append(expr_str)
            results.append(result)

            # Update available numbers
            for n in nums_used:
                available.remove(n)
            available.append(result)

            distance = abs(target - result)
            best_distance = min(best_distance, distance)

            if result == target:
                target_reached = True
                break
        except (ValueError, ZeroDivisionError):
            break

    return {
        "target_reached": target_reached,
        "best_distance": best_distance,
        "turns_used": len(expressions),
        "expressions": expressions,
        "results": results,
    }


def score_problem(
    episodes: list[dict],
    k_values: list[int] | None = None,
) -> dict:
    """Score multiple episodes (samples) for one problem.

    Args:
        episodes: list of evaluate_countdown_episode results.
        k_values: k values for pass@k computation.

    Returns:
        dict with pass@k values, best_distance stats, turn stats.
    """
    if k_values is None:
        k_values = [1, 4, 16]

    n = len(episodes)
    c = sum(1 for ep in episodes if ep["target_reached"])

    result: dict = {
        "n_samples": n,
        "n_correct": c,
    }

    for k in k_values:
        pk = pass_at_k(n, c, k)
        if pk is not None:
            result[f"pass@{k}"] = pk

    # Distance statistics
    distances = [ep["best_distance"] for ep in episodes]
    result["mean_best_distance"] = float(np.mean(distances))
    result["min_best_distance"] = int(min(distances))

    # Turn statistics
    turns = [ep["turns_used"] for ep in episodes]
    result["mean_turns"] = float(np.mean(turns))

    return result


def aggregate_results(
    problem_results: list[dict],
    problems: list[dict],
    k_values: list[int] | None = None,
) -> dict:
    """Aggregate per-problem results into overall metrics.

    Args:
        problem_results: list of score_problem outputs, one per problem.
        problems: list of problem dicts (must have same length).
        k_values: k values for pass@k.

    Returns:
        dict with overall pass@k, per-difficulty pass@k, distance stats.
    """
    if k_values is None:
        k_values = [1, 4, 16]

    overall: dict = {}

    # Overall pass@k
    for k in k_values:
        key = f"pass@{k}"
        values = [r[key] for r in problem_results if key in r]
        if values:
            overall[key] = float(np.mean(values))

    # Per-difficulty breakdown
    difficulty_groups: dict[str, list[dict]] = {}
    for prob, result in zip(problems, problem_results):
        diff = prob.get("difficulty", "unknown")
        if diff not in difficulty_groups:
            difficulty_groups[diff] = []
        difficulty_groups[diff].append(result)

    by_difficulty: dict[str, dict] = {}
    for diff, results in difficulty_groups.items():
        by_difficulty[diff] = {}
        for k in k_values:
            key = f"pass@{k}"
            values = [r[key] for r in results if key in r]
            if values:
                by_difficulty[diff][key] = float(np.mean(values))
        by_difficulty[diff]["n_problems"] = len(results)

    overall["by_difficulty"] = by_difficulty
    overall["n_problems"] = len(problem_results)
    overall["mean_best_distance"] = float(
        np.mean([r["mean_best_distance"] for r in problem_results])
    )

    return overall


# ---------------------------------------------------------------------------
# Incremental results (JSONL sidecar) — same pattern as eval/pass_at_k.py
# ---------------------------------------------------------------------------

def _sidecar_path(out_dir: Path, step: int) -> Path:
    """Path to the per-problem JSONL sidecar file."""
    return out_dir / f"step_{step:04d}.partial.jsonl"


def _load_partial(sidecar: Path) -> dict[int, dict]:
    """Load already-evaluated problem results from JSONL. Returns {idx: record}."""
    results = {}
    if not sidecar.exists():
        return results
    with open(sidecar) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            results[rec["idx"]] = rec
    return results


def _append_result(sidecar: Path, record: dict) -> None:
    """Append a single problem result to the JSONL sidecar."""
    with open(sidecar, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Multi-turn episode runner (async, drives CountdownMessageEnv)
# ---------------------------------------------------------------------------

async def _run_episode(
    env: CountdownMessageEnv,
    sample_fn,
    max_turns: int,
    max_tokens: int,
    temperature: float,
) -> dict:
    """Run a single multi-turn episode.

    Args:
        env: A fresh CountdownMessageEnv instance.
        sample_fn: async callable(messages) -> str that generates model response.
        max_turns: max turns per episode.
        max_tokens: max tokens per turn.
        temperature: sampling temperature.

    Returns:
        evaluate_countdown_episode-compatible result dict.
    """
    messages = await env.initial_observation()
    model_outputs: list[str] = []

    for _turn in range(max_turns):
        # Get model response
        response = await sample_fn(messages)
        model_outputs.append(response)

        # Step the environment
        step_result = await env.step({"role": "assistant", "content": response})

        if step_result.episode_done:
            break

        # Continue with updated conversation
        messages = step_result.next_messages

    # Build result using the same format as evaluate_countdown_episode
    return evaluate_countdown_episode(
        model_outputs=model_outputs,
        target=env.target,
        numbers=env.initial_numbers,
        max_turns=max_turns,
    )


# ---------------------------------------------------------------------------
# Tinker sampling setup
# ---------------------------------------------------------------------------

async def _create_tinker_sampling_client(model: str | None, tinker_path: str | None):
    """Create a Tinker sampling client.

    Two modes:
    1. Base model (step 0): create a LoRA training client, then get sampling client
    2. Checkpoint: restore from tinker_path, then get sampling client

    Returns:
        (sampling_client, tokenizer, tinker_module)
    """
    from dotenv import load_dotenv
    load_dotenv()

    import tinker
    from tinker import ServiceClient

    print("[tinker] Connecting to Tinker API ...", flush=True)
    service = ServiceClient()

    if tinker_path:
        print(f"[tinker] Restoring checkpoint: {tinker_path}", flush=True)
        training_client = await service.create_training_client_from_state_async(
            path=tinker_path
        )
        tokenizer = training_client.get_tokenizer()
        print("[tinker] Creating sampling client from checkpoint ...", flush=True)
        sampling_client = training_client.save_weights_and_get_sampling_client()
    else:
        assert model is not None, "Must provide --model for base model eval"
        print(f"[tinker] Creating LoRA training client for base model: {model}", flush=True)
        training_client = await service.create_lora_training_client_async(
            base_model=model,
            rank=32,
        )
        tokenizer = training_client.get_tokenizer()
        print("[tinker] Creating sampling client for base model ...", flush=True)
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name="base"
        )

    print("[tinker] Sampling client ready.", flush=True)
    return sampling_client, tokenizer, tinker


def _make_sample_fn(sampling_client, tokenizer, tinker_module, max_tokens: int, temperature: float):
    """Create an async sample function that calls tinker.

    Returns an async callable(messages: list[dict]) -> str
    """
    sampling_params = tinker_module.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    async def sample_fn(messages: list[dict]) -> str:
        """Apply chat template, sample from model, decode response."""
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_input = tinker_module.types.ModelInput.from_ints(prompt_token_ids)

        result = sampling_client.sample(
            prompt=prompt_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)

    return sample_fn


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def _evaluate_problem(
    problem: dict,
    episodes: list[dict],
    k_values: list[int],
    idx: int,
) -> dict:
    """Score a single problem from its episode results."""
    scores = score_problem(episodes, k_values)
    return {
        "idx": idx,
        "target": problem["target"],
        "numbers": problem["numbers"],
        "difficulty": problem.get("difficulty", "unknown"),
        **scores,
    }


async def _run_eval(args) -> dict:
    """Main async evaluation loop."""
    # Load problems
    with open(args.problems_path) as f:
        problems = json.load(f)
    n_problems = len(problems)

    # Determine n_samples: default to max(k_values) to be frugal
    n_samples = args.n_samples
    if n_samples is None:
        n_samples = max(args.k_values)
    # Ensure we have enough samples for the requested k values
    if n_samples < max(args.k_values):
        print(
            f"WARNING: n_samples={n_samples} < max(k_values)={max(args.k_values)}. "
            f"Setting n_samples={max(args.k_values)}",
            flush=True,
        )
        n_samples = max(args.k_values)

    print(f"Loaded {n_problems} problems", flush=True)
    print(f"n_samples={n_samples}, k_values={args.k_values}", flush=True)

    # Resume support
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = _sidecar_path(out_dir, args.step)
    partial = _load_partial(sidecar)
    n_done = len(partial)
    if n_done > 0:
        print(f"Resuming: {n_done}/{n_problems} problems already evaluated.", flush=True)

    # Create tinker sampling client
    sampling_client, tokenizer, tinker_module = await _create_tinker_sampling_client(
        model=args.model, tinker_path=args.tinker_path
    )
    sample_fn = _make_sample_fn(
        sampling_client, tokenizer, tinker_module,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Evaluate remaining problems
    remaining = [i for i in range(n_problems) if i not in partial]
    t0 = time.time()

    pbar = progress(remaining, total=n_problems, desc="Problems", initial=n_done)
    for idx in pbar:
        problem = problems[idx]
        target = problem["target"]
        numbers = problem["numbers"]

        # Run n_samples episodes for this problem
        episodes = []
        for _s in range(n_samples):
            env = CountdownMessageEnv(
                target=target,
                numbers=list(numbers),
                max_turns=args.max_turns,
            )
            episode = await _run_episode(
                env=env,
                sample_fn=sample_fn,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            episodes.append(episode)

        record = _evaluate_problem(problem, episodes, args.k_values, idx)
        _append_result(sidecar, record)
        partial[idx] = record

    elapsed = time.time() - t0
    print(f"\nEvaluation complete in {elapsed:.1f}s", flush=True)

    # Finalize: aggregate all results
    # Rebuild problem_results in order
    problem_results = []
    for i in range(n_problems):
        if i in partial:
            problem_results.append(partial[i])

    agg = aggregate_results(problem_results, problems, args.k_values)

    result = {
        "step": args.step,
        "n_problems": len(problem_results),
        "n_samples": n_samples,
        **{k: v for k, v in agg.items() if k not in ("by_difficulty", "n_problems", "mean_best_distance")},
        "mean_best_distance": agg["mean_best_distance"],
        "by_difficulty": agg.get("by_difficulty", {}),
        "per_problem": problem_results,
        "wall_clock_seconds": elapsed,
    }

    if args.model:
        result["model"] = args.model
    if args.tinker_path:
        result["tinker_path"] = args.tinker_path

    # Save final result
    output_path = out_dir / f"step_{args.step:04d}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {output_path}", flush=True)

    # Print summary
    print(f"\n{'='*50}", flush=True)
    print(f"Step {args.step} — {len(problem_results)} problems, {n_samples} samples each", flush=True)
    for k in args.k_values:
        key = f"pass@{k}"
        if key in result:
            print(f"  {key}: {result[key]:.4f}", flush=True)
    print(f"  mean_best_distance: {result['mean_best_distance']:.2f}", flush=True)
    if "by_difficulty" in result:
        for diff in sorted(result["by_difficulty"]):
            d = result["by_difficulty"][diff]
            parts = [f"n={d.get('n_problems', '?')}"]
            for k in args.k_values:
                key = f"pass@{k}"
                if key in d:
                    parts.append(f"{key}={d[key]:.4f}")
            print(f"  {diff}: {', '.join(parts)}", flush=True)
    print(f"{'='*50}", flush=True)

    return result


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Countdown model")
    parser.add_argument(
        "--problems_path",
        type=str,
        required=True,
        help="Path to problems JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples per problem (default: max(k_values), i.e. frugal)",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1],
        help="k values for pass@k (default: [1])",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Training step (for result file naming)",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=5,
        help="Max turns per episode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Base model name (for step 0 eval without checkpoint)",
    )
    parser.add_argument(
        "--tinker_path",
        type=str,
        default=None,
        help="Tinker checkpoint path (e.g. tinker://run-id/weights/step_100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Max tokens per turn",
    )
    args = parser.parse_args()

    # Validate: must specify exactly one of --model or --tinker_path
    if not args.model and not args.tinker_path:
        parser.error("Must specify either --model (base model) or --tinker_path (checkpoint)")
    if args.model and args.tinker_path:
        parser.error("Specify only one of --model or --tinker_path")

    asyncio.run(_run_eval(args))


if __name__ == "__main__":
    main()
