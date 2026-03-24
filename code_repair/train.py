"""
GRPO training script for code repair using the Tinker API.

Follows the same pattern as wordle/recipes/train.py: multi-turn episodes
with per-turn reward computation, parallel rollouts via Tinker futures,
and GRPO advantage normalization.

Usage:
    python -m code_repair.train --reward {sparse,dense_passes,dense_full} [options]
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
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    import tinker
    from tinker import ServiceClient
except ImportError as exc:
    print(
        f"[code-repair-train] ERROR: Could not import 'tinker'.\n"
        f"  Install with: pip install tinker\n"
        f"  Original error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

from code_repair.env.code_repair_env import (
    SYSTEM_PROMPT,
    extract_repair,
    format_initial_prompt,
    format_feedback,
    format_test_results,
)
from code_repair.env.sandbox import run_tests
from code_repair.env.rewards import get_reward_fn


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-turn code repair GRPO training with Tinker API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reward", required=True,
        choices=["sparse", "dense_passes", "dense_full"],
        help="Reward function.",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=16,
                        help="Number of episodes per problem (G in GRPO).")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of problems per step.")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient.")
    parser.add_argument("--max_completion_tokens", type=int, default=1024,
                        help="Max tokens per turn (code can be long).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--problems_path", type=str,
                        default="code_repair/data/problems/train.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--loss_fn", type=str, default="cispo",
                        choices=["importance_sampling", "ppo", "cispo"])
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--refresh_steps", type=int, default=5)
    parser.add_argument("--sandbox_timeout", type=int, default=5,
                        help="Timeout for test execution in seconds.")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="discovery-code-repair")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_problems(path: str) -> list[dict]:
    with open(path) as f:
        problems = json.load(f)
    print(f"[code-repair] Loaded {len(problems)} problems from {path}")
    return problems


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def run_episode(
    problem: dict,
    sampling_client,
    tokenizer,
    args,
    reward_fn,
    max_turns: int = 4,
) -> dict:
    """Run one multi-turn code repair episode.

    Returns dict with per-turn tokens, logprobs, rewards, and episode metadata.
    """
    # Initialize
    initial_results = run_tests(
        problem["buggy_code"], problem["test_code"], problem["entry_point"],
        timeout=args.sandbox_timeout, detailed=True,
    )

    user_msg = format_initial_prompt(
        problem["buggy_code"], initial_results, max_turns,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    prompt_tokens_per_turn: list[list[int]] = []
    completion_tokens_per_turn: list[list[int]] = []
    completion_texts: list[str] = []
    logprobs_per_turn: list[list[float]] = []
    per_turn_rewards: list[float] = []

    hw_passing = 0
    all_passed = False
    test_results_history = [initial_results]

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_completion_tokens,
        temperature=args.temperature,
    )

    for turn in range(1, max_turns + 1):
        # Encode conversation
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_input = tinker.ModelInput.from_ints(prompt_tokens)

        # Sample
        result = sampling_client.sample(
            prompt=prompt_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        seq = result.sequences[0]
        completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)

        prompt_tokens_per_turn.append(prompt_tokens)
        completion_tokens_per_turn.append(list(seq.tokens))
        completion_texts.append(completion_text)
        logprobs_per_turn.append(
            [lp if lp is not None else -100.0 for lp in (seq.logprobs or [])]
        )

        # Parse repair
        repair = extract_repair(completion_text)

        if repair is None:
            # Format violation
            prev_results = test_results_history[-1]
            prev_passing = {r.name for r in prev_results if r.passed}
            info = {
                "turn": turn,
                "max_turns": max_turns,
                "format_violation": True,
                "num_tests": len(prev_results),
                "prev_passing": len(prev_passing),
                "curr_passing": len(prev_passing),
                "hw_passing": hw_passing,
                "old_hw": hw_passing,
                "all_passed": False,
                "no_crash_failing": 0,
                "type_match_failing": 0,
                "shape_match_failing": 0,
            }
            is_terminal = turn >= max_turns
            r, _ = reward_fn(info, is_terminal, completion_text)
            per_turn_rewards.append(r)

            remaining = max_turns - turn
            feedback_text = (
                f"No valid <repair> tag found. ({remaining} turn(s) remaining)"
            )
            messages.append({"role": "assistant", "content": completion_text})
            messages.append({"role": "user", "content": feedback_text})

            if is_terminal:
                break
            continue

        # Run tests on repair
        test_results = run_tests(
            repair, problem["test_code"], problem["entry_point"],
            timeout=args.sandbox_timeout, detailed=True,
        )
        test_results_history.append(test_results)

        prev_results = test_results_history[-2]
        prev_passing = {r.name for r in prev_results if r.passed}
        curr_passing = {r.name for r in test_results if r.passed}
        num_tests = max(len(test_results), 1)

        old_hw = hw_passing
        hw_passing = max(hw_passing, len(curr_passing))

        all_passed = (
            len(curr_passing) == len(test_results)
            and len(test_results) > 0
            and not (len(test_results) == 1 and test_results[0].name in
                     ("timeout", "syntax_error", "runtime_error", "import_error"))
        )

        no_crash_count = sum(1 for r in test_results if r.no_crash and not r.passed)
        type_match_count = sum(1 for r in test_results if not r.passed and r.no_crash and r.return_type)
        shape_match_count = sum(1 for r in test_results if not r.passed and r.return_shape is not None)

        is_terminal = all_passed or turn >= max_turns
        info = {
            "turn": turn,
            "max_turns": max_turns,
            "num_tests": num_tests,
            "prev_passing": len(prev_passing),
            "curr_passing": len(curr_passing),
            "hw_passing": hw_passing,
            "old_hw": old_hw,
            "all_passed": all_passed,
            "no_crash_failing": no_crash_count,
            "type_match_failing": type_match_count,
            "shape_match_failing": shape_match_count,
        }
        r, _ = reward_fn(info, is_terminal, completion_text)
        per_turn_rewards.append(r)

        # Build feedback
        if all_passed:
            feedback_text = f"All {num_tests} tests passing! Fixed in {turn} turn(s)."
        else:
            feedback_text = format_feedback(
                test_results, repair, turn=turn, max_turns=max_turns,
                prev_passing=prev_passing,
            )

        messages.append({"role": "assistant", "content": completion_text})
        messages.append({"role": "user", "content": feedback_text})

        if is_terminal:
            break

    return {
        "prompt_tokens_per_turn": prompt_tokens_per_turn,
        "completion_tokens_per_turn": completion_tokens_per_turn,
        "completion_texts": completion_texts,
        "logprobs_per_turn": logprobs_per_turn,
        "per_turn_rewards": per_turn_rewards,
        "all_passed": all_passed,
        "total_turns": len(per_turn_rewards),
        "task_id": problem["task_id"],
    }


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def compute_advantages(rewards: np.ndarray, group_size: int) -> tuple[np.ndarray, int]:
    """Compute GRPO advantages: group-normalize rewards.

    Returns (advantages, n_skipped_groups) where groups with zero variance
    are zeroed out.
    """
    n = len(rewards)
    assert n % group_size == 0
    n_groups = n // group_size
    advantages = np.zeros_like(rewards)
    skipped = 0

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group = rewards[start:end]
        std = group.std()
        if std < 1e-8:
            skipped += 1
            continue
        advantages[start:end] = (group - group.mean()) / (std + 1e-8)

    return advantages, skipped


# ---------------------------------------------------------------------------
# GRPO step (parallel rollouts)
# ---------------------------------------------------------------------------

def grpo_step(
    step_num: int,
    batch_problems: list[dict],
    sampling_client,
    training_client,
    tokenizer,
    args,
    reward_fn,
) -> dict:
    """Execute one GRPO step: rollout episodes, compute advantages, train."""
    t0 = time.time()

    # Run episodes (sequentially for now — sandbox calls are CPU-bound)
    episodes = []
    for problem in batch_problems:
        for _ in range(args.group_size):
            ep = run_episode(
                problem, sampling_client, tokenizer, args, reward_fn,
                max_turns=args.max_turns,
            )
            episodes.append(ep)

    t_gen = time.time() - t0

    # Compute total episode rewards
    total_rewards = np.array([sum(ep["per_turn_rewards"]) for ep in episodes])
    advantages, n_skipped = compute_advantages(total_rewards, args.group_size)

    # Build training data
    data = []
    for ep_idx, ep in enumerate(episodes):
        adv = advantages[ep_idx]
        for turn_idx in range(ep["total_turns"]):
            prompt_tokens = ep["prompt_tokens_per_turn"][turn_idx]
            comp_tokens = ep["completion_tokens_per_turn"][turn_idx]
            logprobs = ep["logprobs_per_turn"][turn_idx]

            full_tokens = prompt_tokens + comp_tokens
            n_prompt = len(prompt_tokens)
            n_comp = len(comp_tokens)

            # Token-level advantages: 0 for prompt, episode advantage for completion
            token_advantages = [0.0] * n_prompt + [adv] * n_comp

            data.append(tinker.Datum(
                model_input=tinker.ModelInput.from_ints(full_tokens),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(full_tokens, dtype="int64"),
                    "logprobs": tinker.TensorData(logprobs, dtype="float32"),
                    "advantages": tinker.TensorData(token_advantages, dtype="float32"),
                },
            ))

    # Forward-backward + optimizer step
    clip_cfg = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
    fwdbwd = training_client.forward_backward(
        data=data, loss_fn=args.loss_fn, loss_fn_config=clip_cfg,
    )
    optim = training_client.optim_step(
        tinker.AdamParams(learning_rate=args.lr, grad_clip_norm=args.grad_clip_norm)
    )

    t_total = time.time() - t0

    # Metrics
    solve_rate = np.mean([ep["all_passed"] for ep in episodes])
    avg_turns = np.mean([ep["total_turns"] for ep in episodes])
    avg_reward = np.mean(total_rewards)

    return {
        "solve_rate": solve_rate,
        "avg_turns": avg_turns,
        "avg_reward": avg_reward,
        "reward_std": np.std(total_rewards),
        "n_skipped_groups": n_skipped,
        "gen_time": t_gen,
        "total_time": t_total,
        "loss": getattr(fwdbwd, "loss", None),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load problems
    problems = load_problems(args.problems_path)

    # Setup Tinker
    client = ServiceClient()
    training_client = client.training_client(
        model=args.model,
        lora_rank=args.lora_rank,
    )
    sampling_client = client.sampling_client(model=args.model)
    tokenizer = client.tokenizer(model=args.model)

    reward_fn = get_reward_fn(args.reward)

    # Setup wandb
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"code-repair-{args.reward}-s{args.seed}",
        )

    # Output dir
    output_dir = args.output_dir or f"checkpoints/code_repair_{args.reward}_s{args.seed}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[code-repair] Starting training: reward={args.reward}, "
          f"steps={args.max_steps}, batch={args.batch_size}x{args.group_size}")

    for step in range(1, args.max_steps + 1):
        # Refresh sampling client periodically for on-policy rollouts
        if step % args.refresh_steps == 0:
            sampling_client = client.sampling_client(model=args.model)

        # Sample batch of problems
        batch = random.choices(problems, k=args.batch_size)

        metrics = grpo_step(
            step, batch, sampling_client, training_client, tokenizer,
            args, reward_fn,
        )

        log_msg = (
            f"[step {step}/{args.max_steps}] "
            f"solve={metrics['solve_rate']:.3f} "
            f"turns={metrics['avg_turns']:.1f} "
            f"reward={metrics['avg_reward']:.3f}±{metrics['reward_std']:.3f} "
            f"gen={metrics['gen_time']:.1f}s total={metrics['total_time']:.1f}s"
        )
        print(log_msg, flush=True)

        if use_wandb:
            wandb.log({
                "step": step,
                "solve_rate": metrics["solve_rate"],
                "avg_turns": metrics["avg_turns"],
                "avg_reward": metrics["avg_reward"],
                "reward_std": metrics["reward_std"],
                "gen_time": metrics["gen_time"],
                "total_time": metrics["total_time"],
            }, step=step)

        if args.log_file:
            with open(args.log_file, "a") as f:
                f.write(json.dumps({"step": step, **metrics}) + "\n")

        # Save checkpoint
        if step % args.save_steps == 0:
            save_path = os.path.join(output_dir, f"step_{step}")
            training_client.save(save_path)
            print(f"  Saved checkpoint to {save_path}")

    # Final save
    final_path = os.path.join(output_dir, f"step_{args.max_steps}")
    training_client.save(final_path)
    print(f"Training complete. Final checkpoint: {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
