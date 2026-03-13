"""GRPO training for Countdown using Tinker API directly (no tinker-cookbook).

Multi-turn rollouts: for each problem, the model plays up to max_turns of
Countdown, with the environment providing feedback after each turn. The full
trajectory is then scored and used for a GRPO update.

Usage:
    python -m iterative_countdown.recipes.train_tinker \
        --reward binary --max_steps 10 --batch_size 4 --group_size 4 \
        --wandb_project discovery-countdown
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import tinker
from tinker import ServiceClient

from ..environment.countdown_env import CountdownMessageEnv
from ..environment.expression_parser import strip_think_tags


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Countdown GRPO training with Tinker API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reward", required=True, choices=["binary", "dense", "prime"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens per turn")
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--train_data", type=str, default="iterative_countdown/data/train_problems.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--loss_fn", type=str, default="importance_sampling")
    parser.add_argument("--eval_steps", type=int, default=50, help="Run eval every N steps (0 to disable)")
    parser.add_argument("--eval_problems", type=str, default="iterative_countdown/data/eval_50.json")
    parser.add_argument("--eval_n_problems", type=int, default=20, help="Number of eval problems to sample (for speed)")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="discovery-countdown")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None, help="W&B group for grouping related runs")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def get_reward_module(reward_type: str):
    """Import the appropriate reward module."""
    mod = importlib.import_module(f"iterative_countdown.rewards.{reward_type}_reward")
    return mod


# ---------------------------------------------------------------------------
# Periodic eval during training
# ---------------------------------------------------------------------------

def run_periodic_eval(
    step: int,
    sampling_client,
    tokenizer,
    eval_problems: list[dict],
    n_problems: int,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    output_dir: str,
) -> dict:
    """Run a quick eval on a subset of problems and save results + traces.

    Uses the eval module's _run_episode and scoring, but drives it with the
    training loop's sampling client (no need to create a new one).
    """
    import asyncio
    from ..evaluation.eval_pass_at_k import (
        _run_episode,
        _make_sample_fn,
        score_problem,
        _sidecar_path,
        _append_result,
    )
    from pathlib import Path

    rng = np.random.default_rng(seed=step)
    subset_idx = rng.choice(len(eval_problems), size=min(n_problems, len(eval_problems)), replace=False)
    subset = [eval_problems[i] for i in subset_idx]

    # Build sample_fn from the existing sampling client
    sampling_params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature)

    async def sample_fn(messages):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_input = tinker.ModelInput.from_ints(prompt_tokens)
        result = sampling_client.sample(
            prompt=prompt_input, num_samples=1, sampling_params=sampling_params,
        ).result()
        decoded = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=False)
        return strip_think_tags(decoded)

    loop = asyncio.new_event_loop()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = _sidecar_path(out_dir, step)

    t0 = time.time()
    all_results = []
    for i, problem in enumerate(subset):
        env = CountdownMessageEnv(
            target=problem["target"],
            numbers=list(problem["numbers"]),
            max_turns=max_turns,
        )
        episode = loop.run_until_complete(_run_episode(
            env=env, sample_fn=sample_fn,
            max_turns=max_turns, max_tokens=max_tokens, temperature=temperature,
        ))

        scores = score_problem([episode], k_values=[1])
        record = {
            "idx": int(subset_idx[i]),
            "target": problem["target"],
            "numbers": problem["numbers"],
            "difficulty": problem.get("difficulty", "unknown"),
            **scores,
        }
        # Save traces selectively (every 5th problem)
        if i % 5 == 0:
            record["traces"] = [{
                "model_outputs": [o[:2000] for o in episode.get("model_outputs", [])],
                "target_reached": episode["target_reached"],
                "best_distance": episode["best_distance"],
                "turns_used": episode["turns_used"],
            }]
        _append_result(sidecar, record)
        all_results.append(record)

    loop.close()
    elapsed = time.time() - t0

    # Aggregate
    n_correct = sum(1 for r in all_results if r.get("n_correct", 0) > 0)
    mean_dist = float(np.mean([r["mean_best_distance"] for r in all_results]))
    pass_1 = n_correct / len(all_results) if all_results else 0.0

    summary = {
        "step": step,
        "eval_n_problems": len(all_results),
        "eval_pass@1": pass_1,
        "eval_mean_best_distance": mean_dist,
        "eval_time": round(elapsed, 1),
    }
    print(
        f"  [eval step {step}] pass@1={pass_1:.2f}  dist={mean_dist:.1f}  "
        f"({len(all_results)} problems, {elapsed:.1f}s)",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# Multi-turn rollout
# ---------------------------------------------------------------------------

def run_episode(
    problem: dict,
    sampling_client,
    tokenizer,
    sampling_params,
    max_turns: int,
    _loop=None,
) -> dict:
    """Run a single multi-turn Countdown episode.

    Returns dict with episode info + per-turn token data for training.
    """
    import asyncio
    if _loop is None:
        _loop = asyncio.new_event_loop()

    target = problem["target"]
    numbers = problem["numbers"]
    env = CountdownMessageEnv(target=target, numbers=list(numbers), max_turns=max_turns)

    messages = _loop.run_until_complete(env.initial_observation())

    model_outputs: list[str] = []
    all_prompt_tokens: list[list[int]] = []
    all_completion_tokens: list[list[int]] = []
    all_logprobs: list[list[float]] = []

    for turn in range(max_turns):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_input = tinker.ModelInput.from_ints(prompt_tokens)

        result = sampling_client.sample(
            prompt=prompt_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        seq = result.sequences[0]
        completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=False)
        answer_text = strip_think_tags(completion_text)
        model_outputs.append(completion_text)
        all_prompt_tokens.append(prompt_tokens)
        all_completion_tokens.append(list(seq.tokens))
        all_logprobs.append(
            [lp if lp is not None else 0.0 for lp in seq.logprobs]
            if seq.logprobs else [0.0] * len(seq.tokens)
        )

        step_result = _loop.run_until_complete(
            env.step({"role": "assistant", "content": answer_text})
        )

        if step_result.episode_done:
            break

        messages = step_result.next_messages

    return {
        "target": target,
        "numbers": numbers,
        "initial_distance": abs(target),
        "best_distance": env.best_distance,
        "target_reached": env.best_distance == 0,
        "turns_used": len(model_outputs),
        "model_outputs": model_outputs,
        "all_prompt_tokens": all_prompt_tokens,
        "all_completion_tokens": all_completion_tokens,
        "all_logprobs": all_logprobs,
    }


def compute_episode_reward(episode: dict, reward_module, max_turns: int = 5) -> float:
    """Compute total episode reward = sum(turn_rewards) + episode_bonus."""
    import inspect

    # Sum of per-turn rewards (the dense signal)
    turn_total = sum(episode.get("turn_rewards", []))

    # Episode-level bonus
    sig = inspect.signature(reward_module.compute_episode_reward)
    kwargs: dict = {
        "target_reached": episode["target_reached"],
        "best_distance": episode["best_distance"],
        "initial_distance": episode["initial_distance"],
        "total_turns": episode["turns_used"],
        "max_turns": max_turns,
    }
    if "trajectory_texts" in sig.parameters:
        kwargs["trajectory_texts"] = episode["model_outputs"]
    episode_bonus, _ = reward_module.compute_episode_reward(**kwargs)

    return turn_total + episode_bonus


# ---------------------------------------------------------------------------
# GRPO step
# ---------------------------------------------------------------------------

def grpo_step(
    step: int,
    batch_problems: list[dict],
    sampling_client,
    training_client,
    tokenizer,
    reward_module,
    args,
) -> dict:
    """Execute one GRPO training step with multi-turn rollouts."""
    t0 = time.time()

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # --- 1. Generate rollouts (batched by turn depth) ---
    import asyncio
    loop = asyncio.new_event_loop()

    t_gen_start = time.time()

    # Initialize all episodes
    n_total = len(batch_problems) * args.group_size
    episodes_state: list[dict] = []
    for problem in batch_problems:
        for _g in range(args.group_size):
            target = problem["target"]
            numbers = problem["numbers"]
            env = CountdownMessageEnv(target=target, numbers=list(numbers), max_turns=args.max_turns)
            messages = loop.run_until_complete(env.initial_observation())
            episodes_state.append({
                "env": env,
                "messages": messages,
                "target": target,
                "numbers": numbers,
                "initial_distance": abs(target),
                "prev_distance": float(abs(target)),
                "turn_rewards": [],  # accumulate per-turn rewards
                "model_outputs": [],
                "all_prompt_tokens": [],
                "all_completion_tokens": [],
                "all_logprobs": [],
                "done": False,
            })

    # Run turns in batches — all active episodes at the same turn depth
    for turn in range(args.max_turns):
        active = [i for i, ep in enumerate(episodes_state) if not ep["done"]]
        if not active:
            break

        # Fire all sample requests as futures (parallel)
        futures = []
        for i in active:
            ep = episodes_state[i]
            prompt_text = tokenizer.apply_chat_template(
                ep["messages"], tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_input = tinker.ModelInput.from_ints(prompt_tokens)

            future = sampling_client.sample(
                prompt=prompt_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            futures.append((i, future, prompt_tokens))

        # Collect all results
        for i, future, prompt_tokens in futures:
            ep = episodes_state[i]
            result = future.result()
            seq = result.sequences[0]
            completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=False)
            # Strip thinking content — only pass the answer to env/rewards
            answer_text = strip_think_tags(completion_text)

            ep["model_outputs"].append(completion_text)
            ep["all_prompt_tokens"].append(prompt_tokens)
            ep["all_completion_tokens"].append(list(seq.tokens))
            ep["all_logprobs"].append(
                [lp if lp is not None else 0.0 for lp in seq.logprobs]
                if seq.logprobs else [0.0] * len(seq.tokens)
            )

            step_result = loop.run_until_complete(
                ep["env"].step({"role": "assistant", "content": answer_text})
            )

            # Compute turn reward from our reward module
            env = ep["env"]
            metrics = step_result.metrics
            expr_valid = metrics.get("invalid_expression", 0.0) == 0.0
            turn_num = int(metrics.get("turn", len(ep["turn_rewards"]) + 1))
            current_distance = metrics.get("distance_to_target", ep["prev_distance"])

            # Build kwargs for compute_turn_reward (signature varies by module)
            turn_kwargs: dict = {
                "expression_valid": expr_valid,
                "result": int(current_distance) if expr_valid else None,  # approximate
                "target": ep["target"],
                "available_numbers": list(env.available_numbers),
                "turn": turn_num,
                "max_turns": args.max_turns,
                "best_distance": env.best_distance,
                "initial_distance": ep["initial_distance"],
            }
            # Dense and prime modules need extra kwargs
            import inspect
            sig = inspect.signature(reward_module.compute_turn_reward)
            if "numbers_used" in sig.parameters:
                turn_kwargs["numbers_used"] = []  # env already consumed them
            if "prev_distance" in sig.parameters:
                turn_kwargs["prev_distance"] = ep["prev_distance"]
            if "model_text" in sig.parameters:
                turn_kwargs["model_text"] = answer_text
            if "weights" in sig.parameters:
                turn_kwargs["weights"] = None
            # Fix result: use actual result from env history if available
            if env.history:
                _, last_result, _ = env.history[-1]
                turn_kwargs["result"] = last_result

            turn_reward, _ = reward_module.compute_turn_reward(**turn_kwargs)
            ep["turn_rewards"].append(turn_reward)
            ep["prev_distance"] = float(metrics.get("best_distance", ep["prev_distance"]))

            if step_result.episode_done:
                ep["done"] = True
            else:
                ep["messages"] = step_result.next_messages

    loop.close()

    # Build final episode dicts
    all_episodes: list[dict] = []
    rewards: list[float] = []
    for ep in episodes_state:
        episode = {
            "target": ep["target"],
            "numbers": ep["numbers"],
            "initial_distance": abs(ep["target"]),
            "best_distance": ep["env"].best_distance,
            "target_reached": ep["env"].best_distance == 0,
            "turns_used": len(ep["model_outputs"]),
            "turn_rewards": ep["turn_rewards"],
            "model_outputs": ep["model_outputs"],
            "all_prompt_tokens": ep["all_prompt_tokens"],
            "all_completion_tokens": ep["all_completion_tokens"],
            "all_logprobs": ep["all_logprobs"],
        }
        all_episodes.append(episode)
        reward = compute_episode_reward(episode, reward_module, args.max_turns)
        rewards.append(reward)

    t_gen = time.time() - t_gen_start
    rewards_arr = np.array(rewards, dtype=np.float32)

    # --- 2. Compute advantages (mean-center per group) ---
    n_problems = len(batch_problems)
    advantages = np.zeros_like(rewards_arr)
    groups_skipped = 0

    for i in range(n_problems):
        start = i * args.group_size
        end = start + args.group_size
        group_rewards = rewards_arr[start:end]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std()

        if group_std < 1e-8:
            advantages[start:end] = 0.0
            groups_skipped += 1
        else:
            advantages[start:end] = (group_rewards - group_mean) / (group_std + 1e-8)

    # --- 3. Build Datum objects ---
    # Per-turn training: create one Datum per turn per episode so all model
    # decisions receive gradient signal, not just the last turn.
    data = []
    for idx, episode in enumerate(all_episodes):
        if abs(advantages[idx]) < 1e-8:
            continue

        adv_value = float(advantages[idx])
        n_turns = len(episode["all_prompt_tokens"])
        for turn in range(n_turns):
            prompt_tokens = episode["all_prompt_tokens"][turn]
            completion_tokens = episode["all_completion_tokens"][turn]
            logprobs = episode["all_logprobs"][turn]

            n_prompt = len(prompt_tokens)
            n_completion = len(completion_tokens)
            full_tokens = prompt_tokens + completion_tokens

            model_input = tinker.ModelInput.from_ints(full_tokens)

            target_tokens = tinker.TensorData(
                data=full_tokens,
                dtype="int64",
            )

            token_advantages = tinker.TensorData(
                data=[0.0] * n_prompt + [adv_value] * n_completion,
                dtype="float32",
            )

            lp_list = [0.0] * n_prompt + logprobs[:n_completion]
            # Pad if logprobs shorter than completion
            if len(lp_list) < n_prompt + n_completion:
                lp_list.extend([0.0] * (n_prompt + n_completion - len(lp_list)))

            logprobs_tensor = tinker.TensorData(
                data=lp_list,
                dtype="float32",
            )

            datum = tinker.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "logprobs": logprobs_tensor,
                    "advantages": token_advantages,
                },
            )
            data.append(datum)

    if not data:
        return {
            "step": step,
            "mean_reward": float(rewards_arr.mean()),
            "loss": 0.0,
            "groups_skipped": groups_skipped,
            "n_train_samples": 0,
            "target_reached_rate": float(np.mean([e["target_reached"] for e in all_episodes])),
            "mean_turns": float(np.mean([e["turns_used"] for e in all_episodes])),
            "mean_best_distance": float(np.mean([e["best_distance"] for e in all_episodes])),
            "time_gen": round(t_gen, 2),
            "time_total": round(time.time() - t0, 2),
        }

    # --- 4. Forward-backward + optimizer step ---
    t_train_start = time.time()

    fwdbwd_future = training_client.forward_backward(
        data=data,
        loss_fn=args.loss_fn,
    )
    optim_future = training_client.optim_step(
        tinker.AdamParams(learning_rate=args.lr)
    )

    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    t_train = time.time() - t_train_start

    loss = fwdbwd_result.metrics.get("loss:sum", fwdbwd_result.metrics.get("loss", 0.0))

    metrics = {
        "step": step,
        "mean_reward": float(rewards_arr.mean()),
        "max_reward": float(rewards_arr.max()),
        "min_reward": float(rewards_arr.min()),
        "reward_std": float(rewards_arr.std()),
        "loss": float(loss),
        "groups_skipped": groups_skipped,
        "n_train_samples": len(data),
        "target_reached_rate": float(np.mean([e["target_reached"] for e in all_episodes])),
        "mean_turns": float(np.mean([e["turns_used"] for e in all_episodes])),
        "mean_best_distance": float(np.mean([e["best_distance"] for e in all_episodes])),
        "time_gen": round(t_gen, 2),
        "time_train": round(t_train, 2),
        "time_total": round(time.time() - t0, 2),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or f"checkpoints/countdown_{args.reward}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "training_log.jsonl")
    wandb_name = args.wandb_name or f"countdown-{args.reward}"

    print(f"[countdown] === GRPO Training (Tinker) ===")
    print(f"[countdown] Reward     : {args.reward}")
    print(f"[countdown] Model      : {args.model}")
    print(f"[countdown] LoRA rank  : {args.lora_rank}")
    print(f"[countdown] LR         : {args.lr}")
    print(f"[countdown] Max turns  : {args.max_turns}")
    print(f"[countdown] Batch size : {args.batch_size} problems x {args.group_size} rollouts")
    print(f"[countdown] Max steps  : {args.max_steps}")
    print(f"[countdown] Output dir : {output_dir}")

    # Load training data
    with open(args.train_data) as f:
        all_problems = json.load(f)
    print(f"[countdown] Loaded {len(all_problems)} training problems")

    # Load eval data (for periodic eval)
    eval_problems = []
    if args.eval_steps > 0 and os.path.exists(args.eval_problems):
        with open(args.eval_problems) as f:
            eval_problems = json.load(f)
        print(f"[countdown] Loaded {len(eval_problems)} eval problems (sampling {args.eval_n_problems} per eval, every {args.eval_steps} steps)")
    elif args.eval_steps > 0:
        print(f"[countdown] WARNING: eval_problems not found at {args.eval_problems}, skipping periodic eval")

    # Initialize Tinker
    print(f"[countdown] Connecting to Tinker API ...")
    service = ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()
    sampling_client = training_client.save_weights_and_get_sampling_client(name="init")
    print(f"[countdown] Tinker ready.")

    # Reward module
    reward_module = get_reward_module(args.reward)

    # W&B
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            group=args.wandb_group,
            config=vars(args),
        )
        # Allow eval metrics to arrive out-of-order (async eval)
        wandb.define_metric("eval_pass@1", step_metric="step", step_sync=False)
        wandb.define_metric("eval_mean_best_distance", step_metric="step", step_sync=False)
        wandb.define_metric("eval_n_problems", step_metric="step", step_sync=False)
        wandb.define_metric("eval_time", step_metric="step", step_sync=False)
        print(f"[countdown] W&B run: {wandb.run.url}")

    # Background eval executor (single thread — evals queue up)
    eval_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    eval_future = None
    eval_sampling_client = None

    # Training loop
    rng = np.random.default_rng(seed=42)
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        # Sample batch of problems
        indices = rng.choice(len(all_problems), size=args.batch_size, replace=True)
        batch_problems = [all_problems[i] for i in indices]

        metrics = grpo_step(
            step=step,
            batch_problems=batch_problems,
            sampling_client=sampling_client,
            training_client=training_client,
            tokenizer=tokenizer,
            reward_module=reward_module,
            args=args,
        )

        # Log
        print(
            f"[step {step:4d}/{args.max_steps}] "
            f"reward={metrics['mean_reward']:.3f}  "
            f"loss={metrics['loss']:.4f}  "
            f"solved={metrics['target_reached_rate']:.2f}  "
            f"dist={metrics['mean_best_distance']:.1f}  "
            f"time={metrics['time_total']:.1f}s"
        )

        with open(log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if use_wandb:
            wandb.log(metrics, step=step)

        # Save checkpoint (less frequent, costs Tinker storage)
        if step % args.save_steps == 0:
            ckpt_name = f"step_{step:04d}"
            print(f"[countdown] Saving checkpoint: {ckpt_name}")
            save_result = training_client.save_state(ckpt_name).result()
            ckpt_path = getattr(save_result, "path", None)
            print(f"[countdown] Checkpoint saved: {ckpt_path}")

            manifest_path = os.path.join(output_dir, "checkpoints.jsonl")
            with open(manifest_path, "a") as f:
                f.write(json.dumps({"step": step, "name": ckpt_name, "path": ckpt_path}) + "\n")

            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=ckpt_name
            )

        # Periodic eval (more frequent, lightweight — no persistent checkpoint)
        if eval_problems and args.eval_steps > 0 and step % args.eval_steps == 0:
            # Wait for previous eval to finish before starting next
            if eval_future is not None:
                try:
                    prev_eval_metrics = eval_future.result()
                    if use_wandb:
                        wandb.log(prev_eval_metrics)
                    print(f"  [eval step {prev_eval_metrics['step']}] logged to wandb", flush=True)
                except Exception as e:
                    print(f"  [eval] previous eval failed: {e}", flush=True)

            # Snapshot current weights for eval
            eval_sampling_client = training_client.save_weights_and_get_sampling_client(
                name="eval"
            )

            eval_dir = os.path.join(output_dir, "eval")
            eval_future = eval_executor.submit(
                run_periodic_eval,
                step=step,
                sampling_client=eval_sampling_client,
                tokenizer=tokenizer,
                eval_problems=eval_problems,
                n_problems=args.eval_n_problems,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                output_dir=eval_dir,
            )

    # Collect any remaining background eval
    if eval_future is not None:
        try:
            last_eval = eval_future.result(timeout=300)
            if use_wandb:
                wandb.log(last_eval)
            print(f"  [eval step {last_eval['step']}] final eval logged", flush=True)
        except Exception as e:
            print(f"  [eval] final eval failed: {e}", flush=True)
    eval_executor.shutdown(wait=False)

    # Final
    elapsed = time.time() - t_start
    print(f"\n[countdown] === Training Complete ===")
    print(f"[countdown] Total time: {elapsed / 60:.1f} min")

    save_result = training_client.save_state("final").result()
    ckpt_path = getattr(save_result, "path", None)
    print(f"[countdown] Final checkpoint: {ckpt_path}")

    manifest_path = os.path.join(output_dir, "checkpoints.jsonl")
    with open(manifest_path, "a") as f:
        f.write(json.dumps({"step": args.max_steps, "name": "final", "path": ckpt_path}) + "\n")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
