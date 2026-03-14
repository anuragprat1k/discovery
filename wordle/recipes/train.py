"""
GRPO training script for Wordle using the Tinker API.

Follows the same pattern as train_tinker.py (single-turn MATH) but handles
multi-turn Wordle episodes: each rollout is a sequence of sample→feedback
cycles (up to max_turns), with rewards computed per-turn (dense) or at
episode end only (sparse).

Usage:
    python -m wordle.recipes.train --reward {dense,sparse} [options]
"""
from __future__ import annotations

import argparse
import asyncio
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
        f"[wordle-train] ERROR: Could not import 'tinker'.\n"
        f"  Install with: pip install tinker\n"
        f"  Original error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

from wordle.environment.feedback import TileColor, compute_feedback, feedback_to_emoji
from wordle.environment.constraints import RevealedConstraints, compute_constraint_violation_rate
from wordle.environment.wordle_env import load_word_list, _extract_guess, SYSTEM_PROMPT
from wordle.rewards import dense_reward, sparse_reward


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-turn Wordle GRPO training with Tinker API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reward", required=True, choices=["dense", "sparse"],
        help="Reward function: 'dense' for per-turn shaping, 'sparse' for terminal-only.",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_turns", type=int, default=6,
                        help="Max guesses per Wordle episode.")
    parser.add_argument("--group_size", type=int, default=8,
                        help="Number of episodes per target word (G in GRPO).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of target words per step.")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient.")
    parser.add_argument("--max_completion_tokens", type=int, default=256,
                        help="Max tokens per turn (brief reasoning + guess).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--answers_path", type=str, default=None)
    parser.add_argument("--guesses_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--loss_fn", type=str, default="importance_sampling",
                        choices=["importance_sampling", "ppo", "cispo"])
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="discovery-wordle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace_words", type=str, nargs="*", default=None,
                        help="Specific probe words for trajectory saving. If not set, picked randomly.")
    parser.add_argument("--n_trace_words", type=int, default=5,
                        help="Number of probe words for trajectory saving (used when --trace_words not set).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_words(args) -> tuple[list[str], set[str]]:
    """Load answer and valid guess word lists."""
    data_dir = Path(__file__).parent.parent / "data"
    answers_path = Path(args.answers_path) if args.answers_path else data_dir / "wordle_answers.txt"
    guesses_path = Path(args.guesses_path) if args.guesses_path else data_dir / "wordle_guesses.txt"

    answers = load_word_list(answers_path)
    extra_guesses = load_word_list(guesses_path)
    valid_guesses = set(answers) | set(extra_guesses)

    print(f"[wordle] Loaded {len(answers)} answers, {len(valid_guesses)} total valid guesses")
    return answers, valid_guesses


# ---------------------------------------------------------------------------
# Multi-turn episode rollout
# ---------------------------------------------------------------------------

def run_episode(
    target: str,
    valid_guesses: set[str],
    sampling_client,
    tokenizer,
    args,
    max_turns: int = 6,
) -> dict:
    """Run one multi-turn Wordle episode.

    Returns dict with:
        - prompt_tokens: list of all tokens fed to model across turns
        - completion_tokens_per_turn: list of token lists per turn
        - logprobs_per_turn: list of logprob lists per turn
        - history: list of (guess, feedback) tuples
        - target_reached: bool
        - total_turns: int
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Guess a 5-letter word. Reply with <guess>WORD</guess>."},
    ]

    history: list[tuple[str, list[TileColor]]] = []
    completion_tokens_per_turn: list[list[int]] = []
    logprobs_per_turn: list[list[float]] = []
    prompt_tokens_per_turn: list[list[int]] = []
    target_reached = False

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_completion_tokens,
        temperature=args.temperature,
    )

    for turn in range(1, max_turns + 1):
        # Encode current conversation as prompt
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_input = tinker.ModelInput.from_ints(prompt_tokens)

        # Sample one completion
        result = sampling_client.sample(
            prompt=prompt_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        seq = result.sequences[0]
        completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)

        prompt_tokens_per_turn.append(prompt_tokens)
        completion_tokens_per_turn.append(list(seq.tokens))
        logprobs_per_turn.append(
            [lp if lp is not None else 0.0 for lp in (seq.logprobs or [])]
        )

        # Extract guess
        guess = _extract_guess(completion_text)

        if guess is None or guess not in valid_guesses:
            # Invalid: all grey, turn consumed
            guess = guess or "?????"
            feedback = [TileColor.GREY] * 5
            history.append((guess, feedback))
        else:
            feedback = compute_feedback(guess, target)
            history.append((guess, feedback))
            if guess == target:
                target_reached = True

        # Build feedback message for next turn
        emoji = feedback_to_emoji(feedback)
        remaining = max_turns - turn
        if target_reached:
            feedback_text = (
                f"Turn {turn}: {guess.upper()} \u2192 {emoji}  "
                f"Correct! You got it in {turn} guess(es)!"
            )
        else:
            feedback_text = (
                f"Turn {turn}: {guess.upper()} \u2192 {emoji}  "
                f"({remaining} turn(s) remaining)"
            )

        # Update conversation
        messages.append({"role": "assistant", "content": completion_text})
        messages.append({"role": "user", "content": feedback_text})

        if target_reached or turn >= max_turns:
            break

    return {
        "prompt_tokens_per_turn": prompt_tokens_per_turn,
        "completion_tokens_per_turn": completion_tokens_per_turn,
        "logprobs_per_turn": logprobs_per_turn,
        "history": history,
        "target_reached": target_reached,
        "total_turns": len(history),
    }


# ---------------------------------------------------------------------------
# Reward computation for an episode
# ---------------------------------------------------------------------------

def compute_episode_rewards(
    episode: dict,
    reward_type: str,
    max_turns: int,
) -> list[float]:
    """Compute per-turn rewards for an episode.

    Returns a list of rewards, one per turn.
    """
    history = episode["history"]
    target_reached = episode["target_reached"]
    total_turns = episode["total_turns"]

    reward_mod = dense_reward if reward_type == "dense" else sparse_reward
    per_turn_rewards = []

    for turn_idx, (guess, feedback) in enumerate(history):
        turn = turn_idx + 1
        prev_guesses = [g for g, _ in history[:turn_idx]]
        prev_feedbacks = [f for _, f in history[:turn_idx]]

        is_final_turn = (turn == total_turns)
        turn_target_reached = target_reached and is_final_turn

        tr, _ = reward_mod.compute_turn_reward(
            guess=guess,
            feedback=feedback,
            prev_feedbacks=prev_feedbacks,
            prev_guesses=prev_guesses,
            turn=turn,
            max_turns=max_turns,
            target_reached=turn_target_reached,
        )
        per_turn_rewards.append(tr)

    # Add episode reward to last turn
    ep_reward, _ = reward_mod.compute_episode_reward(
        target_reached=target_reached,
        total_turns=total_turns,
        max_turns=max_turns,
    )
    per_turn_rewards[-1] += ep_reward

    return per_turn_rewards


# ---------------------------------------------------------------------------
# GRPO step
# ---------------------------------------------------------------------------

def grpo_step(
    step: int,
    batch_targets: list[str],
    valid_guesses: set[str],
    sampling_client,
    training_client,
    tokenizer,
    args,
) -> dict:
    """Execute one GRPO step over a batch of target words.

    For each target, runs group_size episodes, computes rewards, advantages,
    and trains.  Rollouts are parallelized by turn depth: at each turn, all
    active episodes fire sampling requests concurrently via Tinker futures.
    """
    t0 = time.time()
    t_gen_start = time.time()

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_completion_tokens,
        temperature=args.temperature,
    )

    # Initialize all episode states
    n_total = len(batch_targets) * args.group_size
    episodes_state: list[dict] = []
    for target in batch_targets:
        for _ in range(args.group_size):
            episodes_state.append({
                "target": target,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Guess a 5-letter word. Reply with <guess>WORD</guess>."},
                ],
                "history": [],
                "prompt_tokens_per_turn": [],
                "completion_tokens_per_turn": [],
                "logprobs_per_turn": [],
                "target_reached": False,
                "done": False,
            })

    # Run turns in batches — all active episodes sample in parallel
    for turn in range(1, args.max_turns + 1):
        active = [i for i, ep in enumerate(episodes_state) if not ep["done"]]
        if not active:
            break

        # Fire all sample requests as futures (parallel)
        futures = []
        for i in active:
            ep = episodes_state[i]
            prompt_text = tokenizer.apply_chat_template(
                ep["messages"], add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_input = tinker.ModelInput.from_ints(prompt_tokens)

            future = sampling_client.sample(
                prompt=prompt_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            futures.append((i, future, prompt_tokens))

        # Collect all results and process feedback
        for i, future, prompt_tokens in futures:
            ep = episodes_state[i]
            target = ep["target"]

            result = future.result()
            seq = result.sequences[0]
            completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)

            ep["prompt_tokens_per_turn"].append(prompt_tokens)
            ep["completion_tokens_per_turn"].append(list(seq.tokens))
            ep["logprobs_per_turn"].append(
                [lp if lp is not None else 0.0 for lp in (seq.logprobs or [])]
            )

            # Extract guess and compute feedback
            guess = _extract_guess(completion_text)

            if guess is None or guess not in valid_guesses:
                guess = guess or "?????"
                feedback = [TileColor.GREY] * 5
                ep["history"].append((guess, feedback))
            else:
                feedback = compute_feedback(guess, target)
                ep["history"].append((guess, feedback))
                if guess == target:
                    ep["target_reached"] = True

            # Build feedback message
            emoji = feedback_to_emoji(feedback)
            remaining = args.max_turns - turn
            if ep["target_reached"]:
                feedback_text = (
                    f"Turn {turn}: {guess.upper()} \u2192 {emoji}  "
                    f"Correct! You got it in {turn} guess(es)!"
                )
            else:
                feedback_text = (
                    f"Turn {turn}: {guess.upper()} \u2192 {emoji}  "
                    f"({remaining} turn(s) remaining)"
                )

            ep["messages"].append({"role": "assistant", "content": completion_text})
            ep["messages"].append({"role": "user", "content": feedback_text})

            if ep["target_reached"] or turn >= args.max_turns:
                ep["done"] = True

    # Convert episode states to the format expected by reward computation
    all_episodes = []
    group_indices = []
    for i, ep in enumerate(episodes_state):
        all_episodes.append({
            "prompt_tokens_per_turn": ep["prompt_tokens_per_turn"],
            "completion_tokens_per_turn": ep["completion_tokens_per_turn"],
            "logprobs_per_turn": ep["logprobs_per_turn"],
            "history": ep["history"],
            "target_reached": ep["target_reached"],
            "total_turns": len(ep["history"]),
        })

    for i in range(len(batch_targets)):
        start = i * args.group_size
        end = start + args.group_size
        group_indices.append((start, end))

    t_gen = time.time() - t_gen_start

    # Compute total reward per episode (sum of per-turn rewards)
    episode_total_rewards = []
    all_per_turn_rewards = []
    for ep in all_episodes:
        per_turn = compute_episode_rewards(ep, args.reward, args.max_turns)
        all_per_turn_rewards.append(per_turn)
        episode_total_rewards.append(sum(per_turn))

    rewards = np.array(episode_total_rewards, dtype=np.float32)

    # Compute advantages per group (mean-center, normalize by std)
    advantages = np.zeros_like(rewards)
    groups_skipped = 0

    for start, end in group_indices:
        group_rewards = rewards[start:end]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std()

        if group_std < 1e-8:
            advantages[start:end] = 0.0
            groups_skipped += 1
        else:
            advantages[start:end] = (group_rewards - group_mean) / (group_std + 1e-8)

    # Build Datum objects — one per turn per episode (with non-zero advantage)
    data = []
    for ep_idx, episode in enumerate(all_episodes):
        if abs(advantages[ep_idx]) < 1e-8:
            continue

        adv_value = float(advantages[ep_idx])

        for turn_idx in range(episode["total_turns"]):
            prompt_tokens = episode["prompt_tokens_per_turn"][turn_idx]
            completion_tokens = episode["completion_tokens_per_turn"][turn_idx]
            sample_logprobs = episode["logprobs_per_turn"][turn_idx]

            n_prompt = len(prompt_tokens)
            n_completion = len(completion_tokens)
            full_tokens = prompt_tokens + completion_tokens

            target_tokens = tinker.TensorData(
                data=full_tokens, dtype="int64",
            )

            token_advantages = tinker.TensorData(
                data=[0.0] * n_prompt + [adv_value] * n_completion,
                dtype="float32",
            )

            # Pad logprobs for prompt positions
            lp_list = [0.0] * n_prompt + sample_logprobs
            # Ensure length matches
            if len(lp_list) < len(full_tokens):
                lp_list.extend([0.0] * (len(full_tokens) - len(lp_list)))
            lp_list = lp_list[:len(full_tokens)]

            logprobs_tensor = tinker.TensorData(
                data=lp_list, dtype="float32",
            )

            datum = tinker.Datum(
                model_input=tinker.ModelInput.from_ints(full_tokens),
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "logprobs": logprobs_tensor,
                    "advantages": token_advantages,
                },
            )
            data.append(datum)

    # Behavioral metrics (computed before early return so they're always available)
    win_rate = np.mean([ep["target_reached"] for ep in all_episodes])
    avg_turns = np.mean([ep["total_turns"] for ep in all_episodes])
    episode_histories = [ep["history"] for ep in all_episodes]
    violation_rate = compute_constraint_violation_rate(episode_histories)

    if not data:
        return {
            "step": step,
            "mean_reward": float(rewards.mean()),
            "loss": 0.0,
            "groups_skipped": groups_skipped,
            "n_train_samples": 0,
            "n_problems": len(batch_targets),
            "win_rate": float(win_rate),
            "avg_turns": float(avg_turns),
            "constraint_violation_rate": float(violation_rate),
            "time_gen": round(t_gen, 2),
            "time_total": round(time.time() - t0, 2),
        }

    # Forward-backward + optimizer step
    t_train_start = time.time()

    fwdbwd_future = training_client.forward_backward(
        data=data, loss_fn=args.loss_fn,
    )
    optim_future = training_client.optim_step(
        tinker.AdamParams(learning_rate=args.lr)
    )

    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    t_train = time.time() - t_train_start

    loss = fwdbwd_result.metrics.get("loss", None)
    if loss is None:
        all_logprobs = []
        for output in fwdbwd_result.loss_fn_outputs:
            if "logprobs" in output:
                all_logprobs.extend(output["logprobs"].data)
        loss = -np.mean(all_logprobs) if all_logprobs else 0.0

    metrics = {
        "step": step,
        "mean_reward": float(rewards.mean()),
        "max_reward": float(rewards.max()),
        "min_reward": float(rewards.min()),
        "reward_std": float(rewards.std()),
        "loss": float(loss),
        "groups_skipped": groups_skipped,
        "n_problems": len(batch_targets),
        "n_train_samples": len(data),
        "win_rate": float(win_rate),
        "avg_turns": float(avg_turns),
        "constraint_violation_rate": float(violation_rate),
        "time_gen": round(t_gen, 2),
        "time_train": round(t_train, 2),
        "time_total": round(time.time() - t0, 2),
    }
    return metrics


# ---------------------------------------------------------------------------
# Trajectory saving (probe words)
# ---------------------------------------------------------------------------

def save_trajectories(
    step: int,
    probe_words: list[str],
    valid_guesses: set[str],
    sampling_client,
    tokenizer,
    args,
    output_dir: str,
) -> None:
    """Run one episode per probe word and save full conversation traces."""
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    episodes_data = []
    for target in probe_words:
        episode = run_episode(
            target=target,
            valid_guesses=valid_guesses,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            args=args,
            max_turns=args.max_turns,
        )

        # Reconstruct readable messages from the episode
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Guess a 5-letter word. Reply with <guess>WORD</guess>."},
        ]
        for turn_idx, (guess, feedback) in enumerate(episode["history"]):
            completion_tokens = episode["completion_tokens_per_turn"][turn_idx]
            completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            messages.append({
                "role": "assistant",
                "content": completion_text,
                "tokens": len(completion_tokens),
            })
            emoji = feedback_to_emoji(feedback)
            remaining = args.max_turns - (turn_idx + 1)
            if episode["target_reached"] and turn_idx == episode["total_turns"] - 1:
                fb_text = (
                    f"Turn {turn_idx + 1}: {guess.upper()} \u2192 {emoji}  "
                    f"Correct! You got it in {turn_idx + 1} guess(es)!"
                )
            else:
                fb_text = (
                    f"Turn {turn_idx + 1}: {guess.upper()} \u2192 {emoji}  "
                    f"({remaining} turn(s) remaining)"
                )
            messages.append({"role": "user", "content": fb_text})

        episodes_data.append({
            "target": target.upper(),
            "solved": episode["target_reached"],
            "turns": episode["total_turns"],
            "messages": messages,
        })

    trace_path = os.path.join(traj_dir, f"step_{step:04d}.json")
    with open(trace_path, "w") as f:
        json.dump({"step": step, "episodes": episodes_data}, f, indent=2, ensure_ascii=False)
    print(f"[wordle] Trajectories saved: {trace_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or f"checkpoints/wordle_{args.reward}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(output_dir, "training_log.jsonl")

    print(f"[wordle] === Multi-turn Wordle GRPO Training ===")
    print(f"[wordle] Reward       : {args.reward}")
    print(f"[wordle] Model        : {args.model}")
    print(f"[wordle] Max turns    : {args.max_turns}")
    print(f"[wordle] LoRA rank    : {args.lora_rank}")
    print(f"[wordle] LR           : {args.lr}")
    print(f"[wordle] Loss fn      : {args.loss_fn}")
    print(f"[wordle] Batch size   : {args.batch_size} words x {args.group_size} episodes")
    print(f"[wordle] Max steps    : {args.max_steps}")
    print(f"[wordle] Output dir   : {output_dir}")

    # Initialize Tinker
    print(f"[wordle] Connecting to Tinker API ...")
    service = ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=args.model, rank=args.lora_rank,
    )
    sampling_client = training_client.save_weights_and_get_sampling_client(name="init")
    tokenizer = training_client.get_tokenizer()
    print(f"[wordle] Tinker clients initialized.")

    # Load word lists
    answers, valid_guesses = load_words(args)

    # Select probe words for trajectory saving
    probe_rng = random.Random(args.seed)
    if args.trace_words:
        probe_words = [w.lower() for w in args.trace_words]
    else:
        probe_words = probe_rng.sample(answers, min(args.n_trace_words, len(answers)))
    print(f"[wordle] Probe words   : {[w.upper() for w in probe_words]}")

    # Save step-0 trajectories (before any training)
    save_trajectories(
        step=0, probe_words=probe_words, valid_guesses=valid_guesses,
        sampling_client=sampling_client, tokenizer=tokenizer,
        args=args, output_dir=output_dir,
    )

    # W&B
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"wordle_grpo_{args.reward}",
            config=vars(args),
        )
        print(f"[wordle] W&B run: {wandb.run.url}")

    # Training loop
    rng = random.Random(args.seed)
    all_metrics = []
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        batch_targets = [rng.choice(answers) for _ in range(args.batch_size)]

        metrics = grpo_step(
            step=step,
            batch_targets=batch_targets,
            valid_guesses=valid_guesses,
            sampling_client=sampling_client,
            training_client=training_client,
            tokenizer=tokenizer,
            args=args,
        )
        all_metrics.append(metrics)

        reward_str = f"{metrics['mean_reward']:.3f}"
        loss_str = f"{metrics['loss']:.4f}"
        win_str = f"{metrics['win_rate']:.1%}"
        viol_str = f"{metrics['constraint_violation_rate']:.1%}"
        time_str = f"{metrics['time_total']:.1f}s"
        print(
            f"[step {step:4d}/{args.max_steps}] "
            f"reward={reward_str}  loss={loss_str}  "
            f"win={win_str}  violations={viol_str}  time={time_str}"
        )

        with open(log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if use_wandb:
            wandb.log(metrics, step=step)

        # Save checkpoint + refresh sampling client
        if step % args.save_steps == 0:
            ckpt_name = f"step_{step:04d}"
            print(f"[wordle] Saving checkpoint: {ckpt_name} ...")
            save_result = training_client.save_state(ckpt_name).result()
            ckpt_path = getattr(save_result, "path", None)
            print(f"[wordle] Checkpoint saved: {ckpt_path}")

            manifest_path = os.path.join(output_dir, "checkpoints.jsonl")
            with open(manifest_path, "a") as f:
                f.write(json.dumps({"step": step, "name": ckpt_name, "path": ckpt_path}) + "\n")

            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=ckpt_name
            )
            print(f"[wordle] Sampling client refreshed.")

            # Save probe word trajectories at this checkpoint
            save_trajectories(
                step=step, probe_words=probe_words, valid_guesses=valid_guesses,
                sampling_client=sampling_client, tokenizer=tokenizer,
                args=args, output_dir=output_dir,
            )

    # Final summary
    elapsed = time.time() - t_start
    mean_rewards = [m["mean_reward"] for m in all_metrics]
    win_rates = [m["win_rate"] for m in all_metrics]
    violation_rates = [m["constraint_violation_rate"] for m in all_metrics]
    print(f"\n[wordle] === Training Complete ===")
    print(f"[wordle] Total time       : {elapsed / 60:.1f} min")
    print(f"[wordle] Final reward     : {mean_rewards[-1]:.3f}")
    print(f"[wordle] Final win rate   : {win_rates[-1]:.1%}")
    print(f"[wordle] Final violations : {violation_rates[-1]:.1%}")

    # Save final
    print(f"[wordle] Saving final model ...")
    save_result = training_client.save_state("final").result()
    ckpt_path = getattr(save_result, "path", None)
    print(f"[wordle] Final checkpoint saved: {ckpt_path}")

    manifest_path = os.path.join(output_dir, "checkpoints.jsonl")
    with open(manifest_path, "a") as f:
        f.write(json.dumps({"step": args.max_steps, "name": "final", "path": ckpt_path}) + "\n")

    if use_wandb:
        wandb.finish()

    print(f"[wordle] Done.")


if __name__ == "__main__":
    main()
