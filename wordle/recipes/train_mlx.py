"""MLX-based GRPO training for Wordle on Apple Silicon.

Multi-turn interactive episodes with dense or sparse rewards, using LoRA
fine-tuning. Mirrors wordle/recipes/train.py (Tinker) structure so results
are directly comparable.

Usage:
    python -m wordle.recipes.train_mlx --reward {dense,sparse} [options]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from wordle.environment.feedback import TileColor, compute_feedback, feedback_to_emoji
from wordle.environment.constraints import compute_constraint_violation_rate
from wordle.environment.wordle_env import load_word_list, _extract_guess, SYSTEM_PROMPT
from wordle.rewards import dense_reward, sparse_reward
from wordle.recipes.mlx_utils import (
    load_model_with_lora,
    generate_with_logprobs,
    compute_logprobs_for_sequence,
    save_checkpoint,
    load_checkpoint,
    clear_cache,
    get_memory_stats,
)
from wordle.recipes.mlx_grpo import compute_advantages, grpo_step


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLX-based multi-turn Wordle GRPO training for Apple Silicon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reward", required=True, choices=["dense", "sparse"],
        help="Reward function: 'dense' for per-turn shaping, 'sparse' for terminal-only.",
    )
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-0.6B")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="Episodes per micro-batch for gradient accumulation.")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_layers", type=int, default=16,
                        help="Number of layers to apply LoRA to.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient.")
    parser.add_argument("--clip_low", type=float, default=0.8)
    parser.add_argument("--clip_high", type=float, default=1.2)
    parser.add_argument("--max_completion_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--answers_path", type=str, default=None)
    parser.add_argument("--guesses_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint directory to resume from.")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="discovery-wordle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace_words", type=str, nargs="*", default=None)
    parser.add_argument("--n_trace_words", type=int, default=5)
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
# Multi-turn episode rollout (MLX local generation)
# ---------------------------------------------------------------------------

def run_episode_mlx(
    target: str,
    valid_guesses: set[str],
    model,
    tokenizer,
    args,
    max_turns: int = 6,
) -> dict:
    """Run one multi-turn Wordle episode using local MLX generation.

    Returns dict matching the format from train.py's run_episode.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Guess a 5-letter word."},
    ]

    history: list[tuple[str, list[TileColor]]] = []
    completion_tokens_per_turn: list[list[int]] = []
    completion_texts: list[str] = []
    logprobs_per_turn: list[list[float]] = []
    prompt_tokens_per_turn: list[list[int]] = []
    target_reached = False

    for turn in range(1, max_turns + 1):
        # Encode current conversation as prompt
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Generate with logprobs
        gen_tokens, gen_logprobs, completion_text = generate_with_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=args.max_completion_tokens,
            temperature=args.temperature,
        )

        prompt_tokens_per_turn.append(prompt_tokens)
        completion_tokens_per_turn.append(gen_tokens)
        completion_texts.append(completion_text)
        logprobs_per_turn.append(gen_logprobs)

        # Extract guess
        guess = _extract_guess(completion_text)

        if guess is None or guess not in valid_guesses:
            guess = guess or "?????"
            feedback = [TileColor.GREY] * 5
            history.append((guess, feedback))
        else:
            feedback = compute_feedback(guess, target)
            history.append((guess, feedback))
            if guess == target:
                target_reached = True

        # Build feedback message
        emoji = feedback_to_emoji(feedback)
        remaining = max_turns - turn
        if target_reached:
            feedback_text = (
                f"Turn {turn}: {guess.upper()} → {emoji}  "
                f"Correct! You got it in {turn} guess(es)!"
            )
        else:
            feedback_text = (
                f"Turn {turn}: {guess.upper()} → {emoji}  "
                f"({remaining} turn(s) remaining)"
            )

        messages.append({"role": "assistant", "content": completion_text})
        messages.append({"role": "user", "content": feedback_text})

        if target_reached or turn >= max_turns:
            break

        # Clear cache between turns to prevent memory buildup
        clear_cache()

    return {
        "prompt_tokens_per_turn": prompt_tokens_per_turn,
        "completion_tokens_per_turn": completion_tokens_per_turn,
        "completion_texts": completion_texts,
        "logprobs_per_turn": logprobs_per_turn,
        "history": history,
        "target_reached": target_reached,
        "total_turns": len(history),
    }


# ---------------------------------------------------------------------------
# Reward computation (reused from train.py)
# ---------------------------------------------------------------------------

def compute_episode_rewards(
    episode: dict,
    reward_type: str,
    max_turns: int,
) -> tuple[list[float], dict[str, float]]:
    """Compute per-turn rewards for an episode."""
    history = episode["history"]
    target_reached = episode["target_reached"]
    total_turns = episode["total_turns"]
    completion_texts = episode.get("completion_texts", [""] * total_turns)

    reward_mod = dense_reward if reward_type == "dense" else sparse_reward
    per_turn_rewards = []
    format_compliant_turns = 0

    for turn_idx, (guess, feedback) in enumerate(history):
        turn = turn_idx + 1
        prev_guesses = [g for g, _ in history[:turn_idx]]
        prev_feedbacks = [f for _, f in history[:turn_idx]]

        is_final_turn = (turn == total_turns)
        turn_target_reached = target_reached and is_final_turn

        completion_text = completion_texts[turn_idx] if turn_idx < len(completion_texts) else ""

        tr, turn_metrics = reward_mod.compute_turn_reward(
            guess=guess,
            feedback=feedback,
            prev_feedbacks=prev_feedbacks,
            prev_guesses=prev_guesses,
            turn=turn,
            max_turns=max_turns,
            target_reached=turn_target_reached,
            completion_text=completion_text,
        )
        per_turn_rewards.append(tr)
        if turn_metrics.get("format_compliance", 0.0) > 0:
            format_compliant_turns += 1

    # Add episode reward to last turn
    ep_reward, _ = reward_mod.compute_episode_reward(
        target_reached=target_reached,
        total_turns=total_turns,
        max_turns=max_turns,
    )
    per_turn_rewards[-1] += ep_reward

    ep_metrics = {
        "format_compliant_turns": format_compliant_turns,
        "total_turns": total_turns,
    }
    return per_turn_rewards, ep_metrics


# ---------------------------------------------------------------------------
# Trajectory saving
# ---------------------------------------------------------------------------

def save_trajectories(
    step: int,
    probe_words: list[str],
    valid_guesses: set[str],
    model,
    tokenizer,
    args,
    output_dir: str,
) -> None:
    """Run one episode per probe word and save conversation traces."""
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    print(f"[wordle] Saving trajectories for step {step} ({len(probe_words)} probe words)...", flush=True)
    t_start = time.time()

    episodes_data = []
    for pi, target in enumerate(probe_words):
        t_ep = time.time()
        episode = run_episode_mlx(
            target=target,
            valid_guesses=valid_guesses,
            model=model,
            tokenizer=tokenizer,
            args=args,
            max_turns=args.max_turns,
        )

        # Reconstruct readable messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Guess a 5-letter word."},
        ]
        for turn_idx, (guess, feedback) in enumerate(episode["history"]):
            completion_text = episode["completion_texts"][turn_idx]
            messages.append({
                "role": "assistant",
                "content": completion_text,
                "tokens": len(episode["completion_tokens_per_turn"][turn_idx]),
            })
            emoji = feedback_to_emoji(feedback)
            remaining = args.max_turns - (turn_idx + 1)
            if episode["target_reached"] and turn_idx == episode["total_turns"] - 1:
                fb_text = (
                    f"Turn {turn_idx + 1}: {guess.upper()} → {emoji}  "
                    f"Correct! You got it in {turn_idx + 1} guess(es)!"
                )
            else:
                fb_text = (
                    f"Turn {turn_idx + 1}: {guess.upper()} → {emoji}  "
                    f"({remaining} turn(s) remaining)"
                )
            messages.append({"role": "user", "content": fb_text})

        solved_str = "solved" if episode["target_reached"] else "failed"
        print(
            f"  [trace {pi+1}/{len(probe_words)}] {target.upper()} → "
            f"{episode['total_turns']} turns, {solved_str} ({time.time() - t_ep:.1f}s)",
            flush=True,
        )

        episodes_data.append({
            "target": target.upper(),
            "solved": episode["target_reached"],
            "turns": episode["total_turns"],
            "messages": messages,
        })

    trace_path = os.path.join(traj_dir, f"step_{step:04d}.json")
    with open(trace_path, "w") as f:
        json.dump({"step": step, "episodes": episodes_data}, f, indent=2, ensure_ascii=False)
    print(f"[wordle] Trajectories saved: {trace_path} ({time.time() - t_start:.1f}s total)", flush=True)

    clear_cache()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or f"checkpoints/wordle_mlx_{args.reward}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(output_dir, "training_log.jsonl")

    print(f"[wordle] === MLX Multi-turn Wordle GRPO Training ===")
    print(f"[wordle] Reward       : {args.reward}")
    print(f"[wordle] Model        : {args.model}")
    print(f"[wordle] Max turns    : {args.max_turns}")
    print(f"[wordle] LoRA rank    : {args.lora_rank}")
    print(f"[wordle] LoRA layers  : {args.lora_layers}")
    print(f"[wordle] LR           : {args.lr}")
    print(f"[wordle] KL coef      : {args.beta}")
    print(f"[wordle] CISPO clip   : [{args.clip_low}, {args.clip_high}]")
    print(f"[wordle] Grad clip    : {args.grad_clip_norm}")
    print(f"[wordle] Batch size   : {args.batch_size} words x {args.group_size} episodes")
    print(f"[wordle] Micro-batch  : {args.micro_batch_size}")
    print(f"[wordle] Max steps    : {args.max_steps}")
    print(f"[wordle] Output dir   : {output_dir}")

    # Load model with LoRA
    print(f"[wordle] Loading model with LoRA ...")
    model, tokenizer = load_model_with_lora(
        model_path=args.model,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
    )
    mem = get_memory_stats()
    print(f"[wordle] Memory after load: {mem['active_gb']:.2f} GB active, {mem['peak_gb']:.2f} GB peak")

    # Set up optimizer
    optimizer = optim.Adam(learning_rate=args.lr)

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        print(f"[wordle] Resuming from {args.resume_from} ...")
        start_step = load_checkpoint(model, optimizer, args.resume_from)
        print(f"[wordle] Resuming from step {start_step}")

    # Load word lists
    answers, valid_guesses = load_words(args)

    # Select probe words for trajectory saving
    probe_rng = random.Random(args.seed)
    if args.trace_words:
        probe_words = [w.lower() for w in args.trace_words]
    else:
        probe_words = probe_rng.sample(answers, min(args.n_trace_words, len(answers)))
    print(f"[wordle] Probe words   : {[w.upper() for w in probe_words]}")

    # Save step-0 trajectories (before any training) if not resuming
    if start_step == 0:
        save_trajectories(
            step=0, probe_words=probe_words, valid_guesses=valid_guesses,
            model=model, tokenizer=tokenizer,
            args=args, output_dir=output_dir,
        )

    # W&B
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"wordle_mlx_{args.reward}",
            config=vars(args),
            resume="allow" if args.resume_from else None,
        )
        print(f"[wordle] W&B run: {wandb.run.url}")

    # Training loop
    rng = random.Random(args.seed)
    # Advance RNG past already-completed steps
    for _ in range(start_step):
        for _ in range(args.batch_size):
            rng.choice(answers)

    all_metrics = []
    t_start = time.time()

    for step in range(start_step + 1, args.max_steps + 1):
        t0 = time.time()
        batch_targets = [rng.choice(answers) for _ in range(args.batch_size)]

        # --- ROLLOUT ---
        t_gen_start = time.time()
        all_episodes = []
        group_indices = []

        for i, target in enumerate(batch_targets):
            start_idx = len(all_episodes)
            for _ in range(args.group_size):
                episode = run_episode_mlx(
                    target=target,
                    valid_guesses=valid_guesses,
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    max_turns=args.max_turns,
                )
                all_episodes.append(episode)
                clear_cache()

            end_idx = len(all_episodes)
            group_indices.append((start_idx, end_idx))

        t_gen = time.time() - t_gen_start
        n_total = len(all_episodes)

        # --- REWARD ---
        episode_total_rewards = []
        all_per_turn_rewards = []
        total_format_compliant = 0
        total_episode_turns = 0

        for ep in all_episodes:
            per_turn, ep_metrics = compute_episode_rewards(ep, args.reward, args.max_turns)
            all_per_turn_rewards.append(per_turn)
            episode_total_rewards.append(sum(per_turn))
            total_format_compliant += ep_metrics["format_compliant_turns"]
            total_episode_turns += ep_metrics["total_turns"]

        rewards = np.array(episode_total_rewards, dtype=np.float32)

        # --- ADVANTAGE ---
        advantages, groups_skipped = compute_advantages(rewards, args.group_size)

        # Collect old logprobs as mx.arrays for GRPO loss
        old_logprobs_batch = []
        for ep in all_episodes:
            turn_lps = []
            for turn_idx in range(ep["total_turns"]):
                lps = ep["logprobs_per_turn"][turn_idx]
                turn_lps.append(mx.array(lps))
            old_logprobs_batch.append(turn_lps)

        advantages_list = [float(a) for a in advantages]

        # Behavioral metrics
        win_rate = np.mean([ep["target_reached"] for ep in all_episodes])
        avg_turns = np.mean([ep["total_turns"] for ep in all_episodes])
        episode_histories = [ep["history"] for ep in all_episodes]
        violation_rate = compute_constraint_violation_rate(episode_histories)
        format_compliance_rate = (
            total_format_compliant / total_episode_turns
            if total_episode_turns > 0 else 0.0
        )

        # --- TRAIN ---
        t_train_start = time.time()

        # Filter to episodes with non-zero advantage
        train_episodes = []
        train_old_lps = []
        train_advs = []
        for i in range(n_total):
            if abs(advantages_list[i]) >= 1e-8:
                train_episodes.append(all_episodes[i])
                train_old_lps.append(old_logprobs_batch[i])
                train_advs.append(advantages_list[i])

        if train_episodes:
            train_result = grpo_step(
                model=model,
                optimizer=optimizer,
                episodes_batch=train_episodes,
                old_logprobs_batch=train_old_lps,
                advantages_batch=train_advs,
                clip_low=args.clip_low,
                clip_high=args.clip_high,
                beta=args.beta,
                grad_clip_norm=args.grad_clip_norm,
                micro_batch_size=args.micro_batch_size,
            )
            loss = train_result["loss"]
            grad_norm = train_result["grad_norm"]
        else:
            loss = 0.0
            grad_norm = 0.0

        t_train = time.time() - t_train_start
        clear_cache()

        # --- METRICS ---
        mem = get_memory_stats()

        abs_advs = [abs(float(a)) for a in advantages]
        pos_advs = sum(1 for a in advantages if float(a) > 0)
        mean_abs_advantage = float(np.mean(abs_advs)) if abs_advs else 0.0
        frac_positive_adv = pos_advs / len(advantages) if len(advantages) > 0 else 0.0

        completion_lens = []
        for ep in all_episodes:
            for toks in ep["completion_tokens_per_turn"]:
                completion_lens.append(len(toks))
        mean_completion_len = float(np.mean(completion_lens)) if completion_lens else 0.0

        metrics = {
            "step": step,
            "mean_reward": float(rewards.mean()),
            "max_reward": float(rewards.max()),
            "min_reward": float(rewards.min()),
            "reward_std": float(rewards.std()),
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "groups_skipped": groups_skipped,
            "n_problems": len(batch_targets),
            "n_train_samples": len(train_episodes),
            "win_rate": float(win_rate),
            "avg_turns": float(avg_turns),
            "constraint_violation_rate": float(violation_rate),
            "format_compliance_rate": float(format_compliance_rate),
            "mean_abs_advantage": mean_abs_advantage,
            "frac_positive_adv": frac_positive_adv,
            "mean_completion_len": mean_completion_len,
            "memory_active_gb": mem["active_gb"],
            "memory_peak_gb": mem["peak_gb"],
            "time_gen": round(t_gen, 2),
            "time_train": round(t_train, 2),
            "time_total": round(time.time() - t0, 2),
        }
        all_metrics.append(metrics)

        # --- LOG ---
        reward_str = f"{metrics['mean_reward']:.3f}"
        loss_str = f"{metrics['loss']:.4f}"
        win_str = f"{metrics['win_rate']:.1%}"
        viol_str = f"{metrics['constraint_violation_rate']:.1%}"
        fmt_str = f"{metrics['format_compliance_rate']:.1%}"
        mem_str = f"{mem['active_gb']:.1f}GB"
        time_str = f"{metrics['time_total']:.1f}s"
        print(
            f"[step {step:4d}/{args.max_steps}] "
            f"reward={reward_str}  loss={loss_str}  "
            f"win={win_str}  violations={viol_str}  format={fmt_str}  "
            f"mem={mem_str}  time={time_str}"
        )

        # Flush to JSONL immediately for crash recovery
        with open(log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
            f.flush()

        if use_wandb:
            wandb.log(metrics, step=step)

        # Memory warning
        if mem["active_gb"] > 6.0:
            print(
                f"[wordle] WARNING: Active memory {mem['active_gb']:.2f} GB "
                f"exceeds 6 GB threshold. Consider reducing batch/group size."
            )

        # --- CHECKPOINT ---
        if step % args.save_steps == 0:
            ckpt_path = os.path.join(output_dir, f"step_{step:04d}")
            save_checkpoint(model, optimizer, step, ckpt_path, metrics)

            # Save probe word trajectories
            save_trajectories(
                step=step, probe_words=probe_words, valid_guesses=valid_guesses,
                model=model, tokenizer=tokenizer,
                args=args, output_dir=output_dir,
            )

    # --- FINAL ---
    elapsed = time.time() - t_start
    if all_metrics:
        mean_rewards = [m["mean_reward"] for m in all_metrics]
        win_rates = [m["win_rate"] for m in all_metrics]
        violation_rates = [m["constraint_violation_rate"] for m in all_metrics]
        print(f"\n[wordle] === Training Complete ===")
        print(f"[wordle] Total time       : {elapsed / 60:.1f} min")
        print(f"[wordle] Final reward     : {mean_rewards[-1]:.3f}")
        print(f"[wordle] Final win rate   : {win_rates[-1]:.1%}")
        print(f"[wordle] Final violations : {violation_rates[-1]:.1%}")

    # Save final checkpoint
    final_path = os.path.join(output_dir, "final")
    save_checkpoint(model, optimizer, args.max_steps, final_path)

    if use_wandb:
        wandb.finish()

    print(f"[wordle] Done.")


if __name__ == "__main__":
    main()
