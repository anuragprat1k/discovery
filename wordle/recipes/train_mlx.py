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
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from wordle.environment.feedback import TileColor, compute_feedback, feedback_to_emoji
from wordle.environment.constraints import compute_constraint_violation_rate, RevealedConstraints
from wordle.environment.wordle_env import load_word_list, _extract_guess, SYSTEM_PROMPT, SYSTEM_PROMPT_ENV_ELIMINATED, SYSTEM_PROMPT_THINKING
from wordle.rewards import dense_reward, sparse_reward
from wordle.recipes.mlx_utils import (
    load_model_with_lora,
    generate_with_logprobs,
    generate_batch_with_logprobs,
    compute_logprobs_for_sequence,
    save_checkpoint,
    load_checkpoint,
    clear_cache,
    get_memory_stats,
)
from wordle.recipes.mlx_grpo import compute_advantages, grpo_step
from wordle.recipes.sft_data import generate_sft_examples


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
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-4B-bf16")
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
    parser.add_argument("--no_batch_inference", action="store_true",
                        help="Disable batched inference (use sequential rollouts).")
    parser.add_argument("--inference_batch_size", type=int, default=0,
                        help="Max sequences per batch call (0 = all at once).")
    parser.add_argument("--sft_warmup_steps", type=int, default=0,
                        help="SFT warmup steps for eliminated-letter tracking before GRPO.")
    parser.add_argument("--sft_batch_size", type=int, default=8,
                        help="Batch size for SFT warmup (can be larger than GRPO batch).")
    parser.add_argument("--sft_warmup_data", type=str, default=None,
                        help="JSONL file for SFT warmup data (default: None = use synthetic).")
    parser.add_argument("--expert_buffer_path", type=str, default=None,
                        help="Path to JSONL expert replay buffer (enables SFT replay).")
    parser.add_argument("--sft_replay_batch_size", type=int, default=4,
                        help="Examples per SFT replay step.")
    parser.add_argument("--env_eliminated", action="store_true",
                        help="Environment provides eliminated letters in feedback (model just guesses).")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable Qwen3 thinking mode (<think>...</think> before answer).")
    parser.add_argument("--max_step_time", type=int, default=600,
                        help="Abort training if a single step exceeds this many seconds (swap thrashing guard).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prompt / thinking mode helpers
# ---------------------------------------------------------------------------

def get_system_prompt(args) -> str:
    """Select system prompt based on flags."""
    if getattr(args, "thinking", False):
        return SYSTEM_PROMPT_THINKING
    if getattr(args, "env_eliminated", False):
        return SYSTEM_PROMPT_ENV_ELIMINATED
    return SYSTEM_PROMPT


def use_thinking(args) -> bool:
    """Whether to enable Qwen3 thinking mode in chat template."""
    return getattr(args, "thinking", False)


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
    env_eliminated = getattr(args, "env_eliminated", False)
    sys_prompt = get_system_prompt(args)
    messages = [
        {"role": "system", "content": sys_prompt},
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
            enable_thinking=use_thinking(args),
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
            # Append eliminated letters from env
            if env_eliminated:
                constraints = RevealedConstraints.from_history(
                    [g for g, _ in history], [f for _, f in history],
                )
                elim_str = ", ".join(c.upper() for c in sorted(constraints.grey))
                feedback_text += f"\nEliminated letters: {elim_str}"

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
# Batched multi-turn episode rollout (turn-level parallelism)
# ---------------------------------------------------------------------------

def run_episodes_batched_mlx(
    batch_targets: list[str],
    valid_guesses: set[str],
    model,
    tokenizer,
    args,
    inference_batch_size: int = 0,
) -> list[dict]:
    """Run batch_size * group_size episodes with turn-level batched generation.

    Mirrors the Tinker turn-level batching pattern: at each turn depth, all
    active (not yet done) episodes generate in parallel via BatchKVCache.

    Returns list of episode dicts with same format as run_episode_mlx().
    """
    max_turns = args.max_turns
    N = len(batch_targets) * args.group_size
    env_eliminated = getattr(args, "env_eliminated", False)
    sys_prompt = get_system_prompt(args)

    # Initialize episode states
    episodes = []
    for target in batch_targets:
        for _ in range(args.group_size):
            episodes.append({
                "target": target,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "Guess a 5-letter word."},
                ],
                "history": [],
                "completion_tokens_per_turn": [],
                "completion_texts": [],
                "logprobs_per_turn": [],
                "prompt_tokens_per_turn": [],
                "target_reached": False,
                "done": False,
            })

    for turn in range(1, max_turns + 1):
        # Collect active episode indices
        active_indices = [i for i in range(N) if not episodes[i]["done"]]
        if not active_indices:
            break

        # Build prompt tokens for each active episode
        active_prompts = []
        for i in active_indices:
            ep = episodes[i]
            prompt_text = tokenizer.apply_chat_template(
                ep["messages"], add_generation_prompt=True, tokenize=False,
                enable_thinking=use_thinking(args),
            )
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            ep["prompt_tokens_per_turn"].append(prompt_tokens)
            active_prompts.append(prompt_tokens)

        # Batched generation (split into chunks if inference_batch_size > 0)
        ibs = inference_batch_size if inference_batch_size > 0 else len(active_prompts)
        all_results = []
        for chunk_start in range(0, len(active_prompts), ibs):
            chunk = active_prompts[chunk_start:chunk_start + ibs]
            chunk_results = generate_batch_with_logprobs(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens_list=chunk,
                max_tokens=args.max_completion_tokens,
                temperature=args.temperature,
            )
            all_results.extend(chunk_results)

        # Process results for each active episode
        for result_idx, ep_idx in enumerate(active_indices):
            ep = episodes[ep_idx]
            gen_tokens, gen_logprobs, completion_text = all_results[result_idx]

            ep["completion_tokens_per_turn"].append(gen_tokens)
            ep["completion_texts"].append(completion_text)
            ep["logprobs_per_turn"].append(gen_logprobs)

            # Extract guess and compute feedback
            guess = _extract_guess(completion_text)
            target = ep["target"]

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
            remaining = max_turns - turn
            if ep["target_reached"]:
                feedback_text = (
                    f"Turn {turn}: {guess.upper()} → {emoji}  "
                    f"Correct! You got it in {turn} guess(es)!"
                )
            else:
                feedback_text = (
                    f"Turn {turn}: {guess.upper()} → {emoji}  "
                    f"({remaining} turn(s) remaining)"
                )
                # Append eliminated letters from env
                if env_eliminated:
                    constraints = RevealedConstraints.from_history(
                        [g for g, _ in ep["history"]], [f for _, f in ep["history"]],
                    )
                    elim_str = ", ".join(c.upper() for c in sorted(constraints.grey))
                    feedback_text += f"\nEliminated letters: {elim_str}"

            ep["messages"].append({"role": "assistant", "content": completion_text})
            ep["messages"].append({"role": "user", "content": feedback_text})

            if ep["target_reached"] or turn >= max_turns:
                ep["done"] = True

        clear_cache()

    # Build return dicts (same format as run_episode_mlx)
    results = []
    for ep in episodes:
        results.append({
            "prompt_tokens_per_turn": ep["prompt_tokens_per_turn"],
            "completion_tokens_per_turn": ep["completion_tokens_per_turn"],
            "completion_texts": ep["completion_texts"],
            "logprobs_per_turn": ep["logprobs_per_turn"],
            "history": ep["history"],
            "target_reached": ep["target_reached"],
            "total_turns": len(ep["history"]),
        })
    return results


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
        completion_tokens_list = episode.get("completion_tokens_per_turn", [])
        n_completion_tokens = len(completion_tokens_list[turn_idx]) if turn_idx < len(completion_tokens_list) else None

        tr, turn_metrics = reward_mod.compute_turn_reward(
            guess=guess,
            feedback=feedback,
            prev_feedbacks=prev_feedbacks,
            prev_guesses=prev_guesses,
            turn=turn,
            max_turns=max_turns,
            target_reached=turn_target_reached,
            completion_text=completion_text,
            completion_tokens=n_completion_tokens,
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
# SFT warmup step
# ---------------------------------------------------------------------------

def sft_step(
    model,
    tokenizer,
    batch: list[dict],
    optimizer,
    grad_clip_norm: float = 1.0,
) -> float:
    """One SFT gradient step: cross-entropy loss on completion tokens only.

    Each item in batch has 'messages' (prompt) and 'completion' (target text).
    Returns scalar loss value.
    """

    # Pre-tokenize batch outside the loss function
    tokenized_batch = []
    for example in batch:
        prompt_text = tokenizer.apply_chat_template(
            example["messages"], add_generation_prompt=True,
            tokenize=False, enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_tokens = tokenizer.encode(example["completion"], add_special_tokens=False)
        if len(completion_tokens) == 0:
            continue
        tokenized_batch.append({
            "full_tokens": prompt_tokens + completion_tokens,
            "prompt_len": len(prompt_tokens),
        })

    if not tokenized_batch:
        return 0.0

    # Use closure form matching grpo_step pattern — avoids gradient tree
    # structure mismatch with optimizer state for frozen parameters
    def _sft_loss():
        total_loss = mx.array(0.0)
        n_tokens = 0
        for item in tokenized_batch:
            logprobs = compute_logprobs_for_sequence(
                model, item["full_tokens"], item["prompt_len"],
            )
            total_loss = total_loss - mx.sum(logprobs)
            n_tokens += logprobs.size
        if n_tokens == 0:
            return mx.array(0.0)
        return total_loss / n_tokens

    loss_and_grad_fn = nn.value_and_grad(model, _sft_loss)
    loss, grads = loss_and_grad_fn()
    mx.eval(loss, grads)

    # Gradient clipping — scale in-place to preserve tree structure
    grad_norm_sq = sum(
        mx.sum(g * g).item()
        for _, g in tree_flatten(grads)
        if isinstance(g, mx.array)
    )
    grad_norm = grad_norm_sq ** 0.5
    if grad_clip_norm > 0 and grad_norm > grad_clip_norm:
        scale = grad_clip_norm / grad_norm
        def _scale_tree(node):
            if isinstance(node, dict):
                return {k: _scale_tree(v) for k, v in node.items()}
            elif isinstance(node, list):
                return [_scale_tree(v) for v in node]
            elif isinstance(node, mx.array):
                return node * scale
            return node
        grads = _scale_tree(grads)

    model.update(optimizer.apply_gradients(grads, model))
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


# ---------------------------------------------------------------------------
# Expert replay buffer helpers
# ---------------------------------------------------------------------------

def load_replay_buffer(path: str) -> list[dict]:
    """Load SFT examples from a JSONL replay buffer file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def episode_to_sft_examples(episode: dict, args) -> list[dict]:
    """Convert a winning episode to multi-turn SFT examples.

    For each turn, creates {"messages": [...], "completion": "..."} where
    the completion is the model's actual output that led to a win.
    """
    sys_prompt = get_system_prompt(args)
    history = episode["history"]
    completion_texts = episode["completion_texts"]
    max_turns = args.max_turns
    env_eliminated = getattr(args, "env_eliminated", False)

    examples = []
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "Guess a 5-letter word."},
    ]

    for turn_idx, (guess, feedback) in enumerate(history):
        turn = turn_idx + 1
        completion_text = completion_texts[turn_idx]

        # Save SFT example: current messages as prompt, model output as completion
        examples.append({
            "messages": [dict(m) for m in messages],
            "completion": completion_text,
        })

        # Advance conversation
        messages.append({"role": "assistant", "content": completion_text})
        emoji = feedback_to_emoji(feedback)
        remaining = max_turns - turn
        if episode["target_reached"] and turn_idx == len(history) - 1:
            fb_text = (
                f"Turn {turn}: {guess.upper()} → {emoji}  "
                f"Correct! You got it in {turn} guess(es)!"
            )
        else:
            fb_text = (
                f"Turn {turn}: {guess.upper()} → {emoji}  "
                f"({remaining} turn(s) remaining)"
            )
            if env_eliminated:
                constraints = RevealedConstraints.from_history(
                    [g for g, _ in history[:turn_idx + 1]],
                    [f for _, f in history[:turn_idx + 1]],
                )
                elim_str = ", ".join(c.upper() for c in sorted(constraints.grey))
                fb_text += f"\nEliminated letters: {elim_str}"
        messages.append({"role": "user", "content": fb_text})

    return examples


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

    # Use batched inference for probe words (group_size=1 per target)
    # Create a temporary args with group_size=1 for trajectory probes
    class _ProbeArgs:
        pass
    probe_args = _ProbeArgs()
    for k, v in vars(args).items():
        setattr(probe_args, k, v)
    probe_args.group_size = 1

    use_batch = not getattr(args, "no_batch_inference", False)
    if use_batch:
        all_probe_episodes = run_episodes_batched_mlx(
            batch_targets=probe_words,
            valid_guesses=valid_guesses,
            model=model,
            tokenizer=tokenizer,
            args=probe_args,
            inference_batch_size=getattr(args, "inference_batch_size", 0),
        )
    else:
        all_probe_episodes = []
        for target in probe_words:
            episode = run_episode_mlx(
                target=target,
                valid_guesses=valid_guesses,
                model=model,
                tokenizer=tokenizer,
                args=args,
                max_turns=args.max_turns,
            )
            all_probe_episodes.append(episode)
            clear_cache()

    env_eliminated = getattr(args, "env_eliminated", False)
    sys_prompt = get_system_prompt(args)

    episodes_data = []
    for pi, (target, episode) in enumerate(zip(probe_words, all_probe_episodes)):
        # Reconstruct readable messages
        messages = [
            {"role": "system", "content": sys_prompt},
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
                if env_eliminated:
                    all_guesses = [g for g, _ in episode["history"][:turn_idx + 1]]
                    all_fbs = [f for _, f in episode["history"][:turn_idx + 1]]
                    constraints = RevealedConstraints.from_history(all_guesses, all_fbs)
                    elim_str = ", ".join(c.upper() for c in sorted(constraints.grey))
                    fb_text += f"\nEliminated letters: {elim_str}"
            messages.append({"role": "user", "content": fb_text})

        solved_str = "solved" if episode["target_reached"] else "failed"
        print(
            f"  [trace {pi+1}/{len(probe_words)}] {target.upper()} → "
            f"{episode['total_turns']} turns, {solved_str}",
            flush=True,
        )

        # Compute per-turn rewards for validation
        reward_type = args.reward
        per_turn_rewards, _ = compute_episode_rewards(episode, reward_type, args.max_turns)

        episodes_data.append({
            "target": target.upper(),
            "solved": episode["target_reached"],
            "turns": episode["total_turns"],
            "messages": messages,
            "per_turn_rewards": per_turn_rewards,
            "reward_type": reward_type,
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
    print(f"[wordle] Max step time: {args.max_step_time}s")
    batch_inf_str = "off" if args.no_batch_inference else (
        f"on (max {args.inference_batch_size})" if args.inference_batch_size > 0 else "on (all)"
    )
    print(f"[wordle] Batch infer  : {batch_inf_str}")
    print(f"[wordle] Max steps    : {args.max_steps}")
    if args.sft_warmup_steps > 0:
        print(f"[wordle] SFT warmup   : {args.sft_warmup_steps} steps (batch={args.sft_batch_size})")
    if args.expert_buffer_path:
        print(f"[wordle] Expert replay: {args.expert_buffer_path} (batch={args.sft_replay_batch_size})")
    if args.env_eliminated:
        print(f"[wordle] Env elim     : ON (env provides eliminated letters)")
    if args.thinking:
        print(f"[wordle] Thinking     : ON (Qwen3 <think> mode)")
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

    # W&B (init before SFT so warmup is logged too)
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"wordle_mlx_{args.reward}",
            config=vars(args),
            resume="allow" if args.resume_from else None,
        )
        print(f"[wordle] W&B run: {wandb.run.url}", flush=True)

    # SFT warmup phase (eliminated-letter tracking)
    if args.sft_warmup_steps > 0 and start_step == 0:
        print(f"\n[sft] === SFT Warmup: {args.sft_warmup_steps} steps, batch_size={args.sft_batch_size} ===", flush=True)
        valid_guesses_list = list(valid_guesses)
        n_sft_examples = args.sft_warmup_steps * args.sft_batch_size
        if args.sft_warmup_data:
            sft_examples = load_replay_buffer(args.sft_warmup_data)
            if len(sft_examples) < n_sft_examples:
                sft_examples = sft_examples * (n_sft_examples // len(sft_examples) + 1)
            random.Random(args.seed).shuffle(sft_examples)
            sft_examples = sft_examples[:n_sft_examples]
            print(f"[sft] Loaded {n_sft_examples} examples from {args.sft_warmup_data}", flush=True)
        else:
            sft_examples = generate_sft_examples(
                answers, valid_guesses_list, n_examples=n_sft_examples, seed=args.seed,
            )
            print(f"[sft] Generated {len(sft_examples)} synthetic examples", flush=True)
        t_sft_start = time.time()
        for sft_step_idx in range(args.sft_warmup_steps):
            t_sft_step = time.time()
            batch = sft_examples[
                sft_step_idx * args.sft_batch_size : (sft_step_idx + 1) * args.sft_batch_size
            ]
            loss = sft_step(model, tokenizer, batch, optimizer, grad_clip_norm=args.grad_clip_norm)
            sft_time = time.time() - t_sft_step
            mem = get_memory_stats()

            sft_metrics = {
                "sft_step": sft_step_idx + 1,
                "sft_loss": loss,
                "sft_time": round(sft_time, 2),
                "memory_active_gb": mem["active_gb"],
                "memory_peak_gb": mem["peak_gb"],
                "phase": "sft_warmup",
            }

            # Log every step to JSONL
            with open(log_file, "a") as f:
                f.write(json.dumps(sft_metrics) + "\n")
                f.flush()

            if use_wandb:
                wandb.log(sft_metrics)

            if (sft_step_idx + 1) % 5 == 0 or sft_step_idx == 0:
                print(
                    f"[sft] step {sft_step_idx+1}/{args.sft_warmup_steps}  "
                    f"loss={loss:.4f}  time={sft_time:.1f}s  mem={mem['active_gb']:.1f}GB",
                    flush=True,
                )
            clear_cache()
        t_sft_elapsed = time.time() - t_sft_start
        print(f"[sft] Warmup complete in {t_sft_elapsed:.1f}s\n", flush=True)
        sft_ckpt_path = os.path.join(output_dir, "sft_checkpoint")
        save_checkpoint(model, optimizer, 0, sft_ckpt_path, {"phase": "post_sft_warmup"})
        print(f"[sft] Saved post-warmup checkpoint to {sft_ckpt_path}", flush=True)

    # Load expert replay buffer
    replay_buffer: list[dict] = []
    replay_buffer_path = None
    if args.expert_buffer_path:
        replay_buffer = load_replay_buffer(args.expert_buffer_path)
        replay_buffer_path = os.path.join(output_dir, "replay_buffer.jsonl")
        # Copy initial buffer to output dir so it grows alongside training
        import shutil
        shutil.copy2(args.expert_buffer_path, replay_buffer_path)
        print(f"[replay] Loaded {len(replay_buffer)} expert examples from {args.expert_buffer_path}")

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
        n_episodes = args.batch_size * args.group_size
        print(
            f"[step {step:4d}] ROLLOUT: generating {n_episodes} episodes "
            f"({args.batch_size} words x {args.group_size} group) ...",
            flush=True,
        )

        if args.no_batch_inference:
            # Sequential fallback
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
        else:
            # Batched turn-level inference
            all_episodes = run_episodes_batched_mlx(
                batch_targets=batch_targets,
                valid_guesses=valid_guesses,
                model=model,
                tokenizer=tokenizer,
                args=args,
                inference_batch_size=args.inference_batch_size,
            )
            group_indices = [
                (i * args.group_size, (i + 1) * args.group_size)
                for i in range(len(batch_targets))
            ]

        t_gen = time.time() - t_gen_start
        n_total = len(all_episodes)
        wins_in_batch = sum(1 for ep in all_episodes if ep["target_reached"])
        print(
            f"[step {step:4d}] ROLLOUT done: {t_gen:.1f}s, "
            f"{wins_in_batch}/{n_total} wins, "
            f"targets={[t.upper() for t in batch_targets]}",
            flush=True,
        )

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
        print(
            f"[step {step:4d}] TRAIN: reward={float(rewards.mean()):.3f} "
            f"(±{float(rewards.std()):.3f}), "
            f"win={float(win_rate):.1%}, viol={float(violation_rate):.1%} ...",
            flush=True,
        )

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
        print(
            f"[step {step:4d}] TRAIN done: {t_train:.1f}s, "
            f"loss={float(loss):.4f}, grad_norm={float(grad_norm):.4f}, "
            f"{len(train_episodes)}/{n_total} samples used",
            flush=True,
        )
        clear_cache()

        # --- SFT REPLAY ---
        replay_loss = 0.0
        replay_new_examples = 0
        if args.expert_buffer_path is not None:
            # Collect winning episodes into replay buffer
            new_replay = 0
            for ep in all_episodes:
                if ep["target_reached"]:
                    sft_examples = episode_to_sft_examples(ep, args)
                    replay_buffer.extend(sft_examples)
                    new_replay += len(sft_examples)
                    # Persist to disk
                    if replay_buffer_path:
                        with open(replay_buffer_path, "a") as f:
                            for ex in sft_examples:
                                f.write(json.dumps(ex) + "\n")
            replay_new_examples = new_replay
            if new_replay > 0:
                print(
                    f"[step {step:4d}] REPLAY: +{new_replay} examples from wins "
                    f"(buffer={len(replay_buffer)} total)",
                    flush=True,
                )

            # SFT replay step
            if replay_buffer:
                batch = random.sample(
                    replay_buffer,
                    min(args.sft_replay_batch_size, len(replay_buffer)),
                )
                replay_loss = sft_step(
                    model, tokenizer, batch, optimizer, args.grad_clip_norm,
                )
                print(
                    f"[step {step:4d}] REPLAY: sft_loss={replay_loss:.4f}",
                    flush=True,
                )
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
            "sft_replay_loss": float(replay_loss),
            "replay_new_examples": replay_new_examples,
            "replay_buffer_size": len(replay_buffer),
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
            wandb.log(metrics)

        # Memory warning
        if mem["active_gb"] > 6.0:
            print(
                f"[wordle] WARNING: Active memory {mem['active_gb']:.2f} GB "
                f"exceeds 6 GB threshold. Consider reducing batch/group size."
            )

        # Step-time watchdog (swap thrashing guard)
        step_time = metrics["time_total"]
        if args.max_step_time > 0 and step_time > args.max_step_time:
            print(
                f"\n[wordle] ABORT: Step {step} took {step_time:.0f}s "
                f"(limit: {args.max_step_time}s). Likely swap thrashing. "
                f"Try --micro_batch_size 2 or reduce --group_size.",
                flush=True,
            )
            # Save emergency checkpoint before aborting
            ckpt_path = os.path.join(output_dir, f"step_{step:04d}")
            if not os.path.exists(ckpt_path):
                save_checkpoint(model, optimizer, step, ckpt_path, metrics)
                print(f"[wordle] Emergency checkpoint saved to {ckpt_path}")
            break

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
