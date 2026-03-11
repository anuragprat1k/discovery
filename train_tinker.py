"""
GRPO training script using the Tinker API.

Mirrors the behavior of train.py (TRL-based) but uses Tinker's low-level API
for training and sampling. This enables:
  - Faster iteration (~10-15x vs RTX 5090)
  - Direct control over advantages (supports per-token/process rewards)
  - Async pipelining for overlapped rollout + training

Target model: Qwen/Qwen3-8B (hosted by Tinker)
Reward functions: reuses rewards/reward_fns.py (binary or dense)

Usage:
    python train_tinker.py --reward {binary,dense} [options]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load .env (handles both "KEY=val" and "export KEY=val")
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
        f"[train_tinker.py] ERROR: Could not import 'tinker'.\n"
        f"  Install with: pip install tinker\n"
        f"  Original error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from rewards.reward_fns import (
        compute_score_binary,
        compute_score_partial_credit,
        compute_score_rubric_sync,
    )
except ImportError as exc:
    print(
        f"[train_tinker.py] ERROR: Could not import reward functions.\n"
        f"  Original error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO training with Tinker API + LoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reward",
        required=True,
        choices=["binary", "dense", "rubric"],
        help="Reward function: 'binary' for sparse 0/1, 'dense' for partial-credit, 'rubric' for LLM judge.",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-8B",
                        help="Model to use as LLM judge (only with --reward rubric).")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--group_size", type=int, default=8,
                        help="Number of completions per problem (G in GRPO).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of problems per step.")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient.")
    parser.add_argument("--max_completion_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--data_path", type=str, default="data/train_128.parquet")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to write per-step metrics as JSONL.")
    parser.add_argument("--loss_fn", type=str, default="importance_sampling",
                        choices=["importance_sampling", "ppo", "cispo"],
                        help="Tinker loss function for policy gradient.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="discovery",
                        help="W&B project name.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tinker client initialization
# ---------------------------------------------------------------------------

def init_tinker_clients(args):
    """Initialize Tinker ServiceClient, TrainingClient, SamplingClient, and optionally a judge client."""
    print(f"[tinker] Connecting to Tinker API ...")
    service = ServiceClient()

    print(f"[tinker] Creating LoRA training client: model={args.model}, rank={args.lora_rank}")
    training_client = service.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank,
    )

    print(f"[tinker] Creating initial sampling client ...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="init"
    )

    tokenizer = training_client.get_tokenizer()

    # Create judge client for rubric reward
    judge_client = None
    judge_tokenizer = None
    if args.reward == "rubric":
        print(f"[tinker] Creating judge sampling client: model={args.judge_model}")
        judge_client = service.create_sampling_client(base_model=args.judge_model)
        if args.judge_model == args.model:
            judge_tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            judge_tokenizer = AutoTokenizer.from_pretrained(
                args.judge_model, trust_remote_code=True
            )

    print(f"[tinker] Clients initialized successfully.")
    return training_client, sampling_client, tokenizer, judge_client, judge_tokenizer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(data_path: str) -> pd.DataFrame:
    """Load training dataset from parquet."""
    if not os.path.isabs(data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, data_path)
        if os.path.exists(candidate):
            data_path = candidate

    if not os.path.exists(data_path):
        print(f"[tinker] ERROR: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(data_path)
    required = {"prompt", "ground_truth", "solution"}
    missing = required - set(df.columns)
    if missing:
        print(f"[tinker] ERROR: Dataset missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"[tinker] Loaded {len(df)} problems from {data_path}")
    return df


# ---------------------------------------------------------------------------
# Reward function wrapper
# ---------------------------------------------------------------------------

def make_reward_fn(reward_type: str):
    """Return the appropriate reward function (for non-rubric types)."""
    if reward_type == "binary":
        return compute_score_binary
    elif reward_type == "dense":
        return compute_score_partial_credit
    else:
        # Rubric is handled separately in grpo_step since it needs judge_client
        return None


# ---------------------------------------------------------------------------
# Core GRPO step
# ---------------------------------------------------------------------------

def grpo_step(
    step: int,
    batch_df: pd.DataFrame,
    sampling_client,
    training_client,
    tokenizer,
    reward_fn,
    args,
    judge_client=None,
    judge_tokenizer=None,
) -> dict:
    """Execute one GRPO training step.

    1. Generate rollouts via sampling_client.sample()
    2. Compute rewards using reward_fn
    3. Compute advantages (mean-center per group, filter constant groups)
    4. Build Datum objects with logprobs + advantages
    5. Call forward_backward() with importance_sampling loss
    6. Call optim_step()

    Returns dict of metrics.
    """
    t0 = time.time()

    # --- 1. Generate rollouts ---
    # sample() takes a single prompt and num_samples, so we call per-problem
    # and collect all results
    all_completions = []      # list of (prompt_tokens, completion_tokens, completion_text)
    ground_truths = []
    solutions = []
    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_completion_tokens,
        temperature=args.temperature,
    )

    t_gen_start = time.time()

    # Launch all sample requests (they return futures)
    sample_futures = []
    prompt_token_lists = []
    for _, row in batch_df.iterrows():
        prompt_messages = row["prompt"]
        if isinstance(prompt_messages, str):
            prompt_messages = json.loads(prompt_messages)

        # Use tokenizer's chat template to encode the prompt
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_input = tinker.ModelInput.from_ints(prompt_tokens)

        future = sampling_client.sample(
            prompt=prompt_input,
            num_samples=args.group_size,
            sampling_params=sampling_params,
        )
        sample_futures.append((future, prompt_tokens, row))
        prompt_token_lists.append(prompt_tokens)

    # Collect results
    for future, prompt_tokens, row in sample_futures:
        result = future.result()
        for seq in result.sequences:
            completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            all_completions.append({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": seq.tokens,
                "completion_text": completion_text,
                "sample_logprobs": seq.logprobs,  # logprobs from sampling policy
            })
            ground_truths.append(row["ground_truth"])
            solutions.append(row.get("solution", ""))

    t_gen = time.time() - t_gen_start

    # --- 2. Compute rewards ---
    completion_texts = [c["completion_text"] for c in all_completions]
    if args.reward == "rubric":
        # Collect problem texts for the rubric prompt
        problems = []
        for _, row in batch_df.iterrows():
            prompt_messages = row["prompt"]
            if isinstance(prompt_messages, str):
                prompt_messages = json.loads(prompt_messages)
            problem_text = prompt_messages[-1]["content"] if prompt_messages else ""
            problems.extend([problem_text] * args.group_size)
        rewards = compute_score_rubric_sync(
            completion_texts,
            ground_truth=ground_truths,
            solution=solutions,
            problem=problems,
            judge_client=judge_client,
            tokenizer=judge_tokenizer,
        )
    elif args.reward == "binary":
        rewards = reward_fn(completion_texts, ground_truth=ground_truths)
    else:
        rewards = reward_fn(completion_texts, ground_truth=ground_truths, solution=solutions)

    rewards = np.array(rewards, dtype=np.float32)

    # --- 3. Compute advantages (mean-center per group, normalize by std) ---
    n_problems = len(batch_df)
    advantages = np.zeros_like(rewards)
    groups_skipped = 0

    for i in range(n_problems):
        start = i * args.group_size
        end = start + args.group_size
        group_rewards = rewards[start:end]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std()

        if group_std < 1e-8:
            # All rewards identical — no gradient signal
            advantages[start:end] = 0.0
            groups_skipped += 1
        else:
            advantages[start:end] = (group_rewards - group_mean) / (group_std + 1e-8)

    # --- 4. Build Datum objects for forward_backward ---
    # For samples without logprobs from sampling (shouldn't happen normally),
    # we need to compute them separately. But sample() returns logprobs
    # in SampledSequence when available.

    # For samples missing logprobs, compute them in batch
    logprob_futures = {}
    for idx, comp in enumerate(all_completions):
        if abs(advantages[idx]) < 1e-8:
            continue
        if comp["sample_logprobs"] is None:
            full_tokens = comp["prompt_tokens"] + list(comp["completion_tokens"])
            logprob_futures[idx] = sampling_client.compute_logprobs(
                tinker.ModelInput.from_ints(full_tokens)
            )

    data = []
    for idx, comp in enumerate(all_completions):
        if abs(advantages[idx]) < 1e-8:
            continue  # Skip zero-advantage samples

        completion_tokens = list(comp["completion_tokens"])
        prompt_tokens = comp["prompt_tokens"]
        n_prompt = len(prompt_tokens)
        n_completion = len(completion_tokens)
        n_total = n_prompt + n_completion

        # Full sequence = prompt + completion
        full_tokens = prompt_tokens + completion_tokens
        model_input = tinker.ModelInput.from_ints(full_tokens)

        # All loss_fn_inputs must have length == n_total (full sequence length).
        # Prompt positions get dummy values; only completion positions matter.

        # target_tokens: prompt tokens followed by completion tokens
        target_tokens = tinker.TensorData(
            data=full_tokens,
            dtype="int64",
        )

        # advantages: 0 for prompt positions, actual value for completion
        adv_value = float(advantages[idx])
        token_advantages = tinker.TensorData(
            data=[0.0] * n_prompt + [adv_value] * n_completion,
            dtype="float32",
        )

        # logprobs: need full-sequence logprobs from the sampling policy
        sample_logprobs = comp["sample_logprobs"]
        if sample_logprobs is not None:
            # sample() returns logprobs only for completion tokens;
            # pad with 0.0 for prompt positions
            lp_list = [0.0] * n_prompt + [
                lp if lp is not None else 0.0 for lp in sample_logprobs
            ]
        else:
            # Use compute_logprobs result (covers full sequence)
            all_lp = logprob_futures[idx].result()
            lp_list = [lp if lp is not None else 0.0 for lp in all_lp]

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
            "mean_reward": float(rewards.mean()),
            "loss": 0.0,
            "groups_skipped": groups_skipped,
            "n_train_samples": 0,
            "time_gen": round(t_gen, 2),
            "time_total": round(time.time() - t0, 2),
        }

    # --- 5. Forward-backward + optimizer step ---
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

    # Extract loss from metrics or compute from logprobs outputs
    loss = fwdbwd_result.metrics.get("loss", None)
    if loss is None:
        # Compute mean loss from loss_fn_outputs logprobs
        all_logprobs = []
        for output in fwdbwd_result.loss_fn_outputs:
            if "logprobs" in output:
                all_logprobs.extend(output["logprobs"].data)
        loss = -np.mean(all_logprobs) if all_logprobs else 0.0

    # --- 6. Collect metrics ---
    metrics = {
        "step": step,
        "mean_reward": float(rewards.mean()),
        "max_reward": float(rewards.max()),
        "min_reward": float(rewards.min()),
        "reward_std": float(rewards.std()),
        "loss": float(loss),
        "groups_skipped": groups_skipped,
        "n_problems": n_problems,
        "n_train_samples": len(data),
        "time_gen": round(t_gen, 2),
        "time_train": round(t_train, 2),
        "time_total": round(time.time() - t0, 2),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or f"checkpoints/tinker_{args.reward}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(output_dir, "training_log.jsonl")

    print(f"[tinker] === GRPO Training with Tinker API ===")
    print(f"[tinker] Reward       : {args.reward}")
    if args.reward == "rubric":
        print(f"[tinker] Judge model  : {args.judge_model}")
    print(f"[tinker] Model        : {args.model}")
    print(f"[tinker] LoRA rank    : {args.lora_rank}")
    print(f"[tinker] LR           : {args.lr}")
    print(f"[tinker] Beta (KL)    : {args.beta}")
    print(f"[tinker] Loss fn      : {args.loss_fn}")
    print(f"[tinker] Batch size   : {args.batch_size} problems x {args.group_size} completions")
    print(f"[tinker] Max steps    : {args.max_steps}")
    print(f"[tinker] Save steps   : {args.save_steps}")
    print(f"[tinker] Output dir   : {output_dir}")
    print(f"[tinker] Log file     : {log_file}")

    # Initialize
    training_client, sampling_client, tokenizer, judge_client, judge_tokenizer = init_tinker_clients(args)
    dataset = load_dataset(args.data_path)
    reward_fn = make_reward_fn(args.reward)

    # W&B logging
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"tinker_grpo_{args.reward}",
            config={
                "reward": args.reward,
                "model": args.model,
                "lora_rank": args.lora_rank,
                "lr": args.lr,
                "beta": args.beta,
                "loss_fn": args.loss_fn,
                "batch_size": args.batch_size,
                "group_size": args.group_size,
                "max_steps": args.max_steps,
                "max_completion_tokens": args.max_completion_tokens,
                "temperature": args.temperature,
                "backend": "tinker",
            },
        )
        print(f"[tinker] W&B run: {wandb.run.url}")
    else:
        print(f"[tinker] W&B logging disabled.")

    # Training loop
    rng = np.random.default_rng(seed=42)
    all_metrics = []
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        # Sample batch of problems (with replacement)
        batch_indices = rng.choice(len(dataset), size=args.batch_size, replace=True)
        batch_df = dataset.iloc[batch_indices].reset_index(drop=True)

        metrics = grpo_step(
            step=step,
            batch_df=batch_df,
            sampling_client=sampling_client,
            training_client=training_client,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            args=args,
            judge_client=judge_client,
            judge_tokenizer=judge_tokenizer,
        )
        all_metrics.append(metrics)

        # Log
        reward_str = f"{metrics['mean_reward']:.3f}"
        loss_str = f"{metrics['loss']:.4f}" if metrics['loss'] is not None else "N/A"
        skip_str = f"{metrics['groups_skipped']}/{args.batch_size}"
        time_str = f"{metrics['time_total']:.1f}s"
        print(
            f"[step {step:4d}/{args.max_steps}] "
            f"reward={reward_str}  loss={loss_str}  "
            f"skipped={skip_str}  time={time_str}"
        )

        # Write to log file
        with open(log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Log to W&B
        if use_wandb:
            wandb.log(metrics, step=step)

        # Save checkpoint + refresh sampling client
        if step % args.save_steps == 0:
            ckpt_name = f"step_{step:04d}"
            print(f"[tinker] Saving checkpoint: {ckpt_name} ...")
            save_result = training_client.save_state(ckpt_name).result()
            ckpt_path = getattr(save_result, "path", None)
            print(f"[tinker] Checkpoint saved: {ckpt_path}")

            # Log checkpoint path to a manifest file for eval
            manifest_path = os.path.join(output_dir, "checkpoints.jsonl")
            with open(manifest_path, "a") as f:
                f.write(json.dumps({"step": step, "name": ckpt_name, "path": ckpt_path}) + "\n")

            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=ckpt_name
            )
            print(f"[tinker] Sampling client refreshed.")

    # Final summary
    elapsed = time.time() - t_start
    mean_rewards = [m["mean_reward"] for m in all_metrics]
    print(f"\n[tinker] === Training Complete ===")
    print(f"[tinker] Total time   : {elapsed / 60:.1f} min")
    print(f"[tinker] Avg reward   : {np.mean(mean_rewards):.3f}")
    print(f"[tinker] Final reward : {mean_rewards[-1]:.3f}")
    print(f"[tinker] Avg step time: {elapsed / args.max_steps:.1f}s")

    # Save final checkpoint
    print(f"[tinker] Saving final model ...")
    save_result = training_client.save_state("final").result()
    ckpt_path = getattr(save_result, "path", None)
    print(f"[tinker] Final checkpoint saved: {ckpt_path}")

    manifest_path = os.path.join(output_dir, "checkpoints.jsonl")
    with open(manifest_path, "a") as f:
        f.write(json.dumps({"step": args.max_steps, "name": "final", "path": ckpt_path}) + "\n")

    if use_wandb:
        wandb.finish()

    print(f"[tinker] Done.")


if __name__ == "__main__":
    main()
