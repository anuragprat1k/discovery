"""
Multi-turn GRPO training for code repair.

Each episode: model sees buggy code + test results, outputs <repair>,
gets updated test results, repeats for up to max_turns. Rewards computed
per-turn (dense) or at episode end only (sparse).

Uses HF model for generation with per-turn token/logprob collection,
then computes GRPO loss manually (no TRL GRPOTrainer — we need full
control over multi-turn rollouts).

Usage:
    python -m code_repair.train --reward {sparse,dense_passes,dense_full} [options]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from dotenv import load_dotenv

load_dotenv()

from code_repair.env.sandbox import run_tests, TestResult
from code_repair.env.code_repair_env import (
    SYSTEM_PROMPT,
    extract_repair,
    format_initial_prompt,
    format_feedback,
    format_test_results,
)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--reward", required=True, choices=["sparse", "dense_passes", "dense_full"])
    p.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--max_turns", type=int, default=4)
    p.add_argument("--problems_path", default="code_repair/data/problems_combined/train.json")
    p.add_argument("--eval_problems_path", default="code_repair/data/problems_combined/eval.json")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4, help="Problems per step.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.04, help="KL penalty coefficient.")
    p.add_argument("--clip_low", type=float, default=0.8)
    p.add_argument("--clip_high", type=float, default=1.2)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--eval_samples", type=int, default=8)
    p.add_argument("--eval_temperature", type=float, default=0.6)
    p.add_argument("--sandbox_timeout", type=int, default=5)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="discovery-code-repair")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model / tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_lora_model(model_name: str, lora_rank: int, lora_alpha: int):
    print(f"[train] Loading base model in bf16: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

TERMINAL_WIN = 3.0
SPEED_BONUS = 0.1
TERMINAL_LOSS = -1.0
FORMAT_PENALTY = -1.0
PASS_REWARD_PER_TEST = 0.4
NO_CRASH_BONUS = 0.05
TYPE_MATCH_BONUS = 0.05
SHAPE_MATCH_BONUS = 0.05


def compute_turn_reward_sparse(info: dict, is_terminal: bool) -> float:
    if info.get("format_violation"):
        return FORMAT_PENALTY
    if is_terminal:
        if info["all_passed"]:
            return TERMINAL_WIN + SPEED_BONUS * (info["max_turns"] - info["turn"])
        return TERMINAL_LOSS
    return 0.0


def compute_turn_reward_dense_passes(info: dict, is_terminal: bool) -> float:
    r = 0.0
    if info.get("format_violation"):
        return FORMAT_PENALTY
    # HWM delta (potential-based)
    n_tests = max(info["num_tests"], 1)
    hwm_delta = info["hw_passing"] - info["old_hw"]
    if hwm_delta > 0:
        r += PASS_REWARD_PER_TEST * hwm_delta / n_tests
    # Terminal
    if is_terminal:
        if info["all_passed"]:
            r += TERMINAL_WIN + SPEED_BONUS * (info["max_turns"] - info["turn"])
        else:
            r += TERMINAL_LOSS
    return r


def compute_turn_reward_dense_full(info: dict, is_terminal: bool) -> float:
    r = 0.0
    if info.get("format_violation"):
        return FORMAT_PENALTY
    n_tests = max(info["num_tests"], 1)
    # HWM delta
    hwm_delta = info["hw_passing"] - info["old_hw"]
    if hwm_delta > 0:
        r += PASS_REWARD_PER_TEST * hwm_delta / n_tests
    # Partial-correctness (non-potential)
    r += NO_CRASH_BONUS * info.get("no_crash_failing", 0) / n_tests
    r += TYPE_MATCH_BONUS * info.get("type_match_failing", 0) / n_tests
    r += SHAPE_MATCH_BONUS * info.get("shape_match_failing", 0) / n_tests
    # Terminal
    if is_terminal:
        if info["all_passed"]:
            r += TERMINAL_WIN + SPEED_BONUS * (info["max_turns"] - info["turn"])
        else:
            r += TERMINAL_LOSS
    return r


REWARD_FNS = {
    "sparse": compute_turn_reward_sparse,
    "dense_passes": compute_turn_reward_dense_passes,
    "dense_full": compute_turn_reward_dense_full,
}


# ---------------------------------------------------------------------------
# Multi-turn episode rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_episode(
    problem: dict,
    model,
    tokenizer,
    reward_fn,
    max_turns: int = 4,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    sandbox_timeout: int = 5,
) -> dict:
    """Run one multi-turn code repair episode, collecting per-turn tokens and logprobs."""
    device = next(model.parameters()).device

    # Initial state
    initial_results = run_tests(
        problem["buggy_code"], problem["test_code"], problem["entry_point"],
        timeout=sandbox_timeout, detailed=True,
    )
    user_msg = format_initial_prompt(
        problem["buggy_code"], initial_results, max_turns,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    prompt_tokens_per_turn = []
    completion_tokens_per_turn = []
    logprobs_per_turn = []
    per_turn_rewards = []
    completion_texts = []

    hw_passing = 0
    all_passed = False
    test_results_history = [initial_results]

    for turn in range(1, max_turns + 1):
        # Encode conversation
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False,
                                       return_tensors="pt").to(device)
        n_prompt = prompt_ids.shape[1]

        # Generate
        output = model.generate(
            prompt_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=True,
            return_dict_in_generate=True, output_logits=True,
        )
        gen_ids = output.sequences[0, n_prompt:]  # completion tokens only

        # Compute logprobs from logits
        # output.logits is a tuple of (vocab_size,) tensors, one per generated token
        n_logits = len(output.logits)
        n_gen = len(gen_ids)
        # Truncate to the shorter of the two (can differ by 1 near EOS)
        n_use = min(n_logits, n_gen)
        logits_stack = torch.stack(output.logits[:n_use], dim=0)  # (n_use, batch, vocab)
        if logits_stack.dim() == 3:
            logits_stack = logits_stack[:, 0, :]  # squeeze batch dim -> (n_use, vocab)
        log_probs_all = F.log_softmax(logits_stack, dim=-1)
        gen_ids_trunc = gen_ids[:n_use]
        token_logprobs = log_probs_all[range(n_use), gen_ids_trunc].cpu().tolist()
        # Pad if gen_ids was longer (unlikely but safe)
        if n_gen > n_use:
            token_logprobs.extend([-100.0] * (n_gen - n_use))
        gen_ids = gen_ids[:n_gen]  # keep original length

        completion_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        prompt_tokens_per_turn.append(prompt_ids[0].cpu().tolist())
        completion_tokens_per_turn.append(gen_ids.cpu().tolist())
        logprobs_per_turn.append(token_logprobs)
        completion_texts.append(completion_text)

        # Parse repair and run tests
        repair = extract_repair(completion_text)

        if repair is None:
            prev_results = test_results_history[-1]
            info = {
                "turn": turn, "max_turns": max_turns,
                "format_violation": True,
                "num_tests": len(prev_results),
                "hw_passing": hw_passing, "old_hw": hw_passing,
                "all_passed": False,
                "no_crash_failing": 0, "type_match_failing": 0, "shape_match_failing": 0,
            }
            is_terminal = turn >= max_turns
            per_turn_rewards.append(reward_fn(info, is_terminal))

            remaining = max_turns - turn
            feedback = f"No valid <repair> tag found. ({remaining} turn(s) remaining)"
            messages.append({"role": "assistant", "content": completion_text})
            messages.append({"role": "user", "content": feedback})
            if is_terminal:
                break
            continue

        # Run tests
        test_results = run_tests(
            repair, problem["test_code"], problem["entry_point"],
            timeout=sandbox_timeout, detailed=True,
        )
        test_results_history.append(test_results)

        prev_results = test_results_history[-2]
        prev_passing = {r.name for r in prev_results if r.passed}
        curr_passing = {r.name for r in test_results if r.passed}
        num_tests = max(len(test_results), 1)

        old_hw = hw_passing
        hw_passing = max(hw_passing, len(curr_passing))

        all_passed = (
            len(curr_passing) == len(test_results) and len(test_results) > 0
            and not (len(test_results) == 1 and test_results[0].name in
                     ("timeout", "syntax_error", "runtime_error", "import_error"))
        )

        no_crash = sum(1 for r in test_results if r.no_crash and not r.passed)
        type_match = sum(1 for r in test_results if not r.passed and r.no_crash and r.return_type)
        shape_match = sum(1 for r in test_results if not r.passed and r.return_shape is not None)

        is_terminal = all_passed or turn >= max_turns
        info = {
            "turn": turn, "max_turns": max_turns, "num_tests": num_tests,
            "hw_passing": hw_passing, "old_hw": old_hw,
            "all_passed": all_passed,
            "no_crash_failing": no_crash, "type_match_failing": type_match,
            "shape_match_failing": shape_match,
        }
        per_turn_rewards.append(reward_fn(info, is_terminal))

        # Build feedback
        if all_passed:
            feedback = f"All {num_tests} tests passing! Fixed in {turn} turn(s)."
        else:
            feedback = format_feedback(
                test_results, repair, turn=turn, max_turns=max_turns,
                prev_passing=prev_passing,
            )
        messages.append({"role": "assistant", "content": completion_text})
        messages.append({"role": "user", "content": feedback})

        if is_terminal:
            break

    return {
        "prompt_tokens_per_turn": prompt_tokens_per_turn,
        "completion_tokens_per_turn": completion_tokens_per_turn,
        "logprobs_per_turn": logprobs_per_turn,
        "per_turn_rewards": per_turn_rewards,
        "completion_texts": completion_texts,
        "all_passed": all_passed,
        "total_turns": len(per_turn_rewards),
        "hw_passing": hw_passing,
        "task_id": problem.get("task_id", ""),
    }


# ---------------------------------------------------------------------------
# GRPO loss (CISPO variant)
# ---------------------------------------------------------------------------

def compute_grpo_loss(
    model, episodes: list[dict], advantages: np.ndarray,
    beta: float, clip_low: float, clip_high: float,
) -> tuple[torch.Tensor, dict]:
    """Compute GRPO/CISPO loss over a batch of multi-turn episodes.

    Each turn is a separate forward pass: prompt + completion tokens.
    Advantage is per-episode, applied uniformly to all turns' completion tokens.
    """
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)
    n_tokens = 0
    total_kl = 0.0

    for ep_idx, ep in enumerate(episodes):
        adv = advantages[ep_idx]
        if abs(adv) < 1e-8:
            continue

        for turn_idx in range(ep["total_turns"]):
            prompt_tok = ep["prompt_tokens_per_turn"][turn_idx]
            comp_tok = ep["completion_tokens_per_turn"][turn_idx]
            old_logprobs = ep["logprobs_per_turn"][turn_idx]

            if not comp_tok:
                continue

            full_ids = torch.tensor(prompt_tok + comp_tok, dtype=torch.long, device=device).unsqueeze(0)
            n_prompt = len(prompt_tok)
            n_comp = len(comp_tok)

            # Forward pass
            outputs = model(full_ids)
            logits = outputs.logits[0, n_prompt - 1: n_prompt + n_comp - 1, :]  # shifted
            new_log_probs = F.log_softmax(logits, dim=-1)

            comp_ids = torch.tensor(comp_tok, dtype=torch.long, device=device)
            # Safety: clamp comp_ids to vocab range
            vocab_size = new_log_probs.shape[-1]
            comp_ids = comp_ids.clamp(0, vocab_size - 1)
            new_lp = new_log_probs[range(n_comp), comp_ids]
            old_lp = torch.tensor(old_logprobs[:n_comp], dtype=torch.float32, device=device)
            # Replace padding values with current logprobs (no gradient signal)
            pad_mask = old_lp < -99.0
            if pad_mask.any():
                old_lp[pad_mask] = new_lp[pad_mask].detach()

            # Importance ratio
            ratio = torch.exp(new_lp - old_lp)
            clipped_ratio = torch.clamp(ratio, clip_low, clip_high)

            # CISPO surrogate
            adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
            if adv > 0:
                surrogate = -torch.min(ratio * adv_t, clipped_ratio * adv_t)
            else:
                surrogate = -torch.max(ratio * adv_t, clipped_ratio * adv_t)

            # KL penalty
            kl = (old_lp - new_lp).mean()
            total_kl += kl.item() * n_comp

            turn_loss = surrogate.mean() + beta * kl
            total_loss = total_loss + turn_loss * n_comp
            n_tokens += n_comp

    if n_tokens > 0:
        total_loss = total_loss / n_tokens

    metrics = {"kl": total_kl / max(n_tokens, 1), "n_tokens": n_tokens}
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, problems, n_samples, max_new_tokens, temperature,
             max_turns, sandbox_timeout, step):
    """Lightweight eval: run episodes, report metrics."""
    from code_repair.eval import evaluate
    model.eval()
    results = evaluate(
        model=model, tokenizer=tokenizer, problems=problems,
        n_samples=n_samples, max_new_tokens=max_new_tokens,
        temperature=temperature, sandbox_timeout=sandbox_timeout, verbose=True,
    )
    model.train()
    print(f"[eval] step={step}: pass@1={results['pass_at_1']:.3f} "
          f"pass@k={results['pass_at_k']:.3f} format={results['format_rate']:.3f} "
          f"unique={results['unique_solved']}/{results['n_problems']}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir or f"checkpoints/code_repair_{args.reward}_s{args.seed}"
    os.makedirs(output_dir, exist_ok=True)
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb

    print(f"[train] Reward      : {args.reward}")
    print(f"[train] Model       : {args.model_name}")
    print(f"[train] Max turns   : {args.max_turns}")
    print(f"[train] Batch       : {args.batch_size} problems × {args.group_size} episodes")
    print(f"[train] Max steps   : {args.max_steps}")
    print(f"[train] Output dir  : {output_dir}")

    tokenizer = load_tokenizer(args.model_name)
    model = load_lora_model(args.model_name, args.lora_rank, args.lora_alpha)
    reward_fn = REWARD_FNS[args.reward]

    with open(args.problems_path) as f:
        problems = json.load(f)
    print(f"[train] Loaded {len(problems)} training problems")

    eval_problems = []
    if os.path.exists(args.eval_problems_path):
        with open(args.eval_problems_path) as f:
            eval_problems = json.load(f)
        print(f"[train] Loaded {len(eval_problems)} eval problems")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"code-repair-{args.reward}-s{args.seed}",
        )

    print("[train] Starting training...", flush=True)

    for step in range(1, args.max_steps + 1):
        t0 = time.time()

        # Sample batch of problems
        batch = random.choices(problems, k=args.batch_size)

        # Run episodes
        model.eval()  # no dropout during rollout
        all_episodes = []
        for problem in batch:
            for _ in range(args.group_size):
                ep = run_episode(
                    problem, model, tokenizer, reward_fn,
                    max_turns=args.max_turns,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    sandbox_timeout=args.sandbox_timeout,
                )
                all_episodes.append(ep)
        t_gen = time.time() - t0

        # Compute episode rewards and GRPO advantages
        episode_rewards = np.array([sum(ep["per_turn_rewards"]) for ep in all_episodes])
        advantages = np.zeros_like(episode_rewards)
        groups_skipped = 0
        n_groups = args.batch_size
        for g in range(n_groups):
            start = g * args.group_size
            end = start + args.group_size
            group = episode_rewards[start:end]
            std = group.std()
            if std < 1e-8:
                groups_skipped += 1
            else:
                advantages[start:end] = (group - group.mean()) / (std + 1e-8)

        # Training step
        model.train()
        optimizer.zero_grad()
        loss, loss_metrics = compute_grpo_loss(
            model, all_episodes, advantages,
            beta=args.beta, clip_low=args.clip_low, clip_high=args.clip_high,
        )
        if loss.requires_grad:
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
        t_total = time.time() - t0

        # Metrics
        solve_rate = np.mean([ep["all_passed"] for ep in all_episodes])
        avg_turns = np.mean([ep["total_turns"] for ep in all_episodes])
        avg_reward = episode_rewards.mean()
        avg_hw = np.mean([ep["hw_passing"] for ep in all_episodes])
        format_ok = np.mean([
            1.0 if not any("format_violation" in str(r) for r in ep.get("completion_texts", []))
            else 0.0
            for ep in all_episodes
        ])

        log_msg = (
            f"[step {step}/{args.max_steps}] "
            f"solve={solve_rate:.3f} turns={avg_turns:.1f} "
            f"reward={avg_reward:.2f}±{episode_rewards.std():.2f} "
            f"hwm={avg_hw:.1f} loss={loss.item():.4f} "
            f"kl={loss_metrics['kl']:.4f} "
            f"gen={t_gen:.0f}s total={t_total:.0f}s"
        )
        print(log_msg, flush=True)

        if use_wandb:
            wandb.log({
                "train/solve_rate": solve_rate,
                "train/avg_turns": avg_turns,
                "train/avg_reward": avg_reward,
                "train/reward_std": episode_rewards.std(),
                "train/avg_hwm": avg_hw,
                "train/loss": loss.item(),
                "train/kl": loss_metrics["kl"],
                "train/n_tokens": loss_metrics["n_tokens"],
                "train/groups_skipped": groups_skipped,
                "train/gen_time": t_gen,
                "train/step_time": t_total,
            }, step=step)

        # Eval
        if eval_problems and step % args.eval_steps == 0:
            eval_results = run_eval(
                model, tokenizer, eval_problems,
                n_samples=args.eval_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.eval_temperature,
                max_turns=args.max_turns,
                sandbox_timeout=args.sandbox_timeout,
                step=step,
            )
            if use_wandb:
                wandb.log({
                    "eval/pass_at_1": eval_results["pass_at_1"],
                    "eval/pass_at_k": eval_results["pass_at_k"],
                    "eval/format_rate": eval_results["format_rate"],
                    "eval/unique_solved": eval_results["unique_solved"],
                    "eval/unique_solved_frac": eval_results["unique_solved_frac"],
                }, step=step)

        # Save checkpoint
        if step % args.save_steps == 0:
            save_path = os.path.join(output_dir, f"step_{step}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  Saved checkpoint to {save_path}")

    # Final save
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[train] Training complete. Final model: {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
