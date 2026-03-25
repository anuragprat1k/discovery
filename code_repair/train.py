"""
GRPO training script for code repair using TRL's GRPOTrainer with LoRA.

Single-turn: model sees buggy code + test results, outputs one <repair>.
Reward function parses the repair, runs tests in sandbox, computes reward.

Three reward conditions:
  - sparse:       terminal only (all-pass → +3, else -1)
  - dense_passes: +0.4 per newly-passing test (potential-based, normalized)
  - dense_full:   dense_passes + partial-correctness (no-crash, type, shape)

Eval callback runs every --eval_steps steps on held-out problems and logs
pass@1, pass@k, format_rate, unique_solved to wandb.

Usage:
    python -m code_repair.train --reward {sparse,dense_passes,dense_full} [options]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, TaskType
from dotenv import load_dotenv

load_dotenv()

from code_repair.env.sandbox import run_tests, TestResult
from code_repair.env.code_repair_env import (
    SYSTEM_PROMPT,
    extract_repair,
    format_initial_prompt,
)
from code_repair.eval import evaluate

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO code repair training with TRL + LoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reward", required=True,
        choices=["sparse", "dense_passes", "dense_full"],
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--problems_path", type=str,
                        default="code_repair/data/problems/train.json")
    parser.add_argument("--eval_problems_path", type=str,
                        default="code_repair/data/problems/eval.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--num_generations", type=int, default=16,
                        help="Rollouts per prompt (G in GRPO).")
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--max_completion_length", type=int, default=2048,
                        help="Max tokens per generation (thinking + repair code).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Run eval every N training steps.")
    parser.add_argument("--eval_samples", type=int, default=8,
                        help="Samples per eval problem (for pass@k).")
    parser.add_argument("--eval_temperature", type=float, default=0.6)
    parser.add_argument("--sandbox_timeout", type=int, default=5)
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for generation (faster, requires compatible version).")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="discovery-code-repair")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model / tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_lora_model(model_name: str, lora_rank: int, lora_alpha: int):
    print(f"[train] Loading base model in bf16: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(problems_path: str, sandbox_timeout: int = 5) -> Dataset:
    with open(problems_path) as f:
        problems = json.load(f)

    rows = []
    for p in problems:
        initial_results = run_tests(
            p["buggy_code"], p["test_code"], p["entry_point"],
            timeout=sandbox_timeout, detailed=True,
        )
        user_msg = format_initial_prompt(
            p["buggy_code"], initial_results, max_turns=1,
        )
        row = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "task_id": p["task_id"],
            "buggy_code": p["buggy_code"],
            "canonical_solution": p["canonical_solution"],
            "test_code": p["test_code"],
            "entry_point": p["entry_point"],
            "num_tests": p["num_tests"],
        }
        rows.append(row)

    print(f"[train] Built dataset: {len(rows)} problems")
    return Dataset.from_list(rows)


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

_SANDBOX_TIMEOUT = 5


def _extract_and_test(completion_text: str, test_code: str, entry_point: str
                      ) -> tuple[str | None, list[TestResult]]:
    repair = extract_repair(completion_text)
    if repair is None:
        return None, []
    results = run_tests(repair, test_code, entry_point,
                        timeout=_SANDBOX_TIMEOUT, detailed=True)
    return repair, results


def _reward_sparse(completions, test_code, entry_point, num_tests, **kwargs):
    rewards = []
    for comp, tc, ep, nt in zip(completions, test_code, entry_point, num_tests):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        repair, results = _extract_and_test(text, tc, ep)
        if repair is None:
            rewards.append(FORMAT_PENALTY + TERMINAL_LOSS)
            continue
        passed = sum(1 for r in results if r.passed)
        all_passed = (passed == len(results) and len(results) > 0
                      and not (len(results) == 1 and results[0].name in
                               ("timeout", "syntax_error", "runtime_error")))
        rewards.append(TERMINAL_WIN if all_passed else TERMINAL_LOSS)
    return rewards


def _reward_dense_passes(completions, test_code, entry_point, num_tests, **kwargs):
    rewards = []
    for comp, tc, ep, nt in zip(completions, test_code, entry_point, num_tests):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        repair, results = _extract_and_test(text, tc, ep)
        if repair is None:
            rewards.append(FORMAT_PENALTY + TERMINAL_LOSS)
            continue
        n_tests = max(len(results), int(nt), 1)
        passed = sum(1 for r in results if r.passed)
        all_passed = (passed == len(results) and len(results) > 0
                      and not (len(results) == 1 and results[0].name in
                               ("timeout", "syntax_error", "runtime_error")))
        r = PASS_REWARD_PER_TEST * passed / n_tests
        r += TERMINAL_WIN if all_passed else TERMINAL_LOSS
        rewards.append(r)
    return rewards


def _reward_dense_full(completions, test_code, entry_point, num_tests, **kwargs):
    rewards = []
    for comp, tc, ep, nt in zip(completions, test_code, entry_point, num_tests):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        repair, results = _extract_and_test(text, tc, ep)
        if repair is None:
            rewards.append(FORMAT_PENALTY + TERMINAL_LOSS)
            continue
        n_tests = max(len(results), int(nt), 1)
        passed = sum(1 for r in results if r.passed)
        all_passed = (passed == len(results) and len(results) > 0
                      and not (len(results) == 1 and results[0].name in
                               ("timeout", "syntax_error", "runtime_error")))
        r = PASS_REWARD_PER_TEST * passed / n_tests
        for res in results:
            if res.passed or not res.no_crash:
                continue
            r += NO_CRASH_BONUS / n_tests
            if res.return_type is not None:
                r += TYPE_MATCH_BONUS / n_tests
            if res.return_shape is not None:
                r += SHAPE_MATCH_BONUS / n_tests
        r += TERMINAL_WIN if all_passed else TERMINAL_LOSS
        rewards.append(r)
    return rewards


REWARD_FNS = {
    "sparse": _reward_sparse,
    "dense_passes": _reward_dense_passes,
    "dense_full": _reward_dense_full,
}


# ---------------------------------------------------------------------------
# Eval callback
# ---------------------------------------------------------------------------

class EvalCallback(TrainerCallback):
    """Runs eval on held-out problems every N steps, logs to wandb with trajectory samples."""

    def __init__(
        self,
        eval_problems: list[dict],
        tokenizer,
        eval_steps: int = 50,
        n_samples: int = 8,
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        sandbox_timeout: int = 5,
        n_trace_problems: int = 5,
    ):
        self.eval_problems = eval_problems
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.sandbox_timeout = sandbox_timeout
        self.n_trace_problems = n_trace_problems

    def _collect_traces(self, model, step: int) -> list[dict]:
        """Generate one sample per problem for a subset, capturing full rollouts."""
        device = next(model.parameters()).device
        traces = []
        subset = self.eval_problems[:self.n_trace_problems]

        for p in subset:
            initial_results = run_tests(
                p["buggy_code"], p["test_code"], p["entry_point"],
                timeout=self.sandbox_timeout, detailed=True,
            )
            user_msg = format_initial_prompt(
                p["buggy_code"], initial_results, max_turns=1,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output = model.generate(
                    input_ids, max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature, do_sample=True,
                )
            completion = self.tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True,
            )
            repair = extract_repair(completion)

            test_summary = "no <repair> tag"
            passed = 0
            total = 0
            solved = False
            if repair:
                results = run_tests(
                    repair, p["test_code"], p["entry_point"],
                    timeout=self.sandbox_timeout, detailed=True,
                )
                passed = sum(1 for r in results if r.passed)
                total = len(results)
                solved = passed == total and total > 0
                test_summary = f"{passed}/{total}" + (" SOLVED" if solved else "")

            traces.append({
                "step": step,
                "task_id": p["task_id"],
                "bug_type": p.get("bug_type", ""),
                "test_result": test_summary,
                "solved": solved,
                "passed": passed,
                "total_tests": total,
                "user_prompt": user_msg[:2000],
                "completion": completion[:3000],
                "repair": (repair or "")[:2000],
            })
        return traces

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return
        if model is None:
            return

        print(f"\n[eval] Running eval at step {state.global_step}...", flush=True)
        model.eval()
        results = evaluate(
            model=model,
            tokenizer=self.tokenizer,
            problems=self.eval_problems,
            n_samples=self.n_samples,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            sandbox_timeout=self.sandbox_timeout,
            verbose=True,
        )

        # Collect trajectory samples
        traces = self._collect_traces(model, state.global_step)
        model.train()

        print(f"[eval] step={state.global_step}: "
              f"pass@1={results['pass_at_1']:.3f} "
              f"pass@k={results['pass_at_k']:.3f} "
              f"format={results['format_rate']:.3f} "
              f"unique={results['unique_solved']}/{results['n_problems']}",
              flush=True)

        # Log to wandb
        if _WANDB_AVAILABLE and wandb.run is not None:
            log_dict = {
                "eval/pass_at_1": results["pass_at_1"],
                "eval/pass_at_k": results["pass_at_k"],
                "eval/format_rate": results["format_rate"],
                "eval/unique_solved": results["unique_solved"],
                "eval/unique_solved_frac": results["unique_solved_frac"],
                "eval/raw_solve_rate": results["raw_solve_rate"],
            }
            # Per bug-type pass@1
            for bt, info in results.get("by_bug_type", {}).items():
                log_dict[f"eval/pass1_{bt}"] = info["pass_at_1"]

            # Log trajectory table
            if traces:
                columns = ["step", "task_id", "bug_type", "test_result", "solved",
                           "passed", "total_tests", "user_prompt", "completion", "repair"]
                table = wandb.Table(columns=columns)
                for t in traces:
                    table.add_data(*[t[c] for c in columns])
                log_dict["eval/trajectories"] = table

            wandb.log(log_dict, step=state.global_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    global _SANDBOX_TIMEOUT
    _SANDBOX_TIMEOUT = args.sandbox_timeout

    output_dir = args.output_dir or f"checkpoints/code_repair_{args.reward}_s{args.seed}"
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    report_to = "wandb" if use_wandb else "none"

    print(f"[train] Reward      : {args.reward}")
    print(f"[train] Model       : {args.model_name}")
    print(f"[train] LoRA        : rank={args.lora_rank} alpha={args.lora_alpha}")
    print(f"[train] Max steps   : {args.max_steps}")
    print(f"[train] Generations : {args.num_generations}")
    print(f"[train] Eval every  : {args.eval_steps} steps ({args.eval_samples} samples)")
    print(f"[train] Output dir  : {output_dir}")
    print(f"[train] vLLM        : {args.use_vllm}")

    tokenizer = load_tokenizer(args.model_name)
    model = load_lora_model(args.model_name, args.lora_rank, args.lora_alpha)
    dataset = build_dataset(args.problems_path, sandbox_timeout=args.sandbox_timeout)
    reward_fn = REWARD_FNS[args.reward]

    # Load eval problems
    eval_problems = []
    if os.path.exists(args.eval_problems_path):
        with open(args.eval_problems_path) as f:
            eval_problems = json.load(f)
        print(f"[train] Loaded {len(eval_problems)} eval problems")
    else:
        print(f"[train] WARNING: eval problems not found at {args.eval_problems_path}")

    # Batch size: TRL requires generation_batch_size divisible by num_generations
    train_batch = args.num_generations * args.per_device_batch_size

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        run_name=f"code-repair-{args.reward}-s{args.seed}",
        seed=args.seed,

        max_steps=args.max_steps,

        per_device_train_batch_size=train_batch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,

        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,

        logging_steps=1,
        save_steps=args.save_steps,
        report_to=report_to,
        dataloader_num_workers=0,

        # vLLM settings
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=0.5 if args.use_vllm else 0.3,
    )

    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Build eval callback
    callbacks = []
    if eval_problems:
        eval_cb = EvalCallback(
            eval_problems=eval_problems,
            tokenizer=tokenizer,
            eval_steps=args.eval_steps,
            n_samples=args.eval_samples,
            max_new_tokens=args.max_completion_length,
            temperature=args.eval_temperature,
            sandbox_timeout=args.sandbox_timeout,
        )
        callbacks.append(eval_cb)

    print("[train] Instantiating GRPOTrainer …")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    print("[train] Starting training …")
    trainer.train()

    # Final eval
    if eval_problems:
        print("\n[train] Running final eval...")
        model.eval()
        final_results = evaluate(
            model=model,
            tokenizer=tokenizer,
            problems=eval_problems,
            n_samples=args.eval_samples,
            max_new_tokens=args.max_completion_length,
            temperature=args.eval_temperature,
            sandbox_timeout=args.sandbox_timeout,
        )
        results_path = os.path.join(output_dir, "final_eval.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"[train] Final eval saved to {results_path}")

    final_dir = os.path.join(output_dir, "final")
    print(f"[train] Saving final model to: {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("[train] Training complete.")


if __name__ == "__main__":
    main()
