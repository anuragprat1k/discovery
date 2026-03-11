"""
GRPO training script using TRL's GRPOTrainer with LoRA (PEFT).

Target hardware: single NVIDIA RTX 5090 (32 GB VRAM)
Assumed TRL version: >= 0.13

Memory strategy:
  - Base model loaded in bfloat16, frozen.
  - LoRA adapters (rank=64, alpha=128) are the only trainable parameters,
    ~0.5% of total params, requiring ~200 MB of optimizer state instead of ~8 GB.
  - TRL detects is_peft_model() and skips loading a separate reference model —
    it disables adapters in-place to compute ref logprobs from the frozen base.
  - Net saving vs full fine-tune: ~16 GB (no ref model copy, tiny optimizer state).

Usage:
    python train.py --reward {binary,dense} [options]
"""

import argparse
import os
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import wandb  # noqa: F401
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Reward function imports
# ---------------------------------------------------------------------------

try:
    from rewards.reward_fns import compute_score_binary, compute_score_partial_credit
except ImportError as exc:
    print(
        f"[train.py] ERROR: Could not import reward functions from 'rewards.reward_fns'.\n"
        f"  Make sure 'rewards/reward_fns.py' exists relative to the script directory.\n"
        f"  Original error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning with TRL + LoRA on a single RTX 5090.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reward",
        required=True,
        choices=["binary", "dense"],
        help="Reward function: 'binary' for sparse 0/1, 'dense' for partial-credit.",
    )
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data_path", type=str, default="data/train_128.parquet")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--lora_rank", type=int, default=64,
        help="LoRA rank (default 64; higher = more capacity).",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=128,
        help="LoRA alpha (default 128 = 2× rank, standard scaling).",
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable Weights & Biases logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str) -> transformers.PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_lora_model(model_name: str, lora_rank: int, lora_alpha: int):
    """Load base model in bf16 and wrap with LoRA adapters.

    With PEFT, TRL will:
      - Skip loading a separate reference model entirely.
      - Compute ref logprobs by disabling adapters (reverts to frozen base weights).
    Trainable params: ~0.5% of total → optimizer state ~200 MB vs ~8 GB full.
    """
    print(f"[train.py] Loading base model in bf16: {model_name}")
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
        # Target the attention projection layers — standard for Qwen/LLaMA-style models
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"prompt", "ground_truth", "solution"}


def load_training_dataset(data_path: str):
    if not os.path.isabs(data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, data_path)
        if os.path.exists(candidate):
            data_path = candidate
        elif os.path.exists(data_path):
            data_path = os.path.abspath(data_path)
        else:
            print(
                f"[train.py] ERROR: Data file not found.\n"
                f"  Tried: {candidate}\n"
                f"       : {os.path.abspath(data_path)}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"[train.py] Loading dataset from: {data_path}")
    dataset = load_dataset("parquet", data_files={"train": data_path}, split="train")

    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        print(
            f"[train.py] ERROR: Dataset missing columns: {missing}\n"
            f"  Found: {dataset.column_names}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[train.py] Dataset: {len(dataset)} samples, columns: {dataset.column_names}")
    return dataset


# ---------------------------------------------------------------------------
# Reward function wrapper
# ---------------------------------------------------------------------------

def make_reward_fn(reward_type: str):
    """TRL reward function wrapper.

    TRL (>= 0.13) calls: reward_fn(prompts=..., completions=..., **dataset_kwargs)
    For conversational prompts, each completion is a list of message dicts:
      [{"role": "assistant", "content": "<generated text>"}]
    We extract the text content before scoring.
    """
    score_fn = compute_score_binary if reward_type == "binary" else compute_score_partial_credit

    def reward_fn(completions: list, **kwargs) -> list[float]:
        ground_truths = kwargs.get("ground_truth", [None] * len(completions))
        solutions = kwargs.get("solution", [None] * len(completions))

        rewards = []
        for completion, gt, sol in zip(completions, ground_truths, solutions):
            # TRL passes conversational completions as [{"role": "assistant", "content": "..."}]
            if isinstance(completion, list):
                completion = " ".join(
                    m["content"] for m in completion if isinstance(m, dict) and "content" in m
                )
            try:
                score = score_fn([completion], ground_truth=[gt], solution=[sol])[0]
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[reward_fn] WARNING: score_fn raised an exception; "
                    f"assigning 0.0.  Error: {exc}",
                    file=sys.stderr,
                )
                score = 0.0
            rewards.append(float(score))
        return rewards

    reward_fn.__name__ = f"reward_{reward_type}"
    return reward_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or f"checkpoints/{args.reward}"
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    report_to = "wandb" if use_wandb else "none"

    print(f"[train.py] Reward      : {args.reward}")
    print(f"[train.py] Model       : {args.model_name}")
    print(f"[train.py] LoRA rank   : {args.lora_rank}  alpha: {args.lora_alpha}")
    print(f"[train.py] Max steps   : {args.max_steps}")
    print(f"[train.py] Output dir  : {output_dir}")
    print(f"[train.py] Report to   : {report_to}")

    tokenizer = load_tokenizer(args.model_name)
    model = load_lora_model(args.model_name, args.lora_rank, args.lora_alpha)
    dataset = load_training_dataset(args.data_path)
    reward_fn = make_reward_fn(args.reward)

    training_config = GRPOConfig(
        output_dir=output_dir,
        run_name=f"grpo_{args.reward}",
        seed=42,

        max_steps=args.max_steps,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,

        num_generations=8,
        max_completion_length=1024,
        temperature=1.0,
        beta=0.04,

        learning_rate=1e-4,       # higher LR appropriate for LoRA adapters
        bf16=True,
        gradient_checkpointing=True,

        logging_steps=1,
        save_steps=50,
        report_to=report_to,
        dataloader_num_workers=0,
    )

    print("[train.py] Instantiating GRPOTrainer …")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("[train.py] Starting training …")
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    print(f"[train.py] Saving final model to: {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("[train.py] Training complete.")


if __name__ == "__main__":
    main()
