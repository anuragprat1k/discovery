"""
pass_at_k.py — Evaluate a single checkpoint on an eval parquet using the
unbiased pass@k estimator from the Codex paper:

    pass@k = 1 - C(n-c, k) / C(n, k)

where n = total samples per problem, c = correct samples, k = target k.

Usage (local checkpoint):
    python eval/pass_at_k.py \
        --checkpoint_dir checkpoints/binary/step_0050 \
        --eval_parquet data/eval_200.parquet \
        --output_dir results/binary \
        --step 50 \
        --n_samples 64 \
        --temperature 1.0 \
        --max_new_tokens 1024

Usage (Tinker server-side checkpoint):
    python eval/pass_at_k.py \
        --use_tinker \
        --tinker_checkpoint step_0050 \
        --eval_parquet data/eval_200.parquet \
        --output_dir results/tinker \
        --n_samples 64
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Import reward helpers from sibling package
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from rewards.reward_fns import extract_boxed_answer, answers_match  # noqa: E402

# ---------------------------------------------------------------------------
# Try importing tqdm; fall back to a simple progress printer
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm

    def progress(iterable, total=None, desc=""):
        return _tqdm(iterable, total=total, desc=desc)

except ImportError:
    _tqdm = None

    class _SimplePrinter:
        """Minimal iterator wrapper that prints progress every 10 items."""

        def __init__(self, iterable, total=None, desc=""):
            self._it = iter(iterable)
            self._total = total
            self._desc = desc
            self._n = 0

        def __iter__(self):
            return self

        def __next__(self):
            val = next(self._it)
            self._n += 1
            if self._n % 10 == 0:
                suffix = f"/{self._total}" if self._total is not None else ""
                print(f"  {self._desc}: {self._n}{suffix}", flush=True)
            return val

    def progress(iterable, total=None, desc=""):
        return _SimplePrinter(iterable, total=total, desc=desc)


# ---------------------------------------------------------------------------
# Pass@k computation
# ---------------------------------------------------------------------------

def _pass_at_k(n: int, c: int, k: int) -> float | None:
    """
    Unbiased pass@k estimator.  Returns None when n < k (undefined).
    Returns 0.0 immediately when c == 0 to avoid 0-division edge cases.
    """
    if n < k:
        return None
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_at_k_stats(
    correct_counts: list[int],
    n: int,
    ks: list[int],
) -> dict[int, float | None]:
    """
    Given a list of per-problem correct counts and total samples n,
    return a dict mapping each k to the mean pass@k across all problems.
    Problems where pass@k is undefined (n < k) are excluded from the mean.
    """
    results: dict[int, float | None] = {}
    for k in ks:
        vals = [_pass_at_k(n, c, k) for c in correct_counts]
        defined = [v for v in vals if v is not None]
        if not defined:
            results[k] = None
        else:
            results[k] = sum(defined) / len(defined)
    return results


# ---------------------------------------------------------------------------
# Answer entropy
# ---------------------------------------------------------------------------

def _entropy(answers: list[str]) -> float:
    """Shannon entropy (bits) of an answer distribution."""
    if not answers:
        return 0.0
    counts = Counter(answers)
    total = len(answers)
    h = 0.0
    for cnt in counts.values():
        p = cnt / total
        if p > 0:
            h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _build_prompt(problem: str, tokenizer) -> str:
    """Apply chat template to a math problem."""
    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following math problem step by step. "
                "Put your final answer in a \\boxed{} at the end.\n\n"
                + problem
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    temperature: float,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    """
    Tokenize `prompts`, run model.generate, return decoded *new* tokens only.
    """
    import torch

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_ids = output_ids[:, input_len:]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)


def _generate_samples(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    device: str,
    batch_size: int,
) -> list[str]:
    """
    Generate `n_samples` completions for one prompt, handling OOM by halving
    the batch size and retrying.
    """
    completions: list[str] = []
    remaining = n_samples
    current_batch = batch_size

    while remaining > 0:
        this_batch = min(current_batch, remaining)
        try:
            batch_completions = _generate_batch(
                model,
                tokenizer,
                [prompt] * this_batch,
                temperature,
                max_new_tokens,
                device,
            )
            completions.extend(batch_completions)
            remaining -= this_batch
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and current_batch > 1:
                import torch
                torch.cuda.empty_cache()
                current_batch = max(1, current_batch // 2)
                print(
                    f"  [OOM] Reducing batch size to {current_batch} and retrying.",
                    flush=True,
                )
                if current_batch < 1:
                    raise RuntimeError("OOM at batch_size=1") from exc
            else:
                raise

    return completions


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def _evaluate_vllm(args, df) -> dict:
    """Fast evaluation path using vllm for batched generation."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    problems = df["problem"].tolist()
    ground_truths = df["ground_truth"].tolist()
    levels = df["level"].tolist()
    n_problems = len(problems)

    print(f"Loading vllm engine for: {args.checkpoint_dir}", flush=True)
    llm = LLM(
        model=args.checkpoint_dir,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=4096,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=0.95,
    )

    ks = [1, 4, 16, 64]
    correct_counts: list[int] = []
    unique_correct_per_problem: list[float] = []
    entropy_per_problem: list[float] = []
    level_correct: dict[str, list[int]] = {}

    pbar = progress(range(n_problems), total=n_problems, desc="Problems")
    for idx in pbar:
        problem = problems[idx]
        gt = ground_truths[idx]
        level = str(levels[idx])

        prompt = _build_prompt(problem, tokenizer)
        outputs = llm.generate([prompt], sampling_params)
        completions = [o.text for o in outputs[0].outputs]

        extracted: list = [extract_boxed_answer(c) for c in completions]
        is_correct: list[bool] = [
            pred is not None and answers_match(pred, gt) for pred in extracted
        ]
        c = sum(is_correct)
        correct_counts.append(c)

        correct_answers = [
            pred for pred, ok in zip(extracted, is_correct) if ok and pred is not None
        ]
        unique_correct_per_problem.append(float(len(set(correct_answers))))

        all_answers = [pred if pred is not None else "__NO_ANSWER__" for pred in extracted]
        entropy_per_problem.append(_entropy(all_answers))

        if level not in level_correct:
            level_correct[level] = []
        level_correct[level].append(c)

    n = args.n_samples
    pass_at_k_overall = compute_pass_at_k_stats(correct_counts, n, ks)

    pass_at_k_by_level: dict[str, dict[str, float | None]] = {}
    for k in ks:
        pass_at_k_by_level[str(k)] = {}
        for lvl, counts in sorted(level_correct.items()):
            val = compute_pass_at_k_stats(counts, n, [k])[k]
            pass_at_k_by_level[str(k)][lvl] = val

    mean_correct = sum(correct_counts) / n_problems
    mean_unique_correct = sum(unique_correct_per_problem) / n_problems
    mean_entropy = sum(entropy_per_problem) / n_problems

    return {
        "step": args.step,
        "run": args.run_name,
        "n_problems": n_problems,
        "n_samples": n,
        "eval_method": "vllm",
        "pass_at_k": {str(k): pass_at_k_overall[k] for k in ks},
        "pass_at_k_by_level": pass_at_k_by_level,
        "unique_correct_answers": mean_unique_correct,
        "answer_entropy": mean_entropy,
        "mean_correct_per_problem": mean_correct,
    }


def _evaluate_tinker(args, df) -> dict:
    """Evaluation path using Tinker SamplingClient for server-side checkpoints."""
    from dotenv import load_dotenv

    load_dotenv()

    import tinker
    from tinker import ServiceClient

    problems = df["problem"].tolist()
    ground_truths = df["ground_truth"].tolist()
    levels = df["level"].tolist()
    n_problems = len(problems)

    print(f"[tinker] Connecting to Tinker API ...", flush=True)
    service = ServiceClient()

    print(f"[tinker] Restoring checkpoint: {args.tinker_checkpoint}", flush=True)
    training_client = service.create_training_client_from_state(
        name=args.tinker_checkpoint
    )
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"eval_{args.tinker_checkpoint}"
    )
    tokenizer = training_client.get_tokenizer()
    print(f"[tinker] Sampling client ready.", flush=True)

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    ks = [1, 4, 16, 64]
    correct_counts: list[int] = []
    unique_correct_per_problem: list[float] = []
    entropy_per_problem: list[float] = []
    level_correct: dict[str, list[int]] = {}

    pbar = progress(range(n_problems), total=n_problems, desc="Problems")
    for idx in pbar:
        problem = problems[idx]
        gt = ground_truths[idx]
        level = str(levels[idx])

        prompt = _build_prompt(problem, tokenizer)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_input = tinker.ModelInput.from_ints(prompt_tokens)

        result = sampling_client.sample(
            prompt=prompt_input,
            num_samples=args.n_samples,
            sampling_params=sampling_params,
        ).result()

        completions = [
            tokenizer.decode(seq.tokens, skip_special_tokens=True)
            for seq in result.sequences
        ]

        extracted: list = [extract_boxed_answer(c) for c in completions]
        is_correct: list[bool] = [
            pred is not None and answers_match(pred, gt) for pred in extracted
        ]
        c = sum(is_correct)
        correct_counts.append(c)

        correct_answers = [
            pred for pred, ok in zip(extracted, is_correct) if ok and pred is not None
        ]
        unique_correct_per_problem.append(float(len(set(correct_answers))))

        all_answers = [pred if pred is not None else "__NO_ANSWER__" for pred in extracted]
        entropy_per_problem.append(_entropy(all_answers))

        if level not in level_correct:
            level_correct[level] = []
        level_correct[level].append(c)

    n = args.n_samples
    pass_at_k_overall = compute_pass_at_k_stats(correct_counts, n, ks)

    pass_at_k_by_level: dict[str, dict[str, float | None]] = {}
    for k in ks:
        pass_at_k_by_level[str(k)] = {}
        for lvl, counts in sorted(level_correct.items()):
            val = compute_pass_at_k_stats(counts, n, [k])[k]
            pass_at_k_by_level[str(k)][lvl] = val

    mean_correct = sum(correct_counts) / n_problems
    mean_unique_correct = sum(unique_correct_per_problem) / n_problems
    mean_entropy = sum(entropy_per_problem) / n_problems

    return {
        "step": args.step,
        "run": args.run_name,
        "n_problems": n_problems,
        "n_samples": n,
        "eval_method": "tinker",
        "tinker_checkpoint": args.tinker_checkpoint,
        "pass_at_k": {str(k): pass_at_k_overall[k] for k in ks},
        "pass_at_k_by_level": pass_at_k_by_level,
        "unique_correct_answers": mean_unique_correct,
        "answer_entropy": mean_entropy,
        "mean_correct_per_problem": mean_correct,
    }


def evaluate(args) -> dict:
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- Load eval data ---
    print(f"Loading eval parquet: {args.eval_parquet}", flush=True)
    df = pd.read_parquet(args.eval_parquet)

    # Validate columns
    if "problem" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError(
            f"Parquet must have 'problem' and 'ground_truth' columns. "
            f"Found: {list(df.columns)}"
        )
    if "level" not in df.columns:
        df["level"] = -1

    # --- Dispatch to tinker or vllm path if requested ---
    if getattr(args, "use_tinker", False):
        return _evaluate_tinker(args, df)
    if getattr(args, "use_vllm", False):
        return _evaluate_vllm(args, df)

    problems = df["problem"].tolist()
    ground_truths = df["ground_truth"].tolist()
    levels = df["level"].tolist()
    n_problems = len(problems)
    print(f"  {n_problems} problems loaded.", flush=True)

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()
    print(f"  Model loaded on {args.device}.", flush=True)

    # --- Generate samples and score ---
    ks = [1, 4, 16, 64]
    correct_counts: list[int] = []
    unique_correct_per_problem: list[float] = []
    entropy_per_problem: list[float] = []

    # Per-level tracking: level -> list of correct counts
    level_correct: dict[str, list[int]] = {}

    pbar = progress(range(n_problems), total=n_problems, desc="Problems")
    for idx in pbar:
        problem = problems[idx]
        gt = ground_truths[idx]
        level = str(levels[idx])

        prompt = _build_prompt(problem, tokenizer)

        completions = _generate_samples(
            model,
            tokenizer,
            prompt,
            n_samples=args.n_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            batch_size=args.batch_size,
        )

        # Score each completion
        extracted: list[str | None] = [extract_boxed_answer(c) for c in completions]
        is_correct: list[bool] = [
            pred is not None and answers_match(pred, gt)
            for pred in extracted
        ]
        c = sum(is_correct)
        correct_counts.append(c)

        # Unique correct answers
        correct_answers = [
            pred for pred, ok in zip(extracted, is_correct) if ok and pred is not None
        ]
        unique_correct_per_problem.append(float(len(set(correct_answers))))

        # Entropy over all answers (treat None as the string "NONE")
        all_answers = [pred if pred is not None else "__NO_ANSWER__" for pred in extracted]
        entropy_per_problem.append(_entropy(all_answers))

        # Per-level accumulation
        if level not in level_correct:
            level_correct[level] = []
        level_correct[level].append(c)

    # --- Compute pass@k metrics ---
    n = args.n_samples
    pass_at_k_overall = compute_pass_at_k_stats(correct_counts, n, ks)

    pass_at_k_by_level: dict[str, dict[str, float | None]] = {}
    for k in ks:
        pass_at_k_by_level[str(k)] = {}
        for lvl, counts in sorted(level_correct.items()):
            val = compute_pass_at_k_stats(counts, n, [k])[k]
            pass_at_k_by_level[str(k)][lvl] = val

    # Aggregate summary metrics
    mean_correct = sum(correct_counts) / n_problems
    mean_unique_correct = sum(unique_correct_per_problem) / n_problems
    mean_entropy = sum(entropy_per_problem) / n_problems

    result = {
        "step": args.step,
        "run": args.run_name,
        "n_problems": n_problems,
        "n_samples": n,
        "eval_method": "hf_transformers",
        "pass_at_k": {
            str(k): pass_at_k_overall[k] for k in ks
        },
        "pass_at_k_by_level": pass_at_k_by_level,
        "unique_correct_answers": mean_unique_correct,
        "answer_entropy": mean_entropy,
        "mean_correct_per_problem": mean_correct,
    }

    return result


# ---------------------------------------------------------------------------
# Argument parsing + entrypoint
# ---------------------------------------------------------------------------

def _infer_run_name(checkpoint_dir: str) -> str:
    """
    Infer run name from checkpoint path.
    e.g. "checkpoints/binary/step_0050" -> "binary"
    """
    parts = Path(checkpoint_dir).resolve().parts
    # Walk from the end; skip the step_XXXX dir, take the next one up
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("step_"):
            if i > 0:
                return parts[i - 1]
    # Fallback: parent directory name
    return Path(checkpoint_dir).resolve().parent.name


def _infer_step(checkpoint_dir: str) -> int:
    """
    Infer step number from directory name.
    e.g. "step_0050" -> 50
    """
    name = Path(checkpoint_dir).name
    if name.startswith("step_"):
        try:
            return int(name[len("step_"):])
        except ValueError:
            pass
    raise ValueError(
        f"Cannot infer step from directory name '{name}'. "
        "Please pass --step explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint with unbiased pass@k."
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Path to the HuggingFace checkpoint directory (not required with --use_tinker).",
    )
    parser.add_argument(
        "--eval_parquet",
        default="data/eval_200.parquet",
        help="Path to the eval parquet file (default: data/eval_200.parquet).",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory to write result JSON (default: results).",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help=(
            "Name of the training run, used for output subdirectory. "
            "Inferred from checkpoint_dir if not provided."
        ),
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Training step number. Inferred from checkpoint_dir name if not provided.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=64,
        help="Number of samples to generate per problem (default: 64).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens per generation (default: 1024).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of completions to generate in one forward pass (default: 8).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Device to run on. Defaults to "cuda" if available, else "cpu".',
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vllm for fast batched generation (requires vllm installed).",
    )
    parser.add_argument(
        "--use_tinker",
        action="store_true",
        help="Use Tinker API for evaluation (server-side checkpoints).",
    )
    parser.add_argument(
        "--tinker_checkpoint",
        type=str,
        default=None,
        help="Tinker checkpoint name to restore (e.g. 'step_0050'). Required with --use_tinker.",
    )
    args = parser.parse_args()

    # Validate tinker args
    if args.use_tinker:
        if args.tinker_checkpoint is None:
            parser.error("--tinker_checkpoint is required when using --use_tinker")
    elif args.checkpoint_dir is None:
        parser.error("--checkpoint_dir is required (unless using --use_tinker)")

    # Fill in inferred defaults
    if args.run_name is None:
        if args.use_tinker:
            args.run_name = "tinker"
        else:
            args.run_name = _infer_run_name(args.checkpoint_dir)

    if args.step is None:
        if args.use_tinker and args.tinker_checkpoint is not None:
            # Try to parse step from checkpoint name like "step_0050"
            ckpt = args.tinker_checkpoint
            if ckpt.startswith("step_"):
                try:
                    args.step = int(ckpt[len("step_"):])
                except ValueError:
                    args.step = 0
            else:
                args.step = 0
        else:
            args.step = _infer_step(args.checkpoint_dir)

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    return args


def main():
    args = parse_args()

    print(f"Run: {args.run_name}  |  Step: {args.step}", flush=True)
    print(f"Device: {args.device}", flush=True)

    result = evaluate(args)

    # --- Write output ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step_{args.step:04d}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults written to: {out_path}", flush=True)
    print(
        "pass@k: "
        + ", ".join(
            f"k={k}: {v:.4f}" if v is not None else f"k={k}: N/A"
            for k, v in result["pass_at_k"].items()
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
