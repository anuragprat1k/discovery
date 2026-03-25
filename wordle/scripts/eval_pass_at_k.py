"""Evaluate pass@k for Wordle checkpoints.

Plays N games per target word, computes pass@1 and pass@k using the
unbiased estimator from Chen et al. 2021.

Usage:
    # vLLM on GPU (fast):
    python -u -m wordle.scripts.eval_pass_at_k --use_vllm --model /workspace/checkpoints/sft_merged \
        --n_samples 16 --word_set grpo --word_file results/eval/eval_words.json --output results/eval/sft_grpo.json

    # With LoRA:
    python -u -m wordle.scripts.eval_pass_at_k --use_vllm --model /workspace/checkpoints/sft_merged \
        --lora wordle/autoresearch/results/dense_proxy_sft100/checkpoint-1000 \
        --n_samples 16 --word_set grpo --word_file results/eval/eval_words.json --output results/eval/dense_grpo.json

    # HF transformers on CPU/MPS (slow):
    python -u -m wordle.scripts.eval_pass_at_k --model checkpoints/sft_merged --n_samples 16 --n_words 50
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from wordle.environment.wordle_env import _extract_guess, SYSTEM_PROMPT_ENV_ELIMINATED
from wordle.environment.wordle_gym import WordleGymEnv, load_default_word_lists
from wordle.data.split_words import get_word_splits


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def play_game_hf(model, tokenizer, target, valid_guesses, answers,
                 max_turns=6, temperature=0.8, device="mps"):
    """Play one game using HF transformers."""
    import torch
    env = WordleGymEnv(word_list=answers, valid_guesses=valid_guesses,
                       max_turns=max_turns, target=target.upper())
    env.reset()

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT_ENV_ELIMINATED},
        {"role": "user", "content": "Guess a 5-letter word."},
    ]

    guesses = []
    for turn in range(max_turns):
        prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=50, temperature=temperature,
                do_sample=True, top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)
        guess = _extract_guess(generated)

        if guess is None:
            conversation.append({"role": "assistant", "content": generated})
            conversation.append({"role": "user", "content":
                "I couldn't find a 5-letter word in your response. "
                "Please guess a 5-letter word in <guess>WORD</guess> tags."})
            continue

        guesses.append(guess)
        obs, reward, done, info = env.step(generated)
        conversation.append({"role": "assistant", "content": generated})
        if done:
            break
        feedback_text = obs["messages"][0]["content"] if isinstance(obs, dict) else obs
        conversation.append({"role": "user", "content": feedback_text})

    return {"target": target, "solved": env.target_reached,
            "turns": len(guesses), "guesses": guesses}


def play_game_vllm(llm, tokenizer, target, valid_guesses, answers,
                   max_turns=6, temperature=0.8):
    """Play one game using vLLM."""
    from vllm import SamplingParams

    env = WordleGymEnv(word_list=answers, valid_guesses=valid_guesses,
                       max_turns=max_turns, target=target.upper())
    env.reset()

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT_ENV_ELIMINATED},
        {"role": "user", "content": "Guess a 5-letter word."},
    ]

    sampling_params = SamplingParams(
        temperature=temperature, top_p=0.95, max_tokens=50,
    )

    guesses = []
    for turn in range(max_turns):
        prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        generated = outputs[0].outputs[0].text

        guess = _extract_guess(generated)
        if guess is None:
            conversation.append({"role": "assistant", "content": generated})
            conversation.append({"role": "user", "content":
                "I couldn't find a 5-letter word in your response. "
                "Please guess a 5-letter word in <guess>WORD</guess> tags."})
            continue

        guesses.append(guess)
        obs, reward, done, info = env.step(generated)
        conversation.append({"role": "assistant", "content": generated})
        if done:
            break
        feedback_text = obs["messages"][0]["content"] if isinstance(obs, dict) else obs
        conversation.append({"role": "user", "content": feedback_text})

    return {"target": target, "solved": env.target_reached,
            "turns": len(guesses), "guesses": guesses}


def main():
    parser = argparse.ArgumentParser(description="Wordle pass@k evaluation")
    parser.add_argument("--model", type=str, required=True, help="Base model path")
    parser.add_argument("--lora", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for inference")
    parser.add_argument("--n_samples", type=int, default=16, help="Samples per word")
    parser.add_argument("--n_words", type=int, default=50, help="Number of target words")
    parser.add_argument("--word_set", choices=["grpo", "sft", "probe"], default="grpo")
    parser.add_argument("--word_file", type=str, default=None,
                        help="JSON file with fixed word lists")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load word lists
    splits = get_word_splits()
    _, valid_guesses = load_default_word_lists()
    valid_guesses_set = set(valid_guesses)
    all_answers = splits["sft"] + splits["grpo"] + splits["probe"]

    # Load target words
    if args.word_file:
        with open(args.word_file) as f:
            word_lists = json.load(f)
        key = {"grpo": "grpo_50", "sft": "sft_20", "probe": "probe"}[args.word_set]
        if key in word_lists:
            target_words = word_lists[key]
        else:
            target_words = splits[args.word_set]
            if args.n_words < len(target_words):
                target_words = random.sample(target_words, args.n_words)
    else:
        target_words = splits[args.word_set]
        if args.n_words < len(target_words):
            target_words = random.sample(target_words, args.n_words)

    print(f"Evaluating {len(target_words)} words × {args.n_samples} samples "
          f"= {len(target_words) * args.n_samples} games")

    # Load model
    if args.use_vllm:
        from vllm import LLM
        from transformers import AutoTokenizer

        if args.lora:
            # Merge LoRA first, then load into vLLM
            import torch
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            import tempfile

            print(f"Merging LoRA {args.lora} into {args.model} (on CPU)...")
            base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cpu")
            merged = PeftModel.from_pretrained(base, args.lora, device_map="cpu")
            merged = merged.merge_and_unload()

            # Use /workspace/tmp if available (container root may be small)
            tmp_base = "/workspace/tmp" if Path("/workspace").exists() else None
            if tmp_base:
                Path(tmp_base).mkdir(exist_ok=True)
            merge_dir = tempfile.mkdtemp(prefix="merged_", dir=tmp_base)
            merged.save_pretrained(merge_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            tokenizer.save_pretrained(merge_dir)
            del base, merged
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            print(f"Loading merged model into vLLM from {merge_dir}...")
            llm = LLM(model=merge_dir, gpu_memory_utilization=0.5, max_model_len=512)
            # Clean up merged dir after loading
            import shutil
            shutil.rmtree(merge_dir, ignore_errors=True)
        else:
            print(f"Loading {args.model} into vLLM...")
            llm = LLM(model=args.model, gpu_memory_utilization=0.5, max_model_len=512)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        play_fn = lambda t: play_game_vllm(llm, tokenizer, t, valid_guesses_set,
                                           all_answers, temperature=args.temperature)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {args.model} on {device}...")
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        if args.lora:
            print(f"Loading LoRA adapter from {args.lora}...")
            model = PeftModel.from_pretrained(model, args.lora)
            model = model.merge_and_unload()

        model = model.to(device)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        play_fn = lambda t: play_game_hf(model, tokenizer, t, valid_guesses_set,
                                         all_answers, temperature=args.temperature,
                                         device=device)

    # Run evaluation
    results = []
    total_games = 0
    total_wins = 0
    t0 = time.time()

    for wi, word in enumerate(target_words):
        wins = 0
        for s in range(args.n_samples):
            result = play_fn(word.lower())
            if result["solved"]:
                wins += 1
            total_games += 1
            total_wins += 1 if result["solved"] else 0

        pk1 = pass_at_k(args.n_samples, wins, 1)
        pk4 = pass_at_k(args.n_samples, wins, min(4, args.n_samples))
        pk16 = pass_at_k(args.n_samples, wins, min(16, args.n_samples))

        results.append({
            "word": word, "n": args.n_samples, "wins": wins,
            "pass@1": round(pk1, 4), "pass@4": round(pk4, 4),
            "pass@16": round(pk16, 4),
        })

        elapsed = time.time() - t0
        rate = total_games / elapsed
        eta = (len(target_words) * args.n_samples - total_games) / rate if rate > 0 else 0
        print(f"  [{wi+1}/{len(target_words)}] {word.upper()}: {wins}/{args.n_samples} wins, "
              f"pass@1={pk1:.3f}, pass@16={pk16:.3f} "
              f"({rate:.1f} games/s, ETA {eta/60:.1f}min)")

    # Aggregate
    avg_pass1 = sum(r["pass@1"] for r in results) / len(results)
    avg_pass4 = sum(r["pass@4"] for r in results) / len(results)
    avg_pass16 = sum(r["pass@16"] for r in results) / len(results)
    win_rate = total_wins / total_games

    print(f"\n{'='*50}")
    print(f"Model: {args.model}" + (f" + LoRA {args.lora}" if args.lora else ""))
    print(f"Word set: {args.word_set} ({len(target_words)} words × {args.n_samples} samples)")
    print(f"Win rate: {win_rate:.1%} ({total_wins}/{total_games})")
    print(f"pass@1:  {avg_pass1:.4f}")
    print(f"pass@4:  {avg_pass4:.4f}")
    print(f"pass@16: {avg_pass16:.4f}")
    print(f"Time: {(time.time()-t0)/60:.1f} min")

    output = {
        "model": args.model,
        "lora": args.lora,
        "word_set": args.word_set,
        "n_words": len(target_words),
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "win_rate": round(win_rate, 4),
        "pass@1": round(avg_pass1, 4),
        "pass@4": round(avg_pass4, 4),
        "pass@16": round(avg_pass16, 4),
        "total_games": total_games,
        "total_wins": total_wins,
        "per_word": results,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
