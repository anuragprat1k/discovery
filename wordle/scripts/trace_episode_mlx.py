"""Run a single Wordle episode using local MLX generation and print the trace.

Usage:
    python -m wordle.scripts.trace_episode_mlx [--target WORD] [--model MODEL]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from wordle.environment.feedback import compute_feedback, feedback_to_emoji, TileColor
from wordle.environment.wordle_env import load_word_list, _extract_guess, SYSTEM_PROMPT


def generate(model, tokenizer, prompt_tokens, max_tokens=256, temperature=0.7):
    """Simple generation with KV cache. Returns (tokens, text)."""
    generated = []
    eos = getattr(tokenizer, "eos_token_id", None)
    cache = make_prompt_cache(model)

    tokens = mx.array([prompt_tokens])
    logits = model(tokens, cache=cache)
    next_logits = logits[:, -1, :]

    for _ in range(max_tokens):
        if temperature <= 0:
            tid = mx.argmax(next_logits, axis=-1).item()
        else:
            tid = mx.random.categorical(next_logits / temperature).item()
        generated.append(tid)
        if eos is not None and tid == eos:
            break
        tokens = mx.array([[tid]])
        logits = model(tokens, cache=cache)
        next_logits = logits[:, -1, :]

    return generated, tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="crane")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-4B-bf16")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write JSON trace file.")
    args = parser.parse_args()

    # Load valid guesses
    data_dir = Path(__file__).parent.parent / "data"
    answers = load_word_list(data_dir / "wordle_answers.txt")
    guesses = load_word_list(data_dir / "wordle_guesses.txt")
    valid_guesses = set(answers) | set(guesses)

    # Load model (no LoRA — pure base model inference)
    print(f"Loading model: {args.model} ...")
    model, tokenizer = load(args.model)

    target = args.target.lower()
    print(f"\n{'='*60}")
    print(f"TARGET: {target.upper()}")
    print(f"{'='*60}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Guess a 5-letter word."},
    ]

    trace = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Guess a 5-letter word."},
    ]
    result_info = {"target": target.upper(), "solved": False, "turns": 0}

    print(f"[system] {SYSTEM_PROMPT}\n")
    print(f"[user] Guess a 5-letter word.\n")

    for turn in range(1, args.max_turns + 1):
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        gen_tokens, completion_text = generate(
            model, tokenizer, prompt_tokens,
            max_tokens=args.max_tokens, temperature=args.temperature,
        )
        n_tokens = len(gen_tokens)

        trace.append({"role": "assistant", "content": completion_text, "tokens": n_tokens})

        print(f"[assistant] ({n_tokens} tokens)")
        print(completion_text)
        print()

        guess = _extract_guess(completion_text)

        if guess is None or guess not in valid_guesses:
            guess_display = (guess or "?????").upper()
            feedback = [TileColor.GREY] * 5
            emoji = feedback_to_emoji(feedback)
            note = " (INVALID)" if guess else " (NO PARSE)"
            print(f"  >>> Turn {turn}: {guess_display}{note} → {emoji}")
        else:
            feedback = compute_feedback(guess, target)
            emoji = feedback_to_emoji(feedback)
            print(f"  >>> Turn {turn}: {guess.upper()} → {emoji}")

            if guess == target:
                result_info["solved"] = True
                result_info["turns"] = turn
                print(f"\n  *** SOLVED in {turn} turn(s)! ***\n")
                break

        remaining = args.max_turns - turn
        if remaining == 0:
            result_info["turns"] = turn
            print(f"\n  *** FAILED — target was {target.upper()} ***\n")
            break

        feedback_text = (
            f"Turn {turn}: {(guess or '?????').upper()} → {emoji}  "
            f"({remaining} turn(s) remaining)"
        )
        messages.append({"role": "assistant", "content": completion_text})
        messages.append({"role": "user", "content": feedback_text})
        trace.append({"role": "user", "content": feedback_text})
        print(f"[user] {feedback_text}\n")

        mx.clear_cache()

    # Dump trace to file
    out_path = Path(args.output) if args.output else Path(f"results/trace_{target}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {"result": result_info, "messages": trace}
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    print(f"Trace written to {out_path}")


if __name__ == "__main__":
    main()
