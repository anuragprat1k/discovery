"""Run a single Wordle episode and print the full conversation trace.

Usage:
    python -m wordle.scripts.trace_episode [--target WORD] [--max_tokens 256]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    import tinker
    from tinker import ServiceClient
except ImportError as exc:
    print(f"ERROR: Could not import 'tinker': {exc}", file=sys.stderr)
    sys.exit(1)

from wordle.environment.feedback import compute_feedback, feedback_to_emoji, TileColor
from wordle.environment.wordle_env import load_word_list, _extract_guess, SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="crane")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
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

    # Connect to Tinker
    print(f"Connecting to Tinker (model={args.model})...")
    service = ServiceClient()
    sampling_client = service.create_sampling_client(base_model=args.model)
    tokenizer = sampling_client.get_tokenizer()

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    target = args.target.lower()
    print(f"\n{'='*60}")
    print(f"TARGET: {target.upper()}")
    print(f"{'='*60}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Guess a 5-letter word. Reply with <guess>WORD</guess>."},
    ]

    # Trace collects all messages for JSON output
    trace = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Guess a 5-letter word. Reply with <guess>WORD</guess>."},
    ]
    result_info = {"target": target.upper(), "solved": False, "turns": 0}

    print(f"[system] {SYSTEM_PROMPT}\n")
    print(f"[user] Guess a 5-letter word. Reply with <guess>WORD</guess>.\n")

    for turn in range(1, args.max_turns + 1):
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_input = tinker.ModelInput.from_ints(prompt_tokens)

        result = sampling_client.sample(
            prompt=prompt_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        seq = result.sequences[0]
        completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        n_tokens = len(seq.tokens)

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

    # Dump trace to file
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {"result": result_info, "messages": trace}
        out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
        print(f"Trace written to {out_path}")


if __name__ == "__main__":
    main()
