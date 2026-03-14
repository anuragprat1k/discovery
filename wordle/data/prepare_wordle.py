"""Download official Wordle word lists."""
from __future__ import annotations

import urllib.request
from pathlib import Path

ANSWERS_URL = "https://gist.githubusercontent.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/wordle-answers-alphabetical.txt"
GUESSES_URL = "https://gist.githubusercontent.com/cfreshman/cdcdf777450c5b5301e439061d29694c/raw/wordle-allowed-guesses.txt"

DATA_DIR = Path(__file__).parent


def download_word_list(url: str, output_path: Path) -> list[str]:
    """Download a word list from a URL and save it."""
    print(f"Downloading {output_path.name}...")
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    words = [w.strip().lower() for w in text.splitlines() if w.strip()]
    output_path.write_text("\n".join(words) + "\n")
    print(f"  Saved {len(words)} words to {output_path}")
    return words


def main():
    answers_path = DATA_DIR / "wordle_answers.txt"
    guesses_path = DATA_DIR / "wordle_guesses.txt"

    answers = download_word_list(ANSWERS_URL, answers_path)
    guesses = download_word_list(GUESSES_URL, guesses_path)

    print(f"\nAnswers: {len(answers)}, Additional guesses: {len(guesses)}")
    print(f"Total valid guesses: {len(set(answers) | set(guesses))}")


if __name__ == "__main__":
    main()
