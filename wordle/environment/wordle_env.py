"""Wordle multi-turn environment for RL training.

Implements the MessageEnv interface: each turn the model proposes a 5-letter
word guess, the environment validates it, computes colored tile feedback, and
returns the result. Training is done via the Tinker API directly (see
wordle/recipes/train.py).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .feedback import TileColor, compute_feedback, feedback_to_emoji

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def _get_text_content(message: dict) -> str:
    """Extract text content from a message dict."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p.get("type") == "text")


def _extract_guess(text: str) -> str | None:
    """Extract a 5-letter guess from model output.

    Only accepts <guess>WORD</guess> tags — no fallback parsing.
    This prevents the model's reasoning text from being mistaken for a guess.
    """
    tag_match = re.search(r"<guess>\s*([a-zA-Z]{5})\s*</guess>", text, re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).lower()
    return None


def load_word_list(path: Path) -> list[str]:
    """Load a word list file (one word per line)."""
    return [w.strip().lower() for w in path.read_text().splitlines() if w.strip()]


@dataclass
class MessageStepResult:
    """Result of a message-level environment step."""

    reward: float
    episode_done: bool
    next_messages: list[dict[str, str]]
    metrics: dict[str, float] = field(default_factory=dict)
    next_stop_condition: Any = None


SYSTEM_PROMPT = """\
You are playing Wordle. Guess a 5-letter English word.
After each guess, you receive feedback:
\U0001f7e9 = correct letter, correct position
\U0001f7e8 = correct letter, wrong position
\u2b1c = letter not in the word

You have 6 attempts. Before each guess, track eliminated letters in tags, then guess:
<eliminated>A, B, C</eliminated>
<guess>WORD</guess>\
"""


class WordleMessageEnv:
    """Message-level environment for Wordle.

    Each turn:
    1. Model proposes a 5-letter word guess
    2. Environment validates and provides colored feedback
    3. Episode ends when word guessed correctly or 6 turns exhausted

    Does NOT depend on tinker-cookbook (testable standalone).
    """

    def __init__(self, target: str, valid_guesses: set[str], max_turns: int = 6):
        self.target = target.lower()
        self.valid_guesses = valid_guesses
        self.max_turns = max_turns
        self.turn = 0
        self.history: list[tuple[str, list[TileColor]]] = []

    async def initial_observation(self) -> list[dict]:
        """Return system + user messages for turn 1."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Guess a 5-letter word."},
        ]

    async def step(self, message: dict) -> MessageStepResult:
        """Extract guess from model output, compute feedback, return result."""
        self.turn += 1
        text = _get_text_content(message)
        guess = _extract_guess(text)

        metrics: dict[str, float] = {"turn": float(self.turn)}

        if guess is None:
            return self._make_invalid_result(
                "I couldn't find a 5-letter word in your response. "
                "Please guess a single 5-letter word.",
                metrics,
            )

        if guess not in self.valid_guesses:
            # Invalid word: consume turn, all-grey feedback, penalty
            feedback = [TileColor.GREY] * 5
            self.history.append((guess, feedback))
            metrics["invalid_word"] = 1.0
            return self._make_feedback_result(
                guess, feedback, target_reached=False, metrics=metrics,
                extra_note="(not a valid word) ",
            )

        # Valid guess: compute real feedback
        feedback = compute_feedback(guess, self.target)
        self.history.append((guess, feedback))
        target_reached = guess == self.target
        if target_reached:
            metrics["target_reached"] = 1.0
            metrics["turns_to_solve"] = float(self.turn)

        return self._make_feedback_result(
            guess, feedback, target_reached=target_reached, metrics=metrics,
        )

    def _make_feedback_result(
        self,
        guess: str,
        feedback: list[TileColor],
        target_reached: bool,
        metrics: dict[str, float],
        extra_note: str = "",
    ) -> MessageStepResult:
        """Build a MessageStepResult with feedback."""
        from ..rewards.dense_reward import compute_turn_reward, compute_episode_reward

        episode_done = target_reached or self.turn >= self.max_turns

        # Compute per-turn reward
        prev_guesses = [g for g, _ in self.history[:-1]]
        prev_feedbacks = [f for _, f in self.history[:-1]]
        turn_reward, turn_metrics = compute_turn_reward(
            guess=guess,
            feedback=feedback,
            prev_feedbacks=prev_feedbacks,
            prev_guesses=prev_guesses,
            turn=self.turn,
            max_turns=self.max_turns,
            target_reached=target_reached,
        )
        metrics.update(turn_metrics)

        # Add episode reward if done
        if episode_done:
            ep_reward, ep_metrics = compute_episode_reward(
                target_reached=target_reached,
                total_turns=self.turn,
                max_turns=self.max_turns,
            )
            turn_reward += ep_reward
            metrics.update(ep_metrics)
            if not target_reached:
                metrics["max_turns_exceeded"] = 1.0

        # Build feedback text
        emoji = feedback_to_emoji(feedback)
        remaining = self.max_turns - self.turn
        if target_reached:
            feedback_text = (
                f"Turn {self.turn}: {guess.upper()} \u2192 {emoji}  "
                f"Correct! You got it in {self.turn} guess(es)!"
            )
        else:
            feedback_text = (
                f"Turn {self.turn}: {extra_note}{guess.upper()} \u2192 {emoji}  "
                f"({remaining} turn(s) remaining)"
            )

        # Build full conversation history
        next_messages = await_free_initial_observation()
        for i, (h_guess, h_feedback) in enumerate(self.history):
            next_messages.append(
                {"role": "assistant", "content": h_guess.upper()}
            )
            h_emoji = feedback_to_emoji(h_feedback)
            h_remaining = self.max_turns - (i + 1)
            if i < len(self.history) - 1:
                # Previous turns: abbreviated
                next_messages.append(
                    {"role": "user", "content": f"Turn {i+1}: {h_guess.upper()} \u2192 {h_emoji}  ({h_remaining} turn(s) remaining)"}
                )
            else:
                # Current turn: full feedback
                next_messages.append({"role": "user", "content": feedback_text})

        return MessageStepResult(
            reward=turn_reward,
            episode_done=episode_done,
            next_messages=next_messages,
            metrics=metrics,
        )

    def _make_invalid_result(
        self, error_msg: str, metrics: dict[str, float]
    ) -> MessageStepResult:
        """Create a result for unparseable model output."""
        episode_done = self.turn >= self.max_turns
        metrics["invalid_parse"] = 1.0

        next_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Guess a 5-letter word."},
        ]
        # Replay previous turns
        for i, (h_guess, h_feedback) in enumerate(self.history):
            next_messages.append(
                {"role": "assistant", "content": h_guess.upper()}
            )
            h_emoji = feedback_to_emoji(h_feedback)
            h_remaining = self.max_turns - (i + 1)
            next_messages.append(
                {"role": "user", "content": f"Turn {i+1}: {h_guess.upper()} \u2192 {h_emoji}  ({h_remaining} turn(s) remaining)"}
            )
        # Add error message
        remaining = self.max_turns - self.turn
        next_messages.append(
            {"role": "user", "content": f"{error_msg}\n({remaining} turn(s) remaining)"}
        )

        return MessageStepResult(
            reward=-0.1,
            episode_done=episode_done,
            next_messages=next_messages,
            metrics=metrics,
        )


def _sync_initial_messages() -> list[dict]:
    """Return initial messages without async."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Guess a 5-letter word."},
    ]


# Alias used in _make_feedback_result to avoid async call
await_free_initial_observation = _sync_initial_messages

