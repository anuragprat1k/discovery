"""Comprehensive test suite for the Wordle environment."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from wordle.environment.feedback import TileColor, compute_feedback, feedback_to_emoji
from wordle.environment.constraints import (
    RevealedConstraints,
    compute_constraint_violation_rate,
)
from wordle.environment.wordle_env import WordleMessageEnv, _extract_guess, load_word_list
from wordle.rewards import dense_reward, sparse_reward
from wordle.rewards.reward_utils import (
    count_new_greens,
    count_new_yellows,
    has_constraint_violation,
)

G = TileColor.GREEN
Y = TileColor.YELLOW
X = TileColor.GREY


# ---------------------------------------------------------------------------
# TestFeedback
# ---------------------------------------------------------------------------
class TestFeedback:
    def test_all_green(self):
        assert compute_feedback("crane", "crane") == [G, G, G, G, G]

    def test_all_grey(self):
        assert compute_feedback("plumb", "dirty") == [X, X, X, X, X]

    def test_basic_yellow(self):
        fb = compute_feedback("arose", "renal")
        # a: in target? yes (pos 3 in renAl) -> yellow (not at pos 0)
        # r: in target? yes (pos 0 in Renal) -> yellow (not at pos 1)
        # o: not in target -> grey
        # s: not in target -> grey
        # e: in target? yes (pos 1 in rEnal) -> yellow (not at pos 4)
        assert fb == [Y, Y, X, X, Y]

    def test_duplicate_letter_guess_has_extra(self):
        """Guess has 2 of a letter, target has 1 -> first gets color, second GREY."""
        fb = compute_feedback("speed", "abide")
        # s: not in target -> grey
        # p: not in target -> grey
        # e: target has 'e' at pos 4 (abidE). Not at pos 2 -> yellow
        # e: second 'e'. Target 'e' at pos 4 already consumed -> grey
        # d: target has 'd' at pos 3 (abiDe). Not at pos 4 -> yellow
        assert fb == [X, X, Y, X, Y]

    def test_duplicate_letter_both_present(self):
        """Both guess and target have 2 of a letter -> both colored."""
        fb = compute_feedback("geese", "eagle")
        # g: in target at pos 2 (eaGle). Not at pos 0 -> yellow
        # e: target has 'e' at pos 0 (Eagle). Not at pos 1 -> yellow? Let's trace:
        #   Pass 1 greens: pos 4 g[4]='e', t[4]='e' -> GREEN. consume t[4]
        #   Pass 2: pos 0 g='g', t_remaining=[e,a,g,l,None]. 'g' in remaining -> yellow, consume t[2]
        #   pos 1 g='e', t_remaining=[e,a,None,l,None]. 'e' in remaining -> yellow, consume t[0]
        #   pos 2 g='e', t_remaining=[None,a,None,l,None]. 'e' not in remaining -> grey
        #   pos 3 g='s'. not in remaining -> grey
        assert fb == [Y, Y, X, X, G]

    def test_green_takes_priority(self):
        """Green match should be marked before yellow."""
        fb = compute_feedback("refer", "revel")
        # Pass 1: pos 0 r==r GREEN, pos 1 e==e GREEN, pos 2 f!=v, pos 3 e!=e? no v!=e, pos 4 r!=l
        # Actually: target = r,e,v,e,l
        # Pass 1: pos 0 r==r GREEN, pos 1 e==e GREEN, pos 2 f!=v, pos 3 e==e GREEN, pos 4 r!=l
        # remaining: [None, None, v, None, l]
        # Pass 2: pos 2 f not in [v,l] -> grey. pos 4 r not in [v,l] -> grey
        assert fb == [G, G, X, G, X]

    def test_emoji_output(self):
        fb = [G, Y, X, G, X]
        emoji = feedback_to_emoji(fb)
        assert "\U0001f7e9" in emoji
        assert "\U0001f7e8" in emoji
        assert "\u2b1c" in emoji


# ---------------------------------------------------------------------------
# TestConstraintViolation
# ---------------------------------------------------------------------------
class TestConstraintViolation:
    def test_grey_letter_reuse(self):
        # After guessing "crane" against "dusty", c/r/a/n/e are all grey
        guesses = ["crane"]
        feedbacks = [[X, X, X, X, X]]
        c = RevealedConstraints.from_history(guesses, feedbacks)
        violations = c.check_violations("crest")
        assert any("grey letter" in v.lower() for v in violations)

    def test_green_not_placed(self):
        guesses = ["crane"]
        feedbacks = [[G, X, X, X, X]]  # c is green at 0
        c = RevealedConstraints.from_history(guesses, feedbacks)
        violations = c.check_violations("dusty")  # no c at pos 0
        assert any("green letter" in v.lower() for v in violations)

    def test_yellow_same_position(self):
        guesses = ["crane"]
        feedbacks = [[X, Y, X, X, X]]  # r is yellow at pos 1
        c = RevealedConstraints.from_history(guesses, feedbacks)
        violations = c.check_violations("dried")  # r at pos 1 again... wait, d at 0, r at 1
        # Actually "dried" has r at pos 1 -> violation
        # But also missing known letter r... no, r IS in "dried" at pos 1
        # The violation is: yellow letter 'r' placed at position 1 where excluded
        assert any("yellow letter" in v.lower() and "position 1" in v for v in violations)

    def test_known_letter_missing(self):
        guesses = ["crane"]
        feedbacks = [[X, Y, X, X, X]]  # r is in word
        c = RevealedConstraints.from_history(guesses, feedbacks)
        violations = c.check_violations("dusty")  # no r
        assert any("missing" in v.lower() for v in violations)

    def test_no_violations_first_turn(self):
        c = RevealedConstraints.from_history([], [])
        violations = c.check_violations("crane")
        assert violations == []

    def test_duplicate_letter_grey_but_green_elsewhere(self):
        """A letter grey at one position but green at another is NOT absent."""
        # guess "llama" target "light" -> l at 0 is GREEN, l at 1 is GREY
        guesses = ["level"]
        feedbacks = [[G, X, X, X, X]]  # l green at 0, e grey, v grey, e grey, l grey
        # 'l' appeared GREEN at pos 0 -> it's in the word -> not in grey set
        c = RevealedConstraints.from_history(guesses, feedbacks)
        assert "l" not in c.grey
        assert "l" in c.known_in_word

    def test_constraint_violation_rate(self):
        episodes = [
            [
                ("crane", [X, X, X, X, X]),  # turn 1: no constraints
                ("crane", [X, X, X, X, X]),  # turn 2: reuses grey letters -> violation
            ],
        ]
        rate = compute_constraint_violation_rate(episodes)
        assert rate == 1.0  # 1 of 1 non-first turns has violations

    def test_constraint_violation_rate_no_violations(self):
        episodes = [
            [
                ("crane", [G, G, G, G, G]),  # solved turn 1
            ],
        ]
        rate = compute_constraint_violation_rate(episodes)
        assert rate == 0.0  # no non-first turns


# ---------------------------------------------------------------------------
# TestWordleMessageEnv
# ---------------------------------------------------------------------------
class TestWordleMessageEnv:
    VALID = {"crane", "slate", "arose", "dusty", "light", "crane", "plumb",
             "speed", "geese", "eagle", "revel", "refer", "level", "dried",
             "crest", "hello", "world", "dirty", "renal", "abide", "llama",
             "clues", "outer", "stare", "audio", "adieu"}

    def test_initial_observation(self):
        env = WordleMessageEnv("crane", self.VALID)
        msgs = asyncio.run(env.initial_observation())
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "<guess>" in msgs[1]["content"]

    def test_correct_guess_turn_1(self):
        env = WordleMessageEnv("crane", self.VALID)
        asyncio.run(env.initial_observation())
        result = asyncio.run(env.step({"role": "assistant", "content": "crane"}))
        assert result.episode_done is True
        assert result.metrics.get("target_reached") == 1.0
        assert result.metrics.get("turns_to_solve") == 1.0

    def test_multi_turn_flow(self):
        env = WordleMessageEnv("crane", self.VALID)
        asyncio.run(env.initial_observation())

        # Turn 1: wrong guess
        r1 = asyncio.run(env.step({"role": "assistant", "content": "slate"}))
        assert r1.episode_done is False
        # Conversation should have: system, user, assistant, user (feedback)
        assert len(r1.next_messages) == 4

        # Turn 2: correct guess
        r2 = asyncio.run(env.step({"role": "assistant", "content": "crane"}))
        assert r2.episode_done is True
        assert r2.metrics.get("target_reached") == 1.0
        # system, user, assistant(slate), user(t1), assistant(crane), user(t2)
        assert len(r2.next_messages) == 6

    def test_max_turns_exceeded(self):
        env = WordleMessageEnv("crane", self.VALID, max_turns=2)
        asyncio.run(env.initial_observation())

        r1 = asyncio.run(env.step({"role": "assistant", "content": "slate"}))
        assert r1.episode_done is False

        r2 = asyncio.run(env.step({"role": "assistant", "content": "dusty"}))
        assert r2.episode_done is True
        assert r2.metrics.get("max_turns_exceeded") == 1.0

    def test_invalid_word(self):
        env = WordleMessageEnv("crane", self.VALID)
        asyncio.run(env.initial_observation())
        # "zzzzz" not in valid guesses
        result = asyncio.run(env.step({"role": "assistant", "content": "zzzzz"}))
        assert result.metrics.get("invalid_word") == 1.0
        assert env.turn == 1  # turn still consumed

    def test_guess_extraction(self):
        # <guess> tag is highest priority
        assert _extract_guess("<guess>CRANE</guess>") == "crane"
        assert _extract_guess("I'll try <guess>slate</guess> next") == "slate"
        assert _extract_guess("<think>blah</think><guess>AROSE</guess>") == "arose"
        # Falls back to stripping think blocks
        assert _extract_guess("<think>I think about words</think>SLATE") == "slate"
        assert _extract_guess("<think>maybe crane</think>slate is good") == "slate"
        # Unclosed think block stripped
        assert _extract_guess("<think>Let me think about crane and other") == "other"
        # Plain text fallback
        assert _extract_guess("HELLO") == "hello"
        assert _extract_guess("My answer: slate") == "slate"
        # No match
        assert _extract_guess("no 123") is None
        assert _extract_guess("ab") is None


# ---------------------------------------------------------------------------
# TestRewards
# ---------------------------------------------------------------------------
class TestRewards:
    def test_dense_new_greens(self):
        reward, metrics = dense_reward.compute_turn_reward(
            guess="crane",
            feedback=[G, X, X, X, X],
            prev_feedbacks=[],
            prev_guesses=[],
            turn=1,
            max_turns=6,
            target_reached=False,
        )
        assert metrics["new_greens"] == 1.0
        assert reward >= 0.3  # +0.3 per new green

    def test_dense_new_yellows(self):
        reward, metrics = dense_reward.compute_turn_reward(
            guess="arose",
            feedback=[X, Y, X, X, Y],
            prev_feedbacks=[],
            prev_guesses=[],
            turn=1,
            max_turns=6,
            target_reached=False,
        )
        assert metrics["new_yellows"] == 2.0
        assert reward >= 0.2  # +0.1 per yellow * 2

    def test_dense_constraint_violation(self):
        # First guess: "crane" all grey -> c,r,a,n,e are grey
        # Second guess reuses grey letter
        reward, metrics = dense_reward.compute_turn_reward(
            guess="crest",
            feedback=[X, X, X, X, X],
            prev_feedbacks=[[X, X, X, X, X]],
            prev_guesses=["crane"],
            turn=2,
            max_turns=6,
            target_reached=False,
        )
        assert metrics["constraint_violation"] == 1.0
        assert reward <= -0.5  # should have -0.5 penalty

    def test_dense_episode_win(self):
        reward, metrics = dense_reward.compute_episode_reward(
            target_reached=True, total_turns=3, max_turns=6,
        )
        assert reward == 3.0 + 0.1 * 3  # 3.0 + 0.1 * (6-3)
        assert metrics["target_reached"] == 1.0

    def test_dense_episode_loss(self):
        reward, metrics = dense_reward.compute_episode_reward(
            target_reached=False, total_turns=6, max_turns=6,
        )
        assert reward == -1.0

    def test_sparse_turn_always_zero(self):
        reward, _ = sparse_reward.compute_turn_reward(
            guess="crane",
            feedback=[G, G, G, G, G],
            prev_feedbacks=[],
            prev_guesses=[],
            turn=1,
            max_turns=6,
            target_reached=True,
        )
        assert reward == 0.0

    def test_sparse_episode_win(self):
        reward, metrics = sparse_reward.compute_episode_reward(
            target_reached=True, total_turns=2, max_turns=6,
        )
        assert reward == 3.0 + 0.1 * 4  # turns_remaining = 4
        assert metrics["target_reached"] == 1.0

    def test_sparse_episode_loss(self):
        reward, metrics = sparse_reward.compute_episode_reward(
            target_reached=False, total_turns=6, max_turns=6,
        )
        assert reward == -1.0


# ---------------------------------------------------------------------------
# TestRewardUtils
# ---------------------------------------------------------------------------
class TestRewardUtils:
    def test_count_new_greens_first_turn(self):
        fb = [G, X, G, X, X]
        assert count_new_greens(fb, []) == 2

    def test_count_new_greens_repeated(self):
        prev = [[G, X, X, X, X]]
        fb = [G, G, X, X, X]  # pos 0 was already green
        assert count_new_greens(fb, prev) == 1  # only pos 1 is new

    def test_count_new_yellows(self):
        fb = [X, Y, X, Y, X]
        assert count_new_yellows(fb, []) == 2

    def test_has_constraint_violation_no_history(self):
        assert has_constraint_violation("crane", [], []) is False


# ---------------------------------------------------------------------------
# TestDataLoading
# ---------------------------------------------------------------------------
class TestDataLoading:
    @pytest.fixture
    def data_dir(self):
        return Path(__file__).parent.parent / "data"

    def test_answers_file(self, data_dir):
        path = data_dir / "wordle_answers.txt"
        if not path.exists():
            pytest.skip("Word lists not downloaded yet (run prepare_wordle.py)")
        words = load_word_list(path)
        assert len(words) == 2315
        assert all(len(w) == 5 for w in words)
        assert all(w.isalpha() and w.islower() for w in words)

    def test_guesses_file(self, data_dir):
        path = data_dir / "wordle_guesses.txt"
        if not path.exists():
            pytest.skip("Word lists not downloaded yet (run prepare_wordle.py)")
        words = load_word_list(path)
        assert len(words) == 10657
        assert all(len(w) == 5 for w in words)
        assert all(w.isalpha() and w.islower() for w in words)

    def test_answers_subset_of_valid(self, data_dir):
        answers_path = data_dir / "wordle_answers.txt"
        guesses_path = data_dir / "wordle_guesses.txt"
        if not answers_path.exists() or not guesses_path.exists():
            pytest.skip("Word lists not downloaded yet (run prepare_wordle.py)")
        answers = set(load_word_list(answers_path))
        guesses = set(load_word_list(guesses_path))
        all_valid = answers | guesses
        assert answers.issubset(all_valid)
