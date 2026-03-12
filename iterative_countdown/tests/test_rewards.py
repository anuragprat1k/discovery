"""Tests for all reward functions."""
from __future__ import annotations

import math

import pytest

from iterative_countdown.rewards import reward_utils
from iterative_countdown.rewards import binary_reward
from iterative_countdown.rewards import dense_reward
from iterative_countdown.rewards import prime_reward


# ---------------------------------------------------------------------------
# reward_utils
# ---------------------------------------------------------------------------

class TestProximityScore:
    def test_exact_match(self):
        assert reward_utils.proximity_score(100, 100) == 1.0

    def test_far_away(self):
        score = reward_utils.proximity_score(0, 100)
        assert 0.0 < score < 0.5  # far away -> low score

    def test_close(self):
        score = reward_utils.proximity_score(99, 100)
        assert score > 0.9  # very close -> high score

    def test_custom_max_distance(self):
        score = reward_utils.proximity_score(50, 100, max_distance=200)
        # scale = 200, distance = 50 -> exp(-50/200) ≈ 0.778
        assert 0.7 < score < 0.85

    def test_negative_target(self):
        # Should handle negative targets (scale uses abs)
        score = reward_utils.proximity_score(-10, -10)
        assert score == 1.0

    def test_zero_target(self):
        # scale = max(0, 1) = 1
        score = reward_utils.proximity_score(0, 0)
        assert score == 1.0

    def test_zero_target_nonzero_value(self):
        score = reward_utils.proximity_score(5, 0)
        # scale = 1, distance = 5 -> exp(-5) ≈ 0.0067
        assert score == pytest.approx(math.exp(-5), abs=1e-6)


class TestProximityImprovement:
    def test_positive_improvement(self):
        # Got closer: distance went from 50 to 10
        score = reward_utils.proximity_improvement(50, 10, 100)
        assert score > 0

    def test_negative_improvement(self):
        # Got further: distance went from 10 to 50
        score = reward_utils.proximity_improvement(10, 50, 100)
        assert score < 0

    def test_no_change(self):
        score = reward_utils.proximity_improvement(30, 30, 100)
        assert score == 0.0

    def test_normalization(self):
        # Improvement of 40 with target 100 -> 0.4
        score = reward_utils.proximity_improvement(50, 10, 100)
        assert score == pytest.approx(0.4)


class TestValidityScore:
    def test_all_valid(self):
        assert reward_utils.validity_score(True, True) == 1.0

    def test_invalid_syntax(self):
        assert reward_utils.validity_score(False, False) == 0.0

    def test_valid_syntax_wrong_numbers(self):
        assert reward_utils.validity_score(True, False) == 0.5

    def test_invalid_syntax_valid_numbers(self):
        # syntax invalid trumps everything
        assert reward_utils.validity_score(False, True) == 0.0


class TestConstraintScore:
    def test_all_valid(self):
        assert reward_utils.constraint_score([1, 2], [1, 2, 3]) == 1.0

    def test_one_invalid(self):
        score = reward_utils.constraint_score([1, 99], [1, 2, 3])
        assert score == pytest.approx(0.5)

    def test_all_invalid(self):
        score = reward_utils.constraint_score([88, 99], [1, 2, 3])
        assert score == 0.0

    def test_empty_used(self):
        assert reward_utils.constraint_score([], [1, 2, 3]) == 1.0

    def test_duplicate_use(self):
        # Using 1 twice but only one 1 available
        score = reward_utils.constraint_score([1, 1], [1, 2, 3])
        assert score == pytest.approx(0.5)

    def test_duplicate_available(self):
        # Two 1s available, using both is fine
        score = reward_utils.constraint_score([1, 1], [1, 1, 3])
        assert score == 1.0


class TestProgressScore:
    def test_full_progress_early(self):
        # Reached target on turn 0 with max_turns 5
        score = reward_utils.progress_score(0, 5, 0, 100)
        assert score == 1.0

    def test_no_progress(self):
        # Distance unchanged from initial
        score = reward_utils.progress_score(3, 5, 100, 100)
        assert score == 0.0

    def test_half_progress_midway(self):
        score = reward_utils.progress_score(2, 5, 50, 100)
        assert 0.0 < score < 1.0

    def test_zero_initial_distance(self):
        score = reward_utils.progress_score(0, 5, 0, 0)
        assert score == 1.0


class TestCorrectnessScore:
    def test_correct(self):
        assert reward_utils.correctness_score(42, 42) == 1.0

    def test_incorrect(self):
        assert reward_utils.correctness_score(41, 42) == 0.0

    def test_zero(self):
        assert reward_utils.correctness_score(0, 0) == 1.0


# ---------------------------------------------------------------------------
# binary_reward
# ---------------------------------------------------------------------------

class TestBinaryReward:
    def test_turn_reward_always_zero(self):
        reward, metrics = binary_reward.compute_turn_reward(
            expression_valid=True,
            result=50,
            target=100,
            available_numbers=[1, 2, 3],
            turn=1,
            max_turns=5,
            best_distance=50,
            initial_distance=100,
        )
        assert reward == 0.0
        assert metrics["turn"] == 1
        assert metrics["valid"] == 1.0
        assert metrics["distance"] == 50

    def test_turn_reward_invalid(self):
        reward, metrics = binary_reward.compute_turn_reward(
            expression_valid=False,
            result=None,
            target=100,
            available_numbers=[1, 2, 3],
            turn=2,
            max_turns=5,
            best_distance=100,
            initial_distance=100,
        )
        assert reward == 0.0
        assert metrics["valid"] == 0.0
        assert "distance" not in metrics

    def test_episode_reward_success(self):
        reward, metrics = binary_reward.compute_episode_reward(
            target_reached=True,
            best_distance=0,
            initial_distance=100,
            total_turns=3,
            max_turns=5,
        )
        assert reward == 1.0
        assert metrics["target_reached"] == 1.0

    def test_episode_reward_failure(self):
        reward, metrics = binary_reward.compute_episode_reward(
            target_reached=False,
            best_distance=10,
            initial_distance=100,
            total_turns=5,
            max_turns=5,
        )
        assert reward == 0.0
        assert metrics["target_reached"] == 0.0


# ---------------------------------------------------------------------------
# dense_reward
# ---------------------------------------------------------------------------

class TestDenseReward:
    def test_valid_expression_closer(self):
        reward, metrics = dense_reward.compute_turn_reward(
            expression_valid=True,
            result=90,
            target=100,
            available_numbers=[1, 2, 3, 4, 5, 90],
            numbers_used=[90],
            turn=1,
            max_turns=5,
            best_distance=100,
            initial_distance=100,
            prev_distance=100,
        )
        assert reward > 0  # got closer -> positive reward
        assert metrics["distance"] == 10
        assert metrics["distance_improvement"] == 90

    def test_invalid_expression_penalty(self):
        reward, metrics = dense_reward.compute_turn_reward(
            expression_valid=False,
            result=None,
            target=100,
            available_numbers=[1, 2, 3],
            numbers_used=None,
            turn=1,
            max_turns=5,
            best_distance=100,
            initial_distance=100,
            prev_distance=100,
        )
        assert reward == pytest.approx(-0.1)

    def test_exact_match_high_reward(self):
        reward, metrics = dense_reward.compute_turn_reward(
            expression_valid=True,
            result=100,
            target=100,
            available_numbers=[25, 4, 100],
            numbers_used=[25, 4],
            turn=1,
            max_turns=5,
            best_distance=100,
            initial_distance=100,
            prev_distance=100,
        )
        # correctness component alone is 0.4 * 1.0 = 0.4
        assert reward > 0.3

    def test_further_away_lower_reward(self):
        # Moving further from target should give lower reward than closer
        reward_closer, _ = dense_reward.compute_turn_reward(
            expression_valid=True,
            result=90,
            target=100,
            available_numbers=[90],
            numbers_used=[90],
            turn=1,
            max_turns=5,
            best_distance=100,
            initial_distance=100,
            prev_distance=100,
        )
        reward_further, _ = dense_reward.compute_turn_reward(
            expression_valid=True,
            result=10,
            target=100,
            available_numbers=[10],
            numbers_used=[10],
            turn=1,
            max_turns=5,
            best_distance=100,
            initial_distance=100,
            prev_distance=100,
        )
        assert reward_closer > reward_further

    def test_custom_weights(self):
        reward, _ = dense_reward.compute_turn_reward(
            expression_valid=True,
            result=100,
            target=100,
            available_numbers=[100],
            numbers_used=[100],
            turn=1,
            max_turns=5,
            best_distance=100,
            initial_distance=100,
            prev_distance=100,
            weights={"proximity": 0, "validity": 0, "progress": 0, "correctness": 1.0},
        )
        # Only correctness matters, and result == target
        assert reward == pytest.approx(1.0)

    def test_episode_reward_success(self):
        reward, metrics = dense_reward.compute_episode_reward(
            target_reached=True,
            best_distance=0,
            initial_distance=100,
            total_turns=1,
            max_turns=5,
        )
        # efficiency_bonus = 0.2 * (1 - 1/5) = 0.16
        assert reward == pytest.approx(0.16)
        assert metrics["target_reached"] == 1.0

    def test_episode_reward_failure_partial_credit(self):
        reward, metrics = dense_reward.compute_episode_reward(
            target_reached=False,
            best_distance=20,
            initial_distance=100,
            total_turns=5,
            max_turns=5,
        )
        # 0.1 * (1 - 20/100) = 0.08
        assert reward == pytest.approx(0.08)

    def test_episode_reward_failure_no_progress(self):
        reward, _ = dense_reward.compute_episode_reward(
            target_reached=False,
            best_distance=100,
            initial_distance=100,
            total_turns=5,
            max_turns=5,
        )
        assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# prime_reward
# ---------------------------------------------------------------------------

class TestPrimeReward:
    def test_valid_expression_small_reward(self):
        reward, metrics = prime_reward.compute_turn_reward(
            expression_valid=True,
            result=50,
            target=100,
            available_numbers=[1, 2, 3, 50],
            numbers_used=[50],
            turn=1,
            max_turns=5,
            best_distance=50,
            initial_distance=100,
            prev_distance=100,
        )
        assert reward == pytest.approx(0.05)
        assert metrics["distance"] == 50

    def test_invalid_expression_small_penalty(self):
        reward, metrics = prime_reward.compute_turn_reward(
            expression_valid=False,
            result=None,
            target=100,
            available_numbers=[1, 2, 3],
            numbers_used=None,
            turn=1,
            max_turns=5,
            best_distance=100,
            initial_distance=100,
            prev_distance=100,
        )
        assert reward == pytest.approx(-0.05)

    def test_exact_match_full_reward(self):
        reward, metrics = prime_reward.compute_turn_reward(
            expression_valid=True,
            result=100,
            target=100,
            available_numbers=[25, 4, 100],
            numbers_used=[25, 4],
            turn=1,
            max_turns=5,
            best_distance=0,
            initial_distance=100,
            prev_distance=100,
        )
        assert reward == 1.0

    def test_episode_reward_success_efficient(self):
        reward, metrics = prime_reward.compute_episode_reward(
            target_reached=True,
            best_distance=0,
            initial_distance=100,
            total_turns=1,
            max_turns=5,
        )
        # 1.0 + 0.3 * (1 - 1/5) = 1.24
        assert reward == pytest.approx(1.24)

    def test_episode_reward_success_slow(self):
        reward, metrics = prime_reward.compute_episode_reward(
            target_reached=True,
            best_distance=0,
            initial_distance=100,
            total_turns=5,
            max_turns=5,
        )
        # 1.0 + 0.3 * 0 = 1.0
        assert reward == pytest.approx(1.0)

    def test_episode_reward_failure(self):
        reward, metrics = prime_reward.compute_episode_reward(
            target_reached=False,
            best_distance=50,
            initial_distance=100,
            total_turns=5,
            max_turns=5,
        )
        # 0.3 * (1 - 50/100) = 0.15
        assert reward == pytest.approx(0.15)

    def test_episode_with_trajectory(self):
        reward_no_traj, _ = prime_reward.compute_episode_reward(
            target_reached=True,
            best_distance=0,
            initial_distance=100,
            total_turns=3,
            max_turns=5,
        )
        texts = [
            "Expression: 25 * 4",
            "Expression: 100 + 50",
            "Expression: 150 - 2",
        ]
        reward_with_traj, metrics = prime_reward.compute_episode_reward(
            target_reached=True,
            best_distance=0,
            initial_distance=100,
            total_turns=3,
            max_turns=5,
            trajectory_texts=texts,
        )
        # Should get a process quality bonus
        assert reward_with_traj > reward_no_traj
        assert "process_quality" in metrics

    def test_process_quality_diverse_ops(self):
        texts = [
            "Expression: 25 * 4",
            "Expression: 100 + 50",
            "Expression: 150 - 2",
        ]
        quality = prime_reward._evaluate_process_quality(texts)
        # 3 unique ops * 0.15 = 0.45
        assert quality > 0.3

    def test_process_quality_repeated_expressions(self):
        texts = [
            "Expression: 25 * 4",
            "Expression: 25 * 4",
            "Expression: 25 * 4",
        ]
        quality = prime_reward._evaluate_process_quality(texts)
        # 1 op * 0.15 = 0.15, minus 2 repeats * 0.2 = -0.4 -> net -0.25 -> clamped to 0
        assert quality == 0.0

    def test_process_quality_empty(self):
        assert prime_reward._evaluate_process_quality([]) == 0.0

    def test_process_quality_no_expressions(self):
        texts = ["Some random text", "Another line"]
        assert prime_reward._evaluate_process_quality(texts) == 0.0
