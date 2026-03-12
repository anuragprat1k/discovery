"""Tests for problem generation and the Countdown dataset."""

from __future__ import annotations

import pytest

from iterative_countdown.environment.problem_generator import (
    generate_problems,
    solve_countdown,
)
from iterative_countdown.environment.countdown_env import (
    CountdownDataset,
    CountdownEnvGroupBuilder,
)


class TestSolveCountdown:
    """Tests for the Countdown solver."""

    def test_trivial_single_number(self):
        solutions = solve_countdown(25, [25, 50, 75])
        assert len(solutions) >= 1
        assert "25" in solutions

    def test_simple_addition(self):
        solutions = solve_countdown(75, [25, 50])
        assert len(solutions) >= 1
        # At least one solution should involve 25 and 50
        assert any("25" in s and "50" in s for s in solutions)

    def test_simple_multiplication(self):
        solutions = solve_countdown(100, [25, 4])
        assert len(solutions) >= 1

    def test_no_solution(self):
        # 7 cannot be made from [2, 4] with exact integer arithmetic
        solutions = solve_countdown(7, [2, 4])
        assert len(solutions) == 0

    def test_max_solutions_limit(self):
        solutions = solve_countdown(10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], max_solutions=5)
        assert len(solutions) <= 5

    def test_multi_step_solution(self):
        # 103 = 25 * 4 + 3
        solutions = solve_countdown(103, [25, 4, 3, 2])
        assert len(solutions) >= 1


class TestGenerateProblems:
    """Tests for problem generation."""

    def test_generates_correct_count(self):
        problems = generate_problems(20, seed=123)
        assert len(problems) == 20

    def test_problem_structure(self):
        problems = generate_problems(10, seed=42)
        for p in problems:
            assert "target" in p
            assert "numbers" in p
            assert "difficulty" in p
            assert "min_steps" in p
            assert "has_exact_solution" in p
            assert "closest_possible" in p
            assert isinstance(p["target"], int)
            assert isinstance(p["numbers"], list)
            assert p["difficulty"] in ("easy", "medium", "hard")

    def test_difficulty_distribution(self):
        problems = generate_problems(100, seed=42)
        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        for p in problems:
            difficulties[p["difficulty"]] += 1

        # Default is 30/40/30
        assert difficulties["easy"] == 30
        assert difficulties["medium"] == 40
        assert difficulties["hard"] == 30

    def test_custom_difficulty_tiers(self):
        problems = generate_problems(
            100, seed=42, difficulty_tiers={"easy": 0.5, "medium": 0.3, "hard": 0.2}
        )
        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        for p in problems:
            difficulties[p["difficulty"]] += 1

        assert difficulties["easy"] == 50
        assert difficulties["medium"] == 30
        assert difficulties["hard"] == 20

    def test_reproducibility(self):
        p1 = generate_problems(20, seed=42)
        p2 = generate_problems(20, seed=42)
        for a, b in zip(p1, p2):
            assert a["target"] == b["target"]
            assert a["numbers"] == b["numbers"]
            assert a["difficulty"] == b["difficulty"]

    def test_easy_problems_small_target(self):
        problems = generate_problems(
            30, seed=42, difficulty_tiers={"easy": 1.0, "medium": 0.0, "hard": 0.0}
        )
        for p in problems:
            assert p["difficulty"] == "easy"
            assert len(p["numbers"]) == 2
            # Easy problems should have exact solutions
            assert p["has_exact_solution"] is True

    def test_hard_problems_have_six_numbers(self):
        problems = generate_problems(
            30, seed=42, difficulty_tiers={"easy": 0.0, "medium": 0.0, "hard": 1.0}
        )
        for p in problems:
            assert p["difficulty"] == "hard"
            assert len(p["numbers"]) == 6


class TestCountdownDataset:
    """Tests for CountdownDataset."""

    def test_get_batch(self):
        problems = [
            {"target": 10, "numbers": [5, 5]},
            {"target": 20, "numbers": [10, 10]},
            {"target": 30, "numbers": [15, 15]},
            {"target": 40, "numbers": [20, 20]},
        ]
        dataset = CountdownDataset(
            problems=tuple(problems),
            batch_size=2,
            group_size=3,
            max_turns=5,
            renderer_name="role_colon",
            model_name_for_tokenizer="Qwen/Qwen2.5-0.5B",
            reward_type="dense",
            max_trajectory_tokens=4096,
        )

        batch = dataset.get_batch(0)
        assert len(batch) == 2
        assert all(isinstance(b, CountdownEnvGroupBuilder) for b in batch)

        # Check first builder
        b0 = batch[0]
        assert isinstance(b0, CountdownEnvGroupBuilder)
        assert b0.target == 10
        assert b0.numbers == (5, 5)
        assert b0.num_envs == 3

    def test_dataset_length(self):
        problems = [{"target": i, "numbers": [i]} for i in range(10)]
        dataset = CountdownDataset(
            problems=tuple(problems),
            batch_size=3,
            group_size=1,
            max_turns=5,
            renderer_name="role_colon",
            model_name_for_tokenizer="Qwen/Qwen2.5-0.5B",
            reward_type="dense",
            max_trajectory_tokens=4096,
        )
        assert len(dataset) == 3  # 10 // 3 = 3

    def test_batch_wraps_around(self):
        problems = [
            {"target": 10, "numbers": [5, 5]},
            {"target": 20, "numbers": [10, 10]},
        ]
        dataset = CountdownDataset(
            problems=tuple(problems),
            batch_size=3,
            group_size=1,
            max_turns=5,
            renderer_name="role_colon",
            model_name_for_tokenizer="Qwen/Qwen2.5-0.5B",
            reward_type="dense",
            max_trajectory_tokens=4096,
        )

        batch = dataset.get_batch(0)
        assert len(batch) == 3
        # Should wrap: problems[0], problems[1], problems[0]
        targets = [b.target for b in batch]
        assert targets == [10, 20, 10]
