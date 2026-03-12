"""Brute-force Countdown solver and problem generation.

Generates Countdown numbers game problems at varying difficulty levels and
verifies solvability using an iterative-deepening solver.
"""

from __future__ import annotations

import itertools
import random
from typing import Any


def solve_countdown(
    target: int,
    numbers: list[int],
    max_solutions: int = 100,
) -> list[str]:
    """Find solutions to a Countdown numbers problem.

    Uses iterative deepening: tries combinations of 1 number, then 2, then 3,
    etc. For each subset size, tries all permutations and all operator
    combinations to find expressions that equal the target.

    Args:
        target: The target number to reach.
        numbers: Available source numbers.
        max_solutions: Maximum number of solutions to return.

    Returns:
        List of expression strings that evaluate to the target.
    """
    solutions: list[str] = []
    ops = ["+", "-", "*", "/"]

    # Iterative deepening: try using 1 number, then 2, etc.
    for n_nums in range(1, len(numbers) + 1):
        if len(solutions) >= max_solutions:
            break
        # Try all subsets of size n_nums
        for combo in itertools.combinations(range(len(numbers)), n_nums):
            if len(solutions) >= max_solutions:
                break
            subset = [numbers[i] for i in combo]

            if n_nums == 1:
                if subset[0] == target:
                    solutions.append(str(subset[0]))
                continue

            # Try all permutations of the subset
            for perm in itertools.permutations(subset):
                if len(solutions) >= max_solutions:
                    break
                # Try all operator combinations
                for op_combo in itertools.product(ops, repeat=n_nums - 1):
                    expr_str = _build_left_to_right_expr(list(perm), list(op_combo))
                    result = _eval_left_to_right(list(perm), list(op_combo))
                    if result is not None and result == target:
                        if expr_str not in solutions:
                            solutions.append(expr_str)
                            if len(solutions) >= max_solutions:
                                break

    return solutions


def _build_left_to_right_expr(nums: list[int], ops: list[str]) -> str:
    """Build a left-to-right expression string (with parentheses for clarity)."""
    if len(nums) == 1:
        return str(nums[0])
    expr = str(nums[0])
    for i, op in enumerate(ops):
        if i > 0:
            expr = f"({expr})"
        expr = f"{expr} {op} {nums[i + 1]}"
    return expr


def _eval_left_to_right(nums: list[int], ops: list[str]) -> int | None:
    """Evaluate a left-to-right expression. Returns None if invalid."""
    result = nums[0]
    for i, op in enumerate(ops):
        val = nums[i + 1]
        if op == "+":
            result = result + val
        elif op == "-":
            result = result - val
        elif op == "*":
            result = result * val
        elif op == "/":
            if val == 0 or result % val != 0:
                return None
            result = result // val
    return result


def _find_min_steps(target: int, numbers: list[int]) -> int:
    """Find the minimum number of operations needed to reach the target."""
    ops = ["+", "-", "*", "/"]

    for n_nums in range(1, min(len(numbers) + 1, 7)):
        for combo in itertools.combinations(range(len(numbers)), n_nums):
            subset = [numbers[i] for i in combo]
            if n_nums == 1:
                if subset[0] == target:
                    return 0  # no operations needed
                continue
            for perm in itertools.permutations(subset):
                for op_combo in itertools.product(ops, repeat=n_nums - 1):
                    result = _eval_left_to_right(list(perm), list(op_combo))
                    if result == target:
                        return n_nums - 1  # number of operations
    return -1  # no exact solution found


def _find_closest(target: int, numbers: list[int]) -> int:
    """Find the closest achievable value to the target."""
    best = numbers[0] if numbers else 0
    best_dist = abs(target - best)
    ops = ["+", "-", "*", "/"]

    for n_nums in range(1, min(len(numbers) + 1, 5)):  # limit search depth
        for combo in itertools.combinations(range(len(numbers)), n_nums):
            subset = [numbers[i] for i in combo]
            if n_nums == 1:
                dist = abs(target - subset[0])
                if dist < best_dist:
                    best_dist = dist
                    best = subset[0]
                continue
            for perm in itertools.permutations(subset):
                for op_combo in itertools.product(ops, repeat=n_nums - 1):
                    result = _eval_left_to_right(list(perm), list(op_combo))
                    if result is not None:
                        dist = abs(target - result)
                        if dist < best_dist:
                            best_dist = dist
                            best = result
                        if dist == 0:
                            return best
    return best


def _generate_easy_problem(rng: random.Random) -> dict[str, Any]:
    """Generate an easy problem: 2 numbers, small target, 1-2 steps."""
    # Pick 2 source numbers from small set
    small_numbers = list(range(1, 11))
    nums = rng.sample(small_numbers, 2)
    ops = ["+", "-", "*"]
    op = rng.choice(ops)

    a, b = nums[0], nums[1]
    if op == "+":
        target = a + b
    elif op == "-":
        target = max(a, b) - min(a, b)
        nums = [max(a, b), min(a, b)]
    else:
        target = a * b

    return {
        "target": target,
        "numbers": nums,
        "difficulty": "easy",
        "min_steps": 1,
        "has_exact_solution": True,
        "closest_possible": target,
    }


def _generate_medium_problem(rng: random.Random) -> dict[str, Any]:
    """Generate a medium problem: 4 numbers, target 100-500, 2-3 steps."""
    small = list(range(1, 11))
    large = [25, 50, 75, 100]

    # Pick 2 small and 2 large (or mixed)
    nums = rng.sample(large, min(2, len(large))) + rng.sample(small, 2)
    rng.shuffle(nums)

    # Generate target by combining numbers
    target = _generate_target_from_numbers(rng, nums, min_ops=2, max_ops=3)
    if target is None or target < 100 or target > 500:
        # Fallback: pick a random target in range
        target = rng.randint(100, 500)

    min_steps = _find_min_steps(target, nums)
    has_exact = min_steps >= 0
    if not has_exact:
        closest = _find_closest(target, nums)
        min_steps = 3  # estimate
    else:
        closest = target

    return {
        "target": target,
        "numbers": nums,
        "difficulty": "medium",
        "min_steps": max(min_steps, 2),
        "has_exact_solution": has_exact,
        "closest_possible": closest,
    }


def _generate_hard_problem(rng: random.Random) -> dict[str, Any]:
    """Generate a hard problem: 6 numbers, target 100-999, 3+ steps."""
    small = list(range(1, 11))
    large = [25, 50, 75, 100]

    n_large = rng.randint(1, 4)
    n_small = 6 - n_large
    nums = rng.sample(large, min(n_large, len(large)))
    if len(nums) < n_large:
        nums += rng.choices(large, k=n_large - len(nums))
    nums += rng.sample(small, min(n_small, len(small)))
    if len(nums) < 6:
        nums += rng.choices(small, k=6 - len(nums))
    rng.shuffle(nums)

    # Generate a target
    target = _generate_target_from_numbers(rng, nums, min_ops=3, max_ops=4)
    if target is None or target < 100 or target > 999:
        target = rng.randint(100, 999)

    min_steps = _find_min_steps(target, nums)
    has_exact = min_steps >= 0
    if not has_exact:
        closest = _find_closest(target, nums)
        min_steps = 4  # estimate
    else:
        closest = target

    return {
        "target": target,
        "numbers": nums,
        "difficulty": "hard",
        "min_steps": max(min_steps, 3),
        "has_exact_solution": has_exact,
        "closest_possible": closest,
    }


def _generate_target_from_numbers(
    rng: random.Random,
    nums: list[int],
    min_ops: int,
    max_ops: int,
) -> int | None:
    """Generate a reachable target by randomly combining some numbers."""
    ops = ["+", "-", "*", "/"]
    n_ops = rng.randint(min_ops, max_ops)
    n_nums = n_ops + 1

    if n_nums > len(nums):
        n_nums = len(nums)
        n_ops = n_nums - 1

    selected = rng.sample(nums, n_nums)
    selected_ops = [rng.choice(ops) for _ in range(n_ops)]

    result = _eval_left_to_right(selected, selected_ops)
    if result is not None and result > 0:
        return result
    return None


def generate_problems(
    n: int,
    seed: int = 42,
    difficulty_tiers: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Generate Countdown problems with specified difficulty distribution.

    Args:
        n: Number of problems to generate.
        seed: Random seed for reproducibility.
        difficulty_tiers: Distribution of difficulties, e.g.,
            {"easy": 0.3, "medium": 0.4, "hard": 0.3}. Defaults to this
            distribution if not provided.

    Returns:
        List of problem dicts with keys: target, numbers, difficulty,
        min_steps, has_exact_solution, closest_possible.
    """
    if difficulty_tiers is None:
        difficulty_tiers = {"easy": 0.3, "medium": 0.4, "hard": 0.3}

    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []

    generators = {
        "easy": _generate_easy_problem,
        "medium": _generate_medium_problem,
        "hard": _generate_hard_problem,
    }

    # Compute counts for each tier
    tier_counts: dict[str, int] = {}
    remaining = n
    for i, (tier, fraction) in enumerate(difficulty_tiers.items()):
        if i == len(difficulty_tiers) - 1:
            tier_counts[tier] = remaining
        else:
            count = round(n * fraction)
            tier_counts[tier] = count
            remaining -= count

    for tier, count in tier_counts.items():
        gen = generators[tier]
        for _ in range(count):
            problem = gen(rng)
            problems.append(problem)

    # Shuffle problems
    rng.shuffle(problems)
    return problems
