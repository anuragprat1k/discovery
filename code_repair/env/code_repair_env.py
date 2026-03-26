"""Multi-turn code repair environment compatible with TRL's GRPOTrainer.

Each episode: model receives buggy code + test results, outputs <repair>...</repair>,
gets updated test results, repeats for up to max_turns.

Follows the same Gymnasium-style reset()/step() pattern as WordleGymEnv.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from code_repair.env.sandbox import run_tests, TestResult


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a code repair assistant. You will be given a buggy Python function and \
test results showing which tests pass and which fail. Fix the function so all \
tests pass.

IMPORTANT: You MUST output your repaired function inside <repair>...</repair> tags. \
Do NOT use markdown code blocks. Only output inside <repair> tags will be evaluated.

Example of correct output format:
<repair>
def add(a, b):
    return a + b
</repair>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_repair(text: str) -> str | None:
    """Extract code from <repair>...</repair> tags in model output."""
    match = re.search(r"<repair>(.*?)</repair>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def format_test_results(
    results: list[TestResult],
    turn: int,
    max_turns: int,
    prev_passing: set[str] | None = None,
) -> str:
    """Format test results for the model prompt."""
    lines = [f"## Test Results (Turn {turn} of {max_turns})"]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        detail = ""
        if not r.passed:
            if r.returned is not None and r.expected is not None:
                detail = f" (expected {r.expected}, got {r.returned})"
            elif r.error:
                detail = f" ({r.error[:100]})"
        newly = ""
        if r.passed and prev_passing is not None and r.name not in prev_passing:
            newly = " ✓ NEWLY PASSING"
        lines.append(f"- {r.name}: {status}{detail}{newly}")

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    lines.append(f"\n{passed}/{total} tests passing")
    return "\n".join(lines)


def format_initial_prompt(
    buggy_code: str,
    test_results: list[TestResult],
    max_turns: int,
) -> str:
    """Format the initial user message with buggy code and test results."""
    results_str = format_test_results(test_results, turn=0, max_turns=max_turns)

    return f"""\
## Buggy Function
```python
{buggy_code}
```

{results_str}

Fix the function:"""


def format_feedback(
    test_results: list[TestResult],
    prev_repair: str,
    turn: int,
    max_turns: int,
    prev_passing: set[str] | None = None,
) -> str:
    """Format feedback after a repair attempt."""
    results_str = format_test_results(
        test_results, turn=turn, max_turns=max_turns,
        prev_passing=prev_passing,
    )

    return f"""\
{results_str}

## Previous Repair
```python
{prev_repair}
```

Fix the function:"""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CodeRepairEnv:
    """Multi-turn code repair environment.

    Compatible with TRL's GRPOTrainer via reset()/step() interface.
    Rewards are computed externally via reward functions in rewards.py.
    """

    def __init__(
        self,
        problems: list[dict],
        max_turns: int = 4,
        sandbox_timeout: int = 5,
    ):
        self.problems = problems
        self.max_turns = max_turns
        self.sandbox_timeout = sandbox_timeout

        # Episode state
        self.current_problem: dict = {}
        self.turn: int = 0
        self.done: bool = False
        self.all_passed: bool = False
        self.test_results_history: list[list[TestResult]] = []
        self.repairs: list[str] = []
        self.hw_passing: int = 0  # high-water mark of passing tests

    def reset(self, *, problem_idx: int | None = None, seed: int | None = None) -> dict:
        """Start a new code repair episode.

        Returns initial observation with system prompt and buggy code + test results.
        """
        if seed is not None:
            random.seed(seed)

        if problem_idx is not None:
            self.current_problem = self.problems[problem_idx]
        else:
            self.current_problem = random.choice(self.problems)

        self.turn = 0
        self.done = False
        self.all_passed = False
        self.test_results_history = []
        self.repairs = []
        self.hw_passing = 0

        p = self.current_problem

        # Run tests on buggy code to get initial results
        initial_results = run_tests(
            p["buggy_code"], p["test_code"], p["entry_point"],
            timeout=self.sandbox_timeout, detailed=True,
        )
        self.test_results_history.append(initial_results)

        # Build initial prompt
        user_msg = format_initial_prompt(
            p["buggy_code"], initial_results, self.max_turns,
        )

        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        }

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        """Process model's repair attempt.

        Args:
            action: Raw model output containing <repair>...</repair> tags.

        Returns:
            (observation, reward, done, info)
            reward is always 0.0 — actual rewards computed externally.
        """
        if self.done:
            return ({"messages": []}, 0.0, True, {"error": "episode already done"})

        self.turn += 1
        p = self.current_problem
        info: dict = {
            "turn": self.turn,
            "task_id": p["task_id"],
            "max_turns": self.max_turns,
        }

        # Parse repair
        repair = extract_repair(action)
        if repair is None:
            # Format penalty — no valid repair tag
            episode_done = self.turn >= self.max_turns
            self.done = episode_done
            info["invalid_parse"] = True
            info["format_violation"] = True
            remaining = self.max_turns - self.turn

            # Reuse last test results
            prev_results = self.test_results_history[-1]
            prev_passing = {r.name for r in prev_results if r.passed}
            info["test_results"] = prev_results
            info["prev_passing"] = len(prev_passing)
            info["curr_passing"] = len(prev_passing)
            info["hw_passing"] = self.hw_passing
            info["num_tests"] = len(prev_results)
            info["all_passed"] = False
            info["episode_done"] = episode_done

            feedback_text = (
                f"I couldn't find a valid repair in your response. "
                f"Please output your fix inside <repair>...</repair> tags. "
                f"({remaining} turn(s) remaining)"
            )
            return (
                {"messages": [{"role": "user", "content": feedback_text}]},
                0.0,
                episode_done,
                info,
            )

        self.repairs.append(repair)
        info["repair"] = repair

        # Run tests on repaired code
        test_results = run_tests(
            repair, p["test_code"], p["entry_point"],
            timeout=self.sandbox_timeout, detailed=True,
        )
        self.test_results_history.append(test_results)

        # Compute metrics
        prev_results = self.test_results_history[-2]
        prev_passing = {r.name for r in prev_results if r.passed}
        curr_passing = {r.name for r in test_results if r.passed}
        num_tests = max(len(test_results), 1)

        # Update high-water mark
        old_hw = self.hw_passing
        self.hw_passing = max(self.hw_passing, len(curr_passing))

        all_passed = len(curr_passing) == len(test_results) and len(test_results) > 0
        # Also verify no error-only results (like timeout)
        if len(test_results) == 1 and test_results[0].name in ("timeout", "syntax_error", "runtime_error", "import_error"):
            all_passed = False

        self.all_passed = all_passed
        episode_done = all_passed or self.turn >= self.max_turns
        self.done = episode_done

        info["test_results"] = test_results
        info["prev_passing"] = len(prev_passing)
        info["curr_passing"] = len(curr_passing)
        info["hw_passing"] = self.hw_passing
        info["old_hw"] = old_hw
        info["num_tests"] = num_tests
        info["all_passed"] = all_passed
        info["episode_done"] = episode_done
        info["target_reached"] = all_passed

        # Partial-correctness signals for non-potential reward
        no_crash_count = sum(1 for r in test_results if r.no_crash and not r.passed)
        type_match_count = 0
        shape_match_count = 0
        for r in test_results:
            if r.passed or not r.no_crash:
                continue
            # Check type match (return_type from detailed results)
            if hasattr(r, "return_type") and r.return_type is not None:
                # We need expected type - stored in raw results
                type_match_count += 1  # approximate: no_crash implies some type returned
            if r.return_shape is not None:
                shape_match_count += 1

        info["no_crash_failing"] = no_crash_count
        info["type_match_failing"] = type_match_count
        info["shape_match_failing"] = shape_match_count

        if all_passed:
            feedback_text = (
                f"All {num_tests} tests passing! "
                f"You fixed the bug in {self.turn} turn(s)."
            )
        else:
            feedback_text = format_feedback(
                test_results, repair,
                turn=self.turn, max_turns=self.max_turns,
                prev_passing=prev_passing,
            )

        return (
            {"messages": [{"role": "user", "content": feedback_text}]},
            0.0,
            episode_done,
            info,
        )
