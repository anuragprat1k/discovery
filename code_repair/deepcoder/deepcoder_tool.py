from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any

from code_repair.deepcoder.code_grading import (
    extract_code_from_model,
    sandbox_check_correctness,
)
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool
from tinker_cookbook.utils import logtree


@dataclass(frozen=True)
class DeepcoderTask:
    """A single code task with problem statement and test cases."""

    problem: str
    tests: list[dict[str, Any]]
    starter_code: str | None = None


class DeepcoderTool:
    """Tool for testing code against a task's test cases.

    Each DeepcoderTool instance is bound to a specific task (its tests).
    """

    def __init__(
        self,
        task: DeepcoderTask,
        sandbox_backend: SandboxBackend | None = None,
        timeout: int = 6,
    ):
        self._task = task
        self._sandbox_backend = sandbox_backend
        self._timeout = timeout

    @tool
    async def check_solution(
        self,
        code: Annotated[str, "Python code implementing the solution."],
    ) -> ToolResult:
        """Execute the proposed solution against the task's test cases.

        Use this to test your code before providing your final answer.
        """
        try:
            passed, details = await sandbox_check_correctness(
                self._task.tests,
                code,
                timeout=self._timeout,
                backend=self._sandbox_backend,
            )
            content = json.dumps(
                {"passed": passed, "details": details},
                ensure_ascii=False,
            )
            return simple_tool_result(content)
        except Exception as e:
            return simple_tool_result(json.dumps({"error": str(e), "passed": False}))


@dataclass
class DeepcoderReward:
    """Reward function for code tasks with sparse/dense/dense_full modes.

    Grades the final answer by extracting code from the last assistant message
    and running it against the task's tests.

    Reward modes:
    - sparse: format_coef * (has_code - 1) + all_correct (binary)
    - dense: format_coef * (has_code - 1) + tests_passed / tests_total
    - dense_full: dense + 0.1 * (code ran without crash for failing tests)

    Called once at episode end with the full message history.
    """

    task: DeepcoderTask
    sandbox_backend: SandboxBackend | None = None
    timeout: int = 6
    format_coef: float = 0.1
    reward_type: str = "sparse"  # "sparse", "dense", "dense_full"

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """Grade the completed episode by extracting code from final assistant message."""
        # Find the last assistant message
        final_message = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                final_message = msg
                break

        if final_message is None:
            logtree.log_text("No assistant message found in history.")
            return 0.0, {"format": 0.0, "correct": 0.0, "tests_passed": 0.0, "tests_total": 0.0}

        content = get_text_content(final_message)
        code = extract_code_from_model(content)
        has_code_block = code is not None

        tests_passed = 0
        tests_total = len(self.task.tests) or 1
        all_correct = False

        if code is not None:
            try:
                passed, details = await sandbox_check_correctness(
                    self.task.tests,
                    code,
                    timeout=self.timeout,
                    backend=self.sandbox_backend,
                )
                all_correct = passed
                tests_passed = details.get("tests_passed", tests_total if passed else 0)
                tests_total = details.get("tests_total", tests_total)
            except Exception as e:
                logtree.log_text(f"Error running tests: {e}")
        else:
            logtree.log_text("No code block detected in response.")

        # Compute reward based on mode
        format_score = float(has_code_block)
        format_reward = self.format_coef * (format_score - 1.0)

        if self.reward_type == "sparse":
            reward = format_reward + float(all_correct)
        elif self.reward_type == "dense":
            reward = format_reward + (tests_passed / max(tests_total, 1))
        elif self.reward_type == "dense_full":
            frac = tests_passed / max(tests_total, 1)
            # Bonus: code ran at all (didn't crash on import/syntax)
            ran_at_all = 0.1 if (has_code_block and tests_passed > 0) else 0.0
            reward = format_reward + frac + ran_at_all
        else:
            reward = format_reward + float(all_correct)

        metrics = {
            "format": format_score,
            "correct": float(all_correct),
            "tests_passed": float(tests_passed),
            "tests_total": float(tests_total),
            "test_frac": float(tests_passed / max(tests_total, 1)),
        }

        logtree.log_text(
            f"Format: {'✓' if has_code_block else '✗'}, "
            f"Tests: {tests_passed}/{tests_total}, "
            f"Reward ({self.reward_type}): {reward:.3f}"
        )

        return reward, metrics
