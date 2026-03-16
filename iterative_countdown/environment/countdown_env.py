"""Countdown numbers game environment for multi-turn RL.

Implements the MessageEnv interface: each turn the model proposes an arithmetic
expression, the environment validates it, computes the result, updates the
available numbers, and returns feedback. Also provides EnvGroupBuilder,
RLDataset, and RLDatasetBuilder for integration with tinker-cookbook's RL
training loop.

The CountdownMessageEnv class can be used standalone without tinker-cookbook
(for testing). The tinker-cookbook integration classes (CountdownEnvGroupBuilder,
CountdownDataset, CountdownDatasetBuilder) require tinker-cookbook at runtime.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .expression_parser import extract_expression, parse_expression
from .problem_generator import generate_problems

logger = logging.getLogger(__name__)


def _get_text_content(message: dict) -> str:
    """Extract text content from a message dict."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p.get("type") == "text")


# Simple dataclass to mirror tinker_cookbook.rl.message_env.MessageStepResult
# so CountdownMessageEnv can work without tinker-cookbook installed.
@dataclass
class MessageStepResult:
    """Result of a message-level environment step."""
    reward: float
    episode_done: bool
    next_messages: list[dict[str, str]]
    metrics: dict[str, float] = field(default_factory=dict)
    next_stop_condition: Any = None


SYSTEM_PROMPT = """\
You are playing the Countdown numbers game.

Goal: Reach the target number by combining available numbers with +, -, *, /.

Rules:
- Each turn, pick SOME (not necessarily all) available numbers and write one arithmetic expression.
- You may only use numbers currently in the available pool, each at most once.
- The numbers you use are REMOVED from the pool. The result is ADDED as a new available number.
- Division must be exact (no remainders). Parentheses are fine.
- You have a limited number of turns. Try to reach the target exactly.

Example:
  Target: 15, Available: [2, 3, 10]
  Turn 1 → Expression: 2 + 3   (result: 5, pool becomes [10, 5])
  Turn 2 → Expression: 10 + 5  (result: 15 ✓ target reached!)

Think step-by-step inside <think>...</think> tags, then write your final expression as:
Expression: <your expression>\
"""


class CountdownMessageEnv:
    """Message-level environment for the iterative Countdown numbers game.

    Each turn:
    1. The model proposes an expression using available numbers.
    2. The environment validates and evaluates it.
    3. Used numbers are removed and the result is added to available numbers.
    4. The episode ends when the target is reached, max turns exceeded, or
       the model fails to produce a valid expression.

    This class does NOT depend on tinker-cookbook and can be tested standalone.
    """

    def __init__(self, target: int, numbers: list[int], max_turns: int = 5):
        self.target = target
        self.initial_numbers = list(numbers)
        self.available_numbers = list(numbers)
        self.turn = 0
        self.max_turns = max_turns
        self.history: list[tuple[str, int, list[int]]] = []
        self.best_distance = abs(target)

    async def initial_observation(self) -> list[dict]:
        numbers_str = ", ".join(str(n) for n in self.initial_numbers)
        user_content = (
            f"Target: {self.target}\n"
            f"Available numbers: [{numbers_str}]\n"
            f"Turns remaining: {self.max_turns}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    async def step(self, message: dict) -> MessageStepResult:
        self.turn += 1
        text = _get_text_content(message)

        # Extract expression from model output
        expr = extract_expression(text)
        if expr is None:
            return self._make_error_result(
                "I couldn't find an expression in your response. "
                "Please write: Expression: <your expression>"
            )

        # Parse and validate the expression
        try:
            result, numbers_used = parse_expression(expr, self.available_numbers)
        except ValueError as e:
            return self._make_error_result(f"Invalid expression: {e}")

        # Update available numbers: remove used, add result
        new_available = list(self.available_numbers)
        for num in numbers_used:
            new_available.remove(num)
        new_available.append(result)
        self.available_numbers = new_available

        # Record history
        self.history.append((expr, result, numbers_used))

        # Calculate distance to target
        distance = abs(self.target - result)
        old_best = self.best_distance
        self.best_distance = min(self.best_distance, distance)

        # Calculate reward
        reward = 0.0
        target_reached = result == self.target
        metrics: dict[str, float] = {}

        if target_reached:
            reward = 1.0
            metrics["target_reached"] = 1.0
            metrics["turns_to_solve"] = float(self.turn)
        else:
            # Small reward for valid expression
            reward += 0.05
            # Bonus for improving proximity
            if self.best_distance < old_best:
                # Normalized improvement
                improvement = (old_best - self.best_distance) / max(self.target, 1)
                reward += min(improvement, 0.3)

        metrics["distance_to_target"] = float(distance)
        metrics["best_distance"] = float(self.best_distance)
        metrics["turn"] = float(self.turn)

        # Check episode termination
        episode_done = target_reached or self.turn >= self.max_turns

        if episode_done and not target_reached:
            metrics["max_turns_exceeded"] = 1.0

        # Build feedback message
        numbers_str = ", ".join(str(n) for n in self.available_numbers)
        if target_reached:
            feedback = (
                f"Result: {result}. That's the target! Well done!\n"
                f"You solved it in {self.turn} turn(s)."
            )
        else:
            feedback = (
                f"Result: {result}. "
                f"Distance to target: {distance}.\n"
                f"Available numbers: [{numbers_str}]\n"
                f"Turns remaining: {self.max_turns - self.turn}"
            )
            if not episode_done:
                feedback += "\n\nMake another expression. Write: Expression: <expr>"

        # Build next_messages as the full conversation so far
        next_messages = await self.initial_observation()
        # Replay all turns
        for i, (hist_expr, hist_result, _) in enumerate(self.history):
            # Add the assistant message
            next_messages.append(
                {"role": "assistant", "content": f"Expression: {hist_expr}"}
            )
            # Add the environment feedback
            if i < len(self.history) - 1:
                # Previous turns: abbreviated feedback
                next_messages.append(
                    {"role": "user", "content": f"Result: {hist_result}. Continue."}
                )
            else:
                # Current turn: full feedback
                next_messages.append({"role": "user", "content": feedback})

        return MessageStepResult(
            reward=reward,
            episode_done=episode_done,
            next_messages=next_messages,
            metrics=metrics,
        )

    def _make_error_result(self, error_msg: str) -> MessageStepResult:
        """Create a result for an invalid expression."""
        episode_done = self.turn >= self.max_turns

        # Rebuild conversation with error feedback
        initial_numbers_str = ", ".join(str(n) for n in self.initial_numbers)
        current_numbers_str = ", ".join(str(n) for n in self.available_numbers)
        initial_msgs: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Target: {self.target}\n"
                    f"Available numbers: [{initial_numbers_str}]\n"
                    f"Turns remaining: {self.max_turns}"
                ),
            },
        ]

        # Replay previous turns
        next_messages_out = list(initial_msgs)
        for hist_expr, hist_result, _ in self.history:
            next_messages_out.append(
                {"role": "assistant", "content": f"Expression: {hist_expr}"}
            )
            next_messages_out.append(
                {"role": "user", "content": f"Result: {hist_result}. Continue."}
            )

        # Add error feedback for this turn
        error_feedback = (
            f"{error_msg}\n"
            f"Available numbers: [{current_numbers_str}]\n"
            f"Turns remaining: {self.max_turns - self.turn}"
        )
        if not episode_done:
            error_feedback += "\n\nTry again. Write: Expression: <expr>"

        next_messages_out.append({"role": "user", "content": error_feedback})

        return MessageStepResult(
            reward=-0.1,  # small penalty for invalid expression
            episode_done=episode_done,
            next_messages=next_messages_out,
            metrics={
                "invalid_expression": 1.0,
                "turn": float(self.turn),
                "distance_to_target": float(self.best_distance),
                "best_distance": float(self.best_distance),
            },
        )


# ---------------------------------------------------------------------------
# tinker-cookbook integration classes (require tinker-cookbook at runtime)
# ---------------------------------------------------------------------------


def _import_tinker_cookbook():
    """Lazy import of tinker-cookbook to avoid torch dependency at module level."""
    from tinker_cookbook.rl.message_env import EnvFromMessageEnv
    from tinker_cookbook.rl.types import (
        Env,
        EnvGroupBuilder,
        Metrics,
        RLDataset,
        RLDatasetBuilder,
        Trajectory,
    )
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    import chz
    return EnvFromMessageEnv, Env, EnvGroupBuilder, Metrics, RLDataset, RLDatasetBuilder, Trajectory, get_renderer, get_tokenizer, chz


# We define the tinker-cookbook classes using lazy imports so the module
# can be imported without torch for testing CountdownMessageEnv.
# The classes themselves will fail at instantiation if tinker-cookbook is missing.

try:
    import chz
    from tinker_cookbook.rl.message_env import EnvFromMessageEnv
    from tinker_cookbook.rl.types import (
        Env,
        EnvGroupBuilder,
        Metrics,
        RLDataset,
        RLDatasetBuilder,
        Trajectory,
    )
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    _HAS_TINKER_COOKBOOK = True
except ImportError:
    _HAS_TINKER_COOKBOOK = False


if _HAS_TINKER_COOKBOOK:
    @dataclass(frozen=True)
    class CountdownEnvGroupBuilder(EnvGroupBuilder):
        """Builds a group of Countdown environments for a single problem."""

        target: int
        numbers: tuple[int, ...]
        num_envs: int
        max_turns: int
        renderer_name: str
        model_name_for_tokenizer: str
        reward_type: str = "dense"
        max_trajectory_tokens: int | None = 4096

        async def make_envs(self) -> Sequence[Env]:
            tokenizer = get_tokenizer(self.model_name_for_tokenizer)
            renderer = get_renderer(self.renderer_name, tokenizer)
            envs = []
            for _ in range(self.num_envs):
                msg_env = CountdownMessageEnv(
                    self.target, list(self.numbers), self.max_turns
                )
                env = EnvFromMessageEnv(
                    renderer,
                    msg_env,
                    failed_parse_reward=-0.5,
                    terminate_on_parse_error=False,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                )
                envs.append(env)
            return envs

        async def compute_group_rewards(
            self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
        ) -> list[tuple[float, Metrics]]:
            results: list[tuple[float, Metrics]] = []

            for traj, env in zip(trajectory_group, env_group):
                metrics: Metrics = {}
                bonus = 0.0

                # Check if target was reached by looking at per-step metrics
                target_reached = any(
                    t.metrics.get("target_reached", 0.0) > 0 for t in traj.transitions
                )

                if self.reward_type == "binary":
                    step_total = sum(t.reward for t in traj.transitions)
                    if target_reached:
                        bonus = 1.0 - step_total
                    else:
                        bonus = 0.0 - step_total
                    metrics["binary_reward"] = 1.0 if target_reached else 0.0
                elif self.reward_type == "dense":
                    if target_reached:
                        bonus = 0.5
                    metrics["dense_bonus"] = bonus
                else:
                    pass

                metrics["target_reached"] = 1.0 if target_reached else 0.0
                results.append((bonus, metrics))

            return results

        def logging_tags(self) -> list[str]:
            return ["countdown"]

    @dataclass
    class CountdownDataset(RLDataset):
        """Dataset of Countdown problems."""

        problems: tuple[dict[str, Any], ...]
        batch_size: int
        group_size: int
        max_turns: int
        renderer_name: str
        model_name_for_tokenizer: str
        reward_type: str
        max_trajectory_tokens: int | None

        def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
            n = len(self.problems)
            start = (index * self.batch_size) % n
            batch_problems = [
                self.problems[(start + i) % n] for i in range(self.batch_size)
            ]
            return [
                CountdownEnvGroupBuilder(
                    target=p["target"],
                    numbers=tuple(p["numbers"]),
                    num_envs=self.group_size,
                    max_turns=self.max_turns,
                    renderer_name=self.renderer_name,
                    model_name_for_tokenizer=self.model_name_for_tokenizer,
                    reward_type=self.reward_type,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                )
                for p in batch_problems
            ]

        def __len__(self) -> int:
            return max(1, len(self.problems) // self.batch_size)

    @chz.chz
    class CountdownDatasetBuilder(RLDatasetBuilder):
        """Builder for Countdown RL datasets."""

        batch_size: int
        model_name_for_tokenizer: str
        renderer_name: str
        group_size: int
        max_turns: int = 5
        reward_type: str = "dense"
        train_problems_path: str | None = None
        eval_problems_path: str | None = None
        n_train: int = 500
        n_eval: int = 100
        seed: int = 42
        max_trajectory_tokens: int | None = 4096

        async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
            if self.train_problems_path and Path(self.train_problems_path).exists():
                with open(self.train_problems_path) as f:
                    train_problems = json.load(f)
            else:
                train_problems = generate_problems(self.n_train, seed=self.seed)

            if self.eval_problems_path and Path(self.eval_problems_path).exists():
                with open(self.eval_problems_path) as f:
                    eval_problems = json.load(f)
            else:
                eval_problems = generate_problems(self.n_eval, seed=self.seed + 1000)

            train_dataset = CountdownDataset(
                problems=tuple(train_problems),
                batch_size=self.batch_size,
                group_size=self.group_size,
                max_turns=self.max_turns,
                renderer_name=self.renderer_name,
                model_name_for_tokenizer=self.model_name_for_tokenizer,
                reward_type=self.reward_type,
                max_trajectory_tokens=self.max_trajectory_tokens,
            )

            eval_dataset = CountdownDataset(
                problems=tuple(eval_problems),
                batch_size=self.batch_size,
                group_size=max(1, self.group_size // 2),
                max_turns=self.max_turns,
                renderer_name=self.renderer_name,
                model_name_for_tokenizer=self.model_name_for_tokenizer,
                reward_type=self.reward_type,
                max_trajectory_tokens=self.max_trajectory_tokens,
            )

            return train_dataset, eval_dataset

else:
    # Standalone dataclasses when tinker-cookbook is not available.
    # Core logic (get_batch, __len__) works without tinker; only methods
    # that need the tinker runtime (make_envs, compute_group_rewards,
    # __call__) raise ImportError when invoked.

    @dataclass(frozen=True)
    class CountdownEnvGroupBuilder:  # type: ignore[no-redef]
        """Countdown env group builder (standalone, no tinker base class)."""

        target: int
        numbers: tuple[int, ...]
        num_envs: int
        max_turns: int
        renderer_name: str
        model_name_for_tokenizer: str
        reward_type: str = "dense"
        max_trajectory_tokens: int | None = 4096

        async def make_envs(self) -> Sequence[Any]:
            raise ImportError("make_envs requires tinker-cookbook and torch")

        async def compute_group_rewards(self, trajectory_group: list, env_group: Sequence) -> list:
            raise ImportError("compute_group_rewards requires tinker-cookbook and torch")

        def logging_tags(self) -> list[str]:
            return ["countdown"]

    @dataclass
    class CountdownDataset:  # type: ignore[no-redef]
        """Countdown dataset (standalone, no tinker base class)."""

        problems: tuple[dict[str, Any], ...]
        batch_size: int
        group_size: int
        max_turns: int
        renderer_name: str
        model_name_for_tokenizer: str
        reward_type: str
        max_trajectory_tokens: int | None

        def get_batch(self, index: int) -> Sequence[CountdownEnvGroupBuilder]:
            n = len(self.problems)
            start = (index * self.batch_size) % n
            batch_problems = [
                self.problems[(start + i) % n] for i in range(self.batch_size)
            ]
            return [
                CountdownEnvGroupBuilder(
                    target=p["target"],
                    numbers=tuple(p["numbers"]),
                    num_envs=self.group_size,
                    max_turns=self.max_turns,
                    renderer_name=self.renderer_name,
                    model_name_for_tokenizer=self.model_name_for_tokenizer,
                    reward_type=self.reward_type,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                )
                for p in batch_problems
            ]

        def __len__(self) -> int:
            return max(1, len(self.problems) // self.batch_size)

    class CountdownDatasetBuilder:  # type: ignore[no-redef]
        """Stub: requires tinker-cookbook (with torch) to be installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("CountdownDatasetBuilder requires tinker-cookbook and torch")
