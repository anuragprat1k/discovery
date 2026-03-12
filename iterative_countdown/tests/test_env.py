"""Tests for the Countdown message environment."""

from __future__ import annotations

import asyncio

import pytest

from iterative_countdown.environment.countdown_env import (
    CountdownMessageEnv,
    SYSTEM_PROMPT,
    MessageStepResult,
)


class TestCountdownMessageEnv:
    """Tests for CountdownMessageEnv."""

    def test_initial_observation(self):
        env = CountdownMessageEnv(target=100, numbers=[25, 4, 3, 2])
        messages = asyncio.run(env.initial_observation())

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Countdown" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "100" in messages[1]["content"]
        assert "25" in messages[1]["content"]

    def test_target_reached_in_one_turn(self):
        env = CountdownMessageEnv(target=100, numbers=[25, 4, 3, 2])
        asyncio.run(env.initial_observation())

        result = asyncio.run(
            env.step({"role": "assistant", "content": "Expression: 25 * 4"})
        )

        assert result.reward == 1.0
        assert result.episode_done is True
        assert result.metrics.get("target_reached") == 1.0
        assert result.metrics.get("turns_to_solve") == 1.0

    def test_multi_turn_episode(self):
        # Target: 103 = (25 * 4) + 3
        env = CountdownMessageEnv(target=103, numbers=[25, 4, 3, 2])
        asyncio.run(env.initial_observation())

        # Turn 1: 25 * 4 = 100
        result1 = asyncio.run(
            env.step({"role": "assistant", "content": "Expression: 25 * 4"})
        )
        assert not result1.episode_done
        assert result1.reward > 0  # valid expression + proximity improvement
        assert 100 in env.available_numbers
        assert 25 not in env.available_numbers
        assert 4 not in env.available_numbers
        # 3 and 2 should still be available
        assert 3 in env.available_numbers
        assert 2 in env.available_numbers

        # Turn 2: 100 + 3 = 103 (target!)
        result2 = asyncio.run(
            env.step({"role": "assistant", "content": "Expression: 100 + 3"})
        )
        assert result2.episode_done is True
        assert result2.reward == 1.0
        assert result2.metrics.get("target_reached") == 1.0
        assert result2.metrics.get("turns_to_solve") == 2.0

    def test_max_turns_exceeded(self):
        env = CountdownMessageEnv(target=999, numbers=[1, 2, 3], max_turns=2)
        asyncio.run(env.initial_observation())

        # Turn 1
        result1 = asyncio.run(
            env.step({"role": "assistant", "content": "Expression: 1 + 2"})
        )
        assert not result1.episode_done

        # Turn 2 (max_turns=2, so this ends the episode)
        result2 = asyncio.run(
            env.step({"role": "assistant", "content": "Expression: 3 + 3"})
        )
        assert result2.episode_done is True
        assert result2.metrics.get("max_turns_exceeded") == 1.0

    def test_invalid_expression_handling(self):
        env = CountdownMessageEnv(target=100, numbers=[25, 4, 3, 2], max_turns=3)
        asyncio.run(env.initial_observation())

        # Invalid: number not available
        result = asyncio.run(
            env.step({"role": "assistant", "content": "Expression: 99 + 1"})
        )
        assert result.reward < 0  # penalty
        assert result.metrics.get("invalid_expression") == 1.0
        # Episode should continue (turn 1 of 3)
        assert not result.episode_done

    def test_no_expression_found(self):
        env = CountdownMessageEnv(target=100, numbers=[25, 4, 3, 2], max_turns=3)
        asyncio.run(env.initial_observation())

        result = asyncio.run(
            env.step({"role": "assistant", "content": "I'm not sure what to do."})
        )
        assert result.reward < 0
        assert result.metrics.get("invalid_expression") == 1.0

    def test_available_numbers_update(self):
        env = CountdownMessageEnv(target=200, numbers=[25, 50, 75, 100])
        asyncio.run(env.initial_observation())

        # Use 25 and 75 -> result 100
        asyncio.run(
            env.step({"role": "assistant", "content": "Expression: 25 + 75"})
        )

        # 25 and 75 removed, 100 (result) added
        # So available should be: [50, 100, 100]
        assert sorted(env.available_numbers) == [50, 100, 100]

    def test_conversation_history_grows(self):
        env = CountdownMessageEnv(target=200, numbers=[25, 50, 75, 100])
        asyncio.run(env.initial_observation())

        result1 = asyncio.run(
            env.step({"role": "assistant", "content": "Expression: 25 + 75"})
        )
        # next_messages should contain: system, user (initial), assistant, user (feedback)
        assert len(result1.next_messages) == 4
        assert result1.next_messages[0]["role"] == "system"
        assert result1.next_messages[1]["role"] == "user"
        assert result1.next_messages[2]["role"] == "assistant"
        assert result1.next_messages[3]["role"] == "user"

        result2 = asyncio.run(
            env.step({"role": "assistant", "content": "Expression: 100 + 100"})
        )
        # Should have 6 messages: system, user, assistant1, user1, assistant2, user2
        assert len(result2.next_messages) == 6
        assert result2.episode_done is True  # 200 == target

    def test_proximity_reward(self):
        """Getting closer to target should yield positive reward."""
        env = CountdownMessageEnv(target=100, numbers=[90, 5, 3, 2])
        asyncio.run(env.initial_observation())

        # 90 + 5 = 95, distance = 5 (much closer than initial 100)
        result = asyncio.run(
            env.step({"role": "assistant", "content": "Expression: 90 + 5"})
        )
        # Should get base reward (0.05) + proximity improvement
        assert result.reward > 0.05
        assert result.metrics["distance_to_target"] == 5.0
