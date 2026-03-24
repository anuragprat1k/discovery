"""Tests for code repair reward functions."""
import pytest
from code_repair.env.rewards import (
    sparse_reward,
    dense_passes_reward,
    dense_full_reward,
    TERMINAL_WIN,
    TERMINAL_LOSS,
    SPEED_BONUS,
    FORMAT_PENALTY,
    PASS_REWARD_PER_TEST,
)


def _make_info(
    turn=1, max_turns=4, num_tests=5,
    prev_passing=0, curr_passing=0,
    hw_passing=0, old_hw=0,
    all_passed=False,
    format_violation=False,
    no_crash_failing=0,
    type_match_failing=0,
    shape_match_failing=0,
):
    return {
        "turn": turn,
        "max_turns": max_turns,
        "num_tests": num_tests,
        "prev_passing": prev_passing,
        "curr_passing": curr_passing,
        "hw_passing": hw_passing,
        "old_hw": old_hw,
        "all_passed": all_passed,
        "format_violation": format_violation,
        "no_crash_failing": no_crash_failing,
        "type_match_failing": type_match_failing,
        "shape_match_failing": shape_match_failing,
    }


class TestSparseReward:
    def test_non_terminal_zero(self):
        info = _make_info(turn=1, curr_passing=2)
        r, m = sparse_reward(info, is_terminal=False, completion_text="<repair>x</repair>")
        assert r == 0.0

    def test_terminal_win(self):
        info = _make_info(turn=2, all_passed=True)
        r, m = sparse_reward(info, is_terminal=True)
        expected = TERMINAL_WIN + SPEED_BONUS * (4 - 2)
        assert r == pytest.approx(expected)

    def test_terminal_loss(self):
        info = _make_info(turn=4, all_passed=False)
        r, m = sparse_reward(info, is_terminal=True)
        assert r == pytest.approx(TERMINAL_LOSS)

    def test_format_penalty(self):
        info = _make_info(turn=1, format_violation=True)
        r, m = sparse_reward(info, is_terminal=False)
        assert r == pytest.approx(FORMAT_PENALTY)


class TestDensePassesReward:
    def test_hwm_increase(self):
        """Reward when high-water mark increases."""
        info = _make_info(num_tests=5, hw_passing=3, old_hw=1)
        r, m = dense_passes_reward(info, is_terminal=False, completion_text="<repair>x</repair>")
        expected = PASS_REWARD_PER_TEST * 2 / 5  # delta=2, normalized by 5 tests
        assert r == pytest.approx(expected)

    def test_no_hwm_change(self):
        """No reward when HWM doesn't increase (even if tests regress)."""
        info = _make_info(num_tests=5, hw_passing=3, old_hw=3, curr_passing=1)
        r, m = dense_passes_reward(info, is_terminal=False, completion_text="<repair>x</repair>")
        assert r == 0.0

    def test_potential_based(self):
        """Verify reward only depends on HWM delta (potential-based)."""
        # Same delta, different current passing
        info1 = _make_info(num_tests=5, hw_passing=3, old_hw=2, curr_passing=3)
        info2 = _make_info(num_tests=5, hw_passing=3, old_hw=2, curr_passing=1)
        r1, _ = dense_passes_reward(info1, is_terminal=False)
        r2, _ = dense_passes_reward(info2, is_terminal=False)
        assert r1 == pytest.approx(r2)  # Same HWM delta → same reward

    def test_terminal_adds_episode_reward(self):
        info = _make_info(turn=2, all_passed=True, hw_passing=5, old_hw=3, num_tests=5)
        r, m = dense_passes_reward(info, is_terminal=True)
        pass_r = PASS_REWARD_PER_TEST * 2 / 5
        ep_r = TERMINAL_WIN + SPEED_BONUS * 2
        assert r == pytest.approx(pass_r + ep_r)


class TestDenseFullReward:
    def test_includes_partial_correctness(self):
        """Dense full should include no-crash and type-match bonuses."""
        info = _make_info(
            num_tests=5, hw_passing=2, old_hw=2,
            no_crash_failing=2, type_match_failing=1, shape_match_failing=1,
        )
        r, m = dense_full_reward(info, is_terminal=False, completion_text="<repair>x</repair>")
        assert r > 0.0  # Should have partial-correctness reward
        assert m["partial_reward"] > 0.0

    def test_non_potential(self):
        """Partial-correctness can decrease (non-potential): regression is possible."""
        # Turn 1: function runs but returns wrong values (no_crash=3)
        info1 = _make_info(num_tests=5, hw_passing=0, old_hw=0, no_crash_failing=3)
        r1, _ = dense_full_reward(info1, is_terminal=False)

        # Turn 2: function now crashes on everything (no_crash=0)
        info2 = _make_info(num_tests=5, hw_passing=0, old_hw=0, no_crash_failing=0)
        r2, _ = dense_full_reward(info2, is_terminal=False)

        assert r1 > r2  # Regression possible — this is non-potential

    def test_superset_of_dense_passes(self):
        """Dense full ≥ dense passes (adds partial-correctness on top)."""
        info = _make_info(
            num_tests=5, hw_passing=3, old_hw=1,
            no_crash_failing=1, type_match_failing=1,
        )
        r_full, _ = dense_full_reward(info, is_terminal=False)
        r_passes, _ = dense_passes_reward(info, is_terminal=False)
        assert r_full >= r_passes


class TestFormatPenalty:
    def test_all_rewards_penalize_format_violation(self):
        info = _make_info(format_violation=True)
        for fn in [sparse_reward, dense_passes_reward, dense_full_reward]:
            r, m = fn(info, is_terminal=False)
            assert r == pytest.approx(FORMAT_PENALTY), f"{fn.__name__} should penalize format violation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
