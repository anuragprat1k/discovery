"""Tests for bug injection pipeline."""
import random
import pytest
from code_repair.data.bug_injection import (
    _inject_off_by_one,
    _inject_wrong_comparison,
    _inject_wrong_operator,
    _inject_wrong_init,
    _inject_missing_return,
    _inject_edge_case_removal,
    inject_bug,
)


class TestOffByOne:
    def test_modifies_range(self):
        source = "for i in range(n):\n    total += i"
        rng = random.Random(42)
        result = _inject_off_by_one(source, rng)
        assert result is not None
        assert result != source
        assert "range(" in result
        assert ("n - 1" in result or "n + 1" in result)

    def test_no_range_returns_none(self):
        source = "x = 5\ny = x + 1"
        result = _inject_off_by_one(source, random.Random(42))
        assert result is None


class TestWrongComparison:
    def test_swaps_operator(self):
        source = "if x <= 5:\n    return True"
        rng = random.Random(42)
        result = _inject_wrong_comparison(source, rng)
        assert result is not None
        assert result != source

    def test_no_comparison_returns_none(self):
        source = "x = 5"
        result = _inject_wrong_comparison(source, random.Random(42))
        assert result is None


class TestWrongOperator:
    def test_swaps_arithmetic(self):
        source = "return a + b"
        rng = random.Random(42)
        result = _inject_wrong_operator(source, rng)
        assert result is not None
        assert result != source


class TestWrongInit:
    def test_changes_initial_value(self):
        source = "total = 0\nfor i in items:\n    total += i"
        rng = random.Random(42)
        result = _inject_wrong_init(source, rng)
        assert result is not None
        assert "= 1" in result or "= 0" not in result


class TestMissingReturn:
    def test_needs_multiple_returns(self):
        source = "def f(x):\n    return x"
        result = _inject_missing_return(source, random.Random(42))
        assert result is None  # Only 1 return, won't delete

    def test_removes_one_return(self):
        source = "def f(x):\n    if x > 0:\n        return x\n    return -x"
        rng = random.Random(42)
        result = _inject_missing_return(source, rng)
        assert result is not None
        assert result.count("return") < source.count("return")


class TestInjectBug:
    def test_simple_function(self):
        prompt = "def add(a, b):\n"
        canonical = "    return a + b\n"
        test_code = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0
"""
        result = inject_bug(prompt, canonical, test_code, "add", random.Random(42))
        if result is not None:
            assert "buggy_code" in result
            assert "bug_type" in result
            assert result["buggy_code"] != prompt + canonical


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
