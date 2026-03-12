"""Tests for the Countdown expression parser."""

from __future__ import annotations

import pytest

from iterative_countdown.environment.expression_parser import (
    extract_expression,
    parse_expression,
)


class TestParseExpression:
    """Tests for parse_expression."""

    def test_simple_addition(self):
        result, used = parse_expression("25 + 50", [25, 50, 75])
        assert result == 75
        assert sorted(used) == [25, 50]

    def test_simple_subtraction(self):
        result, used = parse_expression("100 - 25", [25, 50, 75, 100])
        assert result == 75
        assert sorted(used) == [25, 100]

    def test_simple_multiplication(self):
        result, used = parse_expression("25 * 4", [25, 4, 3, 2])
        assert result == 100
        assert sorted(used) == [4, 25]

    def test_simple_division(self):
        result, used = parse_expression("100 / 25", [100, 25, 50])
        assert result == 4
        assert sorted(used) == [25, 100]

    def test_parenthesized_expression(self):
        result, used = parse_expression("(100 + 50) * 2", [100, 50, 2, 3])
        assert result == 300
        assert sorted(used) == [2, 50, 100]

    def test_complex_expression(self):
        result, used = parse_expression("(75 - 25) * 4 + 3", [75, 25, 4, 3])
        assert result == 203
        assert sorted(used) == [3, 4, 25, 75]

    def test_floor_div_operator(self):
        result, used = parse_expression("100 // 25", [100, 25])
        assert result == 4
        assert sorted(used) == [25, 100]

    def test_single_number(self):
        result, used = parse_expression("42", [42, 10, 20])
        assert result == 42
        assert used == [42]

    def test_duplicate_available_numbers(self):
        """If a number appears multiple times in available, it can be used multiple times."""
        result, used = parse_expression("5 + 5", [5, 5, 10])
        assert result == 10
        assert used == [5, 5]

    def test_negative_result(self):
        """Subtraction can produce negative results."""
        result, used = parse_expression("3 - 7", [3, 7])
        assert result == -4
        assert sorted(used) == [3, 7]

    # --- Error cases ---

    def test_unavailable_number(self):
        with pytest.raises(ValueError, match="not available"):
            parse_expression("99 + 1", [25, 50, 75])

    def test_number_used_too_many_times(self):
        with pytest.raises(ValueError, match="used 2 time"):
            parse_expression("5 + 5", [5, 10])  # only one 5 available

    def test_division_by_zero(self):
        with pytest.raises(ValueError, match="Division by zero"):
            parse_expression("25 / 0", [25, 0])

    def test_non_integer_division(self):
        with pytest.raises(ValueError, match="does not produce an integer"):
            parse_expression("7 / 3", [7, 3])

    def test_empty_expression(self):
        with pytest.raises(ValueError, match="Empty expression"):
            parse_expression("", [1, 2, 3])

    def test_syntax_error(self):
        with pytest.raises(ValueError, match="Syntax error"):
            parse_expression("25 +* 3", [25, 3])

    def test_disallowed_operation(self):
        """Exponentiation and other operators should be rejected."""
        with pytest.raises(ValueError, match="Unsupported"):
            parse_expression("2 ** 3", [2, 3])

    def test_float_literal(self):
        with pytest.raises(ValueError, match="Only integers"):
            parse_expression("2.5 + 3", [3])

    def test_string_in_expression(self):
        with pytest.raises(ValueError):
            parse_expression("'hello' + 3", [3])

    def test_function_call_rejected(self):
        with pytest.raises(ValueError):
            parse_expression("abs(-5)", [5])


class TestExtractExpression:
    """Tests for extract_expression."""

    def test_expression_tag(self):
        text = "Let me think about this.\nExpression: 25 * 4"
        assert extract_expression(text) == "25 * 4"

    def test_expression_tag_case_insensitive(self):
        text = "expression: 100 + 50"
        assert extract_expression(text) == "100 + 50"

    def test_expression_with_equals(self):
        text = "I'll calculate:\n25 * 4 = 100"
        assert extract_expression(text) == "25 * 4"

    def test_arithmetic_last_line(self):
        text = "I need to combine numbers.\n25 + 75"
        assert extract_expression(text) == "25 + 75"

    def test_embedded_arithmetic(self):
        text = "My calculation is 100 + 50 * 2 which should work."
        result = extract_expression(text)
        assert result is not None
        assert "100" in result and "50" in result

    def test_empty_input(self):
        assert extract_expression("") is None

    def test_no_arithmetic(self):
        assert extract_expression("I don't know what to do.") is None

    def test_expression_with_parentheses(self):
        text = "Expression: (75 + 25) * 4"
        assert extract_expression(text) == "(75 + 25) * 4"

    def test_expression_tag_with_trailing_equals(self):
        text = "Expression: 25 * 4 = 100"
        result = extract_expression(text)
        assert result == "25 * 4"

    def test_multiline_reasoning_then_expression(self):
        text = (
            "The target is 300.\n"
            "I can multiply 75 by 4 to get 300.\n"
            "Expression: 75 * 4"
        )
        assert extract_expression(text) == "75 * 4"
