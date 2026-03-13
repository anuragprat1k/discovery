"""Robust arithmetic expression parser for the Countdown numbers game.

Parses expressions like "25 * 4", "(100 + 50) * 2", "75 - 25" using the ast
module for safe evaluation. Validates that only available numbers are used and
each number is consumed at most once. Division must produce exact integers.
"""

from __future__ import annotations

import ast
import re
from collections import Counter


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and special tokens from model output.

    Works whether the tags are present as literal text (skip_special_tokens=False)
    or stripped by the tokenizer (skip_special_tokens=True). When tags are stripped,
    we can't recover the boundary, so callers should decode with
    skip_special_tokens=False for best results.
    """
    # Remove <think>...</think> blocks (handles multiple)
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # Remove unclosed <think> block (model hit token limit mid-thinking)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    # Remove any leftover special tokens from Qwen chat format
    text = re.sub(r'<\|[^>]+\|>', '', text)
    return text.strip()


def parse_expression(expr: str, available: list[int]) -> tuple[int, list[int]]:
    """Parse and evaluate an arithmetic expression using available numbers.

    Args:
        expr: An arithmetic expression string (e.g., "25 * 4", "(100 + 50) * 2").
        available: List of integers currently available to use.

    Returns:
        A tuple of (result, numbers_used) where result is the integer outcome
        and numbers_used is the list of numbers consumed from available.

    Raises:
        ValueError: If the expression is invalid, uses unavailable numbers,
            uses a number more times than it appears in available, involves
            non-integer division, or contains disallowed operations.
    """
    expr = expr.strip()
    if not expr:
        raise ValueError("Empty expression")

    # Parse the expression into an AST
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in expression: {e}") from e

    # Extract all integer literals from the expression
    numbers_used: list[int] = []
    _collect_numbers(tree.body, numbers_used)

    # Validate that all numbers are available (respecting multiplicity)
    available_counts = Counter(available)
    used_counts = Counter(numbers_used)
    for num, count in used_counts.items():
        if count > available_counts.get(num, 0):
            if num not in available_counts:
                raise ValueError(
                    f"Number {num} is not available. Available: {available}"
                )
            raise ValueError(
                f"Number {num} used {count} time(s) but only available "
                f"{available_counts[num]} time(s)"
            )

    # Must use at least one number
    if not numbers_used:
        raise ValueError("Expression must use at least one number")

    # Evaluate safely
    result = _safe_eval(tree.body)

    # Result must be a non-negative integer (standard Countdown rule: intermediates > 0)
    if not isinstance(result, int):
        raise ValueError(f"Result is not an integer: {result}")

    return result, numbers_used


def _collect_numbers(node: ast.expr, numbers: list[int]) -> None:
    """Recursively collect all integer literals from an AST node."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            raise ValueError(f"Only integers are allowed, got: {node.value!r}")
        if node.value < 0:
            raise ValueError(f"Negative literals are not allowed: {node.value}")
        numbers.append(node.value)
    elif isinstance(node, ast.BinOp):
        _collect_numbers(node.left, numbers)
        _collect_numbers(node.right, numbers)
    elif isinstance(node, ast.UnaryOp):
        # Allow unary minus for expressions like -(25 - 50) but not negative literals
        if isinstance(node.op, ast.USub):
            _collect_numbers(node.operand, numbers)
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    else:
        raise ValueError(
            f"Unsupported expression element: {type(node).__name__}. "
            "Only numbers and +, -, *, / are allowed."
        )


def _safe_eval(node: ast.expr) -> int:
    """Safely evaluate an AST expression node, allowing only +, -, *, /."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            raise ValueError(f"Only integers are allowed, got: {node.value!r}")
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("Division by zero")
            if left % right != 0:
                raise ValueError(
                    f"Division {left} / {right} does not produce an integer"
                )
            return left // right
        elif isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ValueError("Division by zero")
            if left % right != 0:
                raise ValueError(
                    f"Division {left} // {right} does not produce an integer"
                )
            return left // right
        else:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_safe_eval(node.operand)
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    else:
        raise ValueError(f"Unsupported expression: {type(node).__name__}")


def extract_expression(text: str) -> str | None:
    """Extract an arithmetic expression from model output text.

    Looks for patterns in the following order of priority:
    1. "Expression: <expr>" on any line
    2. A line containing "=" where the left side has arithmetic
    3. The last line that contains arithmetic operators with numbers

    Args:
        text: The full text output from the model.

    Returns:
        The extracted expression string, or None if no expression found.
    """
    text = text.strip()
    if not text:
        return None

    # Pattern 1: "Expression: <expr>" — take the LAST match (safety net for think-tag leaks)
    matches = list(re.finditer(r"[Ee]xpression:\s*(.+)", text))
    if matches:
        match = matches[-1]
        expr = match.group(1).strip()
        # Remove trailing punctuation or explanation
        expr = re.split(r"\s*[=;,]?\s*$", expr)[0].strip()
        # If the expression contains '=', take the left side
        if "=" in expr:
            expr = expr.split("=")[0].strip()
        if expr:
            return expr

    # Pattern 2: Line with "=" where left side has arithmetic
    for line in text.splitlines():
        line = line.strip()
        if "=" in line:
            lhs = line.split("=")[0].strip()
            if re.search(r"\d+\s*[+\-*/]\s*\d+", lhs):
                return lhs

    # Pattern 3: Last line with arithmetic (numbers and operators)
    arithmetic_pattern = re.compile(r"^[\d\s+\-*/()]+$")
    # Also match lines that have arithmetic embedded
    arith_line_pattern = re.compile(r"(\d+(?:\s*[+\-*/]\s*[\d()]+)+(?:\s*[+\-*/]\s*\d+)*)")

    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        # Check if entire line is arithmetic
        if arithmetic_pattern.match(line) and re.search(r"\d", line) and re.search(r"[+\-*/]", line):
            return line
        # Check for embedded arithmetic
        match = arith_line_pattern.search(line)
        if match:
            return match.group(1).strip()

    return None
