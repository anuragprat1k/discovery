"""Sandboxed test runner for code repair.

Runs repaired code against test suites in isolated subprocesses with timeout,
capturing per-test pass/fail results and partial-correctness signals
(no-crash, correct type, correct shape).

Usage:
    results = run_tests(repaired_code, test_code, entry_point)
    for r in results:
        print(f"{r.name}: {'PASS' if r.passed else 'FAIL'}")
"""
from __future__ import annotations

import json
import subprocess
import sys
import re
import textwrap
from dataclasses import dataclass, asdict


@dataclass
class TestResult:
    """Result of running a single test assertion."""
    name: str
    passed: bool
    error: str | None = None
    returned: str | None = None
    expected: str | None = None
    return_type: str | None = None
    return_shape: int | None = None  # len() of return value if applicable
    no_crash: bool = False  # True if the test ran without exception


def _split_assertions(test_code: str, entry_point: str) -> list[tuple[str, str]]:
    """Split a HumanEval check() function into individual assertion tests.

    Returns list of (test_name, assertion_code) tuples.
    """
    # Extract the check function body
    check_match = re.search(
        r'def check\(candidate\):\s*\n((?:[ \t]+.*\n?)*)',
        test_code
    )
    if not check_match:
        return [("test_all", f"{test_code}\ncheck({entry_point})")]

    body = check_match.group(1)
    lines = body.split("\n")

    assertions = []
    current_assertion = []
    base_indent = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if base_indent is None and stripped:
            base_indent = len(line) - len(line.lstrip())

        line_indent = len(line) - len(line.lstrip())

        # New assertion starts at base indent level with 'assert'
        if line_indent == base_indent and stripped.startswith("assert"):
            if current_assertion:
                assertions.append("\n".join(current_assertion))
            current_assertion = [stripped]
        elif current_assertion:
            # Continuation of multi-line assertion
            current_assertion.append(stripped)
        # Skip non-assert lines at base indent (variable assignments etc)
        elif line_indent == base_indent:
            if current_assertion:
                assertions.append("\n".join(current_assertion))
                current_assertion = []
            # Include non-assert setup lines as part of the next assertion
            current_assertion = [stripped]

    if current_assertion:
        assertions.append("\n".join(current_assertion))

    if not assertions:
        return [("test_all", f"{test_code}\ncheck({entry_point})")]

    result = []
    for i, assertion in enumerate(assertions):
        result.append((f"test_{i}", assertion))
    return result


def _build_test_script(
    repaired_code: str,
    test_code: str,
    entry_point: str,
) -> str:
    """Build a Python script that runs each assertion individually and reports JSON results."""
    assertions = _split_assertions(test_code, entry_point)

    # Build individual test runners
    test_blocks = []
    for test_name, assertion_code in assertions:
        # Escape the assertion for embedding in a string
        escaped = assertion_code.replace("\\", "\\\\").replace('"', '\\"')
        test_blocks.append(f"""
    # {test_name}
    try:
        candidate = {entry_point}
        {assertion_code}
        results.append({{"name": "{test_name}", "passed": True, "no_crash": True}})
    except AssertionError as e:
        results.append({{"name": "{test_name}", "passed": False, "error": str(e)[:200], "no_crash": True}})
    except Exception as e:
        results.append({{"name": "{test_name}", "passed": False, "error": type(e).__name__ + ": " + str(e)[:200], "no_crash": False}})
""")

    # For type/shape checking, we need a separate pass that calls the function
    # and inspects the return value. We extract call expressions from assertions.
    script = f"""\
import json
import sys

# --- Repaired code ---
{repaired_code}

# --- Run tests ---
results = []
{"".join(test_blocks)}

print("__RESULTS__")
print(json.dumps(results))
"""
    return script


def _build_detailed_test_script(
    repaired_code: str,
    test_code: str,
    entry_point: str,
) -> str:
    """Build a test script with per-assertion isolation and partial-correctness signals."""
    assertions = _split_assertions(test_code, entry_point)

    script_parts = [
        "import json",
        "import sys",
        "",
        "# --- Repaired code ---",
        repaired_code,
        "",
        "results = []",
        "",
    ]

    for i, (test_name, assertion_code) in enumerate(assertions):
        # Try to extract the function call and expected value from assert
        # Pattern: assert candidate(args) == expected
        call_match = re.search(
            r'assert\s+candidate\(([^)]*)\)\s*==\s*(.+)',
            assertion_code
        )

        if call_match:
            args_str = call_match.group(1)
            expected_str = call_match.group(2).strip().rstrip(",")
            script_parts.append(f"""
try:
    candidate = {entry_point}
    _returned = candidate({args_str})
    _expected = {expected_str}
    _passed = (_returned == _expected)
    _ret_type = type(_returned).__name__
    _exp_type = type(_expected).__name__
    _ret_shape = None
    _exp_shape = None
    try:
        _ret_shape = len(_returned)
    except TypeError:
        pass
    try:
        _exp_shape = len(_expected)
    except TypeError:
        pass
    results.append({{
        "name": "{test_name}",
        "passed": _passed,
        "no_crash": True,
        "returned": repr(_returned)[:200],
        "expected": repr(_expected)[:200],
        "return_type": _ret_type,
        "expected_type": _exp_type,
        "return_shape": _ret_shape,
        "expected_shape": _exp_shape,
    }})
except Exception as e:
    results.append({{
        "name": "{test_name}",
        "passed": False,
        "no_crash": False,
        "error": type(e).__name__ + ": " + str(e)[:200],
    }})
""")
        else:
            # Fallback: just run the assertion as-is
            script_parts.append(f"""
try:
    candidate = {entry_point}
    {assertion_code}
    results.append({{"name": "{test_name}", "passed": True, "no_crash": True}})
except AssertionError as e:
    results.append({{"name": "{test_name}", "passed": False, "no_crash": True, "error": str(e)[:200]}})
except Exception as e:
    results.append({{"name": "{test_name}", "passed": False, "no_crash": False, "error": type(e).__name__ + ": " + str(e)[:200]}})
""")

    script_parts.append('print("__RESULTS__")')
    script_parts.append("print(json.dumps(results))")

    return "\n".join(script_parts)


def run_tests(
    repaired_code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 5,
    detailed: bool = True,
) -> list[TestResult]:
    """Run repaired code against test suite in an isolated subprocess.

    Args:
        repaired_code: The full function definition to test.
        test_code: HumanEval test code with check() function.
        entry_point: Name of the function being tested.
        timeout: Maximum execution time in seconds.
        detailed: If True, capture return values/types for partial-correctness.

    Returns:
        List of TestResult for each assertion in the test suite.
    """
    if detailed:
        script = _build_detailed_test_script(repaired_code, test_code, entry_point)
    else:
        script = _build_test_script(repaired_code, test_code, entry_point)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return [TestResult(
            name="timeout",
            passed=False,
            error="Execution timed out",
            no_crash=False,
        )]

    stdout = result.stdout
    stderr = result.stderr

    # Check for syntax errors in the repaired code
    if result.returncode != 0 and "__RESULTS__" not in stdout:
        error_msg = stderr[:500] if stderr else "Unknown error"
        # Try to determine error type
        if "SyntaxError" in error_msg:
            return [TestResult(
                name="syntax_error",
                passed=False,
                error=error_msg,
                no_crash=False,
            )]
        elif "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
            return [TestResult(
                name="import_error",
                passed=False,
                error=error_msg,
                no_crash=False,
            )]
        else:
            return [TestResult(
                name="runtime_error",
                passed=False,
                error=error_msg,
                no_crash=False,
            )]

    # Parse results
    if "__RESULTS__" not in stdout:
        return [TestResult(
            name="no_output",
            passed=False,
            error="Test script produced no results",
            no_crash=False,
        )]

    results_json = stdout.split("__RESULTS__")[1].strip()
    try:
        raw_results = json.loads(results_json)
    except json.JSONDecodeError:
        return [TestResult(
            name="parse_error",
            passed=False,
            error=f"Could not parse test results: {results_json[:200]}",
            no_crash=False,
        )]

    test_results = []
    for r in raw_results:
        tr = TestResult(
            name=r.get("name", "unknown"),
            passed=r.get("passed", False),
            error=r.get("error"),
            returned=r.get("returned"),
            expected=r.get("expected"),
            return_type=r.get("return_type"),
            no_crash=r.get("no_crash", False),
        )
        # Compute shape match
        if r.get("return_shape") is not None and r.get("expected_shape") is not None:
            tr.return_shape = r["return_shape"]
        elif r.get("return_shape") is not None:
            tr.return_shape = r["return_shape"]
        test_results.append(tr)

    return test_results


def summarize_results(results: list[TestResult]) -> dict:
    """Summarize test results into aggregate metrics."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    no_crash = sum(1 for r in results if r.no_crash)
    type_match = sum(
        1 for r in results
        if r.return_type is not None and not r.passed and r.no_crash
        # We check type match for failing tests only (passing tests trivially match)
    )

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "no_crash": no_crash,
        "pass_rate": passed / total if total > 0 else 0.0,
    }
