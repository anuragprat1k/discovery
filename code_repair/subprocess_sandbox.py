"""Simple subprocess-based sandbox for LCB code execution.

No Docker, no Modal — just subprocess.run() with timeout.
Works anywhere Python is installed. Less secure but fine for
LeetCode-style problems in a training environment.

Supports both stdin/stdout and functional (class method) test types.
Runs ALL tests independently (no short-circuit on first failure).
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


async def check_correctness(
    tests: list[dict[str, Any]],
    code: str,
    timeout: int = 6,
) -> tuple[bool, dict[str, Any]]:
    """Run code against test cases in an isolated subprocess.

    Args:
        tests: List of test dicts with input/output/testtype/metadata
        code: Generated Python code to test
        timeout: Per-test timeout in seconds

    Returns:
        (all_passed, details) where details has tests_passed, tests_total, per_test, errors
    """
    # Determine test type
    fn_name = None
    if tests and tests[0].get("testtype") == "functional":
        fn_name = tests[0].get("metadata", {}).get("func_name")

    # Build test runner script
    script = _build_runner(code, tests, fn_name, timeout)

    # Run in async subprocess (non-blocking, enables true parallelism)
    total_timeout = (timeout + 2) * len(tests) + 10
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=total_timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, {
                "tests_passed": 0, "tests_total": len(tests),
                "per_test": [False] * len(tests),
                "errors": ["Global timeout"],
                "stdout": "", "stderr": "Global timeout",
            }
        stdout = stdout_bytes.decode().strip()
        stderr = stderr_bytes.decode().strip()
    except Exception as e:
        return False, {
            "tests_passed": 0, "tests_total": len(tests),
            "per_test": [False] * len(tests),
            "errors": [str(e)],
            "stdout": "", "stderr": str(e),
        }
    finally:
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass

    # Parse results from stdout — find the JSON summary line
    if stdout:
        try:
            # Find the line that starts with {"passed":
            summary = None
            summary_idx = 0
            for idx, line in enumerate(stdout.split("\n")):
                line = line.strip()
                if line.startswith('{"passed"'):
                    summary = json.loads(line)
                    summary_idx = idx
                    break
            if summary is None:
                raise ValueError("No summary JSON found")
            tp = summary.get("passed", 0)
            tt = summary.get("total", len(tests))
            per_test = summary.get("per_test", [])
            errors = summary.get("errors", [])
            # Remaining lines are error details
            error_lines = stdout.split("\n")[summary_idx + 1:]
            for line in error_lines[:10]:
                try:
                    err = json.loads(line)
                    if isinstance(err, dict):
                        errors.append(err)
                except:
                    pass
            return tp == tt and tt > 0, {
                "tests_passed": tp, "tests_total": tt,
                "per_test": per_test, "errors": errors,
                "stdout": stdout, "stderr": stderr,
            }
        except (json.JSONDecodeError, IndexError):
            pass

    # Fallback: couldn't parse
    return False, {
        "tests_passed": 0, "tests_total": len(tests),
        "per_test": [False] * len(tests),
        "errors": [stderr[:300] if stderr else "No output"],
        "stdout": stdout, "stderr": stderr,
    }


def _build_runner(code: str, tests: list[dict], fn_name: str | None, timeout: int) -> str:
    """Build a Python script that runs each test independently."""
    parts = [
        "import json, sys, signal, io",
        "signal.alarm = lambda x: None  # noop if no SIGALRM",
        "",
        "# Common imports",
        "from typing import *",
        "from collections import *",
        "from itertools import *",
        "from functools import *",
        "from heapq import *",
        "from bisect import *",
        "from math import *",
        "from copy import *",
        "from operator import *",
        "from string import *",
        "import collections, itertools, functools, heapq, bisect, math, re, sys, copy, operator, string",
        "try:\n    import numpy\nexcept ImportError: pass",
        "try:\n    from sortedcontainers import SortedList, SortedDict, SortedSet\nexcept ImportError: pass",
        "sys.setrecursionlimit(10**5)",
        "",
        "# User code",
        code,
        "",
        "results = []",
        "errors = []",
    ]

    if fn_name:
        # Functional tests (class method calls)
        parts.append(f"""
try:
    if 'class Solution' in '''{code}''':
        _obj = Solution()
    else:
        _obj = type('_', (), {{}})()
        _obj.{fn_name} = {fn_name}
except Exception as e:
    # Can't instantiate — all tests fail
    print(json.dumps({{"passed": 0, "total": {len(tests)}, "per_test": [False]*{len(tests)}, "errors": [str(e)[:200]]}}))
    sys.exit(0)
""")
        for i, test in enumerate(tests):
            inp = test.get("input", "")
            expected = test.get("output", "")
            parts.append(f"""
try:
    _inputs = [json.loads(line) for line in {repr(inp)}.split("\\n")]
    _expected = json.loads({repr(expected)})
    _output = _obj.{fn_name}(*_inputs)
    if isinstance(_output, tuple): _output = list(_output)
    _passed = (_output == _expected)
    if not _passed and isinstance(_expected, list) and _expected:
        _passed = (_output == _expected[0])
    results.append(_passed)
    if not _passed:
        errors.append(json.dumps({{"inputs": {repr(inp[:200])}, "expected": str(_expected)[:200], "output": str(_output)[:200], "error_message": "Wrong Answer"}}))
except Exception as e:
    results.append(False)
    errors.append(json.dumps({{"inputs": {repr(inp[:100])}, "error": str(e)[:200], "error_message": "Runtime Error"}}))
""")
    else:
        # stdin/stdout tests
        for i, test in enumerate(tests):
            inp = test.get("input", "")
            expected = test.get("output", "")
            parts.append(f"""
try:
    import subprocess as _sp
    _r = _sp.run([sys.executable, "-c", {repr(code)}],
                 input={repr(inp)}, capture_output=True, text=True, timeout={timeout})
    _output = _r.stdout
    _passed = _output.strip() == {repr(expected)}.strip()
    results.append(_passed)
    if not _passed:
        errors.append(json.dumps({{"inputs": {repr(inp[:200])}, "expected": {repr(expected[:200])}, "output": _output[:200], "error_message": "Wrong Answer"}}))
except _sp.TimeoutExpired:
    results.append(False)
    errors.append(json.dumps({{"inputs": {repr(inp[:100])}, "error_message": "Time Limit Exceeded"}}))
except Exception as e:
    results.append(False)
    errors.append(json.dumps({{"inputs": {repr(inp[:100])}, "error": str(e)[:200], "error_message": "Runtime Error"}}))
""")

    parts.append("""
passed = sum(1 for x in results if x)
total = len(results)
print(json.dumps({"passed": passed, "total": total, "per_test": results, "errors": []}))
for err in errors[:10]:
    print(err)
""")

    return "\n".join(parts)
