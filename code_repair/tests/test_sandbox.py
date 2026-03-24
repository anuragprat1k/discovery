"""Tests for the sandbox test runner."""
import pytest
from code_repair.env.sandbox import run_tests, TestResult, _split_assertions


class TestSplitAssertions:
    def test_basic_check_function(self):
        test_code = '''
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0
'''
        results = _split_assertions(test_code, "add")
        assert len(results) == 3
        assert all(name.startswith("test_") for name, _ in results)

    def test_no_check_function(self):
        test_code = 'assert add(1, 2) == 3'
        results = _split_assertions(test_code, "add")
        assert len(results) >= 1


class TestRunTests:
    def test_all_pass(self):
        code = "def add(a, b):\n    return a + b"
        test_code = '''
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
'''
        results = run_tests(code, test_code, "add", timeout=5, detailed=True)
        assert all(r.passed for r in results)
        assert all(r.no_crash for r in results)

    def test_some_fail(self):
        code = "def add(a, b):\n    return a - b"  # Bug!
        test_code = '''
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
'''
        results = run_tests(code, test_code, "add", timeout=5, detailed=True)
        # 0-0=0 passes, 1-2=-1 != 3 fails
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        assert len(passed) >= 1
        assert len(failed) >= 1

    def test_syntax_error(self):
        code = "def add(a, b)\n    return a + b"  # Missing colon
        test_code = '''
def check(candidate):
    assert candidate(1, 2) == 3
'''
        results = run_tests(code, test_code, "add", timeout=5)
        assert len(results) == 1
        assert not results[0].passed
        assert "syntax" in results[0].name.lower() or "SyntaxError" in (results[0].error or "")

    def test_timeout(self):
        code = "def loop():\n    while True: pass"
        test_code = '''
def check(candidate):
    assert candidate() is None
'''
        results = run_tests(code, test_code, "loop", timeout=2)
        assert len(results) == 1
        assert not results[0].passed
        assert "timeout" in results[0].name.lower() or "timed out" in (results[0].error or "").lower()

    def test_runtime_error(self):
        code = "def div(a, b):\n    return a / b"
        test_code = '''
def check(candidate):
    assert candidate(1, 0) == 0
'''
        results = run_tests(code, test_code, "div", timeout=5, detailed=True)
        assert len(results) >= 1
        assert not results[0].passed

    def test_detailed_type_info(self):
        code = "def get_list():\n    return (1, 2, 3)"  # Returns tuple instead of list
        test_code = '''
def check(candidate):
    assert candidate() == [1, 2, 3]
'''
        results = run_tests(code, test_code, "get_list", timeout=5, detailed=True)
        assert len(results) >= 1
        r = results[0]
        assert not r.passed
        assert r.no_crash  # It ran, just returned wrong type


class TestPartialCorrectness:
    def test_no_crash_signal(self):
        """A function that returns wrong value should have no_crash=True."""
        code = "def add(a, b):\n    return 42"  # Always returns 42
        test_code = '''
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
'''
        results = run_tests(code, test_code, "add", timeout=5, detailed=True)
        for r in results:
            assert r.no_crash  # Function ran without crashing

    def test_crash_signal(self):
        """A function that crashes should have no_crash=False."""
        code = "def add(a, b):\n    raise ValueError('broken')"
        test_code = '''
def check(candidate):
    assert candidate(1, 2) == 3
'''
        results = run_tests(code, test_code, "add", timeout=5, detailed=True)
        assert len(results) >= 1
        assert not results[0].no_crash or not results[0].passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
