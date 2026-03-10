"""
TRL-compatible reward functions for GRPO training on MATH problems.

Two reward functions:
  - compute_score_binary: 1.0 if final boxed answer matches ground_truth, else 0.0
  - compute_score_partial_credit: blends binary correctness with intermediate step matching
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content of the last \\boxed{...} in text, handling nested braces."""
    pattern = r'\\boxed\{'
    matches = [(m.start(), m.end()) for m in re.finditer(pattern, text)]
    if not matches:
        return None

    # Use the last match
    _, content_start = matches[-1]
    depth = 1
    i = content_start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth != 0:
        return None  # Unmatched braces

    raw = text[content_start:i - 1]
    return _normalize_answer(raw)


def _normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = s.strip()

    # Strip surrounding $ signs
    s = s.strip('$').strip()

    # Convert \frac{a}{b} -> a/b  (handle simple one-level fracs)
    frac_pattern = re.compile(r'\\frac\{([^{}]+)\}\{([^{}]+)\}')
    s = frac_pattern.sub(lambda m: f"{m.group(1).strip()}/{m.group(2).strip()}", s)

    # Remove remaining LaTeX commands like \left, \right, spaces around operators
    s = re.sub(r'\\(?:left|right|,|;|!|quad|qquad)', '', s)

    # Strip whitespace again after substitutions
    s = s.strip()

    # Remove trailing zeros after decimal point: 1.500 -> 1.5, 2.0 -> 2
    def strip_trailing_zeros(num_str: str) -> str:
        if '.' in num_str:
            num_str = num_str.rstrip('0').rstrip('.')
        return num_str

    # Apply trailing-zero stripping when the whole string looks like a number
    if re.fullmatch(r'-?\d+\.\d+', s):
        s = strip_trailing_zeros(s)

    return s


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def _to_float(s: str) -> Optional[float]:
    """Try to parse s as a float, handling simple a/b fractions."""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    # Try fraction a/b
    frac_match = re.fullmatch(r'(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)', s)
    if frac_match:
        num, den = float(frac_match.group(1)), float(frac_match.group(2))
        if den != 0.0:
            return num / den
    return None


def answers_match(pred: str, ref: str) -> bool:
    """Return True if pred and ref represent the same answer."""
    pred_norm = _normalize_answer(pred)
    ref_norm = _normalize_answer(ref)

    # Exact string match
    if pred_norm == ref_norm:
        return True

    # Numeric equality
    pred_val = _to_float(pred_norm)
    ref_val = _to_float(ref_norm)
    if pred_val is not None and ref_val is not None:
        return abs(pred_val - ref_val) <= 1e-6

    return False


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------

def extract_numbers(text: str) -> list[float]:
    """
    Extract all numbers from text, including:
      - integers and decimals
      - simple fractions a/b
      - percentages (value stored as the numeric part, e.g. 50% -> 50.0)
      - LaTeX \\frac{a}{b}
      - simple integers inside \\text{...}
    Returns a deduplicated list of floats (order-preserving by first occurrence).
    """
    numbers: list[float] = []
    seen: set[float] = set()

    def _add(val: float) -> None:
        # Round to avoid float noise when deduplicating
        key = round(val, 9)
        if key not in seen:
            seen.add(key)
            numbers.append(val)

    # Working copy; we'll consume/replace matched regions to avoid double-counting
    working = text

    # 1. LaTeX \frac{a}{b}
    frac_latex = re.compile(r'\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}')
    for m in frac_latex.finditer(working):
        num, den = float(m.group(1)), float(m.group(2))
        if den != 0:
            _add(num / den)
    working = frac_latex.sub(' ', working)

    # 2. \text{<integer>}
    text_int = re.compile(r'\\text\{(-?\d+)\}')
    for m in text_int.finditer(working):
        _add(float(m.group(1)))
    working = text_int.sub(' ', working)

    # 3. Simple fractions a/b (must appear before plain number matching)
    simple_frac = re.compile(r'(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)')
    for m in simple_frac.finditer(working):
        num, den = float(m.group(1)), float(m.group(2))
        if den != 0:
            _add(num / den)
    working = simple_frac.sub(' ', working)

    # 4. Percentages: digits followed immediately by %
    percent_pat = re.compile(r'(-?\d+(?:\.\d+)?)\s*%')
    for m in percent_pat.finditer(working):
        _add(float(m.group(1)))
    working = percent_pat.sub(' ', working)

    # 5. Plain integers and decimals
    plain_num = re.compile(r'-?\d+(?:\.\d+)?')
    for m in plain_num.finditer(working):
        _add(float(m.group(0)))

    return numbers


# ---------------------------------------------------------------------------
# Core reward functions
# ---------------------------------------------------------------------------

def compute_score_binary(
    completions: list[str],
    ground_truth: list[str],
    **kwargs,
) -> list[float]:
    """
    TRL-compatible reward function.

    Returns 1.0 if the final \\boxed{} answer in each completion matches the
    corresponding ground_truth entry, else 0.0.
    """
    scores: list[float] = []
    for completion, gt in zip(completions, ground_truth):
        pred = extract_boxed_answer(completion)
        if pred is not None and answers_match(pred, gt):
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores


def compute_score_partial_credit(
    completions: list[str],
    ground_truth: list[str],
    solution: list[str],
    **kwargs,
) -> list[float]:
    """
    TRL-compatible reward function.

    Returns 0.5 * binary_score + 0.5 * intermediate_score where:
      intermediate_score = |model_nums ∩ ref_nums| / |ref_nums|

    If the reference solution has no extractable numbers, falls back to pure
    binary score. All returned values are in [0, 1].
    """
    binary_scores = compute_score_binary(completions, ground_truth)
    scores: list[float] = []

    for completion, ref_sol, binary in zip(completions, solution, binary_scores):
        ref_nums = extract_numbers(ref_sol)

        if not ref_nums:
            # No reference numbers to compare against — fall back to binary
            scores.append(float(binary))
            continue

        model_nums = extract_numbers(completion)

        # Build a set of rounded reference values for tolerance-based matching
        tol = 1e-4
        ref_rounded: set[float] = {round(v / tol) * tol for v in ref_nums}

        matched = 0
        for v in model_nums:
            v_r = round(v / tol) * tol
            if v_r in ref_rounded:
                matched += 1

        intermediate_score = matched / len(ref_nums)
        # Clamp to [0, 1] (matched can exceed len(ref_nums) if model repeats values,
        # but we deduplicate model_nums so this is an edge case; still, be safe)
        intermediate_score = min(intermediate_score, 1.0)

        score = 0.5 * binary + 0.5 * intermediate_score
        scores.append(score)

    return scores


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import statistics

    CORRECT_ANSWER = "42"
    CORRECT_COMPLETION_TEMPLATE = (
        "We start with the equation and solve step by step.\n"
        "First, we simplify: $6 \\times 7 = 42$.\n"
        "Therefore, the answer is $\\boxed{42}$."
    )
    INCORRECT_COMPLETION_TEMPLATE = (
        "We start with the equation and solve step by step.\n"
        "First, we simplify: $6 \\times 7 = 7$.\n"  # deliberate error
        "Therefore, the answer is $\\boxed{7}$."
    )

    REFERENCE_SOLUTION = (
        "We know that $6 \\times 7 = 42$. "
        "The intermediate result is $21 + 21 = 42$. "
        "So the final answer is $\\boxed{42}$."
    )

    n = 10
    correct_completions = [CORRECT_COMPLETION_TEMPLATE] * n
    incorrect_completions = [INCORRECT_COMPLETION_TEMPLATE] * n
    ground_truths_correct = [CORRECT_ANSWER] * n
    ground_truths_incorrect = [CORRECT_ANSWER] * n
    solutions = [REFERENCE_SOLUTION] * n

    # --- Binary scores ---
    binary_correct = compute_score_binary(correct_completions, ground_truths_correct)
    binary_incorrect = compute_score_binary(incorrect_completions, ground_truths_incorrect)

    print("=== Binary reward ===")
    print(f"Correct   completions: {binary_correct}")
    print(f"Incorrect completions: {binary_incorrect}")
    print(f"Mean correct  : {statistics.mean(binary_correct):.3f}  (expected 1.0)")
    print(f"Mean incorrect: {statistics.mean(binary_incorrect):.3f}  (expected 0.0)")

    assert statistics.mean(binary_correct) == 1.0, "Binary correct mean should be 1.0"
    assert statistics.mean(binary_incorrect) == 0.0, "Binary incorrect mean should be 0.0"
    print("Binary assertions passed.\n")

    # --- Partial credit scores ---
    partial_correct = compute_score_partial_credit(
        correct_completions, ground_truths_correct, solutions
    )
    partial_incorrect = compute_score_partial_credit(
        incorrect_completions, ground_truths_incorrect, solutions
    )

    print("=== Partial credit reward ===")
    print(f"Correct   completions: {[round(s, 4) for s in partial_correct]}")
    print(f"Incorrect completions: {[round(s, 4) for s in partial_incorrect]}")
    print(f"Mean correct  : {statistics.mean(partial_correct):.4f}")
    print(f"Mean incorrect: {statistics.mean(partial_incorrect):.4f}")

    # --- Extra unit tests for helper functions ---
    print("\n=== Helper function tests ===")

    # extract_boxed_answer
    assert extract_boxed_answer(r"The answer is \boxed{42}.") == "42"
    assert extract_boxed_answer(r"\boxed{\frac{1}{2}}") == "1/2"
    assert extract_boxed_answer(r"\boxed{3.50}") == "3.5"
    assert extract_boxed_answer("no boxed here") is None
    assert extract_boxed_answer(r"\boxed{x+1} then \boxed{99}") == "99"
    print("extract_boxed_answer: OK")

    # answers_match
    assert answers_match("42", "42")
    assert answers_match("1/2", "0.5")
    assert answers_match("3.14", "3.140000")
    assert not answers_match("42", "43")
    print("answers_match: OK")

    # extract_numbers
    nums = extract_numbers(r"We get \frac{3}{4} and 50% and 7/2 and 3.14.")
    assert 0.75 in nums, f"Expected 0.75 in {nums}"
    assert 50.0 in nums, f"Expected 50.0 in {nums}"
    assert 3.5 in nums, f"Expected 3.5 in {nums}"
    assert 3.14 in nums, f"Expected 3.14 in {nums}"
    print("extract_numbers: OK")

    print("\nAll tests passed.")
