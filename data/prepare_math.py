"""
prepare_math.py

Downloads DigitalLearningGmbH/MATH-lighteval from HuggingFace and produces:
  - train_128.parquet  (128 stratified training problems)
  - eval_200.parquet   (200 stratified eval problems)
"""

import argparse
import re
import os
import sys

import pandas as pd
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "DigitalLearningGmbH/MATH-lighteval"

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step and put your final answer in \\boxed{}."
)

TRAIN_TOTAL = 128
EVAL_TOTAL = 200

# Desired train counts per difficulty bucket
TRAIN_EASY_COUNT = 42    # level 1-2
TRAIN_MEDIUM_COUNT = 64  # level 3
TRAIN_HARD_COUNT = 22    # level 4-5

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name present in df (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def parse_level(value: str) -> int | None:
    """Extract integer 1-5 from strings like 'Level 3' or '3'."""
    if value is None:
        return None
    m = re.search(r"(\d)", str(value))
    if m:
        lvl = int(m.group(1))
        if 1 <= lvl <= 5:
            return lvl
    return None


def extract_boxed_answer(solution: str) -> str:
    r"""
    Extract the content of the last \boxed{...} in a solution string.
    Handles nested braces.
    """
    if not solution:
        return ""

    # Find all occurrences of \boxed{ and extract balanced brace content
    results = []
    pattern = r"\\boxed\{"
    for match in re.finditer(pattern, solution):
        start = match.end()  # position just after the opening {
        depth = 1
        idx = start
        while idx < len(solution) and depth > 0:
            if solution[idx] == "{":
                depth += 1
            elif solution[idx] == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            results.append(solution[start : idx - 1])

    return results[-1] if results else ""


def build_prompt(problem: str) -> list[dict]:
    """Return a TRL-style chat prompt (list of role/content dicts)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


# ---------------------------------------------------------------------------
# Data loading & normalisation
# ---------------------------------------------------------------------------

def load_and_normalise(split: str) -> pd.DataFrame:
    """Load the dataset split and return a normalised DataFrame."""
    print(f"Loading split='{split}' from {DATASET_ID} …")
    ds = load_dataset(DATASET_ID, split=split)
    df = ds.to_pandas()

    print(f"  Raw columns : {list(df.columns)}")
    print(f"  Raw rows    : {len(df)}")

    # Resolve column names flexibly
    problem_col  = _find_column(df, ["problem"])
    solution_col = _find_column(df, ["solution"])
    level_col    = _find_column(df, ["level", "Level"])
    subject_col  = _find_column(df, ["subject", "type", "problem_type"])

    missing = [
        name for name, col in [
            ("problem", problem_col),
            ("solution", solution_col),
            ("level", level_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(f"Required columns not found in dataset: {missing}. "
                         f"Available columns: {list(df.columns)}")

    # Build normalised frame
    out = pd.DataFrame()
    out["problem"]      = df[problem_col].astype(str)
    out["solution"]     = df[solution_col].astype(str)
    out["level"]        = df[level_col].apply(parse_level)
    out["problem_type"] = df[subject_col].astype(str) if subject_col else "unknown"

    # Drop rows where level could not be parsed
    before = len(out)
    out = out.dropna(subset=["level"])
    out["level"] = out["level"].astype(int)
    after = len(out)
    if before != after:
        print(f"  Dropped {before - after} rows with unparseable level values.")

    out["ground_truth"] = out["solution"].apply(extract_boxed_answer)
    out["prompt"]       = out["problem"].apply(build_prompt)

    print(f"  Usable rows after normalisation: {len(out)}")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample_train(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Sample 128 training problems with the fixed bucket allocation:
      easy   (level 1-2): 42
      medium (level 3)  : 64
      hard   (level 4-5): 22
    """
    rng_state = seed

    easy   = df[df["level"].isin([1, 2])]
    medium = df[df["level"] == 3]
    hard   = df[df["level"].isin([4, 5])]

    def _safe_sample(subset: pd.DataFrame, n: int, rs: int) -> pd.DataFrame:
        available = len(subset)
        if available < n:
            print(f"  WARNING: requested {n} but only {available} available – using all.")
            n = available
        return subset.sample(n=n, random_state=rs)

    parts = [
        _safe_sample(easy,   TRAIN_EASY_COUNT,   rng_state),
        _safe_sample(medium, TRAIN_MEDIUM_COUNT, rng_state + 1),
        _safe_sample(hard,   TRAIN_HARD_COUNT,   rng_state + 2),
    ]
    result = pd.concat(parts, ignore_index=True)
    result = result.sample(frac=1, random_state=rng_state).reset_index(drop=True)
    return result


def stratified_sample_eval(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """
    Sample n_total eval problems proportionally across all 5 levels.
    """
    level_counts = df["level"].value_counts().sort_index()
    total_available = level_counts.sum()

    # Compute proportional allocation
    allocation = {}
    running = 0
    levels_sorted = sorted(level_counts.index)
    for i, lvl in enumerate(levels_sorted):
        proportion = level_counts[lvl] / total_available
        if i < len(levels_sorted) - 1:
            n = round(n_total * proportion)
        else:
            # Last bucket gets the remainder to ensure exact total
            n = n_total - running
        n = max(1, n)
        allocation[lvl] = n
        running += n

    # Adjust if total overshoots (due to rounding + max(1, …))
    while sum(allocation.values()) > n_total:
        # Shave from the largest bucket
        largest = max(allocation, key=allocation.get)
        allocation[largest] -= 1

    print(f"  Eval allocation by level: {allocation}")

    parts = []
    for lvl in levels_sorted:
        subset = df[df["level"] == lvl]
        n = allocation[lvl]
        available = len(subset)
        if available < n:
            print(f"  WARNING: level {lvl}: requested {n} but only {available} available.")
            n = available
        parts.append(subset.sample(n=n, random_state=seed))

    result = pd.concat(parts, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = ["prompt", "ground_truth", "solution", "level", "problem_type", "problem"]


def main():
    parser = argparse.ArgumentParser(description="Prepare MATH-lighteval datasets.")
    parser.add_argument(
        "--output_dir",
        default="/workspace/discovery/data",
        help="Directory where parquet files will be written.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load train split ---
    train_df = load_and_normalise("train")

    print(f"\nSampling {TRAIN_TOTAL} training problems …")
    train_sample = stratified_sample_train(train_df, seed=SEED)
    print(f"  Train sample size: {len(train_sample)}")
    print(f"  Level distribution:\n{train_sample['level'].value_counts().sort_index().to_string()}")

    # --- Load test split ---
    eval_df = load_and_normalise("test")

    print(f"\nSampling {EVAL_TOTAL} eval problems …")
    eval_sample = stratified_sample_eval(eval_df, n_total=EVAL_TOTAL, seed=SEED)
    print(f"  Eval sample size: {len(eval_sample)}")
    print(f"  Level distribution:\n{eval_sample['level'].value_counts().sort_index().to_string()}")

    # --- Save ---
    train_path = os.path.join(args.output_dir, "train_128.parquet")
    eval_path  = os.path.join(args.output_dir, "eval_200.parquet")

    train_sample[OUTPUT_COLUMNS].to_parquet(train_path, index=False)
    eval_sample[OUTPUT_COLUMNS].to_parquet(eval_path, index=False)

    print(f"\nSaved train data to : {train_path}")
    print(f"Saved eval data to  : {eval_path}")

    # --- 3-row sample preview ---
    print("\n--- 3-row sample of training data ---")
    preview = train_sample[["problem", "level", "ground_truth"]].head(3).copy()
    preview["problem"] = preview["problem"].str[:100]
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
