"""
plot_trajectories.py

Load all step JSONs for both runs (binary + dense), compute delta relative to
step 0, and produce trajectory plots.

Usage:
    python analysis/plot_trajectories.py \
        --results_dir results \
        --output_dir analysis \
        [--runs binary dense] \
        [--format png] \
        [--dpi 150]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
COLORS = {
    "binary": "#e74c3c",
    "dense": "#3498db",
}

LINE_STYLES = {
    "binary": "-",
    "dense": "--",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_data(results_dir: Path, run: str) -> list[dict]:
    """Load all step JSONs for a given run, sorted by step number."""
    run_dir = results_dir / run
    if not run_dir.exists():
        warnings.warn(f"Directory not found for run '{run}': {run_dir}. Skipping.")
        return []

    json_files = sorted(run_dir.glob("step_*.json"))
    if not json_files:
        warnings.warn(f"No step_XXXX.json files found in {run_dir}. Skipping run '{run}'.")
        return []

    records = []
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            records.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            warnings.warn(f"Could not read {jf}: {exc}. Skipping file.")

    # Sort by step field (use filename-derived step as fallback)
    records.sort(key=lambda d: d.get("step", 0))
    return records


def records_to_dataframe(records: list[dict], run: str) -> pd.DataFrame:
    """Convert a list of step records into a tidy DataFrame."""
    if not records:
        return pd.DataFrame()

    rows = []
    for rec in records:
        step = rec.get("step", 0)
        pak = rec.get("pass_at_k", {})
        row = {
            "run": run,
            "step": step,
            "pass@1": pak.get("1", np.nan),
            "pass@4": pak.get("4", np.nan),
            "pass@16": pak.get("16", np.nan),
            "pass@64": pak.get("64", np.nan),
            "unique_correct_answers": rec.get("unique_correct_answers", np.nan),
            "answer_entropy": rec.get("answer_entropy", np.nan),
            "mean_correct_per_problem": rec.get("mean_correct_per_problem", np.nan),
            "_raw": rec,  # keep raw for difficulty breakdown
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add Δpass@k columns relative to step 0 (or first available step)."""
    if df.empty:
        return df

    df = df.copy().reset_index(drop=True)

    # Find baseline step
    if 0 in df["step"].values:
        baseline_idx = df.index[df["step"] == 0][0]
    else:
        baseline_idx = 0
        first_step = df.loc[0, "step"]
        warnings.warn(
            f"Step 0 not found for run '{df['run'].iloc[0]}'. "
            f"Using step {first_step} as baseline."
        )

    for k in ("1", "4", "16", "64"):
        col = f"pass@{k}"
        base_val = df.loc[baseline_idx, col]
        df[f"delta_pass@{k}"] = (df[col] - base_val) * 100  # in percentage points

    return df


def get_difficulty_pass64(rec: dict) -> dict[int, float]:
    """Extract pass@64 by difficulty level from a raw record."""
    pak_by_level = rec.get("pass_at_k_by_level", {})
    level_data = pak_by_level.get("64", {})
    result = {}
    for lvl in range(1, 6):
        val = level_data.get(str(lvl), np.nan)
        result[lvl] = float(val) if val is not None else np.nan
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(run_dfs: dict[str, pd.DataFrame]) -> None:
    """Print a formatted summary table to stdout."""
    header = f"{'Run':<10} {'Step':>6}  {'pass@1':>7}  {'pass@4':>7}  {'pass@16':>8}  {'pass@64':>8}  {'Δpass@1':>8}  {'Δpass@64':>9}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for run, df in run_dfs.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            def fmt_val(v):
                return f"{v:.4f}" if not np.isnan(v) else "   nan"

            def fmt_delta(v):
                if np.isnan(v):
                    return "     nan"
                sign = "+" if v >= 0 else ""
                return f"{sign}{v:.2f}"

            print(
                f"{run:<10} {int(row['step']):>6}  "
                f"{fmt_val(row['pass@1']):>7}  "
                f"{fmt_val(row['pass@4']):>7}  "
                f"{fmt_val(row['pass@16']):>8}  "
                f"{fmt_val(row['pass@64']):>8}  "
                f"{fmt_delta(row['delta_pass@1']):>8}  "
                f"{fmt_delta(row['delta_pass@64']):>9}"
            )

    print(sep)


# ---------------------------------------------------------------------------
# Figure 1: Trajectory in (Δpass@1, Δpass@64) space
# ---------------------------------------------------------------------------

def plot_trajectory_delta_space(
    run_dfs: dict[str, pd.DataFrame],
    output_path: Path,
    fmt: str = "png",
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for run, df in run_dfs.items():
        if df.empty:
            continue

        color = COLORS.get(run, "gray")
        ls = LINE_STYLES.get(run, "-")

        x = df["delta_pass@1"].values
        y = df["delta_pass@64"].values
        steps = df["step"].values

        # Draw the trajectory line
        ax.plot(x, y, color=color, linestyle=ls, linewidth=2, label=run, zorder=3)

        # Arrows showing direction (on each segment)
        for i in range(len(x) - 1):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            if abs(dx) + abs(dy) < 1e-9:
                continue
            # Place arrow at midpoint of segment
            mx, my = (x[i] + x[i + 1]) / 2, (y[i] + y[i + 1]) / 2
            ax.annotate(
                "",
                xy=(mx + dx * 0.01, my + dy * 0.01),
                xytext=(mx - dx * 0.01, my - dy * 0.01),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                zorder=4,
            )

        # Label points at every other checkpoint
        for i, (xi, yi, s) in enumerate(zip(x, y, steps)):
            if i % 2 == 0:
                ax.annotate(
                    str(int(s)),
                    xy=(xi, yi),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    zorder=5,
                )
            ax.scatter(xi, yi, color=color, s=30, zorder=5)

    # Quadrant dividers
    ax.axhline(0, color="lightgray", linestyle="--", linewidth=1, zorder=1)
    ax.axvline(0, color="lightgray", linestyle="--", linewidth=1, zorder=1)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Expand limits slightly so labels are visible even if data is near zero
    x_pad = max(abs(xlim[1] - xlim[0]) * 0.08, 0.5)
    y_pad = max(abs(ylim[1] - ylim[0]) * 0.08, 0.5)
    ax.set_xlim(xlim[0] - x_pad, xlim[1] + x_pad)
    ax.set_ylim(ylim[0] - y_pad, ylim[1] + y_pad)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    quad_kw = dict(fontsize=10, alpha=0.5, ha="center", va="center")
    ax.text(
        xlim[1] * 0.6, ylim[1] * 0.6,
        "(+,+)\nDiscovery", color="green", **quad_kw
    )
    ax.text(
        xlim[1] * 0.6, ylim[0] * 0.6,
        "(+,\u2212)\nSharpening", color="orange", **quad_kw
    )
    ax.text(
        xlim[0] * 0.6, ylim[1] * 0.6,
        "(\u2212,+)\nExpansion", color="purple", **quad_kw
    )
    ax.text(
        xlim[0] * 0.6, ylim[0] * 0.6,
        "(\u2212,\u2212)\nRegression", color="red", **quad_kw
    )

    ax.set_xlabel("\u0394pass@1 (percentage points from step 0)", fontsize=12)
    ax.set_ylabel("\u0394pass@64 (percentage points from step 0)", fontsize=12)
    ax.set_title("Reward Trajectory in (\u0394pass@1, \u0394pass@64) Space", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_path.with_suffix(f".{fmt}")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: pass@k over training steps
# ---------------------------------------------------------------------------

def plot_pass_at_k_over_steps(
    run_dfs: dict[str, pd.DataFrame],
    output_path: Path,
    fmt: str = "png",
    dpi: int = 150,
) -> None:
    ks = ["1", "4", "16", "64"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for ax_idx, k in enumerate(ks):
        ax = axes_flat[ax_idx]
        col = f"pass@{k}"
        baseline_drawn = False

        for run, df in run_dfs.items():
            if df.empty:
                continue
            color = COLORS.get(run, "gray")
            ls = LINE_STYLES.get(run, "-")

            steps = df["step"].values
            vals = df[col].values

            ax.plot(steps, vals, color=color, linestyle=ls, linewidth=2, label=run, marker="o", markersize=4)

            # Step-0 baseline horizontal dashed line (draw once per subplot)
            if not baseline_drawn and len(df) > 0:
                # Use the first row (step 0 or earliest step)
                base_val = df.iloc[0][col]
                if not np.isnan(base_val):
                    ax.axhline(
                        base_val,
                        color="gray",
                        linestyle=":",
                        linewidth=1.2,
                        alpha=0.7,
                        label=f"step-0 baseline",
                    )
                    baseline_drawn = True

        ax.set_title(f"pass@{k}", fontsize=12)
        ax.set_xlabel("Training step", fontsize=10)
        ax.set_ylabel(f"pass@{k}", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("pass@k Over Training Steps", fontsize=15, y=1.01)
    fig.tight_layout()
    out = output_path.with_suffix(f".{fmt}")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Difficulty breakdown (final checkpoint only)
# ---------------------------------------------------------------------------

def plot_difficulty_breakdown(
    run_dfs: dict[str, pd.DataFrame],
    output_path: Path,
    fmt: str = "png",
    dpi: int = 150,
) -> None:
    levels = [1, 2, 3, 4, 5]
    runs_with_data = [(run, df) for run, df in run_dfs.items() if not df.empty]

    if not runs_with_data:
        warnings.warn("No data available for difficulty breakdown plot. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    n_runs = len(runs_with_data)
    bar_width = 0.35
    x = np.arange(len(levels))

    for run_idx, (run, df) in enumerate(runs_with_data):
        # Final checkpoint = last row by step
        final_rec = df.iloc[-1]["_raw"]
        level_vals = get_difficulty_pass64(final_rec)
        heights = [level_vals.get(lvl, np.nan) for lvl in levels]

        color = COLORS.get(run, "gray")
        offset = (run_idx - (n_runs - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            heights,
            width=bar_width,
            color=color,
            alpha=0.85,
            label=f"{run} (step {int(df.iloc[-1]['step'])})",
            zorder=3,
        )

        # Annotate bar heights
        for bar, h in zip(bars, heights):
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {lvl}" for lvl in levels], fontsize=11)
    ax.set_xlabel("Difficulty Level", fontsize=12)
    ax.set_ylabel("pass@64", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("pass@64 by Difficulty Level (Final Checkpoint)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_path.with_suffix(f".{fmt}")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot training trajectories from step JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Root directory containing per-run subdirectories with step_XXXX.json files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis"),
        help="Directory where output plot files will be saved.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["binary", "dense"],
        help="Names of runs to process (must match subdirectory names in results_dir).",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output file format.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster output formats.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    results_dir: Path = args.results_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data for each run
    # ------------------------------------------------------------------
    run_dfs: dict[str, pd.DataFrame] = {}
    for run in args.runs:
        records = load_run_data(results_dir, run)
        df = records_to_dataframe(records, run)
        df = compute_deltas(df)
        run_dfs[run] = df

    # Check that at least one run has data
    if all(df.empty for df in run_dfs.values()):
        print("ERROR: No data found for any run. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print_summary(run_dfs)

    # ------------------------------------------------------------------
    # Figure 1: Trajectory in delta space
    # ------------------------------------------------------------------
    plot_trajectory_delta_space(
        run_dfs,
        output_dir / "trajectory_delta_space",
        fmt=args.format,
        dpi=args.dpi,
    )

    # ------------------------------------------------------------------
    # Figure 2: pass@k over steps
    # ------------------------------------------------------------------
    plot_pass_at_k_over_steps(
        run_dfs,
        output_dir / "pass_at_k_over_steps",
        fmt=args.format,
        dpi=args.dpi,
    )

    # ------------------------------------------------------------------
    # Figure 3: Difficulty breakdown
    # ------------------------------------------------------------------
    plot_difficulty_breakdown(
        run_dfs,
        output_dir / "difficulty_breakdown",
        fmt=args.format,
        dpi=args.dpi,
    )

    print("Done.")


if __name__ == "__main__":
    main()
