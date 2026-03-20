#!/usr/bin/env python3
"""Autonomous experiment queue runner for Wordle autoresearch.

Reads experiments from queue.jsonl, runs them sequentially, evaluates,
logs results to results.tsv, and applies keep/revert logic for reward constants.

Fully decoupled from the Claude agent — runs as a standalone process.
The agent's job is to populate queue.jsonl; this script executes it.

Usage:
    nohup python3 -m wordle.autoresearch.run_queue </dev/null > wordle/autoresearch/runner.log 2>&1 &

Queue format (queue.jsonl, one JSON object per line):
    {
        "name": "experiment_name",
        "hypothesis": "what we're testing",
        "reward_constants": {
            "LENGTH_PENALTY_PER_TOKEN": -0.005,
            "LENGTH_PENALTY_FREE_TOKENS": 20,
            "CONSTRAINT_VIOLATION_PENALTY": 0.0
        },
        "cli_args": "--lr 1e-4 --max_completion_tokens 128"
    }
"""
from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AUTORESEARCH_DIR = REPO_ROOT / "wordle" / "autoresearch"
RESULTS_DIR = AUTORESEARCH_DIR / "results"
QUEUE_FILE = AUTORESEARCH_DIR / "queue.jsonl"
RESULTS_TSV = AUTORESEARCH_DIR / "results.tsv"
DENSE_REWARD_FILE = REPO_ROOT / "wordle" / "rewards" / "dense_reward.py"

# Default CLI args for all experiments
BASE_CLI_ARGS = (
    "--env_eliminated "
    "--save_steps 5 --wandb_project discovery-wordle-autoresearch"
)

# Reward constants and their defaults
DEFAULT_REWARD_CONSTANTS = {
    "LENGTH_PENALTY_PER_TOKEN": 0.0,
    "LENGTH_PENALTY_FREE_TOKENS": 40,
    "CONSTRAINT_VIOLATION_PENALTY": 0.0,
}


def log(msg: str) -> None:
    """Print with timestamp."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_best_score() -> float:
    """Read results.tsv and return the best score so far."""
    if not RESULTS_TSV.exists():
        return 0.0
    best = 0.0
    with open(RESULTS_TSV) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # header
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                try:
                    best = max(best, float(parts[2]))
                except ValueError:
                    pass
    return best


def set_reward_constants(constants: dict) -> None:
    """Write reward constants to dense_reward.py."""
    content = DENSE_REWARD_FILE.read_text()
    for name, value in constants.items():
        # Match pattern: NAME = <value>  (with optional comment)
        pattern = rf"^({name}\s*=\s*).*$"
        if isinstance(value, float):
            replacement = rf"\g<1>{value}"
        else:
            replacement = rf"\g<1>{value}"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    DENSE_REWARD_FILE.write_text(content)


def get_reward_constants() -> dict:
    """Read current reward constants from dense_reward.py."""
    content = DENSE_REWARD_FILE.read_text()
    constants = {}
    for name in DEFAULT_REWARD_CONSTANTS:
        match = re.search(rf"^{name}\s*=\s*([-\d.eE]+)", content, re.MULTILINE)
        if match:
            constants[name] = float(match.group(1))
    return constants


def revert_reward_constants() -> None:
    """Reset reward constants to defaults."""
    set_reward_constants(DEFAULT_REWARD_CONSTANTS)


def append_results_tsv(row: dict) -> None:
    """Append a row to results.tsv."""
    fields = [
        row.get("name", ""),
        row.get("hypothesis", ""),
        str(row.get("score", "")),
        str(row.get("win_rate", "")),
        str(row.get("violation_rate", "")),
        str(row.get("format_compliance", "")),
        str(row.get("mean_completion_len", "")),
        str(row.get("memory_peak_gb", "")),
        str(row.get("wall_time_min", "")),
        str(row.get("steps_completed", "")),
        str(row.get("max_steps", "")),
        row.get("early_stopped", "no"),
        row.get("early_stop_reason", ""),
        row.get("status", ""),
        row.get("timestamp", datetime.date.today().isoformat()),
    ]
    with open(RESULTS_TSV, "a") as f:
        f.write("\t".join(fields) + "\n")


def read_queue() -> list[dict]:
    """Read all experiments from queue.jsonl."""
    if not QUEUE_FILE.exists():
        return []
    experiments = []
    with open(QUEUE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                experiments.append(json.loads(line))
            except json.JSONDecodeError:
                log(f"WARNING: Skipping malformed queue line: {line[:80]}")
    return experiments


def pop_queue() -> dict | None:
    """Pop the first experiment from queue.jsonl."""
    experiments = read_queue()
    if not experiments:
        return None
    first = experiments[0]
    # Rewrite queue without the first entry
    with open(QUEUE_FILE, "w") as f:
        for exp in experiments[1:]:
            f.write(json.dumps(exp) + "\n")
    return first


def run_experiment(exp: dict) -> dict:
    """Run a single experiment end-to-end. Returns result dict."""
    name = exp["name"]
    hypothesis = exp.get("hypothesis", "")
    reward_constants = exp.get("reward_constants", {})
    cli_args = exp.get("cli_args", "")

    exp_dir = RESULTS_DIR / name
    log_file = exp_dir / "training_log.jsonl"
    stdout_file = exp_dir / "stdout.log"

    log(f"{'='*60}")
    log(f"EXPERIMENT: {name}")
    log(f"Hypothesis: {hypothesis}")
    log(f"Reward constants: {reward_constants}")
    log(f"Extra CLI args: {cli_args}")
    log(f"{'='*60}")

    # Clean up any previous partial run
    if exp_dir.exists():
        log(f"Cleaning up previous {name} dir")
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True)

    # Set reward constants
    if reward_constants:
        merged = {**DEFAULT_REWARD_CONSTANTS, **reward_constants}
        set_reward_constants(merged)
        log(f"Set reward constants: {merged}")
    else:
        revert_reward_constants()
        log("Using default reward constants")

    # Build command
    cmd = (
        f"python3 -m wordle.recipes.train_mlx {BASE_CLI_ARGS} "
        f"--output_dir {exp_dir} --log_file {log_file} {cli_args}"
    )
    log(f"CMD: {cmd}")

    # Run training
    t_start = time.time()
    with open(stdout_file, "w") as stdout_f:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=stdout_f,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
        # Save PID
        (exp_dir / "pid").write_text(str(proc.pid))

        # Poll until done, printing progress
        while proc.poll() is None:
            time.sleep(60)
            # Read last JSONL line for progress
            if log_file.exists():
                try:
                    lines = log_file.read_text().strip().split("\n")
                    if lines and lines[-1]:
                        last = json.loads(lines[-1])
                        step = last.get("step", "?")
                        max_steps = exp.get("max_steps", 10)
                        win = last.get("win_rate", 0)
                        viol = last.get("constraint_violation_rate", 0)
                        mem = last.get("memory_peak_gb", 0)
                        log(
                            f"  [{name}] step {step}/{max_steps} "
                            f"win={win:.1%} viol={viol:.1%} mem={mem:.1f}GB"
                        )
                except (json.JSONDecodeError, IndexError):
                    pass

    wall_time = time.time() - t_start
    exit_code = proc.returncode
    log(f"Training finished: exit_code={exit_code}, wall_time={wall_time/60:.1f}min")

    # Evaluate
    eval_result = evaluate(name, log_file, max_steps=exp.get("max_steps", 10))
    eval_result["wall_time_actual"] = round(wall_time / 60, 1)

    return eval_result


def evaluate(name: str, log_file: Path, max_steps: int = 10) -> dict:
    """Run evaluate_run.py and parse output."""
    if not log_file.exists() or log_file.stat().st_size == 0:
        log(f"No training data for {name}")
        return {
            "name": name, "score": 0.0, "status": "NO_DATA",
            "steps_completed": 0, "early_stopped": "yes",
            "early_stop_reason": "no_data",
        }

    # Count steps
    lines = [l for l in log_file.read_text().strip().split("\n") if l.strip()]
    grpo_lines = []
    for l in lines:
        try:
            d = json.loads(l)
            if d.get("phase") != "sft_warmup" and "step" in d:
                grpo_lines.append(d)
        except json.JSONDecodeError:
            pass
    steps_completed = len(grpo_lines)

    result = subprocess.run(
        ["python3", "-m", "wordle.autoresearch.evaluate_run", str(log_file)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )

    log(f"Evaluator exit_code={result.returncode}")
    if result.stderr:
        log(f"Evaluator stderr: {result.stderr.strip()}")

    early_stopped = result.returncode == 1
    output = result.stdout.strip()

    if not output:
        return {
            "name": name, "score": 0.0, "status": "EVAL_ERROR",
            "steps_completed": steps_completed, "early_stopped": "yes",
            "early_stop_reason": "eval_error",
        }

    parts = output.split("\t")
    status = parts[7] if len(parts) > 7 else "UNKNOWN"
    early_stop_reason = ""
    if status.startswith("EARLY_STOPPED:"):
        early_stop_reason = status.split(":", 1)[1]
        status = "EARLY_STOPPED"

    return {
        "name": name,
        "score": float(parts[0]) if parts[0] else 0.0,
        "win_rate": float(parts[1]) if len(parts) > 1 else 0.0,
        "violation_rate": float(parts[2]) if len(parts) > 2 else 0.0,
        "format_compliance": float(parts[3]) if len(parts) > 3 else 0.0,
        "mean_completion_len": float(parts[4]) if len(parts) > 4 else 0.0,
        "memory_peak_gb": float(parts[5]) if len(parts) > 5 else 0.0,
        "wall_time_min": float(parts[6]) if len(parts) > 6 else 0.0,
        "status": status,
        "steps_completed": steps_completed,
        "max_steps": max_steps,
        "early_stopped": "yes" if early_stopped else "no",
        "early_stop_reason": early_stop_reason,
    }


def main():
    log("=" * 60)
    log("AUTORESEARCH QUEUE RUNNER STARTED")
    log(f"Queue file: {QUEUE_FILE}")
    log(f"Results: {RESULTS_TSV}")
    log("=" * 60)

    if not QUEUE_FILE.exists():
        log("No queue.jsonl found. Nothing to run.")
        return

    best_score = get_best_score()
    log(f"Current best score: {best_score:.4f}")

    experiments_run = 0

    while True:
        exp = pop_queue()
        if exp is None:
            log("Queue empty. Done.")
            break

        name = exp["name"]
        hypothesis = exp.get("hypothesis", "")

        # Run the experiment
        result = run_experiment(exp)
        score = result.get("score", 0.0)

        log(f"RESULT: {name} → score={score:.4f} (best={best_score:.4f})")

        # Keep or revert
        if score > best_score:
            log(f"NEW BEST! {score:.4f} > {best_score:.4f} — keeping reward changes")
            best_score = score
        else:
            log(f"No improvement — reverting reward constants")
            revert_reward_constants()

        # Log to results.tsv
        result["hypothesis"] = hypothesis
        result["timestamp"] = datetime.date.today().isoformat()
        append_results_tsv(result)
        log(f"Logged to results.tsv")

        experiments_run += 1
        log(f"Experiments completed: {experiments_run}")
        remaining = len(read_queue())
        log(f"Queue remaining: {remaining}")

    log("=" * 60)
    log(f"QUEUE RUNNER COMPLETE — {experiments_run} experiments run")
    log(f"Best score: {best_score:.4f}")
    log("=" * 60)


if __name__ == "__main__":
    main()
