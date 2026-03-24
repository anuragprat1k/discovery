#!/usr/bin/env python3
"""GPU queue runner for Wordle autoresearch on RunPod.

Reads experiments from queue.jsonl, runs them sequentially via train_2gpu.py,
evaluates with evaluate_run.py, and logs to results.tsv.

No source-code mutation — each experiment specifies its reward module by name.

Usage:
    nohup python3 -m wordle.autoresearch.run_queue_gpu </dev/null > runner.log 2>&1 &

Queue format (queue.jsonl, one JSON object per line):
    {
        "name": "potential_sft100",
        "hypothesis": "entropy shaping after SFT teaches info-theoretic strategy",
        "reward": "potential",
        "model_path": "/workspace/checkpoints/sft_merged",
        "cli_args": "--max_steps 200 --num_generations 8 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --vllm_gpu_memory_utilization 0.5"
    }
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AUTORESEARCH_DIR = REPO_ROOT / "wordle" / "autoresearch"
RESULTS_DIR = AUTORESEARCH_DIR / "results"
QUEUE_FILE = AUTORESEARCH_DIR / "queue_gpu.jsonl"
RESULTS_TSV = AUTORESEARCH_DIR / "results_gpu.tsv"

# Default CLI args shared by all experiments
BASE_CLI_ARGS = (
    "--save_steps 50 --trace_steps 25 --n_trace_words 5 "
    "--wandb_project wordle-discovery"
)


def log(msg: str) -> None:
    """Print with timestamp."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def init_results_tsv() -> None:
    """Create results TSV with header if it doesn't exist."""
    if not RESULTS_TSV.exists():
        header = "\t".join([
            "name", "hypothesis", "reward", "score", "win_rate",
            "violation_rate", "format_compliance", "mean_completion_len",
            "memory_peak_gb", "wall_time_min", "status", "timestamp",
        ])
        RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_TSV, "w") as f:
            f.write(header + "\n")


def append_results_tsv(row: dict) -> None:
    """Append a row to results TSV."""
    fields = [
        str(row.get("name", "")),
        str(row.get("hypothesis", "")),
        str(row.get("reward", "")),
        str(row.get("score", "")),
        str(row.get("win_rate", "")),
        str(row.get("violation_rate", "")),
        str(row.get("format_compliance", "")),
        str(row.get("mean_completion_len", "")),
        str(row.get("memory_peak_gb", "")),
        str(row.get("wall_time_min", "")),
        str(row.get("status", "")),
        row.get("timestamp", datetime.date.today().isoformat()),
    ]
    with open(RESULTS_TSV, "a") as f:
        f.write("\t".join(fields) + "\n")


def read_queue() -> list[dict]:
    """Read all experiments from queue file."""
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
    """Pop the first experiment from queue."""
    experiments = read_queue()
    if not experiments:
        return None
    first = experiments[0]
    with open(QUEUE_FILE, "w") as f:
        for exp in experiments[1:]:
            f.write(json.dumps(exp) + "\n")
    return first


def run_experiment(exp: dict) -> dict:
    """Run a single experiment end-to-end. Returns result dict."""
    name = exp["name"]
    hypothesis = exp.get("hypothesis", "")
    reward = exp.get("reward", "dense")
    model_path = exp.get("model_path", "Qwen/Qwen3-4B")
    cli_args = exp.get("cli_args", "")

    exp_dir = RESULTS_DIR / name
    log_file = exp_dir / "training_log.jsonl"
    stdout_file = exp_dir / "stdout.log"

    log(f"{'=' * 60}")
    log(f"EXPERIMENT: {name}")
    log(f"Hypothesis: {hypothesis}")
    log(f"Reward: {reward}")
    log(f"Model: {model_path}")
    log(f"CLI args: {cli_args}")
    log(f"{'=' * 60}")

    # Clean up any previous partial run
    if exp_dir.exists():
        log(f"Cleaning up previous {name} dir")
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True)

    # Build command
    model_args = f"--model_path {model_path}" if model_path and not model_path.startswith("Qwen/") else f"--model {model_path}"
    cmd = (
        f"python3 -m wordle.recipes.train_2gpu "
        f"--reward {reward} {model_args} "
        f"--output_dir {exp_dir} --log_file {log_file} "
        f"--run_name {name} "
        f"{BASE_CLI_ARGS} {cli_args}"
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
        (exp_dir / "pid").write_text(str(proc.pid))

        # Poll until done, printing progress
        while proc.poll() is None:
            time.sleep(60)
            if log_file.exists():
                try:
                    lines = log_file.read_text().strip().split("\n")
                    if lines and lines[-1]:
                        last = json.loads(lines[-1])
                        step = last.get("step", "?")
                        log(f"  [{name}] step {step}")
                except (json.JSONDecodeError, IndexError):
                    pass

    wall_time = time.time() - t_start
    exit_code = proc.returncode
    log(f"Training finished: exit_code={exit_code}, wall_time={wall_time / 60:.1f}min")

    # Evaluate
    eval_result = evaluate(name, log_file)
    eval_result["reward"] = reward
    eval_result["hypothesis"] = hypothesis
    eval_result["wall_time_actual"] = round(wall_time / 60, 1)

    return eval_result


def evaluate(name: str, log_file: Path) -> dict:
    """Run evaluate_run.py and parse output."""
    if not log_file.exists() or log_file.stat().st_size == 0:
        log(f"No training data for {name}")
        return {"name": name, "score": 0.0, "status": "NO_DATA"}

    result = subprocess.run(
        ["python3", "-m", "wordle.autoresearch.evaluate_run", str(log_file)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )

    log(f"Evaluator exit_code={result.returncode}")
    if result.stderr:
        log(f"Evaluator stderr: {result.stderr.strip()}")

    output = result.stdout.strip()
    if not output:
        return {"name": name, "score": 0.0, "status": "EVAL_ERROR"}

    parts = output.split("\t")
    status = parts[7] if len(parts) > 7 else "UNKNOWN"

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
    }


def main():
    log("=" * 60)
    log("GPU AUTORESEARCH QUEUE RUNNER STARTED")
    log(f"Queue file: {QUEUE_FILE}")
    log(f"Results: {RESULTS_TSV}")
    log("=" * 60)

    if not QUEUE_FILE.exists():
        log("No queue_gpu.jsonl found. Nothing to run.")
        return

    init_results_tsv()
    experiments_run = 0

    while True:
        exp = pop_queue()
        if exp is None:
            log("Queue empty. Done.")
            break

        name = exp["name"]
        result = run_experiment(exp)
        score = result.get("score", 0.0)
        log(f"RESULT: {name} → score={score:.4f}")

        # Log to results TSV
        result["timestamp"] = datetime.date.today().isoformat()
        append_results_tsv(result)
        log(f"Logged to {RESULTS_TSV}")

        experiments_run += 1
        remaining = len(read_queue())
        log(f"Experiments completed: {experiments_run}, queue remaining: {remaining}")

    log("=" * 60)
    log(f"QUEUE RUNNER COMPLETE — {experiments_run} experiments run")
    log("=" * 60)


if __name__ == "__main__":
    main()
