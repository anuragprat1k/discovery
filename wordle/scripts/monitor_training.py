#!/usr/bin/env python3
"""Monitor training_log.jsonl and print a status summary.

Exit codes:
  0 = healthy
  1 = warning (e.g. win_rate collapsed, loss spike)
  2 = critical (NaN loss, process dead, no progress for >30 min)
"""
import json
import sys
import time
from pathlib import Path

LOG = Path("checkpoints/wordle_mlx_4b_dense/training_log.jsonl")

def load_metrics():
    if not LOG.exists():
        return []
    with open(LOG) as f:
        return [json.loads(line) for line in f if line.strip()]

def analyze(metrics):
    if not metrics:
        print("STATUS: NO DATA YET")
        return 0

    latest = metrics[-1]
    step = latest.get("step", "?")
    n = len(metrics)

    # Print latest
    print(f"=== Step {step} ({n} entries logged) ===")
    for k in ["loss", "mean_reward", "win_rate", "format_compliance_rate",
              "constraint_violation_rate", "time_gen", "time_total",
              "memory_active_gb", "memory_peak_gb"]:
        if k in latest:
            print(f"  {k}: {latest[k]}")

    # Check for NaN
    loss = latest.get("loss")
    if loss is not None and (loss != loss or str(loss).lower() == "nan"):
        print("CRITICAL: Loss is NaN!")
        return 2

    # Check for loss explosion (>100)
    if loss is not None and abs(loss) > 100:
        print(f"WARNING: Loss exploded to {loss}")
        return 1

    # Check win_rate collapse: if we had >5% win_rate at some point but
    # last 10 steps are all 0%, that's a collapse
    if n >= 20:
        win_rates = [m.get("win_rate", 0) for m in metrics]
        peak_wr = max(win_rates)
        recent_wr = win_rates[-10:]
        if peak_wr > 0.05 and all(w == 0 for w in recent_wr):
            print(f"WARNING: Win rate collapsed! Peak was {peak_wr:.2%}, last 10 steps all 0%")
            return 1

    # Check for stall: no new entries in >30 min
    # (use file mtime as proxy)
    mtime = LOG.stat().st_mtime
    age_min = (time.time() - mtime) / 60
    if age_min > 30:
        print(f"CRITICAL: No log update in {age_min:.0f} minutes — process may be dead")
        return 2

    # Rolling averages for last 5 and last 20 steps
    if n >= 5:
        last5 = metrics[-5:]
        avg_wr5 = sum(m.get("win_rate", 0) for m in last5) / 5
        avg_reward5 = sum(m.get("mean_reward", 0) for m in last5) / 5
        avg_loss5 = sum(m.get("loss", 0) for m in last5) / 5
        print(f"\n  [Last 5 avg] win_rate={avg_wr5:.3f} reward={avg_reward5:.3f} loss={avg_loss5:.4f}")

    if n >= 20:
        last20 = metrics[-20:]
        avg_wr20 = sum(m.get("win_rate", 0) for m in last20) / 20
        avg_reward20 = sum(m.get("mean_reward", 0) for m in last20) / 20
        print(f"  [Last 20 avg] win_rate={avg_wr20:.3f} reward={avg_reward20:.3f}")

    # Memory
    peak_mem = max(m.get("memory_peak_gb", 0) for m in metrics)
    print(f"\n  Peak memory (all time): {peak_mem:.2f} GB")

    # Estimated time remaining
    if n >= 2:
        times = [m.get("time_total", 0) for m in metrics[-10:]]
        avg_time = sum(times) / len(times)
        remaining = (200 - step) * avg_time / 60
        print(f"  Avg step time: {avg_time:.1f}s → ~{remaining:.0f} min remaining")

    print("\nSTATUS: HEALTHY")
    return 0

if __name__ == "__main__":
    metrics = load_metrics()
    code = analyze(metrics)
    sys.exit(code)
