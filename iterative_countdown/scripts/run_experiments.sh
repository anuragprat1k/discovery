#!/bin/bash
# Run all Countdown RL experiments
# Usage: bash scripts/run_experiments.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

# Generate dataset first
echo "=== Generating dataset ==="
if [[ "$DRY_RUN" == "false" ]]; then
    python -m iterative_countdown.data.generate_dataset \
        --n_train 500 --n_eval 100 --seed 42 \
        --output_dir iterative_countdown/data/
fi

# Experiment 1: Binary reward
echo "=== Experiment 1: Binary reward ==="
if [[ "$DRY_RUN" == "false" ]]; then
    python -m iterative_countdown.recipes.e1_binary
fi

# Experiment 2: Dense reward
echo "=== Experiment 2: Dense reward ==="
if [[ "$DRY_RUN" == "false" ]]; then
    python -m iterative_countdown.recipes.e2_dense
fi

# Experiment 3: PRIME reward
echo "=== Experiment 3: PRIME reward ==="
if [[ "$DRY_RUN" == "false" ]]; then
    python -m iterative_countdown.recipes.e3_prime
fi

# Plot comparison
echo "=== Plotting results ==="
if [[ "$DRY_RUN" == "false" ]]; then
    python -m iterative_countdown.scripts.plot_results \
        --results_dir ~/countdown_logs \
        --runs binary dense prime
fi

echo "=== All experiments complete ==="
