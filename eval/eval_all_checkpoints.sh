#!/bin/bash
# eval_all_checkpoints.sh — Evaluate all saved checkpoints for one training run.
#
# Usage:
#   bash eval/eval_all_checkpoints.sh {binary|dense}
#
# For each checkpoint found in checkpoints/{run}/step_*/ the script calls
# pass_at_k.py and writes results to results/{run}/step_XXXX.json.
# Already-evaluated checkpoints are skipped, making the script safely resumable.

set -e

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

if [[ -z "${1:-}" ]]; then
    echo "Error: no run name provided." >&2
    echo "Usage: bash eval/eval_all_checkpoints.sh {binary|dense}" >&2
    exit 1
fi

RUN="$1"
CKPT_ROOT="checkpoints/${RUN}"
RESULTS_DIR="results/${RUN}"

if [[ ! -d "${CKPT_ROOT}" ]]; then
    echo "Error: checkpoint directory '${CKPT_ROOT}' does not exist." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Collect and sort checkpoint directories numerically by step number
# ---------------------------------------------------------------------------

# Build a list of (step_number, path) pairs so we can sort numerically.
# Use an array of "NNNN path" strings, then sort -n on the numeric prefix.
mapfile -t SORTED_CKPTS < <(
    for ckpt_dir in "${CKPT_ROOT}"/step_*/; do
        # Strip trailing slash, extract directory name
        dir_name="$(basename "${ckpt_dir%/}")"
        if [[ "${dir_name}" =~ ^step_([0-9]+)$ ]]; then
            step_num="${BASH_REMATCH[1]}"
            # Emit "numeric_step path" so sort -n works correctly
            printf '%d\t%s\n' "$((10#${step_num}))" "${ckpt_dir%/}"
        fi
    done | sort -k1,1n | awk -F'\t' '{print $2}'
)

if [[ ${#SORTED_CKPTS[@]} -eq 0 ]]; then
    echo "No checkpoint directories found under '${CKPT_ROOT}'." >&2
    exit 1
fi

echo "Found ${#SORTED_CKPTS[@]} checkpoint(s) for run '${RUN}'."
mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Evaluate each checkpoint
# ---------------------------------------------------------------------------

for ckpt_dir in "${SORTED_CKPTS[@]}"; do
    dir_name="$(basename "${ckpt_dir}")"

    # Extract numeric step (strip leading zeros for arithmetic, but keep
    # zero-padded form for the output filename check)
    if [[ "${dir_name}" =~ ^step_([0-9]+)$ ]]; then
        step_padded="${BASH_REMATCH[1]}"
        step_num="$((10#${step_padded}))"
    else
        echo "Skipping unrecognised directory: ${ckpt_dir}" >&2
        continue
    fi

    # Build the expected output filename with 4-digit zero-padding
    result_file="${RESULTS_DIR}/$(printf 'step_%04d.json' "${step_num}")"

    # Skip if already evaluated (resumable)
    if [[ -f "${result_file}" ]]; then
        echo "Skipping step ${step_num} (${result_file} already exists)."
        continue
    fi

    echo "Evaluating step ${step_num}..."

    python eval/pass_at_k.py \
        --checkpoint_dir "${ckpt_dir}" \
        --run_name "${RUN}" \
        --step "${step_num}" \
        --output_dir "${RESULTS_DIR}"

    echo "Step ${step_num} complete -> ${result_file}"
done

echo ""
echo "All checkpoints evaluated for run '${RUN}'."
