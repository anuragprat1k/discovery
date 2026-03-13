#!/bin/bash
# Monitor training logs and trigger evals for steps 40 and 50
# Usage: bash monitor_and_eval.sh

BINARY_LOG="/workspace/discovery-full-exp/checkpoints/countdown_binary/training_log.jsonl"
DENSE_LOG="/workspace/discovery-full-exp/checkpoints/countdown_dense/training_log.jsonl"
BINARY_CKPT="/workspace/discovery-full-exp/checkpoints/countdown_binary/checkpoints.jsonl"
DENSE_CKPT="/workspace/discovery-full-exp/checkpoints/countdown_dense/checkpoints.jsonl"
EVAL_PROBLEMS="iterative_countdown/data/eval_50.json"

cd /workspace/discovery-full-exp

eval_step() {
    local run_type=$1  # binary or dense
    local step=$2
    local tinker_path=$3
    local eval_dir="checkpoints/countdown_${run_type}/eval"

    echo "[$(date)] Starting eval: ${run_type} step ${step}"
    python -m iterative_countdown.evaluation.eval_pass_at_k \
        --problems_path "$EVAL_PROBLEMS" \
        --tinker_path "$tinker_path" \
        --output_dir "$eval_dir" \
        --step "$step" --k_values 1 --n_samples 1 2>&1
    echo "[$(date)] Completed eval: ${run_type} step ${step}"
}

get_max_step() {
    local logfile=$1
    if [ -f "$logfile" ]; then
        tail -1 "$logfile" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['step'])" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

get_tinker_path() {
    local ckpt_file=$1
    local step=$2
    local step_name=$(printf "step_%04d" "$step")
    if [ -f "$ckpt_file" ]; then
        grep "$step_name" "$ckpt_file" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['path'])" 2>/dev/null
    fi
}

# Track which evals we've launched
declare -A done

for step in 40 50; do
    done["binary_${step}"]=0
    done["dense_${step}"]=0
done

echo "[$(date)] Monitoring for steps 40, 50..."

while true; do
    all_done=1

    for step in 40 50; do
        # Binary
        if [ "${done[binary_${step}]}" -eq 0 ]; then
            max_binary=$(get_max_step "$BINARY_LOG")
            if [ "$max_binary" -ge "$step" ]; then
                tinker_path=$(get_tinker_path "$BINARY_CKPT" "$step")
                if [ -n "$tinker_path" ]; then
                    eval_step "binary" "$step" "$tinker_path" &
                    done["binary_${step}"]=1
                else
                    all_done=0
                fi
            else
                all_done=0
            fi
        fi

        # Dense
        if [ "${done[dense_${step}]}" -eq 0 ]; then
            max_dense=$(get_max_step "$DENSE_LOG")
            if [ "$max_dense" -ge "$step" ]; then
                tinker_path=$(get_tinker_path "$DENSE_CKPT" "$step")
                if [ -n "$tinker_path" ]; then
                    eval_step "dense" "$step" "$tinker_path" &
                    done["dense_${step}"]=1
                else
                    all_done=0
                fi
            else
                all_done=0
            fi
        fi
    done

    if [ "$all_done" -eq 1 ]; then
        echo "[$(date)] All evals launched. Waiting for background jobs..."
        wait
        echo "[$(date)] All done!"
        break
    fi

    sleep 30
done
