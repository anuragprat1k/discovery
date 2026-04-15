#!/bin/bash
# Sync artifacts from RunPod every 30 minutes
# Usage: nohup bash code_repair/runpod_sync/sync.sh &
set -euo pipefail

REMOTE="root@91.199.227.82"
PORT=34417
KEY="$HOME/.ssh/id_ed25519"
LOCAL_DIR="$(dirname "$0")"
SSH="ssh -o StrictHostKeyChecking=no -i $KEY -p $PORT"

while true; do
    echo "[$(date)] Syncing..."

    # Checkpoints (metrics, eval trajectories, etc.)
    rsync -avz -e "$SSH" "$REMOTE:/workspace/discovery/checkpoints/" "$LOCAL_DIR/checkpoints/" 2>&1 || echo "rsync checkpoints failed"

    # Training logs
    rsync -avz -e "$SSH" "$REMOTE:/workspace/train_path_indep.log" "$REMOTE:/workspace/train_path_dep.log" "$REMOTE:/workspace/overnight_monitor.log" "$LOCAL_DIR/" 2>&1 || echo "rsync logs failed"

    echo "[$(date)] Sync complete"
    sleep 1800  # 30 minutes
done
