#!/bin/bash
# Safe training launcher: auto-resume on crash/OOM
set -e

PROJECT_DIR="/root/ly/BEV/Sparse4D"
CONFIG="work_dirs/fisheye_sparse4d/fisheye_v3_r50_4x.py"
WORK_DIR="$PROJECT_DIR/work_dirs/fisheye_sparse4d"
MAX_RETRIES=100

cd "$PROJECT_DIR"
source /root/ly/BEV/mm_sparse4d/bin/activate
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

retry=0
while [ $retry -lt $MAX_RETRIES ]; do
    LATEST_CKPT=$(ls -t "$WORK_DIR"/iter_*.pth 2>/dev/null | head -1)

    if [ -n "$LATEST_CKPT" ]; then
        echo "========================================="
        echo " [$retry] Resuming from: $LATEST_CKPT"
        echo "========================================="
        python tools/train.py "$CONFIG" --resume-from "$LATEST_CKPT" 2>&1 | tee -a "$WORK_DIR/train.log"
    else
        echo "========================================="
        echo " [$retry] Starting fresh training"
        echo "========================================="
        python tools/train.py "$CONFIG" 2>&1 | tee -a "$WORK_DIR/train.log"
    fi

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully!"
        break
    fi

    retry=$((retry + 1))
    echo "========================================="
    echo " Training stopped (code=$EXIT_CODE)."
    echo " Auto-retry in 60s... (attempt $retry/$MAX_RETRIES)"
    echo "========================================="
    sleep 60
done

echo "All attempts exhausted or training done."
