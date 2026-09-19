#!/bin/bash
set -e

PROJECT_DIR="/root/ly/BEV/Sparse4D"
CONFIG="work_dirs/fisheye_sparse4d/fisheye_v3_r50_4x.py"
WORK_DIR="$PROJECT_DIR/work_dirs/fisheye_sparse4d"

cd "$PROJECT_DIR"
source /root/ly/BEV/mm_sparse4d/bin/activate
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

LATEST_CKPT=$(ls -t "$WORK_DIR"/iter_*.pth 2>/dev/null | head -1)

if [ -n "$LATEST_CKPT" ]; then
    echo "========================================="
    echo " Resuming from: $LATEST_CKPT"
    echo "========================================="
    python tools/train.py "$CONFIG" --resume-from "$LATEST_CKPT"
else
    echo "========================================="
    echo " Starting fresh training (load_from=sparse4dv3_r50.pth)"
    echo "========================================="
    python tools/train.py "$CONFIG"
fi
