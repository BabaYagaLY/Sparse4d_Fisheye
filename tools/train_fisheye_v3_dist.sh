#!/bin/bash
set -e

PROJECT_DIR="/root/ly/BEV/Sparse4D"
CONFIG="work_dirs/fisheye_sparse4d/fisheye_v3_r50_4x.py"
WORK_DIR="$PROJECT_DIR/work_dirs/fisheye_sparse4d"

# Distributed parameters (with defaults)
NODE_RANK=${1:-0}
NNODES=${2:-2}
MASTER_ADDR=${3:-"10.141.210.14"}
MASTER_PORT=${4:-29500}
GPUS_PER_NODE=${5:-1}

cd "$PROJECT_DIR"
source /root/ly/BEV/mm_sparse4d/bin/activate
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

LATEST_CKPT=$(ls -t "$WORK_DIR"/iter_*.pth 2>/dev/null | head -1)

if [ -n "$LATEST_CKPT" ]; then
    echo "========================================="
    echo " Resuming from: $LATEST_CKPT"
    echo " Distributed: node $NODE_RANK / $NNODES"
    echo " Master: $MASTER_ADDR:$MASTER_PORT"
    echo " GPUs per node: $GPUS_PER_NODE"
    echo "========================================="
    torchrun \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --nproc_per_node=$GPUS_PER_NODE \
        tools/train.py "$CONFIG" --launcher pytorch --resume-from "$LATEST_CKPT"
else
    echo "========================================="
    echo " Starting fresh training (load_from=sparse4dv3_r50.pth)"
    echo " Distributed: node $NODE_RANK / $NNODES"
    echo " Master: $MASTER_ADDR:$MASTER_PORT"
    echo " GPUs per node: $GPUS_PER_NODE"
    echo "========================================="
    torchrun \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --nproc_per_node=$GPUS_PER_NODE \
        tools/train.py "$CONFIG" --launcher pytorch
fi
