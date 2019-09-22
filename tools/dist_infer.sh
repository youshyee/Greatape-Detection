#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS=$1

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/infer.py  --launcher pytorch ${@:2}
