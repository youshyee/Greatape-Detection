#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS=$1
CONFIG=$2
CHECKPOINT=$3
DATAPATH=$4

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT $DATAPATH --launcher pytorch ${@:5}
