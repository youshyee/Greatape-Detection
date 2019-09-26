#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS=$1
CONFIG=$2
DATAPATH=$3

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --datapath $DATAPATH --launcher pytorch ${@:4}
