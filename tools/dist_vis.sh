#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS=$1

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/chimp_vis.py  --launcher pytorch --config ../configs/chimptest.py  --input ../../../dataspace/PanAfrica/videos/ --checkpoint ../models/retina_r50_scm_tcm_trainseq7-68041edf.pth 
