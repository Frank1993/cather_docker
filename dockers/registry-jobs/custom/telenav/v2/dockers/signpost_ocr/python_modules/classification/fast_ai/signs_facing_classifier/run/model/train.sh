#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="0"
TRAIN_CFG_JSON="./train_cfg.json"

echo 'Parameters:'
echo 'MATCH_CFG_JSON = ' $TRAIN_CFG_JSON
echo 'CUDA_VISIBLE_DEVICES = ' $CUDA_VISIBLE_DEVICES

set -e
PYTHONPATH=../../../../../:../../../../../apollo_python_common/protobuf:$PYTHONPATH
export PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2.

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ../../utils/train.py --train_cfg_json $TRAIN_CFG_JSON