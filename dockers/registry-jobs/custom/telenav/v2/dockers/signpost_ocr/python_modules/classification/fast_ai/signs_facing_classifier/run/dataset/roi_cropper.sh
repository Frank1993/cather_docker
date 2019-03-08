#!/usr/bin/env bash
CROPPER_CFG_JSON="./roi_cropper_cfg.json"

echo 'Parameters:'
echo 'CROPPER_CFG_JSON = ' $CROPPER_CFG_JSON

set -e
PYTHONPATH=../../../../../:../../../../../apollo_python_common/protobuf:$PYTHONPATH
export PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2.

python ../../utils/roi_cropper.py \
    --config $CROPPER_CFG_JSON