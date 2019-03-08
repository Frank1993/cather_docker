#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES="0"
CONFIG_FILE="../../../config.json"

echo 'Parameters:'
echo 'CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

cd ./python_modules/ocr/deploy/

PYTHONPATH=../../:../../ocr/attention_ocr:../../ocr/attention_ocr/datasets:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u ../scripts/prediction/mq_ocr_predictor.py \
    --config_file $CONFIG_FILE
