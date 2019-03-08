#!/usr/bin/env bash
set -e
CONFIG_FILE="./config.json"

echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u ./quick_tag_roi_dataset_generator.py \
          --config_file $CONFIG_FILE
