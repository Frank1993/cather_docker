#!/usr/bin/env bash
set -e
CONFIG_FILE="./config.json"

echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u ./roi_image_index_filter.py \
          --config_file $CONFIG_FILE
