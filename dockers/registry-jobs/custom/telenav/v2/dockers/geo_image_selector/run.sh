#!/usr/bin/env bash
set -e
CONFIG_FILE="./config.json"

echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

PYTHONPATH=python_modules/:python_modules/apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u ./python_modules/mq_selectors/deploy/geo_image_selector/geo_selector.py \
          --config_file $CONFIG_FILE
