#!/usr/bin/env bash
set -e
CONFIG_FILE="../config/sign_positioning_config.json"

echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'
PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python3 -u ../mq/predictor.py --config_file $CONFIG_FILE
