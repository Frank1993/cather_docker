#!/bin/sh

EVALUATION_CONFIG_PATH="./config/evaluation_config.json"

echo 'Parameters:'
echo 'EVALUATION_CONFIG_PATH = ' $EVALUATION_CONFIG_PATH

set -e
PYTHONPATH=../:../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python ./evaluate_clusters.py \
    --evaluation_config_file $EVALUATION_CONFIG_PATH
    