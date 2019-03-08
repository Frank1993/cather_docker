#!/usr/bin/env bash

set -e
CONFIG_FILE="../config/clustering_config.json"

PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python3 -u ../mq/predictor.py --config_file $CONFIG_FILE