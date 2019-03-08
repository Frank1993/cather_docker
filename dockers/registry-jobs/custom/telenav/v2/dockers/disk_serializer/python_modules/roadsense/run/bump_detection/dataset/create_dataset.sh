#!/bin/sh

CONFIG_JSON="./dataset_config.json"

echo 'Parameters:'
echo 'CONFIG_JSON = ' $CONFIG_JSON

set -e
PYTHONPATH=../../../../:$PYTHONPATH
export PYTHONPATH

python ../../../scripts/general/dataset_builder.py \
    --config_json $CONFIG_JSON
    