#!/bin/sh

CONFIG_JSON="./fake_dataset_config.json"

echo 'Parameters:'
echo 'CONFIG_JSON = ' $CONFIG_JSON

set -e
PYTHONPATH=../../../../:$PYTHONPATH
export PYTHONPATH

python ../../../scripts/fake_dataset/signpost_fake_dataset_generator.py \
    --config_json $CONFIG_JSON
    