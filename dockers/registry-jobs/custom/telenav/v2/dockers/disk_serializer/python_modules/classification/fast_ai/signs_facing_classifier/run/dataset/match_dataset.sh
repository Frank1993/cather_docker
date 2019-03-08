#!/usr/bin/env bash
MATCH_CFG_JSON="./match_dataset_cfg.json"

echo 'Parameters:'
echo 'MATCH_CFG_JSON = ' $MATCH_CFG_JSON

set -e
PYTHONPATH=../../../../../:../../../../../apollo_python_common/protobuf:$PYTHONPATH
export PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2.

python ../../utils/dataset_matcher.py \
    --match_cfg_json $MATCH_CFG_JSON
