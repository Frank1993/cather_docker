#!/usr/bin/env bash
set -e
CONFIG_FILE="../../config.json"

echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

cd ./python_modules/mq_disk_serializer/

PYTHONPATH=../:../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u ./mq_disk_serializer.py \
    --config_file $CONFIG_FILE
