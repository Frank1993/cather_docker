#!/usr/bin/env bash
set -e
CONFIG_FILE="retinanet_config.json"

cd apollo/imagerecognition/python_modules/object_detection/retinanet/tools/

if [ ! -z "$1" ]; then
    CONFIG_FILE=$1
fi


echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE

PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u ../mq/predictor.py --config_files $CONFIG_FILE
