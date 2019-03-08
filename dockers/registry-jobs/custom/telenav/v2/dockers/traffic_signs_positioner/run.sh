#!/usr/bin/env bash
set -e
#CONFIG_FILE="../../../../config.json"
CONFIG_FILE="../../../config.json"

echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

#cd ./imagerecognition/python_modules/sign_positioning/tools/
cd ./python_modules/sign_positioning/tools/

PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python3 -u ../mq/predictor.py --config_file $CONFIG_FILE
