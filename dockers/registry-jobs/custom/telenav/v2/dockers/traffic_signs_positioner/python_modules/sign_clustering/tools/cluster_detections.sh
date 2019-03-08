#!/usr/bin/env bash

set -e
INPUT_FILE="/Users/adrianpopovici/Workspace/SL_git/base/python_modules/sign_clustering/data/localization3.bin"
CONFIG_FILE="../config/clustering_config.json"
OUTPUT_FOLDER="/Users/adrianpopovici/Workspace/SL_git/base/python_modules/sign_clustering/data/"
THREADS_NUMBER=4


echo 'Parameters:'
echo 'INPUT_FILE='$INPUT_FILE
echo 'CONFIG_FILE='$CONFIG_FILE
echo 'OUTPUT_FOLDER='$OUTPUT_FOLDER
echo 'THREADS_NUMBER='$THREADS_NUMBER

PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python ../clustering.py -i $INPUT_FILE -c $CONFIG_FILE -o $OUTPUT_FOLDER -t $THREADS_NUMBER