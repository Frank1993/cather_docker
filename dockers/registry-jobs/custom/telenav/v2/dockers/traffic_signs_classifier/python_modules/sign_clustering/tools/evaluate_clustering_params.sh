#!/usr/bin/env bash

set -e
INPUT_FILE="/Users/adrianpopovici/Workspace/SL_git/base/python_modules/sign_clustering/data/localization3.bin"
CONFIG_FILE="../config/clustering_config.json"
GROUND_TRUTH_FILE="/Users/adrianpopovici/Workspace/SL_git/base/python_modules/sign_clustering/data/clustersP.csv"
THREADS_NUMBER=4


echo 'Parameters:'
echo 'INPUT_FILE='$INPUT_FILE
echo 'CONFIG_FILE='$CONFIG_FILE
echo 'GROUND_TRUTH_FILE='$GROUND_TRUTH_FILE
echo 'THREADS_NUMBER='$THREADS_NUMBER

PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python ../evaluate_clusters.py -i $INPUT_FILE -c $CONFIG_FILE -g $GROUND_TRUTH_FILE -t $THREADS_NUMBER