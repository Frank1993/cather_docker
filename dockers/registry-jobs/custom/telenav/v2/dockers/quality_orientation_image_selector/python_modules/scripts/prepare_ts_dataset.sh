#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES='1'
ROOT_FOLDER = '/data/traffic_signs_09_2018'
EXCLUDE_FROM_TRAIN_FILE = '/home/adrianm/work/exclude_from_train.csv'

echo 'Parameters:'
echo 'ROOT_FOLDER:' $ROOT_FOLDER
echo 'EXCLUDE_FROM_TRAIN_FILE:' $EXCLUDE_FROM_TRAIN_FILE
echo '------------------------------------'
PYTHONPATH=../:../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u prepare_ts_dataset.py \
    --root_folder $ROOT_FOLDER \
    --exclude_from_train_file $EXCLUDE_FROM_TRAIN_FILE
