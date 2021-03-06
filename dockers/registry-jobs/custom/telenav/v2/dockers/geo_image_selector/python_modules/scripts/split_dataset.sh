#!/usr/bin/env bash

IMAGES_FOLDER='/data/datasets/traffic_sign_components_09_2018/v_0_1/filtered_dataset_new/'
IMAGES_OUT_FOLDER='/data/datasets/traffic_sign_components_09_2018/v_0_1/filtered_dataset_2018_12/'
SPLIT_RATIO=0.9
GENERATE_METADATA=true

echo 'Parameters:'
echo 'IMAGES_FOLDER:' $IMAGES_FOLDER
echo 'IMAGES_OUT_FOLDER:' $IMAGES_OUT_FOLDER
echo 'SPLIT_RATIO:' $SPLIT_RATIO
echo 'GENERATE_METADATA' $GENERATE_METADATA
echo '------------------------------------'
PYTHONPATH=../:../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u split_dataset.py \
    --images_folder $IMAGES_FOLDER \
    --images_out_folder $IMAGES_OUT_FOLDER \
    --split_ratio $SPLIT_RATIO \
    --generate_metadata $GENERATE_METADATA
