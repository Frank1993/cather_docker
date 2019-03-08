#!/usr/bin/env bash

IMAGES_FOLDER='/Users/sergiuc/data/big_sample/crops/'
IMAGES_OUT_FOLDER='/Users/sergiuc/data/big_sample/example/'
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
