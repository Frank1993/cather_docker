#!/bin/sh

INPUT_FOLDER="/home/docker/apollo/datasets/image_orientation/osc-images/unittest_performance/raw_imgs/up"

echo 'Parameters:'
echo 'INPUT_FOLDER = ' $INPUT_FOLDER

set -e
PYTHONPATH=../../../../:$PYTHONPATH
export PYTHONPATH

python -u ./create_orientation_dataset.py \
    --input_folder $INPUT_FOLDER 