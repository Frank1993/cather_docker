#!/usr/bin/env bash

set -e

# Run our conversion
PYTHONPATH=../../python_modules/:../../python_modules/apollo_python_common/protobuf:$PYTHONPATH
export PYTHONPATH
echo $PYTHONPATH

python roi_sampler.py \
 --rois_file=/Users/mihaic7/Development/projects/telenav/roi_angle_predictor/mount/traffic_signs_04_2018/train/rois.bin \
 --output_dir=/Users/mihaic7/Development/projects/telenav/roi_angle_predictor/mount/roi_angle_predictor \
 --num_batches=6 \
 --num_rois=5000 \
 --num_classes=55