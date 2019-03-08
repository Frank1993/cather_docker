#!/usr/bin/env bash

set -e

# Run our conversion
PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH
echo $PYTHONPATH

python ../py_scripts/roi_sampler.py \
 --rois_file=/Users/mihaic7/Development/projects/telenav/roi_angle_predictor/mount/traffic_signs_09_2018/train_new/rois.bin \
 --output_dir=/Users/mihaic7/Development/projects/telenav/roi_angle_predictor/mount/roi_angle_predictor/new_samples \
 --excl_samples_dir=/Users/mihaic7/Development/projects/telenav/roi_angle_predictor/mount/roi_angle_predictor/new_samples/exclude \
 --num_batches=6 \
 --num_rois=5000 \
 --num_classes=87
