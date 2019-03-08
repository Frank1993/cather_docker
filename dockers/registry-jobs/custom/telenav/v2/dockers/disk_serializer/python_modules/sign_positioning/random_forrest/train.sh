#!/usr/bin/env bash
set -e

PHONE_LENSES_PATH=../config/phone_lenses.prototxt
ROIS_DIMENSION_PATH=../config/roi_dimensions.prototxt

PREDICTED_ROIS_PATH=./train_data/calculated_position_set.bin
GROUND_TRUTH_ROIS_PATH=./train_data/gt_position_set.bin
TEST_ROIS_PATH=./test_data/output.bin
TEST_OUTPUT_PATH=./test_data/

PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python3 -u rf_trainer.py --predicted_rois_path $PREDICTED_ROIS_PATH \
                         --ground_truth_rois_path $GROUND_TRUTH_ROIS_PATH \
                         --test_rois_path $TEST_ROIS_PATH \
                         --test_output_path $TEST_OUTPUT_PATH \
                         --phone_lenses_path $PHONE_LENSES_PATH \
                         --rois_dimension_path $ROIS_DIMENSION_PATH
