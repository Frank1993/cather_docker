#!/usr/bin/env bash

TEST_ROI='rois_test_signposts.bin'
PREDICT_ROI='rois_retinanet.bin'
RESULT_FILE='statistics.txt'
SELECTED_CLASSES='eval_selected_classes.json'
CLASSES_THRESHOLDS='classes_thresholds_signpost.json'
MIN_SIZE=3
echo 'Parameters:'
echo 'TEST_ROI:' $TEST_ROI
echo 'PREDICT_ROI:' $PREDICT_ROI
echo 'RESULT_FILE:' $RESULT_FILE
echo 'MIN_SIZE:' $MIN_SIZE
echo '------------------------------------'

PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u ../../../apollo_python_common/obj_detection_evaluator/protobuf_evaluator.py \
    --expected_rois_file $TEST_ROI \
    --actual_rois_file $PREDICT_ROI \
    --result_file $RESULT_FILE \
    --min_size $MIN_SIZE \
    --is_for_signposts 1 \
    --classes_thresholds_file $CLASSES_THRESHOLDS
#    --selected_classes_file $SELECTED_CLASSES
