#!/usr/bin/env bash

GT_ROI='rois_test_signposts.bin'
PRED_ROI='rois_retinanet.bin'
RESULT_FILE='classes_thresholds_signpost.json'
SELECTED_CLASSES_FILE=''

MIN_SIZE=3
echo 'Parameters:'
echo 'GT_ROI:' $GT_ROI
echo 'PRED_ROI:' $PRED_ROI
echo 'RESULT_FILE:' $RESULT_FILE
echo 'MIN_SIZE:' $MIN_SIZE
echo '------------------------------------'
PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH


CUDA_VISIBLE_DEVICES='-1' python -u ../generate_best_thresholds.py \
    --is_for_signposts true \
    -p $PRED_ROI \
    -g $GT_ROI \
    -o $RESULT_FILE \
    -m $MIN_SIZE
#    -c $SELECTED_CLASSES_FILE
