#!/usr/bin/env bash

GT_ROI='rois_test.bin'
PRED_ROI='rois_retinanet.bin'
RESULT_FILE='classes_thresholds.json'
SELECTED_CLASSES_FILE='eval_selected_classes.json'

MIN_SIZE=25
echo 'Parameters:'
echo 'GT_ROI:' $GT_ROI
echo 'PRED_ROI:' $PRED_ROI
echo 'RESULT_FILE:' $RESULT_FILE
echo 'MIN_SIZE:' $MIN_SIZE
echo '------------------------------------'
PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH


CUDA_VISIBLE_DEVICES='-1' python -u ../generate_best_thresholds.py \
    -p $PRED_ROI \
    -g $GT_ROI \
    -o $RESULT_FILE \
    -m $MIN_SIZE \
    -c $SELECTED_CLASSES_FILE
