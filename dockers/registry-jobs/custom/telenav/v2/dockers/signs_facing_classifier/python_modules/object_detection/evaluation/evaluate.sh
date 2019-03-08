#!/bin/sh

GT_PATH="/home/docker/apollo/datasets/roi/traffic_signs_04_2018/test/rois.bin"
PRED_PATH="/home/docker/apollo/datasets/roi_classifier/retinanet_preds/rois_retinanet_test_v4.bin"
SELECTED_CLASSES_PATH="/home/docker/apollo/datasets/roi_classifier/retinanet_preds/most_classes.json"
CLASS_THRESHOLDS_PATH="/home/docker/apollo/datasets/roi_classifier/retinanet_preds/classes_thresholds_v4_ontrain.json"
OUTPUT_CSV_PATH="./metrics.csv"

echo 'Parameters:'
echo 'GT_PATH = ' $GT_PATH
echo 'PRED_PATH = ' $PRED_PATH
echo 'SELECTED_CLASSES_PATH = ' $SELECTED_CLASSES_PATH
echo 'CLASS_THRESHOLDS_PATH = ' $CLASS_THRESHOLDS_PATH
echo 'OUTPUT_CSV_PATH = ' $OUTPUT_CSV_PATH

set -e
PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2.

python evaluate.py \
    --gt_path $GT_PATH \
    --pred_path $PRED_PATH \
    --selected_classes_path $SELECTED_CLASSES_PATH \
    --class_thresholds_path $CLASS_THRESHOLDS_PATH\
    --output_csv_path $OUTPUT_CSV_PATH