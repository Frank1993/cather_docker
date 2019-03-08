#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES="3"
WEIGHTS='./resnet101_traffic_signs_21_FP32.pb'
TRAIN_META_FILE='/data/datasets/traffic_sign_components_09_2018/v_0_1/filtered_dataset_2018_12/train/rois.bin'
INPUT_PATH='/data/datasets/traffic_sign_components_09_2018/v_0_1/filtered_dataset_2018_12/test'
THRESHOLD_FILE="SAME"
LOWEST_SCORE_THRESHOLD="0.1"
OUTPUT_PATH='output'

echo 'Parameters:'
echo 'CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES
echo 'LOWEST_SCORE_THRESHOLD='$LOWEST_SCORE_THRESHOLD
echo 'WEIGHTS='$WEIGHTS
echo 'TRAIN_META_FILE='$TRAIN_META_FILE
echo 'INPUT_PATH='$INPUT_PATH
echo 'OUTPUT_PATH='$OUTPUT_PATH
echo '----------------------------------------'
PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u ../predict.py \
    --weights_file $WEIGHTS \
    --input_images_path $INPUT_PATH \
    --output_images_path $OUTPUT_PATH \
    --train_meta_file $TRAIN_META_FILE \
    --lowest_score_threshold $LOWEST_SCORE_THRESHOLD \
    --threshold_file $THRESHOLD_FILE \
    --backbone 'resnet101' \
    --resolutions 300 \
    --cut_below_vanishing_point 0 \
    --is_for_signposts true
