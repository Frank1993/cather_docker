#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES="2"
WEIGHTS='./snapshots_101_imagenet/resnet101_traffic_signs_01.h5'
TRAIN_META_FILE='/data/traffic_signs_04_2018/train/rois.bin'
INPUT_PATH='/data/traffic_signs_04_2018/test'
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
    --backbone 'resnet50' \
    --resolutions 400 600
