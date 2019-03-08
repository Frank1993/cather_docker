#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="0"

FTP_BUNDLE_PATH="/ORBB/data/image_quality/good_bundle.zip"
OUTPUT_FOLDER="./tf_models/"

echo 'Parameters:'
echo 'FTP_BUNDLE_PATH:' $FTP_BUNDLE_PATH
echo 'OUTPUT_FOLDER:' $OUTPUT_FOLDER

PYTHONPATH=../../../:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u ./keras_file_2_pb_file.py \
    --ftp_bundle_path $FTP_BUNDLE_PATH \
    --output_folder $OUTPUT_FOLDER \
