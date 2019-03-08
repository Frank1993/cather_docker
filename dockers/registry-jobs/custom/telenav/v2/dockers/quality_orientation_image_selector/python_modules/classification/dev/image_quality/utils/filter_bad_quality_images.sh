#!/bin/sh

CUDA_VISIBLE_DEVICES="0"
INPUT_FOLDER="/home/docker/apollo/datasets/image_quality/correct_quality/quality_src"
OUTPUT_FOLDER="/home/docker/apollo/datasets/image_quality/correct_quality/quality_dst"
FTP_BUNDLE_PATH='/ORBB/data/image_quality/good_bundle.zip'

echo 'Parameters:'
echo 'INPUT_FOLDER = ' $INPUT_FOLDER
echo 'OUTPUT_FOLDER = ' $OUTPUT_FOLDER
echo 'FTP_BUNDLE_PATH = ' $FTP_BUNDLE_PATH

set -e
PYTHONPATH=../../../../:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u ./filter_bad_quality_images.py \
    --input_folder $INPUT_FOLDER \
    --output_folder $OUTPUT_FOLDER \
    --ftp_bundle_path $FTP_BUNDLE_PATH
