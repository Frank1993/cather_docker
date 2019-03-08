#!/bin/sh

CUDA_VISIBLE_DEVICES="0"
INPUT_FOLDER="/home/docker/apollo/datasets/image_orientation/correct_orientation/orientation_src"
OUTPUT_FOLDER="/home/docker/apollo/datasets/image_orientation/correct_orientation/orientation_dst"
FTP_BUNDLE_PATH='/ORBB/data/image_orientation/good_bundle.zip'

echo 'Parameters:'
echo 'INPUT_FOLDER = ' $INPUT_FOLDER
echo 'OUTPUT_FOLDER = ' $OUTPUT_FOLDER
echo 'FTP_BUNDLE_PATH = ' $FTP_BUNDLE_PATH

set -e
PYTHONPATH=../../../../:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u ./filter_bad_orientation_images.py \
    --input_folder $INPUT_FOLDER \
    --output_folder $OUTPUT_FOLDER \
    --ftp_bundle_path $FTP_BUNDLE_PATH
