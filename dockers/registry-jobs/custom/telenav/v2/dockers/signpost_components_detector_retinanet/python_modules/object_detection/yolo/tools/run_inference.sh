#!/usr/bin/env bash

set -e

INPUT_FOLDER=/data/faces_license_plates_2018_09/test
OUTPUT_FOLDER=/home/flaviub/faces_blur_data_out

echo 'Parameters:'
echo 'OUTPUT_FOLDER='$OUTPUT_FOLDER
echo 'INPUT_FOLDER='$INPUT_FOLDER
echo '----------------------------------------'

PYTHONPATH="${PYTHONPATH}:../../:../../apollo_python_common/protobuf/:./"
export PYTHONPATH

echo $PYTHONPATH

YOLO_FOLDER=./../
cd $YOLO_FOLDER

python3 ./tools/test_yolo_model.py --input_path $INPUT_FOLDER --output_path $OUTPUT_FOLDER
