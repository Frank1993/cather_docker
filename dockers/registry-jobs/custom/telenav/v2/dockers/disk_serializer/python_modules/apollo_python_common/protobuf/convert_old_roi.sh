#!/usr/bin/env bash

INPUT_ROI_PATH='/Users/adrianpopovici/Workspace/TagData/rois.bin'
OUTPUT_FILE_NAME='new_rois.bin'

echo 'Parameters:'
echo 'INPUT_ROI_PATH:' $INPUT_ROI_PATH
echo 'OUTPUT_FILE_NAME:' $OUTPUT_FILE_NAME
echo '------------------------------------'

PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u convert_old_roi.py -i $INPUT_ROI_PATH -o $OUTPUT_FILE_NAME