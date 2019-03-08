#!/usr/bin/env bash

set -e
CONFIG_FILE="../config/config.cfg"
DOWNLOAD_FOLDER="./downloads"
THREADS_NUMBER=4
UPDATE_IMAGE_SET_FILE="rois_test.bin"


echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE
echo 'DOWNLOAD_FOLDER='$DOWNLOAD_FOLDER
echo 'THREADS_NUMBER='$THREADS_NUMBER

PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python ../download_trips.py -c $CONFIG_FILE -d $DOWNLOAD_FOLDER -t $THREADS_NUMBER \
--update_image_set_file $UPDATE_IMAGE_SET_FILE
